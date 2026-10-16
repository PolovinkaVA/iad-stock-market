"""Exact centered Ridge contributions, block permutation, PDP and period review."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .msft_features import ROOT, PRICE_FEATURES, VOLUME_FEATURES, CALENDAR_FEATURES, split_rows, sha, write_json
from .msft_modeling import input_data, metrics, predict_close, versions

LABELS = {
    'return_1':'Изменение цены за 1 сессию', 'return_2':'Изменение цены за 2 сессии',
    'return_5':'Изменение цены за 5 сессий', 'return_20':'Изменение цены за 20 сессий',
    'return_lag_1':'Доходность с лагом 1', 'return_lag_2':'Доходность с лагом 2',
    'return_lag_5':'Доходность с лагом 5', 'close_to_mean_5':'Цена относительно средней за 5 сессий',
    'close_to_mean_20':'Цена относительно средней за 20 сессий',
    'volatility_5':'Волатильность за 5 сессий', 'volatility_20':'Волатильность за 20 сессий',
    'intraday_range':'Диапазон последней свечи', 'intraday_body':'Изменение от открытия до закрытия',
    'opening_gap':'Разрыв цены на открытии', 'log_volume':'Логарифм объёма торгов',
    'volume_to_mean_5':'Объём относительно среднего за 5 сессий',
    'volume_to_mean_20':'Объём относительно среднего за 20 сессий', 'volume_change':'Изменение объёма',
    'weekday':'День недели', 'month':'Месяц',
}
RAW_FEATURES = PRICE_FEATURES + VOLUME_FEATURES + CALENDAR_FEATURES


def load_review(config, root=ROOT):
    root = Path(root)
    rows, metadata = input_data(config, root)
    protocol = json.loads((root/config['reports_dir']/'msft_protocol.json').read_text(encoding='utf-8'))
    check = dict(protocol)
    protocol_hash = check.pop('protocol_sha256')
    if sha(check) != protocol_hash or metadata != protocol['dataset'] or protocol['versions'] != versions():
        raise ValueError('Research data, protocol or model environment differs from the frozen experiment')
    name = protocol['best_learned_model']
    if name != 'ridge__market':
        raise ValueError('This explanation module is designed for the frozen Ridge market model')
    path = root/config['model_dir']/(name+'.joblib')
    bundle = joblib.load(path)
    if bundle['protocol_sha256'] != protocol_hash:
        raise ValueError('Model does not belong to this experiment')
    parts = split_rows(rows, config)
    fit = pd.DataFrame(parts['train']+parts['validation'])
    test = pd.DataFrame(parts['test'])
    return {'model':bundle['pipeline'], 'fit':fit, 'test':test, 'metadata':metadata,
            'protocol':protocol, 'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def transformed_groups(model):
    names = model.named_steps['features'].get_feature_names_out()
    groups = []
    for name in names:
        family, feature = name.split('__',1)
        if family == 'numeric':
            groups.append(feature)
        elif family == 'category':
            groups.append(next(k for k in CALENDAR_FEATURES if feature.startswith(k+'_')))
        else:
            raise ValueError('Unexpected feature transformer for exact Ridge explanations')
    return groups


def reference(model, fit):
    return np.asarray(model.named_steps['features'].transform(fit)).mean(axis=0)


def contributions(model, frame, center):
    """Group centered transformed coefficients back to original input features."""
    z = np.asarray(model.named_steps['features'].transform(frame))
    reg = model.named_steps['regressor']
    delta = (z-center) * reg.coef_
    groups = transformed_groups(model)
    grouped = {k:delta[:,np.array(groups)==k].sum(axis=1) for k in RAW_FEATURES}
    base_return = float(reg.intercept_ + center @ reg.coef_)
    last = frame['close_at_forecast'].to_numpy(dtype=float)
    return grouped, last*(1+base_return), base_return


def explain_row(model, row, center):
    frame = pd.DataFrame([row])
    grouped, base, base_return = contributions(model, frame, center)
    last = float(row['close_at_forecast'])
    values = [{'feature':k, 'label':LABELS[k], 'value':row[k],
               'contribution_usd':float(v[0]*last), 'contribution_bps':float(v[0]*10000)}
              for k,v in grouped.items()]
    prediction = float(predict_close(model,frame)[0])
    reconstructed = float(base[0]+sum(x['contribution_usd'] for x in values))
    if not np.isclose(prediction,reconstructed,rtol=0,atol=1e-8):
        raise ValueError('Local contributions do not reconstruct the prediction')
    return {'target_date':row['target_date'], 'forecast_at':row['forecast_at'],
            'last_close':last, 'base_price':float(base[0]), 'base_adjustment_usd':float(last*base_return),
            'prediction':prediction, 'actual':row.get('target_close'),
            'reconstruction_error_usd':abs(prediction-reconstructed),
            'contributions':sorted(values,key=lambda x:abs(x['contribution_usd']),reverse=True)}


def contribution_importance(model, frame, center):
    grouped, _, _ = contributions(model,frame,center)
    close = frame['close_at_forecast'].to_numpy(dtype=float)
    return [{'feature':k, 'mean_abs_usd':float(np.abs(v*close).mean()),
             'mean_abs_bps':float(np.abs(v*10000).mean())} for k,v in grouped.items()]


def block_indices(n, rng, block_size=20):
    blocks = [np.arange(i,min(i+block_size,n)) for i in range(0,n,block_size)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def permutation_importance(model, frame, repeats=10, seed=42, block_size=20):
    original = metrics(frame['target_close'],predict_close(model,frame))['mae_usd']
    result = []
    for index,feature in enumerate(RAW_FEATURES):
        rng = np.random.default_rng(seed+index)
        changes = []
        for _ in range(repeats):
            shuffled = frame.copy()
            shuffled[feature] = frame[feature].to_numpy()[block_indices(len(frame),rng,block_size)]
            changes.append(metrics(frame['target_close'],predict_close(model,shuffled))['mae_usd']-original)
        result.append({'feature':feature,'delta_mae_usd':float(np.mean(changes)),
                       'std_delta_mae_usd':float(np.std(changes,ddof=1)) if repeats>1 else 0.0})
    return result


def partial_dependence(model, fit, feature, grid_size=25, background_size=100, seed=42):
    if feature not in PRICE_FEATURES+VOLUME_FEATURES:
        raise ValueError('PDP supports numeric inputs only')
    grid = np.unique(np.quantile(fit[feature],np.linspace(.05,.95,grid_size)))
    background = fit.sample(n=min(background_size,len(fit)),random_state=seed).copy()
    output = []
    for value in grid:
        changed = background.copy()
        changed[feature] = value
        output.append({'value':float(value),
                       'mean_predicted_change_bps':float(model.predict(changed).mean()*10000),
                       'mean_predicted_price_usd':float(predict_close(model,changed).mean())})
    return output


def build_explanations(config, root=ROOT, repeats=10):
    root = Path(root)
    data = load_review(config,root)
    model,fit,test = data['model'],data['fit'],data['test']
    center = reference(model,fit)
    periods = {'full':test,'first_half':test.iloc[:len(test)//2], 'second_half':test.iloc[len(test)//2:]}
    result = {'protocol_sha256':data['protocol']['protocol_sha256'],
              'model_sha256':data['model_sha256'], 'model_name':'ridge__market',
              'selected_forecast_model':data['protocol']['selected_model'],
              'reference':'Mean transformed features on train+validation used to fit this frozen Ridge.',
              'permutation':{'repeats':repeats,'block_size':20,'seed':config['seed'],'metric':'MAE USD'},
              'periods':{}}
    for name,frame in periods.items():
        result['periods'][name] = {'rows':len(frame),'start':frame['target_date'].iloc[0],'end':frame['target_date'].iloc[-1],
            'ridge_metrics':metrics(frame['target_close'],predict_close(model,frame)),
            'naive_metrics':metrics(frame['target_close'],predict_close(None,frame)),
            'contribution_importance':contribution_importance(model,frame,center),
            'permutation_importance':permutation_importance(model,frame,repeats,config['seed'])}
    result['pdp'] = {k:partial_dependence(model,fit,k,seed=config['seed']) for k in PRICE_FEATURES+VOLUME_FEATURES}
    locals_ = [explain_row(model,row,center) for row in test.to_dict('records')]
    result['local_explanations'] = {'rows':len(locals_), 'max_reconstruction_error_usd':max(r['reconstruction_error_usd'] for r in locals_),
                                    'example':locals_[-1]}
    result['limitations'] = ['Associations in the fitted model are not causal effects.',
        'Correlated features share information; permutations/PDP can create unusual combinations.',
        'Permutation spread is variation across shuffles, not a confidence interval.',
        'Test interpretation is descriptive, not a new selection/tuning step.',
        'Ridge did not outperform the selected naive forecast.']
    write_json(root/config['reports_dir']/'msft_interpretability.json',result)
    write_json(root/config['processed_dir']/'local_explanations.json',locals_)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/msft_experiment.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    result=build_explanations(config)
    print(json.dumps({'model':result['model_name'], 'periods':list(result['periods']),
                      'explained_rows':result['local_explanations']['rows'],
                      'max_reconstruction_error_usd':result['local_explanations']['max_reconstruction_error_usd']},indent=2))
    return 0
