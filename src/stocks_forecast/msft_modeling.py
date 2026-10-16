"""Chronological MSFT experiments with a separate, frozen holdout stage."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .msft_features import (ROOT, PRICE_FEATURES, VOLUME_FEATURES, TEXT_FEATURES,
                            CALENDAR_FEATURES, sha, split_rows, write_json)

GROUPS = ('price_only', 'market', 'market_text')


def model_pipeline(family, group, config, alpha=None):
    numeric = list(PRICE_FEATURES)
    if group != 'price_only':
        numeric += VOLUME_FEATURES
    if group == 'market_text':
        numeric += TEXT_FEATURES
    transforms = [('numeric', StandardScaler(), numeric)]
    if group != 'price_only':
        categories = CALENDAR_FEATURES + (['event_category'] if group == 'market_text' else [])
        transforms.append(('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories))
    if group == 'market_text':
        transforms.append(('text', TfidfVectorizer(stop_words='english', sublinear_tf=True,
                           token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b', **config['tfidf']), 'news_text'))
    model = Ridge(alpha=alpha, solver='lsqr') if family == 'ridge' else RandomForestRegressor(
        random_state=config['seed'], n_jobs=1, **config['forest'])
    return Pipeline([('features', ColumnTransformer(transforms, sparse_threshold=0)), ('regressor', model)])


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('Nonfinite predictions or labels')
    delta = actual-predicted
    return {'mae_usd': float(np.mean(np.abs(delta))), 'rmse_usd': float(np.sqrt(np.mean(delta**2))),
            'mape_percent': float(100*np.mean(np.abs(delta)/actual))}


def predict_close(model, frame):
    last = frame['close_at_forecast'].to_numpy(dtype=float)
    return last if model is None else last * (1 + model.predict(frame))


def input_data(config, root):
    folder = Path(root)/config['processed_dir']
    metadata = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows = json.loads((folder/'features.json').read_text(encoding='utf-8'))
    if metadata['config_sha256'] != sha(config) or metadata['features_sha256'] != sha(rows):
        raise ValueError('Configuration or features changed; prepare and validate a new experiment explicitly')
    return rows, metadata


def versions():
    return {'python': platform.python_version(), **{p:importlib.metadata.version(p) for p in
            ['numpy','pandas','scikit-learn','scipy','pandas_market_calendars','joblib']}}


def predictions_file(path, frame, values):
    result = frame[['price_date','target_date','forecast_at','target_close','close_at_forecast']].copy()
    for name, prediction in values.items():
        result[name] = prediction
    result.to_csv(path, index=False)


def validate(config, root=ROOT):
    root = Path(root)
    rows, metadata = input_data(config, root)
    splits = split_rows(rows, config)
    train, val = pd.DataFrame(splits['train']), pd.DataFrame(splits['validation'])
    candidates = [{'name':'naive_last_close', 'family':'naive', 'group':'last_close', 'alpha':None,
                   'metrics':metrics(val['target_close'], predict_close(None, val))}]
    prediction_values = {'naive_last_close':predict_close(None,val)}
    trials = []
    for family in ('ridge','random_forest'):
        for group in GROUPS:
            options = config['ridge_alphas'] if family == 'ridge' else [None]
            best, best_prediction = None, None
            for alpha in options:
                model = model_pipeline(family, group, config, alpha)
                model.fit(train, train['target_return'])
                prediction = predict_close(model, val)
                trial = {'name':f'{family}__{group}', 'family':family, 'group':group, 'alpha':alpha,
                         'metrics':metrics(val['target_close'], prediction)}
                trials.append(trial)
                if best is None or trial['metrics']['mae_usd'] < best['metrics']['mae_usd']:
                    best, best_prediction = trial, prediction
            candidates.append(best)
            prediction_values[best['name']] = best_prediction
            print(f"Validation {best['name']}: MAE {best['metrics']['mae_usd']:.4f} USD", flush=True)
    winner = min(candidates, key=lambda c:c['metrics']['mae_usd'])
    learned = min(candidates[1:], key=lambda c:c['metrics']['mae_usd'])
    baseline_mae = candidates[0]['metrics']['mae_usd']
    for candidate in candidates:
        candidate['improvement_vs_naive_percent'] = 100*(1-candidate['metrics']['mae_usd']/baseline_mae)
    protocol = {'dataset':metadata, 'config':config, 'candidates':candidates,
                'selected_model':winner['name'], 'best_learned_model':learned['name'],
                'selection':'Minimum validation MAE; ties keep earlier candidate. No test-based tuning.',
                'estimator_target':'Next-session simple return, converted to closing price using last close. Final target/metrics are USD price.',
                'versions':versions()}
    protocol['protocol_sha256'] = sha(protocol)
    reports = root/config['reports_dir']
    report = {**protocol, 'trials':trials, 'test_evaluated':False,
              'text_effect':{family:{'mae_difference_text_minus_market':
                next(c['metrics']['mae_usd'] for c in candidates if c['name']==family+'__market_text')-
                next(c['metrics']['mae_usd'] for c in candidates if c['name']==family+'__market')}
                for family in ('ridge','random_forest')}}
    protocol_path = reports/'msft_protocol.json'
    test_path = reports/'msft_test.json'
    if test_path.exists():
        old = json.loads(test_path.read_text(encoding='utf-8'))
        if old['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Holdout was already evaluated for another protocol. Preserve it; use a new report directory for a disclosed new experiment.')
    write_json(protocol_path, protocol)
    write_json(reports/'msft_validation.json', report)
    predictions_file(root/config['processed_dir']/'validation_predictions.csv', val, prediction_values)
    return report


def evaluate_test(config, root=ROOT):
    root = Path(root)
    rows, metadata = input_data(config, root)
    reports = root/config['reports_dir']
    protocol = json.loads((reports/'msft_protocol.json').read_text(encoding='utf-8'))
    check = dict(protocol)
    expected = check.pop('protocol_sha256')
    if sha(check) != expected or protocol['dataset'] != metadata or protocol['config'] != config or protocol['versions'] != versions():
        raise ValueError('Frozen protocol/data/environment mismatch')
    path = reports/'msft_test.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['protocol_sha256'] != expected:
            raise ValueError('Another holdout protocol already exists')
        return result
    splits = split_rows(rows, config)
    fit = pd.DataFrame(splits['train']+splits['validation'])
    test = pd.DataFrame(splits['test'])
    model_dir = root/config['model_dir']
    model_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = [], {}
    for candidate in protocol['candidates']:
        model = None
        if candidate['family'] != 'naive':
            model = model_pipeline(candidate['family'], candidate['group'], config, candidate['alpha'])
            model.fit(fit, fit['target_return'])
            joblib.dump({'pipeline':model, 'candidate':candidate, 'protocol_sha256':expected,
                         'fit_last_target':fit['target_date'].iloc[-1], 'purpose':'frozen holdout model'},
                        model_dir/(candidate['name']+'.joblib'))
        prediction = predict_close(model,test)
        predictions[candidate['name']] = prediction
        results.append({'name':candidate['name'], 'metrics':metrics(test['target_close'],prediction)})
    base = results[0]['metrics']['mae_usd']
    for row in results:
        row['improvement_vs_naive_percent'] = 100*(1-row['metrics']['mae_usd']/base)
    result = {'protocol_sha256':expected,'dataset_sha256':metadata['snapshot_sha256'],
              'selected_on_validation':protocol['selected_model'],
              'best_learned_on_validation':protocol['best_learned_model'],
              'test_period':metadata['splits']['test'],'results':results,
              'interpretation':'All variants fixed on validation before test evaluation. Test results do not change the selected model.',
              'text_effect':{family:{'mae_difference_text_minus_market':
                next(r['metrics']['mae_usd'] for r in results if r['name']==family+'__market_text')-
                next(r['metrics']['mae_usd'] for r in results if r['name']==family+'__market')}
                for family in ('ridge','random_forest')}}
    predictions_file(root/config['processed_dir']/'test_predictions.csv',test,predictions)
    write_json(path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/msft_experiment.json')
    parser.add_argument('--stage', choices=('validate','test'), default='validate')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    result = validate(config) if args.stage=='validate' else evaluate_test(config)
    keys = ['selected_model','best_learned_model','text_effect','protocol_sha256'] if args.stage=='validate' else ['selected_on_validation','test_period','results','text_effect']
    print(json.dumps({k:result[k] for k in keys}, indent=2))
    return 0
