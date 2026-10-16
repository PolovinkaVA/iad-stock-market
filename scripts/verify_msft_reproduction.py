"""Rebuild a frozen experiment in a new directory from its retained snapshot."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stocks_forecast.msft_features import prepare, write_json
from stocks_forecast.msft_modeling import validate, evaluate_test, versions
from stocks_forecast.msft_explain import build_explanations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New, empty directory under project tmp/')
    args=parser.parse_args()
    target=args.output.resolve()
    if not target.is_relative_to((ROOT/'tmp').resolve()) or target==(ROOT/'tmp').resolve():
        raise ValueError('Choose a new subdirectory of project tmp/')
    target.mkdir(parents=True,exist_ok=False)
    for folder in ('src','scripts','config','tests','.streamlit'):
        shutil.copytree(ROOT/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cfg=json.loads((ROOT/'config/msft_experiment.json').read_text(encoding='utf-8'))
    snapshot_path=ROOT/cfg['processed_dir']/'snapshot.json'
    original_predictions=pd.read_csv(ROOT/cfg['processed_dir']/'test_predictions.csv')
    frozen={name:(ROOT/cfg['reports_dir']/name).read_bytes() for name in ('msft_protocol.json','msft_test.json')}
    (target/cfg['reports_dir']).mkdir()
    for name,data in frozen.items():
        (target/cfg['reports_dir']/name).write_bytes(data)
    quality=prepare(cfg,target,snapshot=json.loads(snapshot_path.read_text(encoding='utf-8')))
    validation=validate(cfg,target)
    evaluate_test(cfg,target,rebuild_artifacts=True)
    rebuilt=pd.read_csv(target/cfg['processed_dir']/'test_predictions.csv')
    if original_predictions.shape!=rebuilt.shape:
        raise ValueError('Prediction shape differs')
    numeric=list(original_predictions.select_dtypes(include='number').columns)
    categorical=[x for x in original_predictions.columns if x not in numeric]
    assert original_predictions[categorical].equals(rebuilt[categorical])
    delta=float(np.max(np.abs(original_predictions[numeric].to_numpy()-rebuilt[numeric].to_numpy())))
    if delta>1e-10:
        raise ValueError(f'Prediction difference: {delta}')
    explanations=build_explanations(cfg,target)
    assert (target/cfg['reports_dir']/'msft_test.json').read_bytes()==frozen['msft_test.json']
    for name,data in frozen.items():
        assert (ROOT/cfg['reports_dir']/name).read_bytes()==data
    result={'status':'passed','environment':versions(),'database_used':False,
            'input':'Retained research snapshot; live sources were not re-downloaded',
            'snapshot_sha256':quality['snapshot_sha256'],'protocol_sha256':validation['protocol_sha256'],
            'model_files_rebuilt':len(list((target/cfg['model_dir']).glob('*.joblib'))),
            'test_rows':len(rebuilt),'max_prediction_difference_usd':delta,
            'test_report_unchanged':True,'explained_rows':explanations['local_explanations']['rows'],
            'limitations':['Same Windows machine, new Python virtual environment; not a different operating system.',
                           'Source availability was tested separately; reproduction requires the retained snapshot.']}
    write_json(ROOT/'reports/msft_reproduction.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
