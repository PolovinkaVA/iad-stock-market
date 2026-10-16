"""Refresh a separate operational estimate without changing the frozen test."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from .msft_ingest import Ingestion, ROOT
from .msft_features import (load_snapshot, schedule, validate_prices, check_actions,
                            build_rows, sha, write_json, utc_timestamp)
from .msft_modeling import model_pipeline
from .msft_explain import reference, explain_row


def calculation_timing(created_at, forecast_at):
    return 'before_cutoff' if utc_timestamp(created_at)<utc_timestamp(forecast_at) else 'after_cutoff'


def calculate(config, created_at=None, root=ROOT):
    root=Path(root)
    created=utc_timestamp(created_at or datetime.now(timezone.utc))
    end=(created.tz_convert('America/New_York').date()-timedelta(days=1)).isoformat()
    active={**config,'end':end}
    snapshot=load_snapshot(active,root)
    snapshot['articles']=[a for a in snapshot['articles'] if utc_timestamp(a['published_at'])<=created]
    sessions=schedule(active)
    quality=validate_prices(snapshot['prices'],sessions,active)
    check_actions(snapshot['events'],snapshot['prices'],sessions,active)
    rows=build_rows(snapshot,sessions,active)
    fit=pd.DataFrame([r for r in rows if r['target_close'] is not None])
    latest=rows[-1]
    cutoff=utc_timestamp(latest['forecast_at'])
    if any(utc_timestamp(x)>=cutoff for x in fit['target_available_at']):
        raise ValueError('Operational fit includes a future label')
    protocol=json.loads((root/config['reports_dir']/'msft_protocol.json').read_text(encoding='utf-8'))
    check=dict(protocol); expected=check.pop('protocol_sha256')
    if sha(check)!=expected or protocol['config']!=config:
        raise ValueError('Frozen model-selection protocol differs')
    candidate=next(c for c in protocol['candidates'] if c['name']==protocol['best_learned_model'])
    if candidate['name']!='ridge__market' or protocol['selected_model']!='naive_last_close':
        raise ValueError('Operational implementation expects this project\'s frozen model choice')
    model=model_pipeline(candidate['family'],candidate['group'],config,candidate['alpha'])
    model.fit(fit,fit['target_return'])
    explanation=explain_row(model,latest,reference(model,fit))
    output={'created_at':created.isoformat(),'forecast_at':latest['forecast_at'],
            'target_date':latest['target_date'],'price_date':latest['price_date'],
            'timing':calculation_timing(created,cutoff), 'primary_model':'naive_last_close',
            'primary_prediction':latest['close_at_forecast'],'ridge_prediction':explanation['prediction'],
            'fit_through':fit['target_date'].iloc[-1],'fit_rows':len(fit),
            'price_rows':quality['price_rows'],'article_rows':len(snapshot['articles']),
            'snapshot_sha256':sha(snapshot),'protocol_sha256':expected,
            'explanation':explanation,'source_status':'cached_database',
            'limitations':['Calculation time may differ from nominal 09:00 cutoff; this is not proof of a forecast issued at that historical time.',
                           'Before cutoff the estimate is provisional; later publications may change inputs.',
                           'Price/text revisions and data-source delays remain limitations.']}
    folder=root/'data/processed/msft_live'; folder.mkdir(parents=True,exist_ok=True)
    model_dir=root/'models/msft_live';model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':model,'candidate':candidate,'fit_through':output['fit_through'],
                 'purpose':'operational estimate','protocol_sha256':expected},model_dir/'ridge.joblib')
    write_json(folder/'latest.json',output)
    return output


def update(config, root=ROOT, refresh=True, created_at=None):
    root=Path(root)
    log={'started_at':datetime.now(timezone.utc).isoformat(),'refresh':refresh}
    folder=root/'data/processed/msft_live';folder.mkdir(parents=True,exist_ok=True)
    try:
        if refresh:
            ingestion_config=json.loads((root/'config/msft.json').read_text(encoding='utf-8'))
            end=(datetime.now(ZoneInfo('America/New_York')).date()-timedelta(days=1)).isoformat()
            job=Ingestion(ingestion_config,end,root=root)
            try:
                job.update_prices();job.update_articles()
            except Exception as exc:
                job.error('run',None,exc)
            finally:
                ingestion=job.finish();job.db.close()
            log['ingestion']=ingestion
            if ingestion['status']!='success':
                raise RuntimeError('Source refresh incomplete; previous estimate retained. See ingestion errors.')
        result=calculate(config,created_at,root)
        result['source_status']='refreshed' if refresh else 'cached_database'
        write_json(folder/'latest.json',result)
        log.update(status='success',target_date=result['target_date'],snapshot_sha256=result['snapshot_sha256'])
        return result
    except Exception as exc:
        log.update(status='failed',error=str(exc))
        raise
    finally:
        log['finished_at']=datetime.now(timezone.utc).isoformat()
        with (folder/'runs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(log,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/msft_experiment.json')
    parser.add_argument('--refresh',action='store_true',help='Refresh both public sources before calculation')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    try:
        result=update(config,refresh=args.refresh)
        print(json.dumps({k:result[k] for k in ['target_date','timing','primary_prediction','ridge_prediction','fit_through']},indent=2))
        return 0
    except Exception as exc:
        print(f'Update failed: {exc}')
        return 1
