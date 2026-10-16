"""One feature function for both historical rows and unlabeled forecasts."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from .msft_ingest import ROOT, canonical

PRICE_FEATURES = ['return_1', 'return_2', 'return_5', 'return_20',
                  'return_lag_1', 'return_lag_2', 'return_lag_5',
                  'close_to_mean_5', 'close_to_mean_20', 'volatility_5', 'volatility_20',
                  'intraday_range', 'intraday_body', 'opening_gap']
VOLUME_FEATURES = ['log_volume', 'volume_to_mean_5', 'volume_to_mean_20', 'volume_change']
TEXT_FEATURES = ['news_count_7d', 'news_count_1d', 'news_words_mean', 'news_age_days']
CALENDAR_FEATURES = ['weekday', 'month']


def sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def utc_timestamp(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError('Timestamp must include its timezone')
    return stamp.tz_convert('UTC')


def schedule(config):
    return mcal.get_calendar(config['calendar']).schedule(
        start_date=pd.Timestamp(config['start']) - pd.Timedelta(days=10),
        end_date=pd.Timestamp(config['end']) + pd.Timedelta(days=14))


def load_snapshot(config, root=ROOT):
    path = Path(root) / config['database']
    if not path.exists():
        raise FileNotFoundError('Run MSFT ingestion first')
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        prices = [dict(r) for r in db.execute('''SELECT p.trade_date,p.open,p.high,p.low,p.close,p.volume,
          p.adjusted_close,p.version_hash,r.sha256 AS raw_sha256 FROM prices p
          JOIN raw_responses r ON p.raw_id=r.id
          WHERE p.ticker=? AND p.trade_date BETWEEN ? AND ? ORDER BY p.trade_date''',
          (config['ticker'], config['start'], config['end']))]
        articles = [dict(r) for r in db.execute('''SELECT a.url,a.title,a.text,a.published_at,a.modified_at,
          a.first_fetched_at,a.version_hash,r.sha256 AS raw_sha256 FROM articles a
          JOIN raw_responses r ON a.raw_id=r.id ORDER BY a.published_at,a.url''')]
        # Event responses are versioned too; use the latest observation for each key.
        events = [dict(r) for r in db.execute('''SELECT kind,event_key,payload_json FROM (
          SELECT c.*,ROW_NUMBER() OVER(PARTITION BY ticker,kind,event_key
          ORDER BY r.last_fetched_at DESC,r.id DESC) AS rank
          FROM corporate_actions c JOIN raw_responses r ON c.raw_id=r.id WHERE ticker=?)
          WHERE rank=1 ORDER BY kind,event_key''', (config['ticker'],))]
    finally:
        db.close()
    return {'prices': prices, 'articles': articles, 'events': events}


def validate_prices(prices, sessions, config):
    if not prices:
        raise ValueError('Empty prices')
    days = [r['trade_date'] for r in prices]
    expected = list(sessions.loc[config['start']:config['end']].index.strftime('%Y-%m-%d'))
    missing, extra = sorted(set(expected) - set(days)), sorted(set(days) - set(expected))
    if missing or extra or days != sorted(set(days)):
        raise ValueError(f'Calendar mismatch: missing={missing}, extra={extra}, duplicate/unsorted={days != sorted(set(days))}')
    for row in prices:
        o, h, l, c, v = (row[k] for k in ['open', 'high', 'low', 'close', 'volume'])
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (o,h,l,c,v)) or min(o,h,l,c) <= 0 or v <= 0 or h < max(o,c) or l > min(o,c) or h < l:
            raise ValueError(f'Invalid OHLCV: {row["trade_date"]}')
    returns = np.diff(np.log([r['close'] for r in prices]))
    large = [{'date': days[i+1], 'log_return': float(r)} for i,r in enumerate(returns) if abs(r) > 0.15]
    return {'expected_sessions': len(expected), 'price_rows': len(prices), 'missing_sessions': missing,
            'extra_sessions': extra, 'large_moves_over_15pct_log': large,
            'outlier_policy': 'Keep valid large market moves; no clipping or forward/back filling.'}


def check_actions(events, prices, sessions, config):
    days = set(sessions.index.strftime('%Y-%m-%d'))
    dividends, splits = [], []
    for event in events:
        data = json.loads(event['payload_json'])
        day = pd.Timestamp(data['date'], unit='s', tz='UTC').tz_convert('America/New_York').date().isoformat()
        if not config['start'] <= day <= config['end']:
            continue
        if day not in days:
            raise ValueError(f'Corporate event on non-session: {day}')
        if event['kind'] == 'dividends':
            amount = data['amount']
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError('Invalid dividend amount')
            dividends.append({'date': day, 'amount_usd': amount})
        elif event['kind'] == 'splits':
            splits.append({'date': day, **data})
    if splits:
        raise ValueError('Splits require an explicit point-in-time adjustment policy before training')
    ratios = [r['adjusted_close']/r['close'] for r in prices if r['adjusted_close'] is not None]
    return {'dividends': dividends, 'splits': splits, 'adjusted_to_close_min': min(ratios) if ratios else None,
            'adjusted_to_close_max': max(ratios) if ratios else None,
            'policy': 'Use ordinary OHLC/close, not dividend-adjusted close. Dividends stay in actual price changes; no future dividend/event predictors. No splits in the downloaded period.',
            'verification_scope': 'Yahoo events and calendar consistency; not an independent reconciliation of every dividend amount.'}


def event_category(title):
    title = title.lower()
    if re.search(r'earnings|financial results|quarterly results', title):
        return 'earnings'
    if 'dividend' in title:
        return 'dividend'
    if re.search(r'acqui|merger', title):
        return 'acquisition'
    if re.search(r'launch|introduc|unveil|announces.*(?:surface|windows|azure)', title):
        return 'product'
    return 'other'


def feature_at(prices, articles, forecast_at, target_date, config):
    """No labels or learned transforms. Caller supplies only completed price rows."""
    cutoff = utc_timestamp(forecast_at)
    if len(prices) < 21:
        raise ValueError('At least 21 completed price sessions required')
    if prices[-1]['trade_date'] >= target_date:
        raise ValueError('Target-day prices are not available at forecast time')
    closes = np.array([r['close'] for r in prices[-21:]], dtype=float)
    volumes = np.array([r['volume'] for r in prices[-21:]], dtype=float)
    returns = closes[1:] / closes[:-1] - 1
    last = prices[-1]
    result = {'price_date': last['trade_date'], 'target_date': target_date,
              'forecast_at': cutoff.isoformat(), 'close_at_forecast': float(closes[-1])}
    for lag in (1,2,5,20):
        result[f'return_{lag}'] = float(closes[-1]/closes[-lag-1]-1)
    for lag in (1,2,5):
        result[f'return_lag_{lag}'] = float(returns[-lag-1])
    for window in (5,20):
        result[f'close_to_mean_{window}'] = float(closes[-1]/closes[-window:].mean()-1)
        result[f'volatility_{window}'] = float(returns[-window:].std(ddof=1))
        result[f'volume_to_mean_{window}'] = float(volumes[-1]/volumes[-window:].mean()-1)
    result.update(intraday_range=(last['high']-last['low'])/last['close'],
                  intraday_body=last['close']/last['open']-1,
                  opening_gap=last['open']/closes[-2]-1,
                  log_volume=float(np.log1p(volumes[-1])), volume_change=float(volumes[-1]/volumes[-2]-1))
    target = pd.Timestamp(target_date)
    result.update(weekday=str(target.dayofweek), month=str(target.month))
    lower = cutoff - pd.Timedelta(days=config['text_window_days'])
    selected = [a for a in articles if lower < utc_timestamp(a['published_at']) <= cutoff]
    selected.sort(key=lambda a: (utc_timestamp(a['published_at']), a['url']))
    result.update(news_count_7d=len(selected),
                  news_count_1d=sum(utc_timestamp(a['published_at']) > cutoff-pd.Timedelta(days=1) for a in selected),
                  news_words_mean=float(np.mean([len(a['text'].split()) for a in selected])) if selected else 0.0,
                  news_age_days=(cutoff-utc_timestamp(selected[-1]['published_at'])).total_seconds()/86400 if selected else float(config['text_window_days']),
                  event_category=event_category(selected[-1]['title']) if selected else 'none',
                  news_text='\n'.join(a['text'] for a in selected),
                  article_urls=[a['url'] for a in selected],
                  max_publication_at=selected[-1]['published_at'] if selected else None)
    return result


def build_rows(snapshot, sessions, config):
    prices, articles = snapshot['prices'], snapshot['articles']
    days = list(sessions.index.strftime('%Y-%m-%d'))
    positions = {day:i for i,day in enumerate(days)}
    rows = []
    for i in range(20, len(prices)):
        target = days[positions[prices[i]['trade_date']]+1]
        forecast = sessions.loc[target, 'market_open'] - pd.Timedelta(minutes=config['minutes_before_open'])
        row = feature_at(prices[:i+1], articles, forecast, target, config)
        row['target_available_at'] = sessions.loc[target, 'market_close'].isoformat()
        if i+1 < len(prices):
            if prices[i+1]['trade_date'] != target:
                raise ValueError('Label is not the next calendar session')
            row['target_close'] = prices[i+1]['close']
            row['target_return'] = row['target_close']/row['close_at_forecast']-1
        else:
            row['target_close'] = row['target_return'] = None
        rows.append(row)
    return rows


def split_rows(rows, config):
    labeled = [r for r in rows if r['target_close'] is not None]
    n_val, n_test = config['validation_sessions'], config['test_sessions']
    first_val, first_test = len(labeled)-n_val-n_test, len(labeled)-n_test
    if first_val < 252:
        raise ValueError('Need at least 252 training rows plus validation and test')
    splits = {'train': labeled[:first_val], 'validation': labeled[first_val:first_test], 'test': labeled[first_test:]}
    for previous, following in (('train','validation'), ('validation','test')):
        cutoff = utc_timestamp(splits[following][0]['forecast_at'])
        if any(utc_timestamp(r['target_available_at']) >= cutoff for r in splits[previous]):
            raise ValueError('Training labels cross the next forecast cutoff')
    return splits


def prepare(config, root=ROOT, snapshot=None):
    root = Path(root)
    sessions = schedule(config)
    snapshot = load_snapshot(config, root) if snapshot is None else snapshot
    quality = validate_prices(snapshot['prices'], sessions, config)
    quality['corporate_actions'] = check_actions(snapshot['events'], snapshot['prices'], sessions, config)
    rows = build_rows(snapshot, sessions, config)
    splits = split_rows(rows, config)
    folder = root / config['processed_dir']
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_hash = sha(snapshot)
    feature_hash = sha(rows)
    metadata = {'snapshot_sha256': snapshot_hash, 'features_sha256': feature_hash,
                'config_sha256': sha(config), 'config': config,
                'calendar_version': importlib.metadata.version('pandas_market_calendars'),
                'splits': {k:{'rows':len(v),'first_target':v[0]['target_date'],'last_target':v[-1]['target_date']} for k,v in splits.items()}}
    if (folder/'manifest.json').exists():
        previous = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if previous != metadata:
            raise ValueError('Frozen dataset differs. Preserve this snapshot and choose new processed/reports/model directories for a new experiment.')
    write_json(folder/'snapshot.json', snapshot)
    write_json(folder/'features.json', rows)
    write_json(folder/'manifest.json', metadata)
    # Human-readable numeric export; full texts remain in local JSON.
    pd.DataFrame(rows).drop(columns=['news_text','article_urls']).to_csv(folder/'features.csv', index=False)
    sessions.to_csv(folder/'calendar.csv', index_label='session')
    quality.update(metadata)
    quality.update(feature_rows=len(rows)-1, warmup_sessions=20,
                   latest_unlabeled_target=rows[-1]['target_date'],
                   text_rows_by_split={k:{'with_text':sum(bool(r['news_text']) for r in v),'total':len(v)} for k,v in splits.items()},
                   forecast_rule='30 minutes before the next NASDAQ regular session opens (09:00 New York for this period). Previous session OHLCV only.',
                   limitations=['Current text editions assigned to original published_at; historical revision leakage cannot be excluded.',
                                'Current downloaded price revisions are used; actual historical Yahoo availability at 09:00 is assumed, not measured.',
                                'Downloaded Microsoft press-release archive does not represent all market news.'])
    write_json(root/config['reports_dir']/'msft_data_quality.json', quality)
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/msft_experiment.json')
    parser.add_argument('--snapshot', type=Path, help='Rebuild features from a retained JSON snapshot instead of the mutable database')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8')) if args.snapshot else None
    result = prepare(config, snapshot=snapshot)
    print(json.dumps({k:result[k] for k in ['price_rows','feature_rows','missing_sessions','splits','text_rows_by_split','snapshot_sha256']}, indent=2))
    return 0
