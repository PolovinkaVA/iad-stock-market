"""Incremental MSFT ingestion, immutable raw responses, versions and run logs.

Only the standard library is needed (Windows also needs IANA timezone data).
Calendar completeness and point-in-time availability are checked downstream.
"""
import argparse
import hashlib
import json
import math
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .microsoft_source import Page, article_text, parse_date, url_date

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('open', 'high', 'low', 'close', 'volume')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
        status TEXT NOT NULL, parameters_json TEXT NOT NULL, summary_json TEXT);
      CREATE TABLE IF NOT EXISTS raw_responses (
        id INTEGER PRIMARY KEY, url TEXT NOT NULL, sha256 TEXT NOT NULL,
        path TEXT NOT NULL, first_fetched_at TEXT NOT NULL, last_fetched_at TEXT NOT NULL,
        UNIQUE(url, sha256));
      CREATE TABLE IF NOT EXISTS observations (
        run_id TEXT NOT NULL REFERENCES runs(id), raw_id INTEGER NOT NULL REFERENCES raw_responses(id),
        fetched_at TEXT NOT NULL, PRIMARY KEY(run_id, raw_id, fetched_at));
      CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
        source TEXT NOT NULL, url TEXT, message TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS article_queue (
        url TEXT PRIMARY KEY, publication_date TEXT NOT NULL, last_success_at TEXT);
      CREATE TABLE IF NOT EXISTS article_versions (
        id INTEGER PRIMARY KEY, url TEXT NOT NULL REFERENCES article_queue(url),
        version_hash TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
        published_at TEXT NOT NULL, modified_at TEXT, text_sha256 TEXT NOT NULL,
        first_fetched_at TEXT NOT NULL, last_fetched_at TEXT NOT NULL,
        raw_id INTEGER NOT NULL REFERENCES raw_responses(id), UNIQUE(url, version_hash));
      CREATE TABLE IF NOT EXISTS price_versions (
        id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, trade_date TEXT NOT NULL,
        version_hash TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
        close REAL NOT NULL, volume REAL NOT NULL, adjusted_close REAL,
        first_fetched_at TEXT NOT NULL, last_fetched_at TEXT NOT NULL,
        raw_id INTEGER NOT NULL REFERENCES raw_responses(id), UNIQUE(ticker, trade_date, version_hash));
      CREATE TABLE IF NOT EXISTS corporate_actions (
        ticker TEXT NOT NULL, kind TEXT NOT NULL, event_key TEXT NOT NULL, payload_json TEXT NOT NULL,
        raw_id INTEGER NOT NULL REFERENCES raw_responses(id),
        PRIMARY KEY(ticker, kind, event_key, raw_id));
      CREATE VIEW IF NOT EXISTS articles AS
        SELECT * FROM (SELECT *, ROW_NUMBER() OVER
          (PARTITION BY url ORDER BY last_fetched_at DESC, id DESC) AS version_rank
          FROM article_versions) WHERE version_rank=1;
      CREATE VIEW IF NOT EXISTS prices AS
        SELECT * FROM (SELECT *, ROW_NUMBER() OVER
          (PARTITION BY ticker, trade_date ORDER BY last_fetched_at DESC, id DESC) AS version_rank
          FROM price_versions) WHERE version_rank=1;
    ''')
    return db


class HTTP:
    def __init__(self, config):
        self.config = config
        self.last_request = 0
        self.requests = 0

    def get(self, url):
        for attempt in range(self.config['attempts']):
            time.sleep(max(0, self.config['request_interval_seconds'] - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            self.requests += 1
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'MSFTSemesterResearch/0.3 (educational project)'})
                with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                    return response.read(), now()
            except (urllib.error.URLError, TimeoutError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt + 1 == self.config['attempts']:
                    raise
                retry_after = exc.headers.get('Retry-After', '0') if isinstance(exc, urllib.error.HTTPError) else '0'
                delay = float(retry_after) if retry_after.isdigit() else 0
                time.sleep(min(30, max(delay, 2 ** (attempt + 1))))


def parse_prices(raw, ticker, start, end, zone):
    chart = json.loads(raw)['chart']
    if chart.get('error') or not chart.get('result'):
        raise ValueError(f'Yahoo returned no result: {chart.get("error")}')
    data = chart['result'][0]
    if data['meta']['symbol'] != ticker or data['meta'].get('currency') != 'USD':
        raise ValueError('Unexpected instrument or currency')
    stamps = data.get('timestamp', [])
    if not stamps:
        raise ValueError('Empty price response; inspect source/calendar before continuing')
    values = data['indicators']['quote'][0]
    adjusted = data['indicators'].get('adjclose', [{}])[0].get('adjclose', [None] * len(stamps))
    if any(len(values[k]) != len(stamps) for k in FIELDS) or len(adjusted) != len(stamps):
        raise ValueError('Inconsistent price array lengths')
    rows, seen = [], set()
    for i, stamp in enumerate(stamps):
        day = datetime.fromtimestamp(stamp, ZoneInfo(zone)).date().isoformat()
        if not start <= day <= end:
            continue
        if day in seen:
            raise ValueError(f'Duplicate price date: {day}')
        seen.add(day)
        row = {k: values[k][i] for k in FIELDS}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row.values()):
            raise ValueError(f'Missing/nonfinite OHLCV: {day}')
        if min(row[k] for k in FIELDS[:4]) <= 0 or row['volume'] < 0 or row['high'] < max(row['open'], row['close']) or row['low'] > min(row['open'], row['close']) or row['high'] < row['low']:
            raise ValueError(f'Invalid OHLCV: {day}')
        adj = adjusted[i]
        if adj is not None and (not isinstance(adj, (int, float)) or not math.isfinite(adj) or adj <= 0):
            raise ValueError(f'Invalid adjusted close: {day}')
        row.update(trade_date=day, adjusted_close=adj)
        rows.append(row)
    if [r['trade_date'] for r in rows] != sorted(seen):
        raise ValueError('Unsorted trading dates')
    return rows, data.get('events', {})


class Ingestion:
    def __init__(self, config, end, *, root=ROOT, fetcher=None, mode='network', full=False, source='all'):
        self.config, self.end, self.root = config, end, Path(root)
        self.start, self.full = config['start'], full
        self.db = connect(self.root / config['database'])
        self.raw_dir = self.root / config['raw_dir']
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.http = fetcher or HTTP(config)
        self.run_id = str(uuid.uuid4())
        self.summary = {'mode': mode, 'prices_seen': 0, 'price_versions_added': 0,
                        'articles_seen': 0, 'article_versions_added': 0, 'archive_pages': 0,
                        'errors': 0, 'network_requests': 0}
        self.db.execute('INSERT INTO runs VALUES(?,?,NULL,?,?,NULL)',
                        (self.run_id, now(), 'running', canonical({'config': config, 'end': end, 'mode': mode, 'full': full, 'source': source})))
        self.db.commit()

    def error(self, source, url, exc):
        self.db.rollback()
        self.db.execute('INSERT INTO errors(run_id,source,url,message) VALUES(?,?,?,?)',
                        (self.run_id, source, url, str(exc)))
        self.db.commit()
        self.summary['errors'] += 1
        print(f'{source}: {exc}', flush=True)

    def save_raw(self, url, raw, fetched_at):
        fetched_at = parse_date(fetched_at).astimezone(timezone.utc).isoformat()
        checksum = digest(raw)
        path = self.raw_dir / (checksum + '.bin')
        if path.exists():
            if digest(path.read_bytes()) != checksum:
                raise ValueError(f'Raw file checksum mismatch: {path}')
        else:
            temp = path.with_suffix('.' + self.run_id + '.tmp')
            temp.write_bytes(raw)
            temp.replace(path)
        self.db.execute('''INSERT INTO raw_responses(url,sha256,path,first_fetched_at,last_fetched_at)
          VALUES(?,?,?,?,?) ON CONFLICT(url,sha256) DO UPDATE SET
          first_fetched_at=MIN(first_fetched_at,excluded.first_fetched_at),
          last_fetched_at=MAX(last_fetched_at,excluded.last_fetched_at)''',
          (url, checksum, str(path.relative_to(self.root)), fetched_at, fetched_at))
        raw_id = self.db.execute('SELECT id FROM raw_responses WHERE url=? AND sha256=?', (url, checksum)).fetchone()[0]
        self.db.execute('INSERT OR IGNORE INTO observations VALUES(?,?,?)', (self.run_id, raw_id, fetched_at))
        self.db.commit()  # Keep invalid source responses for diagnosis, too.
        return raw_id, fetched_at

    def save_article(self, url, raw, fetched_at):
        raw_id, fetched_at = self.save_raw(url, raw, fetched_at)
        page = Page()
        page.feed(raw.decode('utf-8'))
        published = parse_date(page.meta.get('article:published_time'))
        modified = parse_date(page.meta.get('article:modified_time'))
        text = article_text(page)
        if published is None or not text.strip():
            raise ValueError(f'Missing publication timestamp/body: {url}')
        title = page.meta.get('og:title', '')
        published = published.astimezone(timezone.utc).isoformat()
        modified = modified.astimezone(timezone.utc).isoformat() if modified else None
        version = digest(canonical([title, text, published, modified]).encode())
        self.db.execute('INSERT OR IGNORE INTO article_queue(url,publication_date) VALUES(?,?)', (url, url_date(url)))
        cursor = self.db.execute('''INSERT OR IGNORE INTO article_versions
          (url,version_hash,title,text,published_at,modified_at,text_sha256,first_fetched_at,last_fetched_at,raw_id)
          VALUES(?,?,?,?,?,?,?,?,?,?)''',
          (url, version, title, text, published, modified, digest(text.encode()), fetched_at, fetched_at, raw_id))
        added = cursor.rowcount
        self.db.execute('''UPDATE article_versions SET first_fetched_at=MIN(first_fetched_at,?),
          last_fetched_at=MAX(last_fetched_at,?) WHERE url=? AND version_hash=?''', (fetched_at, fetched_at, url, version))
        self.db.execute('UPDATE article_queue SET last_success_at=MAX(COALESCE(last_success_at,?),?) WHERE url=?', (fetched_at, fetched_at, url))
        self.db.commit()
        self.summary['articles_seen'] += 1
        self.summary['article_versions_added'] += added

    def save_prices(self, url, raw, fetched_at):
        raw_id, fetched_at = self.save_raw(url, raw, fetched_at)
        rows, events = parse_prices(raw, self.config['ticker'], self.start, self.end, self.config['timezone'])
        added = 0
        for row in rows:
            version = digest(canonical(row).encode())
            key = (self.config['ticker'], row['trade_date'], version)
            cursor = self.db.execute('''INSERT OR IGNORE INTO price_versions
              (ticker,trade_date,version_hash,open,high,low,close,volume,adjusted_close,first_fetched_at,last_fetched_at,raw_id)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
              (*key, *(row[k] for k in FIELDS), row['adjusted_close'], fetched_at, fetched_at, raw_id))
            added += cursor.rowcount
            self.db.execute('''UPDATE price_versions SET first_fetched_at=MIN(first_fetched_at,?),
              last_fetched_at=MAX(last_fetched_at,?) WHERE ticker=? AND trade_date=? AND version_hash=?''', (fetched_at, fetched_at, *key))
        for kind, entries in events.items():
            for event_key, payload in entries.items():
                self.db.execute('INSERT OR IGNORE INTO corporate_actions VALUES(?,?,?,?,?)',
                                (self.config['ticker'], kind, event_key, canonical(payload), raw_id))
        self.db.commit()
        self.summary['prices_seen'] += len(rows)
        self.summary['price_versions_added'] += added

    def import_audit(self):
        cache = self.root / self.config['audit_cache']
        files = sorted(cache.glob('*.json'))
        if not files:
            raise FileNotFoundError(f'Audit cache is empty: {cache}')
        for path in files:
            url = str(path)
            try:
                metadata = json.loads(path.read_text(encoding='utf-8'))
                url = metadata['url']
                # Windows text-mode output replaced each LF with CRLF, including
                # existing CRLF -> CRCRLF. Undo that once without universal-newline
                # decoding, which would irreversibly merge distinct line endings.
                raw = path.with_suffix('.txt').read_bytes()
                if digest(raw) != metadata['sha256']:
                    raw = raw.replace(b'\r\n', b'\n')
                if digest(raw) != metadata['sha256']:
                    raise ValueError(f'Audit checksum mismatch: {url}')
                if 'query1.finance.yahoo.com/v8/finance/chart/' in url:
                    self.save_prices(url, raw, metadata['fetched_at_utc'])
                elif '/source/tag/press-releases/' in url:
                    self.save_raw(url, raw, metadata['fetched_at_utc'])
                    self.summary['archive_pages'] += 1
                elif self.start <= url_date(url) <= self.end:
                    self.save_article(url, raw, metadata['fetched_at_utc'])
            except Exception as exc:
                self.error('audit_import', url, exc)

    def update_prices(self):
        latest = self.db.execute('SELECT MAX(trade_date) FROM prices WHERE ticker=? AND trade_date<=?', (self.config['ticker'], self.end)).fetchone()[0]
        start = self.start if self.full or not latest else max(self.start, (date.fromisoformat(latest) - timedelta(days=self.config['price_overlap_days'])).isoformat())
        begin = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
        stop = int((datetime.fromisoformat(self.end).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{self.config["ticker"]}?period1={begin}&period2={stop}&interval=1d&events=div%2Csplits'
        try:
            self.save_prices(url, *self.http.get(url))
        except Exception as exc:
            self.error('yahoo', url, exc)

    def update_articles(self):
        latest = self.db.execute('SELECT MAX(publication_date) FROM article_queue WHERE last_success_at IS NOT NULL AND publication_date<=?', (self.end,)).fetchone()[0]
        boundary = self.start if self.full or not latest else max(self.start, (date.fromisoformat(latest) - timedelta(days=self.config['discovery_overlap_days'])).isoformat())
        reached = False
        for number in range(1, self.config['max_archive_pages'] + 1):
            url = self.config['archive_url'] + (f'page/{number}/' if number > 1 else '')
            try:
                raw, fetched = self.http.get(url)
                self.save_raw(url, raw, fetched)
                page = Page()
                page.feed(raw.decode('utf-8'))
                if not page.links:
                    raise ValueError('Archive has no dated article links')
                for link in sorted(page.links):
                    if self.start <= url_date(link) <= self.end:
                        self.db.execute('INSERT OR IGNORE INTO article_queue(url,publication_date) VALUES(?,?)', (link, url_date(link)))
                self.db.commit()
                self.summary['archive_pages'] += 1
                if max(url_date(link) for link in page.links) < boundary:
                    reached = True
                    break
            except Exception as exc:
                self.error('archive', url, exc)
                break
        if not reached:
            self.error('archive', self.config['archive_url'], ValueError('Discovery boundary not reached; archive scan incomplete'))
        pending = self.db.execute('SELECT url FROM article_queue WHERE last_success_at IS NULL AND publication_date BETWEEN ? AND ? ORDER BY publication_date,url', (self.start, self.end)).fetchall()
        stale_before = (datetime.now(timezone.utc) - timedelta(days=self.config['article_refresh_days'])).isoformat()
        if self.full:
            stale = self.db.execute('SELECT url FROM article_queue WHERE last_success_at IS NOT NULL AND publication_date BETWEEN ? AND ? ORDER BY last_success_at,url', (self.start, self.end)).fetchall()
        else:
            stale = self.db.execute('SELECT url FROM article_queue WHERE last_success_at < ? AND publication_date BETWEEN ? AND ? ORDER BY last_success_at,url LIMIT ?', (stale_before, self.start, self.end, self.config['article_refresh_limit'])).fetchall()
        for row in [*pending, *stale]:
            try:
                self.save_article(row['url'], *self.http.get(row['url']))
            except Exception as exc:
                self.error('article', row['url'], exc)

    def finish(self):
        self.summary['network_requests'] = getattr(self.http, 'requests', 0)
        self.summary['price_rows_total'] = self.db.execute('SELECT COUNT(*) FROM prices').fetchone()[0]
        self.summary['articles_total'] = self.db.execute('SELECT COUNT(*) FROM articles').fetchone()[0]
        self.summary['pending_articles'] = self.db.execute('SELECT COUNT(*) FROM article_queue WHERE last_success_at IS NULL').fetchone()[0]
        status = 'partial' if self.summary['errors'] else 'success'
        if self.summary['errors'] and not (self.summary['prices_seen'] or self.summary['articles_seen'] or self.summary['archive_pages']):
            status = 'failed'
        self.db.execute('UPDATE runs SET finished_at=?,status=?,summary_json=? WHERE id=?', (now(), status, canonical(self.summary), self.run_id))
        self.db.commit()
        return {'run_id': self.run_id, 'status': status, **self.summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/msft.json')
    parser.add_argument('--end', help='Inclusive last date; defaults to yesterday in New York')
    parser.add_argument('--import-audit', action='store_true', help='Import verified audit snapshots, no network')
    parser.add_argument('--full', action='store_true', help='Refresh full price history, archive and all known articles')
    parser.add_argument('--source', choices=('all', 'prices', 'articles'), default='all', help='Select a source for a network update')
    parser.add_argument('--report', type=Path, help='Write this run summary as JSON')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    yesterday = datetime.now(ZoneInfo(config['timezone'])).date() - timedelta(days=1)
    end = date.fromisoformat(args.end) if args.end else yesterday
    if not date.fromisoformat(config['start']) <= end <= yesterday:
        parser.error('end must be between configured start and yesterday in New York (completed daily bars only)')
    if args.full and args.import_audit:
        parser.error('--full cannot be used with --import-audit')
    if args.import_audit and args.source != 'all':
        parser.error('--source only applies to network updates')
    job = Ingestion(config, end.isoformat(), mode='audit_import' if args.import_audit else 'network', full=args.full, source=args.source)
    try:
        if args.import_audit:
            job.import_audit()
        else:
            if args.source in ('all', 'prices'):
                job.update_prices()
            if args.source in ('all', 'articles'):
                job.update_articles()
    except Exception as exc:
        job.error('run', None, exc)
    finally:
        result = job.finish()
        job.db.close()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return int(result['status'] != 'success')
