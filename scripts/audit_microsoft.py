"""Audit MSFT prices and Microsoft press releases; not a training pipeline.

Python standard library only. Raw downloads stay in ignored tmp/us_sources.
Run: python scripts/audit_microsoft.py --end 2026-10-13
Repeat with --offline to reproduce the report from downloaded snapshots.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp/us_sources/audit_cache'
REPORT = ROOT / 'reports/microsoft_audit.json'
MANIFEST = ROOT / 'reports/microsoft_articles.csv'
ARCHIVE = 'https://news.microsoft.com/source/tag/press-releases/'


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_main = False
        self.skip = 0
        self.links = set()
        self.meta = {}
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'main':
            self.in_main = True
        if self.in_main and tag in ('script', 'style', 'noscript'):
            self.skip += 1
        if tag == 'meta':
            self.meta[attrs.get('property', attrs.get('name', ''))] = attrs.get('content', '')
        if self.in_main and tag == 'a':
            href = attrs.get('href', '')
            if re.fullmatch(r'https://news\.microsoft\.com/source/\d{4}/\d{2}/\d{2}/[^/?#]+/', href):
                self.links.add(href)

    def handle_endtag(self, tag):
        if tag == 'main':
            self.in_main = False
        if self.in_main and tag in ('script', 'style', 'noscript'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if self.in_main and not self.skip:
            self.parts.append(text)

    @property
    def text(self):
        return ' '.join(' '.join(self.parts).split())


class Fetcher:
    def __init__(self, offline=False):
        self.offline = offline
        self.network_requests = 0
        self.cache_hits = 0
        self.lock = threading.Lock()
        self.last_request = 0.0
        CACHE.mkdir(parents=True, exist_ok=True)

    def get(self, url, refresh=False):
        key = hashlib.sha256(url.encode()).hexdigest()
        path = CACHE / (key + '.txt')
        metadata = CACHE / (key + '.json')
        if path.exists() and (self.offline or not refresh):
            self.cache_hits += 1
            return path.read_text(encoding='utf-8'), json.loads(metadata.read_text(encoding='utf-8'))
        if self.offline:
            raise FileNotFoundError(f'No cached response: {url}')
        for attempt in range(3):
            try:
                # At most two request starts per second across all workers.
                with self.lock:
                    delay = 0.6 - (time.monotonic() - self.last_request)
                    if delay > 0:
                        time.sleep(delay)
                    self.last_request = time.monotonic()
                    self.network_requests += 1
                request = urllib.request.Request(url, headers={'User-Agent': 'SemesterDataResearch/0.2 (educational source feasibility audit)'})
                with urllib.request.urlopen(request, timeout=20) as response:
                    raw = response.read()
                    meta = {'url': url, 'final_url': response.url, 'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                            'sha256': hashlib.sha256(raw).hexdigest(), 'status': response.status}
                text = raw.decode('utf-8')
                path.write_text(text, encoding='utf-8')
                metadata.write_text(json.dumps(meta, indent=2), encoding='utf-8')
                return text, meta
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))


def url_date(url):
    return '-'.join(re.search(r'/source/(\d{4})/(\d{2})/(\d{2})/', url).groups())


def parse_date(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamp has no timezone')
    return parsed


def inspect_article(url, fetcher):
    record = {'url': url, 'url_date': url_date(url)}
    try:
        html, metadata = fetcher.get(url)
        page = Page()
        page.feed(html)
        published_raw = page.meta.get('article:published_time', '')
        modified_raw = page.meta.get('article:modified_time', '')
        published, modified = parse_date(published_raw), parse_date(modified_raw)
        lag = (modified - published).total_seconds() / 86400 if published and modified else None
        text = page.text
        # Remove press-contact / editor boilerplate before counting usable text.
        for marker in ['For more information, press only:', 'Note to editors:']:
            text = text.split(marker, 1)[0]
        record.update({'status': 'ok', 'published_at': published_raw, 'modified_at': modified_raw,
                       'modification_lag_days': round(lag, 3) if lag is not None else None,
                       'text_chars': len(text), 'text_words': len(text.split()),
                       'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'raw_sha256': metadata['sha256'], 'fetched_at_utc': metadata['fetched_at_utc'],
                       'date_matches_url_utc': published.date().isoformat() == url_date(url) if published else False,
                       'late_modified': lag is not None and lag > 1,
                       'short_text': len(text.split()) < 80,
                       'earnings_link_notice': 'earnings-press-release-available' in url})
    except Exception as exc:
        record.update({'status': 'error', 'error': str(exc)})
    return record


def audit_prices(fetcher, start, end):
    begin = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    stop = int((datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/MSFT?period1={begin}&period2={stop}&interval=1d'
    text, metadata = fetcher.get(url)
    data = json.loads(text)['chart']['result'][0]
    stamps = data['timestamp']
    values = data['indicators']['quote'][0]
    dates = [datetime.fromtimestamp(t, timezone.utc).date().isoformat() for t in stamps]
    missing = {k: sum(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values[k]) for k in ['open', 'high', 'low', 'close', 'volume']}
    invalid = 0
    for i in range(len(stamps)):
        o, h, l, c, v = (values[k][i] for k in ['open', 'high', 'low', 'close', 'volume'])
        if any(x is None or not math.isfinite(x) for x in (o, h, l, c, v)):
            continue
        invalid += int(min(o, h, l, c) <= 0 or v < 0 or h < max(o, c) or l > min(o, c) or h < l)
    summary = {'url': url, 'rows': len(stamps), 'first_date': min(dates), 'last_date': max(dates),
               'duplicate_dates': len(dates) - len(set(dates)), 'sorted_dates': dates == sorted(dates),
               'missing_or_nonfinite': missing, 'invalid_ohlcv': invalid, 'raw_sha256': metadata['sha256'],
               'note': 'Checks validate returned rows, not exchange-calendar completeness or corporate-action adjustments.'}
    return summary, set(dates)


def run(args):
    fetcher = Fetcher(args.offline)
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'period': {'start': args.start, 'end': args.end},
              'archive_pages': [], 'limitations': [
                  'Archive tag coverage is not all Microsoft news or all market news.',
                  'Current pages are not point-in-time text snapshots. Modified dates do not prove historical content availability.',
                  'No model trained; forecasting benefit of texts is not established.',
                  'A second read checks discovery/deduplication only, not a production ingestion pipeline.'
              ]}
    links = set()
    boundary = False
    for number in range(1, 71):
        url = ARCHIVE + (f'page/{number}/' if number > 1 else '')
        try:
            html, _ = fetcher.get(url)
            page = Page()
            page.feed(html)
            dates = sorted(url_date(u) for u in page.links)
            if not dates:
                raise ValueError('No dated article links found')
            result['archive_pages'].append({'page': number, 'count': len(dates), 'first': dates[0], 'last': dates[-1]})
            links.update(u for u in page.links if args.start <= url_date(u) <= args.end)
            if number % 10 == 0:
                print(f'Archive page {number}; unique links: {len(links)}', flush=True)
            if dates[-1] < args.start:
                boundary = True
                break
        except Exception as exc:
            result['archive_pages'].append({'page': number, 'error': str(exc)})
            print(f'Archive page {number}: {exc}', flush=True)
            break
    result['archive_boundary_reached'] = boundary
    print(f'Discovered {len(links)} articles; checking full text', flush=True)
    articles = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, article in enumerate(pool.map(lambda url: inspect_article(url, fetcher), sorted(links)), 1):
            articles.append(article)
            if index % 20 == 0:
                print(f'Texts {index}/{len(links)}; errors: {sum(a["status"] != "ok" for a in articles)}', flush=True)
    result['articles_discovered'] = len(links)
    good = [a for a in articles if a['status'] == 'ok']
    result['article_checks'] = {'downloaded': len(good), 'errors': [a for a in articles if a['status'] != 'ok'],
        'missing_publication_timestamp': sum(not a['published_at'] for a in good),
        'missing_modified_timestamp': sum(not a['modified_at'] for a in good),
        'modified_more_than_one_day_later': sum(a['late_modified'] for a in good),
        'publication_date_differs_from_url_utc': sum(not a['date_matches_url_utc'] for a in good),
        'short_texts_under_80_words': sum(a['short_text'] for a in good),
        'earnings_link_notices': sum(a['earnings_link_notice'] for a in good),
        'duplicate_text_hashes': len(good) - len(set(a['text_sha256'] for a in good)),
        'modified_date_counts': dict(Counter(a['modified_at'][:10] for a in good).most_common(10)),
        'minimum_text_words': min((a['text_words'] for a in good), default=0)}
    result['by_year'] = {}
    for year in sorted(set(a['url_date'][:4] for a in articles)):
        year_rows = [a for a in articles if a['url_date'].startswith(year)]
        result['by_year'][year] = {'discovered': len(year_rows), 'downloaded': sum(a['status'] == 'ok' for a in year_rows),
                                 'late_modified': sum(a.get('late_modified', False) for a in year_rows)}
    try:
        result['prices'], trading_dates = audit_prices(fetcher, args.start, args.end)
        publication_days = set(a['url_date'] for a in good)
        result['calendar_date_overlap'] = {'trading_days_with_publication': len(publication_days & trading_dates),
            'total_trading_days': len(trading_dates), 'percent': round(100 * len(publication_days & trading_dates) / len(trading_dates), 2),
            'note': 'URL date intersection only; not a leak-safe assignment to forecast cutoffs.'}
    except Exception as exc:
        result['prices'] = {'error': str(exc)}
    try:
        html, _ = fetcher.get(ARCHIVE, refresh=True)
        page = Page()
        page.feed(html)
        current_links = {u for u in page.links if args.start <= url_date(u) <= args.end}
        result['repeat_discovery'] = {'first_page_links': len(current_links), 'new_urls_vs_collected': len(current_links - links),
                                      'mode': 'cached replay' if args.offline else 'fresh network request'}
    except Exception as exc:
        result['repeat_discovery'] = {'error': str(exc)}
    result['requests'] = {'network': fetcher.network_requests, 'cache_hits': fetcher.cache_hits}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted(set().union(*(row.keys() for row in articles))) if articles else ['url', 'status']
    with MANIFEST.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(articles)
    result['manifest_sha256'] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'archive_pages'}, ensure_ascii=False, indent=2), flush=True)
    return 0 if boundary and not result['article_checks']['errors'] and 'error' not in result['prices'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', required=True)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    if date.fromisoformat(args.start) > date.fromisoformat(args.end):
        parser.error('start must not be after end')
    raise SystemExit(run(args))
