"""Behavioral checks: recovery, revisions, integrity and price validation."""
import hashlib
import json
import sys
import shutil
import unittest
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from stocks_forecast.msft_ingest import HTTP, Ingestion, parse_prices

URL = 'https://news.microsoft.com/source/2024/05/08/example/'
OLDER_URL = 'https://news.microsoft.com/source/2023/12/01/old/'
T1, T2 = '2026-10-14T10:00:00+00:00', '2026-10-15T10:00:00+00:00'


def article(body='Original release', modified='2024-05-08T09:00:00Z'):
    return (f'<meta property="article:published_time" content="2024-05-08T08:00:00Z">'
            f'<meta property="article:modified_time" content="{modified}">'
            f'<meta property="og:title" content="Title"><main>{body}</main>').encode()


def prices(close=10, bad=False):
    stamp = int(datetime(2024, 5, 8, 13, 30, tzinfo=timezone.utc).timestamp())
    return json.dumps({'chart': {'error': None, 'result': [{
        'meta': {'symbol': 'MSFT', 'currency': 'USD'}, 'timestamp': [stamp],
        'indicators': {'quote': [{'open': [10], 'high': [15], 'low': [5],
                                 'close': [None if bad else close], 'volume': [1000]}],
                       'adjclose': [{'adjclose': [9.5]}]},
        'events': {'dividends': {str(stamp): {'date': stamp, 'amount': 0.75}}}
    }]}}).encode()


class FakeHTTP:
    def __init__(self, responses):
        self.responses, self.urls, self.requests = responses, [], 0

    def get(self, url):
        self.urls.append(url)
        self.requests += 1
        value = self.responses(url)
        if isinstance(value, Exception):
            raise value
        return value, T2


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'tmp' / 'ingestion_tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.clean_workspace_fixture)
        self.config = json.loads((ROOT / 'config/msft.json').read_text())
        self.config.update(start='2024-01-01', max_archive_pages=2, article_refresh_limit=0)

    def clean_workspace_fixture(self):
        target = self.root.resolve()
        parent = (ROOT / 'tmp' / 'ingestion_tests').resolve()
        if target.parent != parent:
            raise ValueError('Test cleanup escaped fixture directory')
        shutil.rmtree(target)

    def job(self, **kwargs):
        job = Ingestion(self.config, '2026-10-15', root=self.root, **kwargs)
        self.addCleanup(job.db.close)
        return job

    def test_replay_is_idempotent_and_revisions_preserve_history(self):
        job = self.job()
        job.save_article(URL, article(), T1)
        job.save_article(URL, article(), T1)
        job.save_prices('https://prices', prices(), T1)
        job.save_prices('https://prices', prices(), T1)
        self.assertEqual(job.summary['article_versions_added'], 1)
        self.assertEqual(job.summary['price_versions_added'], 1)
        job.save_article(URL, article('Revised release'), T2)
        job.save_prices('https://prices', prices(close=12), T2)
        # An older imported snapshot must not replace a newer observation.
        job.save_article(URL, article(), T1)
        job.save_prices('https://prices', prices(), T1)
        self.assertEqual(job.db.execute('SELECT text FROM articles').fetchone()[0], 'Revised release')
        self.assertEqual(job.db.execute('SELECT close FROM prices').fetchone()[0], 12)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM article_versions').fetchone()[0], 2)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM price_versions').fetchone()[0], 2)
        self.assertEqual(job.finish()['status'], 'success')

    def test_failed_price_batch_keeps_previous_prices_and_raw_response(self):
        job = self.job()
        job.save_prices('https://prices', prices(), T1)
        with self.assertRaises(ValueError):
            job.save_prices('https://prices', prices(bad=True), T2)
        self.assertEqual(job.db.execute('SELECT close FROM prices').fetchone()[0], 10)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM raw_responses').fetchone()[0], 2)

    def test_partial_failure_and_retry_of_pending_article(self):
        archive = self.config['archive_url']
        first = f'<main><a href="{URL}">release</a></main>'.encode()
        last = f'<main><a href="{OLDER_URL}">older</a></main>'.encode()
        def response(url):
            if 'finance.yahoo.com' in url:
                return TimeoutError('simulated price timeout')
            return {archive: first, archive + 'page/2/': last, URL: TimeoutError('simulated text timeout')}[url]
        job = self.job(fetcher=FakeHTTP(response))
        job.update_prices()
        job.update_articles()
        result = job.finish()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['pending_articles'], 1)
        job.db.close()
        # Recovery remains possible even when that URL no longer occurs in a recent page.
        job2 = self.job(fetcher=FakeHTTP(lambda url: article() if url == URL else last))
        job2.update_articles()
        result2 = job2.finish()
        self.assertEqual(result2['status'], 'success')
        self.assertEqual(result2['pending_articles'], 0)
        self.assertEqual(result2['articles_total'], 1)
        self.assertEqual(job2.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 2)

    def test_incremental_price_request_revisits_overlap(self):
        http = FakeHTTP(lambda url: prices())
        job = self.job(fetcher=http)
        job.save_prices('https://prices', prices(), T1)
        job.update_prices()
        begin = int(datetime(2024, 5, 1, tzinfo=timezone.utc).timestamp())
        self.assertIn(f'period1={begin}', http.urls[0])
        self.assertEqual(job.summary['price_versions_added'], 1)

    def test_price_instrument_and_invalid_data_rejected(self):
        for raw in (prices(bad=True), prices(close=20), prices().replace(b'MSFT', b'AAPL')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_prices(raw, 'MSFT', '2024-01-01', '2024-12-31', 'America/New_York')

    def test_audit_import_restores_windows_bytes_and_rejects_tampering(self):
        cache = self.root / self.config['audit_cache']
        cache.mkdir(parents=True)
        raw = article('Original\r\nrelease\nbody')
        (cache / 'test.txt').write_bytes(raw.replace(b'\n', b'\r\n'))
        (cache / 'test.json').write_text(json.dumps({'url': URL, 'sha256': hashlib.sha256(raw).hexdigest(), 'fetched_at_utc': T1}))
        job = self.job()
        job.import_audit()
        self.assertEqual(job.summary['articles_seen'], 1)
        self.assertEqual(job.summary['errors'], 0)
        (cache / 'test.txt').write_bytes(b'tampered')
        job.import_audit()
        self.assertEqual(job.summary['errors'], 1)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM articles').fetchone()[0], 1)

    def test_modified_timestamp_is_part_of_a_version(self):
        job = self.job()
        job.save_article(URL, article(), T1)
        job.save_article(URL, article(modified='2026-10-14T11:00:00Z'), T2)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM article_versions').fetchone()[0], 2)

    def test_missing_timezone_cannot_enter_article_table(self):
        job = self.job()
        with self.assertRaises(ValueError):
            job.save_article(URL, article().replace(b'2024-05-08T08:00:00Z', b'2024-05-08T08:00:00'), T1)
        self.assertEqual(job.db.execute('SELECT COUNT(*) FROM articles').fetchone()[0], 0)


class HTTPTests(unittest.TestCase):
    @patch('stocks_forecast.msft_ingest.time.sleep')
    @patch('stocks_forecast.msft_ingest.urllib.request.urlopen')
    def test_retry_temporary_error_but_not_forbidden(self, urlopen, sleep):
        cfg = {'attempts': 3, 'request_interval_seconds': 0, 'timeout_seconds': 1}
        for code, attempts in ((503, 3), (403, 1)):
            with self.subTest(code=code):
                urlopen.reset_mock()
                error = urllib.error.HTTPError('https://example.test', code, 'error', {}, None)
                self.addCleanup(error.close)
                urlopen.side_effect = error
                http = HTTP(cfg)
                with self.assertRaises(urllib.error.HTTPError):
                    http.get('https://example.test')
                self.assertEqual(urlopen.call_count, attempts)


if __name__ == '__main__':
    unittest.main()
