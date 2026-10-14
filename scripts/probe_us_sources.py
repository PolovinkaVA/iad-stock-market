"""Read-only feasibility probe; stores evidence separately from the SBER pipeline."""
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp' / 'us_sources' / 'cache'
REPORT = ROOT / 'reports' / 'us_source_probe.json'


def fetch(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / (hashlib.sha256(url.encode()).hexdigest() + '.html')
    if target.exists():
        return target.read_text(encoding='utf-8')
    time.sleep(0.6)
    request = urllib.request.Request(url, headers={'User-Agent': 'SemesterDataResearch/0.1 (educational source feasibility check)'})
    with urllib.request.urlopen(request, timeout=25) as response:
        content = response.read().decode('utf-8')
    target.write_text(content, encoding='utf-8')
    return content


def article_links(html):
    return sorted(set(re.findall(r'https://news\.microsoft\.com/source/\d{4}/\d{2}/\d{2}/[^\s"<>]+/', html)))


def inspect_article(url):
    html = fetch(url)
    dates = re.findall(r'(?:datePublished|dateModified)["\s:]+([^"<]+)', html)
    meta_dates = re.findall(r'<meta[^>]+(?:published_time|modified_time)[^>]+>', html)
    title = re.search(r'<title>(.*?)</title>', html, re.S)
    # Inspect the article region only; exclude navigation and script content.
    # Older templates contain an inner </article> before the actual body.
    # The main region bounds the complete post in both inspected templates.
    region = re.search(r'<main\b[^>]*>(.*?)</main>', html, re.S)
    text = region.group(1) if region else ''
    text = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', text, flags=re.S)
    text = ' '.join(unescape(re.sub(r'<[^>]+>', ' ', text)).split())
    return {'url': url, 'title': unescape(title.group(1)) if title else None,
            'date_fields': dates, 'date_meta': meta_dates, 'article_chars': len(text),
            'article_excerpt': text[:350], 'html_bytes': len(html.encode())}


def main():
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'archives': [], 'articles': []}
    for page in [1, 8, 20, 40, 60]:
        url = 'https://news.microsoft.com/source/tag/press-releases/' + (f'page/{page}/' if page > 1 else '')
        try:
            links = article_links(fetch(url))
            record = {'page': page, 'url': url, 'count': len(links), 'first': links[0] if links else None, 'last': links[-1] if links else None}
            result['archives'].append(record)
            print(json.dumps(record), flush=True)
            if links:
                article = inspect_article(links[0])
                result['articles'].append(article)
                print(json.dumps(article, ensure_ascii=False), flush=True)
        except Exception as exc:
            result['archives'].append({'page': page, 'url': url, 'error': str(exc)})
            print(page, str(exc), flush=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
