"""Microsoft Source HTML parsing shared by the audit and regular ingestion."""
import re
from datetime import datetime
from html.parser import HTMLParser


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


def url_date(url):
    return '-'.join(re.search(r'/source/(\d{4})/(\d{2})/(\d{2})/', url).groups())


def parse_date(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamp has no timezone')
    return parsed


def article_text(page):
    text = page.text
    for marker in ['For more information, press only:', 'Note to editors:']:
        text = text.split(marker, 1)[0]
    return text
