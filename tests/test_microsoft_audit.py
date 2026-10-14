"""Regression checks for source parsing and timestamp safety; no network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_microsoft import Page, parse_date


class MicrosoftParsingTests(unittest.TestCase):
    def test_legacy_body_after_inner_article_is_not_lost(self):
        page = Page()
        page.feed('<nav>Navigation</nav><main><article>Title</article>'
                  '<section><p>Actual historical body &amp; facts.</p></section></main>'
                  '<footer>Footer</footer>')
        self.assertEqual(page.text, 'Title Actual historical body & facts.')

    def test_script_and_navigation_links_are_excluded(self):
        article = 'https://news.microsoft.com/source/2024/05/08/example/'
        unrelated = 'https://news.microsoft.com/source/2020/01/01/other/'
        page = Page()
        page.feed(f'<nav><a href="{unrelated}">menu</a></nav><main>'
                  f'<a href="{article}">release</a><script>fake content</script></main>')
        self.assertEqual(page.links, {article})
        self.assertEqual(page.text, 'release')

    def test_published_and_modified_are_separate(self):
        page = Page()
        page.feed('<meta content="2018-07-24T07:00:00+00:00" property="article:published_time">'
                  '<meta property="article:modified_time" content="2025-05-07T16:38:22+00:00">')
        self.assertGreater((parse_date(page.meta['article:modified_time']) -
                            parse_date(page.meta['article:published_time'])).days, 2000)

    def test_missing_timezone_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_date('2024-05-08T12:00:00')
        self.assertIsNone(parse_date(''))
        self.assertEqual(parse_date('2024-05-08T12:00:00Z').utcoffset().total_seconds(), 0)


if __name__ == '__main__':
    unittest.main()
