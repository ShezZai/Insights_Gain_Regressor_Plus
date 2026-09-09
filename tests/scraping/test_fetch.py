from ticker_news.scraping.fetch import http_looks_bad
from ticker_news.scraping.models import RawPage


def _raw(status=200, html="x" * 2000):
    return RawPage(url="u", final_url="u", status=status, html=html, method="http")


def test_good_page_is_fine():
    assert http_looks_bad(_raw()) is False


def test_none_is_bad():
    assert http_looks_bad(None) is True


def test_blocked_statuses_are_bad():
    for code in (401, 403, 404, 429, 500, 503):
        assert http_looks_bad(_raw(status=code)) is True


def test_tiny_body_is_bad():
    assert http_looks_bad(_raw(html="too short")) is True


def test_challenge_page_is_bad():
    html = "<html><body>Just a moment... Checking your browser</body></html>" + "x" * 2000
    assert http_looks_bad(_raw(html=html)) is True
