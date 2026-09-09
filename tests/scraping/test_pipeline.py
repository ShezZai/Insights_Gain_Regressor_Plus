import asyncio
import pytest

from ticker_news.scraping.config import Settings
from ticker_news.scraping.models import ArticleJob, RawPage, Article
from ticker_news.scraping.pipeline import DomainLimiter, process_job


def test_domain_limiter_serializes_per_domain():
    async def go():
        limiter = DomainLimiter(per_domain=1, delay=0.0)
        order = []

        async def worker(n):
            async with limiter.slot("fool.com"):
                order.append(("start", n))
                await asyncio.sleep(0.01)
                order.append(("end", n))

        await asyncio.gather(worker(1), worker(2))
        # With per_domain=1 the two must not interleave.
        return order

    order = asyncio.run(go())
    assert order in (
        [("start", 1), ("end", 1), ("start", 2), ("end", 2)],
        [("start", 2), ("end", 2), ("start", 1), ("end", 1)],
    )


class FakeFetcher:
    def __init__(self, http=None, browser=None):
        self._http, self._browser = http, browser
        self.browser_calls = 0

    async def http_get(self, url):
        return self._http

    async def browser_get(self, url):
        self.browser_calls += 1
        return self._browser


class FakeStore:
    def __init__(self):
        self.saved = []

    def exists_ok(self, url):
        return False

    def save(self, **kw):
        self.saved.append(kw)


class FakeRobots:
    def __init__(self, allow):
        self._allow = allow

    def allowed(self, url):
        return self._allow


def _good_raw(html="x" * 2000):
    return RawPage(url="u", final_url="https://fool.com/a", status=200, html=html, method="http")


def test_process_job_ok_via_http(monkeypatch):
    monkeypatch.setattr(
        "ticker_news.scraping.pipeline.extract",
        lambda raw, min_words: Article("T", "lots of words " * 60, None, None, "en"),
    )
    job = ArticleJob(url="https://fool.com/a", tickers=["NVDA"], published_utc=None, publisher="Fool")
    fetcher, store = FakeFetcher(http=_good_raw()), FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    res = asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=False), limiter, robots=None))
    assert res == "ok"
    assert fetcher.browser_calls == 0          # http was good, no escalation
    assert store.saved[0]["status"] == "ok"
    assert store.saved[0]["fetch_method"] == "http"


def test_process_job_escalates_to_browser_when_http_bad(monkeypatch):
    monkeypatch.setattr(
        "ticker_news.scraping.pipeline.extract",
        lambda raw, min_words: Article("T", "lots of words " * 60, None, None, "en"),
    )
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    bad = RawPage(url="u", final_url="https://fool.com/a", status=403, html="", method="http")
    fetcher = FakeFetcher(http=bad, browser=_good_raw())
    store = FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    res = asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=False), limiter, robots=None))
    assert res == "ok"
    assert fetcher.browser_calls == 1          # escalated
    assert store.saved[0]["fetch_method"] == "playwright"


def test_process_job_records_error_when_all_fail(monkeypatch):
    monkeypatch.setattr("ticker_news.scraping.pipeline.extract", lambda raw, min_words: None)
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    fetcher = FakeFetcher(http=None, browser=None)
    store = FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    res = asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=False), limiter, robots=None))
    assert res == "error"
    assert store.saved[0]["status"] == "error"
    assert store.saved[0]["error"] == "fetch_failed"


def test_process_job_blocked_by_robots():
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    fetcher, store = FakeFetcher(http=_good_raw()), FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    res = asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=True),
                                  limiter, robots=FakeRobots(allow=False)))
    assert res == "error"
    assert store.saved[0]["status"] == "error"
    assert store.saved[0]["error"] == "blocked_by_robots"
    assert fetcher.browser_calls == 0          # never fetched a blocked URL


def test_process_job_empty_when_extract_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "ticker_news.scraping.pipeline.extract",
        lambda raw, min_words: Article("T", "", None, None, None),
    )
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    fetcher = FakeFetcher(http=_good_raw(), browser=None)   # browser also yields nothing
    store = FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    res = asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=False), limiter, robots=None))
    assert res == "empty"
    assert store.saved[0]["status"] == "empty"
    assert fetcher.browser_calls == 1          # empty body is weak -> escalated


def test_process_job_keeps_http_when_browser_result_is_bad(monkeypatch):
    # http body is weak (2 words) so we escalate, but the browser returns a 403
    # challenge -> we must keep the original HTTP response, not the bad page.
    monkeypatch.setattr(
        "ticker_news.scraping.pipeline.extract",
        lambda raw, min_words: Article("T", "short body", None, None, "en"),
    )
    job = ArticleJob(url="https://fool.com/a", tickers=[], published_utc=None, publisher="Fool")
    good_http = RawPage(url="u", final_url="https://fool.com/a", status=200, html="y" * 2000, method="http")
    bad_browser = RawPage(url="u", final_url="https://fool.com/a", status=403, html="", method="playwright")
    fetcher = FakeFetcher(http=good_http, browser=bad_browser)
    store = FakeStore()
    limiter = DomainLimiter(per_domain=2, delay=0.0)

    asyncio.run(process_job(job, fetcher, store, Settings(respect_robots=False), limiter, robots=None))
    assert fetcher.browser_calls == 1                       # it tried the browser
    assert store.saved[0]["fetch_method"] == "http"         # but kept the http result
    assert store.saved[0]["http_status"] == 200
