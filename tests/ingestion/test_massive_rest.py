from datetime import datetime, timedelta, timezone

import pytest
import requests

from ticker_news.ingestion import massive_rest as mr
from ticker_news.ingestion.massive_rest import MassiveAPIError, MassiveRestSource


def _resp(status, payload=None):
    class R:
        status_code = status

        def raise_for_status(self):
            if status >= 400:
                raise requests.HTTPError(f"{status}", response=self)

        def json(self):
            return payload or {}

    return R()


def _session(responses, calls):
    class FakeSession:
        def get(self, url, params=None, timeout=None):
            calls.append(url)
            return responses[len(calls) - 1]

    return FakeSession()


def test_request_does_not_retry_client_errors(monkeypatch):
    calls = []
    session = _session([_resp(404)], calls)
    monkeypatch.setattr(mr.time, "sleep",
                        lambda s: pytest.fail("sleep called for non-transient error"))
    with pytest.raises(MassiveAPIError):
        mr._request(session, "https://api.massive.com/news", None)
    assert len(calls) == 1


def test_request_retries_transient_then_succeeds(monkeypatch):
    calls = []
    sleeps = []
    session = _session([_resp(503), _resp(200, {"results": [1]})], calls)
    monkeypatch.setattr(mr.time, "sleep", lambda s: sleeps.append(s))
    assert mr._request(session, "https://api.massive.com/news", None) == {"results": [1]}
    assert len(calls) == 2
    assert len(sleeps) == 1


def test_fetch_articles_rest_paginates_reattaching_api_key(monkeypatch):
    calls = []

    def fake_request(session, url, params):
        calls.append((url, params))
        if url == mr.BASE_URL:
            return {"results": [{"article_url": "https://x.com/1"}],
                    "next_url": "https://api.massive.com/page2"}
        return {"results": [{"article_url": "https://x.com/2"}]}

    monkeypatch.setattr(mr, "_request", fake_request)
    articles = mr.fetch_articles_rest(
        "NVDA", "2025-01-01", until_iso="2025-01-31", key="k"
    )
    assert [a["article_url"] for a in articles] == ["https://x.com/1", "https://x.com/2"]
    first_url, first_params = calls[0]
    assert first_url == mr.BASE_URL
    assert first_params["ticker"] == "NVDA"
    assert first_params["published_utc.gte"] == "2025-01-01"
    assert first_params["published_utc.lte"] == "2025-01-31"
    assert first_params["order"] == "asc"
    assert first_params["sort"] == "published_utc"
    assert first_params["apiKey"] == "k"
    # next_url carries the cursor + filters but not the apiKey: re-attach it.
    assert calls[1] == ("https://api.massive.com/page2", {"apiKey": "k"})


def test_fetch_articles_rest_omits_lte_without_until(monkeypatch):
    calls = []

    def fake_request(session, url, params):
        calls.append((url, params))
        return {"results": []}

    monkeypatch.setattr(mr, "_request", fake_request)
    mr.fetch_articles_rest("NVDA", "2025-01-01", key="k")
    assert "published_utc.lte" not in calls[0][1]


ARTICLE_A = {
    "article_url": "https://example.com/a",
    "published_utc": "2026-06-09T10:00:00Z",
    "publisher": {"name": "Benzinga"},
    "insights": [{"ticker": "NVDA", "sentiment": "positive", "sentiment_reasoning": "beat"}],
}
ARTICLE_A_AMD = {**ARTICLE_A, "insights": [{"ticker": "AMD", "sentiment": "neutral", "sentiment_reasoning": ""}]}
ARTICLE_B = {
    "article_url": "https://example.com/b",
    "published_utc": "2026-06-09T11:00:00Z",
    "publisher": {"name": "Reuters"},
    "insights": [],
}


def _source(pages, tickers=("NVDA", "AMD")):
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        list(tickers), poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=1,
    )
    return src, calls


async def _collect(src):
    return [item async for item in src.stream()]


async def test_groups_same_url_across_tickers_merging_tickers():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": [ARTICLE_A_AMD]})
    items = await _collect(src)
    assert len(items) == 1
    assert items[0].url == "https://example.com/a"
    assert sorted(items[0].tickers) == ["AMD", "NVDA"]


async def test_sentiment_lands_in_source_meta_per_ticker():
    src, _ = _source({"NVDA": [ARTICLE_A], "AMD": []})
    items = await _collect(src)
    meta = items[0].source_meta
    assert meta["sentiments"]["NVDA"]["sentiment"] == "positive"


async def test_cursor_advances_past_seen_articles():
    src, calls = _source({"NVDA": [ARTICLE_A, ARTICLE_B], "AMD": []})
    await _collect(src)
    # after one poll, the per-ticker cursor moved to the newest published_utc
    assert src.cursor("NVDA") == datetime(2026, 6, 9, 11, 0, tzinfo=timezone.utc)


async def test_second_poll_does_not_reyield_seen_urls():
    pages = {"NVDA": [ARTICLE_A], "AMD": []}
    calls = []

    def fake_fetch(ticker, since_iso):
        calls.append((ticker, since_iso))
        return pages.get(ticker, [])

    src = MassiveRestSource(
        ["NVDA", "AMD"], poll_interval_s=0.01, lookback=timedelta(days=1),
        fetch_articles=fake_fetch, max_polls=2,
    )
    items = await _collect(src)
    assert len(items) == 1  # second poll returns the same article; not re-yielded
