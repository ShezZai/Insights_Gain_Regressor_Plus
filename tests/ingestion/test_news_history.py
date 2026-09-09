import pytest

from ticker_news.ingestion import massive_rest as mr
from ticker_news.ingestion import news_history as nh
from ticker_news.ingestion.massive_rest import MassiveAPIError


def _articles(ticker):
    return [{
        "article_url": f"https://x.com/{ticker.lower()}",
        "published_utc": "2025-01-02T03:04:05Z",
        "publisher": {"name": "Pub"},
        "insights": [
            {"ticker": ticker, "sentiment": "positive", "sentiment_reasoning": "why"},
            {"ticker": "OTHER", "sentiment": "neutral", "sentiment_reasoning": ""},
        ],
    }]


def test_fetch_news_rows_extracts_per_ticker_sentiment(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert rows == [{
        "ticker": "NVDA",
        "article_url": "https://x.com/nvda",
        "published_utc": "2025-01-02T03:04:05Z",
        "sentiment": "positive",
        "sentiment_reasoning": "why",
        "publisher_name": "Pub",
    }]


def test_fetch_news_rows_dedupes_ticker_url_pairs(monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles("NVDA") * 2)
    rows = nh.fetch_news_rows(["NVDA"], "2025-01-01", "2025-01-31")
    assert len(rows) == 1


def test_write_news_csv_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(nh, "fetch_range", lambda t, s, e, key=None: _articles(t))
    out = tmp_path / "news.csv"
    nh.fetch_news_csv(["NVDA"], "2025-01-01", "2025-01-31", output_path=str(out))
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "ticker,article_url,published_utc,sentiment,sentiment_reasoning,publisher_name"
    )
    assert "NVDA,https://x.com/nvda" in text


def test_fetch_range_delegates_to_shared_pagination(monkeypatch):
    calls = []

    def fake_fetch(ticker, since_iso, *, until_iso=None, key=None):
        calls.append((ticker, since_iso, until_iso, key))
        return [{"article_url": "https://x.com/1"}]

    monkeypatch.setattr(nh, "fetch_articles_rest", fake_fetch)
    articles = nh.fetch_range("NVDA", "2025-01-01", "2025-01-31", key="k")
    assert articles == [{"article_url": "https://x.com/1"}]
    assert calls == [("NVDA", "2025-01-01", "2025-01-31", "k")]


def test_fetch_range_requires_api_key(monkeypatch):
    monkeypatch.setattr(mr, "get_settings", lambda: type("S", (), {"massive_api_key": None})())
    with pytest.raises(MassiveAPIError, match="MASSIVE_API_KEY"):
        nh.fetch_range("NVDA", "2025-01-01", "2025-01-31")
