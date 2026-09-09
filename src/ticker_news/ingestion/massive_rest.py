"""Massive.com REST news poller — the 'live today' NewsFeedSource.

Polls the news endpoint per universe ticker on an interval, keeping a
per-ticker published_utc cursor and deduping URLs across tickers (one
FeedItem per article with all its tickers merged). The HTTP plumbing
(_request retry/backoff, cursor pagination) is ported from the legacy
ticker_news.py script (pre-package, kept in git history).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Callable, Iterable, Optional

import requests

from ticker_news.ingestion.feed import FeedItem
from ticker_news.shared.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.massive.com/v2/reference/news"
PAGE_LIMIT = 1000
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
REQUEST_TIMEOUT = 30
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class MassiveAPIError(RuntimeError):
    """Raised when the Massive API returns a non-recoverable error."""


def _request(session: requests.Session, url: str, params: Optional[dict]) -> dict:
    """GET with retries on transient errors (429 / 5xx / network)."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in TRANSIENT_STATUSES:
                raise requests.HTTPError(f"transient {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            resp_status = getattr(getattr(exc, "response", None), "status_code", None)
            if resp_status is not None and resp_status not in TRANSIENT_STATUSES:
                raise MassiveAPIError(f"Request to {url} failed: {exc}") from exc
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise MassiveAPIError(f"Request to {url} failed: {last_exc}")


def _parse_utc(value: str | None) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_articles_rest(
    ticker: str,
    since_iso: str,
    *,
    until_iso: str | None = None,
    key: str | None = None,
) -> list[dict]:
    """One blocking fetch of every article for `ticker` since `since_iso`
    (optionally bounded above by `until_iso`).

    Cursor pagination ported from the legacy _iter_articles: follow next_url,
    re-attaching the apiKey (next_url carries filters but not the key).
    """
    key = key or get_settings().massive_api_key
    if not key:
        raise MassiveAPIError("MASSIVE_API_KEY is not set (put it in .env).")
    out: list[dict] = []
    params: Optional[dict] = {
        "ticker": ticker,
        "published_utc.gte": since_iso,
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
        "apiKey": key,
    }
    if until_iso is not None:
        params["published_utc.lte"] = until_iso
    url = BASE_URL
    with requests.Session() as session:
        while url:
            payload = _request(session, url, params)
            out.extend(payload.get("results", []) or [])
            url = payload.get("next_url")
            params = {"apiKey": key} if url else None
    return out


class MassiveRestSource:
    """Poll-based NewsFeedSource over the Massive news API.

    max_polls is for tests/drain runs; None means poll forever.

    Note: _seen_urls grows unbounded over a long-running process. The DB
    enqueue is the real dedupe (ON CONFLICT DO NOTHING) — this set just
    avoids re-yield churn within the service process. Acceptable for now.
    """

    def __init__(
        self,
        tickers: Iterable[str],
        *,
        poll_interval_s: float = 60.0,
        lookback: timedelta = timedelta(hours=24),
        fetch_articles: Callable[[str, str], list[dict]] = fetch_articles_rest,
        max_polls: int | None = None,
        fetch_concurrency: int = 8,
    ):
        self.tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        self.poll_interval_s = poll_interval_s
        self.lookback = lookback
        self.fetch_articles = fetch_articles
        self.max_polls = max_polls
        self.fetch_concurrency = fetch_concurrency
        # Cursors are in-memory only: a restart re-fetches from now-lookback.
        # The DB enqueue (ON CONFLICT DO NOTHING) absorbs re-emits; articles
        # are only LOST if the process is down longer than `lookback`.
        self._cursors: dict[str, datetime] = {}
        self._seen_urls: set[str] = set()

    def cursor(self, ticker: str) -> datetime | None:
        return self._cursors.get(ticker)

    def _since(self, ticker: str, now: datetime) -> datetime:
        return self._cursors.get(ticker, now - self.lookback)

    async def stream(self) -> AsyncIterator[FeedItem]:
        polls = 0
        while self.max_polls is None or polls < self.max_polls:
            polls += 1
            now = datetime.now(timezone.utc)
            # url -> (tickers, article) accumulated across this poll
            batch: dict[str, tuple[list[str], dict]] = {}
            sem = asyncio.Semaphore(self.fetch_concurrency)

            async def _fetch_one(ticker: str, since: datetime):
                async with sem:
                    try:
                        articles = await asyncio.to_thread(
                            self.fetch_articles, ticker, since.isoformat()
                        )
                        return ticker, articles, None
                    except Exception as exc:
                        return ticker, [], exc

            results = await asyncio.gather(
                *(_fetch_one(t, self._since(t, now)) for t in self.tickers)
            )
            for ticker, articles, error in results:
                if error is not None:
                    logger.warning("massive poll failed for %s: %r", ticker, error)
                    continue
                newest = self._cursors.get(ticker)
                for article in articles:
                    published = _parse_utc(article.get("published_utc"))
                    if published and (newest is None or published > newest):
                        newest = published
                    url = (article.get("article_url") or "").strip()
                    if not url:
                        continue
                    tickers_for_url, _ = batch.setdefault(url, ([], article))
                    if ticker not in tickers_for_url:
                        tickers_for_url.append(ticker)
                if newest is not None:
                    self._cursors[ticker] = newest

            for url, (url_tickers, article) in batch.items():
                if url in self._seen_urls:
                    continue
                self._seen_urls.add(url)
                sentiments = {}
                for insight in article.get("insights") or []:
                    t = (insight.get("ticker") or "").upper()
                    if t:
                        sentiments[t] = {
                            "sentiment": insight.get("sentiment", ""),
                            "sentiment_reasoning": insight.get("sentiment_reasoning", ""),
                        }
                yield FeedItem(
                    url=url,
                    tickers=url_tickers,
                    published_utc=_parse_utc(article.get("published_utc")),
                    publisher=(article.get("publisher") or {}).get("name") or None,
                    source_meta={"sentiments": sentiments, "provider": "massive"},
                )

            if self.max_polls is None or polls < self.max_polls:
                await asyncio.sleep(self.poll_interval_s)
