"""Massive market-data plumbing shared by the research tools.

One copy of the market-time constants and bar fetching that the legacy
scripts (ticker_candles, scan_ranges, catalyst_returns) each duplicated.
"""

from __future__ import annotations

import time as _time
from datetime import time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import requests

MARKET_TZ = ZoneInfo("America/New_York")
PREMARKET_OPEN = dtime(4, 0)
REGULAR_OPEN = dtime(9, 30)
REGULAR_CLOSE = dtime(16, 0)
AFTER_HOURS_CLOSE = dtime(20, 0)

AGGS_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{frm}/{to}"
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0
REQUEST_TIMEOUT = 30
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


def _settings_key() -> str:
    from ticker_news.shared.config import get_settings

    return get_settings().massive_api_key or ""


def api_key(explicit: Optional[str] = None) -> str:
    """Explicit value wins; else MASSIVE_API_KEY from settings; else error."""
    key = explicit or _settings_key()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY is not set (put it in .env or pass --api-key).")
    return key


def get_json(url: str, params: dict) -> dict:
    """GET with retry/backoff on 429/5xx/network errors (legacy `_get` port)."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code in TRANSIENT_STATUSES:
                raise requests.HTTPError(f"transient {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            resp_status = getattr(getattr(exc, "response", None), "status_code", None)
            if resp_status is not None and resp_status not in TRANSIENT_STATUSES:
                raise RuntimeError(f"Massive request failed for {url}: {exc!r}") from exc
            last = exc
            if attempt < MAX_RETRIES - 1:
                _time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"Massive request failed for {url}: {last!r}")


def fetch_bars(
    ticker: str,
    *,
    span: str = "minute",
    multiplier: int = 1,
    frm: str,
    to: str,
    key: Optional[str] = None,
    adjusted: bool = True,
    limit: int = 50000,
) -> list[dict]:
    """All aggregate bars for `ticker` in [frm, to], following next_url pages."""
    k = api_key(key)
    url = AGGS_URL.format(ticker=ticker, multiplier=multiplier, span=span, frm=frm, to=to)
    params: dict = {"adjusted": str(adjusted).lower(), "sort": "asc",
                    "limit": limit, "apiKey": k}
    out: list[dict] = []
    while url:
        payload = get_json(url, params)
        out.extend(payload.get("results", []) or [])
        url = payload.get("next_url")
        params = {"apiKey": k}
    return out


def session_of(t: dtime) -> str:
    """premarket | regular | after_hours | closed (extended-hours convention)."""
    if PREMARKET_OPEN <= t < REGULAR_OPEN:
        return "premarket"
    if REGULAR_OPEN <= t < REGULAR_CLOSE:
        return "regular"
    if REGULAR_CLOSE <= t < AFTER_HOURS_CLOSE:
        return "after_hours"
    return "closed"
