"""The provider-agnostic live-feed port.

Any news source — REST poller, websocket consumer, CSV backfill — implements
NewsFeedSource. The service consumes the stream and enqueues pipeline jobs;
nothing downstream knows or cares where items came from. When the real-time
provider is chosen, it becomes one new file implementing this protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class FeedItem(BaseModel):
    url: str
    tickers: list[str] = Field(default_factory=list)
    published_utc: datetime | None = None
    publisher: str | None = None
    source_meta: dict[str, object] = Field(default_factory=dict)


@runtime_checkable
class NewsFeedSource(Protocol):
    def stream(self) -> AsyncIterator[FeedItem]: ...
