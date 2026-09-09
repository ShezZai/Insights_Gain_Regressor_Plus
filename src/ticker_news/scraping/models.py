from dataclasses import dataclass
from datetime import datetime


@dataclass
class ArticleJob:
    url: str
    tickers: list[str]
    published_utc: datetime | None
    publisher: str | None


@dataclass
class RawPage:
    url: str
    final_url: str
    status: int
    html: str
    method: str  # 'http' | 'playwright'


@dataclass
class Article:
    title: str | None
    text: str
    author: str | None
    published: datetime | None
    lang: str | None

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def is_weak(self, min_words: int) -> bool:
        return (not self.text.strip()) or self.word_count < min_words
