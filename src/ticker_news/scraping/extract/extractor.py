from collections.abc import Callable
from datetime import datetime

import trafilatura

from ..models import Article, RawPage
from ..urls import domain_of

# domain -> function(html) -> body text (or None). Populated by @register.
SITE_OVERRIDES: dict[str, Callable[[str], str | None]] = {}


def register(domain: str):
    def decorator(fn: Callable[[str], str | None]):
        SITE_OVERRIDES[domain] = fn
        return fn
    return decorator


def _parse_date(value) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def generic_extract(raw: RawPage) -> Article | None:
    doc = trafilatura.bare_extraction(
        raw.html, with_metadata=True, favor_recall=True,
    )
    if doc is None:
        return None
    data = doc.as_dict() if hasattr(doc, "as_dict") else doc
    text = (data.get("text") or "").strip()
    return Article(
        title=data.get("title"),
        text=text,
        author=data.get("author"),
        published=_parse_date(data.get("date")),
        lang=data.get("language"),
    )


def extract(raw: RawPage, min_words: int) -> Article | None:
    art = generic_extract(raw)
    override = SITE_OVERRIDES.get(domain_of(raw.final_url))
    if override and (art is None or art.is_weak(min_words)):
        body = override(raw.html)
        if body and body.strip():
            if art is None:
                art = Article(title=None, text="", author=None, published=None, lang=None)
            art.text = body.strip()
    return art


# Load all override modules so their @register() decorators run.
from . import overrides as _overrides  # noqa: E402,F401
