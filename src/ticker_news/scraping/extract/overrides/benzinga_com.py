"""Benzinga.com extractor override.

Benzinga's article pages embed the full article body in a NewsArticle JSON-LD
script, while the visible DOM article-content-body container is often truncated.
We extract from the structured data first, then fall back to the largest
React server-component div (id matching 'S:N').
"""
from __future__ import annotations

import html as html_module
import json
import re

from bs4 import BeautifulSoup

from ..extractor import register


@register("benzinga.com")
def extract_benzinga(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")

    # Primary: NewsArticle JSON-LD structured data has the cleanest full body.
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text())
        except (json.JSONDecodeError, ValueError):
            continue
        if data.get("@type") == "NewsArticle" and data.get("articleBody"):
            body = html_module.unescape(data["articleBody"]).strip()
            if body:
                return body

    # Fallback: pick the React streaming server-component div (id="S:N") with
    # the most words — these divs hold the pre-hydration article content.
    best_text = ""
    best_wc = 0
    for div in soup.find_all("div", id=re.compile(r"^S:\d+$")):
        text = div.get_text(" ", strip=True)
        wc = len(text.split())
        if wc > best_wc:
            best_wc = wc
            best_text = text

    if best_wc >= 50:
        return best_text

    return None
