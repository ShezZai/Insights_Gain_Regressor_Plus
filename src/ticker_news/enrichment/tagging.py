"""Tag each article with its primary ticker/segment and any secondary ones.

The ticker -> (company_name, AI-segment) lookup is loaded from the
``public.ticker_data`` table, populated by ``load_ticker_data.py`` from the
market-universe CSV. It derives four columns on ``public.articles``:

    primary_ticker   text     -- the first ticker in `tickers`
    primary_segment  text     -- that ticker's AI segment (NULL if unknown)
    more_tickers     text[]   -- every OTHER universe ticker mentioned in the
                                 article (its symbol or company name appears in
                                 title/content), unioned with the rest of the
                                 `tickers` array; NULL if none
    more_segments    text[]   -- distinct additional segments among `more_tickers`
                                 (excludes primary_segment; NULL if none)

``more_tickers`` is found by scanning the article text for each universe
ticker's symbol (in cashtag/parenthesis/exchange context, plus bare uppercase
for unambiguous symbols) and its company name.

Re-runnable: by default only rows that don't have a primary_ticker yet are
filled, so the job is resumable and a re-run won't re-tag already-tagged rows.
Pass --not-only-missing to recompute every row (e.g. after a ticker_data change).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import psycopg

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

BATCH_SIZE = 1000


# Short / common-word symbols that would generate false positives if matched as
# bare tokens; these are only recognized in cashtag/parenthesis/exchange context.
# common English words / very short symbols where a bare token is unreliable.
_AMBIGUOUS_SYMBOLS = {"AI", "ON", "S", "TT", "ARM", "NOW", "ALL", "SE", "GO", "IT"}

# Trailing "category" words that aren't part of the distinctive brand. A leading
# brand word is registered as a short alias only when EVERY following word is one
# of these (so "Marvell Technology" -> "Marvell", but "Hewlett Packard
# Enterprise" is left alone since "Packard" isn't a category word).
_CATEGORY_WORDS = {
    "technology", "technologies", "holdings", "holding", "systems", "system",
    "networks", "network", "labs", "instruments", "semiconductor", "semiconductors",
    "group", "design", "realty", "trust", "platforms", "industries", "scientific",
    "motion", "test", "materials", "research", "innovation", "enterprise",
    "electric", "services", "vernova", "photonics", "microelectronics",
    "manufacturing", "devices", "integrations", "storage", "solutions",
    "communications",
}
# Leading words too generic to be a safe standalone alias.
_GENERIC_FIRST = {
    "advanced", "applied", "analog", "power", "core", "global", "united",
    "international", "texas", "western", "digital", "pure", "super", "monolithic",
    "onto", "carrier", "general", "american", "national", "micro", "open", "first",
    "silicon",   # "Silicon Motion" — would hit "silicon wafer/chips" everywhere
    "taiwan",    # "Taiwan Semiconductor" — the country appears constantly
    "powell",    # "Powell Industries" — collides with Fed Chair Jerome Powell
}


def _short_alias(name: str) -> Optional[str]:
    """Return a safe leading-brand alias for a full company name, or None."""
    tokens = name.split()
    if len(tokens) < 2:
        return None
    first = tokens[0]
    rest = tokens[1:]
    if not all(t.strip(".,").lower() in _CATEGORY_WORDS for t in rest):
        return None
    if len(first) < 4 or first.lower() in _GENERIC_FIRST:
        return None
    return first


def load_ticker_data(conn: psycopg.Connection) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Load ticker -> (company_name, segment) from public.ticker_data."""
    with conn.cursor() as cur:
        cur.execute("SELECT ticker, company_name, segment FROM public.ticker_data")
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(
            "public.ticker_data is empty or missing. Populate it first:\n"
            "    ticker-news load-universe"
        )
    return {t.strip().upper(): (name, seg) for t, name, seg in rows if t and t.strip()}


_SYM_GROUPS = ("cash", "exch", "parnum", "paren", "bare")


def _build_patterns(data: Dict[str, Tuple[Optional[str], Optional[str]]]):
    """Compile the company-name and ticker-symbol regexes shared by the matcher
    and the annotator. Returns (name_to_tickers, name_re, sym_re)."""
    # company name / alias (lowercased) -> set of tickers (a name like
    # "Alphabet" can map to several share classes).
    name_to_tickers: Dict[str, Set[str]] = defaultdict(set)
    for tk, (name, _seg) in data.items():
        if name and name.strip():
            full = name.strip()
            name_to_tickers[full.lower()].add(tk)
            alias = _short_alias(full)
            if alias:
                name_to_tickers[alias.lower()].add(tk)

    names_sorted = sorted(name_to_tickers, key=len, reverse=True)  # prefer longest
    name_re = (
        re.compile(r"(?<![\w.])(" + "|".join(re.escape(n) for n in names_sorted) + r")(?![\w.])",
                   re.IGNORECASE)
        if names_sorted else None
    )

    symbols = sorted(data, key=len, reverse=True)
    sym_alt = "|".join(re.escape(s) for s in symbols)
    safe_alt = "|".join(
        re.escape(s) for s in symbols if len(s) >= 3 and s not in _AMBIGUOUS_SYMBOLS
    )
    # Strong, unambiguous signals matched for EVERY symbol (so a bare "(AI)"
    # meaning "artificial intelligence" is NOT mistaken for the C3.ai ticker):
    #   $AVGO            cashtag
    #   NASDAQ: AVGO     exchange prefix
    #   (NVDA +2.24%)    parenthesised symbol followed by a price/percent
    # Plain "(GOOG)" and bare "NVDA" are only allowed for unambiguous symbols.
    alts = [
        rf"\$(?P<cash>{sym_alt})\b",
        rf"(?i:NYSE|NASDAQ|NYSEARCA|AMEX|OTC|OTCMKTS)\s*:?\s*(?P<exch>{sym_alt})\b",
        rf"\(\s*(?P<parnum>{sym_alt})\s+[+\-]?\d",
    ]
    if safe_alt:
        alts.append(rf"\(\s*(?P<paren>{safe_alt})\s*\)")
        alts.append(rf"\b(?P<bare>{safe_alt})\b")
    sym_re = re.compile("|".join(alts))  # symbols are case-sensitive (uppercase)
    return name_to_tickers, name_re, sym_re


def build_matcher(
    data: Dict[str, Tuple[Optional[str], Optional[str]]]
) -> Callable[[str], List[str]]:
    """Return a function text -> ordered list of universe tickers mentioned.

    Detects, in one pass each: (a) company names (whole-word, case-insensitive)
    and (b) ticker symbols — always in $/parenthesis/exchange context, and bare
    uppercase tokens for unambiguous symbols.
    """
    name_to_tickers, name_re, sym_re = _build_patterns(data)

    def find(text: str) -> List[str]:
        if not text:
            return []
        hits: List[Tuple[int, str]] = []  # (position, ticker) for ordering
        if name_re:
            for m in name_re.finditer(text):
                for tk in name_to_tickers[m.group(1).lower()]:
                    hits.append((m.start(), tk))
        for m in sym_re.finditer(text):
            d = m.groupdict()
            tk = next((d[k] for k in _SYM_GROUPS if d.get(k)), None)
            if tk:
                hits.append((m.start(), tk))
        # de-dupe, preserving first-appearance order
        seen: Set[str] = set()
        ordered: List[str] = []
        for _pos, tk in sorted(hits, key=lambda h: h[0]):
            if tk not in seen:
                seen.add(tk)
                ordered.append(tk)
        return ordered

    return find


def build_annotator(
    data: Dict[str, Tuple[Optional[str], Optional[str]]]
) -> Callable[[str], str]:
    """Return a function text -> text that cross-annotates ticker/company mentions.

    Each ticker symbol gets its company name in parentheses and each company name
    gets its ticker(s) — e.g. ``NVDA`` -> ``NVDA (NVIDIA)`` and ``NVIDIA`` ->
    ``NVIDIA (NVDA)``. Only the first mention of each ticker entity is annotated,
    and a mention already followed by its counterpart in parentheses is left
    alone (so re-running, or text that already pairs them, is idempotent).
    """
    name_to_tickers, name_re, sym_re = _build_patterns(data)
    sym_to_company = {tk: (name or "").strip() for tk, (name, _s) in data.items()}

    def _followed_by_counterpart(text: str, end: int, counterparts: Sequence[str]) -> bool:
        m = re.match(r"\s*\(([^)]*)\)", text[end:])
        if not m:
            return False
        inside = m.group(1).lower()
        return any(cp and cp.lower() in inside for cp in counterparts)

    def annotate(text: str) -> str:
        if not text:
            return text
        # (start, end, counterpart_tokens, entity_key, annotation_or_None)
        events: List[Tuple[int, int, List[str], object, Optional[str]]] = []
        for m in sym_re.finditer(text):
            d = m.groupdict()
            grp = next((g for g in _SYM_GROUPS if d.get(g)), None)
            if not grp:
                continue
            tk = d[grp]
            s, e = m.span(grp)
            comp = sym_to_company.get(tk, "")
            events.append((s, e, [comp], tk, f" ({comp})" if comp else None))
        if name_re:
            for m in name_re.finditer(text):
                tickers = sorted(name_to_tickers.get(m.group(1).lower(), []))
                if not tickers:
                    continue
                s, e = m.span(1)
                key = tickers[0] if len(tickers) == 1 else tuple(tickers)
                events.append((s, e, tickers, key, f" ({', '.join(tickers)})"))
        if not events:
            return text

        # longest match wins on ties; drop overlaps left-to-right
        events.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        out: List[str] = []
        pos = 0
        done: Set[object] = set()
        for s, e, counterparts, key, ann in events:
            if s < pos:  # overlaps a region already emitted
                continue
            out.append(text[pos:e])
            pos = e
            if key in done:
                continue
            done.add(key)  # annotate (or suppress) each entity only once
            if ann and not _followed_by_counterpart(text, e, counterparts):
                out.append(ann)
        out.append(text[pos:])
        return "".join(out)

    return annotate


def ensure_schema(conn: psycopg.Connection) -> None:
    """Add the segment columns if they don't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE public.articles "
            "ADD COLUMN IF NOT EXISTS primary_ticker  text, "
            "ADD COLUMN IF NOT EXISTS primary_segment text, "
            "ADD COLUMN IF NOT EXISTS more_tickers    text[], "
            "ADD COLUMN IF NOT EXISTS more_segments   text[]"
        )
    conn.commit()


def compute_row(
    tickers: Optional[Sequence[str]],
    text: str,
    data: Dict[str, Tuple[Optional[str], Optional[str]]],
    find: Callable[[str], List[str]],
) -> Tuple[Optional[str], Optional[str], Optional[List[str]], Optional[List[str]]]:
    """Derive (primary_ticker, primary_segment, more_tickers, more_segments).

    ``primary_ticker`` is the first entry of the article's ``tickers`` array.
    ``more_tickers`` is every other universe ticker either named in ``text`` or
    listed in the rest of the array, in order: the array remainder first, then
    text-only finds by first appearance.
    """
    array = [t.strip().upper() for t in (tickers or []) if t and t.strip()]
    if not array:
        return None, None, None, None

    primary_ticker = array[0]
    primary_segment = data.get(primary_ticker, (None, None))[1]

    in_text = find(text)
    more: List[str] = []
    for tk in list(array[1:]) + in_text:  # array remainder first, then text-only
        if tk != primary_ticker and tk not in more:
            more.append(tk)

    # distinct segments contributed by the other tickers, beyond the primary one
    more_segments: List[str] = []
    seen = {primary_segment} if primary_segment else set()
    for tk in more:
        seg = data.get(tk, (None, None))[1]
        if seg and seg not in seen:
            seen.add(seg)
            more_segments.append(seg)

    return (
        primary_ticker,
        primary_segment,
        more or None,
        more_segments or None,
    )


def create_indexes(conn: psycopg.Connection) -> None:
    """Build helpful indexes for filtering by ticker/segment (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_primary_ticker_idx "
            "ON public.articles (primary_ticker);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_primary_segment_idx "
            "ON public.articles (primary_segment);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS articles_more_segments_idx "
            "ON public.articles USING gin (more_segments);"
        )
    conn.commit()


def tag_all(
    only_missing: bool = True,
    build_index: bool = True,
) -> int:
    """Populate the segment columns for every (or only untagged) article."""
    from ticker_news.shared.db import connect

    conn = connect()
    try:
        data = load_ticker_data(conn)
        find = build_matcher(data)
        print(f"Loaded {len(data)} tickers from public.ticker_data")

        ensure_schema(conn)

        where = "WHERE primary_ticker IS NULL" if only_missing else ""
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, tickers, title, content FROM public.articles {where} "
                f"ORDER BY id"
            )
            rows = cur.fetchall()

        if not rows:
            print("Nothing to tag.")
            return 0

        print(f"Tagging {len(rows)} article(s) (scanning text for tickers/names) ...")
        unmatched: set[str] = set()  # primary tickers not present in the mapping
        with_primary_seg = 0
        with_more = 0

        updates: List[tuple] = []
        row_iter = tqdm(rows, unit="article") if tqdm else rows
        for rid, tickers, title, content in row_iter:
            text = f"{title or ''}\n{content or ''}"
            pt, ps, mt, ms = compute_row(tickers, text, data, find)
            if pt and ps is None:
                unmatched.add(pt)
            if ps:
                with_primary_seg += 1
            if mt:
                with_more += 1
            updates.append((pt, ps, mt, ms, rid))

        batches = range(0, len(updates), BATCH_SIZE)
        iterator = tqdm(batches, unit="batch") if tqdm else batches
        for start in iterator:
            chunk = updates[start : start + BATCH_SIZE]
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE public.articles SET primary_ticker = %s, "
                    "primary_segment = %s, more_tickers = %s, more_segments = %s "
                    "WHERE id = %s",
                    chunk,
                )
            conn.commit()

        print(
            f"Done. {with_primary_seg} row(s) got a primary_segment, "
            f"{with_more} row(s) got more_segments."
        )
        if unmatched:
            sample = ", ".join(sorted(unmatched)[:15])
            print(
                f"{len(unmatched)} primary ticker(s) had no segment in the mapping "
                f"(e.g. {sample})"
            )

        if build_index:
            print("Building segment/ticker indexes ...")
            create_indexes(conn)
        return len(updates)
    finally:
        conn.close()
