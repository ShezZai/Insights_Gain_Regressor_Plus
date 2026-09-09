#!/usr/bin/env python3
"""Two-pass insight distillation INTO the DB: public.distilled_article_insights.

DB-writing sibling of extract_insights_new.py, same two-pass logic:
  pass 1  (gemini-2.5-flash-lite) extract evidance-event / informative boxes
          -> stored in `first_label`
  pass 2  (gemini-2.5-flash-lite) strict per-box gate, keep/relabel/drop
          -> stored in `second_label` ('evidance-event' | 'informative' | 'DROP')

EVERY extracted box is stored (so you can see exactly where the gate dropped or
relabeled). ALL boxes — kept AND dropped — are embedded with OpenAI
text-embedding-3-small, so dropped boxes remain available for later analysis
(filter on `second_label` at query time). --no-embed skips embedding entirely.

The table mirrors public.article_insights and adds `first_label` / `second_label`.
The extraction + gate prompts are embedded directly in this file. Per-run token
cost (Gemini extract, Gemini gate, OpenAI embeddings) is logged.

Resumable: skips articles already stamped with
`articles.distilled_insights_extracted_at` unless --reprocess (or --ids, which
re-does those). Each article's rows are fully replaced on (re)processing.

Usage:
    python distill_insights_to_db.py --limit 5          # smoke test
    python distill_insights_to_db.py                    # all un-processed articles
    python distill_insights_to_db.py --ids 20074,9491,8721
    python distill_insights_to_db.py --workers 8
    python distill_insights_to_db.py --reprocess --ids 8721
    python distill_insights_to_db.py --no-embed         # skip OpenAI embeddings

Requires GOOGLE_API_KEY (Gemini) and OPENAI_API_KEY (embeddings, unless
--no-embed). Connection from NEWS_DB_DSN / DATABASE_URL.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DB_DSN = os.getenv("NEWS_DB_DSN") or os.getenv("DATABASE_URL") or "dbname=news"

EXTRACT_MODEL = "gemini-2.5-flash-lite"   # pass 1
GATE_MODEL = "gemini-2.5-flash-lite"      # pass 2 (per-box gate)
FALLBACK_MODEL = "gemini-2.5-flash"       # extraction fallback on 504
GEMINI_TIMEOUT_MS = 120_000
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_ARTICLE_CHARS = 48_000
DEFAULT_QUOTE_THRESHOLD = 0.75

# Approximate published USD per token (verify against your billing console).
RATE = {
    "ext_in":  0.10 / 1e6,   # gemini-2.5-flash-lite input
    "ext_out": 0.40 / 1e6,   # gemini-2.5-flash-lite output
    "gate_in": 0.10 / 1e6,   # gemini-2.5-flash-lite input
    "gate_out": 0.40 / 1e6,  # gemini-2.5-flash-lite output
    "embed":   0.02 / 1e6,   # text-embedding-3-small
}

LABELS = ["evidance-event", "informative"]

# --------------------------------------------------------------------------- #
# Prompts (embedded directly — kept in sync with distill_insights_prompt.txt)
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are an equity analyst building the EVIDENCE FILE that a downstream trading judge will later read to decide whether a NEW breaking-news article is worth ACTING ON. You are NOT summarizing the article and NOT scoring sentiment. You extract only durable, fact-bearing context worth recalling later, and you label each piece by how the judge should use it.

FIRST, a gate. If the article is fundamentally one of these, it has NO extractable evidence — return {{"boxes": []}} and stop:
- a market-research / industry-forecast / TAM report ("market to reach $X by 20YY", CAGR, drivers, "key players", regional breakdown);
- an awards / recognition / "named to list" release, or a conference / event / magazine promo, or a trade-show / "unveils new product line" / multi-product showcase;
- an analyst-opinion or stock-pitch piece (price targets, "could surge N%", "most upgraded", "why X is a buy", valuation calls);
- a single-company THESIS / explainer / "deep dive" / "is X a buy" essay whose substance is the author's argument, competitive analysis, growth story, or valuation — even if it cites real figures;
- a workforce / education / careers / "talent pipeline" / training-program piece, or other human-interest / institutional news with no tradeable event;
- a product-marketing / capability / "how it works" piece with no event behind it.
Naming real companies or citing real numbers does NOT exempt it — promotional or argumentative framing of facts is still not evidence. When such a piece DOES contain a genuine hard event (a reported result, a signed deal, a filed lawsuit), extract ONLY that event and drop the surrounding argument.

Otherwise, create a box ONLY if it is one of these TWO kinds. Label it accordingly:

1. evidance-event — a concrete, first-hand, MATERIAL fact that verifiably happened or was reported, with a real consequence for a company, sector, or the market. It must be a FACT you can quote — not an opinion, pitch, or forecast. May be bullish OR bearish.
   Qualifies: reported financial results / guidance / margins; completed M&A; signed contracts or deals with real substance; lawsuits filed, recalls, layoffs, restructurings, insolvency; regulatory rulings; material macro releases (rate decisions, jobs, inflation, tariffs, sanctions); confirmed product general-availability with business impact; disclosed insider/institutional positions.
   Test: "Did a specific thing verifiably happen, and can I point to the fact in a verbatim quote?"

2. informative — a REAL, SPECIFIC, ANNOUNCED but SOFT development: an announced partnership / collaboration / MoU / LoI / proposed or pending deal with no terms; a launched or announced product/expansion/funding round; a planned or announced-future action ("trial next year"); a self-reported milestone not independently confirmed. There must be a discrete announced HAPPENING — not a description, a trend, or an argument.
   informative is NOT a catch-all. NEVER use it for: a company's competitive position, market share, "moat", or "growth potential"; valuation, price targets, ratings, or "could/should" claims; capex/industry/demand TRENDS or "AI tailwinds"; product capabilities or "how it works"; analyst-sentiment roundups; or background/explanatory context. The ONLY analyst item that qualifies is a SINGLE named major upgrade/downgrade that is itself the article's headline event (e.g. "Goldman downgrades X to Sell") — never a list of who is "most upgraded".
   Test: "Is there a discrete, announced happening here — and is it NOT opinion, trend, capability, or description?"

Decide in that order: try evidance-event, else informative, else DROP. When genuinely unsure, DROP.

ONE-BOX RULE for a product launch / release: emit AT MOST ONE box (the launch itself, as informative). Do NOT add separate boxes for the product's features, capabilities, specs, "how it works", benefits, use-cases, or the company's description/history — those are marketing, not events. Same for a deal or partnership: one box for the deal, none for the partner's product capabilities.

DROP — create NO box — for anything else, especially:
- Self-promotion: awards, accolades, "recognized as", CEO/exec praise of their own product, platform, or results.
- Marketing benefit-narratives: "enables / accelerates / scalable / seamless / best-in-class" capability copy with no event behind it.
- Market-size / TAM projections and syndicated research stats ("market to reach $X by 20YY", CAGR of Y%) — these are NOT informative; drop them.
- Industry-trend / "the market is driven by X" / "the technology enables Y" generalizations — durable-sounding but promotional; DROP (do not treat as background reasoning).
- Analyst OPINION: price targets, ratings, "undervalued / overvalued", return forecasts, "could rally N%", upgrade/downgrade ROUNDUPS, "most upgraded". (Only a single named major rating CHANGE that is the headline event -> "informative".)
- Abstract theses, frameworks, book ideas, motivational or thought-leadership content.
- Routine immaterial notices: recurring share-buyback / repurchase disclosures, insider share-incentive / RSU grant filings, unchanged/flat dividends, daily market wraps ("futures slightly higher", "Asia mixed", "sector X led gains"), NAV / calendar notices, board re-elections.
- A company's competitive position, market share, "moat", strategy, or growth-story narrative — even with real numbers; this is argument, not an event.
- Boilerplate, safe-harbor / forward-looking disclaimers, legal solicitations, subscription / ad copy, generic "about the company".

Principles:
- New facts > opinion. Materiality > popularity. If it is a prediction, a pitch, or a vibe, DROP it.
- Judge materiality, NOT sentiment; ignore which specific ticker is named.
- Be directionally honest: never soften or omit negatives, and never inflate promotional framing into substance. A downplayed risk is worth more than a loud positive.
- Quality over coverage. Most promotional articles should yield ZERO or ONE box. If nothing qualifies, return {{"boxes": []}}.
- Emit a box ONLY if you can attach at least one VERBATIM, fact-bearing quote from the article that justifies its label. No quotable fact -> no box.

Return ONE JSON object: {{"boxes": [ ... ]}}. Each element is an object:
{{"label": "evidance-event" | "informative",
  "topic": "<short label, 3-6 words>",
  "insight": "<1-2 sentences: the fact and its concrete implication; state bullish/bearish/neutral ONLY when the fact supports it>",
  "quotes": ["<verbatim substring of the article>", "..."]}}

Rules:
- quotes are VERBATIM substrings, copied character-for-character; 1-3 per box; never paraphrase, shorten, or stitch non-adjacent text.
- Never invent facts or quotes. Output ONLY the JSON object — no markdown, no preamble.

Example "boxes" element:
{{"label": "evidance-event", "topic": "Q3 revenue beat", "insight": "The company topped estimates on cloud strength and raised guidance — a bullish, confirmed demand signal.", "quotes": ["Revenue rose 18% year over year to $4.2 billion.", "The company raised its full-year outlook."]}}

ARTICLE:
\"\"\"
{article}
\"\"\""""

GATE_PROMPT_TEMPLATE = """You are the STRICT GATE on an EVIDENCE FILE for a trading ACT/DO_NOT_ACT judge. You receive ONE already-extracted box. Decide its fate. DEFAULT TO DROP: keep ONLY if it clears a high bar.

Return exactly one decision:
- "evidance-event": a concrete, first-hand, MATERIAL fact that verifiably HAPPENED or was REPORTED in a disclosure / filing / release — reported earnings/guidance/margins/segment figures, completed M&A, a signed deal with substance, a lawsuit filed, a recall, layoffs, a regulatory ruling, a material macro release (rates/jobs/inflation/tariffs/GDP), a disclosed insider/institutional position, a real price move caused by news. A FACT, bullish or bearish.
- "informative": ONE discrete ANNOUNCED soft development — an announced partnership/MoU/LoI/proposed deal, a product LAUNCH (the launch itself), an announced funding round/expansion, a planned future action, or a single named major analyst upgrade/downgrade that is itself the news.
- "DROP": everything else.

THESIS / OPINION TEST — this is where almost every mistake happens. DROP the box if it is the author's ARGUMENT rather than a reported happening, EVEN WHEN it cites real numbers:
- a company's competitive position, market share, "moat", "dominance", strategy, or growth story;
- valuation, price targets, multiples ("trading at Nx"), "undervalued/overvalued", "could/should/may", "potential upside", return forecasts, ratings opinion;
- capex / industry / demand TRENDS, "AI tailwinds", market-size / TAM / CAGR projections;
- product features / capabilities / specs / "how it works" / benefits; company description or history;
- analyst or pundit views about whether a stock is cheap, a buy, or will rise/fall;
- speculation, predictions, "is X the next Y", motivational or thought-leadership content;
- political claims, talking points, or policy opinions;
- routine buybacks / flat dividends / market wraps / index moves.

KEY DISTINCTION: a number is "evidance-event" ONLY when it is a SPECIFIC result DISCLOSED in an earnings report or filing (e.g. "Q3 revenue rose 19% to $4.2B", "raised guidance to $X"). The SAME kind of number is DROP when it is used to ARGUE a case ("~40% market share", "$117B run-rate" cited as a growth story, "trading at 21x forward"). If you removed the author's opinion and NO discrete event/filing remains, DROP.

MARKET-MECHANISM TEST (for political / government / legal / geopolitical / human-interest items). Many things really HAPPEN without being market evidence. Keep such an item ONLY if it acts on a SPECIFIC company, sector, traded asset, interest rate, or the broad market through a CONCRETE, identifiable channel — e.g. a government equity stake in a named public company, a tariff or sanction on traded goods, a rate decision, a regulatory ruling on an industry, a major macro/fiscal release. DROP it when it is political / governance / legal / cultural / human-interest news with no direct market mechanism — agency restructurings, FBI raids or criminal referrals, security-clearance revocations, election or voter-trend claims, social-media accounts, sports events, missing-persons or trafficking policy, personnel feuds, partisan talking points — EVEN THOUGH each really occurred. Ask: "through what concrete channel would this move a specific stock, sector, rate, or commodity?" No clear answer -> DROP.
  KEEP: "US government acquires a 10% equity stake in Intel ($11B)" (direct stake in a named public company); "US imposes 25% tariff on India" (tariff on traded goods); "Fed holds rates" (rate decision).
  DROP: "FBI raids a former official's home"; "Navy deploys ships near Venezuela"; "4.5M voters switched parties"; "agency to be restructured"; "White House launches a TikTok account".

When genuinely unsure, DROP.

BOX:
TOPIC: {topic}
INSIGHT: {insight}
Return only {{"decision": "evidance-event" | "informative" | "DROP"}}."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN)
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone() is None:
            raise SystemExit("pgvector extension is not installed; run "
                             "CREATE EXTENSION vector; in this database first.")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS public.distilled_article_insights (
                id            bigserial PRIMARY KEY,
                article_id    bigint NOT NULL
                              REFERENCES public.articles(id) ON DELETE CASCADE,
                source_url       text,
                article_headline text,
                box_index     int  NOT NULL,
                topic         text,
                insight       text,
                quotes        text[],
                box_text      text NOT NULL,
                first_label   text,     -- label from pass 1 (flash-lite extraction)
                second_label  text,     -- decision from pass 2 (flash gate): keep tier or DROP
                model         text,     -- extraction model that answered
                embedding     vector({EMBED_DIM}),  -- all boxes embedded (kept + dropped)
                created_at    timestamptz NOT NULL DEFAULT now(),
                UNIQUE (article_id, box_index)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS distilled_article_insights_article_id_idx "
                    "ON public.distilled_article_insights (article_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS distilled_article_insights_embedding_idx "
                    "ON public.distilled_article_insights "
                    "USING hnsw (embedding vector_cosine_ops);")
        # processed marker (covers articles that yielded zero boxes)
        cur.execute("ALTER TABLE public.articles "
                    "ADD COLUMN IF NOT EXISTS distilled_insights_extracted_at timestamptz")
    conn.commit()


def articles_to_process(conn, reprocess: bool, limit: Optional[int],
                        ids: Optional[Sequence[int]]):
    clauses = ["content IS NOT NULL", "char_length(content) > 0"]
    params: List[object] = []
    if ids:
        clauses.append("id = ANY(%s)")
        params.append(list(ids))
    elif not reprocess:
        clauses.append("distilled_insights_extracted_at IS NULL")
    sql = ("SELECT id, COALESCE(url_canonical, url), title, content "
           "FROM public.articles WHERE " + " AND ".join(clauses) + " ORDER BY id")
    if limit:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
_client = None
_ext_cfg = None
_gate_cfg = None


def load_gemini():
    global _client, _ext_cfg, _gate_cfg
    if _client is None:
        from google import genai
        from google.genai import types

        if not os.getenv("GOOGLE_API_KEY"):
            raise SystemExit("GOOGLE_API_KEY is not set (put it in .env).")
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        _ext_cfg = types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {"boxes": {"type": "ARRAY", "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "enum": LABELS},
                        "topic": {"type": "STRING"},
                        "insight": {"type": "STRING"},
                        "quotes": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["label", "topic", "insight", "quotes"]}}},
                "required": ["boxes"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        _gate_cfg = types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {"decision": {"type": "STRING",
                                            "enum": LABELS + ["DROP"]}},
                "required": ["decision"],
            },
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
    return _client, _ext_cfg, _gate_cfg


def _retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in ("deadline_exceeded", "504", "503", "unavailable",
                                "timeout", "timed out", "429", "resource_exhausted"))


def _usage(resp) -> Tuple[int, int]:
    um = getattr(resp, "usage_metadata", None)
    return (getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0)


def generate_boxes(client, cfg, article: str, retries: int = 5):
    """Pass 1. Return (boxes, in_tok, out_tok, model_used)."""
    prompt = PROMPT_TEMPLATE.format(article=article[:MAX_ARTICLE_CHARS])
    model, last = EXTRACT_MODEL, "no response"
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            ti, to = _usage(resp)
            data = json.loads(resp.text or "{}")
            if isinstance(data.get("boxes"), list):
                return [b for b in data["boxes"] if isinstance(b, dict)], ti, to, model
            last = "no boxes array"
            if model != FALLBACK_MODEL and attempt >= 1:
                model = FALLBACK_MODEL
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)
            if model != FALLBACK_MODEL and _retryable(exc):
                model = FALLBACK_MODEL
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"extraction failed after {retries} tries: {last}")


def gate_decision(client, cfg, topic: str, insight: str, first_label: str,
                  retries: int = 4):
    """Pass 2. Return (decision, in_tok, out_tok). Fail-OPEN -> keep w/ first_label."""
    prompt = GATE_PROMPT_TEMPLATE.format(topic=topic, insight=(insight or "")[:600])
    model = GATE_MODEL
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            ti, to = _usage(resp)
            d = json.loads(resp.text or "{}").get("decision")
            return (d or first_label or "informative"), ti, to
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return first_label or "informative", 0, 0   # fail-open keep


# --------------------------------------------------------------------------- #
# Verbatim quotes + embeddings
# --------------------------------------------------------------------------- #
_WRAP = " \t\"'“”‘’"


def _verbatim(quote: str, article: str, thr: float) -> Optional[str]:
    q = quote.strip().strip(_WRAP).strip()
    if not q:
        return None
    if q in article:
        return q
    sm = difflib.SequenceMatcher(None, article, q, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    a0 = max(0, blocks[0].a - blocks[0].b)
    last = blocks[-1]
    a1 = min(len(article), last.a + last.size + (len(q) - (last.b + last.size)))
    if a1 <= a0:
        return None
    cand = article[a0:a1]
    return cand if difflib.SequenceMatcher(None, q, cand).ratio() >= thr else None


def verbatimize(quotes, article: str, thr: float) -> List[str]:
    out, seen = [], set()
    for q in quotes or []:
        m = _verbatim(str(q), article, thr)
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


_openai = None


def embed_texts(texts: List[str]) -> Tuple[List[List[float]], int]:
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI()  # OPENAI_API_KEY
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data], (resp.usage.total_tokens or 0)


def build_box_text(headline: str, topic: str, insight: str, quotes: List[str]) -> str:
    lines = [f"ARTICLE_HEADLINE: {headline or ''}", f"TOPIC: {topic}",
             f"INSIGHT: {insight}", "QUOTES:"]
    lines.extend(quotes)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# DB write
# --------------------------------------------------------------------------- #
def write_article(conn, article_id: int, rows: List[dict]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.distilled_article_insights WHERE article_id = %s",
                    (article_id,))
        for r in rows:
            cur.execute(
                """INSERT INTO public.distilled_article_insights
                   (article_id, source_url, article_headline, box_index, topic,
                    insight, quotes, box_text, first_label, second_label, model,
                    embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (r["article_id"], r["source_url"], r["article_headline"],
                 r["box_index"], r["topic"], r["insight"], r["quotes"],
                 r["box_text"], r["first_label"], r["second_label"], r["model"],
                 r["embedding"]),
            )
        cur.execute("UPDATE public.articles SET distilled_insights_extracted_at = now() "
                    "WHERE id = %s", (article_id,))
    conn.commit()


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", help="comma-separated article ids (implies reprocess)")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--reprocess", action="store_true",
                   help="re-do already-processed articles")
    p.add_argument("--no-embed", action="store_true",
                   help="skip OpenAI embeddings (survivors keep NULL embedding)")
    p.add_argument("--quote-threshold", type=float, default=DEFAULT_QUOTE_THRESHOLD)
    args = p.parse_args(argv)

    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    client, ext_cfg, gate_cfg = load_gemini()

    conn = get_conn()
    ensure_schema(conn)
    todo = articles_to_process(conn, args.reprocess, args.limit, ids)
    print(f"distilling {len(todo)} articles -> public.distilled_article_insights "
          f"(extract={EXTRACT_MODEL}, gate={GATE_MODEL}"
          f"{', no-embed' if args.no_embed else ', embed=' + EMBED_MODEL})")

    cost = Counter()

    def process(article):
        aid, url, title, content = article
        body = content or ""
        text = f"ARTICLE_HEADLINE: {title}\n\n{body}".strip()
        c = Counter()
        if not text:
            return aid, [], c, "ok"
        try:
            boxes, ti, to, emodel = generate_boxes(client, ext_cfg, text)
        except Exception as exc:  # one bad article must NOT kill the whole run
            return aid, None, c, f"extract-error: {repr(exc)[:120]}"
        c["ext_in"] += ti
        c["ext_out"] += to
        rows, to_embed = [], []
        for bi, b in enumerate(boxes):
            first = (b.get("label") or "").strip()
            topic = (b.get("topic") or "").strip()
            insight = (b.get("insight") or "").strip()
            quotes = verbatimize(b.get("quotes"), body, args.quote_threshold)
            decision, gi, go = gate_decision(client, gate_cfg, topic, insight, first)
            c["gate_in"] += gi
            c["gate_out"] += go
            row = {
                "article_id": aid, "source_url": url, "article_headline": title,
                "box_index": bi, "topic": topic, "insight": insight,
                "quotes": quotes,
                "box_text": build_box_text(title, topic, insight, quotes),
                "first_label": first, "second_label": decision,
                "model": emodel, "embedding": None,
            }
            rows.append(row)
            if not args.no_embed:
                to_embed.append(row)   # embed ALL boxes (incl. DROP) for later checks
        if to_embed:
            try:
                vecs, etok = embed_texts([r["box_text"] for r in to_embed])
                c["embed_tok"] += etok
                for r, v in zip(to_embed, vecs):
                    r["embedding"] = v
            except Exception:  # keep boxes (NULL embedding) — backfill later, don't lose the article
                pass
        return aid, rows, c, "ok"

    n_box = n_drop = n_done = n_failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, a) for a in todo]
        for fut in as_completed(futs):
            aid, rows, c, status = fut.result()
            cost.update(c)
            if rows is None:                      # extraction failed -> skip, leave unstamped to retry
                n_failed += 1
                print(f"  [skip] article {aid}: {status}", flush=True)
                continue
            write_article(conn, aid, rows)        # main-thread DB write (also stamps article)
            n_box += len(rows)
            n_drop += sum(1 for r in rows if r["second_label"] == "DROP")
            n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                running = (cost["ext_in"] * RATE["ext_in"] + cost["ext_out"] * RATE["ext_out"]
                           + cost["gate_in"] * RATE["gate_in"] + cost["gate_out"] * RATE["gate_out"]
                           + cost["embed_tok"] * RATE["embed"])
                print(f"  {n_done}/{len(todo)} articles | {n_box} boxes "
                      f"({n_box - n_drop} kept / {n_drop} dropped) | "
                      f"{n_failed} failed | ${running:.4f} so far", flush=True)
    conn.close()

    ext = cost["ext_in"] * RATE["ext_in"] + cost["ext_out"] * RATE["ext_out"]
    gate = cost["gate_in"] * RATE["gate_in"] + cost["gate_out"] * RATE["gate_out"]
    emb = cost["embed_tok"] * RATE["embed"]
    print("\n=== token cost (approx; verify vs billing) ===")
    print(f"  extract (flash-lite): in {cost['ext_in']:>9,}  out {cost['ext_out']:>8,}  ${ext:.4f}")
    print(f"  gate    (flash-lite): in {cost['gate_in']:>9,}  out {cost['gate_out']:>8,}  ${gate:.4f}")
    print(f"  embed   (3-small)   : tok {cost['embed_tok']:>8,}  {' ' * 18}${emb:.4f}")
    print(f"  TOTAL                                                  ${ext + gate + emb:.4f}")
    print(f"\nboxes stored: {n_box}  (kept {n_box - n_drop} / dropped {n_drop})")
    if n_failed:
        print(f"articles skipped after extraction failures: {n_failed} "
              f"(left unstamped — re-run to retry them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
