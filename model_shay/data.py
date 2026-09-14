"""Loading, filtering and preprocessing for the regressor (spec §1).

The corpus is `public.articles` in the `news_trading_window` DB, already
restricted to NYSE trading days between the open and 90 minutes before the
close, so the +90-minute target window always lands inside the session.

Only rows with all five model fields present are usable. On top of that the
81 rows flagged `insights_too_long_more_than_500` are DROPPED rather than
truncated, so no example loses content at the 512-token ceiling -- which is
what lets §1 pad to a flat 512 with a clear conscience.

Torch and transformers are imported inside the functions that need them, so
`baselines` and `data-report` work without pulling in the whole stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from model_shay.config import Config

FLOAT_COLUMNS = ["float_1", "float_2", "float_3"]
MODEL_SETS = ("training", "validation", "test")

# Which columns fill float_1..3 and target depends on cfg.window (§1). In the
# at_article window float_1 is VIXY as a raw PRICE LEVEL, which decays over the
# corpus and so partly encodes calendar time -- accepted, not corrected. In the
# 30m_before window it is a VIX *gain* instead, which does not have that drift.
QUERY = """
SELECT id,
       {text_column}                                          AS text,
       {float_1}                                              AS float_1,
       {float_2}                                              AS float_2,
       {float_3}                                              AS float_3,
       {target}                                               AS target,
       COALESCE(first_mentioned_ticker, first_listed_ticker)  AS ticker,
       (published_utc AT TIME ZONE 'America/New_York')::date   AS et_date,
       model_set,
       test_unseen,
       is_act_from_insights,
       sentiment
FROM public.articles
WHERE {where}
ORDER BY id
"""


def _identifier(name: str) -> str:
    """Guard the columns we interpolate into SQL (same check as the enrichment scripts)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"not a plain SQL identifier: {name!r}")
    return name


def load_frame(cfg: Config, *, require_target: bool = True,
               require_split: bool = True) -> pd.DataFrame:
    """Every usable row as a DataFrame: id, text, float_1..3, target, ticker,
    et_date, model_set, group."""
    if cfg.input_path:
        df = _load_file(Path(cfg.input_path))
    else:
        df = _load_db(cfg, require_target=require_target, require_split=require_split)

    missing = {"text", *FLOAT_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"input is missing column(s) {sorted(missing)}")

    before = len(df)
    subset = list(FLOAT_COLUMNS) + ["text"] + (["target"] if require_target else [])
    if cfg.use_sentiment:
        subset.append("sentiment")
    if cfg.use_act:
        subset.append("is_act_from_insights")
    df = df.dropna(subset=subset)
    if require_split and "model_set" in df.columns:
        df = df[df["model_set"].isin(MODEL_SETS)]
    df = df.reset_index(drop=True)

    df["group"] = _group_key(df, cfg.group_by)
    if cfg.limit:
        df = df.head(int(cfg.limit)).reset_index(drop=True)
    print(f"[data] {len(df)} usable row(s) of {before} loaded "
          f"({cfg.ticker_variant} ticker, window={cfg.window})", flush=True)
    print(f"[data] target={cfg.target_column}", flush=True)
    if cfg.act_filter is not None:
        print(f"[data] is_act_from_insights filter: keeping "
              f"{'actionable' if cfg.act_filter else 'non-actionable'} rows only", flush=True)
    elif "is_act_from_insights" in df.columns:
        act = df["is_act_from_insights"]
        print(f"[data] is_act_from_insights: {int(act.sum())} actionable / "
              f"{int((~act.astype(bool)).sum())} not (unfiltered)", flush=True)
    if cfg.use_floats:
        print(f"[data] floats={list(cfg.float_columns)}", flush=True)
    if cfg.use_act:
        n_act = int(df["is_act_from_insights"].fillna(False).astype(bool).sum())
        print(f"[data] act one-hot input (2 dims): {n_act} actionable / "
              f"{len(df) - n_act} not", flush=True)
    if cfg.use_sentiment:
        counts = df["sentiment"].value_counts().sort_index().to_dict()
        print("[data] sentiment one-hot input (5 dims): "
              + str({int(k): int(v) for k, v in counts.items()}), flush=True)
    return df


def _load_db(cfg: Config, *, require_target: bool, require_split: bool) -> pd.DataFrame:
    float_1, float_2, float_3 = cfg.float_columns
    where = [f"{_identifier(cfg.text_column)} IS NOT NULL", "published_utc IS NOT NULL"]
    # Require the floats non-null even in a text-only run, so the row set stays
    # comparable across experiments (vix/aiq are populated for every row, so
    # this costs nothing in practice).
    where += [f"{c} IS NOT NULL" for c in (float_1, float_2, float_3)]
    if require_target:
        where.append(f"{cfg.target_column} IS NOT NULL")
    if require_split:
        where.append("model_set IS NOT NULL")
    if cfg.exclude_long_insights:
        # The 81 over-budget rows (§1). COALESCE so an unflagged row is kept.
        where.append("NOT COALESCE(insights_too_long_more_than_500, false)")
    if cfg.use_sentiment:
        where.append("sentiment IS NOT NULL")
    if cfg.use_act:
        where.append("is_act_from_insights IS NOT NULL")
    if cfg.act_filter is not None:
        # IS TRUE / IS FALSE rather than = so NULL never sneaks into either side.
        where.append(f"is_act_from_insights IS {'TRUE' if cfg.act_filter else 'FALSE'}")
    if cfg.window == "30m_before" and cfg.exclude_first_market_30m:
        # Their -30m anchor is clamped to the open, so the pre-window is short
        # and the target spans under 120m -- a different measurement.
        where.append("NOT COALESCE(article_in_first_market_30m, false)")

    sql = QUERY.format(text_column=_identifier(cfg.text_column),
                       float_1=float_1, float_2=float_2, float_3=float_3,
                       target=cfg.target_column,
                       where=" AND ".join(where))
    with _connect(cfg) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _connect(cfg: Config):
    if cfg.dsn:
        import psycopg
        return psycopg.connect(cfg.dsn)
    from ticker_news.shared.db import connect
    return connect()


def _load_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    raise ValueError(f"unsupported input {path.suffix!r}; use .csv/.tsv/.parquet")


def _group_key(df: pd.DataFrame, group_by: Sequence[str]) -> pd.Series:
    """The GroupKFold key (§5): rows sharing it must never straddle a fold.

    46% of the corpus shares a (ticker, et_date) with another row and the
    largest such cluster holds 30 articles -- two pieces on the same name
    minutes apart share most of their 90-minute forward window, so their
    targets are near-duplicates.
    """
    available = [c for c in group_by if c in df.columns]
    if not available:
        print(f"[data] WARNING none of group_by={list(group_by)} are present; "
              f"falling back to one group per row, which DISABLES leakage "
              f"protection in k-fold.", flush=True)
        return pd.Series(df.index.astype(str), index=df.index)
    if len(available) != len(group_by):
        print(f"[data] WARNING grouping on {available}, missing "
              f"{sorted(set(group_by) - set(available))}", flush=True)
    return df[available].astype(str).agg("|".join, axis=1)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The model_set split written by assign_model_set.py -- never re-derived (§1)."""
    out = {name: df[df["model_set"] == name].reset_index(drop=True) for name in MODEL_SETS}
    print("[split] " + "  ".join(f"{k}={len(v)}" for k, v in out.items()), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
@dataclass
class FloatScaler:
    """Standardize the 3 floats. Fit on the training fold ONLY (§1)."""

    mean: list[float]
    scale: list[float]
    log_transform: bool = False

    @staticmethod
    def _maybe_log(x: np.ndarray, on: bool) -> np.ndarray:
        # Signed log1p: the gains go negative, so plain log would be undefined.
        return np.sign(x) * np.log1p(np.abs(x)) if on else x

    @classmethod
    def fit(cls, x: np.ndarray, log_transform: bool = False) -> "FloatScaler":
        x = cls._maybe_log(np.asarray(x, dtype=np.float64), log_transform)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mean=x.mean(axis=0).tolist(), scale=scale.tolist(),
                   log_transform=log_transform)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._maybe_log(np.asarray(x, dtype=np.float64), self.log_transform)
        return ((x - np.array(self.mean)) / np.array(self.scale)).astype(np.float32)


@dataclass
class TargetTransform:
    """Maps the target into training space and back.

    Not in the spec, and it matters: the raw target is a 90-minute return with
    sigma = 1.17%, so `HuberLoss(delta=1.0)` would keep every residual inside
    the quadratic region and silently degenerate to MSE -- making `loss_type`
    a no-op. Standardizing (default) puts residuals on a scale where
    huber_delta actually bites. Metrics are always inverse-transformed back to
    return units so they stay comparable across settings and against §7.
    """

    kind: str
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, y: np.ndarray, kind: str) -> "TargetTransform":
        y = np.asarray(y, dtype=np.float64)
        if kind == "classes":
            return cls("classes")
        if kind == "standardize":
            std = float(y.std()) or 1.0
            return cls("standardize", float(y.mean()), std)
        return cls(kind)

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.kind == "classes":
            # 1..5 label -> 0..4 class index, which is what CrossEntropyLoss wants.
            return (y - 1).astype(np.int64)
        if self.kind == "standardize":
            return ((y - self.mean) / self.std).astype(np.float32)
        if self.kind == "percent":
            return (y * 100.0).astype(np.float32)
        return y.astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        if self.kind == "classes":
            return np.asarray(z, dtype=np.int64) + 1      # class index -> 1..5
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "standardize":
            return z * self.std + self.mean
        if self.kind == "percent":
            return z / 100.0
        return z


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def get_tokenizer(cfg: Config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.encoder_name)


def length_report(texts: Sequence[str], tokenizer, max_length: int) -> dict[str, Any]:
    """The startup length check (§1).

    After the row filter the truncation count should be 0. A non-zero value
    means the filter or `insights_too_long_more_than_500` is stale, so it warns
    loudly rather than letting content vanish quietly.
    """
    lengths = np.array([len(tokenizer(t, add_special_tokens=True,
                                      truncation=False)["input_ids"]) for t in texts])
    truncated = int((lengths > max_length).sum())
    report = {
        "n": int(lengths.size), "min": int(lengths.min()), "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)), "max": int(lengths.max()),
        "max_length": max_length, "truncated": truncated,
    }
    print(f"[tokens] n={report['n']} min={report['min']} median={report['median']} "
          f"p95={report['p95']} max={report['max']} (max_length={max_length})", flush=True)
    if truncated:
        print(f"[tokens] WARNING {truncated} example(s) exceed max_length and WILL be "
              f"truncated. After the §1 row filter this should be 0 -- re-run "
              f"flag_long_insights.py, or the filter is not being applied.", flush=True)
    else:
        print("[tokens] no truncation: every retained row fits the ceiling.", flush=True)
    return report


SENTIMENT_CLASSES = 5
ACT_CLASSES = 2


def _onehot(codes: np.ndarray, n: int) -> np.ndarray:
    """0-based class codes -> an n-wide one-hot block. Out-of-range rows stay
    all-zero rather than raising, so a stray NULL degrades to "no signal"."""
    out = np.zeros((len(codes), n), dtype=np.float32)
    ok = (codes >= 0) & (codes < n)
    out[np.flatnonzero(ok), codes[ok]] = 1.0
    return out


def sentiment_onehot(values) -> np.ndarray:
    """articles.sentiment (1..5) -> a 5-wide one-hot row per article."""
    return _onehot(np.asarray(values, dtype=np.int64) - 1, SENTIMENT_CLASSES)


def act_onehot(values) -> np.ndarray:
    """is_act_from_insights -> [1,0] not actionable, [0,1] actionable."""
    v = pd.Series(values).fillna(False).astype(bool).to_numpy().astype(np.int64)
    return _onehot(v, ACT_CLASSES)


def build_features(df: pd.DataFrame, cfg: Config, scaler: FloatScaler) -> np.ndarray:
    """The dense vector concatenated to the pooled text: standardized floats,
    then the sentiment one-hot. Either half can be switched off."""
    parts: list[np.ndarray] = []
    if cfg.use_floats:
        parts.append(scaler.transform(df[FLOAT_COLUMNS].to_numpy()))
    if cfg.use_sentiment:
        # Deliberately NOT through the scaler: a one-hot is already on the right
        # scale, and centering it would smear the categories together.
        parts.append(sentiment_onehot(df["sentiment"].to_numpy()))
    if cfg.use_act:
        parts.append(act_onehot(df["is_act_from_insights"].to_numpy()))
    if not parts:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_dataset(df: pd.DataFrame, tokenizer, cfg: Config,
                  scaler: FloatScaler, target_tf: TargetTransform | None):
    """A TensorDataset of (input_ids, attention_mask, dense_features, target)."""
    import torch

    enc = tokenizer(list(df["text"]), max_length=cfg.max_length, truncation=True,
                    padding=cfg.padding, return_tensors="pt")
    floats = torch.from_numpy(build_features(df, cfg, scaler))
    if "target" in df.columns and target_tf is not None:
        y = torch.from_numpy(target_tf.apply(df["target"].to_numpy()))
    else:
        y = torch.zeros(len(df), dtype=torch.float32)
    return torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], floats, y)
