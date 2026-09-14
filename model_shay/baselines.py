"""Sanity baselines (spec §7) -- run once, outside the training loop.

These are load-bearing, not decoration. The target is a 90-minute forward
return with sigma ~117bps and mean ~0, so a neural R2 near zero is an expected
outcome; the only way to read it honestly is to know where the trivial
predictors sit. Three references:

    constant    predict the training mean -- R2 = 0 by construction on train,
                and the number every other model has to beat
    floats      linear regression and gradient boosting on the 3 floats alone
    text        TF-IDF + ridge on the insight text alone

Run with the same split and (optionally) the same GroupKFold as the neural run
so the comparison is apples to apples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from model_shay.config import Config
from model_shay.data import FLOAT_COLUMNS, load_frame, split_frames

METRIC_KEYS = {"regression": ("mse", "rmse", "mae", "mae_bps", "r2"),
               "classification": ("acc", "f1_macro", "within_1")}
HEADLINE = {"regression": ("mae_bps", "r2"), "classification": ("acc", "f1_macro")}


def _fit_score(name: str, fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
               train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> dict[str, float]:
    from model_shay.train import compute_metrics

    pred = fit_predict(train, val)
    return {"model": name, **compute_metrics(cfg, val["target"].to_numpy(), pred)}


def _constant(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), float(train["target"].mean()))


def _floats_linear(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(train[FLOAT_COLUMNS], train["target"])
    return pipe.predict(val[FLOAT_COLUMNS])


def _floats_gbdt(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    est = HistGradientBoostingRegressor(random_state=0)
    est.fit(train[FLOAT_COLUMNS], train["target"])
    return est.predict(val[FLOAT_COLUMNS])


def _text_tfidf(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(min_df=3, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True),
        Ridge(alpha=1.0))
    pipe.fit(train["text"], train["target"])
    return pipe.predict(val["text"])


def _majority(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), train["target"].mode().iloc[0])


def _floats_logreg(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    est = HistGradientBoostingClassifier(random_state=0)
    est.fit(train[FLOAT_COLUMNS], train["target"])
    return est.predict(val[FLOAT_COLUMNS])


def _text_tfidf_logreg(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(min_df=3, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(train["text"], train["target"])
    return pipe.predict(val["text"])


BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "constant_mean": _constant,
    "floats_linear": _floats_linear,
    "floats_gbdt": _floats_gbdt,
    "text_tfidf_ridge": _text_tfidf,
}
# Same three ideas, classifier flavour, for target_kind="sentiment".
CLASSIFIER_BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "majority_class": _majority,
    "floats_gbdt": _floats_logreg,
    "text_tfidf_logreg": _text_tfidf_logreg,
}


def run_baselines(cfg: Config, out_dir: Path, use_kfold: bool | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(cfg)
    splits = split_frames(df)
    use_kfold = cfg.use_kfold if use_kfold is None else use_kfold
    # A text-only run skips the floats baselines: with target_kind
    # intraday_gain_till, float_2 IS the target, so they would score ~1.0 by
    # reading the answer off the input.
    table_all = CLASSIFIER_BASELINES if cfg.task == "classification" else BASELINES
    active = {k: v for k, v in table_all.items()
              if cfg.use_floats or not k.startswith("floats")}
    if len(active) != len(table_all):
        print(f"[baselines] text-only run: skipping "
              f"{sorted(set(table_all) - set(active))}", flush=True)

    if use_kfold:
        from sklearn.model_selection import GroupKFold

        pool = pd.concat([splits["training"], splits["validation"]], ignore_index=True)
        groups = pool["group"].to_numpy()
        folds = list(GroupKFold(n_splits=cfg.k_folds).split(pool, groups=groups))
        print(f"[baselines] GroupKFold k={cfg.k_folds} over {len(pool)} pooled row(s)", flush=True)
        rows: list[dict] = []
        for name, fn in active.items():
            per_fold = [_fit_score(name, fn, pool.iloc[tr], pool.iloc[va], cfg)
                        for tr, va in folds]
            rows.append({"model": name, "n_folds": len(folds),
                         **{f"{k}_mean": float(np.mean([f[k] for f in per_fold]))
                            for k in METRIC_KEYS[cfg.task]},
                         **{f"{k}_std": float(np.std([f[k] for f in per_fold]))
                            for k in METRIC_KEYS[cfg.task]}})
        table = rows
        left, right = (f"{k}_mean" for k in HEADLINE[cfg.task])
    else:
        train, val = splits["training"], splits["validation"]
        print(f"[baselines] train={len(train)} val={len(val)}", flush=True)
        table = [_fit_score(name, fn, train, val, cfg) for name, fn in active.items()]
        left, right = HEADLINE[cfg.task]

    print(f"\n  {'baseline':<20}{left:>14}{right:>12}")
    for row in table:
        print(f"  {row['model']:<20}{row[left]:>14.4f}{row[right]:>12.4f}")
    print()

    result = {"mode": "kfold" if use_kfold else "single", "baselines": table}
    (out_dir / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(table).to_csv(out_dir / "baselines.csv", index=False)
    print(f"[baselines] wrote {out_dir/'baselines.json'}", flush=True)
    return result
