"""Tree model over model_shay's dense feature vector.

Inputs are whatever model_shay would concatenate to the pooled text -- up to 10
columns:

    float_1..3   the market floats for the chosen window (standardized)
    sentiment_1..5   one-hot of articles.sentiment          (--sentiment)
    act_false/act_true  one-hot of is_act_from_insights     (--act)

Everything about row selection, splitting and grouping is imported from
model_shay, so a tree run and a neural run on the same flags see exactly the
same rows. Scaling is a no-op for a tree, but it is applied anyway so the two
models are fed byte-identical features.

The headline output is permutation importance: with a near-zero-signal target,
which columns the tree leans on says more than its R2 does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_shay.config import Config
from model_shay.data import FLOAT_COLUMNS, FloatScaler, build_features, load_frame, split_frames
from model_shay.metrics import compute_metrics

MODELS = ("hgb", "rf", "extra")


@dataclass
class TreeParams:
    """Hyperparameters, all exposed on the CLI -- these are the sensitivity knobs."""

    model: str = "hgb"
    max_iter: int = 400          # hgb: boosting rounds; rf/extra: n_estimators
    learning_rate: float = 0.05  # hgb only
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0   # hgb only
    max_features: float = 1.0        # rf/extra only
    seed: int = 42


def feature_names(cfg: Config) -> list[str]:
    """Must mirror model_shay.data.build_features column order exactly."""
    names: list[str] = []
    if cfg.use_floats:
        names += list(cfg.float_columns)
    if cfg.use_sentiment:
        names += [f"sentiment_{i}" for i in range(1, 6)]
    if cfg.use_act:
        names += ["act_false", "act_true"]
    return names


def build_estimator(cfg: Config, p: TreeParams):
    classify = cfg.task == "classification"
    if p.model == "hgb":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        cls = HistGradientBoostingClassifier if classify else HistGradientBoostingRegressor
        return cls(max_iter=p.max_iter, learning_rate=p.learning_rate,
                   max_depth=p.max_depth, min_samples_leaf=p.min_samples_leaf,
                   l2_regularization=p.l2_regularization, early_stopping="auto",
                   random_state=p.seed)
    from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    if p.model == "rf":
        cls = RandomForestClassifier if classify else RandomForestRegressor
    else:
        cls = ExtraTreesClassifier if classify else ExtraTreesRegressor
    return cls(n_estimators=p.max_iter, max_depth=p.max_depth,
               min_samples_leaf=p.min_samples_leaf, max_features=p.max_features,
               n_jobs=-1, random_state=p.seed)


def xy(df: pd.DataFrame, cfg: Config, scaler: FloatScaler) -> tuple[np.ndarray, np.ndarray]:
    y = df["target"].to_numpy()
    return build_features(df, cfg, scaler), (y.astype(int) if cfg.task == "classification" else y)


def fit_fold(train: pd.DataFrame, val: pd.DataFrame, cfg: Config,
             p: TreeParams) -> tuple[Any, FloatScaler, dict[str, float], np.ndarray]:
    """Fit on the training fold only -- scaler included, so no leakage."""
    scaler = FloatScaler.fit(train[FLOAT_COLUMNS].to_numpy(), cfg.log_transform_floats)
    x_tr, y_tr = xy(train, cfg, scaler)
    x_va, y_va = xy(val, cfg, scaler)
    est = build_estimator(cfg, p).fit(x_tr, y_tr)
    pred = est.predict(x_va)
    return est, scaler, compute_metrics(cfg, y_va, pred), pred


def constant_reference(train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """The line every model has to beat: predict the training fold's mean (or
    its majority class). Reported alongside so an R2 near 0 is readable."""
    y_tr, y_va = train["target"].to_numpy(), val["target"].to_numpy()
    const = (train["target"].mode().iloc[0] if cfg.task == "classification" else y_tr.mean())
    return compute_metrics(cfg, y_va, np.full(len(y_va), const))


def importances(est, x: np.ndarray, y: np.ndarray, names: list[str],
                cfg: Config, seed: int) -> list[dict[str, float]]:
    """Permutation importance on the held-out fold.

    HistGradientBoosting exposes no feature_importances_, and impurity-based
    importance is misleading for one-hot columns anyway -- it rewards splits
    that never generalized. Permutation is measured on data the tree did not
    fit, so a column that matters only in-sample scores ~0 here.
    """
    from sklearn.inspection import permutation_importance

    r = permutation_importance(est, x, y, n_repeats=20, random_state=seed, n_jobs=-1)
    rows = [{"feature": n, "importance": float(m), "std": float(s)}
            for n, m, s in zip(names, r.importances_mean, r.importances_std)]
    return sorted(rows, key=lambda d: -d["importance"])


def run(cfg: Config, p: TreeParams, out_dir: Path, use_kfold: bool,
        eval_test: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_frame(cfg)
    splits = split_frames(df)
    names = feature_names(cfg)
    print(f"[tree] {p.model} on {len(names)} feature(s): {names}", flush=True)
    if not names:
        raise SystemExit("[tree] no features selected -- a tree needs at least one of "
                         "--floats / --sentiment / --act")

    folds: list[dict] = []
    if use_kfold:
        from sklearn.model_selection import GroupKFold

        pool = pd.concat([splits["training"], splits["validation"]], ignore_index=True)
        groups = pool["group"].to_numpy()
        print(f"[tree] GroupKFold k={cfg.k_folds} over {len(pool)} rows / "
              f"{len(set(groups))} groups", flush=True)
        splitter = GroupKFold(n_splits=cfg.k_folds)
        for i, (tr, va) in enumerate(splitter.split(pool, groups=groups), start=1):
            train, val = pool.iloc[tr], pool.iloc[va]
            est, scaler, metrics, _ = fit_fold(train, val, cfg, p)
            folds.append({"fold": f"fold_{i}", **metrics,
                          "constant": constant_reference(train, val, cfg)})
            print(f"  fold_{i}: " + _fmt(metrics, cfg), flush=True)
        last_train, last_val = pool.iloc[tr], pool.iloc[va]
    else:
        train, val = splits["training"], splits["validation"]
        est, scaler, metrics, _ = fit_fold(train, val, cfg, p)
        folds.append({"fold": "single", **metrics,
                      "constant": constant_reference(train, val, cfg)})
        print(f"  single: " + _fmt(metrics, cfg), flush=True)
        last_train, last_val = train, val

    keys = [k for k in folds[0] if k not in ("fold", "constant")]
    aggregate = {k: {"mean": float(np.mean([f[k] for f in folds])),
                     "std": float(np.std([f[k] for f in folds]))} for k in keys}
    const_agg = {k: float(np.mean([f["constant"][k] for f in folds])) for k in keys}

    x_va, y_va = xy(last_val, cfg, scaler)
    imp = importances(est, x_va, y_va, names, cfg, p.seed)
    print("\n[tree] permutation importance (held-out fold)")
    for row in imp:
        bar = "#" * max(0, min(40, int(round(row["importance"] * 2000))))
        print(f"  {row['feature']:<48}{row['importance']:>+9.5f} +-{row['std']:.5f} {bar}")

    summary = {"model": p.model, "features": names, "n_features": len(names),
               "config": cfg.as_dict(), "params": p.__dict__,
               "folds": folds, "aggregate": aggregate,
               "constant_reference": const_agg, "importance": imp}

    if eval_test and not splits["test"].empty:
        x_te, y_te = xy(splits["test"], cfg, scaler)
        summary["test"] = compute_metrics(cfg, y_te, est.predict(x_te))
        print(f"\n[test] n={len(y_te)} " + _fmt(summary["test"], cfg), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(imp).to_csv(out_dir / "importance.csv", index=False)
    import joblib
    joblib.dump({"estimator": est, "scaler": scaler, "config": cfg.as_dict(),
                 "features": names}, out_dir / "model.joblib")
    print(f"\n[tree] wrote {out_dir}/summary.json, importance.csv, model.joblib", flush=True)
    _report(aggregate, const_agg, cfg)
    return summary


def _fmt(m: dict[str, float], cfg: Config) -> str:
    return (f"acc={m['acc']*100:.1f}% f1={m['f1_macro']:.3f}" if cfg.task == "classification"
            else f"mae={m['mae_bps']:.1f}bps r2={m['r2']:+.4f}")


def _report(agg: dict, const: dict, cfg: Config) -> None:
    key = "acc" if cfg.task == "classification" else "r2"
    print(f"\n[tree] {key}: {agg[key]['mean']:+.4f} +- {agg[key]['std']:.4f}   "
          f"constant reference {const[key]:+.4f}   "
          f"=> {'BEATS' if agg[key]['mean'] > const[key] else 'does not beat'} it", flush=True)
