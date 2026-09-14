"""model-tree: gradient-boosted tree over the dense features, no text encoder.

    model-tree train --30m --sentiment --act
    model-tree train --30m --sentiment --act --kfold --out-dir runs/tree-30m
    model-tree train --30m --sentiment --act --model rf --max-depth 6
    model-tree predict --model-dir runs/tree-30m --model-set test --out preds.csv

Data flags mean exactly what they mean in model-shay, and are read from the
same config, so `model-tree train --30m --sentiment --act` and
`model-shay train --30m --sentiment --act` see identical rows and identical
dense features. The difference is the text.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from model_shay.config import CHOICES, DEFAULT_CONFIG_PATH, Config
from model_tree.tree import MODELS, TreeParams

CONFIG_FIELDS = {f.name for f in fields(Config)}


def add_args(p: argparse.ArgumentParser) -> None:
    flag = argparse.BooleanOptionalAction
    p.add_argument("--config", help=f"YAML config (default {DEFAULT_CONFIG_PATH.name})")

    g = p.add_argument_group("data (shared with model-shay)")
    g.add_argument("--dsn")
    g.add_argument("--input-path")
    g.add_argument("--window", choices=CHOICES["window"])
    g.add_argument("--30m", dest="window_30m", action="store_true",
                   help="shorthand for --window 30m_before")
    g.add_argument("--30m_till", dest="window_30m_till", action="store_true",
                   help="shorthand for --window 30m_before --target-kind intraday_gain_till")
    g.add_argument("--ticker-variant", choices=CHOICES["ticker_variant"])
    g.add_argument("--target-kind", choices=CHOICES["target_kind"])
    g.add_argument("--exclude-first-market-30m", action=flag, default=None)
    g.add_argument("--act-filter", dest="act_filter", action=flag, default=None,
                   help="row filter on is_act_from_insights (not a feature)")
    g.add_argument("--limit", type=int)

    g = p.add_argument_group("features (the tree's whole input)")
    g.add_argument("--floats", dest="use_floats", action=flag, default=None,
                   help="the 3 market floats (default on)")
    g.add_argument("--sentiment", dest="use_sentiment", action=flag, default=None,
                   help="5-dim one-hot of articles.sentiment")
    g.add_argument("--act", dest="use_act", action=flag, default=None,
                   help="2-dim one-hot of is_act_from_insights")
    g.add_argument("--log-transform-floats", action=flag, default=None)

    g = p.add_argument_group("tree")
    g.add_argument("--model", choices=MODELS, default="hgb")
    g.add_argument("--max-iter", type=int, default=400,
                   help="boosting rounds (hgb) or n_estimators (rf/extra)")
    g.add_argument("--learning-rate", type=float, default=0.05, help="hgb only")
    g.add_argument("--max-depth", type=int, default=None)
    g.add_argument("--min-samples-leaf", type=int, default=20)
    g.add_argument("--l2-regularization", type=float, default=0.0, help="hgb only")
    g.add_argument("--max-features", type=float, default=1.0, help="rf/extra only")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("evaluation")
    g.add_argument("--kfold", dest="use_kfold", action=flag, default=None,
                   help="GroupKFold over pooled training+validation (default from config)")
    g.add_argument("--k-folds", type=int)
    g.add_argument("--eval-test", action="store_true")
    g.add_argument("--out-dir", default="runs/tree")


def make_config(args: argparse.Namespace) -> Config:
    overrides = {k: v for k, v in vars(args).items() if k in CONFIG_FIELDS}
    if getattr(args, "window_30m", False):
        overrides["window"] = "30m_before"
    if getattr(args, "window_30m_till", False):
        overrides["window"] = "30m_before"
        overrides["target_kind"] = "intraday_gain_till"
    overrides.pop("seed", None)          # the tree's own --seed, not the config's
    return Config.load(args.config, **overrides)


def cmd_train(args: argparse.Namespace) -> None:
    from model_tree.tree import run

    cfg = make_config(args)
    params = TreeParams(model=args.model, max_iter=args.max_iter,
                        learning_rate=args.learning_rate, max_depth=args.max_depth,
                        min_samples_leaf=args.min_samples_leaf,
                        l2_regularization=args.l2_regularization,
                        max_features=args.max_features, seed=args.seed)
    use_kfold = cfg.use_kfold if args.use_kfold is None else args.use_kfold
    run(cfg, params, Path(args.out_dir), use_kfold=use_kfold, eval_test=args.eval_test)


def cmd_predict(args: argparse.Namespace) -> None:
    import joblib
    import numpy as np
    import pandas as pd

    from model_shay.data import load_frame
    from model_shay.metrics import compute_metrics
    from model_tree.tree import xy

    bundle = joblib.load(Path(args.model_dir) / "model.joblib")
    cfg = Config(**bundle["config"])
    df = load_frame(cfg, require_target=False, require_split=False)
    if args.model_set:
        df = df[df["model_set"] == args.model_set].reset_index(drop=True)
    if df.empty:
        raise SystemExit("[predict] no rows selected")

    x, y = xy(df, cfg, bundle["scaler"])
    pred = bundle["estimator"].predict(x)
    out = pd.DataFrame({c: df[c].to_numpy() for c in
                        ("id", "ticker", "et_date", "model_set", "test_unseen", "target")
                        if c in df.columns})
    out["prediction"] = pred
    if "target" in out.columns:
        out["error"] = out["prediction"] - out["target"]
        if out["target"].notna().all():
            m = compute_metrics(cfg, y, pred)
            print(f"[predict] n={len(out)} " + (
                f"acc={m['acc']*100:.1f}% f1={m['f1_macro']:.3f}"
                if cfg.task == "classification"
                else f"mae={m['mae_bps']:.1f}bps r2={m['r2']:+.4f}"), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[predict] wrote {args.out}", flush=True)
    else:
        print(out.head(20).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="model-tree", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="fit the tree and report permutation importance")
    add_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="run a saved tree against rows")
    p.add_argument("--model-dir", required=True, help="directory holding model.joblib")
    p.add_argument("--model-set", choices=("training", "validation", "test"))
    p.add_argument("--out")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
