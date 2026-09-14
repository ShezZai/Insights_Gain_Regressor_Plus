"""model-shay: CLI for the text + market-floats regressor (model_shay_spec.md).

    model-shay data-report                          # §1 length/row check, no training
    model-shay baselines                            # §7 reference lines
    model-shay train --no-kfold --eval-test         # single split off model_set
    model-shay train --kfold --k-folds 5            # GroupKFold CV (§5)
    model-shay train --train-mode lora --lora-r 16
    model-shay train --loss-type mse --huber-delta 2.0
    model-shay predict --checkpoint runs/.../best.pt --model-set test --out preds.csv

Every knob that moves results is a flag, so a sensitivity sweep is a shell
loop and never an edit. Flags default to None and override --config only when
passed, so `--config mine.yaml --loss-type mse` changes exactly one thing.

Heavy imports (torch, transformers, sklearn) happen inside the subcommand that
needs them, so --help and data-report stay usable in a bare environment.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from model_shay.config import CHOICES, DEFAULT_CONFIG_PATH, Config

CONFIG_FIELDS = {f.name for f in fields(Config)}


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Every sensitivity-relevant config key as a flag (default None = leave YAML alone)."""
    flag = argparse.BooleanOptionalAction

    g = parser.add_argument_group("data (§1)")
    g.add_argument("--dsn", help="Postgres DSN (default: DATABASE_URL via ticker_news)")
    g.add_argument("--input-path", help=".csv/.parquet export instead of the DB")
    g.add_argument("--text-column", help="default consolidated_insights")
    g.add_argument("--ticker-variant", choices=CHOICES["ticker_variant"],
                   help="which float_2/target pair: primary_ticker or first_mentioned_ticker")
    g.add_argument("--window", choices=CHOICES["window"],
                   help="at_article = features/target measured at the article; "
                        "30m_before = measured 30 minutes earlier, target -30m..+90m")
    g.add_argument("--30m", dest="window_30m", action="store_true",
                   help="shorthand for --window 30m_before")
    g.add_argument("--30m_till", dest="window_30m_till", action="store_true",
                   help="shorthand for --window 30m_before --target-kind intraday_gain_till: "
                        "inputs are the market state up to 30m BEFORE the article (plus the "
                        "text); the target is the move over those last 30 minutes, up to the "
                        "article. Input and target do not overlap, so the floats stay on")
    g.add_argument("--sentiment", dest="use_sentiment", action=flag, default=None,
                   help="feed articles.sentiment (1-5) as a 5-dim ONE-HOT INPUT appended "
                        "to the dense vector. To PREDICT sentiment instead, use "
                        "--target-kind sentiment")
    g.add_argument("--act", dest="use_act", action=flag, default=None,
                   help="feed articles.is_act_from_insights as a 2-dim ONE-HOT INPUT "
                        "([1,0] not actionable, [0,1] actionable), same idea as --sentiment")
    g.add_argument("--act-filter", dest="act_filter", action=flag, default=None,
                   help="ROW FILTER (not a feature): keep only actionable rows, or only "
                        "non-actionable with --no-act-filter; omit for every row")
    g.add_argument("--exclude-first-market-30m", action=flag, default=None,
                   help="30m window: drop articles inside the first 30 market minutes, "
                        "whose -30m anchor is clamped to the open (default: drop)")
    g.add_argument("--target-kind", choices=CHOICES["target_kind"],
                   help="gain_90m_after = the real task; intraday_gain_till = the "
                        "already-realized move, a positive control (needs --no-floats)")
    g.add_argument("--floats", dest="use_floats", action=flag, default=None,
                   help="feed the 3 market floats alongside the text (--no-floats = text only)")
    g.add_argument("--exclude-long-insights", action=flag, default=None,
                   help="drop the 81 rows over 500 tokens instead of truncating them")
    g.add_argument("--max-length", type=int, help="token ceiling (default 512)")
    g.add_argument("--padding", help="max_length (spec) | longest")
    g.add_argument("--log-transform-floats", action=flag, default=None,
                   help="signed log1p on the 3 floats")
    g.add_argument("--target-transform", choices=CHOICES["target_transform"],
                   help="training space for the target; standardize keeps huber_delta meaningful")
    g.add_argument("--limit", type=int, help="first N rows only (smoke tests)")

    g = parser.add_argument_group("model (§2)")
    g.add_argument("--encoder-name", help="HF backbone (default microsoft/deberta-v3-base)")
    g.add_argument("--pooling", choices=CHOICES["pooling"])
    g.add_argument("--head-hidden-dim", type=int)
    g.add_argument("--head-dropout", type=float)

    g = parser.add_argument_group("loss (§3)")
    g.add_argument("--loss-type", choices=CHOICES["loss_type"])
    g.add_argument("--huber-delta", type=float)

    g = parser.add_argument_group("training (§4)")
    g.add_argument("--train-mode", choices=CHOICES["train_mode"])
    g.add_argument("--freeze-except-last-n", type=int)
    g.add_argument("--lora-r", type=int)
    g.add_argument("--lora-alpha", type=int)
    g.add_argument("--lora-dropout", type=float)
    g.add_argument("--lora-target-modules", nargs="+",
                   help="override the autodetected attention q/v projections")
    g.add_argument("--head-lr", type=float)
    g.add_argument("--encoder-lr", type=float)
    g.add_argument("--phase2-min-lr", type=float, help="discriminative-LR floor in phase 2")
    g.add_argument("--phase1-epochs", type=int, help="only capped when phase 2 is on")
    g.add_argument("--phase2", dest="unfreeze_all_phase2", action=flag, default=None,
                   help="second phase over the whole encoder (partial_freeze only)")
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--warmup-ratio", type=float)
    g.add_argument("--max-grad-norm", type=float)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--grad-accum", type=int, help="steps per optimizer update")
    g.add_argument("--gradient-checkpointing", action=flag, default=None,
                   help="recompute activations to fit phase 2 on a 16GB card")
    g.add_argument("--max-epochs", type=int)
    g.add_argument("--precision", choices=CHOICES["precision"])
    g.add_argument("--seed", type=int)

    g = parser.add_argument_group("early stopping (§6)")
    g.add_argument("--patience", type=int)
    g.add_argument("--monitor", choices=CHOICES["monitor"])

    g = parser.add_argument_group("k-fold (§5)")
    g.add_argument("--kfold", dest="use_kfold", action=flag, default=None,
                   help="GroupKFold CV over pooled training+validation")
    g.add_argument("--k-folds", type=int)


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {k: v for k, v in vars(args).items() if k in CONFIG_FIELDS}
    if getattr(args, "window_30m", False):
        overrides["window"] = "30m_before"
    if getattr(args, "window_30m_till", False):
        overrides["window"] = "30m_before"
        overrides["target_kind"] = "intraday_gain_till"
    cfg = Config.load(args.config, **overrides)
    print(f"[config] {args.config or DEFAULT_CONFIG_PATH}", flush=True)
    return cfg


def default_out_dir(cfg: Config) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "cv" if cfg.use_kfold else "single"
    return Path("runs") / f"{stamp}-{kind}-{cfg.train_mode}-{cfg.loss_type}"


# --------------------------------------------------------------------------- #
def cmd_data_report(args: argparse.Namespace) -> None:
    from model_shay.data import get_tokenizer, length_report, load_frame, split_frames

    cfg = config_from_args(args)
    df = load_frame(cfg)
    split_frames(df)
    print(f"[groups] {df['group'].nunique()} distinct {cfg.group_by} group(s); "
          f"largest holds {df['group'].value_counts().max()} row(s)", flush=True)
    y = df["target"]
    print(f"[target] mean={y.mean():+.6f} sd={y.std():.6f} "
          f"p1={y.quantile(0.01):+.4f} p50={y.median():+.6f} p99={y.quantile(0.99):+.4f}",
          flush=True)
    length_report(df["text"].tolist(), get_tokenizer(cfg), cfg.max_length)


def cmd_baselines(args: argparse.Namespace) -> None:
    from model_shay.baselines import run_baselines

    cfg = config_from_args(args)
    run_baselines(cfg, Path(args.out_dir) if args.out_dir else default_out_dir(cfg) / "baselines")


def cmd_train(args: argparse.Namespace) -> None:
    from model_shay.train import run_kfold, run_single

    cfg = config_from_args(args)
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(cfg)
    print(f"[out] {out_dir}", flush=True)
    if cfg.use_kfold:
        run_kfold(cfg, out_dir, eval_test=args.eval_test, final_fit=args.final_fit)
    else:
        run_single(cfg, out_dir, eval_test=args.eval_test)
    print(f"[done] {out_dir}/summary.json", flush=True)


def cmd_predict(args: argparse.Namespace) -> None:
    from model_shay.predict import run_predict

    cfg = config_from_args(args)
    ids = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids else None
    run_predict(args.checkpoint, cfg, args.out, model_set=args.model_set, ids=ids)


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="model-shay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def new(name: str, help_: str, fn) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--config", help=f"YAML config (default {DEFAULT_CONFIG_PATH.name})")
        p.set_defaults(func=fn)
        add_config_args(p)
        return p

    new("data-report", "Row counts, group sizes, target and token-length distribution (§1).",
        cmd_data_report)

    p = new("baselines", "Constant / floats-only / text-only reference metrics (§7).",
            cmd_baselines)
    p.add_argument("--out-dir", help="where baselines.json/.csv land")

    p = new("train", "Train the regressor: GroupKFold CV or a single model_set split.",
            cmd_train)
    p.add_argument("--out-dir", help="run directory (default runs/<stamp>-...)")
    p.add_argument("--eval-test", action="store_true",
                   help="score the untouched test holdout once at the end")
    p.add_argument("--final-fit", action="store_true",
                   help="after CV, refit on the whole pool for deployment (no held-out val)")

    p = new("predict", "Run a saved .pt against rows and write predictions.", cmd_predict)
    p.add_argument("--checkpoint", required=True, help="path to a best.pt")
    p.add_argument("--model-set", choices=("training", "validation", "test"),
                   help="restrict to one split")
    p.add_argument("--ids", help="comma-separated article ids")
    p.add_argument("--out", help="CSV to write (default: print the first rows)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
