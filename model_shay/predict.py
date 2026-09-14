"""Run a saved checkpoint against rows (the .pt is the deliverable).

The .pt carries its own config, float scaler and target transform, so nothing
outside it is needed -- no YAML, no peft (LoRA runs are merged before saving),
and no reliance on the caller remembering which ticker_variant was trained.

That last point is a real footgun: scoring `first`-variant rows against a
`primary`-trained model would silently compare different tickers' returns. So
the data-shaping keys come from the CHECKPOINT, and a conflicting CLI flag is
reported rather than honoured.

    model-shay predict --checkpoint runs/cv/fold_1/best.pt --model-set test
    model-shay predict --checkpoint runs/cv/final/best.pt --input-path new.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_shay.config import Config
from model_shay.data import build_dataset, get_tokenizer, load_frame
from model_shay.model import load_checkpoint

# Keys that decide WHICH rows and HOW they are encoded: taken from the
# checkpoint, never from the CLI, so inference cannot drift from training.
CHECKPOINT_OWNED = ("window", "ticker_variant", "target_kind", "use_floats",
                    "use_sentiment", "use_act",
                    "exclude_first_market_30m", "text_column", "max_length", "padding",
                    "exclude_long_insights", "log_transform_floats", "encoder_name")

OUTPUT_COLUMNS = ("id", "ticker", "et_date", "model_set", "test_unseen", "target")


def _predict(model, cfg: Config, scaler, target_tf, df: pd.DataFrame, tokenizer,
             batch_size: int | None, device: torch.device) -> pd.DataFrame:
    ds = build_dataset(df, tokenizer, cfg, scaler, target_tf=None)
    loader = DataLoader(ds, batch_size=batch_size or cfg.batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for input_ids, mask, floats, _ in loader:
            out = model(input_ids.to(device), mask.to(device), floats.to(device))
            preds.append(out.float().cpu().numpy())

    raw = np.concatenate(preds)
    columns = {c: df[c].to_numpy() for c in OUTPUT_COLUMNS if c in df.columns}
    if cfg.task == "classification":
        # Softmax the logits so the per-class confidence ships with the label.
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        columns["prediction"] = target_tf.inverse(raw.argmax(axis=1))
        columns["confidence"] = probs.max(axis=1)
        for i in range(raw.shape[1]):
            columns[f"p{i + 1}"] = probs[:, i]
    else:
        columns["prediction"] = target_tf.inverse(raw)
    out = pd.DataFrame(columns)
    if "target" in out.columns:
        out["error"] = out["prediction"] - out["target"]
    return out


def predict_frame(checkpoint: str | Path, df: pd.DataFrame, *, tokenizer=None,
                  batch_size: int | None = None, device: str | None = None) -> pd.DataFrame:
    """Predictions in RETURN UNITS for every row of `df`.

    Adds `target`/`error` when the frame is labelled, so one call serves both
    scoring a holdout and predicting fresh rows.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, scaler, target_tf, _ = load_checkpoint(checkpoint, dev)
    return _predict(model, cfg, scaler, target_tf, df, tokenizer or get_tokenizer(cfg),
                    batch_size, dev)


def run_predict(checkpoint: str, cfg: Config, out_path: str | None,
                model_set: str | None = None, ids: list[int] | None = None) -> pd.DataFrame:
    """CLI entry: load rows the way training did, score them, write CSV."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_cfg, scaler, target_tf, metrics = load_checkpoint(checkpoint, dev)
    if metrics:
        trained = (f"acc={metrics.get('acc', float('nan'))*100:.1f}% "
                   f"f1={metrics.get('f1_macro', float('nan')):.3f}"
                   if train_cfg.task == "classification" else
                   f"mae={metrics.get('mae_bps', float('nan')):.1f}bps "
                   f"r2={metrics.get('r2', float('nan')):+.4f}")
        print(f"[checkpoint] trained metrics: {trained} "
              f"(best epoch {metrics.get('best_epoch')})", flush=True)

    # The checkpoint decides how rows are selected and encoded; keep only the
    # caller's own choices (dsn, input_path, limit, batch_size).
    load_cfg = Config(**{**cfg.as_dict(),
                         **{k: getattr(train_cfg, k) for k in CHECKPOINT_OWNED}})
    conflicts = [k for k in CHECKPOINT_OWNED if getattr(cfg, k) != getattr(train_cfg, k)]
    if conflicts:
        print(f"[predict] using the checkpoint's {conflicts} over the config's -- "
              f"inference must match training.", flush=True)

    if cfg.act_filter != train_cfg.act_filter:
        # Row SELECTION, unlike the shaping keys above, is left to the caller:
        # scoring an act-trained model on every row is a fair experiment. Say so
        # rather than silently changing the population.
        print(f"[predict] note: trained with act_filter={train_cfg.act_filter}, "
              f"predicting with act_filter={cfg.act_filter}", flush=True)
    df = load_frame(load_cfg, require_target=False, require_split=False)
    if model_set:
        df = df[df["model_set"] == model_set].reset_index(drop=True)
    if ids:
        df = df[df["id"].isin(ids)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("[predict] no rows selected")

    out = _predict(model, train_cfg, scaler, target_tf, df, get_tokenizer(train_cfg),
                   cfg.batch_size, dev)

    if "target" in out.columns and out["target"].notna().all():
        from model_shay.train import compute_metrics
        scored = compute_metrics(train_cfg, out["target"].to_numpy(),
                                 out["prediction"].to_numpy())
        detail = (f"acc={scored['acc']*100:.1f}% f1={scored['f1_macro']:.3f} "
                  f"within1={scored['within_1']*100:.1f}%"
                  if train_cfg.task == "classification" else
                  f"mae={scored['mae_bps']:.1f}bps mse={scored['mse']:.3e} "
                  f"r2={scored['r2']:+.4f}")
        print(f"[predict] n={len(out)} {detail}", flush=True)
    else:
        print(f"[predict] n={len(out)} (no targets to score against)", flush=True)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"[predict] wrote {out_path}", flush=True)
    else:
        print(out.head(20).to_string(index=False))
    return out
