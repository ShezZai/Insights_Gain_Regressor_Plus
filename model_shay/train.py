"""Training loop, early stopping and k-fold CV (spec §4, §5, §6, §8).

One loop serves both trainable modes; `model.prepare()` is the only thing that
differs. `partial_freeze` optionally runs a second phase over the whole encoder
with discriminative LRs, sharing the same epoch machinery.

Metrics are always reported in RETURN UNITS -- predictions are inverse-
transformed out of training space first -- so a Huber run, an MSE run and the
§7 baselines are directly comparable. `mae_bps` is the same number in basis
points, which is the readable one given sigma is ~117 bps.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_shay import model as M
from model_shay.config import Config
from model_shay.data import (FLOAT_COLUMNS, FloatScaler, TargetTransform, build_dataset,
                             get_tokenizer, length_report, load_frame, split_frames)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_precision(cfg: Config, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """(autocast dtype, whether a GradScaler is needed). fp32 -> (None, False)."""
    if device.type != "cuda" or cfg.precision == "fp32":
        return None, False
    if cfg.precision == "bf16":
        return torch.bfloat16, False
    if cfg.precision == "fp16":
        return torch.float16, True
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE/MAE/RMSE/R2 in return units, plus MAE in bps for readability."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse, mae = float(np.mean(err ** 2)), float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae,
            "mae_bps": mae * 1e4, "r2": (1.0 - mse / var) if var > 0 else float("nan")}


# --------------------------------------------------------------------------- #
# Metrics where BIGGER is better, so the tracker compares their negation.
HIGHER_BETTER = {"val_acc", "val_f1"}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1, and the ordinal-friendly within-one-class rate."""
    from sklearn.metrics import f1_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"acc": float((y_true == y_pred).mean()),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "within_1": float((np.abs(y_true - y_pred) <= 1).mean())}


def compute_metrics(cfg: Config, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return (classification_metrics(y_true, y_pred) if cfg.task == "classification"
            else regression_metrics(y_true, y_pred))


class BestTracker:
    """Early stopping (§6): lowest `monitor` wins, best weights kept in RAM."""

    def __init__(self, patience: int, monitor: str):
        self.patience, self.monitor = patience, monitor
        self.best = math.inf
        self.best_epoch = -1
        self.best_state: dict[str, torch.Tensor] | None = None
        self.stale = 0

    def update(self, epoch: int, metrics: dict[str, float], model: torch.nn.Module) -> bool:
        value = metrics[self.monitor]
        if self.monitor in HIGHER_BETTER:
            value = -value
        if value < self.best - 1e-12:
            self.best, self.best_epoch, self.stale = value, epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.stale >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _optimizer(groups: list[dict], cfg: Config, model: torch.nn.Module):
    """AdamW with no weight decay on biases and norms (ndim <= 1)."""
    no_decay = {id(p) for p in model.parameters() if p.ndim <= 1}
    split: list[dict] = []
    for g in groups:
        params = [p for p in g["params"] if p.requires_grad]
        decay = [p for p in params if id(p) not in no_decay]
        plain = [p for p in params if id(p) in no_decay]
        if decay:
            split.append({"params": decay, "lr": g["lr"], "weight_decay": cfg.weight_decay})
        if plain:
            split.append({"params": plain, "lr": g["lr"], "weight_decay": 0.0})
    return torch.optim.AdamW(split)


@torch.no_grad()
def evaluate(model, loader, loss_fn, target_tf: TargetTransform, device,
             amp_dtype, cfg: Config) -> tuple[float, dict[str, float], np.ndarray]:
    """Loss in training space; metrics in the target's own units (returns, or
    1..5 sentiment classes)."""
    model.eval()
    total, n = 0.0, 0
    preds, trues = [], []
    for input_ids, mask, floats, y in loader:
        input_ids, mask = input_ids.to(device), mask.to(device)
        floats, y = floats.to(device), y.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids, mask, floats)
            loss = loss_fn(out.float(), y)
        total += float(loss) * y.size(0)
        n += y.size(0)
        if cfg.task == "classification":
            preds.append(out.float().argmax(-1).cpu().numpy())
            trues.append(y.cpu().numpy())
        else:
            preds.append(out.float().cpu().numpy())
            trues.append(y.float().cpu().numpy())
    pred_out = target_tf.inverse(np.concatenate(preds))
    true_out = target_tf.inverse(np.concatenate(trues))
    return total / max(n, 1), compute_metrics(cfg, true_out, pred_out), pred_out


def run_phase(*, model, cfg: Config, train_loader, val_loader, loss_fn, target_tf,
              device, groups: list[dict], n_epochs: int, tracker: BestTracker,
              phase: str, amp_dtype, use_scaler: bool, epoch_offset: int = 0) -> int:
    """Train `n_epochs`, early-stopping through `tracker`. Returns epochs run."""
    if n_epochs <= 0:
        return 0
    optimizer = _optimizer(groups, cfg, model)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * n_epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ran = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        started, running, seen = time.monotonic(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (input_ids, mask, floats, y) in enumerate(train_loader, start=1):
            input_ids, mask = input_ids.to(device), mask.to(device)
            floats, y = floats.to(device), y.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(input_ids, mask, floats).float(), y)
            if not torch.isfinite(loss):
                # Bail on the first bad step instead of burning epochs on NaN:
                # a whole run of `train_loss=nan` reports nothing about the model.
                raise SystemExit(
                    f"[{phase}] non-finite loss at epoch {epoch}, step {step}. "
                    f"Check that the encoder holds fp32 weights "
                    f"(dtype={next(model.encoder.parameters()).dtype}) and try "
                    f"--precision fp32, or a lower --head-lr / --encoder-lr.")
            scaler.scale(loss / cfg.grad_accum).backward()
            running += float(loss) * y.size(0)
            seen += y.size(0)
            if step % cfg.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_loss, metrics, _ = evaluate(model, val_loader, loss_fn, target_tf,
                                        device, amp_dtype, cfg)
        if cfg.task == "classification":
            row = {"val_loss": val_loss, "val_acc": metrics["acc"],
                   "val_f1": metrics["f1_macro"]}
            detail = (f"acc={metrics['acc']*100:.1f}% f1={metrics['f1_macro']:.3f} "
                      f"within1={metrics['within_1']*100:.1f}%")
        else:
            row = {"val_loss": val_loss, "val_mae": metrics["mae"], "val_mse": metrics["mse"]}
            detail = (f"mae={metrics['mae_bps']:.1f}bps mse={metrics['mse']:.3e} "
                      f"r2={metrics['r2']:+.4f}")
        global_epoch = epoch_offset + epoch
        improved = tracker.update(global_epoch, row, model)
        print(f"[{phase} e{global_epoch:02d}] train_loss={running / max(seen,1):.5f} "
              f"val_loss={val_loss:.5f} {detail} "
              f"({time.monotonic() - started:.0f}s){' *' if improved else ''}", flush=True)
        ran += 1
        if device.type == "cuda" and epoch == 1:
            print(f"[{phase}] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()
        if tracker.should_stop:
            print(f"[{phase}] early stop: no improvement in {cfg.patience} epoch(s); "
                  f"best {cfg.monitor}={tracker.best:.5f} @ e{tracker.best_epoch}", flush=True)
            break
    return ran


def train_one(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer,
              out_dir: Path, tag: str) -> dict[str, Any]:
    """Fit one model on one (train, val) pair and save its best checkpoint."""
    set_seed(cfg.seed)
    device = pick_device()
    amp_dtype, use_scaler = pick_precision(cfg, device)

    # Fit BOTH transforms on the training fold only -- no leakage (§1).
    scaler = FloatScaler.fit(train_df[FLOAT_COLUMNS].to_numpy(), cfg.log_transform_floats)
    # Classification maps 1..5 -> 0..4; target_transform is a regression concept.
    tf_kind = "classes" if cfg.task == "classification" else cfg.target_transform
    target_tf = TargetTransform.fit(train_df["target"].to_numpy(), tf_kind)

    train_ds = build_dataset(train_df, tokenizer, cfg, scaler, target_tf)
    val_ds = build_dataset(val_df, tokenizer, cfg, scaler, target_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = M.TextFloatsRegressor(cfg).to(device)
    M.prepare(model, cfg)
    if cfg.gradient_checkpointing:
        # use_reentrant=False is required: the reentrant variant errors when
        # part of the encoder is frozen, which is exactly phase 1.
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[memory] gradient checkpointing on", flush=True)
    loss_fn = M.build_loss(cfg)
    tracker = BestTracker(cfg.patience, cfg.monitor)

    print(f"[{tag}] train={len(train_df)} val={len(val_df)} device={device} "
          f"precision={'fp32' if amp_dtype is None else str(amp_dtype).split('.')[-1]} "
          f"loss={'cross_entropy' if cfg.task == 'classification' else cfg.loss_type} "
          f"mode={cfg.train_mode}", flush=True)

    # Phase 1 (or the single LoRA phase). Only partial_freeze + phase 2 caps
    # phase 1 at phase1_epochs; otherwise this phase owns the whole budget.
    two_phase = cfg.train_mode == "partial_freeze" and cfg.unfreeze_all_phase2
    phase1_epochs = min(cfg.phase1_epochs, cfg.max_epochs) if two_phase else cfg.max_epochs
    done = run_phase(model=model, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
                     loss_fn=loss_fn, target_tf=target_tf, device=device,
                     groups=M.head_and_encoder_groups(model, cfg), n_epochs=phase1_epochs,
                     tracker=tracker, phase=f"{tag}/p1", amp_dtype=amp_dtype,
                     use_scaler=use_scaler)

    if two_phase and not tracker.should_stop:
        tracker.stale = 0   # phase 2 changes the regime; give it its own patience
        run_phase(model=model, cfg=cfg, train_loader=train_loader, val_loader=val_loader,
                  loss_fn=loss_fn, target_tf=target_tf, device=device,
                  groups=M.discriminative_groups(model, cfg),
                  n_epochs=cfg.max_epochs - done, tracker=tracker, phase=f"{tag}/p2",
                  amp_dtype=amp_dtype, use_scaler=use_scaler, epoch_offset=done)

    tracker.restore(model)
    val_loss, metrics, _ = evaluate(model, val_loader, loss_fn, target_tf,
                                    device, amp_dtype, cfg)
    metrics = {**metrics, "val_loss": val_loss, "best_epoch": tracker.best_epoch,
               "n_train": len(train_df), "n_val": len(val_df)}
    ckpt = M.save_checkpoint(out_dir / tag / "best.pt", model, cfg, scaler, target_tf,
                             metrics=metrics, extra={"tag": tag})
    (out_dir / tag / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    best = -tracker.best if cfg.monitor in HIGHER_BETTER else tracker.best
    summary = (f"acc={metrics['acc']*100:.1f}% f1={metrics['f1_macro']:.3f}"
               if cfg.task == "classification"
               else f"mae={metrics['mae_bps']:.1f}bps r2={metrics['r2']:+.4f}")
    print(f"[{tag}] best {cfg.monitor}={best:.5f} @ e{tracker.best_epoch} :: {summary}",
          flush=True)
    return {**metrics, "checkpoint": str(ckpt), "tag": tag}


# --------------------------------------------------------------------------- #
def _prepare_run(cfg: Config, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.json")          # §8: config beside the checkpoints
    df = load_frame(cfg)
    tokenizer = get_tokenizer(cfg)
    report = length_report(df["text"].tolist(), tokenizer, cfg.max_length)
    (out_dir / "length_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return df, tokenizer


def run_single(cfg: Config, out_dir: Path, eval_test: bool = False) -> dict[str, Any]:
    """Standard train/val run straight off the model_set labels (§5, use_kfold: false)."""
    df, tokenizer = _prepare_run(cfg, out_dir)
    splits = split_frames(df)
    if splits["training"].empty or splits["validation"].empty:
        raise SystemExit("[single] training or validation split is empty")

    result = train_one(cfg, splits["training"], splits["validation"], tokenizer, out_dir, "single")
    summary = {"mode": "single", "config": cfg.as_dict(), "validation": result}

    if eval_test and not splits["test"].empty:
        summary["test"] = _score_split(cfg, result["checkpoint"], splits["test"], tokenizer)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_kfold(cfg: Config, out_dir: Path, eval_test: bool = False,
              final_fit: bool = False) -> dict[str, Any]:
    """GroupKFold CV over the pooled training+validation rows (§5).

    Grouping is mandatory, not optional: 46% of the corpus shares a
    (ticker, trading-day) with another row, so ungrouped folds would split
    near-duplicate targets across the boundary and inflate every metric.
    """
    from sklearn.model_selection import GroupKFold

    df, tokenizer = _prepare_run(cfg, out_dir)
    splits = split_frames(df)
    pool = pd.concat([splits["training"], splits["validation"]], ignore_index=True)
    groups = pool["group"].to_numpy()
    n_groups = len(set(groups))
    if n_groups < cfg.k_folds:
        raise SystemExit(f"[kfold] {n_groups} group(s) < k_folds={cfg.k_folds}")
    print(f"[kfold] pool={len(pool)} rows over {n_groups} group(s), k={cfg.k_folds}", flush=True)

    per_fold: list[dict] = []
    splitter = GroupKFold(n_splits=cfg.k_folds)
    for fold, (tr, va) in enumerate(splitter.split(pool, groups=groups), start=1):
        per_fold.append(train_one(cfg, pool.iloc[tr].reset_index(drop=True),
                                  pool.iloc[va].reset_index(drop=True),
                                  tokenizer, out_dir, f"fold_{fold}"))

    keys = (["val_loss", "acc", "f1_macro", "within_1"] if cfg.task == "classification"
            else ["val_loss", "mse", "rmse", "mae", "mae_bps", "r2"])
    aggregate = {k: {"mean": float(np.mean([f[k] for f in per_fold])),
                     "std": float(np.std([f[k] for f in per_fold]))} for k in keys}
    summary: dict[str, Any] = {"mode": "kfold", "k_folds": cfg.k_folds,
                               "config": cfg.as_dict(), "folds": per_fold,
                               "aggregate": aggregate}

    cols = (("val_loss", "acc", "f1_macro", "within_1") if cfg.task == "classification"
            else ("val_loss", "mae_bps", "mse", "r2"))
    print("\n[kfold] per-fold results")
    print("  " + f"{'fold':<9}" + "".join(f"{c:>12}" for c in cols))
    for f in per_fold:
        print("  " + f"{f['tag']:<9}" + "".join(f"{f[c]:>12.5f}" for c in cols))
    print("  " + f"{'mean':<9}" + "".join(f"{aggregate[c]['mean']:>12.5f}" for c in cols))
    print("  " + f"{'sd':<9}" + "".join(f"{aggregate[c]['std']:>12.5f}" for c in cols) + "\n",
          flush=True)

    pd.DataFrame(per_fold).to_csv(out_dir / "folds.csv", index=False)

    if final_fit:
        # §5: a deployment fit on the whole pool. NO held-out validation set --
        # its "val" numbers below are in-sample and must not be read as CV.
        print("[final] retraining on the FULL pool -- this model has NO held-out "
              "validation set; its metrics are in-sample.", flush=True)
        summary["final"] = train_one(cfg, pool, pool, tokenizer, out_dir, "final")
        summary["final"]["warning"] = "in-sample: trained and scored on the same pool"
        if eval_test and not splits["test"].empty:
            summary["test"] = _score_split(cfg, summary["final"]["checkpoint"],
                                           splits["test"], tokenizer)
    elif eval_test:
        print("[test] skipped: --eval-test needs --final-fit in k-fold mode "
              "(there is no single model to score).", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _score_split(cfg: Config, checkpoint: str, df: pd.DataFrame, tokenizer) -> dict[str, Any]:
    """Score the untouched holdout, once, at the very end (§1)."""
    from model_shay.predict import predict_frame

    print(f"[test] scoring {len(df)} held-out row(s) -- final holdout, one look.", flush=True)
    out = predict_frame(checkpoint, df, tokenizer=tokenizer)
    metrics = compute_metrics(cfg, df["target"].to_numpy(), out["prediction"].to_numpy())
    unseen = df["test_unseen"] if "test_unseen" in df.columns else None
    result: dict[str, Any] = {"n": len(df), **metrics}
    if unseen is not None and unseen.any():
        mask = unseen.fillna(False).to_numpy().astype(bool)
        result["test_unseen"] = {"n": int(mask.sum()),
                                 **compute_metrics(cfg, df["target"].to_numpy()[mask],
                                                   out["prediction"].to_numpy()[mask])}
    print("[test] " + (f"acc={metrics['acc']*100:.1f}% f1={metrics['f1_macro']:.3f} "
                       f"within1={metrics['within_1']*100:.1f}%"
                       if cfg.task == "classification" else
                       f"mae={metrics['mae_bps']:.1f}bps mse={metrics['mse']:.3e} "
                       f"r2={metrics['r2']:+.4f}"), flush=True)
    return result
