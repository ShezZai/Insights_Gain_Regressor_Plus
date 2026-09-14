# Text + Market Floats Regression Model — Build Spec

## Objective
Build a PyTorch/HuggingFace training pipeline that predicts a single continuous
target from:
- a text field (a distilled insight block, median ~156 words / ~200 tokens,
  padded to a flat 512 tokens)
- 3 numeric market float features

Train on ~3.1k examples, validate on ~1.1k, with support for **k-fold
cross-validation** and a **configurable loss (MSE or Huber)**.

Target hardware: single local GPU, 16GB VRAM.

---

## 1. Data

### Source

The data is not a standalone file — it is `public.articles` in the
`news_trading_window` Postgres DB, already filtered to NYSE trading days
between the open and 90 minutes before the close (see
`scripts/enrichment/build_nyse_trading_window.sql`, which is why the
+90-minute target window always lands inside the regular session). The
pipeline reads it directly (`ticker_news.shared.db.connect`) or from a
CSV/Parquet export of the same query.

Column mapping:

| spec field | `public.articles` column |
|---|---|
| `text`    | `consolidated_insights` |
| `float_1` | `vix_at_article_time` |
| `float_2` | `intraday_gain_till_article` |
| `float_3` | `aiq_ai_etf_intraday_gain_till_article` |
| `target`  | `gain_90m_after_article` |

- **Text is `consolidated_insights`**, built by
  `build_consolidated_insights.sql` from raw `public.article_insights`. Not
  `distilled_article_insights` — that table's second-pass DROP gate is
  deliberately not applied here.
- `float_1` is VIXY (the short-term VIX futures ETF, standing in for the VIX
  index Massive does not serve) as a **raw price level**, used as-is. Note for
  the record that VIXY decays, so its level drifts down over the corpus
  (quarterly mean 48.8 → 19.4 from 2024Q3 to 2026Q3) and partly encodes
  calendar time; this is accepted, not corrected.
- `float_2` / `target` have `_first` variants measured on
  `first_mentioned_ticker` instead of `primary_ticker`. Pick one pair with a
  config flag and never mix them — they disagree on ~700 rows.

### Split

- Use the existing `articles.model_set` labels (`training` / `validation` /
  `test`), written by `assign_model_set.py`: 3,173 / 1,096 / 1,237. That split
  is already ticker-balanced and reserves the most recent trading days
  wholesale as a temporal holdout (`test_unseen = true` on 885 of the test
  rows). Do not re-derive a random split.
- `test` is a final holdout — untouched by training, validation, and k-fold CV.
- For k-fold, merge `training` + `validation` into one pool; see §5.

### Row filter

Only rows with all five fields non-NULL are usable. On top of that:

- **Drop the rows flagged `insights_too_long_more_than_500`** (81 rows, >500
  cl100k tokens). They are excluded rather than truncated, so no example loses
  content to the 512-token ceiling.
- Log the eligible/excluded counts at startup.

### Text length

Measured over the 5,506 rows with insights: median 156 words, p95 302, max 793,
at ~1.28 tokens per word. With the 81 long rows dropped, everything remaining
fits inside 512 tokens.

- `max_length=512` (config, adjustable).
- **Fixed padding: `padding="max_length"`** — every batch is a flat
  512-token block. Not `DataCollatorWithPadding`. This trades the compute
  saving for deterministic, worst-case-equals-average memory, which makes the
  16GB budget in §10 a fixed number rather than a per-batch gamble.
- Still log the tokenized length distribution once at startup
  (min/median/p95/max) plus the count of examples hitting `max_length`. After
  the row filter that count should be 0 — a non-zero value means the filter or
  the flag column is stale, so warn loudly.

### Preprocessing

- Standardize the 3 floats (fit scaler on train fold only, apply to val fold — no leakage).
- Optionally log-transform floats/target if skewed (flag as config option, default off).
- Tokenize with the chosen encoder's tokenizer, `max_length=512`,
  `truncation=True` (a backstop only, given the row filter),
  `padding="max_length"`.

## 2. Model architecture

```
text_ids/attention_mask -> encoder (DeBERTa-v3-base) -> pooled_output (CLS or mean pooling)
float_1, float_2, float_3 -> normalized -> concat directly (no MLP needed at this size)
concat([pooled_output, normalized_floats]) -> Dense(hidden) -> Dropout -> Dense(1) -> scalar output
```

- Backbone: `microsoft/deberta-v3-base` (default). Make this a config string so it's swappable.
- Pooling: mean pooling over last hidden state (attention-mask weighted), not just CLS — expose as config option (`cls` | `mean`).
- Regression head: `Linear(hidden_size + 3, 128) -> GELU -> Dropout(0.2-0.3) -> Linear(128, 1)`.
- Dropout in head: configurable, default 0.25.

## 3. Loss function (configurable)

Support a `loss_type` config flag: `"mse"` or `"huber"`.

- MSE: `torch.nn.MSELoss()`
- Huber: `torch.nn.HuberLoss(delta=...)` — expose `delta` as a config value (default 1.0).
- Loss selection should be a single factory function so switching requires no other code changes.
- Report both MSE and MAE as evaluation metrics regardless of which loss is used for optimization, so results are comparable across the two settings.

## 4. Training strategy (configurable)

Support two trainable modes via a config flag `train_mode`: `"partial_freeze"` or `"lora"`.

### partial_freeze (default)
- Freeze all encoder layers except the last N (config, default N=3) + pooler.
- Phase 1: train head + unfrozen layers, `head_lr=1e-3`, `encoder_lr=2e-5`, few epochs (config, default 4).
- Phase 2 (optional, config flag `unfreeze_all_phase2`): unfreeze full encoder, discriminative LR (e.g. `5e-6` early layers → `2e-5` late layers), continue training with early stopping.

### lora
- Apply LoRA adapters to attention query/value projections (use `peft` library).
- Config: `r=8, lora_alpha=16, lora_dropout=0.1` (all configurable).
- Regression head + LoRA params trainable; base weights frozen.
- Same LR schedule structure as above but single phase (no need for phase 2 since LoRA is already regularized).

Both modes should share the same training loop/trainer code — only the parameter-freezing/adapter-injection step differs.

## 5. K-fold cross-validation

- Config flag: `use_kfold: true/false`.
- If true:
  - Merge the `training` + `validation` rows into a single pool (~4.2k before
    the completeness/length filters, less after). `test` stays out.
  - **Use `sklearn.model_selection.GroupKFold`, grouping on
    `(ticker, ET trading date)`. This is mandatory, not a TODO.** 2,529 of the
    5,506 insight-bearing rows (46%) share a ticker-day with at least one other
    row, and the largest such cluster holds 30 articles. Two articles on the
    same name 20 minutes apart share 70 of their 90 forward minutes, so their
    targets are near-identical — plain `KFold` would put them on both sides of
    the fold boundary and inflate every metric.
  - `k=4` or `k=5` (config, default 5).
  - For each fold: fit float scaler on fold-train only, train model from scratch (fresh encoder weights each fold — do not carry over weights between folds), evaluate on fold-val.
  - Aggregate metrics (mean ± std of MSE/MAE/Huber loss across folds) and report per-fold results in a summary table/JSON.
  - After CV, optionally retrain a final model on the whole pool using the best config found, for deployment. This final model has no held-out validation set — note this clearly in output/logs. Evaluate it once on the untouched `test` split.
- If false: single split straight from `model_set` (`training` vs `validation`), with early stopping on val loss.

## 6. Early stopping & checkpointing

- Monitor validation loss (or MAE) each epoch.
- `patience` config (default 3 epochs of no improvement).
- Save best checkpoint per fold (if k-fold) or per run (if single split).
- Max epochs config (default 20) — rely on early stopping, not fixed epoch count.

## 7. Baselines (for sanity-checking, run once, not part of main loop)

- Floats-only baseline: linear regression or gradient boosting on the 3 floats alone.
- Text-only baseline: TF-IDF + linear regression on text alone.
- Log both baseline metrics alongside the neural model's metrics so the incremental value of combining both signals is visible.
- Treat these as load-bearing results, not a formality. The target is
  90-minute forward return: σ = 1.17%, mean −0.008%, p1/p99 = −3.6%/+3.1%. On
  ~3k examples a DeBERTa regressor may well land at R² ≈ 0, and the honest
  read of that outcome depends entirely on knowing where the two trivial
  baselines sit. Report the constant-prediction (predict-the-mean) MSE/MAE as a
  third reference line.

## 8. Outputs / logging

- Log per-epoch train/val loss, MAE, MSE.
- Save config used for each run (JSON) alongside checkpoints for reproducibility.
- If k-fold: save per-fold metrics + aggregate summary (mean/std) as a JSON or CSV table.

## 9. Config file (all of the above should be driven by one YAML/JSON config)

Example keys to include:
```yaml
# data
text_column: "consolidated_insights"
ticker_variant: "primary"      # "primary" | "first" -- picks float_2/target pair
exclude_long_insights: true    # drop the 81 insights_too_long_more_than_500 rows
padding: "max_length"          # flat 512-token batches; not dynamic padding
group_by: ["ticker", "et_date"]  # GroupKFold key -- see §5

encoder_name: "microsoft/deberta-v3-base"
max_length: 512   # every retained row fits; truncation=True is a backstop only
pooling: "mean"          # "cls" | "mean"
head_hidden_dim: 128
head_dropout: 0.25
loss_type: "huber"        # "mse" | "huber"
huber_delta: 1.0
train_mode: "partial_freeze"  # "partial_freeze" | "lora"
freeze_except_last_n: 3
lora_r: 8
lora_alpha: 16
lora_dropout: 0.1
head_lr: 1e-3
encoder_lr: 2e-5
unfreeze_all_phase2: true
batch_size: 16
max_epochs: 20
patience: 3
use_kfold: true
k_folds: 5           # GroupKFold, grouped on group_by
seed: 42
```

## 10. Hardware notes
- Target: RTX 5060 Ti, 16GB VRAM (Blackwell / sm_120 — needs a cu128 torch
  build; see the `[model]` extra in `pyproject.toml`).
- Because §1 pads to a flat 512, every batch is the worst case by
  construction — memory use is constant across steps, so a batch size that
  survives the first step survives the whole run.
- Phase 2 (§4) is the memory peak, not phase 1: unfreezing all 12 layers
  retains every layer's activations, and DeBERTa materializes a
  `batch × heads × 512 × 512` attention tensor per layer. Measured peak
  allocation in phase 2 (bf16, RTX 5060 Ti, ~13.1 GiB usable after the
  display's 2.4 GiB):

  | batch | checkpointing off | on | samples/s (on) |
  |---|---|---|---|
  | 16 | 12.6 GiB (no headroom) | 3.6 GiB | 20.2 |
  | 32 | OOM | 5.0 GiB | 20.5 |
  | 64 | — | 7.8 GiB | 23.7 |

  So `gradient_checkpointing` defaults to **on**: it is what makes the 16GB
  target hold, and the recompute is cheap enough that a larger batch more than
  pays for it. Batch is still a result-affecting knob — hold it fixed across
  any loss/train_mode comparison.
- Use mixed precision training (`torch.cuda.amp` or HF `Trainer(fp16=True)`).
- Gradient accumulation as a config option if larger effective batch size is desired.
