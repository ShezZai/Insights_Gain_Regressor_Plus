# tries.md — what was tried, and why the conclusion is negative

**Conclusion: the AI-compute news corpus and its distilled insights do not support a
profitable model. Not for forecasting the move after an article, and not even for
describing the move that already happened in the 30 minutes before it.**

This is not a tuning failure. Across 6 targets, 8 model classes, a 50× range of
trainable capacity, two learning rates and two measurement windows, every
forward-looking configuration landed at R² ≈ 0 with predictions collapsed to a
constant. The single measurable signal anywhere in the project is a weak
*hindsight* correlation (~0.13–0.17) that does not convert into positive R² and is
substantially day-level market state rather than article-level information.

---

## 1. The data

`news_trading_window` (Postgres), built by `src/ticker_news/` + `scripts/enrichment/`.

| | |
|---|---|
| articles | 6,333 (5,506 with distilled insights) |
| insight boxes | 25,000 (`article_insights`) |
| universe | 118 tickers (AI/compute segments) |
| span | 2024-09-10 → 2026-09-08, 497 NYSE trading days |
| filter | trading days only, between the open and 90 min before the close |
| text | `consolidated_insights` — median 196 tokens, p95 357, max 502 |
| split | `model_set` 60/20/20, ticker-balanced, trailing-date holdout (`test_unseen`) |

Model input, per `model_shay`: the insight text (DeBERTa-v3-base, flat 512 padding)
plus a dense vector of up to 10 dims — 3 market floats, a 5-dim `sentiment` one-hot,
a 2-dim `is_act_from_insights` one-hot.

---

## 2. Guardrails — why these nulls are trustworthy

Negative results are only worth as much as the methodology behind them.

- **GroupKFold on `(ticker, trading-day)` is mandatory, not optional.** 2,529 of
  5,506 rows (46%) share a ticker-day with another row; the largest cluster holds
  30 articles. Two pieces on the same name minutes apart share most of their
  forward window, so plain `KFold` would straddle near-duplicate targets across the
  boundary and inflate every metric. All CV numbers below are grouped.
- **Scalers and target transforms are fitted on the training fold only.**
- **The `test` split was never trained on**, and `test_unseen` (dates absent from
  train/val entirely) is reported separately throughout.
- **Leakage guards are enforced in code.** `target_kind=intraday_gain_till` with
  floats enabled is *refused*, because `float_2` is that column. The `--30m_till`
  target was chosen as the −30m→article segment precisely because the alternative
  (open→article) correlates **0.9526** with an input float, versus **0.0057** for
  the segment.
- **One shared metric module** (`model_shay/metrics.py`) serves the neural and tree
  models, so their numbers are directly comparable.

---

## 3. Forward-looking targets — everything is zero

Target sd is ~114–136 bps, so **compare R², never MAE in bps** across rows.

### 3.1 Baselines (GroupKFold, the line every model must beat)

| baseline | at_article, R² | 30m window, R² |
|---|---|---|
| constant (predict the mean) | **−0.0029** | **−0.0049** |
| floats — linear | −0.0152 | −0.0097 |
| floats — gradient boosting | −0.1358 | −0.0993 |
| text — TF-IDF + ridge | −0.0726 | −0.0689 |

**No feature-using model beats the constant.** TF-IDF and GBDT sit far *below* it —
the signature of fitting noise.

### 3.2 DeBERTa-v3-base, 5-fold GroupKFold, `gain_90m_after_article_first`

| run | loss | per-fold R² | mean |
|---|---|---|---|
| `cv-huber` | Huber | −0.0007, −0.0029, −0.0171, −0.0002, −0.0020 | **−0.0046 ± 0.0063** |
| `cv-mse` | MSE | −0.0112, −0.0014, −0.0076, −0.0051, −0.0099 | **−0.0070 ± 0.0035** |

Ten folds, ten negative. Prediction spread was **2.4%** of target spread — the model
emits a near-constant. RMSE equals the target's own standard deviation.

### 3.3 The 30-minute window (−30m → +90m), single split, 1,127 test rows

| run | trainable params | test R² | corr | pred spread | best epoch |
|---|---|---|---|---|---|
| `30m-single` (freeze last 3) | 21.4M | −0.0015 | −0.0152 | 2.6% | 2 |
| `30m-freeze1` (freeze last 1) | 7.3M | −0.0025 | −0.0315 | 2.7% | 2 |
| `30m-lora` (r=8) | 0.4M | −0.0086 | −0.0606 | 2.8% | 5 |
| `30m-sent` (+ sentiment one-hot) | 21.4M | −0.0060 | — | — | 3 |
| `atart-act` (actionable subset) | 21.4M | −0.0110 | — | — | 2 |

**Trainable parameters vary 50×. Prediction spread varies 2.6% → 2.8%.**

Sign agreement on `30m-single`: **51.5%**, *below* the 52.6% majority-class rate
(binomial p = 0.79). Because the model outputs a near-constant, 90.7% of its
predictions are positive and "sign agreement" merely re-reads the base rate. The
`30m-lora` run settled on a negative constant instead and scored 46.2% — same
phenomenon, opposite side. Confidence deciles show no trend.

Adding the distilled `sentiment` and `is_act_from_insights` one-hots as inputs
changed nothing (`30m-sent`, test R² −0.0060).

---

## 4. Hindsight targets — weak, and it does not survive

If the text cannot predict the future, can it at least describe the *past*? This is
the positive control, and the answer is "barely, and not usefully."

### 4.1 At-article hindsight (open → article), text-only

`control-gain-till`: val R² **+0.0109**, test R² **+0.0009**, prediction spread
10.2%, corr +0.074. Statistically alive, economically nothing.

For reference, a crude up/down keyword counter on the same text achieves corr
**+0.082** — separating +31 bps (net-positive wording) from −47 bps (net-negative).
The transformer matched a regex.

TF-IDF on the realized move, by text source:

| source | R² | corr |
|---|---|---|
| `consolidated_insights` | −0.0279 | +0.1001 |
| `title` only | −0.0757 | +0.1142 |
| title + insights | −0.0208 | +0.1204 |

The distillation did not hide the signal — the ceiling is low everywhere.

### 4.2 Long hindsight (open → −30m), text-only — the one real signal

`30m-till`: test R² **+0.0508**, corr **+0.2315**, prediction spread **27.3%**,
best epoch 7. This is the only run in the project that genuinely committed.

But it decomposes badly:

| | n | R² | corr |
|---|---|---|---|
| seen dates | 312 | **+0.1541** | +0.4107 |
| unseen dates | 815 | **−0.0056** | +0.1271 |

Nearly all the R² comes from test rows whose *calendar day* appears in training —
the model learns "that morning was strong" from other articles on the same date and
applies it to a different ticker. This is **day-level market state, not article-level
information**. It is not row duplication: only 110 rows (9.8%) share a ticker-day
with training, and those scored *worse* (+0.0115) than the clean rows (+0.0530).

What survives on genuinely unseen dates is real but small: corr +0.1271 at row level,
**+0.1678 aggregated to ticker-day means (t = 3.94, n = 537 groups)**. Statistically
solid, R² still negative. The model ranks slightly better than chance and its
magnitudes do not generalize across days.

### 4.3 30-minute hindsight (−30m → article) — null

The sharpest version of the question: predict the move over the last 30 minutes
before publication, from the text plus the market state up to that point. Input and
target do not overlap (corr 0.0057), so the floats are legitimate features.

`30m_till-sent`, DeBERTa + 10 dense features, 5-fold GroupKFold:

```
folds: -0.0002, -0.0089, -0.0056, -0.0057, +0.0029
mean:  -0.0035 +- 0.0042
```

**Nothing.** Strip the multi-hour window that let the model infer the day's market
direction, and the remaining article-level signal is indistinguishable from zero.
This is the result that closes the question: the insights do not even describe the
move that had just happened when they were published.

---

## 5. Trees on the dense features alone (`model_tree`)

GroupKFold, 10 features (3 floats + 5 sentiment + 2 act), HistGradientBoosting:

| target | features | R² | constant ref |
|---|---|---|---|
| forward (−30m → +90m) | 10 | **−0.1441 ± 0.0416** | −0.0049 |
| forward, labels only | 7 | **−0.0088 ± 0.0053** | −0.0049 |
| hindsight segment | 10 | **−0.1754 ± 0.0853** | −0.0027 |

Test R² on the first: **−0.2616**. Permutation importance on held-out folds:

```
aiq_ai_etf_..._30m_before   +0.0905     <- market-wide, identical per day
vix_..._30m_before          +0.0630     <- market-wide, identical per day
intraday_gain_till_30m      +0.0064
act_false                   +0.0049
sentiment_4                 +0.0037
act_true                    -0.0001
sentiment_2                 -0.0009
sentiment_3                 -0.0051
sentiment_1                 -0.0058
sentiment_5                 -0.0069
```

Because R² is negative, this measures *reliance*, not usefulness: the tree leans on
the two market-wide series and that is exactly why it loses to a constant — it is
memorising days. **Four of five `sentiment` dimensions have negative importance:
shuffling them improves held-out performance.** The label-only run (R² −0.0088 vs
constant −0.0049) is the clean measure of what the distilled labels contribute to
the forward return: nothing.

---

## 6. Ruling out the alternative explanations

Every "maybe it's the setup" hypothesis was tested and eliminated.

| hypothesis | test | result |
|---|---|---|
| **Broken training** | fp16 weight bug (transformers ≥5 `dtype="auto"` loaded fp16 params → NaN) | found and fixed; all later runs finite |
| **Underfitting / LR too low** | LR probe, 12 epochs on a small subset | train loss 0.388 → 0.192 while val rose 0.307 → 0.349. At 5× LR: train 0.495 → 0.150, val → 0.435. Memorises freely |
| **Too much capacity** | LoRA 0.4M → freeze-1 7.3M → freeze-3 21.4M | flat: R² −0.009 to −0.002, spread 2.6–2.8% |
| **Needs deeper fine-tuning** | phase 2, full 12-layer unfreeze | worse in every run (control: +0.011 → −0.001 → −0.062 while train loss fell) |
| **Wrong loss** | Huber vs MSE, 5-fold | −0.0046 vs −0.0070, both null |
| **Wrong ticker attribution** | `primary_ticker` vs `first_mentioned_ticker` | both null |
| **Wrong window** | at_article vs −30m | both null |
| **Too much noise in the corpus** | restrict to `is_act_from_insights` | *worse*: 1,023 training rows collapsed the one working control (spread 27.3% → 3.2%). The full-population model beat the act-trained model **on the act rows themselves** (corr +0.127 vs +0.041) — a data-quantity effect |
| **Pipeline can't learn anything** | predict `sentiment` (5-way) from text | TF-IDF + logistic: **55.0% acc / 0.444 macro-F1** vs majority 42.7% / 0.120. The text is plainly readable |

That last row matters most. The pipeline reads text fine, and the model commits when
there is something to commit to (27.3% spread on the long-hindsight control, best
epoch 7). It collapses to the mean only when there is nothing to find — which is the
correct behaviour, not a bug: under MSE/Huber the loss-minimising output with no
usable conditioning information *is* the unconditional mean.

---

## 7. Why more tuning will not fix it

- The target is ~1.2% sd with mean ≈ 0. At 3–4k training rows, the signal-to-noise
  needed for positive R² is far above what a ~0.1 correlation provides.
- Capacity, learning rate, loss, window, ticker attribution and subset filtering
  have each been varied and none moved the result off zero.
- The one genuine signal is **hindsight, day-level, and worth ~0.13–0.17
  correlation** — it neither forecasts nor produces positive R².
- The articles describe what the market already did. That is what the data contains,
  and it is what the models found.

---

## 8. What would have to change

Not a model change — a data or problem change:

1. **A different question.** Sign classification with cross-entropy and AUC would at
   least make a 0.1 correlation legible instead of erasing it under squared error
   (`--target-kind sentiment` already demonstrates the classification path works).
2. **Information that precedes the move**, not text published after it. This corpus
   is by construction retrospective.
3. **Much more data.** The act-subset experiment showed 1,023 rows is below the floor
   for this setup; 3k is likely below it too for a ~0.1 correlation.
4. **A market-adjusted target.** The strongest thing the features explain is the
   day's market direction. Residualising against AIQ would strip that and show
   whether anything ticker-specific remains — the current evidence says little does.

---

## 9. Reproducing

```bash
model-shay baselines --30m
model-shay train --30m --kfold --batch-size 32 --eval-test --out-dir runs/cv-30m
model-shay train --30m_till --sentiment --act --kfold --out-dir runs/30m_till-sent
model-tree  train --30m --sentiment --act --kfold --eval-test --out-dir runs/tree-30m
```

Artifacts: `model_shay/runs/*/summary.json` (per-fold metrics, config, length report),
`model_tree/runs/*/{summary.json,importance.csv,model.joblib}`.
Every run directory carries the exact config that produced it.
