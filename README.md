# Human Activity Recognition — Analysis & Rework Log

A step-by-step record of profiling, auditing and rebuilding a working HAR notebook:
what was measured, what was expected, what actually happened, and what each result implies.

**Credit.** The starting point is *Project 18: Human Activity Recognition with Smartphones*
by **DataThinkers** ([video](https://youtu.be/0AxDJPp7ssE) ·
[GitHub](https://github.com/DataThinkers)). The problem framing, dataset choice and Tkinter
front-end come from that walkthrough. Everything below is a critique of a tutorial that
works — the point is not that it is bad, but that a working notebook and a deployable one
are separated by a set of specific, measurable gaps.

**Dataset.** [UCI HAR on Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) —
7,352 train rows, 2,947 test rows, 561 features plus `subject` and `Activity`.

---

## Outcome

| Metric | Before | After | Change |
|---|---|---|---|
| Accuracy on unseen subjects | 0.9253 | **0.9617** | +3.6 pts |
| Reported accuracy | 0.9816 | 0.9617 | −2.0 pts (the old figure measured the wrong thing) |
| GroupKFold estimate | not measured | 0.9169 ± 0.0379 | — |
| Training path runtime | ~6.5–8 min | **10.5 s** | ~40× |
| Full notebook runtime | ~6.5–8 min | 88.5 s | ~5× (and it now runs far more diagnostics) |
| Correctly named output classes | 1 of 6 | **6 of 6** | the largest user-visible fix |

Seven investigations produced these. Four of the seven ended somewhere other than where
the hypothesis pointed.

---

## Ground rules for every number below

Measurement discipline decides whether an optimization log is evidence or anecdote, so the
protocol is fixed before any change is made:

- **Timing** — wall clock via `time.perf_counter()`, single-threaded on one CPU core.
  scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4. Absolute values shrink on a multi-core
  machine; ratios between configurations hold.
- **Accuracy** — always the official `test.csv`. Its nine participants appear nowhere in
  `train.csv`, which makes it the only split in this dataset that answers the operational
  question: *how well does this work on a person it has never seen?*
- **Stability** — `GroupKFold` over `train.csv`, grouped by `subject`.
- **Projections** — where running a step to completion costs minutes, its cost is
  reconstructed from timed sub-fits and labelled a projection, never quoted as a stopwatch
  reading.
- **One variable at a time** — when re-scoring the original models in Step 4, nothing else
  changed: same features, same hyperparameters, only the evaluation split.

---

## Step 0 — Inventory

Reading before touching anything. The original runs 70 cells:

1. Load `train.csv` / `test.csv`
2. Drop duplicate columns via `data.T.duplicated()` → 563 columns become 542
3. `X = data.drop('Activity', axis=1)`, `LabelEncoder` on `y`
4. `train_test_split(X, y, test_size=0.20, random_state=42)`
5. `LogisticRegression()` → **0.9803**
6. `RandomForestClassifier()` → **0.9816**
7. `SelectKBest(f_classif, k=200)` → `RFE(RandomForestClassifier(), n_features_to_select=100)`
8. Retrain forest on the 100 survivors → **0.9769**
9. `joblib.dump` of three separate objects
10. Tkinter GUI that reloads all three and maps predictions back to class names

Two things stand out before a single measurement:

- **Feature selection lowers the reported score** (0.9816 → 0.9769) and is kept anyway.
  The notebook never comments on this.
- **Three separate persisted objects** must be re-applied in the right order by the caller.
  Nothing enforces the order.

Both are noted as leads. Profiling comes first.

---

## Step 1 — Profile before optimizing

**Trigger.** Seven thousand rows and 561 columns is a small problem. A notebook that size
should not take minutes.

**Method.** Time each stage individually rather than guessing from the code shape.

**Expected.** The random forest fits dominate — they are the only obviously heavy models
present.

**Actual.**

| Stage | Time |
|---|---|
| `RFE(RandomForest, 200 → 100, step=1)` | **≈ 370–483 s (projected)** |
| `RandomForestClassifier()` fit, 541 features | 7.7 s |
| `data.T.duplicated()` | 1.0–3.1 s |
| `LogisticRegression()` fit | 1.0 s |
| `read_csv` (both files) | 1.6–3.3 s |

One line is 95% of the runtime, and it is not a model fit — it is a feature selector.

**Why RFE costs that much.** `RFE(estimator, n_features_to_select=100)` leaves `step` at its
default of **1**. Going from 200 features to 100 therefore retrains the forest **100 times**,
discarding one feature per round. The parameter that controls this never appears in the
original code, so the cost is invisible at the call site.

**How the projection was built.** Running 100 rounds to completion wastes minutes per
experiment, so the cost was reconstructed: fit the same forest at 200 / 175 / 150 / 125 / 101
features, then integrate the timing curve across the 100 rounds. Two runs on the same
machine gave 4.28 / 4.07 / 3.69 / 3.42 / 3.08 s → **370 s**, and 5.61 / 5.35 / 4.87 / 4.44 /
4.09 s → **483 s**. The spread is background load; the conclusion is not sensitive to it.

**Insight.** The expensive line looked cheap. `RFE(estimator, n_features_to_select=k)` reads
like a one-shot transform and behaves like a loop with an invisible bound. Cost lives in
defaults at least as often as in the code you wrote — profile, don't infer.

---

## Step 2 — Does RFE earn six minutes?

**Trigger.** Before optimizing that line, check whether it should exist. The original's own
numbers already hint at the answer (0.9816 → 0.9769 after selection).

**Hypothesis.** Selection trades a small amount of accuracy for a smaller feature set. If
the trade is roughly even, speed it up with `step=0.25`. If it costs real accuracy, delete it.

**Method.** Take the selected subsets, retrain from scratch, score on `test.csv`. Also test
a cheaper selector (L1-penalised `LinearSVC` inside `SelectFromModel`) to separate *"RFE is
bad"* from *"selection is bad here."*

**Expected.** A 1-point drop at most. 100 features out of 561 seemed like plenty for six
well-separated classes.

**Actual.**

| Feature set | RandomForest | LinearSVC |
|---|---|---|
| All 540 | 0.9253 | **0.9617** |
| RFE, top 100 | 0.8897 | — |
| L1-`LinearSVC`, top 83 | 0.8884 | 0.9471 |

Both selectors hurt, and the damage is nearly identical (0.8897 vs 0.8884). Six to eight
minutes of compute buys **−3.6 points** on the forest and **−1.5** on the linear model.

**Why.** The 561 features are not a redundant dump from an automated generator. They are a
designed set — mean, std, MAD, energy, entropy, IQR, correlation, FFT band energies, and
gravity-vector angles — each capturing a distinct property of a windowed signal. Redundancy
is what feature selection exploits, and there is comparatively little here. Cutting to 100
cuts signal.

**Insight.** The matched failure of two unrelated selectors is the informative part. Had
only RFE degraded, the selector would be at fault. Both failing at the same magnitude points
at the premise: on hand-engineered features, feature selection has much less slack to work
with than on raw or auto-generated ones. **The stage was deleted, not optimized** — faster
and more accurate at once, which is rare enough to be worth stating plainly.

For the case where a deployment target genuinely constrains feature count, the cheap paths
are documented in the notebook: `SelectFromModel` fits once (4.6 s), and `RFE(..., step=0.25)`
turns 100 sequential fits into about five (5.6 s vs 370 s, a **66×** reduction). Both still
cost accuracy, which is why neither ships.

---

## Step 3 — The duplicate-column scan

**Trigger.** Second-largest line in the profile, and structurally odd: `data.T.duplicated()`
transposes 7,352 × 563 into 563 × 7,352 to compare columns as rows.

**Method.** Replace the transpose with a byte-signature hash — read each column's raw bytes,
push into a `set`, one linear pass, no copy of the frame. Assert both methods return the
same column list.

**Expected.** Maybe 5×. The transpose is one memory copy.

**Actual.** **1.0–3.1 s → 0.02–0.04 s, a 50–70× reduction**, both finding the same 21 columns.
The transpose was not just copying — it was forcing pairwise row comparison over a layout
that fights the CPU cache, and it costs the same whether zero or two hundred duplicates exist.

**Then the more serious problem surfaced.** While confirming the two methods agreed, a
second question came up: agreed *on which frame?* The original recomputes duplicates
independently on `train`, on `test`, and again inside the GUI on whatever file the user
opens. That list is data-dependent — a 5-row file will show far more columns as byte-identical
than a 7,352-row file. The surviving column count then stops matching what the model was
fitted on, and `transform()` either raises a shape error or, worse, aligns the wrong columns
silently.

On the shipped `train.csv` / `test.csv` pair the two lists happen to coincide — 21 columns
both times — which is exactly why the bug never fires during the tutorial and would fire in
production.

**Insight.** A latent bug found while verifying an optimization, not while looking for bugs.
The general rule it illustrates: **any preprocessing decision derived from data must be fitted
on train and then frozen.** Deduplication is preprocessing, even though it does not look like
it and has no `.fit()` method. In the rebuild the list is computed once from `train` and
reused everywhere downstream.

---

## Step 4 — Auditing the 0.98

**Trigger.** A default `RandomForestClassifier()` hitting 0.9816 on a 6-class problem with
7,000 samples, no tuning, is high enough to warrant checking how it was scored. Then reading
the split code:

```python
X = data.drop('Activity', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

Two problems in two lines.

**Problem 1 — `subject` survives as a feature.** Only `Activity` is dropped, so the
participant's ID number stays in the matrix. The model is told *who* produced each row. At
inference that column is absent or meaningless, so any accuracy it earned is unrecoverable.

**Problem 2 — the split cuts across people, not between them.** `train_test_split` shuffles
rows, so consecutive windows from a single walking session land on both sides. Gait is close
to a biometric. Matching a person's own stride to itself is a much easier task than
recognising a stranger's, and the model is being scored on the easier one.

UCI already partitions the data correctly — a fact the tutorial does not use, since it splits
`train.csv` again and only touches `test.csv` at the very end for an unscored demo prediction.

**Method.** Confirm the participant partition, then re-score the original two models against
`test.csv`. Nothing else changed: same features including `subject`, same hyperparameters,
same seed. Only the evaluation split moved.

**Expected.** A 1–2 point drop. Enough to matter, not enough to change conclusions.

**Actual.**

| Model, exactly as written in the original | Random 80/20 | Held-out subjects | Gap |
|---|---|---|---|
| `LogisticRegression()` | 0.9803 | 0.9539 | −2.6 pts |
| `RandomForestClassifier()` | 0.9816 | **0.9237** | **−5.8 pts** |

`train` subjects: 21. `test` subjects: 9. Overlap: **none**.

**Insight.** Nearly six points of the forest's score were recognition of individuals, not
activities — and the leak is model-dependent, which is the part worth internalising. The
forest, splitting on one feature at a time, can isolate a participant's idiosyncratic
sensor signature; the linear model, forced to combine features additively, cannot exploit
it as cleanly and loses less than half as much. **A leaked split does not inflate all models
equally, so it corrupts model selection and not just the headline number.** The forest looks
like the best model under the leaked protocol and is the worst under the honest one — the
ranking inverts.

This is the step that reframes the whole exercise. The goal stops being "beat 0.98" and
becomes "measure honestly, then beat 0.9237."

---

## Step 5 — Choosing a model family

**Trigger.** With honest evaluation in place, the forest sits at 0.9237. Is that the ceiling
or the wrong family?

**Reasoning before measuring.** Three signals point the same way:

- **Shape.** 7,352 samples against 561 features is wide and low-sample. Linear decision
  boundaries are well-determined here; a forest has to spend its capacity discovering
  axis-aligned structure from relatively few examples per leaf.
- **The features are already the non-linearity.** They are hand-engineered frequency-domain
  statistics. The transformations a tree ensemble would learn have largely been applied
  upstream by the dataset authors.
- **The boundaries are dense, not sparse.** Trees split one feature at a time; separating
  these classes needs combinations across many correlated columns at once.

**One more clue in the original output.** It prints a `ConvergenceWarning` for
`LogisticRegression()` and moves on. That warning means the solver hit its default
`max_iter=100` without converging — **the logistic regression baseline was never fully
trained**. The cause is conditioning: the features are pre-normalised to [-1, 1], but
per-column variances still differ by orders of magnitude, which stretches the loss surface.
The fix is `StandardScaler`, not a larger iteration cap.

**Method.** Fit each candidate on all of `train.csv`, score on `test.csv`, record fit time.

**Expected.** Linear models competitive with the forest, maybe a point ahead.

**Actual.**

| Model | Fit time | Accuracy on `test.csv` |
|---|---|---|
| `LinearSVC(C=1)` + scaler | 21.9 s | 0.9623 |
| **`LinearSVC(C=0.1)` + scaler** | **4.5 s** | **0.9617** |
| `SVC(rbf, C=10)` + scaler | 1.7 s | 0.9555 |
| `LogisticRegression` + scaler | 1.4 s | 0.9549 |
| `RandomForest` | 10.2 s | 0.9253 |

A 3.6-point spread — larger than expected, and the entire linear family beats the forest.

**On `C`.** `C=1` wins by 0.0006: six ten-thousandths, under two rows out of 2,947, while
costing 4.8× the fit time. That is inside run-to-run noise. **`C=0.1` is selected** —
indistinguishable accuracy, and stronger regularisation generalises more conservatively to
new participants, which is the actual deployment condition.

**Insight.** The largest single accuracy gain in this project came from matching the model
family to the data's shape, and it cost one line. No tuning, no ensembling, no additional
features. Worth weighing against the reflex to reach for a grid search first — and note that
this decision was only *visible* because Step 4 fixed the evaluation. Under the leaked
protocol both families read 0.98 and the choice looks like a coin flip.

---

## Step 6 — Validating the choice under GroupKFold

**Trigger.** The 0.9617 rests on nine participants. If two are unusual, the number moves.

**Method.** `GroupKFold` over `train.csv` with `subject` as the group key, so every
participant sits entirely inside one fold. Three folds over 21 participants hold out seven
each, closely matching the nine unseen subjects in `test.csv` — the folds simulate deployment
at roughly the right scale.

`KFold` and plain `cross_val_score` are unusable here: they shuffle rows and would reintroduce
the exact same-person leakage removed in Step 4, reporting the same inflated 0.98.

**Expected.** Something near 0.96, confirming the test-set figure.

**Actual.** **0.9169 ± 0.0379.** Folds: 0.9594, 0.8674, 0.9238.

**Reading it.** Lower than 0.9617 for a structural reason rather than a contradictory one:
each fold trains on 14 participants instead of all 21, so the model sees a third less
diversity. Treat it as a conservative lower bound, and `test.csv` — trained on all 21 — as
the realistic figure.

**Insight.** The spread is more useful than the mean. Nearly ten points separate the best
and worst folds, which says **which people are held out matters far more than any
hyperparameter tuned here.** That sets the noise floor, and it retroactively confirms the
Step 5 call: a 0.0006 gap between `C=0.1` and `C=1` is invisible against ±0.038 of
subject-to-subject variance. It also sets the bar for future work — an improvement under
1 point is not distinguishable from luck at this sample size.

---

## Step 7 — The label-decoding bug

**Trigger.** Found by accident. While rewriting the GUI for the new single-artifact model,
this line needed porting:

```python
# standing : 0, sitting : 1, laying : 2, WALKING_DOWNSTAIRS: 3,
# walking_upstairs: 4, walking : 5
y_pred = y_pred.map({0: 'Standing', 1: 'Sitting', 2: 'Laying',
                     3: 'Walking_downstairs', 4: 'Walking_upstairs',
                     5: "Walking"})
```

A hand-written integer-to-name table. `LabelEncoder` assigns codes by **sorting the class
strings alphabetically**, which is not the order anyone would guess from reading the data.

**Method.** Print the fitted `label_encoder.classes_` beside the hard-coded table.

**Expected.** One or two classes transposed.

**Actual.**

| Code | `LabelEncoder` | Original table | |
|---|---|---|---|
| 0 | LAYING | Standing | **MISMATCH** |
| 1 | SITTING | Sitting | ok |
| 2 | STANDING | Laying | **MISMATCH** |
| 3 | WALKING | Walking_downstairs | **MISMATCH** |
| 4 | WALKING_DOWNSTAIRS | Walking_upstairs | **MISMATCH** |
| 5 | WALKING_UPSTAIRS | Walking | **MISMATCH** |

**Five of six classes are wrong.** Only SITTING lands on itself, by coincidence.

**Impact.** The classifier predicts correctly. `accuracy_score` reads 0.96 because it compares
integers. Then the integers are renamed wrongly on the way out. A GUI user receives a CSV in
which roughly one row in six carries the right activity name. No exception is raised, and no
cell in the original notebook would ever display the discrepancy — every metric it computes
lives upstream of the rename.

**Fix.** Stop writing the mapping down. `label_encoder.inverse_transform()` reads it from the
fitted object, so decoding cannot drift from encoding. The encoder ships inside the same
`.joblib` bundle as the pipeline, which makes the two impossible to separate.

**Insight.** This is the most damaging defect in the project and the cheapest to fix, and it
was found while refactoring rather than while testing — no accuracy metric could have caught
it. The pattern generalises: **`accuracy_score` validates the model, not the product.**
Everything downstream of `argmax` — decoding, formatting, file writing, display — is untested
surface in most ML notebooks, and it is the only part the end user actually sees. A single
end-to-end assertion from raw CSV to final string, of the kind added in Step 8, would have
caught this on day one.

---

## Step 8 — Packaging and the inference path

**Trigger.** Three persisted artifacts (`model_rfe`, `k_best_selector`, `rfe_selector`) whose
correct application order lives only in the author's memory and in the sequence of GUI
statements. Getting it wrong raises nothing — it just returns wrong predictions.

**Changes.**

- **One `Pipeline`, one file.** The order becomes data rather than convention, so
  `.predict()` cannot be called out of sequence. `FEATURES` and the fitted `LabelEncoder`
  travel in the same bundle.
- **One inference entry point** (`predict_activities`) shared by notebook and GUI, replacing
  duplicated preprocessing in `process_data()`.
- **Columns selected by the stored `FEATURES` list**, never recomputed from the incoming
  file — this is the Step 3 bug closed at its actual point of impact.
- **Missing columns checked up front** and reported by name, instead of surfacing as a shape
  error from deep inside scikit-learn.
- **Model loaded once at GUI startup**, not on every button press: three `joblib.load` calls
  per file become zero.
- **`askopenfilename` replaces `askopenfile`**, which returned a file handle that was opened
  and never closed.
- **`to_csv(index=False)`** stops an unnamed index column appearing in every output file.
- **The success dialog reports predicted class counts**, so a wrong result is visible
  immediately rather than after opening the file elsewhere.

**Verification.** A round-trip test — save, reload from disk, predict on `test.csv`, compare
strings against `test['Activity']` — returns 0.9617, matching the in-memory model. This is
the end-to-end assertion whose absence let Step 7's bug survive.

---

## Consolidated results

| Change | Measured against | Result |
|---|---|---|
| Delete RFE stage | wall clock + `test.csv` | −370 to −483 s, **+3.6 pts** |
| Hash-based dedup | wall clock, output asserted identical | 1.0–3.1 s → 0.02–0.04 s (**50–70×**) |
| Freeze dedup list from train | robustness of inference path | removes a data-dependent shape failure |
| Drop `subject` from features | `test.csv` | removes identity leakage |
| Evaluate on `test.csv` | vs. random 80/20 | exposes a **5.8 pt** overstatement |
| `RandomForest` → `LinearSVC(C=0.1)` + scaler | `test.csv` | 0.9253 → **0.9617** |
| `GroupKFold(n_splits=3)` | `train.csv`, grouped | 0.9169 ± 0.0379 (lower bound + noise floor) |
| `inverse_transform` for decoding | class-by-class comparison | **5 of 6** labels corrected |
| Single `.joblib` bundle | round-trip test | 0.9617 reproduced from disk |

**Runtime, timed end to end on one core:** training path (load → clean → fit → evaluate)
**10.5 s**; full notebook including every diagnostic cell **88.5 s**; original **≈ 6.5–8 min**.

---

## Where the error now lives

The per-class breakdown localises what remains:

| Class | Precision | Recall |
|---|---|---|
| LAYING | 1.00 | 0.98 |
| **SITTING** | 0.96 | **0.87** |
| **STANDING** | **0.88** | 0.97 |
| WALKING | 0.97 | 0.99 |
| WALKING_DOWNSTAIRS | 1.00 | 0.98 |
| WALKING_UPSTAIRS | 0.98 | 0.97 |

113 errors total. **60 of them are SITTING read as STANDING** — 53% of all error in one
ordered pair; adding the 15 in the reverse direction brings it to 66%.

The cause is a sensor limitation rather than a modelling failure. Both postures are static,
so the accelerometer registers no motion energy, and the only discriminating information is
torso tilt — recoverable from the `tGravityAcc` and `angle(..., gravityMean)` features alone.
Every dynamic class exceeds 0.97 on both metrics.

**Insight.** Two thirds of the remaining error is one confusion pair with a physical
explanation. Broad interventions — more trees, more features, a wider grid search — spend
effort across four classes that are effectively solved.

---

## Corrections made during this work

Kept for transparency, since both were caught by re-measurement rather than review:

- **RandomForest baseline first quoted as 0.9237.** That figure came from a forest trained on
  80% of `train.csv` with `subject` still present. Re-measured under the final protocol
  (full train, `subject` dropped) it is **0.9253**. The corrected figure is used throughout.
- **"End-to-end under 30 seconds" was an unverified estimate.** Timed properly, the training
  path is 10.5 s and the full notebook 88.5 s. The claim had been extrapolated from component
  timings rather than measured, which is precisely the error Step 1 exists to avoid.
- **The RFE projection is a range, not a point.** Two runs on the same machine projected
  370 s and 483 s. Background load explains the spread; the decision to delete the stage does
  not depend on which figure is right.

---

## What generalises

1. **Profile before optimizing.** The expensive line was a feature selector, not a model fit,
   and its cost came from an unstated default (`step=1`) rather than from anything written
   at the call site.
2. **Ask whether a stage should exist before making it fast.** RFE was a candidate for a
   66× speedup. Measurement said delete it — faster *and* more accurate.
3. **A leaked split corrupts model selection, not just the score.** The forest ranked first
   under the leaked protocol and last under the honest one. Fixing evaluation was the
   precondition for every downstream decision being meaningful.
4. **Match the model family to the data's shape.** Wide, low-sample, hand-engineered
   features favour linear models. The biggest accuracy gain here cost one line.
5. **Freeze anything data-derived at fit time.** Duplicate-column detection has no `.fit()`
   method and is still preprocessing. Recomputing it at inference is the same class of bug
   as refitting a scaler on the test set.
6. **`accuracy_score` validates the model, not the product.** The worst defect in the
   original lived entirely downstream of the metric. One end-to-end assertion from raw CSV
   to final string would have caught it immediately.
7. **Report the variance, not only the mean.** ±0.038 across folds set the threshold below
   which any comparison — including the `C=0.1` vs `C=1` decision — is noise.

---

## Reproducing

```
.
├── Human_Activity_Recognition_Reworked.ipynb
├── README.md
├── train.csv
└── test.csv
```

Edit `DATA_DIR` in the Configuration cell if the CSVs live elsewhere, then run all cells.
Training writes `har_model.joblib`; the GUI is opened by calling `launch_gui()` after that
file exists.

Requires `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `joblib`. Tkinter ships
with CPython on Windows and macOS; on Debian/Ubuntu install `python3-tk`. The GUI needs a
display and will not open under Colab or a headless server.

---

*Reworked from the DataThinkers walkthrough: [video](https://youtu.be/0AxDJPp7ssE) ·
[GitHub](https://github.com/DataThinkers)*
