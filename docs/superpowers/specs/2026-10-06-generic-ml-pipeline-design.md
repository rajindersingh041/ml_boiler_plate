# Generic ML Pipeline (A+B) — Design Spec

Date: 2026-10-06
Status: approved in chat (all 5 sections)
Scope: extend-in-place (A) + local registry & tuning (B)

## 1. Goal

One generic supervised pipeline covering classification, regression, and
single-series timeseries forecasting, end to end:

`data → EDA → validation split → CV → tuning → final train →
evaluation (incl. distributions) → registry → serve → monitor`

with extensive offline tests at every stage. No network required for
tests. Existing `Task` abstraction is preserved and extended, not
replaced.

## 2. Architecture (what is added, what is untouched)

Untouched: `io.py`, `data.py`, `features.py` (shared ColumnTransformer),
`metrics.py` (extended only), `predict.py` (reads from registry next),
`tasks/base.py` shape (methods added, none removed).

New modules under `src/ml_boilerplate/`:

| Module | Purpose |
|---|---|
| `validation.py` | train/val/test splitter; stratified for classification, chronological for timeseries; returns 6-tuple |
| `crossval.py` | k-fold / stratified-k-fold / time-series-split CV over the full sklearn Pipeline; per-fold + mean±std metrics |
| `tuning.py` | Grid + randomized search over `model__*` params via sklearn search CV; returns best pipeline + run record |
| `evaluation.py` | final held-out test gate: metrics + error-distribution artifacts + pass/fail vs thresholds |
| `distributions.py` | data + prediction + residual distribution profiles and plotly HTML (backs EDA and monitor) |
| `registry.py` | local file registry: `runs/<run_id>/` (model, metrics, params, profile); `promote(run_id)` copies to `production/` |
| `api.py` | FastAPI app: `GET /health`, `POST /predict` (loads production model, task-aware features) |
| `monitor.py` | compares live batch profile vs training profile: PSI/KS per column, target/prediction drift chart |

Config additions (`config.py`, all optional with defaults):

- `validation: { val_size, test_size }`
- `cross_validation: { folds, shuffle }`
- `tuning: { enabled, method: grid|random, param_grid, n_iter }`
- `evaluation: { fail_on_regression: bool, min_metrics: dict }`
- `registry: { runs_dir, production_dir }`
- `serving: { host, port }`

CLI additions (`main.py`): `validate`, `crossval`, `tune`, `evaluate`,
`serve`, `monitor` alongside existing `train`, `predict`, `eda`.

## 3. Validation / CV / tuning

- `validation.split_train_val_test(df, cfg, task)`:
  classification → stratified; timeseries → chronological
  (train | val | test in time order); regression → random.
- `crossval.run_cv(pipeline_fn, X, y, cfg, task)`: builds the same
  preprocess+model pipeline per fold; returns
  `{per_fold: [...], aggregate: {mean, std}}` saved to
  `artifacts/cv_results.json`.
- `tuning.run_tuning(X_train, y_train, cfg, task)`: param names are
  `model__<param>` so preprocessor is never tuned by accident; best
  estimator refit on full train split; best params + CV score written
  into the run record.
- Reproducibility: every split/search takes `random_state` from config;
  run record stores config snapshot + package versions.

## 4. Distributions (data + errors + drift)

`distributions.py` exposes three pure functions (DataFrame in, dict/paths out):

1. `data_distributions(df, target)` — histograms, target distribution,
   missingness, cardinality (extends current EDA; reuses `eda.py`).
2. `error_distributions(y_true, y_pred, y_proba, task)` — residuals
   (regression), calibration + ROC (classification), prediction
   histogram; saved as plotly HTML.
3. `drift_report(train_profile, serve_df)` — PSI + KS per column,
   drift chart; consumed by `monitor.py`.

All charts are standalone plotly HTML (existing convention).

## 5. Production (registry → serve → monitor)

- `registry.save_run(pipeline, metrics, cfg)` → `runs/<utc-id>/`
  containing `model.joblib`, `metrics.json`, `params.json`,
  `profile.json` (training data profile for drift).
- `registry.promote(run_id)` → copies run to `production/` (atomic
  directory swap); `predict.py` and `api.py` load from `production/`
  by default with explicit `--model-path` override retained.
- `api.py`: FastAPI, `/health` returns model version + task;
  `/predict` validates input schema (columns) and returns predictions
  + run id. No training code in the serving path.
- `monitor.py`: `check(live_csv)` loads production profile, runs
  `drift_report`, exits nonzero on PSI > 0.25 (configurable) so it
  works as a CI gate.
- `Dockerfile`: `uv`-based image running `uvicorn api:app`;
  model baked via `production/` mount (not into the image layer).

## 6. Testing strategy

All tests offline (synthetic data), runnable with `uv run pytest`:

| Layer | Tests |
|---|---|
| unit | `validation` split ratios/stratification/chronology; `crossval` fold counts + aggregation math; `tuning` best-params plumbing (tiny grid); `distributions` empty-frame + zero-target edge cases; `registry` save/promote round-trip (tmp dir); `monitor` PSI known-shift detection |
| integration | `train → evaluate → register → predict` end to end per task; `tune → evaluate` smoke on tiny grid |
| contract | FastAPI `TestClient`: `/health`, `/predict` schema + error on missing column |
| config | existing example-config sanity extended to new blocks (parse only, no network) |

Failure policy: unit tests never touch disk outside `tmp_path`;
integration tests use `synthetic` source only.

## 7. Non-goals (this spec)

- No hosted tracker (no MLflow/Weights&Biases); local file registry only.
- No walk-forward rolling-origin CV (single chronological split + TimeSeriesSplit folds only).
- No xgboost/lightgbm; sklearn registries only (unchanged).
- No auth/rate-limiting on the API.

## 8. Self-review

- Placeholders: none — all modules, config keys, CLI commands named.
- Consistency: `predict.py`/`api.py` both read `production/`; tuning param prefix `model__` matches pipeline step name in `train.py`.
- Scope: single implementation plan; tracker/search kept local-only per B-lite agreement.
- Ambiguity resolved: "distributions" = data + error + drift; "production" = registry + FastAPI + Docker + monitor gate; CV = train/val/test + k-fold (not nested).
