# ML Boilerplate

Generic, extensible scaffold for a supervised ML learning task — classification,
regression, or single-series time-series forecasting — driven by one YAML config
and a small CLI: **data → EDA → features → train → evaluate → predict**, with
interactive charts at every stage.

## Structure

```
configs/
  classification.yaml   # example 1: binary classification
  regression.yaml         # example 2: tabular regression
  timeseries.yaml           # example 3: single-series forecasting
src/ml_boilerplate/
  config.py                 # dataclasses + YAML loader (task type, I/O, EDA, models, ...)
  io.py                     # read_table()/write_table(): csv or parquet, local path or http(s) URL
  eda.py                    # profile_dataframe() (schema/missing/stats) + run_eda() -> report + charts
  plotting.py               # plotly charts (EDA + per-task result charts), saved as standalone HTML
  features.py               # shared ColumnTransformer: numeric/categorical impute + encode + scale
  model.py                  # CLASSIFIER_REGISTRY / REGRESSOR_REGISTRY (linear, RF, boosting, bagging)
  metrics.py                 # classification_metrics() + regression_metrics() (incl. MAPE/sMAPE)
  tasks/                     # the abstraction layer — one class per problem type
    base.py                  # Task ABC: split(), model_registry(), compute_metrics(), plot_results()
    classification.py         # stratified random split, confusion matrix + ROC chart
    regression.py               # random split, predicted-vs-actual + residuals chart
    timeseries.py                 # lag/rolling features, chronological split, forecast chart
  data.py                    # synthetic generator per task, or load via io.py for real data
  train.py                   # generic driver: wires the above into one Task-agnostic training run
  predict.py                 # generic driver: load model, score new data, write csv/parquet
  main.py                    # CLI: `train`, `predict`, `eda` subcommands
tests/                       # one smoke test per task + one for EDA
artifacts/<task>/            # per-example model, metrics.json, and plots/ (created after training)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## The abstraction layer

Everything that differs between task types lives on a `Task` subclass
(`tasks/base.py`): how to split the data, which models are valid, which
metrics to compute, and which chart to draw as a result. `train.py` and
`predict.py` never branch on task type themselves — they just call
`get_task(cfg.task)` and use whatever it returns. To add a new problem type,
subclass `Task` and register it in `tasks/__init__.py`; to add a new model,
add one line to `CLASSIFIER_REGISTRY`/`REGRESSOR_REGISTRY` in `model.py`.

## The three examples

```bash
# 1. Classification (logistic regression, random forest, boosting, bagging, ...)
python -m ml_boilerplate.main --config configs/classification.yaml train

# 2. Regression (linear/ridge, random forest, gradient boosting, ...)
python -m ml_boilerplate.main --config configs/regression.yaml train

# 3. Time-series forecasting (lag/rolling features + chronological split)
python -m ml_boilerplate.main --config configs/timeseries.yaml train
```

Each writes `artifacts/<task>/model.joblib`, `artifacts/<task>/metrics.json`, and
a set of interactive HTML charts under `artifacts/<task>/plots/` — note each
example config uses its own artifacts directory so training one doesn't
overwrite another's saved model.

`--config`/`-v` work whether given before or after the subcommand.

## EDA

Run standalone, without training:

```bash
python -m ml_boilerplate.main --config configs/regression.yaml eda
```

Writes `eda_report.json` (shape, dtypes, missing value count/% per column,
numeric summary stats, categorical cardinality) plus three charts —
`missing_values.html`, `histograms.html`, `correlation_heatmap.html` — into
`eda.output_dir`. It also runs automatically at the start of `train` unless
`eda.enabled: false`.

## Missing values

Reported (not silently dropped) by the EDA step above. Actual handling is
via `SimpleImputer` inside the shared preprocessing pipeline
(`features.py`), configurable per config:

```yaml
features:
  missing_strategy_numeric: median          # mean | median | most_frequent | constant
  missing_strategy_categorical: most_frequent
```

## Metrics

- Classification: accuracy, precision, recall, f1, roc_auc
- Regression / time series: MAE, RMSE, R2, MAPE, sMAPE (`metrics.py`)

Note MAPE is undefined/unstable when the target is at or near zero (a known
limitation of the metric, not a bug) — sMAPE is more robust in that case.

## Charts

All rendered with plotly and saved as standalone interactive HTML (open
directly in a browser, no server needed):

| Task           | Charts                                                     |
|----------------|--------------------------------------------------------------|
| any (EDA)      | missing values bar chart, histograms, correlation heatmap    |
| classification | confusion matrix, ROC curve                                   |
| regression     | predicted vs actual, residuals                                 |
| timeseries     | forecast — actual vs predicted over time, train/test boundary marked |

## Input data

Set in each config's `data:` block:

```yaml
data:
  source: file              # "synthetic" (default, for quick starts) | "file"
  path: data/train.csv       # local path OR an http(s) URL — both work out of the box for csv
  format: csv                 # "csv" | "parquet" | omit to infer from the path's extension
```

Parquet needs `pyarrow` (already a dependency). Parquet-over-URL isn't
supported yet (would need `fsspec`); CSV-over-URL works natively via pandas.

For `task: timeseries`, also set `date_column`, and optionally tune
`n_lags` (default `[1, 7, 14]`) and `rolling_windows` (default `[7, 14]`).

## Predicting on new data

```bash
python -m ml_boilerplate.main --config configs/classification.yaml predict \
  --input new_data.csv --output predictions.parquet
```

Output format is inferred from `--output`'s extension unless
`output.format` is set explicitly in the config. For `task: timeseries`,
`--input` must look like the raw historical data (a date column + the
target column's history, not a single future row) — lag/rolling features
are rebuilt from that history, the same way as during training; the first
`max(n_lags + rolling_windows)` rows are dropped as a warm-up window.

## Tests

```bash
pytest
```

## Non-goals (for now)

- No walk-forward / rolling-origin cross-validation for time series (single
  chronological train/test split only).
- No xgboost/lightgbm — the model registries are scikit-learn only
  (`gradient_boosting`, `hist_gradient_boosting`, `bagging`, `random_forest`
  cover most boosting/bagging needs); add your own registry entries if you
  need them.
