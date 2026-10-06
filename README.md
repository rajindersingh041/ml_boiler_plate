# ML Boilerplate

Generic, extensible scaffold for supervised ML — classification, regression,
or single-series time-series forecasting — usable two ways: driven by one
YAML config and a small CLI (**data → EDA → features → train → validate →
tune → evaluate → register → serve → monitor**), or à la carte as a plain
Python library (every stage is an importable function, no YAML required).
Managed end-to-end with [`uv`](https://docs.astral.sh/uv/).

## Structure

```
configs/
  classification.yaml       # binary classification (Titanic)
  classification-penguins.yaml  # 3-class classification (Penguins)
  classification-iris.yaml      # 3-class classification, clean (Iris)
  regression.yaml             # tabular regression (California Housing)
  regression-diamonds.yaml    # large tabular regression (Diamonds, 54k)
  regression-mpg.yaml         # small tabular regression (Auto MPG)
  timeseries.yaml               # monthly forecasting (Airline Passengers)
  timeseries-births.yaml        # daily forecasting (Female Births 1959)
  timeseries-sunspots.yaml      # long monthly forecasting (Sunspots 1749-1983)
data/                       # vendored offline CSV copies of the six newer examples
src/ml_boilerplate/
  config.py                 # dataclasses + YAML loader (task type, I/O, EDA, models, ...)
  io.py                     # read_table()/write_table(): csv or parquet, local path or http(s) URL
  eda.py                    # profile_dataframe() (schema/missing/stats) + run_eda() -> report + charts
  engineering.py            # add_datetime_features() (parts/cyclical/holidays) + add_lag_features() + featurize()
  plotting.py               # plotly charts (EDA + per-task result charts), saved as standalone HTML
  features.py               # shared ColumnTransformer: numeric/categorical impute + encode + scale
  model.py                  # CLASSIFIER_REGISTRY / REGRESSOR_REGISTRY (linear, RF, boosting, bagging)
  metrics.py                # classification_metrics() (binary + multiclass) + regression_metrics() (MAPE/sMAPE)
  validation.py             # split_train_val_test(): stratified / chronological / random
  crossval.py               # run_cv(): k-fold over the full pipeline, per-fold + mean±std
  tuning.py                 # run_tuning(): grid/random search over model__* params
  distributions.py          # data/error/drift distribution profiles (JSON-serializable)
  evaluation.py             # evaluate_on_test(): metrics + pass/fail gate vs min_metrics
  registry.py               # save_run()/promote()/load_production(): local file model registry
  api.py                    # FastAPI: GET /health, POST /predict over the production model
  monitor.py                # check_drift(): per-column PSI vs the training profile (CI gate)
  tasks/                     # the abstraction layer — one class per problem type
    base.py                  # Task ABC: split(), model_registry(), compute_metrics(), plot_results()
    classification.py         # stratified random split, confusion matrix + ROC chart
    regression.py               # random split, predicted-vs-actual + residuals chart
    timeseries.py                 # lag/rolling features, chronological split, forecast chart
  data.py                    # synthetic generator per task, or load via io.py for real data
  train.py                   # generic driver: wires the above into one Task-agnostic training run
  predict.py                 # generic driver: load model, score new data, write csv/parquet
  main.py                    # CLI: train/predict/eda/validate/crossval/tune/evaluate/promote/serve/monitor
notebooks/
  interview_boilerplate.ipynb  # live-interview EDA kit: set 4 vars, run all, narrate
  library_usage.ipynb          # à la carte recipes: every stage as a function call (runs offline)
tests/                       # unit + smoke tests per module, offline config sanity tests
artifacts/<task>/            # per-example model, metrics.json, and plots/ (created after training)
```

## Setup

```bash
uv sync
```

Creates `.venv` and installs everything (including dev deps from
`[dependency-groups].dev`) pinned exactly as recorded in `uv.lock`. No manual
venv/pip steps — every command below runs through `uv run`, which uses that
same environment automatically.

## Install anywhere (including a live interview)

The package installs straight from git into any environment — verified in a
clean venv — so `git pull` (or one install line) is the whole setup:

```bash
pip install "ml-boilerplate @ git+https://github.com/rajindersingh041/ml_boiler_plate.git"
# or: uv pip install "ml-boilerplate @ git+https://github.com/rajindersingh041/ml_boiler_plate.git"
```

then `import ml_boilerplate` and use it as below. For the notebooks,
clone the repo instead (vendored `data/*.csv` files make both notebooks
runnable offline).

## Use it as a library (no YAML, no CLI)

`Config(task=...)` built in code replaces every YAML file; only
`run_training`/`main` need YAML. See `notebooks/library_usage.ipynb`
(executed green, runs offline) for the full recipes — the shape is:

```python
from sklearn.pipeline import Pipeline
from ml_boilerplate.config import Config, ModelConfig
from ml_boilerplate.data import load_data
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task
from ml_boilerplate.validation import split_train_val_test
from ml_boilerplate.evaluation import evaluate_on_test

cfg = Config(task="regression")          # plain object, set attrs in code
df = load_data(cfg.data, "regression")   # synthetic offline; or read_table("data/mpg.csv")

task = get_task("regression")
X = df.drop(columns=["target"]); y = df["target"]
Xtr, _, Xte, ytr, _, yte = split_train_val_test(X, y, cfg, "regression")
pipe = Pipeline([("preprocess", task.build_preprocessor(Xtr, cfg)),
                 ("model", build_model(ModelConfig(type="random_forest"), task.model_registry()))])
pipe.fit(Xtr, ytr)
gate = evaluate_on_test(yte, pipe.predict(Xte), None, "regression", {"r2": 0.5})
print(gate["passed"], gate["metrics"]["r2"])
```

More entry points: `profile_dataframe` (EDA dict), `data_distributions` /
`error_distributions`, `build_model` + registries, `run_tuning`,
`task.build_preprocessor(X, cfg)` (shared impute/encode/scale),
`create_app` (FastAPI).

## The abstraction layer

Everything that differs between task types lives on a `Task` subclass
(`tasks/base.py`): how to split the data, which models are valid, which
metrics to compute, and which chart to draw as a result. `train.py` and
`predict.py` never branch on task type themselves — they just call
`get_task(cfg.task)` and use whatever it returns. To add a new problem type,
subclass `Task` and register it in `tasks/__init__.py`; to add a new model,
add one line to `CLASSIFIER_REGISTRY`/`REGRESSOR_REGISTRY` in `model.py`.

## The nine examples — real datasets, zero setup

Each config pulls a well-known real dataset straight from a public CSV
mirror over http(s) (no Kaggle login needed to run these); the same
datasets are also published as Kaggle datasets/competitions if you'd rather
fetch them via `kaggle competitions download`/`kaggle datasets download` and
point `data.path` at the local file instead. Vendored offline copies of the
six newer datasets live in `data/` (see each config's header comment).

```bash
# Classification
uv run python -m ml_boilerplate.main --config configs/classification.yaml train            # Titanic, binary (Age ~20% missing)
uv run python -m ml_boilerplate.main --config configs/classification-penguins.yaml train  # Penguins, 3-class (real missing values)
uv run python -m ml_boilerplate.main --config configs/classification-iris.yaml train      # Iris, clean 3-class smoke test

# Regression
uv run python -m ml_boilerplate.main --config configs/regression.yaml train          # California Housing (total_bedrooms ~1% missing)
uv run python -m ml_boilerplate.main --config configs/regression-diamonds.yaml train # Diamonds, 54k rows (hist_gradient_boosting)
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml train      # Auto MPG (horsepower 6 rows missing)

# Time-series (single-series forecasting)
uv run python -m ml_boilerplate.main --config configs/timeseries.yaml train          # Airline Passengers, monthly 1949-1960
uv run python -m ml_boilerplate.main --config configs/timeseries-births.yaml train   # Daily births, 1959 (daily lags)
uv run python -m ml_boilerplate.main --config configs/timeseries-sunspots.yaml train # Monthly sunspots, 1749-1983 (~11y cycle)
```

Each writes `artifacts/<task>/model.joblib`, `artifacts/<task>/metrics.json`, and
a set of interactive HTML charts under `artifacts/<task>/plots/` — note each
example config uses its own artifacts directory so training one doesn't
overwrite another's saved model.

Prefer to start from synthetic data instead (e.g. offline, or to sanity-check
a change without network access)? Set `data.source: synthetic` in any config —
`data.py` has a generator for each task type.

`--config`/`-v` work whether given before or after the subcommand.

Note on the timeseries example: Random Forest (and tree models generally)
can't extrapolate beyond the target range they were trained on, so a
strongly trending series like this one will under/over-shoot at the trend's
edges — a real, teachable limitation, not a bug. Try `model.type:
gradient_boosting`, or add an explicit trend feature, to see the difference.

## Full pipeline (CLI)

Beyond `train`/`predict`/`eda`, every stage runs standalone:

```bash
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml validate  # split sizes
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml crossval  # k-fold -> artifacts/cv_results.json
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml tune      # needs tuning.param_grid
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml evaluate  # gate vs evaluation.min_metrics (exit 1 on fail)
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml promote --run-id <id>
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml serve     # FastAPI on serving.host:port
uv run python -m ml_boilerplate.main --config configs/regression-mpg.yaml monitor --input live.csv  # PSI gate (exit 1 on drift)
```

## Feature engineering

`engineering.py`, shared by all tasks (same functions at train and predict
time, so no train/serve skew). Opt-in via `features:` — defaults reproduce
the legacy timeseries builder exactly:

```yaml
features:
  datetime_columns: [signup_date]  # tabular date cols -> parts below; original col dropped
  cyclical_encoding: true          # month/dow sin+cos
  holiday_country: "US"            # is_holiday + days_to_holiday (holidays lib; null disables)
  rolling_stats: [mean, std]       # any of mean | std | min | max
  lag_diffs: false                 # target.diff(lag) features (timeseries)
```

Lag/rolling windows always build on `.shift(1)` first — no feature ever
sees the current row's target. See `timeseries-births.yaml` for a live
example (cyclical + US holidays on).

## EDA

Run standalone, without training:

```bash
uv run python -m ml_boilerplate.main --config configs/regression.yaml eda
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

- Classification: accuracy, precision, recall, f1 (binary average for 2
  classes, weighted for multiclass), roc_auc (binary only)
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
  source: file              # "synthetic" (offline quick-start) | "file"
  path: data/train.csv       # local path OR an http(s) URL — both work out of the box for csv
  format: csv                 # "csv" | "parquet" | omit to infer from the path's extension
```

Parquet needs `pyarrow` (already a dependency). Parquet-over-URL isn't
supported yet (would need `fsspec`); CSV-over-URL works natively via pandas —
that's how the example configs above pull real data with no download
step.

For `task: timeseries`, also set `date_column`, and optionally tune
`n_lags` (default `[1, 7, 14]`, daily-shaped — the shipped monthly example
overrides this to `[1, 12]`) and `rolling_windows` (default `[7, 14]`,
overridden to `[3, 12]` for the monthly example).

## Predicting on new data

```bash
uv run python -m ml_boilerplate.main --config configs/classification.yaml predict \
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
uv run pytest
```

The per-task, engineering, metrics, registry, API, and monitor tests run
fully offline against synthetic data; a separate `test_example_configs.py`
only parses the nine real-dataset YAML files (no network call) so a config
typo still fails fast in CI.

## Adding your own dependency

```bash
uv add <package>          # runtime dependency
uv add --group dev <package>  # dev-only dependency
```

Both update `pyproject.toml` and `uv.lock` together — commit both.

## Non-goals (for now)

- No walk-forward / rolling-origin evaluation for time series (single
  chronological train/test split; CV does offer `TimeSeriesSplit` folds).
- No xgboost/lightgbm — the model registries are scikit-learn only
  (`gradient_boosting`, `hist_gradient_boosting`, `bagging`, `random_forest`
  cover most boosting/bagging needs); add your own registry entries if you
  need them.
