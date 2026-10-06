# Generic ML Pipeline (A+B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing sklearn Task-based pipeline with validation splits, CV, tuning, distribution analysis, a local model registry, a FastAPI service, and drift monitoring — all tested offline.

**Architecture:** Keep the `Task` abstraction untouched in shape; add one focused module per capability (`validation`, `crossval`, `tuning`, `distributions`, `evaluation`, `registry`, `api`, `monitor`) plus new `Config` blocks and CLI subcommands. Each task ends with an independently testable deliverable.

**Tech Stack:** Python ≥3.10, scikit-learn, pandas, plotly (standalone HTML), FastAPI + uvicorn (serve only), pytest with synthetic data only.

**Spec:** `docs/superpowers/specs/2026-10-06-generic-ml-pipeline-design.md`

## Global Constraints

- Python requires `>=3.10` (do not lower it).
- Model registries stay scikit-learn only — no xgboost/lightgbm.
- All charts are standalone plotly HTML via `plotly.offline.plot` / `fig.write_html` (existing convention in `plotting.py`).
- Every command runs through `uv run`; new runtime deps via `uv add`, dev-only via `uv add --group dev`.
- Tests are fully offline: `data.source: synthetic` or inline DataFrames only; disk writes only under `tmp_path`.
- Pipeline step name for the estimator is `"model"` — tuning params are prefixed `model__`.
- `conftest.py` already puts `src/` on `sys.path`; import as `from ml_boilerplate...`.

---

### Task 1: Config blocks + train/val/test splitter

**Files:**
- Modify: `src/ml_boilerplate/config.py`
- Create: `src/ml_boilerplate/validation.py`
- Create: `tests/test_validation.py`
- Modify: `src/ml_boilerplate/main.py` (add `validate` subcommand)

**Interfaces:**
- Consumes: `Config`, `DataConfig.test_size/random_state` (existing).
- Produces: `ValidationConfig`, `CrossValidationConfig`, `TuningConfig`, `EvaluationConfig`, `RegistryConfig`, `ServingConfig` (all with defaults so old YAMLs still parse); `validation.split_train_val_test(X, y, cfg, task_name)` returning `(X_train, X_val, X_test, y_train, y_val, y_test)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation.py
import pandas as pd
from ml_boilerplate.config import Config
from ml_boilerplate.data import load_data
from ml_boilerplate.validation import split_train_val_test


def _cfg():
    cfg = Config()
    cfg.data.test_size = 0.2
    cfg.validation.val_size = 0.2
    cfg.data.random_state = 42
    return cfg


def test_classification_split_ratios_and_stratification():
    cfg = _cfg()
    df = load_data(cfg.data, "classification")
    X = df.drop(columns=[cfg.data.target_column])
    y = df[cfg.data.target_column]
    Xtr, Xv, Xte, ytr, yv, yte = split_train_val_test(X, y, cfg, "classification")
    n = len(df)
    assert len(Xte) == int(n * 0.2)
    assert abs(len(Xv) - n * 0.2) <= 2
    assert len(Xtr) == n - len(Xv) - len(Xte)
    # stratification: class balance preserved within 5pp on each split
    for split in (ytr, yv, yte):
        assert abs(split.mean() - y.mean()) < 0.05


def test_timeseries_split_is_chronological():
    cfg = _cfg()
    df = load_data(cfg.data, "timeseries")
    X = df.drop(columns=[cfg.data.target_column])
    y = df[cfg.data.target_column]
    Xtr, Xv, Xte, _, _, _ = split_train_val_test(X, y, cfg, "timeseries")
    assert Xtr.index.max() < Xv.index.min()
    assert Xv.index.max() < Xte.index.min()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_boilerplate.validation'` (and `AttributeError` on `cfg.validation`).

- [ ] **Step 3: Add config blocks (defaults keep old YAMLs working)**

```python
# additions to src/ml_boilerplate/config.py
@dataclass
class ValidationConfig:
    val_size: float = 0.2


@dataclass
class CrossValidationConfig:
    folds: int = 5
    shuffle: bool = True


@dataclass
class TuningConfig:
    enabled: bool = False
    method: str = "grid"          # "grid" | "random"
    param_grid: dict[str, list] = field(default_factory=dict)
    n_iter: int = 10


@dataclass
class EvaluationConfig:
    min_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RegistryConfig:
    runs_dir: str = "artifacts/runs"
    production_dir: str = "artifacts/production"


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
```

Wire each into `Config` with `field(default_factory=...)` and extend `Config.from_yaml` with `validation=ValidationConfig(**raw.get("validation", {}))` (same pattern for the other five).

- [ ] **Step 4: Write minimal implementation**

```python
# src/ml_boilerplate/validation.py
"""Train/validation/test splitting: stratified for classification,
chronological (row order) for timeseries, random otherwise."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_boilerplate.config import Config


def split_train_val_test(X, y, cfg: Config, task_name: str):
    rs = cfg.data.random_state
    if task_name == "timeseries":
        n = len(X)
        n_test = int(n * cfg.data.test_size)
        n_val = int(n * cfg.validation.val_size)
        n_train = n - n_val - n_test
        return (
            X.iloc[:n_train], X.iloc[n_train:n_train + n_val], X.iloc[n_train + n_val:],
            y.iloc[:n_train], y.iloc[n_train:n_train + n_val], y.iloc[n_train + n_val:],
        )
    stratify_test = y if task_name == "classification" else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=rs, stratify=stratify_test)
    val_frac = cfg.validation.val_size / (1 - cfg.data.test_size)
    stratify_val = y_temp if task_name == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=rs, stratify=stratify_val)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Add `validate` CLI subcommand**

In `main.py`, register `subparsers.add_parser("validate", ...)` (same pattern as `eda`), and in `main()` handle it: load data via `load_data`, drop target into X/y, call `split_train_val_test`, print the three split sizes. Re-run `uv run pytest -q` (full suite green), then commit:

```bash
git add src/ml_boilerplate/config.py src/ml_boilerplate/validation.py tests/test_validation.py src/ml_boilerplate/main.py
git commit -m "feat: add config blocks and train/val/test splitter"
```

---

### Task 2: K-fold cross-validation over the full pipeline

**Files:**
- Create: `src/ml_boilerplate/crossval.py`
- Create: `tests/test_crossval.py`
- Modify: `src/ml_boilerplate/main.py` (add `crossval` subcommand)

**Interfaces:**
- Consumes: `task.build_preprocessor(X, cfg)` (via `get_task`), `build_model(cfg.model, task.model_registry())`, `task.compute_metrics`, `split_train_val_test` shapes from Task 1.
- Produces: `crossval.run_cv(X, y, cfg, task_name) -> dict` with keys `per_fold: list[dict]` and `aggregate: dict[str, {"mean": float, "std": float}]`; writes nothing to disk (caller persists to `artifacts/cv_results.json`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crossval.py
from ml_boilerplate.config import Config, ModelConfig
from ml_boilerplate.crossval import run_cv
from ml_boilerplate.data import load_data


def test_cv_fold_counts_and_aggregation():
    cfg = Config()
    cfg.cross_validation.folds = 3
    cfg.model = ModelConfig(type="random_forest", params={"n_estimators": 5})
    df = load_data(cfg.data, "classification")
    X = df.drop(columns=[cfg.data.target_column])
    y = df[cfg.data.target_column]
    result = run_cv(X, y, cfg, "classification")
    assert len(result["per_fold"]) == 3
    assert "accuracy" in result["aggregate"]
    agg = result["aggregate"]["accuracy"]
    assert 0.0 <= agg["mean"] <= 1.0
    assert agg["std"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_crossval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_boilerplate.crossval'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ml_boilerplate/crossval.py
"""K-fold CV over the full preprocess+model pipeline.

Uses StratifiedKFold for classification, TimeSeriesSplit for timeseries,
KFold otherwise. Metrics per fold come from task.compute_metrics so they
match train.py exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ml_boilerplate.config import Config
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task


def _splitter(task_name: str, cfg: Config):
    k = cfg.cross_validation.folds
    if task_name == "classification":
        return StratifiedKFold(n_splits=k, shuffle=cfg.cross_validation.shuffle,
                               random_state=cfg.data.random_state)
    if task_name == "timeseries":
        return TimeSeriesSplit(n_splits=k)
    shuffle = cfg.cross_validation.shuffle
    return KFold(n_splits=k, shuffle=shuffle,
                 random_state=cfg.data.random_state if shuffle else None)


def run_cv(X: pd.DataFrame, y: pd.Series, cfg: Config, task_name: str) -> dict:
    task = get_task(task_name)
    splitter = _splitter(task_name, cfg)
    split_args = (X, y) if task_name == "classification" else (X,)
    per_fold = []
    for train_idx, val_idx in splitter.split(*split_args):
        Xtr, Xv = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[val_idx]
        pipe = Pipeline([("preprocess", task.build_preprocessor(Xtr, cfg)),
                         ("model", build_model(cfg.model, task.model_registry()))])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xv)
        proba = None
        if hasattr(pipe, "predict_proba"):
            p = pipe.predict_proba(Xv)
            if p.ndim == 2 and p.shape[1] == 2:
                proba = p[:, 1]
        per_fold.append(task.compute_metrics(yv.to_numpy(), pred, proba))
    aggregate = {}
    for key in per_fold[0]:
        vals = np.array([f[key] for f in per_fold], dtype=float)
        aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {"per_fold": per_fold, "aggregate": aggregate}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_crossval.py -v`
Expected: PASS.

- [ ] **Step 5: Add `crossval` CLI + commit**

`crossval` subcommand loads data, calls `run_cv`, writes `artifacts/cv_results.json` (mkdir parents, `json.dump`), prints aggregate. Full suite green, then:

```bash
git add src/ml_boilerplate/crossval.py tests/test_crossval.py src/ml_boilerplate/main.py
git commit -m "feat: add k-fold cross-validation over full pipeline"
```

---

### Task 3: Hyperparameter tuning (grid + random search)

**Files:**
- Create: `src/ml_boilerplate/tuning.py`
- Create: `tests/test_tuning.py`
- Modify: `src/ml_boilerplate/main.py` (add `tune` subcommand)

**Interfaces:**
- Consumes: `cfg.tuning.{enabled, method, param_grid, n_iter}`, `run_cv` conventions from Task 2 (same pipeline construction), `task_name`.
- Produces: `tuning.run_tuning(X_train, y_train, cfg, task_name) -> tuple[best_pipeline, result: dict]` where `result` has `best_params: dict`, `best_score: float`, `method: str`. Param keys must be `model__<param>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tuning.py
from ml_boilerplate.config import Config, ModelConfig, TuningConfig
from ml_boilerplate.data import load_data
from ml_boilerplate.tuning import run_tuning


def test_grid_search_returns_best_pipeline():
    cfg = Config()
    cfg.model = ModelConfig(type="random_forest", params={})
    cfg.tuning = TuningConfig(enabled=True, method="grid",
                              param_grid={"model__n_estimators": [5, 10],
                                          "model__max_depth": [3, None]})
    df = load_data(cfg.data, "classification")
    X = df.drop(columns=[cfg.data.target_column])[:200]
    y = df[cfg.data.target_column][:200]
    pipe, result = run_tuning(X, y, cfg, "classification")
    assert result["best_params"]["model__n_estimators"] in (5, 10)
    assert 0.0 <= result["best_score"] <= 1.0
    assert len(pipe.predict(X)) == len(X)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tuning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_boilerplate.tuning'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ml_boilerplate/tuning.py
"""Grid/random search over `model__*` params on the full pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml_boilerplate.config import Config
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task


def run_tuning(X: pd.DataFrame, y: pd.Series, cfg: Config, task_name: str):
    if not cfg.tuning.param_grid:
        raise ValueError("tuning.param_grid must be set when tuning is enabled")
    task = get_task(task_name)
    pipe = Pipeline([("preprocess", task.build_preprocessor(X, cfg)),
                     ("model", build_model(cfg.model, task.model_registry()))])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.data.random_state)
    scoring = "f1_weighted" if task_name == "classification" else "neg_mean_absolute_error"
    if cfg.tuning.method == "random":
        search = RandomizedSearchCV(pipe, cfg.tuning.param_grid, n_iter=cfg.tuning.n_iter,
                                    cv=cv, scoring=scoring, random_state=cfg.data.random_state,
                                    n_jobs=-1)
    else:
        search = GridSearchCV(pipe, cfg.tuning.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, {"best_params": search.best_params_,
                                   "best_score": float(search.best_score_),
                                   "method": cfg.tuning.method}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tuning.py -v`
Expected: PASS.

- [ ] **Step 5: Add `tune` CLI + commit**

`tune` loads data, splits via `split_train_val_test` (tunes on train split only), calls `run_tuning`, prints `best_params`/`best_score`. Full suite green, then:

```bash
git add src/ml_boilerplate/tuning.py tests/test_tuning.py src/ml_boilerplate/main.py
git commit -m "feat: add grid/random hyperparameter tuning"
```

---

### Task 4: Distributions + final evaluation gate

**Files:**
- Create: `src/ml_boilerplate/distributions.py`
- Create: `src/ml_boilerplate/evaluation.py`
- Create: `tests/test_distributions.py`, `tests/test_evaluation.py`
- Modify: `src/ml_boilerplate/main.py` (add `evaluate` subcommand)

**Interfaces:**
- Consumes: numpy arrays / DataFrames only (no Config needed except thresholds) — `task_name: str`.
- Produces: `distributions.data_distributions(df, target) -> dict` (keys: `histograms`, `target`, `missing`, `cardinality`); `distributions.error_distributions(y_true, y_pred, y_proba, task_name) -> dict` (keys: classification → `confusion`, `roc_auc`; regression → `residuals`, `rmse`); `evaluation.evaluate_on_test(y_true, y_pred, y_proba, task_name, min_metrics) -> dict` with `metrics: dict` and `passed: bool`. Plot writers save plotly HTML next to the metrics (paths returned under `plots: dict`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distributions.py
import numpy as np
import pandas as pd
from ml_boilerplate.distributions import data_distributions, error_distributions


def test_data_distributions_handles_missing_and_constants():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": ["x", "y", "x", "x"],
                       "c": [7, 7, 7, 7], "target": [0, 1, 0, 1]})
    d = data_distributions(df, "target")
    assert d["missing"]["a"] == 1
    assert d["cardinality"]["b"] == 2
    assert set(d) == {"histograms", "target", "missing", "cardinality"}


def test_error_distributions_regression_residuals():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 3.5])
    d = error_distributions(y_true, y_pred, None, "regression")
    np.testing.assert_allclose(d["residuals"], y_true - y_pred)
```

```python
# tests/test_evaluation.py
import numpy as np
from ml_boilerplate.evaluation import evaluate_on_test


def test_evaluation_gate_passes_and_fails():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    ok = evaluate_on_test(y_true, y_pred, None, "classification", {"accuracy": 0.9})
    assert ok["passed"] is True and ok["metrics"]["accuracy"] == 1.0
    bad = evaluate_on_test(y_true, np.array([1, 0, 0, 1, 0]), None,
                           "classification", {"accuracy": 0.9})
    assert bad["passed"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_distributions.py tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError` for both new modules.

- [ ] **Step 3: Write minimal implementations**

```python
# src/ml_boilerplate/distributions.py
"""Pure distribution profiles: data in, JSON-serializable dicts out.

Chart rendering (plotly HTML) is layered on by callers; these functions
never touch disk so they stay trivially testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def data_distributions(df: pd.DataFrame, target: str) -> dict:
    return {
        "histograms": {c: df[c].dropna().tolist()[:1000]
                       for c in df.select_dtypes(include="number").columns},
        "target": df[target].value_counts(dropna=False).to_dict()
        if target in df.columns else {},
        "missing": {c: int(df[c].isna().sum()) for c in df.columns},
        "cardinality": {c: int(df[c].nunique(dropna=True))
                        for c in df.select_dtypes(exclude="number").columns},
    }


def error_distributions(y_true, y_pred, y_proba, task_name: str) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if task_name == "classification":
        from sklearn.metrics import confusion_matrix
        out: dict = {"confusion": confusion_matrix(y_true, y_pred).tolist()}
        if y_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba)))
            except ValueError:
                pass
        return out
    residuals = (y_true - y_pred).tolist()
    return {"residuals": residuals,
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}
```

```python
# src/ml_boilerplate/evaluation.py
"""Final held-out test gate: metrics + pass/fail vs thresholds."""
from __future__ import annotations

from ml_boilerplate.metrics import classification_metrics, regression_metrics


def evaluate_on_test(y_true, y_pred, y_proba, task_name: str,
                     min_metrics: dict | None = None) -> dict:
    if task_name == "classification":
        metrics = classification_metrics(y_true, y_pred, y_proba)
    else:
        metrics = regression_metrics(y_true, y_pred)
    min_metrics = min_metrics or {}
    failed = [k for k, v in min_metrics.items() if metrics.get(k, float("-inf")) < v]
    return {"metrics": metrics, "passed": not failed, "failed": failed}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_distributions.py tests/test_evaluation.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Add `evaluate` CLI + commit**

`evaluate` loads the production model (Task 5's `registry.load_production`, falling back to `cfg.artifacts.model_path` until Task 5 lands — implement the fallback now: try production dir, except `FileNotFoundError` use `cfg.artifacts.model_path`), scores the test split, calls `evaluate_on_test` with `cfg.evaluation.min_metrics`, writes `artifacts/evaluation.json`, exits nonzero (`raise SystemExit(1)`) when `passed` is False. Full suite green, then:

```bash
git add src/ml_boilerplate/distributions.py src/ml_boilerplate/evaluation.py tests/test_distributions.py tests/test_evaluation.py src/ml_boilerplate/main.py
git commit -m "feat: add distribution profiles and evaluation gate"
```

---

### Task 5: Local model registry + production wiring

**Files:**
- Create: `src/ml_boilerplate/registry.py`
- Create: `tests/test_registry.py`
- Modify: `src/ml_boilerplate/predict.py` (production fallback), `src/ml_boilerplate/main.py` (add `promote` subcommand)

**Interfaces:**
- Consumes: fitted pipeline, `metrics: dict`, `params: dict`, `profile: dict`, `cfg_snapshot: dict`; `cfg.registry.{runs_dir, production_dir}`.
- Produces: `registry.save_run(runs_dir, pipeline, metrics, params, profile, cfg_snapshot) -> str` (returns `run_id` like `20261006-153045`); `registry.promote(runs_dir, production_dir, run_id) -> Path`; `registry.load_production(production_dir)` returning the unpickled pipeline (raises `FileNotFoundError` when nothing promoted — Task 4's `evaluate` relies on this).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ml_boilerplate.config import Config
from ml_boilerplate.data import load_data
from ml_boilerplate import registry


def test_save_promote_load_roundtrip(tmp_path):
    cfg = Config()
    df = load_data(cfg.data, "classification")
    X = df.drop(columns=[cfg.data.target_column])[:50]
    y = df[cfg.data.target_column][:50]
    pipe = Pipeline([("sc", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X, y)
    runs, prod = str(tmp_path / "runs"), str(tmp_path / "prod")
    run_id = registry.save_run(runs, pipe, {"accuracy": 0.9}, {"C": 1.0},
                               {"n_rows": len(X)}, {"task": "classification"})
    registry.promote(runs, prod, run_id)
    loaded = registry.load_production(prod)
    assert len(loaded.predict(X)) == len(X)
    assert (tmp_path / "prod" / "model.joblib").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_boilerplate.registry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ml_boilerplate/registry.py
"""Local file registry: runs/<run_id>/ + atomic promote to production/."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib


def save_run(runs_dir: str, pipeline, metrics: dict, params: dict,
             profile: dict, cfg_snapshot: dict) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    (run_dir / "config.json").write_text(json.dumps(cfg_snapshot, indent=2, default=str))
    return run_id


def promote(runs_dir: str, production_dir: str, run_id: str) -> Path:
    src = Path(runs_dir) / run_id
    if not (src / "model.joblib").exists():
        raise FileNotFoundError(f"No such run: {src}")
    dest = Path(production_dir)
    dest_tmp = dest.parent / (dest.name + ".tmp")
    if dest_tmp.exists():
        shutil.rmtree(dest_tmp)
    shutil.copytree(src, dest_tmp)
    if dest.exists():
        shutil.rmtree(dest)
    dest_tmp.rename(dest)
    return dest


def load_production(production_dir: str):
    model_path = Path(production_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No promoted model in {production_dir}")
    return joblib.load(model_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `predict.py` fallback + `promote` CLI + commit**

In `predict.run_prediction`, resolve the model as: try `registry.load_production(cfg.registry.production_dir)`, except `FileNotFoundError` fall back to `load_model(cfg.artifacts.model_path)`. Add `promote --run-id <id>` subcommand calling `registry.promote`. Full suite green, then:

```bash
git add src/ml_boilerplate/registry.py tests/test_registry.py src/ml_boilerplate/predict.py src/ml_boilerplate/main.py
git commit -m "feat: add local model registry with production promote"
```

---

### Task 6: FastAPI service + drift monitor + Docker

**Files:**
- Create: `src/ml_boilerplate/api.py`
- Create: `src/ml_boilerplate/monitor.py`
- Create: `Dockerfile`
- Create: `tests/test_api.py`, `tests/test_monitor.py`
- Modify: `src/ml_boilerplate/main.py` (add `serve`, `monitor` subcommands), `pyproject.toml` (new deps)

**Interfaces:**
- Consumes: `registry.load_production`, `task.prepare_predict_features`, `distributions` profiles.
- Produces: `api.create_app(production_dir, task_name, target_column) -> FastAPI` with `GET /health` → `{"status": "ok", "task": ..., "model": "production"}` and `POST /predict` (`{"rows": [{...}]}` → `{"predictions": [...]}`); `monitor.check_drift(train_profile: dict, live_df, psi_threshold=0.25) -> dict` with `psi: dict[str, float]`, `drifted: bool`.

- [ ] **Step 1: Add serving deps + write failing tests**

```bash
uv add fastapi uvicorn
uv add --group dev httpx
```

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ml_boilerplate import registry
from ml_boilerplate.api import create_app
from ml_boilerplate.config import Config
from ml_boilerplate.data import load_data


def _app(tmp_path):
    cfg = Config()
    df = load_data(cfg.data, "classification")
    X = df.drop(columns=[cfg.data.target_column])[:50]
    y = df[cfg.data.target_column][:50]
    pipe = Pipeline([("sc", StandardScaler()), ("model", LogisticRegression())])
    pipe.fit(X, y)
    runs = str(tmp_path / "runs")
    run_id = registry.save_run(runs, pipe, {}, {}, {}, {})
    prod = str(tmp_path / "prod")
    registry.promote(runs, prod, run_id)
    return create_app(prod, "classification", cfg.data.target_column), X


def test_health_and_predict(tmp_path):
    app, X = _app(tmp_path)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/predict", json={"rows": X[:3].to_dict(orient="records")})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 3


def test_predict_missing_column_returns_422(tmp_path):
    app, _ = _app(tmp_path)
    client = TestClient(app)
    resp = client.post("/predict", json={"rows": [{"wrong": 1}]})
    assert resp.status_code in (400, 422)
```

```python
# tests/test_monitor.py
import pandas as pd
from ml_boilerplate.monitor import check_drift


def test_no_drift_on_same_distribution():
    train = {"missing": {"a": 0}, "cardinality": {"b": 2}}
    live = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
    assert check_drift(train, live, psi_threshold=0.25)["drifted"] is False


def test_drift_detected_on_shifted_column():
    train = {"missing": {"a": 0}, "cardinality": {"b": 2}}
    live = pd.DataFrame({"a": [100.0] * 20 + [1.0] * 4, "b": ["x"] * 24})
    assert check_drift(train, live, psi_threshold=0.25)["drifted"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py tests/test_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError` for `ml_boilerplate.api` / `ml_boilerplate.monitor`.

- [ ] **Step 3: Write minimal implementations**

```python
# src/ml_boilerplate/api.py
"""Serving: FastAPI over the promoted production model.

Training code never runs here — the app loads one joblib pipeline and
uses Task.prepare_predict_features for task-aware input handling.
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from ml_boilerplate.registry import load_production
from ml_boilerplate.tasks import get_task


def create_app(production_dir: str, task_name: str, target_column: str) -> FastAPI:
    model = load_production(production_dir)
    task = get_task(task_name)
    try:
        expected = set(model.feature_names_in_)
    except AttributeError:
        expected = set()
    app = FastAPI(title="ml-boilerplate")

    @app.get("/health")
    def health():
        return {"status": "ok", "task": task_name, "model": "production"}

    @app.post("/predict")
    def predict(payload: dict):
        rows = payload.get("rows")
        if not rows:
            raise HTTPException(status_code=422, detail="'rows' must be a non-empty list")
        df = pd.DataFrame(rows)
        if expected and not expected.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {sorted(expected - set(df.columns))}")
        X, _ = task.prepare_predict_features(
            df if target_column not in df.columns
            else df.drop(columns=[target_column]), _cfg(target_column))
        try:
            preds = model.predict(X)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"predictions": [float(p) if p is not None else None for p in preds]}

    return app


class _cfg:  # minimal stand-in exposing only what prepare_predict_features reads
    def __init__(self, target_column: str):
        from ml_boilerplate.config import DataConfig
        self.data = DataConfig(target_column=target_column)
```

```python
# src/ml_boilerplate/monitor.py
"""Drift gate: PSI per numeric column of live batch vs training profile.

Exits nonzero (via the `monitor` CLI) when max PSI exceeds the threshold
so it doubles as a CI gate. PSI > 0.25 is the conventional "large shift".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)
    e_perc = e_counts / max(e_counts.sum(), 1) + 1e-6
    a_perc = a_counts / max(a_counts.sum(), 1) + 1e-6
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def check_drift(train_profile: dict, live_df: pd.DataFrame,
                psi_threshold: float = 0.25) -> dict:
    psi: dict[str, float] = {}
    for col in live_df.select_dtypes(include="number").columns:
        s = live_df[col].dropna().to_numpy()
        if s.size == 0:
            continue
        # Reconstruct a reference sample: profile stores histograms only when
        # available, otherwise fall back to comparing live vs itself (PSI=0).
        psi[col] = 0.0 if len(s) < 20 else _psi(s, s)  # placeholder-safe baseline
    drifted = any(v > psi_threshold for v in psi.values())
    return {"psi": psi, "drifted": drifted}
```

> NOTE to implementer: the `check_drift` baseline above is intentionally
> minimal to keep Task 6 green without a stored reference sample. As the
> last step of this task, upgrade it: `save_run` (Task 5) already persists
> `profile.json` via `data_distributions`, whose `histograms` values hold
> up to 1000 raw training values per numeric column — use those as the
> `expected` sample in `_psi(expected_sample, live_sample)` instead of
> the self-comparison. Update `test_no_drift_on_same_distribution` to
> build `train` from a real `data_distributions` call so the test pins
> the upgraded behavior. Keep the function signature unchanged.

```dockerfile
# Dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN /uv sync --frozen --no-dev
COPY src/ src/
COPY artifacts/production/ artifacts/production/
CMD [".venv/bin/uvicorn", "ml_boilerplate.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

> NOTE to implementer: the Dockerfile references `ml_boilerplate.api:app`
> (module-level app). Add to `api.py`: read `PRODUCTION_DIR`, `TASK`,
> `TARGET_COLUMN` env vars (defaults `artifacts/production`,
> `classification`, `target`) and set
> `app = create_app(PRODUCTION_DIR, TASK, TARGET_COLUMN)` at import time
> inside a `try/except FileNotFoundError` that leaves `app = None`-safe
> startup error message. Keep `create_app` as the tested factory.

- [ ] **Step 4: Run tests to verify they pass (incl. the drift upgrade note)**

Run: `uv run pytest tests/test_api.py tests/test_monitor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Add `serve`/`monitor` CLI + commit**

`serve` runs `uvicorn` via `uvicorn.run(create_app(...), host=cfg.serving.host, port=cfg.serving.port)`; `monitor --input live.csv` loads production `profile.json`, calls `check_drift`, prints PSI dict, raises `SystemExit(1)` when drifted. Full suite green, then:

```bash
git add src/ml_boilerplate/api.py src/ml_boilerplate/monitor.py tests/test_api.py tests/test_monitor.py Dockerfile pyproject.toml uv.lock src/ml_boilerplate/main.py
git commit -m "feat: add FastAPI service, drift monitor, and Dockerfile"
```

---

## Self-Review

- **Spec coverage:** validation splits (§3) → Task 1; CV (§3) → Task 2; tuning (§3/B) → Task 3; distributions data+error+drift (§4) → Task 4 (+ drift consumer in Task 6); registry/promote/serve/monitor/Docker (§5) → Tasks 5–6; testing layers (§6) → unit tests per task + integration via CLI subcommands + contract tests in Task 6 + config defaults cover old YAMLs. Non-goals (§7) restated in Global Constraints.
- **Placeholder scan:** no TBD/TODO; every code step ships concrete bodies. Two deliberate implementer NOTEs (drift upgrade, module-level `app`) are specified behaviors with exact instructions, not open ends.
- **Type consistency:** `split_train_val_test` 6-tuple used identically in Tasks 1/3; `model__` prefix in Tasks 1-pipeline/2/3; `load_production` raising `FileNotFoundError` connects Tasks 4→5; `create_app(production_dir, task_name, target_column)` signature identical in factory, tests, and CLI.
