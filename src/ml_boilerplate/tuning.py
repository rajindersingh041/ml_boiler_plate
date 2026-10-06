"""Grid/random search over `model__*` params on the full pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ml_boilerplate.config import Config
from ml_boilerplate.model import build_model
from ml_boilerplate.tasks import get_task
from ml_boilerplate.crossval import get_splitter


def run_tuning(X: pd.DataFrame, y: pd.Series, cfg: Config, task_name: str):
    if not cfg.tuning.param_grid:
        raise ValueError("tuning.param_grid must be set when tuning is enabled")
    task = get_task(task_name)
    pipe = Pipeline([("preprocess", task.build_preprocessor(X, cfg)),
                     ("model", build_model(cfg.model, task.model_registry()))])
    cv = get_splitter(task_name, cfg)
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
