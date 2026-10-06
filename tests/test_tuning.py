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


def test_grid_search_works_for_regression():
    cfg = Config()
    cfg.cross_validation.folds = 3
    cfg.model = ModelConfig(type="random_forest", params={})
    cfg.tuning = TuningConfig(enabled=True, method="grid",
                              param_grid={"model__n_estimators": [5, 10]})
    df = load_data(cfg.data, "regression")
    X = df.drop(columns=[cfg.data.target_column])[:200]
    y = df[cfg.data.target_column][:200]
    pipe, result = run_tuning(X, y, cfg, "regression")
    assert result["best_params"]["model__n_estimators"] in (5, 10)
    assert isinstance(result["best_score"], float)
    assert len(pipe.predict(X)) == len(X)
