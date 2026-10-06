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
