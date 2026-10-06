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
