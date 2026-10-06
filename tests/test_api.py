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
