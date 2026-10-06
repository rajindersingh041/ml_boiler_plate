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
