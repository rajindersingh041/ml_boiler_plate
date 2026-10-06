"""Unit tests: classification metrics handle binary and multiclass targets."""

import numpy as np

from ml_boilerplate.metrics import classification_metrics


def test_binary_uses_binary_average():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    m = classification_metrics(y_true, y_pred, np.array([0.1, 0.9, 0.4, 0.2, 0.8]))
    assert m["accuracy"] == 0.8
    assert m["precision"] == 1.0  # binary average: 2/2
    assert "roc_auc" in m


def test_multiclass_uses_weighted_average():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 4 / 6
    assert 0.0 <= m["precision"] <= 1.0
    assert "roc_auc" not in m  # no binary probabilities available
