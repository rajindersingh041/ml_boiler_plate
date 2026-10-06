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
