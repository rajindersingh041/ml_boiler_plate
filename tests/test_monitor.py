import pandas as pd
from ml_boilerplate.distributions import data_distributions
from ml_boilerplate.monitor import check_drift


def _train_base():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0] * 25,
            "b": ["x", "y"] * 50,
        }
    )


def test_no_drift_on_same_distribution():
    train = data_distributions(_train_base(), target="b")
    live = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0] * 6,
            "b": ["x", "y"] * 12,
        }
    )
    assert check_drift(train, live, psi_threshold=0.25)["drifted"] is False


def test_drift_detected_on_shifted_column():
    train = data_distributions(_train_base(), target="b")
    live = pd.DataFrame({"a": [100.0] * 20 + [1.0] * 4, "b": ["x"] * 24})
    assert check_drift(train, live, psi_threshold=0.25)["drifted"] is True
