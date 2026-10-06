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
