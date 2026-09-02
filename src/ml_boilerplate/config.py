"""Configuration loading and dataclasses.

Keeps every tunable knob in one place, loaded from a YAML file so
experiments are reproducible and diffable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"          # "synthetic" | "file"
    path: str | None = None            # local path or http(s) URL, when source == "file"
    format: str | None = None          # "csv" | "parquet" | None (infer from extension)
    target_column: str = "target"
    date_column: str | None = None     # required for task == "timeseries"
    n_lags: list[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14])
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FeatureConfig:
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    scale_numeric: bool = True
    missing_strategy_numeric: str = "median"       # passed to sklearn SimpleImputer
    missing_strategy_categorical: str = "most_frequent"


@dataclass
class ModelConfig:
    type: str = "random_forest"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    cv_folds: int = 5


@dataclass
class ArtifactConfig:
    model_path: str = "artifacts/model.joblib"
    metrics_path: str = "artifacts/metrics.json"
    plots_dir: str = "artifacts/plots"


@dataclass
class EdaConfig:
    enabled: bool = True
    output_dir: str = "artifacts/plots"


@dataclass
class OutputConfig:
    format: str | None = None          # "csv" | "parquet" | None (infer from output path)


@dataclass
class Config:
    task: str = "classification"       # "classification" | "regression" | "timeseries"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    eda: EdaConfig = field(default_factory=EdaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            task=raw.get("task", "classification"),
            data=DataConfig(**raw.get("data", {})),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            artifacts=ArtifactConfig(**raw.get("artifacts", {})),
            eda=EdaConfig(**raw.get("eda", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
