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
class ValidationConfig:
    val_size: float = 0.2


@dataclass
class CrossValidationConfig:
    folds: int = 5
    shuffle: bool = True


@dataclass
class TuningConfig:
    enabled: bool = False
    method: str = "grid"          # "grid" | "random"
    param_grid: dict[str, list] = field(default_factory=dict)
    n_iter: int = 10


@dataclass
class EvaluationConfig:
    min_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RegistryConfig:
    runs_dir: str = "artifacts/runs"
    production_dir: str = "artifacts/production"


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


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
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

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
            validation=ValidationConfig(**raw.get("validation", {})),
            cross_validation=CrossValidationConfig(**raw.get("cross_validation", {})),
            tuning=TuningConfig(**raw.get("tuning", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            registry=RegistryConfig(**raw.get("registry", {})),
            serving=ServingConfig(**raw.get("serving", {})),
        )
