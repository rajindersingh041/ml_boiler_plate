"""Task registry: resolves config.task -> a Task instance."""

from __future__ import annotations

from ml_boilerplate.tasks.base import Task
from ml_boilerplate.tasks.classification import ClassificationTask
from ml_boilerplate.tasks.regression import RegressionTask
from ml_boilerplate.tasks.timeseries import TimeSeriesTask

_REGISTRY: dict[str, type[Task]] = {
    "classification": ClassificationTask,
    "regression": RegressionTask,
    "timeseries": TimeSeriesTask,
}


def get_task(name: str) -> Task:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown task {name!r}. Available: {list(_REGISTRY)}") from exc


__all__ = ["Task", "ClassificationTask", "RegressionTask", "TimeSeriesTask", "get_task"]
