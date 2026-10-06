"""CLI entrypoint.

    python -m ml_boilerplate.main --config configs/classification.yaml train
    python -m ml_boilerplate.main --config configs/classification.yaml predict --input new.csv --output preds.csv
    python -m ml_boilerplate.main --config configs/regression.yaml eda

`--config`/`-v` work whether given before or after the subcommand. Note:
argparse subparsers parse into a *fresh* namespace and copy every one of
their own attributes onto the parent's, so if these flags carried a real
default on the subparser copies too, that default would silently clobber
whatever the top-level parser already captured. To avoid that, the shared
copies use `default=SUPPRESS` (so an unset flag leaves nothing to copy)
and the real default is applied once, explicitly, in `main()`.
"""

from __future__ import annotations

import argparse
import logging

from ml_boilerplate.config import Config
from ml_boilerplate.data import load_data
from ml_boilerplate.eda import run_eda
from ml_boilerplate.predict import run_prediction
from ml_boilerplate.train import run_training

DEFAULT_CONFIG_PATH = "configs/classification.yaml"


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=argparse.SUPPRESS, help="Path to YAML config file"
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="Enable debug logging",
    )

    parser = argparse.ArgumentParser(prog="ml-boilerplate", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "train", help="Train a model and save it + its metrics + result charts", parents=[common]
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Score new data with a saved model", parents=[common]
    )
    predict_parser.add_argument(
        "--input", required=True, help="CSV/Parquet file (local path or http(s) URL) with feature columns"
    )
    predict_parser.add_argument("--output", required=True, help="Where to write predictions (csv or parquet)")

    subparsers.add_parser(
        "eda", help="Profile the data and render EDA charts, without training", parents=[common]
    )

    subparsers.add_parser(
        "validate", help="Show train/validation/test split sizes", parents=[common]
    )

    subparsers.add_parser(
        "crossval", help="Run k-fold CV over the full pipeline", parents=[common]
    )

    subparsers.add_parser(
        "tune", help="Tune hyperparameters via grid/random search", parents=[common]
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = getattr(args, "config", DEFAULT_CONFIG_PATH)
    verbose = getattr(args, "verbose", False)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = Config.from_yaml(config_path)

    if args.command == "train":
        metrics = run_training(cfg)
        print(metrics)
    elif args.command == "predict":
        output_path = run_prediction(cfg, args.input, args.output)
        print(f"Wrote predictions to {output_path}")
    elif args.command == "eda":
        df = load_data(cfg.data, cfg.task)
        paths = run_eda(df, cfg.eda, report_path=f"{cfg.eda.output_dir}/eda_report.json")
        print(paths)
    elif args.command == "validate":
        from ml_boilerplate.validation import split_train_val_test

        df = load_data(cfg.data, cfg.task)
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        X_train, X_val, X_test, _, _, _ = split_train_val_test(X, y, cfg, cfg.task)
        print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    elif args.command == "crossval":
        import json
        from pathlib import Path

        from ml_boilerplate.crossval import run_cv

        df = load_data(cfg.data, cfg.task)
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        result = run_cv(X, y, cfg, cfg.task)
        out_path = Path("artifacts/cv_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(result["aggregate"])
    elif args.command == "tune":
        from ml_boilerplate.tuning import run_tuning
        from ml_boilerplate.validation import split_train_val_test

        df = load_data(cfg.data, cfg.task)
        X = df.drop(columns=[cfg.data.target_column])
        y = df[cfg.data.target_column]
        X_train, _, _, y_train, _, _ = split_train_val_test(X, y, cfg, cfg.task)
        _, result = run_tuning(X_train, y_train, cfg, cfg.task)
        print(result["best_params"])
        print(result["best_score"])


if __name__ == "__main__":
    main()
