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


if __name__ == "__main__":
    main()
