# src/ml_boilerplate/registry.py
"""Local file registry: runs/<run_id>/ + atomic promote to production/."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib


def save_run(runs_dir: str, pipeline, metrics: dict, params: dict,
             profile: dict, cfg_snapshot: dict) -> str:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))
    (run_dir / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    (run_dir / "config.json").write_text(json.dumps(cfg_snapshot, indent=2, default=str))
    return run_id


def promote(runs_dir: str, production_dir: str, run_id: str) -> Path:
    src = Path(runs_dir) / run_id
    if not (src / "model.joblib").exists():
        raise FileNotFoundError(f"No such run: {src}")
    dest = Path(production_dir)
    dest_tmp = dest.parent / (dest.name + ".tmp")
    if dest_tmp.exists():
        shutil.rmtree(dest_tmp)
    shutil.copytree(src, dest_tmp)
    if dest.exists():
        shutil.rmtree(dest)
    dest_tmp.rename(dest)
    return dest


def load_production(production_dir: str):
    model_path = Path(production_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No promoted model in {production_dir}")
    return joblib.load(model_path)
