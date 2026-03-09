"""CLI entrypoint for training the RIR project model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rir_project.trainer import RIRTrainer, TrainingConfig


def _str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _field_type_for_arg(field_type: Any):
    origin = get_origin(field_type)
    if origin is Union:
        args = [a for a in get_args(field_type) if a is not type(None)]
        if len(args) == 1:
            return _field_type_for_arg(args[0])
    if field_type is bool:
        return _str_to_bool
    if field_type in (int, float, str):
        return field_type
    return str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Physics-Informed RIR model")
    parser.add_argument("--hf-cache-dir", dest="hf_cache_dir", type=str, default=None,
                        help="local HuggingFace datasets cache directory (e.g. drive mount)")

    for f in fields(TrainingConfig):
        arg_name = f"--{f.name.replace('_', '-')}"
        if arg_name == "--hf-cache-dir":
            continue  # already added above
        arg_type = _field_type_for_arg(f.type)
        parser.add_argument(
            arg_name,
            dest=f.name,
            type=arg_type,
            default=f.default,
            help=f"TrainingConfig.{f.name} (default: {f.default})",
        )

    return parser


def _coerce_optional_seed(cfg_dict: dict[str, Any]) -> None:
    # Accept --seed none/null for convenience.
    seed_val = cfg_dict.get("seed")
    if isinstance(seed_val, str) and seed_val.strip().lower() in {"none", "null"}:
        cfg_dict["seed"] = None


def main() -> dict[str, list]:
    parser = build_parser()
    args = parser.parse_args()
    cfg_dict = vars(args)
    # normalize empty string -> None
    if cfg_dict.get("hf_cache_dir") == "":
        cfg_dict["hf_cache_dir"] = None
    _coerce_optional_seed(cfg_dict)

    cfg = TrainingConfig(**cfg_dict)
    trainer = RIRTrainer(cfg)
    history = trainer.fit()

    print("[train.py] Completed training run")
    print(json.dumps({"config": cfg_dict, "history": history}, indent=2, default=float))
    return history


if __name__ == "__main__":
    history = main()
