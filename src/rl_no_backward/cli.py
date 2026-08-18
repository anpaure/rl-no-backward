"""Command-line interface for experiments, diagnostics, and plotting."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .experiment import ExperimentConfig, run_benchmark
from .gsm8k_experiment import GSM8KExperimentConfig, run_gsm8k_benchmark


def _load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, dict):
        raise TypeError("configuration root must be a mapping")
    return ExperimentConfig.from_mapping(mapping)


def _load_gsm8k_config(path: str | Path) -> GSM8KExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, dict):
        raise TypeError("configuration root must be a mapping")
    return GSM8KExperimentConfig.from_mapping(mapping)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run configured training methods")
    benchmark.add_argument("--config", required=True, help="YAML experiment config")
    benchmark.add_argument("--output", required=True, help="artifact output directory")
    gsm8k = subparsers.add_parser("gsm8k", help="run the small GSM8K RLVR benchmark")
    gsm8k.add_argument("--config", required=True, help="YAML GSM8K config")
    gsm8k.add_argument("--output", required=True, help="artifact output directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        output = run_benchmark(_load_config(args.config), args.output)
        print(output.resolve())
    elif args.command == "gsm8k":
        output = run_gsm8k_benchmark(_load_gsm8k_config(args.config), args.output)
        print(output.resolve())


if __name__ == "__main__":
    main()
