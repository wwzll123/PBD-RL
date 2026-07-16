#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def as_bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"Missing required config value: {label}")
    return Path(value).expanduser().resolve()


def collect_inputs(cfg: dict[str, Any], validate: bool = True) -> list[Path]:
    input_cfg = cfg.get("input", {})
    input_file = input_cfg.get("input_file")
    input_dir = input_cfg.get("input_dir")
    glob_pattern = input_cfg.get("glob", "*.yaml")
    if input_file and input_dir:
        raise ValueError("Set only one of input.input_file or input.input_dir")
    if input_file:
        path = require_path(input_file, "input.input_file")
        if validate and not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return [path]
    if input_dir:
        path = require_path(input_dir, "input.input_dir")
        if validate and not path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {path}")
        if not validate:
            return [path / str(glob_pattern)]
        files = sorted(p for p in path.glob(str(glob_pattern)) if p.is_file())
        if not files:
            raise FileNotFoundError(f"No input files matched {glob_pattern!r} in {path}")
        return files
    raise ValueError("Set input.input_file or input.input_dir")


def build_command(cfg: dict[str, Any], input_path: Path, output_dir: Path, validate: bool = True) -> list[str]:
    px_cfg = cfg.get("pxdesign", {})
    sampling = cfg.get("sampling", {})
    runtime = cfg.get("runtime", {})

    checkpoint_dir = require_path(px_cfg.get("load_checkpoint_dir"), "pxdesign.load_checkpoint_dir")
    model_name = str(px_cfg.get("model_name", "final"))
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    if validate and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cmd = [
        "pxdesign",
        "infer",
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--dtype",
        str(sampling.get("dtype", "bf16")),
        "--N_sample",
        str(sampling.get("n_sample", 1)),
        "--N_step",
        str(sampling.get("n_step", 400)),
        "--eta_type",
        str(sampling.get("eta_type", "const")),
        "--eta_min",
        str(sampling.get("eta_min", 2.5)),
        "--eta_max",
        str(sampling.get("eta_max", 2.5)),
        "--model_name",
        model_name,
        "--load_checkpoint_dir",
        str(checkpoint_dir),
        "--load_strict",
        as_bool_text(px_cfg.get("load_strict", True)),
        "--num_workers",
        str(runtime.get("num_workers", 0)),
        "--use_msa",
        as_bool_text(runtime.get("use_msa", True)),
        "--use_fast_ln",
        as_bool_text(runtime.get("use_fast_ln", True)),
    ]

    seeds = sampling.get("seeds") or []
    if seeds:
        cmd.extend(["--seeds", ",".join(str(seed) for seed in seeds)])
    extra_args = runtime.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise TypeError("runtime.extra_args must be a list")
    cmd.extend(str(x) for x in extra_args)
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PXDesign inference with a config-selected checkpoint.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/inference.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    output_cfg = cfg.get("output", {})
    base_out = require_path(output_cfg.get("dump_dir"), "output.dump_dir")
    overwrite = bool(output_cfg.get("overwrite", False))
    validate_paths = not args.dry_run
    inputs = collect_inputs(cfg, validate=validate_paths)

    env = os.environ.copy()
    cuda_visible_devices = cfg.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    for input_path in inputs:
        out_dir = base_out if len(inputs) == 1 else base_out / input_path.stem
        if validate_paths and out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory is non-empty. Set output.overwrite=true: {out_dir}")
        if validate_paths:
            out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(cfg, input_path, out_dir, validate=validate_paths)
        print("[cmd]", " ".join(shlex.quote(part) for part in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
