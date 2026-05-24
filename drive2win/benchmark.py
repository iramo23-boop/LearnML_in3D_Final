"""Canonical local benchmark wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .eval import run_policy, score_runs
from .viz import plot_checkpoint_progress, plot_multi_run_paths, plot_path_overlay


def run_benchmark(tag: str, model_path: str, data_path: str, seeds: list[int], out_dir: str = "benchmarks") -> dict:
    Path(out_dir).mkdir(exist_ok=True)
    package = dict(np.load(model_path, allow_pickle=True))
    runs = [run_policy(package, seed=s) for s in seeds]
    summary = score_runs(runs)

    plot_multi_run_paths(runs, Path(out_dir) / f"{tag}_paths.png")
    plot_checkpoint_progress(runs, Path(out_dir) / f"{tag}_progress.png")
    plot_path_overlay(data_path, runs, Path(out_dir) / f"{tag}_overlay.png")

    result = {
        "tag": tag,
        "model_path": model_path,
        "data_path": data_path,
        "seeds": seeds,
        "summary": summary,
        "runs": [{k: v for k, v in r.items() if k not in ["path", "track"]} for r in runs],
    }
    with open(Path(out_dir) / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
