"""Visualization helpers for the project."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_action_histograms(y, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(y[:, 0], bins=30, alpha=0.7, label="Throttle")
    plt.hist(y[:, 1], bins=30, alpha=0.7, label="Steering")
    plt.xlabel("Action value")
    plt.ylabel("Frequency")
    plt.title("Action distribution in training data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_heading_vs_steering(x, y, out_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(x[:, 5], y[:, 1], s=8, alpha=0.45)
    plt.xlabel("Heading error")
    plt.ylabel("Steering")
    plt.title("Heading error vs steering")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_multi_run_paths(runs, out_path):
    plt.figure(figsize=(7, 7))
    for r in runs:
        plt.plot(r["track"][:, 0], r["track"][:, 1], linewidth=1)
        plt.plot(r["path"][:, 0], r["path"][:, 1], linewidth=1.4, label=f"seed {r['seed']}")
    plt.axis("equal")
    plt.title("Benchmark paths across seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_checkpoint_progress(runs, out_path):
    labels = [str(r["seed"]) for r in runs]
    cps = [r["max_cp"] for r in runs]
    crashes = [r["crashes"] for r in runs]
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.18, cps, width=0.36, label="checkpoints")
    plt.bar(x + 0.18, crashes, width=0.36, label="crashes/100 steps")
    plt.xticks(x, labels)
    plt.xlabel("Seed")
    plt.title("Progress and stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_path_overlay(data_path, runs, out_path):
    d = np.load(data_path)
    expert_xy = d["positions"]
    plt.figure(figsize=(7, 7))
    plt.scatter(expert_xy[:, 0], expert_xy[:, 1], s=3, alpha=0.25, label="expert data")
    for r in runs:
        plt.plot(r["path"][:, 0], r["path"][:, 1], linewidth=1.2, label=f"model seed {r['seed']}")
    plt.axis("equal")
    plt.title("Expert data vs model trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_iteration_history(benchmark_dir: str | Path, out_path):
    benchmark_dir = Path(benchmark_dir)
    rows = []
    for p in sorted(benchmark_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows.append((obj["tag"], obj["summary"]["mean_checkpoints"], obj["summary"]["mean_crashes"]))
    if not rows:
        return
    tags, cps, crashes = zip(*rows)
    x = np.arange(len(tags))
    plt.figure(figsize=(9, 5))
    plt.plot(x, cps, marker="o", label="mean checkpoints")
    plt.plot(x, crashes, marker="o", label="mean crashes")
    plt.xticks(x, tags, rotation=30, ha="right")
    plt.title("Iteration history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
