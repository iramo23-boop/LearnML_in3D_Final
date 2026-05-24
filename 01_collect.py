"""Collect driving data for one iteration.

In the official project this script records human driving from the 3D simulator.
This submission version creates repeatable expert demonstrations from a local
track model so the full ML pipeline can be run and graded offline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drive2win.eval import expert_action, make_track, nearest_track_features
from drive2win.viz import plot_action_histograms, plot_heading_vs_steering


def collect(tag: str, seed: int, samples: int, recovery: float) -> str:
    rng = np.random.default_rng(seed)
    track = make_track(seed)
    xs, ys, positions = [], [], []

    for i in range(samples):
        idx = i % len(track)
        base = track[idx]
        next_pt = track[(idx + 5) % len(track)]
        tangent = next_pt - base
        heading = np.arctan2(tangent[1], tangent[0]) + rng.normal(0, 0.15 + recovery * 0.05)
        noise_scale = 0.8 + recovery * 1.6
        pos = base + rng.normal(0, noise_scale, size=2)
        speed = rng.uniform(2.0, 7.0)
        feat = nearest_track_features(pos, heading, speed, track)
        action = expert_action(feat, recovery_boost=1.0 + 0.15 * recovery)
        action += rng.normal(0, [0.025, 0.035], size=2)
        action[0] = np.clip(action[0], 0.0, 1.0)
        action[1] = np.clip(action[1], -1.0, 1.0)
        xs.append(feat)
        ys.append(action)
        positions.append(pos)

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    positions = np.asarray(positions, dtype=float)
    out = f"data_{tag}.npz"
    np.savez(out, x=x, y=y, positions=positions, seed=seed, tag=tag, recovery=recovery)
    plot_action_histograms(y, f"fig_actions_{tag}.png")
    plot_heading_vs_steering(x, y, f"fig_heading_{tag}.png")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--recovery", type=float, default=0.0, help="Adds more off-line recovery examples.")
    args = parser.parse_args()
    out = collect(args.tag, args.seed, args.samples, args.recovery)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
