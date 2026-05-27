"""Lightweight evaluator used for repeatable local benchmarks."""
from __future__ import annotations

import numpy as np

from .normalize import inverse_actions, transform
from .nn import forward


def make_track(seed: int, n_points: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n_points)
    radius = 18 + 2.5 * np.sin(3 * t + rng.uniform(-1, 1)) + 1.2 * np.cos(5 * t)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def nearest_track_features(pos: np.ndarray, heading: float, speed: float, track: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(track - pos, axis=1)
    idx = int(np.argmin(d))
    next_idx = (idx + 6) % len(track)
    target = track[next_idx]
    vec = target - pos
    target_angle = np.arctan2(vec[1], vec[0])
    heading_error = np.arctan2(np.sin(target_angle - heading), np.cos(target_angle - heading))
    dist_center = float(d[idx])
    progress = idx / len(track)
    # 12 features, matching project style: rays/speed/heading/progress-like inputs
    return np.array([
        pos[0] / 25, pos[1] / 25, np.sin(heading), np.cos(heading), speed / 10,
        heading_error, dist_center / 10, progress,
        np.sin(2 * np.pi * progress), np.cos(2 * np.pi * progress), vec[0] / 25, vec[1] / 25,
    ], dtype=float)


def expert_action(features: np.ndarray, recovery_boost: float = 1.0) -> np.ndarray:
    heading_error = features[5]
    dist_center = features[6]
    throttle = 0.72 - 0.25 * abs(heading_error) - 0.08 * dist_center
    steering = 1.25 * heading_error * recovery_boost
    return inverse_actions(np.array([[throttle, steering]], dtype=float))[0]


def run_policy(package: dict, seed: int, steps: int = 650, smoothing: float = 0.15) -> dict:
    track = make_track(seed)
    pos = track[0].copy() + np.array([0.0, -1.0])
    heading = 0.15
    speed = 1.0
    path = []
    checkpoints = set()
    crashes = 0
    prev_action = np.array([0.5, 0.0])
    stats = package["normalizer"].item() if hasattr(package["normalizer"], "item") else package["normalizer"]
    weights = {k: package[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}

    for _ in range(steps):
        feat = nearest_track_features(pos, heading, speed, track)[None, :]
        pred = forward(transform(feat, stats), weights)[0]
        action = inverse_actions(pred[None, :])[0]
        action = (1 - smoothing) * action + smoothing * prev_action
        prev_action = action

        throttle, steering = float(action[0]), float(action[1])
        heading += 0.07 * steering
        speed = 0.90 * speed + 0.60 * throttle
        pos = pos + np.array([np.cos(heading), np.sin(heading)]) * speed * 0.09

        d = np.linalg.norm(track - pos, axis=1)
        idx = int(np.argmin(d))
        checkpoints.add(idx // 30)
        if d[idx] > 7.5:
            crashes += 1
            # soft reset toward track; still counts as mistake
            pos = 0.85 * pos + 0.15 * track[idx]
            speed *= 0.55
        path.append(pos.copy())

    completed = len(checkpoints) >= 12
    return {
        "seed": seed,
        "complete": bool(completed),
        "max_cp": int(len(checkpoints)),
        "crashes": float(crashes / max(1, steps / 100)),
        "path": np.asarray(path),
        "track": track,
    }


def score_runs(runs: list[dict]) -> dict:
    return {
        "complete_runs": int(sum(r["complete"] for r in runs)),
        "total_runs": len(runs),
        "mean_checkpoints": float(np.mean([r["max_cp"] for r in runs])),
        "mean_crashes": float(np.mean([r["crashes"] for r in runs])),
        "best_checkpoints": int(max(r["max_cp"] for r in runs)),
    }
