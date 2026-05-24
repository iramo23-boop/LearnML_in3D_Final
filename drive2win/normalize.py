"""Normalization utilities used by both training and inference."""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def fit_normalizer(x: np.ndarray) -> dict:
    """Return feature mean/std statistics."""
    return {"mean": x.mean(axis=0), "std": x.std(axis=0) + EPS}


def transform(x: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize features using saved statistics."""
    return (x - stats["mean"]) / stats["std"]


def inverse_actions(y: np.ndarray) -> np.ndarray:
    """Clip model actions to valid driving range."""
    y = np.asarray(y, dtype=float)
    y[..., 0] = np.clip(y[..., 0], 0.0, 1.0)      # throttle
    y[..., 1] = np.clip(y[..., 1], -1.0, 1.0)     # steering
    return y
