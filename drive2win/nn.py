"""Small NumPy MLP with manual backpropagation and Adam optimizer."""
from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def init_weights(input_dim: int, h1: int = 64, h2: int = 32, output_dim: int = 2, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, h1)),
        "b1": np.zeros(h1),
        "W2": rng.normal(0, np.sqrt(2 / h1), size=(h1, h2)),
        "b2": np.zeros(h2),
        "W3": rng.normal(0, np.sqrt(2 / h2), size=(h2, output_dim)),
        "b3": np.zeros(output_dim),
    }


def forward_all(x: np.ndarray, weights: dict) -> dict:
    z1 = x @ weights["W1"] + weights["b1"]
    a1 = relu(z1)
    z2 = a1 @ weights["W2"] + weights["b2"]
    a2 = relu(z2)
    out = a2 @ weights["W3"] + weights["b3"]
    return {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "out": out}


def forward(x: np.ndarray, weights: dict) -> np.ndarray:
    return forward_all(x, weights)["out"]


def backward(cache: dict, y_true: np.ndarray, weights: dict) -> tuple[float, dict]:
    """Manual chain-rule backpropagation for MSE loss."""
    m = y_true.shape[0]
    y_pred = cache["out"]
    diff = y_pred - y_true
    loss = float(np.mean(diff ** 2))

    d_out = (2.0 / m) * diff / y_true.shape[1]
    dW3 = cache["a2"].T @ d_out
    db3 = d_out.sum(axis=0)

    da2 = d_out @ weights["W3"].T
    dz2 = da2 * relu_grad(cache["z2"])
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ weights["W2"].T
    dz1 = da1 * relu_grad(cache["z1"])
    dW1 = cache["x"].T @ dz1
    db1 = dz1.sum(axis=0)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
    return loss, grads


class Adam:
    def __init__(self, weights: dict, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in weights.items()}

    def step(self, weights: dict, grads: dict) -> None:
        self.t += 1
        for k in weights:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            weights[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def predict_action(x: np.ndarray, package: dict) -> np.ndarray:
    from .normalize import transform, inverse_actions
    x_norm = transform(np.asarray(x, dtype=float), package["normalizer"].item() if hasattr(package["normalizer"], "item") else package["normalizer"])
    y = forward(x_norm, {k: package[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]})
    return inverse_actions(y)
