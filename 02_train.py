"""Train the navigation neural network for one iteration."""
from __future__ import annotations

import argparse

import numpy as np

from drive2win import nn
from drive2win.normalize import fit_normalizer, transform
from drive2win.viz import plot_loss_curves


def train(data_path: str, tag: str, epochs: int, lr: float, h1: int, h2: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    data = np.load(data_path)
    x, y = data["x"], data["y"]
    idx = rng.permutation(len(x))
    split = int(len(x) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    normalizer = fit_normalizer(x[train_idx])
    x_train = transform(x[train_idx], normalizer)
    x_val = transform(x[val_idx], normalizer)
    y_train = y[train_idx]
    y_val = y[val_idx]

    weights = nn.init_weights(input_dim=x.shape[1], h1=h1, h2=h2, seed=seed)
    opt = nn.Adam(weights, lr=lr)
    batch_size = 64
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            cache = nn.forward_all(x_train[b], weights)
            loss, grads = nn.backward(cache, y_train[b], weights)
            opt.step(weights, grads)
            batch_losses.append(loss)
        train_losses.append(float(np.mean(batch_losses)))
        val_pred = nn.forward(x_val, weights)
        val_losses.append(float(np.mean((val_pred - y_val) ** 2)))

    model_path = f"nav_{tag}.npz"
    np.savez(model_path, **weights, normalizer=normalizer, tag=tag, h1=h1, h2=h2)
    plot_loss_curves(train_losses, val_losses, f"fig_loss_{tag}.png")
    print(f"Final train loss: {train_losses[-1]:.5f}")
    print(f"Final validation loss: {val_losses[-1]:.5f}")
    print(f"Saved {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--h1", type=int, default=64)
    parser.add_argument("--h2", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.tag, args.epochs, args.lr, args.h1, args.h2, args.seed)


if __name__ == "__main__":
    main()
