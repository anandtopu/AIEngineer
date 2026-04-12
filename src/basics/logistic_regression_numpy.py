from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic_binary_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)

    x0 = rng.normal(loc=(-1.0, -1.0), scale=1.0, size=(n // 2, 2))
    x1 = rng.normal(loc=(1.0, 1.0), scale=1.0, size=(n // 2, 2))

    x = np.vstack([x0, x1])
    y = np.vstack([np.zeros((n // 2, 1)), np.ones((n // 2, 1))])

    idx = rng.permutation(n)
    return x[idx], y[idx]


def fit_logreg_gd(x: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 2000):
    n, d = x.shape
    w = np.zeros((d, 1))
    b = 0.0

    for _ in range(steps):
        logits = x @ w + b
        probs = sigmoid(logits)
        grad_w = (x.T @ (probs - y)) / n
        grad_b = float(np.mean(probs - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def predict(x: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    probs = sigmoid(x @ w + b)
    return (probs >= threshold).astype(int)


def main():
    x, y = make_synthetic_binary_data()
    w, b = fit_logreg_gd(x, y)

    y_pred = predict(x, w, b)
    acc = float(np.mean(y_pred == y))

    print(f"train accuracy={acc:.3f}")
    print(f"w={w.ravel()}, b={b:.3f}")


if __name__ == "__main__":
    main()
