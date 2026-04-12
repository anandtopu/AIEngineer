import numpy as np


def make_synthetic_linear_data(n: int = 200, noise_std: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=(n, 1))
    true_w = np.array([[2.5]])
    true_b = -0.7
    y = x @ true_w + true_b + rng.normal(0, noise_std, size=(n, 1))
    return x, y, true_w, true_b


def fit_closed_form(x: np.ndarray, y: np.ndarray):
    X = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    w = theta[:1]
    b = float(theta[1])
    return w, b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def main():
    x, y, true_w, true_b = make_synthetic_linear_data()
    w, b = fit_closed_form(x, y)

    y_pred = x @ w + b

    print(f"True w={true_w.ravel()[0]:.3f}, b={true_b:.3f}")
    print(f"Fit  w={w.ravel()[0]:.3f}, b={b:.3f}")
    print(f"MSE={mse(y, y_pred):.4f}")


if __name__ == "__main__":
    main()
