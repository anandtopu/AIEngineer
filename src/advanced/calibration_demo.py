from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(float).ravel()
    prob = prob.astype(float).ravel()
    return float(np.mean((prob - y_true) ** 2))


def main():
    X, y = make_classification(
        n_samples=8000,
        n_features=25,
        n_informative=8,
        n_redundant=8,
        weights=[0.9, 0.1],
        random_state=0,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")

    print(f"Brier={brier_score(y_te, prob):.4f}")
    print("Calibration points (mean_pred -> frac_pos):")
    for mp, fp in zip(mean_pred, frac_pos, strict=True):
        print(f"{mp:.3f} -> {fp:.3f}")


if __name__ == "__main__":
    main()
