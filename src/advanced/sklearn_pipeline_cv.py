from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cross_val_pr_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []

    for tr, va in cv.split(X, y):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[va])[:, 1]
        scores.append(float(average_precision_score(y[va], prob)))

    return float(np.mean(scores))


def main():
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=6,
        n_redundant=6,
        weights=[0.95, 0.05],
        random_state=0,
    )

    pr_auc = cross_val_pr_auc(X, y)
    print(f"CV PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
