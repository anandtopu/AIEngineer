from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_synthetic_churn(n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(0, 60, size=n)
    monthly_spend = rng.gamma(shape=2.0, scale=25.0, size=n)
    tickets_last_30d = rng.poisson(lam=1.0, size=n)
    country = rng.choice(["US", "IN", "BR", "DE"], size=n, p=[0.4, 0.3, 0.2, 0.1])

    logit = (
        -1.5
        + 0.02 * (monthly_spend - 40)
        + 0.35 * (tickets_last_30d)
        - 0.015 * tenure_months
        + (country == "BR") * 0.25
    )

    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, size=n) < p).astype(int)

    X = {
        "tenure_months": tenure_months.astype(float),
        "monthly_spend": monthly_spend.astype(float),
        "tickets_last_30d": tickets_last_30d.astype(float),
        "country": country.astype(object),
    }
    return X, y


def main():
    X, y = make_synthetic_churn()

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ["tenure_months", "monthly_spend", "tickets_last_30d"]
    categorical_features = ["country"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train = {k: np.asarray(v)[train_idx] for k, v in X.items()}
    y_train = y[train_idx]
    X_test = {k: np.asarray(v)[test_idx] for k, v in X.items()}
    y_test = y[test_idx]

    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)

    print(f"ROC-AUC={roc:.4f}")
    print(f"PR-AUC={ap:.4f}")


if __name__ == "__main__":
    main()
