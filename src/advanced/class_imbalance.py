"""Handling class imbalance: class weights, resampling, threshold tuning.

Interview goal: explain why accuracy is misleading on imbalanced data,
and walk through the standard mitigation menu:
  - class_weight='balanced' (cheap, no data change)
  - random oversampling / undersampling
  - decision-threshold tuning on PR curve, not the default 0.5
  - choose the metric that matches the cost (recall vs precision vs F1)
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def best_threshold_for_f1(y_true, y_score):
    p, r, t = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra point with no threshold.
    f1 = 2 * p * r / (p + r + 1e-12)
    best = int(np.nanargmax(f1[:-1]))
    return float(t[best]), float(f1[best])


def main():
    # 5% positive class — typical fraud / churn ratio.
    X, y = make_classification(
        n_samples=4000, n_features=12, n_informative=6,
        weights=[0.95, 0.05], flip_y=0.01, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
    print(f"Positive rate: train={y_tr.mean():.3f}  test={y_te.mean():.3f}")

    def evaluate(name, model):
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        thr, f1 = best_threshold_for_f1(y_te, proba)
        pred_tuned = (proba >= thr).astype(int)
        print(f"\n--- {name} ---")
        print(f"  ROC-AUC          : {roc_auc_score(y_te, proba):.3f}")
        print(f"  F1 @ 0.5         : {f1_score(y_te, pred):.3f}")
        print(f"  F1 @ tuned thr   : {f1:.3f}  (thr={thr:.3f})")
        print(f"  positives caught : {pred_tuned.sum()} / {y_te.sum()}")

    base = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)

    # Manual oversampling: duplicate minority rows until balanced.
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    pos_up = np.random.default_rng(0).choice(pos_idx, size=len(neg_idx), replace=True)
    idx = np.concatenate([neg_idx, pos_up])
    over = LogisticRegression(max_iter=1000).fit(X_tr[idx], y_tr[idx])

    evaluate("baseline (no fix)", base)
    evaluate("class_weight=balanced", weighted)
    evaluate("random oversampling", over)

    print("\nKey takeaway: ROC-AUC is similar across the three, but the")
    print("decision THRESHOLD and effective recall differ a lot. Pick the")
    print("knob (weights, sampling, threshold) that matches your cost.")


if __name__ == "__main__":
    main()
