"""Learning-to-rank baseline with LambdaMART-style pointwise approach."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def make_synthetic_ranking_data(n_queries: int = 1000, docs_per_query: int = 10):
    """Generate synthetic ranking data."""
    rng = np.random.default_rng(42)

    X_list, y_list, qid_list = [], [], []
    for q in range(n_queries):
        X_q, _ = make_classification(
            n_samples=docs_per_query,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=rng.integers(0, 10000),
        )
        # Simulate relevance scores (0, 1, 2)
        relevance = rng.integers(0, 3, size=docs_per_query)
        relevance = np.sort(relevance)[::-1]  # Higher relevance at top (ideal ranking)

        X_list.append(X_q)
        y_list.append(relevance)
        qid_list.extend([q] * docs_per_query)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    qids = np.array(qid_list)
    return X, y, qids


def dcg(scores: np.ndarray, k: int | None = None) -> float:
    """Discounted Cumulative Gain."""
    if k is not None:
        scores = scores[:k]
    gains = (2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum(gains))


def ndcg(scores: np.ndarray, k: int | None = None) -> float:
    """Normalized DCG."""
    ideal = np.sort(scores)[::-1]
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(scores, k) / ideal_dcg


def evaluate_ranking(y_true: np.ndarray, y_pred: np.ndarray, qids: np.ndarray, k: int = 5):
    """Evaluate ranking predictions grouped by query."""
    unique_qids = np.unique(qids)
    ndcg_scores = []

    for q in unique_qids:
        mask = qids == q
        y_q = y_true[mask]
        pred_q = y_pred[mask]

        # Sort by predicted score, get true relevance in that order
        order = np.argsort(-pred_q)
        ranked_true = y_q[order]

        ndcg_scores.append(ndcg(ranked_true, k=k))

    return float(np.mean(ndcg_scores))


def main():
    X, y, qids = make_synthetic_ranking_data(n_queries=500, docs_per_query=10)

    # Stratified split by query (prevent leakage)
    unique_qids = np.unique(qids)
    train_qids, test_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)

    train_mask = np.isin(qids, train_qids)
    test_mask = np.isin(qids, test_qids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test, qids_test = X[test_mask], y[test_mask], qids[test_mask]

    # Pointwise approach: treat as classification/regression
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)
    # Use probability of highest relevance class as score
    if y_pred.shape[1] > 1:
        scores = y_pred[:, -1]  # P(relevance=2)
    else:
        scores = y_pred.ravel()

    ndcg_5 = evaluate_ranking(y_test, scores, qids_test, k=5)
    ndcg_10 = evaluate_ranking(y_test, scores, qids_test, k=10)

    print(f"NDCG@5={ndcg_5:.4f}")
    print(f"NDCG@10={ndcg_10:.4f}")

    # Baseline: random ranking
    random_scores = np.random.default_rng(42).uniform(0, 1, size=len(y_test))
    ndcg_random = evaluate_ranking(y_test, random_scores, qids_test, k=5)
    print(f"Random NDCG@5={ndcg_random:.4f}")


if __name__ == "__main__":
    main()
