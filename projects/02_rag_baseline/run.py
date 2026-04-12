from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.retrieval_eval_metrics import RetrievalExample, aggregate_metrics


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str


def build_index(docs: list[Doc]):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform([d.text for d in docs])
    return vec, X


def retrieve(query: str, vec: TfidfVectorizer, X, docs: list[Doc], top_k: int = 3):
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
    return [(docs[i], float(score)) for i, score in ranked]


def evaluate_retrieval(queries: list[str], gold_doc_ids: list[str], vec, X, docs: list[Doc], k: int = 3):
    examples: list[RetrievalExample] = []
    for q, gold in zip(queries, gold_doc_ids, strict=True):
        got = [d.doc_id for d, _ in retrieve(q, vec, X, docs, top_k=k)]
        examples.append(RetrievalExample(query=q, gold_doc_id=gold, ranked_doc_ids=got))

    return aggregate_metrics(examples, k=k)


def main():
    docs = [
        Doc("bias", "Bias is error from incorrect assumptions; high bias typically underfits."),
        Doc("variance", "Variance is error from sensitivity to training data; high variance typically overfits."),
        Doc("leakage", "Data leakage uses future/test information during training and inflates offline metrics."),
        Doc("prauc", "PR-AUC is often better than ROC-AUC for highly imbalanced problems."),
        Doc("rag", "RAG retrieves relevant documents to ground answers and reduce hallucinations."),
    ]

    vec, X = build_index(docs)

    queries = [
        "What is data leakage?",
        "How do I explain overfitting?",
        "What metric is better for imbalance?",
    ]
    gold = ["leakage", "variance", "prauc"]

    metrics = evaluate_retrieval(queries, gold, vec, X, docs, k=3)
    print("Retrieval metrics (k=3):")
    for name, value in metrics.items():
        print(f"- {name}={value:.3f}")

    q = "How do I explain bias vs variance?"
    retrieved = retrieve(q, vec, X, docs, top_k=3)

    print("\nRetrieved:")
    for d, s in retrieved:
        print(f"- {d.doc_id} score={s:.3f} text={d.text}")

    print("\nBaseline answer:")
    print(retrieved[0][0].text)


if __name__ == "__main__":
    main()
