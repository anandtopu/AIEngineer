from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


def retrieve(query: str, docs: list[Document], top_k: int = 3):
    corpus = [d.text for d in docs]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(corpus)
    q = vectorizer.transform([query])

    sims = cosine_similarity(q, X).ravel()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]

    return [(docs[i], float(score)) for i, score in ranked]


def generate_answer(query: str, retrieved: list[tuple[Document, float]]) -> str:
    # This is a no-LLM baseline: we simply return the most relevant context.
    # In an interview, explain that the next step would be an LLM prompt that cites sources.
    if not retrieved:
        return "No relevant context found."

    best_doc, _ = retrieved[0]
    return (
        "Answer (baseline):\n"
        f"- Query: {query}\n"
        f"- Most relevant context: {best_doc.text}\n"
    )


def main():
    docs = [
        Document(
            "d1",
            "Bias is error from incorrect assumptions in the learning algorithm; high bias leads to underfitting.",
        ),
        Document(
            "d2",
            "Variance is error from sensitivity to small fluctuations in the training set; high variance leads to overfitting.",
        ),
        Document(
            "d3",
            "Precision measures how many predicted positives are actually positive; recall measures how many actual positives are found.",
        ),
        Document(
            "d4",
            "Data leakage happens when information from the test set or future is used during training, inflating offline metrics.",
        ),
        Document(
            "d5",
            "RAG systems retrieve relevant documents and provide them as context to a model to reduce hallucinations and improve factuality.",
        ),
    ]

    query = "How do you explain bias and variance in interviews?"
    retrieved = retrieve(query, docs, top_k=3)

    print("Retrieved:")
    for d, s in retrieved:
        print(f"- {d.doc_id}: score={s:.3f} text={d.text}")

    print("\n" + generate_answer(query, retrieved))


if __name__ == "__main__":
    main()
