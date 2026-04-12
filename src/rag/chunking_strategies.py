"""Document chunking strategies for RAG, with empirical comparison.

Interview goal: chunking quality dominates RAG quality. Be ready to
discuss the trade-offs:
  - too small : context lost, more chunks = more retrieval noise
  - too large : retriever gets diluted, generator wastes context
  - overlap   : guards against the answer landing on a chunk boundary
  - semantic  : respects sentence/paragraph structure (best of both)

We implement four strategies and measure how often the gold-answer
sentence ends up in the same chunk as the retrieved chunk.
"""

from __future__ import annotations

import re
from typing import Callable, List


def fixed_chunks(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def sliding_chunks(text: str, size: int, overlap: int) -> List[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(1, len(text) - overlap), step)]


def sentence_chunks(text: str, max_chars: int) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return chunks


def paragraph_chunks(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# --- toy retrieval: count keyword overlap ---

def tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


def retrieve(query: str, chunks: List[str]) -> int:
    q = tokens(query)
    scored = [(len(q & tokens(c)), i) for i, c in enumerate(chunks)]
    scored.sort(reverse=True)
    return scored[0][1]


def evaluate(strategy_name, chunker: Callable[[], List[str]], qa_pairs):
    chunks = chunker()
    hits = 0
    for q, gold_phrase in qa_pairs:
        idx = retrieve(q, chunks)
        if gold_phrase.lower() in chunks[idx].lower():
            hits += 1
    print(f"  {strategy_name:30s} chunks={len(chunks):>3d}  "
          f"recall={hits}/{len(qa_pairs)}")


def main():
    text = (
        "The Transformer architecture was introduced in the 2017 paper "
        "Attention is All You Need. It replaces recurrence with self-attention.\n\n"
        "BERT is a bidirectional encoder. It is pretrained with masked language "
        "modeling on a large corpus.\n\n"
        "GPT is an autoregressive decoder. GPT models predict the next token "
        "given the previous ones.\n\n"
        "Retrieval Augmented Generation (RAG) grounds an LLM in external "
        "documents fetched at query time. The retriever finds relevant chunks "
        "and the generator conditions on them.\n\n"
        "LoRA injects low-rank trainable matrices into a frozen base model. "
        "It dramatically reduces the number of trainable parameters.\n\n"
        "Quantization reduces numerical precision of model weights. INT8 is "
        "common; INT4 is used for very large models."
    )
    qa = [
        ("When was the Transformer introduced?", "2017"),
        ("What kind of model is BERT?", "bidirectional encoder"),
        ("How does GPT generate text?", "next token"),
        ("What does LoRA freeze?", "frozen base model"),
        ("Why use INT8?", "INT8"),
    ]

    print(f"Document length: {len(text)} chars\n")
    print("Strategy results (recall = answer is in the retrieved chunk):")
    evaluate("fixed(80)",        lambda: fixed_chunks(text, 80), qa)
    evaluate("fixed(200)",       lambda: fixed_chunks(text, 200), qa)
    evaluate("sliding(120,40)",  lambda: sliding_chunks(text, 120, 40), qa)
    evaluate("sentence(<=160)",  lambda: sentence_chunks(text, 160), qa)
    evaluate("paragraph",        lambda: paragraph_chunks(text), qa)

    print("\nTakeaways:")
    print("  - tiny fixed chunks lose surrounding context")
    print("  - sliding overlap rescues answers near boundaries")
    print("  - sentence/paragraph chunking is usually the best default")


if __name__ == "__main__":
    main()
