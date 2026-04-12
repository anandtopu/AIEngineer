# Project 08 -- Production-style hybrid RAG with RAGAS-style eval

End-to-end RAG pipeline that mirrors what a real production system
looks like:

```
query -> BM25 + dense retrieval -> RRF fusion (top 20)
      -> cross-encoder rerank (top 3)
      -> grounded answer generation
      -> faithfulness + context_precision scoring
```

Every stage is a pure-Python function you can swap for its production
counterpart (FAISS, Cohere rerank, Anthropic API, RAGAS). This project
is meant to show you what boxes a real interviewer expects you to
draw, and to give you a codebase you can iterate on locally.

## Run

```bash
python projects/08_hybrid_rag/run.py
```

## What the output shows

- For each query: the reranked top document ids, the generated
  answer, and the RAGAS-style metrics.
- Aggregate scores over all queries at the end.

## Extension ideas

- Add **HyDE**: generate a hypothetical answer with the mock LLM
  and retrieve against its embedding. Does MRR improve?
- Add **query expansion**: produce 3 paraphrases and fuse their
  rankings with RRF. Compare to the baseline.
- Add **answer relevancy** and **context recall** metrics (see
  `src/rag/ragas_style_eval.py`).
- Compute **bootstrap 95% confidence intervals** on each metric.
- Replace the extractive generator with a real LLM call that
  conditions on the top chunks.
- Swap the fake hash embedding for a real sentence-transformer
  embedding (requires sentence-transformers or an API call).
