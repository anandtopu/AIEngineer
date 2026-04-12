# Module 09 — LLM + RAG Primer

## Goals

- Explain RAG components and tradeoffs
- Build a simple retrieval + answer generation baseline

## RAG building blocks

- Documents
- Chunking
- Index / retrieval
- Optional reranking
- Prompting with retrieved context
- Evaluation (faithfulness + relevance)

## Practice (no API keys)

- Run `src/rag/tfidf_rag_demo.py`
- Run `projects/02_rag_baseline/run.py` and inspect:
  - `precision@k`, `recall@k`, `mrr`, `ndcg@k`
- Improve retrieval by:
  - changing chunk size
  - removing stopwords
  - using bigrams (exercise)
