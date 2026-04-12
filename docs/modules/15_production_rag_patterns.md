# Module 15 -- Production RAG Patterns

Goal: go from "I built a RAG demo with TF-IDF" to "I have an opinion
on every knob in a production pipeline." This module covers the
patterns that show up in 2026 GenAI system design interviews and in
every RAG-in-production post-mortem.

## Mental model

```
              +-------------------+
query ----->  |  input guardrails |
              +-------------------+
                      |
                      v
              +-------------------+
              |  query rewrite /  |  (HyDE, query expansion,
              |  decomposition    |   multi-query)
              +-------------------+
                      |
                      v
        +--------------+----------------+
        |                               |
        v                               v
    BM25 top K                   dense top K
        \                             /
         \--- RRF / weighted sum ----/
                    |
                    v
             cross-encoder rerank (top 5-10)
                    |
                    v
           context builder (de-dup, truncate, cite)
                    |
                    v
                   LLM
                    |
                    v
            +-------------------+
            | output guardrails |  (faithfulness, schema, citations)
            +-------------------+
                    |
                    v
                 response
```

## What to read / run

- `src/rag/chunking_strategies.py` -- fixed/sliding/sentence/paragraph.
- `src/rag/embeddings_knn_search.py` -- dense retrieval mechanics.
- `src/rag/hybrid_search_bm25.py` -- BM25 + dense + RRF.
- `src/rag/cross_encoder_rerank.py` -- 2-stage cascade.
- `src/rag/semantic_cache.py` -- cache the LLM call, not just the retrieval.
- `src/rag/ragas_style_eval.py` -- faithfulness / relevancy / precision / recall.
- `projects/08_hybrid_rag/run.py` -- end-to-end project.
- `projects/02_rag_baseline/run.py` -- the TF-IDF baseline this replaces.

## Concepts every AI Engineer should own

### Chunking
- **Fixed** -- simple and fast, but often splits sentences mid-thought.
- **Sliding window with overlap** -- overlap rescues answers at boundaries.
- **Sentence / paragraph** -- usually the best default for prose.
- **Semantic** -- compute embedding similarity between consecutive
  sentences and cut where similarity drops. Best quality, higher cost.
- Typical sizes: 300-800 tokens per chunk, 50-100 token overlap.

### Embedding
- Start with an off-the-shelf model (`bge`, `e5`, OpenAI `text-embedding-3`).
- L2-normalize so inner product = cosine similarity.
- Train domain embeddings only when the gap to off-the-shelf is measured.

### Hybrid retrieval
- BM25 is the free strong baseline. Keep it.
- Dense retrieval catches paraphrase; BM25 catches exact matches (names,
  codes, error strings).
- Fuse with **Reciprocal Rank Fusion (RRF)**: `1 / (k + rank)`, sum
  across rankers. No score calibration required.

### Reranking
- Cross-encoder scores (query, document) jointly. Much higher quality,
  much slower -- use on 50-200 candidates, not on the full corpus.
- The 2-stage cascade (retriever -> reranker) is the industry default.
- Models to know: `bge-reranker`, `cohere-rerank-v3`, `ms-marco-MiniLM`.

### Query-side tricks
- **HyDE** (Hypothetical Document Embeddings): ask the LLM to write a
  fake answer first, then retrieve against THAT embedding. Surprisingly
  effective for long-tail queries.
- **Multi-query expansion**: ask the LLM for 3-5 paraphrases, retrieve
  with each, union the results.
- **Query decomposition**: for multi-hop questions, split into sub-queries.

### Evaluation
- **Faithfulness** -- is every claim in the answer supported by context?
- **Answer relevancy** -- does the answer actually address the question?
- **Context precision** -- are retrieved chunks actually relevant?
- **Context recall** -- did we retrieve everything needed?
- Use the RAGAS naming so you can talk to other engineers without
  inventing new terms.

### Operational concerns
- **Index staleness**: treat it like a cache. Log build time, have a
  re-index pipeline, monitor recall over a held-out eval set.
- **Semantic cache** (see module 16): saves 30-70% of inference cost.
- **Per-slice monitoring**: overall scores hide regressions on specific
  topics. Always report by slice.

## Common interview questions

- "Design a RAG chatbot for our product docs." (Go through the box
  diagram above out loud.)
- "Your answers are plausible but wrong. Where do you start?"
  (Faithfulness low -> LLM hallucinating; context precision low ->
  retriever failing. Diagnose with metrics first.)
- "Why hybrid? BM25 is so old." (BM25 is the cheap strong baseline
  for exact matches; dense for paraphrase.)
- "Chunks of 2000 tokens or 200?" (Depends on query type; measure
  on your own eval set.)
- "How do you keep the index fresh?"

## Drills

1. Add **HyDE** to `projects/08_hybrid_rag/run.py`: generate a fake
   answer with the mock LLM and retrieve against its embedding. Show
   whether MRR goes up or down on the test queries.
2. Add **query expansion**: take each query, generate 3 paraphrases,
   retrieve with each, merge with RRF.
3. Compute **bootstrap 95% confidence intervals** on each RAGAS metric
   across your eval set. Production eval reports always include them.
