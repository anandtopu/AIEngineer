# AI Engineer Interview Preparation (Comprehensive)

This repository is a **complete, hands-on tutorial** to prepare for **AI Engineer / ML Engineer** interviews.

It is designed for:
- Beginners with little experience who want a clear path
- Engineers switching into AI/ML
- Candidates preparing for end-to-end AI Engineering interviews (ML + Deep Learning + LLMs + MLOps + System Design)

## How to Use This Repo

- Follow the roadmap in `docs/ROADMAP.md`.
- For the full navigation index, use `INDEX.md`.
- Use `docs/MODULES.md` to jump into a module.
- If you have 3 months, follow `docs/3_MONTH_PLAN.md`.
- For FAANG-style preparation (coding + ML + system design), follow `docs/3_MONTH_PLAN_FAANG.md`.
- For each module, read the notes and then run the practice code in `src/`.
- Use `interview/` for question banks, checklists, and mock interview practice.

## Repository Map

- `docs/`
  - Guides and structured learning path
- `interview/`
  - Interview questions, cheat sheets, mock interview loops
- `src/`
  - Runnable code examples (from-scratch + PyTorch)
- `projects/`
  - Mini-projects you can build and discuss in interviews

## Projects (recommended)

Start here once you finish the fundamentals in `docs/`.

```bash
python projects/01_churn_prediction/train.py      # classical ML end-to-end
python projects/02_rag_baseline/run.py            # minimal RAG
python projects/03_monitoring_drift/drift_check.py  # production drift
python projects/04_ranking_baseline/train.py      # learning to rank
python projects/05_time_series/baseline.py        # forecasting
python projects/06_mini_llm_eval/run.py           # LLM eval harness
python projects/07_agent_tool_use/run.py          # ReAct agent with tools
python projects/08_hybrid_rag/run.py              # BM25 + dense + rerank + RAGAS
python projects/09_llm_gateway/run.py             # guardrails + cache + budgets
```

## Hands-on labs for modern AI Engineer interviews (2026)

Every script below is self-contained, runnable, and maps to a topic
that 2026 AI Engineer interviews at FAANG and startups drill on.

### Transformers, inference, and training internals
- **Attention from scratch** -- `src/deep_learning/attention_from_scratch.py`
- **Transformer encoder block (PyTorch)** -- `src/deep_learning/transformer_encoder_torch.py`
- **Backprop from first principles** -- `src/deep_learning/autograd_micro.py`
- **Rotary Positional Embeddings (LLaMA default)** -- `src/deep_learning/rope_positional.py`
- **KV cache** -- `src/deep_learning/kv_cache.py`
- **Speculative decoding** -- `src/deep_learning/speculative_decoding.py`
- **LoRA** -- `src/deep_learning/lora_from_scratch.py`
- **DPO (preference learning)** -- `src/deep_learning/dpo_from_scratch.py`
- **Knowledge distillation** -- `src/deep_learning/knowledge_distillation.py`
- **CLIP-style contrastive dual encoder** -- `src/deep_learning/clip_dual_encoder.py`
- **INT8 quantization** -- `src/advanced/quantization_int8.py`

### LLM application layer
- **ReAct agent loop** -- `src/llm/agent_react_loop.py`
- **Prompt patterns (few-shot, CoT, self-consistency)** -- `src/llm/prompt_patterns.py`
- **Structured output with self-healing retry** -- `src/llm/structured_output.py`
- **Streaming tokens, stop sequences, incremental JSON** -- `src/llm/streaming_generator.py`
- **Guardrails (PII, prompt injection, output policy)** -- `src/llm/guardrails.py`
- **Token economics and cost budgeting** -- `src/llm/token_economics.py`
- **Span-based tracing for RAG and agents** -- `src/llm/observability_tracing.py`

### Production RAG
- **Hybrid BM25 + dense + RRF** -- `src/rag/hybrid_search_bm25.py`
- **Cross-encoder reranking** -- `src/rag/cross_encoder_rerank.py`
- **Semantic cache** -- `src/rag/semantic_cache.py`
- **RAGAS-style eval (faithfulness / relevancy / precision / recall)** -- `src/rag/ragas_style_eval.py`
- **Chunking strategies** -- `src/rag/chunking_strategies.py`
- **Dense embeddings + k-NN** -- `src/rag/embeddings_knn_search.py`

### Foundations
- **Tokenization (BPE)** -- `src/advanced/bpe_tokenizer.py`
- **LLM decoding strategies** -- `src/advanced/llm_sampling.py`
- **Class imbalance & threshold tuning** -- `src/advanced/class_imbalance.py`
- **A/B test statistics** -- `src/advanced/ab_test_significance.py`

### Matching module notes
- `docs/modules/12_transformer_internals.md`
- `docs/modules/13_llm_inference_and_finetuning.md`
- `docs/modules/14_llm_agents_and_tool_use.md`
- `docs/modules/15_production_rag_patterns.md`
- `docs/modules/16_llm_observability_cost_safety.md`
- `docs/modules/17_modern_training_and_alignment.md`
- `docs/modules/18_ai_engineer_2026_capabilities.md` -- role map for
  FAANG, LLM labs, startups, and enterprise teams.

### Interview question banks
- `interview/llm_agents_and_production_questions.md` -- 24 Qs on
  agents, production RAG, guardrails, cost, alignment.
- `interview/genai_system_design_cases.md` -- 10 worked case studies.

## Quickstart

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run an example

```bash
python src/basics/linear_regression_numpy.py
```

Other examples:

```bash
python src/basics/metrics_from_scratch.py
python src/sql/sql_practice_sqlite.py
python src/rag/tfidf_rag_demo.py
```

## What You’ll Learn

- ML fundamentals (bias/variance, regularization, metrics, CV)
- Data work (pandas, feature engineering, leakage, imbalance)
- Deep Learning (PyTorch training loops, debugging, optimization)
- LLM fundamentals (tokenization, transformers, finetuning concepts)
- RAG (chunking, retrieval, reranking, evaluation)
- MLOps/LLMOps (experiment tracking concepts, CI checks, monitoring)
- ML system design (requirements → data → modeling → serving → monitoring)


