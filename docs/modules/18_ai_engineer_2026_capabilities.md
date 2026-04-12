# Module 18 -- What AI Engineer Roles Actually Require in 2026

Goal: map every concept in this repo to the skills that hiring managers
and JDs at FAANG and startups explicitly ask for. Use this as a
checklist before you apply.

## The 2026 signal (from JDs and industry reports)

- **LinkedIn Jobs on the Rise 2026**: the top three skills on AI
  Engineer JDs are **LangChain, Retrieval-Augmented Generation (RAG),
  and PyTorch**. Agent building is listed as a "core requirement, not
  a nice-to-have."
- **GenAI system design interviews** now replace ~75% of the classical
  ML system design questions at major LLM-focused teams. Expect
  "Design an LLM-powered X" as the primary prompt.
- **Portfolio over credentials**: hiring managers want to see working
  AI applications, not a transcript. You are expected to ship.
- **Production concerns** (cost, latency, eval, safety) are no longer
  "senior-only" topics. Mid-level AI Engineer interviews ask about
  them directly.

Sources: LinkedIn Jobs on the Rise 2026, aijobs.com startup listings,
Microsoft Azure Agent Factory blog, IBM Agentic RAG guide, lockedinai
and Exponent interview guides, tryexponent.com ML system design guide,
RAGAS docs, Anthropic Guardrails docs.

## Core buckets and where each is covered

### 1. Foundations (never skipped even in LLM interviews)

| Topic                       | Where                                          |
|-----------------------------|------------------------------------------------|
| Linear algebra, calculus    | `docs/modules/03_linear_algebra.md`            |
| Probability & statistics    | `docs/modules/02_probability_stats.md`         |
| Classical ML                | `docs/modules/04_supervised_ml.md`, `src/basics/` |
| Evaluation metrics          | `docs/modules/05_evaluation_metrics.md`, `src/advanced/calibration_demo.py` |
| Data pitfalls / leakage     | `docs/modules/06_data_pitfalls.md`             |
| Class imbalance & thresholds| `src/advanced/class_imbalance.py`              |
| A/B testing                 | `src/advanced/ab_test_significance.py`         |
| SQL for ML                  | `docs/modules/07_sql_for_ai_engineers.md`      |

### 2. Deep learning fundamentals

| Topic                              | Where                                       |
|------------------------------------|---------------------------------------------|
| PyTorch training loop              | `src/deep_learning/pytorch_training_loop_mnist.py` |
| Autograd / backprop from scratch   | `src/deep_learning/autograd_micro.py`       |
| Attention from scratch             | `src/deep_learning/attention_from_scratch.py` |
| Transformer encoder block          | `src/deep_learning/transformer_encoder_torch.py` |
| RoPE positional embeddings         | `src/deep_learning/rope_positional.py`      |
| KV caching                         | `src/deep_learning/kv_cache.py`             |
| Speculative decoding               | `src/deep_learning/speculative_decoding.py` |
| LoRA / PEFT                        | `src/deep_learning/lora_from_scratch.py`    |
| DPO preference learning            | `src/deep_learning/dpo_from_scratch.py`     |
| Knowledge distillation             | `src/deep_learning/knowledge_distillation.py` |
| CLIP-style contrastive             | `src/deep_learning/clip_dual_encoder.py`    |
| INT8 quantization                  | `src/advanced/quantization_int8.py`         |

### 3. LLM application layer (the 2026 core)

| Topic                          | Where                                      |
|--------------------------------|--------------------------------------------|
| Tokenization (BPE)             | `src/advanced/bpe_tokenizer.py`            |
| Sampling (greedy/top-k/top-p)  | `src/advanced/llm_sampling.py`             |
| Prompt engineering patterns    | `src/llm/prompt_patterns.py`               |
| Structured output / JSON       | `src/llm/structured_output.py`             |
| Streaming generation           | `src/llm/streaming_generator.py`           |
| ReAct agent loop               | `src/llm/agent_react_loop.py`              |
| Guardrails (input + output)    | `src/llm/guardrails.py`                    |
| Token economics / cost         | `src/llm/token_economics.py`               |
| Observability / tracing        | `src/llm/observability_tracing.py`         |

### 4. Retrieval and RAG

| Topic                        | Where                                     |
|------------------------------|-------------------------------------------|
| TF-IDF retrieval             | `src/rag/tfidf_rag_demo.py`               |
| Retrieval metrics            | `src/rag/retrieval_eval_metrics.py`       |
| Dense embeddings + k-NN      | `src/rag/embeddings_knn_search.py`        |
| Chunking strategies          | `src/rag/chunking_strategies.py`          |
| Hybrid search + RRF          | `src/rag/hybrid_search_bm25.py`           |
| Cross-encoder reranking      | `src/rag/cross_encoder_rerank.py`         |
| Semantic cache               | `src/rag/semantic_cache.py`               |
| RAGAS-style eval             | `src/rag/ragas_style_eval.py`             |

### 5. Projects (portfolio)

| Project                            | Shows                                    |
|------------------------------------|------------------------------------------|
| `01_churn_prediction`              | Classical ML + feature engineering       |
| `02_rag_baseline`                  | Minimal RAG                              |
| `03_monitoring_drift`              | Production monitoring                    |
| `04_ranking_baseline`              | Learning to rank                         |
| `05_time_series`                   | Forecasting discipline                   |
| `06_mini_llm_eval`                 | Eval harness with graders + slices       |
| `07_agent_tool_use`                | ReAct + tools + memory                   |
| `08_hybrid_rag`                    | Production RAG pipeline with RAGAS eval  |
| `09_llm_gateway`                   | Guardrails + routing + cache + budgets   |

### 6. System design (ML and GenAI)

- Classical ML: `docs/modules/10_ml_system_design_primer.md`,
  `interview/ml_system_design.md`, `interview/system_design_drills.md`.
- GenAI: the modules 14-17 above cover the vocabulary. Expect prompts
  like "Design a customer-support AI", "Design a code review agent",
  "Design semantic search over our documentation", "Design an LLM
  gateway for our product", "Design a RAG pipeline that stays fresh".

### 7. MLOps / LLMOps

- `docs/modules/11_mlops_llmops_primer.md`
- `docs/modules/16_llm_observability_cost_safety.md`
- `projects/03_monitoring_drift/drift_check.py`
- `projects/09_llm_gateway/run.py`

## Company archetype playbook

### Big tech / FAANG
- Expect **more classical ML system design** mixed with GenAI. Know
  ranking, recommendations, fraud, search.
- Expect **strong coding DSA** screens. See `interview/coding_dsa_plan.md`.
- Expect depth on **one area**: ranking, search, ads, trust/safety,
  training infra. Pick one and go deep in your follow-ups.
- Will probe **fundamentals**: bias/variance, metrics, calibration,
  leakage, experiment design.

### LLM-focused labs (OpenAI, Anthropic, Google DeepMind, Meta FAIR)
- Expect **research-flavored questions**: derive attention, explain
  RLHF/DPO, analyze a scaling law graph.
- Expect **inference infrastructure** questions: KV cache, paged
  attention, continuous batching, speculative decoding, distillation.
- Expect **evaluation rigor**: contamination, judge bias, slice
  analysis, bootstrap CIs.

### AI-first startups
- Expect **end-to-end product questions**: "How would you build this
  feature in a week?"
- Expect **cost and latency** to be discussed in every answer.
- Expect **agent / RAG / guardrails** as the core technical ask.
- Expect **framework familiarity**: LangChain / LlamaIndex / LangGraph
  / Anthropic SDK / OpenAI SDK / pgvector / FAISS / Pinecone.

### Enterprise AI teams
- Expect **compliance and safety** questions: PII, auditability,
  tenant isolation, HIPAA / SOC2.
- Expect **evaluation and monitoring** questions about shipped
  systems.
- Will reward answers that start from business metrics, not from
  model metrics.

## Six-week cram plan (if you already know classical ML)

1. **Week 1**: Modules 8-9 (PyTorch, LLM/RAG primer) + `src/deep_learning/*`.
2. **Week 2**: Module 12 (Transformer internals) + attention / RoPE / KV cache scripts.
3. **Week 3**: Module 13 + module 17 (inference / LoRA / DPO /
   distillation / quantization).
4. **Week 4**: Modules 14-15 (agents, production RAG) + projects 07 and 08.
5. **Week 5**: Module 16 + project 09 + all the guardrails / cost /
   observability scripts.
6. **Week 6**: Mock interviews -- 3 coding, 3 ML system design, 3
   GenAI system design. Use `interview/mock_interview_loop.md`.
