# Repository Index

## Start here

- `README.md`
- `docs/3_MONTH_PLAN.md`
- `docs/ROADMAP.md`
- `docs/MODULES.md`

## Docs (modules)

- `docs/modules/01_python_for_ml.md`
- `docs/modules/02_probability_stats.md`
- `docs/modules/03_linear_algebra.md`
- `docs/modules/04_supervised_ml.md`
- `docs/modules/05_evaluation_metrics.md`
- `docs/modules/06_data_pitfalls.md`
- `docs/modules/07_sql_for_ai_engineers.md`
- `docs/modules/08_pytorch_core.md`
- `docs/modules/09_llm_rag_primer.md`
- `docs/modules/10_ml_system_design_primer.md`
- `docs/modules/11_mlops_llmops_primer.md`
- `docs/modules/12_transformer_internals.md`
- `docs/modules/13_llm_inference_and_finetuning.md`
- `docs/modules/14_llm_agents_and_tool_use.md`
- `docs/modules/15_production_rag_patterns.md`
- `docs/modules/16_llm_observability_cost_safety.md`
- `docs/modules/17_modern_training_and_alignment.md`
- `docs/modules/18_ai_engineer_2026_capabilities.md`

## Hands-on examples (src/)

### basics
- `src/basics/linear_regression_numpy.py`
- `src/basics/logistic_regression_numpy.py`
- `src/basics/metrics_from_scratch.py`

### deep learning
- `src/deep_learning/pytorch_training_loop_mnist.py`
- `src/deep_learning/autograd_micro.py` -- micrograd-style scalar autograd
- `src/deep_learning/attention_from_scratch.py` -- scaled dot-product + multi-head attention
- `src/deep_learning/transformer_encoder_torch.py` -- minimal Transformer encoder block
- `src/deep_learning/rope_positional.py` -- Rotary Positional Embeddings (LLaMA/Mistral default)
- `src/deep_learning/kv_cache.py` -- KV cache for fast autoregressive decoding
- `src/deep_learning/speculative_decoding.py` -- lossless inference speedup
- `src/deep_learning/lora_from_scratch.py` -- LoRA wrapped around a frozen Linear layer
- `src/deep_learning/dpo_from_scratch.py` -- Direct Preference Optimization loss
- `src/deep_learning/knowledge_distillation.py` -- teacher-student compression
- `src/deep_learning/clip_dual_encoder.py` -- contrastive dual-encoder (CLIP recipe)

### advanced ML
- `src/advanced/sklearn_pipeline_cv.py`
- `src/advanced/calibration_demo.py`
- `src/advanced/class_imbalance.py` -- weights, resampling, threshold tuning
- `src/advanced/ab_test_significance.py` -- z-test, CIs, sample size, peeking
- `src/advanced/bpe_tokenizer.py` -- BPE training and encoding
- `src/advanced/llm_sampling.py` -- greedy / temperature / top-k / top-p
- `src/advanced/quantization_int8.py` -- symmetric and asymmetric weight quantization

### retrieval and RAG
- `src/rag/tfidf_rag_demo.py`
- `src/rag/retrieval_eval_metrics.py`
- `src/rag/embeddings_knn_search.py` -- dense retrieval baseline
- `src/rag/chunking_strategies.py` -- fixed / sliding / sentence / paragraph chunking
- `src/rag/hybrid_search_bm25.py` -- BM25 + dense + Reciprocal Rank Fusion
- `src/rag/cross_encoder_rerank.py` -- 2-stage retrieval cascade with cross-encoder rerank
- `src/rag/semantic_cache.py` -- cache LLM answers by embedding similarity
- `src/rag/ragas_style_eval.py` -- faithfulness / relevancy / precision / recall

### LLM application layer
- `src/llm/agent_react_loop.py` -- ReAct agent loop with tools
- `src/llm/prompt_patterns.py` -- zero-shot, few-shot, CoT, self-consistency, critique
- `src/llm/structured_output.py` -- JSON schema + self-healing retry
- `src/llm/streaming_generator.py` -- streaming tokens, stop sequences, incremental JSON
- `src/llm/guardrails.py` -- PII redaction, prompt injection, output policy
- `src/llm/token_economics.py` -- token counting, context budgeting, cost across models
- `src/llm/observability_tracing.py` -- span-based tracing for RAG and agents

### SQL
- `src/sql/sql_practice_sqlite.py`
- `src/sql/window_functions_practice_sqlite.py`

## Interview prep

- `interview/README.md`
- `interview/ml_fundamentals_questions.md`
- `interview/deep_learning_questions.md`
- `interview/llm_rag_questions.md`
- `interview/ml_system_design.md`
- `interview/system_design_drills.md`
- `interview/behavioral_prep.md`
- `interview/mock_interview_loop.md`
- `interview/llm_agents_and_production_questions.md`
- `interview/genai_system_design_cases.md`
- `interview/cheat_sheets/`

## Projects

- `projects/README.md`
- `projects/01_churn_prediction/`
- `projects/02_rag_baseline/`
- `projects/03_monitoring_drift/`
- `projects/04_ranking_baseline/`
- `projects/05_time_series/`
- `projects/06_mini_llm_eval/` -- LLM evaluation harness with EM / F1 / ROUGE-L / judge
- `projects/07_agent_tool_use/` -- ReAct agent with three tools + memory + step budget
- `projects/08_hybrid_rag/` -- hybrid BM25 + dense + rerank + RAGAS-style eval
- `projects/09_llm_gateway/` -- guardrails + routing + semantic cache + token budgets

## Runnability

- `requirements.txt`
- `validate.py`

Run:

```bash
python validate.py
```
