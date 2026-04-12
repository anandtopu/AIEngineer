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
- `src/deep_learning/lora_from_scratch.py` -- LoRA wrapped around a frozen Linear layer

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
- `interview/cheat_sheets/`

## Projects

- `projects/README.md`
- `projects/01_churn_prediction/`
- `projects/02_rag_baseline/`
- `projects/03_monitoring_drift/`
- `projects/04_ranking_baseline/`
- `projects/05_time_series/`
- `projects/06_mini_llm_eval/` -- LLM evaluation harness with EM / F1 / ROUGE-L / judge

## Runnability

- `requirements.txt`
- `validate.py`

Run:

```bash
python validate.py
```
