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
python projects/01_churn_prediction/train.py
python projects/02_rag_baseline/run.py
python projects/03_monitoring_drift/drift_check.py
python projects/04_ranking_baseline/train.py
python projects/05_time_series/baseline.py
python projects/06_mini_llm_eval/run.py
```

## Hands-on labs for modern AI Engineer interviews

The `src/` tree includes from-scratch demos for the topics LLM-era
interviews actually drill on:

- **Transformers & attention** -- `src/deep_learning/attention_from_scratch.py`,
  `src/deep_learning/transformer_encoder_torch.py`
- **Backprop from first principles** -- `src/deep_learning/autograd_micro.py`
- **LoRA fine-tuning** -- `src/deep_learning/lora_from_scratch.py`
- **Tokenization (BPE)** -- `src/advanced/bpe_tokenizer.py`
- **LLM decoding strategies** -- `src/advanced/llm_sampling.py`
- **INT8 quantization** -- `src/advanced/quantization_int8.py`
- **Dense retrieval & chunking** -- `src/rag/embeddings_knn_search.py`,
  `src/rag/chunking_strategies.py`
- **Class imbalance & threshold tuning** -- `src/advanced/class_imbalance.py`
- **A/B test statistics** -- `src/advanced/ab_test_significance.py`

See `docs/modules/12_transformer_internals.md` and
`docs/modules/13_llm_inference_and_finetuning.md` for the matching notes.

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


