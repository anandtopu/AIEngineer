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
```

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

## Status

This repo is intentionally structured like a course. If you want me to tailor it:
- Tell me your target companies/roles (startup vs FAANG-style)
- Tell me your timeline (2 weeks, 1 month, 3 months)
