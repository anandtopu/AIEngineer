# 3-Month Plan (FAANG-Style AI Engineer)

This plan optimizes for typical FAANG loops:
- Coding (DSA) rounds
- ML fundamentals + applied debugging
- ML/LLM system design
- Behavioral

Assumptions:
- 12-18 hours/week
- You do at least some coding practice **daily**

## Weekly time split (recommended)

- Coding/DSA: 5-7 hrs/week
- ML fundamentals + metrics/debugging: 3-4 hrs/week
- System design (ML + LLM/RAG): 3-4 hrs/week
- Behavioral: 1-2 hrs/week

## Daily routine (Mon–Fri)

- 45–60 min: coding practice (DSA)
- 45–90 min: ML/LLM study or implementation
- 15 min: 1 interview question out loud

Weekend:
- 1 mock interview (rotate: coding → ML → system design)
- 1 project iteration block (2-3 hours)

---

## Month 1 — Core ML + Interview Mechanics

### Week 1: Python + Metrics + Coding warm-up

- Learning:
  - `docs/modules/01_python_for_ml.md`
  - `docs/modules/05_evaluation_metrics.md`
- Run:
  - `src/basics/metrics_from_scratch.py`
- Coding focus:
  - Arrays/strings + hashmaps + two pointers
- Output goal:
  - Be able to explain precision/recall/F1 tradeoffs in 2 minutes

### Week 2: Supervised ML + CV + Calibration

- Learning:
  - `docs/modules/04_supervised_ml.md`
  - `src/advanced/sklearn_pipeline_cv.py`
  - `src/advanced/calibration_demo.py`
- Coding focus:
  - Binary search patterns + intervals
- Output goal:
  - Answer: leakage vs overfitting vs distribution shift

### Week 3: Data pitfalls + SQL (FAANG data round style)

- Learning:
  - `docs/modules/06_data_pitfalls.md`
  - `docs/modules/07_sql_for_ai_engineers.md`
- Run:
  - `src/sql/sql_practice_sqlite.py`
  - `src/sql/window_functions_practice_sqlite.py`
- Coding focus:
  - Stacks/queues + monotonic stack basics
- Output goal:
  - Solve cohort retention and funnel questions in SQL

### Week 4: Project 01 + System Design Intro

- Project:
  - `projects/01_churn_prediction/`
  - Add 1 feature + compare models
- System design:
  - Read: `interview/ml_system_design.md`
  - Drill: `interview/system_design_drills.md` (pick 2)
- Coding focus:
  - Trees + BFS/DFS patterns
- Mock interview:
  - 1 coding mock + 1 ML fundamentals mock

---

## Month 2 — Deep Learning + LLM/RAG + More Design

### Week 5: PyTorch training loop + debugging

- Learning:
  - `docs/modules/08_pytorch_core.md`
- Run:
  - `src/deep_learning/pytorch_training_loop_mnist.py`
- Coding focus:
  - Heaps + top-k patterns
- Mock:
  - 1 debugging-style ML mock (nan loss, overfitting)

### Week 6: LLM fundamentals + RAG evaluation

- Learning:
  - `docs/modules/09_llm_rag_primer.md`
  - `interview/llm_rag_questions.md`
- Run:
  - `projects/02_rag_baseline/run.py` (inspect retrieval metrics)
- Coding focus:
  - Graph basics (BFS shortest path)
- System design drill:
  - Internal KB assistant

### Week 7: Project 02 (RAG) + LLMOps

- Project:
  - Improve `projects/02_rag_baseline/` by adding docs and gold queries
- Learning:
  - `docs/modules/11_mlops_llmops_primer.md`
- Coding focus:
  - Dynamic programming intro
- Mock:
  - 1 LLM/RAG system design mock

### Week 8: Monitoring + drift + incident thinking

- Run:
  - `projects/03_monitoring_drift/drift_check.py`
- Practice:
  - Define monitoring dashboards + alert thresholds
- Coding focus:
  - Backtracking patterns
- Mock:
  - 1 system design mock (fraud or ranking)

---

## Month 3 — FAANG Loop Simulation (coding + ML + design + behavioral)

### Week 9: Ranking project + ranking system design

- Project:
  - `projects/04_ranking_baseline/`
- System design drills:
  - Search ranking
  - Recommendations feed
- Coding focus:
  - Advanced graphs (topological sort / union-find)

### Week 10: Time series project + experimentation

- Project:
  - `projects/05_time_series/`
- ML thinking:
  - Offline vs online metrics, A/B tests, guardrails
- Coding focus:
  - Mixed sets; revisit weak patterns

### Week 11: Mock interviews intensive

- 2 coding mocks
- 1 ML fundamentals mock
- 1 ML system design mock
- 1 behavioral mock

### Week 12: Final polish + interview readiness

- Re-run `validate.py`
- Revisit weak areas
- Prepare:
  - 6-8 STAR stories
  - 2-minute explanations for: bias/variance, calibration, PR-AUC vs ROC-AUC, leakage
  - 3 full system designs (ML + LLM)

---

## Milestones (FAANG readiness)

By end of:
- Week 4: 1 project + 2 system design drills + 15 coding problems
- Week 8: 2 projects + 6 drills + 40 coding problems
- Week 12: 5 projects + 12 drills + 80+ coding problems + 8 STAR stories
