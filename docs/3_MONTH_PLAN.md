# 3-Month AI Engineer Interview Prep Plan (Generalist Track)

This plan assumes 10-15 hours/week. Adjust based on your pace.

## Month 1: Foundations + Core ML

### Week 1: Python + Math/Stats Basics
- **Days 1-2**: Python data structures, complexity, `dataclasses`, type hints
  - Read: `docs/modules/01_python_for_ml.md`
  - Practice: Implement `top_k_frequent()`, `group_by_key()`
  - Run: `src/basics/metrics_from_scratch.py`

- **Days 3-4**: Probability, Bayes rule, distributions
  - Read: `docs/modules/02_probability_stats.md`
  - Practice: Derive precision/recall from confusion matrix

- **Days 5-7**: Linear algebra essentials
  - Read: `docs/modules/03_linear_algebra.md`
  - Run: `src/basics/linear_regression_numpy.py`, explain normal equation

### Week 2: Supervised ML + Evaluation
- **Days 8-10**: Linear/logistic regression, trees, ensembles
  - Read: `docs/modules/04_supervised_ml.md`
  - Run: `src/basics/logistic_regression_numpy.py`
  - Add L2 regularization to the gradient descent (exercise)

- **Days 11-12**: Metrics, calibration, thresholding
  - Read: `docs/modules/05_evaluation_metrics.md`
  - Run: `src/advanced/calibration_demo.py`
  - Practice: Explain when PR-AUC beats ROC-AUC

- **Days 13-14**: Bias/variance, overfitting, regularization
  - Read: `interview/ml_fundamentals_questions.md` Q1-5
  - Do mock interview: Answer 3 questions out loud, time yourself

### Week 3: Data Pitfalls + SQL
- **Days 15-17**: Data leakage, imbalance, shift
  - Read: `docs/modules/06_data_pitfalls.md`
  - Practice: List 5 leakage examples from your experience or imagination

- **Days 18-20**: SQL for AI Engineers
  - Read: `docs/modules/07_sql_for_ai_engineers.md`
  - Run: `src/sql/sql_practice_sqlite.py`
  - Exercise: Modify queries for retention cohort analysis

- **Days 21**: Review + catch up

### Week 4: First Project + ML System Design Intro
- **Days 22-25**: Churn prediction project
  - Run: `projects/01_churn_prediction/train.py`
  - Modify: Add a new feature, try a tree model, compare PR-AUC
  - Write: One-page design doc (metrics, risks, monitoring)

- **Days 26-28**: ML System Design fundamentals
  - Read: `docs/modules/10_ml_system_design_primer.md`, `interview/ml_system_design.md`
  - Practice: Do a 30-min design for "recommend top 10 products"

- **Days 29-30**: Mock interviews (2 sessions)
  - Use: `interview/mock_interview_loop.md`
  - Record yourself or practice with a friend

## Month 2: Deep Learning + LLMs + RAG

### Week 5: PyTorch + Training Loops
- **Days 31-33**: PyTorch fundamentals
  - Read: `docs/modules/08_pytorch_core.md`
  - Run: `src/deep_learning/pytorch_training_loop_mnist.py`
  - Modify: Add learning rate scheduling, try different optimizers

- **Days 34-36**: Debugging DL training
  - Read: `interview/deep_learning_questions.md`
  - Practice: Introduce a bug (e.g., no `zero_grad`), find and fix it

- **Days 37-38**: Cross-validation with sklearn pipelines
  - Run: `src/advanced/sklearn_pipeline_cv.py`
  - Exercise: Add stratified K-fold to churn project

### Week 6: LLM + RAG Fundamentals
- **Days 39-41**: LLM basics, tokenization, transformers
  - Read: `interview/llm_rag_questions.md` Q1-5
  - Practice: Explain self-attention in 2 minutes (record yourself)

- **Days 42-44**: RAG building blocks
  - Read: `docs/modules/09_llm_rag_primer.md`
  - Run: `src/rag/tfidf_rag_demo.py` and `projects/02_rag_baseline/run.py`
  - Modify: Try different chunk sizes, measure `Recall@k` impact

- **Days 45-46**: Evaluation for RAG
  - Practice: Design an eval suite (context relevance, faithfulness)

### Week 7: Second Project (RAG)
- **Days 47-50**: Build and evaluate RAG baseline
  - Run: `projects/02_rag_baseline/run.py`
  - Extend: Add 10 more documents, create 5 test queries with gold answers
  - Measure: `Recall@k`, experiment with hybrid retrieval

- **Days 51-53**: RAG system design
  - Practice: Design "internal knowledge base Q&A" for a company
  - Focus: Chunking strategy, retrieval, reranking, guardrails

### Week 8: MLOps + Monitoring
- **Days 54-56**: MLOps/LLMOps fundamentals
  - Read: `docs/modules/11_mlops_llmops_primer.md`
  - Run: `projects/03_monitoring_drift/drift_check.py`
  - Exercise: Set up PSI thresholds, simulate alerts

- **Days 57-59**: Monitoring + drift detection
  - Practice: Design monitoring for churn model (what to log, what to alert)
  - Read: Interview questions on failure modes

- **Days 60**: Review + catch up

## Month 3: Advanced Topics + Intensive Interview Practice

### Week 9: Advanced ML Topics
- **Days 61-63**: Ranking/Recommendation basics
  - Read: `projects/04_ranking_baseline/README.md`
  - Run: `projects/04_ranking_baseline/train.py`
  - Understand: NDCG, pairwise vs pointwise loss

- **Days 64-66**: Time-series forecasting
  - Read: `projects/05_time_series/README.md`
  - Run: `projects/05_time_series/baseline.py`
  - Understand: Feature engineering, train/test split by time

- **Days 67-69**: Advanced system design cases
  - Practice: Design fraud detection, search ranking, content moderation
  - Use template: `interview/ml_system_design.md`

### Week 10: Coding Interview Prep
- **Days 70-72**: ML coding patterns
  - Implement: Custom metrics, data generators, feature transforms
  - Practice: NumPy vectorization, pandas efficiency

- **Days 73-75**: SQL deep dive
  - Run: Complex queries with window functions
  - Exercise: Cohort retention, sessionization, funnel analysis

- **Days 76-77**: PyTorch coding exercises
  - Implement: Custom loss, custom dataset, data augmentation

### Week 11: Mock Interview Intensive
- **Days 78-80**: ML fundamentals mocks
  - 3 sessions: Bias/variance, metrics, debugging
  - Focus: Clear explanations, tradeoffs, failure modes

- **Days 81-83**: System design mocks
  - 3 designs: Pick from churn, fraud, search, recommendation, RAG
  - Record: 45-min sessions, review your structure

- **Days 84-86**: Behavioral prep
  - Read: `interview/behavioral_prep.md`
  - Write: STAR stories for 5 common questions
  - Practice: Tell me about yourself, Why this company, Conflict resolution

### Week 12: Final Review + Strategy
- **Days 87-89**: Review weak areas
  - Re-run projects, fix gaps
  - Re-read tricky interview questions

- **Days 90-92**: Final mock interviews
  - Full loop: coding + ML + system design + behavioral

- **Days 93-95**: Interview strategy
  - Company research, questions to ask, negotiation prep

- **Days 96-100**: Buffer/final polish

## Daily Habits (keep throughout)

1. **Morning (30 min)**: Review flashcards or cheat sheets
2. **Deep work (90 min)**: Learn or code
3. **Evening (30 min)**: 1-2 interview questions out loud

## Key Milestones

| Week | Milestone |
|------|-----------|
| 4 | Completed churn project + first mock |
| 7 | Completed RAG project + system design practice |
| 8 | PSI drift detection working |
| 10 | Ranking + time-series projects done |
| 12 | Ready for interviews |

## Success Metrics

- Can explain any algorithm in 2 minutes with tradeoffs
- Can debug training issues quickly (nan loss, overfitting)
- Can design a full ML system in 45 minutes
- Has 5 portfolio projects to discuss
- Has 10 STAR stories ready
