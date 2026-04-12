# ML System Design — Interview Guide

Use this as a reusable template in system design interviews.

## 1) Clarify requirements

- Users and use-cases
- Latency (p50/p95), throughput, availability
- Online vs batch predictions
- Interpretability requirements
- Data privacy/regulatory constraints

## 2) Define success metrics

- Business metric
- Offline metrics
- Online metrics (A/B tests)
- Guardrail metrics (fairness, safety)

## 3) Data

- Sources
- Labeling strategy
- Data quality checks
- Train/serving skew risks

## 4) Baseline

- Simple heuristic
- Linear model
- Existing model or rules

## 5) Modeling

- Feature engineering vs representation learning
- Candidate models and why
- Training strategy
- Tuning strategy

## 6) Evaluation

- Slicing (by cohort)
- Calibration
- Robustness tests

## 7) Serving architecture

- Batch scoring vs online service
- Model packaging
- Feature store vs on-the-fly
- Caching

## 8) Monitoring + iteration

- Data drift
- Performance drift
- Alerting and rollback

## 9) Failure modes

- Missing features
- Feedback loops
- Adversarial inputs
