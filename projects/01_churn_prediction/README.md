# Project 01 — Churn Prediction (Classic ML)

## What you’ll practice

- Problem framing and metrics
- Train/validation split choices
- Feature preprocessing with a pipeline
- Baseline model + evaluation
- Error analysis and next steps

## Run

```bash
python projects/01_churn_prediction/train.py
```

## Interview talk track

- Define churn and the prediction horizon (e.g., churn in next 30 days)
- Offline metrics (PR-AUC, recall@k) vs business metric
- Leakage risks (using post-churn activity)
- Serving: batch scoring daily vs real-time
- Monitoring: drift + calibration + alerting
