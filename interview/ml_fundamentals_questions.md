# ML Fundamentals — Interview Questions (with what interviewers look for)

## Core concepts

1) Explain **bias vs variance**. How do you diagnose each?
- What they look for:
  - Bias: underfitting; high training + validation error
  - Variance: overfitting; low training error but high validation error
  - Tools: learning curves, CV, regularization, more data, simpler model

2) What’s the difference between **generative vs discriminative** models?

3) When would you prefer **precision/recall** over accuracy?

4) Explain **ROC-AUC vs PR-AUC** and when each is more informative.

5) What is **data leakage**? Give examples.

6) How does **regularization** help? Compare L1 vs L2.

7) What is **calibration**? How do you check it and improve it?

8) How do you choose a **decision threshold**?

9) Explain **cross-validation**. When can CV be misleading?

10) How do you handle **class imbalance**?

## Practical ML

11) Walk me through an ML project end-to-end.
- Problem framing
- Metrics
- Data collection/labeling
- Baseline
- Feature engineering
- Training/tuning
- Evaluation
- Deployment
- Monitoring + iteration

12) How do you perform **hyperparameter tuning** safely?

13) What’s the difference between **train/val/test** vs **train/test** only?

14) Explain **feature importance** for trees vs linear models.

15) How do you detect **distribution shift**?

## Debugging

16) Your model performs great offline but poorly in production. Why?

17) Training loss decreases but validation loss increases. What do you do?

18) Your model performance is unstable between runs. Why?

## Mini case prompts

19) Design a model to predict customer churn. What are pitfalls?

20) You have 0.5% positive rate. What metric + validation strategy?
