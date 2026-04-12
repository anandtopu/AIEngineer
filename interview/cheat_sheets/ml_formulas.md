# ML Formulas Cheat Sheet

## Classification Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Balanced classes |
| Precision | TP / (TP + FP) | Cost of FP is high |
| Recall | TP / (TP + FN) | Cost of FN is high |
| F1 | 2 * (P * R) / (P + R) | Balance precision/recall |
| Specificity | TN / (TN + FP) | True negative rate |

## Probabilistic Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Log Loss | -Σ[y log(p) + (1-y)log(1-p)] / n | Penalizes confident wrong predictions |
| Brier Score | Σ(y - p)² / n | MSE for probabilities |
| AUC-ROC | ∫ TPR(FPR) dFPR | Ranking quality, threshold-invariant |

## Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | Σ(y - ŷ)² / n | Sensitive to outliers |
| RMSE | √MSE | Same units as target |
| MAE | Σ\|y - ŷ\| / n | Robust to outliers |
| MAPE | Σ\|(y - ŷ)/y\| / n | Scale-independent, avoid when y≈0 |
| R² | 1 - SSE/SST | Explained variance (can be negative!) |

## Information Theory

| Concept | Formula |
|---------|---------|
| Entropy | H(X) = -Σ p(x) log p(x) |
| Cross-Entropy | H(p,q) = -Σ p(x) log q(x) |
| KL Divergence | DKL(p\|\|q) = Σ p(x) log(p(x)/q(x)) |
| Mutual Info | I(X;Y) = H(X) - H(X\|Y) |

## Regularization

| Type | Formula | Effect |
|------|---------|--------|
| L1 (Lasso) | λ Σ\|wi\| | Sparse weights |
| L2 (Ridge) | λ Σ wi² | Small weights |
| Elastic Net | λ₁ Σ\|wi\| + λ₂ Σ wi² | Combines both |
| Dropout | Randomly zero p% of activations | Ensemble effect |

## Optimization

| Concept | Formula/Update |
|---------|----------------|
| SGD | w ← w - α ∇L |
| Momentum | v ← βv + ∇L; w ← w - αv |
| Adam | Adaptive lr per parameter |
| Weight Decay | w ← w - α(∇L + λw) |

## Distance/Similarity

| Measure | Formula | Use Case |
|---------|---------|----------|
| Euclidean | √Σ(ai - bi)² | Dense vectors |
| Cosine | (a·b) / (\|a\|\|b\|) | Direction similarity |
| Manhattan | Σ\|ai - bi\| | Grid-like spaces |
| Jaccard | \|A∩B\| / \|A∪B\| | Sets/boolean |

## Statistical Tests

| Test | Use | Key Assumption |
|------|-----|---------------|
| t-test | Compare means | Normality, equal variance |
| χ² test | Independence | Expected counts ≥ 5 |
| KS test | Distribution comparison | Continuous data |
| Mann-Whitney U | Non-parametric means | Independent samples |

## Bias-Variance

- **Total Error** = Bias² + Variance + Irreducible Error
- **High Bias**: Underfitting, oversimplified model
- **High Variance**: Overfitting, too complex for data

## Learning Rate Rules of Thumb

- Start: 1e-3 for Adam, 1e-2 for SGD
- Decay: step, cosine, or plateau-based
- If loss diverges: lr too high
- If loss plateaus: lr too low or stuck in local min

## Batch Size Tradeoffs

| Size | Pros | Cons |
|------|------|------|
| Small (16-32) | Better generalization, more updates | Noisy gradients, slower |
| Large (256+) | Stable gradients, fast | Memory, sharp minima |

## Sequence Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| BLEU | n-gram precision | Translation |
| ROUGE | n-gram recall | Summarization |
| Perplexity | exp(cross-entropy) | Language modeling |

## Ranking Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| DCG@k | Σ (2^rel - 1) / log₂(i+1) | Cumulative gain |
| NDCG@k | DCG / Ideal DCG | Normalized 0-1 |
| MAP | Avg precision@k per query | Mean over queries |
| MRR | 1 / rank of first relevant | Reciprocal rank |

## PSI (Population Stability Index)

```
PSI = Σ (%Actual - %Expected) × ln(%Actual / %Expected)
```

- PSI < 0.1: No change
- PSI 0.1-0.25: Moderate change
- PSI > 0.25: Significant change
