# ML System Design Patterns Cheat Sheet

## Two-Tower Architecture (Candidate Generation)

```
Query Tower ─┐
             ├──► Dot Product ──► Score
Item Tower ──┘
```

- **Use**: Large catalog, fast retrieval
- **Pros**: Scalable, precomputable item embeddings
- **Cons**: Limited feature cross-interaction

## Two-Stage: Candidate Generation + Ranking

```
Stage 1: Fast retrieval (heuristics, ANN, two-tower)
  ↓  Top-K candidates (100-1000)
Stage 2: Slow precise ranking (heavy model with full features)
  ↓  Final N results (10-50)
```

- **Tradeoff**: Latency vs quality
- **Re-ranking**: Diversity, business rules, freshness boosts

## Feature Store Pattern

```
Training:   Offline features ──► Training data
                  ↓
Serving:    Real-time context ──┐
            Precomputed features ─┴─► Model
```

- **Point-in-time correctness**: Prevent leakage
- **Consistency**: Same transformation train/serving

## Online vs Batch Prediction

| Aspect | Batch | Online |
|--------|-------|--------|
| Trigger | Schedule | Request |
| Latency | Minutes-hours | Milliseconds-seconds |
| Features | Precomputed | Real-time + precomputed |
| Use Case | Recommendations, churn | Search, fraud detection |

## Caching Strategies

| Level | What | TTL | Invalidation |
|-------|------|-----|---------------|
| Model | Prediction | 1-60 min | Model update |
| Feature | Embeddings | Hours-days | Data change |
| Result | Top-K lists | Minutes | User action |

## A/B Testing for ML

1. **Control**: Current model
2. **Treatment**: New model
3. **Metrics**:
   - Primary: Business metric (CTR, revenue)
   - Guardrail: Latency, error rate
   - ML: Prediction distribution
4. **Duration**: Minimum sample size for power
5. **Analysis**: Check for SRM (sample ratio mismatch)

## Shadow Mode / Canary

```
Production Traffic ──► Production Model ──► Serve
              │
              └──► Candidate Model ──► Log only (no serve)
```

- **Shadow**: Test on real traffic without risk
- **Canary**: Route 1-10% traffic, monitor, rollback if needed

## Monitoring Dimensions

| Type | What | How to Detect |
|------|------|---------------|
| Data Drift | Feature distributions | PSI, KS test, Wasserstein |
| Concept Drift | P(Y\|X) changed | Performance drop, error analysis |
| Label Shift | P(Y) changed | Class distribution monitoring |
| Upstream | Data pipeline issues | Schema validation, null checks |

## Rollback Strategies

| Strategy | When | How |
|----------|------|-----|
| Model version | Performance drop | Deploy previous version |
| Feature fallback | Feature missing | Use default/imputed values |
| Heuristic fallback | Model down | Rule-based backup |
| Circuit breaker | High error rate | Return cached/default |

## RAG Architecture Patterns

### Simple RAG
```
Query ──► Retrieve ──► LLM Prompt ──► Answer
```

### Advanced RAG
```
Query ──► Query Rewriting ──► Retrieve (Dense + Sparse)
                                    ↓
                              Re-ranking ──► Top-K Context
                                    ↓
                              Prompt + Guardrails ──► LLM ──► Answer + Citations
```

### RAG Evaluation
- **Retrieval**: Recall@k, MRR, NDCG
- **Generation**: Faithfulness, answer relevance, citation accuracy
- **End-to-end**: Task completion, user satisfaction

## LLM Serving Optimizations

| Technique | What | Tradeoff |
|-----------|------|----------|
| KV Cache | Store attention keys/values | Memory for speed |
| Quantization | Lower precision (INT8, INT4) | Quality for speed |
| Speculative Decoding | Draft model + verify | Complexity for throughput |
| Continuous Batching | Dynamic batching | Latency variability |
| Prefix Caching | Cache common prefixes | Memory for repeated prefixes |

## Cost Controls for LLMs

| Strategy | Implementation | Savings |
|----------|---------------|---------|
| Model routing | Small model for easy queries, large for hard | 50-80% |
| Response caching | Cache similar queries | 20-40% |
| Prompt compression | Summarize history | 30-50% tokens |
| Batch processing | Group requests | 20-30% |
