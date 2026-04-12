# Project 09 -- LLM Gateway

The layer that sits between users and a raw LLM in every production
AI application. This project wires all of the non-model pieces
together into one entry point:

1. Input guardrails (PII redaction + prompt injection detection)
2. Rate limiting per user
3. Semantic cache (short-circuit near-duplicate queries)
4. Model routing by intent
5. Token budget per tenant
6. Output guardrails
7. Structured tracing you could export to OpenTelemetry

The LLM is mocked so the project runs offline. Swap `mock_llm` for
a real provider and you have 80% of a real gateway (the other 20%
is durable state + auth + infra).

## Run

```bash
python projects/09_llm_gateway/run.py
```

## What the output shows

For each synthetic request:

- Which policy fired (rate limit, guardrail, cache hit, router,
  budget, LLM call, output policy).
- Final verdict: answer, rate_limited, blocked, budget_exceeded.

At the end: per-user spend vs budget.

## Extension ideas

- Add a **durable budget store** (SQLite) so spend survives restart.
- Add **circuit breakers** on upstream provider errors, with a
  fall-through secondary provider.
- Add **prompt-cache identifiers** so the gateway can take
  advantage of server-side prompt caching when the provider
  supports it.
- Add **per-tenant policy config**: each tenant declares its own
  blocked topics and redaction rules.
- Export traces in **OpenTelemetry GenAI** semantic convention format.
- Add **p50/p95 latency histograms** by model and by intent.
- Add a **safety classifier stub** and log every block for review.
