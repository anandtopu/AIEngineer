# Module 16 -- LLM Observability, Cost, and Safety

Goal: be the engineer who can stand up the non-model parts of a
production LLM application. This is the side of the stack that
determines whether your launch survives contact with real users.

## What to read / run

- `src/llm/guardrails.py` -- PII redaction, prompt injection defense,
  output policy checks.
- `src/llm/token_economics.py` -- token counting, context budgeting,
  cost calculation across models.
- `src/llm/observability_tracing.py` -- span-based tracing for RAG and
  agent pipelines.
- `src/rag/semantic_cache.py` -- cost optimization via semantic cache.
- `projects/09_llm_gateway/run.py` -- a gateway that wires all of the
  above into one service.

## Observability

Modern LLM apps are traced the same way as microservices:

- **Trace** = a single user request.
- **Span** = one unit of work (retrieve, rerank, llm.generate, tool.call).
- Spans nest into a tree; each carries attributes and a status.
- Standards: OpenTelemetry GenAI semantic conventions (2025). Vendors:
  LangSmith, LangFuse, Arize Phoenix, Helicone, Weights & Biases Weave.

**What to log on every LLM call**:
- trace_id, parent_id
- model, version, temperature, top_p, max_tokens
- prompt hash (not the prompt itself unless you've redacted PII)
- retrieved doc ids + rerank scores
- token counts (input, output) + unit cost + total cost
- latency buckets (TTFT, total)
- guardrail verdicts and any blocked/redacted content
- user id, tenant id, feature flag state

## Cost

Token economics is the most common reason a promising AI app is killed
before it ships. The three knobs:

1. **Shrink the system prompt** -- it is replayed on every call.
   Factor shared instructions into a system message; keep examples
   in a retrievable store and fetch them on demand.
2. **Cap retrieved context** -- top-3 is usually within 1 point of
   top-10 on quality, at one-third the prompt tokens.
3. **Route by intent** -- short factoids go to a cheap small model;
   synthesis goes to a big one. Two-tier routing is standard.

Then layer:

- **Semantic cache** for repeat / near-duplicate queries. Typical hit
  rates in chat deployments: 20-40%. Cache by `(query_embedding,
  model, system_prompt_version)`.
- **Prompt caching** (Anthropic / OpenAI feature): reuse server-side
  KV cache for a static prompt prefix.
- **Speculative decoding** for inference-heavy workloads (see
  `src/deep_learning/speculative_decoding.py`).
- **Distillation** for tasks with stable distributions (see
  `src/deep_learning/knowledge_distillation.py`).

## Safety and guardrails

Every production deployment has a guardrail layer. The Anthropic /
NVIDIA NeMo / Guardrails AI stacks all chain the same primitives:

### Input guardrails
- **PII detection & redaction** (emails, SSNs, cards, phones, IPs).
- **Prompt injection detection**: heuristics for phrases like
  "ignore previous instructions", role-flip templates, HTML-style
  fake system tags.
- **Topic allow-list** or refusal classifier for off-topic requests.
- **Length caps** (bounds the worst case token spend).

### Output guardrails
- **Schema validation** for structured output (see
  `src/llm/structured_output.py`).
- **Toxicity / policy classifier** pass.
- **Secret leakage regex** (never echo API keys, passwords).
- **Citation check** for RAG: if an answer makes a claim but has no
  `[n]` citation, flag or reject.

### Rate limiting and budgets
- Token bucket per user per minute.
- USD budget per tenant per day.
- Circuit breakers: if upstream LLM error rate > X%, fail fast.

### Audit logging
- Every request: user, tenant, prompt hash, policy verdicts, tokens,
  cost, latency.
- Separate log for policy violations with the raw content (so you
  can triage jailbreak attempts).

## Common interview questions

- "How do you detect and block prompt injection?"
- "Our LLM bill is $40k/month. Give me a plan to cut it 50%."
- "A user says your bot leaked their data. How do you investigate?"
  (Trace by user id, audit log, redaction-before-logging policy.)
- "Walk me through the layers of a production LLM gateway."
- "Where does OpenTelemetry fit into an agent app?"

## Drills

1. Add a **toxicity classifier stub** to
   `src/llm/guardrails.py` that rejects outputs matching a small
   blocklist and logs the verdict.
2. In `projects/09_llm_gateway/run.py`, add per-model latency
   histograms and print p50/p95 at the end of the run.
3. Sketch a dashboard: top 5 metrics you would page on at 3 AM for
   a production RAG service. (Hint: faithfulness drop, p95 latency,
   error rate, cost per 1K requests, cache hit rate.)
