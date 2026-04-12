# LLM Agents and Production -- Interview Question Bank

2026-oriented question bank covering the ground that modern AI
Engineer loops at both FAANG and startups actually hit. Pair with
`docs/modules/14`-`18` and `src/llm/*`.

Use the format: try to answer out loud in 60-90 seconds, then read
the reference answer and score yourself.

---

## A. Agents and tool use

### 1. Walk me through the ReAct loop.
**Reference:** `Thought -> Action -> Observation -> ...` until
`Final Answer`. The LLM emits a short reasoning trace plus a tool
invocation; the runtime executes the tool and appends the result to
the prompt. It works because tool outputs give the model a grounded
anchor on each turn. Failure modes: infinite loops (enforce
`max_steps`), hallucinated arguments (validate against a JSON schema
and retry), context bloat (summarize old observations).

### 2. Function calling vs "just prompt for JSON". What's the difference?
**Reference:** Function calling is a server-side feature: the API
exposes a tool schema, constrains decoding so the output matches, and
returns a typed object. Prompting for JSON works but relies on the
model's cooperation and a post-hoc parser. Function calling eliminates
most parse errors and supports parallel tool calls.

### 3. Compare ReAct vs Plan-and-Execute.
**Reference:** ReAct interleaves thinking and acting; great for
open-ended exploration, but can loop on complex tasks. Plan-and-Execute
has a planner that emits the whole plan up front, and a worker that
executes it; reported ~3.6x speedup and ~92% task completion on
complex multi-step tasks because replanning is rare. Pick Plan-and-
Execute when the task is predictable and latency matters.

### 4. How do you add short-term and long-term memory to an agent?
**Reference:** Short-term: a scratchpad appended to the prompt for
the current session, plus structured slots for intermediate tool
outputs. Long-term: a vector store keyed by user id, retrieved on
each turn. De-duplicate and time-weight old memories. Summarize
older entries to control context bloat.

### 5. Your agent calls the calculator tool ten times in a row. Debug.
**Reference:** (a) Check the stop condition. If the parser expects a
specific `Final Answer:` token and the model varied casing, it will
loop. (b) Check whether observations are being appended. Some
implementations drop them. (c) Check the planner prompt for "always
use calculator" bias. (d) Add a tool-call cache so repeated identical
calls short-circuit. (e) Add step budget + cost cap as hard rails.

### 6. How would you evaluate an agent?
**Reference:** Offline: trajectory-level metrics on a labeled test
set -- task success rate, step count, tool-call correctness,
end-answer quality. Online: user feedback, task completion rate,
abandon rate, cost per successful task. Log every (thought, action,
observation) for post-hoc analysis; spot-check with a human rater
on a sampled slice.

---

## B. Production RAG

### 7. Design a RAG chatbot for our product documentation.
**Reference:** Go through the pipeline out loud:
1. Ingest: scrape docs, normalize, chunk (sentence-aware, 400-600
   tokens, 80 overlap), embed, index in a vector DB + keep BM25.
2. Query: guardrail input -> rewrite/expand query -> hybrid retrieve
   (BM25 + dense) -> RRF fuse -> cross-encoder rerank top 50 to 5 ->
   build context with citations -> LLM with a grounded system prompt.
3. Evaluate: RAGAS metrics on a golden set; slice by topic.
4. Ops: re-index on doc change, monitor faithfulness, log traces,
   cache semantically, budget per tenant.

### 8. Why hybrid retrieval and not pure dense?
**Reference:** Dense embeddings generalize over paraphrase but miss
exact matches (product codes, names, rare jargon). BM25 nails exact
matches. They fail on different queries; RRF fusion captures both
without needing score calibration. In practice hybrid beats either
baseline on ~every corpus I've seen, which is why it's the industry
default.

### 9. Your answers are "plausible but wrong" in production. Diagnose.
**Reference:** Measure RAGAS-style. If `faithfulness` is high but
`answer_relevancy` is low, retrieval is bringing back junk. If
`faithfulness` is low but `relevancy` is high, the LLM is
hallucinating -- tighten the prompt ("only answer from the context,
otherwise say 'I don't know'") and add a citation requirement. If
both are low, your chunking or query rewrite is broken. Always
bisect with metrics before touching the prompt.

### 10. Pros and cons of cross-encoder reranking.
**Reference:** Pros: much higher quality than bi-encoder; models like
`bge-reranker` routinely add 10-20 points of MRR. Cons: it's N
forward passes per query, so it only scales to 50-200 candidates.
Shape the pipeline as `retriever (1K) -> reranker (100 -> 5) -> LLM`.
You can also distill the reranker into a smaller model once you
know your task.

### 11. How would you keep the index fresh?
**Reference:** Treat it like a cache. (a) Event-driven re-index on
CMS / DB change notifications. (b) Scheduled re-index every N hours
for things without events. (c) Version the index; blue/green
deploy. (d) Run a small eval suite against the new index before
promoting. (e) Alert on recall drop vs the old index.

### 12. Semantic cache -- when is it dangerous?
**Reference:** When the "same" query changes meaning over time
("what's the weather today"), or when the cached answer depends on
a user-specific context (auth, tenant, locale). Always version the
cache key by model + system prompt, and exclude time-sensitive
intents. Use a conservative threshold (>=0.93 cosine) for safety-
critical domains.

---

## C. Prompt engineering and structured output

### 13. Chain-of-thought -- when does it NOT help?
**Reference:** On tasks that are already shallow (single-step
classification, straightforward extraction). The extra tokens cost
latency and can introduce errors. CoT pays off on multi-step
reasoning (arithmetic, logic, multi-hop QA). Self-consistency (sample
N chains, majority vote) pays off even more, at N times the cost.

### 14. How do you defend against prompt injection?
**Reference:** Defense in depth: (a) detect in the input guardrail
with heuristics or a classifier; (b) fence untrusted input in
delimiters that the system prompt names explicitly ("the text
between triple hashes"); (c) use "spotlighting" -- preprocess the
input to mark it as untrusted (`<untrusted>...</untrusted>`); (d)
never give a tool with destructive side effects access to untrusted
output without a human-in-the-loop confirmation; (e) output
guardrails to catch the model echoing a hostile instruction it
received.

### 15. When would you use structured output with retries?
**Reference:** Any time downstream code expects typed data --
extraction, classification with scores, tool calls. Declare a JSON
schema, prompt the model with it, validate the output, and on
failure re-prompt with the validation error included ("self-
healing"). Two retries are usually enough; more indicates a prompt
problem.

---

## D. Observability, cost, and safety

### 16. What do you log on every LLM call in production?
**Reference:** `trace_id`, `parent_span_id`, model, version,
temperature, top_p, input/output token counts, dollar cost, TTFT,
total latency, prompt hash (NOT raw prompt unless redacted),
retrieved doc ids, rerank scores, guardrail verdicts, user id,
tenant id. Emit this in OpenTelemetry GenAI semantic conventions
format.

### 17. Your LLM bill is $40K/month. Cut it in half.
**Reference:** (a) Route by intent -- cheap model for short
factoids, big model for synthesis. (b) Semantic cache (20-40% hit
rates are common). (c) Trim the system prompt (it's replayed every
call). (d) Reduce retrieved context from top-10 to top-3 and
measure quality. (e) Switch to prompt caching where available. (f)
Distill or fine-tune a smaller model for the highest-volume intents.
(g) For heavy batch workloads, move to open-weight models on your
own GPUs with vLLM + continuous batching.

### 18. How do you detect hallucination in a RAG system?
**Reference:** Offline: RAGAS faithfulness measures the fraction of
answer sentences supported by retrieved context. Online: a faithful-
ness classifier or LLM judge on a sampled slice of traffic. Also
enforce citations -- any claim without a `[n]` citation is a red
flag. Pair with user feedback loops ("was this answer helpful?")
and human-grade a daily sample.

### 19. A user reports their data leaked via the chatbot. How do you investigate?
**Reference:** (a) Find their trace by user id. (b) Confirm whether
their PII appears in logs -- if it does, you have an input redaction
gap. (c) Check whether their data was in retrieved chunks for
someone else (index tenant isolation gap). (d) Check whether the
LLM echoed their data in an output to a different user (training-
on-your-inputs problem, or a cache key that ignored tenant). (e)
Roll the fix, notify, and add a regression test that catches each
root cause.

### 20. Design an LLM gateway from scratch.
**Reference:** See `projects/09_llm_gateway/run.py` for the full
skeleton. Layers: auth -> rate limiter -> input guardrails (PII +
injection) -> semantic cache -> router (intent -> model) ->
budget check -> llm call -> output guardrails (schema, secrets,
policy) -> response + async trace log. Add circuit breakers on
upstream errors and per-tenant budgets in a durable store.

---

## E. Modern training and alignment

### 21. DPO vs PPO-RLHF. Why did DPO take over?
**Reference:** The classic RLHF pipeline needs SFT -> reward model
-> PPO. It works but is brittle: three models, a reward model
mis-specification can break everything, and PPO is finicky. DPO
(Rafailov et al., 2023) proves the RLHF optimum has a closed form
and derives a loss that only needs a frozen reference model and
the policy. Fewer moving parts, similar quality on most benchmarks,
much faster to iterate. Most open-source labs shipped DPO or KTO in
2024-2025.

### 22. Explain RoPE and why it replaced sinusoidal PE.
**Reference:** Sinusoidal / learned PE is added to the token
embedding once. RoPE rotates Q and K by an angle proportional to
position BEFORE the dot product. The resulting attention score
depends only on `(i - j)`, i.e. relative position, for free. RoPE
extrapolates further beyond training context and is compatible with
lightweight scaling tricks (position interpolation, NTK-aware,
YARN). It's now the default in LLaMA, Mistral, Qwen, Gemma, Phi.

### 23. How does KV caching speed up inference?
**Reference:** In autoregressive decoding, at step t you only need
the query for token t; K and V for all previous tokens were already
computed. Store them in a cache and avoid recomputation. Reduces
per-step cost from O(T^2) to O(T). The cost is memory: cache size
is `n_layers * n_heads * head_dim * T * 2 (for K and V)`, which
dominates at long context. Grouped-query / multi-query attention,
FP8 caches, and paged attention all target this memory cost.

### 24. When would you distill vs fine-tune vs prompt-engineer?
**Reference:** Prompt-engineer when the task is expressible in a
few in-context examples and quality is good enough. Fine-tune
(LoRA / QLoRA) when you have 1K-100K labeled examples, the task is
stable, and prompting plateaus. Distill when you've already got a
great large model but latency or cost is killing you in production
-- train a smaller model on the large model's outputs. Distillation
is compression; fine-tuning is adaptation; prompting is zero-shot
steering.

---

## F. GenAI system design drills (30-minute out-loud practice)

For each drill, follow the 9-point template in `interview/ml_system_design.md`:

1. **Design a customer-support RAG bot** for an e-commerce company.
2. **Design a code review agent** that comments on pull requests.
3. **Design a semantic search** system over internal wiki + tickets.
4. **Design an LLM gateway** that many product teams share.
5. **Design an eval system** for a chat product shipping weekly.
6. **Design a personalized tutor** that adapts to a student over time.
7. **Design an AI meeting-notes assistant** with speaker diarization.
8. **Design a text-to-SQL** assistant over a large analytics warehouse.
9. **Design an image + text search** feature using CLIP-style embeddings.
10. **Design a fine-tuning pipeline** for a customer-specific LLM.
