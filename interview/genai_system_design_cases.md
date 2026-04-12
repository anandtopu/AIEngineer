# GenAI System Design Case Studies (2026)

10 full worked cases you can use as mock interviews. Each follows the
same template so you build muscle memory:

1. Clarify scope and non-functional requirements.
2. Define success metrics (business + ML).
3. Sketch data sources, labeling, and eval set.
4. Draw the pipeline from query to response.
5. Pick model(s) and discuss cost / latency tradeoffs.
6. Call out the safety / guardrail story.
7. Describe monitoring and feedback loops.
8. Discuss failure modes and mitigations.
9. Describe the v1 -> v2 roadmap.

Run each case in 25-35 minutes and grade yourself against the rubric
in `interview/system_design_drills.md`.

---

## Case 1 -- RAG chatbot for product documentation

**Prompt:** "We sell developer tools. Build a chatbot that answers
product questions from our public docs."

**Clarifications to ask:**
- Traffic? (1K/day or 1M/day changes the architecture.)
- Languages?
- Must-have latency target? (p95 <= 3s is standard.)
- Is answer correctness critical or is it informational?
- How fresh must the index be?

**Metrics:**
- Business: deflection rate (% of tickets avoided), CSAT.
- ML: faithfulness >= 0.9, answer_relevancy >= 0.8, p95 latency.

**Pipeline:** hybrid BM25 + dense retrieval -> RRF -> cross-encoder
rerank -> grounded LLM with `[n]` citation requirement -> guardrail.

**Model choice:** mid-tier closed model (sonnet / gpt-4o) for answer;
cheap embeddings (bge / openai-small) for retrieval.

**Guardrails:** PII redaction on input, injection detection, citation
enforcement on output, refusal on off-topic.

**Monitoring:** RAGAS per slice, thumbs up/down, latency + cost,
index freshness.

**Failure modes:** stale docs (mitigate with scheduled re-index),
hallucinated APIs (citation + low-temp + faithfulness gate), prompt
injection via doc content (detect and sanitize at ingest).

---

## Case 2 -- Code review agent

**Prompt:** "Build an agent that reviews pull requests and posts
inline comments."

**Clarifications:** which repos, which languages, comment style
(advisory vs blocking), how to gate merges.

**Metrics:** precision of comments (acceptance rate), recall of real
bugs (needs labeled corpus), reviewer satisfaction, time-to-review.

**Pipeline:**
1. Webhook on PR open/update.
2. Fetch diff; chunk per file and per hunk.
3. Retrieve relevant code from the rest of the repo (dense +
   symbol-aware BM25 on identifiers).
4. Agent tools: `read_file`, `search_symbol`, `run_linter`,
   `run_tests_affected_by(files)`, `post_comment(path, line, body)`.
5. LLM loop: think about the diff -> fetch context -> write comments
   -> verify each comment cites a line.
6. Output guardrails: no comments on unchanged lines, no duplicates,
   no more than N comments per PR.

**Model:** large model for planning, small fast model for comment
drafting. Consider distillation for the drafting model.

**Safety:** never run arbitrary code from the diff; sandbox tests
strictly; never auto-approve.

**Monitoring:** acceptance rate per reviewer, false-positive rate,
cost per PR. Human-rate a weekly sample.

---

## Case 3 -- Semantic search over internal wiki + tickets

**Prompt:** "Build search over Confluence + Jira + Slack threads."

Key twist: multi-source, permission-aware.

**Pipeline:**
1. Ingest each source with its own connector; normalize to a common
   `Doc(id, text, acl, updated_at, source)` schema.
2. **Permission filter** BEFORE retrieval: convert user identity to
   a set of group ids, filter index shards accordingly.
3. Hybrid retrieve + rerank.
4. Generate a grounded answer with sources.

**Pitfalls:**
- Cross-tenant leakage if ACL is applied post-retrieval. Apply it
  BEFORE or as a vector DB filter.
- Fresh Slack threads vs stale Confluence pages -- weight recency.
- De-dup near-identical content from cross-posting.

**Metrics:** search NDCG on labeled queries, click-through, time-to-
find, ACL violation rate (must be zero).

---

## Case 4 -- Shared LLM gateway for product teams

**Prompt:** "Many teams want to call LLMs. Build a gateway."

**Requirements:** auth, rate limit, cost budgets, model choice,
logging, guardrails, observability.

**Design:** see `projects/09_llm_gateway/run.py`. Key additions
vs the toy:
- Durable cost ledger (e.g. per-tenant Postgres row updated in a
  transaction with the request).
- Circuit breakers per upstream provider; fall over to a backup.
- Prompt caching for recurring system prompts.
- A policy DSL so teams can declare their own guardrails without
  editing core code.
- OpenTelemetry export to a shared tracing backend.

**Metrics:** p95 latency, error rate, cost per request, cache hit
rate, policy violation rate, per-tenant spend.

---

## Case 5 -- Eval harness for a weekly-shipping chat product

**Prompt:** "We ship every week. How do we know quality isn't
regressing?"

**Design:**
- Golden set of 200-1000 examples per capability, versioned in git.
- Automated eval on every release candidate:
  - reference-based (exact / token-F1 / ROUGE-L)
  - reference-free (faithfulness, toxicity, policy)
  - LLM-as-judge with inter-judge agreement monitoring
- Bootstrap 95% CIs, block release if CI on any key metric is
  below the current model - epsilon.
- Slice by intent, language, user persona, complexity.
- Contamination check: flag if any golden example appears in the
  model's own outputs verbatim.
- Human spot-check on a weekly sample.

**Watch out for:** judge-model bias (prefers its own family); fix
with calibration and rotate judges.

---

## Case 6 -- Personalized tutor

**Prompt:** "Build an AI tutor that helps students learn math."

**Design constraints:** must not just give the answer; must adapt to
the student; must be safe for minors.

**Pipeline:**
- Student state store: past problems, errors, concepts mastered.
- Retriever over a curriculum graph.
- Agent tools: `check_prereq(topic)`, `generate_problem(topic, difficulty)`,
  `grade_solution(problem, answer)`, `show_hint(problem, level)`.
- LLM prompted to Socratic-method: ask questions, never dump the
  answer, escalate hint level on failure.

**Safety:** strict profanity / grooming filters; never collect PII;
log parents' consent; escalate distress keywords.

**Metrics:** learning gain (pre/post test delta), engagement time,
dropout rate, hint-level distribution.

---

## Case 7 -- AI meeting-notes assistant

**Prompt:** "Transcribe meetings and produce action items."

**Pipeline:**
- ASR (speech to text) with speaker diarization.
- Streaming chunking (30-60 second windows).
- Structured extraction: `{summary, decisions, action_items[]}`.
- Per-speaker tagging of action items.
- Storage with ACL on the meeting owner.

**Evaluation:** action-item F1 vs human labels, hallucination rate
on sampled meetings, speaker-attribution accuracy.

**Scaling:** batch processing for async uploads vs streaming for
live meetings. Different models for each: fast streaming ASR vs
higher-accuracy batch ASR.

---

## Case 8 -- Text-to-SQL over analytics warehouse

**Prompt:** "Let business users ask questions in English over our
warehouse."

**Pipeline:**
1. Retrieve relevant table schemas (dense search on column names +
   descriptions).
2. Few-shot prompt with 3-5 example (question, SQL) pairs from a
   stored library.
3. Model emits SQL; validate against the schema + a read-only
   execution plan.
4. Run SQL with a **query cap** (row limit, cost limit, timeout).
5. Present results + the SQL as a citation.

**Safety:** never allow writes; blocklist DDL; role-scope the
executing user; log every query.

**Metrics:** execution success rate, result correctness (on a
labeled set), user satisfaction, cost per query.

**Failure modes:** hallucinated column names (catch with schema
validation), ambiguous questions (ask clarifying question instead
of guessing).

---

## Case 9 -- Image + text search with CLIP-style embeddings

**Prompt:** "Users search our product catalog with text OR an
uploaded image."

**Pipeline:**
- Pre-compute CLIP-style embeddings for every product image and
  description at index time. Project both into a shared space.
- Query: embed the text or image, nearest-neighbor in the shared
  space, hybrid with a BM25 text filter.
- Rerank with a cross-encoder.
- Result diversification (MMR) so the top K aren't all near-
  duplicates.

**Metrics:** recall@K on labeled queries, click-through, diversity
score, catalog coverage.

**Pitfalls:** shared space means image queries can match text-only
products. Calibrate thresholds per modality.

---

## Case 10 -- Fine-tuning pipeline for a customer-specific LLM

**Prompt:** "Customer X wants their own fine-tuned model."

**Pipeline:**
1. Intake: agree on data license, PII scope, eval set.
2. Data cleaning + deduplication + PII redaction.
3. Base model choice: open-weight (LLaMA / Qwen / Mistral) vs
   closed API FT.
4. Training: LoRA or QLoRA on a modest GPU.
5. Evaluation: customer-provided golden set + safety eval.
6. Serving: private endpoint with their ACL + audit log.
7. Drift monitoring; retrain cadence.

**Pitfalls:** overfitting to a tiny dataset, catastrophic forgetting,
contamination with public benchmarks, hidden PII in training data,
fine-tuning hides but does not remove base-model weaknesses.

**Discussion points:** when to recommend RAG instead of FT (answer:
most of the time). FT wins for style, format, and niche vocab; RAG
wins for factuality and freshness.
