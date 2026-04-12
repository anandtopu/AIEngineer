# System Design Drills (ML + LLM)

Use these drills to build speed and structure.

## How to use

- Pick 1 drill/day.
- Timebox: 35-45 minutes.
- Use `interview/ml_system_design.md` as the template.
- After each drill, write 5 bullets:
  - key requirements
  - primary metric + guardrails
  - biggest risks/failure modes
  - monitoring plan
  - what you’d ship in v1

## Scoring Rubric (self-check)

Score each dimension 0-2.

- Requirements clarity
- Metrics correctness
- Data + labeling realism
- Leakage/skew awareness
- Model choice justification
- Serving architecture appropriateness
- Monitoring + rollback plan
- Safety/privacy considerations

Total: 16 points.

## ML System Design Drills

1) **Fraud detection (online)**
- Constraints: p95 latency < 100ms; very imbalanced labels; adversarial actors.

2) **Search ranking**
- Two-stage system: candidate generation + ranking.
- Discuss offline eval and online A/B.

3) **Recommendations feed**
- Discuss cold start and exploration vs exploitation.

4) **Content moderation**
- Multi-modal possibility; human-in-the-loop.

5) **Churn prediction (batch)**
- Define horizon and actionability; focus on segmentation.

## LLM / RAG System Design Drills

6) **Internal knowledge base assistant**
- RAG pipeline, access control, citations, eval.

7) **Customer support copilot**
- Response suggestions, safe actions, guardrails.

8) **Code assistant for internal APIs**
- RAG + tool use, prompt injection defense.

9) **Meeting notes summarizer**
- Long context handling, privacy, evaluation.

10) **SQL agent over analytics database**
- Tool execution safety, SQL injection protection, audit logging.
