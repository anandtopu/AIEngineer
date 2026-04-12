# Project 07 -- Agent with tool use, memory, and self-correction

An end-to-end ReAct agent in pure Python. No LangChain, no API key.
The LLM is mocked with scripted trajectories so the project runs
offline; swap the mock for a real Anthropic / OpenAI call and nothing
else changes.

## What you'll learn

- How the ReAct loop looks at the parser level (Thought / Action /
  Observation / Final Answer).
- How to register tools with per-tool input validation.
- How to keep session memory so the agent can skip repeated tool calls.
- How to enforce a step budget as a hard safety rail.
- What data structure you'd emit to a tracing backend (LangSmith,
  LangFuse, OpenTelemetry).

## Run

```bash
python projects/07_agent_tool_use/run.py
```

## Extension ideas

- Replace `mock_llm` with a real Anthropic call (`anthropic` SDK)
  or OpenAI function-calling. Keep the rest of the file unchanged.
- Add a `web_search` tool over a small offline corpus and a test
  that asserts the agent calls it exactly once for queries that
  need it.
- Add a USD budget tied to `src/llm/token_economics.py` and stop
  the loop when the estimated cost exceeds it.
- Emit traces compatible with `src/llm/observability_tracing.py`
  and print a waterfall.
- Handle failures: retry tool on error, escalate to a larger model
  on repeated failures.
