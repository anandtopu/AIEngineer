# Module 14 -- LLM Agents and Tool Use

Goal: build the agentic skill set that every 2026 AI Engineer JD lists
as a core requirement (LinkedIn Jobs on the Rise 2026 puts "agent
building" alongside LangChain and RAG as the top three). Be the person
who can actually reason about an agent loop instead of shipping a
LangChain one-liner.

## What to read / run

- `src/llm/agent_react_loop.py` -- ReAct loop from scratch.
- `src/llm/prompt_patterns.py` -- zero-shot, few-shot, CoT, self-consistency.
- `src/llm/structured_output.py` -- function-calling-style JSON with schema validation + retry.
- `src/llm/streaming_generator.py` -- streaming, stop sequences, incremental JSON.
- `projects/07_agent_tool_use/run.py` -- end-to-end agent with three tools + memory.

## Concepts

### Patterns (from the Microsoft / Weaviate / ByteByteGo literature)

1. **ReAct** -- Thought / Action / Observation loop. Simple, debuggable,
   the default starting point.
2. **Plan-and-Execute** -- a planner LLM emits a plan once, then a
   worker walks the plan. Cited to give up to ~3.6x speedup over
   sequential ReAct on complex tasks and ~92% task completion.
3. **Reflexion / Self-critique** -- after each attempt, ask the model
   to critique and retry. Expensive; use when correctness matters
   more than latency.
4. **Multi-agent orchestration** -- specialized agents (planner,
   researcher, coder, verifier) communicate through a shared bus.
   Mirrors microservices: modular, but protocols and prompts get
   hairy fast.
5. **Agentic RAG** -- the retriever is a tool the agent calls, and the
   agent can decide to re-retrieve after each thought.

### Tool use / function calling

- A "tool" is a named function with a JSON schema for its arguments.
- The model is shown the schema and emits a JSON blob matching it.
- The runtime dispatches to the function and feeds the result back as
  an `Observation`.
- Modern APIs (Anthropic, OpenAI, Gemini) do this natively -- you do
  not need the regex parser we build in the module, but you should
  understand what it's doing.

### Failure modes in production

- **Infinite loops**: always cap `max_steps` AND wall clock AND cost.
- **Hallucinated arguments**: validate every tool call against the
  schema before dispatch. Retry with the validation error in the
  prompt ("self-healing").
- **Wrong tool choice**: usually a prompt problem. Give a short
  description per tool and one example per tool.
- **Context bloat**: observations pile up. Truncate, summarize, or
  move old observations to a retrievable scratchpad.
- **Cost blowups**: budget per user per day; alert on outliers.

## Common interview questions

- Walk me through how ReAct works. Where does it fail?
- How is function calling different from just prompting for JSON?
- Compare Plan-and-Execute vs pure ReAct -- when would you pick each?
- Your agent keeps calling the same tool forever. Debug it.
- How do you handle tool outputs bigger than the context window?
- How would you add observability to an agent?

## Drills

1. Modify `agent_react_loop.py` to add a budget in USD and stop when
   the estimated cost exceeds it.
2. Extend `projects/07_agent_tool_use/run.py` with a `web_search`
   tool that's deterministic over a small offline corpus, and write
   a test that asserts the agent calls it exactly once for a query
   that needs it.
3. Swap the ReAct parser for a JSON-based "native function call"
   interface and measure how many fewer parse errors you get.
