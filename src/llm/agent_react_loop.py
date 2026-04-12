"""ReAct agent loop from scratch (no LangChain).

Interview goal: every 2026 AI Engineer JD lists "agents" as a core skill.
Be able to build the loop yourself so you can reason about where it
fails in production (infinite loops, wrong tool, hallucinated args,
step budget, cost control).

The pattern, from the ReAct paper (Yao et al., 2022):
  Thought: <reasoning>
  Action: <tool_name>
  Action Input: <json args>
  Observation: <tool output>
  ... repeat until ...
  Final Answer: <answer>

We use a deterministic "mock LLM" that plays a scripted trajectory, so
this script runs offline. Swap mock_llm for a real API call and the
loop is unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------- tools ----------

def tool_calculator(expr: str) -> str:
    # SECURITY: in production, never use eval on LLM output. Parse an AST
    # or restrict to a known grammar. Here we whitelist characters.
    if not re.fullmatch(r"[0-9+\-*/()., ]+", expr):
        return "error: invalid characters"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


def tool_wiki_lookup(topic: str) -> str:
    facts = {
        "transformer": "The Transformer was introduced in 'Attention Is All You Need' (2017).",
        "lora": "LoRA adds trainable low-rank matrices to a frozen base model.",
        "bpe": "BPE is a subword tokenization scheme used by GPT-style models.",
        "faiss": "FAISS is a library for efficient similarity search over dense vectors.",
    }
    return facts.get(topic.lower().strip(), f"no article for {topic!r}")


def tool_unit_convert(spec: str) -> str:
    m = re.match(r"\s*([\d.]+)\s*(km|m|cm|mi)\s*to\s*(km|m|cm|mi)\s*", spec)
    if not m:
        return "error: use format '<value> <unit> to <unit>'"
    v, a, b = float(m.group(1)), m.group(2), m.group(3)
    to_m = {"km": 1000, "m": 1, "cm": 0.01, "mi": 1609.34}
    return f"{v * to_m[a] / to_m[b]:.4f} {b}"


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": tool_calculator,
    "wiki_lookup": tool_wiki_lookup,
    "unit_convert": tool_unit_convert,
}


# ---------- agent loop ----------

SYSTEM_PROMPT = """You are a ReAct agent. On each turn, output EITHER:

Thought: <short reasoning>
Action: <one of: {tools}>
Action Input: <single-line input to the tool>

... OR, once you have enough information:

Thought: <short reasoning>
Final Answer: <concise answer>

Do not invent tool output. Wait for the Observation."""


@dataclass
class Trace:
    question: str
    steps: List[Dict[str, str]] = field(default_factory=list)
    final: str = ""


def parse_action(text: str):
    m = re.search(r"Action:\s*(.+?)\nAction Input:\s*(.+)", text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def parse_final(text: str):
    m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    return m.group(1).strip() if m else None


def run_agent(llm: Callable[[str], str], question: str, max_steps: int = 6) -> Trace:
    prompt = SYSTEM_PROMPT.format(tools=", ".join(TOOLS)) + f"\n\nQuestion: {question}\n"
    trace = Trace(question=question)

    for step in range(max_steps):
        out = llm(prompt)
        trace.steps.append({"llm_output": out})
        final = parse_final(out)
        if final is not None:
            trace.final = final
            return trace
        parsed = parse_action(out)
        if parsed is None:
            trace.final = "error: could not parse action"
            return trace
        name, arg = parsed
        if name not in TOOLS:
            obs = f"error: unknown tool {name!r}"
        else:
            obs = TOOLS[name](arg)
        trace.steps[-1]["observation"] = obs
        prompt += out + f"\nObservation: {obs}\n"

    trace.final = "error: step budget exhausted"
    return trace


# ---------- mock LLM with canned trajectories ----------

def make_mock_llm():
    """A deterministic LLM that walks a scripted ReAct trajectory per question."""
    scripts: Dict[str, List[str]] = {
        "What is 17 * 23 plus 5?": [
            "Thought: I will compute 17*23 first.\nAction: calculator\nAction Input: 17*23",
            "Thought: Now add 5.\nAction: calculator\nAction Input: 391+5",
            "Thought: I have the result.\nFinal Answer: 396",
        ],
        "Who introduced the Transformer and how far is 3 km in miles?": [
            "Thought: First look up the Transformer.\nAction: wiki_lookup\nAction Input: transformer",
            "Thought: Now convert 3 km to miles.\nAction: unit_convert\nAction Input: 3 km to mi",
            "Thought: I have both facts.\nFinal Answer: The Transformer was introduced in the 2017 paper 'Attention Is All You Need'; 3 km is about 1.864 mi.",
        ],
    }
    state = {"idx": {q: 0 for q in scripts}}

    def llm(prompt: str) -> str:
        # Find which scripted question this prompt is working on.
        for q, steps in scripts.items():
            if q in prompt:
                i = state["idx"][q]
                state["idx"][q] = min(i + 1, len(steps) - 1)
                return steps[i]
        return "Final Answer: I don't know."
    return llm


def main():
    llm = make_mock_llm()
    for q in [
        "What is 17 * 23 plus 5?",
        "Who introduced the Transformer and how far is 3 km in miles?",
    ]:
        trace = run_agent(llm, q)
        print(f"\n=== {q} ===")
        for i, s in enumerate(trace.steps, 1):
            print(f"\n-- step {i} --")
            print(s["llm_output"])
            if "observation" in s:
                print(f"Observation: {s['observation']}")
        print(f"\nFINAL: {trace.final}")

    print("\nProduction notes:")
    print("  - Always enforce max_steps and a wall-clock budget.")
    print("  - Log every (thought, action, observation) tuple for debugging.")
    print("  - Validate tool arguments against a JSON schema before running.")
    print("  - Use function-calling APIs when available; they eliminate the parser.")


if __name__ == "__main__":
    main()
