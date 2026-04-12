"""Project 07 -- Agent with tool use, memory, and self-correction.

An end-to-end agent that can solve multi-step questions using three
tools (calculator, knowledge base, unit converter), short-term memory,
and a retry-on-error loop. The LLM is mocked deterministically so the
project runs offline; swap `mock_llm` for any real API and nothing
else changes.

Topics demonstrated:
  - ReAct loop with strict parse + retry
  - Tool schema validation
  - Short-term memory across turns
  - Structured trace you could send to LangSmith / LangFuse
  - Hard step budget to prevent infinite loops
  - Fall-through to "I don't know" on failure
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------
# Tools
# ---------------------------------------------------------------

def tool_calc(expr: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        return "error: invalid chars"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


KB = {
    "lora": "LoRA (Low-Rank Adaptation) adds trainable low-rank matrices "
            "B @ A to frozen weights, training <1% of the parameters.",
    "rag": "RAG retrieves relevant chunks and conditions the LLM on them "
           "to reduce hallucination and ground answers in fresh data.",
    "transformer": "The Transformer (2017) replaced recurrence with "
                   "self-attention and is the backbone of modern LLMs.",
    "quantization": "Quantization reduces numerical precision of model "
                    "weights (FP32 -> INT8 / INT4) to shrink memory and "
                    "speed up inference.",
}


def tool_kb(topic: str) -> str:
    return KB.get(topic.strip().lower(), f"no entry for {topic!r}")


def tool_convert(spec: str) -> str:
    m = re.match(r"\s*([\d.]+)\s*(km|mi|kg|lb)\s*to\s*(km|mi|kg|lb)\s*", spec)
    if not m:
        return "error: use '<n> <unit> to <unit>'"
    v, a, b = float(m.group(1)), m.group(2), m.group(3)
    tab = {"km": ("length", 1000.0), "mi": ("length", 1609.34),
           "kg": ("mass", 1.0),      "lb": ("mass", 0.4536)}
    if tab[a][0] != tab[b][0]:
        return "error: dimension mismatch"
    return f"{v * tab[a][1] / tab[b][1]:.4f} {b}"


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": tool_calc,
    "knowledge_base": tool_kb,
    "convert": tool_convert,
}


# ---------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------

SYSTEM = """You are a careful agent. Each turn output EITHER:

Thought: <reasoning>
Action: <tool>
Action Input: <arg>

OR when done:

Thought: <reasoning>
Final Answer: <answer>

Available tools: {tools}
"""


@dataclass
class Step:
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None


@dataclass
class Session:
    question: str
    steps: List[Step] = field(default_factory=list)
    memory: Dict[str, str] = field(default_factory=dict)


def parse_turn(text: str) -> Step:
    thought = ""
    mt = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):)", text, re.DOTALL)
    if mt:
        thought = mt.group(1).strip()
    mf = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if mf:
        return Step(thought=thought, final_answer=mf.group(1).strip())
    ma = re.search(r"Action:\s*(.+?)\nAction Input:\s*(.+)", text)
    if ma:
        return Step(thought=thought, action=ma.group(1).strip(),
                    action_input=ma.group(2).strip())
    return Step(thought=thought)


def run_agent(llm: Callable[[str], str], question: str, max_steps: int = 6) -> Session:
    session = Session(question=question)
    prompt = SYSTEM.format(tools=", ".join(TOOLS)) + f"\nQuestion: {question}\n"

    for _ in range(max_steps):
        text = llm(prompt)
        step = parse_turn(text)
        session.steps.append(step)

        if step.final_answer:
            return session

        if not step.action or step.action not in TOOLS:
            step.observation = f"error: unknown or missing action"
            prompt += text + f"\nObservation: {step.observation}\n"
            continue

        step.observation = TOOLS[step.action](step.action_input or "")
        # Memory: store successful (tool, input, output) for reuse
        session.memory[f"{step.action}:{step.action_input}"] = step.observation
        prompt += text + f"\nObservation: {step.observation}\n"

    session.steps.append(Step(thought="budget exhausted",
                              final_answer="I could not complete the task."))
    return session


# ---------------------------------------------------------------
# Mock LLM with scripted trajectories for three test questions
# ---------------------------------------------------------------

def make_mock():
    scripts: Dict[str, List[str]] = {
        "What is LoRA, and what is 12 * 11 + 4?": [
            "Thought: First look up LoRA.\n"
            "Action: knowledge_base\n"
            "Action Input: lora",

            "Thought: Now compute 12*11+4.\n"
            "Action: calculator\n"
            "Action Input: 12*11+4",

            "Thought: I have both pieces.\n"
            "Final Answer: LoRA adds low-rank adapters to a frozen base model. "
            "12*11+4 = 136.",
        ],
        "Convert 5 km to miles and explain what RAG is.": [
            "Thought: Convert km to miles.\n"
            "Action: convert\n"
            "Action Input: 5 km to mi",

            "Thought: Look up RAG.\n"
            "Action: knowledge_base\n"
            "Action Input: rag",

            "Thought: I have both facts.\n"
            "Final Answer: 5 km is about 3.107 miles. RAG retrieves relevant "
            "chunks and conditions the LLM on them to reduce hallucination.",
        ],
        "What is (25 + 17) * 2?": [
            "Thought: Inner parens first.\n"
            "Action: calculator\n"
            "Action Input: 25+17",

            "Thought: Now multiply by 2.\n"
            "Action: calculator\n"
            "Action Input: 42*2",

            "Thought: Done.\n"
            "Final Answer: 84",
        ],
    }
    idx = {q: 0 for q in scripts}

    def llm(prompt: str) -> str:
        for q, steps in scripts.items():
            if q in prompt:
                i = idx[q]
                idx[q] = min(i + 1, len(steps) - 1)
                return steps[i]
        return "Thought: unknown question.\nFinal Answer: I don't know."

    return llm


def print_session(s: Session) -> None:
    print(f"\nQUESTION: {s.question}")
    for i, st in enumerate(s.steps, 1):
        print(f"  step {i}: thought={st.thought[:60]!r}")
        if st.action:
            print(f"          action={st.action}({st.action_input!r}) -> {st.observation!r}")
        if st.final_answer:
            print(f"          FINAL: {st.final_answer}")
    print(f"  tool calls made: {sum(1 for st in s.steps if st.action)}")
    print(f"  memory entries : {len(s.memory)}")


def main():
    llm = make_mock()
    questions = [
        "What is LoRA, and what is 12 * 11 + 4?",
        "Convert 5 km to miles and explain what RAG is.",
        "What is (25 + 17) * 2?",
    ]
    for q in questions:
        print_session(run_agent(llm, q))

    print("\n=== What this project teaches ===")
    print("  1. Clean ReAct parsing with strict regex.")
    print("  2. Tool registry with per-tool input validation.")
    print("  3. Step budget as a hard safety rail.")
    print("  4. Session memory for cross-turn caching.")
    print("  5. The data structure you'd emit to a tracer.")


if __name__ == "__main__":
    main()
