"""Streaming LLM output: generators, partial-JSON parsing, early stop.

Interview goal: every production chat UI streams tokens. Understand:
  - why streaming matters (perceived latency -> user satisfaction)
  - the generator pattern in Python
  - how to handle incremental JSON during a tool-calling response
  - early termination on stop sequences or token budget
"""

from __future__ import annotations

import time
from typing import Iterator


def mock_token_stream(text: str, token_size: int = 3, delay: float = 0.002) -> Iterator[str]:
    """Pretend we're the OpenAI streaming API: yield chunks over time."""
    for i in range(0, len(text), token_size):
        time.sleep(delay)
        yield text[i : i + token_size]


def stream_with_stop(source: Iterator[str], stop: str, max_chars: int) -> Iterator[str]:
    buf = ""
    for chunk in source:
        buf += chunk
        if stop and stop in buf:
            cut = buf.index(stop)
            remaining = buf[:cut]
            # yield up to stop point (minus already-yielded prefix)
            already = len(buf) - len(chunk)
            if cut > already:
                yield remaining[already:]
            return
        if len(buf) >= max_chars:
            # Respect token budget.
            yield chunk
            return
        yield chunk


def accumulate(source: Iterator[str]) -> str:
    return "".join(source)


def incremental_json_fields(source: Iterator[str], fields: list[str]) -> Iterator[tuple[str, str]]:
    """Extract values of known top-level fields as they appear in the stream.

    This is a *pragmatic* incremental parser for the common case where the
    LLM emits a flat JSON object. A real impl would use a streaming parser
    (ijson) or a state machine over the token stream.
    """
    import re
    buf = ""
    seen = set()
    for chunk in source:
        buf += chunk
        for f in fields:
            if f in seen:
                continue
            m = re.search(rf'"{f}"\s*:\s*"([^"]*)"', buf)
            if m:
                seen.add(f)
                yield f, m.group(1)


def main():
    answer = "The capital of France is Paris. It is known for the Eiffel Tower."
    print("=== Plain streaming (observe text appearing progressively) ===")
    t0 = time.perf_counter()
    first_token_at = None
    out = ""
    for chunk in mock_token_stream(answer):
        if first_token_at is None:
            first_token_at = time.perf_counter() - t0
        out += chunk
    total = time.perf_counter() - t0
    print(out)
    print(f"  TTFT (time to first token): {first_token_at*1000:.1f} ms")
    print(f"  Total:                      {total*1000:.1f} ms")

    print("\n=== Stop sequence cuts generation early ===")
    truncated = accumulate(stream_with_stop(mock_token_stream(answer), stop=".", max_chars=500))
    print(f"  stop='.' -> {truncated!r}")

    print("\n=== Incremental JSON field extraction ===")
    json_text = '{"name": "Ada", "role": "engineer", "focus": "LLM agents"}'
    stream = mock_token_stream(json_text, token_size=4)
    for field, value in incremental_json_fields(stream, ["name", "role", "focus"]):
        print(f"  as soon as available -> {field} = {value}")

    print("\nProduction notes:")
    print("  - TTFT dominates perceived latency; prioritize it over total time.")
    print("  - Always enforce a max token budget server-side.")
    print("  - Stream tool-calling arguments as they arrive to start tool")
    print("    execution in parallel with the rest of the model's output.")


if __name__ == "__main__":
    main()
