"""LLM observability: span-based tracing for RAG and agent pipelines.

Interview goal: when a RAG answer is wrong in production, the FIRST
question is "which stage broke?" Observability tools (LangSmith,
LangFuse, Arize Phoenix, OpenTelemetry GenAI semantic conventions)
all model a request as a TREE OF SPANS: a root request span, child
spans per retrieval / rerank / LLM / tool call, with timing and
metadata on each.

This file builds the core span primitive, a context manager that
nests spans automatically, and a tiny formatter that prints a waterfall.
Swap the print for an exporter and you have a working tracing client.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: Optional[str]
    start: float
    end: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000 if self.end else 0.0


class Tracer:
    def __init__(self):
        self.spans: List[Span] = []
        self._stack: List[str] = []

    @contextmanager
    def span(self, name: str, **attrs):
        sid = uuid.uuid4().hex[:8]
        parent = self._stack[-1] if self._stack else None
        s = Span(name=name, span_id=sid, parent_id=parent, start=time.perf_counter(), attrs=dict(attrs))
        self.spans.append(s)
        self._stack.append(sid)
        try:
            yield s
        except Exception as e:
            s.status = f"ERROR:{type(e).__name__}"
            raise
        finally:
            s.end = time.perf_counter()
            self._stack.pop()

    def print_waterfall(self):
        # Build child map
        children: Dict[Optional[str], List[Span]] = {}
        for s in self.spans:
            children.setdefault(s.parent_id, []).append(s)

        root_start = min(s.start for s in self.spans)
        total = max(s.end or s.start for s in self.spans) - root_start

        def walk(parent_id: Optional[str], depth: int):
            for s in children.get(parent_id, []):
                offset = (s.start - root_start) / total
                width = (s.duration_ms / 1000) / total
                bar_start = int(offset * 40)
                bar_width = max(1, int(width * 40))
                bar = " " * bar_start + "#" * bar_width
                bar = bar[:40].ljust(40)
                extras = " ".join(f"{k}={v}" for k, v in s.attrs.items())
                print(f"  |{bar}| {s.duration_ms:>6.1f}ms  "
                      f"{'  ' * depth}{s.name:<22s} {s.status:<8s} {extras}")
                walk(s.span_id, depth + 1)

        print(f"Total request: {total*1000:.1f} ms")
        walk(None, 0)


# ---------- demo pipeline ----------

def demo_rag(tracer: Tracer, question: str):
    with tracer.span("rag.request", user="u123") as root:
        root.attrs["question"] = question[:60]

        with tracer.span("guardrail.input", policy="v1"):
            time.sleep(0.002)

        with tracer.span("retrieve.hybrid", k=20) as s:
            with tracer.span("retrieve.bm25"):
                time.sleep(0.005)
            with tracer.span("retrieve.dense"):
                time.sleep(0.008)
            with tracer.span("retrieve.rrf_merge"):
                time.sleep(0.001)
            s.attrs["candidates"] = 20

        with tracer.span("rerank.cross_encoder", k=5):
            time.sleep(0.012)

        with tracer.span("llm.generate", model="sonnet-mid") as llm:
            time.sleep(0.040)
            llm.attrs["input_tokens"] = 820
            llm.attrs["output_tokens"] = 140

        with tracer.span("guardrail.output", policy="v1"):
            time.sleep(0.002)


def main():
    tracer = Tracer()
    demo_rag(tracer, "Compare LoRA and QLoRA for memory and cost.")
    tracer.print_waterfall()

    print("\n=== Derived metrics you should aggregate ===")
    latency_by_stage: Dict[str, float] = {}
    for s in tracer.spans:
        latency_by_stage.setdefault(s.name, 0)
        latency_by_stage[s.name] += s.duration_ms
    for name, ms in sorted(latency_by_stage.items(), key=lambda x: -x[1]):
        print(f"  {name:<22s} {ms:>6.1f} ms")

    llm_span = next(s for s in tracer.spans if s.name == "llm.generate")
    print(f"\n  model         : {llm_span.attrs['model']}")
    print(f"  input_tokens  : {llm_span.attrs['input_tokens']}")
    print(f"  output_tokens : {llm_span.attrs['output_tokens']}")

    print("\nWhat to log in production:")
    print("  - trace_id, parent_id (for fan-out across services)")
    print("  - model, version, temperature, top_p")
    print("  - prompt fingerprint (hash), retrieved doc ids, rerank scores")
    print("  - token counts + unit cost + total cost")
    print("  - guardrail verdicts and any blocked content")


if __name__ == "__main__":
    main()
