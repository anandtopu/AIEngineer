"""Token economics: counting, budgeting, and costing LLM calls.

Interview goal: be able to reason about LLM cost BEFORE shipping. Every
production system has three dials:
  1. input tokens (prompt + retrieved context + history)
  2. output tokens (response + structured scaffolding)
  3. model choice (small / cheap vs big / expensive)

We implement a rough tokenizer (char / 4 heuristic, close enough to
tiktoken for back-of-envelope math), a budgeted context builder that
fits as much history as possible under a token cap, and a cost
calculator across a few 2026 price points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def approx_token_count(text: str) -> int:
    """Char / 4 is within ~15% of tiktoken for typical English text."""
    return max(1, (len(text) + 3) // 4)


# Rough 2026 pricing in USD per 1M tokens. Update as you ship.
PRICES = {
    # name:            (prompt $/M, output $/M)
    "haiku-small":     (0.25,  1.25),
    "sonnet-mid":      (3.00, 15.00),
    "opus-large":      (15.00, 75.00),
    "open-7b":         (0.10,  0.20),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES[model]
    return prompt_tokens / 1_000_000 * pin + completion_tokens / 1_000_000 * pout


@dataclass
class Message:
    role: str
    text: str

    @property
    def tokens(self) -> int:
        return approx_token_count(self.text) + 4  # role overhead


def build_context(
    system_prompt: str,
    retrieved: List[str],
    history: List[Message],
    user_query: str,
    max_context_tokens: int,
    reserve_for_output: int,
) -> List[Message]:
    """Pack as much recent history + retrieved context as fits.

    Priority: system > user_query > retrieved (top-first) > history (latest-first).
    """
    budget = max_context_tokens - reserve_for_output
    msgs: List[Message] = []

    sys_msg = Message("system", system_prompt)
    user_msg = Message("user", user_query)
    mandatory = sys_msg.tokens + user_msg.tokens
    if mandatory > budget:
        raise ValueError(
            f"system + user alone use {mandatory} tokens, over budget {budget}"
        )
    budget -= mandatory

    # Retrieved chunks first -- they're load-bearing for RAG quality.
    used_retrieved: List[Message] = []
    for chunk in retrieved:
        m = Message("context", chunk)
        if m.tokens > budget:
            break
        used_retrieved.append(m)
        budget -= m.tokens

    # Then history, most recent first.
    used_hist: List[Message] = []
    for m in reversed(history):
        if m.tokens > budget:
            break
        used_hist.append(m)
        budget -= m.tokens
    used_hist.reverse()

    return [sys_msg] + used_retrieved + used_hist + [user_msg]


def main():
    system_prompt = (
        "You are a careful retrieval-augmented assistant. "
        "Only answer from the provided context. Cite sources."
    )
    retrieved = [
        "Doc 1: LoRA injects trainable low-rank matrices into frozen layers.",
        "Doc 2: QLoRA combines 4-bit quantized weights with LoRA adapters.",
        "Doc 3: Prefix tuning is a cheaper alternative to full fine-tuning.",
        "Doc 4: Adapter tuning adds small bottleneck layers to each block.",
    ]
    history = [Message("user", f"previous question {i}: "
                               f"what about technique number {i}?") for i in range(12)]
    query = "Compare LoRA and QLoRA in terms of memory and training cost."

    context = build_context(
        system_prompt=system_prompt,
        retrieved=retrieved,
        history=history,
        user_query=query,
        max_context_tokens=500,
        reserve_for_output=150,
    )

    prompt_tokens = sum(m.tokens for m in context)
    print(f"Context built: {len(context)} messages, {prompt_tokens} tokens")
    for m in context:
        print(f"  {m.role:<8s} {m.tokens:>4d} tok  {m.text[:60]!r}")

    print("\n=== Cost per model for this call (assume 150 output tokens) ===")
    for name in PRICES:
        c = cost_usd(name, prompt_tokens, 150)
        print(f"  {name:<12s}  ${c:.5f} per request")

    print("\n=== Budget planning: 1M requests/month ===")
    for name in PRICES:
        c = cost_usd(name, prompt_tokens, 150) * 1_000_000
        print(f"  {name:<12s}  ${c:>10,.0f} / month")

    print("\nDesign levers to reduce cost:")
    print("  - shrink system prompt (it's replayed on every call)")
    print("  - cap retrieved chunks (top-3 usually as good as top-10)")
    print("  - summarise old history instead of replaying it")
    print("  - route by intent: cheap model for classification, big model for synthesis")
    print("  - semantic cache for repeat questions")


if __name__ == "__main__":
    main()
