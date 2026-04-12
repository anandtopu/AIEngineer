"""Speculative decoding: a small draft model accelerates a big target model.

Interview goal: explain how speculative decoding gets ~2-3x inference
speedup without changing the output distribution of the target model.

The trick:
  1. A cheap DRAFT model proposes k candidate tokens autoregressively.
  2. The TARGET model scores all k proposals in a SINGLE forward pass.
  3. For each proposed token, accept it with probability
        min(1, p_target(t) / p_draft(t))
     and reject the rest. This rejection-sampling rule exactly matches
     the target distribution (Leviathan et al., 2022).
  4. On rejection, resample from an adjusted residual distribution.

We simulate draft + target with deterministic mock distributions and
measure acceptance rate. Our comparison is purely conceptual: in real
systems, the target forward pass cost is what dominates, and the win
is "spend one target call to advance (accepted+1) tokens instead of 1".
"""

from __future__ import annotations

import numpy as np


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def draft_model(prefix_len: int, vocab: int, rng) -> np.ndarray:
    # Pretend a small model that is biased towards lower ids.
    logits = -np.arange(vocab).astype(float) * 0.3 + rng.normal(0, 0.1, vocab)
    return softmax(logits)


def target_model(prefix_len: int, vocab: int, rng) -> np.ndarray:
    # A stronger model with slightly different bias.
    logits = -np.arange(vocab).astype(float) * 0.25 + rng.normal(0, 0.05, vocab)
    return softmax(logits)


def speculative_step(draft_p: np.ndarray, target_p: np.ndarray, k: int, rng):
    """Propose k tokens from draft, accept/reject under target."""
    accepted = []
    for _ in range(k):
        t = int(rng.choice(len(draft_p), p=draft_p))
        r = rng.random()
        if r < min(1.0, target_p[t] / draft_p[t]):
            accepted.append(t)
        else:
            # Rejection: resample from adjusted residual
            # p'(x) proportional to max(0, target_p(x) - draft_p(x))
            resid = np.clip(target_p - draft_p, 0, None)
            if resid.sum() > 0:
                resid /= resid.sum()
                t_new = int(rng.choice(len(resid), p=resid))
            else:
                t_new = int(np.argmax(target_p))
            accepted.append(t_new)
            return accepted  # stop at first rejection
    return accepted


def main():
    rng = np.random.default_rng(0)
    vocab = 32
    k = 4  # draft proposes k tokens per target call
    n_target_calls = 200

    total_accepted = 0
    total_generated = 0
    for step in range(n_target_calls):
        d = draft_model(step, vocab, rng)
        t = target_model(step, vocab, rng)
        toks = speculative_step(d, t, k, rng)
        total_generated += len(toks)

    avg_per_call = total_generated / n_target_calls
    # A plain autoregressive decoder advances 1 token per target call.
    speedup = avg_per_call
    print(f"k (draft lookahead)     : {k}")
    print(f"target calls            : {n_target_calls}")
    print(f"tokens generated        : {total_generated}")
    print(f"avg tokens per target call: {avg_per_call:.2f}")
    print(f"theoretical speedup     : {speedup:.2f}x vs plain decoding")
    print("\nNotes:")
    print("  - The acceptance rule guarantees the output distribution")
    print("    exactly matches the target model: speculative decoding")
    print("    is LOSSLESS, not just an approximation.")
    print("  - Real systems pair a 7B draft with a 70B target, or a")
    print("    distilled small version of the target itself.")
    print("  - Modern variants: Medusa (extra heads), EAGLE (feature-")
    print("    level draft), and LookAhead decoding.")


if __name__ == "__main__":
    main()
