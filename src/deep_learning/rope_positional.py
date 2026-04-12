"""Rotary Positional Embeddings (RoPE) from scratch.

Interview goal: 2026 open-weight models (LLaMA 2/3, Mistral, Qwen,
Gemma, Phi) all use RoPE. Be able to explain why:

  - Absolute positional encodings (sinusoid or learned) are added to the
    token embedding once. They do not generalize well past the context
    length they were trained on.
  - RoPE instead *rotates* the Q and K vectors by an angle that depends
    on the position, BEFORE the dot product. Because a dot product of
    two rotated vectors only depends on the rotation DIFFERENCE, the
    attention score between positions i and j depends only on (i - j).
    That is relative position, built into attention for free.
  - RoPE extrapolates further and is compatible with "position
    interpolation" tricks like NTK-aware scaling for long context.

This file implements RoPE on a small Q, K tensor and checks the key
invariant: score(Q_i, K_j) depends only on (i - j).
"""

from __future__ import annotations

import numpy as np


def build_freqs(dim: int, base: float = 10000.0) -> np.ndarray:
    # Angular frequency for each pair of dimensions (d must be even).
    assert dim % 2 == 0
    return 1.0 / (base ** (np.arange(0, dim, 2) / dim))


def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """x: (T, d). Returns x with RoPE applied per position."""
    T, d = x.shape
    positions = np.arange(T)[:, None]          # (T, 1)
    angles = positions * freqs[None, :]        # (T, d/2)
    cos, sin = np.cos(angles), np.sin(angles)

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]
    # Rotate each (x_even, x_odd) pair by the angle:
    # [ x_even' ]   [ cos  -sin ] [ x_even ]
    # [ x_odd'  ] = [ sin   cos ] [ x_odd  ]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    out = np.empty_like(x)
    out[:, 0::2] = rot_even
    out[:, 1::2] = rot_odd
    return out


def attention_score(q: np.ndarray, k: np.ndarray) -> float:
    return float(q @ k)


def main():
    rng = np.random.default_rng(0)
    dim = 8
    T = 12
    freqs = build_freqs(dim)

    # Same underlying Q and K vectors at every position so we can isolate
    # the effect of RoPE. If RoPE works, scoring (Q at pos i, K at pos j)
    # should depend only on (i - j), not on (i, j) individually.
    base_q = rng.standard_normal(dim)
    base_k = rng.standard_normal(dim)

    Q = np.tile(base_q, (T, 1))
    K = np.tile(base_k, (T, 1))
    Qr = apply_rope(Q, freqs)
    Kr = apply_rope(K, freqs)

    print("Score(Q_i, K_j) after RoPE, i, j in [0, 5]:")
    print("     " + "  ".join(f"j={j}" for j in range(6)))
    for i in range(6):
        row = [f"{attention_score(Qr[i], Kr[j]):+.3f}" for j in range(6)]
        print(f"i={i}  " + "  ".join(row))

    print("\nKey check: diagonal (i == j) is constant:")
    diag = [attention_score(Qr[i], Kr[i]) for i in range(T)]
    print(f"  values     : {[round(x, 3) for x in diag]}")
    print(f"  std dev    : {np.std(diag):.2e}  (should be ~0)")
    assert np.std(diag) < 1e-9, "RoPE should preserve same-position score"

    print("\nKey check: score depends only on (j - i):")
    # Fix delta=2 and walk along the diagonal.
    for delta in [0, 1, 2, 3]:
        scores = [attention_score(Qr[i], Kr[i + delta]) for i in range(T - delta)]
        std = np.std(scores)
        print(f"  delta={delta}: mean={np.mean(scores):+.3f}  std={std:.2e}")
        assert std < 1e-9, f"RoPE score should not depend on absolute i for fixed delta={delta}"

    print("\nRoPE invariant holds: attention is a function of RELATIVE position.")
    print("This is why RoPE models can extrapolate beyond training context")
    print("with lightweight tricks (position interpolation, NTK-aware, YARN).")


if __name__ == "__main__":
    main()
