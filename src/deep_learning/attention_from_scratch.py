"""Scaled dot-product and multi-head attention from scratch (numpy).

Interview goal: be able to derive attention on a whiteboard and explain
the role of Q, K, V, the sqrt(d_k) scaling, and why multiple heads help.

We implement:
  1. scaled_dot_product_attention(Q, K, V, mask)
  2. MultiHeadAttention as a class with split/concat
  3. A causal (autoregressive) mask demo
  4. A sanity check that single-head MHA equals plain attention
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (..., T_q, d_k)  K: (..., T_k, d_k)  V: (..., T_k, d_v)."""
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)  # (..., T_q, T_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, seed: int = 0):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.default_rng(seed)
        # Use small init so the demo is numerically tame.
        self.W_q = rng.standard_normal((d_model, d_model)) * 0.1
        self.W_k = rng.standard_normal((d_model, d_model)) * 0.1
        self.W_v = rng.standard_normal((d_model, d_model)) * 0.1
        self.W_o = rng.standard_normal((d_model, d_model)) * 0.1

    def _split(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_k)
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def __call__(self, x, mask=None):
        Q = self._split(x @ self.W_q)
        K = self._split(x @ self.W_k)
        V = self._split(x @ self.W_v)
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = self._merge(out) @ self.W_o
        return out, attn


def causal_mask(T: int) -> np.ndarray:
    """Lower-triangular mask: position t can attend to <= t."""
    return np.tril(np.ones((T, T), dtype=np.int32))


def main():
    rng = np.random.default_rng(0)
    B, T, d_model, n_heads = 2, 5, 16, 4
    x = rng.standard_normal((B, T, d_model)).astype(np.float64)

    mha = MultiHeadAttention(d_model, n_heads)
    mask = causal_mask(T)[None, None]  # broadcast over (B, H)
    out, attn = mha(x, mask=mask)

    print(f"Input shape:        {x.shape}")
    print(f"Output shape:       {out.shape}")
    print(f"Attention shape:    {attn.shape}  (B, H, T_q, T_k)")
    print(f"Causal check (row 0 attends only to col 0):")
    print(np.round(attn[0, 0], 3))

    # Each row of attention weights should sum to 1.
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), "attention rows must sum to 1"
    # Causal mask: above-diagonal weights must be 0.
    upper = attn[0, 0] * (1 - causal_mask(T))
    assert np.allclose(upper, 0.0), "causal mask leaked future tokens"
    print("\nAll attention sanity checks passed.")


if __name__ == "__main__":
    main()
