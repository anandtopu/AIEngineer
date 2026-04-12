"""INT8 weight quantization (symmetric and asymmetric) on a real matmul.

Interview goal: explain WHY quantization works (weight distributions are
roughly Gaussian, error per element is small, and matmul is linear so
errors don't blow up too quickly), and the difference between symmetric
(zero-point = 0) and asymmetric (zero-point != 0) schemes.

We quantize a small linear layer's weights to int8, run a forward pass
in int and in float, and report the resulting reconstruction error.
"""

from __future__ import annotations

import numpy as np


def quantize_symmetric(W: np.ndarray, n_bits: int = 8):
    qmax = 2 ** (n_bits - 1) - 1  # 127 for int8
    scale = W.abs().max() / qmax if hasattr(W, "abs") else np.abs(W).max() / qmax
    q = np.round(W / scale).clip(-qmax - 1, qmax).astype(np.int8)
    return q, float(scale)


def dequantize_symmetric(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def quantize_asymmetric(W: np.ndarray, n_bits: int = 8):
    qmin, qmax = 0, 2 ** n_bits - 1  # 0..255 for uint8
    w_min, w_max = float(W.min()), float(W.max())
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = round(qmin - w_min / scale)
    q = np.round(W / scale + zero_point).clip(qmin, qmax).astype(np.uint8)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale


def main():
    rng = np.random.default_rng(0)
    in_dim, out_dim = 256, 128

    W = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.1
    x = rng.standard_normal((4, in_dim)).astype(np.float32)
    y_fp = x @ W.T

    # Symmetric.
    q_sym, scale = quantize_symmetric(W)
    W_sym = dequantize_symmetric(q_sym, scale)
    y_sym = x @ W_sym.T

    # Asymmetric.
    q_asym, s, zp = quantize_asymmetric(W)
    W_asym = dequantize_asymmetric(q_asym, s, zp)
    y_asym = x @ W_asym.T

    def err(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(a))

    fp_bytes = W.nbytes
    int8_bytes = q_sym.nbytes
    print(f"Weight tensor: {W.shape}")
    print(f"FP32 size:  {fp_bytes:>8,} bytes")
    print(f"INT8 size:  {int8_bytes:>8,} bytes  ({fp_bytes / int8_bytes:.1f}x smaller)")
    print()
    print("Reconstruction error (relative L2):")
    print(f"  symmetric  : {err(W, W_sym):.4e}")
    print(f"  asymmetric : {err(W, W_asym):.4e}")
    print()
    print("Forward-pass error (relative L2):")
    print(f"  symmetric  : {err(y_fp, y_sym):.4e}")
    print(f"  asymmetric : {err(y_fp, y_asym):.4e}")
    print()
    print("Note: symmetric works well for zero-mean weights (typical of")
    print("trained linear layers); asymmetric is preferred for activations")
    print("after ReLU because they are non-negative.")


if __name__ == "__main__":
    main()
