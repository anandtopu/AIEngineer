"""LLM decoding strategies: greedy, temperature, top-k, top-p (nucleus).

Interview goal: explain how a language model turns a probability vector
over the vocabulary into a single next token, and how each knob trades
off determinism vs. diversity vs. quality.

We don't need a real LM here — we operate directly on a fixed logit
vector to make the math obvious and reproducible.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def temperature_sample(logits: np.ndarray, T: float, rng) -> int:
    # T -> 0  : approaches greedy (peakier distribution)
    # T -> inf: approaches uniform (more diverse but less coherent)
    probs = softmax(logits / T)
    return int(rng.choice(len(probs), p=probs))


def top_k_sample(logits: np.ndarray, k: int, rng) -> int:
    # Keep the k highest-prob tokens, renormalize, sample.
    idx = np.argpartition(-logits, k)[:k]
    masked = np.full_like(logits, -np.inf)
    masked[idx] = logits[idx]
    probs = softmax(masked)
    return int(rng.choice(len(probs), p=probs))


def top_p_sample(logits: np.ndarray, p: float, rng) -> int:
    # Nucleus: smallest set of tokens whose cumulative prob >= p.
    probs = softmax(logits)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cum, p) + 1  # include the token that crosses p
    keep = order[:cutoff]
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked /= masked.sum()
    return int(rng.choice(len(probs), p=masked))


def repetition_penalty(logits: np.ndarray, history: list[int], penalty: float) -> np.ndarray:
    # Down-weight tokens already used (a common LLM-decoding hack).
    out = logits.copy()
    for tok in set(history):
        out[tok] = out[tok] / penalty if out[tok] > 0 else out[tok] * penalty
    return out


def main():
    rng = np.random.default_rng(42)
    vocab = ["the", "a", "an", "cat", "dog", "ran", "slept", "."]
    logits = np.array([3.5, 2.8, 1.2, 2.5, 2.4, 0.5, 0.3, -1.0])
    print("Vocab :", vocab)
    print("Probs :", np.round(softmax(logits), 3).tolist())

    print(f"\nGreedy           -> {vocab[greedy(logits)]}")

    print("\nTemperature sweep (5 samples each):")
    for T in [0.3, 1.0, 2.0]:
        seq = [vocab[temperature_sample(logits, T, rng)] for _ in range(5)]
        print(f"  T={T:>3}: {seq}")

    print("\nTop-k sweep (k=3, 5 samples):")
    seq = [vocab[top_k_sample(logits, 3, rng)] for _ in range(5)]
    print(f"  {seq}")

    print("\nTop-p (nucleus) sweep (5 samples):")
    for p in [0.5, 0.9]:
        seq = [vocab[top_p_sample(logits, p, rng)] for _ in range(5)]
        print(f"  p={p}: {seq}")

    print("\nRepetition penalty: down-weight 'the' after using it twice")
    penalised = repetition_penalty(logits, history=[0, 0], penalty=2.0)
    print(f"  before: {vocab[greedy(logits)]}   after: {vocab[greedy(penalised)]}")


if __name__ == "__main__":
    main()
