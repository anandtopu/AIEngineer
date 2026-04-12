"""Byte-Pair Encoding (BPE) tokenizer trained on a toy corpus.

Interview goal: explain how subword tokenizers (BPE / WordPiece) bridge
the gap between character-level (long sequences) and word-level (huge
vocab + OOV) representations, and why GPT-style models use BPE.

Algorithm:
  1. start with characters (+ end-of-word marker '</w>')
  2. repeatedly find the most frequent adjacent pair and merge it
  3. repeat until you reach the target vocab size

We then encode unseen words using the learned merge table.
"""

from __future__ import annotations

import collections
import re
from typing import Dict, List, Tuple


def get_stats(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    new_vocab = {}
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word, freq in vocab.items():
        joined = " ".join(word)
        merged = pattern.sub("".join(pair), joined)
        new_vocab[tuple(merged.split(" "))] = freq
    return new_vocab


def train_bpe(corpus: List[str], num_merges: int) -> Tuple[List[Tuple[str, str]], Dict]:
    # Initialize each word as a tuple of characters with </w> end marker.
    word_freq = collections.Counter(corpus)
    vocab = {tuple(list(w) + ["</w>"]): f for w, f in word_freq.items()}

    merges: List[Tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if pairs[best] < 2:
            break  # nothing left worth merging
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return merges, vocab


def encode(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    tokens = list(word) + ["</w>"]
    # Apply learned merges greedily, in the order they were learned.
    for a, b in merges:
        i = 0
        out = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def main():
    corpus = (
        "low low low low low "
        "lower lower "
        "newest newest newest newest newest newest "
        "widest widest widest"
    ).split()

    merges, vocab = train_bpe(corpus, num_merges=10)
    print("Learned merges (in order):")
    for i, m in enumerate(merges, 1):
        print(f"  {i:2d}. {m[0]!r} + {m[1]!r} -> {(m[0] + m[1])!r}")

    print("\nFinal vocab fragments:")
    fragments = sorted({tok for word in vocab for tok in word})
    print("  " + ", ".join(fragments))

    print("\nEncode unseen words:")
    for w in ["newest", "lowest", "newer", "wildest"]:
        print(f"  {w:10s} -> {encode(w, merges)}")


if __name__ == "__main__":
    main()
