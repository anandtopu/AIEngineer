"""A minimal Transformer encoder block in PyTorch.

Interview goal: explain the wiring of a Transformer block end-to-end:
  embeddings + positional encoding -> [MHA -> Add&Norm -> FFN -> Add&Norm] x N

This file builds the smallest version of that stack, runs a forward pass
on toy data, and checks that the output shape and gradients are sane.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-norm variant: more stable to train than the original post-norm.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(attn_out)
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids):
        x = self.pos(self.embed(ids))
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)  # (B, T, vocab)


def main():
    torch.manual_seed(0)
    vocab_size, B, T = 50, 4, 10
    model = TinyTransformer(vocab_size)
    ids = torch.randint(0, vocab_size, (B, T))
    logits = model(ids)
    print(f"Input ids:  {tuple(ids.shape)}")
    print(f"Logits:     {tuple(logits.shape)}  expected (B, T, vocab)")

    # One training step to confirm gradients flow through every layer.
    targets = torch.randint(0, vocab_size, (B, T))
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    grad_ok = all(p.grad is not None for p in model.parameters())
    print(f"Loss:       {loss.item():.4f}")
    print(f"Params:     {n_params:,}")
    print(f"Grad flow:  {'OK' if grad_ok else 'BROKEN'}")


if __name__ == "__main__":
    main()
