"""LoRA (Low-Rank Adaptation) wrapped around a frozen Linear layer.

Interview goal: explain WHY LoRA works (rank-r update is enough to capture
task-specific adaptation) and how it saves memory/compute (only A, B are
trained; the base weight stays frozen and can be shared across tasks).

We freeze a base nn.Linear, attach a LoRA adapter (B @ A) of rank r,
train ONLY the adapter on a synthetic regression task, and verify that
the base weight is unchanged after training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=True)
        for p in self.base.parameters():
            p.requires_grad = False  # freeze the pretrained weights

        # Low-rank update: delta_W = (alpha / r) * B @ A
        # Standard init: A ~ N(0, sigma^2), B = 0  => initial delta_W = 0.
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)


def main():
    torch.manual_seed(0)
    in_dim, out_dim, n = 16, 8, 256

    # Synthetic linear task with a "target" weight we want to recover.
    W_true = torch.randn(out_dim, in_dim)
    X = torch.randn(n, in_dim)
    y = X @ W_true.T

    layer = LoRALinear(in_dim, out_dim, r=4, alpha=8.0)
    base_weight_before = layer.base.weight.detach().clone()

    trainable = [p for p in layer.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in layer.parameters())
    print(f"Trainable params (LoRA only): {n_train} / {n_total} total "
          f"({100 * n_train / n_total:.1f}%)")

    opt = torch.optim.Adam(trainable, lr=5e-2)
    for epoch in range(200):
        pred = layer(X)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Final MSE after LoRA training: {loss.item():.4f}")

    # Critical invariant: the base weight must NOT have moved.
    base_weight_after = layer.base.weight.detach()
    drift = (base_weight_after - base_weight_before).abs().max().item()
    print(f"Max drift of frozen base weight: {drift:.2e}  (must be 0)")
    assert drift == 0.0, "base weight moved — freeze is broken"

    # Show that A, B together encode a meaningful low-rank update.
    delta = layer.scaling * (layer.B @ layer.A)
    print(f"||delta_W|| (Frobenius): {delta.norm().item():.3f}")
    print("LoRA training succeeded with frozen base weights.")


if __name__ == "__main__":
    main()
