"""Knowledge distillation: compressing a teacher into a smaller student.

Interview goal: distillation is the dominant way to ship production
models once you have a great-but-slow teacher. Be able to explain:

  - Hard targets are class labels (cross-entropy vs ground truth).
  - Soft targets are the teacher's SOFTMAX OUTPUT (probabilities over
    every class). They carry dark knowledge -- relationships between
    classes that one-hot labels throw away.
  - Distillation loss (Hinton et al., 2015):
        L = alpha * CE(student, labels)
          + (1-alpha) * T^2 * KL(soft_student || soft_teacher)
    with temperature T > 1 to flatten the soft targets.

We train a tiny teacher on a 10-class synthetic task, then distill it
into a 4x smaller student and show the student beats one trained with
hard labels alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_data(n: int, seed: int = 0):
    torch.manual_seed(seed)
    X = torch.randn(n, 32)
    # Non-linear teacher target: sign of random projections into 10 buckets.
    W = torch.randn(32, 10)
    y = (X @ W).argmax(dim=1)
    return X, y


class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, epochs, loss_fn, opt):
    for _ in range(epochs):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    X_tr, y_tr = make_data(2000, seed=0)
    X_te, y_te = make_data(500,  seed=1)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    # --- 1. Train the teacher (big MLP) ---
    teacher = MLP(hidden=128)
    train(teacher, loader, epochs=8,
          loss_fn=nn.CrossEntropyLoss(),
          opt=torch.optim.Adam(teacher.parameters(), lr=1e-3))
    t_acc = accuracy(teacher, X_te, y_te)
    print(f"Teacher ({count_params(teacher):,} params): test acc = {t_acc:.3f}")

    # --- 2. Small student trained with HARD labels only ---
    student_hard = MLP(hidden=32)
    train(student_hard, loader, epochs=8,
          loss_fn=nn.CrossEntropyLoss(),
          opt=torch.optim.Adam(student_hard.parameters(), lr=1e-3))
    h_acc = accuracy(student_hard, X_te, y_te)
    print(f"Student hard  ({count_params(student_hard):,} params): test acc = {h_acc:.3f}")

    # --- 3. Same-size student distilled from the teacher ---
    student_dist = MLP(hidden=32)
    opt = torch.optim.Adam(student_dist.parameters(), lr=1e-3)
    for _ in range(8):
        for xb, yb in loader:
            with torch.no_grad():
                t_logits = teacher(xb)
            s_logits = student_dist(xb)
            loss = distillation_loss(s_logits, t_logits, yb, T=4.0, alpha=0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()
    d_acc = accuracy(student_dist, X_te, y_te)
    print(f"Student dist  ({count_params(student_dist):,} params): test acc = {d_acc:.3f}")

    compression = count_params(teacher) / count_params(student_dist)
    print(f"\nCompression ratio: {compression:.1f}x")
    print(f"Accuracy gap closed by distillation: "
          f"{(d_acc - h_acc)*100:+.1f} pts vs hard-label student")
    print("\nThis is the core pattern behind DistilBERT, DistilGPT,")
    print("and most 'small chat' models shipped by labs in 2025-2026.")


if __name__ == "__main__":
    main()
