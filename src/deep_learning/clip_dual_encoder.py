"""CLIP-style dual encoder with contrastive (InfoNCE) loss.

Interview goal: multimodal search, image-text retrieval, and sentence
embedding models all use the SAME core recipe:

  1. Two encoders (one per modality or one shared).
  2. Project both sides into the same vector space, L2-normalize.
  3. Contrastive loss: make matched pairs similar, push unmatched apart.

The InfoNCE loss on a batch of N pairs is simply symmetric cross-
entropy over an NxN similarity matrix, with the diagonal being the
"correct" class. Training is embarrassingly simple; the intuition
scales from CLIP to sentence-BERT to ColBERT to recommender
two-tower models.

Here we train on a synthetic "text <-> structured feature" task: each
pair shares a random latent vector, so the model's job is to
discover that latent from both views.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.1):
    """Symmetric contrastive loss over a batch of N matched pairs."""
    logits = z_a @ z_b.T / temperature                 # (N, N)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def make_pairs(n: int, latent_dim: int = 8):
    torch.manual_seed(0)
    z = torch.randn(n, latent_dim)
    # View A: a linear projection + noise
    Wa = torch.randn(latent_dim, 24)
    Wb = torch.randn(latent_dim, 32)
    A = z @ Wa + 0.2 * torch.randn(n, 24)
    B = z @ Wb + 0.2 * torch.randn(n, 32)
    return A, B


def recall_at_k(z_a, z_b, k: int) -> float:
    sim = z_a @ z_b.T
    topk = sim.topk(k, dim=-1).indices
    gold = torch.arange(z_a.size(0)).unsqueeze(-1)
    return (topk == gold).any(dim=-1).float().mean().item()


def main():
    torch.manual_seed(0)
    n_train, n_test = 512, 256
    A_tr, B_tr = make_pairs(n_train)
    A_te, B_te = make_pairs(n_test + n_train)
    A_te, B_te = A_te[n_train:], B_te[n_train:]

    enc_a = Encoder(24, 16)
    enc_b = Encoder(32, 16)
    params = list(enc_a.parameters()) + list(enc_b.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)

    # Pre-training retrieval quality
    with torch.no_grad():
        r1_before = recall_at_k(enc_a(A_te), enc_b(B_te), k=1)
        r5_before = recall_at_k(enc_a(A_te), enc_b(B_te), k=5)

    for step in range(200):
        idx = torch.randperm(n_train)[:64]
        za = enc_a(A_tr[idx])
        zb = enc_b(B_tr[idx])
        loss = info_nce(za, zb, temperature=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss={loss.item():.4f}")

    with torch.no_grad():
        r1_after = recall_at_k(enc_a(A_te), enc_b(B_te), k=1)
        r5_after = recall_at_k(enc_a(A_te), enc_b(B_te), k=5)

    print(f"\nRetrieval quality (A -> B, over {n_test} candidates):")
    print(f"  before training  R@1 = {r1_before:.3f}   R@5 = {r5_before:.3f}")
    print(f"  after  training  R@1 = {r1_after :.3f}   R@5 = {r5_after :.3f}")
    print("\nThis is the full recipe: two encoders, one contrastive loss,")
    print("shared embedding space. Swap Encoder for a ViT and a text")
    print("Transformer and you have CLIP.")


if __name__ == "__main__":
    main()
