"""Direct Preference Optimization (DPO) loss from scratch.

Interview goal: 2026 alignment interviews ask about RLHF vs DPO. Be
able to say why DPO replaced the PPO-based pipeline at most labs:

  - RLHF trains a reward model, then runs PPO against it. Stable to
    train, but brittle: needs a reward model, a reference model, and
    a carefully tuned PPO loop.
  - DPO (Rafailov et al., 2023) shows that the RLHF objective has a
    CLOSED-FORM optimum, letting you skip the reward model entirely.
    You just need pairs (prompt, chosen, rejected) and two models:
    the policy (the one you're training) and a frozen reference.

The DPO loss:

  L = -log sigmoid( beta * (
          log pi(chosen|x) - log pi_ref(chosen|x)
        - log pi(rejected|x) + log pi_ref(rejected|x)
      ) )

We implement it in PyTorch on a toy 2-class "preference" task (the
"model" is just a Linear + softmax) and verify that after training,
the policy assigns more probability to "chosen" responses than to
"rejected" responses on held-out pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyPolicy(nn.Module):
    def __init__(self, vocab: int, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def log_prob(self, context_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        # Mean pool context -> one hidden state, score all response tokens.
        h = self.embed(context_ids).mean(dim=1)          # (B, dim)
        logits = self.head(h)                            # (B, vocab)
        log_probs = F.log_softmax(logits, dim=-1)        # (B, vocab)
        return log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)  # (B,)


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean()


def main():
    torch.manual_seed(0)
    vocab = 20

    # Toy preference data. Given a context (a 3-token id sequence), the
    # "chosen" response is always the context[0] itself (copy the first
    # token). The "rejected" response is some other token.
    def make_batch(n):
        ctx = torch.randint(0, vocab, (n, 3))
        chosen = ctx[:, 0].clone()
        rejected = (chosen + torch.randint(1, vocab, (n,))) % vocab
        return ctx, chosen, rejected

    policy = TinyPolicy(vocab)
    reference = TinyPolicy(vocab)
    reference.load_state_dict(policy.state_dict())
    for p in reference.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=5e-2)

    for step in range(200):
        ctx, ch, rj = make_batch(128)
        pi_c = policy.log_prob(ctx, ch)
        pi_r = policy.log_prob(ctx, rj)
        with torch.no_grad():
            ref_c = reference.log_prob(ctx, ch)
            ref_r = reference.log_prob(ctx, rj)
        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0:
            margin = (pi_c - pi_r).mean().item()
            print(f"step {step:3d}  loss={loss.item():.4f}  margin={margin:+.3f}")

    # Evaluate
    ctx, ch, rj = make_batch(500)
    with torch.no_grad():
        pi_c = policy.log_prob(ctx, ch)
        pi_r = policy.log_prob(ctx, rj)
    wins = (pi_c > pi_r).float().mean().item()
    print(f"\nHeld-out preference win-rate: {wins*100:.1f}%")
    print("(random policy would score ~50%)")
    print("\nKey property: the frozen reference prevents the policy from")
    print("drifting arbitrarily from its pretrained behavior, which is")
    print("why DPO does not catastrophically break general capability.")


if __name__ == "__main__":
    main()
