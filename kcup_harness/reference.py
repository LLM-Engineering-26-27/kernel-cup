"""The slow reference every speedup is measured against, and the oracle the suite checks with.

They are separate on purpose: a change to one cannot hide an error in the other (§4.3).
"""

from __future__ import annotations

import torch

from kcup_harness.cases import IGNORE_INDEX


@torch.enable_grad()
def eager_reference(logits: torch.Tensor, targets: torch.Tensor):
    """The eager fp32 chain plus autograd: the baseline of the score (1.0x)."""
    x = logits.detach().float().requires_grad_(True)
    m = x.amax(dim=-1, keepdim=True).detach()
    lse = (m + (x - m).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)
    valid = targets != IGNORE_INDEX
    picked = x.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    loss = torch.where(valid, lse - picked, torch.zeros_like(lse)).sum() / valid.sum()
    loss.backward()
    return loss.detach(), x.grad.to(torch.bfloat16)


@torch.no_grad()
def oracle(logits: torch.Tensor, targets: torch.Tensor):
    """The exact loss (fp32 scalar) and gradient (fp32) in the stable closed form."""
    x = logits.to(torch.float32, copy=True)  # the closed form below works in place
    valid = targets != IGNORE_INDEX
    n_valid = valid.sum()
    index = targets.clamp_min(0)
    lse = torch.logsumexp(x, dim=-1)
    picked = x.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    loss = torch.where(valid, lse - picked, torch.zeros_like(lse)).sum() / n_valid
    grad = x.sub_(lse.unsqueeze(-1)).exp_()
    rows = valid.nonzero().squeeze(1)
    grad[rows, index[rows]] -= 1
    grad.mul_((valid.float() / n_valid).unsqueeze(-1))
    return loss, grad
