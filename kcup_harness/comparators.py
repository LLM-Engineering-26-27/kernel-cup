"""The comparison lines of the plots (§6.3). Never ranked."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kcup_harness.cases import IGNORE_INDEX


def _loss_expr(x, targets):
    x = x.float()
    m = x.amax(dim=-1, keepdim=True).detach()
    lse = (m + (x - m).exp().sum(dim=-1, keepdim=True).log()).squeeze(-1)
    valid = targets != IGNORE_INDEX
    picked = x.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, lse - picked, torch.zeros_like(lse)).sum() / valid.sum()


def torch_bf16(logits, targets):
    x = logits.detach().requires_grad_(True)
    loss = F.cross_entropy(x, targets, ignore_index=IGNORE_INDEX)
    loss.backward()
    return loss.detach().float(), x.grad


_compiled = None


def compiled(logits, targets):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_loss_expr)
    x = logits.detach().requires_grad_(True)
    loss = _compiled(x, targets)
    loss.backward()
    return loss.detach(), x.grad


def quack(logits, targets):
    """QuACK's fused forward with the gradient, which leaves the logits intact. A class weight of 1 / n_valid
    scales each valid row's loss and gradient, so the mean costs no extra pass. On UC3 it beat the autograd
    path on every benchmark shape (1.4x to 3.4x) and passed the suite's cases QuACK supports (Task 15, §6.3)."""
    from quack.cross_entropy import cross_entropy_fwd

    n_valid = (targets != IGNORE_INDEX).sum()
    weight = (1.0 / n_valid.float()).expand(logits.shape[1]).contiguous()
    loss, grad = cross_entropy_fwd(logits, targets, weight=weight, ignore_index=IGNORE_INDEX, return_dx=True)
    return loss.sum(), grad


def liger_on_clone(logits, targets):
    """Calibration only: Liger writes the gradient into its input."""
    from liger_kernel.transformers import LigerCrossEntropyLoss

    x = logits.detach().clone().requires_grad_(True)
    loss = LigerCrossEntropyLoss(ignore_index=IGNORE_INDEX)(x, targets)
    loss.backward()
    return loss.detach().float(), x.grad


def starter(logits, targets):
    from kcup_harness import starter_kernel

    return starter_kernel.cross_entropy_fwd_bwd(logits, targets)


def supports(name: str, n: int, v: int) -> bool:
    """QuACK copies at least 32 bits per thread, so an odd V (one bf16 element, 16 bits) does not compile.
    Measured on UC3: V = 31 and 50,257 fail, every suite case at V = 2, 1,000, 8,192, 50,304, 128,256 and
    262,144 passes (calibration of 2026-11-10)."""
    return v % 2 == 0 if name == "quack" else True


TIMED = {"starter": starter, "torch_bf16": torch_bf16, "compile": compiled, "quack": quack}
