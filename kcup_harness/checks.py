"""Contract checks and value checks for one call of a submission (§4.1, §5)."""

from __future__ import annotations

import torch

from kcup_harness.cases import IGNORE_INDEX

# Calibrated on UC3's H100 (docs/lecture-reviews/evidence/kernel-cup/calibration-2026-11-10/) with the
# starter, QuACK, torch.compile, bf16 F.cross_entropy and Liger, the loss from the three with an fp32 loss.
# Only row_sum_atol is bounded by the seven mutants of the starter: it must lie between 0.032 and 0.39.
# The other four come from the legitimate errors. loss_rtol and grad_atol are about twice the largest.
# grad_rtol is the smallest recorded rtol that bf16 F.cross_entropy passes. loss_atol is a floor that no
# measured case needs. Every mutant fails.
TOLERANCES = {"loss_atol": 1e-4, "loss_rtol": 2e-4, "grad_atol": 2.5e-4, "grad_rtol": 1.6e-2, "row_sum_atol": 6.4e-2}
_CHUNK_ELEMENTS = 2**26


def contract_problems(logits, targets, out, *, logits_before, targets_before) -> list[str]:
    if not (isinstance(out, tuple) and len(out) == 2):
        return [f"must return a tuple (loss, grad), got {type(out).__name__}"]
    loss, grad = out
    problems = [f"{name} must be a plain torch.Tensor, got {type(t).__name__}"
                for name, t in (("loss", loss), ("grad", grad)) if type(t) is not torch.Tensor]
    if problems:
        return problems
    if loss.dtype != torch.float32 or loss.dim() != 0:
        problems.append(f"loss must be a float32 scalar, got {loss.dtype} with shape {tuple(loss.shape)}")
    if grad.dtype != torch.bfloat16 or grad.shape != logits.shape:
        problems.append(f"grad must be bfloat16 {tuple(logits.shape)}, got {grad.dtype} {tuple(grad.shape)}")
    if loss.device != logits.device or grad.device != logits.device:
        problems.append("loss and grad must be on the logits' device")
    if not grad.is_contiguous():
        problems.append("grad must be contiguous")
    if grad.data_ptr() == logits.data_ptr():
        problems.append("grad must be a new tensor, not the logits")
    if not torch.equal(logits, logits_before):
        problems.append("the logits were modified")
    if not torch.equal(targets, targets_before):
        problems.append("the targets were modified")
    return problems


def value_problems(loss, grad, ref_loss, ref_grad, targets, tolerances=None) -> list[str]:
    tol = tolerances or TOLERANCES
    if not torch.isfinite(ref_loss):
        raise ValueError(f"the reference loss is not finite ({float(ref_loss)}): an all-ignored batch?")
    problems = []
    got, want = float(loss), float(ref_loss)
    if not torch.isfinite(loss):
        problems.append(f"loss is not finite ({got})")
    elif abs(got - want) > tol["loss_atol"] + tol["loss_rtol"] * abs(want):
        problems.append(f"loss {got:.7g}, expected {want:.7g}")
    n_valid = int((targets != IGNORE_INDEX).sum())
    rows = max(1, _CHUNK_ELEMENTS // grad.shape[1])
    bad, worst, nonzero_ignored, off_rows, row_sum = 0, 0.0, 0, 0, 0.0
    for g, r, t in zip(grad.split(rows), ref_grad.split(rows), targets.split(rows)):  # chunked for memory
        if not torch.isfinite(r).all():
            raise ValueError("the reference gradient is not finite")
        ignored = t == IGNORE_INDEX
        nonzero_ignored += int((g[ignored] != 0).sum())  # exact zeros, whatever the tolerances
        g, r = g.float() * n_valid, r * n_valid  # scaled by n_valid
        err = (g - r).abs()
        wrong = ~torch.isfinite(g) | (err > tol["grad_atol"] + tol["grad_rtol"] * r.abs())
        bad += int(wrong.sum())
        worst = max(worst, float(torch.nan_to_num(err, nan=float("inf")).max()))
        # A valid row's scaled gradient is softmax minus one-hot: it sums to zero. At large V most softmax
        # entries are far below grad_atol, so this is what catches a wrong softmax term there (§5).
        sums = torch.nan_to_num(g[~ignored].sum(-1), nan=float("inf")).abs()
        off_rows += int((sums > tol["row_sum_atol"]).sum())
        row_sum = max(row_sum, float(sums.max())) if sums.numel() else row_sum
    if nonzero_ignored:
        problems.append(f"{nonzero_ignored} gradient entries on ignored rows are not zero")
    if bad:
        problems.append(f"{bad} gradient entries outside the tolerance (largest error x n_valid: {worst:.3g})")
    if off_rows:
        rows_text = "1 valid row" if off_rows == 1 else f"{off_rows} valid rows"
        problems.append(f"the scaled gradient of {rows_text} does not sum to zero (largest |sum|: {row_sum:.3g})")
    return problems
