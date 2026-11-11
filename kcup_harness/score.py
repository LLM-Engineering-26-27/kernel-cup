"""Speedups, the leaderboard score and % of speed of light (§6.2)."""

from __future__ import annotations

import math

PLAUSIBLE_SOL_PERCENT = 105.0


def ideal_bytes(n: int, v: int) -> int:
    """One bf16 read of the logits, one bf16 write of the gradient, the int64 targets."""
    return 4 * n * v + 8 * n


def copy_bandwidth(n: int, v: int, t_copy: float) -> float:
    """Bytes per second of out.copy_(logits): it reads 2 * n * v bytes and writes as many."""
    return 4 * n * v / t_copy


def sol_percent(n: int, v: int, t: float, t_copy: float) -> float:
    return 100.0 * (ideal_bytes(n, v) / t) / copy_bandwidth(n, v, t_copy)


def speedup(t_reference: float, t: float) -> float:
    return t_reference / t


def geomean(values) -> float:
    values = list(values)
    if not values or not all(x > 0 and math.isfinite(x) for x in values):
        raise ValueError(f"geomean needs positive values, got {values}")
    return math.exp(sum(math.log(x) for x in values) / len(values))


def summarize(rows: list[dict]) -> dict:
    """rows: n, v, t_reference, t_submission, t_copy (seconds)."""
    return {
        "score": geomean(speedup(r["t_reference"], r["t_submission"]) for r in rows),
        "sol_percent": geomean(sol_percent(r["n"], r["v"], r["t_submission"], r["t_copy"]) for r in rows),
    }


def plausible(summary: dict) -> bool:
    """Single small shapes may beat copy (its launch overhead dominates there), the summary may not."""
    return summary["sol_percent"] <= PLAUSIBLE_SOL_PERCENT
