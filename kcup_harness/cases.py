"""The benchmark grid, the correctness suite's cases and their input generators."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch

IGNORE_INDEX = -100
GRID_N = (256, 2048, 8192)
GRID_V = (8192, 50304, 128256, 262144)
SUITE_N = (1, 7, 256, 1000, 2048, 4096, 8192)
SUITE_V = (2, 31, 1000, 8192, 50257, 50304, 128256, 262144)
MAX_ELEMENTS = 2**31
SMALL_ELEMENTS = 2**24
DISTRIBUTIONS = ("normal", "x3", "x0.01", "negated", "spike", "offset1000", "neginf")
TARGET_PATTERNS = ("random", "first", "last", "ignore10", "all_but_one")
LARGE_COMBOS = (("normal", "random"), ("offset1000", "random"), ("normal", "ignore10"))


@dataclass(frozen=True)
class Case:
    n: int
    v: int
    distribution: str
    targets: str

    @property
    def id(self) -> str:
        return f"N{self.n}-V{self.v}-{self.distribution}-{self.targets}"

    @property
    def seed(self) -> int:
        return zlib.crc32(self.id.encode())


def benchmark_shapes() -> list[tuple[int, int]]:
    return [(n, v) for n in GRID_N for v in GRID_V]


def suite_cases(shape: tuple[int, int] | None = None, max_elements: int = MAX_ELEMENTS) -> list[Case]:
    selected = []
    for n in SUITE_N:
        for v in SUITE_V:
            if n * v > max_elements or (shape is not None and (n, v) != tuple(shape)):
                continue
            if n * v <= SMALL_ELEMENTS:
                combos = [(d, t) for d in DISTRIBUTIONS for t in TARGET_PATTERNS]
            else:
                combos = list(LARGE_COMBOS)
            selected.extend(Case(n, v, d, t) for d, t in combos)
    return selected


def mix_seed(seed: int, secret: int | None) -> int:
    """GPU MODE's Cantor pairing: the public seed tells nothing about the mixed one."""
    if secret is None:
        return seed
    return (seed + (seed + secret) * (seed + secret + 1) // 2) % (2**63 - 1)


def generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def make_targets(n: int, v: int, pattern: str, g: torch.Generator, device: torch.device) -> torch.Tensor:
    if pattern == "first":
        return torch.zeros(n, dtype=torch.int64, device=device)
    if pattern == "last":
        return torch.full((n,), v - 1, dtype=torch.int64, device=device)
    targets = torch.randint(0, v, (n,), generator=g, device=device)
    if pattern == "ignore10":
        drop = torch.rand(n, generator=g, device=device) < 0.1
        drop[0] = False
        targets[drop] = IGNORE_INDEX
    elif pattern == "all_but_one":
        keep = int(torch.randint(0, n, (1,), generator=g, device=device))
        kept = targets[keep].clone()
        targets.fill_(IGNORE_INDEX)
        targets[keep] = kept
    elif pattern != "random":
        raise ValueError(f"unknown target pattern {pattern!r}")
    return targets


def make_logits(n: int, v: int, distribution: str, targets: torch.Tensor, g: torch.Generator,
                device: torch.device) -> torch.Tensor:
    x = torch.randn(n, v, generator=g, device=device, dtype=torch.bfloat16)
    if distribution == "x3":
        x.mul_(3)
    elif distribution == "x0.01":
        x.mul_(0.01)
    elif distribution == "negated":
        x.neg_()
    elif distribution == "offset1000":
        x.add_(1000)
    elif distribution == "spike":
        cols = torch.randint(0, v, (n,), generator=g, device=device)
        x[torch.arange(n, device=device), cols] += 80
    elif distribution == "neginf":
        mask = torch.rand(n, v, generator=g, device=device) < 0.1
        valid = targets != IGNORE_INDEX
        rows = valid.nonzero().squeeze(1)
        mask[rows, targets[rows]] = False  # the target stays finite
        mask[~valid, 0] = False  # an ignored row keeps one finite entry
        x[mask] = float("-inf")
    elif distribution != "normal":
        raise ValueError(f"unknown distribution {distribution!r}")
    return x


def make_inputs(case: Case, device: torch.device, secret: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    g = generator(device, mix_seed(case.seed, secret))
    targets = make_targets(case.n, case.v, case.targets, g, device)
    return make_logits(case.n, case.v, case.distribution, targets, g, device), targets


def make_benchmark_buffers(n: int, v: int, count: int, device: torch.device, seed: int):
    """Standard-normal logits and uniform targets without ignored rows (§4.4)."""
    g = generator(device, seed)
    return [(torch.randn(n, v, generator=g, device=device, dtype=torch.bfloat16),
             torch.randint(0, v, (n,), generator=g, device=device)) for _ in range(count)]


def refill(buffers, g: torch.Generator) -> None:
    """New random data in place, so a kernel that caches by buffer address goes stale."""
    for logits, targets in buffers:
        logits.normal_(generator=g)
        targets.random_(0, logits.shape[1], generator=g)
