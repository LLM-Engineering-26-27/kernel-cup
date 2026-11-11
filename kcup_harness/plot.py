"""The panels plot (§7): % of speed of light against V, one panel per N."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kcup_harness import loading, score  # noqa: E402

INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
HIGHLIGHT = "#2a78d6"
# (impl, label, colour, linestyle, marker); the muted lines differ in dash and marker, not in hue.
LINES = [
    ("submission", "Your kernel", HIGHLIGHT, "-", "o"),
    ("starter", "Starter", "#8a8985", "-", "s"),
    ("reference", "Eager reference", "#8a8985", "--", "^"),
    ("quack", "QuACK", "#8a8985", ":", "D"),
]
EXTRA = [
    ("torch_bf16", "PyTorch bf16", "#8a8985", "-.", "v"),
    ("compile", "torch.compile", "#8a8985", (0, (1, 3)), "P"),
]


def plot(results: dict, out: str | Path, *, all_lines: bool = False) -> matplotlib.figure.Figure:
    lines = LINES + (EXTRA if all_lines else [])
    ns = sorted({s["n"] for s in results["shapes"]})
    figure, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns) + 1.2, 4.2), dpi=150, squeeze=False,
                                sharey=True, facecolor=SURFACE)
    for ax, n in zip(axes[0], ns):
        shapes = sorted((s for s in results["shapes"] if s["n"] == n), key=lambda s: s["v"])
        ax.set_facecolor(SURFACE)
        for impl, label, colour, style, marker in lines:
            points = [(s["v"], score.sol_percent(n, s["v"], s["impls"][impl]["median_s"],
                                                 s["impls"]["copy"]["median_s"]))
                      for s in shapes if "median_s" in s["impls"].get(impl, {})]  # a failed comparator has none
            if not points:
                continue
            top = impl == "submission"
            ax.plot(*zip(*points), color=colour, linestyle=style, marker=marker, label=label,
                    linewidth=2.5 if top else 1.5, markersize=8 if top else 6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3 if top else 2)
        ax.axhline(100, color=INK_2, linestyle=(0, (5, 3)), linewidth=1.2, label="copy (speed of light)", zorder=1)
        ax.set_xscale("log")
        ax.set_title(f"N = {n}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("V (vocabulary size)", color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][0].set_ylabel("% of speed of light", color=INK_2)
    axes[0][0].set_ylim(0, 115)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center right", frameon=False, labelcolor=INK)
    summary = results["summary"]
    figure.suptitle(f"{results['team']['nickname']}: {summary['score']:.2f}x, "
                    f"{summary['sol_percent']:.1f}% of speed of light, {results['started_at'][:10]}",
                    color=INK, fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 0.86 if len(ns) > 1 else 0.78, 0.94))
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", help="a results JSON")
    parser.add_argument("--latest", action="store_true", help="the newest results/*.json")
    parser.add_argument("--all", action="store_true", help="add F.cross_entropy and torch.compile")
    args = parser.parse_args(argv)
    if args.latest == bool(args.results):
        parser.error("give a results path or --latest")
    path = Path(args.results) if args.results else max(loading.results_dir().glob("*.json"), key=lambda p: p.name)
    out = path.with_suffix(".png")
    plot(json.loads(path.read_text()), out, all_lines=args.all)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
