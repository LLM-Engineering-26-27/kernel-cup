"""Times the submission against the reference and the comparators and writes the results JSON.

    python -m kcup_harness.bench --full
    python -m kcup_harness.bench --shape N V
    python -m kcup_harness.bench --final --round R          (staff re-run: reference, copy, submission)
    python -m kcup_harness.bench --comparators-only --round R
    python -m kcup_harness.bench --profile-call N V [--torch-trace PATH]

The protocol is section 6.1 of the design, after Bonus 3's benchmark_softmax.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
import zlib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import jsonschema
import torch

from kcup_harness import KIT_VERSION, cases, census, checks, comparators, loading, reference, score

LOOPS = 5
MIN_LOOP_SECONDS = 0.05
MAX_REPS = 100
SAMPLED_CHECKS = 2
PACKAGES = ("torch", "triton", "nvidia-cutlass-dsl", "quack-kernels", "liger-kernel")
SCHEMA = Path(__file__).with_name("schema.json")


def l2_size(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).L2_cache_size
    return 32 * 2**20


def buffer_count(n: int, v: int, *, l2_bytes: int) -> int:
    """Enough rotating buffers that their logits total at least four times the L2."""
    return max(1, math.ceil(4 * l2_bytes / (2 * n * v)))


def copy_impl(logits, targets):
    """The speed of light: moves exactly the ideal bytes of the logits and the gradient."""
    return torch.empty_like(logits).copy_(logits)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, run) -> tuple[float, float]:
    """(device seconds, wall seconds) of run(). CUDA events on the current stream, synchronized
    once, inside a wall clock that includes the final synchronize. On CPU both are the wall clock."""
    _sync(device)  # earlier work (a refill) stays outside the wall clock
    t0 = time.perf_counter()
    if device.type != "cuda":
        run()
        wall = time.perf_counter() - t0
        return wall, wall
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    return start.elapsed_time(end) / 1e3, wall


def time_impl(fn, buffers, *, device: torch.device, gen: torch.Generator) -> dict:
    """LOOPS loops of back-to-back calls over the rotating buffers, refilled between loops."""
    for _ in range(2):  # compilation and caches stay outside the timing
        fn(*buffers[0])
    estimate, _ = _timed(device, lambda: fn(*buffers[0]))
    reps = min(MAX_REPS, max(len(buffers), math.ceil(MIN_LOOP_SECONDS / estimate)))

    def loop():
        for i in range(reps):
            fn(*buffers[i % len(buffers)])

    loop_s, wall_s = [], []
    for index in range(LOOPS):
        if index:
            cases.refill(buffers, gen)
        device_s, wall = _timed(device, loop)
        loop_s.append(device_s / reps)
        wall_s.append(wall / reps)
    median = statistics.median(loop_s)
    return {"median_s": median, "spread": (max(loop_s) - min(loop_s)) / median, "loop_s": loop_s,
            "wall_s": wall_s, "reps": reps}


def post_check(fn, buffers, gen: torch.Generator) -> list[str]:
    """Refill sampled buffers once more and check the outputs on them against the oracle, so a
    kernel that caches its outputs by buffer address returns stale results and fails."""
    picks = random.Random(gen.initial_seed()).sample(range(len(buffers)), min(SAMPLED_CHECKS, len(buffers)))
    problems = []
    for i in picks:
        cases.refill([buffers[i]], gen)
        logits, targets = buffers[i]
        ref_loss, ref_grad = reference.oracle(logits, targets)
        loss, grad = fn(logits, targets)
        if (loss.shape != () or loss.dtype != torch.float32
                or grad.shape != logits.shape or grad.dtype != torch.bfloat16):  # a stale output of another shape
            problems.append(f"buffer {i}: expected a float32 scalar loss and a bfloat16 {tuple(logits.shape)} grad, "
                            f"got {loss.dtype} {tuple(loss.shape)} and {grad.dtype} {tuple(grad.shape)}")
        else:
            problems += [f"buffer {i}: {p}" for p in checks.value_problems(loss, grad, ref_loss, ref_grad, targets)]
        del ref_grad, grad
    return problems


def _nvidia_smi(fields: str) -> list[str]:
    """The first visible GPU's values of an nvidia-smi query (a job sees only its own GPU)."""
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True).stdout
    return [value.strip() for value in out.splitlines()[0].split(",")]


def _sm_clock(device: torch.device) -> int | None:
    return int(_nvidia_smi("clocks.sm")[0]) if device.type == "cuda" else None


def _scores(n: int, v: int, t_reference: float, t: float, t_copy: float) -> dict:
    return {"speedup": score.speedup(t_reference, t), "sol_percent": score.sol_percent(n, v, t, t_copy),
            "copy_bandwidth": score.copy_bandwidth(n, v, t_copy)}


def bench_shape(n: int, v: int, impls: dict, *, device: torch.device, seed: int,
                checked=("submission", "starter"), l2_bytes: int | None = None) -> dict:
    """One shape: every implementation timed on the same buffers, the checked ones post-checked.
    The speedup and % of speed of light are the submission's, or the starter's when it stands in.
    A comparator that raises is recorded as an error, so it does not cost the pair the job."""
    subject = "submission" if "submission" in impls else "starter"
    count = buffer_count(n, v, l2_bytes=l2_size(device) if l2_bytes is None else l2_bytes)
    buffers = cases.make_benchmark_buffers(n, v, count, device, seed)
    gen = cases.generator(device, seed + 1)
    results = {}
    for name, fn in impls.items():
        if not comparators.supports(name, n, v):
            continue
        try:
            result = time_impl(fn, buffers, device=device, gen=gen)
            if name in checked:
                result["check"] = post_check(fn, buffers, gen)
        except Exception as err:
            if name in ("reference", "copy", subject):
                raise
            results[name] = {"error": f"{type(err).__name__}: {err}"}
        else:
            result["sm_clock_mhz"] = _sm_clock(device)
            results[name] = result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    median = {name: results[name]["median_s"] for name in ("reference", "copy", subject)}
    return {"n": n, "v": v, "buffers": count, "impls": results,
            **_scores(n, v, median["reference"], median[subject], median["copy"])}


def environment(device: torch.device) -> dict:
    env = {"device": device.type, "gpu": None, "gpu_memory_bytes": None, "sm_count": None, "l2_bytes": None,
           "driver": None, "sm_clock_mhz": None, "max_sm_clock_mhz": None, "max_memory_clock_mhz": None,
           "torch": torch.__version__, "cuda": torch.version.cuda, "packages": {},
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(), "hostname": socket.gethostname()}
    for package in PACKAGES:
        try:
            env["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env["packages"][package] = "absent"
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        driver, sm, max_sm, max_memory = _nvidia_smi("driver_version,clocks.sm,clocks.max.sm,clocks.max.memory")
        env.update(gpu=props.name, gpu_memory_bytes=props.total_memory, sm_count=props.multi_processor_count,
                   l2_bytes=props.L2_cache_size, driver=driver, sm_clock_mhz=int(sm), max_sm_clock_mhz=int(max_sm),
                   max_memory_clock_mhz=int(max_memory))
    return env


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_row(row: dict, names) -> None:
    cells = [row["impls"].get(k, {}) for k in names]
    times = "".join(f"{c['median_s'] * 1e3:>15.3f}" if "median_s" in c else f"{'error' if c else '-':>15}"
                    for c in cells)
    print(f"{row['n']:>6} x {row['v']:<7}{times}{row['speedup']:>9.2f}x{row['sol_percent']:>9.1f}%", flush=True)
    for name, result in row["impls"].items():
        if "error" in result:
            print(f"    {name} failed: {result['error']}", flush=True)


def run(mode: str, *, shape: tuple[int, int] | None = None, round_: int | None = None) -> dict:
    started = _now()
    device = loading.device()
    team_toml = loading.read_team()
    team = {key: team_toml[key] for key in ("nickname", "division", "language")}
    impls = {"reference": reference.eager_reference, "copy": copy_impl}
    if mode == "comparators":  # the starter takes the submission's place. It needs no marker and no status.json
        subject, correctness, identity = "starter", None, loading.starter_identity()
        impls.update(comparators.TIMED)
    else:
        subject = "submission"
        correctness = loading.require_marker(f"{shape[0]},{shape[1]}" if mode == "shape" else "full",
                                             seed_mode="secret" if mode == "final" else None)
        impls["submission"] = loading.load_submission()
        identity = loading.kernel_identity()
        if mode != "final":
            impls.update(comparators.TIMED)
    shapes = [tuple(shape)] if mode == "shape" else cases.benchmark_shapes()
    secret = loading.secret_seed()

    print(f"{'shape':<16}" + "".join(f"{k + ' ms':>15}" for k in impls) + f"{'speedup':>10}{'% SoL':>10}")
    rows, violations, calls = [], [], {}
    for n, v in shapes:
        key = f"{n},{v}"
        seed = cases.mix_seed(zlib.crc32(f"bench-{key}".encode()), secret)
        row = bench_shape(n, v, impls, device=device, seed=seed)
        rows.append(row)
        violations += [f"{key}: {p}" for p in row["impls"][subject]["check"]]
        logits, targets = cases.make_benchmark_buffers(n, v, 1, device, seed)[0]
        calls[key] = census.profile_call(impls[subject], logits, targets, device=device)
        del logits, targets
        violations += [f"{key}: {p}" for p in census.violations(calls[key], n, device=device)]
        _print_row(row, impls)

    summary = score.summarize([{"n": r["n"], "v": r["v"], "t_reference": r["impls"]["reference"]["median_s"],
                                "t_submission": r["impls"][subject]["median_s"],
                                "t_copy": r["impls"]["copy"]["median_s"]} for r in rows])
    summary["full_grid"] = {(r["n"], r["v"]) for r in rows} >= set(cases.benchmark_shapes())
    results = {
        "schema_version": 1, "kit_version": KIT_VERSION, "mode": mode, "round": round_,
        "started_at": started, "finished_at": _now(), "team": team, **identity,
        "environment": environment(device), "correctness": correctness,
        "census": {"passed": not violations, "violations": violations,
                   "kernels_per_call": {k: census.kernels_per_call(record) for k, record in calls.items()},
                   "calls": calls},
        "shapes": rows, "summary": summary,
    }
    jsonschema.validate(results, json.loads(SCHEMA.read_text()))
    stamp = started.replace("-", "").replace(":", "")
    name = f"{stamp}-{mode}-r{round_}.json" if round_ is not None else f"{stamp}-{loading.job_id()}.json"
    path = loading.results_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")

    grid = "full grid" if summary["full_grid"] else "partial (the leaderboard ignores it)"
    print(f"\nscore {summary['score']:.2f}x, {summary['sol_percent']:.1f}% of speed of light, {grid}")
    for problem in violations:
        print(f"census: {problem}")
    print(f"wrote {path}")
    return results


def profile_call(n: int, v: int, torch_trace: str | Path | None = None) -> None:
    """One warm-up call, then three calls inside NVTX ranges "kcup-call" for nsys and ncu."""
    device = loading.device()
    fn = loading.load_submission()
    logits, targets = cases.make_benchmark_buffers(n, v, 1, device, seed=0)[0]
    fn(logits, targets)
    _sync(device)

    def calls():
        for _ in range(3):
            with torch.cuda.nvtx.range("kcup-call"):
                fn(logits, targets)
        _sync(device)

    if torch_trace is None:
        calls()
        return
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        calls()
    prof.export_chrome_trace(str(torch_trace))
    print(profile_table(prof))
    print(f"wrote {torch_trace}")


def profile_table(prof) -> str:
    """Every operator and kernel, by CUDA time. PyTorch's kernel names differ only after their
    first 55 characters (the default width), so the names are printed up to 160 characters."""
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1, max_name_column_width=160)


def fake_results_for_tests() -> dict:
    """A complete results dict that schema.json accepts: an eligible H100 run whose grid is one
    shape. The board's tests start from it."""
    n, v = 8192, 128256
    ms = {"reference": 56.48, "copy": 1.933, "submission": 3.1, "starter": 9.5, "torch_bf16": 8.419,
          "compile": 6.2, "quack": 2.973}
    impls = {name: {"median_s": t / 1e3, "spread": 0.01, "loop_s": [t / 1e3] * LOOPS, "wall_s": [t / 1e3] * LOOPS,
                    "reps": 10, "sm_clock_mhz": 1980} for name, t in ms.items()}
    impls["submission"]["check"] = impls["starter"]["check"] = []

    def files(*paths):
        return {p: {"blob": hashlib.sha1(p.encode()).hexdigest(), "sha256": hashlib.sha256(p.encode()).hexdigest()}
                for p in paths}

    key = f"{n},{v}"
    t_ref, t_sub, t_copy = (ms[k] / 1e3 for k in ("reference", "submission", "copy"))
    return {
        "schema_version": 1, "kit_version": KIT_VERSION, "mode": "full", "round": None,
        "started_at": "2026-12-01T10:00:00Z", "finished_at": "2026-12-01T10:08:00Z",
        "team": {"nickname": "fake", "division": "Unassisted", "language": "CuTe DSL"},
        "kernel": {"files": files("submission/kernel.py"), "sha256": hashlib.sha256(b"kernel").hexdigest(),
                   "head": hashlib.sha1(b"head").hexdigest(), "uncommitted": False},
        "harness": {"files": files("kcup_harness/bench.py", "job.sh"), "sha256": hashlib.sha256(b"harness").hexdigest()},
        "environment": {"device": "cuda", "gpu": "NVIDIA H100", "gpu_memory_bytes": 99_975_000_000, "sm_count": 132,
                        "l2_bytes": 60 * 2**20, "driver": "615.71.09", "sm_clock_mhz": 1980, "max_sm_clock_mhz": 1980,
                        "max_memory_clock_mhz": 2619, "torch": "2.13.0a0+9186a08b2c.nv26.07", "cuda": "13.3",
                        "packages": {p: "1.0" for p in PACKAGES}, "slurm_job_id": "1234567", "node": "uc3n105",
                        "hostname": "uc3n105"},
        "correctness": {"kernel_sha256": hashlib.sha256(b"kernel").hexdigest(),
                        "harness_sha256": hashlib.sha256(b"harness").hexdigest(), "scope": "full", "cases": 400,
                        "seed_mode": "public"},
        "census": {"passed": True, "violations": [], "kernels_per_call": {key: 1},
                   "calls": {key: {"ops": [{"name": "aten::empty", "shapes": [[n, v]]}],
                                   "kernels": [{"name": "fill", "stream": 7}, {"name": "kernel", "stream": 7}]}}},
        "shapes": [{"n": n, "v": v, "buffers": 1, "impls": impls, **_scores(n, v, t_ref, t_sub, t_copy)}],
        "summary": {**score.summarize([{"n": n, "v": v, "t_reference": t_ref, "t_submission": t_sub,
                                        "t_copy": t_copy}]), "full_grid": True},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--full", action="store_true", help="the whole grid with every comparator")
    what.add_argument("--shape", nargs=2, type=int, metavar=("N", "V"), help="one shape during development")
    what.add_argument("--final", action="store_true", help="staff re-run: reference, copy and the submission")
    what.add_argument("--comparators-only", action="store_true", help="staff re-run: the comparators' visit")
    what.add_argument("--profile-call", nargs=2, type=int, metavar=("N", "V"), help="three calls for a profiler")
    parser.add_argument("--round", type=int, help="the final re-run's round")
    parser.add_argument("--torch-trace", type=Path, help="with --profile-call: write a torch.profiler trace here")
    args = parser.parse_args(argv)
    if (args.final or args.comparators_only) != (args.round is not None):
        parser.error("--round goes with --final and --comparators-only, and only with them")
    if args.round is not None and args.round < 1:
        parser.error("--round must be a positive integer")
    if args.shape and tuple(args.shape) not in cases.benchmark_shapes():
        parser.error(f"--shape must be a benchmark shape: {', '.join(f'{n} {v}' for n, v in cases.benchmark_shapes())}")
    if args.torch_trace and not args.profile_call:
        parser.error("--torch-trace goes with --profile-call")
    if args.profile_call:
        profile_call(*args.profile_call, torch_trace=args.torch_trace)
    elif args.shape:
        run("shape", shape=tuple(args.shape))
    else:
        run("full" if args.full else "final" if args.final else "comparators", round_=args.round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
