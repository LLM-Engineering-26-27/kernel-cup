"""Paths, the device, the submission, team.toml and the correctness marker."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path

import torch

from kcup_harness import manifest


def kit_root() -> Path:
    return Path(__file__).resolve().parents[1]


def state_dir() -> Path:
    return Path(os.environ.get("KCUP_STATE_DIR", kit_root() / ".kcup"))


def results_dir() -> Path:
    return Path(os.environ.get("KCUP_RESULTS_DIR", kit_root() / "results"))


def device() -> torch.device:
    return torch.device(os.environ.get("KCUP_DEVICE", "cuda"))


def job_id() -> str:
    return os.environ.get("KCUP_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"


def marker_path() -> Path:
    """This job's correctness marker. Each job has its own, as it has its own status file, so a --shape
    job's suite cannot replace or remove the marker a full job's benchmark is about to read."""
    return state_dir() / f"correctness-{job_id()}.json"


def secret_seed() -> int | None:
    path = os.environ.get("KCUP_SECRET_SEED_FILE")
    return int(Path(path).read_text().strip()) if path else None


def load_submission(path: str | Path | None = None):
    path = Path(path or kit_root() / "submission/kernel.py")
    sys.path.insert(0, str(path.parent))  # a submission may import its own helpers
    spec = importlib.util.spec_from_file_location("kcup_submission", path)
    module = importlib.util.module_from_spec(spec)
    before = set(sys.modules)
    sys.modules[spec.name] = module  # dataclasses and inspect.getmodule need the module registered
    spec.loader.exec_module(module)
    refuse_kit_imports(set(sys.modules) - before, path.parent)
    return module.cross_entropy_fwd_bwd


def refuse_kit_imports(names, submission: Path) -> None:
    """The final re-run mounts only submission/ and the harness, so a module the kernel imports from
    anywhere else in the kit stops the run. load_submission checks what loading the kernel imported,
    the suite's last test what its calls imported."""
    root = kit_root().resolve()
    allowed = (submission.resolve(), root / "kcup_harness")
    for name in names:
        file = getattr(sys.modules.get(name), "__file__", None)
        if file is None:
            continue
        file = Path(file).resolve()
        if file.is_relative_to(root) and not any(file.is_relative_to(a) for a in allowed):
            raise SystemExit(f"the kernel imports {name} from {file.relative_to(root)}. The final re-run "
                             "has only submission/, so move that file into submission/.")


def read_team() -> dict:
    """team.toml, its fields checked here: the results schema would refuse a wrong type only after the
    whole benchmark had run."""
    with open(kit_root() / "submission/team.toml", "rb") as handle:
        team = tomllib.load(handle)
    for key, kind, what in (("nickname", str, "a string in quotes"), ("division", str, "a string in quotes"),
                            ("language", str, "a string in quotes"), ("publish_code", bool, "true or false")):
        if not isinstance(team.get(key), kind):
            raise SystemExit(f"submission/team.toml: {key} must be {what}. The comments in the file show the "
                             "values it takes.")
    return team


def _files_block(root: Path, paths: list[str]) -> dict:
    """Blob ids and hashes of the files that exist, from the bytes this process reads."""
    present = [p for p in paths if (root / p).is_file()]
    return {"files": {p: {"blob": manifest.git_blob_id(root / p), "sha256": manifest.sha256_file(root / p)}
                      for p in present},
            "sha256": manifest.combined_sha256(root, present)}


def kernel_identity() -> dict:
    """The files the host listed, hashed again here. A file that changed or vanished since the host
    recorded it (an edit during the job) makes the measurement uncommitted."""
    status = manifest.read_status(state_dir(), job_id())
    if "submission/kernel.py" not in status["kernel_files"]:
        raise SystemExit("git tracks no submission/kernel.py in this checkout. "
                         "Commit it (git add submission/kernel.py) and run the job again.")
    root = kit_root()
    kernel = _files_block(root, status["kernel_files"])
    edited = len(kernel["files"]) != len(status["kernel_files"]) or any(
        entry["blob"] != status["kernel_blobs"].get(p) for p, entry in kernel["files"].items())
    kernel.update(head=status["head"], uncommitted=status["uncommitted"] or edited)
    return {"kernel": kernel, "harness": _files_block(root, status["harness_files"])}


def starter_identity() -> dict:
    """The comparators visit's identity: the starter as the kernel and the harness files on disk,
    hashed here, so the visit needs no status.json (the staff copy carries no git history)."""
    root = kit_root()
    harness = sorted(p.relative_to(root).as_posix() for p in (root / "kcup_harness").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    harness += [p for p in manifest.HARNESS_EXTRA if (root / p).exists()]
    return {"kernel": {**_files_block(root, ["kcup_harness/starter_kernel.py"]), "head": None, "uncommitted": False},
            "harness": _files_block(root, harness)}


def write_marker(*, passed: bool, kernel_sha256: str, harness_sha256: str, scope: str, cases: int,
                 seed_mode: str) -> None:
    """The hashes are the ones taken when the suite started, so an edit during the suite
    leaves a marker that no longer matches the files on disk."""
    path = marker_path()
    if not passed:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kernel_sha256": kernel_sha256, "harness_sha256": harness_sha256, "scope": scope,
                                "cases": cases, "seed_mode": seed_mode}, indent=2) + "\n")


def require_marker(scope: str, *, seed_mode: str | None = None) -> dict:
    """bench.py runs only after the suite passed for this kernel and this harness, at this scope or
    the full one. The final re-run also needs the suite's run with the secret seed."""
    path = marker_path()
    if not path.exists():
        raise SystemExit("no passing correctness run for this kernel in this job. Run the benchmark through "
                         "job.sh, which runs the tests first in the same job.")
    marker = json.loads(path.read_text())
    identity = kernel_identity()
    if marker["kernel_sha256"] != identity["kernel"]["sha256"]:
        raise SystemExit("the kernel changed since the tests passed: run them again")
    if marker.get("harness_sha256") != identity["harness"]["sha256"]:
        raise SystemExit("the harness changed since the tests passed: run them again")
    if marker["scope"] not in ("full", scope):
        raise SystemExit(f"the tests passed for scope {marker['scope']}, the benchmark needs {scope}")
    if seed_mode is not None and marker["seed_mode"] != seed_mode:
        raise SystemExit(f"the tests ran with the {marker['seed_mode']} seed, this run needs the {seed_mode} seed")
    return marker
