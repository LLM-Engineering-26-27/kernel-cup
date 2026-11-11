#!/usr/bin/env python3
# competitions/kernel_cup/kcup_harness/manifest.py
"""File lists and git state of a kit checkout, written on the host before each job (§4.2).

Git runs on the host because inside Apptainer a git worktree's .git file points into a
clone the container cannot see. Standard library only, Python 3.9 or later: the host has
no torch and may have an old python3.

    python3 kcup_harness/manifest.py write --state-dir .kcup --job-id "$SLURM_JOB_ID"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXCLUDED_FROM_KERNEL = ("submission/team.toml", "submission/WRITEUP.md")
HARNESS_PREFIX = "kcup_harness/"
HARNESS_EXTRA = ("job.sh", "pytest.ini")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"git {' '.join(args)} failed in {root}: {err.stderr.strip()}") from None


def _paths(output: str) -> list[str]:
    return sorted(p for p in output.split("\0") if p)


def tracked(root: Path) -> list[str]:
    return _paths(_git(root, "ls-files", "-z"))


def kernel_files(root: Path) -> list[str]:
    return [p for p in tracked(root) if p.startswith("submission/") and p not in EXCLUDED_FROM_KERNEL]


def harness_files(root: Path) -> list[str]:
    return [p for p in tracked(root) if p.startswith(HARNESS_PREFIX) or p in HARNESS_EXTRA]


def untracked(root: Path) -> list[str]:
    return _paths(_git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", "submission/"))


def git_blob_id(path: Path) -> str:
    """The id git gives a file: SHA-1 of the blob header and the bytes."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blob_ids(root: Path, paths: list[str]) -> dict[str, str]:
    paths = [p for p in paths if (root / p).is_file()]  # a tracked file deleted from the tree has no blob
    if not paths:
        return {}
    return dict(zip(paths, _git(root, "hash-object", "--", *paths).split()))


def build_status(root: Path, job_id: str) -> dict:
    kernel, harness = kernel_files(root), harness_files(root)
    loose = untracked(root)
    changed = bool(_git(root, "status", "--porcelain", "-z", "--", "submission/",
                        *(f":(exclude){p}" for p in EXCLUDED_FROM_KERNEL)))
    return {
        "job_id": str(job_id),
        "head": _git(root, "rev-parse", "HEAD").strip(),
        "kernel_files": kernel,
        "harness_files": harness,
        "kernel_blobs": blob_ids(root, kernel),
        "harness_blobs": blob_ids(root, harness),
        "untracked": loose,
        "uncommitted": changed or bool(loose),
    }


def status_path(state_dir: Path, job_id: str) -> Path:
    return Path(state_dir) / f"status-{job_id}.json"


def write_status(root: Path, state_dir: Path, job_id: str) -> dict:
    status = build_status(root, job_id)
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path(state_dir, job_id).write_text(json.dumps(status, indent=2) + "\n")
    return status


def read_status(state_dir: Path, job_id: str) -> dict:
    path = status_path(state_dir, job_id)
    if not path.is_file():
        raise RuntimeError(f"no manifest for job {job_id} in {state_dir}. Run the job through job.sh.")
    status = json.loads(path.read_text())
    if status["job_id"] != str(job_id):
        raise RuntimeError(f"{path} was written by job {status['job_id']}, not by this job ({job_id}). "
                           "Run the job through job.sh.")
    return status


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combined_sha256(root: Path, paths: list[str]) -> str:
    lines = "".join(f"{p}\0{sha256_file(Path(root) / p)}\n" for p in sorted(paths))
    return hashlib.sha256(lines.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--state-dir", type=Path, required=True)
    write.add_argument("--job-id", required=True)
    write.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    status = write_status(args.root, args.state_dir, args.job_id)
    print(f"manifest: {len(status['kernel_files'])} kernel files, "
          f"{'uncommitted' if status['uncommitted'] else 'committed'} ({status['head'][:12]})")
    if status["uncommitted"]:
        print("manifest: submission/ has changes that are not committed. This run will not count on the board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
