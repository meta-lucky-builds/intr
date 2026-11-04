"""Helpers for .github/workflows/experiments.yml. Standard library only, so the setup job needs no torch.

    request   validate .ci/run-request.json and write the matrix to GITHUB_OUTPUT
    job-info  record one experiment job's outcome, machine and minutes next to its results
    minutes   print a table of minutes per job from the GitHub jobs API response
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import merge_results

FAMILIES = ("pendulum", "puck", "drone")  # the same as intr.envs.FAMILIES, which this file cannot import without torch
REQUEST_KEYS = {"thread", "seeds", "families", "note"}
MAX_SEED = 999
MAX_JOBS = 40
MAX_NOTE = 200


class RequestError(ValueError):
    """The run request cannot be accepted."""


def validate_request(raw) -> dict:
    """Example: {"thread": "a", "seeds": [0], "families": ["pendulum", "puck"], "note": "pilot"}."""
    if not isinstance(raw, dict):
        raise RequestError("the request must be a JSON object")
    unknown = sorted(set(raw) - REQUEST_KEYS)
    if unknown:
        raise RequestError(f"unknown keys: {unknown}")
    thread = raw.get("thread")
    if thread not in merge_results.THREADS:
        raise RequestError(f"thread must be one of {merge_results.THREADS}, got {thread!r}")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise RequestError("seeds must be a non-empty list")
    for seed in seeds:
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise RequestError(f"each seed must be an integer from 0 to {MAX_SEED}, got {seed!r}")
    if len(set(seeds)) != len(seeds):
        raise RequestError(f"seeds has duplicates: {seeds}")
    families = raw.get("families")
    if not isinstance(families, list) or not families:
        raise RequestError("families must be a non-empty list")
    for family in families:
        if family not in FAMILIES:
            raise RequestError(f"each family must be one of {FAMILIES}, got {family!r}")
    if len(set(families)) != len(families):
        raise RequestError(f"families has duplicates: {families}")
    if len(seeds) * len(families) > MAX_JOBS:
        raise RequestError(f"{len(seeds) * len(families)} jobs is more than the limit of {MAX_JOBS}")
    note = raw.get("note", "")
    if not isinstance(note, str) or len(note) > MAX_NOTE or not note.isprintable():
        raise RequestError(f"note must be one printable line of at most {MAX_NOTE} characters")
    return {"thread": thread, "seeds": seeds, "families": families, "note": note}


def src_fingerprint(root: Path) -> dict:
    """The src/ part of thread_a.source_fingerprint, with the same keys."""
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src" / "intr").glob("*.py"))
    }


def preflight(request: dict, root: Path) -> None:
    """Refuse now what the merge would refuse after hours of compute. Config is checked at merge, as it needs torch."""
    path = merge_results.result_path(root, request["thread"])
    if not path.exists():
        return
    prior = json.loads(path.read_bytes())
    present = set(merge_results.job_pairs(prior))
    overlap = sorted((s, f) for s in request["seeds"] for f in request["families"] if (f, s) in present)
    if overlap:
        raise RequestError(f"{path.name} already holds these (seed, family) jobs: {overlap}")
    changed = merge_results.src_hash_differences(prior["metadata"]["source_sha256"], src_fingerprint(root))
    if changed:
        raise RequestError(f"src/ differs from the code behind {path.name}, so results could not be merged: {changed}")


def matrix_outputs(request: dict) -> dict:
    return {
        "thread": request["thread"],
        "seeds": json.dumps(request["seeds"], separators=(",", ":")),
        "families": json.dumps(request["families"], separators=(",", ":")),
        "seeds_args": " ".join(str(s) for s in request["seeds"]),
        "families_args": " ".join(request["families"]),
        "note": request["note"],
    }


def cpu_info() -> dict:
    """Which machine ran a job, since instruction sets are one source of numeric differences."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"cores": os.cpu_count()}
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        fields.setdefault(key.strip(), value.strip())
    flags = set(fields.get("flags", "").split())
    return {
        "model": fields.get("model name"),
        "cores": os.cpu_count(),
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
    }


def job_info(thread: str, seed: int, family: str, outcome: str, started_epoch: float, ended_epoch: float) -> dict:
    return {
        "thread": thread,
        "seed": seed,
        "family": family,
        "outcome": outcome,
        "minutes": round((ended_epoch - started_epoch) / 60.0, 2),
        "cpu": cpu_info(),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "sha": os.environ.get("GITHUB_SHA"),
        "runner": os.environ.get("RUNNER_NAME"),
    }


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def minutes_table(payload: dict, now: datetime | None = None) -> str:
    """Billed minutes are per job, rounded up. A job still running is counted up to now."""
    now = now or datetime.now(timezone.utc)
    rows, total, billed_total = [], 0.0, 0
    for job in payload.get("jobs", []):
        minutes = 0.0
        if job.get("started_at"):
            ended = _parse_time(job["completed_at"]) if job.get("completed_at") else now
            minutes = max((ended - _parse_time(job["started_at"])).total_seconds(), 0.0) / 60.0
        billed = math.ceil(minutes)
        total += minutes
        billed_total += billed
        rows.append(f"| {job.get('name')} | {job.get('conclusion') or job.get('status')} | {minutes:.1f} | {billed} |")
    head = ["### Minutes per job", "", "| Job | Conclusion | Minutes | Rounded up |", "| --- | --- | ---: | ---: |"]
    return "\n".join(head + rows + [f"| **Total** | | **{total:.1f}** | **{billed_total}** |"]) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    request = commands.add_parser("request")
    request.add_argument("--file", type=Path, required=True)
    request.add_argument("--root", type=Path, default=Path("."))
    request.add_argument("--github-output", type=Path, default=None)

    info = commands.add_parser("job-info")
    info.add_argument("--out", type=Path, required=True)
    info.add_argument("--thread", required=True)
    info.add_argument("--seed", type=int, required=True)
    info.add_argument("--family", required=True)
    info.add_argument("--outcome", required=True)
    info.add_argument("--started-epoch", type=float, required=True)

    minutes = commands.add_parser("minutes")
    minutes.add_argument("--jobs-json", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "request":
        try:
            if not args.file.exists():
                raise RequestError(f"{args.file} does not exist")
            try:
                raw = json.loads(args.file.read_bytes())
            except ValueError as err:
                raise RequestError(f"{args.file} is not valid JSON: {err}") from err
            checked = validate_request(raw)
            preflight(checked, args.root)
        except RequestError as err:
            print(f"::error::run request refused: {err}")
            return 1
        outputs = matrix_outputs(checked)
        text = "".join(f"{key}={value}\n" for key, value in outputs.items())
        print(text, end="")
        if args.github_output is not None:
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write(text)
    elif args.command == "job-info":
        record = job_info(args.thread, args.seed, args.family, args.outcome, args.started_epoch, time.time())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        line = f"seed {args.seed} on {args.family}: {args.outcome}, {record['minutes']} minutes, {record['cpu'].get('model')}\n"
        print(line, end="")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(line)
    else:
        print(minutes_table(json.loads(args.jobs_json.read_bytes())), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
