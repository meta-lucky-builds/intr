"""Runs one process per job, each pinned to a slot. Standard library only.

A job is finished when its output file exists, parses, says status ok, and carries the expected fingerprint. Workers write that file
atomically, so a crash leaves either the old file or nothing. A rerun skips finished jobs. A failed job gets one retry, on any free slot,
and a final failure is reported without stopping the others. No secret reaches a worker: variables that look like credentials are
removed from its environment, so nothing a worker prints or stores can contain them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|CREDENTIAL", re.IGNORECASE)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass
class Job:
    name: str
    argv: list[str]
    output: Path
    log: Path
    fingerprint: str | None = None


@dataclass(frozen=True)
class Slot:
    gpu: str | None  # None runs on the CPU
    threads: int = 2


@dataclass
class RunReport:
    finished: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, dict] = field(default_factory=dict)
    retried: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)


def atomic_write_json(path: Path, obj) -> None:
    """Write beside the target, flush to disk, then rename. Readers see the old file or the new one, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def is_complete(job: Job) -> bool:
    try:
        data = json.loads(Path(job.output).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("status") != "ok":
        return False
    return job.fingerprint is None or data.get("fingerprint") == job.fingerprint


def worker_env(slot: Slot, base: dict | None = None) -> dict:
    env = {k: v for k, v in (os.environ if base is None else base).items() if not _SECRET.search(k)}
    env["CUDA_VISIBLE_DEVICES"] = "" if slot.gpu is None else str(slot.gpu)
    for name in THREAD_VARS:
        env[name] = str(slot.threads)
    env["INTR_THREADS"] = str(slot.threads)
    return env


def log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def format_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f} min"


def run_jobs(jobs: list[Job], slots: list[Slot], label: str = "jobs", poll_s: float = 1.0, max_attempts: int = 2, progress=print, progress_every_s: float = 60.0, prior_job_s: float | None = None) -> RunReport:
    """Blocks until every job finished or failed. progress receives plain lines."""
    if not slots:
        raise ValueError("no slots to run on")
    report = RunReport()
    pending: deque = deque()
    for job in jobs:
        if is_complete(job):
            report.skipped.append(job.name)
        else:
            pending.append((job, 1))
    total = len(pending)
    free = list(range(len(slots)))
    running: dict[int, tuple] = {}
    started = time.perf_counter()
    last_print = started
    if report.skipped:
        progress(f"[{label}] {len(report.skipped)} already finished, skipped")
    progress(f"[{label}] {total} to run on {len(slots)} slots")

    def line() -> str:
        elapsed = time.perf_counter() - started
        done = len(report.finished) + len(report.failed)
        mean = sum(report.durations.values()) / len(report.durations) if report.durations else prior_job_s
        eta = "unknown" if mean is None else format_minutes(mean * (total - done) / len(slots))
        return f"[{label}] {done}/{total} done, {len(report.failed)} failed, {len(running)} running, elapsed {format_minutes(elapsed)}, ETA {eta}"

    try:
        while pending or running:
            while pending and free:
                job, attempt = pending.popleft()
                index = free.pop(0)
                Path(job.log).parent.mkdir(parents=True, exist_ok=True)
                with open(job.log, "ab") as log:
                    process = subprocess.Popen(job.argv, env=worker_env(slots[index]), stdout=log, stderr=subprocess.STDOUT)
                running[index] = (job, process, attempt, time.perf_counter())
            for index, (job, process, attempt, began) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                del running[index]
                free.append(index)
                if code == 0 and is_complete(job):
                    report.finished.append(job.name)
                    report.durations[job.name] = time.perf_counter() - began
                elif attempt < max_attempts:
                    report.retried.append(job.name)
                    progress(f"[{label}] {job.name} failed with code {code}, retrying once. Log: {job.log}")
                    pending.append((job, attempt + 1))
                else:
                    report.failed[job.name] = {"code": code, "log": str(job.log), "tail": log_tail(job.log)}
                    progress(f"[{label}] {job.name} FAILED twice with code {code}. Others continue. Log: {job.log}")
            if time.perf_counter() - last_print >= progress_every_s:
                progress(line())
                last_print = time.perf_counter()
            if pending or running:
                time.sleep(poll_s)
    finally:
        for _, process, _, _ in running.values():
            process.terminate()
    progress(line())
    return report


# ---------------------------------------------------------------- machine facts


def parse_gpu_query(text: str) -> list[dict]:
    """Output of nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits."""
    gpus = []
    for row in text.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                gpus.append({"index": parts[0], "name": parts[1], "memory_mb": int(float(parts[2]))})
            except ValueError:
                continue
    return gpus


def discover_gpus() -> list[dict]:
    """GPUs nvidia-smi reports, limited to a CUDA_VISIBLE_DEVICES of plain indices. None found means an empty list."""
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = parse_gpu_query(text)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and all(p.strip().isdigit() for p in visible.split(",") if p.strip()):
        keep = {p.strip() for p in visible.split(",") if p.strip()}
        gpus = [g for g in gpus if g["index"] in keep]
    return gpus


def plan_slots(gpus: list[dict], cores: int, jobs_per_gpu: int, max_threads: int = 4) -> list[Slot]:
    """How many jobs and how many threads each, without oversubscribing cores. No GPU: CPU jobs of 2 threads each."""
    cores = max(int(cores), 1)
    if not gpus:
        threads = min(2, cores)
        return [Slot(None, threads) for _ in range(max(cores // threads, 1))]
    per_gpu = max(min(jobs_per_gpu, cores // len(gpus)), 1)
    threads = max(min(cores // (per_gpu * len(gpus)), max_threads), 1)
    return [Slot(g["index"], threads) for _ in range(per_gpu) for g in gpus]


def choose_concurrency(rates: dict[int, float], memory_ok: dict[int, bool], min_gain: float = 0.15) -> int:
    """rates: {jobs on one GPU: total steps per second}. Add a job only while total throughput rises by at least min_gain."""
    best = min(rates)
    for n in sorted(rates):
        if n == best or not memory_ok.get(n, True):
            continue
        if rates[n] >= rates[best] * (1.0 + min_gain):
            best = n
        else:
            break
    return best
