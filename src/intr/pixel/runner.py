"""The supervisor: preflight, pilot ladder, then stages. Standard library only; every torch use is in a worker process.

    python -m intr.pixel.runner [--smoke]      normally started by scripts/gpu_run.sh

Stages, so the main verdicts cannot wait on or be damaged by the rest:
    main    H1p, H12p, H4p, H3p
    second  H5p, H14p
    last    H16, H17, both descriptive

A summary and a tar are written after every stage. The folder name includes a fingerprint of the source and settings, so a rerun of the same
code resumes, and changed code starts a new folder instead of mixing results.

--repair-variant idm|state runs the pilot stage alone, at rung 0, on the pilot seed, with that encoder objective, and then stops. No registered
seed starts, in the tiny preflight job either. --action-repeat 3 holds each action for 3 simulator steps, and is refused without a repair variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import ACTION_REPEATS, EMBED_DIMS, REPAIR_VARIANTS, ThreadPConfig, config_for, estimate_job_seconds
from .publish import TOKEN_VAR, make_tar, publish_results, redact, scrub_origin
from .scheduler import Job, RunReport, Slot, atomic_write_json, choose_concurrency, discover_gpus, is_complete, plan_slots, run_jobs, worker_env

ROOT = Path(__file__).resolve().parents[3]
EXIT_PREFLIGHT = 2


def say(text: str = "") -> None:
    print(redact(str(text)), flush=True)


def source_hashes(root: Path) -> dict:
    files = sorted((root / "src" / "intr").glob("*.py")) + sorted((root / "src" / "intr" / "pixel").glob("*.py"))
    files += [root / "experiments" / "thread_p.py", root / "scripts" / "gpu_run.sh"]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def fingerprint_of(hashes: dict, cfg: ThreadPConfig) -> str:
    text = json.dumps({"source": hashes, "config": cfg.fingerprint_view()}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Paths:
    def __init__(self, base: Path, fingerprint: str):
        self.base = base
        self.run = base / fingerprint
        self.jobs = self.run / "jobs"
        self.pilot = self.run / "pilot"
        self.logs = self.run / "logs"
        self.state = self.run / "state.json"
        self.tar = base / f"thread_p_{fingerprint}.tar.gz"


def worker(mode: str, kind: str, *extra: str) -> list[str]:
    return [sys.executable, str(ROOT / "experiments" / "thread_p.py"), "--kind", kind, "--mode", mode, *extra]


def variant_args(cfg: ThreadPConfig) -> list[str]:
    return ["--repair-variant", cfg.repair_variant, "--action-repeat", str(cfg.action_repeat)] if cfg.repair_variant else []


def job_for(paths: Paths, mode: str, kind: str, family: str, seed: int, fingerprint: str, rung: int = 0) -> Job:
    name = f"{kind}_{family}_s{seed}"
    return Job(
        name,
        worker(mode, kind, "--family", family, "--seed", str(seed), "--rung", str(rung), "--run-dir", str(paths.run), "--out", str(paths.jobs / f"{name}.json"), "--fingerprint", fingerprint),
        paths.jobs / f"{name}.json",
        paths.logs / f"{name}.log",
        fingerprint,
    )


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def machine_facts(root: Path, gpus: list[dict], cores: int, hashes: dict, cfg: ThreadPConfig, fingerprint: str) -> dict:
    def text(args: list[str]) -> str:
        try:
            return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=60, env=worker_env(Slot(None, 1))).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    versions = text([sys.executable, "-c", "import torch, numpy; print(torch.__version__, torch.version.cuda, numpy.__version__)"]).split()
    return {
        "fingerprint": fingerprint,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": text(["git", "rev-parse", "HEAD"]),
        "git_dirty_paths": text(["git", "status", "--porcelain", "--", "src", "experiments", "scripts"]),
        "gpus": gpus,
        "cpu_cores": cores,
        "python": platform.python_version(),
        "torch": versions[0] if versions else None,
        "torch_cuda": versions[1] if len(versions) > 1 else None,
        "numpy": versions[2] if len(versions) > 2 else None,
        "platform": platform.platform(),
        "source_sha256": hashes,
        "config": asdict(cfg),
    }


def run_tests(root: Path, paths: Paths) -> bool:
    """Fast pixel tests, everything but the end-to-end smoke. Returns False on failure."""
    say("Preflight: fast tests")
    log = paths.logs / "preflight_tests.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    tests = [str(p) for p in sorted((root / "tests").glob("test_pixel_*.py"))]
    with open(log, "wb") as handle:
        code = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", "not smoke", *tests], cwd=root, stdout=handle, stderr=subprocess.STDOUT, env=worker_env(Slot(None, 2))).returncode
    if code != 0:
        say(f"Fast tests FAILED. Do not spend on this run. Log: {log}")
        say(log.read_text(encoding="utf-8", errors="replace")[-3000:])
        return False
    say("Fast tests passed.")
    return True


def preflight_each_gpu(paths: Paths, mode: str, cfg: ThreadPConfig, slots_one_each: list[Slot], fingerprint: str) -> bool:
    """Structural checks and one tiny end-to-end job, per GPU."""
    root = paths.run / "preflight"
    families = ["--families", *cfg.families]
    checks = [
        Job(f"preflight_{i}", worker(mode, "preflight", *families, *variant_args(cfg), "--out", str(root / f"checks_{i}.json"), "--fingerprint", fingerprint), root / f"checks_{i}.json", paths.logs / f"preflight_{i}.log", fingerprint)
        for i in range(len(slots_one_each))
    ]
    report = run_jobs(checks, slots_one_each, "preflight", max_attempts=1, progress=say)
    for name, failure in report.failed.items():
        say(f"PREFLIGHT FAILED on {name}. Do not spend on this run.\n{failure['tail']}")
    if report.failed:
        return False
    for job in checks:
        data = read_json(job.output, {})
        say(f"  {job.name}: {data.get('checks')}")

    def tiny_argv(i: int) -> list[str]:
        if cfg.repair_variant:  # validation of the pilot seed only: a main job would touch seed 10 and create test tiers
            return worker("smoke", "pilot", "--family", cfg.families[0], "--seed", str(cfg.pilot_seed), *variant_args(cfg), "--out", str(root / f"tiny_{i}.json"), "--fingerprint", fingerprint)
        return worker("smoke", "main", "--family", cfg.families[0], "--seed", "10", "--run-dir", str(root / f"tiny_{i}"), "--out", str(root / f"tiny_{i}.json"), "--fingerprint", fingerprint)

    tiny = [Job(f"tiny_{i}", tiny_argv(i), root / f"tiny_{i}.json", paths.logs / f"tiny_{i}.log", fingerprint) for i in range(len(slots_one_each))]
    report = run_jobs(tiny, slots_one_each, "tiny end-to-end", max_attempts=1, progress=say)
    for name, failure in report.failed.items():
        say(f"PREFLIGHT FAILED: the tiny job {name} crashed. Do not spend on this run.\n{failure['tail']}")
    return not report.failed


def measure_concurrency(paths: Paths, mode: str, cfg: ThreadPConfig, gpu: dict, cores: int, fingerprint: str) -> tuple[int, dict]:
    """Run 1, 2, 3, 4 copies of a JEPA benchmark on one GPU together, and keep adding while total throughput rises."""
    root = paths.run / "bench"
    rates: dict[int, float] = {}
    memory_ok: dict[int, bool] = {}
    first: dict = {}
    biggest = max(r.size for r in cfg.rungs)
    frames_mb = cfg.train_envs * cfg.train_traj * (cfg.train_steps + cfg.stack) * 3 * biggest * biggest / 1e6
    for n in (1, 2, 3, 4):
        if cores // max(n, 1) < 1:
            break
        jobs = [
            Job(
                f"bench_{n}_{i}",
                worker(mode, "bench", "--seconds", "15", *([] if (n == 1) else ["--jepa-only"]), "--out", str(root / f"bench_{n}_{i}.json"), "--fingerprint", fingerprint),
                root / f"bench_{n}_{i}.json",
                paths.logs / f"bench_{n}_{i}.log",
                fingerprint,
            )
            for i in range(n)
        ]
        threads = max(min(cores // (n * 2), 2), 1)
        report = run_jobs(jobs, [Slot(gpu["index"], threads) for _ in range(n)], f"bench x{n}", max_attempts=1, progress=say)
        results = [read_json(j.output, {}) for j in jobs]
        if report.failed or any("jepa_steps_per_s" not in r for r in results):
            break
        rates[n] = sum(r["jepa_steps_per_s"] for r in results)
        peak = max(r.get("peak_memory_mb", 0.0) for r in results)
        memory_ok[n] = n * (peak + frames_mb) <= 0.85 * gpu["memory_mb"]
        if n == 1:
            first = results[0]
        elif rates[n] < rates[n - 1] * 1.15:
            break
    if not rates:
        return 1, {}
    chosen = choose_concurrency(rates, memory_ok)
    bench = {**first, "jepa_share": rates[chosen] / rates[min(rates)], "rates": rates, "chosen": chosen}
    return chosen, bench


def print_estimate(cfg: ThreadPConfig, bench: dict, jobs_per_gpu: int, n_slots: int) -> None:
    if "core_steps_per_s" not in bench:
        return
    totals = {"main": 0.0, "second": 0.0, "last": 0.0}
    for family in cfg.families:
        est = estimate_job_seconds(cfg, family, bench, jobs_per_gpu)
        totals["main"] += est["main"] * len(cfg.seeds)
        totals["second"] += (est["h5"] + est["h14"]) * len(cfg.seeds)
        totals["last"] += est["h16"] * len(cfg.seeds) + est["h17"] * len(cfg.h17_seeds)
    wall = {k: v / n_slots for k, v in totals.items()}
    say(f"Estimated wall time on {n_slots} slots: main {wall['main'] / 60:.0f} min, H5p and H14p {wall['second'] / 60:.0f} min, H16 and H17 {wall['last'] / 60:.0f} min, about {sum(wall.values()) / 3600:.1f} h in total, plus the pilot. Rough, from short benchmarks; ETA below comes from finished jobs.")


def run_pilot_ladder(paths: Paths, mode: str, cfg: ThreadPConfig, slots: list[Slot], fingerprint: str) -> dict:
    """One seed per family, smallest passing rung on validation. Thresholds never change. An oracle failure stops that family's ladder."""
    choice_path = paths.pilot / "choice.json"
    saved = read_json(choice_path)
    if saved is not None and saved.get("fingerprint") == fingerprint and set(cfg.families) <= set(saved["rungs"]) | set(saved["skipped"]):
        say(f"Pilot already done: {saved}")
        return saved
    rungs: dict[str, int] = {}
    skipped: dict[str, str] = {}
    pending = {family: 0 for family in cfg.families}
    results: dict = {}
    while pending:
        jobs = {}
        for family, rung in pending.items():
            name = f"pilot_{family}_r{rung}"
            out = paths.pilot / f"{name}.json"
            jobs[family] = Job(name, worker(mode, "pilot", "--family", family, "--seed", str(cfg.pilot_seed), "--rung", str(rung), *variant_args(cfg), "--out", str(out), "--fingerprint", fingerprint), out, paths.logs / f"{name}.log", fingerprint)
        report = run_jobs(list(jobs.values()), slots, "pilot", progress=say)
        next_pending = {}
        for family, job in jobs.items():
            rung = pending[family]
            data = read_json(job.output, {}) if job.name not in report.failed else {}
            gate = data.get("gate", {})
            results[job.name] = gate
            say(f"  pilot {family} rung {rung} ({cfg.rungs[rung]}): {gate.get('checks', 'job failed')}, wall {data.get('wall_s', 0.0) / 60:.1f} min")
            if gate.get("pass") or (not cfg.enforce_gates and data):
                rungs[family] = rung
            elif data and not gate["checks"].get("oracle_interp", True) and cfg.enforce_gates:
                skipped[family] = "the oracle fails on validation, which no rung can change"
            elif rung + 1 < len(cfg.rungs):
                next_pending[family] = rung + 1
            else:
                skipped[family] = "no rung of the ladder passed Gate GP on validation"
        pending = next_pending
    for family, reason in skipped.items():
        say(f"FAMILY {family} cannot pass Gate GP: {reason}. It is not evidence for or against any method. Continuing with the others.")
    choice = {"fingerprint": fingerprint, "status": "ok", "rungs": rungs, "skipped": skipped, "gates": results, "pilot_seed": cfg.pilot_seed}
    atomic_write_json(choice_path, choice)
    return choice


def finish_repair(paths: Paths, cfg: ThreadPConfig) -> int:
    """The end of a repair run, pass or fail on the gate. A pilot job that crashed is an error, not a gate result."""
    crashed = [f for f in cfg.families if read_json(paths.pilot / f"pilot_{f}_r0.json", {}).get("status") != "ok"]
    if crashed:
        say(f"Repair variant {cfg.repair_variant}: the pilot job crashed for {crashed}. See {paths.logs}")
        return 1
    say(f"Repair variant {cfg.repair_variant}: stopping after rung 0 of the pilot on seed {cfg.pilot_seed}. No registered seed was started.")
    return 0


def summarize(paths: Paths, mode: str, cfg: ThreadPConfig, fingerprint: str) -> None:
    """In a process of its own, so a bug here cannot touch job files. Then always tar."""
    log = paths.logs / "summary.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as handle:
        code = subprocess.run(worker(mode, "summarize", "--run-dir", str(paths.run), "--fingerprint", fingerprint, "--families", *cfg.families), stdout=handle, stderr=subprocess.STDOUT, env=worker_env(Slot(None, 2))).returncode
    if code != 0:
        say(f"Summary failed with code {code}. Job files are intact. Log: {log}")
    elif (paths.run / "summary.txt").exists():
        say((paths.run / "summary.txt").read_text(encoding="utf-8"))
    make_tar(paths.run, paths.tar)
    say(f"Results folder: {paths.run}")
    say(f"Tar: {paths.tar}")


def update_state(paths: Paths, report: RunReport, skipped: list[str]) -> None:
    state = read_json(paths.state, {"failed_jobs": {}, "skipped_families": []})
    for name in report.finished + report.skipped:
        state["failed_jobs"].pop(name, None)
    for name, failure in report.failed.items():
        state["failed_jobs"][name] = {"code": failure["code"], "log": failure["log"]}
    state["skipped_families"] = skipped
    atomic_write_json(paths.state, state)


def run_stage(name: str, jobs: list[Job], slots: list[Slot], paths: Paths, skipped: list[str], prior_s: float | None) -> RunReport:
    say(f"\n=== Stage {name}: {len(jobs)} jobs ===")
    report = run_jobs(jobs, slots, name, progress=say, prior_job_s=prior_s) if jobs else RunReport()
    update_state(paths, report, skipped)
    if report.retried:
        say(f"Retried once: {report.retried}")
    for job, failure in report.failed.items():
        say(f"Failed after one retry: {job}. Log: {failure['log']}\n{failure['tail']}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny settings on any machine, minutes on a CPU; numbers mean nothing")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_p_gpu")
    parser.add_argument("--families", nargs="+", default=None, choices=sorted(EMBED_DIMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="run the full settings without a GPU, which would take days")
    parser.add_argument("--jobs-per-gpu", type=int, default=None, help="skip the measurement")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--repair-variant", default=None, choices=sorted(REPAIR_VARIANTS), help="pilot stage only, rung 0, pilot seed, with this encoder objective")
    parser.add_argument("--action-repeat", type=int, default=1, choices=ACTION_REPEATS, help="simulator steps per action, with --repair-variant only")
    args = parser.parse_args(argv)
    if args.repair_variant and args.seeds:
        parser.error("--repair-variant runs the pilot seed only, so it takes no --seeds")
    if args.action_repeat != 1 and not args.repair_variant:
        parser.error("--action-repeat above 1 runs only with --repair-variant: at that repeat the later stages have not been scaled")

    mode = "smoke" if args.smoke else "full"
    cfg = config_for(mode, args.repair_variant, args.action_repeat)
    if args.families:
        cfg.families = args.families
    if args.seeds:
        cfg.seeds = args.seeds
    cfg.h17_seeds = [s for s in cfg.h17_seeds if s in cfg.seeds]

    clean = scrub_origin(ROOT)
    if clean:
        say(f"Origin: {clean}")
    hashes = source_hashes(ROOT)
    fingerprint = fingerprint_of(hashes, cfg)
    paths = Paths(args.out, fingerprint)
    for folder in (paths.jobs, paths.pilot, paths.logs):
        folder.mkdir(parents=True, exist_ok=True)

    gpus = discover_gpus()
    cores = os.cpu_count() or 1
    say(f"Run {fingerprint} in {paths.run}. GPUs: {[g['name'] for g in gpus] or 'none'}. CPU cores: {cores}. Mode: {mode}.")
    if not gpus and not args.smoke and not args.allow_cpu:
        say("No GPU found. Stopping before spending anything. Check nvidia-smi, or pass --smoke to test on a CPU.")
        return EXIT_PREFLIGHT
    atomic_write_json(paths.run / "manifest.json", machine_facts(ROOT, gpus, cores, hashes, cfg, fingerprint))

    # The test suite names seeds 10 and 11 in its tiny checks. A repair run leaves it to pixel-check.
    if not args.skip_tests and not cfg.repair_variant and not run_tests(ROOT, paths):
        return EXIT_PREFLIGHT
    one_each = [Slot(g["index"], 2) for g in gpus] or [Slot(None, min(2, cores))]
    if not preflight_each_gpu(paths, mode, cfg, one_each, fingerprint):
        return EXIT_PREFLIGHT

    bench: dict = {}
    if gpus and args.jobs_per_gpu is None and not cfg.repair_variant:
        jobs_per_gpu, bench = measure_concurrency(paths, mode, cfg, gpus[0], cores, fingerprint)
        say(f"Jobs per GPU: {jobs_per_gpu}. Measured: {bench}")
    else:
        jobs_per_gpu = args.jobs_per_gpu or 1
    slots = plan_slots(gpus, cores, jobs_per_gpu)
    say(f"Slots: {len(slots)}, {slots[0].threads} CPU threads each.")
    print_estimate(cfg, bench, jobs_per_gpu, len(slots))

    choice = run_pilot_ladder(paths, mode, cfg, slots, fingerprint)
    if cfg.repair_variant:
        return finish_repair(paths, cfg)
    families = [f for f in cfg.families if f in choice["rungs"]]
    skipped = sorted(choice["skipped"])
    pilot_walls = [read_json(p, {}).get("wall_s") for p in paths.pilot.glob("pilot_*.json")]
    pilot_walls = [w for w in pilot_walls if w]
    prior = 1.5 * sum(pilot_walls) / len(pilot_walls) if pilot_walls else None
    if prior:
        say(f"Pilot basis: a main job should take about {prior / 60:.0f} min, so about {prior * len(families) * len(cfg.seeds) / len(slots) / 60:.0f} min for the main stage.")

    make = lambda kind, family, seed: job_for(paths, mode, kind, family, seed, fingerprint, choice["rungs"].get(family, 0))
    main_jobs = [make("main", f, s) for s in cfg.seeds for f in families]
    run_stage("main", main_jobs, slots, paths, skipped, prior)
    summarize(paths, mode, cfg, fingerprint)

    finished = {(f, s) for s in cfg.seeds for f in families if is_complete(make("main", f, s))}
    second = [make(kind, f, s) for kind in ("h5", "h14") for s in cfg.seeds for f in families if (f, s) in finished]
    try:
        # These train on the CPU and use little GPU memory, so they can run wider than the main stage where cores allow.
        wide = plan_slots(gpus, cores, min(6, max(jobs_per_gpu, cores // (max(len(gpus), 1) * 2)))) if gpus else slots
        run_stage("second (H5p, H14p)", second, wide, paths, skipped, None)
    finally:
        summarize(paths, mode, cfg, fingerprint)
    last = [make("h16", f, s) for s in cfg.seeds for f in families if (f, s) in finished]
    last += [make("h17", f, s) for s in cfg.h17_seeds for f in families if (f, s) in finished]
    try:
        run_stage("last (H16, H17)", last, slots, paths, skipped, None)
    finally:
        summarize(paths, mode, cfg, fingerprint)

    if args.no_publish:
        say(f"Publishing skipped. Download the tar from Jupyter: {paths.tar}")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = publish_results(paths.run, ROOT, stamp)
    if result["pushed"]:
        say(f"Results pushed to branch {result['branch']}. Tar also at: {paths.tar}")
    else:
        say(f"Not pushed: {result['reason']}. Download the tar from Jupyter: {paths.tar}")
    say("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
