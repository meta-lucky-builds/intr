"""One runner stage per GitHub Actions job, for .github/workflows/pixel-pendulum.yml. The GPU path does not use this file.

Nothing here decides what runs. The job list comes from the runner's own main, called with its effects replaced, so the list cannot drift from
it. This file is outside the source the fingerprint hashes, so adding it leaves the fingerprint as it was. Standard library only, except time-h14.

    context     print the fingerprint and run folder, fail if the fingerprint is not the expected one
    pilot       the runner's main up to and including the pilot ladder, preflight included
    plan        the matrices, from the pilot's choice
    status      has this job finished in an earlier attempt? Fail on a foreign record, or if its main job did not finish
    run         one job, through the runner's run_stage, on all of this machine's cores
    summarize   merge state, then the runner's normal summarize. Fail afterwards on a foreign record
    time-h14    seconds per error-fed update at full settings on random data. Touches no seed's data
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intr.pixel import runner  # noqa: E402
from intr.pixel.config import config_for  # noqa: E402
from intr.pixel.scheduler import Job, RunReport, Slot, atomic_write_json, is_complete  # noqa: E402

MUST_MATCH = ("fingerprint", "git_commit", "python", "torch", "numpy", "source_sha256")


def emit(key: str, value: str) -> None:
    """A step output on Actions, a plain line anywhere."""
    print(f"{key}={value}", flush=True)
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"{key}={value}\n")


def complain(problems: list[str]) -> int:
    for problem in problems:
        print(f"::error::{problem}", flush=True)
    return 1 if problems else 0


class Context:
    def __init__(self, mode: str, family: str, out: Path):
        self.mode, self.family, self.out = mode, family, Path(out)
        self.cfg = config_for(mode)
        self.cfg.families = [family]
        self.hashes = runner.source_hashes(ROOT)
        self.fingerprint = runner.fingerprint_of(self.hashes, self.cfg)
        self.paths = runner.Paths(self.out, self.fingerprint)

    @property
    def choice(self) -> dict | None:
        return runner.read_json(self.paths.pilot / "choice.json")


def drive_runner(ctx: Context, **stubs) -> tuple[int, list[Job]]:
    """The runner's real main, with its stages caught and not run. is_complete says yes, so jobs after main are listed as if main had finished."""
    caught: list[Job] = []

    def catch(name, jobs, slots, paths, skipped, prior):
        caught.extend(jobs)
        return RunReport()

    argv = ["--allow-cpu", "--skip-tests", "--no-publish", "--out", str(ctx.out), "--families", ctx.family, *(["--smoke"] if ctx.mode == "smoke" else [])]
    effects = {"run_stage": catch, "summarize": lambda *a: None, "is_complete": lambda job: True, **stubs}
    with mock.patch.multiple(runner, **effects), contextlib.redirect_stdout(sys.stderr if stubs.get("machine_facts") else sys.stdout):
        code = runner.main(argv)
    return code, caught


def listed_jobs(ctx: Context) -> list[Job]:
    """Exactly the jobs the runner would run after this pilot choice. Nothing is run or written."""
    choice = ctx.choice
    if choice is None:
        raise SystemExit(f"No pilot choice at {ctx.paths.pilot}")
    stubs = {
        "scrub_origin": lambda root: None,
        "discover_gpus": lambda: [],
        "machine_facts": lambda *a: {},
        "atomic_write_json": lambda *a: None,
        "preflight_each_gpu": lambda *a: True,
        "run_pilot_ladder": lambda *a: choice,
    }
    code, jobs = drive_runner(ctx, **stubs)
    if code != 0:
        raise SystemExit(f"The runner's main returned {code} while listing jobs")
    return jobs


def kind_of(job: Job) -> str:
    return job.argv[job.argv.index("--kind") + 1]


def seed_of(job: Job) -> int:
    return int(job.argv[job.argv.index("--seed") + 1])


def main_name(job: Job, family: str) -> str:
    return f"main_{family}_s{seed_of(job)}"


def matrix(jobs: list[Job], family: str) -> tuple[list[dict], list[dict]]:
    main, extras = [], []
    for job in jobs:
        entry = {"name": job.name, "kind": kind_of(job), "seed": seed_of(job)}
        if entry["kind"] == "main":
            main.append(entry)
        else:
            extras.append({**entry, "main": main_name(job, family)})
    return main, extras


def job_named(ctx: Context, name: str) -> Job:
    for job in listed_jobs(ctx):
        if job.name == name:
            return job
    raise SystemExit(f"::error::{name} is not a job the runner would create")


def record_problems(paths, fingerprint: str) -> list[str]:
    """Every record here must carry this run's fingerprint."""
    problems = []
    for path in sorted([*paths.jobs.glob("*.json"), *paths.pilot.glob("*.json")]):
        got = (runner.read_json(path) or {}).get("fingerprint")
        if got != fingerprint:
            problems.append(f"{path.name} has fingerprint {got}, this run is {fingerprint}")
    return problems


def manifest_problems(paths) -> list[str]:
    """Every job against the pilot's manifest: same commit, same source, same package versions, clean tree."""
    base = runner.read_json(paths.run / "manifest.json")
    if base is None:
        return ["the pilot's manifest.json is missing"]
    problems = []
    for path in [paths.run / "manifest.json", *sorted((paths.run / "manifests").glob("*.json"))]:
        facts = runner.read_json(path) or {}
        for key in MUST_MATCH:
            if facts.get(key) != base.get(key):
                problems.append(f"{path.name}: {key} is {facts.get(key)}, the pilot had {base.get(key)}")
        if facts.get("git_dirty_paths"):
            problems.append(f"{path.name}: source differs from the commit: {facts['git_dirty_paths']}")
    return problems


def merge_state(ctx: Context) -> RunReport:
    """state.json as the runner's stages would leave it. A job after main that never existed, because its main failed, is not listed as failed."""
    choice = ctx.choice
    report = RunReport()
    if choice is None:
        report.failed["pilot"] = {"code": None, "log": str(ctx.paths.logs)}
        runner.update_state(ctx.paths, report, [])
        return report
    jobs = listed_jobs(ctx)
    by_name = {job.name: job for job in jobs}
    for job in jobs:
        if is_complete(job):
            report.finished.append(job.name)
        elif kind_of(job) == "main" or is_complete(by_name[main_name(job, ctx.family)]):
            report.failed[job.name] = {"code": None, "log": str(job.log)}
    runner.update_state(ctx.paths, report, sorted(choice["skipped"]))
    return report


# ---------------------------------------------------------------- commands


def cmd_context(ctx: Context, args) -> int:
    emit("fingerprint", ctx.fingerprint)
    emit("run", ctx.paths.run.relative_to(ROOT).as_posix() if ctx.paths.run.is_relative_to(ROOT) else str(ctx.paths.run))
    if args.expect and args.expect != ctx.fingerprint:
        return complain([f"this checkout has fingerprint {ctx.fingerprint}, the pilot had {args.expect}"])
    return 0


def cmd_pilot(ctx: Context, args) -> int:
    cores = os.cpu_count() or 1
    code, _ = drive_runner(ctx, plan_slots=lambda gpus, c, n: [Slot(None, cores)])
    if code != 0:
        return code
    return complain(record_problems(ctx.paths, ctx.fingerprint))


def cmd_plan(ctx: Context, args) -> int:
    main, extras = matrix(listed_jobs(ctx), ctx.family)
    emit("main", json.dumps(main, separators=(",", ":")))
    emit("extras", json.dumps(extras, separators=(",", ":")))
    print(f"{len(main)} main jobs, {len(extras)} jobs after main. Pilot choice: {ctx.choice['rungs']}, skipped {ctx.choice['skipped']}", flush=True)
    return 0


def cmd_status(ctx: Context, args) -> int:
    if ctx.fingerprint != args.expect:
        return complain([f"this checkout has fingerprint {ctx.fingerprint}, the pilot had {args.expect}"])
    problems = record_problems(ctx.paths, ctx.fingerprint)
    if problems:
        return complain(problems)
    job = job_named(ctx, args.name)
    done = is_complete(job)
    if kind_of(job) == "main":
        stem = f"{ctx.family}_s{seed_of(job)}.pt"
        done = done and (ctx.paths.run / "checkpoints" / stem).exists() and (ctx.paths.run / "cache" / stem).exists()
    elif not done and not is_complete(job_named(ctx, main_name(job, ctx.family))):
        return complain([f"{main_name(job, ctx.family)} did not finish, so {job.name} cannot run. Re-run failed jobs after fixing it."])
    print(f"{job.name}: {'finished in an earlier attempt, nothing to repeat' if done else 'to run'}", flush=True)
    emit("done", "true" if done else "false")
    return 0


def cmd_run(ctx: Context, args) -> int:
    job = job_named(ctx, args.name)
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.log.touch()  # so the upload always holds two folders and keeps its layout
    cores = os.cpu_count() or 1
    facts = runner.machine_facts(ROOT, [], cores, ctx.hashes, ctx.cfg, ctx.fingerprint)
    atomic_write_json(ctx.paths.run / "manifests" / f"{job.name}.json", facts)
    problems = manifest_problems(ctx.paths) + record_problems(ctx.paths, ctx.fingerprint)
    if problems:
        return complain(problems)
    print(f"{job.name}: {cores} threads, commit {facts['git_commit']}, torch {facts['torch']}, numpy {facts['numpy']}, python {facts['python']}", flush=True)
    report = runner.run_stage(kind_of(job), [job], [Slot(None, cores)], ctx.paths, [], None)
    return 1 if report.failed else 0


def cmd_summarize(ctx: Context, args) -> int:
    report = merge_state(ctx)
    print(f"{len(report.finished)} jobs finished, not finished: {sorted(report.failed)}", flush=True)
    runner.summarize(ctx.paths, ctx.mode, ctx.cfg, ctx.fingerprint)
    problems = record_problems(ctx.paths, ctx.fingerprint) + manifest_problems(ctx.paths)
    if not (ctx.paths.run / "summary.txt").exists():
        problems.append("summarize wrote no summary.txt")
    return complain(problems)


def cmd_time_h14(ctx: Context, args) -> int:
    """Training is what H14 adds over other jobs, and its cost does not depend on the numbers in the data."""
    import torch

    from intr.envs import make_env
    from intr.looped import SequenceData
    from intr.pixel.config import EMBED_DIMS
    from intr.pixel.error_fed import ErrorFedConfig, ErrorFedTrainConfig, train_error_fed

    cores = os.cpu_count() or 1
    torch.set_num_threads(cores)
    cfg, ef = config_for("full"), config_for("full").error_fed
    embed, act_dim = EMBED_DIMS[ctx.family], make_env(ctx.family, 1, "train", "bounds").act_dim
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(cfg.train_envs, cfg.train_traj, cfg.train_steps + 1, embed, generator=gen)
    act = 2.0 * torch.rand(cfg.train_envs, cfg.train_traj, cfg.train_steps, act_dim, generator=gen) - 1.0
    model = ErrorFedConfig(embed, act_dim, ef.z_dim, ef.emb_dim, ef.hidden, ef.loops, ef.grad_loops, True)
    tcfg = ErrorFedTrainConfig(steps=args.updates, batch_seqs=ef.batch_seqs, filter_max=ef.filter_max, horizon_max=ef.horizon_max)
    wall = train_error_fed(SequenceData(emb, act), model, tcfg, "timing", emb.min(), emb.max()).stats["wall_s"]
    per_update = wall / args.updates
    total = per_update * ef.steps * 3
    print(f"H14 timing on {cores} threads: {per_update:.3f} s per update, so {total / 60:.0f} min to train the three arms at {ef.steps} updates each", flush=True)
    return 0


COMMANDS = {"context": cmd_context, "pilot": cmd_pilot, "plan": cmd_plan, "status": cmd_status, "run": cmd_run, "summarize": cmd_summarize, "time-h14": cmd_time_h14}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--mode", default="full", choices=["full", "smoke"])
    parser.add_argument("--family", default="pendulum")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_p_gpu")
    parser.add_argument("--name", help="a job name, as the runner names it")
    parser.add_argument("--expect", default="", help="the fingerprint the pilot job reported")
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args(argv)
    return COMMANDS[args.command](Context(args.mode, args.family, args.out), args)


if __name__ == "__main__":
    sys.exit(main())
