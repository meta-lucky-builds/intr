"""The Actions helper: the job list must be the runner's, and a foreign record must fail loudly. No worker runs."""

import importlib.util
import json
from pathlib import Path

import pytest

from intr.pixel import runner
from intr.pixel.config import config_for
from intr.pixel.scheduler import atomic_write_json

_spec = importlib.util.spec_from_file_location("pixel_actions", Path(__file__).resolve().parents[1] / "scripts" / "pixel_actions.py")
actions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(actions)


@pytest.fixture(autouse=True)
def no_step_outputs(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)  # this suite runs inside Actions steps


def context(tmp_path, mode="full", rungs=None, skipped=None):
    ctx = actions.Context(mode, "pendulum", tmp_path)
    atomic_write_json(ctx.paths.pilot / "choice.json", {"fingerprint": ctx.fingerprint, "status": "ok", "rungs": {"pendulum": 0} if rungs is None else rungs, "skipped": skipped or {}})
    return ctx


def finish(ctx, name, fingerprint=None):
    atomic_write_json(ctx.paths.jobs / f"{name}.json", {"status": "ok", "fingerprint": fingerprint or ctx.fingerprint})


def test_full_list_is_the_registered_design(tmp_path):
    ctx = context(tmp_path)
    jobs = actions.listed_jobs(ctx)
    names = [j.name for j in jobs]
    assert len(names) == len(set(names)) == 43
    for kind, seeds in (("main", range(10, 20)), ("h5", range(10, 20)), ("h14", range(10, 20)), ("h16", range(10, 20)), ("h17", (10, 11, 12))):
        assert [n for n in names if n.startswith(f"{kind}_")] == [f"{kind}_pendulum_s{s}" for s in seeds]


def test_jobs_are_the_runners_own(tmp_path):
    ctx = context(tmp_path, rungs={"pendulum": 2})
    for job in actions.listed_jobs(ctx):
        kind, seed = actions.kind_of(job), actions.seed_of(job)
        expected = runner.job_for(ctx.paths, "full", kind, "pendulum", seed, ctx.fingerprint, 2)
        assert job.argv == expected.argv and job.output == expected.output and job.log == expected.log


def test_listing_writes_nothing_and_keeps_the_fingerprint(tmp_path):
    ctx = context(tmp_path)
    actions.listed_jobs(ctx)
    assert not (ctx.paths.run / "manifest.json").exists() and not (ctx.paths.run / "state.json").exists()
    cfg = config_for("full")
    cfg.families = ["pendulum"]
    assert ctx.fingerprint == runner.fingerprint_of(runner.source_hashes(actions.ROOT), cfg)
    assert "scripts/pixel_actions.py" not in ctx.hashes


def test_smoke_list(tmp_path):
    names = [j.name for j in actions.listed_jobs(context(tmp_path, "smoke"))]
    assert names == ["main_pendulum_s10", "main_pendulum_s11", "h5_pendulum_s10", "h5_pendulum_s11", "h14_pendulum_s10", "h14_pendulum_s11", "h16_pendulum_s10", "h16_pendulum_s11", "h17_pendulum_s10"]


def test_matrix(tmp_path):
    ctx = context(tmp_path)
    main, extras = actions.matrix(actions.listed_jobs(ctx), "pendulum")
    assert len(main) == 10 and len(extras) == 33
    assert main[0] == {"name": "main_pendulum_s10", "kind": "main", "seed": 10}
    assert {"name": "h17_pendulum_s12", "kind": "h17", "seed": 12, "main": "main_pendulum_s12"} in extras


def test_a_family_that_fails_the_pilot_gets_no_jobs(tmp_path):
    ctx = context(tmp_path, rungs={}, skipped={"pendulum": "no rung passed"})
    assert actions.listed_jobs(ctx) == []
    assert actions.merge_state(ctx).failed == {}
    assert runner.read_json(ctx.paths.state)["skipped_families"] == ["pendulum"]


def test_foreign_fingerprint_is_found(tmp_path):
    ctx = context(tmp_path)
    finish(ctx, "main_pendulum_s10")
    assert actions.record_problems(ctx.paths, ctx.fingerprint) == []
    finish(ctx, "main_pendulum_s11", "somethingelse")
    problems = actions.record_problems(ctx.paths, ctx.fingerprint)
    assert len(problems) == 1 and "main_pendulum_s11" in problems[0]


def test_status_skips_a_finished_job_and_refuses_foreign_records(tmp_path, capsys):
    out = ["--mode", "full", "--family", "pendulum", "--out", str(tmp_path)]
    ctx = context(tmp_path)
    expect = ["--expect", ctx.fingerprint]
    assert actions.main(["status", "--name", "main_pendulum_s10", *expect, *out]) == 0
    assert "done=false" in capsys.readouterr().out
    finish(ctx, "main_pendulum_s10")  # the record alone is not enough: later jobs need the weights too
    assert actions.main(["status", "--name", "main_pendulum_s10", *expect, *out]) == 0
    assert "done=false" in capsys.readouterr().out
    for folder in ("checkpoints", "cache"):
        (ctx.paths.run / folder).mkdir()
        (ctx.paths.run / folder / "pendulum_s10.pt").touch()
    assert actions.main(["status", "--name", "main_pendulum_s10", *expect, *out]) == 0
    assert "done=true" in capsys.readouterr().out
    assert actions.main(["status", "--name", "main_pendulum_s10", "--expect", "other", *out]) == 1
    finish(ctx, "main_pendulum_s11", "somethingelse")
    assert actions.main(["status", "--name", "main_pendulum_s10", *expect, *out]) == 1


def test_a_job_after_main_fails_loudly_without_its_main(tmp_path, capsys):
    out = ["--mode", "full", "--family", "pendulum", "--out", str(tmp_path)]
    ctx = context(tmp_path)
    assert actions.main(["status", "--name", "h5_pendulum_s10", "--expect", ctx.fingerprint, *out]) == 1
    finish(ctx, "main_pendulum_s10")
    assert actions.main(["status", "--name", "h5_pendulum_s10", "--expect", ctx.fingerprint, *out]) == 0
    with pytest.raises(SystemExit):
        actions.job_named(ctx, "h17_pendulum_s13")  # registered on three seeds only


def test_merge_state_matches_what_the_stages_would_leave(tmp_path):
    ctx = context(tmp_path)
    finish(ctx, "main_pendulum_s10")
    finish(ctx, "h5_pendulum_s10")
    report = actions.merge_state(ctx)
    assert sorted(report.finished) == ["h5_pendulum_s10", "main_pendulum_s10"]
    failed = set(report.failed)
    assert "main_pendulum_s11" in failed and "h14_pendulum_s10" in failed
    assert "h5_pendulum_s11" not in failed  # its main failed, so the runner would never have created it
    assert set(runner.read_json(ctx.paths.state)["failed_jobs"]) == failed


def test_merge_state_without_a_pilot(tmp_path):
    ctx = actions.Context("full", "pendulum", tmp_path)
    assert list(actions.merge_state(ctx).failed) == ["pilot"]


def test_manifests_must_agree(tmp_path):
    ctx = context(tmp_path)
    assert actions.manifest_problems(ctx.paths)  # no pilot manifest
    facts = {"fingerprint": ctx.fingerprint, "git_commit": "abc", "python": "3.13", "torch": "2", "numpy": "2", "source_sha256": {}, "git_dirty_paths": ""}
    atomic_write_json(ctx.paths.run / "manifest.json", facts)
    atomic_write_json(ctx.paths.run / "manifests" / "main_pendulum_s10.json", facts)
    assert actions.manifest_problems(ctx.paths) == []
    atomic_write_json(ctx.paths.run / "manifests" / "main_pendulum_s11.json", {**facts, "numpy": "3"})
    atomic_write_json(ctx.paths.run / "manifests" / "main_pendulum_s12.json", {**facts, "git_commit": "def"})
    atomic_write_json(ctx.paths.run / "manifests" / "main_pendulum_s13.json", {**facts, "git_dirty_paths": "M src/x.py"})
    assert len(actions.manifest_problems(ctx.paths)) == 3


def test_emit_writes_a_step_output(tmp_path, monkeypatch):
    target = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    actions.emit("main", json.dumps([{"seed": 10}], separators=(",", ":")))
    assert target.read_text() == 'main=[{"seed":10}]\n'


def test_h14_timing_runs(tmp_path, capsys):
    assert actions.main(["time-h14", "--updates", "2", "--out", str(tmp_path)]) == 0
    assert "H14 timing" in capsys.readouterr().out
