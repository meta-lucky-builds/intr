"""The scheduler with fake GPUs and fake workers; publishing with a fake token. No hardware and no network."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from intr.pixel import config
from intr.pixel import publish as pub
from intr.pixel.scheduler import Job, Slot, atomic_write_json, choose_concurrency, is_complete, parse_gpu_query, plan_slots, run_jobs, worker_env

FAKE_TOKEN = "ghp_FAKE0123456789fake0123456789FAKEtoken"
FP = "abc123"

WORKER = r"""
import json, os, sys, time
out, marker, behaviour = sys.argv[1], sys.argv[2], sys.argv[3]
with open(marker, "a") as handle:
    handle.write(os.environ["CUDA_VISIBLE_DEVICES"] + "\n")
attempts = sum(1 for _ in open(marker))
print("env", dict(os.environ))
time.sleep(0.05)
if behaviour == "fail" or (behaviour == "flaky" and attempts == 1):
    sys.exit(3)
if behaviour == "no_output":
    sys.exit(0)
tmp = out + ".part"
json.dump({"status": "ok", "fingerprint": sys.argv[4], "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "threads": os.environ["OMP_NUM_THREADS"]}, open(tmp, "w"))
os.replace(tmp, out)
"""


def fake_job(tmp: Path, name: str, behaviour: str = "ok", fingerprint: str = FP) -> Job:
    return Job(name, [sys.executable, "-c", WORKER, str(tmp / f"{name}.json"), str(tmp / f"{name}.marker"), behaviour, fingerprint], tmp / f"{name}.json", tmp / "logs" / f"{name}.log", FP)


def attempts(tmp: Path, name: str) -> int:
    path = tmp / f"{name}.marker"
    return len(path.read_text().splitlines()) if path.exists() else 0


def quiet(_line) -> None:
    pass


def run(jobs, slots):
    return run_jobs(jobs, slots, poll_s=0.02, progress=quiet)


def test_jobs_are_pinned_to_their_slot_and_get_thread_limits(tmp_path):
    jobs = [fake_job(tmp_path, f"j{i}") for i in range(6)]
    report = run(jobs, [Slot("0", 3), Slot("1", 3)])
    assert sorted(report.finished) == sorted(j.name for j in jobs) and not report.failed
    outputs = [json.loads(j.output.read_text()) for j in jobs]
    assert {o["gpu"] for o in outputs} == {"0", "1"}
    assert all(o["threads"] == "3" for o in outputs)


def test_cpu_slot_hides_gpus(tmp_path):
    job = fake_job(tmp_path, "c")
    run([job], [Slot(None, 1)])
    assert json.loads(job.output.read_text())["gpu"] == ""


def test_resume_skips_finished_jobs(tmp_path):
    job = fake_job(tmp_path, "r")
    run([job], [Slot("0")])
    assert attempts(tmp_path, "r") == 1
    report = run([job], [Slot("0")])
    assert report.skipped == ["r"] and not report.finished
    assert attempts(tmp_path, "r") == 1


def test_output_with_another_fingerprint_is_not_finished(tmp_path):
    job = fake_job(tmp_path, "s")
    atomic_write_json(job.output, {"status": "ok", "fingerprint": "older code"})
    assert not is_complete(job)
    job.output.write_text("{ truncated")
    assert not is_complete(job)


def test_one_failure_does_not_stop_others_and_gets_one_retry(tmp_path):
    jobs = [fake_job(tmp_path, "good1"), fake_job(tmp_path, "bad", "fail"), fake_job(tmp_path, "good2")]
    report = run(jobs, [Slot("0"), Slot("1")])
    assert sorted(report.finished) == ["good1", "good2"]
    assert list(report.failed) == ["bad"] and report.failed["bad"]["code"] == 3
    assert report.retried == ["bad"] and attempts(tmp_path, "bad") == 2


def test_flaky_job_succeeds_on_retry(tmp_path):
    report = run([fake_job(tmp_path, "f", "flaky")], [Slot("0")])
    assert report.finished == ["f"] and report.retried == ["f"] and not report.failed


def test_clean_exit_without_output_counts_as_failure(tmp_path):
    report = run([fake_job(tmp_path, "n", "no_output")], [Slot("0")])
    assert list(report.failed) == ["n"]


def test_atomic_write_leaves_old_file_on_crash(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    atomic_write_json(path, {"v": 1})

    def boom(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(path, {"v": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_plan_slots():
    gpus = [{"index": str(i), "memory_mb": 24000} for i in range(4)]
    slots = plan_slots(gpus, 32, 2)
    assert len(slots) == 8 and {s.gpu for s in slots} == {"0", "1", "2", "3"} and slots[0].threads == 4
    assert [s.gpu for s in slots[:4]] == ["0", "1", "2", "3"]  # spread before doubling up
    limited = plan_slots(gpus, 8, 4)
    assert len(limited) == 8 and limited[0].threads == 1
    assert len(plan_slots(gpus[:1], 2, 8)) == 2
    assert len(plan_slots(gpus * 2, 8, 1)) == 8
    cpu = plan_slots([], 8, 1)
    assert all(s.gpu is None for s in cpu) and len(cpu) == 4


def test_parse_gpu_query():
    text = "0, NVIDIA GeForce RTX 4090, 24564\n1, NVIDIA GeForce RTX 4090, 24564\n\ngarbage\n"
    assert parse_gpu_query(text) == [{"index": "0", "name": "NVIDIA GeForce RTX 4090", "memory_mb": 24564}, {"index": "1", "name": "NVIDIA GeForce RTX 4090", "memory_mb": 24564}]


def test_choose_concurrency():
    assert choose_concurrency({1: 100.0, 2: 180.0, 3: 190.0}, {}) == 2
    assert choose_concurrency({1: 100.0, 2: 105.0, 3: 400.0}, {}) == 1
    assert choose_concurrency({1: 100.0, 2: 180.0}, {2: False}) == 1


def test_estimate_is_positive_and_grows_with_slower_rates():
    bench = {"jepa_steps_per_s": 150.0, "core_steps_per_s": 400.0, "model_calls_per_s": 800.0, "jepa_share": 1.5}
    cfg = config.full_config()
    fast = config.estimate_job_seconds(cfg, "pendulum", bench, 2)
    slow = config.estimate_job_seconds(cfg, "pendulum", {**bench, "core_steps_per_s": 100.0}, 2)
    assert all(v > 0 for v in fast.values()) and slow["main"] > fast["main"]
    assert config.estimate_job_seconds(cfg, "puck", bench, 2)["main"] > fast["main"]


def test_fingerprint_ignores_seed_list_but_not_settings():
    from intr.pixel.runner import fingerprint_of

    a, b, c = config.full_config(), config.full_config(), config.full_config()
    b.seeds = [10]
    c.stack = 3
    assert fingerprint_of({"x": "1"}, a) == fingerprint_of({"x": "1"}, b)
    assert fingerprint_of({"x": "1"}, a) != fingerprint_of({"x": "1"}, c)
    assert fingerprint_of({"x": "1"}, a) != fingerprint_of({"x": "2"}, a)


def test_supervisor_modules_do_not_import_torch():
    code = "import sys; import intr.pixel.runner, intr.pixel.scheduler, intr.pixel.publish, intr.pixel.config; sys.exit(1 if 'torch' in sys.modules else 0)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


# ---------------------------------------------------------------- the token


def scan(folder: Path) -> list[str]:
    found = []
    for path in folder.rglob("*"):
        if path.is_file() and FAKE_TOKEN.encode() in path.read_bytes():
            found.append(str(path))
    return found


def test_workers_never_receive_a_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("GH_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
    env = worker_env(Slot("0"))
    assert not any(FAKE_TOKEN in value for value in env.values())
    jobs = [fake_job(tmp_path, "t"), fake_job(tmp_path, "t2", "fail")]
    run(jobs, [Slot("0")])  # the worker prints its whole environment into its log
    assert scan(tmp_path) == []


def test_redact_and_clean_remote(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    assert FAKE_TOKEN not in pub.redact(f"fatal: bad {FAKE_TOKEN} here")
    assert pub.clean_remote(f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r.git") == "https://github.com/o/r.git"
    assert pub.clean_remote(f"https://{FAKE_TOKEN}@github.com/o/r") == "https://github.com/o/r"
    assert pub.clean_remote("git@github.com:o/r.git") == "https://github.com/o/r.git"
    assert pub.clean_remote("/tmp/remote.git") == "/tmp/remote.git"


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(*args, cwd=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def results_folder(tmp: Path) -> Path:
    folder = tmp / "run"
    (folder / "jobs").mkdir(parents=True)
    (folder / "cache").mkdir()
    (folder / "checkpoints").mkdir()
    (folder / "jobs" / "main.json").write_text("{}")
    (folder / "summary.txt").write_text("table")
    (folder / "cache" / "big.pt").write_bytes(b"0" * 10)
    (folder / "checkpoints" / "w.pt").write_bytes(b"0" * 10)
    return folder


@needs_git
def test_push_goes_only_to_a_new_results_branch(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    remote = tmp_path / "remote.git"
    git("init", "-q", "--bare", str(remote))
    seed = tmp_path / "seed"
    seed.mkdir()
    git("init", "-q", "-b", "main", cwd=seed)
    (seed / "f").write_text("main")
    git("add", "f", cwd=seed)
    git("commit", "-q", "-m", "first", cwd=seed)
    git("push", "-q", str(remote), "main", "main:mission/world-model-techniques", cwd=seed)
    before = git("rev-parse", "main", "mission/world-model-techniques", cwd=remote)

    result = pub.publish_results(results_folder(tmp_path), seed, "20261104-010101", remote=str(remote))
    assert result == {"pushed": True, "branch": "results/thread-p-20261104-010101"}
    assert git("rev-parse", "main", "mission/world-model-techniques", cwd=remote) == before
    files = git("ls-tree", "-r", "--name-only", "results/thread-p-20261104-010101", cwd=remote).splitlines()
    assert "results/thread_p/20261104-010101/summary.txt" in files
    assert not any("cache" in f or "checkpoints" in f for f in files)
    assert scan(tmp_path) == []
    assert FAKE_TOKEN not in git("config", "--list", cwd=seed)


@needs_git
def test_failed_push_reports_without_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_TOKEN)
    result = pub.publish_results(results_folder(tmp_path), tmp_path, "s", remote=str(tmp_path / "does_not_exist.git"))
    assert result["pushed"] is False and FAKE_TOKEN not in result["reason"]


def test_no_token_means_no_push(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert pub.publish_results(results_folder(tmp_path), tmp_path, "s")["pushed"] is False


@needs_git
def test_origin_with_embedded_token_is_scrubbed_on_disk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    git("remote", "add", "origin", f"https://x-access-token:{FAKE_TOKEN}@github.com/o/r.git", cwd=repo)
    assert pub.scrub_origin(repo) == "https://github.com/o/r.git"
    assert scan(repo) == []


def test_tar_holds_everything_but_the_cache(tmp_path):
    folder = results_folder(tmp_path)
    tar = pub.make_tar(folder, tmp_path / "out.tar.gz")
    with tarfile.open(tar) as handle:
        names = handle.getnames()
    assert "run/summary.txt" in names and "run/checkpoints/w.pt" in names
    assert not any("cache" in n for n in names)


def test_script_never_traces_or_echoes_the_token():
    text = (Path(__file__).resolve().parents[1] / "scripts" / "gpu_run.sh").read_text()
    code_lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    assert not any("set -x" in line for line in code_lines)
    assert not any("GITHUB_TOKEN" in line for line in code_lines)
