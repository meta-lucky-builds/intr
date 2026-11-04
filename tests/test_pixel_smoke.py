"""The whole path on tiny settings, CPU, through the real runner and worker processes. Numbers mean nothing here."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("thread_p")
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run(
        [sys.executable, "-m", "intr.pixel.runner", "--smoke", "--skip-tests", "--no-publish", "--families", "pendulum", "--out", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0], out, env, done.stdout


def job(run: Path, name: str) -> dict:
    return json.loads((run / "jobs" / f"{name}.json").read_text())


def test_smoke_writes_every_job_kind_and_a_table(smoke_run):
    run, out, _, stdout = smoke_run
    for name in ("main_pendulum_s10", "main_pendulum_s11", "h5_pendulum_s10", "h14_pendulum_s10", "h16_pendulum_s10", "h17_pendulum_s10"):
        assert job(run, name)["status"] == "ok"
    summary = json.loads((run / "summary.json").read_text())
    fam = summary["families"]["pendulum"]
    assert {"H1p", "H12p", "H4p", "H3p", "H5p", "H14p", "H16", "H17"} <= set(fam)
    text = (run / "summary.txt").read_text()
    for hyp in ("H1p", "H12p", "H4p", "H3p", "H5p", "H14p"):
        assert hyp in text
    assert "Thread P verdicts" in stdout
    assert list(out.glob("thread_p_*.tar.gz"))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["source_sha256"] and "torch" in manifest


def test_same_data_across_methods_and_no_test_tier_in_the_probe(smoke_run):
    run = smoke_run[0]
    main = job(run, "main_pendulum_s10")
    assert main["same_data_check"] is True
    assert main["train"]["probe"]["source_tiers"] == ["train"]
    for tier in ("train", "interp", "extrap", "oof"):
        for k, cells in main["tiers"][tier]["k"].items():
            assert len({cell["data_sha256"] for cell in cells.values()}) == 1, (tier, k)
    assert main["tiers"]["interp"]["k"]["1"]["rls@latent"]["data_sha256"] != main["tiers"]["interp"]["k"]["3"]["rls@latent"]["data_sha256"]
    h14 = job(run, "h14_pendulum_s10")
    assert h14["tiers"]["interp"]["1"]["error_fed"]["data_sha256"] == main["tiers"]["interp"]["k"]["1"]["rls@latent"]["data_sha256"]


def test_selection_saw_validation_only_and_frozen_keeps_its_starting_point(smoke_run):
    main = job(smoke_run[0], "main_pendulum_s10")
    assert main["rls_best"] in ("rls@latent", "rls@nolatent")
    assert len(main["val_errors"]["rls@latent"]) == 2
    assert main["h3"]["selection"]["hp"] == {"lr": 0.1, "passes": 1}
    interp = main["tiers"]["interp"]
    assert "oracle_closed_loop" in interp and "closed_loop" in interp["k"]["1"]["frozen@latent"]
    assert "closed_loop" not in interp["k"]["3"]["frozen@latent"]
    assert "closed_loop" not in main["tiers"]["train"]["k"]["1"]["frozen@latent"]
    assert interp["k"]["1"]["frozen@latent"]["open_loop"]["h0"] == interp["persistence"]["h0"]  # both decode the same true embedding


def test_controls_are_real_manipulations(smoke_run):
    h14 = job(smoke_run[0], "h14_pendulum_s10")["tiers"]["interp"]["1"]
    assert h14["error_fed"]["fed_differs_from_own"] == 0.0
    assert h14["no_error"]["fed_differs_from_own"] == 0.0
    assert h14["shuffled"]["fed_differs_from_own"] > 0.0
    assert h14["error_fed"]["adapt_counts"]["test_time_gradients"] == 0


def test_appearance_changes_pixels_only(smoke_run):
    h16 = job(smoke_run[0], "h16_pendulum_s10")["appearance"]
    assert h16["physics_unchanged"] is True
    assert h16["pixel_distance"] > 0.01


def test_h5_arms_share_starts_and_amount_of_data(smoke_run):
    h5 = job(smoke_run[0], "h5_pendulum_s10")["tiers"]["interp"]
    assert h5["starts_match_default"] is True
    assert {cell["n_transitions"] for cell in h5["1"].values()} == {10}
    assert {cell["n_transitions"] for cell in h5["3"].values()} == {30}


def test_rerun_skips_everything_and_a_crashed_extra_leaves_main_alone(smoke_run):
    run, out, env, _ = smoke_run
    (run / "jobs" / "h17_pendulum_s10.json").unlink()
    before = (run / "jobs" / "main_pendulum_s10.json").read_bytes()
    done = subprocess.run(
        [sys.executable, "-m", "intr.pixel.runner", "--smoke", "--skip-tests", "--no-publish", "--families", "pendulum", "--out", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-3000:]
    assert "already finished, skipped" in done.stdout
    assert "[last (H16, H17)] 1 to run" in done.stdout  # only the job that went missing
    assert (run / "jobs" / "main_pendulum_s10.json").read_bytes() == before
    assert (run / "jobs" / "h17_pendulum_s10.json").exists()
