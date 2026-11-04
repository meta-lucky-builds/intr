"""Thread G: the choice rule, the guards against an easier task, and the whole pipeline on tiny settings."""

import json
import math
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import merge_results  # noqa: E402
import thread_a  # noqa: E402
import thread_g  # noqa: E402
from intr.envs import PuckEnv, make_env  # noqa: E402
from intr.models import ModelConfig  # noqa: E402


def _arm(name):
    return next(a for a in thread_g.ARMS if a.name == name)


def test_round_one_arm_is_the_first_round_config():
    a, g = thread_a.full_config(), thread_g.full_config()
    for key in ("z_dim", "hidden", "n_hidden", "feature_dim", "train_envs", "train_traj", "train_steps", "n_candidates", "n_elites", "n_iters"):
        assert getattr(a, key) == getattr(g, key), key
    assert thread_g.arm_train_config(_arm("round_one"), g) == a.train
    env = make_env("puck", 2, "train", 0)
    assert thread_g.arm_model_config(_arm("round_one"), env, g, 8) == ModelConfig(8, 2, a.z_dim, a.hidden, a.n_hidden, a.feature_dim)


def test_default_seeds_exclude_the_ones_that_informed_the_hypotheses():
    assert thread_g.full_config().seeds == [5, 6, 7, 8, 9]


def test_no_arm_can_change_the_task_or_the_planner():
    allowed = {
        "name", "width_scale", "extra_layers", "update_scale", "loss", "latent_samples_per_env",
        "latent_block_steps", "latent_lr_decay", "input_extra", "contact_data", "planner_rounds", "puck_only",
    }  # fmt: skip
    assert {f.name for f in fields(thread_g.Arm)} == allowed
    env = make_env("puck", 2, "train", 0)
    cem = thread_g.cem_for(env, thread_g.full_config())
    assert (cem.horizon, cem.n_candidates, cem.n_elites, cem.n_iters) == (10, 32, 6, 3)
    assert (cem.init_std, cem.min_std, cem.noise_beta) == (0.6, 0.05, 0.5)
    assert (PuckEnv.success_dist, PuckEnv.episode_steps, PuckEnv.plan_horizon) == (0.15, 50, 10)
    assert (PuckEnv.f_max, PuckEnv.contact_k, PuckEnv.contact_c) == (10.0, 400.0, 2.0)


def _cost(macs, transitions=1):
    return {"macs": macs, "transitions": transitions}


def test_choice_takes_the_cheapest_passing_arm():
    candidates = [("a", _cost(3.0), False), ("b", _cost(5.0), True), ("c", _cost(4.0), True)]
    assert thread_g.choose_arm(candidates) == "c"
    assert thread_g.choose_arm([("a", _cost(4.0, 2), True), ("b", _cost(4.0, 1), True)]) == "b"
    assert thread_g.choose_arm([("a", _cost(4.0), True), ("b", _cost(4.0), True)]) == "a"


def test_choice_says_none_when_no_arm_passes():
    assert thread_g.choose_arm([("a", _cost(1.0), False)]) == thread_g.NO_ARM
    assert thread_g.choose_arm([]) == thread_g.NO_ARM


def _scores(val=0.8, interp=0.8, latent_gain=0.2, heldout_val=0.1, heldout_interp=0.1, fresh=0.1):
    return {
        "success_val_in_latent": val,
        "success_interp_latent": interp,
        "latent_gain": latent_gain,
        "heldout_gain_val_in": heldout_val,
        "heldout_gain_interp": heldout_interp,
        "fresh_gain": fresh,
    }


def test_gates():
    assert thread_g.gates_for(_scores()) == {"g1_val": True, "g1": True, "latent_generalizes": True}
    assert not thread_g.gates_for(_scores(val=0.49))["g1_val"]
    assert not thread_g.gates_for(_scores(interp=0.49))["g1"]
    assert not thread_g.gates_for(_scores(latent_gain=0.09))["g1"]
    assert not thread_g.gates_for(_scores(heldout_val=-0.01))["g1_val"]
    assert not thread_g.gates_for(_scores(heldout_interp=-0.01))["latent_generalizes"]
    assert not thread_g.gates_for(_scores(fresh=-0.01))["latent_generalizes"]
    assert not thread_g.gates_for(_scores(val=math.nan))["g1_val"]


def test_validation_gate_does_not_read_test_scores():
    high_test = thread_g.gates_for(_scores(val=0.0, interp=1.0))
    assert high_test["g1"] and not high_test["g1_val"]
    low_test = thread_g.gates_for(_scores(val=1.0, interp=0.0, heldout_interp=-1.0))
    assert low_test["g1_val"] and not low_test["g1"]


def test_cost_orders_the_arms_as_intended():
    env = make_env("puck", 2, "train", 0)
    cfg = thread_g.full_config()
    cost = {a.name: thread_g.arm_cost(a, env, cfg) for a in thread_g.ARMS}
    assert cost["combined"]["macs"] > cost["capacity_updates"]["macs"] > cost["round_one"]["macs"]
    assert cost["latent_fit"]["macs"] > cost["round_one"]["macs"]
    assert cost["contact_data"]["transitions"] == 2 * cost["round_one"]["transitions"]
    assert cost["huber"] == cost["round_one"]


def test_latent_update_count_in_the_cost_matches_training():
    """The cost must describe what train_latent does, so the same block arithmetic is checked against it."""
    from intr.train import TrainConfig, TransitionData, train_latent
    from intr.envs import make_generator

    cfg = thread_g.smoke_config()
    env = make_env("puck", cfg.train_envs, "train", 0)
    data = TransitionData.collect(env, cfg.train_traj, cfg.train_steps, make_generator("cost"))
    arm = _arm("latent_fit")
    tcfg = thread_g.arm_train_config(arm, cfg)
    backbone = train_latent(data, thread_g.arm_model_config(arm, env, cfg, cfg.z_dim), tcfg, seed=0)
    blocks = 1 + math.ceil((tcfg.model_steps - tcfg.warmup_steps) / tcfg.block_steps)
    assert backbone.stats["latent_updates"] == blocks * tcfg.latent_block_steps
    assert isinstance(tcfg, TrainConfig)


def test_puck_only_settings_do_nothing_elsewhere():
    active = thread_g.effective_arm(_arm("combined"), "drone")
    assert active.input_extra == "" and active.contact_data is False
    assert active.planner_rounds == _arm("combined").planner_rounds
    assert thread_g.effective_arm(_arm("combined"), "puck") == _arm("combined")


@pytest.fixture(scope="module")
def small_runs():
    """Three arms that between them use every data path, on both families. Takes a minute or so."""
    cfg = thread_g.smoke_config()
    arms = (_arm("round_one"), _arm("relative_inputs"), _arm("combined"))
    return cfg, {family: {str(seed): thread_g.run_job(family, seed, cfg, arms) for seed in (5, 6)} for family in ("puck", "drone")}


def test_job_has_what_was_asked_for(small_runs):
    _, runs = small_runs
    job = runs["puck"]["5"]
    scores = job["arms"]["round_one"]["scores"]
    for key in (
        "train_mse_latent", "train_mse_nolatent", "latent_gain", "own_gain", "fresh_gain", "r2_mass", "r2_friction",
        "success_interp_latent", "success_interp_nolatent", "success_val_in_latent", "h1_interp_latent", "h10_interp_latent",
        "h30_interp_latent", "contact_windows_rmse_latent", "free_windows_rmse_latent", "heldout_gain_val_in", "heldout_gain_interp",
    ):  # fmt: skip
        assert key in scores, key
    assert set(job["oracle"]) == {"val_in", "interp", "extrap", "oof"}
    assert set(job["persistence_open_loop"]["interp"]) == {"h1", "h10", "h30"}
    assert job["chosen_on_this_seed"] in ("round_one", "relative_inputs", "combined", thread_g.NO_ARM)
    assert math.isfinite(scores["contact_windows_rmse_latent"]) or math.isfinite(scores["free_windows_rmse_latent"])


def test_arms_on_a_seed_are_paired(small_runs):
    _, runs = small_runs
    job = runs["puck"]["5"]
    a, c = job["arms"]["round_one"], job["arms"]["combined"]
    assert c["data"]["transitions_per_env"] > a["data"]["transitions_per_env"]
    assert a["train"]["latent"]["stats"]["model_updates"] < c["train"]["latent"]["stats"]["model_updates"]


def test_drone_skips_puck_only_arms_and_scores_a_second_family_gate(small_runs):
    cfg, runs = small_runs
    job = runs["drone"]["5"]
    assert "skipped" in job["arms"]["relative_inputs"]
    assert math.isnan(job["arms"]["round_one"]["scores"]["contact_windows_rmse_latent"])
    summary = thread_g.summarize(runs["drone"], cfg.n_boot)
    assert summary["second_family_gate"]["arm"] in summary["arms"]
    assert "relative_inputs" not in summary["arms"]


def test_summary_survives_results_file_and_merge_and_agrees(small_runs):
    """null in a results file comes back as NaN at the merge, so the summary must read the same in both cases."""
    cfg, runs = small_runs
    direct = thread_g._jsonable({f: thread_g.summarize(j, cfg.n_boot) for f, j in runs.items()})
    stored = json.loads(json.dumps(thread_g._jsonable(runs), allow_nan=False))
    merged = merge_results.recompute_summary("g", stored, asdict(cfg))
    json.dumps(merged, allow_nan=False)
    assert merged["puck"]["chosen_arm"] == direct["puck"]["chosen_arm"]
    assert merged["puck"]["statement"] == direct["puck"]["statement"]
    assert merged["drone"]["second_family_gate"] == direct["drone"]["second_family_gate"]
    assert merged["puck"]["arms"]["round_one"]["means"] == direct["puck"]["arms"]["round_one"]["means"]


def test_statement_says_plainly_when_no_arm_passes(small_runs):
    cfg, runs = small_runs
    summary = thread_g.summarize(runs["puck"], cfg.n_boot)
    if summary["chosen_arm"] == thread_g.NO_ARM:  # expected at smoke sizes
        assert summary["statement"].startswith("NO ARM PASSES")
        assert summary["no_arm_passes"] is True
    else:
        assert summary["chosen_arm"] in summary["statement"]


def test_rule_in_summary_uses_validation_only(small_runs):
    cfg, runs = small_runs
    stored = json.loads(json.dumps(thread_g._jsonable(runs["puck"])))
    for job in stored.values():
        for record in job["arms"].values():
            if "scores" in record:
                record["scores"]["success_interp_latent"] = 1.0
                record["scores"]["success_val_in_latent"] = 0.0
                record["gates"] = thread_g.gates_for(record["scores"])
    summary = thread_g.summarize(merge_results.revive_nan(stored), cfg.n_boot)
    assert summary["chosen_arm"] == thread_g.NO_ARM
    assert summary["statement"].startswith("NO ARM PASSES")


def test_command_line_smoke_writes_a_file_the_merge_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(thread_g, "ARMS", (_arm("round_one"),))
    monkeypatch.setattr(thread_g.run_job, "__defaults__", ((_arm("round_one"),),))
    monkeypatch.setattr(sys, "argv", ["thread_g.py", "--smoke", "--seed-list", "5", "--families", "drone", "--threads", "1", "--out", str(tmp_path)])
    thread_g.main()
    path = tmp_path / "thread_g_smoke.json"
    doc = json.loads(path.read_text())
    assert {"g_gates", "jobs", "summary", "config", "metadata"} <= set(doc)
    assert list(doc["jobs"]["drone"]) == ["5"]
    assert doc["summary"]["drone"]["statement"]
    loaded = merge_results.load_input(path, "g")
    assert merge_results.merge("g", [loaded]).merged["summary"]["drone"]["chosen_arm"] == doc["summary"]["drone"]["chosen_arm"]


def test_full_flag_is_what_the_workflow_calls():
    """The workflow passes exactly these flags. Parsing only: --help exits before any work."""
    done = subprocess.run([sys.executable, str(ROOT / "experiments" / "thread_g.py"), "--help"], capture_output=True, text=True, check=True)
    for flag in ("--full", "--smoke", "--seed-list", "--families", "--threads", "--out"):
        assert flag in done.stdout
