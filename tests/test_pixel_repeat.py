"""Action repeat and the noise-floor diagnostic: repeat 1 is today, repeat 3 keeps physics and physical time, and the diagnostic touches no result."""

import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))

import thread_p  # noqa: E402
from test_pixel_repair import fake_runner, kind_of, seed_of  # noqa: E402
from intr.adapt import AdaptedPredictor  # noqa: E402
from intr.envs import FAMILIES, OBS_DT, make_env, make_generator  # noqa: E402
from intr.eval import CEMConfig, OraclePredictor, compute_obs_scale  # noqa: E402
from intr.looped import SequenceData  # noqa: E402
from intr.models import DynamicsModel, ModelConfig  # noqa: E402
from intr.pixel import runner  # noqa: E402
from intr.pixel.config import PILOT_SEED, REGISTERED_SEEDS, config_for, decision_steps, full_config  # noqa: E402
from intr.pixel.error_fed import sample_sequences  # noqa: E402
from intr.pixel.planning import EvalContext, closed_loop_model, closed_loop_oracle, make_windows  # noqa: E402
from intr.pixel.render import render  # noqa: E402
from intr.pixel.sim import burn_in, collect_states, env_on, rollout_policy  # noqa: E402
from intr.pixel.world import ConvEncoder, Observed, StateProbe, embedding_bounds, noise_floor  # noqa: E402

CPU = torch.device("cpu")
CEM = CEMConfig(horizon=3, n_candidates=8, n_elites=2, n_iters=2)


def three_steps(env, state, action):
    for _ in range(3):
        state = env.step(state, action)
    return state


# ---------------------------------------------------------------- repeat 1 is today


def test_repeat_1_is_todays_settings_and_simulator():
    assert asdict(config_for("full", None, 1)) == asdict(full_config())
    cfg = full_config()
    assert cfg.action_repeat == 1 and cfg.train_steps == 60 and cfg.train_traj == 16
    for family in FAMILIES:
        env = make_env(family, 3, "train", 0)
        moved = env_on(env, CPU, 1)
        assert "step" not in vars(moved) and "success_window" not in vars(moved)
        assert thread_p.cem_for(env, cfg).horizon == env.plan_horizon
        assert thread_p.episode_steps(env, cfg) == env.episode_steps


@pytest.mark.parametrize("family", FAMILIES)
def test_repeat_1_gives_todays_numbers(family):
    env = make_env(family, 3, "train", 0)
    a = collect_states(env, env_on(env, CPU), 2, 8, env.default_probe, make_generator("r", family), 1, CPU)
    b = collect_states(env, env_on(env, CPU, 1), 2, 8, env.default_probe, make_generator("r", family), 1, CPU)
    assert torch.equal(a.states, b.states) and torch.equal(a.obs, b.obs) and torch.equal(a.act, b.act)


# ---------------------------------------------------------------- repeat 3 keeps physics


@pytest.mark.parametrize("family", FAMILIES)
def test_a_held_step_is_three_true_steps_and_changes_nothing_else(family):
    env = make_env(family, 3, "train", 0)
    before = {k: v.clone() for k, v in env.params.items()}
    held = env_on(env, CPU, 3)
    plain = env_on(env, CPU)
    state = env.reset(2, make_generator("s", family))
    action = 2.0 * torch.rand(3, 2, env.act_dim, generator=make_generator("a", family)) - 1.0
    assert torch.equal(held.step(state, action), three_steps(plain, state, action))
    assert torch.equal(env.step(state, action), plain.step(state, action))  # the original stays one step
    assert held.dt == env.dt and held.n_substeps == env.n_substeps
    assert all(torch.equal(before[k], held.params[k]) and torch.equal(before[k], env.params[k]) for k in before)


@pytest.mark.parametrize("family", FAMILIES)
def test_burn_in_and_data_hold_each_action_and_alignment_survives(family):
    env = make_env(family, 3, "train", 0)
    held, plain = env_on(env, CPU, 3), env_on(env, CPU)
    s0 = env.reset(2, make_generator("s0", family))
    burn = 1
    traj = rollout_policy(held, s0, 6, "iid", make_generator("iid", family), burn)
    zero = torch.zeros(3, 2, env.act_dim, dtype=OBS_DT)
    assert torch.equal(traj.states[:, :, 1], three_steps(plain, s0, zero))
    assert torch.equal(burn_in(held, s0, burn)[-1], traj.states[:, :, burn])
    for t in range(6):
        assert torch.equal(traj.states[:, :, t + burn + 1], three_steps(plain, traj.states[:, :, t + burn], traj.act[:, :, t]))
        assert torch.equal(traj.obs[:, :, t], plain.obs(traj.states[:, :, t + burn]))
    assert render(family, traj.states, 16).shape[2] == 2 + 6  # one frame per decision, stack + T


@pytest.mark.parametrize("family", FAMILIES)
def test_the_oracles_model_holds_the_action_too(family):
    env = make_env(family, 3, "train", 0)
    held, plain = env_on(env, CPU, 3), env_on(env, CPU)
    state = env.reset(2, make_generator("o", family))
    action = torch.zeros(3, 2, env.act_dim, dtype=OBS_DT)
    predicted = OraclePredictor(held).step(plain.obs(state), action)
    truth = plain.obs(three_steps(plain, plain.state_from_obs(plain.obs(state)), action))
    assert torch.equal(predicted, truth)


def test_the_closed_loop_spends_three_simulator_steps_per_decision(monkeypatch):
    env = make_env("pendulum", 2, "interp", 0)
    calls = []
    real = type(env).step

    def counting(self, state, action):
        calls.append(1)
        return real(self, state, action)

    monkeypatch.setattr(type(env), "step", counting)
    held = env_on(env, CPU, 3)
    D = 5
    model = DynamicsModel(ModelConfig(D, env.act_dim, 0, 8, 1, 4)).requires_grad_(False)
    encoder = ConvEncoder(6, 16, D, 8).eval().requires_grad_(False)
    probe = StateProbe(D, env.obs_dim, 8).requires_grad_(False)
    low, high = embedding_bounds(torch.randn(50, D, generator=make_generator("b")))
    ctx = EvalContext(probe, compute_obs_scale("pendulum"), env.obs_low, env.obs_high, low, high)
    calls.clear()
    out = closed_loop_model(held, AdaptedPredictor(model), ctx, "pendulum", 16, 2, encoder, env.sample_task(make_generator("t")), CEM, make_generator("g"), 2)
    assert len(calls) == 3 * (1 + 2)  # one burn-in decision and two planned ones
    assert out["actions"].shape[1] == 2
    assert closed_loop_oracle(held, env.sample_task(make_generator("t")), CEM, make_generator("g"), 2, 1)["actions"].shape[1] == 2


# ---------------------------------------------------------------- scaling


@pytest.mark.parametrize("family,episode,horizon", [("pendulum", 20, 4), ("drone", 20, 4), ("puck", 17, 4)])
def test_closed_loop_scaling_keeps_physical_time(family, episode, horizon):
    cfg = config_for("full", "idm", 3)
    env = make_env(family, 1, "train", "b")
    assert thread_p.episode_steps(env, cfg) == episode and thread_p.cem_for(env, cfg).horizon == horizon
    assert env.episode_steps <= 3 * episode < env.episode_steps + 3
    assert env.plan_horizon <= 3 * horizon < env.plan_horizon + 3


def test_data_scaling_keeps_transitions_and_physical_time():
    base, cfg = full_config(), config_for("full", "state", 3)
    assert (cfg.train_steps, cfg.train_traj) == (20, 48)
    assert cfg.train_steps * cfg.train_traj == base.train_steps * base.train_traj
    assert 3 * cfg.train_steps == base.train_steps
    assert (cfg.holdout_steps, cfg.holdout_traj, cfg.adapt_steps) == (base.holdout_steps, base.holdout_traj, base.adapt_steps)
    E, B, T, A, D, O = 2, 2, cfg.holdout_steps, 1, 4, 3
    make_windows(torch.zeros(E, B, T + 1, D), torch.zeros(E, B, T + 1, O), torch.zeros(E, B, T, A), CPU)


def test_smoke_overrides_stay_in_decisions():
    cfg = config_for("smoke", "idm", 3)
    env = make_env("puck", 1, "train", "b")
    assert thread_p.episode_steps(env, cfg) == cfg.episode_override and thread_p.cem_for(env, cfg).horizon == cfg.horizon_override


@pytest.mark.parametrize("family", ["pendulum", "drone"])
def test_success_window_keeps_its_physical_span(family):
    env = make_env(family, 2, "train", 0)
    held = env_on(env, CPU, 3)
    assert env.success_window == 10 and held.success_window == 4 == decision_steps(10, 3)
    assert (held.success_window - 1) * 3 == env.success_window - 1


def test_puck_success_needs_no_window():
    assert not hasattr(env_on(make_env("puck", 2, "train", 0), CPU, 3), "success_window")


def test_held_success_reads_the_last_four_observations():
    env = make_env("pendulum", 2, "train", 0)
    held = env_on(env, CPU, 3)
    hanging, upright = torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])

    def ending_with(n):
        traj = hanging.expand(2, 21, 3).clone()
        traj[:, -n:] = upright
        return traj

    assert held.is_success(ending_with(4), None).all()
    assert not held.is_success(ending_with(3), None).any()
    assert not env.is_success(ending_with(4), None).any()  # one step at a time, it still wants 10


def test_every_simulator_in_the_worker_takes_the_repeat():
    text = (ROOT / "experiments" / "thread_p.py").read_text(encoding="utf-8")
    calls = re.findall(r"env_on\([^)]*\)", text)
    assert calls and all(call == "env_on(env, device, cfg.action_repeat)" for call in calls)


def test_error_fed_phases_do_not_fit_at_repeat_3():
    """Pins the misfit that keeps repeat 3 on the repair path. If this stops raising, that guard can be reviewed."""
    cfg = config_for("full", None, 3)
    O, A = 3, 1
    data = SequenceData(torch.zeros(8, 4, cfg.train_steps + 1, O), torch.zeros(8, 4, cfg.train_steps, A))
    for filter_len, horizon in ((cfg.error_fed.filter_max, 1), (1, cfg.error_fed.horizon_max)):
        with pytest.raises(ValueError):
            sample_sequences(data, 2, filter_len, horizon, 0.5, make_generator("q"))


# ---------------------------------------------------------------- noise floor


def synthetic(sigma, E=4, B=8, T=20):
    gen = make_generator("floor")
    obs = torch.cumsum(0.1 * torch.randn(E, B, T + 1, 2, generator=gen), dim=2)
    mix = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.3, 0.3]])
    emb = obs @ mix.t() + sigma * torch.randn(E, B, T + 1, 3, generator=gen)
    return Observed(emb, obs, torch.zeros(E, B, T, 1))


def test_noise_floor_arithmetic_and_units():
    data = synthetic(0.2)
    out = noise_floor(data, 32, 600, 128, 0, CPU)
    delta = (data.emb[:, :, 1:] - data.emb[:, :, :-1]).reshape(-1, 3)
    assert torch.allclose(torch.tensor(out["delta_variance_per_dim"]), delta.var(0))
    assert torch.allclose(torch.tensor(out["embedding_variance_per_dim"]), data.emb.reshape(-1, 3).var(0))
    mse, var = torch.tensor(out["heldout_mse_per_dim"]), torch.tensor(out["delta_variance_per_dim"])
    assert torch.allclose(torch.tensor(out["ratio_per_dim"]), mse / var)
    assert out["ratio"] == pytest.approx((mse / var).mean().item(), rel=1e-5)
    assert out["ratio_pooled"] == pytest.approx((mse.sum() / var.sum()).item(), rel=1e-5)
    assert out["delta_variance"] == pytest.approx(var.mean().item(), rel=1e-5)
    assert len(out["ratio_per_dim"]) == 3


def test_noise_floor_sees_noise_that_obs_cannot_explain():
    quiet = noise_floor(synthetic(0.02), 32, 600, 128, 0, CPU)
    noisy = noise_floor(synthetic(0.3), 32, 600, 128, 0, CPU)
    assert noisy["ratio"] > quiet["ratio"] and noisy["heldout_mse"] > 2 * quiet["heldout_mse"]
    assert 0.3**2 / 2 < noisy["heldout_mse"] < 0.3**2 * 3


def test_noise_floor_holds_out_whole_trajectories_from_the_fit():
    data = synthetic(0.1)
    base = noise_floor(data, 16, 50, 64, 0, CPU)
    assert base["n_heldout"] == 4 * 2 * 21 and base["n_fit"] == 4 * 6 * 21
    poisoned = data.emb.clone()
    poisoned[:, [3, 7]] += 100.0
    again = noise_floor(Observed(poisoned, data.obs, data.act), 16, 50, 64, 0, CPU)
    assert again["fit_final_loss"] == base["fit_final_loss"]
    assert again["heldout_mse"] > 100 * base["heldout_mse"]


def test_noise_floor_needs_at_least_two_trajectories_and_works_there():
    assert noise_floor(synthetic(0.1, B=2), 8, 5, 16, 0, CPU)["n_heldout"] == 4 * 21


def test_noise_floor_changes_neither_its_input_nor_the_random_state():
    data = synthetic(0.1)
    emb, obs = data.emb.clone(), data.obs.clone()
    torch.manual_seed(7)
    before = torch.get_rng_state()
    noise_floor(data, 8, 5, 16, 0, CPU)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(data.emb, emb) and torch.equal(data.obs, obs)


def test_smoke_pilot_numbers_do_not_depend_on_the_diagnostic(monkeypatch):
    cfg = config_for("smoke")
    family, seed = "pendulum", PILOT_SEED
    worlds = []
    real_build = thread_p.build_world

    def spy(*a, **k):
        out = real_build(*a, **k)
        world = out[0]
        worlds.append((world, copy.deepcopy(world.encoder.state_dict()), {n: copy.deepcopy(b.model.state_dict()) for n, b in world.backbones.items()}))
        return out

    monkeypatch.setattr(thread_p, "build_world", spy)
    real = thread_p.run_pilot(family, seed, cfg, cfg.rungs[0], CPU)
    world, encoder, core = worlds[-1]
    assert all(torch.equal(encoder[k], v) for k, v in world.encoder.state_dict().items())
    assert all(torch.equal(core[n][k], v) for n, b in world.backbones.items() for k, v in b.model.state_dict().items())

    monkeypatch.setattr(thread_p, "noise_floor", lambda *a, **k: {})
    without = thread_p.run_pilot(family, seed, cfg, cfg.rungs[0], CPU)
    assert real["gate"] == without["gate"] and real["collapse"] == without["collapse"]
    assert real["oracle"] == without["oracle"] and real["frozen"] == without["frozen"]
    assert real["train"]["exact_train_loss"] == without["train"]["exact_train_loss"]
    assert real["train"]["latent_r2"] == pytest.approx(without["train"]["latent_r2"], rel=1e-9)  # a threaded solve differs in the last bits between runs
    assert real["train"]["jepa"]["log"] == without["train"]["jepa"]["log"]

    assert real["action_repeat"] == 1
    assert len(real["noise_floor"]["ratio_per_dim"]) == 16
    assert {"ratio", "delta_variance", "embedding_variance"} <= set(real["noise_floor"])


# ---------------------------------------------------------------- fingerprints and guards


def test_fingerprints_keep_every_arm_apart():
    hashes = {"x": "1"}
    arms = [full_config(), config_for("full", "idm", 1), config_for("full", "idm", 3), config_for("full", "state", 1), config_for("full", "state", 3)]
    assert len({runner.fingerprint_of(hashes, cfg) for cfg in arms}) == len(arms)
    assert full_config().fingerprint_view()["action_repeat"] == 1


def test_repeat_3_changes_the_fingerprint_on_its_own():
    a, b = full_config(), full_config()
    b.action_repeat = 3
    assert runner.fingerprint_of({"x": "1"}, a) != runner.fingerprint_of({"x": "1"}, b)


@pytest.mark.parametrize("variant,repeat", [("idm", 1), ("idm", 3), ("state", 3)])
def test_repair_with_a_repeat_stays_on_the_pilot_seed(monkeypatch, tmp_path, variant, repeat):
    code, jobs = fake_runner(monkeypatch, tmp_path, ["--repair-variant", variant, "--action-repeat", str(repeat), "--families", *FAMILIES])
    assert code == 0
    assert {kind_of(job.argv) for job in jobs} == {"preflight", "pilot"}
    assert all(seed_of(job.argv) in (None, PILOT_SEED) for job in jobs)
    assert sorted(job.name for job in jobs if job.name.startswith("pilot_")) == sorted(f"pilot_{f}_r0" for f in FAMILIES)
    for job in jobs:
        assert job.argv[job.argv.index("--action-repeat") + 1] == str(repeat)
        assert job.argv[job.argv.index("--repair-variant") + 1] == variant


@pytest.mark.parametrize("argv", [["--action-repeat", "3"], ["--repair-variant", "idm", "--action-repeat", "2"], ["--repair-variant", "idm", "--action-repeat", "3", "--seeds", "10"]])
def test_runner_refuses(argv):
    with pytest.raises(SystemExit):
        runner.main(argv)


def test_config_refuses_another_repeat():
    with pytest.raises(ValueError):
        config_for("full", "idm", 2)


def test_worker_refuses_a_repeat_without_a_variant(tmp_path):
    with pytest.raises(SystemExit):
        thread_p.main(["--kind", "pilot", "--mode", "smoke", "--family", "pendulum", "--seed", str(PILOT_SEED), "--action-repeat", "3", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("seed", sorted(REGISTERED_SEEDS))
def test_worker_refuses_a_registered_seed_with_a_repeat(seed, tmp_path):
    with pytest.raises(SystemExit):
        thread_p.main(["--kind", "pilot", "--mode", "smoke", "--family", "pendulum", "--seed", str(seed), "--repair-variant", "idm", "--action-repeat", "3", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("kind", ["main", "h5", "h14", "h16", "h17", "bench", "summarize"])
def test_worker_refuses_any_other_job_kind_with_a_repeat(kind, tmp_path):
    with pytest.raises(SystemExit):
        thread_p.main(["--kind", kind, "--mode", "smoke", "--family", "pendulum", "--seed", str(PILOT_SEED), "--repair-variant", "state", "--action-repeat", "3", "--run-dir", str(tmp_path), "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("family,variant", [("pendulum", "idm"), ("drone", "state")])
def test_repeat_smoke_through_the_real_runner(tmp_path, family, variant):
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run(
        [sys.executable, "-m", "intr.pixel.runner", "--smoke", "--skip-tests", "--no-publish", "--families", family, "--repair-variant", variant, "--action-repeat", "3", "--out", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    assert "No registered seed was started" in done.stdout
    run = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert not list((run / "jobs").glob("*"))
    record = json.loads((run / "pilot" / f"pilot_{family}_r0.json").read_text())
    assert record["status"] == "ok" and record["seed"] == PILOT_SEED and record["action_repeat"] == 3 and record["repair_variant"] == variant
    assert record["noise_floor"]["ratio"] >= 0.0
    assert len(record["noise_floor"]["ratio_per_dim"]) == len(record["noise_floor"]["delta_variance_per_dim"])
    tiny = json.loads((run / "preflight" / "tiny_0.json").read_text())
    assert tiny["kind"] == "pilot" and tiny["action_repeat"] == 3
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["config"]["action_repeat"] == 3
