"""The encoder repair: new terms are off by default, align with the data, stay out of what is kept, and cannot reach a registered seed."""

import copy
import inspect
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import thread_p  # noqa: E402
from intr.envs import FAMILIES, derive_seed, make_env, make_generator  # noqa: E402
from intr.pixel import runner, world  # noqa: E402
from intr.pixel.config import EMBED_DIMS, PILOT_SEED, REGISTERED_SEEDS, JepaSettings, Rung, config_for, full_config  # noqa: E402
from intr.pixel.render import render, stack_windows  # noqa: E402
from intr.pixel.scheduler import RunReport, atomic_write_json  # noqa: E402
from intr.pixel.sim import collect_mixed_states, collect_states, env_on  # noqa: E402
from intr.pixel.world import ConvEncoder, InverseDynamicsHead, JepaPredictor, StateAuxHead, jepa_loss, train_jepa, var_cov_terms  # noqa: E402
from intr.train import _cosine_lr, _set_lr  # noqa: E402

CPU = torch.device("cpu")
STACK, EMBED, SIZE = 2, 6, 16
RUNG = Rung(SIZE, 8)
SETTINGS = JepaSettings(steps=4, batch=8, predictor_hidden=16)


def tiny_data(family="pendulum", T=8):
    env = make_env(family, 3, "train", 0)
    traj = collect_states(env, env_on(env, CPU), 2, T, env.default_probe, make_generator("repair", family), STACK - 1, CPU)
    return env, traj, render(family, traj.states, SIZE)


def nets(act_dim=1):
    torch.manual_seed(0)
    online = ConvEncoder(3 * STACK, SIZE, EMBED, RUNG.width)
    predictor = JepaPredictor(EMBED, act_dim, SETTINGS.predictor_hidden)
    return online, copy.deepcopy(online).requires_grad_(False), predictor


def one_batch(H=3):
    _, traj, frames = tiny_data()
    return frames.flatten(0, 1)[:, : H + STACK], traj.act.flatten(0, 1)[:, :H], traj.obs.flatten(0, 1)[:, 0], traj.obs


def heads(train_obs, act_dim=1):
    torch.manual_seed(1)
    return InverseDynamicsHead(EMBED, act_dim, 16), StateAuxHead(EMBED, train_obs, 16)


# ---------------------------------------------------------------- defaults are today's objective


def todays_loss(online, target, predictor, frames, actions, stack, s):
    """jepa_loss as it stood at the pilot, copied here so a change to it shows."""
    b, H = actions.shape[0], actions.shape[1]
    windows = torch.stack([frames[:, i : i + stack] for i in range(1, H + 1)], dim=1)
    with torch.no_grad():
        goal = target(stack_windows(windows.flatten(0, 1))).reshape(b, H, -1).float()
    s0 = online(stack_windows(frames[:, :stack]))
    current = s0
    sim = 0.0
    for i in range(H):
        current = predictor(current, actions[:, i])
        sim = sim + ((current.float() - goal[:, i]) ** 2).mean()
    sim = sim / H
    variance, covariance = var_cov_terms(s0.float())
    return s.sim_weight * sim + s.var_weight * variance + s.cov_weight * covariance


def test_default_loss_is_exactly_todays_and_logs_no_new_term():
    frames, actions, _, _ = one_batch()
    online, target, predictor = nets()
    mine, parts = jepa_loss(online, target, predictor, frames, actions, STACK, SETTINGS, "cpu", False)
    assert torch.equal(mine, todays_loss(online, target, predictor, frames, actions, STACK, SETTINGS))
    assert set(parts) == {"sim", "var", "cov"}


def test_default_training_is_exactly_todays():
    """The same weights after a few steps as the loop at the pilot, with obs given or not."""
    _, traj, frames = tiny_data()
    s = SETTINGS
    E, B = frames.shape[:2]
    T = traj.act.shape[2]
    H = min(s.rollout, T)
    torch.manual_seed(derive_seed("init", 0, "jepa"))
    online = ConvEncoder(3 * STACK, SIZE, EMBED, RUNG.width)
    predictor = JepaPredictor(EMBED, traj.act.shape[-1], s.predictor_hidden)
    target = copy.deepcopy(online).requires_grad_(False)
    params = list(online.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=s.lr)
    gen = make_generator("train", 0, "jepa")
    for step in range(s.steps):
        _set_lr(optimizer, _cosine_lr(s.lr, step, s.steps, s.lr_floor))
        e, b = torch.randint(E, (s.batch, 1), generator=gen), torch.randint(B, (s.batch, 1), generator=gen)
        t = torch.randint(T - H + 1, (s.batch, 1), generator=gen)
        loss = todays_loss(online, target, predictor, frames[e, b, t + torch.arange(H + STACK)[None]], traj.act[e, b, t + torch.arange(H)[None]], STACK, s)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, s.grad_clip)
        optimizer.step()
        with torch.no_grad():
            for p_target, p_online in zip(target.parameters(), online.parameters()):
                p_target.lerp_(p_online, 1.0 - s.ema)
    for kwargs in ({}, {"obs": traj.obs}):
        mine, stats = train_jepa(frames, traj.act, STACK, EMBED, RUNG, s, 0, CPU, **kwargs)
        assert all(torch.equal(x, y) for x, y in zip(mine.state_dict().values(), target.state_dict().values()))
        assert all(set(row) == {"step", "loss", "sim", "var", "cov"} for row in stats["log"])


def test_full_config_defaults_are_unchanged():
    cfg = full_config()
    assert cfg.jepa.idm_weight == 0.0 and cfg.jepa.state_weight == 0.0
    assert cfg.repair_variant is None and len(cfg.rungs) == 4 and tuple(cfg.seeds) == REGISTERED_SEEDS


# ---------------------------------------------------------------- gradients


def encoder_gradients(s, idm_head=None, state_head=None):
    frames, actions, obs0, _ = one_batch()
    online, target, predictor = nets()
    loss, parts = jepa_loss(online, target, predictor, frames, actions, STACK, s, "cpu", False, idm_head, state_head, obs0)
    loss.backward()
    return [p.grad.clone() for p in online.parameters()], parts


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_each_term_changes_encoder_gradients_only_when_its_weight_is_nonzero():
    train_obs = one_batch()[3]
    base, _ = encoder_gradients(SETTINGS)
    idm_head, state_head = heads(train_obs)
    zero, parts = encoder_gradients(SETTINGS, idm_head, state_head)
    assert same(base, zero) and set(parts) == {"sim", "var", "cov"}
    assert all(p.grad is None for p in list(idm_head.parameters()) + list(state_head.parameters()))

    idm, parts = encoder_gradients(replace(SETTINGS, idm_weight=1.0), idm_head, state_head)
    assert not same(base, idm) and "idm" in parts and "state" not in parts
    assert all(p.grad is not None for p in idm_head.parameters()) and all(p.grad is None for p in state_head.parameters())

    idm_head, state_head = heads(train_obs)
    state, parts = encoder_gradients(replace(SETTINGS, state_weight=1.0), idm_head, state_head)
    assert not same(base, state) and not same(idm, state) and "state" in parts and "idm" not in parts
    assert all(p.grad is not None for p in state_head.parameters()) and all(p.grad is None for p in idm_head.parameters())


def test_the_weight_scales_the_term():
    frames, actions, obs0, train_obs = one_batch()
    idm_head, state_head = heads(train_obs)
    online, target, predictor = nets()
    base, _ = jepa_loss(online, target, predictor, frames, actions, STACK, SETTINGS, "cpu", False)
    both, parts = jepa_loss(online, target, predictor, frames, actions, STACK, replace(SETTINGS, idm_weight=2.0, state_weight=3.0), "cpu", False, idm_head, state_head, obs0)
    assert torch.allclose(both, base + 2.0 * parts["idm"] + 3.0 * parts["state"])


def test_each_term_is_logged_and_recorded_in_training():
    _, traj, frames = tiny_data()
    _, stats = train_jepa(frames, traj.act, STACK, EMBED, RUNG, replace(SETTINGS, idm_weight=1.0, state_weight=1.0), 0, CPU, obs=traj.obs)
    assert all({"idm", "state"} <= set(row) for row in stats["log"])
    assert not any(isinstance(v, nn.Module) for v in stats.values())


def test_variants_start_from_the_same_weights_and_batches():
    """Heads draw after the encoder and predictor, so a difference between arms is the objective and not the start."""
    _, traj, frames = tiny_data()
    seen = {}

    def spy_for(name, real=jepa_loss):
        def spy(online, target, predictor, batch_frames, *rest):
            seen.setdefault(name, (copy.deepcopy(online.state_dict()), copy.deepcopy(predictor.state_dict()), batch_frames.clone()))
            return real(online, target, predictor, batch_frames, *rest)

        return spy

    patch = pytest.MonkeyPatch()
    try:
        for name, s in (("base", SETTINGS), ("idm", replace(SETTINGS, idm_weight=1.0)), ("state", replace(SETTINGS, state_weight=1.0))):
            patch.setattr(world, "jepa_loss", spy_for(name))
            train_jepa(frames, traj.act, STACK, EMBED, RUNG, s, 0, CPU, obs=traj.obs)
    finally:
        patch.undo()
    for name in ("idm", "state"):
        for part in (0, 1):
            assert all(torch.equal(seen["base"][part][k], seen[name][part][k]) for k in seen["base"][part])
        assert torch.equal(seen["base"][2], seen[name][2])


# ---------------------------------------------------------------- alignment, against sim.py's burn-in note


@pytest.mark.parametrize("family", FAMILIES)
def test_obs_t_is_the_state_of_window_ts_last_frame_and_action_t_leads_to_window_t_plus_1s(family):
    env, traj, frames = tiny_data(family)
    env_dev = env_on(env, CPU)
    T = traj.act.shape[2]
    assert frames.shape[2] == STACK + T and traj.obs.shape[2] == T + 1
    for t in range(T):
        last_frame_state = traj.states[:, :, t + STACK - 1]
        assert torch.equal(traj.obs[:, :, t], env.obs(last_frame_state))
        assert torch.equal(frames[:, :, t + STACK - 1], render(family, last_frame_state, SIZE))
        assert torch.allclose(env_dev.step(last_frame_state, traj.act[:, :, t]), traj.states[:, :, t + STACK], atol=1e-12)


def test_training_pairs_each_window_with_its_last_frames_obs():
    """Each frame is painted with its index, and obs holds the index of the frame it belongs to."""
    E, B, T = 2, 2, 8
    index = torch.arange(STACK + T, dtype=torch.uint8)
    frames = index[None, None, :, None, None, None].expand(E, B, STACK + T, 3, SIZE, SIZE).clone()
    act = torch.randn(E, B, T, 1, generator=make_generator("act"))
    obs = (torch.arange(T + 1, dtype=torch.float32) + STACK - 1)[None, None, :, None].expand(E, B, T + 1, 1)
    seen = []
    real = world.jepa_loss

    def spy(online, target, predictor, batch_frames, actions, stack, s, device_type, amp, idm_head, state_head, obs0):
        seen.append((batch_frames.clone(), obs0.clone()))
        return real(online, target, predictor, batch_frames, actions, stack, s, device_type, amp, idm_head, state_head, obs0)

    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(world, "jepa_loss", spy)
        train_jepa(frames, act, STACK, EMBED, RUNG, replace(SETTINGS, state_weight=1.0), 0, CPU, obs=obs)
    finally:
        patch.undo()
    assert seen
    for batch_frames, obs0 in seen:
        assert torch.equal(batch_frames[:, STACK - 1, 0, 0, 0].float(), obs0[:, 0])


class RecordingEncoder(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.calls = inner, []

    def forward(self, x):
        self.calls.append(x.clone())
        return self.inner(x)


class ZeroHead(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        self.act_dim, self.inputs = act_dim, None

    def forward(self, s0, s1):
        self.inputs = (s0, s1)
        return torch.zeros(s0.shape[0], self.act_dim)


def test_inverse_dynamics_reads_windows_0_and_1_and_targets_action_0():
    frames, actions, _, _ = one_batch()
    online, target, predictor = nets()
    recording, head = RecordingEncoder(online), ZeroHead(actions.shape[-1])
    _, parts = jepa_loss(recording, target, predictor, frames, actions, STACK, replace(SETTINGS, idm_weight=1.0), "cpu", False, head)
    assert len(recording.calls) == 2  # exactly one extra online forward
    assert torch.equal(recording.calls[0], stack_windows(frames[:, :STACK]))
    assert torch.equal(recording.calls[1], stack_windows(frames[:, 1 : 1 + STACK]))
    assert torch.allclose(parts["idm"], (actions[:, 0] ** 2).mean())
    assert head.inputs[0].requires_grad and head.inputs[1].requires_grad  # the gradient reaches the encoder through both


def test_no_extra_online_forward_when_the_weight_is_zero():
    frames, actions, _, _ = one_batch()
    online, target, predictor = nets()
    recording = RecordingEncoder(online)
    jepa_loss(recording, target, predictor, frames, actions, STACK, SETTINGS, "cpu", False)
    assert len(recording.calls) == 1


# ---------------------------------------------------------------- state control: observable only, train scale only


@pytest.mark.parametrize("family", FAMILIES)
def test_obs_cannot_read_a_hidden_parameter(family):
    env, traj, _ = tiny_data(family)
    poisoned = copy.copy(env)
    poisoned.params = {k: torch.full_like(v, float("nan")) for k, v in env.params.items()}
    assert torch.equal(poisoned.obs(traj.states), env.obs(traj.states))
    assert torch.isfinite(poisoned.obs(traj.states)).all()
    assert traj.obs.shape[-1] == env.obs_dim


def test_train_jepa_takes_no_environment():
    names = set(inspect.signature(train_jepa).parameters)
    assert names == {"frames", "act", "stack", "embed_dim", "rung", "s", "seed", "device", "amp", "obs"}


def test_state_scale_comes_from_the_observations_given():
    train_obs = torch.randn(4, 3, 9, 5, generator=make_generator("o")) * 3.0 + 1.0
    head = StateAuxHead(EMBED, train_obs, 8)
    flat = train_obs.reshape(-1, 5)
    assert torch.equal(head.obs_mean, flat.mean(0)) and torch.equal(head.obs_std, flat.std(0))
    assert torch.allclose(head.standardize(flat).mean(0), torch.zeros(5), atol=1e-5)
    assert not any(p is head.obs_mean or p is head.obs_std for p in head.parameters())


def test_state_weight_without_observations_is_refused():
    _, traj, frames = tiny_data()
    with pytest.raises(ValueError):
        train_jepa(frames, traj.act, STACK, EMBED, RUNG, replace(SETTINGS, state_weight=1.0), 0, CPU)


def test_bench_works_with_the_state_term():
    assert thread_p.run_bench(config_for("smoke", "state"), 0.01, CPU, True)["jepa_steps_per_s"] > 0


@pytest.mark.parametrize("variant", ["idm", "state"])
def test_world_keeps_no_head_and_the_state_target_is_train_obs(variant, monkeypatch):
    cfg = config_for("smoke", variant)
    family, seed = "pendulum", cfg.pilot_seed
    captured = {}
    real = thread_p.train_jepa

    def boom(*a, **k):
        raise AssertionError("a training head ran after training")

    def spy(*a, **k):
        out = real(*a, **k)
        captured["obs"] = k["obs"].clone()
        monkeypatch.setattr(InverseDynamicsHead, "forward", boom)
        monkeypatch.setattr(StateAuxHead, "forward", boom)
        return out

    monkeypatch.setattr(thread_p, "train_jepa", spy)
    built, _, record = thread_p.build_world(family, seed, cfg, cfg.rungs[0], CPU)

    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    burn = cfg.stack - 1
    traj = collect_mixed_states(env, env_on(env, CPU), cfg.train_traj, cfg.train_steps, env.data_policies, make_generator("data", family, seed), burn, CPU)
    poisoned = copy.copy(env)
    poisoned.params = {k: torch.full_like(v, float("nan")) for k, v in env.params.items()}
    assert torch.equal(captured["obs"], poisoned.obs(traj.states[:, :, burn:]))

    assert type(built.encoder) is ConvEncoder
    plain = ConvEncoder(3 * cfg.stack, cfg.rungs[0].size, EMBED_DIMS[family], cfg.rungs[0].width)
    saved = thread_p.world_to_dict(built, built.ctx.emb_low, built.ctx.emb_high)
    assert set(saved["encoder"]) == set(plain.state_dict())
    assert not any("idm" in key or "state_head" in key or "aux" in key for key in saved)
    assert variant in record["jepa"]["log"][-1]


# ---------------------------------------------------------------- settings and fingerprints


def test_variant_is_the_real_settings_with_one_change():
    for variant, changed in (("idm", "idm_weight"), ("state", "state_weight")):
        mine, base = asdict(config_for("full", variant)), asdict(full_config())
        assert mine["jepa"].pop(changed) == 1.0 and base["jepa"].pop(changed) == 0.0
        assert mine["jepa"] == base["jepa"]
        assert mine["pilot_seed"] == PILOT_SEED == base["pilot_seed"] and mine["pilot_seed"] not in REGISTERED_SEEDS
        assert mine["rungs"] == base["rungs"][:1] and mine["rungs"][0]["size"] == 32
        assert mine["seeds"] == [] and mine["h17_seeds"] == []
        ignored = {"jepa", "rungs", "seeds", "h17_seeds", "repair_variant"}
        assert {k: v for k, v in mine.items() if k not in ignored} == {k: v for k, v in base.items() if k not in ignored}


def test_fingerprints_differ_and_the_weights_alone_change_them():
    hashes = {"x": "1"}
    prints = [runner.fingerprint_of(hashes, cfg) for cfg in (full_config(), config_for("full", "idm"), config_for("full", "state"))]
    assert len(set(prints)) == 3
    for field in ("idm_weight", "state_weight"):
        cfg = full_config()
        cfg.jepa = replace(cfg.jepa, **{field: 1.0})
        assert runner.fingerprint_of(hashes, cfg) != prints[0]
    assert {"idm_weight", "state_weight"} <= set(full_config().fingerprint_view()["jepa"])


# ---------------------------------------------------------------- the repair path cannot start a registered seed


def seed_of(argv):
    return int(argv[argv.index("--seed") + 1]) if "--seed" in argv else None


def kind_of(argv):
    return argv[argv.index("--kind") + 1]


def fake_runner(monkeypatch, tmp_path, argv, gate_pass=False, write=True):
    """The real supervisor on the real settings, with every worker replaced. Returns its exit code and each job it asked for."""
    jobs = []

    def fake_run_jobs(batch, slots, label="jobs", **kwargs):
        jobs.extend(batch)
        if write:
            for job in batch:
                job.output.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(job.output, {"status": "ok", "fingerprint": job.fingerprint, "wall_s": 1.0, "checks": {}, "gate": {"pass": gate_pass, "checks": {"oracle_interp": True}}})
        return RunReport()

    monkeypatch.setattr(runner, "run_jobs", fake_run_jobs)
    monkeypatch.setattr(runner, "discover_gpus", lambda: [])
    monkeypatch.setattr(runner, "scrub_origin", lambda root: None)
    monkeypatch.setattr(runner, "machine_facts", lambda *a: {})
    monkeypatch.setattr(runner, "run_tests", lambda *a: pytest.fail("a repair run must not start the test suite"))
    monkeypatch.setattr(runner, "summarize", lambda *a: None)
    code = runner.main(["--allow-cpu", "--no-publish", "--out", str(tmp_path), *argv])
    return code, jobs


@pytest.mark.parametrize("variant", ["idm", "state"])
@pytest.mark.parametrize("gate_pass", [True, False])
def test_repair_runs_only_the_pilot_at_rung_0_on_the_real_settings(monkeypatch, tmp_path, variant, gate_pass):
    code, jobs = fake_runner(monkeypatch, tmp_path, ["--repair-variant", variant, "--families", *FAMILIES], gate_pass)
    assert code == 0
    pilots = [job for job in jobs if kind_of(job.argv) == "pilot" and job.name.startswith("pilot_")]
    assert sorted(job.name for job in pilots) == sorted(f"pilot_{f}_r0" for f in FAMILIES)
    assert {kind_of(job.argv) for job in jobs} == {"preflight", "pilot"}
    for job in jobs:
        assert seed_of(job.argv) in (None, PILOT_SEED)
    for job in pilots:
        assert job.argv[job.argv.index("--mode") + 1] == "full"
        assert job.argv[job.argv.index("--repair-variant") + 1] == variant


def test_a_crashed_pilot_job_is_an_error_and_starts_nothing_else(monkeypatch, tmp_path):
    code, jobs = fake_runner(monkeypatch, tmp_path, ["--repair-variant", "idm", "--families", "puck"], write=False)
    assert code == 1
    assert {kind_of(job.argv) for job in jobs} == {"preflight", "pilot"}
    assert all(seed_of(job.argv) in (None, PILOT_SEED) for job in jobs)


def test_the_ordinary_path_still_goes_on_to_registered_seeds(monkeypatch, tmp_path):
    """The check above could pass by accident if the fake runner never reached a main job."""
    code, jobs = fake_runner(monkeypatch, tmp_path, ["--skip-tests", "--families", "pendulum", "--seeds", "10"], gate_pass=True)
    assert code == 0
    assert any(kind_of(job.argv) == "main" and seed_of(job.argv) == 10 for job in jobs)


def test_runner_refuses_seeds_with_a_variant():
    with pytest.raises(SystemExit):
        runner.main(["--repair-variant", "idm", "--seeds", "10"])


@pytest.mark.parametrize("seed", sorted(REGISTERED_SEEDS))
def test_worker_refuses_a_registered_seed_with_a_variant(seed, tmp_path):
    with pytest.raises(SystemExit):
        thread_p.main(["--kind", "pilot", "--mode", "smoke", "--family", "pendulum", "--seed", str(seed), "--repair-variant", "idm", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("kind", ["main", "h5", "h14", "h16", "h17", "bench"])
def test_worker_refuses_any_other_job_kind_with_a_variant(kind, tmp_path):
    with pytest.raises(SystemExit):
        thread_p.main(["--kind", kind, "--mode", "smoke", "--family", "pendulum", "--seed", str(PILOT_SEED), "--repair-variant", "state", "--run-dir", str(tmp_path), "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("family,variant", [("pendulum", "idm"), ("puck", "state")])
def test_repair_smoke_through_the_real_runner(tmp_path, family, variant):
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = ""
    done = subprocess.run(
        [sys.executable, "-m", "intr.pixel.runner", "--smoke", "--skip-tests", "--no-publish", "--families", family, "--repair-variant", variant, "--out", str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    assert "No registered seed was started" in done.stdout
    runs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    assert not list((run / "jobs").glob("*")), "a stage after the pilot started"
    assert not (run / "checkpoints").exists() and not (run / "cache").exists()
    assert not list((run / "preflight").rglob("checkpoints"))
    for log in (run / "logs").glob("*.log"):
        assert not log.name.startswith(("main_", "h5_", "h14_", "h16_", "h17_"))

    record = json.loads((run / "pilot" / f"pilot_{family}_r0.json").read_text())
    assert record["status"] == "ok" and record["seed"] == PILOT_SEED and record["repair_variant"] == variant
    assert {"frozen_closed_loop_interp", "oracle_interp", "latent_reduction", "checks", "pass"} <= set(record["gate"])
    assert len(record["collapse"]["r2"]["per_dim"]) == make_env(family, 1, "train", 0).obs_dim
    assert record["train"]["latent_r2"] and {"latent", "nolatent"} <= set(record["train"]["exact_train_loss"])
    assert variant in record["train"]["jepa"]["log"][-1]
    assert record["jepa_settings"][f"{variant}_weight"] == 1.0

    tiny = json.loads((run / "preflight" / "tiny_0.json").read_text())
    assert tiny["kind"] == "pilot" and tiny["seed"] == PILOT_SEED
