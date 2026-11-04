"""Encoder, probe boundaries, RLS on embeddings against batch ridge, planner parity with eval.py, and causality."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import thread_f  # noqa: E402
import thread_g  # noqa: E402
import thread_p  # noqa: E402
from intr import ebayes  # noqa: E402
from intr.adapt import AdaptedPredictor, adapt  # noqa: E402
from intr.envs import FAMILIES, Transitions, collect_trajectories, make_env, make_generator  # noqa: E402
from intr.eval import CEMConfig, OraclePredictor, cem_plan, closed_loop, collect_holdout, compute_obs_scale, open_loop_errors  # noqa: E402
from intr.models import DynamicsModel, ModelConfig, clip_obs  # noqa: E402
from intr.pixel.config import CORE_FULL, NO_WRITE_LAMBDA, JepaSettings, Rung, smoke_config  # noqa: E402
from intr.pixel.design import dopt_scores, sample_action_sequences  # noqa: E402
from intr.pixel.planning import EvalContext, closed_loop_model, closed_loop_oracle, decoded_errors, make_windows, plan  # noqa: E402
from intr.pixel.render import render  # noqa: E402
from intr.pixel.sim import collect_states, env_on  # noqa: E402
from intr.pixel.world import (  # noqa: E402
    ConvEncoder,
    ProbeLeakError,
    backbone_from_dict,
    backbone_to_dict,
    embed_frames,
    embedding_bounds,
    fit_probe,
    on_device,
    train_jepa,
    var_cov_terms,
)
from intr.train import Backbone  # noqa: E402

CPU = torch.device("cpu")
CEM = CEMConfig(horizon=3, n_candidates=8, n_elites=2, n_iters=2)


def tiny_frames(family="pendulum", stack=2, T=8, size=16):
    env = make_env(family, 3, "train", 0)
    traj = collect_states(env, env_on(env, CPU), 2, T, env.default_probe, make_generator("f"), stack - 1, CPU)
    return render(family, traj.states, size), traj


# ---------------------------------------------------------------- encoder


def test_encoder_output_does_not_depend_on_the_batch():
    encoder = ConvEncoder(6, 16, 5, 8).eval()
    x = torch.rand(4, 6, 16, 16)
    with torch.no_grad():
        assert torch.allclose(encoder(x)[:1], encoder(x[:1]), atol=1e-6)


@pytest.mark.parametrize("size", [16, 24, 32, 48, 64])
def test_encoder_sizes(size):
    assert ConvEncoder(6, size, 7, 8)(torch.rand(2, 6, size, size)).shape == (2, 7)


def test_full_rung_parameter_counts_are_small():
    for rung, D in ((Rung(32), 16), (Rung(48), 32), (Rung(64), 32), (Rung(64, 48, 2), 32)):
        count = sum(p.numel() for p in ConvEncoder(6, rung.size, D, rung.width).parameters())
        assert 100_000 < count < 1_000_000


def test_anti_collapse_terms_see_collapse():
    collapsed = torch.ones(64, 8)
    variance, _ = var_cov_terms(collapsed)
    assert variance.item() > 0.9
    healthy = torch.randn(4096, 8, generator=make_generator("h"))
    variance, covariance = var_cov_terms(healthy)
    assert variance.item() < 0.05 and covariance.item() < 0.05
    duplicated = healthy[:, :1].expand(-1, 8)
    assert var_cov_terms(duplicated)[1].item() > 1.0


def test_jepa_runs_and_is_finite_and_seeded():
    frames, traj = tiny_frames()
    s = JepaSettings(steps=6, batch=8, predictor_hidden=16)
    a, stats = train_jepa(frames, traj.act, 2, 6, Rung(16, 8), s, 0, CPU)
    b, _ = train_jepa(frames, traj.act, 2, 6, Rung(16, 8), s, 0, CPU)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not any(p.requires_grad for p in a.parameters())
    assert stats["amp"] is False and stats["log"]


def test_embed_frames_alignment_and_chunking():
    frames, traj = tiny_frames(T=6)
    encoder = ConvEncoder(6, 16, 4, 8).eval()
    a = embed_frames(encoder, frames, 2, chunk=2)
    b = embed_frames(encoder, frames, 2, chunk=4096)
    assert a.shape == (3, 2, 7, 4) and torch.allclose(a, b, atol=1e-5)
    window = (frames[1, 0, 3:5].flatten(0, 1)[None].float() / 255.0)
    with torch.no_grad():
        assert torch.allclose(encoder(window)[0], a[1, 0, 3], atol=1e-5)


# ---------------------------------------------------------------- probe


def test_probe_refuses_any_tier_but_train():
    emb, obs = torch.randn(32, 4), torch.randn(32, 3)
    for tier in ("val_in", "interp", "extrap", "oof"):
        with pytest.raises(ProbeLeakError):
            fit_probe(emb, obs, tier, torch.ones(3), 8, 2, 4, 0, CPU)
    _, stats = fit_probe(emb, obs, "train", torch.ones(3), 8, 2, 4, 0, CPU)
    assert stats["source_tiers"] == ["train"]


def test_probe_learns_and_touches_no_encoder():
    frames, traj = tiny_frames(T=8)
    encoder = ConvEncoder(6, 16, 6, 8).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    emb = embed_frames(encoder, frames, 2)
    obs = traj.obs
    probe, _ = fit_probe(emb.reshape(-1, 6), obs.reshape(-1, 3), "train", torch.ones(3), 32, 300, 32, 0, CPU)
    start_loss = ((obs - obs.mean((0, 1, 2))) ** 2).mean()
    assert ((probe(emb) - obs) ** 2).mean() < start_loss
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())
    assert not any(p.requires_grad for p in probe.parameters())


# ---------------------------------------------------------------- RLS on embeddings


def embedding_core(D=6, A=1, z=0, F=8):
    torch.manual_seed(0)
    model = DynamicsModel(ModelConfig(D, A, z, 16, 2, F))
    return Backbone(model, "nolatent").freeze()


def embedding_transitions(E=3, N=12, D=6, A=1):
    gen = make_generator("emb")
    obs = torch.randn(E, N, D, generator=gen)
    act = torch.rand(E, N, A, generator=gen) * 2 - 1
    next_obs = obs + 0.1 * torch.randn(E, N, D, generator=gen)
    return Transitions(obs, act, next_obs)


@pytest.mark.parametrize("lam", [1e-3, 1.0, 100.0])
def test_rls_on_embeddings_matches_batch_ridge(lam):
    backbone = embedding_core()
    data = embedding_transitions()
    predictor = adapt("rls", backbone, data, lambda_=lam).predictor
    with torch.no_grad():
        phi = backbone.model.features(data.obs, data.act).double()
    v = backbone.model.target(data.obs, data.next_obs).double()
    W0 = backbone.model.W.detach().double()
    for e in range(3):
        A = phi[e].t() @ phi[e] + lam * torch.eye(8, dtype=torch.float64)
        W = W0 + torch.linalg.solve(A, phi[e].t() @ (v[e] - phi[e] @ W0.t())).t()
        assert torch.allclose(predictor.rls.W[e], W, atol=1e-6)


def test_evidence_write_runs_on_embeddings_and_can_refuse():
    backbone = embedding_core()
    result = adapt("ebrls", backbone, embedding_transitions())
    summary = thread_p.compact_info(result.info)["eb_summary"]
    assert summary["median_edf"] >= 0.0
    assert ebayes.NO_WRITE_LAMBDA == NO_WRITE_LAMBDA


def test_on_device_predictor_equals_the_adapted_one():
    backbone = embedding_core()
    data = embedding_transitions()
    adapted = adapt("rls", backbone, data, lambda_=0.1).predictor
    moved = on_device(adapted, backbone.model, CPU)
    assert torch.equal(adapted.step(data.obs, data.act), moved.step(data.obs, data.act))


def test_backbone_round_trip():
    backbone = embedding_core()
    copy = backbone_from_dict(backbone_to_dict(backbone))
    data = embedding_transitions()
    assert torch.equal(backbone.model(data.obs, data.act), copy.model(data.obs, data.act))
    assert not any(p.requires_grad for p in copy.model.parameters())


def test_config_core_matches_thread_g_capacity_arm():
    arm = {a.name: a for a in thread_g.ARMS}["capacity_updates"]
    cfg = thread_g.full_config()
    tcfg = thread_g.arm_train_config(arm, cfg)
    core = CORE_FULL["puck"]
    assert (core.model_steps, core.warmup_steps, core.block_steps, core.latent_block_steps) == (tcfg.model_steps, tcfg.warmup_steps, tcfg.block_steps, tcfg.latent_block_steps)
    model = thread_g.arm_model_config(arm, make_env("puck", 1, "train", "b"), cfg, 8)
    assert (core.hidden, core.n_hidden, core.feature_dim) == (model.hidden, model.n_hidden, model.feature_dim)


# ---------------------------------------------------------------- open loop and planner parity


@pytest.mark.parametrize("family", FAMILIES)
def test_decoded_errors_with_identity_probe_equal_open_loop_errors(family):
    env = make_env(family, 3, "interp", 0)
    holdout = collect_holdout(env, 2, 60, make_generator("h", family))
    scale = compute_obs_scale(family)
    oracle = OraclePredictor(env)
    original = open_loop_errors(oracle, holdout, scale, env.obs_low, env.obs_high)
    ctx = EvalContext(torch.nn.Identity(), scale, env.obs_low, env.obs_high, env.obs_low, env.obs_high)
    windows = make_windows(holdout.obs, holdout.obs, holdout.act, CPU)
    mine = decoded_errors(oracle, windows, ctx, (1, 10, 30))
    for h in (1, 10, 30):
        assert torch.allclose(original[h], mine[h], atol=1e-6)


def test_imagination_reads_no_later_embedding():
    """Poison every embedding that is not a window start. Rollouts start from the starts and read actions only."""
    env = make_env("pendulum", 2, "interp", 0)
    holdout = collect_holdout(env, 2, 60, make_generator("p"))
    emb = holdout.obs.clone()
    backbone = embedding_core(D=3)
    frozen = AdaptedPredictor(backbone.model)
    scale = compute_obs_scale("pendulum")
    ctx = EvalContext(torch.nn.Identity(), scale, env.obs_low, env.obs_high, env.obs_low, env.obs_high)
    clean = decoded_errors(frozen, make_windows(emb, holdout.obs, holdout.act, CPU), ctx)
    poisoned = emb.clone()
    keep = torch.zeros(emb.shape[2], dtype=torch.bool)
    keep[[0, 10, 20, 30]] = True
    poisoned[:, :, ~keep] = 1e6
    again = decoded_errors(frozen, make_windows(poisoned, holdout.obs, holdout.act, CPU), ctx)
    for h in clean:
        assert torch.equal(clean[h], again[h])


@pytest.mark.parametrize("family", FAMILIES)
def test_plan_matches_cem_plan(family):
    env = make_env(family, 3, "interp", 0)
    task = env.sample_task(make_generator("t", family))
    o = env.obs(task.state)[:, 0]
    mean = torch.zeros(3, CEM.horizon, env.act_dim)
    oracle = OraclePredictor(env)
    a1, m1 = cem_plan(oracle, env, o, task.goal, mean, CEM, make_generator("g"))
    a2, m2 = plan(o, lambda x, a: clip_obs(oracle.step(x, a), env.obs_low, env.obs_high), lambda x, a: env.cost(x, a, task.goal), mean, CEM, make_generator("g"))
    assert torch.equal(a1, a2) and torch.equal(m1, m2)


@pytest.mark.parametrize("family", FAMILIES)
def test_oracle_episode_matches_closed_loop(family):
    env = make_env(family, 3, "interp", 0)
    task = env.sample_task(make_generator("t", family))
    original = closed_loop(env, OraclePredictor(env), task, CEM, make_generator("g"), 4)
    mine = closed_loop_oracle(env_on(env, CPU), task, CEM, make_generator("g"), 4, 0)
    assert torch.equal(original["success"], mine["success"])
    assert torch.allclose(original["mean_cost"], mine["mean_cost"])


def _pixel_setup(family="pendulum"):
    env = make_env(family, 2, "interp", 0)
    D = 5
    model = DynamicsModel(ModelConfig(D, env.act_dim, 0, 8, 1, 4)).requires_grad_(False)
    encoder = ConvEncoder(6, 16, D, 8).eval().requires_grad_(False)
    from intr.pixel.world import StateProbe

    probe = StateProbe(D, env.obs_dim, 8).requires_grad_(False)
    low, high = embedding_bounds(torch.randn(50, D, generator=make_generator("b")))
    ctx = EvalContext(probe, compute_obs_scale(family), env.obs_low, env.obs_high, low, high)
    return env, AdaptedPredictor(model), encoder, ctx


def test_closed_loop_renders_one_image_per_real_step_and_no_future():
    env, predictor, encoder, ctx = _pixel_setup()
    task = env.sample_task(make_generator("t"))
    dev = env_on(env, CPU)
    calls = []

    def counting(*a, **k):
        calls.append(a[1].clone())
        return render(*a, **k)

    short = closed_loop_model(dev, predictor, ctx, "pendulum", 16, 2, encoder, task, CEM, make_generator("g"), 3, None, counting)
    assert len(calls) == 1 + 3  # the burn-in frame, then one per planning step
    long = closed_loop_model(dev, predictor, ctx, "pendulum", 16, 2, encoder, task, CEM, make_generator("g"), 6, None, render)
    assert torch.equal(short["actions"], long["actions"][:, :3])  # later images cannot change earlier actions


def test_closed_loop_image_palette_reaches_the_camera():
    env, predictor, encoder, ctx = _pixel_setup("puck")
    task = env.sample_task(make_generator("t"))
    dev = env_on(env, CPU)
    from intr.pixel.render import appearance_palette

    palette = appearance_palette("puck", 2, make_generator("a"))
    seen = []

    def spy(family, state, size, given):
        seen.append(given)
        return render(family, state, size, given)

    closed_loop_model(dev, predictor, ctx, "puck", 16, 2, encoder, task, CEM, make_generator("g"), 3, palette, spy)
    assert len(seen) == 4 and all(given is palette for given in seen)


# ---------------------------------------------------------------- D-optimal design, same data


def test_dopt_scores_match_thread_f():
    backbone = embedding_core(D=7, A=2)
    env_like = AdaptedPredictor(backbone.model, rls=ebayes.RLSState(backbone.model.W.detach()[None].expand(3, -1, -1), 1.0))
    s = torch.randn(3, 7, generator=make_generator("s"))
    seqs = sample_action_sequences(3, 6, 3, 2, make_generator("q"))
    assert torch.equal(seqs, thread_f.sample_action_sequences(3, 6, 3, 2, make_generator("q")))
    low, high = -100 * torch.ones(7), 100 * torch.ones(7)
    assert torch.equal(dopt_scores(env_like, s, seqs, low, high), thread_f.dopt_scores(env_like, s, seqs, low, high))


def test_k1_data_is_a_prefix_and_does_not_depend_on_the_list_of_k():
    """Problem 12: the draw was a hidden noise source. Here K only cuts one fixed draw."""
    family = "pendulum"
    env = make_env(family, 3, "interp", 0)
    a_cfg, b_cfg = smoke_config(), smoke_config()
    a_cfg.ks, b_cfg.ks = [1], [1, 3]
    tag = ("test", family, 10, "interp")
    _, _, probes_a = thread_p.collect_tier(env, family, tag, a_cfg, CPU)
    _, _, probes_b = thread_p.collect_tier(env, family, tag, b_cfg, CPU)
    assert torch.equal(probes_a.obs, probes_b.obs) and probes_a.obs.shape[1] == 3
    plain = collect_trajectories(env, 1, 5, env.default_probe, make_generator(tag, "probe"))
    assert plain.obs.shape[1] == 1  # what a config-dependent draw would have done: a different shape, so different numbers


def test_edges_are_reported():
    grid = [{"lambda_": 1e-3}, {"lambda_": 1.0}, {"lambda_": NO_WRITE_LAMBDA}]
    assert thread_p.edges_of(grid, {"lambda_": 1e-3}) == {"lambda_": "low"}
    assert thread_p.edges_of(grid, {"lambda_": 1.0}) == {}
    assert thread_p.edges_of(grid, {"lambda_": NO_WRITE_LAMBDA}) == {"lambda_": "no_write"}
