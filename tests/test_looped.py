import inspect
import math

import numpy as np
import pytest
import torch

from intr.envs import PuckEnv, Trajectories, collect_mixed, make_env, make_generator
from intr.eval import PersistencePredictor, collect_holdout, compute_obs_scale, open_loop_errors, paired_comparison
from intr.looped import (
    TWO_X_THRESHOLD,
    GaussianResidualModel,
    LoopConfig,
    LoopedDynamics,
    LoopedPredictor,
    LoopTrainConfig,
    MemorylessPredictor,
    PlanningPredictor,
    SequenceData,
    auroc,
    auroc_groups,
    check_ablations,
    collect_contact_trajectories,
    collect_switching,
    contact_masks,
    draw_pre_calls,
    h7_outcome,
    h8_outcome,
    h9_target_passes,
    loop_relation,
    make_windows,
    mlp_macs,
    mlp_matching_macs,
    mlp_matching_params,
    mlp_param_count,
    replay_contact_flags,
    rollout_grid,
    rollout_metrics,
    shift_auroc,
    step_with_contact,
    teacher_forced_trace,
    to_signal_time,
    train_looped,
    window_contact_mask,
)
from intr.models import DynamicsModel, ModelConfig

ARMS = {
    "full": {},
    "no_reinject": {"reinject_input": False, "action_every_loop": False},
    "no_norm": {"normalize": False},
    "no_mixed": {"mixed_depth": False},
    "plain": {"reinject_input": False, "action_every_loop": False, "normalize": False, "mixed_depth": False},
    "prelude_only": {"action_every_loop": False},
    "reinit": {"carry": False},
}


def tiny_cfg(**overrides) -> LoopConfig:
    settings = dict(obs_dim=3, act_dim=1, z_dim=12, emb_dim=6, hidden=16, loops=4, grad_loops=2)
    settings.update(overrides)
    return LoopConfig(**settings)


def tiny_model(seed=0, **overrides) -> LoopedDynamics:
    torch.manual_seed(seed)
    return LoopedDynamics(tiny_cfg(**overrides))


def same_weights(base: LoopedDynamics, **overrides) -> LoopedDynamics:
    model = LoopedDynamics(tiny_cfg(**overrides))
    model.load_state_dict(base.state_dict())
    return model


def inputs(n=8, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, generator=gen), 2.0 * torch.rand(n, 1, generator=gen) - 1.0


def tiny_sequences(family="pendulum", n_env=2, n_traj=2, n_steps=8):
    env = make_env(family, n_env, "train", 0)
    trajs = collect_mixed(env, n_traj, n_steps, ("smooth",), make_generator("test_looped", family))
    return env, SequenceData.from_trajectories(trajs)


# ------------------------------------------------------------ normalization


@pytest.mark.parametrize("loops", [1, 4, 64, 256])
def test_normalization_holds_state_rms(loops):
    model = tiny_model(normalize=True, rms_target=0.85)
    o, a = inputs()
    z_prev = 50.0 * torch.randn(8, 12)
    with torch.no_grad():
        z0, inj = model.begin_step(o, a, z_prev)
        z = model.solve(z0, inj, loops)
    expected = torch.full((8,), 0.85)
    assert torch.allclose(z0.pow(2).mean(-1).sqrt(), expected, atol=1e-4)
    assert torch.allclose(z.pow(2).mean(-1).sqrt(), expected, atol=1e-4)


def test_without_normalization_rms_is_free():
    base = tiny_model(normalize=True)
    model = same_weights(base, normalize=False)
    o, a = inputs()
    with torch.no_grad():
        z0, inj = model.begin_step(o, a, None)
        z = model.solve(z0, inj, 16)
    rms = z.pow(2).mean(-1).sqrt()
    assert (rms - 0.85).abs().max() > 1e-2


def test_rescale_keeps_a_zero_state_finite():
    model = tiny_model(normalize=True)
    out = model.rescale(torch.zeros(2, 12))
    assert torch.isfinite(out).all() and (out == 0).all()


# ------------------------------------------------------------ switches


def test_arms_have_identical_parameter_count_and_all_differ():
    counts = {name: sum(p.numel() for p in LoopedDynamics(tiny_cfg(**o)).parameters()) for name, o in ARMS.items()}
    assert len(set(counts.values())) == 1
    record = check_ablations(tiny_cfg(), ARMS)
    assert set(record) == set(ARMS) - {"full"}
    assert record["no_norm"]["max_output_diff"] > 1e-6
    assert record["no_mixed"]["schedule_differs"] and record["no_mixed"]["max_output_diff"] == 0.0


def test_check_ablations_rejects_a_no_op():
    with pytest.raises(RuntimeError):
        check_ablations(tiny_cfg(), {"fake": {"rms_target": 0.85}})


def test_input_reinjection_switch_controls_what_the_loop_sees():
    on = tiny_model(reinject_input=True, action_every_loop=False)
    off = same_weights(on, reinject_input=False, action_every_loop=False)
    o, a = inputs()
    other_o, _ = inputs(seed=2)
    with torch.no_grad():
        z0, _ = on.begin_step(o, a)
        e_a = on.enc_action(a)
        e1, e2 = on.enc_state(on.norm_obs(o)), on.enc_state(on.norm_obs(other_o))
        # Same z^0, different state embedding: only the re-injecting arm can notice.
        changed_on = (on.solve(z0, on.injection(e1, e_a), 4) - on.solve(z0, on.injection(e2, e_a), 4)).abs().max()
        changed_off = (off.solve(z0, off.injection(e1, e_a), 4) - off.solve(z0, off.injection(e2, e_a), 4)).abs().max()
    assert changed_on > 1e-5
    assert changed_off == 0.0


def test_prelude_only_action_cannot_reach_the_loop_but_reaches_the_output():
    every = tiny_model(action_every_loop=True)
    prelude = same_weights(every, action_every_loop=False)
    o, a = inputs()
    a2 = -a
    with torch.no_grad():
        z0, _ = every.begin_step(o, a)
        e_s = every.enc_state(every.norm_obs(o))
        e1, e2 = every.enc_action(a), every.enc_action(a2)
        changed_every = (every.solve(z0, every.injection(e_s, e1), 4) - every.solve(z0, every.injection(e_s, e2), 4)).abs().max()
        changed_prelude = (prelude.solve(z0, prelude.injection(e_s, e1), 4) - prelude.solve(z0, prelude.injection(e_s, e2), 4)).abs().max()
        through_prelude = (prelude.step(o, a)[0] - prelude.step(o, a2)[0]).abs().max()
    assert changed_every > 1e-5
    assert changed_prelude == 0.0
    assert through_prelude > 1e-5  # the action still matters, via z^0


def test_depth_schedules():
    mixed, fixed = tiny_cfg(loops=8, grad_loops=4), tiny_cfg(loops=8, grad_loops=4, mixed_depth=False)
    assert mixed.pre_call_range() == (0, 8) and mixed.trained_depth_range() == (4, 12)
    assert fixed.pre_call_range() == (4, 4) and fixed.trained_depth_range() == (8, 8)
    gen = torch.Generator().manual_seed(0)
    draws = [draw_pre_calls(mixed, gen) for _ in range(2000)]
    assert set(draws) == set(range(9))
    assert np.mean(draws) == pytest.approx(4.0, abs=0.3)  # mean total depth equals K in both arms
    assert {draw_pre_calls(fixed, gen) for _ in range(20)} == {4}
    with pytest.raises(ValueError):
        tiny_cfg(loops=2, grad_loops=3)


def test_gradient_depth_is_bounded_by_grad_loops():
    model = tiny_model(loops=8, grad_loops=2)
    enabled = []
    model.update.register_forward_hook(lambda module, args, out: enabled.append(torch.is_grad_enabled()))
    o, a = inputs()
    pred, z = model.training_step(o, a, None, pre_calls=5)
    assert enabled.count(False) == 5 and enabled.count(True) == 2
    assert z.requires_grad and pred.requires_grad


@pytest.mark.parametrize("arm", ["no_reinject", "prelude_only", "plain"])
def test_encoders_get_gradient_through_no_grad_calls(arm):
    # Without the pass-through these arms would train their action encoder only when r = 0.
    model = tiny_model(**ARMS[arm])
    o, a = inputs()
    pred, _ = model.training_step(o, a, None, pre_calls=3)
    pred.pow(2).mean().backward()
    assert model.enc_action[0].weight.grad.abs().max() > 0
    assert model.prelude_net[0].weight.grad.abs().max() > 0


def test_pass_through_does_not_change_the_forward_value():
    model = tiny_model(loops=6, grad_loops=2)
    o, a = inputs()
    pred, _ = model.training_step(o, a, None, pre_calls=4)
    with torch.no_grad():
        z0, inj = model.begin_step(o, a)
        direct = model.decode(model.solve(z0, inj, 6))
    assert torch.allclose(pred, direct, atol=1e-5)


# ------------------------------------------------------------ stationarity


def test_update_does_not_depend_on_loop_index():
    model = tiny_model()
    o, a = inputs()
    assert list(inspect.signature(model.cell).parameters) == ["z", "inj"]
    with torch.no_grad():
        z0, inj = model.begin_step(o, a)
        # Splitting a solve anywhere gives the same result, which index conditioning would break.
        whole = model.solve(z0, inj, 9)
        split = model.solve(model.solve(z0, inj, 4), inj, 5)
        # An update applied at call 7 equals one applied to the same state at call 1.
        z6 = model.solve(z0, inj, 6)
        z7 = model.solve(z0, inj, 7)
    assert torch.equal(whole, split)
    assert torch.equal(model.cell(z6, inj), z7)
    # Every call uses one set of weights.
    names = [n for n, _ in model.named_parameters() if n.startswith("update.")]
    assert len(names) == 4


def test_solve_traced_matches_solve_and_defines_the_residual():
    model = tiny_model()
    o, a = inputs()
    with torch.no_grad():
        z0, inj = model.begin_step(o, a)
        z, residuals = model.solve_traced(z0, inj, 5)
        z4 = model.solve(z0, inj, 4)
        expected = (z - z4).norm(dim=-1) / z4.norm(dim=-1)
    assert torch.equal(z, model.solve(z0, inj, 5))
    assert residuals.shape == (8, 5)
    assert torch.allclose(residuals[:, -1], expected)


# ------------------------------------------------------------ carrying


def test_state_carrying_as_defined():
    carried = tiny_model(carry=True)
    reinit = same_weights(carried, carry=False)
    o, a = inputs()
    z_prev = torch.randn(8, 12)
    with torch.no_grad():
        # First step: no previous state, so the two arms agree.
        assert torch.equal(carried.step(o, a, None)[0], reinit.step(o, a, None)[0])
        # Later steps: carried depends on z_prev, re-init ignores it.
        assert (carried.step(o, a, z_prev)[0] - carried.step(o, a, None)[0]).abs().max() > 1e-6
        assert torch.equal(reinit.step(o, a, z_prev)[0], reinit.step(o, a, None)[0])


def test_rollout_passes_final_state_to_next_step():
    model = tiny_model(carry=True)
    predictor = LoopedPredictor(model, 3)
    o, a = inputs()
    with torch.no_grad():
        o1, z1 = predictor.advance(o, a, None)
        o2, z2 = predictor.advance(o1, a, z1)
        manual_o1, manual_z1 = model.step(o, a, None, 3)
        manual_o2, _ = model.step(manual_o1, a, manual_z1, 3)
    assert torch.equal(o2, manual_o2)
    assert predictor.t_calls == 6


def test_planning_predictor_resets_each_imagination():
    model = tiny_model()
    planner = PlanningPredictor(model, 2, horizon=2)
    o, a = inputs()
    with torch.no_grad():
        first_a = planner.step(o, a)
        second = planner.step(first_a, a)
        first_b = planner.step(o, a)  # a new imagination must not see the previous state
        z_first = model.step(o, a, None, 2)[1]
        expected_second = model.step(first_a, a, z_first, 2)[0]
    assert torch.equal(first_a, first_b)
    assert torch.equal(second, expected_second)


# ------------------------------------------------------------ training


@pytest.mark.parametrize("arm", ["full", "plain"])
def test_tiny_model_overfits_small_batch(arm):
    env, data = tiny_sequences()
    cfg = LoopConfig(env.obs_dim, env.act_dim, z_dim=16, emb_dim=8, hidden=32, loops=2, grad_loops=1, **ARMS[arm])
    tcfg = LoopTrainConfig(steps=800, batch_windows=8, window=2, lr=3e-3, log_every=200)
    trained = train_looped(data, cfg, tcfg, seed=0)
    assert trained.stats["skipped_nonfinite_updates"] == 0
    assert trained.stats["final_loss"] < 0.3 * trained.stats["first_loss"]
    assert not any(p.requires_grad for p in trained.model.parameters())


def test_training_counts_and_pairing():
    env, data = tiny_sequences()
    tcfg = LoopTrainConfig(steps=5, batch_windows=4, window=2, log_every=2)
    settings = dict(z_dim=8, emb_dim=4, hidden=8, loops=4, grad_loops=2)
    mixed = train_looped(data, LoopConfig(env.obs_dim, env.act_dim, **settings), tcfg, seed=3)
    fixed = train_looped(data, LoopConfig(env.obs_dim, env.act_dim, mixed_depth=False, **settings), tcfg, seed=3)
    # Backward depth is identical; fixed spends exactly K - g no-grad calls per step.
    assert mixed.stats["t_calls_grad"] == fixed.stats["t_calls_grad"] == 5 * 2 * 2
    assert fixed.stats["t_calls_no_grad"] == 5 * 2 * 2
    assert mixed.stats["counts"]["backward_calls"] == 5


def test_nonfinite_loss_skips_the_update_and_is_counted(monkeypatch):
    env, data = tiny_sequences()
    tcfg = LoopTrainConfig(steps=4, batch_windows=4, window=2)
    cfg = LoopConfig(env.obs_dim, env.act_dim, z_dim=8, emb_dim=4, hidden=8, loops=2, grad_loops=1)
    healthy = train_looped(data, cfg, LoopTrainConfig(steps=1, batch_windows=4, window=2, lr=0.0), seed=0).model
    monkeypatch.setattr(LoopedDynamics, "decode", lambda self, z: self.decoder(z) * float("nan"))
    trained = train_looped(data, cfg, tcfg, seed=0)
    assert trained.stats["skipped_nonfinite_updates"] == 4
    assert trained.stats["final_loss"] is None
    for a, b in zip(trained.model.parameters(), healthy.parameters()):
        assert torch.equal(a, b)


# ------------------------------------------------------------ baselines


def test_mlp_counts_match_the_real_model():
    cfg = ModelConfig(8, 2, 0, 37, 2, 16)
    model = DynamicsModel(cfg)
    assert mlp_param_count(cfg) == sum(p.numel() for p in model.parameters())
    linear = sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert mlp_macs(cfg) == linear + model.W.numel()


def test_baselines_match_their_targets():
    looped = LoopedDynamics(LoopConfig(8, 2, loops=8, grad_loops=4))
    params = sum(p.numel() for p in looped.parameters())
    macs = looped.macs_per_step()
    assert macs == looped.base_macs() + 8 * looped.cell_macs()
    by_params = mlp_matching_params(params, 8, 2, 64)
    by_macs = mlp_matching_macs(macs, 8, 2, 64)
    assert mlp_param_count(by_params) == pytest.approx(params, rel=0.05)
    assert mlp_macs(by_macs) == pytest.approx(macs, rel=0.05)
    # The point of having both: they differ a lot, in the expected direction.
    assert mlp_macs(by_params) < macs / 2
    assert mlp_param_count(by_macs) > params * 2


# ------------------------------------------------------------ rollouts


class _BadAfter:
    """Returns the input until after_step, then replaces the masked rollouts with a fixed value."""

    def __init__(self, after_step, mask, value, in_carry=False):
        self.after_step, self.mask, self.value, self.in_carry = after_step, mask, value, in_carry
        self.t = 0

    def advance(self, o, a, carry):
        self.t += 1
        out = o.clone()
        state = torch.zeros(*o.shape[:-1], 2)
        if self.t > self.after_step:
            if self.in_carry:
                state[self.mask] = self.value
            else:
                out[self.mask] = self.value
        return out, state


def _fake_windows():
    true = torch.zeros(1, 4, 7, 3)
    act = torch.zeros(1, 4, 6, 1)
    return true, act, torch.ones(3), torch.full((3,), -2.0), torch.full((3,), 2.0)


def test_nonfinite_rollouts_are_counted_not_hidden():
    true, act, scale, low, high = _fake_windows()
    mask = torch.tensor([[True, False, False, False]])
    out = rollout_metrics(_BadAfter(2, mask, float("nan")), true, act, (1, 2, 3, 6), scale, low, high)
    assert out["h2"]["nonfinite_frac"] == 0.0
    assert out["h3"]["nonfinite_frac"] == 0.25
    assert out["h6"]["nonfinite_frac"] == 0.25  # sticky
    assert out["h3"]["blowup_frac"] == 0.25
    assert math.isfinite(out["h6"]["err"]) and out["h6"]["err"] > 0  # bounded, and it scores badly
    assert out["h6"]["err_intact"] == 0.0  # the three good rollouts alone are perfect


def test_clipped_rollouts_are_counted_separately():
    true, act, scale, low, high = _fake_windows()
    mask = torch.tensor([[True, True, False, False]])
    out = rollout_metrics(_BadAfter(0, mask, 100.0), true, act, (1,), scale, low, high)
    assert out["h1"]["clipped_frac"] == 0.5
    assert out["h1"]["nonfinite_frac"] == 0.0
    # Clipped to the bound of 2 in each dimension on half of the rollouts.
    assert out["h1"]["err"] == pytest.approx(math.sqrt(0.5 * 4.0))


def test_nonfinite_carried_state_counts_even_if_prediction_looks_fine():
    true, act, scale, low, high = _fake_windows()
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = rollout_metrics(_BadAfter(0, mask, float("inf"), in_carry=True), true, act, (1,), scale, low, high)
    assert out["h1"]["nonfinite_frac"] == 1.0
    assert out["h1"]["err_intact"] is None
    assert out["h1"]["z_explode_frac"] == 1.0


def test_all_rollouts_blown_up_still_returns_a_record():
    true, act, scale, low, high = _fake_windows()
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = rollout_metrics(_BadAfter(0, mask, float("nan")), true, act, (1, 6), scale, low, high)
    assert out["h6"]["blowup_frac"] == 1.0 and out["h6"]["err_intact"] is None


def test_rollout_metrics_agrees_with_eval_open_loop_errors():
    env = make_env("pendulum", 3, "interp", 0)
    holdout = collect_holdout(env, 2, 8, make_generator("agree"))
    scale = compute_obs_scale("pendulum")
    low, high = env.obs_low, env.obs_high
    torch.manual_seed(0)
    mlp = DynamicsModel(ModelConfig(env.obs_dim, env.act_dim, 0, 16, 2, 8))
    starts = (0, 2)
    true, act = make_windows(holdout, starts, 5)
    for reference, wrapped in [
        (PersistencePredictor(), MemorylessPredictor(PersistencePredictor().step)),
        (mlp, MemorylessPredictor(mlp.step)),
    ]:
        expected = open_loop_errors(reference, holdout, scale, low, high, (1, 5), starts)
        got = rollout_metrics(wrapped, true, act, (1, 5), scale, low, high, keep_per_env=True)
        assert torch.allclose(torch.tensor(got["h5"]["per_env"]), expected[5], atol=1e-5)
        assert torch.allclose(torch.tensor(got["h1"]["per_env"]), expected[1], atol=1e-5)


def test_grid_records_trained_depth_and_relation():
    env, data = tiny_sequences()
    cfg = LoopConfig(env.obs_dim, env.act_dim, z_dim=8, emb_dim=4, hidden=8, loops=6, grad_loops=2)
    model = train_looped(data, cfg, LoopTrainConfig(steps=2, batch_windows=2, window=2), seed=0).model
    scale = compute_obs_scale("pendulum")
    holdout = collect_holdout(env, 2, 4, make_generator("grid"))
    true, act = make_windows(holdout, (0,), 3)
    grid = rollout_grid(model, true, act, (1, 4, 64), (1, 3), scale, env.obs_low, env.obs_high)
    assert grid["trained_loops"] == 6 and grid["trained_depth_range"] == [2, 10]
    assert set(grid["loops"]) == {"1", "4", "6", "64"}  # the trained count is added
    assert grid["loops"]["1"]["relation"] == "below_trained"
    assert grid["loops"]["4"]["relation"] == "within_trained"
    assert grid["loops"]["64"]["relation"] == "above_trained"
    assert grid["loops"]["6"]["is_trained_loops"] and not grid["loops"]["4"]["is_trained_loops"]
    assert set(grid["loops"]["64"]["horizons"]) == {"h1", "h3"}
    assert grid["loops"]["64"]["t_calls"] == 64 * 3
    assert loop_relation(8, 8, 8) == "within_trained"


# ------------------------------------------------------------ contact


def _puck(n=1):
    return PuckEnv.from_values([0.5] * n, [0.1] * n)


def test_substepped_contact_step_equals_env_step_exactly():
    env = make_env("puck", 4, "train", 0)
    gen = make_generator("contact_exact")
    state = env.reset_diverse(3, gen)
    action = 2.0 * torch.rand(4, 3, 2, generator=gen) - 1.0
    for _ in range(5):
        expected = env.step(state, action)
        got, _, _ = step_with_contact(env, state, action)
        assert torch.equal(got, expected)
        state = expected


def test_contact_flags_on_known_states():
    env = _puck()
    a = torch.zeros(1, 1, 2)
    touching = torch.tensor([[[0.0, 0.0, 0.15, 0.0, 0, 0, 0, 0]]], dtype=torch.float64)
    apart = torch.tensor([[[-0.5, 0.0, 0.5, 0.0, 0, 0, 0, 0]]], dtype=torch.float64)
    at_wall = torch.tensor([[[0.95, 0.0, -0.5, 0.0, 0, 0, 0, 0]]], dtype=torch.float64)
    flags = lambda state: tuple(x.item() for x in step_with_contact(env, state, a)[1:])
    assert flags(touching) == (True, False)
    assert flags(apart) == (False, False)
    assert flags(at_wall) == (False, True)


def test_contact_inside_a_control_step_is_caught_though_boundaries_miss_it():
    env = _puck()
    # Agent racing at the puck: no overlap at the start of the step, so a boundary check at t would say no.
    state = torch.tensor([[[-0.30, 0.0, 0.0, 0.0, 3.0, 0.0, 0, 0]]], dtype=torch.float64)
    assert (state[..., 2:4] - state[..., :2]).norm(dim=-1).item() > 2 * env.radius
    _, contact, _ = step_with_contact(env, state, torch.tensor([[[1.0, 0.0]]]))
    assert contact.item()


def test_replayed_flags_match_collected_flags():
    env = make_env("puck", 3, "interp", 0)
    trajs, ap, wall = collect_contact_trajectories(env, 4, 20, "push", make_generator("replay"))
    replay_ap, replay_wall = replay_contact_flags(env, trajs)
    assert ap.any()  # the push policy does produce contact
    # Replay starts from float32 observations, so allow an isolated disagreement at a grazing touch.
    assert (replay_ap != ap).float().mean() < 0.02
    assert (replay_wall != wall).float().mean() < 0.02


def test_window_contact_mask_order_matches_make_windows():
    ap = torch.zeros(1, 2, 6, dtype=torch.bool)
    ap[0, 1, 3] = True
    mask = window_contact_mask(ap, (0, 2), 2)
    # Order: (traj 0, start 0), (traj 0, start 2), (traj 1, start 0), (traj 1, start 2).
    assert mask.tolist() == [[False, False, False, True]]


def test_contact_masks_on_known_sequence():
    ap = torch.zeros(1, 8, dtype=torch.bool)
    ap[0, 3:5] = True
    wall = torch.zeros(1, 8, dtype=torch.bool)
    lag = contact_masks(ap, wall, tol=1, burn=1, alignment="lag")
    assert lag["positive"][0].nonzero().flatten().tolist() == [4, 5]
    assert lag["negative"][0].nonzero().flatten().tolist() == [1, 2, 7]
    lead = contact_masks(ap, wall, tol=1, burn=1, alignment="lead")
    assert lead["positive"][0].nonzero().flatten().tolist() == [3]
    assert lead["negative"][0].nonzero().flatten().tolist() == [1, 7]
    assert not (lag["positive"] & lag["negative"]).any()


def test_contact_masks_exclude_walls_from_negatives_and_reset_contact_from_onsets():
    ap = torch.zeros(1, 8, dtype=torch.bool)
    ap[0, 0] = True  # already touching at reset
    wall = torch.zeros(1, 8, dtype=torch.bool)
    wall[0, 5] = True
    masks = contact_masks(ap, wall, tol=1, burn=1)
    assert not masks["positive"].any()
    assert masks["negative"][0].nonzero().flatten().tolist() == [3, 4]


# ------------------------------------------------------------ AUROC and signals


def test_auroc_on_known_data():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auroc([5, 5, 5, 5], [0, 1, 0, 1]) == pytest.approx(0.5)  # all ties
    assert auroc([1, 2, 2, 3], [0, 1, 0, 1]) == pytest.approx(0.875)  # one tied pair of four counts half
    assert math.isnan(auroc([1, 2], [1, 1]))
    rng = np.random.default_rng(0)
    assert auroc(rng.normal(size=20000), rng.integers(0, 2, 20000)) == pytest.approx(0.5, abs=0.02)
    # Separation d' = 1 has the known AUROC Phi(1 / sqrt 2) = 0.760.
    x = np.concatenate([rng.normal(1.0, 1.0, 20000), rng.normal(0.0, 1.0, 20000)])
    y = np.concatenate([np.ones(20000), np.zeros(20000)])
    assert auroc(x, y) == pytest.approx(0.760, abs=0.01)


def test_auroc_groups_reports_nonfinite_as_alarms():
    out = auroc_groups(torch.tensor([float("nan"), 2.0]), torch.tensor([1.0, 0.0]))
    assert out["n_nonfinite"] == 1 and out["n_pos"] == 2 and out["n_neg"] == 2
    assert out["auroc"] == 1.0


def test_shift_auroc_on_constructed_signal():
    T = 20
    control = torch.zeros(4, T)
    switched = torch.zeros(4, T)
    switched[:, 11:] = 1.0  # rises exactly when post-switch dynamics first become visible
    assert shift_auroc(switched, control, t_switch=10, window=5, burn=2)["auroc"] == 1.0
    # A signal late by the whole onset window has no skill.
    late = torch.zeros(4, T)
    late[:, 16:] = 1.0
    assert shift_auroc(late, control, t_switch=10, window=5, burn=2)["auroc"] == pytest.approx(0.5)
    keep = torch.tensor([True, False, False, False])
    assert shift_auroc(switched, control, 10, 5, 2, keep)["n_pos"] == 5


def test_gaussian_model_nll_and_cusum():
    gen = torch.Generator().manual_seed(0)
    clean = torch.randn(5000, 3, generator=gen, dtype=torch.float64) * torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    model = GaussianResidualModel.fit(clean)
    assert model.mean.abs().max() < 0.1
    # Mahalanobis of N(0, Sigma) has expectation D.
    assert model.mahalanobis(clean).mean().item() == pytest.approx(3.0, abs=0.15)
    assert model.nll(torch.tensor([[0.0, 0.0, 0.0]])) < model.nll(torch.tensor([[8.0, 8.0, 8.0]]))
    # NLL at the mean of a unit Gaussian is D/2 log(2 pi) plus half the log determinant.
    unit = GaussianResidualModel(torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64), 0.0, 2)
    assert unit.nll(torch.zeros(1, 2)).item() == pytest.approx(math.log(2 * math.pi))
    seq = clean[:40].reshape(1, 40, 3).clone()
    seq[:, 20:] *= 5.0
    stat = model.cusum(seq, kappa=2.0)
    assert (stat >= 0).all()
    assert stat[0, 25:].mean() > stat[0, :20].mean() + 5.0
    nan_seq = seq.clone()
    nan_seq[0, 10] = float("nan")
    assert torch.isfinite(model.cusum(nan_seq)).all()
    with pytest.raises(ValueError):
        GaussianResidualModel.fit(torch.full((10, 3), float("nan")))


def test_to_signal_time_alignment():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = to_signal_time(x)
    assert math.isnan(out[0, 0].item()) and out[0, 1:].tolist() == [1.0, 2.0]


def test_teacher_forced_trace_shapes_and_carry():
    carried = tiny_model(carry=True)
    reinit = same_weights(carried, carry=False)
    gen = torch.Generator().manual_seed(0)
    obs = torch.randn(4, 6, 3, generator=gen)
    act = torch.rand(4, 5, 1, generator=gen)
    with torch.no_grad():
        a = teacher_forced_trace(carried, obs, act)
        b = teacher_forced_trace(reinit, obs, act)
    assert a["residual_last"].shape == (4, 5) and a["residual_mean"].shape == (4, 5) and a["pred"].shape == (4, 5, 3)
    assert (a["residual_last"] >= 0).all()
    assert torch.equal(a["pred"][:, 0], b["pred"][:, 0])  # nothing to carry yet
    assert (a["pred"][:, 1:] - b["pred"][:, 1:]).abs().max() > 1e-6
    # Re-init is memoryless: changing earlier observations cannot change a later prediction.
    changed = obs.clone()
    changed[:, 0] += 1.0
    with torch.no_grad():
        c = teacher_forced_trace(reinit, changed, act)
    assert torch.equal(c["pred"][:, 2], b["pred"][:, 2])


# ------------------------------------------------------------ regime switches


def test_collect_switching_matches_control_until_the_switch():
    pre = make_env("pendulum", 2, "interp", 0)
    post = make_env("pendulum", 2, "extrap", 0)
    switched = collect_switching(pre, post, 3, 12, 5, "smooth", make_generator("switch"))
    control = collect_switching(pre, pre, 3, 12, 5, "smooth", make_generator("switch"))
    assert torch.equal(switched.obs[:, :, :6], control.obs[:, :, :6])
    assert torch.equal(switched.act, control.act)
    assert (switched.obs[:, :, 6] - control.obs[:, :, 6]).abs().max() > 1e-6
    assert switched.obs.shape == (2, 3, 13, 3) and isinstance(switched, Trajectories)


# ------------------------------------------------------------ outcome rules


def _verdicts(no_reinject, no_norm, no_mixed, plain):
    return {"no_reinject": no_reinject, "no_norm": no_norm, "no_mixed": no_mixed, "plain": plain}


def test_two_x_threshold_means_two_x():
    full = [1.0, 1.0, 1.01, 0.99, 1.0]
    assert paired_comparison(full, [2.1, 2.2, 2.1, 2.2, 2.15], threshold=TWO_X_THRESHOLD)["verdict"] == "a_wins"
    assert paired_comparison(full, [1.8, 1.9, 1.8, 1.85, 1.8], threshold=TWO_X_THRESHOLD)["verdict"] == "no_detectable_difference"


def test_h7_outcome_rule():
    win, none = "a_wins", "no_detectable_difference"
    assert h7_outcome(_verdicts(win, win, win, win), 0.5, True) == "supported"
    assert h7_outcome(_verdicts(win, none, win, win), 0.5, True) == "not_supported_some_ablations_under_2x"
    assert h7_outcome(_verdicts(win, win, win, none), 0.0, True) == "killed_plain_core_stable"
    # Under 2x worse on a saturated metric, but blowing up, is not stable.
    assert h7_outcome(_verdicts(win, win, win, none), 0.4, True) == "supported"
    assert h7_outcome(_verdicts(none, none, none, none), 0.4, True) == "not_supported_some_ablations_under_2x"
    assert h7_outcome(_verdicts(win, win, win, win), 0.0, False).startswith("uninformative")
    assert h7_outcome(_verdicts("insufficient_seeds", win, win, win), 0.0, True) == "insufficient_seeds"


def test_h8_and_h9_rules():
    assert h8_outcome("a_wins") == "supported"
    assert h8_outcome("no_detectable_difference").startswith("killed")
    assert h9_target_passes(0.8, "a_wins")
    assert not h9_target_passes(0.69, "a_wins")
    assert not h9_target_passes(0.9, "no_detectable_difference")
    assert not h9_target_passes(float("nan"), "a_wins")
