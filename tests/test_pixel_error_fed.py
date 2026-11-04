"""The error-fed core was moved in from an interrupted run, unreviewed. These pin down the design points where a bug would bias H14p."""

import math

import pytest
import torch

from intr.envs import make_generator
from intr.looped import SequenceData
from intr.pixel.error_fed import (
    ErrorFedConfig,
    ErrorFedDynamics,
    ErrorFedPredictor,
    ErrorFedTrainConfig,
    make_donor,
    other_env_donor,
    run_filter,
    sample_sequences,
    sequence_loss,
    train_error_fed,
)

O, A = 5, 2


def model(use_error=True, seed=0):
    torch.manual_seed(seed)
    return ErrorFedDynamics(ErrorFedConfig(O, A, z_dim=12, emb_dim=6, hidden=16, loops=3, grad_loops=2, use_error=use_error)).requires_grad_(False)


def sequence(N=4, T=8):
    gen = make_generator("seq")
    return torch.randn(N, T + 1, O, generator=gen), torch.rand(N, T, A, generator=gen) * 2 - 1


def test_prediction_at_t_reads_nothing_after_t():
    m = model()
    obs, act = sequence()
    _, base = run_filter(m, obs, act)
    t = 3
    changed = obs.clone()
    changed[:, t + 1 :] += 5.0
    _, again = run_filter(m, changed, act)
    assert torch.equal(base["pred_next"][:, : t + 1], again["pred_next"][:, : t + 1])
    assert not torch.equal(base["pred_next"][:, t + 1 :], again["pred_next"][:, t + 1 :])


def test_error_channel_is_live_on_the_core_and_dead_on_the_no_error_control():
    obs, act = sequence()
    changed = obs.clone()
    changed[:, -1] += 5.0  # only the last error changes, so only the belief's pending block can differ
    on = model(True)
    a, _ = run_filter(on, obs, act)
    b, _ = run_filter(on, changed, act)
    assert not torch.equal(a.pending[:, : O], b.pending[:, : O])
    query, action = obs[:, 0], act[:, 0]
    assert not torch.equal(on.step(query, action, a.z, a.pending)[0], on.step(query, action, b.z, b.pending)[0])
    off = model(False)
    c, _ = run_filter(off, obs, act)
    d, _ = run_filter(off, changed, act)
    assert torch.equal(off.step(query, action, c.z, c.pending)[0], off.step(query, action, d.z, d.pending)[0])


def test_controls_share_parameters_at_init():
    a, b = model(True), model(False)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_shuffled_control_feeds_another_environment_error():
    m = model()
    obs, act = sequence()
    donor = other_env_donor(4)
    assert (donor != torch.arange(4)).all()
    _, trace = run_filter(m, obs, act, donor)
    assert torch.equal(trace["fed_error"], trace["own_error"][donor])
    assert not torch.equal(trace["fed_error"], trace["own_error"])
    for n in (2, 5):
        d = make_donor(n, make_generator("d", n))
        assert (d != torch.arange(n)).all() and sorted(d.tolist()) == list(range(n))


def test_imagination_consumes_the_pending_error_once():
    m = model()
    obs, act = sequence()
    belief, _ = run_filter(m, obs, act)
    predictor = ErrorFedPredictor(m, belief, horizon=3, online=False)
    o, a = obs[:, :2], act[:, :2]  # (N, 2, O), a batch of 2 per environment
    first = predictor.step(o, a)
    second = predictor.step(o, a)
    assert not torch.equal(first, second)
    # By hand: the first step uses z and pending, the second the carried z and the explicit no-observation block.
    from intr.pixel.error_fed import match_batch

    next_o, z = m.step(o, a, match_batch(belief.z, o), match_batch(belief.pending, o))
    assert torch.equal(first, next_o)
    assert torch.equal(second, m.step(o, a, z, m.no_observation(o))[0])


def test_step_resets_to_the_belief_each_horizon():
    m = model()
    obs, act = sequence()
    belief, _ = run_filter(m, obs, act)
    predictor = ErrorFedPredictor(m, belief, horizon=2, online=False)
    o, a = obs[:, :2], act[:, :2]
    first = predictor.step(o, a)
    predictor.step(o, a)
    assert torch.equal(first, predictor.step(o, a))
    predictor.observe(obs[:, 0], act[:, 0], obs[:, 1])  # online False: the belief stays where adaptation left it
    assert predictor.belief is belief


def test_no_observation_is_distinct_from_zero_error():
    m = model()
    obs, act = sequence()
    zero_error = m.innovation(torch.zeros(4, O), act[:, 0])
    none = m.no_observation(obs[:, 0])
    assert zero_error[:, -1].eq(1.0).all() and none.eq(0.0).all()
    assert not torch.equal(m.step(obs[:, 0], act[:, 0], None, zero_error)[0], m.step(obs[:, 0], act[:, 0], None, none)[0])


def test_nonfinite_error_stays_finite():
    m = model()
    err = torch.full((4, O), float("nan"))
    assert torch.isfinite(m.innovation(err, torch.zeros(4, A))).all()


@pytest.mark.parametrize("L,H", [(0, 3), (5, 3), (8, 8), (6, 6)])
def test_sample_sequences_shapes_and_hops(L, H):
    E, B, T = 6, 3, 8
    gen = make_generator("s")
    data = SequenceData(torch.randn(E, B, T + 1, O, generator=gen), torch.rand(E, B, T, A, generator=gen))
    batch = sample_sequences(data, 4, L, H, 0.5, make_generator("b", L, H))
    assert batch.filter_obs.shape == (4, L + 1, O) and batch.filter_act.shape == (4, L, A)
    assert batch.imag_obs.shape == (4, H + 1, O) and batch.imag_act.shape == (4, H, A)
    assert len(set(batch.env_idx.tolist())) == 4


def test_sequence_loss_runs_with_no_filter_phase_and_has_gradient_through_z():
    torch.manual_seed(0)
    m = ErrorFedDynamics(ErrorFedConfig(O, A, z_dim=12, emb_dim=6, hidden=16, loops=3, grad_loops=2))
    gen = make_generator("s")
    data = SequenceData(torch.randn(6, 3, 9, O, generator=gen), torch.rand(6, 3, 8, A, generator=gen))
    tcfg = ErrorFedTrainConfig(steps=1, batch_seqs=4, filter_max=4, horizon_max=3)
    for L in (0, 4):
        batch = sample_sequences(data, 4, L, 3, 0.5, make_generator("b"))
        loss, info = sequence_loss(m, batch, tcfg, make_generator("depth"))
        assert math.isfinite(loss.item()) and info["steps"] == L + 3
    m.zero_grad()
    loss.backward()
    assert m.prelude_net[-1].weight.grad is not None and m.update[-1].weight.grad.abs().sum() > 0


def test_training_is_seeded_shared_across_arms_and_finite():
    gen = make_generator("t")
    data = SequenceData(torch.randn(6, 2, 9, O, generator=gen), torch.rand(6, 2, 8, A, generator=gen))
    cfg = dict(obs_dim=O, act_dim=A, z_dim=8, emb_dim=4, hidden=8, loops=2, grad_loops=1)
    tcfg = ErrorFedTrainConfig(steps=6, batch_seqs=3, filter_max=3, horizon_max=2)  # several updates, so some has a filter phase
    low, high = -50 * torch.ones(O), 50 * torch.ones(O)
    a = train_error_fed(data, ErrorFedConfig(**cfg), tcfg, 0, low, high)
    b = train_error_fed(data, ErrorFedConfig(**cfg), tcfg, 0, low, high)
    assert all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()))
    assert a.stats["skipped_nonfinite_updates"] == 0
    shuffled = train_error_fed(data, ErrorFedConfig(**cfg), tcfg, 0, low, high, shuffle_errors=True)
    control = train_error_fed(data, ErrorFedConfig(**cfg, use_error=False), tcfg, 0, low, high)
    assert not all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), shuffled.model.state_dict().values()))
    assert not all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(), control.model.state_dict().values()))
