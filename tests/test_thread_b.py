"""Thread B measurement: synthetic data with known answers, then the pipeline on tiny settings."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import thread_a  # noqa: E402
import thread_b  # noqa: E402
from intr.adapt import RLSState, adapt  # noqa: E402
from intr.calibration import (  # noqa: E402
    P_TRACE_LIMIT,
    Schedule,
    StreamPolicy,
    SurpriseConfig,
    SurpriseDetector,
    across_env_spearman,
    average_ranks,
    build_stream,
    coverage,
    fit_sigma2,
    gaussian_nll,
    n_regimes_needed,
    partial_spearman,
    post_switch_mask,
    query_signals,
    regime_sequence,
    rmse_over,
    rollout_starts,
    run_prequential,
    simulate_stream,
    spearman,
    steady_mask,
    subset_env,
    well_specified_spearman,
    windup_summary,
    within_env_spearman,
)
from intr.envs import make_env, make_generator  # noqa: E402
from intr.train import TrainConfig  # noqa: E402

DOUBLE = torch.float64


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- ranking


def test_average_ranks_handles_ties():
    assert average_ranks([10.0, 20.0, 20.0, 5.0]).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_spearman_is_exact_on_monotone_data_and_undefined_on_constants():
    x = np.arange(50.0)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -np.exp(x / 10)) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones(50)))
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))


def test_spearman_drops_non_finite_pairs():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([1.0, 2.0, 3.0, 4.0, -100.0])
    assert spearman(x, y) == pytest.approx(1.0)


def test_informative_variance_scores_high_and_uninformative_scores_near_zero():
    rng = np.random.default_rng(0)
    n = 5000
    variance = np.exp(2.0 * rng.standard_normal(n))
    # Squared errors truly governed by the variance, with a little multiplicative noise.
    close = variance * np.exp(0.2 * rng.standard_normal(n))
    assert spearman(variance, close) > 0.95
    # One-dimensional Gaussian draws: informative, but capped by the chi-square draw.
    drawn = variance * rng.standard_normal(n) ** 2
    assert 0.5 < spearman(variance, drawn) < 0.8
    unrelated = np.exp(2.0 * rng.standard_normal(n))
    assert abs(spearman(variance, unrelated)) < 0.05


def test_partial_spearman_removes_a_shared_confounder_and_keeps_real_signal():
    rng = np.random.default_rng(1)
    n = 5000
    norm = rng.standard_normal(n)
    # Both grow with the norm and share nothing else.
    variance = norm + 0.3 * rng.standard_normal(n)
    error = norm + 0.3 * rng.standard_normal(n)
    assert spearman(variance, error) > 0.8
    # Not exactly 0: the rank of variance is a slightly nonlinear function of the rank of norm.
    assert abs(partial_spearman(variance, error, [norm])) < 0.1
    # Now the variance knows something the norm does not.
    extra = rng.standard_normal(n)
    variance2 = norm + extra
    error2 = norm + extra + 0.3 * rng.standard_normal(n)
    assert partial_spearman(variance2, error2, [norm]) > 0.8


def test_partial_spearman_is_undefined_when_the_control_is_the_signal():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(200)
    assert np.isnan(partial_spearman(x, rng.standard_normal(200), [x]))


def test_within_and_across_answer_different_questions():
    rng = np.random.default_rng(3)
    E, N = 60, 400
    # Across only: each environment has a level that signal and error share; inside it, no relation.
    level = rng.standard_normal((E, 1))
    signal = level + 0.01 * rng.standard_normal((E, N))
    err = level + 0.01 * rng.standard_normal((E, N))
    assert abs(np.nanmean(within_env_spearman(signal, err))) < 0.1
    assert across_env_spearman(signal, err) > 0.9
    # Within only: queries rank together, and the environment levels are unrelated.
    shared = rng.standard_normal((E, N))
    signal2 = shared + rng.standard_normal((E, 1)) * 10
    err2 = shared + rng.standard_normal((E, 1)) * 10
    assert np.nanmean(within_env_spearman(signal2, err2)) > 0.99
    assert abs(across_env_spearman(signal2, err2)) < 0.6


# ---------------------------------------------------------------- calibration


def _gaussian(seed, sigma2, q, D=2):
    E, N = q.shape
    std = (torch.tensor(sigma2, dtype=DOUBLE) * (1.0 + q)[..., None]).sqrt()
    return std * torch.randn(E, N, D, generator=_gen(seed), dtype=DOUBLE)


def test_sigma_and_coverage_are_correct_on_known_gaussian_data():
    q = torch.rand(10, 4000, generator=_gen(0), dtype=DOUBLE) * 5.0
    residual = _gaussian(1, [0.25, 4.0], q)
    sigma2 = fit_sigma2(residual, q)
    assert torch.allclose(sigma2, torch.tensor([0.25, 4.0], dtype=DOUBLE), rtol=0.03)
    for level in (0.5, 0.9):
        assert coverage(residual, q, sigma2, level).mean().item() == pytest.approx(level, abs=0.01)


def test_overconfident_and_underconfident_intervals_are_detected():
    q = torch.zeros(5, 4000, dtype=DOUBLE)
    residual = _gaussian(2, [1.0, 1.0], q)
    too_narrow = coverage(residual, q, torch.tensor([0.25, 0.25], dtype=DOUBLE), 0.9).mean().item()
    too_wide = coverage(residual, q, torch.tensor([9.0, 9.0], dtype=DOUBLE), 0.5).mean().item()
    assert too_narrow < 0.7  # true value is P(|z| < 0.82) = 0.59
    assert too_wide > 0.9


def test_constant_variance_covers_marginally_but_misses_conditionally_and_loses_on_nll():
    q = torch.exp(1.5 * torch.randn(5, 8000, generator=_gen(3), dtype=DOUBLE))
    residual = _gaussian(4, [1.0, 1.0], q)
    sigma_q, sigma_c = fit_sigma2(residual, q), fit_sigma2(residual, None)
    assert gaussian_nll(residual, q, sigma_q).mean() < gaussian_nll(residual, None, sigma_c).mean() - 0.05
    top = q > q.quantile(0.9)
    half = 1.6449 * sigma_c.sqrt()
    inside_top = (residual.abs() <= half)[top].to(DOUBLE).mean().item()
    assert inside_top < 0.8  # the constant interval undercovers exactly where q says it should
    half_q = 1.6449 * (sigma_q * (1 + q)[..., None]).sqrt()
    assert (residual.abs() <= half_q)[top].to(DOUBLE).mean().item() == pytest.approx(0.9, abs=0.03)


def test_well_specified_reference_tracks_the_spread_of_q():
    wide = 100.0 * torch.exp(2.0 * torch.randn(4, 2000, generator=_gen(5), dtype=DOUBLE))
    sigma2 = torch.tensor([1.0], dtype=DOUBLE)
    assert 0.5 < np.mean(well_specified_spearman(wide, sigma2, _gen(6))) < 0.8
    # With q far below 1 the predictive variance is flat, so even a correct model ranks at about 0.
    tiny = 1e-6 * torch.rand(4, 2000, generator=_gen(7), dtype=DOUBLE)
    assert abs(np.mean(well_specified_spearman(tiny, sigma2, _gen(8)))) < 0.1
    # More output dimensions average out the chi-square draw.
    sigma8 = torch.ones(8, dtype=DOUBLE)
    assert np.mean(well_specified_spearman(wide, sigma8, _gen(9))) > np.mean(well_specified_spearman(wide, sigma2, _gen(9))) + 0.1


def test_query_signals_agree_with_the_predictor(latent_backbone):
    env = make_env("pendulum", 3, "interp", "qs")
    from intr.envs import collect_trajectories

    data = collect_trajectories(env, 1, 20, "smooth", make_generator("qs_data")).flat()
    holdout = collect_trajectories(env, 2, 15, "iid", make_generator("qs_holdout")).flat()
    predictor = adapt("rls", latent_backbone, data, lambda_=0.1).predictor
    before = predictor.rls.P.clone()
    sig = query_signals(predictor, holdout)
    assert torch.equal(predictor.rls.P, before)  # a query must not write
    assert torch.allclose(sig.q, predictor.variance(holdout.obs, holdout.act), atol=1e-10)
    target = latent_backbone.model.target(holdout.obs, holdout.next_obs)
    assert torch.allclose(sig.residual.float(), target - predictor.predict_delta(holdout.obs, holdout.act), atol=1e-3)
    assert sig.sq_err.shape == holdout.obs.shape[:2]


# ---------------------------------------------------------------- prequential


def _problem(seed, E=2, T=25, D=2, F=6):
    gen = _gen(seed)
    phi = torch.randn(E, T, F, generator=gen, dtype=DOUBLE)
    v = torch.randn(E, T, D, generator=gen, dtype=DOUBLE)
    W0 = torch.randn(E, D, F, generator=gen, dtype=DOUBLE)
    return phi, v, W0


def _ridge_prediction(phi, v, W0, t, lambda_, gamma=1.0):
    """Prediction for step t from batch weighted ridge on steps before t: independent of the RLS code."""
    E, _, F = phi.shape
    past_phi, past_v = phi[:, :t], v[:, :t]
    weights = gamma ** torch.arange(t - 1, -1, -1, dtype=DOUBLE)[None, :, None]
    A = past_phi.transpose(1, 2) @ (past_phi * weights) + lambda_ * gamma**t * torch.eye(F, dtype=DOUBLE)
    residual = past_v - past_phi @ W0.transpose(1, 2)
    delta = torch.linalg.solve(A, past_phi.transpose(1, 2) @ (residual * weights))  # (E, F, D)
    W = W0 + delta.transpose(1, 2)
    return (W @ phi[:, t][:, :, None]).squeeze(-1)


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_every_prequential_score_equals_batch_ridge_on_strictly_earlier_data(gamma):
    phi, v, W0 = _problem(0)
    lambda_ = 0.5
    result = run_prequential(phi, v, W0, StreamPolicy("p", lambda_, forgetting=gamma))
    for t in range(1, phi.shape[1]):
        expected = ((v[:, t] - _ridge_prediction(phi, v, W0, t, lambda_, gamma)) ** 2).mean(-1)
        assert torch.allclose(result.err[:, t], expected, atol=1e-8, rtol=1e-8)


def test_first_score_uses_the_prior_even_when_a_write_would_fit_it_exactly():
    phi, v, W0 = _problem(1)
    result = run_prequential(phi, v, W0, StreamPolicy("p", 1e-8))
    expected = ((v[:, 0] - (W0 @ phi[:, 0][:, :, None]).squeeze(-1)) ** 2).mean(-1)
    assert torch.allclose(result.err[:, 0], expected, atol=1e-10)
    assert (result.err[:, 0] > 1e-3).all()  # write-then-score would have given about 0 here


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {},
        {"forgetting": 0.9},
        {"surprise": SurpriseConfig(threshold=1.5, min_gap=2)},
    ],
)
def test_the_future_cannot_change_a_score(policy_kwargs):
    phi, v, W0 = _problem(2, T=40)
    t0 = 20
    base = run_prequential(phi, v, W0, StreamPolicy("p", 0.5, **policy_kwargs))
    # Change the target at t0 and everything after. Scores before t0 must be identical.
    v2 = v.clone()
    v2[:, t0:] = v2[:, t0:] + 100.0
    changed_target = run_prequential(phi, v2, W0, StreamPolicy("p", 0.5, **policy_kwargs))
    assert torch.equal(base.err[:, :t0], changed_target.err[:, :t0])
    # The change must arrive: from t0 on, the scores differ.
    assert (base.err[:, t0] != changed_target.err[:, t0]).all()
    # Change only later features. Score at t0 reads phi at t0 and earlier, so it is unchanged too.
    phi2 = phi.clone()
    phi2[:, t0 + 1 :] = phi2[:, t0 + 1 :] * 3.0
    changed_phi = run_prequential(phi2, v, W0, StreamPolicy("p", 0.5, **policy_kwargs))
    assert torch.equal(base.err[:, : t0 + 1], changed_phi.err[:, : t0 + 1])


def test_writes_do_reach_later_scores():
    phi, v, W0 = _problem(3)
    v2 = v.clone()
    v2[:, 5] = v2[:, 5] + 10.0
    a = run_prequential(phi, v, W0, StreamPolicy("p", 0.5))
    b = run_prequential(phi, v2, W0, StreamPolicy("p", 0.5))
    assert torch.equal(a.err[:, :5], b.err[:, :5])
    assert (a.err[:, 6] != b.err[:, 6]).all()


def test_forgetting_of_one_equals_persist_and_the_plain_rls_recursion():
    phi, v, W0 = _problem(4, T=60)
    stream = _FakeStream(phi.shape[0], phi.shape[1])
    persist = run_prequential(phi, v, W0, thread_b.make_policy("persist", {"lambda_": 0.5}, stream))
    one = run_prequential(phi, v, W0, thread_b.make_policy("forgetting", {"lambda_": 0.5, "forgetting": 1.0}, stream))
    assert torch.equal(persist.err, one.err)
    assert torch.equal(persist.trace, one.trace)
    rls = RLSState(W0, 0.5)
    for t in range(phi.shape[1]):
        pred = (rls.W @ phi[:, t][:, :, None]).squeeze(-1)
        assert torch.equal(persist.err[:, t], ((v[:, t] - pred) ** 2).mean(-1))
        rls.write(phi[:, t], v[:, t])
    below_one = run_prequential(phi, v, W0, thread_b.make_policy("forgetting", {"lambda_": 0.5, "forgetting": 0.9}, stream))
    assert not torch.equal(persist.err[:, 10:], below_one.err[:, 10:])


class _FakeStream:
    """Just the reset schedules make_policy reads."""

    def __init__(self, E, T, every=10):
        self.episode_start = (torch.arange(T) % every == 0)[None].expand(E, -1)
        self.switch = torch.zeros(E, T, dtype=torch.bool)


def test_reset_really_clears_state_and_only_where_asked():
    phi, v, W0 = _problem(5, T=30)
    t0 = 12
    reset_at = torch.zeros(2, 30, dtype=torch.bool)
    reset_at[0, t0] = True  # stream 0 only
    result = run_prequential(phi, v, W0, StreamPolicy("p", 0.5, reset_at=reset_at))
    fresh = run_prequential(phi[:, t0:], v[:, t0:], W0, StreamPolicy("p", 0.5))
    persist = run_prequential(phi, v, W0, StreamPolicy("p", 0.5))
    # Stream 0 after the reset is a run started fresh at t0: all data before t0 is gone, W and P both.
    assert torch.allclose(result.err[0, t0:], fresh.err[0], atol=1e-12)
    expected = ((v[0, t0] - W0[0] @ phi[0, t0]) ** 2).mean()
    assert result.err[0, t0].item() == pytest.approx(expected.item())
    assert torch.equal(result.err[0, :t0], persist.err[0, :t0])
    assert torch.equal(result.err[1], persist.err[1])
    assert result.resets.tolist() == [1, 0]
    assert result.trace[0, t0].item() == pytest.approx(fresh.trace[0, 0].item())


def test_detector_is_silent_when_stationary_and_fires_soon_after_a_jump():
    cfg = SurpriseConfig(threshold=3.0)
    detector = SurpriseDetector(2, cfg)
    fired = []
    for t in range(100):
        err = torch.tensor([1.0, 10.0 if t >= 60 else 1.0], dtype=DOUBLE)
        fired.append(detector.update(err))
    fired = torch.stack(fired)  # (T, 2)
    assert not fired[:, 0].any()
    assert not fired[:60, 1].any()
    assert fired[60:66, 1].any()


def test_detector_waits_min_gap_after_a_reset():
    cfg = SurpriseConfig(threshold=2.0, min_gap=10)
    detector = SurpriseDetector(1, cfg)
    for _ in range(30):
        detector.update(torch.tensor([1.0], dtype=DOUBLE))
    detector.clear(torch.tensor([True]))
    # The first error after a reset seeds the averages, then errors rise at once.
    triggers = [detector.update(torch.tensor([1.0 * 2.0**i], dtype=DOUBLE)).item() for i in range(12)]
    assert not any(triggers[:9])
    assert any(triggers[9:])


def test_surprise_reset_adapts_after_a_real_regime_change():
    gen = _gen(6)
    E, T, D, F = 3, 300, 1, 4
    phi = torch.randn(E, T, F, generator=gen, dtype=DOUBLE)
    W_a = torch.randn(E, D, F, generator=gen, dtype=DOUBLE)
    W_b = W_a + 5.0
    v = phi @ W_a.transpose(1, 2)
    v[:, 150:] = phi[:, 150:] @ W_b.transpose(1, 2)
    W0 = torch.zeros(E, D, F, dtype=DOUBLE)
    surprise = run_prequential(phi, v, W0, StreamPolicy("s", 0.1, surprise=SurpriseConfig(threshold=3.0)))
    persist = run_prequential(phi, v, W0, StreamPolicy("p", 0.1))
    assert (surprise.resets >= 1).all()
    assert surprise.err[:, 200:].mean() < 0.1 * persist.err[:, 200:].mean()


# ---------------------------------------------------------------- wind-up and numerical failure


def test_trace_and_eigenvalues_track_wind_up_exactly_without_excitation():
    E, T, F = 2, 30, 3
    gamma, lambda_ = 0.9, 2.0
    phi = torch.zeros(E, T, F, dtype=DOUBLE)
    v = torch.zeros(E, T, 1, dtype=DOUBLE)
    W0 = torch.zeros(E, 1, F, dtype=DOUBLE)
    result = run_prequential(phi, v, W0, StreamPolicy("f", lambda_, forgetting=gamma), eig_every=1)
    expected = F / lambda_ * gamma ** -(torch.arange(T, dtype=DOUBLE) + 1.0)
    assert torch.allclose(result.trace[0], expected, rtol=1e-10)
    assert torch.allclose(result.lam_max[0], expected / F, rtol=1e-10)
    assert result.failures.sum() == 0
    summary = windup_summary(result)
    assert summary["log10_trace_growth"] > 1.0
    assert summary["lam_max_max"] == pytest.approx((expected / F).max().item())


def test_trace_limit_failure_is_counted_reset_and_reported():
    E, T, F = 1, 12, 2
    phi = torch.zeros(E, T, F, dtype=DOUBLE)
    v = torch.zeros(E, T, 1, dtype=DOUBLE)
    W0 = torch.zeros(E, 1, F, dtype=DOUBLE)
    # gamma = 0.5 doubles P each step from trace 2: 4, 8, 16, 32, 64, 128 at t = 5, which breaks the limit.
    result = run_prequential(phi, v, W0, StreamPolicy("f", 1.0, forgetting=0.5), trace_limit=100.0)
    assert result.first_failure.item() == 5
    assert result.trace[0, 5].item() == pytest.approx(128.0)  # the record keeps the value that failed
    assert result.trace[0, 6].item() == pytest.approx(4.0)  # back at the prior, then one more write
    assert result.failures.item() == 2  # again at t = 11
    summary = windup_summary(result)
    assert summary["n_failures"] == 2
    assert summary["failed_stream_fraction"] == 1.0
    assert summary["first_failure_step"] == 5
    assert summary["trace_max"] == pytest.approx(128.0)


def test_non_finite_data_is_counted_and_the_stream_keeps_going():
    phi, v, W0 = _problem(7, T=20)
    phi[0, 5] = float("nan")
    low = torch.full_like(v, -50.0)
    high = torch.full_like(v, 50.0)
    result = run_prequential(phi, v, W0, StreamPolicy("p", 0.5), bounds=(low, high))
    assert result.failures.tolist() == [1, 0]
    assert result.first_failure.tolist() == [5, -1]
    assert result.clipped[0, 5]  # the NaN prediction was recorded as clipped, not dropped
    assert torch.isfinite(result.err).all()
    # The healthy stream is untouched by the other one's failure.
    clean = run_prequential(phi[1:], v[1:], W0[1:], StreamPolicy("p", 0.5), bounds=(low[1:], high[1:]))
    assert torch.allclose(result.err[1], clean.err[0], atol=1e-10)


def test_default_trace_limit_leaves_ordinary_forgetting_alone():
    phi, v, W0 = _problem(8, T=200)
    result = run_prequential(phi, v, W0, StreamPolicy("f", 1.0, forgetting=0.98), eig_every=20)
    assert result.failures.sum() == 0
    assert result.trace.max() < P_TRACE_LIMIT
    assert (result.lam_min > 0).all()


def test_clipping_bounds_a_diverging_readout():
    phi, v, W0 = _problem(9, T=5)
    huge = W0 * 1e8
    low, high = torch.full_like(v, -1.0), torch.full_like(v, 1.0)
    target = torch.zeros_like(v)
    result = run_prequential(phi, target, huge, StreamPolicy("frozen", 1.0, write=False), bounds=(low, high))
    assert result.clipped.all()
    assert torch.allclose(result.err, torch.ones_like(result.err))


# ---------------------------------------------------------------- schedules and streams


def test_regime_sequences_have_the_right_switch_times():
    gen = _gen(10)
    E, T, L = 3, 200, 50
    none = regime_sequence(Schedule("n", "none"), E, T, L, gen)
    assert (none == 0).all()
    fixed = regime_sequence(Schedule("f", "fixed", interval=35), E, T, L, gen)
    assert fixed[0, 34] == 0 and fixed[0, 35] == 1 and fixed[0, 70] == 2
    aligned = regime_sequence(Schedule("a", "aligned"), E, T, L, gen)
    assert aligned[0, 49] == 0 and aligned[0, 50] == 1
    cycle = regime_sequence(Schedule("c", "cycle", interval=40, cycle_len=2), E, T, L, gen)
    assert cycle[0, 0] == 0 and cycle[0, 40] == 1 and cycle[0, 80] == 0 and cycle[0, 120] == 1
    schedule = Schedule("v", "varied", min_interval=20, max_interval=60)
    varied = regime_sequence(schedule, E, T, L, gen)
    assert not torch.equal(varied[0], varied[1])  # streams switch at different times
    for s in range(E):
        changes = (varied[s, 1:] != varied[s, :-1]).nonzero().flatten() + 1
        gaps = changes[1:] - changes[:-1]
        assert (gaps >= 20).all() and (gaps <= 60).all()
        assert varied[s].max() + 1 <= n_regimes_needed(schedule, T, L)


def test_build_stream_gives_distinct_regimes_and_real_returns():
    E, T = 3, 130
    fixed = build_stream("pendulum", "interp", Schedule("f", "fixed", interval=40), E, T, 0, "t1")
    for s in range(E):
        changes = [0] + (fixed.switch[s]).nonzero().flatten().tolist()
        ids = fixed.env_ids[s, changes].tolist()
        assert len(set(ids)) == len(ids)
    assert len(set(fixed.env_ids[:, 0].tolist())) == E  # no two streams share a regime
    back = build_stream("pendulum", "interp", Schedule("c", "cycle", interval=40, cycle_len=2), E, T, 0, "t2")
    assert (back.env_ids[:, 0] == back.env_ids[:, 80]).all()
    assert (back.env_ids[:, 0] != back.env_ids[:, 40]).all()
    none = build_stream("pendulum", "interp", Schedule("n", "none"), E, T, 0, "t3")
    assert not none.switch.any()


def test_simulated_transitions_follow_the_regime_of_their_step():
    E, T = 3, 90
    schedule = Schedule("f", "fixed", interval=25)
    pool = make_env("pendulum", E * 4, "interp", "sim")
    table = torch.randperm(E * 4, generator=_gen(11)).reshape(E, 4)
    regimes = regime_sequence(schedule, E, T, pool.episode_steps, _gen(12))
    env_ids = table.gather(1, regimes)
    stream = simulate_stream(pool, env_ids, make_generator("sim_run"))
    for t in (0, 24, 25, 26, 50, 75):
        env = subset_env(pool, env_ids[:, t])
        state = env.state_from_obs(stream.obs[:, t][:, None])
        next_obs = env.obs(env.step(state, stream.act[:, t][:, None]))[:, 0]
        assert torch.allclose(next_obs, stream.next_obs[:, t], atol=1e-3)
    # The transition at a switch runs under the new regime, not the old one.
    t = 25
    old = subset_env(pool, env_ids[:, t - 1])
    state = old.state_from_obs(stream.obs[:, t][:, None])
    wrong = old.obs(old.step(state, stream.act[:, t][:, None]))[:, 0]
    assert (wrong - stream.next_obs[:, t]).abs().max() > 1e-3


def test_state_carries_across_a_switch_and_resets_at_an_episode():
    E, T = 2, 130
    stream = build_stream("pendulum", "interp", Schedule("f", "fixed", interval=25), E, T, 0, "t4")
    L = make_env("pendulum", 1, "train", "bounds").episode_steps
    assert stream.switch[:, 25].all()
    for t in range(1, T):
        if t % L != 0:
            assert torch.equal(stream.obs[:, t], stream.next_obs[:, t - 1])
        else:
            assert not torch.equal(stream.obs[:, t], stream.next_obs[:, t - 1])
            assert stream.episode_start[:, t].all()


def test_masks_cover_the_right_steps():
    stream = build_stream("pendulum", "interp", Schedule("f", "fixed", interval=30), 2, 100, 0, "t5")
    post = post_switch_mask(stream, 5)
    assert not post[:, :30].any()  # the start of the stream is not a switch
    assert post[:, 30:35].all() and not post[:, 35:60].any() and post[:, 60:65].all()
    short = post_switch_mask(stream, 50)  # cut short by the next switch, still only steps after one
    assert short[:, 30:].all()
    steady = steady_mask(stream, 20)
    assert steady[:, 20:30].all() and not steady[:, 30:50].any() and steady[:, 50:60].all()


def test_rmse_over_is_nan_only_for_an_empty_mask():
    err = torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=DOUBLE)
    mask = torch.tensor([[True, False], [False, False]])
    out = rmse_over(err, mask)
    assert out[0].item() == pytest.approx(1.0)
    assert torch.isnan(out[1])
    assert rmse_over(err)[0].item() == pytest.approx(np.sqrt(2.5))


def test_rollout_windows_stay_inside_one_episode():
    starts = rollout_starts(200, 50, 10, 5)
    assert starts
    assert all(t % 50 + 10 <= 50 for t in starts)
    assert 40 in starts and 45 not in starts
    assert rollout_starts(5, 50, 10, 1) == []


def test_episode_reset_equals_oracle_reset_on_the_aligned_schedule():
    stream = build_stream("pendulum", "interp", Schedule("a", "aligned"), 2, 130, 0, "t6")
    phi, v, W0 = _problem(13, E=2, T=130)
    a = run_prequential(phi, v, W0, thread_b.make_policy("episode_reset", {"lambda_": 0.5}, stream))
    b = run_prequential(phi, v, W0, thread_b.make_policy("oracle_reset", {"lambda_": 0.5}, stream))
    assert torch.equal(a.err, b.err)


# ---------------------------------------------------------------- verdict logic


def _stat(mean, low=None, high=None):
    return {"mean": mean, "ci_low": low if low is not None else mean - 0.01, "ci_high": high if high is not None else mean + 0.01}


def _h4_tier(mean, partial_low=0.1, verdict="a_wins", reference=0.7):
    return {
        "within_P": _stat(mean),
        "partial_phi_norm": _stat(partial_low + 0.05, low=partial_low),
        "vs_phi_norm": {"verdict": verdict},
        "well_specified_reference": _stat(reference),
    }


def test_h4_outcome_rules():
    good = {"interp": _h4_tier(0.6), "extrap": _h4_tier(0.55)}
    assert thread_b.h4_outcome(good)["outcome"] == "supported"
    # Raw target met, but P adds nothing beyond the norm: that must not read as support.
    norm = {"interp": _h4_tier(0.6, verdict="no_detectable_difference"), "extrap": _h4_tier(0.55)}
    assert thread_b.h4_outcome(norm)["outcome"] == "target_met_but_not_beyond_norm"
    no_partial = {"interp": _h4_tier(0.6, partial_low=-0.01), "extrap": _h4_tier(0.55)}
    assert thread_b.h4_outcome(no_partial)["outcome"] == "target_met_but_not_beyond_norm"
    killed = thread_b.h4_outcome({"interp": _h4_tier(0.29), "extrap": _h4_tier(0.9)})
    assert killed["outcome"] == "killed"
    assert killed["killed_h5_h6_end_as_stated"]
    assert thread_b.h4_outcome({"interp": _h4_tier(0.4), "extrap": _h4_tier(0.6)})["outcome"] == "below_target"
    assert thread_b.h4_outcome({"interp": _h4_tier(0.6), "extrap": _h4_tier(0.45)})["outcome"] == "below_target"
    assert thread_b.h4_outcome({})["outcome"] == "incomplete"


def test_h4_outcome_flags_sensitivity_and_reachability():
    tier = _h4_tier(0.31, reference=0.2)
    tier["within_P"] = _stat(0.31, low=0.25, high=0.4)
    out = thread_b.h4_outcome({"interp": tier, "extrap": _h4_tier(0.6)})
    assert out["ci_straddles_threshold_add_seeds"]["interp"]
    assert not out["reachability_if_well_specified"]["interp"]["kill_threshold_reachable"]


def _h10_cell(vs_reset="a_wins", vs_persist="a_wins", vs_surprise="no_detectable_difference", tainted=False, persist_worst=True):
    return {"vs_reset": vs_reset, "vs_persist": vs_persist, "vs_surprise": vs_surprise, "tainted": tainted, "persist_worst": persist_worst}


def test_h10_outcome_rules():
    assert thread_b.h10_tier_outcome({"a": _h10_cell(), "b": _h10_cell()})["outcome"] == "supported"
    # Kill: reset matches forgetting everywhere, or beats it.
    killed = {"a": _h10_cell(vs_reset="no_detectable_difference"), "b": _h10_cell(vs_reset="b_wins")}
    assert thread_b.h10_tier_outcome(killed)["outcome"] == "killed"
    partial = {"a": _h10_cell(), "b": _h10_cell(vs_reset="no_detectable_difference")}
    out = thread_b.h10_tier_outcome(partial)
    assert out["outcome"] == "partial"
    assert out["forgetting_beats_reset"] == ["a"]
    assert thread_b.h10_tier_outcome({"a": _h10_cell(vs_reset="insufficient_seeds")})["outcome"] == "insufficient_seeds"


def test_h10_a_win_with_numerical_failure_does_not_count():
    out = thread_b.h10_tier_outcome({"a": _h10_cell(tainted=True), "b": _h10_cell(vs_reset="no_detectable_difference")})
    assert out["outcome"] == "inconclusive_numerical_failure"  # neither a win nor evidence for the kill
    assert out["forgetting_beats_reset"] == []
    assert out["tainted_schedules"] == ["a"]
    clean_win = thread_b.h10_tier_outcome({"a": _h10_cell(tainted=True), "b": _h10_cell()})
    assert clean_win["outcome"] == "partial"
    assert clean_win["forgetting_beats_both"] == ["b"]


def test_selection_skips_failing_entries_and_flags_when_nothing_is_clean():
    clean = {"hp": {"x": 1}, "score": 0.9, "failed_stream_fraction": 0.0}
    better_but_failing = {"hp": {"x": 2}, "score": 0.5, "failed_stream_fraction": 0.1}
    picked = thread_b._pick([better_but_failing, clean])
    assert picked["hp"] == {"x": 1}
    assert not picked["all_entries_failed"]
    picked = thread_b._pick([better_but_failing])
    assert picked["hp"] == {"x": 2}
    assert picked["all_entries_failed"]


# ---------------------------------------------------------------- the pipeline on tiny settings


TINY_TRAIN = TrainConfig(model_steps=30, warmup_steps=10, block_steps=10, batch_size=32, log_every=10, context_envs_per_step=4)


def _tiny_a():
    return thread_a.ThreadAConfig(
        mode="test", seeds=[0], z_dim=2, hidden=16, feature_dim=8, train=TINY_TRAIN, train_envs=6, train_traj=2,
        train_steps=10, n_eval_envs=4, n_val_envs=3, adapt_steps=10, holdout_traj=2, n_boot=100,
    )


def _tiny_b():
    return thread_b.ThreadBConfig(
        mode="test",
        a=_tiny_a(),
        h10_tiers=["interp"],
        schedules=[Schedule("none", "none"), Schedule("medium", "fixed", interval=45), Schedule("aligned", "aligned")],
        stream_steps=130,
        n_streams=3,
        n_val_streams=3,
        forgetting_grid=[0.98],
        surprise_thresholds=[3.0],
    )


def test_backbones_match_thread_a():
    a = _tiny_a()
    ours, _ = thread_b.train_backbones("pendulum", 0, a)
    theirs, _ = thread_a.train_all("pendulum", 0, a)
    for name in ("latent", "nolatent"):
        mine, other = ours[name].model.state_dict(), theirs[name].model.state_dict()
        assert mine.keys() == other.keys()
        for key in mine:
            assert torch.equal(mine[key], other[key]), f"{name}.{key}"


def test_calibration_data_is_validation_only_and_shares_thread_a_holdout():
    a = _tiny_a()
    assert {cv.tier for cv in thread_b.calibration_val_sets("pendulum", 0, a)} == {"val_in", "val_shift"}
    theirs = thread_a.val_sets_for("pendulum", 0, a, "smooth")
    mine = thread_b.calibration_val_sets("pendulum", 0, a)
    for cv, val in zip(mine, theirs):
        assert torch.equal(cv.holdout.obs, val.holdout.flat().obs)


def test_stream_selection_sees_validation_tiers_only(nolatent_backbone):
    env = make_env("pendulum", 1, "train", "bounds")
    cfg = _tiny_b()
    val = thread_b.validation_streams("pendulum", 0, nolatent_backbone, env.obs_low, env.obs_high, cfg)
    assert [fs.stream.tier for fs in val] == ["val_in", "val_shift"]
    selection = thread_b.select_stream_methods(val, cfg)
    for method in thread_b.H10_METHODS + thread_b.H10_REFERENCES:
        assert "hp" in selection[method]
    assert selection["forgetting"]["hp"]["forgetting"] == 0.98
    assert selection["surprise_reset"]["hp"]["threshold"] == 3.0


@pytest.fixture(scope="module")
def tiny_job():
    return thread_b.run_job("pendulum", 0, _tiny_b())


def test_job_records_what_the_report_needs(tiny_job):
    assert tiny_job["rls_backbone"] in ("latent", "nolatent")
    cell = tiny_job["h4"]["latent"]["tiers"]["interp"]["rollout"]["1"]
    for name in ("P", "phi_norm", "input_norm", "pred_norm"):
        assert len(cell["within"][name]["per_env"]) == 4
    assert "P|phi_norm" in cell["partial"]
    assert "adapt_surprise" in cell["across"]
    assert set(cell["calibration"]) == set(thread_b.VARIANCE_MODELS)
    assert 0.0 <= cell["calibration"]["rls"]["coverage"]["0.9"] <= 1.0
    assert set(tiny_job["h4"]["latent"]["tiers"]) == {"interp", "extrap", "oof"}
    assert cell["by_mechanism"] == {}
    assert tiny_job["h4"]["latent"]["tiers"]["oof"]["rollout"]["1"]["by_mechanism"]

    methods = tiny_job["h10"]["tiers"]["interp"]["medium"]["methods"]
    assert set(methods) == set(thread_b.H10_METHODS + thread_b.H10_REFERENCES)
    forgetting = methods["forgetting"]
    assert forgetting["hp"]["forgetting"] == 0.98
    assert forgetting["windup"]["log10_trace_mean"]
    assert "lam_max_max" in forgetting["windup"]
    assert forgetting["post_switch_20"]["mean"] == forgetting["post_switch_20"]["mean"]  # not NaN with switches
    assert thread_b.ROLLOUT_KEY in forgetting
    # The stationarity control has no switch, so its post-switch metric is undefined, not zero.
    none = tiny_job["h10"]["tiers"]["interp"]["none"]["methods"]["persist"]
    assert none["post_switch_20"]["mean"] != none["post_switch_20"]["mean"]
    assert tiny_job["h10"]["tiers"]["interp"]["none"]["n_switches_per_stream"] == 0
    assert methods["frozen"]["windup"]["log10_trace_growth"] == 0.0


def test_summary_runs_serializes_and_states_the_primary(tiny_job):
    # The same job twice checks the plumbing only; a repeated seed gives a degenerate CI, so verdicts mean nothing.
    jobs = {"0": tiny_job, "1": tiny_job}
    summary = thread_b.summarize(jobs, 100)
    assert "within-environment" in summary["H4"]["primary_ranking"]
    outcome = summary["H4"]["outcomes"]["selected"]["rollout"]["1"]
    assert outcome["outcome"] in ("killed", "supported", "target_met_but_not_beyond_norm", "below_target")
    assert set(outcome["adds_beyond_norm"]) == {"interp", "extrap"}
    h10 = summary["H10"]["outcomes"]["interp"]
    assert h10["outcome"] in ("killed", "supported", "partial", "inconclusive_numerical_failure")
    assert h10["n_switching_schedules"] == 2  # none is a control and decides nothing
    table = summary["H10"]["tiers"]["interp"]["medium"]
    assert table["comparisons"]["prequential"]["forgetting_vs_reset"]["verdict"] in ("a_wins", "b_wins", "no_detectable_difference")
    assert table["means"]["persist"]["prequential"]["n_seeds"] == 2
    assert "post_switch_curve" not in table["means"]["persist"]
    assert summary["H10"]["tiers"]["interp"]["none"]["comparisons"]["post_switch_20"]["forgetting_vs_reset"]["verdict"] == "undefined"
    json.dumps(thread_a._jsonable(summary), allow_nan=False)
    json.dumps(thread_a._jsonable(tiny_job), allow_nan=False)


def test_one_seed_is_reported_as_insufficient(tiny_job):
    summary = thread_b.summarize({"0": tiny_job}, 100)
    assert summary["H10"]["outcomes"]["interp"]["outcome"] == "insufficient_seeds"


def test_configs_build_and_keep_the_protocol_sizes():
    full = thread_b.full_b_config()
    assert len(full.a.seeds) >= 5
    assert full.a.n_eval_envs >= 20 and full.n_streams >= 20
    assert thread_b.PRIMARY_POST_N in full.post_switch_ns
    assert any(s.kind == "none" for s in full.schedules)
    assert any(s.kind == "cycle" for s in full.schedules)
    assert full.a.train == thread_a.full_config().train
    smoke = thread_b.smoke_b_config()
    assert smoke.a.train == thread_a.smoke_config().train


def test_fingerprint_covers_this_thread():
    hashes = thread_b.fingerprint()
    assert "experiments/thread_b.py" in hashes
    assert "experiments/thread_a.py" in hashes
    assert "src/intr/calibration.py" in hashes
