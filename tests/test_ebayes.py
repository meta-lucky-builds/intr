"""Empirical-Bayes RLS (H12) and prior-preserving forgetting (H13), against brute force and known answers."""

import math

import pytest
import torch

import intr.ebayes as eb
from intr.adapt import RLSState, adapt
from intr.calibration import StreamPolicy, run_prequential
from intr.envs import collect_trajectories, make_env, make_generator

DOUBLE = torch.float64
SHAPES = [(6, 10), (10, 4)]  # N < F and N > F


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randn(gen, *shape):
    return torch.randn(*shape, generator=gen, dtype=DOUBLE)


def _problem(N, F, D=2, E=3, seed=0):
    gen = _gen(seed)
    return _randn(gen, E, N, F), _randn(gen, E, N, D)


def _brute_log_marginal(phi, r, lam, sigma2):
    """One environment, by the full covariance. phi (N, F), r (N, D), sigma2 (D,)."""
    N = phi.shape[0]
    C = torch.eye(N, dtype=DOUBLE) + phi @ phi.t() / lam
    total = 0.0
    for d in range(r.shape[1]):
        cov = sigma2[d] * C
        sign, logdet = torch.linalg.slogdet(cov)
        assert sign > 0
        total -= 0.5 * (N * math.log(2 * math.pi) + logdet + r[:, d] @ torch.linalg.solve(cov, r[:, d]))
    return total.item()


# ---------------------------------------------------------------- evidence formula


@pytest.mark.parametrize("N,F", SHAPES)
@pytest.mark.parametrize("sigma_mode", eb.SIGMA_MODES)
def test_evidence_matches_the_full_covariance(N, F, sigma_mode):
    phi, r = _problem(N, F)
    lambdas = [0.01, 0.3, 1.0, 50.0]
    grid = eb.evidence_grid(phi, r, lambdas, sigma_mode)
    for e in range(phi.shape[0]):
        for i, lam in enumerate(lambdas):
            brute = _brute_log_marginal(phi[e], r[e], lam, grid.sigma2[e, i])
            assert grid.log_evidence[e, i].item() == pytest.approx(brute, abs=1e-8)


@pytest.mark.parametrize("N,F", SHAPES)
def test_evidence_at_any_given_sigma_matches_too(N, F):
    phi, r = _problem(N, F, seed=1)
    spec = eb.spectrum(phi, r)
    lam = torch.full((phi.shape[0], 1), 0.7, dtype=DOUBLE)
    quad, logdet = eb.quadratic_and_logdet(spec, lam)
    sigma2 = torch.tensor([0.4, 2.5], dtype=DOUBLE)
    values = eb.log_marginal(quad[:, 0], logdet[:, 0], sigma2, N)
    for e in range(phi.shape[0]):
        assert values[e].item() == pytest.approx(_brute_log_marginal(phi[e], r[e], 0.7, sigma2), abs=1e-8)


@pytest.mark.parametrize("N,F", SHAPES)
def test_dual_and_primal_paths_agree(N, F):
    phi, r = _problem(N, F, seed=2)
    lambdas = [0.01, 1.0, 100.0]
    dual = eb.evidence_grid(phi, r, lambdas, path="dual")
    primal = eb.evidence_grid(phi, r, lambdas, path="primal")
    assert dual.spectrum.path == "dual" and primal.spectrum.path == "primal"
    assert torch.allclose(dual.log_evidence, primal.log_evidence, atol=1e-7, rtol=0.0)
    W0 = torch.zeros(phi.shape[0], r.shape[2], F, dtype=DOUBLE)
    lam = torch.full((phi.shape[0],), 0.1, dtype=DOUBLE)
    W_d, P_d = eb.posterior(dual.spectrum, W0, lam)
    W_p, P_p = eb.posterior(primal.spectrum, W0, lam)
    assert torch.allclose(W_d, W_p, atol=1e-7, rtol=0.0)
    assert torch.allclose(P_d, P_p, atol=1e-7, rtol=0.0)


def test_auto_path_takes_the_smaller_matrix():
    assert eb.spectrum(*_problem(6, 10)).path == "dual"
    assert eb.spectrum(*_problem(10, 4)).path == "primal"


@pytest.mark.parametrize("N,F", SHAPES)
def test_posterior_is_the_state_rls_reaches(N, F):
    phi, v = _problem(N, F, seed=3)
    W0 = _randn(_gen(4), phi.shape[0], v.shape[2], F)
    rls = RLSState(W0, 0.5)
    rls.write_sequence(phi, v)
    fit = eb.fit_empirical_bayes(phi, v, W0, lambdas=[0.5])
    assert torch.allclose(fit.W, rls.W, atol=1e-8, rtol=0.0)
    assert torch.allclose(fit.P, rls.P, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("N,F", SHAPES)
@pytest.mark.parametrize("sigma_mode", eb.SIGMA_MODES)
def test_chain_rule_gives_the_same_evidence_in_time_order(N, F, sigma_mode):
    """The evidence is also the product of one-step predictive densities, which sequential RLS computes independently."""
    phi, v = _problem(N, F, seed=5)
    W0 = _randn(_gen(6), phi.shape[0], v.shape[2], F)
    fit = eb.fit_empirical_bayes(phi, v, W0, lambdas=[0.01, 0.1, 1.0, 10.0], sigma_mode=sigma_mode)
    checks = eb.innovation_diagnostics(phi, v, W0, fit.lambda_, fit.sigma2)
    assert torch.allclose(torch.tensor(checks["chain_log_evidence"], dtype=DOUBLE), fit.log_evidence, atol=1e-6, rtol=0.0)


def test_profiled_sigma_is_a_maximum():
    phi, r = _problem(8, 5, seed=7)
    spec = eb.spectrum(phi, r)
    lam = torch.full((phi.shape[0], 1), 1.0, dtype=DOUBLE)
    quad, logdet = eb.quadratic_and_logdet(spec, lam)
    for mode in eb.SIGMA_MODES:
        best = eb.profile_sigma2(quad[:, 0], 8, mode)
        at_best = eb.log_marginal(quad[:, 0], logdet[:, 0], best, 8)
        for factor in (0.8, 1.25):
            assert (eb.log_marginal(quad[:, 0], logdet[:, 0], best * factor, 8) < at_best).all()
    # And per_output can only fit at least as well as shared.
    shared = eb.log_marginal(quad[:, 0], logdet[:, 0], eb.profile_sigma2(quad[:, 0], 8, "shared"), 8)
    per_output = eb.log_marginal(quad[:, 0], logdet[:, 0], eb.profile_sigma2(quad[:, 0], 8, "per_output"), 8)
    assert (per_output >= shared - 1e-10).all()


def test_no_write_entry_is_the_limit_of_the_grid():
    phi, r = _problem(20, 5, seed=8)
    grid = eb.evidence_grid(phi, r, [eb.NO_WRITE_LAMBDA])
    assert torch.allclose(grid.log_evidence[:, 0], grid.no_write_log_evidence, atol=1e-6, rtol=0.0)


def test_sigma_modes_coincide_with_one_output():
    phi, r = _problem(9, 4, D=1, seed=9)
    a = eb.evidence_grid(phi, r, [0.1, 1.0], "shared")
    b = eb.evidence_grid(phi, r, [0.1, 1.0], "per_output")
    assert torch.allclose(a.log_evidence, b.log_evidence, atol=1e-10, rtol=0.0)


def test_per_output_choice_ignores_output_units_and_shared_is_the_registered_default():
    phi, v = _problem(30, 6, seed=10)
    v = v + phi @ _randn(_gen(11), 3, 2, 6).transpose(1, 2)  # some signal
    W0 = torch.zeros(3, 2, 6, dtype=DOUBLE)
    scaled = v * torch.tensor([100.0, 1.0], dtype=DOUBLE)
    a = eb.fit_empirical_bayes(phi, v, W0, sigma_mode="per_output")
    b = eb.fit_empirical_bayes(phi, scaled, W0, sigma_mode="per_output")
    assert torch.equal(a.index, b.index)
    assert eb.fit_empirical_bayes(phi, v, W0).sigma_mode == "shared"


# ---------------------------------------------------------------- what the evidence chooses


def test_predictable_residuals_get_a_strong_write_with_more_data_than_features():
    E, N, F, D = 8, 60, 8, 2
    gen = _gen(20)
    phi = _randn(gen, E, N, F)
    W_true = _randn(gen, E, D, F)
    v = phi @ W_true.transpose(1, 2) + 0.01 * _randn(gen, E, N, D)
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE))
    assert fit.edf.median().item() > 0.9 * F
    assert ((fit.log_evidence - fit.no_write_log_evidence) / N).median().item() > 1.0
    assert (fit.W - W_true).abs().max().item() < 0.05
    assert fit.sigma2.max().item() < 1e-2


def _decaying_features(gen, E, N, F):
    """Real features have a wide spectrum. A flat one leaves lambda unidentified (see the flat-spectrum test)."""
    scales = torch.logspace(0.0, -2.0, F, dtype=DOUBLE)
    return _randn(gen, E, N, F) * scales


def test_predictable_residuals_get_a_strong_write_with_fewer_data_than_features():
    E, N, F, D = 8, 12, 24, 2
    gen = _gen(21)
    phi = _decaying_features(gen, E, N, F)
    v = phi @ _randn(gen, E, D, F).transpose(1, 2) + 1e-3 * _randn(gen, E, N, D)
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE))
    assert fit.path == "dual"
    left = ((v - phi @ fit.W.transpose(1, 2)) ** 2).sum(dim=(1, 2)) / (v**2).sum(dim=(1, 2))
    assert left.median().item() < 0.5
    assert ((fit.log_evidence - fit.no_write_log_evidence) / N).median().item() > 0.5


def test_pure_noise_gets_no_write_with_more_data_than_features():
    E, N, F, D = 200, 60, 8, 2
    gen = _gen(22)
    phi = _randn(gen, E, N, F)
    v = _randn(gen, E, N, D)  # nothing to do with phi
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE))
    # The null sits on the boundary, so about half the environments go exactly to no write and the rest write a little.
    assert fit.edf.mean().item() < 0.35 * F
    assert (fit.edf < 0.5).double().mean().item() > 0.3
    assert ((fit.log_evidence - fit.no_write_log_evidence) / N).mean().item() < 0.1


def test_pure_noise_gets_a_weak_write_with_fewer_data_than_features():
    E, N, F, D = 100, 12, 24, 2
    gen = _gen(23)
    phi = _decaying_features(gen, E, N, F)
    v = _randn(gen, E, N, D)
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE))
    assert fit.edf.mean().item() < 0.35 * N
    assert ((fit.log_evidence - fit.no_write_log_evidence) / N).mean().item() < 0.25


def test_flat_spectrum_leaves_lambda_unidentified_and_ties_go_to_no_write():
    """A stated risk: profiled evidence is exactly flat when Phi Phi' is a multiple of I. The width shows it."""
    E, N, F, D = 2, 4, 8, 2
    phi = torch.zeros(E, N, F, dtype=DOUBLE)
    diagonal = torch.arange(N)
    phi[:, diagonal, diagonal] = 3.0
    v = _randn(_gen(24), E, N, D)
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE))
    assert (fit.index == len(eb.DEFAULT_LAMBDAS) - 1).all()
    span = math.log10(max(eb.DEFAULT_LAMBDAS)) - math.log10(min(eb.DEFAULT_LAMBDAS))
    assert fit.width_2nats.min().item() == pytest.approx(span, abs=1e-9)


def test_exact_prior_keeps_w0_and_stays_finite():
    E, N, F, D = 3, 10, 4, 2
    gen = _gen(25)
    phi = _randn(gen, E, N, F)
    W0 = _randn(gen, E, D, F)
    fit = eb.fit_empirical_bayes(phi, phi @ W0.transpose(1, 2), W0)  # residual exactly zero
    assert torch.isfinite(fit.log_evidence).all()
    assert (fit.sigma2 == eb.SIGMA2_FLOOR).all()
    assert (fit.index == len(eb.DEFAULT_LAMBDAS) - 1).all()
    assert torch.allclose(fit.W, W0, atol=1e-12, rtol=0.0)


def test_selected_values_lie_in_the_grid_and_are_reported():
    E, N, F, D = 5, 15, 6, 2
    gen = _gen(26)
    phi = _randn(gen, E, N, F)
    v = phi @ _randn(gen, E, D, F).transpose(1, 2) + 0.1 * _randn(gen, E, N, D)
    grid = [0.1, 1.0, 10.0]
    fit = eb.fit_empirical_bayes(phi, v, torch.zeros(E, D, F, dtype=DOUBLE), lambdas=grid)
    assert all(lam in grid for lam in fit.lambda_.tolist())
    info = fit.info()
    for key in ("lambda", "log10_lambda", "sigma2", "edf", "no_write", "at_lower_edge", "at_upper_edge", "log_bf_vs_no_write_per_transition", "log10_lambda_width_2nats"):
        assert len(info[key]) == E, key
    assert (info["grid_low"], info["grid_high"], info["n_transitions"], info["feature_dim"]) == (0.1, 10.0, N, F)
    # sigma = 0.1 and prior scale 1 put the answer at 0.01, under this grid: the edge has to show it.
    assert sum(info["at_lower_edge"]) >= E - 1
    assert not any(info["at_upper_edge"])


def test_default_grid_is_wider_than_round_one_on_both_sides_and_ascends():
    grid = eb.DEFAULT_LAMBDAS
    assert list(grid) == sorted(grid)
    assert min(grid) < 1e-3 and max(grid) > 100.0 and max(grid) == eb.NO_WRITE_LAMBDA


def test_innovations_expose_correlated_noise():
    E, N, F, D = 16, 200, 3, 1
    gen = _gen(27)
    phi = _randn(gen, E, N, F)
    iid = _randn(gen, E, N, D)
    ar = torch.zeros(E, N, D, dtype=DOUBLE)
    for t in range(1, N):
        ar[:, t] = 0.9 * ar[:, t - 1] + math.sqrt(1 - 0.81) * iid[:, t]
    W0 = torch.zeros(E, D, F, dtype=DOUBLE)
    lam = torch.full((E,), 1e3, dtype=DOUBLE)
    sigma2 = torch.ones(E, D, dtype=DOUBLE)
    white = eb.innovation_diagnostics(phi, iid, W0, lam, sigma2)
    corr = eb.innovation_diagnostics(phi, ar, W0, lam, sigma2)
    assert abs(sum(white["lag1_autocorr"]) / E) < 0.1
    assert sum(corr["lag1_autocorr"]) / E > 0.6
    assert sum(corr["n_eff"]) / E < 0.3 * N


def test_logdet_gain_and_covariance_match_direct_computation():
    E, N, F = 2, 7, 4
    phi = _randn(_gen(28), E, N, F)
    A = torch.eye(F, dtype=DOUBLE) * 0.5 + phi.transpose(1, 2) @ phi
    direct = torch.linalg.slogdet(A)[1] - F * math.log(0.5)
    assert torch.allclose(eb.logdet_precision_gain(phi, 0.5), direct, atol=1e-9, rtol=0.0)
    assert torch.allclose(eb.posterior_covariance(phi, 0.5), torch.linalg.inv(A), atol=1e-9, rtol=0.0)
    wide = _randn(_gen(29), E, 3, 6)  # the dual path
    A = torch.eye(6, dtype=DOUBLE) * 0.5 + wide.transpose(1, 2) @ wide
    assert torch.allclose(eb.posterior_covariance(wide, 0.5), torch.linalg.inv(A), atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------- registry


def _adapt_data(n_traj=1, n_steps=20):
    env = make_env("pendulum", 5, "interp", 1)
    return collect_trajectories(env, n_traj, n_steps, env.default_probe, make_generator("ebayes_test", n_traj)).flat()


@pytest.mark.parametrize("n_traj", [1, 3])  # 20 and 60 transitions against 24 features, so both paths run
def test_ebrls_runs_on_both_backbones_and_reports(n_traj, latent_backbone, nolatent_backbone):
    data = _adapt_data(n_traj)
    for backbone in (latent_backbone, nolatent_backbone):
        result = adapt("ebrls", backbone, data)
        info = result.info["eb"]
        assert info["path"] == ("dual" if n_traj == 1 else "primal")
        assert info["n_transitions"] == 20 * n_traj
        assert all(lam in eb.DEFAULT_LAMBDAS for lam in info["lambda"])
        pred = result.predictor.predict_delta(data.obs, data.act)
        assert torch.isfinite(pred).all()
        assert result.counts.backward_calls == 0
        # It is a normal RLS state, so it can keep writing.
        result.predictor.observe(data.obs[:, 0], data.act[:, 0], data.next_obs[:, 0])
        assert result.predictor.variance(data.obs, data.act).shape == data.obs.shape[:2]


def test_ebrls_equals_rls_at_a_single_lambda(nolatent_backbone):
    data = _adapt_data()
    ours = adapt("ebrls", nolatent_backbone, data, lambdas=[1.0]).predictor
    theirs = adapt("rls", nolatent_backbone, data, lambda_=1.0).predictor
    assert torch.allclose(ours.rls.W, theirs.rls.W, atol=1e-7, rtol=0.0)
    assert torch.allclose(ours.rls.P, theirs.rls.P, atol=1e-7, rtol=0.0)


def test_hybrid_eb_is_latent_opt_then_ebrls(latent_backbone):
    data = _adapt_data()
    latent = adapt("latent_opt", latent_backbone, data, lr_scale=2.0).predictor
    hybrid = adapt("hybrid_eb", latent_backbone, data, lr_scale=2.0)
    assert torch.equal(hybrid.predictor.z, latent.z)
    assert "z_saturation" in hybrid.info and "eb" in hybrid.info
    assert hybrid.counts.backward_calls == 40
    # With the write switched off, it is latent_opt, and ebrls is frozen.
    no_write = adapt("hybrid_eb", latent_backbone, data, lr_scale=2.0, lambdas=[eb.NO_WRITE_LAMBDA]).predictor
    assert torch.allclose(no_write.predict_delta(data.obs, data.act), latent.predict_delta(data.obs, data.act), atol=1e-4, rtol=0.0)
    frozen = adapt("frozen", latent_backbone, data).predictor
    off = adapt("ebrls", latent_backbone, data, lambdas=[eb.NO_WRITE_LAMBDA]).predictor
    assert torch.allclose(off.predict_delta(data.obs, data.act), frozen.predict_delta(data.obs, data.act), atol=1e-4, rtol=0.0)


# ---------------------------------------------------------------- H13


def _stream(E=2, T=30, F=4, D=2, seed=30):
    gen = _gen(seed)
    return _randn(gen, E, T, F), _randn(gen, E, T, D), _randn(gen, E, D, F)


def test_prior_preserving_with_gamma_one_is_exactly_rls():
    phi, v, W0 = _stream()
    a = eb.PriorPreservingRLS(W0, 0.5, gamma=1.0)
    b = RLSState(W0, 0.5)
    a.write_sequence(phi, v)
    b.write_sequence(phi, v)
    assert torch.equal(a.W, b.W)
    assert torch.equal(a.P, b.P)


def test_with_no_data_it_relaxes_to_the_prior_at_the_stated_rates():
    phi, v, W0 = _stream()
    gamma, lam = 0.9, 0.5
    state = eb.PriorPreservingRLS(W0, lam, gamma=gamma)
    state.write_sequence(phi, v)
    W_gap = (state.W - state.W_prior).norm()
    P_gap = (state.P - state.P_prior).norm()
    assert W_gap > 0.1 and P_gap > 0.1
    n = 50
    for _ in range(n):
        state.relax()
    assert (state.W - state.W_prior).norm().item() == pytest.approx(W_gap.item() * gamma ** (n / 2), rel=1e-6)
    assert (state.P - state.P_prior).norm().item() == pytest.approx(P_gap.item() * gamma**n, rel=1e-6)
    for _ in range(2000):
        state.relax()
    assert torch.allclose(state.W, W0, atol=1e-9, rtol=0.0)
    assert torch.allclose(state.P, torch.eye(4, dtype=DOUBLE).expand(2, -1, -1) / lam, atol=1e-9, rtol=0.0)


def test_prior_is_a_fixed_point():
    _, _, W0 = _stream()
    state = eb.PriorPreservingRLS(W0, 0.5, gamma=0.8)
    W, P = state.W.clone(), state.P.clone()
    state.relax()
    assert torch.allclose(state.W, W, atol=1e-12, rtol=0.0)
    assert torch.allclose(state.P, P, atol=1e-12, rtol=0.0)


def test_plain_forgetting_loses_the_prior_and_prior_preserving_does_not():
    """The distinction H13 tests. Without excitation plain P winds up and W stays put; this one goes home, with P <= P0."""
    phi, v, W0 = _stream()
    gamma, lam, F = 0.9, 0.5, 4
    plain = RLSState(W0, lam, forgetting=gamma)
    prior = eb.PriorPreservingRLS(W0, lam, gamma=gamma)
    plain.write_sequence(phi, v)
    prior.write_sequence(phi, v)
    W_after_data = plain.W.clone()
    silent, nothing = torch.zeros(2, F, dtype=DOUBLE), torch.zeros(2, 2, dtype=DOUBLE)
    for _ in range(200):
        plain.write(silent, nothing)
        prior.write(silent, nothing)
    assert torch.allclose(plain.W, W_after_data, atol=1e-12, rtol=0.0)
    trace_prior0 = F / lam
    assert plain.P.diagonal(dim1=1, dim2=2).sum(dim=1).min().item() > 1e6 * trace_prior0
    assert torch.allclose(prior.W, W0, atol=1e-2, rtol=0.0)  # the gap shrinks by 0.9^100, about 3e-5
    assert prior.P.diagonal(dim1=1, dim2=2).sum(dim=1).max().item() == pytest.approx(trace_prior0, rel=1e-6)


def test_precision_never_falls_below_the_prior():
    phi, v, W0 = _stream(T=100)
    lam = 0.5
    state = eb.PriorPreservingRLS(W0, lam, gamma=0.9)
    P0 = state.P_prior
    for t in range(100):
        state.write(phi[:, t], v[:, t])
        assert torch.linalg.eigvalsh(P0 - state.P).min().item() > -1e-9


def test_update_is_the_exact_filter_of_the_stated_drift():
    """Brute force: Gaussian conditioning on the whole chain, with Cov(w_i, w_j) = gamma^(|i - j| / 2) / lambda and unit noise."""
    T, F = 6, 3
    gen = _gen(31)
    phi = _randn(gen, 1, T, F)
    v = _randn(gen, 1, T, 1)
    W0 = _randn(gen, 1, 1, F)
    gamma, lam = 0.8, 0.7
    state = eb.PriorPreservingRLS(W0, lam, gamma=gamma)
    state.write_sequence(phi, v)

    p = phi[0]  # (T, F)
    r = v[0, :, 0] - p @ W0[0, 0]
    ages = torch.arange(T, dtype=DOUBLE)
    decay = gamma ** (torch.abs(ages[:, None] - ages[None, :]) / 2.0)
    K = (p @ p.t()) * decay / lam + torch.eye(T, dtype=DOUBLE)
    # Cov(w_last, y_j) = gamma^((T - 1 - j) / 2) phi_j / lambda, shape (T, F).
    C = p * (gamma ** ((T - 1 - ages) / 2.0))[:, None] / lam
    mean = W0[0, 0] + C.t() @ torch.linalg.solve(K, r)
    cov = torch.eye(F, dtype=DOUBLE) / lam - C.t() @ torch.linalg.solve(K, C)
    assert torch.allclose(state.W[0, 0], mean, atol=1e-10, rtol=0.0)
    assert torch.allclose(state.P[0], cov, atol=1e-10, rtol=0.0)


def test_prequential_loop_matches_thread_b_at_gamma_one():
    phi, v, W0 = _stream(E=3, T=40, seed=32)
    bounds = (torch.full_like(v, -1.5), torch.full_like(v, 1.5))
    theirs = run_prequential(phi, v, W0, StreamPolicy("persist", 0.5), bounds=bounds, eig_every=10, snapshot_at=[5, 20])
    ours = eb.run_prequential_prior(phi, v, W0, 0.5, 1.0, bounds=bounds, eig_every=10, snapshot_at=[5, 20])
    assert torch.equal(ours.err, theirs.err)
    assert torch.equal(ours.clipped, theirs.clipped)
    assert torch.equal(ours.trace, theirs.trace)
    assert ours.n_writes == theirs.n_writes
    assert ours.eig_steps == theirs.eig_steps
    assert set(ours.snapshots) == {5, 20}
    assert torch.equal(ours.snapshots[20], theirs.snapshots[20])
    assert torch.equal(ours.failures, theirs.failures)


def test_after_a_regime_switch_prior_preserving_recovers_faster_than_persist():
    """A known answer on a stream: two regimes, far apart. Long memory hurts after the switch; relaxing limits it."""
    E, T, F, D = 8, 400, 4, 1
    gen = _gen(33)
    phi = _randn(gen, E, T, F)
    first, second = _randn(gen, E, D, F), _randn(gen, E, D, F)
    v = torch.zeros(E, T, D, dtype=DOUBLE)
    v[:, :200] = phi[:, :200] @ first.transpose(1, 2)
    v[:, 200:] = phi[:, 200:] @ second.transpose(1, 2)
    v = v + 0.01 * _randn(gen, E, T, D)
    W0 = torch.zeros(E, D, F, dtype=DOUBLE)
    persist = run_prequential(phi, v, W0, StreamPolicy("persist", 1.0))
    relaxing = eb.run_prequential_prior(phi, v, W0, 1.0, 0.95)
    assert relaxing.err[:, 200:240].mean().item() < 0.5 * persist.err[:, 200:240].mean().item()
    assert relaxing.failures.sum().item() == 0
