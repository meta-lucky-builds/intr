"""RLS must equal batch ridge exactly. H1, H2, H3, H4 and H10 all rest on this."""

import pytest
import torch

from intr.adapt import RLSState

DOUBLE = torch.float64


def _problem(seed, E=3, N=30, D=2, F=8, collinear=False):
    gen = torch.Generator().manual_seed(seed)
    phi = torch.randn(E, N, F, generator=gen, dtype=DOUBLE)
    if collinear:
        # Smooth trajectory: all features nearly one direction, a small perturbation apart.
        direction = torch.randn(E, 1, F, generator=gen, dtype=DOUBLE)
        amplitude = torch.linspace(0.5, 1.5, N, dtype=DOUBLE)[None, :, None]
        phi = direction * amplitude + 1e-3 * phi
    v = torch.randn(E, N, D, generator=gen, dtype=DOUBLE)
    W0 = torch.randn(E, D, F, generator=gen, dtype=DOUBLE)
    return phi, v, W0


def _batch_ridge(phi, v, W0, lambda_, weights=None):
    """Closed form: W = W0 + ((Phi' D Phi + lambda I)^-1 Phi' D (V - Phi W0'))'."""
    E, N, F = phi.shape
    w = torch.ones(E, N, dtype=DOUBLE) if weights is None else weights
    A = phi.transpose(1, 2) @ (phi * w[:, :, None]) + lambda_ * torch.eye(F, dtype=DOUBLE)
    residual = v - phi @ W0.transpose(1, 2)  # (E, N, D)
    delta = torch.linalg.solve(A, phi.transpose(1, 2) @ (residual * w[:, :, None]))  # (E, F, D)
    return W0 + delta.transpose(1, 2), torch.linalg.inv(A)


@pytest.mark.parametrize("lambda_", [1e-3, 0.1, 10.0])
@pytest.mark.parametrize("collinear", [False, True])
def test_matches_batch_ridge(lambda_, collinear):
    phi, v, W0 = _problem(0, collinear=collinear)
    rls = RLSState(W0, lambda_)
    rls.write_sequence(phi, v)
    W_ridge, P_ridge = _batch_ridge(phi, v, W0, lambda_)
    assert torch.allclose(rls.W, W_ridge, atol=1e-6, rtol=1e-6)
    assert torch.allclose(rls.P, P_ridge, atol=1e-6, rtol=1e-6)


def test_underdetermined_matches_batch_ridge():
    # K = 1 in the experiment gives fewer transitions than features in some settings.
    phi, v, W0 = _problem(1, N=5, F=20)
    rls = RLSState(W0, 0.01)
    rls.write_sequence(phi, v)
    W_ridge, _ = _batch_ridge(phi, v, W0, 0.01)
    assert torch.allclose(rls.W, W_ridge, atol=1e-7, rtol=1e-7)


def test_weighted_matches_batch_ridge():
    phi, v, W0 = _problem(2)
    weights = torch.full(phi.shape[:2], 0.5, dtype=DOUBLE)
    weights[:, ::2] = 3.0
    rls = RLSState(W0, 0.1)
    for t in range(phi.shape[1]):
        rls.write(phi[:, t], v[:, t], weights[:, t])
    W_ridge, _ = _batch_ridge(phi, v, W0, 0.1, weights)
    assert torch.allclose(rls.W, W_ridge, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("collinear", [False, True])
def test_order_invariant(collinear):
    phi, v, W0 = _problem(3, collinear=collinear)
    forward = RLSState(W0, 1e-2)
    forward.write_sequence(phi, v)
    N = phi.shape[1]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(9))
    shuffled = RLSState(W0, 1e-2)
    shuffled.write_sequence(phi[:, perm], v[:, perm])
    backward = RLSState(W0, 1e-2)
    backward.write_sequence(phi.flip(1), v.flip(1))
    assert torch.allclose(forward.W, shuffled.W, atol=1e-6, rtol=1e-6)
    assert torch.allclose(forward.W, backward.W, atol=1e-6, rtol=1e-6)
    assert torch.allclose(forward.P, shuffled.P, atol=1e-6, rtol=1e-6)


def test_environments_are_independent():
    phi, v, W0 = _problem(4)
    together = RLSState(W0, 0.1)
    together.write_sequence(phi, v)
    alone = RLSState(W0[1:2], 0.1)
    alone.write_sequence(phi[1:2], v[1:2])
    assert torch.allclose(together.W[1], alone.W[0], atol=1e-10)


@pytest.mark.parametrize("forgetting", [1.0, 0.95])
@pytest.mark.parametrize("collinear", [False, True])
def test_P_stays_symmetric_positive_definite(forgetting, collinear):
    phi, v, W0 = _problem(5, N=400, collinear=collinear)
    rls = RLSState(W0, 1e-3, forgetting=forgetting)
    for t in range(phi.shape[1]):
        rls.write(phi[:, t], v[:, t])
        if t % 50 == 0 or t == phi.shape[1] - 1:
            assert torch.equal(rls.P, rls.P.transpose(1, 2))
            assert (torch.linalg.eigvalsh(rls.P) > 0).all()
    torch.linalg.cholesky(rls.P)  # raises if not positive definite


def test_forgetting_matches_exponentially_weighted_ridge():
    phi, v, W0 = _problem(6)
    gamma, lambda_ = 0.9, 0.5
    rls = RLSState(W0, lambda_, forgetting=gamma)
    rls.write_sequence(phi, v)
    E, N, F = phi.shape
    # The t-th of N observations carries gamma^(N-1-t); the prior carries gamma^N lambda.
    weights = gamma ** torch.arange(N - 1, -1, -1, dtype=DOUBLE)[None].expand(E, -1)
    W_ridge, P_ridge = _batch_ridge(phi, v, W0, lambda_ * gamma**N, weights)
    assert torch.allclose(rls.W, W_ridge, atol=1e-7, rtol=1e-7)
    assert torch.allclose(rls.P, P_ridge, atol=1e-7, rtol=1e-7)


def test_variance_is_prior_then_shrinks_with_data():
    phi, v, W0 = _problem(7)
    lambda_ = 0.5
    rls = RLSState(W0, lambda_)
    query = phi[:, :4]  # (E, 4, F)
    prior = rls.variance(query)
    expected = (query**2).sum(-1) / lambda_  # phi' (I / lambda) phi
    assert torch.allclose(prior, expected, atol=1e-10)
    rls.write_sequence(phi, v)
    posterior = rls.variance(query)
    assert posterior.shape == query.shape[:2]
    assert (posterior < prior).all()
    assert (posterior >= 0).all()


def test_recovers_true_readout():
    gen = torch.Generator().manual_seed(8)
    E, N, D, F = 2, 200, 3, 10
    phi = torch.randn(E, N, F, generator=gen, dtype=DOUBLE)
    W_true = torch.randn(E, D, F, generator=gen, dtype=DOUBLE)
    v = phi @ W_true.transpose(1, 2)
    rls = RLSState(torch.zeros(E, D, F, dtype=DOUBLE), 1e-6)
    rls.write_sequence(phi, v)
    assert torch.allclose(rls.W, W_true, atol=1e-4)


def test_rejects_bad_settings():
    W0 = torch.zeros(1, 2, 3, dtype=DOUBLE)
    with pytest.raises(ValueError):
        RLSState(W0, 0.0)
    with pytest.raises(ValueError):
        RLSState(W0, 1.0, forgetting=1.5)
