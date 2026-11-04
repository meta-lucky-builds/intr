"""Empirical-Bayes write strength (H12) and prior-preserving forgetting (H13). Registers ebrls and hybrid_eb.

Importing this module adds the methods to adapt.METHODS, so a caller needs ``import intr.ebayes`` first.

The model (H12), for one environment. Phi (N, F), V (N, D), W (D, F) with rows w_d:

    v_d = Phi w_d + eps_d,  eps_d ~ N(0, sigma_d^2 I_N)
    w_d ~ N(w0_d, (sigma_d^2 / lambda) I_F)

Integrating w_d out, the residual r_d = v_d - Phi w0_d has

    r_d ~ N(0, sigma_d^2 C),  C = I_N + Phi Phi' / lambda

    log p(R | lambda, sigma) = -1/2 sum_d [ N log(2 pi sigma_d^2) + log|C| + r_d' C^-1 r_d / sigma_d^2 ]

The maximizing sigma_d^2 has a closed form given lambda, so only lambda needs a grid:

    shared      sigma^2   = sum_d r_d' C^-1 r_d / (N D)
    per_output  sigma_d^2 = r_d' C^-1 r_d / N

N against F. C is N x N and Phi'Phi is F x F. Two forms, each with cost set by its own size:
    dual (N <= F):   eigendecompose Phi Phi' = U S U'. log|C| = sum log(1 + s_i / lambda) and
                     r' C^-1 r = sum (u_i' r)^2 lambda / (lambda + s_i).
    primal (N > F):  Sylvester: log|I_N + Phi Phi' / lambda| = log|I_F + Phi'Phi / lambda|.
                     Woodbury: r' C^-1 r = r'r - r' Phi (lambda I + Phi'Phi)^-1 Phi' r.
Both forms are exact for any N and F, so tests compare them on the same data, and against the full covariance.

Shared sigma. One P serves all outputs, and the posterior mean of W does not depend on sigma, for any diagonal noise
covariance with one lambda (the matrix-normal model). So both modes give a posterior RLS can represent, and they differ
only in which lambda they pick. Per-output gives every output an equal vote in log units, whatever its size. A kinematic
output that the model nearly fits (position delta from velocity) can then drive lambda down by itself, to remove a
residual too small to matter, at the expense of the outputs that carry the shift. Shared sigma weighs outputs by residual
energy in normalized delta units, the same units the rollout metric and the RLS objective see, but it does depend on that
normalization. Shared is the primary rule, since it is also the model H12 states. per_output is a reported sensitivity.

Known ways this model is wrong here (thread_f measures each; none is corrected for):
    1. Simulators are deterministic, so "noise" is misfit, and it is not iid. Along one trajectory residuals are smooth in
       time. The evidence treats 40 correlated transitions as 40 independent ones, which overstates its confidence.
       innovation_diagnostics gives the lag-1 autocorrelation of the standardized one-step-ahead errors, and an
       effective sample size from it.
    2. The evidence scores one-step residuals, and the verdict metric is horizon 10.
    3. Flat evidence. If the spectrum of Phi Phi' is flat, or lambda is far below every s_i, profiled evidence does not
       depend on lambda: with s_i >> lambda, the N log(lambda) terms of the quadratic and the log-determinant cancel.
       lambda is then unidentified from below. log10_lambda_width_2nats reports this per environment. Ties go to the
       largest lambda, the weakest write.
    4. For hybrid_eb, z was already optimized on the same data, so its residuals are not draws from the prior predictive.
       Its evidence is conditional on z.

H13, prior-preserving forgetting. Keep these two distinct:

    plain forgetting (RLSState with gamma < 1, adapt.py:41-44):
        P_pred = P / gamma, W_pred = W, equivalently A_t = gamma A_{t-1} + phi phi' with A_0 = lambda I.
        The prior's precision in A_t decays as gamma^t lambda. After enough steps nothing remembers W0: W stays wherever it
        was, and P grows without bound in unexcited directions (wind-up). It forgets the prior along with the data.

    prior-preserving forgetting (PriorPreservingRLS below):
        W_t - W0 = sqrt(gamma) (W_{t-1} - W0) + xi_t,  xi_t ~ N(0, (1 - gamma) (sigma^2 / lambda) I)
    An Ornstein-Uhlenbeck drift whose stationary distribution is exactly the prior. Its exact Kalman time update is

        W_pred = W0 + sqrt(gamma) (W - W0)
        P_pred = gamma P + (1 - gamma) P0,  P0 = I / lambda

    followed by the standard RLS measurement update. So only the evidence from data decays. gamma = 1 is exactly RLS. With no
    data W returns to W0 and P to P0, geometrically. P <= P0 always (a convex combination, then a write that can only lower
    it), so there is no wind-up by construction. The update is O(F^2), since the time update is a diagonal shift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapt import RLS_DTYPE, AdaptedPredictor, RLSState, _features_and_targets, _optimize_latent, _start_z, register
from .calibration import _NONFINITE_PRED, P_TRACE_LIMIT, PrequentialResult, reset_envs
from .envs import Transitions
from .models import Counts
from .train import Backbone

NO_WRITE_LAMBDA = 1e12  # P0 = 1e-12 I, so q is about 1e-11 and W cannot move. A finite stand-in, safe in JSON.
# Half decades. The first round's grid (1e-3 to 100) was pinned at both edges (docs/RESULTS.md, Problem 7).
DEFAULT_LAMBDAS = tuple(10.0 ** (e / 2.0) for e in range(-12, 13)) + (NO_WRITE_LAMBDA,)
SIGMA2_FLOOR = 1e-12  # targets are float32, so residuals under about 1e-6 are rounding; also keeps log sigma^2 finite
TIE_TOL = 1e-6  # nats. Within this the larger lambda wins, which resolves the flat region below every s_i.
NO_WRITE_EDF = 0.01  # under this many effective written directions, a write counts as no write
SIGMA_MODES = ("shared", "per_output")


# ---------------------------------------------------------------- evidence


@dataclass
class Spectrum:
    """Everything needed to evaluate evidence and posterior at any lambda, batched by environment.

    dual:   s, basis are the eigenpairs of Phi Phi' (M = N); coeff = U' R.
    primal: s, basis are the eigenpairs of Phi' Phi (M = F); coeff = V' Phi' R.
    """

    path: str
    phi: Tensor  # (E, N, F), float64
    residual: Tensor  # (E, N, D), float64
    s: Tensor  # (E, M), non-negative
    basis: Tensor  # (E, M, M)
    coeff: Tensor  # (E, M, D)
    r2: Tensor  # (E, D), ||r_d||^2

    @property
    def n(self) -> int:
        return self.phi.shape[1]


def spectrum(phi: Tensor, residual: Tensor, path: str = "auto") -> Spectrum:
    """phi (E, N, F), residual (E, N, D). path: auto picks the smaller matrix. Non-finite input raises."""
    phi = phi.to(RLS_DTYPE)
    residual = residual.to(RLS_DTYPE)
    if not (torch.isfinite(phi).all() and torch.isfinite(residual).all()):
        raise ValueError("evidence needs finite features and residuals")
    _, N, F = phi.shape
    if path == "auto":
        path = "dual" if N <= F else "primal"
    phi_t = phi.transpose(1, 2)
    if path == "dual":
        s, basis = torch.linalg.eigh(phi @ phi_t)  # (E, N), (E, N, N)
        coeff = basis.transpose(1, 2) @ residual  # (E, N, D)
    elif path == "primal":
        s, basis = torch.linalg.eigh(phi_t @ phi)  # (E, F), (E, F, F)
        coeff = basis.transpose(1, 2) @ (phi_t @ residual)  # (E, F, D)
    else:
        raise ValueError(f"unknown path {path!r}")
    return Spectrum(path, phi, residual, s.clamp_min(0.0), basis, coeff, (residual**2).sum(dim=1))


def quadratic_and_logdet(spec: Spectrum, lam: Tensor) -> tuple[Tensor, Tensor]:
    """lam (E, L). Returns r_d' C^-1 r_d (E, L, D) and log|C| (E, L)."""
    s = spec.s[:, None, :]  # (E, 1, M)
    lam3 = lam[:, :, None]  # (E, L, 1)
    c2 = spec.coeff[:, None, :, :] ** 2  # (E, 1, M, D)
    logdet = torch.log1p(s / lam3).sum(dim=-1)
    if spec.path == "dual":
        quad = ((lam3 / (lam3 + s))[..., None] * c2).sum(dim=2)
    else:
        # Subtraction can go a rounding error below 0 when the fit is nearly exact.
        quad = (spec.r2[:, None, :] - ((1.0 / (lam3 + s))[..., None] * c2).sum(dim=2)).clamp_min(0.0)
    return quad, logdet


def profile_sigma2(quad: Tensor, n: int, sigma_mode: str = "shared", floor: float = SIGMA2_FLOOR) -> Tensor:
    """Maximizing sigma^2 given lambda. quad (..., D) -> (..., D), so both modes read the same downstream."""
    if sigma_mode == "shared":
        shared = quad.sum(dim=-1, keepdim=True) / (n * quad.shape[-1])
        return shared.clamp_min(floor).expand_as(quad)
    if sigma_mode == "per_output":
        return (quad / n).clamp_min(floor)
    raise ValueError(f"unknown sigma_mode {sigma_mode!r}, expected one of {SIGMA_MODES}")


def log_marginal(quad: Tensor, logdet: Tensor, sigma2: Tensor, n: int) -> Tensor:
    """quad, sigma2 (..., D), logdet (...). Written in full, so it stays right when the floor binds."""
    D = quad.shape[-1]
    return -0.5 * (
        n * D * math.log(2.0 * math.pi) + D * logdet + n * torch.log(sigma2).sum(dim=-1) + (quad / sigma2).sum(dim=-1)
    )


@dataclass
class EvidenceGrid:
    spectrum: Spectrum
    lambdas: Tensor  # (L,), ascending
    log_evidence: Tensor  # (E, L), at the profiled sigma^2
    sigma2: Tensor  # (E, L, D)
    no_write_log_evidence: Tensor  # (E,), the exact lambda = infinity limit, with its own profiled sigma^2
    no_write_sigma2: Tensor  # (E, D)


def evidence_grid(
    phi: Tensor, residual: Tensor, lambdas=DEFAULT_LAMBDAS, sigma_mode: str = "shared", path: str = "auto", floor: float = SIGMA2_FLOOR
) -> EvidenceGrid:
    spec = spectrum(phi, residual, path)
    grid = torch.as_tensor(sorted(lambdas), dtype=RLS_DTYPE)  # (L,)
    if (grid <= 0).any():
        raise ValueError("lambdas must be positive")
    lam = grid[None].expand(spec.phi.shape[0], -1)  # (E, L)
    quad, logdet = quadratic_and_logdet(spec, lam)
    sigma2 = profile_sigma2(quad, spec.n, sigma_mode, floor)
    no_write_sigma2 = profile_sigma2(spec.r2, spec.n, sigma_mode, floor)  # C = I
    no_write = log_marginal(spec.r2, torch.zeros_like(spec.r2[:, 0]), no_write_sigma2, spec.n)
    return EvidenceGrid(spec, grid, log_marginal(quad, logdet, sigma2, spec.n), sigma2, no_write, no_write_sigma2)


def select_lambda(log_evidence: Tensor, tie_tol: float = TIE_TOL) -> Tensor:
    """Index of the best lambda per environment, (E,). Among near ties, the largest lambda, the weakest write."""
    best = log_evidence.max(dim=1, keepdim=True).values
    near = log_evidence >= best - tie_tol
    return (near.long() * torch.arange(log_evidence.shape[1])).argmax(dim=1)


def effective_dof(s: Tensor, lam: Tensor) -> Tensor:
    """sum s_i / (s_i + lambda): how many feature directions the write really fits. s (E, M), lam (E,) -> (E,).

    Unlike lambda, this can be compared across backbones, because it accounts for the scale of the features.
    """
    return (s / (s + lam[:, None])).sum(dim=1)


def posterior(spec: Spectrum, W0: Tensor, lam: Tensor) -> tuple[Tensor, Tensor]:
    """Posterior mean and scale-free covariance at a per-environment lambda. W0 (E, D, F), lam (E,).

    Returns W (E, D, F) and P = (Phi'Phi + lambda I)^-1, (E, F, F), the state RLS would reach with these data in any order.
    """
    W0 = W0.to(RLS_DTYPE)
    inv = 1.0 / (spec.s + lam[:, None])  # (E, M)
    if spec.path == "dual":
        # Phi'(K + lambda I)^-1 R, and P = (I - Phi'(K + lambda I)^-1 Phi) / lambda, by the push-through identity.
        A = spec.phi.transpose(1, 2) @ spec.basis  # (E, F, N)
        A_inv = A * inv[:, None, :]
        delta = A_inv @ spec.coeff  # (E, F, D)
        eye = torch.eye(A.shape[1], dtype=RLS_DTYPE)
        P = (eye - A_inv @ A.transpose(1, 2)) / lam[:, None, None]
    else:
        V_inv = spec.basis * inv[:, None, :]  # (E, F, F)
        delta = V_inv @ spec.coeff  # (E, F, D)
        P = V_inv @ spec.basis.transpose(1, 2)
    return W0 + delta.transpose(1, 2), 0.5 * (P + P.transpose(1, 2))


def rls_state_from_posterior(W: Tensor, P: Tensor, forgetting: float = 1.0) -> RLSState:
    """An RLSState holding a given posterior. It may hold a different lambda per environment, and can keep writing."""
    state = RLSState(W, 1.0, forgetting)
    state.P = P.to(RLS_DTYPE).clone()
    return state


def posterior_covariance(phi: Tensor, lam: float) -> Tensor:
    """P for features alone, one lambda for all environments. phi (E, N, F) -> (E, F, F)."""
    E, N, F = phi.shape
    residual = torch.zeros(E, N, 1, dtype=RLS_DTYPE)
    W0 = torch.zeros(E, 1, F, dtype=RLS_DTYPE)
    return posterior(spectrum(phi, residual), W0, torch.full((E,), float(lam), dtype=RLS_DTYPE))[1]


def logdet_precision_gain(phi: Tensor, lam: float) -> Tensor:
    """log det(lambda I + Phi'Phi) - log det(lambda I), the D-optimality measure of a data set. phi (E, N, F) -> (E,)."""
    E, N, _ = phi.shape
    spec = spectrum(phi, torch.zeros(E, N, 1, dtype=RLS_DTYPE))
    return torch.log1p(spec.s / float(lam)).sum(dim=1)


@dataclass
class EmpiricalBayesFit:
    W: Tensor  # (E, D, F)
    P: Tensor  # (E, F, F)
    lambda_: Tensor  # (E,)
    sigma2: Tensor  # (E, D)
    log_evidence: Tensor  # (E,)
    no_write_log_evidence: Tensor  # (E,)
    no_write_sigma2: Tensor  # (E, D)
    edf: Tensor  # (E,)
    index: Tensor  # (E,), into lambdas
    width_2nats: Tensor  # (E,), log10 span of lambdas within 2 nats of the best: large means poorly identified
    lambdas: Tensor  # (L,)
    n: int
    feature_dim: int
    path: str
    sigma_mode: str

    def info(self) -> dict:
        """Per-environment values, as plain lists. Selected values are reported from here (H12's stated risk)."""
        last = self.lambdas.shape[0] - 1
        return {
            "sigma_mode": self.sigma_mode,
            "path": self.path,
            "n_transitions": self.n,
            "feature_dim": self.feature_dim,
            "grid_low": self.lambdas[0].item(),
            "grid_high": self.lambdas[-1].item(),
            "n_grid": self.lambdas.shape[0],
            "lambda": self.lambda_.tolist(),
            "log10_lambda": self.lambda_.log10().tolist(),
            "sigma2": self.sigma2.tolist(),
            "sigma2_mean": self.sigma2.mean(dim=1).tolist(),
            "log_evidence": self.log_evidence.tolist(),
            "log_evidence_no_write": self.no_write_log_evidence.tolist(),
            "log_bf_vs_no_write_per_transition": ((self.log_evidence - self.no_write_log_evidence) / self.n).tolist(),
            "edf": self.edf.tolist(),
            "no_write": (self.edf < NO_WRITE_EDF).tolist(),
            "at_lower_edge": (self.index == 0).tolist(),
            "at_upper_edge": (self.index == last).tolist(),
            "log10_lambda_width_2nats": self.width_2nats.tolist(),
        }


def fit_empirical_bayes(
    phi: Tensor,
    v: Tensor,
    W0: Tensor,
    lambdas=None,
    sigma_mode: str = "shared",
    path: str = "auto",
    tie_tol: float = TIE_TOL,
    floor: float = SIGMA2_FLOOR,
) -> EmpiricalBayesFit:
    """Choose lambda and sigma^2 per environment by evidence on these data alone. phi (E, N, F), v (E, N, D), W0 (E, D, F)."""
    phi = phi.to(RLS_DTYPE)
    v = v.to(RLS_DTYPE)
    W0 = W0.to(RLS_DTYPE)
    residual = v - phi @ W0.transpose(1, 2)  # (E, N, F) @ (E, F, D)
    grid = evidence_grid(phi, residual, DEFAULT_LAMBDAS if lambdas is None else lambdas, sigma_mode, path, floor)
    E = phi.shape[0]
    index = select_lambda(grid.log_evidence, tie_tol)
    rows = torch.arange(E)
    lam = grid.lambdas[index]
    W, P = posterior(grid.spectrum, W0, lam)

    log_lambdas = grid.lambdas.log10()
    within = grid.log_evidence >= grid.log_evidence.max(dim=1, keepdim=True).values - 2.0
    high = torch.where(within, log_lambdas, torch.full_like(log_lambdas, -math.inf)).max(dim=1).values
    low = torch.where(within, log_lambdas, torch.full_like(log_lambdas, math.inf)).min(dim=1).values
    return EmpiricalBayesFit(
        W=W,
        P=P,
        lambda_=lam,
        sigma2=grid.sigma2[rows, index],
        log_evidence=grid.log_evidence[rows, index],
        no_write_log_evidence=grid.no_write_log_evidence,
        no_write_sigma2=grid.no_write_sigma2,
        edf=effective_dof(grid.spectrum.s, lam),
        index=index,
        width_2nats=high - low,
        lambdas=grid.lambdas,
        n=phi.shape[1],
        feature_dim=phi.shape[2],
        path=grid.spectrum.path,
        sigma_mode=sigma_mode,
    )


@torch.no_grad()
def innovation_diagnostics(phi: Tensor, v: Tensor, W0: Tensor, lambda_: Tensor, sigma2: Tensor) -> dict:
    """Replay the data in time order at a fixed per-environment lambda and sigma^2. All per environment.

    The chain rule gives log p(R) = sum_t log N(e_t; 0, sigma^2 (1 + q_t)) with e_t the prequential error, so chain_log_evidence
    must equal the closed form. If the model were true, e_t / sqrt(1 + q_t) would be white. Its lag-1 autocorrelation
    therefore measures the iid violation directly, and N (1 - rho) / (1 + rho) is the sample size an AR(1) reading supports.
    log_bf_scale is that size over N: a crude factor for how far the Bayes factor against no write may be overstated.
    """
    phi = phi.to(RLS_DTYPE)
    v = v.to(RLS_DTYPE)
    E, N, F = phi.shape
    sigma2 = sigma2.to(RLS_DTYPE)
    eye = torch.eye(F, dtype=RLS_DTYPE).expand(E, -1, -1)
    state = rls_state_from_posterior(W0.to(RLS_DTYPE), eye / lambda_[:, None, None])
    white = torch.zeros(E, N, v.shape[-1], dtype=RLS_DTYPE)
    chain = torch.zeros(E, dtype=RLS_DTYPE)
    for t in range(N):
        p = phi[:, t]
        q = state.variance(p[:, None, :])[:, 0]  # (E,)
        err = v[:, t] - (state.W @ p[:, :, None]).squeeze(-1)  # (E, D)
        var = sigma2 * (1.0 + q)[:, None]  # (E, D)
        chain = chain - 0.5 * (torch.log(2.0 * math.pi * var) + err**2 / var).sum(dim=-1)
        white[:, t] = err / var.sqrt()
        state.write(p, v[:, t])
    denom = (white**2).sum(dim=(1, 2)).clamp_min(1e-30)
    rho = (white[:, :-1] * white[:, 1:]).sum(dim=(1, 2)) / denom
    rho_safe = rho.clamp(-0.95, 0.95)
    n_eff = (N * (1.0 - rho_safe) / (1.0 + rho_safe)).clamp_max(float(N))
    return {
        "chain_log_evidence": chain.tolist(),
        "lag1_autocorr": rho.tolist(),
        "n_eff": n_eff.tolist(),
        "log_bf_scale": (n_eff / N).tolist(),
    }


# ---------------------------------------------------------------- registry


def _eb_write(backbone: Backbone, data: Transitions, z: Tensor | None, lambdas, sigma_mode: str, counts: Counts) -> tuple[RLSState, dict]:
    phi, v = _features_and_targets(backbone.model, z, data, counts)
    E = data.obs.shape[0]
    W0 = backbone.model.W.detach()[None].expand(E, -1, -1)
    fit = fit_empirical_bayes(phi, v, W0, lambdas, sigma_mode)
    counts.writes += 1  # one closed-form solve per environment, no sequence of updates
    return rls_state_from_posterior(fit.W, fit.P), fit.info()


@register(
    "ebrls",
    backbones=("latent", "nolatent"),
    defaults={"sigma_mode": "shared", "lambdas": None},
)
def ebrls(backbone: Backbone, data: Transitions, sigma_mode: str, lambdas):
    """RLS from W0 with lambda and sigma^2 chosen per environment by evidence. No validation set, no hyperparameter."""
    counts = Counts()
    z = _start_z(backbone, data.obs.shape[0])
    state, eb = _eb_write(backbone, data, z, lambdas, sigma_mode, counts)
    return AdaptedPredictor(backbone.model, z=z, rls=state), {"eb": eb}, counts


@register(
    "hybrid_eb",
    backbones=("latent",),
    defaults={"steps": 40, "lr_scale": 1.0, "sigma_mode": "shared", "lambdas": None},
    grid=[{"lr_scale": s} for s in (0.5, 1.0, 2.0)],
)
def hybrid_eb(backbone: Backbone, data: Transitions, steps: int, lr_scale: float, sigma_mode: str, lambdas):
    """latent_opt, then ebrls at phi(s, a, z*). lr_scale belongs to the latent stage and may be validated. lambda never is."""
    counts = Counts()
    z, info = _optimize_latent(backbone, data, steps, lr_scale, counts)
    state, eb = _eb_write(backbone, data, z, lambdas, sigma_mode, counts)
    info["eb"] = eb
    return AdaptedPredictor(backbone.model, z=z, rls=state), info, counts


# ---------------------------------------------------------------- H13


class PriorPreservingRLS(RLSState):
    """RLS whose posterior relaxes to the prior. The exact Kalman filter of the drift in the module docstring.

    Do not confuse with RLSState(forgetting=gamma), which inflates P by 1 / gamma and leaves W alone, so the prior itself
    decays. self.forgetting stays 1 here, so the parent's write is the plain measurement update.
    """

    def __init__(self, W0: Tensor, lambda_: float, gamma: float = 1.0):
        super().__init__(W0, lambda_, 1.0)
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma
        self.W_prior = self.W.clone()
        self.P_prior = self.P.clone()

    def relax(self) -> None:
        """Time update: one step of drift back toward the prior. gamma = 1 is exactly no change."""
        if self.gamma == 1.0:
            return
        self.W = self.W_prior + math.sqrt(self.gamma) * (self.W - self.W_prior)
        self.P = self.gamma * self.P + (1.0 - self.gamma) * self.P_prior
        self._W32 = None

    def write(self, phi: Tensor, v: Tensor, w: float | Tensor = 1.0) -> None:
        """Time update, then measurement update."""
        self.relax()
        super().write(phi, v, w)


@torch.no_grad()
def run_prequential_prior(
    phi: Tensor,
    v: Tensor,
    W0: Tensor,
    lambda_: float,
    gamma: float,
    bounds: tuple[Tensor, Tensor] | None = None,
    eig_every: int = 0,
    snapshot_at=(),
    trace_limit: float = P_TRACE_LIMIT,
) -> PrequentialResult:
    """calibration.run_prequential for PriorPreservingRLS, with the same scoring, clipping and failure rules.

    Same order, predict then score then write, except that the time update comes first, since the Kalman prediction for step t
    is the relaxed mean. Separate because run_prequential builds its own RLSState. With gamma = 1 this reproduces its persist
    policy exactly, which a test checks.
    """
    phi = phi.to(RLS_DTYPE)
    v = v.to(RLS_DTYPE)
    E, T, _ = phi.shape
    if W0.dim() == 2:
        W0 = W0[None].expand(E, -1, -1)
    rls = PriorPreservingRLS(W0, lambda_, gamma)
    W_init, P_init = rls.W.clone(), rls.P.clone()
    wanted = set(snapshot_at)

    err = torch.zeros(E, T, dtype=RLS_DTYPE)
    clipped = torch.zeros(E, T, dtype=torch.bool)
    trace = torch.zeros(E, T, dtype=RLS_DTYPE)
    failures = torch.zeros(E, dtype=torch.long)
    first_failure = torch.full((E,), -1, dtype=torch.long)
    eig_steps, lam_max, lam_min = [], [], []
    eig_errors, n_writes = 0, 0
    snapshots = {}

    def fail(bad: Tensor, t: int) -> None:
        nonlocal first_failure, failures
        failures = failures + bad.long()
        first_failure = torch.where((first_failure < 0) & bad, torch.full_like(first_failure, t), first_failure)
        reset_envs(rls, bad, W_init, P_init)

    for t in range(T):
        rls.relax()
        if t in wanted:
            snapshots[t] = rls.W.clone()

        p = phi[:, t]  # (E, F)
        pred = (rls.W @ p[:, :, None]).squeeze(-1)  # (E, D)
        scored = pred
        if bounds is not None:
            safe = torch.nan_to_num(pred, nan=_NONFINITE_PRED, posinf=_NONFINITE_PRED, neginf=-_NONFINITE_PRED)
            scored = torch.max(torch.min(safe, bounds[1][:, t]), bounds[0][:, t])
            clipped[:, t] = (scored != pred).any(dim=-1)
        err[:, t] = ((v[:, t] - scored) ** 2).mean(dim=-1)

        RLSState.write(rls, p, v[:, t])  # measurement update only; relax already ran for this step
        n_writes += E

        diag = rls.P.diagonal(dim1=1, dim2=2)
        tr = diag.sum(dim=1)
        trace[:, t] = tr
        bad = (
            ~torch.isfinite(rls.W).all(dim=(1, 2))
            | ~torch.isfinite(rls.P).all(dim=(1, 2))
            | (diag <= 0).any(dim=1)
            | (tr > trace_limit)
        )
        if bad.any():
            fail(bad, t)

        if eig_every > 0 and (t % eig_every == 0 or t == T - 1):
            try:
                eig = torch.linalg.eigvalsh(rls.P)
                eig_steps.append(t)
                lam_max.append(eig[:, -1])
                lam_min.append(eig[:, 0])
                indefinite = eig[:, 0] <= 0
                if indefinite.any():
                    fail(indefinite, t)
            except torch.linalg.LinAlgError:
                eig_errors += 1

    return PrequentialResult(
        err=err,
        clipped=clipped,
        trace=trace,
        resets=torch.zeros(E, dtype=torch.long),
        failures=failures,
        first_failure=first_failure,
        eig_steps=eig_steps,
        lam_max=torch.stack(lam_max, dim=1) if lam_max else None,
        lam_min=torch.stack(lam_min, dim=1) if lam_min else None,
        eig_errors=eig_errors,
        snapshots=snapshots,
        w_shift=(rls.W - W_init).flatten(1).norm(dim=1),
        n_writes=n_writes,
    )
