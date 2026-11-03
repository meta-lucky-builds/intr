"""Measurement code for Thread B: H4 (is phi' P phi calibrated) and H10 (forgetting under regime switches).

Part 1, ranking and calibration: Spearman with ties, partial Spearman, per-environment and across-environment
ranking, a Gaussian predictive sigma^2 (1 + q) with coverage and NLL.

Part 2, streams: regime schedules, a simulator that switches hidden parameters mid-stream, and the prequential
loop. Prequential means every transition is scored by a model that has seen only earlier transitions.

Units. Errors here are in the RLS target space, the normalized delta (DynamicsModel.target), because that is
where the Bayesian reading of P lives. The rollout metric uses the Thread A unit (obs_scale), so it is
comparable with Thread A. The two are not comparable with each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import torch
from torch import Tensor

from .adapt import RLS_DTYPE, AdaptedPredictor, RLSState, adapt
from .envs import BatchEnv, Transitions, _policy_action, make_env, make_generator
from .models import DynamicsModel, rollout
from .train import Backbone

COVERAGE_LEVELS = (0.5, 0.9)

# Numerical failure, not wind-up as such. P is float64, so at trace 1e12 an entry carries absolute rounding of
# about 1e-4, which is larger than the q of a well-excited direction. Wind-up below this limit is only reported.
P_TRACE_LIMIT = 1e12

_NONFINITE_PRED = 1e30


def to_numpy(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------- ranking


def average_ranks(x) -> np.ndarray:
    """1-based ranks, with ties sharing their average rank. x (N,) without NaN."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    starts = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat((starts + ends - 1) / 2.0 + 1.0, ends - starts)
    return ranks


def _rank_corr(a: np.ndarray, b: np.ndarray, n: int) -> float:
    """Pearson of two centered rank vectors. NaN when either has no variation left.

    The floor is relative to the sum of squares of untied ranks, n^3 / 12, so a control that explains a rank
    vector exactly gives NaN and not a correlation of rounding noise.
    """
    floor = 1e-10 * n**3 / 12.0
    ss_a, ss_b = float(a @ a), float(b @ b)
    if ss_a <= floor or ss_b <= floor:
        return float("nan")
    return float(a @ b / math.sqrt(ss_a * ss_b))


def spearman(x, y) -> float:
    """Spearman with average ranks. Pairs with a non-finite value are dropped. NaN if undefined."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < 3:
        return float("nan")
    rx, ry = average_ranks(x[keep]), average_ranks(y[keep])
    return _rank_corr(rx - rx.mean(), ry - ry.mean(), n)


def partial_spearman(x, y, controls) -> float:
    """Spearman of x and y with the rank of each control regressed out of both ranks.

    x, y (N,); controls: a list of (N,) arrays. NaN if undefined.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    controls = [np.asarray(c, dtype=float) for c in controls]
    keep = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        keep &= np.isfinite(c)
    n = int(keep.sum())
    if n < len(controls) + 3:
        return float("nan")
    Z = np.stack([np.ones(n)] + [average_ranks(c[keep]) for c in controls], axis=1)  # (n, 1 + C)
    R = np.stack([average_ranks(x[keep]), average_ranks(y[keep])], axis=1)  # (n, 2)
    beta, *_ = np.linalg.lstsq(Z, R, rcond=None)
    res = R - Z @ beta
    return _rank_corr(res[:, 0], res[:, 1], n)


def within_env_spearman(signal, err) -> np.ndarray:
    """Does the signal flag which queries are uncertain? (E, N), (E, N) -> (E,)."""
    s, e = to_numpy(signal), to_numpy(err)
    return np.array([spearman(s[i], e[i]) for i in range(s.shape[0])])


def within_env_partial(signal, err, controls) -> np.ndarray:
    """within_env_spearman with controls removed. controls: a list of (E, N). -> (E,)."""
    s, e = to_numpy(signal), to_numpy(err)
    cs = [to_numpy(c) for c in controls]
    return np.array([partial_spearman(s[i], e[i], [c[i] for c in cs]) for i in range(s.shape[0])])


def across_env_spearman(signal, err) -> float:
    """Does the signal flag which environments are poorly adapted? Ranks per-environment means."""
    return spearman(to_numpy(signal).mean(axis=1), to_numpy(err).mean(axis=1))


# ---------------------------------------------------------------- predictive calibration


@dataclass
class QuerySignals:
    """Everything H4 reads at held-out queries, float64. All (E, N) except residual (E, N, D)."""

    q: Tensor  # phi' P phi
    residual: Tensor  # target minus prediction
    phi_norm2: Tensor  # ||phi||^2; its ranks equal those of the prior variance phi' (I / lambda) phi
    input_norm2: Tensor  # ||normalized obs||^2, state magnitude and velocity
    pred_norm2: Tensor  # ||W phi||^2, the size of the predicted change

    @property
    def sq_err(self) -> Tensor:
        """Mean over output dimensions."""
        return (self.residual**2).mean(dim=-1)


@torch.no_grad()
def query_signals(predictor: AdaptedPredictor, data: Transitions) -> QuerySignals:
    """Read an RLS predictor on (E, N, ·) transitions. Does not write."""
    model = predictor.model
    phi = predictor.features(data.obs, data.act)  # (E, N, F)
    q = predictor.rls.variance(phi)
    phi64 = phi.to(RLS_DTYPE)
    pred = phi64 @ predictor.rls.W.transpose(1, 2)  # (E, N, F) @ (E, F, D)
    target = model.target(data.obs, data.next_obs).to(RLS_DTYPE)
    return QuerySignals(
        q=q,
        residual=target - pred,
        phi_norm2=(phi64**2).sum(-1),
        input_norm2=(model.norm_obs(data.obs).to(RLS_DTYPE) ** 2).sum(-1),
        pred_norm2=(pred**2).sum(-1),
    )


def fit_sigma2(residual: Tensor, extra: Tensor | None = None) -> Tensor:
    """Maximum likelihood sigma^2 per output dimension, for r_d ~ N(0, sigma_d^2 (1 + extra)).

    residual (E, N, D), extra (E, N) or None for constant variance. Returns (D,).
    """
    r2 = residual.to(RLS_DTYPE) ** 2
    if extra is not None:
        r2 = r2 / (1.0 + extra.to(RLS_DTYPE))[..., None]
    return r2.mean(dim=(0, 1)).clamp_min(1e-30)


def predictive_var(sigma2: Tensor, extra: Tensor | None, like: Tensor) -> Tensor:
    """(E, N, D)."""
    if extra is None:
        return sigma2.to(RLS_DTYPE).expand_as(like)
    return sigma2.to(RLS_DTYPE) * (1.0 + extra.to(RLS_DTYPE))[..., None]


def coverage(residual: Tensor, extra: Tensor | None, sigma2: Tensor, level: float) -> Tensor:
    """Per-environment fraction of residuals inside the central interval of the given nominal level. (E,)."""
    r = residual.to(RLS_DTYPE)
    half = NormalDist().inv_cdf(0.5 + level / 2.0) * predictive_var(sigma2, extra, r).sqrt()
    return (r.abs() <= half).to(RLS_DTYPE).mean(dim=(1, 2))


def gaussian_nll(residual: Tensor, extra: Tensor | None, sigma2: Tensor) -> Tensor:
    """Per-environment mean negative log density per output dimension. A proper score. (E,)."""
    r = residual.to(RLS_DTYPE)
    var = predictive_var(sigma2, extra, r)
    return 0.5 * (torch.log(2.0 * math.pi * var) + r**2 / var).mean(dim=(1, 2))


def well_specified_spearman(q: Tensor, sigma2: Tensor, gen: torch.Generator, n_reps: int = 3) -> np.ndarray:
    """Within-environment Spearman expected if r_d ~ N(0, sigma_d^2 (1 + q)) were exactly true. (E,).

    A single squared error is variance times a chi-square draw, so even a perfect model has a Spearman well
    below 1, and it depends on the spread of q and on D. With q much below 1 the predictive variance barely
    varies, so a correct model scores near 0. This makes a low Spearman readable. It is a reference and not a
    strict bound: a model whose misfit happens to scale exactly with q can exceed it.
    """
    q64 = q.to(RLS_DTYPE)
    E, N = q64.shape
    D = sigma2.shape[0]
    std = predictive_var(sigma2, q64, torch.zeros(E, N, D, dtype=RLS_DTYPE)).sqrt()
    reps = []
    for _ in range(n_reps):
        r = std * torch.randn(E, N, D, generator=gen, dtype=RLS_DTYPE)
        reps.append(within_env_spearman(q64, (r**2).mean(dim=-1)))
    return np.mean(reps, axis=0)


# ---------------------------------------------------------------- regime schedules


@dataclass
class Schedule:
    """kind: none, fixed (every interval steps), varied (log-uniform per stream), cycle (returns to regimes
    already seen), aligned (a new regime at every episode start, the timing that favours per-episode reset).
    """

    name: str
    kind: str
    interval: int = 0
    min_interval: int = 0
    max_interval: int = 0
    cycle_len: int = 2


def n_regimes_needed(schedule: Schedule, n_steps: int, episode_steps: int) -> int:
    if schedule.kind == "none":
        return 1
    if schedule.kind == "fixed":
        return (n_steps - 1) // schedule.interval + 1
    if schedule.kind == "aligned":
        return (n_steps - 1) // episode_steps + 1
    if schedule.kind == "cycle":
        return schedule.cycle_len
    if schedule.kind == "varied":
        return (n_steps - 1) // schedule.min_interval + 1
    raise ValueError(f"unknown schedule kind {schedule.kind!r}")


def regime_sequence(schedule: Schedule, n_streams: int, n_steps: int, episode_steps: int, gen: torch.Generator) -> Tensor:
    """Regime index per stream and step, (E, T), starting at 0."""
    t = torch.arange(n_steps)[None].expand(n_streams, -1)
    if schedule.kind == "none":
        return torch.zeros_like(t)
    if schedule.kind == "fixed":
        return t // schedule.interval
    if schedule.kind == "aligned":
        return t // episode_steps
    if schedule.kind == "cycle":
        return (t // schedule.interval) % schedule.cycle_len
    if schedule.kind == "varied":
        low, high = math.log(schedule.min_interval), math.log(schedule.max_interval)
        seq = torch.zeros(n_streams, n_steps, dtype=torch.long)
        for s in range(n_streams):
            start, regime = 0, 0
            while start < n_steps:
                interval = int(round(math.exp(low + (high - low) * torch.rand(1, generator=gen).item())))
                seq[s, start : start + interval] = regime
                start += interval
                regime += 1
        return seq
    raise ValueError(f"unknown schedule kind {schedule.kind!r}")


@dataclass
class Stream:
    """One stream per row. Transition t runs under the regime at t, so a switch at t makes transition t the
    first one in the new regime. All tensors are (E, T, ·) or (E, T), except episode_start.
    """

    obs: Tensor
    act: Tensor
    next_obs: Tensor
    env_ids: Tensor  # index into the pool, (E, T)
    switch: Tensor  # bool, True at the first step of every regime after the first
    since_switch: Tensor  # steps since the regime began, 0 at a switch and at t = 0
    episode_start: Tensor  # bool (E, T)
    tier: str = ""
    schedule: str = ""

    @property
    def transitions(self) -> Transitions:
        return Transitions(self.obs, self.act, self.next_obs)


def subset_env(pool: BatchEnv, idx: Tensor) -> BatchEnv:
    """A batch of environments picked from a pool. idx (E,)."""
    params = {k: v[idx] for k, v in pool.params.items()}
    return type(pool)(params, [pool.mechanisms[i] for i in idx.tolist()], pool.tier)


@torch.no_grad()
def simulate_stream(pool: BatchEnv, env_ids: Tensor, gen: torch.Generator, policy: str | None = None, tier: str = "", schedule: str = "") -> Stream:
    """Roll out streams whose hidden parameters follow env_ids. env_ids (E, T) indexes the pool.

    The physical state resets every pool.episode_steps. A switch away from an episode start keeps the state, so
    only the dynamics change. A stored transition never crosses an episode reset.
    """
    policy = pool.default_probe if policy is None else policy
    E, T = env_ids.shape
    episode_steps = pool.episode_steps
    obs, act, next_obs = [], [], []
    env, state, noise = None, None, None
    for t in range(T):
        if t == 0 or (env_ids[:, t] != env_ids[:, t - 1]).any():
            env = subset_env(pool, env_ids[:, t])
        if t % episode_steps == 0:
            state = env.reset(1, gen)  # (E, 1, S)
            noise = None
        o = env.obs(state)  # (E, 1, O)
        a, noise = _policy_action(env, policy, o, noise, gen)  # (E, 1, A)
        state = env.step(state, a)
        obs.append(o[:, 0])
        act.append(a[:, 0])
        next_obs.append(env.obs(state)[:, 0])
    switch = torch.zeros(E, T, dtype=torch.bool)
    switch[:, 1:] = env_ids[:, 1:] != env_ids[:, :-1]
    since = torch.zeros(E, T, dtype=torch.long)
    for t in range(1, T):
        since[:, t] = torch.where(switch[:, t], torch.zeros(E, dtype=torch.long), since[:, t - 1] + 1)
    episode_start = (torch.arange(T) % episode_steps == 0)[None].expand(E, -1).clone()
    return Stream(
        torch.stack(obs, dim=1), torch.stack(act, dim=1), torch.stack(next_obs, dim=1),
        env_ids, switch, since, episode_start, tier, schedule,
    )


def build_stream(family: str, tier: str, schedule: Schedule, n_streams: int, n_steps: int, env_seed, tag) -> Stream:
    """Draw the regimes and simulate. Regimes within a stream are distinct environments of one tier, apart from
    the deliberate returns of a cycle. A random assignment mixes the mechanisms of the oof tier over regimes.
    """
    episode_steps = make_env(family, 1, "train", "bounds").episode_steps
    R = n_regimes_needed(schedule, n_steps, episode_steps)
    pool = make_env(family, n_streams * R, tier, env_seed)
    table = torch.randperm(n_streams * R, generator=make_generator(tag, "assign")).reshape(n_streams, R)  # (E, R)
    regimes = regime_sequence(schedule, n_streams, n_steps, episode_steps, make_generator(tag, "regimes"))
    env_ids = table.gather(1, regimes)  # (E, T)
    return simulate_stream(pool, env_ids, make_generator(tag, "simulate"), tier=tier, schedule=schedule.name)


def post_switch_mask(stream: Stream, n: int) -> Tensor:
    """First n steps of every regime after the first, cut short by the next switch. (E, T)."""
    switched = stream.switch.long().cumsum(dim=1) > 0
    return switched & (stream.since_switch < n)


def steady_mask(stream: Stream, settle: int) -> Tensor:
    """Steps at least settle steps into a regime. Empty for a schedule that switches sooner. (E, T)."""
    return stream.since_switch >= settle


def rmse_over(err: Tensor, mask: Tensor | None = None) -> Tensor:
    """Per-stream root mean of squared errors, the same form as open_loop_errors. NaN where the mask is empty.

    err (E, T), mask (E, T) bool. Returns (E,).
    """
    if mask is None:
        return err.mean(dim=1).sqrt()
    m = mask.to(err.dtype)
    count = m.sum(dim=1)
    out = ((err * m).sum(dim=1) / count.clamp_min(1.0)).sqrt()
    return torch.where(count > 0, out, torch.full_like(out, float("nan")))


# ---------------------------------------------------------------- prequential loop


@dataclass
class SurpriseConfig:
    """Reset when a fast average of the prequential error exceeds threshold times a slow one."""

    threshold: float
    tau_fast: float = 5.0
    tau_slow: float = 50.0
    min_gap: int = 20  # steps after a reset before the next can fire, so high errors after a reset cannot chain


class SurpriseDetector:
    """Causal, scale-free, per stream. Uses only errors already scored."""

    def __init__(self, n_envs: int, cfg: SurpriseConfig):
        self.cfg = cfg
        self.fast = torch.zeros(n_envs, dtype=RLS_DTYPE)
        self.slow = torch.zeros(n_envs, dtype=RLS_DTYPE)
        self.age = torch.zeros(n_envs, dtype=torch.long)  # scored steps since the last reset

    def update(self, err: Tensor) -> Tensor:
        """err (E,). Returns a bool (E,) saying which streams should reset now."""
        fresh = self.age == 0
        fast = torch.where(fresh, err, self.fast + (err - self.fast) / self.cfg.tau_fast)
        trigger = ~fresh & (self.age + 1 >= self.cfg.min_gap) & (fast > self.cfg.threshold * self.slow.clamp_min(1e-12))
        # The slow average is compared before it sees err, so a spike cannot dilute its own detection.
        self.slow = torch.where(fresh, err, self.slow + (err - self.slow) / self.cfg.tau_slow)
        self.fast = fast
        self.age = self.age + 1
        return trigger

    def clear(self, mask: Tensor) -> None:
        self.age = torch.where(mask, torch.zeros_like(self.age), self.age)


@dataclass
class StreamPolicy:
    """What a method does with an RLS state along a stream.

    reset_at (E, T) bool: known in advance (episode start, or an oracle), applied before scoring step t.
    surprise: a data-driven reset, decided after scoring step t and applied before writing it.
    write = False scores W0 only, the frozen reference.
    """

    name: str
    lambda_: float = 1.0
    forgetting: float = 1.0
    reset_at: Tensor | None = None
    surprise: SurpriseConfig | None = None
    write: bool = True


@dataclass
class PrequentialResult:
    err: Tensor  # (E, T), mean over output dimensions, scored before transition t is written
    clipped: Tensor  # (E, T) bool, the prediction was outside the observation bounds or non-finite
    trace: Tensor  # (E, T), trace of P after step t, before any failure reset
    resets: Tensor  # (E,), resets the policy asked for
    failures: Tensor  # (E,), numerical failures, each followed by a forced reset
    first_failure: Tensor  # (E,), step of the first failure, -1 if none
    eig_steps: list[int] = field(default_factory=list)
    lam_max: Tensor | None = None  # (E, len(eig_steps))
    lam_min: Tensor | None = None
    eig_errors: int = 0
    snapshots: dict[int, Tensor] = field(default_factory=dict)  # {t: (E, D, F)}, W used to score step t
    w_shift: Tensor | None = None  # (E,), final ||W - W0||_F
    n_writes: int = 0


def reset_envs(rls: RLSState, mask: Tensor, W_init: Tensor, P_init: Tensor) -> None:
    """Back to the prior in the streams selected by mask (E,) bool. The others are untouched."""
    rls.W = torch.where(mask[:, None, None], W_init, rls.W)
    rls.P = torch.where(mask[:, None, None], P_init, rls.P)
    rls._W32 = None


@torch.no_grad()
def run_prequential(
    phi: Tensor,
    v: Tensor,
    W0: Tensor,
    policy: StreamPolicy,
    bounds: tuple[Tensor, Tensor] | None = None,
    eig_every: int = 0,
    snapshot_at=(),
    trace_limit: float = P_TRACE_LIMIT,
) -> PrequentialResult:
    """Predict, score, maybe reset, write, in that order, one transition at a time.

    phi (E, T, F), v (E, T, D), W0 (D, F) or (E, D, F).
    bounds: (low, high), each (E, T, D), target-space images of the observation bounds. A prediction outside is
    clipped before scoring, as rollouts clip in eval.py, so a diverging readout has a large but bounded error
    and cannot swamp a mean. ``clipped`` records every time this happened.

    Numerical failure is a non-finite W or P, a non-positive diagonal or eigenvalue of P, or a trace above
    trace_limit. The stream is then forced back to the prior so that the run can continue, and the event is
    counted. A forced reset can flatter a method, so callers must treat a method with failures as tainted.
    """
    phi = phi.to(RLS_DTYPE)
    v = v.to(RLS_DTYPE)
    E, T, _ = phi.shape
    if W0.dim() == 2:
        W0 = W0[None].expand(E, -1, -1)
    rls = RLSState(W0, policy.lambda_, policy.forgetting)
    W_init, P_init = rls.W.clone(), rls.P.clone()
    detector = SurpriseDetector(E, policy.surprise) if policy.surprise is not None else None
    wanted = set(snapshot_at)

    err = torch.zeros(E, T, dtype=RLS_DTYPE)
    clipped = torch.zeros(E, T, dtype=torch.bool)
    trace = torch.zeros(E, T, dtype=RLS_DTYPE)
    resets = torch.zeros(E, dtype=torch.long)
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
        if detector is not None:
            detector.clear(bad)

    for t in range(T):
        if policy.reset_at is not None and t > 0:
            mask = policy.reset_at[:, t]
            if mask.any():
                reset_envs(rls, mask, W_init, P_init)
                resets = resets + mask.long()
                if detector is not None:
                    detector.clear(mask)
        if t in wanted:
            snapshots[t] = rls.W.clone()

        # Score. Nothing from transition t has touched rls yet.
        p = phi[:, t]  # (E, F)
        pred = (rls.W @ p[:, :, None]).squeeze(-1)  # (E, D)
        scored = pred
        if bounds is not None:
            safe = torch.nan_to_num(pred, nan=_NONFINITE_PRED, posinf=_NONFINITE_PRED, neginf=-_NONFINITE_PRED)
            scored = torch.max(torch.min(safe, bounds[1][:, t]), bounds[0][:, t])
            clipped[:, t] = (scored != pred).any(dim=-1)  # NaN compares unequal, so it is counted
        err_t = ((v[:, t] - scored) ** 2).mean(dim=-1)
        err[:, t] = err_t

        if detector is not None:
            trigger = detector.update(err_t)
            if trigger.any():
                reset_envs(rls, trigger, W_init, P_init)
                resets = resets + trigger.long()
                detector.clear(trigger)

        if policy.write:
            rls.write(p, v[:, t])
            n_writes += E

        diag = rls.P.diagonal(dim1=1, dim2=2)  # (E, F)
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
            # After the check above, so P is finite here.
            try:
                eig = torch.linalg.eigvalsh(rls.P)  # (E, F), ascending
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
        resets=resets,
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


def windup_summary(result: PrequentialResult, n_points: int = 11) -> dict:
    """What the record needs to show wind-up and blow-up. An infinite or NaN trace is reported as 1e300."""
    trace = torch.nan_to_num(result.trace, nan=1e300, posinf=1e300, neginf=1e300).clamp_min(1e-300)
    log_trace = trace.log10()  # (E, T)
    T = trace.shape[1]
    points = sorted({round(i * (T - 1) / max(n_points - 1, 1)) for i in range(n_points)})
    failed = result.failures > 0
    out = {
        "trace_steps": points,
        "log10_trace_mean": [log_trace[:, t].mean().item() for t in points],
        "log10_trace_max": [log_trace[:, t].max().item() for t in points],
        "log10_trace_growth": (log_trace[:, -1] - log_trace[:, 0]).mean().item(),
        "trace_max": trace.max().item(),
        "n_failures": int(result.failures.sum()),
        "failed_stream_fraction": failed.to(RLS_DTYPE).mean().item(),
        "first_failure_step": int(result.first_failure[failed].min()) if failed.any() else None,
        "eig_errors": result.eig_errors,
        "w_shift_final_mean": None if result.w_shift is None else result.w_shift.mean().item(),
    }
    if result.lam_max is not None:
        out["eig_steps"] = result.eig_steps
        out["lam_max_mean"] = result.lam_max.mean(dim=0).tolist()
        out["lam_max_max"] = result.lam_max.max().item()
        out["lam_min_min"] = result.lam_min.min().item()
    return out


# ---------------------------------------------------------------- streams through a backbone


@dataclass
class FeatureStream:
    """A stream with its features on one frozen backbone. Features do not depend on the method, so every method
    in a cell replays the same numbers.
    """

    stream: Stream
    phi: Tensor  # (E, T, F)
    v: Tensor  # (E, T, D)
    bounds: tuple[Tensor, Tensor]  # target-space image of the observation bounds, each (E, T, D)
    W0: Tensor  # (E, D, F)
    model: DynamicsModel
    z: Tensor | None  # (E, Z)
    forward_samples: int = 0


@torch.no_grad()
def prepare_stream(backbone: Backbone, stream: Stream, obs_low: Tensor, obs_high: Tensor) -> FeatureStream:
    model = backbone.model
    frozen = adapt("frozen", backbone, stream.transitions).predictor  # mean training latent, trained W
    E, T, _ = stream.obs.shape
    phi = frozen.features(stream.obs, stream.act).to(RLS_DTYPE)
    v = model.target(stream.obs, stream.next_obs).to(RLS_DTYPE)
    low = model.target(stream.obs, obs_low).to(RLS_DTYPE)
    high = model.target(stream.obs, obs_high).to(RLS_DTYPE)
    W0 = model.W.detach()[None].expand(E, -1, -1)
    return FeatureStream(stream, phi, v, (low, high), W0, model, frozen.z, E * T)


def rollout_starts(n_steps: int, episode_steps: int, horizon: int, every: int) -> list[int]:
    """Start steps whose window stays inside one episode, so the truth has no state reset in it."""
    return [t for t in range(0, n_steps - horizon + 1, every) if t % episode_steps + horizon <= episode_steps]


@torch.no_grad()
def prequential_rollout_rmse(fs: FeatureStream, snapshots: dict[int, Tensor], horizon: int, obs_scale: Tensor, obs_low: Tensor, obs_high: Tensor) -> tuple[Tensor, AdaptedPredictor | None]:
    """Open-loop error at a horizon with the W held at each start, per stream. (E,).

    Each W has seen only transitions before its start, so this is prequential too. Windows may cross a regime
    switch, which is the point. Same unit and form as eval.open_loop_errors. NaN when there are no windows.
    """
    E = fs.phi.shape[0]
    starts = sorted(snapshots)
    if not starts:
        return torch.full((E,), float("nan"), dtype=RLS_DTYPE), None
    S = len(starts)
    W = torch.stack([snapshots[t] for t in starts], dim=1)  # (E, S, D, F)
    W = W.reshape(E * S, *W.shape[2:]).to(torch.float32)
    z = None
    if fs.z is not None:
        z = fs.z[:, None, :].expand(-1, S, -1).reshape(E * S, -1)
    predictor = AdaptedPredictor(fs.model, z=z, W=W)
    stream = fs.stream
    o0 = stream.obs[:, starts].reshape(E * S, 1, -1)  # (E S, 1, O)
    act = torch.stack([stream.act[:, t : t + horizon] for t in starts], dim=1)  # (E, S, H, A)
    act = act.reshape(E * S, 1, horizon, -1)  # (E S, 1, H, A)
    truth = torch.stack([stream.next_obs[:, t + horizon - 1] for t in starts], dim=1)  # (E, S, O)
    pred = rollout(predictor.step, o0, act, obs_low, obs_high)[:, 0, horizon]  # (E S, O)
    pred = pred.reshape(E, S, -1)
    scaled = (pred - truth) / obs_scale
    return (scaled**2).mean(dim=(1, 2)).sqrt().to(RLS_DTYPE), predictor


def stream_metrics(result: PrequentialResult, stream: Stream, post_ns, settle: int, curve_len: int = 30) -> dict:
    """Prequential, post-switch and steady-state RMSE, as a mean over streams and per stream."""

    def record(values: Tensor) -> dict:
        return {"mean": torch.nanmean(values).item(), "per_stream": values.tolist()}

    out = {"prequential": record(rmse_over(result.err))}
    for n in post_ns:
        out[f"post_switch_{n}"] = record(rmse_over(result.err, post_switch_mask(stream, n)))
    out["steady"] = record(rmse_over(result.err, steady_mask(stream, settle)))
    switched = stream.switch.long().cumsum(dim=1) > 0
    curve = []
    for s in range(curve_len):
        mask = switched & (stream.since_switch == s)
        curve.append(result.err[mask].mean().sqrt().item() if mask.any() else float("nan"))
    out["post_switch_curve"] = curve
    out["clipped_fraction"] = result.clipped.to(RLS_DTYPE).mean().item()
    out["resets_per_stream"] = result.resets.to(RLS_DTYPE).mean().item()
    return out
