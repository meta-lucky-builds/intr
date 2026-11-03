"""Paired evaluation. Model-agnostic: a predictor is anything with step(o, a) and a counts attribute.

step takes o (E, B, O), a (E, B, A) and returns (E, B, O).

Open-loop error at horizon h: predict h steps by feeding predictions back, then take the RMSE of the h-th
prediction, per observation dimension scaled by a family constant (``compute_obs_scale``), so the unit
is the same for every method, seed and tier. Error is per environment, shape (E,).

Rollouts clip predictions to the family's observation bounds. A diverging model thus has a large but
bounded error, and one bad environment cannot dominate a mean. This caps how bad any method can look.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .adapt import METHODS, adapt
from .envs import BatchEnv, Task, Trajectories, Transitions, collect_mixed, make_env, make_generator
from .models import Counts, clip_obs, rollout
from .train import Backbone

HORIZONS = (1, 10, 30)
START_STEPS = (0, 10, 20, 30)
EFFECT_THRESHOLD = 0.10


class OraclePredictor:
    """The true simulator as a model: the ceiling for planning, and a check that a task is feasible."""

    def __init__(self, env: BatchEnv):
        self.env = env
        self.counts = Counts()

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        self.counts.forward_calls += 1
        self.counts.forward_samples += o.shape[0] * o.shape[1]
        return self.env.obs(self.env.step(self.env.state_from_obs(o), a))


class PersistencePredictor:
    """Next observation equals current. A model that cannot beat this has not learned the dynamics."""

    def __init__(self):
        self.counts = Counts()

    def step(self, o: Tensor, a: Tensor) -> Tensor:
        return o


@torch.no_grad()
def compute_obs_scale(family: str) -> Tensor:
    """Per-dimension observation std on a fixed training-range sample. (O,). Independent of seed."""
    env = make_env(family, 64, "train", "obs_scale")
    trajs = collect_mixed(env, 4, 40, env.data_policies, make_generator("obs_scale", family))
    return trajs.obs.reshape(-1, env.obs_dim).std(0).clamp_min(1e-6)


def collect_holdout(env: BatchEnv, n_traj: int, n_steps: int, gen) -> Trajectories:
    """Held-out trajectories: half from the default probe, half with random actions.

    The adaptation data comes from the default probe only, so half of the queries leave the action
    distribution seen during adaptation. Readout writes do not get to fit only the region they query.
    """
    return collect_mixed(env, n_traj, n_steps, (env.default_probe, "iid"), gen)


@torch.no_grad()
def open_loop_errors(
    predictor,
    holdout: Trajectories,
    obs_scale: Tensor,
    obs_low: Tensor,
    obs_high: Tensor,
    horizons=HORIZONS,
    starts=START_STEPS,
) -> dict[int, Tensor]:
    """Returns {h: (E,)}. Holdout needs at least max(starts) + max(horizons) steps."""
    E, B, T, A = holdout.act.shape
    O = holdout.obs.shape[-1]
    H = max(horizons)
    if max(starts) + H > T:
        raise ValueError(f"holdout of {T} steps is too short for start {max(starts)} and horizon {H}")
    S = len(starts)
    # Take a window from each start: obs (E, B, S, H + 1, O), act (E, B, S, H, A).
    true = torch.stack([holdout.obs[:, :, s : s + H + 1] for s in starts], dim=2)
    act = torch.stack([holdout.act[:, :, s : s + H] for s in starts], dim=2)
    true = true.reshape(E, B * S, H + 1, O)
    act = act.reshape(E, B * S, H, A)
    pred = rollout(predictor.step, true[:, :, 0], act, obs_low, obs_high)  # (E, B S, H + 1, O)
    out = {}
    for h in horizons:
        scaled = (pred[:, :, h] - true[:, :, h]) / obs_scale  # (E, B S, O)
        out[h] = (scaled**2).mean(dim=(1, 2)).sqrt()  # (E,)
    return out


@dataclass
class CEMConfig:
    horizon: int = 10
    n_candidates: int = 32
    n_elites: int = 6
    n_iters: int = 3
    init_std: float = 0.6
    min_std: float = 0.05
    noise_beta: float = 0.5  # AR(1) correlation of candidate noise over planning steps


def cem_plan(predictor, env, obs: Tensor, goal: Tensor | None, mean: Tensor, cfg: CEMConfig, gen) -> tuple[Tensor, Tensor]:
    """One batched CEM plan inside the predictor. env supplies cost and obs bounds only.

    obs (E, O), mean (E, H, A). Returns the first action of the best sequence seen, (E, A), and the
    shifted mean for warm starting, (E, H, A). The best sequence is more robust than the elite mean
    with so few candidates, and it keeps bang-bang swing-up torques from being averaged away.
    """
    E, H, A = mean.shape
    N = cfg.n_candidates
    std = torch.full_like(mean, cfg.init_std)
    best_cost = torch.full((E,), float("inf"))
    best_action = mean[:, 0].clone()
    keep = math.sqrt(1.0 - cfg.noise_beta**2)
    for _ in range(cfg.n_iters):
        noise = torch.randn(E, N, H, A, generator=gen)
        for t in range(1, H):
            noise[:, :, t] = cfg.noise_beta * noise[:, :, t - 1] + keep * noise[:, :, t]
        candidates = (mean[:, None] + std[:, None] * noise).clamp(-1.0, 1.0)  # (E, N, H, A)
        o = obs[:, None, :].expand(E, N, -1)  # (E, N, O)
        cost = torch.zeros(E, N)
        for t in range(H):
            a = candidates[:, :, t]
            o = clip_obs(predictor.step(o, a), env.obs_low, env.obs_high)
            cost = cost + env.cost(o, a, goal)
        top_cost, idx = cost.topk(cfg.n_elites, dim=1, largest=False)  # (E, n_elites), sorted ascending
        elites = candidates.gather(1, idx[:, :, None, None].expand(-1, -1, H, A))  # (E, n_elites, H, A)
        mean = elites.mean(dim=1)
        std = elites.std(dim=1, unbiased=False).clamp_min(cfg.min_std)
        better = top_cost[:, 0] < best_cost
        best_cost = torch.where(better, top_cost[:, 0], best_cost)
        best_action = torch.where(better[:, None], elites[:, 0, 0], best_action)
    next_mean = torch.cat([mean[:, 1:], torch.zeros(E, 1, A)], dim=1)
    return best_action, next_mean


@torch.no_grad()
def closed_loop(env: BatchEnv, predictor, task: Task, cfg: CEMConfig, gen, steps: int | None = None) -> dict:
    """Plan in the predictor, act in the true simulator. Use the same task and gen seed across methods.

    Returns success (E,) bool and mean_cost (E,), the mean true per-step cost.
    """
    steps = env.episode_steps if steps is None else steps
    E = env.n_envs
    state = task.state  # (E, 1, S)
    o = env.obs(state)[:, 0]  # (E, O)
    mean = torch.zeros(E, cfg.horizon, env.act_dim)
    observed = [o]
    total_cost = torch.zeros(E)
    started = time.perf_counter()
    for _ in range(steps):
        action, mean = cem_plan(predictor, env, o, task.goal, mean, cfg, gen)  # (E, A), (E, H, A)
        state = env.step(state, action[:, None, :])
        o = env.obs(state)[:, 0]
        observed.append(o)
        total_cost += env.cost(o[:, None], action[:, None], task.goal)[:, 0].to(total_cost.dtype)
    traj = torch.stack(observed, dim=1)  # (E, steps + 1, O)
    return {
        "success": env.is_success(traj, task.goal),
        "mean_cost": total_cost / max(steps, 1),
        "wall_s": time.perf_counter() - started,
    }


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. values: one number per training seed.

    With 5 seeds, this has at most 126 distinct resamples and is known to be somewhat narrow. It is
    what the protocol specifies; report per-seed values next to it.
    """
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(a, b, threshold: float = EFFECT_THRESHOLD, n_boot: int = 10000, seed: int = 0) -> dict:
    """Protocol verdict for lower-is-better metrics (error, failure rate), per training seed.

    A side wins only if the CI of mean(a - b) excludes 0 and its reduction relative to the other
    side's mean is at least the threshold.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired metrics must align by seed, got {a.shape} and {b.shape}")
    diff = a - b
    result = {
        "n_seeds": int(len(a)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "per_seed_diff": diff.tolist(),
        "seed_std_diff": float(diff.std(ddof=1)) if len(a) > 1 else None,
    }
    if len(a) < 2:
        result.update(ci_low=None, ci_high=None, reduction=None, verdict="insufficient_seeds")
        return result
    low, high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    ma, mb = result["mean_a"], result["mean_b"]
    verdict = "no_detectable_difference"
    reduction = 0.0
    if ma < mb and mb != 0:
        reduction = (mb - ma) / abs(mb)
        if high < 0 and reduction >= threshold:
            verdict = "a_wins"
    elif mb < ma and ma != 0:
        reduction = (ma - mb) / abs(ma)
        if low > 0 and reduction >= threshold:
            verdict = "b_wins"
    result.update(ci_low=low, ci_high=high, reduction=reduction, verdict=verdict)
    return result


@torch.no_grad()
def feature_collinearity(predictor, data: Transitions, lambda_: float = 1e-3) -> dict:
    """Collinearity of phi over adaptation data, averaged over environments (H3).

    With fewer transitions than features Phi'Phi is singular, so a raw condition number is meaningless.
    Reported instead:
        log10_cond_reg       log10 of (eig_max + lambda) / (eig_min + lambda)
        log10_cond_nonzero   log10 of the condition number over eigenvalues above 1e-10 eig_max
        participation_ratio  (sum eig)^2 / sum eig^2, the effective number of feature directions
        top_eig_fraction     share of total energy in the top direction

    Reordering transitions changes none of these. Only different data can reduce collinearity.
    """
    phi = predictor.features(data.obs, data.act).double()  # (E, N, F)
    eig = torch.linalg.eigvalsh(phi.transpose(1, 2) @ phi).clamp_min(0.0)  # (E, F), ascending
    top = eig[:, -1].clamp_min(1e-30)
    cond_reg = (eig[:, -1] + lambda_) / (eig[:, 0] + lambda_)
    nonzero_min = torch.where(eig > 1e-10 * top[:, None], eig, top[:, None]).min(dim=1).values
    participation = eig.sum(1) ** 2 / (eig**2).sum(1).clamp_min(1e-30)
    return {
        "log10_cond_reg": cond_reg.log10().mean().item(),
        "log10_cond_nonzero": (top / nonzero_min).log10().mean().item(),
        "participation_ratio": participation.mean().item(),
        "top_eig_fraction": (eig[:, -1] / eig.sum(1).clamp_min(1e-30)).mean().item(),
        "n_transitions": data.obs.shape[1],
        "feature_dim": phi.shape[2],
    }


@dataclass
class ValSet:
    data: Transitions
    holdout: Trajectories


def select_hyperparameters(
    method: str,
    backbone: Backbone,
    val_sets: list[ValSet],
    obs_scale: Tensor,
    obs_low: Tensor,
    obs_high: Tensor,
    grid: list[dict] | None = None,
    horizon: int = 10,
    starts=START_STEPS,
) -> dict:
    """Pick the grid entry with the lowest mean validation error relative to no adaptation.

    The score is the mean over val_sets of (method error / frozen error) at one horizon. Using a ratio
    stops the set with larger errors from deciding alone. Non-finite scores never win. Returns the
    choice and the whole table, for the record.
    """
    grid = METHODS[method].grid if grid is None else grid

    def error(predictor, val: ValSet) -> float:
        errs = open_loop_errors(predictor, val.holdout, obs_scale, obs_low, obs_high, (horizon,), starts)
        return errs[horizon].mean().item()

    reference = None
    if "frozen" in METHODS and backbone.kind in METHODS["frozen"].backbones:
        reference = [error(adapt("frozen", backbone, v.data).predictor, v) for v in val_sets]
    table, best_hp, best_score = [], None, float("inf")
    for hp in grid:
        errors = [error(adapt(method, backbone, v.data, **hp).predictor, v) for v in val_sets]
        if reference is None:
            score = float(np.mean(errors))
        else:
            score = float(np.mean([e / max(r, 1e-12) for e, r in zip(errors, reference)]))
        table.append({"hp": hp, "score": score, "errors": errors})
        if math.isfinite(score) and score < best_score:
            best_score, best_hp = score, hp
    if best_hp is None:
        best_hp = grid[0]  # every entry diverged; recorded through the table
    return {"hp": best_hp, "score": best_score, "table": table, "horizon": horizon, "relative_to_frozen": reference is not None}
