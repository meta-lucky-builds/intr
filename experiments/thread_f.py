"""Thread F and the coverage part of Thread B: H1r, H12, H13, H5 (docs/HYPOTHESES.md, "Second round").

    python experiments/thread_f.py --smoke     # checks the pipeline; numbers mean nothing
    python experiments/thread_f.py --full      # seeds 5-9 on pendulum and drone. Puck runs only when asked: --families puck

Writes results/thread_f/thread_f_<mode>.json after every (seed, family) job. Second-round hypotheses are post hoc, so they
are tested on seeds 5-9 and on a new family, never on the seeds that formed them.

A job, per training seed and family:
    1. Train latent, nolatent and the matched control with Thread A's functions and settings.
    2. Select on val_in and val_shift only. Wider grids than round one, each with a no-write entry (Problem 7).
    3. Per test tier, one set of adaptation trajectories, held-out trajectories and tasks serves every arm of a comparison
       (Problem 12). data_sha256 records this. Open loop at every K; closed loop at K = 1.
    4. H13: thread_b's prequential streams, one feature stream per cell, replayed by every method.
    5. H5: every arm starts each trajectory from the same initial states, with the same number of transitions.

The registered rules live in pure functions (h1r_rule, h12_rule, h13_rule, h5_rule), so stored outcomes can be read against
HYPOTHESES.md and are tested directly (Problems 2-4). FIXED_CHOICES states what HYPOTHESES.md leaves open.

Scope: small state-based simulators on CPU. Nothing here supports a claim about pixels, large models, or real robots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import intr.ebayes as ebayes  # noqa: E402  (registers ebrls and hybrid_eb)
from intr.adapt import METHODS, AdaptedPredictor, RLSState, _start_z, adapt  # noqa: E402
from intr.calibration import (  # noqa: E402
    COVERAGE_LEVELS,
    Schedule,
    build_stream,
    coverage,
    gaussian_nll,
    prepare_stream,
    prequential_rollout_rmse,
    query_signals,
    rmse_over,
    rollout_starts,
    run_prequential,
    spearman,
    stream_metrics,
    windup_summary,
)
from intr.envs import (  # noqa: E402
    BatchEnv,
    Trajectories,
    Transitions,
    _policy_action,
    collect_iid_states,
    collect_trajectories,
    derive_seed,
    make_env,
    make_generator,
)
from intr.eval import (  # noqa: E402
    EFFECT_THRESHOLD,
    HORIZONS,
    START_STEPS,
    OraclePredictor,
    PersistencePredictor,
    ValSet,
    closed_loop,
    collect_holdout,
    compute_obs_scale,
    open_loop_errors,
    select_hyperparameters,
)
from intr.looped import saturation_reference  # noqa: E402
from intr.models import Counts, ModelConfig, clip_obs  # noqa: E402
from intr.train import (  # noqa: E402
    Backbone,
    TransitionData,
    continued_training,
    latent_r2,
    train_latent,
    train_nolatent,
)
from thread_a import (  # noqa: E402
    PRIMARY_HORIZON,
    PRIMARY_K,
    Condition,
    ThreadAConfig,
    _errors_record,
    _jsonable,
    cem_for,
    smoke_config as a_smoke_config,
    source_fingerprint,
    val_sets_for,
)
from thread_b import (  # noqa: E402
    NO_SWITCH_KIND,
    PRIMARY_POST_N,
    ROLLOUT_KEY,
    _is_finite,
    _paired,
    _pick,
    _score_entry,
    _stat,
    default_schedules,
    evaluate_method,
    make_policy,
    validation_streams,
)

ORACLE_GATE = 0.5  # HYPOTHESES.md, second round, for closed-loop validity and for the family gates
LATENT_GATE = 0.10  # the latent must lower training loss by this much
WEAK_WRITE_EDF = 1.0  # "picks weak writes": median effective written directions at or below this
DESIGN_LAMBDA_MAX = 100.0  # first round's top value; D-optimal design at a no-write lambda would only maximize ||phi||
H5_REGISTERED_FAMILIES = ("pendulum", "drone")
H13_METHODS = ("episode_reset", "persist", "forgetting", "surprise_reset", "prior_forgetting")
H13_REFERENCES = ("oracle_reset", "frozen")
H5_ARMS = ("default", "random_action", "dopt", "iid_state")
USABLE = ("a_wins", "b_wins", "no_detectable_difference")
SCOPE_LIMIT = "Small state-based simulators on CPU; no claim about pixels, large models, or real robots."

FIXED_CHOICES = {
    "no_worse": "Protocol reading: worse means the paired CI excludes 0 and the difference is at least 10% (verdict b_wins).",
    "rls_comparator": "RLS in H1r and H12 is the better of rls@latent and rls@nolatent by raw validation h10, per seed (fixes Problem 4).",
    "better_single": (
        "H12's 'better single method': per seed, the lower validation error of latent_opt and the better RLS, on val_in for interp "
        "and val_shift for extrap and oof. The kill also fires on H12's named comparisons (latent_opt on interp, RLS on oof). "
        "The same verdict against the lower test mean is stored as a sensitivity, and comparator_sensitive flags a disagreement."
    ),
    "hybrid_baseline": (
        "H2's hybrid gets a stronger search than round one: lambda over the wide grid and lr_scale, both on validation. "
        "hybrid_eb validates lr_scale only, since that belongs to the latent stage; its lambda and sigma^2 never see validation. "
        "A win over this baseline is conservative. latent_opt also gets a wider lr_scale grid."
    ),
    "writes_do_not_help": "A family where writes do not help: the better RLS does not win against its frozen model in any tier.",
    "weak_writes": f"Weak write: median effective written directions (edf) at most {WEAK_WRITE_EDF} in every tier.",
    "h13_metric": "H13 is decided on prequential one-step RMSE, as H10 was, over all switching cells. Horizon 10 gets its own outcome field.",
    "h13_kill": "Kill 'no better than plain forgetting': zero switching cells where prior forgetting wins against plain, with no numerical failure.",
    "h13_update": "Kalman filter of an OU drift whose stationary law is the prior. See intr/ebayes.py.",
    "h5_tiers": (
        "H5 names no tier. Per tier, K = 1, h10, against random actions. Killed if no tier gains. Supported if all tiers gain against "
        "both random and default. Otherwise partial, with the tiers listed."
    ),
    "h5_manipulation": (
        "Before reading H5, check that D-optimal raised log det of precision over random, CI excluding 0. The kill is computed as "
        "written either way; the outcome says whether coverage moved, so that a null is not read as evidence about coverage if it did not."
    ),
    "h5_design": (
        "Receding horizon. Score candidate action sequences by the sum of log(1 + phi' P phi) on predicted features, with P updated on "
        "predicted features inside the sequence, which is the log det gain by the determinant lemma. Execute the first action in the true "
        "simulator, then write the real transition. Trajectories of K > 1 run in sequence and share P. Features come from the frozen "
        "backbone; predicted next states use the current online W. Design lambda is the selected RLS lambda, capped at "
        f"{DESIGN_LAMBDA_MAX}. Each arm's final lambda is selected on validation data from the same kind of probe."
    ),
    "closed_loop": (
        f"Closed-loop verdicts need tier-mean oracle success of at least {ORACLE_GATE}, and compare failure rate on the tasks the oracle "
        "solved (same tasks and planning noise across arms). Below the gate they are reported as invalid. They decide no hypothesis."
    ),
    "family_gates": (
        f"Oracle interp success at least {ORACLE_GATE}, and latent lowers exact training loss by at least {LATENT_GATE:.0%}. "
        f"Puck also needs frozen@latent closed-loop interp at least {ORACLE_GATE} (G1). Outcomes are computed regardless (rule_outcome); "
        "gated_outcome is invalid when these fail."
    ),
    "evidence_diagnostics": (
        "Risks stored, deciding nothing: selected values per tier, width of the 2-nat lambda set, lag-1 autocorrelation of whitened "
        "innovations and N_eff, held-out coverage and NLL of the evidence's own sigma^2 against no write, and a lambda response curve on "
        "the test tier against h10 (lambda_curve), which reads test data and so is a measurement, never a choice."
    ),
}


@dataclass
class ThreadFConfig(ThreadAConfig):
    families: list[str] = field(default_factory=lambda: ["pendulum", "drone"])
    open_loop_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    h3_tiers: list[str] = field(default_factory=list)  # inherited field, unused here
    val_lambdas: list[float] = field(default_factory=lambda: [10.0**e for e in range(-6, 7)] + [ebayes.NO_WRITE_LAMBDA])
    lr_scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    curve_lambdas: list[float] = field(default_factory=lambda: [10.0**e for e in range(-6, 7)] + [ebayes.NO_WRITE_LAMBDA])
    # H13, thread_b's streams
    h13_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    schedules: list[Schedule] = field(default_factory=default_schedules)
    val_schedule: Schedule = field(default_factory=lambda: Schedule("varied", "varied", min_interval=20, max_interval=300))
    stream_steps: int = 800
    n_streams: int = 21
    n_val_streams: int = 20
    stream_lambdas: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3])
    forgetting_grid: list[float] = field(default_factory=lambda: [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995])
    surprise_thresholds: list[float] = field(default_factory=lambda: [1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
    post_switch_ns: list[int] = field(default_factory=lambda: [5, PRIMARY_POST_N])
    steady_after: int = 60
    eig_every: int = 50
    rollout_every: int = 10
    # H5
    h5_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    dopt_candidates: int = 16
    dopt_horizon: int = 4


def full_config() -> ThreadFConfig:
    return ThreadFConfig(mode="full", seeds=[5, 6, 7, 8, 9])


def smoke_config() -> ThreadFConfig:
    a = a_smoke_config()
    grid = [1e-3, 1.0, ebayes.NO_WRITE_LAMBDA]
    return ThreadFConfig(
        mode="smoke",
        seeds=[5, 6],
        z_dim=a.z_dim,
        hidden=a.hidden,
        feature_dim=a.feature_dim,
        train=a.train,
        train_envs=a.train_envs,
        train_traj=a.train_traj,
        train_steps=a.train_steps,
        n_eval_envs=a.n_eval_envs,
        n_val_envs=a.n_val_envs,
        adapt_steps=a.adapt_steps,
        holdout_traj=a.holdout_traj,
        n_candidates=a.n_candidates,
        n_elites=a.n_elites,
        n_iters=a.n_iters,
        horizon_override=a.horizon_override,
        episode_override=a.episode_override,
        n_boot=a.n_boot,
        val_lambdas=grid,
        lr_scales=[1.0],
        curve_lambdas=grid,
        schedules=[Schedule("none", NO_SWITCH_KIND), Schedule("fast", "fixed", interval=35), Schedule("aligned", "aligned")],
        stream_steps=150,
        n_streams=6,
        n_val_streams=4,
        stream_lambdas=[0.1, 10.0],
        forgetting_grid=[0.95, 0.99],
        surprise_thresholds=[3.0],
        dopt_candidates=6,
        dopt_horizon=2,
    )


def conditions_for(cfg: ThreadFConfig) -> list[Condition]:
    rls_grid = [{"lambda_": lam} for lam in cfg.val_lambdas]
    lr_grid = [{"lr_scale": s} for s in cfg.lr_scales]
    hybrid_grid = [{"lambda_": lam, "lr_scale": s} for lam in cfg.val_lambdas for s in cfg.lr_scales]
    return [
        Condition("frozen@latent", "frozen", "latent"),
        Condition("frozen@nolatent", "frozen", "nolatent"),
        Condition("frozen@control", "frozen", "control"),
        Condition("latent_opt@latent", "latent_opt", "latent", lr_grid),
        Condition("rls@latent", "rls", "latent", rls_grid),
        Condition("rls@nolatent", "rls", "nolatent", rls_grid),
        Condition("hybrid@latent", "hybrid", "latent", hybrid_grid),
        Condition("ebrls@latent", "ebrls", "latent"),
        Condition("ebrls@nolatent", "ebrls", "nolatent"),
        Condition("hybrid_eb@latent", "hybrid_eb", "latent", lr_grid),
    ]


# ---------------------------------------------------------------- training and small helpers


@torch.no_grad()
def exact_train_losses(backbones: dict[str, Backbone], data: TransitionData) -> dict:
    """MSE over all training transitions, so gate G1 does not read a noisy minibatch log. latent uses its own z per environment."""
    latent = backbones["latent"]
    N = data.obs.shape[1]
    target = latent.model.target(data.obs, data.next_obs)
    own_z = latent.z_table[:, None, :].expand(-1, N, -1)
    mean_z = latent.z_default[None, None, :].expand(data.obs.shape[0], N, -1)
    out = {
        "latent": ((latent.model(data.obs, data.act, own_z) - target) ** 2).mean().item(),
        "latent_at_mean_z": ((latent.model(data.obs, data.act, mean_z) - target) ** 2).mean().item(),
    }
    for name in ("nolatent", "control"):
        model = backbones[name].model
        out[name] = ((model(data.obs, data.act) - model.target(data.obs, data.next_obs)) ** 2).mean().item()
    return out


def train_backbones(family: str, seed: int, cfg: ThreadFConfig) -> tuple[dict[str, Backbone], dict]:
    """thread_a.train_all without context. Each trainer seeds itself, so skipping it changes nothing else."""
    gen = make_generator("data", family, seed)
    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    data = TransitionData.collect(env, cfg.train_traj, cfg.train_steps, gen)
    with_z = ModelConfig(env.obs_dim, env.act_dim, cfg.z_dim, cfg.hidden, cfg.n_hidden, cfg.feature_dim)
    no_z = ModelConfig(env.obs_dim, env.act_dim, 0, cfg.hidden, cfg.n_hidden, cfg.feature_dim)
    latent = train_latent(data, with_z, cfg.train, seed)
    nolatent = train_nolatent(data, no_z, cfg.train, seed)
    control = continued_training(nolatent, data, latent.stats["latent_updates"], cfg.train, seed)
    backbones = {"latent": latent, "nolatent": nolatent, "control": control}
    record = {name: {"stats": b.stats, "final_log": b.log[-3:]} for name, b in backbones.items()}
    record["latent_r2"] = dict(zip(env.hidden_names, latent_r2(latent.z_table, env.hidden_matrix())))
    record["exact_train_loss"] = exact_train_losses(backbones, data)
    return backbones, record


def data_sha256(data: Transitions) -> str:
    digest = hashlib.sha256()
    for x in (data.obs, data.act, data.next_obs):
        digest.update(x.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _val_row(selection: dict) -> list[float]:
    """Raw validation errors of the chosen entry, [val_in, val_shift]. Scores are ratios to each backbone's frozen model."""
    for row in selection["table"]:
        if row["hp"] == selection["hp"]:
            return row["errors"]
    return [float("inf"), float("inf")]


def _better_by_val(selection: dict, labels: tuple[str, str]) -> str:
    return min(labels, key=lambda label: float(np.mean(_val_row(selection[label]))))


def saturation_for(holdout: Trajectories, scale: Tensor) -> dict:
    E, B, _, _ = holdout.act.shape
    O = holdout.obs.shape[-1]
    H = max(HORIZONS)
    true = torch.stack([holdout.obs[:, :, s : s + H + 1] for s in START_STEPS], dim=2).reshape(E, B * len(START_STEPS), H + 1, O)
    return saturation_reference(true, HORIZONS, scale, len(START_STEPS))


def run_closed(env: BatchEnv, predictor, tag, cfg: ThreadFConfig, oracle_mask: Tensor | None = None) -> dict:
    """Same tasks and planning noise for every arm in a tier. oracle_mask (E,) bool marks tasks the oracle solved."""
    predictor.counts = Counts()
    task = env.sample_task(make_generator((tag, "task")))
    out = closed_loop(env, predictor, task, cem_for(env, cfg), make_generator(tag, "cem"), cfg.episode_override)
    success = out["success"].bool()
    record = {
        "success_rate": success.float().mean().item(),
        "failure_rate": 1.0 - success.float().mean().item(),
        "mean_cost": out["mean_cost"].mean().item(),
        "per_env_success": success.tolist(),
        "per_env_cost": out["mean_cost"].tolist(),
        "wall_s": out["wall_s"],
        "counts": predictor.counts.as_dict(),
    }
    if oracle_mask is not None:
        solved = oracle_mask.sum().item()
        record["failure_rate_on_oracle_solved"] = (
            1.0 - success[oracle_mask].float().mean().item() if solved > 0 else float("nan")
        )
    return record


# ---------------------------------------------------------------- H12 diagnostics


@torch.no_grad()
def evidence_checks(predictor: AdaptedPredictor, data: Transitions, holdout: Transitions, eb: dict) -> dict:
    """Did the evidence's own confidence hold? Innovation whiteness on the adaptation data; coverage and NLL on held-out data."""
    model = predictor.model
    E = data.obs.shape[0]
    phi = predictor.features(data.obs, data.act)
    v = model.target(data.obs, data.next_obs)
    W0 = model.W.detach()[None].expand(E, -1, -1)
    lambda_ = torch.tensor(eb["lambda"], dtype=torch.float64)  # (E,)
    sigma2 = torch.tensor(eb["sigma2"], dtype=torch.float64)  # (E, D)
    innovations = ebayes.innovation_diagnostics(phi, v, W0, lambda_, sigma2)

    sig = query_signals(predictor, holdout)
    per_env_sigma2 = sigma2[:, None, :]  # (E, 1, D), broadcasts against (E, N, D)
    frozen_residual = model.target(holdout.obs, holdout.next_obs).double() - predictor.features(holdout.obs, holdout.act).double() @ W0.double().transpose(1, 2)
    fit = fit_for_no_write(phi, v, W0)
    return {
        "innovations": innovations,
        "heldout_coverage": {str(level): coverage(sig.residual, sig.q, per_env_sigma2, level).mean().item() for level in COVERAGE_LEVELS},
        "heldout_nll": gaussian_nll(sig.residual, sig.q, per_env_sigma2).mean().item(),
        "heldout_nll_no_write": gaussian_nll(frozen_residual, None, fit[:, None, :]).mean().item(),
        "heldout_coverage_no_write": {str(level): coverage(frozen_residual, None, fit[:, None, :], level).mean().item() for level in COVERAGE_LEVELS},
    }


def fit_for_no_write(phi: Tensor, v: Tensor, W0: Tensor) -> Tensor:
    """The no-write model's profiled sigma^2, (E, D)."""
    phi64, v64 = phi.double(), v.double()
    residual = v64 - phi64 @ W0.double().transpose(1, 2)
    return ebayes.profile_sigma2((residual**2).sum(dim=1), phi.shape[1], "shared")


@torch.no_grad()
def lambda_curve(predictor: AdaptedPredictor, data: Transitions, holdout: Trajectories, scale, low, high, lambdas: list[float]) -> dict:
    """Does evidence on one-step residuals track h10 as lambda moves? This reads test data, so it is a measurement only."""
    model = predictor.model
    E = data.obs.shape[0]
    phi = predictor.features(data.obs, data.act).double()
    v = model.target(data.obs, data.next_obs).double()
    W0 = model.W.detach().double()[None].expand(E, -1, -1)
    grid = ebayes.evidence_grid(phi, v - phi @ W0.transpose(1, 2), lambdas)
    errors = []
    for i in range(grid.lambdas.shape[0]):
        lam = grid.lambdas[i].expand(E)
        W, P = ebayes.posterior(grid.spectrum, W0, lam)
        candidate = AdaptedPredictor(model, z=predictor.z, rls=ebayes.rls_state_from_posterior(W, P))
        errors.append(open_loop_errors(candidate, holdout, scale, low, high, (PRIMARY_HORIZON,))[PRIMARY_HORIZON])
    err = torch.stack(errors, dim=1)  # (E, L)
    best_fixed = err.mean(dim=0).min().item()
    agreement = [spearman(grid.log_evidence[i].numpy(), (-err[i]).numpy()) for i in range(E)]
    finite = [x for x in agreement if _is_finite(x)]
    return {
        "lambdas": grid.lambdas.tolist(),
        "mean_h10_by_lambda": err.mean(dim=0).tolist(),
        "h10_best_fixed_lambda": best_fixed,
        "h10_best_per_env_lambda": err.min(dim=1).values.mean().item(),
        "evidence_vs_error_spearman": float(np.mean(finite)) if finite else float("nan"),
    }


# ---------------------------------------------------------------- H5 probes


def probe_from_state(env: BatchEnv, state: Tensor, n_steps: int, policy: str, gen) -> Trajectories:
    """collect_trajectories from given initial states. With the same generator after reset, the output is bit-identical to it."""
    o = env.obs(state)
    obs, act = [o], []
    noise = None
    for _ in range(n_steps):
        a, noise = _policy_action(env, policy, o, noise, gen)
        state = env.step(state, a)
        o = env.obs(state)
        obs.append(o)
        act.append(a)
    return Trajectories(torch.stack(obs, dim=2), torch.stack(act, dim=2))


def sample_action_sequences(E: int, C: int, H: int, A: int, gen) -> Tensor:
    """(E, C, H, A) in [-1, 1]. A quarter constant at corners, a quarter constant inside, and the rest iid per step."""
    seqs = (2.0 * torch.rand(E, C, 1, A, generator=gen) - 1.0).expand(-1, -1, H, -1).clone()
    n_iid = C // 2
    if H > 1 and n_iid > 0:
        seqs[:, C - n_iid :] = 2.0 * torch.rand(E, n_iid, H, A, generator=gen) - 1.0
    n_corner = C // 4
    if n_corner > 0:
        seqs[:, :n_corner] = torch.sign(seqs[:, :n_corner])
    return seqs


@torch.no_grad()
def dopt_scores(predictor, o: Tensor, seqs: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Predicted log det gain of each candidate sequence: sum_h log(1 + phi_h' P_h phi_h), with P_h updated on predicted features.

    o (E, O), seqs (E, C, H, A); returns (E, C). predictor needs features, model.apply_delta and rls (P, W32), and is not changed.
    """
    E, C, H, _ = seqs.shape
    P = predictor.rls.P[:, None]  # (E, 1, F, F), broadcast until the first update makes it per-candidate
    W = predictor.rls.W32.transpose(1, 2)  # (E, F, D)
    obs = o[:, None].expand(E, C, -1)
    score = torch.zeros(E, C, dtype=torch.float64)
    for h in range(H):
        phi = predictor.features(obs, seqs[:, :, h])  # (E, C, F)
        p64 = phi.to(torch.float64)
        u = (P @ p64[..., None]).squeeze(-1)  # (E, C, F)
        q = (p64 * u).sum(-1).clamp_min(0.0)  # (E, C)
        score = score + torch.log1p(q)
        if h + 1 < H:
            P = P - u[..., None] * u[..., None, :] / (1.0 + q)[..., None, None]
            obs = clip_obs(predictor.model.apply_delta(obs, phi @ W), low, high)
    return score


@torch.no_grad()
def collect_dopt(env: BatchEnv, s0: Tensor, n_steps: int, predictor, n_candidates: int, horizon: int, gen, low: Tensor, high: Tensor) -> tuple[Trajectories, dict]:
    """Greedy receding-horizon D-optimal probe. s0 (E, B, S). Real transitions go through the true simulator and update the RLS state.

    predictor.rls is the design state, written in place. Trajectories run in sequence, so later ones benefit from earlier ones.
    """
    E, B, _ = s0.shape
    rows = torch.arange(E)
    started = time.perf_counter()
    obs_all, act_all = [], []
    for b in range(B):
        state = s0[:, [b]]  # (E, 1, S)
        o = env.obs(state)[:, 0]  # (E, O)
        obs, act = [o], []
        for _ in range(n_steps):
            seqs = sample_action_sequences(E, n_candidates, horizon, env.act_dim, gen)
            best = dopt_scores(predictor, o, seqs, low, high).argmax(dim=1)
            a = seqs[rows, best, 0]  # (E, A)
            state = env.step(state, a[:, None, :])
            next_o = env.obs(state)[:, 0]
            predictor.observe(o, a, next_o)
            obs.append(next_o)
            act.append(a)
            o = next_o
        obs_all.append(torch.stack(obs, dim=1))  # (E, T + 1, O)
        act_all.append(torch.stack(act, dim=1))
    info = {
        "wall_s": time.perf_counter() - started,
        "forward_samples": E * B * n_steps * n_candidates * horizon,
        "n_candidates": n_candidates,
        "horizon": horizon,
    }
    return Trajectories(torch.stack(obs_all, dim=1), torch.stack(act_all, dim=1)), info


def design_predictor(backbone: Backbone, n_envs: int, design_lambda: float) -> AdaptedPredictor:
    z = _start_z(backbone, n_envs)
    W0 = backbone.model.W.detach()[None].expand(n_envs, -1, -1)
    return AdaptedPredictor(backbone.model, z=z, rls=RLSState(W0, design_lambda))


def collect_arm(arm: str, env: BatchEnv, tag, max_k: int, cfg: ThreadFConfig, backbone: Backbone, design_lambda: float, low, high) -> tuple[Trajectories, dict]:
    """One probe arm with max_k trajectories. default, random_action and dopt share initial states, by construction of gen."""
    n = cfg.adapt_steps
    if arm == "iid_state":  # the ceiling; not reachable by a real robot
        return collect_iid_states(env, max_k * n, make_generator(tag, "iid_state")), {}
    gen = make_generator(tag, "probe")
    s0 = env.reset(max_k, gen)  # (E, max_k, S), the same draw collect_trajectories makes first
    if arm == "default":
        return probe_from_state(env, s0, n, env.default_probe, gen), {}
    if arm == "random_action":
        return probe_from_state(env, s0, n, "iid", make_generator(tag, "h5", "random_action")), {}
    if arm == "dopt":
        predictor = design_predictor(backbone, env.n_envs, design_lambda)
        return collect_dopt(env, s0, n, predictor, cfg.dopt_candidates, cfg.dopt_horizon, make_generator(tag, "h5", "dopt"), low, high)
    raise ValueError(f"unknown arm {arm!r}")


def cut_arm(arm: str, trajs: Trajectories, k: int, n_steps: int) -> Transitions:
    """iid_state stores one-step trajectories, so K trajectories mean K * n of them. Every arm returns K * n transitions."""
    return trajs.take(k * n_steps if arm == "iid_state" else k).flat()


@torch.no_grad()
def coverage_measures(backbone: Backbone, data: Transitions, holdout: Transitions, lambda_: float) -> dict:
    """The manipulation check. All at one lambda, on the frozen features, so arms are comparable."""
    E = data.obs.shape[0]
    frozen = AdaptedPredictor(backbone.model, z=_start_z(backbone, E))
    phi = frozen.features(data.obs, data.act).double()
    gain = ebayes.logdet_precision_gain(phi, lambda_)  # (E,)
    P = ebayes.posterior_covariance(phi, lambda_)  # (E, F, F)
    query = frozen.features(holdout.obs, holdout.act).double()  # (E, Q, F)
    q = ((query @ P) * query).sum(-1).clamp_min(0.0)  # (E, Q)
    return {
        "lambda": lambda_,
        "logdet_gain": gain.mean().item(),
        "logdet_gain_per_env": gain.tolist(),
        "query_log1p_q": torch.log1p(q).mean().item(),  # lower: the data cover the held-out queries better
        "state_spread": backbone.model.norm_obs(data.obs).std(dim=1).mean().item(),
    }


def run_h5(
    family: str, seed: int, backbone: Backbone, label: str, selection: dict, contexts: dict, scale, low, high, cfg: ThreadFConfig
) -> dict:
    started = time.perf_counter()
    selected = selection[label]["hp"]["lambda_"]
    design_lambda = min(selected, DESIGN_LAMBDA_MAX)
    grid = [{"lambda_": lam} for lam in cfg.val_lambdas]

    # Final lambda per arm, on validation data from the same kind of probe. default is what the main selection already used.
    arm_selection = {"default": selection[label]}
    for arm in H5_ARMS[1:]:
        val = []
        for tier in ("val_in", "val_shift"):
            env = make_env(family, cfg.n_val_envs, tier, derive_seed("val_envs", seed))
            tag = ("val", family, seed, tier)
            holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
            trajs, _ = collect_arm(arm, env, tag, PRIMARY_K, cfg, backbone, design_lambda, low, high)
            val.append(ValSet(cut_arm(arm, trajs, PRIMARY_K, cfg.adapt_steps), holdout))
        arm_selection[arm] = select_hyperparameters("rls", backbone, val, scale, low, high, grid=grid, horizon=PRIMARY_HORIZON)
    selection_wall = time.perf_counter() - started

    started = time.perf_counter()
    tiers = {}
    for tier in cfg.h5_tiers:
        ctx = contexts[tier]
        env, tag, holdout = ctx["env"], ctx["tag"], ctx["holdout"]
        holdout_flat = holdout.flat()
        trajectories, collect_info = {}, {}
        for arm in H5_ARMS:
            trajectories[arm], collect_info[arm] = collect_arm(arm, env, tag, max(cfg.ks), cfg, backbone, design_lambda, low, high)
        tier_record = {"k": {}, "collect": collect_info, "initial_obs_sha256": {}}
        for arm in ("default", "random_action", "dopt"):
            tier_record["initial_obs_sha256"][arm] = hashlib.sha256(trajectories[arm].obs[:, :, 0].numpy().tobytes()).hexdigest()
        for k in cfg.ks:
            k_record = {}
            for arm in H5_ARMS:
                data = cut_arm(arm, trajectories[arm], k, cfg.adapt_steps)
                hp = arm_selection[arm]["hp"]
                result = adapt("rls", backbone, data, **hp)
                entry = {
                    "hp": hp,
                    "n_transitions": data.obs.shape[1],
                    "data_sha256": data_sha256(data),
                    "adapt_counts": result.counts.as_dict(),
                    "adapt_wall_s": result.wall_s,
                    "open_loop": _errors_record(open_loop_errors(result.predictor, holdout, scale, low, high), env.mechanisms),
                    "coverage": coverage_measures(backbone, data, holdout_flat, design_lambda),
                }
                if k == PRIMARY_K and tier in cfg.closed_loop_tiers:
                    entry["closed_loop"] = run_closed(env, result.predictor, tag, cfg, ctx["oracle_mask"])
                k_record[arm] = entry
            tier_record["k"][str(k)] = k_record
        tiers[tier] = tier_record
    return {
        "backbone": label,
        "design_lambda": design_lambda,
        "selection": arm_selection,
        "tiers": tiers,
        "wall_s": {"selection": selection_wall, "eval": time.perf_counter() - started},
    }


# ---------------------------------------------------------------- H13


def _score_prior(val, hp: dict, reference: list[float]) -> dict:
    ratios, failed = [], []
    for fs, ref in zip(val, reference):
        result = ebayes.run_prequential_prior(fs.phi, fs.v, fs.W0, hp["lambda_"], hp["gamma"], bounds=fs.bounds)
        ratios.append(rmse_over(result.err).mean().item() / max(ref, 1e-12))
        failed.append((result.failures > 0).double().mean().item())
    return {"hp": hp, "score": float(np.mean(ratios)), "failed_stream_fraction": float(np.mean(failed))}


def select_h13(val, cfg: ThreadFConfig) -> dict:
    """On validation streams only. Plain and prior-preserving forgetting search the same lambda and gamma grids, so neither is favoured."""
    reference = []
    for fs in val:
        frozen = run_prequential(fs.phi, fs.v, fs.W0, make_policy("frozen", {}, fs.stream), bounds=fs.bounds)
        reference.append(rmse_over(frozen.err).mean().item())
    selection = {}
    for method in ("persist", "episode_reset"):
        selection[method] = _pick([_score_entry(val, method, {"lambda_": lam}, reference) for lam in cfg.stream_lambdas])
    lam_reset = selection["episode_reset"]["hp"]["lambda_"]
    selection["forgetting"] = _pick([
        _score_entry(val, "forgetting", {"lambda_": lam, "forgetting": gamma}, reference)
        for lam in cfg.stream_lambdas
        for gamma in cfg.forgetting_grid
    ])
    selection["prior_forgetting"] = _pick([
        _score_prior(val, {"lambda_": lam, "gamma": gamma}, reference)
        for lam in cfg.stream_lambdas
        for gamma in cfg.forgetting_grid
    ])
    selection["surprise_reset"] = _pick([
        _score_entry(val, "surprise_reset", {"lambda_": lam_reset, "threshold": th}, reference) for th in cfg.surprise_thresholds
    ])
    selection["oracle_reset"] = {"hp": {"lambda_": lam_reset}, "note": "reference; uses the lambda chosen for episode_reset"}
    selection["frozen"] = {"hp": {}, "note": "reference; no writes"}
    return selection


def evaluate_prior(fs, hp: dict, starts: list[int], scale, low, high, cfg: ThreadFConfig) -> dict:
    """thread_b.evaluate_method for prior_forgetting, with the same record."""
    started = time.perf_counter()
    result = ebayes.run_prequential_prior(
        fs.phi, fs.v, fs.W0, hp["lambda_"], hp["gamma"], bounds=fs.bounds, eig_every=cfg.eig_every, snapshot_at=starts
    )
    stream_wall = time.perf_counter() - started
    record = stream_metrics(result, fs.stream, cfg.post_switch_ns, cfg.steady_after)
    rollout_err, predictor = prequential_rollout_rmse(fs, result.snapshots, PRIMARY_HORIZON, scale, low, high)
    record[ROLLOUT_KEY] = {"mean": torch.nanmean(rollout_err).item(), "per_stream": rollout_err.tolist()}
    record["windup"] = windup_summary(result)
    record["hp"] = hp
    counts = Counts(writes=result.n_writes)
    if predictor is not None:
        counts.forward_calls = predictor.counts.forward_calls
        counts.forward_samples = predictor.counts.forward_samples
    record["counts_excluding_shared_features"] = counts.as_dict()
    record["wall_s"] = {"stream": stream_wall, "total": time.perf_counter() - started}
    return record


def run_h13(family: str, seed: int, backbone: Backbone, scale, low, high, cfg: ThreadFConfig) -> dict:
    started = time.perf_counter()
    val = validation_streams(family, seed, backbone, low, high, cfg)
    selection = select_h13(val, cfg)
    selection_wall = time.perf_counter() - started
    episode_steps = make_env(family, 1, "train", "bounds").episode_steps
    starts = rollout_starts(cfg.stream_steps, episode_steps, PRIMARY_HORIZON, cfg.rollout_every)

    started = time.perf_counter()
    tiers = {}
    for tier in cfg.h13_tiers:
        tiers[tier] = {}
        for schedule in cfg.schedules:
            stream = build_stream(
                family, tier, schedule, cfg.n_streams, cfg.stream_steps,
                derive_seed("stream_envs", seed), ("stream", family, seed, tier, schedule.name),
            )
            fs = prepare_stream(backbone, stream, low, high)
            methods = {}
            for method in H13_METHODS + H13_REFERENCES:
                hp = selection[method]["hp"]
                if method == "prior_forgetting":
                    methods[method] = evaluate_prior(fs, hp, starts, scale, low, high, cfg)
                else:
                    methods[method] = evaluate_method(fs, method, hp, starts, scale, low, high, cfg)
            tiers[tier][schedule.name] = {
                "kind": schedule.kind,
                "n_switches_per_stream": stream.switch.double().sum(dim=1).mean().item(),
                "shared_feature_forward_samples": fs.forward_samples,
                "methods": methods,
            }
    return {"selection": selection, "tiers": tiers, "wall_s": {"selection": selection_wall, "eval": time.perf_counter() - started}}


# ---------------------------------------------------------------- one job


def run_job(family: str, seed: int, cfg: ThreadFConfig) -> dict:
    started = time.perf_counter()
    backbones, train_record = train_backbones(family, seed, cfg)
    train_wall = time.perf_counter() - started
    print(f"  trained in {train_wall:.1f}s; exact loss {train_record['exact_train_loss']}", flush=True)
    scale = compute_obs_scale(family)
    bounds_env = make_env(family, 1, "train", "bounds")
    low, high = bounds_env.obs_low, bounds_env.obs_high
    conditions = conditions_for(cfg)

    started = time.perf_counter()
    val = val_sets_for(family, seed, cfg, "smooth")
    selection = {
        c.label: select_hyperparameters(c.method, backbones[c.backbone], val, scale, low, high, grid=c.grid, horizon=PRIMARY_HORIZON)
        for c in conditions
    }
    rls_best = _better_by_val(selection, ("rls@latent", "rls@nolatent"))
    ebrls_best = _better_by_val(selection, ("ebrls@latent", "ebrls@nolatent"))
    best_kind = rls_best.split("@")[1]
    val_errors = {label: _val_row(entry) for label, entry in selection.items()}
    selection_wall = time.perf_counter() - started
    print(f"  selected in {selection_wall:.1f}s; better RLS {rls_best}", flush=True)

    started = time.perf_counter()
    tiers, contexts = {}, {}
    for tier in cfg.open_loop_tiers:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
        probes = collect_trajectories(env, max(cfg.ks), cfg.adapt_steps, env.default_probe, make_generator(tag, "probe"))
        holdout_flat = holdout.flat()
        tier_record = {
            "mechanisms": env.mechanisms,
            "persistence": _errors_record(open_loop_errors(PersistencePredictor(), holdout, scale, low, high), env.mechanisms),
            "saturation": saturation_for(holdout, scale),
            "k": {},
            "lambda_curve": {},
        }
        run_cl = tier in cfg.closed_loop_tiers
        oracle_mask = None
        if run_cl:
            oracle = run_closed(env, OraclePredictor(env), tag, cfg)
            tier_record["oracle_closed_loop"] = oracle
            oracle_mask = torch.tensor(oracle["per_env_success"], dtype=torch.bool)
        contexts[tier] = {"env": env, "tag": tag, "holdout": holdout, "oracle_mask": oracle_mask}
        for k in cfg.ks:
            data = probes.take(k).flat()
            sha = data_sha256(data)
            k_record = {}
            for cond in conditions:
                hp = selection[cond.label]["hp"]
                result = adapt(cond.method, backbones[cond.backbone], data, **hp)
                entry = {
                    "hp": hp,
                    "data_sha256": sha,
                    "adapt_wall_s": result.wall_s,
                    "adapt_counts": result.counts.as_dict(),
                    "info": result.info,
                    "open_loop": _errors_record(open_loop_errors(result.predictor, holdout, scale, low, high), env.mechanisms),
                }
                if "eb" in result.info:
                    entry["evidence_checks"] = evidence_checks(result.predictor, data, holdout_flat, result.info["eb"])
                    if k == PRIMARY_K:
                        tier_record["lambda_curve"][cond.label] = lambda_curve(result.predictor, data, holdout, scale, low, high, cfg.curve_lambdas)
                if run_cl and k == PRIMARY_K:
                    entry["closed_loop"] = run_closed(env, result.predictor, tag, cfg, oracle_mask)
                k_record[cond.label] = entry
            if k == PRIMARY_K:
                # Sensitivity to the shared-sigma choice, open loop only. Same data, same lr_scale.
                hp = selection["hybrid_eb@latent"]["hp"]
                result = adapt("hybrid_eb", backbones["latent"], data, sigma_mode="per_output", **hp)
                k_record["hybrid_eb_per_output"] = {
                    "hp": hp,
                    "data_sha256": sha,
                    "adapt_wall_s": result.wall_s,
                    "adapt_counts": result.counts.as_dict(),
                    "info": result.info,
                    "open_loop": _errors_record(open_loop_errors(result.predictor, holdout, scale, low, high), env.mechanisms),
                }
            tier_record["k"][str(k)] = k_record
        tiers[tier] = tier_record
        print(f"  tier {tier} done at {time.perf_counter() - started:.1f}s", flush=True)
    eval_wall = time.perf_counter() - started

    h13 = run_h13(family, seed, backbones[best_kind], scale, low, high, cfg)
    print(f"  H13 done: selection {h13['wall_s']['selection']:.1f}s, eval {h13['wall_s']['eval']:.1f}s", flush=True)
    h5 = run_h5(family, seed, backbones[best_kind], rls_best, selection, contexts, scale, low, high, cfg)
    print(f"  H5 done: selection {h5['wall_s']['selection']:.1f}s, eval {h5['wall_s']['eval']:.1f}s", flush=True)
    return {
        "family": family,
        "train": train_record,
        "selection": selection,
        "val_errors": val_errors,
        "rls_best": rls_best,
        "ebrls_best": ebrls_best,
        "frozen_best": f"frozen@{best_kind}",
        "frozen_ebrls_best": f"frozen@{ebrls_best.split('@')[1]}",
        "tiers": tiers,
        "h13": h13,
        "h5": h5,
        "wall_s": {"train": train_wall, "selection": selection_wall, "eval": eval_wall},
    }


# ---------------------------------------------------------------- registered rules, pure and tested


def _usable(*verdicts) -> bool:
    return all(v in USABLE for v in verdicts)


def worse(verdict: str) -> bool:
    """a is worse than b, by the protocol: CI excludes 0 and the difference is at least 10%."""
    return verdict == "b_wins"


def h1r_rule(v_interp: str, v_oof: str) -> str:
    """a = latent_opt, b = the better RLS. Kill: one method wins both tiers."""
    if not _usable(v_interp, v_oof):
        return "insufficient_seeds"
    if v_interp == "a_wins" and v_oof == "b_wins":
        return "supported"
    if v_interp == v_oof and v_interp in ("a_wins", "b_wins"):
        return "killed_one_method_wins_both"
    return "not_supported_no_crossover"


def h12_rule(tiers: dict) -> dict:
    """a = hybrid_eb in every verdict. tiers: {tier: {vs_hybrid, vs_latent_opt, vs_rls, vs_better_single, vs_frozen, rls_vs_frozen, median_edf}}.

    Registered predictions: beats H2's hybrid on oof; no worse than latent_opt on interp and no worse than RLS on oof; where writes
    do not help, picks weak writes and is no worse than frozen. Kill: worse than the better single method in any tier.
    """
    named = ("vs_hybrid", "vs_latent_opt", "vs_rls", "vs_better_single", "vs_frozen", "rls_vs_frozen")
    if "interp" not in tiers or "oof" not in tiers or not all(_usable(*(t[n] for n in named)) for t in tiers.values()):
        return {"outcome": "insufficient_seeds"}
    worse_better_single = [t for t, row in tiers.items() if worse(row["vs_better_single"])]
    worse_latent_interp = worse(tiers["interp"]["vs_latent_opt"])
    worse_rls_oof = worse(tiers["oof"]["vs_rls"])
    killed = bool(worse_better_single) or worse_latent_interp or worse_rls_oof
    beats_h2_oof = tiers["oof"]["vs_hybrid"] == "a_wins"
    writes_help = any(row["rls_vs_frozen"] == "a_wins" for row in tiers.values())
    weak_clause = None
    if not writes_help:
        weak = all(_is_finite(row["median_edf"]) and row["median_edf"] <= WEAK_WRITE_EDF for row in tiers.values())
        weak_clause = weak and not any(worse(row["vs_frozen"]) for row in tiers.values())
    if killed:
        outcome = "killed"
    elif beats_h2_oof and weak_clause is not False:
        outcome = "supported"
    else:
        outcome = "not_supported_no_kill"
    return {
        "outcome": outcome,
        "kill_fired": killed,
        "worse_than_better_single_in": worse_better_single,
        "worse_than_latent_opt_on_interp": worse_latent_interp,
        "worse_than_rls_on_oof": worse_rls_oof,
        "beats_h2_hybrid_on_oof": beats_h2_oof,
        "writes_help_on_this_family": writes_help,
        "weak_writes_and_no_worse_than_frozen": weak_clause,  # None when writes do help, so the clause does not apply
    }


def h13_rule(cells: dict) -> dict:
    """a = prior_forgetting. cells: {cell: {vs_plain, vs_reset, vs_surprise, tainted}}, switching schedules only.

    Prediction: beats plain forgetting and matches or beats reset. Kill: no better than plain forgetting. A win with numerical
    failures does not count.
    """
    if not cells:
        return {"outcome": "incomplete"}
    if not all(_usable(c["vs_plain"], c["vs_reset"]) for c in cells.values()):
        return {"outcome": "insufficient_seeds"}
    beats_plain = [name for name, c in cells.items() if c["vs_plain"] == "a_wins" and not c["tainted"]]
    worse_than_reset = [name for name, c in cells.items() if worse(c["vs_reset"])]
    beats_reset = [name for name, c in cells.items() if c["vs_reset"] == "a_wins" and not c["tainted"]]
    if not beats_plain:
        spoiled = any(c["vs_plain"] == "a_wins" and c["tainted"] for c in cells.values())
        outcome = "inconclusive_numerical_failure" if spoiled else "killed"
    elif len(beats_plain) == len(cells) and not worse_than_reset:
        outcome = "supported"
    else:
        outcome = "partial"
    return {
        "outcome": outcome,
        "n_switching_cells": len(cells),
        "beats_plain": beats_plain,
        "worse_than_reset": worse_than_reset,
        "beats_reset": beats_reset,
        "surprise_beats_it": [name for name, c in cells.items() if worse(c.get("vs_surprise", ""))],
        "tainted": [name for name, c in cells.items() if c["tainted"]],
    }


def h5_rule(tiers: dict) -> dict:
    """a = dopt. tiers: {tier: {vs_random, vs_default, coverage_vs_random}}, K = 1, h10. coverage a_wins: dopt raised log det over random.

    Kill as written: no detectable gain over random-action probes at K = 1.
    """
    if not tiers:
        return {"outcome": "incomplete"}
    if not all(_usable(t["vs_random"], t["vs_default"], t["coverage_vs_random"]) for t in tiers.values()):
        return {"outcome": "insufficient_seeds"}
    gains = [t for t, row in tiers.items() if row["vs_random"] == "a_wins"]
    both = [t for t in gains if tiers[t]["vs_default"] == "a_wins"]
    moved = [t for t, row in tiers.items() if row["coverage_vs_random"] == "a_wins"]
    if not gains:
        outcome = "killed_coverage_moved_no_error_gain" if moved else "killed_coverage_did_not_move"
    elif len(both) == len(tiers):
        outcome = "supported"
    else:
        outcome = "partial"
    return {
        "outcome": outcome,
        "kill_fired": not gains,
        "gain_over_random_in": gains,
        "gain_over_both_in": both,
        "coverage_moved_in": moved,
        "manipulation_failed_in": [t for t in tiers if t not in moved],
        "gain_without_moved_coverage_in": [t for t in gains if t not in moved],
    }


def gated(rule_outcome: str, gates: dict) -> dict:
    return {
        "rule_outcome": rule_outcome,
        "family_gates_pass": gates["pass"],
        "gated_outcome": rule_outcome if gates["pass"] else "invalid_family_gates_failed",
    }


# ---------------------------------------------------------------- summaries


def _finite(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def _mean(values) -> float:
    x = _finite(values)
    return float(x.mean()) if x.size else float("nan")


def _median(values) -> float:
    x = _finite(values)
    return float(np.median(x)) if x.size else float("nan")


def _seeds(jobs: dict) -> list[str]:
    return sorted(jobs, key=int)


def resolve(job: dict, label: str, tier: str) -> str:
    """Per-seed comparators, each chosen on validation only."""
    if label in ("rls_best", "ebrls_best", "frozen_best", "frozen_ebrls_best"):
        return job[label]
    if label == "better_single":
        index = 0 if tier == "interp" else 1  # val_in for interp, val_shift for the shifted tiers
        rls = job["rls_best"]
        latent_err = job["val_errors"]["latent_opt@latent"][index]
        rls_err = job["val_errors"][rls][index]
        latent_err = latent_err if _is_finite(latent_err) else float("inf")
        rls_err = rls_err if _is_finite(rls_err) else float("inf")
        return "latent_opt@latent" if latent_err <= rls_err else rls
    return label


def _entry(job: dict, tier: str, k: int, label: str) -> dict:
    return job["tiers"][tier]["k"].get(str(k), {}).get(resolve(job, label, tier), {})


def series(jobs: dict, tier: str, k: int, label: str, metric: str, kind: str = "open_loop") -> list:
    return [_entry(jobs[s], tier, k, label).get(kind, {}).get(metric, float("nan")) for s in _seeds(jobs)]


def compare_open(jobs: dict, tier: str, k: int, a: str, b: str, n_boot: int) -> dict:
    return {f"h{h}": _paired(series(jobs, tier, k, a, f"h{h}"), series(jobs, tier, k, b, f"h{h}"), n_boot) for h in HORIZONS}


def oracle_gate_for(jobs: dict, tier: str) -> dict:
    per_seed = [jobs[s]["tiers"][tier].get("oracle_closed_loop", {}).get("success_rate", float("nan")) for s in _seeds(jobs)]
    mean = _mean(per_seed)
    return {"gate": ORACLE_GATE, "oracle_success": mean, "per_seed": per_seed, "valid": _is_finite(mean) and mean >= ORACLE_GATE}


def closed_verdict(gate: dict, a_failures: list, b_failures: list, n_boot: int) -> dict:
    """Failure rate on oracle-solved tasks, or a stated invalid. a_failures, b_failures: one number per seed."""
    if not gate["valid"]:
        return {"valid": False, "gate": gate["gate"], "oracle_success": gate["oracle_success"], "verdict": "invalid_oracle_gate"}
    return {"valid": True, "gate": gate["gate"], "oracle_success": gate["oracle_success"], **_paired(a_failures, b_failures, n_boot)}


def compare_closed(jobs: dict, tier: str, a: str, b: str, n_boot: int) -> dict:
    metric = "failure_rate_on_oracle_solved"
    return closed_verdict(
        oracle_gate_for(jobs, tier),
        series(jobs, tier, PRIMARY_K, a, metric, "closed_loop"),
        series(jobs, tier, PRIMARY_K, b, metric, "closed_loop"),
        n_boot,
    )


def summarize_gates(jobs: dict, tiers: list[str]) -> dict:
    """Numbers the family gates need, and whether they pass the registered thresholds. Train loss with and without the latent is exact."""
    seeds = _seeds(jobs)
    family = jobs[seeds[0]].get("family", "")
    latent = [jobs[s]["train"]["exact_train_loss"]["latent"] for s in seeds]
    nolatent = [jobs[s]["train"]["exact_train_loss"]["nolatent"] for s in seeds]
    mean_latent, mean_nolatent = _mean(latent), _mean(nolatent)
    reduction = 1.0 - mean_latent / mean_nolatent if _is_finite(mean_nolatent) and mean_nolatent > 0 else float("nan")
    oracle = {tier: oracle_gate_for(jobs, tier) for tier in tiers}
    frozen_success = {
        label: _mean(series(jobs, "interp", PRIMARY_K, label, "success_rate", "closed_loop")) if "interp" in tiers else float("nan")
        for label in ("frozen@latent", "frozen@nolatent", "frozen@control")
    }
    oracle_ok = "interp" in oracle and oracle["interp"]["valid"]
    latent_ok = _is_finite(reduction) and reduction >= LATENT_GATE
    needs_g1 = family == "puck"
    g1_ok = _is_finite(frozen_success["frozen@latent"]) and frozen_success["frozen@latent"] >= ORACLE_GATE
    return {
        "family": family,
        "oracle_gate": ORACLE_GATE,
        "latent_gate": LATENT_GATE,
        "oracle": oracle,
        "exact_train_loss": {
            "latent": latent,
            "nolatent": nolatent,
            "control": [jobs[s]["train"]["exact_train_loss"]["control"] for s in seeds],
            "latent_at_mean_z": [jobs[s]["train"]["exact_train_loss"]["latent_at_mean_z"] for s in seeds],
            "reduction_latent_vs_nolatent": reduction,
            "per_seed_reduction": [1.0 - a / b if _is_finite(a) and _is_finite(b) and b > 0 else float("nan") for a, b in zip(latent, nolatent)],
        },
        "latent_r2": {s: jobs[s]["train"]["latent_r2"] for s in seeds},
        "frozen_closed_loop_interp": frozen_success,
        "checks": {"oracle_interp": oracle_ok, "latent_lowers_loss": latent_ok, "g1_closed_loop": g1_ok, "g1_required": needs_g1},
        "pass": bool(oracle_ok and latent_ok and (g1_ok or not needs_g1)),
    }


def means_table(jobs: dict, tiers: list[str], ks: list[str]) -> dict:
    seeds = _seeds(jobs)
    out = {}
    for tier in tiers:
        out[tier] = {
            "saturation": {h: _mean([jobs[s]["tiers"][tier]["saturation"][h] for s in seeds]) for h in (f"h{x}" for x in HORIZONS)},
            "persistence": {h: _mean([jobs[s]["tiers"][tier]["persistence"][h] for s in seeds]) for h in (f"h{x}" for x in HORIZONS)},
            "k": {},
        }
        for k in ks:
            labels = sorted({label for s in seeds for label in jobs[s]["tiers"][tier]["k"].get(k, {})})
            out[tier]["k"][k] = {}
            for label in labels:
                ik = int(k)
                row = {f"h{h}": _mean(series(jobs, tier, ik, label, f"h{h}")) for h in HORIZONS}
                row["per_seed_primary"] = series(jobs, tier, ik, label, f"h{PRIMARY_HORIZON}")
                success = series(jobs, tier, ik, label, "success_rate", "closed_loop")
                if _finite(success).size:
                    row["success_rate"] = _mean(success)
                    row["failure_on_oracle_solved"] = _mean(series(jobs, tier, ik, label, "failure_rate_on_oracle_solved", "closed_loop"))
                entries = [jobs[s]["tiers"][tier]["k"].get(k, {}).get(label, {}) for s in seeds]
                row["adapt_wall_s"] = _mean([e.get("adapt_wall_s", float("nan")) for e in entries])
                row["adapt_counts"] = {
                    key: _mean([e.get("adapt_counts", {}).get(key, float("nan")) for e in entries])
                    for key in ("forward_calls", "forward_samples", "backward_calls", "writes")
                }
                out[tier]["k"][k][label] = row
    return out


def grid_edges(selection: dict) -> dict:
    """For each searched hyperparameter, whether the choice sits at an edge of its grid. Problem 7 made visible."""
    table = selection.get("table")
    if not table or len(table) < 2:
        return {}
    out = {}
    for key, chosen in selection["hp"].items():
        values = sorted({row["hp"][key] for row in table if key in row["hp"]})
        if len(values) > 1:
            out[key] = {"chosen": chosen, "at_low": chosen == values[0], "at_high": chosen == values[-1]}
    return out


def summarize_edges(jobs: dict) -> dict:
    seeds = _seeds(jobs)
    sources = {
        "main": lambda job: job["selection"],
        "h13": lambda job: job["h13"]["selection"],
        "h5": lambda job: job["h5"]["selection"],
    }
    out = {}
    for name, source in sources.items():
        out[name] = {}
        for label in source(jobs[seeds[0]]):
            rows = [grid_edges(source(jobs[s]).get(label, {})) for s in seeds]
            keys = sorted({key for row in rows for key in row})
            if keys:
                out[name][label] = {
                    key: {
                        "chosen_per_seed": [row.get(key, {}).get("chosen") for row in rows],
                        "n_at_low": sum(bool(row.get(key, {}).get("at_low")) for row in rows),
                        "n_at_high": sum(bool(row.get(key, {}).get("at_high")) for row in rows),
                    }
                    for key in keys
                }
    return out


def eb_selected(jobs: dict, tier: str, k: int, label: str, n_boot: int) -> dict:
    """Selected values and evidence checks for one arm in one tier. One number per seed, then _stat. Per-environment lists stay in jobs."""
    entries = [_entry(jobs[s], tier, k, label) for s in _seeds(jobs)]

    def eb_stat(key: str, reducer=_median) -> dict:
        return _stat([reducer(e["info"]["eb"][key]) if "info" in e and "eb" in e["info"] else float("nan") for e in entries], n_boot)

    def check_stat(getter) -> dict:
        return _stat([getter(e["evidence_checks"]) if "evidence_checks" in e else float("nan") for e in entries], n_boot)

    return {
        "median_log10_lambda": eb_stat("log10_lambda"),
        "median_sigma2": eb_stat("sigma2_mean"),
        "median_edf": eb_stat("edf"),
        "fraction_no_write": eb_stat("no_write", _mean),
        "fraction_at_lower_edge": eb_stat("at_lower_edge", _mean),
        "fraction_at_upper_edge": eb_stat("at_upper_edge", _mean),
        "median_log_bf_vs_no_write_per_transition": eb_stat("log_bf_vs_no_write_per_transition"),
        "median_log10_lambda_width_2nats": eb_stat("log10_lambda_width_2nats"),
        "median_innovation_lag1": check_stat(lambda c: _median(c["innovations"]["lag1_autocorr"])),
        "median_n_eff": check_stat(lambda c: _median(c["innovations"]["n_eff"])),
        "heldout_nll": check_stat(lambda c: c["heldout_nll"]),
        "heldout_nll_no_write": check_stat(lambda c: c["heldout_nll_no_write"]),
        "heldout_coverage": {str(level): check_stat(lambda c, lv=str(level): c["heldout_coverage"][lv]) for level in COVERAGE_LEVELS},
        "heldout_coverage_no_write": {str(level): check_stat(lambda c, lv=str(level): c["heldout_coverage_no_write"][lv]) for level in COVERAGE_LEVELS},
    }


def summarize_h1r(jobs: dict, tiers: list[str], gates: dict, n_boot: int) -> dict:
    primary = f"h{PRIMARY_HORIZON}"
    out = {"primary_horizon": PRIMARY_HORIZON, "k": PRIMARY_K, "comparisons": {}}
    for tier in tiers:
        out["comparisons"][tier] = {
            "latent_opt_vs_better_rls": compare_open(jobs, tier, PRIMARY_K, "latent_opt@latent", "rls_best", n_boot),
            "latent_opt_vs_rls_latent": compare_open(jobs, tier, PRIMARY_K, "latent_opt@latent", "rls@latent", n_boot),
            "latent_opt_vs_rls_nolatent": compare_open(jobs, tier, PRIMARY_K, "latent_opt@latent", "rls@nolatent", n_boot),
            "latent_opt_vs_control": compare_open(jobs, tier, PRIMARY_K, "latent_opt@latent", "frozen@control", n_boot),
            "better_rls_vs_control": compare_open(jobs, tier, PRIMARY_K, "rls_best", "frozen@control", n_boot),
            "closed_loop": compare_closed(jobs, tier, "latent_opt@latent", "rls_best", n_boot),
            "z_saturation": _mean([_entry(jobs[s], tier, PRIMARY_K, "latent_opt@latent").get("info", {}).get("z_saturation", float("nan")) for s in _seeds(jobs)]),
        }
    out["better_rls_by_seed"] = {s: jobs[s]["rls_best"] for s in _seeds(jobs)}
    if {"interp", "oof"} <= set(tiers):
        v_in = out["comparisons"]["interp"]["latent_opt_vs_better_rls"][primary]["verdict"]
        v_out = out["comparisons"]["oof"]["latent_opt_vs_better_rls"][primary]["verdict"]
        out["verdicts"] = {"interp": v_in, "oof": v_out}
        out.update(gated(h1r_rule(v_in, v_out), gates))
    return out


def summarize_h12(jobs: dict, tiers: list[str], ks: list[str], gates: dict, n_boot: int) -> dict:
    primary = f"h{PRIMARY_HORIZON}"
    a = "hybrid_eb@latent"
    out = {"primary_horizon": PRIMARY_HORIZON, "k": PRIMARY_K, "comparisons": {}, "selected": {}, "lambda_curve": {}}
    rule_input, sensitivity_input = {}, {}
    seeds = _seeds(jobs)
    for tier in tiers:
        latent_mean = _mean(series(jobs, tier, PRIMARY_K, "latent_opt@latent", primary))
        rls_mean = _mean(series(jobs, tier, PRIMARY_K, "rls_best", primary))
        test_mean_single = "latent_opt@latent" if latent_mean <= rls_mean else "rls_best"
        comparisons = {
            "vs_hybrid": compare_open(jobs, tier, PRIMARY_K, a, "hybrid@latent", n_boot),
            "vs_latent_opt": compare_open(jobs, tier, PRIMARY_K, a, "latent_opt@latent", n_boot),
            "vs_rls": compare_open(jobs, tier, PRIMARY_K, a, "rls_best", n_boot),
            "vs_better_single": compare_open(jobs, tier, PRIMARY_K, a, "better_single", n_boot),
            "vs_better_single_by_test_mean": compare_open(jobs, tier, PRIMARY_K, a, test_mean_single, n_boot),
            "vs_frozen": compare_open(jobs, tier, PRIMARY_K, a, "frozen@latent", n_boot),
            "vs_control": compare_open(jobs, tier, PRIMARY_K, a, "frozen@control", n_boot),
            "rls_vs_frozen": compare_open(jobs, tier, PRIMARY_K, "rls_best", "frozen_best", n_boot),
            "vs_per_output_sigma": compare_open(jobs, tier, PRIMARY_K, a, "hybrid_eb_per_output", n_boot),
            "ebrls_vs_better_rls": compare_open(jobs, tier, PRIMARY_K, "ebrls_best", "rls_best", n_boot),
            "ebrls_vs_frozen": compare_open(jobs, tier, PRIMARY_K, "ebrls_best", "frozen_ebrls_best", n_boot),
            "closed_loop_vs_hybrid": compare_closed(jobs, tier, a, "hybrid@latent", n_boot),
            "closed_loop_vs_better_rls": compare_closed(jobs, tier, a, "rls_best", n_boot),
            "closed_loop_vs_latent_opt": compare_closed(jobs, tier, a, "latent_opt@latent", n_boot),
        }
        out["comparisons"][tier] = comparisons
        out["comparisons"][tier]["better_single_by_seed"] = {s: resolve(jobs[s], "better_single", tier) for s in seeds}
        selected = {
            k: {label: eb_selected(jobs, tier, int(k), label, n_boot) for label in (a, "ebrls@latent", "ebrls@nolatent")} for k in ks
        }
        out["selected"][tier] = selected
        out["lambda_curve"][tier] = {
            s: jobs[s]["tiers"][tier].get("lambda_curve", {}) for s in seeds
        }
        verdict = lambda name: comparisons[name][primary]["verdict"]
        row = {
            "vs_hybrid": verdict("vs_hybrid"),
            "vs_latent_opt": verdict("vs_latent_opt"),
            "vs_rls": verdict("vs_rls"),
            "vs_better_single": verdict("vs_better_single"),
            "vs_frozen": verdict("vs_frozen"),
            "rls_vs_frozen": verdict("rls_vs_frozen"),
            "median_edf": selected[str(PRIMARY_K)][a]["median_edf"]["mean"] if str(PRIMARY_K) in selected else float("nan"),
        }
        rule_input[tier] = row
        sensitivity_input[tier] = {**row, "vs_better_single": verdict("vs_better_single_by_test_mean")}
    rule = h12_rule(rule_input)
    sensitivity = h12_rule(sensitivity_input)
    out["rule"] = rule
    out["rule_input"] = rule_input
    out["outcome_with_test_mean_comparator"] = sensitivity["outcome"]
    out["comparator_sensitive"] = sensitivity["outcome"] != rule["outcome"]
    out.update(gated(rule["outcome"], gates))
    return out


def summarize_h13(jobs: dict, gates: dict, n_boot: int) -> dict:
    seeds = _seeds(jobs)
    first = jobs[seeds[0]]["h13"]["tiers"]
    out = {"decided_on": "prequential one-step RMSE, as H10", "cells": {}, "selected": {}}
    for method in H13_METHODS:
        out["selected"][method] = {s: jobs[s]["h13"]["selection"][method]["hp"] for s in seeds}
    prequential_cells, rollout_cells = {}, {}
    pairs = {
        "vs_plain": "forgetting",
        "vs_reset": "episode_reset",
        "vs_surprise": "surprise_reset",
        "vs_persist": "persist",
        "vs_oracle_reset": "oracle_reset",
        "vs_frozen": "frozen",
    }
    for tier, schedules in first.items():
        out["cells"][tier] = {}
        for name, cell in schedules.items():
            metrics = ["prequential", ROLLOUT_KEY, f"post_switch_{PRIMARY_POST_N}", "steady"]

            def values(method: str, metric: str) -> list:
                return [jobs[s]["h13"]["tiers"][tier][name]["methods"][method].get(metric, {}).get("mean", float("nan")) for s in seeds]

            comparisons = {
                metric: {key: _paired(values("prior_forgetting", metric), values(other, metric), n_boot) for key, other in pairs.items()}
                for metric in metrics
            }
            windup = {}
            for method in H13_METHODS:
                records = [jobs[s]["h13"]["tiers"][tier][name]["methods"][method]["windup"] for s in seeds]
                windup[method] = {
                    "n_failures": sum(int(r["n_failures"]) for r in records),
                    "trace_max": max(float(r["trace_max"]) for r in records),
                    "log10_trace_growth": _mean([r["log10_trace_growth"] for r in records]),
                }
            tainted = windup["prior_forgetting"]["n_failures"] > 0
            out["cells"][tier][name] = {
                "means": {
                    method: {metric: _mean(values(method, metric)) for metric in metrics} for method in H13_METHODS + H13_REFERENCES
                },
                "comparisons": comparisons,
                "windup": windup,
                "tainted": tainted,
                "kind": cell["kind"],
            }
            if cell["kind"] == NO_SWITCH_KIND:
                continue  # the stationarity control decides nothing
            key = f"{tier}/{name}"
            for target, metric in ((prequential_cells, "prequential"), (rollout_cells, ROLLOUT_KEY)):
                target[key] = {
                    "vs_plain": comparisons[metric]["vs_plain"]["verdict"],
                    "vs_reset": comparisons[metric]["vs_reset"]["verdict"],
                    "vs_surprise": comparisons[metric]["vs_surprise"]["verdict"],
                    "tainted": tainted,
                }
    rule = h13_rule(prequential_cells)
    out["rule"] = rule
    out["rule_at_h10"] = h13_rule(rollout_cells)
    out.update(gated(rule["outcome"], gates))
    return out


def summarize_h5(jobs: dict, gates: dict, n_boot: int) -> dict:
    seeds = _seeds(jobs)
    family = jobs[seeds[0]].get("family", "")
    first = jobs[seeds[0]]["h5"]
    primary = f"h{PRIMARY_HORIZON}"
    out = {
        "registered_scope": family in H5_REGISTERED_FAMILIES,
        "primary_horizon": PRIMARY_HORIZON,
        "k": PRIMARY_K,
        "design_lambda": {s: jobs[s]["h5"]["design_lambda"] for s in seeds},
        "selected": {arm: {s: jobs[s]["h5"]["selection"][arm]["hp"] for s in seeds} for arm in H5_ARMS},
        "tiers": {},
    }
    rule_input = {}
    for tier, tier_first in first["tiers"].items():
        out["tiers"][tier] = {"k": {}}

        def arm_metric(arm: str, k: str, metric: str, kind: str = "open_loop") -> list:
            return [jobs[s]["h5"]["tiers"][tier]["k"][k][arm].get(kind, {}).get(metric, float("nan")) for s in seeds]

        for k in tier_first["k"]:
            row = {
                "means": {
                    arm: {
                        **{f"h{h}": _mean(arm_metric(arm, k, f"h{h}")) for h in HORIZONS},
                        "logdet_gain": _mean(arm_metric(arm, k, "logdet_gain", "coverage")),
                        "query_log1p_q": _mean(arm_metric(arm, k, "query_log1p_q", "coverage")),
                        "state_spread": _mean(arm_metric(arm, k, "state_spread", "coverage")),
                        "per_seed_primary": arm_metric(arm, k, primary),
                    }
                    for arm in H5_ARMS
                },
                "dopt_vs_random": {f"h{h}": _paired(arm_metric("dopt", k, f"h{h}"), arm_metric("random_action", k, f"h{h}"), n_boot) for h in HORIZONS},
                "dopt_vs_default": {f"h{h}": _paired(arm_metric("dopt", k, f"h{h}"), arm_metric("default", k, f"h{h}"), n_boot) for h in HORIZONS},
                "dopt_vs_iid_state": {f"h{h}": _paired(arm_metric("dopt", k, f"h{h}"), arm_metric("iid_state", k, f"h{h}"), n_boot) for h in HORIZONS},
                "random_vs_default": {f"h{h}": _paired(arm_metric("random_action", k, f"h{h}"), arm_metric("default", k, f"h{h}"), n_boot) for h in HORIZONS},
            }
            # Higher log det is better, so compare its negative with no effect threshold: a_wins means dopt raised coverage.
            negated = lambda arm: [-x if _is_finite(x) else float("nan") for x in arm_metric(arm, k, "logdet_gain", "coverage")]
            row["coverage_dopt_vs_random"] = _paired(negated("dopt"), negated("random_action"), n_boot, threshold=0.0)
            row["coverage_dopt_vs_default"] = _paired(negated("dopt"), negated("default"), n_boot, threshold=0.0)
            row["coverage_random_vs_default"] = _paired(negated("random_action"), negated("default"), n_boot, threshold=0.0)
            row["query_coverage_dopt_vs_random"] = _paired(arm_metric("dopt", k, "query_log1p_q", "coverage"), arm_metric("random_action", k, "query_log1p_q", "coverage"), n_boot, threshold=0.0)
            random_err, dopt_err, iid_err = (row["means"][arm][primary] for arm in ("random_action", "dopt", "iid_state"))
            room = random_err - iid_err
            row["fraction_of_gap_to_ceiling_closed"] = (random_err - dopt_err) / room if _is_finite(room) and room > 0 else float("nan")
            out["tiers"][tier]["k"][k] = row
        # Closed loop, K = 1, on the same tasks, conditional on the oracle gate.
        gate = oracle_gate_for(jobs, tier)
        metric = "failure_rate_on_oracle_solved"
        key = str(PRIMARY_K)
        out["tiers"][tier]["closed_loop"] = {
            "dopt_vs_random": closed_verdict(gate, arm_metric("dopt", key, metric, "closed_loop"), arm_metric("random_action", key, metric, "closed_loop"), n_boot),
            "dopt_vs_default": closed_verdict(gate, arm_metric("dopt", key, metric, "closed_loop"), arm_metric("default", key, metric, "closed_loop"), n_boot),
        }
        out["tiers"][tier]["same_initial_states"] = all(
            len(set(jobs[s]["h5"]["tiers"][tier]["initial_obs_sha256"].values())) == 1 for s in seeds
        )
        primary_row = out["tiers"][tier]["k"][key]
        rule_input[tier] = {
            "vs_random": primary_row["dopt_vs_random"][primary]["verdict"],
            "vs_default": primary_row["dopt_vs_default"][primary]["verdict"],
            "coverage_vs_random": primary_row["coverage_dopt_vs_random"]["verdict"],
        }
    rule = h5_rule(rule_input)
    out["rule"] = rule
    out["rule_input"] = rule_input
    out.update(gated(rule["outcome"], gates))
    if not out["registered_scope"]:
        out["gated_outcome"] = "outside_registered_scope"
    return out


def summarize(jobs: dict, n_boot: int) -> dict:
    """Verdicts for one family from whichever seeds finished. Accepts null or NaN for undefined values: merge_results passes NaN."""
    if not jobs:
        return {}
    first = jobs[_seeds(jobs)[0]]
    tiers = list(first["tiers"])
    ks = list(first["tiers"][tiers[0]]["k"])
    gates = summarize_gates(jobs, tiers)
    return {
        "n_seeds": len(jobs),
        "gates": gates,
        "means": means_table(jobs, tiers, ks),
        "grid_edges": summarize_edges(jobs),
        "H1r": summarize_h1r(jobs, tiers, gates, n_boot),
        "H12": summarize_h12(jobs, tiers, ks, gates, n_boot),
        "H13": summarize_h13(jobs, gates, n_boot),
        "H5": summarize_h5(jobs, gates, n_boot),
    }


# ---------------------------------------------------------------- output


def fingerprint() -> dict:
    """src/intr, thread_a and thread_b (which supply training and streams), and this file."""
    out = source_fingerprint()
    for path in (ROOT / "experiments" / "thread_b.py", Path(__file__).resolve()):
        out[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def f_design() -> dict:
    """The thread marker for merge_results. Must not depend on the job: every input has to match exactly."""
    return {
        "hypotheses": ["H1r", "H12", "H13", "H5"],
        "primary_horizon": PRIMARY_HORIZON,
        "primary_k": PRIMARY_K,
        "effect_threshold": EFFECT_THRESHOLD,
        "oracle_gate": ORACLE_GATE,
        "latent_gate": LATENT_GATE,
        "weak_write_edf": WEAK_WRITE_EDF,
        "no_write_edf": ebayes.NO_WRITE_EDF,
        "evidence_grid": {
            "low": min(ebayes.DEFAULT_LAMBDAS),
            "high": max(ebayes.DEFAULT_LAMBDAS),
            "n": len(ebayes.DEFAULT_LAMBDAS),
            "no_write_lambda": ebayes.NO_WRITE_LAMBDA,
            "sigma_mode": "shared",
            "sensitivity": "per_output",
        },
        "fixed_choices": FIXED_CHOICES,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny settings, checks the pipeline")
    mode.add_argument("--full", action="store_true", help="protocol settings")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="exact training seeds; default 5 to 9")
    parser.add_argument("--families", nargs="+", default=None, help="default pendulum and drone; puck only when named")
    parser.add_argument("--max-minutes", type=float, default=None, help="start no new job after this long")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_f")
    args = parser.parse_args(argv)

    cfg = smoke_config() if args.smoke else full_config()
    if args.seed_list is not None:
        cfg.seeds = args.seed_list
    if args.families:
        cfg.families = args.families

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"thread_f_{cfg.mode}.json"
    started = time.perf_counter()
    output = {
        "config": asdict(cfg),
        "f_design": f_design(),
        "conditions": [asdict(c) for c in conditions_for(cfg)],
        "registered_methods": sorted(METHODS),
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "source_sha256": fingerprint(),
        },
        "scope_limit": SCOPE_LIMIT,
        "jobs": {f: {} for f in cfg.families},
        "summary": {},
        "stopped_early": False,
    }

    def save() -> None:
        output["metadata"]["elapsed_s"] = time.perf_counter() - started
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(output), indent=1, allow_nan=False), encoding="utf-8")
        tmp.replace(path)

    for seed in cfg.seeds:
        for family in cfg.families:
            elapsed = (time.perf_counter() - started) / 60.0
            if args.max_minutes is not None and elapsed >= args.max_minutes:
                output["stopped_early"] = True
                save()
                print(f"stopped at {elapsed:.1f} minutes before seed {seed}, {family}", flush=True)
                return
            print(f"seed {seed}, {family} (elapsed {elapsed:.1f} min)", flush=True)
            job_started = time.perf_counter()
            # Through _jsonable now, so summarize sees what a reader of the file or merge_results sees.
            output["jobs"][family][str(seed)] = _jsonable(run_job(family, seed, cfg))
            print(f"  job took {(time.perf_counter() - job_started) / 60.0:.2f} minutes", flush=True)
            output["summary"] = {f: summarize(j, cfg.n_boot) for f, j in output["jobs"].items() if j}
            save()
    save()
    print(f"wrote {path} in {(time.perf_counter() - started) / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
