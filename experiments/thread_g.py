"""Thread G: benchmark repair. Scores gate G1 (docs/HYPOTHESES.md, second round) and reports what fixes puck.

    python experiments/thread_g.py --smoke     # a few minutes, checks the pipeline; numbers mean nothing
    python experiments/thread_g.py --full      # under 60 minutes per seed-family job on 4 cores (not measured yet)

Writes results/thread_g/thread_g_<mode>.json, rewritten after every (seed, family) job.

Gate G1 needs two things that have different causes, so a single fix may not pass:
    - closed-loop success: limited by how well the model fits contact where the planner goes.
    - the latent lowering loss by 10%: limited by how much of the remaining error depends on the environment at all.
      At round-one fit, environment-dependent variance is small next to contact error, so this needs the fit fixed first.

Each arm changes training only. An Arm has no field for the task, the simulator, the planner or the oracle, so no arm
can pass by making the task easier. run_job records those settings per job, and tests compare them with round one.

Why these arms (reasoned before any result; see DROPPED for what was left out):
    round_one         the first-round config, as the baseline. Same training code path and same sizes.
    latent_fit        32 samples per environment per latent step, and latent lr decayed with the model's. Adam normalizes
                      gradient scale, so at low signal-to-noise each step moves z by about lr whatever the gradient says.
                      A constant 0.03 over 200 steps random-walks about 0.4 on a [0, 1] range. That fits the 41-44% of z at
                      the clip bound. Samples alone raise signal-to-noise; only the decay lets z settle. Expected to raise
                      R2 and not closed-loop success.
    huber             tests the "spikes dominate" explanation directly. Expected to help the free-flight fit and R2,
                      and possibly to hurt closed-loop: where contact is unresolved, the conditional median is "no contact".
    relative_inputs   the contact boundary is a circle in relative position. Train error equals test error, so this is
                      underfitting, and this is the cheapest test of whether the input representation is why. Puck only.
                      It adds known geometry, so it is outside the first round's plain model class. Reported as such.
    contact_data      doubles the data with slow, repeated pushes from close by, the contact the task needs. The default
                      push is one hard collision. Puck only.
    planner_data      two rounds of data from a planner using a pilot model, the standard fix for a planner exploiting
                      model error where there was no data. Neural planning is cheap, so this costs little.
    capacity_updates  twice the width, one more layer, three times the updates. Joined because the first round's
                      continued-training control gained only 4% with no extra capacity, and more width without more updates
                      rarely fits better.
    combined          everything above except huber, since the gates need two different causes fixed at once.

Risks handled in code:
    An arm that changes the objective: the gate's gain is plain MSE on the base data in fixed units, for every arm.
        Each arm's own objective is reported next to it as own_gain.
    A latent that memorizes: fresh_gain uses unseen transitions from training environments. heldout_gain_<tier> fits z on
        new environments and scores it on other transitions from them, against the no-latent model.
    Choosing on test data: choose_arm reads validation scores and shared training scores only. Interp is reported beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intr.adapt import adapt  # noqa: E402
from intr.envs import (  # noqa: E402
    FAMILIES,
    BatchEnv,
    Trajectories,
    Transitions,
    collect_contact_rich,
    collect_mixed,
    collect_trajectories,
    derive_seed,
    make_env,
    make_generator,
)
from intr.eval import (  # noqa: E402
    START_STEPS,
    CEMConfig,
    OraclePredictor,
    PersistencePredictor,
    closed_loop,
    collect_holdout,
    collect_planned,
    compute_obs_scale,
    open_loop_errors,
    paired_comparison,
)
from intr.looped import replay_contact_flags, window_contact_mask  # noqa: E402
from intr.models import INPUT_EXTRAS, ModelConfig, rollout  # noqa: E402
from intr.train import (  # noqa: E402
    Backbone,
    TrainConfig,
    TransitionData,
    latent_r2,
    regression_loss,
    train_latent,
    train_nolatent,
)
from thread_a import _jsonable  # noqa: E402

G1_SUCCESS = 0.5
G1_LATENT_GAIN = 0.10
ORACLE_GATE = 0.5
MIN_HELDOUT_GAIN = 0.0  # a latent fitted on a new environment must beat no latent on unseen transitions
NO_ARM = "none"
BASELINE_ARM = "round_one"
CONTACT_HORIZON = 10
BACKWARD_FACTOR = 3  # a training sample costs about a forward and two forwards' worth of backward


@dataclass(frozen=True)
class Arm:
    """Training options only. Scales apply to the thread's config, so smoke settings stay tiny."""

    name: str
    width_scale: int = 1
    extra_layers: int = 0
    update_scale: int = 1
    loss: str = "mse"
    latent_samples_per_env: int = 0
    latent_block_steps: int = 0
    latent_lr_decay: bool = False
    input_extra: str = ""  # applies on puck only
    contact_data: bool = False  # applies on puck only
    planner_rounds: int = 0
    puck_only: bool = False  # the arm has nothing to test on other families


ARMS = (
    Arm(BASELINE_ARM),
    Arm("latent_fit", latent_samples_per_env=32, latent_block_steps=50, latent_lr_decay=True),
    Arm("huber", loss="huber"),
    Arm("relative_inputs", input_extra="puck_relative", puck_only=True),
    Arm("contact_data", contact_data=True, puck_only=True),
    Arm("planner_data", planner_rounds=2),
    Arm("capacity_updates", width_scale=2, extra_layers=1, update_scale=3, latent_block_steps=200),
    Arm(
        "combined",
        width_scale=2,
        extra_layers=1,
        update_scale=3,
        latent_samples_per_env=32,
        latent_block_steps=50,
        latent_lr_decay=True,
        input_extra="puck_relative",
        contact_data=True,
        planner_rounds=2,
    ),
)

DROPPED = {
    "finer_dt": (
        "The flow map is already continuous at dt = 0.1: a collision's effect ramps up over a gap of about v * 0.09 m, "
        "0.17 m at 2 m/s. So geometry looks like the harder part. Halving dt doubles planning cost and rollout compounding, "
        "and every adaptation method would need a wrapper. Revisit if relative_inputs and capacity_updates both fail."
    ),
    "per_dimension_normalization": (
        "Targets are already normalized per dimension. A robust scale only reweights spikes against free flight, "
        "which is the question the huber arm asks."
    ),
    "more_updates_alone_and_capacity_alone": "Joined as capacity_updates; see the module docstring.",
}


def effective_arm(arm: Arm, family: str) -> Arm:
    """What the arm means on this family."""
    return arm if family == "puck" else replace(arm, input_extra="", contact_data=False)


@dataclass
class ThreadGConfig:
    mode: str
    seeds: list[int]
    families: list[str] = field(default_factory=lambda: ["puck", "drone"])
    z_dim: int = 8
    hidden: int = 128
    n_hidden: int = 2
    feature_dim: int = 64
    train: TrainConfig = field(default_factory=TrainConfig)
    train_envs: int = 128
    train_traj: int = 8
    train_steps: int = 40
    n_eval_envs: int = 42  # twice round one, since a threshold on success needs more resolution than 1/21
    n_val_envs: int = 40
    adapt_steps: int = 40
    fit_k: int = 3  # trajectories to fit z on a held-out environment
    score_traj: int = 4  # trajectories to score that z on
    holdout_traj: int = 8
    holdout_steps: int = 60
    open_loop_tiers: list[str] = field(default_factory=lambda: ["train", "interp", "extrap", "oof"])
    closed_loop_tiers: list[str] = field(default_factory=lambda: ["val_in", "interp", "extrap", "oof"])
    n_candidates: int = 32
    n_elites: int = 6
    n_iters: int = 3
    horizon_override: int | None = None
    episode_override: int | None = None
    planner_rollouts: int = 4
    planner_action_noise: float = 0.1
    n_boot: int = 10000


def smoke_config() -> ThreadGConfig:
    return ThreadGConfig(
        mode="smoke",
        seeds=[5],
        z_dim=4,
        hidden=32,
        feature_dim=16,
        train=TrainConfig(model_steps=150, warmup_steps=30, block_steps=30, batch_size=64, log_every=50),
        train_envs=12,
        train_traj=3,
        train_steps=15,
        n_eval_envs=6,
        n_val_envs=4,
        adapt_steps=15,
        fit_k=2,
        score_traj=2,
        holdout_traj=2,
        n_candidates=8,
        n_elites=2,
        n_iters=2,
        horizon_override=4,
        episode_override=6,
        planner_rollouts=1,
        n_boot=200,
    )


def full_config() -> ThreadGConfig:
    return ThreadGConfig(mode="full", seeds=[5, 6, 7, 8, 9])  # seeds 0-4 informed the hypotheses


def cem_for(env: BatchEnv, cfg: ThreadGConfig) -> CEMConfig:
    return CEMConfig(
        horizon=cfg.horizon_override or env.plan_horizon,
        n_candidates=cfg.n_candidates,
        n_elites=cfg.n_elites,
        n_iters=cfg.n_iters,
    )


def arm_train_config(arm: Arm, cfg: ThreadGConfig) -> TrainConfig:
    t, s = cfg.train, arm.update_scale
    block = t.block_steps * s
    return replace(
        t,
        model_steps=t.model_steps * s,
        warmup_steps=t.warmup_steps * s,
        block_steps=block,
        loss=arm.loss,
        latent_samples_per_env=arm.latent_samples_per_env,
        latent_block_steps=min(arm.latent_block_steps, block),
        latent_lr_decay=arm.latent_lr_decay,
    )


def arm_model_config(arm: Arm, env: BatchEnv, cfg: ThreadGConfig, z_dim: int) -> ModelConfig:
    return ModelConfig(
        env.obs_dim,
        env.act_dim,
        z_dim,
        cfg.hidden * arm.width_scale,
        cfg.n_hidden + arm.extra_layers,
        cfg.feature_dim * arm.width_scale,
        arm.input_extra,
    )


def forward_macs(model: ModelConfig) -> int:
    """Multiply-accumulates per sample through the MLP and the readout."""
    extra = INPUT_EXTRAS[model.input_extra][0] if model.input_extra else 0
    d = model.obs_dim + model.act_dim + model.z_dim + extra
    total = 0
    for _ in range(model.n_hidden):
        total += d * model.hidden
        d = model.hidden
    return total + d * model.feature_dim + model.feature_dim * model.obs_dim


def arm_cost(arm: Arm, env: BatchEnv, cfg: ThreadGConfig) -> dict:
    """Planned compute, the same on every machine, so the choice cannot depend on runner speed. Evaluation is not included."""
    t = arm_train_config(arm, cfg)
    with_z = forward_macs(arm_model_config(arm, env, cfg, cfg.z_dim))
    no_z = forward_macs(arm_model_config(arm, env, cfg, 0))
    warmup = min(t.warmup_steps, t.model_steps)
    blocks = 1 + math.ceil(max(t.model_steps - warmup, 0) / t.block_steps)
    latent_updates = blocks * (t.latent_block_steps or t.block_steps)
    latent_batch = cfg.train_envs * t.latent_samples_per_env if t.latent_samples_per_env > 0 else t.batch_size
    macs = BACKWARD_FACTOR * (t.model_steps * t.batch_size * (with_z + no_z) + latent_updates * latent_batch * with_z)
    transitions = cfg.train_envs * cfg.train_traj * cfg.train_steps
    if arm.contact_data:
        transitions += cfg.train_envs * cfg.train_traj * cfg.train_steps
    if arm.planner_rounds > 0:
        pilot = forward_macs(ModelConfig(env.obs_dim, env.act_dim, 0, cfg.hidden, cfg.n_hidden, cfg.feature_dim, arm.input_extra))
        cem = cem_for(env, cfg)
        episode = cfg.episode_override or env.episode_steps
        pilot_train = BACKWARD_FACTOR * cfg.train.model_steps * cfg.train.batch_size * pilot
        plan = cfg.planner_rollouts * episode * cfg.train_envs * cem.n_candidates * cem.n_iters * cem.horizon * pilot
        macs += arm.planner_rounds * (pilot_train + plan)
        transitions += arm.planner_rounds * cfg.planner_rollouts * episode * cfg.train_envs
    return {"macs": float(macs), "transitions": int(transitions)}


# ---------------------------------------------------------------- data


def frozen_predictor(backbone: Backbone, n_envs: int):
    """No adaptation. frozen only reads the number of environments from its data."""
    empty = torch.zeros(n_envs, 1, backbone.model.cfg.obs_dim)
    return adapt("frozen", backbone, Transitions(empty, empty, empty)).predictor


def arm_data(arm: Arm, family: str, seed: int, cfg: ThreadGConfig, env: BatchEnv, base: TransitionData) -> TransitionData:
    """The base data, unchanged and first, plus what the arm adds. All from the training environments."""
    data = base
    if arm.contact_data:
        rich = collect_contact_rich(env, cfg.train_traj, cfg.train_steps, make_generator("data_contact", family, seed))
        data = data.extended(rich.flat())
    if arm.planner_rounds > 0:
        pilot_model = ModelConfig(env.obs_dim, env.act_dim, 0, cfg.hidden, cfg.n_hidden, cfg.feature_dim, arm.input_extra)
        for round_index in range(arm.planner_rounds):
            pilot = train_nolatent(data, pilot_model, cfg.train, ("pilot", seed, round_index))
            predictor = frozen_predictor(pilot, env.n_envs)
            for rollout_index in range(cfg.planner_rollouts):
                tag = ("planner", family, seed, arm.name, round_index, rollout_index)
                task = env.sample_task(make_generator(tag, "task"))
                trajs = collect_planned(
                    env, predictor, task, cem_for(env, cfg), make_generator(tag, "cem"), cfg.episode_override, cfg.planner_action_noise
                )
                data = data.extended(trajs.flat())
    return data


def contact_fraction(env: BatchEnv, family: str, obs: torch.Tensor, act: torch.Tensor, next_obs: torch.Tensor) -> float:
    """Share of transitions with agent-puck contact. NaN on families without contact."""
    if family != "puck":
        return float("nan")
    flags, _ = replay_contact_flags(env, Trajectories(torch.stack([obs, next_obs], dim=2), act[:, :, None]))
    return flags.float().mean().item()


@dataclass
class TierData:
    """Everything fixed per tier, so all arms see the same environments, tasks and holdouts."""

    env: BatchEnv
    tag: tuple
    holdout: Trajectories
    fit: Transitions
    score: Transitions
    task: object
    contact_windows: torch.Tensor | None  # (E, B S)


def prepare_tier(family: str, seed: int, tier: str, cfg: ThreadGConfig) -> TierData:
    """Validation and test environments come from different labels, so they cannot coincide."""
    if tier.startswith("val"):
        env = make_env(family, cfg.n_val_envs, tier, derive_seed("val_envs", seed))
    else:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
    tag = ("g", family, seed, tier)
    holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
    fit = collect_trajectories(env, cfg.fit_k, cfg.adapt_steps, env.default_probe, make_generator(tag, "fit")).flat()
    score = collect_mixed(env, cfg.score_traj, cfg.train_steps, env.data_policies, make_generator(tag, "score")).flat()
    task = env.sample_task(make_generator(tag, "task"))
    contact = None
    if family == "puck":
        flags, _ = replay_contact_flags(env, holdout)
        contact = window_contact_mask(flags, START_STEPS, CONTACT_HORIZON)
    return TierData(env, tag, holdout, fit, score, task, contact)


# ---------------------------------------------------------------- scores


@torch.no_grad()
def plain_mse(model, z: torch.Tensor | None, data: Transitions, delta_std: torch.Tensor) -> float:
    """MSE in units of the base data's delta std, whatever loss or normalization the arm trained with. z is (E, Z) or None."""
    E, N, _ = data.obs.shape
    z_all = None if z is None else z[:, None, :].expand(E, N, -1)
    err = (model.step(data.obs, data.act, z_all) - data.next_obs) / delta_std
    return (err**2).mean().item()


@torch.no_grad()
def own_loss(backbone: Backbone, z: torch.Tensor | None, data: TransitionData, tcfg: TrainConfig) -> float:
    """The arm's training objective over its whole training data, with final parameters."""
    E, N, _ = data.obs.shape
    z_all = None if z is None else z[:, None, :].expand(E, N, -1)
    model = backbone.model
    return regression_loss(model(data.obs, data.act, z_all), model.target(data.obs, data.next_obs), tcfg).item()


def gain(with_latent: float, without: float) -> float:
    return 1.0 - with_latent / max(without, 1e-12)


@torch.no_grad()
def contact_free_rmse(predictor, td: TierData, scale, low, high, horizon: int = CONTACT_HORIZON) -> tuple[float, float]:
    """RMSE at the horizon over rollout windows with contact and without. Same windows and units as open_loop_errors."""
    if td.contact_windows is None:
        return float("nan"), float("nan")
    holdout = td.holdout
    E, B, _, A = holdout.act.shape
    O = holdout.obs.shape[-1]
    S = len(START_STEPS)
    true = torch.stack([holdout.obs[:, :, s : s + horizon + 1] for s in START_STEPS], dim=2).reshape(E, B * S, horizon + 1, O)
    act = torch.stack([holdout.act[:, :, s : s + horizon] for s in START_STEPS], dim=2).reshape(E, B * S, horizon, A)
    pred = rollout(predictor.step, true[:, :, 0], act, low, high)
    squared = (((pred[:, :, horizon] - true[:, :, horizon]) / scale) ** 2).mean(dim=-1)  # (E, B S)
    mask = td.contact_windows
    contact = squared[mask].mean().sqrt().item() if mask.any() else float("nan")
    free = squared[~mask].mean().sqrt().item() if (~mask).any() else float("nan")
    return contact, free


def run_closed(predictor, td: TierData, cfg: ThreadGConfig) -> dict:
    """Same task and planning noise for every predictor on a tier."""
    out = closed_loop(td.env, predictor, td.task, cem_for(td.env, cfg), make_generator(td.tag, "cem"), cfg.episode_override)
    return {
        "success_rate": out["success"].float().mean().item(),
        "mean_cost": out["mean_cost"].mean().item(),
        "wall_s": out["wall_s"],
    }


def evaluate_arm(arm: Arm, family: str, seed: int, cfg: ThreadGConfig, env, base, fresh, tiers, scale, low, high, delta_std) -> dict:
    started = time.perf_counter()
    data = arm_data(arm, family, seed, cfg, env, base)
    tcfg = arm_train_config(arm, cfg)
    data_wall = time.perf_counter() - started

    started = time.perf_counter()
    nolatent = train_nolatent(data, arm_model_config(arm, env, cfg, 0), tcfg, seed)
    latent = train_latent(data, arm_model_config(arm, env, cfg, cfg.z_dim), tcfg, seed)
    train_wall = time.perf_counter() - started

    started = time.perf_counter()
    z = latent.z_table
    base_trans = Transitions(base.obs, base.act, base.next_obs)
    scores = {
        "train_mse_latent": plain_mse(latent.model, z, base_trans, delta_std),
        "train_mse_nolatent": plain_mse(nolatent.model, None, base_trans, delta_std),
        "own_loss_latent": own_loss(latent, z, data, tcfg),
        "own_loss_nolatent": own_loss(nolatent, None, data, tcfg),
        "fresh_mse_latent": plain_mse(latent.model, z, fresh, delta_std),
        "fresh_mse_nolatent": plain_mse(nolatent.model, None, fresh, delta_std),
        "z_saturation": ((z <= 1e-4) | (z >= 1.0 - 1e-4)).float().mean().item(),
    }
    scores["latent_gain"] = gain(scores["train_mse_latent"], scores["train_mse_nolatent"])
    scores["own_gain"] = gain(scores["own_loss_latent"], scores["own_loss_nolatent"])
    scores["fresh_gain"] = gain(scores["fresh_mse_latent"], scores["fresh_mse_nolatent"])
    for name, value in zip(env.hidden_names, latent_r2(z, env.hidden_matrix())):
        scores[f"r2_{name}"] = value

    # Is the latent usable where it was not trained? z fitted on some transitions, scored on others.
    adapted = {}
    for tier in ("val_in", "interp"):
        td = tiers[tier]
        adapted[tier] = adapt("latent_opt", latent, td.fit)
        fitted = plain_mse(latent.model, adapted[tier].predictor.z, td.score, delta_std)
        without = plain_mse(nolatent.model, None, td.score, delta_std)
        scores[f"heldout_mse_fitted_{tier}"] = fitted
        scores[f"heldout_mse_nolatent_{tier}"] = without
        scores[f"heldout_gain_{tier}"] = gain(fitted, without)
        scores[f"heldout_z_saturation_{tier}"] = adapted[tier].info["z_saturation"]

    for tier in cfg.closed_loop_tiers:
        td = tiers[tier]
        for label, backbone in (("latent", latent), ("nolatent", nolatent)):
            out = run_closed(frozen_predictor(backbone, td.env.n_envs), td, cfg)
            scores[f"success_{tier}_{label}"] = out["success_rate"]
            scores[f"cost_{tier}_{label}"] = out["mean_cost"]
    # Diagnostic only: what fitting z buys. G1 is defined with no adaptation.
    out = run_closed(adapted["interp"].predictor, tiers["interp"], cfg)
    scores["success_interp_fitted"] = out["success_rate"]

    for tier in cfg.open_loop_tiers:
        td = tiers[tier]
        for label, backbone in (("latent", latent), ("nolatent", nolatent)):
            errors = open_loop_errors(frozen_predictor(backbone, td.env.n_envs), td.holdout, scale, low, high)
            for h, e in errors.items():
                scores[f"h{h}_{tier}_{label}"] = e.mean().item()
    for label, backbone in (("latent", latent), ("nolatent", nolatent)):
        td = tiers["interp"]
        contact, free = contact_free_rmse(frozen_predictor(backbone, td.env.n_envs), td, scale, low, high)
        scores[f"contact_windows_rmse_{label}"] = contact
        scores[f"free_windows_rmse_{label}"] = free
    eval_wall = time.perf_counter() - started

    return {
        "arm": asdict(arm),
        "cost": arm_cost(arm, env, cfg),
        "data": {
            "transitions_per_env": data.obs.shape[1],
            "contact_fraction": contact_fraction(env, family, data.obs, data.act, data.next_obs),
        },
        "train": {
            "latent": {"stats": latent.stats, "final_log": latent.log[-3:]},
            "nolatent": {"stats": nolatent.stats, "final_log": nolatent.log[-3:]},
        },
        "scores": scores,
        "gates": gates_for(scores),
        "wall_s": {"data": data_wall, "train": train_wall, "eval": eval_wall},
    }


def gates_for(scores: dict) -> dict:
    """Comparisons with NaN are False, so a missing score fails its gate.

    g1_val is what choose_arm reads: closed-loop success on val_in, and the latent gain on training data, with a held-out
    check on validation environments. g1 is the gate as registered, on interp. It is reported and never used to choose.
    """
    return {
        "g1_val": bool(
            scores["success_val_in_latent"] >= G1_SUCCESS
            and scores["latent_gain"] >= G1_LATENT_GAIN
            and scores["heldout_gain_val_in"] > MIN_HELDOUT_GAIN
        ),
        "g1": bool(scores["success_interp_latent"] >= G1_SUCCESS and scores["latent_gain"] >= G1_LATENT_GAIN),
        "latent_generalizes": bool(scores["fresh_gain"] > MIN_HELDOUT_GAIN and scores["heldout_gain_interp"] > MIN_HELDOUT_GAIN),
    }


def choose_arm(candidates: list[tuple[str, dict, bool]]) -> str:
    """The rule: the cheapest arm passing G1 on validation. Cost is planned MACs, then transitions, then list order.

    candidates: (name, cost, passed validation), in arm order. Takes no test score by construction.
    """
    passing = [(cost["macs"], cost["transitions"], i, name) for i, (name, cost, passed) in enumerate(candidates) if passed]
    return min(passing)[3] if passing else NO_ARM


def task_fingerprint(env: BatchEnv, cfg: ThreadGConfig) -> dict:
    """What no arm may change, recorded per job, so a later reader can check it against round one."""
    cem = cem_for(env, cfg)
    constants = ("episode_steps", "plan_horizon", "n_substeps", "dt", "success_dist", "success_window", "f_max", "contact_k", "contact_c")
    return {
        "env": {name: getattr(env, name) for name in constants if hasattr(env, name)},
        "cem": asdict(cem),
    }


def run_job(family: str, seed: int, cfg: ThreadGConfig, arms=ARMS) -> dict:
    """One training seed on one family. Paired across arms: same base data, seeds, environments and tasks."""
    started = time.perf_counter()
    # The same draws as thread_a.train_all, so the base data is the first round's recipe on these seeds.
    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    base = TransitionData.collect(env, cfg.train_traj, cfg.train_steps, make_generator("data", family, seed))
    fresh = collect_mixed(env, cfg.train_traj, cfg.train_steps, env.data_policies, make_generator("fresh", family, seed)).flat()
    delta_std = (base.next_obs - base.obs).reshape(-1, env.obs_dim).std(0).clamp_min(1e-6)
    scale = compute_obs_scale(family)
    low, high = env.obs_low, env.obs_high
    names = sorted(set(cfg.open_loop_tiers) | set(cfg.closed_loop_tiers) | {"val_in", "interp"})
    tiers = {tier: prepare_tier(family, seed, tier, cfg) for tier in names}

    oracle = {}
    for tier in cfg.closed_loop_tiers:
        oracle[tier] = run_closed(OraclePredictor(tiers[tier].env), tiers[tier], cfg)
    persistence_closed = run_closed(PersistencePredictor(), tiers["interp"], cfg)  # the floor: a planner that believes nothing moves
    persistence_open = {}
    for tier in cfg.open_loop_tiers:
        errors = open_loop_errors(PersistencePredictor(), tiers[tier].holdout, scale, low, high)
        persistence_open[tier] = {f"h{h}": e.mean().item() for h, e in errors.items()}
    setup_wall = time.perf_counter() - started
    print(f"  oracle success {[(t, round(r['success_rate'], 2)) for t, r in oracle.items()]}; setup {setup_wall:.0f}s", flush=True)

    records = {}
    for arm in arms:
        if arm.puck_only and family != "puck":
            records[arm.name] = {"skipped": "tests contact, and this family has none"}
            continue
        active = effective_arm(arm, family)
        arm_started = time.perf_counter()
        records[arm.name] = evaluate_arm(active, family, seed, cfg, env, base, fresh, tiers, scale, low, high, delta_std)
        scores = records[arm.name]["scores"]
        print(
            f"  {arm.name}: val {scores['success_val_in_latent']:.2f}, interp {scores['success_interp_latent']:.2f}, "
            f"gain {scores['latent_gain']:.3f}, heldout {scores['heldout_gain_interp']:.3f}, "
            f"{records[arm.name]['gates']}, {(time.perf_counter() - arm_started) / 60.0:.1f} min",
            flush=True,
        )

    chosen = choose_arm([(n, r["cost"], r["gates"]["g1_val"]) for n, r in records.items() if "skipped" not in r])
    return {
        "family": family,
        "seed": seed,
        "task": task_fingerprint(env, cfg),
        "base_contact_fraction": contact_fraction(env, family, base.obs, base.act, base.next_obs),
        "oracle": oracle,
        "persistence_closed_loop_interp": persistence_closed,
        "persistence_open_loop": persistence_open,
        "arms": records,
        "chosen_on_this_seed": chosen,
        "wall_s": {"total": time.perf_counter() - started, "setup": setup_wall},
    }


# ---------------------------------------------------------------- summary


def _mean(values) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def summarize(jobs: dict, n_boot: int) -> dict:
    """One family, from whichever seeds have finished.

    An arm passes across seeds when its mean scores pass and it passes on a strict majority of seeds. Means alone would let
    one lucky seed decide, and requiring every seed is too strict for binomial noise of about 0.08 at 40 environments.
    Results files store NaN as null and the merge gives them back as NaN. So this reads numbers and strings, never None.
    """
    seeds = sorted(jobs, key=int)
    if not seeds:
        return {}
    first = jobs[seeds[0]]
    family = first["family"]
    active = [name for name, record in first["arms"].items() if "skipped" not in record]
    table = {}
    for name in active:
        records = [jobs[s]["arms"][name] for s in seeds]
        per_seed = {key: [r["scores"][key] for r in records] for key in records[0]["scores"]}
        means = {key: _mean(values) for key, values in per_seed.items()}
        val_seeds = [r["gates"]["g1_val"] for r in records]
        test_seeds = [r["gates"]["g1"] for r in records]
        table[name] = {
            "cost": records[0]["cost"],
            "means": means,
            "per_seed": per_seed,
            "g1_val_per_seed": val_seeds,
            "g1_per_seed": test_seeds,
            "latent_generalizes_per_seed": [r["gates"]["latent_generalizes"] for r in records],
            "g1_val_pass": bool(gates_for(means)["g1_val"] and 2 * sum(val_seeds) > len(seeds)),
            "g1_pass": bool(gates_for(means)["g1"] and 2 * sum(test_seeds) > len(seeds)),
        }
    if BASELINE_ARM in table:
        reference = table[BASELINE_ARM]["per_seed"]
        for name in active:
            row = table[name]["per_seed"]
            table[name]["vs_round_one"] = {
                "failure_rate_interp": paired_comparison(
                    [1.0 - v for v in row["success_interp_latent"]], [1.0 - v for v in reference["success_interp_latent"]], n_boot=n_boot
                ),
                "h10_interp": paired_comparison(row["h10_interp_latent"], reference["h10_interp_latent"], n_boot=n_boot),
            }

    chosen = choose_arm([(name, table[name]["cost"], table[name]["g1_val_pass"]) for name in active])
    oracle_interp = [jobs[s]["oracle"]["interp"]["success_rate"] for s in seeds]
    out = {
        "family": family,
        "seeds": [int(s) for s in seeds],
        "oracle": {tier: _mean(jobs[s]["oracle"][tier]["success_rate"] for s in seeds) for tier in first["oracle"]},
        "oracle_interp_per_seed": oracle_interp,
        "persistence_closed_loop_interp": _mean(jobs[s]["persistence_closed_loop_interp"]["success_rate"] for s in seeds),
        "persistence_open_loop": {
            tier: {key: _mean(jobs[s]["persistence_open_loop"][tier][key] for s in seeds) for key in first["persistence_open_loop"][tier]}
            for tier in first["persistence_open_loop"]
        },
        "base_contact_fraction": _mean(jobs[s]["base_contact_fraction"] for s in seeds),
        "arms": table,
        "chosen_arm": chosen,
        "no_arm_passes": chosen == NO_ARM,
    }
    interp_passers = [name for name in active if table[name]["g1_pass"]]
    out["arms_passing_g1_on_interp"] = interp_passers
    if chosen == NO_ARM:
        out["chosen_arm_g1_on_interp"] = False
        statement = f"NO ARM PASSES G1 on validation for {family} over seeds {out['seeds']}."
        if interp_passers:
            statement += f" Some pass on interp ({interp_passers}), but the rule does not read test tiers, so none is chosen."
    else:
        out["chosen_arm_g1_on_interp"] = table[chosen]["g1_pass"]
        statement = (
            f"Chosen for {family} by the fixed rule (cheapest passing G1 on validation): {chosen}. "
            f"On interp, G1 {'passes' if table[chosen]['g1_pass'] else 'FAILS'}: "
            f"success {table[chosen]['means']['success_interp_latent']:.2f}, latent gain {table[chosen]['means']['latent_gain']:.3f}."
        )
    out["statement"] = statement
    if family != "puck":
        # Second working family: the oracle passes on interp and the latent lowers training loss, on the arm used.
        arm_name = chosen if chosen != NO_ARM else BASELINE_ARM
        gain_mean = table[arm_name]["means"]["latent_gain"] if arm_name in table else float("nan")
        out["second_family_gate"] = {
            "arm": arm_name,
            "oracle_interp": _mean(oracle_interp),
            "latent_gain": gain_mean,
            "passes": bool(_mean(oracle_interp) >= ORACLE_GATE and gain_mean >= G1_LATENT_GAIN),
        }
    return out


G_GATES = {
    "G1": {
        "text": "Closed-loop success on interp with no adaptation at least 0.5, and the latent lowers training loss by at least 10%.",
        "success": G1_SUCCESS,
        "latent_gain": G1_LATENT_GAIN,
        "model": "frozen@latent: the latent backbone at its mean training latent",
        "loss": "plain MSE on the base training data, in units of its delta std, for every arm",
    },
    "second_family": {"oracle_interp": ORACLE_GATE, "latent_gain": G1_LATENT_GAIN},
    "selection_rule": (
        "Cheapest arm by planned training MACs, then transitions, then list order, among arms passing on validation: "
        "success on val_in at least 0.5, latent gain at least 10%, and a latent fitted on new val_in environments beating "
        "no latent on unseen transitions. Across seeds: mean scores pass and a strict majority of seeds pass."
    ),
    "verdicts_at": "summary.<family>: chosen_arm, no_arm_passes, statement, arms.<arm>.g1_pass; per seed under jobs.<family>.<seed>.arms.<arm>.gates",
    "dropped_arms": DROPPED,
    "invariant": "Arms change training only. Simulator, task, success criterion, planner and oracle are shared; see jobs.<family>.<seed>.task.",
}


def source_fingerprint() -> dict:
    files = sorted((ROOT / "src" / "intr").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny settings, checks the pipeline")
    mode.add_argument("--full", action="store_true", help="protocol settings")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="exact training seeds to run")
    parser.add_argument("--families", nargs="+", default=None, choices=list(FAMILIES))
    parser.add_argument("--max-minutes", type=float, default=None, help="start no new job after this long")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_g")
    args = parser.parse_args()

    cfg = smoke_config() if args.smoke else full_config()
    if args.seed_list is not None:
        cfg.seeds = args.seed_list
    if args.families:
        cfg.families = args.families

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"thread_g_{cfg.mode}.json"
    started = time.perf_counter()
    output = {
        "config": asdict(cfg),
        "g_gates": G_GATES,
        "arms": [asdict(a) for a in ARMS],
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "argv": sys.argv[1:],
            "source_sha256": source_fingerprint(),
        },
        "scope_limit": "Small state-based simulators on CPU; no claim about pixels, large models, or real robots.",
        "jobs": {f: {} for f in cfg.families},
        "summary": {},
        "stopped_early": False,
    }

    def save() -> None:
        output["metadata"]["elapsed_s"] = time.perf_counter() - started
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(output), indent=1))
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
            job = run_job(family, seed, cfg)
            output["jobs"][family][str(seed)] = job
            output["summary"] = {f: summarize(j, cfg.n_boot) for f, j in output["jobs"].items() if j}
            save()
            print(f"  seed {seed} alone chooses: {job['chosen_on_this_seed']}", flush=True)
            print(f"  {output['summary'][family]['statement']}", flush=True)
    save()
    print(f"wrote {path} in {(time.perf_counter() - started) / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
