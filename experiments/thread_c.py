"""Thread C: looped dynamics (H7, H8, H9 in docs/HYPOTHESES.md).

    python experiments/thread_c.py --smoke     # under 3 minutes, checks the pipeline; numbers mean nothing
    python experiments/thread_c.py --full      # estimated 1 to 2 hours on 2 CPU cores, not yet measured

Writes results/thread_c/thread_c_<mode>.json after hyperparameter selection and after every (seed, family)
job, so a stopped run keeps what it finished. --max-minutes stops starting new work after that long.

Design of a run:
    1. Per family, once: choose the trained loop count K on the full recipe, then a learning rate per arm, on
       validation environments only (val_in and val_shift), with short training runs on a separate seed.
    2. Per seed and family, train every arm on the same data from the same initial weights, then measure.

Arms are in ARMS and BASELINES. Baselines:
    mlp_params          about the parameters of a looped model
    mlp_params_matched  mlp_params continued until its total forward cost in training matches the full recipe,
                        the protocol's matched continued-training control
    mlp_compute         about the forward cost per step of a looped model at K

Choices HYPOTHESES.md leaves open, fixed here before any result exists:
    - H7's "long-rollout error" is horizon 100 at trained loops, on the interp tier.
    - H7's "stable" is less than 2x worse than the full recipe with under 5% blow-ups (looped.h7_outcome).
    - H7 is also evaluated at the "product" cell (product_loops, horizon 100), reported as a separate outcome,
      because Solv3R's ablations mostly separate above trained depth.
    - H8 is decided on the puck at horizon 10 on interp. Pendulum is the non-contact comparison. Contact and
      contact-free windows are reported separately, descriptively.
    - H9 uses the final-loop residual of the full recipe, teacher-forced with carried z. The primary alignment
      gives the residual and the baseline the same information, observations up to obs[t]. "No detectable gain"
      applies paired_comparison to 1 - AUROC. Shift onset is the mean over the extrap and oof tiers.

Limits worth stating with any result:
    - H7's 2x test can saturate. Rollout error is bounded by the state range, so once the full recipe sits above
      half of the saturation reference, only blow-up can show 2x. two_x_reachable records this per tier.
    - H9 tests detection only, not the "usable to allocate loops or gate writes" clause of the hypothesis.
    - The ARCADE baseline is simplified: fixed Gaussian and CUSUM, no Bayesian decoder or beam.
    - Closed loop runs a subset of arms on one tier, and state does not carry across real steps
      (looped.PlanningPredictor). Learning rates are selected per arm, but K comes from the full recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intr.envs import BatchEnv, Trajectories, collect_mixed, derive_seed, make_env, make_generator  # noqa: E402
from intr.eval import (  # noqa: E402
    CEMConfig,
    OraclePredictor,
    PersistencePredictor,
    closed_loop,
    collect_holdout,
    compute_obs_scale,
    paired_comparison,
)
from intr.looped import (  # noqa: E402
    BLOWUP_LIMIT,
    TWO_X_THRESHOLD,
    GaussianResidualModel,
    LoopConfig,
    LoopedDynamics,
    LoopedPredictor,
    LoopTrainConfig,
    MemorylessPredictor,
    PlanningPredictor,
    SequenceData,
    auroc_groups,
    check_ablations,
    collect_contact_trajectories,
    collect_switching,
    contact_masks,
    count_params,
    h7_outcome,
    h8_outcome,
    h9_target_passes,
    make_windows,
    mlp_macs,
    mlp_matching_macs,
    mlp_matching_params,
    mlp_param_count,
    replay_contact_flags,
    rollout_grid,
    rollout_metrics,
    saturation_reference,
    sensitivity,
    shift_auroc,
    teacher_forced_trace,
    to_signal_time,
    train_looped,
    window_contact_mask,
)
from intr.models import ModelConfig  # noqa: E402
from intr.train import TrainConfig, TransitionData, continued_training, train_nolatent  # noqa: E402

ARMS = {
    "full": {},
    "no_reinject": {"reinject_input": False, "action_every_loop": False},
    "no_norm": {"normalize": False},
    "no_mixed": {"mixed_depth": False},
    "plain": {"reinject_input": False, "action_every_loop": False, "normalize": False, "mixed_depth": False},
    "prelude_only": {"action_every_loop": False},
    "reinit": {"carry": False},
}
H7_ABLATIONS = ("no_reinject", "no_norm", "no_mixed", "plain")
BASELINES = ("mlp_params", "mlp_params_matched", "mlp_compute")
SELECTED_BASELINES = ("mlp_params", "mlp_compute")  # mlp_params_matched continues mlp_params at its learning rate
CLOSED_LOOP_ARMS = ("full", "plain", "prelude_only", "mlp_params", "mlp_compute")
DECISION_TIER = "interp"


@dataclass
class ThreadCConfig:
    mode: str
    seeds: list[int]
    families: list[str] = field(default_factory=lambda: ["pendulum", "puck"])
    z_dim: int = 48
    emb_dim: int = 24
    hidden: int = 96
    mlp_feature_dim: int = 64
    loop_choices: list[int] = field(default_factory=lambda: [4, 8])
    lr_choices: list[float] = field(default_factory=lambda: [1e-3, 3e-4])
    selection_steps: int = 600
    train: LoopTrainConfig = field(default_factory=LoopTrainConfig)
    train_envs: int = 128  # the same training data sizes as Thread A
    train_traj: int = 8
    train_steps: int = 40
    n_eval_envs: int = 21
    n_val_envs: int = 20
    holdout_traj: int = 8
    starts: list[int] = field(default_factory=lambda: [0, 10, 20, 30])  # eval.START_STEPS
    primary_horizon: int = 10
    long_horizon: int = 100
    at_trained_horizons: list[int] = field(default_factory=lambda: [1, 10, 30, 100])
    at_trained_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    loop_grid: list[int] = field(default_factory=lambda: [1, 4, 16, 64, 256])
    horizon_grid: list[int] = field(default_factory=lambda: [1, 10, 30, 100, 300])
    grid_tiers: list[str] = field(default_factory=lambda: ["interp"])
    grid_traj: int = 4  # with start 0 only, so one rollout per trajectory: 84 rollouts over 21 environments
    product_loops: int = 64
    closed_loop_tiers: list[str] = field(default_factory=lambda: ["interp"])
    n_candidates: int = 32
    n_elites: int = 6
    n_iters: int = 3
    horizon_override: int | None = None
    episode_override: int | None = None
    h9_traj: int = 8
    h9_steps: int = 60
    h9_switch: int = 30
    h9_window: int = 5  # signals counted as "onset" after the switch
    h9_burn: int = 5  # the first signals follow z = 0 and are excluded
    h9_tol: int = 1
    h9_kappa: float = 2.0
    h9_shift_tiers: list[str] = field(default_factory=lambda: ["extrap", "oof"])
    n_boot: int = 10000


def smoke_config() -> ThreadCConfig:
    return ThreadCConfig(
        mode="smoke",
        seeds=[0, 1],
        z_dim=12,
        emb_dim=6,
        hidden=16,
        mlp_feature_dim=8,
        loop_choices=[2],
        lr_choices=[1e-3, 3e-4],
        selection_steps=15,
        train=LoopTrainConfig(steps=40, batch_windows=8, window=2, log_every=20),
        train_envs=8,
        train_traj=2,
        train_steps=10,
        n_eval_envs=6,
        n_val_envs=4,
        holdout_traj=2,
        starts=[0, 2],
        long_horizon=12,
        at_trained_horizons=[1, 10, 12],
        at_trained_tiers=["interp", "oof"],
        loop_grid=[1, 4],
        horizon_grid=[1, 10, 12],
        grid_traj=2,
        product_loops=4,
        n_candidates=6,
        n_elites=2,
        n_iters=1,
        horizon_override=3,
        episode_override=4,
        h9_traj=2,
        h9_steps=24,
        h9_switch=12,
        h9_window=3,
        h9_burn=2,
        h9_shift_tiers=["oof"],
        n_boot=200,
    )


def full_config() -> ThreadCConfig:
    return ThreadCConfig(mode="full", seeds=[0, 1, 2, 3, 4])


# ---------------------------------------------------------------- building and training


def arm_config(env: BatchEnv, arm: str, loops: int, cfg: ThreadCConfig) -> LoopConfig:
    return LoopConfig(
        env.obs_dim,
        env.act_dim,
        z_dim=cfg.z_dim,
        emb_dim=cfg.emb_dim,
        hidden=cfg.hidden,
        loops=loops,
        grad_loops=max(1, loops // 2),
        **ARMS[arm],
    )


def training_data(family: str, seed, cfg: ThreadCConfig) -> tuple[BatchEnv, SequenceData, TransitionData]:
    """One data set per seed, as both windows and transitions, so baselines see the same trajectories."""
    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    trajs = collect_mixed(env, cfg.train_traj, cfg.train_steps, env.data_policies, make_generator("data", family, seed))
    flat = trajs.flat()
    transitions = TransitionData(flat.obs, flat.act, flat.next_obs, cfg.train_traj, cfg.train_steps)
    return env, SequenceData.from_trajectories(trajs), transitions


def baseline_config(env: BatchEnv, name: str, loops: int, cfg: ThreadCConfig) -> ModelConfig:
    reference = LoopedDynamics(arm_config(env, "full", loops, cfg))
    if name == "mlp_params":
        return mlp_matching_params(count_params(reference), env.obs_dim, env.act_dim, cfg.mlp_feature_dim)
    if name == "mlp_compute":
        return mlp_matching_macs(reference.macs_per_step(), env.obs_dim, env.act_dim, cfg.mlp_feature_dim)
    raise ValueError(name)


def mlp_train_config(cfg: ThreadCConfig, lr: float, steps: int) -> TrainConfig:
    return TrainConfig(
        model_steps=steps,
        batch_size=cfg.train.batch_windows * cfg.train.window,  # same transitions per update as looped arms
        lr=lr,
        lr_floor=cfg.train.lr_floor,
        grad_clip=cfg.train.grad_clip,
        log_every=cfg.train.log_every,
    )


def looped_train_config(cfg: ThreadCConfig, lr: float, steps: int) -> LoopTrainConfig:
    t = cfg.train
    return LoopTrainConfig(steps, t.batch_windows, t.window, lr, t.lr_floor, t.grad_clip, t.log_every)


def persistence() -> MemorylessPredictor:
    return MemorylessPredictor(PersistencePredictor().step)


# ---------------------------------------------------------------- hyperparameters, validation data only


def select_for_family(family: str, cfg: ThreadCConfig) -> dict:
    """K on the full recipe, then a learning rate per arm. Score: mean over val tiers of error / persistence."""
    started = time.perf_counter()
    seed = "hp_select"
    env, sequences, transitions = training_data(family, seed, cfg)
    scale = compute_obs_scale(family)
    low, high = env.obs_low, env.obs_high
    h = cfg.primary_horizon
    val_sets = []
    for tier in ("val_in", "val_shift"):
        val_env = make_env(family, cfg.n_val_envs, tier, derive_seed("val_envs", seed))
        holdout = collect_holdout(val_env, cfg.holdout_traj, max(cfg.starts) + h, make_generator("val", family, tier))
        true, act = make_windows(holdout, cfg.starts, h)
        reference = rollout_metrics(persistence(), true, act, (h,), scale, low, high)[f"h{h}"]["err"]
        val_sets.append((true, act, reference))

    def score(predict_factory) -> float:
        ratios = []
        for true, act, reference in val_sets:
            err = rollout_metrics(predict_factory(), true, act, (h,), scale, low, high)[f"h{h}"]["err"]
            ratios.append(err / max(reference, 1e-12))
        value = float(np.mean(ratios))
        return value if math.isfinite(value) else float("inf")

    def looped_score(arm: str, loops: int, lr: float) -> float:
        trained = train_looped(sequences, arm_config(env, arm, loops, cfg), looped_train_config(cfg, lr, cfg.selection_steps), seed)
        return score(lambda: LoopedPredictor(trained.model))

    def mlp_score(name: str, loops: int, lr: float) -> float:
        backbone = train_nolatent(transitions, baseline_config(env, name, loops, cfg), mlp_train_config(cfg, lr, cfg.selection_steps), seed)
        return score(lambda: MemorylessPredictor(backbone.model.step))

    table = []
    best_loops, best_lr, best = cfg.loop_choices[0], cfg.lr_choices[0], float("inf")
    for loops in cfg.loop_choices:
        for lr in cfg.lr_choices:
            value = looped_score("full", loops, lr)
            table.append({"arm": "full", "loops": loops, "lr": lr, "score": value})
            if value < best:
                best_loops, best_lr, best = loops, lr, value
    lrs = {"full": best_lr}
    for arm in list(ARMS) + list(SELECTED_BASELINES):
        if arm == "full":
            continue
        arm_best, arm_lr = float("inf"), cfg.lr_choices[0]
        for lr in cfg.lr_choices:
            value = mlp_score(arm, best_loops, lr) if arm in SELECTED_BASELINES else looped_score(arm, best_loops, lr)
            table.append({"arm": arm, "loops": best_loops, "lr": lr, "score": value})
            if value < arm_best:
                arm_best, arm_lr = value, lr
        lrs[arm] = arm_lr
    return {
        "loops": best_loops,
        "grad_loops": max(1, best_loops // 2),
        "lr": lrs,
        "table": table,
        "rule": "lowest mean over val_in and val_shift of error / persistence error at the primary horizon",
        "wall_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------- measurements


def attach_strata(cells: dict, contact_by_h: dict | None) -> None:
    """Replace each cell's per-window errors with contact and contact-free RMSE."""
    for key, cell in cells.items():
        sq = cell.pop("_sq", None)
        if sq is None or contact_by_h is None:
            continue
        mask = contact_by_h[key]
        cell["err_contact_windows"] = sq[mask].mean().sqrt().item() if mask.any() else None
        cell["err_free_windows"] = sq[~mask].mean().sqrt().item() if (~mask).any() else None
        cell["frac_contact_windows"] = mask.float().mean().item()


def measure_step_us(advance, o: torch.Tensor, a: torch.Tensor, repeats: int = 20) -> float:
    """Wall-clock per rollout step on one batch."""
    carry = None
    with torch.no_grad():
        started = time.perf_counter()
        for _ in range(repeats):
            _, carry = advance(o, a, carry)
    return (time.perf_counter() - started) / repeats * 1e6


def cem_for(env: BatchEnv, cfg: ThreadCConfig) -> CEMConfig:
    return CEMConfig(
        horizon=cfg.horizon_override or env.plan_horizon,
        n_candidates=cfg.n_candidates,
        n_elites=cfg.n_elites,
        n_iters=cfg.n_iters,
    )


def flatten_trajs(trajs: Trajectories) -> tuple[torch.Tensor, torch.Tensor]:
    E, B, T, A = trajs.act.shape
    return trajs.obs.reshape(E * B, T + 1, -1), trajs.act.reshape(E * B, T, A)


@torch.no_grad()
def h9_signals(model: LoopedDynamics, trajs: Trajectories, scale: torch.Tensor, gauss, kappa: float) -> dict:
    """Every signal on signal time, from one forward pass of one model over the same trajectories."""
    obs, act = flatten_trajs(trajs)
    trace = teacher_forced_trace(model, obs, act)
    out = {"residual_last": trace["residual_last"], "residual_mean": trace["residual_mean"]}
    r = (obs[:, 1:] - trace["pred"]) / scale  # (N, T, O); r[:, t] is the surprise on obs[t + 1]
    out["_innovation"] = r
    if gauss is not None:
        out["gauss_nll"] = to_signal_time(gauss.nll(r))
        out["gauss_cusum"] = to_signal_time(gauss.cusum(r, kappa))
    return out


def contact_scores(signals: dict, ap: torch.Tensor, wall: torch.Tensor, cfg: ThreadCConfig) -> dict:
    E, B, T = ap.shape
    ap_flat, wall_flat = ap.reshape(E * B, T), wall.reshape(E * B, T)
    lag = contact_masks(ap_flat, wall_flat, cfg.h9_tol, cfg.h9_burn, "lag")
    lead = contact_masks(ap_flat, wall_flat, cfg.h9_tol, cfg.h9_burn, "lead")
    out = {}
    for name, sig in signals.items():
        if name.startswith("_"):
            continue
        out[name] = {
            "onset": auroc_groups(sig[lag["positive"]], sig[lag["negative"]]),
            "onset_lead": auroc_groups(sig[lead["positive"]], sig[lead["negative"]]),
            "sustained": auroc_groups(sig[lag["sustained"]], sig[lag["negative"]]),
        }
    return out


def contact_set(env, cfg: ThreadCConfig, tag) -> tuple[Trajectories, torch.Tensor, torch.Tensor]:
    """Half push, half smooth: pushes make contact, smooth motion supplies free flight."""
    n_push = cfg.h9_traj - cfg.h9_traj // 2
    a = collect_contact_trajectories(env, n_push, cfg.h9_steps, "push", make_generator(tag, "contact_push"))
    b = collect_contact_trajectories(env, cfg.h9_traj // 2, cfg.h9_steps, "smooth", make_generator(tag, "contact_smooth"))
    trajs = Trajectories(torch.cat([a[0].obs, b[0].obs], dim=1), torch.cat([a[0].act, b[0].act], dim=1))
    return trajs, torch.cat([a[1], b[1]], dim=1), torch.cat([a[2], b[2]], dim=1)


def switch_pair(pre, post, cfg: ThreadCConfig, tag) -> tuple[Trajectories, Trajectories]:
    gen_tag = (tag, "switch")
    switched = collect_switching(pre, post, cfg.h9_traj, cfg.h9_steps, cfg.h9_switch, pre.default_probe, make_generator(gen_tag))
    control = collect_switching(pre, pre, cfg.h9_traj, cfg.h9_steps, cfg.h9_switch, pre.default_probe, make_generator(gen_tag))
    return switched, control


def h9_data(family: str, seed, cfg: ThreadCConfig) -> dict:
    """Generated once per job, so every arm and every signal is scored on the same trajectories."""
    data = {}
    val_pre = make_env(family, cfg.n_val_envs, "val_in", derive_seed("h9_val", seed))
    val_post = make_env(family, cfg.n_val_envs, "val_shift", derive_seed("h9_val", seed))
    data["val_switch"] = switch_pair(val_pre, val_post, cfg, ("h9_val", family, seed))
    test_pre = make_env(family, cfg.n_eval_envs, "interp", derive_seed("h9_test", seed))
    data["test_shift"] = {}
    for tier in cfg.h9_shift_tiers:
        post = make_env(family, cfg.n_eval_envs, tier, derive_seed("h9_test", seed))
        data["test_shift"][tier] = {
            "pair": switch_pair(test_pre, post, cfg, ("h9_test", family, seed, tier)),
            "mechanisms": post.mechanisms,
        }
    if family == "puck":
        data["val_contact"] = contact_set(val_pre, cfg, ("h9_val", family, seed))
        data["test_contact"] = contact_set(test_pre, cfg, ("h9_test", family, seed))
    return data


def _better_signal(scores: dict) -> str:
    """Between the two baseline signals, by validation AUROC. NaN never wins."""
    nll, cusum = scores["gauss_nll"], scores["gauss_cusum"]
    if not math.isfinite(cusum):
        return "gauss_nll"
    if not math.isfinite(nll):
        return "gauss_cusum"
    return "gauss_nll" if nll >= cusum else "gauss_cusum"


def h9_for_arm(model: LoopedDynamics, data: dict, cfg: ThreadCConfig, scale: torch.Tensor) -> dict:
    """AUROC of the loop residual and of the Gaussian baseline, on the same data from the same model."""
    val_switched, val_control = data["val_switch"]
    clean = h9_signals(model, val_control, scale, None, cfg.h9_kappa)
    r = clean["_innovation"][:, cfg.h9_burn :]
    try:
        gauss = GaussianResidualModel.fit(r.reshape(-1, r.shape[-1]))
    except (ValueError, RuntimeError) as err:  # a diverged arm has no usable residuals
        gauss = None
        fit_error = str(err)
    else:
        fit_error = None
    out = {"gaussian_fit_error": fit_error, "gaussian_rollouts": int(r.shape[0])}
    if gauss is None:
        return out
    kappa = cfg.h9_kappa
    args = (cfg.h9_switch, cfg.h9_window, cfg.h9_burn)

    # Which baseline signal to use is chosen per target on validation environments.
    val_sw = h9_signals(model, val_switched, scale, gauss, kappa)
    val_ct = h9_signals(model, val_control, scale, gauss, kappa)
    val_shift_scores = {s: shift_auroc(val_sw[s], val_ct[s], *args)["auroc"] for s in ("gauss_nll", "gauss_cusum")}
    shift_baseline = _better_signal(val_shift_scores)
    out["shift_baseline"] = {"chosen": shift_baseline, "val_auroc": val_shift_scores}

    out["shift"] = {}
    for tier, entry in data["test_shift"].items():
        switched, control = entry["pair"]
        sw = h9_signals(model, switched, scale, gauss, kappa)
        ct = h9_signals(model, control, scale, gauss, kappa)
        names = [s for s in sw if not s.startswith("_")]
        scores = {s: shift_auroc(sw[s], ct[s], *args) for s in names}
        record = {"signals": scores, "residual": scores["residual_last"]["auroc"], "baseline": scores[shift_baseline]["auroc"]}
        mechanisms = entry["mechanisms"]
        if len(set(mechanisms)) > 1:
            B = switched.act.shape[1]
            record["by_mechanism"] = {}
            for name in sorted(set(mechanisms)):
                keep = torch.tensor([m == name for m in mechanisms]).repeat_interleave(B)
                record["by_mechanism"][name] = {
                    s: shift_auroc(sw[s], ct[s], *args, keep=keep)["auroc"] for s in ("residual_last", shift_baseline)
                }
        out["shift"][tier] = record

    if "test_contact" in data:
        val_trajs, val_ap, val_wall = data["val_contact"]
        val_scores = contact_scores(h9_signals(model, val_trajs, scale, gauss, kappa), val_ap, val_wall, cfg)
        choice = {s: val_scores[s]["onset"]["auroc"] for s in ("gauss_nll", "gauss_cusum")}
        contact_baseline = _better_signal(choice)
        trajs, ap, wall = data["test_contact"]
        scores = contact_scores(h9_signals(model, trajs, scale, gauss, kappa), ap, wall, cfg)
        out["contact_baseline"] = {"chosen": contact_baseline, "val_auroc": choice}
        out["contact"] = {
            "signals": scores,
            "residual": scores["residual_last"]["onset"]["auroc"],
            "baseline": scores[contact_baseline]["onset"]["auroc"],
        }
    return out


# ---------------------------------------------------------------- one job


def run_job(family: str, seed: int, cfg: ThreadCConfig, hp: dict) -> dict:
    """One training seed on one family."""
    env, sequences, transitions = training_data(family, seed, cfg)
    loops = hp["loops"]
    scale = compute_obs_scale(family)
    low, high = env.obs_low, env.obs_high
    ablation_record = check_ablations(arm_config(env, "full", loops, cfg), ARMS, seed)

    # Training.
    started = time.perf_counter()
    models: dict[str, torch.nn.Module] = {}
    arms: dict[str, dict] = {}
    sample_o = transitions.obs[:, 0:2].reshape(-1, env.obs_dim)[:256]
    sample_a = transitions.act[:, 0:2].reshape(-1, env.act_dim)[:256]
    for arm in ARMS:
        config = arm_config(env, arm, loops, cfg)
        trained = train_looped(sequences, config, looped_train_config(cfg, hp["lr"][arm], cfg.train.steps), seed)
        model = trained.model
        models[arm] = model
        low_depth, high_depth = config.trained_depth_range()
        arms[arm] = {
            "kind": "looped",
            "switches": ARMS[arm],
            "lr": hp["lr"][arm],
            "params": count_params(model),
            "forward_macs_per_step": model.macs_per_step(),
            "forward_macs_by_loops": {str(n): model.macs_per_step(n) for n in sorted(set(cfg.loop_grid) | {loops})},
            "measured_step_us": measure_step_us(LoopedPredictor(model).advance, sample_o, sample_a),
            "depth": {
                "trained_loops": loops,
                "grad_loops": config.grad_loops,
                "trained_depth_range": [low_depth, high_depth],
                "train_window": cfg.train.window,
            },
            "train": trained.stats,
            "log": trained.log,
            "sensitivity": sensitivity(model, sample_o, sample_a, make_generator("sensitivity", family, seed, arm)),
        }
        print(f"  {arm}: {trained.stats['wall_s']:.1f}s, final loss {trained.stats['final_loss']}, "
              f"skipped {trained.stats['skipped_nonfinite_updates']}", flush=True)

    # Baselines. Same data; one-step loss, which is what teacher forcing gives the looped arms.
    full_macs = models["full"].macs_per_step()
    baseline_cfgs = {}
    for name in ("mlp_params", "mlp_compute"):
        model_cfg = baseline_config(env, name, loops, cfg)
        baseline_cfgs[name] = model_cfg
        backbone = train_nolatent(transitions, model_cfg, mlp_train_config(cfg, hp["lr"][name], cfg.train.steps), seed)
        models[name] = backbone.model
        arms[name] = _mlp_record(backbone, model_cfg, hp["lr"][name], cfg, sample_o, sample_a, cfg.train.steps)
        if name == "mlp_params":
            ratio = full_macs / mlp_macs(model_cfg)
            extra = max(0, round(cfg.train.steps * (ratio - 1.0)))
            control = continued_training(backbone, transitions, extra, mlp_train_config(cfg, hp["lr"][name], cfg.train.steps), seed)
            models["mlp_params_matched"] = control.model
            record = _mlp_record(control, model_cfg, hp["lr"][name], cfg, sample_o, sample_a, cfg.train.steps + extra)
            record["extra_updates"] = extra
            arms["mlp_params_matched"] = record
    train_wall = time.perf_counter() - started
    print(f"  trained all arms in {train_wall:.1f}s (K = {loops})", flush=True)

    def predictor_for(arm: str, n: int | None = None):
        if arms[arm]["kind"] == "looped":
            return LoopedPredictor(models[arm], n)
        return MemorylessPredictor(models[arm].step)

    # At trained depth, per tier.
    started = time.perf_counter()
    references: dict[str, dict] = {}
    max_h = max(cfg.at_trained_horizons)
    for arm in arms:
        arms[arm]["at_trained"] = {}
    for tier in cfg.at_trained_tiers:
        test_env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("thread_c", family, seed, tier)
        holdout = collect_holdout(test_env, cfg.holdout_traj, max(cfg.starts) + max_h, make_generator(tag, "holdout"))
        true, act = make_windows(holdout, cfg.starts, max_h)
        contact_by_h = None
        if family == "puck":
            ap, _ = replay_contact_flags(test_env, holdout)
            contact_by_h = {f"h{h}": window_contact_mask(ap, cfg.starts, h) for h in cfg.at_trained_horizons}
        references[tier] = {
            "persistence": rollout_metrics(persistence(), true, act, cfg.at_trained_horizons, scale, low, high),
            "saturation": saturation_reference(true, cfg.at_trained_horizons, scale, len(cfg.starts)),
            "mechanisms": test_env.mechanisms,
        }
        for arm in arms:
            cells = rollout_metrics(
                predictor_for(arm), true, act, cfg.at_trained_horizons, scale, low, high,
                keep_per_env=True, per_window=contact_by_h is not None,
            )
            attach_strata(cells, contact_by_h)
            arms[arm]["at_trained"][tier] = cells
    at_trained_wall = time.perf_counter() - started
    print(f"  at-trained evaluation in {at_trained_wall:.1f}s", flush=True)

    # Loops by horizon. Long rollouts with many loops dominate the cost of evaluation.
    started = time.perf_counter()
    grid_reference = {}
    for arm in arms:
        arms[arm]["grid"] = {}
    for tier in cfg.grid_tiers:
        test_env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("thread_c_grid", family, seed, tier)
        holdout = collect_holdout(test_env, cfg.grid_traj, max(cfg.horizon_grid), make_generator(tag, "holdout"))
        true, act = make_windows(holdout, (0,), max(cfg.horizon_grid))
        grid_reference[tier] = {
            "persistence": rollout_metrics(persistence(), true, act, cfg.horizon_grid, scale, low, high),
            "saturation": saturation_reference(true, cfg.horizon_grid, scale, 1),
        }
        for arm in arms:
            if arms[arm]["kind"] == "looped":
                arms[arm]["grid"][tier] = rollout_grid(models[arm], true, act, cfg.loop_grid, cfg.horizon_grid, scale, low, high)
            else:
                arms[arm]["grid"][tier] = {
                    "no_loops": True,
                    "horizons": rollout_metrics(predictor_for(arm), true, act, cfg.horizon_grid, scale, low, high),
                }
    grid_wall = time.perf_counter() - started
    print(f"  loop-by-horizon grid in {grid_wall:.1f}s", flush=True)

    # Closed loop.
    started = time.perf_counter()
    closed = {}
    for tier in cfg.closed_loop_tiers:
        test_env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("thread_c_closed", family, seed, tier)
        cem = cem_for(test_env, cfg)
        task_seed = (tag, "task")
        oracle = OraclePredictor(test_env)
        out = closed_loop(test_env, oracle, test_env.sample_task(make_generator(task_seed)), cem, make_generator(tag, "cem"), cfg.episode_override)
        closed[tier] = {
            "oracle": {
                "success_rate": out["success"].float().mean().item(),
                "mean_cost": out["mean_cost"].mean().item(),
                "wall_s": out["wall_s"],
                "counts": oracle.counts.as_dict(),
            },
            "arms": {},
        }
        for arm in CLOSED_LOOP_ARMS:
            if arms[arm]["kind"] == "looped":
                predictor = PlanningPredictor(models[arm], None, cem.horizon)
            else:
                predictor = MemorylessPredictor(models[arm].step)
            out = closed_loop(test_env, predictor, test_env.sample_task(make_generator(task_seed)), cem, make_generator(tag, "cem"), cfg.episode_override)
            success = out["success"].float().mean().item()
            closed[tier]["arms"][arm] = {
                "success_rate": success,
                "failure_rate": 1.0 - success,
                "mean_cost": out["mean_cost"].mean().item(),
                "wall_s": out["wall_s"],
                "counts": predictor.counts.as_dict(),
                "t_calls": getattr(predictor, "t_calls", None),
            }
    closed_wall = time.perf_counter() - started
    print(f"  closed loop in {closed_wall:.1f}s", flush=True)

    # H9.
    started = time.perf_counter()
    data = h9_data(family, seed, cfg)
    h9 = {arm: h9_for_arm(models[arm], data, cfg, scale) for arm in ARMS}
    h9_wall = time.perf_counter() - started
    print(f"  H9 in {h9_wall:.1f}s", flush=True)

    return {
        "trained_loops": loops,
        "ablation_check": ablation_record,
        "baseline_configs": {k: asdict(v) for k, v in baseline_cfgs.items()},
        "arms": arms,
        "references": references,
        "grid_references": grid_reference,
        "closed_loop": closed,
        "h9": h9,
        "wall_s": {"train": train_wall, "at_trained": at_trained_wall, "grid": grid_wall, "closed_loop": closed_wall, "h9": h9_wall},
    }


def _mlp_record(backbone, model_cfg: ModelConfig, lr: float, cfg: ThreadCConfig, o, a, updates: int) -> dict:
    predictor = MemorylessPredictor(backbone.model.step)
    return {
        "kind": "mlp",
        "lr": lr,
        "width": model_cfg.hidden,
        "params": mlp_param_count(model_cfg),
        "forward_macs_per_step": mlp_macs(model_cfg),
        "measured_step_us": measure_step_us(predictor.advance, o, a),
        "depth": None,
        "train": {
            **backbone.stats,
            "updates": updates,
            "forward_macs_total": updates * cfg.train.batch_windows * cfg.train.window * mlp_macs(model_cfg),
            "backward_calls": updates,
        },
        "log": backbone.log,
    }


# ---------------------------------------------------------------- summaries


def _paired(a, b, n_boot: int, threshold: float | None = None) -> dict:
    """paired_comparison over seeds where both values are finite."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"n_seeds": len(pairs), "verdict": "insufficient_seeds"}
    xs, ys = zip(*pairs)
    if threshold is None:
        return paired_comparison(xs, ys, n_boot=n_boot)
    return paired_comparison(xs, ys, threshold=threshold, n_boot=n_boot)


def _mean(values) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def summarize(family: str, jobs: dict, cfg: ThreadCConfig) -> dict:
    """Protocol verdicts from whichever seeds have finished."""
    if not jobs:
        return {}
    seeds = sorted(jobs, key=int)
    n_boot = cfg.n_boot
    per = lambda fn: [fn(jobs[s]) for s in seeds]
    at = lambda arm, tier, h, key="err": per(lambda j: j["arms"][arm]["at_trained"][tier][f"h{h}"].get(key))
    ref = lambda kind, tier, h: per(
        lambda j: j["references"][tier][kind][f"h{h}"]["err"] if kind == "persistence" else j["references"][tier][kind][f"h{h}"]
    )
    primary, long_h = cfg.primary_horizon, cfg.long_horizon
    tiers = cfg.at_trained_tiers
    arm_names = list(jobs[seeds[0]]["arms"])
    summary: dict = {"n_seeds": len(seeds), "decision_tier": DECISION_TIER, "primary_horizon": primary, "long_horizon": long_h}

    # Compute, for every arm.
    summary["compute"] = {}
    for arm in arm_names:
        train = per(lambda j: j["arms"][arm]["train"])
        summary["compute"][arm] = {
            "params": _mean(per(lambda j: j["arms"][arm]["params"])),
            "forward_macs_per_step": _mean(per(lambda j: j["arms"][arm]["forward_macs_per_step"])),
            "measured_step_us": _mean(per(lambda j: j["arms"][arm]["measured_step_us"])),
            "train_wall_s": _mean([t.get("wall_s") for t in train]),
            "train_forward_macs_total": _mean([t.get("forward_macs_total") for t in train]),
            "train_updates": _mean([t.get("updates") for t in train]),
            "t_calls_grad": _mean([t.get("t_calls_grad") for t in train]),
            "t_calls_no_grad": _mean([t.get("t_calls_no_grad") for t in train]),
            "skipped_nonfinite_updates": _mean([t.get("skipped_nonfinite_updates") for t in train]),
        }
    summary["trained_loops"] = jobs[seeds[0]]["trained_loops"]

    # Means at trained depth, with persistence and saturation for scale.
    summary["means_at_trained"] = {}
    for tier in tiers:
        row = {arm: {f"h{h}": _mean(at(arm, tier, h)) for h in cfg.at_trained_horizons} for arm in arm_names}
        for arm in arm_names:
            row[arm]["blowup_frac_long"] = _mean(at(arm, tier, long_h, "blowup_frac"))
            row[arm]["nonfinite_frac_long"] = _mean(at(arm, tier, long_h, "nonfinite_frac"))
        row["persistence"] = {f"h{h}": _mean(ref("persistence", tier, h)) for h in cfg.at_trained_horizons}
        row["saturation"] = {f"h{h}": _mean(ref("saturation", tier, h)) for h in cfg.at_trained_horizons}
        summary["means_at_trained"][tier] = row

    # Loops by horizon, means over seeds, with relation to the trained depth.
    summary["grid"] = {}
    for tier in cfg.grid_tiers:
        table = {}
        for arm in arm_names:
            grid = jobs[seeds[0]]["arms"][arm]["grid"][tier]
            if grid.get("no_loops"):
                table[arm] = {
                    f"h{h}": {
                        "err": _mean(per(lambda j: j["arms"][arm]["grid"][tier]["horizons"][f"h{h}"]["err"])),
                        "blowup_frac": _mean(per(lambda j: j["arms"][arm]["grid"][tier]["horizons"][f"h{h}"]["blowup_frac"])),
                    }
                    for h in cfg.horizon_grid
                }
                continue
            table[arm] = {
                "trained_loops": grid["trained_loops"],
                "trained_depth_range": grid["trained_depth_range"],
                "loops": {},
            }
            for n, row in grid["loops"].items():
                cell = lambda h, key: per(lambda j: j["arms"][arm]["grid"][tier]["loops"][n]["horizons"][f"h{h}"][key])
                table[arm]["loops"][n] = {
                    "relation": row["relation"],
                    "is_trained_loops": row["is_trained_loops"],
                    "horizons": {
                        f"h{h}": {
                            "err": _mean(cell(h, "err")),
                            "err_intact": _mean(cell(h, "err_intact")),
                            "nonfinite_frac": _mean(cell(h, "nonfinite_frac")),
                            "clipped_frac": _mean(cell(h, "clipped_frac")),
                            "blowup_frac": _mean(cell(h, "blowup_frac")),
                            "z_rms_median": _mean(cell(h, "z_rms_median")),
                            "z_collapse_frac": _mean(cell(h, "z_collapse_frac")),
                            "z_spread": _mean(cell(h, "z_spread")),
                        }
                        for h in cfg.horizon_grid
                    },
                }
        summary["grid"][tier] = table

    # H7. a = full, b = ablated; a_wins at threshold 0.5 means the ablated arm is at least 2x worse.
    h7: dict = {
        "operational_definitions": {
            "long_rollout": f"horizon {long_h} at trained loops on {DECISION_TIER}; HYPOTHESES.md names no horizon",
            "stable": f"less than 2x worse than full, and blowup_frac below {BLOWUP_LIMIT}",
            "note": "2610.31943 predicts no_norm may be better; no_norm_vs_full records either outcome",
        },
        "tiers": {},
    }
    for tier in tiers:
        full_long = _mean(at("full", tier, long_h))
        saturation = _mean(ref("saturation", tier, long_h))
        tier_out = {
            "full_beats_persistence_primary": (_mean(at("full", tier, primary)) or math.inf) < (_mean(ref("persistence", tier, primary)) or 0.0),
            "two_x_reachable": full_long is not None and saturation is not None and 2.0 * full_long < saturation,
            "ablations": {},
        }
        for arm in H7_ABLATIONS:
            tier_out["ablations"][arm] = {
                "primary": _paired(at("full", tier, primary), at(arm, tier, primary), n_boot),
                "long_protocol": _paired(at("full", tier, long_h), at(arm, tier, long_h), n_boot),
                "long_2x": _paired(at("full", tier, long_h), at(arm, tier, long_h), n_boot, TWO_X_THRESHOLD),
                "blowup_frac_long": _mean(at(arm, tier, long_h, "blowup_frac")),
                "per_seed_long": at(arm, tier, long_h),
            }
        tier_out["full_per_seed_long"] = at("full", tier, long_h)
        tier_out["full_blowup_frac_long"] = _mean(at("full", tier, long_h, "blowup_frac"))
        tier_out["no_norm_vs_full"] = tier_out["ablations"]["no_norm"]["long_protocol"]["verdict"]
        h7["tiers"][tier] = tier_out
    if DECISION_TIER in h7["tiers"]:
        decision = h7["tiers"][DECISION_TIER]
        verdicts = {arm: decision["ablations"][arm]["long_2x"]["verdict"] for arm in H7_ABLATIONS}
        h7["verdicts_long_2x"] = verdicts
        h7["outcome"] = h7_outcome(verdicts, decision["ablations"]["plain"]["blowup_frac_long"] or 0.0, decision["full_beats_persistence_primary"])
    if DECISION_TIER in cfg.grid_tiers:
        n = str(cfg.product_loops)
        cell = lambda arm, key: per(lambda j: j["arms"][arm]["grid"][DECISION_TIER]["loops"][n]["horizons"][f"h{long_h}"][key])
        product = {"loops": cfg.product_loops, "horizon": long_h, "ablations": {}}
        for arm in H7_ABLATIONS:
            product["ablations"][arm] = {
                "long_2x": _paired(cell("full", "err"), cell(arm, "err"), n_boot, TWO_X_THRESHOLD),
                "blowup_frac": _mean(cell(arm, "blowup_frac")),
            }
        product["full_blowup_frac"] = _mean(cell("full", "blowup_frac"))
        if DECISION_TIER in h7["tiers"]:
            verdicts = {arm: product["ablations"][arm]["long_2x"]["verdict"] for arm in H7_ABLATIONS}
            product["outcome"] = h7_outcome(verdicts, product["ablations"]["plain"]["blowup_frac"] or 0.0, h7["tiers"][DECISION_TIER]["full_beats_persistence_primary"])
        h7["product_above_trained_depth"] = product
    summary["H7"] = h7

    # H8. a = every loop, b = prelude-only.
    h8: dict = {"a": "full", "b": "prelude_only", "decisive_family": "puck", "family": family, "tiers": {}}
    for tier in tiers:
        tier_out = {
            "primary": _paired(at("full", tier, primary), at("prelude_only", tier, primary), n_boot),
            "long": _paired(at("full", tier, long_h), at("prelude_only", tier, long_h), n_boot),
        }
        if family == "puck":
            tier_out["contact_windows_primary"] = _paired(
                at("full", tier, primary, "err_contact_windows"), at("prelude_only", tier, primary, "err_contact_windows"), n_boot
            )
            tier_out["free_windows_primary"] = _paired(
                at("full", tier, primary, "err_free_windows"), at("prelude_only", tier, primary, "err_free_windows"), n_boot
            )
        h8["tiers"][tier] = tier_out
    if DECISION_TIER in h8["tiers"]:
        h8["outcome"] = h8_outcome(h8["tiers"][DECISION_TIER]["primary"]["verdict"])
    h8["action_sensitivity"] = {a: _mean(per(lambda j: j["arms"][a]["sensitivity"]["action_sensitivity"])) for a in ("full", "prelude_only")}
    summary["H8"] = h8

    # Carried state against re-initialization, and whether carrying had any effect at all.
    summary["carry"] = {
        tier: {
            "primary": _paired(at("full", tier, primary), at("reinit", tier, primary), n_boot),
            "long": _paired(at("full", tier, long_h), at("reinit", tier, long_h), n_boot),
        }
        for tier in tiers
    }
    summary["carry"]["carry_sensitivity_full"] = _mean(per(lambda j: j["arms"]["full"]["sensitivity"]["carry_sensitivity"]))

    # Baselines: a = full, b = baseline, so b_wins means the non-looped model is better.
    summary["baselines"] = {
        tier: {
            name: {
                "primary": _paired(at("full", tier, primary), at(name, tier, primary), n_boot),
                "long": _paired(at("full", tier, long_h), at(name, tier, long_h), n_boot),
            }
            for name in BASELINES
        }
        for tier in tiers
    }

    # Closed loop, gated by whether the oracle solves the task.
    closed = {}
    for tier in cfg.closed_loop_tiers:
        oracle = _mean(per(lambda j: j["closed_loop"][tier]["oracle"]["success_rate"]))
        fail = lambda arm: per(lambda j: j["closed_loop"][tier]["arms"][arm]["failure_rate"])
        cost = lambda arm: per(lambda j: j["closed_loop"][tier]["arms"][arm]["mean_cost"])
        closed[tier] = {
            "oracle_success_rate": oracle,
            "valid_as_evidence": oracle is not None and oracle >= 0.5,
            "success_rate": {arm: _mean(per(lambda j: j["closed_loop"][tier]["arms"][arm]["success_rate"])) for arm in CLOSED_LOOP_ARMS},
            "full_vs": {
                arm: {"failure_rate": _paired(fail("full"), fail(arm), n_boot), "mean_cost": _paired(cost("full"), cost(arm), n_boot)}
                for arm in CLOSED_LOOP_ARMS
                if arm != "full"
            },
        }
    summary["closed_loop"] = closed

    summary["H9"] = _summarize_h9(family, jobs, seeds, cfg)
    return summary


def _summarize_h9(family: str, jobs: dict, seeds: list, cfg: ThreadCConfig) -> dict:
    """a = residual, b = baseline, compared on 1 - AUROC, where lower is better."""
    n_boot = cfg.n_boot
    nan = float("nan")

    def get_target(job, target, tier=None, arm="full") -> tuple[float, float]:
        entry = job["h9"][arm].get(target)
        if entry is None:
            return nan, nan
        if tier is not None:
            entry = entry[tier]
        return entry["residual"], entry["baseline"]

    def judge(pairs: list[tuple[float, float]]) -> dict:
        residual = [p[0] for p in pairs]
        baseline = [p[1] for p in pairs]
        comparison = _paired([1.0 - x for x in residual], [1.0 - x for x in baseline], n_boot)
        mean_residual = _mean(residual)
        return {
            "mean_auroc_residual": mean_residual,
            "mean_auroc_baseline": _mean(baseline),
            "per_seed_residual": residual,
            "per_seed_baseline": baseline,
            "comparison_on_1_minus_auroc": comparison,
            "passes": h9_target_passes(mean_residual if mean_residual is not None else nan, comparison["verdict"]),
        }

    targets = {}
    tier_pairs = {tier: [get_target(jobs[s], "shift", tier) for s in seeds] for tier in cfg.h9_shift_tiers}

    def pooled_mean(i: int, side: int) -> float:
        value = _mean([tier_pairs[t][i][side] for t in cfg.h9_shift_tiers])
        return nan if value is None else value

    pooled = [(pooled_mean(i, 0), pooled_mean(i, 1)) for i in range(len(seeds))]
    targets["shift_onset"] = judge(pooled)
    targets["shift_onset"]["by_tier"] = {t: judge(p) for t, p in tier_pairs.items()}
    if family == "puck":
        targets["contact_onset"] = judge([get_target(jobs[s], "contact") for s in seeds])
    verdicts = [t["comparison_on_1_minus_auroc"]["verdict"] for t in targets.values()]
    if any(v == "insufficient_seeds" for v in verdicts):
        outcome = "insufficient_seeds"
    else:
        outcome = "supported" if all(t["passes"] for t in targets.values()) else "killed"
    out = {"signal": "residual_last on the full arm", "targets": targets, "outcome": outcome}

    # Descriptive: the same AUROCs on the other arms. Not part of the verdict.
    other = {}
    for arm in ARMS:
        if arm == "full":
            continue
        row = {t: _mean([get_target(jobs[s], "shift", t, arm)[0] for s in seeds]) for t in cfg.h9_shift_tiers}
        if family == "puck":
            row["contact"] = _mean([get_target(jobs[s], "contact", None, arm)[0] for s in seeds])
        other[arm] = row
    out["residual_auroc_other_arms"] = other
    return out


# ---------------------------------------------------------------- output


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return _jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None  # strict JSON has no NaN or infinity
    return x


def source_fingerprint() -> dict:
    """Hash of the code that ran, so results stay traceable without git."""
    files = sorted((ROOT / "src" / "intr").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny settings, checks the pipeline in under 3 minutes")
    mode.add_argument("--full", action="store_true", help="protocol settings, estimated 1 to 2 hours on 2 cores")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of training seeds")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="exact training seeds to run; overrides --seeds")
    parser.add_argument("--families", nargs="+", default=None, choices=["pendulum", "puck"])
    parser.add_argument("--max-minutes", type=float, default=None, help="start no new work after this long")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_c")
    args = parser.parse_args()

    cfg = smoke_config() if args.smoke else full_config()
    if args.seeds is not None:
        cfg.seeds = list(range(args.seeds))
    if args.seed_list is not None:
        cfg.seeds = args.seed_list
    if args.families:
        cfg.families = args.families

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"thread_c_{cfg.mode}.json"
    started = time.perf_counter()
    output = {
        "config": asdict(cfg),
        "arms": ARMS,
        "baselines": list(BASELINES),
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
        "selection": {},
        "jobs": {f: {} for f in cfg.families},
        "summary": {},
        "stopped_early": False,
    }

    def save() -> None:
        output["metadata"]["elapsed_s"] = time.perf_counter() - started
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(output), indent=1))
        tmp.replace(path)

    def out_of_time(what: str) -> bool:
        elapsed = (time.perf_counter() - started) / 60.0
        if args.max_minutes is not None and elapsed >= args.max_minutes:
            output["stopped_early"] = True
            save()
            print(f"stopped at {elapsed:.1f} minutes before {what}", flush=True)
            return True
        return False

    for family in cfg.families:
        if out_of_time(f"selection for {family}"):
            return
        print(f"selecting hyperparameters for {family}", flush=True)
        output["selection"][family] = select_for_family(family, cfg)
        print(f"  K = {output['selection'][family]['loops']} in {output['selection'][family]['wall_s']:.1f}s", flush=True)
        save()

    # Seeds run in the outer loop, so a stopped run still has both families at equal depth.
    for seed in cfg.seeds:
        for family in cfg.families:
            if out_of_time(f"seed {seed}, {family}"):
                return
            elapsed = (time.perf_counter() - started) / 60.0
            print(f"seed {seed}, {family} (elapsed {elapsed:.1f} min)", flush=True)
            output["jobs"][family][str(seed)] = run_job(family, seed, cfg, output["selection"][family])
            output["summary"] = {f: summarize(f, j, cfg) for f, j in output["jobs"].items() if j}
            save()
    save()
    print(f"wrote {path} in {(time.perf_counter() - started) / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
