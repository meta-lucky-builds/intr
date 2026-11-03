"""Thread A: what should adapt (H1, H2, H3 in docs/HYPOTHESES.md).

    python experiments/thread_a.py --smoke     # under 2 minutes, checks the pipeline; numbers mean nothing
    python experiments/thread_a.py --full      # about 1 hour on 2 CPU cores

Writes results/thread_a/thread_a_<mode>.json, rewritten after every (seed, family) job, so a stopped run
keeps what it finished. --max-minutes stops starting new jobs after that long.

Design of a run, per training seed and family:
    1. Train backbones: latent (T2-Z), context, nolatent, control (nolatent continued).
    2. Choose hyperparameters per condition on validation environments only (val_in and val_shift).
       val_shift uses a weakened actuator, a mechanism no test tier uses.
    3. On fresh environments per tier: adapt from K trajectories, then measure open-loop error (all K)
       and closed-loop success (K = 1). All methods in a cell see the same data, held-out trajectories,
       initial states, goals and planning noise. Test environments differ per training seed, so the
       CI over seeds includes environment sampling.
    4. H3 conditions: same methods, four kinds of adaptation data.

The primary horizon for the H1 and H2 outcome is 10. HYPOTHESES.md does not name one, so all horizons are
reported, and closed-loop comparisons appear next to them.
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

from intr.adapt import GRAD_WRITE_ONE_PASS_GRID, METHODS, adapt  # noqa: E402
from intr.envs import (  # noqa: E402
    BatchEnv,
    Transitions,
    collect_iid_states,
    collect_trajectories,
    derive_seed,
    make_env,
    make_generator,
)
from intr.eval import (  # noqa: E402
    CEMConfig,
    OraclePredictor,
    PersistencePredictor,
    ValSet,
    closed_loop,
    collect_holdout,
    compute_obs_scale,
    feature_collinearity,
    open_loop_errors,
    paired_comparison,
    select_hyperparameters,
)
from intr.models import ModelConfig  # noqa: E402
from intr.train import (  # noqa: E402
    TrainConfig,
    TransitionData,
    continued_training,
    latent_r2,
    train_context,
    train_latent,
    train_nolatent,
)

PRIMARY_HORIZON = 10
PRIMARY_K = 1
H3_CONDITIONS = ("smooth", "shuffled", "random_action", "iid_state")


@dataclass
class Condition:
    label: str
    method: str
    backbone: str  # key into the dict of trained backbones
    grid: list[dict] | None = None  # None means the method's registered grid


CONDITIONS = [
    Condition("frozen@latent", "frozen", "latent"),
    Condition("context@context", "context", "context"),
    Condition("latent_opt@latent", "latent_opt", "latent"),
    Condition("rls@latent", "rls", "latent"),
    Condition("grad_write@latent", "grad_write", "latent"),
    Condition("last_block@latent", "last_block", "latent"),
    Condition("hybrid@latent", "hybrid", "latent"),
    # rls@latent uses the mean latent, a z no training environment had, which may handicap it.
    # rls@nolatent is the same write on a backbone trained for the average environment.
    Condition("frozen@nolatent", "frozen", "nolatent"),
    Condition("rls@nolatent", "rls", "nolatent"),
    # Matched continued-training control for what the T2-Z scheme adds at training time.
    Condition("frozen@control", "frozen", "control"),
]

H3_METHODS = [
    Condition("rls", "rls", "latent"),
    Condition("grad_write_1pass", "grad_write", "latent", GRAD_WRITE_ONE_PASS_GRID),
    Condition("grad_write_tuned", "grad_write", "latent"),
]


@dataclass
class ThreadAConfig:
    mode: str
    seeds: list[int]
    families: list[str] = field(default_factory=lambda: ["pendulum", "puck"])
    z_dim: int = 8
    hidden: int = 128
    n_hidden: int = 2
    feature_dim: int = 64
    train: TrainConfig = field(default_factory=TrainConfig)
    train_envs: int = 128
    train_traj: int = 8
    train_steps: int = 40
    n_eval_envs: int = 21  # at least 20 per protocol; 21 balances the three puck mechanisms
    n_val_envs: int = 20
    ks: list[int] = field(default_factory=lambda: [1, 3])
    adapt_steps: int = 40  # transitions per adaptation trajectory
    holdout_traj: int = 8
    holdout_steps: int = 60
    open_loop_tiers: list[str] = field(default_factory=lambda: ["train", "interp", "extrap", "oof"])
    closed_loop_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    h3_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    n_candidates: int = 32
    n_elites: int = 6
    n_iters: int = 3
    horizon_override: int | None = None
    episode_override: int | None = None
    oracle: bool = True
    n_boot: int = 10000


def smoke_config() -> ThreadAConfig:
    return ThreadAConfig(
        mode="smoke",
        seeds=[0, 1],
        z_dim=4,
        hidden=32,
        feature_dim=16,
        train=TrainConfig(model_steps=150, warmup_steps=30, block_steps=30, batch_size=64, log_every=50, context_envs_per_step=8),
        train_envs=12,
        train_traj=3,
        train_steps=15,
        n_eval_envs=6,
        n_val_envs=4,
        adapt_steps=15,
        holdout_traj=2,
        n_candidates=8,
        n_elites=2,
        n_iters=2,
        horizon_override=4,
        episode_override=6,
        n_boot=500,
    )


def full_config() -> ThreadAConfig:
    return ThreadAConfig(mode="full", seeds=[0, 1, 2, 3, 4])


def make_adapt_data(env: BatchEnv, condition: str, k: int, n_steps: int, tag) -> Transitions:
    """All H3 conditions yield k * n_steps transitions from the same environments.

    smooth and shuffled are the same transitions by construction (same generator tag), so features and
    their collinearity are identical. Shuffling changes only the order. RLS should be unaffected, so
    any difference there would be a bug. random_action and iid_state change the data itself.
    """
    if condition == "smooth":
        return collect_trajectories(env, k, n_steps, env.default_probe, make_generator(tag, "probe")).flat()
    if condition == "shuffled":
        data = collect_trajectories(env, k, n_steps, env.default_probe, make_generator(tag, "probe")).flat()
        return data.permuted(make_generator(tag, "permute"))
    if condition == "random_action":
        return collect_trajectories(env, k, n_steps, "iid", make_generator(tag, "random_action")).flat()
    if condition == "iid_state":
        return collect_iid_states(env, k * n_steps, make_generator(tag, "iid_state")).flat()
    raise ValueError(condition)


def cem_for(env: BatchEnv, cfg: ThreadAConfig) -> CEMConfig:
    return CEMConfig(
        horizon=cfg.horizon_override or env.plan_horizon,
        n_candidates=cfg.n_candidates,
        n_elites=cfg.n_elites,
        n_iters=cfg.n_iters,
    )


def train_all(family: str, seed: int, cfg: ThreadAConfig):
    gen = make_generator("data", family, seed)
    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    data = TransitionData.collect(env, cfg.train_traj, cfg.train_steps, gen)
    with_z = ModelConfig(env.obs_dim, env.act_dim, cfg.z_dim, cfg.hidden, cfg.n_hidden, cfg.feature_dim)
    no_z = ModelConfig(env.obs_dim, env.act_dim, 0, cfg.hidden, cfg.n_hidden, cfg.feature_dim)
    latent = train_latent(data, with_z, cfg.train, seed)
    context = train_context(data, with_z, cfg.train, seed)
    nolatent = train_nolatent(data, no_z, cfg.train, seed)
    control = continued_training(nolatent, data, latent.stats["latent_updates"], cfg.train, seed)
    backbones = {"latent": latent, "context": context, "nolatent": nolatent, "control": control}
    record = {
        name: {"stats": b.stats, "final_log": b.log[-3:], "log": b.log} for name, b in backbones.items()
    }
    record["latent_r2"] = dict(zip(env.hidden_names, latent_r2(latent.z_table, env.hidden_matrix())))
    return backbones, record


def val_sets_for(family: str, seed: int, cfg: ThreadAConfig, condition: str) -> list[ValSet]:
    sets = []
    for tier in ("val_in", "val_shift"):
        env = make_env(family, cfg.n_val_envs, tier, derive_seed("val_envs", seed))
        tag = ("val", family, seed, tier)
        holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
        data = make_adapt_data(env, condition, PRIMARY_K, cfg.adapt_steps, tag)
        sets.append(ValSet(data, holdout))
    return sets


def _errors_record(errors: dict[int, torch.Tensor], mechanisms: list[str]) -> dict:
    record = {f"h{h}": e.mean().item() for h, e in errors.items()}
    record["per_env"] = {f"h{h}": e.tolist() for h, e in errors.items()}
    names = sorted(set(mechanisms))
    if len(names) > 1:
        record["by_mechanism"] = {
            name: {
                f"h{h}": e[torch.tensor([m == name for m in mechanisms])].mean().item() for h, e in errors.items()
            }
            for name in names
        }
    return record


def run_job(family: str, seed: int, cfg: ThreadAConfig) -> dict:
    """One training seed on one family."""
    started = time.perf_counter()
    backbones, train_record = train_all(family, seed, cfg)
    train_wall = time.perf_counter() - started
    print(f"  trained in {train_wall:.1f}s; latent R2 {train_record['latent_r2']}", flush=True)
    scale = compute_obs_scale(family)
    probe_env = make_env(family, 1, "train", "bounds")
    low, high = probe_env.obs_low, probe_env.obs_high

    # Hyperparameter selection, on validation environments only.
    started = time.perf_counter()
    selection = {}
    val_smooth = val_sets_for(family, seed, cfg, "smooth")
    for cond in CONDITIONS:
        selection[cond.label] = select_hyperparameters(
            cond.method, backbones[cond.backbone], val_smooth, scale, low, high, grid=cond.grid, horizon=PRIMARY_HORIZON
        )
    h3_selection = {}
    for data_kind in H3_CONDITIONS:
        val = val_smooth if data_kind == "smooth" else val_sets_for(family, seed, cfg, data_kind)
        h3_selection[data_kind] = {
            m.label: select_hyperparameters(
                m.method, backbones[m.backbone], val, scale, low, high, grid=m.grid, horizon=PRIMARY_HORIZON
            )
            for m in H3_METHODS
        }
    selection_wall = time.perf_counter() - started
    print(f"  selected hyperparameters in {selection_wall:.1f}s", flush=True)

    # Test tiers.
    started = time.perf_counter()
    tiers = {}
    for tier in cfg.open_loop_tiers:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
        probes = collect_trajectories(env, max(cfg.ks), cfg.adapt_steps, env.default_probe, make_generator(tag, "probe"))
        tier_record = {
            "mechanisms": env.mechanisms,
            "persistence": _errors_record(open_loop_errors(PersistencePredictor(), holdout, scale, low, high), env.mechanisms),
            "k": {},
        }
        task_seed = (tag, "task")
        run_closed = tier in cfg.closed_loop_tiers
        if run_closed and cfg.oracle:
            oracle = OraclePredictor(env)
            out = closed_loop(env, oracle, env.sample_task(make_generator(task_seed)), cem_for(env, cfg), make_generator(tag, "cem"), cfg.episode_override)
            tier_record["oracle_closed_loop"] = {
                "success_rate": out["success"].float().mean().item(),
                "mean_cost": out["mean_cost"].mean().item(),
                "wall_s": out["wall_s"],
                "counts": oracle.counts.as_dict(),
            }
        for k in cfg.ks:
            data = probes.take(k).flat()
            k_record = {}
            for cond in CONDITIONS:
                hp = selection[cond.label]["hp"]
                result = adapt(cond.method, backbones[cond.backbone], data, **hp)
                entry = {
                    "hp": hp,
                    "adapt_wall_s": result.wall_s,
                    "adapt_counts": result.counts.as_dict(),
                    "info": result.info,
                    "open_loop": _errors_record(open_loop_errors(result.predictor, holdout, scale, low, high), env.mechanisms),
                }
                if run_closed and k == PRIMARY_K:
                    result.predictor.counts = type(result.predictor.counts)()
                    out = closed_loop(
                        env, result.predictor, env.sample_task(make_generator(task_seed)), cem_for(env, cfg),
                        make_generator(tag, "cem"), cfg.episode_override,
                    )
                    entry["closed_loop"] = {
                        "success_rate": out["success"].float().mean().item(),
                        "failure_rate": 1.0 - out["success"].float().mean().item(),
                        "mean_cost": out["mean_cost"].mean().item(),
                        "per_env_success": out["success"].tolist(),
                        "per_env_cost": out["mean_cost"].tolist(),
                        "wall_s": out["wall_s"],
                        "counts": result.predictor.counts.as_dict(),
                    }
                k_record[cond.label] = entry
            tier_record["k"][str(k)] = k_record
        tiers[tier] = tier_record
        print(f"  tier {tier} done at {time.perf_counter() - started:.1f}s", flush=True)
    eval_wall = time.perf_counter() - started

    # H3: same methods, different adaptation data, open loop at K = 1.
    started = time.perf_counter()
    h3 = {}
    for tier in cfg.h3_tiers:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        holdout = collect_holdout(env, cfg.holdout_traj, cfg.holdout_steps, make_generator(tag, "holdout"))
        h3[tier] = {}
        for data_kind in H3_CONDITIONS:
            data = make_adapt_data(env, data_kind, PRIMARY_K, cfg.adapt_steps, tag)
            frozen_pred = adapt("frozen", backbones["latent"], data).predictor
            cell = {"collinearity": feature_collinearity(frozen_pred, data), "methods": {}}
            for m in H3_METHODS:
                hp = h3_selection[data_kind][m.label]["hp"]
                result = adapt(m.method, backbones[m.backbone], data, **hp)
                cell["methods"][m.label] = {
                    "hp": hp,
                    "adapt_wall_s": result.wall_s,
                    "adapt_counts": result.counts.as_dict(),
                    "open_loop": _errors_record(open_loop_errors(result.predictor, holdout, scale, low, high), env.mechanisms),
                }
            h3[tier][data_kind] = cell
    h3_wall = time.perf_counter() - started

    return {
        "train": train_record,
        "selection": selection,
        "h3_selection": h3_selection,
        "tiers": tiers,
        "h3": h3,
        "wall_s": {"train": train_wall, "selection": selection_wall, "eval": eval_wall, "h3": h3_wall},
    }


def _seed_metric(jobs: dict, tier: str, k: int, label: str, metric: str, kind: str = "open_loop") -> list[float]:
    """One number per completed seed, in seed order. Seeds are keys of jobs."""
    values = []
    for seed in sorted(jobs, key=int):
        entry = jobs[seed]["tiers"][tier]["k"][str(k)][label]
        if kind in entry:
            values.append(entry[kind][metric])
    return values


def _available_horizons(jobs: dict, tier: str, k: int, label: str) -> list[str]:
    any_seed = next(iter(jobs))
    return [m for m in jobs[any_seed]["tiers"][tier]["k"][str(k)][label]["open_loop"] if m.startswith("h")]


def summarize(jobs: dict, n_boot: int) -> dict:
    """Protocol verdicts per family, computed from whichever seeds have finished."""
    summary = {"means": {}}
    if not jobs:
        return summary
    any_seed = next(iter(jobs))
    tiers = list(jobs[any_seed]["tiers"].keys())
    ks = list(jobs[any_seed]["tiers"][tiers[0]]["k"].keys())

    def compare(tier, k, a, b):
        out = {}
        for h in _available_horizons(jobs, tier, k, a):
            out[h] = paired_comparison(_seed_metric(jobs, tier, k, a, h), _seed_metric(jobs, tier, k, b, h), n_boot=n_boot)
        fail_a = _seed_metric(jobs, tier, k, a, "failure_rate", "closed_loop")
        fail_b = _seed_metric(jobs, tier, k, b, "failure_rate", "closed_loop")
        if fail_a and fail_b:
            out["failure_rate"] = paired_comparison(fail_a, fail_b, n_boot=n_boot)
            out["closed_loop_cost"] = paired_comparison(
                _seed_metric(jobs, tier, k, a, "mean_cost", "closed_loop"),
                _seed_metric(jobs, tier, k, b, "mean_cost", "closed_loop"),
                n_boot=n_boot,
            )
        return out

    for tier in tiers:
        summary["means"][tier] = {}
        for k in ks:
            summary["means"][tier][k] = {}
            for cond in CONDITIONS:
                row = {h: float(np.mean(_seed_metric(jobs, tier, int(k), cond.label, h))) for h in _available_horizons(jobs, tier, int(k), cond.label)}
                row["per_seed_primary"] = _seed_metric(jobs, tier, int(k), cond.label, f"h{PRIMARY_HORIZON}")
                success = _seed_metric(jobs, tier, int(k), cond.label, "success_rate", "closed_loop")
                if success:
                    row["success_rate"] = float(np.mean(success))
                summary["means"][tier][k][cond.label] = row

    primary = f"h{PRIMARY_HORIZON}"
    k = PRIMARY_K
    # H1: a = latent_opt, b = rls.
    h1 = {"primary_horizon": PRIMARY_HORIZON, "k": k, "comparisons": {}}
    for tier in ("interp", "oof"):
        if tier in tiers:
            h1["comparisons"][tier] = {
                "latent_opt_vs_rls": compare(tier, k, "latent_opt@latent", "rls@latent"),
                "latent_opt_vs_rls_nolatent": compare(tier, k, "latent_opt@latent", "rls@nolatent"),
                "latent_opt_vs_control": compare(tier, k, "latent_opt@latent", "frozen@control"),
            }
    if {"interp", "oof"} <= set(tiers):
        v_in = h1["comparisons"]["interp"]["latent_opt_vs_rls"][primary]["verdict"]
        v_out = h1["comparisons"]["oof"]["latent_opt_vs_rls"][primary]["verdict"]
        if v_in == "a_wins" and v_out == "b_wins":
            h1["outcome"] = "supported"
        elif v_in == v_out and v_in in ("a_wins", "b_wins"):
            h1["outcome"] = "killed_one_method_wins_both"
        else:
            h1["outcome"] = "not_supported_no_crossover"
        h1["verdicts"] = {"interp": v_in, "oof": v_out}
    summary["H1"] = h1

    # H2: a = hybrid, b = each single method, then rule from HYPOTHESES.md.
    h2 = {"primary_horizon": PRIMARY_HORIZON, "k": k, "tiers": {}}
    worse_anywhere, beats_both_somewhere = False, False
    for tier in ("interp", "extrap", "oof"):
        if tier not in tiers:
            continue
        vs_latent = compare(tier, k, "hybrid@latent", "latent_opt@latent")
        vs_rls = compare(tier, k, "hybrid@latent", "rls@latent")
        latent_mean = vs_latent[primary]["mean_b"]
        rls_mean = vs_rls[primary]["mean_b"]
        versus_better = vs_latent if latent_mean <= rls_mean else vs_rls
        if versus_better[primary]["verdict"] == "b_wins":
            worse_anywhere = True
        if vs_latent[primary]["verdict"] == "a_wins" and vs_rls[primary]["verdict"] == "a_wins":
            beats_both_somewhere = True
        h2["tiers"][tier] = {
            "vs_latent_opt": vs_latent,
            "vs_rls": vs_rls,
            "better_single": "latent_opt" if latent_mean <= rls_mean else "rls",
        }
    h2["outcome"] = "killed" if worse_anywhere else ("supported" if beats_both_somewhere else "no_worse_but_no_gain")
    summary["H2"] = h2

    # H3: a = rls, b = gradient write. Gap = gradient error minus RLS error, per seed.
    h3 = {"primary_horizon": PRIMARY_HORIZON, "tiers": {}}
    seeds = sorted(jobs, key=int)
    for tier in jobs[any_seed]["h3"]:
        tier_out = {}
        gaps = {}
        for data_kind in H3_CONDITIONS:
            get = lambda m, metric: [jobs[s]["h3"][tier][data_kind]["methods"][m]["open_loop"][metric] for s in seeds]
            cond_out = {
                "collinearity": {
                    key: float(np.mean([jobs[s]["h3"][tier][data_kind]["collinearity"][key] for s in seeds]))
                    for key in ("log10_cond_reg", "log10_cond_nonzero", "participation_ratio", "top_eig_fraction")
                },
                "rls_vs_grad_1pass": {h: paired_comparison(get("rls", h), get("grad_write_1pass", h), n_boot=n_boot) for h in ("h1", "h10", "h30")},
                "rls_vs_grad_tuned": {h: paired_comparison(get("rls", h), get("grad_write_tuned", h), n_boot=n_boot) for h in ("h1", "h10", "h30")},
            }
            tier_out[data_kind] = cond_out
            rls_err = np.asarray(get("rls", primary))
            grad_err = np.asarray(get("grad_write_1pass", primary))
            gaps[data_kind] = (grad_err - rls_err) / np.maximum(rls_err, 1e-12)
        # Does the relative gap shrink when collinearity is reduced? a = reduced, b = smooth;
        # a_wins (reduced has the smaller gap) is what H3 predicts.
        tier_out["gap_vs_smooth"] = {
            kind: paired_comparison(gaps[kind], gaps["smooth"], threshold=0.0, n_boot=n_boot)
            for kind in ("random_action", "iid_state")
        }
        # Exact RLS is order invariant, so these two must agree to rounding. Anything else is a bug.
        tier_out["rls_order_invariance_max_abs_diff"] = max(
            abs(jobs[s]["h3"][tier]["smooth"]["methods"]["rls"]["open_loop"][primary] - jobs[s]["h3"][tier]["shuffled"]["methods"]["rls"]["open_loop"][primary])
            for s in seeds
        )
        h3["tiers"][tier] = tier_out
    summary["H3"] = h3
    return summary


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
    mode.add_argument("--smoke", action="store_true", help="tiny settings, checks the pipeline in under 2 minutes")
    mode.add_argument("--full", action="store_true", help="protocol settings, about 1 hour on 2 cores")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of training seeds")
    parser.add_argument("--families", nargs="+", default=None, choices=["pendulum", "puck"])
    parser.add_argument("--z-dim", type=int, default=None, help="override latent dimension")
    parser.add_argument("--no-oracle", action="store_true", help="skip the true-simulator planning ceiling")
    parser.add_argument("--max-minutes", type=float, default=None, help="start no new job after this long")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_a")
    args = parser.parse_args()

    cfg = smoke_config() if args.smoke else full_config()
    if args.seeds is not None:
        cfg.seeds = list(range(args.seeds))
    if args.families:
        cfg.families = args.families
    if args.z_dim is not None:
        cfg.z_dim = args.z_dim
    if args.no_oracle:
        cfg.oracle = False

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"thread_a_{cfg.mode}.json"
    started = time.perf_counter()
    output = {
        "config": asdict(cfg),
        "conditions": [asdict(c) for c in CONDITIONS],
        "h3_methods": [asdict(c) for c in H3_METHODS],
        "h3_data_conditions": list(H3_CONDITIONS),
        "registered_methods": sorted(METHODS),
        "selection_rule": (
            "per seed and condition: lowest mean over val_in and val_shift of error / frozen error at h10, K=1. "
            "Test tiers are never used; per-condition tables are under jobs.<family>.<seed>.selection."
        ),
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

    # Seeds run in the outer loop, so a stopped run still has both families at equal depth.
    for seed in cfg.seeds:
        for family in cfg.families:
            elapsed = (time.perf_counter() - started) / 60.0
            if args.max_minutes is not None and elapsed >= args.max_minutes:
                output["stopped_early"] = True
                save()
                print(f"stopped at {elapsed:.1f} minutes before seed {seed}, {family}", flush=True)
                return
            print(f"seed {seed}, {family} (elapsed {elapsed:.1f} min)", flush=True)
            output["jobs"][family][str(seed)] = run_job(family, seed, cfg)
            output["summary"] = {f: summarize(j, cfg.n_boot) for f, j in output["jobs"].items() if j}
            save()
    save()
    print(f"wrote {path} in {(time.perf_counter() - started) / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
