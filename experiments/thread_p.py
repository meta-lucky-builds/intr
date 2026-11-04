"""Thread P worker: one job in one process. The runner (intr.pixel.runner) schedules these; docs/GPU_RUN.md is how to start a run.

Job kinds, in the order the runner uses them:
    preflight   structural checks on one GPU: images, encoder, oracle
    bench       measured rates for the time estimate and for jobs per GPU
    pilot       train one world on the pilot seed and read Gate GP on validation environments only, for the ladder
    main        H1p, H12p, H4p, H3p on one seed and family. Writes weights and embedded data for later kinds
    h5          H5p probes
    h14         H14p error-fed core and its controls
    h16         appearance tier, descriptive. Runs last, and cannot affect main
    h17         latent sizes, descriptive. Runs last, and cannot affect main
    summarize   read every job file, apply the rules, write summary.json and summary.txt

Each job writes one JSON atomically on success, and nothing on failure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intr import ebayes  # noqa: E402,F401  (registers ebrls and hybrid_eb)
from intr.adapt import adapt  # noqa: E402
from intr.calibration import fit_sigma2, query_signals, well_specified_spearman, within_env_partial, within_env_spearman  # noqa: E402
from intr.envs import Trajectories, Transitions, derive_seed, make_env, make_generator  # noqa: E402
from intr.eval import HORIZONS, START_STEPS, CEMConfig, PersistencePredictor, compute_obs_scale  # noqa: E402
from intr.looped import SequenceData, saturation_reference  # noqa: E402
from intr.models import clip_obs  # noqa: E402
from intr.pixel import rules  # noqa: E402
from intr.pixel.config import EMBED_DIMS, PROBE_TRAJECTORIES, Rung, ThreadPConfig, config_for  # noqa: E402
from intr.pixel.design import collect_dopt, coverage_measures, design_predictor  # noqa: E402
from intr.pixel.error_fed import Belief, ErrorFedConfig, ErrorFedPredictor, ErrorFedTrainConfig, other_env_donor, run_filter, train_error_fed  # noqa: E402
from intr.pixel.planning import EvalContext, Windows, closed_loop_model, closed_loop_oracle, decoded_errors, errors_record, make_windows, record_of  # noqa: E402
from intr.pixel.render import appearance_palette, offscreen_fraction, render  # noqa: E402
from intr.pixel.scheduler import atomic_write_json  # noqa: E402
from intr.pixel.sim import collect_iid_states_with_burn, collect_mixed_states, collect_states, env_on, rollout_policy, state_sha  # noqa: E402
from intr.pixel.world import (  # noqa: E402
    ConvEncoder,
    Observed,
    StateProbe,
    backbone_from_dict,
    backbone_to_dict,
    embed_frames,
    embedding_bounds,
    embedding_stats,
    exact_train_losses,
    fit_probe,
    model_config,
    observe,
    on_device,
    probe_r2,
    train_config,
    train_core,
    train_jepa,
    transition_data,
)
from intr.train import Backbone, latent_r2, train_latent  # noqa: E402

PRIMARY_H = 10
SCOPE_LIMIT = "Own renderer and small models. Not a published benchmark. No claim about large models or real robots."
FIXED_CHOICES = {
    "encoder": "Shared frozen EMA encoder per seed and family, so latent and nolatent losses share units.",
    "cost": "Models pay env.cost on the probe's decoding; the oracle pays it on true observations. Same goal, same success test.",
    "burn_in": "stack - 1 zero-action steps in the true simulator before every trajectory and episode, oracle included.",
    "ladder": "Smallest rung passing Gate GP on val_in of the pilot seed. Thresholds are never changed.",
    "hybrid": "hybrid and hybrid_eb use latent_opt's validated lr_scale. hybrid's lambda is validated, hybrid_eb's is evidence-chosen.",
    "h14_closed_loop": "The error-fed belief stays where adaptation left it, as a latent does, so it gets the same data as other methods.",
    "h4_lambda": "H4p uses the better RLS and its h10-selected lambda, as thread_b's rollout rule.",
}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Tensor):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def transitions_sha(data: Transitions) -> str:
    digest = hashlib.sha256()
    for x in (data.obs, data.act, data.next_obs):
        digest.update(x.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_torch(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def label_kind(label: str) -> str:
    return label.split("@")[1]


# ---------------------------------------------------------------- the world: encoder, probe, core


@dataclass
class World:
    family: str
    seed: int
    rung: Rung
    stack: int
    embed_dim: int
    act_dim: int
    encoder: ConvEncoder
    probe: StateProbe
    backbones: dict[str, Backbone]  # CPU
    dev_models: dict
    ctx: EvalContext
    device: torch.device


def make_context(probe, scale: Tensor, family: str, emb_low: Tensor, emb_high: Tensor, device) -> EvalContext:
    env = make_env(family, 1, "train", "bounds")
    return EvalContext(probe, scale.to(device), env.obs_low.to(device), env.obs_high.to(device), emb_low.to(device), emb_high.to(device))


def assemble_world(family, seed, rung, cfg, encoder, probe, backbones, emb_low, emb_high, act_dim, device) -> World:
    dev_models = {name: copy.deepcopy(b.model).to(device) for name, b in backbones.items()}
    ctx = make_context(probe, compute_obs_scale(family), family, emb_low, emb_high, device)
    return World(family, seed, rung, cfg.stack, EMBED_DIMS[family], act_dim, encoder, probe, backbones, dev_models, ctx, device)


def build_world(family: str, seed: int, cfg: ThreadPConfig, rung: Rung, device) -> tuple[World, Observed, dict]:
    """Train encoder, probe and core. Every number comes from training environments. No validation or test tier exists yet."""
    started = time.perf_counter()
    burn = cfg.stack - 1
    env = make_env(family, cfg.train_envs, "train", derive_seed("train_envs", seed))
    env_dev = env_on(env, device)
    traj = collect_mixed_states(env, env_dev, cfg.train_traj, cfg.train_steps, env.data_policies, make_generator("data", family, seed), burn, device)
    frames = render(family, traj.states, rung.size)
    embed_dim = EMBED_DIMS[family]
    encoder, jepa_stats = train_jepa(frames, traj.act, cfg.stack, embed_dim, rung, cfg.jepa, seed, device)
    emb = embed_frames(encoder, frames, cfg.stack)
    del frames
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  JEPA done at {time.perf_counter() - started:.0f}s: {jepa_stats['log'][-1]}", flush=True)
    observed = Observed(emb, traj.obs.cpu(), traj.act.cpu())
    if not torch.isfinite(emb).all():
        raise RuntimeError("training embeddings are not finite")
    scale = compute_obs_scale(family)
    probe, probe_stats = fit_probe(emb.reshape(-1, embed_dim), observed.obs.reshape(-1, env.obs_dim), "train", scale, cfg.probe_hidden, cfg.probe_steps, cfg.probe_batch, seed, device)
    data = transition_data(observed)
    core_started = time.perf_counter()
    backbones = train_core(data, embed_dim, env.act_dim, cfg.z_dim, cfg.core[family], seed)
    core_wall = time.perf_counter() - core_started
    print(f"  core done in {core_wall:.0f}s", flush=True)
    emb_low, emb_high = embedding_bounds(emb)
    world = assemble_world(family, seed, rung, cfg, encoder, probe, backbones, emb_low, emb_high, env.act_dim, device)
    record = {
        "jepa": jepa_stats,
        "probe": probe_stats,
        "core_wall_s": core_wall,
        "exact_train_loss": exact_train_losses(backbones, data),
        "latent_r2": dict(zip(env.hidden_names, latent_r2(backbones["latent"].z_table, env.hidden_matrix()))),
        "embedding": embedding_stats(emb),
        "offscreen_fraction": offscreen_fraction(family, traj.states),
        "core_settings": asdict(cfg.core[family]),
        "wall_s": time.perf_counter() - started,
    }
    return world, observed, record


def world_to_dict(world: World, emb_low: Tensor, emb_high: Tensor) -> dict:
    return {
        "family": world.family,
        "seed": world.seed,
        "rung": asdict(world.rung),
        "stack": world.stack,
        "embed_dim": world.embed_dim,
        "act_dim": world.act_dim,
        "encoder": {k: v.cpu() for k, v in world.encoder.state_dict().items()},
        "probe": {k: v.cpu() for k, v in world.probe.state_dict().items()},
        "probe_shape": (world.probe.net[0].in_features, world.probe.net[0].out_features, world.probe.net[-1].out_features),
        "backbones": {name: backbone_to_dict(b) for name, b in world.backbones.items()},
        "emb_low": emb_low.cpu(),
        "emb_high": emb_high.cpu(),
    }


def world_from_dict(d: dict, cfg: ThreadPConfig, device) -> World:
    rung = Rung(**d["rung"])
    encoder = ConvEncoder(3 * d["stack"], rung.size, d["embed_dim"], rung.width)
    encoder.load_state_dict(d["encoder"])
    encoder = encoder.to(device).eval().requires_grad_(False)
    embed, hidden, obs_dim = d["probe_shape"]
    probe = StateProbe(embed, obs_dim, hidden)
    probe.load_state_dict(d["probe"])
    probe = probe.to(device).eval().requires_grad_(False)
    backbones = {name: backbone_from_dict(b) for name, b in d["backbones"].items()}
    return assemble_world(d["family"], d["seed"], rung, cfg, encoder, probe, backbones, d["emb_low"], d["emb_high"], d["act_dim"], device)


# ---------------------------------------------------------------- data for a tier


@dataclass
class TierData:
    holdout: Observed
    probes: Observed  # always PROBE_TRAJECTORIES, cut to K later, so K = 1 data does not depend on the list of K (Problem 12)
    windows: Windows
    mechanisms: list[str]
    state_sha: str
    offscreen: float

    def adaptation(self, k: int) -> Transitions:
        return self.probes.take(k).trajectories().flat()

    def holdout_flat(self) -> Transitions:
        return self.holdout.trajectories().flat()

    def holdout_next_obs(self) -> Tensor:
        return Trajectories(self.holdout.obs, self.holdout.act).flat().next_obs


def collect_tier(env, family: str, tag, cfg: ThreadPConfig, device) -> tuple:
    """Same tag, same trajectories: the states of a tier do not depend on which method or which palette reads them."""
    env_dev = env_on(env, device)
    burn = cfg.stack - 1
    holdout = collect_mixed_states(env, env_dev, cfg.holdout_traj, cfg.holdout_steps, (env.default_probe, "iid"), make_generator(tag, "holdout"), burn, device)
    probes = collect_states(env, env_dev, PROBE_TRAJECTORIES, cfg.adapt_steps, env.default_probe, make_generator(tag, "probe"), burn, device)
    return env_dev, holdout, probes


def embed_tier(world: World, holdout, probes, mechanisms: list[str], palette: Tensor | None = None) -> TierData:
    held = observe(holdout, world.family, world.rung.size, world.encoder, world.stack, palette)
    prob = observe(probes, world.family, world.rung.size, world.encoder, world.stack, palette)
    if not (torch.isfinite(held.emb).all() and torch.isfinite(prob.emb).all()):
        raise RuntimeError("embeddings of a tier are not finite")
    return TierData(held, prob, make_windows(held.emb, held.obs, held.act, world.device), mechanisms, state_sha(probes) + state_sha(holdout), offscreen_fraction(world.family, holdout.states))


def tier_to_dict(t: TierData) -> dict:
    return {"holdout": t.holdout.as_dict(), "probes": t.probes.as_dict(), "mechanisms": t.mechanisms, "state_sha": t.state_sha, "offscreen": t.offscreen}


def tier_from_dict(d: dict, device) -> TierData:
    holdout = Observed(**d["holdout"])
    return TierData(holdout, Observed(**d["probes"]), make_windows(holdout.emb, holdout.obs, holdout.act, device), d["mechanisms"], d["state_sha"], d["offscreen"])


def val_tier(world: World, tier: str, cfg: ThreadPConfig) -> tuple[TierData, object]:
    env = make_env(world.family, cfg.n_val_envs, tier, derive_seed("val_envs", world.seed))
    tag = ("val", world.family, world.seed, tier)
    env_dev, holdout, probes = collect_tier(env, world.family, tag, cfg, world.device)
    return embed_tier(world, holdout, probes, env.mechanisms), env


def eval_tier(world: World, tier: str, cfg: ThreadPConfig) -> tuple[TierData, object, tuple]:
    env = make_env(world.family, cfg.n_eval_envs, tier, derive_seed("test_envs", world.seed))
    tag = ("test", world.family, world.seed, tier)
    env_dev, holdout, probes = collect_tier(env, world.family, tag, cfg, world.device)
    return embed_tier(world, holdout, probes, env.mechanisms), env, tag


# ---------------------------------------------------------------- selection, validation only


def error_of(method: str, world: World, backbone_name: str, data: Transitions, windows: Windows, hp: dict, horizon: int = PRIMARY_H) -> float:
    result = adapt(method, world.backbones[backbone_name], data, **hp)
    predictor = on_device(result.predictor, world.dev_models[backbone_name], world.device)
    return decoded_errors(predictor, windows, world.ctx, (horizon,))[horizon].mean().item()


def edges_of(grid: list[dict], hp: dict) -> dict:
    """Which grid edge the choice sits at, per hyperparameter. An edge at the explicit no-write value is named as such."""
    out = {}
    for key in hp:
        values = sorted({row[key] for row in grid if isinstance(row.get(key), (int, float))})
        if len(values) < 2 or not isinstance(hp[key], (int, float)):
            continue
        if hp[key] == values[0]:
            out[key] = "low"
        elif hp[key] == values[-1]:
            out[key] = "no_write" if hp[key] >= ebayes.NO_WRITE_LAMBDA else "high"
    return out


def select(method: str, world: World, backbone_name: str, val: list[TierData], grid: list[dict]) -> dict:
    """Lowest mean over validation sets of (error / frozen error) at h10, as eval.select_hyperparameters, in decoded units."""
    data = [v.adaptation(1) for v in val]
    reference = [error_of("frozen", world, backbone_name, d, v.windows, {}) for d, v in zip(data, val)]
    table, best_hp, best_score = [], None, float("inf")
    for hp in grid:
        errors = [error_of(method, world, backbone_name, d, v.windows, hp) for d, v in zip(data, val)]
        score = float(np.mean([e / max(r, 1e-12) for e, r in zip(errors, reference)]))
        table.append({"hp": hp, "score": score, "errors": errors})
        if math.isfinite(score) and score < best_score:
            best_score, best_hp = score, hp
    if best_hp is None:
        best_hp = grid[0]
    chosen_errors = next(row["errors"] for row in table if row["hp"] == best_hp)
    return {"hp": best_hp, "score": best_score, "errors": chosen_errors, "table": table, "at_edge": edges_of(grid, best_hp)}


def conditions_after_latent(cfg: ThreadPConfig, lr_scale: float) -> list[tuple]:
    return [
        ("hybrid@latent", "hybrid", "latent", [{"lambda_": lam, "lr_scale": lr_scale} for lam in cfg.val_lambdas]),
        ("ebrls@latent", "ebrls", "latent", [{}]),
        ("ebrls@nolatent", "ebrls", "nolatent", [{}]),
        ("hybrid_eb@latent", "hybrid_eb", "latent", [{"lr_scale": lr_scale}]),
    ]


def select_all(world: World, val: list[TierData], cfg: ThreadPConfig) -> dict:
    lambdas = [{"lambda_": lam} for lam in cfg.val_lambdas]
    selection = {
        "frozen@latent": select("frozen", world, "latent", val, [{}]),
        "frozen@nolatent": select("frozen", world, "nolatent", val, [{}]),
        "frozen@control": select("frozen", world, "control", val, [{}]),
        "latent_opt@latent": select("latent_opt", world, "latent", val, [{"lr_scale": s} for s in cfg.lr_scales]),
        "rls@latent": select("rls", world, "latent", val, lambdas),
        "rls@nolatent": select("rls", world, "nolatent", val, lambdas),
    }
    lr_scale = selection["latent_opt@latent"]["hp"]["lr_scale"]
    for label, method, backbone, grid in conditions_after_latent(cfg, lr_scale):
        selection[label] = select(method, world, backbone, val, grid)
    return selection


METHOD_OF = {
    "frozen@latent": ("frozen", "latent"),
    "frozen@nolatent": ("frozen", "nolatent"),
    "frozen@control": ("frozen", "control"),
    "latent_opt@latent": ("latent_opt", "latent"),
    "rls@latent": ("rls", "latent"),
    "rls@nolatent": ("rls", "nolatent"),
    "hybrid@latent": ("hybrid", "latent"),
    "ebrls@latent": ("ebrls", "latent"),
    "ebrls@nolatent": ("ebrls", "nolatent"),
    "hybrid_eb@latent": ("hybrid_eb", "latent"),
}


def better_rls(selection: dict) -> str:
    """By raw validation h10, per seed. Scores are ratios to each backbone's own frozen model, so they do not compare backbones."""
    return min(("rls@latent", "rls@nolatent"), key=lambda label: float(np.mean(selection[label]["errors"])))


def compact_info(info: dict) -> dict:
    """Per-environment evidence lists are large and decide nothing. Selected values per tier stay, as H12 asked."""
    out = {k: v for k, v in info.items() if k != "eb"}
    eb = info.get("eb")
    if eb is not None:
        out["eb_summary"] = {
            "median_edf": float(np.median(eb["edf"])),
            "median_log10_lambda": float(np.median(eb["log10_lambda"])),
            "at_lower_edge": float(np.mean(eb["at_lower_edge"])),
            "at_upper_edge": float(np.mean(eb["at_upper_edge"])),
            "no_write": float(np.mean(eb["no_write"])),
            "median_width_2nats": float(np.median(eb["log10_lambda_width_2nats"])),
        }
    return out


# ---------------------------------------------------------------- H4p


@torch.no_grad()
def h4_signals(world: World, backbone_name: str, probe_cpu, scale: Tensor, low: Tensor, high: Tensor, data: Transitions, holdout: Transitions, next_obs: Tensor, lambda_: float):
    """Signals at held-out queries, and the physical residual after decoding. CPU."""
    backbone = world.backbones[backbone_name]
    predictor = adapt("rls", backbone, data, lambda_=lambda_).predictor
    sig = query_signals(predictor, holdout)
    next_emb = backbone.model.apply_delta(holdout.obs, predictor.predict_delta(holdout.obs, holdout.act))
    decoded = clip_obs(probe_cpu(next_emb), low, high)
    return sig, (decoded - next_obs) / scale


def run_h4(world: World, label: str, lambda_: float, val: list[TierData], tiers: dict, cfg: ThreadPConfig) -> dict:
    backbone_name = label_kind(label)
    probe_cpu = copy.deepcopy(world.probe).cpu()
    scale = compute_obs_scale(world.family)
    bounds = make_env(world.family, 1, "train", "bounds")
    low, high = bounds.obs_low, bounds.obs_high
    out: dict = {}
    calibrators = {}
    for k in cfg.ks:
        signals = [h4_signals(world, backbone_name, probe_cpu, scale, low, high, v.adaptation(k), v.holdout_flat(), v.holdout_next_obs(), lambda_) for v in val]
        q = torch.cat([s.q for s, _ in signals], dim=0)
        calibrators[k] = {
            "physical": fit_sigma2(torch.cat([r for _, r in signals], dim=0), q),
            "target": fit_sigma2(torch.cat([s.residual for s, _ in signals], dim=0), q),
        }
    for tier, (data_tier, tag) in tiers.items():
        out[tier] = {}
        for k in cfg.ks:
            sig, physical = h4_signals(world, backbone_name, probe_cpu, scale, low, high, data_tier.adaptation(k), data_tier.holdout_flat(), data_tier.holdout_next_obs(), lambda_)
            cell = {"lambda": lambda_, "backbone": backbone_name, "q_median": sig.q.flatten().median().item()}
            for space, err, sigma2 in (("physical", (physical**2).mean(-1), calibrators[k]["physical"]), ("target", sig.sq_err, calibrators[k]["target"])):
                gen = make_generator(tag, "h4", space, k)
                cell[space] = {
                    "within_P": float(np.nanmean(within_env_spearman(sig.q, err))),
                    "within_phi": float(np.nanmean(within_env_spearman(sig.phi_norm2, err))),
                    "partial_given_phi": float(np.nanmean(within_env_partial(sig.q, err, [sig.phi_norm2]))),
                    "reference": float(np.nanmean(well_specified_spearman(sig.q, sigma2, gen))),
                }
            out[tier][str(k)] = cell
    return out


# ---------------------------------------------------------------- main job


def cem_for(env, cfg: ThreadPConfig) -> CEMConfig:
    return CEMConfig(horizon=cfg.horizon_override or env.plan_horizon, n_candidates=cfg.n_candidates, n_elites=cfg.n_elites, n_iters=cfg.n_iters)


def episode_steps(env, cfg: ThreadPConfig) -> int:
    return cfg.episode_override or env.episode_steps


def run_closed(world: World, env, env_dev, predictor, tag, cfg: ThreadPConfig, oracle_mask: Tensor | None, palette: Tensor | None = None) -> dict:
    """Same tasks and planning noise for every arm of a tier."""
    out = closed_loop_model(
        env_dev, predictor, world.ctx, world.family, world.rung.size, world.stack, world.encoder,
        env.sample_task(make_generator((tag, "task"))), cem_for(env, cfg), make_generator(tag, "cem"), episode_steps(env, cfg), palette,
    )
    return record_of(out, oracle_mask)


def run_oracle(env, env_dev, tag, cfg: ThreadPConfig) -> dict:
    out = closed_loop_oracle(env_dev, env.sample_task(make_generator((tag, "task"))), cem_for(env, cfg), make_generator(tag, "cem"), episode_steps(env, cfg), cfg.stack - 1)
    return record_of(out)


def saturation_of(windows: Windows, ctx: EvalContext) -> dict:
    return saturation_reference(windows.true, HORIZONS, ctx.obs_scale, len(START_STEPS))


def collapse_check(world: World, val_in: TierData, cfg: ThreadPConfig) -> dict:
    r2 = probe_r2(world.probe, val_in.holdout.emb, val_in.holdout.obs)
    return {"r2": r2, "threshold": cfg.collapse_r2, "passes": bool(math.isfinite(r2["mean"]) and r2["mean"] >= cfg.collapse_r2), "embedding": embedding_stats(val_in.holdout.emb)}


def run_main(family: str, seed: int, cfg: ThreadPConfig, rung: Rung, device, run_dir: Path) -> dict:
    started = time.perf_counter()
    world, train_observed, train_record = build_world(family, seed, cfg, rung, device)
    val_in, _ = val_tier(world, "val_in", cfg)
    val_shift, _ = val_tier(world, "val_shift", cfg)
    val = [val_in, val_shift]
    collapse = collapse_check(world, val_in, cfg)
    print(f"  collapse check: {collapse['r2']['mean']:.3f}", flush=True)

    selection = select_all(world, val, cfg)
    rls_best = better_rls(selection)
    val_errors = {label: row["errors"] for label, row in selection.items()}
    best_kind = label_kind(rls_best)
    grad_grid = [{"lr": lr, "passes": p} for lr in cfg.grad_lrs for p in cfg.grad_passes]
    grad_selection = select("grad_write", world, best_kind, val, grad_grid)
    select_wall = time.perf_counter() - started
    print(f"  selected at {select_wall:.0f}s; better RLS {rls_best}", flush=True)

    emb_low, emb_high = world.ctx.emb_low.cpu(), world.ctx.emb_high.cpu()
    save_torch(world_to_dict(world, emb_low, emb_high), run_dir / "checkpoints" / f"{family}_s{seed}.pt")

    tiers_record, tier_cache, h4_tiers = {}, {}, {}
    same_data = True
    h3_tiers: dict = {}
    for tier in ("train", "interp", "extrap", "oof"):
        data_tier, env, tag = eval_tier(world, tier, cfg)
        tier_cache[tier] = tier_to_dict(data_tier)
        env_dev = env_on(env, device)
        record = {
            "mechanisms": env.mechanisms,
            "persistence": errors_record(decoded_errors(PersistencePredictor(), data_tier.windows, world.ctx), env.mechanisms),
            "saturation": saturation_of(data_tier.windows, world.ctx),
            "offscreen_fraction": data_tier.offscreen,
            "state_sha": data_tier.state_sha,
            "probe_r2": probe_r2(world.probe, data_tier.holdout.emb, data_tier.holdout.obs),
            "k": {},
        }
        run_cl = tier != "train"
        oracle_mask = None
        if run_cl:
            oracle = run_oracle(env, env_dev, tag, cfg)
            record["oracle_closed_loop"] = oracle
            oracle_mask = torch.tensor(oracle["per_env_success"], dtype=torch.bool)
            h4_tiers[tier] = (data_tier, tag)
        for k in cfg.ks:
            data = data_tier.adaptation(k)
            sha = transitions_sha(data)
            k_record = {}
            for label, (method, backbone_name) in METHOD_OF.items():
                hp = selection[label]["hp"]
                result = adapt(method, world.backbones[backbone_name], data, **hp)
                predictor = on_device(result.predictor, world.dev_models[backbone_name], device)
                same_data = same_data and transitions_sha(data) == sha
                entry = {
                    "hp": hp,
                    "data_sha256": sha,
                    "adapt_wall_s": result.wall_s,
                    "adapt_counts": result.counts.as_dict(),
                    "info": compact_info(result.info),
                    "open_loop": errors_record(decoded_errors(predictor, data_tier.windows, world.ctx), env.mechanisms),
                }
                frozen_label = f"frozen@{backbone_name}"
                if not label.startswith("frozen") and frozen_label in k_record:
                    entry["same_as_frozen"] = entry["open_loop"]["per_env"] == k_record[frozen_label]["open_loop"]["per_env"]
                if run_cl and k == 1:
                    entry["closed_loop"] = run_closed(world, env, env_dev, predictor, tag, cfg, oracle_mask)
                k_record[label] = entry
            record["k"][str(k)] = k_record
        if tier != "train":
            data = data_tier.adaptation(1)
            grad = adapt("grad_write", world.backbones[best_kind], data, **grad_selection["hp"])
            grad_predictor = on_device(grad.predictor, world.dev_models[best_kind], device)
            h3_tiers[tier] = {
                "rls": record["k"]["1"][rls_best]["open_loop"],
                "grad_tuned": errors_record(decoded_errors(grad_predictor, data_tier.windows, world.ctx), env.mechanisms),
                "rls_writes": record["k"]["1"][rls_best]["adapt_counts"]["writes"],
                "grad_writes": grad.counts.writes,
            }
        tiers_record[tier] = record
        print(f"  tier {tier} done at {time.perf_counter() - started:.0f}s", flush=True)

    h4 = run_h4(world, rls_best, selection[rls_best]["hp"]["lambda_"], val, h4_tiers, cfg)
    save_torch(
        {"val_in": tier_to_dict(val_in), "val_shift": tier_to_dict(val_shift), "tiers": tier_cache, "train": train_observed.as_dict(), "selection": {k: v["hp"] for k, v in selection.items()}, "rls_best": rls_best},
        run_dir / "cache" / f"{family}_s{seed}.pt",
    )
    return {
        "kind": "main",
        "family": family,
        "seed": seed,
        "rung": asdict(rung),
        "train": train_record,
        "collapse": collapse,
        "selection": selection,
        "val_errors": val_errors,
        "rls_best": rls_best,
        "tiers": tiers_record,
        "same_data_check": same_data,
        "h3": {"selection": grad_selection, "backbone": best_kind, "tiers": h3_tiers},
        "h4": h4,
        "fixed_choices": FIXED_CHOICES,
        "scope_limit": SCOPE_LIMIT,
        "wall_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------- pilot


def run_pilot(family: str, seed: int, cfg: ThreadPConfig, rung: Rung, device) -> dict:
    """Validation only. No test tier is created, so the ladder cannot see test data."""
    started = time.perf_counter()
    world, _, train_record = build_world(family, seed, cfg, rung, device)
    val_in, env = val_tier(world, "val_in", cfg)
    env_dev = env_on(env, device)
    tag = ("val", family, seed, "val_in")
    oracle = run_oracle(env, env_dev, tag, cfg)
    result = adapt("frozen", world.backbones["latent"], val_in.adaptation(1))
    frozen = run_closed(world, env, env_dev, on_device(result.predictor, world.dev_models["latent"], device), tag, cfg, None)
    collapse = collapse_check(world, val_in, cfg)
    losses = train_record["exact_train_loss"]
    gate = rules.gate_gp([frozen["success_rate"]], [oracle["success_rate"]], [losses["latent"]], [losses["nolatent"]], 1, 1)
    gate["checks"]["collapse"] = collapse["passes"]
    gate["pass"] = all(gate["checks"].values())
    return {
        "kind": "pilot",
        "family": family,
        "seed": seed,
        "rung": asdict(rung),
        "gate": gate,
        "collapse": collapse,
        "oracle": oracle["success_rate"],
        "frozen": frozen["success_rate"],
        "train": train_record,
        "wall_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------- jobs after main


def load_stage(family: str, seed: int, cfg: ThreadPConfig, device, run_dir: Path) -> tuple[World, dict]:
    world_dict = torch.load(run_dir / "checkpoints" / f"{family}_s{seed}.pt", map_location="cpu", weights_only=False)
    cache = torch.load(run_dir / "cache" / f"{family}_s{seed}.pt", map_location="cpu", weights_only=False)
    return world_from_dict(world_dict, cfg, device), cache


def main_job_record(run_dir: Path, family: str, seed: int) -> dict:
    return json.loads((run_dir / "jobs" / f"main_{family}_s{seed}.json").read_text(encoding="utf-8"))


def oracle_mask_from(main: dict, tier: str) -> Tensor | None:
    success = main["tiers"][tier].get("oracle_closed_loop", {}).get("per_env_success")
    return None if success is None else torch.tensor(success, dtype=torch.bool)


def run_h5(family: str, seed: int, cfg: ThreadPConfig, device, run_dir: Path) -> dict:
    started = time.perf_counter()
    world, cache = load_stage(family, seed, cfg, device, run_dir)
    main = main_job_record(run_dir, family, seed)
    rls_best = main["rls_best"]
    kind = label_kind(rls_best)
    backbone = world.backbones[kind]
    selected = cache["selection"][rls_best]["lambda_"]
    design_lambda = min(selected, cfg.design_lambda_max)
    burn = cfg.stack - 1
    low_cpu, high_cpu = world.ctx.emb_low.cpu(), world.ctx.emb_high.cpu()

    def collect_arm(arm: str, env, tag, n_traj: int) -> Observed:
        env_dev = env_on(env, device)
        if arm == "iid_state":
            traj = collect_iid_states_with_burn(env, env_dev, n_traj * cfg.adapt_steps, make_generator(tag, "iid_state"), burn, device)
            return observe(traj, family, world.rung.size, world.encoder, world.stack)
        s0 = env.reset(n_traj, make_generator(tag, "probe")).to(device)  # the first draw collect_states makes, so arms share starts
        if arm == "random_action":
            traj = rollout_policy(env_dev, s0, cfg.adapt_steps, "iid", make_generator(tag, "h5", "random_action"), burn)
            return observe(traj, family, world.rung.size, world.encoder, world.stack)
        if arm == "dopt":
            predictor = design_predictor(backbone, env.n_envs, design_lambda)
            observed, _ = collect_dopt(env_dev, s0, cfg.adapt_steps, predictor, family, world.rung.size, world.stack, world.encoder, cfg.dopt_candidates, cfg.dopt_horizon, make_generator(tag, "h5", "dopt"), low_cpu, high_cpu)
            return observed
        raise ValueError(arm)

    def cut(arm: str, observed: Observed, k: int) -> Transitions:
        return observed.take(k * cfg.adapt_steps if arm == "iid_state" else k).trajectories().flat()

    grid = [{"lambda_": lam} for lam in cfg.val_lambdas]
    arm_hp = {"default": cache["selection"][rls_best]}
    arm_selection: dict = {}
    for arm in ("random_action", "dopt", "iid_state"):
        val_data = []
        for tier_name in ("val_in", "val_shift"):
            env = make_env(family, cfg.n_val_envs, tier_name, derive_seed("val_envs", seed))
            held = tier_from_dict(cache[tier_name], device)
            val_data.append((cut(arm, collect_arm(arm, env, ("val", family, seed, tier_name), PROBE_TRAJECTORIES), 1), held.windows))
        reference = [error_of("frozen", world, kind, d, w, {}) for d, w in val_data]
        table, best_hp, best_score = [], grid[0], float("inf")
        for hp in grid:
            errors = [error_of("rls", world, kind, d, w, hp) for d, w in val_data]
            score = float(np.mean([e / max(r, 1e-12) for e, r in zip(errors, reference)]))
            table.append({"hp": hp, "score": score, "errors": errors})
            if math.isfinite(score) and score < best_score:
                best_score, best_hp = score, hp
        arm_hp[arm] = best_hp
        arm_selection[arm] = {"hp": best_hp, "score": best_score, "table": table, "at_edge": edges_of(grid, best_hp)}

    tiers: dict = {}
    for tier in rules.VERDICT_TIERS:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        env_dev = env_on(env, device)
        data_tier = tier_from_dict(cache["tiers"][tier], device)
        holdout_flat = data_tier.holdout_flat()
        arms = {"default": data_tier.probes}
        for arm in ("random_action", "dopt", "iid_state"):
            arms[arm] = collect_arm(arm, env, tag, PROBE_TRAJECTORIES)
        starts_match = bool(torch.equal(arms["default"].obs[:, :, 0], arms["random_action"].obs[:, :, 0]) and torch.equal(arms["default"].obs[:, :, 0], arms["dopt"].obs[:, :, 0]))
        mask = oracle_mask_from(main, tier)
        tiers[tier] = {"starts_match_default": starts_match}
        for k in cfg.ks:
            cells = {}
            for arm, observed in arms.items():
                data = cut(arm, observed, k)
                hp = arm_hp[arm]
                result = adapt("rls", backbone, data, **hp)
                predictor = on_device(result.predictor, world.dev_models[kind], device)
                cell = {
                    "hp": hp,
                    "n_transitions": data.obs.shape[1],
                    "data_sha256": transitions_sha(data),
                    "open_loop": errors_record(decoded_errors(predictor, data_tier.windows, world.ctx), env.mechanisms),
                    "coverage": coverage_measures(backbone, data, holdout_flat, design_lambda),
                }
                if k == 1 and arm != "iid_state":
                    cell["closed_loop"] = run_closed(world, env, env_dev, predictor, tag, cfg, mask)
                cells[arm] = cell
            tiers[tier][str(k)] = cells
    return {"kind": "h5", "family": family, "seed": seed, "backbone": rls_best, "design_lambda": design_lambda, "selection": arm_selection, "tiers": tiers, "wall_s": time.perf_counter() - started}


H14_ARMS = {"error_fed": (True, False), "no_error": (False, False), "shuffled": (True, True)}  # use_error, shuffle_errors


def train_h14_arm(arm: str, train: dict, world: World, cfg: ThreadPConfig, seed: int):
    use_error, shuffled = H14_ARMS[arm]
    ef = cfg.error_fed
    model_cfg = ErrorFedConfig(world.embed_dim, world.act_dim, ef.z_dim, ef.emb_dim, ef.hidden, ef.loops, ef.grad_loops, use_error)
    tcfg = ErrorFedTrainConfig(steps=ef.steps, batch_seqs=ef.batch_seqs, filter_max=ef.filter_max, horizon_max=ef.horizon_max)
    data = SequenceData(train["emb"], train["act"])
    return train_error_fed(data, model_cfg, tcfg, seed, world.ctx.emb_low.cpu(), world.ctx.emb_high.cpu(), shuffle_errors=shuffled)


@torch.no_grad()
def filter_probes(model, probes: Observed, k: int, donor: Tensor | None) -> tuple[Belief, dict]:
    """K trajectories in sequence, on the CPU. Same transitions as every other method."""
    belief = None
    trace = {}
    for b in range(k):
        belief, trace = run_filter(model, probes.emb[:, b], probes.act[:, b], donor, belief)
    return belief, trace


def run_h14(family: str, seed: int, cfg: ThreadPConfig, device, run_dir: Path) -> dict:
    started = time.perf_counter()
    world, cache = load_stage(family, seed, cfg, device, run_dir)
    main = main_job_record(run_dir, family, seed)
    horizon = max(HORIZONS)
    models = {}
    train_stats: dict = {"skipped": {}, "stats": {}}
    for arm in H14_ARMS:
        trained = train_h14_arm(arm, cache["train"], world, cfg, seed)
        models[arm] = trained.model
        train_stats["skipped"][arm] = trained.stats["skipped_nonfinite_updates"]
        train_stats["stats"][arm] = trained.stats
        print(f"  {arm} trained at {time.perf_counter() - started:.0f}s; loss {trained.stats['final_loss']}", flush=True)
    tiers: dict = {}
    for tier in rules.VERDICT_TIERS:
        env = make_env(family, cfg.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        env_dev = env_on(env, device)
        data_tier = tier_from_dict(cache["tiers"][tier], device)
        donor = other_env_donor(env.n_envs)
        mask = oracle_mask_from(main, tier)
        tiers[tier] = {}
        for k in cfg.ks:
            cells = {}
            for arm, model in models.items():
                model_dev = copy.deepcopy(model).to(device)
                belief, trace = filter_probes(model, data_tier.probes, k, donor if H14_ARMS[arm][1] else None)
                on_dev = Belief(belief.z.to(device), belief.pending.to(device))
                predictor = ErrorFedPredictor(model_dev, on_dev, horizon=horizon, online=False)
                open_loop = errors_record(decoded_errors(predictor, data_tier.windows, world.ctx), env.mechanisms)
                blind = ErrorFedPredictor(model_dev, None, horizon=horizon, online=False)
                blind_loop = decoded_errors(blind, data_tier.windows, world.ctx, (PRIMARY_H,))[PRIMARY_H].mean().item()
                own, fed = trace["own_error"], trace["fed_error"]
                cell = {
                    "open_loop": open_loop,
                    "h10_without_belief": blind_loop,  # manipulation check: a belief that changes nothing is inert
                    "own_error_rms": own.pow(2).mean().sqrt().item(),
                    "fed_differs_from_own": (fed - own).pow(2).mean().sqrt().item(),  # zero except on the shuffled control
                    "belief_std_across_envs": belief.z.std(dim=0).mean().item(),
                    "adapt_counts": {"forward_transitions": k * cfg.adapt_steps, "test_time_gradients": 0},
                    "data_sha256": transitions_sha(data_tier.adaptation(k)),
                }
                if k == 1:
                    planner = ErrorFedPredictor(model_dev, on_dev, horizon=cem_for(env, cfg).horizon, online=False)
                    cell["closed_loop"] = run_closed(world, env, env_dev, planner, tag, cfg, mask)
                cells[arm] = cell
            tiers[tier][str(k)] = cells
    return {"kind": "h14", "family": family, "seed": seed, "train": train_stats, "tiers": tiers, "wall_s": time.perf_counter() - started}


def run_h16(family: str, seed: int, cfg: ThreadPConfig, device, run_dir: Path) -> dict:
    """Same environments, tasks and states as interp. Only the palette differs."""
    started = time.perf_counter()
    world, cache = load_stage(family, seed, cfg, device, run_dir)
    main = main_job_record(run_dir, family, seed)
    env = make_env(family, cfg.n_eval_envs, "interp", derive_seed("test_envs", seed))
    tag = ("test", family, seed, "interp")
    env_dev, holdout, probes = collect_tier(env, family, tag, cfg, device)
    palette = appearance_palette(family, env.n_envs, make_generator(tag, "appearance"))
    data_tier = embed_tier(world, holdout, probes, env.mechanisms, palette)
    physics_unchanged = data_tier.state_sha == cache["tiers"]["interp"]["state_sha"]
    sample = holdout.states[:, 0, 0]
    pixel_distance = (render(family, sample, world.rung.size).float() - render(family, sample, world.rung.size, palette).float()).abs().mean().item() / 255.0
    mask = oracle_mask_from(main, "interp")
    rls_best = cache["rls_best"]
    cells = {}
    data = data_tier.adaptation(1)
    for label in ("frozen@latent", "latent_opt@latent", rls_best, "hybrid_eb@latent"):
        method, backbone_name = METHOD_OF[label]
        result = adapt(method, world.backbones[backbone_name], data, **cache["selection"][label])
        predictor = on_device(result.predictor, world.dev_models[backbone_name], device)
        cells[label] = {
            "open_loop": errors_record(decoded_errors(predictor, data_tier.windows, world.ctx), env.mechanisms),
            "closed_loop": run_closed(world, env, env_dev, predictor, tag, cfg, mask, palette),
            "data_sha256": transitions_sha(data),
        }
    appearance = {"physics_unchanged": physics_unchanged, "pixel_distance": pixel_distance, "k": {"1": cells}, "probe_r2": probe_r2(world.probe, data_tier.holdout.emb, data_tier.holdout.obs)}
    return {"kind": "h16", "family": family, "seed": seed, "appearance": appearance, "wall_s": time.perf_counter() - started}


def run_h17(family: str, seed: int, cfg: ThreadPConfig, device, run_dir: Path) -> dict:
    """Latent sizes on the stored encoder and data. latent_opt's lr_scale is selected on validation at each size."""
    started = time.perf_counter()
    world, cache = load_stage(family, seed, cfg, device, run_dir)
    train = Observed(**cache["train"])
    data = transition_data(train)
    val = [tier_from_dict(cache["val_in"], device), tier_from_dict(cache["val_shift"], device)]
    latents: dict = {}
    core = cfg.core[family]
    for z_dim in cfg.h17_latents:
        backbone = train_latent(data, model_config(world.embed_dim, world.act_dim, z_dim, core), train_config(core), seed)
        name = f"latent_z{z_dim}"
        world.backbones[name] = backbone
        world.dev_models[name] = copy.deepcopy(backbone.model).to(device)
        selection = select("latent_opt", world, name, val, [{"lr_scale": s} for s in cfg.lr_scales])
        tiers = {}
        for tier in rules.VERDICT_TIERS:
            data_tier = tier_from_dict(cache["tiers"][tier], device)
            adaptation = data_tier.adaptation(1)
            frozen = adapt("frozen", backbone, adaptation)
            adapted = adapt("latent_opt", backbone, adaptation, **selection["hp"])
            tiers[tier] = {
                "frozen": errors_record(decoded_errors(on_device(frozen.predictor, world.dev_models[name], device), data_tier.windows, world.ctx), data_tier.mechanisms),
                "latent_opt": errors_record(decoded_errors(on_device(adapted.predictor, world.dev_models[name], device), data_tier.windows, world.ctx), data_tier.mechanisms),
                "z_saturation": adapted.info.get("z_saturation"),
            }
        latents[str(z_dim)] = {"selection": selection, "tiers": tiers, "train_stats": backbone.stats}
        print(f"  z = {z_dim} done at {time.perf_counter() - started:.0f}s", flush=True)
    return {"kind": "h17", "family": family, "seed": seed, "latents": latents, "wall_s": time.perf_counter() - started}


# ---------------------------------------------------------------- preflight and bench


def run_preflight(cfg: ThreadPConfig, device) -> dict:
    """Structural checks that would make a paid run useless. Raises with a plain message on the first failure."""
    checks: dict = {"device": str(device)}
    if device.type == "cuda":
        checks["gpu"] = torch.cuda.get_device_name(0)
    for family in cfg.families:
        env = make_env(family, 8, "interp", "preflight")
        env_dev = env_on(env, device)
        state = env.reset(1, make_generator("preflight", family)).to(device)
        frames = render(family, state, cfg.rungs[0].size)
        if int(frames.max()) == int(frames.min()):
            raise RuntimeError(f"{family}: rendered images are constant")
        encoder = ConvEncoder(3 * cfg.stack, cfg.rungs[0].size, EMBED_DIMS[family], cfg.rungs[0].width).to(device)
        with torch.no_grad():
            out = encoder(frames[:, 0].float().repeat(1, cfg.stack, 1, 1) / 255.0)
        if not torch.isfinite(out).all():
            raise RuntimeError(f"{family}: encoder output is not finite")
        task = env.sample_task(make_generator("preflight", family, "task"))
        oracle = closed_loop_oracle(env_dev, task, cem_for(env, cfg), make_generator("preflight", family, "cem"), episode_steps(env, cfg), cfg.stack - 1)
        rate = oracle["success"].float().mean().item()
        checks[f"oracle_{family}"] = rate
        if cfg.enforce_gates and rate < rules.GATE_SUCCESS:
            raise RuntimeError(f"{family}: the oracle solves only {rate:.2f} of tasks on the true simulator, so its gate cannot pass. Do not spend on this run.")
    return checks


def run_bench(cfg: ThreadPConfig, seconds: float, device, jepa_only: bool) -> dict:
    """Measured rates at the middle rung, on random frames for JEPA, so no result of any kind is touched."""
    rung = cfg.rungs[min(1, len(cfg.rungs) - 1)]
    settings = cfg.jepa
    T = 8
    frames = torch.randint(0, 256, (16, 4, cfg.stack + T, 3, rung.size, rung.size), dtype=torch.uint8).to(device)
    act = (2.0 * torch.rand(16, 4, T, 2) - 1.0).to(device)
    n = 0
    began = time.perf_counter()
    from dataclasses import replace

    step = 20
    while time.perf_counter() - began < seconds:
        train_jepa(frames, act, cfg.stack, 32, rung, replace(settings, steps=step), n, device)
        n += 1
    jepa_rate = n * step * rung.jepa_scale / (time.perf_counter() - began)
    out = {"jepa_steps_per_s": jepa_rate, "device": str(device)}
    if device.type == "cuda":
        out["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
    if jepa_only:
        return out
    core = cfg.core["pendulum"]
    model = model_config(32, 2, 0, core)
    from intr.models import DynamicsModel

    cpu_model = DynamicsModel(model)
    optimizer = torch.optim.Adam(cpu_model.parameters(), lr=1e-3)
    began = time.perf_counter()
    count = 0
    while time.perf_counter() - began < min(seconds, 5.0):
        o, a = torch.randn(256, 32), torch.randn(256, 2)
        loss = (cpu_model(o, a) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        count += 1
    out["core_steps_per_s"] = count / (time.perf_counter() - began)
    dev_model = copy.deepcopy(cpu_model).to(device).requires_grad_(False)
    s, a = torch.randn(42, 64, 32, device=device), torch.randn(42, 64, 2, device=device)
    began = time.perf_counter()
    count = 0
    with torch.no_grad():
        while time.perf_counter() - began < min(seconds, 5.0):
            dev_model(s, a)
            count += 1
    out["model_calls_per_s"] = count / (time.perf_counter() - began)
    return out


# ---------------------------------------------------------------- summary


def load_kind(jobs_dir: Path, kind: str, family: str, fingerprint: str | None) -> dict:
    out = {}
    for path in sorted(jobs_dir.glob(f"{kind}_{family}_s*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("status") == "ok" and (fingerprint is None or data.get("fingerprint") == fingerprint):
            out[str(data["seed"])] = data
    return out


def run_summary(run_dir: Path, cfg: ThreadPConfig, fingerprint: str | None) -> dict:
    jobs_dir = run_dir / "jobs"
    families = {}
    for family in cfg.families:
        main = load_kind(jobs_dir, "main", family, fingerprint)
        extras = {kind: load_kind(jobs_dir, kind, family, fingerprint) for kind in ("h5", "h14", "h16", "h17")}
        families[family] = rules.summarize_family(family, main, extras, cfg.n_boot, cfg.min_seeds)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    summary = {
        "status": "ok",
        "fingerprint": fingerprint,
        "scope_limit": SCOPE_LIMIT,
        "fixed_choices": FIXED_CHOICES,
        "families": families,
        "failed_jobs": state.get("failed_jobs", {}),
        "skipped_families": state.get("skipped_families", []),
    }
    summary = jsonable(summary)
    atomic_write_json(run_dir / "summary.json", summary)
    text = rules.verdict_table(summary)
    (run_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return summary


# ---------------------------------------------------------------- entry point


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=["preflight", "bench", "pilot", "main", "h5", "h14", "h16", "h17", "summarize"])
    parser.add_argument("--mode", default="full", choices=["full", "smoke"])
    parser.add_argument("--family")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rung", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--jepa-only", action="store_true")
    args = parser.parse_args(argv)

    cfg = config_for(args.mode)
    if args.families:
        cfg.families = args.families
    torch.set_num_threads(int(os.environ.get("INTR_THREADS", cfg.threads)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False  # a chosen convolution algorithm must not depend on timing

    if args.kind == "summarize":
        run_summary(args.run_dir, cfg, args.fingerprint)
        return 0
    if args.kind == "preflight":
        record = {"kind": "preflight", "checks": run_preflight(cfg, device)}
    elif args.kind == "bench":
        record = {"kind": "bench", **run_bench(cfg, args.seconds, device, args.jepa_only)}
    elif args.kind == "pilot":
        record = run_pilot(args.family, args.seed, cfg, cfg.rungs[args.rung], device)
    elif args.kind == "main":
        record = run_main(args.family, args.seed, cfg, cfg.rungs[args.rung], device, args.run_dir)
    elif args.kind == "h5":
        record = run_h5(args.family, args.seed, cfg, device, args.run_dir)
    elif args.kind == "h14":
        record = run_h14(args.family, args.seed, cfg, device, args.run_dir)
    elif args.kind == "h16":
        record = run_h16(args.family, args.seed, cfg, device, args.run_dir)
    else:
        record = run_h17(args.family, args.seed, cfg, device, args.run_dir)
    record.update(status="ok", fingerprint=args.fingerprint)
    atomic_write_json(args.out, jsonable(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
