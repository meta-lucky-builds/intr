"""Thread B: H4 (calibration of RLS epistemic variance) and H10 (forgetting under regime switches).

    python experiments/thread_b.py --smoke     # target under 3 minutes, checks the pipeline; numbers mean nothing
    python experiments/thread_b.py --full      # estimated 30 to 60 minutes on 2 CPU cores, not yet measured

Writes results/thread_b/thread_b_<mode>.json, rewritten after every (seed, family) job, so a stopped run keeps
what it finished. --max-minutes stops starting new jobs after that long.

Backbones are trained by Thread A's functions on Thread A's config and seeds, so latent and nolatent match its
weights (test_backbones_match_thread_a), and H4 uses the same test environments and held-out trajectories.
The context and control backbones are skipped: neither has an RLS readout.

H4, per seed, family, backbone, tier and K:
    - After an RLS write, q = phi' P phi and the squared one-step error on held-out transitions.
    - PRIMARY ranking is within-environment (does q flag which queries are uncertain). Across-environment
      (does q flag which environments are poorly adapted) is reported beside it and decides nothing.
    - P is judged on what it adds beyond ||phi||^2, whose ranks equal those of the prior variance. "Adds" means
      paired_comparison on 1 - Spearman says P wins AND partial Spearman given ||phi||^2 has a CI above 0.
    - Lambda is chosen on validation environments. Primary rule ("rollout"): Thread A's selection, so q is the
      variance of the write Thread A evaluates. Sensitivity rule ("val_calibration"): the lambda with the best
      validation partial Spearman, the most charitable validation-only choice. It decides nothing by itself.
    - sigma^2 per K is fit on val_in and val_shift only, for three variance models: sigma^2 (1 + q),
      sigma^2 (1 + ||phi||^2 / lambda) and constant sigma^2. Coverage and NLL are reported for all three.

H10, per seed, family, tier and schedule, on the backbone chosen on validation:
    - Methods: episode_reset, persist, forgetting, surprise_reset. References outside any verdict: oracle_reset
      (told the true switch times) and frozen (no writes).
    - One hyperparameter choice per seed and family, on val_in and val_shift streams under the varied schedule,
      used for every schedule, since a deployed method does not know the schedule.

Choices HYPOTHESES.md leaves open, fixed here before any result exists:
    - Backbone: the better of rls@latent and rls@nolatent by raw validation error at horizon 10, per seed, as
      HYPOTHESES.md fixes for H1. Both are reported for H4. H10 runs on the chosen one only, for compute.
    - H4 thresholds apply to the mean over seeds of the per-environment Spearman, with K = 1 as the headline. A
      CI that straddles a threshold sets a sensitivity flag, which by protocol means adding seeds.
    - H10 "beats" uses RMSE, the protocol's 10% effect threshold and paired_comparison, cell by cell.
    - An "episode" in a stream is env.episode_steps. Switch intervals are chosen not to line up with it, except
      in the aligned schedule, where per-episode reset has oracle timing. Kill counts aligned like any other.

Departures from the protocol, stated so nobody assumes otherwise:
    - H4 and H10 are decided on one-step error, as they are specified, not on the protocol's horizon 10. H10 also
      reports a prequential horizon-10 error. Nothing here is decided on it.
    - No closed-loop control is measured in this thread.

Limits worth stating with any result:
    - The simulators are deterministic, so sigma^2 stands for model misfit, not noise. The iid Gaussian reading
      is an assumption under test, and non-Gaussian errors alone will miss nominal coverage.
    - P depends on features only, never on targets. It cannot see how wrong the features are in an environment,
      so across-environment ranking can reflect only how well a trajectory covers the queries.
    - One squared error is a variance times a chi-square draw, so a well-specified model has Spearman well below
      1. well_specified_reference records what it would score. A low Spearman with a low reference says little.
    - A numerical failure forces a reset, which can flatter forgetting, so a win in a cell with any failure does
      not count (cell "tainted").
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
sys.path.insert(0, str(ROOT / "experiments"))

from intr.adapt import METHODS, adapt  # noqa: E402
from intr.calibration import (  # noqa: E402
    COVERAGE_LEVELS,
    FeatureStream,
    QuerySignals,
    Schedule,
    StreamPolicy,
    SurpriseConfig,
    across_env_spearman,
    build_stream,
    coverage,
    fit_sigma2,
    gaussian_nll,
    prepare_stream,
    prequential_rollout_rmse,
    query_signals,
    rmse_over,
    rollout_starts,
    run_prequential,
    stream_metrics,
    well_specified_spearman,
    windup_summary,
    within_env_partial,
    within_env_spearman,
)
from intr.envs import Trajectories, Transitions, collect_trajectories, derive_seed, make_env, make_generator  # noqa: E402
from intr.eval import EFFECT_THRESHOLD, bootstrap_ci, collect_holdout, compute_obs_scale, paired_comparison, select_hyperparameters  # noqa: E402
from intr.models import Counts, ModelConfig  # noqa: E402
from intr.train import Backbone, TransitionData, latent_r2, train_latent, train_nolatent  # noqa: E402
from thread_a import PRIMARY_HORIZON, PRIMARY_K, ThreadAConfig, _jsonable, full_config, smoke_config, source_fingerprint, val_sets_for  # noqa: E402

KILL_SPEARMAN = 0.3
TARGET_SPEARMAN = 0.5
H4_BACKBONES = ("latent", "nolatent")
LAMBDA_SOURCES = ("rollout", "val_calibration")
VARIANCE_MODELS = ("rls", "prior_norm", "constant")
H10_METHODS = ("episode_reset", "persist", "forgetting", "surprise_reset")
H10_REFERENCES = ("oracle_reset", "frozen")
PRIMARY_POST_N = 20
ROLLOUT_KEY = f"rollout_h{PRIMARY_HORIZON}"
NO_SWITCH_KIND = "none"

DESIGN_NOTES = {
    "primary_ranking": (
        "H4 is decided on within-environment Spearman (does q flag which queries are uncertain). "
        "Across-environment Spearman (does q flag which environments are poorly adapted) is reported and decides nothing."
    ),
    "baseline_rule": "H4 is judged on whether P adds beyond ||phi||^2, whose ranks equal those of the prior variance.",
    "h10_primary": "H10 cells are decided on prequential one-step RMSE in the normalized target space.",
    "protocol_departures": [
        "H4 and H10 are decided on one-step error, as specified for them, not on rollout horizon 10.",
        "No closed-loop control is measured in this thread.",
    ],
    "selection": "Test tiers are never used for lambda, sigma^2, forgetting or surprise threshold.",
}


def default_schedules() -> list[Schedule]:
    """Intervals do not line up with episodes of 50 or 60 steps, except in aligned."""
    return [
        Schedule("none", NO_SWITCH_KIND),
        Schedule("fast", "fixed", interval=35),
        Schedule("medium", "fixed", interval=85),
        Schedule("slow", "fixed", interval=235),
        Schedule("varied", "varied", min_interval=20, max_interval=300),
        Schedule("return", "cycle", interval=120, cycle_len=2),
        Schedule("aligned", "aligned"),
    ]


@dataclass
class ThreadBConfig:
    mode: str
    a: ThreadAConfig  # training, environments and held-out data, exactly as Thread A has them
    h4_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    h10_tiers: list[str] = field(default_factory=lambda: ["interp", "extrap", "oof"])
    schedules: list[Schedule] = field(default_factory=default_schedules)
    val_schedule: Schedule = field(default_factory=lambda: Schedule("varied", "varied", min_interval=20, max_interval=300))
    stream_steps: int = 800
    n_streams: int = 21
    n_val_streams: int = 20
    forgetting_grid: list[float] = field(default_factory=lambda: [0.95, 0.98, 0.99, 0.995, 0.999])
    surprise_thresholds: list[float] = field(default_factory=lambda: [2.0, 3.0, 5.0, 10.0])
    post_switch_ns: list[int] = field(default_factory=lambda: [5, PRIMARY_POST_N])
    steady_after: int = 60
    eig_every: int = 50
    rollout_every: int = 10


def smoke_b_config() -> ThreadBConfig:
    return ThreadBConfig(
        mode="smoke",
        a=smoke_config(),
        stream_steps=150,
        n_streams=6,
        n_val_streams=4,
        forgetting_grid=[0.95, 0.99],
        surprise_thresholds=[3.0],
    )


def full_b_config() -> ThreadBConfig:
    return ThreadBConfig(mode="full", a=full_config())


def train_backbones(family: str, seed: int, a: ThreadAConfig) -> tuple[dict[str, Backbone], dict]:
    """thread_a.train_all without context and control. Each trainer seeds itself, so skipping them changes nothing."""
    gen = make_generator("data", family, seed)
    env = make_env(family, a.train_envs, "train", derive_seed("train_envs", seed))
    data = TransitionData.collect(env, a.train_traj, a.train_steps, gen)
    with_z = ModelConfig(env.obs_dim, env.act_dim, a.z_dim, a.hidden, a.n_hidden, a.feature_dim)
    no_z = ModelConfig(env.obs_dim, env.act_dim, 0, a.hidden, a.n_hidden, a.feature_dim)
    latent = train_latent(data, with_z, a.train, seed)
    nolatent = train_nolatent(data, no_z, a.train, seed)
    backbones = {"latent": latent, "nolatent": nolatent}
    record = {name: {"stats": b.stats, "final_log": b.log[-3:]} for name, b in backbones.items()}
    record["latent_r2"] = dict(zip(env.hidden_names, latent_r2(latent.z_table, env.hidden_matrix())))
    return backbones, record


# ---------------------------------------------------------------- H4


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _env_stat(per_env) -> dict:
    x = np.asarray(per_env, dtype=float)
    finite = x[np.isfinite(x)]
    return {
        "mean": _nanmean(x),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "frac_ge_target": float((finite >= TARGET_SPEARMAN).mean()) if finite.size else float("nan"),
        "n_undefined": int(x.size - finite.size),
        "per_env": x.tolist(),
    }


def _variance_extras(sig: QuerySignals, lambda_: float) -> dict:
    return {"rls": sig.q, "prior_norm": sig.phi_norm2 / lambda_, "constant": None}


@dataclass
class CalVal:
    """A validation tier with held-out transitions and probes for every K."""

    tier: str
    holdout: Transitions
    probes: Trajectories


def calibration_val_sets(family: str, seed: int, a: ThreadAConfig) -> list[CalVal]:
    """Same environments and held-out data as thread_a.val_sets_for, with probes that can be cut to any K."""
    sets = []
    for tier in ("val_in", "val_shift"):
        env = make_env(family, a.n_val_envs, tier, derive_seed("val_envs", seed))
        tag = ("val", family, seed, tier)
        holdout = collect_holdout(env, a.holdout_traj, a.holdout_steps, make_generator(tag, "holdout")).flat()
        probes = collect_trajectories(env, max(a.ks), a.adapt_steps, env.default_probe, make_generator(tag, "probe"))
        sets.append(CalVal(tier, holdout, probes))
    return sets


def select_lambda_for_calibration(backbone: Backbone, cal_val: list[CalVal]) -> dict:
    """Highest validation within-environment partial Spearman given ||phi||^2, at the primary K."""
    table, best_hp, best_score = [], None, -float("inf")
    for hp in METHODS["rls"].grid:
        partial, raw = [], []
        for cv in cal_val:
            predictor = adapt("rls", backbone, cv.probes.take(PRIMARY_K).flat(), **hp).predictor
            sig = query_signals(predictor, cv.holdout)
            partial.append(_nanmean(within_env_partial(sig.q, sig.sq_err, [sig.phi_norm2])))
            raw.append(_nanmean(within_env_spearman(sig.q, sig.sq_err)))
        score = _nanmean(partial)
        table.append({"hp": hp, "partial_spearman": score, "spearman": _nanmean(raw)})
        if math.isfinite(score) and score > best_score:
            best_hp, best_score = hp, score
    if best_hp is None:
        best_hp = METHODS["rls"].grid[0]  # nothing was defined; visible through the table
    return {"hp": best_hp, "score": best_score, "table": table}


def fit_calibrators(backbone: Backbone, cal_val: list[CalVal], lambda_: float, k: int) -> dict:
    """sigma^2 per variance model, pooled over validation tiers. Also its in-sample coverage, a check that
    separates non-Gaussian errors (validation already misses) from shift (only test tiers miss).
    """
    signals = []
    for cv in cal_val:
        predictor = adapt("rls", backbone, cv.probes.take(k).flat(), lambda_=lambda_).predictor
        signals.append(query_signals(predictor, cv.holdout))
    residual = torch.cat([s.residual for s in signals], dim=0)
    sigma2, val_coverage = {}, {}
    per_tier_extras = [_variance_extras(s, lambda_) for s in signals]
    for name in VARIANCE_MODELS:
        extra = None if per_tier_extras[0][name] is None else torch.cat([e[name] for e in per_tier_extras], dim=0)
        sigma2[name] = fit_sigma2(residual, extra)
        val_coverage[name] = {str(level): coverage(residual, extra, sigma2[name], level).mean().item() for level in COVERAGE_LEVELS}
    return {"sigma2": sigma2, "val_coverage": val_coverage}


def h4_cell(backbone: Backbone, data: Transitions, holdout: Transitions, lambda_: float, sigma2: dict, mechanisms: list[str], first_policy_n: int, gen) -> dict:
    """One tier and K. first_policy_n: held-out transitions at the front that came from the default probe."""
    result = adapt("rls", backbone, data, lambda_=lambda_)
    predictor = result.predictor
    sig = query_signals(predictor, holdout)
    err = sig.sq_err
    model = backbone.model

    signals = {"P": sig.q, "phi_norm": sig.phi_norm2, "input_norm": sig.input_norm2, "pred_norm": sig.pred_norm2}
    within_p = within_env_spearman(sig.q, err)
    within = {"P": _env_stat(within_p)}
    for name in ("phi_norm", "input_norm", "pred_norm"):
        within[name] = _env_stat(within_env_spearman(signals[name], err))
    partial = {
        "P|phi_norm": _env_stat(within_env_partial(sig.q, err, [sig.phi_norm2])),
        "P|phi_norm,input_norm": _env_stat(within_env_partial(sig.q, err, [sig.phi_norm2, sig.input_norm2])),
    }

    # What a stream of predict-then-write would have seen during adaptation; the only signal here using targets.
    with torch.no_grad():
        phi_data = predictor.features(data.obs, data.act)
        v_data = model.target(data.obs, data.next_obs)
    E = data.obs.shape[0]
    W0 = model.W.detach()[None].expand(E, -1, -1)
    adapt_err = run_prequential(phi_data, v_data, W0, StreamPolicy("adapt", lambda_)).err
    adapt_surprise = adapt_err[:, adapt_err.shape[1] // 2 :].mean(dim=1, keepdim=True)  # (E, 1)
    across = {name: across_env_spearman(s, err) for name, s in signals.items()}
    across["adapt_surprise"] = across_env_spearman(adapt_surprise, err)

    by_policy = {
        "default_probe": _nanmean(within_env_spearman(sig.q[:, :first_policy_n], err[:, :first_policy_n])),
        "iid": _nanmean(within_env_spearman(sig.q[:, first_policy_n:], err[:, first_policy_n:])),
    }
    names = sorted(set(mechanisms))
    by_mechanism = {}
    if len(names) > 1:  # descriptive only, per the known weakness in HYPOTHESES.md
        for name in names:
            by_mechanism[name] = _nanmean([within_p[i] for i, m in enumerate(mechanisms) if m == name])

    extras = _variance_extras(sig, lambda_)
    calibration = {}
    for name in VARIANCE_MODELS:
        calibration[name] = {
            "nll": gaussian_nll(sig.residual, extras[name], sigma2[name]).mean().item(),
            "coverage": {
                str(level): coverage(sig.residual, extras[name], sigma2[name], level).mean().item() for level in COVERAGE_LEVELS
            },
        }

    with torch.no_grad():
        frozen = adapt("frozen", backbone, data).predictor
        frozen_err = ((model.target(holdout.obs, holdout.next_obs) - frozen.predict_delta(holdout.obs, holdout.act)) ** 2).mean(dim=-1)
    q_flat = sig.q.flatten()
    return {
        "lambda": lambda_,
        "within": within,
        "partial": partial,
        "across": across,
        "by_policy": by_policy,
        "by_mechanism": by_mechanism,
        "well_specified_reference": _nanmean(well_specified_spearman(sig.q, sigma2["rls"], gen)),
        "calibration": calibration,
        "one_step_rmse": err.mean(dim=1).sqrt().mean().item(),
        "frozen_one_step_rmse": frozen_err.mean(dim=1).sqrt().mean().item(),
        "q_median": q_flat.median().item(),
        "q_max": q_flat.max().item(),
        "adapt_wall_s": result.wall_s,
        "adapt_counts": result.counts.as_dict(),
    }


def run_h4(family: str, seed: int, backbones: dict[str, Backbone], selection: dict, cfg: ThreadBConfig) -> dict:
    a = cfg.a
    cal_val = calibration_val_sets(family, seed, a)
    out = {}
    # Test data once per tier: both backbones and both lambda rules read the same transitions.
    test = {}
    for tier in cfg.h4_tiers:
        env = make_env(family, a.n_eval_envs, tier, derive_seed("test_envs", seed))
        tag = ("test", family, seed, tier)
        holdout = collect_holdout(env, a.holdout_traj, a.holdout_steps, make_generator(tag, "holdout")).flat()
        probes = collect_trajectories(env, max(a.ks), a.adapt_steps, env.default_probe, make_generator(tag, "probe"))
        test[tier] = (env, holdout, probes, tag)
    first_policy_n = ((a.holdout_traj + 1) // 2) * a.holdout_steps  # collect_mixed puts the default probe first

    for label in H4_BACKBONES:
        backbone = backbones[label]
        cal_selection = select_lambda_for_calibration(backbone, cal_val)
        lambdas = {
            "rollout": selection[label]["hp"]["lambda_"],
            "val_calibration": cal_selection["hp"]["lambda_"],
        }
        record = {"lambda": lambdas, "selection_rollout": selection[label], "selection_calibration": cal_selection, "sigma2": {}, "val_coverage": {}, "tiers": {}}
        for source, lambda_ in lambdas.items():
            record["sigma2"][source] = {}
            record["val_coverage"][source] = {}
            calibrators = {k: fit_calibrators(backbone, cal_val, lambda_, k) for k in a.ks}
            for k, fitted in calibrators.items():
                record["sigma2"][source][str(k)] = {m: s.tolist() for m, s in fitted["sigma2"].items()}
                record["val_coverage"][source][str(k)] = fitted["val_coverage"]
            for tier, (env, holdout, probes, tag) in test.items():
                tier_record = record["tiers"].setdefault(tier, {"mechanisms": env.mechanisms})
                tier_record[source] = {}
                for k in a.ks:
                    gen = make_generator(tag, "reference", label, source, k)
                    tier_record[source][str(k)] = h4_cell(
                        backbone, probes.take(k).flat(), holdout, lambda_, calibrators[k]["sigma2"], env.mechanisms, first_policy_n, gen
                    )
        out[label] = record
    return out


# ---------------------------------------------------------------- H10


def make_policy(method: str, hp: dict, stream) -> StreamPolicy:
    lambda_ = hp.get("lambda_", 1.0)
    if method == "persist":
        return StreamPolicy(method, lambda_)
    if method == "episode_reset":
        return StreamPolicy(method, lambda_, reset_at=stream.episode_start)
    if method == "oracle_reset":
        return StreamPolicy(method, lambda_, reset_at=stream.switch)
    if method == "forgetting":
        return StreamPolicy(method, lambda_, forgetting=hp["forgetting"])
    if method == "surprise_reset":
        return StreamPolicy(method, lambda_, surprise=SurpriseConfig(threshold=hp["threshold"]))
    if method == "frozen":
        return StreamPolicy(method, lambda_, write=False)
    raise ValueError(f"unknown stream method {method!r}")


def validation_streams(family: str, seed: int, backbone: Backbone, obs_low, obs_high, cfg: ThreadBConfig) -> list[FeatureStream]:
    streams = []
    for tier in ("val_in", "val_shift"):
        stream = build_stream(
            family, tier, cfg.val_schedule, cfg.n_val_streams, cfg.stream_steps,
            derive_seed("val_streams", seed), ("val_stream", family, seed, tier),
        )
        streams.append(prepare_stream(backbone, stream, obs_low, obs_high))
    return streams


def _score_entry(val: list[FeatureStream], method: str, hp: dict, reference: list[float]) -> dict:
    """Mean over validation tiers of prequential RMSE / frozen RMSE, as Thread A scores relative to frozen."""
    ratios, failed = [], []
    for fs, ref in zip(val, reference):
        result = run_prequential(fs.phi, fs.v, fs.W0, make_policy(method, hp, fs.stream), bounds=fs.bounds)
        ratios.append(rmse_over(result.err).mean().item() / max(ref, 1e-12))
        failed.append((result.failures > 0).double().mean().item())
    return {"hp": hp, "score": float(np.mean(ratios)), "failed_stream_fraction": float(np.mean(failed))}


def _pick(table: list[dict]) -> dict:
    """Lowest score among entries with no numerical failure. If every entry failed, that is flagged, not hidden."""
    finite = [row for row in table if math.isfinite(row["score"])]
    clean = [row for row in finite if row["failed_stream_fraction"] == 0.0]
    pool = clean or finite or table
    best = min(pool, key=lambda row: row["score"] if math.isfinite(row["score"]) else float("inf"))
    return {"hp": best["hp"], "score": best["score"], "all_entries_failed": not clean, "table": table}


def select_stream_methods(val: list[FeatureStream], cfg: ThreadBConfig) -> dict:
    """Choose on validation streams only. Forgetting tries lambdas chosen for persist and reset, since its
    regularizer decays as gamma^t and lambda matters only early. Surprise resets use the lambda chosen for reset.
    """
    reference = []
    for fs in val:
        frozen = run_prequential(fs.phi, fs.v, fs.W0, make_policy("frozen", {}, fs.stream), bounds=fs.bounds)
        reference.append(rmse_over(frozen.err).mean().item())
    lambdas = [g["lambda_"] for g in METHODS["rls"].grid]
    selection = {}
    for method in ("persist", "episode_reset"):
        selection[method] = _pick([_score_entry(val, method, {"lambda_": lam}, reference) for lam in lambdas])
    lam_persist = selection["persist"]["hp"]["lambda_"]
    lam_reset = selection["episode_reset"]["hp"]["lambda_"]
    forgetting_lambdas = sorted({lam_persist, lam_reset})
    selection["forgetting"] = _pick([
        _score_entry(val, "forgetting", {"lambda_": lam, "forgetting": gamma}, reference)
        for lam in forgetting_lambdas
        for gamma in cfg.forgetting_grid
    ])
    selection["surprise_reset"] = _pick([
        _score_entry(val, "surprise_reset", {"lambda_": lam_reset, "threshold": th}, reference)
        for th in cfg.surprise_thresholds
    ])
    selection["oracle_reset"] = {"hp": {"lambda_": lam_reset}, "note": "reference; uses the lambda chosen for episode_reset"}
    selection["frozen"] = {"hp": {}, "note": "reference; no writes"}
    return selection


def evaluate_method(fs: FeatureStream, method: str, hp: dict, starts: list[int], scale, low, high, cfg: ThreadBConfig) -> dict:
    started = time.perf_counter()
    result = run_prequential(
        fs.phi, fs.v, fs.W0, make_policy(method, hp, fs.stream), bounds=fs.bounds, eig_every=cfg.eig_every, snapshot_at=starts
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


def run_h10(family: str, seed: int, backbone: Backbone, scale, low, high, cfg: ThreadBConfig) -> dict:
    started = time.perf_counter()
    val = validation_streams(family, seed, backbone, low, high, cfg)
    selection = select_stream_methods(val, cfg)
    selection_wall = time.perf_counter() - started
    hps = {m: selection[m]["hp"] for m in H10_METHODS + H10_REFERENCES}
    episode_steps = make_env(family, 1, "train", "bounds").episode_steps
    starts = rollout_starts(cfg.stream_steps, episode_steps, PRIMARY_HORIZON, cfg.rollout_every)

    started = time.perf_counter()
    tiers = {}
    for tier in cfg.h10_tiers:
        tiers[tier] = {}
        for schedule in cfg.schedules:
            stream = build_stream(
                family, tier, schedule, cfg.n_streams, cfg.stream_steps,
                derive_seed("stream_envs", seed), ("stream", family, seed, tier, schedule.name),
            )
            fs = prepare_stream(backbone, stream, low, high)
            methods = {m: evaluate_method(fs, m, hps[m], starts, scale, low, high, cfg) for m in H10_METHODS + H10_REFERENCES}
            tiers[tier][schedule.name] = {
                "kind": schedule.kind,
                "n_switches_per_stream": stream.switch.double().sum(dim=1).mean().item(),
                "shared_feature_forward_samples": fs.forward_samples,
                "methods": methods,
            }
    return {
        "selection": selection,
        "tiers": tiers,
        "wall_s": {"selection": selection_wall, "eval": time.perf_counter() - started},
    }


# ---------------------------------------------------------------- one job


def _raw_val_error(selection: dict) -> float:
    """Raw validation error of the chosen entry. Selection scores are ratios to each backbone's own frozen model,
    so they cannot compare backbones.
    """
    for row in selection["table"]:
        if row["hp"] == selection["hp"]:
            return float(np.mean(row["errors"]))
    return float("inf")


def run_job(family: str, seed: int, cfg: ThreadBConfig) -> dict:
    a = cfg.a
    started = time.perf_counter()
    backbones, train_record = train_backbones(family, seed, a)
    train_wall = time.perf_counter() - started
    print(f"  trained in {train_wall:.1f}s", flush=True)
    scale = compute_obs_scale(family)
    probe_env = make_env(family, 1, "train", "bounds")
    low, high = probe_env.obs_low, probe_env.obs_high

    started = time.perf_counter()
    val_sets = val_sets_for(family, seed, a, "smooth")
    selection = {
        label: select_hyperparameters("rls", backbones[label], val_sets, scale, low, high, horizon=PRIMARY_HORIZON)
        for label in H4_BACKBONES
    }
    val_errors = {label: _raw_val_error(selection[label]) for label in H4_BACKBONES}
    chosen = min(H4_BACKBONES, key=lambda label: val_errors[label] if math.isfinite(val_errors[label]) else float("inf"))
    h4 = run_h4(family, seed, backbones, selection, cfg)
    h4_wall = time.perf_counter() - started
    print(f"  H4 done in {h4_wall:.1f}s; rls backbone {chosen}", flush=True)

    h10 = run_h10(family, seed, backbones[chosen], scale, low, high, cfg)
    print(f"  H10 done; selection {h10['wall_s']['selection']:.1f}s, eval {h10['wall_s']['eval']:.1f}s", flush=True)
    return {
        "train": train_record,
        "rls_backbone": chosen,
        "backbone_val_error": val_errors,
        "h4": h4,
        "h10": h10,
        "wall_s": {"train": train_wall, "h4": h4_wall},
    }


# ---------------------------------------------------------------- summaries and verdicts


def _is_finite(x) -> bool:
    return x is not None and math.isfinite(x)


def _stat(values, n_boot: int) -> dict:
    """Mean and bootstrap CI over training seeds, with the per-seed values next to them."""
    x = [float(v) for v in values if _is_finite(v)]
    if not x:
        return {"mean": None, "ci_low": None, "ci_high": None, "per_seed": [], "seed_std": None, "n_seeds": 0}
    low, high = bootstrap_ci(x, n_boot=n_boot) if len(x) >= 2 else (None, None)
    return {
        "mean": float(np.mean(x)),
        "ci_low": low,
        "ci_high": high,
        "per_seed": x,
        "seed_std": float(np.std(x, ddof=1)) if len(x) >= 2 else None,
        "n_seeds": len(x),
    }


def _paired(a, b, n_boot: int, threshold: float = EFFECT_THRESHOLD) -> dict:
    """paired_comparison over seeds where both values exist."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if _is_finite(x) and _is_finite(y)]
    if not pairs:
        return {"verdict": "undefined", "n_seeds": 0}
    return paired_comparison([p[0] for p in pairs], [p[1] for p in pairs], threshold=threshold, n_boot=n_boot)


def _straddles(stat: dict, threshold: float) -> bool:
    return stat.get("ci_low") is not None and stat["ci_low"] < threshold <= stat["ci_high"]


def _adds_beyond_norm(tier: dict) -> bool:
    partial = tier["partial_phi_norm"]
    return tier["vs_phi_norm"].get("verdict") == "a_wins" and partial.get("ci_low") is not None and partial["ci_low"] > 0


def h4_outcome(tiers: dict) -> dict:
    """The H4 rule from HYPOTHESES.md, with the baseline requirement.

    tiers: {tier: {"within_P": stat, "partial_phi_norm": stat, "vs_phi_norm": paired, "well_specified_reference": stat}}.
    Kill and target read the mean over seeds. A straddling CI only sets a flag: the protocol says add seeds.
    """
    if "interp" not in tiers or tiers["interp"]["within_P"]["mean"] is None:
        return {"outcome": "incomplete"}
    decisive = [t for t in ("interp", "extrap") if t in tiers and tiers[t]["within_P"]["mean"] is not None]
    interp_mean = tiers["interp"]["within_P"]["mean"]
    killed = interp_mean < KILL_SPEARMAN
    target_met = len(decisive) == 2 and all(tiers[t]["within_P"]["mean"] >= TARGET_SPEARMAN for t in decisive)
    adds = {t: _adds_beyond_norm(tiers[t]) for t in decisive}
    all_add = len(decisive) == 2 and all(adds.values())
    if killed:
        outcome = "killed"
    elif target_met and all_add:
        outcome = "supported"
    elif target_met:
        outcome = "target_met_but_not_beyond_norm"
    else:
        outcome = "below_target"
    sensitive = {t: _straddles(tiers[t]["within_P"], KILL_SPEARMAN) or _straddles(tiers[t]["within_P"], TARGET_SPEARMAN) for t in decisive}
    reachable = {}
    for t in decisive:
        reference = tiers[t]["well_specified_reference"]["mean"]
        reachable[t] = {
            "well_specified_reference": reference,
            "kill_threshold_reachable": None if reference is None else reference >= KILL_SPEARMAN,
            "target_reachable": None if reference is None else reference >= TARGET_SPEARMAN,
        }
    return {
        "outcome": outcome,
        "killed_h5_h6_end_as_stated": killed,
        "interp_mean": interp_mean,
        "adds_beyond_norm": adds,
        "ci_straddles_threshold_add_seeds": sensitive,
        "reachability_if_well_specified": reachable,
    }


def summarize_h4(jobs: dict, n_boot: int) -> dict:
    seeds = sorted(jobs, key=int)

    def cell(seed, label, tier, source, k):
        job = jobs[seed]
        name = job["rls_backbone"] if label == "selected" else label
        return job["h4"][name]["tiers"][tier][source][k]

    any_job = jobs[seeds[0]]
    first_label = H4_BACKBONES[0]
    tiers = list(any_job["h4"][first_label]["tiers"])
    ks = list(any_job["h4"][first_label]["tiers"][tiers[0]][LAMBDA_SOURCES[0]])
    out = {"primary_ranking": DESIGN_NOTES["primary_ranking"], "tables": {}, "outcomes": {}}
    for label in ("selected",) + H4_BACKBONES:
        out["tables"][label] = {}
        out["outcomes"][label] = {}
        for source in LAMBDA_SOURCES:
            out["tables"][label][source] = {}
            out["outcomes"][label][source] = {}
            for k in ks:
                table = {}
                for tier in tiers:
                    cells = [cell(s, label, tier, source, k) for s in seeds]
                    within = lambda name: [c["within"][name]["mean"] for c in cells]
                    within_p = within("P")
                    row = {
                        "within_P": _stat(within_p, n_boot),
                        "within_baselines": {n: _stat(within(n), n_boot) for n in ("phi_norm", "input_norm", "pred_norm")},
                        "partial_phi_norm": _stat([c["partial"]["P|phi_norm"]["mean"] for c in cells], n_boot),
                        "partial_phi_input_norm": _stat([c["partial"]["P|phi_norm,input_norm"]["mean"] for c in cells], n_boot),
                        # a = P, b = baseline, on 1 - Spearman: a_wins means P ranks better.
                        "vs_phi_norm": _paired([1.0 - v if _is_finite(v) else None for v in within_p], [1.0 - v if _is_finite(v) else None for v in within("phi_norm")], n_boot),
                        "across": {n: _stat([c["across"][n] for c in cells], n_boot) for n in cells[0]["across"]},
                        "well_specified_reference": _stat([c["well_specified_reference"] for c in cells], n_boot),
                        "by_policy": {n: _stat([c["by_policy"][n] for c in cells], n_boot) for n in ("default_probe", "iid")},
                        "one_step_rmse": _stat([c["one_step_rmse"] for c in cells], n_boot),
                        "frozen_one_step_rmse": _stat([c["frozen_one_step_rmse"] for c in cells], n_boot),
                        "calibration": {},
                    }
                    for model in VARIANCE_MODELS:
                        row["calibration"][model] = {
                            "nll": _stat([c["calibration"][model]["nll"] for c in cells], n_boot),
                            "coverage": {
                                str(level): _stat([c["calibration"][model]["coverage"][str(level)] for c in cells], n_boot)
                                for level in COVERAGE_LEVELS
                            },
                        }
                    # NLL can be negative, so only the sign and CI mean anything here, not a relative threshold.
                    row["nll_rls_vs_constant"] = _paired(
                        [c["calibration"]["rls"]["nll"] for c in cells], [c["calibration"]["constant"]["nll"] for c in cells], n_boot, threshold=0.0
                    )
                    row["nll_rls_vs_prior_norm"] = _paired(
                        [c["calibration"]["rls"]["nll"] for c in cells], [c["calibration"]["prior_norm"]["nll"] for c in cells], n_boot, threshold=0.0
                    )
                    table[tier] = row
                if "interp" in table and "oof" in table:
                    # a = interp, b = oof, on 1 - Spearman: a_wins means ranking weakens out of family, as H4 expects.
                    flip = lambda stat: [1.0 - v for v in stat["per_seed"]]
                    if table["interp"]["within_P"]["n_seeds"] == table["oof"]["within_P"]["n_seeds"]:
                        table["oof_weakening"] = _paired(flip(table["interp"]["within_P"]), flip(table["oof"]["within_P"]), n_boot, threshold=0.0)
                out["tables"][label][source][k] = table
                out["outcomes"][label][source][k] = h4_outcome({t: table[t] for t in tiers})
    return out


def h10_tier_outcome(cells: dict) -> dict:
    """The H10 rule from HYPOTHESES.md for one tier. cells: {schedule: {"vs_reset", "vs_persist", "vs_surprise":
    verdict with a = forgetting, "tainted": bool, "persist_worst": bool}} over switching schedules only.

    Kill: reset matches forgetting in every switching schedule, so forgetting beats reset in none. A win in a
    tainted cell (numerical failures) does not count.
    """
    if not cells:
        return {"outcome": "incomplete"}
    unusable = ("insufficient_seeds", "undefined")
    if any(c["vs_reset"] in unusable or c["vs_persist"] in unusable for c in cells.values()):
        return {"outcome": "insufficient_seeds"}
    beats_reset = [s for s, c in cells.items() if c["vs_reset"] == "a_wins" and not c["tainted"]]
    beats_both = [s for s in beats_reset if cells[s]["vs_persist"] == "a_wins"]
    if not beats_reset:
        # A win spoiled by numerics is not evidence that reset matches forgetting.
        spoiled = any(c["vs_reset"] == "a_wins" and c["tainted"] for c in cells.values())
        outcome = "inconclusive_numerical_failure" if spoiled else "killed"
    elif len(beats_both) == len(cells):
        outcome = "supported"
    else:
        outcome = "partial"
    return {
        "outcome": outcome,
        "n_switching_schedules": len(cells),
        "forgetting_beats_reset": beats_reset,
        "forgetting_beats_both": beats_both,
        "tainted_schedules": [s for s, c in cells.items() if c["tainted"]],
        "surprise_beats_forgetting": [s for s, c in cells.items() if c["vs_surprise"] == "b_wins"],
        "persist_worst_after_switch": [s for s, c in cells.items() if c["persist_worst"]],
    }


def summarize_h10(jobs: dict, n_boot: int) -> dict:
    seeds = sorted(jobs, key=int)
    any_tiers = jobs[seeds[0]]["h10"]["tiers"]
    primary_post = f"post_switch_{PRIMARY_POST_N}"
    out = {"tiers": {}, "outcomes": {}}
    for tier, schedules in any_tiers.items():
        out["tiers"][tier] = {}
        decisions = {}
        for name, schedule_cell in schedules.items():
            method_names = list(schedule_cell["methods"])
            metrics = [m for m in schedule_cell["methods"][method_names[0]] if m == "prequential" or m == "steady" or m == ROLLOUT_KEY or m.startswith("post_switch_") and m != "post_switch_curve"]

            def vals(method, metric):
                return [jobs[s]["h10"]["tiers"][tier][name]["methods"][method][metric]["mean"] for s in seeds]

            def compare(metric):
                return {
                    "forgetting_vs_reset": _paired(vals("forgetting", metric), vals("episode_reset", metric), n_boot),
                    "forgetting_vs_persist": _paired(vals("forgetting", metric), vals("persist", metric), n_boot),
                    "forgetting_vs_surprise": _paired(vals("forgetting", metric), vals("surprise_reset", metric), n_boot),
                    "persist_vs_reset": _paired(vals("persist", metric), vals("episode_reset", metric), n_boot),
                    "surprise_vs_reset": _paired(vals("surprise_reset", metric), vals("episode_reset", metric), n_boot),
                    "surprise_vs_persist": _paired(vals("surprise_reset", metric), vals("persist", metric), n_boot),
                }

            comparisons = {metric: compare(metric) for metric in metrics}
            windup = {}
            for method in method_names:
                records = [jobs[s]["h10"]["tiers"][tier][name]["methods"][method] for s in seeds]
                windup[method] = {
                    "failed_stream_fraction": _stat([r["windup"]["failed_stream_fraction"] for r in records], n_boot),
                    "n_failures": sum(r["windup"]["n_failures"] for r in records),
                    "log10_trace_growth": _stat([r["windup"]["log10_trace_growth"] for r in records], n_boot),
                    "trace_max": max(r["windup"]["trace_max"] for r in records),
                    "clipped_fraction": _stat([r["clipped_fraction"] for r in records], n_boot),
                    "resets_per_stream": _stat([r["resets_per_stream"] for r in records], n_boot),
                }
            tainted = windup["forgetting"]["n_failures"] > 0
            out["tiers"][tier][name] = {
                "means": {method: {metric: _stat(vals(method, metric), n_boot) for metric in metrics} for method in method_names},
                "comparisons": comparisons,
                "windup": windup,
                "forgetting_tainted_by_numerical_failure": tainted,
            }
            if schedule_cell["kind"] == NO_SWITCH_KIND:
                continue  # the stationarity control shows the cost of forgetting and reset; it decides nothing
            post = comparisons.get(primary_post, {})
            decisions[name] = {
                "vs_reset": comparisons["prequential"]["forgetting_vs_reset"]["verdict"],
                "vs_persist": comparisons["prequential"]["forgetting_vs_persist"]["verdict"],
                "vs_surprise": comparisons["prequential"]["forgetting_vs_surprise"]["verdict"],
                "tainted": tainted,
                "persist_worst": (
                    bool(post)
                    and post["forgetting_vs_persist"]["verdict"] == "a_wins"
                    and post["persist_vs_reset"]["verdict"] == "b_wins"
                    and post["surprise_vs_persist"]["verdict"] == "a_wins"
                ),
            }
        out["outcomes"][tier] = h10_tier_outcome(decisions)
    return out


def summarize(jobs: dict, n_boot: int) -> dict:
    """Verdicts for one family, from whichever seeds have finished."""
    if not jobs:
        return {}
    return {
        "H4": summarize_h4(jobs, n_boot),
        "H10": summarize_h10(jobs, n_boot),
        "rls_backbone_by_seed": {s: jobs[s]["rls_backbone"] for s in sorted(jobs, key=int)},
    }


def fingerprint() -> dict:
    """thread_a's hashes (src/intr and thread_a.py, which supplies training) plus this file."""
    out = source_fingerprint()
    me = Path(__file__).resolve()
    out[str(me.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(me.read_bytes()).hexdigest()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny settings, checks the pipeline")
    mode.add_argument("--full", action="store_true", help="protocol settings")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of training seeds")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="exact training seeds to run; overrides --seeds")
    parser.add_argument("--families", nargs="+", default=None, choices=["pendulum", "puck"])
    parser.add_argument("--max-minutes", type=float, default=None, help="start no new job after this long")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "thread_b")
    args = parser.parse_args()

    cfg = smoke_b_config() if args.smoke else full_b_config()
    if args.seeds is not None:
        cfg.a.seeds = list(range(args.seeds))
    if args.seed_list is not None:
        cfg.a.seeds = args.seed_list
    if args.families:
        cfg.a.families = args.families

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"thread_b_{cfg.mode}.json"
    started = time.perf_counter()
    output = {
        "config": asdict(cfg),
        "design_notes": DESIGN_NOTES,
        "thresholds": {"h4_kill": KILL_SPEARMAN, "h4_target": TARGET_SPEARMAN, "effect": EFFECT_THRESHOLD},
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "argv": sys.argv[1:],
            "source_sha256": fingerprint(),
        },
        "scope_limit": "Small state-based simulators on CPU; no claim about pixels, large models, or real robots.",
        "jobs": {f: {} for f in cfg.a.families},
        "summary": {},
        "stopped_early": False,
    }

    def save() -> None:
        output["metadata"]["elapsed_s"] = time.perf_counter() - started
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_jsonable(output), indent=1))
        tmp.replace(path)

    # Seeds run in the outer loop, so a stopped run still has both families at equal depth.
    for seed in cfg.a.seeds:
        for family in cfg.a.families:
            elapsed = (time.perf_counter() - started) / 60.0
            if args.max_minutes is not None and elapsed >= args.max_minutes:
                output["stopped_early"] = True
                save()
                print(f"stopped at {elapsed:.1f} minutes before seed {seed}, {family}", flush=True)
                return
            print(f"seed {seed}, {family} (elapsed {elapsed:.1f} min)", flush=True)
            output["jobs"][family][str(seed)] = run_job(family, seed, cfg)
            output["summary"] = {f: summarize(j, cfg.a.n_boot) for f, j in output["jobs"].items() if j}
            save()
    save()
    print(f"wrote {path} in {(time.perf_counter() - started) / 60.0:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
