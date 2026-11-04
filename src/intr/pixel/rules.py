"""Registered rules for Thread P, and the summary computed from job files. Rule functions are pure and tested on fabricated verdicts.

Stored outcomes come only from these functions, which follow docs/HYPOTHESES.md "Thread P". Every comparison uses eval.paired_comparison,
over training seeds, on the primary metric unless stated: decoded physical error at horizon 10, K = 1.

Readings the registration leaves open, fixed here before any result:
    any tier       H12p's and H3p's "any tier" means interp, extrap, oof, as in Thread F. Train is a sanity tier and appearance is H16.
    better RLS     the lower of rls@latent and rls@nolatent by raw validation h10, per seed, as HYPOTHESES.md requires for H1.
    better single  H12p: per seed, the lower validation error of latent_opt and the better RLS, on val_in for interp and val_shift for
                   the shifted tiers. The verdict against the lower test mean is stored as a sensitivity.
    H4p's space    The registration says one-step error, and also that embedding-space error is not used for verdicts. P's Bayesian reading
                   lives in the RLS target space. These conflict, so the verdict uses decoded physical one-step error, with its reference
                   computed on the same residuals, and the target-space reading is stored and flagged if it disagrees.
    H4p's "adds"   As fixed in thread_b's header: paired win on 1 - Spearman against ||phi||^2, and partial Spearman CI above 0.
    H5p            Per tier at K = 1. Coverage: paired log det of precision, dopt against random, threshold 0. Killed if coverage moved in some
                   tier and no such tier gained. Inconclusive if coverage moved in no tier.
    H14p           Controls and latent_opt on interp, RLS on oof. "Matches" means not worse by the effect rule.
    collapse       A seed whose probe R2 on val_in is under the configured number gets no verdict and leaves the paired set. It is listed.
    min seeds      Fewer than the protocol's 5 usable seeds gives insufficient_seeds.
"""

from __future__ import annotations

import math

import numpy as np

from ..eval import EFFECT_THRESHOLD, bootstrap_ci, paired_comparison

A, B, NDD = "a_wins", "b_wins", "no_detectable_difference"
USABLE = (A, B, NDD)
GATE_SUCCESS = 0.5
GATE_LATENT = 0.10
PRIMARY = "h10"
PRIMARY_K = "1"
VERDICT_TIERS = ("interp", "extrap", "oof")
INVALID = "invalid_family_gates_failed"


def is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def mean_of(values) -> float:
    x = [float(v) for v in values if is_finite(v)]
    return float(np.mean(x)) if x else float("nan")


def paired(a, b, n_boot: int, threshold: float = EFFECT_THRESHOLD) -> dict:
    """paired_comparison over seeds where both values exist. Seeds are aligned by the caller."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if is_finite(x) and is_finite(y)]
    if not pairs:
        return {"verdict": "undefined", "n_seeds": 0}
    return paired_comparison([p[0] for p in pairs], [p[1] for p in pairs], threshold=threshold, n_boot=n_boot)


def stat(values, n_boot: int) -> dict:
    x = [float(v) for v in values if is_finite(v)]
    if not x:
        return {"mean": None, "ci_low": None, "ci_high": None, "per_seed": [], "n_seeds": 0}
    low, high = bootstrap_ci(x, n_boot=n_boot) if len(x) >= 2 else (None, None)
    return {"mean": float(np.mean(x)), "ci_low": low, "ci_high": high, "per_seed": x, "n_seeds": len(x)}


# ---------------------------------------------------------------- rules


def _usable(*verdicts) -> bool:
    return all(v in USABLE for v in verdicts)


def h1p_rule(v_interp: str, v_oof: str) -> str:
    """a = latent_opt, b = the better RLS."""
    if not _usable(v_interp, v_oof):
        return "insufficient_seeds"
    if v_interp == A and v_oof == B:
        return "supported"
    if v_interp == v_oof and v_interp in (A, B):
        return "killed_one_method_wins_both"
    return "not_supported_no_crossover"


def h12p_rule(vs_better_single: dict) -> dict:
    """a = hybrid_eb. vs_better_single: {tier: verdict}. No worse in both tiers; killed if worse in any tier."""
    if "interp" not in vs_better_single or "oof" not in vs_better_single or not _usable(*vs_better_single.values()):
        return {"outcome": "insufficient_seeds", "kill_fired": False, "worse_in": []}
    worse_in = [t for t, v in vs_better_single.items() if v == B]
    return {"outcome": "killed" if worse_in else "supported", "kill_fired": bool(worse_in), "worse_in": worse_in}


def h4p_rule(within_p: float | None, reference: float | None, vs_phi_verdict: str, partial_ci_low: float | None) -> dict:
    """Interp. Killed if P ranks below half its well-specified reference, or adds nothing beyond ||phi||^2."""
    if not (is_finite(within_p) and is_finite(reference)):
        return {"outcome": "incomplete"}
    below_half = within_p < 0.5 * reference
    adds = vs_phi_verdict == A and is_finite(partial_ci_low) and partial_ci_low > 0
    return {
        "outcome": "killed" if (below_half or not adds) else "supported",
        "below_half_reference": bool(below_half),
        "adds_beyond_phi_norm": bool(adds),
        "reference_has_no_force": bool(reference < 0.2),  # Problem 11: a near-zero reference makes half of it an empty test
    }


def h14p_rule(vs_no_error: str, vs_shuffled: str, vs_latent_interp: str, vs_rls_oof: str) -> dict:
    """a = the error-fed core in every comparison. Kill: no better than the no-error control on interp."""
    verdicts = {"vs_no_error": vs_no_error, "vs_shuffled": vs_shuffled, "vs_latent_interp": vs_latent_interp, "vs_rls_oof": vs_rls_oof}
    if not _usable(*verdicts.values()):
        return {"outcome": "insufficient_seeds", "verdicts": verdicts}
    predictions = {
        "beats_no_error": vs_no_error == A,
        "beats_shuffled": vs_shuffled == A,
        "matches_latent_opt": vs_latent_interp in (A, NDD),
        "loses_to_rls_oof": vs_rls_oof == B,
    }
    if not predictions["beats_no_error"]:
        outcome = "killed"
    elif all(predictions.values()):
        outcome = "supported"
    else:
        outcome = "not_killed_some_predictions_failed"
    return {"outcome": outcome, "kill_fired": outcome == "killed", "predictions": predictions, "verdicts": verdicts}


def h5p_rule(tiers: dict) -> dict:
    """a = dopt. tiers: {tier: {vs_random, coverage_vs_random}}. a_wins on coverage means dopt raised log det over random."""
    if not tiers or not all(_usable(r["vs_random"], r["coverage_vs_random"]) for r in tiers.values()):
        return {"outcome": "insufficient_seeds"}
    moved = [t for t, r in tiers.items() if r["coverage_vs_random"] == A]
    gains = [t for t in moved if tiers[t]["vs_random"] == A]
    if not moved:
        outcome = "inconclusive_coverage_unmoved"
    elif not gains:
        outcome = "killed"
    elif len(gains) == len(tiers):
        outcome = "supported"
    else:
        outcome = "partial"
    return {
        "outcome": outcome,
        "kill_fired": outcome == "killed",
        "coverage_moved_in": moved,
        "gain_with_moved_coverage_in": gains,
        "gain_without_moved_coverage_in": [t for t, r in tiers.items() if r["vs_random"] == A and t not in moved],
        "worse_than_random_in": [t for t, r in tiers.items() if r["vs_random"] == B],
    }


def h3p_rule(tiers: dict) -> dict:
    """a = RLS, b = validation-tuned gradient writes. Kill: gradient wins any tier. tiers: {tier: verdict}."""
    if "interp" not in tiers or not _usable(*tiers.values()):
        return {"outcome": "insufficient_seeds"}
    wins = [t for t, v in tiers.items() if v == B]
    shifted = [t for t in tiers if t != "interp"]
    expected = tiers["interp"] == NDD and bool(shifted) and all(tiers[t] == A for t in shifted)
    if wins:
        outcome = "killed"
    elif expected:
        outcome = "supported"
    else:
        outcome = "not_killed_expectation_differs"
    return {"outcome": outcome, "kill_fired": bool(wins), "gradient_wins_in": wins, "matches_expectation": expected}


def gate_gp(frozen_success, oracle_success, loss_latent, loss_nolatent, n_usable: int, min_seeds: int) -> dict:
    """Gate GP over usable seeds, mean scores. Lists of one number per seed. A plain statement goes with a failure."""
    frozen, oracle = mean_of(frozen_success), mean_of(oracle_success)
    mean_latent, mean_nolatent = mean_of(loss_latent), mean_of(loss_nolatent)
    reduction = 1.0 - mean_latent / mean_nolatent if is_finite(mean_nolatent) and mean_nolatent > 0 else float("nan")
    checks = {
        "frozen_closed_loop_interp": is_finite(frozen) and frozen >= GATE_SUCCESS,
        "oracle_interp": is_finite(oracle) and oracle >= GATE_SUCCESS,
        "latent_lowers_prediction_loss": is_finite(reduction) and reduction >= GATE_LATENT,
        "enough_usable_seeds": n_usable >= min_seeds,
    }
    numbers = {"frozen_closed_loop_interp": frozen, "oracle_interp": oracle, "latent_reduction": reduction, "n_usable_seeds": n_usable}
    return {**numbers, "checks": checks, "pass": all(checks.values())}


def gated(rule_outcome: str, gate: dict) -> dict:
    return {"rule_outcome": rule_outcome, "gate_pass": gate["pass"], "gated_outcome": rule_outcome if gate["pass"] else INVALID}


def gate_statement(family: str, gate: dict) -> str:
    if gate["pass"]:
        return f"{family} passes Gate GP."
    failed = [name for name, ok in gate["checks"].items() if not ok]
    return (
        f"{family} FAILS Gate GP ({', '.join(failed)}; frozen {gate['frozen_closed_loop_interp']:.3f}, oracle {gate['oracle_interp']:.3f}, "
        f"latent reduction {gate['latent_reduction']:.3f}, usable seeds {gate['n_usable_seeds']}). Its results are not evidence for or against any method."
    )


# ---------------------------------------------------------------- reading job files


def usable_seeds(jobs: dict) -> tuple[list[str], list[str]]:
    seeds = sorted(jobs, key=int)
    good = [s for s in seeds if jobs[s]["collapse"]["passes"]]
    return good, [s for s in seeds if s not in good]


def resolve(job: dict, label: str, tier: str) -> str:
    if label == "rls_best":
        return job["rls_best"]
    if label == "frozen_best":
        return "frozen@" + job["rls_best"].split("@")[1]
    if label == "better_single":
        index = 0 if tier == "interp" else 1  # val_in for interp, val_shift for shifted tiers
        rls = job["rls_best"]
        latent_err, rls_err = job["val_errors"]["latent_opt@latent"][index], job["val_errors"][rls][index]
        latent_err = latent_err if is_finite(latent_err) else float("inf")
        rls_err = rls_err if is_finite(rls_err) else float("inf")
        return "latent_opt@latent" if latent_err <= rls_err else rls
    return label


def entry(job: dict, tier: str, k: str, label: str) -> dict:
    return job["tiers"][tier]["k"].get(k, {}).get(resolve(job, label, tier), {})


def series(jobs: dict, seeds: list[str], tier: str, k: str, label: str, metric: str = PRIMARY, kind: str = "open_loop") -> list:
    return [entry(jobs[s], tier, k, label).get(kind, {}).get(metric, float("nan")) for s in seeds]


def compare(jobs, seeds, tier, k, a, b, n_boot, metric=PRIMARY) -> dict:
    return paired(series(jobs, seeds, tier, k, a, metric), series(jobs, seeds, tier, k, b, metric), n_boot)


def enough(result: dict, min_seeds: int) -> str:
    """The verdict, unless too few seeds support one."""
    return result.get("verdict", "undefined") if result.get("n_seeds", 0) >= min_seeds else "insufficient_seeds"


def family_gate(jobs: dict, seeds: list[str], min_seeds: int) -> dict:
    frozen = series(jobs, seeds, "interp", PRIMARY_K, "frozen@latent", "success_rate", "closed_loop")
    oracle = [jobs[s]["tiers"]["interp"].get("oracle_closed_loop", {}).get("success_rate", float("nan")) for s in seeds]
    latent = [jobs[s]["train"]["exact_train_loss"]["latent"] for s in seeds]
    nolatent = [jobs[s]["train"]["exact_train_loss"]["nolatent"] for s in seeds]
    return gate_gp(frozen, oracle, latent, nolatent, len(seeds), min_seeds)


def summarize_h1p(jobs, seeds, k, gate, n_boot, min_seeds) -> dict:
    out = {"k": k, "comparisons": {}}
    for tier in VERDICT_TIERS:
        out["comparisons"][tier] = {
            "latent_opt_vs_better_rls": compare(jobs, seeds, tier, k, "latent_opt@latent", "rls_best", n_boot),
            "latent_opt_vs_frozen": compare(jobs, seeds, tier, k, "latent_opt@latent", "frozen@latent", n_boot),
            "latent_opt_vs_control": compare(jobs, seeds, tier, k, "latent_opt@latent", "frozen@control", n_boot),
            "better_rls_vs_control": compare(jobs, seeds, tier, k, "rls_best", "frozen@control", n_boot),
        }
    v_in = enough(out["comparisons"]["interp"]["latent_opt_vs_better_rls"], min_seeds)
    v_oof = enough(out["comparisons"]["oof"]["latent_opt_vs_better_rls"], min_seeds)
    out["verdicts"] = {"interp": v_in, "oof": v_oof}
    out.update(gated(h1p_rule(v_in, v_oof), gate))
    out["better_rls_by_seed"] = {s: jobs[s]["rls_best"] for s in seeds}
    return out


def summarize_h12p(jobs, seeds, k, gate, n_boot, min_seeds) -> dict:
    a = "hybrid_eb@latent"
    out = {"k": k, "comparisons": {}}
    by_val, by_test = {}, {}
    for tier in VERDICT_TIERS:
        latent_mean = mean_of(series(jobs, seeds, tier, k, "latent_opt@latent"))
        rls_mean = mean_of(series(jobs, seeds, tier, k, "rls_best"))
        test_single = "latent_opt@latent" if latent_mean <= rls_mean else "rls_best"
        row = {
            "vs_better_single": compare(jobs, seeds, tier, k, a, "better_single", n_boot),
            "vs_better_single_by_test_mean": compare(jobs, seeds, tier, k, a, test_single, n_boot),
            "vs_hybrid": compare(jobs, seeds, tier, k, a, "hybrid@latent", n_boot),
            "vs_frozen": compare(jobs, seeds, tier, k, a, "frozen@latent", n_boot),
            "ebrls_vs_better_rls": compare(jobs, seeds, tier, k, "ebrls@latent", "rls_best", n_boot),
            "better_single_by_seed": {s: resolve(jobs[s], "better_single", tier) for s in seeds},
            "median_edf": mean_of([entry(jobs[s], tier, k, a).get("info", {}).get("eb_summary", {}).get("median_edf") for s in seeds]),
            "median_log10_lambda": mean_of([entry(jobs[s], tier, k, a).get("info", {}).get("eb_summary", {}).get("median_log10_lambda") for s in seeds]),
        }
        out["comparisons"][tier] = row
        by_val[tier] = enough(row["vs_better_single"], min_seeds)
        by_test[tier] = enough(row["vs_better_single_by_test_mean"], min_seeds)
    rule = h12p_rule(by_val)
    sensitivity = h12p_rule(by_test)
    out["rule"] = rule
    out["outcome_with_test_mean_comparator"] = sensitivity["outcome"]
    out["comparator_sensitive"] = sensitivity["outcome"] != rule["outcome"]
    out.update(gated(rule["outcome"], gate))
    return out


def summarize_h4p(jobs, seeds, k, gate, n_boot) -> dict:
    out = {"k": k, "tiers": {}}
    for tier in VERDICT_TIERS:
        row = {}
        for space in ("physical", "target"):
            cell = lambda s: jobs[s]["h4"].get(tier, {}).get(k, {}).get(space, {})
            p = [cell(s).get("within_P", float("nan")) for s in seeds]
            phi = [cell(s).get("within_phi", float("nan")) for s in seeds]
            row[space] = {
                "within_P": stat(p, n_boot),
                "within_phi": stat(phi, n_boot),
                "partial_given_phi": stat([cell(s).get("partial_given_phi", float("nan")) for s in seeds], n_boot),
                "reference": stat([cell(s).get("reference", float("nan")) for s in seeds], n_boot),
                "vs_phi": paired([1.0 - x if is_finite(x) else float("nan") for x in p], [1.0 - x if is_finite(x) else float("nan") for x in phi], n_boot),
            }
        row["lambda_by_seed"] = {s: jobs[s]["h4"].get(tier, {}).get(k, {}).get("lambda") for s in seeds}
        out["tiers"][tier] = row

    def rule_for(space: str) -> dict:
        r = out["tiers"]["interp"][space]
        return h4p_rule(r["within_P"]["mean"], r["reference"]["mean"], r["vs_phi"].get("verdict", "undefined"), r["partial_given_phi"]["ci_low"])

    rule = rule_for("physical")
    target_rule = rule_for("target")
    out["rule"] = rule
    out["target_space_outcome"] = target_rule["outcome"]
    out["space_sensitive"] = target_rule["outcome"] != rule["outcome"]
    out.update(gated(rule["outcome"], gate))
    return out


def summarize_h3p(jobs, seeds, gate, n_boot, min_seeds) -> dict:
    out = {"tiers": {}}
    verdicts = {}
    for tier in VERDICT_TIERS:
        rls = [jobs[s]["h3"]["tiers"].get(tier, {}).get("rls", {}).get(PRIMARY, float("nan")) for s in seeds]
        grad = [jobs[s]["h3"]["tiers"].get(tier, {}).get("grad_tuned", {}).get(PRIMARY, float("nan")) for s in seeds]
        result = paired(rls, grad, n_boot)
        writes_rls = mean_of([jobs[s]["h3"]["tiers"].get(tier, {}).get("rls_writes") for s in seeds])
        writes_grad = mean_of([jobs[s]["h3"]["tiers"].get(tier, {}).get("grad_writes") for s in seeds])
        out["tiers"][tier] = {"rls_vs_grad_tuned": result, "writes_rls": writes_rls, "writes_grad": writes_grad}
        verdicts[tier] = enough(result, min_seeds)
    rule = h3p_rule(verdicts)
    out["rule"] = rule
    out["grad_hp_by_seed"] = {s: jobs[s]["h3"]["selection"]["hp"] for s in seeds}
    out.update(gated(rule["outcome"], gate))
    return out


def summarize_h5p(extra, seeds, k, gate, n_boot, min_seeds) -> dict:
    """extra: {seed: h5 job}. Coverage is compared on negated log det, so a_wins means dopt covered more."""
    out = {"k": k, "tiers": {}}
    seeds = [s for s in seeds if s in extra]
    rule_input = {}
    for tier in VERDICT_TIERS:
        get = lambda arm, group, metric: [extra[s]["tiers"].get(tier, {}).get(k, {}).get(arm, {}).get(group, {}).get(metric, float("nan")) for s in seeds]
        err = lambda arm: get(arm, "open_loop", PRIMARY)
        neg = lambda arm: [-x if is_finite(x) else float("nan") for x in get(arm, "coverage", "logdet_gain")]
        vs_random = paired(err("dopt"), err("random_action"), n_boot)
        coverage = paired(neg("dopt"), neg("random_action"), n_boot, threshold=0.0)
        out["tiers"][tier] = {
            "dopt_vs_random": vs_random,
            "dopt_vs_default": paired(err("dopt"), err("default"), n_boot),
            "coverage_dopt_vs_random": coverage,
            "means": {arm: mean_of(err(arm)) for arm in ("default", "random_action", "dopt", "iid_state")},
            "logdet_gain": {arm: mean_of(get(arm, "coverage", "logdet_gain")) for arm in ("default", "random_action", "dopt", "iid_state")},
            "iid_state_ceiling": mean_of(err("iid_state")),
        }
        rule_input[tier] = {"vs_random": enough(vs_random, min_seeds), "coverage_vs_random": enough(coverage, min_seeds)}
    rule = h5p_rule(rule_input)
    out["rule"] = rule
    out.update(gated(rule["outcome"], gate))
    return out


def summarize_h14p(jobs, extra, seeds, k, gate, n_boot, min_seeds) -> dict:
    """extra: {seed: h14 job}. Latent and RLS numbers come from the main job on the same stored data."""
    seeds = [s for s in seeds if s in extra]
    out = {"k": k, "tiers": {}}
    arm = lambda tier, name: [extra[s]["tiers"].get(tier, {}).get(k, {}).get(name, {}).get("open_loop", {}).get(PRIMARY, float("nan")) for s in seeds]
    for tier in VERDICT_TIERS:
        out["tiers"][tier] = {
            "means": {name: mean_of(arm(tier, name)) for name in ("error_fed", "no_error", "shuffled")},
            "vs_no_error": paired(arm(tier, "error_fed"), arm(tier, "no_error"), n_boot),
            "vs_shuffled": paired(arm(tier, "error_fed"), arm(tier, "shuffled"), n_boot),
            "vs_latent_opt": paired(arm(tier, "error_fed"), series(jobs, seeds, tier, k, "latent_opt@latent"), n_boot),
            "vs_rls": paired(arm(tier, "error_fed"), series(jobs, seeds, tier, k, "rls_best"), n_boot),
            "skipped_updates": {s: extra[s].get("train", {}).get("skipped", {}) for s in seeds},
        }
    interp, oof = out["tiers"]["interp"], out["tiers"]["oof"]
    rule = h14p_rule(enough(interp["vs_no_error"], min_seeds), enough(interp["vs_shuffled"], min_seeds), enough(interp["vs_latent_opt"], min_seeds), enough(oof["vs_rls"], min_seeds))
    out["rule"] = rule
    out.update(gated(rule["outcome"], gate))
    return out


def summarize_h16(jobs, extra, seeds, n_boot) -> dict:
    """Descriptive, no verdict. Same environments and states as interp. Only colours differ."""
    seeds = [s for s in seeds if s in extra]
    if not seeds:
        return {"status": "not_run"}
    labels = ("frozen@latent", "latent_opt@latent", "rls_best", "hybrid_eb@latent")
    get = lambda s, label: extra[s]["appearance"]["k"][PRIMARY_K].get(resolve(jobs[s], label, "interp"), {}).get("open_loop", {})
    rows = {}
    for label in labels:
        shifted = [get(s, label).get(PRIMARY, float("nan")) for s in seeds]
        base = series(jobs, seeds, "interp", PRIMARY_K, label)
        rows[label] = {"appearance_h10": mean_of(shifted), "interp_h10": mean_of(base), "appearance_vs_interp": paired(shifted, base, n_boot)}
    frozen = [get(s, "frozen@latent").get(PRIMARY, float("nan")) for s in seeds]
    recovery = {label: paired([get(s, label).get(PRIMARY, float("nan")) for s in seeds], frozen, n_boot) for label in labels[1:]}
    return {
        "status": "descriptive_no_verdict",
        "rows": rows,
        "writes_vs_frozen_on_appearance": recovery,
        "probe_floor_h0": {"appearance": mean_of([get(s, "frozen@latent").get("h0") for s in seeds]), "interp": mean_of(series(jobs, seeds, "interp", PRIMARY_K, "frozen@latent", "h0"))},
        "pixel_distance_from_training_palette": mean_of([extra[s]["appearance"].get("pixel_distance") for s in seeds]),
        "seeds": seeds,
    }


def summarize_h17(jobs, extra, seeds, n_boot) -> dict:
    """Descriptive. latent_opt by latent size, with 8 from the main job. extra: {seed: h17 job} holding every other size."""
    seeds = [s for s in seeds if s in extra]
    if not seeds:
        return {"status": "not_run"}
    sizes = sorted({int(z) for s in seeds for z in extra[s]["latents"]})
    table = {}
    for tier in VERDICT_TIERS:
        table[tier] = {"8": mean_of(series(jobs, seeds, tier, PRIMARY_K, "latent_opt@latent"))}
        for z in sizes:
            table[tier][str(z)] = mean_of([extra[s]["latents"][str(z)]["tiers"].get(tier, {}).get("latent_opt", {}).get(PRIMARY, float("nan")) for s in seeds])
    return {"status": "descriptive_no_verdict", "latent_opt_h10_by_size": table, "n_seeds": len(seeds), "seeds": seeds, "note": "Three seeds: means only, no verdict."}


def diagnostics(jobs: dict, seeds: list[str]) -> dict:
    """Things a reader must see before trusting a verdict."""
    edges: dict = {}
    same_as_frozen: dict = {}
    for s in seeds:
        job = jobs[s]
        for label, row in job["selection"].items():
            for key, where in row.get("at_edge", {}).items():
                if where:
                    edges.setdefault(f"{label}.{key}.{where}", []).append(s)
        for tier in VERDICT_TIERS:
            for label, e in job["tiers"][tier]["k"].get(PRIMARY_K, {}).items():
                if e.get("same_as_frozen"):
                    same_as_frozen.setdefault(label, []).append(f"{s}:{tier}")
    return {
        "grid_edges": edges,
        "same_as_frozen": same_as_frozen,
        "same_data_across_methods": all(job.get("same_data_check", False) for job in (jobs[s] for s in seeds)) if seeds else None,
        "persistence_h10": {t: mean_of([jobs[s]["tiers"][t]["persistence"].get(PRIMARY) for s in seeds]) for t in VERDICT_TIERS},
        "saturation_h10": {t: mean_of([jobs[s]["tiers"][t]["saturation"].get(PRIMARY) for s in seeds]) for t in VERDICT_TIERS},
        "probe_r2_val_in": {s: jobs[s]["collapse"]["r2"]["mean"] for s in seeds},
        "offscreen_fraction": {t: mean_of([jobs[s]["tiers"][t].get("offscreen_fraction") for s in seeds]) for t in VERDICT_TIERS},
        "oracle_success": {t: mean_of([jobs[s]["tiers"][t].get("oracle_closed_loop", {}).get("success_rate") for s in seeds]) for t in VERDICT_TIERS},
        "frozen_success": {t: mean_of(series(jobs, seeds, t, PRIMARY_K, "frozen@latent", "success_rate", "closed_loop")) for t in VERDICT_TIERS},
    }


def closed_loop_table(jobs, seeds, n_boot) -> dict:
    """Reported, deciding nothing. Failure on tasks the oracle solved, valid when the tier's oracle mean reaches the gate."""
    out = {}
    for tier in VERDICT_TIERS:
        oracle = mean_of([jobs[s]["tiers"][tier].get("oracle_closed_loop", {}).get("success_rate") for s in seeds])
        metric = "failure_rate_on_oracle_solved"
        fail = lambda label: series(jobs, seeds, tier, PRIMARY_K, label, metric, "closed_loop")
        row = {"oracle_success": oracle, "valid": is_finite(oracle) and oracle >= GATE_SUCCESS}
        if row["valid"]:
            row["latent_opt_vs_better_rls"] = paired(fail("latent_opt@latent"), fail("rls_best"), n_boot)
            row["hybrid_eb_vs_better_rls"] = paired(fail("hybrid_eb@latent"), fail("rls_best"), n_boot)
        else:
            row["verdict"] = "invalid_oracle_gate"
        out[tier] = row
    return out


def summarize_family(family: str, jobs: dict, extras: dict, n_boot: int, min_seeds: int) -> dict:
    """jobs: {seed: main job}. extras: {kind: {seed: job}}. Each extra is optional, so a crashed one never touches the main verdicts."""
    if not jobs:
        return {"family": family, "status": "no_finished_jobs"}
    seeds, excluded = usable_seeds(jobs)
    out = {"family": family, "status": "ok", "seeds": seeds, "excluded_collapsed": excluded}
    if not seeds:
        gate = gate_gp([], [], [], [], 0, min_seeds)
        out.update(gate=gate, statement=gate_statement(family, gate))
        return out
    gate = family_gate(jobs, seeds, min_seeds)
    out.update(gate=gate, statement=gate_statement(family, gate))
    out["H1p"] = {k: summarize_h1p(jobs, seeds, k, gate, n_boot, min_seeds) for k in ("1", "3")}
    out["H12p"] = {k: summarize_h12p(jobs, seeds, k, gate, n_boot, min_seeds) for k in ("1", "3")}
    out["H4p"] = {k: summarize_h4p(jobs, seeds, k, gate, n_boot) for k in ("1", "3")}
    out["H3p"] = summarize_h3p(jobs, seeds, gate, n_boot, min_seeds)
    out["closed_loop"] = closed_loop_table(jobs, seeds, n_boot)
    out["diagnostics"] = diagnostics(jobs, seeds)
    # Extras in their own guards, so a bad file costs only its own table.
    for name, fn in (
        ("H5p", lambda: {k: summarize_h5p(extras.get("h5", {}), seeds, k, gate, n_boot, min_seeds) for k in ("1", "3")}),
        ("H14p", lambda: {k: summarize_h14p(jobs, extras.get("h14", {}), seeds, k, gate, n_boot, min_seeds) for k in ("1", "3")}),
        ("H16", lambda: summarize_h16(jobs, extras.get("h16", {}), seeds, n_boot)),
        ("H17", lambda: summarize_h17(jobs, extras.get("h17", {}), seeds, n_boot)),
    ):
        try:
            out[name] = fn()
        except Exception as error:  # noqa: BLE001
            out[name] = {"status": "summary_failed", "error": repr(error)}
    return out


# ---------------------------------------------------------------- plain table


def _number(result: dict) -> str:
    if not isinstance(result, dict) or result.get("verdict") in (None, "undefined"):
        return "-"
    ma, mb, reduction = result.get("mean_a"), result.get("mean_b"), result.get("reduction")
    if not is_finite(ma) or not is_finite(mb):
        return "-"
    return f"{ma:.3f} vs {mb:.3f}" + (f", {100 * reduction:.0f}%" if is_finite(reduction) else "")


def _deciding(hyp: str, cell: dict) -> str:
    try:
        if hyp == "H1p":
            c = cell["comparisons"]
            return f"interp {_number(c['interp']['latent_opt_vs_better_rls'])}; oof {_number(c['oof']['latent_opt_vs_better_rls'])}"
        if hyp == "H12p":
            c = cell["comparisons"]
            return "; ".join(f"{t} {_number(c[t]['vs_better_single'])}" for t in VERDICT_TIERS)
        if hyp == "H4p":
            r = cell["tiers"]["interp"]["physical"]
            return f"P {r['within_P']['mean']:.3f}, ref {r['reference']['mean']:.3f}, phi {r['within_phi']['mean']:.3f}"
        if hyp == "H3p":
            return "; ".join(f"{t} {_number(cell['tiers'][t]['rls_vs_grad_tuned'])}" for t in VERDICT_TIERS)
        if hyp == "H5p":
            return "; ".join(f"{t} {_number(cell['tiers'][t]['dopt_vs_random'])}" for t in VERDICT_TIERS)
        if hyp == "H14p":
            t = cell["tiers"]
            return f"vs no-error {_number(t['interp']['vs_no_error'])}; vs latent {_number(t['interp']['vs_latent_opt'])}; oof vs RLS {_number(t['oof']['vs_rls'])}"
    except (KeyError, TypeError):
        pass
    return "-"


HYPOTHESES = ("H1p", "H12p", "H4p", "H3p", "H5p", "H14p")


def verdict_table(summary: dict) -> str:
    """One row per hypothesis and family. K = 1 decides. K = 3 is reported."""
    lines = ["Thread P verdicts (decoded physical error, horizon 10; K = 1 decides, K = 3 reported). Scope: own renderer, small models, not a published benchmark.", ""]
    header = f"{'hypothesis':<10} {'family':<9} {'gated outcome (K=1)':<36} {'rule outcome (K=1)':<36} {'K=3 rule':<36} deciding numbers (K=1)"
    lines += [header, "-" * len(header)]
    for family, fam in summary["families"].items():
        for hyp in HYPOTHESES:
            block = fam.get(hyp)
            if not block:
                lines.append(f"{hyp:<10} {family:<9} {'not_run':<36}")
                continue
            cell = block.get("1", block) if isinstance(block, dict) else {}
            cell3 = block.get("3", {}) if isinstance(block, dict) else {}
            lines.append(
                f"{hyp:<10} {family:<9} {str(cell.get('gated_outcome', block.get('status', 'not_run'))):<36} {str(cell.get('rule_outcome', '-')):<36} "
                f"{str(cell3.get('rule_outcome', '-')):<36} {_deciding(hyp, cell)}"
            )
    lines += ["", "Gate GP:"]
    for family, fam in summary["families"].items():
        lines.append("  " + fam.get("statement", f"{family}: {fam.get('status', 'no data')}"))
        if fam.get("excluded_collapsed"):
            lines.append(f"  {family}: seeds excluded by the collapse check: {fam['excluded_collapsed']}")
        diag = fam.get("diagnostics", {})
        if diag.get("grid_edges"):
            lines.append(f"  {family}: selections at a grid edge: {sorted(diag['grid_edges'])}")
        if diag.get("same_as_frozen"):
            lines.append(f"  {family}: methods identical to frozen on some cell (write had no effect): {sorted(diag['same_as_frozen'])}")
        if fam.get("H4p", {}).get("1", {}).get("space_sensitive"):
            lines.append(f"  {family}: H4p differs between physical and target-space readings; physical decides.")
        if fam.get("H12p", {}).get("1", {}).get("comparator_sensitive"):
            lines.append(f"  {family}: H12p depends on how the better single method is chosen.")
    lines += ["", "Descriptive, no verdict: H16 appearance and H17 latent size are in summary.json."]
    if summary.get("failed_jobs"):
        lines += ["", f"Failed jobs after one retry: {summary['failed_jobs']}"]
    if summary.get("skipped_families"):
        lines += ["", f"Families skipped after the pilot ladder failed its validation gate (not evidence for or against any method): {summary['skipped_families']}"]
    return "\n".join(lines) + "\n"
