"""Registered rules on fabricated inputs, including a family that fails its gate."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import thread_f  # noqa: E402
from intr.pixel import rules  # noqa: E402
from intr.pixel.rules import A, B, NDD  # noqa: E402

NB = 400


@pytest.mark.parametrize("interp", [A, B, NDD, "insufficient_seeds"])
@pytest.mark.parametrize("oof", [A, B, NDD, "insufficient_seeds"])
def test_h1p_equals_the_state_rule(interp, oof):
    assert rules.h1p_rule(interp, oof) == thread_f.h1r_rule(interp, oof)


def test_h1p_kill_and_support():
    assert rules.h1p_rule(A, B) == "supported"
    assert rules.h1p_rule(A, A) == "killed_one_method_wins_both"
    assert rules.h1p_rule(B, B) == "killed_one_method_wins_both"
    assert rules.h1p_rule(NDD, B) == "not_supported_no_crossover"


def test_h12p():
    assert rules.h12p_rule({"interp": NDD, "extrap": A, "oof": NDD})["outcome"] == "supported"
    killed = rules.h12p_rule({"interp": NDD, "extrap": B, "oof": NDD})
    assert killed["outcome"] == "killed" and killed["worse_in"] == ["extrap"]
    assert rules.h12p_rule({"interp": NDD, "oof": "insufficient_seeds"})["outcome"] == "insufficient_seeds"


def test_h4p_is_the_revised_rule_not_the_old_threshold():
    # 0.28 is below the superseded 0.3 kill. By the registered rule it survives if it clears half the reference and adds beyond phi.
    assert rules.h4p_rule(0.28, 0.06, A, 0.1)["outcome"] == "supported"
    assert rules.h4p_rule(0.28, 0.06, A, 0.1)["reference_has_no_force"] is True
    below = rules.h4p_rule(0.2, 0.7, A, 0.1)
    assert below["outcome"] == "killed" and below["below_half_reference"]
    assert rules.h4p_rule(0.8, 0.7, NDD, 0.1)["outcome"] == "killed"
    assert rules.h4p_rule(0.8, 0.7, A, -0.01)["outcome"] == "killed"
    assert rules.h4p_rule(None, 0.7, A, 0.1)["outcome"] == "incomplete"


def test_h14p():
    assert rules.h14p_rule(A, A, NDD, B)["outcome"] == "supported"
    assert rules.h14p_rule(A, A, A, B)["outcome"] == "supported"
    assert rules.h14p_rule(NDD, A, NDD, B)["outcome"] == "killed"
    assert rules.h14p_rule(B, A, NDD, B)["outcome"] == "killed"
    assert rules.h14p_rule(A, NDD, NDD, B)["outcome"] == "not_killed_some_predictions_failed"
    assert rules.h14p_rule(A, A, B, B)["outcome"] == "not_killed_some_predictions_failed"
    assert rules.h14p_rule("insufficient_seeds", A, A, B)["outcome"] == "insufficient_seeds"


def test_h5p():
    moved = lambda v: {"vs_random": v, "coverage_vs_random": A}
    still = lambda v: {"vs_random": v, "coverage_vs_random": NDD}
    assert rules.h5p_rule({"interp": still(NDD), "oof": still(A)})["outcome"] == "inconclusive_coverage_unmoved"
    assert rules.h5p_rule({"interp": moved(NDD), "oof": still(A)})["outcome"] == "killed"
    assert rules.h5p_rule({"interp": moved(A), "oof": moved(A)})["outcome"] == "supported"
    partial = rules.h5p_rule({"interp": moved(A), "oof": moved(NDD)})
    assert partial["outcome"] == "partial" and partial["gain_with_moved_coverage_in"] == ["interp"]
    assert rules.h5p_rule({"oof": still(A), "interp": moved(NDD)})["gain_without_moved_coverage_in"] == ["oof"]


def test_h3p():
    assert rules.h3p_rule({"interp": NDD, "extrap": A, "oof": A})["outcome"] == "supported"
    assert rules.h3p_rule({"interp": B, "extrap": A, "oof": A})["outcome"] == "killed"
    assert rules.h3p_rule({"interp": NDD, "extrap": A, "oof": B})["gradient_wins_in"] == ["oof"]
    assert rules.h3p_rule({"interp": A, "extrap": A, "oof": NDD})["outcome"] == "not_killed_expectation_differs"


def test_gate_thresholds_and_statement():
    good = rules.gate_gp([0.6], [0.9], [0.8], [1.0], 5, 5)
    assert good["pass"] and "passes" in rules.gate_statement("drone", good)
    assert rules.gate_gp([0.5], [0.5], [0.875], [1.0], 5, 5)["pass"]  # success exactly at its threshold, 12.5% reduction
    for bad in (([0.49], [0.9], [0.5], [1.0], 5, 5), ([0.9], [0.49], [0.5], [1.0], 5, 5), ([0.9], [0.9], [0.95], [1.0], 5, 5), ([0.9], [0.9], [0.5], [1.0], 4, 5), ([float("nan")], [0.9], [0.5], [1.0], 5, 5)):
        assert not rules.gate_gp(*bad)["pass"]
    text = rules.gate_statement("puck", rules.gate_gp([0.1], [0.9], [0.5], [1.0], 10, 5))
    assert "FAILS" in text and "not evidence" in text


def test_gated_hides_outcome_only_in_gated_field():
    failing = {"pass": False}
    row = rules.gated("supported", failing)
    assert row == {"rule_outcome": "supported", "gate_pass": False, "gated_outcome": rules.INVALID}


# ---------------------------------------------------------------- fabricated jobs through the summary


LABELS = ("frozen@latent", "frozen@nolatent", "frozen@control", "latent_opt@latent", "rls@latent", "rls@nolatent", "hybrid@latent", "ebrls@latent", "ebrls@nolatent", "hybrid_eb@latent")


def fabricated_job(seed: int, frozen_success=0.8, oracle=1.0, latent_loss=0.5, r2=0.9, h10=None, rls_best="rls@latent"):
    """h10: {tier: {label: value}}. Small per-seed wobble, so a CI has width."""
    wobble = 0.002 * (seed - 14)
    tiers = {}
    for tier in ("train", "interp", "extrap", "oof"):
        k = {}
        for label in LABELS:
            value = (h10 or {}).get(tier, {}).get(label, 0.5) + wobble
            entry = {"open_loop": {"h0": 0.05, "h1": value / 2, "h10": value, "h30": value * 1.2}, "info": {}, "data_sha256": "x"}
            if tier != "train":
                entry["closed_loop"] = {"success_rate": frozen_success, "failure_rate": 1 - frozen_success, "failure_rate_on_oracle_solved": 1 - frozen_success}
            k[label] = entry
        record = {"k": {"1": k, "3": k}, "persistence": {"h10": 1.0}, "saturation": {"h10": 1.1}, "offscreen_fraction": 0.0}
        if tier != "train":
            record["oracle_closed_loop"] = {"success_rate": oracle}
        tiers[tier] = record
    cell = {"physical": {"within_P": 0.8, "within_phi": 0.3 + wobble, "partial_given_phi": 0.5 + wobble, "reference": 0.7}, "target": {"within_P": 0.8, "within_phi": 0.3 + wobble, "partial_given_phi": 0.5 + wobble, "reference": 0.7}, "lambda": 1.0}
    grad_tiers = {t: {"rls": {"h10": 0.5 + wobble}, "grad_tuned": {"h10": 0.5 + wobble}, "rls_writes": 40, "grad_writes": 800} for t in rules.VERDICT_TIERS}
    return {
        "seed": seed,
        "status": "ok",
        "collapse": {"passes": r2 >= 0.5, "r2": {"mean": r2}},
        "train": {"exact_train_loss": {"latent": latent_loss, "nolatent": 1.0}},
        "selection": {label: {"hp": {}, "at_edge": {}} for label in LABELS},
        "val_errors": {label: [0.4, 0.6] for label in LABELS},
        "rls_best": rls_best,
        "same_data_check": True,
        "tiers": tiers,
        "h3": {"selection": {"hp": {"lr": 0.1, "passes": 5}}, "tiers": grad_tiers},
        "h4": {t: {"1": cell, "3": cell} for t in rules.VERDICT_TIERS},
    }


CROSSOVER = {
    "interp": {"latent_opt@latent": 0.2, "rls@latent": 0.5, "hybrid_eb@latent": 0.22},
    "extrap": {"latent_opt@latent": 0.5, "rls@latent": 0.5, "hybrid_eb@latent": 0.5},
    "oof": {"latent_opt@latent": 0.8, "rls@latent": 0.4, "hybrid_eb@latent": 0.42},
}


def jobs_of(**kwargs):
    return {str(s): fabricated_job(s, h10=CROSSOVER, **kwargs) for s in range(10, 20)}


def test_summary_on_a_passing_family_applies_the_rules():
    out = rules.summarize_family("pendulum", jobs_of(), {}, NB, 5)
    assert out["gate"]["pass"]
    assert out["H1p"]["1"]["gated_outcome"] == "supported"
    assert out["H12p"]["1"]["rule_outcome"] == "supported"
    assert out["H4p"]["1"]["rule_outcome"] == "supported"
    assert out["H3p"]["rule_outcome"] == "not_killed_expectation_differs"
    assert out["H5p"]["1"]["rule_outcome"] == "insufficient_seeds"
    assert out["H14p"]["1"]["rule_outcome"] == "insufficient_seeds"
    assert out["H16"]["status"] == "not_run" and out["H17"]["status"] == "not_run"


def test_summary_on_a_gate_failure_keeps_rule_outcome_and_invalidates_gated():
    out = rules.summarize_family("puck", jobs_of(frozen_success=0.1), {}, NB, 5)
    assert not out["gate"]["pass"]
    assert "FAILS" in out["statement"]
    assert out["H1p"]["1"]["rule_outcome"] == "supported"
    assert out["H1p"]["1"]["gated_outcome"] == rules.INVALID
    assert out["H12p"]["1"]["gated_outcome"] == rules.INVALID
    assert out["H4p"]["1"]["gated_outcome"] == rules.INVALID
    table = rules.verdict_table({"families": {"puck": out}, "failed_jobs": {}, "skipped_families": []})
    assert rules.INVALID in table and "FAILS Gate GP" in table


def test_latent_that_lowers_loss_too_little_fails_the_gate():
    assert not rules.summarize_family("drone", jobs_of(latent_loss=0.95), {}, NB, 5)["gate"]["pass"]


def test_collapsed_seeds_leave_the_paired_set_and_too_many_invalidate():
    jobs = jobs_of()
    jobs["12"]["collapse"] = {"passes": False, "r2": {"mean": 0.1}}
    out = rules.summarize_family("pendulum", jobs, {}, NB, 5)
    assert out["excluded_collapsed"] == ["12"] and "12" not in out["seeds"]
    assert out["H1p"]["1"]["comparisons"]["interp"]["latent_opt_vs_better_rls"]["n_seeds"] == 9
    all_bad = jobs_of(r2=0.1)
    out = rules.summarize_family("pendulum", all_bad, {}, NB, 5)
    assert not out["gate"]["pass"] and out["seeds"] == []


def test_too_few_seeds_gives_no_verdict():
    jobs = {s: j for s, j in jobs_of().items() if int(s) < 14}
    out = rules.summarize_family("pendulum", jobs, {}, NB, 5)
    assert out["H1p"]["1"]["rule_outcome"] == "insufficient_seeds"
    assert not out["gate"]["pass"]


def test_better_single_is_chosen_on_validation_per_tier():
    job = fabricated_job(10)
    job["val_errors"]["latent_opt@latent"] = [0.1, 0.9]
    job["val_errors"]["rls@latent"] = [0.5, 0.5]
    assert rules.resolve(job, "better_single", "interp") == "latent_opt@latent"
    assert rules.resolve(job, "better_single", "oof") == "rls@latent"
    assert rules.resolve(job, "better_single", "extrap") == "rls@latent"


def test_a_bad_extra_file_costs_only_its_own_table():
    jobs = jobs_of()
    extras = {"h16": {"10": {"broken": True}}, "h14": {"10": {"tiers": None}}}
    out = rules.summarize_family("pendulum", jobs, extras, NB, 5)
    assert out["H1p"]["1"]["gated_outcome"] == "supported"
    assert out["H16"]["status"] == "summary_failed"
    assert out["H14p"]["status"] == "summary_failed"


def test_h14_and_h5_extras_reach_their_rules():
    jobs = jobs_of()
    h14, h5 = {}, {}
    for s in jobs:
        wobble = 0.002 * (int(s) - 14)
        arm = lambda v: {"open_loop": {"h10": v + wobble}}
        fed = {"interp": (0.2, 0.5, 0.5), "extrap": (0.5, 0.5, 0.5), "oof": (0.8, 0.8, 0.8)}
        h14[s] = {"seed": int(s), "train": {"skipped": {}}, "tiers": {t: {k: {"error_fed": arm(v[0]), "no_error": arm(v[1]), "shuffled": arm(v[2])} for k in ("1", "3")} for t, v in fed.items()}}
        cell = lambda e, c: {"open_loop": {"h10": e + wobble}, "coverage": {"logdet_gain": c + wobble}}
        h5[s] = {"seed": int(s), "tiers": {t: {k: {"default": cell(0.6, 8.0), "random_action": cell(0.5, 9.0), "dopt": cell(0.3, 12.0), "iid_state": cell(0.2, 14.0)} for k in ("1", "3")} for t in rules.VERDICT_TIERS}}
    out = rules.summarize_family("pendulum", jobs, {"h14": h14, "h5": h5}, NB, 5)
    assert out["H14p"]["1"]["rule_outcome"] == "supported"
    assert out["H5p"]["1"]["rule_outcome"] == "supported"
    assert out["H5p"]["1"]["tiers"]["interp"]["iid_state_ceiling"] == pytest.approx(0.2, abs=0.01)
