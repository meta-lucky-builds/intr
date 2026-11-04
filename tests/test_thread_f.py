"""Thread F: registered rules on fabricated verdicts, D-optimal design on known cases, then the pipeline on tiny settings."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import merge_results  # noqa: E402
import thread_a  # noqa: E402
import thread_f  # noqa: E402
import thread_g  # noqa: E402
from intr.adapt import RLSState  # noqa: E402
from intr.ebayes import NO_WRITE_LAMBDA, logdet_precision_gain  # noqa: E402
from intr.envs import FAMILIES, collect_trajectories, make_env, make_generator  # noqa: E402
from intr.models import ModelConfig  # noqa: E402
from intr.train import TrainConfig  # noqa: E402

A, B, NDD = "a_wins", "b_wins", "no_detectable_difference"
DOUBLE = torch.float64


# ---------------------------------------------------------------- rules


def test_no_worse_is_the_protocol_reading():
    assert thread_f.worse(B)
    assert not thread_f.worse(NDD)  # a CI that excludes 0 with a small effect is NDD, so it is not worse
    assert not thread_f.worse(A)


@pytest.mark.parametrize(
    "interp,oof,expected",
    [
        (A, B, "supported"),
        (A, A, "killed_one_method_wins_both"),
        (B, B, "killed_one_method_wins_both"),
        (A, NDD, "not_supported_no_crossover"),
        (NDD, NDD, "not_supported_no_crossover"),
        (B, A, "not_supported_no_crossover"),
        ("insufficient_seeds", B, "insufficient_seeds"),
        (A, "undefined", "insufficient_seeds"),
    ],
)
def test_h1r_rule(interp, oof, expected):
    assert thread_f.h1r_rule(interp, oof) == expected


def _h12_row(**over):
    row = {"vs_hybrid": NDD, "vs_latent_opt": NDD, "vs_rls": NDD, "vs_better_single": NDD, "vs_frozen": NDD, "rls_vs_frozen": A, "median_edf": 5.0}
    row.update(over)
    return row


def _h12_tiers(**tiers):
    out = {t: _h12_row() for t in ("interp", "extrap", "oof")}
    for tier, over in tiers.items():
        out[tier] = _h12_row(**over)
    return out


def test_h12_supported_needs_the_oof_win_and_no_kill():
    assert thread_f.h12_rule(_h12_tiers(oof={"vs_hybrid": A}))["outcome"] == "supported"
    assert thread_f.h12_rule(_h12_tiers())["outcome"] == "not_supported_no_kill"


@pytest.mark.parametrize(
    "tiers",
    [
        {"oof": {"vs_hybrid": A, "vs_rls": B}},  # worse than RLS out of family
        {"oof": {"vs_hybrid": A}, "interp": {"vs_latent_opt": B}},  # worse than latent on interp
        {"oof": {"vs_hybrid": A}, "extrap": {"vs_better_single": B}},  # any tier, including one the prediction does not name
    ],
)
def test_h12_kill_fires_on_any_worse_tier_even_with_the_oof_win(tiers):
    result = thread_f.h12_rule(_h12_tiers(**tiers))
    assert result["outcome"] == "killed" and result["kill_fired"]


def test_h12_small_effects_do_not_kill():
    result = thread_f.h12_rule(_h12_tiers(oof={"vs_hybrid": A, "vs_rls": NDD}, interp={"vs_latent_opt": NDD}))
    assert not result["kill_fired"]


def test_h12_clause_for_a_family_where_writes_do_not_help():
    no_help = {"rls_vs_frozen": NDD, "median_edf": 0.2}
    tiers = {t: _h12_row(**no_help) for t in ("interp", "extrap", "oof")}
    tiers["oof"]["vs_hybrid"] = A
    good = thread_f.h12_rule(tiers)
    assert not good["writes_help_on_this_family"]
    assert good["weak_writes_and_no_worse_than_frozen"] is True
    assert good["outcome"] == "supported"

    tiers["interp"]["median_edf"] = 7.0  # it wrote hard where nothing helps
    strong = thread_f.h12_rule(tiers)
    assert strong["weak_writes_and_no_worse_than_frozen"] is False
    assert strong["outcome"] == "not_supported_no_kill"

    tiers["interp"]["median_edf"] = 0.2
    tiers["interp"]["vs_frozen"] = B
    assert thread_f.h12_rule(tiers)["weak_writes_and_no_worse_than_frozen"] is False

    # Where writes do help, the clause does not apply.
    assert thread_f.h12_rule(_h12_tiers(oof={"vs_hybrid": A}))["weak_writes_and_no_worse_than_frozen"] is None


def test_h12_unusable_verdicts_are_not_verdicts():
    assert thread_f.h12_rule(_h12_tiers(oof={"vs_rls": "insufficient_seeds"}))["outcome"] == "insufficient_seeds"
    assert thread_f.h12_rule({"interp": _h12_row()})["outcome"] == "insufficient_seeds"


def _h13_cell(plain=NDD, reset=NDD, surprise=NDD, tainted=False):
    return {"vs_plain": plain, "vs_reset": reset, "vs_surprise": surprise, "tainted": tainted}


def test_h13_rule():
    assert thread_f.h13_rule({"a": _h13_cell(A), "b": _h13_cell(A, A)})["outcome"] == "supported"
    # Matches reset counts. Losing to reset does not.
    assert thread_f.h13_rule({"a": _h13_cell(A, B), "b": _h13_cell(A)})["outcome"] == "partial"
    assert thread_f.h13_rule({"a": _h13_cell(A), "b": _h13_cell(NDD)})["outcome"] == "partial"
    assert thread_f.h13_rule({"a": _h13_cell(), "b": _h13_cell(B)})["outcome"] == "killed"
    # Beats reset everywhere, but not plain: the kill is about plain forgetting.
    assert thread_f.h13_rule({"a": _h13_cell(NDD, A)})["outcome"] == "killed"
    assert thread_f.h13_rule({"a": _h13_cell("insufficient_seeds")})["outcome"] == "insufficient_seeds"
    assert thread_f.h13_rule({})["outcome"] == "incomplete"


def test_h13_win_with_numerical_failures_does_not_count():
    result = thread_f.h13_rule({"a": _h13_cell(A, tainted=True)})
    assert result["outcome"] == "inconclusive_numerical_failure"
    assert result["beats_plain"] == [] and result["tainted"] == ["a"]


def _h5_row(random=NDD, default=NDD, cover=NDD):
    return {"vs_random": random, "vs_default": default, "coverage_vs_random": cover}


def test_h5_rule_reads_the_manipulation_check():
    # Problem 5: a null with coverage unmoved says nothing about coverage, and the outcome must say which null this is.
    unmoved = thread_f.h5_rule({"interp": _h5_row(), "oof": _h5_row()})
    assert unmoved["outcome"] == "killed_coverage_did_not_move" and unmoved["kill_fired"]
    assert unmoved["manipulation_failed_in"] == ["interp", "oof"]
    moved = thread_f.h5_rule({"interp": _h5_row(cover=A), "oof": _h5_row()})
    assert moved["outcome"] == "killed_coverage_moved_no_error_gain" and moved["kill_fired"]
    assert moved["manipulation_failed_in"] == ["oof"]


def test_h5_rule_outcomes():
    win = _h5_row(A, A, A)
    assert thread_f.h5_rule({"interp": win, "oof": win})["outcome"] == "supported"
    partial = thread_f.h5_rule({"interp": win, "oof": _h5_row(cover=A)})
    assert partial["outcome"] == "partial" and not partial["kill_fired"]
    assert partial["gain_over_random_in"] == ["interp"]
    # Better than random but not than default is not full support.
    assert thread_f.h5_rule({"interp": _h5_row(A, NDD, A)})["outcome"] == "partial"
    assert thread_f.h5_rule({"interp": _h5_row(A, A, NDD)})["gain_without_moved_coverage_in"] == ["interp"]
    assert thread_f.h5_rule({"interp": _h5_row("insufficient_seeds")})["outcome"] == "insufficient_seeds"


def test_closed_loop_verdict_is_conditional_on_the_stated_gate():
    a, b = [0.1, 0.2, 0.1, 0.2, 0.1], [0.5, 0.6, 0.5, 0.6, 0.5]
    below = thread_f.closed_verdict({"gate": 0.5, "oracle_success": 0.33, "valid": False}, a, b, 200)
    assert below["verdict"] == "invalid_oracle_gate" and not below["valid"] and below["gate"] == 0.5
    above = thread_f.closed_verdict({"gate": 0.5, "oracle_success": 0.9, "valid": True}, a, b, 200)
    assert above["verdict"] == A and above["valid"]


def test_gated_outcome_keeps_the_rule_outcome_visible():
    assert thread_f.gated("supported", {"pass": True})["gated_outcome"] == "supported"
    failed = thread_f.gated("supported", {"pass": False})
    assert failed["gated_outcome"] == "invalid_family_gates_failed" and failed["rule_outcome"] == "supported"


def _fake_entry(h10):
    return {"open_loop": {"h1": h10 / 3.0, "h10": h10, "h30": h10 * 2.0}}


def test_h1r_reads_the_rls_chosen_on_validation_not_rls_latent():
    """Problem 4. rls@latent alone would make latent lose on interp; the registered better-of-two is rls@nolatent."""
    jobs = {}
    for i, seed in enumerate("56789"):
        wiggle = 0.01 * i
        jobs[seed] = {
            "rls_best": "rls@nolatent",
            "tiers": {
                "interp": {"k": {"1": {"latent_opt@latent": _fake_entry(0.1 + wiggle), "rls@latent": _fake_entry(0.05 + wiggle), "rls@nolatent": _fake_entry(0.5 + wiggle)}}},
                "oof": {"k": {"1": {"latent_opt@latent": _fake_entry(0.5 + wiggle), "rls@latent": _fake_entry(0.9 + wiggle), "rls@nolatent": _fake_entry(0.1 + wiggle)}}},
            },
        }
    out = thread_f.summarize_h1r(jobs, ["interp", "oof"], {"pass": True}, 500)
    assert out["verdicts"] == {"interp": A, "oof": B}
    assert out["rule_outcome"] == "supported"
    assert out["comparisons"]["interp"]["latent_opt_vs_rls_latent"]["h10"]["verdict"] == B  # the other reading is still stored
    assert out["comparisons"]["interp"]["closed_loop"]["verdict"] == "invalid_oracle_gate"  # no oracle data, so no verdict


def test_better_single_is_chosen_on_matching_validation_tier():
    job = {"rls_best": "rls@latent", "val_errors": {"latent_opt@latent": [0.1, 0.9], "rls@latent": [0.2, 0.3]}}
    assert thread_f.resolve(job, "better_single", "interp") == "latent_opt@latent"
    assert thread_f.resolve(job, "better_single", "oof") == "rls@latent"
    assert thread_f.resolve(job, "better_single", "extrap") == "rls@latent"
    assert thread_f.resolve(job, "rls_best", "oof") == "rls@latent"
    assert thread_f.resolve(job, "frozen@latent", "oof") == "frozen@latent"


def test_grid_edges_flag_pinned_choices():
    table = [{"hp": {"lambda_": x}} for x in (0.1, 1.0, 10.0)]
    assert thread_f.grid_edges({"hp": {"lambda_": 0.1}, "table": table})["lambda_"]["at_low"]
    assert thread_f.grid_edges({"hp": {"lambda_": 10.0}, "table": table})["lambda_"]["at_high"]
    middle = thread_f.grid_edges({"hp": {"lambda_": 1.0}, "table": table})["lambda_"]
    assert not middle["at_low"] and not middle["at_high"]
    assert thread_f.grid_edges({"hp": {}, "note": "reference"}) == {}


# ---------------------------------------------------------------- D-optimal design


class _ToyModel:
    def apply_delta(self, o, delta):
        return o + delta

    def target(self, o, next_o):
        return next_o - o


class _BucketPredictor:
    """phi is a one-hot of the action's bucket, so coverage is exactly how evenly the buckets are visited."""

    model = _ToyModel()

    def __init__(self, n_envs, n_buckets, lam):
        self.n_buckets = n_buckets
        self.rls = RLSState(torch.zeros(n_envs, 1, n_buckets), lam)

    def features(self, o, a):
        index = ((a[..., 0] + 1.0) * self.n_buckets / 2.0).long().clamp(0, self.n_buckets - 1)
        return torch.nn.functional.one_hot(index, self.n_buckets).to(torch.float32)

    def observe(self, o, a, next_o):
        phi = self.features(o[:, None], a[:, None])[:, 0]
        self.rls.write(phi, self.model.target(o, next_o))


class _StillEnv:
    """The state never changes. Only used to separate the design from any dynamics."""

    act_dim = 1

    def __init__(self, n_envs):
        self.n_envs = n_envs

    def obs(self, state):
        return state.to(torch.float32)

    def step(self, state, action):
        return state


def test_greedy_design_raises_logdet_more_than_random_on_a_known_case():
    # 16 steps over 8 buckets. Random misses a bucket about 12% of the time, which costs a lot at this small lambda.
    E, n, lam, buckets, candidates = 64, 16, 0.1, 8, 16
    env = _StillEnv(E)
    s0 = torch.zeros(E, 1, 1)
    low, high = torch.tensor([-10.0]), torch.tensor([10.0])
    greedy, info = thread_f.collect_dopt(env, s0, n, _BucketPredictor(E, buckets, lam), candidates, 1, torch.Generator().manual_seed(0), low, high)
    random = thread_f.probe_from_state(env, s0, n, "iid", torch.Generator().manual_seed(1))
    assert greedy.act.shape == random.act.shape == (E, 1, n, 1)
    assert info["forward_samples"] == E * n * candidates

    reader = _BucketPredictor(E, buckets, lam)

    def gain(trajs):
        data = trajs.flat()
        return logdet_precision_gain(reader.features(data.obs, data.act).double(), lam)

    greedy_gain, random_gain = gain(greedy), gain(random)
    assert greedy_gain.mean().item() > random_gain.mean().item() + 0.4
    # Even visits are the maximum, 8 log(1 + 2 / lam), and greedy comes close to it.
    assert greedy_gain.mean().item() > 0.96 * 8 * torch.log(torch.tensor(1.0 + 2.0 / lam)).item()


class _SmoothPredictor:
    model = _ToyModel()

    def __init__(self, n_envs, lam):
        self.rls = RLSState(torch.zeros(n_envs, 1, 4), lam)

    def features(self, o, a):
        return torch.cat([a, a**2, torch.ones_like(a), o[..., :1]], dim=-1)


def test_scores_equal_the_logdet_gain_of_the_whole_sequence():
    """The determinant lemma: the sum of log(1 + q) with P updated in between is log det(A_H) - log det(A_0)."""
    E, C, H = 2, 3, 3
    lam = 0.5
    predictor = _SmoothPredictor(E, lam)
    gen = torch.Generator().manual_seed(2)
    seqs = 2.0 * torch.rand(E, C, H, 1, generator=gen) - 1.0
    o = torch.rand(E, 1, generator=gen)
    low, high = torch.tensor([-10.0]), torch.tensor([10.0])
    P_before = predictor.rls.P.clone()
    scores = thread_f.dopt_scores(predictor, o, seqs, low, high)
    assert torch.equal(predictor.rls.P, P_before)  # planning does not touch the state
    for e in range(E):
        for c in range(C):
            phi = predictor.features(o[e][None, None].expand(1, H, -1), seqs[e, c][None]).double()
            assert scores[e, c].item() == pytest.approx(logdet_precision_gain(phi, lam)[0].item(), abs=1e-5)


def test_later_steps_discount_what_earlier_steps_already_covered():
    lam = 1.0
    predictor = _SmoothPredictor(1, lam)
    o = torch.zeros(1, 1)
    low, high = torch.tensor([-10.0]), torch.tensor([10.0])
    same = torch.full((1, 1, 2, 1), 0.5)
    different = torch.tensor([0.5, -0.9])[None, None, :, None]
    assert thread_f.dopt_scores(predictor, o, different, low, high).item() > thread_f.dopt_scores(predictor, o, same, low, high).item()


def test_action_sequences_stay_in_range_and_include_corners():
    seqs = thread_f.sample_action_sequences(3, 16, 4, 2, torch.Generator().manual_seed(3))
    assert seqs.shape == (3, 16, 4, 2)
    assert seqs.abs().max().item() <= 1.0
    assert (seqs[:, :4].abs() == 1.0).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_probe_from_state_reproduces_collect_trajectories(family):
    env = make_env(family, 3, "interp", 1)
    theirs = collect_trajectories(env, 2, 8, env.default_probe, make_generator("pair", family))
    gen = make_generator("pair", family)
    s0 = env.reset(2, gen)
    ours = thread_f.probe_from_state(env, s0, 8, env.default_probe, gen)
    assert torch.equal(ours.obs, theirs.obs)
    assert torch.equal(ours.act, theirs.act)


# ---------------------------------------------------------------- pipeline on tiny settings


def _tiny() -> thread_f.ThreadFConfig:
    cfg = thread_f.smoke_config()
    cfg.train = TrainConfig(model_steps=80, warmup_steps=20, block_steps=20, batch_size=32, log_every=40)
    cfg.train_envs = 6
    cfg.train_traj = 2
    cfg.train_steps = 10
    cfg.n_eval_envs = 4
    cfg.n_val_envs = 2
    cfg.adapt_steps = 12  # against 16 features: K = 1 takes the dual path and K = 3 the primal one
    cfg.stream_steps = 90
    cfg.n_streams = 3
    cfg.n_val_streams = 2
    cfg.n_boot = 100
    return cfg


@pytest.fixture(scope="module")
def spied_job():
    """One tiny job, recording which data every direct call to adapt received."""
    calls = []
    real = thread_f.adapt

    def spy(method, backbone, data, **hp):
        calls.append((method, thread_f.data_sha256(data), data.obs.shape[1]))
        return real(method, backbone, data, **hp)

    patch = pytest.MonkeyPatch()
    patch.setattr(thread_f, "adapt", spy)
    try:
        job = thread_f.run_job("pendulum", 5, _tiny())
    finally:
        patch.undo()
    return thread_f._jsonable(job), calls


@pytest.fixture(scope="module")
def job(spied_job):
    return spied_job[0]


def test_every_non_probe_arm_of_a_cell_reads_the_same_trajectories(spied_job):
    """Problem 12, checked from the data itself: regenerate the probes independently and compare with what adapt received."""
    job, calls = spied_job
    cfg = _tiny()
    expected = {}
    for tier in cfg.open_loop_tiers:
        env = make_env("pendulum", cfg.n_eval_envs, tier, thread_f.derive_seed("test_envs", 5))
        probes = collect_trajectories(env, max(cfg.ks), cfg.adapt_steps, env.default_probe, make_generator(("test", "pendulum", 5, tier), "probe"))
        for k in cfg.ks:
            sha = thread_f.data_sha256(probes.take(k).flat())
            expected[(tier, k)] = sha
            for label, entry in job["tiers"][tier]["k"][str(k)].items():
                assert entry["data_sha256"] == sha, (tier, k, label)
    allowed = set(expected.values())
    seen_methods = set()
    for method, sha, _ in calls:
        if method != "rls":  # rls also runs on the H5 probes
            assert sha in allowed, method
            seen_methods.add(method)
    assert {"frozen", "latent_opt", "hybrid", "ebrls", "hybrid_eb"} <= seen_methods


def test_h5_default_arm_is_the_main_probe_and_all_arms_match_in_size(job):
    cfg = _tiny()
    best = job["rls_best"]
    for tier, record in job["h5"]["tiers"].items():
        for k in cfg.ks:
            cell = record["k"][str(k)]
            assert set(cell) == set(thread_f.H5_ARMS)
            main = job["tiers"][tier]["k"][str(k)][best]
            assert cell["default"]["data_sha256"] == main["data_sha256"]
            assert cell["default"]["open_loop"]["h10"] == pytest.approx(main["open_loop"]["h10"])
            sizes = {entry["n_transitions"] for entry in cell.values()}
            assert sizes == {k * cfg.adapt_steps}
            hashes = {entry["data_sha256"] for entry in cell.values()}
            assert len(hashes) == len(thread_f.H5_ARMS)  # the arms really differ
        assert len(set(record["initial_obs_sha256"].values())) == 1  # same initial states


def test_h5_records_the_manipulation_check(job):
    cell = job["h5"]["tiers"]["interp"]["k"]["1"]
    for arm in thread_f.H5_ARMS:
        coverage = cell[arm]["coverage"]
        assert coverage["lambda"] == job["h5"]["design_lambda"]
        assert coverage["logdet_gain"] > 0.0
        assert len(coverage["logdet_gain_per_env"]) == 4
        assert "closed_loop" in cell[arm]
    assert job["h5"]["design_lambda"] <= thread_f.DESIGN_LAMBDA_MAX
    assert job["h5"]["tiers"]["interp"]["collect"]["dopt"]["n_candidates"] == 6


def test_job_stores_per_task_oracle_success_and_gate_inputs(job):
    for tier in ("interp", "extrap", "oof"):
        oracle = job["tiers"][tier]["oracle_closed_loop"]
        assert len(oracle["per_env_success"]) == 4
        entry = job["tiers"][tier]["k"]["1"]["hybrid_eb@latent"]["closed_loop"]
        assert len(entry["per_env_success"]) == 4
        assert "failure_rate_on_oracle_solved" in entry
    assert "closed_loop" not in job["tiers"]["interp"]["k"]["3"]["hybrid_eb@latent"]
    loss = job["train"]["exact_train_loss"]
    assert set(loss) == {"latent", "latent_at_mean_z", "nolatent", "control"}
    assert all(v > 0.0 for v in loss.values())


def test_eb_arms_report_selected_values_and_checks(job):
    for tier in ("interp", "oof"):
        for k, path in (("1", "dual"), ("3", "primal")):
            entry = job["tiers"][tier]["k"][k]["hybrid_eb@latent"]
            info = entry["info"]["eb"]
            assert info["path"] == path and len(info["lambda"]) == 4
            checks = entry["evidence_checks"]
            assert len(checks["innovations"]["lag1_autocorr"]) == 4
            assert 0.0 <= checks["heldout_coverage"]["0.9"] <= 1.0
            assert 0.0 <= checks["heldout_coverage_no_write"]["0.9"] <= 1.0
            # Chain rule against the closed form, on real features. Loose, since sequential RLS at lambda = 1e-6 on
            # near-collinear features loses digits. test_ebayes holds the exact version on conditioned data.
            for a, b in zip(checks["innovations"]["chain_log_evidence"], info["log_evidence"]):
                assert a == pytest.approx(b, abs=0.5 + 1e-2 * abs(b))
        curve = job["tiers"][tier]["lambda_curve"]["hybrid_eb@latent"]
        assert len(curve["mean_h10_by_lambda"]) == 3
        assert curve["h10_best_per_env_lambda"] <= curve["h10_best_fixed_lambda"] + 1e-6
    assert "hybrid_eb_per_output" in job["tiers"]["oof"]["k"]["1"]
    assert "hybrid_eb_per_output" not in job["tiers"]["oof"]["k"]["3"]


def test_selection_grids_include_no_write_and_rls_comparator_is_recorded(job):
    assert job["rls_best"] in ("rls@latent", "rls@nolatent")
    assert job["ebrls_best"] in ("ebrls@latent", "ebrls@nolatent")
    assert job["frozen_best"] == "frozen@" + job["rls_best"].split("@")[1]
    table = job["selection"]["rls@latent"]["table"]
    assert NO_WRITE_LAMBDA in [row["hp"]["lambda_"] for row in table]
    assert job["val_errors"]["latent_opt@latent"][0] > 0.0 and len(job["val_errors"]["rls@latent"]) == 2


def test_h13_uses_validation_and_every_method_replays_each_cell(job):
    tiers = job["h13"]["tiers"]
    assert set(tiers) == {"interp", "extrap", "oof"}
    methods = tiers["interp"]["fast"]["methods"]
    assert set(methods) == set(thread_f.H13_METHODS + thread_f.H13_REFERENCES)
    prior = methods["prior_forgetting"]
    assert prior["hp"]["gamma"] in _tiny().forgetting_grid
    assert thread_f.ROLLOUT_KEY in prior
    assert prior["windup"]["n_failures"] == 0
    assert set(prior) >= set(methods["forgetting"]) - {"counts_excluding_shared_features"}


def test_summary_runs_on_both_null_and_nan_and_gives_the_same_answer(job):
    jobs = {"5": job, "6": job}
    direct = thread_f.summarize(jobs, 100)
    revived = thread_f.summarize(merge_results.revive_nan(json.loads(json.dumps(jobs))), 100)
    assert json.dumps(thread_f._jsonable(direct), sort_keys=True, allow_nan=False) == json.dumps(thread_f._jsonable(revived), sort_keys=True, allow_nan=False)


def test_summary_states_gates_outcomes_and_selected_values(job):
    # The same job twice checks plumbing only. A repeated seed gives a degenerate CI, so verdicts mean nothing.
    summary = thread_f.summarize({"5": job, "6": job}, 100)
    gates = summary["gates"]
    assert gates["oracle_gate"] == 0.5 and gates["latent_gate"] == 0.1
    assert gates["oracle"]["interp"]["gate"] == 0.5
    assert isinstance(gates["pass"], bool)
    assert "reduction_latent_vs_nolatent" in gates["exact_train_loss"]
    for hypothesis in ("H1r", "H12", "H13", "H5"):
        block = summary[hypothesis]
        assert block["rule_outcome"] != "insufficient_seeds", hypothesis
        assert block["family_gates_pass"] == gates["pass"]
        assert block["gated_outcome"] in (block["rule_outcome"], "invalid_family_gates_failed")
    selected = summary["H12"]["selected"]["oof"]["1"]["hybrid_eb@latent"]
    assert selected["median_log10_lambda"]["n_seeds"] == 2
    assert selected["heldout_coverage"]["0.5"]["n_seeds"] == 2
    assert summary["H5"]["tiers"]["interp"]["same_initial_states"]
    assert "coverage_dopt_vs_random" in summary["H5"]["tiers"]["interp"]["k"]["1"]
    assert summary["H5"]["registered_scope"]
    assert "main" in summary["grid_edges"]


def test_one_seed_is_reported_as_insufficient(job):
    summary = thread_f.summarize({"5": job}, 100)
    for hypothesis in ("H1r", "H12", "H13", "H5"):
        assert summary[hypothesis]["rule_outcome"] == "insufficient_seeds", hypothesis


def test_puck_is_outside_h5_scope_in_the_output(job):
    fake = json.loads(json.dumps(job))
    fake["family"] = "puck"
    summary = thread_f.summarize({"5": fake, "6": fake}, 100)
    assert not summary["H5"]["registered_scope"]
    assert summary["H5"]["gated_outcome"] == "outside_registered_scope"
    assert summary["gates"]["checks"]["g1_required"]


def test_main_writes_files_that_merge(job, tmp_path, monkeypatch):
    monkeypatch.setattr(thread_f, "run_job", lambda family, seed, cfg: {**job, "seed_marker": seed})
    inputs = []
    for seed in (5, 6):
        out = tmp_path / f"job{seed}"
        thread_f.main(["--smoke", "--seed-list", str(seed), "--families", "pendulum", "--threads", "1", "--out", str(out)])
        inputs.append(merge_results.load_input(out / "thread_f_smoke.json", "f"))
    assert inputs[0].doc["f_design"]["oracle_gate"] == 0.5
    assert inputs[0].doc["config"]["families"] == ["pendulum"]
    assert not inputs[0].doc["stopped_early"]
    result = merge_results.merge("f", inputs)
    assert result.report["seeds"] == [5, 6]
    assert result.merged["summary"]["pendulum"]["n_seeds"] == 2
    assert result.merged["config"]["seeds"] == [5, 6]


def test_families_with_different_training_merge(job, tmp_path, monkeypatch):
    """Files name only their own family and seed. Training differs per family, and the shared config and f_design must not."""
    monkeypatch.setattr(thread_f, "run_job", lambda family, seed, cfg: job)
    inputs = []
    for family, seed in (("puck", 5), ("pendulum", 6)):
        out = tmp_path / family
        thread_f.main(["--smoke", "--seed-list", str(seed), "--families", family, "--threads", "1", "--out", str(out)])
        inputs.append(merge_results.load_input(out / "thread_f_smoke.json", "f"))
    assert inputs[0].doc["f_design"] == inputs[1].doc["f_design"]
    merged = merge_results.merge("f", inputs).merged
    assert sorted(merged["jobs"]) == ["pendulum", "puck"]
    assert merged["config"]["family_arms"] == thread_f.FAMILY_ARMS


def test_max_minutes_stops_before_starting_a_job(tmp_path):
    thread_f.main(["--smoke", "--seed-list", "5", "--families", "pendulum", "--max-minutes", "0", "--threads", "1", "--out", str(tmp_path)])
    doc = json.loads((tmp_path / "thread_f_smoke.json").read_text(encoding="utf-8"))
    assert doc["stopped_early"] and doc["jobs"]["pendulum"] == {}


@pytest.mark.skipif("drone" not in FAMILIES, reason="the drone family is not in this checkout")
def test_second_family_runs_through_the_job():
    job = thread_f._jsonable(thread_f.run_job("drone", 5, _tiny()))
    assert job["family"] == "drone"
    summary = thread_f.summarize({"5": job, "6": job}, 100)
    assert summary["H5"]["registered_scope"]
    assert not summary["gates"]["checks"]["g1_required"]


def test_puck_runs_through_the_job_at_its_larger_size():
    """Every adaptation method, H13 and H5 on the larger model. Tiny, but with K = 3 under F, on the dual path as at full size."""
    cfg = _tiny()
    cfg.n_eval_envs, cfg.n_val_envs, cfg.n_streams, cfg.n_val_streams = 6, 4, 6, 4  # smoke's values, for puck's mechanisms
    cfg.adapt_steps = 10
    arm = next(a for a in thread_g.ARMS if a.name == "capacity_updates")
    feature_dim = cfg.feature_dim * arm.width_scale
    assert max(cfg.ks) * cfg.adapt_steps <= feature_dim

    job = thread_f._jsonable(thread_f.run_job("puck", 5, cfg))
    assert job["train"]["settings"]["arm"] == "capacity_updates"
    assert job["train"]["settings"]["model_latent"]["feature_dim"] == feature_dim
    assert job["train"]["control"]["stats"]["extra_steps"] == job["train"]["latent"]["stats"]["latent_updates"]
    info = job["tiers"]["interp"]["k"]["3"]["hybrid_eb@latent"]["info"]["eb"]
    assert info["feature_dim"] == feature_dim and info["path"] == "dual"
    summary = thread_f.summarize({"5": job, "6": job}, 100)
    assert summary["gates"]["checks"]["g1_required"]
    assert summary["H5"]["gated_outcome"] == "outside_registered_scope"


@pytest.mark.parametrize("make_cfg", [thread_f.full_config, thread_f.smoke_config])
def test_training_per_family_resolves_as_intended(make_cfg):
    cfg = make_cfg()
    for family in ("pendulum", "drone"):  # the first-round settings, unchanged
        env = make_env(family, 2, "train", 0)
        with_z, no_z, tcfg = thread_f.training_for(family, env, cfg)
        assert tcfg == cfg.train
        assert with_z == ModelConfig(env.obs_dim, env.act_dim, cfg.z_dim, cfg.hidden, cfg.n_hidden, cfg.feature_dim)
        assert no_z == ModelConfig(env.obs_dim, env.act_dim, 0, cfg.hidden, cfg.n_hidden, cfg.feature_dim)

    env = make_env("puck", 2, "train", 0)
    arm = next(a for a in thread_g.ARMS if a.name == "capacity_updates")
    with_z, no_z, tcfg = thread_f.training_for("puck", env, cfg)
    assert with_z == thread_g.arm_model_config(arm, env, cfg, cfg.z_dim)
    assert no_z == thread_g.arm_model_config(arm, env, cfg, 0)
    assert tcfg == thread_g.arm_train_config(arm, cfg)
    assert with_z.feature_dim > cfg.feature_dim and with_z.n_hidden > cfg.n_hidden and tcfg.model_steps > cfg.train.model_steps

    with pytest.raises(ValueError):
        thread_f.family_arm("not_a_family", cfg)


def test_base_settings_are_the_first_round_config():
    a, f = thread_a.full_config(), thread_f.full_config()
    assert f.train == a.train
    for key in ("z_dim", "hidden", "n_hidden", "feature_dim", "train_envs", "train_traj", "train_steps"):
        assert getattr(f, key) == getattr(a, key), key


def test_config_and_design_do_not_depend_on_the_job():
    def doc(seeds, families):
        cfg = thread_f.full_config()
        cfg.seeds, cfg.families = seeds, families
        stored = thread_f._jsonable({"config": asdict(cfg), "f_design": thread_f.f_design(cfg)})
        return json.loads(json.dumps(stored))

    a, b = doc([5], ["puck"]), doc([9], ["drone"])
    assert merge_results.core_config(a, "f") == merge_results.core_config(b, "f")
    assert a["f_design"] == b["f_design"]

    choices = a["f_design"]["fixed_choices"]
    assert choices["g1_status"] == thread_f.G1_STATUS
    settings = choices["family_training_settings"]
    assert set(settings) == {"pendulum", "drone", "puck"}
    assert settings["puck"]["arm"]["name"] == "capacity_updates"
    assert settings["pendulum"]["arm"]["name"] == settings["drone"]["arm"]["name"] == thread_g.BASELINE_ARM
    assert settings["drone"]["train"] == asdict(thread_f.full_config().train)


def test_configs_keep_the_protocol_sizes_and_the_requested_defaults():
    full = thread_f.full_config()
    assert full.seeds == [5, 6, 7, 8, 9]
    assert full.families == ["pendulum", "drone", "puck"]
    assert full.n_eval_envs >= 20 and full.n_streams >= 20 and full.n_val_envs >= 20
    assert full.ks[0] == thread_f.PRIMARY_K
    assert min(full.val_lambdas) < 1e-3 and max(full.val_lambdas) == NO_WRITE_LAMBDA
    assert full.lr_scales[-1] > 2.0  # round one pinned at 2.0
    assert min(full.forgetting_grid) < 0.95 and max(full.forgetting_grid) > 0.999
    assert min(full.surprise_thresholds) < 2.0 and max(full.surprise_thresholds) > 10.0
    assert any(s.kind == thread_f.NO_SWITCH_KIND for s in full.schedules)
    # What merge_results reads from a Thread A style config.
    assert full.n_boot > 0
    labels = [c.label for c in thread_f.conditions_for(full)]
    assert len(labels) == len(set(labels))


def test_design_block_does_not_depend_on_the_job():
    assert thread_f.f_design() == thread_f.f_design()
    text = json.dumps(thread_f.f_design(), allow_nan=False)
    assert "no_worse" in text


def test_fingerprint_covers_the_threads_it_reuses():
    hashes = thread_f.fingerprint()
    for name in ("experiments/thread_f.py", "experiments/thread_a.py", "experiments/thread_b.py", "experiments/thread_g.py", "src/intr/ebayes.py"):
        assert name in hashes
