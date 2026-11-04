"""Merging per-job results and validating run requests, on small fabricated files."""

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import ci_tools  # noqa: E402
import merge_results  # noqa: E402
from merge_results import MergeRefused, NothingToMerge, load_input, merge  # noqa: E402

MARKERS = {"a": "h3_methods", "b": "design_notes", "c": "baselines"}


def make_doc(thread="a", seed=0, family="pendulum", job=None, n_boot=50, src="aaa", script="bbb", python="3.13.1", **config_extra):
    settings = {"mode": "test", "seeds": [seed], "families": [family], "n_boot": n_boot, **config_extra}
    doc = {
        "config": {"mode": "test", "a": settings} if thread == "b" else settings,
        MARKERS[thread]: ["shared"],
        "metadata": {
            "started_utc": f"2026-11-04T0{seed % 10}:00:00+00:00",
            "python": python,
            "torch": "2.11.0+cpu",
            "numpy": "2.3.0",
            "platform": "Linux",
            "torch_threads": 2,
            "argv": ["--full", "--seed-list", str(seed), "--families", family],
            "source_sha256": {"src/intr/envs.py": src, f"experiments/thread_{thread}.py": script},
            "elapsed_s": 60.0,
        },
        "jobs": {family: {str(seed): {"value": seed} if job is None else job}},
        "summary": {"stale": True},
        "stopped_early": False,
    }
    if thread == "c":
        doc["selection"] = {
            family: {"loops": 4, "grad_loops": 2, "lr": {"full": 0.001}, "rule": "r", "table": [{"score": 0.5}], "wall_s": seed + 1.0}
        }
    return doc


def put(tmp_path, name, doc, thread="a"):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"thread_{thread}_full.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def fake_summary(thread, jobs, config):
    return {"jobs": {family: sorted(seeds) for family, seeds in jobs.items()}, "from_merge": True}


def run(tmp_path, docs, thread="a", prior=None, **kwargs):
    inputs = [load_input(put(tmp_path, f"job{i}", doc, thread), thread) for i, doc in enumerate(docs)]
    prior_input = None if prior is None else load_input(put(tmp_path, "prior", prior, thread), thread)
    return merge(thread, inputs, prior_input, summarize=fake_summary, **kwargs)


# ---------------------------------------------------------------- merging


def test_merge_writes_the_normal_format_with_what_is_actually_present(tmp_path):
    result = run(tmp_path, [make_doc(seed=10, family="puck"), make_doc(seed=2, family="puck"), make_doc(seed=2, family="pendulum")])
    merged = result.merged
    assert merged["config"]["seeds"] == [2, 10]
    assert merged["config"]["families"] == ["pendulum", "puck"]
    assert list(merged["jobs"]["puck"]) == ["2", "10"]  # numeric order, not string order
    assert merged["jobs"]["pendulum"] == {"2": {"value": 2}}
    assert merged["summary"]["from_merge"]  # recomputed, not carried from an input
    assert merged["h3_methods"] == ["shared"]
    assert merged["stopped_early"] is False
    assert merged["metadata"]["elapsed_s"] == 180.0
    assert merged["metadata"]["python"] == "3.13.1"
    assert merged["metadata"]["argv"] == "mixed"


def test_merge_records_inputs_and_per_job_metadata(tmp_path):
    merged = run(tmp_path, [make_doc(seed=0), make_doc(seed=1)]).merged
    record = merged["metadata"]["merge"]
    assert [i["name"] for i in record["inputs"]] == ["job0/thread_a_full.json", "job1/thread_a_full.json"]
    assert all(len(i["sha256"]) == 64 for i in record["inputs"])
    assert record["inputs"][1]["jobs"] == [{"family": "pendulum", "seed": 1}]
    job = record["jobs"]["pendulum"]["1"]
    assert job["input"] == "job1/thread_a_full.json"
    assert job["argv"] == ["--full", "--seed-list", "1", "--families", "pendulum"]
    assert job["source_sha256"]["src/intr/envs.py"] == "aaa"


def test_job_info_next_to_an_input_is_recorded(tmp_path):
    path = put(tmp_path, "job", make_doc())
    (path.parent / "job_info.json").write_text(json.dumps({"minutes": 12.5}), encoding="utf-8")
    merged = merge("a", [load_input(path, "a")], summarize=fake_summary).merged
    assert merged["metadata"]["merge"]["jobs"]["pendulum"]["0"]["ci"] == {"minutes": 12.5}


def test_missing_pairs_are_reported(tmp_path):
    result = run(tmp_path, [make_doc(seed=0, family="pendulum"), make_doc(seed=1, family="puck")], expected_seeds=[0, 1, 2], expected_families=["pendulum", "puck"])
    expected = [
        {"seed": 0, "family": "puck"},
        {"seed": 1, "family": "pendulum"},
        {"seed": 2, "family": "pendulum"},
        {"seed": 2, "family": "puck"},
    ]
    assert result.report["missing"] == expected
    assert result.merged["metadata"]["merge"]["missing"] == expected
    assert result.merged["config"]["seeds"] == [0, 1]  # only what is present
    message = merge_results.commit_message(result.report, note="pilot", run_url="http://run", sha="abc")
    assert "Missing: seed 0 on puck, seed 1 on pendulum, seed 2 on pendulum, seed 2 on puck" in message
    assert "Seeds now present: 0, 1" in message
    assert "Families now present: pendulum, puck" in message
    assert "Note: pilot" in message
    assert "seed 2 on puck" in merge_results.render_markdown(result.report)


def test_commit_message_says_none_when_nothing_is_missing(tmp_path):
    result = run(tmp_path, [make_doc(seed=0), make_doc(seed=0, family="puck")], expected_seeds=[0], expected_families=["pendulum", "puck"])
    message = merge_results.commit_message(result.report)
    assert message.startswith("Add thread a results: seeds 0; pendulum, puck\n")
    assert "Missing: none" in message


def test_different_environment_is_recorded_but_not_refused(tmp_path):
    result = run(tmp_path, [make_doc(seed=0), make_doc(seed=1, python="3.12.0", script="zzz")])
    assert result.merged["metadata"]["python"] == "mixed"
    assert result.merged["metadata"]["source_sha256"]["experiments/thread_a.py"] == "mixed"
    assert result.merged["metadata"]["source_sha256"]["src/intr/envs.py"] == "aaa"
    warnings = " ".join(result.report["warnings"])
    assert "python differs" in warnings
    assert "experiments/thread_a.py" in warnings


# ---------------------------------------------------------------- refusals


def test_refuses_different_config(tmp_path):
    with pytest.raises(MergeRefused, match="config differs.*z_dim"):
        run(tmp_path, [make_doc(seed=0, z_dim=8), make_doc(seed=1, z_dim=4)])


def test_refuses_config_present_on_one_side(tmp_path):
    with pytest.raises(MergeRefused, match="config differs"):
        run(tmp_path, [make_doc(seed=0), make_doc(seed=1, z_dim=4)])


def test_refuses_different_nested_config_in_thread_b_but_allows_its_seeds_and_families(tmp_path):
    allowed = run(tmp_path, [make_doc("b", seed=0), make_doc("b", seed=1, family="puck")], thread="b").merged
    assert allowed["config"]["a"]["seeds"] == [0, 1]
    assert allowed["config"]["a"]["families"] == ["pendulum", "puck"]
    with pytest.raises(MergeRefused, match="config differs"):
        run(tmp_path, [make_doc("b", seed=0, hidden=8), make_doc("b", seed=1, hidden=16)], thread="b")


def test_refuses_different_shared_fields(tmp_path):
    other = make_doc(seed=1)
    other["h3_methods"] = ["changed"]
    with pytest.raises(MergeRefused, match="shared fields differ"):
        run(tmp_path, [make_doc(seed=0), other])


def test_refuses_different_src_hashes(tmp_path):
    with pytest.raises(MergeRefused, match="src/ hashes differ"):
        run(tmp_path, [make_doc(seed=0), make_doc(seed=1, src="changed")])


def test_refuses_a_src_file_on_one_side_only(tmp_path):
    other = make_doc(seed=1)
    other["metadata"]["source_sha256"]["src/intr/new.py"] = "ccc"
    with pytest.raises(MergeRefused, match="new.py"):
        run(tmp_path, [make_doc(seed=0), other])


def test_refuses_duplicate_jobs_even_when_identical(tmp_path):
    with pytest.raises(MergeRefused, match="seed 0 on pendulum appears in both"):
        run(tmp_path, [make_doc(seed=0), make_doc(seed=0)])


def test_refuses_a_file_of_another_thread(tmp_path):
    path = put(tmp_path, "job", make_doc("c"), "a")
    with pytest.raises(MergeRefused, match="not a thread a results file"):
        load_input(path, "a")


def test_refuses_invalid_json_and_missing_hashes(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(MergeRefused, match="not valid JSON"):
        load_input(bad, "a")
    doc = make_doc()
    doc["metadata"]["source_sha256"] = {"experiments/thread_a.py": "bbb"}
    with pytest.raises(MergeRefused, match="no src/ hashes"):
        load_input(put(tmp_path, "job", doc), "a")


# ---------------------------------------------------------------- jobs that did not finish


def test_nothing_to_merge_when_no_job_finished(tmp_path):
    empty = make_doc("c")
    empty["jobs"] = {"pendulum": {}}  # Thread C saves after selection, so a job that times out leaves this
    with pytest.raises(NothingToMerge):
        run(tmp_path, [empty], thread="c")
    with pytest.raises(NothingToMerge):
        merge("a", [], summarize=fake_summary)


def test_input_without_jobs_is_recorded_and_skipped(tmp_path):
    empty = make_doc("c", seed=1, family="puck", z_dim=99)  # its config must not matter
    empty["jobs"] = {"puck": {}}
    merged = run(tmp_path, [make_doc("c", seed=0), empty], thread="c").merged
    assert merged["metadata"]["merge"]["inputs_without_jobs"] == ["job1/thread_c_full.json"]
    assert merged["config"]["families"] == ["pendulum"]
    assert list(merged["selection"]) == ["pendulum"]


def test_main_with_no_input_writes_nothing_and_exits_3(tmp_path, capsys):
    out = tmp_path / "results" / "thread_a_full.json"
    report = tmp_path / "report.json"
    code = merge_results.main(
        ["--thread", "a", "--inputs", "--out", str(out), "--report", str(report), "--expected-seeds", "0", "--expected-families", "puck"]
    )
    assert code == merge_results.EXIT_NOTHING
    assert not out.exists()
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "nothing_to_merge"
    assert written["missing"] == [{"seed": 0, "family": "puck"}]
    assert "seed 0 on puck" in capsys.readouterr().err


def test_main_refusal_exits_2_and_leaves_the_output_alone(tmp_path):
    out = tmp_path / "merged.json"
    out.write_text("unchanged", encoding="utf-8")
    paths = [str(put(tmp_path, "a", make_doc(seed=0))), str(put(tmp_path, "b", make_doc(seed=0)))]
    assert merge_results.main(["--thread", "a", "--inputs", *paths, "--out", str(out)], summarize=fake_summary) == merge_results.EXIT_REFUSED
    assert out.read_text(encoding="utf-8") == "unchanged"


def test_main_writes_strict_json_and_a_commit_message(tmp_path):
    out = tmp_path / "results" / "thread_a" / "thread_a_full.json"
    message = tmp_path / "message.txt"
    report = tmp_path / "report.json"
    path = put(tmp_path, "job", make_doc())
    code = merge_results.main(
        ["--thread", "a", "--inputs", str(path), "--out", str(out), "--commit-message", str(message), "--report", str(report), "--note=pilot"],
        summarize=fake_summary,
    )
    assert code == merge_results.EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["jobs"]["pendulum"] == {"0": {"value": 0}}
    assert not out.with_suffix(".tmp").exists()
    assert "Note: pilot" in message.read_text(encoding="utf-8")
    assert json.loads(report.read_text(encoding="utf-8"))["size_bytes"] > 0


# ---------------------------------------------------------------- combining with committed results


def test_later_request_combines_with_committed_results(tmp_path):
    first = run(tmp_path / "one", [make_doc(seed=0)]).merged
    second = run(tmp_path / "two", [make_doc(seed=1)], prior=first)
    merged = second.merged
    assert merged["config"]["seeds"] == [0, 1]
    assert list(merged["jobs"]["pendulum"]) == ["0", "1"]
    assert second.report["added"] == [{"seed": 1, "family": "pendulum"}]
    record = merged["metadata"]["merge"]
    assert len(record["inputs"]) == 2  # the first merge's input is kept
    assert record["jobs"]["pendulum"]["0"]["input"] == "job0/thread_a_full.json"  # not the merged file
    assert merged["metadata"]["elapsed_s"] == 120.0


def test_prior_is_held_to_the_same_refusals(tmp_path):
    first = run(tmp_path / "one", [make_doc(seed=0)]).merged
    with pytest.raises(MergeRefused, match="src/ hashes differ"):
        run(tmp_path / "two", [make_doc(seed=1, src="changed")], prior=first)
    with pytest.raises(MergeRefused, match="config differs"):
        run(tmp_path / "three", [make_doc(seed=1, z_dim=4)], prior=first)


def test_prior_with_different_content_for_a_job_is_a_refusal(tmp_path):
    first = run(tmp_path / "one", [make_doc(seed=0)]).merged
    with pytest.raises(MergeRefused, match="already in"):
        run(tmp_path / "two", [make_doc(seed=0, job={"value": "other"})], prior=first)


def test_rerunning_the_merge_with_the_same_job_changes_nothing(tmp_path):
    first = run(tmp_path / "one", [make_doc(seed=0)]).merged
    again = run(tmp_path / "two", [make_doc(seed=0)], prior=first)
    assert again.merged is None
    assert again.report["status"] == "already_merged"
    assert again.report["already_present"] == [{"seed": 0, "family": "pendulum"}]


# ---------------------------------------------------------------- Thread C selection


def test_thread_c_selection_comes_from_each_family(tmp_path):
    merged = run(tmp_path, [make_doc("c", seed=0), make_doc("c", seed=1), make_doc("c", seed=0, family="puck")], thread="c").merged
    assert list(merged["selection"]) == ["pendulum", "puck"]


def test_thread_c_refuses_different_selected_hyperparameters(tmp_path):
    other = make_doc("c", seed=1)
    other["selection"]["pendulum"]["loops"] = 8
    with pytest.raises(MergeRefused, match="selection for pendulum differs.*loops"):
        run(tmp_path, [make_doc("c", seed=0), other], thread="c")


def test_thread_c_warns_on_different_selection_scores_with_the_same_choice(tmp_path):
    other = make_doc("c", seed=1)
    other["selection"]["pendulum"]["table"] = [{"score": 0.50001}]
    result = run(tmp_path, [make_doc("c", seed=0), other], thread="c")
    assert result.merged is not None
    assert any("selection scores" in w for w in result.report["warnings"])
    assert not any("selection scores" in w for w in run(tmp_path / "same", [make_doc("c", seed=0), make_doc("c", seed=1)], thread="c").report["warnings"])


def test_thread_c_refuses_jobs_without_a_selection(tmp_path):
    doc = make_doc("c")
    doc["selection"] = {}
    with pytest.raises(MergeRefused, match="no selection"):
        run(tmp_path, [doc], thread="c")


# ---------------------------------------------------------------- each thread's own summary


def _thread_a_job(base):
    """Just enough for thread_a.summarize: one tier, one K, the primary horizon, no H3 tiers."""
    import thread_a

    cells = {c.label: {"open_loop": {"h10": base + i / 10}} for i, c in enumerate(thread_a.CONDITIONS)}
    return {"tiers": {"interp": {"k": {"1": cells}}}, "h3": {}}


def test_summary_is_thread_a_own_function_over_all_jobs(tmp_path):
    import thread_a

    jobs = {"2": _thread_a_job(1.0), "10": _thread_a_job(2.0)}
    docs = [make_doc(seed=10, job=jobs["10"]), make_doc(seed=2, job=jobs["2"])]
    inputs = [load_input(put(tmp_path, f"job{i}", doc), "a") for i, doc in enumerate(docs)]
    merged = merge("a", inputs).merged
    assert merged["summary"]["pendulum"] == thread_a._jsonable(thread_a.summarize(jobs, 50))
    assert merged["summary"]["pendulum"]["means"]["interp"]["1"]["frozen@latent"]["per_seed_primary"] == [1.0, 2.0]
    json.dumps(merged, allow_nan=False)


def test_a_null_metric_reaches_the_summary_as_nan_and_stays_null_in_the_jobs(tmp_path):
    """A full run's summary sees NaN, and np.mean raises on the null a results file stores in its place."""
    job = _thread_a_job(1.0)
    job["tiers"]["interp"]["k"]["1"]["rls@latent"]["open_loop"]["h10"] = None
    docs = [make_doc(seed=0, job=job), make_doc(seed=1, job=_thread_a_job(2.0))]
    inputs = [load_input(put(tmp_path, f"job{i}", doc), "a") for i, doc in enumerate(docs)]
    merged = merge("a", inputs).merged
    assert merged["jobs"]["pendulum"]["0"]["tiers"]["interp"]["k"]["1"]["rls@latent"]["open_loop"]["h10"] is None
    row = merged["summary"]["pendulum"]["means"]["interp"]["1"]["rls@latent"]
    assert row["h10"] is None
    assert row["per_seed_primary"][0] is None
    json.dumps(merged, allow_nan=False)


def test_thread_b_summary_gets_its_nested_n_boot(monkeypatch):
    import thread_b

    calls = []
    monkeypatch.setattr(thread_b, "summarize", lambda jobs, n_boot: calls.append(n_boot) or {"seeds": list(jobs)})
    out = merge_results.recompute_summary("b", {"puck": {"0": {}}, "pendulum": {}}, {"a": {"n_boot": 7}})
    assert calls == [7]
    assert out == {"puck": {"seeds": ["0"]}}


def test_thread_c_summary_gets_a_real_config_rebuilt_from_json(monkeypatch):
    import thread_c

    seen = {}

    def fake(family, jobs, cfg):
        seen.update(family=family, steps=cfg.train.steps, n_boot=cfg.n_boot, horizons=cfg.at_trained_horizons)
        return {"ok": True}

    monkeypatch.setattr(thread_c, "summarize", fake)
    config = json.loads(json.dumps(dataclasses.asdict(thread_c.full_config())))
    assert merge_results.recompute_summary("c", {"puck": {"0": {}}}, config) == {"puck": {"ok": True}}
    full = thread_c.full_config()
    assert seen == {"family": "puck", "steps": full.train.steps, "n_boot": full.n_boot, "horizons": full.at_trained_horizons}


# ---------------------------------------------------------------- run requests


def test_valid_request_builds_the_matrix():
    checked = ci_tools.validate_request({"thread": "a", "seeds": [0, 3], "families": ["pendulum", "puck"], "note": "pilot"})
    outputs = ci_tools.matrix_outputs(checked)
    assert outputs["seeds"] == "[0,3]"
    assert outputs["families"] == '["pendulum","puck"]'
    assert outputs["seeds_args"] == "0 3"
    assert outputs["families_args"] == "pendulum puck"
    assert outputs["note"] == "pilot"
    assert ci_tools.validate_request({"thread": "c", "seeds": [1], "families": ["puck"]})["note"] == ""


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"seeds": [0], "families": ["puck"]},
        {"thread": "d", "seeds": [0], "families": ["puck"]},
        {"thread": "a", "seeds": [], "families": ["puck"]},
        {"thread": "a", "seeds": 0, "families": ["puck"]},
        {"thread": "a", "seeds": [True], "families": ["puck"]},
        {"thread": "a", "seeds": ["0"], "families": ["puck"]},
        {"thread": "a", "seeds": [-1], "families": ["puck"]},
        {"thread": "a", "seeds": [1000], "families": ["puck"]},
        {"thread": "a", "seeds": [0, 0], "families": ["puck"]},
        {"thread": "a", "seeds": [0], "families": []},
        {"thread": "a", "seeds": [0], "families": ["cartpole"]},
        {"thread": "a", "seeds": [0], "families": ["puck", "puck"]},
        {"thread": "a", "seeds": list(range(21)), "families": ["pendulum", "puck"]},
        {"thread": "a", "seeds": [0], "families": ["puck"], "note": "two\nlines"},
        {"thread": "a", "seeds": [0], "families": ["puck"], "note": "x" * 201},
        {"thread": "a", "seeds": [0], "families": ["puck"], "mode": "smoke"},
    ],
)
def test_bad_requests_are_refused(raw):
    with pytest.raises(ci_tools.RequestError):
        ci_tools.validate_request(raw)


def _repo_with_results(tmp_path, jobs):
    (tmp_path / "src" / "intr").mkdir(parents=True)
    (tmp_path / "src" / "intr" / "envs.py").write_text("code\n", encoding="utf-8")
    path = merge_results.result_path(tmp_path, "a")
    path.parent.mkdir(parents=True)
    doc = make_doc()
    doc["jobs"] = jobs
    doc["metadata"]["source_sha256"] = ci_tools.src_fingerprint(tmp_path)
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_preflight_refuses_jobs_already_committed(tmp_path):
    _repo_with_results(tmp_path, {"pendulum": {"0": {}}})
    ci_tools.preflight({"thread": "a", "seeds": [0], "families": ["puck"], "note": ""}, tmp_path)
    ci_tools.preflight({"thread": "a", "seeds": [1], "families": ["pendulum"], "note": ""}, tmp_path)
    with pytest.raises(ci_tools.RequestError, match="already holds"):
        ci_tools.preflight({"thread": "a", "seeds": [0, 1], "families": ["pendulum"], "note": ""}, tmp_path)


def test_preflight_refuses_changed_library_code(tmp_path):
    _repo_with_results(tmp_path, {"pendulum": {"0": {}}})
    (tmp_path / "src" / "intr" / "envs.py").write_text("different\n", encoding="utf-8")
    with pytest.raises(ci_tools.RequestError, match="src/ differs"):
        ci_tools.preflight({"thread": "a", "seeds": [1], "families": ["pendulum"], "note": ""}, tmp_path)


def test_preflight_passes_without_committed_results(tmp_path):
    ci_tools.preflight({"thread": "b", "seeds": [0], "families": ["puck"], "note": ""}, tmp_path)


def test_fingerprint_matches_the_one_experiments_record():
    import thread_a

    expected = {k: v for k, v in thread_a.source_fingerprint().items() if k.startswith("src/")}
    assert ci_tools.src_fingerprint(ROOT) == expected


def test_request_command_writes_outputs_and_fails_on_a_bad_file(tmp_path):
    request = tmp_path / "request.json"
    output = tmp_path / "github_output"
    request.write_text('{"thread": "a", "seeds": [0], "families": ["pendulum", "puck"], "note": "pilot"}', encoding="utf-8")
    assert ci_tools.main(["request", "--file", str(request), "--root", str(tmp_path), "--github-output", str(output)]) == 0
    assert "seeds=[0]\n" in output.read_text(encoding="utf-8")
    request.write_text("{broken", encoding="utf-8")
    assert ci_tools.main(["request", "--file", str(request), "--root", str(tmp_path)]) == 1
    assert ci_tools.main(["request", "--file", str(tmp_path / "absent.json"), "--root", str(tmp_path)]) == 1


# ---------------------------------------------------------------- minutes


def test_minutes_table_counts_each_job_and_a_running_one_up_to_now():
    payload = {
        "jobs": [
            {"name": "tests", "conclusion": "success", "started_at": "2026-11-04T00:00:00Z", "completed_at": "2026-11-04T00:02:30Z"},
            {"name": "skipped", "conclusion": "skipped", "started_at": None, "completed_at": None},
            {"name": "merge", "conclusion": None, "status": "in_progress", "started_at": "2026-11-04T00:10:00Z", "completed_at": None},
        ]
    }
    table = ci_tools.minutes_table(payload, now=datetime(2026, 11, 4, 0, 11, tzinfo=timezone.utc))
    assert "| tests | success | 2.5 | 3 |" in table
    assert "| skipped | skipped | 0.0 | 0 |" in table
    assert "| merge | in_progress | 1.0 | 1 |" in table
    assert "**3.5**" in table
    assert "**4**" in table
