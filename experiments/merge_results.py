"""Merge per-job result files of one thread into the normal results format.

    python experiments/merge_results.py --thread a --inputs job1/thread_a_full.json job2/thread_a_full.json \
        --prior results/thread_a/thread_a_full.json --out results/thread_a/thread_a_full.json

Refuses (exit 2) if config apart from seeds and families, other shared top-level fields, or the src/ hashes differ
across inputs, or if a seed-family job appears twice. Exits 3 if no input holds a finished job, and writes nothing.

Families have to be excluded along with seeds: every per-job file names only its own family.

The summary comes from the thread's own function, over all jobs, in seed order. Result files store NaN as null,
which the summary functions do not all accept, so jobs are given back their NaNs first. They then see what a full
run's summary sees. Only the summary input is changed; the jobs written out are the ones read in.

A job in --prior identical to an input is reported as already present, which keeps a re-run of the merge harmless.
Different content for the same pair is a refusal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

THREADS = ("a", "b", "c")
THREAD_MARKERS = {"a": "h3_methods", "b": "design_notes", "c": "baselines"}  # a top-level key only that thread writes
ENV_KEYS = ("python", "torch", "numpy", "platform", "torch_threads")
WARN_IF_MIXED = ("python", "torch", "numpy", "torch_threads")
PER_JOB_KEYS = {"config", "jobs", "summary", "metadata", "stopped_early", "selection"}  # all else must match exactly
SELECTION_CHOICE = ("loops", "grad_loops", "lr", "rule")  # what Thread C's selection decides
EXIT_OK, EXIT_REFUSED, EXIT_NOTHING = 0, 2, 3


class MergeRefused(ValueError):
    """The inputs must not be combined."""


class NothingToMerge(ValueError):
    """No input holds a finished job."""


@dataclass
class Input:
    name: str
    sha256: str
    doc: dict
    ci: dict | None = None


@dataclass
class MergeResult:
    merged: dict | None  # None when every job was already in the prior file
    report: dict


def job_pairs(doc: dict) -> list[tuple[str, int]]:
    return [(family, int(seed)) for family, jobs in doc["jobs"].items() for seed in jobs]


def result_path(root: Path, thread: str) -> Path:
    return Path(root) / "results" / f"thread_{thread}" / f"thread_{thread}_full.json"


def load_input(path, thread: str) -> Input:
    path = Path(path)
    raw = path.read_bytes()
    name = f"{path.parent.name}/{path.name}"
    try:
        doc = json.loads(raw)
    except ValueError as err:
        raise MergeRefused(f"{name} is not valid JSON: {err}") from err
    for key in ("config", "metadata", "jobs"):
        if key not in doc:
            raise MergeRefused(f"{name} has no {key!r}, so it is not a results file")
    if THREAD_MARKERS[thread] not in doc or (thread == "b" and "a" not in doc["config"]):
        raise MergeRefused(f"{name} is not a thread {thread} results file")
    source = doc["metadata"].get("source_sha256") or {}
    if not any(f.startswith("src/") for f in source):
        raise MergeRefused(f"{name} records no src/ hashes, so its code cannot be checked")
    for family, jobs in doc["jobs"].items():
        for seed in jobs:
            if not seed.isdecimal() or str(int(seed)) != seed:
                raise MergeRefused(f"{name} has a job under {family} with a bad seed key {seed!r}")
    ci = None
    sidecar = path.parent / "job_info.json"
    if sidecar.exists():
        try:
            ci = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            ci = {"error": "job_info.json was not valid JSON"}
    return Input(name, hashlib.sha256(raw).hexdigest(), doc, ci)


def core_config(doc: dict, thread: str) -> dict:
    """Config without what legitimately differs per job."""
    config = copy.deepcopy(doc["config"])
    holder = config["a"] if thread == "b" else config
    holder.pop("seeds", None)
    holder.pop("families", None)
    return config


def set_scope(config: dict, thread: str, seeds: list[int], families: list[str]) -> None:
    holder = config["a"] if thread == "b" else config
    holder["seeds"] = seeds
    holder["families"] = families


def differences(a, b, path: str = "") -> list[str]:
    """Dotted paths where two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}{key} (on one side only)")
            else:
                out.extend(differences(a[key], b[key], f"{path}{key}."))
        return out
    return [] if a == b else [path.rstrip(".") or "<root>"]


def src_hash_differences(old: dict, new: dict) -> list[str]:
    """Both directions, so an added or removed library file also counts."""
    return sorted(f for f in set(old) | set(new) if f.startswith("src/") and old.get(f) != new.get(f))


def check_compatible(reference: Input, other: Input, thread: str) -> None:
    between = f"between {reference.name} and {other.name}"
    diffs = differences(core_config(reference.doc, thread), core_config(other.doc, thread))
    if diffs:
        raise MergeRefused(f"config differs {between}: {diffs[:8]}")
    shared = lambda doc: {k: v for k, v in doc.items() if k not in PER_JOB_KEYS}
    diffs = differences(shared(reference.doc), shared(other.doc))
    if diffs:
        raise MergeRefused(f"shared fields differ {between}: {diffs[:8]}")
    changed = src_hash_differences(reference.doc["metadata"]["source_sha256"], other.doc["metadata"]["source_sha256"])
    if changed:
        raise MergeRefused(f"src/ hashes differ {between}: {changed}")


def merge_selection(items: list[Input], warnings: list[str]) -> dict:
    """Thread C selects per family inside each job's process, so runs must agree on what was selected.

    Jobs with different choices are not comparable. Same choices with different scores are only a sign of rounding
    differences between machines, so that is a warning. wall_s always differs.
    """
    chosen: dict[str, tuple[str, dict]] = {}
    for item in items:
        for family in sorted({f for f, _ in job_pairs(item.doc)}):
            entry = item.doc.get("selection", {}).get(family)
            if entry is None:
                raise MergeRefused(f"{item.name} has jobs on {family} but no selection for it")
            if family not in chosen:
                chosen[family] = (item.name, entry)
                continue
            first_name, first = chosen[family]
            for key in SELECTION_CHOICE:
                if first.get(key) != entry.get(key):
                    raise MergeRefused(
                        f"selection for {family} differs between {first_name} and {item.name} on {key}: "
                        f"{first.get(key)!r} and {entry.get(key)!r}"
                    )
            if first.get("table") != entry.get("table"):
                warnings.append(f"selection scores for {family} differ between {first_name} and {item.name}; same choices")
    return {family: chosen[family][1] for family in sorted(chosen)}


def revive_nan(x):
    """Undo _jsonable's NaN to null. Harmless on real nulls: summaries treat NaN and None alike where None occurs."""
    if x is None:
        return float("nan")
    if isinstance(x, dict):
        return {k: revive_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [revive_nan(v) for v in x]
    return x


def recompute_summary(thread: str, jobs: dict, config: dict) -> dict:
    """The thread's own summary function, exactly as its main() calls it, over {family: {seed: job}}."""
    module = importlib.import_module(f"thread_{thread}")
    revived = revive_nan(jobs)
    if thread == "a":
        summary = {f: module.summarize(j, config["n_boot"]) for f, j in revived.items() if j}
    elif thread == "b":
        summary = {f: module.summarize(j, config["a"]["n_boot"]) for f, j in revived.items() if j}
    else:
        cfg = module.ThreadCConfig(**{**config, "train": module.LoopTrainConfig(**config["train"])})
        summary = {f: module.summarize(f, j, cfg) for f, j in revived.items() if j}
    return module._jsonable(summary)


def missing_pairs(present, expected_seeds, expected_families) -> list[dict]:
    """Every pair of the requested or present seeds and families that has no job. Never hidden."""
    present = set(present)
    seeds = sorted(set(expected_seeds) | {s for _, s in present})
    families = sorted(set(expected_families) | {f for f, _ in present})
    return [{"seed": s, "family": f} for s in seeds for f in families if (f, s) not in present]


def job_record(item: Input) -> dict:
    meta = item.doc["metadata"]
    return {
        "input": item.name,
        **{key: meta.get(key) for key in ENV_KEYS},
        "started_utc": meta.get("started_utc"),
        "elapsed_s": meta.get("elapsed_s"),
        "argv": meta.get("argv"),
        "source_sha256": meta["source_sha256"],
        "ci": item.ci,
    }


def job_records(item: Input) -> dict[tuple[str, int], dict]:
    """A file that is itself a merge keeps the records of the files it came from."""
    carried = item.doc["metadata"].get("merge", {}).get("jobs", {})
    base = job_record(item)
    return {(f, s): carried.get(f, {}).get(str(s), base) for f, s in job_pairs(item.doc)}


def input_records(item: Input) -> list[dict]:
    carried = item.doc["metadata"].get("merge", {}).get("inputs")
    if carried is not None:
        return list(carried)
    meta = item.doc["metadata"]
    return [{
        "name": item.name,
        "sha256": item.sha256,
        "jobs": [{"family": f, "seed": s} for f, s in sorted(job_pairs(item.doc))],
        "started_utc": meta.get("started_utc"),
        "elapsed_s": meta.get("elapsed_s"),
    }]


def _canonical(x) -> str:
    return json.dumps(x, sort_keys=True)


def _single_or_mixed(values):
    values = list(values)
    return values[0] if all(v == values[0] for v in values) else "mixed"


def merge(thread, inputs, prior=None, expected_seeds=(), expected_families=(), summarize=None) -> MergeResult:
    without_jobs = [i.name for i in inputs if not job_pairs(i.doc)]
    fresh = [i for i in inputs if job_pairs(i.doc)]
    if not fresh:
        raise NothingToMerge(f"no finished jobs in {len(inputs)} input file(s)")
    if prior is not None and not job_pairs(prior.doc):
        prior = None
    known = ([prior] if prior else []) + fresh
    reference = known[0]
    for other in known[1:]:
        check_compatible(reference, other, thread)

    jobs: dict[tuple[str, int], dict] = {}
    records: dict[tuple[str, int], dict] = {}
    if prior:
        jobs.update({(f, s): prior.doc["jobs"][f][str(s)] for f, s in job_pairs(prior.doc)})
        records.update(job_records(prior))
    added, already, added_from, contributors = [], [], {}, ([prior] if prior else [])
    for item in fresh:
        item_records = job_records(item)
        contributed = False
        for pair in job_pairs(item.doc):
            family, seed = pair
            job = item.doc["jobs"][family][str(seed)]
            if pair in added_from:
                raise MergeRefused(f"seed {seed} on {family} appears in both {added_from[pair]} and {item.name}")
            if pair in jobs:
                if _canonical(jobs[pair]) != _canonical(job):
                    raise MergeRefused(f"seed {seed} on {family} is already in {prior.name} with different content; see {item.name}")
                already.append(pair)
                continue
            jobs[pair] = job
            records[pair] = item_records[pair]
            added_from[pair] = item.name
            added.append(pair)
            contributed = True
        if contributed:
            contributors.append(item)

    seeds = sorted({s for _, s in jobs})
    families = sorted({f for f, _ in jobs})
    missing = missing_pairs(jobs, expected_seeds, expected_families)
    report = {
        "status": "merged" if added else "already_merged",
        "thread": thread,
        "added": [{"seed": s, "family": f} for f, s in sorted(added, key=lambda p: (p[1], p[0]))],
        "already_present": [{"seed": s, "family": f} for f, s in sorted(already, key=lambda p: (p[1], p[0]))],
        "seeds": seeds,
        "families": families,
        "missing": missing,
        "inputs_without_jobs": without_jobs,
        "warnings": [],
    }
    if not added:
        return MergeResult(None, report)

    warnings = report["warnings"]
    selection = merge_selection(contributors, warnings) if thread == "c" else None
    metas = list(records.values())
    for key in WARN_IF_MIXED:
        if _single_or_mixed(m.get(key) for m in metas) == "mixed":
            warnings.append(f"{key} differs across jobs; see metadata.merge.jobs")
    source_files = sorted({f for m in metas for f in m["source_sha256"]})
    source = {f: _single_or_mixed(m["source_sha256"].get(f) for m in metas) for f in source_files}
    mixed_files = [f for f, value in source.items() if value == "mixed"]
    if mixed_files:
        warnings.append(f"not refused, since only src/ is compared, but these differ across jobs: {mixed_files}")

    inputs_list = [record for item in contributors for record in input_records(item)]
    started = [m["started_utc"] for m in metas if m.get("started_utc")]
    jobs_out = {f: {str(s): jobs[(f, s)] for s in seeds if (f, s) in jobs} for f in families}
    records_out = {f: {str(s): records[(f, s)] for s in seeds if (f, s) in records} for f in families}
    config = copy.deepcopy(reference.doc["config"])
    set_scope(config, thread, seeds, families)
    merged = dict(reference.doc)
    merged["config"] = config
    merged["metadata"] = {
        "started_utc": min(started) if started else None,
        **{key: _single_or_mixed(m.get(key) for m in metas) for key in ENV_KEYS},
        "argv": _single_or_mixed(m.get("argv") for m in metas),
        "source_sha256": source,
        "elapsed_s": sum(r.get("elapsed_s") or 0.0 for r in inputs_list),
        "merge": {
            "merged_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": seeds,
            "families": families,
            "missing": missing,
            "inputs": inputs_list,
            "inputs_without_jobs": without_jobs,
            "jobs": records_out,
            "warnings": warnings,
        },
    }
    merged["jobs"] = jobs_out
    if selection is not None:
        merged["selection"] = selection
    merged["summary"] = (summarize or recompute_summary)(thread, jobs_out, config)
    merged["stopped_early"] = any(item.doc.get("stopped_early") for item in contributors)
    return MergeResult(merged, report)


def _pair_text(pairs: list[dict]) -> str:
    return ", ".join(f"seed {p['seed']} on {p['family']}" for p in pairs) or "none"


def commit_message(report: dict, note: str = "", run_url: str = "", sha: str = "") -> str:
    added_seeds = sorted({p["seed"] for p in report["added"]})
    added_families = sorted({p["family"] for p in report["added"]})
    lines = [
        f"Add thread {report['thread']} results: seeds {', '.join(map(str, added_seeds))}; {', '.join(added_families)}",
        "",
        f"Added: {_pair_text(report['added'])}",
        f"Seeds now present: {', '.join(map(str, report['seeds']))}",
        f"Families now present: {', '.join(report['families'])}",
        f"Missing: {_pair_text(report['missing'])}",
    ]
    if report["warnings"]:
        lines.extend(f"Warning: {w}" for w in report["warnings"])
    if note:
        lines.append(f"Note: {note}")
    if run_url:
        lines.append(f"Run: {run_url}")
    if sha:
        lines.append(f"Code: {sha}")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict) -> str:
    lines = [f"### Merge of thread {report['thread']}: {report['status']}", ""]
    if report.get("reason"):
        lines.append(f"Reason: {report['reason']}")
    for label, key in (("Added", "added"), ("Already present", "already_present"), ("Missing", "missing")):
        if key in report:
            lines.append(f"- {label}: {_pair_text(report[key])}")
    if "seeds" in report:
        lines.append(f"- Seeds now present: {', '.join(map(str, report['seeds'])) or 'none'}")
        lines.append(f"- Families now present: {', '.join(report['families']) or 'none'}")
    if report.get("inputs_without_jobs"):
        lines.append(f"- Inputs with no finished job: {', '.join(report['inputs_without_jobs'])}")
    if "size_bytes" in report:
        lines.append(f"- Merged file size: {report['size_bytes'] / 1e6:.2f} MB")
    lines.extend(f"- Warning: {w}" for w in report.get("warnings", []))
    return "\n".join(lines) + "\n"


def main(argv=None, summarize=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thread", required=True, choices=THREADS)
    parser.add_argument("--inputs", nargs="*", default=[], type=Path, help="per-job result files")
    parser.add_argument("--prior", type=Path, default=None, help="merged file to extend; may not exist yet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=[], help="report these if they have no job")
    parser.add_argument("--expected-families", nargs="*", default=[], help="report these if they have no job")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    parser.add_argument("--markdown", type=Path, default=None, help="write a readable report here")
    parser.add_argument("--commit-message", type=Path, default=None, help="write a commit message here if jobs were added")
    parser.add_argument("--note", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--sha", default="")
    args = parser.parse_args(argv)

    report = {"thread": args.thread}
    code = EXIT_OK
    try:
        inputs = [load_input(p, args.thread) for p in sorted(args.inputs)]
        prior = load_input(args.prior, args.thread) if args.prior is not None and args.prior.exists() else None
        result = merge(args.thread, inputs, prior, args.expected_seeds, args.expected_families, summarize)
    except MergeRefused as err:
        report.update(status="refused", reason=str(err))
        code = EXIT_REFUSED
    except NothingToMerge as err:
        report.update(
            status="nothing_to_merge", reason=str(err), added=[],
            missing=missing_pairs([], args.expected_seeds, args.expected_families),
            inputs_without_jobs=[str(p) for p in args.inputs],
        )
        code = EXIT_NOTHING
    else:
        report = result.report
        if result.merged is not None:
            text = json.dumps(result.merged, indent=1, allow_nan=False)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.out.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(args.out)
            report["size_bytes"] = len(text)
            if args.commit_message is not None:
                args.commit_message.write_text(commit_message(report, args.note, args.run_url, args.sha), encoding="utf-8")

    markdown = render_markdown(report)
    print(markdown, file=sys.stderr if code else sys.stdout)
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(markdown, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
