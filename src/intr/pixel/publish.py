"""Tar the results, and push them to a new branch when GITHUB_TOKEN is set. Standard library only.

Token rules. It is read from the environment at the moment of the push. It is passed only through a credential helper that names the variable,
so it is on no command line, in no remote URL and in no file. Pushes go from a throwaway repository, as a new orphan branch holding only
results, with an explicit refspec, so main and the mission branch cannot be touched. Everything printed goes through redact.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

TOKEN_VAR = "GITHUB_TOKEN"
BRANCH_PREFIX = "results/thread-p-"
# A literal shell snippet. git runs it, the shell expands the variable, and only the variable's name appears in any argument list.
_CREDENTIAL_HELPER = '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
EXCLUDED_FROM_PUSH = ("cache", "checkpoints")


def redact(text: str, token: str | None = None) -> str:
    token = os.environ.get(TOKEN_VAR) if token is None else token
    return text.replace(token, "***") if token else text


def clean_remote(url: str) -> str:
    """Remove any credentials from a URL, and turn a GitHub SSH address into https, which a token can use."""
    url = url.strip()
    match = re.match(r"^git@github\.com:(.+)$", url)
    if match:
        return f"https://github.com/{match.group(1)}"
    return re.sub(r"^(https?://)[^/@]+@", r"\1", url)


def _git(args: list[str], cwd=None, env=None, timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def scrub_origin(repo: Path) -> str | None:
    """If origin holds credentials on disk, rewrite it clean. Returns the clean URL, or None when there is no origin."""
    result = _git(["remote", "get-url", "origin"], cwd=repo)
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    clean = re.sub(r"^(https?://)[^/@]+@", r"\1", url)
    if clean != url:
        _git(["remote", "set-url", "origin", clean], cwd=repo)
    return clean


def make_tar(folder: Path, tar_path: Path) -> Path:
    """Everything in the folder but the rebuildable cache."""
    folder, tar_path = Path(folder), Path(tar_path)
    tmp = tar_path.with_name(tar_path.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(folder, arcname=folder.name, filter=lambda info: None if f"/{folder.name}/cache" in f"/{info.name}" else info)
    os.replace(tmp, tar_path)
    return tar_path


def publish_results(folder: Path, repo: Path, stamp: str, remote: str | None = None, token: str | None = None) -> dict:
    """Returns {"pushed": bool, "branch" or "reason": ...}. Never raises on a push failure, and never includes the token."""
    token = os.environ.get(TOKEN_VAR) if token is None else token
    if not token:
        return {"pushed": False, "reason": f"{TOKEN_VAR} is not set"}
    branch = f"{BRANCH_PREFIX}{stamp}"
    try:
        target = remote if remote is not None else scrub_origin(Path(repo))
        if not target:
            return {"pushed": False, "reason": "the repository has no origin remote"}
        target = clean_remote(target)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "results_repo"
            destination = work / "results" / "thread_p" / stamp
            shutil.copytree(folder, destination, ignore=shutil.ignore_patterns(*EXCLUDED_FROM_PUSH, "*.tar.gz", "*.tmp*"))
            env = {k: v for k, v in os.environ.items() if k != TOKEN_VAR}
            env["GIT_TERMINAL_PROMPT"] = "0"
            for step in (["init", "-q"], ["add", "-A"], ["-c", "user.name=thread-p-runner", "-c", "user.email=thread-p-runner@users.noreply.github.com", "commit", "-q", "-m", f"Thread P results {stamp}"]):
                done = _git(step, cwd=work, env=env)
                if done.returncode != 0:
                    return {"pushed": False, "reason": redact(f"git {step[0]} failed: {done.stderr.strip()}", token)}
            push_env = {**env, TOKEN_VAR: token}
            done = _git(["-c", "credential.helper=", "-c", f"credential.helper={_CREDENTIAL_HELPER}", "push", target, f"HEAD:refs/heads/{branch}"], cwd=work, env=push_env, timeout=900.0)
            if done.returncode != 0:
                return {"pushed": False, "reason": redact(f"git push failed: {done.stderr.strip()}", token)}
    except (OSError, subprocess.SubprocessError) as error:
        return {"pushed": False, "reason": redact(repr(error), token)}
    return {"pushed": True, "branch": branch}
