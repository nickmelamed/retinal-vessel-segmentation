#!/usr/bin/env python3
"""Keep Claude working until the repo's fast checks pass.

Runs as a Claude Code Stop hook. When there are changes since the last
passing check, it runs each command in ``.claude/gate-commands`` (one per
line) plus a check that tests were not weakened. If anything fails it exits
with code 2, which sends the output back to Claude and keeps the turn going.
After three failed rounds in a row it lets Claude stop so it can report the
problem to the owner instead of looping.

Set ``AGENT_ALLOW_TEST_CHANGES=1`` in the environment that launches Claude
Code to allow a session to remove assertions or add skips on purpose.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_ROUNDS = 3
TIMEOUT_S = 900
DEFAULT_COMMANDS = ["python3 scripts/agent/check_style.py ."]
ASSERTS = re.compile(r"^\s*assert\b|pytest\.raises|pytest\.approx|\bassert_\w+\(|"
                     r"\bexpect_\w+\(", re.M)
SKIPS = re.compile(r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|unittest\.skip")


def git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def fingerprint(root):
    h = hashlib.sha256()
    h.update(git(root, "rev-parse", "HEAD").encode())
    h.update(git(root, "diff", "HEAD").encode())
    untracked = [p for p in git(root, "ls-files", "--others",
                                "--exclude-standard").splitlines()
                 if not p.endswith(".gate-state.json")]
    for p in untracked:
        h.update(p.encode())
        try:
            h.update((root / p).read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def base_ref(root):
    for ref in ("origin/main", "main", "origin/master", "master"):
        base = git(root, "merge-base", "HEAD", ref).strip()
        if base:
            return base
    return "HEAD"


def is_test_file(path):
    name = Path(path).name
    return path.endswith(".py") and (
        path.startswith("tests/") or "/tests/" in path
        or name.startswith("test_") or name.endswith("_test.py"))


def weakened_tests(root):
    if os.environ.get("AGENT_ALLOW_TEST_CHANGES") == "1":
        return []
    base = base_ref(root)
    changed = git(root, "diff", "--name-only", base).splitlines()
    problems = []
    for path in filter(is_test_file, changed):
        before = git(root, "show", f"{base}:{path}")
        f = root / path
        after = f.read_text(encoding="utf-8") if f.is_file() else ""
        a0, a1 = len(ASSERTS.findall(before)), len(ASSERTS.findall(after))
        s0, s1 = len(SKIPS.findall(before)), len(SKIPS.findall(after))
        if a1 < a0:
            problems.append(f"{path}: assertions went from {a0} to {a1}")
        if s1 > s0:
            problems.append(f"{path}: skip/xfail markers went from {s0} to {s1}")
    return problems


def load_commands(root):
    f = root / ".claude" / "gate-commands"
    if not f.is_file():
        return DEFAULT_COMMANDS
    return [line.strip() for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    root = Path(payload.get("cwd") or Path.cwd())
    if not git(root, "rev-parse", "--is-inside-work-tree").strip():
        return 0

    state_file = root / ".claude" / ".gate-state.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}

    current = fingerprint(root)
    if state.get("passed") == current:
        return 0

    failures = []
    for cmd in load_commands(root):
        try:
            r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True,
                               text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            failures.append(f"$ {cmd}\ntimed out after {TIMEOUT_S}s")
            continue
        if r.returncode != 0:
            tail = (r.stdout + r.stderr).strip().splitlines()[-40:]
            failures.append(f"$ {cmd}\n" + "\n".join(tail))
    weak = weakened_tests(root)
    if weak:
        failures.append("Tests were weakened compared with the base branch:\n"
                        + "\n".join(weak)
                        + "\nRestore them, or stop and ask the owner to approve.")

    state_file.parent.mkdir(exist_ok=True)
    if not failures:
        state_file.write_text(json.dumps({"passed": current, "rounds": 0}))
        return 0

    rounds = state.get("rounds", 0) + 1
    if rounds > MAX_ROUNDS:
        state_file.write_text(json.dumps({"rounds": 0}))
        return 0
    state_file.write_text(json.dumps({"rounds": rounds}))

    msg = "\n\n".join(failures)
    if rounds == MAX_ROUNDS:
        msg += ("\n\nThis is the last automatic round. If you cannot fix this, "
                "stop and tell the owner exactly what is failing and why.")
    print(f"Definition of done not met (round {rounds} of {MAX_ROUNDS}).\n\n{msg}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
