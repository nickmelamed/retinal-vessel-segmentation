#!/usr/bin/env python3
"""Require the owner's approval before Claude edits a protected file.

Runs as a Claude Code PreToolUse hook on Edit, Write, and MultiEdit. Protected
files are listed as glob patterns in ``.claude/protected-paths`` (one per
line, relative to the repo root, ``#`` for comments). A matching edit gets a
permission prompt with the reason, even in auto mode.
"""
import fnmatch
import json
import sys
from pathlib import Path


def load_patterns(root):
    f = root / ".claude" / "protected-paths"
    if not f.is_file():
        return []
    entries = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, reason = line.partition("  #")
        entries.append((pattern.strip(), reason.strip()))
    return entries


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    file_path = payload.get("tool_input", {}).get("file_path")
    if not file_path:
        return 0
    root = Path(payload.get("cwd") or Path.cwd())
    try:
        rel = Path(file_path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return 0

    for pattern, reason in load_patterns(root):
        if fnmatch.fnmatch(rel, pattern):
            why = reason or "listed in .claude/protected-paths"
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": f"{rel} is protected ({why}).",
                }
            }))
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
