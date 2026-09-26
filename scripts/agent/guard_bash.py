#!/usr/bin/env python3
"""Block git commands that bypass checks or rewrite shared history.

Runs as a Claude Code PreToolUse hook on Bash. Exit code 2 blocks the
command and shows the reason to Claude. Permission rules in settings.json
cover the common spellings, and this catches the rest (flags in a different
order, short flags, chained commands).
"""
import json
import re
import shlex
import sys

BLOCKED = [
    (r"\bgit\b[^;&|]*\s--no-verify\b",
     "Never bypass hooks with --no-verify. Fix what the hook reports."),
    (r"\bgit\s+commit\b[^;&|]*\s-[a-zA-Z]*n[a-zA-Z]*\b",
     "git commit -n skips hooks. Fix what the hook reports instead."),
    (r"\bgit\s+push\b[^;&|]*\s(--force\b|--force-with-lease\b|-[a-zA-Z]*f\b|\+\S+)",
     "Force-pushing is not allowed. Ask the owner if history really must change."),
    (r"\bgit\s+reset\b[^;&|]*--hard\b",
     "git reset --hard discards work. Ask the owner first."),
    (r"\bgit\s+clean\b[^;&|]*\s-[a-zA-Z]*f",
     "git clean -f deletes untracked files. Ask the owner first."),
    (r"\bgit\s+(checkout|restore)\s+(--\s+)?\.\s*$",
     "This discards all uncommitted changes. Ask the owner first."),
    (r"\bgit\s+config\b[^;&|]*\bcore\.hooksPath\b",
     "Changing the hooks path disables pre-commit. Ask the owner first."),
]


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    command = payload.get("tool_input", {}).get("command", "")
    if not command:
        return 0
    try:
        normalized = " ".join(shlex.split(command, posix=True))
    except ValueError:
        normalized = command
    for text in (command, normalized):
        for pattern, reason in BLOCKED:
            if re.search(pattern, text):
                print(f"Blocked: {reason}", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
