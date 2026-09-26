#!/usr/bin/env python3
"""Check that every reported number in the docs comes from a generated artifact.

Finds decimals and percentages in the given documents (outside code blocks,
inline code, and links) and fails if a number does not appear in any file
under the source directories. Write ``TBD`` for numbers that do not exist
yet, and add ``numbers: ok`` to a line whose numbers are not results (a
dataset fact from its paper, for example).

Example::

    python3 scripts/agent/check_numbers.py README.md MODEL_CARD.md \\
        --sources results/tables results/release
"""
import argparse
import re
import sys
from pathlib import Path

IGNORE_MARK = "numbers: ok"
NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?%|\d+\.\d+)(?![\w.]*\d)")
LINK_TARGET = re.compile(r"\]\([^)]*\)|https?://\S+")
INLINE_CODE = re.compile(r"`[^`]*`")
VERSION = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def reported_numbers(text):
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or IGNORE_MARK in line or stripped.startswith("<!--"):
            continue
        clean = VERSION.sub("", INLINE_CODE.sub("", LINK_TARGET.sub("", line)))
        for m in NUMBER.finditer(clean):
            yield lineno, m.group(1)


def source_numbers(dirs):
    found = set()
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            found.update(NUMBER.findall(text))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("docs", nargs="+")
    parser.add_argument("--sources", nargs="+", default=["results/tables"])
    args = parser.parse_args()

    if not any(Path(d).exists() for d in args.sources):
        print("check_numbers: no source directories yet, skipping.")
        return 0
    known = source_numbers(args.sources)

    problems = []
    for doc in args.docs:
        path = Path(doc)
        if not path.is_file():
            continue
        for lineno, num in reported_numbers(path.read_text(encoding="utf-8")):
            if num not in known:
                problems.append(f"{doc}:{lineno}: {num} is not in "
                                f"{', '.join(args.sources)}")
    if problems:
        print("\n".join(problems))
        print("Regenerate the tables and copy numbers from them, write TBD, "
              "or mark a non-result line with 'numbers: ok'.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
