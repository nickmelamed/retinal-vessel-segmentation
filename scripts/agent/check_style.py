#!/usr/bin/env python3
"""Flag comment, docstring, and Markdown patterns that read as machine-written.

Pass files or directories to scan. With --hook, read a Claude Code hook
payload from stdin and check only the edited file. Add ``style: ok`` to a
line to skip it. Glob patterns in ``.claude/style-ignore`` (one per line)
exclude files from the check.
"""
import argparse
import ast
import fnmatch
import io
import json
import re
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", "build",
    "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache", "renv", ".quarto",
}
DEFAULT_IGNORE = [
    "CLAUDE.md", "AGENTS.md", "PROGRESS.md", "CHANGELOG.md", ".claude/*",
    "*handoff*.md",
]
IGNORE_MARK = "style: ok"

# An en dash between two digits is a numeric range ("images 21–40"), not
# sentence punctuation, so it is allowed.
DASH = re.compile(r"—|(?<!\d)–|–(?!\d)")
BANNED = re.compile(
    r"\b(robust|seamless(ly)?|comprehensive|leverag(e|es|ed|ing)|delve|"
    r"crucial|utiliz(e|es|ed|ing)|meticulous(ly)?)\b",
    re.I,
)
AGENT_REF = re.compile(r"CLAUDE\.md|AGENTS\.md|as instructed", re.I)
HISTORY = re.compile(
    r"^(updated|now handles|now supports|fixed|changed|modified|refactored)\b",
    re.I,
)
BANNER = re.compile(r"^[=\-*#~_]{3,}|[=\-*#~_]{4,}\s*$")
BOLD_LEADIN = re.compile(r"\*\*[^*]+:\*\*|\*\*[^*]+\*\*\s*:")
INLINE_CODE = re.compile(r"`[^`]*`")


def load_ignore(root):
    patterns = list(DEFAULT_IGNORE)
    f = root / ".claude" / "style-ignore"
    if f.is_file():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def is_ignored(path, root, patterns):
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.name, p)
               for p in patterns)


def prose_problems(text, allow_semicolon=False):
    text = INLINE_CODE.sub("", text)
    found = []
    if DASH.search(text):
        found.append("em/en dash used as punctuation")
    m = BANNED.search(text)
    if m:
        found.append(f"filler word '{m.group(0)}'")
    if AGENT_REF.search(text):
        found.append("references agent instructions")
    if not allow_semicolon and ";" in text:
        found.append("semicolon in prose")
    return found


def check_python(source):
    findings = []
    lines = source.splitlines()

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            lineno = tok.start[0]
            if IGNORE_MARK in lines[lineno - 1]:
                continue
            body = tok.string.lstrip("#").strip()
            if body.startswith("!") or body.startswith("type:") or "noqa" in body:
                continue
            probs = prose_problems(body, allow_semicolon=True)
            if HISTORY.search(body):
                probs.append("describes edit history")
            if body and BANNER.search(body):
                probs.append("section banner")
            findings += [(lineno, p) for p in probs]
    except (tokenize.TokenError, IndentationError):
        pass

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings
    nodes = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for node in nodes:
        body = getattr(node, "body", [])
        if not body or not isinstance(body[0], ast.Expr):
            continue
        val = body[0].value
        if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
            continue
        for offset, line in enumerate(val.value.splitlines()):
            lineno = val.lineno + offset
            if lineno <= len(lines) and IGNORE_MARK in lines[lineno - 1]:
                continue
            findings += [(lineno, p) for p in prose_problems(line)]
    return findings


def check_markdown(source):
    findings = []
    in_fence = False
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence or IGNORE_MARK in line:
            continue
        probs = prose_problems(line)
        if BOLD_LEADIN.search(INLINE_CODE.sub("", line)):
            probs.append("bold lead-in with colon")
        findings += [(lineno, p) for p in probs]
    return findings


def check_file(path):
    if path.suffix not in {".py", ".md", ".qmd"}:
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return check_python(source) if path.suffix == ".py" else check_markdown(source)


def iter_files(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not SKIP_DIRS.intersection(f.parts):
                    yield f
        elif p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["."])
    parser.add_argument("--hook", action="store_true",
                        help="read a Claude Code hook payload from stdin")
    args = parser.parse_args()
    root = Path.cwd()

    if args.hook:
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError:
            return 0
        file_path = payload.get("tool_input", {}).get("file_path")
        if not file_path:
            return 0
        root = Path(payload.get("cwd") or root)
        files = [Path(file_path)]
    else:
        files = iter_files(args.paths)

    patterns = load_ignore(root)
    report = []
    for f in files:
        if is_ignored(f, root, patterns):
            continue
        for lineno, problem in check_file(f):
            report.append(f"{f}:{lineno}: {problem}")

    if not report:
        return 0
    out = sys.stderr if args.hook else sys.stdout
    print("\n".join(report), file=out)
    if args.hook:
        print("Fix these style issues in comments, docstrings, or docs "
              "before continuing.", file=out)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
