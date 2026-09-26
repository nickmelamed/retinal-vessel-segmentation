---
name: finish-phase
description: Close out a build phase. Runs checks, independent review, the section 2 rules check, changelog, and summary before asking to open a PR.
disable-model-invocation: true
---

Close out the current phase (see PROGRESS.md).

1. Run `make ci` and show the output. Fix failures before going on.
2. Run the spec-reviewer agent on `git diff main...HEAD`, pointing it at the
   SPEC sections this phase covers. Fix every problem it reports as affecting
   correctness, requirements, or the non-negotiable rules. List its optional
   items for the owner without acting on them.
3. Run the style-reviewer agent on the same diff and apply its rewrites.
4. Check every changed document against the non-negotiable rules in
   CLAUDE.md, and run `scripts/agent/check_numbers.py` on them.
5. Make sure any design decision is in docs/DECISIONS.md.
6. Update CHANGELOG.md under "Unreleased" and tick the finished items in
   PROGRESS.md.
7. Rewrite HANDOFF.md for planning the next phase. Cover where things
   stand, what the next phase must deliver, decisions to raise with the
   owner, and working notes. Replace anything that is out of date.
8. Commit the fixes as atomic Conventional Commits.
9. Give the owner a short summary covering what was built, the commits,
   decisions logged, anything left open from review, and the evidence that
   checks pass.
10. Ask before pushing or opening the pull request (merge commit, not squash).
