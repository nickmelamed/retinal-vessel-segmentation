---
name: release
description: Cut a tagged release of the DRIVE project. Only the owner starts this.
disable-model-invocation: true
---

Prepare release $ARGUMENTS (for example v0.1.0).

1. Confirm the tree is clean, on `main`, with CI green.
2. Confirm every run whose results appear in a document is marked reported,
   came from a clean tree, and records the data checksum (rule 7).
3. Run `make snapshot tables figures report` and show that nothing changed
   that should not have.
4. Run `scripts/agent/check_numbers.py` over README.md, MODEL_CARD.md, and
   docs/one_pager.md, and check each against the non-negotiable rules.
5. For v1.0.0 and later, run `make reproduce` from a fresh clone inside
   Docker and record the runtime.
6. Move CHANGELOG.md's "Unreleased" entries under the version and date, and
   make sure the README names the tag that produced its numbers.
7. Commit as `chore(release): <version>`.
8. Show the owner the changelog entry and the exact `git tag -a` and
   `git push` commands, then wait for approval.
