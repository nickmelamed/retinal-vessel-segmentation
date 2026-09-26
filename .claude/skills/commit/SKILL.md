---
name: commit
description: Make one atomic Conventional Commit through the pre-commit hooks. Use for every commit in this repo.
---

Commit the staged change $ARGUMENTS.

1. Check that the staged diff is one logical change, with code and its tests together.
2. Write the message to a file in the scratchpad: `type(scope): subject`
   (lowercase, imperative, no period, 72 characters at most), a blank line,
   then a body saying what changed and why, wrapped at 72. End it with the
   attribution line from the system reminder.
3. Run `git commit -F <file>`. Never pass the message inline, because the
   Bash guard matches blocked flags anywhere in the command text.
4. If a hook fails or rewrites files, fix the cause, restage, and commit
   again. Never skip the hooks.
5. Show the resulting `git log --oneline -1`.
