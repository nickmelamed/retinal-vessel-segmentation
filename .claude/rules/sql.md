---
paths:
  - "sql/**"
  - "src/retinal_vessels/db.py"
---

# SQL standards

- `sqlfluff` with the SQLite dialect. Uppercase keywords, snake_case
  identifiers, explicit column lists (no `SELECT *` in committed queries).
- Every query in `sql/queries/` starts with a comment stating the question it
  answers.
- Schema changes go through `sql/schema.sql` with a `schema_version` table.
  Never alter the database by hand.
- Never store images, patches, or arrays in the database.
- `04_leakage_audit.sql` and `09_frozen_model_audit.sql` must return zero rows
  on a valid database and rows on a deliberately broken one. Both are
  protected. Tests enforce this (SPEC section 14).
