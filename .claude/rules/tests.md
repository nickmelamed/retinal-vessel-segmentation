---
paths:
  - "tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
---

# Tests

- Tests are the oracle for the code, so never weaken one to make it pass. Do
  not delete assertions, loosen tolerances, or add skip or xfail markers
  without the owner's approval. The Stop hook checks this against the base
  branch.
- When fixing a bug, first write a test that fails because of it.
- Prefer small synthetic fixtures with known answers. Tests never touch real
  or private data.
- For numeric code, add property-based tests with Hypothesis (bounds,
  symmetry, invariance) alongside the hand-worked examples.
- Coverage is a floor, not the goal. `mutmut` on core modules is the real
  measure of whether tests catch bugs. Do not write hollow tests to raise a
  number.
- Mark slow tests with `@pytest.mark.slow` so the fast gate stays fast.
- The required unit and integration tests are listed in SPEC section 14.
  Tests never touch real DRIVE data. Use `tests/fixtures/`.
