---
paths:
  - "R/**"
  - "site/**"
---

# R standards

- Tidyverse style, enforced by `styler` and `lintr`. Packages locked with
  `renv`.
- The report reads `results/experiments.db` through `DBI` and `RSQLite`, sets
  a seed, and prints `sessionInfo()` at the end.
- `ggplot2` for plots. Key plots are exported to `figures/`.
- With n = 20 (or n = 3 for the pathology subgroup), report effect sizes and
  intervals and say plainly what the sample size cannot support. Do not
  over-interpret p-values.
- CHASE_DB1 intervals use a cluster bootstrap by child.
