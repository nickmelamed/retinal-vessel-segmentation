# Repository settings

These settings live in GitHub, not in the repository, so the owner applies
them by hand.

## Secrets

Add `ANTHROPIC_API_KEY` under Settings, Secrets and variables, Actions. The
Claude review workflow (`.github/workflows/claude-review.yml`) uses it to
review each pull request. Without it, that workflow fails and the main CI is
unaffected.

## Branch protection on `main`

Under Settings, Branches, add a rule for `main` that requires a pull request
before merging and requires the `ci` check from the CI workflow to pass. Allow merge
commits, since phases merge with a merge commit rather than a squash.

## Description and topics

Description:

> U-Net vessel segmentation on DRIVE with cross-validation, external validation, uncertainty maps, a SQL audit trail, R statistics, and LSTM-autoencoder narrowing detection. Learning project, not for clinical use.

Topics: `medical-imaging`, `image-segmentation`, `retinal-imaging`, `u-net`,
`tensorflow`, `uncertainty-quantification`, `model-governance`, `r`,
`sqlite`, `reproducible-research`.

## Social preview image

`make presentation` writes `figures/social_preview.png`, 1280 by 640 pixels,
from the hero figure. Upload it under Settings, General, Social preview, and
upload it again whenever `make figures` changes the hero.

## Badges

The README shows CI, Python, license, and "not for clinical use" badges. It
has no coverage badge, since CI prints coverage but publishes nothing a badge
can read (D-022). Adding one means a CI change, which is revisited in phase 10.

## Added in later phases

The website field and GitHub Pages (phase 8) point at the Quarto results site.
