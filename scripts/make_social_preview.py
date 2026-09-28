"""Build the 1280x640 repository preview image from the hero figure.

GitHub shows this image when the repository link is shared. Uploading it
is a manual step under Settings, General (docs/REPO_SETTINGS.md). The image
holds the project title above the hero figure, caption included, so it is
rebuilt whenever ``make figures`` redraws the hero.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from retinal_vessels.config import ConfigError, FiguresConfig, load_reporting_config
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("make_social_preview")

REPO = Path(__file__).resolve().parents[1]
# GitHub's recommended preview size.
WIDTH_PX, HEIGHT_PX = 1280, 640
DPI = 100
TITLE = "Retinal vessel segmentation on DRIVE"
SUBTITLE = "A self-directed learning project on public data. Not for clinical use."
# Fractions of the canvas: the title band on top, the hero figure below it.
TITLE_Y, SUBTITLE_Y = 0.9, 0.82
HERO_BOX = (0.03, 0.03, 0.94, 0.74)
TITLE_SIZE, SUBTITLE_SIZE = 30, 15


def build(hero: Path, out: Path, colors: FiguresConfig) -> Path:
    """Write the preview to ``out`` and return it. A missing hero raises ``FileNotFoundError``."""
    if not hero.is_file():
        raise FileNotFoundError(f"hero figure not found at {hero}. Run make figures first.")
    with Image.open(hero) as img:
        pixels = np.asarray(img.convert("RGB"))
    fig = Figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, facecolor=colors.surface)
    fig.text(0.5, TITLE_Y, TITLE, ha="center", va="center", fontsize=TITLE_SIZE, color=colors.ink)
    fig.text(
        0.5,
        SUBTITLE_Y,
        SUBTITLE,
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        color=colors.secondary_ink,
    )
    ax = fig.add_axes(HERO_BOX)
    ax.imshow(pixels, interpolation="lanczos")
    ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=colors.surface, metadata={"Software": None})
    logger.info("Wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hero", type=Path, default=REPO / "figures" / "hero.png")
    parser.add_argument("--out", type=Path, default=REPO / "figures" / "social_preview.png")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "reporting.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        build(args.hero, args.out, load_reporting_config(args.config).figures)
    except (ConfigError, FileNotFoundError) as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
