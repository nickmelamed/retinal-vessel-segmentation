"""Build the 1280x640 repository preview image from the hero figure.

GitHub shows this image when the repository link is shared. Uploading it
is a manual step under Settings, General (docs/REPO_SETTINGS.md). The image
holds the project title above the hero figure, caption included, so
``make figures`` rebuilds it after redrawing the hero. Its JSON sidecar
carries the hero's run, tag, commit, and data hash. The layout is the
``preview`` section of ``configs/reporting.yaml``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from retinal_vessels.config import (
    ConfigError,
    FiguresConfig,
    PreviewConfig,
    load_reporting_config,
)
from retinal_vessels.utils import setup_logging

logger = logging.getLogger("make_social_preview")

REPO = Path(__file__).resolve().parents[1]


def build(hero: Path, out: Path, colors: FiguresConfig, layout: PreviewConfig) -> Path:
    """Write the preview and its sidecar to ``out`` and return it.

    A missing hero figure or hero sidecar raises ``FileNotFoundError``, since
    the preview's lineage comes from the hero's.
    """
    lineage = hero.with_suffix(".json")
    for path in (hero, lineage):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run make figures first.")
    with Image.open(hero) as img:
        pixels = np.asarray(img.convert("RGB"))
    size = (layout.width_px / layout.dpi, layout.height_px / layout.dpi)
    fig = Figure(figsize=size, dpi=layout.dpi, facecolor=colors.surface)
    fig.text(
        0.5,
        layout.title_y,
        layout.title,
        ha="center",
        va="center",
        fontsize=layout.title_size,
        color=colors.ink,
    )
    fig.text(
        0.5,
        layout.subtitle_y,
        layout.subtitle,
        ha="center",
        va="center",
        fontsize=layout.subtitle_size,
        color=colors.secondary_ink,
    )
    ax = fig.add_axes(layout.hero_box)
    ax.imshow(pixels, interpolation="lanczos")
    ax.set_axis_off()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=colors.surface, metadata={"Software": None})
    size = out.stat().st_size
    if size > colors.max_bytes:
        out.unlink()
        raise ValueError(
            f"{out.name} is {size} bytes, over the {colors.max_bytes} byte limit for "
            "committed files. Shrink its layout rather than raising the limit."
        )
    sidecar = json.loads(lineage.read_text(encoding="utf-8"))
    sidecar.update(
        figure=out.name, generated_by="scripts/make_social_preview.py", built_from=hero.name
    )
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
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
        config = load_reporting_config(args.config)
        build(args.hero, args.out, config.figures, config.preview)
    except (ConfigError, FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
