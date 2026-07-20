from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


MODEL_STYLES = {
    "ACM-4 (coupled)": dict(color="black", linestyle="-", marker="o"),
    "FCM": dict(color="0.40", linestyle="--", marker="s"),
    "ACM-2": dict(color="0.25", linestyle=":", marker="^"),
    "ACM-3": dict(color="0.55", linestyle="-.", marker="v"),
    "ACM-4 (independent)": dict(color="0.68", linestyle=(0, (5, 2)), marker="D"),
    "SGC-GNN": dict(color="0.15", linestyle=(0, (3, 1, 1, 1)), marker="P"),
    "Bayesian network": dict(color="0.60", linestyle=(0, (1, 1)), marker="X"),
    "PPO": dict(color="0.32", linestyle=(0, (7, 2)), marker="*"),
    "Robust-LP": dict(color="0.76", linestyle=(0, (3, 2)), marker="h"),
    "Equal allocation": dict(color="0.86", linestyle=(0, (1, 2)), marker="+"),
    "Current stress": dict(color="0.18", linestyle=(0, (6, 2)), marker="x"),
    "Admissions": dict(color="0.62", linestyle=(0, (2, 2)), marker="d"),
}


def apply_nature_style(dpi: int = 600) -> None:
    """Apply the exact grayscale constraints supplied with the manuscript."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "grid.linewidth": 0.55,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "mathtext.fontset": "dejavuserif",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, f"({label})", transform=ax.transAxes, ha="left", va="bottom")


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(which="both", top=True, right=True)
    if grid:
        ax.grid(True, which="major", axis="both")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str] = ("png", "tiff"),
    dpi: int = 600,
) -> list[Path]:
    """Save raster-only publication figures. PDF is deliberately forbidden."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig.tight_layout()
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt == "pdf":
            raise ValueError("PDF output is disabled by the study specification.")
        if fmt not in {"png", "tif", "tiff"}:
            raise ValueError(f"Unsupported raster format: {fmt}")
        path = output_dir / f"{stem}.{fmt}"
        temporary = output_dir / f".{stem}.tmp.{fmt}"
        temporary.unlink(missing_ok=True)
        last_error: Exception | None = None
        for _ in range(3):
            fig.savefig(
                temporary,
                format=fmt,
                dpi=dpi,
                bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"} if fmt in {"tif", "tiff"} else None,
            )
            try:
                with Image.open(temporary) as image:
                    image.verify()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
        if last_error is not None:
            raise RuntimeError(f"Failed to create a valid {fmt.upper()} figure for {stem}: {last_error}")
        temporary.replace(path)
        temporary.unlink(missing_ok=True)
        paths.append(path)
    plt.close(fig)
    return paths
