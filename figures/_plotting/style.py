"""Shared visual style for the figure scripts.

Call ``setup_style()`` once at the top of ``main()`` before any plotting.
Call ``finish_figure(fig, axes)`` on each figure before saving.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


_CJK_FALLBACK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "SimHei",
    "Arial Unicode MS",
)


@lru_cache(maxsize=1)
def _get_cjk_fallback_font_path() -> str | None:
    """Return the path of an installed sans-serif font with broad CJK coverage."""
    fonts_by_name = {}
    for font in font_manager.fontManager.ttflist:
        fonts_by_name.setdefault(font.name, font.fname)
    for candidate in _CJK_FALLBACK_FONT_CANDIDATES:
        if candidate in fonts_by_name:
            return fonts_by_name[candidate]
    return None


def setup_style() -> None:
    """Configure seaborn theme and matplotlib rcParams for formal plots."""
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.2,
    )
    rc_params = {
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    }
    plt.rcParams.update(rc_params)


def setup_grid_style(figsize: tuple[float, float] | None = (12, 6), font_size: float = 16) -> None:
    """Whitegrid seaborn style used by the extra (non-paper) plot scripts."""
    sns.set_style("whitegrid")
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = font_size


def get_text_fontfamily(text: str | None) -> list[str] | None:
    """Return a font-family list for strings that contain CJK code points."""
    if not text:
        return None
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return None

    cjk_font_path = _get_cjk_fallback_font_path()
    if cjk_font_path is None:
        return None

    cjk_name = font_manager.FontProperties(fname=cjk_font_path).get_name()
    return ["DejaVu Sans", cjk_name]


def finish_figure(fig, axes=None, despine_kwargs=None) -> None:
    """Apply final polish: despine and tighten layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    axes : optional axes or array of axes.  If *None*, despine all axes in *fig*.
    despine_kwargs : optional dict passed to ``sns.despine`` (e.g. ``left=True``).
    """
    kwargs = despine_kwargs or {}
    if axes is not None:
        for ax in np.atleast_1d(axes).flat:
            sns.despine(ax=ax, **kwargs)
    else:
        sns.despine(fig=fig, **kwargs)
    fig.tight_layout()
