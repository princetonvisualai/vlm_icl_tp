"""Model registry for the figure scripts: colors, families, order, display names.

Colors use ColorBrewer sequential palettes so that models within a family
share a hue (darker = larger model) while remaining colorblind-friendly.

Family hues: Qwen3-VL blue, Qwen2.5-VL green, Qwen3.5 red,
InternVL3.5 yellow/orange, Gemma4 purple.
"""

from __future__ import annotations

MODEL_COLORS: dict[str, str] = {
    # Qwen3-VL — blues (light → dark by size)
    "Qwen3-VL-2B-Instruct":        "#a6cee3",
    "Qwen3-VL-4B-Instruct":        "#6baed6",
    "Qwen3-VL-8B-Instruct":        "#3182bd",
    "Qwen3-VL-32B-Instruct":       "#08519c",
    "Qwen3-VL-235B-A22B-Instruct": "#08306b",
    # Qwen2.5-VL — greens
    "Qwen2.5-VL-3B-Instruct":      "#a1d99b",
    "Qwen2.5-VL-7B-Instruct":      "#41ab5d",
    "Qwen2.5-VL-32B-Instruct":     "#238b45",
    "Qwen2.5-VL-72B-Instruct":     "#005a32",
    # Qwen3.5 — reds (light → dark by size)
    "Qwen3.5-4B":                   "#fc8d6a",
    "Qwen3.5-27B":                  "#de2d26",
    # InternVL3.5 — yellows/oranges (light → dark by size)
    "InternVL3-5-2B":               "#fff7bc",
    "InternVL3-5-4B":               "#fee391",
    "InternVL3-5-8B":               "#fec44f",
    "InternVL3-5-38B":              "#f59e0b",
    "InternVL3-5-78B":              "#d95f02",
    # Gemma4 — purples (light → dark by size)
    "gemma-4-4b-it":                "#cbc9e2",
    "gemma-4-E4B-it":               "#cbc9e2",
    "gemma-4-12b-it":               "#9e9ac8",
    "gemma-4-27b-it":               "#756bb1",
    "gemma-4-31b-it":               "#54278f",
}


MODEL_FAMILIES: dict[str, list[str]] = {
    "qwen25vl": ["qwen2.5", "qwen2_5"],
    "qwen35vl": ["qwen3.5", "qwen3_5"],
    "qwen3vl": ["qwen3"],
    "internvl35": ["internvl"],
    "gemma4": ["gemma"],
}

# Display-friendly family names for plot labels
FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "qwen3vl": "Qwen3VL",
    "qwen25vl": "Qwen2.5VL",
    "qwen35vl": "Qwen3.5",
    "internvl35": "InternVL3.5",
    "gemma4": "Gemma4",
}

# Canonical ordering for legend columns
FAMILY_ORDER: list[str] = ["qwen3vl", "qwen25vl", "qwen35vl", "internvl35", "gemma4"]

# Display-friendly per-model labels (used in plot legends)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "InternVL3-5-38B": "InternVL3.5-38B",
    "gemma-4-31b-it": "Gemma4-31B-it",
}


def get_family_display_name(family: str) -> str:
    """Return display-friendly family name."""
    return FAMILY_DISPLAY_NAMES.get(family, family)


_NORMALIZED_DISPLAY_NAMES: dict[str, str] = {
    name.lower().replace("_", "-").replace(".", "-"): label
    for name, label in MODEL_DISPLAY_NAMES.items()
}


def get_model_display_name(model_name: str) -> str:
    """Return display-friendly model label, falling back to the raw name."""
    if model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    norm = model_name.lower().replace("_", "-").replace(".", "-")
    return _NORMALIZED_DISPLAY_NAMES.get(norm, model_name)


def get_model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    for family, patterns in MODEL_FAMILIES.items():
        if any(p in name_lower for p in patterns):
            return family
    return model_name


def _normalize(name: str) -> str:
    """Normalize model name for fuzzy matching: lowercase, collapse separators."""
    return name.lower().replace("_", "-").replace(".", "-")


# Build a normalized lookup once at import time.
_NORMALIZED_COLORS: dict[str, str] = {_normalize(k): v for k, v in MODEL_COLORS.items()}


def get_model_color(model_name: str) -> str:
    """Return the assigned color for a model, falling back to gray.

    Matches are case-insensitive and treat ``_``, ``-``, and ``.`` as
    equivalent so that directory names like ``InternVL3_5-38B`` resolve
    correctly against canonical keys like ``InternVL3-5-38B``.
    """
    # Try exact match first, then normalized.
    color = MODEL_COLORS.get(model_name)
    if color is not None:
        return color
    return _NORMALIZED_COLORS.get(_normalize(model_name), "#888888")


def add_family_legend(fig, ax, *, fontsize=None) -> None:
    """Add a grid legend below the plot: each column is a model family, rows are sizes."""
    import matplotlib.lines as mlines

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    handle_map = dict(zip(labels, handles))

    # Group by family, preserving MODEL_FAMILIES order
    family_models: dict[str, list[str]] = {}
    for name in labels:
        family = get_model_family(name)
        family_models.setdefault(family, []).append(name)

    family_order = [f for f in FAMILY_ORDER if f in family_models]
    for f in family_models:
        if f not in family_order:
            family_order.append(f)

    max_rows = max(len(v) for v in family_models.values())
    ncol = len(family_order)

    # Build grid in column-major order (matplotlib legend fills top→bottom per column)
    dummy = mlines.Line2D([], [], color="none", marker="None", linestyle="None")
    grid_handles: list = []
    grid_labels: list[str] = []
    for fam in family_order:
        models = family_models[fam]
        for row in range(max_rows):
            if row < len(models):
                grid_handles.append(handle_map[models[row]])
                grid_labels.append(models[row])
            else:
                grid_handles.append(dummy)
                grid_labels.append(" ")

    # Remove per-axes legend, add figure-level grid legend
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    fig.legend(
        grid_handles,
        grid_labels,
        loc="lower center",
        ncol=ncol,
        fontsize=fontsize or "x-small",
        frameon=True,
        edgecolor="0.8",
        facecolor="white",
        framealpha=0.9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.08 + 0.05 * max_rows)


def family_models(family: str) -> list[str]:
    """Registered model names of *family*, in registry (small to large) order."""
    return [m for m in MODEL_COLORS if get_model_family(m) == family]


# Named model orders used by the extra plot scripts.
QWEN3_MODELS: list[str] = family_models("qwen3vl")
QWEN25_MODELS: list[str] = family_models("qwen25vl")
API_MODELS: list[str] = ["gpt", "gpt_noreasoning", "gemini"]
