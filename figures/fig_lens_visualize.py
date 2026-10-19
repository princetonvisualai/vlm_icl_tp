"""LatentLens Visual Token Grid Visualization — spatial grid of decoded text per image patch."""

import argparse
import ijson
import json
import multiprocessing as mp
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn2, venn3
import numpy as np
from PIL import Image
from scipy import stats
from tqdm import tqdm

from figures._plotting.style import (
    setup_style,
    finish_figure,
    get_text_fontfamily,
)

from vlm_icl.data import prepare_data
from vlm_icl.images import resolve_image
from vlm_icl.lens import kneedle_threshold

# Matplotlib emits one warning per unsupported glyph/script when labels contain
# multilingual text, which can overwhelm long SLURM logs without affecting output.
warnings.filterwarnings(
    "ignore",
    message=r"Glyph \d+ .* missing from font\(s\) .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Matplotlib currently does not support .* natively\.",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Keyword highlight configuration
# ---------------------------------------------------------------------------
# Preset distract_cue keywords per task (hand-curated). useful_cue keywords
# are derived dynamically from the ground_truth labels seen in each task's
# results (see _build_task_keywords).
DISTRACT_CUE_KEYWORDS = {
    "color_identification": ["tiger", "frog", "capybara", "raven", "flamingo",
                             "scarlet", "ibis"],
    "shape_identification": ["yield", "stop", "warning", "speed", "railroad",
                             "crossing", "school", "caution"],
    "counting": [
        "alpaca", "antelope", "avocet", "bird", "bison", "boar", "buffalo",
        "camel", "caracal", "caribou", "cassowary", "cat", "cheetah", "chicken",
        "coyote", "crane", "deer", "dog", "donkey", "duck", "egret", "eland",
        "elephant", "elk", "emu", "fox", "gazelle", "giraffe", "gnu", "goat",
        "hare", "heron", "hippopotamus", "horse", "ibex", "ibis", "impala",
        "jackal", "jaguar", "kudu", "lapwing", "leopard", "lion", "llama",
        "moose", "mule", "okapi", "ostrich", "oystercatcher", "peacock", "pig",
        "plover", "pronghorn", "puma", "rabbit", "reindeer", "rhea", "rooster",
        "sandpiper", "serval", "sheep", "spoonbill", "springbok", "stilt",
        "stork", "swan", "tiger", "turkey", "warthog", "wildebeest", "wolf",
        "yak", "zebra",
    ],
}

CATEGORY_COLORS = {
    "ground_truth":  (0.1, 0.7, 0.3),   # green       — this entry's ground truth
    "expected_bias": (0.9, 0.3, 0.1),   # orange-red  — this entry's expected bias
    "distract_cue":  (0.2, 0.4, 0.9),   # blue        — distractor / context
    "_both":         (0.6, 0.2, 0.7),   # purple      — token encodes multiple (fallback)
    "_distract_gt":  (0.1, 0.6, 0.6),   # teal        — distract_cue + ground_truth
    "_distract_eb":  (0.8, 0.5, 0.2),   # orange      — distract_cue + expected_bias
}


def _build_entry_keywords(entry_data, task):
    """Assemble keyword lists for a single entry.

    ground_truth:  [entry's ground_truth label]
    expected_bias: [entry's expected_bias label] (if present)
    distract_cue:  preset hand-curated list from DISTRACT_CUE_KEYWORDS.
    """
    keywords = {}
    gt = entry_data.get("ground_truth")
    if gt:
        keywords["ground_truth"] = [str(gt).lower()]
    eb = entry_data.get("expected_bias")
    if eb:
        keywords["expected_bias"] = [str(eb).lower()]
    distract_cue = DISTRACT_CUE_KEYWORDS.get(task)
    if distract_cue:
        # Allow overlap with ground_truth/expected_bias — distinct colors show these cases.
        keywords["distract_cue"] = [kw.lower() for kw in distract_cue]
    return keywords or None

TARGET_COLORS = {
    "ground_truth":   (0.1, 0.7, 0.3),   # green
    "expected_bias":  (0.9, 0.3, 0.1),   # red
    "distract":       (0.2, 0.4, 0.9),   # blue
}

def _compile_keyword_patterns(task_keywords):
    """Pre-compile regex patterns for a task's keyword dict.

    Returns {category: [(compiled_re, keyword_str), ...]}.
    """
    compiled = {}
    for cat, kws in task_keywords.items():
        compiled[cat] = [(re.compile(r"\b" + kw + r"\b", re.IGNORECASE), kw)
                         for kw in kws]
    return compiled


def match_token_keywords(token_neighbors, compiled_patterns):
    """Check top-k neighbors for keyword matches.

    Returns dict mapping category -> (matched_keyword, best_similarity).
    Empty dict if no matches.
    """
    matches = {}  # category -> (keyword, similarity)
    for nn in token_neighbors:
        text = nn["token_str"]
        sim = nn["similarity"]
        for cat, patterns in compiled_patterns.items():
            for pat, kw in patterns:
                if pat.search(text):
                    if cat not in matches or sim > matches[cat][1]:
                        matches[cat] = (kw, sim)
                    break  # one keyword match per category per neighbor suffices
    return matches


def _get_target_texts(question, distract=None):
    """Return dict of target text categories for heatmap plots."""
    targets = {
        "ground_truth": question.get("ground_truth"),
        "expected_bias": question.get("expected_bias"),
        "distract": distract,
    }
    return {k: v for k, v in targets.items() if v is not None}


def _text_style_kwargs(text):
    """Use a CJK-capable fallback only for strings that need it."""
    fontfamily = get_text_fontfamily(text)
    return {"fontfamily": fontfamily} if fontfamily else {}



def _match_target_in_neighbors(token_neighbors, target_text):
    """Return highest similarity among neighbors matching target_text, or None."""
    pat = re.compile(r"\b" + re.escape(target_text) + r"\b", re.IGNORECASE)
    best = None
    for nn in token_neighbors:
        if pat.search(nn["token_str"]):
            sim = nn["similarity"]
            if best is None or sim > best:
                best = sim
    return best


def _average_demo_sets(demo_set_list):
    """Average multiple demo set results into a single layer_data dict.

    Each element in *demo_set_list* is a dict ``{layer_key: token_neighbors_list}``
    where ``token_neighbors_list`` is a list (one per spatial token) of lists of
    neighbor dicts ``{"token_str": ..., "similarity": ..., "caption": ...}``.

    Returns a single dict with the same structure.  For each (layer, token, rank),
    the similarity is averaged across demo sets; token_str and caption are taken
    from the first demo set.
    """
    if not demo_set_list:
        return {}
    if len(demo_set_list) == 1:
        return demo_set_list[0]

    ref = demo_set_list[0]
    averaged = {}
    for layer_key in ref:
        ref_tokens = ref[layer_key]
        n_tokens = len(ref_tokens)
        merged_tokens = []
        for t_idx in range(n_tokens):
            ref_nns = ref_tokens[t_idx]
            n_nns = len(ref_nns)
            merged_nns = []
            for nn_idx in range(n_nns):
                base_nn = dict(ref_nns[nn_idx])
                sims = []
                for ds in demo_set_list:
                    ds_tokens = ds.get(layer_key, [])
                    if t_idx < len(ds_tokens) and nn_idx < len(ds_tokens[t_idx]):
                        sims.append(ds_tokens[t_idx][nn_idx]["similarity"])
                if sims:
                    base_nn["similarity"] = sum(sims) / len(sims)
                merged_nns.append(base_nn)
            merged_tokens.append(merged_nns)
        averaged[layer_key] = merged_tokens
    return averaged


def discover_result_files(base_path):
    """Discover formal_lens_visualize.json files under base_path.

    Returns list of (task, model, experiment, json_path).
    """
    base = Path(base_path)
    entries = []
    for json_path in base.rglob("formal_lens_visualize.json"):
        parts = json_path.relative_to(base).parts
        # Expected: task/model/experiment/formal_lens_visualize.json
        if len(parts) == 4:
            task, model, experiment, _ = parts
            entries.append((task, model, experiment, json_path))
    return entries


def load_results(json_path):
    try:
        with open(json_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {json_path}: {e}")
        return None


_SKIP_KEYS = frozenset({"caption"})


def _finish_value(parser, event):
    """Consume remaining events for a value whose first event was ``event``."""
    if event == "start_map":
        depth = 1
        while depth:
            _, e, _ = next(parser)
            if e == "start_map":
                depth += 1
            elif e == "end_map":
                depth -= 1
    elif event == "start_array":
        depth = 1
        while depth:
            _, e, _ = next(parser)
            if e == "start_array":
                depth += 1
            elif e == "end_array":
                depth -= 1


def _build_value(parser, event, value):
    if event == "start_map":
        d = {}
        while True:
            _, e, v = next(parser)
            if e == "end_map":
                return d
            key = v
            _, e2, v2 = next(parser)
            if key in _SKIP_KEYS:
                _finish_value(parser, e2)
            else:
                d[key] = _build_value(parser, e2, v2)
    if event == "start_array":
        a = []
        while True:
            _, e, v = next(parser)
            if e == "end_array":
                return a
            a.append(_build_value(parser, e, v))
    return value


def load_metadata(json_path):
    """Read only the top-level 'metadata' dict without loading the full file."""
    try:
        with open(json_path, "rb") as f:
            return next(ijson.items(f, "metadata", use_float=True), {})
    except Exception as e:
        print(f"Warning: Could not read metadata from {json_path}: {e}")
        return {}


def stream_results_entries(json_path, max_entries=None):
    """Yield (entry_key, entry_data) via ijson.parse, dropping 'caption' subtrees
    at parse time so caption bytes are never allocated as Python objects."""
    count = 0
    try:
        with open(json_path, "rb") as f:
            parser = ijson.parse(f, use_float=True)
            for prefix, event, _ in parser:
                if prefix == "results" and event == "start_map":
                    break
            else:
                return
            while True:
                _, event, value = next(parser)
                if event == "end_map":
                    return
                entry_key = value  # map_key
                _, event, value = next(parser)
                if event != "start_map":
                    _finish_value(parser, event)
                    continue
                entry_data = _build_value(parser, "start_map", None)
                if "conditions" not in entry_data:
                    continue
                if max_entries is not None and count >= max_entries:
                    return
                yield entry_key, entry_data
                count += 1
    except Exception as e:
        print(f"Warning: Error streaming {json_path}: {e}")


def render_token_grid(image, grid_thw, token_labels, token_similarities,
                      title="", output_path=None, dpi=200, font_size=6):
    """Render text labels overlaid on the query image in a spatial grid."""
    t, h, w = grid_thw
    num_tokens = h * w

    # Pad or truncate labels to match grid
    if len(token_labels) < num_tokens:
        token_labels = token_labels + [""] * (num_tokens - len(token_labels))
        token_similarities = token_similarities + [0.0] * (num_tokens - len(token_similarities))
    token_labels = token_labels[:num_tokens]
    token_similarities = token_similarities[:num_tokens]

    cell_px = max(60, 400 // max(h, w))
    fig_w = w * cell_px / dpi * 2.5
    fig_h = h * cell_px / dpi * 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if image is not None:
        img_resized = image.resize((w * cell_px, h * cell_px), Image.BILINEAR)
        ax.imshow(np.array(img_resized), extent=[0, w, h, 0])
    else:
        ax.set_facecolor("#f0f0f0")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    for row in range(h):
        for col in range(w):
            idx = row * w + col
            label = token_labels[idx]
            sim = token_similarities[idx]

            # Semi-transparent background box colored by similarity
            alpha = 0.3 + 0.5 * min(sim, 1.0)
            rect = mpatches.FancyBboxPatch(
                (col + 0.02, row + 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=(1.0, 1.0, 1.0, alpha),
                edgecolor=(0.5, 0.5, 0.5, 0.3),
                linewidth=0.3,
            )
            ax.add_patch(rect)

            # Truncate long labels
            display = label[:10] if len(label) > 10 else label
            ax.text(
                col + 0.5, row + 0.5, display,
                fontsize=font_size, ha="center", va="center",
                color="black", fontweight="bold",
                **_text_style_kwargs(display),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=font_size + 2, pad=4)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_highlight_grid(image, grid_thw, token_matches, category_names,
                          title="", output_path=None, dpi=200, font_size=6):
    """Render keyword-highlight visualization: colored cells for keyword matches."""
    t, h, w = grid_thw
    num_tokens = h * w

    if len(token_matches) < num_tokens:
        token_matches = token_matches + [{}] * (num_tokens - len(token_matches))
    token_matches = token_matches[:num_tokens]

    cell_px = max(60, 400 // max(h, w))
    fig_w = w * cell_px / dpi * 2.5
    fig_h = h * cell_px / dpi * 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if image is not None:
        img_resized = image.resize((w * cell_px, h * cell_px), Image.BILINEAR)
        ax.imshow(np.array(img_resized), extent=[0, w, h, 0])
    else:
        ax.set_facecolor("#f0f0f0")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    for row in range(h):
        for col in range(w):
            idx = row * w + col
            m = token_matches[idx]

            if not m:
                # Unmatched — very faint overlay, no text
                rect = mpatches.FancyBboxPatch(
                    (col + 0.02, row + 0.02), 0.96, 0.96,
                    boxstyle="round,pad=0.02",
                    facecolor=(0.8, 0.8, 0.8, 0.12),
                    edgecolor=(0.5, 0.5, 0.5, 0.1),
                    linewidth=0.2,
                )
                ax.add_patch(rect)
                continue

            matched_cats = set(m.keys())
            if len(matched_cats) >= 2:
                # Check for specific overlap pairs
                if matched_cats == {"distract_cue", "ground_truth"}:
                    rgb = CATEGORY_COLORS["_distract_gt"]
                elif matched_cats == {"distract_cue", "expected_bias"}:
                    rgb = CATEGORY_COLORS["_distract_eb"]
                else:
                    rgb = CATEGORY_COLORS["_both"]
                keywords = [m[c][0] for c in matched_cats]
                display = "/".join(keywords)
                best_sim = max(m[c][1] for c in matched_cats)
            else:
                cat = next(iter(matched_cats))
                rgb = CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
                display = m[cat][0]
                best_sim = m[cat][1]

            alpha = 0.35 + 0.4 * min(best_sim, 1.0)
            rect = mpatches.FancyBboxPatch(
                (col + 0.02, row + 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor=(*rgb, alpha),
                edgecolor=(*rgb, min(alpha + 0.15, 1.0)),
                linewidth=0.4,
            )
            ax.add_patch(rect)

            display = display[:10] if len(display) > 10 else display
            ax.text(
                col + 0.5, row + 0.5, display,
                fontsize=font_size, ha="center", va="center",
                color="white", fontweight="bold",
                **_text_style_kwargs(display),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend — include overlap colors if they occurred
    legend_handles = []
    for cat in category_names:
        rgb = CATEGORY_COLORS.get(cat, (0.5, 0.5, 0.5))
        legend_handles.append(mpatches.Patch(color=rgb, label=cat))
    # Check for overlap occurrences and add to legend
    overlap_labels = {
        "_distract_gt": "distract+gt",
        "_distract_eb": "distract+eb",
    }
    added_overlaps = set()
    for m in token_matches:
        if len(m) >= 2:
            cats = set(m.keys())
            if cats == {"distract_cue", "ground_truth"}:
                key = "_distract_gt"
            elif cats == {"distract_cue", "expected_bias"}:
                key = "_distract_eb"
            else:
                continue
            if key not in added_overlaps:
                added_overlaps.add(key)
                rgb = CATEGORY_COLORS[key]
                legend_handles.append(mpatches.Patch(color=rgb, label=overlap_labels[key]))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.04), ncol=len(legend_handles),
                  fontsize=font_size * 3, frameon=False)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _compute_support_sets(token_neighbors_list, target_texts, num_tokens):
    """Return {target_name: set of token indices} where that target passes Kneedle threshold."""
    support = {tname: set() for tname in target_texts}
    for idx in range(min(len(token_neighbors_list), num_tokens)):
        nns = token_neighbors_list[idx]
        sims = [nn["similarity"] for nn in nns] if nns else []
        threshold = kneedle_threshold(sims) if sims else 0.0
        for tname, ttext in target_texts.items():
            sim = _match_target_in_neighbors(nns, ttext)
            if sim is not None and sim >= threshold:
                support[tname].add(idx)
    return support


def render_similarity_heatmap(image, grid_thw, token_neighbors_list, target_texts,
                              title="", output_path=None, dpi=200):
    """Cosine similarity heatmap for each target text overlaid on the image."""
    t, h, w = grid_thw
    num_tokens = h * w
    if len(token_neighbors_list) < num_tokens:
        token_neighbors_list = token_neighbors_list + [[]] * (num_tokens - len(token_neighbors_list))
    token_neighbors_list = token_neighbors_list[:num_tokens]

    targets = list(target_texts.items())
    ncols = 1 + len(targets)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    img_arr = np.array(image.resize((w * 14, h * 14), Image.BILINEAR)) if image else None

    # Original image
    if img_arr is not None:
        axes[0].imshow(img_arr)
    axes[0].set_title("original", fontsize=8)
    axes[0].axis("off")

    # Pre-compute grids for shared vmin/vmax
    grids = []
    for tname, ttext in targets:
        grid = np.zeros((h, w), dtype=np.float32)
        for row in range(h):
            for col in range(w):
                idx = row * w + col
                sim = _match_target_in_neighbors(token_neighbors_list[idx], ttext)
                grid[row, col] = sim if sim is not None else 0.0
        grids.append(grid)

    vmin = 0.0
    vmax = max((g.max() for g in grids), default=1.0)
    if vmax <= 0:
        vmax = 1.0  # Keep 0 as darkest when all values are zero

    for i, ((tname, ttext), grid) in enumerate(zip(targets, grids)):
        ax = axes[i + 1]
        if img_arr is not None:
            ax.imshow(img_arr, alpha=0.3)
        im = ax.imshow(grid, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax,
                       extent=[0, img_arr.shape[1] if img_arr is not None else w,
                               img_arr.shape[0] if img_arr is not None else h, 0],
                       interpolation="bilinear")
        ax.set_title(f"{tname}: {ttext}", fontsize=8,
                     **_text_style_kwargs(f"{tname}: {ttext}"))
        ax.axis("off")

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def render_cross_condition_heatmap(image, grid_thw, conditions_data, target_texts,
                                   title="", output_path=None, dpi=200):
    """Cross-condition heatmap: rows = targets (ground_truth, expected_bias, distract),
    cols = conditions, with one shared colorbar on the right."""
    t, h, w = grid_thw
    num_tokens = h * w

    # Canonical row order
    row_order = [k for k in ["ground_truth", "expected_bias", "distract"]
                 if k in target_texts]
    if not row_order:
        return
    cond_names = list(conditions_data.keys())
    ncols = len(cond_names)
    nrows = len(row_order)
    if ncols == 0:
        return

    img_arr = np.array(image.resize((w * 14, h * 14), Image.BILINEAR)) if image else None

    # Pre-compute all grids: all_grids[row][col]
    all_grids = []
    for tname in row_order:
        ttext = target_texts[tname]
        row_grids = []
        for cond_name in cond_names:
            nns_list = conditions_data[cond_name]
            if len(nns_list) < num_tokens:
                nns_list = nns_list + [[]] * (num_tokens - len(nns_list))
            grid = np.zeros((h, w), dtype=np.float32)
            for row in range(h):
                for col in range(w):
                    idx = row * w + col
                    sim = _match_target_in_neighbors(nns_list[idx], ttext)
                    grid[row, col] = sim if sim is not None else 0.0
            row_grids.append(grid)
        all_grids.append(row_grids)

    vmin = 0.0
    vmax = max(g.max() for rg in all_grids for g in rg)
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 2.0, 4 * nrows),
                              squeeze=False)

    for ri, tname in enumerate(row_order):
        ttext = target_texts[tname]
        for ci, cond_name in enumerate(cond_names):
            ax = axes[ri, ci]
            grid = all_grids[ri][ci]
            if img_arr is not None:
                ax.imshow(img_arr, alpha=0.3)
            im = ax.imshow(grid, cmap="hot", alpha=0.7, vmin=vmin, vmax=vmax,
                           extent=[0, img_arr.shape[1] if img_arr is not None else w,
                                   img_arr.shape[0] if img_arr is not None else h, 0],
                           interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ri == 0:
                ax.set_title(cond_name, fontsize=32, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{ttext}", fontsize=32, fontweight="bold",
                              **_text_style_kwargs(f"{ttext}"))
            if ci == ncols - 1:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(tname, fontsize=32, fontweight="bold",
                               rotation=270, labelpad=32)

    # Single colorbar on the right — pad keeps it off the last column
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.02, pad=0.20, shrink=0.8)
    cbar.ax.tick_params(labelsize=20)

    fig.tight_layout(rect=[0, 0, 0.80, 1.0])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _condition_to_shots(cond_name):
    """Convert condition name to ICL shot count (zero_shot -> 0, icl_N -> N)."""
    if cond_name == "zero_shot":
        return 0
    if cond_name.startswith("icl_"):
        return int(cond_name.split("_")[1])
    return None


def render_token_change_plot(entries_data, title="", output_path=None, dpi=200):
    """Plot mean similarity of target tokens (ground_truth, expected_bias, distract) vs ICL shots.

    entries_data: list of dicts, each with keys:
        "target_texts": {target_name: target_text, ...}
        "conditions": {cond_name: token_neighbors_list, ...}  (last layer only)
    """
    # Collect per-(target, shot_count) similarities across entries
    # target_sims[target_name][shot_count] = list of per-entry mean similarities
    target_sims = {}
    for ed in entries_data:
        targets = ed["target_texts"]
        for cond_name, token_neighbors in ed["conditions"].items():
            shots = _condition_to_shots(cond_name)
            if shots is None:
                continue
            for tname, ttext in targets.items():
                sims = []
                for token_nns in token_neighbors:
                    s = _match_target_in_neighbors(token_nns, ttext)
                    sims.append(s if s is not None else 0.0)
                mean_sim = np.mean(sims) if sims else 0.0
                target_sims.setdefault(tname, {}).setdefault(shots, []).append(mean_sim)

    if not target_sims:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    reg_lines = []

    for tname in ["ground_truth", "expected_bias", "distract"]:
        if tname not in target_sims:
            continue
        shot_map = target_sims[tname]
        shots_sorted = sorted(shot_map.keys())
        means = [np.mean(shot_map[s]) for s in shots_sorted]
        sems = [np.std(shot_map[s]) / max(np.sqrt(len(shot_map[s])), 1) for s in shots_sorted]
        color = TARGET_COLORS.get(tname, (0.5, 0.5, 0.5))

        ax.errorbar(shots_sorted, means, yerr=sems, marker="o", color=color,
                    label=tname, capsize=3, linewidth=1.5, markersize=5)

        # Linear regression
        x = np.array(shots_sorted, dtype=float)
        y = np.array(means, dtype=float)
        if len(x) >= 2:
            res = stats.linregress(x, y)
            reg_lines.append(f"{tname}: slope={res.slope:.4f}, R²={res.rvalue**2:.3f}, p={res.pvalue:.4g}")
            ax.plot(x, res.intercept + res.slope * x, "--", color=color, alpha=0.5, linewidth=1)

    ax.set_xlabel("Number of ICL demos")
    ax.set_ylabel("Mean similarity")
    ax.legend(fontsize=8)
    if reg_lines:
        ax.text(0.02, 0.02, "\n".join(reg_lines), transform=ax.transAxes,
                fontsize=6.5, verticalalignment="bottom", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if title:
        ax.set_title(title)
    finish_figure(fig, ax)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Print regression results
    for line in reg_lines:
        print(f"  [token_change] {line}")


# Hatch pattern per set — colorblind-friendly texture for Venn patches.
_VENN_HATCHES = {
    "distract": "///",
    "ground_truth": "\\\\",
    "expected_bias": "xx",
}


def _pretty_task_name(task):
    return task.replace("_", " ").title()


def _pairwise_jaccards_from_triple(tc):
    """Return (J(D,G), J(D,E), J(G,E)) from a triple-counts dict."""
    def _j(num, denom):
        return num / denom if denom > 0 else 0.0
    dg = _j(tc["110"] + tc["111"],
            tc["100"] + tc["010"] + tc["110"] + tc["101"] + tc["011"] + tc["111"])
    de = _j(tc["101"] + tc["111"],
            tc["100"] + tc["001"] + tc["110"] + tc["101"] + tc["011"] + tc["111"])
    ge = _j(tc["011"] + tc["111"],
            tc["010"] + tc["001"] + tc["110"] + tc["101"] + tc["011"] + tc["111"])
    return dg, de, ge


def _render_single_venn3(ax, tc, title_prefix="", title_fontsize=16,
                         show_set_labels=True, show_title=True):
    """Render a single 3-set Venn (Distractor / Ground Truth / Expected Bias) into ax.

    Returns (j_dg, j_de, j_ge, union, neither) or None if no data.
    """
    union = sum(tc[k] for k in ("100", "010", "001", "110", "101", "011", "111"))
    neither = tc.get("neither", 0)

    if union == 0:
        ax.text(0.5, 0.5, "No overlap data", ha="center", va="center",
                transform=ax.transAxes, fontsize=16, color="gray")
        if show_title:
            ax.set_title(f"{title_prefix}Distractor / Ground Truth / Expected Bias",
                         fontsize=title_fontsize)
        ax.axis("off")
        return None

    set_names = ("Distractor", "Ground Truth", "Expected Bias") if show_set_labels else ("", "", "")
    v = venn3(
        subsets=(tc["100"], tc["010"], tc["110"],
                 tc["001"], tc["101"], tc["011"], tc["111"]),
        set_labels=set_names,
        set_colors=(CATEGORY_COLORS["distract_cue"],
                    CATEGORY_COLORS["ground_truth"],
                    CATEGORY_COLORS["expected_bias"]),
        alpha=0.45,
        ax=ax,
    )

    for rid, hatch_key in (("100", "distract"),
                           ("010", "ground_truth"),
                           ("001", "expected_bias")):
        p = v.get_patch_by_id(rid)
        if p is not None:
            p.set_hatch(_VENN_HATCHES[hatch_key])

    for rid in ("100", "010", "001", "110", "101", "011", "111"):
        lbl = v.get_label_by_id(rid)
        if lbl is not None and tc[rid] > 0:
            pct = 100 * tc[rid] / union
            lbl.set_text(f"{pct:.1f}%")
            lbl.set_fontsize(18)
            lbl.set_fontweight("bold")
        elif lbl is not None:
            lbl.set_text("")

    if show_set_labels:
        for set_id in ("A", "B", "C"):
            set_lbl = v.get_label_by_id(set_id)
            if set_lbl is not None:
                set_lbl.set_fontsize(20)
                set_lbl.set_fontweight("bold")

    j_dg, j_de, j_ge = _pairwise_jaccards_from_triple(tc)
    if show_title:
        ax.set_title(
            f"{title_prefix}n={union}, excl. neither={neither}\n"
            f"J(D\u2229G)={j_dg:.3f}, J(D\u2229E)={j_de:.3f}, J(G\u2229E)={j_ge:.3f}",
            fontsize=title_fontsize,
        )
    return j_dg, j_de, j_ge, union, neither


def _accumulate_triple_counts(entries_data):
    """Return {shots: triple_counts_dict} aggregated across *entries_data*.

    Shares logic with `render_mutual_exclusivity_venns` so cross-task aggregation
    in `main()` can reuse the same region definitions.
    """
    triple_counts = {}
    for ed in entries_data:
        targets = ed["target_texts"]
        num_tokens = ed["num_tokens"]
        for cond_name, token_neighbors in ed["conditions"].items():
            shots = _condition_to_shots(cond_name)
            if shots is None:
                continue
            support = _compute_support_sets(token_neighbors, targets, num_tokens)
            if not all(k in support for k in ("distract", "ground_truth", "expected_bias")):
                continue
            all_idx = set(range(num_tokens))
            sA, sB, sC = support["distract"], support["ground_truth"], support["expected_bias"]
            regions = {
                "100": len((sA - sB) - sC),
                "010": len((sB - sA) - sC),
                "001": len((sC - sA) - sB),
                "110": len((sA & sB) - sC),
                "101": len((sA & sC) - sB),
                "011": len((sB & sC) - sA),
                "111": len(sA & sB & sC),
                "neither": len(all_idx - (sA | sB | sC)),
            }
            tc = triple_counts.setdefault(shots, {k: 0 for k in regions})
            for k, v in regions.items():
                tc[k] += v
    return triple_counts


def render_combined_task_venn3(task_triple_counts, shots, output_path, dpi=200,
                                task_order=("counting", "color_identification",
                                            "shape_identification")):
    """1x3 grid of venn3 subplots, one per task.

    task_triple_counts: {task_name: tc_dict}. Missing tasks render a placeholder.
    shots: int number of ICL demos (0 for zero-shot).
    """
    cond_label = "Zero-Shot" if shots == 0 else f"{shots}-Shot"
    ncols = len(task_order)
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 * ncols, 7.5), squeeze=False)
    fig.suptitle(cond_label, fontsize=26, fontweight="bold", y=0.98)

    for i, task in enumerate(task_order):
        ax = axes[0, i]
        tc = task_triple_counts.get(task)
        if tc is None:
            ax.text(0.5, 0.5, f"N/A\n{_pretty_task_name(task)}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=18, color="gray")
            ax.axis("off")
            continue
        stats = _render_single_venn3(ax, tc, show_set_labels=False, show_title=False)
        ax.set_title(_pretty_task_name(task), fontsize=18, fontweight="bold", pad=12)
        if stats:
            j_dg, j_de, j_ge, union, neither = stats
            ax.text(0.5, -0.06,
                    f"n = {union}  (excl. neither = {neither})\n"
                    f"J(D\u2229G) = {j_dg:.3f}   J(D\u2229E) = {j_de:.3f}   J(G\u2229E) = {j_ge:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=15, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              alpha=0.85, edgecolor="0.8"))

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS["expected_bias"], alpha=0.6,
                       label="Expected Bias"),
        mpatches.Patch(color=CATEGORY_COLORS["ground_truth"], alpha=0.6,
                       label="Ground Truth"),
        mpatches.Patch(color=CATEGORY_COLORS["distract_cue"], alpha=0.6,
                       label="Distractor"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=18, frameon=True, edgecolor="0.8",
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_mutual_exclusivity_venns(entries_data, title="", output_path=None, dpi=200):
    """Plot Venn diagrams for distract vs ground_truth and distract vs expected_bias.

    Writes multiple files under the directory of ``output_path``:
      - ``venn2_{cond}.png`` — 2-set Venns (distract vs GT, distract vs EB) for each condition
      - ``venn3_{cond}.png`` — single 3-set Venn (distract, GT, EB) for each condition

    Percentages are computed against the union (excluding tokens matching neither).

    entries_data: list of dicts, each with keys:
        "target_texts": {target_name: target_text, ...}
        "conditions": {cond_name: token_neighbors_list, ...}  (last layer only)
        "num_tokens": int
    """
    # Per-condition support accumulators
    # pair_tables[shots][(distract, other)] = 2x2 array for 2-set Venns
    # triple_counts[shots] = dict of 7 regions for 3-set Venn
    pair_tables = {}
    triple_counts = {}
    pairs = [("distract", "ground_truth"), ("distract", "expected_bias")]

    for ed in entries_data:
        targets = ed["target_texts"]
        num_tokens = ed["num_tokens"]
        for cond_name, token_neighbors in ed["conditions"].items():
            shots = _condition_to_shots(cond_name)
            if shots is None:
                continue
            support = _compute_support_sets(token_neighbors, targets, num_tokens)
            all_idx = set(range(num_tokens))

            # --- 2-set tables ---
            for dist_name, other_name in pairs:
                if dist_name not in support or other_name not in support:
                    continue
                s_dist, s_other = support[dist_name], support[other_name]
                both = len(s_dist & s_other)
                dist_only = len(s_dist - s_other)
                other_only = len(s_other - s_dist)
                neither = len(all_idx - (s_dist | s_other))

                key = (dist_name, other_name)
                pair_tables.setdefault(shots, {})
                if key not in pair_tables[shots]:
                    pair_tables[shots][key] = np.zeros((2, 2), dtype=int)
                pair_tables[shots][key][0, 0] += both
                pair_tables[shots][key][0, 1] += dist_only
                pair_tables[shots][key][1, 0] += other_only
                pair_tables[shots][key][1, 1] += neither

            # --- 3-set regions: keys '100','010','001','110','101','011','111' ---
            if all(k in support for k in ("distract", "ground_truth", "expected_bias")):
                sA = support["distract"]
                sB = support["ground_truth"]
                sC = support["expected_bias"]
                regions = {
                    "100": len((sA - sB) - sC),
                    "010": len((sB - sA) - sC),
                    "001": len((sC - sA) - sB),
                    "110": len((sA & sB) - sC),
                    "101": len((sA & sC) - sB),
                    "011": len((sB & sC) - sA),
                    "111": len(sA & sB & sC),
                    "neither": len(all_idx - (sA | sB | sC)),
                }
                tc = triple_counts.setdefault(shots, {k: 0 for k in regions})
                for k, v in regions.items():
                    tc[k] += v

    if not pair_tables and not triple_counts:
        return

    pair_titles = {
        ("distract", "ground_truth"): "Distractor vs Ground Truth",
        ("distract", "expected_bias"): "Distractor vs Expected Bias",
    }
    right_set_labels = {
        ("distract", "ground_truth"): "Ground Truth",
        ("distract", "expected_bias"): "Expected Bias",
    }
    right_set_colors = {
        ("distract", "ground_truth"): CATEGORY_COLORS["ground_truth"],
        ("distract", "expected_bias"): CATEGORY_COLORS["expected_bias"],
    }

    out_dir = Path(output_path).parent if output_path else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- 2-set Venns per condition -----------------
    shots_sorted = sorted(pair_tables.keys())
    available_pairs = [p for p in pairs if any(p in pair_tables.get(s, {}) for s in shots_sorted)]

    for shots in shots_sorted:
        cond_label = "zero-shot" if shots == 0 else f"{shots}-shot"
        slug = "zeroshot" if shots == 0 else f"{shots}shot"
        ncols = len(available_pairs)
        if ncols == 0:
            break
        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.0), squeeze=False)

        for j, pair in enumerate(available_pairs):
            ax = axes[0, j]
            if pair not in pair_tables.get(shots, {}):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=18, color="gray")
                ax.set_title(f"{cond_label}: {pair_titles[pair]}", fontsize=16)
                ax.axis("off")
                continue

            table = pair_tables[shots][pair]
            both = int(table[0, 0])
            dist_only = int(table[0, 1])
            other_only = int(table[1, 0])
            neither = int(table[1, 1])
            union = both + dist_only + other_only
            jaccard = both / union if union > 0 else 0.0

            subtitle = (f"{cond_label}: {pair_titles[pair]}\n"
                        f"Jaccard = {jaccard:.3f}  "
                        f"(n={union}, excl. neither={neither})")

            if union == 0:
                ax.text(0.5, 0.5, "No overlap data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=16, color="gray")
                ax.set_title(subtitle, fontsize=16)
                ax.axis("off")
                continue

            v = venn2(
                subsets=(dist_only, other_only, both),
                set_labels=("Distractor", right_set_labels[pair]),
                set_colors=(CATEGORY_COLORS["distract_cue"], right_set_colors[pair]),
                alpha=0.45,
                ax=ax,
            )
            # Hatching on singleton regions — overlap stays clean.
            other_hatch_key = pair[1]  # "ground_truth" or "expected_bias"
            for rid, hatch_key in (("10", "distract"), ("01", other_hatch_key)):
                p = v.get_patch_by_id(rid)
                if p is not None:
                    p.set_hatch(_VENN_HATCHES[hatch_key])
            for region_id, count in (("10", dist_only), ("01", other_only), ("11", both)):
                lbl = v.get_label_by_id(region_id)
                if lbl is not None:
                    pct = 100 * count / union
                    lbl.set_text(f"{count}\n({pct:.1f}%)")
                    lbl.set_fontsize(18)
                    lbl.set_fontweight("bold")
            for set_id in ("A", "B"):
                set_lbl = v.get_label_by_id(set_id)
                if set_lbl is not None:
                    set_lbl.set_fontsize(18)
                    set_lbl.set_fontweight("bold")
            ax.set_title(subtitle, fontsize=16)

        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"venn2_{slug}.png",
                        dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    # ----------------- 3-set Venn per condition -----------------
    for shots in sorted(triple_counts.keys()):
        cond_label = "zero-shot" if shots == 0 else f"{shots}-shot"
        slug = "zeroshot" if shots == 0 else f"{shots}shot"
        tc = triple_counts[shots]

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        _render_single_venn3(ax, tc, title_prefix=f"{cond_label}  ",
                             title_fontsize=16)

        plt.tight_layout()
        if out_dir is not None:
            fig.savefig(out_dir / f"venn3_{slug}.png",
                        dpi=dpi, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    # Print summary
    for shots in shots_sorted:
        cond_label = "zero-shot" if shots == 0 else f"{shots}-shot"
        for pair in available_pairs:
            if pair in pair_tables.get(shots, {}):
                table = pair_tables[shots][pair]
                both = int(table[0, 0])
                dist_only = int(table[0, 1])
                other_only = int(table[1, 0])
                neither = int(table[1, 1])
                union = both + dist_only + other_only
                jaccard = both / union if union > 0 else 0.0
                if union > 0:
                    pct_d = 100 * dist_only / union
                    pct_b = 100 * both / union
                    pct_o = 100 * other_only / union
                else:
                    pct_d = pct_b = pct_o = 0.0
                print(f"  [mutual_excl] {cond_label} {pair[1]}: "
                      f"Jaccard={jaccard:.3f}  "
                      f"dist_only={pct_d:.1f}% both={pct_b:.1f}% other_only={pct_o:.1f}%  "
                      f"(neither={neither} excluded)")


def _render_map_job(job):
    image, grid_thw, labels, sims, title, out_path, dpi, font_size = job
    render_token_grid(image, grid_thw, labels, sims,
                      title=title, output_path=out_path,
                      dpi=dpi, font_size=font_size)


def _render_highlight_job(job):
    image, grid_thw, tok_matches, cat_names, title, out_path, dpi, font_size = job
    render_highlight_grid(image, grid_thw, tok_matches, cat_names,
                          title=title, output_path=out_path,
                          dpi=dpi, font_size=font_size)


def _render_heatmap_job(job):
    image, grid_thw, token_nns, targets, title, out_path, dpi = job
    render_similarity_heatmap(image, grid_thw, token_nns, targets,
                              title=title, output_path=out_path, dpi=dpi)


def _render_cross_heatmap_job(job):
    image, grid_thw, conds_data, target_texts, title, out_path, dpi = job
    render_cross_condition_heatmap(image, grid_thw, conds_data, target_texts,
                                  title=title, output_path=out_path, dpi=dpi)


def _render_job(job):
    kind, payload = job
    if kind == "map":
        _render_map_job(payload)
    elif kind == "highlight":
        _render_highlight_job(payload)
    elif kind == "cross_heatmap":
        _render_cross_heatmap_job(payload)


def main():
    setup_style()
    parser = argparse.ArgumentParser(description="LatentLens Visual Token Grid Visualization")
    parser.add_argument("--base_path", type=str, default=None,
                        help="Result root (default: <results_dir>/formal_lens_visualize/<dataset>)")
    parser.add_argument("--results_dir", type=str, default="output",
                        help="Root of the experiment results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="visual_reasoning")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Filter to specific model names (substring match against directory name).")
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                        choices=["early", "middle", "late"],
                        help="Which layer groups to plot (default: all). "
                             "Maps to the three saved layers in order.")
    parser.add_argument("--plot_type", type=str, default="all",
                        choices=["map", "highlight", "both", "token_change", "mutual_exclusivity", "all"],
                        help="Which plot type(s) to generate (default: all).")
    parser.add_argument("--max_entries_per_task", type=int, default=20,
                        help="Max entries to load per task for statistical aggregation (default: 20).")
    parser.add_argument("--max_vis_entries", type=int, default=5,
                        help="Max entries for per-image visualization plots (map/highlight/heatmap, default: 5).")
    parser.add_argument("--cross_condition_heatmap", action="store_true", default=True,
                        help="Also generate cross-condition combined heatmaps (default: True).")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--font_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 1, 8),
                        help="Number of parallel workers for plotting (default: min(cpu_count, 8)). "
                             "Set to 1 to disable multiprocessing.")
    args = parser.parse_args()
    if args.base_path is None:
        args.base_path = str(Path(args.results_dir) / "formal_lens_visualize" / args.dataset)

    output_base = Path(args.output_dir) if args.output_dir else Path("output_vis/formal_lens_visualize")

    # Discover result files
    entries = discover_result_files(args.base_path)
    if not entries:
        print(f"No result files found under {args.base_path}")
        return
    print(f"Found {len(entries)} result file(s)")

    if args.tasks:
        entries = [e for e in entries if e[0] in args.tasks]
    if args.models:
        entries = [e for e in entries
                   if any(m.lower() in e[1].lower() for m in args.models)]

    # Detect dataset from result metadata (fall back to --dataset flag)
    dataset = args.dataset
    for _, _, _, json_path in entries:
        meta = load_metadata(json_path)
        if meta:
            dataset = meta.get("dataset", dataset)
            break
    print(f"Using dataset: {dataset}")

    # Load dataset for image lookup
    try:
        data = prepare_data(dataset=dataset, tasks=args.tasks)
        image_lookup = {entry["image_id"]: entry for entry in data}
        print(f"Loaded {len(image_lookup)} images from dataset")
    except Exception as e:
        print(f"Warning: Could not load dataset for image backgrounds: {e}")
        image_lookup = {}

    do_map = args.plot_type in ("map", "both", "all")
    do_highlight = args.plot_type in ("highlight", "both", "all")
    do_token_change = args.plot_type in ("token_change", "all")
    do_mutual_excl = args.plot_type in ("mutual_exclusivity", "all")

    # First pass: collect all plot jobs so we can show a progress bar
    map_jobs = []
    highlight_jobs = []
    cross_heatmap_jobs = []
    # token_change: keyed by (task, model, experiment) -> list of entry dicts
    token_change_data = {}
    # mutual_exclusivity: same keying as token_change
    mutual_excl_data = {}
    # track visualization entries per group (limit to max_vis_entries)
    vis_entry_counts = {}

    for task, model_name, experiment, json_path in tqdm(entries, desc="Files"):
        for entry_key, entry_data in tqdm(
                stream_results_entries(json_path, max_entries=args.max_entries_per_task),
                desc=f"  {model_name}/{experiment}", leave=False):

            image_id = entry_data.get("image_id", entry_key.split("__")[0])
            qtype = entry_data.get("question_type", entry_key.split("__")[-1])
            grid_thw = entry_data.get("grid_thw", [1, 1, 1])

            entry_keywords = _build_entry_keywords(entry_data, task)
            compiled_pats = _compile_keyword_patterns(entry_keywords) if entry_keywords else None
            category_names = list(entry_keywords.keys()) if entry_keywords else []

            # Gate per-image visualizations by max_vis_entries
            agg_key = (task, model_name, experiment)
            vis_count = vis_entry_counts.get(agg_key, 0)
            do_vis_for_entry = vis_count < args.max_vis_entries

            # Build target texts for heatmap/token_change/mutual_excl plots
            target_texts = {}
            ds_entry = image_lookup.get(image_id)
            if args.cross_condition_heatmap or do_token_change or do_mutual_excl:
                # Prefer distract from result JSON, fall back to dataset entry
                distract = entry_data.get("distract")
                if distract is None and ds_entry:
                    distract = ds_entry.get("distract")
                if ds_entry:
                    for q in ds_entry.get("questions", []):
                        if q.get("type") == qtype:
                            target_texts = _get_target_texts(q, distract=distract)
                            break

            # Only decode the image when this entry will actually emit per-image figures.
            query_image = None
            if do_vis_for_entry and image_id in image_lookup:
                try:
                    query_image = resolve_image(image_lookup[image_id]["image"], prefer_pil=True)
                    if isinstance(query_image, str):
                        query_image = Image.open(query_image)
                    query_image = query_image.convert("RGB")
                except Exception:
                    pass

            # Collect per-condition neighbor data for cross-condition heatmap
            cross_cond_data = {}  # layer_key -> {condition -> token_neighbors}

            for condition, raw_layer_data in entry_data["conditions"].items():
                # Multi-demo-set format: ICL conditions stored as list of layer_data dicts
                if isinstance(raw_layer_data, list):
                    layer_data = _average_demo_sets(raw_layer_data)
                else:
                    layer_data = raw_layer_data
                # Map layer keys (sorted) to early/middle/late
                sorted_layer_keys = sorted(layer_data.keys(),
                                           key=lambda k: int(k.split("_")[-1]))
                label_map = {k: lbl for k, lbl in
                             zip(sorted_layer_keys, ["early", "middle", "late"])}
                if args.layers:
                    sorted_layer_keys = [k for k in sorted_layer_keys
                                         if label_map.get(k) in args.layers]

                for layer_key in sorted_layer_keys:
                    token_neighbors = layer_data[layer_key]
                    fname = f"{image_id}__{qtype}_{condition}_{layer_key}.png"
                    title = f"{image_id} | {qtype} | {condition} | {layer_key}"
                    layer_label = label_map[layer_key]

                    # --- Map jobs (limited to max_vis_entries) ---
                    if do_map and do_vis_for_entry:
                        labels = []
                        sims = []
                        for token_nns in token_neighbors:
                            if token_nns:
                                labels.append(token_nns[0]["token_str"])
                                sims.append(token_nns[0]["similarity"])
                            else:
                                labels.append("")
                                sims.append(0.0)

                        map_path = (output_base / task / model_name / experiment
                                    / "map" / layer_label / fname)
                        map_jobs.append((query_image, tuple(grid_thw), labels,
                                         sims, title, map_path,
                                         args.dpi, args.font_size))

                    # --- Highlight jobs (limited to max_vis_entries) ---
                    if do_highlight and do_vis_for_entry and compiled_pats is not None:
                        token_sims = [
                            float(token_nns[0]["similarity"]) if token_nns else 0.0
                            for token_nns in token_neighbors
                        ]
                        sim_threshold = float(np.percentile(token_sims, 25)) if token_sims else 0.0

                        tok_matches = []
                        for token_nns, sim in zip(token_neighbors, token_sims):
                            if token_nns and sim >= sim_threshold:
                                tok_matches.append(
                                    match_token_keywords(token_nns, compiled_pats))
                            else:
                                tok_matches.append({})

                        hl_path = (output_base / task / model_name / experiment
                                   / "highlight" / layer_label / fname)
                        highlight_jobs.append((query_image, tuple(grid_thw),
                                               tok_matches, category_names,
                                               title, hl_path,
                                               args.dpi, args.font_size))

                    # Accumulate for cross-condition heatmap
                    if args.cross_condition_heatmap and target_texts:
                        cross_cond_data.setdefault(layer_key, {})[condition] = token_neighbors

            # --- Cross-condition heatmap jobs (limited to max_vis_entries) ---
            if args.cross_condition_heatmap and target_texts and do_vis_for_entry:
                for layer_key, conds in cross_cond_data.items():
                    layer_label = label_map.get(layer_key, layer_key)
                    fname = f"{image_id}__{qtype}_{layer_key}.png"
                    title = f"{image_id} | {qtype} | {layer_key}"
                    xhm_path = (output_base / task / model_name / experiment
                                / "heatmap_cross" / layer_label / fname)
                    cross_heatmap_jobs.append((query_image, tuple(grid_thw),
                                                conds, target_texts,
                                                title, xhm_path, args.dpi))

            # Increment visualization counter
            if do_vis_for_entry:
                vis_entry_counts[agg_key] = vis_count + 1

            # --- Token change / mutual exclusivity data collection (last layer only) ---
            if (do_token_change or do_mutual_excl) and target_texts:
                # Use the last (late) layer for each condition
                cond_last_layer = {}
                for condition, raw_ld in entry_data["conditions"].items():
                    ld = _average_demo_sets(raw_ld) if isinstance(raw_ld, list) else raw_ld
                    last_key = sorted(ld.keys(), key=lambda k: int(k.split("_")[-1]))[-1]
                    cond_last_layer[condition] = ld[last_key]
                agg_key = (task, model_name, experiment)
                entry_record = {
                    "target_texts": target_texts,
                    "conditions": cond_last_layer,
                    "num_tokens": grid_thw[1] * grid_thw[2],
                }
                if do_token_change:
                    token_change_data.setdefault(agg_key, []).append(entry_record)
                if do_mutual_excl:
                    mutual_excl_data.setdefault(agg_key, []).append(entry_record)

    all_jobs = ([("map", j) for j in map_jobs]
                + [("highlight", j) for j in highlight_jobs]
                + [("cross_heatmap", j) for j in cross_heatmap_jobs])
    total = len(all_jobs)
    counts = (f"{len(map_jobs)} map, {len(highlight_jobs)} highlight, "
              f"{len(cross_heatmap_jobs)} cross_heatmap")
    print(f"Rendering {total} plots ({counts}) with {args.num_workers} worker(s)...")

    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(_render_job, all_jobs),
                          total=total, desc="Plotting"):
                pass
    else:
        for job in tqdm(all_jobs, desc="Plotting"):
            _render_job(job)

    # --- Token change plots (aggregate, not parallelized) ---
    token_change_count = 0
    for (tc_task, tc_model, tc_exp), tc_entries in token_change_data.items():
        tc_path = output_base / tc_task / tc_model / tc_exp / "token_change" / "token_change.png"
        tc_title = f"{tc_task} | {tc_model} | token similarity vs ICL shots"
        print(f"\n[token_change] {tc_task}/{tc_model}/{tc_exp} ({len(tc_entries)} entries)")
        render_token_change_plot(tc_entries, title=tc_title, output_path=tc_path, dpi=args.dpi)
        token_change_count += 1

    # --- Mutual exclusivity plots (aggregate) ---
    mutual_excl_count = 0
    for (me_task, me_model, me_exp), me_entries in mutual_excl_data.items():
        me_path = (output_base / me_task / me_model / me_exp
                   / "mutual_exclusivity" / "mutual_exclusivity.png")
        me_title = f"{me_task} | {me_model} | support-set Jaccard vs ICL shots"
        print(f"\n[mutual_excl] {me_task}/{me_model}/{me_exp} ({len(me_entries)} entries)")
        render_mutual_exclusivity_venns(me_entries, title=me_title, output_path=me_path, dpi=args.dpi)
        mutual_excl_count += 1

    # --- Cross-task combined venn3 (counting + color_identification + shape_identification) ---
    crosstask_count = 0
    if do_mutual_excl:
        crosstask_tasks = ("counting", "color_identification", "shape_identification")
        crosstask_data = {}  # (model, experiment) -> {task -> {shots -> tc}}
        for (me_task, me_model, me_exp), me_entries in mutual_excl_data.items():
            if me_task not in crosstask_tasks:
                continue
            tc_by_shots = _accumulate_triple_counts(me_entries)
            if not tc_by_shots:
                continue
            crosstask_data.setdefault((me_model, me_exp), {})[me_task] = tc_by_shots

        for (ct_model, ct_exp), by_task in crosstask_data.items():
            all_shots = set()
            for task_map in by_task.values():
                all_shots.update(task_map.keys())
            for shots in sorted(all_shots):
                task_tc = {t: by_task[t][shots]
                           for t in crosstask_tasks
                           if t in by_task and shots in by_task[t]}
                if not task_tc:
                    continue
                slug = "zeroshot" if shots == 0 else f"{shots}shot"
                ct_path = (output_base / "_crosstask" / ct_model / ct_exp
                           / f"venn3_combined_{slug}.png")
                print(f"\n[crosstask] {ct_model}/{ct_exp} {slug} "
                      f"({len(task_tc)}/{len(crosstask_tasks)} tasks present)")
                render_combined_task_venn3(task_tc, shots, ct_path, dpi=args.dpi,
                                           task_order=crosstask_tasks)
                crosstask_count += 1

    total_generated = (total + token_change_count + mutual_excl_count
                       + crosstask_count)
    print(f"\nDone! Generated {total_generated} plots under {output_base}")


if __name__ == "__main__":
    main()

# Example commands:
# Basic run (plots all tasks/models found under base_path):
#   uv run python -m figures.fig_lens_visualize --base_path output/formal_lens_visualize/visual_reasoning_lens
#   uv run python -m figures.fig_lens_visualize --base_path output/formal_lens_visualize/visual_reasoning_control --layers late
#   uv run python -m figures.fig_lens_visualize --base_path output/formal_lens_visualize/visual_reasoning_control --layers late --plot_type highlight
"""
  uv run python -m figures.fig_lens_visualize \
    --base_path output/formal_lens_visualize/visual_reasoning_control \
    --models Qwen2.5-VL-7B Qwen3-VL-8B\
    --plot_type highlight

    uv run python -m figures.fig_lens_visualize \
        --base_path output/formal_lens_visualize/visual_reasoning \
        --plot_type mutual_exclusivity
        # or higher    
"""