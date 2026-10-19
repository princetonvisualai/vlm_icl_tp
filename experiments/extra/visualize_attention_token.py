"""Extract token-level attention maps, average across heads and examples, save heatmap PNGs."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from vlm_icl.data import prepare_data
from vlm_icl.models import load_model
from vlm_icl.images import resolve_image
from vlm_icl.paths import build_analysis_path


def _interpolate_attention(attn: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Bilinear-interpolate a (seq, seq) attention map to (grid_size, grid_size)."""
    # attn: (S, S) -> (1, 1, S, S) for F.interpolate
    attn_4d = attn.unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(attn_4d, size=(grid_size, grid_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0)


def main():
    from vlm_icl.args import add_model_args, add_data_args, add_run_config_args

    parser = argparse.ArgumentParser(description="Visualize token-level attention heatmaps")
    add_model_args(parser)
    add_data_args(parser)
    add_run_config_args(parser)
    parser.set_defaults(sample_per_split=20)
    parser.add_argument("--grid_size", type=int, default=1024,
                        help="Interpolation grid size for attention maps (default: 1024)")
    parser.add_argument("--layer_stride", type=int, default=2,
                        help="Sample every N-th layer (default: 2)")
    args = parser.parse_args()

    if args.backend == "vllm":
        raise ValueError("visualize_attention_token requires HF backend (not vllm)")

    random.seed(args.seed)

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, enable_thinking=args.enable_thinking)
    model_type = type(model).__name__
    if model_type not in ("Qwen3VLModel", "Qwen35Model"):
        raise ValueError(f"forward_attentions not supported for {model_type} (API models not supported)")
    print(f"Model loaded: {model_type}")

    print(f"\nPreparing sampled dataset ({args.dataset})...")
    data_kwargs = {"tasks": args.tasks, "sample_per_split": args.sample_per_split}
    data = prepare_data(dataset=args.dataset, seed=args.seed, **data_kwargs, data_root=args.data_root, data_source=args.data_source)
    print(f"Sampled data: {len(data)} entries")

    if not data:
        print("No data entries found!")
        return

    # Group by (task, split)
    groups = defaultdict(list)
    for entry in data:
        groups[(entry["task"], entry.get("split", "default"))].append(entry)

    prefer_pil = True

    for (task_name, split_name), entries in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"Task: {task_name}, Split: {split_name} ({len(entries)} entries)")
        print(f"{'='*60}")

        # Accumulate attention maps per layer
        layer_accum = defaultdict(lambda: torch.zeros(args.grid_size, args.grid_size))
        layer_count = defaultdict(int)

        for entry in tqdm(entries, desc=f"{task_name}/{split_name}", unit="entry"):
            image_obj = entry.get("image")
            image = resolve_image(image_obj, prefer_pil=prefer_pil) if image_obj is not None else None
            extra_imgs = {}
            for extra_key in ("image2", "image3", "image4"):
                extra_obj = entry.get(extra_key)
                if extra_obj is not None:
                    extra_imgs[extra_key] = resolve_image(extra_obj, prefer_pil=prefer_pil)

            for question in entry.get("questions", []):
                try:
                    attn_dict = model.forward_attentions(
                        image=image,
                        prompt=question["prompt"],
                        layer_stride=args.layer_stride,
                        **extra_imgs,
                    )
                    for layer_idx, attn_map in attn_dict.items():
                        interpolated = _interpolate_attention(attn_map, args.grid_size)
                        layer_accum[layer_idx] += interpolated
                        layer_count[layer_idx] += 1

                except torch.cuda.OutOfMemoryError:
                    tqdm.write(f"  CUDA OOM: image_id={entry['image_id']}, skipping")
                    torch.cuda.empty_cache()
                except Exception as e:
                    tqdm.write(f"  Error: {e}, skipping")

        if not layer_count:
            print("  No attention maps collected, skipping.")
            continue

        # Build output path
        leaf = f"sps{args.sample_per_split}_stride{args.layer_stride}"
        output_path = build_analysis_path(
            args.output_dir, "visualize_attention_token", args.dataset,
            task_name, args.model_path, f"{leaf}/{split_name}",
        )
        output_path.mkdir(parents=True, exist_ok=True)

        # Save one heatmap per layer
        for layer_idx in sorted(layer_accum.keys()):
            avg_attn = (layer_accum[layer_idx] / layer_count[layer_idx]).numpy()

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
            ax.set_title(f"Layer {layer_idx} — {task_name}/{split_name} (n={layer_count[layer_idx]})")
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            out_file = output_path / f"layer_{layer_idx}.png"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Saved {len(layer_accum)} heatmaps to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
