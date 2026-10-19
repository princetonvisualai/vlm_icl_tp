"""Create visual_reasoning_sample: 5 rows per unique ground_truth per (task, split)."""

import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

SRC_DIR = Path(__file__).parent.parent / "data" / "visual_reasoning"
OUT_DIR = Path(__file__).parent.parent / "data" / "visual_reasoning_sample"

PER_LABEL = 5
SEED = 42


def _sample_indices(table: pa.Table, per_label: int, rng: random.Random) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    gts = table["ground_truth"].to_pylist()
    for i, gt in enumerate(gts):
        groups[gt].append(i)

    picked: list[int] = []
    for gt in sorted(groups.keys()):
        idxs = groups[gt]
        n = min(per_label, len(idxs))
        picked.extend(rng.sample(idxs, n))
    picked.sort()
    return picked


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(SRC_DIR.glob("*.arrow"))
    if not files:
        raise FileNotFoundError(f"No .arrow partitions under {SRC_DIR}")

    print(f"Source: {SRC_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Per-label: {PER_LABEL}, seed: {SEED}\n")

    rng = random.Random(SEED)
    total_in = 0
    total_out = 0

    for src in files:
        reader = ipc.open_file(pa.memory_map(str(src), "r"))
        table = reader.read_all()

        indices = _sample_indices(table, PER_LABEL, rng)
        sampled = table.take(pa.array(indices, type=pa.int64()))

        dst = OUT_DIR / src.name
        with ipc.new_file(str(dst), sampled.schema) as writer:
            writer.write_table(sampled)

        n_labels = len(set(table["ground_truth"].to_pylist()))
        print(f"{src.name}: {table.num_rows} -> {sampled.num_rows} rows ({n_labels} labels)")
        total_in += table.num_rows
        total_out += sampled.num_rows

    print(f"\nTotal: {total_in} -> {total_out} rows across {len(files)} files")


if __name__ == "__main__":
    main()
