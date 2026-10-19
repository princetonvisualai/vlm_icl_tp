"""Convert BLINK dataset from parquet to Arrow IPC format."""

from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from vlm_icl.blink import BLINK_RAW_TASKS, convert_rows, rows_to_table

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "blink"
OUT_DIR = Path(__file__).parent.parent / "data" / "blink"


def _convert_split(task_raw: str, task_arrow: str, split: str) -> list:
    """Convert one parquet split file to a list of Arrow-ready row dicts. Returns [] if missing."""
    parquet_path = RAW_DIR / task_raw / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        print(f"  Skipping {task_raw}/{split} (file not found)")
        return []

    return convert_rows(pq.read_table(parquet_path), task_arrow, split)


def _write_arrow(rows: list, out_path: Path) -> None:
    """Write a list of row dicts to an Arrow IPC file."""
    if not rows:
        return

    table = rows_to_table(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(str(out_path), table.schema) as writer:
        writer.write_table(table)
    print(f"  Wrote {table.num_rows} rows → {out_path}")


def main():
    """Convert all BLINK tasks from parquet to Arrow IPC."""
    print(f"Raw data: {RAW_DIR}")
    print(f"Output:   {OUT_DIR}\n")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw BLINK data not found at {RAW_DIR}")

    for task_raw, task_arrow in sorted(BLINK_RAW_TASKS.items()):
        print(f"Processing {task_raw} → {task_arrow}")

        all_rows = []
        for split in ("val", "test"):
            rows = _convert_split(task_raw, task_arrow, split)
            if rows:
                # Write per-split file
                _write_arrow(rows, OUT_DIR / f"{task_arrow}_{split}.arrow")
                all_rows.extend(rows)

        # Write combined default file (val + usable test rows)
        if all_rows:
            _write_arrow(all_rows, OUT_DIR / f"{task_arrow}_default.arrow")

    print("\nDone!")


if __name__ == "__main__":
    main()
