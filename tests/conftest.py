import io
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
LABELS = ["1", "2", "3"]
TOPICS = ["animals", "food"]
ROWS_PER_PARTITION = 9


def pytest_configure(config):
    config.addinivalue_line("filterwarnings", "ignore:Partition file not found:UserWarning")


FIXTURE_FILES = {
    "visual_reasoning": [
        "counting_simple.arrow",
        "counting_simple_noprior.arrow",
        "counting_complex.arrow",
        "color_identification_simple.arrow",
        "color_identification_simple_noprior.arrow",
        "color_identification_complex.arrow",
    ],
    "visual_reasoning_stresstest": [
        "counting_simple_stresstest.arrow",
        "counting_simple_noprior_stresstest.arrow",
        "counting_complex_stresstest.arrow",
    ],
    "visual_reasoning_control": [
        "color_identification_control.arrow",
        "color_identification_control2.arrow",
        "color_identification_expand.arrow",
    ],
}


def png_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def write_partition(path, stem):
    rows = range(ROWS_PER_PARTITION)
    base = sum(map(ord, stem)) % 200
    table = pa.table({
        "ID": [f"{stem}_{i}" for i in rows],
        "image": pa.array(
            [{"bytes": png_bytes((base, i * 25, 255 - i * 25)), "path": None} for i in rows],
            type=IMAGE_TYPE,
        ),
        "topic": [TOPICS[i % 2] for i in rows],
        "prompt": [f"How many in {stem} {i}?" for i in rows],
        "ground_truth": [LABELS[i % 3] for i in rows],
        "expected_bias": [LABELS[(i + 1) % 3] for i in rows],
        "verified": [i != 4 for i in rows],
        "pixels": [i * 10 for i in rows],
    })
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for dataset, files in FIXTURE_FILES.items():
        (root / dataset).mkdir()
        for name in files:
            write_partition(root / dataset / name, Path(name).stem)
    return root
