import datasets
import pytest

from data_hf.push_dataset import build_config
from vlm_icl import data
from vlm_icl.data import (
    DATA_SOURCES, HUB_REPOS, VR_DATASETS, dataset_dir, hub_split_name, prepare_data, resolve_data_source,
)
from vlm_icl.images import resolve_image

from conftest import ROWS_PER_PARTITION


def counts(entries):
    out = {}
    for e in entries:
        out[(e["task"], e["split"])] = out.get((e["task"], e["split"]), 0) + 1
    return out


def test_load_with_root_and_dataset_dir(data_root):
    a = prepare_data("visual_reasoning", data_root=str(data_root), data_source="local")
    b = prepare_data("visual_reasoning", data_root=str(data_root / "visual_reasoning"), data_source="local")
    assert len(a) == len(b) == 6 * ROWS_PER_PARTITION
    assert [e["image_id"] for e in a] == [e["image_id"] for e in b]


def test_entry_fields(data_root):
    entry = prepare_data("visual_reasoning", tasks=["counting"], data_root=str(data_root))[0]
    assert entry["image_id"] == "counting_simple_0"
    assert entry["task"] == "counting" and entry["split"] == "simple"
    q = entry["questions"][0]
    assert q["ground_truth"] == "1" and q["expected_bias"] == "2" and q["prompt"]
    assert resolve_image(entry["image"], prefer_pil=True).size == (4, 4)


def test_task_filter(data_root):
    entries = prepare_data("visual_reasoning", tasks=["color_identification"], data_root=str(data_root))
    assert {e["task"] for e in entries} == {"color_identification"}
    assert len(entries) == 3 * ROWS_PER_PARTITION


def test_unknown_task_warns(data_root):
    with pytest.warns(UserWarning, match="Unknown task"):
        assert prepare_data("visual_reasoning", tasks=["nope"], data_root=str(data_root)) == []


def test_topic_and_verified_filter(data_root):
    entries = prepare_data("visual_reasoning", tasks=["counting"], topics=["food"], data_root=str(data_root))
    assert entries and {e["topic"] for e in entries} == {"food"}
    verified = prepare_data("visual_reasoning", tasks=["counting"], verified_only=True, data_root=str(data_root))
    assert all(e["verified"] for e in verified)
    assert len(verified) == 3 * (ROWS_PER_PARTITION - 1)


def test_split_aliases(data_root):
    entries = prepare_data("visual_reasoning", data_root=str(data_root))
    assert {e["split"] for e in entries if e["task"] == "counting"} == {"simple", "noprior", "complex"}
    assert {e["split"] for e in entries if e["task"] == "color_identification"} == {"simple", "noprior", "complex"}
    control = prepare_data("visual_reasoning_control", data_root=str(data_root))
    assert {e["split"] for e in control} == {"control", "control2", "expand"}


def test_stresstest_partitions(data_root):
    entries = prepare_data("visual_reasoning_stresstest", data_root=str(data_root))
    assert counts(entries) == {
        ("counting", split): ROWS_PER_PARTITION for split in ("simple", "noprior", "complex")
    }


def test_sample_per_split(data_root):
    kwargs = dict(dataset="visual_reasoning", sample_per_split=4, data_root=str(data_root))
    a = prepare_data(seed=1, **kwargs)
    b = prepare_data(seed=1, **kwargs)
    c = prepare_data(seed=2, **kwargs)
    ids = lambda es: [e["image_id"] for e in es]
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    assert set(counts(a).values()) == {4}
    assert len(counts(a)) == 6


def test_sample_per_split_caps_at_partition_size(data_root):
    entries = prepare_data("visual_reasoning", tasks=["counting"], sample_per_split=100, seed=0, data_root=str(data_root))
    assert len(entries) == 3 * ROWS_PER_PARTITION


def test_invalid_data_source(data_root):
    with pytest.raises(ValueError, match="data_source"):
        prepare_data("visual_reasoning", data_root=str(data_root), data_source="s3")


def test_unknown_dataset(data_root):
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_data("nonexistent", data_root=str(data_root))


def test_hub_without_mirror(data_root):
    with pytest.raises(ValueError, match="no Hub mirror"):
        prepare_data("visual_reasoning_sample", data_root=str(data_root), data_source="hub")


def test_dataset_dir_forms(tmp_path):
    (tmp_path / "visual_reasoning").mkdir()
    assert dataset_dir("visual_reasoning", str(tmp_path)) == tmp_path / "visual_reasoning"
    assert dataset_dir("visual_reasoning", str(tmp_path / "visual_reasoning")) == tmp_path / "visual_reasoning"


def test_resolve_data_source(tmp_path, monkeypatch):
    assert DATA_SOURCES == ("auto", "local", "hub")
    assert resolve_data_source("visual_reasoning", str(tmp_path), "auto") == "local"
    assert resolve_data_source("visual_reasoning", None, "hub") == "hub"
    assert resolve_data_source("visual_reasoning", str(tmp_path), "local") == "local"
    monkeypatch.setattr(data, "GLOBAL_DATA_DIR", tmp_path / "empty")
    assert resolve_data_source("visual_reasoning", None, "auto") == "hub"
    monkeypatch.setattr(data, "GLOBAL_DATA_DIR", tmp_path)
    (tmp_path / "visual_reasoning").mkdir()
    assert resolve_data_source("visual_reasoning", None, "auto") == "local"
    with pytest.raises(ValueError):
        resolve_data_source("visual_reasoning", None, "bogus")


def test_hub_repos_are_datasets():
    assert set(HUB_REPOS) <= set(VR_DATASETS)


@pytest.mark.parametrize("dataset", sorted(VR_DATASETS))
def test_hub_split_names(dataset):
    partitions, _ = VR_DATASETS[dataset]
    for task, files in partitions.items():
        names = [hub_split_name(task, f) for f, _ in files]
        assert len(set(names)) == len(names), (dataset, task)
        for name in names:
            assert name and ".arrow" not in name and "stresstest" not in name
            assert not name.startswith(task)


def test_hub_split_name_examples():
    assert hub_split_name("counting", "counting_simple_noprior.arrow") == "simple_noprior"
    assert hub_split_name("counting", "counting_simple_noprior_stresstest.arrow") == "simple_noprior"
    assert hub_split_name("color_identification", "color_identification_control2.arrow") == "control2"


@pytest.mark.parametrize("variant,tasks", [
    ("visual_reasoning", ["counting", "color_identification"]),
    ("visual_reasoning_stresstest", ["counting"]),
    ("visual_reasoning_control", ["color_identification"]),
])
def test_hub_round_trip(variant, tasks, data_root, monkeypatch):
    built = {task: build_config(variant, task, str(data_root)) for task in tasks}
    monkeypatch.setattr(datasets, "load_dataset", lambda repo, config, split: built[config][split])

    local = prepare_data(variant, tasks=tasks, data_root=str(data_root), data_source="local")
    hub = prepare_data(variant, tasks=tasks, data_source="hub")
    assert local and len(local) == len(hub)
    for a, b in zip(local, hub):
        assert {k: v for k, v in a.items() if k != "image"} == {k: v for k, v in b.items() if k != "image"}
        pa_, pb = (resolve_image(x["image"], prefer_pil=True) for x in (a, b))
        assert pa_.size == pb.size and pa_.tobytes() == pb.tobytes()
