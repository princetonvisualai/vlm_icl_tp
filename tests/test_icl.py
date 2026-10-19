import random

import pytest

from vlm_icl.data import prepare_data
from vlm_icl.icl import (
    build_control_demo_set, build_demo_pool, build_icl_context_pools, build_label_space, corrupt_icl_answers,
    ensure_gt_demo, reposition_gt_demo, select_from_demo_pool, shuffle_icl_answers,
)

KEY = ("counting", "simple")


@pytest.fixture(scope="module")
def pools(data_root):
    entries = prepare_data("visual_reasoning", tasks=["counting"], data_root=str(data_root))
    return build_icl_context_pools(entries)


@pytest.fixture(scope="module")
def demo_pool(pools):
    return build_demo_pool(pools, seed=0)


def ids(demos):
    return [d["entry"]["image_id"] for d in demos]


def test_pools_group_by_task_and_split(pools):
    assert set(pools) == {("counting", "simple"), ("counting", "noprior"), ("counting", "complex")}
    assert all(len(items) == 9 for items in pools.values())


def test_demo_pool_deterministic_and_balanced(pools):
    a, b = build_demo_pool(pools, seed=3), build_demo_pool(pools, seed=3)
    assert {k: ids(v) for k, v in a.items()} == {k: ids(v) for k, v in b.items()}
    labels = [d["question"]["ground_truth"] for d in a[KEY]]
    assert {labels.count(x) for x in set(labels)} == {3}


def test_selection_deterministic_and_excludes_query(pools, demo_pool):
    for item in pools[KEY]:
        a = select_from_demo_pool(item["entry"], item["question"], demo_pool, 4, seed=7)
        b = select_from_demo_pool(item["entry"], item["question"], demo_pool, 4, seed=7)
        assert ids(a) == ids(b) and len(a) == 4
        assert item["entry"]["image_id"] not in ids(a)
        assert len(set(ids(a))) == 4


def test_selection_varies_with_seed(pools, demo_pool):
    item = pools[KEY][0]
    runs = {tuple(ids(select_from_demo_pool(item["entry"], item["question"], demo_pool, 4, seed=s))) for s in range(10)}
    assert len(runs) > 1


def test_selection_too_few_candidates(pools, demo_pool):
    item = pools[KEY][0]
    assert select_from_demo_pool(item["entry"], item["question"], demo_pool, 50, seed=0) is None
    assert select_from_demo_pool({"task": "x", "split": "y", "image_id": "z"}, item["question"], demo_pool, 2) is None


def test_ensure_query_gt_and_position(pools, demo_pool):
    for item in pools[KEY]:
        gt = item["question"]["ground_truth"]
        for position in ("beginning", "end"):
            demos = select_from_demo_pool(
                item["entry"], item["question"], demo_pool, 3, seed=5, query_gt_position=position, ensure_query_gt=True,
            )
            labels = [d["question"]["ground_truth"] for d in demos]
            assert gt in labels
            assert labels.index(gt) == (0 if position == "beginning" else len(labels) - 1)


def test_ensure_gt_demo_swaps_one_in(pools):
    items = pools[KEY]
    selected = [i for i in items if i["question"]["ground_truth"] != "1"][:3]
    out = ensure_gt_demo(selected, "1", items, random.Random(0))
    assert any(d["question"]["ground_truth"] == "1" for d in out)
    assert len(out) == 3
    assert all(d["question"]["ground_truth"] != "1" for d in selected)


def test_ensure_gt_demo_noop_when_present(pools):
    items = pools[KEY]
    selected = [i for i in items if i["question"]["ground_truth"] == "1"][:2]
    assert ensure_gt_demo(selected, "1", items, random.Random(0)) is selected


def test_reposition_gt_demo(pools):
    items = pools[KEY]
    gt = "2"
    mixed = [next(i for i in items if i["question"]["ground_truth"] == g) for g in ("1", "2", "3")]
    assert reposition_gt_demo(mixed, gt, "beginning")[0]["question"]["ground_truth"] == gt
    assert reposition_gt_demo(mixed, gt, "end")[-1]["question"]["ground_truth"] == gt
    assert reposition_gt_demo(mixed, gt, "random") is mixed
    assert ids(reposition_gt_demo(mixed, gt, "end")) != ids(mixed)
    assert len(mixed) == 3


def test_control_demo_set(pools):
    labels = build_label_space(pools)
    demos_a, corrupt_a = build_control_demo_set(pools, max_demos=6, seed=11)
    demos_b, corrupt_b = build_control_demo_set(pools, max_demos=6, seed=11)
    assert {k: ids(v) for k, v in demos_a.items()} == {k: ids(v) for k, v in demos_b.items()}
    assert corrupt_a == corrupt_b
    for key, demos in demos_a.items():
        assert len(demos) == 6 and len(corrupt_a[key]) == 6
        assert set(corrupt_a[key]) <= set(labels[key])
        assert len(set(ids(demos))) == 6


def test_shuffle_and_corrupt_answers(pools):
    context = pools[KEY]
    truth = [c["question"]["ground_truth"] for c in context]
    assert shuffle_icl_answers(context, None, random.Random(0)) == truth
    shuffled = shuffle_icl_answers(context, "random_icd", random.Random(0))
    assert sorted(shuffled) == sorted(truth)
    labels = build_label_space(pools)[KEY]
    corrupted = corrupt_icl_answers(context, "corrupt_icd", random.Random(0), labels)
    assert len(corrupted) == len(truth) and set(corrupted) <= set(labels)
    first = corrupt_icl_answers(context, "corrupt_icd_beginning", random.Random(0), labels)
    assert first[2:] == truth[2:]
