"""公开对话训练 pack 的大能力级定向验证。"""
from pathlib import Path

from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_training_contrast import (
    build_dialogue_training_contrast,
)
from pure_integer_ai.experiments.run_conversation_training import default_course_paths


def test_public_course_pack_has_train_heldout_negative_and_replays() -> None:
    root = Path(__file__).parents[1] / "data" / "ph2"
    paths = default_course_paths(root.parents[1])
    pack = load_dialogue_training_pack(paths)
    counts = dict(pack.split_counts)
    assert counts["train"] >= 300
    assert counts["heldout"] >= 100
    assert counts["negative"] >= 100
    assert len(pack.training_items()) == counts["train"]
    assert len(pack.training_items(split="heldout")) == counts["heldout"]
    replay = load_dialogue_training_pack(paths)
    assert replay.pack_sha256 == pack.pack_sha256
    assert tuple(item.canonical_record() for item in replay.cases) == tuple(
        item.canonical_record() for item in pack.cases)
    contrast = build_dialogue_training_contrast(pack)
    assert contrast.heldout_id_overlap_count == 0
    assert contrast.negative_id_overlap_count == 0
    assert contrast.heldout_case_count >= 175
    assert contrast.negative_case_count >= 173
    # authored 记录的元数据不能进入语言表层；结构载体则必须保留原文。
    assert all("evaluation_dimension" not in item.raw_text
               for item in pack.cases)
    assert any(item.raw_text.startswith("# 计划") for item in pack.cases)
    assert any(item.raw_text.startswith("<article") for item in pack.cases)
