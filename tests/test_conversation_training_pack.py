"""公开对话训练 pack 的大能力级定向验证。"""
from pathlib import Path
import json

from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_training_contrast import (
    build_dialogue_training_contrast,
)
from pure_integer_ai.experiments.dialogue_training_typed_adapter import (
    TypedDialogueCourseAdapter,
)
from pure_integer_ai.experiments.run_conversation_training import default_course_paths
from pure_integer_ai.experiments.integer_token_index import (
    build_integer_aggregate_index, build_integer_token_index,
    load_integer_aggregate_index, write_integer_aggregate_index,
    write_integer_token_index,
)
from pure_integer_ai.experiments.split_indexed_dialogue_course import (
    split_indexed_dialogue_course,
)


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


def test_typed_course_is_explicit_and_stable() -> None:
    """只有已登记 generation payload 进入 typed 适配器，且键可重放。"""
    root = Path(__file__).parents[1] / "data" / "ph2"
    pack = load_dialogue_training_pack(default_course_paths(root.parents[1]))
    adapter = TypedDialogueCourseAdapter()
    report = adapter.report(pack.cases)
    assert report.typed_items == 40
    assert dict(report.by_kind) == {
        "GenerationAdoptionPostcheckQuery": 12,
        "GenerationGeneralizationCandidateV1": 28,
    }
    assert len(report.request_keys) == 27
    assert report.request_keys == adapter.report(pack.cases).request_keys
    assert all(case.typed_payload is None or case.payload_kind is not None
               for case in pack.cases)


def test_compact_course_reconstructs_surface_from_integer_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "course.jsonl.tokens.int.json"
    course = tmp_path / "course.jsonl"
    index = build_integer_token_index(("甲乙重复", "丙丁"),
                                      sequence_keys=("a", "b"))
    write_integer_token_index(sidecar, index)
    rows = [
        {"sample_id": "a", "split": "train", "token_index_file": sidecar.name,
         "token_index_ordinal": 0, "token_index_sha256": index.sha256},
        {"sample_id": "b", "split": "heldout", "token_index_file": sidecar.name,
         "token_index_ordinal": 1, "token_index_sha256": index.sha256},
    ]
    course.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                  for row in rows), encoding="utf-8")
    pack = load_dialogue_training_pack((course,),
                                       source_path_identities={course: "compact/course.jsonl"})
    assert [case.raw_text for case in pack.cases] == ["甲乙重复", "丙丁"]
    assert dict(pack.split_counts) == {"train": 1, "heldout": 1, "negative": 0}


def test_integer_index_deduplicates_repeated_sequences() -> None:
    index = build_integer_token_index(("重复内容", "重复内容", "另一条"),
                                      sequence_keys=("a", "b", "c"))
    assert len(index.sequences) == 2
    assert index.occurrence_ordinals == (0, 0, 1)
    assert [index.render(i) for i in range(3)] == ["重复内容", "重复内容", "另一条"]


def test_integer_index_duplicate_roundtrip_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    index = build_integer_token_index(("相同", "相同", "不同"),
                                      sequence_keys=("a", "b", "c"))
    write_integer_token_index(path, index)
    from pure_integer_ai.experiments.integer_token_index import load_integer_token_index
    replay = load_integer_token_index(path)
    assert replay.sha256 == index.sha256
    assert replay.occurrence_ordinals == (0, 0, 1)


def test_integer_aggregate_index_reuses_sequence_references(tmp_path: Path) -> None:
    """重复结构只保留一条 aggregate，渲染仍由整数引用完整恢复。"""
    tokens = build_integer_token_index(
        ("甲", "乙", "甲乙"), sequence_keys=("a", "b", "c"))
    # 0/1/2 指向 token sequence；3 指向此前登记的 aggregate 0。
    aggregate = build_integer_aggregate_index(
        tokens,
        (("first", (0, 1)), ("second", (0, 1)), ("nested", (3, 2))),
    )
    assert len(aggregate.aggregate_sequences) == 2
    assert aggregate.occurrence_ordinals == (0, 0, 1)
    assert [aggregate.render(tokens, i) for i in range(3)] == [
        "甲乙", "甲乙", "甲乙甲乙"]
    path = tmp_path / "aggregate.json"
    write_integer_aggregate_index(path, aggregate)
    replay = load_integer_aggregate_index(path)
    assert replay.sha256 == aggregate.sha256
    assert replay.render(tokens, 2) == "甲乙甲乙"


def test_integer_aggregate_index_rejects_forward_reference() -> None:
    tokens = build_integer_token_index(("甲",), sequence_keys=("a",))
    try:
        build_integer_aggregate_index(tokens, (("bad", (2,)),))
    except ValueError as error:
        assert "此前 aggregate" in str(error)
    else:
        raise AssertionError("forward aggregate reference must fail closed")


def test_aggregate_course_projection_replays_without_surface_storage(
        tmp_path: Path) -> None:
    """课程 occurrence 只保存 aggregate ordinal，重复段落共享聚合正文。"""
    token_path = tmp_path / "course.jsonl.tokens.int.json"
    aggregate_path = tmp_path / "course.jsonl.aggregates.int.json"
    course_path = tmp_path / "course.jsonl"
    token_index = build_integer_token_index(
        ("重复段落", "重复段落", "另一段"),
        sequence_keys=("source:a", "source:b", "source:c"),
    )
    aggregate_index = build_integer_aggregate_index(
        token_index,
        ((key, (token_index.occurrence_ordinals[index],))
         for index, key in enumerate(("source:a", "source:b", "source:c"))),
    )
    write_integer_token_index(token_path, token_index)
    write_integer_aggregate_index(aggregate_path, aggregate_index)
    rows = [
        {"sample_id": key, "split": "train",
         "token_index_file": token_path.name,
         "token_index_sha256": token_index.sha256,
         "aggregate_index_file": aggregate_path.name,
         "aggregate_index_ordinal": index,
         "aggregate_index_sha256": aggregate_index.sha256}
        for index, key in enumerate(("source:a", "source:b", "source:c"))
    ]
    course_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pack = load_dialogue_training_pack((course_path,))
    assert pack.cases[0].surfaces == ()
    assert pack.cases[0].raw_text == "重复段落"
    assert len(pack.cases[0].aggregate_index.aggregate_sequences) == 2
    items = pack.training_items(defer_indexed_surface=True)
    assert [item.token_values() for item in items] == [
        tuple("重复段落"), tuple("重复段落"), tuple("另一段")]
    items[0].materialize_tokens()
    items[0].release_index_tokens()
    assert items[0].tokens == []


def test_indexed_course_default_projection_keeps_training_tokens(tmp_path: Path) -> None:
    sidecar = tmp_path / "course.jsonl.tokens.int.json"
    course = tmp_path / "course.jsonl"
    index = build_integer_token_index(("保留训练信号",), sequence_keys=("a",))
    write_integer_token_index(sidecar, index)
    course.write_text(json.dumps({
        "sample_id": "a", "split": "train", "token_index_file": sidecar.name,
        "token_index_ordinal": 0, "token_index_sha256": index.sha256,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    pack = load_dialogue_training_pack((course,))
    item = pack.training_items()[0]
    assert item.tokens == list("保留训练信号")
    assert item.raw_text == "保留训练信号"


def test_indexed_course_deferred_projection_materializes_and_releases(tmp_path: Path) -> None:
    sidecar = tmp_path / "course.jsonl.tokens.int.json"
    course = tmp_path / "course.jsonl"
    index = build_integer_token_index(("按需恢复",), sequence_keys=("a",))
    write_integer_token_index(sidecar, index)
    course.write_text(json.dumps({
        "sample_id": "a", "split": "train", "token_index_file": sidecar.name,
        "token_index_ordinal": 0, "token_index_sha256": index.sha256,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    pack = load_dialogue_training_pack((course,))
    item = pack.training_items(defer_indexed_surface=True)[0]
    assert item.tokens == []
    assert item.token_values() == tuple("按需恢复")
    assert item.materialize_tokens() == list("按需恢复")
    item.release_index_tokens()
    assert item.tokens == []


def test_indexed_course_split_preserves_sidecar_and_records(tmp_path: Path) -> None:
    sidecar = tmp_path / "course.jsonl.tokens.int.json"
    course = tmp_path / "course.jsonl"
    index = build_integer_token_index(("甲一", "乙二", "丙三"),
                                      sequence_keys=("a", "b", "c"))
    write_integer_token_index(sidecar, index)
    course.write_text("".join(json.dumps({
        "sample_id": key, "split": "train", "token_index_file": sidecar.name,
        "token_index_ordinal": ordinal, "token_index_sha256": index.sha256,
    }, ensure_ascii=False) + "\n" for ordinal, key in enumerate(("a", "b", "c"))),
                        encoding="utf-8")
    output = tmp_path / "shards"
    reports = split_indexed_dialogue_course(course, output, shard_size=2,
                                            require_k_drive=False)
    assert [item["record_count"] for item in reports] == [2, 1]
    assert (output / sidecar.name).read_bytes() == sidecar.read_bytes()
    shard_paths = tuple(output / str(item["path"]).split("/")[-1]
                        for item in reports)
    pack = load_dialogue_training_pack(shard_paths)
    assert [case.raw_text for case in pack.cases] == ["甲一", "乙二", "丙三"]
