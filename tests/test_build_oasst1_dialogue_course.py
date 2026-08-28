import gzip
import hashlib
import json

import pytest

from pure_integer_ai.experiments.build_oasst1_dialogue_course import (
    build_oasst1_dialogue_course,
    build_openassistant_dialogue_course,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.language_observation import (
    _split_item_to_segments,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend


def _message(identity, role, text, *, parent=None, replies=None, rank=None,
             synthetic=False, language="zh"):
    return {
        "message_id": identity,
        "parent_id": parent,
        "text": text,
        "role": role,
        "lang": language,
        "review_count": 2,
        "review_result": True,
        "deleted": False,
        "rank": rank,
        "synthetic": synthetic,
        "replies": [] if replies is None else replies,
    }


def _source(tmp_path):
    followup = _message("a4", "assistant", "可以先补充预算范围。",
                        parent="u3", rank=0)
    user_followup = _message("u3", "prompter", "那应该先做什么？",
                             parent="a2", replies=[followup])
    rejected = _message("a5", "assistant", "合成回复。", parent="u1",
                        rank=0, synthetic=True)
    assistant = _message("a2", "assistant", "先确认目标，再比较方案。",
                         parent="u1", replies=[user_followup], rank=0)
    root = _message("u1", "prompter", "请帮我做一个计划。",
                    replies=[assistant, rejected])
    tree = {"message_tree_id": "u1", "tree_state": "ready_for_export",
            "prompt": root}
    payload = gzip.compress(
        (json.dumps(tree, ensure_ascii=False, separators=(",", ":")) + "\n")
        .encode("utf-8"), mtime=0)
    path = tmp_path / "source.jsonl.gz"
    path.write_bytes(payload)
    return path, payload


def test_builds_reviewed_human_paths_and_pack(tmp_path):
    source, payload = _source(tmp_path)
    output = tmp_path / "course.jsonl"
    result = build_oasst1_dialogue_course(
        source, output, source_sha256=hashlib.sha256(payload).hexdigest(),
        dataset_url="https://example.org/oasst1",
        file_url="https://example.org/oasst1/ready.jsonl.gz",
        heldout_percent=10, require_k_drive=False)
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert result["record_count"] == 2
    assert result["multi_turn_record_count"] == 1
    assert {row["message_id"] for row in rows} == {"a2", "a4"}
    assert rows[1]["input_surface"].endswith("那应该先做什么？")
    assert rows[1]["response_surface"] == "可以先补充预算范围。"
    assert rows[0]["split"] == rows[1]["split"]
    assert all(len(row["source_ref_key"]) == 11 for row in rows)
    assert rows[1]["context_turn_count"] == 3
    assert rows[1]["prompt_turn_ordinal"] == 3
    assert rows[1]["response_turn_ordinal"] == 4
    assert [turn["speaker_role"] for turn in rows[1]["dialogue_turns"]] == [
        1, 2, 1, 2,
    ]
    pack = load_dialogue_training_pack((output,))
    assert len(pack.cases) == 2
    case = pack.cases[1]
    assert "用户：" not in case.raw_text
    assert "助手：" not in case.raw_text
    assert "那应该先做什么？" in case.raw_text
    assert case.raw_text.endswith("可以先补充预算范围。")
    assert len(case.dialogue_turns) == 4
    assert case.source_ref is not None
    item = pack.training_items()[1]
    assert len(item.speaker_spans) == 4
    assert item.speaker_spans[0].speaker != item.speaker_spans[1].speaker
    assert item.speaker_spans[-1].end == len(item.raw_text)
    segments = _split_item_to_segments(item)
    assert [segment.dialogue_turn_ordinal for segment in segments] == [1, 2, 3, 4]
    assert segments[0].speaker_identity == segments[2].speaker_identity
    assert segments[1].speaker_identity == segments[3].speaker_identity
    assert segments[0].speaker_identity != segments[1].speaker_identity
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        speaker_refs = tuple(
            context.graph_ontology.materialize(span.speaker)
            for span in item.speaker_spans
        )
        assert speaker_refs[0] == speaker_refs[2]
        assert speaker_refs[1] == speaker_refs[3]
        assert speaker_refs[0] != speaker_refs[1]
    finally:
        backend.close()


def test_rejects_hash_drift(tmp_path):
    source, _payload = _source(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_oasst1_dialogue_course(
            source, tmp_path / "bad.jsonl", source_sha256="0" * 64,
            dataset_url="https://example.org/oasst1",
            file_url="https://example.org/oasst1/ready.jsonl.gz",
            require_k_drive=False)


def test_oasst2_has_distinct_release_identity_and_reuses_turn_protocol(
        tmp_path):
    source, payload = _source(tmp_path)
    output = tmp_path / "oasst2-course.jsonl"
    build_openassistant_dialogue_course(
        source, output, source_sha256=hashlib.sha256(payload).hexdigest(),
        dataset_url="https://example.org/oasst2",
        file_url="https://example.org/oasst2/ready.jsonl.gz",
        release_id="oasst2", heldout_percent=10, require_k_drive=False)
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert {row["format"] for row in rows} == {
        "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2",
    }
    assert {row["family"] for row in rows} == {"oasst2-human-zh-v1"}
    assert {row["source_title"] for row in rows} == {"OpenAssistant OASST2"}
    pack = load_dialogue_training_pack((output,))
    assert pack.dialogue_structure_counts == (
        ("structured_cases", 2),
        ("turns", 6),
        ("prompt_response_pairs", 2),
        ("multiturn_cases", 1),
    )
