from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.experiments.build_kdconv_dialogue_course import (
    build_kdconv_dialogue_course,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)


_COMMIT = "12" * 20


def _snapshot(root: Path) -> None:
    head = root / ".git" / "refs" / "heads" / "master"
    head.parent.mkdir(parents=True)
    head.write_text(_COMMIT + "\n", encoding="ascii")
    (root / "LICENSE").write_text(
        "Apache License\nVersion 2.0\n", encoding="utf-8")
    for domain in ("film", "music", "travel"):
        directory = root / "data" / domain
        directory.mkdir(parents=True)
        value = [{
            "name": "主题",
            "messages": [
                {"message": "最近怎么样？"},
                {"message": "挺好的。", "attrs": [{"name": "主题"}]},
                {"message": "那就好，我们接着聊。"},
            ],
        }]
        for split in ("train", "dev", "test"):
            (directory / f"{split}.json").write_text(
                json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_builds_adjacent_turn_course_with_grounding_boundary(
        tmp_path: Path) -> None:
    source = tmp_path / "source"
    _snapshot(source)
    output = tmp_path / "course.jsonl"
    result = build_kdconv_dialogue_course(
        source, output, expected_commit=_COMMIT, require_k_drive=False)
    rows = [json.loads(item) for item in output.read_text(
        "utf-8").splitlines()]
    assert result["record_count"] == 18
    assert result["split_counts"] == [["train", 6], ["heldout", 12]]
    assert result["grounded_response_count"] == 9
    assert result["intent_support_count"] == 9
    assert {row["format"] for row in rows} == {
        "PURE_INTEGER_AI_KDCONV_DIALOGUE_COURSE_V1"}
    assert all(len(row["source_ref_key"]) == 11 for row in rows)
    assert all([turn["speaker_role"] for turn in row["dialogue_turns"]][-2:]
               == [1, 2] for row in rows)
    assert {row["intent_support"] for row in rows} == {0, 1}
    pack = load_dialogue_training_pack((output,))
    assert len(pack.cases) == 18
