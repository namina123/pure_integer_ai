import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.build_llm_assisted_dialogue_course import (
    COURSE_FORMAT,
    build_llm_assisted_dialogue_course,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _source(root: Path, *, tamper_done: bool = False) -> str:
    root.mkdir()
    family = "supportive_listening"
    source_id = "llm-dialogue-sol-supportive-listening-test"
    records = []
    for episode_ordinal in range(1, 36):
        episode_id = f"{family}-{episode_ordinal:04d}"
        turns = []
        for turn_ordinal in range(1, 9):
            role = 1 if turn_ordinal % 2 else 2
            prefix = "用户补充" if role == 1 else "助手承接"
            turns.append({
                "speaker_role": role,
                "surface": f"{prefix}第{episode_ordinal}段第{turn_ordinal}轮的具体内容。",
                "turn_ordinal": turn_ordinal,
            })
        records.append({
            "contains_external_fact": 0,
            "episode_id": episode_id,
            "family": family,
            "format": "PURE_INTEGER_AI_LLM_DIALOGUE_SOURCE_V1",
            "generator_model": "gpt-5.6-sol",
            "human_generated": 0,
            "language": "zh-CN",
            "quality_tags": ["context_update", "supportive"],
            "source_id": source_id,
            "turns": turns,
        })
    files = {
        "GENERATION_CONTRACT.md": b"contract\n",
        "TASK.md": b"task\n",
        "dataset.jsonl": b"".join(_canonical(item) for item in records),
        "generation_report.json": _canonical({
            "assistant_turn_count": 140,
            "duplicate_assistant_surface_count": 0,
            "episode_count": 35,
            "external_fact_episode_count": 0,
            "family": family,
            "validation_status": "PASS",
        }),
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    manifest = {
        "assistant_turn_count": 140,
        "creation_method": "isolated_codex_authored_dialogue",
        "episode_count": 35,
        "family": family,
        "files": [{
            "bytes": len(payload), "name": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        } for name, payload in sorted(files.items())],
        "format": "PURE_INTEGER_AI_LLM_DIALOGUE_SOURCE_MANIFEST_V1",
        "generator_model": "gpt-5.6-sol",
        "human_generated": 0,
        "license_id": "CC0-1.0",
        "schema_version": 1,
        "source_id": source_id,
    }
    manifest_payload = _canonical(manifest)
    (root / "source_manifest.json").write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    (root / "DONE").write_bytes(
        (("0" * 64 if tamper_done else manifest_sha) + "\n").encode("ascii"))
    return manifest_sha


def test_builds_episode_isolated_heldout_and_structured_turn_course(
        tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest_sha = _source(source)
    output = tmp_path / "course.jsonl"
    result = build_llm_assisted_dialogue_course(
        (source,), output,
        expected_source_manifest_sha256s=(manifest_sha,),
        require_k_drive=False)
    records = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert result["record_count"] == 140
    assert {record["format"] for record in records} == {COURSE_FORMAT}
    assert {record["human_generated"] for record in records} == {0}
    assert {record["license_id"] for record in records} == {"CC0-1.0"}
    split_by_episode = {}
    for record in records:
        episode = "/".join(record["sample_id"].split("/")[:-1])
        split_by_episode.setdefault(episode, set()).add(record["split"])
    assert all(len(splits) == 1 for splits in split_by_episode.values())
    assert {next(iter(splits)) for splits in split_by_episode.values()} == {
        "train", "heldout"}
    final = max(records, key=lambda item: item["path_turn_count"])
    assert final["path_turn_count"] == 8
    assert [turn["speaker_role"] for turn in final["dialogue_turns"]] == [
        1, 2, 1, 2, 1, 2, 1, 2]
    pack = load_dialogue_training_pack((output,))
    assert len(pack.cases) == 140
    assert max(len(case.dialogue_turns) for case in pack.cases) == 8


def test_rejects_non_manifest_last_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest_sha = _source(source, tamper_done=True)
    with pytest.raises(ValueError, match="manifest-last"):
        build_llm_assisted_dialogue_course(
            (source,), tmp_path / "course.jsonl",
            expected_source_manifest_sha256s=(manifest_sha,),
            require_k_drive=False)
