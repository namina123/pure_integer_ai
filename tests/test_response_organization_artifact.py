"""OASST1 response organization artifact and consumer regression."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pure_integer_ai.experiments.build_response_organization_artifact import (
    build_response_organization_artifact,
    load_response_organization_artifact,
)
from pure_integer_ai.experiments.conversation_response_organization import (
    BLOCK_BULLET_LIST,
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_HTML,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    ResponseOrganizationModel,
    organize_response_surface,
    profile_response_surface,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
)


_PACK_SHA = "42" * 32
_SOURCE_SHA = "24" * 32


def _record(sample_id: str, split: str, response: str) -> dict[str, object]:
    return {
        "format": "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2",
        "human_generated": 1,
        "license_id": "Apache-2.0",
        "response_surface": response,
        "sample_id": sample_id,
        "source_sha256": _SOURCE_SHA,
        "split": split,
    }


def _write_course(path: Path) -> str:
    complex_response = (
        "# 步骤\n\n- 第一项\n- 第二项\n\n"
        "```python\nprint(1)\n```\n\n"
        "| 名称 | 值 |\n| --- | --- |\n| 甲 | 一 |\n\n"
        "<div>补充结构</div>")
    rows = (
        _record("train-1", "train", "这是第一句。这是第二句。这是第三句。"),
        _record("train-2", "train", complex_response),
        _record("train-3", "train", "1. 先检查\n2. 再执行"),
        _record("heldout-1", "heldout", "# 结果\n\n- 已完成\n- 可恢复"),
    )
    payload = "".join(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_training_run(root: Path, course_sha: str) -> None:
    root.mkdir()
    (root / "training_cursor.int").write_bytes(b"cursor-v1")
    (root / "training_summary.json").write_text(json.dumps({
        "pack_sha256": _PACK_SHA,
        "run_id": "ft-b-test",
        "stages_completed": [1, 2],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    (root / "dialogue_pack_manifest.json").write_text(json.dumps({
        "pack_sha256": _PACK_SHA,
        "source_files": [["data/ph2/course.jsonl", course_sha, 4]],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_profile_covers_markdown_html_and_has_no_content_model() -> None:
    surface = (
        "开头。\n\n# 标题\n\n- 甲\n- 乙\n\n"
        "```\ncode\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "<section>尾部</section>")
    profile = profile_response_surface(surface)
    assert profile.signature == (
        BLOCK_PARAGRAPH, BLOCK_HEADING, BLOCK_BULLET_LIST,
        BLOCK_CODE, BLOCK_TABLE, BLOCK_HTML,
    )


def test_build_load_and_consume_response_organization(tmp_path: Path) -> None:
    course = tmp_path / "course.jsonl"
    course_sha = _write_course(course)
    training = tmp_path / "training"
    _write_training_run(training, course_sha)
    artifact_root = tmp_path / "artifact"

    artifact = build_response_organization_artifact(
        course_path=course,
        training_run_root=training,
        artifact_root=artifact_root,
        expected_course_sha256=course_sha,
        expected_pack_sha256=_PACK_SHA,
        require_k_drive=False,
    )
    assert artifact.status == "PASS"
    assert artifact.capability_status == "PASS"
    restored = load_response_organization_artifact(
        artifact_root,
        expected_run_id="ft-b-test",
        expected_pack_sha256=_PACK_SHA,
        require_k_drive=False,
    )
    assert ResponseOrganizationModel.from_integer_stream(
        restored.model.integer_stream()) == restored.model

    structured = organize_response_surface(
        restored.model, "# 结论\n\n- 第一项\n- 第二项")
    assert structured.used is True
    assert structured.reason == "organization_validated"
    assert structured.surface == "# 结论\n\n- 第一项\n- 第二项"

    long_plain = (
        "第一部分说明当前条件和范围。第二部分说明执行过程和限制。"
        "第三部分说明最终结果和依据。第四部分说明恢复方式和下一步安排。")
    long_plain += (
        "第五部分说明来源和许可。第六部分说明未知信息的处理方式。"
        "第七部分说明性能边界。第八部分说明后续学习入口。")
    organized = organize_response_surface(restored.model, long_plain)
    assert organized.used is True
    assert "\n\n" in organized.surface
    assert organized.surface.replace("\n", "") == long_plain

    observation = SimpleNamespace(
        run_id="ft-b-test", graph_size=10, training_item_count=3)
    runtime = TrainedSurfaceRuntime(
        observation, {}, organization_model=restored.model)
    rendered = runtime.render("# 结论\n\n- 第一项\n- 第二项")
    assert rendered.used is True
    assert rendered.reason == "organization_validated"
