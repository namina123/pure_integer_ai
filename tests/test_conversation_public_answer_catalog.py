"""DLG-RAW-09 独立 ANSWER catalog 的内容锁与实际运行前闭合专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_answer_catalog import (
    PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1,
    PublicAnswerFrameCatalogError,
    load_public_answer_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameRuntimeRecipe,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]


def _closure() -> PublicSourcePayloadClosureV1:
    """只在 host test boundary 从当前公开资源根构造完整 closure。"""
    return load_public_source_payload_closure_from_root(_ROOT)


def _manifest(closure: PublicSourcePayloadClosureV1) -> dict[str, object]:
    """读取唯一 canonical ANSWER manifest，不允许宽松 JSON transport。"""
    payload = closure.payload_for(PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1)
    assert payload.endswith(b"\n")
    return parse_canonical_json_bytes(payload[:-1], require_object=True)


def _closure_with_answer_manifest(
        closure: PublicSourcePayloadClosureV1,
        manifest: dict[str, object],
        ) -> PublicSourcePayloadClosureV1:
    """只替换内存中的 ANSWER transport，验证 loader 不依赖物理路径。"""
    replacement = canonical_json_line(manifest)
    return build_public_source_payload_closure_v1(tuple(
        public_source_payload_record_from_u8_v1(
            record.logical_key,
            replacement,
        ) if record.logical_key == PUBLIC_ANSWER_FRAME_CATALOG_LOGICAL_KEY_V1
        else record
        for record in closure.records
    ))


def _contains_bytes(record: tuple[int, ...], value: bytes) -> bool:
    """检查 canonical integer record 是否错误地嵌入一段原始 transport bytes。"""
    needle = tuple(value)
    return any(record[index:index + len(needle)] == needle
               for index in range(len(record) - len(needle) + 1))


def test_answer_catalog_closes_pattern_and_claim_source_before_runtime() -> None:
    """ANSWER recipe 必须在 loader 内锁定实际 pattern、structure 与 raw claim span。"""
    closure = _closure()
    catalog = load_public_answer_frame_catalog_from_closure(closure)

    assert len(catalog.frames) == 1
    frame = catalog.frames[0]
    recipe = frame.recipe
    assert isinstance(recipe, PublicFrameRuntimeRecipe)
    assert frame.frame_key == "dlg-raw-public-answer-pier-time-v1"
    assert "".join(chr(value) for value in frame.surface_scalars) == "澄川码头何时启用？"
    assert "".join(chr(value) for value in recipe.claim_scalars) == "澄川码头于2023年启用"
    claim_source = next(
        item for item in frame.source_records
        if item.record_id == recipe.claim_source_record_id)
    course = closure.payload_for(recipe.course_relative_path.encode("ascii"))
    assert course[claim_source.span[0]:claim_source.span[1]] == bytes(
        claim_source.span_bytes)
    assert claim_source.span_scalars == recipe.claim_scalars
    assert recipe.candidate_state.stable_key() == (1, 0)
    assert len(recipe.candidate_evidence) == 1
    assert all(not _contains_bytes(frame.canonical_record(), forbidden)
               for forbidden in (b'"answer_plan"', b'"response_act"'))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pattern_id", 1, "pattern 不属于当前内容锁课程"),
        ("structure_id", 1, "pattern/structure 漂移"),
    ),
)
def test_answer_catalog_rejects_recipe_not_learned_from_locked_course(
        field: str,
        value: int,
        message: str,
        ) -> None:
    """pattern 或 structure 漂移必须在 catalog load 前拒绝，不能拖到 RAW-02。"""
    closure = _closure()
    manifest = _manifest(closure)
    manifest[field] = value

    with pytest.raises(PublicAnswerFrameCatalogError, match=message):
        load_public_answer_frame_catalog_from_closure(
            _closure_with_answer_manifest(closure, manifest))
