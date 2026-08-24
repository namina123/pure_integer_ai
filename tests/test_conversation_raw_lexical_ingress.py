"""DLG-RAW-01 公开词汇入口的首轮和 fail-closed 专项验证。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_context_runtime import (
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_BOM,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_raw_lexical_ingress import (
    ConversationRawLexicalIngressError,
    ConversationRawLexicalIngressResult,
    ingress_raw_lexical_frame,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_OCCURRENCE_KEY = (65001, 32, 1)


@pytest.fixture(scope="module")
def public_runtime() -> PublicDialogueRuntimeV1:
    """host 先冻结完整 closure，测试侧不再把物理根送入 catalog core。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(_REPOSITORY_ROOT))


@pytest.fixture(scope="module")
def public_catalog(public_runtime: PublicDialogueRuntimeV1) -> PublicFrameCatalog:
    """RAW-01 首轮只消费 runtime 已绑定的 V1 base catalog。"""
    return public_runtime.base_catalog


def test_exact_first_turn_materializes_request_from_public_frame(
        public_catalog: PublicFrameCatalog) -> None:
    """精确公开首句必须只经 RAW-00/RAW-01 形成完整 F-00 request。"""
    frame = public_catalog.frames[0]
    intake = intake_raw_conversation_vector(frame.surface_bytes)

    result = ingress_raw_lexical_frame(intake, public_catalog, _OCCURRENCE_KEY)

    assert result.result_code == DLG_RAW_ACCEPT
    assert result.accepted is True
    assert result.matched_frame_count == 1
    assert result.frame == frame
    assert result.representations == tuple(item.representation for item in frame.routes)
    assert result.language_atoms == tuple(item.atom for item in frame.routes)
    assert result.request is not None
    assert result.request.target == frame.question.target
    assert result.request.trace == (*frame.question.trace_prefix, *_OCCURRENCE_KEY)
    assert result.context_read is None
    assert result.state_delta == ()


def test_raw_rejection_does_not_scan_catalog(
        monkeypatch: pytest.MonkeyPatch,
        public_catalog: PublicFrameCatalog,
        ) -> None:
    """RAW-00 拒绝必须在 lexical/构式索引读取前直接返回。"""
    intake = intake_raw_conversation_vector((0xEF, 0xBB, 0xBF, 0x61))

    def forbidden_scan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("raw rejection 不得扫描 public catalog")

    monkeypatch.setattr(PublicFrameCatalog, "matching_frames", forbidden_scan)
    monkeypatch.setattr(PublicFrameCatalog, "has_construction", forbidden_scan)
    result = ingress_raw_lexical_frame(intake, public_catalog, _OCCURRENCE_KEY)

    assert result.result_code == DLG_RAW_REJECT_BOM
    assert result.matched_frame_count == 0
    assert result.frame is None
    assert result.request is None
    assert result.context_read is None
    assert result.state_delta == ()


def test_decodable_unlearned_surface_is_lexical_miss(
        public_catalog: PublicFrameCatalog) -> None:
    """可解码但不在公开 catalog 的首句不得调用词义或生成 fallback。"""
    intake = intake_raw_conversation_vector(
        tuple("北川站西门何时启用？".encode("utf-8")))

    result = ingress_raw_lexical_frame(intake, public_catalog, _OCCURRENCE_KEY)

    assert result.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert result.matched_frame_count == 0
    assert result.frame is None
    assert result.representations == result.language_atoms == ()
    assert result.request is None
    assert result.state_delta == ()


def test_duplicate_exact_frames_are_ambiguous_not_silently_selected(
        public_catalog: PublicFrameCatalog) -> None:
    """相同 surface 的多个公开 frame 必须拒绝，不能按 tuple 顺序暗选。"""
    frame = public_catalog.frames[0]
    duplicate_catalog = PublicFrameCatalog(
        public_catalog.source_sha256,
        (frame, frame),
    )
    intake = intake_raw_conversation_vector(frame.surface_bytes)

    result = ingress_raw_lexical_frame(intake, duplicate_catalog, _OCCURRENCE_KEY)

    assert result.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert result.matched_frame_count == 2
    assert result.frame is None
    assert result.representations == result.language_atoms == ()
    assert result.request is None
    assert result.state_delta == ()


def test_missing_construction_registry_entry_rejects_after_lexical_match(
        public_catalog: PublicFrameCatalog) -> None:
    """词汇匹配不等于构式已注册，缺构式时不得生成 request。"""
    frame = public_catalog.frames[0]
    construction_miss_catalog = PublicFrameCatalog(
        public_catalog.source_sha256,
        public_catalog.frames,
        (frame.routes[0].representation.stable_key(),),
    )
    intake = intake_raw_conversation_vector(frame.surface_bytes)

    result = ingress_raw_lexical_frame(
        intake,
        construction_miss_catalog,
        _OCCURRENCE_KEY,
    )

    assert result.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS
    assert result.matched_frame_count == 1
    assert result.frame == frame
    assert result.representations
    assert result.language_atoms
    assert result.request is None
    assert result.context_read is None
    assert result.state_delta == ()


def test_none_frame_rejects_supplied_context_read(
        public_catalog: PublicFrameCatalog) -> None:
    """首轮 NONE frame 即使收到有效 context read 也不得把它并入 request。"""
    frame = public_catalog.frames[0]
    intake = intake_raw_conversation_vector(frame.surface_bytes)
    context_read = start_conversation_context((65001, 32, 9)).read(0)

    result = ingress_raw_lexical_frame(
        intake,
        public_catalog,
        _OCCURRENCE_KEY,
        context_read=context_read,
    )

    assert result.result_code == DLG_RAW_REJECT_CONTEXT
    assert result.matched_frame_count == 1
    assert result.frame == frame
    assert result.request is None
    assert result.context_read == context_read
    assert result.state_delta == ()


def test_source_conflict_shape_carries_no_semantic_materialization(
        public_catalog: PublicFrameCatalog) -> None:
    """DLG-RAW-08 code 13 在 ingress 边界不得携带 frame、route、request 或 context。"""
    intake = intake_raw_conversation_vector(public_catalog.frames[0].surface_bytes)
    result = ConversationRawLexicalIngressResult(
        DLG_RAW_REJECT_SOURCE_CONFLICT,
        intake,
        public_catalog,
        matched_frame_count=1,
    )

    assert result.accepted is False
    assert result.frame is None
    assert result.representations == ()
    assert result.language_atoms == ()
    assert result.request is None
    assert result.context_read is None
    assert result.state_delta == ()
    with pytest.raises(ConversationRawLexicalIngressError,
                       match="source conflict"):
        ConversationRawLexicalIngressResult(
            DLG_RAW_REJECT_SOURCE_CONFLICT,
            intake,
            public_catalog,
            matched_frame_count=1,
            frame=public_catalog.frames[0],
        )
