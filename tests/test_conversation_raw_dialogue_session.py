"""DLG-RAW-04 通过完整公开 logical runtime 的显式 context 回归。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    conversation_raw_answer_runtime,
    conversation_raw_dialogue_session,
    conversation_raw_response_act_runtime,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR,
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_course_prepare import (
    PublicCoursePreparationCache,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    run_public_frame_dialogue_turn,
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SourceBoundSlotCompositionResolution,
    resolve_source_bound_slot_composition,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dialogue_rejects_negative_conversation_key() -> None:
    """会话 identity 必须可进入无符号整数 snapshot/framing。"""
    with pytest.raises(ValueError, match="非负"):
        start_public_frame_dialogue((-1,))


@pytest.fixture(scope="module")
def public_runtime() -> PublicDialogueRuntimeV1:
    """host 只在此处构造 closure；所有 session transition 均只收 runtime。"""
    closure = load_public_source_payload_closure_from_root(_REPOSITORY_ROOT)
    return build_public_dialogue_runtime_v1(closure)


def _frame(catalog: PublicFrameCatalog, requirement: int):
    """按冻结 context requirement 选择唯一 frame，不依赖 manifest 行序。"""
    matches = tuple(item for item in catalog.frames
                    if item.context_requirement == requirement)
    assert len(matches) == 1
    return matches[0]


def _frame_by_key(catalog: PublicFrameCatalog, frame_key: str):
    """按冻结 frame key 取值，避免目录、文件或宿主排序参与语义。"""
    matches = tuple(item for item in catalog.frames if item.frame_key == frame_key)
    assert len(matches) == 1
    return matches[0]


def _source_bound_surface(runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
    """从 runtime 已验证的 V2 family/binding 拼出可回答 alias 首句。"""
    return _source_bound_surface_for_binding(
        runtime,
        "north-east-side-entrance-time-v2",
        "北川站东侧入口何时启用？",
    )


def _source_bound_surface_for_binding(
        runtime: PublicDialogueRuntimeV1,
        binding_key: str,
        expected_text: str,
        ) -> tuple[int, ...]:
    """只由 runtime 内的 V2 槽片段重组预期 surface，避免测试走静态问句。"""
    slot_catalog = runtime.source_bound_slot_catalog
    binding = next(item for item in slot_catalog.bindings
                   if item.binding_key == binding_key)
    expected = tuple(ord(character) for character in expected_text)
    surfaces = tuple(
        (*family.prefix_scalars, *binding.entity_scalars,
         *family.suffix_scalars)
        for family in slot_catalog.families
        if (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars) == expected
    )
    assert surfaces == (expected,)
    return expected


def test_follow_up_requires_actual_first_run_context_and_then_appends(
        public_runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """冷启动代词拒绝；真实 target anchor 才能解锁第二轮。"""
    catalog = public_runtime.base_catalog
    first_frame = _frame(catalog, PUBLIC_FRAME_CONTEXT_NONE)
    follow_up_frame = _frame(catalog, PUBLIC_FRAME_CONTEXT_TARGET_ANCHOR)
    initial = start_public_frame_dialogue((65001, 38, 1))
    preparation_cache = PublicCoursePreparationCache()

    cold = run_public_frame_dialogue_turn(
        initial, follow_up_frame.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert cold.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert cold.context_written == 0
    assert cold.after.context == initial.context
    assert cold.after.next_operation_ordinal == 2

    failed_first = run_public_frame_dialogue_turn(
        initial, (0x78,), public_runtime, preparation_cache=preparation_cache)
    assert failed_first.answer.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert failed_first.context_written == 0
    after_failed_follow_up = run_public_frame_dialogue_turn(
        failed_first.after, follow_up_frame.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert after_failed_follow_up.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert after_failed_follow_up.context_written == 0
    assert after_failed_follow_up.after.context.revision == 0

    first = run_public_frame_dialogue_turn(
        initial, first_frame.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert first.answer.result_code == DLG_RAW_ACCEPT
    assert first.context_written == 1
    assert first.context_read is None
    assert first.after.context.revision == 1

    follow_up = run_public_frame_dialogue_turn(
        first.after, follow_up_frame.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert follow_up.answer.result_code == DLG_RAW_ACCEPT
    assert follow_up.context_written == 1
    assert follow_up.context_read is not None
    assert follow_up.context_read.turns[-1].target_key == follow_up_frame.context_target_key
    assert follow_up.after.context.revision == 2
    assert follow_up.after.context.turns[-1].context_read == follow_up.context_read
    assert bytes(follow_up.answer.output_bytes).decode("utf-8") == "北川站东门于2024年启用。"
    assert all(type(item) is int for item in follow_up.canonical_record())
    assert preparation_cache.entry_count == 1
    assert preparation_cache.miss_count == 1
    assert preparation_cache.hit_count == 1


def test_contextual_ellipsis_requires_actual_matching_target_and_then_appends(
        public_runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """省略主语只消费上一轮实际同 target 的 typed context。"""
    catalog = public_runtime.active_catalog
    wrong_anchor = _frame_by_key(catalog, "dlg-raw-public-v2-clarify-site")
    matching_anchor = _frame_by_key(
        catalog, "dlg-raw-public-v3-derived-unknown-budget-omission")
    contextual = _frame_by_key(
        catalog, "dlg-raw-public-v4-contextual-unknown-budget-ellipsis")
    initial = start_public_frame_dialogue((65001, 38, 2))
    preparation_cache = PublicCoursePreparationCache()

    cold = run_public_frame_dialogue_turn(
        initial, contextual.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert cold.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert cold.answer.output_bytes == ()
    assert cold.context_written == 0
    assert cold.after.context == initial.context

    wrong_first = run_public_frame_dialogue_turn(
        initial, wrong_anchor.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert wrong_first.answer.result_code == DLG_RAW_ACCEPT
    assert wrong_first.context_written == 1
    assert wrong_first.after.context.turns[-1].target_key != contextual.context_target_key
    wrong_follow_up = run_public_frame_dialogue_turn(
        wrong_first.after, contextual.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert wrong_follow_up.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert wrong_follow_up.answer.output_bytes == ()
    assert wrong_follow_up.context_written == 0
    assert wrong_follow_up.after.context == wrong_first.after.context

    first = run_public_frame_dialogue_turn(
        initial, matching_anchor.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert first.answer.result_code == DLG_RAW_ACCEPT
    assert first.context_written == 1
    assert first.after.context.revision == 1
    assert first.after.context.turns[-1].target_key == contextual.context_target_key
    follow_up = run_public_frame_dialogue_turn(
        first.after, contextual.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert follow_up.answer.result_code == DLG_RAW_ACCEPT
    assert follow_up.context_written == 1
    assert follow_up.context_read is not None
    assert follow_up.context_read.turns[-1].target_key == contextual.context_target_key
    assert follow_up.after.context.revision == 2
    assert bytes(follow_up.answer.output_bytes).decode("utf-8") == "现有来源没有提供所问信息。"

    cleared = start_public_frame_dialogue((65001, 38, 3))
    after_clear = run_public_frame_dialogue_turn(
        cleared, contextual.surface_bytes, public_runtime,
        preparation_cache=preparation_cache)
    assert after_clear.answer.result_code == DLG_RAW_REJECT_CONTEXT
    assert after_clear.context_written == 0


def test_source_bound_slot_composition_uses_dynamic_frame_only_after_static_miss(
        public_runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """RAW-06 仅在 static exact miss 后组成动态 NONE frame。"""
    dynamic_surface = _source_bound_surface(public_runtime)
    assert not public_runtime.active_catalog.matching_frames(dynamic_surface)
    initial = start_public_frame_dialogue((65001, 38, 6))

    dynamic = run_public_frame_dialogue_turn(
        initial, encode_utf8_v1(dynamic_surface), public_runtime)
    assert dynamic.answer.result_code == DLG_RAW_ACCEPT
    assert dynamic.context_written == 1
    assert bytes(dynamic.answer.output_bytes).decode("utf-8") == "北川站东门于2024年启用。"

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        """static exact 路径触达 resolver 即说明优先级退化。"""
        raise AssertionError("static exact 不得调用 RAW-06 resolver")

    monkeypatch.setattr(
        conversation_raw_dialogue_session,
        "resolve_source_bound_slot_composition",
        fail_if_called,
    )
    static = run_public_frame_dialogue_turn(
        initial,
        _frame(public_runtime.base_catalog,
               PUBLIC_FRAME_CONTEXT_NONE).surface_bytes,
        public_runtime,
    )
    assert static.answer.result_code == DLG_RAW_ACCEPT


def test_v2_star_bridge_alias_runs_existing_clarify_response_act(
        public_runtime: PublicDialogueRuntimeV1,
        ) -> None:
    """星桥项目 alias 必须动态接入既有 CLARIFY recipe，而非建立新回答捷径。"""
    surface = _source_bound_surface_for_binding(
        public_runtime,
        "star-bridge-project-site-v2",
        "星桥项目的试验地点在哪里？",
    )
    clarify = _frame_by_key(
        public_runtime.active_catalog,
        "dlg-raw-public-v2-clarify-site",
    )
    assert not public_runtime.active_catalog.matching_frames(surface)
    initial = start_public_frame_dialogue((65001, 38, 8))

    turn = run_public_frame_dialogue_turn(
        initial,
        encode_utf8_v1(surface),
        public_runtime,
        preparation_cache=PublicCoursePreparationCache(),
    )

    assert turn.answer.result_code == DLG_RAW_ACCEPT
    assert turn.answer.accepted
    assert turn.answer.ingress.frame is not None
    assert turn.answer.ingress.catalog.frames == (turn.answer.ingress.frame,)
    assert turn.answer.ingress.frame.frame_key != clarify.frame_key
    assert turn.answer.ingress.frame.question.target == clarify.question.target
    assert turn.answer.ingress.frame.recipe == clarify.recipe
    assert turn.answer.run is not None
    assert turn.context_written == 1
    assert turn.after.context.revision == 1
    assert bytes(turn.answer.output_bytes).decode("utf-8") == (
        "存在多个候选，请补充更明确的限定。")


def test_v2_source_counterevidence_stops_before_materialization_or_generation(
        public_runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """code 13 不得创建 request、G-01/G-03/G-04、backend、输出或 context append。"""
    surface = _source_bound_surface_for_binding(
        public_runtime,
        "north-east-side-passage-conflict-v2",
        "北川站东侧通道何时启用？",
    )
    assert not public_runtime.active_catalog.matching_frames(surface)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("source conflict 不得进入回答 materialization 或 generation")

    for module, attribute in (
            (conversation_raw_answer_runtime,
             "materialize_public_frame_candidate"),
            (conversation_raw_answer_runtime, "_generation_protocols"),
            (conversation_raw_answer_runtime, "_build_lifecycle"),
            (conversation_raw_answer_runtime, "GroundedAnswerRunLocalFactory"),
            (conversation_raw_answer_runtime, "DictBackend"),
            (conversation_raw_answer_runtime, "encode_utf8_v1"),
            (conversation_raw_response_act_runtime, "_materialize_planning"),
            (conversation_raw_response_act_runtime, "_generation_protocols"),
            (conversation_raw_response_act_runtime, "_build_lifecycle"),
            (conversation_raw_response_act_runtime,
             "GroundedResponseActRunLocalFactory"),
            (conversation_raw_response_act_runtime, "DictBackend"),
            (conversation_raw_response_act_runtime, "encode_utf8_v1"),
    ):
        monkeypatch.setattr(module, attribute, forbidden)
    monkeypatch.setattr(
        conversation_raw_dialogue_session.ConversationContextState,
        "append",
        forbidden,
    )
    monkeypatch.setattr(
        conversation_raw_dialogue_session.ConversationContextState,
        "append_consumed",
        forbidden,
    )
    initial = start_public_frame_dialogue((65001, 38, 9))

    turn = run_public_frame_dialogue_turn(
        initial,
        encode_utf8_v1(surface),
        public_runtime,
    )

    assert turn.answer.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT
    assert turn.answer.accepted is False
    assert turn.answer.ingress.frame is None
    assert turn.answer.ingress.request is None
    assert turn.answer.ingress.representations == ()
    assert turn.answer.ingress.language_atoms == ()
    assert turn.answer.run is None
    assert turn.answer.output_scalars == ()
    assert turn.answer.output_bytes == ()
    assert turn.answer.output_readback is None
    assert turn.context_read is None
    assert turn.context_written == 0
    assert turn.after.context == initial.context


def test_source_bound_slot_rejections_preserve_codes_and_zero_context_writes(
        public_runtime: PublicDialogueRuntimeV1,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """RAW-06 的 7/8/9 保持零输出、零 context append。"""
    slot_catalog = public_runtime.source_bound_slot_catalog
    surface = _source_bound_surface(public_runtime)
    accepted = resolve_source_bound_slot_composition(
        slot_catalog,
        public_runtime.base_catalog,
        public_runtime.active_catalog,
        surface,
        public_runtime.source_payload_closure,
    )
    assert accepted.accepted
    assert accepted.frame is not None
    assert accepted.public_frame_catalog is not None
    initial = start_public_frame_dialogue((65001, 38, 7))

    ambiguous = resolve_source_bound_slot_composition(
        slot_catalog,
        public_runtime.base_catalog,
        public_runtime.active_catalog,
        tuple(ord(item) for item in "东岸入口何时启用？"),
        public_runtime.source_payload_closure,
    )
    conflict = resolve_source_bound_slot_composition(
        slot_catalog,
        public_runtime.base_catalog,
        public_runtime.active_catalog,
        tuple(ord(item) for item in "北川站东侧通道何时启用？"),
        public_runtime.source_payload_closure,
    )
    assert ambiguous.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
    assert conflict.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT
    cases = (
        SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_LEXICAL_MISS, 0, surface, slot_catalog),
        ambiguous,
        SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, 1, surface, slot_catalog),
        conflict,
    )
    for ordinal, resolution in enumerate(cases):
        monkeypatch.setattr(
            conversation_raw_dialogue_session,
            "resolve_source_bound_slot_composition",
            lambda *_args, _resolution=resolution, **_kwargs: _resolution,
        )
        turn = run_public_frame_dialogue_turn(
            initial, encode_utf8_v1(resolution.input_scalars), public_runtime)
        assert turn.answer.result_code == resolution.result_code, ordinal
        assert turn.answer.output_bytes == (), ordinal
        assert turn.context_written == 0, ordinal
        assert turn.after.context == initial.context, ordinal
