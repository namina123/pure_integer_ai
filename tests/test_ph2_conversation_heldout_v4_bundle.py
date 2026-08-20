"""DLG-05 v4 source bundle 的窄合同专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4BundleError,
    ConversationHeldOutV4Candidate,
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4ExecutionInput,
    ConversationHeldOutV4Representation,
    ConversationHeldOutV4SourceBundle,
    ConversationHeldOutV4SourceRecord,
    ConversationHeldOutV4Turn,
    build_v4_source_bundle,
    build_v4_source_bundle_from_executions,
    build_v4_turn_from_candidates,
    build_v4_turn_from_execution,
    digest_from_hex,
    export_v4_candidate_set,
    unicode_scalars,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from tests.test_f00_question_answer_runtime import _fixture, _rendered_text


_FAMILY = ProtocolKey((20260821, 405))
_CASE = ProtocolKey((20260821, 405, 1))
_TURN = ProtocolKey((20260821, 405, 1, 1))
_ZERO_SHA = digest_from_hex("00" * 32)
_DEPENDENCIES = ConversationHeldOutV4DependencyBinding(
    _ZERO_SHA, _ZERO_SHA, _ZERO_SHA)


def _bundle_fixture():
    """从真实 F-00 candidate 和 renderer 构造最小无标签 bundle。"""
    fixture = _fixture(EVIDENCE_SUPPORT, answer_text="事实")
    run = fixture.runtime.run(fixture.request)
    assert run.complete
    assert run.generation is not None and run.generation.rendered is not None
    rendered = run.generation.rendered
    candidate = ConversationHeldOutV4Candidate.from_candidate(
        run.query_result.candidates[0], tuple(rendered.units),
        rendered.representations)
    representations = tuple(
        ConversationHeldOutV4Representation(item, index, tuple(unit for unit in
            fixture.renderer.render((item,)).units))
        for index, item in enumerate(rendered.representations)
    )
    source = ConversationHeldOutV4SourceRecord(
        fixture.request.source,
        unicode_scalars("公开来源正文"),
        tuple(hashlib.sha256("公开来源正文".encode("utf-8")).digest()),
        unicode_scalars("CC-BY"),
        unicode_scalars("项目公开来源"),
    )
    turn = ConversationHeldOutV4Turn(
        _CASE, _TURN, 1, fixture.request, representations,
        representations,
        (candidate,), (source.source,), _DEPENDENCIES)
    bundle = build_v4_source_bundle(
        version=1, family_key=_FAMILY, dependencies=_DEPENDENCIES,
        turns=(turn,), sources=(source,))
    return fixture, bundle, turn, source


def test_v4_bundle_keeps_full_payload_and_is_deterministic():
    """完整对象、来源原文和整数身份都必须可复核且重复构造相等。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        assert turn.payload_size == len(turn.canonical_payload)
        assert len(turn.payload_sha256) == 32
        assert bundle.payload_size == len(bundle.canonical_payload)
        assert bundle.canonical_payload == bundle.canonical_payload
        assert bundle.turn_for(_CASE, _TURN) is turn
        assert bundle.source_for(source.source) is source
        assert source.raw_text == "公开来源正文"
        assert not hasattr(turn, "selected_candidate")
        assert not hasattr(turn, "response_act")
        assert not hasattr(bundle, "labels")
    finally:
        fixture.close()


def test_v4_bundle_rejects_source_hash_or_representation_drift():
    """来源正文和 Representation content 不能仅靠摘要继续施工。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        with pytest.raises(ConversationHeldOutV4BundleError, match="内容 hash"):
            replace(source, content_sha256=_ZERO_SHA)
        representation = turn.representations[0]
        with pytest.raises(ConversationHeldOutV4BundleError, match="scalar"):
            replace(representation, scalars=(0x4E00,))
        with pytest.raises(ConversationHeldOutV4BundleError, match="surface"):
            replace(turn.candidates[0], surface_scalars=(0x4E00,))
        with pytest.raises(ConversationHeldOutV4BundleError, match="SourceRef"):
            ConversationHeldOutV4SourceBundle(
                bundle.version, bundle.family_key, bundle.dependencies,
                (replace(turn, source_keys=()),), bundle.sources)
    finally:
        fixture.close()


def test_v4_candidate_export_is_forward_only_and_keeps_full_candidate_set():
    """候选导出只消费真实 candidate 和已验证 surface，不接受摘要键。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        exported = export_v4_candidate_set(
            (turn.candidates[0].candidate,),
            lambda candidate: fixture.renderer.render(
                turn.candidates[0].surface_representations),
        )
        assert exported == turn.candidates
        assert exported[0].candidate_key == turn.candidates[0].candidate_key
        with pytest.raises(ConversationHeldOutV4BundleError, match="RenderedSurface"):
            export_v4_candidate_set(
                (turn.candidates[0].candidate,), lambda _candidate: object())
    finally:
        fixture.close()


def test_v4_turn_builder_requires_source_records_at_construction_point():
    """turn builder 在 source bundle 组装前就拒绝缺失 SourceRecord。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        with pytest.raises(ConversationHeldOutV4BundleError, match="SourceRecord"):
            build_v4_turn_from_candidates(
                case_key=turn.case_key,
                turn_key=turn.turn_key,
                ordinal=turn.ordinal,
                request=turn.request,
                representations=turn.representations,
                surface_representations=turn.surface_representations,
                candidates=(turn.candidates[0].candidate,),
                render_candidate=lambda _candidate: fixture.renderer.render(
                    turn.candidates[0].surface_representations),
                source_records=(),
                dependencies=bundle.dependencies,
            )
        rebuilt = build_v4_turn_from_candidates(
            case_key=turn.case_key,
            turn_key=turn.turn_key,
            ordinal=turn.ordinal,
            request=turn.request,
            representations=turn.representations,
            surface_representations=turn.surface_representations,
            candidates=(turn.candidates[0].candidate,),
            render_candidate=lambda _candidate: fixture.renderer.render(
                turn.candidates[0].surface_representations),
            source_records=(source,),
            dependencies=bundle.dependencies,
        )
        assert rebuilt == turn
    finally:
        fixture.close()


def test_v4_execution_adapter_uses_same_result_candidate_tuple():
    """execution adapter 不允许调用方用另一个候选 tuple 替换真实结果。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        query = fixture.runtime.run(fixture.request).query_result
        rebuilt = build_v4_turn_from_execution(
            case_key=turn.case_key,
            turn_key=turn.turn_key,
            ordinal=turn.ordinal,
            request=fixture.request,
            execution=query,
            representations=turn.representations,
            surface_representations=turn.surface_representations,
            render_candidate=lambda _candidate: fixture.renderer.render(
                turn.candidates[0].surface_representations),
            source_records=(source,),
            dependencies=bundle.dependencies,
        )
        assert rebuilt.candidates == turn.candidates
        with pytest.raises(ConversationHeldOutV4BundleError, match="request"):
            build_v4_turn_from_execution(
                case_key=turn.case_key,
                turn_key=turn.turn_key,
                ordinal=turn.ordinal,
                request=replace(fixture.request, trace=(1, 2, 3)),
                execution=query,
                representations=turn.representations,
                surface_representations=turn.surface_representations,
                render_candidate=lambda _candidate: fixture.renderer.render(
                    turn.candidates[0].surface_representations),
                source_records=(source,),
                dependencies=bundle.dependencies,
            )
    finally:
        fixture.close()


def test_v4_turn_allows_a_verified_empty_candidate_set():
    """无候选 UNKNOWN turn 可以没有答案 surface，但仍保留输入映射。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        empty = replace(
            turn,
            surface_representations=(),
            candidates=(),
        )
        assert empty.candidates == ()
        assert empty.surface_representations == ()
    finally:
        fixture.close()


def test_v4_execution_inputs_aggregate_without_reusing_old_observations():
    """多 execution 聚合保留每个完整 request/candidate/source，而非摘要。"""
    fixture, bundle, turn, source = _bundle_fixture()
    try:
        execution = fixture.runtime.run(fixture.request).query_result
        item = ConversationHeldOutV4ExecutionInput(
            turn.case_key,
            turn.turn_key,
            turn.ordinal,
            fixture.request,
            execution,
            turn.representations,
            turn.surface_representations,
            (source,),
            bundle.dependencies,
        )
        rebuilt = build_v4_source_bundle_from_executions(
            version=1,
            family_key=bundle.family_key,
            inputs=(item,),
            render_candidate=lambda _candidate: fixture.renderer.render(
                turn.candidates[0].surface_representations),
        )
        assert rebuilt.turns[0].candidates == turn.candidates
        assert rebuilt.sources == bundle.sources
    finally:
        fixture.close()
