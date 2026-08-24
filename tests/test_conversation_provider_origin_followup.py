"""DLG-RAW-11C 来源内追问 reducer 与公开课程的定向验证。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
    provider_origin_provider_binding_from_public_provider_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    start_mixed_conversation_context_v2,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupCatalogV1,
    ProviderOriginFollowupError,
    compare_nonnegative_integer_records_v1,
    order_items_by_nonnegative_integer_record_v1,
    run_provider_origin_followup_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PublicProofSentenceProviderV1,
    load_public_proof_sentence_provider_from_root,
    run_public_proof_sentence_provider_vector_with_typed_proof,
)
from pure_integer_ai.experiments.conversation_public_provider_origin_followup_catalog import (
    load_public_provider_origin_followup_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
    intake_raw_conversation_vector,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provider() -> PublicProofSentenceProviderV1:
    """加载已冻结的公开 proof provider，不重建课程或读取 private 数据。"""
    return load_public_proof_sentence_provider_from_root(_ROOT)


@pytest.fixture(scope="module")
def catalog(provider: PublicProofSentenceProviderV1) -> ProviderOriginFollowupCatalogV1:
    """用公开 closure 和同次 proof 编译一次来源内 follow-up catalog。"""
    return load_public_provider_origin_followup_catalog_from_closure(
        load_public_source_payload_closure_from_root(_ROOT),
        provider,
    )


def _anchor_context(provider: PublicProofSentenceProviderV1, question: str):
    """以真实 provider 同次 proof 写入一条 V2 provider-origin tail。"""
    same_dispatch = run_public_proof_sentence_provider_vector_with_typed_proof(
        provider,
        tuple(question.encode("utf-8")),
    )
    assert same_dispatch.demo_proof_projection is not None
    sparse = same_dispatch.demo_proof_projection.sparse_proof_projection
    assert sparse is not None
    anchor = project_provider_origin_anchor_v1(
        provider_origin_provider_binding_from_public_provider_v1(provider),
        same_dispatch.provider_result,
        provider_origin_legacy_proof_from_same_dispatch_v1(sparse),
    )
    assert anchor.accepted
    state = start_mixed_conversation_context_v2((65111, 1, 1))
    return state.admit_provider_origin_projection(anchor, state.read(0)).after


def _followup_intake():
    """形成已来源化构式的 canonical raw input record。"""
    return intake_raw_conversation_vector(tuple("它的原因是什么？".encode("utf-8")))


def test_integer_record_order_is_explicit_and_prefix_stable() -> None:
    """catalog order 使用注册的整数词典序，不能回退到宿主 tuple/hash 规则。"""
    assert compare_nonnegative_integer_records_v1(
        (1,), (1, 0), label="test") == -1
    assert compare_nonnegative_integer_records_v1(
        (1, 256), (1, 255), label="test") == 1
    ordered = order_items_by_nonnegative_integer_record_v1(
        ((2,), (1, 256), (1,), (1, 255)),
        key=lambda item: item,
        label="test order",
    )
    assert ordered == ((1,), (1, 255), (1, 256), (2,))
    with pytest.raises(ProviderOriginFollowupError, match="重复"):
        order_items_by_nonnegative_integer_record_v1(
            ((1,), (1,)), key=lambda item: item, label="test duplicate")


def test_real_effect_focus_reuses_the_bound_cause_occurrence(
        provider: PublicProofSentenceProviderV1,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> None:
    """结果焦点的真实 provider tail 只能以 profile 指定 span 复用原因 bytes。"""
    context = _anchor_context(provider, "寒潮导致什么？")
    result = run_provider_origin_followup_v1(
        _followup_intake(),
        context.read(1),
        catalog,
    )

    assert result.accepted
    assert result.context_write_origin == 0
    assert result.candidate is not None
    assert result.candidate.answer_start == 0
    assert bytes(result.output_u8).decode("utf-8") == "寒潮"
    assert result.output_u8 == result.candidate.output_u8


def test_non_effect_focus_cannot_match_by_same_sentence_or_source(
        provider: PublicProofSentenceProviderV1,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> None:
    """相同完整输出里的原因焦点不允许因字符串或 SourceRef 相同被反向匹配。"""
    context = _anchor_context(provider, "什么导致路面结冰？")
    result = run_provider_origin_followup_v1(
        _followup_intake(),
        context.read(1),
        catalog,
    )

    assert not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_CONTEXT
    assert result.output_u8 == ()
    assert result.candidate is None


def test_two_distinct_structural_profiles_are_reference_ambiguity(
        provider: PublicProofSentenceProviderV1,
        catalog: ProviderOriginFollowupCatalogV1,
        ) -> None:
    """两个完整闭合 candidate 必须返回 14，不能按配置/字符串次序任选。"""
    profile = next(
        item for item in catalog.profiles
        if bytes(item.profile_key_u8) == b"provider-origin-causal-effect-cold-alias-v1")
    duplicate_meaning = replace(
        profile,
        profile_key_u8=tuple(
            b"provider-origin-causal-effect-cold-alias-shadow-v1"),
        profile_identity_u8=(),
    )
    ambiguous_catalog = ProviderOriginFollowupCatalogV1(
        catalog.source_payload_closure_identity_u8,
        catalog.forms,
        tuple(sorted(
            (*catalog.profiles, duplicate_meaning),
            key=lambda item: item.canonical_record(),
        )),
    )
    context = _anchor_context(provider, "寒潮导致什么？")
    result = run_provider_origin_followup_v1(
        _followup_intake(),
        context.read(1),
        ambiguous_catalog,
    )

    assert not result.accepted
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_REFERENCE_AMBIGUOUS
    assert result.candidate_count == 2
    assert result.output_scalars == result.output_u8 == ()
