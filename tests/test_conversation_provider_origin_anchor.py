"""DLG-RAW-11A provider-origin anchor 的闭锁专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
    PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
    PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
    PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
    ProviderOriginProviderBindingV1,
    project_provider_origin_anchor_v1,
    provider_origin_legacy_proof_from_same_dispatch_v1,
    provider_origin_provider_binding_from_public_provider_v1,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
    PublicProofSentenceProviderResultV1,
    load_public_proof_sentence_provider_from_root,
    run_public_proof_sentence_provider_vector_with_typed_proof,
    verify_public_proof_sentence_provider_result,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    build_public_sentence_demo_catalog,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    run_sparse_qa_query_with_typed_proof,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_public_sparse_qa_runtime_snapshot,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def anchor_inputs():
    """从冻结 snapshot 取得真实 same-dispatch proof，再构造已验证 result carrier。"""
    runtime = load_public_sparse_qa_runtime_snapshot(
        _ROOT / "data" / "ph2" / "sparse_qa_runtime_snapshot_v1.json",
        repository=_ROOT,
    )
    catalog = build_public_sentence_demo_catalog(runtime)
    route = next(item for item in catalog.routes
                 if item.route_kind == PUBLIC_SENTENCE_DEMO_ROUTE_EXACT)
    raw = tuple(route.request.question_surface.encode("utf-8"))
    proof_projection = run_sparse_qa_query_with_typed_proof(
        runtime,
        route.request,
    )
    carrier = provider_origin_legacy_proof_from_same_dispatch_v1(
        proof_projection)
    assert carrier is not None
    provider_identity = (1,) * 32
    runtime_identity = (2,) * 32
    catalog_record = (3, 5, 8)
    binding = ProviderOriginProviderBindingV1(
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        provider_identity,
        runtime_identity,
        catalog_record,
        tuple(portable_sha256_v1(
            PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
            (catalog_record,),
        )),
    )
    provider_result = PublicProofSentenceProviderResultV1(
        PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
        DLG_RAW_ACCEPT,
        intake_raw_conversation_vector(raw),
        provider_identity,
        runtime_identity,
        catalog_record,
        demo_record=(1,),
        route_kind=route.route_kind,
        source_record_key=carrier.source_record_key,
        output_scalars=carrier.generated_proposition_scalars,
        output_bytes=carrier.generated_proposition_u8,
        context_policy=PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    )
    return (
        binding,
        provider_result,
        carrier,
    )


def _assert_none(anchor) -> None:
    """所有失败路径都必须收敛到无 payload 的 canonical NONE。"""
    assert anchor.anchor_status == PROVIDER_ORIGIN_ANCHOR_STATUS_NONE
    assert not anchor.accepted
    assert anchor.provider_identity_u8 == ()
    assert anchor.source_record_key == ()
    assert anchor.output_u8 == ()


def test_valid_provider_result_and_typed_proof_emit_full_anchor(anchor_inputs) -> None:
    """有效链路保留所有可供后续 context admission 审计的结构 identity。"""
    binding, result, carrier = anchor_inputs
    anchor = project_provider_origin_anchor_v1(binding, result, carrier)

    assert anchor.anchor_status == PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER
    assert anchor.accepted
    assert anchor.source_record_key == carrier.source_record_key
    assert anchor.source_ref_stable_key == carrier.source_ref_stable_key
    assert anchor.proposition_key == carrier.proposition_key
    assert anchor.predicate_key == carrier.predicate_key
    assert anchor.focus_role_binding_key == carrier.focus_role_binding_key
    assert anchor.focus_occurrence_key == carrier.focus_occurrence_key
    assert anchor.output_scalars == result.output_scalars
    assert anchor.output_u8 == result.output_bytes
    assert len(anchor.provider_result_identity_u8) == 32
    assert len(anchor.input_intake_identity_u8) == 32
    assert len(anchor.output_readback_identity_u8) == 32
    assert len(anchor.anchor_identity_u8) == 32
    assert all(type(item) is int for item in anchor.canonical_record())


def test_output_source_key_and_relation_drift_fail_closed(anchor_inputs) -> None:
    """不可信 carrier/result 的任何承重字段漂移不得产生可写入锚点。"""
    binding, result, carrier = anchor_inputs
    shortened_scalars = result.output_scalars[:-1]
    output_drift = replace(
        result,
        output_scalars=shortened_scalars,
        output_bytes=encode_utf8_v1(shortened_scalars),
    )
    source_drift = replace(result, source_record_key=(999_991,))
    key_drift = replace(carrier, candidate_predicate_key=(999_992,))
    relation_drift = replace(carrier, relation_kind_code=999_993)

    _assert_none(project_provider_origin_anchor_v1(
        binding, output_drift, carrier))
    _assert_none(project_provider_origin_anchor_v1(
        binding, source_drift, carrier))
    _assert_none(project_provider_origin_anchor_v1(
        binding, result, key_drift))
    _assert_none(project_provider_origin_anchor_v1(
        binding, result, relation_drift))


def test_actual_provider_same_dispatch_projects_anchor_without_text_reverse_lookup() -> None:
    """真实 provider 的 result 与 typed proof 必须来自同一 dispatch，锚点不重跑或匹配 output 文本。"""
    provider = load_public_proof_sentence_provider_from_root(_ROOT)
    route = next(item for item in provider.legacy_catalog.routes
                 if item.route_kind == PUBLIC_SENTENCE_DEMO_ROUTE_EXACT)
    raw = tuple(route.request.question_surface.encode("utf-8"))

    same_dispatch = run_public_proof_sentence_provider_vector_with_typed_proof(
        provider,
        raw,
    )
    assert same_dispatch.provider_result.accepted
    assert same_dispatch.demo_proof_projection is not None
    sparse_projection = same_dispatch.demo_proof_projection.sparse_proof_projection
    assert sparse_projection is not None
    carrier = provider_origin_legacy_proof_from_same_dispatch_v1(
        sparse_projection)
    assert carrier is not None
    assert verify_public_proof_sentence_provider_result(
        provider,
        raw,
        same_dispatch.provider_result,
    )

    anchor = project_provider_origin_anchor_v1(
        provider_origin_provider_binding_from_public_provider_v1(provider),
        same_dispatch.provider_result,
        carrier,
    )

    assert anchor.accepted
    assert anchor.output_u8 == same_dispatch.provider_result.output_bytes
    assert anchor.source_record_key == same_dispatch.provider_result.source_record_key
