"""DLG-RAW-10 provider 的整数 binding 与 proof 回投专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
    PublicProofSentenceProviderError,
    load_public_proof_sentence_provider_from_root,
    run_public_proof_sentence_provider_vector,
    verify_public_proof_sentence_provider_result,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
    PublicSentenceDemoResult,
    PublicSentenceDemoSameDispatchProofProjection,
    run_public_sentence_demo_vector,
)
from pure_integer_ai.experiments import conversation_public_proof_sentence_provider
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_RUNTIME,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    parse_canonical_json_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]
_VECTOR = Path(__file__).parent / "fixtures" / (
    "dlg_raw_public_proof_sentence_provider_v1_conformance.json")


@pytest.fixture(scope="module")
def provider():
    """从已发布 public snapshot 构造一次 provider，禁止临时 rebuild。"""
    return load_public_proof_sentence_provider_from_root(_ROOT)


def _raw(route) -> tuple[int, ...]:
    """测试 host 边缘把既有 route 表层一次复制成 u8 tuple。"""
    return tuple(route.request.question_surface.encode("utf-8"))


def test_provider_projects_every_published_route_from_the_same_proof(provider) -> None:
    """24 条 exact/alias/implicit 路由都只能复制同次 W03-W05 proof output。"""
    observed = []
    for route in provider.legacy_catalog.routes:
        result = run_public_proof_sentence_provider_vector(provider, _raw(route))
        direct = run_public_sentence_demo_vector(
            provider.legacy_runtime,
            provider.legacy_catalog,
            _raw(route),
        )
        assert result.provider_status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER
        assert result.mapped_dlg_result_code == DLG_RAW_ACCEPT
        assert result.accepted
        assert result.route_kind == route.route_kind
        assert result.source_record_key == route.source_record_key
        assert result.output_scalars == direct.generated_proposition_scalars
        assert result.output_bytes == direct.output_bytes
        assert direct.canonical_record() == result.demo_record
        observed.append(result.canonical_record())

    assert len(observed) == 24
    assert {route.route_kind for route in provider.legacy_catalog.routes} == {1, 2, 3}


def test_provider_miss_has_no_output_and_is_distinct_from_runtime_unknown(provider) -> None:
    """未学习输入保留 DLG lexical miss，不能被 provider 组织为自然语言。"""
    result = run_public_proof_sentence_provider_vector(
        provider,
        tuple("未学习的公开问题？".encode("utf-8")),
    )

    assert result.provider_status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert result.output_scalars == result.output_bytes == ()
    assert result.source_record_key == ()


def test_provider_binding_rejects_catalog_or_runtime_identity_drift(provider) -> None:
    """legacy adapter 的任何 catalog/runtime identity 漂移必须在 provider 构造时失败。"""
    with pytest.raises(PublicProofSentenceProviderError):
        replace(provider, runtime_identity=(0,) * 32)
    with pytest.raises(PublicProofSentenceProviderError):
        replace(provider, catalog_record=provider.catalog_record[:-1])


def test_provider_replay_rejects_forged_nested_demo_output(provider) -> None:
    """RAW-04 只能消费同一 provider 重放出的完整 carrier，不能信任形状相同的 output。"""
    route = provider.legacy_catalog.routes[0]
    raw = _raw(route)
    actual = run_public_proof_sentence_provider_vector(provider, raw)
    forged = replace(
        actual,
        output_scalars=(65,),
        output_bytes=(65,),
    )

    assert verify_public_proof_sentence_provider_result(
        provider,
        raw,
        actual,
    )
    assert not verify_public_proof_sentence_provider_result(
        provider,
        raw,
        forged,
    )


def test_provider_runtime_unknown_is_not_downgraded_to_lexical_miss(
        provider,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """已匹配 route 的 legacy UNKNOWN 是 runtime 拒绝，不能让主链继续 fallback。"""
    route = provider.legacy_catalog.routes[0]
    raw = _raw(route)
    direct = run_public_sentence_demo_vector(
        provider.legacy_runtime,
        provider.legacy_catalog,
        raw,
    )
    unknown = PublicSentenceDemoResult(
        PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
        direct.intake,
        matched_route_count=1,
        selected_route_kind=direct.selected_route_kind,
        selected_source_record_key=direct.selected_source_record_key,
        dispatch_status_code=PUBLIC_SENTENCE_DEMO_DISPATCH_UNKNOWN,
    )
    monkeypatch.setattr(
        conversation_public_proof_sentence_provider,
        "run_public_sentence_demo_vector_with_typed_proof",
        lambda *_args, **_kwargs: PublicSentenceDemoSameDispatchProofProjection(
            unknown,
            None,
        ),
    )

    result = run_public_proof_sentence_provider_vector(provider, raw)

    assert result.provider_status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT
    assert result.mapped_dlg_result_code == DLG_RAW_REJECT_RUNTIME
    assert not result.accepted


def test_provider_matches_the_frozen_language_neutral_conformance_vector(provider) -> None:
    """固定 input/binding 必须得到同一整数结果摘要，供其他语言复现。"""
    payload = _VECTOR.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    raw = tuple(value["input_u8"])
    result = run_public_proof_sentence_provider_vector(provider, raw)

    assert value["schema"] == 1
    assert tuple(value["provider_identity_u8"]) == provider.provider_identity
    assert tuple(value["runtime_identity_u8"]) == provider.runtime_identity
    assert result.provider_status == value["provider_status"]
    assert result.mapped_dlg_result_code == value["mapped_dlg_result_code"]
    assert result.route_kind == value["route_kind"]
    assert result.source_record_key == tuple(value["source_record_key"])
    assert result.context_policy == value["context_policy"]
    assert result.output_bytes == tuple(value["output_u8"])
    assert tuple(portable_sha256_v1(
        b"PURE-INTEGER-AI/DLG-RAW-10/DEMO-RECORD/V1",
        (result.demo_record,),
    )) == tuple(value["demo_record_sha256_u8"])
    assert tuple(portable_sha256_v1(
        b"PURE-INTEGER-AI/DLG-RAW-10/RESULT-RECORD/V1",
        (result.canonical_record(),),
    )) == tuple(value["result_record_sha256_u8"])
