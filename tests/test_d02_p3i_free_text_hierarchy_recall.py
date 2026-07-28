"""D-02G P3-Ia 自由文本层级、中心与改写召回合同 T0。"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    StableRecordKey,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_catalog import (
    SAMPLE_RELATIVE_PATH,
    build_t0_cases,
    build_t0_sample_bytes,
    load_t0_sample,
    validate_case_catalog,
)
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    AbsoluteSpan,
    CandidateEvidence,
    CenterCandidate,
    FreeTextHierarchyRecallCase,
    FreeTextHierarchyRecallContractError,
    OwnerIsolation,
    RecallBudget,
    RecallReceipt,
)


_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / SAMPLE_RELATIVE_PATH


def _case() -> FreeTextHierarchyRecallCase:
    return build_t0_cases()[0]


def _replace_candidate(case, index, **changes):
    values = list(case.hierarchy_candidates)
    values[index] = replace(values[index], **changes)
    return replace(case, hierarchy_candidates=tuple(values))


def test_formal_sample_is_canonical_deterministic_and_round_trips():
    cases = build_t0_cases()
    assert _SAMPLE.read_bytes() == build_t0_sample_bytes()
    assert load_t0_sample(_SAMPLE) == cases
    assert tuple(FreeTextHierarchyRecallCase.from_dict(item.to_dict())
                 for item in cases) == cases
    assert tuple(canonical_json_bytes(item.to_dict()) for item in cases) == tuple(
        canonical_json_bytes(
            FreeTextHierarchyRecallCase.from_dict(item.to_dict()).to_dict())
        for item in cases)


def test_contract_reuses_source_ref_and_keeps_query_source_distinct():
    case = _case()
    assert isinstance(case.document.source_ref, SourceRef)
    assert isinstance(case.query.source_ref, SourceRef)
    assert case.query.source_ref != case.document.source_ref
    assert all(item.span.source_ref == case.document.source_ref
               for item in case.hierarchy_candidates)
    assert case.receipt.citations[0].source_ref == case.document.source_ref


def test_bad_empty_and_out_of_document_ranges_fail_closed():
    case = _case()
    span = case.hierarchy_candidates[0].span
    with pytest.raises(FreeTextHierarchyRecallContractError, match="非空半开"):
        AbsoluteSpan(span.span_key, span.source_ref, span.start, span.start)
    bad = replace(span, end=len(case.document.raw_text) + 1)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="超出"):
        _replace_candidate(case, 0, span=bad)


def test_missing_parent_and_parent_cycle_fail_before_consumption():
    case = _case()
    with pytest.raises(FreeTextHierarchyRecallContractError, match="parent 越界"):
        _replace_candidate(case, 2, parent_key=StableRecordKey((999, 1)))
    values = list(case.hierarchy_candidates)
    values[0] = replace(values[0], parent_key=values[1].candidate_key)
    values[1] = replace(values[1], parent_key=values[0].candidate_key)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="出现环"):
        replace(case, hierarchy_candidates=tuple(values))


def test_duplicate_candidate_span_and_evidence_identities_fail_closed():
    case = _case()
    duplicate_span = replace(
        case.hierarchy_candidates[5].span,
        span_key=case.hierarchy_candidates[4].span.span_key,
    )
    with pytest.raises(FreeTextHierarchyRecallContractError, match="identity 重复"):
        _replace_candidate(case, 5, span=duplicate_span)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="evidence identity"):
        replace(case, evidence=case.evidence + (case.evidence[-1],))


def test_sibling_ordinal_must_be_positive_contiguous_strict_int():
    case = _case()
    with pytest.raises(FreeTextHierarchyRecallContractError, match="从 1 连续"):
        _replace_candidate(case, 1, ordinal=3)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="正严格整数"):
        replace(case.hierarchy_candidates[0], ordinal=True)


def test_cross_source_or_version_span_and_evidence_fail_closed():
    case = _case()
    source = case.document.source_ref
    changed_source = replace(
        source,
        versions=VersionBundle(
            source.versions.corpus,
            ParserVersion(source.versions.parser.value + 1),
            source.versions.primitive,
            source.versions.curriculum,
        ),
    )
    changed_span = replace(case.hierarchy_candidates[0].span,
                           source_ref=changed_source)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="跨 source/version"):
        _replace_candidate(case, 0, span=changed_span)
    changed_evidence = replace(case.evidence[0], source_ref=changed_source)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="Evidence 跨"):
        replace(case, evidence=(changed_evidence, *case.evidence[1:]))


def test_candidate_cannot_read_evaluator_owner_or_label():
    case = _case()
    evidence = case.evidence[0]
    with pytest.raises(FreeTextHierarchyRecallContractError, match="枚举非法"):
        CandidateEvidence(
            evidence.evidence_key,
            evidence.evidence_kind,
            "EVALUATOR",
            evidence.source_ref,
            evidence.span_key,
            evidence.subject_key,
        )
    owner = case.owner_isolation
    with pytest.raises(FreeTextHierarchyRecallContractError, match="偷看 evaluator"):
        OwnerIsolation(
            owner.source_owner_key,
            owner.candidate_owner_key,
            owner.teacher_owner_key,
            owner.evaluator_owner_key,
            owner.source_relative_path,
            owner.candidate_relative_path,
            owner.teacher_relative_path,
            owner.evaluator_relative_path,
            owner.candidate_visible_owner_kinds,
            owner.candidate_visible_evidence_keys,
            1,
        )


def test_owner_keys_and_physical_paths_must_be_pairwise_distinct():
    owner = _case().owner_isolation
    with pytest.raises(FreeTextHierarchyRecallContractError, match="owner 必须物理隔离"):
        replace(owner, evaluator_owner_key=owner.teacher_owner_key)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="owner path"):
        replace(owner, evaluator_relative_path=owner.teacher_relative_path)


def test_activation_never_authorizes_center_adoption():
    center = _case().center_candidates[0]
    with pytest.raises(FreeTextHierarchyRecallContractError, match="activation 不得"):
        CenterCandidate(
            center.center_key,
            center.query_key,
            center.situation_key,
            center.target_kind,
            center.target_key,
            center.state,
            center.evidence_keys,
            1,
            1,
        )


def test_acl_mismatch_and_payload_read_before_denial_fail_closed():
    case = _case()
    bad_obligation = replace(
        case.obligation,
        acl_key=StableRecordKey((777, 1)),
    )
    with pytest.raises(FreeTextHierarchyRecallContractError, match="ACL 不一致"):
        replace(case, obligation=bad_obligation)
    denied_obligation = replace(case.obligation, failure_state="UNAUTHORIZED")
    denied_receipt = replace(
        case.receipt,
        result_keys=(),
        citations=(),
        stop_reason="UNAUTHORIZED",
    )
    with pytest.raises(FreeTextHierarchyRecallContractError, match="payload 读取前"):
        replace(
            case,
            obligation=denied_obligation,
            receipt=denied_receipt,
        )


def test_authorized_denial_has_zero_payload_reads_and_explicit_stop():
    case = _case()
    obligation = replace(case.obligation, failure_state="UNAUTHORIZED")
    receipt = RecallReceipt(
        obligation.obligation_key,
        (case.receipt.index_keys[0],),
        (),
        1,
        0,
        0,
        (),
        (),
        "UNAUTHORIZED",
        0,
    )
    denied = replace(
        case,
        obligation=obligation,
        receipt=receipt,
        evaluator_label=replace(case.evaluator_label, citation_keys=()),
    )
    assert denied.receipt.segment_payload_gets == 0
    assert denied.receipt.segment_payload_bytes == 0


def test_budget_omission_and_budget_overrun_fail_closed():
    case = _case()
    value = case.to_dict()
    del value["obligation"]["budget"]["max_results"]
    with pytest.raises(FreeTextHierarchyRecallContractError, match="字段集合"):
        FreeTextHierarchyRecallCase.from_dict(value)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="正严格整数"):
        RecallBudget(True, 1, 1, 1)
    overrun = replace(
        case.receipt,
        segment_payload_bytes=case.obligation.budget.max_segment_payload_bytes + 1,
    )
    with pytest.raises(FreeTextHierarchyRecallContractError, match="超出 recall budget"):
        replace(case, receipt=overrun)


def test_resolved_receipt_requires_exact_citation_and_matching_source():
    case = _case()
    with pytest.raises(FreeTextHierarchyRecallContractError, match="缺 result/citation"):
        replace(case, receipt=replace(case.receipt, citations=()))
    other_source = replace(case.document.source_ref, document_id=99)
    with pytest.raises(FreeTextHierarchyRecallContractError, match="citation span 跨"):
        replace(case.receipt.citations[0], source_ref=other_source)


def test_cluster_may_repeat_within_split_but_never_across_splits():
    train, held_out = build_t0_cases()
    leaked = replace(
        held_out,
        split_identity=replace(
            held_out.split_identity,
            source_cluster_key=train.split_identity.source_cluster_key,
        ),
    )
    with pytest.raises(FreeTextHierarchyRecallContractError, match="cluster 跨 split"):
        validate_case_catalog((train, leaked))


def test_noncanonical_unknown_and_missing_fields_fail_closed(tmp_path):
    case = _case()
    value = case.to_dict()
    value["unexpected"] = 1
    with pytest.raises(FreeTextHierarchyRecallContractError, match="字段集合"):
        FreeTextHierarchyRecallCase.from_dict(value)
    value = case.to_dict()
    del value["receipt"]
    with pytest.raises(FreeTextHierarchyRecallContractError, match="字段集合"):
        FreeTextHierarchyRecallCase.from_dict(value)
    noncanonical = tmp_path / "pretty.jsonl"
    noncanonical.write_text(
        json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不是规范形式"):
        load_t0_sample(noncanonical)
