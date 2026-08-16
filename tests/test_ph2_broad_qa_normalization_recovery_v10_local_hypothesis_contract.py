"""覆盖 recovery-v10 局部 hypothesis 与整句闭合合同。"""
from __future__ import annotations

import hashlib

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_contract import (
    LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY,
    SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION,
    build_normalization_recovery_v10_local_span_hypothesis,
    derive_normalization_recovery_v10_local_hypothesis_preflight,
    execute_normalization_recovery_v10_local_hypothesis_contract,
)


def _sha(label: str) -> str:
    """返回稳定测试 SHA。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _query(input_text: str, source: str) -> dict[str, object]:
    """构造结构自洽 query。"""
    return {
        "input_text": input_text,
        "official_source_text": source,
        "structure_tokens": list(localization_structure_tokens(input_text)),
    }


def _span(
        *, start: int, end: int, input_text: str, output_text: str,
        source: str = "File", authorized: bool = False,
        identity_veto_count: int = 0, conflict_count: int = 0,
        ) -> dict[str, object]:
    """构造 hypothesis-only 或三来源 source-context 授权 span。"""
    kind = (
        SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION
        if authorized else LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY)
    return build_normalization_recovery_v10_local_span_hypothesis(
        input_start=start,
        input_end=end,
        input_text=input_text,
        output_text=output_text,
        authorization_kind=kind,
        evidence_ids=[_sha(f"evidence:{start}:{end}:{input_text}")],
        opencc_route_ids=[_sha(f"route:{input_text}:{output_text}")],
        training_support_families=(
            list(V8_TRAIN_FAMILIES)
            if authorized else list(V8_TRAIN_FAMILIES[:2])),
        source_context_authorization_id=(
            _sha(f"source-context:{source}:{start}:{end}")
            if authorized else ""),
        authorized_official_source_text=source if authorized else "",
        identity_veto_count=identity_veto_count,
        conflict_count=conflict_count,
    )


def test_hypothesis_only_span_never_becomes_answer_output() -> None:
    """OpenCC与两来源证据只能形成trace，不能提交局部或整句输出。"""
    result = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(start=0, end=1, input_text="檔", output_text="档"),
        ),
    )
    assert result["behavior"] == "UNKNOWN"
    assert result["output_text"] == ""
    assert result["reason"] == "LOCAL_HYPOTHESIS_NOT_SENTENCE_AUTHORIZATION"
    assert result["hypothesis_only_span_count"] == 1
    assert result["partial_commit_count"] == 0


def test_all_source_context_spans_may_commit_whole_sentence() -> None:
    """全部changed span获三来源exact-source授权后才可提交合成句。"""
    result = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="檔", output_text="档",
                authorized=True),
        ),
    )
    assert result["behavior"] == "EXACT"
    assert result["output_text"] == "档案"
    assert result["all_changed_spans_authorized"] == 1
    assert result["reason"] == "ALL_CHANGED_SPANS_SOURCE_CONTEXT_AUTHORIZED"


def test_mixed_authorization_identity_conflict_and_source_scope_fail_closed(
        ) -> None:
    """任一span未授权、被否决、冲突或source不符都必须整句UNKNOWN。"""
    query = _query("檔案儲存", "File")
    authorized = _span(
        start=0, end=1, input_text="檔", output_text="档", authorized=True)
    hypothesis = _span(
        start=2, end=3, input_text="儲", output_text="储")
    mixed = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=query, span_hypotheses=(authorized, hypothesis))
    assert mixed["behavior"] == "UNKNOWN"
    assert mixed["output_text"] == ""
    assert mixed["reason"] == "LOCAL_HYPOTHESIS_NOT_SENTENCE_AUTHORIZATION"

    vetoed = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="檔", output_text="档",
                authorized=True, identity_veto_count=1),
        ))
    assert vetoed["reason"] == "IDENTITY_VETO_REJECTED"
    assert vetoed["output_text"] == ""

    conflicted = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="檔", output_text="档",
                authorized=True, conflict_count=1),
        ))
    assert conflicted["reason"] == "SPAN_CONFLICT_REJECTED"
    assert conflicted["output_text"] == ""

    wrong_source = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "Document"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="檔", output_text="档",
                source="File", authorized=True),
        ))
    assert wrong_source["reason"] == "SOURCE_CONTEXT_AUTHORIZATION_MISMATCH"
    assert wrong_source["output_text"] == ""


def test_geometry_source_slice_and_structure_fail_closed() -> None:
    """重叠、原文slice漂移或结构token破坏均不得提交。"""
    overlap = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="檔", output_text="档",
                authorized=True),
            _span(
                start=0, end=2, input_text="檔案", output_text="文件",
                authorized=True),
        ))
    assert overlap["reason"] == "SPAN_GEOMETRY_REJECTED"
    assert overlap["output_text"] == ""

    source_slice = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("檔案", "File"),
        span_hypotheses=(
            _span(
                start=0, end=1, input_text="錯", output_text="对",
                authorized=True),
        ))
    assert source_slice["reason"] == "SOURCE_SLICE_MISMATCH_REJECTED"
    assert source_slice["output_text"] == ""

    structure = execute_normalization_recovery_v10_local_hypothesis_contract(
        query=_query("開啟 %s", "Open %s"),
        span_hypotheses=(
            _span(
                start=3, end=5, input_text="%s", output_text="檔案",
                source="Open %s", authorized=True),
        ))
    assert structure["reason"] == "STRUCTURE_PRESERVATION_HARD_GATE_REJECTED"
    assert structure["structure_mismatch_count"] == 1
    assert structure["output_text"] == ""


def test_preflight_separates_trace_and_answer_and_rejects_weak_authorization(
        ) -> None:
    """预检覆盖EXACT/UNKNOWN，且两来源不得伪装source-context授权。"""
    hypothesis = _span(
        start=0, end=1, input_text="檔", output_text="档")
    authorized = _span(
        start=0, end=1, input_text="檔", output_text="档", authorized=True)
    preflight = derive_normalization_recovery_v10_local_hypothesis_preflight((
        {
            "expected_behavior": "UNKNOWN",
            "expected_reason": "LOCAL_HYPOTHESIS_NOT_SENTENCE_AUTHORIZATION",
            "query": _query("檔案", "File"),
            "span_hypotheses": (hypothesis,),
        },
        {
            "expected_behavior": "EXACT",
            "expected_reason": "ALL_CHANGED_SPANS_SOURCE_CONTEXT_AUTHORIZED",
            "query": _query("檔案", "File"),
            "span_hypotheses": (authorized,),
        },
    ))
    assert preflight["status"] == "PASS"
    assert preflight["failure_count"] == 0
    assert preflight["exact_count"] == 1
    assert preflight["unknown_count"] == 1

    with pytest.raises(BroadQaExternalDataError, match="family 未闭合"):
        build_normalization_recovery_v10_local_span_hypothesis(
            input_start=0,
            input_end=1,
            input_text="檔",
            output_text="档",
            authorization_kind=SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION,
            evidence_ids=[_sha("weak-evidence")],
            opencc_route_ids=[_sha("weak-route")],
            training_support_families=list(V8_TRAIN_FAMILIES[:2]),
            source_context_authorization_id=_sha("weak-source-context"),
            authorized_official_source_text="File",
        )
