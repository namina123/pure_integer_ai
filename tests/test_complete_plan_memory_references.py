"""长会话的全部 Plan 引用及 O/H/E 必须在原显式预算内核验，后页损坏不得漏审。"""
from copy import deepcopy
from dataclasses import replace

import pytest

from pure_integer_ai.experiments.query_memory_ohe_audit import (
    QueryMemoryOheEvidenceError, verify_query_memory_ohe,
)
from tests.test_query_memory_ohe_audit import _fixture


def _combined_sources(count):
    """合并独立来源的真实分型 O/H/E 测试记录，重建全部原整数引用。"""
    captured = None
    declarations, manifests = {}, {}
    for document_id in range(1, count + 1):
        current, rows, manifest, _evidence = _fixture(document_id=document_id)
        declarations.update(rows)
        manifests[manifest.source] = manifest
        if captured is None:
            captured = current
            continue
        target, origin = captured["trace"], current["trace"]
        table = target["integer_key_table"]

        def remap(ref):
            """只重建 compact 表索引；完整成员值和来源均不改动。"""
            key = origin["integer_key_table"][ref - 1]
            if key not in table:
                table.append(key)
            return table.index(key) + 1

        for field in ("roots", "bindings", "evidence"):
            target["state_key"][field].extend(remap(ref) for ref in origin["state_key"][field])
        for hop in origin["hops"]:
            target["hops"].append({**hop, "proposition_ref": remap(hop["proposition_ref"]),
                                   "predicate_ref": remap(hop["predicate_ref"])})
        for field in ("evidence_refs", "memory_refs"):
            target["response_plan"][field].extend(origin["response_plan"][field])
    return captured, declarations, manifests


def _verify_combined(captured, declarations, manifests):
    """保持既有 256 声明预算；完整读取所有来源，未执行任何 query/generation。"""
    return verify_query_memory_ohe(captured,
        declaration_reader=lambda ref: (declarations[ref],) if ref in declarations else (),
        current_manifest_reader=manifests.__getitem__,
        source_hash_reader=lambda source: 19 + (source.document_id - 1) * 1000,
        max_state_members=1024, max_ohe_declarations=256)


def test_complete_plan_over_one_page_checks_all_original_ohe_without_changing_budget():
    """130 个原 Plan 引用覆盖 65 个完整来源，全部 195 个声明均真实核验。"""
    captured, declarations, manifests = _combined_sources(65)
    original = deepcopy(captured)
    result = _verify_combined(captured, declarations, manifests)
    assert len(captured["trace"]["response_plan"]["memory_refs"]) == 130
    assert result["inspected_ohe_declaration_count"] == 195
    assert result["independent_evidence_count"] == 65
    assert result["max_ohe_declarations"] == 256
    assert captured == original


def test_total_declaration_budget_still_rejects_even_when_plan_reference_count_fits():
    """258 个声明超出原显式 256 总量，不能仅凭 172 个 Plan 引用而放行。"""
    captured, declarations, manifests = _combined_sources(86)
    with pytest.raises(QueryMemoryOheEvidenceError, match="explicit declaration audit budget"):
        _verify_combined(captured, declarations, manifests)


def test_missing_declaration_on_second_page_is_still_rejected():
    """移除第 65 来源的物理 H，第二页同样必须失败关闭。"""
    captured, declarations, manifests = _combined_sources(65)
    missing = next(ref for ref, row in declarations.items() if row.event_hash == 64102)
    del declarations[missing]
    with pytest.raises(QueryMemoryOheEvidenceError, match="unique declaration"):
        _verify_combined(captured, declarations, manifests)


@pytest.mark.parametrize("mutation,match", (
    ("duplicate", "duplicated"),
    ("unqualified", "current QueryState"),
    ("owner", "owner/version/space"),
))
def test_extra_memory_references_cannot_enter_through_larger_valid_roster(mutation, match):
    """总量合法仍须逐项满足唯一、当前归属及 namespace，不信任 Plan 自报引用。"""
    captured, declarations, manifests = _combined_sources(65)
    refs = captured["trace"]["response_plan"]["memory_refs"]
    if mutation == "duplicate":
        refs.append(refs[-1])
    else:
        foreign, _rows, _manifest, _entry = _fixture(document_id=1000)
        key = foreign["trace"]["response_plan"]["memory_refs"][0]
        if mutation == "owner":
            from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
            ref = MemoryObjectRef.from_stable_key(tuple(key))
            key = list(replace(ref, owner=replace(ref.owner, user_id=99)).stable_key())
        refs.append(key)
    with pytest.raises(QueryMemoryOheEvidenceError, match=match):
        _verify_combined(captured, declarations, manifests)
