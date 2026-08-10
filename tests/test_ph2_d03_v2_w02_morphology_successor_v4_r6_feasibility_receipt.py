"""W02 V4-first R6 feasibility receipt 的公开专项测试。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_r6_feasibility_receipt import (
    W02_MORPH_V4_R6_DIMENSION_COUNTS,
    W02_MORPH_V4_R6_FEASIBILITY_OWNER_ID,
    W02_MORPH_V4_R6_SPLIT_COUNTS,
    read_w02_morphology_successor_v4_r6_feasibility_receipt,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_r6_feasibility_receipt_is_payload_free_and_frozen() -> None:
    """receipt 只能包含安全聚合，并保持 formal 未授权。"""
    receipt = read_w02_morphology_successor_v4_r6_feasibility_receipt(REPOSITORY)
    assert receipt["owner_id"] == W02_MORPH_V4_R6_FEASIBILITY_OWNER_ID
    assert receipt["case_count"] == 500
    assert receipt["dimension_counts"] == W02_MORPH_V4_R6_DIMENSION_COUNTS
    assert receipt["split_counts"] == W02_MORPH_V4_R6_SPLIT_COUNTS
    assert receipt["formal_owner_transport_authorized"] == 0
    assert receipt["formal_private_evaluation_runs"] == 0
    assert receipt["main_session_case_payload_reads"] == 0
    assert receipt["main_session_conllu_content_reads"] == 0
    serialized = str(receipt)
    assert "D:\\" not in serialized
    assert "eligible-token-span-cases.jsonl.gz" not in serialized


def test_r6_feasibility_receipt_preserves_revision_lineage() -> None:
    """Revision A 阻塞与 Revision B 非正式 payload 边界必须可审计。"""
    receipt = read_w02_morphology_successor_v4_r6_feasibility_receipt(REPOSITORY)
    assert receipt["revision_a_blocker"] == "CONLLU_MORPHOLOGY_MISSING"
    assert receipt["revision_a_payload_reuse_authorized"] == 0
    assert receipt["revision_b_private_case_reuse_as_formal_payload_authorized"] == 0
    assert receipt["source_repository_key"] == "UD_CLASSICAL_CHINESE_TUECL"
