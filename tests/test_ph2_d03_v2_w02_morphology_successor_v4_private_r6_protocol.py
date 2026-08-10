"""W02 V4-first R6 公共协议与一次性 guard 的专项测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_private_r6_protocol import (
    W02_MORPH_V4_PRIVATE_R6_LAYOUTS,
    W02_MORPH_V4_PRIVATE_R6_OWNER_METADATA_VERSION,
    W02_MORPH_V4_PRIVATE_R6_PATHS,
    W02MorphologySuccessorV4PrivateR6ProtocolError,
    consume_w02_morphology_successor_v4_private_r6_guard,
    publish_w02_morphology_successor_v4_private_r6_family_root,
    read_w02_morphology_successor_v4_private_r6_owner_metadata,
    read_w02_morphology_successor_v4_private_r6_protocol_freeze,
    verify_w02_morphology_successor_v4_private_r6_consumed_guard,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_r6_feasibility_receipt import (
    W02_MORPH_V4_R6_DIMENSION_COUNTS,
    W02_MORPH_V4_R6_SPLIT_COUNTS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _digest(value: str) -> str:
    """为合成 metadata 产生稳定 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_inventory() -> list[dict[str, object]]:
    """构造只含身份、不含 payload 的标准七文件 inventory。"""
    rows = []
    for ordinal, layout in enumerate(W02_MORPH_V4_PRIVATE_R6_LAYOUTS, start=1):
        if layout == "PRIVATE_SOURCE":
            kind, split, count = "source_ref", "", 500
        else:
            kind = "observation" if layout.endswith("OBSERVATION") else "evaluator_label"
            split = layout.removeprefix("PRIVATE_").removesuffix(
                "_OBSERVATION").removesuffix("_LABEL").lower()
            count = W02_MORPH_V4_R6_SPLIT_COUNTS[split]
        rows.append({
            "content_sha256": _digest(f"content:{layout}"),
            "content_size_bytes": count,
            "first_record_key": [91, ordinal, 1],
            "last_record_key": [91, ordinal, count],
            "layout_key": layout,
            "license_ids": ["CC-BY-SA-4.0"],
            "record_count": count,
            "record_kind": kind,
            "relative_path": W02_MORPH_V4_PRIVATE_R6_PATHS[layout],
            "split": split,
            "transport_sha256": _digest(f"transport:{layout}"),
            "transport_size_bytes": count,
        })
    return rows


def _owner_metadata(path: Path) -> Path:
    """发布一个合成 safe owner metadata，供 family/guard 测试。"""
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_V4_R6_FORMAL_OWNER_METADATA",
        "artifact_version": W02_MORPH_V4_PRIVATE_R6_OWNER_METADATA_VERSION,
        "candidate_calls": 0,
        "case_selection_commitment": (
            "812658e2b8fa1eaf925d4dfd2a101ee80dda2e6c1854fc0c4238cf200504cf3a"),
        "commitments": {
            "case_commitment": _digest("case"),
            "cluster_commitment": _digest("cluster"),
            "label_commitment": _digest("label"),
            "payload_commitment": _digest("payload"),
        },
        "dimension_counts": W02_MORPH_V4_R6_DIMENSION_COUNTS,
        "double_pass_equal": 1,
        "feasibility_case_transport_reads": 1,
        "feasibility_metadata_sha256": (
            "5c027695c9c46e3763a1a91dc6ff126b69730c670650459b0f8a0e4c4f4c37e8"),
        "file_inventory": _file_inventory(),
        "formal_private_evaluation_runs": 0,
        "label_imputation_count": 0,
        "main_session_private_payload_reads": 0,
        "old_owner_payload_reads": 0,
        "owner_family_key": "SYNTHETIC-R6-OWNER",
        "owner_id": "synthetic-r6-owner",
        "pair_count": 500,
        "source_count": 500,
        "source_key": "UD_LZH_TUECL_R2_18_TOKEN_SPAN_BLIND_PRIVATE",
        "split_counts": W02_MORPH_V4_R6_SPLIT_COUNTS,
        "status": "R6_FORMAL_OWNER_FROZEN",
        "teacher_calls": 0,
        "v1_v2_v3_v4_calls": 0,
    }, path)
    return path


def test_r6_public_protocol_is_zero_run_and_v4_bound() -> None:
    """公共 freeze 必须绑定 V4、R5 FAIL 且保持 formal 未运行。"""
    freeze = read_w02_morphology_successor_v4_private_r6_protocol_freeze(REPOSITORY)
    assert freeze["status"] == "W02_V4_R6_PUBLIC_PROTOCOL_FROZEN_FORMAL_OWNER_PENDING"
    assert freeze["formal_private_evaluation_runs"] == 0
    assert freeze["private_payload_reads"] == 0
    assert freeze["dimension_denominator_counts"] == W02_MORPH_V4_R6_DIMENSION_COUNTS
    assert freeze["artifact_chain"]["v4_artifact_semantic_sha256"] == (
        "55a64c12007aaa5b8fc625c0b477d1dd539bb6eabb9bacf90408c624bbc7f332")


def test_r6_family_guard_is_one_shot(tmp_path: Path) -> None:
    """全新 family 只能从 AVAILABLE 转为 CONSUMED 一次。"""
    metadata = _owner_metadata(tmp_path / "owner-public-metadata.json")
    owner = read_w02_morphology_successor_v4_private_r6_owner_metadata(metadata)
    assert len(owner.files) == 7
    family = publish_w02_morphology_successor_v4_private_r6_family_root(
        REPOSITORY, tmp_path / "formal-family", metadata)
    consume_w02_morphology_successor_v4_private_r6_guard(
        REPOSITORY, family, metadata)
    intent = verify_w02_morphology_successor_v4_private_r6_consumed_guard(family)
    assert intent["run_id"] == 1
    with pytest.raises(W02MorphologySuccessorV4PrivateR6ProtocolError):
        consume_w02_morphology_successor_v4_private_r6_guard(
            REPOSITORY, family, metadata)
