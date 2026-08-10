"""发布不含 private 内容的 TueCL R6 feasibility owner receipt。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v7 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V7_PATH,
    TUECL_FEASIBILITY_CASE_CONTENT_SHA256,
    TUECL_FEASIBILITY_CASE_TRANSPORT_SHA256,
    TUECL_FEASIBILITY_METADATA_SHA256,
    TUECL_FEASIBILITY_SELECTION_COMMITMENT,
    TUECL_SOURCE_KEY,
    build_blind_private_source_extension_v7_manifest,
    read_blind_private_source_extension_v7_manifest,
)


W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-V4-R6-FEASIBILITY-OWNER-RECEIPT-V1"
)
W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_r6_"
    "feasibility_owner_receipt_v1.json"
)
W02_MORPH_V4_R6_FEASIBILITY_METADATA_SIZE_BYTES = 3_777
W02_MORPH_V4_R6_FEASIBILITY_OWNER_ID = (
    "d30af7d75f6442b3be432af748f5aa19"
)
W02_MORPH_V4_R6_FEASIBILITY_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-V4-TUECL-TOKEN-SPAN-FEASIBILITY-"
    "d30af7d75f6442b3be432af748f5aa19"
)
W02_MORPH_V4_R6_DIMENSION_COUNTS = {
    "W-02-V2-BOUNDARY-WITHDRAWAL": 100,
    "W-02-V2-GENERATION-HARD-CONJUNCT": 100,
    "W-02-V2-MULTI-CANDIDATE": 100,
    "W-02-V2-NEW-CONTENT-MORPHOLOGY": 100,
    "W-02-V2-OOV": 100,
}
W02_MORPH_V4_R6_SPLIT_COUNTS = {
    "adversarial": 100,
    "held_out": 350,
    "wall": 50,
}


# object-model: exception
class W02MorphologySuccessorV4R6FeasibilityReceiptError(RuntimeError):
    """safe metadata、公开 receipt 或 V7 依赖发生漂移。"""


def _repository_file(root: Path, relative: str) -> Path:
    """解析仓内普通文件，并拒绝链接与路径逃逸。"""
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility 公开路径非法")
    return target


def _sha256_file(path: Path) -> tuple[int, str]:
    """流式计算文件长度与 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _require_dict(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    """要求 object 字段集合精确一致。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            f"{where} 字段漂移")
    return value


def validate_w02_morphology_successor_v4_r6_feasibility_metadata(
        raw: object) -> dict[str, Any]:
    """只验证 Revision-B 安全投影，不接触 eligibility gzip。"""
    value = _require_dict(raw, {
        "artifact_kind", "candidate_calls", "candidate_evaluation_runs",
        "case_file_identity", "case_selection_commitment",
        "dimension_counts", "double_pass_equal", "evaluator_calls",
        "formal_private_evaluation_runs", "llm_calls",
        "main_session_conllu_content_reads", "near_duplicate_algorithm",
        "near_duplicate_threshold_basis_points", "owner_family_key",
        "owner_id", "previous_blocked_owner",
        "previous_owner_implementation_reads", "previous_owner_payload_reads",
        "private_payload_reads_by_owner", "public_base", "public_comparison",
        "r6_design_authorized", "r6_formal_run_authorized",
        "rejection_counts", "revision_policy", "selected_case_count",
        "selected_sentence_cluster_count", "source",
        "source_level_kyoto_intersection_count", "split_counts", "status",
        "target_case_count", "target_marked_case_duplicate_unit",
        "teacher_calls", "test_payload_transport_reads", "v1_v2_v3_v4_calls",
    }, where="R6 feasibility metadata")
    case_file = _require_dict(value["case_file_identity"], {
        "content_sha256", "content_size_bytes", "record_count",
        "relative_path", "transport_sha256", "transport_size_bytes",
    }, where="R6 feasibility case identity")
    source = _require_dict(value["source"], {
        "commit_sha1", "file_identities", "language", "license_id",
        "repository_url", "sentence_count", "source_key", "token_count",
    }, where="R6 feasibility source")
    previous = _require_dict(value["previous_blocked_owner"], {
        "blocker_code", "owner_id", "payload_reuse_authorized", "status",
    }, where="R6 revision-A lineage")
    if (value["artifact_kind"]
            != "PH2_D03_V2_W02_V4_TUECL_FEASIBILITY_OWNER_METADATA"
            or value["status"] != "PASS_500_TOKEN_SPAN_CASES_FEASIBLE"
            or value["owner_id"] != W02_MORPH_V4_R6_FEASIBILITY_OWNER_ID
            or value["owner_family_key"]
            != W02_MORPH_V4_R6_FEASIBILITY_OWNER_FAMILY_KEY
            or value["case_selection_commitment"]
            != TUECL_FEASIBILITY_SELECTION_COMMITMENT
            or value["dimension_counts"] != W02_MORPH_V4_R6_DIMENSION_COUNTS
            or value["split_counts"] != W02_MORPH_V4_R6_SPLIT_COUNTS
            or value["selected_case_count"] != 500
            or value["target_case_count"] != 500
            or value["selected_sentence_cluster_count"] != 84
            or value["double_pass_equal"] != 1
            or value["r6_design_authorized"] != 1
            or value["r6_formal_run_authorized"] != 0
            or value["main_session_conllu_content_reads"] != 0
            or any(value[name] != 0 for name in (
                "candidate_calls", "candidate_evaluation_runs",
                "evaluator_calls", "formal_private_evaluation_runs",
                "llm_calls", "teacher_calls", "v1_v2_v3_v4_calls"))):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility 主状态漂移")
    if (case_file["relative_path"]
            != "private/eligible-token-span-cases.jsonl.gz"
            or case_file["record_count"] != 500
            or case_file["content_size_bytes"] != 200_208
            or case_file["content_sha256"]
            != TUECL_FEASIBILITY_CASE_CONTENT_SHA256
            or case_file["transport_size_bytes"] != 32_669
            or case_file["transport_sha256"]
            != TUECL_FEASIBILITY_CASE_TRANSPORT_SHA256):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility case 身份漂移")
    if (source["source_key"]
            != "UD_LZH_TUECL_R2_18_TOKEN_SPAN_BLIND_PRIVATE_CANDIDATE"
            or source["language"] != "lzh"
            or source["license_id"] != "CC-BY-SA-4.0"
            or source["sentence_count"] != 100
            or source["token_count"] != 648
            or value["source_level_kyoto_intersection_count"] != 0
            or previous["status"] != "BLOCKED"
            or previous["blocker_code"] != "CONLLU_MORPHOLOGY_MISSING"
            or previous["payload_reuse_authorized"] != 0
            or value["previous_owner_payload_reads"] != 0):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility 来源或 revision lineage 漂移")
    return value


def build_w02_morphology_successor_v4_r6_feasibility_receipt(
        raw: object) -> dict[str, object]:
    """将已验证 safe metadata 缩成公开、无路径的冻结 receipt。"""
    value = validate_w02_morphology_successor_v4_r6_feasibility_metadata(raw)
    case_file = value["case_file_identity"]
    source = value["source"]
    rejection = value["rejection_counts"]
    public = value["public_comparison"]
    assert isinstance(case_file, dict)
    assert isinstance(source, dict)
    assert isinstance(rejection, dict)
    assert isinstance(public, dict)
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_V4_R6_FEASIBILITY_OWNER_RECEIPT"),
        "artifact_version": W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_VERSION,
        "candidate_evaluation_runs": 0,
        "case_content_sha256": case_file["content_sha256"],
        "case_content_size_bytes": case_file["content_size_bytes"],
        "case_count": case_file["record_count"],
        "case_selection_commitment": value["case_selection_commitment"],
        "case_transport_sha256": case_file["transport_sha256"],
        "case_transport_size_bytes": case_file["transport_size_bytes"],
        "dimension_counts": dict(value["dimension_counts"]),
        "double_pass_equal": 1,
        "feasibility_metadata_sha256": TUECL_FEASIBILITY_METADATA_SHA256,
        "feasibility_metadata_size_bytes":
            W02_MORPH_V4_R6_FEASIBILITY_METADATA_SIZE_BYTES,
        "formal_owner_transport_authorized": 0,
        "formal_private_evaluation_runs": 0,
        "label_imputation_count": 0,
        "main_session_case_payload_reads": 0,
        "main_session_conllu_content_reads": 0,
        "near_duplicate_algorithm": value["near_duplicate_algorithm"],
        "near_duplicate_threshold_basis_points":
            value["near_duplicate_threshold_basis_points"],
        "owner_family_key": value["owner_family_key"],
        "owner_id": value["owner_id"],
        "public_comparison_token_count": public["token_count"],
        "rejection_counts": dict(rejection),
        "revision_a_blocker": "CONLLU_MORPHOLOGY_MISSING",
        "revision_a_payload_reuse_authorized": 0,
        "revision_b_private_case_reuse_as_formal_payload_authorized": 0,
        "selected_sentence_cluster_count":
            value["selected_sentence_cluster_count"],
        "source_commit_sha1": source["commit_sha1"],
        "source_key": TUECL_SOURCE_KEY,
        "source_language": source["language"],
        "source_license_id": source["license_id"],
        "source_repository_key": "UD_CLASSICAL_CHINESE_TUECL",
        "source_sentence_count": source["sentence_count"],
        "source_token_count": source["token_count"],
        "split_counts": dict(value["split_counts"]),
        "status": "R6_FEASIBILITY_OWNER_RECEIPT_FROZEN_PAYLOAD_UNREAD",
        "teacher_calls": 0,
        "v7_source_manifest_sha256": hashlib.sha256(canonical_json_bytes(
            build_blind_private_source_extension_v7_manifest())).hexdigest(),
    }


def publish_w02_morphology_successor_v4_r6_feasibility_receipt(
        repository_root: str | Path,
        safe_metadata_path: str | Path,
        path: str | Path | None = None,
        ) -> Path:
    """只读 safe metadata，核对固定字节身份后发布 receipt。"""
    repository = Path(repository_root).resolve()
    metadata_path = Path(safe_metadata_path).resolve()
    size, digest = _sha256_file(metadata_path)
    if (size != W02_MORPH_V4_R6_FEASIBILITY_METADATA_SIZE_BYTES
            or digest != TUECL_FEASIBILITY_METADATA_SHA256):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility metadata 字节身份漂移")
    raw = read_canonical_object(metadata_path)
    target = (
        repository / Path(*PurePosixPath(
            W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH).parts)
        if path is None else Path(path).resolve())
    write_immutable_json(
        build_w02_morphology_successor_v4_r6_feasibility_receipt(raw), target)
    read_w02_morphology_successor_v4_r6_feasibility_receipt(repository, target)
    return target


def read_w02_morphology_successor_v4_r6_feasibility_receipt(
        repository_root: str | Path,
        path: str | Path | None = None,
        ) -> dict[str, object]:
    """回读公开 receipt，并重新闭合 V7 来源依赖。"""
    repository = Path(repository_root).resolve()
    target = (
        _repository_file(repository, W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH)
        if path is None else Path(path).resolve())
    receipt = read_canonical_object(target)
    expected_fields = {
        "artifact_kind", "artifact_version", "candidate_evaluation_runs",
        "case_content_sha256", "case_content_size_bytes", "case_count",
        "case_selection_commitment", "case_transport_sha256",
        "case_transport_size_bytes", "dimension_counts", "double_pass_equal",
        "feasibility_metadata_sha256", "feasibility_metadata_size_bytes",
        "formal_owner_transport_authorized", "formal_private_evaluation_runs",
        "label_imputation_count", "main_session_case_payload_reads",
        "main_session_conllu_content_reads", "near_duplicate_algorithm",
        "near_duplicate_threshold_basis_points", "owner_family_key",
        "owner_id", "public_comparison_token_count", "rejection_counts",
        "revision_a_blocker", "revision_a_payload_reuse_authorized",
        "revision_b_private_case_reuse_as_formal_payload_authorized",
        "selected_sentence_cluster_count", "source_commit_sha1", "source_key",
        "source_language", "source_license_id", "source_repository_key",
        "source_sentence_count", "source_token_count", "split_counts", "status",
        "teacher_calls", "v7_source_manifest_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility receipt 字段漂移")
    source = read_blind_private_source_extension_v7_manifest(repository)
    source_sha = hashlib.sha256(canonical_json_bytes(source)).hexdigest()
    if (receipt.get("artifact_version")
            != W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_VERSION
            or receipt.get("status")
            != "R6_FEASIBILITY_OWNER_RECEIPT_FROZEN_PAYLOAD_UNREAD"
            or receipt.get("case_count") != 500
            or receipt.get("dimension_counts")
            != W02_MORPH_V4_R6_DIMENSION_COUNTS
            or receipt.get("split_counts") != W02_MORPH_V4_R6_SPLIT_COUNTS
            or receipt.get("formal_owner_transport_authorized") != 0
            or receipt.get("formal_private_evaluation_runs") != 0
            or receipt.get("main_session_case_payload_reads") != 0
            or receipt.get("main_session_conllu_content_reads") != 0
            or receipt.get("source_key") != TUECL_SOURCE_KEY
            or receipt.get("v7_source_manifest_sha256") != source_sha):
        raise W02MorphologySuccessorV4R6FeasibilityReceiptError(
            "R6 feasibility receipt 状态漂移")
    return receipt


__all__ = [
    "W02_MORPH_V4_R6_DIMENSION_COUNTS",
    "W02_MORPH_V4_R6_FEASIBILITY_METADATA_SIZE_BYTES",
    "W02_MORPH_V4_R6_FEASIBILITY_OWNER_FAMILY_KEY",
    "W02_MORPH_V4_R6_FEASIBILITY_OWNER_ID",
    "W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_PATH",
    "W02_MORPH_V4_R6_FEASIBILITY_RECEIPT_VERSION",
    "W02_MORPH_V4_R6_SPLIT_COUNTS",
    "W02MorphologySuccessorV4R6FeasibilityReceiptError",
    "build_w02_morphology_successor_v4_r6_feasibility_receipt",
    "publish_w02_morphology_successor_v4_r6_feasibility_receipt",
    "read_w02_morphology_successor_v4_r6_feasibility_receipt",
    "validate_w02_morphology_successor_v4_r6_feasibility_metadata",
]
