"""Strict public receipt for the isolated successor V2 blind owner."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    BLIND_PRIVATE_SOURCE_EXTENSION_PATH,
    read_blind_private_source_extension_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorResourceBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import W02FileFreeze


W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_private_owner_receipt_v1.json"
)
W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-PRIVATE-OWNER-RECEIPT-V1"
)
W02_MORPH_V2_PRIVATE_OWNER_METADATA_SHA256 = (
    "70589d0d62007c5e7be7a2d99c9457361a11e77d03c008eb1a38aa058087cd2a"
)
W02_MORPH_V2_PRIVATE_OWNER_METADATA_SIZE_BYTES = 9277
W02_MORPH_V2_PRIVATE_OWNER_ID = "884d5696c8e244ab"
W02_MORPH_V2_PRIVATE_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V2-UD-CFL-HK-BLIND-PRIVATE-V1"
)
W02_MORPH_V2_PRIVATE_LAYOUTS = (
    "PRIVATE_SOURCE",
    "PRIVATE_HELD_OUT_OBSERVATION",
    "PRIVATE_ADVERSARIAL_OBSERVATION",
    "PRIVATE_WALL_OBSERVATION",
    "PRIVATE_HELD_OUT_LABEL",
    "PRIVATE_ADVERSARIAL_LABEL",
    "PRIVATE_WALL_LABEL",
)
W02_MORPH_V2_PRIVATE_PATHS = {
    "PRIVATE_SOURCE": "source/source_refs.jsonl.gz",
    "PRIVATE_HELD_OUT_OBSERVATION": "observations/held_out.jsonl.gz",
    "PRIVATE_ADVERSARIAL_OBSERVATION": "observations/adversarial.jsonl.gz",
    "PRIVATE_WALL_OBSERVATION": "observations/wall.jsonl.gz",
    "PRIVATE_HELD_OUT_LABEL": "evaluator/held_out.labels.jsonl.gz",
    "PRIVATE_ADVERSARIAL_LABEL": "evaluator/adversarial.labels.jsonl.gz",
    "PRIVATE_WALL_LABEL": "evaluator/wall.labels.jsonl.gz",
}
W02_MORPH_V2_PRIVATE_SPLITS = ("held_out", "adversarial", "wall")
W02_MORPH_V2_PRIVATE_SOURCE_KEYS = (
    "UD_ZH_CFL_R2_18_BLIND_PRIVATE",
    "UD_ZH_HK_R2_18_BLIND_PRIVATE",
)
W02_MORPH_V2_PRIVATE_SOURCE_SNAPSHOTS = {
    "UD_ZH_CFL_R2_18_BLIND_PRIVATE": (
        "a608b5d73136200246e5320f91c8e1009a32d3dce10be8266962548cb62397e8"
    ),
    "UD_ZH_HK_R2_18_BLIND_PRIVATE": (
        "cea5801bc5db2d9cd3c7055c827268cd3fff50a29a0075cb50d1511e8f735ba2"
    ),
}
W02_MORPH_V2_PRIVATE_PAIR_COUNT = 1409
W02_MORPH_V2_PRIVATE_SOURCE_COUNT = 1409


# object-model: exception
class W02MorphologySuccessorV2PrivateOwnerError(RuntimeError):
    """The payload-free owner receipt or its public dependencies drifted."""


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            f"{where} is not lowercase SHA-256")
    return value


def _exact(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV2PrivateOwnerError(
            f"{where} fields drifted")
    return value


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or not target.is_relative_to(repository) or target.is_symlink()
            or not target.is_file()):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt repository path is invalid")
    return target


def _validate_zero_overlap(value: object, *, where: str) -> None:
    row = _exact(value, {
        "authorized_observation_count", "exact_case_overlap_count",
        "exact_cluster_overlap_count", "exact_content_overlap_count",
        "normalized_content_overlap_count", "transport_commitment",
    }, where=where)
    if (type(row["authorized_observation_count"]) is not int
            or row["authorized_observation_count"] <= 0
            or any(row[key] != 0 for key in (
                "exact_case_overlap_count", "exact_cluster_overlap_count",
                "exact_content_overlap_count",
                "normalized_content_overlap_count"))):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            f"{where} contamination audit did not close")
    _sha256(row["transport_commitment"], where=f"{where} commitment")


def _validate_file_inventory(value: object) -> tuple[W02FileFreeze, ...]:
    if not isinstance(value, list) or len(value) != len(W02_MORPH_V2_PRIVATE_LAYOUTS):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt file inventory is incomplete")
    files_list = []
    for row in value:
        raw = _exact(row, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "relative_path", "root_key", "split",
            "transport_sha256", "transport_size_bytes",
        }, where="owner file identity")
        layout = raw["layout_key"]
        if (layout not in W02_MORPH_V2_PRIVATE_PATHS
                or raw["relative_path"] != W02_MORPH_V2_PRIVATE_PATHS[layout]):
            raise W02MorphologySuccessorV2PrivateOwnerError(
                "owner receipt relative path drifted")
        files_list.append(W02FileFreeze.from_dict({
            key: item for key, item in raw.items() if key != "relative_path"
        }))
    files = tuple(files_list)
    if tuple(row.layout_key for row in files) != W02_MORPH_V2_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt layout order drifted")
    by_layout = {row.layout_key: row for row in files}
    if (by_layout["PRIVATE_SOURCE"].record_count
            != W02_MORPH_V2_PRIVATE_SOURCE_COUNT
            or by_layout["PRIVATE_SOURCE"].record_kind != "source_ref"
            or by_layout["PRIVATE_SOURCE"].split
            or any(row.license_ids != ("CC-BY-SA-4.0",) for row in files)):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt source/license inventory drifted")
    for split in W02_MORPH_V2_PRIVATE_SPLITS:
        name = split.upper()
        observation = by_layout[f"PRIVATE_{name}_OBSERVATION"]
        label = by_layout[f"PRIVATE_{name}_LABEL"]
        if (observation.record_kind != "observation"
                or label.record_kind != "evaluator_label"
                or observation.split != split or label.split != split
                or observation.record_count != label.record_count):
            raise W02MorphologySuccessorV2PrivateOwnerError(
                "owner receipt observation/label inventory drifted")
    if sum(by_layout[f"PRIVATE_{split.upper()}_OBSERVATION"].record_count
           for split in W02_MORPH_V2_PRIVATE_SPLITS
           ) != W02_MORPH_V2_PRIVATE_PAIR_COUNT:
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt pair count drifted")
    return files


def validate_w02_morphology_successor_v2_private_owner_receipt(
        value: object,
        repository_root: str | Path,
        ) -> tuple[dict[str, Any], tuple[W02FileFreeze, ...]]:
    """Validate the complete aggregate receipt without opening owner payload."""
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "audit_report",
        "candidate_evaluation_runs", "commitments", "contamination_audit",
        "dimension_denominator_counts", "file_count", "files",
        "formal_private_evaluation_runs", "main_session_private_payload_reads",
        "next_action", "old_private_source_identity_audit",
        "owner_family_key", "owner_id", "owner_metadata_sha256",
        "owner_metadata_size_bytes", "owner_teacher_llm_provenance",
        "pair_count", "public_repository", "resource_budget", "source_count",
        "source_snapshot_commitments", "status", "within_owner_duplicate_audit",
    }, where="owner receipt")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_PRIVATE_OWNER_RECEIPT"
            or raw["artifact_version"]
            != W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_VERSION
            or raw["status"] != "OWNER_METADATA_INGESTED_PAYLOAD_UNREAD"
            or raw["next_action"] != "BUILD_PRIVATE_FAMILY_REGISTRATION_FREEZE"
            or raw["owner_family_key"] != W02_MORPH_V2_PRIVATE_OWNER_FAMILY_KEY
            or raw["owner_id"] != W02_MORPH_V2_PRIVATE_OWNER_ID
            or raw["owner_metadata_sha256"]
            != W02_MORPH_V2_PRIVATE_OWNER_METADATA_SHA256
            or raw["owner_metadata_size_bytes"]
            != W02_MORPH_V2_PRIVATE_OWNER_METADATA_SIZE_BYTES
            or raw["file_count"] != len(W02_MORPH_V2_PRIVATE_LAYOUTS)
            or raw["pair_count"] != W02_MORPH_V2_PRIVATE_PAIR_COUNT
            or raw["source_count"] != W02_MORPH_V2_PRIVATE_SOURCE_COUNT
            or any(raw[key] != 0 for key in (
                "candidate_evaluation_runs", "formal_private_evaluation_runs",
                "main_session_private_payload_reads"))):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner receipt identity/status drifted")
    files = _validate_file_inventory(raw["files"])
    commitments = _exact(raw["commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="owner commitments")
    for key, digest in commitments.items():
        _sha256(digest, where=f"owner {key}")
    if len(set(commitments.values())) != len(commitments):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner commitments are not independent")
    dimensions = raw["dimension_denominator_counts"]
    if (not isinstance(dimensions, dict)
            or len(dimensions) != 5
            or any(type(count) is not int or count <= 0
                   for count in dimensions.values())
            or sum(dimensions.values()) != W02_MORPH_V2_PRIVATE_PAIR_COUNT):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner dimension denominators drifted")
    contamination = _exact(
        raw["contamination_audit"], {"dev", "shadow", "train"},
        where="owner contamination audit")
    for split in ("train", "dev", "shadow"):
        _validate_zero_overlap(contamination[split], where=f"owner {split}")
    duplicates = _exact(raw["within_owner_duplicate_audit"], {
        "exact_case_duplicate_count", "exact_cluster_collision_count",
        "exact_content_duplicate_count", "near_duplicate_pair_count",
        "normalized_content_duplicate_count",
    }, where="owner duplicate audit")
    if any(value != 0 for value in duplicates.values()):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner duplicate audit did not close")
    old_private = _exact(raw["old_private_source_identity_audit"], {
        "new_source_keys", "old_private_payload_reads", "parent_schema_sha256",
        "parent_source_keys", "policy", "source_key_intersection_count",
    }, where="old private identity audit")
    if (tuple(old_private["new_source_keys"]) != W02_MORPH_V2_PRIVATE_SOURCE_KEYS
            or old_private["old_private_payload_reads"] != 0
            or old_private["source_key_intersection_count"] != 0
            or old_private["policy"]
            != "SOURCE_IDENTITY_DISJOINT_WITH_ZERO_OLD_PRIVATE_READS"):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "old private source identity audit drifted")
    _sha256(old_private["parent_schema_sha256"], where="parent schema")
    if raw["source_snapshot_commitments"] != W02_MORPH_V2_PRIVATE_SOURCE_SNAPSHOTS:
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner source snapshot commitments drifted")
    audit = _exact(raw["audit_report"], {
        "relative_path", "sha256", "size_bytes",
    }, where="owner audit report")
    if (audit["relative_path"] != "owner-audit-report.json"
            or type(audit["size_bytes"]) is not int or audit["size_bytes"] <= 0):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner audit report identity drifted")
    _sha256(audit["sha256"], where="owner audit report")
    provenance = _exact(raw["owner_teacher_llm_provenance"], {
        "deterministic_adapter_runs", "llm_calls", "teacher_calls", "tool",
    }, where="owner teacher provenance")
    if (provenance["deterministic_adapter_runs"] != 2
            or provenance["llm_calls"] != 0
            or provenance["teacher_calls"] != 0
            or provenance["tool"] != "NONE"):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner teacher provenance drifted")
    budget = V2EvaluatorResourceBudget.from_dict(raw["resource_budget"])
    if (sum(row.transport_size_bytes for row in files) > budget.max_payload_bytes
            or W02_MORPH_V2_PRIVATE_SOURCE_COUNT
            + W02_MORPH_V2_PRIVATE_PAIR_COUNT * 2 > budget.max_records):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner inventory exceeds evaluator budget")
    public = _exact(raw["public_repository"], {
        "head_commit_sha1", "required_base_commit_sha1",
        "source_extension_sha256",
    }, where="owner public repository")
    if (public["head_commit_sha1"]
            != "aa5f695d993eef582195b814a4b63c7798186b90"
            or public["required_base_commit_sha1"] != public["head_commit_sha1"]):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner public base commit drifted")
    extension = read_blind_private_source_extension_manifest(repository_root)
    extension_path = _repository_file(
        Path(repository_root).resolve(), BLIND_PRIVATE_SOURCE_EXTENSION_PATH)
    extension_sha = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    if (extension["status"] != "BLIND_PRIVATE_SOURCE_EXTENSION_APPROVED"
            or public["source_extension_sha256"] != extension_sha):
        raise W02MorphologySuccessorV2PrivateOwnerError(
            "owner source extension binding drifted")
    return raw, files


def read_w02_morphology_successor_v2_private_owner_receipt(
        repository_root: str | Path,
        ) -> tuple[dict[str, Any], tuple[W02FileFreeze, ...]]:
    """Read the canonical public receipt; no private path is accepted."""
    repository = Path(repository_root).resolve()
    target = _repository_file(repository, W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_PATH)
    value = read_canonical_object(target)
    return validate_w02_morphology_successor_v2_private_owner_receipt(
        value, repository)


__all__ = [
    "W02_MORPH_V2_PRIVATE_LAYOUTS",
    "W02_MORPH_V2_PRIVATE_OWNER_FAMILY_KEY",
    "W02_MORPH_V2_PRIVATE_OWNER_METADATA_SHA256",
    "W02_MORPH_V2_PRIVATE_OWNER_METADATA_SIZE_BYTES",
    "W02_MORPH_V2_PRIVATE_OWNER_RECEIPT_PATH",
    "W02_MORPH_V2_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V2_PRIVATE_SOURCE_COUNT",
    "W02MorphologySuccessorV2PrivateOwnerError",
    "read_w02_morphology_successor_v2_private_owner_receipt",
    "validate_w02_morphology_successor_v2_private_owner_receipt",
]
