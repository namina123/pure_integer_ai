"""在唯一formal guard后物化 recovery-v9 GIMP labels。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
    read_normalization_recovery_v9_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_pack import (
    materialize_normalization_recovery_v9_source_pairs_after_guard,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_EVALUATION_RECORD_V1")


def _sha256(value: object) -> str:
    """返回物化roster或source identity的SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _records(
        pairs: tuple[dict[str, object], ...], *,
        source_manifest_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """把完整GIMP pair roster转为不筛选的formal records。"""
    values = []
    for pair in pairs:
        source_identity = pair.get("source_identity")
        zh_hans = pair.get("zh_hans")
        zh_hant = pair.get("zh_hant")
        if (not isinstance(source_identity, dict)
                or not isinstance(zh_hans, dict)
                or not isinstance(zh_hant, dict)
                or not isinstance(pair.get("official_source_text"), str)
                or not pair["official_source_text"]):
            raise BroadQaExternalDataError("v9 GIMP evaluation pair schema漂移")
        input_text = zh_hant.get("msgstr")
        expected = zh_hans.get("msgstr")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(expected, str) or not expected):
            raise BroadQaExternalDataError("v9 GIMP evaluation label非法")
        values.append({
            "contains_han_both": pair["contains_han_both"],
            "equal_length": pair["equal_length"],
            "evaluation_eligible": pair["v9_evaluation_eligible"],
            "evaluation_id": pair["pair_id"],
            "expected_output": expected,
            "format_version": 1,
            "identity_preservation": pair["identity_preservation"],
            "input_text": input_text,
            "official_source_text": pair["official_source_text"],
            "record_kind": NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND,
            "single_han_difference": pair["single_han_difference"],
            "source_identity": source_identity,
            "source_identity_sha256": pair["source_identity_sha256"],
            "source_pack_manifest_sha256": source_manifest_sha256,
            "structure_equal": pair["structure_equal"],
            "structure_tokens": pair["zh_hant_structure_tokens"],
            "variable_length": 1 - int(pair["equal_length"]),
            "within_scalar_limit": pair["within_scalar_limit"],
            "zh_hans_source_file_id": zh_hans["source_file_id"],
            "zh_hant_source_file_id": zh_hant["source_file_id"],
        })
    result = tuple(values)
    if (not result or len({item["evaluation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError("v9 GIMP evaluation roster非法")
    return result


def _aggregate(
        records: tuple[dict[str, object], ...],
        ) -> dict[str, int]:
    """从完整formal records重算commitment十项aggregate。"""
    outputs_by_input: dict[str, set[str]] = {}
    for item in records:
        outputs_by_input.setdefault(str(item["input_text"]), set()).add(
            str(item["expected_output"]))
    return {
        "contains_han_both_count": sum(
            int(item["contains_han_both"]) for item in records),
        "equal_length_count": sum(int(item["equal_length"]) for item in records),
        "evaluation_eligible_count": sum(
            int(item["evaluation_eligible"]) for item in records),
        "identity_count": sum(
            int(item["identity_preservation"]) for item in records),
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "nonidentity_count": sum(
            1 - int(item["identity_preservation"]) for item in records),
        "single_han_difference_count": sum(
            int(item["single_han_difference"]) for item in records),
        "structure_equal_count": sum(
            int(item["structure_equal"]) for item in records),
        "structure_unequal_count": sum(
            1 - int(item["structure_equal"]) for item in records),
        "variable_length_count": sum(
            int(item["variable_length"]) for item in records),
    }


def materialize_normalization_recovery_v9_labels_after_guard(
        *, guard_consumed: int,
        gimp_source_pack_dir: str | Path,
        expected_gimp_source_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        runtime_gate_dir: str | Path,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """guard后重建GIMP labels，并核对identity与完整分母。"""
    if type(guard_consumed) is not int or guard_consumed != 1:
        raise BroadQaExternalDataError(
            "v9 GIMP labels只能在formal guard后物化")
    commitment = read_normalization_recovery_v9_evaluation_commitment(
        evaluation_commitment_dir,
        source_pack_dir=gimp_source_pack_dir,
        runtime_gate_dir=runtime_gate_dir,
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256))
    source_manifest, pairs, summary = (
        materialize_normalization_recovery_v9_source_pairs_after_guard(
            gimp_source_pack_dir,
            expected_manifest_sha256=expected_gimp_source_manifest_sha256,
            guard_consumed=guard_consumed))
    records = _records(
        pairs, source_manifest_sha256=str(source_manifest["manifest_sha256"]))
    denominator = commitment.get("denominator")
    buckets = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    if (commitment.get("manifest_sha256")
            != expected_evaluation_commitment_manifest_sha256
            or source_manifest.get("manifest_sha256")
            != expected_gimp_source_manifest_sha256
            or not isinstance(buckets, dict)
            or denominator.get("record_count") != len(records)
            or _aggregate(records) != buckets
            or summary.get("plain_pair_count") != len(records)
            or denominator.get("identity_artifact", {}).get("record_count")
            != len(records)):
        raise BroadQaExternalDataError(
            "v9 GIMP label/identity/denominator漂移")
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_record_roster_sha256": _sha256(records),
        "gimp_archive_parse_count": 1,
        "gimp_source_manifest_sha256": source_manifest["manifest_sha256"],
        "gimp_source_payload_read_count": 1,
        "inventory_identity_sha256": denominator["identity_artifact"]["sha256"],
        "label_materialization_count": len(records),
        "source_identity_reselection_count": 0,
    }
    return materialization, records


__all__ = [
    "NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND",
    "materialize_normalization_recovery_v9_labels_after_guard",
]
