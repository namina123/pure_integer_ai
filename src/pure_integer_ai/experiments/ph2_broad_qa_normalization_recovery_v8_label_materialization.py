"""在唯一 formal guard 后物化 recovery-v8 VLC labels。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    strict_json_equal,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vlc_source_pack import (
    VLC_ARCHIVE_NAME,
    read_normalization_recovery_v7_vlc_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vlc_source_records import (
    parse_normalization_recovery_v7_vlc_archive,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
    read_normalization_recovery_v8_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_VLC_EVALUATION_RECORD_V1")


def _sha256(value: object) -> str:
    """返回物化 roster 或来源 identity 的 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identity(pair: dict[str, object]) -> dict[str, object]:
    """从 guard 后 pair 重建 source pack 的无 label identity。"""
    return {
        "format_version": 1,
        "pair_id": pair["pair_id"],
        "record_kind": "VLC_HELD_OUT_IDENTITY_WITHOUT_LABEL_V1",
        "source_identity": pair["source_identity"],
        "source_identity_sha256": pair["source_identity_sha256"],
        "zh_hans_source_file_id": pair["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": pair["zh_hant"]["source_file_id"],
    }


def _records(
        pairs: tuple[dict[str, object], ...], *, source_manifest_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """把完整 VLC pair roster 转为不筛选的 formal records。"""
    values = []
    for pair in pairs:
        source_identity = pair.get("source_identity")
        zh_hans = pair.get("zh_hans")
        zh_hant = pair.get("zh_hant")
        if (not isinstance(source_identity, dict)
                or not isinstance(zh_hans, dict) or not isinstance(zh_hant, dict)
                or not isinstance(source_identity.get("msgid"), str)
                or not source_identity["msgid"]):
            raise BroadQaExternalDataError("v8 VLC evaluation pair schema 漂移")
        input_text = zh_hant.get("msgstr")
        expected = zh_hans.get("msgstr")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(expected, str) or not expected):
            raise BroadQaExternalDataError("v8 VLC evaluation label 非法")
        values.append({
            "equal_length": pair["equal_length"],
            "evaluation_id": pair["pair_id"],
            "expected_output": expected,
            "format_version": 1,
            "identity_preservation": pair["identity_preservation"],
            "input_text": input_text,
            "official_source_text": source_identity["msgid"],
            "record_kind": NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND,
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
        raise BroadQaExternalDataError("v8 VLC evaluation roster 非法")
    return result


def materialize_normalization_recovery_v8_labels_after_guard(
        *, guard_consumed: int, vlc_source_pack_dir: str | Path,
        expected_vlc_source_manifest_sha256: str,
        v7_commitment_dir: str | Path,
        expected_v7_commitment_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """guard 后双重派生 VLC labels，并逐项核对 identity 与分母。"""
    if type(guard_consumed) is not int or guard_consumed != 1:
        raise BroadQaExternalDataError("v8 VLC labels 只能在 formal guard 后物化")
    source_root = Path(vlc_source_pack_dir).resolve()
    commitment = read_normalization_recovery_v8_evaluation_commitment(
        evaluation_commitment_dir,
        v7_commitment_dir=v7_commitment_dir,
        expected_v7_commitment_manifest_sha256=(
            expected_v7_commitment_manifest_sha256),
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256))
    source_manifest, _source_files, stored_inventory = (
        read_normalization_recovery_v7_vlc_source_pack(source_root))
    try:
        archive_payload = (source_root / VLC_ARCHIVE_NAME).read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError("v8 VLC raw archive 不可读") from error
    _derived_files, pairs, summary = (
        parse_normalization_recovery_v7_vlc_archive(archive_payload))
    derived_inventory = tuple(_identity(pair) for pair in pairs)
    denominator = commitment.get("denominator")
    buckets = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    checks = {
        "equal_length_count": summary.get("equal_length_pair_count"),
        "identity_count": summary.get("identity_pair_count"),
        "nonidentity_count": summary.get("nonidentity_pair_count"),
        "single_han_difference_count": summary.get(
            "single_han_difference_count"),
        "structure_equal_count": summary.get("structure_equal_count"),
        "variable_length_count": summary.get("variable_length_pair_count"),
    }
    if (source_manifest.get("manifest_sha256")
            != expected_vlc_source_manifest_sha256
            or commitment.get("manifest_sha256")
            != expected_evaluation_commitment_manifest_sha256
            or commitment.get("source_exclusion", {}).get(
                "excluded_source_pack_manifest_sha256")
            != source_manifest.get("manifest_sha256")
            or not strict_json_equal(derived_inventory, stored_inventory)
            or not isinstance(buckets, dict)
            or denominator.get("record_count") != len(pairs)
            or any(checks[name] != buckets.get(name) for name in checks)):
        raise BroadQaExternalDataError("v8 VLC label/identity/denominator 漂移")
    records = _records(
        pairs, source_manifest_sha256=str(source_manifest["manifest_sha256"]))
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_record_roster_sha256": _sha256(records),
        "inventory_identity_sha256": denominator["identity_artifact"]["sha256"],
        "label_materialization_count": len(records),
        "source_identity_reselection_count": 0,
        "vlc_archive_parse_count": 2,
        "vlc_source_manifest_sha256": source_manifest["manifest_sha256"],
        "vlc_source_payload_read_count": 1,
    }
    return materialization, records


__all__ = [
    "NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND",
    "materialize_normalization_recovery_v8_labels_after_guard",
]
