"""在唯一 formal guard 后物化 recovery-v6 Qt held-out labels。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V5_DIMENSIONS,
    read_normalization_recovery_v5_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_qt_source_pack import (
    QT_ARCHIVE_NAME,
    read_normalization_recovery_v5_qt_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_qt_source_records import (
    parse_normalization_recovery_v5_qt_archive,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_QT_EVALUATION_RECORD_V1")


def _sha256(payload: bytes) -> str:
    """返回 inventory、record roster 或 label payload identity。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(left, right)
                        for left, right in zip(value, expected)))
    return value == expected


def _identity(pair: dict[str, object]) -> dict[str, object]:
    """从 guard 后重派生 pair 构造冻结的无 label identity。"""
    return {
        "format_version": 1,
        "pair_id": pair["pair_id"],
        "record_kind": "QT_TRANSLATIONS_HELD_OUT_IDENTITY_V1",
        "source_identity": pair["source_identity"],
        "source_identity_sha256": pair["source_identity_sha256"],
        "zh_hans_source_file_id": pair["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": pair["zh_hant"]["source_file_id"],
    }


def _single_han_difference(
        input_text: str,
        expected_output: str,
        ) -> tuple[str, str, int] | None:
    """返回唯一不同 scalar 的输入、输出和偏移。"""
    if len(input_text) != len(expected_output):
        return None
    values = tuple(
        (left, right, offset)
        for offset, (left, right) in enumerate(zip(input_text, expected_output))
        if left != right)
    return values[0] if len(values) == 1 else None


def _evaluation_records(
        pairs: tuple[dict[str, object], ...],
        *,
        source_pack_manifest_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """把完整 Qt pair roster 转成不筛选的八维 evaluation records。"""
    mapping_outputs: dict[str, set[str]] = defaultdict(set)
    output_by_input: dict[str, set[str]] = defaultdict(set)
    differences = {}
    for pair in pairs:
        input_text = str(pair["zh_hant"]["translation"])
        expected = str(pair["zh_hans"]["translation"])
        output_by_input[input_text].add(expected)
        if pair["single_han_difference"] == 1:
            difference = _single_han_difference(input_text, expected)
            if difference is None:
                raise BroadQaExternalDataError("v6 Qt single-Han feature 漂移")
            differences[str(pair["pair_id"])] = difference
            mapping_outputs[difference[0]].add(difference[1])
    values = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        input_text = str(pair["zh_hant"]["translation"])
        expected = str(pair["zh_hans"]["translation"])
        difference = differences.get(pair_id)
        context_conditioned = int(
            difference is not None and len(mapping_outputs[difference[0]]) > 1)
        local_mapping = int(difference is not None and not context_conditioned)
        record = {
            "contains_han_both": pair["contains_han_both"],
            "context_conditioned": context_conditioned,
            "equal_length": pair["equal_length"],
            "evaluation_id": pair_id,
            "expected_output": expected,
            "format_version": 1,
            "identity_preservation": pair["identity_preservation"],
            "input_scalar_count": len(input_text),
            "input_text": input_text,
            "local_mapping": local_mapping,
            "output_scalar_count": len(expected),
            "pair_id": pair_id,
            "record_kind": NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND,
            "single_han_difference": pair["single_han_difference"],
            "source_conflict": int(len(output_by_input[input_text]) > 1),
            "source_identity": pair["source_identity"],
            "source_identity_sha256": pair["source_identity_sha256"],
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "structure_equal": pair["structure_equal"],
            "structure_tokens": pair["zh_hant_structure_tokens"],
            "variable_length": 1 - int(pair["equal_length"]),
            "within_scalar_limit": pair["within_scalar_limit"],
            "zh_hans_source_file_id": pair["zh_hans"]["source_file_id"],
            "zh_hant_source_file_id": pair["zh_hant"]["source_file_id"],
        }
        if difference is not None:
            record.update({
                "mapping_expected_character": difference[1],
                "mapping_input_character": difference[0],
                "mapping_offset": difference[2],
            })
        values.append(record)
    result = tuple(values)
    if (not result or len({item["evaluation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError("v6 Qt evaluation identity 非法")
    return result


def materialize_normalization_recovery_v6_labels_after_guard(
        *,
        guard_consumed: int,
        qt_source_pack_dir: str | Path,
        expected_qt_source_manifest_sha256: str,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """guard 后双重派生 Qt labels，并逐项核对冻结 identity 与分母。"""
    if type(guard_consumed) is not int or guard_consumed != 1:
        raise BroadQaExternalDataError("v6 Qt labels 只能在 formal guard 后物化")
    source_root = Path(qt_source_pack_dir).resolve()
    commitment = read_normalization_recovery_v5_evaluation_commitment(
        evaluation_commitment_dir,
        qt_source_pack_dir=source_root,
        expected_qt_source_manifest_sha256=(
            expected_qt_source_manifest_sha256),
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256),
    )
    source_manifest, _source_files, stored_inventory = (
        read_normalization_recovery_v5_qt_source_pack(source_root))
    try:
        archive_payload = (source_root / QT_ARCHIVE_NAME).read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError("v6 Qt raw archive 不可读") from error
    _derived_files, pairs, summary = (
        parse_normalization_recovery_v5_qt_archive(archive_payload))
    derived_inventory = tuple(_identity(pair) for pair in pairs)
    denominator = commitment.get("denominator")
    aggregates = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    if (source_manifest.get("manifest_sha256")
            != expected_qt_source_manifest_sha256
            or commitment.get("manifest_sha256")
            != expected_evaluation_commitment_manifest_sha256
            or commitment.get("source_exclusion", {}).get(
                "excluded_source_pack_manifest_sha256")
            != source_manifest.get("manifest_sha256")
            or not _strict_equal(commitment.get("dimensions"),
                                 NORMALIZATION_RECOVERY_V5_DIMENSIONS)
            or not _strict_equal(derived_inventory, stored_inventory)
            or not isinstance(aggregates, dict)
            or denominator.get("record_count") != len(pairs)
            or summary.get("identity_pair_count")
            != aggregates.get("identity_count")
            or summary.get("nonidentity_pair_count")
            != aggregates.get("nonidentity_count")
            or summary.get("equal_length_pair_count")
            != aggregates.get("equal_length_count")
            or summary.get("variable_length_pair_count")
            != aggregates.get("variable_length_count")
            or summary.get("single_han_difference_count")
            != aggregates.get("single_han_difference_count")):
        raise BroadQaExternalDataError("v6 Qt label/identity/denominator 漂移")
    records = _evaluation_records(
        pairs, source_pack_manifest_sha256=str(
            source_manifest["manifest_sha256"]))
    if (len(records) != len(stored_inventory)
            or sum(item["identity_preservation"] for item in records)
            != aggregates["identity_count"]
            or sum(item["variable_length"] for item in records)
            != aggregates["variable_length_count"]
            or sum(item["single_han_difference"] for item in records)
            != aggregates["single_han_difference_count"]
            or sum(item["local_mapping"] + item["context_conditioned"]
                   for item in records)
            != aggregates["single_han_difference_count"]):
        raise BroadQaExternalDataError("v6 Qt materialized record 分账漂移")
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "evaluation_record_roster_sha256": _sha256(
            canonical_json_bytes(records)),
        "inventory_identity_sha256": denominator[
            "identity_artifact"]["sha256"],
        "label_materialization_count": len(records),
        "qt_archive_parse_count": 2,
        "qt_source_manifest_sha256": source_manifest["manifest_sha256"],
        "qt_source_payload_read_count": 1,
        "source_identity_reselection_count": 0,
    }
    return materialization, records


__all__ = [
    "NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND",
    "materialize_normalization_recovery_v6_labels_after_guard",
]
