"""派生 recovery-v10 新 TRAIN family 的自包含 source pack records。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content import (
    read_normalization_recovery_v10_source_expansion_content_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content_records import (
    V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256,
    derive_normalization_recovery_v10_candidate_source_records,
    read_normalization_recovery_v10_candidate_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    read_normalization_recovery_v10_source_expansion_roster,
)


V10_SOURCE_EXPANSION_CONTENT_MANIFEST_SHA256 = (
    "537fe632f835f0ed40b54b3e075775768b8a396535a3a549704268ce154fe56d")
V10_SOURCE_EXPANSION_SOURCE_PACK_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_SOURCE_PACK_CENSUS_V1")
V10_SOURCE_EXPANSION_SOURCE_FAMILIES = (
    "MIXXX_PROJECT",
    "MUMBLE_PROJECT",
)


def read_normalization_recovery_v10_source_expansion_source_pack_state(
        *,
        source_family: str,
        roster_dir: str | Path,
        content_dir: str | Path,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """严格回读名册与aggregate content，并选择唯一family记录。"""
    if source_family not in V10_SOURCE_EXPANSION_SOURCE_FAMILIES:
        raise BroadQaExternalDataError(
            "v10 source pack family 未支持")
    _roster_manifest, roster_outputs = (
        read_normalization_recovery_v10_source_expansion_roster(
            roster_dir,
            expected_manifest_sha256=(
                V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256),
        ))
    _content_manifest, content_outputs = (
        read_normalization_recovery_v10_source_expansion_content_aggregate(
            content_dir,
            expected_manifest_sha256=(
                V10_SOURCE_EXPANSION_CONTENT_MANIFEST_SHA256),
        ))
    roster = {
        str(item.get("source_family")): item
        for item in roster_outputs["source-candidates.jsonl"]
        if item.get("selection_status")
        == "SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING"
    }
    content = {
        str(item.get("source_family")): item
        for item in content_outputs["source-content.jsonl"]
    }
    if (set(roster) != set(V10_SOURCE_EXPANSION_SOURCE_FAMILIES)
            or set(content) != set(V10_SOURCE_EXPANSION_SOURCE_FAMILIES)):
        raise BroadQaExternalDataError(
            "v10 source pack predecessor inventory 漂移")
    return roster[source_family], content[source_family]


def _validate_content_record(
        roster: dict[str, object],
        content: dict[str, object],
        *,
        parser_summary: dict[str, object],
        pair_count: int,
        ) -> None:
    """要求pack完整重派生与sealed aggregate逐字段一致。"""
    license_value = roster.get("license")
    locale_files = roster.get("locale_files")
    if (not isinstance(license_value, dict)
            or not isinstance(locale_files, list)
            or content.get("content_outcome")
            != "PASS_NONZERO_ACTIVE_COMMON_PAIR"
            or content.get("selection_outcome")
            != "PASS_CONTENT_AND_PREDECESSOR_SOURCE_INPUT_OVERLAP"
            or content.get("source_family") != roster.get("source_family")
            or content.get("license_expression")
            != license_value.get("expression")
            or content.get("locale_file_read_count") != len(locale_files)
            or content.get("transient_pair_count") != pair_count
            or content.get("parser_summary") != parser_summary):
        raise BroadQaExternalDataError(
            "v10 source pack content aggregate 漂移")


def derive_normalization_recovery_v10_source_expansion_source_pack_records(
        *,
        roster: dict[str, object],
        content: dict[str, object],
        source_root: str | Path,
        ) -> tuple[
            dict[str, bytes],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
            dict[str, object],
            str,
        ]:
    """逐blob核验并派生完整source files、pairs、census和scope。"""
    family = roster.get("source_family")
    if family not in V10_SOURCE_EXPANSION_SOURCE_FAMILIES:
        raise BroadQaExternalDataError(
            "v10 source pack roster family 漂移")
    payloads = read_normalization_recovery_v10_candidate_payloads(
        roster, source_root)
    file_records, pairs, parser_summary = (
        derive_normalization_recovery_v10_candidate_source_records(
            roster, payloads))
    _validate_content_record(
        roster, content,
        parser_summary=parser_summary,
        pair_count=len(pairs),
    )
    license_value = roster.get("license")
    locale_files = roster.get("locale_files")
    if (not isinstance(license_value, dict)
            or not isinstance(license_value.get("files"), list)
            or not isinstance(locale_files, list)):
        raise BroadQaExternalDataError(
            "v10 source pack roster files 漂移")
    license_paths = {
        str(item["relative_path"]) for item in license_value["files"]}
    locale_paths = {str(item["relative_path"]) for item in locale_files}
    scopes = {str(item.get("source_policy_scope")) for item in pairs}
    if (set(payloads) != license_paths.union(locale_paths)
            or len(file_records) != len(locale_paths)
            or len(scopes) != 1 or "" in scopes):
        raise BroadQaExternalDataError(
            "v10 source pack raw/parser inventory 漂移")
    source_policy_scope = next(iter(scopes))
    eligible = int(parser_summary["v8_training_eligible_pair_count"])
    census = {
        "changed_pair_count": int(content["changed_pair_count"]),
        "excluded_or_ineligible_pair_count": len(pairs) - eligible,
        "format_version": 1,
        "identity_pair_count": int(parser_summary["identity_pair_count"]),
        "license_file_count": len(license_paths),
        "locale_file_count": len(locale_paths),
        "pair_record_count": len(pairs),
        "pair_surface_public_git_count": 0,
        "raw_blob_count": len(payloads),
        "record_kind": V10_SOURCE_EXPANSION_SOURCE_PACK_CENSUS_KIND,
        "source_family": family,
        "source_file_record_count": len(file_records),
        "source_pack_family_vote_count": 1,
        "structure_equal_pair_count": int(
            parser_summary["structure_equal_count"]),
        "structure_unequal_pair_count": (
            len(pairs) - int(parser_summary["structure_equal_count"])),
        "v10_training_eligible_pair_count": eligible,
    }
    return (
        payloads,
        file_records,
        pairs,
        census,
        parser_summary,
        source_policy_scope,
    )


__all__ = [
    "V10_SOURCE_EXPANSION_CONTENT_MANIFEST_SHA256",
    "V10_SOURCE_EXPANSION_SOURCE_FAMILIES",
    "V10_SOURCE_EXPANSION_SOURCE_PACK_CENSUS_KIND",
    "derive_normalization_recovery_v10_source_expansion_source_pack_records",
    "read_normalization_recovery_v10_source_expansion_source_pack_state",
]
