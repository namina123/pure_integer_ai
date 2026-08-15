"""派生 recovery-v7 neutral upstream source projection 记录。

本模块只消费调用方已经严格回读的四套 TRAIN source pack。Godot 与
LibreOffice 投影 gettext ``msgid``，VS Code 仅投影 JSON key leaf 候选；
Thunderbird 明确记录为无 neutral surface。返回值不包含 source、input 或
output 原文，只保留可从冻结 source pack 重建的承诺与支持事实。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


NEUTRAL_SOURCE_PROJECTION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_V1")
NEUTRAL_SOURCE_FAMILY_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_FAMILY_V1")
NEUTRAL_SOURCE_SUPPORT_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_SUPPORT_V1")

GODOT_SOURCE_FAMILY = "GODOT_ENGINE_PROJECT"
LIBREOFFICE_SOURCE_FAMILY = "LIBREOFFICE_PROJECT"
VSCODE_SOURCE_FAMILY = "MICROSOFT_VSCODE_PROJECT"
THUNDERBIRD_SOURCE_FAMILY = "THUNDERBIRD_PROJECT"

GETTEXT_SOURCE_PROJECTION = "GETTEXT_MSGID_SOURCE_SURFACE"
VSCODE_LEAF_PROJECTION = "VSCODE_JSON_KEY_LEAF_CANDIDATE"
NEUTRAL_SURFACE_UNAVAILABLE = "NEUTRAL_SURFACE_UNAVAILABLE"

_ASCII_LETTER = re.compile(r"[A-Za-z]")
_SOURCE_ORDER = (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)
_PAIRWISE_ORDER = (
    (GODOT_SOURCE_FAMILY, LIBREOFFICE_SOURCE_FAMILY),
    (GODOT_SOURCE_FAMILY, VSCODE_SOURCE_FAMILY),
    (LIBREOFFICE_SOURCE_FAMILY, VSCODE_SOURCE_FAMILY),
)


def _sha256(payload: bytes) -> str:
    """返回规范值或 UTF-8 surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _binary(value: object, *, label: str) -> int:
    """读取严格 JSON 二值，拒绝 bool、缺失与其他整数。"""
    if type(value) is not int or value not in (0, 1):
        raise BroadQaExternalDataError(
            f"v7 neutral source projection {label} 非二值")
    return value


def _text_commitment(value: str) -> dict[str, object]:
    """形成不泄露原文的 UTF-8 文本承诺。"""
    if not isinstance(value, str) or not value:
        raise BroadQaExternalDataError(
            "v7 neutral source projection surface 为空或非法")
    encoded = value.encode("utf-8")
    return {
        "bytes": len(encoded),
        "scalar_length": len(value),
        "sha256": _sha256(encoded),
    }


def _manifest_evidence(
        manifest: dict[str, object],
        *,
        expected_family: str,
        ) -> dict[str, str]:
    """从冻结 source manifest 提取来源、策略与许可承诺。"""
    license_value = manifest.get("license")
    manifest_sha256 = manifest.get("manifest_sha256")
    policy = manifest.get("source_policy_scope")
    if (manifest.get("source_family") != expected_family
            or not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or not isinstance(policy, str) or not policy
            or not isinstance(license_value, dict)
            or not isinstance(license_value.get("license_id"), str)):
        raise BroadQaExternalDataError(
            "v7 neutral source projection source manifest 非法")
    return {
        "license_evidence_sha256": _sha256(
            canonical_json_bytes(license_value)),
        "license_id": str(license_value["license_id"]),
        "source_pack_manifest_sha256": manifest_sha256,
        "source_policy_scope": policy,
    }


def _identity_commitment(value: dict[str, object]) -> str:
    """承诺完整 adapter source identity，不发布 identity surface。"""
    if not isinstance(value, dict) or not value:
        raise BroadQaExternalDataError(
            "v7 neutral source projection source identity 非法")
    return _sha256(canonical_json_bytes(value))


def _row(
        *,
        pair_id: object,
        source_family: str,
        evidence: dict[str, str],
        projection_kind: str,
        source_identity_kind: str,
        source_identity: dict[str, object],
        neutral_surface: str,
        output_text: str,
        variable_length: object,
        structured: bool,
        candidate_sentence_like: int,
        ) -> dict[str, object]:
    """构造一个含瞬时原文的内部投影行。"""
    if (not isinstance(pair_id, str) or len(pair_id) != 64
            or type(variable_length) is not bool
            or candidate_sentence_like not in (0, 1)):
        raise BroadQaExternalDataError(
            "v7 neutral source projection pair 字段非法")
    surface = _text_commitment(neutral_surface)
    output = _text_commitment(output_text)
    identity_sha256 = _identity_commitment(source_identity)
    projection_identity = {
        "adapter_projection_kind": projection_kind,
        "pair_id": pair_id,
        "source_family": source_family,
        "source_identity_sha256": identity_sha256,
        "source_pack_manifest_sha256": evidence[
            "source_pack_manifest_sha256"],
    }
    return {
        "_neutral_surface": neutral_surface,
        "adapter_projection_kind": projection_kind,
        "candidate_sentence_like": candidate_sentence_like,
        "format_version": 1,
        "license_evidence_sha256": evidence[
            "license_evidence_sha256"],
        "license_id": evidence["license_id"],
        "neutral_surface_bytes": surface["bytes"],
        "neutral_surface_scalar_length": surface["scalar_length"],
        "neutral_surface_sha256": surface["sha256"],
        "output_bytes": output["bytes"],
        "output_scalar_length": output["scalar_length"],
        "output_sha256": output["sha256"],
        "pair_id": pair_id,
        "projection_id": _sha256(canonical_json_bytes(
            projection_identity)),
        "record_kind": NEUTRAL_SOURCE_PROJECTION_RECORD_KIND,
        "source_family": source_family,
        "source_identity_kind": source_identity_kind,
        "source_identity_sha256": identity_sha256,
        "source_pack_manifest_sha256": evidence[
            "source_pack_manifest_sha256"],
        "source_policy_scope": evidence["source_policy_scope"],
        "structured": int(structured),
        "variable_length": int(variable_length),
    }


def _godot_rows(
        manifest: dict[str, object],
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """投影 Godot eligible gettext ``msgid`` 与简体 output。"""
    evidence = _manifest_evidence(
        manifest, expected_family=GODOT_SOURCE_FAMILY)
    rows = []
    for item in pairs:
        if _binary(
                item.get("training_eligible"),
                label="Godot training_eligible") == 0:
            continue
        identity = item.get("source_identity")
        hans = item.get("zh_hans")
        hant = item.get("zh_hant")
        equal_length = _binary(
            item.get("equal_length"), label="Godot equal_length")
        if (not isinstance(identity, dict)
                or not isinstance(hans, dict)
                or not isinstance(hant, dict)
                or identity.get("msgid_plural") != ""
                or not isinstance(identity.get("msgid"), str)
                or not identity["msgid"]
                or not isinstance(hans.get("msgstr"), str)
                or not isinstance(hant.get("msgstr"), str)):
            raise BroadQaExternalDataError(
                "v7 neutral source projection Godot pair 非法")
        rows.append(_row(
            pair_id=item.get("pair_id"),
            source_family=GODOT_SOURCE_FAMILY,
            evidence=evidence,
            projection_kind=GETTEXT_SOURCE_PROJECTION,
            source_identity_kind="GETTEXT_MSGCTXT_MSGID_MSGID_PLURAL",
            source_identity=identity,
            neutral_surface=identity["msgid"],
            output_text=hans["msgstr"],
            variable_length=equal_length == 0,
            structured=bool(
                hans.get("structure_tokens")
                or hant.get("structure_tokens")),
            candidate_sentence_like=0,
        ))
    return tuple(rows)


def _libreoffice_rows(
        manifest: dict[str, object],
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """投影 LibreOffice eligible gettext ``msgid`` 与简体 output。"""
    evidence = _manifest_evidence(
        manifest, expected_family=LIBREOFFICE_SOURCE_FAMILY)
    rows = []
    for item in pairs:
        if _binary(
                item.get("training_eligible"),
                label="LibreOffice training_eligible") == 0:
            continue
        identity = item.get("source_identity")
        hans = item.get("zh_hans")
        equal_length = _binary(
            item.get("equal_length"), label="LibreOffice equal_length")
        if (not isinstance(identity, dict) or not isinstance(hans, dict)
                or identity.get("msgid_plural") != ""
                or not isinstance(identity.get("msgid"), str)
                or not identity["msgid"]
                or not isinstance(hans.get("msgstr"), str)):
            raise BroadQaExternalDataError(
                "v7 neutral source projection LibreOffice pair 非法")
        rows.append(_row(
            pair_id=item.get("pair_id"),
            source_family=LIBREOFFICE_SOURCE_FAMILY,
            evidence=evidence,
            projection_kind=GETTEXT_SOURCE_PROJECTION,
            source_identity_kind="GETTEXT_MSGCTXT_MSGID_MSGID_PLURAL",
            source_identity=identity,
            neutral_surface=identity["msgid"],
            output_text=hans["msgstr"],
            variable_length=equal_length == 0,
            structured=bool(
                item.get("zh_hans_structure_tokens")
                or item.get("zh_hant_structure_tokens")),
            candidate_sentence_like=0,
        ))
    return tuple(rows)


def _vscode_rows(
        manifest: dict[str, object],
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """投影 VS Code 完整 JSON path 的 leaf 候选与简体 output。"""
    evidence = _manifest_evidence(
        manifest, expected_family=VSCODE_SOURCE_FAMILY)
    rows = []
    for item in pairs:
        if _binary(
                item.get("training_eligible"),
                label="VS Code training_eligible") == 0:
            continue
        json_path = item.get("json_path")
        output = item.get("zh_hans_text")
        equal_length = _binary(
            item.get("equal_length"), label="VS Code equal_length")
        if (not isinstance(json_path, list) or not json_path
                or any(not isinstance(part, str) or not part
                       for part in json_path)
                or not isinstance(output, str) or not output
                or not isinstance(
                    item.get("translation_relative_path"), str)
                or not item["translation_relative_path"]
                or any(not isinstance(item.get(key), str)
                       or len(str(item.get(key))) != 64
                       for key in (
                           "zh_hans_file_id", "zh_hant_file_id"))):
            raise BroadQaExternalDataError(
                "v7 neutral source projection VS Code pair 非法")
        leaf = json_path[-1]
        if _ASCII_LETTER.search(leaf) is None:
            raise BroadQaExternalDataError(
                "v7 neutral source projection VS Code leaf 非候选")
        identity = {
            "json_path": json_path,
            "translation_relative_path": item.get(
                "translation_relative_path"),
            "zh_hans_file_id": item.get("zh_hans_file_id"),
            "zh_hant_file_id": item.get("zh_hant_file_id"),
        }
        rows.append(_row(
            pair_id=item.get("pair_id"),
            source_family=VSCODE_SOURCE_FAMILY,
            evidence=evidence,
            projection_kind=VSCODE_LEAF_PROJECTION,
            source_identity_kind=(
                "VSCODE_RELATIVE_PATH_COMPLETE_JSON_KEY_PATH"),
            source_identity=identity,
            neutral_surface=leaf,
            output_text=output,
            variable_length=equal_length == 0,
            structured=bool(
                item.get("zh_hans_structure_tokens")
                or item.get("zh_hant_structure_tokens")),
            candidate_sentence_like=int(
                " " in leaf or len(leaf) >= 8),
        ))
    return tuple(rows)


def _public_projection(row: dict[str, object]) -> dict[str, object]:
    """移除只允许在派生过程内存中存在的 neutral surface。"""
    return {key: value for key, value in row.items()
            if not key.startswith("_")}


def _family_record(
        *,
        source_family: str,
        manifest: dict[str, object],
        source_record_count: int,
        rows: tuple[dict[str, object], ...],
        projection_kind: str,
        availability: str,
        ) -> dict[str, object]:
    """形成一个 adapter projection availability 记录。"""
    evidence = _manifest_evidence(
        manifest, expected_family=source_family)
    unique_surfaces = {
        str(item["_neutral_surface"]) for item in rows}
    identity = {
        "adapter_projection_kind": projection_kind,
        "source_family": source_family,
        "source_pack_manifest_sha256": evidence[
            "source_pack_manifest_sha256"],
    }
    return {
        "adapter_projection_kind": projection_kind,
        "candidate_sentence_like_count": sum(
            int(item["candidate_sentence_like"]) for item in rows),
        "format_version": 1,
        "license_evidence_sha256": evidence[
            "license_evidence_sha256"],
        "license_id": evidence["license_id"],
        "neutral_surface_availability": availability,
        "projected_record_count": len(rows),
        "projection_family_id": _sha256(canonical_json_bytes(identity)),
        "record_kind": NEUTRAL_SOURCE_FAMILY_RECORD_KIND,
        "source_family": source_family,
        "source_pack_manifest_sha256": evidence[
            "source_pack_manifest_sha256"],
        "source_policy_scope": evidence["source_policy_scope"],
        "source_record_count": source_record_count,
        "unique_neutral_surface_count": len(unique_surfaces),
    }


def _support_records(
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """按 transient exact surface 派生至少双来源支持的承诺路由。"""
    by_surface: dict[str, dict[str, list[dict[str, object]]]] = {}
    commitment_surface: dict[tuple[str, int], str] = {}
    for family, rows in rows_by_family.items():
        for row in rows:
            surface = str(row["_neutral_surface"])
            key = (
                str(row["neutral_surface_sha256"]),
                int(row["neutral_surface_scalar_length"]),
            )
            previous = commitment_surface.setdefault(key, surface)
            if previous != surface:
                raise BroadQaExternalDataError(
                    "v7 neutral source projection surface commitment collision")
            by_surface.setdefault(surface, {}).setdefault(family, []).append(row)
    records = []
    for surface, support in by_surface.items():
        if len(support) < 2:
            continue
        family_support = []
        all_rows = []
        for family in sorted(support):
            family_rows = support[family]
            all_rows.extend(family_rows)
            outputs = sorted({str(item["output_sha256"])
                              for item in family_rows})
            family_support.append({
                "adapter_projection_kinds": sorted({
                    str(item["adapter_projection_kind"])
                    for item in family_rows}),
                "distinct_output_count": len(outputs),
                "license_evidence_sha256s": sorted({
                    str(item["license_evidence_sha256"])
                    for item in family_rows}),
                "license_ids": sorted({
                    str(item["license_id"]) for item in family_rows}),
                "output_sha256s": outputs,
                "record_count": len(family_rows),
                "source_family": family,
                "source_identity_sha256s": sorted({
                    str(item["source_identity_sha256"])
                    for item in family_rows}),
            })
        unique = all(item["distinct_output_count"] == 1
                     for item in family_support)
        unique_outputs = {
            str(item["output_sha256s"][0])
            for item in family_support if item["distinct_output_count"] == 1}
        consensus = unique and len(unique_outputs) == 1
        surface_commitment = _text_commitment(surface)
        identity = {
            "neutral_surface_sha256": surface_commitment["sha256"],
            "support_families": sorted(support),
        }
        records.append({
            "all_families_unique_output": int(unique),
            "all_variable": int(all(
                int(item["variable_length"]) == 1 for item in all_rows)),
            "any_structured": int(any(
                int(item["structured"]) == 1 for item in all_rows)),
            "any_variable": int(any(
                int(item["variable_length"]) == 1 for item in all_rows)),
            "consensus_output_sha256": (
                next(iter(unique_outputs)) if consensus else ""),
            "family_support": family_support,
            "format_version": 1,
            "neutral_surface_bytes": surface_commitment["bytes"],
            "neutral_surface_scalar_length": surface_commitment[
                "scalar_length"],
            "neutral_surface_sha256": surface_commitment["sha256"],
            "output_consensus": int(consensus),
            "record_kind": NEUTRAL_SOURCE_SUPPORT_RECORD_KIND,
            "support_families": sorted(support),
            "support_family_count": len(support),
            "support_id": _sha256(canonical_json_bytes(identity)),
        })
    records.sort(key=lambda item: str(item["support_id"]))
    return tuple(records)


def _pairwise_summary(
        left_family: str,
        right_family: str,
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, int]:
    """汇总一对 source family 的 exact neutral key 支持事实。"""
    maps = {}
    for family in (left_family, right_family):
        values: dict[str, list[dict[str, object]]] = {}
        for row in rows_by_family[family]:
            values.setdefault(str(row["_neutral_surface"]), []).append(row)
        maps[family] = values
    common = set(maps[left_family]).intersection(maps[right_family])
    unique = []
    same_comparisons = 0
    comparisons = 0
    for surface in common:
        left_rows = maps[left_family][surface]
        right_rows = maps[right_family][surface]
        for left in left_rows:
            for right in right_rows:
                comparisons += 1
                same_comparisons += int(
                    left["output_sha256"] == right["output_sha256"])
        left_outputs = {str(item["output_sha256"]) for item in left_rows}
        right_outputs = {str(item["output_sha256"]) for item in right_rows}
        if len(left_outputs) == 1 and len(right_outputs) == 1:
            unique.append(surface)
    consensus = []
    for surface in unique:
        left_output = maps[left_family][surface][0]["output_sha256"]
        right_output = maps[right_family][surface][0]["output_sha256"]
        if left_output == right_output:
            consensus.append(surface)
    return {
        "all_families_unique_output_count": len(unique),
        "common_exact_surface_count": len(common),
        "conflict_count": len(unique) - len(consensus),
        "consensus_all_variable_count": sum(
            all(int(item["variable_length"]) == 1
                for rows in (
                    maps[left_family][surface],
                    maps[right_family][surface])
                for item in rows)
            for surface in consensus),
        "consensus_any_structured_count": sum(
            any(int(item["structured"]) == 1
                for rows in (
                    maps[left_family][surface],
                    maps[right_family][surface])
                for item in rows)
            for surface in consensus),
        "consensus_any_variable_count": sum(
            any(int(item["variable_length"]) == 1
                for rows in (
                    maps[left_family][surface],
                    maps[right_family][surface])
                for item in rows)
            for surface in consensus),
        "consensus_count": len(consensus),
        "record_comparison_count": comparisons,
        "same_output_record_comparison_count": same_comparisons,
    }


def _three_family_summary(
        rows_by_family: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """汇总 Godot/LibreOffice/VS Code 三来源 exact key 事实。"""
    families = (
        GODOT_SOURCE_FAMILY,
        LIBREOFFICE_SOURCE_FAMILY,
        VSCODE_SOURCE_FAMILY,
    )
    maps = {}
    for family in families:
        values: dict[str, list[dict[str, object]]] = {}
        for row in rows_by_family[family]:
            values.setdefault(str(row["_neutral_surface"]), []).append(row)
        maps[family] = values
    common = set.intersection(*(set(maps[family]) for family in families))
    unique = [surface for surface in common if all(
        len({str(item["output_sha256"])
             for item in maps[family][surface]}) == 1
        for family in families)]
    consensus = [surface for surface in unique if len({
        str(maps[family][surface][0]["output_sha256"])
        for family in families}) == 1]
    return {
        "all_families_unique_output_count": len(unique),
        "common_exact_surface_count": len(common),
        "conflict_count": len(unique) - len(consensus),
        "consensus_all_variable_count": sum(
            all(int(item["variable_length"]) == 1
                for family in families
                for item in maps[family][surface])
            for surface in consensus),
        "consensus_any_structured_count": sum(
            any(int(item["structured"]) == 1
                for family in families
                for item in maps[family][surface])
            for surface in consensus),
        "consensus_any_variable_count": sum(
            any(int(item["variable_length"]) == 1
                for family in families
                for item in maps[family][surface])
            for surface in consensus),
        "consensus_count": len(consensus),
        "consensus_identity_set_sha256": _sha256(canonical_json_bytes(
            sorted(consensus))),
    }


def derive_neutral_upstream_source_projection_records(
        *,
        godot_manifest: dict[str, object],
        godot_pairs: tuple[dict[str, object], ...],
        libreoffice_manifest: dict[str, object],
        libreoffice_pairs: tuple[dict[str, object], ...],
        vscode_manifest: dict[str, object],
        vscode_pairs: tuple[dict[str, object], ...],
        thunderbird_manifest: dict[str, object],
        thunderbird_pairs: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生 family、projection、cross-support 与 section-80 census。"""
    rows_by_family = {
        GODOT_SOURCE_FAMILY: _godot_rows(godot_manifest, godot_pairs),
        LIBREOFFICE_SOURCE_FAMILY: _libreoffice_rows(
            libreoffice_manifest, libreoffice_pairs),
        VSCODE_SOURCE_FAMILY: _vscode_rows(vscode_manifest, vscode_pairs),
    }
    family_records = (
        _family_record(
            source_family=GODOT_SOURCE_FAMILY,
            manifest=godot_manifest,
            source_record_count=len(godot_pairs),
            rows=rows_by_family[GODOT_SOURCE_FAMILY],
            projection_kind=GETTEXT_SOURCE_PROJECTION,
            availability="NEUTRAL_SOURCE_SURFACE_AVAILABLE",
        ),
        _family_record(
            source_family=LIBREOFFICE_SOURCE_FAMILY,
            manifest=libreoffice_manifest,
            source_record_count=len(libreoffice_pairs),
            rows=rows_by_family[LIBREOFFICE_SOURCE_FAMILY],
            projection_kind=GETTEXT_SOURCE_PROJECTION,
            availability="NEUTRAL_SOURCE_SURFACE_AVAILABLE",
        ),
        _family_record(
            source_family=VSCODE_SOURCE_FAMILY,
            manifest=vscode_manifest,
            source_record_count=len(vscode_pairs),
            rows=rows_by_family[VSCODE_SOURCE_FAMILY],
            projection_kind=VSCODE_LEAF_PROJECTION,
            availability="NEUTRAL_KEY_OR_SOURCE_SURFACE_CANDIDATE",
        ),
        _family_record(
            source_family=THUNDERBIRD_SOURCE_FAMILY,
            manifest=thunderbird_manifest,
            source_record_count=len(thunderbird_pairs),
            rows=(),
            projection_kind=NEUTRAL_SURFACE_UNAVAILABLE,
            availability=NEUTRAL_SURFACE_UNAVAILABLE,
        ),
    )
    projections = tuple(sorted(
        (_public_projection(row)
         for family in _SOURCE_ORDER[:-1]
         for row in rows_by_family[family]),
        key=lambda item: str(item["projection_id"]),
    ))
    support_records = _support_records(rows_by_family)
    if (len({item["projection_id"] for item in projections})
            != len(projections)
            or len({item["support_id"] for item in support_records})
            != len(support_records)):
        raise BroadQaExternalDataError(
            "v7 neutral source projection output identity 重复")
    pairwise = {
        f"{left}__{right}": _pairwise_summary(
            left, right, rows_by_family)
        for left, right in _PAIRWISE_ORDER
    }
    summary = {
        "cross_family_support_record_count": len(support_records),
        "pairwise_exact_key_overlap": pairwise,
        "projection_class_counts": dict(sorted(Counter(
            str(item["adapter_projection_kind"])
            for item in family_records).items())),
        "projection_record_count": len(projections),
        "raw_input_output_or_source_surface_published": 0,
        "source_family_count": len(family_records),
        "source_family_projection_counts": {
            str(item["source_family"]): int(item["projected_record_count"])
            for item in family_records
        },
        "surface_commitment_collision_count": 0,
        "three_family_exact_key_overlap": _three_family_summary(
            rows_by_family),
        "vscode_ascii_letter_leaf_count": len(
            rows_by_family[VSCODE_SOURCE_FAMILY]),
        "vscode_sentence_like_leaf_count": sum(
            int(item["candidate_sentence_like"])
            for item in rows_by_family[VSCODE_SOURCE_FAMILY]),
    }
    return family_records, projections, support_records, summary


__all__ = [
    "GETTEXT_SOURCE_PROJECTION",
    "GODOT_SOURCE_FAMILY",
    "LIBREOFFICE_SOURCE_FAMILY",
    "NEUTRAL_SOURCE_FAMILY_RECORD_KIND",
    "NEUTRAL_SOURCE_PROJECTION_RECORD_KIND",
    "NEUTRAL_SOURCE_SUPPORT_RECORD_KIND",
    "NEUTRAL_SURFACE_UNAVAILABLE",
    "THUNDERBIRD_SOURCE_FAMILY",
    "VSCODE_LEAF_PROJECTION",
    "VSCODE_SOURCE_FAMILY",
    "derive_neutral_upstream_source_projection_records",
]
