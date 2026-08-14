"""从三套独立 UI 本地化来源派生 recovery-v4 TRAIN records。

v4 复用冻结 v3 observation/fragment schema 与确定性 SequenceMatcher 算法，
只新增 VS Code source adapter 和作用域明确的 group record。跨产品 target 候选
必须由至少两个独立 UI source family 支持；单来源重复 context 只能成为
source-scoped candidate，不能获得 target policy scope。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_godot_source_pack import (
    GODOT_LICENSE_ID,
    GODOT_PO_PAIR_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack import (
    THUNDERBIRD_L10N_LICENSE_ID,
    THUNDERBIRD_PATTERN_PAIR_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    FRAGMENT_KINDS,
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
    derive_normalization_recovery_v3_fragments,
    derive_normalization_recovery_v3_pair_observations,
    normalization_recovery_v3_pair_observation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_LICENSE_ID,
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_records import (
    VSCODE_TRANSLATION_PAIR_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V4_GROUP_KIND = (
    "NORMALIZATION_RECOVERY_V4_PHRASE_GROUP_V1")
RECOVERY_V4_TARGET_POLICY_SCOPE = (
    "ZH_CN_CROSS_PRODUCT_UI_LOCALIZATION_TRANSFER_V4")

V4_SOURCE_FAMILIES = (
    GODOT_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)
V4_SOURCE_POLICY_SCOPES = (
    GODOT_SOURCE_POLICY_SCOPE,
    VSCODE_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
)
V4_GROUP_DISPOSITIONS = (
    "CONFLICT_DEFER",
    "CROSS_FAMILY_CONSENSUS_CANDIDATE",
    "DEFER_INSUFFICIENT_AUTHORITY",
    "SOURCE_SCOPED_CANDIDATE",
)


def _sha256(payload: bytes) -> str:
    """返回规范 group identity。"""
    return hashlib.sha256(payload).hexdigest()


def _vscode_observations(
        *,
        manifest_sha256: str,
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """把严格 VS Code pair 适配到冻结 v3 observation schema。"""
    values = []
    for pair in pairs:
        if pair.get("record_kind") != VSCODE_TRANSLATION_PAIR_RECORD_KIND:
            raise BroadQaExternalDataError("v4 VS Code pair kind 漂移")
        if pair.get("training_eligible") != 1:
            continue
        input_text = pair.get("zh_hant_text")
        output_text = pair.get("zh_hans_text")
        input_tokens = pair.get("zh_hant_structure_tokens")
        output_tokens = pair.get("zh_hans_structure_tokens")
        if (not isinstance(input_text, str)
                or not isinstance(output_text, str)
                or not isinstance(input_tokens, list)
                or not isinstance(output_tokens, list)
                or any(not isinstance(item, str)
                       for item in input_tokens + output_tokens)
                or input_tokens != output_tokens
                or pair.get("structure_equal") != 1
                or pair.get("contains_han_both") != 1
                or pair.get("within_scalar_limit") != 1):
            raise BroadQaExternalDataError(
                "v4 VS Code eligible pair surface/structure 漂移")
        values.append(normalization_recovery_v3_pair_observation(
            source_family=VSCODE_SOURCE_FAMILY,
            source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
            license_id=VSCODE_LICENSE_ID,
            source_pack_manifest_sha256=manifest_sha256,
            source_pair_id=str(pair["pair_id"]),
            input_text=input_text,
            output_text=output_text,
            structure_tokens=tuple(input_tokens),
            source_commitment={
                "json_path_sha256": pair["json_path_sha256"],
                "translation_relative_path": (
                    pair["translation_relative_path"]),
                "zh_hans_file_id": pair["zh_hans_file_id"],
                "zh_hans_text_sha256": pair["zh_hans_text_sha256"],
                "zh_hant_file_id": pair["zh_hant_file_id"],
                "zh_hant_text_sha256": pair["zh_hant_text_sha256"],
            },
        ))
    result = tuple(sorted(values, key=lambda item: str(item["source_pair_id"])))
    if (not result or len({item["observation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError(
            "v4 VS Code observation identity 漂移")
    return result


def derive_normalization_recovery_v4_pair_observations(
        *,
        thunderbird_manifest_sha256: str,
        thunderbird_pairs: tuple[dict[str, object], ...],
        godot_manifest_sha256: str,
        godot_pairs: tuple[dict[str, object], ...],
        vscode_manifest_sha256: str,
        vscode_pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从三套 source pack 派生完整 TRAIN observations。"""
    legacy = derive_normalization_recovery_v3_pair_observations(
        thunderbird_manifest_sha256=thunderbird_manifest_sha256,
        thunderbird_pairs=thunderbird_pairs,
        godot_manifest_sha256=godot_manifest_sha256,
        godot_pairs=godot_pairs,
    )
    vscode = _vscode_observations(
        manifest_sha256=vscode_manifest_sha256,
        pairs=vscode_pairs,
    )
    result = tuple(sorted(legacy + vscode, key=lambda item: (
        str(item["source_family"]), str(item["source_pair_id"]))))
    if (len({item["observation_id"] for item in result}) != len(result)
            or {str(item["source_family"]) for item in result}
            != set(V4_SOURCE_FAMILIES)):
        raise BroadQaExternalDataError("v4 observation roster 漂移")
    return result


def derive_normalization_recovery_v4_fragments(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """复用冻结 v3 确定性 fragment 算法。"""
    return derive_normalization_recovery_v3_fragments(observations)


def derive_normalization_recovery_v4_groups(
        fragments: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """形成跨来源 target、单来源 scoped、冲突与 defer 四类 group。"""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for fragment in fragments:
        kind = str(fragment["fragment_kind"])
        source_family = str(fragment["source_family"])
        source_policy = str(fragment["source_policy_scope"])
        if (kind not in FRAGMENT_KINDS
                or source_family not in V4_SOURCE_FAMILIES
                or source_policy not in V4_SOURCE_POLICY_SCOPES):
            raise BroadQaExternalDataError(
                "v4 fragment kind/source scope 非法")
        grouped[(kind, str(fragment["input_text"]))].append(fragment)
    values = []
    for (kind, input_text), items in sorted(grouped.items()):
        outputs: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            outputs[str(item["output_text"])].append(item)
        variants = []
        all_families = set()
        all_policies = set()
        for output_text, supports in sorted(outputs.items()):
            families = sorted({str(item["source_family"])
                               for item in supports})
            policies = sorted({str(item["source_policy_scope"])
                               for item in supports})
            all_families.update(families)
            all_policies.update(policies)
            variants.append({
                "fragment_ids": sorted(str(item["fragment_id"])
                                       for item in supports),
                "output_text": output_text,
                "source_families": families,
                "source_policy_scopes": policies,
                "support_count": len(supports),
            })
        sole = variants[0] if len(variants) == 1 else None
        if len(variants) > 1:
            disposition = "CONFLICT_DEFER"
        elif (len(input_text) >= 2 and sole is not None
              and len(sole["source_families"]) >= 2):
            disposition = "CROSS_FAMILY_CONSENSUS_CANDIDATE"
        elif (kind == "CONTEXT_HUNK" and len(input_text) >= 2
              and sole is not None and sole["support_count"] >= 2
              and len(sole["source_families"]) == 1):
            disposition = "SOURCE_SCOPED_CANDIDATE"
        else:
            disposition = "DEFER_INSUFFICIENT_AUTHORITY"
        identity = {"fragment_kind": kind, "input_text": input_text}
        values.append({
            "candidate_scope_kind": (
                "TARGET_CROSS_FAMILY"
                if disposition == "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                else "SOURCE_ONLY"
                if disposition == "SOURCE_SCOPED_CANDIDATE"
                else "NONE"),
            "disposition": disposition,
            "format_version": 1,
            "fragment_kind": kind,
            "group_id": _sha256(canonical_json_bytes({
                **identity,
                "record_kind": NORMALIZATION_RECOVERY_V4_GROUP_KIND,
            })),
            "input_text": input_text,
            "negative_evidence_required_before_execution": int(
                disposition.endswith("CANDIDATE")),
            "output_variants": variants,
            "record_kind": NORMALIZATION_RECOVERY_V4_GROUP_KIND,
            "source_families": sorted(all_families),
            "source_policy_scopes": sorted(all_policies),
            "target_policy_scope": (
                RECOVERY_V4_TARGET_POLICY_SCOPE
                if disposition == "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                else ""),
            "unscoped_execution_allowed": 0,
        })
    result = tuple(values)
    if (not result or len({item["group_id"] for item in result})
            != len(result)
            or any(item["disposition"] not in V4_GROUP_DISPOSITIONS
                   for item in result)):
        raise BroadQaExternalDataError("v4 phrase group identity 漂移")
    return result


__all__ = [
    "NORMALIZATION_RECOVERY_V4_GROUP_KIND",
    "RECOVERY_V4_TARGET_POLICY_SCOPE",
    "V4_GROUP_DISPOSITIONS",
    "V4_SOURCE_FAMILIES",
    "V4_SOURCE_POLICY_SCOPES",
    "derive_normalization_recovery_v4_fragments",
    "derive_normalization_recovery_v4_groups",
    "derive_normalization_recovery_v4_pair_observations",
]
