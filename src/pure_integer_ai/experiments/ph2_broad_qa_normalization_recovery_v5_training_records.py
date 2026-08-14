"""从四套独立 UI 本地化来源派生 recovery-v5 TRAIN records。

v5 保留冻结的 edit-core/context-hunk 算法，并新增完整输入 `WHOLE_INPUT`
fragment。局部与等长整句可由两 source family 共识形成 target candidate；
变长整句要求三 family，或两个 family 各自至少两条独立 observation 支持。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    FRAGMENT_KINDS,
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
    derive_normalization_recovery_v3_fragments,
    normalization_recovery_v3_pair_observation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    derive_normalization_recovery_v4_pair_observations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_pack import (
    LIBREOFFICE_LICENSE_ID,
    LIBREOFFICE_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_records import (
    LIBREOFFICE_PO_PAIR_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V5_WHOLE_INPUT_FRAGMENT_KIND = (
    "NORMALIZATION_RECOVERY_V5_WHOLE_INPUT_FRAGMENT_V1")
NORMALIZATION_RECOVERY_V5_GROUP_KIND = (
    "NORMALIZATION_RECOVERY_V5_PHRASE_GROUP_V1")
RECOVERY_V5_TARGET_POLICY_SCOPE = (
    "ZH_CN_CROSS_PRODUCT_UI_LOCALIZATION_TRANSFER_V5")

V5_FRAGMENT_KINDS = (*FRAGMENT_KINDS, "WHOLE_INPUT")
V5_SOURCE_FAMILIES = (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
)
V5_SOURCE_POLICY_BY_FAMILY = {
    GODOT_SOURCE_FAMILY: GODOT_SOURCE_POLICY_SCOPE,
    LIBREOFFICE_SOURCE_FAMILY: LIBREOFFICE_SOURCE_POLICY_SCOPE,
    VSCODE_SOURCE_FAMILY: VSCODE_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY: THUNDERBIRD_SOURCE_POLICY_SCOPE,
}
V5_GROUP_DISPOSITIONS = (
    "CONFLICT_DEFER",
    "CROSS_FAMILY_CONSENSUS_CANDIDATE",
    "DEFER_INSUFFICIENT_AUTHORITY",
    "SOURCE_SCOPED_CANDIDATE",
)


def _sha256(payload: bytes) -> str:
    """返回规范 fragment/group identity。"""
    return hashlib.sha256(payload).hexdigest()


def _libreoffice_observations(
        *,
        manifest_sha256: str,
        pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """把严格 LibreOffice pair 适配到冻结 observation schema。"""
    values = []
    for pair in pairs:
        if pair.get("record_kind") != LIBREOFFICE_PO_PAIR_RECORD_KIND:
            raise BroadQaExternalDataError("v5 LibreOffice pair kind 漂移")
        if pair.get("training_eligible") != 1:
            continue
        zh_hant = pair.get("zh_hant")
        zh_hans = pair.get("zh_hans")
        input_tokens = pair.get("zh_hant_structure_tokens")
        output_tokens = pair.get("zh_hans_structure_tokens")
        if (not isinstance(zh_hant, dict) or not isinstance(zh_hans, dict)
                or not isinstance(input_tokens, list)
                or not isinstance(output_tokens, list)
                or any(not isinstance(item, str)
                       for item in input_tokens + output_tokens)
                or input_tokens != output_tokens
                or pair.get("structure_equal") != 1
                or pair.get("within_scalar_limit") != 1):
            raise BroadQaExternalDataError(
                "v5 LibreOffice eligible pair structure 漂移")
        input_text = zh_hant.get("msgstr")
        output_text = zh_hans.get("msgstr")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text):
            raise BroadQaExternalDataError(
                "v5 LibreOffice eligible pair surface 漂移")
        values.append(normalization_recovery_v3_pair_observation(
            source_family=LIBREOFFICE_SOURCE_FAMILY,
            source_policy_scope=LIBREOFFICE_SOURCE_POLICY_SCOPE,
            license_id=LIBREOFFICE_LICENSE_ID,
            source_pack_manifest_sha256=manifest_sha256,
            source_pair_id=str(pair["pair_id"]),
            input_text=input_text,
            output_text=output_text,
            structure_tokens=tuple(input_tokens),
            source_commitment={
                "source_identity_sha256": pair["source_identity_sha256"],
                "zh_hans_entry_semantic_sha256": (
                    zh_hans["entry_semantic_sha256"]),
                "zh_hans_source_file_id": zh_hans["source_file_id"],
                "zh_hant_entry_semantic_sha256": (
                    zh_hant["entry_semantic_sha256"]),
                "zh_hant_source_file_id": zh_hant["source_file_id"],
            },
        ))
    result = tuple(sorted(values, key=lambda item: str(
        item["source_pair_id"])))
    if (not result or len({item["observation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError(
            "v5 LibreOffice observation identity 漂移")
    return result


def derive_normalization_recovery_v5_pair_observations(
        *,
        thunderbird_manifest_sha256: str,
        thunderbird_pairs: tuple[dict[str, object], ...],
        godot_manifest_sha256: str,
        godot_pairs: tuple[dict[str, object], ...],
        vscode_manifest_sha256: str,
        vscode_pairs: tuple[dict[str, object], ...],
        libreoffice_manifest_sha256: str,
        libreoffice_pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从四套 source pack 派生完整 TRAIN observations。"""
    legacy = derive_normalization_recovery_v4_pair_observations(
        thunderbird_manifest_sha256=thunderbird_manifest_sha256,
        thunderbird_pairs=thunderbird_pairs,
        godot_manifest_sha256=godot_manifest_sha256,
        godot_pairs=godot_pairs,
        vscode_manifest_sha256=vscode_manifest_sha256,
        vscode_pairs=vscode_pairs,
    )
    libreoffice = _libreoffice_observations(
        manifest_sha256=libreoffice_manifest_sha256,
        pairs=libreoffice_pairs,
    )
    result = tuple(sorted(legacy + libreoffice, key=lambda item: (
        str(item["source_family"]), str(item["source_pair_id"]))))
    if (len({item["observation_id"] for item in result}) != len(result)
            or {str(item["source_family"]) for item in result}
            != set(V5_SOURCE_FAMILIES)):
        raise BroadQaExternalDataError("v5 observation roster 漂移")
    return result


def _whole_input_fragment(
        observation: dict[str, object],
        ) -> dict[str, object] | None:
    """从一条非恒等 observation 形成完整输入 phrase fragment。"""
    input_text = str(observation["input_text"])
    output_text = str(observation["output_text"])
    if (input_text == output_text or len(input_text) < 2
            or len(input_text) > 320 or len(output_text) > 320
            or observation["contains_han"] != 1):
        return None
    identity = {
        "fragment_kind": "WHOLE_INPUT",
        "observation_id": observation["observation_id"],
    }
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "format_version": 1,
        "fragment_id": _sha256(canonical_json_bytes({
            **identity,
            "record_kind": NORMALIZATION_RECOVERY_V5_WHOLE_INPUT_FRAGMENT_KIND,
        })),
        "fragment_kind": "WHOLE_INPUT",
        "input_end": len(input_text),
        "input_start": 0,
        "input_text": input_text,
        "license_id": observation["license_id"],
        "observation_id": observation["observation_id"],
        "opcode_ordinal": 0,
        "output_end": len(output_text),
        "output_start": 0,
        "output_text": output_text,
        "record_kind": NORMALIZATION_RECOVERY_V5_WHOLE_INPUT_FRAGMENT_KIND,
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
        "structure_tokens": observation["structure_tokens"],
    }


def derive_normalization_recovery_v5_fragments(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """派生冻结局部 fragments，并追加完整输入 fragments。"""
    legacy = derive_normalization_recovery_v3_fragments(observations)
    whole = tuple(
        fragment for fragment in (
            _whole_input_fragment(item) for item in observations)
        if fragment is not None)
    result = tuple(sorted(legacy + whole, key=lambda item: (
        str(item["fragment_kind"]), str(item["input_text"]),
        str(item["output_text"]), str(item["fragment_id"]))))
    if (not result or len({item["fragment_id"] for item in result})
            != len(result)
            or {str(item["fragment_kind"]) for item in result}
            != set(V5_FRAGMENT_KINDS)):
        raise BroadQaExternalDataError("v5 fragment roster 漂移")
    return result


def _variant(
        output_text: str,
        supports: list[dict[str, object]],
        ) -> dict[str, object]:
    """汇总一个输出 variant 的独立来源和长度事实。"""
    family_counts = Counter(str(item["source_family"]) for item in supports)
    equal_lengths = {int(item["equal_length"]) for item in supports}
    if len(equal_lengths) != 1:
        raise BroadQaExternalDataError("v5 group variant length fact 冲突")
    return {
        "equal_length": equal_lengths.pop(),
        "fragment_ids": sorted(str(item["fragment_id"]) for item in supports),
        "output_text": output_text,
        "source_families": sorted(family_counts),
        "source_family_support_counts": dict(sorted(family_counts.items())),
        "source_policy_scopes": sorted({
            str(item["source_policy_scope"]) for item in supports}),
        "support_count": len(supports),
    }


def _candidate_disposition(
        *,
        fragment_kind: str,
        input_text: str,
        variants: list[dict[str, object]],
        ) -> tuple[str, str, int]:
    """按 fragment class 返回 disposition、authority basis 与最小 family 数。"""
    if len(variants) > 1:
        return "CONFLICT_DEFER", "OUTPUT_CONFLICT", 0
    sole = variants[0]
    families = sole["source_families"]
    family_counts = sole["source_family_support_counts"]
    if len(input_text) < 2:
        return "DEFER_INSUFFICIENT_AUTHORITY", "INPUT_TOO_SHORT", 0
    if fragment_kind == "WHOLE_INPUT":
        if sole["equal_length"] == 1 and len(families) >= 2:
            return (
                "CROSS_FAMILY_CONSENSUS_CANDIDATE",
                "EQUAL_LENGTH_WHOLE_INPUT_TWO_FAMILY_CONSENSUS",
                2,
            )
        replicated_two_family = (
            len(families) >= 2
            and all(int(family_counts[family]) >= 2 for family in families))
        if sole["equal_length"] == 0 and (
                len(families) >= 3 or replicated_two_family):
            return (
                "CROSS_FAMILY_CONSENSUS_CANDIDATE",
                "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS",
                3 if len(families) >= 3 else 2,
            )
        if len(families) == 1 and sole["support_count"] >= 2:
            return (
                "SOURCE_SCOPED_CANDIDATE",
                "REPEATED_SOURCE_WHOLE_INPUT",
                1,
            )
        return (
            "DEFER_INSUFFICIENT_AUTHORITY",
            "WHOLE_INPUT_AUTHORITY_INSUFFICIENT",
            3 if sole["equal_length"] == 0 else 2,
        )
    if len(families) >= 2:
        return (
            "CROSS_FAMILY_CONSENSUS_CANDIDATE",
            "LOCAL_OR_CONTEXT_TWO_FAMILY_CONSENSUS",
            2,
        )
    if (fragment_kind == "CONTEXT_HUNK"
            and sole["support_count"] >= 2):
        return (
            "SOURCE_SCOPED_CANDIDATE",
            "REPEATED_SOURCE_CONTEXT",
            1,
        )
    return (
        "DEFER_INSUFFICIENT_AUTHORITY",
        "LOCAL_OR_CONTEXT_AUTHORITY_INSUFFICIENT",
        2,
    )


def derive_normalization_recovery_v5_groups(
        fragments: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """形成分 fragment class 的 target、source、conflict 与 defer group。"""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for fragment in fragments:
        kind = str(fragment["fragment_kind"])
        family = str(fragment["source_family"])
        policy = str(fragment["source_policy_scope"])
        if (kind not in V5_FRAGMENT_KINDS
                or family not in V5_SOURCE_POLICY_BY_FAMILY
                or policy != V5_SOURCE_POLICY_BY_FAMILY[family]):
            raise BroadQaExternalDataError(
                "v5 fragment kind/source scope 非法")
        grouped[(kind, str(fragment["input_text"]))].append(fragment)
    values = []
    for (kind, input_text), items in sorted(grouped.items()):
        outputs: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            outputs[str(item["output_text"])].append(item)
        variants = [_variant(output_text, supports)
                    for output_text, supports in sorted(outputs.items())]
        disposition, authority_basis, required_families = (
            _candidate_disposition(
                fragment_kind=kind,
                input_text=input_text,
                variants=variants,
            ))
        all_families = sorted({
            family for variant in variants
            for family in variant["source_families"]})
        all_policies = sorted({
            policy for variant in variants
            for policy in variant["source_policy_scopes"]})
        identity = {"fragment_kind": kind, "input_text": input_text}
        values.append({
            "authority_basis": authority_basis,
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
                "record_kind": NORMALIZATION_RECOVERY_V5_GROUP_KIND,
            })),
            "input_text": input_text,
            "negative_evidence_required_before_execution": int(
                disposition.endswith("CANDIDATE")),
            "observed_distinct_source_family_count": len(all_families),
            "output_variants": variants,
            "record_kind": NORMALIZATION_RECOVERY_V5_GROUP_KIND,
            "required_distinct_source_family_count": required_families,
            "source_families": all_families,
            "source_policy_scopes": all_policies,
            "target_policy_scope": (
                RECOVERY_V5_TARGET_POLICY_SCOPE
                if disposition == "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                else ""),
            "unscoped_execution_allowed": 0,
            "variable_length": int(any(
                variant["equal_length"] == 0 for variant in variants)),
        })
    result = tuple(values)
    if (not result or len({item["group_id"] for item in result})
            != len(result)
            or any(item["disposition"] not in V5_GROUP_DISPOSITIONS
                   for item in result)):
        raise BroadQaExternalDataError("v5 phrase group identity 漂移")
    return result


__all__ = [
    "NORMALIZATION_RECOVERY_V5_GROUP_KIND",
    "NORMALIZATION_RECOVERY_V5_WHOLE_INPUT_FRAGMENT_KIND",
    "RECOVERY_V5_TARGET_POLICY_SCOPE",
    "V5_FRAGMENT_KINDS",
    "V5_GROUP_DISPOSITIONS",
    "V5_SOURCE_FAMILIES",
    "V5_SOURCE_POLICY_BY_FAMILY",
    "derive_normalization_recovery_v5_fragments",
    "derive_normalization_recovery_v5_groups",
    "derive_normalization_recovery_v5_pair_observations",
]
