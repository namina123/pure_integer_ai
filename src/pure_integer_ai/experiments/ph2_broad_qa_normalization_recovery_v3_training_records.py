"""从独立 UI 本地化来源派生 recovery-v3 TRAIN 记录。

本模块只做纯记录变换：整句 observation、确定性 edit core、带上下文的
phrase hunk 和冲突/authority 分组。它不读取路径、evaluation、reserve、
candidate 或 formal artifact，也不把候选组直接升级为可执行规则。
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import re

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
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_KIND = (
    "NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_V1")
NORMALIZATION_RECOVERY_V3_FRAGMENT_KIND = (
    "NORMALIZATION_RECOVERY_V3_PHRASE_FRAGMENT_V1")
NORMALIZATION_RECOVERY_V3_GROUP_KIND = (
    "NORMALIZATION_RECOVERY_V3_PHRASE_GROUP_V1")

THUNDERBIRD_SOURCE_FAMILY = "THUNDERBIRD_PROJECT"
GODOT_SOURCE_FAMILY = "GODOT_ENGINE_PROJECT"
THUNDERBIRD_SOURCE_POLICY_SCOPE = (
    "THUNDERBIRD_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1")
GODOT_SOURCE_POLICY_SCOPE = "GODOT_EDITOR_ZH_HANT_TO_ZH_HANS_V1"
RECOVERY_V3_TARGET_POLICY_SCOPE = (
    "ZH_CN_CROSS_PRODUCT_UI_LOCALIZATION_TRANSFER_V3")

FRAGMENT_KINDS = ("CONTEXT_HUNK", "EDIT_CORE")
GROUP_DISPOSITIONS = (
    "CONFLICT_DEFER",
    "CROSS_FAMILY_CONSENSUS_CANDIDATE",
    "DEFER_INSUFFICIENT_AUTHORITY",
    "MOZILLA_REPEATED_CONTEXT_CANDIDATE",
)
_HAN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def _sha256(payload: bytes) -> str:
    """返回规范记录 identity。"""
    return hashlib.sha256(payload).hexdigest()


def _pair_observation(
        *,
        source_family: str,
        source_policy_scope: str,
        license_id: str,
        source_pack_manifest_sha256: str,
        source_pair_id: str,
        input_text: str,
        output_text: str,
        structure_tokens: tuple[str, ...],
        source_commitment: dict[str, object],
        ) -> dict[str, object]:
    """构造一条来源、许可与表面结构分账的整句 observation。"""
    identity = {
        "source_pack_manifest_sha256": source_pack_manifest_sha256,
        "source_pair_id": source_pair_id,
        "source_policy_scope": source_policy_scope,
    }
    return {
        "contains_han": int(bool(_HAN.search(input_text + output_text))),
        "equal_length": int(len(input_text) == len(output_text)),
        "format_version": 1,
        "identity_preservation": int(input_text == output_text),
        "input_text": input_text,
        "license_id": license_id,
        "observation_id": _sha256(canonical_json_bytes({
            **identity,
            "record_kind": NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_KIND,
        })),
        "output_text": output_text,
        "record_kind": NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_KIND,
        "source_commitment": source_commitment,
        "source_family": source_family,
        "source_pack_manifest_sha256": source_pack_manifest_sha256,
        "source_pair_id": source_pair_id,
        "source_policy_scope": source_policy_scope,
        "structure_tokens": list(structure_tokens),
    }


def derive_normalization_recovery_v3_pair_observations(
        *,
        thunderbird_manifest_sha256: str,
        thunderbird_pairs: tuple[dict[str, object], ...],
        godot_manifest_sha256: str,
        godot_pairs: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从两套 source pack 机械派生 TRAIN 整句 observations。"""
    values = []
    for pair in thunderbird_pairs:
        if pair.get("record_kind") != THUNDERBIRD_PATTERN_PAIR_RECORD_KIND:
            raise BroadQaExternalDataError("v3 Thunderbird pair kind 漂移")
        zh_tw = pair.get("zh_tw")
        zh_cn = pair.get("zh_cn")
        if not isinstance(zh_tw, dict) or not isinstance(zh_cn, dict):
            raise BroadQaExternalDataError("v3 Thunderbird pair locale 缺失")
        input_text = zh_tw.get("surface_text")
        output_text = zh_cn.get("surface_text")
        if (pair.get("plain_pair_eligible") != 1
                or not isinstance(input_text, str)
                or not isinstance(output_text, str)
                or not _HAN.search(input_text + output_text)):
            continue
        values.append(_pair_observation(
            source_family=THUNDERBIRD_SOURCE_FAMILY,
            source_policy_scope=THUNDERBIRD_SOURCE_POLICY_SCOPE,
            license_id=THUNDERBIRD_L10N_LICENSE_ID,
            source_pack_manifest_sha256=thunderbird_manifest_sha256,
            source_pair_id=str(pair["pair_id"]),
            input_text=input_text,
            output_text=output_text,
            structure_tokens=(),
            source_commitment={
                "attribute_id": pair["attribute_id"],
                "entry_kind": pair["entry_kind"],
                "message_id": pair["message_id"],
                "relative_path": pair["relative_path"],
                "zh_cn_file_sha256": zh_cn["file_sha256"],
                "zh_cn_source_slice_sha256": zh_cn["source_slice_sha256"],
                "zh_tw_file_sha256": zh_tw["file_sha256"],
                "zh_tw_source_slice_sha256": zh_tw["source_slice_sha256"],
            },
        ))
    for pair in godot_pairs:
        if pair.get("record_kind") != GODOT_PO_PAIR_RECORD_KIND:
            raise BroadQaExternalDataError("v3 Godot pair kind 漂移")
        zh_hant = pair.get("zh_hant")
        zh_hans = pair.get("zh_hans")
        if not isinstance(zh_hant, dict) or not isinstance(zh_hans, dict):
            raise BroadQaExternalDataError("v3 Godot pair locale 缺失")
        if pair.get("training_eligible") != 1:
            continue
        input_text = zh_hant.get("msgstr")
        output_text = zh_hans.get("msgstr")
        tokens = zh_hant.get("structure_tokens")
        if (not isinstance(input_text, str) or not isinstance(output_text, str)
                or not isinstance(tokens, list)
                or any(not isinstance(item, str) for item in tokens)):
            raise BroadQaExternalDataError("v3 Godot pair surface 漂移")
        values.append(_pair_observation(
            source_family=GODOT_SOURCE_FAMILY,
            source_policy_scope=GODOT_SOURCE_POLICY_SCOPE,
            license_id=GODOT_LICENSE_ID,
            source_pack_manifest_sha256=godot_manifest_sha256,
            source_pair_id=str(pair["pair_id"]),
            input_text=input_text,
            output_text=output_text,
            structure_tokens=tuple(tokens),
            source_commitment={
                "source_identity": pair["source_identity"],
                "zh_hans_entry_linenum": zh_hans["entry_linenum"],
                "zh_hans_entry_semantic_sha256": (
                    zh_hans["entry_semantic_sha256"]),
                "zh_hant_entry_linenum": zh_hant["entry_linenum"],
                "zh_hant_entry_semantic_sha256": (
                    zh_hant["entry_semantic_sha256"]),
            },
        ))
    result = tuple(sorted(values, key=lambda item: (
        item["source_family"], item["source_pair_id"])))
    if (not result or len({item["observation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError("v3 pair observation identity 漂移")
    return result


def _fragment(
        observation: dict[str, object],
        *,
        fragment_kind: str,
        opcode_ordinal: int,
        input_start: int,
        input_end: int,
        output_start: int,
        output_end: int,
        ) -> dict[str, object] | None:
    """从一个确定 span 构造非恒等 phrase fragment。"""
    input_text = str(observation["input_text"])[input_start:input_end]
    output_text = str(observation["output_text"])[output_start:output_end]
    if (not input_text or input_text == output_text
            or len(input_text) > 40 or len(output_text) > 80
            or not _HAN.search(input_text + output_text)):
        return None
    identity = {
        "fragment_kind": fragment_kind,
        "input_end": input_end,
        "input_start": input_start,
        "observation_id": observation["observation_id"],
        "opcode_ordinal": opcode_ordinal,
        "output_end": output_end,
        "output_start": output_start,
    }
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "format_version": 1,
        "fragment_id": _sha256(canonical_json_bytes({
            **identity,
            "record_kind": NORMALIZATION_RECOVERY_V3_FRAGMENT_KIND,
        })),
        "fragment_kind": fragment_kind,
        "input_end": input_end,
        "input_start": input_start,
        "input_text": input_text,
        "license_id": observation["license_id"],
        "observation_id": observation["observation_id"],
        "opcode_ordinal": opcode_ordinal,
        "output_end": output_end,
        "output_start": output_start,
        "output_text": output_text,
        "record_kind": NORMALIZATION_RECOVERY_V3_FRAGMENT_KIND,
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
    }


def derive_normalization_recovery_v3_fragments(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """以固定 SequenceMatcher 参数派生 edit core 与一字符 context hunk。"""
    values = []
    for observation in observations:
        input_text = str(observation["input_text"])
        output_text = str(observation["output_text"])
        if (input_text == output_text or observation["structure_tokens"]
                or len(input_text) > 320 or len(output_text) > 320):
            continue
        matcher = SequenceMatcher(None, input_text, output_text, autojunk=False)
        for ordinal, (tag, i1, i2, j1, j2) in enumerate(
                matcher.get_opcodes()):
            if tag == "equal":
                continue
            if i1 < i2:
                core = _fragment(
                    observation,
                    fragment_kind="EDIT_CORE",
                    opcode_ordinal=ordinal,
                    input_start=i1,
                    input_end=i2,
                    output_start=j1,
                    output_end=j2,
                )
                if core is not None:
                    values.append(core)
            hunk = _fragment(
                observation,
                fragment_kind="CONTEXT_HUNK",
                opcode_ordinal=ordinal,
                input_start=max(0, i1 - 1),
                input_end=min(len(input_text), i2 + 1),
                output_start=max(0, j1 - 1),
                output_end=min(len(output_text), j2 + 1),
            )
            if hunk is not None:
                values.append(hunk)
    result = tuple(sorted(values, key=lambda item: (
        item["fragment_kind"], item["input_text"], item["output_text"],
        item["fragment_id"])))
    if (not result or len({item["fragment_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError("v3 phrase fragment identity 漂移")
    return result


def derive_normalization_recovery_v3_groups(
        fragments: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 kind/input 分组，保留全部输出冲突与来源支持。"""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for fragment in fragments:
        kind = str(fragment["fragment_kind"])
        if kind not in FRAGMENT_KINDS:
            raise BroadQaExternalDataError("v3 fragment kind 非法")
        grouped[(kind, str(fragment["input_text"]))].append(fragment)
    values = []
    for (kind, input_text), items in sorted(grouped.items()):
        outputs: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            outputs[str(item["output_text"])].append(item)
        variants = []
        all_families = set()
        for output_text, supports in sorted(outputs.items()):
            families = sorted({str(item["source_family"]) for item in supports})
            policies = sorted({
                str(item["source_policy_scope"]) for item in supports})
            all_families.update(families)
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
              and sole["source_families"] == [THUNDERBIRD_SOURCE_FAMILY]):
            disposition = "MOZILLA_REPEATED_CONTEXT_CANDIDATE"
        else:
            disposition = "DEFER_INSUFFICIENT_AUTHORITY"
        identity = {"fragment_kind": kind, "input_text": input_text}
        values.append({
            "disposition": disposition,
            "format_version": 1,
            "fragment_kind": kind,
            "group_id": _sha256(canonical_json_bytes({
                **identity,
                "record_kind": NORMALIZATION_RECOVERY_V3_GROUP_KIND,
            })),
            "input_text": input_text,
            "negative_evidence_required_before_execution": int(
                disposition.endswith("CANDIDATE")),
            "output_variants": variants,
            "record_kind": NORMALIZATION_RECOVERY_V3_GROUP_KIND,
            "source_families": sorted(all_families),
            "target_policy_scope": (
                RECOVERY_V3_TARGET_POLICY_SCOPE
                if disposition.endswith("CANDIDATE") else ""),
            "unscoped_execution_allowed": 0,
        })
    result = tuple(values)
    if (not result or len({item["group_id"] for item in result})
            != len(result)
            or any(item["disposition"] not in GROUP_DISPOSITIONS
                   for item in result)):
        raise BroadQaExternalDataError("v3 phrase group identity 漂移")
    return result


__all__ = [
    "FRAGMENT_KINDS",
    "GODOT_SOURCE_FAMILY",
    "GODOT_SOURCE_POLICY_SCOPE",
    "GROUP_DISPOSITIONS",
    "NORMALIZATION_RECOVERY_V3_FRAGMENT_KIND",
    "NORMALIZATION_RECOVERY_V3_GROUP_KIND",
    "NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_KIND",
    "RECOVERY_V3_TARGET_POLICY_SCOPE",
    "THUNDERBIRD_SOURCE_FAMILY",
    "THUNDERBIRD_SOURCE_POLICY_SCOPE",
    "derive_normalization_recovery_v3_fragments",
    "derive_normalization_recovery_v3_groups",
    "derive_normalization_recovery_v3_pair_observations",
]
