"""结算 recovery-v4 TRAIN family 并派生完整 learner 输出。

本模块负责 family 合并、conflict/defer、summary 与 checkpoint 计数；候选作用域、
Evidence、occurrence、rule/defeater/index 由独立 learning-evidence 模块负责。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_phrase_learning import (
    normalization_phrase_output_payloads,
    normalization_phrase_prefix_output_counts,
    require_normalization_phrase_work_alignment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_contract import (
    NORMALIZATION_RECOVERY_V4_CONFLICT_KIND,
    NORMALIZATION_RECOVERY_V4_DEFERRED_KIND,
    NORMALIZATION_RECOVERY_V4_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_evidence import (
    normalization_recovery_v4_candidate_occurrence_evidence,
    normalization_recovery_v4_candidate_occurrence_index,
    normalization_recovery_v4_eligible_outputs,
    normalization_recovery_v4_family_candidates,
    normalization_recovery_v4_support_evidence,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    V4_GROUP_DISPOSITIONS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _sha256(payload: bytes) -> str:
    """返回规范记录摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _scope_fields(candidate: dict[str, object] | None) -> dict[str, object]:
    """返回 conflict/defer 共用的显式 scope identity。"""
    if candidate is None:
        return {
            "candidate_scope_kind": "NONE",
            "source_execution_family": "",
            "source_execution_policy_scope": "",
            "target_policy_scope": "",
        }
    return {
        "candidate_scope_kind": candidate["candidate_scope_kind"],
        "source_execution_family": candidate["source_execution_family"],
        "source_execution_policy_scope": candidate[
            "source_execution_policy_scope"],
        "target_policy_scope": candidate["target_policy_scope"],
    }


def _input_families(
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """合并跨 fragment kind 的同 input group，并固定最后结算 group。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    group_ordinal = {}
    for ordinal, group in enumerate(groups):
        input_text = group.get("input_text")
        group_id = group.get("group_id")
        disposition = group.get("disposition")
        scope_kind = group.get("candidate_scope_kind")
        target_scope = group.get("target_policy_scope")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(group_id, str) or group_id in group_ordinal
                or disposition not in V4_GROUP_DISPOSITIONS
                or scope_kind not in {"NONE", "SOURCE_ONLY",
                                      "TARGET_CROSS_FAMILY"}
                or (scope_kind == "TARGET_CROSS_FAMILY")
                != (target_scope == RECOVERY_V4_TARGET_POLICY_SCOPE)
                or (scope_kind != "TARGET_CROSS_FAMILY" and target_scope != "")
                or group.get("unscoped_execution_allowed") != 0):
            raise BroadQaExternalDataError("v4 learner phrase group 漂移")
        grouped[input_text].append(group)
        group_ordinal[group_id] = ordinal
    values = []
    for input_text, members in sorted(grouped.items()):
        group_ids = sorted(str(item["group_id"]) for item in members)
        identity = {"input_text": input_text, "group_ids": group_ids}
        values.append({
            **identity,
            "emission_group_id": max(
                group_ids, key=lambda value: group_ordinal[value]),
            "family_id": _record_id(identity),
            "fragment_kinds": sorted({
                str(item["fragment_kind"]) for item in members}),
            "members": members,
        })
    return tuple(values)


def _family_variants(
        family: dict[str, object],
        fragment_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """把 family 内各 group variant 合并为 output 级支持账本。"""
    grouped: dict[str, set[str]] = defaultdict(set)
    for group in family["members"]:
        for variant in group["output_variants"]:
            grouped[str(variant["output_text"])].update(
                str(value) for value in variant["fragment_ids"])
    values = []
    for output_text, fragment_ids in sorted(grouped.items()):
        try:
            fragments = [fragment_by_id[value]
                         for value in sorted(fragment_ids)]
        except KeyError as error:
            raise BroadQaExternalDataError(
                "v4 learner family fragment 缺失") from error
        values.append({
            "fragment_ids": sorted(fragment_ids),
            "license_ids": sorted({str(item["license_id"])
                                   for item in fragments}),
            "output_text": output_text,
            "source_families": sorted({str(item["source_family"])
                                       for item in fragments}),
            "source_policy_scopes": sorted({
                str(item["source_policy_scope"]) for item in fragments}),
            "support_count": len(fragment_ids),
        })
    return tuple(values)


def _conflict_record(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        conflict_kind: str,
        variants: tuple[dict[str, object], ...],
        candidate: dict[str, object] | None,
        support_evidence_ids: list[str],
        refute_evidence_ids: list[str],
        context_signature_ids: list[str],
        ) -> dict[str, object]:
    """保留 output 或 scoped support/refute context 冲突。"""
    identity = {
        "candidate_id": "" if candidate is None else candidate["candidate_id"],
        "conflict_kind": conflict_kind,
        "family_id": family["family_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "conflict_id": _record_id(identity),
        "context_signature_ids": context_signature_ids,
        "format_version": 1,
        "group_ids": family["group_ids"],
        "input_text": family["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V4_CONFLICT_KIND,
        "refute_evidence_ids": refute_evidence_ids,
        "resolution_state": "DEFERRED_NO_OVERRIDE",
        "support_evidence_ids": support_evidence_ids,
        "unscoped_execution_allowed": 0,
        "variants": list(variants),
    }


def _deferred_record(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        defer_reason: str,
        candidate: dict[str, object] | None,
        candidate_output: str,
        evidence_ids: list[str],
        conflict_id: str,
        unaligned_occurrence_count: int,
        ) -> dict[str, object]:
    """形成每个未执行 scoped 假设的明确恢复理由。"""
    identity = {
        "candidate_id": "" if candidate is None else candidate["candidate_id"],
        "defer_reason": defer_reason,
        "family_id": family["family_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "candidate_output": candidate_output,
        "conflict_id": conflict_id,
        "deferred_id": _record_id(identity),
        "evidence_ids": evidence_ids,
        "format_version": 1,
        "fragment_kinds": family["fragment_kinds"],
        "group_ids": family["group_ids"],
        "input_text": family["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V4_DEFERRED_KIND,
        "unaligned_occurrence_count": unaligned_occurrence_count,
    }


def _derive_material(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]],
                   dict[str, object], tuple[dict[str, object], ...]]:
    """派生全部 v4 输出，并返回每个 input family 的 checkpoint 增量。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v4 learner protocol manifest")
    contract = protocol_manifest.get("learner_contract")
    if (not observations or not fragments or not groups or not work
            or protocol_manifest.get("target_policy_scope")
            != RECOVERY_V4_TARGET_POLICY_SCOPE
            or not isinstance(contract, dict)
            or contract.get("source_scoped_candidate_target_upgrade_allowed")
            != 0
            or contract.get(
                "target_candidate_min_distinct_ui_source_family_count") != 2):
        raise BroadQaExternalDataError(
            "v4 learner protocol material 为空或 scope 漂移")
    require_normalization_phrase_work_alignment(
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
        label="v4 learner",
    )
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    fragment_by_id = {str(item["fragment_id"]): item for item in fragments}
    if (len(observation_by_id) != len(observations)
            or len(fragment_by_id) != len(fragments)):
        raise BroadQaExternalDataError(
            "v4 learner observation/fragment identity 重复")
    families = _input_families(groups)
    family_material = []
    candidate_inputs = set()
    for family in families:
        variants = _family_variants(family, fragment_by_id)
        candidates = () if len(variants) != 1 else (
            normalization_recovery_v4_family_candidates(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                variant=variants[0],
                fragment_by_id=fragment_by_id,
            ))
        family_material.append((family, variants, candidates))
        candidate_inputs.update(str(item["input_text"]) for item in candidates)
    occurrence_index = normalization_recovery_v4_candidate_occurrence_index(
        observations, candidate_inputs)
    alignment_by_observation: dict[str, dict[int, set[int]]] = {}
    evidence = []
    target_rules = []
    source_rules = []
    defeaters = []
    conflicts = []
    deferred = []
    target_overlap = []
    source_overlap = []
    emission_counts = []
    for family, variants, candidates in family_material:
        family_evidence_count = 0
        family_result_count = 0
        if len(variants) != 1:
            conflict = _conflict_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                conflict_kind="TRAIN_OUTPUT_CONFLICT",
                variants=variants,
                candidate=None,
                support_evidence_ids=[],
                refute_evidence_ids=[],
                context_signature_ids=[],
            )
            conflicts.append(conflict)
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                defer_reason="TRAIN_OUTPUT_CONFLICT",
                candidate=None,
                candidate_output="",
                evidence_ids=[],
                conflict_id=str(conflict["conflict_id"]),
                unaligned_occurrence_count=0,
            ))
            family_result_count = 2
        elif not candidates:
            has_target_candidate = any(
                item["disposition"] == "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                for item in family["members"])
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family=family,
                defer_reason=(
                    "TARGET_TRANSFER_REQUIRES_CONTEXTUAL_EQUAL_LENGTH_EDIT_CORE"
                    if has_target_candidate
                    else "INSUFFICIENT_POSITIVE_AUTHORITY"),
                candidate=None,
                candidate_output=str(variants[0]["output_text"]),
                evidence_ids=[],
                conflict_id="",
                unaligned_occurrence_count=0,
            ))
            family_result_count = 1
        else:
            for candidate in candidates:
                support_fragments = tuple(
                    fragment_by_id[str(value)]
                    for value in candidate["support_fragment_ids"])
                support_evidence = tuple(
                    normalization_recovery_v4_support_evidence(
                        protocol_manifest_sha256=protocol_sha,
                        candidate=candidate,
                        fragment=fragment,
                        observation=observation_by_id[
                            str(fragment["observation_id"])],
                    ) for fragment in support_fragments)
                family_refutes, support_signatures, unaligned = (
                    normalization_recovery_v4_candidate_occurrence_evidence(
                        protocol_manifest_sha256=protocol_sha,
                        candidate=candidate,
                        occurrence_index=occurrence_index,
                        observation_by_id=observation_by_id,
                        support_fragments=support_fragments,
                        alignment_by_observation=alignment_by_observation,
                    ))
                evidence.extend(support_evidence)
                evidence.extend(family_refutes)
                family_evidence_count += (
                    len(support_evidence) + len(family_refutes))
                support_ids = sorted(str(item["evidence_id"])
                                     for item in support_evidence)
                refute_ids = sorted(str(item["evidence_id"])
                                    for item in family_refutes)
                collided = sorted(support_signatures.intersection({
                    str(item["context_signature"]["context_signature_id"])
                    for item in family_refutes}))
                all_evidence_ids = sorted(set(support_ids + refute_ids))
                if collided:
                    conflict = _conflict_record(
                        protocol_manifest_sha256=protocol_sha,
                        family=family,
                        conflict_kind=(
                            "SUPPORT_REFUTE_CONTEXT_SIGNATURE_CONFLICT"),
                        variants=variants,
                        candidate=candidate,
                        support_evidence_ids=support_ids,
                        refute_evidence_ids=refute_ids,
                        context_signature_ids=collided,
                    )
                    conflicts.append(conflict)
                    deferred.append(_deferred_record(
                        protocol_manifest_sha256=protocol_sha,
                        family=family,
                        defer_reason="CONTEXT_SIGNATURE_CONFLICT",
                        candidate=candidate,
                        candidate_output=str(candidate["output_text"]),
                        evidence_ids=all_evidence_ids,
                        conflict_id=str(conflict["conflict_id"]),
                        unaligned_occurrence_count=unaligned,
                    ))
                    family_result_count += 2
                    continue
                if not family_refutes:
                    deferred.append(_deferred_record(
                        protocol_manifest_sha256=protocol_sha,
                        family=family,
                        defer_reason="NO_REAL_REFUTE_EVIDENCE",
                        candidate=candidate,
                        candidate_output=str(candidate["output_text"]),
                        evidence_ids=support_ids,
                        conflict_id="",
                        unaligned_occurrence_count=unaligned,
                    ))
                    family_result_count += 1
                    continue
                rule, family_defeaters, index = (
                    normalization_recovery_v4_eligible_outputs(
                        protocol_manifest_sha256=protocol_sha,
                        family=family,
                        candidate=candidate,
                        support_evidence=support_evidence,
                        refutes=family_refutes,
                    ))
                defeaters.extend(family_defeaters)
                if candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY":
                    target_rules.append(rule)
                    target_overlap.append(index)
                else:
                    source_rules.append(rule)
                    source_overlap.append(index)
                family_result_count += 2 + len(family_defeaters)
        emission_counts.append({
            "evidence_increment": family_evidence_count,
            "group_id": family["emission_group_id"],
            "result_increment": family_result_count,
        })
    raw_outputs = {
        "evidence.jsonl": evidence,
        "target-phrase-rules.jsonl": target_rules,
        "source-phrase-rules.jsonl": source_rules,
        "defeaters.jsonl": defeaters,
        "conflict-ledger.jsonl": conflicts,
        "deferred-groups.jsonl": deferred,
        "target-overlap-index.jsonl": target_overlap,
        "source-overlap-index.jsonl": source_overlap,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_RECOVERY_V4_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        if len({item[identity_key] for item in values}) != len(values):
            raise BroadQaExternalDataError(
                f"v4 learner {name} identity 重复")
        outputs[name] = values
    stance_counts = Counter(
        str(item["stance"]) for item in outputs["evidence.jsonl"])
    defer_counts = Counter(
        str(item["defer_reason"])
        for item in outputs["deferred-groups.jsonl"])
    conflict_counts = Counter(
        str(item["conflict_kind"])
        for item in outputs["conflict-ledger.jsonl"])
    source_rule_counts = Counter(
        str(item["source_execution_family"])
        for item in outputs["source-phrase-rules.jsonl"])
    result_count = sum(
        len(values) for name, values in outputs.items()
        if name != "evidence.jsonl")
    all_rules = (outputs["target-phrase-rules.jsonl"]
                 + outputs["source-phrase-rules.jsonl"])
    summary = {
        "conflict_count": len(outputs["conflict-ledger.jsonl"]),
        "conflict_kind_counts": dict(sorted(conflict_counts.items())),
        "defeater_count": len(outputs["defeaters.jsonl"]),
        "defer_reason_counts": dict(sorted(defer_counts.items())),
        "deferred_candidate_count": len(outputs["deferred-groups.jsonl"]),
        "equal_length_rule_count": sum(
            item["equal_length"] == 1 for item in all_rules),
        "evidence_count": len(outputs["evidence.jsonl"]),
        "evidence_stance_counts": {
            "REFUTE": stance_counts["REFUTE"],
            "SUPPORT": stance_counts["SUPPORT"],
        },
        "input_family_count": len(families),
        "result_record_count": result_count,
        "source_overlap_index_count": len(
            outputs["source-overlap-index.jsonl"]),
        "source_rule_count": len(outputs["source-phrase-rules.jsonl"]),
        "source_rule_family_counts": dict(sorted(source_rule_counts.items())),
        "target_overlap_index_count": len(
            outputs["target-overlap-index.jsonl"]),
        "target_rule_count": len(outputs["target-phrase-rules.jsonl"]),
        "variable_length_rule_count": sum(
            item["variable_length"] == 1 for item in all_rules),
    }
    expected_evidence = sum(
        int(item["evidence_increment"]) for item in emission_counts)
    expected_results = sum(
        int(item["result_increment"]) for item in emission_counts)
    if (expected_evidence != summary["evidence_count"]
            or expected_results != result_count
            or summary["target_rule_count"]
            != summary["target_overlap_index_count"]
            or summary["source_rule_count"]
            != summary["source_overlap_index_count"]):
        raise BroadQaExternalDataError(
            "v4 learner output/prefix count 漂移")
    return outputs, summary, tuple(emission_counts)


def derive_normalization_recovery_v4_learning_outputs(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]],
                   dict[str, object], tuple[dict[str, object], ...]]:
    """公开纯派生入口；第三项供 append-only checkpoint 前缀计数。"""
    return _derive_material(
        protocol_manifest=protocol_manifest,
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
    )


def normalization_recovery_v4_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """机械计算任意 v4 ordered-work 前缀的 Evidence/result 数。"""
    return normalization_phrase_prefix_output_counts(
        work=work,
        emission_counts=emission_counts,
        processed_item_count=processed_item_count,
        label="v4 learner",
    )


def normalization_recovery_v4_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把完整 v4 输出转为 manifest-last writer 的规范 JSONL。"""
    return normalization_phrase_output_payloads(
        outputs,
        output_file_roles=NORMALIZATION_RECOVERY_V4_OUTPUT_FILE_ROLES,
        label="v4 learner",
    )


__all__ = [
    "derive_normalization_recovery_v4_learning_outputs",
    "normalization_recovery_v4_output_payloads",
    "normalization_recovery_v4_prefix_output_counts",
]
