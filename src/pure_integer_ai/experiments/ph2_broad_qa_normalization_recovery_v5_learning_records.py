"""结算 recovery-v5 TRAIN groups 并派生完整 learner 输出。"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_phrase_learning import (
    normalization_phrase_output_payloads,
    require_normalization_phrase_work_alignment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_CONFLICT_KIND,
    NORMALIZATION_RECOVERY_V5_DEFERRED_KIND,
    NORMALIZATION_RECOVERY_V5_IDENTITY_OBSERVATION_KIND,
    NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
    NORMALIZATION_RECOVERY_V5_RULE_CLASSES,
    normalization_recovery_v5_rule_class,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_evidence import (
    normalization_recovery_v5_candidate_occurrence_evidence,
    normalization_recovery_v5_candidate_occurrence_index,
    normalization_recovery_v5_eligible_outputs,
    normalization_recovery_v5_group_candidates,
    normalization_recovery_v5_support_evidence,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
    V5_FRAGMENT_KINDS,
    V5_GROUP_DISPOSITIONS,
    V5_SOURCE_FAMILIES,
    V5_SOURCE_POLICY_BY_FAMILY,
    derive_normalization_recovery_v5_groups,
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


def _rule_class(group: dict[str, object]) -> str:
    """从单输出 group 返回规则分桶；冲突组返回空分桶。"""
    variants = group["output_variants"]
    if len(variants) != 1:
        return ""
    try:
        return normalization_recovery_v5_rule_class(
            str(group["fragment_kind"]),
            equal_length=int(variants[0]["equal_length"]),
        )
    except (TypeError, ValueError) as error:
        raise BroadQaExternalDataError(
            "v5 learner group rule class 漂移") from error


def _validate_observations(
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """核验 observation 的来源、结构、identity 与长度事实。"""
    result = {}
    for observation in observations:
        observation_id = observation.get("observation_id")
        input_text = observation.get("input_text")
        output_text = observation.get("output_text")
        source_family = observation.get("source_family")
        structure_tokens = observation.get("structure_tokens")
        if (not isinstance(observation_id, str) or not observation_id
                or observation_id in result
                or not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text
                or source_family not in V5_SOURCE_POLICY_BY_FAMILY
                or observation.get("source_policy_scope")
                != V5_SOURCE_POLICY_BY_FAMILY[source_family]
                or not isinstance(structure_tokens, list)
                or any(not isinstance(item, str) for item in structure_tokens)
                or observation.get("identity_preservation")
                != int(input_text == output_text)
                or observation.get("equal_length")
                != int(len(input_text) == len(output_text))):
            raise BroadQaExternalDataError(
                "v5 learner observation schema/identity 漂移")
        result[observation_id] = observation
    source_roster = {str(item["source_family"]) for item in observations}
    if (len(source_roster) not in {3, 4}
            or not source_roster.issubset(set(V5_SOURCE_FAMILIES))):
        raise BroadQaExternalDataError(
            "v5 learner observation source roster 漂移")
    return result


def _validate_fragments(
        fragments: tuple[dict[str, object], ...],
        observation_by_id: dict[str, dict[str, object]],
        ) -> dict[str, dict[str, object]]:
    """核验 fragment span、来源与 WHOLE_INPUT 结构 token 未丢失。"""
    result = {}
    for fragment in fragments:
        fragment_id = fragment.get("fragment_id")
        observation_id = fragment.get("observation_id")
        if (not isinstance(fragment_id, str) or not fragment_id
                or fragment_id in result
                or not isinstance(observation_id, str)
                or observation_id not in observation_by_id
                or fragment.get("fragment_kind") not in V5_FRAGMENT_KINDS):
            raise BroadQaExternalDataError(
                "v5 learner fragment identity/kind 漂移")
        observation = observation_by_id[observation_id]
        positions = tuple(fragment.get(key) for key in (
            "input_start", "input_end", "output_start", "output_end"))
        if (any(type(value) is not int for value in positions)
                or not 0 <= positions[0] < positions[1] <= len(
                    str(observation["input_text"]))
                or not 0 <= positions[2] <= positions[3] <= len(
                    str(observation["output_text"]))
                or fragment.get("input_text")
                != str(observation["input_text"])[positions[0]:positions[1]]
                or fragment.get("output_text")
                != str(observation["output_text"])[positions[2]:positions[3]]
                or fragment.get("equal_length") != int(
                    len(str(fragment.get("input_text")))
                    == len(str(fragment.get("output_text"))))
                or fragment.get("license_id") != observation["license_id"]
                or fragment.get("source_family")
                != observation["source_family"]
                or fragment.get("source_policy_scope")
                != observation["source_policy_scope"]):
            raise BroadQaExternalDataError(
                "v5 learner fragment span/source 漂移")
        if fragment["fragment_kind"] == "WHOLE_INPUT":
            if (positions != (0, len(str(observation["input_text"])),
                              0, len(str(observation["output_text"])))
                    or fragment.get("structure_tokens")
                    != observation["structure_tokens"]):
                raise BroadQaExternalDataError(
                    "v5 learner WHOLE_INPUT structure/span 漂移")
        elif observation["structure_tokens"]:
            raise BroadQaExternalDataError(
                "v5 learner structured observation 产生局部 fragment")
        result[fragment_id] = fragment
    return result


def _validate_groups(
        groups: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        ) -> None:
    """重聚合 fragments，核验 group authority、scope 与 variant 逐字段一致。"""
    expected = derive_normalization_recovery_v5_groups(fragments)
    if canonical_json_bytes(groups) != canonical_json_bytes(expected):
        raise BroadQaExternalDataError(
            "v5 learner group/fragment authority 重派生漂移")


def _family_ids(
        groups: tuple[dict[str, object], ...],
        ) -> dict[str, str]:
    """为同一 input 的全部 fragment class 形成稳定 family identity。"""
    grouped: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        grouped[str(group["input_text"])].append(str(group["group_id"]))
    return {
        input_text: _record_id({
            "group_ids": sorted(group_ids),
            "input_text": input_text,
        })
        for input_text, group_ids in grouped.items()
    }


def _identity_observation_record(
        *,
        protocol_manifest_sha256: str,
        observation: dict[str, object],
        ) -> dict[str, object]:
    """把 identity observation 保留成 hard audit bucket 的一等记录。"""
    identity = {
        "observation_id": observation["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
    }
    return {
        **identity,
        "format_version": 1,
        "identity_record_id": _record_id(identity),
        "input_text": observation["input_text"],
        "license_id": observation["license_id"],
        "output_text": observation["output_text"],
        "record_kind": NORMALIZATION_RECOVERY_V5_IDENTITY_OBSERVATION_KIND,
        "source_commitment": observation["source_commitment"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "structure_tokens": observation["structure_tokens"],
    }


def _conflict_record(
        *,
        protocol_manifest_sha256: str,
        family_id: str,
        group: dict[str, object],
        conflict_kind: str,
        candidate: dict[str, object] | None,
        support_evidence_ids: list[str],
        refute_evidence_ids: list[str],
        context_signature_ids: list[str],
        ) -> dict[str, object]:
    """保留 output 或 scoped support/refute context 冲突。"""
    identity = {
        "candidate_id": "" if candidate is None else candidate["candidate_id"],
        "conflict_kind": conflict_kind,
        "family_id": family_id,
        "group_id": group["group_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "authority_basis": group["authority_basis"],
        "conflict_id": _record_id(identity),
        "context_signature_ids": context_signature_ids,
        "format_version": 1,
        "fragment_kind": group["fragment_kind"],
        "input_text": group["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V5_CONFLICT_KIND,
        "refute_evidence_ids": refute_evidence_ids,
        "resolution_state": "DEFERRED_NO_OVERRIDE",
        "rule_class": _rule_class(group),
        "support_evidence_ids": support_evidence_ids,
        "unscoped_execution_allowed": 0,
        "variants": group["output_variants"],
    }


def _deferred_record(
        *,
        protocol_manifest_sha256: str,
        family_id: str,
        group: dict[str, object],
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
        "family_id": family_id,
        "group_id": group["group_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "authority_basis": group["authority_basis"],
        "candidate_output": candidate_output,
        "conflict_id": conflict_id,
        "deferred_id": _record_id(identity),
        "evidence_ids": evidence_ids,
        "format_version": 1,
        "fragment_kind": group["fragment_kind"],
        "input_text": group["input_text"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V5_DEFERRED_KIND,
        "rule_class": _rule_class(group),
        "unaligned_occurrence_count": unaligned_occurrence_count,
    }


def _protocol_contract(protocol_manifest: dict[str, object]) -> str:
    """核验 v5 learner 依赖的协议门并返回 manifest SHA。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v5 learner protocol manifest")
    contract = protocol_manifest.get("learner_contract")
    if (protocol_manifest.get("target_policy_scope")
            != RECOVERY_V5_TARGET_POLICY_SCOPE
            or not isinstance(contract, dict)
            or contract.get("source_scoped_candidate_target_upgrade_allowed")
            != 0
            or contract.get("target_equal_length_min_distinct_source_family_count")
            != 2
            or contract.get("target_variable_length_min_distinct_source_family_count")
            != 3
            or contract.get("target_variable_length_two_family_replicated_support_allowed")
            != 1
            or contract.get("identity_preservation_hard_gate_required") != 1
            or contract.get("negative_evidence_required_before_execution") != 1
            or contract.get("whole_input_exact_precedes_phrase_lexicon") != 1):
        raise BroadQaExternalDataError(
            "v5 learner protocol contract 漂移")
    return protocol_sha


def _derive_material(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]],
                   dict[str, object], tuple[dict[str, object], ...]]:
    """派生全部 v5 输出，并返回 observation/group checkpoint 增量。"""
    if not observations or not fragments or not groups or not work:
        raise BroadQaExternalDataError("v5 learner material 为空")
    protocol_sha = _protocol_contract(protocol_manifest)
    require_normalization_phrase_work_alignment(
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
        label="v5 learner",
    )
    observation_by_id = _validate_observations(observations)
    fragment_by_id = _validate_fragments(fragments, observation_by_id)
    _validate_groups(groups, fragments)
    family_ids = _family_ids(groups)
    candidate_material = []
    candidate_inputs = set()
    for group in groups:
        candidates = normalization_recovery_v5_group_candidates(
            protocol_manifest_sha256=protocol_sha,
            family_id=family_ids[str(group["input_text"])],
            group=group,
            fragment_by_id=fragment_by_id,
        )
        candidate_material.append((group, candidates))
        candidate_inputs.update(str(item["input_text"])
                                for item in candidates)
    occurrence_index = normalization_recovery_v5_candidate_occurrence_index(
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
    identity_observations = [
        _identity_observation_record(
            protocol_manifest_sha256=protocol_sha,
            observation=observation,
        )
        for observation in observations
        if observation["identity_preservation"] == 1
    ]
    emission_counts = [{
        "evidence_increment": 0,
        "record_id": item["observation_id"],
        "result_increment": 1,
        "work_kind": "PAIR_OBSERVATION",
    } for item in observations if item["identity_preservation"] == 1]
    for group, candidates in candidate_material:
        family_evidence_count = 0
        family_result_count = 0
        family_id = family_ids[str(group["input_text"])]
        variants = group["output_variants"]
        if group["disposition"] == "CONFLICT_DEFER":
            conflict = _conflict_record(
                protocol_manifest_sha256=protocol_sha,
                family_id=family_id,
                group=group,
                conflict_kind="TRAIN_OUTPUT_CONFLICT",
                candidate=None,
                support_evidence_ids=[],
                refute_evidence_ids=[],
                context_signature_ids=[],
            )
            conflicts.append(conflict)
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family_id=family_id,
                group=group,
                defer_reason="TRAIN_OUTPUT_CONFLICT",
                candidate=None,
                candidate_output="",
                evidence_ids=[],
                conflict_id=str(conflict["conflict_id"]),
                unaligned_occurrence_count=0,
            ))
            family_result_count = 2
        elif not candidates:
            deferred.append(_deferred_record(
                protocol_manifest_sha256=protocol_sha,
                family_id=family_id,
                group=group,
                defer_reason=str(group["authority_basis"]),
                candidate=None,
                candidate_output=(
                    "" if len(variants) != 1
                    else str(variants[0]["output_text"])),
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
                    normalization_recovery_v5_support_evidence(
                        protocol_manifest_sha256=protocol_sha,
                        candidate=candidate,
                        fragment=fragment,
                        observation=observation_by_id[
                            str(fragment["observation_id"])],
                    ) for fragment in support_fragments)
                family_refutes, support_signatures, unaligned = (
                    normalization_recovery_v5_candidate_occurrence_evidence(
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
                        family_id=family_id,
                        group=group,
                        conflict_kind=(
                            "SUPPORT_REFUTE_CONTEXT_SIGNATURE_CONFLICT"),
                        candidate=candidate,
                        support_evidence_ids=support_ids,
                        refute_evidence_ids=refute_ids,
                        context_signature_ids=collided,
                    )
                    conflicts.append(conflict)
                    deferred.append(_deferred_record(
                        protocol_manifest_sha256=protocol_sha,
                        family_id=family_id,
                        group=group,
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
                        family_id=family_id,
                        group=group,
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
                    normalization_recovery_v5_eligible_outputs(
                        protocol_manifest_sha256=protocol_sha,
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
            "record_id": group["group_id"],
            "result_increment": family_result_count,
            "work_kind": "PHRASE_GROUP",
        })
    raw_outputs = {
        "evidence.jsonl": evidence,
        "target-phrase-rules.jsonl": target_rules,
        "source-phrase-rules.jsonl": source_rules,
        "defeaters.jsonl": defeaters,
        "conflict-ledger.jsonl": conflicts,
        "deferred-groups.jsonl": deferred,
        "identity-observations.jsonl": identity_observations,
        "target-overlap-index.jsonl": target_overlap,
        "source-overlap-index.jsonl": source_overlap,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        if len({item[identity_key] for item in values}) != len(values):
            raise BroadQaExternalDataError(
                f"v5 learner {name} identity 重复")
        outputs[name] = values
    stance_counts = Counter(
        str(item["stance"]) for item in outputs["evidence.jsonl"])
    defer_counts = Counter(
        str(item["defer_reason"])
        for item in outputs["deferred-groups.jsonl"])
    conflict_counts = Counter(
        str(item["conflict_kind"])
        for item in outputs["conflict-ledger.jsonl"])
    all_rules = (outputs["target-phrase-rules.jsonl"]
                 + outputs["source-phrase-rules.jsonl"])
    target_class_counts = Counter(
        str(item["rule_class"])
        for item in outputs["target-phrase-rules.jsonl"])
    source_class_counts = Counter(
        str(item["rule_class"])
        for item in outputs["source-phrase-rules.jsonl"])
    candidate_class_counts = Counter(
        _rule_class(item) for item in groups
        if item["disposition"].endswith("CANDIDATE"))
    target_candidate_class_counts = Counter(
        _rule_class(item) for item in groups
        if item["candidate_scope_kind"] == "TARGET_CROSS_FAMILY")
    source_candidate_class_counts = Counter(
        _rule_class(item) for item in groups
        if item["candidate_scope_kind"] == "SOURCE_ONLY")
    target_deferred_class_counts = Counter(
        str(item["rule_class"])
        for item in outputs["deferred-groups.jsonl"]
        if item["candidate_scope_kind"] == "TARGET_CROSS_FAMILY")
    source_deferred_class_counts = Counter(
        str(item["rule_class"])
        for item in outputs["deferred-groups.jsonl"]
        if item["candidate_scope_kind"] == "SOURCE_ONLY")
    source_rule_counts = Counter(
        str(item["source_execution_family"])
        for item in outputs["source-phrase-rules.jsonl"])
    result_count = sum(
        len(values) for name, values in outputs.items()
        if name != "evidence.jsonl")
    summary = {
        "candidate_group_rule_class_counts": {
            key: candidate_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
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
        "identity_observation_count": len(
            outputs["identity-observations.jsonl"]),
        "identity_source_family_counts": dict(sorted(Counter(
            str(item["source_family"])
            for item in outputs["identity-observations.jsonl"]).items())),
        "identity_structured_observation_count": sum(
            bool(item["structure_tokens"])
            for item in outputs["identity-observations.jsonl"]),
        "negative_evidence_closed_rule_count": sum(
            bool(item["negative_evidence_ids"]) for item in all_rules),
        "refute_identity_preservation_count": sum(
            item["stance"] == "REFUTE"
            and item["identity_preservation"] == 1
            for item in outputs["evidence.jsonl"]),
        "result_record_count": result_count,
        "source_overlap_index_count": len(
            outputs["source-overlap-index.jsonl"]),
        "source_candidate_group_rule_class_counts": {
            key: source_candidate_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "source_deferred_candidate_rule_class_counts": {
            key: source_deferred_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "source_rule_class_counts": {
            key: source_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "source_rule_count": len(outputs["source-phrase-rules.jsonl"]),
        "source_rule_family_counts": dict(sorted(source_rule_counts.items())),
        "structured_rule_count": sum(
            any(item["structure_token_variants"]) for item in all_rules),
        "target_overlap_index_count": len(
            outputs["target-overlap-index.jsonl"]),
        "target_candidate_group_rule_class_counts": {
            key: target_candidate_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "target_deferred_candidate_rule_class_counts": {
            key: target_deferred_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "target_rule_class_counts": {
            key: target_class_counts[key]
            for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES},
        "target_rule_count": len(outputs["target-phrase-rules.jsonl"]),
        "variable_length_rule_count": sum(
            item["variable_length"] == 1 for item in all_rules),
    }
    expected_evidence = sum(
        int(item["evidence_increment"]) for item in emission_counts)
    expected_results = sum(
        int(item["result_increment"]) for item in emission_counts)
    candidate_resolution_equal = all(
        target_candidate_class_counts[key]
        == target_class_counts[key] + target_deferred_class_counts[key]
        and source_candidate_class_counts[key]
        == source_class_counts[key] + source_deferred_class_counts[key]
        for key in NORMALIZATION_RECOVERY_V5_RULE_CLASSES)
    if (expected_evidence != summary["evidence_count"]
            or expected_results != result_count
            or not candidate_resolution_equal
            or summary["identity_observation_count"]
            != sum(item["identity_preservation"] == 1
                   for item in observations)
            or summary["negative_evidence_closed_rule_count"] != len(all_rules)
            or summary["target_rule_count"]
            != summary["target_overlap_index_count"]
            or summary["source_rule_count"]
            != summary["source_overlap_index_count"]):
        raise BroadQaExternalDataError(
            "v5 learner output/prefix/audit count 漂移")
    return outputs, summary, tuple(emission_counts)


def derive_normalization_recovery_v5_learning_outputs(
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


def normalization_recovery_v5_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """计算 observation identity 与 group 输出共同构成的任意 work 前缀。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError("v5 learner processed prefix 非法")
    by_emission = {
        (str(item["work_kind"]), str(item["record_id"])): item
        for item in emission_counts}
    if len(by_emission) != len(emission_counts):
        raise BroadQaExternalDataError(
            "v5 learner emission identity 重复")
    evidence_count = 0
    result_count = 0
    seen = set()
    for item in work[:processed_item_count]:
        kind = str(item["work_kind"])
        if kind not in {"PAIR_OBSERVATION", "PHRASE_FRAGMENT", "PHRASE_GROUP"}:
            raise BroadQaExternalDataError("v5 learner work kind 非法")
        key = (kind, str(item["record_id"]))
        counts = by_emission.get(key)
        if counts is None:
            continue
        seen.add(key)
        evidence_count += int(counts["evidence_increment"])
        result_count += int(counts["result_increment"])
    if processed_item_count == len(work) and seen != set(by_emission):
        raise BroadQaExternalDataError(
            "v5 learner emission 未被 work 覆盖")
    return evidence_count, result_count


def normalization_recovery_v5_checkpoint_prefix_context(
        *,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[int, ...]]:
    """按 emission work ordinal 预计算 checkpoint 的整数累计查询表。"""
    work_ordinal = {
        (str(item["work_kind"]), str(item["record_id"])): ordinal
        for ordinal, item in enumerate(work)}
    if len(work_ordinal) != len(work):
        raise BroadQaExternalDataError(
            "v5 learner checkpoint work identity 重复")
    ordered = []
    seen = set()
    for emission in emission_counts:
        key = (str(emission["work_kind"]), str(emission["record_id"]))
        if key in seen or key not in work_ordinal:
            raise BroadQaExternalDataError(
                "v5 learner checkpoint emission/work 漂移")
        seen.add(key)
        evidence = emission["evidence_increment"]
        result = emission["result_increment"]
        if (type(evidence) is not int or evidence < 0
                or type(result) is not int or result < 0):
            raise BroadQaExternalDataError(
                "v5 learner checkpoint emission count 非法")
        ordered.append((work_ordinal[key] + 1, evidence, result))
    ordered.sort()
    positions = []
    evidence_prefix = []
    result_prefix = []
    evidence_total = 0
    result_total = 0
    for position, evidence, result in ordered:
        evidence_total += evidence
        result_total += result
        positions.append(position)
        evidence_prefix.append(evidence_total)
        result_prefix.append(result_total)
    return {
        "evidence_prefix": tuple(evidence_prefix),
        "positions": tuple(positions),
        "result_prefix": tuple(result_prefix),
    }


def normalization_recovery_v5_checkpoint_prefix_counts(
        *,
        work_item_count: int,
        prefix_context: dict[str, tuple[int, ...]],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """以二分查询返回任意 checkpoint 前缀的 Evidence/result 计数。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= work_item_count
            or set(prefix_context)
            != {"evidence_prefix", "positions", "result_prefix"}):
        raise BroadQaExternalDataError(
            "v5 learner checkpoint prefix context 非法")
    positions = prefix_context["positions"]
    evidence_prefix = prefix_context["evidence_prefix"]
    result_prefix = prefix_context["result_prefix"]
    if (not isinstance(positions, tuple)
            or not isinstance(evidence_prefix, tuple)
            or not isinstance(result_prefix, tuple)
            or not len(positions) == len(evidence_prefix) == len(result_prefix)
            or any(type(value) is not int for values in (
                positions, evidence_prefix, result_prefix) for value in values)):
        raise BroadQaExternalDataError(
            "v5 learner checkpoint prefix table 漂移")
    count = bisect_right(positions, processed_item_count)
    if count == 0:
        return 0, 0
    return evidence_prefix[count - 1], result_prefix[count - 1]


def normalization_recovery_v5_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把完整 v5 输出转为 manifest-last writer 的规范 JSONL。"""
    return normalization_phrase_output_payloads(
        outputs,
        output_file_roles=NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
        label="v5 learner",
    )


__all__ = [
    "derive_normalization_recovery_v5_learning_outputs",
    "normalization_recovery_v5_checkpoint_prefix_context",
    "normalization_recovery_v5_checkpoint_prefix_counts",
    "normalization_recovery_v5_output_payloads",
    "normalization_recovery_v5_prefix_output_counts",
]
