"""Recovery-v4 scoped candidate、Evidence 与可执行规则的纯派生。"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_phrase_learning import (
    normalization_phrase_alignment_boundary_map,
    normalization_phrase_context_signature,
    normalization_phrase_observed_output,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_contract import (
    NORMALIZATION_RECOVERY_V4_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V4_EVIDENCE_KIND,
    NORMALIZATION_RECOVERY_V4_SOURCE_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V4_SOURCE_RULE_KIND,
    NORMALIZATION_RECOVERY_V4_TARGET_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V4_TARGET_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    V4_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _sha256(payload: bytes) -> str:
    """返回规范记录摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _scope_fields(candidate: dict[str, object]) -> dict[str, object]:
    """返回规则相关记录共享的显式 scope identity。"""
    return {
        "candidate_scope_kind": candidate["candidate_scope_kind"],
        "source_execution_family": candidate["source_execution_family"],
        "source_execution_policy_scope": candidate[
            "source_execution_policy_scope"],
        "target_policy_scope": candidate["target_policy_scope"],
    }


def _candidate(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        variant: dict[str, object],
        candidate_scope_kind: str,
        source_execution_family: str,
        source_execution_policy_scope: str,
        support_fragment_ids: list[str],
        ) -> dict[str, object]:
    """形成一条 target 或 source-only 假设 identity。"""
    target_scope = (
        RECOVERY_V4_TARGET_POLICY_SCOPE
        if candidate_scope_kind == "TARGET_CROSS_FAMILY" else "")
    identity = {
        "candidate_scope_kind": candidate_scope_kind,
        "family_id": family["family_id"],
        "input_text": family["input_text"],
        "output_text": variant["output_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_execution_family": source_execution_family,
        "source_execution_policy_scope": source_execution_policy_scope,
        "target_policy_scope": target_scope,
    }
    return {
        **identity,
        "candidate_id": _record_id(identity),
        "support_fragment_ids": support_fragment_ids,
    }


def normalization_recovery_v4_family_candidates(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        variant: dict[str, object],
        fragment_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """按 group authority 形成互斥 target 或逐 family source 假设。"""
    target_members = [item for item in family["members"]
                      if item["disposition"]
                      == "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                      and item["fragment_kind"] == "EDIT_CORE"]
    source_members = [item for item in family["members"]
                      if item["disposition"] == "SOURCE_SCOPED_CANDIDATE"]
    if target_members:
        if any(item["candidate_scope_kind"] != "TARGET_CROSS_FAMILY"
               or item["target_policy_scope"]
               != RECOVERY_V4_TARGET_POLICY_SCOPE
               or len(item["source_families"]) < 2
               for item in target_members):
            raise BroadQaExternalDataError(
                "v4 learner target group authority 漂移")
        support_fragment_ids = sorted({
            str(fragment_id)
            for member in target_members
            for group_variant in member["output_variants"]
            if group_variant["output_text"] == variant["output_text"]
            for fragment_id in group_variant["fragment_ids"]
        })
        support_families = {
            str(fragment_by_id[value]["source_family"])
            for value in support_fragment_ids}
        if (len(str(family["input_text"])) < 3
                or len(str(family["input_text"]))
                != len(str(variant["output_text"]))
                or len(support_families) < 2):
            return ()
        return (_candidate(
            protocol_manifest_sha256=protocol_manifest_sha256,
            family=family,
            variant=variant,
            candidate_scope_kind="TARGET_CROSS_FAMILY",
            source_execution_family="",
            source_execution_policy_scope="",
            support_fragment_ids=support_fragment_ids,
        ),)
    source_families = set()
    for group in source_members:
        families = group.get("source_families")
        if (group.get("candidate_scope_kind") != "SOURCE_ONLY"
                or group.get("target_policy_scope") != ""
                or not isinstance(families, list) or len(families) != 1
                or families[0] not in V4_SOURCE_FAMILIES):
            raise BroadQaExternalDataError(
                "v4 learner source group authority 漂移")
        source_families.add(str(families[0]))
    values = []
    for source_family in sorted(source_families):
        fragments = [fragment_by_id[str(value)]
                     for value in variant["fragment_ids"]
                     if fragment_by_id[str(value)]["source_family"]
                     == source_family]
        policies = sorted({str(item["source_policy_scope"])
                           for item in fragments})
        if not fragments or len(policies) != 1:
            raise BroadQaExternalDataError(
                "v4 learner source candidate fragment scope 漂移")
        values.append(_candidate(
            protocol_manifest_sha256=protocol_manifest_sha256,
            family=family,
            variant=variant,
            candidate_scope_kind="SOURCE_ONLY",
            source_execution_family=source_family,
            source_execution_policy_scope=policies[0],
            support_fragment_ids=sorted(
                str(item["fragment_id"]) for item in fragments),
        ))
    return tuple(values)


def normalization_recovery_v4_support_evidence(
        *,
        protocol_manifest_sha256: str,
        candidate: dict[str, object],
        fragment: dict[str, object],
        observation: dict[str, object],
        ) -> dict[str, object]:
    """把 candidate scope 内的物化 fragment 绑定为 SUPPORT。"""
    signature = normalization_phrase_context_signature(
        str(observation["input_text"]),
        int(fragment["input_start"]),
        int(fragment["input_end"]),
        identity_builder=_record_id,
    )
    identity = {
        "candidate_id": candidate["candidate_id"],
        "family_id": candidate["family_id"],
        "fragment_id": fragment["fragment_id"],
        "hypothesis_input": candidate["input_text"],
        "hypothesis_output": candidate["output_text"],
        "observation_id": fragment["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "stance": "SUPPORT",
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "context_signature": signature,
        "equal_length": fragment["equal_length"],
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "fragment_kind": fragment["fragment_kind"],
        "input_end": fragment["input_end"],
        "input_start": fragment["input_start"],
        "license_id": fragment["license_id"],
        "output_end": fragment["output_end"],
        "output_start": fragment["output_start"],
        "record_kind": NORMALIZATION_RECOVERY_V4_EVIDENCE_KIND,
        "source_commitment": observation["source_commitment"],
        "source_family": fragment["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": fragment["source_policy_scope"],
    }


def _refute_evidence(
        *,
        protocol_manifest_sha256: str,
        candidate: dict[str, object],
        observation: dict[str, object],
        start: int,
        end: int,
        observed_output: str,
        signature: dict[str, object],
        ) -> dict[str, object]:
    """记录 candidate scope 内未产生候选 output 的真实 REFUTE。"""
    reason = (
        "IDENTITY_PRESERVATION"
        if observed_output == candidate["input_text"] else "ALTERNATIVE_OUTPUT")
    identity = {
        "candidate_id": candidate["candidate_id"],
        "candidate_output": candidate["output_text"],
        "family_id": candidate["family_id"],
        "hypothesis_input": candidate["input_text"],
        "observation_id": observation["observation_id"],
        "occurrence_end": end,
        "occurrence_start": start,
        "observed_output": observed_output,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "stance": "REFUTE",
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "context_signature": signature,
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "license_id": observation["license_id"],
        "record_kind": NORMALIZATION_RECOVERY_V4_EVIDENCE_KIND,
        "refute_reason": reason,
        "source_commitment": observation["source_commitment"],
        "source_family": observation["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": observation["source_policy_scope"],
    }


def normalization_recovery_v4_candidate_occurrence_index(
        observations: tuple[dict[str, object], ...],
        candidate_inputs: set[str],
        ) -> dict[str, tuple[tuple[str, int, int], ...]]:
    """单遍扫描 TRAIN，按首 scalar 分桶索引全部候选 occurrence。"""
    buckets: dict[str, list[str]] = defaultdict(list)
    for phrase in sorted(candidate_inputs):
        if not phrase:
            raise BroadQaExternalDataError("v4 learner candidate input 为空")
        buckets[phrase[0]].append(phrase)
    for values in buckets.values():
        values.sort(key=lambda item: (-len(item), item))
    occurrences: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for observation in observations:
        observation_id = str(observation["observation_id"])
        text = str(observation["input_text"])
        for start, scalar in enumerate(text):
            for phrase in buckets.get(scalar, ()):
                if text.startswith(phrase, start):
                    occurrences[phrase].append(
                        (observation_id, start, start + len(phrase)))
    return {key: tuple(values) for key, values in occurrences.items()}


def normalization_recovery_v4_candidate_occurrence_evidence(
        *,
        protocol_manifest_sha256: str,
        candidate: dict[str, object],
        occurrence_index: dict[str, tuple[tuple[str, int, int], ...]],
        observation_by_id: dict[str, dict[str, object]],
        support_fragments: tuple[dict[str, object], ...],
        alignment_by_observation: dict[str, dict[int, set[int]]],
        ) -> tuple[tuple[dict[str, object], ...], set[str], int]:
    """返回 scope 内 REFUTE、正上下文签名与未对齐 occurrence 数。"""
    input_text = str(candidate["input_text"])
    source_family = str(candidate["source_execution_family"])
    support_locations = {
        (str(item["observation_id"]), int(item["input_start"]),
         int(item["input_end"])) for item in support_fragments
    }
    support_signatures = set()
    for fragment in support_fragments:
        observation = observation_by_id[str(fragment["observation_id"])]
        signature = normalization_phrase_context_signature(
            str(observation["input_text"]),
            int(fragment["input_start"]),
            int(fragment["input_end"]),
            identity_builder=_record_id,
        )
        support_signatures.add(str(signature["context_signature_id"]))
    refutes = []
    unaligned = 0
    for observation_id, start, end in occurrence_index.get(input_text, ()):
        observation = observation_by_id[observation_id]
        if source_family and observation["source_family"] != source_family:
            continue
        signature = normalization_phrase_context_signature(
            str(observation["input_text"]), start, end,
            identity_builder=_record_id,
        )
        if (observation_id, start, end) in support_locations:
            support_signatures.add(str(signature["context_signature_id"]))
            continue
        boundaries = alignment_by_observation.get(observation_id)
        if boundaries is None:
            boundaries = normalization_phrase_alignment_boundary_map(
                str(observation["input_text"]),
                str(observation["output_text"]),
            )
            alignment_by_observation[observation_id] = boundaries
        observed_output = normalization_phrase_observed_output(
            observation, start, end, boundaries, label="v4 learner")
        if observed_output is None:
            unaligned += 1
            continue
        if observed_output == candidate["output_text"]:
            support_signatures.add(str(signature["context_signature_id"]))
            continue
        refutes.append(_refute_evidence(
            protocol_manifest_sha256=protocol_manifest_sha256,
            candidate=candidate,
            observation=observation,
            start=start,
            end=end,
            observed_output=observed_output,
            signature=signature,
        ))
    result = tuple(sorted(
        {str(item["evidence_id"]): item for item in refutes}.values(),
        key=lambda item: str(item["evidence_id"]),
    ))
    return result, support_signatures, unaligned


def normalization_recovery_v4_eligible_outputs(
        *,
        protocol_manifest_sha256: str,
        family: dict[str, object],
        candidate: dict[str, object],
        support_evidence: tuple[dict[str, object], ...],
        refutes: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   dict[str, object]]:
    """从 scoped 正负 Evidence 形成禁用态 rule、defeater 与 index。"""
    rule_kind = (
        NORMALIZATION_RECOVERY_V4_TARGET_RULE_KIND
        if candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
        else NORMALIZATION_RECOVERY_V4_SOURCE_RULE_KIND)
    rule_identity = {
        "candidate_id": candidate["candidate_id"],
        "family_id": family["family_id"],
        "input_text": family["input_text"],
        "output_text": candidate["output_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        **_scope_fields(candidate),
    }
    rule_id = _record_id(rule_identity)
    by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)
    for evidence in refutes:
        signature = evidence["context_signature"]
        by_signature[str(signature["context_signature_id"])].append(evidence)
    defeaters = []
    for signature_id, records in sorted(by_signature.items()):
        signature = records[0]["context_signature"]
        identity = {
            "context_signature_id": signature_id,
            "rule_id": rule_id,
            **_scope_fields(candidate),
        }
        defeaters.append({
            **identity,
            "action": "BLOCK_PHRASE_RULE_USE_BACKOFF",
            "defeater_id": _record_id(identity),
            "format_version": 1,
            "left_boundary": signature["left_boundary"],
            "left_context": signature["left_context"],
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_V4_DEFEATER_KIND,
            "refute_evidence_ids": sorted(
                str(item["evidence_id"]) for item in records),
            "right_boundary": signature["right_boundary"],
            "right_context": signature["right_context"],
        })
    defeater_ids = sorted(str(item["defeater_id"]) for item in defeaters)
    support_families = sorted({str(item["source_family"])
                               for item in support_evidence})
    if (candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
            and len(support_families) < 2):
        raise BroadQaExternalDataError(
            "v4 learner target rule 缺少跨 family SUPPORT")
    input_match = (
        "WHOLE_INPUT_EXACT"
        if candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
        else "EXACT_SCALAR_SUBSEQUENCE")
    rule = {
        **rule_identity,
        "application_scope": {
            "candidate_scope_kind": candidate["candidate_scope_kind"],
            "defeater_required": 1,
            "input_match": input_match,
            "source_execution_family": candidate["source_execution_family"],
            "source_execution_policy_scope": candidate[
                "source_execution_policy_scope"],
            "unscoped_execution_allowed": 0,
        },
        "defeater_ids": defeater_ids,
        "equal_length": int(
            len(str(family["input_text"]))
            == len(str(candidate["output_text"]))),
        "format_version": 1,
        "fragment_kinds": family["fragment_kinds"],
        "group_ids": family["group_ids"],
        "license_ids": sorted({str(item["license_id"])
                               for item in support_evidence}),
        "negative_evidence_ids": sorted(
            str(item["evidence_id"]) for item in refutes),
        "positive_evidence_ids": sorted(
            str(item["evidence_id"]) for item in support_evidence),
        "production_enabled": 0,
        "record_kind": rule_kind,
        "rule_id": rule_id,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "source_families": support_families,
        "source_fragment_ids": sorted(
            str(item["fragment_id"]) for item in support_evidence),
        "source_policy_scopes": sorted({
            str(item["source_policy_scope"]) for item in support_evidence}),
        "variable_length": int(
            len(str(family["input_text"]))
            != len(str(candidate["output_text"]))),
    }
    first_scalar = ord(str(family["input_text"])[0])
    target = candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
    index_identity = {
        "candidate_scope_kind": candidate["candidate_scope_kind"],
        "first_scalar": first_scalar,
        "input_scalar_length": len(str(family["input_text"])),
        "rule_id": rule_id,
        "source_execution_family": candidate["source_execution_family"],
        "target_policy_scope": candidate["target_policy_scope"],
    }
    index = {
        **index_identity,
        "candidate_input": family["input_text"],
        "format_version": 1,
        "index_id": _record_id(index_identity),
        "priority_contract": (
            "WHOLE_INPUT_EXACT_ONLY" if target else
            "WHOLE_INPUT_EXACT_THEN_LONGEST_MATCH_THEN_CHARACTER_BACKOFF"),
        "record_kind": (
            NORMALIZATION_RECOVERY_V4_TARGET_OVERLAP_KIND if target
            else NORMALIZATION_RECOVERY_V4_SOURCE_OVERLAP_KIND),
    }
    return rule, tuple(defeaters), index


__all__ = [
    "normalization_recovery_v4_candidate_occurrence_evidence",
    "normalization_recovery_v4_candidate_occurrence_index",
    "normalization_recovery_v4_eligible_outputs",
    "normalization_recovery_v4_family_candidates",
    "normalization_recovery_v4_support_evidence",
]
