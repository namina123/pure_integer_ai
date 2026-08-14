"""Recovery-v5 scoped candidate、Evidence、rule 与 defeater 的纯派生。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_phrase_learning import (
    normalization_phrase_alignment_boundary_map,
    normalization_phrase_context_signature,
    normalization_phrase_observed_output,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V5_EVIDENCE_KIND,
    NORMALIZATION_RECOVERY_V5_SOURCE_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V5_SOURCE_RULE_KIND,
    NORMALIZATION_RECOVERY_V5_TARGET_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V5_TARGET_RULE_KIND,
    normalization_recovery_v5_rule_class,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
    V5_SOURCE_FAMILIES,
    V5_SOURCE_POLICY_BY_FAMILY,
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


def _rule_class(fragment_kind: str, equal_length: int) -> str:
    """把非法 fragment class 转成统一外部数据错误。"""
    try:
        return normalization_recovery_v5_rule_class(
            fragment_kind, equal_length=equal_length)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "v5 learner fragment class 漂移") from error


def _structure_token_variants(
        fragments: tuple[dict[str, object], ...],
        ) -> list[list[str]]:
    """保留全部互异结构 token 序，不把标记结构压成纯文本。"""
    variants = set()
    for fragment in fragments:
        tokens = fragment.get("structure_tokens", [])
        if (not isinstance(tokens, list)
                or any(not isinstance(item, str) for item in tokens)):
            raise BroadQaExternalDataError(
                "v5 learner fragment structure token 漂移")
        variants.add(tuple(tokens))
    return [list(value) for value in sorted(variants)]


def _validate_target_authority(
        group: dict[str, object],
        variant: dict[str, object],
        ) -> None:
    """逐 rule class 复核 target group 的冻结 authority basis。"""
    fragment_kind = str(group["fragment_kind"])
    equal_length = int(variant["equal_length"])
    families = variant["source_families"]
    counts = variant["source_family_support_counts"]
    basis = group["authority_basis"]
    if (not isinstance(families, list)
            or any(item not in V5_SOURCE_FAMILIES for item in families)
            or not isinstance(counts, dict)
            or set(counts) != set(families)
            or any(type(value) is not int or value <= 0
                   for value in counts.values())):
        raise BroadQaExternalDataError(
            "v5 learner target source authority schema 漂移")
    if fragment_kind == "WHOLE_INPUT" and equal_length == 0:
        replicated_two_family = (
            len(families) >= 2
            and all(counts[family] >= 2 for family in families))
        if (basis != "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS"
                or not (len(families) >= 3 or replicated_two_family)):
            raise BroadQaExternalDataError(
                "v5 learner variable whole-input authority 漂移")
        return
    expected_basis = (
        "EQUAL_LENGTH_WHOLE_INPUT_TWO_FAMILY_CONSENSUS"
        if fragment_kind == "WHOLE_INPUT"
        else "LOCAL_OR_CONTEXT_TWO_FAMILY_CONSENSUS")
    if basis != expected_basis or len(families) < 2:
        raise BroadQaExternalDataError(
            "v5 learner target authority basis 漂移")


def normalization_recovery_v5_group_candidates(
        *,
        protocol_manifest_sha256: str,
        family_id: str,
        group: dict[str, object],
        fragment_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """按单个 group 的 authority 形成 target 或 source-only 假设。"""
    variants = group["output_variants"]
    if len(variants) != 1 or not group["disposition"].endswith("CANDIDATE"):
        return ()
    variant = variants[0]
    try:
        support_fragments = tuple(
            fragment_by_id[str(value)] for value in variant["fragment_ids"])
    except KeyError as error:
        raise BroadQaExternalDataError(
            "v5 learner candidate fragment 缺失") from error
    if (not support_fragments
            or any(item["fragment_kind"] != group["fragment_kind"]
                   or item["input_text"] != group["input_text"]
                   or item["output_text"] != variant["output_text"]
                   for item in support_fragments)):
        raise BroadQaExternalDataError(
            "v5 learner candidate fragment/group 漂移")
    equal_lengths = {int(item["equal_length"]) for item in support_fragments}
    if len(equal_lengths) != 1 or equal_lengths != {variant["equal_length"]}:
        raise BroadQaExternalDataError(
            "v5 learner candidate length fact 漂移")
    equal_length = equal_lengths.pop()
    rule_class = _rule_class(str(group["fragment_kind"]), equal_length)
    scope_kind = str(group["candidate_scope_kind"])
    source_family = ""
    source_policy = ""
    target_scope = ""
    if scope_kind == "TARGET_CROSS_FAMILY":
        if (group["disposition"] != "CROSS_FAMILY_CONSENSUS_CANDIDATE"
                or group["target_policy_scope"]
                != RECOVERY_V5_TARGET_POLICY_SCOPE):
            raise BroadQaExternalDataError(
                "v5 learner target group scope 漂移")
        _validate_target_authority(group, variant)
        target_scope = RECOVERY_V5_TARGET_POLICY_SCOPE
    elif scope_kind == "SOURCE_ONLY":
        families = variant["source_families"]
        basis = group["authority_basis"]
        expected_basis = (
            "REPEATED_SOURCE_WHOLE_INPUT"
            if group["fragment_kind"] == "WHOLE_INPUT"
            else "REPEATED_SOURCE_CONTEXT")
        if (group["disposition"] != "SOURCE_SCOPED_CANDIDATE"
                or group["target_policy_scope"] != ""
                or basis != expected_basis
                or not isinstance(families, list) or len(families) != 1
                or families[0] not in V5_SOURCE_POLICY_BY_FAMILY):
            raise BroadQaExternalDataError(
                "v5 learner source group authority 漂移")
        source_family = str(families[0])
        source_policy = V5_SOURCE_POLICY_BY_FAMILY[source_family]
    else:
        raise BroadQaExternalDataError(
            "v5 learner candidate scope kind 漂移")
    identity = {
        "authority_basis": group["authority_basis"],
        "candidate_scope_kind": scope_kind,
        "family_id": family_id,
        "fragment_kind": group["fragment_kind"],
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "output_text": variant["output_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "rule_class": rule_class,
        "source_execution_family": source_family,
        "source_execution_policy_scope": source_policy,
        "target_policy_scope": target_scope,
    }
    return ({
        **identity,
        "candidate_id": _record_id(identity),
        "observed_distinct_source_family_count": group[
            "observed_distinct_source_family_count"],
        "required_distinct_source_family_count": group[
            "required_distinct_source_family_count"],
        "structure_token_variants": _structure_token_variants(
            support_fragments),
        "support_fragment_ids": sorted(
            str(item["fragment_id"]) for item in support_fragments),
    },)


def normalization_recovery_v5_support_evidence(
        *,
        protocol_manifest_sha256: str,
        candidate: dict[str, object],
        fragment: dict[str, object],
        observation: dict[str, object],
        ) -> dict[str, object]:
    """把 candidate scope 内的物化 fragment 绑定为结构保真的 SUPPORT。"""
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
        "group_id": candidate["group_id"],
        "hypothesis_input": candidate["input_text"],
        "hypothesis_output": candidate["output_text"],
        "observation_id": fragment["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "stance": "SUPPORT",
        **_scope_fields(candidate),
    }
    return {
        **identity,
        "authority_basis": candidate["authority_basis"],
        "context_signature": signature,
        "equal_length": fragment["equal_length"],
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "fragment_kind": candidate["fragment_kind"],
        "identity_preservation": observation["identity_preservation"],
        "input_end": fragment["input_end"],
        "input_start": fragment["input_start"],
        "license_id": fragment["license_id"],
        "output_end": fragment["output_end"],
        "output_start": fragment["output_start"],
        "record_kind": NORMALIZATION_RECOVERY_V5_EVIDENCE_KIND,
        "rule_class": candidate["rule_class"],
        "source_commitment": observation["source_commitment"],
        "source_family": fragment["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": fragment["source_policy_scope"],
        "structure_tokens": observation["structure_tokens"],
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
        "group_id": candidate["group_id"],
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
        "authority_basis": candidate["authority_basis"],
        "context_signature": signature,
        "evidence_id": _record_id(identity),
        "format_version": 1,
        "fragment_kind": candidate["fragment_kind"],
        "identity_preservation": observation["identity_preservation"],
        "license_id": observation["license_id"],
        "record_kind": NORMALIZATION_RECOVERY_V5_EVIDENCE_KIND,
        "refute_reason": reason,
        "rule_class": candidate["rule_class"],
        "source_commitment": observation["source_commitment"],
        "source_family": observation["source_family"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
        "source_policy_scope": observation["source_policy_scope"],
        "structure_tokens": observation["structure_tokens"],
    }


def normalization_recovery_v5_candidate_occurrence_index(
        observations: tuple[dict[str, object], ...],
        candidate_inputs: set[str],
        ) -> dict[str, tuple[tuple[str, int, int], ...]]:
    """以确定性 scalar trie 单遍索引全部候选 literal occurrence。"""
    transitions: list[dict[str, int]] = [{}]
    terminals: list[list[str]] = [[]]
    for phrase in sorted(candidate_inputs):
        if not phrase:
            raise BroadQaExternalDataError("v5 learner candidate input 为空")
        node = 0
        for scalar in phrase:
            next_node = transitions[node].get(scalar)
            if next_node is None:
                next_node = len(transitions)
                transitions[node][scalar] = next_node
                transitions.append({})
                terminals.append([])
            node = next_node
        terminals[node].append(phrase)
    occurrences: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for observation in observations:
        observation_id = str(observation["observation_id"])
        text = str(observation["input_text"])
        for start in range(len(text)):
            node = 0
            matches = []
            for end in range(start, len(text)):
                next_node = transitions[node].get(text[end])
                if next_node is None:
                    break
                node = next_node
                matches.extend(terminals[node])
            for phrase in sorted(matches, key=lambda item: (-len(item), item)):
                occurrences[phrase].append(
                    (observation_id, start, start + len(phrase)))
    return {key: tuple(values) for key, values in occurrences.items()}


def normalization_recovery_v5_candidate_occurrence_evidence(
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
            observation, start, end, boundaries, label="v5 learner")
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


def normalization_recovery_v5_eligible_outputs(
        *,
        protocol_manifest_sha256: str,
        candidate: dict[str, object],
        support_evidence: tuple[dict[str, object], ...],
        refutes: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   dict[str, object]]:
    """从 scoped 正负 Evidence 形成禁用态 rule、defeater 与 index。"""
    if not support_evidence or not refutes:
        raise BroadQaExternalDataError(
            "v5 learner executable candidate 缺少正负 Evidence")
    rule_kind = (
        NORMALIZATION_RECOVERY_V5_TARGET_RULE_KIND
        if candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
        else NORMALIZATION_RECOVERY_V5_SOURCE_RULE_KIND)
    rule_identity = {
        "authority_basis": candidate["authority_basis"],
        "candidate_id": candidate["candidate_id"],
        "family_id": candidate["family_id"],
        "fragment_kind": candidate["fragment_kind"],
        "group_id": candidate["group_id"],
        "input_text": candidate["input_text"],
        "output_text": candidate["output_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "rule_class": candidate["rule_class"],
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
            "rule_class": candidate["rule_class"],
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
            "record_kind": NORMALIZATION_RECOVERY_V5_DEFEATER_KIND,
            "refute_evidence_ids": sorted(
                str(item["evidence_id"]) for item in records),
            "right_boundary": signature["right_boundary"],
            "right_context": signature["right_context"],
        })
    support_families = sorted({str(item["source_family"])
                               for item in support_evidence})
    required = int(candidate["required_distinct_source_family_count"])
    if (candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
            and len(support_families) < required):
        raise BroadQaExternalDataError(
            "v5 learner target rule 缺少 authority SUPPORT")
    input_match = (
        "WHOLE_INPUT_EXACT"
        if candidate["fragment_kind"] == "WHOLE_INPUT"
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
            "structure_match_required": int(any(
                candidate["structure_token_variants"])),
            "structure_token_variants": candidate[
                "structure_token_variants"],
            "unscoped_execution_allowed": 0,
        },
        "defeater_ids": sorted(str(item["defeater_id"])
                               for item in defeaters),
        "equal_length": int(
            len(str(candidate["input_text"]))
            == len(str(candidate["output_text"]))),
        "format_version": 1,
        "license_ids": sorted({str(item["license_id"])
                               for item in support_evidence}),
        "negative_evidence_ids": sorted(
            str(item["evidence_id"]) for item in refutes),
        "observed_distinct_source_family_count": candidate[
            "observed_distinct_source_family_count"],
        "positive_evidence_ids": sorted(
            str(item["evidence_id"]) for item in support_evidence),
        "production_enabled": 0,
        "record_kind": rule_kind,
        "required_distinct_source_family_count": required,
        "rule_id": rule_id,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "source_families": support_families,
        "source_fragment_ids": sorted(
            str(item["fragment_id"]) for item in support_evidence),
        "source_policy_scopes": sorted({
            str(item["source_policy_scope"]) for item in support_evidence}),
        "structure_token_variants": candidate["structure_token_variants"],
        "variable_length": int(
            len(str(candidate["input_text"]))
            != len(str(candidate["output_text"]))),
    }
    first_scalar = ord(str(candidate["input_text"])[0])
    target = candidate["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
    index_identity = {
        "candidate_scope_kind": candidate["candidate_scope_kind"],
        "first_scalar": first_scalar,
        "input_scalar_length": len(str(candidate["input_text"])),
        "rule_class": candidate["rule_class"],
        "rule_id": rule_id,
        "source_execution_family": candidate["source_execution_family"],
        "target_policy_scope": candidate["target_policy_scope"],
    }
    index = {
        **index_identity,
        "candidate_input": candidate["input_text"],
        "format_version": 1,
        "index_id": _record_id(index_identity),
        "input_match": input_match,
        "priority_contract": (
            "WHOLE_INPUT_EXACT_PRECEDES_LONGEST_LOCAL_MATCH"
            if candidate["fragment_kind"] == "WHOLE_INPUT"
            else "LONGEST_MATCH_THEN_RULE_CLASS_THEN_RULE_ID"),
        "record_kind": (
            NORMALIZATION_RECOVERY_V5_TARGET_OVERLAP_KIND if target
            else NORMALIZATION_RECOVERY_V5_SOURCE_OVERLAP_KIND),
    }
    return rule, tuple(defeaters), index


def normalization_recovery_v5_support_family_counts(
        evidence: tuple[dict[str, object], ...],
        ) -> dict[str, int]:
    """返回测试与审计可复用的 SUPPORT family 计数。"""
    return dict(sorted(Counter(
        str(item["source_family"]) for item in evidence).items()))


__all__ = [
    "normalization_recovery_v5_candidate_occurrence_evidence",
    "normalization_recovery_v5_candidate_occurrence_index",
    "normalization_recovery_v5_eligible_outputs",
    "normalization_recovery_v5_group_candidates",
    "normalization_recovery_v5_support_evidence",
    "normalization_recovery_v5_support_family_counts",
]
