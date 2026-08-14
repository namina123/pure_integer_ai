"""从冻结 successor TRAIN protocol 纯派生 Evidence 与禁用态规则记录。

本模块不读取路径、不写 artifact、不导入 evaluation。所有输出只由 protocol
manifest identity、observation、group、context 和 ordered work 决定。
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    SUCCESSOR_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_SUCCESSOR_EVIDENCE_KIND = "NORMALIZATION_SUCCESSOR_EVIDENCE_V1"
NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND = (
    "NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_V1")
NORMALIZATION_SUCCESSOR_CONFLICT_KIND = (
    "NORMALIZATION_SUCCESSOR_SOURCE_POLICY_CONFLICT_V1")
NORMALIZATION_SUCCESSOR_DECISION_KIND = (
    "NORMALIZATION_SUCCESSOR_GROUP_DECISION_V1")
NORMALIZATION_SUCCESSOR_CONTEXT_REPLAY_KIND = (
    "NORMALIZATION_SUCCESSOR_CONTEXT_REPLAY_RESULT_V1")
NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND = (
    "NORMALIZATION_SUCCESSOR_CONTEXT_RULE_V1")
NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES = (
    ("evidence.jsonl", "LEARNED_EVIDENCE", "evidence_id"),
    ("consensus-rules.jsonl", "LEARNED_CONSENSUS_RULES", "rule_id"),
    ("conflict-ledger.jsonl", "LEARNED_CONFLICT_LEDGER", "conflict_id"),
    ("group-decisions.jsonl", "LEARNED_GROUP_DECISIONS", "decision_id"),
    ("context-replays.jsonl", "LEARNED_CONTEXT_REPLAYS", "replay_id"),
    ("context-rules.jsonl", "LEARNED_CONTEXT_RULES", "context_rule_id"),
)
SOURCE_POLICY_SCOPES = (OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE)


def _sha256(payload: bytes) -> str:
    """返回规范记录或来源 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """对协议绑定的完整记录 identity 计算稳定摘要。"""
    return _sha256(canonical_json_bytes(identity))


def _observation_evidence(
        *,
        protocol_manifest_sha256: str,
        observation: dict[str, object],
        group: dict[str, object],
        ) -> dict[str, object]:
    """把一条来源 observation 投影为 SUPPORT Evidence。"""
    consensus = group["group_kind"] == "CROSS_SOURCE_CONSENSUS"
    target_scope = SUCCESSOR_TARGET_POLICY_SCOPE if consensus else ""
    hypothesis_kind = (
        "TARGET_POLICY_MAPPING" if consensus else "SOURCE_POLICY_MAPPING")
    identity = {
        "hypothesis_kind": hypothesis_kind,
        "hypothesis_output": (
            group["consensus_output"] if consensus
            else observation["expected_output"]),
        "input_text": observation["input_text"],
        "observation_id": observation["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_commitment_sha256": _sha256(canonical_json_bytes(
            observation["source_commitment"])),
        "source_policy_scope": observation["source_policy_scope"],
        "stance": "SUPPORT",
        "target_policy_scope": target_scope,
    }
    return {
        **identity,
        "evidence_id": _record_id(identity),
        "evidence_source_scope": observation["evidence_source_scope"],
        "format_version": 1,
        "record_kind": NORMALIZATION_SUCCESSOR_EVIDENCE_KIND,
        "source_commitment": observation["source_commitment"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
    }


def _context_refute_evidence(
        *,
        protocol_manifest_sha256: str,
        context: dict[str, object],
        phrase: dict[str, object],
        ) -> dict[str, object]:
    """为 phrase override 构造反驳逐字符 base replay 的 Evidence。"""
    identity = {
        "context_id": context["context_id"],
        "hypothesis_kind": "SOURCE_POLICY_CONTEXT_BASE_REPLAY",
        "hypothesis_output": context["base_output"],
        "input_text": context["input_text"],
        "observation_id": phrase["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_commitment_sha256": _sha256(canonical_json_bytes(
            phrase["source_commitment"])),
        "source_policy_scope": context["source_policy_scope"],
        "stance": "REFUTE",
        "target_policy_scope": "",
    }
    return {
        **identity,
        "evidence_id": _record_id(identity),
        "evidence_source_scope": phrase["evidence_source_scope"],
        "format_version": 1,
        "record_kind": NORMALIZATION_SUCCESSOR_EVIDENCE_KIND,
        "source_commitment": phrase["source_commitment"],
        "source_pack_manifest_sha256": phrase[
            "source_pack_manifest_sha256"],
    }


def _consensus_rule(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        observation_by_id: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """从两个来源 policy 的同输出 Evidence 形成目标 policy 规则。"""
    observation_ids = list(group["observation_ids"])
    observations = [observation_by_id[str(value)] for value in observation_ids]
    evidence_ids = sorted(
        str(evidence_by_observation[str(value)]["evidence_id"])
        for value in observation_ids)
    identity = {
        "evidence_ids": evidence_ids,
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "output_text": group["consensus_output"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_policy_scopes": sorted(
            str(item["source_policy_scope"]) for item in observations),
        "target_policy_scope": SUCCESSOR_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "application_domain": {
            "input_match": "EXACT_SCALAR_SEQUENCE",
            "normalization_policy_scope": SUCCESSOR_TARGET_POLICY_SCOPE,
        },
        "evidence_source_scopes": sorted(
            str(item["evidence_source_scope"]) for item in observations),
        "format_version": 1,
        "mapping_kind": group["mapping_kind"],
        "production_enabled": 0,
        "record_kind": NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
        "rule_id": _record_id(identity),
        "runtime_state": "LEARNED_PACK_DISABLED",
    }


def _conflict_record(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """保留所有来源 policy 输出，不选择或合并冲突结果。"""
    evidence_ids = sorted(
        str(evidence_by_observation[str(value)]["evidence_id"])
        for value in group["observation_ids"])
    identity = {
        "evidence_ids": evidence_ids,
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_policy_outputs": group["source_policy_outputs"],
        "target_policy_scope": SUCCESSOR_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "conflict_id": _record_id(identity),
        "format_version": 1,
        "production_enabled": 0,
        "record_kind": NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
        "resolution_state": "POLICY_SCOPE_REQUIRED",
        "unscoped_application_allowed": 0,
    }


def _group_decision(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """记录单来源 defer 或共识 identity no-op，禁止伪造学习规则。"""
    if group["group_kind"] == "SINGLE_SOURCE":
        decision_kind = "SINGLE_SOURCE_DEFER"
        target_scope = ""
    elif (group["group_kind"] == "CROSS_SOURCE_CONSENSUS"
          and group["consensus_is_identity"] == 1):
        decision_kind = "IDENTITY_CONSENSUS_NOOP"
        target_scope = SUCCESSOR_TARGET_POLICY_SCOPE
    else:
        raise BroadQaExternalDataError(
            "successor group decision 输入不是 defer/no-op")
    evidence_ids = sorted(
        str(evidence_by_observation[str(value)]["evidence_id"])
        for value in group["observation_ids"])
    identity = {
        "decision_kind": decision_kind,
        "evidence_ids": evidence_ids,
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_policy_outputs": group["source_policy_outputs"],
        "target_policy_scope": target_scope,
    }
    return {
        **identity,
        "decision_id": _record_id(identity),
        "eligible_rule": 0,
        "format_version": 1,
        "production_enabled": 0,
        "record_kind": NORMALIZATION_SUCCESSOR_DECISION_KIND,
    }


def _context_outputs(
        *,
        protocol_manifest_sha256: str,
        context: dict[str, object],
        observation_by_id: dict[str, dict[str, object]],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> tuple[
            dict[str, object],
            dict[str, object] | None,
            dict[str, object] | None,
        ]:
    """构造 context replay，并在 override 时增加 REFUTE 与 exact rule。"""
    phrase_id = str(context["phrase_observation_id"])
    phrase = observation_by_id.get(phrase_id)
    if phrase is None:
        raise BroadQaExternalDataError(
            "successor context phrase observation 缺失")
    base_ids = [str(value) for value in context["base_observation_ids"]]
    if (any(value not in observation_by_id for value in base_ids)
            or phrase["source_policy_scope"] != context["source_policy_scope"]
            or any(observation_by_id[value]["source_policy_scope"]
                   != context["source_policy_scope"] for value in base_ids)):
        raise BroadQaExternalDataError(
            "successor context observation policy/reference 漂移")
    phrase_evidence_id = str(
        evidence_by_observation[phrase_id]["evidence_id"])
    base_evidence_ids = sorted(
        str(evidence_by_observation[value]["evidence_id"])
        for value in base_ids)
    override = context["qualification_kind"] == "SOURCE_REPLAY_OVERRIDE"
    refute = None
    context_rule = None
    if override:
        refute = _context_refute_evidence(
            protocol_manifest_sha256=protocol_manifest_sha256,
            context=context,
            phrase=phrase,
        )
        rule_identity = {
            "base_evidence_ids": base_evidence_ids,
            "base_output": context["base_output"],
            "context_id": context["context_id"],
            "input_text": context["input_text"],
            "observed_output": context["observed_output"],
            "phrase_evidence_id": phrase_evidence_id,
            "protocol_manifest_sha256": protocol_manifest_sha256,
            "refute_evidence_id": refute["evidence_id"],
            "source_policy_scope": context["source_policy_scope"],
        }
        context_rule = {
            **rule_identity,
            "application_domain": {
                "exact_input": context["input_text"],
                "source_policy_scope": context["source_policy_scope"],
            },
            "context_rule_id": _record_id(rule_identity),
            "format_version": 1,
            "production_enabled": 0,
            "record_kind": NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND,
            "runtime_state": "LEARNED_PACK_DISABLED",
            "target_policy_scope": "",
        }
    replay_identity = {
        "base_evidence_ids": base_evidence_ids,
        "base_output": context["base_output"],
        "context_id": context["context_id"],
        "context_rule_id": (
            "" if context_rule is None else context_rule["context_rule_id"]),
        "input_text": context["input_text"],
        "observed_output": context["observed_output"],
        "phrase_evidence_id": phrase_evidence_id,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "qualification_kind": context["qualification_kind"],
        "refute_evidence_id": (
            "" if refute is None else refute["evidence_id"]),
        "source_policy_scope": context["source_policy_scope"],
    }
    replay = {
        **replay_identity,
        "exact_context_required": context["exact_context_required"],
        "format_version": 1,
        "record_kind": NORMALIZATION_SUCCESSOR_CONTEXT_REPLAY_KIND,
        "replay_id": _record_id(replay_identity),
    }
    return replay, refute, context_rule


def _require_work_alignment(
        *,
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> None:
    """要求 ordered work 逐阶段精确覆盖三类 material identity。"""
    expected = (
        [("SOURCE_OBSERVATION_INGEST", "OBSERVATION", item["observation_id"])
         for item in observations]
        + [("CROSS_SOURCE_GROUP_RESOLUTION", "GROUP", item["group_id"])
           for item in groups]
        + [("CONTEXT_REPLAY_RESOLUTION", "CONTEXT", item["context_id"])
           for item in contexts]
    )
    observed = [
        (item.get("phase"), item.get("work_kind"), item.get("record_id"))
        for item in work
    ]
    if (observed != expected
            or [item.get("work_ordinal") for item in work]
            != list(range(len(work)))):
        raise BroadQaExternalDataError(
            "successor learning ordered work/material 漂移")


def normalization_successor_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """机械计算任意冻结 work 前缀已形成的 Evidence 与结果记录数。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError(
            "successor learning processed prefix 非法")
    context_by_id = {str(item["context_id"]): item for item in contexts}
    if len(context_by_id) != len(contexts):
        raise BroadQaExternalDataError(
            "successor learning context identity 重复")
    evidence_count = 0
    result_record_count = 0
    for item in work[:processed_item_count]:
        kind = item["work_kind"]
        if kind == "OBSERVATION":
            evidence_count += 1
        elif kind == "GROUP":
            result_record_count += 1
        elif kind == "CONTEXT":
            context = context_by_id.get(str(item["record_id"]))
            if context is None:
                raise BroadQaExternalDataError(
                    "successor learning work context 缺失")
            result_record_count += 1
            if context["qualification_kind"] == "SOURCE_REPLAY_OVERRIDE":
                evidence_count += 1
                result_record_count += 1
        else:
            raise BroadQaExternalDataError(
                "successor learning work kind 非法")
    return evidence_count, result_record_count


def derive_normalization_successor_learning_outputs(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """从完整 TRAIN protocol 派生唯一 Evidence、规则和账本输出。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="successor learning protocol manifest")
    if (not observations or not groups or not contexts or not work
            or protocol_manifest.get("target_policy_scope")
            != SUCCESSOR_TARGET_POLICY_SCOPE):
        raise BroadQaExternalDataError(
            "successor learning protocol material 为空或 target policy 漂移")
    _require_work_alignment(
        observations=observations, groups=groups, contexts=contexts, work=work)
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    if len(observation_by_id) != len(observations):
        raise BroadQaExternalDataError(
            "successor learning observation identity 重复")
    group_by_observation: dict[str, dict[str, object]] = {}
    for group in groups:
        for observation_id in group["observation_ids"]:
            key = str(observation_id)
            if key not in observation_by_id or key in group_by_observation:
                raise BroadQaExternalDataError(
                    "successor learning group/observation 关系漂移")
            group_by_observation[key] = group
    if set(group_by_observation) != set(observation_by_id):
        raise BroadQaExternalDataError(
            "successor learning group 未覆盖全部 observation")

    observation_evidence = tuple(_observation_evidence(
        protocol_manifest_sha256=protocol_sha,
        observation=item,
        group=group_by_observation[str(item["observation_id"])],
    ) for item in observations)
    evidence_by_observation = {
        str(item["observation_id"]): evidence
        for item, evidence in zip(observations, observation_evidence)
    }
    consensus_rules = []
    conflicts = []
    decisions = []
    for group in groups:
        if (group["group_kind"] == "CROSS_SOURCE_CONSENSUS"
                and group["consensus_is_identity"] == 0):
            consensus_rules.append(_consensus_rule(
                protocol_manifest_sha256=protocol_sha,
                group=group,
                evidence_by_observation=evidence_by_observation,
                observation_by_id=observation_by_id,
            ))
        elif group["group_kind"] == "SOURCE_POLICY_CONFLICT":
            conflicts.append(_conflict_record(
                protocol_manifest_sha256=protocol_sha,
                group=group,
                evidence_by_observation=evidence_by_observation,
            ))
        else:
            decisions.append(_group_decision(
                protocol_manifest_sha256=protocol_sha,
                group=group,
                evidence_by_observation=evidence_by_observation,
            ))
    context_replays = []
    context_refutes = []
    context_rules = []
    for context in contexts:
        replay, refute, context_rule = _context_outputs(
            protocol_manifest_sha256=protocol_sha,
            context=context,
            observation_by_id=observation_by_id,
            evidence_by_observation=evidence_by_observation,
        )
        context_replays.append(replay)
        if refute is not None:
            context_refutes.append(refute)
        if context_rule is not None:
            context_rules.append(context_rule)
    evidence = list(observation_evidence) + context_refutes
    raw_outputs = {
        "evidence.jsonl": evidence,
        "consensus-rules.jsonl": consensus_rules,
        "conflict-ledger.jsonl": conflicts,
        "group-decisions.jsonl": decisions,
        "context-replays.jsonl": context_replays,
        "context-rules.jsonl": context_rules,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        identities = [str(item[identity_key]) for item in values]
        if len(identities) != len(set(identities)):
            raise BroadQaExternalDataError(
                f"successor learning {name} identity 重复")
        outputs[name] = values

    decision_counts = Counter(
        str(item["decision_kind"]) for item in outputs["group-decisions.jsonl"])
    evidence_counts = Counter(
        str(item["stance"]) for item in outputs["evidence.jsonl"])
    summary = {
        "conflict_ledger_count": len(outputs["conflict-ledger.jsonl"]),
        "consensus_rule_count": len(outputs["consensus-rules.jsonl"]),
        "context_override_rule_count": len(outputs["context-rules.jsonl"]),
        "context_replay_count": len(outputs["context-replays.jsonl"]),
        "evidence_count": len(outputs["evidence.jsonl"]),
        "evidence_stance_counts": {
            "REFUTE": evidence_counts["REFUTE"],
            "SUPPORT": evidence_counts["SUPPORT"],
        },
        "identity_consensus_noop_count": decision_counts[
            "IDENTITY_CONSENSUS_NOOP"],
        "result_record_count": sum(
            len(outputs[name]) for name, _role, _identity
            in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES if name != "evidence.jsonl"),
        "single_source_defer_count": decision_counts[
            "SINGLE_SOURCE_DEFER"],
    }
    exact = protocol_manifest["learner_contract"]["exact_output_counts"]
    if (summary["conflict_ledger_count"] != exact["conflict_ledger_count"]
            or summary["consensus_rule_count"]
            != exact["nonidentity_consensus_rule_count"]
            or summary["context_override_rule_count"]
            != exact["context_override_count"]
            or summary["single_source_defer_count"]
            != exact["single_source_defer_count"]
            or summary["context_replay_count"] != len(contexts)
            or summary["evidence_stance_counts"]["SUPPORT"]
            != len(observations)
            or summary["evidence_stance_counts"]["REFUTE"]
            != summary["context_override_rule_count"]
            or len(consensus_rules) + len(conflicts) + len(decisions)
            != len(groups)):
        raise BroadQaExternalDataError(
            "successor learning output contract/count 漂移")
    terminal_counts = normalization_successor_prefix_output_counts(
        work=work, contexts=contexts, processed_item_count=len(work))
    if terminal_counts != (
            summary["evidence_count"], summary["result_record_count"]):
        raise BroadQaExternalDataError(
            "successor learning terminal prefix/output count 漂移")
    return outputs, summary


def normalization_successor_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把六份已排序输出编码为规范 JSONL，并拒绝缺项或空项。"""
    expected = {
        name for name, _role, _identity
        in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES}
    if set(outputs) != expected:
        raise BroadQaExternalDataError(
            "successor learning output 文件角色漂移")
    payloads = {}
    for name, _role, identity_key in NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES:
        values = outputs[name]
        if (not isinstance(values, tuple) or not values
                or [str(item[identity_key]) for item in values]
                != sorted(str(item[identity_key]) for item in values)):
            raise BroadQaExternalDataError(
                f"successor learning {name} 为空或未排序")
        payloads[name] = b"".join(canonical_json_line(item) for item in values)
    return payloads


__all__ = [
    "NORMALIZATION_SUCCESSOR_CONFLICT_KIND",
    "NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND",
    "NORMALIZATION_SUCCESSOR_CONTEXT_REPLAY_KIND",
    "NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND",
    "NORMALIZATION_SUCCESSOR_DECISION_KIND",
    "NORMALIZATION_SUCCESSOR_EVIDENCE_KIND",
    "NORMALIZATION_SUCCESSOR_OUTPUT_FILE_ROLES",
    "derive_normalization_successor_learning_outputs",
    "normalization_successor_output_payloads",
    "normalization_successor_prefix_output_counts",
]
