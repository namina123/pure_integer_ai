"""编译并执行 normalization successor 的 policy-scoped candidate clone。

目标 policy 只消费跨来源共识；来源 policy 只重放各自观察；context rule 仅在
exact input 与 exact source policy 下执行。公开 production gate 始终关闭。
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
    NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
    NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND,
    NORMALIZATION_SUCCESSOR_DECISION_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    SUCCESSOR_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


SOURCE_POLICY_SCOPES = (OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE)


def _sha256(payload: bytes) -> str:
    """返回 clone program、规则或执行结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str, empty: bool = False) -> str:
    """核验小写 SHA-256，可按字段允许空引用。"""
    if empty and value == "":
        return value
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorTargetRule:
    """一个只在冻结目标 normalization policy 下执行的共识规则。"""

    input_text: str
    output_text: str
    rule_id: str
    mapping_kind: str
    target_policy_scope: str

    def __post_init__(self) -> None:
        """核验共识输入输出、identity 与目标 scope。"""
        if (not self.input_text or not self.output_text
                or self.mapping_kind not in {"CHARACTER_INPUT", "PHRASE_INPUT"}
                or self.mapping_kind == "CHARACTER_INPUT"
                and len(self.input_text) != 1
                or self.mapping_kind == "PHRASE_INPUT"
                and len(self.input_text) < 2
                or self.target_policy_scope != SUCCESSOR_TARGET_POLICY_SCOPE):
            raise BroadQaExternalDataError(
                "successor target rule 字段漂移")
        _sha_value(self.rule_id, label="successor target rule id")

    def to_dict(self) -> dict[str, object]:
        """导出可冻结的目标规则结构。"""
        return {
            "input_text": self.input_text,
            "mapping_kind": self.mapping_kind,
            "output_text": self.output_text,
            "rule_id": self.rule_id,
            "target_policy_scope": self.target_policy_scope,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorSourceReplay:
    """一个 exact-input、exact-source-policy 的训练来源重放记录。"""

    source_policy_scope: str
    input_text: str
    output_text: str
    source_record_id: str
    source_record_kind: str
    conflict_id: str = ""

    def __post_init__(self) -> None:
        """核验来源 scope、输入输出和可选冲突引用。"""
        if (self.source_policy_scope not in SOURCE_POLICY_SCOPES
                or not self.input_text or not self.output_text
                or self.source_record_kind not in {
                    NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
                    NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
                    NORMALIZATION_SUCCESSOR_DECISION_KIND,
                }):
            raise BroadQaExternalDataError(
                "successor source replay 字段漂移")
        _sha_value(self.source_record_id, label="successor source record id")
        _sha_value(
            self.conflict_id,
            label="successor source replay conflict id",
            empty=True,
        )
        if ((self.source_record_kind == NORMALIZATION_SUCCESSOR_CONFLICT_KIND)
                != bool(self.conflict_id)):
            raise BroadQaExternalDataError(
                "successor source replay conflict 引用漂移")

    def to_dict(self) -> dict[str, object]:
        """导出来源 replay 结构。"""
        return {
            "conflict_id": self.conflict_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "source_policy_scope": self.source_policy_scope,
            "source_record_id": self.source_record_id,
            "source_record_kind": self.source_record_kind,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorContextRule:
    """一个 source-policy 局部 exact context override。"""

    source_policy_scope: str
    input_text: str
    base_output: str
    observed_output: str
    context_rule_id: str

    def __post_init__(self) -> None:
        """要求 context 为非 identity override 且 scope 完整。"""
        if (self.source_policy_scope not in SOURCE_POLICY_SCOPES
                or not self.input_text or not self.base_output
                or not self.observed_output
                or self.base_output == self.observed_output):
            raise BroadQaExternalDataError(
                "successor context rule 字段漂移")
        _sha_value(self.context_rule_id, label="successor context rule id")

    def to_dict(self) -> dict[str, object]:
        """导出可执行 context override。"""
        return {
            "base_output": self.base_output,
            "context_rule_id": self.context_rule_id,
            "input_text": self.input_text,
            "observed_output": self.observed_output,
            "source_policy_scope": self.source_policy_scope,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorCandidateProgram:
    """绑定 pack、target/source scope 和所有确定性索引输入的 clone program。"""

    rule_pack_manifest_sha256: str
    target_policy_scope: str
    target_rules: tuple[NormalizationSuccessorTargetRule, ...]
    source_replays: tuple[NormalizationSuccessorSourceReplay, ...]
    context_rules: tuple[NormalizationSuccessorContextRule, ...]
    declared_conflict_ids: tuple[str, ...]
    production_enabled: int = 0

    def __post_init__(self) -> None:
        """核验规则排序、唯一性、scope 和禁用态。"""
        _sha_value(
            self.rule_pack_manifest_sha256,
            label="successor candidate pack manifest")
        if (self.target_policy_scope != SUCCESSOR_TARGET_POLICY_SCOPE
                or not self.target_rules or not self.source_replays
                or not self.context_rules
                or self.production_enabled != 0
                or type(self.production_enabled) is not int):
            raise BroadQaExternalDataError(
                "successor candidate program 边界漂移")
        target_keys = tuple(item.input_text for item in self.target_rules)
        replay_keys = tuple(
            (item.source_policy_scope, item.input_text)
            for item in self.source_replays)
        context_keys = tuple(
            (item.source_policy_scope, item.input_text)
            for item in self.context_rules)
        if (target_keys != tuple(sorted(set(target_keys)))
                or replay_keys != tuple(sorted(set(replay_keys)))
                or context_keys != tuple(sorted(set(context_keys)))
                or self.declared_conflict_ids
                != tuple(sorted(set(self.declared_conflict_ids)))
                or not self.declared_conflict_ids):
            raise BroadQaExternalDataError(
                "successor candidate program 规则 identity/排序漂移")
        for value in self.declared_conflict_ids:
            _sha_value(value, label="successor candidate conflict id")

    def to_dict(self) -> dict[str, object]:
        """导出完整 candidate program identity。"""
        return {
            "context_rules": [item.to_dict() for item in self.context_rules],
            "declared_conflict_ids": list(self.declared_conflict_ids),
            "production_enabled": self.production_enabled,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "source_replays": [item.to_dict() for item in self.source_replays],
            "target_policy_scope": self.target_policy_scope,
            "target_rules": [item.to_dict() for item in self.target_rules],
        }

    def sha256(self) -> str:
        """返回完整 clone program 摘要。"""
        return _sha256(canonical_json_bytes(self.to_dict()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorExecutionResult:
    """一次 policy-scoped normalization 的确定性输出与命中 trace。"""

    input_text: str
    output_text: str
    requested_policy_scope: str
    target_rule_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    context_rule_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    scope_mismatch: int
    unscoped_conflict_blocked: int
    production_enabled: int = 0

    def __post_init__(self) -> None:
        """核验执行 trace 的有序唯一 identity 和禁用态。"""
        if (not self.input_text or not isinstance(self.output_text, str)
                or not isinstance(self.requested_policy_scope, str)
                or any(values != tuple(sorted(set(values))) for values in (
                    self.target_rule_ids, self.source_record_ids,
                    self.context_rule_ids, self.conflict_ids))
                or any(type(value) is not int or value not in {0, 1}
                       for value in (
                           self.scope_mismatch,
                           self.unscoped_conflict_blocked,
                           self.production_enabled))
                or self.production_enabled != 0):
            raise BroadQaExternalDataError(
                "successor candidate execution result 漂移")
        for values in (
                self.target_rule_ids, self.source_record_ids,
                self.context_rule_ids, self.conflict_ids):
            for value in values:
                _sha_value(value, label="successor execution trace id")

    def to_dict(self) -> dict[str, object]:
        """导出规范执行结果。"""
        return {
            "conflict_ids": list(self.conflict_ids),
            "context_rule_ids": list(self.context_rule_ids),
            "input_text": self.input_text,
            "output_text": self.output_text,
            "production_enabled": self.production_enabled,
            "requested_policy_scope": self.requested_policy_scope,
            "scope_mismatch": self.scope_mismatch,
            "source_record_ids": list(self.source_record_ids),
            "target_rule_ids": list(self.target_rule_ids),
            "unscoped_conflict_blocked": self.unscoped_conflict_blocked,
        }

    def sha256(self) -> str:
        """返回规范执行结果摘要。"""
        return _sha256(canonical_json_bytes(self.to_dict()))


def _source_outputs(record: dict[str, object]) -> list[dict[str, object]]:
    """核验并返回 group record 的逐 policy 输出。"""
    values = record.get("source_policy_outputs")
    if (not isinstance(values, list) or not values
            or any(not isinstance(item, dict)
                   or item.get("source_policy_scope") not in SOURCE_POLICY_SCOPES
                   or not isinstance(item.get("expected_output"), str)
                   or not item["expected_output"]
                   for item in values)):
        raise BroadQaExternalDataError(
            "successor candidate source policy outputs 漂移")
    return values


def compile_normalization_successor_candidate(
        *,
        rule_pack_manifest: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationSuccessorCandidateProgram:
    """从严格回读 pack 编译目标规则、来源 replay、冲突和 context 索引。"""
    pack_sha = _sha_value(
        rule_pack_manifest.get("manifest_sha256"),
        label="successor candidate rule pack manifest")
    if (rule_pack_manifest.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or rule_pack_manifest.get("production_enabled") != 0
            or rule_pack_manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError(
            "successor candidate pack 已启用或边界漂移")
    target_rules = []
    replay_by_key: dict[tuple[str, str], NormalizationSuccessorSourceReplay] = {}
    conflicts = []
    for record in outputs.get("consensus-rules.jsonl", ()):
        if record.get("record_kind") != NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND:
            raise BroadQaExternalDataError(
                "successor candidate consensus record kind 漂移")
        target = NormalizationSuccessorTargetRule(
            str(record["input_text"]),
            str(record["output_text"]),
            str(record["rule_id"]),
            str(record["mapping_kind"]),
            str(record["target_policy_scope"]),
        )
        target_rules.append(target)
        for policy in record["source_policy_scopes"]:
            replay = NormalizationSuccessorSourceReplay(
                str(policy), target.input_text, target.output_text,
                target.rule_id, NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND)
            replay_by_key[(replay.source_policy_scope, replay.input_text)] = replay
    for record in outputs.get("conflict-ledger.jsonl", ()):
        if record.get("record_kind") != NORMALIZATION_SUCCESSOR_CONFLICT_KIND:
            raise BroadQaExternalDataError(
                "successor candidate conflict record kind 漂移")
        conflict_id = str(record["conflict_id"])
        conflicts.append(conflict_id)
        for item in _source_outputs(record):
            replay = NormalizationSuccessorSourceReplay(
                str(item["source_policy_scope"]),
                str(record["input_text"]),
                str(item["expected_output"]),
                conflict_id,
                NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
                conflict_id,
            )
            replay_by_key[(replay.source_policy_scope, replay.input_text)] = replay
    for record in outputs.get("group-decisions.jsonl", ()):
        if record.get("record_kind") != NORMALIZATION_SUCCESSOR_DECISION_KIND:
            raise BroadQaExternalDataError(
                "successor candidate decision record kind 漂移")
        decision_id = str(record["decision_id"])
        for item in _source_outputs(record):
            replay = NormalizationSuccessorSourceReplay(
                str(item["source_policy_scope"]),
                str(record["input_text"]),
                str(item["expected_output"]),
                decision_id,
                NORMALIZATION_SUCCESSOR_DECISION_KIND,
            )
            key = (replay.source_policy_scope, replay.input_text)
            existing = replay_by_key.get(key)
            if existing is not None and existing.output_text != replay.output_text:
                raise BroadQaExternalDataError(
                    "successor candidate source replay 输出冲突")
            replay_by_key[key] = replay
    context_rules = []
    for record in outputs.get("context-rules.jsonl", ()):
        if record.get("record_kind") != NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND:
            raise BroadQaExternalDataError(
                "successor candidate context record kind 漂移")
        context = NormalizationSuccessorContextRule(
            str(record["source_policy_scope"]),
            str(record["input_text"]),
            str(record["base_output"]),
            str(record["observed_output"]),
            str(record["context_rule_id"]),
        )
        replay = replay_by_key.get(
            (context.source_policy_scope, context.input_text))
        if replay is None or replay.output_text != context.observed_output:
            raise BroadQaExternalDataError(
                "successor context rule/source replay 未闭合")
        context_rules.append(context)
    return NormalizationSuccessorCandidateProgram(
        pack_sha,
        SUCCESSOR_TARGET_POLICY_SCOPE,
        tuple(sorted(target_rules, key=lambda item: item.input_text)),
        tuple(sorted(
            replay_by_key.values(),
            key=lambda item: (item.source_policy_scope, item.input_text))),
        tuple(sorted(
            context_rules,
            key=lambda item: (item.source_policy_scope, item.input_text))),
        tuple(sorted(conflicts)),
        0,
    )


def _exact_target_rule(
        program: NormalizationSuccessorCandidateProgram,
        input_text: str,
        ) -> NormalizationSuccessorTargetRule | None:
    """用二分索引查找 exact-input target rule。"""
    index = bisect_left(
        program.target_rules, input_text, key=lambda item: item.input_text)
    if (index < len(program.target_rules)
            and program.target_rules[index].input_text == input_text):
        return program.target_rules[index]
    return None


def _exact_source_replay(
        program: NormalizationSuccessorCandidateProgram,
        policy: str,
        input_text: str,
        ) -> NormalizationSuccessorSourceReplay | None:
    """用二分索引查找 exact policy/input 来源重放。"""
    key = (policy, input_text)
    index = bisect_left(
        program.source_replays, key,
        key=lambda item: (item.source_policy_scope, item.input_text))
    if index < len(program.source_replays):
        value = program.source_replays[index]
        if (value.source_policy_scope, value.input_text) == key:
            return value
    return None


def _exact_context_rule(
        program: NormalizationSuccessorCandidateProgram,
        policy: str,
        input_text: str,
        ) -> NormalizationSuccessorContextRule | None:
    """用二分索引查找 exact policy/input context rule。"""
    key = (policy, input_text)
    index = bisect_left(
        program.context_rules, key,
        key=lambda item: (item.source_policy_scope, item.input_text))
    if index < len(program.context_rules):
        value = program.context_rules[index]
        if (value.source_policy_scope, value.input_text) == key:
            return value
    return None


def execute_normalization_successor_candidate(
        program: NormalizationSuccessorCandidateProgram,
        input_text: str,
        *,
        policy_scope: str,
        ) -> NormalizationSuccessorExecutionResult:
    """按显式 policy scope 执行索引 clone，未知或无 scope 时保持输入。"""
    if (not isinstance(program, NormalizationSuccessorCandidateProgram)
            or not isinstance(input_text, str) or not input_text
            or not isinstance(policy_scope, str)):
        raise BroadQaExternalDataError(
            "successor candidate execution 输入非法")
    target_ids = []
    source_ids = []
    context_ids = []
    conflict_ids = []
    scope_mismatch = 0
    unscoped_blocked = 0
    output = input_text
    if policy_scope == program.target_policy_scope:
        exact = _exact_target_rule(program, input_text)
        if exact is not None:
            output = exact.output_text
            target_ids.append(exact.rule_id)
        else:
            parts = []
            for character in input_text:
                rule = _exact_target_rule(program, character)
                if rule is None or rule.mapping_kind != "CHARACTER_INPUT":
                    parts.append(character)
                else:
                    parts.append(rule.output_text)
                    target_ids.append(rule.rule_id)
            output = "".join(parts)
    elif policy_scope in SOURCE_POLICY_SCOPES:
        context = _exact_context_rule(program, policy_scope, input_text)
        replay = _exact_source_replay(program, policy_scope, input_text)
        if context is not None:
            output = context.observed_output
            context_ids.append(context.context_rule_id)
        elif replay is not None:
            output = replay.output_text
        if replay is not None:
            source_ids.append(replay.source_record_id)
            if replay.conflict_id:
                conflict_ids.append(replay.conflict_id)
    else:
        scope_mismatch = 1
        conflict_inputs = {
            item.input_text for item in program.source_replays if item.conflict_id}
        if input_text in conflict_inputs:
            unscoped_blocked = 1
    return NormalizationSuccessorExecutionResult(
        input_text,
        output,
        policy_scope,
        tuple(sorted(set(target_ids))),
        tuple(sorted(set(source_ids))),
        tuple(sorted(set(context_ids))),
        tuple(sorted(set(conflict_ids))),
        scope_mismatch,
        unscoped_blocked,
        program.production_enabled,
    )


def reference_normalization_successor_candidate(
        program: NormalizationSuccessorCandidateProgram,
        input_text: str,
        *,
        policy_scope: str,
        ) -> NormalizationSuccessorExecutionResult:
    """用线性扫描独立解释同一 program，供 evaluator 对照索引执行。"""
    if policy_scope == program.target_policy_scope:
        exact = next(
            (item for item in program.target_rules
             if item.input_text == input_text), None)
        if exact is not None:
            output = exact.output_text
            target_ids = (exact.rule_id,)
        else:
            target_ids_list = []
            parts = []
            for character in input_text:
                rule = next(
                    (item for item in program.target_rules
                     if item.input_text == character
                     and item.mapping_kind == "CHARACTER_INPUT"), None)
                if rule is None:
                    parts.append(character)
                else:
                    parts.append(rule.output_text)
                    target_ids_list.append(rule.rule_id)
            output = "".join(parts)
            target_ids = tuple(sorted(set(target_ids_list)))
        return NormalizationSuccessorExecutionResult(
            input_text, output, policy_scope, target_ids, (), (), (), 0, 0, 0)
    if policy_scope in SOURCE_POLICY_SCOPES:
        context = next(
            (item for item in program.context_rules
             if item.source_policy_scope == policy_scope
             and item.input_text == input_text), None)
        replay = next(
            (item for item in program.source_replays
             if item.source_policy_scope == policy_scope
             and item.input_text == input_text), None)
        output = (
            context.observed_output if context is not None
            else replay.output_text if replay is not None else input_text)
        return NormalizationSuccessorExecutionResult(
            input_text,
            output,
            policy_scope,
            (),
            () if replay is None else (replay.source_record_id,),
            () if context is None else (context.context_rule_id,),
            () if replay is None or not replay.conflict_id
            else (replay.conflict_id,),
            0,
            0,
            0,
        )
    blocked = int(any(
        item.input_text == input_text and item.conflict_id
        for item in program.source_replays))
    return NormalizationSuccessorExecutionResult(
        input_text, input_text, policy_scope, (), (), (), (), 1, blocked, 0)


__all__ = [
    "NormalizationSuccessorCandidateProgram",
    "NormalizationSuccessorContextRule",
    "NormalizationSuccessorExecutionResult",
    "NormalizationSuccessorSourceReplay",
    "NormalizationSuccessorTargetRule",
    "compile_normalization_successor_candidate",
    "execute_normalization_successor_candidate",
    "reference_normalization_successor_candidate",
]
