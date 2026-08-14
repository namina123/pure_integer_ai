"""Normalization recovery transfer candidate 的不可变记录合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    RECOVERY_TARGET_POLICY_SCOPE,
    SOURCE_POLICY_SCOPES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


RECOVERY_TRANSFER_PROFILE_KIND = (
    "NORMALIZATION_RECOVERY_AUTHORITY_TO_FIREFOX_TRANSFER_PROFILE_V1")
RECOVERY_TRANSFER_REGION_SCOPE = "ZH_CN"
RECOVERY_TARGET_PRECEDENCE = (
    "REGIONAL_EXACT_WHOLE_INPUT",
    "GENERIC_EXACT_WHOLE_INPUT",
    "REGIONAL_CHARACTER_COMPOSITION",
    "GENERIC_CHARACTER_COMPOSITION",
    "PRESERVE_UNKNOWN",
)
RECOVERY_SOURCE_PRECEDENCE = (
    "SOURCE_POLICY_EXACT_PHRASE_OVERRIDE",
    "SOURCE_POLICY_EXACT_OBSERVATION_REPLAY",
    "SOURCE_POLICY_CHARACTER_COMPOSITION",
    "PRESERVE_UNKNOWN",
)


def _sha256(payload: bytes) -> str:
    """返回 profile、program、规则或执行结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: object, *, label: str, empty: bool = False) -> str:
    """核验小写 SHA-256，可按字段允许空引用。"""
    if empty and value == "":
        return value
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def require_text(value: object, *, label: str, empty: bool = False) -> str:
    """核验文本字段并按合同决定是否允许空串。"""
    if not isinstance(value, str) or (not empty and not value):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryTransferProfile:
    """把 authority rule 投影到待证 Firefox target policy 的显式合同。"""

    rule_pack_manifest_sha256: str
    evaluation_protocol_manifest_sha256: str
    authority_policy_scope: str
    candidate_target_policy_scope: str
    regional_scope: str
    target_precedence: tuple[str, ...]
    source_precedence: tuple[str, ...]
    profile_kind: str = RECOVERY_TRANSFER_PROFILE_KIND

    def __post_init__(self) -> None:
        """核验来源/目标 scope、固定区域与执行次序。"""
        require_sha256(
            self.rule_pack_manifest_sha256,
            label="recovery transfer pack manifest")
        require_sha256(
            self.evaluation_protocol_manifest_sha256,
            label="recovery transfer evaluation manifest")
        if (self.authority_policy_scope != RECOVERY_TARGET_POLICY_SCOPE
                or self.candidate_target_policy_scope
                != NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE
                or self.regional_scope != RECOVERY_TRANSFER_REGION_SCOPE
                or self.target_precedence != RECOVERY_TARGET_PRECEDENCE
                or self.source_precedence != RECOVERY_SOURCE_PRECEDENCE
                or self.profile_kind != RECOVERY_TRANSFER_PROFILE_KIND):
            raise BroadQaExternalDataError(
                "recovery transfer profile scope/precedence 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出可冻结且不含 evaluation payload 的 profile。"""
        return {
            "authority_policy_scope": self.authority_policy_scope,
            "candidate_target_policy_scope": self.candidate_target_policy_scope,
            "evaluation_protocol_manifest_sha256": (
                self.evaluation_protocol_manifest_sha256),
            "profile_kind": self.profile_kind,
            "regional_scope": self.regional_scope,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "source_precedence": list(self.source_precedence),
            "target_precedence": list(self.target_precedence),
        }

    def sha256(self) -> str:
        """返回完整 transfer profile identity。"""
        return _sha256(canonical_json_bytes(self.to_dict()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryTargetRule:
    """一条保留原 authority scope 的 generic 或 regional target rule。"""

    input_text: str
    output_text: str
    rule_id: str
    mapping_kind: str
    rule_scope: str
    authority_policy_scope: str
    regional_scope: str = ""

    def __post_init__(self) -> None:
        """核验 exact sequence、authority scope 与区域限制。"""
        if (not self.input_text or not self.output_text
                or self.mapping_kind not in {"CHARACTER_INPUT", "PHRASE_INPUT"}
                or self.mapping_kind == "CHARACTER_INPUT"
                and len(self.input_text) != 1
                or self.mapping_kind == "PHRASE_INPUT"
                and len(self.input_text) < 2
                or self.rule_scope not in {"GENERIC", "REGIONAL_ZH_CN"}
                or self.authority_policy_scope != RECOVERY_TARGET_POLICY_SCOPE
                or (self.rule_scope == "GENERIC" and self.regional_scope)
                or (self.rule_scope == "REGIONAL_ZH_CN"
                    and self.regional_scope != RECOVERY_TRANSFER_REGION_SCOPE)):
            raise BroadQaExternalDataError(
                "recovery target rule 字段漂移")
        require_sha256(self.rule_id, label="recovery target rule id")

    def to_dict(self) -> dict[str, object]:
        """导出 target rule 结构。"""
        return {
            "authority_policy_scope": self.authority_policy_scope,
            "input_text": self.input_text,
            "mapping_kind": self.mapping_kind,
            "output_text": self.output_text,
            "regional_scope": self.regional_scope,
            "rule_id": self.rule_id,
            "rule_scope": self.rule_scope,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoverySourceReplay:
    """一条 exact source-policy observation replay。"""

    source_policy_scope: str
    input_text: str
    output_text: str
    evidence_id: str
    observation_id: str
    authority_role: str
    conflict_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """核验来源 policy、Evidence identity 与冲突引用。"""
        if (self.source_policy_scope not in SOURCE_POLICY_SCOPES
                or not self.input_text or not self.output_text
                or self.authority_role not in {
                    "GENERIC_T2S_EVIDENCE",
                    "REGIONAL_ZH_CN_EXACT_AUTHORITY",
                }
                or self.conflict_ids != tuple(sorted(set(self.conflict_ids)))):
            raise BroadQaExternalDataError(
                "recovery source replay 字段漂移")
        require_sha256(
            self.evidence_id, label="recovery source replay evidence")
        require_sha256(
            self.observation_id, label="recovery source replay observation")
        for value in self.conflict_ids:
            require_sha256(value, label="recovery source replay conflict")

    def to_dict(self) -> dict[str, object]:
        """导出 source replay 结构。"""
        return {
            "authority_role": self.authority_role,
            "conflict_ids": list(self.conflict_ids),
            "evidence_id": self.evidence_id,
            "input_text": self.input_text,
            "observation_id": self.observation_id,
            "output_text": self.output_text,
            "source_policy_scope": self.source_policy_scope,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryPhraseOverride:
    """一个 exact source-policy、exact whole-input phrase override。"""

    source_policy_scope: str
    input_text: str
    base_output: str
    output_text: str
    rule_id: str
    support_evidence_id: str
    refute_evidence_id: str

    def __post_init__(self) -> None:
        """要求 phrase 非恒等、scope 完整且 Evidence 双引用存在。"""
        if (self.source_policy_scope not in SOURCE_POLICY_SCOPES
                or len(self.input_text) < 2 or not self.base_output
                or not self.output_text
                or self.base_output == self.output_text):
            raise BroadQaExternalDataError(
                "recovery phrase override 字段漂移")
        require_sha256(self.rule_id, label="recovery phrase override id")
        require_sha256(
            self.support_evidence_id,
            label="recovery phrase support evidence")
        require_sha256(
            self.refute_evidence_id,
            label="recovery phrase refute evidence")

    def to_dict(self) -> dict[str, object]:
        """导出 source phrase override。"""
        return {
            "base_output": self.base_output,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "refute_evidence_id": self.refute_evidence_id,
            "rule_id": self.rule_id,
            "source_policy_scope": self.source_policy_scope,
            "support_evidence_id": self.support_evidence_id,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryConflict:
    """一个不得无 scope 执行的 family authority conflict。"""

    input_text: str
    conflict_id: str
    conflict_kind: str
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """核验冲突种类、输入和完整 observation identity。"""
        if (not self.input_text
                or self.conflict_kind not in {
                    "INTRA_FAMILY_CONFLICT", "SOURCE_FAMILY_CONFLICT"}
                or not self.observation_ids
                or self.observation_ids
                != tuple(sorted(set(self.observation_ids)))):
            raise BroadQaExternalDataError(
                "recovery conflict 字段漂移")
        require_sha256(self.conflict_id, label="recovery conflict id")
        for value in self.observation_ids:
            require_sha256(value, label="recovery conflict observation")

    def to_dict(self) -> dict[str, object]:
        """导出 conflict 结构。"""
        return {
            "conflict_id": self.conflict_id,
            "conflict_kind": self.conflict_kind,
            "input_text": self.input_text,
            "observation_ids": list(self.observation_ids),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryCandidateProgram:
    """绑定 pack、transfer profile 与四类执行记录的禁用态 program。"""

    transfer_profile: NormalizationRecoveryTransferProfile
    generic_rules: tuple[NormalizationRecoveryTargetRule, ...]
    regional_rules: tuple[NormalizationRecoveryTargetRule, ...]
    source_replays: tuple[NormalizationRecoverySourceReplay, ...]
    phrase_overrides: tuple[NormalizationRecoveryPhraseOverride, ...]
    conflicts: tuple[NormalizationRecoveryConflict, ...]
    production_enabled: int = 0

    def __post_init__(self) -> None:
        """核验所有索引键排序、唯一性、scope 和禁用态。"""
        if (not isinstance(
                self.transfer_profile, NormalizationRecoveryTransferProfile)
                or not self.generic_rules or not self.regional_rules
                or not self.source_replays or not self.phrase_overrides
                or not self.conflicts
                or type(self.production_enabled) is not int
                or self.production_enabled != 0):
            raise BroadQaExternalDataError(
                "recovery candidate program 边界漂移")
        generic_keys = tuple(item.input_text for item in self.generic_rules)
        regional_keys = tuple(item.input_text for item in self.regional_rules)
        replay_keys = tuple(
            (item.source_policy_scope, item.input_text)
            for item in self.source_replays)
        phrase_keys = tuple(
            (item.source_policy_scope, item.input_text)
            for item in self.phrase_overrides)
        conflict_keys = tuple(item.input_text for item in self.conflicts)
        if (generic_keys != tuple(sorted(set(generic_keys)))
                or regional_keys != tuple(sorted(set(regional_keys)))
                or replay_keys != tuple(sorted(set(replay_keys)))
                or phrase_keys != tuple(sorted(set(phrase_keys)))
                or conflict_keys != tuple(sorted(set(conflict_keys)))):
            raise BroadQaExternalDataError(
                "recovery candidate program 索引排序/identity 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出完整 program identity。"""
        return {
            "conflicts": [item.to_dict() for item in self.conflicts],
            "generic_rules": [item.to_dict() for item in self.generic_rules],
            "phrase_overrides": [
                item.to_dict() for item in self.phrase_overrides],
            "production_enabled": self.production_enabled,
            "regional_rules": [
                item.to_dict() for item in self.regional_rules],
            "source_replays": [
                item.to_dict() for item in self.source_replays],
            "transfer_profile": self.transfer_profile.to_dict(),
            "transfer_profile_sha256": self.transfer_profile.sha256(),
        }

    def sha256(self) -> str:
        """返回完整 candidate program identity。"""
        return _sha256(canonical_json_bytes(self.to_dict()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryExecutionResult:
    """一次 scope-aware 执行的输出和完整命中 trace。"""

    input_text: str
    output_text: str
    requested_policy_scope: str
    regional_scope: str
    target_rule_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    phrase_rule_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]
    transfer_profile_id: str
    projection_used: int
    scope_mismatch: int
    unscoped_conflict_blocked: int
    production_enabled: int = 0

    def __post_init__(self) -> None:
        """核验 trace identity、projection 标记与禁用态。"""
        if (not self.input_text or not isinstance(self.output_text, str)
                or not isinstance(self.requested_policy_scope, str)
                or not isinstance(self.regional_scope, str)
                or any(values != tuple(sorted(set(values))) for values in (
                    self.target_rule_ids, self.source_evidence_ids,
                    self.phrase_rule_ids, self.conflict_ids))
                or any(type(value) is not int or value not in {0, 1}
                       for value in (
                           self.projection_used, self.scope_mismatch,
                           self.unscoped_conflict_blocked,
                           self.production_enabled))
                or self.production_enabled != 0
                or (self.projection_used == 1)
                != bool(self.transfer_profile_id)):
            raise BroadQaExternalDataError(
                "recovery candidate execution result 漂移")
        for values in (
                self.target_rule_ids, self.source_evidence_ids,
                self.phrase_rule_ids, self.conflict_ids):
            for value in values:
                require_sha256(value, label="recovery execution trace id")
        require_sha256(
            self.transfer_profile_id,
            label="recovery execution transfer profile",
            empty=True,
        )

    def to_dict(self) -> dict[str, object]:
        """导出规范执行结果。"""
        return {
            "conflict_ids": list(self.conflict_ids),
            "input_text": self.input_text,
            "output_text": self.output_text,
            "phrase_rule_ids": list(self.phrase_rule_ids),
            "production_enabled": self.production_enabled,
            "projection_used": self.projection_used,
            "regional_scope": self.regional_scope,
            "requested_policy_scope": self.requested_policy_scope,
            "scope_mismatch": self.scope_mismatch,
            "source_evidence_ids": list(self.source_evidence_ids),
            "target_rule_ids": list(self.target_rule_ids),
            "transfer_profile_id": self.transfer_profile_id,
            "unscoped_conflict_blocked": self.unscoped_conflict_blocked,
        }

    def sha256(self) -> str:
        """返回规范执行结果摘要。"""
        return _sha256(canonical_json_bytes(self.to_dict()))


__all__ = [
    "NormalizationRecoveryCandidateProgram",
    "NormalizationRecoveryConflict",
    "NormalizationRecoveryExecutionResult",
    "NormalizationRecoveryPhraseOverride",
    "NormalizationRecoverySourceReplay",
    "NormalizationRecoveryTargetRule",
    "NormalizationRecoveryTransferProfile",
    "RECOVERY_SOURCE_PRECEDENCE",
    "RECOVERY_TARGET_PRECEDENCE",
    "RECOVERY_TRANSFER_PROFILE_KIND",
    "RECOVERY_TRANSFER_REGION_SCOPE",
    "require_sha256",
    "require_text",
]
