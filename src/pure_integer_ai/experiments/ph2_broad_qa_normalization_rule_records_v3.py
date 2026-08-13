"""normalization v3 accepted/rejected 记录及其严格规范字节。

本模块拥有规则候选、SUPPORT accepted rule、context REFUTE rejected trial
三种值结构。跨记录 ledger、checkpoint 和 pack I/O 由上层 pack 模块负责。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    BroadQaNormalizationEvidenceCommitmentV3,
    normalization_contrastive_protocol_scope,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_identity_v3 import (
    normalization_context_defeater,
    normalization_context_trial_hypothesis_key,
    normalization_rule_hypothesis_key,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    SOURCE_INFERENCE_DIRECTIONS,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_rule_pack import (
    SOURCE_INFERENCE_RULE_RUNTIME_STATE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


NORMALIZATION_ACCEPTED_RULE_V3_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ACCEPTED_RULE_V3")
NORMALIZATION_REJECTED_TRIAL_V3_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_REJECTED_TRIAL_V3")
NORMALIZATION_CONTEXT_REJECTION_KIND = "CONTEXTUAL_SOURCE_COUNTEREXAMPLE"


def _sha256(value: object, *, label: str) -> str:
    """要求记录身份为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """要求稳定键为非空严格整数数组或 tuple。"""
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int for item in value)):
        raise BroadQaExternalDataError(f"{label} 必须是严格整数键")
    return tuple(value)


def _identity(
        value: object,
        *,
        label: str,
        object_kind: int,
        ) -> ObjectIdentity:
    """回读指定 object kind 的稳定身份。"""
    try:
        identity = ObjectIdentity.from_stable_key(
            _strict_key(value, label=label))
    except (TypeError, ValueError) as error:
        raise BroadQaExternalDataError(f"{label} 身份非法") from error
    if identity.object_kind != object_kind:
        raise BroadQaExternalDataError(f"{label} object kind 漂移")
    return identity


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaNormalizationRuleCandidateV3:
    """一个由冻结 mapping candidate 约束的 normalization 规则候选。"""

    contrastive_protocol_manifest_sha256: str
    source_pack_manifest_sha256: str
    mapping_candidate_id: str
    input_codepoint: int
    output_codepoint: int
    operator: ObjectIdentity
    operator_version: int
    schema: ObjectIdentity
    direction: str
    application_domain: str
    defeaters: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验来源身份、规则坐标、适用域和非空 defeater 集。"""
        _sha256(
            self.contrastive_protocol_manifest_sha256,
            label="normalization candidate protocol",
        )
        _sha256(
            self.source_pack_manifest_sha256,
            label="normalization candidate source pack",
        )
        _sha256(
            self.mapping_candidate_id,
            label="normalization mapping candidate",
        )
        for label, codepoint in (
                ("input", self.input_codepoint),
                ("output", self.output_codepoint)):
            if (type(codepoint) is not int or not 0 <= codepoint <= 0x10FFFF
                    or 0xD800 <= codepoint <= 0xDFFF):
                raise BroadQaExternalDataError(
                    f"normalization candidate {label} codepoint 非法")
        if (not isinstance(self.operator, ObjectIdentity)
                or self.operator.object_kind != OBJECT_MINIMAL_INSTRUCTION
                or type(self.operator_version) is not int
                or self.operator_version <= 0
                or not isinstance(self.schema, ObjectIdentity)
                or self.schema.object_kind != OBJECT_STRUCTURE_CONCEPT
                or self.direction not in SOURCE_INFERENCE_DIRECTIONS
                or self.application_domain
                != NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN):
            raise BroadQaExternalDataError(
                "normalization candidate rule identity/domain 漂移")
        defeater_keys = tuple(item.stable_key() for item in self.defeaters)
        if (not isinstance(self.defeaters, tuple) or not self.defeaters
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.defeaters)
                or defeater_keys != tuple(sorted(set(defeater_keys)))):
            raise BroadQaExternalDataError(
                "normalization candidate defeaters 必须非空唯一排序")

    def hypothesis(self) -> HypothesisKey:
        """构造绑定 mapping、码点和 protocol scope 的规则 hypothesis。"""
        return normalization_rule_hypothesis_key(
            mapping_candidate_id=self.mapping_candidate_id,
            input_codepoint=self.input_codepoint,
            output_codepoint=self.output_codepoint,
            operator=self.operator,
            schema=self.schema,
            direction=self.direction,
            operator_version=self.operator_version,
            applicability_scope=normalization_contrastive_protocol_scope(
                self.contrastive_protocol_manifest_sha256),
        )

    def trial_hypothesis(self, trial_id: str) -> HypothesisKey:
        """为一个上下文失败 trial 构造独立且可审计的 hypothesis。"""
        return normalization_context_trial_hypothesis_key(
            trial_id=trial_id,
            rule_hypothesis=self.hypothesis(),
        )

    def to_dict(self) -> dict[str, object]:
        """导出字段精确的规则候选。"""
        return {
            "application_domain": self.application_domain,
            "contrastive_protocol_manifest_sha256": (
                self.contrastive_protocol_manifest_sha256),
            "defeater_keys": [
                list(item.stable_key()) for item in self.defeaters],
            "direction": self.direction,
            "input_codepoint": self.input_codepoint,
            "mapping_candidate_id": self.mapping_candidate_id,
            "operator_key": list(self.operator.stable_key()),
            "operator_version": self.operator_version,
            "output_codepoint": self.output_codepoint,
            "schema_key": list(self.schema.stable_key()),
            "source_pack_manifest_sha256": self.source_pack_manifest_sha256,
        }

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaNormalizationRuleCandidateV3":
        """从字段精确 JSON object 恢复规则候选。"""
        expected = {
            "application_domain", "contrastive_protocol_manifest_sha256",
            "defeater_keys", "direction", "input_codepoint",
            "mapping_candidate_id", "operator_key", "operator_version",
            "output_codepoint", "schema_key", "source_pack_manifest_sha256",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or not isinstance(value["defeater_keys"], list)):
            raise BroadQaExternalDataError(
                "normalization candidate 字段漂移")
        return cls(
            value["contrastive_protocol_manifest_sha256"],
            value["source_pack_manifest_sha256"],
            value["mapping_candidate_id"],
            value["input_codepoint"],
            value["output_codepoint"],
            _identity(
                value["operator_key"],
                label="normalization candidate operator",
                object_kind=OBJECT_MINIMAL_INSTRUCTION,
            ),
            value["operator_version"],
            _identity(
                value["schema_key"],
                label="normalization candidate schema",
                object_kind=OBJECT_STRUCTURE_CONCEPT,
            ),
            value["direction"],
            value["application_domain"],
            tuple(_identity(
                item,
                label="normalization candidate defeater",
                object_kind=OBJECT_CONCEPT,
            ) for item in value["defeater_keys"]),
        )


def _validate_commitments(
        *,
        candidate: BroadQaNormalizationRuleCandidateV3,
        commitments: tuple[BroadQaNormalizationEvidenceCommitmentV3, ...],
        expected_hypothesis: HypothesisKey,
        expected_qualification: str,
        label: str,
        ) -> None:
    """核验 commitment 唯一排序并绑定指定 hypothesis 和资格。"""
    if (not isinstance(commitments, tuple) or not commitments
            or any(not isinstance(
                item, BroadQaNormalizationEvidenceCommitmentV3)
                   for item in commitments)):
        raise BroadQaExternalDataError(
            f"normalization {label} Evidence commitments 非法")
    evidence_keys = tuple(item.evidence_key for item in commitments)
    if evidence_keys != tuple(sorted(set(evidence_keys))):
        raise BroadQaExternalDataError(
            f"normalization {label} Evidence 必须唯一规范排序")
    expected_stance = (
        EVIDENCE_SUPPORT
        if expected_qualification == "SOURCE_REPLAY_SUPPORT"
        else EVIDENCE_REFUTE)
    for commitment in commitments:
        evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
        if (commitment.qualification_kind != expected_qualification
                or evidence.hypothesis != expected_hypothesis
                or evidence.stance != expected_stance
                or commitment.contrastive_protocol_manifest_sha256
                != candidate.contrastive_protocol_manifest_sha256
                or commitment.source_pack_manifest_sha256
                != candidate.source_pack_manifest_sha256
                or commitment.candidate_id != candidate.mapping_candidate_id
                or commitment.input_codepoint != candidate.input_codepoint
                or commitment.candidate_output_codepoint
                != candidate.output_codepoint):
            raise BroadQaExternalDataError(
                f"normalization {label} Evidence candidate/stance 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaNormalizationRejectedTrialV3:
    """一个绑定独立 trial hypothesis 的 context REFUTE 记录。"""

    candidate: BroadQaNormalizationRuleCandidateV3
    trial_id: str
    context_defeater: ObjectIdentity
    candidate_parameters_sha256: str
    evidence_commitments: tuple[
        BroadQaNormalizationEvidenceCommitmentV3, ...]
    rejection_kind: str = NORMALIZATION_CONTEXT_REJECTION_KIND

    def __post_init__(self) -> None:
        """要求 context trial、defeater 和 REFUTE Evidence 精确关联。"""
        if not isinstance(self.candidate, BroadQaNormalizationRuleCandidateV3):
            raise TypeError("normalization rejected candidate 类型非法")
        _sha256(self.trial_id, label="normalization rejected trial")
        _sha256(
            self.candidate_parameters_sha256,
            label="normalization rejected parameters",
        )
        if (not isinstance(self.context_defeater, ObjectIdentity)
                or self.context_defeater.object_kind != OBJECT_CONCEPT
                or self.context_defeater not in self.candidate.defeaters
                or self.context_defeater
                != normalization_context_defeater(self.trial_id)
                or self.rejection_kind != NORMALIZATION_CONTEXT_REJECTION_KIND):
            raise BroadQaExternalDataError(
                "normalization rejected trial defeater/kind 漂移")
        _validate_commitments(
            candidate=self.candidate,
            commitments=self.evidence_commitments,
            expected_hypothesis=self.candidate.trial_hypothesis(self.trial_id),
            expected_qualification="SOURCE_REPLAY_REFUTE",
            label="rejected trial",
        )
        if any(item.trial_id != self.trial_id
               for item in self.evidence_commitments):
            raise BroadQaExternalDataError(
                "normalization rejected trial Evidence identity 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出 rejected trial 与独立 REFUTE Evidence。"""
        return {
            "artifact_kind": NORMALIZATION_REJECTED_TRIAL_V3_KIND,
            "candidate": self.candidate.to_dict(),
            "candidate_parameters_sha256": self.candidate_parameters_sha256,
            "context_defeater_key": list(
                self.context_defeater.stable_key()),
            "evidence_commitments": [
                item.to_dict() for item in self.evidence_commitments],
            "format_version": 1,
            "rejection_kind": self.rejection_kind,
            "trial_id": self.trial_id,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 rejected trial。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 rejected trial 的规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaNormalizationRejectedTrialV3":
        """从字段精确 JSON object 恢复 rejected trial。"""
        expected = {
            "artifact_kind", "candidate", "candidate_parameters_sha256",
            "context_defeater_key", "evidence_commitments", "format_version",
            "rejection_kind", "trial_id",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"]
                != NORMALIZATION_REJECTED_TRIAL_V3_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or not isinstance(value["evidence_commitments"], list)):
            raise BroadQaExternalDataError(
                "normalization rejected trial 字段漂移")
        return cls(
            BroadQaNormalizationRuleCandidateV3.from_dict(value["candidate"]),
            value["trial_id"],
            _identity(
                value["context_defeater_key"],
                label="normalization context defeater",
                object_kind=OBJECT_CONCEPT,
            ),
            value["candidate_parameters_sha256"],
            tuple(BroadQaNormalizationEvidenceCommitmentV3.from_dict(item)
                  for item in value["evidence_commitments"]),
            value["rejection_kind"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaNormalizationAcceptedRuleV3:
    """只有 SUPPORT 的 accepted mapping rule 及其 context rejection 引用。"""

    candidate: BroadQaNormalizationRuleCandidateV3
    evidence_commitments: tuple[
        BroadQaNormalizationEvidenceCommitmentV3, ...]
    rejection_record_sha256s: tuple[str, ...]
    runtime_state: str = SOURCE_INFERENCE_RULE_RUNTIME_STATE
    production_enabled: int = 0
    identity_dispatch: int = 0

    def __post_init__(self) -> None:
        """要求 accepted hypothesis 为 SUPPORT 且保持生产禁用。"""
        if not isinstance(self.candidate, BroadQaNormalizationRuleCandidateV3):
            raise TypeError("normalization accepted candidate 类型非法")
        _validate_commitments(
            candidate=self.candidate,
            commitments=self.evidence_commitments,
            expected_hypothesis=self.candidate.hypothesis(),
            expected_qualification="SOURCE_REPLAY_SUPPORT",
            label="accepted rule",
        )
        if (not isinstance(self.rejection_record_sha256s, tuple)
                or not self.rejection_record_sha256s
                or self.rejection_record_sha256s != tuple(sorted(set(
                    self.rejection_record_sha256s)))):
            raise BroadQaExternalDataError(
                "normalization accepted rejection refs 必须非空唯一排序")
        for value in self.rejection_record_sha256s:
            _sha256(value, label="normalization accepted rejection ref")
        if (self.runtime_state != SOURCE_INFERENCE_RULE_RUNTIME_STATE
                or type(self.production_enabled) is not int
                or self.production_enabled != 0
                or type(self.identity_dispatch) is not int
                or self.identity_dispatch != 0):
            raise BroadQaExternalDataError(
                "normalization accepted rule 不得启用生产或 identity dispatch")

    def to_dict(self) -> dict[str, object]:
        """导出 accepted rule、SUPPORT 和 rejection ledger 引用。"""
        return {
            "artifact_kind": NORMALIZATION_ACCEPTED_RULE_V3_KIND,
            "candidate": self.candidate.to_dict(),
            "evidence_commitments": [
                item.to_dict() for item in self.evidence_commitments],
            "format_version": 1,
            "identity_dispatch": self.identity_dispatch,
            "production_enabled": self.production_enabled,
            "rejection_record_sha256s": list(
                self.rejection_record_sha256s),
            "runtime_state": self.runtime_state,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 accepted rule。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 accepted rule 的规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaNormalizationAcceptedRuleV3":
        """从字段精确 JSON object 恢复 accepted rule。"""
        expected = {
            "artifact_kind", "candidate", "evidence_commitments",
            "format_version", "identity_dispatch", "production_enabled",
            "rejection_record_sha256s", "runtime_state",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"] != NORMALIZATION_ACCEPTED_RULE_V3_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or not isinstance(value["evidence_commitments"], list)
                or not isinstance(value["rejection_record_sha256s"], list)):
            raise BroadQaExternalDataError(
                "normalization accepted rule 字段漂移")
        return cls(
            BroadQaNormalizationRuleCandidateV3.from_dict(value["candidate"]),
            tuple(BroadQaNormalizationEvidenceCommitmentV3.from_dict(item)
                  for item in value["evidence_commitments"]),
            tuple(value["rejection_record_sha256s"]),
            value["runtime_state"],
            value["production_enabled"],
            value["identity_dispatch"],
        )


def _parse_record(payload: bytes, *, accepted: bool):
    """严格回读一条 normalization accepted/rejected 规范记录。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError(
            "normalization v3 record 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "normalization v3 record 不是规范 JSON") from error
    record = (
        BroadQaNormalizationAcceptedRuleV3.from_dict(value)
        if accepted else BroadQaNormalizationRejectedTrialV3.from_dict(value)
    )
    if record.canonical_bytes() != payload:
        raise BroadQaExternalDataError(
            "normalization v3 record 字节漂移")
    return record


def parse_normalization_accepted_rule_v3(
        payload: bytes,
        ) -> BroadQaNormalizationAcceptedRuleV3:
    """严格回读一条 normalization accepted rule。"""
    return _parse_record(payload, accepted=True)


def parse_normalization_rejected_trial_v3(
        payload: bytes,
        ) -> BroadQaNormalizationRejectedTrialV3:
    """严格回读一条 normalization rejected trial。"""
    return _parse_record(payload, accepted=False)


__all__ = [
    "BroadQaNormalizationAcceptedRuleV3",
    "BroadQaNormalizationRejectedTrialV3",
    "BroadQaNormalizationRuleCandidateV3",
    "NORMALIZATION_ACCEPTED_RULE_V3_KIND",
    "NORMALIZATION_CONTEXT_REJECTION_KIND",
    "NORMALIZATION_REJECTED_TRIAL_V3_KIND",
    "parse_normalization_accepted_rule_v3",
    "parse_normalization_rejected_trial_v3",
]
