"""来源归纳 v3 rule pack：accepted rule 与失败 trial 独立入账。

v2 把同一规则 hypothesis 的 SUPPORT/REFUTE 同时设为发布前提，会使核心
``HypothesisLedger`` 将其解析为冲突。v3 保留 v2 reader，但用新 record kind
把已接受规则的 SUPPORT 与被拒 trial 的 REFUTE 分开；本模块仍不接生产查询。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
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
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    SOURCE_INFERENCE_DIRECTIONS,
    source_inference_rule_hypothesis_key,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE,
    read_source_inference_learning_chain,
    source_inference_learning_prefix_sha256,
    source_inference_learning_result_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_protocol import (
    SOURCE_INFERENCE_LEARNING_FAMILIES,
    read_source_inference_learning_protocol,
    read_source_inference_learning_slice,
    source_inference_protocol_scope,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_rule_pack import (
    BroadQaSourceInferenceRuleEvidenceCommitment,
    SOURCE_INFERENCE_EVIDENCE_REASON_KEYS,
    SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS,
    SOURCE_INFERENCE_RULE_RUNTIME_STATE,
    read_source_inference_learning_split_inventory,
    validate_source_inference_training_commitments,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


SOURCE_INFERENCE_RULE_PACK_V3_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_RULE_PACK_V3")
SOURCE_INFERENCE_ACCEPTED_RULE_V3_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_ACCEPTED_RULE_V3")
SOURCE_INFERENCE_REJECTED_TRIAL_V3_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REJECTED_TRIAL_V3")
SOURCE_INFERENCE_REJECTION_KINDS = (
    "CONTEXTUAL_COUNTEREXAMPLE",
    "FAILED_CANDIDATE_TRIAL",
)
SOURCE_INFERENCE_RULE_PACK_V3_STATUS = (
    "FROZEN_NOT_EVALUATED_NOT_DEPLOYED")


def _sha256(value: object, *, label: str) -> str:
    """要求 artifact 承诺为小写 SHA-256。"""
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
class BroadQaSourceInferenceRuleCandidateV3:
    """规则候选的协议、算子、方向、适用域和 defeater 坐标。"""

    protocol_manifest_sha256: str
    operator_family: str
    operator: ObjectIdentity
    operator_version: int
    schema: ObjectIdentity
    direction: str
    application_domain: str
    defeaters: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验候选身份和静态适用边界。"""
        _sha256(
            self.protocol_manifest_sha256,
            label="v3 candidate protocol manifest",
        )
        if self.operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
            raise BroadQaExternalDataError("v3 candidate family 未启用")
        if (not isinstance(self.operator, ObjectIdentity)
                or self.operator.object_kind != OBJECT_MINIMAL_INSTRUCTION
                or type(self.operator_version) is not int
                or self.operator_version <= 0
                or not isinstance(self.schema, ObjectIdentity)
                or self.schema.object_kind != OBJECT_STRUCTURE_CONCEPT
                or self.direction not in SOURCE_INFERENCE_DIRECTIONS):
            raise BroadQaExternalDataError("v3 candidate identity 非法")
        if self.application_domain != SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS[
                self.operator_family]:
            raise BroadQaExternalDataError(
                "v3 candidate application domain 漂移")
        if (not isinstance(self.defeaters, tuple) or not self.defeaters
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.defeaters)
                or tuple(item.stable_key() for item in self.defeaters)
                != tuple(sorted({
                    item.stable_key() for item in self.defeaters}))):
            raise BroadQaExternalDataError(
                "v3 candidate defeaters 必须非空唯一排序")

    def hypothesis(self):
        """构造该候选在冻结 protocol scope 下的完整 hypothesis。"""
        return source_inference_rule_hypothesis_key(
            self.operator,
            self.schema,
            self.direction,
            self.operator_version,
            source_inference_protocol_scope(self.protocol_manifest_sha256),
        )

    def to_dict(self) -> dict[str, object]:
        """导出候选的字段精确 JSON 值。"""
        return {
            "application_domain": self.application_domain,
            "defeater_keys": [
                list(item.stable_key()) for item in self.defeaters],
            "direction": self.direction,
            "operator_family": self.operator_family,
            "operator_key": list(self.operator.stable_key()),
            "operator_version": self.operator_version,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "schema_key": list(self.schema.stable_key()),
        }

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceRuleCandidateV3":
        """从字段精确 JSON object 回读候选。"""
        expected = {
            "application_domain", "defeater_keys", "direction",
            "operator_family", "operator_key", "operator_version",
            "protocol_manifest_sha256", "schema_key",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or not isinstance(value["defeater_keys"], list)):
            raise BroadQaExternalDataError("v3 candidate 字段漂移")
        return cls(
            value["protocol_manifest_sha256"],
            value["operator_family"],
            _identity(
                value["operator_key"],
                label="v3 candidate operator",
                object_kind=OBJECT_MINIMAL_INSTRUCTION,
            ),
            value["operator_version"],
            _identity(
                value["schema_key"],
                label="v3 candidate schema",
                object_kind=OBJECT_STRUCTURE_CONCEPT,
            ),
            value["direction"],
            value["application_domain"],
            tuple(_identity(
                item,
                label="v3 candidate defeater",
                object_kind=OBJECT_CONCEPT,
            ) for item in value["defeater_keys"]),
        )


def _validate_commitments(
        candidate: BroadQaSourceInferenceRuleCandidateV3,
        commitments: tuple[BroadQaSourceInferenceRuleEvidenceCommitment, ...],
        *,
        qualification_kind: str,
        stance: int,
        label: str,
        ) -> None:
    """核验 commitment 唯一排序并绑定指定候选和单一立场。"""
    if (not isinstance(commitments, tuple) or not commitments
            or any(not isinstance(
                item, BroadQaSourceInferenceRuleEvidenceCommitment)
                   for item in commitments)):
        raise BroadQaExternalDataError(f"{label} Evidence commitments 非法")
    evidence_keys = tuple(item.evidence_key for item in commitments)
    if evidence_keys != tuple(sorted(set(evidence_keys))):
        raise BroadQaExternalDataError(f"{label} Evidence 必须唯一规范排序")
    expected_hypothesis = candidate.hypothesis()
    expected_reason = SOURCE_INFERENCE_EVIDENCE_REASON_KEYS[
        candidate.operator_family][qualification_kind]
    for commitment in commitments:
        evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
        if (commitment.qualification_kind != qualification_kind
                or evidence.stance != stance
                or evidence.hypothesis != expected_hypothesis
                or evidence.reason_key != expected_reason):
            raise BroadQaExternalDataError(
                f"{label} Evidence hypothesis/stance/reason 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceRejectedTrialV3:
    """一个独立候选 trial 的来源化拒绝记录。"""

    candidate: BroadQaSourceInferenceRuleCandidateV3
    rejection_kind: str
    candidate_parameters_sha256: str
    evidence_commitments: tuple[
        BroadQaSourceInferenceRuleEvidenceCommitment, ...]

    def __post_init__(self) -> None:
        """要求 trial 只携带绑定自身 hypothesis 的 REFUTE Evidence。"""
        if not isinstance(self.candidate, BroadQaSourceInferenceRuleCandidateV3):
            raise TypeError("v3 rejected trial candidate 类型非法")
        if self.rejection_kind not in SOURCE_INFERENCE_REJECTION_KINDS:
            raise BroadQaExternalDataError("v3 rejection kind 未注册")
        _sha256(
            self.candidate_parameters_sha256,
            label="v3 rejected trial parameters",
        )
        _validate_commitments(
            self.candidate,
            self.evidence_commitments,
            qualification_kind="REPLAYED_CANDIDATE_REFUTE",
            stance=EVIDENCE_REFUTE,
            label="v3 rejected trial",
        )

    @property
    def protocol_manifest_sha256(self) -> str:
        """暴露复用 TRAIN 校验所需的 protocol identity。"""
        return self.candidate.protocol_manifest_sha256

    @property
    def operator_family(self) -> str:
        """暴露复用 TRAIN 校验所需的 operator family。"""
        return self.candidate.operator_family

    def to_dict(self) -> dict[str, object]:
        """导出拒绝 trial 及其独立 Evidence。"""
        return {
            "artifact_kind": SOURCE_INFERENCE_REJECTED_TRIAL_V3_KIND,
            "candidate": self.candidate.to_dict(),
            "candidate_parameters_sha256": self.candidate_parameters_sha256,
            "evidence_commitments": [
                item.to_dict() for item in self.evidence_commitments],
            "format_version": 1,
            "rejection_kind": self.rejection_kind,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范拒绝记录。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回拒绝记录规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceRejectedTrialV3":
        """从字段精确 JSON object 回读拒绝 trial。"""
        expected = {
            "artifact_kind", "candidate", "candidate_parameters_sha256",
            "evidence_commitments", "format_version", "rejection_kind",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"]
                != SOURCE_INFERENCE_REJECTED_TRIAL_V3_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or not isinstance(value["evidence_commitments"], list)):
            raise BroadQaExternalDataError("v3 rejected trial 字段漂移")
        return cls(
            BroadQaSourceInferenceRuleCandidateV3.from_dict(
                value["candidate"]),
            value["rejection_kind"],
            value["candidate_parameters_sha256"],
            tuple(BroadQaSourceInferenceRuleEvidenceCommitment.from_dict(item)
                  for item in value["evidence_commitments"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceAcceptedRuleV3:
    """只由 SUPPORT 采用、并显式引用独立 rejection ledger 的规则。"""

    candidate: BroadQaSourceInferenceRuleCandidateV3
    evidence_commitments: tuple[
        BroadQaSourceInferenceRuleEvidenceCommitment, ...]
    rejection_record_sha256s: tuple[str, ...]
    runtime_state: str = SOURCE_INFERENCE_RULE_RUNTIME_STATE
    production_enabled: int = 0
    item_identity_dispatch: int = 0
    title_identity_dispatch: int = 0
    page_identity_dispatch: int = 0

    def __post_init__(self) -> None:
        """要求 accepted hypothesis 只有 SUPPORT 且保持生产禁用。"""
        if not isinstance(self.candidate, BroadQaSourceInferenceRuleCandidateV3):
            raise TypeError("v3 accepted rule candidate 类型非法")
        _validate_commitments(
            self.candidate,
            self.evidence_commitments,
            qualification_kind="REPLAYED_CANDIDATE_SUPPORT",
            stance=EVIDENCE_SUPPORT,
            label="v3 accepted rule",
        )
        if (not isinstance(self.rejection_record_sha256s, tuple)
                or not self.rejection_record_sha256s
                or self.rejection_record_sha256s != tuple(sorted(set(
                    self.rejection_record_sha256s)))):
            raise BroadQaExternalDataError(
                "v3 accepted rule rejection refs 必须非空唯一排序")
        for value in self.rejection_record_sha256s:
            _sha256(value, label="v3 accepted rule rejection ref")
        if (self.runtime_state != SOURCE_INFERENCE_RULE_RUNTIME_STATE
                or type(self.production_enabled) is not int
                or self.production_enabled != 0
                or any(type(value) is not int or value != 0 for value in (
                    self.item_identity_dispatch,
                    self.title_identity_dispatch,
                    self.page_identity_dispatch,
                ))):
            raise BroadQaExternalDataError(
                "v3 accepted rule 不得启用生产或逐 identity dispatch")

    @property
    def protocol_manifest_sha256(self) -> str:
        """暴露复用 TRAIN 校验所需的 protocol identity。"""
        return self.candidate.protocol_manifest_sha256

    @property
    def operator_family(self) -> str:
        """暴露复用 TRAIN 校验所需的 operator family。"""
        return self.candidate.operator_family

    def to_dict(self) -> dict[str, object]:
        """导出 accepted rule、SUPPORT 和 rejection 引用。"""
        return {
            "artifact_kind": SOURCE_INFERENCE_ACCEPTED_RULE_V3_KIND,
            "candidate": self.candidate.to_dict(),
            "evidence_commitments": [
                item.to_dict() for item in self.evidence_commitments],
            "format_version": 1,
            "item_identity_dispatch": self.item_identity_dispatch,
            "page_identity_dispatch": self.page_identity_dispatch,
            "production_enabled": self.production_enabled,
            "rejection_record_sha256s": list(
                self.rejection_record_sha256s),
            "runtime_state": self.runtime_state,
            "title_identity_dispatch": self.title_identity_dispatch,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 accepted rule。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 accepted rule 规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceAcceptedRuleV3":
        """从字段精确 JSON object 回读 accepted rule。"""
        expected = {
            "artifact_kind", "candidate", "evidence_commitments",
            "format_version", "item_identity_dispatch",
            "page_identity_dispatch", "production_enabled",
            "rejection_record_sha256s", "runtime_state",
            "title_identity_dispatch",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"]
                != SOURCE_INFERENCE_ACCEPTED_RULE_V3_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or not isinstance(value["evidence_commitments"], list)
                or not isinstance(value["rejection_record_sha256s"], list)):
            raise BroadQaExternalDataError("v3 accepted rule 字段漂移")
        return cls(
            BroadQaSourceInferenceRuleCandidateV3.from_dict(
                value["candidate"]),
            tuple(BroadQaSourceInferenceRuleEvidenceCommitment.from_dict(item)
                  for item in value["evidence_commitments"]),
            tuple(value["rejection_record_sha256s"]),
            value["runtime_state"],
            value["production_enabled"],
            value["item_identity_dispatch"],
            value["title_identity_dispatch"],
            value["page_identity_dispatch"],
        )


def _parse_record(payload: bytes, *, accepted: bool):
    """严格回读一条 v3 accepted 或 rejected 规范记录。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError("v3 rule pack record 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "v3 rule pack record 不是规范 JSON") from error
    record = (
        BroadQaSourceInferenceAcceptedRuleV3.from_dict(value)
        if accepted else BroadQaSourceInferenceRejectedTrialV3.from_dict(value)
    )
    if record.canonical_bytes() != payload:
        raise BroadQaExternalDataError("v3 rule pack record 字节漂移")
    return record


def parse_source_inference_accepted_rule_v3(
        payload: bytes,
        ) -> BroadQaSourceInferenceAcceptedRuleV3:
    """严格回读一条 accepted rule v3。"""
    return _parse_record(payload, accepted=True)


def parse_source_inference_rejected_trial_v3(
        payload: bytes,
        ) -> BroadQaSourceInferenceRejectedTrialV3:
    """严格回读一条 rejected trial v3。"""
    return _parse_record(payload, accepted=False)


def _validate_epistemic_separation(
        accepted_rules: tuple[BroadQaSourceInferenceAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaSourceInferenceRejectedTrialV3, ...],
        ) -> None:
    """用核心 ledger 证明 accepted=SUPPORTED、trial=REFUTED 且身份互异。"""
    accepted_hypotheses = tuple(item.candidate.hypothesis()
                                for item in accepted_rules)
    rejected_hypotheses = tuple(item.candidate.hypothesis()
                                for item in rejected_trials)
    if (len(set(accepted_hypotheses)) != len(accepted_hypotheses)
            or len(set(rejected_hypotheses)) != len(rejected_hypotheses)
            or set(accepted_hypotheses) & set(rejected_hypotheses)):
        raise BroadQaExternalDataError(
            "v3 accepted/rejected hypothesis 必须独立唯一")
    evidence_ids = tuple(
        EvidenceRecord.from_stable_key(commitment.evidence_key).evidence_id
        for record in accepted_rules + rejected_trials
        for commitment in record.evidence_commitments
    )
    if len(set(evidence_ids)) != len(evidence_ids):
        raise BroadQaExternalDataError(
            "v3 accepted/rejected Evidence id 必须全局唯一")
    ledger = HypothesisLedger()
    for record, expected in (
            *((item, EPISTEMIC_SUPPORTED) for item in accepted_rules),
            *((item, EPISTEMIC_REFUTED) for item in rejected_trials)):
        hypothesis = record.candidate.hypothesis()
        ledger.register(hypothesis)
        for commitment in record.evidence_commitments:
            ledger.append_evidence(EvidenceRecord.from_stable_key(
                commitment.evidence_key))
        if ledger.snapshot(hypothesis).epistemic_status != expected:
            raise BroadQaExternalDataError(
                "v3 accepted/rejected epistemic 状态漂移")


def source_inference_rule_pack_v3_result_sha256(
        *,
        protocol_manifest_sha256: str,
        operator_family: str,
        training_item_ids: tuple[str, ...],
        accepted_rules: tuple[BroadQaSourceInferenceAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaSourceInferenceRejectedTrialV3, ...],
        ) -> str:
    """从 accepted、rejected 及其 Evidence 重算唯一结果摘要。"""
    if (not isinstance(accepted_rules, tuple) or not accepted_rules
            or not isinstance(rejected_trials, tuple) or not rejected_trials):
        raise BroadQaExternalDataError("v3 pack records 不能为空")
    records = accepted_rules + rejected_trials
    evidence_shas = tuple(sorted({
        hashlib.sha256(canonical_json_bytes(
            list(commitment.evidence_key))).hexdigest()
        for record in records
        for commitment in record.evidence_commitments
    }))
    record_shas = tuple(sorted(record.sha256() for record in records))
    return source_inference_learning_result_sha256(
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=operator_family,
        processed_item_ids=training_item_ids,
        evidence_record_sha256s=evidence_shas,
        rule_record_sha256s=record_shas,
    )


def _record_payloads(
        accepted_rules: tuple[BroadQaSourceInferenceAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaSourceInferenceRejectedTrialV3, ...],
        ) -> tuple[bytes, bytes]:
    """要求两类 record 均按自身 SHA 唯一排序并返回规范字节。"""
    for label, records in (
            ("accepted", accepted_rules), ("rejected", rejected_trials)):
        shas = tuple(item.sha256() for item in records)
        if shas != tuple(sorted(set(shas))):
            raise BroadQaExternalDataError(
                f"v3 {label} records 必须按 SHA 唯一排序")
    return (
        b"".join(item.canonical_bytes() for item in accepted_rules),
        b"".join(item.canonical_bytes() for item in rejected_trials),
    )


def _read_v3_training_provenance(
        *,
        protocol_root: Path,
        operator_family: str,
        ) -> tuple[
            dict[str, object],
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
            dict[str, str],
            tuple[str, ...],
        ]:
    """只回读冻结 protocol 的 LEARNER/TRAIN 来源及全局 split inventory。"""
    protocol = read_source_inference_learning_protocol(
        protocol_root / "manifest.json")
    if operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
        raise BroadQaExternalDataError("v3 rule pack family 未启用")
    dossier, census = read_source_inference_learning_slice(
        protocol_dir=protocol_root,
        access_role="LEARNER",
        operator_family=operator_family,
    )
    dossier_by_id = {str(item["item_id"]): item for item in dossier}
    census_by_id = {str(item["item_id"]): item for item in census}
    training_item_ids = tuple(str(item["item_id"]) for item in dossier)
    inventory_identity = protocol["item_split_inventory"]
    inventory_path = (
        protocol_root / inventory_identity["relative_path"]).resolve()
    try:
        inventory_payload = inventory_path.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "v3 rule pack split inventory 不可读") from error
    if (not inventory_path.is_relative_to(protocol_root)
            or len(inventory_payload) != inventory_identity["bytes"]
            or hashlib.sha256(inventory_payload).hexdigest()
            != inventory_identity["sha256"]):
        raise BroadQaExternalDataError(
            "v3 rule pack split inventory commitment 漂移")
    split_by_item = read_source_inference_learning_split_inventory(
        inventory_path)
    if len(split_by_item) != inventory_identity["record_count"]:
        raise BroadQaExternalDataError(
            "v3 rule pack split inventory count 漂移")
    return (
        protocol,
        dossier_by_id,
        census_by_id,
        split_by_item,
        training_item_ids,
    )


def publish_source_inference_rule_pack_v3(
        *,
        protocol_dir: str | Path,
        operator_family: str,
        fresh_accepted_rules: tuple[
            BroadQaSourceInferenceAcceptedRuleV3, ...],
        fresh_rejected_trials: tuple[
            BroadQaSourceInferenceRejectedTrialV3, ...],
        resumed_accepted_rules: tuple[
            BroadQaSourceInferenceAcceptedRuleV3, ...],
        resumed_rejected_trials: tuple[
            BroadQaSourceInferenceRejectedTrialV3, ...],
        target_dir: str | Path,
        fresh_checkpoint_chain_path: str | Path,
        resumed_checkpoint_chain_path: str | Path,
        ) -> dict[str, object]:
    """核验 TRAIN、双链和分账语义后不可覆盖发布禁用态 v3 pack。"""
    protocol_root = Path(protocol_dir).resolve()
    (
        protocol,
        dossier_by_id,
        census_by_id,
        split_by_item,
        training_item_ids,
    ) = _read_v3_training_provenance(
        protocol_root=protocol_root,
        operator_family=operator_family,
    )
    protocol_sha = protocol["manifest_sha256"]
    record_groups = (
        fresh_accepted_rules, fresh_rejected_trials,
        resumed_accepted_rules, resumed_rejected_trials,
    )
    expected_types = (
        BroadQaSourceInferenceAcceptedRuleV3,
        BroadQaSourceInferenceRejectedTrialV3,
        BroadQaSourceInferenceAcceptedRuleV3,
        BroadQaSourceInferenceRejectedTrialV3,
    )
    if (any(not isinstance(group, tuple) or not group
            or any(not isinstance(record, expected_type) for record in group)
            for group, expected_type in zip(record_groups, expected_types))
            or any(record.operator_family != operator_family
                   or record.protocol_manifest_sha256 != protocol_sha
                   for group in record_groups for record in group)):
        raise BroadQaExternalDataError(
            "v3 rule pack records/family/protocol 漂移")
    fresh_payloads = _record_payloads(
        fresh_accepted_rules, fresh_rejected_trials)
    resumed_payloads = _record_payloads(
        resumed_accepted_rules, resumed_rejected_trials)
    if fresh_payloads != resumed_payloads:
        raise BroadQaExternalDataError(
            "v3 rule pack fresh/resume record 字节不等价")
    accepted_rules = fresh_accepted_rules
    rejected_trials = fresh_rejected_trials
    rejected_shas = {item.sha256() for item in rejected_trials}
    referenced_rejections = {
        value for rule in accepted_rules
        for value in rule.rejection_record_sha256s
    }
    if referenced_rejections != rejected_shas:
        raise BroadQaExternalDataError(
            "v3 accepted rule 与 rejection ledger 引用未精确闭合")
    _validate_epistemic_separation(accepted_rules, rejected_trials)
    for record in accepted_rules + rejected_trials:
        validate_source_inference_training_commitments(
            record,
            dossier_by_id=dossier_by_id,
            census_by_id=census_by_id,
            split_by_item=split_by_item,
        )

    maximum_evidence = (
        len(training_item_ids)
        * protocol["learning_stopping_contract"][
            "maximum_evidence_candidates_per_item"])
    maximum_candidates = protocol["learning_stopping_contract"][
        "maximum_rule_candidates_per_family"]
    output_evidence_count = sum(
        len(record.evidence_commitments)
        for record in accepted_rules + rejected_trials)
    output_candidate_count = len(accepted_rules) + len(rejected_trials)
    chain_paths = tuple(Path(path).resolve() for path in (
        fresh_checkpoint_chain_path, resumed_checkpoint_chain_path))
    if (chain_paths[0] == chain_paths[1]
            or any(not path.is_relative_to(protocol_root.parent)
                   for path in chain_paths)):
        raise BroadQaExternalDataError(
            "v3 checkpoint 必须独立且位于 protocol run root")
    checkpoint_lineage = []
    for chain_path in chain_paths:
        chain = read_source_inference_learning_chain(chain_path)
        checkpoint = chain[-1]
        checkpoint_lineage.append({
            "chain_sha256": hashlib.sha256(
                chain_path.read_bytes()).hexdigest(),
            "run_id": checkpoint.run_id,
            "terminal_sha256": checkpoint.sha256(),
        })
        if (checkpoint.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
                or checkpoint.protocol_manifest_sha256 != protocol_sha
                or checkpoint.operator_family != operator_family
                or checkpoint.training_item_count != len(training_item_ids)
                or checkpoint.training_item_order_sha256
                != source_inference_learning_prefix_sha256(training_item_ids)
                or not output_evidence_count
                <= checkpoint.evidence_candidate_count <= maximum_evidence
                or not output_candidate_count
                <= checkpoint.rule_candidate_count <= maximum_candidates):
            raise BroadQaExternalDataError(
                "v3 checkpoint 未证明完整 TRAIN 或候选预算")
    if len({item["run_id"] for item in checkpoint_lineage}) != 2:
        raise BroadQaExternalDataError(
            "v3 fresh/resume checkpoint run identity 必须独立")
    result_sha = source_inference_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        training_item_ids=training_item_ids,
        accepted_rules=accepted_rules,
        rejected_trials=rejected_trials,
    )
    resumed_result_sha = source_inference_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        training_item_ids=training_item_ids,
        accepted_rules=resumed_accepted_rules,
        rejected_trials=resumed_rejected_trials,
    )
    if result_sha != resumed_result_sha:
        raise BroadQaExternalDataError("v3 fresh/resume 重算结果不等价")

    target = Path(target_dir).resolve()
    if not target.is_relative_to(protocol_root.parent):
        raise BroadQaExternalDataError(
            "v3 rule pack target 必须位于 protocol run root")
    if target.exists():
        raise BroadQaExternalDataError("v3 rule pack target 已存在")
    target.mkdir(parents=True)
    accepted_path = target / "accepted-rules.jsonl"
    rejected_path = target / "rejected-trials.jsonl"
    accepted_path.write_bytes(fresh_payloads[0])
    rejected_path.write_bytes(fresh_payloads[1])
    accepted_sha = hashlib.sha256(fresh_payloads[0]).hexdigest()
    rejected_sha = hashlib.sha256(fresh_payloads[1]).hexdigest()
    manifest = {
        "accepted_records_bytes": len(fresh_payloads[0]),
        "accepted_records_count": len(accepted_rules),
        "accepted_records_sha256": accepted_sha,
        "artifact_kind": SOURCE_INFERENCE_RULE_PACK_V3_KIND,
        "format_version": 1,
        "fresh_checkpoint_chain_sha256": checkpoint_lineage[0][
            "chain_sha256"],
        "fresh_checkpoint_terminal_sha256": checkpoint_lineage[0][
            "terminal_sha256"],
        "fresh_result_sha256": result_sha,
        "fresh_run_id": checkpoint_lineage[0]["run_id"],
        "operator_family": operator_family,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "rejected_records_bytes": len(fresh_payloads[1]),
        "rejected_records_count": len(rejected_trials),
        "rejected_records_sha256": rejected_sha,
        "resumed_checkpoint_chain_sha256": checkpoint_lineage[1][
            "chain_sha256"],
        "resumed_checkpoint_terminal_sha256": checkpoint_lineage[1][
            "terminal_sha256"],
        "resumed_result_sha256": resumed_result_sha,
        "resumed_run_id": checkpoint_lineage[1]["run_id"],
        "runtime_state": SOURCE_INFERENCE_RULE_RUNTIME_STATE,
        "status": SOURCE_INFERENCE_RULE_PACK_V3_STATUS,
        "training_item_count": len(training_item_ids),
        "training_item_order_sha256": source_inference_learning_prefix_sha256(
            training_item_ids),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
    }


def read_source_inference_rule_pack_v3(
        target_dir: str | Path,
        *,
        protocol_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[BroadQaSourceInferenceAcceptedRuleV3, ...],
            tuple[BroadQaSourceInferenceRejectedTrialV3, ...],
        ]:
    """依据真实 TRAIN 重放 v3 manifest、accepted rules 和 rejected trials。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    accepted_path = root / "accepted-rules.jsonl"
    rejected_path = root / "rejected-trials.jsonl"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        accepted_payload = accepted_path.read_bytes()
        rejected_payload = rejected_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v3 rule pack 不可读") from error
    expected = {
        "accepted_records_bytes", "accepted_records_count",
        "accepted_records_sha256", "artifact_kind", "format_version",
        "fresh_checkpoint_chain_sha256",
        "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
        "fresh_run_id", "operator_family", "production_enabled",
        "protocol_manifest_sha256", "rejected_records_bytes",
        "rejected_records_count", "rejected_records_sha256",
        "resumed_checkpoint_chain_sha256",
        "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
        "resumed_run_id", "runtime_state", "status",
        "training_item_count", "training_item_order_sha256",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != payload
            or manifest["artifact_kind"] != SOURCE_INFERENCE_RULE_PACK_V3_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"]
            not in SOURCE_INFERENCE_LEARNING_FAMILIES
            or type(manifest["production_enabled"]) is not int
            or manifest["production_enabled"] != 0
            or manifest["runtime_state"] != SOURCE_INFERENCE_RULE_RUNTIME_STATE
            or manifest["status"] != SOURCE_INFERENCE_RULE_PACK_V3_STATUS
            or manifest["fresh_run_id"] == manifest["resumed_run_id"]
            or manifest["fresh_result_sha256"]
            != manifest["resumed_result_sha256"]):
        raise BroadQaExternalDataError("v3 rule pack manifest 漂移")
    for prefix, records_payload in (
            ("accepted", accepted_payload), ("rejected", rejected_payload)):
        if (type(manifest[f"{prefix}_records_count"]) is not int
                or manifest[f"{prefix}_records_count"] <= 0
                or type(manifest[f"{prefix}_records_bytes"]) is not int
                or manifest[f"{prefix}_records_bytes"] != len(records_payload)
                or manifest[f"{prefix}_records_sha256"]
                != hashlib.sha256(records_payload).hexdigest()
                or not records_payload.endswith(b"\n")):
            raise BroadQaExternalDataError(
                f"v3 {prefix} records commitment 漂移")
    for name in (
            "accepted_records_sha256", "fresh_checkpoint_chain_sha256",
            "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
            "fresh_run_id", "protocol_manifest_sha256",
            "rejected_records_sha256", "resumed_checkpoint_chain_sha256",
            "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
            "resumed_run_id", "training_item_order_sha256"):
        _sha256(manifest[name], label=f"v3 manifest {name}")
    if (type(manifest["training_item_count"]) is not int
            or manifest["training_item_count"] <= 0):
        raise BroadQaExternalDataError("v3 training item count 非法")
    accepted = tuple(
        parse_source_inference_accepted_rule_v3(line + b"\n")
        for line in accepted_payload.splitlines()
    )
    rejected = tuple(
        parse_source_inference_rejected_trial_v3(line + b"\n")
        for line in rejected_payload.splitlines()
    )
    if (len(accepted) != manifest["accepted_records_count"]
            or len(rejected) != manifest["rejected_records_count"]):
        raise BroadQaExternalDataError("v3 rule pack record count 漂移")
    _record_payloads(accepted, rejected)
    if ({value for rule in accepted
         for value in rule.rejection_record_sha256s}
            != {item.sha256() for item in rejected}
            or any(record.operator_family != manifest["operator_family"]
                   or record.protocol_manifest_sha256
                   != manifest["protocol_manifest_sha256"]
                   for record in accepted + rejected)):
        raise BroadQaExternalDataError(
            "v3 rule pack record identity/reference 漂移")
    _validate_epistemic_separation(accepted, rejected)

    protocol_root = Path(protocol_dir).resolve()
    (
        protocol,
        dossier_by_id,
        census_by_id,
        split_by_item,
        training_item_ids,
    ) = _read_v3_training_provenance(
        protocol_root=protocol_root,
        operator_family=manifest["operator_family"],
    )
    protocol_sha = protocol["manifest_sha256"]
    if manifest["protocol_manifest_sha256"] != protocol_sha:
        raise BroadQaExternalDataError("v3 rule pack protocol commitment 漂移")
    for record in accepted + rejected:
        validate_source_inference_training_commitments(
            record,
            dossier_by_id=dossier_by_id,
            census_by_id=census_by_id,
            split_by_item=split_by_item,
        )
    expected_order_sha = source_inference_learning_prefix_sha256(
        training_item_ids)
    expected_result_sha = source_inference_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family=manifest["operator_family"],
        training_item_ids=training_item_ids,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    if (manifest["training_item_count"] != len(training_item_ids)
            or manifest["training_item_order_sha256"] != expected_order_sha
            or manifest["fresh_result_sha256"] != expected_result_sha
            or manifest["resumed_result_sha256"] != expected_result_sha):
        raise BroadQaExternalDataError(
            "v3 rule pack TRAIN/result commitment 漂移")
    return ({
        **manifest,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }, accepted, rejected)


__all__ = [
    "BroadQaSourceInferenceAcceptedRuleV3",
    "BroadQaSourceInferenceRejectedTrialV3",
    "BroadQaSourceInferenceRuleCandidateV3",
    "SOURCE_INFERENCE_ACCEPTED_RULE_V3_KIND",
    "SOURCE_INFERENCE_REJECTED_TRIAL_V3_KIND",
    "SOURCE_INFERENCE_REJECTION_KINDS",
    "SOURCE_INFERENCE_RULE_PACK_V3_KIND",
    "SOURCE_INFERENCE_RULE_PACK_V3_STATUS",
    "parse_source_inference_accepted_rule_v3",
    "parse_source_inference_rejected_trial_v3",
    "publish_source_inference_rule_pack_v3",
    "read_source_inference_rule_pack_v3",
    "source_inference_rule_pack_v3_result_sha256",
]
