"""来源归纳 learned rule pack 的严格、默认禁用输出合同。

rule pack 的学习 Evidence 绑定 learning protocol 自身的来源 scope；每条 Evidence
同时保存具体 TRAIN 文档和 span 承诺。该 scope 不冒充 Wikipedia 文档 scope，且
当前 pack 不接入 ``BroadQaSourceDerivation`` 或生产查询。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
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
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


SOURCE_INFERENCE_RULE_PACK_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_RULE_PACK_V2")
SOURCE_INFERENCE_RULE_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNED_RULE_V2")
SOURCE_INFERENCE_RULE_RUNTIME_STATE = "LEARNED_PACK_DISABLED"
SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS = {
    "NORMALIZATION_EQUIVALENCE": "TERMINAL_SOURCE_NORMALIZATION_V1",
    "SOURCE_SPAN_SELECTION": "PROJECTED_TERMINAL_PASSAGE_SELECTION_V1",
}
SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND = 817032
SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_KINDS = {
    "REPLAYED_CANDIDATE_REFUTE": 2,
    "REPLAYED_CANDIDATE_SUPPORT": 1,
}
SOURCE_INFERENCE_EVIDENCE_REASON_KEYS = {
    family: {
        qualification: (817037, family_ordinal, stance)
        for qualification, stance in
        SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_KINDS.items()
    }
    for family_ordinal, family in enumerate(
        SOURCE_INFERENCE_LEARNING_FAMILIES, start=1)
}


def _sha256(value: object, *, label: str) -> str:
    """要求 rule pack 承诺为小写 SHA-256。"""
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


def _source(value: object, *, label: str) -> SourceRef:
    """回读来源稳定身份。"""
    try:
        return SourceRef.from_stable_key(_strict_key(value, label=label))
    except (TypeError, ValueError) as error:
        raise BroadQaExternalDataError(f"{label} 来源身份非法") from error


def source_inference_qualification_input_sha256(
        operator_family: str,
        dossier: dict[str, object],
        ) -> str:
    """承诺 learner 应用候选规则时可见、但不含 census stance 的输入。"""
    if operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
        raise BroadQaExternalDataError("qualification input family 未启用")
    try:
        training = dossier["training_source"]
        terminal = dossier["terminal_source"]
        question = training["question"]
        if operator_family == "NORMALIZATION_EQUIVALENCE":
            source_value = terminal["wikitext"]
        else:
            source_value = [
                {
                    "ordinal": item["ordinal"],
                    "raw_end": item["raw_end"],
                    "raw_sha256": item["raw_sha256"],
                    "raw_start": item["raw_start"],
                    "text": item["text"],
                }
                for item in terminal["passages"]
            ]
    except (KeyError, TypeError) as error:
        raise BroadQaExternalDataError(
            "qualification input dossier 漂移") from error
    return hashlib.sha256(canonical_json_bytes({
        "operator_family": operator_family,
        "question": question,
        "source": source_value,
    })).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceRuleEvidenceCommitment:
    """一条规则 Evidence 对 TRAIN item 和终页 passage 的完整承诺。"""

    item_id: str
    source_key: str
    source_ref: SourceRef
    page_id: int
    revision_id: int
    passage_ordinal: int
    raw_start: int
    raw_end: int
    raw_sha256: str
    candidate_raw_start: int
    candidate_raw_end: int
    candidate_raw_sha256: str
    routing_signal_state: str
    qualification_kind: str
    qualification_input_sha256: str
    qualification_expected_sha256: str
    qualification_observed_sha256: str
    evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 item、文档/span、机械信号和可恢复 Evidence。"""
        _sha256(self.item_id, label="rule evidence item_id")
        if not isinstance(self.source_key, str) or not self.source_key:
            raise BroadQaExternalDataError("rule evidence source_key 非法")
        if (not isinstance(self.source_ref, SourceRef)
                or self.source_ref.source_kind
                != SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND):
            raise BroadQaExternalDataError("rule evidence source_ref 非法")
        for name in (
                "page_id", "revision_id", "passage_ordinal", "raw_end",
                "candidate_raw_end"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise BroadQaExternalDataError(f"rule evidence {name} 非法")
        if (type(self.raw_start) is not int or self.raw_start < 0
                or type(self.candidate_raw_start) is not int
                or self.raw_end <= self.raw_start
                or not self.raw_start <= self.candidate_raw_start
                < self.candidate_raw_end <= self.raw_end
                or self.source_ref.source_id != self.page_id
                or self.source_ref.document_id != self.revision_id):
            raise BroadQaExternalDataError("rule evidence source/span 漂移")
        _sha256(self.raw_sha256, label="rule evidence raw")
        _sha256(
            self.candidate_raw_sha256,
            label="rule evidence candidate raw",
        )
        if self.routing_signal_state not in (
                "MECHANICAL_SUPPORT_SIGNAL", "MECHANICAL_COUNTER_SIGNAL",
                "UNDETERMINED"):
            raise BroadQaExternalDataError(
                "rule evidence routing signal 未注册")
        if self.qualification_kind not in (
                SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_KINDS):
            raise BroadQaExternalDataError(
                "rule evidence qualification kind 未注册")
        _sha256(
            self.qualification_input_sha256,
            label="rule evidence qualification input",
        )
        expected = _sha256(
            self.qualification_expected_sha256,
            label="rule evidence qualification expected",
        )
        observed = _sha256(
            self.qualification_observed_sha256,
            label="rule evidence qualification observed",
        )
        try:
            evidence = EvidenceRecord.from_stable_key(self.evidence_key)
        except (TypeError, ValueError) as error:
            raise BroadQaExternalDataError(
                "rule evidence key 无法恢复") from error
        expected_stance = (
            EVIDENCE_SUPPORT
            if self.qualification_kind == "REPLAYED_CANDIDATE_SUPPORT"
            else EVIDENCE_REFUTE)
        if (evidence.stance != expected_stance
                or evidence.source != self.source_ref
                or evidence.supersedes_evidence_id != 0
                or (expected_stance == EVIDENCE_SUPPORT)
                != (observed == expected)):
            raise BroadQaExternalDataError(
                "rule Evidence qualification/stance/source 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出完整训练 item、source、span 和 Evidence identity。"""
        return {
            "evidence_key": list(self.evidence_key),
            "candidate_raw_end": self.candidate_raw_end,
            "candidate_raw_sha256": self.candidate_raw_sha256,
            "candidate_raw_start": self.candidate_raw_start,
            "item_id": self.item_id,
            "page_id": self.page_id,
            "passage_ordinal": self.passage_ordinal,
            "raw_end": self.raw_end,
            "raw_sha256": self.raw_sha256,
            "raw_start": self.raw_start,
            "qualification_expected_sha256": self.qualification_expected_sha256,
            "qualification_input_sha256": self.qualification_input_sha256,
            "qualification_kind": self.qualification_kind,
            "qualification_observed_sha256": self.qualification_observed_sha256,
            "revision_id": self.revision_id,
            "routing_signal_state": self.routing_signal_state,
            "source_key": self.source_key,
            "source_ref": list(self.source_ref.stable_key()),
        }

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceRuleEvidenceCommitment":
        """从字段精确的 JSON object 回读训练 Evidence 承诺。"""
        expected = {
            "candidate_raw_end", "candidate_raw_sha256",
            "candidate_raw_start", "evidence_key", "item_id", "page_id",
            "passage_ordinal",
            "qualification_expected_sha256", "qualification_input_sha256",
            "qualification_kind", "qualification_observed_sha256", "raw_end",
            "raw_sha256", "raw_start", "revision_id", "routing_signal_state",
            "source_key", "source_ref",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise BroadQaExternalDataError("rule evidence commitment 字段漂移")
        return cls(
            value["item_id"],
            value["source_key"],
            _source(value["source_ref"], label="rule evidence source_ref"),
            value["page_id"],
            value["revision_id"],
            value["passage_ordinal"],
            value["raw_start"],
            value["raw_end"],
            value["raw_sha256"],
            value["candidate_raw_start"],
            value["candidate_raw_end"],
            value["candidate_raw_sha256"],
            value["routing_signal_state"],
            value["qualification_kind"],
            value["qualification_input_sha256"],
            value["qualification_expected_sha256"],
            value["qualification_observed_sha256"],
            _strict_key(value["evidence_key"], label="rule evidence key"),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceLearnedRule:
    """一个协议来源化、带正反 Evidence 和 defeater 的 learned rule。"""

    protocol_manifest_sha256: str
    operator_family: str
    operator: ObjectIdentity
    operator_version: int
    schema: ObjectIdentity
    direction: str
    application_domain: str
    defeaters: tuple[ObjectIdentity, ...]
    evidence_commitments: tuple[
        BroadQaSourceInferenceRuleEvidenceCommitment, ...]
    runtime_state: str = SOURCE_INFERENCE_RULE_RUNTIME_STATE
    production_enabled: int = 0
    item_identity_dispatch: int = 0
    title_identity_dispatch: int = 0
    page_identity_dispatch: int = 0

    def __post_init__(self) -> None:
        """核验规则坐标、协议 scope、正反例和禁止逐题 dispatch。"""
        protocol_sha = _sha256(
            self.protocol_manifest_sha256, label="rule protocol manifest")
        if self.operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
            raise BroadQaExternalDataError("rule operator family 未启用")
        if (not isinstance(self.operator, ObjectIdentity)
                or self.operator.object_kind != OBJECT_MINIMAL_INSTRUCTION
                or type(self.operator_version) is not int
                or self.operator_version <= 0
                or not isinstance(self.schema, ObjectIdentity)
                or self.schema.object_kind != OBJECT_STRUCTURE_CONCEPT
                or self.direction not in SOURCE_INFERENCE_DIRECTIONS):
            raise BroadQaExternalDataError("learned rule identity 非法")
        if self.application_domain != SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS[
                self.operator_family]:
            raise BroadQaExternalDataError("learned rule application domain 漂移")
        if (not isinstance(self.defeaters, tuple) or not self.defeaters
                or any(not isinstance(item, ObjectIdentity)
                       or item.object_kind != OBJECT_CONCEPT
                       for item in self.defeaters)
                or tuple(item.stable_key() for item in self.defeaters)
                != tuple(sorted({item.stable_key() for item in self.defeaters}))):
            raise BroadQaExternalDataError(
                "learned rule defeaters 必须非空唯一排序")
        if (not isinstance(self.evidence_commitments, tuple)
                or not self.evidence_commitments
                or any(not isinstance(
                    item, BroadQaSourceInferenceRuleEvidenceCommitment)
                       for item in self.evidence_commitments)):
            raise BroadQaExternalDataError("learned rule Evidence commitments 非法")
        evidence_keys = tuple(
            item.evidence_key for item in self.evidence_commitments)
        if evidence_keys != tuple(sorted(set(evidence_keys))):
            raise BroadQaExternalDataError(
                "learned rule Evidence 必须唯一规范排序")
        qualification_kinds = {
            item.qualification_kind for item in self.evidence_commitments}
        if qualification_kinds != set(
                SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_KINDS):
            raise BroadQaExternalDataError(
                "learned rule 必须同时保留正例和反例 Evidence")
        expected_hypothesis = source_inference_rule_hypothesis_key(
            self.operator,
            self.schema,
            self.direction,
            self.operator_version,
            source_inference_protocol_scope(protocol_sha),
        )
        for commitment in self.evidence_commitments:
            evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
            expected_reason = SOURCE_INFERENCE_EVIDENCE_REASON_KEYS[
                self.operator_family][commitment.qualification_kind]
            if (evidence.hypothesis != expected_hypothesis
                    or evidence.reason_key != expected_reason):
                raise BroadQaExternalDataError(
                    "learned rule Evidence 未绑定协议来源规则或资格理由")
        if (self.runtime_state != SOURCE_INFERENCE_RULE_RUNTIME_STATE
                or type(self.production_enabled) is not int
                or self.production_enabled != 0
                or any(type(value) is not int or value != 0 for value in (
                    self.item_identity_dispatch,
                    self.title_identity_dispatch,
                    self.page_identity_dispatch,
                ))):
            raise BroadQaExternalDataError(
                "learned rule 不得启用生产或逐 identity dispatch")

    def to_dict(self) -> dict[str, object]:
        """导出严格 rule record，不隐藏 Evidence 或适用域。"""
        return {
            "application_domain": self.application_domain,
            "artifact_kind": SOURCE_INFERENCE_RULE_RECORD_KIND,
            "defeater_keys": [
                list(item.stable_key()) for item in self.defeaters],
            "direction": self.direction,
            "evidence_commitments": [
                item.to_dict() for item in self.evidence_commitments],
            "format_version": 1,
            "item_identity_dispatch": self.item_identity_dispatch,
            "operator_family": self.operator_family,
            "operator_key": list(self.operator.stable_key()),
            "operator_version": self.operator_version,
            "page_identity_dispatch": self.page_identity_dispatch,
            "production_enabled": self.production_enabled,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "runtime_state": self.runtime_state,
            "schema_key": list(self.schema.stable_key()),
            "title_identity_dispatch": self.title_identity_dispatch,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 learned rule 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回 learned rule 的规范内容摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceLearnedRule":
        """从字段精确的 JSON object 回读 learned rule。"""
        expected = {
            "application_domain", "artifact_kind", "defeater_keys",
            "direction", "evidence_commitments", "format_version",
            "item_identity_dispatch", "operator_family", "operator_key",
            "operator_version", "page_identity_dispatch", "production_enabled",
            "protocol_manifest_sha256", "runtime_state", "schema_key",
            "title_identity_dispatch",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"] != SOURCE_INFERENCE_RULE_RECORD_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or not isinstance(value["defeater_keys"], list)
                or not isinstance(value["evidence_commitments"], list)):
            raise BroadQaExternalDataError("learned rule record 字段漂移")
        return cls(
            value["protocol_manifest_sha256"],
            value["operator_family"],
            _identity(
                value["operator_key"],
                label="learned rule operator",
                object_kind=OBJECT_MINIMAL_INSTRUCTION,
            ),
            value["operator_version"],
            _identity(
                value["schema_key"],
                label="learned rule schema",
                object_kind=OBJECT_STRUCTURE_CONCEPT,
            ),
            value["direction"],
            value["application_domain"],
            tuple(_identity(
                item,
                label="learned rule defeater",
                object_kind=OBJECT_CONCEPT,
            ) for item in value["defeater_keys"]),
            tuple(BroadQaSourceInferenceRuleEvidenceCommitment.from_dict(item)
                  for item in value["evidence_commitments"]),
            value["runtime_state"],
            value["production_enabled"],
            value["item_identity_dispatch"],
            value["title_identity_dispatch"],
            value["page_identity_dispatch"],
        )


def parse_source_inference_learned_rule(
        payload: bytes,
        ) -> BroadQaSourceInferenceLearnedRule:
    """严格回读单条规范 rule，拒绝未知字段和非规范编码。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError("learned rule 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError("learned rule 不是规范 JSON") from error
    rule = BroadQaSourceInferenceLearnedRule.from_dict(value)
    if rule.canonical_bytes() != payload:
        raise BroadQaExternalDataError("learned rule 字节漂移")
    return rule


def _read_split_inventory(path: Path) -> dict[str, str]:
    """回读全局 item split inventory，供 pack 阻断泄漏。"""
    values = {}
    expected = {
        "bucket", "format_version", "item_id", "record_kind", "source_key",
        "split", "training_assignment",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                item_id = value.get("item_id") if isinstance(value, dict) else None
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or set(value) != expected
                        or type(value.get("format_version")) is not int
                        or value.get("format_version") != 1
                        or value.get("record_kind")
                        != "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNING_SPLIT_RECORD_V1"
                        or type(value.get("bucket")) is not int
                        or not 0 <= value["bucket"] < 8
                        or not isinstance(item_id, str) or item_id in values
                        or value.get("split") not in (
                            "TRAIN", "VALIDATION", "RESERVE")):
                    raise BroadQaExternalDataError(
                        f"rule pack split inventory 漂移: {line_number}")
                values[item_id] = value["split"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("rule pack split inventory 不可读") from error
    return values


def _validate_rule_training_commitments(
        rule: BroadQaSourceInferenceLearnedRule,
        *,
        dossier_by_id: dict[str, dict[str, object]],
        census_by_id: dict[str, dict[str, object]],
        split_by_item: dict[str, str],
        ) -> None:
    """逐条核验 rule Evidence 只引用 TRAIN 物理切片中的真实 span。"""
    for commitment in rule.evidence_commitments:
        if split_by_item.get(commitment.item_id) != "TRAIN":
            raise BroadQaExternalDataError(
                "learned rule 读取了非 TRAIN item")
        dossier = dossier_by_id.get(commitment.item_id)
        census = census_by_id.get(commitment.item_id)
        if dossier is None or census is None:
            raise BroadQaExternalDataError(
                "learned rule Evidence 不在 family TRAIN slice")
        terminal = dossier["terminal_source"]
        training = dossier["training_source"]
        passages = {
            (
                item["ordinal"], item["raw_start"], item["raw_end"],
                item["raw_sha256"],
            )
            for item in terminal["passages"]
        }
        expected_span = (
            commitment.passage_ordinal,
            commitment.raw_start,
            commitment.raw_end,
            commitment.raw_sha256,
        )
        expected_source = SourceRef(
            SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND,
            terminal["page_id"],
            terminal["revision_id"],
            GLOBAL_OWNER_SCOPE,
            VersionBundle(),
        )
        expected_answer_shas = {
            hashlib.sha256(normalize_external_text(answer).encode(
                "utf-8")).hexdigest()
            for answer in training["gold_answers"]
        }
        wikitext = terminal["wikitext"]
        candidate_raw = wikitext[
            commitment.candidate_raw_start:commitment.candidate_raw_end]
        replayed_observed_sha = hashlib.sha256(
            normalize_external_text(candidate_raw).encode("utf-8")
        ).hexdigest()
        replayed_kind = (
            "REPLAYED_CANDIDATE_SUPPORT"
            if replayed_observed_sha == commitment.qualification_expected_sha256
            else "REPLAYED_CANDIDATE_REFUTE")
        if (commitment.source_key != training["source_key"]
                or commitment.source_ref != expected_source
                or commitment.page_id != terminal["page_id"]
                or commitment.revision_id != terminal["revision_id"]
                or expected_span not in passages
                or hashlib.sha256(candidate_raw.encode("utf-8")).hexdigest()
                != commitment.candidate_raw_sha256
                or commitment.routing_signal_state
                != census["mechanical_signal_state"]
                or commitment.qualification_input_sha256
                != source_inference_qualification_input_sha256(
                    rule.operator_family, dossier)
                or commitment.qualification_expected_sha256
                not in expected_answer_shas
                or commitment.qualification_observed_sha256
                != replayed_observed_sha
                or commitment.qualification_kind != replayed_kind):
            raise BroadQaExternalDataError(
                "learned rule source/item/span commitment 漂移")


def source_inference_rule_pack_result_sha256(
        *,
        protocol_manifest_sha256: str,
        operator_family: str,
        training_item_ids: tuple[str, ...],
        rules: tuple[BroadQaSourceInferenceLearnedRule, ...],
        ) -> str:
    """从规范规则与 Evidence 重算 learner 唯一结果摘要。"""
    if (not isinstance(rules, tuple) or not rules
            or any(not isinstance(item, BroadQaSourceInferenceLearnedRule)
                   for item in rules)):
        raise BroadQaExternalDataError("rule pack result rules 非法")
    evidence_shas = tuple(sorted({
        hashlib.sha256(canonical_json_bytes(
            list(commitment.evidence_key))).hexdigest()
        for rule in rules
        for commitment in rule.evidence_commitments
    }))
    rule_shas = tuple(sorted(rule.sha256() for rule in rules))
    return source_inference_learning_result_sha256(
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=operator_family,
        processed_item_ids=training_item_ids,
        evidence_record_sha256s=evidence_shas,
        rule_record_sha256s=rule_shas,
    )


def publish_source_inference_rule_pack(
        *,
        protocol_dir: str | Path,
        operator_family: str,
        fresh_rules: tuple[BroadQaSourceInferenceLearnedRule, ...],
        resumed_rules: tuple[BroadQaSourceInferenceLearnedRule, ...],
        target_dir: str | Path,
        fresh_checkpoint_chain_path: str | Path,
        resumed_checkpoint_chain_path: str | Path,
        ) -> dict[str, object]:
    """验证并不可覆盖发布单 family rule pack；不启用生产 consumer。"""
    protocol_root = Path(protocol_dir).resolve()
    protocol = read_source_inference_learning_protocol(
        protocol_root / "manifest.json")
    protocol_sha = protocol["manifest_sha256"]
    if operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
        raise BroadQaExternalDataError("rule pack family 未启用")
    if (not isinstance(fresh_rules, tuple) or not fresh_rules
            or not isinstance(resumed_rules, tuple) or not resumed_rules
            or any(not isinstance(item, BroadQaSourceInferenceLearnedRule)
                   for item in fresh_rules + resumed_rules)
            or any(item.operator_family != operator_family
                   or item.protocol_manifest_sha256 != protocol_sha
                   for item in fresh_rules + resumed_rules)):
        raise BroadQaExternalDataError("rule pack rules/family/protocol 漂移")
    fresh_payload = b"".join(item.canonical_bytes() for item in fresh_rules)
    resumed_payload = b"".join(item.canonical_bytes() for item in resumed_rules)
    if fresh_payload != resumed_payload:
        raise BroadQaExternalDataError(
            "rule pack fresh/resume rule 字节不等价")
    rules = fresh_rules
    rule_shas = tuple(item.sha256() for item in rules)
    if rule_shas != tuple(sorted(set(rule_shas))):
        raise BroadQaExternalDataError("rule pack rules 必须按 SHA 唯一排序")

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
    if (not inventory_path.is_relative_to(protocol_root)
            or hashlib.sha256(inventory_path.read_bytes()).hexdigest()
            != inventory_identity["sha256"]):
        raise BroadQaExternalDataError("rule pack split inventory commitment 漂移")
    split_by_item = _read_split_inventory(inventory_path)
    for rule in rules:
        _validate_rule_training_commitments(
            rule,
            dossier_by_id=dossier_by_id,
            census_by_id=census_by_id,
            split_by_item=split_by_item,
        )

    maximum_evidence = (
        len(training_item_ids)
        * protocol["learning_stopping_contract"][
            "maximum_evidence_candidates_per_item"])
    maximum_rules = protocol["learning_stopping_contract"][
        "maximum_rule_candidates_per_family"]
    output_evidence_count = sum(
        len(rule.evidence_commitments) for rule in rules)
    chain_paths = tuple(Path(path).resolve() for path in (
        fresh_checkpoint_chain_path, resumed_checkpoint_chain_path))
    if (chain_paths[0] == chain_paths[1]
            or any(not path.is_relative_to(protocol_root.parent)
                   for path in chain_paths)):
        raise BroadQaExternalDataError(
            "rule pack fresh/resume checkpoint 必须独立且位于 run root")
    checkpoint_lineage = []
    for chain_path in chain_paths:
        checkpoint_chain = read_source_inference_learning_chain(chain_path)
        checkpoint = checkpoint_chain[-1]
        checkpoint_lineage.append({
            "chain_sha256": hashlib.sha256(
                chain_path.read_bytes()).hexdigest(),
            "run_id": checkpoint.run_id,
            "terminal_checkpoint_sha256": checkpoint.sha256(),
        })
        if (checkpoint.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
                or checkpoint.protocol_manifest_sha256 != protocol_sha
                or checkpoint.operator_family != operator_family
                or checkpoint.training_item_count != len(training_item_ids)
                or checkpoint.training_item_order_sha256
                != source_inference_learning_prefix_sha256(training_item_ids)
                or not output_evidence_count
                <= checkpoint.evidence_candidate_count <= maximum_evidence
                or not len(rules)
                <= checkpoint.rule_candidate_count <= maximum_rules):
            raise BroadQaExternalDataError(
                "rule pack checkpoint 未证明完整 TRAIN 或候选预算")
    if len({item["run_id"] for item in checkpoint_lineage}) != 2:
        raise BroadQaExternalDataError(
            "rule pack fresh/resume checkpoint run identity 必须独立")
    expected_result_sha = source_inference_rule_pack_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        training_item_ids=training_item_ids,
        rules=rules,
    )
    fresh_result_sha256 = expected_result_sha
    resumed_result_sha256 = source_inference_rule_pack_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family=operator_family,
        training_item_ids=training_item_ids,
        rules=resumed_rules,
    )
    if fresh_result_sha256 != resumed_result_sha256:
        raise BroadQaExternalDataError("rule pack fresh/resume 重算结果不等价")

    target = Path(target_dir).resolve()
    if not target.is_relative_to(protocol_root.parent):
        raise BroadQaExternalDataError(
            "rule pack target 必须位于 protocol run root 内")
    if target.exists():
        raise BroadQaExternalDataError("rule pack target 已存在")
    target.mkdir(parents=True)
    records_path = target / "rules.jsonl"
    with records_path.open("xb") as handle:
        for rule in rules:
            handle.write(rule.canonical_bytes())
    records_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_RULE_PACK_KIND,
        "format_version": 1,
        "fresh_checkpoint_chain_sha256": checkpoint_lineage[0][
            "chain_sha256"],
        "fresh_checkpoint_terminal_sha256": checkpoint_lineage[0][
            "terminal_checkpoint_sha256"],
        "fresh_result_sha256": fresh_result_sha256,
        "fresh_run_id": checkpoint_lineage[0]["run_id"],
        "operator_family": operator_family,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "record_count": len(rules),
        "records_bytes": records_path.stat().st_size,
        "records_sha256": records_sha,
        "resumed_checkpoint_chain_sha256": checkpoint_lineage[1][
            "chain_sha256"],
        "resumed_checkpoint_terminal_sha256": checkpoint_lineage[1][
            "terminal_checkpoint_sha256"],
        "resumed_result_sha256": resumed_result_sha256,
        "resumed_run_id": checkpoint_lineage[1]["run_id"],
        "runtime_state": SOURCE_INFERENCE_RULE_RUNTIME_STATE,
        "status": "FROZEN_NOT_EVALUATED_NOT_DEPLOYED",
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


def read_source_inference_rule_pack(
        target_dir: str | Path,
        ) -> tuple[dict[str, object], tuple[BroadQaSourceInferenceLearnedRule, ...]]:
    """严格回读 rule pack manifest 和全部规范 records。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    records_path = root / "rules.jsonl"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
        records_payload = records_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("rule pack 不可读") from error
    expected = {
        "artifact_kind", "format_version", "fresh_checkpoint_chain_sha256",
        "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
        "fresh_run_id",
        "operator_family", "production_enabled", "protocol_manifest_sha256",
        "record_count", "records_bytes", "records_sha256",
        "resumed_checkpoint_chain_sha256",
        "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
        "resumed_run_id", "runtime_state", "status", "training_item_count",
        "training_item_order_sha256",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != payload
            or manifest["artifact_kind"] != SOURCE_INFERENCE_RULE_PACK_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"]
            not in SOURCE_INFERENCE_LEARNING_FAMILIES
            or manifest["production_enabled"] != 0
            or manifest["runtime_state"]
            != SOURCE_INFERENCE_RULE_RUNTIME_STATE
            or manifest["status"] != "FROZEN_NOT_EVALUATED_NOT_DEPLOYED"
            or type(manifest["record_count"]) is not int
            or manifest["record_count"] <= 0
            or type(manifest["records_bytes"]) is not int
            or manifest["records_bytes"] <= 0
            or type(manifest["production_enabled"]) is not int
            or type(manifest["training_item_count"]) is not int
            or manifest["training_item_count"] <= 0
            or manifest["fresh_run_id"] == manifest["resumed_run_id"]
            or manifest["records_bytes"] != len(records_payload)
            or manifest["records_sha256"]
            != hashlib.sha256(records_payload).hexdigest()):
        raise BroadQaExternalDataError("rule pack manifest 漂移")
    for name in (
            "fresh_checkpoint_chain_sha256",
            "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
            "fresh_run_id", "protocol_manifest_sha256", "records_sha256",
            "resumed_checkpoint_chain_sha256",
            "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
            "resumed_run_id", "training_item_order_sha256"):
        _sha256(manifest[name], label=f"rule pack {name}")
    if manifest["fresh_result_sha256"] != manifest["resumed_result_sha256"]:
        raise BroadQaExternalDataError("rule pack fresh/resume 漂移")
    if not records_payload.endswith(b"\n"):
        raise BroadQaExternalDataError("rule pack records 截断")
    rules = tuple(
        parse_source_inference_learned_rule(line + b"\n")
        for line in records_payload.splitlines()
    )
    rule_shas = tuple(item.sha256() for item in rules)
    if (len(rules) != manifest["record_count"]
            or rule_shas != tuple(sorted(set(rule_shas)))
            or any(item.operator_family != manifest["operator_family"]
                   or item.protocol_manifest_sha256
                   != manifest["protocol_manifest_sha256"]
                   for item in rules)):
        raise BroadQaExternalDataError("rule pack record count/identity 漂移")
    return ({
        **manifest,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }, rules)


__all__ = [
    "BroadQaSourceInferenceLearnedRule",
    "BroadQaSourceInferenceRuleEvidenceCommitment",
    "SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS",
    "SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_KINDS",
    "SOURCE_INFERENCE_EVIDENCE_REASON_KEYS",
    "SOURCE_INFERENCE_RULE_PACK_KIND",
    "SOURCE_INFERENCE_RULE_RECORD_KIND",
    "SOURCE_INFERENCE_RULE_RUNTIME_STATE",
    "SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND",
    "parse_source_inference_learned_rule",
    "publish_source_inference_rule_pack",
    "read_source_inference_rule_pack",
    "source_inference_qualification_input_sha256",
    "source_inference_rule_pack_result_sha256",
]
