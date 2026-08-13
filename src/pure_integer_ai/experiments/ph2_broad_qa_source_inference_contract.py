"""广域问答来源内可审计归纳的纯合同。

本模块只补齐来源文本到共享 typed relation 之间的接地边界，不发现自然语言
同义词、不执行推导，也不接入生产查询。调用方必须显式提供来源 observation、
角色化 premise、规则身份、负条件和逐字输出路径；任何身份或字节漂移均失败关闭。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    SEMANTIC_OBJECT_KINDS,
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    semantic_source,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


SOURCE_INFERENCE_RECORD_KIND = "PH2_BROAD_QA_SOURCE_INFERENCE_RECORD_V1"
SOURCE_INFERENCE_DIRECTIONS = ("FORWARD", "REVERSE")
SOURCE_INFERENCE_EPISTEMIC_STATUS = "SOURCE_DERIVED_FROM_ASSERTIONS"
SOURCE_INFERENCE_TRUTH_STATUS = "NOT_ADJUDICATED"
SOURCE_INFERENCE_RUNTIME_STATE = "CONTRACT_ONLY_DISABLED"


# object-model: exception
class BroadQaSourceInferenceError(ValueError):
    """来源内归纳合同、稳定身份或规范字节发生漂移。"""


def _exact(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    """要求 JSON object 的字段集合精确匹配合同。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise BroadQaSourceInferenceError(f"{label} 字段漂移")
    return value


def _text(value: object, *, label: str) -> str:
    """要求值是无首尾空白的非空文本。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BroadQaSourceInferenceError(f"{label} 必须是规范文本")
    return value


def _sha256(value: object, *, label: str) -> str:
    """要求值是小写 SHA-256 文本。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaSourceInferenceError(f"{label} 必须是 SHA-256")
    return value


def _positive(value: object, *, label: str) -> int:
    """要求值是正严格整数。"""
    if type(value) is not int or value <= 0:
        raise BroadQaSourceInferenceError(f"{label} 必须是正严格整数")
    return value


def _nonnegative(value: object, *, label: str) -> int:
    """要求值是非负严格整数。"""
    if type(value) is not int or value < 0:
        raise BroadQaSourceInferenceError(f"{label} 必须是非负严格整数")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """恢复非空严格整数键，拒绝 bool 和整数子类。"""
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int for item in value)):
        raise BroadQaSourceInferenceError(f"{label} 必须是严格整数键")
    return tuple(value)


def _identity(value: object, *, label: str) -> ObjectIdentity:
    """从完整稳定键恢复一等对象身份。"""
    try:
        return ObjectIdentity.from_stable_key(_strict_key(value, label=label))
    except (TypeError, ValueError) as error:
        raise BroadQaSourceInferenceError(f"{label} 身份非法") from error


def _source(value: object) -> SourceRef:
    """从完整稳定键恢复来源身份。"""
    try:
        return SourceRef.from_stable_key(_strict_key(value, label="source_ref"))
    except (TypeError, ValueError) as error:
        raise BroadQaSourceInferenceError("source_ref 身份非法") from error


def _scope(value: object) -> ScopeIdentity:
    """从完整稳定键恢复适用域身份。"""
    try:
        return ScopeIdentity.from_stable_key(
            _strict_key(value, label="applicability_scope"))
    except (TypeError, ValueError) as error:
        raise BroadQaSourceInferenceError("applicability_scope 身份非法") from error


def _binding_key(binding: AtomicRoleBinding) -> tuple[int, ...]:
    """返回 Role、ordinal 和 filler 的完整稳定排序键。"""
    return (
        *binding.role.stable_key(),
        binding.ordinal,
        *binding.filler.stable_key(),
    )


def _binding_dict(binding: AtomicRoleBinding) -> dict[str, object]:
    """导出共享原子角色绑定的规范 JSON 值。"""
    return {
        "filler_key": list(binding.filler.stable_key()),
        "ordinal": binding.ordinal,
        "role_key": list(binding.role.stable_key()),
    }


def _binding_from_dict(value: object) -> AtomicRoleBinding:
    """从字段精确的 JSON object 恢复共享原子角色绑定。"""
    raw = _exact(
        value, {"filler_key", "ordinal", "role_key"}, label="role binding")
    try:
        return AtomicRoleBinding(
            _identity(raw["role_key"], label="role binding role"),
            _identity(raw["filler_key"], label="role binding filler"),
            _nonnegative(raw["ordinal"], label="role binding ordinal"),
        )
    except (TypeError, ValueError) as error:
        raise BroadQaSourceInferenceError("role binding 非法") from error


def _definition_dict(
        value: AtomicPropositionDefinition,
        ) -> dict[str, object]:
    """导出共享原子命题定义，不复制另一套命题模型。"""
    return {
        "bindings": [_binding_dict(item) for item in value.bindings],
        "context_key": list(value.context.stable_key()),
        "predicate_key": list(value.predicate.stable_key()),
        "proposition_key": list(value.proposition.stable_key()),
        "source_anchor_key": list(value.source_anchor.stable_key()),
    }


def _definition_from_dict(value: object) -> AtomicPropositionDefinition:
    """从字段精确的 JSON object 恢复共享原子命题定义。"""
    raw = _exact(value, {
        "bindings", "context_key", "predicate_key", "proposition_key",
        "source_anchor_key",
    }, label="proposition definition")
    if not isinstance(raw["bindings"], list):
        raise BroadQaSourceInferenceError("proposition bindings 必须是数组")
    try:
        return AtomicPropositionDefinition(
            _identity(raw["proposition_key"], label="proposition"),
            _identity(raw["predicate_key"], label="predicate"),
            _identity(raw["source_anchor_key"], label="source anchor"),
            _identity(raw["context_key"], label="context"),
            tuple(_binding_from_dict(item) for item in raw["bindings"]),
        )
    except (TypeError, ValueError) as error:
        raise BroadQaSourceInferenceError("proposition definition 非法") from error


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceObservation:
    """一条绑定页面修订、原始 span 和显示文本的来源 observation。"""

    source: SourceRef
    snapshot_id: str
    license_id: str
    title: str
    page_id: int
    revision_id: int
    passage_ordinal: int
    raw_start: int
    raw_end: int
    raw_sha256: str
    evidence_text: str
    selected_start: int
    selected_end: int
    selected_text: str

    def __post_init__(self) -> None:
        """核验来源坐标和选中文本逐字对应，禁止只保存摘要。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("observation source 必须是 SourceRef")
        _text(self.snapshot_id, label="observation snapshot_id")
        _text(self.license_id, label="observation license_id")
        _text(self.title, label="observation title")
        _positive(self.page_id, label="observation page_id")
        _positive(self.revision_id, label="observation revision_id")
        _positive(self.passage_ordinal, label="observation passage_ordinal")
        if (self.source.source_id != self.page_id
                or self.source.document_id != self.revision_id):
            raise BroadQaSourceInferenceError(
                "observation SourceRef 与 page/revision 不一致")
        _nonnegative(self.raw_start, label="observation raw_start")
        _positive(self.raw_end, label="observation raw_end")
        if self.raw_end <= self.raw_start:
            raise BroadQaSourceInferenceError("observation raw span 非法")
        _sha256(self.raw_sha256, label="observation raw_sha256")
        _text(self.evidence_text, label="observation evidence_text")
        _nonnegative(self.selected_start, label="observation selected_start")
        _positive(self.selected_end, label="observation selected_end")
        if (self.selected_end <= self.selected_start
                or self.selected_end > len(self.evidence_text)
                or self.evidence_text[
                    self.selected_start:self.selected_end] != self.selected_text):
            raise BroadQaSourceInferenceError(
                "observation selected span 与来源文本不一致")

    def to_dict(self) -> dict[str, object]:
        """导出完整且不丢失来源字节坐标的规范 JSON 值。"""
        return {
            "evidence_text": self.evidence_text,
            "license_id": self.license_id,
            "page_id": self.page_id,
            "passage_ordinal": self.passage_ordinal,
            "raw_end": self.raw_end,
            "raw_sha256": self.raw_sha256,
            "raw_start": self.raw_start,
            "revision_id": self.revision_id,
            "selected_end": self.selected_end,
            "selected_start": self.selected_start,
            "selected_text": self.selected_text,
            "snapshot_id": self.snapshot_id,
            "source_ref": list(self.source.stable_key()),
            "title": self.title,
        }

    def sha256(self) -> str:
        """返回 observation 的规范内容承诺。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceObservation":
        """从字段精确的 JSON object 恢复 observation。"""
        raw = _exact(value, {
            "evidence_text", "license_id", "page_id", "passage_ordinal",
            "raw_end", "raw_sha256", "raw_start", "revision_id",
            "selected_end", "selected_start", "selected_text", "snapshot_id",
            "source_ref", "title",
        }, label="source observation")
        return cls(
            _source(raw["source_ref"]), raw["snapshot_id"], raw["license_id"],
            raw["title"], raw["page_id"], raw["revision_id"],
            raw["passage_ordinal"], raw["raw_start"], raw["raw_end"],
            raw["raw_sha256"], raw["evidence_text"], raw["selected_start"],
            raw["selected_end"], raw["selected_text"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceRoleSpan:
    """把一个 typed RoleBinding 绑定到 premise observation 的逐字 span。"""

    binding: AtomicRoleBinding
    start: int
    end: int
    surface: str

    def __post_init__(self) -> None:
        """核验角色赋值和局部 span；文本对应由 premise 统一核验。"""
        if not isinstance(self.binding, AtomicRoleBinding):
            raise TypeError("source role span binding 类型非法")
        _nonnegative(self.start, label="source role span start")
        _positive(self.end, label="source role span end")
        if self.end <= self.start:
            raise BroadQaSourceInferenceError("source role span 边界非法")
        _text(self.surface, label="source role span surface")

    def to_dict(self) -> dict[str, object]:
        """导出角色绑定及其来源 span。"""
        return {
            "binding": _binding_dict(self.binding),
            "end": self.end,
            "start": self.start,
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceRoleSpan":
        """从字段精确的 JSON object 恢复角色 span。"""
        raw = _exact(
            value, {"binding", "end", "start", "surface"},
            label="source role span")
        return cls(
            _binding_from_dict(raw["binding"]), raw["start"], raw["end"],
            raw["surface"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourcePremise:
    """一个不裁定绝对真值的来源声明命题及其角色字节证据。"""

    observation: BroadQaSourceObservation
    definition: AtomicPropositionDefinition
    role_spans: tuple[BroadQaSourceRoleSpan, ...]
    epistemic_status: str = "SOURCE_ASSERTED"
    truth_status: str = SOURCE_INFERENCE_TRUTH_STATUS

    def __post_init__(self) -> None:
        """核验命题来源、角色完整性和每个论元的逐字位置。"""
        if not isinstance(self.observation, BroadQaSourceObservation):
            raise TypeError("premise observation 类型非法")
        if not isinstance(self.definition, AtomicPropositionDefinition):
            raise TypeError("premise definition 类型非法")
        if (semantic_source(self.definition.proposition) != self.observation.source
                or semantic_source(self.definition.context)
                != self.observation.source):
            raise BroadQaSourceInferenceError("premise 命题未绑定同一来源")
        expected_anchor = occurrence_identity(
            self.observation.source,
            start=self.observation.raw_start,
            end=self.observation.raw_end,
            ordinal=self.observation.passage_ordinal,
        )
        if self.definition.source_anchor != expected_anchor:
            raise BroadQaSourceInferenceError(
                "premise source anchor 未精确绑定 observation raw span")
        if self.definition.bindings != tuple(sorted(
                self.definition.bindings, key=_binding_key)):
            raise BroadQaSourceInferenceError("premise bindings 未规范排序")
        if (not isinstance(self.role_spans, tuple) or not self.role_spans
                or any(not isinstance(item, BroadQaSourceRoleSpan)
                       for item in self.role_spans)):
            raise BroadQaSourceInferenceError("premise role spans 非法")
        span_bindings = tuple(item.binding for item in self.role_spans)
        if (span_bindings != self.definition.bindings
                or len(set(_binding_key(item) for item in span_bindings))
                != len(span_bindings)):
            raise BroadQaSourceInferenceError(
                "premise role spans 未精确覆盖命题 bindings")
        for item in self.role_spans:
            if (item.end > len(self.observation.evidence_text)
                    or self.observation.evidence_text[
                        item.start:item.end] != item.surface):
                raise BroadQaSourceInferenceError(
                    "premise role span 与 observation 文本不一致")
            filler = item.binding.filler
            if (filler.object_kind in SEMANTIC_OBJECT_KINDS
                    and filler.object_kind != OBJECT_ROLE
                    and semantic_source(filler) != self.observation.source):
                raise BroadQaSourceInferenceError(
                    "premise 来源化 filler 跨越了 observation")
        if (self.epistemic_status != "SOURCE_ASSERTED"
                or self.truth_status != SOURCE_INFERENCE_TRUTH_STATUS):
            raise BroadQaSourceInferenceError("premise 真值边界漂移")

    def to_dict(self) -> dict[str, object]:
        """导出来源声明、命题和论元 span。"""
        return {
            "definition": _definition_dict(self.definition),
            "epistemic_status": self.epistemic_status,
            "observation": self.observation.to_dict(),
            "role_spans": [item.to_dict() for item in self.role_spans],
            "truth_status": self.truth_status,
        }

    def sha256(self) -> str:
        """返回 premise 全链路的规范内容承诺。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourcePremise":
        """从字段精确的 JSON object 恢复 premise。"""
        raw = _exact(value, {
            "definition", "epistemic_status", "observation", "role_spans",
            "truth_status",
        }, label="source premise")
        if not isinstance(raw["role_spans"], list):
            raise BroadQaSourceInferenceError("premise role_spans 必须是数组")
        return cls(
            BroadQaSourceObservation.from_dict(raw["observation"]),
            _definition_from_dict(raw["definition"]),
            tuple(BroadQaSourceRoleSpan.from_dict(item)
                  for item in raw["role_spans"]),
            raw["epistemic_status"], raw["truth_status"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceRoleProjection:
    """声明一个结果 RoleBinding 必须复用哪条 premise 的哪个 filler。"""

    result_role: ObjectIdentity
    result_ordinal: int
    premise_index: int
    premise_role: ObjectIdentity
    premise_ordinal: int

    def __post_init__(self) -> None:
        """核验两端角色身份和零基 premise 坐标。"""
        if (not isinstance(self.result_role, ObjectIdentity)
                or self.result_role.object_kind != OBJECT_ROLE
                or not isinstance(self.premise_role, ObjectIdentity)
                or self.premise_role.object_kind != OBJECT_ROLE):
            raise BroadQaSourceInferenceError("role projection 两端必须是 Role")
        _nonnegative(self.result_ordinal, label="projection result ordinal")
        _nonnegative(self.premise_index, label="projection premise index")
        _nonnegative(self.premise_ordinal, label="projection premise ordinal")

    def to_dict(self) -> dict[str, object]:
        """导出结果角色到 premise 角色的完整映射。"""
        return {
            "premise_index": self.premise_index,
            "premise_ordinal": self.premise_ordinal,
            "premise_role_key": list(self.premise_role.stable_key()),
            "result_ordinal": self.result_ordinal,
            "result_role_key": list(self.result_role.stable_key()),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceRoleProjection":
        """从字段精确的 JSON object 恢复角色映射。"""
        raw = _exact(value, {
            "premise_index", "premise_ordinal", "premise_role_key",
            "result_ordinal", "result_role_key",
        }, label="role projection")
        return cls(
            _identity(raw["result_role_key"], label="projection result role"),
            raw["result_ordinal"], raw["premise_index"],
            _identity(raw["premise_role_key"], label="projection premise role"),
            raw["premise_ordinal"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceDerivation:
    """保存规则身份、适用域、完整前提顺序和已清除负条件。"""

    operator: ObjectIdentity
    operator_version: int
    schema: ObjectIdentity
    direction: str
    applicability_scope: ScopeIdentity
    premise_sha256s: tuple[str, ...]
    rule_evidence_keys: tuple[tuple[int, ...], ...]
    role_projections: tuple[BroadQaSourceRoleProjection, ...]
    defeaters: tuple[ObjectIdentity, ...]
    cleared_defeaters: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        """核验规则坐标、证据、映射和 fail-closed defeater 边界。"""
        if (not isinstance(self.operator, ObjectIdentity)
                or self.operator.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise BroadQaSourceInferenceError("derivation operator 类型非法")
        _positive(self.operator_version, label="derivation operator_version")
        if (not isinstance(self.schema, ObjectIdentity)
                or self.schema.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise BroadQaSourceInferenceError("derivation schema 类型非法")
        if self.direction not in SOURCE_INFERENCE_DIRECTIONS:
            raise BroadQaSourceInferenceError("derivation direction 未注册")
        if not isinstance(self.applicability_scope, ScopeIdentity):
            raise TypeError("derivation applicability_scope 类型非法")
        if (not isinstance(self.premise_sha256s, tuple)
                or not self.premise_sha256s):
            raise BroadQaSourceInferenceError("derivation 必须保留前提顺序")
        for item in self.premise_sha256s:
            _sha256(item, label="derivation premise SHA")
        if (not isinstance(self.rule_evidence_keys, tuple)
                or not self.rule_evidence_keys
                or self.rule_evidence_keys
                != tuple(sorted(set(self.rule_evidence_keys)))):
            raise BroadQaSourceInferenceError(
                "derivation rule Evidence 必须非空、唯一、规范排序")
        for item in self.rule_evidence_keys:
            _strict_key(item, label="derivation rule Evidence")
        if (not isinstance(self.role_projections, tuple)
                or not self.role_projections
                or any(not isinstance(item, BroadQaSourceRoleProjection)
                       for item in self.role_projections)):
            raise BroadQaSourceInferenceError("derivation role projections 非法")
        result_slots = tuple(
            (item.result_role.stable_key(), item.result_ordinal)
            for item in self.role_projections)
        if len(result_slots) != len(set(result_slots)):
            raise BroadQaSourceInferenceError(
                "derivation result Role+ordinal 不得重复")
        for name, values in (
                ("defeaters", self.defeaters),
                ("cleared_defeaters", self.cleared_defeaters)):
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ObjectIdentity)
                           for item in values)
                    or tuple(item.stable_key() for item in values)
                    != tuple(sorted({item.stable_key() for item in values}))):
                raise BroadQaSourceInferenceError(
                    f"derivation {name} 必须唯一规范排序")
        if self.cleared_defeaters != self.defeaters:
            raise BroadQaSourceInferenceError(
                "derivation 存在未清除或伪造的 defeater")

    def to_dict(self) -> dict[str, object]:
        """导出完整规则承诺，不折叠 premise 或负条件身份。"""
        return {
            "applicability_scope_key": list(
                self.applicability_scope.stable_key()),
            "cleared_defeater_keys": [
                list(item.stable_key()) for item in self.cleared_defeaters],
            "defeater_keys": [list(item.stable_key()) for item in self.defeaters],
            "direction": self.direction,
            "operator_key": list(self.operator.stable_key()),
            "operator_version": self.operator_version,
            "premise_sha256s": list(self.premise_sha256s),
            "role_projections": [
                item.to_dict() for item in self.role_projections],
            "rule_evidence_keys": [list(item) for item in self.rule_evidence_keys],
            "schema_key": list(self.schema.stable_key()),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceDerivation":
        """从字段精确的 JSON object 恢复 derivation。"""
        raw = _exact(value, {
            "applicability_scope_key", "cleared_defeater_keys",
            "defeater_keys", "direction", "operator_key", "operator_version",
            "premise_sha256s", "role_projections", "rule_evidence_keys",
            "schema_key",
        }, label="source derivation")
        for label in (
                "cleared_defeater_keys", "defeater_keys", "premise_sha256s",
                "role_projections", "rule_evidence_keys"):
            if not isinstance(raw[label], list):
                raise BroadQaSourceInferenceError(
                    f"derivation {label} 必须是数组")
        return cls(
            _identity(raw["operator_key"], label="derivation operator"),
            raw["operator_version"],
            _identity(raw["schema_key"], label="derivation schema"),
            raw["direction"], _scope(raw["applicability_scope_key"]),
            tuple(raw["premise_sha256s"]),
            tuple(_strict_key(item, label="derivation rule Evidence")
                  for item in raw["rule_evidence_keys"]),
            tuple(BroadQaSourceRoleProjection.from_dict(item)
                  for item in raw["role_projections"]),
            tuple(_identity(item, label="derivation defeater")
                  for item in raw["defeater_keys"]),
            tuple(_identity(item, label="derivation cleared defeater")
                  for item in raw["cleared_defeater_keys"]),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceOutputPart:
    """声明回答中的一段文本来自哪个结果角色及其 premise 原文。"""

    result_role: ObjectIdentity
    result_ordinal: int
    surface: str

    def __post_init__(self) -> None:
        """核验结果角色和非空逐字表层。"""
        if (not isinstance(self.result_role, ObjectIdentity)
                or self.result_role.object_kind != OBJECT_ROLE):
            raise BroadQaSourceInferenceError("output part role 类型非法")
        _nonnegative(self.result_ordinal, label="output part ordinal")
        _text(self.surface, label="output part surface")

    def to_dict(self) -> dict[str, object]:
        """导出输出角色和逐字表层。"""
        return {
            "result_ordinal": self.result_ordinal,
            "result_role_key": list(self.result_role.stable_key()),
            "surface": self.surface,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceOutputPart":
        """从字段精确的 JSON object 恢复输出部分。"""
        raw = _exact(
            value, {"result_ordinal", "result_role_key", "surface"},
            label="source output part")
        return cls(
            _identity(raw["result_role_key"], label="output part role"),
            raw["result_ordinal"], raw["surface"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceDerivedClaim:
    """一个只复用 premise typed filler 和原文表层的来源内派生命题。"""

    premises: tuple[BroadQaSourcePremise, ...]
    derivation: BroadQaSourceDerivation
    definition: AtomicPropositionDefinition
    output_parts: tuple[BroadQaSourceOutputPart, ...]
    rendered_text: str
    epistemic_status: str = SOURCE_INFERENCE_EPISTEMIC_STATUS
    truth_status: str = SOURCE_INFERENCE_TRUTH_STATUS

    def __post_init__(self) -> None:
        """核验支持链、角色复用和输出文本均未越过来源字节。"""
        if (not isinstance(self.premises, tuple) or not self.premises
                or any(not isinstance(item, BroadQaSourcePremise)
                       for item in self.premises)):
            raise BroadQaSourceInferenceError("derived claim premises 非法")
        if not isinstance(self.derivation, BroadQaSourceDerivation):
            raise TypeError("derived claim derivation 类型非法")
        if self.derivation.premise_sha256s != tuple(
                item.sha256() for item in self.premises):
            raise BroadQaSourceInferenceError("derived claim premise 顺序漂移")
        sources = {item.observation.source for item in self.premises}
        if len(sources) != 1:
            raise BroadQaSourceInferenceError(
                "来源内 derivation 不得跨 SourceRef")
        source = next(iter(sources))
        scope = self.derivation.applicability_scope
        if scope.source != source:
            raise BroadQaSourceInferenceError(
                "derivation applicability scope 未绑定来源")
        if not isinstance(self.definition, AtomicPropositionDefinition):
            raise TypeError("derived claim definition 类型非法")
        if (semantic_source(self.definition.proposition) != source
                or semantic_source(self.definition.context) != source):
            raise BroadQaSourceInferenceError("derived claim 未绑定 premise 来源")
        if self.definition.proposition in {
                item.definition.proposition for item in self.premises}:
            raise BroadQaSourceInferenceError(
                "derived claim proposition 不得复用 premise identity")
        if self.definition.bindings != tuple(sorted(
                self.definition.bindings, key=_binding_key)):
            raise BroadQaSourceInferenceError(
                "derived claim bindings 未规范排序")

        premise_slots: dict[tuple[int, tuple[int, ...], int], BroadQaSourceRoleSpan] = {}
        for premise_index, premise in enumerate(self.premises):
            for role_span in premise.role_spans:
                slot = (
                    premise_index,
                    role_span.binding.role.stable_key(),
                    role_span.binding.ordinal,
                )
                premise_slots[slot] = role_span
        result_bindings = {
            (item.role.stable_key(), item.ordinal): item
            for item in self.definition.bindings
        }
        if len(result_bindings) != len(self.definition.bindings):
            raise BroadQaSourceInferenceError(
                "derived claim result Role+ordinal 重复")
        projected_surfaces: dict[tuple[tuple[int, ...], int], str] = {}
        projected_slots = set()
        for projection in self.derivation.role_projections:
            if projection.premise_index >= len(self.premises):
                raise BroadQaSourceInferenceError(
                    "role projection premise index 越界")
            source_slot = (
                projection.premise_index,
                projection.premise_role.stable_key(),
                projection.premise_ordinal,
            )
            source_span = premise_slots.get(source_slot)
            result_slot = (
                projection.result_role.stable_key(), projection.result_ordinal)
            result_binding = result_bindings.get(result_slot)
            if (source_span is None or result_binding is None
                    or result_binding.filler != source_span.binding.filler):
                raise BroadQaSourceInferenceError(
                    "role projection 替换、丢失或伪造了 typed filler")
            projected_slots.add(result_slot)
            projected_surfaces[result_slot] = source_span.surface
        if projected_slots != set(result_bindings):
            raise BroadQaSourceInferenceError(
                "role projections 未精确覆盖结果 bindings")

        if (not isinstance(self.output_parts, tuple) or not self.output_parts
                or any(not isinstance(item, BroadQaSourceOutputPart)
                       for item in self.output_parts)):
            raise BroadQaSourceInferenceError("derived claim output parts 非法")
        output_slots = tuple(
            (item.result_role.stable_key(), item.result_ordinal)
            for item in self.output_parts)
        if len(output_slots) != len(set(output_slots)):
            raise BroadQaSourceInferenceError("derived claim output slot 重复")
        for item, slot in zip(self.output_parts, output_slots, strict=True):
            if projected_surfaces.get(slot) != item.surface:
                raise BroadQaSourceInferenceError(
                    "derived claim output 含来源外字符")
        if self.rendered_text != "".join(item.surface for item in self.output_parts):
            raise BroadQaSourceInferenceError(
                "derived claim rendered_text 不是逐字来源拼接")
        if (self.epistemic_status != SOURCE_INFERENCE_EPISTEMIC_STATUS
                or self.truth_status != SOURCE_INFERENCE_TRUTH_STATUS):
            raise BroadQaSourceInferenceError("derived claim 真值边界漂移")

    def to_dict(self) -> dict[str, object]:
        """导出完整前提、规则、结果命题和逐字输出链。"""
        return {
            "definition": _definition_dict(self.definition),
            "derivation": self.derivation.to_dict(),
            "epistemic_status": self.epistemic_status,
            "output_parts": [item.to_dict() for item in self.output_parts],
            "premises": [item.to_dict() for item in self.premises],
            "rendered_text": self.rendered_text,
            "truth_status": self.truth_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceDerivedClaim":
        """从字段精确的 JSON object 恢复 derived claim。"""
        raw = _exact(value, {
            "definition", "derivation", "epistemic_status", "output_parts",
            "premises", "rendered_text", "truth_status",
        }, label="source derived claim")
        if (not isinstance(raw["premises"], list)
                or not isinstance(raw["output_parts"], list)):
            raise BroadQaSourceInferenceError(
                "derived claim premise/output 必须是数组")
        return cls(
            tuple(BroadQaSourcePremise.from_dict(item)
                  for item in raw["premises"]),
            BroadQaSourceDerivation.from_dict(raw["derivation"]),
            _definition_from_dict(raw["definition"]),
            tuple(BroadQaSourceOutputPart.from_dict(item)
                  for item in raw["output_parts"]),
            raw["rendered_text"], raw["epistemic_status"], raw["truth_status"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceRecord:
    """封装一个默认关闭的来源内归纳合同记录。"""

    claim: BroadQaSourceDerivedClaim
    runtime_state: str = SOURCE_INFERENCE_RUNTIME_STATE
    production_enabled: int = 0

    def __post_init__(self) -> None:
        """阻止合同存在被误记为生产能力已经接通。"""
        if not isinstance(self.claim, BroadQaSourceDerivedClaim):
            raise TypeError("source inference record claim 类型非法")
        if (self.runtime_state != SOURCE_INFERENCE_RUNTIME_STATE
                or type(self.production_enabled) is not int
                or self.production_enabled != 0):
            raise BroadQaSourceInferenceError(
                "source inference contract 不得声明生产启用")

    def to_dict(self) -> dict[str, object]:
        """导出冻结 envelope 和完整 claim。"""
        return {
            "artifact_kind": SOURCE_INFERENCE_RECORD_KIND,
            "claim": self.claim.to_dict(),
            "format_version": 1,
            "production_enabled": self.production_enabled,
            "runtime_state": self.runtime_state,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 JSONL 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回完整规范 record 的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceInferenceRecord":
        """从字段精确的 JSON object 恢复 record。"""
        raw = _exact(value, {
            "artifact_kind", "claim", "format_version", "production_enabled",
            "runtime_state",
        }, label="source inference record")
        if (raw["artifact_kind"] != SOURCE_INFERENCE_RECORD_KIND
                or raw["format_version"] != 1):
            raise BroadQaSourceInferenceError(
                "source inference record envelope 漂移")
        return cls(
            BroadQaSourceDerivedClaim.from_dict(raw["claim"]),
            raw["runtime_state"], raw["production_enabled"],
        )


def parse_broad_qa_source_inference_record(
        payload: bytes,
        ) -> BroadQaSourceInferenceRecord:
    """严格回读单行规范 record，拒绝尾随、缺换行或重编码。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaSourceInferenceError(
            "source inference record 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaSourceInferenceError(
            "source inference record 不是规范 JSON") from error
    record = BroadQaSourceInferenceRecord.from_dict(value)
    if record.canonical_bytes() != payload:
        raise BroadQaSourceInferenceError(
            "source inference record 字节承诺漂移")
    return record


__all__ = [
    "BroadQaSourceDerivation",
    "BroadQaSourceDerivedClaim",
    "BroadQaSourceInferenceError",
    "BroadQaSourceInferenceRecord",
    "BroadQaSourceObservation",
    "BroadQaSourceOutputPart",
    "BroadQaSourcePremise",
    "BroadQaSourceRoleProjection",
    "BroadQaSourceRoleSpan",
    "SOURCE_INFERENCE_DIRECTIONS",
    "SOURCE_INFERENCE_EPISTEMIC_STATUS",
    "SOURCE_INFERENCE_RECORD_KIND",
    "SOURCE_INFERENCE_RUNTIME_STATE",
    "SOURCE_INFERENCE_TRUTH_STATUS",
    "parse_broad_qa_source_inference_record",
]
