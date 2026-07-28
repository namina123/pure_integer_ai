"""D-02D 原创 typed 复合逻辑 seed 的共享纯合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.identity import OBJECT_PROPOSITION
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
OPERATOR_REGISTRY = "AUTHORED_LOGIC_OPERATOR_V1"
STRUCTURE_REGISTRY = "AUTHORED_LOGIC_STRUCTURE_V1"
INSTRUCTION_REGISTRY = "AUTHORED_LOGIC_INSTRUCTION_V1"
ROLE_REGISTRY = "AUTHORED_LOGIC_ROLE_V1"

REQUEST_LOGIC_EXECUTION = 1
OPERATOR_NOT = 1
OPERATOR_AND = 2
OPERATOR_OR = 3
OPERATOR_CONDITION = 4
OPERATOR_EXISTS = 5
OPERATOR_FORALL = 6
OPERATOR_MODAL = 7
STRUCTURE_NOT = 1
STRUCTURE_AND = 2
STRUCTURE_OR = 3
STRUCTURE_CONDITION = 4
STRUCTURE_EXISTS = 5
STRUCTURE_FORALL = 6
STRUCTURE_MODAL = 7
INSTRUCTION_NOT = 1
INSTRUCTION_AND = 2
INSTRUCTION_OR = 3
INSTRUCTION_CONDITION = 4
INSTRUCTION_EXISTS = 5
INSTRUCTION_FORALL = 6
INSTRUCTION_MODAL = 7
ROLE_NOT_OPERAND = 1
ROLE_AND_OPERAND = 2
ROLE_OR_OPERAND = 3
ROLE_CONDITION_ANTECEDENT = 4
ROLE_CONDITION_CONSEQUENT = 5
ROLE_EXISTS_BODY = 6
ROLE_EXISTS_VALUE = 7
ROLE_FORALL_BODY = 8
ROLE_FORALL_VALUE = 9
ROLE_MODAL_CHILD = 10

ALLOWED_PERTURBATIONS = frozenset({
    "NONE",
    "CONTENT_REPLACEMENT",
    "DOUBLE_NEGATION",
    "TARGET_REPLACEMENT",
    "SCOPE_TARGET_SHIFT",
    "PSEUDO_OPERATOR",
    "CLOSED_WORLD_CONFUSION",
    "REFUTE_EVIDENCE_CONFUSION",
    "ANTONYM_CONFUSION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
    "OPERAND_ORDER_SWAP",
    "BRANCH_REPLACEMENT",
    "OPERATOR_CONFUSION",
    "ANTECEDENT_CONSEQUENT_SWAP",
    "CAUSAL_CONFUSION",
    "TEMPORAL_CONFUSION",
    "QUANTIFIER_SWAP",
    "DOMAIN_CLOSURE_CONFUSION",
    "DOMAIN_TYPE_MISMATCH",
    "EMPTY_DOMAIN_CONFUSION",
    "MODAL_SCOPE_SHIFT",
    "RESOLVER_MISSING",
    "RESOLVER_DENIED",
    "BUDGET_UNDECIDED",
    "MISSING_INNER_OPERATOR",
    "DEPTH_REPLACEMENT",
})
REQUIRED_SAMPLE_ROLES = frozenset({
    "support", "refute", "conflict", "supersede"})

_ANCHOR_FIELDS = frozenset({
    "end", "ordinal", "start", "surface_fragment"})
_OPERAND_FIELDS = frozenset({
    "end",
    "evidence_refute",
    "evidence_support",
    "local_id",
    "object_kind",
    "operand_id",
    "ordinal",
    "start",
    "surface_fragment",
})
_BINDING_FIELDS = frozenset({
    "operand_id", "ordinal", "role_kind", "role_registry"})
_REQUEST_FIELDS = frozenset({
    "max_branches", "max_depth", "max_steps", "request_kind"})
_SEED_FIELDS = frozenset({
    "anchor",
    "bindings",
    "consumer_request",
    "context_local_id",
    "expected_payload",
    "expected_state",
    "family",
    "instruction_kind",
    "instruction_registry",
    "label_owner",
    "license_id",
    "logical_order",
    "nesting_depth",
    "operands",
    "operator_family",
    "operator_kind",
    "operator_registry",
    "perturbation_kind",
    "sample_role",
    "seed_id",
    "split",
    "structure_kind",
    "structure_registry",
    "supersedes_seed_id",
    "surface",
    "template_family",
})


class AuthoredLogicCourseError(RuntimeError):
    """原创 logic seed 的 typed 结构、owner、split 或修正链非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 logic seed 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredLogicCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredLogicCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求 operator、身份和预算坐标为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredLogicCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span 与 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredLogicCourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求四态 evidence bit 只能是严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredLogicCourseError(f"{where} 必须是严格整数 0/1")
    return value


@dataclass(frozen=True)
class LogicAnchorSeed:
    """逻辑候选的来源 surface anchor。"""

    surface_fragment: str
    start: int
    end: int
    ordinal: int

    def __post_init__(self) -> None:
        _text(self.surface_fragment, where="LogicAnchorSeed.surface_fragment")
        _nonnegative_int(self.start, where="LogicAnchorSeed.start")
        _nonnegative_int(self.end, where="LogicAnchorSeed.end")
        _nonnegative_int(self.ordinal, where="LogicAnchorSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredLogicCourseError("logic anchor span 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "LogicAnchorSeed":
        """从严格字段集合恢复逻辑 anchor。"""
        if not isinstance(value, dict) or set(value) != _ANCHOR_FIELDS:
            raise AuthoredLogicCourseError("logic anchor 字段集合漂移")
        return cls(
            _text(value["surface_fragment"], where="anchor.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
        )


@dataclass(frozen=True)
class LogicOperandSeed:
    """一个来源化原子 Proposition operand 及其输入 Evidence 四态位。"""

    operand_id: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    object_kind: int
    local_id: int
    evidence_support: int
    evidence_refute: int

    def __post_init__(self) -> None:
        _text(self.operand_id, where="LogicOperandSeed.operand_id")
        _text(self.surface_fragment, where="LogicOperandSeed.surface_fragment")
        _nonnegative_int(self.start, where="LogicOperandSeed.start")
        _nonnegative_int(self.end, where="LogicOperandSeed.end")
        _nonnegative_int(self.ordinal, where="LogicOperandSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredLogicCourseError("logic operand span 必须有正宽度")
        if self.object_kind != OBJECT_PROPOSITION:
            raise AuthoredLogicCourseError(
                "logic operand 必须是一等 Proposition")
        _positive_int(self.local_id, where="LogicOperandSeed.local_id")
        _bit(self.evidence_support, where="LogicOperandSeed.evidence_support")
        _bit(self.evidence_refute, where="LogicOperandSeed.evidence_refute")

    @classmethod
    def from_dict(cls, value: Any) -> "LogicOperandSeed":
        """从严格字段集合恢复逻辑 operand。"""
        if not isinstance(value, dict) or set(value) != _OPERAND_FIELDS:
            raise AuthoredLogicCourseError("logic operand 字段集合漂移")
        return cls(
            _text(value["operand_id"], where="operand_id"),
            _text(value["surface_fragment"], where="operand.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
            value["object_kind"],
            value["local_id"],
            value["evidence_support"],
            value["evidence_refute"],
        )


@dataclass(frozen=True)
class LogicBindingSeed:
    """一个 operator Role/ordinal 到 operand 的显式绑定。"""

    role_registry: str
    role_kind: int
    operand_id: str
    ordinal: int

    def __post_init__(self) -> None:
        if self.role_registry != ROLE_REGISTRY:
            raise AuthoredLogicCourseError("logic Role registry 非冻结坐标")
        _positive_int(self.role_kind, where="LogicBindingSeed.role_kind")
        _text(self.operand_id, where="LogicBindingSeed.operand_id")
        _nonnegative_int(self.ordinal, where="LogicBindingSeed.ordinal")

    @classmethod
    def from_dict(cls, value: Any) -> "LogicBindingSeed":
        """从严格字段集合恢复逻辑 Role binding。"""
        if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
            raise AuthoredLogicCourseError("logic binding 字段集合漂移")
        return cls(
            _text(value["role_registry"], where="role_registry"),
            value["role_kind"],
            _text(value["operand_id"], where="binding.operand_id"),
            value["ordinal"],
        )


@dataclass(frozen=True)
class LogicConsumerRequestSeed:
    """一次逻辑执行的显式递归、分支和步骤预算。"""

    request_kind: int
    max_steps: int
    max_depth: int
    max_branches: int

    def __post_init__(self) -> None:
        if self.request_kind != REQUEST_LOGIC_EXECUTION:
            raise AuthoredLogicCourseError("logic consumer request kind 未注册")
        for name, value in (
                ("max_steps", self.max_steps),
                ("max_depth", self.max_depth),
                ("max_branches", self.max_branches)):
            _positive_int(value, where=f"LogicConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "LogicConsumerRequestSeed":
        """从严格字段集合恢复 logic consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredLogicCourseError("logic consumer 字段集合漂移")
        return cls(
            value["request_kind"],
            value["max_steps"],
            value["max_depth"],
            value["max_branches"],
        )


@dataclass(frozen=True)
class AuthoredLogicSeed:
    """一条可编译为 LogicOperatorDefinition 和 BoundProposition 的候选。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    operator_family: str
    operator_registry: str
    operator_kind: int
    structure_registry: str
    structure_kind: int
    instruction_registry: str
    instruction_kind: int
    surface: str
    anchor: LogicAnchorSeed
    operands: tuple[LogicOperandSeed, ...]
    bindings: tuple[LogicBindingSeed, ...]
    context_local_id: int
    nesting_depth: int
    consumer_request: LogicConsumerRequestSeed
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("operator_family", self.operator_family),
                ("surface", self.surface),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredLogicSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredLogicSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredLogicCourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredLogicCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredLogicCourseError("sample_role 不属于 logic 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredLogicCourseError("supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredLogicCourseError("非 supersede seed 不得声明替代目标")
        if self.operator_registry != OPERATOR_REGISTRY:
            raise AuthoredLogicCourseError("logic operator registry 非冻结坐标")
        if self.structure_registry != STRUCTURE_REGISTRY:
            raise AuthoredLogicCourseError("logic structure registry 非冻结坐标")
        if self.instruction_registry != INSTRUCTION_REGISTRY:
            raise AuthoredLogicCourseError("logic instruction registry 非冻结坐标")
        _positive_int(self.operator_kind, where="AuthoredLogicSeed.operator_kind")
        _positive_int(self.structure_kind, where="AuthoredLogicSeed.structure_kind")
        _positive_int(
            self.instruction_kind,
            where="AuthoredLogicSeed.instruction_kind",
        )
        _positive_int(
            self.context_local_id,
            where="AuthoredLogicSeed.context_local_id",
        )
        _positive_int(self.nesting_depth, where="AuthoredLogicSeed.nesting_depth")
        _positive_int(self.logical_order, where="AuthoredLogicSeed.logical_order")
        if self.consumer_request.max_depth < self.nesting_depth:
            raise AuthoredLogicCourseError("logic consumer max_depth 小于嵌套深度")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredLogicCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredLogicCourseError("logic perturbation 未注册")
        if not self.operands or not self.bindings:
            raise AuthoredLogicCourseError("logic operands/bindings 不能为空")
        if self.anchor.end > len(self.surface) or self.surface[
                self.anchor.start:self.anchor.end] != self.anchor.surface_fragment:
            raise AuthoredLogicCourseError("logic anchor span 与 surface 不一致")
        operand_ids = [item.operand_id for item in self.operands]
        if len(set(operand_ids)) != len(operand_ids):
            raise AuthoredLogicCourseError("logic operand_id 重复")
        operand_spans = [
            (item.start, item.end, item.ordinal) for item in self.operands]
        if operand_spans != sorted(operand_spans):
            raise AuthoredLogicCourseError("logic operand 必须按来源 span 排序")
        previous_end = -1
        for operand in self.operands:
            if operand.end > len(self.surface) or self.surface[
                    operand.start:operand.end] != operand.surface_fragment:
                raise AuthoredLogicCourseError(
                    "logic operand span 与 surface 不一致")
            if operand.start < previous_end:
                raise AuthoredLogicCourseError("logic operand span 不得重叠")
            previous_end = operand.end
        slots = [(item.role_kind, item.ordinal) for item in self.bindings]
        if len(set(slots)) != len(slots):
            raise AuthoredLogicCourseError("logic Role/ordinal slot 重复")
        bound = [item.operand_id for item in self.bindings]
        if any(item not in set(operand_ids) for item in bound):
            raise AuthoredLogicCourseError("logic binding 引用未知 operand")
        if len(bound) != len(operand_ids) or set(bound) != set(operand_ids):
            raise AuthoredLogicCourseError(
                "logic bindings 必须恰好覆盖全部 operand")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredLogicSeed":
        """从严格字段集合恢复一条共享 logic seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredLogicCourseError("logic seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredLogicCourseError("logic seed 必须是 CC0-1.0")
        operands = value["operands"]
        bindings = value["bindings"]
        if not isinstance(operands, list) or not isinstance(bindings, list):
            raise AuthoredLogicCourseError("operands/bindings 必须是列表")
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["operator_family"], where="operator_family"),
            _text(value["operator_registry"], where="operator_registry"),
            value["operator_kind"],
            _text(value["structure_registry"], where="structure_registry"),
            value["structure_kind"],
            _text(
                value["instruction_registry"],
                where="instruction_registry",
            ),
            value["instruction_kind"],
            _text(value["surface"], where="surface"),
            LogicAnchorSeed.from_dict(value["anchor"]),
            tuple(LogicOperandSeed.from_dict(item) for item in operands),
            tuple(LogicBindingSeed.from_dict(item) for item in bindings),
            value["context_local_id"],
            value["nesting_depth"],
            LogicConsumerRequestSeed.from_dict(value["consumer_request"]),
            _text(value["expected_state"], where="expected_state"),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            _text(value["perturbation_kind"], where="perturbation_kind"),
            _text(
                value["supersedes_seed_id"],
                where="supersedes_seed_id",
                allow_empty=True,
            ),
            value["logical_order"],
        )


__all__ = [
    "ALLOWED_PERTURBATIONS",
    "AuthoredLogicCourseError",
    "AuthoredLogicSeed",
    "INSTRUCTION_NOT",
    "INSTRUCTION_AND",
    "INSTRUCTION_OR",
    "INSTRUCTION_CONDITION",
    "INSTRUCTION_EXISTS",
    "INSTRUCTION_FORALL",
    "INSTRUCTION_MODAL",
    "INSTRUCTION_REGISTRY",
    "LICENSE_ID",
    "LogicAnchorSeed",
    "LogicBindingSeed",
    "LogicConsumerRequestSeed",
    "LogicOperandSeed",
    "OPERATOR_NOT",
    "OPERATOR_AND",
    "OPERATOR_OR",
    "OPERATOR_CONDITION",
    "OPERATOR_EXISTS",
    "OPERATOR_FORALL",
    "OPERATOR_MODAL",
    "OPERATOR_REGISTRY",
    "REQUEST_LOGIC_EXECUTION",
    "REQUIRED_SAMPLE_ROLES",
    "ROLE_NOT_OPERAND",
    "ROLE_AND_OPERAND",
    "ROLE_OR_OPERAND",
    "ROLE_CONDITION_ANTECEDENT",
    "ROLE_CONDITION_CONSEQUENT",
    "ROLE_EXISTS_BODY",
    "ROLE_EXISTS_VALUE",
    "ROLE_FORALL_BODY",
    "ROLE_FORALL_VALUE",
    "ROLE_MODAL_CHILD",
    "ROLE_REGISTRY",
    "SOURCE_KEY",
    "STRUCTURE_NOT",
    "STRUCTURE_AND",
    "STRUCTURE_OR",
    "STRUCTURE_CONDITION",
    "STRUCTURE_EXISTS",
    "STRUCTURE_FORALL",
    "STRUCTURE_MODAL",
    "STRUCTURE_REGISTRY",
]
