"""D-02D EXISTS/FORALL 的 Binder、有限域和四态 seed 合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    ALLOWED_PERTURBATIONS,
    INSTRUCTION_REGISTRY,
    LICENSE_ID,
    OPERATOR_REGISTRY,
    REQUIRED_SAMPLE_ROLES,
    ROLE_REGISTRY,
    SOURCE_KEY,
    STRUCTURE_REGISTRY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)


REQUEST_QUANTIFIER_EXECUTION = 2

_ANCHOR_FIELDS = frozenset({
    "end", "ordinal", "start", "surface_fragment"})
_BODY_FIELDS = frozenset({
    "end", "local_id", "ordinal", "start", "surface_fragment"})
_VALUE_FIELDS = frozenset({
    "actual_type_kind",
    "evidence_refute",
    "evidence_support",
    "local_id",
    "value_id",
})
_DOMAIN_FIELDS = frozenset({
    "closed",
    "closure_evidence_local_ids",
    "domain_local_id",
    "values",
})
_REQUEST_FIELDS = frozenset({
    "max_branches",
    "max_depth",
    "max_domain_values",
    "max_steps",
    "request_kind",
})
_SEED_FIELDS = frozenset({
    "anchor",
    "binder_local_id",
    "body",
    "body_role_kind",
    "consumer_request",
    "context_local_id",
    "domain",
    "expected_payload",
    "expected_state",
    "family",
    "instruction_kind",
    "instruction_registry",
    "label_owner",
    "license_id",
    "logical_order",
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
    "value_role_kind",
    "value_type_kind",
    "variable_local_id",
})


class AuthoredQuantifierCourseError(RuntimeError):
    """原创 quantifier seed 的 Binder、域、owner 或修正链非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 quantifier 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredQuantifierCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredQuantifierCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求身份、坐标和预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredQuantifierCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span 与 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredQuantifierCourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 domain closed 和 Evidence 位只能是严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredQuantifierCourseError(f"{where} 必须是严格整数 0/1")
    return value


@dataclass(frozen=True)
class QuantifierAnchorSeed:
    """量词 operator 的来源 anchor。"""

    surface_fragment: str
    start: int
    end: int
    ordinal: int

    def __post_init__(self) -> None:
        _text(self.surface_fragment, where="QuantifierAnchorSeed.surface_fragment")
        _nonnegative_int(self.start, where="QuantifierAnchorSeed.start")
        _nonnegative_int(self.end, where="QuantifierAnchorSeed.end")
        _nonnegative_int(self.ordinal, where="QuantifierAnchorSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredQuantifierCourseError(
                "quantifier anchor span 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "QuantifierAnchorSeed":
        """从严格字段集合恢复 quantifier anchor。"""
        if not isinstance(value, dict) or set(value) != _ANCHOR_FIELDS:
            raise AuthoredQuantifierCourseError(
                "quantifier anchor 字段集合漂移")
        return cls(
            _text(value["surface_fragment"], where="anchor.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
        )


@dataclass(frozen=True)
class QuantifierBodySeed:
    """量词 body 原子 Proposition 的来源 span。"""

    surface_fragment: str
    start: int
    end: int
    ordinal: int
    local_id: int

    def __post_init__(self) -> None:
        _text(self.surface_fragment, where="QuantifierBodySeed.surface_fragment")
        _nonnegative_int(self.start, where="QuantifierBodySeed.start")
        _nonnegative_int(self.end, where="QuantifierBodySeed.end")
        _nonnegative_int(self.ordinal, where="QuantifierBodySeed.ordinal")
        _positive_int(self.local_id, where="QuantifierBodySeed.local_id")
        if self.end <= self.start:
            raise AuthoredQuantifierCourseError(
                "quantifier body span 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "QuantifierBodySeed":
        """从严格字段集合恢复 quantifier body。"""
        if not isinstance(value, dict) or set(value) != _BODY_FIELDS:
            raise AuthoredQuantifierCourseError("quantifier body 字段集合漂移")
        return cls(
            _text(value["surface_fragment"], where="body.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
            value["local_id"],
        )


@dataclass(frozen=True)
class QuantifierValueSeed:
    """有限域中的 typed value 及替换后 body Evidence 四态。"""

    value_id: str
    local_id: int
    actual_type_kind: int
    evidence_support: int
    evidence_refute: int

    def __post_init__(self) -> None:
        _text(self.value_id, where="QuantifierValueSeed.value_id")
        _positive_int(self.local_id, where="QuantifierValueSeed.local_id")
        _positive_int(
            self.actual_type_kind,
            where="QuantifierValueSeed.actual_type_kind",
        )
        _bit(
            self.evidence_support,
            where="QuantifierValueSeed.evidence_support",
        )
        _bit(
            self.evidence_refute,
            where="QuantifierValueSeed.evidence_refute",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "QuantifierValueSeed":
        """从严格字段集合恢复一个 domain value。"""
        if not isinstance(value, dict) or set(value) != _VALUE_FIELDS:
            raise AuthoredQuantifierCourseError(
                "quantifier value 字段集合漂移")
        return cls(
            _text(value["value_id"], where="value_id"),
            value["local_id"],
            value["actual_type_kind"],
            value["evidence_support"],
            value["evidence_refute"],
        )


@dataclass(frozen=True)
class QuantifierDomainSeed:
    """显式 SetExpr、closed 声明、closure Evidence 和 typed values。"""

    domain_local_id: int
    closed: int
    closure_evidence_local_ids: tuple[int, ...]
    values: tuple[QuantifierValueSeed, ...]

    def __post_init__(self) -> None:
        _positive_int(
            self.domain_local_id,
            where="QuantifierDomainSeed.domain_local_id",
        )
        _bit(self.closed, where="QuantifierDomainSeed.closed")
        if (not isinstance(self.closure_evidence_local_ids, tuple)
                or any(type(item) is not int or item <= 0
                       for item in self.closure_evidence_local_ids)
                or len(set(self.closure_evidence_local_ids))
                != len(self.closure_evidence_local_ids)):
            raise AuthoredQuantifierCourseError(
                "closure evidence local id 非法或重复")
        if self.closed and not self.closure_evidence_local_ids:
            raise AuthoredQuantifierCourseError(
                "closed quantifier domain 必须有 closure evidence")
        if not self.closed and self.closure_evidence_local_ids:
            raise AuthoredQuantifierCourseError(
                "open quantifier domain 不得伪造 closure evidence")
        if (not isinstance(self.values, tuple)
                or any(not isinstance(item, QuantifierValueSeed)
                       for item in self.values)):
            raise AuthoredQuantifierCourseError(
                "quantifier domain values 类型非法")
        ids = [item.value_id for item in self.values]
        locals_ = [item.local_id for item in self.values]
        if len(set(ids)) != len(ids) or len(set(locals_)) != len(locals_):
            raise AuthoredQuantifierCourseError(
                "quantifier domain value 重复")

    @classmethod
    def from_dict(cls, value: Any) -> "QuantifierDomainSeed":
        """从严格字段集合恢复量词有限域。"""
        if not isinstance(value, dict) or set(value) != _DOMAIN_FIELDS:
            raise AuthoredQuantifierCourseError(
                "quantifier domain 字段集合漂移")
        closure = value["closure_evidence_local_ids"]
        values = value["values"]
        if not isinstance(closure, list) or not isinstance(values, list):
            raise AuthoredQuantifierCourseError(
                "quantifier domain list 字段非法")
        return cls(
            value["domain_local_id"],
            value["closed"],
            tuple(closure),
            tuple(QuantifierValueSeed.from_dict(item) for item in values),
        )


@dataclass(frozen=True)
class QuantifierConsumerRequestSeed:
    """量词执行的 domain、branch、depth 和 step 预算。"""

    request_kind: int
    max_domain_values: int
    max_branches: int
    max_depth: int
    max_steps: int

    def __post_init__(self) -> None:
        if self.request_kind != REQUEST_QUANTIFIER_EXECUTION:
            raise AuthoredQuantifierCourseError(
                "quantifier consumer request kind 未注册")
        for name, value in (
                ("max_domain_values", self.max_domain_values),
                ("max_branches", self.max_branches),
                ("max_depth", self.max_depth),
                ("max_steps", self.max_steps)):
            _positive_int(value, where=f"QuantifierConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "QuantifierConsumerRequestSeed":
        """从严格字段集合恢复 quantifier consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredQuantifierCourseError(
                "quantifier consumer 字段集合漂移")
        return cls(
            value["request_kind"],
            value["max_domain_values"],
            value["max_branches"],
            value["max_depth"],
            value["max_steps"],
        )


@dataclass(frozen=True)
class AuthoredQuantifierSeed:
    """一条可编译为 QuantifierDefinition 和有限域的课程候选。"""

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
    anchor: QuantifierAnchorSeed
    body: QuantifierBodySeed
    context_local_id: int
    binder_local_id: int
    variable_local_id: int
    value_type_kind: int
    body_role_kind: int
    value_role_kind: int
    domain: QuantifierDomainSeed
    consumer_request: QuantifierConsumerRequestSeed
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
            _text(value, where=f"AuthoredQuantifierSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredQuantifierSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredQuantifierCourseError(
                "label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredQuantifierCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredQuantifierCourseError(
                "sample_role 不属于 quantifier 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredQuantifierCourseError(
                "supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredQuantifierCourseError(
                "非 supersede seed 不得声明替代目标")
        if self.operator_registry != OPERATOR_REGISTRY:
            raise AuthoredQuantifierCourseError(
                "quantifier operator registry 非冻结坐标")
        if self.structure_registry != STRUCTURE_REGISTRY:
            raise AuthoredQuantifierCourseError(
                "quantifier structure registry 非冻结坐标")
        if self.instruction_registry != INSTRUCTION_REGISTRY:
            raise AuthoredQuantifierCourseError(
                "quantifier instruction registry 非冻结坐标")
        for name, value in (
                ("operator_kind", self.operator_kind),
                ("structure_kind", self.structure_kind),
                ("instruction_kind", self.instruction_kind),
                ("context_local_id", self.context_local_id),
                ("binder_local_id", self.binder_local_id),
                ("variable_local_id", self.variable_local_id),
                ("value_type_kind", self.value_type_kind),
                ("body_role_kind", self.body_role_kind),
                ("value_role_kind", self.value_role_kind),
                ("logical_order", self.logical_order)):
            _positive_int(value, where=f"AuthoredQuantifierSeed.{name}")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredQuantifierCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredQuantifierCourseError(
                "quantifier perturbation 未注册")
        if self.anchor.end > len(self.surface) or self.surface[
                self.anchor.start:self.anchor.end] != self.anchor.surface_fragment:
            raise AuthoredQuantifierCourseError(
                "quantifier anchor span 与 surface 不一致")
        if self.body.end > len(self.surface) or self.surface[
                self.body.start:self.body.end] != self.body.surface_fragment:
            raise AuthoredQuantifierCourseError(
                "quantifier body span 与 surface 不一致")
        if self.consumer_request.max_domain_values < len(self.domain.values):
            raise AuthoredQuantifierCourseError(
                "quantifier domain values 超过 consumer 预算")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredQuantifierSeed":
        """从严格字段集合恢复 quantifier seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredQuantifierCourseError(
                "quantifier seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredQuantifierCourseError(
                "quantifier seed 必须是 CC0-1.0")
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
            QuantifierAnchorSeed.from_dict(value["anchor"]),
            QuantifierBodySeed.from_dict(value["body"]),
            value["context_local_id"],
            value["binder_local_id"],
            value["variable_local_id"],
            value["value_type_kind"],
            value["body_role_kind"],
            value["value_role_kind"],
            QuantifierDomainSeed.from_dict(value["domain"]),
            QuantifierConsumerRequestSeed.from_dict(value["consumer_request"]),
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
    "AuthoredQuantifierCourseError",
    "AuthoredQuantifierSeed",
    "LICENSE_ID",
    "QuantifierAnchorSeed",
    "QuantifierBodySeed",
    "QuantifierConsumerRequestSeed",
    "QuantifierDomainSeed",
    "QuantifierValueSeed",
    "REQUEST_QUANTIFIER_EXECUTION",
    "SOURCE_KEY",
]
