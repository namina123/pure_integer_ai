"""D-02D 嵌套作用域的异构 unary operator chain 纯合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.identity import OBJECT_PROPOSITION
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    ALLOWED_PERTURBATIONS,
    EXPECTED_STATES,
    INSTRUCTION_REGISTRY,
    LICENSE_ID,
    OPERATOR_REGISTRY,
    REQUIRED_SAMPLE_ROLES,
    ROLE_REGISTRY,
    STRUCTURE_REGISTRY,
    LogicAnchorSeed,
)
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    AuthoredModalCourseError,
    ModalResolverSeed,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_schema import (
    AuthoredQuantifierCourseError,
    QuantifierDomainSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)


_LAYER_FIELDS = frozenset({
    "anchor",
    "candidate_available",
    "instruction_kind",
    "layer_id",
    "modal_resolver",
    "operator_family",
    "operator_kind",
    "role_kind",
    "structure_kind",
})
_LEAF_FIELDS = frozenset({
    "end",
    "evidence_refute",
    "evidence_support",
    "local_id",
    "object_kind",
    "ordinal",
    "start",
    "surface_fragment",
})
_QUANTIFIER_FIELDS = frozenset({
    "binder_local_id",
    "domain",
    "layer_id",
    "value_role_kind",
    "value_type_kind",
    "variable_local_id",
})
_REQUEST_FIELDS = frozenset({
    "max_branches",
    "max_depth",
    "max_domain_values",
    "max_resolver_calls",
    "max_steps",
})
_SEED_FIELDS = frozenset({
    "consumer_request",
    "context_local_id",
    "expected_payload",
    "expected_state",
    "family",
    "instruction_registry",
    "label_owner",
    "layers",
    "leaf",
    "license_id",
    "logical_order",
    "operator_registry",
    "perturbation_kind",
    "quantifier",
    "role_registry",
    "sample_role",
    "seed_id",
    "split",
    "structure_registry",
    "supersedes_seed_id",
    "surface",
    "template_family",
})


class AuthoredNestedCourseError(RuntimeError):
    """原创 nested seed 的层序、scope、Binder、预算或恢复链非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 nested 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredNestedCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredNestedCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求 nested 身份与预算为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredNestedCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 nested span 与 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredNestedCourseError(f"{where} 必须是非负严格整数")
    return value


def _bit(value: Any, *, where: str) -> int:
    """要求 candidate 和 Evidence 位只能是严格整数 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredNestedCourseError(f"{where} 必须是严格整数 0/1")
    return value


@dataclass(frozen=True)
class NestedLayerSeed:
    """一层 unary operator 的冻结坐标、anchor、可用位和 modal resolver。"""

    layer_id: str
    operator_family: str
    operator_kind: int
    structure_kind: int
    instruction_kind: int
    role_kind: int
    anchor: LogicAnchorSeed
    candidate_available: int
    modal_resolver: ModalResolverSeed | None

    def __post_init__(self) -> None:
        _text(self.layer_id, where="NestedLayerSeed.layer_id")
        _text(
            self.operator_family, where="NestedLayerSeed.operator_family")
        for name, value in (
                ("operator_kind", self.operator_kind),
                ("structure_kind", self.structure_kind),
                ("instruction_kind", self.instruction_kind),
                ("role_kind", self.role_kind)):
            _positive_int(value, where=f"NestedLayerSeed.{name}")
        if not isinstance(self.anchor, LogicAnchorSeed):
            raise AuthoredNestedCourseError("nested layer anchor 类型错误")
        _bit(
            self.candidate_available,
            where="NestedLayerSeed.candidate_available",
        )
        if self.operator_family == "MODAL":
            if not isinstance(self.modal_resolver, ModalResolverSeed):
                raise AuthoredNestedCourseError(
                    "MODAL nested layer 必须有 resolver")
        elif self.modal_resolver is not None:
            raise AuthoredNestedCourseError(
                "非 MODAL nested layer 不得携带 resolver")

    @classmethod
    def from_dict(cls, value: Any) -> "NestedLayerSeed":
        """从严格字段集合恢复一个 nested layer。"""
        if not isinstance(value, dict) or set(value) != _LAYER_FIELDS:
            raise AuthoredNestedCourseError("nested layer 字段集合漂移")
        raw_resolver = value["modal_resolver"]
        try:
            resolver = (
                None if raw_resolver is None
                else ModalResolverSeed.from_dict(raw_resolver))
        except AuthoredModalCourseError as error:
            raise AuthoredNestedCourseError(
                "nested modal resolver 非法") from error
        return cls(
            _text(value["layer_id"], where="layer_id"),
            _text(value["operator_family"], where="operator_family"),
            value["operator_kind"],
            value["structure_kind"],
            value["instruction_kind"],
            value["role_kind"],
            LogicAnchorSeed.from_dict(value["anchor"]),
            value["candidate_available"],
            resolver,
        )


@dataclass(frozen=True)
class NestedLeafSeed:
    """嵌套 chain 最内层原子 Proposition 和四态 Evidence。"""

    surface_fragment: str
    start: int
    end: int
    ordinal: int
    local_id: int
    object_kind: int
    evidence_support: int
    evidence_refute: int

    def __post_init__(self) -> None:
        _text(self.surface_fragment, where="NestedLeafSeed.surface_fragment")
        _nonnegative_int(self.start, where="NestedLeafSeed.start")
        _nonnegative_int(self.end, where="NestedLeafSeed.end")
        _nonnegative_int(self.ordinal, where="NestedLeafSeed.ordinal")
        _positive_int(self.local_id, where="NestedLeafSeed.local_id")
        if self.end <= self.start:
            raise AuthoredNestedCourseError("nested leaf span 必须有正宽度")
        if self.object_kind != OBJECT_PROPOSITION:
            raise AuthoredNestedCourseError(
                "nested leaf 必须是 Proposition")
        _bit(self.evidence_support, where="NestedLeafSeed.evidence_support")
        _bit(self.evidence_refute, where="NestedLeafSeed.evidence_refute")

    @classmethod
    def from_dict(cls, value: Any) -> "NestedLeafSeed":
        """从严格字段集合恢复 nested leaf。"""
        if not isinstance(value, dict) or set(value) != _LEAF_FIELDS:
            raise AuthoredNestedCourseError("nested leaf 字段集合漂移")
        return cls(
            _text(value["surface_fragment"], where="leaf.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
            value["local_id"],
            value["object_kind"],
            value["evidence_support"],
            value["evidence_refute"],
        )


@dataclass(frozen=True)
class NestedQuantifierSeed:
    """chain 中唯一量词层的 Binder、Variable、type 和有限域。"""

    layer_id: str
    binder_local_id: int
    variable_local_id: int
    value_type_kind: int
    value_role_kind: int
    domain: QuantifierDomainSeed

    def __post_init__(self) -> None:
        _text(self.layer_id, where="NestedQuantifierSeed.layer_id")
        for name, value in (
                ("binder_local_id", self.binder_local_id),
                ("variable_local_id", self.variable_local_id),
                ("value_type_kind", self.value_type_kind),
                ("value_role_kind", self.value_role_kind)):
            _positive_int(value, where=f"NestedQuantifierSeed.{name}")
        if not isinstance(self.domain, QuantifierDomainSeed):
            raise AuthoredNestedCourseError("nested quantifier domain 类型错误")
        if any(item.actual_type_kind != self.value_type_kind
               for item in self.domain.values):
            raise AuthoredNestedCourseError(
                "nested quantifier 不得混入错类型 value")

    @classmethod
    def from_dict(cls, value: Any) -> "NestedQuantifierSeed":
        """从严格字段集合恢复 nested quantifier。"""
        if not isinstance(value, dict) or set(value) != _QUANTIFIER_FIELDS:
            raise AuthoredNestedCourseError(
                "nested quantifier 字段集合漂移")
        try:
            domain = QuantifierDomainSeed.from_dict(value["domain"])
        except AuthoredQuantifierCourseError as error:
            raise AuthoredNestedCourseError(
                "nested quantifier domain 非法") from error
        return cls(
            _text(value["layer_id"], where="quantifier.layer_id"),
            value["binder_local_id"],
            value["variable_local_id"],
            value["value_type_kind"],
            value["value_role_kind"],
            domain,
        )


@dataclass(frozen=True)
class NestedConsumerRequestSeed:
    """嵌套执行的 depth、branch、domain、resolver 和 step 预算。"""

    max_steps: int
    max_depth: int
    max_branches: int
    max_domain_values: int
    max_resolver_calls: int

    def __post_init__(self) -> None:
        for name, value in (
                ("max_steps", self.max_steps),
                ("max_depth", self.max_depth),
                ("max_branches", self.max_branches),
                ("max_domain_values", self.max_domain_values),
                ("max_resolver_calls", self.max_resolver_calls)):
            _positive_int(value, where=f"NestedConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "NestedConsumerRequestSeed":
        """从严格字段集合恢复 nested consumer。"""
        if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
            raise AuthoredNestedCourseError(
                "nested consumer 字段集合漂移")
        return cls(
            value["max_steps"],
            value["max_depth"],
            value["max_branches"],
            value["max_domain_values"],
            value["max_resolver_calls"],
        )


@dataclass(frozen=True)
class AuthoredNestedSeed:
    """可恢复为异构 unary bound tree 的嵌套作用域课程记录。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    operator_registry: str
    structure_registry: str
    instruction_registry: str
    role_registry: str
    surface: str
    layers: tuple[NestedLayerSeed, ...]
    leaf: NestedLeafSeed
    context_local_id: int
    quantifier: NestedQuantifierSeed | None
    consumer_request: NestedConsumerRequestSeed
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
                ("surface", self.surface),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredNestedSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredNestedSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredNestedCourseError(
                "label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredNestedCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredNestedCourseError(
                "sample_role 不属于 nested 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredNestedCourseError(
                "supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredNestedCourseError(
                "非 supersede seed 不得声明替代目标")
        if self.operator_registry != OPERATOR_REGISTRY:
            raise AuthoredNestedCourseError("nested operator registry 漂移")
        if self.structure_registry != STRUCTURE_REGISTRY:
            raise AuthoredNestedCourseError("nested structure registry 漂移")
        if self.instruction_registry != INSTRUCTION_REGISTRY:
            raise AuthoredNestedCourseError("nested instruction registry 漂移")
        if self.role_registry != ROLE_REGISTRY:
            raise AuthoredNestedCourseError("nested Role registry 漂移")
        if (not isinstance(self.layers, tuple) or len(self.layers) < 2
                or any(not isinstance(item, NestedLayerSeed)
                       for item in self.layers)):
            raise AuthoredNestedCourseError(
                "nested layers 必须至少有两个 operator")
        layer_ids = [item.layer_id for item in self.layers]
        if len(set(layer_ids)) != len(layer_ids):
            raise AuthoredNestedCourseError("nested layer_id 重复")
        if not isinstance(self.leaf, NestedLeafSeed):
            raise AuthoredNestedCourseError("nested leaf 类型错误")
        _positive_int(
            self.context_local_id,
            where="AuthoredNestedSeed.context_local_id",
        )
        if self.consumer_request.max_depth < len(self.layers):
            raise AuthoredNestedCourseError(
                "nested depth 超过 consumer 预算")
        modal_count = sum(
            item.operator_family == "MODAL" for item in self.layers)
        if self.consumer_request.max_resolver_calls < modal_count:
            raise AuthoredNestedCourseError(
                "nested modal 层超过 resolver 预算")
        quantifier_layers = {
            item.layer_id for item in self.layers
            if item.operator_family in {"EXISTS", "FORALL"}}
        if self.quantifier is None:
            if quantifier_layers:
                raise AuthoredNestedCourseError(
                    "nested quantifier layer 缺少 Binder/domain")
        else:
            if quantifier_layers != {self.quantifier.layer_id}:
                raise AuthoredNestedCourseError(
                    "nested quantifier 必须恰好对应唯一量词层")
            if (len(self.quantifier.domain.values)
                    > self.consumer_request.max_domain_values):
                raise AuthoredNestedCourseError(
                    "nested domain 超过 consumer 预算")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredNestedCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredNestedCourseError("nested perturbation 未注册")
        spans = []
        for layer in self.layers:
            anchor = layer.anchor
            if anchor.end > len(self.surface) or self.surface[
                    anchor.start:anchor.end] != anchor.surface_fragment:
                raise AuthoredNestedCourseError(
                    "nested layer anchor span 与 surface 不一致")
            spans.append((anchor.start, anchor.end, anchor.ordinal))
        if spans != sorted(spans):
            raise AuthoredNestedCourseError(
                "nested layer 必须按 outer-to-inner 来源顺序排列")
        if self.leaf.end > len(self.surface) or self.surface[
                self.leaf.start:self.leaf.end] != self.leaf.surface_fragment:
            raise AuthoredNestedCourseError(
                "nested leaf span 与 surface 不一致")
        if spans[-1][1] > self.leaf.start:
            raise AuthoredNestedCourseError(
                "nested operator anchor 不得与 leaf 重叠")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredNestedSeed":
        """从严格字段集合恢复 nested seed。"""
        if not isinstance(value, dict) or set(value) != _SEED_FIELDS:
            raise AuthoredNestedCourseError("nested seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredNestedCourseError("nested seed 必须是 CC0-1.0")
        layers = value["layers"]
        if not isinstance(layers, list):
            raise AuthoredNestedCourseError("nested layers 必须是列表")
        raw_quantifier = value["quantifier"]
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["operator_registry"], where="operator_registry"),
            _text(value["structure_registry"], where="structure_registry"),
            _text(value["instruction_registry"], where="instruction_registry"),
            _text(value["role_registry"], where="role_registry"),
            _text(value["surface"], where="surface"),
            tuple(NestedLayerSeed.from_dict(item) for item in layers),
            NestedLeafSeed.from_dict(value["leaf"]),
            value["context_local_id"],
            (None if raw_quantifier is None
             else NestedQuantifierSeed.from_dict(raw_quantifier)),
            NestedConsumerRequestSeed.from_dict(value["consumer_request"]),
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
    "AuthoredNestedCourseError",
    "AuthoredNestedSeed",
    "NestedConsumerRequestSeed",
    "NestedLayerSeed",
    "NestedLeafSeed",
    "NestedQuantifierSeed",
]
