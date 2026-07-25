"""S-03 旧 STRUCT_BIND 槽位边到 typed Variable 对应候选的只读适配器。

旧边只说明两个 ConceptRef 槽位存在来源化对应，不是变量赋值。调用方必须显式注入
slot ref 到完整 Variable identity 的映射；适配结果保留每条旧边，不按位置或首行猜测。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.semantic_object import describe_variable
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingFailure,
    BindingFailureProtocol,
    TypeCompatibilityResolver,
    TypeCompatibilityResult,
)
from pure_integer_ai.cognition.shared.types import ConceptRef
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.edge_types import EDGE_STRUCT_BIND


def _require_slot_ref(slot_ref: ConceptRef, *, label: str) -> ConceptRef:
    """核验 legacy ConceptRef 是两个严格整数，不解释空间或槽位语义。"""
    if not isinstance(slot_ref, tuple) or len(slot_ref) != 2:
        raise TypeError(f"{label} 必须是二元 ConceptRef")
    assert_int(*slot_ref, _where=label)
    if any(type(part) is not int for part in slot_ref):
        raise ValueError(f"{label} 必须使用严格整数")
    return slot_ref


def _optional_int(value: object, *, label: str) -> int | None:
    """核验旧边可空整数元数据，禁止 bool 或浮点混入 trace。"""
    if value is None:
        return None
    assert_int(value, _where=label)
    if type(value) is not int:
        raise ValueError(f"{label} 必须是严格整数或 None")
    return value


def _required_int(value: object, *, label: str) -> int:
    """核验旧边必填整数元数据，避免容错转换改变原记录。"""
    assert_int(value, _where=label)
    if type(value) is not int:
        raise ValueError(f"{label} 必须是严格整数")
    return value


def _optional_key(value: int | None) -> tuple[int, ...]:
    """把可空整数编码为无歧义确定性排序分段。"""
    return (0,) if value is None else (1, value)


@dataclass(frozen=True)
class TypedStructBindEndpoint:
    """调用方显式声明的 legacy 槽位与 typed Variable 全身份映射。"""

    slot_ref: ConceptRef
    variable: ObjectIdentity

    def __post_init__(self) -> None:
        _require_slot_ref(self.slot_ref, label="typed STRUCT_BIND slot_ref")
        describe_variable(self.variable)

    def stable_key(self) -> tuple[int, ...]:
        """返回槽位和 Variable 全键，用于候选确定性排序。"""
        variable_key = self.variable.stable_key()
        return (*self.slot_ref, len(variable_key), *variable_key)


@dataclass(frozen=True)
class LegacyStructBindMetadata:
    """一条旧 STRUCT_BIND 边除端点外的完整整数 trace，不赋予真值语义。"""

    strength: int
    base_strength: int
    belief_p: int
    belief_q: int
    sn: int
    tn: int
    tier: int
    source: int
    epistemic_origin: int | None
    subtype: int | None
    order_index: int | None
    role: int | None
    memory_time_attach: int | None
    content_version: int

    def __post_init__(self) -> None:
        """核验完整旧边 trace 的必填和可空列都保持严格整数。"""
        for label, value in (
                ("strength", self.strength),
                ("base_strength", self.base_strength),
                ("belief_p", self.belief_p),
                ("belief_q", self.belief_q),
                ("sn", self.sn),
                ("tn", self.tn),
                ("tier", self.tier),
                ("source", self.source),
                ("content_version", self.content_version)):
            _required_int(value, label=f"legacy metadata.{label}")
        for label, value in (
                ("epistemic_origin", self.epistemic_origin),
                ("subtype", self.subtype),
                ("order_index", self.order_index),
                ("role", self.role),
                ("memory_time_attach", self.memory_time_attach)):
            _optional_int(value, label=f"legacy metadata.{label}")

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "LegacyStructBindMetadata":
        """从存储行逐字段恢复完整 trace，不用 order_index 推断任何对象。"""
        if not isinstance(row, dict):
            raise TypeError("STRUCT_BIND row 必须是 dict")
        return cls(
            _required_int(row.get("strength"), label="edge.strength"),
            _required_int(row.get("base_strength"), label="edge.base_strength"),
            _required_int(row.get("belief_p"), label="edge.belief_p"),
            _required_int(row.get("belief_q"), label="edge.belief_q"),
            _required_int(row.get("sn"), label="edge.sn"),
            _required_int(row.get("tn"), label="edge.tn"),
            _required_int(row.get("tier"), label="edge.tier"),
            _required_int(row.get("source"), label="edge.source"),
            _optional_int(
                row.get("epistemic_origin"), label="edge.epistemic_origin"),
            _optional_int(row.get("subtype"), label="edge.subtype"),
            _optional_int(row.get("order_index"), label="edge.order_index"),
            _optional_int(row.get("role"), label="edge.role"),
            _optional_int(
                row.get("memory_time_attach"),
                label="edge.memory_time_attach"),
            _required_int(
                row.get("content_version"), label="edge.content_version"),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回覆盖旧边全部非端点列的无歧义整数键。"""
        return (
            self.strength,
            self.base_strength,
            self.belief_p,
            self.belief_q,
            self.sn,
            self.tn,
            self.tier,
            self.source,
            *_optional_key(self.epistemic_origin),
            *_optional_key(self.subtype),
            *_optional_key(self.order_index),
            *_optional_key(self.role),
            *_optional_key(self.memory_time_attach),
            self.content_version,
        )


@dataclass(frozen=True)
class TypedStructBindCorrespondence:
    """一条类型明确通过的 source Variable 到 target Variable 对应候选。"""

    source: TypedStructBindEndpoint
    target: TypedStructBindEndpoint
    type_check: TypeCompatibilityResult
    legacy_metadata: LegacyStructBindMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.source, TypedStructBindEndpoint):
            raise TypeError("source 必须是 TypedStructBindEndpoint")
        if not isinstance(self.target, TypedStructBindEndpoint):
            raise TypeError("target 必须是 TypedStructBindEndpoint")
        if not isinstance(self.type_check, TypeCompatibilityResult):
            raise TypeError("type_check 必须是 TypeCompatibilityResult")
        if self.type_check.compatible is not True:
            raise ValueError("typed STRUCT_BIND 候选只能保存明确通过的类型检查")
        source_type = describe_variable(self.source.variable).value_type
        target_type = describe_variable(self.target.variable).value_type
        if (self.type_check.expected_type != target_type
                or self.type_check.actual_type != source_type):
            raise ValueError("typed STRUCT_BIND 类型方向必须是 target expected/source actual")
        if not isinstance(self.legacy_metadata, LegacyStructBindMetadata):
            raise TypeError("legacy_metadata 必须是 LegacyStructBindMetadata")

    def stable_key(self) -> tuple[int, ...]:
        """返回端点、类型 support 和旧边 trace 的完整候选键。"""
        source_key = self.source.stable_key()
        target_key = self.target.stable_key()
        support_keys = tuple(
            identity.stable_key() for identity in self.type_check.support)
        result = [
            len(source_key), *source_key,
            len(target_key), *target_key,
            len(support_keys),
        ]
        for support_key in support_keys:
            result.extend((len(support_key), *support_key))
        metadata_key = self.legacy_metadata.stable_key()
        result.extend((len(metadata_key), *metadata_key))
        return tuple(result)


@dataclass(frozen=True)
class StructBindTypedFailure:
    """保留具体旧边端点和元数据的结构化适配失败。"""

    source_slot: ConceptRef
    target_slot: ConceptRef
    legacy_metadata: LegacyStructBindMetadata
    failure: BindingFailure

    def __post_init__(self) -> None:
        _require_slot_ref(self.source_slot, label="failed source slot")
        _require_slot_ref(self.target_slot, label="failed target slot")
        if not isinstance(self.legacy_metadata, LegacyStructBindMetadata):
            raise TypeError("legacy_metadata 必须是 LegacyStructBindMetadata")
        if not isinstance(self.failure, BindingFailure):
            raise TypeError("failure 必须是 BindingFailure")

    def stable_key(self) -> tuple[int, ...]:
        """返回失败边的确定性键，不把异常文字纳入协议。"""
        reason_key = self.failure.reason.stable_key()
        metadata_key = self.legacy_metadata.stable_key()
        return (
            *self.source_slot,
            *self.target_slot,
            len(reason_key),
            *reason_key,
            len(metadata_key),
            *metadata_key,
        )


@dataclass(frozen=True)
class StructBindTypedReadResult:
    """一次只读适配的全部通过候选和逐边失败，不内置竞争选择。"""

    correspondences: tuple[TypedStructBindCorrespondence, ...]
    failures: tuple[StructBindTypedFailure, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.correspondences, tuple):
            raise TypeError("correspondences 必须是 tuple")
        if not isinstance(self.failures, tuple):
            raise TypeError("failures 必须是 tuple")
        if any(not isinstance(item, TypedStructBindCorrespondence)
               for item in self.correspondences):
            raise TypeError("correspondences 含非法项")
        if any(not isinstance(item, StructBindTypedFailure)
               for item in self.failures):
            raise TypeError("failures 含非法项")
        object.__setattr__(self, "correspondences", tuple(sorted(
            self.correspondences, key=lambda item: item.stable_key())))
        object.__setattr__(self, "failures", tuple(sorted(
            self.failures, key=lambda item: item.stable_key())))


class StructBindTypedAdapter:
    """把旧槽位对应边恢复为 typed Variable 候选并对每条边 fail closed。"""

    def __init__(
            self, resolver: TypeCompatibilityResolver,
            failures: BindingFailureProtocol,
            ) -> None:
        if not hasattr(resolver, "resolve"):
            raise TypeError("resolver 必须实现 TypeCompatibilityResolver")
        if not isinstance(failures, BindingFailureProtocol):
            raise TypeError("failures 必须是 BindingFailureProtocol")
        self._resolver = resolver
        self._failures = failures

    def read_from(
            self, edge_store: EdgeStore, source_slot: ConceptRef,
            endpoints: tuple[TypedStructBindEndpoint, ...],
            ) -> StructBindTypedReadResult:
        """读取 source_slot 全部 STRUCT_BIND 边，显式映射并保留所有竞争项。"""
        if not isinstance(edge_store, EdgeStore):
            raise TypeError("edge_store 必须是 EdgeStore")
        _require_slot_ref(source_slot, label="STRUCT_BIND source_slot")
        if not isinstance(endpoints, tuple):
            raise TypeError("endpoints 必须是 TypedStructBindEndpoint tuple")
        endpoint_map: dict[ConceptRef, TypedStructBindEndpoint] = {}
        for endpoint in endpoints:
            if not isinstance(endpoint, TypedStructBindEndpoint):
                raise TypeError("endpoints 含非法项")
            if endpoint.slot_ref in endpoint_map:
                raise ValueError("同一 slot_ref 不得映射多个 Variable")
            endpoint_map[endpoint.slot_ref] = endpoint

        rows = edge_store.query_from(
            source_slot[0], source_slot[1], edge_type=EDGE_STRUCT_BIND)
        correspondences: list[TypedStructBindCorrespondence] = []
        failures: list[StructBindTypedFailure] = []
        source_endpoint = endpoint_map.get(source_slot)
        for row in rows:
            row_source = (
                _required_int(
                    row.get("space_id_from"), label="edge.space_id_from"),
                _required_int(
                    row.get("local_id_from"), label="edge.local_id_from"),
            )
            target_slot = (
                _required_int(
                    row.get("space_id_to"), label="edge.space_id_to"),
                _required_int(
                    row.get("local_id_to"), label="edge.local_id_to"),
            )
            edge_type = _required_int(
                row.get("edge_type"), label="edge.edge_type")
            if row_source != source_slot or edge_type != EDGE_STRUCT_BIND:
                raise ValueError("EdgeStore 返回了查询范围外的 STRUCT_BIND 行")
            metadata = LegacyStructBindMetadata.from_row(row)
            target_endpoint = endpoint_map.get(target_slot)
            if source_endpoint is None or target_endpoint is None:
                mapped = source_endpoint or target_endpoint
                mapped_descriptor = (
                    describe_variable(mapped.variable)
                    if mapped is not None else None)
                failures.append(StructBindTypedFailure(
                    source_slot,
                    target_slot,
                    metadata,
                    BindingFailure(
                        self._failures.legacy_mapping_missing,
                        variable=mapped.variable if mapped is not None else None,
                        binder=(
                            mapped_descriptor.binder
                            if mapped_descriptor is not None else None),
                        details=(*source_slot, *target_slot),
                    ),
                ))
                continue

            source_descriptor = describe_variable(source_endpoint.variable)
            target_descriptor = describe_variable(target_endpoint.variable)
            type_check = self._resolver.resolve(
                target_descriptor.value_type,
                source_descriptor.value_type,
            )
            if not isinstance(type_check, TypeCompatibilityResult):
                raise TypeError("type resolver 必须返回 TypeCompatibilityResult")
            if (type_check.expected_type != target_descriptor.value_type
                    or type_check.actual_type != source_descriptor.value_type):
                raise ValueError("type resolver 返回了其他类型对的结果")
            if type_check.compatible is not True:
                reason = (
                    self._failures.type_rejected
                    if type_check.compatible is False
                    else self._failures.type_unknown
                )
                failures.append(StructBindTypedFailure(
                    source_slot,
                    target_slot,
                    metadata,
                    BindingFailure(
                        reason,
                        variable=target_endpoint.variable,
                        binder=target_descriptor.binder,
                        expected_type=target_descriptor.value_type,
                        actual_type=source_descriptor.value_type,
                        details=(*source_slot, *target_slot),
                    ),
                ))
                continue
            correspondences.append(TypedStructBindCorrespondence(
                source_endpoint,
                target_endpoint,
                type_check,
                metadata,
            ))
        return StructBindTypedReadResult(
            tuple(correspondences), tuple(failures))


__all__ = [
    "LegacyStructBindMetadata",
    "StructBindTypedAdapter",
    "StructBindTypedFailure",
    "StructBindTypedReadResult",
    "TypedStructBindCorrespondence",
    "TypedStructBindEndpoint",
]
