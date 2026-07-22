"""把 L-06 已核验 occurrence 链投影为 H-06 typed 顺序观察。

adapter 不从词面、concept、绝对位置或旧 role_seq 猜角色。调用方 mapper 必须为每个
候选显式给出 ``OrderPattern`` 及规范 slot 对应哪个真实端点；本模块只核验端点来自
同一条 L-06 事实，并复制完整 assertion、SourceRef、scope、位置和 qualifiers。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFact
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceRecord,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderReader,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


@dataclass(frozen=True)
class OccurrenceOrderStep:
    """一条 L-06 相邻事实及其可回读端点和完整一等身份。"""

    index: int
    previous: OccurrenceRecord
    current: OccurrenceRecord
    previous_identity: ObjectIdentity
    current_identity: ObjectIdentity
    fact: OrderFact

    def __post_init__(self) -> None:
        assert_int(self.index, _where="OccurrenceOrderStep.index")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("OccurrenceOrderStep.index 必须为非负严格整数")
        if not isinstance(self.previous, OccurrenceRecord):
            raise TypeError("previous 必须是 OccurrenceRecord")
        if not isinstance(self.current, OccurrenceRecord):
            raise TypeError("current 必须是 OccurrenceRecord")
        if not isinstance(self.previous_identity, ObjectIdentity):
            raise TypeError("previous_identity 必须是 ObjectIdentity")
        if not isinstance(self.current_identity, ObjectIdentity):
            raise TypeError("current_identity 必须是 ObjectIdentity")
        if not isinstance(self.fact, OrderFact):
            raise TypeError("fact 必须是 OrderFact")

    @property
    def records(self) -> tuple[OccurrenceRecord, OccurrenceRecord]:
        """按来源位置返回事实的两个端点记录。"""
        return self.previous, self.current

    @property
    def identities(self) -> tuple[ObjectIdentity, ObjectIdentity]:
        """按来源位置返回事实的两个完整 occurrence 身份。"""
        return self.previous_identity, self.current_identity


@dataclass(frozen=True)
class TypedOrderProjection:
    """调用方声明一个模式的两个规范 slot 分别对应哪个事实端点。"""

    pattern: OrderPattern
    endpoint_order: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, OrderPattern):
            raise TypeError("TypedOrderProjection.pattern 必须是 OrderPattern")
        if not isinstance(self.endpoint_order, tuple):
            raise TypeError("endpoint_order 必须是整数 tuple")
        assert_int(*self.endpoint_order, _where="TypedOrderProjection.endpoint_order")
        if (len(self.endpoint_order) != 2
                or any(type(item) is not int for item in self.endpoint_order)
                or set(self.endpoint_order) != set(range(2))):
            raise ValueError("endpoint_order 必须完整映射两个不同事实端点")


class OccurrenceOrderMapper(Protocol):
    """S-02 或课程 adapter 注入的 typed 结构、slot 和 context 映射边界。"""

    def __call__(
            self, step: OccurrenceOrderStep,
            ) -> tuple[TypedOrderProjection, ...]: ...


@dataclass(frozen=True)
class MappedOrderObservation:
    """保留模式与由 L-06 事实构造的完整 H-06 观察。"""

    pattern: OrderPattern
    observation: OrderObservation


class OccurrenceOrderHypothesisAdapter:
    """只读消费 L-06 链，并拒绝 mapper 伪造或遗漏事实端点。"""

    def __init__(self, reader: OccurrenceOrderReader) -> None:
        """绑定正式 L-06 reader；adapter 不拥有事实或 Hypothesis ledger。"""
        if not isinstance(reader, OccurrenceOrderReader):
            raise TypeError("reader 必须是 OccurrenceOrderReader")
        self.reader = reader

    def project(
            self, scope: ScopeIdentity,
            mapper: OccurrenceOrderMapper,
            ) -> tuple[MappedOrderObservation, ...]:
        """逐事实调用 typed mapper，并从权威链字段构造不可伪造的观察。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if not callable(mapper):
            raise TypeError("mapper 必须可调用")
        chain = self.reader.read_chain(scope)
        if not chain.facts:
            return ()
        if len(chain.records) != len(chain.facts) + 1:
            raise ValueError("L-06 chain 的记录数与事实数不一致")
        results: list[MappedOrderObservation] = []
        seen: set[tuple[OrderPattern, tuple[int, int], int]] = set()
        ontology = self.reader.occurrences.ontology
        for index, fact in enumerate(chain.facts):
            previous = chain.records[index]
            current = chain.records[index + 1]
            step = OccurrenceOrderStep(
                index,
                previous,
                current,
                ontology.identity_of(previous.occurrence),
                ontology.identity_of(current.occurrence),
                fact,
            )
            projections = mapper(step)
            if (not isinstance(projections, tuple)
                    or any(not isinstance(item, TypedOrderProjection)
                           for item in projections)):
                raise TypeError("mapper 必须返回 TypedOrderProjection tuple")
            for projection in projections:
                marker = (projection.pattern, projection.endpoint_order, index)
                if marker in seen:
                    raise ValueError("mapper 不得重复投影同一事实和模式")
                seen.add(marker)
                results.append(self._observation(step, projection))
        return tuple(results)

    @staticmethod
    def _observation(
            step: OccurrenceOrderStep,
            projection: TypedOrderProjection,
            ) -> MappedOrderObservation:
        """按 mapper 的规范 slot 映射复制真实 occurrence 和 assertion 字段。"""
        pattern = projection.pattern
        identities = step.identities
        records = step.records
        first_index, second_index = projection.endpoint_order
        assertion = step.fact.statement.assertion
        observation = OrderObservation(
            assertion.scope.source,
            assertion.scope,
            assertion.stable_key(),
            pattern.language_branch,
            pattern.structure_family,
            pattern.structure_candidate,
            pattern.first_slot,
            pattern.second_slot,
            pattern.context,
            pattern.conditions,
            identities[first_index],
            identities[second_index],
            records[first_index].document_index,
            records[second_index].document_index,
            assertion.qualifiers,
        )
        return MappedOrderObservation(pattern, observation)


__all__ = [
    "MappedOrderObservation",
    "OccurrenceOrderHypothesisAdapter",
    "OccurrenceOrderMapper",
    "OccurrenceOrderStep",
    "TypedOrderProjection",
]
