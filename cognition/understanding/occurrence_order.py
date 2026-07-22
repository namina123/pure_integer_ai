"""来源 occurrence 位置序的正式 writer。

本模块只把相邻 occurrence 写成一种由调用方注入 predicate 的顺序事实。其他篇章、事件、
程序或生成序使用同一个 ``OrderFactIndex`` 但必须注入不同 predicate，不能复用本 writer
的关系身份或把多个序类型压进 qualifier 枚举。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import SourceRef, TypedRef
from pure_integer_ai.cognition.shared.order_facts import (
    OrderFact,
    OrderFactIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceRecord,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _relation_key(value, *, where: str) -> tuple[int, ...]:
    """校验来源课程或程序注入的开放关系键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class OccurrenceOrderProtocol:
    """指定来源位置序所使用的一等 predicate，不定义宿主关系枚举。"""

    relation_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _relation_key(
            self.relation_key,
            where="OccurrenceOrderProtocol.relation_key",
        )


class OccurrenceOrderIntegrityError(ValueError):
    """来源顺序事实、端点详情或链拓扑无法相互核验。"""


@dataclass(frozen=True)
class OccurrenceOrderChain:
    """一个精确来源 scope 内已完整核验的线性 occurrence 链。"""

    scope: ScopeIdentity
    source: SourceRef
    records: tuple[OccurrenceRecord, ...]
    facts: tuple[OrderFact, ...]

    @property
    def occurrences(self) -> tuple[TypedRef, ...]:
        """按来源位置返回链中的一等 occurrence 引用。"""
        return tuple(record.occurrence for record in self.records)


class OccurrenceOrderWriter:
    """把同一来源 scope 内的相邻 occurrence 写成分型顺序事实。"""

    def __init__(self, facts: OrderFactIndex,
                 protocol: OccurrenceOrderProtocol) -> None:
        if not isinstance(facts, OrderFactIndex):
            raise TypeError("facts 必须是 OrderFactIndex")
        if not isinstance(protocol, OccurrenceOrderProtocol):
            raise TypeError("protocol 必须是 OccurrenceOrderProtocol")
        self.facts = facts
        self.protocol = protocol
        self.relation = relation_concept_identity(protocol.relation_key)

    def record_adjacent(
            self, previous: TypedRef, current: TypedRef, *,
            source: SourceRef,
            scope: ScopeIdentity,
            previous_position: int,
            current_position: int,
            ) -> OrderFact:
        """记录同一来源内相邻端点，并把局部位置只作为 scoped qualifier。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if scope.source != source:
            raise ValueError("occurrence 顺序 scope 必须指向同一 SourceRef")
        assert_int(
            previous_position,
            current_position,
            _where="OccurrenceOrderWriter.record_adjacent",
        )
        if previous_position < 0 or current_position <= previous_position:
            raise ValueError("来源 occurrence 位置必须严格递增")
        return self.facts.record(
            self.relation,
            previous,
            current,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
            qualifiers=(previous_position, current_position),
        )

    def facts_in_scope(
            self, scope: ScopeIdentity, *,
            active_only: bool = True,
            ) -> tuple[OrderFact, ...]:
        """读取一个精确来源 scope 内的 occurrence 位置序。"""
        return self.facts.facts(
            self.relation,
            scope=scope,
            active_only=active_only,
        )

    def count(self, *, active_only: bool = True) -> int:
        """返回当前后端全部来源 scope 的 occurrence 位置序事实数。"""
        return self.facts.count(
            self.relation,
            active_only=active_only,
        )

    def clone_for_context(
            self, facts: OrderFactIndex) -> "OccurrenceOrderWriter":
        """在评测 clone 的 OrderFactIndex 上重建独立 writer。"""
        return OccurrenceOrderWriter(facts, self.protocol)


class OccurrenceOrderReader:
    """从分型事实和 occurrence 详情双向恢复精确来源位置链。"""

    def __init__(self, facts: OrderFactIndex,
                 occurrences: OccurrenceIndex,
                 protocol: OccurrenceOrderProtocol) -> None:
        if not isinstance(facts, OrderFactIndex):
            raise TypeError("facts 必须是 OrderFactIndex")
        if not isinstance(occurrences, OccurrenceIndex):
            raise TypeError("occurrences 必须是 OccurrenceIndex")
        if not isinstance(protocol, OccurrenceOrderProtocol):
            raise TypeError("protocol 必须是 OccurrenceOrderProtocol")
        self.facts = facts
        self.occurrences = occurrences
        self.protocol = protocol
        self.relation = relation_concept_identity(protocol.relation_key)

    def read_chain(self, scope: ScopeIdentity) -> OccurrenceOrderChain:
        """恢复一个 scope 的唯一连续链；歧义、断裂或来源污染时拒绝部分结果。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        source = scope.source
        if source is None:
            raise OccurrenceOrderIntegrityError(
                "occurrence 来源顺序 scope 必须携带 SourceRef")
        facts = self.facts.facts(
            self.relation,
            scope=scope,
            active_only=True,
        )
        if not facts:
            return OccurrenceOrderChain(scope, source, (), ())

        records: dict[TypedRef, OccurrenceRecord] = {}
        outgoing: dict[TypedRef, tuple[TypedRef, OrderFact]] = {}
        incoming: dict[TypedRef, TypedRef] = {}
        for fact in facts:
            previous = self._validated_record(
                fact.statement.subject, scope=scope, source=source)
            current = self._validated_record(
                fact.statement.object, scope=scope, source=source)
            self._validate_assertion(
                fact,
                previous=previous,
                current=current,
                source=source,
            )
            if previous.occurrence in outgoing:
                raise OccurrenceOrderIntegrityError("来源顺序存在分支或重复后继")
            if current.occurrence in incoming:
                raise OccurrenceOrderIntegrityError("来源顺序存在汇合或重复前驱")
            records[previous.occurrence] = previous
            records[current.occurrence] = current
            outgoing[previous.occurrence] = (current.occurrence, fact)
            incoming[current.occurrence] = previous.occurrence

        starts = tuple(sorted(
            (ref for ref in records if ref not in incoming),
            key=TypedRef.stable_key,
        ))
        if len(starts) != 1:
            raise OccurrenceOrderIntegrityError("来源顺序不是唯一连通无环链")

        ordered_records: list[OccurrenceRecord] = []
        ordered_facts: list[OrderFact] = []
        visited: set[TypedRef] = set()
        cursor = starts[0]
        while True:
            if cursor in visited:
                raise OccurrenceOrderIntegrityError("来源顺序形成环")
            visited.add(cursor)
            ordered_records.append(records[cursor])
            step = outgoing.get(cursor)
            if step is None:
                break
            cursor, fact = step
            ordered_facts.append(fact)

        if len(visited) != len(records) or len(ordered_facts) != len(facts):
            raise OccurrenceOrderIntegrityError("来源顺序存在断裂或未连接分量")
        return OccurrenceOrderChain(
            scope,
            source,
            tuple(ordered_records),
            tuple(ordered_facts),
        )

    def clone_for_context(
            self, facts: OrderFactIndex,
            occurrences: OccurrenceIndex,
            ) -> "OccurrenceOrderReader":
        """在评测 clone 的事实和 occurrence 索引上重建独立 reader。"""
        return OccurrenceOrderReader(facts, occurrences, self.protocol)

    def _validated_record(
            self, ref: TypedRef, *,
            scope: ScopeIdentity,
            source: SourceRef,
            ) -> OccurrenceRecord:
        """核验顺序端点确属请求的来源、scope 和 parser version。"""
        try:
            record = self.occurrences.read(ref)
        except (KeyError, TypeError, ValueError) as exc:
            raise OccurrenceOrderIntegrityError(
                "来源顺序端点不是可回读 occurrence") from exc
        if record.scope != scope or record.source != source:
            raise OccurrenceOrderIntegrityError("来源顺序端点跨越了 SourceRef 或 scope")
        if record.parser_version != source.versions.parser.value:
            raise OccurrenceOrderIntegrityError("来源顺序端点 parser version 不一致")
        return record

    @staticmethod
    def _validate_assertion(
            fact: OrderFact, *,
            previous: OccurrenceRecord,
            current: OccurrenceRecord,
            source: SourceRef,
            ) -> None:
        """核对 assertion 来源字段、限定位置和 occurrence 详情完全一致。"""
        assertion = fact.statement.assertion
        if assertion.provenance_kind != source.source_kind:
            raise OccurrenceOrderIntegrityError("来源顺序 provenance 与 SourceRef 不一致")
        if assertion.epistemic_origin != 0:
            raise OccurrenceOrderIntegrityError("来源位置事实不得携带认识论强度来源")
        if assertion.content_version != source.versions.parser.value:
            raise OccurrenceOrderIntegrityError("来源顺序 assertion parser version 不一致")
        expected = (previous.document_index, current.document_index)
        if assertion.qualifiers != expected:
            raise OccurrenceOrderIntegrityError("来源顺序限定位置与端点详情不一致")
        if current.document_index != previous.document_index + 1:
            raise OccurrenceOrderIntegrityError("来源顺序存在位置 gap 或逆序")


__all__ = [
    "OccurrenceOrderChain",
    "OccurrenceOrderIntegrityError",
    "OccurrenceOrderProtocol",
    "OccurrenceOrderReader",
    "OccurrenceOrderWriter",
]
