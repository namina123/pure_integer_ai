"""递归 Span、结构分类、成分关系和候选生命周期的统一领域入口。

Span 身份只表达来源内的精确成员范围。字符、词、短语、命题等结构含义由调用方
注入的一等 StructureConcept 表达；候选活动状态由 Hypothesis 断言和 supersede 事件表达。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    normalize_span_members,
    span_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.assertion_identity import IDENTITY_SPAN_ROLE
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.span import SpanStorageRecord, SpanStore


def _protocol_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或程序注入的开放图关系键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SpanProtocol:
    """指定 Span 的结构、成分、occurrence 和候选关系 predicate。"""

    structure_relation_key: tuple[int, ...]
    constituent_relation_key: tuple[int, ...]
    occurrence_relation_key: tuple[int, ...]
    candidate_relation_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, value in (
                ("structure_relation_key", self.structure_relation_key),
                ("constituent_relation_key", self.constituent_relation_key),
                ("occurrence_relation_key", self.occurrence_relation_key),
                ("candidate_relation_key", self.candidate_relation_key)):
            _protocol_key(value, where=f"SpanProtocol.{label}")
        if len({
                self.structure_relation_key,
                self.constituent_relation_key,
                self.occurrence_relation_key,
                self.candidate_relation_key,
                }) != 4:
            raise ValueError("SpanProtocol 的四类关系 predicate 必须互不相同")


@dataclass(frozen=True)
class SpanRecord:
    """可从 Span 回到来源、完整成员和图内结构关系的读模型。"""

    span: TypedRef
    source: SourceRef
    scope: ScopeIdentity
    raw_text: str
    members: tuple[tuple[int, int], ...]
    ordinal: int
    parser_version: int
    structures: tuple[TypedRef, ...]
    constituents: tuple[GraphStatement, ...]
    occurrences: tuple[GraphStatement, ...]
    candidate_links: tuple[GraphStatement, ...]


@dataclass(frozen=True)
class _SpanDetails:
    """由权威表恢复的不可变 Span 来源详情缓存。"""

    source: SourceRef
    scope: ScopeIdentity
    raw_text: str
    members: tuple[tuple[int, int], ...]
    ordinal: int
    parser_version: int


class SpanIndex:
    """联合图本体、来源记录和 Span 详情表维护递归结构。"""

    def __init__(
            self, ontology: GraphOntology,
            scoped_identities: ScopedIdentityStore,
            protocol: SpanProtocol,
            occurrence_index: OccurrenceIndex | None = None,
            ) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(scoped_identities, ScopedIdentityStore):
            raise TypeError("scoped_identities 必须是 ScopedIdentityStore")
        if not isinstance(protocol, SpanProtocol):
            raise TypeError("protocol 必须是 SpanProtocol")
        if occurrence_index is not None and not isinstance(
                occurrence_index, OccurrenceIndex):
            raise TypeError("occurrence_index 必须是 OccurrenceIndex 或 None")
        self.ontology = ontology
        self.scoped_identities = scoped_identities
        self.protocol = protocol
        self.occurrence_index = occurrence_index
        self._sources = SourceRecordRepository(
            ontology.backend, registry=scoped_identities.registry)
        self._spans = SpanStore(ontology.backend)
        self._span_cache: dict[ObjectIdentity, TypedRef] = {}
        self._predicate_cache: dict[tuple[int, ...], TypedRef] = {}
        self._detail_cache: dict[TypedRef, _SpanDetails] = {}
        self._constituent_cache: dict[TypedRef, tuple[TypedRef, ...]] = {}
        self._structure_statement_cache: dict[
            tuple[TypedRef, TypedRef], GraphStatement
        ] = {}
        self._constituent_statement_cache: dict[
            tuple[TypedRef, TypedRef, int], GraphStatement
        ] = {}

    def ensure(
            self, *, source: SourceRef, raw_text: str,
            scope: ScopeIdentity,
            members: tuple[tuple[int, int], ...],
            ordinal: int = 0,
            structures: tuple[ObjectIdentity | TypedRef, ...] = (),
            ) -> SpanRecord:
        """物化 Span 后完整回读全部当前图关系，供需要快照的调用方使用。"""
        span = self.ensure_ref(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=members,
            ordinal=ordinal,
            structures=structures,
        )
        return self.read(span)

    def ensure_ref(
            self, *, source: SourceRef, raw_text: str,
            scope: ScopeIdentity,
            members: tuple[tuple[int, int], ...],
            ordinal: int = 0,
            structures: tuple[ObjectIdentity | TypedRef, ...] = (),
            ) -> TypedRef:
        """物化精确 Span 并追加结构分类，只返回已核验引用而不扫描派生关系。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text 必须是字符串")
        if not isinstance(scope, ScopeIdentity) or scope.source != source:
            raise ValueError("Span scope 必须指向同一 SourceRef")
        assert_int(ordinal, _where="SpanIndex.ensure.ordinal")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("Span ordinal 必须为非负严格整数")
        normalized = normalize_span_members(members)
        if any(end > len(raw_text) for _, end in normalized):
            raise ValueError("Span member 超出来源原文")
        source_record = self._sources.put(source.stable_key(), raw_text)
        identity = span_identity(
            source,
            members=normalized,
            ordinal=ordinal,
        )
        span = self._span_cache.get(identity)
        if span is None:
            span = self.ontology.materialize(identity, tier=TIER_SHADOW)
            self._span_cache[identity] = span
        scope_hash = self.scoped_identities.register_scope(scope)
        self._spans.add(SpanStorageRecord(
            span.space_id,
            span.local_id,
            source_record.source_hash,
            scope_hash,
            len(normalized),
            ordinal,
            source.versions.parser.value,
            normalized[0][0],
            normalized[-1][1],
        ), normalized)
        details = _SpanDetails(
            source,
            scope,
            raw_text,
            normalized,
            ordinal,
            source.versions.parser.value,
        )
        self._detail_cache[span] = details
        for structure in structures:
            self._add_structure(span, structure, details)
        return span

    def members_of(self, span: TypedRef) -> tuple[tuple[int, int], ...]:
        """返回已核验 Span 本体成员，不读取结构或候选等派生 statement。"""
        return self._details(span).members

    def add_structure(
            self, span: TypedRef,
            structure: ObjectIdentity | TypedRef) -> GraphStatement:
        """把 Span 分类到调用方注入的一等 StructureConcept。"""
        details = self._details(span)
        return self._add_structure(span, structure, details)

    def ensure_role_ordinal(
            self, role: ObjectIdentity | TypedRef) -> int:
        """登记一等 Span 角色，并返回跨物化器碰撞可核验的非零 ordinal。"""
        role_ref = self._role_ref(role, materialize=True)
        role_hash = self.ontology.identity_hash_of(role_ref)
        return self.scoped_identities.registry.register(
            IDENTITY_SPAN_ROLE,
            (role_hash,),
        )

    def resolve_role_ordinal(
            self, role: ObjectIdentity | TypedRef) -> int | None:
        """只读解析已登记角色 ordinal；角色或登记不存在时保持零写并返回 None。"""
        role_ref = self._role_ref(role, materialize=False)
        if role_ref is None:
            return None
        role_hash = self.ontology.identity_hash_of(role_ref)
        return self.scoped_identities.registry.find(
            IDENTITY_SPAN_ROLE,
            (role_hash,),
        )

    def _add_structure(
            self, span: TypedRef,
            structure: ObjectIdentity | TypedRef,
            details: _SpanDetails) -> GraphStatement:
        """使用已核验详情追加结构关系，避免同一写事务重复回读。"""
        structure_ref = self._structure_ref(structure)
        cache_key = (span, structure_ref)
        cached = self._structure_statement_cache.get(cache_key)
        if cached is not None:
            return cached
        statement = self.ontology.relate(
            self._predicate(self.protocol.structure_relation_key),
            span,
            structure_ref,
            scope=details.scope,
            provenance_kind=details.source.source_kind,
            content_version=details.parser_version,
        )
        self._structure_statement_cache[cache_key] = statement
        return statement

    def add_constituent(
            self, parent: TypedRef, child: TypedRef, *,
            member_ordinal: int) -> GraphStatement:
        """追加 parent 到 child 的有序递归成员关系，并拒绝跨来源和成环。"""
        assert_int(member_ordinal, _where="SpanIndex.add_constituent")
        if type(member_ordinal) is not int or member_ordinal < 0:
            raise ValueError("constituent member_ordinal 必须为非负严格整数")
        parent_record = self._details(parent)
        child_record = self._details(child)
        if parent == child:
            raise ValueError("Span 不得把自身作为 constituent")
        if parent_record.source != child_record.source:
            raise ValueError("constituent 必须位于同一来源")
        if parent_record.scope != child_record.scope:
            raise ValueError("constituent 必须位于同一 scope")
        if not self._members_contain(
                parent_record.members, child_record.members):
            raise ValueError("child Span 必须完整落在 parent 成员范围内")
        cache_key = (parent, child, member_ordinal)
        cached = self._constituent_statement_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._reaches(child, parent):
            raise ValueError("constituent 关系不得形成环")
        statement = self.ontology.relate(
            self._predicate(self.protocol.constituent_relation_key),
            parent,
            child,
            scope=parent_record.scope,
            provenance_kind=parent_record.source.source_kind,
            content_version=parent_record.parser_version,
            qualifiers=(member_ordinal,),
        )
        children = set(self._children(parent))
        children.add(child)
        self._constituent_cache[parent] = tuple(sorted(
            children,
            key=lambda ref: ref.stable_key(),
        ))
        self._constituent_statement_cache[cache_key] = statement
        return statement

    def add_occurrence(
            self, span: TypedRef, occurrence: TypedRef, *,
            member_ordinal: int) -> GraphStatement:
        """把 winner Span 关联到同来源 occurrence，不为未决候选伪造 occurrence。"""
        if self.occurrence_index is None:
            raise RuntimeError("SpanIndex 未装配 OccurrenceIndex")
        assert_int(member_ordinal, _where="SpanIndex.add_occurrence")
        if type(member_ordinal) is not int or member_ordinal < 0:
            raise ValueError("occurrence member_ordinal 必须为非负严格整数")
        span_record = self._details(span)
        occurrence_record = self.occurrence_index.read(occurrence)
        if span_record.source != occurrence_record.source:
            raise ValueError("Span 与 occurrence 必须来自同一 SourceRef")
        if span_record.scope != occurrence_record.scope:
            raise ValueError("Span 与 occurrence 必须位于同一 scope")
        if not self._members_contain(
                span_record.members,
                ((occurrence_record.start, occurrence_record.end),)):
            raise ValueError("occurrence 必须完整落在 Span 成员范围内")
        return self.ontology.relate(
            self._predicate(self.protocol.occurrence_relation_key),
            span,
            occurrence,
            scope=span_record.scope,
            provenance_kind=span_record.source.source_kind,
            content_version=span_record.parser_version,
            qualifiers=(member_ordinal,),
        )

    def add_candidate(
            self, hypothesis: HypothesisKey,
            span: TypedRef) -> GraphStatement:
        """把边界 Hypothesis 作为一等候选主体关联到 Span。"""
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        span_record = self._details(span)
        if hypothesis.observation != span_record.source:
            raise ValueError("候选 Hypothesis 与 Span 必须来自同一 observation")
        if hypothesis.scope != span_record.scope:
            raise ValueError("候选 Hypothesis 与 Span 必须使用同一 scope")
        hypothesis_ref = self.ontology.materialize(
            hypothesis.object_identity(), tier=TIER_SHADOW)
        return self.ontology.relate(
            self._predicate(self.protocol.candidate_relation_key),
            hypothesis_ref,
            span,
            scope=span_record.scope,
            provenance_kind=span_record.source.source_kind,
            content_version=span_record.parser_version,
        )

    def supersede_candidate(
            self, old: HypothesisKey, new: HypothesisKey,
            timestamp: LogicalTimestamp) -> int:
        """用 append-only 事件替代边界候选 link，不删除旧 Span 和旧 statement。"""
        old_statement, new_statement = self.validate_candidate_supersede(
            old,
            new,
            timestamp,
        )
        return self.scoped_identities.supersede(
            old_statement.assertion,
            new_statement.assertion,
            timestamp,
        )

    def validate_candidate_supersede(
            self, old: HypothesisKey, new: HypothesisKey,
            timestamp: LogicalTimestamp,
            ) -> tuple[GraphStatement, GraphStatement]:
        """在 ledger 或图写入前核验边界候选替代的完整竞争边界。"""
        if not isinstance(old, HypothesisKey) or not isinstance(
                new, HypothesisKey):
            raise TypeError("候选替代需要两个 HypothesisKey")
        if not isinstance(timestamp, LogicalTimestamp):
            raise TypeError("候选替代需要 LogicalTimestamp")
        if (old == new
                or old.hypothesis_kind != new.hypothesis_kind
                or old.competition_key != new.competition_key
                or old.scope != new.scope
                or old.observation != new.observation):
            raise ValueError("replacement 必须属于同一 Span 竞争组")
        if timestamp.clock.scope != old.scope:
            raise ValueError("候选 supersede 时间戳必须由同一 scope 拥有")
        old_statement = self._candidate_statement(old)
        new_statement = self._candidate_statement(new)
        if self.scoped_identities.assertion_is_superseded(
                new_statement.assertion_hash):
            raise ValueError("replacement Span candidate 必须仍为 active")
        return old_statement, new_statement

    def candidate_statements(
            self, *, active_only: bool = True
            ) -> tuple[GraphStatement, ...]:
        """读取全部 Span 候选 link，并可过滤已被替代的历史断言。"""
        statements = self._statements(self.protocol.candidate_relation_key)
        if active_only:
            statements = tuple(
                statement for statement in statements
                if not self.scoped_identities.assertion_is_superseded(
                    statement.assertion_hash)
            )
        return statements

    def read(self, span: TypedRef) -> SpanRecord:
        """完整核验 Span 图身份、来源、成员、scope 和所有分型关系。"""
        details = self._details(span, use_cache=False)
        structure_statements = self._statements(
            self.protocol.structure_relation_key,
            subject=span,
        )
        structures = tuple(statement.object for statement in structure_statements)
        if any(ref.object_kind != OBJECT_STRUCTURE_CONCEPT for ref in structures):
            raise ValueError("Span 结构关系指向了非 StructureConcept")
        constituents = tuple(sorted(
            self._statements(
                self.protocol.constituent_relation_key,
                subject=span,
            ),
            key=lambda item: (
                item.assertion.qualifiers,
                item.object.stable_key(),
                item.assertion_hash,
            ),
        ))
        if any(item.object.object_kind != OBJECT_SPAN for item in constituents):
            raise ValueError("Span constituent 指向了非 Span 对象")
        self._constituent_cache[span] = tuple(sorted(
            {item.object for item in constituents},
            key=lambda ref: ref.stable_key(),
        ))
        occurrences = tuple(sorted(
            self._statements(
                self.protocol.occurrence_relation_key,
                subject=span,
            ),
            key=lambda item: (
                item.assertion.qualifiers,
                item.object.stable_key(),
                item.assertion_hash,
            ),
        ))
        if any(item.object.object_kind != OBJECT_OCCURRENCE for item in occurrences):
            raise ValueError("Span occurrence 关系指向了非 occurrence")
        candidate_links = self._statements(
            self.protocol.candidate_relation_key,
            object_ref=span,
        )
        if any(item.subject.object_kind != OBJECT_HYPOTHESIS
               for item in candidate_links):
            raise ValueError("Span candidate 关系来自非 Hypothesis")
        scoped_statements = (
            *structure_statements,
            *constituents,
            *occurrences,
            *candidate_links,
        )
        if any(item.assertion.scope != details.scope
               for item in scoped_statements):
            raise ValueError("Span 图关系与详情 scope 不一致")
        if any(
                item.assertion.provenance_kind != details.source.source_kind
                or item.assertion.content_version
                != details.source.versions.parser.value
                for item in scoped_statements):
            raise ValueError("Span 图关系与来源版本不一致")
        return SpanRecord(
            span,
            details.source,
            details.scope,
            details.raw_text,
            details.members,
            details.ordinal,
            details.parser_version,
            structures,
            constituents,
            occurrences,
            candidate_links,
        )

    def span_count(self) -> int:
        """返回当前后端已保存的唯一 Span 数。"""
        return self._spans.span_count()

    def clone_for_context(
            self, ontology: GraphOntology,
            scoped_identities: ScopedIdentityStore,
            occurrence_index: OccurrenceIndex | None,
            ) -> "SpanIndex":
        """在评测 clone 的图、身份和 occurrence 索引上重建 facade。"""
        return SpanIndex(
            ontology,
            scoped_identities,
            self.protocol,
            occurrence_index,
        )

    def clear_runtime_caches(self) -> None:
        """清空 Span facade 的全部已核验缓存，供外部状态替换后重新审计。"""
        self._sources.clear_runtime_caches()
        self._spans.clear_runtime_caches()
        self._span_cache.clear()
        self._predicate_cache.clear()
        self._detail_cache.clear()
        self._constituent_cache.clear()
        self._structure_statement_cache.clear()
        self._constituent_statement_cache.clear()

    def _structure_ref(
            self, structure: ObjectIdentity | TypedRef) -> TypedRef:
        """物化或核验 StructureConcept 端点。"""
        if isinstance(structure, ObjectIdentity):
            if structure.object_kind != OBJECT_STRUCTURE_CONCEPT:
                raise ValueError("Span 结构必须是一等 StructureConcept")
            return self.ontology.materialize(structure, tier=TIER_SHADOW)
        if isinstance(structure, TypedRef):
            identity = self.ontology.identity_of(structure)
            if identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
                raise ValueError("Span 结构必须是一等 StructureConcept")
            return structure
        raise TypeError("structure 必须是 ObjectIdentity 或 TypedRef")

    def _role_ref(
            self, role: ObjectIdentity | TypedRef, *, materialize: bool
            ) -> TypedRef | None:
        """把 Hypothesis/StructureConcept 角色转换为已核验引用，并支持只读缺失。"""
        if isinstance(role, ObjectIdentity):
            if role.object_kind not in {
                    OBJECT_HYPOTHESIS, OBJECT_STRUCTURE_CONCEPT}:
                raise ValueError("Span role 必须是一等 Hypothesis 或 StructureConcept")
            if materialize:
                return self.ontology.materialize(role, tier=TIER_SHADOW)
            return self.ontology.resolve(role)
        if isinstance(role, TypedRef):
            identity = self.ontology.identity_of(role)
            if identity.object_kind not in {
                    OBJECT_HYPOTHESIS, OBJECT_STRUCTURE_CONCEPT}:
                raise ValueError("Span role 必须是一等 Hypothesis 或 StructureConcept")
            return role
        raise TypeError("Span role 必须是 ObjectIdentity 或 TypedRef")

    def _predicate(self, key: tuple[int, ...]) -> TypedRef:
        """幂等物化协议注入的关系 predicate。"""
        predicate = self._predicate_cache.get(key)
        if predicate is None:
            predicate = self.ontology.materialize(
                relation_concept_identity(key), tier=TIER_SHADOW)
            self._predicate_cache[key] = predicate
        return predicate

    def _statements(
            self, key: tuple[int, ...], *,
            subject: TypedRef | None = None,
            object_ref: TypedRef | None = None,
            ) -> tuple[GraphStatement, ...]:
        """只读查询协议关系；predicate 尚未物化时返回空结果。"""
        predicate = self.ontology.resolve(relation_concept_identity(key))
        if predicate is None:
            return ()
        return self.ontology.statements(
            predicate=predicate,
            subject=subject,
            object_ref=object_ref,
        )

    def _candidate_statement(self, hypothesis: HypothesisKey) -> GraphStatement:
        """回读一个 Hypothesis 唯一、真实存在的 Span candidate statement。"""
        hypothesis_ref = self.ontology.resolve(hypothesis.object_identity())
        if hypothesis_ref is None:
            raise LookupError("Hypothesis 尚未物化 Span candidate statement")
        matches = self._statements(
            self.protocol.candidate_relation_key,
            subject=hypothesis_ref,
        )
        if len(matches) != 1:
            raise ValueError("Hypothesis 没有唯一 Span candidate statement")
        return matches[0]

    def _details(
            self, span: TypedRef, *, use_cache: bool = True
            ) -> _SpanDetails:
        """从缓存或权威表恢复 Span 本体详情，不查询派生图关系。"""
        if use_cache:
            cached = self._detail_cache.get(span)
            if cached is not None:
                return cached
        identity = self.ontology.identity_of(span)
        if identity.object_kind != OBJECT_SPAN:
            raise ValueError("read 需要一等 Span 引用")
        stored, members = self._spans.read(span.space_id, span.local_id)
        source_record = self._sources.read(stored.source_hash)
        source = SourceRef.from_stable_key(source_record.source_key)
        expected = span_identity(
            source,
            members=members,
            ordinal=stored.ordinal,
        )
        if expected != identity:
            raise ValueError("Span 图身份与详情表不一致")
        if any(end > len(source_record.raw_text) for _, end in members):
            raise ValueError("Span member 超出来源原文")
        scope = self.scoped_identities.load_scope(stored.scope_hash)
        if scope.source != source:
            raise ValueError("Span scope 与来源记录不一致")
        if stored.parser_version != source.versions.parser.value:
            raise ValueError("Span parser version 与 SourceRef 不一致")
        details = _SpanDetails(
            source,
            scope,
            source_record.raw_text,
            members,
            stored.ordinal,
            stored.parser_version,
        )
        self._detail_cache[span] = details
        return details

    def _children(self, span: TypedRef) -> tuple[TypedRef, ...]:
        """从缓存或图 statement 恢复一个 Span 的直接 constituent。"""
        cached = self._constituent_cache.get(span)
        if cached is not None:
            return cached
        children = tuple(sorted(
            {
                statement.object
                for statement in self._statements(
                    self.protocol.constituent_relation_key,
                    subject=span,
                )
            },
            key=lambda ref: ref.stable_key(),
        ))
        if any(child.object_kind != OBJECT_SPAN for child in children):
            raise ValueError("Span constituent 指向了非 Span 对象")
        self._constituent_cache[span] = children
        return children

    def _reaches(self, start: TypedRef, target: TypedRef) -> bool:
        """沿 constituent 关系检查 target 是否已是 start 的后代。"""
        predicate = self.ontology.resolve(relation_concept_identity(
            self.protocol.constituent_relation_key))
        if predicate is None:
            return False
        frontier = [start]
        seen: set[TypedRef] = set()
        while frontier:
            current = frontier.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self._children(current))
        return False

    @staticmethod
    def _members_contain(
            parent: tuple[tuple[int, int], ...],
            child: tuple[tuple[int, int], ...]) -> bool:
        """判断 child 的每个完整区间都被 parent 某一成员包含。"""
        return all(any(
            parent_start <= child_start and child_end <= parent_end
            for parent_start, parent_end in parent
        ) for child_start, child_end in child)


__all__ = ["SpanIndex", "SpanProtocol", "SpanRecord"]
