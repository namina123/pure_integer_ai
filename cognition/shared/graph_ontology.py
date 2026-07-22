"""一等图对象和动态 predicate statement 的领域 facade。

具体语言、表示、结构角色和执行操作都由调用方注入的概念身份表达。本模块只负责
物化、分型、scope/provenance 绑定和确定性图遍历，不维护自然语言字面量或关系枚举。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_HYPOTHESIS,
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    OwnerScope,
    SourceRef,
    TypedRef,
    VersionBundle,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.scope_identity import (
    AssertionIdentity,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    SEMANTIC_OBJECT_KINDS,
    validate_semantic_identity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.graph_object import (
    GraphObjectIntegrityError,
    GraphObjectIdentitySpec,
    GraphObjectRecord,
    GraphObjectRepository,
)
from pure_integer_ai.storage.graph_statement import (
    GraphStatementRecord,
    GraphStatementStore,
)
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_ROLE_GRAPH_STATEMENT,
)
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_GRAPH_OBJECT,
    IDENTITY_SOURCE_RECORD,
)
from pure_integer_ai.storage.node_store import (
    NODE_CONCEPT,
    NodeStore,
    TIER_SHADOW,
)
from pure_integer_ai.storage.spaces.registry import SpaceIdentity


@dataclass(frozen=True)
class GraphStatement:
    """已从存储和 identity registry 双向核验的领域 statement。"""

    assertion_hash: int
    predicate_identity_hash: int
    predicate: TypedRef
    subject: TypedRef
    object: TypedRef
    scope_hash: int
    assertion: AssertionIdentity


def relation_concept_identity(
        relation_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造动态关系概念身份；具体关系含义由图或课程提供。"""
    if not isinstance(relation_key, tuple) or not relation_key:
        raise ValueError("relation_key 必须是非空严格整数元组")
    assert_int(*relation_key, _where="relation_concept_identity.relation_key")
    if any(type(value) is not int for value in relation_key):
        raise ValueError("relation_key 必须使用严格整数")
    return ObjectIdentity(OBJECT_CONCEPT, relation_key, owner, versions)


class GraphOntology:
    """在一个稳定空间内物化对象、记录 statement 并执行纯图路径查询。"""

    def __init__(self, backend: StorageBackend, *, space_id: int,
                 space_identity: SpaceIdentity,
                 scoped_identities: ScopedIdentityStore) -> None:
        assert_int(space_id, _where="GraphOntology.space_id")
        if type(space_id) is not int or space_id <= 0:
            raise ValueError("GraphOntology.space_id 必须为严格正整数")
        self._backend = backend
        self._space_id = space_id
        self._space_identity = space_identity
        self._scoped_identities = scoped_identities
        self._objects = GraphObjectRepository(
            backend, registry=scoped_identities.registry)
        self._statements = GraphStatementStore(
            backend,
            self._objects,
            scoped_identities.assertion_records,
        )
        self._nodes = NodeStore(backend)
        self._identity_to_ref: dict[ObjectIdentity, TypedRef] = {}
        self._ref_to_identity: dict[TypedRef, ObjectIdentity] = {}
        self._records_by_node: dict[tuple[int, int], GraphObjectRecord] = {}
        self._refs_by_node: dict[tuple[int, int], TypedRef] = {}
        self._source_hashes: dict[SourceRef, int] = {}

    @property
    def space_id(self) -> int:
        """返回当前图对象的运行时空间编址。"""
        return self._space_id

    @property
    def space_identity(self) -> SpaceIdentity:
        """返回不依赖运行时分配顺序的稳定空间身份。"""
        return self._space_identity

    @property
    def backend(self) -> StorageBackend:
        """返回承载当前图对象和 statement 的存储后端。"""
        return self._backend

    def materialize(self, identity: ObjectIdentity, *,
                    tier: int = TIER_SHADOW) -> TypedRef:
        """把权威对象身份幂等物化为一等概念节点，并返回分型引用。"""
        if identity.object_kind in SEMANTIC_OBJECT_KINDS:
            validate_semantic_identity(identity)
        contract = object_contracts_by_kind().get(identity.object_kind)
        if contract is None or not contract.authoritative_identity:
            raise ValueError("兼容投影或未注册对象不得物化为权威图对象")
        cached = self._identity_to_ref.get(identity)
        if cached is not None:
            return cached
        existing_hash = self._objects.find_identity(identity.stable_key())
        if existing_hash is not None:
            record = self._objects.read(existing_hash)
            return self._cache_object(
                identity, self._validate_record(record))

        local_id = self._backend.next_id(self._space_id)
        self._nodes.put(
            self._space_id, local_id,
            node_type=NODE_CONCEPT,
            tier=tier,
        )
        record = self._objects.add(
            self._identity_spec(identity),
            object_kind=identity.object_kind,
            space_id=self._space_id,
            local_id=local_id,
            space_identity_key=self._space_identity.stable_key(),
        )
        return self._cache_object(identity, self._validate_record(record))

    def resolve(self, identity: ObjectIdentity) -> TypedRef | None:
        """只读解析已物化对象，不存在时不登记 identity。"""
        cached = self._identity_to_ref.get(identity)
        if cached is not None:
            return cached
        identity_hash = self._objects.find_identity(identity.stable_key())
        if identity_hash is None:
            return None
        record = self._validate_record(self._objects.read(identity_hash))
        return self._cache_object(identity, record)

    def identity_of(self, ref: TypedRef) -> ObjectIdentity:
        """从分型节点引用恢复完整对象身份并核对 owner、版本和节点。"""
        cached = self._ref_to_identity.get(ref)
        if cached is not None:
            return cached
        record = self._validate_record(
            self._objects.read_by_ref(ref.space_id, ref.local_id))
        identity = ObjectIdentity.from_stable_key(
            self._objects.registry.read_key(
                IDENTITY_GRAPH_OBJECT, record.identity_hash))
        expected = self._typed_ref(identity, record)
        if expected != ref:
            raise GraphObjectIntegrityError("TypedRef 与权威对象身份不一致")
        self._cache_object(identity, record)
        return identity

    def identity_hash_of(self, ref: TypedRef) -> int:
        """返回已完整核验图对象的稳定身份 hash，不把 hash 当作本体。"""
        self.identity_of(ref)
        return self._record_for_ref(ref).identity_hash

    def typed_ref_for_node(self, space_id: int, local_id: int) -> TypedRef:
        """从已物化图节点恢复分型引用；legacy 节点没有映射时 fail closed。"""
        cached = self._refs_by_node.get((space_id, local_id))
        if cached is not None:
            return cached
        record = self._validate_record(
            self._objects.read_by_ref(space_id, local_id))
        identity = ObjectIdentity.from_stable_key(
            self._objects.registry.read_key(
                IDENTITY_GRAPH_OBJECT, record.identity_hash))
        return self._cache_object(identity, record)

    def relate(self, predicate: TypedRef, subject: TypedRef,
               object_ref: TypedRef, *, scope: ScopeIdentity,
               provenance_kind: int, epistemic_origin: int = 0,
               content_version: int = 0,
               qualifiers: tuple[int, ...] = ()) -> GraphStatement:
        """用一等 predicate 概念追加 scoped typed statement。"""
        predicate_identity = self.identity_of(predicate)
        self.identity_of(subject)
        self.identity_of(object_ref)
        if predicate_identity.object_kind != OBJECT_CONCEPT:
            raise ValueError("predicate 必须是通用一等关系概念")
        predicate_record = self._record_for_ref(predicate)
        assertion = AssertionIdentity(
            predicate_record.identity_hash,
            subject,
            object_ref,
            scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        scope_hash = self._scoped_identities.register_scope(scope)
        assertion_hash = self._scoped_identities.register_assertion(
            assertion,
            assertion_role=ASSERTION_ROLE_GRAPH_STATEMENT,
        )
        expected_record = GraphStatementRecord(
            assertion_hash,
            predicate_record.identity_hash,
            predicate.node_ref(),
            (subject.object_kind, subject.space_id, subject.local_id),
            (object_ref.object_kind, object_ref.space_id, object_ref.local_id),
            scope_hash,
        )
        record = self._statements.add(expected_record)
        if record != expected_record:
            raise GraphObjectIntegrityError("statement 写入结果与已知断言不一致")
        return GraphStatement(
            assertion_hash,
            predicate_record.identity_hash,
            predicate,
            subject,
            object_ref,
            scope_hash,
            assertion,
        )

    def clear_runtime_caches(self) -> None:
        """外部 load、迁移或故障注入后清空 facade 运行期核验缓存。"""
        self._identity_to_ref.clear()
        self._ref_to_identity.clear()
        self._records_by_node.clear()
        self._refs_by_node.clear()
        self._source_hashes.clear()
        self._objects.clear_runtime_caches()
        self._statements.clear_runtime_caches()
        self._scoped_identities.clear_runtime_caches()

    def _identity_spec(
            self, identity: ObjectIdentity) -> GraphObjectIdentitySpec:
        """选择可逆物理 codec；依赖身份未登记时退回通用开放编码。"""
        stable_key = identity.stable_key()
        if identity.object_kind in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
            source = SourceRef.from_stable_key(identity.components[:11])
            source_hash = self._registered_source_hash(source)
            if source_hash is None:
                return GraphObjectIdentitySpec.generic(stable_key)
            if identity.object_kind == OBJECT_OCCURRENCE:
                return GraphObjectIdentitySpec.occurrence_source(
                    stable_key,
                    source_hash=source_hash,
                    source_key=source.stable_key(),
                )
            return GraphObjectIdentitySpec.span_source(
                stable_key,
                source_hash=source_hash,
                source_key=source.stable_key(),
            )
        if identity.object_kind == OBJECT_HYPOTHESIS:
            hypothesis = HypothesisKey.from_stable_key(identity.components)
            observation_key = hypothesis.observation.stable_key()
            observation_hash = self._registered_source_hash(
                hypothesis.observation)
            if observation_hash is None:
                return GraphObjectIdentitySpec.generic(stable_key)
            scope_hash = self._scoped_identities.register_scope(
                hypothesis.scope)
            return GraphObjectIdentitySpec.hypothesis(
                stable_key,
                hypothesis_version=identity.components[0],
                hypothesis_kind=hypothesis.hypothesis_kind,
                candidate_key=hypothesis.candidate_key,
                competition_key=hypothesis.competition_key,
                scope_hash=scope_hash,
                scope_key=hypothesis.scope.stable_key(),
                observation_hash=observation_hash,
                observation_key=observation_key,
            )
        return GraphObjectIdentitySpec.generic(stable_key)

    def _registered_source_hash(self, source: SourceRef) -> int | None:
        """缓存已登记 SourceRef 的可核验 hash；缺失结果不缓存以允许后续登记。"""
        cached = self._source_hashes.get(source)
        if cached is not None:
            return cached
        source_hash = self._scoped_identities.registry.find(
            IDENTITY_SOURCE_RECORD, source.stable_key())
        if source_hash is not None:
            self._source_hashes[source] = source_hash
        return source_hash

    def statements(self, *, predicate: TypedRef | None = None,
                   subject: TypedRef | None = None,
                   object_ref: TypedRef | None = None
                   ) -> tuple[GraphStatement, ...]:
        """按任意 predicate/subject/object 组合读取并完整核验 statement。"""
        predicate_hash = None
        if predicate is not None:
            predicate_identity = self.identity_of(predicate)
            if predicate_identity.object_kind != OBJECT_CONCEPT:
                raise ValueError("predicate 必须是通用一等关系概念")
            predicate_hash = self._objects.read_by_ref(
                predicate.space_id, predicate.local_id).identity_hash
        if subject is not None:
            self.identity_of(subject)
        if object_ref is not None:
            self.identity_of(object_ref)
        records = self._statements.query(
            predicate_identity_hash=predicate_hash,
            subject_ref=None if subject is None else subject.node_ref(),
            object_ref=None if object_ref is None else object_ref.node_ref(),
        )
        restored = tuple(self._restore_statement(record) for record in records)
        if subject is not None:
            restored = tuple(item for item in restored if item.subject == subject)
        if object_ref is not None:
            restored = tuple(item for item in restored if item.object == object_ref)
        return restored

    def follow(self, start: TypedRef,
               predicates: tuple[TypedRef, ...]) -> tuple[TypedRef, ...]:
        """沿注入的 predicate 序列逐层遍历，不依赖最小执行指令。"""
        self.identity_of(start)
        frontier: tuple[TypedRef, ...] = (start,)
        for predicate in predicates:
            next_refs: dict[tuple[int, ...], TypedRef] = {}
            for ref in frontier:
                for statement in self.statements(
                        predicate=predicate, subject=ref):
                    next_refs[statement.object.stable_key()] = statement.object
            frontier = tuple(next_refs[key] for key in sorted(next_refs))
            if not frontier:
                break
        return frontier

    def _validate_record(self, record: GraphObjectRecord) -> GraphObjectRecord:
        """核对对象位于当前稳定空间且 concept_node 唯一存在。"""
        if record.space_id != self._space_id:
            raise GraphObjectIntegrityError("对象映射到了其他运行时空间")
        if record.space_identity_key != self._space_identity.stable_key():
            raise GraphObjectIntegrityError("对象稳定空间身份不一致")
        rows = self._backend.select("concept_node", where={
            "space_id": record.space_id,
            "local_id": record.local_id,
        })
        if len(rows) != 1 or rows[0]["type"] != NODE_CONCEPT:
            raise GraphObjectIntegrityError("图对象没有唯一一等概念节点")
        return record

    def _cache_object(self, identity: ObjectIdentity,
                      record: GraphObjectRecord) -> TypedRef:
        """缓存已完整核验的对象身份、分型引用和物化记录。"""
        if identity.object_kind in SEMANTIC_OBJECT_KINDS:
            validate_semantic_identity(identity)
        ref = self._typed_ref(identity, record)
        existing_ref = self._identity_to_ref.get(identity)
        existing_identity = self._ref_to_identity.get(ref)
        if existing_ref is not None and existing_ref != ref:
            raise GraphObjectIntegrityError("同一对象身份缓存到不同节点")
        if existing_identity is not None and existing_identity != identity:
            raise GraphObjectIntegrityError("同一节点缓存到不同对象身份")
        self._identity_to_ref[identity] = ref
        self._ref_to_identity[ref] = identity
        self._records_by_node[ref.node_ref()] = record
        self._refs_by_node[ref.node_ref()] = ref
        return ref

    def _record_for_ref(self, ref: TypedRef) -> GraphObjectRecord:
        """读取已核验对象记录，运行期命中时不重复访问五张身份表。"""
        cached = self._records_by_node.get(ref.node_ref())
        if cached is not None:
            if cached.object_kind != ref.object_kind:
                raise GraphObjectIntegrityError("缓存对象类型与 TypedRef 不一致")
            return cached
        self.identity_of(ref)
        cached = self._records_by_node.get(ref.node_ref())
        if cached is None:
            raise GraphObjectIntegrityError("对象核验后未建立运行期记录缓存")
        return cached

    @staticmethod
    def _typed_ref(identity: ObjectIdentity,
                   record: GraphObjectRecord) -> TypedRef:
        """把稳定对象身份和已核验编址组合为分型节点引用。"""
        if identity.object_kind != record.object_kind:
            raise GraphObjectIntegrityError("对象身份类型与物化记录不一致")
        return TypedRef(
            identity.object_kind,
            record.space_id,
            record.local_id,
            identity.owner,
            identity.versions,
        )

    def _restore_statement(self,
                           record: GraphStatementRecord) -> GraphStatement:
        """联合 statement 行、对象映射和 assertion registry 恢复领域对象。"""
        assertion = self._scoped_identities.load_assertion(
            record.assertion_hash)
        scope = self._scoped_identities.load_scope(record.scope_hash)
        if assertion.scope != scope:
            raise GraphObjectIntegrityError("statement scope 与 assertion 不一致")
        if assertion.relation_kind != record.predicate_identity_hash:
            raise GraphObjectIntegrityError("statement predicate 与 assertion 不一致")
        if record.subject_ref != (
                assertion.subject.object_kind,
                assertion.subject.space_id,
                assertion.subject.local_id):
            raise GraphObjectIntegrityError("statement subject 与 assertion 不一致")
        if record.object_ref != (
                assertion.object.object_kind,
                assertion.object.space_id,
                assertion.object.local_id):
            raise GraphObjectIntegrityError("statement object 与 assertion 不一致")
        predicate_record = self._validate_record(
            self._objects.read(record.predicate_identity_hash))
        if record.predicate_ref != (
                predicate_record.space_id, predicate_record.local_id):
            raise GraphObjectIntegrityError("statement predicate 编址不一致")
        predicate_identity = ObjectIdentity.from_stable_key(
            self._objects.registry.read_key(
                IDENTITY_GRAPH_OBJECT, record.predicate_identity_hash))
        predicate = self._typed_ref(predicate_identity, predicate_record)
        self.identity_of(assertion.subject)
        self.identity_of(assertion.object)
        return GraphStatement(
            record.assertion_hash,
            record.predicate_identity_hash,
            predicate,
            assertion.subject,
            assertion.object,
            record.scope_hash,
            assertion,
        )


__all__ = [
    "GraphOntology",
    "GraphStatement",
    "relation_concept_identity",
]
