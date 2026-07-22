"""S-02 语义候选到 S-00 图拓扑的独立物化与恢复入口。

本模块只保存 compiler 已形成的对象、原子定义和三类开放溯源关系。predicate、
来源元数据和逻辑结构均由调用方注入；图中对象或关系存在不产生 Evidence，也不
表示命题为真。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphStatement,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_HYPOTHESIS,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    TypedRef,
)
from pure_integer_ai.cognition.shared.semantic_graph import (
    MaterializedAtomicProposition,
    SemanticGraph,
    SemanticTopologyError,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBuildResult,
    SemanticPropositionCandidate,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class SemanticBuilderGraphError(RuntimeError):
    """S-02 图协议、批次一致性、溯源拓扑或重放元数据不合法。"""


@dataclass(frozen=True)
class SemanticBuilderTracePredicates:
    """Proposition 到结构、上游候选和 builder 的三个开放 predicate。"""

    proposition_structure: TypedRef
    proposition_upstream: TypedRef
    proposition_builder: TypedRef

    def refs(self) -> tuple[TypedRef, ...]:
        """按冻结协议槽位返回三个 predicate，供统一分型和冲突检查。"""
        return (
            self.proposition_structure,
            self.proposition_upstream,
            self.proposition_builder,
        )


@dataclass(frozen=True)
class MaterializedSemanticCandidate:
    """从图完整恢复的原子定义、结构、上游候选和 builder 溯源。"""

    atomic: MaterializedAtomicProposition
    structure: ObjectIdentity
    upstream_hypothesis: ObjectIdentity
    builder: ObjectIdentity
    structure_ref: TypedRef
    upstream_ref: TypedRef
    builder_ref: TypedRef
    trace_assertion_hashes: tuple[int, int, int]


@dataclass(frozen=True)
class MaterializedSemanticBuild:
    """一次 S-02 批量物化返回的局部对象和语义候选。"""

    objects: tuple[TypedRef, ...]
    candidates: tuple[MaterializedSemanticCandidate, ...]


class SemanticCandidateGraphAdapter:
    """批量预检并物化 S-02 候选，不承担 Evidence 或候选选择职责。"""

    def __init__(
            self, graph: SemanticGraph,
            predicates: SemanticBuilderTracePredicates,
            ) -> None:
        if not isinstance(graph, SemanticGraph):
            raise TypeError("graph 必须是 SemanticGraph")
        if not isinstance(predicates, SemanticBuilderTracePredicates):
            raise TypeError("predicates 必须是 SemanticBuilderTracePredicates")
        self.graph = graph
        self.predicates = predicates
        self.ontology = graph.ontology
        self._validate_predicates()

    def materialize(
            self, result: SemanticBuildResult, *, provenance_kind: int,
            epistemic_origin: int = 0, content_version: int = 0,
            qualifiers: tuple[int, ...] = (),
            ) -> MaterializedSemanticBuild:
        """整批预检后物化对象、原子命题和溯源边，精确重放保持幂等。"""
        if not isinstance(result, SemanticBuildResult):
            raise TypeError("result 必须是 SemanticBuildResult")
        self._validate_metadata(
            result.scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        self._validate_result(result)

        metadata = (
            result.scope,
            provenance_kind,
            epistemic_origin,
            content_version,
            qualifiers,
        )
        for candidate in self._ordered_candidates(result):
            existing = self.graph.preflight_atomic(
                candidate.definition,
                scope=result.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            )
            self._preflight_trace(candidate, existing, metadata)

        object_refs = tuple(
            self.ontology.materialize(item.identity)
            for item in sorted(
                result.objects,
                key=lambda item: item.identity.stable_key())
        )
        materialized: list[MaterializedSemanticCandidate] = []
        for candidate in self._ordered_candidates(result):
            atomic = self.graph.define_atomic(
                candidate.definition,
                scope=result.scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                qualifiers=qualifiers,
            )
            self._write_trace(candidate, atomic.proposition, metadata)
            restored = self.read(atomic.proposition)
            self._require_candidate_trace(candidate, restored)
            materialized.append(restored)
        return MaterializedSemanticBuild(object_refs, tuple(materialized))

    def read(self, proposition: TypedRef) -> MaterializedSemanticCandidate:
        """严格恢复一个命题的三条单值溯源边，并与原子定义元数据交叉核验。"""
        atomic = self.graph.read_atomic(proposition)
        statements = tuple(
            self._single(predicate, proposition, label)
            for predicate, label in zip(
                self.predicates.refs(),
                ("structure", "upstream hypothesis", "builder"),
            )
        )
        self._require_uniform_trace_metadata(statements, atomic)
        identities = tuple(
            self.ontology.identity_of(statement.object)
            for statement in statements
        )
        expected_kinds = (
            OBJECT_STRUCTURE_CONCEPT,
            OBJECT_HYPOTHESIS,
            OBJECT_MINIMAL_INSTRUCTION,
        )
        if tuple(item.object_kind for item in identities) != expected_kinds:
            raise SemanticBuilderGraphError("语义候选溯源端点类型不匹配")
        upstream = identities[1]
        if (upstream.owner != atomic.definition.proposition.owner
                or upstream.versions != atomic.definition.proposition.versions):
            raise SemanticBuilderGraphError(
                "upstream Hypothesis 与 Proposition owner/version 不一致")
        try:
            upstream_key = HypothesisKey.from_stable_key(upstream.components)
        except (TypeError, ValueError) as exc:
            raise SemanticBuilderGraphError(
                "upstream Hypothesis 完整身份无法恢复") from exc
        if upstream_key.object_identity() != upstream:
            raise SemanticBuilderGraphError(
                "upstream Hypothesis 对象身份与完整候选键不一致")
        if (upstream_key.observation != atomic.definition.source
                or upstream_key.scope != atomic.scope):
            raise SemanticBuilderGraphError(
                "upstream Hypothesis 来源或 scope 与 Proposition 不一致")
        return MaterializedSemanticCandidate(
            atomic,
            identities[0],
            upstream,
            identities[2],
            statements[0].object,
            statements[1].object,
            statements[2].object,
            tuple(statement.assertion_hash for statement in statements),
        )

    def read_if_defined(
            self, proposition: ObjectIdentity,
            ) -> MaterializedSemanticCandidate | None:
        """恢复已有 S-02 template；纯 opaque Proposition 不被猜成局部定义。"""
        if not isinstance(proposition, ObjectIdentity):
            raise TypeError("semantic template lookup 必须是 ObjectIdentity")
        if proposition.object_kind != OBJECT_PROPOSITION:
            raise ValueError("semantic template lookup 必须是 Proposition")
        ref = self.ontology.resolve(proposition)
        if ref is None:
            return None
        predicates = self.graph.predicates.refs() + self.predicates.refs()
        topology = tuple(
            statement
            for predicate in predicates
            for statement in self.ontology.statements(
                predicate=predicate,
                subject=ref,
            )
        )
        if not topology:
            return None
        return self.read(ref)

    def lookup(
            self,
            *,
            anchors: tuple[ObjectIdentity, ...] = (),
            fillers: tuple[ObjectIdentity, ...] = (),
            ) -> tuple[MaterializedSemanticCandidate, ...]:
        """合并 anchor 与 filler 反查结果，按完整 Proposition 去重且不选择歧义。"""
        if not isinstance(anchors, tuple) or any(
                not isinstance(item, ObjectIdentity) for item in anchors):
            raise TypeError("semantic lookup anchors 必须是 ObjectIdentity tuple")
        if not isinstance(fillers, tuple) or any(
                not isinstance(item, ObjectIdentity) for item in fillers):
            raise TypeError("semantic lookup fillers 必须是 ObjectIdentity tuple")
        atomics = {}
        for anchor in anchors:
            for atomic in self.graph.lookup_atomic_by_anchor(anchor):
                atomics[atomic.definition.proposition] = atomic
        for filler in fillers:
            for atomic in self.graph.lookup_atomic_by_filler(filler):
                atomics[atomic.definition.proposition] = atomic
        result = []
        for proposition in sorted(atomics, key=ObjectIdentity.stable_key):
            ref = atomics[proposition].proposition
            restored = self.read(ref)
            if restored.atomic != atomics[proposition]:
                raise SemanticBuilderGraphError(
                    "semantic lookup 重复恢复结果不一致")
            result.append(restored)
        return tuple(result)

    def _validate_predicates(self) -> None:
        """核验三个 predicate 为本图内互异 Concept，且不复用 S-00 定义槽。"""
        refs = self.predicates.refs()
        if any(not isinstance(ref, TypedRef) for ref in refs):
            raise TypeError("语义 builder 溯源 predicate 必须全部是 TypedRef")
        all_refs = self.graph.predicates.refs() + refs
        if len({ref.stable_key() for ref in all_refs}) != len(all_refs):
            raise SemanticBuilderGraphError(
                "builder 溯源 predicate 必须互异且不得复用原子命题槽")
        for ref in refs:
            if self.ontology.identity_of(ref).object_kind != OBJECT_CONCEPT:
                raise SemanticBuilderGraphError(
                    "builder 溯源 predicate 必须是 Concept")

    @staticmethod
    def _validate_metadata(
            scope: ScopeIdentity, provenance_kind: int,
            epistemic_origin: int, content_version: int,
            qualifiers: tuple[int, ...],
            ) -> None:
        """校验开放来源元数据，避免首个命题写后才暴露后项参数错误。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if not isinstance(qualifiers, tuple):
            raise TypeError("qualifiers 必须是整数 tuple")
        assert_int(
            provenance_kind,
            epistemic_origin,
            content_version,
            *qualifiers,
            _where="SemanticCandidateGraphAdapter.materialize",
        )
        if type(provenance_kind) is not int or provenance_kind <= 0:
            raise SemanticBuilderGraphError("provenance_kind 必须为严格正整数")
        if type(epistemic_origin) is not int or epistemic_origin < 0:
            raise SemanticBuilderGraphError("epistemic_origin 必须为非负严格整数")
        if type(content_version) is not int or content_version < 0:
            raise SemanticBuilderGraphError("content_version 必须为非负严格整数")
        if any(type(item) is not int for item in qualifiers):
            raise SemanticBuilderGraphError("qualifiers 必须使用严格整数")

    @staticmethod
    def _validate_result(result: SemanticBuildResult) -> None:
        """核验批次来源和 provenance 身份统一，拒绝拼接不同 compiler 结果。"""
        if result.scope.source != result.source:
            raise SemanticBuilderGraphError("构造结果 scope 与 SourceRef 不一致")
        proposition_keys: set[tuple[int, ...]] = set()
        for candidate in result.propositions:
            if candidate.builder != result.builder:
                raise SemanticBuilderGraphError("候选 builder 与批次 builder 不一致")
            if candidate.upstream_hypothesis != result.upstream_hypothesis:
                raise SemanticBuilderGraphError("候选 upstream 与批次 upstream 不一致")
            if candidate.definition.source != result.source:
                raise SemanticBuilderGraphError("候选 Proposition 与批次来源不一致")
            key = candidate.definition.proposition.stable_key()
            if key in proposition_keys:
                raise SemanticBuilderGraphError("同一批次重复 Proposition 身份")
            proposition_keys.add(key)

    @staticmethod
    def _ordered_candidates(
            result: SemanticBuildResult,
            ) -> tuple[SemanticPropositionCandidate, ...]:
        """按完整 Proposition 身份确定批次顺序，不依赖调用方容器顺序。"""
        return tuple(sorted(
            result.propositions,
            key=lambda item: item.definition.proposition.stable_key(),
        ))

    def _expected_trace(
            self, candidate: SemanticPropositionCandidate,
            ) -> tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity]:
        """返回候选声明应保存的结构、上游 Hypothesis 和 builder 完整身份。"""
        return (
            candidate.spec.structure,
            candidate.upstream_hypothesis.object_identity(),
            candidate.builder,
        )

    def _preflight_trace(
            self, candidate: SemanticPropositionCandidate,
            existing: MaterializedAtomicProposition | None,
            metadata: tuple[ScopeIdentity, int, int, int, tuple[int, ...]],
            ) -> None:
        """读取全部溯源槽后裁决新写或完整重放，拒绝部分拓扑和竞争端点。"""
        proposition = self.ontology.resolve(candidate.definition.proposition)
        if proposition is None:
            return
        groups = tuple(
            self.ontology.statements(predicate=predicate, subject=proposition)
            for predicate in self.predicates.refs()
        )
        counts = tuple(len(group) for group in groups)
        if counts == (0, 0, 0):
            if existing is not None:
                raise SemanticBuilderGraphError(
                    "已有原子命题缺少全部 builder 溯源，拒绝部分修补")
            return
        if counts != (1, 1, 1) or existing is None:
            raise SemanticBuilderGraphError(
                "已有 builder 溯源是部分拓扑或存在竞争端点")
        statements = tuple(group[0] for group in groups)
        actual = tuple(
            self.ontology.identity_of(statement.object)
            for statement in statements
        )
        if actual != self._expected_trace(candidate):
            raise SemanticBuilderGraphError("已有 builder 溯源端点与候选不一致")
        if any(self._statement_metadata(item) != metadata for item in statements):
            raise SemanticBuilderGraphError("已有 builder 溯源元数据不一致")

    def _write_trace(
            self, candidate: SemanticPropositionCandidate,
            proposition: TypedRef,
            metadata: tuple[ScopeIdentity, int, int, int, tuple[int, ...]],
            ) -> None:
        """在整批预检通过后幂等追加三条 provenance statement。"""
        scope, provenance, epistemic, content_version, qualifiers = metadata
        for predicate, identity in zip(
                self.predicates.refs(), self._expected_trace(candidate)):
            target = self.ontology.materialize(identity)
            self.ontology.relate(
                predicate,
                proposition,
                target,
                scope=scope,
                provenance_kind=provenance,
                epistemic_origin=epistemic,
                content_version=content_version,
                qualifiers=qualifiers,
            )

    def _single(
            self, predicate: TypedRef, proposition: TypedRef, label: str,
            ) -> GraphStatement:
        """读取恰好一条溯源槽，缺失和多值都不得按顺序私选首行。"""
        statements = self.ontology.statements(
            predicate=predicate, subject=proposition)
        if len(statements) != 1:
            raise SemanticBuilderGraphError(
                f"{label} trace 必须恰有一条，实际 {len(statements)} 条")
        return statements[0]

    @staticmethod
    def _statement_metadata(
            statement: GraphStatement,
            ) -> tuple[ScopeIdentity, int, int, int, tuple[int, ...]]:
        """返回 statement 的完整来源元数据，供定义和溯源交叉核验。"""
        assertion = statement.assertion
        return (
            assertion.scope,
            assertion.provenance_kind,
            assertion.epistemic_origin,
            assertion.content_version,
            assertion.qualifiers,
        )

    def _require_uniform_trace_metadata(
            self, statements: tuple[GraphStatement, ...],
            atomic: MaterializedAtomicProposition,
            ) -> None:
        """要求三条溯源边与原子定义使用完全相同的来源元数据。"""
        expected = (
            atomic.scope,
            atomic.provenance_kind,
            atomic.epistemic_origin,
            atomic.content_version,
            atomic.qualifiers,
        )
        if any(self._statement_metadata(item) != expected
               for item in statements):
            raise SemanticBuilderGraphError(
                "builder 溯源与原子命题定义的来源元数据不一致")

    def _require_candidate_trace(
            self, candidate: SemanticPropositionCandidate,
            restored: MaterializedSemanticCandidate,
            ) -> None:
        """写后核验恢复的定义和三类完整身份都等于 compiler 候选。"""
        if restored.atomic.definition != candidate.definition:
            raise SemanticTopologyError("写后恢复的原子命题定义与候选不一致")
        actual = (
            restored.structure,
            restored.upstream_hypothesis,
            restored.builder,
        )
        if actual != self._expected_trace(candidate):
            raise SemanticBuilderGraphError("写后恢复的 builder 溯源与候选不一致")


__all__ = [
    "MaterializedSemanticBuild",
    "MaterializedSemanticCandidate",
    "SemanticBuilderGraphError",
    "SemanticBuilderTracePredicates",
    "SemanticCandidateGraphAdapter",
]
