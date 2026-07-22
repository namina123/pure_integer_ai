"""S-02 从来源化 Span/Occurrence 编译 typed 语义候选。

compiler 不解析词面、不判断真值，也不直接写图或 Evidence。具体 predicate、Role、
逻辑结构和局部对象均由调用方以一等身份注入；H-00 adapter 只允许登记 unknown
guidance，避免 cue 直接支持命题真值或宣布唯一 parse。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_SET_EXPR,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    PropositionKnowledge,
    context_scope_identity,
    entity_identity,
    event_identity,
    project_proposition_knowledge,
    proposition_hypothesis_key,
    proposition_identity,
    semantic_source,
    set_expr_identity,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
)
from pure_integer_ai.cognition.understanding.span_index import SpanIndex
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_SEMANTIC_BUILD_KEY_VERSION = 1
_BUILDABLE_OBJECT_KINDS = frozenset({
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_SET_EXPR,
})


class SemanticBuildError(ValueError):
    """语义构造计划、来源边界或局部引用不合法。"""


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或 mapper 注入的非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise SemanticBuildError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise SemanticBuildError(f"{where} 必须使用严格整数")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给完整可变长身份段增加长度，避免拼接产生歧义。"""
    return len(key), *key


def _require_identity_kind(
        identity: ObjectIdentity, kind: int, *, label: str,
        ) -> ObjectIdentity:
    """按 object contract 类型核验一等身份，不读取 surface 或局部编号。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != kind:
        raise SemanticBuildError(f"{label} 对象类型不匹配")
    return identity


@dataclass(frozen=True)
class SemanticBuilderProtocol:
    """注入 builder 身份和语义候选的 H-00 kind。"""

    builder: ObjectIdentity
    semantic_hypothesis_kind: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identity_kind(
            self.builder, OBJECT_MINIMAL_INSTRUCTION,
            label="semantic builder")
        _strict_key(
            self.semantic_hypothesis_kind,
            where="semantic_hypothesis_kind")


@dataclass(frozen=True, order=True)
class LocalSemanticRef:
    """引用同一构造计划内的 typed 局部对象或 Proposition。"""

    object_kind: int
    local_key: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_int(self.object_kind, _where="LocalSemanticRef.object_kind")
        if type(self.object_kind) is not int:
            raise SemanticBuildError("LocalSemanticRef.object_kind 必须是严格整数")
        if self.object_kind not in {
                *_BUILDABLE_OBJECT_KINDS, OBJECT_PROPOSITION}:
            raise SemanticBuildError("局部引用对象类型不属于 S-02 构造范围")
        _strict_key(self.local_key, where="LocalSemanticRef.local_key")


@dataclass(frozen=True)
class SemanticObjectSpec:
    """一个 Entity/Event/SetExpr 局部候选槽。"""

    object_kind: int
    local_key: tuple[int, ...]

    def __post_init__(self) -> None:
        assert_int(self.object_kind, _where="SemanticObjectSpec.object_kind")
        if self.object_kind not in _BUILDABLE_OBJECT_KINDS:
            raise SemanticBuildError(
                "SemanticObjectSpec 只构造 Entity/Event/SetExpr")
        _strict_key(self.local_key, where="SemanticObjectSpec.local_key")

    @property
    def local_ref(self) -> LocalSemanticRef:
        """返回可供 Proposition binding 使用的 typed 局部引用。"""
        return LocalSemanticRef(self.object_kind, self.local_key)


@dataclass(frozen=True)
class SemanticFillerSpec:
    """一个 Role filler，必须在局部引用和外部权威身份中二选一。"""

    local_ref: LocalSemanticRef | None = None
    external: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if (self.local_ref is None) == (self.external is None):
            raise SemanticBuildError(
                "SemanticFillerSpec 必须且只能使用一种 filler")
        if self.local_ref is not None and not isinstance(
                self.local_ref, LocalSemanticRef):
            raise TypeError("local_ref 必须是 LocalSemanticRef")
        if self.external is not None and not isinstance(
                self.external, ObjectIdentity):
            raise TypeError("external filler 必须是 ObjectIdentity")


@dataclass(frozen=True)
class SemanticBindingSpec:
    """语义计划中的开放 Role、filler 和同 Role ordinal。"""

    role: ObjectIdentity
    filler: SemanticFillerSpec
    ordinal: int = 0

    def __post_init__(self) -> None:
        _require_identity_kind(self.role, OBJECT_ROLE, label="semantic role")
        if not isinstance(self.filler, SemanticFillerSpec):
            raise TypeError("filler 必须是 SemanticFillerSpec")
        assert_int(self.ordinal, _where="SemanticBindingSpec.ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise SemanticBuildError("binding ordinal 必须为非负严格整数")


@dataclass(frozen=True)
class SemanticPropositionSpec:
    """一个原子或嵌套 Proposition 候选的开放构造说明。"""

    local_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    predicate: ObjectIdentity
    structure: ObjectIdentity
    bindings: tuple[SemanticBindingSpec, ...]
    source_anchor: TypedRef | None = None

    def __post_init__(self) -> None:
        _strict_key(self.local_key, where="SemanticPropositionSpec.local_key")
        _strict_key(
            self.competition_key,
            where="SemanticPropositionSpec.competition_key")
        _require_identity_kind(
            self.predicate, OBJECT_CONCEPT, label="semantic predicate")
        _require_identity_kind(
            self.structure, OBJECT_STRUCTURE_CONCEPT,
            label="semantic structure")
        if not isinstance(self.bindings, tuple):
            raise TypeError("bindings 必须是 SemanticBindingSpec tuple")
        if any(not isinstance(item, SemanticBindingSpec)
               for item in self.bindings):
            raise TypeError("bindings 只能包含 SemanticBindingSpec")
        if self.source_anchor is not None and not isinstance(
                self.source_anchor, TypedRef):
            raise TypeError("source_anchor 必须是 TypedRef 或 None")

    @property
    def local_ref(self) -> LocalSemanticRef:
        """返回可供其他 Proposition 嵌套引用的局部 Proposition。"""
        return LocalSemanticRef(OBJECT_PROPOSITION, self.local_key)


@dataclass(frozen=True)
class SemanticBuildPlan:
    """一个 upstream parse 下可并存的完整语义候选计划。"""

    upstream_hypothesis: HypothesisKey
    context_key: tuple[int, ...]
    objects: tuple[SemanticObjectSpec, ...]
    propositions: tuple[SemanticPropositionSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_hypothesis, HypothesisKey):
            raise TypeError("upstream_hypothesis 必须是 HypothesisKey")
        _strict_key(self.context_key, where="SemanticBuildPlan.context_key")
        if not isinstance(self.objects, tuple):
            raise TypeError("objects 必须是 SemanticObjectSpec tuple")
        if any(not isinstance(item, SemanticObjectSpec)
               for item in self.objects):
            raise TypeError("objects 只能包含 SemanticObjectSpec")
        if not isinstance(self.propositions, tuple) or not self.propositions:
            raise SemanticBuildError("构造计划至少需要一个 Proposition")
        if any(not isinstance(item, SemanticPropositionSpec)
               for item in self.propositions):
            raise TypeError("propositions 只能包含 SemanticPropositionSpec")
        refs = tuple(item.local_ref for item in self.objects) + tuple(
            item.local_ref for item in self.propositions)
        if len(set(refs)) != len(refs):
            raise SemanticBuildError("构造计划不得重复 typed local ref")


@dataclass(frozen=True)
class BuiltSemanticObject:
    """局部对象说明与来源化完整身份的编译结果。"""

    spec: SemanticObjectSpec
    identity: ObjectIdentity

    def __post_init__(self) -> None:
        """核验局部 spec 与来源化对象身份类型一致。"""
        if not isinstance(self.spec, SemanticObjectSpec):
            raise TypeError("spec 必须是 SemanticObjectSpec")
        if not isinstance(self.identity, ObjectIdentity):
            raise TypeError("identity 必须是 ObjectIdentity")
        if self.identity.object_kind != self.spec.object_kind:
            raise SemanticBuildError("局部对象 spec 与 identity 类型不一致")
        semantic_source(self.identity)


@dataclass(frozen=True)
class SemanticPropositionCandidate:
    """一个尚未判真的 Proposition 定义、结构和 H-00 候选。"""

    spec: SemanticPropositionSpec
    definition: AtomicPropositionDefinition
    hypothesis: HypothesisKey
    builder: ObjectIdentity
    upstream_hypothesis: HypothesisKey

    def __post_init__(self) -> None:
        """核验 spec、原子定义、H-00 候选和 provenance 身份没有被拼接。"""
        if not isinstance(self.spec, SemanticPropositionSpec):
            raise TypeError("spec 必须是 SemanticPropositionSpec")
        if not isinstance(self.definition, AtomicPropositionDefinition):
            raise TypeError("definition 必须是 AtomicPropositionDefinition")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("hypothesis 必须是 HypothesisKey")
        _require_identity_kind(
            self.builder, OBJECT_MINIMAL_INSTRUCTION,
            label="candidate builder")
        if not isinstance(self.upstream_hypothesis, HypothesisKey):
            raise TypeError("upstream_hypothesis 必须是 HypothesisKey")
        if self.definition.predicate != self.spec.predicate:
            raise SemanticBuildError("Proposition spec 与定义 predicate 不一致")
        if self.hypothesis.candidate_key != (
                self.definition.proposition.stable_key()):
            raise SemanticBuildError("Hypothesis 未绑定完整 Proposition 身份")
        if self.hypothesis.competition_key != self.spec.competition_key:
            raise SemanticBuildError("Hypothesis 与 spec competition 不一致")
        if (self.hypothesis.observation != self.definition.source
                or self.hypothesis.scope != self.upstream_hypothesis.scope
                or self.upstream_hypothesis.observation
                != self.definition.source):
            raise SemanticBuildError("候选、upstream 与 Proposition 来源不一致")

    def shape_key(self) -> tuple[int, ...]:
        """返回排除内容 identity、保留结构/Role/filler 类型的逻辑形状。"""
        structure = self.spec.structure.stable_key()
        bindings = tuple(sorted((
            binding.role.stable_key(),
            binding.filler.object_kind,
            binding.ordinal,
        ) for binding in self.definition.bindings))
        result: list[int] = [len(structure), *structure, len(bindings)]
        for role, object_kind, ordinal in bindings:
            result.extend((len(role), *role, object_kind, ordinal))
        return tuple(result)


@dataclass(frozen=True)
class SemanticBuildResult:
    """一次纯编译的来源、anchor、局部对象和 Proposition 候选集合。"""

    source: SourceRef
    scope: ScopeIdentity
    root_anchor: ObjectIdentity
    builder: ObjectIdentity
    upstream_hypothesis: HypothesisKey
    context: ObjectIdentity
    objects: tuple[BuiltSemanticObject, ...]
    propositions: tuple[SemanticPropositionCandidate, ...]

    def __post_init__(self) -> None:
        """核验纯编译批次的来源、scope、builder、context 和成员统一。"""
        if not isinstance(self.source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if self.scope.source != self.source:
            raise SemanticBuildError("构造结果 scope 与 source 不一致")
        if not isinstance(self.root_anchor, ObjectIdentity) or (
                self.root_anchor.object_kind not in {
                    OBJECT_SPAN, OBJECT_OCCURRENCE}):
            raise SemanticBuildError("root_anchor 必须是 Span/Occurrence 身份")
        _require_identity_kind(
            self.builder, OBJECT_MINIMAL_INSTRUCTION,
            label="result builder")
        if not isinstance(self.upstream_hypothesis, HypothesisKey):
            raise TypeError("upstream_hypothesis 必须是 HypothesisKey")
        if (self.upstream_hypothesis.observation != self.source
                or self.upstream_hypothesis.scope != self.scope):
            raise SemanticBuildError("result upstream 与来源或 scope 不一致")
        _require_identity_kind(
            self.context, OBJECT_CONTEXT_SCOPE, label="semantic context")
        if semantic_source(self.context) != self.source:
            raise SemanticBuildError("semantic context 与构造来源不一致")
        if not isinstance(self.objects, tuple) or any(
                not isinstance(item, BuiltSemanticObject)
                for item in self.objects):
            raise TypeError("objects 必须是 BuiltSemanticObject tuple")
        if not isinstance(self.propositions, tuple) or not self.propositions:
            raise SemanticBuildError("结果必须包含 Proposition 候选")
        if any(not isinstance(item, SemanticPropositionCandidate)
               for item in self.propositions):
            raise TypeError("propositions 必须是 SemanticPropositionCandidate tuple")
        for candidate in self.propositions:
            if (candidate.builder != self.builder
                    or candidate.upstream_hypothesis
                    != self.upstream_hypothesis
                    or candidate.definition.source != self.source):
                raise SemanticBuildError("结果混入其他 builder/upstream/source 候选")


@dataclass(frozen=True)
class SemanticGuidanceEvidence:
    """只允许作为 unknown guidance 追加到语义候选的外部事件说明。"""

    hypothesis: HypothesisKey
    evidence_id: int
    reason_key: tuple[int, ...]
    source: SourceRef
    timestamp_seq: int
    payload: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("guidance hypothesis 必须是 HypothesisKey")
        if not isinstance(self.payload, tuple) or any(
                type(item) is not int for item in self.payload):
            raise SemanticBuildError("guidance payload 必须是严格整数 tuple")
        assert_int(
            self.evidence_id,
            self.timestamp_seq,
            *self.payload,
            _where="SemanticGuidanceEvidence",
        )
        if type(self.evidence_id) is not int or self.evidence_id <= 0:
            raise SemanticBuildError("guidance evidence_id 必须为严格正整数")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise SemanticBuildError("guidance timestamp 必须为非负严格整数")
        _strict_key(self.reason_key, where="guidance reason_key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("guidance source 必须是 SourceRef")

    def evidence_record(self) -> EvidenceRecord:
        """固定生成 EVIDENCE_UNKNOWN，调用方不能借此注入 support/refute。"""
        return EvidenceRecord(
            self.evidence_id,
            self.hypothesis,
            EVIDENCE_UNKNOWN,
            self.reason_key,
            self.source,
            self.timestamp_seq,
            self.payload,
        )


class SemanticCandidateBuilder:
    """只读 Span/Occurrence 真源并纯编译注入语义计划。"""

    def __init__(
            self, spans: SpanIndex, protocol: SemanticBuilderProtocol,
            occurrence_index: OccurrenceIndex | None = None,
            ) -> None:
        if not isinstance(spans, SpanIndex):
            raise TypeError("spans 必须是 SpanIndex")
        if not isinstance(protocol, SemanticBuilderProtocol):
            raise TypeError("protocol 必须是 SemanticBuilderProtocol")
        if occurrence_index is not None and not isinstance(
                occurrence_index, OccurrenceIndex):
            raise TypeError("occurrence_index 必须是 OccurrenceIndex 或 None")
        if (occurrence_index is not None
                and occurrence_index.ontology is not spans.ontology):
            raise SemanticBuildError("Span 与 Occurrence 必须属于同一 ontology")
        self.spans = spans
        self.protocol = protocol
        self.occurrence_index = occurrence_index
        self.ontology = spans.ontology

    def compile(
            self, root_anchor: TypedRef, plan: SemanticBuildPlan,
            ) -> SemanticBuildResult:
        """核验来源与局部引用后，确定性生成 typed 对象和命题候选。"""
        if not isinstance(root_anchor, TypedRef):
            raise TypeError("root_anchor 必须是 TypedRef")
        if not isinstance(plan, SemanticBuildPlan):
            raise TypeError("plan 必须是 SemanticBuildPlan")
        anchor_cache: dict[
            TypedRef, tuple[SourceRef, ScopeIdentity, ObjectIdentity]
        ] = {}

        def read_anchor(
                anchor: TypedRef,
                ) -> tuple[SourceRef, ScopeIdentity, ObjectIdentity]:
            """在本次纯编译内只恢复一次同一 anchor，并复用不可变结果。"""
            cached = anchor_cache.get(anchor)
            if cached is None:
                cached = self._anchor(anchor)
                anchor_cache[anchor] = cached
            return cached

        source, scope, root_identity = read_anchor(root_anchor)
        upstream = plan.upstream_hypothesis
        if upstream.observation != source or upstream.scope != scope:
            raise SemanticBuildError(
                "upstream Hypothesis 与 root anchor 来源或 scope 不一致")

        context = context_scope_identity(
            source,
            self._declaration_key(
                upstream, root_identity, OBJECT_CONTEXT_SCOPE,
                plan.context_key),
        )
        identities: dict[LocalSemanticRef, ObjectIdentity] = {}
        built_objects: list[BuiltSemanticObject] = []
        for spec in sorted(
                plan.objects,
                key=lambda item: (item.object_kind, item.local_key)):
            identity = self._build_object(
                source, upstream, root_identity, spec)
            identities[spec.local_ref] = identity
            built_objects.append(BuiltSemanticObject(spec, identity))

        proposition_identities: dict[LocalSemanticRef, ObjectIdentity] = {}
        proposition_anchors: dict[LocalSemanticRef, ObjectIdentity] = {}
        for spec in sorted(plan.propositions, key=lambda item: item.local_key):
            anchor_ref = root_anchor if spec.source_anchor is None else spec.source_anchor
            anchor_source, anchor_scope, anchor_identity = read_anchor(anchor_ref)
            if anchor_source != source or anchor_scope != scope:
                raise SemanticBuildError(
                    "Proposition anchor 与 root anchor 来源或 scope 不一致")
            proposition_anchors[spec.local_ref] = anchor_identity
            proposition_identities[spec.local_ref] = proposition_identity(
                source,
                self._proposition_key(
                    upstream, anchor_identity, spec),
            )
        identities.update(proposition_identities)

        candidates: list[SemanticPropositionCandidate] = []
        for spec in sorted(plan.propositions, key=lambda item: item.local_key):
            anchor_identity = proposition_anchors[spec.local_ref]
            bindings = tuple(
                AtomicRoleBinding(
                    binding.role,
                    self._resolve_filler(binding.filler, identities),
                    binding.ordinal,
                )
                for binding in spec.bindings
            )
            definition = AtomicPropositionDefinition(
                proposition_identities[spec.local_ref],
                spec.predicate,
                anchor_identity,
                context,
                bindings,
            )
            hypothesis = proposition_hypothesis_key(
                definition.proposition,
                hypothesis_kind=self.protocol.semantic_hypothesis_kind,
                competition_key=spec.competition_key,
                scope=scope,
            )
            candidates.append(SemanticPropositionCandidate(
                spec,
                definition,
                hypothesis,
                self.protocol.builder,
                upstream,
            ))
        return SemanticBuildResult(
            source,
            scope,
            root_identity,
            self.protocol.builder,
            upstream,
            context,
            tuple(built_objects),
            tuple(candidates),
        )

    def _anchor(
            self, anchor: TypedRef,
            ) -> tuple[SourceRef, ScopeIdentity, ObjectIdentity]:
        """从唯一 Span/Occurrence reader 恢复来源、scope 和完整对象身份。"""
        if anchor.object_kind == OBJECT_SPAN:
            record = self.spans.read(anchor)
            return record.source, record.scope, self.ontology.identity_of(anchor)
        if anchor.object_kind == OBJECT_OCCURRENCE:
            if self.occurrence_index is None:
                raise SemanticBuildError("Occurrence anchor 需要 OccurrenceIndex")
            record = self.occurrence_index.read(anchor)
            return record.source, record.scope, self.ontology.identity_of(anchor)
        raise SemanticBuildError("semantic anchor 必须是 Span 或 Occurrence")

    def _build_object(
            self, source: SourceRef, upstream: HypothesisKey,
            anchor: ObjectIdentity, spec: SemanticObjectSpec,
            ) -> ObjectIdentity:
        """按 spec 类型调用 S-00 来源化对象 identity codec。"""
        key = self._declaration_key(
            upstream, anchor, spec.object_kind, spec.local_key)
        if spec.object_kind == OBJECT_ENTITY:
            return entity_identity(source, key)
        if spec.object_kind == OBJECT_EVENT:
            return event_identity(source, key)
        if spec.object_kind == OBJECT_SET_EXPR:
            return set_expr_identity(source, key)
        raise SemanticBuildError("未注册的 S-02 object kind")

    def _declaration_key(
            self, upstream: HypothesisKey, anchor: ObjectIdentity,
            object_kind: int, local_key: tuple[int, ...],
            ) -> tuple[int, ...]:
        """完整编码 builder、upstream、anchor、类型和局部键。"""
        builder_key = self.protocol.builder.stable_key()
        upstream_key = upstream.stable_key()
        anchor_key = anchor.stable_key()
        return (
            _SEMANTIC_BUILD_KEY_VERSION,
            *_packed(builder_key),
            *_packed(upstream_key),
            *_packed(anchor_key),
            object_kind,
            *_packed(local_key),
        )

    def _proposition_key(
            self, upstream: HypothesisKey, anchor: ObjectIdentity,
            spec: SemanticPropositionSpec,
            ) -> tuple[int, ...]:
        """在通用声明键后保存结构和 predicate，隔离同槽不同语义候选。"""
        base = self._declaration_key(
            upstream, anchor, OBJECT_PROPOSITION, spec.local_key)
        return (
            *base,
            *_packed(spec.structure.stable_key()),
            *_packed(spec.predicate.stable_key()),
        )

    @staticmethod
    def _resolve_filler(
            filler: SemanticFillerSpec,
            identities: dict[LocalSemanticRef, ObjectIdentity],
            ) -> ObjectIdentity:
        """解析局部或外部 filler，未声明局部引用必须失败。"""
        if filler.external is not None:
            return filler.external
        assert filler.local_ref is not None
        resolved = identities.get(filler.local_ref)
        if resolved is None:
            raise SemanticBuildError("binding 引用了未声明的 typed local ref")
        return resolved


class SemanticCandidateLedgerAdapter:
    """批量登记语义候选，并且只追加 unknown guidance。"""

    def __init__(self, ledger: HypothesisLedger) -> None:
        if not isinstance(ledger, HypothesisLedger):
            raise TypeError("ledger 必须是 HypothesisLedger")
        self.ledger = ledger

    def register_unknown(
            self, result: SemanticBuildResult,
            guidance: tuple[SemanticGuidanceEvidence, ...] = (),
            ) -> tuple[PropositionKnowledge, ...]:
        """先在 clone 完整预检，再登记候选和 EVIDENCE_UNKNOWN。"""
        if not isinstance(result, SemanticBuildResult):
            raise TypeError("result 必须是 SemanticBuildResult")
        if not isinstance(guidance, tuple) or any(
                not isinstance(item, SemanticGuidanceEvidence)
                for item in guidance):
            raise TypeError("guidance 必须是 SemanticGuidanceEvidence tuple")
        candidates = {
            item.hypothesis: item for item in result.propositions}
        if any(item.hypothesis not in candidates for item in guidance):
            raise SemanticBuildError("guidance 指向 result 外的 Hypothesis")

        probe = self.ledger.clone()
        self._apply(probe, result, guidance)
        self._apply(self.ledger, result, guidance)
        return tuple(
            project_proposition_knowledge(
                candidate.definition, candidate.hypothesis, self.ledger)
            for candidate in result.propositions
        )

    @staticmethod
    def _apply(
            ledger: HypothesisLedger, result: SemanticBuildResult,
            guidance: tuple[SemanticGuidanceEvidence, ...],
            ) -> None:
        """按完整稳定键登记候选后追加已冻结的 unknown 事件。"""
        for candidate in sorted(
                result.propositions,
                key=lambda item: item.hypothesis.stable_key()):
            ledger.register(candidate.hypothesis)
        for item in sorted(
                guidance,
                key=lambda evidence: evidence.evidence_record().stable_key()):
            ledger.append_evidence(item.evidence_record())


__all__ = [
    "BuiltSemanticObject",
    "LocalSemanticRef",
    "SemanticBindingSpec",
    "SemanticBuildError",
    "SemanticBuildPlan",
    "SemanticBuildResult",
    "SemanticBuilderProtocol",
    "SemanticCandidateBuilder",
    "SemanticCandidateLedgerAdapter",
    "SemanticFillerSpec",
    "SemanticGuidanceEvidence",
    "SemanticObjectSpec",
    "SemanticPropositionCandidate",
    "SemanticPropositionSpec",
]
