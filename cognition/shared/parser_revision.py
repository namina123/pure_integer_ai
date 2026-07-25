"""A-03 跨 ParserVersion 的来源化重解析映射和图投影。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactSchema,
    artifact_identity,
    describe_artifact_identity,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ARTIFACT,
    OBJECT_CONCEPT,
    OBJECT_HYPOTHESIS,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    validate_semantic_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_REVISION_VERSION = 1
_REVISION_DOMAIN = "parser_revision.definition.v1"
_ANCHOR_MAPPING_DOMAIN = "parser_revision.anchor_mapping.v1"
_HYPOTHESIS_MAPPING_DOMAIN = "parser_revision.hypothesis_mapping.v1"
_SOURCE_KEY_SIZE = 11
_ANCHOR_KINDS = frozenset({OBJECT_OCCURRENCE, OBJECT_SPAN})


class ParserRevisionError(ValueError):
    """ParserRevision 身份、来源、映射或图拓扑违反契约时抛出。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验开放键或 trace 是非空严格整数元组。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_positive(value: int, *, where: str) -> int:
    """校验图元数据和预算是严格正整数。"""
    assert_int(value, _where=where)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{where} 必须是严格正整数")
    return value


def parser_lineage_key(source: SourceRef) -> tuple[int, ...]:
    """返回排除 ParserVersion、保留其他来源和版本坐标的 lineage 键。"""
    if not isinstance(source, SourceRef):
        raise TypeError("parser lineage 需要 SourceRef")
    key = source.stable_key()
    return (*key[:8], *key[9:])


def _anchor_source(identity: ObjectIdentity) -> SourceRef:
    """从 Occurrence/Span 完整身份恢复并核验 SourceRef 前缀。"""
    if (not isinstance(identity, ObjectIdentity)
            or identity.object_kind not in _ANCHOR_KINDS):
        raise TypeError("parser revision anchor 必须是 Occurrence 或 Span 身份")
    if len(identity.components) <= _SOURCE_KEY_SIZE:
        raise ParserRevisionError("parser revision anchor 身份被截断")
    source = SourceRef.from_stable_key(
        identity.components[:_SOURCE_KEY_SIZE])
    if source.owner != identity.owner or source.versions != identity.versions:
        raise ParserRevisionError("parser revision anchor 与 SourceRef 不一致")
    return source


def _require_relation(identity: ObjectIdentity, *, label: str) -> None:
    """核验 revision predicate 是有效的一等 Concept。"""
    if (not isinstance(identity, ObjectIdentity)
            or identity.object_kind != OBJECT_CONCEPT):
        raise TypeError(f"{label} 必须是一等 Concept")
    validate_semantic_identity(identity)


@dataclass(frozen=True)
class ParserRevisionProtocol:
    """注入 revision/mapping Artifact 类型、图关系、元数据和硬预算。"""

    revision_kind: ObjectIdentity
    anchor_mapping_kind: ObjectIdentity
    hypothesis_mapping_kind: ObjectIdentity
    schema: ArtifactSchema
    revision_anchor_mapping: ObjectIdentity
    revision_hypothesis_mapping: ObjectIdentity
    mapping_old: ObjectIdentity
    mapping_new: ObjectIdentity
    revision_dimension: ObjectIdentity
    revision_reason: ObjectIdentity
    provenance_kind: int
    epistemic_origin: int
    content_version: int
    qualifiers: tuple[int, ...]
    max_anchor_mappings: int
    max_hypothesis_mappings: int
    max_targets_per_mapping: int

    def __post_init__(self) -> None:
        for label, identity in (
                ("revision kind", self.revision_kind),
                ("anchor mapping kind", self.anchor_mapping_kind),
                ("hypothesis mapping kind", self.hypothesis_mapping_kind)):
            if not isinstance(identity, ObjectIdentity):
                raise TypeError(f"{label} 必须是 ObjectIdentity")
            validate_semantic_identity(identity)
        if len({
                self.revision_kind,
                self.anchor_mapping_kind,
                self.hypothesis_mapping_kind}) != 3:
            raise ValueError("revision 和两类 mapping 必须使用不同 Artifact kind")
        if not isinstance(self.schema, ArtifactSchema):
            raise TypeError("parser revision schema 类型错误")
        relations = (
            self.revision_anchor_mapping,
            self.revision_hypothesis_mapping,
            self.mapping_old,
            self.mapping_new,
            self.revision_dimension,
            self.revision_reason,
        )
        for index, relation in enumerate(relations):
            _require_relation(relation, label=f"parser revision relation[{index}]")
        if len(set(relations)) != len(relations):
            raise ValueError("parser revision 图关系必须互异")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="ParserRevisionProtocol.metadata",
        )
        if type(self.provenance_kind) is not int or self.provenance_kind <= 0:
            raise ValueError("parser revision provenance_kind 必须为严格正整数")
        if (type(self.epistemic_origin) is not int
                or self.epistemic_origin < 0
                or type(self.content_version) is not int
                or self.content_version < 0
                or any(type(item) is not int for item in self.qualifiers)):
            raise ValueError("parser revision 图元数据必须是非负严格整数")
        _strict_positive(
            self.max_anchor_mappings,
            where="parser revision max_anchor_mappings",
        )
        _strict_positive(
            self.max_hypothesis_mappings,
            where="parser revision max_hypothesis_mappings",
        )
        _strict_positive(
            self.max_targets_per_mapping,
            where="parser revision max_targets_per_mapping",
        )

    def relations(self) -> tuple[ObjectIdentity, ...]:
        """返回协议拥有的六个互异图关系。"""
        return (
            self.revision_anchor_mapping,
            self.revision_hypothesis_mapping,
            self.mapping_old,
            self.mapping_new,
            self.revision_dimension,
            self.revision_reason,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 Artifact 分类、图关系、元数据和预算的完整键。"""
        result = [_REVISION_VERSION]
        for identity in (
                self.revision_kind,
                self.anchor_mapping_kind,
                self.hypothesis_mapping_kind):
            result.extend(_packed(identity.stable_key()))
        result.extend(_packed(self.schema.stable_key()))
        for relation in self.relations():
            result.extend(_packed(relation.stable_key()))
        result.extend((
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            len(self.qualifiers),
            *self.qualifiers,
            self.max_anchor_mappings,
            self.max_hypothesis_mappings,
            self.max_targets_per_mapping,
        ))
        return tuple(result)


@dataclass(frozen=True)
class ParserAnchorRevision:
    """一个旧 anchor 到零个或多个新 anchor 的删除/拆分/合并映射。"""

    old: ObjectIdentity
    replacements: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _anchor_source(self.old)
        if (not isinstance(self.replacements, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.replacements)):
            raise TypeError("parser anchor replacements 必须是 ObjectIdentity tuple")
        for item in self.replacements:
            _anchor_source(item)
        if len(set(self.replacements)) != len(self.replacements):
            raise ValueError("parser anchor replacements 不得重复")
        object.__setattr__(self, "replacements", tuple(sorted(
            self.replacements, key=ObjectIdentity.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回旧 anchor 和全部新 anchor 的完整键。"""
        result = [
            _REVISION_VERSION,
            *_packed(self.old.stable_key()),
            len(self.replacements),
        ]
        for item in self.replacements:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ParserHypothesisRevision:
    """一个旧 Hypothesis、跨版本新候选及使旧候选退出的 refute Evidence。"""

    old: HypothesisKey
    replacements: tuple[HypothesisKey, ...]
    refute: EvidenceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.old, HypothesisKey):
            raise TypeError("parser hypothesis old 类型错误")
        if (not isinstance(self.replacements, tuple)
                or any(not isinstance(item, HypothesisKey)
                       for item in self.replacements)):
            raise TypeError("parser hypothesis replacements 类型错误")
        if len(set(self.replacements)) != len(self.replacements):
            raise ValueError("parser hypothesis replacements 不得重复")
        if (not isinstance(self.refute, EvidenceRecord)
                or self.refute.hypothesis != self.old
                or self.refute.stance != EVIDENCE_REFUTE):
            raise ValueError("parser hypothesis revision 必须携带指向 old 的 refute")
        object.__setattr__(self, "replacements", tuple(sorted(
            self.replacements, key=HypothesisKey.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回旧/新候选和定向 refute 的完整键。"""
        result = [
            _REVISION_VERSION,
            *_packed(self.old.stable_key()),
            len(self.replacements),
        ]
        for item in self.replacements:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.refute.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class ParserRevisionRequest:
    """同一原文从旧 ParserVersion 到新版本的完整局部 revision 声明。"""

    old_source: SourceRef
    new_source: SourceRef
    old_scope: ScopeIdentity
    new_scope: ScopeIdentity
    revision_key: tuple[int, ...]
    anchors: tuple[ParserAnchorRevision, ...]
    hypotheses: tuple[ParserHypothesisRevision, ...]
    dimensions: tuple[ObjectIdentity, ...]
    reason: ObjectIdentity
    timestamp_seq: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.old_source, SourceRef)
                or not isinstance(self.new_source, SourceRef)):
            raise TypeError("parser revision source 必须是 SourceRef")
        if parser_lineage_key(self.old_source) != parser_lineage_key(
                self.new_source):
            raise ParserRevisionError("parser revision 新旧来源 lineage 不一致")
        if self.new_source.versions.parser.value <= (
                self.old_source.versions.parser.value):
            raise ParserRevisionError("新 ParserVersion 必须严格高于旧版")
        for scope, source, label in (
                (self.old_scope, self.old_source, "old"),
                (self.new_scope, self.new_source, "new")):
            if (not isinstance(scope, ScopeIdentity)
                    or scope.scope_kind != SCOPE_DOCUMENT
                    or scope.source != source):
                raise ParserRevisionError(
                    f"parser revision {label} scope 必须是对应 document")
        _strict_key(self.revision_key, where="parser revision key")
        if (not isinstance(self.anchors, tuple)
                or any(not isinstance(item, ParserAnchorRevision)
                       for item in self.anchors)):
            raise TypeError("parser revision anchors 类型错误")
        if (not isinstance(self.hypotheses, tuple) or not self.hypotheses
                or any(not isinstance(item, ParserHypothesisRevision)
                       for item in self.hypotheses)):
            raise TypeError("parser revision hypotheses 必须含 revision 项")
        if len({item.old for item in self.anchors}) != len(self.anchors):
            raise ValueError("parser revision old anchor 不得重复")
        if len({item.old for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("parser revision old Hypothesis 不得重复")
        for item in self.anchors:
            if _anchor_source(item.old) != self.old_source:
                raise ParserRevisionError("old anchor 不属于旧 ParserVersion")
            if any(_anchor_source(value) != self.new_source
                   for value in item.replacements):
                raise ParserRevisionError("new anchor 不属于新 ParserVersion")
        for item in self.hypotheses:
            if (item.old.observation != self.old_source
                    or item.old.scope != self.old_scope):
                raise ParserRevisionError("old Hypothesis 不属于旧 parser scope")
            if any(
                    value.observation != self.new_source
                    or value.scope != self.new_scope
                    for value in item.replacements):
                raise ParserRevisionError("replacement Hypothesis 不属于新 parser scope")
            if item.refute.source != self.new_source:
                raise ParserRevisionError("旧候选 refute 必须来自新 ParserVersion")
        if (not isinstance(self.dimensions, tuple) or not self.dimensions
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.dimensions)):
            raise TypeError("parser revision dimensions 必须是非空一等对象 tuple")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("parser revision dimensions 不得重复")
        if (not isinstance(self.reason, ObjectIdentity)
                or self.reason.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("parser revision reason 必须是 MinimalInstruction")
        assert_int(self.timestamp_seq, _where="parser revision timestamp_seq")
        if type(self.timestamp_seq) is not int or self.timestamp_seq < 0:
            raise ValueError("parser revision timestamp_seq 必须是非负严格整数")
        _strict_key(self.trace, where="parser revision trace")
        object.__setattr__(self, "anchors", tuple(sorted(
            self.anchors, key=lambda item: item.old.stable_key())))
        object.__setattr__(self, "hypotheses", tuple(sorted(
            self.hypotheses, key=lambda item: item.old.stable_key())))
        object.__setattr__(self, "dimensions", tuple(sorted(
            self.dimensions, key=ObjectIdentity.stable_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、scope、影响集、理由、逻辑序和 trace 的完整键。"""
        result = [
            _REVISION_VERSION,
            *_packed(self.old_source.stable_key()),
            *_packed(self.new_source.stable_key()),
            *_packed(self.old_scope.stable_key()),
            *_packed(self.new_scope.stable_key()),
            *_packed(self.revision_key),
            len(self.anchors),
        ]
        for item in self.anchors:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.hypotheses))
        for item in self.hypotheses:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.dimensions))
        for item in self.dimensions:
            result.extend(_packed(item.stable_key()))
        result.extend((
            *_packed(self.reason.stable_key()),
            self.timestamp_seq,
            *_packed(self.trace),
        ))
        return tuple(result)

    def revision_identity(
            self, protocol: ParserRevisionProtocol,
            ) -> ObjectIdentity:
        """构造由新来源拥有、固定内容引用绑定完整 request 的 revision Artifact。"""
        if not isinstance(protocol, ParserRevisionProtocol):
            raise TypeError("revision_identity 需要 ParserRevisionProtocol")
        payload = integer_tuple_fingerprint(
            self.stable_key(), domain=_REVISION_DOMAIN)
        return artifact_identity(
            self.new_source,
            protocol.revision_kind,
            protocol.schema,
            self.revision_key,
            payload,
            self.new_scope,
        )


@dataclass(frozen=True)
class MaterializedParserRevision:
    """一次 ParserRevision 的一等 Artifact、mapping Artifact 和 statement 集。"""

    revision: ObjectIdentity
    anchor_mappings: tuple[ObjectIdentity, ...]
    hypothesis_mappings: tuple[ObjectIdentity, ...]
    assertion_hashes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.revision, ObjectIdentity)
                or self.revision.object_kind != OBJECT_ARTIFACT):
            raise TypeError("materialized parser revision 必须是 Artifact")
        for values in (self.anchor_mappings, self.hypothesis_mappings):
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ObjectIdentity)
                           or item.object_kind != OBJECT_ARTIFACT
                           for item in values)):
                raise TypeError("parser revision mapping 必须是 Artifact tuple")
        if not isinstance(self.assertion_hashes, tuple):
            raise TypeError("parser revision assertion_hashes 必须是 tuple")
        assert_int(
            *self.assertion_hashes,
            _where="MaterializedParserRevision.assertion_hashes",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 revision、mapping 和全部 assertion 的确定性键。"""
        result = [
            _REVISION_VERSION,
            *_packed(self.revision.stable_key()),
            len(self.anchor_mappings),
        ]
        for item in self.anchor_mappings:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.hypothesis_mappings))
        for item in self.hypothesis_mappings:
            result.extend(_packed(item.stable_key()))
        result.extend((len(self.assertion_hashes), *self.assertion_hashes))
        return tuple(result)


@dataclass(frozen=True)
class ParserRevisionLineage:
    """从一等 revision 拓扑恢复的旧、新 ParserVersion 有向边。"""

    revision: ObjectIdentity
    old_source: SourceRef
    new_source: SourceRef

    def __post_init__(self) -> None:
        if (not isinstance(self.revision, ObjectIdentity)
                or self.revision.object_kind != OBJECT_ARTIFACT):
            raise TypeError("parser lineage revision 必须是 Artifact")
        if (not isinstance(self.old_source, SourceRef)
                or not isinstance(self.new_source, SourceRef)):
            raise TypeError("parser lineage 端点必须是 SourceRef")
        if (parser_lineage_key(self.old_source)
                != parser_lineage_key(self.new_source)):
            raise ParserRevisionError("parser lineage 边跨越来源 lineage")
        if (self.new_source.versions.parser.value
                <= self.old_source.versions.parser.value):
            raise ParserRevisionError("parser lineage 边没有严格前进")

    def stable_key(self) -> tuple[int, ...]:
        """返回 revision 及两个来源端点的完整确定性键。"""
        return (
            _REVISION_VERSION,
            *_packed(self.revision.stable_key()),
            *_packed(self.old_source.stable_key()),
            *_packed(self.new_source.stable_key()),
        )


class ParserRevisionGraph:
    """把 revision 及分组 mapping 物化为可回读的一等图结构。"""

    def __init__(
            self, ontology: GraphOntology, protocol: ParserRevisionProtocol,
            ) -> None:
        """绑定图 owner 并物化调用方注入的六个 predicate。"""
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ParserRevisionGraph 需要 GraphOntology")
        if not isinstance(protocol, ParserRevisionProtocol):
            raise TypeError("ParserRevisionGraph 需要 protocol")
        self.ontology = ontology
        self.protocol = protocol
        self._relations = {
            identity: ontology.materialize(identity)
            for identity in protocol.relations()
        }

    def lineages(self) -> tuple[ParserRevisionLineage, ...]:
        """从全部协议关系恢复线性 revision 边，并拒绝孤立、分叉和竞争拓扑。"""
        parent_by_mapping: dict[ObjectIdentity, ObjectIdentity] = {}
        mappings_by_revision: dict[
            ObjectIdentity, dict[ObjectIdentity, list[ObjectIdentity]]
        ] = {}
        relation_specs = (
            (self.protocol.revision_anchor_mapping,
             self.protocol.anchor_mapping_kind),
            (self.protocol.revision_hypothesis_mapping,
             self.protocol.hypothesis_mapping_kind),
        )
        for relation, mapping_kind in relation_specs:
            for statement in self.ontology.statements(
                    predicate=self._relations[relation]):
                revision = self.ontology.identity_of(statement.subject)
                mapping = self.ontology.identity_of(statement.object)
                revision_descriptor = self._artifact_descriptor(
                    revision, self.protocol.revision_kind)
                self._require_statement_metadata(
                    statement, revision_descriptor.scope)
                mapping_descriptor = self._artifact_descriptor(
                    mapping, mapping_kind)
                if (mapping_descriptor.source != revision_descriptor.source
                        or mapping_descriptor.scope
                        != revision_descriptor.scope):
                    raise ParserRevisionError(
                        "parser revision mapping 来源或 scope 漂移")
                expected_tail = (
                    1 if mapping_kind == self.protocol.anchor_mapping_kind
                    else 2)
                declaration = mapping_descriptor.declaration_key
                revision_key = revision_descriptor.declaration_key
                if (len(declaration) != len(revision_key) + 2
                        or declaration[:len(revision_key)] != revision_key
                        or declaration[-2] != expected_tail
                        or declaration[-1] < 0):
                    raise ParserRevisionError(
                        "parser revision mapping 声明键与 parent 不一致")
                previous = parent_by_mapping.get(mapping)
                if previous is not None and previous != revision:
                    raise ParserRevisionError(
                        "parser revision mapping 被多个 revision 共享")
                parent_by_mapping[mapping] = revision
                mappings_by_revision.setdefault(revision, {}).setdefault(
                    mapping_kind, []).append(mapping)

        old_by_mapping: dict[ObjectIdentity, list[ObjectIdentity]] = {}
        new_by_mapping: dict[ObjectIdentity, list[ObjectIdentity]] = {}
        for relation, target in (
                (self.protocol.mapping_old, old_by_mapping),
                (self.protocol.mapping_new, new_by_mapping)):
            for statement in self.ontology.statements(
                    predicate=self._relations[relation]):
                mapping = self.ontology.identity_of(statement.subject)
                endpoint = self.ontology.identity_of(statement.object)
                if mapping not in parent_by_mapping:
                    raise ParserRevisionError(
                        "parser revision graph 存在孤立 mapping")
                revision = parent_by_mapping[mapping]
                self._require_statement_metadata(
                    statement,
                    self._artifact_descriptor(
                        revision, self.protocol.revision_kind).scope,
                )
                target.setdefault(mapping, []).append(endpoint)

        reason_by_revision: dict[ObjectIdentity, list[ObjectIdentity]] = {}
        dimensions_by_revision: dict[ObjectIdentity, list[ObjectIdentity]] = {}
        for relation, target in (
                (self.protocol.revision_reason, reason_by_revision),
                (self.protocol.revision_dimension, dimensions_by_revision)):
            for statement in self.ontology.statements(
                    predicate=self._relations[relation]):
                revision = self.ontology.identity_of(statement.subject)
                value = self.ontology.identity_of(statement.object)
                if revision not in mappings_by_revision:
                    raise ParserRevisionError(
                        "parser revision 元数据指向无 mapping 的 revision")
                self._require_statement_metadata(
                    statement,
                    self._artifact_descriptor(
                        revision, self.protocol.revision_kind).scope,
                )
                target.setdefault(revision, []).append(value)

        edges = []
        seen_pairs: dict[
            tuple[SourceRef, SourceRef], ObjectIdentity
        ] = {}
        for revision in sorted(
                mappings_by_revision, key=ObjectIdentity.stable_key):
            descriptor = self._artifact_descriptor(
                revision, self.protocol.revision_kind)
            groups = mappings_by_revision[revision]
            hypothesis_mappings = groups.get(
                self.protocol.hypothesis_mapping_kind, [])
            if not hypothesis_mappings:
                raise ParserRevisionError(
                    "parser revision 缺少 hypothesis mapping")
            if len(reason_by_revision.get(revision, ())) != 1:
                raise ParserRevisionError("parser revision reason 不唯一")
            if not dimensions_by_revision.get(revision):
                raise ParserRevisionError("parser revision dimension 为空")
            if reason_by_revision[revision][0].object_kind != (
                    OBJECT_MINIMAL_INSTRUCTION):
                raise ParserRevisionError("parser revision reason 类型漂移")

            old_sources: set[SourceRef] = set()
            for mapping_kind, mappings in groups.items():
                indexes = []
                for mapping in mappings:
                    mapping_descriptor = self._artifact_descriptor(
                        mapping, mapping_kind)
                    indexes.append(mapping_descriptor.declaration_key[-1])
                    old_values = old_by_mapping.get(mapping, ())
                    if len(old_values) != 1:
                        raise ParserRevisionError(
                            "parser revision mapping 的 old 端点不唯一")
                    old = old_values[0]
                    replacements = new_by_mapping.get(mapping, ())
                    if len(set(replacements)) != len(replacements):
                        raise ParserRevisionError(
                            "parser revision mapping 含重复 new 端点")
                    if mapping_kind == self.protocol.anchor_mapping_kind:
                        old_sources.add(_anchor_source(old))
                        new_sources = tuple(_anchor_source(item)
                                            for item in replacements)
                    else:
                        if old.object_kind != OBJECT_HYPOTHESIS:
                            raise ParserRevisionError(
                                "hypothesis mapping old 端点类型漂移")
                        old_hypothesis = HypothesisKey.from_stable_key(
                            old.components)
                        old_sources.add(old_hypothesis.observation)
                        new_sources = []
                        for item in replacements:
                            if item.object_kind != OBJECT_HYPOTHESIS:
                                raise ParserRevisionError(
                                    "hypothesis mapping new 端点类型漂移")
                            new_sources.append(
                                HypothesisKey.from_stable_key(
                                    item.components).observation)
                    if any(source != descriptor.source
                           for source in new_sources):
                        raise ParserRevisionError(
                            "parser revision new 端点来源漂移")
                if len(set(indexes)) != len(indexes):
                    raise ParserRevisionError(
                        "parser revision mapping index 发生竞争")
            if len(old_sources) != 1:
                raise ParserRevisionError(
                    "parser revision old 端点没有唯一来源")
            old_source = next(iter(old_sources))
            edge = ParserRevisionLineage(
                revision, old_source, descriptor.source)
            pair = edge.old_source, edge.new_source
            previous = seen_pairs.get(pair)
            if previous is not None and previous != revision:
                raise ParserRevisionError(
                    "同一 parser lineage 边存在竞争 revision")
            seen_pairs[pair] = revision
            edges.append(edge)

        outgoing: dict[SourceRef, SourceRef] = {}
        incoming: dict[SourceRef, SourceRef] = {}
        for edge in edges:
            next_source = outgoing.get(edge.old_source)
            prior_source = incoming.get(edge.new_source)
            if next_source is not None and next_source != edge.new_source:
                raise ParserRevisionError("parser revision lineage 出现分叉")
            if prior_source is not None and prior_source != edge.old_source:
                raise ParserRevisionError("parser revision lineage 出现竞争前驱")
            outgoing[edge.old_source] = edge.new_source
            incoming[edge.new_source] = edge.old_source
        return tuple(sorted(edges, key=ParserRevisionLineage.stable_key))

    def preflight(
            self, request: ParserRevisionRequest,
            ) -> MaterializedParserRevision | None:
        """在首写前核验预算和已有 revision 是否完整、精确可重放。"""
        self.lineages()
        self._validate_request(request)
        revision, anchor_mappings, hypothesis_mappings = self._identities(
            request)
        revision_ref = self.ontology.resolve(revision)
        mapping_identities = (*anchor_mappings, *hypothesis_mappings)
        if revision_ref is None:
            if any(self.ontology.resolve(item) is not None
                   for item in mapping_identities):
                raise ParserRevisionError("parser revision graph 存在孤立 mapping")
            return None
        if any(self.ontology.resolve(item) is None for item in mapping_identities):
            raise ParserRevisionError("parser revision graph 缺少 mapping Artifact")
        expected = self._expected_triples(
            request, revision, anchor_mappings, hypothesis_mappings)
        actual = self._actual_triples((revision, *mapping_identities))
        if actual != expected:
            raise ParserRevisionError("parser revision graph 已有拓扑不完整或竞争")
        assertions = tuple(sorted(
            statement.assertion_hash
            for statement in self._statements((revision, *mapping_identities))
        ))
        return MaterializedParserRevision(
            revision, anchor_mappings, hypothesis_mappings, assertions)

    def materialize(
            self, request: ParserRevisionRequest,
            ) -> MaterializedParserRevision:
        """整批预检后物化 revision/mapping Artifact 和精确图关系。"""
        existing = self.preflight(request)
        if existing is not None:
            return existing
        revision, anchor_mappings, hypothesis_mappings = self._identities(
            request)
        triples = self._expected_triples(
            request, revision, anchor_mappings, hypothesis_mappings)
        identities = {
            value
            for triple in triples
            for value in (triple[1], triple[2])
        }
        refs = {
            identity: self.ontology.materialize(identity)
            for identity in sorted(
                identities, key=ObjectIdentity.stable_key)
        }
        statements = []
        for relation, subject, object_value in triples:
            statements.append(self.ontology.relate(
                self._relations[relation],
                refs[subject],
                refs[object_value],
                scope=request.new_scope,
                provenance_kind=self.protocol.provenance_kind,
                epistemic_origin=self.protocol.epistemic_origin,
                content_version=self.protocol.content_version,
                qualifiers=self.protocol.qualifiers,
            ))
        restored = self.preflight(request)
        if restored is None:
            raise RuntimeError("parser revision graph 写后无法恢复")
        expected_hashes = tuple(sorted(item.assertion_hash for item in statements))
        if restored.assertion_hashes != expected_hashes:
            raise ParserRevisionError("parser revision graph 写后 assertion 集漂移")
        return restored

    def _validate_request(self, request: ParserRevisionRequest) -> None:
        """核验 request 类型和三项局部影响预算。"""
        if not isinstance(request, ParserRevisionRequest):
            raise TypeError("ParserRevisionGraph request 类型错误")
        if len(request.anchors) > self.protocol.max_anchor_mappings:
            raise ParserRevisionError("parser revision anchor mapping 超预算")
        if len(request.hypotheses) > self.protocol.max_hypothesis_mappings:
            raise ParserRevisionError("parser revision hypothesis mapping 超预算")
        if any(
                len(item.replacements) > self.protocol.max_targets_per_mapping
                for item in (*request.anchors, *request.hypotheses)):
            raise ParserRevisionError("parser revision 单 mapping target 超预算")

    def _artifact_descriptor(
            self, identity: ObjectIdentity, expected_kind: ObjectIdentity,
            ):
        """核验 revision/mapping Artifact 的 kind、schema 和 document scope。"""
        try:
            descriptor = describe_artifact_identity(identity)
        except (TypeError, ValueError) as exc:
            raise ParserRevisionError(
                "parser revision graph 含非法 Artifact 身份") from exc
        if (descriptor.artifact_kind != expected_kind
                or descriptor.schema != self.protocol.schema
                or descriptor.scope is None
                or descriptor.scope.scope_kind != SCOPE_DOCUMENT
                or descriptor.scope.source != descriptor.source):
            raise ParserRevisionError(
                "parser revision Artifact kind/schema/scope 漂移")
        return descriptor

    def _require_statement_metadata(
            self, statement: GraphStatement, scope: ScopeIdentity,
            ) -> None:
        """核验 lineage 回读到的 statement 使用统一 scope 和协议来源元数据。"""
        assertion = statement.assertion
        if (assertion.scope != scope
                or assertion.provenance_kind
                != self.protocol.provenance_kind
                or assertion.epistemic_origin
                != self.protocol.epistemic_origin
                or assertion.content_version
                != self.protocol.content_version
                or assertion.qualifiers != self.protocol.qualifiers):
            raise ParserRevisionError(
                "parser revision statement 元数据漂移")

    def _identities(
            self, request: ParserRevisionRequest,
            ) -> tuple[
                ObjectIdentity,
                tuple[ObjectIdentity, ...],
                tuple[ObjectIdentity, ...],
            ]:
        """为 revision 和两类 mapping 构造来源化 Artifact 身份。"""
        revision = request.revision_identity(self.protocol)
        anchor_mappings = tuple(
            artifact_identity(
                request.new_source,
                self.protocol.anchor_mapping_kind,
                self.protocol.schema,
                (*request.revision_key, 1, index),
                integer_tuple_fingerprint(
                    item.stable_key(), domain=_ANCHOR_MAPPING_DOMAIN),
                request.new_scope,
            )
            for index, item in enumerate(request.anchors)
        )
        hypothesis_mappings = tuple(
            artifact_identity(
                request.new_source,
                self.protocol.hypothesis_mapping_kind,
                self.protocol.schema,
                (*request.revision_key, 2, index),
                integer_tuple_fingerprint(
                    item.stable_key(), domain=_HYPOTHESIS_MAPPING_DOMAIN),
                request.new_scope,
            )
            for index, item in enumerate(request.hypotheses)
        )
        for identity, expected_kind in (
                (revision, self.protocol.revision_kind),
                *((item, self.protocol.anchor_mapping_kind)
                  for item in anchor_mappings),
                *((item, self.protocol.hypothesis_mapping_kind)
                  for item in hypothesis_mappings)):
            if describe_artifact_identity(identity).artifact_kind != expected_kind:
                raise ParserRevisionError("parser revision Artifact kind 漂移")
        return revision, anchor_mappings, hypothesis_mappings

    def _expected_triples(
            self,
            request: ParserRevisionRequest,
            revision: ObjectIdentity,
            anchor_mappings: tuple[ObjectIdentity, ...],
            hypothesis_mappings: tuple[ObjectIdentity, ...],
            ) -> tuple[tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity], ...]:
        """建立保留 mapping 分组的完整预期图三元组。"""
        triples = []
        for mapping_identity, mapping in zip(anchor_mappings, request.anchors):
            triples.append((
                self.protocol.revision_anchor_mapping,
                revision,
                mapping_identity,
            ))
            triples.append((
                self.protocol.mapping_old,
                mapping_identity,
                mapping.old,
            ))
            triples.extend(
                (self.protocol.mapping_new, mapping_identity, item)
                for item in mapping.replacements)
        for mapping_identity, mapping in zip(
                hypothesis_mappings, request.hypotheses):
            triples.append((
                self.protocol.revision_hypothesis_mapping,
                revision,
                mapping_identity,
            ))
            triples.append((
                self.protocol.mapping_old,
                mapping_identity,
                mapping.old.object_identity(),
            ))
            triples.extend(
                (self.protocol.mapping_new,
                 mapping_identity,
                 item.object_identity())
                for item in mapping.replacements)
        triples.extend(
            (self.protocol.revision_dimension, revision, item)
            for item in request.dimensions)
        triples.append((self.protocol.revision_reason, revision, request.reason))
        return tuple(sorted(
            triples,
            key=lambda item: tuple(
                value
                for identity in item
                for value in _packed(identity.stable_key())
            ),
        ))

    def _statements(
            self, subjects: tuple[ObjectIdentity, ...],
            ) -> tuple[GraphStatement, ...]:
        """读取协议六关系下指定 subject 的全部 statement。"""
        result = []
        for subject in subjects:
            subject_ref = self.ontology.resolve(subject)
            if subject_ref is None:
                continue
            for relation in self.protocol.relations():
                result.extend(self.ontology.statements(
                    predicate=self._relations[relation],
                    subject=subject_ref,
                ))
        return tuple(result)

    def _actual_triples(
            self, subjects: tuple[ObjectIdentity, ...],
            ) -> tuple[tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity], ...]:
        """恢复 statement 身份并核验统一 scope/provenance 元数据。"""
        triples = []
        for statement in self._statements(subjects):
            assertion = statement.assertion
            if (assertion.scope != describe_artifact_identity(subjects[0]).scope
                    or assertion.provenance_kind
                    != self.protocol.provenance_kind
                    or assertion.epistemic_origin
                    != self.protocol.epistemic_origin
                    or assertion.content_version
                    != self.protocol.content_version
                    or assertion.qualifiers != self.protocol.qualifiers):
                raise ParserRevisionError("parser revision statement 元数据漂移")
            triples.append((
                self.ontology.identity_of(statement.predicate),
                self.ontology.identity_of(statement.subject),
                self.ontology.identity_of(statement.object),
            ))
        return tuple(sorted(
            triples,
            key=lambda item: tuple(
                value
                for identity in item
                for value in _packed(identity.stable_key())
            ),
        ))


__all__ = [
    "MaterializedParserRevision",
    "ParserAnchorRevision",
    "ParserHypothesisRevision",
    "ParserRevisionError",
    "ParserRevisionGraph",
    "ParserRevisionLineage",
    "ParserRevisionProtocol",
    "ParserRevisionRequest",
    "parser_lineage_key",
]
