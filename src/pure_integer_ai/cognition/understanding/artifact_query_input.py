"""将 LC-16 载体投影接入统一查询输入，保留完整整数来源与结构。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_TEXT_RANGE,
    PROJECTION_UNDERSTANDING,
    RAW_UNIT_UNICODE_SCALAR,
    ArtifactAnchor,
    ArtifactEnvelope,
    ArtifactReferenceBinding,
    ArtifactSemanticProjection,
    ArtifactStructureNode,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    SPAN_SOURCE,
    STRUCTURE_CLOSED,
    STRUCTURE_OPEN,
    InputCarrierCandidate,
    InputRelationCandidate,
    InputSemanticCandidate,
    InputSpan,
    InputSpanRelation,
    InputToken,
    QueryInputStructure,
    TrainedInputStructure,
)


ARTIFACT_QUERY_INPUT_VERSION = 91540
ARTIFACT_MEMORY_CANDIDATE_VERSION = 91502
ARTIFACT_MEMORY_CATEGORY = 10
ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE = 91541
ARTIFACT_QUERY_EVIDENCE_NAMESPACE = 91542
# Interaction Memory 已冻结其 Observation context 的会话命名空间为 91501。
# 工件仍有独立 hypothesis/evidence namespace；这里只复用可冷恢复的 O/H/E
# context 外壳，避免生成一条不能被同一 Memory 图重新读取的伪 Observation。
INTERACTION_MEMORY_CONTEXT_NAMESPACE = 91501


def _frame(tag: int, *parts: tuple[int, ...]) -> tuple[int, ...]:
    """编码有长度边界的纯整数记录，禁止把身份压成散列摘要。"""
    if type(tag) is not int or tag <= 0:
        raise ValueError("artifact query frame tag 必须为正严格整数")
    values = [tag]
    for part in parts:
        if (type(part) is not tuple
                or any(type(value) is not int or value < 0 for value in part)):
            raise ValueError("artifact query frame part 必须为非负严格整数 tuple")
        values.extend((len(part), *part))
    return tuple(values)


def _frame_parts(key: tuple[int, ...], tag: int) -> tuple[tuple[int, ...], ...]:
    """严格解码长度边界记录，拒绝截断、尾随和布尔值。"""
    if (type(key) is not tuple or not key or key[0] != tag
            or any(type(value) is not int or value < 0 for value in key)):
        raise ValueError("artifact query 整数记录类型或版本非法")
    result = []
    cursor = 1
    while cursor < len(key):
        size = key[cursor]
        cursor += 1
        end = cursor + size
        if end > len(key):
            raise ValueError("artifact query 整数记录被截断")
        result.append(key[cursor:end])
        cursor = end
    return tuple(result)


def artifact_memory_candidate_key(
        projection: ArtifactSemanticProjection,
        ) -> tuple[int, ...]:
    """为一个理解方向载体投影建立精确 Memory Hypothesis 身份。"""
    if not isinstance(projection, ArtifactSemanticProjection):
        raise TypeError("artifact projection 必须是 ArtifactSemanticProjection")
    if PROJECTION_UNDERSTANDING not in projection.directions:
        raise ValueError("只有 UNDERSTANDING projection 才能进入查询 Memory")
    return _frame(
        ARTIFACT_MEMORY_CANDIDATE_VERSION,
        (ARTIFACT_MEMORY_CATEGORY,),
        projection.stable_key(),
    )


def artifact_memory_reference_candidate_key(
        reference: ArtifactReferenceBinding,
        ) -> tuple[int, ...]:
    """为 carrier 图引用建立可路由的 Memory Hypothesis 身份。"""
    if not isinstance(reference, ArtifactReferenceBinding):
        raise TypeError("artifact reference 必须是 ArtifactReferenceBinding")
    return _frame(
        ARTIFACT_MEMORY_CANDIDATE_VERSION,
        (ARTIFACT_MEMORY_CATEGORY,), (1,), reference.stable_key())


def artifact_memory_envelope_candidate_key(
        envelope: ArtifactEnvelope,
        ) -> tuple[int, ...]:
    """为未获语义投影的 carrier 保留 exact 输入 Observation 身份。"""
    if not isinstance(envelope, ArtifactEnvelope):
        raise TypeError("artifact envelope 必须是 ArtifactEnvelope")
    return _frame(
        ARTIFACT_MEMORY_CANDIDATE_VERSION,
        (ARTIFACT_MEMORY_CATEGORY,), (2,), envelope.stable_key())


def artifact_memory_candidate_projection(
        key: tuple[int, ...],
        ) -> ArtifactSemanticProjection:
    """从精确候选键回读投影本体，不按表层或相邻字节猜测。"""
    fields = _frame_parts(key, ARTIFACT_MEMORY_CANDIDATE_VERSION)
    if (len(fields) != 2 or fields[0] != (ARTIFACT_MEMORY_CATEGORY,)):
        raise ValueError("artifact Memory candidate 格式非法")
    projection = ArtifactSemanticProjection.from_stable_key(fields[1])
    if PROJECTION_UNDERSTANDING not in projection.directions:
        raise ValueError("artifact Memory candidate 不含 UNDERSTANDING 方向")
    return projection


def artifact_memory_candidate_reference(
        key: tuple[int, ...],
        ) -> ArtifactReferenceBinding:
    """从带显式 subtype 的整数候选键恢复工件引用本体。"""
    fields = _frame_parts(key, ARTIFACT_MEMORY_CANDIDATE_VERSION)
    if (len(fields) != 3 or fields[0] != (ARTIFACT_MEMORY_CATEGORY,)
            or fields[1] != (1,)):
        raise ValueError("artifact Memory reference candidate 格式非法")
    return ArtifactReferenceBinding.from_stable_key(fields[2])


def artifact_memory_candidate_envelope(
        key: tuple[int, ...],
        ) -> ArtifactEnvelope:
    """恢复不带语义断言的 exact carrier 输入候选。"""
    fields = _frame_parts(key, ARTIFACT_MEMORY_CANDIDATE_VERSION)
    if (len(fields) != 3 or fields[0] != (ARTIFACT_MEMORY_CATEGORY,)
            or fields[1] != (2,)):
        raise ValueError("artifact Memory envelope candidate 格式非法")
    return ArtifactEnvelope.from_stable_key(fields[2])


def artifact_memory_candidate_routes(
        key: tuple[int, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """返回 exact projection 与共享语义对象两类显式结构路由。"""
    fields = _frame_parts(key, ARTIFACT_MEMORY_CANDIDATE_VERSION)
    if len(fields) == 2:
        projection = artifact_memory_candidate_projection(key)
        return tuple(sorted({
            _frame(ARTIFACT_QUERY_INPUT_VERSION, (1,), key),
            _frame(ARTIFACT_QUERY_INPUT_VERSION, (2,),
                   projection.semantic_object.stable_key()),
        }))
    if len(fields) == 3 and fields[1] == (2,):
        envelope = artifact_memory_candidate_envelope(key)
        return tuple(sorted({
            _frame(ARTIFACT_QUERY_INPUT_VERSION, (6,), key),
            _frame(ARTIFACT_QUERY_INPUT_VERSION, (7,),
                   envelope.identity.stable_key()),
        }))
    reference = artifact_memory_candidate_reference(key)
    routes = {
        _frame(ARTIFACT_QUERY_INPUT_VERSION, (3,), key),
        _frame(ARTIFACT_QUERY_INPUT_VERSION, (4,),
               reference.anchor_identity.stable_key()),
    }
    if reference.target_anchor is not None:
        routes.add(_frame(
            ARTIFACT_QUERY_INPUT_VERSION, (5,),
            reference.target_anchor.stable_key()))
    return tuple(sorted(routes))


@dataclass(frozen=True, slots=True)
class ArtifactMemoryObservationDraft:
    """一次载体输入可写入 Interaction Memory 的 O/H/E 草案与追加证据。"""

    draft: ObservationIntakeDraft
    evidence_tails: tuple[tuple[EvidenceRecord, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.draft, ObservationIntakeDraft):
            raise TypeError("artifact Memory draft 必须是 ObservationIntakeDraft")
        if (type(self.evidence_tails) is not tuple
                or any(type(item) is not tuple
                       or any(not isinstance(row, EvidenceRecord) for row in item)
                       for item in self.evidence_tails)
                or len(self.evidence_tails) != len(self.draft.hypotheses)):
            raise TypeError("artifact Memory evidence tails 类型或长度非法")


@dataclass(frozen=True, slots=True)
class ArtifactQueryInput:
    """一次工件查询的完整载体、结构与语义投影，不承担字符串解析或事实晋升。"""

    envelope: ArtifactEnvelope
    anchors: tuple[ArtifactAnchor, ...]
    structure_nodes: tuple[ArtifactStructureNode, ...]
    projections: tuple[ArtifactSemanticProjection, ...]
    references: tuple[ArtifactReferenceBinding, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, ArtifactEnvelope):
            raise TypeError("artifact query envelope 必须是 ArtifactEnvelope")
        if ArtifactEnvelope.from_stable_key(self.envelope.stable_key()) != self.envelope:
            raise ValueError("artifact query envelope 不能稳定 round-trip")
        for label, values, expected in (
                ("anchors", self.anchors, ArtifactAnchor),
                ("structure_nodes", self.structure_nodes, ArtifactStructureNode),
                ("projections", self.projections, ArtifactSemanticProjection),
                ("references", self.references, ArtifactReferenceBinding)):
            if (type(values) is not tuple
                    or any(not isinstance(item, expected) for item in values)
                    or values != tuple(sorted(set(values), key=lambda item: item.stable_key()))):
                raise ValueError(f"artifact query {label} 必须按稳定键排序去重")
        if not self.anchors and not self.structure_nodes:
            raise ValueError("artifact query 至少需要一个 carrier anchor 或 structure node")
        envelope_identity = self.envelope.identity
        source = self.envelope.source
        scope = self.envelope.scope
        for item in (*self.anchors, *self.structure_nodes, *self.references):
            if (item.envelope_identity != envelope_identity
                    or item.source != source or item.scope != scope):
                raise ValueError("artifact query local object 跨 envelope/source/scope")
            if type(item).from_stable_key(item.stable_key()) != item:
                raise ValueError("artifact query local object 不能稳定 round-trip")
        anchor_ids = {item.identity for item in self.anchors}
        node_ids = {item.identity for item in self.structure_nodes}
        for projection in self.projections:
            if (projection.envelope_identity != envelope_identity
                    or projection.source != source or projection.scope != scope):
                raise ValueError("artifact query projection 跨 envelope/source/scope")
            if ArtifactSemanticProjection.from_stable_key(projection.stable_key()) != projection:
                raise ValueError("artifact query projection 不能稳定 round-trip")
            if (not set(projection.anchor_identities) <= anchor_ids
                    or not set(projection.structure_node_identities) <= node_ids):
                raise ValueError("artifact query projection 引用了未登记局部对象")
        for reference in self.references:
            if reference.anchor_identity not in anchor_ids:
                raise ValueError("artifact query reference 引用了未登记源 anchor")

    @property
    def source_ref(self) -> tuple[int, ...]:
        """返回本次载体的完整输入身份，不以 raw body 或哈希作为语义替代。"""
        return _frame(ARTIFACT_QUERY_INPUT_VERSION, (1,),
                      self.envelope.stable_key())

    @property
    def understanding_projections(self) -> tuple[ArtifactSemanticProjection, ...]:
        """只暴露明确授权给理解方向的投影；推理/生成方向不越权充当输入。"""
        return tuple(item for item in self.projections
                     if PROJECTION_UNDERSTANDING in item.directions)

    def unicode_surface(self) -> str:
        """仅在原始单元已声明 Unicode scalar 时还原宿主显示/Memory 边界文本。"""
        if self.envelope.raw_unit_kind != RAW_UNIT_UNICODE_SCALAR:
            raise ValueError("OCTET artifact 不得被伪装为 Unicode 文本")
        if not self.envelope.raw_units:
            raise ValueError("空 artifact 不能形成查询输入")
        return "".join(chr(value) for value in self.envelope.raw_units)

    def memory_candidate_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回由 projection 本体授权的 Memory 路由键。"""
        return tuple(sorted(
            (artifact_memory_envelope_candidate_key(self.envelope),
             *(
                artifact_memory_candidate_key(item)
                for item in self.understanding_projections),
             *(artifact_memory_reference_candidate_key(item)
                for item in self.references))))

    def _projection_spans(self) -> dict[ArtifactSemanticProjection, tuple[InputSpan, ...]]:
        """将 text-range anchor 映射为输入区间；其他 carrier 保留完整来源区间。"""
        values = self.envelope.raw_units
        if not values:
            raise ValueError("空 artifact 不能形成 QueryInputStructure")
        source_ref = self.source_ref
        source_span = InputSpan(source_ref, 0, len(values), SPAN_SOURCE, values)
        anchors = {item.identity: item for item in self.anchors}
        result = {}
        for projection in self.understanding_projections:
            spans = []
            for identity in projection.anchor_identities:
                anchor = anchors[identity]
                if (anchor.anchor_kind == ANCHOR_TEXT_RANGE
                        and len(anchor.coordinates) == 2):
                    start, end = anchor.coordinates
                    if 0 <= start < end <= len(values):
                        spans.append(InputSpan(
                            source_ref, start, end, SPAN_SOURCE,
                            values[start:end]))
            result[projection] = tuple(sorted(
                set(spans) or {source_span}, key=lambda item: item.stable_key()))
        return result

    def query_input_structure(
            self,
            structures: tuple[TrainedInputStructure, ...],
            ) -> QueryInputStructure:
        """将已授权投影与训练后角色拓扑组装为一个 QueryInputStructure。

        原始单元不参与字符近邻或词面匹配。只有 semantic projection 精确命中
        已训练 member identity 时才出现候选；非文本 OCTET 因而可查询图结构，
        但不会被解释成 Unicode 字符。
        """
        if (type(structures) is not tuple
                or any(not isinstance(item, TrainedInputStructure) for item in structures)):
            raise TypeError("artifact query structures 必须是 TrainedInputStructure tuple")
        if not self.envelope.raw_units:
            raise ValueError("空 artifact 不能形成 QueryInputStructure")
        source_ref = self.source_ref
        values = self.envelope.raw_units
        tokens = tuple(InputToken(index, value) for index, value in enumerate(values))
        source_span = InputSpan(source_ref, 0, len(values), SPAN_SOURCE, values)
        projection_spans = self._projection_spans()
        projections_by_object: dict[tuple[int, ...], list[ArtifactSemanticProjection]] = {}
        for projection in self.understanding_projections:
            projections_by_object.setdefault(
                projection.semantic_object.stable_key(), []).append(projection)
        semantic_candidates: set[InputSemanticCandidate] = set()
        carrier_candidates: set[InputCarrierCandidate] = set()
        relation_candidates: set[InputRelationCandidate] = set()
        for structure in structures:
            supported = []
            structure_candidates = []
            for member in structure.members:
                projections = projections_by_object.get(member.candidate_key, ())
                candidates = []
                for projection in projections:
                    for span in projection_spans[projection]:
                        candidate = InputSemanticCandidate(
                            member.candidate_key, member.object_kind,
                            member.projection_kind, span.ref_key,
                            member.proposition_key, member.predicate_key,
                            member.role_key, member.source_ref,
                            member.member_ordinal,
                        )
                        semantic_candidates.add(candidate)
                        candidates.append(candidate)
                if candidates:
                    supported.append(member)
                    structure_candidates.extend(candidates)
            if not supported:
                continue
            required_count = len(structure.members)
            support_count = len(supported)
            predicate_coverage = int(any(
                item.projection_kind == PROJECTION_PREDICATE for item in supported))
            role_coverage = sum(
                item.projection_kind == PROJECTION_FILLER for item in supported)
            state = (STRUCTURE_CLOSED if support_count == required_count
                     else STRUCTURE_OPEN)
            span_refs = tuple(sorted({item.span_ref for item in structure_candidates}))
            relation = InputRelationCandidate(
                structure.proposition_key, structure.predicate_key,
                structure.relation_structure_key(), structure.carrier_key,
                structure.source_ref, span_refs, required_count, support_count,
                role_coverage, predicate_coverage,
                required_count - support_count, state, 0,
            )
            relation_candidates.add(relation)
            carrier_candidates.add(InputCarrierCandidate(
                structure.carrier_key, structure.relation_structure_key(),
                structure.source_ref, span_refs[0], support_count))
        spans = {source_span}
        for rows in projection_spans.values():
            spans.update(rows)
        spans_tuple = tuple(sorted(spans, key=lambda item: item.stable_key()))
        precedes = []
        contains = []
        for left in spans_tuple:
            for right in spans_tuple:
                if left == right:
                    continue
                if left.end <= right.start:
                    precedes.append(InputSpanRelation(left.ref_key, right.ref_key))
                if (left.start <= right.start and right.end <= left.end
                        and (left.start, left.end) != (right.start, right.end)):
                    contains.append(InputSpanRelation(left.ref_key, right.ref_key))
        return QueryInputStructure(
            source_ref,
            tokens,
            source_span.ref_key,
            spans_tuple,
            tuple(sorted(set(precedes), key=lambda item: item.stable_key())),
            tuple(sorted(set(contains), key=lambda item: item.stable_key())),
            tuple(sorted(carrier_candidates, key=lambda item: item.stable_key())),
            tuple(sorted(semantic_candidates, key=lambda item: item.stable_key())),
            tuple(sorted(relation_candidates, key=lambda item: item.stable_key())),
        )

    def memory_observation_draft(
            self,
            source: SourceRef,
            *, session_id: int, speaker_kind: int,
            ) -> ArtifactMemoryObservationDraft:
        """构造来源化 O/H/E 草案；每条原 projection Evidence 保留自己的 stance。"""
        if not isinstance(source, SourceRef):
            raise TypeError("artifact Memory source 必须是 SourceRef")
        if (type(session_id) is not int or session_id <= 0
                or type(speaker_kind) is not int or speaker_kind <= 0):
            raise ValueError("artifact Memory session/speaker 必须为正严格整数")
        projections = self.understanding_projections
        context = ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (INTERACTION_MEMORY_CONTEXT_NAMESPACE, session_id, speaker_kind),
            source.owner,
            source.versions,
        )
        relation_refs = [MemoryLinkedRef.object(context),
                         MemoryLinkedRef.object(self.envelope.identity)]
        relation_refs.extend(MemoryLinkedRef.object(item.identity)
                             for item in self.anchors)
        relation_refs.extend(MemoryLinkedRef.object(item.identity)
                             for item in self.structure_nodes)
        relation_refs.extend(MemoryLinkedRef.object(item.identity)
                             for item in projections)
        relation_refs.extend(MemoryLinkedRef.object(item.identity)
                             for item in self.references)
        relation_refs.extend(MemoryLinkedRef.object(item.semantic_object)
                             for item in projections)
        hypotheses = []
        tails = []
        envelope_candidate = artifact_memory_envelope_candidate_key(self.envelope)
        hypotheses.append(HypothesisIntakeDraft(
            _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE,
                   (source.document_id, 0), envelope_candidate),
            (ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, ARTIFACT_MEMORY_CATEGORY),
            envelope_candidate,
            _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, (3,), self.source_ref),
            EVIDENCE_UNKNOWN,
            reason_key=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (5,),
                              self.envelope.identity.stable_key()),
            detail=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (6,),
                          self.envelope.stable_key()),
        ))
        tails.append(())
        for ordinal, projection in enumerate(projections, start=1):
            evidence = projection.evidence
            first = evidence[0]
            candidate_key = artifact_memory_candidate_key(projection)
            hypotheses.append(HypothesisIntakeDraft(
                _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE,
                       (source.document_id, ordinal), candidate_key),
                (ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, ARTIFACT_MEMORY_CATEGORY),
                candidate_key,
                _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, (1,),
                       projection.semantic_object.stable_key()),
                first.stance,
                reason_key=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (1,),
                                  projection.identity.stable_key(),
                                  first.stable_key()),
                detail=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (2,),
                              projection.identity.stable_key(),
                              first.stable_key()),
            ))
            tails.append(evidence[1:])
        for ordinal, reference in enumerate(
                self.references, start=1 + len(projections)):
            candidate_key = artifact_memory_reference_candidate_key(reference)
            hypotheses.append(HypothesisIntakeDraft(
                _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE,
                       (source.document_id, ordinal), candidate_key),
                (ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, ARTIFACT_MEMORY_CATEGORY),
                candidate_key,
                _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE, (2,),
                       reference.relation.stable_key(),
                       reference.anchor_identity.stable_key()),
                EVIDENCE_UNKNOWN,
                reason_key=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (3,),
                                  reference.identity.stable_key()),
                detail=_frame(ARTIFACT_QUERY_EVIDENCE_NAMESPACE, (4,),
                              reference.stable_key()),
            ))
            tails.append(())
        return ArtifactMemoryObservationDraft(
            ObservationIntakeDraft(
                _frame(ARTIFACT_QUERY_HYPOTHESIS_NAMESPACE,
                       (source.document_id,), self.source_ref),
                MemoryLinkedRef.object(context),
                relation_occurrences=tuple(sorted(
                    set(relation_refs), key=lambda item: item.stable_key())),
                hypotheses=tuple(hypotheses),
            ),
            tuple(tails),
        )

    def integer_trace(self) -> dict[str, object]:
        """导出只含整数的可回读输入投影，不暴露原始正文作为回答来源。"""
        return {
            "schema_version": ARTIFACT_QUERY_INPUT_VERSION,
            "envelope": list(self.envelope.stable_key()),
            "anchors": [list(item.stable_key()) for item in self.anchors],
            "structure_nodes": [list(item.stable_key()) for item in self.structure_nodes],
            "projections": [list(item.stable_key()) for item in self.projections],
            "references": [list(item.stable_key()) for item in self.references],
            "understanding_projections": [
                list(item.stable_key()) for item in self.understanding_projections],
            "memory_candidate_keys": [
                list(item) for item in self.memory_candidate_keys()],
            "stable_key": list(self.stable_key()),
        }

    def stable_key(self) -> tuple[int, ...]:
        """编码完整工件本体引用；不以 identity fingerprint 代替成员。"""
        values = [ARTIFACT_QUERY_INPUT_VERSION, *_frame(1, self.envelope.stable_key())]
        for collection in (
                self.anchors, self.structure_nodes, self.projections,
                self.references):
            values.append(len(collection))
            for item in collection:
                key = item.stable_key()
                values.extend((len(key), *key))
        return tuple(values)


__all__ = [
    "ARTIFACT_MEMORY_CANDIDATE_VERSION",
    "ARTIFACT_MEMORY_CATEGORY",
    "ARTIFACT_QUERY_INPUT_VERSION",
    "ArtifactMemoryObservationDraft",
    "ArtifactQueryInput",
    "artifact_memory_candidate_envelope",
    "artifact_memory_candidate_key",
    "artifact_memory_candidate_projection",
    "artifact_memory_candidate_reference",
    "artifact_memory_envelope_candidate_key",
    "artifact_memory_reference_candidate_key",
    "artifact_memory_candidate_routes",
]
