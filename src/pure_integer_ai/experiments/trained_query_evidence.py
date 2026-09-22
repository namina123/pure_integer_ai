"""把训练图原始证据无损投影到共同查询；当前有效集不覆盖历史。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord, HypothesisKey
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import EvidenceEntry, SPACE_CORE
from pure_integer_ai.cognition.shared.query_state import SPACE_DIALOGUE
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationGenerationInput,
    GraphRelationGeneration,
    RelationSurfaceFrame,
)
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_RECORD,
    IntegerIdentityRegistry,
)


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    """给图引用或整数载荷保留明确长度边界。"""
    return len(key), *key


def _evidence_ref(hypothesis: HypothesisKey, evidence_id: int) -> tuple[int, ...]:
    """以完整候选身份及原证据 id 定位 H-00 事件，不用摘要代替本体。"""
    key = hypothesis.stable_key()
    return 1, len(key), *key, evidence_id


def core_query_evidence(
        proposition: ObjectIdentity,
        history: tuple[EvidenceRecord, ...],
        registry: IntegerIdentityRegistry,
        ) -> tuple[EvidenceEntry, ...]:
    """保留原 stance、来源、scope、理由、逻辑序和完整 supersedes 连接。"""
    if not history:
        raise ValueError("Core 查询命题必须有可恢复的 Evidence 历史")
    return tuple(EvidenceEntry(
        # 训练后只读库可能没有单独写入 source-record header；仍以同一
        # registry 的确定性 identity hash 作为正 source 标识，禁止 0
        # 污染 QueryState 证据（0 仅表示无 parent，不是有效来源）。
        source_ref=((registry.find(IDENTITY_SOURCE_RECORD, item.source.stable_key())
                     or registry.identity_hash(IDENTITY_SOURCE_RECORD,
                                               item.source.stable_key())),
                    *item.source.stable_key()),
        hypothesis_key=proposition.stable_key(),
        space=SPACE_CORE,
        polarity=item.stance,
        evidence_key=_evidence_ref(item.hypothesis, item.evidence_id),
        scope_key=item.hypothesis.scope.stable_key(),
        payload_key=item.stable_key(),
        supersedes_key=(
            _evidence_ref(item.hypothesis, item.supersedes_evidence_id)
            if item.supersedes_evidence_id else ()),
    ) for item in history)


def current_query_evidence(
        evidence: tuple[EvidenceEntry, ...],
        ) -> tuple[EvidenceEntry, ...]:
    """只从已经进入 QueryState 的连接派生活动证据，不删除原集合。"""
    superseded = {(item.space, item.supersedes_key) for item in evidence
                  if item.supersedes_key}
    return tuple(item for item in evidence
                 if (item.space, item.evidence_key) not in superseded)


def dialogue_query_evidence(
        claim: ObjectIdentity,
        frame: RelationSurfaceFrame,
        source: ActiveRelationGenerationInput,
        generation: GraphRelationGeneration | None,
        ) -> EvidenceEntry:
    """保留组织框架的角色序、整数间隔及实际 connector/realization 引用。"""
    if source.proposition.definition.proposition != frame.proposition:
        raise ValueError("Dialogue frame 与来源图命题不一致")
    payload = [1, *_pack(frame.proposition.stable_key()),
               *_pack(frame.predicate.stable_key()), frame.source_hash,
               frame.envelope_start, frame.envelope_end, len(frame.roles)]
    for role in frame.roles:
        payload.extend(_pack(role.stable_key()))
    payload.append(len(frame.gaps))
    for gap in frame.gaps:
        payload.extend(_pack(tuple(ord(value) for value in gap)))
    payload.append(0 if generation is None else 1)
    if generation is not None:
        if (generation.frame_proposition != frame.proposition
                or generation.frame_source_hash != frame.source_hash):
            raise ValueError("Dialogue 证据必须指向真正参与生成的 frame")
        payload.extend((generation.slot_count,
                        *_pack(() if generation.connector is None
                               else generation.connector.stable_key()),
                        len(generation.representations)))
        for representation in generation.representations:
            payload.extend(_pack(representation.stable_key()))
        payload.extend(_pack(generation.trace))
    return EvidenceEntry(
        source_ref=(frame.source_hash, *source.proposition.definition.source.stable_key()),
        hypothesis_key=claim.stable_key(),
        space=SPACE_DIALOGUE,
        evidence_key=(2, *_pack(frame.proposition.stable_key()), frame.source_hash),
        scope_key=source.hypothesis.scope.stable_key(),
        payload_key=tuple(payload),
    )


__all__ = ["core_query_evidence", "current_query_evidence", "dialogue_query_evidence"]
