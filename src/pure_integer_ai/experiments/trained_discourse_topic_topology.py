"""只读恢复训练图中的 W-08 discourse/topic 拓扑。

该 reader 只访问 ``GraphOntology.statements`` 和完整对象身份，不读取课程文件、
SourceRecord 正文、posting 或 successor。它恢复 frame 与 proposition、context、
information、QUD、reference、revision 的关系，并沿已发现 proposition 反查动态
proposition-to-proposition discourse relation。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    ObjectIdentity,
    SourceRef,
)


TOPOLOGY_READER_VERSION = 1
DISCOURSE_GRAPH_INPUT_ROLE = (21601, 1)


def _key(identity: ObjectIdentity) -> tuple[int, ...]:
    return identity.stable_key()


def _source_key(identity: ObjectIdentity) -> tuple[int, ...]:
    """从 W-08 来源化对象恢复完整 SourceRef；普通概念返回空。"""
    source_size = len(SourceRef(1, 1, 0, identity.owner, identity.versions).stable_key())
    if identity.object_kind == OBJECT_OCCURRENCE:
        source_values = identity.components[:source_size]
    elif identity.object_kind in {OBJECT_PROPOSITION, OBJECT_CONTEXT_SCOPE}:
        source_values = identity.components[1:1 + source_size]
    else:
        return ()
    if len(source_values) != source_size:
        return ()
    return SourceRef.from_stable_key(source_values).stable_key()


@dataclass(frozen=True, slots=True)
class DiscourseTopicGraphFrame:
    """一个训练 frame 及其恢复出的 W-08 一等对象引用。"""

    frame_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    proposition_keys: tuple[tuple[int, ...], ...]
    context_keys: tuple[tuple[int, ...], ...]
    information_keys: tuple[tuple[int, ...], ...]
    qud_keys: tuple[tuple[int, ...], ...]
    presupposition_keys: tuple[tuple[int, ...], ...]
    reference_keys: tuple[tuple[int, ...], ...]
    revision_keys: tuple[tuple[int, ...], ...]

    def stable_key(self) -> tuple[int, ...]:
        values = [TOPOLOGY_READER_VERSION, *self.frame_key]
        for group in (
                self.proposition_keys, self.context_keys, self.information_keys,
                self.qud_keys, self.presupposition_keys, self.reference_keys,
                self.revision_keys):
            values.extend((len(group), *(value for key in group for value in (len(key), *key))))
        values.extend((len(self.source_ref), *self.source_ref))
        return tuple(values)


@dataclass(frozen=True, slots=True)
class DiscourseTopicGraphRelation:
    """两个来源化 Proposition 之间的动态 discourse relation。"""

    subject_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    object_key: tuple[int, ...]
    source_ref: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("subject_key", "predicate_key", "object_key"):
            value = getattr(self, name)
            if (type(value) is not tuple or not value
                    or any(type(item) is not int or item < 0 for item in value)):
                raise ValueError(f"{name} 必须是非空非负整数 tuple")
        if type(self.source_ref) is not tuple:
            raise ValueError("source_ref 必须是整数 tuple")
        if self.source_ref and (
                len(self.source_ref) != 11
                or any(type(item) is not int or item < 0
                       for item in self.source_ref)):
            raise ValueError("source_ref 必须是完整 SourceRef 稳定键")

    def stable_key(self) -> tuple[int, ...]:
        return (
            TOPOLOGY_READER_VERSION,
            len(self.subject_key),
            *self.subject_key,
            len(self.predicate_key),
            *self.predicate_key,
            len(self.object_key),
            *self.object_key,
            len(self.source_ref),
            *self.source_ref,
        )


@dataclass(frozen=True, slots=True)
class DiscourseTopicGraphTopology:
    """从物理 graph statement 恢复的不可变 W-08 拓扑快照。"""

    frames: tuple[DiscourseTopicGraphFrame, ...]
    relations: tuple[DiscourseTopicGraphRelation, ...]

    @property
    def proposition_keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted({key for frame in self.frames for key in frame.proposition_keys}))

    def stable_key(self) -> tuple[int, ...]:
        return (
            TOPOLOGY_READER_VERSION,
            *(value for frame in self.frames for value in (len(frame.stable_key()), *frame.stable_key())),
            *(value for relation in self.relations for value in (len(relation.stable_key()), *relation.stable_key())),
        )


def read_discourse_topic_topology(
        ontology: GraphOntology,
        predicate_identities: Mapping[str, ObjectIdentity],
        ) -> DiscourseTopicGraphTopology:
    """只读恢复 W-08 frame、对象链接和动态篇章关系。"""
    if not isinstance(ontology, GraphOntology):
        raise TypeError("ontology 必须是 GraphOntology")
    if not isinstance(predicate_identities, Mapping):
        raise TypeError("predicate_identities 必须是 Mapping")
    by_predicate: dict[tuple[int, ...], str] = {}
    for name, identity in predicate_identities.items():
        if not isinstance(name, str) or not name or not isinstance(identity, ObjectIdentity):
            raise TypeError("W-08 predicate identity 非法")
        by_predicate[identity.stable_key()] = name

    links: dict[tuple[int, ...], dict[str, set[tuple[int, ...]]]] = {}
    proposition_refs: dict[tuple[int, ...], object] = {}
    for predicate_key, name in sorted(by_predicate.items()):
        predicate = ontology.resolve(ObjectIdentity.from_stable_key(predicate_key))
        if predicate is None:
            continue
        for statement in ontology.statements(predicate=predicate):
            subject = ontology.identity_of(statement.subject)
            obj = ontology.identity_of(statement.object)
            subject_key, object_key = _key(subject), _key(obj)
            links.setdefault(subject_key, {}).setdefault(name, set()).add(object_key)
            if obj.object_kind == OBJECT_PROPOSITION:
                proposition_refs[object_key] = statement.object

    relations: set[DiscourseTopicGraphRelation] = set()
    for subject_key, ref in sorted(proposition_refs.items()):
        for statement in ontology.statements(subject=ref):
            predicate = ontology.identity_of(statement.predicate)
            obj = ontology.identity_of(statement.object)
            if (obj.object_kind != OBJECT_PROPOSITION
                    or predicate.stable_key() in by_predicate):
                continue
            object_key = _key(obj)
            source_ref = _source_key(ObjectIdentity.from_stable_key(subject_key))
            relations.add(DiscourseTopicGraphRelation(
                subject_key, predicate.stable_key(), object_key, source_ref))

    frames = []
    for frame_key, grouped in sorted(links.items()):
        propositions = tuple(sorted(grouped.get("frame_proposition", ())))
        linked_keys = tuple(
            key for values in grouped.values() for key in values)
        source_refs = tuple(sorted({
            _source_key(ObjectIdentity.from_stable_key(key))
            for key in linked_keys
            if _source_key(ObjectIdentity.from_stable_key(key))
        }))
        if len(source_refs) > 1:
            raise ValueError("W-08 frame 跨 SourceRef")
        frames.append(DiscourseTopicGraphFrame(
            frame_key,
            source_refs[0] if source_refs else (),
            propositions,
            tuple(sorted(grouped.get("frame_context", ()))),
            tuple(sorted(grouped.get("frame_information", ()))),
            tuple(sorted(grouped.get("frame_qud", ()))),
            tuple(sorted(grouped.get("frame_presupposition", ()))),
            tuple(sorted(grouped.get("frame_reference", ()))),
            tuple(sorted(grouped.get("frame_revision", ()))),
        ))
    return DiscourseTopicGraphTopology(
        tuple(sorted(frames, key=lambda item: item.stable_key())),
        tuple(sorted(relations, key=lambda item: item.stable_key())),
    )


__all__ = [
    "DISCOURSE_GRAPH_INPUT_ROLE",
    "TOPOLOGY_READER_VERSION",
    "DiscourseTopicGraphFrame",
    "DiscourseTopicGraphRelation",
    "DiscourseTopicGraphTopology",
    "read_discourse_topic_topology",
]
