"""LC-05 Event/State、时间锚、区间和体貌到共享 Core 图的训练桥。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    event_identity,
    proposition_identity,
    semantic_source,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_authored_event_time_aspect_course import (
    read_authored_event_time_aspect_seeds,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED


BRIDGE_VERSION = 1
_NS = (21602, BRIDGE_VERSION)
_VERSIONS = VersionBundle(
    CorpusVersion(1), ParserVersion(1),
    PrimitiveVersion(1), CurriculumVersion(1))

_P_FRAME_PROPOSITION = (3, 1)
_P_FRAME_EVENT = (3, 2)
_P_FRAME_ANCHOR = (3, 3)
_P_FRAME_INTERVAL = (3, 4)
_P_FRAME_ASPECT = (3, 5)
_P_FRAME_CONTEXT = (3, 6)
_P_FRAME_OCCURRENCE = (3, 7)
_P_EVENT_KIND = (3, 8)
_P_CANDIDATE_KIND = (3, 9)
_P_ANCHOR_KIND = (3, 10)
_P_INTERVAL_START = (3, 11)
_P_INTERVAL_END = (3, 12)
_P_ASPECT_OBJECT = (3, 13)
_P_ASPECT_ANCHOR = (3, 14)
_P_ASPECT_INTERVAL = (3, 15)
_P_ASPECT_KIND = (3, 16)
_P_NARRATIVE_RELATION = (3, 17)
_P_REVISION = (3, 18)
_P_RETENTION = (3, 19)
_P_EPISTEMIC_STATE = (3, 20)
_P_FRAME_CONNECTOR = (3, 21)
_P_FRAME_INPUT_STRUCTURE = (3, 22)
_P_FRAME_RELATION_PREDICATE = (3, 23)
_P_FRAME_ROLE_BINDING = (3, 24)

_INPUT_STRUCTURE_DOMAIN = "event.time.input.structure.v1"
_GENERATION_FRAME_DOMAIN = "event.time.generation.frame.v1"


def _positive(payload: bytes) -> int:
    """把规范训练身份压成非零 63 位整数。"""
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (value & ((1 << 63) - 1)) or 1


def _key(namespace: str, value: object) -> tuple[int, ...]:
    """为开放课程值生成不携带表层正文的确定性整数键。"""
    return (_positive(f"{namespace}\x00{value}".encode("utf-8")),)


def event_time_source_ref(
        source_name: str, seed_id: str, logical_order: int) -> SourceRef:
    """建立 loader 与图 producer 共用的完整来源身份。"""
    if (not isinstance(source_name, str) or not source_name
            or not isinstance(seed_id, str) or not seed_id
            or type(logical_order) is not int or logical_order <= 0):
        raise ValueError("event-time 来源身份字段非法")
    source_id = _positive(
        f"{source_name}\x00{seed_id}\x00{logical_order}".encode("utf-8"))
    return SourceRef(
        _NS[0], source_id, logical_order,
        GLOBAL_OWNER_SCOPE, _VERSIONS)


def _predicate(code: tuple[int, ...], label: str = "") -> ObjectIdentity:
    """返回固定关系槽或开放课程关系的纯整数概念身份。"""
    suffix = _key("predicate", label) if label else ()
    return relation_concept_identity((*_NS, *code, *suffix))


def event_time_input_structure_identity(
        structure_key: tuple[int, ...],
        ) -> ObjectIdentity:
    """把完整输入拓扑映射为可复算身份；成员仍由 RoleBinding 边保存。"""
    if (type(structure_key) is not tuple or not structure_key
            or any(type(value) is not int or value < 0
                   for value in structure_key)):
        raise ValueError("event-time input structure key 必须是非空非负整数 tuple")
    digest = integer_tuple_fingerprint(
        structure_key, domain=_INPUT_STRUCTURE_DOMAIN)
    return structure_concept_identity((*_NS, 800, *digest))


@dataclass(frozen=True, slots=True)
class EventTimeRelationGenerationSeed:
    """从一个 active EVENT_TIME 事实派生的零复制生成绑定输入。"""

    proposition: ObjectIdentity
    predicate: ObjectIdentity
    context: ObjectIdentity
    connector: ObjectIdentity
    input_structure: ObjectIdentity
    role_bindings: tuple[ObjectIdentity, ...]
    event_refs: tuple[ObjectIdentity, ...]
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if self.proposition.object_kind != OBJECT_PROPOSITION:
            raise ValueError("event-time generation seed proposition 类型错误")
        if not isinstance(self.scope, ScopeIdentity) or self.scope.source is None:
            raise ValueError("event-time generation seed 必须保留来源化 scope")
        if semantic_source(self.proposition) != self.scope.source:
            raise ValueError("event-time generation seed proposition/source 漂移")
        for name in ("predicate", "context", "connector", "input_structure"):
            if not isinstance(getattr(self, name), ObjectIdentity):
                raise TypeError(f"event-time generation seed {name} 类型错误")
        if (type(self.role_bindings) is not tuple or not self.role_bindings
                or any(item.object_kind != OBJECT_ROLE_BINDING
                       for item in self.role_bindings)
                or len(set(self.role_bindings)) != len(self.role_bindings)):
            raise ValueError("event-time generation seed RoleBinding 不闭合")
        if (type(self.event_refs) is not tuple or len(self.event_refs) < 2
                or any(item.object_kind not in {OBJECT_EVENT, OBJECT_PROPOSITION}
                       for item in self.event_refs)
                or len(set(self.event_refs)) != len(self.event_refs)):
            raise ValueError("event-time generation seed Event 端点不闭合")

    def stable_key(self) -> tuple[int, ...]:
        """返回不复制表层、可完整恢复既有对象引用的整数键。"""
        result = [BRIDGE_VERSION]
        for identity in (
                self.proposition, self.predicate, self.context,
                self.connector, self.input_structure):
            key = identity.stable_key()
            result.extend((len(key), *key))
        result.append(len(self.role_bindings))
        for identity in sorted(
                self.role_bindings, key=ObjectIdentity.stable_key):
            key = identity.stable_key()
            result.extend((len(key), *key))
        result.append(len(self.event_refs))
        for identity in sorted(self.event_refs, key=ObjectIdentity.stable_key):
            key = identity.stable_key()
            result.extend((len(key), *key))
        scope_key = self.scope.stable_key()
        result.extend((len(scope_key), *scope_key))
        return tuple(result)


def _generation_frame_identity(
        seed: EventTimeRelationGenerationSeed,
        ) -> ObjectIdentity:
    """用紧凑派生身份索引 frame；完整成员由 statement 本体保存。"""
    digest = integer_tuple_fingerprint(
        seed.stable_key(), domain=_GENERATION_FRAME_DOMAIN)
    source = semantic_source(seed.proposition)
    return structure_concept_identity(
        (*_NS, 801, *digest),
        owner=source.owner,
        versions=source.versions,
    )


class EventTimeStructureTrainingRuntime:
    """把 LC-05 训练 split 幂等物化进同一 ``GraphOntology``。"""

    def __init__(
            self, context, course_paths: tuple[str | Path, ...],
            authoritative_source_keys: tuple[tuple[int, ...], ...] = ()):
        self.context = context
        self.course_paths = tuple(Path(item).resolve() for item in course_paths)
        self.authoritative_source_keys = frozenset(
            tuple(item) for item in authoritative_source_keys)
        if any(not key or any(type(value) is not int or value < 0
                              for value in key)
               for key in self.authoritative_source_keys):
            raise ValueError("event-time authoritative SourceRef 非法")
        context.graph_ontology.enable_physical_statement_projection()
        self._seen: set[tuple[int, ...]] = set()
        self._counts = {
            "seed_count": 0,
            "proposition_count": 0,
            "event_count": 0,
            "state_count": 0,
            "anchor_count": 0,
            "interval_count": 0,
            "aspect_count": 0,
            "narrative_relation_count": 0,
            "revision_count": 0,
            "retention_count": 0,
            "unknown_count": 0,
            "graph_statement_count": 0,
        }
        ontology = context.graph_ontology
        self._predicates = {
            name: ontology.materialize(_predicate(code))
            for name, code in (
                ("frame_proposition", _P_FRAME_PROPOSITION),
                ("frame_event", _P_FRAME_EVENT),
                ("frame_anchor", _P_FRAME_ANCHOR),
                ("frame_interval", _P_FRAME_INTERVAL),
                ("frame_aspect", _P_FRAME_ASPECT),
                ("frame_context", _P_FRAME_CONTEXT),
                ("frame_occurrence", _P_FRAME_OCCURRENCE),
                ("event_kind", _P_EVENT_KIND),
                ("candidate_kind", _P_CANDIDATE_KIND),
                ("anchor_kind", _P_ANCHOR_KIND),
                ("interval_start", _P_INTERVAL_START),
                ("interval_end", _P_INTERVAL_END),
                ("aspect_object", _P_ASPECT_OBJECT),
                ("aspect_anchor", _P_ASPECT_ANCHOR),
                ("aspect_interval", _P_ASPECT_INTERVAL),
                ("aspect_kind", _P_ASPECT_KIND),
                ("revision", _P_REVISION),
                ("retention", _P_RETENTION),
                ("epistemic_state", _P_EPISTEMIC_STATE),
                ("frame_connector", _P_FRAME_CONNECTOR),
                ("frame_input_structure", _P_FRAME_INPUT_STRUCTURE),
                ("frame_relation_predicate", _P_FRAME_RELATION_PREDICATE),
                ("frame_role_binding", _P_FRAME_ROLE_BINDING),
            )
        }

    def _relate(self, predicate, subject, obj, scope) -> None:
        """写一条来源化物理 statement 并更新真实计数。"""
        self.context.graph_ontology.relate(
            predicate, subject, obj, scope=scope,
            provenance_kind=EPI_STRUCTURED,
            content_version=BRIDGE_VERSION)
        self._counts["graph_statement_count"] += 1

    def _materialize(self, path: Path, seed) -> None:
        """物化一个 teacher seed 的全部结构，不保存答案表层。"""
        source = event_time_source_ref(
            path.name, seed.seed_id, seed.logical_order)
        if (self.authoritative_source_keys
                and source.stable_key() not in self.authoritative_source_keys):
            return
        scope = document_scope(source)
        ontology = self.context.graph_ontology
        frame = ontology.materialize(structure_concept_identity(
            (*_NS, 100, *_key("seed", seed.seed_id)),
            owner=source.owner, versions=source.versions))
        context = ontology.materialize(context_scope_identity(
            source, _key(
                "context", seed.surface_scope.to_value()["context_scope_key"])))
        self._relate(self._predicates["frame_context"], frame, context, scope)
        occurrence = ontology.materialize(occurrence_identity(
            source, start=0, end=len(seed.observed_text), ordinal=0))
        self._relate(
            self._predicates["frame_occurrence"], frame, occurrence, scope)
        proposition_by_id: dict[str, object] = {}
        event_by_id: dict[str, object] = {}
        for row_object in seed.event_state_candidates:
            row = row_object.to_value()
            proposition = proposition_by_id.get(row["proposition_key"])
            if proposition is None:
                proposition = ontology.materialize(proposition_identity(
                    source, _key("proposition", row["proposition_key"])))
                proposition_by_id[row["proposition_key"]] = proposition
                self._relate(
                    self._predicates["frame_proposition"],
                    frame, proposition, scope)
                self._counts["proposition_count"] += 1
            event = ontology.materialize(event_identity(
                source, (*_NS, 200, *_key("event", row["candidate_id"]))))
            event_by_id[row["candidate_id"]] = event
            self._relate(self._predicates["frame_event"], frame, event, scope)
            event_kind = ontology.materialize(concept_identity(
                (*_NS, 201, *_key("event-kind", row["semantic_kind"]))))
            self._relate(self._predicates["event_kind"], event, event_kind, scope)
            self._counts[
                "state_count" if row["semantic_kind"] == "STATE"
                else "event_count"] += 1
        candidate_kind = ontology.materialize(concept_identity(
            (*_NS, 202, *_key("candidate-kind", seed.candidate_kind))))
        self._relate(
            self._predicates["candidate_kind"], frame, candidate_kind, scope)
        anchor_by_id: dict[str, object] = {}
        for row_object in seed.temporal_anchors:
            row = row_object.to_value()
            anchor = ontology.materialize(structure_concept_identity(
                (*_NS, 300, *_key("anchor", row["anchor_id"])),
                owner=source.owner, versions=source.versions))
            anchor_by_id[row["anchor_id"]] = anchor
            self._relate(self._predicates["frame_anchor"], frame, anchor, scope)
            kind = ontology.materialize(concept_identity(
                (*_NS, 301, *_key("anchor-kind", row["anchor_kind"]))))
            self._relate(self._predicates["anchor_kind"], anchor, kind, scope)
            self._counts["anchor_count"] += 1
        interval_by_id: dict[str, object] = {}
        for row_object in seed.intervals:
            row = row_object.to_value()
            interval = ontology.materialize(structure_concept_identity(
                (*_NS, 400, *_key("interval", row["interval_id"])),
                owner=source.owner, versions=source.versions))
            interval_by_id[row["interval_id"]] = interval
            self._relate(
                self._predicates["frame_interval"], frame, interval, scope)
            self._relate(self._predicates["interval_start"], interval,
                         anchor_by_id[row["start_anchor_id"]], scope)
            self._relate(self._predicates["interval_end"], interval,
                         anchor_by_id[row["end_anchor_id"]], scope)
            self._counts["interval_count"] += 1
        for row_object in seed.aspect_profiles:
            row = row_object.to_value()
            aspect = ontology.materialize(structure_concept_identity(
                (*_NS, 500, *_key("aspect", row["profile_id"])),
                owner=source.owner, versions=source.versions))
            self._relate(self._predicates["frame_aspect"], frame, aspect, scope)
            self._relate(self._predicates["aspect_object"], aspect,
                         event_by_id[row["object_id"]], scope)
            if row["anchor_id"]:
                self._relate(self._predicates["aspect_anchor"], aspect,
                             anchor_by_id[row["anchor_id"]], scope)
            if row["interval_id"]:
                self._relate(self._predicates["aspect_interval"], aspect,
                             interval_by_id[row["interval_id"]], scope)
            kind = ontology.materialize(concept_identity(
                (*_NS, 501, *_key("aspect-kind", row["aspect_kind"]))))
            self._relate(self._predicates["aspect_kind"], aspect, kind, scope)
            self._counts["aspect_count"] += 1
        for row_object in seed.narrative_relations:
            row = row_object.to_value()
            predicate = ontology.materialize(_predicate(
                _P_NARRATIVE_RELATION, row["relation_kind"]))
            self._relate(predicate, event_by_id[row["from_object_id"]],
                         event_by_id[row["to_object_id"]], scope)
            self._counts["narrative_relation_count"] += 1
        for field, predicate_name, count_name in (
                (seed.supersedes_seed_id, "revision", "revision_count"),
                (seed.retention_anchor_id, "retention", "retention_count")):
            if field:
                target = ontology.materialize(concept_identity(
                    (*_NS, 600, *_key(predicate_name, field))))
                self._relate(self._predicates[predicate_name], frame, target, scope)
                self._counts[count_name] += 1
        state = ontology.materialize(concept_identity(
            (*_NS, 700, *_key("expected-state", seed.expected_state))))
        self._relate(
            self._predicates["epistemic_state"], frame, state, scope)
        if seed.expected_state == "UNKNOWN":
            self._counts["unknown_count"] += 1
        self._counts["seed_count"] += 1

    def consume(self) -> dict[str, int]:
        """按来源路径和课程逻辑序消费 teacher split，重复调用无新增。"""
        for path in self.course_paths:
            for seed in read_authored_event_time_aspect_seeds(path):
                if seed.label_owner != "teacher":
                    continue
                marker = _key("seed", seed.seed_id)
                if marker in self._seen:
                    continue
                self._seen.add(marker)
                self._materialize(path, seed)
        return self.report()

    def report(self) -> dict[str, int]:
        """返回当前 runtime 实际写入的结构计数。"""
        return {
            "bridge_version": BRIDGE_VERSION,
            "course_count": len(self.course_paths),
            **{name: int(value) for name, value in self._counts.items()},
        }


def build_event_time_structure_training_runtime(
        context, course_paths: Iterable[str | Path],
        authoritative_source_keys: Iterable[tuple[int, ...]] = (),
        ) -> EventTimeStructureTrainingRuntime:
    """建立并立即消费绑定当前正式图的 LC-05 producer。"""
    runtime = EventTimeStructureTrainingRuntime(
        context,
        tuple(Path(item).resolve() for item in course_paths),
        tuple(tuple(item) for item in authoritative_source_keys))
    runtime.consume()
    return runtime


def materialize_event_time_relation_generation_bindings(
        context,
        seeds: Iterable[EventTimeRelationGenerationSeed],
        ) -> dict[str, int]:
    """把 active 事实、输入拓扑与既有 connector 接成同图 Event/Time frame。

    本函数不复制 Proposition、RoleBinding、Event、Context 或 connector，也不读取
    来源正文。新增 frame 只是这些一等对象之间的来源化结构投影。
    """
    ordered = tuple(sorted(set(seeds), key=lambda item: item.stable_key()))
    if any(not isinstance(item, EventTimeRelationGenerationSeed)
           for item in ordered):
        raise TypeError("event-time generation bindings 必须是 seed iterable")
    ontology = context.graph_ontology
    ontology.enable_physical_statement_projection()
    predicates = {
        name: ontology.materialize(_predicate(code))
        for name, code in (
            ("frame_proposition", _P_FRAME_PROPOSITION),
            ("frame_event", _P_FRAME_EVENT),
            ("frame_context", _P_FRAME_CONTEXT),
            ("candidate_kind", _P_CANDIDATE_KIND),
            ("frame_connector", _P_FRAME_CONNECTOR),
            ("frame_input_structure", _P_FRAME_INPUT_STRUCTURE),
            ("frame_relation_predicate", _P_FRAME_RELATION_PREDICATE),
            ("frame_role_binding", _P_FRAME_ROLE_BINDING),
        )
    }
    statement_count = 0
    reused_object_keys: set[tuple[int, ...]] = set()
    for seed in ordered:
        frame_identity = _generation_frame_identity(seed)
        frame = ontology.materialize(frame_identity)
        relation_kind = ontology.materialize(concept_identity(
            (*_NS, 802, *integer_tuple_fingerprint(
                seed.predicate.stable_key(),
                domain="event.time.relation.kind.v1"))))
        targets = (
            ("frame_proposition", seed.proposition),
            ("frame_context", seed.context),
            ("candidate_kind", ontology.identity_of(relation_kind)),
            ("frame_connector", seed.connector),
            ("frame_input_structure", seed.input_structure),
            ("frame_relation_predicate", seed.predicate),
            *(('frame_event', item) for item in seed.event_refs),
            *(('frame_role_binding', item) for item in seed.role_bindings),
        )
        for predicate_name, identity in targets:
            target = ontology.materialize(identity)
            ontology.relate(
                predicates[predicate_name], frame, target,
                scope=seed.scope,
                provenance_kind=EPI_STRUCTURED,
                content_version=BRIDGE_VERSION,
            )
            statement_count += 1
            reused_object_keys.add(identity.stable_key())
    return {
        "bridge_version": BRIDGE_VERSION,
        "generation_binding_count": len(ordered),
        "generation_binding_statement_count": statement_count,
        "reused_object_count": len(reused_object_keys),
        "semantic_object_copy_count": 0,
    }


def event_time_structure_predicate_identities() -> dict[str, ObjectIdentity]:
    """返回 reader/query 可复用的固定谓词身份。"""
    return {
        name: _predicate(code)
        for name, code in (
            ("frame_proposition", _P_FRAME_PROPOSITION),
            ("frame_event", _P_FRAME_EVENT),
            ("frame_anchor", _P_FRAME_ANCHOR),
            ("frame_interval", _P_FRAME_INTERVAL),
            ("frame_aspect", _P_FRAME_ASPECT),
            ("frame_context", _P_FRAME_CONTEXT),
            ("frame_occurrence", _P_FRAME_OCCURRENCE),
            ("event_kind", _P_EVENT_KIND),
            ("candidate_kind", _P_CANDIDATE_KIND),
            ("anchor_kind", _P_ANCHOR_KIND),
            ("interval_start", _P_INTERVAL_START),
            ("interval_end", _P_INTERVAL_END),
            ("aspect_object", _P_ASPECT_OBJECT),
            ("aspect_anchor", _P_ASPECT_ANCHOR),
            ("aspect_interval", _P_ASPECT_INTERVAL),
            ("aspect_kind", _P_ASPECT_KIND),
            ("revision", _P_REVISION),
            ("retention", _P_RETENTION),
            ("epistemic_state", _P_EPISTEMIC_STATE),
            ("frame_connector", _P_FRAME_CONNECTOR),
            ("frame_input_structure", _P_FRAME_INPUT_STRUCTURE),
            ("frame_relation_predicate", _P_FRAME_RELATION_PREDICATE),
            ("frame_role_binding", _P_FRAME_ROLE_BINDING),
        )
    }


def event_time_narrative_predicate_prefix() -> tuple[int, ...]:
    """返回开放 narrative relation 谓词共享的整数身份前缀。"""
    return (*_NS, *_P_NARRATIVE_RELATION)


__all__ = [
    "BRIDGE_VERSION",
    "EventTimeRelationGenerationSeed",
    "EventTimeStructureTrainingRuntime",
    "build_event_time_structure_training_runtime",
    "event_time_input_structure_identity",
    "event_time_narrative_predicate_prefix",
    "event_time_source_ref",
    "event_time_structure_predicate_identities",
    "materialize_event_time_relation_generation_bindings",
]
