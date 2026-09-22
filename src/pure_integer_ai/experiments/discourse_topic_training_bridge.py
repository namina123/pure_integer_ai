"""W-08 discourse/topic authored data -> shared integer graph owner.

The authored JSONL files are only consumed while a training ``TrainContext``
is being built.  This bridge materializes source-bound propositions, discourse
relations, topic/information/QUD objects, and revision/reference links into the
same ``GraphOntology`` used by the Core graph.  It never stores an answer
surface as a generation shortcut and never creates a second semantic database.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    concept_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_authored_discourse_compile import (
    compile_discourse_seed,
)
from pure_integer_ai.experiments.ph2_authored_discourse_course import (
    read_authored_discourse_seeds,
)
from pure_integer_ai.experiments.ph2_authored_discourse_information_course import (
    information_source_ref,
    read_authored_discourse_information_seeds,
)
from pure_integer_ai.experiments.trained_discourse_topic_topology import (
    DiscourseTopicGraphTopology,
    read_discourse_topic_topology,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED


BRIDGE_VERSION = 1
_NS = (21601, BRIDGE_VERSION)

_P_FRAME_PROPOSITION = (3, 1)
_P_FRAME_CONTEXT = (3, 2)
_P_FRAME_INFORMATION = (3, 3)
_P_FRAME_QUD = (3, 4)
_P_FRAME_PRESUPPOSITION = (3, 5)
_P_FRAME_REFERENCE = (3, 6)
_P_FRAME_REVISION = (3, 7)
_P_TOPIC = (3, 8)
_P_FOCUS = (3, 9)
_P_STATUS = (3, 10)
_P_OCCURRENCE = (3, 11)
_P_SCOPE = (3, 12)


def _positive(value: bytes) -> int:
    result = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result or 1


def _key(namespace: str, value: object) -> tuple[int, ...]:
    """Create a deterministic, non-surface semantic key at train time."""
    encoded = f"{namespace}\x00{value}".encode("utf-8")
    return (_positive(encoded),)


def _predicate(code: tuple[int, ...], label: str = ""):
    suffix = _key("predicate", label) if label else ()
    return relation_concept_identity((*_NS, *code, *suffix))


@dataclass(frozen=True, slots=True)
class DiscourseTopicTrainingReport:
    course_count: int
    seed_count: int
    proposition_count: int
    relation_count: int
    information_count: int
    qud_count: int
    presupposition_count: int
    occurrence_count: int
    reference_count: int
    revision_count: int
    graph_statement_count: int

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name))
                for name in self.__dataclass_fields__}


class DiscourseTopicTrainingRuntime:
    """Idempotent W-08 producer bound to one formal ``TrainContext``."""

    def __init__(
            self,
            context,
            course_paths: tuple[str | Path, ...],
            authoritative_source_keys: tuple[tuple[int, ...], ...] = (),
            ):
        self.context = context
        self.course_paths = tuple(Path(item).resolve() for item in course_paths)
        self.authoritative_source_keys = tuple(sorted(set(
            tuple(item) for item in authoritative_source_keys)))
        if any(type(key) is not tuple or not key
               or any(type(value) is not int or value < 0 for value in key)
               for key in self.authoritative_source_keys):
            raise ValueError("authoritative_source_keys 必须是整数 SourceRef stable key")
        # W-08 is a publication-facing graph owner: retain the physical
        # statement index so independent readers can recover the materialized
        # discourse topology without replaying course files.
        context.graph_ontology.enable_physical_statement_projection()
        self._seen: set[tuple[int, ...]] = set()
        self._counts = {
            "seed_count": 0,
            "proposition_count": 0,
            "relation_count": 0,
            "information_count": 0,
            "qud_count": 0,
            "presupposition_count": 0,
            "occurrence_count": 0,
            "reference_count": 0,
            "revision_count": 0,
            "graph_statement_count": 0,
        }
        ontology = context.graph_ontology
        self._predicates = {
            name: ontology.materialize(_predicate(code))
            for name, code in (
                ("frame_proposition", _P_FRAME_PROPOSITION),
                ("frame_context", _P_FRAME_CONTEXT),
                ("frame_information", _P_FRAME_INFORMATION),
                ("frame_qud", _P_FRAME_QUD),
                ("frame_presupposition", _P_FRAME_PRESUPPOSITION),
                ("frame_reference", _P_FRAME_REFERENCE),
                ("frame_revision", _P_FRAME_REVISION),
                ("topic", _P_TOPIC),
                ("focus", _P_FOCUS),
                ("status", _P_STATUS),
                ("occurrence", _P_OCCURRENCE),
                ("scope", _P_SCOPE),
            )
        }

    @property
    def topology(self) -> DiscourseTopicGraphTopology:
        """返回当前 formal graph 中 W-08 拓扑的只读恢复快照。"""
        return read_discourse_topic_topology(
            self.context.graph_ontology,
            discourse_topic_predicate_identities(),
        )

    def _relate(self, predicate, subject, obj, scope) -> None:
        self.context.graph_ontology.relate(
            predicate, subject, obj, scope=scope,
            provenance_kind=EPI_STRUCTURED,
            content_version=BRIDGE_VERSION,
        )
        self._counts["graph_statement_count"] += 1

    def _materialize_revision(self, path: Path, seed) -> None:
        compiled = compile_discourse_seed(seed)
        payload = compiled.observation_payload.to_value()
        source_key = payload.get("source_ref_key")
        source = SourceRef.from_stable_key(tuple(source_key))
        if (self.authoritative_source_keys
                and source.stable_key() not in self.authoritative_source_keys):
            return
        scope = document_scope(source)
        frame = structure_concept_identity(
            (*_NS, 100, *_key("seed", seed.seed_id)),
            owner=source.owner, versions=source.versions,
        )
        frame_ref = self.context.graph_ontology.materialize(frame)
        for row in payload.get("occurrences", ()):
            occurrence = occurrence_identity(
                source, start=row["start"], end=row["end"],
                ordinal=row["ordinal"],
            )
            occurrence_ref = self.context.graph_ontology.materialize(occurrence)
            self._relate(self._predicates["occurrence"], frame_ref,
                         occurrence_ref, scope)
            self._counts["occurrence_count"] += 1
        reference = payload.get("reference_plan")
        if isinstance(reference, dict):
            reference_obj = concept_identity(
                (*_NS, 110, *_key("reference", seed.seed_id)))
            reference_ref = self.context.graph_ontology.materialize(reference_obj)
            self._relate(self._predicates["frame_reference"], frame_ref,
                         reference_ref, scope)
            self._counts["reference_count"] += 1
        revision = payload.get("parser_revision_plan")
        if isinstance(revision, dict):
            revision_obj = concept_identity(
                (*_NS, 120, *_key("revision", seed.seed_id)))
            revision_ref = self.context.graph_ontology.materialize(revision_obj)
            self._relate(self._predicates["frame_revision"], frame_ref,
                         revision_ref, scope)
            self._counts["revision_count"] += 1
        self._counts["seed_count"] += 1

    def _materialize_information(self, path: Path, seed) -> None:
        payload = seed.observation_payload().to_value()
        source = information_source_ref(
            path.name, seed.seed_id, seed.logical_order)
        if (self.authoritative_source_keys
                and source.stable_key() not in self.authoritative_source_keys):
            return
        scope = document_scope(source)
        frame = structure_concept_identity(
            (*_NS, 200, *_key("seed", seed.seed_id)),
            owner=source.owner, versions=source.versions,
        )
        frame_ref = self.context.graph_ontology.materialize(frame)
        context_key = _key("context", payload["surface_scope"]["context_scope_key"])
        context_obj = context_scope_identity(source, context_key)
        context_ref = self.context.graph_ontology.materialize(context_obj)
        self._relate(self._predicates["frame_context"], frame_ref,
                     context_ref, scope)
        for row in payload.get("information_structure", ()):
            information = concept_identity(
                (*_NS, 310, *_key("information", row["information_id"])))
            information_ref = self.context.graph_ontology.materialize(information)
            self._relate(self._predicates["frame_information"], frame_ref,
                         information_ref, scope)
            self._counts["information_count"] += 1
        for row in payload.get("qud_candidates", ()):
            qud = concept_identity(
                (*_NS, 320, *_key("qud", row["qud_id"])))
            qud_ref = self.context.graph_ontology.materialize(qud)
            self._relate(self._predicates["frame_qud"], frame_ref,
                         qud_ref, scope)
            self._counts["qud_count"] += 1
        for ordinal, row in enumerate(payload.get("presupposition_obligations", ())):
            obligation_id = row.get("obligation_id", ordinal + 1)
            obligation = concept_identity(
                (*_NS, 330, *_key("obligation", obligation_id)))
            obligation_ref = self.context.graph_ontology.materialize(obligation)
            self._relate(self._predicates["frame_presupposition"], frame_ref,
                         obligation_ref, scope)
            self._counts["presupposition_count"] += 1
        for row in payload.get("proposition_candidates", ()):
            proposition = proposition_identity(
                source, _key("proposition", row["proposition_id"]),
            )
            proposition_ref = self.context.graph_ontology.materialize(proposition)
            self._relate(self._predicates["frame_proposition"], frame_ref,
                         proposition_ref, scope)
            self._counts["proposition_count"] += 1
        proposition_by_id = {
            row["proposition_id"]: proposition_identity(
                source, _key("proposition", row["proposition_id"]),
            ) for row in payload.get("proposition_candidates", ())
        }
        for row in payload.get("discourse_relations", ()):
            subject = proposition_by_id.get(row["source_proposition_id"])
            obj = proposition_by_id.get(row["target_proposition_id"])
            if subject is None or obj is None:
                continue
            subject_ref = self.context.graph_ontology.resolve(subject)
            object_ref = self.context.graph_ontology.resolve(obj)
            if subject_ref is None or object_ref is None:
                continue
            predicate = self.context.graph_ontology.materialize(
                _predicate((4, 1), row["relation_kind"]))
            self._relate(predicate, subject_ref, object_ref, scope)
            self._counts["relation_count"] += 1
        for row in payload.get("information_structure", ()):
            proposition = proposition_by_id.get(row["proposition_id"])
            if proposition is None:
                continue
            proposition_ref = self.context.graph_ontology.resolve(proposition)
            if proposition_ref is None:
                continue
            for name, code, value in (
                    ("topic", _P_TOPIC, row.get("topic_key")),
                    ("focus", _P_FOCUS, row.get("focus_key")),
                    ("status", _P_STATUS, row.get("status_family"))):
                if value:
                    target = self.context.graph_ontology.materialize(
                        concept_identity((*_NS, 300, *code, *_key(name, value))))
                    self._relate(self._predicates[name], proposition_ref,
                                 target, scope)
        self._counts["seed_count"] += 1

    def consume(self) -> dict[str, int]:
        for path in self.course_paths:
            if path.name == "authored_discourse_information_seed_v1.jsonl.sample":
                seeds = read_authored_discourse_information_seeds(path)
                for seed in seeds:
                    if seed.label_owner != "teacher":
                        continue
                    marker = _key("seed", seed.seed_id)
                    if marker in self._seen:
                        continue
                    self._seen.add(marker)
                    self._materialize_information(path, seed)
            elif path.name == "authored_discourse_revision_seed_v1.jsonl.sample":
                seeds = read_authored_discourse_seeds(path)
                for seed in seeds:
                    if seed.label_owner != "teacher":
                        continue
                    marker = _key("seed", seed.seed_id)
                    if marker in self._seen:
                        continue
                    self._seen.add(marker)
                    self._materialize_revision(path, seed)
        return self.report()

    def report(self) -> dict[str, int]:
        return {
            "bridge_version": BRIDGE_VERSION,
            "course_count": len(self.course_paths),
            **{name: int(value) for name, value in self._counts.items()},
        }


def build_discourse_topic_training_runtime(
        context,
        course_paths: Iterable[str | Path],
        authoritative_source_keys: Iterable[tuple[int, ...]] = (),
        ):
    runtime = DiscourseTopicTrainingRuntime(
        context,
        tuple(Path(item).resolve() for item in course_paths),
        tuple(tuple(item) for item in authoritative_source_keys),
    )
    runtime.consume()
    return runtime


def discourse_topic_predicate_identities() -> dict[str, ObjectIdentity]:
    """返回 W-08 固定结构 predicate 的完整整数身份。"""
    return {
        name: _predicate(code)
        for name, code in (
            ("frame_proposition", _P_FRAME_PROPOSITION),
            ("frame_context", _P_FRAME_CONTEXT),
            ("frame_information", _P_FRAME_INFORMATION),
            ("frame_qud", _P_FRAME_QUD),
            ("frame_presupposition", _P_FRAME_PRESUPPOSITION),
            ("frame_reference", _P_FRAME_REFERENCE),
            ("frame_revision", _P_FRAME_REVISION),
            ("topic", _P_TOPIC),
            ("focus", _P_FOCUS),
            ("status", _P_STATUS),
            ("occurrence", _P_OCCURRENCE),
            ("scope", _P_SCOPE),
        )
    }


__all__ = [
    "BRIDGE_VERSION",
    "DiscourseTopicTrainingReport",
    "DiscourseTopicTrainingRuntime",
    "build_discourse_topic_training_runtime",
    "discourse_topic_predicate_identities",
]
