"""来源化 Grounded Answer 结构到共享整数图的训练桥。

课程正文只在编译期读取；正式图只保留 episode、claim、Evidence、response act、
scope 和 reference/revision 的整数身份与关系，不把答案 occurrence 当作运行时回放。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.shared.graph_ontology import relation_concept_identity
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle, concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity, proposition_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedAnswerEpisode, read_grounded_answer_episodes,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED


BRIDGE_VERSION = 1
_SOURCE_KIND = 21620
_NS = (21620, BRIDGE_VERSION)

_PREDICATE_CODES = {
    "episode": (1, 1), "question": (1, 2), "claim": (1, 3),
    "evidence": (1, 4), "response_act": (1, 5), "scope": (1, 6),
    "reference": (1, 7), "antecedent": (1, 8), "referring": (1, 9),
    "surface_structure": (1, 10), "cluster": (1, 11),
    "source": (1, 12), "support": (1, 13), "refute": (1, 14),
    "typed_intent": (1, 15), "realization": (1, 16),
    "accepted": (1, 17), "rejected": (1, 18), "turn": (1, 19),
}


def _positive(value: object) -> int:
    payload = repr(value).encode("utf-8")
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result or 1


def _key(label: str, value: object) -> tuple[int, ...]:
    return (_positive((label, value)),)


def _source(episode: GroundedAnswerEpisode, ordinal: int) -> SourceRef:
    return SourceRef(
        _SOURCE_KIND, _positive((episode.episode_id, ordinal)), ordinal,
        GLOBAL_OWNER_SCOPE, VersionBundle(),
    )


def _predicate(ontology, name: str):
    return ontology.materialize(relation_concept_identity((*_NS, *_PREDICATE_CODES[name])))


class GroundedAnswerGraphTrainingRuntime:
    """幂等地将 Grounded Answer 的结构字段追加到当前 TrainContext 图。"""

    def __init__(self, context, course_paths: tuple[str | Path, ...]):
        self.context = context
        self.course_paths = tuple(Path(item).resolve() for item in course_paths)
        self.episode_count = 0
        self.claim_count = 0
        self.evidence_count = 0
        self.reference_count = 0
        self.statement_count = 0
        self.source_count = 0
        self.accepted_realization_count = 0
        self.rejected_realization_count = 0
        self.turn_count = 0
        self._seen: set[tuple[int, ...]] = set()
        ontology = context.graph_ontology
        ontology.enable_physical_statement_projection()
        self._predicates = {name: _predicate(ontology, name) for name in _PREDICATE_CODES}

    def _relate(self, predicate, subject, obj, scope) -> None:
        self.context.graph_ontology.relate(
            predicate, subject, obj, scope=scope,
            provenance_kind=EPI_STRUCTURED, content_version=BRIDGE_VERSION)
        self.statement_count += 1

    def consume(self) -> dict[str, int]:
        for path in self.course_paths:
            if path.name != "grounded_answer_train_v1.jsonl.sample":
                continue
            for ordinal, episode in enumerate(read_grounded_answer_episodes(path), 1):
                marker = _key("episode", episode.episode_id)
                if marker in self._seen:
                    continue
                self._seen.add(marker)
                source = _source(episode, ordinal)
                scope = document_scope(source)
                ontology = self.context.graph_ontology
                episode_ref = ontology.materialize(structure_concept_identity(
                    (*_NS, 100, *_key("episode", episode.episode_id)),
                    owner=source.owner, versions=source.versions))
                question_ref = ontology.materialize(structure_concept_identity(
                    (*_NS, 101, *_key("question", episode.episode_id)),
                    owner=source.owner, versions=source.versions))
                self._relate(self._predicates["question"], episode_ref, question_ref, scope)
                act_ref = ontology.materialize(concept_identity(
                    (*_NS, 200, *_key("response_act", episode.question.answer_plan.response_act))))
                self._relate(self._predicates["response_act"], question_ref, act_ref, scope)
                intent_ref = ontology.materialize(concept_identity(
                    (*_NS, 201, *_key("typed_intent", episode.question.typed_intent))))
                self._relate(self._predicates["typed_intent"], question_ref, intent_ref, scope)
                scope_ref = ontology.materialize(context_scope_identity(
                    source, _key("response_scope", episode.question.response_scope_id)))
                self._relate(self._predicates["scope"], question_ref, scope_ref, scope)
                claim_refs = {}
                for evidence in episode.question.evidence:
                    claim_ref = claim_refs.get(evidence.proposition_id)
                    if claim_ref is None:
                        claim_ref = ontology.materialize(proposition_identity(
                            source, _key("claim", evidence.proposition_id)))
                        claim_refs[evidence.proposition_id] = claim_ref
                        self._relate(self._predicates["claim"], question_ref, claim_ref, scope)
                        self.claim_count += 1
                    evidence_ref = ontology.materialize(concept_identity(
                        (*_NS, 300, *_key("evidence", evidence.evidence_id))))
                    self._relate(self._predicates["evidence"], claim_ref, evidence_ref, scope)
                    source_ref = ontology.materialize(concept_identity(
                        (*_NS, 301, *_key("source", evidence.source_id))))
                    self._relate(self._predicates["source"], evidence_ref, source_ref, scope)
                    if evidence.support:
                        self._relate(self._predicates["support"], claim_ref, evidence_ref, scope)
                    if evidence.refute:
                        self._relate(self._predicates["refute"], claim_ref, evidence_ref, scope)
                    self.evidence_count += 1
                    self.source_count += 1
                if episode.reference_course is not None:
                    reference_ref = ontology.materialize(concept_identity(
                        (*_NS, 400, *_key("reference", episode.episode_id))))
                    self._relate(self._predicates["reference"], question_ref, reference_ref, scope)
                    antecedent = claim_refs.get(episode.reference_course.antecedent_proposition_id)
                    referring = claim_refs.get(episode.reference_course.referring_proposition_id)
                    if antecedent is not None:
                        self._relate(self._predicates["antecedent"], reference_ref, antecedent, scope)
                    if referring is not None:
                        self._relate(self._predicates["referring"], reference_ref, referring, scope)
                    self.reference_count += 1
                for realization in episode.surfaces.accepted:
                    realization_ref = ontology.materialize(concept_identity(
                        (*_NS, 600, *_key("realization", realization.realization_id))))
                    self._relate(self._predicates["realization"], question_ref, realization_ref, scope)
                    accepted_ref = ontology.materialize(concept_identity(
                        (*_NS, 601, *_key("accepted", realization.realization_id))))
                    self._relate(self._predicates["accepted"], realization_ref, accepted_ref, scope)
                    # Reuse the episode response-act identity; realizations
                    # must not create a second semantic object for the same act.
                    self._relate(self._predicates["response_act"], realization_ref, act_ref, scope)
                    self.accepted_realization_count += 1
                for rejected in episode.surfaces.rejected:
                    realization = rejected.realization
                    realization_ref = ontology.materialize(concept_identity(
                        (*_NS, 610, *_key("realization", realization.realization_id))))
                    self._relate(self._predicates["realization"], question_ref, realization_ref, scope)
                    rejected_ref = ontology.materialize(concept_identity(
                        (*_NS, 611, *_key("rejected", realization.realization_id))))
                    self._relate(self._predicates["rejected"], realization_ref, rejected_ref, scope)
                    self._relate(self._predicates["response_act"], realization_ref, act_ref, scope)
                    self.rejected_realization_count += 1
                for turn in episode.dialogue.turns:
                    turn_ref = ontology.materialize(concept_identity(
                        (*_NS, 700, *_key("turn", (episode.episode_id, turn.turn_id)))))
                    self._relate(self._predicates["turn"], question_ref, turn_ref, scope)
                    self.turn_count += 1
                for cluster_name in ("source", "proposition", "question_construction", "paraphrase"):
                    target = ontology.materialize(concept_identity(
                        (*_NS, 500, *_key(cluster_name, getattr(episode.clusters, cluster_name)))))
                    self._relate(self._predicates["cluster"], question_ref, target, scope)
                self.episode_count += 1
        return self.report()

    def report(self) -> dict[str, int]:
        return {
            "bridge_version": BRIDGE_VERSION,
            "course_count": len(self.course_paths),
            "episode_count": self.episode_count,
            "claim_count": self.claim_count,
            "evidence_count": self.evidence_count,
            "reference_count": self.reference_count,
            "graph_statement_count": self.statement_count,
            "source_count": self.source_count,
            "accepted_realization_count": self.accepted_realization_count,
            "rejected_realization_count": self.rejected_realization_count,
            "turn_count": self.turn_count,
            "semantic_object_copy_count": 0,
        }


def build_grounded_answer_graph_runtime(context, course_paths: Iterable[str | Path]):
    runtime = GroundedAnswerGraphTrainingRuntime(context, tuple(course_paths))
    runtime.consume()
    return runtime


__all__ = ["BRIDGE_VERSION", "GroundedAnswerGraphTrainingRuntime", "build_grounded_answer_graph_runtime"]
