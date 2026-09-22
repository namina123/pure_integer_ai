"""Integer-only discourse topic candidates for the default graph QueryState.

The input projector and Interaction Memory already carry category-7 topic
candidate keys.  This module gives that path an explicit production contract:
it decodes the complete topic identity, retains the relation's graph routes,
and reports which active Memory hypotheses correspond to the current input.
It never reads surface text, successor rows, posting tables, or a language
vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER,
    QueryInputStructure,
    STRUCTURE_CLOSED,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION,
    MEMORY_CATEGORY_DISCOURSE,
    frame_key,
    memory_candidate_category,
    memory_input_candidate_keys,
    memory_topic_candidate,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_CORE, SPACE_DIALOGUE, SPACE_MEMORY, QueryState,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity


DISCOURSE_TOPIC_RUNTIME_VERSION = 91512


@dataclass(frozen=True, slots=True)
class DiscourseTopicCandidate:
    """A complete topic candidate derived from one input relation candidate."""

    candidate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    structure_key: tuple[int, ...]
    carrier_key: tuple[int, ...]
    member_keys: tuple[tuple[int, ...], ...]
    route_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        values = (
            self.candidate_key,
            self.proposition_key,
            self.predicate_key,
            self.structure_key,
            self.carrier_key,
        )
        if any(type(item) is not tuple or not item
               or any(type(value) is not int or value < 0 for value in item)
               for item in values):
            raise ValueError("discourse topic identity must be non-empty integer tuples")
        if type(self.member_keys) is not tuple:
            raise ValueError("discourse topic members must be a tuple")
        if self.member_keys != tuple(sorted(set(self.member_keys))):
            raise ValueError("discourse topic members must be sorted and unique")
        if any(type(item) is not tuple or not item
               or any(type(value) is not int or value < 0 for value in item)
               for item in self.member_keys):
            raise ValueError("discourse topic members must be integer tuples")
        if type(self.route_keys) is not tuple:
            raise ValueError("discourse topic routes must be a tuple")
        if self.route_keys != tuple(sorted(set(self.route_keys))):
            raise ValueError("discourse topic routes must be sorted and unique")
        if any(type(item) is not tuple or not item
               or any(type(value) is not int or value < 0 for value in item)
               for item in self.route_keys):
            raise ValueError("discourse topic routes must be integer tuples")

    def stable_key(self) -> tuple[int, ...]:
        """Return a complete integer identity, including graph routes."""
        return frame_key(
            DISCOURSE_TOPIC_RUNTIME_VERSION,
            self.candidate_key,
            self.proposition_key,
            self.predicate_key,
            self.structure_key,
            self.carrier_key,
            *self.member_keys,
            frame_key(1, *self.route_keys),
        )


def derive_discourse_topic_candidates(
        input_structure: QueryInputStructure,
        ) -> tuple[DiscourseTopicCandidate, ...]:
    """Derive category-7 topic candidates from the current typed structure.

    The candidate key is obtained from the existing Memory candidate compiler;
    this adapter only makes the category explicit for the shared QueryState.
    """
    if not isinstance(input_structure, QueryInputStructure):
        raise TypeError("input_structure must be QueryInputStructure")
    relation_by_key = {}
    for relation in input_structure.relation_candidates:
        graph_closed = (
            relation.state == STRUCTURE_CLOSED
            and relation.required_count >= 2
            and relation.required_count == relation.support_count
            and relation.predicate_coverage == 1)
        if not graph_closed:
            continue
        members = tuple(sorted(set(
            frame_key(
                1,
                (item.object_kind, item.member_ordinal),
                item.candidate_key,
                item.role_key,
                item.source_ref,
            )
            for item in input_structure.semantic_candidates
            if item.proposition_key == relation.proposition_key
            and item.source_ref == relation.source_ref
            and item.projection_kind == PROJECTION_FILLER
        )))
        topic_key = frame_key(
            MEMORY_CANDIDATE_VERSION,
            (MEMORY_CATEGORY_DISCOURSE,),
            memory_topic_candidate_key(
                relation.proposition_key,
                relation.predicate_key,
                relation.structure_key,
                relation.carrier_key,
                members,
            ),
        )
        relation_by_key[topic_key] = relation

    # Confirm these keys are exactly those emitted by the authoritative
    # compiler; no hand-built candidate may enter the production path.
    compiled = {
        key for key in memory_input_candidate_keys(input_structure)
        if (memory_candidate_category(key) == MEMORY_CATEGORY_DISCOURSE
            and key[0] == MEMORY_CANDIDATE_VERSION)
    }
    if set(relation_by_key) != compiled:
        raise ValueError("discourse topic compiler output drifted from Memory keys")

    result = []
    for key in sorted(compiled):
        topic = memory_topic_candidate(key)
        if topic is None:
            raise ValueError("category-7 candidate is not a complete topic")
        result.append(DiscourseTopicCandidate(
            key,
            topic.proposition_key,
            topic.predicate_key,
            topic.structure_key,
            topic.carrier_key,
            topic.members,
            topic.content_routes(),
        ))
    return tuple(result)


def matched_discourse_topic_hypothesis_keys(
        hypotheses: tuple[object, ...],
        candidates: tuple[DiscourseTopicCandidate, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """Return active Memory Hypothesis identities for current topic candidates."""
    candidate_keys = {item.candidate_key for item in candidates}
    result = set()
    for hypothesis in hypotheses:
        candidate_key = getattr(hypothesis, "candidate_key", None)
        kind = getattr(hypothesis, "hypothesis_kind", ())
        hypothesis_key = getattr(hypothesis, "hypothesis_key", None)
        if (type(candidate_key) is tuple
                and candidate_key in candidate_keys
                and type(kind) is tuple
                and kind[-1:] == (MEMORY_CATEGORY_DISCOURSE,)
                and type(hypothesis_key) is tuple):
            result.add(hypothesis_key)
    return tuple(sorted(result))


def restore_discourse_topic_candidates(
        hypotheses: tuple[object, ...],
        candidates: tuple[DiscourseTopicCandidate, ...] = (),
        ) -> tuple[DiscourseTopicCandidate, ...]:
    """Recover topic candidates from the active Memory hypothesis graph.

    A cold topic is eligible only when its complete category-7 integer
    candidate was already restored by the Memory observation closure.  This
    keeps topic recovery on the same O/H/E path as the current query; no
    surface, posting or successor lookup can manufacture a candidate.
    """
    if type(hypotheses) is not tuple:
        raise TypeError("discourse topic hypotheses must be a tuple")
    if type(candidates) is not tuple or any(
            not isinstance(item, DiscourseTopicCandidate) for item in candidates):
        raise TypeError("discourse topic candidates must be a tuple")
    result = {item.candidate_key: item for item in candidates}
    for hypothesis in hypotheses:
        candidate_key = getattr(hypothesis, "candidate_key", None)
        if type(candidate_key) is not tuple or not candidate_key:
            continue
        if candidate_key[0] != MEMORY_CANDIDATE_VERSION:
            continue
        category = memory_candidate_category(candidate_key)
        if category != MEMORY_CATEGORY_DISCOURSE:
            continue
        topic = memory_topic_candidate(candidate_key)
        if topic is None:
            raise ValueError("category-7 Memory hypothesis lacks a topic payload")
        result[candidate_key] = DiscourseTopicCandidate(
            candidate_key,
            topic.proposition_key,
            topic.predicate_key,
            topic.structure_key,
            topic.carrier_key,
            topic.members,
            topic.content_routes(),
        )
    return tuple(sorted(result.values(), key=lambda item: item.stable_key()))


def validate_discourse_topic_query_state(
        state: QueryState,
        candidates: tuple[DiscourseTopicCandidate, ...],
        hypothesis_keys: tuple[tuple[int, ...], ...],
        ) -> tuple[int, ...]:
    """证明话题候选在同一次三图 QueryState 中有真实 Dialogue 根。"""
    if not isinstance(state, QueryState):
        raise TypeError("discourse topic integration requires QueryState")
    if type(candidates) is not tuple or any(
            not isinstance(item, DiscourseTopicCandidate) for item in candidates):
        raise TypeError("discourse topic candidates type error")
    if type(hypothesis_keys) is not tuple or any(
            type(item) is not tuple or not item
            or any(type(value) is not int or value < 0 for value in item)
            for item in hypothesis_keys):
        raise TypeError("discourse topic hypothesis keys type error")
    if not {SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE} <= set(state.active_spaces):
        raise ValueError("话题查询没有同一次三图 active spaces")
    candidate_keys = {item.candidate_key for item in candidates}
    if len(candidate_keys) != len(candidates):
        raise ValueError("话题候选身份重复")
    roots = {(item.owner_space, item.root_key) for item in state.roots}
    missing = tuple(
        key for key in hypothesis_keys if (SPACE_DIALOGUE, key) not in roots)
    if missing:
        raise ValueError("话题 Hypothesis 没有同轮 Dialogue root")
    return frame_key(
        DISCOURSE_TOPIC_RUNTIME_VERSION,
        state.query_key,
        frame_key(1, *(item.stable_key() for item in candidates)),
        frame_key(2, *hypothesis_keys),
    )


def graph_backed_discourse_topic_keys(
        ontology, candidates: tuple[DiscourseTopicCandidate, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """Return topic proposition identities that resolve in the trained graph.

    Query candidates carry complete proposition identity keys.  A topic root is
    considered graph-backed only when that identity is present in the same
    read-only Core ontology; no surface or candidate ordering is accepted as a
    substitute.
    """
    if ontology is None:
        raise TypeError("discourse topic graph validation requires ontology")
    if type(candidates) is not tuple:
        raise TypeError("discourse topic candidates must be a tuple")
    resolved = []
    for candidate in candidates:
        identity = ObjectIdentity.from_stable_key(candidate.proposition_key)
        if ontology.resolve(identity) is not None:
            resolved.append(candidate.proposition_key)
    return tuple(sorted(set(resolved)))


def memory_topic_candidate_key(
        proposition_key: tuple[int, ...],
        predicate_key: tuple[int, ...],
        structure_key: tuple[int, ...],
        carrier_key: tuple[int, ...],
        members: tuple[tuple[int, ...], ...],
        ) -> tuple[int, ...]:
    """Build the topic payload using the existing canonical topic schema."""
    from pure_integer_ai.cognition.understanding.query_memory_candidates import (
        MemoryTopicCandidate,
    )
    return MemoryTopicCandidate(
        proposition_key,
        predicate_key,
        structure_key,
        carrier_key,
        members,
    ).stable_key()


__all__ = [
    "DISCOURSE_TOPIC_RUNTIME_VERSION",
    "DiscourseTopicCandidate",
    "derive_discourse_topic_candidates",
    "graph_backed_discourse_topic_keys",
    "matched_discourse_topic_hypothesis_keys",
    "restore_discourse_topic_candidates",
    "validate_discourse_topic_query_state",
]
