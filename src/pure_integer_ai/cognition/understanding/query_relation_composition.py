"""Bind complete learned question slots to pending candidates across relations."""
from __future__ import annotations

from pure_integer_ai.cognition.shared.relation_composition import PendingRelationComposition
from pure_integer_ai.cognition.understanding.query_input_structure import (
    InputPendingRelationComposition, PROJECTION_FILLER, QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles
from pure_integer_ai.cognition.understanding.query_relation_questions import TrainedRelationQuestionFrame


def match_pending_relation_questions(
        frames: tuple[TrainedRelationQuestionFrame, ...], structure: QueryInputStructure,
        pending: tuple[PendingRelationComposition, ...],
        ) -> tuple[InputPendingRelationComposition, ...]:
    """Require whole-input syntax, exact role spans, full identities and a unique rule proof."""
    if (not frames or not structure.tokens or not pending
            or structure.discourse_compositions or structure.discourse_gaps
            or structure.relation_question_bindings):
        return ()
    available = {item.open_frame().stable_key(): item for item in frames}
    if len(available) != len(frames):
        raise ValueError("pending question frames have competing parse identities")
    spans = {(item.start, item.end): item.ref_key for item in structure.spans}
    result = set()
    for parsed in match_open_roles(tuple(item.open_frame() for item in frames),
                                   structure.token_values, structure.source_ref):
        frame = available[parsed.frame.stable_key()]
        refs, fillers = [], []
        for slot in parsed.bindings:
            ref = spans.get((slot.start, slot.end))
            identities = {item.candidate_key for item in structure.semantic_candidates
                          if item.projection_kind == PROJECTION_FILLER
                          and not item.concept_route_key and item.span_ref == ref}
            if ref is None or len(identities) != 1:
                break
            refs.append(ref)
            fillers.append(next(iter(identities)))
        else:
            if len(refs) != 2:
                continue
            expected = dict(zip(frame.roles, fillers, strict=True))
            for item in pending:
                if (item.candidate.relation.stable_key() == frame.predicate_key
                        and {binding.role.stable_key(): binding.filler.stable_key()
                             for binding in item.candidate.bindings} == expected):
                    result.add(InputPendingRelationComposition(
                        frame.stable_key(), frame.operation_key, item.stable_key(), tuple(refs)))
    return tuple(sorted(result, key=lambda item: item.stable_key())) if len(result) == 1 else ()
