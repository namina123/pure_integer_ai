"""Train-only source compilation for connected compound relation questions."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputDiscourseOperator,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.experiments.connected_relation_question_course import (
    compile_train_relation_question,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    LICENSE_ID, AuthoredRelationSeed,
)
from pure_integer_ai.experiments.trained_input_discourse_conjunction_graph import (
    TrainedInputDiscourseConjunction,
)
from pure_integer_ai.experiments.trained_input_discourse_operation_graph import (
    TrainedInputDiscourseOperationContract,
)


MODE_CONJOINED_RELATION_QUERY = (91672, 5)


@dataclass(frozen=True, slots=True)
class ConjoinedTrainQuestion:
    frames: tuple[TrainedRelationQuestionFrame, ...]
    operator: TrainedInputDiscourseOperator
    contract: TrainedInputDiscourseOperationContract
    conjunction: TrainedInputDiscourseConjunction
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]


def compile_train_conjoined_question(
        seeds: tuple[AuthoredRelationSeed, ...], query: str, *,
        source_case_key: tuple[int, ...], question_operation_key: tuple[int, ...],
        conjunction_operation_key: tuple[int, ...], conjunction_surface: str,
        source_identity: tuple[int, ...], license_identity: tuple[int, ...],
        ) -> ConjoinedTrainQuestion:
    """Remove teacher fillers and retain only exact syntax and join permission.

    The caller must already have audited the source/permission pack.  This
    compiler consumes only train teachers; held-out inputs and expected answers
    are not arguments.  Its contract is joint lookup, not multi-hop inference.
    """
    if (type(seeds) is not tuple or len(seeds) < 2
            or type(query) is not str or not query
            or type(conjunction_surface) is not str or not conjunction_surface):
        raise ValueError("conjoined question needs train facts and an exact source operator")
    for key in (source_case_key, question_operation_key, conjunction_operation_key,
                source_identity, license_identity):
        if type(key) is not tuple or not key:
            raise ValueError("conjoined question source and operation keys must be nonempty")
    pack_record(92030, source_case_key, question_operation_key,
                conjunction_operation_key, source_identity, license_identity)
    if question_operation_key == conjunction_operation_key:
        raise ValueError("question and conjunction operations need distinct contracts")
    if license_identity != tuple(map(ord, LICENSE_ID)):
        raise ValueError("conjoined syntax requires a licensed source identity")
    if any(seed.split != "train" or seed.label_owner != "teacher"
           or seed.sample_role != "support"
           for seed in seeds):
        raise ValueError("conjoined syntax requires licensed train teachers only")
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise ValueError("conjoined question may not repeat a teacher fact")
    if any(type(endpoint.local_id) is not int or endpoint.local_id <= 0
           for seed in seeds for endpoint in seed.endpoints):
        raise ValueError("conjoined endpoint identity is not independently materialized")
    clauses = query.split(conjunction_surface)
    if len(clauses) != len(seeds) or any(not clause for clause in clauses):
        raise ValueError("conjoined operator does not uniquely partition the source question")
    endpoints = [{(item.object_kind, item.local_id) for item in seed.endpoints}
                 for seed in seeds]
    reached = {0}
    while True:
        expanded = reached | {index for index, identities in enumerate(endpoints)
                              if any(identities & endpoints[prior] for prior in reached)}
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != len(seeds):
        raise ValueError("conjoined teacher facts have no shared graph endpoint")
    frames = []
    for ordinal, (seed, clause) in enumerate(zip(seeds, clauses), 1):
        frame = compile_train_relation_question(
            seed, clause, source_case_key=(*source_case_key, ordinal),
            operation_key=question_operation_key)
        if frame is None:
            raise ValueError("conjoined train clause has incomplete or ambiguous role slots")
        frames.append(frame)
    return ConjoinedTrainQuestion(
        tuple(sorted(frames, key=lambda item: item.stable_key())),
        TrainedInputDiscourseOperator(conjunction_operation_key,
                                     tuple(map(ord, conjunction_surface))),
        TrainedInputDiscourseOperationContract(
            conjunction_operation_key, MODE_CONJOINED_RELATION_QUERY),
        TrainedInputDiscourseConjunction(conjunction_operation_key),
        source_identity, license_identity,
    )


__all__ = ["ConjoinedTrainQuestion", "MODE_CONJOINED_RELATION_QUERY",
           "compile_train_conjoined_question"]
