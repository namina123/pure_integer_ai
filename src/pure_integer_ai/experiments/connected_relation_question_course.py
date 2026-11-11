"""Compile train-only connected question syntax without evaluation labels."""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_PLAIN, TrainedRelationQuestionFrame,
)
from pure_integer_ai.experiments.integer_relation_connected_course import (
    ConnectedCourseCompilation,
    ConnectedCourseDraft,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import AuthoredRelationSeed


def compile_train_relation_question(
        seed: AuthoredRelationSeed, query: str, *,
        source_case_key: tuple[int, ...], operation_key: tuple[int, ...],
        ) -> TrainedRelationQuestionFrame | None:
    """只抽取已审 train 教师事实的完整角色问式；不提供事实真值。"""
    if seed.split != "train" or seed.label_owner != "teacher":
        raise ValueError("question syntax requires a train teacher source")
    if seed.sample_role != "support" or len(seed.endpoints) < 2:
        return None
    observed = []
    endpoint_ids = tuple(getattr(item, "endpoint_id", None)
                         for item in seed.endpoints)
    binding_ids = tuple(getattr(item, "endpoint_id", None)
                        for item in seed.bindings)
    if all(item is not None for item in (*endpoint_ids, *binding_ids)):
        endpoint_by_id = dict(zip(endpoint_ids, seed.endpoints))
        if len(endpoint_by_id) != len(seed.endpoints):
            return None
        endpoint_bindings = tuple(
            (endpoint_by_id.get(binding.endpoint_id), binding)
            for binding in seed.bindings)
    elif any(item is not None for item in (*endpoint_ids, *binding_ids)):
        return None
    else:
        # Small structural fixtures may omit endpoint identities; their
        # binding order is the only available source-backed association.
        if len(seed.endpoints) != len(seed.bindings):
            return None
        endpoint_bindings = tuple(zip(seed.endpoints, seed.bindings))
    for endpoint, binding in endpoint_bindings:
        if endpoint is None:
            return None
        start = query.find(endpoint.surface_fragment)
        if start < 0 or query.find(endpoint.surface_fragment, start + 1) >= 0:
            return None
        observed.append((start, start + len(endpoint.surface_fragment), binding))
    observed.sort(key=lambda item: (
        item[0], item[1], item[2].role_kind,
        getattr(item[2], "ordinal", 0)))
    spans = tuple((start, end) for start, end, _binding in observed)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return None
    gaps = (query[:spans[0][0]], *(
        query[left[1]:right[0]] for left, right in zip(spans, spans[1:])),
        query[spans[-1][1]:])
    return TrainedRelationQuestionFrame(
        source_case_key, operation_key,
        authored_relation_identity(seed.relation_kind).stable_key(),
        tuple(authored_relation_role_identity(binding.role_kind).stable_key()
              for _start, _end, binding in observed),
        tuple(tuple(map(ord, gap)) for gap in gaps),
    )


def compile_train_slot_relation_question(
        seed: AuthoredRelationSeed, query: str, *,
        answer_binding_index: int, wh_surface: str,
        source_case_key: tuple[int, ...], operation_key: tuple[int, ...],
        slot_mode: int = SLOT_MODE_PLAIN,
        ) -> TrainedRelationQuestionFrame:
    """Compile one source-backed WH slot without importing its answer surface."""
    if (seed.split != "train" or seed.label_owner != "teacher"
            or seed.sample_role != "support"):
        raise ValueError("slot question syntax requires a train teacher support fact")
    if (type(answer_binding_index) is not int
            or not 0 <= answer_binding_index < len(seed.bindings)):
        raise ValueError("slot question answer binding index is invalid")
    if (not isinstance(query, str) or not query or query.strip() != query
            or not isinstance(wh_surface, str) or not wh_surface
            or wh_surface.strip() != wh_surface):
        raise ValueError("slot question and WH marker must be nonempty trimmed text")
    if type(slot_mode) is not int or slot_mode < 0:
        raise ValueError("slot question mode must be a nonnegative integer")

    endpoint_by_id = {item.endpoint_id: item for item in seed.endpoints}
    if len(endpoint_by_id) != len(seed.endpoints):
        raise ValueError("slot question source repeats an endpoint identity")
    answer_binding = seed.bindings[answer_binding_index]
    answer_endpoint = endpoint_by_id.get(answer_binding.endpoint_id)
    if answer_endpoint is None or answer_endpoint.surface_fragment in query:
        raise ValueError("slot question must omit the answer endpoint surface")

    wh_start = query.find(wh_surface)
    if wh_start < 0 or query.find(wh_surface, wh_start + 1) >= 0:
        raise ValueError("slot question must contain one exact WH marker")
    wh_end = wh_start + len(wh_surface)
    observed = []
    for index, binding in enumerate(seed.bindings):
        if index == answer_binding_index:
            continue
        endpoint = endpoint_by_id.get(binding.endpoint_id)
        if endpoint is None:
            raise ValueError("slot question has no complete observed role projection")
        start = query.find(endpoint.surface_fragment)
        if (start < 0
                or query.find(endpoint.surface_fragment, start + 1) >= 0):
            raise ValueError("slot question must contain each observed endpoint once")
        observed.append((start, index, binding, endpoint))
    observed.sort(key=lambda item: item[0])
    if not observed:
        raise ValueError("slot question has no complete observed role projection")
    spans = []
    for start, _index, _binding, endpoint in observed:
        spans.append((start, start + len(endpoint.surface_fragment)))
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("slot question observed roles are out of source order")

    gaps = (query[:spans[0][0]], *(
        query[left[1]:right[0]] for left, right in zip(spans, spans[1:])),
        query[spans[-1][1]:])
    wh_gaps = []
    for index in range(len(gaps)):
        gap_start = 0 if index == 0 else spans[index - 1][1]
        gap_end = spans[index][0] if index < len(spans) else len(query)
        if gap_start <= wh_start and wh_end <= gap_end:
            wh_gaps.append(index)
    if len(wh_gaps) != 1:
        raise ValueError("WH marker is outside the answer role's source-order gap")

    from pure_integer_ai.experiments.ph2_authored_relation_compile import (
        authored_relation_identity, authored_relation_role_identity,
    )

    return TrainedRelationQuestionFrame(
        source_case_key, operation_key,
        authored_relation_identity(seed.relation_kind).stable_key(),
        tuple(authored_relation_role_identity(binding.role_kind).stable_key()
              for _start, _index, binding, _endpoint in observed),
        tuple(tuple(map(ord, gap)) for gap in gaps),
        authored_relation_role_identity(answer_binding.role_kind).stable_key(),
        tuple(map(ord, wh_surface)),
        slot_mode,
    )


def compile_connected_pair_questions(
        draft: ConnectedCourseDraft,
        compilation: ConnectedCourseCompilation,
        ) -> tuple[TrainedRelationQuestionFrame, ...]:
    """Learn complete role-slot syntax only where a train fact supplies every role.

    This deliberately covers one pair-lookup behavior, not all eight course
    operation labels. Other shapes remain outside this compiler's authority.
    """
    if draft.case_sha256 != compilation.case_sha256:
        raise ValueError("question syntax case identity drifted")
    if draft.source_sha256 != compilation.source_audit.get("source_sha256"):
        raise ValueError("question syntax source identity drifted")
    if compilation.source_audit.get("status") != "PASS_SOURCE_AND_FOUR_WAY_ISOLATION_ONLY":
        raise ValueError("question syntax source audit is not qualified")
    if len(draft.cases) != len(compilation.assignments):
        raise ValueError("question syntax case assignment count drifted")
    seeds = {item.seed_id: item for item in draft.seeds if item.split == "train"}
    frames = []
    for number, (row, assignment) in enumerate(
            zip(draft.cases, compilation.assignments), 1):
        if row["split"] != "train":
            continue
        if (assignment.case_key != (92002, number)
                or assignment.split_code != 1
                or tuple(map(ord, row["input"])) != assignment.input_values):
            raise ValueError("question syntax train assignment identity drifted")
        if (row["operation"] != "ASSERT_RELATION"
                or len(row["seed_ids"]) != 1):
            continue
        seed = seeds[row["seed_ids"][0]]
        frame = compile_train_relation_question(
            seed, row["input"], source_case_key=assignment.case_key,
            operation_key=assignment.operation_key)
        if frame is not None:
            frames.append(frame)
    return tuple(sorted(set(frames), key=lambda item: item.stable_key()))


__all__ = [
    "compile_connected_pair_questions", "compile_train_relation_question",
    "compile_train_slot_relation_question",
]
