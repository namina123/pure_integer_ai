"""Freeze syntax, teacher permissions and evaluation partitions before learning."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle
from pure_integer_ai.experiments.integer_relation_connected_course import (
    _CASE_FIELDS, _FACT_FIELDS, _read_lines, _seed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from pure_integer_ai.cognition.understanding.query_open_roles import match_open_roles
from pure_integer_ai.cognition.understanding.query_relation_questions import _composed_frames
from pure_integer_ai.experiments.independent_relation_fact_admission import _matches
from pure_integer_ai.experiments.integer_relation_graph_course import integer_relation_graph_course_to_w06_adapter


SYNTAX = "data/ph2/syntax_train_facts.jsonl"
QUESTIONS = "data/ph2/syntax_train_questions.jsonl"
TEACHER = "data/ph2/new_teacher_facts.jsonl"
INPUTS = "data/ph2/owners/evaluator/inputs.jsonl"
LABELS = "data/ph2/owners/evaluator/labels.jsonl"
PERMISSION = "source_permission.json"
SOURCE_FILES = {
    SYNTAX: "data/ph2/independent_fact_syntax_train_v1.jsonl.sample",
    QUESTIONS: "data/ph2/independent_fact_syntax_questions_v1.jsonl.sample",
    TEACHER: "data/ph2/independent_new_teacher_facts_v1.jsonl.sample",
    INPUTS: "data/ph2/independent_fact_preflight_inputs_v1.jsonl.sample",
    LABELS: "data/ph2/owners/evaluator/independent_fact_preflight_labels_v1.jsonl.sample",
    PERMISSION: "data/ph2/independent_fact_preflight_permission_v1.json",
}
SPLITS = ("held_out", "negative", "conflict", "unauthorized")
RELATIONS = frozenset({"CAUSES", "EVENT_BEFORE", "REFERS", "PART_OF", "SIMILAR"})
BUDGET = {
    "max_depth": 24, "max_nodes": 512, "max_edges": 1024, "max_reads": 2048,
    "max_parse_steps": 65536, "max_witness_integers": 65536,
    "max_case_wall_seconds": 600, "max_wall_ns": 3600 * 10**9,
    "max_process_peak_bytes": 2 * 1024**3, "max_output_bytes": 1024**3,
    "minimum_free_bytes": 12 * 1024**3,
    "active_check_interval_seconds": 3600, "minimum_active_check_interval_seconds": 1800,
}


def read_rows(path: Path) -> tuple[dict, ...]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("preflight records need one final newline")
    return tuple(parse_canonical_json_bytes(line, require_object=True) for line in raw.splitlines())


def fact_input_source(ordinal: int) -> SourceRef:
    """A named host ingestion source; entity identities still come from teachers."""
    if type(ordinal) is not int or ordinal <= 0:
        raise ValueError("independent input ordinal must be positive")
    return SourceRef(92042, 202610130300 + ordinal, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


def authorized_fact_inputs(root: Path) -> tuple[SourceRef, ...]:
    """Only the training host reads this permission; evaluator labels are absent."""
    permission = read_rows(root / PERMISSION)[0]
    allowed = set(permission["authorized_teacher_seed_ids"])
    return tuple(fact_input_source(ordinal) for ordinal, row in enumerate(
        _read_lines(root / TEACHER, _FACT_FIELDS), 1) if row["seed_id"] in allowed)


def inspect_independent_fact_contract(root: Path) -> dict[str, object]:
    """Audit partition denominators; this result supplies no labels to Core."""
    permissions = read_rows(root / PERMISSION)
    if len(permissions) != 1:
        raise ValueError("independent preflight needs one explicit permission")
    permission = permissions[0]
    if (permission.get("format") != "INDEPENDENT_FACT_PREFLIGHT_PERMISSION_V1"
            or permission.get("license_id") != "CC0-1.0"
            or permission.get("source_role") != "independent_train_teacher"
            or permission.get("scope") != "ADMISSION_AND_LANGUAGE_TRANSFER_PREFLIGHT_NOT_WAVE1"
            or permission.get("external_source_reads") != 0
            or permission.get("production_authorized") != 0
            or permission.get("free_dialogue_complete") != 0
            or not permission.get("rights_basis")):
        raise ValueError("independent preflight source permission drifted")
    syntax_rows = _read_lines(root / SYNTAX, _FACT_FIELDS)
    teacher_rows = _read_lines(root / TEACHER, _FACT_FIELDS)
    syntax = tuple(_seed(row, number) for number, row in enumerate(syntax_rows, 1))
    teacher = tuple(_seed(row, number) for number, row in enumerate(teacher_rows, 1))
    if any(item.split != "train" or item.label_owner != "teacher" for item in (*syntax, *teacher)):
        raise ValueError("syntax and fact sources must be train teachers")
    if (len(syntax) != 5 or len(teacher) != 9
            or {item.relation_family for item in syntax} != RELATIONS
            or {item.relation_family for item in teacher} != RELATIONS
            or any(item.sample_role != "support" for item in syntax)
            or len({item.seed_id for item in (*syntax, *teacher)}) != len(syntax) + len(teacher)):
        raise ValueError("independent preflight source denominator or identity drifted")
    allowed = permission.get("authorized_teacher_seed_ids")
    if (not isinstance(allowed, list) or len(allowed) != 8 or len(set(allowed)) != 8
            or not set(allowed) < {item.seed_id for item in teacher}):
        raise ValueError("explicit teacher input permission is incomplete")
    old_names = {term.surface_fragment for item in syntax for term in item.endpoints}
    new_names = {term.surface_fragment for item in teacher for term in item.endpoints}
    old_ids = {(term.object_kind, term.local_id) for item in syntax for term in item.endpoints}
    new_ids = {(term.object_kind, term.local_id) for item in teacher for term in item.endpoints}
    if old_names & new_names or old_ids & new_ids:
        raise ValueError("new fact entities leak into syntax training")
    questions = _read_lines(root / QUESTIONS, _CASE_FIELDS)
    syntax_ids = {item.seed_id for item in syntax}
    if (len(questions) != 10 or len({row["case_id"] for row in questions}) != 10
            or any(row["split"] != "train" or row["operation"] != "ASSERT_RELATION"
                   or len(row["seed_ids"]) != 1 or row["seed_ids"][0] not in syntax_ids
                   for row in questions)):
        raise ValueError("question training contains a non-syntax owner")
    inputs = read_rows(root / INPUTS)
    counts = {name: 0 for name in SPLITS}
    seen, surfaces = set(), {row["input"] for row in questions}
    for row in inputs:
        if (set(row) != {"case_id", "input", "split"} or row["split"] not in counts
                or not isinstance(row["case_id"], str) or not row["case_id"]
                or row["case_id"] in seen or not isinstance(row["input"], str)
                or not row["input"] or row["input"] in surfaces):
            raise ValueError("evaluation input identity or partition drifted")
        seen.add(row["case_id"])
        counts[row["split"]] += 1
    if counts != dict(zip(SPLITS, (5, 2, 1, 1))):
        raise ValueError("independent preflight evaluation denominator drifted")
    labels = read_rows(root / LABELS)
    if (len(labels) != len(inputs) or len({row.get("case_id") for row in labels}) != len(labels)
            or {row.get("case_id") for row in labels} != seen):
        raise ValueError("independent evaluator labels do not cover the frozen cases")
    by_id = {row["case_id"]: row for row in inputs}
    teacher_ids = {item.seed_id for item in teacher}
    for row in labels:
        if (set(row) != {"case_id", "expected_seed_id", "policy"}
                or row["policy"] not in {"DIRECT_ONLY", "NO_FACTUAL_ANSWER"}
                or (row["expected_seed_id"] and row["expected_seed_id"] not in teacher_ids)
                or (row["policy"] == "DIRECT_ONLY") != (by_id[row["case_id"]]["split"] == "held_out")
                or (row["policy"] == "DIRECT_ONLY" and row["expected_seed_id"] not in allowed)):
            raise ValueError("independent evaluator policy or permission drifted")
    return {
        "syntax_fact_count": len(syntax), "train_question_count": len(questions),
        "independent_teacher_record_count": len(teacher), "authorized_input_count": len(allowed),
        "relation_family_count": 5, "operation_family_count": 1,
        "evaluation_case_counts": counts,
        "new_entity_novel_relative_to_syntax": 1,
        "new_entity_novel_relative_to_teacher": 0,
        "held_out_scope": "UNSEEN_QUESTION_COMPOSITION_ON_NEWLY_ADMITTED_TEACHER_FACTS",
        "teacher_factual_truth_held_out": 0, "source_dialogue_full_isolation": 0,
        "unauthorized_scope": "UNGRANTED_FACT_INPUT_NOT_QUERY_USER_ACL_EVALUATION",
        "multi_relation_inference": "UNKNOWN_SHARED_ENDPOINTS_ARE_NOT_REASONING_EVIDENCE",
        "wave1_qualification": "BLOCKED_SINGLE_OPERATION_AND_INCOMPLETE_NOVEL_SPLITS",
        "free_dialogue_complete": 0,
    }


def audit_prepared_question_carriers(frames: tuple, inputs: tuple[dict, ...], *, teacher_course) -> dict:
    """Join exact syntax to independently declared role surfaces, never support or active truth."""
    declared = integer_relation_graph_course_to_w06_adapter(teacher_course, require_full_curriculum=False)
    rows = []
    for number, row in enumerate(inputs, 1):
        values = tuple(map(ord, row["input"]))
        composed = _composed_frames(frames, values, max_combinations=4096)
        parses = match_open_roles(tuple(item.open_frame() for item in (*frames, *composed)),
                                  values, (92043, number))
        meanings = {(item.frame.predicate, tuple(
            (span.role, span.start, span.end, span.values) for span in item.bindings))
                    for item in parses}
        composed_keys = {item.open_frame().stable_key() for item in composed}
        source_keys = {frame.stable_key() for frame in frames}
        used_composed = tuple(item for item in composed
                              if any(parse.frame.stable_key() == item.open_frame().stable_key()
                                     for parse in parses))
        unbound_composed_meanings = {(item.frame.predicate, tuple(
            (span.role, span.start, span.end, span.values) for span in item.bindings))
                    for item in parses if item.frame.stable_key() in composed_keys}
        qualified = tuple(item for item in parses
                          if any(_matches(item, candidate) for candidate in declared.candidates))
        declared_meanings = {(item.frame.predicate, tuple(
            (span.role, span.start, span.end, span.values) for span in item.bindings)) for item in qualified}
        composed_meanings = {(item.frame.predicate, tuple(
            (span.role, span.start, span.end, span.values) for span in item.bindings))
                    for item in qualified if item.frame.stable_key() in composed_keys}
        if row["split"] == "held_out" and (len(declared_meanings) != 1 or len(composed_meanings) != 1
                or any(item.frame.stable_key() not in composed_keys for item in qualified)):
            raise ValueError("held-out carrier has no unique, unseen source-backed complete parse")
        if any(item.stable_key() in source_keys for item in used_composed):
            raise ValueError("held-out question syntax is a train frame")
        rows.append({"case_id": row["case_id"], "split": row["split"],
                     "complete_meaning_count": len(meanings),
                     "unbound_composed_meaning_count": len(unbound_composed_meanings),
                     "declared_role_meaning_count": len(declared_meanings),
                     "composed_meaning_count": len(composed_meanings),
                     "parse_keys": [list(item.stable_key()) for item in parses],
                     "declared_role_parse_keys": [list(item.stable_key()) for item in qualified],
                     "composed_frame_keys": [list(item.stable_key()) for item in used_composed],
                     "qualification": "DECLARED_ROLE_SURFACE_JOIN_ONLY_REQUIRES_REAL_ADMISSION_AND_CORE"})
    return {"status": "PASS_PREPARED_SOURCE_BACKED_CARRIERS_ONLY", "cases": rows,
            "core_training_count": 0, "query_execution_count": 0, "free_dialogue_complete": 0}


__all__ = ["BUDGET", "SOURCE_FILES", "SYNTAX", "QUESTIONS", "TEACHER", "INPUTS",
           "LABELS", "PERMISSION", "SPLITS", "read_rows", "fact_input_source",
           "authorized_fact_inputs", "inspect_independent_fact_contract",
           "audit_prepared_question_carriers"]
