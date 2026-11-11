"""Compile explicitly authored connected relation cases for Wave 1 preflight."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
)
from pure_integer_ai.experiments.integer_relation_evaluation_graph import (
    EVALUATION_SPLITS, EvaluationGraphAssignment,
    IntegerRelationEvaluationGraphSplitCourse,
    compile_w06_evaluation_graph_splits,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    integer_course_from_w06_adapter,
)
from pure_integer_ai.experiments.integer_relation_graph_split_course import (
    IntegerRelationGraphSplitCase, IntegerRelationGraphSplitCourse,
)
from pure_integer_ai.experiments.integer_relation_graph_admission import (
    validate_integer_relation_graph_admission,
)
from pure_integer_ai.experiments.integer_relation_source_audit import (
    FOUR_WAY_SPLITS, audit_integer_relation_source_pack,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild, AuthoredCourseSpec, publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    LICENSE_ID, RELATION_REGISTRY, ROLE_REGISTRY, SCHEMA_REGISTRY, SOURCE_KEY,
    REQUEST_CAUSAL_VERIFICATION, REQUEST_EVENT_TIME_VERIFICATION,
    REQUEST_MEREOLOGY_QUERY, REQUEST_PROPERTY_SELECTION,
    REQUEST_REFERENCE_RESOLUTION, REQUEST_RELATION_EVALUATION,
    REQUEST_SYMMETRIC_PAIR_QUERY,
    AuthoredRelationSeed, RelationAnchorSeed, RelationBindingSeed,
    RelationConsumerRequestSeed, RelationEndpointSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject, RECORD_OBSERVATION, RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE, canonical_json_bytes, parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w06_adapter import adapt_w06_training_payload
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_registry import W06_RELATION_REGISTRY
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES, validate_w06_relation_seed,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--connected-wave1-v1"
_KIND_CODES = {
    "concept": OBJECT_CONCEPT, "entity": OBJECT_ENTITY,
    "event": OBJECT_EVENT, "proposition": OBJECT_PROPOSITION,
    "set": OBJECT_SET_EXPR,
}
_FACT_FIELDS = frozenset({
    "anchor", "case_id", "relation", "sample_role", "seed_id",
    "split", "surface", "terms",
})
_CASE_FIELDS = frozenset({"case_id", "input", "operation", "seed_ids", "split"})
_STATES = {"support": "TRUE", "refute": "FALSE", "conflict": "CONFLICT"}
_OPERATION_ORDER = (
    "ASSERT_RELATION", "REVERSE_ROLE", "DISTINGUISH_RELATION",
    "INVERSE_RELATION", "CLARIFY_DIRECTION", "DISTINGUISH_TEMPORAL",
    "CONFLICT_CHECK", "CLARIFY_UNSUPPORTED",
)
_OPERATIONS = frozenset(_OPERATION_ORDER)
_OPERATION_CODES = {name: index for index, name in enumerate(_OPERATION_ORDER, 1)}
_SPEC = AuthoredCourseSpec(
    SOURCE_KEY, LICENSE_ID, 1, 1, 1, 1, 1, PACK_NAME, "W-06",
    "CONNECTED_WAVE1", "connected-wave1-authoring-v1",
    "urn:pure-integer-ai:authored:connected-wave1-v1",
    "Pure Integer AI original CC0 connected relation case course",
    "CONNECTED_RELATION_LABEL", "connected_relation", 100,
)


class ConnectedCourseError(ValueError):
    """Authored facts, cases, or their graph identities do not close."""


def _read_lines(path: str | Path, fields: frozenset[str]) -> tuple[dict, ...]:
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ConnectedCourseError("authored JSONL needs one final newline")
    result = []
    for number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = parse_canonical_json_bytes(line, require_object=True)
        except ValueError as error:
            raise ConnectedCourseError(f"line {number} is not canonical JSON") from error
        if set(row) != fields:
            raise ConnectedCourseError(f"line {number} fields drifted")
        result.append(row)
    return tuple(result)


def _positive(value: object, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConnectedCourseError(f"{where} must be a positive integer")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConnectedCourseError(f"{where} must be nonempty trimmed text")
    return value


def _span(surface: str, fragment: str, where: str) -> tuple[int, int]:
    start = surface.find(fragment)
    if start < 0 or surface.find(fragment, start + 1) >= 0:
        raise ConnectedCourseError(f"{where} must occur exactly once")
    return start, start + len(fragment)


def _request(relation: str, terms: tuple[RelationEndpointSeed, ...]) -> RelationConsumerRequestSeed:
    if relation == "PROPERTY":
        return RelationConsumerRequestSeed(
            REQUEST_PROPERTY_SELECTION, terms[0].endpoint_id, (), 12,
            None, None, attribute_endpoint_id=terms[1].endpoint_id,
            max_options=8,
        )
    left, right = terms
    if relation == "CAUSES":
        return RelationConsumerRequestSeed(
            REQUEST_CAUSAL_VERIFICATION, left.endpoint_id, (), 12,
            None, None, effect_endpoint_id=right.endpoint_id,
            max_evidence_requests=12, max_relations=8,
            max_witness_inputs=8,
        )
    if relation in {"PURE_ALIAS", "REFERS"}:
        return RelationConsumerRequestSeed(
            REQUEST_REFERENCE_RESOLUTION, left.endpoint_id,
            (right.object_kind,), 12, 32, 8,
        )
    if relation in {"SUBSET", "MEMBER"}:
        return RelationConsumerRequestSeed(
            REQUEST_RELATION_EVALUATION, left.endpoint_id,
            (right.object_kind,), 12, 32, 8,
        )
    if relation in {"PART_OF", "HAS_PART"}:
        return RelationConsumerRequestSeed(
            REQUEST_MEREOLOGY_QUERY, left.endpoint_id, (), 12, None, None,
            whole_endpoint_id=right.endpoint_id, max_options=8,
            max_closure_statements=16, max_rule_applications=8,
        )
    if relation in {"SIMILAR", "ANTONYM"}:
        return RelationConsumerRequestSeed(
            REQUEST_SYMMETRIC_PAIR_QUERY, left.endpoint_id, (), 12, None, None,
            max_options=8, counterpart_endpoint_id=right.endpoint_id,
            max_total_direct_facts=16,
        )
    if relation.startswith("EVENT_"):
        return RelationConsumerRequestSeed(
            REQUEST_EVENT_TIME_VERIFICATION, left.endpoint_id, (), 12,
            None, None, object_endpoint_id=right.endpoint_id,
            max_evidence_requests=12, max_relations=8,
        )
    raise ConnectedCourseError(f"relation {relation!r} has no compiled request")


def _seed(row: dict, number: int) -> AuthoredRelationSeed:
    relation = _text(row["relation"], "relation")
    profile = W06_RELATION_PROFILES.get(relation)
    if profile is None:
        raise ConnectedCourseError("this source course has no compiled relation profile")
    surface = _text(row["surface"], "surface")
    anchor_text = _text(row["anchor"], "anchor")
    anchor_start, anchor_end = _span(surface, anchor_text, "anchor")
    raw_terms = row["terms"]
    roles = tuple(profile.role_object_kinds)
    if not isinstance(raw_terms, list) or len(raw_terms) != len(roles):
        raise ConnectedCourseError("source terms do not cover relation roles")
    endpoints = []
    for index, term in enumerate(raw_terms):
        if (not isinstance(term, list) or len(term) != 3
                or term[1] not in _KIND_CODES):
            raise ConnectedCourseError("term must declare text, kind, and local id")
        fragment = _text(term[0], "term text")
        start, end = _span(surface, fragment, "term")
        endpoints.append(RelationEndpointSeed(
            f"term-{index + 1}", fragment, start, end, 0,
            _KIND_CODES[term[1]], _positive(term[2], "term local id"),
        ))
    if any(left.end > right.start for left, right in zip(
            endpoints, endpoints[1:])):
        raise ConnectedCourseError("source terms must be ordered and disjoint")
    bindings = tuple(RelationBindingSeed(
        ROLE_REGISTRY, role, endpoints[index].endpoint_id, 0,
        tuple(sorted(allowed)),
    ) for index, (role, allowed) in enumerate(roles))
    sample_role = _text(row["sample_role"], "sample_role")
    if sample_role not in _STATES:
        raise ConnectedCourseError("source fact sample role is not supported")
    split = _text(row["split"], "split")
    if split not in ("train", "held_out"):
        raise ConnectedCourseError("source fact split must be train/held_out")
    normalized = surface
    for endpoint in reversed(endpoints):
        normalized = (normalized[:endpoint.start] + "<?>"
                      + normalized[endpoint.end:])
    seed = AuthoredRelationSeed(
        _text(row["seed_id"], "seed_id"),
        _text(row["case_id"], "case_id"),
        normalized,
        "teacher" if split == "train" else "evaluator",
        split, sample_role, relation, RELATION_REGISTRY,
        profile.relation_kind, SCHEMA_REGISTRY, profile.relation_kind,
        profile.directionality, surface,
        RelationAnchorSeed(anchor_text, anchor_start, anchor_end, 0),
        tuple(endpoints), bindings, number,
        _request(relation, tuple(endpoints)),
        _STATES[sample_role],
        CanonicalJsonObject.from_value({"decision": sample_role}),
        "CONFLICT_SOURCE" if sample_role == "conflict" else "NONE",
        "", number,
    )
    validate_w06_relation_seed(seed)
    return seed


@dataclass(frozen=True)
class ConnectedCourseDraft:
    seeds: tuple[AuthoredRelationSeed, ...]
    cases: tuple[dict, ...]
    source_sha256: str
    case_sha256: str


def prepare_connected_course(
        source_path: str | Path, case_path: str | Path,
        ) -> ConnectedCourseDraft:
    """Validate all authored identities and real within-case shared endpoints."""
    rows = _read_lines(source_path, _FACT_FIELDS)
    cases = _read_lines(case_path, _CASE_FIELDS)
    seeds = tuple(_seed(row, index) for index, row in enumerate(rows, 1))
    by_id = {seed.seed_id: seed for seed in seeds}
    if len(by_id) != len(seeds):
        raise ConnectedCourseError("source seed ids repeat")
    case_ids = set()
    used = set()
    endpoint_owners: dict[tuple[int, int], tuple[str, str]] = {}
    named_endpoint_owners: dict[tuple[int, str], str] = {}
    for case in cases:
        case_id = _text(case["case_id"], "case_id")
        split = _text(case["split"], "case split")
        if case_id in case_ids or split not in FOUR_WAY_SPLITS:
            raise ConnectedCourseError("case identity or split is invalid")
        case_ids.add(case_id)
        _text(case["input"], "natural query")
        if case["operation"] not in _OPERATIONS:
            raise ConnectedCourseError("case operation is not explicitly registered")
        seed_ids = case["seed_ids"]
        if (not isinstance(seed_ids, list) or not seed_ids
                or len(seed_ids) != len(set(seed_ids))):
            raise ConnectedCourseError("case turn order is missing or duplicated")
        if split == "held_out" and len(seed_ids) < 2:
            raise ConnectedCourseError("held-out case is not multi-relation")
        selected = []
        for seed_id in seed_ids:
            seed = by_id.get(seed_id)
            if (seed is None or seed_id in used or seed.family != case_id
                    or seed.split != ("train" if split == "train" else "held_out")):
                raise ConnectedCourseError("case/source turn ownership drifted")
            selected.append(seed)
            used.add(seed_id)
            for endpoint in seed.endpoints:
                name_key = (endpoint.object_kind, endpoint.surface_fragment)
                prior_case = named_endpoint_owners.setdefault(name_key, case_id)
                if prior_case != case_id:
                    raise ConnectedCourseError("named endpoint crosses authored cases")
                key = (endpoint.object_kind, endpoint.local_id)
                prior = endpoint_owners.setdefault(
                    key, (case_id, endpoint.surface_fragment))
                if prior != (case_id, endpoint.surface_fragment):
                    raise ConnectedCourseError("endpoint identity crosses cases or names")
        if len(selected) > 1:
            overlap = set(
                (item.object_kind, item.local_id) for item in selected[0].endpoints)
            for seed in selected[1:]:
                current = {(item.object_kind, item.local_id)
                           for item in seed.endpoints}
                if not overlap.intersection(current):
                    raise ConnectedCourseError("case facts have no declared shared endpoint")
                overlap.update(current)
    if used != set(by_id) or set(FOUR_WAY_SPLITS) - {case["split"] for case in cases}:
        raise ConnectedCourseError("four-way case coverage is incomplete")
    return ConnectedCourseDraft(
        seeds, cases,
        hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        hashlib.sha256(Path(case_path).read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class ConnectedCourseCompilation:
    pack: AuthoredCourseBuild
    train_course: IntegerRelationGraphSplitCourse
    evaluation_courses: dict[str, IntegerRelationEvaluationGraphSplitCourse]
    source_audit: dict[str, object]
    topology_admission: dict[str, object]
    assignments: tuple[ConnectedCaseAssignment, ...]
    case_sha256: str
    wave1_course_coverage: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectedCaseAssignment:
    """Integer-only source turn order and evaluator-side operation assignment."""

    case_key: tuple[int, ...]
    split_code: int
    operation_key: tuple[int, ...]
    input_values: tuple[int, ...]
    turn_values: tuple[tuple[int, ...], ...]
    observation_keys: tuple[tuple[int, ...], ...]
    dialogue_chain_key: tuple[int, ...]

    def to_value(self) -> dict[str, object]:
        return {
            "case_key": list(self.case_key),
            "split_code": self.split_code,
            "operation_key": list(self.operation_key),
            "input_values": list(self.input_values),
            "turn_values": [list(item) for item in self.turn_values],
            "observation_keys": [list(item) for item in self.observation_keys],
            "dialogue_chain_key": list(self.dialogue_chain_key),
        }


def compile_connected_course(
        source_path: str | Path, case_path: str | Path,
        release_root: str | Path,
        *, source_identity: str | None = None,
        ) -> ConnectedCourseCompilation:
    """Publish one CC0 source pack and compile isolated four-way graph views."""
    if source_identity is not None and (not isinstance(source_identity, str)
            or not source_identity.startswith("urn:pure-integer-ai:authored:")
            or source_identity.strip() != source_identity):
        raise ConnectedCourseError("new authored source identity is invalid")
    draft = prepare_connected_course(source_path, case_path)
    substages = {
        seed.seed_id: (
            W06_RELATION_PROFILES[seed.relation_family].substage_key,
            W06_RELATION_REGISTRY[
                W06_RELATION_PROFILES[seed.relation_family].substage_key
            ].teacher_evidence_kinds[0],
        ) for seed in draft.seeds
    }
    pack = publish_authored_course(
        tuple(compile_relation_seed(seed, semantic_identity=True)
              for seed in draft.seeds),
        source_path, release_root,
        _SPEC if source_identity is None else replace(
            _SPEC, snapshot_id=source_identity.rsplit(":", 1)[-1],
            official_url=source_identity),
        seed_substages=substages,
    )
    records: dict[str, list] = {
        RECORD_SOURCE_REF: [], RECORD_OBSERVATION: [], RECORD_TEACHER_EVIDENCE: [],
    }
    for identity in pack.manifest.files:
        if identity.record_kind in records:
            records[identity.record_kind].extend(
                read_record_artifact(pack.pack_root, identity))
    sources = tuple(records[RECORD_SOURCE_REF])
    observations = tuple(records[RECORD_OBSERVATION])
    teachers = tuple(records[RECORD_TEACHER_EVIDENCE])
    source_by_key = {item.stable_key: item for item in sources}
    observation_by_seed = {
        source_by_key[item.source_ref_key].revision_id: item
        for item in observations
    }
    source_identity = pack.manifest.stable_key.components
    license_identity = tuple(map(ord, LICENSE_ID))
    owner_identity = (92001, *source_identity)
    adapter = adapt_w06_training_payload(W06TrainingPayload(
        sources,
        tuple(item for item in observations if item.split == "train"),
        teachers,
    ), require_full_curriculum=False)
    train_graph = integer_course_from_w06_adapter(
        adapter, source_identity=source_identity,
        license_identity=license_identity,
    )
    train_cases = []
    evaluation_assignments = {name: [] for name in EVALUATION_SPLITS}
    source_assignments = {name: [] for name in FOUR_WAY_SPLITS}
    case_assignments = []
    for number, case in enumerate(draft.cases, 1):
        seed_ids = case["seed_ids"]
        observation_keys = tuple(
            observation_by_seed[seed_id].stable_key.components
            for seed_id in seed_ids
        )
        source_assignments[case["split"]].extend(observation_keys)
        turns = [observation_by_seed[seed_id].typed_payload.to_value()["surface"]
                 for seed_id in seed_ids]
        chain_digest = hashlib.sha256(canonical_json_bytes({
            "case_id": case["case_id"], "turns": [*turns, case["input"]],
        })).digest()
        chain_id = int.from_bytes(chain_digest[:8], "big") & ((1 << 63) - 1)
        case_key = (92002, number)
        chain_keys = ((92003, chain_id or 1),)
        input_values = tuple(map(ord, case["input"]))
        case_assignments.append(ConnectedCaseAssignment(
            case_key,
            FOUR_WAY_SPLITS.index(case["split"]) + 1,
            (92004, _OPERATION_CODES[case["operation"]]),
            input_values,
            tuple(tuple(map(ord, turn)) for turn in turns),
            observation_keys,
            chain_keys[0],
        ))
        if case["split"] == "train":
            train_cases.append(IntegerRelationGraphSplitCase(
                case_key, input_values, chain_keys, observation_keys,
            ))
        else:
            evaluation_assignments[case["split"]].append(
                EvaluationGraphAssignment(
                    case["split"], case_key, input_values,
                    chain_keys, observation_keys,
                ))
    train_course = IntegerRelationGraphSplitCourse(
        "train", owner_identity, source_identity, license_identity,
        tuple(sorted(train_cases, key=lambda item: item.case_key)),
        train_graph.records,
    )
    evaluation = compile_w06_evaluation_graph_splits(
        tuple(item for item in observations if item.split == "held_out"),
        sources,
        {name: tuple(values) for name, values in evaluation_assignments.items()},
        owner_identity=owner_identity, source_identity=source_identity,
        license_identity=license_identity,
    )
    source_audit = audit_integer_relation_source_pack(
        pack.pack_root, source_path, assignments=source_assignments,
        source_row_parser=_seed,
    )
    topology = validate_integer_relation_graph_admission(train_course, evaluation)
    result = ConnectedCourseCompilation(
        pack, train_course, evaluation, source_audit, topology,
        tuple(case_assignments),
        draft.case_sha256,
    )
    from pure_integer_ai.experiments.integer_relation_wave1_course_gate import (
        audit_integer_relation_wave1_course,
    )
    return replace(
        result, wave1_course_coverage=audit_integer_relation_wave1_course(result))


__all__ = [
    "ConnectedCaseAssignment", "ConnectedCourseCompilation",
    "ConnectedCourseDraft", "ConnectedCourseError",
    "PACK_NAME", "compile_connected_course", "prepare_connected_course",
]
