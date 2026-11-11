"""Compile one physical licensed train instruction against audited fact teachers."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef, minimal_instruction_identity
from pure_integer_ai.cognition.shared.relation_composition import RelationCompositionWitness
from pure_integer_ai.cognition.shared.typed_relation import CompositionRule
from pure_integer_ai.experiments.independent_relation_fact_source import (
    IndependentRelationFactSource, verify_independent_fact_source_course,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    integer_relation_graph_course_to_w06_adapter,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS


def compile_train_relation_composition(
        source_path: str | Path, facts: IndependentRelationFactSource) -> RelationCompositionWitness:
    """Learn only the explicitly authorized integer join, never example conclusions."""
    verify_independent_fact_source_course(facts)
    raw = Path(source_path).resolve(strict=True).read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("composition source requires one final newline")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    if set(row) != {"instruction", "license_id", "owner", "rule_id", "split"}:
        raise ValueError("composition source fields drifted")
    if row["split"] != "train" or row["owner"] != "teacher" or row["license_id"] != "CC0-1.0":
        raise ValueError("composition instruction requires a CC0 train teacher")
    values, rule_id = row["instruction"], row["rule_id"]
    if (type(values) is not list or len(values) != 9
            or any(type(item) is not int or item <= 0 for item in values)
            or type(rule_id) is not list or not rule_id
            or any(type(item) is not int or item <= 0 for item in rule_id)):
        raise ValueError("composition source requires explicit integer relation/role mappings")
    rule = CompositionRule(
        minimal_instruction_identity(tuple(rule_id), versions=W06_IDENTITY_VERSIONS),
        *(authored_relation_identity(value) if index % 3 == 0
          else authored_relation_role_identity(value) for index, value in enumerate(values)))
    adapter = integer_relation_graph_course_to_w06_adapter(
        facts.course, require_full_curriculum=False)
    schemas = tuple(item for item in adapter.schemas if item.relation in {
        rule.first_relation, rule.second_relation, rule.result_relation})
    digest = hashlib.sha256(raw).digest()
    witness_source = SourceRef(
        92100, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
        0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    source_identity = facts.course.source_identity
    return RelationCompositionWitness(
        rule, schemas, witness_source, tuple(digest), source_identity,
        facts.course.license_identity,
        tuple(sorted({candidate.source_ref for candidate in adapter.candidates},
                     key=SourceRef.stable_key)))
