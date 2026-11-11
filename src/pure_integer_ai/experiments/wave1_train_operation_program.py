"""Compile a source-qualified operation program from a frozen Wave 1 train split.

The program source is an integer train instruction. Its identity and allowed
schemas come from the audited train graph, never from an evaluator label or a
previous response.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import GLOBAL_OWNER_SCOPE, SourceRef
from pure_integer_ai.cognition.shared.integer_operation_program import (
    IntegerOperationProgram, OperationProgramWitness,
)
from pure_integer_ai.cognition.shared.relation_composition import schema_key
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourse, integer_relation_graph_course_to_w06_adapter,
)
from pure_integer_ai.experiments.integer_relation_graph_split_course import (
    read_integer_relation_graph_split_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_identity_protocol import W06_IDENTITY_VERSIONS


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frozen_wave1_train(course_root: Path) -> IntegerRelationGraphCourse:
    """Rejoin the physical split with the original source and license receipt."""
    root = Path(course_root).resolve(strict=True)
    receipt = json.loads((root / "course_receipt.json").read_bytes())
    permission = json.loads((root / "source_permission.json").read_bytes())
    split_path = root / "train.int.json"
    if (receipt.get("split_sha256", {}).get("train") != _sha(split_path)
            or receipt.get("source_sha256") != permission.get("source_sha256")
            or receipt.get("case_sha256") != permission.get("case_sha256")
            or receipt.get("source_permission_sha256") != _sha(root / "source_permission.json")
            or receipt.get("source_sha256")
            != _sha(root / "data/ph2/wave1_real_facts_20261014.jsonl")
            or receipt.get("case_sha256")
            != _sha(root / "data/ph2/wave1_real_cases_20261014.jsonl")
            or permission.get("license_id") != "CC0-1.0"
            or receipt.get("source_audit", {}).get("status")
            != "PASS_SOURCE_AND_FOUR_WAY_ISOLATION_ONLY"
            or receipt.get("topology_admission", {}).get("status")
            != "PASS_STRUCTURAL_ADMISSION_ONLY"
            or receipt.get("free_dialogue_complete") != 0):
        raise ValueError("Wave 1 source, split or license receipt is not frozen")
    split = read_integer_relation_graph_split_course(split_path)
    if (split.split != "train" or len(split.cases) != 20
            or len(split.records) != 20
            or len({record.relation_kind for record in split.records}) != 5
            or tuple(receipt["source_identity"]) != split.source_identity
            or tuple(receipt["license_identity"]) != split.license_identity):
        raise ValueError("Wave 1 train-only graph does not match the five-family receipt")
    return IntegerRelationGraphCourse(
        split.source_identity, split.license_identity, split.records)


def compile_wave1_train_operation_program(
        source_path: Path, course_root: Path) -> OperationProgramWitness:
    """Admit only a canonical physical CC0 train program over actual schemas."""
    path = Path(source_path).resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("operation program source must be one bounded record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    if (set(row) != {"program_key", "license_id", "owner", "split"}
            or row["split"] != "train" or row["owner"] != "teacher"
            or row["license_id"] != "CC0-1.0"
            or type(row["program_key"]) is not list):
        raise ValueError("operation program lacks a canonical CC0 train instruction")
    program = IntegerOperationProgram.from_stable_key(tuple(row["program_key"]))
    course = read_frozen_wave1_train(course_root)
    adapter = integer_relation_graph_course_to_w06_adapter(
        course, require_full_curriculum=False)
    schemas = {schema.relation: schema_key(schema) for schema in adapter.schemas}
    if any(schemas.get(atom.schema.relation) != schema_key(atom.schema)
           for atom in program.atoms):
        raise ValueError("operation program atom is outside the actual train schemas")
    digest = hashlib.sha256(raw).digest()
    source = SourceRef(
        92221, (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1,
        0, GLOBAL_OWNER_SCOPE, W06_IDENTITY_VERSIONS)
    forming = tuple(sorted({candidate.source_ref for candidate in adapter.candidates},
                           key=SourceRef.stable_key))
    return OperationProgramWitness(
        program, source, tuple(digest), course.source_identity,
        course.license_identity, forming)


__all__ = ["compile_wave1_train_operation_program", "read_frozen_wave1_train"]
