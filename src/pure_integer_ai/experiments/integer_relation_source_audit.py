"""Audit authored W-06 source packs before graph-course admission."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import compile_relation_seed
from pure_integer_ai.experiments.ph2_authored_relation_schema import AuthoredRelationSeed
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    validate_artifact_manifest,
    validate_dataset_bundle,
)


FOUR_WAY_SPLITS = ("train", "held_out", "negative", "conflict")
_KINDS = (
    RECORD_SOURCE_REF, RECORD_OBSERVATION,
    RECORD_TEACHER_EVIDENCE, RECORD_EVALUATOR_LABEL,
)


class IntegerRelationSourceAuditError(ValueError):
    """A source, owner, or four-way assignment is not independently closed."""


def _key(value: Sequence[int], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise IntegerRelationSourceAuditError(f"{where} must be a positive integer key")
    return tuple(value)


def audit_integer_relation_source_pack(
        pack_root: str | Path,
        authored_source_path: str | Path,
        *,
        assignments: Mapping[str, Sequence[Sequence[int]]] | None = None,
        source_row_parser: Callable[[dict, int], AuthoredRelationSeed] | None = None,
        training_only: bool = False,
        ) -> dict[str, object]:
    """Verify physical source/owner evidence, then optional four-way isolation.

    Assignments are frozen observation identities from the course, never
    inferred from sample_role, perturbation, or an evaluator's expected answer.
    The two-way source pack alone cannot qualify a four-way Wave 1 course.
    """
    root = Path(pack_root).resolve()
    source_path = Path(authored_source_path).resolve()
    if type(training_only) is not bool or (training_only and assignments is not None):
        raise IntegerRelationSourceAuditError("train-only 来源不得授权四路 assignment")
    manifest = read_artifact_manifest(root / "manifest.json")
    if (manifest.source_key != "AUTHORED_CC0_V1"
            or manifest.license_partition != "CC0-1.0"
            or manifest.redistribution_policy != "PUBLIC"
            or manifest.w_stages != ("W-06",)):
        raise IntegerRelationSourceAuditError("source/permission/stage is not the authored public W-06 contract")
    if source_path.parent.name != "ph2" or source_path.parent.parent.name != "data":
        raise IntegerRelationSourceAuditError("authored source must be a data/ph2 file")
    raw_source = source_path.read_bytes()
    if not raw_source or not raw_source.endswith(b"\n"):
        raise IntegerRelationSourceAuditError("authored source is empty or unterminated")
    source_hash = hashlib.sha256(raw_source).hexdigest()
    source_lines = raw_source.splitlines()
    source_rows = []
    source_seeds = []
    for number, line in enumerate(source_lines, start=1):
        try:
            row = parse_canonical_json_bytes(line, require_object=True)
        except ValueError as error:
            raise IntegerRelationSourceAuditError(
                f"authored source line {number} is not canonical JSON") from error
        if not isinstance(row.get("seed_id"), str) or not row["seed_id"]:
            raise IntegerRelationSourceAuditError(
                f"authored source line {number} has no seed_id")
        source_rows.append(row)
        source_seeds.append(
            source_row_parser(row, number) if source_row_parser is not None
            else AuthoredRelationSeed.from_dict(row)
        )

    records: dict[str, list] = {kind: [] for kind in _KINDS}
    for identity in manifest.files:
        records[identity.record_kind].extend(read_record_artifact(root, identity))
    sources = tuple(records[RECORD_SOURCE_REF])
    observations = tuple(records[RECORD_OBSERVATION])
    teachers = tuple(records[RECORD_TEACHER_EVIDENCE])
    evaluators = tuple(records[RECORD_EVALUATOR_LABEL])
    validate_dataset_bundle(
        sources, observations, teachers, evaluators,
        source_key=manifest.source_key,
        license_partition=manifest.license_partition,
        public_release=True,
    )
    validate_artifact_manifest(
        manifest, sources, observations, training_only=training_only)
    if training_only and (evaluators or any(item.split != "train" for item in observations)):
        raise IntegerRelationSourceAuditError("train-only 来源发现 evaluation owner")
    if len(sources) != len(source_rows):
        raise IntegerRelationSourceAuditError("source line and SourceRef counts differ")
    source_by_key = {item.stable_key: item for item in sources}
    expected_identity_prefix = f"data/ph2/{source_path.name}#"
    for source in sources:
        if not isinstance(source, SourceRefRecord):
            raise IntegerRelationSourceAuditError("source record type drifted")
        ordinal = source.record_ordinal
        if not 1 <= ordinal <= len(source_rows):
            raise IntegerRelationSourceAuditError("SourceRef line ordinal is out of bounds")
        span = source.source_span.to_value()
        if (span != {
                "line_start": ordinal,
                "line_end": ordinal,
                "relative_path": f"data/ph2/{source_path.name}",
                }
                or source.local_sha256 != source_hash
                or source.upstream_checksum != "sha256:" + source_hash
                or source.revision_id != source_rows[ordinal - 1]["seed_id"]
                or source.source_identity
                != expected_identity_prefix + source.revision_id):
            raise IntegerRelationSourceAuditError("SourceRef is not bound to its authored source line")
    observation_by_key = {item.stable_key.components: item for item in observations}
    teacher_by_observation = {}
    evaluator_by_observation = {}
    for owner_records, target in ((teachers, teacher_by_observation),
                                  (evaluators, evaluator_by_observation)):
        for item in owner_records:
            if item.observation_key in target:
                raise IntegerRelationSourceAuditError("an observation has duplicate owner labels")
            target[item.observation_key] = item
    if ({item.owner_key for item in teachers}
            & {item.owner_key for item in evaluators}):
        raise IntegerRelationSourceAuditError("teacher and evaluator owners overlap")
    for observation in observations:
        if not isinstance(observation, ObservationRecord):
            raise IntegerRelationSourceAuditError("observation record type drifted")
        source = source_by_key[observation.source_ref_key]
        row = source_rows[source.record_ordinal - 1]
        seed = source_seeds[source.record_ordinal - 1]
        rebuilt = compile_relation_seed(seed, semantic_identity=True).observation_payload.to_value()
        actual = observation.typed_payload.to_value()
        graph_fields = (
            "candidate_definition", "directionality", "endpoints",
            "relation_family", "relation_schema", "semantic_key",
            "source_anchor_span", "surface",
        )
        if (observation.split != row.get("split")
                or observation.sample_role != seed.sample_role
                or observation.perturbation_kind != seed.perturbation_kind
                or any(actual.get(key) != rebuilt.get(key) for key in graph_fields)):
            raise IntegerRelationSourceAuditError("observation does not match its source line")
        teacher = teacher_by_observation.get(observation.stable_key)
        evaluator = evaluator_by_observation.get(observation.stable_key)
        if observation.split == "train":
            if (teacher is None or evaluator is not None
                    or teacher.source_ref_key != observation.source_ref_key):
                raise IntegerRelationSourceAuditError("train teacher ownership is incomplete")
            expected = teacher.typed_evidence.to_value()
            if expected != {
                    "expected_state": seed.expected_state,
                    "expected_payload": seed.expected_payload.to_value(),
                    "seed_id": seed.seed_id,
                    }:
                raise IntegerRelationSourceAuditError("teacher evidence does not match authored annotation")
        elif (observation.split != "held_out" or teacher is not None
              or evaluator is None or evaluator.owner_mode != "read_only"):
            raise IntegerRelationSourceAuditError("evaluation owner isolation is incomplete")
        elif (evaluator.expected_state != seed.expected_state
              or evaluator.expected_payload != seed.expected_payload):
            raise IntegerRelationSourceAuditError("evaluator label does not match authored annotation")
    if (len(teacher_by_observation) != sum(item.split == "train" for item in observations)
            or len(evaluator_by_observation) != sum(
                item.split == "held_out" for item in observations)):
        raise IntegerRelationSourceAuditError("owner label coverage is incomplete")

    report: dict[str, object] = {
        "status": "PASS_SOURCE_PACK_ONLY",
        "manifest_key": list(manifest.stable_key.components),
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": source_hash,
        "source_key": manifest.source_key,
        "license_id": manifest.license_partition,
        "source_count": len(sources),
        "observation_count": len(observations),
        "teacher_count": len(teachers),
        "evaluator_count": len(evaluators),
        "four_way_qualification": "UNKNOWN_no_frozen_course_assignments",
        "free_dialogue_complete": 0,
        "course_scope": "TRAIN_ONLY" if training_only else "TRAIN_AND_EVALUATION",
    }
    if assignments is None:
        return report
    if not isinstance(assignments, Mapping) or set(assignments) != set(FOUR_WAY_SPLITS):
        raise IntegerRelationSourceAuditError("all four frozen assignments are required")
    split_by_observation: dict[tuple[int, ...], str] = {}
    for split in FOUR_WAY_SPLITS:
        values = assignments[split]
        if not isinstance(values, (tuple, list)) or not values:
            raise IntegerRelationSourceAuditError(f"{split} assignment is empty")
        for value in values:
            key = _key(value, where=f"{split} observation")
            if key in split_by_observation:
                raise IntegerRelationSourceAuditError("observation assigned to multiple splits")
            split_by_observation[key] = split
    if set(split_by_observation) != set(observation_by_key):
        raise IntegerRelationSourceAuditError("four-way assignment does not cover the pack exactly")

    cluster_owner: dict[tuple[str, StableRecordKey], str] = {}
    for key, split in split_by_observation.items():
        observation = observation_by_key[key]
        if (observation.split != ("train" if split == "train" else "held_out")):
            raise IntegerRelationSourceAuditError("course split contradicts the source observation")
        source = source_by_key[observation.source_ref_key]
        clusters = {
            "source": source.source_cluster_key,
            "document": observation.content_group_key,
            "concept": observation.dedup_cluster_key,
            "template": observation.template_group_key,
            "shape": observation.shape_group_key,
        }
        for name, cluster in clusters.items():
            owner_key = (name, cluster)
            prior = cluster_owner.setdefault(owner_key, split)
            if prior != split:
                raise IntegerRelationSourceAuditError(
                    f"{name} cluster crosses {prior}/{split}")
    report["status"] = "PASS_SOURCE_AND_FOUR_WAY_ISOLATION_ONLY"
    report["four_way_qualification"] = "PASS_PROVENANCE_ASSIGNMENT_ONLY"
    report["assigned_counts"] = {
        split: len(assignments[split]) for split in FOUR_WAY_SPLITS
    }
    return report


__all__ = [
    "FOUR_WAY_SPLITS", "IntegerRelationSourceAuditError",
    "audit_integer_relation_source_pack",
]
