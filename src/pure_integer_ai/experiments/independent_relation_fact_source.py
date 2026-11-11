"""Compile a licensed teacher-only fact source without reading query/eval owners."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pure_integer_ai.experiments.connected_relation_training_course import _digest_key
from pure_integer_ai.experiments.integer_relation_connected_course import (
    _FACT_FIELDS, _SPEC, _read_lines, _seed,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourse, integer_course_from_w06_adapter,
)
from pure_integer_ai.experiments.integer_relation_source_audit import audit_integer_relation_source_pack
from pure_integer_ai.experiments.ph2_authored_course_common import AuthoredCourseBuild, publish_authored_course
from pure_integer_ai.experiments.ph2_authored_relation_compile import compile_relation_seed
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION, RECORD_SOURCE_REF, RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest, read_record_artifact
from pure_integer_ai.experiments.ph2_dataset_validation import validate_dataset_bundle
from pure_integer_ai.experiments.ph2_w06_adapter import adapt_w06_training_payload
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_registry import W06_RELATION_REGISTRY
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES


@dataclass(frozen=True, slots=True)
class IndependentRelationFactSource:
    """Audited independent teacher data, not a permission to learn every record."""

    pack: AuthoredCourseBuild
    course: IntegerRelationGraphCourse
    source_audit: dict[str, object]
    source_digest_key: tuple[int, ...]
    source_path: Path


def compile_independent_relation_fact_source(
        source_path: str | Path, release_root: str | Path,
        ) -> IndependentRelationFactSource:
    """Source preparation only; no question syntax, evaluator or Core writes."""
    source = Path(source_path).resolve(strict=True)
    rows = _read_lines(source, _FACT_FIELDS)
    seeds = tuple(_seed(row, index) for index, row in enumerate(rows, 1))
    if (any(seed.split != "train" or seed.label_owner != "teacher" for seed in seeds)
            or len({seed.seed_id for seed in seeds}) != len(seeds)):
        raise ValueError("independent facts require unique train teacher sources")
    names = {}
    for seed in seeds:
        for endpoint in seed.endpoints:
            key = (endpoint.object_kind, endpoint.local_id)
            if names.setdefault(key, endpoint.surface_fragment) != endpoint.surface_fragment:
                raise ValueError("independent endpoint identity has competing names")
    digest = _digest_key(source)
    spec = replace(
        _SPEC, pack_name="AUTHORED_CC0_V1--CC0-1.0--independent-relation-facts-v1",
        substage="INDEPENDENT_RELATION_FACTS_V1",
        snapshot_id="independent-relation-facts-v1-" + "-".join(map(str, digest)),
        official_url="urn:pure-integer-ai:authored:independent-relation-facts-v1",
        attribution="Pure Integer AI original CC0 fictional independent fact evidence",
    )
    substages = {
        seed.seed_id: (
            W06_RELATION_PROFILES[seed.relation_family].substage_key,
            W06_RELATION_REGISTRY[
                W06_RELATION_PROFILES[seed.relation_family].substage_key
            ].teacher_evidence_kinds[0]) for seed in seeds
    }
    pack = publish_authored_course(
        tuple(compile_relation_seed(seed, semantic_identity=True) for seed in seeds),
        source, release_root, spec, seed_substages=substages, training_only=True)
    return read_independent_relation_fact_source(source, pack.pack_root)


def read_independent_relation_fact_source(
        source_path: str | Path, pack_root: str | Path,
        ) -> IndependentRelationFactSource:
    """Restore a prepared teacher source without publishing or rewriting its pack."""
    source, root = Path(source_path).resolve(strict=True), Path(pack_root).resolve(strict=True)
    manifest = read_artifact_manifest(root / "manifest.json")
    audit = audit_integer_relation_source_pack(root, source, source_row_parser=_seed, training_only=True)
    records = {kind: [] for kind in (RECORD_SOURCE_REF, RECORD_OBSERVATION, RECORD_TEACHER_EVIDENCE)}
    for identity in manifest.files:
        if identity.record_kind not in records:
            raise ValueError("independent facts contain an evaluation owner")
        records[identity.record_kind].extend(read_record_artifact(root, identity))
    validation = validate_dataset_bundle(
        tuple(records[RECORD_SOURCE_REF]), tuple(records[RECORD_OBSERVATION]),
        tuple(records[RECORD_TEACHER_EVIDENCE]), (), source_key=manifest.source_key,
        license_partition=manifest.license_partition, public_release=True)
    pack = AuthoredCourseBuild(root, manifest, validation)
    digest = _digest_key(source)
    course = _read_teacher_course(pack, (92034, *manifest.stable_key.components, *digest),
                                 tuple(map(ord, "CC0-1.0")))
    if (audit["status"] != "PASS_SOURCE_PACK_ONLY" or audit["evaluator_count"] != 0
            or audit["teacher_count"] != len(course.records)):
        raise ValueError("independent fact source audit is incomplete")
    return IndependentRelationFactSource(pack, course, audit, digest, source)


def _read_teacher_course(pack, source_identity, license_identity):
    """Read verified physical train artifacts, never caller-supplied teacher labels."""
    records = {kind: [] for kind in (
        RECORD_SOURCE_REF, RECORD_OBSERVATION, RECORD_TEACHER_EVIDENCE)}
    for identity in pack.manifest.files:
        if identity.record_kind not in records:
            raise ValueError("independent facts contain an evaluation owner")
        records[identity.record_kind].extend(read_record_artifact(pack.pack_root, identity))
    adapter = adapt_w06_training_payload(W06TrainingPayload(
        tuple(records[RECORD_SOURCE_REF]), tuple(records[RECORD_OBSERVATION]),
        tuple(records[RECORD_TEACHER_EVIDENCE])), require_full_curriculum=False)
    return integer_course_from_w06_adapter(
        adapter, source_identity=source_identity, license_identity=license_identity)


def verify_independent_fact_source_course(source: IndependentRelationFactSource) -> None:
    """Rejoin the proposed course with the complete independently audited pack."""
    manifest = read_artifact_manifest(source.pack.pack_root / "manifest.json")
    audit = audit_integer_relation_source_pack(
        source.pack.pack_root, source.source_path, source_row_parser=_seed, training_only=True)
    digest = _digest_key(source.source_path)
    digest_hex = "".join(value.to_bytes(4, "big").hex() for value in source.source_digest_key)
    if (manifest != source.pack.manifest or audit != source.source_audit
            or digest != source.source_digest_key
            or audit.get("source_sha256") != digest_hex
            or source.course.source_identity != (92034, *manifest.stable_key.components, *digest)
            or source.course.license_identity != tuple(map(ord, "CC0-1.0"))
            or _read_teacher_course(source.pack, source.course.source_identity,
                                    source.course.license_identity) != source.course):
        raise ValueError("independent fact teacher course or source digest drifted")


__all__ = ["IndependentRelationFactSource", "compile_independent_relation_fact_source",
           "read_independent_relation_fact_source", "verify_independent_fact_source_course"]
