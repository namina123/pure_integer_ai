"""Compile one CC0 pack with explicitly linked W-03 and W-04 records."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_contract import (
    BRIDGE_LICENSE_ID,
    BRIDGE_PACK_NAME,
    BRIDGE_SOURCE_KEY,
    BRIDGE_STAGES,
    BRIDGE_SUBSTAGES,
    AuthoredSemanticPrimitiveBridgeError,
    SemanticPrimitiveBridgeSeed,
    read_authored_semantic_primitive_bridge_seeds,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    FORMAT_VERSION,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    SCHEMA_VERSION,
    ArtifactManifest,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetBundleValidationReport,
    validate_artifact_manifest,
    validate_dataset_bundle,
)


COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
SNAPSHOT_ID = "authored-semantic-primitive-bridge-seed-v1"
OFFICIAL_URL = "urn:pure-integer-ai:ph2:semantic-primitive-bridge-v1"
ATTRIBUTION = "Pure Integer AI PH2 authored semantic primitive bridge seed"


def _stable_key(namespace: str, *parts: Any) -> StableRecordKey:
    payload = canonical_json_bytes({
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class AuthoredSemanticPrimitiveBridgeBuild:
    """Published mixed-stage pack and its public validation report."""

    pack_root: Path
    manifest: ArtifactManifest
    validation: DatasetBundleValidationReport


def _source_record(
        seed: SemanticPrimitiveBridgeSeed,
        *,
        ordinal: int,
        source_path: Path,
        sample_sha256: str,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        source_key: StableRecordKey,
        ) -> SourceRefRecord:
    return SourceRefRecord(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        dataset_key,
        artifact_key,
        source_key,
        BRIDGE_SOURCE_KEY,
        SNAPSHOT_ID,
        seed.bridge_id,
        OFFICIAL_URL,
        f"data/ph2/{source_path.name}#{seed.bridge_id}",
        "sha256:" + sample_sha256,
        sample_sha256,
        BRIDGE_LICENSE_ID,
        "PUBLIC",
        ATTRIBUTION,
        PARSER_VERSION,
        CanonicalJsonObject.from_value({
            "line_end": ordinal,
            "line_start": ordinal,
            "relative_path": f"data/ph2/{source_path.name}",
        }),
        ordinal,
        _stable_key("source_cluster", BRIDGE_SOURCE_KEY, seed.family),
    )


def _observation(
        seed: SemanticPrimitiveBridgeSeed,
        *,
        stage: str,
        source: SourceRefRecord,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        observation_key: StableRecordKey,
        supersedes_key: StableRecordKey | None,
        prerequisite_keys: tuple[StableRecordKey, ...],
        ) -> ObservationRecord:
    is_sense = stage == "W-03"
    substage = BRIDGE_SUBSTAGES[0 if is_sense else 1]
    payload_kind = "SenseBoundaryQuery" if is_sense else "PrimitiveSurfaceQuery"
    payload = (
        {
            "candidate_sense": seed.candidate_sense,
            "context": seed.context,
            "query_kind": "sense_boundary",
            "surface": seed.surface,
        }
        if is_sense else
        {
            "candidate_primitive": {
                "kind": seed.primitive_kind,
                "registry": seed.primitive_registry,
            },
            "context": seed.context,
            "query_kind": "primitive_surface_mapping",
            "surface_form": seed.surface,
        }
    )
    return ObservationRecord(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        dataset_key,
        artifact_key,
        observation_key,
        stage,
        substage,
        seed.split,
        "zh",
        f"typed-{substage.casefold().replace('_', '-')}",
        source.stable_key,
        BRIDGE_LICENSE_ID,
        _stable_key(
            "dedup", BRIDGE_SOURCE_KEY, stage, seed.family,
            seed.surface, seed.context,
            seed.candidate_sense if is_sense else seed.primitive_registry,
            0 if is_sense else seed.primitive_kind,
        ),
        _stable_key(
            "content", BRIDGE_SOURCE_KEY, seed.family,
            seed.surface, seed.context),
        _stable_key("template", stage, seed.template_family),
        _stable_key(
            "shape", stage, seed.family,
            "sense-boundary-query-v1" if is_sense
            else "primitive-surface-query-v1"),
        "forming" if seed.label_owner == "teacher" else "evaluator",
        seed.sample_role,
        payload_kind,
        CanonicalJsonObject.from_value(payload),
        seed.perturbation_kind,
        supersedes_key,
        prerequisite_keys,
        seed.logical_order * 2 - (1 if is_sense else 0),
    )


def _private_payload(
        seed: SemanticPrimitiveBridgeSeed,
        *,
        stage: str,
        ) -> tuple[str, CanonicalJsonObject]:
    if stage == "W-03":
        state = seed.sense_expected_state
        expected = seed.sense_expected_payload
    else:
        state = seed.primitive_expected_state
        expected = seed.primitive_expected_payload
    return state, CanonicalJsonObject.from_value({
        "bridge_id": seed.bridge_id,
        "expected_payload": expected.to_value(),
        "expected_state": state,
    })


def _build_records(
        seeds: tuple[SemanticPrimitiveBridgeSeed, ...],
        source_path: Path,
        sample_sha256: str,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        ) -> tuple[
            tuple[SourceRefRecord, ...],
            tuple[ObservationRecord, ...],
            tuple[TeacherEvidenceRecord, ...],
            tuple[EvaluatorLabelRecord, ...],
        ]:
    source_keys = {
        item.bridge_id: _stable_key(
            "source_ref", BRIDGE_SOURCE_KEY, sample_sha256, item.bridge_id)
        for item in seeds
    }
    observation_keys = {
        (item.bridge_id, stage): _stable_key(
            "observation", BRIDGE_SOURCE_KEY, stage,
            sample_sha256, item.bridge_id)
        for item in seeds for stage in BRIDGE_STAGES
    }
    sources = []
    observations = []
    teachers = []
    evaluators = []
    for ordinal, seed in enumerate(seeds, start=1):
        source = _source_record(
            seed,
            ordinal=ordinal,
            source_path=source_path,
            sample_sha256=sample_sha256,
            dataset_key=dataset_key,
            artifact_key=artifact_key,
            source_key=source_keys[seed.bridge_id],
        )
        sources.append(source)
        for stage in BRIDGE_STAGES:
            is_sense = stage == "W-03"
            observation_key = observation_keys[(seed.bridge_id, stage)]
            supersedes = (
                observation_keys[(seed.supersedes_bridge_id, stage)]
                if seed.supersedes_bridge_id else None
            )
            prerequisites = (
                () if is_sense else
                (observation_keys[(seed.bridge_id, "W-03")],)
            )
            observation = _observation(
                seed,
                stage=stage,
                source=source,
                dataset_key=dataset_key,
                artifact_key=artifact_key,
                observation_key=observation_key,
                supersedes_key=supersedes,
                prerequisite_keys=prerequisites,
            )
            observations.append(observation)
            state, private_payload = _private_payload(seed, stage=stage)
            if seed.label_owner == "teacher":
                teachers.append(TeacherEvidenceRecord(
                    FORMAT_VERSION,
                    SCHEMA_VERSION,
                    COURSE_VERSION,
                    dataset_key,
                    artifact_key,
                    _stable_key(
                        "teacher_evidence", BRIDGE_SOURCE_KEY,
                        stage, seed.bridge_id),
                    observation.stable_key,
                    ("SENSE_BOUNDARY_LABEL" if is_sense
                     else "PRIMITIVE_SURFACE_LABEL"),
                    private_payload,
                    source.stable_key,
                    stage,
                    0,
                    _stable_key(
                        "owner", BRIDGE_SOURCE_KEY, stage, "teacher"),
                ))
            else:
                expected = (
                    seed.sense_expected_payload if is_sense
                    else seed.primitive_expected_payload)
                evaluators.append(EvaluatorLabelRecord(
                    FORMAT_VERSION,
                    SCHEMA_VERSION,
                    COURSE_VERSION,
                    dataset_key,
                    artifact_key,
                    _stable_key(
                        "evaluator_label", BRIDGE_SOURCE_KEY,
                        stage, seed.bridge_id),
                    observation.stable_key,
                    _stable_key(
                        "dimension", BRIDGE_SOURCE_KEY,
                        "sense" if is_sense else "primitive"),
                    state,
                    expected,
                    100,
                    1,
                    stage,
                    _stable_key(
                        "owner", BRIDGE_SOURCE_KEY, stage, "evaluator"),
                ))
    return (
        tuple(sources), tuple(observations),
        tuple(teachers), tuple(evaluators),
    )


def compile_authored_semantic_primitive_bridge_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredSemanticPrimitiveBridgeBuild:
    """Publish linked mixed-stage records without exposing evaluator labels."""
    source_path = Path(sample_path).resolve()
    seeds = read_authored_semantic_primitive_bridge_seeds(source_path)
    sample_sha256 = _sha256_file(source_path)
    dataset_key = _stable_key(
        "dataset", "PH2", BRIDGE_SOURCE_KEY, SCHEMA_VERSION)
    artifact_key = _stable_key(
        "artifact", BRIDGE_SOURCE_KEY, BRIDGE_LICENSE_ID,
        BRIDGE_STAGES, sample_sha256, COURSE_VERSION, ARTIFACT_VERSION)
    sources, observations, teachers, evaluators = _build_records(
        seeds, source_path, sample_sha256, dataset_key, artifact_key)
    validation = validate_dataset_bundle(
        sources,
        observations,
        teachers,
        evaluators,
        source_key=BRIDGE_SOURCE_KEY,
        license_partition=BRIDGE_LICENSE_ID,
        public_release=True,
    )
    pack_root = (
        Path(release_root).resolve() / "packs" / BRIDGE_PACK_NAME).resolve()
    if pack_root.exists():
        raise AuthoredSemanticPrimitiveBridgeError(
            "bridge pack already exists; use a new artifact version")
    source_clusters = tuple(sorted({item.source_cluster_key for item in sources}))
    source_by_key = {item.stable_key: item for item in sources}

    def clusters_for(items: tuple[ObservationRecord, ...]) -> tuple[StableRecordKey, ...]:
        return tuple(sorted({
            source_by_key[item.source_ref_key].source_cluster_key for item in items
        }))

    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    files = (
        write_record_artifact(
            sources,
            pack_root,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
                BRIDGE_LICENSE_ID, source_clusters),
        ),
        write_record_artifact(
            train,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/train.jsonl.gz", "train",
                BRIDGE_LICENSE_ID, clusters_for(train)),
        ),
        write_record_artifact(
            held_out,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/held_out.jsonl.gz", "held_out",
                BRIDGE_LICENSE_ID, clusters_for(held_out)),
        ),
        write_record_artifact(
            teachers,
            pack_root,
            ArtifactWriteSpec(
                RECORD_TEACHER_EVIDENCE, "teacher",
                "owners/teacher/train.evidence.jsonl.gz", "train",
                BRIDGE_LICENSE_ID, clusters_for(train)),
        ),
        write_record_artifact(
            evaluators,
            pack_root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL, "evaluator",
                "owners/evaluator/held_out.labels.jsonl.gz", "held_out",
                BRIDGE_LICENSE_ID, clusters_for(held_out)),
        ),
    )
    manifest = ArtifactManifest(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        dataset_key,
        artifact_key,
        BRIDGE_SOURCE_KEY,
        BRIDGE_LICENSE_ID,
        "PUBLIC",
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        files,
        ("train", "held_out"),
        BRIDGE_STAGES,
        source_clusters,
        (),
        "W-03",
    )
    validate_artifact_manifest(manifest, sources, observations)
    write_artifact_manifest(manifest, pack_root)
    return AuthoredSemanticPrimitiveBridgeBuild(
        pack_root, manifest, validation)


__all__ = [
    "AuthoredSemanticPrimitiveBridgeBuild",
    "compile_authored_semantic_primitive_bridge_course",
]
