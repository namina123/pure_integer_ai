"""把显式链接的 W-04 primitive 与 W-05 atomic seed 编译为同源 CC0 pack。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_atomic_compile import (
    compile_atomic_seed,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_contract import (
    PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
    PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME,
    PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
    PRIMITIVE_ATOMIC_BRIDGE_STAGES,
    PRIMITIVE_ATOMIC_BRIDGE_SUBSTAGES,
    AuthoredPrimitiveAtomicBridgeError,
    PrimitiveAtomicBridgeSeed,
    read_authored_primitive_atomic_bridge_seeds,
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
SNAPSHOT_ID = "authored-primitive-atomic-bridge-v1"
OFFICIAL_URL = "urn:pure-integer-ai:ph2:primitive-atomic-bridge-v1"
ATTRIBUTION = "Pure Integer AI PH2 authored primitive atomic bridge seed"
_STAGE_SPEC = {
    "W-04": (
        PRIMITIVE_ATOMIC_BRIDGE_SUBSTAGES[0],
        "PRIMITIVE_SURFACE_LABEL",
        "primitive-surface-mapping",
    ),
    "W-05": (
        PRIMITIVE_ATOMIC_BRIDGE_SUBSTAGES[1],
        "ATOMIC_PROPOSITION_LABEL",
        "occurrence-role-atomic-proposition",
    ),
}


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
class AuthoredPrimitiveAtomicBridgeBuild:
    """已发布的 W-04/W-05 mixed-stage pack 与公开校验报告。"""

    pack_root: Path
    manifest: ArtifactManifest
    validation: DatasetBundleValidationReport


def _source_record(
        seed: PrimitiveAtomicBridgeSeed,
        *,
        ordinal: int,
        map_path: Path,
        atomic_path: Path,
        map_sha256: str,
        atomic_sha256: str,
        source_commitment: str,
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
        PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
        SNAPSHOT_ID,
        seed.atomic.seed_id,
        OFFICIAL_URL,
        f"data/ph2/{map_path.name}#{seed.atomic.seed_id}",
        "sha256:" + source_commitment,
        source_commitment,
        PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
        "PUBLIC",
        ATTRIBUTION,
        PARSER_VERSION,
        CanonicalJsonObject.from_value({
            "atomic_relative_path": f"data/ph2/{atomic_path.name}",
            "atomic_sha256": atomic_sha256,
            "line_end": ordinal,
            "line_start": ordinal,
            "map_relative_path": f"data/ph2/{map_path.name}",
            "map_sha256": map_sha256,
        }),
        ordinal,
        _stable_key(
            "source_cluster",
            PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            seed.atomic.family,
        ),
    )


def _observation(
        compiled: AuthoredCompiledSeed,
        *,
        stage: str,
        source: SourceRefRecord,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        observation_key: StableRecordKey,
        supersedes_key: StableRecordKey | None,
        prerequisite_keys: tuple[StableRecordKey, ...],
        ) -> ObservationRecord:
    substage = _STAGE_SPEC[stage][0]
    return ObservationRecord(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        dataset_key,
        artifact_key,
        observation_key,
        stage,
        substage,
        compiled.split,
        "zh",
        f"typed-{substage.casefold().replace('_', '-')}",
        source.stable_key,
        PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
        _stable_key(
            "dedup", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage, compiled.family, *compiled.dedup_parts),
        _stable_key(
            "content", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage, compiled.family, *compiled.content_parts),
        _stable_key("template", stage, compiled.template_family),
        _stable_key(
            "shape", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage, compiled.family, *compiled.shape_parts),
        "forming" if compiled.label_owner == "teacher" else "evaluator",
        compiled.sample_role,
        compiled.payload_kind,
        compiled.observation_payload,
        compiled.perturbation_kind,
        supersedes_key,
        prerequisite_keys,
        compiled.logical_order * 2 - (1 if stage == "W-04" else 0),
    )


def _teacher(
        compiled: AuthoredCompiledSeed,
        *,
        stage: str,
        source: SourceRefRecord,
        observation: ObservationRecord,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        ) -> TeacherEvidenceRecord:
    substage, evidence_kind, _ = _STAGE_SPEC[stage]
    return TeacherEvidenceRecord(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        dataset_key,
        artifact_key,
        _stable_key(
            "teacher_evidence",
            PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage,
            compiled.seed_id,
        ),
        observation.stable_key,
        evidence_kind,
        CanonicalJsonObject.from_value({
            "expected_payload": compiled.expected_payload.to_value(),
            "expected_state": compiled.expected_state,
            "seed_id": compiled.seed_id,
        }),
        source.stable_key,
        stage,
        0,
        _stable_key(
            "owner", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            substage, "teacher"),
    )


def _evaluator(
        compiled: AuthoredCompiledSeed,
        *,
        stage: str,
        source: SourceRefRecord,
        observation: ObservationRecord,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        ) -> EvaluatorLabelRecord:
    substage, _, dimension = _STAGE_SPEC[stage]
    return EvaluatorLabelRecord(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        dataset_key,
        artifact_key,
        _stable_key(
            "evaluator_label",
            PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage,
            compiled.seed_id,
        ),
        observation.stable_key,
        _stable_key(
            "dimension", compiled.evaluation_dimension or dimension),
        compiled.expected_state,
        compiled.expected_payload,
        100,
        1,
        stage,
        _stable_key(
            "owner", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            substage, "evaluator"),
    )


def _build_records(
        seeds: tuple[PrimitiveAtomicBridgeSeed, ...],
        *,
        map_path: Path,
        atomic_path: Path,
        map_sha256: str,
        atomic_sha256: str,
        source_commitment: str,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        ) -> tuple[
            tuple[SourceRefRecord, ...],
            tuple[ObservationRecord, ...],
            tuple[TeacherEvidenceRecord, ...],
            tuple[EvaluatorLabelRecord, ...],
        ]:
    source_keys = {
        item.atomic.seed_id: _stable_key(
            "source_ref", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            source_commitment, item.atomic.seed_id)
        for item in seeds
    }
    observation_keys = {
        (item.atomic.seed_id, stage): _stable_key(
            "observation", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
            stage, source_commitment, item.atomic.seed_id)
        for item in seeds for stage in PRIMITIVE_ATOMIC_BRIDGE_STAGES
    }
    sources = []
    observations = []
    teachers = []
    evaluators = []
    for ordinal, seed in enumerate(seeds, start=1):
        source = _source_record(
            seed,
            ordinal=ordinal,
            map_path=map_path,
            atomic_path=atomic_path,
            map_sha256=map_sha256,
            atomic_sha256=atomic_sha256,
            source_commitment=source_commitment,
            dataset_key=dataset_key,
            artifact_key=artifact_key,
            source_key=source_keys[seed.atomic.seed_id],
        )
        sources.append(source)
        compiled_by_stage = {
            "W-04": seed.primitive.compiled(),
            "W-05": compile_atomic_seed(seed.atomic),
        }
        for stage in PRIMITIVE_ATOMIC_BRIDGE_STAGES:
            compiled = compiled_by_stage[stage]
            supersedes = (
                observation_keys[(compiled.supersedes_seed_id, stage)]
                if compiled.supersedes_seed_id else None
            )
            prerequisites = (
                () if stage == "W-04" else
                (observation_keys[(seed.atomic.seed_id, "W-04")],)
            )
            observation = _observation(
                compiled,
                stage=stage,
                source=source,
                dataset_key=dataset_key,
                artifact_key=artifact_key,
                observation_key=observation_keys[(
                    seed.atomic.seed_id, stage)],
                supersedes_key=supersedes,
                prerequisite_keys=prerequisites,
            )
            observations.append(observation)
            if compiled.label_owner == "teacher":
                teachers.append(_teacher(
                    compiled,
                    stage=stage,
                    source=source,
                    observation=observation,
                    dataset_key=dataset_key,
                    artifact_key=artifact_key,
                ))
            else:
                evaluators.append(_evaluator(
                    compiled,
                    stage=stage,
                    source=source,
                    observation=observation,
                    dataset_key=dataset_key,
                    artifact_key=artifact_key,
                ))
    return (
        tuple(sources), tuple(observations),
        tuple(teachers), tuple(evaluators),
    )


def compile_authored_primitive_atomic_bridge_course(
        map_path: str | Path,
        atomic_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredPrimitiveAtomicBridgeBuild:
    """编译并发布 W-04 Observation 显式前置于同源 W-05 Observation 的 pack。"""
    resolved_map = Path(map_path).resolve()
    resolved_atomic = Path(atomic_path).resolve()
    seeds = read_authored_primitive_atomic_bridge_seeds(
        resolved_map, resolved_atomic)
    map_sha256 = _sha256_file(resolved_map)
    atomic_sha256 = _sha256_file(resolved_atomic)
    source_commitment = hashlib.sha256(canonical_json_bytes({
        "atomic_sha256": atomic_sha256,
        "map_sha256": map_sha256,
    })).hexdigest()
    dataset_key = _stable_key(
        "dataset", "PH2", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY, SCHEMA_VERSION)
    artifact_key = _stable_key(
        "artifact", PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
        PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
        *PRIMITIVE_ATOMIC_BRIDGE_STAGES,
        source_commitment,
        COURSE_VERSION,
        ARTIFACT_VERSION,
    )
    sources, observations, teachers, evaluators = _build_records(
        seeds,
        map_path=resolved_map,
        atomic_path=resolved_atomic,
        map_sha256=map_sha256,
        atomic_sha256=atomic_sha256,
        source_commitment=source_commitment,
        dataset_key=dataset_key,
        artifact_key=artifact_key,
    )
    validation = validate_dataset_bundle(
        sources,
        observations,
        teachers,
        evaluators,
        source_key=PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
        license_partition=PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
        public_release=True,
    )
    pack_root = (
        Path(release_root).resolve()
        / "packs" / PRIMITIVE_ATOMIC_BRIDGE_PACK_NAME
    ).resolve()
    if pack_root.exists():
        raise AuthoredPrimitiveAtomicBridgeError(
            "primitive atomic bridge pack 已存在")
    source_clusters = tuple(sorted({
        item.source_cluster_key for item in sources}))
    source_by_key = {item.stable_key: item for item in sources}

    def clusters_for(
            items: tuple[ObservationRecord, ...],
            ) -> tuple[StableRecordKey, ...]:
        return tuple(sorted({
            source_by_key[item.source_ref_key].source_cluster_key
            for item in items
        }))

    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(
        item for item in observations if item.split == "held_out")
    if (not train or not held_out or not teachers or not evaluators
            or len(train) != len(teachers)
            or len(held_out) != len(evaluators)):
        raise AuthoredPrimitiveAtomicBridgeError(
            "bridge train/held-out/owner 分账未闭合")
    files = (
        write_record_artifact(
            sources,
            pack_root,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
                PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID, source_clusters,
            ),
        ),
        write_record_artifact(
            train,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/train.jsonl.gz", "train",
                PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID, clusters_for(train),
            ),
        ),
        write_record_artifact(
            held_out,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/held_out.jsonl.gz", "held_out",
                PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID, clusters_for(held_out),
            ),
        ),
        write_record_artifact(
            teachers,
            pack_root,
            ArtifactWriteSpec(
                RECORD_TEACHER_EVIDENCE, "teacher",
                "owners/teacher/train.evidence.jsonl.gz", "train",
                PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID, clusters_for(train),
            ),
        ),
        write_record_artifact(
            evaluators,
            pack_root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL, "evaluator",
                "owners/evaluator/held_out.labels.jsonl.gz", "held_out",
                PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID, clusters_for(held_out),
            ),
        ),
    )
    manifest = ArtifactManifest(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        dataset_key,
        artifact_key,
        PRIMITIVE_ATOMIC_BRIDGE_SOURCE_KEY,
        PRIMITIVE_ATOMIC_BRIDGE_LICENSE_ID,
        "PUBLIC",
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        files,
        ("train", "held_out"),
        PRIMITIVE_ATOMIC_BRIDGE_STAGES,
        source_clusters,
        (),
        "W-05",
    )
    validate_artifact_manifest(manifest, sources, observations)
    write_artifact_manifest(manifest, pack_root)
    return AuthoredPrimitiveAtomicBridgeBuild(
        pack_root, manifest, validation)


__all__ = [
    "ADAPTER_VERSION",
    "ARTIFACT_VERSION",
    "COURSE_VERSION",
    "GENERATOR_VERSION",
    "PARSER_VERSION",
    "AuthoredPrimitiveAtomicBridgeBuild",
    "compile_authored_primitive_atomic_bridge_course",
]
