"""Strict derived overlay joining the frozen W-03/W-04 pack to W-05."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Iterable

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    AuthoredPrimitiveAtomicBridgeBuild,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    AuthoredSemanticPrimitiveBridgeBuild,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_record_artifact
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
    build_w03_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
    build_w05_v2_public_evaluation_batch,
)


W03_W04_BASE_SAMPLE_SHA256 = (
    "7157f92fcef678da2a0b1d1772adf42bbb30696b66ef8a148b455123b3e0ff23")
W03_W04_BASE_MANIFEST_SHA256 = (
    "c4e9bcb0955fe8161a68e151abf39d3fe3b69c615816cbe618e72fb3be8e4810")
W03_W04_BASE_BRIDGE_SHA256 = (
    "ff10f3f70dcaa7239f911e37dd05a2ae7796570117f042bff08afee1ad6c6209")
W04_W05_DONOR_MAP_SHA256 = (
    "55353709b06f3d682ad4520714e017db748b7cc685486f1d0485e9b038f02e80")
W04_W05_DONOR_ATOMIC_SHA256 = (
    "5c07642ae710f521bed6dabbd1554be3a698157538bdc55211ef72aa3bd01cd9")
W04_W05_DONOR_MANIFEST_SHA256 = (
    "1d569809d8cea3725353759d518d8513a9cbd5c245ddf6de6c8f5e67d91764cb")
VERTICAL_SURFACE = "使得"
VERTICAL_CONTEXT = "暴雨使得河水上涨。"
VERTICAL_PROPOSITION_SURFACE = VERTICAL_CONTEXT
VERTICAL_OVERLAY_VALIDATION_SHA256 = (
    "bd979fd9f27918b0df629fca3d8fd6d6816e729afcded0f0a74d49f8bb826d2b")
W04_W05_OVERLAY_BRIDGE_SHA256 = (
    "269bf786841ef4a58fd9c3f430264020747c96081e8a005f15460847da1db8f6")
W03_W04_W05_VERTICAL_RESULT_SHA256 = (
    "601e738c15a9191171ef9d72524de1c6c21b2e02f4f9de009665be687ddafdf0")
_TRAIN_PERTURBATIONS = frozenset({
    "NONE",
    "ROLE_SWAP",
    "ORDER_REVERSAL",
    "SCOPE_SHIFT",
    "OCCURRENCE_OMISSION",
    "OCCURRENCE_RESTORE",
})


# object-model: exception
class W03W04W05VerticalOverlayError(ValueError):
    """A frozen base, donor record, or external prerequisite drifted."""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _records(build, kind: str) -> tuple[object, ...]:
    values = []
    for identity in build.manifest.files:
        if identity.record_kind == kind:
            values.extend(read_record_artifact(build.pack_root, identity))
    return tuple(values)


def _one(values: Iterable[object], *, where: str):
    selected = tuple(values)
    if len(selected) != 1:
        raise W03W04W05VerticalOverlayError(
            f"{where} must resolve to one record")
    return selected[0]


def _predicate_surface(observation: ObservationRecord) -> str:
    payload = observation.typed_payload.to_value()
    definition = payload.get("candidate_definition")
    occurrences = payload.get("occurrences")
    if not isinstance(definition, dict) or not isinstance(occurrences, list):
        raise W03W04W05VerticalOverlayError(
            "W-05 candidate definition or occurrence inventory drifted")
    anchor = definition.get("source_anchor_key")
    matching = tuple(
        item for item in occurrences
        if isinstance(item, dict) and item.get("identity_key") == anchor)
    occurrence = _one(matching, where="W-05 predicate occurrence")
    surface = occurrence.get("surface_fragment")
    if not isinstance(surface, str) or not surface:
        raise W03W04W05VerticalOverlayError(
            "W-05 predicate occurrence surface drifted")
    return surface


def _manifest_sha(build) -> str:
    return _sha(build.manifest.to_dict())


def _donor_sample_hashes(
        sources: tuple[SourceRefRecord, ...],
        ) -> tuple[str, str]:
    map_hashes = set()
    atomic_hashes = set()
    for source in sources:
        metadata = source.source_span.to_value()
        map_hashes.add(metadata.get("map_sha256"))
        atomic_hashes.add(metadata.get("atomic_sha256"))
    if len(map_hashes) != 1 or len(atomic_hashes) != 1:
        raise W03W04W05VerticalOverlayError(
            "donor source sample hashes are not uniform")
    map_sha = next(iter(map_hashes))
    atomic_sha = next(iter(atomic_hashes))
    if not isinstance(map_sha, str) or not isinstance(atomic_sha, str):
        raise W03W04W05VerticalOverlayError(
            "donor source sample hash metadata drifted")
    return map_sha, atomic_sha


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalOverlay:
    """Three public batches plus the exact externally validated dependency."""

    w03_batch: W03V2PublicEvaluationBatch
    w04_batch: W04V2PublicEvaluationBatch
    w05_batch: W05V2PublicEvaluationBatch
    base_w03_observation: ObservationRecord
    base_w04_observation: ObservationRecord
    overlay_w05_observation: ObservationRecord
    dependency_w04_observations: tuple[ObservationRecord, ...]
    base_sample_sha256: str
    base_manifest_sha256: str
    donor_map_sha256: str
    donor_atomic_sha256: str
    donor_manifest_sha256: str
    validation_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.w03_batch, W03V2PublicEvaluationBatch)
                or not isinstance(self.w04_batch, W04V2PublicEvaluationBatch)
                or not isinstance(self.w05_batch, W05V2PublicEvaluationBatch)
                or not isinstance(self.base_w03_observation, ObservationRecord)
                or not isinstance(self.base_w04_observation, ObservationRecord)
                or not isinstance(self.overlay_w05_observation, ObservationRecord)
                or not self.dependency_w04_observations
                or any(not isinstance(item, ObservationRecord)
                       for item in self.dependency_w04_observations)):
            raise W03W04W05VerticalOverlayError(
                "vertical overlay record inventory drifted")
        for value in (
                self.base_sample_sha256,
                self.base_manifest_sha256,
                self.donor_map_sha256,
                self.donor_atomic_sha256,
                self.donor_manifest_sha256,
                self.validation_sha256):
            if not isinstance(value, str) or len(value) != 64:
                raise W03W04W05VerticalOverlayError(
                    "vertical overlay SHA identity drifted")


def build_w03_w04_w05_vertical_overlay(
        base: AuthoredSemanticPrimitiveBridgeBuild,
        donor: AuthoredPrimitiveAtomicBridgeBuild,
        ) -> W03W04W05VerticalOverlay:
    """Replace only the donor restore SourceRef/prerequisite with the old base."""
    if (not isinstance(base, AuthoredSemanticPrimitiveBridgeBuild)
            or not isinstance(donor, AuthoredPrimitiveAtomicBridgeBuild)):
        raise TypeError("vertical overlay builds are invalid")
    base_manifest_sha = _manifest_sha(base)
    donor_manifest_sha = _manifest_sha(donor)
    if base_manifest_sha != W03_W04_BASE_MANIFEST_SHA256:
        raise W03W04W05VerticalOverlayError("W-03/W-04 base manifest drifted")
    if donor_manifest_sha != W04_W05_DONOR_MANIFEST_SHA256:
        raise W03W04W05VerticalOverlayError("W-05 donor manifest drifted")

    base_sources = tuple(_records(base, RECORD_SOURCE_REF))
    base_observations = tuple(_records(base, RECORD_OBSERVATION))
    base_teachers = tuple(_records(base, RECORD_TEACHER_EVIDENCE))
    donor_sources = tuple(_records(donor, RECORD_SOURCE_REF))
    donor_observations = tuple(_records(donor, RECORD_OBSERVATION))
    donor_teachers = tuple(_records(donor, RECORD_TEACHER_EVIDENCE))
    if (any(not isinstance(item, SourceRefRecord)
            for item in (*base_sources, *donor_sources))
            or any(not isinstance(item, ObservationRecord)
                   for item in (*base_observations, *donor_observations))
            or any(not isinstance(item, TeacherEvidenceRecord)
                   for item in (*base_teachers, *donor_teachers))):
        raise W03W04W05VerticalOverlayError(
            "vertical overlay input record type drifted")
    base_sample_hashes = {item.local_sha256 for item in base_sources}
    if base_sample_hashes != {W03_W04_BASE_SAMPLE_SHA256}:
        raise W03W04W05VerticalOverlayError("W-03/W-04 base sample drifted")
    donor_map_sha, donor_atomic_sha = _donor_sample_hashes(donor_sources)
    if (donor_map_sha != W04_W05_DONOR_MAP_SHA256
            or donor_atomic_sha != W04_W05_DONOR_ATOMIC_SHA256):
        raise W03W04W05VerticalOverlayError("W-05 donor sample drifted")

    base_w04 = _one(
        (
            item for item in base_observations
            if (item.w_stage == "W-04" and item.split == "train"
                and item.typed_payload.to_value().get("surface_form")
                == VERTICAL_SURFACE
                and item.typed_payload.to_value().get("context")
                == VERTICAL_CONTEXT)
        ),
        where="vertical base W-04 Observation",
    )
    base_w03 = _one(
        (
            item for item in base_observations
            if item.stable_key in base_w04.prerequisite_keys
        ),
        where="vertical base W-03 prerequisite",
    )
    if (base_w03.w_stage != "W-03"
            or base_w03.source_ref_key != base_w04.source_ref_key
            or base_w04.prerequisite_keys != (base_w03.stable_key,)):
        raise W03W04W05VerticalOverlayError(
            "base W-03 -> W-04 prerequisite is not exact")
    base_source = _one(
        (
            item for item in base_sources
            if item.stable_key == base_w04.source_ref_key
        ),
        where="vertical base SourceRef",
    )

    donor_train_w05 = tuple(
        item for item in donor_observations
        if item.w_stage == "W-05" and item.split == "train")
    by_perturbation = {
        item.perturbation_kind: item for item in donor_train_w05}
    if (len(by_perturbation) != len(donor_train_w05)
            or frozenset(by_perturbation) != _TRAIN_PERTURBATIONS):
        raise W03W04W05VerticalOverlayError(
            "W-05 donor train perturbations are not the frozen six")
    donor_restore = by_perturbation["OCCURRENCE_RESTORE"]
    if (donor_restore.typed_payload.to_value().get("surface")
            != VERTICAL_PROPOSITION_SURFACE
            or _predicate_surface(donor_restore) != VERTICAL_SURFACE):
        raise W03W04W05VerticalOverlayError(
            "W-05 donor restore does not carry the vertical predicate")
    overlay_w05 = replace(
        donor_restore,
        source_ref_key=base_source.stable_key,
        prerequisite_keys=(base_w04.stable_key,),
    )
    donor_evidence_by_observation = {
        item.observation_key: item for item in donor_teachers}
    donor_restore_evidence = donor_evidence_by_observation.get(
        donor_restore.stable_key)
    if donor_restore_evidence is None:
        raise W03W04W05VerticalOverlayError(
            "W-05 donor restore Evidence is missing")
    overlay_w05_evidence = replace(
        donor_restore_evidence,
        source_ref_key=base_source.stable_key,
    )

    donor_w04_by_source = {
        item.source_ref_key: item for item in donor_observations
        if item.w_stage == "W-04" and item.split == "train"}
    non_restore_w05 = tuple(
        item for item in donor_train_w05
        if item.stable_key != donor_restore.stable_key)
    dependency_w04 = [base_w04]
    for observation in non_restore_w05:
        primitive = donor_w04_by_source.get(observation.source_ref_key)
        if (primitive is None
                or observation.prerequisite_keys != (primitive.stable_key,)
                or _predicate_surface(observation)
                != primitive.typed_payload.to_value().get("surface_form")):
            raise W03W04W05VerticalOverlayError(
                "donor W-05 external prerequisite is not exact")
        dependency_w04.append(primitive)
    w05_observations = tuple(sorted(
        (*non_restore_w05, overlay_w05),
        key=lambda item: item.stable_key,
    ))
    w05_evidence = tuple(sorted(
        (
            *(donor_evidence_by_observation[item.stable_key]
              for item in non_restore_w05),
            overlay_w05_evidence,
        ),
        key=lambda item: item.stable_key,
    ))
    source_by_key = {item.stable_key: item for item in donor_sources}
    source_by_key[base_source.stable_key] = base_source
    w05_sources = tuple(sorted(
        (source_by_key[key]
         for key in {item.source_ref_key for item in w05_observations}),
        key=lambda item: item.stable_key,
    ))
    if (len(w05_observations) != 6 or len(w05_evidence) != 6
            or len(w05_sources) != 6
            or overlay_w05.prerequisite_keys != (base_w04.stable_key,)
            or overlay_w05.source_ref_key != base_source.stable_key
            or overlay_w05.supersedes_key
            != by_perturbation["OCCURRENCE_OMISSION"].stable_key):
        raise W03W04W05VerticalOverlayError(
            "vertical W-05 overlay inventory or links drifted")

    base_w03_train = tuple(
        item for item in base_observations
        if item.w_stage == "W-03" and item.split == "train")
    base_w04_train = tuple(
        item for item in base_observations
        if item.w_stage == "W-04" and item.split == "train")
    base_w03_teachers = tuple(
        item for item in base_teachers if item.visible_from_stage == "W-03")
    base_w04_teachers = tuple(
        item for item in base_teachers if item.visible_from_stage == "W-04")
    w03_batch = build_w03_v2_public_evaluation_batch(W03TrainingPayload(
        base_sources, base_w03_train, base_w03_teachers))
    w04_batch = build_w04_v2_public_evaluation_batch(W04TrainingPayload(
        base_sources, base_w04_train, base_w04_teachers))
    w05_batch = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
        w05_sources, w05_observations, w05_evidence))
    validation_sha = _sha({
        "base_manifest_sha256": base_manifest_sha,
        "base_sample_sha256": W03_W04_BASE_SAMPLE_SHA256,
        "dependency_w04": [
            item.to_dict() for item in sorted(
                dependency_w04, key=lambda value: value.stable_key)
        ],
        "donor_atomic_sha256": donor_atomic_sha,
        "donor_manifest_sha256": donor_manifest_sha,
        "donor_map_sha256": donor_map_sha,
        "overlay_w05": overlay_w05.to_dict(),
        "overlay_w05_evidence": overlay_w05_evidence.to_dict(),
        "policy": "EXACT_EXTERNAL_PREREQUISITES_NO_SURFACE_FALLBACK",
        "w03_source_binding_sha256": w03_batch.source_binding.sha256(),
        "w04_source_binding_sha256": w04_batch.source_binding.sha256(),
        "w05_source_binding_sha256": w05_batch.source_binding.sha256(),
    })
    if validation_sha != VERTICAL_OVERLAY_VALIDATION_SHA256:
        raise W03W04W05VerticalOverlayError(
            "vertical overlay validation commitment drifted")
    return W03W04W05VerticalOverlay(
        w03_batch,
        w04_batch,
        w05_batch,
        base_w03,
        base_w04,
        overlay_w05,
        tuple(sorted(dependency_w04, key=lambda item: item.stable_key)),
        W03_W04_BASE_SAMPLE_SHA256,
        base_manifest_sha,
        donor_map_sha,
        donor_atomic_sha,
        donor_manifest_sha,
        validation_sha,
    )


__all__ = [
    "VERTICAL_CONTEXT",
    "VERTICAL_OVERLAY_VALIDATION_SHA256",
    "VERTICAL_PROPOSITION_SURFACE",
    "VERTICAL_SURFACE",
    "W03_W04_W05_VERTICAL_RESULT_SHA256",
    "W03_W04_BASE_BRIDGE_SHA256",
    "W03_W04_BASE_MANIFEST_SHA256",
    "W03_W04_BASE_SAMPLE_SHA256",
    "W04_W05_DONOR_ATOMIC_SHA256",
    "W04_W05_DONOR_MANIFEST_SHA256",
    "W04_W05_DONOR_MAP_SHA256",
    "W04_W05_OVERLAY_BRIDGE_SHA256",
    "W03W04W05VerticalOverlay",
    "W03W04W05VerticalOverlayError",
    "build_w03_w04_w05_vertical_overlay",
]
