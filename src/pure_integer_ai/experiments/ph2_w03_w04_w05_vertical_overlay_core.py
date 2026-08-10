"""为一个或多个来源命题组装严格的 W03→W04→W05 公开纵向 overlay。"""
from __future__ import annotations

from dataclasses import replace
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
    build_w03_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core_contract import (
    VerticalOverlayProjection,
    VerticalOverlayTargetProjection,
    VerticalOverlayTargetSpec,
    W03W04W05VerticalOverlayCoreError,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    build_w05_v2_public_evaluation_batch,
)


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
        raise W03W04W05VerticalOverlayCoreError(
            f"{where} must resolve to one record")
    return selected[0]


def _manifest_sha(build) -> str:
    return _sha(build.manifest.to_dict())


def _base_sample_sha(
        sources: tuple[SourceRefRecord, ...],
        ) -> str:
    values = {item.local_sha256 for item in sources}
    if len(values) != 1:
        raise W03W04W05VerticalOverlayCoreError(
            "base source sample SHA inventory is not uniform")
    value = next(iter(values))
    if not isinstance(value, str) or len(value) != 64:
        raise W03W04W05VerticalOverlayCoreError(
            "base source sample SHA drifted")
    return value


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
        raise W03W04W05VerticalOverlayCoreError(
            "donor sample SHA inventories are not uniform")
    map_sha = next(iter(map_hashes))
    atomic_sha = next(iter(atomic_hashes))
    if (not isinstance(map_sha, str) or len(map_sha) != 64
            or not isinstance(atomic_sha, str) or len(atomic_sha) != 64):
        raise W03W04W05VerticalOverlayCoreError(
            "donor sample SHA metadata drifted")
    return map_sha, atomic_sha


def _predicate_projection(
        observation: ObservationRecord,
        ) -> str:
    payload = observation.typed_payload.to_value()
    definition = payload.get("candidate_definition")
    occurrences = payload.get("occurrences")
    if not isinstance(definition, dict) or not isinstance(occurrences, list):
        raise W03W04W05VerticalOverlayCoreError(
            "W05 candidate definition or occurrence inventory drifted")
    anchor = definition.get("source_anchor_key")
    matching = tuple(
        item for item in occurrences
        if isinstance(item, dict) and item.get("identity_key") == anchor)
    occurrence = _one(matching, where="W05 predicate occurrence")
    surface = occurrence.get("surface_fragment")
    if not isinstance(surface, str) or not surface:
        raise W03W04W05VerticalOverlayCoreError(
            "W05 predicate projection drifted")
    return surface


def _primitive_projection(
        observation: ObservationRecord,
        ) -> str:
    payload = observation.typed_payload.to_value()
    primitive = payload.get("candidate_primitive")
    surface = payload.get("surface_form")
    if (not isinstance(primitive, dict)
            or not isinstance(surface, str) or not surface):
        raise W03W04W05VerticalOverlayCoreError(
            "W04 primitive projection drifted")
    if (not isinstance(primitive.get("registry"), str)
            or type(primitive.get("kind")) is not int
            or primitive["kind"] <= 0):
        raise W03W04W05VerticalOverlayCoreError(
            "W04 primitive coordinate drifted")
    return surface


def _validate_dependency(
        w04: ObservationRecord,
        w05: ObservationRecord,
        ) -> None:
    w04_surface = _primitive_projection(w04)
    w05_surface = _predicate_projection(w05)
    if (w05.prerequisite_keys != (w04.stable_key,)
            or w05_surface != w04_surface):
        raise W03W04W05VerticalOverlayCoreError(
            "W05 external prerequisite is not exact")


def build_vertical_overlay_projection(
        base: AuthoredSemanticPrimitiveBridgeBuild,
        donor: AuthoredPrimitiveAtomicBridgeBuild,
        targets: tuple[VerticalOverlayTargetSpec, ...],
        ) -> VerticalOverlayProjection:
    """以来源、前置 Observation、predicate occurrence 三重身份组装 overlay。"""
    if (not isinstance(base, AuthoredSemanticPrimitiveBridgeBuild)
            or not isinstance(donor, AuthoredPrimitiveAtomicBridgeBuild)
            or not isinstance(targets, tuple) or not targets
            or any(not isinstance(item, VerticalOverlayTargetSpec)
                   for item in targets)):
        raise TypeError("vertical overlay projection inputs are invalid")
    target_order = tuple(sorted(
        targets,
        key=lambda item: (
            item.surface, item.context, item.proposition_surface),
    ))
    if len({
            (item.surface, item.context, item.proposition_surface)
            for item in target_order
            }) != len(target_order):
        raise W03W04W05VerticalOverlayCoreError(
            "vertical overlay targets are duplicated")

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
        raise W03W04W05VerticalOverlayCoreError(
            "vertical overlay input record type drifted")

    donor_train_w05 = tuple(
        item for item in donor_observations
        if item.w_stage == "W-05" and item.split == "train")
    donor_w04_by_source = {
        item.source_ref_key: item for item in donor_observations
        if item.w_stage == "W-04" and item.split == "train"}
    donor_evidence_by_observation = {
        item.observation_key: item for item in donor_teachers
        if item.visible_from_stage == "W-05"}
    if (not donor_train_w05
            or set(donor_evidence_by_observation)
            != {item.stable_key for item in donor_train_w05}):
        raise W03W04W05VerticalOverlayCoreError(
            "donor W05 Observation/Evidence inventory is not closed")
    for observation in donor_train_w05:
        primitive = donor_w04_by_source.get(observation.source_ref_key)
        if primitive is None:
            raise W03W04W05VerticalOverlayCoreError(
                "donor W05 prerequisite W04 is missing")
        _validate_dependency(primitive, observation)

    projections = []
    selected_w05_keys = set()
    dependency_w04 = []
    for spec in target_order:
        base_w04 = _one(
            (
                item for item in base_observations
                if (item.w_stage == "W-04" and item.split == "train"
                    and item.typed_payload.to_value().get("surface_form")
                    == spec.surface
                    and item.typed_payload.to_value().get("context")
                    == spec.context)
            ),
            where="target base W04 Observation",
        )
        base_w03 = _one(
            (
                item for item in base_observations
                if item.stable_key in base_w04.prerequisite_keys
            ),
            where="target base W03 prerequisite",
        )
        if (base_w03.w_stage != "W-03"
                or base_w03.source_ref_key != base_w04.source_ref_key
                or base_w04.prerequisite_keys != (base_w03.stable_key,)):
            raise W03W04W05VerticalOverlayCoreError(
                "target W03→W04 prerequisite is not exact")
        base_source = _one(
            (
                item for item in base_sources
                if item.stable_key == base_w04.source_ref_key
            ),
            where="target base SourceRef",
        )
        donor_w05 = _one(
            (
                item for item in donor_train_w05
                if (item.perturbation_kind == "OCCURRENCE_RESTORE"
                    and item.typed_payload.to_value().get("surface")
                    == spec.proposition_surface
                    and _predicate_projection(item) == spec.surface)
            ),
            where="target donor W05 restore Observation",
        )
        if donor_w05.stable_key in selected_w05_keys:
            raise W03W04W05VerticalOverlayCoreError(
                "one donor W05 record served multiple targets")
        selected_w05_keys.add(donor_w05.stable_key)
        base_surface = _primitive_projection(base_w04)
        donor_surface = _predicate_projection(donor_w05)
        if base_surface != donor_surface:
            raise W03W04W05VerticalOverlayCoreError(
                "target W04 and donor W05 predicate surfaces differ")
        overlay_w05 = replace(
            donor_w05,
            source_ref_key=base_source.stable_key,
            prerequisite_keys=(base_w04.stable_key,),
        )
        donor_evidence = donor_evidence_by_observation[donor_w05.stable_key]
        overlay_evidence = replace(
            donor_evidence,
            source_ref_key=base_source.stable_key,
        )
        if (overlay_w05.supersedes_key is None
                or overlay_w05.supersedes_key
                not in {item.stable_key for item in donor_train_w05}):
            raise W03W04W05VerticalOverlayCoreError(
                "target W05 restore lacks its omission predecessor")
        projections.append(VerticalOverlayTargetProjection(
            spec,
            base_source,
            base_w03,
            base_w04,
            donor_w05,
            overlay_w05,
            overlay_evidence,
        ))
        dependency_w04.append(base_w04)

    non_target_w05 = tuple(
        item for item in donor_train_w05
        if item.stable_key not in selected_w05_keys)
    for observation in non_target_w05:
        dependency_w04.append(
            donor_w04_by_source[observation.source_ref_key])
    overlay_by_key = {
        item.donor_w05_observation.stable_key: item
        for item in projections
    }
    w05_observations = tuple(sorted(
        (
            overlay_by_key[item.stable_key].overlay_w05_observation
            if item.stable_key in overlay_by_key else item
            for item in donor_train_w05
        ),
        key=lambda item: item.stable_key,
    ))
    w05_evidence = tuple(sorted(
        (
            overlay_by_key[item.stable_key].overlay_w05_evidence
            if item.stable_key in overlay_by_key
            else donor_evidence_by_observation[item.stable_key]
            for item in donor_train_w05
        ),
        key=lambda item: item.stable_key,
    ))
    source_by_key = {item.stable_key: item for item in donor_sources}
    source_by_key.update(
        (item.base_source.stable_key, item.base_source)
        for item in projections
    )
    w05_source_keys = {item.source_ref_key for item in w05_observations}
    if not w05_source_keys.issubset(source_by_key):
        raise W03W04W05VerticalOverlayCoreError(
            "overlay W05 source inventory is incomplete")
    w05_sources = tuple(sorted(
        (source_by_key[key] for key in w05_source_keys),
        key=lambda item: item.stable_key,
    ))

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
    return VerticalOverlayProjection(
        w03_batch,
        w04_batch,
        w05_batch,
        tuple(projections),
        tuple(sorted(
            {item.stable_key: item for item in dependency_w04}.values(),
            key=lambda item: item.stable_key,
        )),
        _base_sample_sha(base_sources),
        _manifest_sha(base),
        *_donor_sample_hashes(donor_sources),
        _manifest_sha(donor),
    )


__all__ = [
    "VerticalOverlayProjection",
    "VerticalOverlayTargetProjection",
    "VerticalOverlayTargetSpec",
    "W03W04W05VerticalOverlayCoreError",
    "build_vertical_overlay_projection",
]
