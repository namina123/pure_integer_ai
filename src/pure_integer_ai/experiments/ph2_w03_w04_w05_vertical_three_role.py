"""冻结两条来源绑定的三 Role W03 -> W04 -> W05 纵向链。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    AuthoredPrimitiveAtomicBridgeBuild,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    AuthoredSemanticPrimitiveBridgeBuild,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core import (
    build_vertical_overlay_projection,
    vertical_overlay_validation_sha256,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core_contract import (
    VerticalOverlayProjection,
    VerticalOverlayTargetProjection,
    VerticalOverlayTargetSpec,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_registry import (
    VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


THREE_ROLE_BASE_SAMPLE_SHA256 = (
    "7978cce4cc91f0967319a5b837d06a88e0326b5f391c66aa004ea1bb594e23a4")
THREE_ROLE_BASE_MANIFEST_SHA256 = (
    "f6ab7b60022aa9d6dcd694f5d9baf7e32927db531eb31b687a5cafbcddde4373")
THREE_ROLE_DONOR_MAP_SHA256 = (
    "319e828f06a67104398e08514e96191adf963ff858f58297ac351a625ac2cbe4")
THREE_ROLE_DONOR_ATOMIC_SHA256 = (
    "73fe807061351c4493e65cb409854337887bc23a3aff610c902b27eebd1993cb")
THREE_ROLE_DONOR_MANIFEST_SHA256 = (
    "9dfd2ffe83f461d838892a6045891aa16035b9d9e0818caa6c9eed87c7013566")
THREE_ROLE_VERTICAL_TARGETS = (
    VerticalOverlayTargetSpec(
        "使得",
        "暴雨在山区使得河水上涨。",
        "暴雨在山区使得河水上涨。",
    ),
    VerticalOverlayTargetSpec(
        "使得",
        "寒潮在桥面使得路面结冰。",
        "寒潮在桥面使得路面结冰。",
    ),
)


# object-model: exception
class W03W04W05ThreeRoleVerticalError(ValueError):
    """公开三 Role 课程或精确前置依赖发生漂移。"""


def _target_projection(
        projection: VerticalOverlayProjection,
        spec: VerticalOverlayTargetSpec,
        ) -> VerticalOverlayTargetProjection:
    values = tuple(item for item in projection.targets if item.spec == spec)
    if len(values) != 1:
        raise W03W04W05ThreeRoleVerticalError(
            "three-Role target projection is not unique")
    return values[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05ThreeRoleVerticalOverlay:
    """保存两条三 Role 事实的公开批次和冻结证明身份。"""

    projection: VerticalOverlayProjection
    validation_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.projection, VerticalOverlayProjection)
                or self.validation_sha256
                != VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256):
            raise W03W04W05ThreeRoleVerticalError(
                "three-Role vertical overlay identity drifted")
        for item in self.projection.targets:
            payload = item.overlay_w05_observation.typed_payload.to_value()
            definition = payload.get("candidate_definition")
            occurrences = payload.get("occurrences")
            bindings = (
                definition.get("role_bindings")
                if isinstance(definition, dict) else None
            )
            if (not isinstance(definition, dict)
                    or not isinstance(bindings, list)
                    or not isinstance(occurrences, list)
                    or len(bindings) != 3
                    or len(occurrences) != 4):
                raise W03W04W05ThreeRoleVerticalError(
                    "target does not retain three roles and full occurrences")

    @property
    def w03_batch(self) -> W03V2PublicEvaluationBatch:
        return self.projection.w03_batch

    @property
    def w04_batch(self) -> W04V2PublicEvaluationBatch:
        return self.projection.w04_batch

    @property
    def w05_batch(self) -> W05V2PublicEvaluationBatch:
        return self.projection.w05_batch

    @property
    def targets(self) -> tuple[VerticalOverlayTargetProjection, ...]:
        return self.projection.targets

    def target(
            self,
            spec: VerticalOverlayTargetSpec,
            ) -> VerticalOverlayTargetProjection:
        return _target_projection(self.projection, spec)


def build_w03_w04_w05_three_role_vertical_overlay(
        base: AuthoredSemanticPrimitiveBridgeBuild,
        donor: AuthoredPrimitiveAtomicBridgeBuild,
        ) -> W03W04W05ThreeRoleVerticalOverlay:
    """构建并冻结公开三 Role 纵向 overlay。"""
    projection = build_vertical_overlay_projection(
        base,
        donor,
        THREE_ROLE_VERTICAL_TARGETS,
    )
    actual = (
        projection.base_sample_sha256,
        projection.base_manifest_sha256,
        projection.donor_map_sha256,
        projection.donor_atomic_sha256,
        projection.donor_manifest_sha256,
    )
    expected = (
        THREE_ROLE_BASE_SAMPLE_SHA256,
        THREE_ROLE_BASE_MANIFEST_SHA256,
        THREE_ROLE_DONOR_MAP_SHA256,
        THREE_ROLE_DONOR_ATOMIC_SHA256,
        THREE_ROLE_DONOR_MANIFEST_SHA256,
    )
    if actual != expected:
        raise W03W04W05ThreeRoleVerticalError(
            "three-Role public course identity drifted")
    validation = vertical_overlay_validation_sha256(projection)
    if validation != VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256:
        raise W03W04W05ThreeRoleVerticalError(
            "three-Role overlay validation commitment drifted")
    return W03W04W05ThreeRoleVerticalOverlay(projection, validation)


__all__ = [
    "THREE_ROLE_BASE_MANIFEST_SHA256",
    "THREE_ROLE_BASE_SAMPLE_SHA256",
    "THREE_ROLE_DONOR_ATOMIC_SHA256",
    "THREE_ROLE_DONOR_MANIFEST_SHA256",
    "THREE_ROLE_DONOR_MAP_SHA256",
    "THREE_ROLE_VERTICAL_TARGETS",
    "VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256",
    "W03W04W05ThreeRoleVerticalError",
    "W03W04W05ThreeRoleVerticalOverlay",
    "build_w03_w04_w05_three_role_vertical_overlay",
]
