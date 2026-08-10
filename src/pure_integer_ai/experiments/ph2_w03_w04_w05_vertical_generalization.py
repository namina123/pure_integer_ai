"""冻结两个独立来源命题共享的 W03→W04→W05 纵向 overlay。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    AuthoredPrimitiveAtomicBridgeBuild,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    AuthoredSemanticPrimitiveBridgeBuild,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core import (
    build_vertical_overlay_projection,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core_contract import (
    VerticalOverlayProjection,
    VerticalOverlayTargetProjection,
    VerticalOverlayTargetSpec,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_registry import (
    VERTICAL_GENERALIZATION_OVERLAY_VALIDATION_SHA256,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


VERTICAL_GENERALIZATION_BASE_SAMPLE_SHA256 = (
    "dcf4b9a33d8ca0a526c568dd0105301271316fe352cb9abccbd81f21cc8f6e96")
VERTICAL_GENERALIZATION_BASE_MANIFEST_SHA256 = (
    "8b9e17d806ee1f196b928a56ea8bc5edb4789d417f6f157aaae70ef7af87a69a")
VERTICAL_GENERALIZATION_DONOR_MAP_SHA256 = (
    "2156bf4a45bf1b26e6a06973363dca4ba4d411e1a89615c4f7c3c4f37ac9a54b")
VERTICAL_GENERALIZATION_DONOR_ATOMIC_SHA256 = (
    "6cb072f139206044d9fc09922839979065e20e912830f0e5cf0a95c7886c6d9b")
VERTICAL_GENERALIZATION_DONOR_MANIFEST_SHA256 = (
    "f5fe0db047099b298fe90b74bf14f394f8bbf0709ffc211fc648c80cabd3cc08")
VERTICAL_GENERALIZATION_TARGETS = (
    VerticalOverlayTargetSpec(
        "使得",
        "暴雨使得河水上涨。",
        "暴雨使得河水上涨。",
    ),
    VerticalOverlayTargetSpec(
        "使得",
        "寒潮使得路面结冰。",
        "寒潮使得路面结冰。",
    ),
)


# object-model: exception
class W03W04W05VerticalGeneralizationError(ValueError):
    """多来源纵向 overlay 的公开资料或冻结承诺发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _target_projection(
        projection: VerticalOverlayProjection,
        spec: VerticalOverlayTargetSpec,
        ) -> VerticalOverlayTargetProjection:
    values = tuple(item for item in projection.targets if item.spec == spec)
    if len(values) != 1:
        raise W03W04W05VerticalGeneralizationError(
            "generalization target projection is not unique")
    return values[0]


def _validation_sha(projection: VerticalOverlayProjection) -> str:
    return _sha({
        "base_manifest_sha256": projection.base_manifest_sha256,
        "base_sample_sha256": projection.base_sample_sha256,
        "dependency_w04": [
            item.to_dict()
            for item in projection.dependency_w04_observations
        ],
        "donor_atomic_sha256": projection.donor_atomic_sha256,
        "donor_manifest_sha256": projection.donor_manifest_sha256,
        "donor_map_sha256": projection.donor_map_sha256,
        "overlay_targets": [
            {
                "base_source": item.base_source.to_dict(),
                "base_w03": item.base_w03_observation.to_dict(),
                "base_w04": item.base_w04_observation.to_dict(),
                "overlay_w05": item.overlay_w05_observation.to_dict(),
                "overlay_w05_evidence": item.overlay_w05_evidence.to_dict(),
                "spec": item.spec.to_dict(),
            }
            for item in projection.targets
        ],
        "policy": (
            "MULTI_SOURCE_EXACT_EXTERNAL_PREREQUISITES_"
            "NO_SURFACE_FALLBACK"
        ),
        "w03_source_binding_sha256": (
            projection.w03_batch.source_binding.sha256()),
        "w04_source_binding_sha256": (
            projection.w04_batch.source_binding.sha256()),
        "w05_source_binding_sha256": (
            projection.w05_batch.source_binding.sha256()),
    })


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalGeneralizationOverlay:
    """两个来源命题共用的公开批次、精确链接和冻结验证摘要。"""

    projection: VerticalOverlayProjection
    validation_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.projection, VerticalOverlayProjection)
                or not isinstance(self.validation_sha256, str)
                or len(self.validation_sha256) != 64):
            raise W03W04W05VerticalGeneralizationError(
                "generalization overlay projection drifted")

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


def build_w03_w04_w05_vertical_generalization_overlay(
        base: AuthoredSemanticPrimitiveBridgeBuild,
        donor: AuthoredPrimitiveAtomicBridgeBuild,
        ) -> W03W04W05VerticalGeneralizationOverlay:
    """构建双来源 overlay，并拒绝任一资料或承诺的静默漂移。"""
    projection = build_vertical_overlay_projection(
        base,
        donor,
        VERTICAL_GENERALIZATION_TARGETS,
    )
    actual = (
        projection.base_sample_sha256,
        projection.base_manifest_sha256,
        projection.donor_map_sha256,
        projection.donor_atomic_sha256,
        projection.donor_manifest_sha256,
    )
    expected = (
        VERTICAL_GENERALIZATION_BASE_SAMPLE_SHA256,
        VERTICAL_GENERALIZATION_BASE_MANIFEST_SHA256,
        VERTICAL_GENERALIZATION_DONOR_MAP_SHA256,
        VERTICAL_GENERALIZATION_DONOR_ATOMIC_SHA256,
        VERTICAL_GENERALIZATION_DONOR_MANIFEST_SHA256,
    )
    if actual != expected:
        raise W03W04W05VerticalGeneralizationError(
            "generalization course identity drifted")
    validation_sha = _validation_sha(projection)
    if (validation_sha
            != VERTICAL_GENERALIZATION_OVERLAY_VALIDATION_SHA256):
        raise W03W04W05VerticalGeneralizationError(
            "generalization overlay validation commitment drifted")
    return W03W04W05VerticalGeneralizationOverlay(
        projection,
        validation_sha,
    )


__all__ = [
    "VERTICAL_GENERALIZATION_BASE_MANIFEST_SHA256",
    "VERTICAL_GENERALIZATION_BASE_SAMPLE_SHA256",
    "VERTICAL_GENERALIZATION_DONOR_ATOMIC_SHA256",
    "VERTICAL_GENERALIZATION_DONOR_MANIFEST_SHA256",
    "VERTICAL_GENERALIZATION_DONOR_MAP_SHA256",
    "VERTICAL_GENERALIZATION_OVERLAY_VALIDATION_SHA256",
    "VERTICAL_GENERALIZATION_TARGETS",
    "W03W04W05VerticalGeneralizationError",
    "W03W04W05VerticalGeneralizationOverlay",
    "build_w03_w04_w05_vertical_generalization_overlay",
]
