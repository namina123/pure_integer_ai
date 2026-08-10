"""冻结 FT08 单来源纵向链，并把通用组装委托给多目标 overlay core。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    AuthoredPrimitiveAtomicBridgeBuild,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    AuthoredSemanticPrimitiveBridgeBuild,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core import (
    build_vertical_overlay_projection,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_core_contract import (
    VerticalOverlayTargetSpec,
    W03W04W05VerticalOverlayCoreError,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_registry import (
    VERTICAL_OVERLAY_VALIDATION_SHA256,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
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
_TARGET = VerticalOverlayTargetSpec(
    VERTICAL_SURFACE,
    VERTICAL_CONTEXT,
    VERTICAL_PROPOSITION_SURFACE,
)


# object-model: exception
class W03W04W05VerticalOverlayError(ValueError):
    """冻结基础资料、donor 记录或外部前置关系发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05VerticalOverlay:
    """三个公开批次以及唯一经过冻结验证的外部依赖。"""

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
    """冻结旧单链身份；组装算法由可复用 core 唯一拥有。"""
    try:
        projection = build_vertical_overlay_projection(
            base,
            donor,
            (_TARGET,),
        )
    except W03W04W05VerticalOverlayCoreError as error:
        raise W03W04W05VerticalOverlayError(str(error)) from error
    actual_identity = (
        projection.base_sample_sha256,
        projection.base_manifest_sha256,
        projection.donor_map_sha256,
        projection.donor_atomic_sha256,
        projection.donor_manifest_sha256,
    )
    frozen_identity = (
        W03_W04_BASE_SAMPLE_SHA256,
        W03_W04_BASE_MANIFEST_SHA256,
        W04_W05_DONOR_MAP_SHA256,
        W04_W05_DONOR_ATOMIC_SHA256,
        W04_W05_DONOR_MANIFEST_SHA256,
    )
    if actual_identity != frozen_identity or len(projection.targets) != 1:
        raise W03W04W05VerticalOverlayError(
            "vertical overlay course identity drifted")
    target = projection.targets[0]
    perturbations = frozenset(
        item.observation.perturbation_kind
        for item in projection.w05_batch.pairs
    )
    omission = tuple(
        item.observation
        for item in projection.w05_batch.pairs
        if item.observation.perturbation_kind == "OCCURRENCE_OMISSION"
    )
    if (len(projection.w05_batch.pairs) != 6
            or perturbations != _TRAIN_PERTURBATIONS
            or len(omission) != 1
            or target.overlay_w05_observation.supersedes_key
            != omission[0].stable_key):
        raise W03W04W05VerticalOverlayError(
            "vertical W05 frozen inventory drifted")
    validation_sha = _sha({
        "base_manifest_sha256": projection.base_manifest_sha256,
        "base_sample_sha256": projection.base_sample_sha256,
        "dependency_w04": [
            item.to_dict()
            for item in projection.dependency_w04_observations
        ],
        "donor_atomic_sha256": projection.donor_atomic_sha256,
        "donor_manifest_sha256": projection.donor_manifest_sha256,
        "donor_map_sha256": projection.donor_map_sha256,
        "overlay_w05": target.overlay_w05_observation.to_dict(),
        "overlay_w05_evidence": target.overlay_w05_evidence.to_dict(),
        "policy": "EXACT_EXTERNAL_PREREQUISITES_NO_SURFACE_FALLBACK",
        "w03_source_binding_sha256": (
            projection.w03_batch.source_binding.sha256()),
        "w04_source_binding_sha256": (
            projection.w04_batch.source_binding.sha256()),
        "w05_source_binding_sha256": (
            projection.w05_batch.source_binding.sha256()),
    })
    if validation_sha != VERTICAL_OVERLAY_VALIDATION_SHA256:
        raise W03W04W05VerticalOverlayError(
            "vertical overlay validation commitment drifted")
    return W03W04W05VerticalOverlay(
        projection.w03_batch,
        projection.w04_batch,
        projection.w05_batch,
        target.base_w03_observation,
        target.base_w04_observation,
        target.overlay_w05_observation,
        projection.dependency_w04_observations,
        projection.base_sample_sha256,
        projection.base_manifest_sha256,
        projection.donor_map_sha256,
        projection.donor_atomic_sha256,
        projection.donor_manifest_sha256,
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
