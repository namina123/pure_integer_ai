"""多来源纵向 overlay core 的不可变目标、记录投影与错误合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


# object-model: exception
class W03W04W05VerticalOverlayCoreError(ValueError):
    """基础资料、donor 资料或精确前置关系无法闭合。"""


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05VerticalOverlayCoreError(
            f"{where} is not canonical text")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class VerticalOverlayTargetSpec:
    """一个必须由同源 W03、W04 与 donor W05 精确闭合的目标。"""

    surface: str
    context: str
    proposition_surface: str

    def __post_init__(self) -> None:
        _text(self.surface, where="vertical target surface")
        _text(self.context, where="vertical target context")
        _text(
            self.proposition_surface,
            where="vertical target proposition surface",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context,
            "proposition_surface": self.proposition_surface,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class VerticalOverlayTargetProjection:
    """一个目标在替换 donor SourceRef 后的全部精确记录。"""

    spec: VerticalOverlayTargetSpec
    base_source: SourceRefRecord
    base_w03_observation: ObservationRecord
    base_w04_observation: ObservationRecord
    donor_w05_observation: ObservationRecord
    overlay_w05_observation: ObservationRecord
    overlay_w05_evidence: TeacherEvidenceRecord

    def __post_init__(self) -> None:
        if (not isinstance(self.spec, VerticalOverlayTargetSpec)
                or not isinstance(self.base_source, SourceRefRecord)
                or not isinstance(
                    self.base_w03_observation, ObservationRecord)
                or not isinstance(
                    self.base_w04_observation, ObservationRecord)
                or not isinstance(
                    self.donor_w05_observation, ObservationRecord)
                or not isinstance(
                    self.overlay_w05_observation, ObservationRecord)
                or not isinstance(
                    self.overlay_w05_evidence, TeacherEvidenceRecord)):
            raise W03W04W05VerticalOverlayCoreError(
                "vertical target projection type drifted")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class VerticalOverlayProjection:
    """可由冻结 facade 进一步承诺的多目标公开 overlay 投影。"""

    w03_batch: W03V2PublicEvaluationBatch
    w04_batch: W04V2PublicEvaluationBatch
    w05_batch: W05V2PublicEvaluationBatch
    targets: tuple[VerticalOverlayTargetProjection, ...]
    dependency_w04_observations: tuple[ObservationRecord, ...]
    base_sample_sha256: str
    base_manifest_sha256: str
    donor_map_sha256: str
    donor_atomic_sha256: str
    donor_manifest_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.w03_batch, W03V2PublicEvaluationBatch)
                or not isinstance(self.w04_batch, W04V2PublicEvaluationBatch)
                or not isinstance(self.w05_batch, W05V2PublicEvaluationBatch)
                or not self.targets
                or any(not isinstance(item, VerticalOverlayTargetProjection)
                       for item in self.targets)
                or not self.dependency_w04_observations
                or any(not isinstance(item, ObservationRecord)
                       for item in self.dependency_w04_observations)):
            raise W03W04W05VerticalOverlayCoreError(
                "vertical overlay projection inventory drifted")
        for value in (
                self.base_sample_sha256,
                self.base_manifest_sha256,
                self.donor_map_sha256,
                self.donor_atomic_sha256,
                self.donor_manifest_sha256):
            if not isinstance(value, str) or len(value) != 64:
                raise W03W04W05VerticalOverlayCoreError(
                    "vertical overlay projection SHA drifted")


__all__ = [
    "VerticalOverlayProjection",
    "VerticalOverlayTargetProjection",
    "VerticalOverlayTargetSpec",
    "W03W04W05VerticalOverlayCoreError",
]
