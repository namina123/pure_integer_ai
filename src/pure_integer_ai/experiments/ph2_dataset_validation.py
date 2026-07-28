"""PH2 统一资料的许可、split、引用、阶段可见性和 supersede 校验。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    JSONL_RECORD_KINDS,
    LOCAL_ONLY_LICENSE_IDS,
    PUBLIC_LICENSE_IDS,
    SPLITS,
    W_STAGES,
    ArtifactManifest,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)


PRIVATE_OBSERVATION_KEYS = frozenset({
    "expected",
    "expected_output",
    "expected_state",
    "teacher_output",
    "teacher_answer",
    "evaluator_label",
    "held_out_label",
    "fixture_label",
})


class DatasetValidationError(RuntimeError):
    """资料 bundle 存在许可、泄漏、引用或图一致性错误。"""


@dataclass(frozen=True)
class DatasetBundleValidationReport:
    """记录一次纯校验得到的分账计数和可审计集合。"""

    format_version: int
    schema_version: int
    course_version: int
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    source_ref_count: int
    observation_count: int
    teacher_evidence_count: int
    evaluator_label_count: int
    source_cluster_count: int
    splits: tuple[str, ...]
    stages: tuple[str, ...]


def _stage_rank(stage: str) -> int:
    """返回冻结 W 阶段的严格顺序索引。"""
    try:
        return W_STAGES.index(stage)
    except ValueError as error:
        raise DatasetValidationError(f"未知 W 阶段: {stage}") from error


def _unique_records(
        records: Iterable[
            SourceRefRecord | ObservationRecord
            | TeacherEvidenceRecord | EvaluatorLabelRecord]) -> None:
    """要求 bundle 内所有记录 stable key 全局唯一。"""
    owners: dict[StableRecordKey, str] = {}
    for record in records:
        prior = owners.get(record.stable_key)
        if prior is not None:
            raise DatasetValidationError(
                f"重复 stable key: {record.stable_key.components} ({prior})")
        owners[record.stable_key] = type(record).__name__


def _identity_tuple(record: SourceRefRecord | ObservationRecord
                    | TeacherEvidenceRecord | EvaluatorLabelRecord
                    ) -> tuple[int, int, int, StableRecordKey, StableRecordKey]:
    """提取每条正式记录必须直接携带的五项数据身份。"""
    return (
        record.format_version,
        record.schema_version,
        record.course_version,
        record.dataset_key,
        record.artifact_key,
    )


def validate_identity_bindings(
        records: tuple[
            SourceRefRecord | ObservationRecord
            | TeacherEvidenceRecord | EvaluatorLabelRecord, ...]
        ) -> tuple[int, int, int, StableRecordKey, StableRecordKey]:
    """要求 bundle 内所有记录显式绑定同一版本、dataset 和 artifact。"""
    if not records:
        raise DatasetValidationError("资料 bundle 不能为空")
    identities = {_identity_tuple(record) for record in records}
    if len(identities) != 1:
        raise DatasetValidationError("资料记录 dataset/artifact/version 绑定漂移")
    return next(iter(identities))


def _validate_private_payload_keys(value: Any, *, where: str) -> None:
    """递归拒绝 Observation 中的 expected/teacher/evaluator 私有字段。"""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_").replace(" ", "_")
            if normalized in PRIVATE_OBSERVATION_KEYS:
                raise DatasetValidationError(f"{where} 含私有字段 {key!r}")
            _validate_private_payload_keys(item, where=f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_private_payload_keys(item, where=f"{where}[{index}]")


def validate_license_partition(
        source_refs: tuple[SourceRefRecord, ...],
        observations: tuple[ObservationRecord, ...],
        *,
        source_key: str,
        license_partition: str,
        public_release: bool) -> None:
    """要求单 pack 只有一个来源键和许可分区，公开包不得含 NOASSERTION。"""
    if not source_refs:
        raise DatasetValidationError("source_refs 不能为空")
    if any(item.source_key != source_key for item in source_refs):
        raise DatasetValidationError("pack 混入其他 source_key")
    if any(item.license_id != license_partition for item in source_refs):
        raise DatasetValidationError("pack 混入其他许可来源")
    if any(item.license_partition != license_partition for item in observations):
        raise DatasetValidationError("Observation 许可分区与 pack 不一致")
    if public_release:
        if license_partition not in PUBLIC_LICENSE_IDS:
            raise DatasetValidationError("许可不清来源不得进入公开 release")
        if any(item.redistribution_policy != "PUBLIC" for item in source_refs):
            raise DatasetValidationError("公开 release 含 LOCAL_ONLY SourceRef")
    elif license_partition in LOCAL_ONLY_LICENSE_IDS:
        if any(item.redistribution_policy != "LOCAL_ONLY" for item in source_refs):
            raise DatasetValidationError("NOASSERTION SourceRef 必须 LOCAL_ONLY")


def validate_reference_integrity(
        source_refs: tuple[SourceRefRecord, ...],
        observations: tuple[ObservationRecord, ...],
        teacher_evidence: tuple[TeacherEvidenceRecord, ...],
        evaluator_labels: tuple[EvaluatorLabelRecord, ...]) -> None:
    """核对 SourceRef、Observation 和 owner 引用均指向本 bundle 唯一记录。"""
    sources = {item.stable_key: item for item in source_refs}
    observation_index = {item.stable_key: item for item in observations}
    if len(sources) != len(source_refs) or len(observation_index) != len(observations):
        raise DatasetValidationError("SourceRef 或 Observation stable key 重复")
    for item in observations:
        source = sources.get(item.source_ref_key)
        if source is None:
            raise DatasetValidationError("Observation 引用缺失 SourceRef")
        if source.license_id != item.license_partition:
            raise DatasetValidationError("Observation 与 SourceRef 许可不一致")
        _validate_private_payload_keys(
            item.typed_payload.to_value(),
            where=f"Observation[{item.stable_key.components}]",
        )
    for item in teacher_evidence:
        if item.observation_key not in observation_index:
            raise DatasetValidationError("TeacherEvidence 引用缺失 Observation")
        if item.source_ref_key not in sources:
            raise DatasetValidationError("TeacherEvidence 引用缺失 SourceRef")
    for item in evaluator_labels:
        if item.observation_key not in observation_index:
            raise DatasetValidationError("EvaluatorLabel 引用缺失 Observation")
    teacher_owners = {item.owner_key for item in teacher_evidence}
    evaluator_owners = {item.owner_key for item in evaluator_labels}
    if teacher_owners & evaluator_owners:
        raise DatasetValidationError("teacher 与 evaluator 不得共用 owner")


def validate_split_leakage(
        source_refs: tuple[SourceRefRecord, ...],
        observations: tuple[ObservationRecord, ...]) -> None:
    """要求来源、去重、内容、模板和形状 cluster 均只属于一个 split。"""
    sources = {item.stable_key: item for item in source_refs}
    assignments: dict[tuple[str, StableRecordKey], str] = {}
    for item in observations:
        source = sources.get(item.source_ref_key)
        if source is None:
            raise DatasetValidationError("split 校验遇到缺失 SourceRef")
        clusters = (
            ("source", source.source_cluster_key),
            ("dedup", item.dedup_cluster_key),
            ("content", item.content_group_key),
            ("template", item.template_group_key),
            ("shape", item.shape_group_key),
        )
        for kind, key in clusters:
            identity = (kind, key)
            prior = assignments.get(identity)
            if prior is not None and prior != item.split:
                raise DatasetValidationError(
                    f"{kind} cluster 跨 split: {prior} -> {item.split}")
            assignments[identity] = item.split


def validate_supersede_and_prerequisite_graph(
        observations: tuple[ObservationRecord, ...]) -> None:
    """要求前置/替代引用存在、只指向过去，并拒绝 supersede 环。"""
    index = {item.stable_key: item for item in observations}
    if len(index) != len(observations):
        raise DatasetValidationError("Observation stable key 重复")
    edges: dict[StableRecordKey, StableRecordKey] = {}
    for item in observations:
        if item.supersedes_key is not None:
            edges[item.stable_key] = item.supersedes_key

    visiting: set[StableRecordKey] = set()
    visited: set[StableRecordKey] = set()

    def visit(key: StableRecordKey) -> None:
        """深度优先遍历单出边 supersede 图并检测回边。"""
        if key in visiting:
            raise DatasetValidationError("supersede 图存在环")
        if key in visited:
            return
        visiting.add(key)
        target = edges.get(key)
        if target is not None:
            visit(target)
        visiting.remove(key)
        visited.add(key)

    for key in sorted(index):
        visit(key)

    for item in observations:
        references = item.prerequisite_keys + (
            (item.supersedes_key,) if item.supersedes_key is not None else ())
        for key in references:
            target = index.get(key)
            if target is None:
                raise DatasetValidationError("Observation 引用 bundle 外记录")
            if _stage_rank(target.w_stage) > _stage_rank(item.w_stage):
                raise DatasetValidationError("Observation 引用了未来阶段记录")
            if target.logical_order >= item.logical_order:
                raise DatasetValidationError("Observation 引用必须指向更早 logical order")


def validate_artifact_manifest(
        manifest: ArtifactManifest,
        source_refs: tuple[SourceRefRecord, ...],
        observations: tuple[ObservationRecord, ...]) -> None:
    """核对 manifest 的四类文件、来源簇、split、阶段、来源和许可集合。"""
    file_kinds = {item.record_kind for item in manifest.files}
    if file_kinds != set(JSONL_RECORD_KINDS):
        raise DatasetValidationError("ArtifactManifest 必须覆盖四类 JSONL record")
    if any(item.source_key != manifest.source_key for item in source_refs):
        raise DatasetValidationError("ArtifactManifest.source_key 与 SourceRef 不一致")
    if any(item.license_id != manifest.license_partition for item in source_refs):
        raise DatasetValidationError("ArtifactManifest.license_partition 与 SourceRef 不一致")
    if any(item.license_partition != manifest.license_partition
           for item in observations):
        raise DatasetValidationError("ArtifactManifest.license_partition 与 Observation 不一致")
    for record in source_refs + observations:
        if (record.format_version != manifest.format_version
                or record.schema_version != manifest.schema_version
                or record.course_version != manifest.course_version
                or record.dataset_key != manifest.dataset_key
                or record.artifact_key != manifest.stable_key):
            raise DatasetValidationError("ArtifactManifest 与记录身份绑定不一致")
    source_clusters = tuple(sorted({
        item.source_cluster_key for item in source_refs
    }))
    if tuple(sorted(manifest.source_cluster_keys)) != source_clusters:
        raise DatasetValidationError("ArtifactManifest.source_cluster_keys 漂移")
    file_clusters = tuple(sorted({
        key for item in manifest.files for key in item.source_cluster_keys
    }))
    if file_clusters != source_clusters:
        raise DatasetValidationError("ArtifactManifest 文件来源簇集合不完整")
    splits = tuple(sorted({item.split for item in observations}, key=SPLITS.index))
    if tuple(manifest.splits) != splits:
        raise DatasetValidationError("ArtifactManifest.splits 与 Observation 不一致")
    stages = tuple(sorted({item.w_stage for item in observations}, key=W_STAGES.index))
    if tuple(manifest.w_stages) != stages:
        raise DatasetValidationError("ArtifactManifest.w_stages 与 Observation 不一致")


def validate_stage_visibility(
        observations: tuple[ObservationRecord, ...],
        teacher_evidence: tuple[TeacherEvidenceRecord, ...],
        evaluator_labels: tuple[EvaluatorLabelRecord, ...],
        *,
        current_stage: str,
        view_kind: str) -> None:
    """校验训练/评测视图的阶段白名单和 owner 物理隔离语义。"""
    current_rank = _stage_rank(current_stage)
    if view_kind not in {"training", "evaluation"}:
        raise DatasetValidationError("view_kind 必须是 training/evaluation")
    for item in observations:
        if _stage_rank(item.w_stage) > current_rank:
            raise DatasetValidationError("当前视图包含未来阶段 Observation")
        if view_kind == "training" and item.split != "train":
            raise DatasetValidationError("training 视图只能含 train Observation")
        if view_kind == "evaluation" and item.split == "train":
            raise DatasetValidationError("evaluation 视图不得含 train Observation")
    for item in teacher_evidence:
        if _stage_rank(item.visible_from_stage) > current_rank:
            raise DatasetValidationError("当前视图包含未来阶段 TeacherEvidence")
    if view_kind == "training" and evaluator_labels:
        raise DatasetValidationError("training 视图不得读取 evaluator label")
    if view_kind == "evaluation" and teacher_evidence:
        raise DatasetValidationError("evaluation 视图不得读取 teacher Evidence")


def validate_dataset_bundle(
        source_refs: Iterable[SourceRefRecord],
        observations: Iterable[ObservationRecord],
        teacher_evidence: Iterable[TeacherEvidenceRecord],
        evaluator_labels: Iterable[EvaluatorLabelRecord],
        *,
        source_key: str,
        license_partition: str,
        public_release: bool) -> DatasetBundleValidationReport:
    """合取 D-02A bundle 的唯一键、许可、引用、split 和替代图证据。"""
    source_tuple = tuple(source_refs)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teacher_evidence)
    evaluator_tuple = tuple(evaluator_labels)
    all_records = source_tuple + observation_tuple + teacher_tuple + evaluator_tuple
    _unique_records(all_records)
    identity = validate_identity_bindings(all_records)
    validate_license_partition(
        source_tuple,
        observation_tuple,
        source_key=source_key,
        license_partition=license_partition,
        public_release=public_release,
    )
    validate_reference_integrity(
        source_tuple, observation_tuple, teacher_tuple, evaluator_tuple)
    validate_split_leakage(source_tuple, observation_tuple)
    validate_supersede_and_prerequisite_graph(observation_tuple)
    return DatasetBundleValidationReport(
        identity[0],
        identity[1],
        identity[2],
        identity[3],
        identity[4],
        len(source_tuple),
        len(observation_tuple),
        len(teacher_tuple),
        len(evaluator_tuple),
        len({item.source_cluster_key for item in source_tuple}),
        tuple(sorted({item.split for item in observation_tuple}, key=SPLITS.index)),
        tuple(sorted({item.w_stage for item in observation_tuple}, key=W_STAGES.index)),
    )


__all__ = [
    "DatasetBundleValidationReport",
    "DatasetValidationError",
    "PRIVATE_OBSERVATION_KEYS",
    "validate_dataset_bundle",
    "validate_identity_bindings",
    "validate_artifact_manifest",
    "validate_license_partition",
    "validate_reference_integrity",
    "validate_split_leakage",
    "validate_stage_visibility",
    "validate_supersede_and_prerequisite_graph",
]
