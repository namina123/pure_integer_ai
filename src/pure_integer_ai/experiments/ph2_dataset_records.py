"""PH2 学生可见 SourceRef 与 Observation 记录合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pure_integer_ai.experiments.ph2_dataset_core import (
    EPISTEMIC_ROLES,
    LOCAL_ONLY_LICENSE_IDS,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    REDISTRIBUTION_POLICIES,
    SAMPLE_ROLES,
    SPLITS,
    W_STAGES,
    CanonicalJsonObject,
    DatasetContractError,
    StableRecordKey,
    _enum_text,
    _license_id,
    _nonempty_text,
    _nonnegative_int,
    _positive_int,
    _record_key_tuple,
    _sha256,
    _upstream_checksum,
)


@dataclass(frozen=True)
class SourceRefRecord:
    """冻结一个来源实体/文件记录及许可、解析和真实来源簇身份。"""

    RECORD_KIND: ClassVar[str] = RECORD_SOURCE_REF

    format_version: int
    schema_version: int
    course_version: int
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    stable_key: StableRecordKey
    source_key: str
    snapshot_id: str
    revision_id: str
    official_url: str
    source_identity: str
    upstream_checksum: str
    local_sha256: str
    license_id: str
    redistribution_policy: str
    attribution: str
    parser_version: int
    source_span: CanonicalJsonObject
    record_ordinal: int
    source_cluster_key: StableRecordKey

    def __post_init__(self) -> None:
        _positive_int(self.format_version, where="SourceRefRecord.format_version")
        _positive_int(self.schema_version, where="SourceRefRecord.schema_version")
        _positive_int(self.course_version, where="SourceRefRecord.course_version")
        if any(not isinstance(value, StableRecordKey) for value in (
                self.dataset_key, self.artifact_key, self.stable_key)):
            raise DatasetContractError("SourceRefRecord 身份键类型错误")
        _nonempty_text(self.source_key, where="SourceRefRecord.source_key")
        if not self.snapshot_id and not self.revision_id:
            raise DatasetContractError("SourceRefRecord snapshot/revision 至少一个非空")
        if self.snapshot_id:
            _nonempty_text(self.snapshot_id, where="SourceRefRecord.snapshot_id")
        if self.revision_id:
            _nonempty_text(self.revision_id, where="SourceRefRecord.revision_id")
        url = _nonempty_text(self.official_url, where="SourceRefRecord.official_url")
        if not url.startswith(("https://", "local:", "urn:")):
            raise DatasetContractError("SourceRefRecord.official_url scheme 非法")
        _nonempty_text(self.source_identity, where="SourceRefRecord.source_identity")
        object.__setattr__(self, "upstream_checksum", _upstream_checksum(
            self.upstream_checksum, where="SourceRefRecord.upstream_checksum"))
        object.__setattr__(self, "local_sha256", _sha256(
            self.local_sha256, where="SourceRefRecord.local_sha256"))
        object.__setattr__(self, "license_id", _license_id(
            self.license_id, where="SourceRefRecord.license_id"))
        _enum_text(
            self.redistribution_policy,
            REDISTRIBUTION_POLICIES,
            where="SourceRefRecord.redistribution_policy",
        )
        if (self.license_id in LOCAL_ONLY_LICENSE_IDS
                and self.redistribution_policy != "LOCAL_ONLY"):
            raise DatasetContractError("许可不清来源只能标 LOCAL_ONLY")
        _nonempty_text(self.attribution, where="SourceRefRecord.attribution")
        _positive_int(self.parser_version, where="SourceRefRecord.parser_version")
        if not isinstance(self.source_span, CanonicalJsonObject):
            raise DatasetContractError("SourceRefRecord.source_span 类型错误")
        _nonnegative_int(
            self.record_ordinal, where="SourceRefRecord.record_ordinal")
        if not isinstance(self.source_cluster_key, StableRecordKey):
            raise DatasetContractError("SourceRefRecord.source_cluster_key 类型错误")

    def to_dict(self) -> dict[str, Any]:
        """导出 SourceRefRecord 的规范 JSON object。"""
        return {
            "attribution": self.attribution,
            "artifact_key": self.artifact_key.to_list(),
            "course_version": self.course_version,
            "dataset_key": self.dataset_key.to_list(),
            "format_version": self.format_version,
            "license_id": self.license_id,
            "local_sha256": self.local_sha256,
            "official_url": self.official_url,
            "parser_version": self.parser_version,
            "record_kind": self.RECORD_KIND,
            "record_ordinal": self.record_ordinal,
            "redistribution_policy": self.redistribution_policy,
            "revision_id": self.revision_id,
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "source_cluster_key": self.source_cluster_key.to_list(),
            "source_identity": self.source_identity,
            "source_key": self.source_key,
            "source_span": self.source_span.to_value(),
            "stable_key": self.stable_key.to_list(),
            "upstream_checksum": self.upstream_checksum,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRefRecord":
        """从已解析 JSON object 恢复 SourceRefRecord。"""
        return cls(
            value["format_version"],
            value["schema_version"],
            value["course_version"],
            StableRecordKey.from_value(value["dataset_key"], where="source.dataset_key"),
            StableRecordKey.from_value(value["artifact_key"], where="source.artifact_key"),
            StableRecordKey.from_value(value["stable_key"], where="source.stable_key"),
            str(value["source_key"]),
            str(value["snapshot_id"]),
            str(value["revision_id"]),
            str(value["official_url"]),
            str(value["source_identity"]),
            str(value["upstream_checksum"]),
            str(value["local_sha256"]),
            str(value["license_id"]),
            str(value["redistribution_policy"]),
            str(value["attribution"]),
            value["parser_version"],
            CanonicalJsonObject.from_value(value["source_span"]),
            value["record_ordinal"],
            StableRecordKey.from_value(
                value["source_cluster_key"], where="source.source_cluster_key"),
        )


@dataclass(frozen=True)
class ObservationRecord:
    """学生可见的来源化课程 observation，不含教师或 evaluator 私有答案。"""

    RECORD_KIND: ClassVar[str] = RECORD_OBSERVATION

    format_version: int
    schema_version: int
    course_version: int
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    stable_key: StableRecordKey
    w_stage: str
    substage: str
    split: str
    language: str
    representation: str
    source_ref_key: StableRecordKey
    license_partition: str
    dedup_cluster_key: StableRecordKey
    content_group_key: StableRecordKey
    template_group_key: StableRecordKey
    shape_group_key: StableRecordKey
    epistemic_role: str
    sample_role: str
    payload_kind: str
    typed_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_key: StableRecordKey | None
    prerequisite_keys: tuple[StableRecordKey, ...]
    logical_order: int

    def __post_init__(self) -> None:
        _positive_int(self.format_version, where="ObservationRecord.format_version")
        _positive_int(self.schema_version, where="ObservationRecord.schema_version")
        _positive_int(self.course_version, where="ObservationRecord.course_version")
        key_fields = (
            self.dataset_key, self.artifact_key, self.stable_key,
            self.source_ref_key, self.dedup_cluster_key,
            self.content_group_key, self.template_group_key,
            self.shape_group_key,
        )
        if any(not isinstance(value, StableRecordKey) for value in key_fields):
            raise DatasetContractError("ObservationRecord 整数键字段类型错误")
        _enum_text(self.w_stage, W_STAGES, where="ObservationRecord.w_stage")
        _nonempty_text(self.substage, where="ObservationRecord.substage")
        _enum_text(self.split, SPLITS, where="ObservationRecord.split")
        _nonempty_text(self.language, where="ObservationRecord.language")
        _nonempty_text(self.representation, where="ObservationRecord.representation")
        object.__setattr__(self, "license_partition", _license_id(
            self.license_partition, where="ObservationRecord.license_partition"))
        _enum_text(
            self.epistemic_role,
            EPISTEMIC_ROLES,
            where="ObservationRecord.epistemic_role",
        )
        _enum_text(self.sample_role, SAMPLE_ROLES, where="ObservationRecord.sample_role")
        _nonempty_text(self.payload_kind, where="ObservationRecord.payload_kind")
        if not isinstance(self.typed_payload, CanonicalJsonObject):
            raise DatasetContractError("ObservationRecord.typed_payload 类型错误")
        _nonempty_text(
            self.perturbation_kind,
            where="ObservationRecord.perturbation_kind",
        )
        if self.supersedes_key is not None and not isinstance(
                self.supersedes_key, StableRecordKey):
            raise DatasetContractError("ObservationRecord.supersedes_key 类型错误")
        _record_key_tuple(
            self.prerequisite_keys,
            where="ObservationRecord.prerequisite_keys",
            allow_empty=True,
        )
        if self.stable_key in self.prerequisite_keys:
            raise DatasetContractError("ObservationRecord 不得前置引用自身")
        if self.supersedes_key == self.stable_key:
            raise DatasetContractError("ObservationRecord 不得 supersede 自身")
        _nonnegative_int(self.logical_order, where="ObservationRecord.logical_order")

    def to_dict(self) -> dict[str, Any]:
        """导出 ObservationRecord 的规范 JSON object。"""
        return {
            "artifact_key": self.artifact_key.to_list(),
            "content_group_key": self.content_group_key.to_list(),
            "course_version": self.course_version,
            "dataset_key": self.dataset_key.to_list(),
            "dedup_cluster_key": self.dedup_cluster_key.to_list(),
            "epistemic_role": self.epistemic_role,
            "format_version": self.format_version,
            "language": self.language,
            "license_partition": self.license_partition,
            "logical_order": self.logical_order,
            "payload_kind": self.payload_kind,
            "perturbation_kind": self.perturbation_kind,
            "prerequisite_keys": [key.to_list() for key in self.prerequisite_keys],
            "record_kind": self.RECORD_KIND,
            "representation": self.representation,
            "sample_role": self.sample_role,
            "schema_version": self.schema_version,
            "shape_group_key": self.shape_group_key.to_list(),
            "source_ref_key": self.source_ref_key.to_list(),
            "split": self.split,
            "stable_key": self.stable_key.to_list(),
            "substage": self.substage,
            "supersedes_key": (
                self.supersedes_key.to_list()
                if self.supersedes_key is not None else None
            ),
            "template_group_key": self.template_group_key.to_list(),
            "typed_payload": self.typed_payload.to_value(),
            "w_stage": self.w_stage,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ObservationRecord":
        """从已解析 JSON object 恢复 ObservationRecord。"""
        raw_supersedes = value["supersedes_key"]
        return cls(
            value["format_version"],
            value["schema_version"],
            value["course_version"],
            StableRecordKey.from_value(
                value["dataset_key"], where="observation.dataset_key"),
            StableRecordKey.from_value(
                value["artifact_key"], where="observation.artifact_key"),
            StableRecordKey.from_value(value["stable_key"], where="observation.stable_key"),
            str(value["w_stage"]),
            str(value["substage"]),
            str(value["split"]),
            str(value["language"]),
            str(value["representation"]),
            StableRecordKey.from_value(
                value["source_ref_key"], where="observation.source_ref_key"),
            str(value["license_partition"]),
            StableRecordKey.from_value(
                value["dedup_cluster_key"], where="observation.dedup_cluster_key"),
            StableRecordKey.from_value(
                value["content_group_key"], where="observation.content_group_key"),
            StableRecordKey.from_value(
                value["template_group_key"], where="observation.template_group_key"),
            StableRecordKey.from_value(
                value["shape_group_key"], where="observation.shape_group_key"),
            str(value["epistemic_role"]),
            str(value["sample_role"]),
            str(value["payload_kind"]),
            CanonicalJsonObject.from_value(value["typed_payload"]),
            str(value["perturbation_kind"]),
            (StableRecordKey.from_value(raw_supersedes, where="observation.supersedes_key")
             if raw_supersedes is not None else None),
            tuple(StableRecordKey.from_value(item, where="observation.prerequisite_key")
                  for item in value["prerequisite_keys"]),
            value["logical_order"],
        )


__all__ = ["ObservationRecord", "SourceRefRecord"]
