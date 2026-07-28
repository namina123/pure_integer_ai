"""D-02B/D 原创精确课程共用的来源化记录构造与 pack 发布器。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
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


class AuthoredCourseCommonError(RuntimeError):
    """原创课程共用 seed、spec 或 pack 发布边界不完整。"""


def _text(value: Any, *, where: str) -> str:
    """要求共用 spec/seed 文本非空且无首尾空白。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AuthoredCourseCommonError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求共用版本、预算和逻辑序为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredCourseCommonError(f"{where} 必须是正严格整数")
    return value


def _stable_key(namespace: str, *parts: Any) -> StableRecordKey:
    """从完整规范值产生版本化正整数身份键。"""
    payload = canonical_json_bytes({
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


def _sha256_file(path: Path) -> str:
    """流式计算原创 sample 文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuthoredCourseSpec:
    """声明一个原创课程 pack 的来源、版本、阶段、owner 和 evaluator 协议。"""

    source_key: str
    license_id: str
    course_version: int
    artifact_version: int
    adapter_version: int
    generator_version: int
    parser_version: int
    pack_name: str
    stage: str
    substage: str
    snapshot_id: str
    official_url: str
    attribution: str
    evidence_kind: str
    dimension_name: str
    evaluation_budget: int

    def __post_init__(self) -> None:
        for name, value in (
                ("source_key", self.source_key),
                ("license_id", self.license_id),
                ("pack_name", self.pack_name),
                ("stage", self.stage),
                ("substage", self.substage),
                ("snapshot_id", self.snapshot_id),
                ("official_url", self.official_url),
                ("attribution", self.attribution),
                ("evidence_kind", self.evidence_kind),
                ("dimension_name", self.dimension_name)):
            _text(value, where=f"AuthoredCourseSpec.{name}")
        for name, value in (
                ("course_version", self.course_version),
                ("artifact_version", self.artifact_version),
                ("adapter_version", self.adapter_version),
                ("generator_version", self.generator_version),
                ("parser_version", self.parser_version),
                ("evaluation_budget", self.evaluation_budget)):
            _positive_int(value, where=f"AuthoredCourseSpec.{name}")
        if self.license_id != "CC0-1.0":
            raise AuthoredCourseCommonError("原创课程共用发布器只接受 CC0-1.0")
        if not self.official_url.startswith(("https://", "urn:")):
            raise AuthoredCourseCommonError("原创课程 official_url scheme 非法")


@dataclass(frozen=True)
class AuthoredCompiledSeed:
    """来源 parser 已核准、可交给共用发布器的 typed 原创课程 seed。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    payload_kind: str
    observation_payload: CanonicalJsonObject
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int
    dedup_parts: tuple[Any, ...]
    content_parts: tuple[Any, ...]
    shape_parts: tuple[Any, ...]
    evaluation_dimension: str = ""

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("sample_role", self.sample_role),
                ("payload_kind", self.payload_kind),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredCompiledSeed.{name}")
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredCourseCommonError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredCourseCommonError("label_owner 与 split 不一致")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredCourseCommonError("expected_state 非四态")
        if (not isinstance(self.observation_payload, CanonicalJsonObject)
                or not isinstance(self.expected_payload, CanonicalJsonObject)):
            raise AuthoredCourseCommonError("原创 seed payload 类型错误")
        if not isinstance(self.supersedes_seed_id, str):
            raise AuthoredCourseCommonError("supersedes_seed_id 必须是字符串")
        if (not isinstance(self.evaluation_dimension, str)
                or self.evaluation_dimension.strip() != self.evaluation_dimension):
            raise AuthoredCourseCommonError("evaluation_dimension 必须是无首尾空白字符串")
        _positive_int(self.logical_order, where="AuthoredCompiledSeed.logical_order")
        for name, value in (
                ("dedup_parts", self.dedup_parts),
                ("content_parts", self.content_parts),
                ("shape_parts", self.shape_parts)):
            if not isinstance(value, tuple) or not value:
                raise AuthoredCourseCommonError(f"{name} 不能为空")
            canonical_json_bytes(list(value))


@dataclass(frozen=True)
class AuthoredCourseBuild:
    """返回已发布原创 pack、manifest 和 bundle 校验报告。"""

    pack_root: Path
    manifest: ArtifactManifest
    validation: DatasetBundleValidationReport


def publish_authored_course(
        seeds: tuple[AuthoredCompiledSeed, ...],
        sample_path: str | Path,
        release_root: str | Path,
        spec: AuthoredCourseSpec) -> AuthoredCourseBuild:
    """把核准 seed 分账、校验并原子发布为单来源单许可 PH2 pack。"""
    if not isinstance(seeds, tuple) or not seeds:
        raise AuthoredCourseCommonError("原创课程 seeds 不能为空")
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredCourseCommonError("原创课程 seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredCourseCommonError("原创课程 logical_order 必须严格递增")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredCourseCommonError("原创课程 supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredCourseCommonError("原创课程 supersede 不得跨 family/split")
    source_path = Path(sample_path).resolve()
    sample_sha256 = _sha256_file(source_path)
    dataset_key = _stable_key(
        "dataset", "PH2", spec.source_key, SCHEMA_VERSION)
    artifact_key = _stable_key(
        "artifact", spec.source_key, spec.license_id, spec.stage,
        spec.substage, sample_sha256, spec.course_version,
        spec.artifact_version,
    )
    source_keys = {
        seed.seed_id: _stable_key(
            "source_ref", spec.source_key, sample_sha256, seed.seed_id)
        for seed in seeds
    }
    observation_keys = {
        seed.seed_id: _stable_key(
            "observation", spec.source_key, spec.stage,
            spec.substage, sample_sha256, seed.seed_id)
        for seed in seeds
    }
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    teacher_owner = _stable_key(
        "owner", spec.source_key, spec.substage, "teacher")
    evaluator_owner = _stable_key(
        "owner", spec.source_key, spec.substage, "evaluator")
    for ordinal, seed in enumerate(seeds, start=1):
        source_cluster = _stable_key(
            "source_cluster", spec.source_key, spec.substage, seed.family)
        source = SourceRefRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            source_keys[seed.seed_id],
            spec.source_key,
            spec.snapshot_id,
            seed.seed_id,
            spec.official_url,
            f"data/ph2/{source_path.name}#{seed.seed_id}",
            "sha256:" + sample_sha256,
            sample_sha256,
            spec.license_id,
            "PUBLIC",
            spec.attribution,
            spec.parser_version,
            CanonicalJsonObject.from_value({
                "line_end": ordinal,
                "line_start": ordinal,
                "relative_path": f"data/ph2/{source_path.name}",
            }),
            ordinal,
            source_cluster,
        )
        sources.append(source)
        observation = ObservationRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            observation_keys[seed.seed_id],
            spec.stage,
            spec.substage,
            seed.split,
            "zh",
            f"typed-{spec.substage.casefold().replace('_', '-')}",
            source.stable_key,
            spec.license_id,
            _stable_key("dedup", spec.substage, seed.family, *seed.dedup_parts),
            _stable_key("content", spec.substage, seed.family, *seed.content_parts),
            _stable_key("template", spec.substage, seed.template_family),
            _stable_key("shape", spec.substage, seed.family, *seed.shape_parts),
            "forming" if seed.label_owner == "teacher" else "evaluator",
            seed.sample_role,
            seed.payload_kind,
            seed.observation_payload,
            seed.perturbation_kind,
            (observation_keys[seed.supersedes_seed_id]
             if seed.supersedes_seed_id else None),
            (),
            seed.logical_order,
        )
        observations.append(observation)
        if seed.label_owner == "teacher":
            teachers.append(TeacherEvidenceRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                spec.course_version,
                dataset_key,
                artifact_key,
                _stable_key(
                    "teacher_evidence", spec.substage, seed.seed_id),
                observation.stable_key,
                spec.evidence_kind,
                CanonicalJsonObject.from_value({
                    "expected_payload": seed.expected_payload.to_value(),
                    "expected_state": seed.expected_state,
                    "seed_id": seed.seed_id,
                }),
                source.stable_key,
                spec.stage,
                0,
                teacher_owner,
            ))
        else:
            evaluators.append(EvaluatorLabelRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                spec.course_version,
                dataset_key,
                artifact_key,
                _stable_key(
                    "evaluator_label", spec.substage, seed.seed_id),
                observation.stable_key,
                _stable_key(
                    "dimension",
                    seed.evaluation_dimension or spec.dimension_name,
                ),
                seed.expected_state,
                seed.expected_payload,
                spec.evaluation_budget,
                1,
                spec.stage,
                evaluator_owner,
            ))
    source_tuple = tuple(sources)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teachers)
    evaluator_tuple = tuple(evaluators)
    validation = validate_dataset_bundle(
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
        source_key=spec.source_key,
        license_partition=spec.license_id,
        public_release=True,
    )
    pack_root = (
        Path(release_root).resolve() / "packs" / spec.pack_name).resolve()
    if pack_root.exists():
        raise AuthoredCourseCommonError("原创 pack 已存在，必须使用新 artifact 版本")
    source_clusters = tuple(sorted({
        item.source_cluster_key for item in source_tuple
    }))
    source_by_key = {item.stable_key: item for item in source_tuple}

    def clusters_for(
            items: tuple[ObservationRecord, ...]) -> tuple[StableRecordKey, ...]:
        """返回一组 Observation 实际引用的来源簇。"""
        return tuple(sorted({
            source_by_key[item.source_ref_key].source_cluster_key for item in items
        }))

    train = tuple(item for item in observation_tuple if item.split == "train")
    held_out = tuple(
        item for item in observation_tuple if item.split == "held_out")
    if not train or not held_out or not teacher_tuple or not evaluator_tuple:
        raise AuthoredCourseCommonError("原创课程必须同时含 train/held-out 和双 owner")
    files = (
        write_record_artifact(
            source_tuple,
            pack_root,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF, "source", "source_refs.jsonl.gz", None,
                spec.license_id, source_clusters,
            ),
        ),
        write_record_artifact(
            train,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/train.jsonl.gz", "train",
                spec.license_id, clusters_for(train),
            ),
        ),
        write_record_artifact(
            held_out,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION, "observation",
                "observations/held_out.jsonl.gz", "held_out",
                spec.license_id, clusters_for(held_out),
            ),
        ),
        write_record_artifact(
            teacher_tuple,
            pack_root,
            ArtifactWriteSpec(
                RECORD_TEACHER_EVIDENCE, "teacher",
                "owners/teacher/train.evidence.jsonl.gz", "train",
                spec.license_id, clusters_for(train),
            ),
        ),
        write_record_artifact(
            evaluator_tuple,
            pack_root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL, "evaluator",
                "owners/evaluator/held_out.labels.jsonl.gz", "held_out",
                spec.license_id, clusters_for(held_out),
            ),
        ),
    )
    manifest = ArtifactManifest(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        spec.course_version,
        spec.artifact_version,
        dataset_key,
        artifact_key,
        spec.source_key,
        spec.license_id,
        "PUBLIC",
        spec.adapter_version,
        spec.generator_version,
        spec.parser_version,
        files,
        ("train", "held_out"),
        (spec.stage,),
        source_clusters,
        (),
        spec.stage,
    )
    validate_artifact_manifest(manifest, source_tuple, observation_tuple)
    write_artifact_manifest(manifest, pack_root)
    return AuthoredCourseBuild(pack_root, manifest, validation)


__all__ = [
    "AuthoredCompiledSeed",
    "AuthoredCourseBuild",
    "AuthoredCourseCommonError",
    "AuthoredCourseSpec",
    "publish_authored_course",
]
