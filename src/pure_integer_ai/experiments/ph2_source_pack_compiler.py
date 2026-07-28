"""D-02 外部来源到现有四类 record/artifact 的统一薄编译器。"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    DatasetArtifactIOError,
    read_artifact_manifest,
    read_record_artifact,
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetBundleValidationReport,
    DatasetValidationError,
    validate_artifact_manifest,
    validate_dataset_bundle,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SOURCE_PACK_CONTRACT_VERSION,
    SourceObservationSeed,
    SourcePackContractError,
    SourcePackCoverageManifest,
    SourcePackSpec,
    stable_source_pack_key,
)


SOURCE_PAYLOAD_KEYS = frozenset({
    "combination_axes",
    "combination_cluster_key",
    "definitive_truth_authoritative",
    "raw_observation",
    "raw_observation_append_only",
    "raw_observation_sha256",
    "source_pack_contract_version",
})
SOURCE_SPAN_BINDING_KEYS = frozenset({
    "raw_observation_sha256",
    "raw_snapshot_manifest_relative_path",
    "raw_snapshot_manifest_sha256",
})
READER_KINDS = ("student", "teacher", "evaluator", "source_audit")


class SourcePackCompilerError(RuntimeError):
    """外部来源 pack 编译、恢复、owner grant 或组合审计失败。"""


@dataclass(frozen=True)
class SourcePackBundle:
    """一个从正式 artifact 重读并合取校验的四 owner bundle。"""

    manifest: ArtifactManifest
    sources: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teachers: tuple[TeacherEvidenceRecord, ...]
    evaluators: tuple[EvaluatorLabelRecord, ...]
    validation: DatasetBundleValidationReport
    combination_audit: CanonicalJsonObject


@dataclass(frozen=True)
class SourcePackBuild:
    """返回 pack 位置、已核 bundle 和 fresh/resume 事实。"""

    pack_root: Path
    bundle: SourcePackBundle
    published: bool
    contract_sha256: str

    @property
    def manifest(self) -> ArtifactManifest:
        """便捷返回正式 ArtifactManifest。"""
        return self.bundle.manifest


def _sha256_value(value: Any) -> str:
    """返回规范 JSON 值 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contract_sha256(
        spec: SourcePackSpec,
        seeds: tuple[SourceObservationSeed, ...]) -> str:
    """绑定 spec 与全部 seed 身份但不复制 raw 正文。"""
    return _sha256_value({
        "contract_version": SOURCE_PACK_CONTRACT_VERSION,
        "seeds": [item.to_contract_dict() for item in seeds],
        "spec": spec.to_contract_dict(),
    })


def _validate_inputs(
        spec: SourcePackSpec,
        seeds: tuple[SourceObservationSeed, ...]) -> None:
    """要求 pack/seed 唯一、顺序稳定且 snapshot manifest 当前字节一致。"""
    if not isinstance(spec, SourcePackSpec):
        raise SourcePackCompilerError("source pack spec 类型错误")
    if not isinstance(seeds, tuple) or not seeds:
        raise SourcePackCompilerError("source pack seeds 不能为空")
    if any(not isinstance(item, SourceObservationSeed) for item in seeds):
        raise SourcePackCompilerError("source pack seed 类型错误")
    ids = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if len(ids) != len(set(ids)):
        raise SourcePackCompilerError("source pack seed_id 重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise SourcePackCompilerError("source pack logical_order 必须严格递增")
    pack_path = Path(spec.pack_name)
    if pack_path.name != spec.pack_name or spec.pack_name in {".", ".."}:
        raise SourcePackCompilerError("source pack_name 必须是单层安全目录名")


def _records_from_inputs(
        spec: SourcePackSpec,
        seeds: tuple[SourceObservationSeed, ...],
        ) -> tuple[
            tuple[SourceRefRecord, ...],
            tuple[ObservationRecord, ...],
            tuple[TeacherEvidenceRecord, ...],
            tuple[EvaluatorLabelRecord, ...],
            str,
        ]:
    """纯函数构造四类现有 record，不执行文件或 backend 写入。"""
    _validate_inputs(spec, seeds)
    contract_sha256 = _contract_sha256(spec, seeds)
    dataset_key = stable_source_pack_key(
        "dataset", "PH2", spec.source_key, SCHEMA_VERSION)
    artifact_key = stable_source_pack_key(
        "artifact", spec.source_key, spec.license_id, spec.stage,
        spec.substage, spec.course_version, spec.artifact_version,
        contract_sha256,
    )
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    for ordinal, seed in enumerate(seeds, start=1):
        source_key = stable_source_pack_key(
            "source_ref", spec.source_key, spec.snapshot_id, seed.seed_id,
            seed.local_sha256, seed.source_identity)
        observation_key = stable_source_pack_key(
            "observation", spec.source_key, spec.stage, spec.substage,
            seed.seed_id, seed.raw_observation_sha256)
        source_cluster = stable_source_pack_key(
            "source_cluster", spec.source_key, *seed.source_cluster_parts)
        combination_cluster = stable_source_pack_key(
            "combination_cluster", spec.source_key,
            *seed.combination_parts)
        raw_span = seed.source_span.to_value()
        if set(raw_span) & SOURCE_SPAN_BINDING_KEYS:
            raise SourcePackCompilerError("source_span 占用保留 snapshot 绑定字段")
        source_span = dict(raw_span)
        source_span.update({
            "raw_observation_sha256": seed.raw_observation_sha256,
            "raw_snapshot_manifest_relative_path": (
                spec.raw_snapshot_manifest_relative_path),
            "raw_snapshot_manifest_sha256": (
                spec.raw_snapshot_manifest_sha256),
        })
        source = SourceRefRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            source_key,
            spec.source_key,
            spec.snapshot_id,
            seed.seed_id,
            spec.official_url,
            seed.source_identity,
            seed.upstream_checksum,
            seed.local_sha256,
            spec.license_id,
            spec.redistribution_policy,
            spec.attribution,
            spec.parser_version,
            CanonicalJsonObject.from_value(source_span),
            ordinal,
            source_cluster,
        )
        observation = ObservationRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            observation_key,
            spec.stage,
            spec.substage,
            seed.split,
            seed.language,
            seed.representation,
            source.stable_key,
            spec.license_id,
            stable_source_pack_key(
                "dedup", spec.source_key, *seed.dedup_parts),
            stable_source_pack_key(
                "content", spec.source_key, *seed.content_parts),
            stable_source_pack_key(
                "template", spec.source_key, *seed.template_parts),
            stable_source_pack_key(
                "shape", spec.source_key, *seed.shape_parts),
            "forming" if seed.split == "train" else "evaluator",
            seed.sample_role,
            "RAW_SOURCE_OBSERVATION_V1",
            CanonicalJsonObject.from_value({
                "combination_axes": seed.combination_axes.to_value(),
                "combination_cluster_key": combination_cluster.to_list(),
                "definitive_truth_authoritative": 0,
                "raw_observation": seed.raw_observation.to_value(),
                "raw_observation_append_only": 1,
                "raw_observation_sha256": seed.raw_observation_sha256,
                "source_pack_contract_version": SOURCE_PACK_CONTRACT_VERSION,
            }),
            seed.perturbation_kind,
            None,
            (),
            seed.logical_order,
        )
        teacher_owner = stable_source_pack_key(
            "owner", spec.source_key, spec.substage, seed.split, "teacher")
        evaluator_owner = stable_source_pack_key(
            "owner", spec.source_key, spec.substage, seed.split, "evaluator")
        teacher = TeacherEvidenceRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            stable_source_pack_key(
                "teacher_evidence", spec.source_key, seed.seed_id),
            observation.stable_key,
            "SOURCE_PARSER_RECEIPT_V1",
            CanonicalJsonObject.from_value({
                "definitive_truth_authoritative": 0,
                "parser_version": spec.parser_version,
                "raw_observation_sha256": seed.raw_observation_sha256,
                "source_ref_key": source.stable_key.to_list(),
            }),
            source.stable_key,
            spec.stage,
            3,
            teacher_owner,
        )
        evaluator = EvaluatorLabelRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            dataset_key,
            artifact_key,
            stable_source_pack_key(
                "evaluator_label", spec.source_key, seed.seed_id),
            observation.stable_key,
            stable_source_pack_key(
                "dimension", "SOURCE_OBSERVATION_INTEGRITY_V1"),
            "TRUE",
            CanonicalJsonObject.from_value({
                "definitive_truth_authoritative": 0,
                "raw_observation_sha256": seed.raw_observation_sha256,
                "source_binding_required": 1,
            }),
            1,
            1,
            spec.stage,
            evaluator_owner,
        )
        sources.append(source)
        observations.append(observation)
        teachers.append(teacher)
        evaluators.append(evaluator)
    return (
        tuple(sources), tuple(observations), tuple(teachers),
        tuple(evaluators), contract_sha256,
    )


def validate_source_pack_payloads(
        observations: tuple[ObservationRecord, ...]) -> CanonicalJsonObject:
    """核验 raw 不变、组合轴显式存在且组合 cluster 不跨 split。"""
    if not observations:
        raise SourcePackCompilerError("source pack Observation 不能为空")
    assignments: dict[StableRecordKey, str] = {}
    rows: list[dict[str, Any]] = []
    for item in observations:
        payload = item.typed_payload.to_value()
        if set(payload) != SOURCE_PAYLOAD_KEYS:
            raise SourcePackCompilerError("source Observation payload 字段不精确")
        if (payload["source_pack_contract_version"]
                != SOURCE_PACK_CONTRACT_VERSION):
            raise SourcePackCompilerError("source pack contract version 漂移")
        if payload["raw_observation_append_only"] != 1:
            raise SourcePackCompilerError("raw Observation 未声明 append-only")
        if payload["definitive_truth_authoritative"] != 0:
            raise SourcePackCompilerError("外部 Observation 不得成为真值权威")
        raw_sha256 = _sha256_value(payload["raw_observation"])
        if raw_sha256 != payload["raw_observation_sha256"]:
            raise SourcePackCompilerError("raw Observation hash 漂移")
        combination_key = StableRecordKey.from_value(
            payload["combination_cluster_key"],
            where="source pack combination cluster key",
        )
        prior = assignments.get(combination_key)
        if prior is not None and prior != item.split:
            raise SourcePackCompilerError("combination cluster 跨 split")
        assignments[combination_key] = item.split
        axes = payload["combination_axes"]
        if not isinstance(axes, dict) or not axes:
            raise SourcePackCompilerError("combination axes 不能为空")
        rows.append({
            "combination_axes_sha256": _sha256_value(axes),
            "combination_cluster_key": combination_key.to_list(),
            "observation_key": item.stable_key.to_list(),
            "raw_observation_sha256": raw_sha256,
            "split": item.split,
        })
    rows.sort(key=lambda value: tuple(value["observation_key"]))
    return CanonicalJsonObject.from_value({
        "combination_cluster_count": len(assignments),
        "raw_observation_append_only": 1,
        "rows": rows,
    })


def _clusters_for(
        observations: tuple[ObservationRecord, ...],
        source_by_key: dict[StableRecordKey, SourceRefRecord],
        ) -> tuple[StableRecordKey, ...]:
    """返回文件内 Observation 实际引用的来源簇。"""
    return tuple(sorted({
        source_by_key[item.source_ref_key].source_cluster_key
        for item in observations
    }))


def _write_fresh_pack(
        spec: SourcePackSpec,
        seeds: tuple[SourceObservationSeed, ...],
        release_root: Path,
        ) -> tuple[Path, str]:
    """在同父目录 staging 中构建完整 pack，再原子发布目录。"""
    sources, observations, teachers, evaluators, contract_sha256 = (
        _records_from_inputs(spec, seeds))
    validation = validate_dataset_bundle(
        sources,
        observations,
        teachers,
        evaluators,
        source_key=spec.source_key,
        license_partition=spec.license_id,
        public_release=spec.redistribution_policy == "PUBLIC",
    )
    validate_source_pack_payloads(observations)
    packs_root = (release_root / "packs").resolve()
    packs_root.mkdir(parents=True, exist_ok=True)
    target = (packs_root / spec.pack_name).resolve()
    if not target.is_relative_to(packs_root) or target.exists():
        raise SourcePackCompilerError("source pack 目标已存在或逃逸")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{spec.pack_name}.building-", dir=packs_root)).resolve()
    if not staging.is_relative_to(packs_root):
        raise SourcePackCompilerError("source pack staging 逃逸")
    try:
        source_clusters = tuple(sorted({
            item.source_cluster_key for item in sources}))
        source_by_key = {item.stable_key: item for item in sources}
        files = [write_record_artifact(
            sources,
            staging,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF,
                "source",
                "source_refs.jsonl.gz",
                None,
                spec.license_id,
                source_clusters,
            ),
        )]
        for split in sorted({item.split for item in observations}):
            split_observations = tuple(
                item for item in observations if item.split == split)
            split_teachers = tuple(
                item for item in teachers
                if any(observation.stable_key == item.observation_key
                       for observation in split_observations))
            split_evaluators = tuple(
                item for item in evaluators
                if any(observation.stable_key == item.observation_key
                       for observation in split_observations))
            clusters = _clusters_for(split_observations, source_by_key)
            files.extend((
                write_record_artifact(
                    split_observations,
                    staging,
                    ArtifactWriteSpec(
                        RECORD_OBSERVATION,
                        "observation",
                        f"observations/{split}.jsonl.gz",
                        split,
                        spec.license_id,
                        clusters,
                    ),
                ),
                write_record_artifact(
                    split_teachers,
                    staging,
                    ArtifactWriteSpec(
                        RECORD_TEACHER_EVIDENCE,
                        "teacher",
                        f"owners/teacher/{split}.evidence.jsonl.gz",
                        split,
                        spec.license_id,
                        clusters,
                    ),
                ),
                write_record_artifact(
                    split_evaluators,
                    staging,
                    ArtifactWriteSpec(
                        RECORD_EVALUATOR_LABEL,
                        "evaluator",
                        f"owners/evaluator/{split}.labels.jsonl.gz",
                        split,
                        spec.license_id,
                        clusters,
                    ),
                ),
            ))
        manifest = ArtifactManifest(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            spec.course_version,
            spec.artifact_version,
            sources[0].dataset_key,
            sources[0].artifact_key,
            spec.source_key,
            spec.license_id,
            spec.redistribution_policy,
            spec.adapter_version,
            spec.generator_version,
            spec.parser_version,
            tuple(files),
            tuple(sorted({item.split for item in observations})),
            (spec.stage,),
            source_clusters,
            (),
            spec.earliest_invalidated_stage,
        )
        validate_artifact_manifest(manifest, sources, observations)
        if validation.artifact_key != manifest.stable_key:
            raise SourcePackCompilerError("source pack validation/manifest 漂移")
        write_artifact_manifest(manifest, staging)
        os.replace(staging, target)
    except Exception:
        if staging.exists() and staging.is_relative_to(packs_root):
            shutil.rmtree(staging)
        raise
    return target, contract_sha256


def read_source_pack(pack_root: str | Path) -> SourcePackBundle:
    """重读全部 owner 文件并合取既有 D-02 与组合/raw 不变量。"""
    root = Path(pack_root).resolve()
    manifest = read_artifact_manifest(root / "manifest.json")
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    for identity in manifest.files:
        records = read_record_artifact(root, identity)
        for record in records:
            if isinstance(record, SourceRefRecord):
                sources.append(record)
            elif isinstance(record, ObservationRecord):
                observations.append(record)
            elif isinstance(record, TeacherEvidenceRecord):
                teachers.append(record)
            elif isinstance(record, EvaluatorLabelRecord):
                evaluators.append(record)
            else:
                raise SourcePackCompilerError("source pack 含未知 record")
    source_tuple = tuple(sources)
    observation_tuple = tuple(observations)
    teacher_tuple = tuple(teachers)
    evaluator_tuple = tuple(evaluators)
    validation = validate_dataset_bundle(
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
        source_key=manifest.source_key,
        license_partition=manifest.license_partition,
        public_release=manifest.redistribution_policy == "PUBLIC",
    )
    validate_artifact_manifest(manifest, source_tuple, observation_tuple)
    combination = validate_source_pack_payloads(observation_tuple)
    return SourcePackBundle(
        manifest,
        source_tuple,
        observation_tuple,
        teacher_tuple,
        evaluator_tuple,
        validation,
        combination,
    )


def compile_or_resume_source_pack(
        spec: SourcePackSpec,
        seeds: tuple[SourceObservationSeed, ...],
        release_root: str | Path,
        ) -> SourcePackBuild:
    """fresh 时原子发布；resume 时逐记录核对相同合同，禁止领养漂移产物。"""
    _validate_inputs(spec, seeds)
    release = Path(release_root).resolve()
    pack_root = (release / "packs" / spec.pack_name).resolve()
    expected = _records_from_inputs(spec, seeds)
    contract_sha256 = expected[4]
    published = False
    if pack_root.exists():
        if not pack_root.is_dir():
            raise SourcePackCompilerError("source pack resume 路径不是目录")
    else:
        pack_root, actual_contract = _write_fresh_pack(spec, seeds, release)
        if actual_contract != contract_sha256:
            raise SourcePackCompilerError("source pack fresh contract 漂移")
        published = True
    bundle = read_source_pack(pack_root)
    actual_groups = (
        bundle.sources, bundle.observations, bundle.teachers, bundle.evaluators)
    expected_groups = expected[:4]
    if any(
            {item.stable_key: item.to_dict() for item in actual}
            != {item.stable_key: item.to_dict() for item in planned}
            for actual, planned in zip(actual_groups, expected_groups)):
        raise SourcePackCompilerError("source pack resume 记录与输入合同漂移")
    return SourcePackBuild(pack_root, bundle, published, contract_sha256)


def read_source_pack_view(
        pack_root: str | Path,
        *,
        reader_kind: Literal["student", "teacher", "evaluator", "source_audit"],
        split: str | None = None,
        ) -> tuple[
            ObservationRecord | TeacherEvidenceRecord
            | EvaluatorLabelRecord | SourceRefRecord, ...]:
    """按显式 grant 读取单 owner；student 正式路径只能得到 Observation。"""
    if reader_kind not in READER_KINDS:
        raise SourcePackCompilerError("source pack reader_kind 未注册")
    root = Path(pack_root).resolve()
    manifest = read_artifact_manifest(root / "manifest.json")
    owner = {
        "student": "observation",
        "teacher": "teacher",
        "evaluator": "evaluator",
        "source_audit": "source",
    }[reader_kind]
    if reader_kind == "source_audit":
        if split is not None:
            raise SourcePackCompilerError("source_audit 不接受 split")
    elif split not in manifest.splits:
        raise SourcePackCompilerError("owner view split 不在 manifest")
    identities = tuple(
        item for item in manifest.files
        if item.owner_kind == owner
        and (reader_kind == "source_audit" or item.split == split)
    )
    if not identities:
        raise SourcePackCompilerError("owner view 没有授权文件")
    records = tuple(
        record
        for identity in identities
        for record in read_record_artifact(root, identity)
    )
    expected_type = {
        "student": ObservationRecord,
        "teacher": TeacherEvidenceRecord,
        "evaluator": EvaluatorLabelRecord,
        "source_audit": SourceRefRecord,
    }[reader_kind]
    if any(not isinstance(item, expected_type) for item in records):
        raise SourcePackCompilerError("owner view 出现越权 record")
    return records


def write_source_pack_coverage_manifest(
        manifest: SourcePackCoverageManifest,
        path: str | Path) -> Path:
    """独占或幂等发布来源覆盖账，禁止原地改版。"""
    if not isinstance(manifest, SourcePackCoverageManifest):
        raise SourcePackCompilerError("coverage manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise SourcePackCompilerError("coverage manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise SourcePackCompilerError("coverage manifest 无法发布") from error
    return target


def read_source_pack_coverage_manifest(
        path: str | Path) -> SourcePackCoverageManifest:
    """严格回读规范来源覆盖账。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise SourcePackCompilerError("coverage manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = SourcePackCoverageManifest.from_dict(value)
    except SourcePackCompilerError:
        raise
    except (OSError, SourcePackContractError) as error:
        raise SourcePackCompilerError("coverage manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise SourcePackCompilerError("coverage manifest 非规范字节")
    return manifest


__all__ = [
    "READER_KINDS",
    "SOURCE_PAYLOAD_KEYS",
    "SourcePackBuild",
    "SourcePackBundle",
    "SourcePackCompilerError",
    "compile_or_resume_source_pack",
    "read_source_pack",
    "read_source_pack_coverage_manifest",
    "read_source_pack_view",
    "validate_source_pack_payloads",
    "write_source_pack_coverage_manifest",
]
