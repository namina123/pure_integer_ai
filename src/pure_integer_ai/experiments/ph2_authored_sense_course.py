"""把 AUTHORED_CC0_V1 极小 seed 编译为 W-03 sense/概念边界资料 pack。"""
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
    SAMPLE_ROLES,
    SCHEMA_VERSION,
    ArtifactManifest,
    CanonicalJsonObject,
    DatasetContractError,
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
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_validation import (
    DatasetBundleValidationReport,
    validate_artifact_manifest,
    validate_dataset_bundle,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--sense-v1"
STAGE = "W-03"
SUBSTAGE = "SENSE_CONCEPT_BOUNDARY"

_SEED_FIELDS = frozenset({
    "candidate_sense",
    "context",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "sample_role",
    "seed_id",
    "split",
    "supersedes_seed_id",
    "surface",
    "template_family",
})
_COURSE_SAMPLE_ROLES = frozenset({"support", "refute", "conflict", "supersede"})


class AuthoredSenseCourseError(RuntimeError):
    """原创 sense seed 或编译输出违反版本、owner、顺序或许可边界。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求 seed 字段为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredSenseCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredSenseCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求 seed 顺序为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredSenseCourseError(f"{where} 必须是正严格整数")
    return value


def _stable_key(namespace: str, *parts: Any) -> StableRecordKey:
    """从完整规范值产生版本化正整数身份键，摘要只作键分量。"""
    payload = canonical_json_bytes({
        "namespace": namespace,
        "parts": list(parts),
        "version": 1,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


def _sha256_file(path: Path) -> str:
    """流式计算 sample 文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuthoredSenseSeed:
    """一条项目原创 sense/概念边界 seed 及其私有 owner 标签。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    surface: str
    context: str
    candidate_sense: str
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("surface", self.surface),
                ("context", self.context),
                ("candidate_sense", self.candidate_sense),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredSenseSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredSenseSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredSenseCourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredSenseCourseError("label_owner 与 split 不一致")
        if self.sample_role not in _COURSE_SAMPLE_ROLES:
            raise AuthoredSenseCourseError("sample_role 不属于首类 sense 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredSenseCourseError("supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredSenseCourseError("非 supersede seed 不得声明替代目标")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredSenseCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredSenseCourseError("expected_payload 类型错误")
        _positive_int(self.logical_order, where="AuthoredSenseSeed.logical_order")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredSenseSeed":
        """从严格字段集合恢复一条 seed，不接受静默扩展字段。"""
        if set(value) != _SEED_FIELDS:
            raise AuthoredSenseCourseError("sense seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredSenseCourseError("AUTHORED_CC0_V1 seed 必须是 CC0-1.0")
        return cls(
            str(value["seed_id"]),
            str(value["family"]),
            str(value["template_family"]),
            str(value["label_owner"]),
            str(value["split"]),
            str(value["sample_role"]),
            str(value["surface"]),
            str(value["context"]),
            str(value["candidate_sense"]),
            str(value["expected_state"]),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            str(value["perturbation_kind"]),
            str(value["supersedes_seed_id"]),
            value["logical_order"],
        )


@dataclass(frozen=True)
class AuthoredSenseCourseBuild:
    """返回已发布 pack、manifest 和 bundle 校验报告。"""

    pack_root: Path
    manifest: ArtifactManifest
    validation: DatasetBundleValidationReport


def read_authored_sense_seeds(path: str | Path) -> tuple[AuthoredSenseSeed, ...]:
    """严格读取规范 JSONL sample，并核对 owner family、顺序和替代目标。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredSenseCourseError("sense sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredSenseCourseError("sense sample 必须非空并以换行结束")
    seeds: list[AuthoredSenseSeed] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredSenseCourseError(f"sense sample 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredSenseCourseError(
                f"sense sample 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        seeds.append(AuthoredSenseSeed.from_dict(value))
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredSenseCourseError("sense seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredSenseCourseError("sense logical_order 必须严格递增且唯一")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredSenseCourseError("sense supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredSenseCourseError("sense supersede 不得跨 family/split")
    teacher_families = {seed.family for seed in seeds if seed.label_owner == "teacher"}
    evaluator_families = {seed.family for seed in seeds if seed.label_owner == "evaluator"}
    teacher_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "teacher"
    }
    evaluator_templates = {
        seed.template_family for seed in seeds if seed.label_owner == "evaluator"
    }
    if (not teacher_families or not evaluator_families
            or teacher_families & evaluator_families
            or teacher_templates & evaluator_templates):
        raise AuthoredSenseCourseError("teacher/evaluator family 必须非空且互斥")
    if {seed.sample_role for seed in seeds} != _COURSE_SAMPLE_ROLES:
        raise AuthoredSenseCourseError("首类 sense sample 必须覆盖四种 sample role")
    return tuple(seeds)


def _build_records(
        seeds: tuple[AuthoredSenseSeed, ...],
        sample_path: Path,
        sample_sha256: str,
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        ) -> tuple[
            tuple[SourceRefRecord, ...],
            tuple[ObservationRecord, ...],
            tuple[TeacherEvidenceRecord, ...],
            tuple[EvaluatorLabelRecord, ...]]:
    """把私有 seed 分账为学生 Observation、teacher Evidence 和 evaluator label。"""
    source_keys = {
        seed.seed_id: _stable_key("source_ref", sample_sha256, seed.seed_id)
        for seed in seeds
    }
    observation_keys = {
        seed.seed_id: _stable_key("observation", sample_sha256, seed.seed_id)
        for seed in seeds
    }
    sources: list[SourceRefRecord] = []
    observations: list[ObservationRecord] = []
    teachers: list[TeacherEvidenceRecord] = []
    evaluators: list[EvaluatorLabelRecord] = []
    teacher_owner = _stable_key("owner", SOURCE_KEY, "teacher-sense-v1")
    evaluator_owner = _stable_key("owner", SOURCE_KEY, "evaluator-sense-v1")
    for ordinal, seed in enumerate(seeds, start=1):
        source_cluster = _stable_key("source_cluster", SOURCE_KEY, seed.family)
        source = SourceRefRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            COURSE_VERSION,
            dataset_key,
            artifact_key,
            source_keys[seed.seed_id],
            SOURCE_KEY,
            "authored-sense-seed-v1",
            seed.seed_id,
            "urn:pure-integer-ai:ph2:authored-sense-v1",
            f"data/ph2/{sample_path.name}#{seed.seed_id}",
            "sha256:" + sample_sha256,
            sample_sha256,
            LICENSE_ID,
            "PUBLIC",
            "Pure Integer AI PH2 authored sense seed",
            PARSER_VERSION,
            CanonicalJsonObject.from_value({
                "line_end": ordinal,
                "line_start": ordinal,
                "relative_path": f"data/ph2/{sample_path.name}",
            }),
            ordinal,
            source_cluster,
        )
        sources.append(source)
        supersedes_key = (
            observation_keys[seed.supersedes_seed_id]
            if seed.supersedes_seed_id else None
        )
        observation = ObservationRecord(
            FORMAT_VERSION,
            SCHEMA_VERSION,
            COURSE_VERSION,
            dataset_key,
            artifact_key,
            observation_keys[seed.seed_id],
            STAGE,
            SUBSTAGE,
            seed.split,
            "zh",
            "typed-sense-boundary-query",
            source.stable_key,
            LICENSE_ID,
            _stable_key(
                "dedup", seed.family, seed.surface,
                seed.context, seed.candidate_sense),
            _stable_key("content", seed.family, seed.surface, seed.context),
            _stable_key("template", seed.template_family),
            _stable_key("shape", seed.family, "sense-boundary-query-v1"),
            "forming" if seed.label_owner == "teacher" else "evaluator",
            seed.sample_role,
            "SenseBoundaryQuery",
            CanonicalJsonObject.from_value({
                "candidate_sense": seed.candidate_sense,
                "context": seed.context,
                "query_kind": "sense_boundary",
                "surface": seed.surface,
            }),
            seed.perturbation_kind,
            supersedes_key,
            (),
            seed.logical_order,
        )
        observations.append(observation)
        private_payload = CanonicalJsonObject.from_value({
            "expected_payload": seed.expected_payload.to_value(),
            "expected_state": seed.expected_state,
            "seed_id": seed.seed_id,
        })
        if seed.label_owner == "teacher":
            teachers.append(TeacherEvidenceRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                COURSE_VERSION,
                dataset_key,
                artifact_key,
                _stable_key("teacher_evidence", seed.seed_id),
                observation.stable_key,
                "SENSE_BOUNDARY_LABEL",
                private_payload,
                source.stable_key,
                STAGE,
                0,
                teacher_owner,
            ))
        else:
            evaluators.append(EvaluatorLabelRecord(
                FORMAT_VERSION,
                SCHEMA_VERSION,
                COURSE_VERSION,
                dataset_key,
                artifact_key,
                _stable_key("evaluator_label", seed.seed_id),
                observation.stable_key,
                _stable_key("dimension", "sense-concept-boundary"),
                seed.expected_state,
                seed.expected_payload,
                100,
                1,
                STAGE,
                evaluator_owner,
            ))
    return tuple(sources), tuple(observations), tuple(teachers), tuple(evaluators)


def compile_authored_sense_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredSenseCourseBuild:
    """编译并原子发布首个 W-03 极小 pack，不触碰 Core、Memory 或 mastered。"""
    source_path = Path(sample_path).resolve()
    seeds = read_authored_sense_seeds(source_path)
    sample_sha256 = _sha256_file(source_path)
    dataset_key = _stable_key("dataset", "PH2", SOURCE_KEY, SCHEMA_VERSION)
    artifact_key = _stable_key(
        "artifact", SOURCE_KEY, LICENSE_ID, sample_sha256,
        COURSE_VERSION, ARTIFACT_VERSION,
    )
    sources, observations, teachers, evaluators = _build_records(
        seeds, source_path, sample_sha256, dataset_key, artifact_key)
    validation = validate_dataset_bundle(
        sources,
        observations,
        teachers,
        evaluators,
        source_key=SOURCE_KEY,
        license_partition=LICENSE_ID,
        public_release=True,
    )
    pack_root = (Path(release_root).resolve() / "packs" / PACK_NAME).resolve()
    if pack_root.exists():
        raise AuthoredSenseCourseError("sense pack 已存在，必须使用新 artifact 版本")
    source_clusters = tuple(sorted({item.source_cluster_key for item in sources}))
    source_by_key = {item.stable_key: item for item in sources}

    def clusters_for_observations(
            items: tuple[ObservationRecord, ...]) -> tuple[StableRecordKey, ...]:
        """返回一组 Observation 实际引用的真实来源簇。"""
        return tuple(sorted({
            source_by_key[item.source_ref_key].source_cluster_key for item in items
        }))

    train = tuple(item for item in observations if item.split == "train")
    held_out = tuple(item for item in observations if item.split == "held_out")
    files = (
        write_record_artifact(
            sources,
            pack_root,
            ArtifactWriteSpec(
                RECORD_SOURCE_REF,
                "source",
                "source_refs.jsonl.gz",
                None,
                LICENSE_ID,
                source_clusters,
            ),
        ),
        write_record_artifact(
            train,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION,
                "observation",
                "observations/train.jsonl.gz",
                "train",
                LICENSE_ID,
                clusters_for_observations(train),
            ),
        ),
        write_record_artifact(
            held_out,
            pack_root,
            ArtifactWriteSpec(
                RECORD_OBSERVATION,
                "observation",
                "observations/held_out.jsonl.gz",
                "held_out",
                LICENSE_ID,
                clusters_for_observations(held_out),
            ),
        ),
        write_record_artifact(
            teachers,
            pack_root,
            ArtifactWriteSpec(
                RECORD_TEACHER_EVIDENCE,
                "teacher",
                "owners/teacher/train.evidence.jsonl.gz",
                "train",
                LICENSE_ID,
                clusters_for_observations(train),
            ),
        ),
        write_record_artifact(
            evaluators,
            pack_root,
            ArtifactWriteSpec(
                RECORD_EVALUATOR_LABEL,
                "evaluator",
                "owners/evaluator/held_out.labels.jsonl.gz",
                "held_out",
                LICENSE_ID,
                clusters_for_observations(held_out),
            ),
        ),
    )
    manifest = ArtifactManifest(
        FORMAT_VERSION,
        SCHEMA_VERSION,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        dataset_key,
        artifact_key,
        SOURCE_KEY,
        LICENSE_ID,
        "PUBLIC",
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        files,
        ("train", "held_out"),
        (STAGE,),
        source_clusters,
        (),
        STAGE,
    )
    validate_artifact_manifest(manifest, sources, observations)
    write_artifact_manifest(manifest, pack_root)
    return AuthoredSenseCourseBuild(pack_root, manifest, validation)


__all__ = [
    "ADAPTER_VERSION",
    "ARTIFACT_VERSION",
    "AuthoredSenseCourseBuild",
    "AuthoredSenseCourseError",
    "AuthoredSenseSeed",
    "COURSE_VERSION",
    "GENERATOR_VERSION",
    "LICENSE_ID",
    "PACK_NAME",
    "PARSER_VERSION",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_sense_course",
    "read_authored_sense_seeds",
]
