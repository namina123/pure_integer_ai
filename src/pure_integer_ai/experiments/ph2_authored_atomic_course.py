"""W-05 原创 typed seed 的规范读取、覆盖校验和 D-02 pack 编排 facade。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_atomic_compile import (
    compile_atomic_seed,
)
from pure_integer_ai.experiments.ph2_authored_atomic_schema import (
    ALLOWED_ROLE_KINDS,
    LICENSE_ID,
    PACK_NAME,
    PREDICATE_REGISTRY,
    REQUIRED_PERTURBATIONS,
    REQUIRED_SAMPLE_ROLES,
    ROLE_ACTOR,
    ROLE_LOCATION,
    ROLE_PATIENT,
    ROLE_RECIPIENT,
    ROLE_REGISTRY,
    SOURCE_KEY,
    STAGE,
    SUBSTAGE,
    AtomicBindingSeed,
    AtomicOccurrenceSeed,
    AuthoredAtomicCourseError,
    AuthoredAtomicSeed,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


_SPEC = AuthoredCourseSpec(
    SOURCE_KEY,
    LICENSE_ID,
    1,
    1,
    1,
    1,
    1,
    PACK_NAME,
    STAGE,
    SUBSTAGE,
    "authored-atomic-seed-v1",
    "urn:pure-integer-ai:ph2:authored-atomic-v1",
    "Pure Integer AI PH2 authored atomic proposition seed",
    "ATOMIC_PROPOSITION_LABEL",
    "occurrence-role-atomic-proposition",
    100,
)


def read_authored_atomic_seeds(
        path: str | Path) -> tuple[AuthoredAtomicSeed, ...]:
    """读取规范 atomic JSONL，并核对 owner、覆盖面和 supersede 顺序。"""
    sample_path = Path(path)
    try:
        payload = sample_path.read_bytes()
    except OSError as error:
        raise AuthoredAtomicCourseError("atomic sample 无法读取") from error
    if not payload or not payload.endswith(b"\n"):
        raise AuthoredAtomicCourseError("atomic sample 必须非空并以换行结束")
    seeds: list[AuthoredAtomicSeed] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line == b"\n":
            raise AuthoredAtomicCourseError(
                f"atomic sample 第 {line_number} 行为空或缺换行")
        try:
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
        except DatasetContractError as error:
            raise AuthoredAtomicCourseError(
                f"atomic sample 第 {line_number} 行不是规范 JSON") from error
        assert isinstance(value, dict)
        try:
            seeds.append(AuthoredAtomicSeed.from_dict(value))
        except DatasetContractError as error:
            raise AuthoredAtomicCourseError(
                f"atomic sample 第 {line_number} 行 payload 非法") from error
    if len({seed.seed_id for seed in seeds}) != len(seeds):
        raise AuthoredAtomicCourseError("atomic seed_id 重复")
    orders = [seed.logical_order for seed in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredAtomicCourseError("atomic logical_order 必须严格递增")
    index = {seed.seed_id: seed for seed in seeds}
    for seed in seeds:
        if not seed.supersedes_seed_id:
            continue
        target = index.get(seed.supersedes_seed_id)
        if target is None or target.logical_order >= seed.logical_order:
            raise AuthoredAtomicCourseError("atomic supersede 必须指向更早 seed")
        if target.family != seed.family or target.split != seed.split:
            raise AuthoredAtomicCourseError("atomic supersede 不得跨 family/split")
        if (target.perturbation_kind != "OCCURRENCE_OMISSION"
                or seed.perturbation_kind != "OCCURRENCE_RESTORE"):
            raise AuthoredAtomicCourseError("atomic supersede 必须修正 occurrence omission")
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
        raise AuthoredAtomicCourseError("teacher/evaluator family 必须非空且互斥")
    if {seed.sample_role for seed in seeds} != REQUIRED_SAMPLE_ROLES:
        raise AuthoredAtomicCourseError("atomic sample 必须覆盖四种 sample role")
    if not REQUIRED_PERTURBATIONS.issubset({
            seed.perturbation_kind for seed in seeds}):
        raise AuthoredAtomicCourseError("atomic sample 缺少必需反向破坏")
    return tuple(seeds)


def compile_authored_atomic_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译并发布 W-05 occurrence/角色/原子命题极小 pack。"""
    seeds = read_authored_atomic_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_atomic_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredAtomicCourseError("atomic pack 发布失败") from error


__all__ = [
    "ALLOWED_ROLE_KINDS",
    "AuthoredAtomicCourseError",
    "AuthoredAtomicSeed",
    "AtomicBindingSeed",
    "AtomicOccurrenceSeed",
    "LICENSE_ID",
    "PACK_NAME",
    "PREDICATE_REGISTRY",
    "ROLE_ACTOR",
    "ROLE_LOCATION",
    "ROLE_PATIENT",
    "ROLE_RECIPIENT",
    "ROLE_REGISTRY",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_atomic_course",
    "read_authored_atomic_seeds",
]
