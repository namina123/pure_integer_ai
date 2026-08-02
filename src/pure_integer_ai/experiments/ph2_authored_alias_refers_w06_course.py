"""发布与篇章指代物理分离的 W-06 PURE_ALIAS/REFERS v2 课程。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_authored_alias_refers_course import (
    read_authored_alias_refers_seeds,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseBuild,
    AuthoredCourseCommonError,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    compile_relation_seed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    LICENSE_ID,
    SOURCE_KEY,
    AuthoredRelationCourseError,
    AuthoredRelationSeed,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    validate_w06_relation_seed,
)


PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--alias-refers-w06-v2"
STAGE = "W-06"
SUBSTAGE = "PURE_ALIAS_REFERS"
_SPEC = AuthoredCourseSpec(
    SOURCE_KEY,
    LICENSE_ID,
    2,
    2,
    1,
    2,
    1,
    PACK_NAME,
    STAGE,
    SUBSTAGE,
    "authored-alias-refers-w06-seed-v2",
    "urn:pure-integer-ai:ph2:authored-alias-refers-w06-v2",
    "Pure Integer AI PH2 authored stable alias/refers W-06 seed",
    "ALIAS_REFERS_STABLE_RELATION_LABEL",
    "pure-alias-refers-stable",
    100,
)


def read_authored_alias_refers_w06_seeds(
        path: str | Path,
        ) -> tuple[AuthoredRelationSeed, ...]:
    """读取 v2 seed，并拒绝 occurrence-bound 或篇章 REFERS。"""
    seeds = read_authored_alias_refers_seeds(path)
    for seed in seeds:
        validate_w06_relation_seed(seed)
    return seeds


def compile_authored_alias_refers_w06_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    """编译 append-only v2 pack，不覆盖已冻结的 v1 课程。"""
    seeds = read_authored_alias_refers_w06_seeds(sample_path)
    try:
        return publish_authored_course(
            tuple(compile_relation_seed(seed) for seed in seeds),
            sample_path,
            release_root,
            _SPEC,
        )
    except AuthoredCourseCommonError as error:
        raise AuthoredRelationCourseError(
            "W-06 stable alias/refers v2 pack 发布失败") from error


__all__ = [
    "PACK_NAME",
    "STAGE",
    "SUBSTAGE",
    "compile_authored_alias_refers_w06_course",
    "read_authored_alias_refers_w06_seeds",
]
