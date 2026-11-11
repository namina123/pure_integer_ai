"""操作训练共用的物理来源核验；课程与运行模型分离。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.authored_relation_sqlite_course import (
    build_sqlite_relation_adapter, read_teacher_relation_course,
)
from pure_integer_ai.experiments.independent_relation_fact_source import (
    IndependentRelationFactSource, verify_independent_fact_source_course,
)
from pure_integer_ai.experiments.integer_relation_connected_course import (
    _FACT_FIELDS, _read_lines, _seed,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourse, integer_course_from_w06_adapter,
    integer_relation_graph_course_to_w06_adapter,
)


SQLITE_OPERATION_SOURCE = 92430


@dataclass(frozen=True, slots=True)
class SqliteOperationFactSource:
    """训练来源值记录；完整课程、物理来源与原型身份均须重新验证。"""

    source_path: Path
    prototype_root: Path
    course: IntegerRelationGraphCourse
    source_sha256: tuple[int, ...]
    prototype_sha256: tuple[tuple[str, str], ...]


def read_sqlite_operation_fact_source(database, prototype_root):
    """沿现役教师编译链恢复来源，不生成交换文件或写入模型正文。"""
    path, prototypes = Path(database).resolve(strict=True), Path(prototype_root).resolve(strict=True)
    adapter, receipt = build_sqlite_relation_adapter(path, prototypes)
    digest = tuple(bytes.fromhex(receipt["course_sha256"]))
    course = integer_course_from_w06_adapter(adapter,
        source_identity=(SQLITE_OPERATION_SOURCE, *digest),
        license_identity=tuple(map(ord, receipt["license_id"])))
    return SqliteOperationFactSource(path, prototypes, course, digest,
        tuple(sorted(receipt["prototype_sha256"].items())))


def verify_operation_fact_source(source):
    """重新连接物理来源、完整课程和示例；返回核验后的adapter与seed。

    来源类只区分物理恢复协议，不能授予真值。调用方替换任何课程成员、
    来源身份、许可或原型都须在训练前失败；既有独立来源保留原pack审计。
    """
    if type(source) is IndependentRelationFactSource:
        verify_independent_fact_source_course(source)
        adapter = integer_relation_graph_course_to_w06_adapter(
            source.course, require_full_curriculum=False)
        seeds = tuple(_seed(row, index) for index, row in enumerate(
            _read_lines(source.source_path, _FACT_FIELDS), 1))
    elif type(source) is SqliteOperationFactSource:
        current = read_sqlite_operation_fact_source(source.source_path, source.prototype_root)
        if current != source:
            raise ValueError("typed operation teacher course or source identity drifted")
        seeds, _meta, digest, prototypes = read_teacher_relation_course(
            source.source_path, source.prototype_root)
        if (tuple(digest) != source.source_sha256
                or tuple(sorted(prototypes.items())) != source.prototype_sha256):
            raise ValueError("typed operation teacher changed during source verification")
        adapter = integer_relation_graph_course_to_w06_adapter(
            source.course, require_full_curriculum=False)
    else:
        raise ValueError("operation training requires a verified physical teacher source")
    if (not seeds or len({seed.seed_id for seed in seeds}) != len(seeds)
            or any(seed.split != "train" or seed.label_owner != "teacher" for seed in seeds)
            or len(seeds) != len(source.course.records)):
        raise ValueError("operation teacher members or train ownership differ")
    return adapter, seeds
