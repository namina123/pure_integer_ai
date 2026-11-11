"""SQLite 教师课程必须沿真实 W-06 来源、角色和证据链编译。"""
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.experiments.authored_relation_sqlite_course import (
    COURSE_FORMAT, COURSE_SCHEMA, build_sqlite_relation_adapter,
    file_sha256, read_teacher_relation_course,
)
from pure_integer_ai.experiments.ph2_authored_mereology_course import read_authored_mereology_seeds
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data/ph2"
FILENAME = "authored_relation_mereology_seed_v1.jsonl.sample"


def _course(tmp_path):
    """用现役许可原型与两个独立端点声明组成最小有类型教师来源。"""
    prototype = next(seed for seed in read_authored_mereology_seeds(SOURCE_ROOT / FILENAME)
                     if seed.label_owner == "teacher" and seed.sample_role == "support")
    path = tmp_path / "course.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(COURSE_SCHEMA)
        db.executemany("INSERT INTO course_meta VALUES(?,?)", (
            ("format", COURSE_FORMAT), ("split", "train"), ("label_owner", "teacher"),
            ("license_id", "CC0-1.0"), ("source_uri", "urn:test:sqlite-teacher"),
            ("attribution", "CC0 test annotation"), ("record_count", "1")))
        db.execute("INSERT INTO domain VALUES(1,?)", (b"test",))
        db.execute("INSERT INTO prototype VALUES(1,?,?,?)",
                   (FILENAME, prototype.seed_id, file_sha256(SOURCE_ROOT / FILENAME)))
        db.execute("INSERT INTO teacher_relation VALUES(1,1,1,1)")
        db.executemany("INSERT INTO teacher_endpoint VALUES(1,?,?,?)", (
            (0, encode_integer_tuple(tuple(map(ord, "新部件"))), 9000001),
            (1, encode_integer_tuple(tuple(map(ord, "新整体"))), 9000002)))
    return path, prototype


def test_sqlite_course_reaches_real_w06_typed_adapter(tmp_path):
    """端点替换须保留来源间隔/角色、更新教师整数目标并通过实际 adapter。"""
    path, original = _course(tmp_path)
    seeds, _meta, digest, _sources = read_teacher_relation_course(path, SOURCE_ROOT)
    seed, = seeds
    assert seed.bindings == original.bindings
    assert seed.surface == "新部件是新整体的一部分。"
    assert seed.expected_payload.to_value()["expected_whole_local_id"] == 9000002
    assert seed.surface[seed.anchor.start:seed.anchor.end] == seed.anchor.surface_fragment
    adapter, receipt = build_sqlite_relation_adapter(path, SOURCE_ROOT)
    assert len(adapter.observations) == len(adapter.evidence) == 1
    assert not adapter.rejections
    assert receipt["course_sha256"] == digest.hex()
    assert receipt["evaluator_read_count"] == 0
    source = adapter.observations[0].source.record
    assert source.local_sha256 == digest.hex()
    assert source.official_url == "urn:test:sqlite-teacher"
    assert source.license_id == "CC0-1.0"


def test_sqlite_course_incremental_slices_preserve_full_observation_and_source_identities(tmp_path):
    """规模探针后只学习后续记录；切片不改变来源、原序或任何候选身份。"""
    path, _original = _course(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE course_meta SET value='2' WHERE name='record_count'")
        db.execute('INSERT INTO teacher_relation VALUES(2,1,1,2)')
        db.executemany('INSERT INTO teacher_endpoint VALUES(2,?,?,?)', (
            (0, encode_integer_tuple(tuple(map(ord, '另一部件'))), 9000003),
            (1, encode_integer_tuple(tuple(map(ord, '另一整体'))), 9000004)))
    complete, _ = build_sqlite_relation_adapter(path, SOURCE_ROOT)
    first, _ = build_sqlite_relation_adapter(path, SOURCE_ROOT, limit=1)
    following, receipt = build_sqlite_relation_adapter(path, SOURCE_ROOT, offset=1, limit=1)
    key = lambda candidate: candidate.observation.logical_order
    assert sorted(complete.candidates, key=key) == sorted((*first.candidates, *following.candidates), key=key)
    assert receipt['record_count'] == receipt['selection_offset'] == 1
    with pytest.raises(ValueError):
        build_sqlite_relation_adapter(path, SOURCE_ROOT, offset=2)


@pytest.mark.parametrize("mutation", ("prototype", "evaluator", "endpoint"))
def test_sqlite_course_rejects_source_owner_or_role_drift(tmp_path, mutation):
    """来源身份、训练 owner 或角色缺失不得借 SQLite 载体取得资格。"""
    path, _original = _course(tmp_path)
    with sqlite3.connect(path) as db:
        if mutation == "prototype":
            db.execute("UPDATE prototype SET sha256=?", (bytes(32),))
        elif mutation == "evaluator":
            db.execute("UPDATE course_meta SET value='evaluator' WHERE name='label_owner'")
        else:
            db.execute("DELETE FROM teacher_endpoint WHERE endpoint_ordinal=1")
    with pytest.raises(ValueError):
        build_sqlite_relation_adapter(path, SOURCE_ROOT)
