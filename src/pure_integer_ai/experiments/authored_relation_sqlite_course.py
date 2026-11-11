"""把 K 盘有类型教师关系课程接入现役 authored/W-06 编译链。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCourseSpec, build_authored_course_records,
)
from pure_integer_ai.experiments.relation_sqlite_prototypes import (
    SqliteRelationPrototype, compile_sqlite_relation_prototype, read_sqlite_relation_prototypes,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_w06_adapter import adapt_w06_training_payload
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.storage.integer_codec import decode_integer_tuple


COURSE_FORMAT = "PURE_INTEGER_AUTHORED_RELATION_SQLITE_COURSE_V1"
COURSE_SCHEMA = """
CREATE TABLE course_meta(name TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
CREATE TABLE domain(domain_id INTEGER PRIMARY KEY,name_utf8 BLOB NOT NULL) STRICT;
CREATE TABLE prototype(prototype_id INTEGER PRIMARY KEY,filename TEXT NOT NULL,
    seed_id TEXT NOT NULL,sha256 BLOB NOT NULL CHECK(length(sha256)=32)) STRICT;
CREATE TABLE teacher_relation(case_id INTEGER PRIMARY KEY,
    domain_id INTEGER NOT NULL REFERENCES domain,prototype_id INTEGER NOT NULL REFERENCES prototype,
    logical_order INTEGER NOT NULL UNIQUE CHECK(logical_order>0)) STRICT;
CREATE TABLE teacher_endpoint(case_id INTEGER NOT NULL REFERENCES teacher_relation,
    endpoint_ordinal INTEGER NOT NULL CHECK(endpoint_ordinal>=0),
    surface_values BLOB NOT NULL,local_id INTEGER NOT NULL CHECK(local_id>0),
    PRIMARY KEY(case_id,endpoint_ordinal)) STRICT;
"""


def file_sha256(path: Path) -> bytes:
    """分块计算课程或来源身份，避免整库读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _map_payload(value, replacements, *, endpoint_field=False):
    """只替换显式local_id字段；相等的预算、方向或强度整数不能被端点污染。"""
    if type(value) is int and endpoint_field:
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_map_payload(item, replacements, endpoint_field=endpoint_field) for item in value]
    if isinstance(value, dict):
        return {key: _map_payload(item, replacements,
            endpoint_field=key.endswith(('_local_id', '_local_ids'))) for key, item in value.items()}
    return value


def read_teacher_relation_course(database: Path, prototype_root: Path, *, limit=None, offset=0):
    """兼容既有公开调用，返回原关系seed及完整课程身份。"""
    records, meta, digest, sources = _read_teacher_relation_records(
        database, prototype_root, limit=limit, offset=offset)
    return tuple(item.seed for item in records), meta, digest, sources


def _read_teacher_relation_records(database: Path, prototype_root: Path, *, limit=None, offset=0):
    """恢复教师 seed；课程和原型身份固定，拒绝开发标签及端点跨度漂移。

    SQLite 只声明领域、原型和端点。固定原型提供先已许可的 schema/角色/间隔，
    课程提供新知识；所有自然语言值都来自这两份来源，算法不内置实体或答案。
    """
    path = database.resolve(strict=True)
    source_root = prototype_root.resolve(strict=True)
    if path.drive.upper() != "K:":
        raise ValueError("教师课程必须位于显式 K 盘")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("课程规模探针必须是正整数")
    if type(offset) is not int or offset < 0:
        raise ValueError("课程续接偏移必须是非负整数")
    course_sha = file_sha256(path)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT name,value FROM course_meta ORDER BY name"))
        prototypes = tuple(db.execute("SELECT prototype_id,filename,seed_id,sha256 FROM prototype ORDER BY prototype_id"))
        rows = tuple(db.execute("SELECT case_id,domain_id,prototype_id,logical_order FROM teacher_relation ORDER BY logical_order"))
        endpoints = tuple(db.execute("SELECT case_id,endpoint_ordinal,surface_values,local_id FROM teacher_endpoint ORDER BY case_id,endpoint_ordinal"))
        if (db.execute("PRAGMA integrity_check").fetchone() != ("ok",)
                or tuple(db.execute("PRAGMA foreign_key_check"))):
            raise ValueError("教师课程 SQLite 完整性失败")
    if (meta.get("format") != COURSE_FORMAT or meta.get("license_id") != "CC0-1.0"
            or meta.get("split") != "train" or meta.get("label_owner") != "teacher"
            or not meta.get("source_uri", "").startswith("urn:")
            or not meta.get("attribution") or not rows
            or len(rows) != int(meta.get("record_count", "0"))):
        raise ValueError("教师课程的来源、许可或 train-only owner 不闭合")
    index, source_hashes = {}, {}
    for code, filename, seed_id, expected_sha in prototypes:
        source = source_root / filename
        if source.name != filename:
            raise ValueError('教师原型必须是登记的直接来源文件')
        actual_sha = file_sha256(source)
        if actual_sha != expected_sha:
            raise ValueError("教师关系原型来源 SHA 漂移")
        found = tuple(item for item in read_sqlite_relation_prototypes(source) if item.seed.seed_id == seed_id
                      and item.seed.label_owner == "teacher" and item.seed.split == "train"
                      and item.seed.sample_role == "support" and item.seed.perturbation_kind == "NONE")
        if len(found) != 1:
            raise ValueError("教师关系原型没有唯一 train/support seed")
        index[code] = found[0]
        source_hashes[filename] = actual_sha.hex()
    by_case = {}
    for case, ordinal, payload, local_id in endpoints:
        values = decode_integer_tuple(payload)
        if not values or any(not 0 <= value <= 0x10FFFF for value in values):
            raise ValueError("教师端点不是合法整数码点序列")
        by_case.setdefault(case, []).append((ordinal, "".join(map(chr, values)), local_id))
    seeds = []
    selected = rows[offset:] if limit is None else rows[offset:offset+limit]
    if not selected:
        raise ValueError("课程续接范围没有新训练记录")
    for case, domain, prototype, order in selected:
        record = index[prototype]
        original = record.seed
        declared = by_case.get(case, ())
        if tuple(item[0] for item in declared) != tuple(range(len(original.endpoints))):
            raise ValueError("教师端点数量和顺序未覆盖原型")
        local_ids = {old.local_id: item[2] for old, item in zip(original.endpoints, declared, strict=True)}
        if len(set(local_ids.values())) != len(local_ids):
            raise ValueError("同一命题的端点身份竞争")
        # 只替换来源中已声明的端点区间；其余间隔与锚点仍逐码点保留。
        cursor, text, replaced = 0, "", []
        shift_before_anchor = 0
        for old, (_ordinal, surface, local_id) in zip(original.endpoints, declared, strict=True):
            if old.start < original.anchor.end and old.end > original.anchor.start:
                raise ValueError("教师关系锚点不能与角色端点重叠")
            text += original.surface[cursor:old.start]
            start = len(text)
            text += surface
            replaced.append(replace(old, surface_fragment=surface, start=start,
                                    end=len(text), local_id=local_id))
            if old.end <= original.anchor.start:
                shift_before_anchor += len(surface) - (old.end - old.start)
            cursor = old.end
        text += original.surface[cursor:]
        seed = replace(original, seed_id=f"sqlite-teacher-{course_sha.hex()}-{case}",
            family=f"sqlite-teacher-domain-{domain}", template_family=f"sqlite-teacher-prototype-{prototype}",
            surface=text, anchor=replace(original.anchor,
                start=original.anchor.start + shift_before_anchor,
                end=original.anchor.end + shift_before_anchor), endpoints=tuple(replaced),
            expected_payload=CanonicalJsonObject.from_value(
                _map_payload(original.expected_payload.to_value(), local_ids)),
            context_local_id=case, logical_order=order)
        seeds.append(SqliteRelationPrototype(seed, record.rational_role_values))
    if file_sha256(path) != course_sha:
        raise ValueError("教师课程在恢复过程中发生变化")
    return tuple(seeds), meta, course_sha, source_hashes


def build_sqlite_relation_adapter(database: Path, prototype_root: Path, *, limit=None, offset=0):
    """将有类型课程直接编译为现役 W-06 输入，不落盘 JSONL 或读取开发答案。"""
    records, meta, course_sha, sources = _read_teacher_relation_records(
        database, prototype_root, limit=limit, offset=offset)
    compiled, substages = [], {}
    for record in records:
        item, substage = compile_sqlite_relation_prototype(record)
        compiled.append(item)
        substages[record.seed.seed_id] = substage
    spec = AuthoredCourseSpec("AUTHORED_CC0_V1", "CC0-1.0", 1, 1, 1, 1, 1,
        "SQLITE_TEACHER_RELATIONS_V1", "W-06", "SQLITE_RELATION",
        course_sha.hex(), meta["source_uri"], meta["attribution"],
        "TYPED_RELATION_LABEL", "typed-relation", 100)
    source, observation, teacher, evaluator, _validation = build_authored_course_records(
        tuple(compiled), database, spec, seed_substages=substages, training_only=True,
        source_ordinal_offset=offset)
    if evaluator:
        raise ValueError("教师训练不得包含 evaluator")
    adapter = adapt_w06_training_payload(W06TrainingPayload(source, observation, teacher),
                                        require_full_curriculum=False)
    return adapter, {"course_sha256": course_sha.hex(), "record_count": len(records),
        "relation_families": sorted({record.seed.relation_family for record in records}),
        "relation_substages": sorted({value[0] for value in substages.values()}),
        "source_uri": meta["source_uri"], "license_id": "CC0-1.0",
        "prototype_sha256": sources, "evaluator_read_count": 0, "selection_offset": offset}
