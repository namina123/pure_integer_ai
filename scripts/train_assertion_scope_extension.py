"""从已冻结 train SQLite 学习陈述作用域图，课程与开发数据不进入模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.cognition.understanding.query_assertion_scope import TrainedAssertionScopeFrame
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.trained_assertion_scope_graph import (
    materialize_assertion_scope_graph, read_assertion_scope_graph,
)


def train(course_database, target):
    """消费完整来源化角色跨度，剥离中性端点，仅学习算子与角色间隔。"""
    course, target = course_database.resolve(strict=True), target.resolve()
    if (course.drive.upper() != "K:" or target.drive.upper() != "K:"
            or target.exists() or not target.parent.is_dir()):
        raise ValueError("scope 训练需要 K 盘冻结输入与新模型路径")
    original = course.read_bytes()
    sha = hashlib.sha256(original).digest()
    parts = tuple(int.from_bytes(sha[index:index + 4], "big") for index in range(0, 32, 4))
    with sqlite3.connect(course.as_uri() + "?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM course_meta ORDER BY key"))
        rows = db.execute("SELECT case_id,split_code,relation_kind,mode,input_utf8,source_sha256 "
                          "FROM scope_case ORDER BY case_id").fetchall()
        slots = db.execute("SELECT case_id,ordinal,start,end,role_kind FROM scope_slot ORDER BY case_id,ordinal").fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    if (metadata.get("format") != "PURE_INTEGER_ASSERTION_SCOPE_COURSE_V1"
            or metadata.get("split") != "train" or metadata.get("license_id") != "CC0-1.0"
            or metadata.get("label_owner") != "teacher" or not metadata.get("source_uri")
            or integrity != "ok" or not rows):
        raise ValueError("scope 课程来源、划分或完整性不合格")
    grouped = {row[0]: [] for row in rows}
    for case_id, ordinal, start, end, role in slots:
        if case_id not in grouped:
            raise ValueError("scope 课程存在孤儿角色")
        grouped[case_id].append((ordinal, start, end, role))
    frames = []
    for case_id, split, relation, mode, payload, source_sha in rows:
        text = payload.decode("utf-8", "strict")
        members = grouped[case_id]
        if (split != 1 or len(members) != 2 or len(source_sha) != 32
                or tuple(member[0] for member in members) != (0, 1)
                or any(not 0 <= start < end <= len(text) for _, start, end, _ in members)
                or members[0][2] > members[1][1]):
            raise ValueError("scope 教师角色跨度或 split 不合格")
        gaps = (text[:members[0][1]], text[members[0][2]:members[1][1]], text[members[1][2]:])
        source_parts = tuple(int.from_bytes(source_sha[index:index + 4], "big") for index in range(0, 32, 4))
        syntax = OpenRoleFrame((92386, *parts, case_id), authored_relation_identity(relation).stable_key(),
            (92385, *source_parts), tuple(authored_relation_role_identity(member[3]).stable_key() for member in members),
            tuple(tuple(map(ord, gap)) for gap in gaps), (0, len(text)))
        frames.append(TrainedAssertionScopeFrame(syntax, mode, (92384, *parts), (91696, 1)))
    if course.read_bytes() != original:
        raise ValueError("scope 课程在学习期间发生变化")
    learned = tuple(sorted(frames, key=lambda frame: frame.stable_key()))
    receipt = materialize_assertion_scope_graph(learned, target)
    if read_assertion_scope_graph(target) != learned:
        raise ValueError("scope 图冷恢复与学习状态不同")
    return {**receipt, "course_sha256": sha.hex(), "graph_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "course_metadata": metadata, "source_text_included": 0, "evaluators_read": 0,
            "free_dialogue_complete": 0, "production_authorized": 0}


def main():
    """显式输入课程与 K 模型目标，仅输出紧凑训练收据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = train(args.course_database, args.output)
    with args.output.with_suffix(".receipt.json").open("x", encoding="ascii") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
