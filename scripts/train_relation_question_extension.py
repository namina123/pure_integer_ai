"""从 K 盘有类型课程增量训练问式整数图，不把课程或答案带入运行包。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pure_integer_ai.experiments.connected_relation_question_course import (
    compile_train_relation_question, compile_train_slot_relation_question,
)
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_PLAIN, SLOT_MODE_REFERENCE,
)
from pure_integer_ai.experiments.ph2_authored_alias_refers_course import (
    read_authored_alias_refers_seeds,
)
from pure_integer_ai.experiments.ph2_authored_causes_course import read_authored_causes_seeds
from pure_integer_ai.experiments.ph2_authored_mereology_course import read_authored_mereology_seeds
from pure_integer_ai.experiments.ph2_authored_subset_member_course import read_authored_subset_member_seeds
from pure_integer_ai.experiments.ph2_authored_semantic_pair_course import read_authored_semantic_pair_seeds
from pure_integer_ai.experiments.ph2_authored_precedes_course import read_authored_precedes_seeds
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph, read_relation_question_graph,
)

COURSE_FORMAT = "PURE_INTEGER_AUTHORED_QUESTION_EXTENSION_COURSE_V1"
SEED_LOADERS = {
    1: ("causes", read_authored_causes_seeds),
    2: ("mereology", read_authored_mereology_seeds),
    3: ("subset_member", read_authored_subset_member_seeds),
    4: ("alias_refers", read_authored_alias_refers_seeds),
    5: ("similar_antonym", read_authored_semantic_pair_seeds),
    6: ("precedes", read_authored_precedes_seeds),
}


def _sha(path: Path) -> str:
    """流式计算训练输入身份，避免将模型整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(course_database: Path, parent_graph: Path, seed_root: Path,
          output_root: Path) -> dict[str, object]:
    """只消费 train 教师语法，保留父图及来源身份并冷恢复新整数图。

    问式只贡献谓词、角色和字面间隔；事实、答案和生成路由由现有模型提供。
    课程与输出必须在 K 盘，输出身份不得覆盖，课程不能包含其它 split。
    """
    course = course_database.resolve(strict=True)
    parent = parent_graph.resolve(strict=True)
    seeds_path = seed_root.resolve(strict=True)
    output = output_root.resolve()
    if course.drive.upper() != "K:" or output.drive.upper() != "K:":
        raise ValueError("question training course/output must be on K:")
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    course_sha = _sha(course)
    with sqlite3.connect(course.as_uri() + "?mode=ro", uri=True) as db:
        metadata = dict(db.execute("SELECT key,value FROM course_meta ORDER BY key"))
        rows = db.execute(
            "SELECT case_id,split_code,seed_family,seed_id,input_text,"
            "answer_binding_index,wh_surface,operation_code,slot_mode "
            "FROM question_case ORDER BY case_id").fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    if (metadata.get("format") != COURSE_FORMAT or metadata.get("split") != "train"
            or metadata.get("license_id") != "CC0-1.0"
            or metadata.get("label_owner") != "teacher"
            or not metadata.get("source_uri") or integrity != "ok"
            or not rows or len({row[0] for row in rows}) != len(rows)):
        raise ValueError("question course provenance/schema is not qualified")
    existing, parent_meta = read_relation_question_graph(parent)
    seed_files = {}
    seeds = {}
    for code in sorted({row[2] for row in rows}):
        if code not in SEED_LOADERS:
            raise ValueError("unregistered teacher seed family")
        name, loader = SEED_LOADERS[code]
        path = seeds_path / f"authored_relation_{name}_seed_v1.jsonl.sample"
        seed_files[name] = {"path": str(path), "sha256": _sha(path)}
        seeds[code] = {item.seed_id: item for item in loader(path)
                       if item.split == "train" and item.label_owner == "teacher"
                       and item.sample_role == "support"}
    digest = bytes.fromhex(course_sha)
    source_parts = tuple(int.from_bytes(digest[index:index + 4], "big")
                         for index in range(0, len(digest), 4))
    added = []
    for case_id, split, family, seed_id, text, answer, wh, operation, mode in rows:
        if (type(case_id) is not int or case_id <= 0 or split != 1
                or operation != 1 or type(mode) is not int
                or mode not in {SLOT_MODE_PLAIN, SLOT_MODE_REFERENCE} or not isinstance(text, str)
                or not text or text.strip() != text or type(answer) is not int):
            raise ValueError("question row is outside the train-only contract")
        seed = seeds[family].get(seed_id)
        if seed is None:
            raise ValueError("question row lacks a train teacher support seed")
        arguments = {"source_case_key": (92029, *source_parts, case_id),
                     "operation_key": (92004, operation)}
        if answer == -1 and wh == "":
            if mode != SLOT_MODE_PLAIN:
                raise ValueError("complete question cannot carry reference slot mode")
            frame = compile_train_relation_question(seed, text, **arguments)
        elif answer >= 0 and isinstance(wh, str) and wh:
            frame = compile_train_slot_relation_question(
                seed, text, answer_binding_index=answer, wh_surface=wh,
                slot_mode=mode, **arguments)
        else:
            raise ValueError("question row mixes complete and WH slot fields")
        if frame is None:
            raise ValueError("question row lacks complete role spans")
        added.append(frame)
    if _sha(course) != course_sha:
        raise ValueError("question course changed during compilation")
    merged = tuple(sorted(set((*existing, *added)), key=lambda item: item.stable_key()))
    if len(merged) != len(existing) + len(rows):
        raise ValueError("question course repeats a physical frame identity")
    output.mkdir(parents=False)
    target = output / "relation_questions.sqlite3"
    graph = materialize_relation_question_graph(
        merged, target, source_identity=tuple(parent_meta["source_identity"]),
        license_identity=tuple(parent_meta["license_identity"]))
    restored, _ = read_relation_question_graph(target)
    if restored != merged:
        raise ValueError("question graph cold restoration differs from training")
    receipt = {
        "format": "PURE_INTEGER_QUESTION_EXTENSION_TRAINING_RECEIPT_V1",
        "status": "TRAINED_AND_COLD_RESTORED", "course_metadata": metadata,
        "course_sha256": course_sha, "trainer_sha256": _sha(Path(__file__)),
        "seed_sources": seed_files, "parent_graph_sha256": _sha(parent),
        "graph_sha256": _sha(target), "parent_frame_count": len(existing),
        "added_frame_count": len(added), "frame_count": len(restored),
        "graph": graph, "source_text_included": 0, "evaluator_read_count": 0,
        "free_dialogue_complete": 0, "production_authorized": 0,
    }
    with (output / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                               separators=(",", ":")) + "\n")
    return receipt


def main() -> int:
    """显式接收课程、父图与 K 盘 run root，打印紧凑训练收据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-database", type=Path, required=True)
    parser.add_argument("--parent-graph", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, default=ROOT / "data/ph2")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.course_database, args.parent_graph, args.seed_root,
                           args.output_root), ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
