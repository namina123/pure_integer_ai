"""只读核验知识增量的完成边界；收据不能替代已提交 checkpoint。"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.authored_relation_sqlite_course import file_sha256


@dataclass(frozen=True, slots=True)
class CompletedKnowledgeTraining:
    """已核对的训练身份，不声明自然语言能力或发布资格。"""

    model_sha256: str
    parent_core_sha256: str
    course_sha256: str
    generated_relation_count: int
    completed_stage_count: int = 4


def model_source_body_count(database: Path) -> int:
    """只读统计实际 Core 正文，不以收据中的零值替代物理存储检查。"""
    with closing(sqlite3.connect(database.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT count(*) FROM source_record "
            "WHERE raw_text IS NOT NULL AND length(raw_text)>0").fetchone()[0]


def _sha(value) -> bytes:
    """身份必须是完整 32 字节 SHA，不能接受截断摘要。"""
    if type(value) is not str or len(value) != 64:
        raise ValueError("训练收据 SHA 不完整")
    result = bytes.fromhex(value)
    if len(result) != 32:
        raise ValueError("训练收据 SHA 不完整")
    return result


def _keys(values) -> tuple[tuple[int, ...], ...]:
    """恢复完整命题成员；禁止空、重复或非整数身份。"""
    if (type(values) is not list or not values
            or any(type(key) is not list or not key
                   or any(type(item) is not int or item < 0 for item in key)
                   for key in values)):
        raise ValueError("训练生成命题身份不完整")
    result = tuple(tuple(key) for key in values)
    if len(set(result)) != len(result):
        raise ValueError("训练生成命题身份重复")
    return result


def verify_completed_knowledge_training(run_root: Path) -> CompletedKnowledgeTraining:
    """在恢复或启动后继前核对四阶段、模型、冻结身份和实际生成分母。

    只读已结束运行；运行中的 cursor 不作为监控入口。权威进度来自有类型
    SQLite，JSON 收据仅携带元数据，不能把未提交阶段升级为训练成功。
    """
    root = run_root.resolve(strict=True)
    if root.drive.upper() != "K:" or root == Path(root.anchor):
        raise ValueError("完成校验必须使用显式 K 盘运行根")
    cursor_path = root / "training_cursor.sqlite3"
    with closing(sqlite3.connect(cursor_path.as_uri() + "?mode=ro", uri=True)) as cursor:
        if cursor.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("训练 checkpoint SQLite 完整性失败")
        stages = tuple(cursor.execute("SELECT stage_id,core_sha256 FROM stage ORDER BY stage_id"))
        resources = tuple(cursor.execute(
            "SELECT stage_id,endpoint_utf8,artifact_bytes,free_disk_bytes "
            "FROM resource_sample ORDER BY stage_id"))
        frozen = cursor.execute("SELECT value_utf8 FROM run_identity WHERE name='frozen'").fetchone()
    if (tuple(row[0] for row in stages) != (1, 2, 3, 4)
            or any(type(row[1]) is not bytes or len(row[1]) != 32 for row in stages)
            or tuple(row[0] for row in resources) != (1, 2, 3, 4) or frozen is None):
        raise ValueError("训练四阶段 checkpoint 或资源记录未完整提交")
    for _stage, endpoint, artifact_bytes, free_bytes in resources:
        sample = json.loads(endpoint)
        if (artifact_bytes > 4 * 1024**3 or free_bytes < 10 * 1024**3
                or max(sample.get("working_set_bytes", 0),
                       sample.get("process_peak_working_set_bytes", 0)) > 8 * 1024**3):
            raise ValueError("训练 checkpoint 超出冻结资源预算")
    receipt = json.loads((root / "training_receipt.json").read_bytes())
    generation = json.loads((root / "generation_receipt.json").read_bytes())
    identity = json.loads(frozen[0])
    count = receipt.get("new_generated_relations")
    if (receipt.get("format") != "PURE_INTEGER_KNOWLEDGE_EXTENSION_TRAINING_V1"
            or receipt.get("identity") != identity or type(count) is not int or count <= 0
            or type(identity.get("limit")) is not int or count > identity["limit"]
            or receipt.get("source", {}).get("record_count") != count
            or receipt.get("parent_core_unchanged") != 1
            or receipt.get("source_body_restored") != 0
            or receipt.get("free_dialogue_complete") != 0
            or receipt.get("production_authorized") != 0):
        raise ValueError("训练收据与冻结身份、来源或完成分母不同")
    source_sha = _sha(identity["course_sha256"])
    if _sha(receipt["source"]["course_sha256"]) != source_sha:
        raise ValueError("训练来源课程身份漂移")
    model_sha = _sha(receipt["model_sha256"])
    if stages[-1][1] != model_sha or file_sha256(root / "model/training.sqlite3") != model_sha:
        raise ValueError("已提交 checkpoint 与当前模型 SHA 不同")
    if identity.get("source_body_storage") == "EXTERNAL_TYPED_ARCHIVE_V1" and (
            receipt.get("model_source_body_count") != 0
            or model_source_body_count(root / "model/training.sqlite3") != 0
            or _sha(receipt.get("training_source_archive_sha256"))
               != file_sha256(root / "training_sources.sqlite3")):
        raise ValueError("外部训练来源归档或模型零正文合同未闭合")
    outputs = receipt.get("outputs")
    if type(outputs) is not list or len(outputs) != count or any(
            type(item) is not dict or not item.get("surface") for item in outputs):
        raise ValueError("实际图生成输出分母不完整")
    output_keys = _keys([item.get("proposition") for item in outputs])
    generated_keys = _keys(generation.get("proposition_keys"))
    if (set(output_keys) != set(generated_keys) or len(output_keys) != count
            or generation.get("connector_count") != count
            or generation.get("selected_proposition_count") != count
            or generation.get("selection_explicit") != 1
            or generation.get("uncovered_proposition_keys") != []):
        raise ValueError("训练收据与实际生成命题覆盖不同")
    return CompletedKnowledgeTraining(model_sha.hex(),
        _sha(identity["parent_core_sha256"]).hex(), source_sha.hex(), count)


def main() -> int:
    """提供只读完成信号，不重新训练、查询、生成或修补旧 checkpoint。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    result = verify_completed_knowledge_training(parser.parse_args().run_root)
    print(json.dumps(asdict(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
