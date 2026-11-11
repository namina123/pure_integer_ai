"""中断、旧收据及模型漂移不能跨越知识训练完成边界。"""
import hashlib
import json
import sqlite3

import pytest

from pure_integer_ai.experiments.relation_knowledge_checkpoint import (
    verify_completed_knowledge_training,
)


def _completed(tmp_path):
    """生成真实 SQLite checkpoint 与独立收据，训练模型仅作为摘要输入。"""
    root = tmp_path / "run"
    (root / "model").mkdir(parents=True)
    core = root / "model/training.sqlite3"
    with sqlite3.connect(core) as db:
        db.execute("CREATE TABLE trained_value(value INTEGER NOT NULL) STRICT")
        db.execute("INSERT INTO trained_value VALUES(42)")
        db.execute("CREATE TABLE source_record(raw_text TEXT NOT NULL) STRICT")
    digest = hashlib.sha256(core.read_bytes()).digest()
    identity = {"course_sha256": "11" * 32, "parent_core_sha256": "22" * 32, "limit": 2}
    encoded = json.dumps(identity, sort_keys=True).encode("ascii")
    with sqlite3.connect(root / "training_cursor.sqlite3") as db:
        db.executescript("""
            CREATE TABLE run_identity(name TEXT PRIMARY KEY,value_utf8 BLOB NOT NULL) STRICT;
            CREATE TABLE stage(stage_id INTEGER PRIMARY KEY,core_sha256 BLOB NOT NULL) STRICT;
            CREATE TABLE resource_sample(stage_id INTEGER PRIMARY KEY,endpoint_utf8 BLOB NOT NULL,
                artifact_bytes INTEGER NOT NULL,free_disk_bytes INTEGER NOT NULL) STRICT;
        """)
        db.execute("INSERT INTO run_identity VALUES('frozen',?)", (encoded,))
        db.executemany("INSERT INTO stage VALUES(?,?)", ((stage, digest) for stage in range(1, 5)))
        db.executemany("INSERT INTO resource_sample VALUES(?,?,?,?)", (
            (stage, b'{"working_set_bytes":1024}', 4096, 20 * 1024**3) for stage in range(1, 5)))
    receipt = {"format": "PURE_INTEGER_KNOWLEDGE_EXTENSION_TRAINING_V1",
        "identity": identity, "new_generated_relations": 2,
        "source": {"record_count": 2, "course_sha256": "11" * 32},
        "parent_core_unchanged": 1, "source_body_restored": 0,
        "free_dialogue_complete": 0, "production_authorized": 0,
        "model_sha256": digest.hex(),
        "outputs": [{"proposition": [7, 101], "surface": "first"},
                    {"proposition": [7, 102], "surface": "second"}]}
    generation = {"proposition_keys": [[7, 101], [7, 102]], "connector_count": 2,
        "selected_proposition_count": 2, "selection_explicit": 1, "uncovered_proposition_keys": []}
    (root / "training_receipt.json").write_text(json.dumps(receipt), encoding="ascii")
    (root / "generation_receipt.json").write_text(json.dumps(generation), encoding="ascii")
    return root


def test_completed_checkpoint_matches_current_model_and_generation(tmp_path):
    """完成身份是值对象；校验不改变任何权威进度或模型文件。"""
    root = _completed(tmp_path)
    before = {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = verify_completed_knowledge_training(root)
    assert result.generated_relation_count == 2
    assert result.completed_stage_count == 4
    assert before == {path.name: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("mutation", ("last_stage", "middle_stage", "resource"))
def test_receipt_cannot_complete_interrupted_checkpoint(tmp_path, mutation):
    """即使最终 JSON 已写出，缺少任一已提交阶段或资源记录仍必须拒绝。"""
    root = _completed(tmp_path)
    with sqlite3.connect(root / "training_cursor.sqlite3") as db:
        if mutation == "resource":
            db.execute("DELETE FROM resource_sample WHERE stage_id=4")
        else:
            db.execute("DELETE FROM stage WHERE stage_id=?", (4 if mutation == "last_stage" else 2,))
    with pytest.raises(ValueError, match="四阶段"):
        verify_completed_knowledge_training(root)


def test_completed_receipt_cannot_hide_core_change(tmp_path):
    """保存的完成 SHA 与实际 Core 逐字节不同必须关闭后继。"""
    root = _completed(tmp_path)
    with sqlite3.connect(root / "model/training.sqlite3") as db:
        db.execute("UPDATE trained_value SET value=43")
    with pytest.raises(ValueError, match="模型 SHA"):
        verify_completed_knowledge_training(root)


@pytest.mark.parametrize("mutation", ("identity", "source", "coverage", "resource_budget"))
def test_completed_receipt_cannot_hide_identity_or_coverage_drift(tmp_path, mutation):
    """恢复不能把别次课程、未生成命题或越界阶段冒充本轮训练完成。"""
    root = _completed(tmp_path)
    path = root / ("generation_receipt.json" if mutation == "coverage" else "training_receipt.json")
    value = json.loads(path.read_bytes())
    if mutation == "identity":
        value["identity"]["limit"] = 3
    elif mutation == "source":
        value["source"]["course_sha256"] = "33" * 32
    elif mutation == "coverage":
        value["proposition_keys"][1] = [7, 103]
    else:
        with sqlite3.connect(root / "training_cursor.sqlite3") as db:
            db.execute("UPDATE resource_sample SET artifact_bytes=? WHERE stage_id=4", (5 * 1024**3,))
    path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError):
        verify_completed_knowledge_training(root)


@pytest.mark.parametrize("mutation", ("valid", "body", "archive"))
def test_external_archive_contract_checks_physical_model_and_archive(tmp_path, mutation):
    """零正文声明不能覆盖实际模型正文；训练归档也须匹配本轮完整 SHA。"""
    root = _completed(tmp_path)
    receipt_path = root / "training_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["identity"]["source_body_storage"] = "EXTERNAL_TYPED_ARCHIVE_V1"
    archive = root / "training_sources.sqlite3"
    with sqlite3.connect(archive) as db:
        db.execute("CREATE TABLE source_record(raw_text TEXT NOT NULL) STRICT")
        db.execute("INSERT INTO source_record VALUES('teacher')")
    receipt["model_source_body_count"] = 0
    receipt["training_source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    if mutation == "body":
        with sqlite3.connect(root / "model/training.sqlite3") as db:
            db.execute("INSERT INTO source_record VALUES('body must stay outside model')")
        receipt["model_sha256"] = hashlib.sha256((root / "model/training.sqlite3").read_bytes()).hexdigest()
    elif mutation == "archive":
        receipt["training_source_archive_sha256"] = "44" * 32
    with sqlite3.connect(root / "training_cursor.sqlite3") as db:
        db.execute("UPDATE run_identity SET value_utf8=? WHERE name='frozen'",
                   (json.dumps(receipt["identity"], sort_keys=True).encode("ascii"),))
        db.execute("UPDATE stage SET core_sha256=? WHERE stage_id=4", (bytes.fromhex(receipt["model_sha256"]),))
    receipt_path.write_text(json.dumps(receipt), encoding="ascii")
    if mutation == "valid":
        assert verify_completed_knowledge_training(root).generated_relation_count == 2
    else:
        with pytest.raises(ValueError, match="归档或模型零正文"):
            verify_completed_knowledge_training(root)
