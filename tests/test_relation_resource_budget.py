"""真实 Memory owner 的写前门、阶段检查、块检查和失败收据验证。"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from pure_integer_ai.experiments.batch_trace_storage import save_case_trace
from pure_integer_ai.experiments.natural_relation_preflight import write_preflight_json
from pure_integer_ai.experiments.relation_resource_budget import (
    RelationCaseResources, RelationResourceBudgetError, RelationResourceLimits,
    _tree_bytes, before_resource_write, bounded_relation_resources, resource_checkpoint,
)
from pure_integer_ai.experiments.relation_runtime_phase import capture_relation_phases, relation_phase
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from test_batch_case_storage import _captured


LIMITS = RelationResourceLimits(32 * 1024**2, 16 * 1024**2, 8 * 1024**2,
    2 * 1024**3, 2 * 1024**3, 1024**2, 65536)


def _memory(root):
    """建立空 Memory 并提交初始化；probe 只验证真实后端的事务可见性。"""
    if root.drive.upper() != "K:":
        pytest.skip("resource scope requires a named K test root")
    memory = TrainedDialogueMemoryGraph(root / "memory.sqlite3")
    memory.backend._conn.execute("CREATE TABLE budget_probe (value INTEGER)")
    memory.backend.commit()
    return memory


def _receipt(root):
    """读取独占诊断收据，不执行业务查询。"""
    return json.loads((root / "resource/receipt.json").read_bytes())


def test_resource_tree_counts_nested_unmanaged_files_on_every_scan(tmp_path):
    """隐藏、嵌套及两次扫描之间的未管理新增/增长均计入实际长度。"""
    nested = tmp_path / "unmanaged" / "nested"
    nested.mkdir(parents=True)
    (tmp_path / ".hidden").write_bytes(b"a" * 17)
    (nested / "data.bin").write_bytes(b"b" * 29)
    (nested / "empty").mkdir()
    assert _tree_bytes(tmp_path) == 46
    (nested / "data.bin").write_bytes(b"c" * 73)
    (nested.parent / "new.bin").write_bytes(b"d" * 11)
    assert _tree_bytes(tmp_path) == 101


@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "junction"])
def test_resource_tree_rejects_links_and_junctions(tmp_path, kind, outside):
    """真实内部/外部链接及Windows junction均在枚举时拒绝，不消费其目标。"""
    root = tmp_path / "tree"
    root.mkdir()
    target = (tmp_path if outside else root) / "target"
    is_directory = kind != "file_symlink"
    if is_directory:
        target.mkdir()
        (target / "data.bin").write_bytes(b"target")
    else:
        target.write_bytes(b"target")
    link = root / "linked"
    if kind == "junction":
        if os.name != "nt":
            pytest.skip("junction requires Windows")
        subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)
    else:
        try:
            link.symlink_to(target, target_is_directory=is_directory)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows symbolic-link privilege is unavailable")
            raise
    with pytest.raises(RelationResourceBudgetError, match="linked"):
        _tree_bytes(root)


def test_unmanaged_nested_growth_stops_case_and_rolls_back(tmp_path):
    """未走受管理写门的真实新增文件仍由下一进度点拦截并回滚Memory。"""
    memory = _memory(tmp_path)
    with pytest.raises(RelationResourceBudgetError, match="ARTIFACT_PROGRESS_LIMIT"):
        with bounded_relation_resources(tmp_path, tmp_path / "resource", memory, LIMITS) as owner:
            memory.backend._conn.execute("INSERT INTO budget_probe VALUES (1)")
            owner.limits = replace(LIMITS,
                max_artifact_bytes=_tree_bytes(tmp_path) + LIMITS.diagnostic_reserve_bytes + 1024)
            nested = tmp_path / "unmanaged" / "nested"
            nested.mkdir(parents=True)
            (nested / "growth.bin").write_bytes(b"x" * 8192)
            resource_checkpoint("unmanaged_growth")
    assert memory.backend is None
    assert _receipt(tmp_path)["memory_committed"] == 0
    with sqlite3.connect(memory.path) as reader:
        assert reader.execute("SELECT count(*) FROM budget_probe").fetchone()[0] == 0


def test_actual_phase_and_compression_blocks_share_one_resource_owner(tmp_path):
    """真实写门和编码器触发同 owner；只有完整保存后提交的行可见。"""
    memory = _memory(tmp_path)
    _capture, raw = _captured()
    try:
        with bounded_relation_resources(tmp_path, tmp_path / "resource", memory, LIMITS) as owner:
            with capture_relation_phases(tmp_path / "phases", run_identity="resource-contract",
                    attempt_key="complete", case_ordinal=1, required_phases=("query",)):
                with relation_phase("query", "resource-test"):
                    memory.backend._conn.execute("INSERT INTO budget_probe VALUES (1)")
                    memory.backend.commit()
                    with sqlite3.connect(memory.path) as reader:
                        assert reader.execute("SELECT count(*) FROM budget_probe").fetchone()[0] == 0
                    saved = tmp_path / "saved"
                    saved.mkdir()
                    save_case_trace(saved, json.loads(raw), compressed=True,
                        checkpoint=lambda: resource_checkpoint("trace_encode_block", retain=False),
                        before_write=before_resource_write)
                    write_preflight_json(tmp_path, "managed.json", {"complete": 1})
                owner.commit()
        receipt = _receipt(tmp_path)
        assert receipt["status"] == "PASS" and receipt["memory_committed"] == 1
        assert receipt["progress_check_count"] > len(receipt["samples"])
        assert any(row["progress_point"] == "begin:query" for row in receipt["samples"])
        assert receipt["working_set_limit_scope"] == "PROGRESS_CHECKS_ONLY_NO_OS_HARD_LIMIT"
        with sqlite3.connect(memory.path) as reader:
            assert reader.execute("SELECT count(*) FROM budget_probe").fetchone()[0] == 1
    finally:
        memory.close()


@pytest.mark.parametrize("trigger", ["phase", "write", "block"])
def test_exceeded_budget_rolls_back_and_abandons_memory(tmp_path, trigger):
    """三个实际检查入口均停止；即便捕获超限，也不能提交部分 Memory。"""
    memory = _memory(tmp_path)
    _capture, raw = _captured()
    with pytest.raises(RelationResourceBudgetError):
        with bounded_relation_resources(tmp_path, tmp_path / "resource", memory, LIMITS) as owner:
            memory.backend._conn.execute("INSERT INTO budget_probe VALUES (1)")
            memory.backend.commit()
            try:
                if trigger == "write":
                    before_resource_write(tmp_path / "oversize.bin", LIMITS.max_artifact_bytes)
                else:
                    owner.limits = replace(LIMITS, max_working_set_bytes=1)
                    if trigger == "phase":
                        with relation_phase("query", "budget-stop"):
                            pytest.fail("over-limit phase began business work")
                    else:
                        saved = tmp_path / "saved"
                        saved.mkdir()
                        save_case_trace(saved, json.loads(raw), compressed=True,
                            checkpoint=lambda: resource_checkpoint("trace_encode_block"),
                            before_write=before_resource_write)
            except RelationResourceBudgetError:
                owner.commit()
    assert memory.backend is None
    assert _receipt(tmp_path)["memory_committed"] == 0
    assert _receipt(tmp_path)["status"] == "FAIL"
    assert (tmp_path / "resource/resource-stop.json").exists()
    with sqlite3.connect(memory.path) as reader:
        assert reader.execute("SELECT count(*) FROM budget_probe").fetchone()[0] == 0


def test_post_commit_failure_reports_durable_commit_truthfully(tmp_path, monkeypatch):
    """提交后阶段收尾失败不能回滚；收据明确 FAIL 且 memory_committed=1。"""
    memory = _memory(tmp_path)
    original = RelationCaseResources.checkpoint

    def checkpoint(owner, label, **options):
        """在真实 commit 后的阶段末端注入诊断错误，验证持久化边界。"""
        if label == "end:memory_commit":
            raise LookupError("post-commit diagnostic failure")
        return original(owner, label, **options)

    monkeypatch.setattr(RelationCaseResources, "checkpoint", checkpoint)
    with pytest.raises(LookupError, match="post-commit"):
        with bounded_relation_resources(tmp_path, tmp_path / "resource", memory, LIMITS) as owner:
            memory.backend._conn.execute("INSERT INTO budget_probe VALUES (1)")
            owner.commit()
    assert memory.backend is None
    assert _receipt(tmp_path)["status"] == "FAIL"
    assert _receipt(tmp_path)["memory_committed"] == 1
    with sqlite3.connect(memory.path) as reader:
        assert reader.execute("SELECT count(*) FROM budget_probe").fetchone()[0] == 1


def test_receipt_failure_preserves_original_business_error(tmp_path, monkeypatch):
    """诊断文件失败只添加说明，不覆盖原来的业务异常。"""
    memory = _memory(tmp_path)

    def failed_save(owner, name, value):
        """模拟诊断介质拒绝写入，不制造业务状态。"""
        raise OSError("diagnostic unavailable")

    monkeypatch.setattr(RelationCaseResources, "_save", failed_save)
    with pytest.raises(LookupError, match="business failure") as caught:
        with bounded_relation_resources(tmp_path, tmp_path / "resource", memory, LIMITS):
            raise LookupError("business failure")
    assert any("Resource receipt failed" in note for note in caught.value.__notes__)
    assert memory.backend is None
