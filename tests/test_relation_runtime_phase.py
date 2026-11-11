"""实际调用点诊断的嵌套、业务失败和单次被动硬预算边界。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pure_integer_ai.experiments.relation_case_budget import passive_relation_case_budget
from pure_integer_ai.experiments.relation_runtime_phase import (
    capture_relation_phases, observed_relation_phase,
)


def _k_path(tmp_path: Path) -> Path:
    """诊断落盘只在显式 K 临时目录运行，其他平台保留原运行时测试。"""
    if tmp_path.drive.upper() != "K:":
        pytest.skip("phase diagnostics require an explicit K test directory")
    return tmp_path


def test_actual_call_sites_overlap_without_changing_values_or_failures(tmp_path):
    """外层查询与内层 Core 分流，原返回值/异常及未运行阶段均保持。"""
    root = _k_path(tmp_path)

    @observed_relation_phase("core")
    def inner(value):
        """提供明确行为以验证诊断不会改变调用结果。"""
        if value < 0:
            raise LookupError("business failure")
        return value + 1

    @observed_relation_phase("query")
    def outer(value):
        """嵌套真实 inner 调用，不制造空阶段。"""
        return inner(value)

    assert outer(3) == 4
    with capture_relation_phases(
            root / "success", run_identity="contract", attempt_key="success",
            case_ordinal=1, required_phases=("query", "core")) as phases:
        assert outer(3) == 4
    assert phases.report["status"] == "PASS"
    assert phases.report["positive_chain_phase_coverage"] == 0
    assert phases.report["case_phase_presence"]["generation"] == 0
    assert len(phases.report["streams"]) == 2
    with pytest.raises(LookupError, match="business failure"):
        with capture_relation_phases(
                root / "failure", run_identity="contract", attempt_key="failure",
                case_ordinal=1, required_phases=("query", "core")) as failed:
            outer(-1)
    assert failed.report["status"] == "FAIL"
    assert all(row["journal"]["status"] == "FAIL" for row in failed.report["streams"])


def test_passive_case_budget_cancels_on_completion_and_fails_mid_call(tmp_path):
    """自然退出取消唯一 waiter；超时子进程留 stop 与未闭合 BEGIN，返回 124。"""
    root = _k_path(tmp_path)
    with passive_relation_case_budget(root / "completed", max_wall_seconds=2):
        pass
    completion = json.loads((root / "completed/completion.json").read_bytes())
    assert completion["status"] == "PASS" and completion["process_poll_count"] == 0
    assert not (root / "completed/hard_budget_stop.json").exists()
    code = (
        "from pathlib import Path; from threading import Event; "
        "from pure_integer_ai.experiments.relation_case_budget import passive_relation_case_budget; "
        "from pure_integer_ai.experiments.relation_runtime_phase import capture_relation_phases,relation_phase; "
        "root=Path(__import__('os').environ['PIA_CASE_BUDGET_TEST_ROOT']); "
        "budget=passive_relation_case_budget(root/'stopped',max_wall_seconds=1); budget.__enter__(); "
        "scope=capture_relation_phases(root/'phases',run_identity='contract',attempt_key='one',"
        "case_ordinal=8,required_phases=('query',)); scope.__enter__(); "
        "phase=relation_phase('query','blocked-call'); phase.__enter__(); Event().wait(10)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "PIA_CASE_BUDGET_TEST_ROOT": str(root)},
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False)
    assert result.returncode == 124, result.stderr
    stop = json.loads((root / "stopped/hard_budget_stop.json").read_bytes())
    assert stop["process_poll_count"] == 0 and stop["restart_authorized"] == 0
    assert not (root / "stopped/completion.json").exists()
    boundary = json.loads((root / "phases/001-query/00000001.json").read_bytes())
    assert boundary["event"] == "BEGIN" and boundary["case_ordinal"] == 8
    assert not (root / "phases/001-query/00000002.json").exists()
