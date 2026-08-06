"""JF2-04 逐文件隔离 T3 runner 的有界契约测试。"""
from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath
import subprocess

import pytest

from scripts.t3_shard_checkpoint import prepare_state, read_state, write_state
from scripts.t3_shard_contract import (
    T3ShardRunnerError,
    build_inventory,
    select_files,
)
from scripts.t3_shard_runner import run_state, summarize_state


def _git(repository: Path, *arguments: str) -> None:
    """在有界临时仓库执行测试所需的 Git 操作。"""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))


def _make_repository(root: Path, tests: dict[str, str]) -> Path:
    """建立只含极小测试文件的 clean Git 仓库。"""
    root.mkdir()
    tests_root = root / "tests"
    tests_root.mkdir()
    (root / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")
    for name, source in tests.items():
        (tests_root / name).write_text(source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "T3 Test")
    _git(root, "config", "user.email", "t3@example.invalid")
    _git(root, "add", ".gitignore", "tests")
    _git(root, "commit", "-q", "-m", "freeze tests")
    return root


def _prepare(
    repository: Path,
    state_root: Path,
    *,
    resume: bool = False,
    continue_on_failure: bool = False,
    carry_forward_from: tuple[Path, ...] = (),
):
    """用固定单分片参数建立或恢复测试 checkpoint。"""
    return prepare_state(
        repository,
        state_root,
        resume=resume,
        shard_count=1,
        shard_index=0,
        start_at=None,
        end_at=None,
        file_timeout_seconds=30,
        continue_on_failure=continue_on_failure,
        carry_forward_from=carry_forward_from,
    )


def test_inventory_selection_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    """相同 inventory 的两片选择稳定、不重叠且完整覆盖。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "test_c.py": "def test_c():\n    assert True\n",
            "test_a.py": "def test_a():\n    assert True\n",
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    inventory = build_inventory(repository)
    first = select_files(
        inventory,
        shard_count=2,
        shard_index=0,
        start_at=None,
        end_at=None,
    )
    second = select_files(
        inventory,
        shard_count=2,
        shard_index=1,
        start_at=None,
        end_at=None,
    )
    assert set(first).isdisjoint(second)
    assert sorted((*first, *second)) == [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
    ]


def test_each_file_gets_fresh_process_log_and_resumable_checkpoint(tmp_path: Path) -> None:
    """执行预算中断后只续跑 pending 文件，PASS 尝试和日志不变。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "test_a.py": "import os\ndef test_a():\n    assert os.environ['PYTHONHASHSEED'] == '0'\n",
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    state_root = tmp_path / "state"
    state = _prepare(repository, state_root)
    first = run_state(repository, state_root, state, retry_failed=False, max_files=1)
    first_attempt = first["results"]["tests/test_a.py"]["attempts"][0]
    assert summarize_state(first)["status_counts"] == {"PASS": 1, "PENDING": 1}
    assert "1 passed" in first_attempt["pytest_summary"]
    assert (state_root / first_attempt["log_path"]).is_file()

    resumed = _prepare(repository, state_root, resume=True)
    completed = run_state(
        repository,
        state_root,
        resumed,
        retry_failed=False,
        max_files=None,
    )
    assert summarize_state(completed)["aggregate_status"] == "PASS"
    assert completed["results"]["tests/test_a.py"]["attempts"] == [first_attempt]
    assert len(completed["results"]["tests/test_b.py"]["attempts"]) == 1
    assert read_state(state_root) == completed


def test_resume_rejects_head_or_inventory_drift(tmp_path: Path) -> None:
    """测试内容或 HEAD 改变后，旧 checkpoint 必须 fail closed。"""
    repository = _make_repository(
        tmp_path / "repo",
        {"test_a.py": "def test_a():\n    assert True\n"},
    )
    state_root = tmp_path / "state"
    _prepare(repository, state_root)
    (repository / "tests" / "test_a.py").write_text(
        "def test_a():\n    assert False\n",
        encoding="utf-8",
    )
    with pytest.raises(T3ShardRunnerError, match="clean"):
        _prepare(repository, state_root, resume=True)
    _git(repository, "add", "tests/test_a.py")
    _git(repository, "commit", "-q", "-m", "drift")
    with pytest.raises(T3ShardRunnerError, match="漂移"):
        _prepare(repository, state_root, resume=True)


def test_failure_is_checkpointed_and_fail_fast_leaves_suffix_pending(tmp_path: Path) -> None:
    """文件失败应留下正式汇总，并在 fail-fast 下保留后续 pending。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "test_a.py": "def test_a():\n    assert False\n",
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    state_root = tmp_path / "state"
    state = _prepare(repository, state_root)
    completed = run_state(
        repository,
        state_root,
        state,
        retry_failed=False,
        max_files=None,
    )
    summary = summarize_state(completed)
    assert summary["aggregate_status"] == "FAIL"
    assert summary["status_counts"] == {"FAIL": 1, "PENDING": 1}
    attempt = completed["results"]["tests/test_a.py"]["attempts"][0]
    assert attempt["return_code"] == 1
    assert "1 failed" in attempt["pytest_summary"]


def test_continue_on_failure_runs_remaining_files(tmp_path: Path) -> None:
    """continue 模式必须保留失败并继续取得后续文件的正式汇总。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "test_a.py": "def test_a():\n    assert False\n",
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    state_root = tmp_path / "state"
    state = _prepare(repository, state_root, continue_on_failure=True)
    completed = run_state(
        repository,
        state_root,
        state,
        retry_failed=False,
        max_files=None,
    )
    assert summarize_state(completed)["status_counts"] == {"FAIL": 1, "PASS": 1}


def test_checkpoint_is_canonical_and_must_stay_outside_repository(tmp_path: Path) -> None:
    """状态必须 canonical，且公开仓库内路径一律拒绝。"""
    repository = _make_repository(
        tmp_path / "repo",
        {"test_a.py": "def test_a():\n    assert True\n"},
    )
    with pytest.raises(T3ShardRunnerError, match="Git 根之外"):
        _prepare(repository, repository / ".t3-state")
    state_root = tmp_path / "state"
    _prepare(repository, state_root)
    payload = (state_root / "state.json").read_bytes()
    assert payload.endswith(b"\n")
    assert payload == (
        json.dumps(
            json.loads(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def test_resume_seals_abandoned_attempt_and_uses_a_new_log(tmp_path: Path) -> None:
    """强制中断留下的 RUNNING attempt 必须封存，恢复不得覆盖旧日志。"""
    repository = _make_repository(
        tmp_path / "repo",
        {"test_a.py": "def test_a():\n    assert True\n"},
    )
    state_root = tmp_path / "state"
    state = _prepare(repository, state_root)
    old_log = "logs/0000-abandoned-attempt-001.log"
    (state_root / "logs").mkdir()
    (state_root / old_log).write_text("abandoned\n", encoding="utf-8")
    state["results"]["tests/test_a.py"] = {
        "status": "RUNNING",
        "attempts": [
            {
                "status": "RUNNING",
                "attempt": 1,
                "started_at_utc": "2026-08-06T00:00:00+00:00",
                "log_path": old_log,
            }
        ],
    }
    state["aggregate_status"] = "RUNNING"
    write_state(state_root, state)

    resumed = _prepare(repository, state_root, resume=True)
    completed = run_state(
        repository,
        state_root,
        resumed,
        retry_failed=False,
        max_files=None,
    )
    attempts = completed["results"]["tests/test_a.py"]["attempts"]
    assert [item["status"] for item in attempts] == ["INTERRUPTED", "PASS"]
    assert attempts[0]["log_path"] == old_log
    assert attempts[1]["log_path"] != old_log
    assert (state_root / old_log).read_text(encoding="utf-8") == "abandoned\n"


def test_carry_forward_keeps_unchanged_pass_and_leaves_changed_file_pending(
    tmp_path: Path,
) -> None:
    """后继 HEAD 只改一个测试时，只继承另一个文件的 PASS provenance。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "test_a.py": "def test_a():\n    assert True\n",
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    source_root = tmp_path / "source-state"
    source = _prepare(repository, source_root)
    source = run_state(
        repository,
        source_root,
        source,
        retry_failed=False,
        max_files=None,
    )
    (repository / "tests" / "test_b.py").write_text(
        "def test_b():\n    assert False\n",
        encoding="utf-8",
    )
    _git(repository, "add", "tests/test_b.py")
    _git(repository, "commit", "-q", "-m", "change suffix")
    target_root = tmp_path / "target-state"
    target = _prepare(
        repository,
        target_root,
        carry_forward_from=(source_root,),
    )
    assert target["results"]["tests/test_a.py"]["status"] == "PASS"
    carried = target["results"]["tests/test_a.py"]["carried_pass"]
    assert carried["source_head"]
    assert carried["changed_path_count"] == 1
    assert len(carried["changed_paths_sha256"]) == 64
    assert len(carried["source_log_sha256"]) == 64
    assert "tests/test_b.py" not in target["results"]


def test_carry_forward_rejects_global_and_production_changes(tmp_path: Path) -> None:
    """conftest 或 src 改动必须使整个来源 fail closed。"""
    for changed_path in ("tests/nested/conftest.py", "src/package/runtime.py"):
        case = changed_path.replace("/", "-").replace(".", "-")
        repository = _make_repository(
            tmp_path / f"repo-{case}",
            {"test_a.py": "def test_a():\n    assert True\n"},
        )
        source_root = tmp_path / f"source-{case}"
        source = _prepare(repository, source_root)
        run_state(repository, source_root, source, retry_failed=False, max_files=None)
        target = repository / Path(*PurePosixPath(changed_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
        _git(repository, "add", changed_path)
        _git(repository, "commit", "-q", "-m", "global drift")
        with pytest.raises(T3ShardRunnerError, match="全局/生产"):
            _prepare(
                repository,
                tmp_path / f"target-{case}",
                carry_forward_from=(source_root,),
            )


def test_carry_forward_tracks_recursive_test_dependencies(tmp_path: Path) -> None:
    """递归 tests.* 辅助模块改变时，依赖它的 PASS 不得继承。"""
    repository = _make_repository(
        tmp_path / "repo",
        {
            "helper.py": "VALUE = True\n",
            "middle.py": "from tests.helper import VALUE\n",
            "test_a.py": (
                "from tests.middle import VALUE\n"
                "def test_a():\n"
                "    assert VALUE\n"
            ),
            "test_b.py": "def test_b():\n    assert True\n",
        },
    )
    source_root = tmp_path / "source-state"
    source = _prepare(repository, source_root)
    run_state(repository, source_root, source, retry_failed=False, max_files=None)
    (repository / "tests" / "helper.py").write_text("VALUE = False\n", encoding="utf-8")
    _git(repository, "add", "tests/helper.py")
    _git(repository, "commit", "-q", "-m", "dependency drift")
    target = _prepare(
        repository,
        tmp_path / "target-state",
        carry_forward_from=(source_root,),
    )
    assert "tests/test_a.py" not in target["results"]
    assert target["results"]["tests/test_b.py"]["status"] == "PASS"


def test_carry_forward_rejects_non_ancestor_and_log_escape(tmp_path: Path) -> None:
    """分叉来源与逃逸日志路径均不得成为 PASS provenance。"""
    repository = _make_repository(
        tmp_path / "repo",
        {"test_a.py": "def test_a():\n    assert True\n"},
    )
    (repository / "marker.txt").write_text("source\n", encoding="utf-8")
    _git(repository, "add", "marker.txt")
    _git(repository, "commit", "-q", "-m", "source branch")
    source_root = tmp_path / "source-state"
    source = _prepare(repository, source_root)
    source = run_state(
        repository, source_root, source, retry_failed=False, max_files=None
    )

    escaped = source["results"]["tests/test_a.py"]["attempts"][-1]
    escaped["log_path"] = "logs/../../state.json"
    write_state(source_root, source)
    (repository / "tests" / "helper.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    _git(repository, "add", "tests/helper.py")
    _git(repository, "commit", "-q", "-m", "successor")
    with pytest.raises(T3ShardRunnerError, match="日志路径无效"):
        _prepare(
            repository,
            tmp_path / "escape-target",
            carry_forward_from=(source_root,),
        )

    _git(repository, "checkout", "-q", "HEAD~2")
    (repository / "branch.txt").write_text("other\n", encoding="utf-8")
    _git(repository, "add", "branch.txt")
    _git(repository, "commit", "-q", "-m", "other branch")
    with pytest.raises(T3ShardRunnerError, match="不是当前 HEAD 的祖先"):
        _prepare(
            repository,
            tmp_path / "branch-target",
            carry_forward_from=(source_root,),
        )


def test_resume_rejects_carry_forward_sources(tmp_path: Path) -> None:
    """恢复与创建期继承不能混用，以免静默忽略调用者参数。"""
    repository = _make_repository(
        tmp_path / "repo",
        {"test_a.py": "def test_a():\n    assert True\n"},
    )
    state_root = tmp_path / "state"
    _prepare(repository, state_root)
    with pytest.raises(T3ShardRunnerError, match="不得同时"):
        _prepare(
            repository,
            state_root,
            resume=True,
            carry_forward_from=(tmp_path / "source",),
        )
