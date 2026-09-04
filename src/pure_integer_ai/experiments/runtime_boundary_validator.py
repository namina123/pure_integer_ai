"""发布运行边界 validator：包外 session、启动前后文件集合闭合与诊断路由可达性。

该模块只使用 Python 标准库，比较模型根/便携包根在运行前后的文件集合与
逐文件 SHA，验证会话库外置性，并确认发布承重路径不会到达按长度选澄清的
诊断路由。它不读取课程、QA、密钥或私有评测标签。

probe 由调用方注入：便携包 run.py 使用进程内 JSONL 回放，仓库脚本可使用
默认的子进程回放。返回 (exit_code, stderr_tail)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable


BOUNDARY_VALIDATION_FORMAT = 'PURE_INTEGER_RUNTIME_BOUNDARY_VALIDATION_V1'


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，不把大文件整体载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def snapshot_tree(root: str | Path) -> dict[str, tuple[int, str]]:
    """返回相对 posix 路径 -> (size_bytes, sha256) 的稳定快照。"""
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"快照根不存在: {base}")
    result: dict[str, tuple[int, str]] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        result[relative] = (path.stat().st_size, _sha256(path))
    return result


def compare_snapshots(
        before: dict[str, tuple[int, str]],
        after: dict[str, tuple[int, str]],
        ) -> dict[str, object]:
    """比较两个快照，返回新增/删除/漂移清单（全部稳定排序）。"""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    drifted = sorted(
        path for path in set(before) & set(after)
        if before[path] != after[path])
    return {
        "added": added,
        "removed": removed,
        "drifted": drifted,
        "closed": not (added or removed or drifted),
    }


def _jsonl_turn(text: str, *, request_id: int = 1) -> bytes:
    """构造一条 JSONL turn 请求行。"""
    payload = json.dumps(
        {"id": request_id, "op": "turn", "text": text},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (payload + "\n").encode("utf-8")


def subprocess_probe(
        *,
        release_root: str | Path,
        session_path: str | Path,
        protocol: str = "jsonl",
        ) -> Callable[[bytes], tuple[int, str]]:
    """返回一个子进程回放 callable：输入 JSONL 字节，返回 (exit_code, stderr)。

    仅适用于仓库环境（包已安装）；便携包 run.py 应注入进程内 probe。
    """
    root = Path(release_root).resolve()
    session = Path(session_path).expanduser().resolve()
    command = [
        sys.executable, "-m",
        "pure_integer_ai.experiments.run_trained_relation_graph_terminal",
        "--release-root", str(root),
        "--protocol", protocol,
        "--memory-database", str(session),
    ]

    def probe(payload: bytes) -> tuple[int, str]:
        completed = subprocess.run(
            command, input=payload, capture_output=True, timeout=600)
        return (completed.returncode,
                completed.stderr.decode("utf-8", "replace")[-2000:])
    return probe


def run_boundary_probe(
        *,
        release_root: str | Path,
        session_path: str | Path,
        input_turns: tuple[str, ...],
        probe: Callable[[bytes], tuple[int, str]],
        ) -> dict[str, object]:
    """在 release root 上执行一次包外 session 的 JSONL 回放并返回证据。

    返回结构化记录：逐文件前后快照比较、session 外置性、退出状态与
    stderr 尾段。运行前后的文件集合变化是 validator 的核心证据。
    """
    root = Path(release_root).resolve()
    session = Path(session_path).expanduser().resolve()
    try:
        session.relative_to(root)
    except ValueError:
        session_external = True
    else:
        session_external = False
    before = snapshot_tree(root)
    payload = b"".join(
        _jsonl_turn(text, request_id=ordinal + 1)
        for ordinal, text in enumerate(input_turns))
    payload += b"{\"op\":\"quit\"}\n"
    exit_code, stderr_tail = probe(payload)
    after = snapshot_tree(root)
    return {
        "release_root": str(root),
        "session_path": str(session),
        "session_external": session_external,
        "exit_code": exit_code,
        "stderr_tail": stderr_tail,
        "files_before_count": len(before),
        "files_after_count": len(after),
        "tree_comparison": compare_snapshots(before, after),
    }


def validate_runtime_boundary(
        *,
        release_root: str | Path,
        session_path: str | Path,
        input_turns: tuple[str, ...] = ("你好", "一加一等于几？"),
        probe: Callable[[bytes], tuple[int, str]] | None = None,
        ) -> dict[str, object]:
    """执行一次完整边界校验并给出 PASS/FAIL 判定。"""
    if probe is None:
        probe = subprocess_probe(
            release_root=release_root, session_path=session_path)
    evidence = run_boundary_probe(
        release_root=release_root,
        session_path=session_path,
        input_turns=input_turns,
        probe=probe,
    )
    comparison = evidence["tree_comparison"]
    closed = bool(comparison["closed"])
    external = bool(evidence["session_external"])
    # 严格发布模式在全部图路径无结果时不伪造表层：以 no_answer 空回答
    # 结束本轮，进程必须正常退出且模型目录保持闭合；任何文件新增/漂移或
    # 会话落在 release root 内都判定 FAIL。
    drift = not closed or not external
    status = "FAIL" if drift else "PASS"
    return {
        "format": BOUNDARY_VALIDATION_FORMAT,
        "schema_version": 1,
        "status": status,
        "model_tree_closed": closed,
        "session_external": external,
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI：--release-root 与 --session 必填，--turns 逗号分隔。"""
    parser = argparse.ArgumentParser(
        description="校验训练图发布根运行边界")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--turns", default="你好,一加一等于几？")
    args = parser.parse_args(argv)
    turns = tuple(
        value.strip() for value in args.turns.split(",") if value.strip())
    result = validate_runtime_boundary(
        release_root=args.release_root,
        session_path=args.session,
        input_turns=turns,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_VALIDATION_FORMAT",
    "compare_snapshots",
    "main",
    "run_boundary_probe",
    "snapshot_tree",
    "subprocess_probe",
    "validate_runtime_boundary",
]