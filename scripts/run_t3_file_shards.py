"""运行可恢复的逐文件隔离 T3 分片。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.t3_shard_checkpoint import prepare_state  # noqa: E402
from scripts.t3_shard_contract import (  # noqa: E402
    T3ShardRunnerError,
    build_inventory,
    read_head,
    select_files,
)
from scripts.t3_shard_runner import run_state, summarize_state  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """定义分片、恢复和单次执行预算参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--start-at")
    parser.add_argument("--end-at")
    parser.add_argument("--file-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--plan-only", action="store_true")
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument(
        "--fail-fast",
        dest="continue_on_failure",
        action="store_false",
        help="任一文件不通过即停止，默认行为",
    )
    failure.add_argument(
        "--continue-on-failure",
        dest="continue_on_failure",
        action="store_true",
        help="记录失败后继续其余文件",
    )
    parser.set_defaults(continue_on_failure=False)
    return parser


def default_state_root(repository_root: Path, shard_count: int, shard_index: int) -> Path:
    """把默认证据目录放在公开 Git 根之外。"""
    head = read_head(repository_root)[:12]
    return (
        repository_root.resolve().parent
        / "t3_public_validation_artifacts"
        / head
        / f"file-shard-{shard_index:03d}-of-{shard_count:03d}"
    )


def plan(args: argparse.Namespace) -> dict[str, object]:
    """只读输出当前 inventory 和确定性分片选择，不创建 checkpoint。"""
    repository_root = args.repository_root.resolve()
    inventory = build_inventory(repository_root)
    selected = select_files(
        inventory,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        start_at=args.start_at,
        end_at=args.end_at,
    )
    return {
        "inventory_file_count": inventory["file_count"],
        "inventory_sha256": inventory["sha256"],
        "selected_file_count": len(selected),
        "first_selected_file": selected[0],
        "last_selected_file": selected[-1],
        "selected_files": list(selected),
    }


def main(argv: list[str] | None = None) -> int:
    """装配 CLI，执行或恢复一个分片并返回稳定退出码。"""
    args = build_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    if args.plan_only:
        print(json.dumps(plan(args), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    state_root = (
        args.state_root.resolve()
        if args.state_root is not None
        else default_state_root(repository_root, args.shard_count, args.shard_index)
    )
    try:
        state = prepare_state(
            repository_root,
            state_root,
            resume=args.resume,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            start_at=args.start_at,
            end_at=args.end_at,
            file_timeout_seconds=args.file_timeout_seconds,
            continue_on_failure=args.continue_on_failure,
        )
        state = run_state(
            repository_root,
            state_root,
            state,
            retry_failed=args.retry_failed,
            max_files=args.max_files,
        )
    except T3ShardRunnerError as error:
        print(f"T3_RUNNER_ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("T3_RUNNER_INTERRUPTED", file=sys.stderr)
        return 130
    summary = summarize_state(state)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    if summary["aggregate_status"] == "PASS":
        return 0
    if summary["aggregate_status"] == "FAIL":
        return 1
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
