"""正式中文 PH2 W-01 阶段 0 的独立命令行入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pure_integer_ai.experiments.ph2_w01_contract import (
    D03_GLOBAL_MANIFEST_PATH,
    W01_ALLOWED_MODES,
)
from pure_integer_ai.experiments.ph2_w01_faults import (
    W01FaultPoint,
    W01InjectedFault,
)
from pure_integer_ai.experiments.ph2_w01_runtime import (
    W01RuntimeConfig,
    run_language_stage0,
)


def _integer_key(value: str) -> tuple[int, ...]:
    """把逗号分隔的严格整数解析为非空 base fence key。"""
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("base fence key 必须是逗号分隔整数") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("base fence key 必须非空且非负")
    return result


def _parser() -> argparse.ArgumentParser:
    """构造不暴露 corpus、teacher 或 evaluator path 的最小 CLI。"""
    parser = argparse.ArgumentParser(
        description="执行或恢复正式中文 PH2 W-01 阶段 0 协议")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--global-manifest-path",
        default=D03_GLOBAL_MANIFEST_PATH,
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sqlite-path", type=Path, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--parent-run-id", type=int, default=0)
    parser.add_argument("--base-run-id", type=int, default=0)
    parser.add_argument(
        "--base-fence-key", type=_integer_key, default=(1, 0, 20260729))
    parser.add_argument("--worker-count", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--mode", choices=W01_ALLOWED_MODES, default="fresh")
    parser.add_argument(
        "--fault-point", choices=W01FaultPoint.injectable_points())
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行独立 W-01 orchestrator，并输出不含私有数据的规范摘要。"""
    args = _parser().parse_args(argv)
    try:
        outcome = run_language_stage0(W01RuntimeConfig(
            repository_root=args.repository_root,
            global_manifest_path=args.global_manifest_path,
            run_root=args.run_root,
            sqlite_path=args.sqlite_path,
            run_id=args.run_id,
            parent_run_id=args.parent_run_id,
            base_run_id=args.base_run_id,
            base_fence_key=args.base_fence_key,
            worker_count=args.worker_count,
            mode=args.mode,
            fault_point=args.fault_point,
        ))
    except W01InjectedFault as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps({
        "artifact_digest": outcome.artifact_digest,
        "cursor_digest": outcome.cursor_digest,
        "logical_state_digest": outcome.logical_state_digest,
        "report_digest": outcome.report_digest,
        "run_manifest_path": str(outcome.run_manifest_path),
        "status": outcome.report["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
