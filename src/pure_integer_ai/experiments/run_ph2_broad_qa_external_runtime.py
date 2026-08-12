"""执行标签隔离的外部证据预测或独立评分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_broad_qa_external_runtime import (
    predict_external_evidence,
    score_external_evidence,
)


def _work_path(value: str) -> Path:
    """要求正式外部评测输入输出均为显式 K 盘绝对路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("evaluation paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("evaluation paths must be on K:")
    return resolved


def _parser() -> argparse.ArgumentParser:
    """构造互相隔离的 predict/score 子命令。"""
    parser = argparse.ArgumentParser(
        description="Predict or score external evidence selection.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--questions", type=_work_path, required=True)
    predict.add_argument("--predictions", type=_work_path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--questions", type=_work_path, required=True)
    score.add_argument("--predictions", type=_work_path, required=True)
    score.add_argument("--labels", type=_work_path, required=True)
    score.add_argument("--aggregate", type=_work_path, required=True)
    score.add_argument(
        "--scope", choices=("DEVELOPMENT", "FORMAL_HELD_OUT"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行所选阶段，并输出不含逐题金答案的聚合 JSON。"""
    args = _parser().parse_args(argv)
    if args.command == "predict":
        report = predict_external_evidence(
            args.questions, predictions_path=args.predictions)
    else:
        report = score_external_evidence(
            args.questions, args.predictions, args.labels,
            aggregate_path=args.aggregate, scope=args.scope)
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
