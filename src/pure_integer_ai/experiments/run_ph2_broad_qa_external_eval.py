"""冻结外部中文问答 source pack；正式评分由独立命令消费。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    ExternalQaSourceFile,
    freeze_external_source_pack,
    load_external_qa_sources,
    official_external_qa_sources,
    select_external_source_pack,
)




def _work_path(value: str) -> Path:
    """要求评测来源与产物使用显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("evaluation paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("evaluation paths must be on K:")
    return resolved


def _parser() -> argparse.ArgumentParser:
    """构造只包含不可覆盖 freeze 的命令行协议。"""
    parser = argparse.ArgumentParser(
        description="Freeze a label-isolated external Chinese QA pack.")
    parser.add_argument("--cmrc-root", type=_work_path, required=True)
    parser.add_argument("--drcd-root", type=_work_path, required=True)
    parser.add_argument("--target", type=_work_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """核验官方来源并冻结 200 dev/300 held-out，无系统运行。"""
    args = _parser().parse_args(argv)
    items, source_report = load_external_qa_sources(
        official_external_qa_sources(args.cmrc_root, args.drcd_root))
    selected = select_external_source_pack(items)
    report = freeze_external_source_pack(
        selected, target_dir=args.target, source_report=source_report)
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
