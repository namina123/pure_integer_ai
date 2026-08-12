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
    select_external_source_pack,
)


_CMRC_REVISION = "c0eb1b6ba219847457e6af3180da722bbeb656af"
_DRCD_REVISION = "b944790de5af02c5fbb7cd9cb1473d27d169eebf"
_CMRC_FILES = (
    ("train", "data/cmrc2018_train.json",
     "d935a2d2c3ea8fef4b6b3d7f1c876f5870b720fc0c9caf4d14989f394a3d4745"),
    ("dev", "data/cmrc2018_dev.json",
     "5cfe4414c28a8ecbb51670f78c0dc7d1049f286c2d5769b52f1f94bcc0752cf1"),
    ("trial", "data/cmrc2018_trial.json",
     "a976d1fd5efc173bd58ff1c57e958de5f49fed633a7bfb8e0e402e5490d75f5e"),
)
_DRCD_FILES = (
    ("training", "DRCD_training.json",
     "5e6268091ab98f0bb858f03c77fba85e23b31e49a91638e4f22d0d8ec703f79a"),
    ("dev", "DRCD_dev.json",
     "e236df03861ba241bd7b29628cd48b3b8e339c60137e041a4358b5320ed61928"),
    ("test", "DRCD_test.json",
     "d9de0b4d247a8391ac8daee6c02c2c8272b75562a1e2377b50ae04ecc09a7438"),
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


def _sources(cmrc_root: Path, drcd_root: Path) -> tuple[ExternalQaSourceFile, ...]:
    """把冻结 commit、文件 hash 与许可绑定到本地官方 checkout。"""
    values = []
    for partition, relative, sha256 in _CMRC_FILES:
        values.append(ExternalQaSourceFile(
            "CMRC2018", partition, "CMRC2018", cmrc_root / relative,
            sha256, _CMRC_REVISION, "CC-BY-SA-4.0",
            "https://github.com/ymcui/cmrc2018",
        ))
    for partition, relative, sha256 in _DRCD_FILES:
        values.append(ExternalQaSourceFile(
            "DRCD", partition, "DRCD", drcd_root / relative,
            sha256, _DRCD_REVISION, "CC-BY-SA-3.0",
            "https://github.com/DRCKnowledgeTeam/DRCD",
        ))
    return tuple(values)


def main(argv: list[str] | None = None) -> int:
    """核验官方来源并冻结 200 dev/300 held-out，无系统运行。"""
    args = _parser().parse_args(argv)
    items, source_report = load_external_qa_sources(
        _sources(args.cmrc_root, args.drcd_root))
    selected = select_external_source_pack(items)
    report = freeze_external_source_pack(
        selected, target_dir=args.target, source_report=source_report)
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
