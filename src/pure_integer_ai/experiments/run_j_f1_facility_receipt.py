"""临时生成、严格回读并首次发布公开 J-F1 facility receipt。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from pure_integer_ai.experiments.j_f1_facility_receipt import (
    JF1ReceiptError,
    J_F1_RECEIPT_RELATIVE_PATH,
    build_j_f1_facility_receipt,
    publish_j_f1_facility_receipt,
    read_j_f1_facility_receipt,
    write_j_f1_facility_receipt,
)


def stage_and_publish_j_f1_receipt(repository_root: str | Path) -> tuple[Path, str]:
    """先在系统临时目录闭合 canonical 回读，再排他发布正式目标。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(J_F1_RECEIPT_RELATIVE_PATH).parts)
    if target.exists():
        raise JF1ReceiptError("J-F1 正式 receipt 已存在，拒绝重复发布")
    receipt = build_j_f1_facility_receipt(repository)
    with TemporaryDirectory(prefix="j-f1-receipt-stage-") as temporary:
        staged = Path(temporary) / "j_f1_facility_receipt_v1.json"
        write_j_f1_facility_receipt(receipt, staged)
        restored = read_j_f1_facility_receipt(
            repository, receipt_path=staged, verify_runtime=False)
        if (restored != receipt
                or staged.read_bytes() != receipt.canonical_bytes()):
            raise JF1ReceiptError("J-F1 临时 receipt 回读不闭合")
    published = publish_j_f1_facility_receipt(repository, receipt)
    final = read_j_f1_facility_receipt(repository, verify_runtime=False)
    if final != receipt or published.read_bytes() != receipt.canonical_bytes():
        raise JF1ReceiptError("J-F1 正式 receipt 发布后回读不闭合")
    return published, final.sha256()


def _parser() -> argparse.ArgumentParser:
    """构造只接受公开仓库根的首次发布命令。"""
    parser = argparse.ArgumentParser(description="首次发布 J-F1 facility receipt")
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行临时闭合和 append-only 发布，并打印公开 identity。"""
    args = _parser().parse_args(argv)
    target, digest = stage_and_publish_j_f1_receipt(args.repository_root)
    print(json.dumps({
        "receipt_path": target.as_posix(),
        "receipt_sha256": digest,
        "status": "FACILITY_EVIDENCED",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["stage_and_publish_j_f1_receipt"]
