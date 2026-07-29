"""生成正式中文 PH2 W-01 run bundle 和 self-excluded receipt。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_w01_contract import D03_GLOBAL_MANIFEST_PATH
from pure_integer_ai.experiments.ph2_w01_receipt import (
    W01_FORMAL_RECEIPT_PATH,
    W01_FORMAL_ROOT,
    build_w01_formal_receipt,
    read_w01_formal_receipt,
    write_w01_formal_receipt,
)
from pure_integer_ai.experiments.ph2_w01_report import run_directory
from pure_integer_ai.experiments.ph2_w01_runtime import (
    W01RuntimeConfig,
    run_language_stage0,
)


def _parser() -> argparse.ArgumentParser:
    """构造只接受仓库与独立 SQLite scratch 的正式生成参数。"""
    parser = argparse.ArgumentParser(description="生成正式 W-01 report/receipt")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """首次执行正式 worker=1 run；随后不可覆盖地生成并规范回读 receipt。"""
    args = _parser().parse_args(argv)
    repository = args.repository_root.resolve()
    scratch = args.scratch_root.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    formal_root = repository / W01_FORMAL_ROOT
    formal_run = run_directory(formal_root, 1)
    if not formal_run.is_dir():
        run_language_stage0(W01RuntimeConfig(
            repository_root=repository,
            global_manifest_path=D03_GLOBAL_MANIFEST_PATH,
            run_root=formal_root,
            sqlite_path=scratch / "w01-formal.sqlite3",
            run_id=1,
            parent_run_id=0,
            base_run_id=0,
            base_fence_key=(1, 0, 20260729),
            worker_count=1,
            mode="fresh",
        ))
    receipt = build_w01_formal_receipt(repository)
    target = repository / W01_FORMAL_RECEIPT_PATH
    write_w01_formal_receipt(receipt, target)
    restored = read_w01_formal_receipt(repository)
    print(json.dumps({
        "receipt_path": str(target),
        "receipt_sha256": restored.sha256(),
        "status": restored.to_dict()["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
