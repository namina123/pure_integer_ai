"""执行一次 W-02 compile/freeze，不启动 Candidate 或 private evaluator。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    compile_formal_w02_stage,
)


def main() -> int:
    """解析显式根路径，完成正式 W-02 pack 编译并打印安全摘要。"""
    parser = argparse.ArgumentParser(description="Compile and freeze PH2-D03-V2 W-02")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--formal-root", required=True, type=Path)
    args = parser.parse_args()
    result = compile_formal_w02_stage(
        args.repository_root, args.workspace_root, args.formal_root)
    print("W02_COMPILE_FREEZE_COMPLETE=1")
    print(f"W02_FREEZE_SHA256={result.freeze.sha256()}")
    print(f"W02_PACK_COMMITMENT={result.freeze.pack_commitment}")
    print(f"W02_TRAIN_OBSERVATIONS={result.freeze.plan.split_total('train')}")
    print(f"W02_TOTAL_OBSERVATIONS={result.freeze.plan.total_observations()}")
    print(f"WIKTIONARY_PAGES_SCANNED={result.wiktionary_pages_scanned}")
    print("FORMAL_TRAINING_RUNS=0")
    print("FORMAL_PRIVATE_EVALUATION_RUNS=0")
    print("PRIVATE_PAYLOAD_READS=0")
    print("TEACHER_CALLS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
