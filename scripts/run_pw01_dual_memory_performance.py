"""运行 PW-01 双 Memory 三档完整回答性能基线。"""
from __future__ import annotations

import argparse
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.pw01_dual_memory_performance import (
    PW01_PERFORMANCE_SCALES,
    run_pw01_dual_memory_scale_curve,
)


def _parser() -> argparse.ArgumentParser:
    """构造排他数据库和报告路径参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-index", action="store_true")
    return parser


def main() -> int:
    """执行预注册规模并以 create-new 语义发布基线报告。"""
    args = _parser().parse_args()
    database = args.database.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"PW-01 performance report 已存在: {output}")
    report = run_pw01_dual_memory_scale_curve(
        database,
        scales=PW01_PERFORMANCE_SCALES,
        use_query_index=args.query_index,
    )
    payload = canonical_json_bytes(report.as_dict())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
