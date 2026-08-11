"""安装后可用的 FT26 public term/短语查询入口。"""
from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from pure_integer_ai.experiments.ph2_w03_public_sense_contract import (
    W03PublicSenseQuery,
)
from pure_integer_ai.experiments.ph2_w03_public_sense_runtime import (
    load_w03_public_sense_artifact,
    query_w03_public_sense,
)
from pure_integer_ai.experiments.ph2_w03_w04_source_bound_primitive import (
    project_w03_public_sense_to_w04_primitives,
    query_w03_w04_source_bound_primitives,
)


def _parser() -> argparse.ArgumentParser:
    """构造不接触 formal/training 状态的只读 CLI 参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Query the experimental public source-bound term/sense artifact."),
    )
    parser.add_argument("surface", help="raw term or short phrase")
    parser.add_argument(
        "--context",
        default=None,
        help="optional exact learned definition/context",
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="base language or explicit language variant",
    )
    parser.add_argument(
        "--primitive",
        action="store_true",
        help=(
            "project matching senses as source-asserted W-04 primitives; "
            "does not adjudicate them as facts"),
    )
    return parser


def main(
        argv: list[str] | None = None,
        *,
        stdout: TextIO | None = None,
        ) -> int:
    """加载一次 compact artifact，执行一次查询并输出规范 JSON。"""
    args = _parser().parse_args(argv)
    runtime = load_w03_public_sense_artifact()
    query = W03PublicSenseQuery(
        args.surface, args.context, args.language)
    if args.primitive:
        result = query_w03_w04_source_bound_primitives(
            project_w03_public_sense_to_w04_primitives(runtime), query)
    else:
        result = query_w03_public_sense(runtime, query)
    stream = sys.stdout if stdout is None else stdout
    stream.write(json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    stream.write("\n")
    stream.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
