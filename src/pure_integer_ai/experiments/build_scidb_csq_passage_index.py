"""CSQ 来源知识段索引的 K 盘构建入口。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.scidb_csq_passage_index import (
    build_scidb_csq_passage_index,
)


def main(argv: list[str] | None = None) -> int:
    """解析显式来源承诺并打印单条规范构建结果。"""
    parser = argparse.ArgumentParser(
        description="build train-only SciDB CSQ source passage index")
    parser.add_argument("--course", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--course-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    arguments = parser.parse_args(argv)
    result = build_scidb_csq_passage_index(
        course_path=arguments.course,
        artifact_root=arguments.artifact_root,
        expected_course_sha256=arguments.course_sha256,
        expected_source_sha256=arguments.source_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
