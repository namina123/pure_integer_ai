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
from pure_integer_ai.experiments.ph2_w04_w05_source_bound_proposition import (
    project_w04_primitives_to_w05_source_bound_propositions,
    query_w04_w05_source_bound_propositions,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa import (
    answer_w05_raw_definition_question,
)
from pure_integer_ai.experiments.ph2_w05_definition_rendering import (
    render_w05_definition_answer,
)
from pure_integer_ai.experiments.ph2_w05_raw_definition_qa_contract import (
    W05RawDefinitionRequest,
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
        "--artifact-version",
        choices=("v1", "v2", "v3", "v4"),
        default="v1",
        help="explicit compact public-sense artifact version",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--primitive",
        action="store_true",
        help=(
            "project matching senses as source-asserted W-04 primitives; "
            "does not adjudicate them as facts"),
    )
    modes.add_argument(
        "--proposition",
        action="store_true",
        help=(
            "project matching primitives as source-bound W-05 typed "
            "propositions; authorizes only the source claim"),
    )
    modes.add_argument(
        "--definition",
        action="store_true",
        help=(
            "interpret surface as a raw definition question and answer only "
            "from a unique active source definition"),
    )
    modes.add_argument(
        "--display-definition",
        action="store_true",
        help=(
            "render an already unique source definition with explicit "
            "citation; unsupported or ambiguous markup fails closed"),
    )
    return parser


def main(
        argv: list[str] | None = None,
        *,
        stdout: TextIO | None = None,
        ) -> int:
    """加载一次 compact artifact，执行一次查询并输出规范 JSON。"""
    args = _parser().parse_args(argv)
    runtime = load_w03_public_sense_artifact(
        artifact_version=args.artifact_version)
    query = W03PublicSenseQuery(args.surface, args.context, args.language)
    if args.definition or args.display_definition:
        primitive_runtime = project_w03_public_sense_to_w04_primitives(runtime)
        proposition_runtime = (
            project_w04_primitives_to_w05_source_bound_propositions(
                primitive_runtime))
        result = answer_w05_raw_definition_question(
            proposition_runtime,
            W05RawDefinitionRequest(
                args.surface, args.context, args.language),
        )
        if args.display_definition:
            result = render_w05_definition_answer(result)
    elif args.proposition:
        primitive_runtime = project_w03_public_sense_to_w04_primitives(runtime)
        result = query_w04_w05_source_bound_propositions(
            project_w04_primitives_to_w05_source_bound_propositions(
                primitive_runtime),
            query,
        )
    elif args.primitive:
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
