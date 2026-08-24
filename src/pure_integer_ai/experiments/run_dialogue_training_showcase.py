"""公开对话训练与只读展示的一键编排入口。

该入口把公开 JSONL 课程的真实 formal_train 运行和已有广域问答展示串起来。
训练、SQLite 和展示摘要仍全部写入调用方指定的 K 盘 run；展示只读消费训练
状态和外部 QA 索引，不把检索结果或模板回放写回 Core/Runtime 学习账。

typed language generation 尚未由本入口隐式伪造：当调用方请求 Stage 3/4 而
没有 generation owner 时，formal_train 会按既有 gate 停止，随后仍可生成
只读展示，摘要会同时保留阻塞状态。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    build_dialogue_scale_showcase,
    write_dialogue_scale_showcase,
)
from pure_integer_ai.experiments.run_conversation_training import (
    run_conversation_training,
)


def _paths(values: Iterable[str | Path]) -> tuple[str, ...]:
    """把 CLI 多值稳定化为不可变字符串序列。"""
    return tuple(str(Path(value)) for value in values)


def run_training_and_showcase(
        *,
        project_root: str | Path,
        run_root: str | Path,
        run_id: str,
        qa_database: str | Path | None = None,
        active_stages: tuple[int, ...] = (1, 2, 3, 4),
        with_heldout_probe: bool = True,
        extra_course_paths: tuple[str | Path, ...] = (),
        extra_variant_course_paths: tuple[str | Path, ...] = (),
        extra_variant_evidence_paths: tuple[str | Path, ...] = (),
        extra_order_course_paths: tuple[str | Path, ...] = (),
        extra_order_evidence_paths: tuple[str | Path, ...] = (),
        ) -> dict[str, object]:
    """先执行真实公开训练，再按需建立只读中文对话展示。"""
    summary = run_conversation_training(
        project_root=project_root,
        run_root=run_root,
        run_id=run_id,
        active_stages=active_stages,
        with_heldout_probe=with_heldout_probe,
        extra_course_paths=extra_course_paths,
    )
    result: dict[str, object] = {
        "training": summary,
        "showcase": None,
    }
    if qa_database is None:
        return result
    database_path = Path(str(summary["database"])).resolve()
    training_run_root = database_path.parent
    showcase = build_dialogue_scale_showcase(
        project_root=project_root,
        database_path=qa_database,
        training_run_root=training_run_root,
        extra_training_course_paths=extra_course_paths,
        extra_variant_course_paths=extra_variant_course_paths,
        extra_variant_evidence_paths=extra_variant_evidence_paths,
        extra_order_course_paths=extra_order_course_paths,
        extra_order_evidence_paths=extra_order_evidence_paths,
    )
    output_path = training_run_root / "dialogue_showcase.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    write_dialogue_scale_showcase(showcase, output_path)
    result["showcase"] = {
        "output": str(output_path),
        "turn_count": len(showcase.get("turns", ())),
        "turns": showcase.get("turns", ()),
        "pack": showcase.get("pack"),
        "trained_surface_consumer": showcase.get("trained_surface_consumer"),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run public dialogue training and optional read-only showcase")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--qa-database", default=None,
                        help="可选的 K 盘广域 QA SQLite 索引")
    parser.add_argument("--stages", default="1,2,3,4")
    parser.add_argument("--without-heldout-probe", action="store_true")
    parser.add_argument("--extra-course", action="append", default=[])
    parser.add_argument("--variant-course", action="append", default=[])
    parser.add_argument("--variant-evidence", action="append", default=[])
    parser.add_argument("--order-course", action="append", default=[])
    parser.add_argument("--order-evidence", action="append", default=[])
    args = parser.parse_args(argv)
    value = run_training_and_showcase(
        project_root=args.project_root,
        run_root=args.run_root,
        run_id=args.run_id,
        qa_database=args.qa_database,
        active_stages=tuple(int(item) for item in args.stages.split(",")
                            if item),
        with_heldout_probe=not args.without_heldout_probe,
        extra_course_paths=_paths(args.extra_course),
        extra_variant_course_paths=_paths(args.variant_course),
        extra_variant_evidence_paths=_paths(args.variant_evidence),
        extra_order_course_paths=_paths(args.order_course),
        extra_order_evidence_paths=_paths(args.order_evidence),
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_training_and_showcase"]
