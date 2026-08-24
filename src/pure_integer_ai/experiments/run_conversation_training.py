"""运行公开对话 pack 的第一条真实 formal_train 切片。

大输入与 checkpoint 只写显式 K 盘 run root。该入口默认只跑 observe/skeleton
阶段，先让真实 dialogue case 改变图状态；后续 reward 阶段由独立课程切片开启。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_integer_ai.config import gates
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_training_contrast import (
    build_dialogue_training_contrast,
)
from pure_integer_ai.experiments.formal_train import FormalTrainConfig, formal_train
from pure_integer_ai.storage.backend import SQLiteBackend


def default_course_paths(project_root: str | Path) -> tuple[Path, ...]:
    """返回当前仓库登记的公开 authored/dialogue 课程文件。"""
    root = Path(project_root).resolve() / "data" / "ph2"
    paths = sorted(root.glob("authored_*.jsonl.sample"))
    paths.extend(sorted(root.glob("dlg_raw_public_*course_v1.jsonl.sample")))
    paths.extend(sorted(root.glob("lc16_*_carrier_v1.jsonl.sample")))
    surface = root / "dlg_raw16_surface_organization_v1.jsonl.sample"
    if surface.is_file():
        paths.append(surface)
    return tuple(paths)


def _write_json(path: Path, value: object) -> None:
    """以单次创建写入紧凑、可回读的运行摘要。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n", encoding="utf-8")


def run_conversation_training(*, project_root: str | Path,
                              run_root: str | Path,
                              run_id: str = "dialogue-pack-v1",
                              active_stages: tuple[int, ...] = (1,),
                              resume_from: str | None = None,
                              with_heldout_probe: bool = False,
                              causal_only: bool = False,
                              extra_course_paths: tuple[str | Path, ...] = (),
                              ) -> dict[str, object]:
    """消费公开 train split，并产出真实 SQLite graph/checkpoint 摘要。"""
    root = Path(run_root).resolve()
    if root.drive.upper() != "K:" or not root.is_dir():
        raise ValueError("run_root 必须是已存在的 K 盘目录")
    paths = (*default_course_paths(project_root), *tuple(
        Path(item).resolve() for item in extra_course_paths))
    if len(paths) != len(set(paths)):
        raise ValueError("extra course path 与默认课程重复")
    pack = load_dialogue_training_pack(paths)
    train_items = pack.training_items(causal_only=causal_only)
    heldout_items = pack.training_items(split="heldout", causal_only=causal_only)
    contrast = build_dialogue_training_contrast(pack)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "dialogue_pack_manifest.json", {
        "protocol": 1,
        "pack_sha256": pack.pack_sha256,
        "source_files": pack.source_files,
        "case_count": len(pack.cases),
        "split_counts": pack.split_counts,
        "train_surface_count": len(train_items),
        "heldout_surface_count": len(heldout_items),
        "causal_only": causal_only,
        "extra_course_paths": tuple(Path(item).resolve().as_posix()
                                     for item in extra_course_paths),
    })
    _write_json(run_dir / "contrast_report.json", contrast.to_dict())
    database_path = run_dir / "training.sqlite3"
    backend = SQLiteBackend(str(database_path))
    corpus = train_items + heldout_items if with_heldout_probe else train_items
    previous = gates.TRAINING_MODE
    try:
        gates.TRAINING_MODE = True
        result = formal_train(
            FormalTrainConfig(
                run_dir=str(root),
                run_id=run_id,
                rounds_per_stage=1,
                active_training_stages=active_stages,
                resume=resume_from is not None,
                base_run_id=resume_from,
                probe_holdout=len(heldout_items) if with_heldout_probe else 0,
                probe_version=1 if with_heldout_probe else 0,
                persist_graph_dump=True,
            ),
            corpus,
            backend=backend,
        )
    finally:
        gates.TRAINING_MODE = previous
        backend.commit()
        backend.close()
    summary = {
        "run_id": run_id,
        "pack_sha256": pack.pack_sha256,
        "case_count": len(pack.cases),
        "split_counts": pack.split_counts,
        "training_item_count": len(train_items),
        "heldout_probe_count": len(heldout_items) if with_heldout_probe else 0,
        "causal_only": causal_only,
        "lang_generalization": None if result.lang_generalization is None else {
            "total_held_out": result.lang_generalization.total_held_out,
            "recognized": result.lang_generalization.recognized,
            "verified": result.lang_generalization.verified,
            "lang_rate_permille": result.lang_generalization.lang_rate_permille,
        },
        "active_stages": active_stages,
        "resume_from": resume_from,
        "stages_completed": tuple(result.stages_completed),
        "weaning_ready": bool(result.weaning_ready),
        "weaning_blockers": tuple(result.weaning_blockers),
        "database": str(database_path),
    }
    _write_json(run_dir / "training_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run public dialogue training slice")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="dialogue-pack-v1")
    parser.add_argument("--stages", default="1",
                        help="comma-separated formal_train stages")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--with-heldout-probe", action="store_true")
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--extra-course", action="append", default=[],
                        help="可选的公开课程 JSONL；不改变默认 v6 pack")
    args = parser.parse_args(argv)
    summary = run_conversation_training(
        project_root=args.project_root,
        run_root=args.run_root,
        run_id=args.run_id,
        active_stages=tuple(int(item) for item in args.stages.split(",") if item),
        resume_from=args.resume_from,
        with_heldout_probe=args.with_heldout_probe,
        causal_only=args.causal_only,
        extra_course_paths=tuple(args.extra_course),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["default_course_paths", "main", "run_conversation_training"]
