"""公开对话 pack 的 train/heldout/negative 大能力级对照摘要。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import argparse
import json
from pathlib import Path

from pure_integer_ai.experiments.conversation_training_pack import (
    DialogueTrainingPack,
)


def _sha_ids(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _surface_scalars(values: tuple[str, ...]) -> set[int]:
    return {ord(character) for value in values for character in value}


@dataclass(frozen=True, slots=True)
class DialogueTrainingContrast:
    """只描述 split 隔离与表层 novelty，不宣称问答正确率。"""

    train_case_count: int
    heldout_case_count: int
    negative_case_count: int
    train_case_id_sha256: str
    heldout_case_id_sha256: str
    negative_case_id_sha256: str
    heldout_id_overlap_count: int
    negative_id_overlap_count: int
    heldout_exact_surface_overlap_count: int
    negative_exact_surface_overlap_count: int
    heldout_novel_scalar_count: int
    negative_novel_scalar_count: int
    train_causal_case_count: int
    heldout_causal_case_count: int
    negative_causal_case_count: int

    def to_dict(self) -> dict[str, object]:
        """导出紧凑、可回读的整数 split 对照。"""
        return {
            "format_version": 1,
            "heldout_case_count": self.heldout_case_count,
            "heldout_causal_case_count": self.heldout_causal_case_count,
            "heldout_exact_surface_overlap_count": self.heldout_exact_surface_overlap_count,
            "heldout_id_overlap_count": self.heldout_id_overlap_count,
            "heldout_novel_scalar_count": self.heldout_novel_scalar_count,
            "heldout_case_id_sha256": self.heldout_case_id_sha256,
            "negative_case_count": self.negative_case_count,
            "negative_causal_case_count": self.negative_causal_case_count,
            "negative_exact_surface_overlap_count": self.negative_exact_surface_overlap_count,
            "negative_id_overlap_count": self.negative_id_overlap_count,
            "negative_novel_scalar_count": self.negative_novel_scalar_count,
            "negative_case_id_sha256": self.negative_case_id_sha256,
            "train_case_count": self.train_case_count,
            "train_causal_case_count": self.train_causal_case_count,
            "train_case_id_sha256": self.train_case_id_sha256,
        }


def build_dialogue_training_contrast(pack: DialogueTrainingPack) -> DialogueTrainingContrast:
    """从公开 pack 计算 split 对照；不读取答案标签，也不改训练状态。"""
    if type(pack) is not DialogueTrainingPack:
        raise TypeError("contrast 需要 DialogueTrainingPack")
    by_split = {
        split: tuple(item for item in pack.cases if item.split == split)
        for split in ("train", "heldout", "negative")
    }
    train = by_split["train"]
    heldout = by_split["heldout"]
    negative = by_split["negative"]
    train_ids = {item.case_id for item in train}
    train_surfaces = {item.raw_text for item in train}
    train_scalars = _surface_scalars(tuple(item.raw_text for item in train))
    heldout_scalars = _surface_scalars(tuple(item.raw_text for item in heldout))
    negative_scalars = _surface_scalars(tuple(item.raw_text for item in negative))
    return DialogueTrainingContrast(
        len(train), len(heldout), len(negative),
        _sha_ids(tuple(item.case_id for item in train)),
        _sha_ids(tuple(item.case_id for item in heldout)),
        _sha_ids(tuple(item.case_id for item in negative)),
        len(train_ids & {item.case_id for item in heldout}),
        len(train_ids & {item.case_id for item in negative}),
        sum(item.raw_text in train_surfaces for item in heldout),
        sum(item.raw_text in train_surfaces for item in negative),
        len(heldout_scalars - train_scalars),
        len(negative_scalars - train_scalars),
        sum(bool(item.causal_pairs) for item in train),
        sum(bool(item.causal_pairs) for item in heldout),
        sum(bool(item.causal_pairs) for item in negative),
    )


__all__ = ["DialogueTrainingContrast", "build_dialogue_training_contrast", "main"]


def main(argv: list[str] | None = None) -> int:
    """生成公开 pack 对照摘要，不创建训练或 SQLite 状态。"""
    parser = argparse.ArgumentParser(description="Dialogue split contrast")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from pure_integer_ai.experiments.run_conversation_training import default_course_paths
    from pure_integer_ai.experiments.conversation_training_pack import load_dialogue_training_pack
    pack = load_dialogue_training_pack(default_course_paths(args.project_root))
    output = Path(args.output).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_dialogue_training_contrast(pack).to_dict(),
                                 ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
