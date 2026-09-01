"""把公开 authored relation pack 接入正式对话训练图。\n\n+JSONL 仍只是课程交换格式；本模块在训练启动时把已声明的 W-06 relation
records 适配为共享 H-05/R-00 owner，并将 owner 绑定到 formal_train 当前的
TrainContext。发布运行时不读取这里的课程文件。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06TypedAdapterOutput,
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload


_COURSE_BUILDERS = {
    "authored_relation_alias_refers_w06_seed_v2.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_alias_refers_w06_course",
        "compile_authored_alias_refers_w06_course",
    ),
    "authored_relation_subset_member_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_subset_member_course",
        "compile_authored_subset_member_course",
    ),
    "authored_relation_property_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_property_course",
        "compile_authored_property_course",
    ),
    "authored_relation_mereology_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_mereology_course",
        "compile_authored_mereology_course",
    ),
    "authored_relation_similar_antonym_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_semantic_pair_course",
        "compile_authored_semantic_pair_course",
    ),
    "authored_relation_precedes_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_precedes_course",
        "compile_authored_precedes_course",
    ),
    "authored_relation_causes_seed_v1.jsonl.sample": (
        "pure_integer_ai.experiments.ph2_authored_causes_course",
        "compile_authored_causes_course",
    ),
}


def _course_builder(path: Path):
    """按已登记文件名加载对应 authored parser/compiler。"""
    import importlib

    entry = _COURSE_BUILDERS.get(path.name)
    if entry is None:
        return None
    module = importlib.import_module(entry[0])
    return getattr(module, entry[1])


def build_authored_w06_adapter(
        course_paths: Iterable[str | Path],
        pack_root: str | Path,
        ) -> W06TypedAdapterOutput:
    """编译公开 relation samples 并形成统一 W-06 typed adapter output。"""
    paths = tuple(sorted(Path(item).resolve() for item in course_paths))
    selected = tuple(path for path in paths if _course_builder(path) is not None)
    if not selected:
        raise ValueError("未找到已登记 authored relation sample")
    output_root = Path(pack_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_refs = []
    observations = []
    teacher_evidence = []
    for ordinal, path in enumerate(selected, start=1):
        builder = _course_builder(path)
        assert builder is not None
        build = builder(path, output_root / f"pack-{ordinal:02d}")
        manifest = read_artifact_manifest(build.pack_root / "manifest.json")
        for identity in manifest.files:
            records = read_record_artifact(build.pack_root, identity)
            if identity.record_kind == RECORD_SOURCE_REF:
                source_refs.extend(records)
            elif identity.record_kind == RECORD_OBSERVATION:
                observations.extend(
                    item for item in records if item.split == "train")
            elif identity.record_kind == RECORD_TEACHER_EVIDENCE:
                teacher_evidence.extend(records)
    return adapt_w06_training_payload(W06TrainingPayload(
        tuple(source_refs), tuple(observations), tuple(teacher_evidence)))


def build_authored_w06_learning_runtime(
        backend,
        context,
        course_paths: Iterable[str | Path],
        pack_root: str | Path,
        ) -> W06RelationLearningRuntime:
    """在指定正式 TrainContext 上消费 authored relation，返回共享 owner。"""
    adapter = build_authored_w06_adapter(course_paths, pack_root)
    return build_w06_learning_runtime(backend, adapter, context=context)


__all__ = [
    "build_authored_w06_adapter",
    "build_authored_w06_learning_runtime",
]
