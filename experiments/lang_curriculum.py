"""语言关系课程定义和显式关系范围规划。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import CurriculumVersion
from pure_integer_ai.training.stages import (
    STAGE1_SKELETON, STAGE2_CAUSES_ABS, STAGE3_REWARD,
    STAGE4_PROMOTE_WEAN,
)

LANGUAGE_BOOT_RELATIONS = frozenset({
    "alias", "is_a", "abstract", "similar", "mereology", "antonym",
    "causes", "sense", "number",
})


@dataclass(frozen=True)
class LanguageCurriculumStage:
    name: str
    add_relations: frozenset[str] = frozenset()
    load_all: bool = False
    result_field: str | None = None
    capability_kind: str = "reuse"
    training_stage: int = STAGE2_CAUSES_ABS

    def __post_init__(self) -> None:
        if self.load_all and self.add_relations:
            raise ValueError("load_all stages cannot also add a finite relation set")


@dataclass(frozen=True)
class LanguageCurriculumState:
    index: int
    stage: LanguageCurriculumStage
    active_relations: frozenset[str] | None
    boot_relations: frozenset[str]
    curriculum_version: CurriculumVersion


LANGUAGE_RELATION_CURRICULUM: tuple[LanguageCurriculumStage, ...] = (
    LanguageCurriculumStage(
        "T-L0  alias", frozenset({"alias"}),
        result_field="alias_edges_seeded", capability_kind="live",
        training_stage=STAGE1_SKELETON),
    LanguageCurriculumStage(
        "T-L1a is_a", frozenset({"is_a", "abstract"}),
        result_field="abstract_is_a_edges_seeded", capability_kind="derive"),
    LanguageCurriculumStage(
        "T-L1b similar", frozenset({"similar"}),
        result_field="similar_edges_seeded", capability_kind="derive"),
    LanguageCurriculumStage(
        "T-L1d mereology", frozenset({"mereology"}),
        result_field="mereology_edges_seeded", capability_kind="dead"),
    LanguageCurriculumStage(
        "T-L1e antonym", frozenset({"antonym"}),
        result_field="antonym_edges_seeded", capability_kind="dead"),
    LanguageCurriculumStage(
        "T-L2b causes", frozenset({"causes"}), capability_kind="correspondence",
        training_stage=STAGE3_REWARD),
    LanguageCurriculumStage(
        "T-L6d quant", capability_kind="reuse", training_stage=STAGE3_REWARD),
    LanguageCurriculumStage(
        "T-FULL all", load_all=True, capability_kind="all",
        training_stage=STAGE4_PROMOTE_WEAN),
)


def build_language_curriculum_plan(
        limit: int | None = None,
        stages: tuple[LanguageCurriculumStage, ...] = LANGUAGE_RELATION_CURRICULUM,
        curriculum_version: CurriculumVersion = CurriculumVersion(1),
        ) -> list[LanguageCurriculumState]:
    """解析累积有限范围和显式全关系状态。"""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    selected = stages if limit is None else stages[:limit]
    cumulative: frozenset[str] = frozenset()
    all_loaded = False
    plan: list[LanguageCurriculumState] = []
    for index, stage in enumerate(selected):
        if stage.load_all:
            boot_relations = LANGUAGE_BOOT_RELATIONS - cumulative
            all_loaded = True
        elif not all_loaded:
            boot_relations = stage.add_relations - cumulative
            cumulative = cumulative | stage.add_relations
        else:
            boot_relations = frozenset()
        active = None if all_loaded else cumulative
        plan.append(LanguageCurriculumState(
            index=index, stage=stage, active_relations=active,
            boot_relations=boot_relations,
            curriculum_version=curriculum_version))
    return plan
