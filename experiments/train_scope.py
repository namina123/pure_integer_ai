"""正式训练的阶段和关系范围解析。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


@dataclass(frozen=True)
class TrainScope:
    """一次正式训练调用的显式执行范围。"""

    training_stages: tuple[int, ...]
    active_relations: frozenset[str] | None
    boot_relations: frozenset[str] | None

    def relation_enabled(self, relation: str) -> bool:
        """判断本调用是否允许 boot 指定关系。"""
        return self.boot_relations is None or relation in self.boot_relations


def resolve_train_scope(*,
                        known_stages: tuple[int, ...],
                        requested_stages: tuple[int, ...] | None,
                        active_relations: frozenset[str] | None,
                        boot_relations: frozenset[str] | None) -> TrainScope:
    """校验阶段并解析课程 boot delta。"""
    stages = known_stages if requested_stages is None else requested_stages
    if len(set(stages)) != len(stages):
        raise ValueError("active_training_stages cannot contain duplicates")
    for stage in stages:
        assert_int(stage, _where="resolve_train_scope.training_stage")
        if stage not in known_stages:
            raise ValueError(
                f"active_training_stages contains unknown stage: {stage}")
    effective_boot = active_relations if boot_relations is None else boot_relations
    return TrainScope(
        training_stages=tuple(stages),
        active_relations=active_relations,
        boot_relations=effective_boot,
    )


__all__ = ["TrainScope", "resolve_train_scope"]
