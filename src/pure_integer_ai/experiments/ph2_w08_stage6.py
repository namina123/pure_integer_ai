"""汇总 P3-Ia、开放生成与 LC-16 的薄 W08-06 facade。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_lc16 import W08LC16QualificationReceipt
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08OpenGenerationAblationReport,
    W08OpenGenerationAuditReceipt,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_contract import (
    W08P3IaAuditReceipt,
    W08P3IaStageAblationReport,
    W08P3IaSupportingAblationReport,
    W08_P3IA_COMPONENT_KEYS,
)


class W08Stage6Error(ValueError):
    """W08-06 aggregate 或 formal 前状态发生漂移。"""


@dataclass(frozen=True)
class W08Stage6AuditReceipt:
    p3ia_key: tuple[int, ...]
    open_generation_key: tuple[int, ...]
    lc16_key: tuple[int, ...]
    dimension_outcomes: tuple[tuple[str, str], ...]
    supporting_p3ia_components: tuple[str, ...]
    hard_conjuncts: tuple[tuple[str, str], ...]
    state: str
    W08_STARTED: int = 0
    formal_w08_training_runs: int = 0
    teacher_calls: int = 0
    memory_learning_writes: int = 0
    LANGUAGE_CAPABILITY_MASTERED: int = 0
    LANGUAGE_READINESS: int = 0
    W09_STARTED: int = 0
    OPEN_GENERATION: str = "NE_NOT_YET_EVALUABLE"

    def __post_init__(self) -> None:
        for name in ("p3ia_key", "open_generation_key", "lc16_key"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not value or any(
                type(item) is not int for item in value
            ):
                raise W08Stage6Error(f"W08-06 {name} is invalid")
        if tuple(key for key, _ in self.dimension_outcomes) != W08_DIMENSION_KEYS:
            raise W08Stage6Error("W08-06 dimension order drifted")
        if any(value != "PASS" for _, value in self.dimension_outcomes):
            raise W08Stage6Error("W08-06 dimension precondition did not pass")
        if self.supporting_p3ia_components != W08_P3IA_COMPONENT_KEYS:
            raise W08Stage6Error("W08-06 supporting P3-Ia ablations are incomplete")
        if self.hard_conjuncts != (
            ("OPEN_GENERATION", "PASS"),
            ("LC16_DISCOURSE_REFERENCE_GENERATION", "PASS"),
        ):
            raise W08Stage6Error("W08-06 hard conjuncts are incomplete")
        if self.state != "PASS":
            raise W08Stage6Error("W08-06 public bounded aggregate is not PASS")
        if (
            self.W08_STARTED,
            self.formal_w08_training_runs,
            self.teacher_calls,
            self.memory_learning_writes,
            self.LANGUAGE_CAPABILITY_MASTERED,
            self.LANGUAGE_READINESS,
            self.W09_STARTED,
        ) != (0, 0, 0, 0, 0, 0, 0):
            raise W08Stage6Error("W08-06 changed pre-formal execution state")
        if self.OPEN_GENERATION != "NE_NOT_YET_EVALUABLE":
            raise W08Stage6Error("W08-06 changed OPEN_GENERATION before formal run")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "p3ia": list(self.p3ia_key),
                "open_generation": list(self.open_generation_key),
                "lc16": list(self.lc16_key),
                "dimensions": dict(self.dimension_outcomes),
                "supporting_p3ia": list(self.supporting_p3ia_components),
                "hard_conjuncts": dict(self.hard_conjuncts),
                "state": self.state,
                "W08_STARTED": self.W08_STARTED,
                "formal_w08_training_runs": self.formal_w08_training_runs,
                "OPEN_GENERATION": self.OPEN_GENERATION,
            }
        )


class W08Stage6Facade:
    """在不改变 W08 formal 状态的前提下关闭公开 W08-06。"""

    def close(
        self,
        *,
        p3ia: W08P3IaAuditReceipt,
        p3ia_supporting_ablations: tuple[W08P3IaSupportingAblationReport, ...],
        p3ia_stage_ablation: W08P3IaStageAblationReport,
        open_generation: W08OpenGenerationAuditReceipt,
        open_generation_ablation: W08OpenGenerationAblationReport,
        lc16: W08LC16QualificationReceipt,
        dimension_outcomes: dict[str, str],
    ) -> W08Stage6AuditReceipt:
        if not isinstance(p3ia, W08P3IaAuditReceipt) or p3ia.state != "RESOLVED":
            raise W08Stage6Error("W08-06 requires resolved P3-Ia")
        if tuple(item.component_key for item in p3ia_supporting_ablations) != (
            W08_P3IA_COMPONENT_KEYS
        ):
            raise W08Stage6Error("W08-06 P3-Ia supporting ablations drifted")
        if not isinstance(p3ia_stage_ablation, W08P3IaStageAblationReport) or (
            p3ia_stage_ablation.affected_dimensions != ("W-08-P3IA",)
        ):
            raise W08Stage6Error("W08-06 P3-Ia stage ablation did not bear")
        if (
            not isinstance(open_generation, W08OpenGenerationAuditReceipt)
            or open_generation.state != "RESOLVED"
        ):
            raise W08Stage6Error("W08-06 requires resolved open generation")
        if (
            not isinstance(open_generation_ablation, W08OpenGenerationAblationReport)
            or open_generation_ablation.affected_layers != ("SURFACE_MORPHOLOGY",)
        ):
            raise W08Stage6Error("W08-06 template replay ablation did not bear")
        if not isinstance(lc16, W08LC16QualificationReceipt) or lc16.state != "PASS":
            raise W08Stage6Error("W08-06 LC-16 qualification did not pass")
        if set(dimension_outcomes) != set(W08_DIMENSION_KEYS):
            raise W08Stage6Error("W08-06 dimension inventory drifted")
        ordered = tuple((key, dimension_outcomes[key]) for key in W08_DIMENSION_KEYS)
        return W08Stage6AuditReceipt(
            p3ia.canonical_key(),
            open_generation.canonical_key(),
            lc16.canonical_key(),
            ordered,
            W08_P3IA_COMPONENT_KEYS,
            (
                ("OPEN_GENERATION", "PASS"),
                ("LC16_DISCOURSE_REFERENCE_GENERATION", "PASS"),
            ),
            "PASS",
        )


__all__ = [
    "W08Stage6AuditReceipt",
    "W08Stage6Error",
    "W08Stage6Facade",
]
