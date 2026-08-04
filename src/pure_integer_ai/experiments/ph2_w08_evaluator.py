"""W08-09 五维、五消融、开放生成和 LC-16 private evaluator 核心。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    EvaluatorLabelRecord,
    ObservationRecord,
)
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CARRIER_KEYS,
    W08_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    W08PrivateDimensionResult,
    W08PrivateEvaluationError,
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w08_lc16 import W08_LC16_SCOPE_KEY
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08_OPEN_GENERATION_COVERAGE_KEYS,
    W08_OPEN_GENERATION_LAYER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import (
    W08_RUNTIME_HARD_CONJUNCT_KEYS,
    W08RunOutcome,
)


@dataclass(frozen=True, order=True)
class W08PrivateEvaluationPair:
    """一个已在 private phase 配对的 held-out Observation 与 evaluator label。"""

    pack_key: str
    observation: ObservationRecord
    label: EvaluatorLabelRecord

    def __post_init__(self) -> None:
        if not self.pack_key:
            raise W08PrivateEvaluationError("W08 private pair pack key 为空")
        if self.observation.split != "held_out":
            raise W08PrivateEvaluationError("W08 private pair 不是 held-out")
        if (
            self.label.observation_key != self.observation.stable_key
            or self.label.owner_mode != "read_only"
        ):
            raise W08PrivateEvaluationError("W08 private pair label 引用或 owner 漂移")


@dataclass(frozen=True, order=True)
class W08PrivateCaseOutcome:
    """Candidate 对一个 private case 的实际输出及 U/R/G/shortcut 账。"""

    dimension_key: str
    observation_key: tuple[int, ...]
    actual_state: str
    actual_payload_sha256: str
    consumer_states: tuple[tuple[str, str], ...]
    shortcut_counts: tuple[tuple[str, int], ...]
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dimension_key not in W08_DIMENSION_KEYS:
            raise W08PrivateEvaluationError("W08 private case outcome 维度非法")
        for value in (self.observation_key, self.outcome_key):
            if not value or any(type(item) is not int for item in value):
                raise W08PrivateEvaluationError("W08 private case outcome key 非法")
        if self.actual_state not in EXPECTED_STATES:
            raise W08PrivateEvaluationError("W08 private case outcome 状态非法")
        if (
            len(self.actual_payload_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in self.actual_payload_sha256
            )
        ):
            raise W08PrivateEvaluationError("W08 private case payload commitment 非法")
        if tuple(key for key, _ in self.consumer_states) != W08_CONSUMER_KEYS:
            raise W08PrivateEvaluationError("W08 private case U/R/G inventory 漂移")
        expected_shortcuts = (
            "exact_surface_reads",
            "fifo_or_recency_choices",
            "full_recompute_runs",
            "preloaded_hot_records",
            "w09_future_reads",
        )
        if (
            tuple(key for key, _ in self.shortcut_counts) != expected_shortcuts
            or any(
                type(value) is not int or value < 0
                for _, value in self.shortcut_counts
            )
        ):
            raise W08PrivateEvaluationError("W08 private case shortcut account 非法")


@dataclass(frozen=True)
class W08EvaluatorSnapshot:
    """从 Candidate canonical dump 恢复的只读承重能力摘要。"""

    dimension_artifact_keys: tuple[tuple[int, ...], ...]
    use_cells: tuple[tuple[str, str, str], ...]
    hard_conjunct_states: tuple[tuple[str, str], ...]
    semantic_state_key: tuple[int, ...]
    dump_manifest_sha256: str
    future_payload_reads: int
    evaluator_label_reads: int
    host_learning_writes: int
    memory_learning_writes: int

    def __post_init__(self) -> None:
        if (
            len(self.dimension_artifact_keys) != len(W08_DIMENSION_KEYS)
            or any(not item for item in self.dimension_artifact_keys)
        ):
            raise W08PrivateEvaluationError("W08 Candidate 五维 artifact 不完整")
        expected_cells = tuple(
            (dimension, consumer, "RESOLVED")
            for dimension in W08_DIMENSION_KEYS
            for consumer in W08_CONSUMER_KEYS
        )
        if self.use_cells != expected_cells:
            raise W08PrivateEvaluationError("W08 Candidate U/R/G Use 不完整")
        if self.hard_conjunct_states != tuple(
            (key, "PUBLIC_BOUNDED_PASS") for key in W08_RUNTIME_HARD_CONJUNCT_KEYS
        ):
            raise W08PrivateEvaluationError("W08 Candidate hard conjunct 不完整")
        if (
            not self.semantic_state_key
            or len(self.dump_manifest_sha256) != 64
            or any(
                (
                    self.future_payload_reads,
                    self.evaluator_label_reads,
                    self.host_learning_writes,
                    self.memory_learning_writes,
                )
            )
        ):
            raise W08PrivateEvaluationError("W08 Candidate snapshot 越过禁止边界")


def snapshot_from_w08_outcome(outcome: W08RunOutcome) -> W08EvaluatorSnapshot:
    if not isinstance(outcome, W08RunOutcome):
        raise TypeError("W08 Candidate dump outcome 类型非法")
    return W08EvaluatorSnapshot(
        tuple(item.artifact_key for item in outcome.artifacts),
        tuple(
            (item.dimension_key, item.consumer_key, item.outcome_state)
            for item in outcome.uses
        ),
        tuple((item.conjunct_key, item.state) for item in outcome.hard_conjuncts),
        outcome.semantic_state_key,
        outcome.dump_manifest_sha256,
        outcome.future_payload_reads,
        outcome.evaluator_label_reads,
        outcome.host_learning_writes,
        outcome.memory_learning_writes,
    )


def evaluate_w08_private_pairs(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    ablation_key: str | None = None,
    case_outcomes: tuple[W08PrivateCaseOutcome, ...] = (),
) -> tuple[W08PrivateDimensionResult, ...]:
    """只按实际 private 输出评定五维；缺输出是 NE，不能按 artifact 存在判 PASS。"""
    if not isinstance(snapshot, W08EvaluatorSnapshot):
        raise TypeError("W08 evaluator snapshot 类型非法")
    if (
        not isinstance(pairs, tuple)
        or not pairs
        or any(not isinstance(item, W08PrivateEvaluationPair) for item in pairs)
    ):
        raise W08PrivateEvaluationError("W08 private case family 为空或非法")
    if len({item.observation.stable_key for item in pairs}) != len(pairs):
        raise W08PrivateEvaluationError("W08 private Observation 重复")
    if len({item.label.stable_key for item in pairs}) != len(pairs):
        raise W08PrivateEvaluationError("W08 private label 重复")
    if not isinstance(case_outcomes, tuple) or any(
        not isinstance(item, W08PrivateCaseOutcome) for item in case_outcomes
    ):
        raise W08PrivateEvaluationError("W08 private case outcome inventory 非法")
    outcome_by_case = {
        (item.dimension_key, item.observation_key): item
        for item in case_outcomes
    }
    if len(outcome_by_case) != len(case_outcomes):
        raise W08PrivateEvaluationError("W08 private case outcome 重复")
    expected_outcome_keys = {
        (dimension, tuple(pair.observation.stable_key.components))
        for dimension in W08_DIMENSION_KEYS
        for pair in pairs
    }
    if set(outcome_by_case) - expected_outcome_keys:
        raise W08PrivateEvaluationError("W08 private case outcome 越出冻结 inventory")
    target = None
    if ablation_key is not None:
        if ablation_key not in W08_ABLATION_KEYS:
            raise W08PrivateEvaluationError("W08 evaluator ablation 未注册")
        target = W08_DIMENSION_KEYS[W08_ABLATION_KEYS.index(ablation_key)]
    results = []
    pair_commitment = evidence_commitment({
        "observations": [
            list(item.observation.stable_key.components) for item in pairs
        ],
        "labels": [list(item.label.stable_key.components) for item in pairs],
        "packs": [item.pack_key for item in pairs],
    })
    for ordinal, dimension in enumerate(W08_DIMENSION_KEYS):
        required = len(pairs)
        passed = failed = ne = 0
        outcome_commitments = []
        for pair in pairs:
            outcome = outcome_by_case.get((
                dimension,
                tuple(pair.observation.stable_key.components),
            ))
            if dimension == target:
                failed += 1
                continue
            if outcome is None:
                ne += 1
                continue
            expected_payload = evidence_commitment(
                pair.label.expected_payload.to_value()
            )
            outcome_passed = all((
                outcome.actual_state == pair.label.expected_state,
                outcome.actual_payload_sha256 == expected_payload,
                all(state == "RESOLVED" for _, state in outcome.consumer_states),
                not any(value for _, value in outcome.shortcut_counts),
                bool(snapshot.dimension_artifact_keys[ordinal]),
            ))
            passed += int(outcome_passed)
            failed += int(not outcome_passed)
            outcome_commitments.append(list(outcome.outcome_key))
        status = "FAIL" if failed else "NE" if ne else "PASS"
        results.append(W08PrivateDimensionResult(
            dimension,
            status,
            passed,
            required,
            failed,
            ne,
            evidence_commitment({
                "ablation": ablation_key or "NONE",
                "artifact": list(snapshot.dimension_artifact_keys[ordinal]),
                "dimension": dimension,
                "outcome_commitments": outcome_commitments,
                "pair_commitment": pair_commitment,
            }),
        ))
    return tuple(results)


def assess_w08_orthogonal_ablations(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    case_outcomes: tuple[W08PrivateCaseOutcome, ...] = (),
) -> list[dict[str, object]]:
    result = []
    for ordinal, ablation in enumerate(W08_ABLATION_KEYS):
        dimensions = evaluate_w08_private_pairs(
            snapshot,
            pairs,
            ablation_key=ablation,
            case_outcomes=case_outcomes,
        )
        expected = tuple(
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W08_DIMENSION_KEYS))
        )
        statuses = tuple(item.status for item in dimensions)
        result.append({
            "ablation_key": ablation,
            "dimension_statuses": list(statuses),
            "status": "PASS" if statuses == expected else "FAIL",
            "target_dimension_key": W08_DIMENSION_KEYS[ordinal],
        })
    return result


def assess_w08_private_open_generation(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    layer_states: tuple[tuple[str, str], ...] = (),
) -> dict[str, object]:
    """用完整 held-out 组合 identity 闭合五层，不读取或回放 exact surface。"""
    if layer_states and (
        tuple(key for key, _ in layer_states) != W08_OPEN_GENERATION_LAYER_KEYS
        or any(state not in {"PASS", "FAIL", "NE"} for _, state in layer_states)
    ):
        raise W08PrivateEvaluationError("W08 open-generation layer 结果非法")
    combinations = {
        (
            item.pack_key,
            item.observation.payload_kind,
            item.observation.content_group_key.components,
            item.observation.template_group_key.components,
            item.observation.shape_group_key.components,
            item.observation.perturbation_kind,
        )
        for item in pairs
    }
    passed = (
        bool(combinations)
        and layer_states
        == tuple((key, "PASS") for key in W08_OPEN_GENERATION_LAYER_KEYS)
        and dict(snapshot.hard_conjunct_states).get("OPEN_GENERATION")
        == "PUBLIC_BOUNDED_PASS"
    )
    status = (
        "PASS"
        if passed
        else "FAIL"
        if any(state == "FAIL" for _, state in layer_states)
        else "NE"
    )
    return {
        "combination_count": len(combinations),
        "complete_template_replay_count": 0,
        "coverage_keys": list(W08_OPEN_GENERATION_COVERAGE_KEYS),
        "exact_surface_read_count": 0,
        "layer_states": [list(item) for item in layer_states] if layer_states else [
            [key, "NE"] for key in W08_OPEN_GENERATION_LAYER_KEYS
        ],
        "source_replay_count": 0,
        "status": status,
    }


def assess_w08_private_lc16(
    snapshot: W08EvaluatorSnapshot,
    *,
    bearing_cell_states: tuple[str, ...] = (),
) -> dict[str, object]:
    if bearing_cell_states and (
        len(bearing_cell_states)
        != len(W08_CARRIER_KEYS) * len(W08_CONSUMER_KEYS)
        or any(item not in {"PASS", "FAIL", "NE"} for item in bearing_cell_states)
    ):
        raise W08PrivateEvaluationError("W08 LC-16 bearing cell 结果非法")
    passed = dict(snapshot.hard_conjunct_states).get(
        "LC16_DISCOURSE_REFERENCE_GENERATION"
    ) == "PUBLIC_BOUNDED_PASS" and (
        len(bearing_cell_states) == len(W08_CARRIER_KEYS) * len(W08_CONSUMER_KEYS)
        and all(item == "PASS" for item in bearing_cell_states)
    )
    status = (
        "PASS"
        if passed
        else "FAIL"
        if "FAIL" in bearing_cell_states
        else "NE"
    )
    return {
        "bearing_cell_count": len(W08_CARRIER_KEYS) * len(W08_CONSUMER_KEYS),
        "bearing_scope_key": W08_LC16_SCOPE_KEY,
        "schema_required_state": "NE",
        "status": status,
        "wall_dimension_state": "NE",
    }


__all__ = [
    "W08EvaluatorSnapshot",
    "W08PrivateCaseOutcome",
    "W08PrivateEvaluationPair",
    "assess_w08_orthogonal_ablations",
    "assess_w08_private_lc16",
    "assess_w08_private_open_generation",
    "evaluate_w08_private_pairs",
    "snapshot_from_w08_outcome",
]
