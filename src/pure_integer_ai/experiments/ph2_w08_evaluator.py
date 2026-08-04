"""W08-09 五维、真实消融、开放生成和 LC-16 private evaluator 核心。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import (
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
from pure_integer_ai.experiments.ph2_w08_inference import (
    validate_w08_inference_outcome,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
    W08CandidateInferenceOutcome,
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


_COURSE_PAYLOAD_KINDS = {
    "AttributionQuotationCandidateV1",
    "DiscourseInformationCandidateV1",
    "OpenSetClarificationCandidateV1",
}


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


@dataclass(frozen=True)
class W08EvaluatorSnapshot:
    """从 Candidate canonical dump 恢复的只读承重能力与 inference 摘要。"""

    dimension_artifact_keys: tuple[tuple[int, ...], ...]
    use_cells: tuple[tuple[str, str, str], ...]
    hard_conjunct_states: tuple[tuple[str, str], ...]
    semantic_state_key: tuple[int, ...]
    dump_manifest_sha256: str
    inference_state_key: tuple[int, ...]
    inference_state_sha256: str
    inference_interface_version: str
    inference_rule_count: int
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
        sha_values = (self.dump_manifest_sha256, self.inference_state_sha256)
        if (
            not self.semantic_state_key
            or not self.inference_state_key
            or any(
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in sha_values
            )
            or self.inference_interface_version
            != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
            or type(self.inference_rule_count) is not int
            or self.inference_rule_count <= 0
            or any((
                self.future_payload_reads,
                self.evaluator_label_reads,
                self.host_learning_writes,
                self.memory_learning_writes,
            ))
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
        outcome.inference_state_key,
        outcome.inference_state_sha256,
        outcome.inference_interface_version,
        outcome.inference_rule_count,
        outcome.future_payload_reads,
        outcome.evaluator_label_reads,
        outcome.host_learning_writes,
        outcome.memory_learning_writes,
    )


def _normalize_operation_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    prefix, marker, suffix = value.partition("_")
    return suffix if marker and len(prefix) == 1 and suffix else value


def _course_payload_matches(
    actual: dict[str, object],
    expected: dict[str, object],
) -> bool:
    result = actual.get("result")
    if not isinstance(result, dict):
        return False
    accepted = expected.get("accepted")
    outputs = result.get("generated_outputs")
    accepted_outputs = expected.get("accepted_surfaces")
    receipt = result.get("render_receipt")
    render_policy = result.get("render_policy")
    if (
        accepted not in {0, 1}
        or result.get("accepted") != accepted
        or _normalize_operation_key(result.get("operation_key"))
        != _normalize_operation_key(expected.get("analysis_key"))
        or not isinstance(outputs, list)
        or not isinstance(accepted_outputs, list)
        or not isinstance(receipt, dict)
        or receipt.get("postcheck_state") != "PASS"
    ):
        return False
    if accepted == 0:
        return not outputs and not accepted_outputs and render_policy == "NO_TEXT"
    if render_policy == "COPY_VISIBLE_TEXT":
        return outputs == accepted_outputs and bool(outputs)
    return (
        render_policy == "STRUCTURAL_GENERATOR"
        and bool(outputs)
        and bool(accepted_outputs)
        and all(isinstance(item, str) and item for item in outputs)
        and receipt.get("output_count") == len(outputs)
        and len(receipt.get("output_sha256", ())) == len(outputs)
    )


def _outcome_passes_pair(
    snapshot: W08EvaluatorSnapshot,
    pair: W08PrivateEvaluationPair,
    outcome: W08CandidateInferenceOutcome,
    ordinal: int,
) -> bool:
    if (
        outcome.actual_state != pair.label.expected_state
        or outcome.component_state != "ACTIVE"
        or outcome.state_commitment_sha256 != snapshot.inference_state_sha256
        or any(state != "RESOLVED" for _, state in outcome.consumer_states)
        or not validate_w08_inference_outcome(pair.observation, outcome)
        or not snapshot.dimension_artifact_keys[ordinal]
    ):
        return False
    actual = outcome.actual_payload.to_value()
    expected = pair.label.expected_payload.to_value()
    if pair.observation.payload_kind in _COURSE_PAYLOAD_KINDS:
        return _course_payload_matches(actual, expected)
    return actual.get("result") == expected


def _outcome_inventory(
    pairs: tuple[W08PrivateEvaluationPair, ...],
    case_outcomes: tuple[W08CandidateInferenceOutcome, ...],
) -> dict[tuple[str, tuple[int, ...]], W08CandidateInferenceOutcome]:
    if not isinstance(case_outcomes, tuple) or any(
        not isinstance(item, W08CandidateInferenceOutcome) for item in case_outcomes
    ):
        raise W08PrivateEvaluationError("W08 private case outcome inventory 非法")
    values = {
        (item.dimension_key, item.observation_key): item for item in case_outcomes
    }
    if len(values) != len(case_outcomes):
        raise W08PrivateEvaluationError("W08 private case outcome 重复")
    allowed = {
        (dimension, tuple(pair.observation.stable_key.components))
        for dimension in W08_DIMENSION_KEYS
        for pair in pairs
    }
    if set(values) - allowed:
        raise W08PrivateEvaluationError("W08 private case outcome 越出冻结 inventory")
    return values


def evaluate_w08_private_pairs(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    ablation_key: str | None = None,
    case_outcomes: tuple[W08CandidateInferenceOutcome, ...] = (),
) -> tuple[W08PrivateDimensionResult, ...]:
    """只按 adapter 实际输出评定五维；缺输出是 NE，错误输出是 FAIL。"""
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
    if ablation_key is not None and ablation_key not in W08_ABLATION_KEYS:
        raise W08PrivateEvaluationError("W08 evaluator ablation 未注册")
    outcome_by_case = _outcome_inventory(pairs, case_outcomes)
    pair_commitment = evidence_commitment({
        "observations": [
            list(item.observation.stable_key.components) for item in pairs
        ],
        "labels": [list(item.label.stable_key.components) for item in pairs],
        "packs": [item.pack_key for item in pairs],
    })
    results = []
    for ordinal, dimension in enumerate(W08_DIMENSION_KEYS):
        required = len(pairs)
        passed = failed = ne = 0
        outcome_commitments = []
        for pair in pairs:
            outcome = outcome_by_case.get((
                dimension,
                tuple(pair.observation.stable_key.components),
            ))
            if outcome is None:
                ne += 1
                continue
            outcome_passed = _outcome_passes_pair(snapshot, pair, outcome, ordinal)
            passed += int(outcome_passed)
            failed += int(not outcome_passed)
            outcome_commitments.append(list(outcome.invocation_key))
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
    outcome_families: tuple[
        tuple[str, tuple[W08CandidateInferenceOutcome, ...]], ...
    ] = (),
) -> list[dict[str, object]]:
    if tuple(key for key, _ in outcome_families) != W08_ABLATION_KEYS:
        raise W08PrivateEvaluationError("W08 ablation outcome family 顺序漂移")
    result = []
    for ordinal, (ablation, outcomes) in enumerate(outcome_families):
        dimensions = evaluate_w08_private_pairs(
            snapshot,
            pairs,
            ablation_key=ablation,
            case_outcomes=outcomes,
        )
        expected = tuple(
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W08_DIMENSION_KEYS))
        )
        statuses = tuple(item.status for item in dimensions)
        target = W08_DIMENSION_KEYS[ordinal]
        real_component_pattern = all(
            item.component_state
            == ("DISABLED" if item.dimension_key == target else "ACTIVE")
            for item in outcomes
        ) and len({item.invocation_key for item in outcomes}) == len(outcomes)
        complete = _complete_outcomes(pairs, outcomes)
        has_fail = any(item.status == "FAIL" for item in dimensions)
        result.append({
            "ablation_key": ablation,
            "dimension_statuses": list(statuses),
            "invocation_count": len(outcomes),
            "real_component_disabled": int(real_component_pattern),
            "status": (
                "PASS"
                if complete and statuses == expected and real_component_pattern
                else "FAIL"
                if complete and (has_fail or not real_component_pattern)
                else "NE"
            ),
            "target_dimension_key": target,
        })
    return result


def _complete_outcomes(
    pairs: tuple[W08PrivateEvaluationPair, ...],
    outcomes: tuple[W08CandidateInferenceOutcome, ...],
) -> bool:
    return len(outcomes) == len(pairs) * len(W08_DIMENSION_KEYS) and len({
        (item.dimension_key, item.observation_key) for item in outcomes
    }) == len(outcomes)


def assess_w08_private_open_generation(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    case_outcomes: tuple[W08CandidateInferenceOutcome, ...] = (),
) -> dict[str, object]:
    """五层均从同一 baseline actual output 计算，不接受外部 PASS 字符串。"""
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
    if not case_outcomes:
        layers = tuple((key, "NE") for key in W08_OPEN_GENERATION_LAYER_KEYS)
    else:
        outcome_by_case = _outcome_inventory(pairs, case_outcomes)
        if not _complete_outcomes(pairs, case_outcomes):
            layers = tuple((key, "NE") for key in W08_OPEN_GENERATION_LAYER_KEYS)
            complete = False
        else:
            complete = True
            content = complete and all(
                _outcome_passes_pair(
                    snapshot,
                    pair,
                    outcome_by_case[(dimension, pair.observation.stable_key.components)],
                    ordinal,
                )
                for ordinal, dimension in enumerate(W08_DIMENSION_KEYS)
                for pair in pairs
            )
            structure = complete and all(
                validate_w08_inference_outcome(
                    next(
                        pair.observation
                        for pair in pairs
                        if pair.observation.stable_key.components == item.observation_key
                    ),
                    item,
                )
                for item in case_outcomes
            )
            discourse = complete and all(
                bool(item.source_key and item.scope_key and item.reference_key)
                and item.component_state == "ACTIVE"
                for item in case_outcomes
            )
            morphology = complete and all(
                (
                    item.actual_payload.to_value().get("payload_kind")
                    not in _COURSE_PAYLOAD_KINDS
                )
                or isinstance(
                    item.actual_payload.to_value().get("result", {}).get(
                        "generated_outputs"
                    ),
                    list,
                )
                and item.actual_payload.to_value().get("result", {}).get(
                    "render_receipt", {}
                ).get("postcheck_state") == "PASS"
                for item in case_outcomes
            )
            task = complete and len({item.invocation_key for item in case_outcomes}) == len(
                case_outcomes
            ) and all(
                all(state == "RESOLVED" for _, state in item.consumer_states)
                for item in case_outcomes
            )
            layer_values = (content, structure, discourse, morphology, task)
            layers = tuple(
                (key, "PASS" if passed else "FAIL")
                for key, passed in zip(W08_OPEN_GENERATION_LAYER_KEYS, layer_values)
            )
    hard_pass = dict(snapshot.hard_conjunct_states).get("OPEN_GENERATION") == (
        "PUBLIC_BOUNDED_PASS"
    )
    status = (
        "PASS"
        if hard_pass and layers == tuple(
            (key, "PASS") for key in W08_OPEN_GENERATION_LAYER_KEYS
        )
        else "FAIL"
        if any(state == "FAIL" for _, state in layers)
        else "NE"
    )
    return {
        "combination_count": len(combinations),
        "complete_template_replay_count": 0,
        "coverage_keys": list(W08_OPEN_GENERATION_COVERAGE_KEYS),
        "exact_surface_read_count": sum(
            dict(item.shortcut_counts)["exact_surface_reads"] for item in case_outcomes
        ),
        "layer_states": [list(item) for item in layers],
        "output_invocation_count": len(case_outcomes),
        "source_replay_count": 0,
        "status": status,
    }


def assess_w08_private_lc16(
    snapshot: W08EvaluatorSnapshot,
    pairs: tuple[W08PrivateEvaluationPair, ...],
    *,
    case_outcomes: tuple[W08CandidateInferenceOutcome, ...] = (),
) -> dict[str, object]:
    """27 个 carrier×U/R/G cell 逐一消费 baseline outcome commitment。"""
    if case_outcomes:
        _outcome_inventory(pairs, case_outcomes)
    complete = bool(case_outcomes) and _complete_outcomes(pairs, case_outcomes)
    cells = []
    for carrier in W08_CARRIER_KEYS:
        for consumer in W08_CONSUMER_KEYS:
            passed = complete and all(
                dict(item.consumer_states)[consumer] == "RESOLVED"
                and item.component_state == "ACTIVE"
                for item in case_outcomes
            )
            cells.append({
                "carrier_key": carrier,
                "consumer_key": consumer,
                "evidence_sha256": evidence_commitment({
                    "carrier_key": carrier,
                    "consumer_key": consumer,
                    "invocations": [list(item.invocation_key) for item in case_outcomes],
                }),
                "state": "PASS" if passed else "FAIL" if case_outcomes else "NE",
            })
    hard_pass = dict(snapshot.hard_conjunct_states).get(
        "LC16_DISCOURSE_REFERENCE_GENERATION"
    ) == "PUBLIC_BOUNDED_PASS"
    if not complete:
        cells = [dict(item, state="NE") for item in cells]
    states = tuple(item["state"] for item in cells)
    status = (
        "PASS" if hard_pass and states and all(item == "PASS" for item in states)
        else "FAIL" if "FAIL" in states
        else "NE"
    )
    return {
        "bearing_cell_count": len(cells),
        "bearing_scope_key": W08_LC16_SCOPE_KEY,
        "cell_commitment": evidence_commitment(cells),
        "output_invocation_count": len(case_outcomes),
        "schema_required_state": "NE",
        "status": status,
        "wall_dimension_state": "NE",
    }


__all__ = [
    "W08EvaluatorSnapshot",
    "W08PrivateEvaluationPair",
    "assess_w08_orthogonal_ablations",
    "assess_w08_private_lc16",
    "assess_w08_private_open_generation",
    "evaluate_w08_private_pairs",
    "snapshot_from_w08_outcome",
]
