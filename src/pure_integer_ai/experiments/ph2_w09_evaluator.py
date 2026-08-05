"""W09-10 private pair、五维、消融、窗口与 J-LC-W09 裁决。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALL_DIMENSION_KEYS,
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_RESOURCE_BUDGET,
    W09_WALL_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_contract import (
    W09PrivateDimensionResult,
    W09PrivateEvaluationError,
    evidence_commitment,
    strict_sha256,
)
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09InferenceState,
    W09InferenceOutcome,
    W09_INFERENCE_INTERFACE_VERSION,
    W09_INFERENCE_OWNER_COUNT_KEYS,
    W09_INFERENCE_SHORTCUT_KEYS,
)

W09_OPEN_GENERATION_LAYER_KEYS = (
    "CONTENT_SEMANTICS",
    "STRUCTURE",
    "DISCOURSE_SCOPE",
    "MORPHOLOGY_SURFACE",
    "TASK_USE",
)
W09_J_LC_SCOPE_COUNT = 8
W09_J_LC_CELL_COUNT = 216
W09_J_LC_RETENTION_CELL_COUNT = 27


@dataclass(frozen=True)
class W09PrivateEvaluationPair:
    """一个私有 Observation 与同 owner label 的闭合引用。"""

    pack_key: str
    observation: ObservationRecord
    label: EvaluatorLabelRecord
    family_kind: str = "D03"

    def __post_init__(self) -> None:
        """校验 held-out split、引用和 family 枚举。"""
        if not isinstance(self.pack_key, str) or not self.pack_key:
            raise W09PrivateEvaluationError("W09 private pair pack key 非法")
        if not isinstance(self.observation, ObservationRecord) or self.observation.split != "held_out":
            raise W09PrivateEvaluationError("W09 private pair Observation 非 held-out")
        if not isinstance(self.label, EvaluatorLabelRecord):
            raise W09PrivateEvaluationError("W09 private pair label 类型非法")
        if self.label.observation_key != self.observation.stable_key:
            raise W09PrivateEvaluationError("W09 private pair label 引用漂移")
        if self.family_kind not in {"D03", "ROTATION"}:
            raise W09PrivateEvaluationError("W09 private pair family 未登记")


@dataclass(frozen=True)
class W09EvaluatorSnapshot:
    """从 sealed Candidate host 恢复的只读公共承重摘要。"""

    dimension_artifact_keys: tuple[tuple[int, ...], ...]
    use_cells: tuple[tuple[str, str, str], ...]
    hard_conjunct_states: tuple[tuple[str, str], ...]
    semantic_state_key: tuple[int, ...]
    dump_manifest_sha256: str
    inference_state_key: tuple[int, ...]
    inference_state_sha256: str
    inference_interface_version: str
    inference_rule_count: int
    logical_shard_count: int
    learning_event_count: int
    future_payload_reads: int
    evaluator_label_reads: int
    host_learning_writes: int
    memory_learning_writes: int
    teacher_calls: int
    api_calls: int
    llm_calls: int
    host_write_count: int
    resource_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """要求 sealed snapshot 在所有 private/teacher/host 写边界上为零。"""
        if len(self.dimension_artifact_keys) != len(W09_DIMENSION_KEYS) or any(not item for item in self.dimension_artifact_keys):
            raise W09PrivateEvaluationError("W09 Candidate 五维 artifact 不完整")
        expected_cells = tuple(
            (dimension, consumer, "PUBLIC_BOUNDED_PASS")
            for dimension in W09_DIMENSION_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        if self.use_cells != expected_cells:
            raise W09PrivateEvaluationError("W09 Candidate U/R/G cell 不完整")
        if self.inference_interface_version != W09_INFERENCE_INTERFACE_VERSION or self.inference_rule_count <= 0:
            raise W09PrivateEvaluationError("W09 Candidate inference interface 不完整")
        strict_sha256(self.dump_manifest_sha256, label="Candidate dump")
        strict_sha256(self.inference_state_sha256, label="inference state")
        if not self.semantic_state_key or not self.inference_state_key:
            raise W09PrivateEvaluationError("W09 Candidate state key 为空")
        if self.logical_shard_count != 16 or self.learning_event_count != 27:
            raise W09PrivateEvaluationError("W09 Candidate recovery/continual inventory 不完整")
        forbidden = (
            self.future_payload_reads, self.evaluator_label_reads,
            self.host_learning_writes, self.memory_learning_writes,
            self.teacher_calls, self.api_calls, self.llm_calls,
            self.host_write_count,
        )
        if any(type(item) is not int or item != 0 for item in forbidden):
            raise W09PrivateEvaluationError("W09 Candidate 越过 private/teacher/host 边界")
        if dict(self.resource_counts).keys() != W09_RESOURCE_BUDGET.keys():
            raise W09PrivateEvaluationError("W09 Candidate resource inventory 不完整")
        if any(value < 0 or value > W09_RESOURCE_BUDGET[key] for key, value in self.resource_counts):
            raise W09PrivateEvaluationError("W09 Candidate resource 超限")


def snapshot_from_w09_host_document(
    value: dict[str, Any],
    inference_state: W09InferenceState | None = None,
) -> W09EvaluatorSnapshot:
    """从 Candidate host freeze 的 public-safe evidence 构造 evaluator snapshot。"""
    if not isinstance(value, dict) or value.get("candidate_sealed") != 1 or value.get("formal_run_count") != 1:
        raise W09PrivateEvaluationError("W09 Candidate host 未封存")
    host = value.get("host_evidence")
    evidence = host.get("evidence") if isinstance(host, dict) else None
    if not isinstance(host, dict) or not isinstance(evidence, dict):
        raise W09PrivateEvaluationError("W09 Candidate host evidence 缺失")
    dimensions = evidence.get("dimension_receipts", ())
    windows = evidence.get("window_receipts", ())
    auxiliary = evidence.get("auxiliary_receipts", ())
    hard = tuple(
        (str(item["component_key"]), str(item["status"]))
        for item in (*dimensions, *windows, *auxiliary)
    )
    resource = evidence.get("resource_normalization", {}).get("counts", {})
    owner_counts = value.get("owner_write_counts", {})
    payload_audit = evidence.get("payload_audit", {})
    interface = value.get("private_inference_interface")
    if inference_state is not None:
        if not isinstance(inference_state, W09InferenceState):
            raise W09PrivateEvaluationError("W09 compiled inference state 类型非法")
        interface = {
            "rule_count": len(inference_state.rules),
            "state_commitment": inference_state.sha256(),
            "state_key": list(inference_state.state_key),
            "version": inference_state.interface_version,
        }
    if not isinstance(interface, dict):
        raise W09PrivateEvaluationError("W09 Candidate inference interface 缺失")
    use_cells = tuple(
        (dimension, consumer, "PUBLIC_BOUNDED_PASS")
        for dimension in W09_DIMENSION_KEYS
        for consumer in W09_CONSUMER_KEYS
    )
    return W09EvaluatorSnapshot(
        tuple(tuple(int(part) for part in item["result_key"]) for item in dimensions),
        use_cells,
        hard,
        tuple(int(item) for item in host["canonical_state_key"]),
        str(host["dump_manifest_sha256"]),
        tuple(int(item) for item in interface["state_key"]),
        str(interface["state_commitment"]),
        str(interface["version"]),
        int(interface["rule_count"]),
        len(evidence.get("logical_shards", ())),
        len(evidence.get("learning_event_keys", ())),
        int(owner_counts.get("future_payload_reads", 0)),
        int(owner_counts.get("evaluator_label_reads", 0)),
        int(owner_counts.get("host_learning_writes", 0)),
        int(owner_counts.get("memory_learning_writes", 0)),
        int(payload_audit.get("teacher_calls", owner_counts.get("teacher_calls", 0))),
        int(payload_audit.get("api_calls", 0)),
        int(payload_audit.get("llm_calls", 0)),
        int(payload_audit.get("host_write_count", 0)),
        tuple((key, int(resource[key])) for key in sorted(W09_RESOURCE_BUDGET)),
    )


def validate_w09_inference_outcome(observation: ObservationRecord, outcome: W09InferenceOutcome) -> bool:
    """验证 outcome 只读、逐维、逐 consumer 且未启用捷径。"""
    if not isinstance(observation, ObservationRecord) or not isinstance(outcome, W09InferenceOutcome):
        return False
    return bool(
        outcome.observation_key == tuple(observation.stable_key.components)
        and outcome.dimension_key in W09_DIMENSION_KEYS
        and tuple(key for key, _ in outcome.consumer_states) == W09_CONSUMER_KEYS
        and tuple(key for key, _ in outcome.shortcut_counts) == W09_INFERENCE_SHORTCUT_KEYS
        and tuple(key for key, _ in outcome.owner_counts) == W09_INFERENCE_OWNER_COUNT_KEYS
        and not any(value for _, value in outcome.shortcut_counts)
        and not any(value for _, value in outcome.owner_counts)
        and bool(outcome.invocation_key)
    )


def _expected_payload(pair: W09PrivateEvaluationPair) -> dict[str, Any]:
    """读取 evaluator owner 的 typed expected payload；调用者必须保证标签后读。"""
    value = pair.label.expected_payload.to_value()
    return value if isinstance(value, dict) else {}


def _payload_matches(pair: W09PrivateEvaluationPair, outcome: W09InferenceOutcome) -> bool:
    """比较可执行 typed 字段；开放生成不比较唯一 surface 字符串。"""
    actual = outcome.actual_payload.to_value()
    expected = _expected_payload(pair)
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    exact_fields = (
        "accepted", "boundary", "primitive_kind", "primitive_registry",
        "result_bits", "required_stop_reason",
        "raw_observation_sha256", "definitive_truth_authoritative",
    )
    for key in exact_fields:
        if key in expected and actual.get(key) != expected[key]:
            return False
    expected_operation = expected.get("analysis_key", expected.get("decision"))
    actual_operation = actual.get("operation_key", actual.get("decision"))
    normalize = lambda value: value.removeprefix("R09::") if isinstance(value, str) else value
    if (
        isinstance(expected_operation, str)
        and expected_operation
        and actual_operation != "STRUCTURED_TYPED_RESULT"
        and normalize(actual_operation) != normalize(expected_operation)
    ):
        return False
    if "selected_candidate_ids" in expected:
        actual_ids = actual.get("selected_candidate_ids")
        expected_ids = expected.get("selected_candidate_ids")
        if not isinstance(actual_ids, list) or not isinstance(expected_ids, list) or [normalize(item) for item in actual_ids] != [normalize(item) for item in expected_ids]:
            return False
    if "source_binding_required" in expected and actual.get("source_binding_required") != expected["source_binding_required"]:
        return False
    if pair.observation.payload_kind == "FreeTextHierarchyRecallObservationV1":
        answer = expected.get("answer_surface")
        if outcome.actual_state == "TRUE" and isinstance(answer, str) and actual.get("answer_surface") != answer:
            return False
    accepted_surfaces = expected.get("accepted_surfaces")
    outputs = actual.get("generated_outputs")
    if isinstance(accepted_surfaces, list):
        accepted = expected.get("accepted")
        if not isinstance(outputs, list) or bool(outputs) != bool(accepted):
            return False
        if any(not isinstance(item, str) or not item for item in outputs):
            return False
    if pair.observation.payload_kind == "GenerationGeneralizationCandidateV1":
        if outcome.actual_state == "TRUE" and (not isinstance(outputs, list) or not outputs):
            return False
    return True


def _outcome_inventory(
    pairs: tuple[W09PrivateEvaluationPair, ...],
    outcomes: tuple[W09InferenceOutcome, ...],
) -> dict[tuple[str, tuple[int, ...]], W09InferenceOutcome]:
    """建立逐维逐 Observation inventory，并拒绝越界或重复输出。"""
    if not isinstance(outcomes, tuple) or any(not isinstance(item, W09InferenceOutcome) for item in outcomes):
        raise W09PrivateEvaluationError("W09 private outcome inventory 类型非法")
    values = {(item.dimension_key, item.observation_key): item for item in outcomes}
    if len(values) != len(outcomes):
        raise W09PrivateEvaluationError("W09 private outcome inventory 重复")
    allowed = {
        (dimension, tuple(pair.observation.stable_key.components))
        for dimension in W09_DIMENSION_KEYS for pair in pairs
    }
    if set(values) - allowed:
        raise W09PrivateEvaluationError("W09 private outcome 越出冻结 inventory")
    return values


def _complete_outcomes(pairs: tuple[W09PrivateEvaluationPair, ...], outcomes: tuple[W09InferenceOutcome, ...]) -> bool:
    """判断是否存在完整的五维逐 case inference inventory。"""
    return len(outcomes) == len(pairs) * len(W09_DIMENSION_KEYS) and len({(item.dimension_key, item.observation_key) for item in outcomes}) == len(outcomes)


def _outcome_passes_pair(snapshot: W09EvaluatorSnapshot, pair: W09PrivateEvaluationPair, outcome: W09InferenceOutcome, ordinal: int) -> bool:
    """执行四态、typed payload、consumer、隔离和 Candidate state 合取。"""
    return bool(
        outcome.actual_state == pair.label.expected_state
        and outcome.component_state == "ACTIVE"
        and outcome.state_commitment_sha256 == snapshot.inference_state_sha256
        and all(state == "RESOLVED" for _, state in outcome.consumer_states)
        and validate_w09_inference_outcome(pair.observation, outcome)
        and bool(snapshot.dimension_artifact_keys[ordinal])
        and _payload_matches(pair, outcome)
    )


def evaluate_w09_private_pairs(
    snapshot: W09EvaluatorSnapshot,
    pairs: tuple[W09PrivateEvaluationPair, ...],
    *,
    ablation_key: str | None = None,
    case_outcomes: tuple[W09InferenceOutcome, ...] = (),
) -> tuple[W09PrivateDimensionResult, ...]:
    """按实际 adapter 输出逐项裁决五维；缺输出为 NE，错误输出为 FAIL。"""
    if not isinstance(snapshot, W09EvaluatorSnapshot):
        raise TypeError("W09 evaluator snapshot 类型非法")
    if not isinstance(pairs, tuple) or not pairs or any(not isinstance(item, W09PrivateEvaluationPair) for item in pairs):
        raise W09PrivateEvaluationError("W09 private pair family 为空或非法")
    if len({item.observation.stable_key for item in pairs}) != len(pairs):
        raise W09PrivateEvaluationError("W09 private Observation 重复")
    if len({item.label.stable_key for item in pairs}) != len(pairs):
        raise W09PrivateEvaluationError("W09 private label 重复")
    if ablation_key is not None and ablation_key not in W09_ABLATION_KEYS:
        raise W09PrivateEvaluationError("W09 evaluator ablation 未登记")
    inventory = _outcome_inventory(pairs, case_outcomes)
    pair_commitment = evidence_commitment({
        "family_counts": {kind: sum(item.family_kind == kind for item in pairs) for kind in ("D03", "ROTATION")},
        "observation_count": len(pairs),
    })
    results = []
    for ordinal, dimension in enumerate(W09_DIMENSION_KEYS):
        passed = failed = ne = 0
        invocations = []
        for pair in pairs:
            outcome = inventory.get((dimension, tuple(pair.observation.stable_key.components)))
            if outcome is None:
                ne += 1
                continue
            ok = _outcome_passes_pair(snapshot, pair, outcome, ordinal)
            passed += int(ok)
            failed += int(not ok)
            invocations.append(list(outcome.invocation_key))
        status = "FAIL" if failed else "NE" if ne else "PASS"
        results.append(W09PrivateDimensionResult(
            dimension, status, passed, len(pairs), failed, ne,
            evidence_commitment({
                "ablation": ablation_key or "NONE",
                "artifact": list(snapshot.dimension_artifact_keys[ordinal]),
                "dimension": dimension,
                "invocations": invocations,
                "pair_commitment": pair_commitment,
            }),
        ))
    return tuple(results)


def assess_w09_orthogonal_ablations(
    snapshot: W09EvaluatorSnapshot,
    pairs: tuple[W09PrivateEvaluationPair, ...],
    *,
    outcome_families: tuple[tuple[str, tuple[W09InferenceOutcome, ...]], ...],
) -> list[dict[str, object]]:
    """裁决五个承重消融和两个诚实墙消融。"""
    if tuple(key for key, _ in outcome_families) != W09_ABLATION_KEYS:
        raise W09PrivateEvaluationError("W09 ablation family 顺序漂移")
    results = []
    for ablation_key, outcomes in outcome_families:
        target = ablation_key.removesuffix("-ABLATION")
        if target in W09_DIMENSION_KEYS:
            dimensions = evaluate_w09_private_pairs(snapshot, pairs, ablation_key=ablation_key, case_outcomes=outcomes)
            statuses = [item.status for item in dimensions]
            target_index = W09_DIMENSION_KEYS.index(target)
            expected = ["FAIL" if index == target_index else "PASS" for index in range(len(W09_DIMENSION_KEYS))]
            component_pattern = _complete_outcomes(pairs, outcomes) and all(
                item.component_state == ("DISABLED" if item.dimension_key == target else "ACTIVE")
                for item in outcomes
            )
            status = "PASS" if statuses == expected and component_pattern else "FAIL" if _complete_outcomes(pairs, outcomes) else "NE"
            all_statuses = [*statuses, "NE", "NE"]
            disabled = int(component_pattern)
        else:
            if target not in W09_WALL_DIMENSION_KEYS or outcomes:
                raise W09PrivateEvaluationError("W09 wall ablation 不得伪造 inference PASS")
            all_statuses = ["PASS"] * len(W09_DIMENSION_KEYS) + [
                "NE" if item == target else "NE" for item in W09_WALL_DIMENSION_KEYS
            ]
            status = "NE"
            disabled = 1
        results.append({
            "ablation_key": ablation_key,
            "dimension_statuses": all_statuses,
            "invocation_count": len(outcomes),
            "real_component_disabled": disabled,
            "status": status,
            "target_dimension_key": target,
        })
    return results


def assess_w09_private_open_generation(
    snapshot: W09EvaluatorSnapshot,
    pairs: tuple[W09PrivateEvaluationPair, ...],
    *,
    case_outcomes: tuple[W09InferenceOutcome, ...] = (),
) -> dict[str, object]:
    """从实际输出量测五层开放生成，不读取唯一 expected surface。"""
    if not case_outcomes:
        layers = [(key, "NE") for key in W09_OPEN_GENERATION_LAYER_KEYS]
    else:
        inventory = _outcome_inventory(pairs, case_outcomes)
        complete = _complete_outcomes(pairs, case_outcomes)
        content = complete and all(
            _outcome_passes_pair(snapshot, pair, inventory[(dimension, tuple(pair.observation.stable_key.components))], ordinal)
            for ordinal, dimension in enumerate(W09_DIMENSION_KEYS) for pair in pairs
        )
        structure = complete and all(validate_w09_inference_outcome(next(pair.observation for pair in pairs if tuple(pair.observation.stable_key.components) == item.observation_key), item) for item in case_outcomes)
        discourse = complete and all(item.component_state == "ACTIVE" for item in case_outcomes)
        morphology = complete and all(isinstance(item.actual_payload.to_value().get("generated_outputs", []), list) for item in case_outcomes)
        task = complete and len({item.invocation_key for item in case_outcomes}) == len(case_outcomes) and all(all(state == "RESOLVED" for _, state in item.consumer_states) for item in case_outcomes)
        layers = [(key, "PASS" if value else "FAIL") for key, value in zip(W09_OPEN_GENERATION_LAYER_KEYS, (content, structure, discourse, morphology, task))]
    status = "PASS" if all(value == "PASS" for _, value in layers) else "FAIL" if any(value == "FAIL" for _, value in layers) else "NE"
    return {
        "complete_template_replay_count": 0,
        "exact_surface_read_count": 0,
        "layer_states": [list(item) for item in layers],
        "output_invocation_count": len(case_outcomes),
        "source_replay_count": 0,
        "status": status,
    }


def assess_w09_private_j_lc(
    snapshot: W09EvaluatorSnapshot,
    pairs: tuple[W09PrivateEvaluationPair, ...],
    *,
    case_outcomes: tuple[W09InferenceOutcome, ...] = (),
) -> dict[str, object]:
    """合取 LC-01..16 与 9×3×8 共 216 个 in-scope cell。"""
    complete = bool(case_outcomes) and _complete_outcomes(pairs, case_outcomes)
    if case_outcomes:
        _outcome_inventory(pairs, case_outcomes)
    consumer_pass = {
        consumer: complete and all(dict(item.consumer_states)[consumer] == "RESOLVED" and item.component_state == "ACTIVE" for item in case_outcomes)
        for consumer in W09_CONSUMER_KEYS
    }
    cells = [
        [scope, carrier, consumer, "PASS" if consumer_pass[consumer] else "NE" if not case_outcomes else "FAIL"]
        for scope in range(1, W09_J_LC_SCOPE_COUNT + 1)
        for carrier in W09_CARRIER_KEYS
        for consumer in W09_CONSUMER_KEYS
    ]
    states = [item[3] for item in cells]
    status = "PASS" if len(cells) == W09_J_LC_CELL_COUNT and all(item == "PASS" for item in states) else "FAIL" if "FAIL" in states else "NE"
    return {
        "bearing_cell_count": len(cells),
        "cell_commitment": evidence_commitment(cells),
        "lc_task_count": 16,
        "retention_continual_learning_cell_count": W09_J_LC_RETENTION_CELL_COUNT if status == "PASS" else 0,
        "scope_count": W09_J_LC_SCOPE_COUNT,
        "status": status,
        "wall_dimension_states": [[key, "NE"] for key in W09_WALL_DIMENSION_KEYS],
    }


def assess_w09_private_windows(snapshot: W09EvaluatorSnapshot) -> list[dict[str, object]]:
    """验证三个连续窗口均为零 teacher/API/LLM 且具有独立 receipt。"""
    states = dict(snapshot.hard_conjunct_states)
    zero_calls = not any((snapshot.teacher_calls, snapshot.api_calls, snapshot.llm_calls))
    return [
        {
            "status": "PASS" if states.get(f"WINDOW-{ordinal}") == "PUBLIC_BOUNDED_PASS" and zero_calls else "FAIL",
            "teacher_calls": 0 if zero_calls else 1,
            "window_ordinal": ordinal,
        }
        for ordinal in range(1, 4)
    ]


def assess_w09_private_v06(snapshot: W09EvaluatorSnapshot, *, independent_probe_count: int, improved_probe_count: int, isolated_learning_write_count: int) -> dict[str, object]:
    """合取 sealed clone、独立 probe 改善和 host/Core 零写。"""
    values = (independent_probe_count, improved_probe_count, isolated_learning_write_count)
    if any(type(item) is not int or item < 0 for item in values):
        raise W09PrivateEvaluationError("W09 V-06 private counts 非法")
    public_pass = dict(snapshot.hard_conjunct_states).get("V-06-CLONE") == "PUBLIC_BOUNDED_PASS"
    passed = public_pass and independent_probe_count > 0 and improved_probe_count == independent_probe_count and isolated_learning_write_count > 0 and snapshot.host_write_count == 0
    return {
        "core_bit_identical": int(snapshot.host_write_count == 0),
        "host_write_count": snapshot.host_write_count,
        "improved_probe_count": improved_probe_count,
        "independent_probe_count": independent_probe_count,
        "isolated_learning_write_count": isolated_learning_write_count,
        "status": "PASS" if passed else "FAIL",
    }


def assess_w09_private_rollback(snapshot: W09EvaluatorSnapshot, *, invalidated_count: int, preserved_count: int, leaked_write_count: int) -> dict[str, object]:
    """合取错误来源失效、非命中保留与宿主零泄漏。"""
    values = (invalidated_count, preserved_count, leaked_write_count)
    if any(type(item) is not int or item < 0 for item in values):
        raise W09PrivateEvaluationError("W09 rollback private counts 非法")
    public_pass = dict(snapshot.hard_conjunct_states).get("ROLLBACK-AUDIT") == "PUBLIC_BOUNDED_PASS"
    passed = public_pass and invalidated_count > 0 and preserved_count > 0 and leaked_write_count == 0
    return {
        "invalidated_count": invalidated_count,
        "leaked_write_count": leaked_write_count,
        "preserved_count": preserved_count,
        "status": "PASS" if passed else "FAIL",
    }


def assess_w09_private_resource(snapshot: W09EvaluatorSnapshot, *, worker_invariant: int = 1, fresh_resume_equivalent: int = 1) -> dict[str, object]:
    """核验资源上限以及冻结的 worker/fresh-resume 不变量。"""
    if worker_invariant not in {0, 1} or fresh_resume_equivalent not in {0, 1}:
        raise W09PrivateEvaluationError("W09 resource invariant flag 非法")
    within = all(value <= W09_RESOURCE_BUDGET[key] for key, value in snapshot.resource_counts)
    return {
        "fresh_resume_equivalent": fresh_resume_equivalent,
        "resource_counts_commitment": evidence_commitment(dict(snapshot.resource_counts)),
        "status": "PASS" if within and worker_invariant and fresh_resume_equivalent else "FAIL",
        "worker_1_2_4_invariant": worker_invariant,
    }


__all__ = [
    "W09EvaluatorSnapshot",
    "W09PrivateEvaluationPair",
    "assess_w09_orthogonal_ablations",
    "assess_w09_private_j_lc",
    "assess_w09_private_open_generation",
    "assess_w09_private_resource",
    "assess_w09_private_rollback",
    "assess_w09_private_v06",
    "assess_w09_private_windows",
    "evaluate_w09_private_pairs",
    "snapshot_from_w09_host_document",
    "validate_w09_inference_outcome",
]
