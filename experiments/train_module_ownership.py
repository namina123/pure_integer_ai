"""正式训练周边模块的职责所有权和依赖边界。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainModuleOwnership:
    """描述一个训练模块唯一负责的职责及其禁止承担的职责。"""

    module: str
    owns: tuple[str, ...]
    forbids: tuple[str, ...]
    may_import_runner: bool = False


TRAIN_MODULE_OWNERSHIP: tuple[TrainModuleOwnership, ...] = (
    TrainModuleOwnership(
        module="experiments.formal_train",
        owns=("assembly", "stage_calls", "transaction_boundary", "report_merge"),
        forbids=("new_language_parsing", "new_evidence_model", "new_verifier_routing"),
        may_import_runner=True,
    ),
    TrainModuleOwnership(
        module="experiments.train_context",
        owns=("context_type", "storage_bootstrap", "scope_identity_helpers"),
        forbids=("stage_loop", "domain_learning"),
    ),
    TrainModuleOwnership(
        module="experiments.train_scope",
        owns=("stage_scope", "relation_scope"),
        forbids=("graph_write", "domain_learning"),
    ),
    TrainModuleOwnership(
        module="experiments.train_execution",
        owns=("execution_counts", "external_clock", "execution_report"),
        forbids=("graph_state", "readiness_decision"),
    ),
    TrainModuleOwnership(
        module="experiments.train_gate_profile",
        owns=("production_training_gate_profile", "context_gate_lifecycle"),
        forbids=("stage_loop", "domain_learning", "graph_write"),
    ),
    TrainModuleOwnership(
        module="experiments.language_observation",
        owns=("word_form_projection", "boundary_projection", "span_materialization", "segment_build"),
        forbids=("stage_loop", "training_report"),
    ),
    TrainModuleOwnership(
        module="experiments.language_protocol_runtime",
        owns=("language_protocol_installation", "protocol_dependency_validation"),
        forbids=("stage_loop", "language_parsing", "training_report"),
    ),
    TrainModuleOwnership(
        module="experiments.language_structure_runtime",
        owns=("language_structure_discovery", "recognition", "structure_tally"),
        forbids=("stage_loop", "graph_dump"),
    ),
    TrainModuleOwnership(
        module="experiments.round_runtime",
        owns=(
            "round_lifecycle",
            "observation_execution",
            "verifier_execution",
            "reward_round",
        ),
        forbids=("stage_loop", "graph_dump", "training_report"),
    ),
    TrainModuleOwnership(
        module="experiments.task_generation_runtime",
        owns=("task_generation", "generation_verification", "generation_summary"),
        forbids=("stage_loop", "graph_dump", "training_report"),
    ),
    TrainModuleOwnership(
        module="experiments.arithmetic_structure_runtime",
        owns=("arithmetic_structure_discovery", "recognition", "held_out_verification"),
        forbids=("stage_loop", "graph_dump", "generation"),
    ),
    TrainModuleOwnership(
        module="experiments.evaluation_runtime",
        owns=("calibration", "held_out_observation", "floor_measurement", "offline_eval"),
        forbids=("training_commit", "stage_loop", "host_state_write"),
    ),
    TrainModuleOwnership(
        module="experiments.preflight_runtime",
        owns=("preflight_trial", "release_gate", "preflight_report"),
        forbids=("training_commit", "formal_stage_loop", "graph_dump"),
    ),
    TrainModuleOwnership(
        module="experiments.train_diagnostics",
        owns=("graph_counts", "anti_collapse_summary", "weaning_blockers"),
        forbids=("graph_write", "stage_loop", "release_decision"),
    ),
    TrainModuleOwnership(
        module="experiments.stage_learning_runtime",
        owns=("base_frequency_intake", "candidate_promotion"),
        forbids=("stage_loop", "graph_dump", "report_merge"),
    ),
    TrainModuleOwnership(
        module="experiments.verification_dispatch",
        owns=("legacy_verifier_applicability", "legacy_dimension_identity"),
        forbids=("verifier_execution", "graph_write"),
    ),
    TrainModuleOwnership(
        module="experiments.verification_orchestration",
        owns=(
            "typed_verifier_registration",
            "multi_dimension_execution",
            "verification_result_schema",
            "effect_commit_boundary",
        ),
        forbids=("domain_semantics", "direct_graph_write", "scalar_merge"),
    ),
    TrainModuleOwnership(
        module="experiments.event_time_verification",
        owns=("event_time_r09_adapter", "event_time_verdict_mapping"),
        forbids=("event_time_fact_inference", "causal_write", "scalar_merge"),
    ),
    TrainModuleOwnership(
        module="experiments.precedence_relation_runtime",
        owns=(
            "occurrence_to_order_learning_orchestration",
            "structure_order_lifecycle_orchestration",
            "typed_order_consumption_report",
        ),
        forbids=(
            "surface_rule_inference",
            "role_name_hardcode",
            "legacy_precedes_read",
            "event_time_semantics",
        ),
    ),
    TrainModuleOwnership(
        module="experiments.causal_relation_runtime",
        owns=(
            "causal_relation_forming_orchestration",
            "independent_causal_verification",
            "provisional_causal_execution",
            "causal_generation_use_ledger",
        ),
        forbids=(
            "surface_cue_semantics",
            "legacy_causes_read",
            "event_time_fact_inference",
            "definitive_truth",
        ),
    ),
    TrainModuleOwnership(
        module="experiments.causal_relation_course",
        owns=(
            "causal_course_request_schema",
            "formal_causal_round_orchestration",
            "causal_runtime_installation",
            "causal_evaluation_clone",
        ),
        forbids=(
            "surface_language_mapping",
            "legacy_causes_read",
            "verifier_semantics",
            "definitive_truth",
        ),
    ),
    TrainModuleOwnership(
        module="experiments.evaluation_isolation",
        owns=("evaluation_clone", "host_write_guard"),
        forbids=("training_commit", "runner_type_definition"),
    ),
)


def ownership_for(module: str) -> TrainModuleOwnership:
    """按完整模块名返回唯一 ownership 记录，未知模块直接失败。"""
    matches = [item for item in TRAIN_MODULE_OWNERSHIP if item.module == module]
    if len(matches) != 1:
        raise KeyError(f"训练模块 ownership 未唯一登记: {module}")
    return matches[0]


__all__ = [
    "TRAIN_MODULE_OWNERSHIP",
    "TrainModuleOwnership",
    "ownership_for",
]
