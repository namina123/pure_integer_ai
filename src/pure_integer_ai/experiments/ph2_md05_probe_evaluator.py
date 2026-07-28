"""MD-05 独立 held-out evaluator；MD-04 runtime 不导入本模块。"""
from __future__ import annotations

from collections import defaultdict

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    MD_BASELINE_KEYS,
)
from pure_integer_ai.experiments.ph2_md04_probe_contract import (
    FORMAT_VERSION,
    MD04_ABLATION_KEYS,
    MD04_PREREGISTRATION_VERSION,
    MD05_DECISION_VERSION,
    MD04ProbeContractError,
    MD04ProbePlan,
    MD04ProbeRunArtifact,
    MD05DecisionArtifact,
    ProbeCaseOutcome,
    ProbeEvaluatorLabel,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MD_METRIC_KEYS,
    MemoryDynamicsRunReport,
    zero_execution_state,
)


_PRIMARY = "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"


def _key(*values: int) -> StableRecordKey:
    return StableRecordKey(tuple(values))


def _expected_rows() -> tuple[
        tuple[int, int, str, tuple[int, ...], tuple[int, ...], int, int], ...]:
    """冻结 evaluator-only 正解；candidate ordinal 是 plan 外部稳定身份。"""
    return (
        (1, 1, "ACCESS_BLOCKED", (), (), 0, 0),
        (2, 1, "BUDGET_EXHAUSTED", (), (), 0, 0),
        (3, 1, "GROUNDING_BLOCKED", (), (), 0, 0),
        (4, 1, "RESOLVED", (102,), (), 0, 1),
        (5, 1, "RESOLVED", (102,), (), 1, 1),
        (6, 1, "GROUNDING_BLOCKED", (), (1,), 0, 0),
        (7, 1, "CLARIFY", (), (), 0, 1),
        (8, 1, "RESOLVED", (1,), (), 0, 0),
        (9, 1, "RESOLVED", (1,), (), 0, 0),
        (10, 1, "RESOLVED", (1,), (), 0, 0),
        (11, 1, "RESOLVED", (101,), (), 0, 1),
        (11, 2, "RESOLVED", (102,), (), 0, 1),
        (11, 3, "RESOLVED", (103,), (), 0, 1),
        (12, 1, "UNKNOWN", (), (), 0, 0),
    )


def build_md05_labels(plan: MD04ProbePlan) -> tuple[ProbeEvaluatorLabel, ...]:
    """在 evaluator owner 内建立 held-out 标签并核 plan 引用存在。"""
    if not isinstance(plan, MD04ProbePlan):
        raise MD04ProbeContractError("MD-05 labels plan 类型错误")
    cases = {item.case_key.components[-1]: item for item in plan.cases}
    labels = []
    for ordinal, row in enumerate(_expected_rows(), start=1):
        case_ordinal, center_ordinal, status, correct, forbidden, structure, distance = row
        case = cases.get(case_ordinal)
        if case is None:
            raise MD04ProbeContractError("MD-05 label 引用缺失 case")
        center_key = _key(44140, case_ordinal, center_ordinal)
        if center_key not in {item.center_key for item in case.center_refs}:
            raise MD04ProbeContractError("MD-05 label 引用缺失 center")
        correct_keys = tuple(sorted(
            _key(44200, case_ordinal, value) for value in correct))
        forbidden_keys = tuple(sorted(
            _key(44200, case_ordinal, value) for value in forbidden))
        all_candidates = {
            item.candidate_key
            for item in (*case.hot_candidates, *case.cold_candidates)
        }
        if not set((*correct_keys, *forbidden_keys)).issubset(all_candidates):
            raise MD04ProbeContractError("MD-05 label candidate 不在 plan")
        labels.append(ProbeEvaluatorLabel(
            _key(44600, ordinal),
            _key(44610, 1),
            case.case_key,
            center_key,
            status,
            correct_keys,
            forbidden_keys,
            _key(44620, case_ordinal, center_ordinal),
            structure,
            distance,
            "held_out",
        ))
    return tuple(sorted(labels))


def _labels_by_case(
        labels: tuple[ProbeEvaluatorLabel, ...],
        ) -> dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]]:
    grouped: dict[StableRecordKey, list[ProbeEvaluatorLabel]] = defaultdict(list)
    for label in labels:
        grouped[label.case_key].append(label)
    return {
        key: tuple(sorted(values)) for key, values in grouped.items()
    }


def _status_map(outcome: ProbeCaseOutcome) -> dict[StableRecordKey, str]:
    return {
        item.center_key: item.status for item in outcome.stop_decisions
    }


def _case_quality(
        outcome: ProbeCaseOutcome,
        labels: tuple[ProbeEvaluatorLabel, ...],
        ) -> dict[str, int]:
    statuses = _status_map(outcome)
    adopted_or_generated = set((
        *outcome.adopted_candidate_keys,
        *outcome.generated_candidate_keys,
    ))
    correct = 0
    wrong = 0
    classification = 0
    missed_refute = 0
    generation_safe = 0
    generation_recoverable = 0
    expected_candidates = {
        key for label in labels for key in label.correct_candidate_keys
    }
    correct = len(adopted_or_generated & expected_candidates)
    wrong = len(adopted_or_generated - expected_candidates)
    for label in labels:
        actual_status = statuses.get(label.center_key)
        classification += int(actual_status == label.expected_status)
        if label.expected_status == "CLARIFY" and actual_status != "CLARIFY":
            missed_refute += 1
        if label.forbidden_generation_keys:
            safe = not set(label.forbidden_generation_keys).intersection(
                outcome.generated_candidate_keys)
            generation_safe += int(safe)
            generation_recoverable += int(safe)
    return {
        "ADOPTED_CORRECT": correct,
        "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION": classification,
        "GENERATION_ADDRESSEE_RECOVERABILITY": generation_recoverable,
        "GENERATION_SEMANTIC_PRESERVATION": generation_safe,
        "MISSED_REFUTE": missed_refute,
        "WRONG_ADOPTION": wrong,
    }


def _hard_failures(
        outcomes: tuple[ProbeCaseOutcome, ...],
        labels_by_case: dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]],
        ) -> tuple[str, ...]:
    failures: set[str] = set()
    for outcome in outcomes:
        statuses = _status_map(outcome)
        audit = outcome.audit_values.to_value()
        for label in labels_by_case[outcome.case_key]:
            actual = statuses[label.center_key]
            if (label.expected_status == "BUDGET_EXHAUSTED"
                    and actual == "UNKNOWN"):
                failures.add("BUDGET_NOT_UNKNOWN")
            if set(label.forbidden_generation_keys).intersection(
                    outcome.generated_candidate_keys):
                failures.update({
                    "HARD_VETO_NOT_OFFSET",
                    "UNAUTHORIZED_GENERATION_ZERO",
                })
        if audit["held_out_train_overlap_count"]:
            failures.add("HELD_OUT_COMBINATION_NOT_TRAIN")
        if audit["host_learning_write_count"]:
            failures.add("HOLDOUT_EVALUATOR_HOST_WRITES_ZERO")
        if (audit["unrelated_revision_change_count"]
                or not audit["unaffected_projection_bit_identical"]):
            failures.add("UNRELATED_REVISION_CHANGES_ZERO")
    return tuple(sorted(failures))


def _metrics(
        outcomes: tuple[ProbeCaseOutcome, ...],
        labels_by_case: dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]],
        ) -> CanonicalJsonObject:
    values = {key: 0 for key in MD_METRIC_KEYS}
    for outcome in outcomes:
        quality = _case_quality(outcome, labels_by_case[outcome.case_key])
        for key, value in quality.items():
            values[key] += value
        query = outcome.query_metrics.to_value()
        audit = outcome.audit_values.to_value()
        values["AGENDA_ENTRIES"] += sum(
            len(item.agenda_candidate_keys)
            for item in outcome.receipt_records)
        values["CONSUMED_OBJECTS"] += sum(
            len(item.consumed_candidate_keys)
            for item in outcome.receipt_records)
        values["LOGIC_STEPS"] += len(outcome.receipt_records)
        values["OPENED_PAGE_SEGMENT"] += query["segment_reads"]
        values["RECOMPUTED_OBJECTS"] += sum(
            item.recompute_count for item in outcome.receipt_records)
        values["SCANNED_OBJECTS"] += query["page_in_records"] + sum(
            item.scanned_object_count
            for item in outcome.receipt_records
            if item.channel_key != "L4_SEALED_PAGE")
        values["OLD_OBSERVATION_EVIDENCE_PRESERVED"] += audit[
            "old_evidence_preserved"]
        values["OWNER_SCOPE_VERSION_VIOLATION"] += audit[
            "host_learning_write_count"]
        values["RECEIPT_COMPLETENESS"] += len(outcome.receipt_records)
        values["TEACHER_HELD_OUT_LEAKAGE"] += (
            audit["teacher_call_count"]
            + audit["held_out_train_overlap_count"])
        values["UNAFFECTED_PROJECTION_BIT_IDENTITY"] += audit[
            "unaffected_projection_bit_identical"]
    return CanonicalJsonObject.from_value(values)


def _report(
        strategy: str,
        outcomes: tuple[ProbeCaseOutcome, ...],
        labels_by_case: dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]],
        ) -> MemoryDynamicsRunReport:
    failures = _hard_failures(outcomes, labels_by_case)
    metrics = _metrics(outcomes, labels_by_case)
    quality = metrics.to_value()
    total_labels = sum(len(labels_by_case[item.case_key]) for item in outcomes)
    quality_complete = (
        quality["CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION"] == total_labels
        and quality["WRONG_ADOPTION"] == 0
        and quality["MISSED_REFUTE"] == 0
    )
    decision = "PASS" if not failures and quality_complete else "REJECT"
    center_keys = tuple(sorted({
        item.center_key
        for outcome in outcomes for item in outcome.stop_decisions
    }))
    profile_keys = tuple(sorted({
        _key(44170, outcome.case_key.components[-1])
        for outcome in outcomes
    }))
    receipt_keys = tuple(sorted({
        item.receipt_key
        for outcome in outcomes for item in outcome.receipt_records
    }))
    decision_keys = tuple(sorted({
        item.decision_key
        for outcome in outcomes for item in outcome.stop_decisions
    }))
    return MemoryDynamicsRunReport(
        FORMAT_VERSION,
        f"MD-05-{strategy}-run-report-v1",
        _key(44630, MD_BASELINE_KEYS.index(strategy) + 1),
        MD04_PREREGISTRATION_VERSION,
        strategy,
        "COMPLETE",
        center_keys,
        profile_keys,
        receipt_keys,
        decision_keys,
        metrics,
        failures,
        1,
        decision,
        zero_execution_state(),
    )


def _quality_vector(report: MemoryDynamicsRunReport) -> dict[str, int]:
    values = report.metric_values.to_value()
    return {
        key: values[key] for key in (
            "ADOPTED_CORRECT",
            "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION",
            "GENERATION_ADDRESSEE_RECOVERABILITY",
            "GENERATION_SEMANTIC_PRESERVATION",
            "MISSED_REFUTE",
            "WRONG_ADOPTION",
        )
    }


def _no_quality_regression(
        candidate: MemoryDynamicsRunReport,
        baselines: tuple[MemoryDynamicsRunReport, ...],
        ) -> int:
    primary = _quality_vector(candidate)
    higher = {
        "ADOPTED_CORRECT",
        "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION",
        "GENERATION_ADDRESSEE_RECOVERABILITY",
        "GENERATION_SEMANTIC_PRESERVATION",
    }
    lower = {"MISSED_REFUTE", "WRONG_ADOPTION"}
    for baseline in baselines:
        values = _quality_vector(baseline)
        if any(primary[key] < values[key] for key in higher):
            return 0
        if any(primary[key] > values[key] for key in lower):
            return 0
    return 1


def _challenge_improvements(
        runs: MD04ProbeRunArtifact,
        labels_by_case: dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]],
        ) -> int:
    grouped = {
        strategy: {
            item.case_key: item for item in runs.strategy_outcomes
            if item.strategy_key == strategy
        }
        for strategy in MD_BASELINE_KEYS
    }
    improved = 0
    for case_key, primary in grouped[_PRIMARY].items():
        primary_quality = _case_quality(primary, labels_by_case[case_key])
        primary_score = (
            primary_quality["CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION"],
            primary_quality["ADOPTED_CORRECT"],
            -primary_quality["WRONG_ADOPTION"],
            -primary_quality["MISSED_REFUTE"],
        )
        if any(primary_score > (
                _case_quality(grouped[strategy][case_key],
                              labels_by_case[case_key])[
                    "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION"],
                _case_quality(grouped[strategy][case_key],
                              labels_by_case[case_key])["ADOPTED_CORRECT"],
                -_case_quality(grouped[strategy][case_key],
                               labels_by_case[case_key])["WRONG_ADOPTION"],
                -_case_quality(grouped[strategy][case_key],
                               labels_by_case[case_key])["MISSED_REFUTE"],
                ) for strategy in MD_BASELINE_KEYS if strategy != _PRIMARY):
            improved += 1
    return improved


def _resource_growth_violations(runs: MD04ProbeRunArtifact) -> int:
    grouped: dict[StableRecordKey, list[ProbeCaseOutcome]] = defaultdict(list)
    for outcome in runs.scale_outcomes:
        grouped[outcome.case_key].append(outcome)
    violations = 0
    for outcomes in grouped.values():
        ordered = sorted(outcomes, key=lambda item: item.scale_factor)
        for key in (
                "cold_read_bytes", "page_in_records", "peak_hot_objects",
                "segment_reads"):
            values = [item.query_metrics.to_value()[key] for item in ordered]
            if len(set(values)) != 1:
                violations += 1
    return violations


def _far_source_chain_recovered(
        outcomes: tuple[ProbeCaseOutcome, ...],
        plan: MD04ProbePlan,
        ) -> int:
    case = next(item for item in plan.cases
                if item.case_key.components[-1] == 4)
    correct = next(item for item in case.cold_candidates
                   if item.candidate_key == _key(44200, 4, 102))
    outcome = next(item for item in outcomes
                   if item.case_key == case.case_key)
    return int(
        correct.candidate_key in outcome.adopted_candidate_keys
        and any(
            correct.evidence_key in receipt.evidence_keys
            and correct.source_key in receipt.dependency_keys
            for receipt in outcome.receipt_records)
    )


def _ablation_evidence(
        runs: MD04ProbeRunArtifact,
        labels_by_case: dict[StableRecordKey, tuple[ProbeEvaluatorLabel, ...]],
        ) -> CanonicalJsonObject:
    primary = tuple(
        item for item in runs.strategy_outcomes if item.strategy_key == _PRIMARY)
    primary_metrics = _metrics(primary, labels_by_case).to_value()
    result: dict[str, list[str]] = {}
    for ablation in runs.ablation_outcomes:
        metrics = _metrics(ablation.outcomes, labels_by_case).to_value()
        dimensions = []
        for key in MD_METRIC_KEYS:
            if metrics[key] != primary_metrics[key]:
                dimensions.append(key)
        primary_audit = {
            "UNRELATED_REVISION_CHANGES": sum(
                item.audit_values.to_value()["unrelated_revision_change_count"]
                for item in primary),
            "UNAUTHORIZED_GENERATION": sum(
                len(item.generated_candidate_keys) for item in primary
                if item.case_key.components[-1] == 6),
            "COLD_READ_BYTES": sum(
                item.query_metrics.to_value()["cold_read_bytes"]
                for item in primary),
        }
        ablation_audit = {
            "UNRELATED_REVISION_CHANGES": sum(
                item.audit_values.to_value()["unrelated_revision_change_count"]
                for item in ablation.outcomes),
            "UNAUTHORIZED_GENERATION": sum(
                len(item.generated_candidate_keys)
                for item in ablation.outcomes
                if item.case_key.components[-1] == 6),
            "COLD_READ_BYTES": sum(
                item.query_metrics.to_value()["cold_read_bytes"]
                for item in ablation.outcomes),
        }
        dimensions.extend(
            key for key in sorted(ablation_audit)
            if ablation_audit[key] != primary_audit[key])
        result[ablation.ablation_key] = sorted(set(dimensions))
    return CanonicalJsonObject.from_value(result)


def evaluate_md05_probe(
        plan: MD04ProbePlan,
        runs: MD04ProbeRunArtifact,
        *,
        plan_relative_path: str,
        plan_sha256: str,
        run_relative_path: str,
        run_sha256: str,
        ) -> MD05DecisionArtifact:
    """只读消费 raw outcomes，形成四报告、消融证据和合取决断。"""
    if not isinstance(plan, MD04ProbePlan):
        raise MD04ProbeContractError("MD-05 plan 类型错误")
    if not isinstance(runs, MD04ProbeRunArtifact):
        raise MD04ProbeContractError("MD-05 runs 类型错误")
    labels = build_md05_labels(plan)
    labels_by_case = _labels_by_case(labels)
    outcomes_by_strategy = {
        strategy: tuple(
            item for item in runs.strategy_outcomes
            if item.strategy_key == strategy)
        for strategy in MD_BASELINE_KEYS
    }
    reports = tuple(
        _report(strategy, outcomes_by_strategy[strategy], labels_by_case)
        for strategy in MD_BASELINE_KEYS
    )
    primary_report = reports[MD_BASELINE_KEYS.index(_PRIMARY)]
    other_reports = tuple(item for item in reports
                          if item.strategy_key != _PRIMARY)
    primary_outcomes = outcomes_by_strategy[_PRIMARY]
    primary_failures = _hard_failures(primary_outcomes, labels_by_case)
    ablations = _ablation_evidence(runs, labels_by_case)
    comparison = CanonicalJsonObject.from_value({
        "challenge_improvement_count": _challenge_improvements(
            runs, labels_by_case),
        "far_source_chain_recovered": _far_source_chain_recovered(
            primary_outcomes, plan),
        "held_out_combination_overlap_count": sum(
            item.audit_values.to_value()["held_out_train_overlap_count"]
            for item in primary_outcomes),
        "irrelevant_query_cold_read_bytes": max(
            item.query_metrics.to_value()["cold_read_bytes"]
            for item in runs.scale_outcomes
            if item.case_key.components[-1] == 9),
        "no_quality_regression": _no_quality_regression(
            primary_report, other_reports),
        "resource_growth_violation_count": _resource_growth_violations(runs),
        "time_advance_full_store_rewrites": sum(
            item.audit_values.to_value()["full_store_rewrite_count"]
            for item in primary_outcomes),
    })
    pass_conditions = (
        not primary_failures
        and primary_report.probe_decision == "PASS"
        and comparison.to_value()["no_quality_regression"] == 1
        and comparison.to_value()["challenge_improvement_count"] >= 1
        and comparison.to_value()["resource_growth_violation_count"] == 0
        and comparison.to_value()["irrelevant_query_cold_read_bytes"] == 0
        and comparison.to_value()["time_advance_full_store_rewrites"] == 0
        and comparison.to_value()["held_out_combination_overlap_count"] == 0
        and comparison.to_value()["far_source_chain_recovered"] == 1
        and all(ablations.to_value()[key] for key in MD04_ABLATION_KEYS)
    )
    return MD05DecisionArtifact(
        FORMAT_VERSION,
        MD05_DECISION_VERSION,
        plan_relative_path,
        plan_sha256,
        run_relative_path,
        run_sha256,
        labels,
        reports,
        comparison,
        ablations,
        primary_failures,
        "PASS" if pass_conditions else "REJECT",
        1,
        0,
        zero_execution_state(),
    )


__all__ = ["build_md05_labels", "evaluate_md05_probe"]
