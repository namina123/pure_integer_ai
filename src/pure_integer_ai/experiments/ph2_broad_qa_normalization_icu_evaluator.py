"""按冻结 ICU v2 协议评估 normalization candidate clone。

本模块只包含纯判分逻辑。它不负责文件发布、唯一运行 guard，也不读取 reserve。
direct mapping、defeater、context application 和 runtime 分开出账。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_candidate_clone import (
    NormalizationCandidateCloneProgram,
    execute_normalization_candidate_clone,
    reference_normalization_candidate_rewrite,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_protocol import (
    NORMALIZATION_ICU_EVALUATION_DIMENSIONS,
    NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_ICU_EVALUATION_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_records_v3 import (
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_ICU_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_EVALUATION_REPORT_V1")
NORMALIZATION_EVALUATION_OUTCOMES = ("FAIL", "NE", "PASS")
_DIMENSION_ORDER = (
    "DIRECT_MAPPING_CONSISTENCY",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "CONTEXT_APPLICATION_OUTCOME",
    "RUNTIME_PRODUCTION_BEHAVIOR",
)
_EVALUATION_RECORD_FIELDS = {
    "evaluation_id", "expected_output", "format_version", "input_text",
    "mapping_kind", "record_kind", "source_arrow",
    "source_pack_manifest_sha256", "source_rule_byte_end",
    "source_rule_byte_start", "source_rule_line_end_ordinal",
    "source_rule_line_start_ordinal", "source_rule_sha256", "split",
}


def _sha256(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _metrics(values: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """把非负整数计数规范化为排序 tuple。"""
    if (not isinstance(values, dict) or not values
            or any(not isinstance(key, str) or not key
                   or type(value) is not int or value < 0
                   for key, value in values.items())):
        raise BroadQaExternalDataError(
            "normalization evaluation metrics 非法")
    return tuple(sorted(values.items()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationEvaluationDimensionResult:
    """一个 bearing dimension 的原始整数计数和结论。"""

    dimension_key: str
    outcome: str
    metrics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """要求维度、结论和计数规范。"""
        if (self.dimension_key not in _DIMENSION_ORDER
                or self.outcome not in NORMALIZATION_EVALUATION_OUTCOMES
                or not isinstance(self.metrics, tuple)
                or self.metrics != tuple(sorted(set(self.metrics)))
                or any(not isinstance(key, str) or not key
                       or type(value) is not int or value < 0
                       for key, value in self.metrics)):
            raise BroadQaExternalDataError(
                "normalization evaluation dimension result 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范维度结果。"""
        return {
            "dimension_key": self.dimension_key,
            "metrics": {key: value for key, value in self.metrics},
            "outcome": self.outcome,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationIcuEvaluationReport:
    """冻结 protocol 与 candidate clone 的四维评测报告。"""

    protocol_manifest_sha256: str
    source_pack_manifest_sha256: str
    rule_pack_manifest_sha256: str
    candidate_clone_sha256: str
    evaluation_inventory_sha256: str
    evaluation_record_count: int
    dimensions: tuple[NormalizationEvaluationDimensionResult, ...]
    overall_outcome: str
    learned_pack_read_count: int
    evaluation_run_count: int
    production_enabled: int
    reserve_label_read_count: int
    teacher_api_llm_call_count: int

    def __post_init__(self) -> None:
        """要求四维完整、有序且所有禁用计数未漂移。"""
        for label, value in (
                ("protocol manifest", self.protocol_manifest_sha256),
                ("source pack manifest", self.source_pack_manifest_sha256),
                ("rule pack manifest", self.rule_pack_manifest_sha256),
                ("candidate clone", self.candidate_clone_sha256),
                ("evaluation inventory", self.evaluation_inventory_sha256)):
            _sha256(value, label=f"normalization evaluation {label}")
        outcomes = tuple(item.outcome for item in self.dimensions)
        expected_overall = (
            "FAIL" if "FAIL" in outcomes else
            "NE" if "NE" in outcomes else "PASS")
        if (type(self.evaluation_record_count) is not int
                or self.evaluation_record_count <= 0
                or tuple(item.dimension_key for item in self.dimensions)
                != _DIMENSION_ORDER
                or self.overall_outcome != expected_overall
                or type(self.learned_pack_read_count) is not int
                or self.learned_pack_read_count != 1
                or type(self.evaluation_run_count) is not int
                or self.evaluation_run_count != 1
                or any(type(value) is not int or value != 0 for value in (
                    self.production_enabled,
                    self.reserve_label_read_count,
                    self.teacher_api_llm_call_count))):
            raise BroadQaExternalDataError(
                "normalization evaluation report 边界漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范正式或 synthetic 报告。"""
        return {
            "artifact_kind": NORMALIZATION_ICU_EVALUATION_REPORT_KIND,
            "candidate_clone_sha256": self.candidate_clone_sha256,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evaluation_inventory_sha256": self.evaluation_inventory_sha256,
            "evaluation_record_count": self.evaluation_record_count,
            "evaluation_run_count": self.evaluation_run_count,
            "format_version": 1,
            "learned_pack_read_count": self.learned_pack_read_count,
            "overall_outcome": self.overall_outcome,
            "production_enabled": self.production_enabled,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "reserve_label_read_count": self.reserve_label_read_count,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "source_pack_manifest_sha256": self.source_pack_manifest_sha256,
            "teacher_api_llm_call_count": self.teacher_api_llm_call_count,
        }

    def sha256(self) -> str:
        """返回规范报告摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _validate_protocol(
        manifest: dict[str, object],
        ) -> tuple[str, str, str]:
    """核验调用方传入的是未消费、未运行的冻结 v2 协议。"""
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind")
            != NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND
            or manifest.get("status") != NORMALIZATION_ICU_EVALUATION_STATUS
            or manifest.get("format_version") != 2
            or not _strict_equal(
                manifest.get("dimensions"),
                NORMALIZATION_ICU_EVALUATION_DIMENSIONS)
            or not _strict_equal(
                manifest.get("metric_contract"),
                NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT)
            or any(manifest.get(name) != 0 for name in (
                "evaluation_run_count", "learned_pack_read_count",
                "mastery_claimed", "production_enabled",
                "reserve_labels_published"))):
        raise BroadQaExternalDataError(
            "normalization evaluator protocol 已消费或漂移")
    return (
        _sha256(
            manifest.get("manifest_sha256"),
            label="normalization evaluator protocol manifest"),
        _sha256(
            manifest.get("source_pack_manifest_sha256"),
            label="normalization evaluator source manifest"),
        _sha256(
            manifest.get("evaluation_inventory", {}).get("sha256")
            if isinstance(manifest.get("evaluation_inventory"), dict)
            else None,
            label="normalization evaluator inventory"),
    )


def _validate_evaluation_inventory(
        records: tuple[dict[str, object], ...],
        *,
        source_pack_manifest_sha256: str,
        expected_count: int,
        ) -> None:
    """拒绝 reserve、未知字段、错来源或非规范 evaluation identity。"""
    if (not isinstance(records, tuple) or len(records) != expected_count
            or not records):
        raise BroadQaExternalDataError(
            "normalization evaluator inventory 数量漂移")
    ids = []
    for record in records:
        if (not isinstance(record, dict)
                or set(record) != _EVALUATION_RECORD_FIELDS
                or record["split"] != "EVALUATION"
                or record["format_version"] != 1
                or record["source_pack_manifest_sha256"]
                != source_pack_manifest_sha256
                or not isinstance(record["input_text"], str)
                or not record["input_text"]
                or not isinstance(record["expected_output"], str)
                or not record["expected_output"]
                or record["mapping_kind"]
                not in {"CHARACTER", "IDENTITY", "PHRASE"}):
            raise BroadQaExternalDataError(
                "normalization evaluator inventory 含 reserve/未知字段或来源漂移")
        ids.append(_sha256(
            record["evaluation_id"],
            label="normalization evaluator evaluation id"))
    if len(set(ids)) != len(ids):
        raise BroadQaExternalDataError(
            "normalization evaluator evaluation identity 重复")


def _trial_by_id(
        trials: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """建立 TRAIN_SOURCE trial 身份索引并拒绝重复。"""
    result = {}
    for trial in trials:
        trial_id = trial.get("trial_id") if isinstance(trial, dict) else None
        if (not isinstance(trial_id, str) or trial_id in result):
            raise BroadQaExternalDataError(
                "normalization evaluator context trial identity 漂移")
        result[trial_id] = trial
    return result


def _dimension(
        key: str,
        outcome: str,
        values: dict[str, int],
        ) -> NormalizationEvaluationDimensionResult:
    """构造规范 bearing dimension 结果。"""
    return NormalizationEvaluationDimensionResult(
        key, outcome, _metrics(values))


def _direct_mapping_result(
        program: NormalizationCandidateCloneProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> NormalizationEvaluationDimensionResult:
    """用独立线性 reference 与 clone 对照计算 direct mapping 维度。"""
    values = {
        "applicable_independent_rule_count": 0,
        "independent_false_accept_count": 0,
        "independent_false_reject_count": 0,
        "independent_support_count": 0,
        "unresolved_independent_conflict_count": 0,
    }
    if program is None:
        return _dimension("DIRECT_MAPPING_CONSISTENCY", "NE", values)
    inputs = {item.input_codepoint for item in program.rules}
    for record in records:
        input_text = record["input_text"]
        if not any(ord(item) in inputs for item in input_text):
            continue
        values["applicable_independent_rule_count"] += 1
        expected = record["expected_output"]
        reference = reference_normalization_candidate_rewrite(
            program, input_text)
        actual = execute_normalization_candidate_clone(program, input_text)
        supported = reference.output_text == expected
        if supported:
            values["independent_support_count"] += 1
            if actual.output_text != expected:
                values["independent_false_reject_count"] += 1
        else:
            values["unresolved_independent_conflict_count"] += 1
        if (actual.output_text != input_text
                and actual.output_text != expected):
            values["independent_false_accept_count"] += 1
    if values["applicable_independent_rule_count"] == 0:
        outcome = "NE"
    elif (values["independent_support_count"] < 1
          or values["independent_false_accept_count"] > 0
          or values["independent_false_reject_count"] > 0
          or values["unresolved_independent_conflict_count"] > 0):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("DIRECT_MAPPING_CONSISTENCY", outcome, values)


def _defeater_result(
        program: NormalizationCandidateCloneProgram | None,
        ) -> NormalizationEvaluationDimensionResult:
    """核对 declared defeater 是否全部形成非 identity context 程序。"""
    values = {
        "declared_defeater_count": 0,
        "executable_defeater_count": 0,
        "identity_only_defeater_count": 0,
        "malformed_defeater_count": 0,
    }
    if program is None:
        return _dimension(
            "DEFEATER_REPRESENTATION_EXECUTABILITY", "NE", values)
    declared = {
        key for rule in program.rules for key in rule.declared_defeater_keys}
    executable = {
        item.defeater_key
        for rule in program.rules for item in rule.defeaters}
    values["declared_defeater_count"] = len(declared)
    values["executable_defeater_count"] = len(executable)
    values["malformed_defeater_count"] = len(declared - executable)
    if declared and not executable:
        values["identity_only_defeater_count"] = len(declared)
    if not declared:
        outcome = "NE"
    elif (values["identity_only_defeater_count"] > 0
          or values["malformed_defeater_count"] > 0
          or executable != declared):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension(
        "DEFEATER_REPRESENTATION_EXECUTABILITY", outcome, values)


def _context_result(
        program: NormalizationCandidateCloneProgram | None,
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        trials: tuple[dict[str, object], ...],
        ) -> NormalizationEvaluationDimensionResult:
    """在 support/refute phrase 的原 offset 重放 context application。"""
    values = {
        "context_false_accept_count": 0,
        "context_false_reject_count": 0,
        "defeater_hit_count": 0,
        "negative_context_case_count": 0,
        "positive_context_case_count": 0,
    }
    if program is None:
        return _dimension("CONTEXT_APPLICATION_OUTCOME", "NE", values)
    indexed = _trial_by_id(trials)
    for accepted in accepted_rules:
        for evidence in accepted.evidence_commitments:
            trial = indexed.get(evidence.trial_id)
            if (not isinstance(trial, dict)
                    or trial.get("qualification_kind")
                    != "SOURCE_REPLAY_SUPPORT"):
                raise BroadQaExternalDataError(
                    "normalization evaluator positive context 来源漂移")
            result = execute_normalization_candidate_clone(
                program, trial.get("phrase_source"))
            offset = trial.get("source_codepoint_offset")
            if type(offset) is not int or not 0 <= offset < len(result.steps):
                raise BroadQaExternalDataError(
                    "normalization evaluator positive context offset 漂移")
            values["positive_context_case_count"] += 1
            if (result.steps[offset].output_codepoint
                    != trial.get("observed_output_codepoint")):
                values["context_false_reject_count"] += 1
    for rejected in rejected_trials:
        trial = indexed.get(rejected.trial_id)
        if (not isinstance(trial, dict)
                or trial.get("qualification_kind") != "SOURCE_REPLAY_REFUTE"):
            raise BroadQaExternalDataError(
                "normalization evaluator negative context 来源漂移")
        result = execute_normalization_candidate_clone(
            program, trial.get("phrase_source"))
        offset = trial.get("source_codepoint_offset")
        if type(offset) is not int or not 0 <= offset < len(result.steps):
            raise BroadQaExternalDataError(
                "normalization evaluator negative context offset 漂移")
        step = result.steps[offset]
        values["negative_context_case_count"] += 1
        if rejected.trial_id in step.defeater_trial_ids:
            values["defeater_hit_count"] += 1
        if step.output_codepoint == rejected.candidate.output_codepoint:
            values["context_false_accept_count"] += 1
    if (values["positive_context_case_count"] == 0
            or values["negative_context_case_count"] == 0):
        outcome = "NE"
    elif (values["context_false_accept_count"] > 0
          or values["context_false_reject_count"] > 0
          or values["defeater_hit_count"]
          != values["negative_context_case_count"]):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("CONTEXT_APPLICATION_OUTCOME", outcome, values)


def _runtime_result(
        program: NormalizationCandidateCloneProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> NormalizationEvaluationDimensionResult:
    """对全部 evaluation input 双遍执行并核对规范结果确定性。"""
    values = {
        "public_production_enabled": 0,
        "runtime_exception_count": 0,
        "runtime_executed_input_count": 0,
        "runtime_replay_mismatch_count": 0,
    }
    if program is None:
        return _dimension("RUNTIME_PRODUCTION_BEHAVIOR", "NE", values)
    values["public_production_enabled"] = program.production_enabled
    for record in records:
        try:
            first = execute_normalization_candidate_clone(
                program, record["input_text"])
            second = execute_normalization_candidate_clone(
                program, record["input_text"])
        except (BroadQaExternalDataError, TypeError, ValueError):
            values["runtime_exception_count"] += 1
            continue
        values["runtime_executed_input_count"] += 1
        if first.sha256() != second.sha256():
            values["runtime_replay_mismatch_count"] += 1
    if (values["public_production_enabled"] != 0
            or values["runtime_exception_count"] > 0
            or values["runtime_replay_mismatch_count"] > 0
            or values["runtime_executed_input_count"] != len(records)):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("RUNTIME_PRODUCTION_BEHAVIOR", outcome, values)


def evaluate_normalization_icu_candidate(
        *,
        protocol_manifest: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        contrastive_trials: tuple[dict[str, object], ...],
        program: NormalizationCandidateCloneProgram | None,
        ) -> NormalizationIcuEvaluationReport:
    """按冻结 v2 门生成四维报告，不读取 reserve 或改变生产状态。"""
    protocol_sha, source_sha, inventory_sha = _validate_protocol(
        protocol_manifest)
    expected_count = protocol_manifest.get("summary", {}).get(
        "evaluation_record_count") if isinstance(
            protocol_manifest.get("summary"), dict) else None
    if type(expected_count) is not int or expected_count <= 0:
        raise BroadQaExternalDataError(
            "normalization evaluator protocol summary 漂移")
    _validate_evaluation_inventory(
        evaluation_records,
        source_pack_manifest_sha256=source_sha,
        expected_count=expected_count,
    )
    if (not isinstance(accepted_rules, tuple)
            or any(not isinstance(item, BroadQaNormalizationAcceptedRuleV3)
                   for item in accepted_rules)
            or not isinstance(rejected_trials, tuple)
            or any(not isinstance(item, BroadQaNormalizationRejectedTrialV3)
                   for item in rejected_trials)
            or not isinstance(contrastive_trials, tuple)):
        raise BroadQaExternalDataError(
            "normalization evaluator learned records 类型漂移")
    dimensions = (
        _direct_mapping_result(program, evaluation_records),
        _defeater_result(program),
        _context_result(
            program, accepted_rules, rejected_trials, contrastive_trials),
        _runtime_result(program, evaluation_records),
    )
    outcomes = tuple(item.outcome for item in dimensions)
    overall = (
        "FAIL" if "FAIL" in outcomes else
        "NE" if "NE" in outcomes else "PASS")
    if program is None:
        clone_sha = "0" * 64
        rule_pack_sha = "0" * 64
    else:
        clone_sha = program.sha256()
        rule_pack_sha = program.rule_pack_manifest_sha256
    return NormalizationIcuEvaluationReport(
        protocol_sha,
        source_sha,
        rule_pack_sha,
        clone_sha,
        inventory_sha,
        len(evaluation_records),
        dimensions,
        overall,
        1,
        1,
        0,
        0,
        0,
    )


__all__ = [
    "NORMALIZATION_EVALUATION_OUTCOMES",
    "NORMALIZATION_ICU_EVALUATION_REPORT_KIND",
    "NormalizationEvaluationDimensionResult",
    "NormalizationIcuEvaluationReport",
    "evaluate_normalization_icu_candidate",
]
