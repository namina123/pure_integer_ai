"""按冻结六维协议评估 normalization successor candidate clone。

本模块只做纯判分，不读路径、不发布 guard、不读取 reserve。局部 transfer、整句
coverage、来源冲突、context facility 和 runtime 均独立出账。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_candidate_clone import (
    NormalizationSuccessorCandidateProgram,
    execute_normalization_successor_candidate,
    reference_normalization_successor_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_protocol import (
    NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS,
    NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
    NORMALIZATION_SUCCESSOR_EVALUATION_STATUS,
    NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
    NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_SUCCESSOR_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_EVALUATION_REPORT_V1")
NORMALIZATION_SUCCESSOR_EVALUATION_OUTCOMES = ("FAIL", "NE", "PASS")
NORMALIZATION_SUCCESSOR_DIMENSION_ORDER = (
    "LOCAL_MAPPING_TRANSFER",
    "END_TO_END_COVERAGE",
    "SOURCE_POLICY_CONFLICT",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "INDEPENDENT_CONTEXT_TRANSFER",
    "RUNTIME_PRODUCTION_BEHAVIOR",
)
SOURCE_POLICY_SCOPES = (OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE)


def _sha_value(value: object, *, label: str) -> str:
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
    """把非负整数计数规范化为唯一排序 tuple。"""
    if (not isinstance(values, dict) or not values
            or any(not isinstance(key, str) or not key
                   or type(value) is not int or value < 0
                   for key, value in values.items())):
        raise BroadQaExternalDataError(
            "successor evaluation metrics 非法")
    return tuple(sorted(values.items()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorDimensionResult:
    """一个 bearing dimension 的规范整数计数和结论。"""

    dimension_key: str
    outcome: str
    metrics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """要求维度、结论和 metrics 完整规范。"""
        if (self.dimension_key not in NORMALIZATION_SUCCESSOR_DIMENSION_ORDER
                or self.outcome not in NORMALIZATION_SUCCESSOR_EVALUATION_OUTCOMES
                or self.metrics != tuple(sorted(set(self.metrics)))
                or any(not isinstance(key, str) or not key
                       or type(value) is not int or value < 0
                       for key, value in self.metrics)):
            raise BroadQaExternalDataError(
                "successor evaluation dimension result 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范维度结果。"""
        return {
            "dimension_key": self.dimension_key,
            "metrics": {key: value for key, value in self.metrics},
            "outcome": self.outcome,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationSuccessorEvaluationReport:
    """冻结 protocol、pack 和 clone 的六维唯一运行报告。"""

    protocol_manifest_sha256: str
    evaluation_source_pack_manifest_sha256: str
    rule_pack_manifest_sha256: str
    candidate_clone_sha256: str
    evaluation_inventory_sha256: str
    evaluation_record_count: int
    dimensions: tuple[NormalizationSuccessorDimensionResult, ...]
    overall_outcome: str
    runtime_result_sha256: str
    learned_pack_read_count: int
    evaluation_run_count: int
    production_enabled: int
    reserve_payload_read_count: int
    teacher_api_llm_call_count: int

    def __post_init__(self) -> None:
        """核验六维顺序、总结果、身份与全部零边界。"""
        for label, value in (
                ("protocol", self.protocol_manifest_sha256),
                ("evaluation source", self.evaluation_source_pack_manifest_sha256),
                ("rule pack", self.rule_pack_manifest_sha256),
                ("candidate clone", self.candidate_clone_sha256),
                ("evaluation inventory", self.evaluation_inventory_sha256),
                ("runtime result", self.runtime_result_sha256)):
            _sha_value(value, label=f"successor evaluation {label}")
        outcomes = tuple(item.outcome for item in self.dimensions)
        expected_overall = (
            "FAIL" if "FAIL" in outcomes else
            "NE" if "NE" in outcomes else "PASS")
        if (type(self.evaluation_record_count) is not int
                or self.evaluation_record_count <= 0
                or tuple(item.dimension_key for item in self.dimensions)
                != NORMALIZATION_SUCCESSOR_DIMENSION_ORDER
                or self.overall_outcome != expected_overall
                or type(self.learned_pack_read_count) is not int
                or self.learned_pack_read_count != 1
                or type(self.evaluation_run_count) is not int
                or self.evaluation_run_count != 1
                or any(type(value) is not int or value != 0 for value in (
                    self.production_enabled,
                    self.reserve_payload_read_count,
                    self.teacher_api_llm_call_count))):
            raise BroadQaExternalDataError(
                "successor evaluation report 边界漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范正式或 synthetic 六维报告。"""
        return {
            "artifact_kind": NORMALIZATION_SUCCESSOR_EVALUATION_REPORT_KIND,
            "candidate_clone_sha256": self.candidate_clone_sha256,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evaluation_inventory_sha256": self.evaluation_inventory_sha256,
            "evaluation_record_count": self.evaluation_record_count,
            "evaluation_run_count": self.evaluation_run_count,
            "evaluation_source_pack_manifest_sha256": (
                self.evaluation_source_pack_manifest_sha256),
            "format_version": 1,
            "learned_pack_read_count": self.learned_pack_read_count,
            "overall_outcome": self.overall_outcome,
            "production_enabled": self.production_enabled,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "reserve_payload_read_count": self.reserve_payload_read_count,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "runtime_result_sha256": self.runtime_result_sha256,
            "teacher_api_llm_call_count": self.teacher_api_llm_call_count,
        }

    def sha256(self) -> str:
        """返回规范报告摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _dimension(
        key: str,
        outcome: str,
        values: dict[str, int],
        ) -> NormalizationSuccessorDimensionResult:
    """构造一个规范 bearing dimension。"""
    return NormalizationSuccessorDimensionResult(key, outcome, _metrics(values))


def _validate_protocol(manifest: dict[str, object]) -> tuple[str, str, str, int]:
    """核验调用方传入未消费、学习前冻结的 successor 协议。"""
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind")
            != NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_KIND
            or manifest.get("status") != NORMALIZATION_SUCCESSOR_EVALUATION_STATUS
            or manifest.get("format_version") != 1
            or manifest.get("target_policy_scope")
            != NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE
            or not _strict_equal(
                manifest.get("dimensions"),
                NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS)
            or not _strict_equal(
                manifest.get("metric_contract"),
                NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT)
            or any(manifest.get(name) != 0 for name in (
                "candidate_pack_read_count", "evaluation_run_count",
                "failed_icu_evaluation_read_count", "learned_pack_read_count",
                "mastery_claimed", "production_enabled",
                "reserve_label_read_count", "successor_training_source_read_count",
                "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError(
            "successor evaluator protocol 已消费或冻结字段漂移")
    inventory = manifest.get("evaluation_inventory")
    summary = manifest.get("inventory_summary")
    if (not isinstance(inventory, dict) or not isinstance(summary, dict)
            or type(summary.get("evaluation_count")) is not int
            or summary["evaluation_count"] <= 0):
        raise BroadQaExternalDataError(
            "successor evaluator protocol inventory 漂移")
    return (
        _sha_value(manifest.get("manifest_sha256"), label="successor protocol"),
        _sha_value(
            manifest.get("source_pack_manifest_sha256"),
            label="successor evaluation source"),
        _sha_value(inventory.get("sha256"), label="successor inventory"),
        summary["evaluation_count"],
    )


def _validate_inventory(
        records: tuple[dict[str, object], ...],
        *,
        source_pack_manifest_sha256: str,
        expected_count: int,
        ) -> None:
    """拒绝 reserve、未知 family、错来源、错字段或非规范次序。"""
    if (not isinstance(records, tuple) or len(records) != expected_count
            or not records):
        raise BroadQaExternalDataError(
            "successor evaluator inventory 数量漂移")
    identities = []
    common = {
        "evaluation_id", "expected_output", "family_keys", "format_version",
        "input_text", "record_kind", "source_key", "source_license_id",
        "source_line_ordinal", "source_line_sha256",
        "source_pack_manifest_sha256", "source_record_id", "source_revision",
        "split", "target_policy_scope",
    }
    local_fields = common | {"source_property"}
    phrase_fields = common | {
        "context_sensitive", "position_expectations", "source_table"}
    for record in records:
        if (not isinstance(record, dict)
                or frozenset(record) not in {
                    frozenset(local_fields), frozenset(phrase_fields)}
                or record["split"] != "EVALUATION"
                or record["format_version"] != 1
                or record["record_kind"]
                != NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND
                or record["source_pack_manifest_sha256"]
                != source_pack_manifest_sha256
                or record["target_policy_scope"]
                != NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE
                or not isinstance(record["input_text"], str)
                or not record["input_text"]
                or not isinstance(record["expected_output"], str)
                or not record["expected_output"]
                or not isinstance(record["family_keys"], list)
                or not record["family_keys"]
                or type(record["source_line_ordinal"]) is not int
                or record["source_line_ordinal"] <= 0):
            raise BroadQaExternalDataError(
                "successor evaluator inventory 含 reserve/未知字段或来源漂移")
        _sha_value(record["source_line_sha256"], label="successor source line")
        _sha_value(record["source_record_id"], label="successor source record")
        if "source_property" in record:
            if (record["source_key"] != "UNICODE_UNIHAN"
                    or record["family_keys"] != ["LOCAL_MAPPING_TRANSFER"]
                    or len(record["input_text"]) != 1
                    or len(record["expected_output"]) != 1):
                raise BroadQaExternalDataError(
                    "successor evaluator local record 漂移")
        else:
            expected_families = ["END_TO_END_COVERAGE"]
            if record["context_sensitive"] == 1:
                expected_families.append("INDEPENDENT_CONTEXT_TRANSFER")
            if (record["source_key"] != "MEDIAWIKI_CORE"
                    or record["family_keys"] != expected_families
                    or type(record["context_sensitive"]) is not int
                    or record["context_sensitive"] not in {0, 1}
                    or not isinstance(record["position_expectations"], list)
                    or not record["position_expectations"]):
                raise BroadQaExternalDataError(
                    "successor evaluator phrase record 漂移")
            for position in record["position_expectations"]:
                if (not isinstance(position, dict)
                        or set(position) != {
                            "expected_output", "input_text", "scalar_offset"}
                        or not isinstance(position["input_text"], str)
                        or len(position["input_text"]) != 1
                        or not isinstance(position["expected_output"], str)
                        or len(position["expected_output"]) != 1
                        or type(position["scalar_offset"]) is not int
                        or not 0 <= position["scalar_offset"]
                        < len(record["input_text"])):
                    raise BroadQaExternalDataError(
                        "successor evaluator phrase position 漂移")
        identities.append(_sha_value(
            record["evaluation_id"], label="successor evaluation id"))
    if identities != sorted(set(identities)):
        raise BroadQaExternalDataError(
            "successor evaluator evaluation identity/次序漂移")


def _local_mapping_result(
        program: NormalizationSuccessorCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> NormalizationSuccessorDimensionResult:
    """按单一 Unihan edge 判定显式 target-policy 局部 transfer。"""
    values = {
        "applicable_mapping_count": 0,
        "conflicting_mapping_count": 0,
        "indexed_reference_mismatch_count": 0,
        "local_mapping_inventory_count": 0,
        "scope_mismatch_execution_count": 0,
        "supported_mapping_count": 0,
        "unscoped_rule_count": 0,
    }
    local = tuple(
        item for item in records
        if "LOCAL_MAPPING_TRANSFER" in item["family_keys"])
    values["local_mapping_inventory_count"] = len(local)
    if program is None:
        return _dimension("LOCAL_MAPPING_TRANSFER", "NE", values)
    values["unscoped_rule_count"] = sum(
        item.target_policy_scope != NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE
        for item in program.target_rules)
    for record in local:
        indexed = execute_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        reference = reference_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if indexed.target_rule_ids:
            values["applicable_mapping_count"] += 1
            if indexed.output_text == record["expected_output"]:
                values["supported_mapping_count"] += 1
            else:
                values["conflicting_mapping_count"] += 1
        for policy in SOURCE_POLICY_SCOPES:
            scoped = execute_normalization_successor_candidate(
                program, record["input_text"], policy_scope=policy)
            if scoped.target_rule_ids:
                values["scope_mismatch_execution_count"] += 1
    gate = NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
        "LOCAL_MAPPING_TRANSFER"]
    if (values["conflicting_mapping_count"] > gate[
            "conflicting_mapping_count_max"]
            or values["scope_mismatch_execution_count"] > gate[
                "scope_mismatch_execution_count_max"]
            or values["unscoped_rule_count"] > gate["unscoped_rule_count_max"]
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif values["applicable_mapping_count"] < gate[
            "applicable_mapping_count_min"]:
        outcome = "NE"
    elif values["supported_mapping_count"] != values[
            "applicable_mapping_count"]:
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("LOCAL_MAPPING_TRANSFER", outcome, values)


def _coverage_result(
        program: NormalizationSuccessorCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> NormalizationSuccessorDimensionResult:
    """逐 scalar 区分已覆盖、未覆盖与错误改写，再单独统计完整输出。"""
    values = {
        "covered_changed_position_count": 0,
        "evaluation_phrase_count": 0,
        "expected_changed_position_count": 0,
        "full_output_match_count": 0,
        "indexed_reference_mismatch_count": 0,
        "output_length_mismatch_count": 0,
        "partial_coverage_count": 0,
        "uncovered_position_count": 0,
        "wrong_changed_position_count": 0,
    }
    phrases = tuple(
        item for item in records
        if "END_TO_END_COVERAGE" in item["family_keys"])
    values["evaluation_phrase_count"] = len(phrases)
    if program is None:
        return _dimension("END_TO_END_COVERAGE", "NE", values)
    for record in phrases:
        indexed = execute_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        reference = reference_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if len(indexed.output_text) != len(record["expected_output"]):
            values["output_length_mismatch_count"] += 1
            continue
        if indexed.output_text == record["expected_output"]:
            values["full_output_match_count"] += 1
        elif indexed.output_text != record["input_text"]:
            values["partial_coverage_count"] += 1
        for input_item, expected_item, actual_item in zip(
                record["input_text"], record["expected_output"],
                indexed.output_text):
            if input_item == expected_item:
                if actual_item != input_item:
                    values["wrong_changed_position_count"] += 1
                continue
            values["expected_changed_position_count"] += 1
            if actual_item == expected_item:
                values["covered_changed_position_count"] += 1
            elif actual_item == input_item:
                values["uncovered_position_count"] += 1
            else:
                values["wrong_changed_position_count"] += 1
    gate = NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
        "END_TO_END_COVERAGE"]
    if (values["output_length_mismatch_count"] > gate[
            "output_length_mismatch_count_max"]
            or values["wrong_changed_position_count"] > gate[
                "wrong_changed_position_count_max"]
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif (values["evaluation_phrase_count"] < gate[
            "evaluation_phrase_count_min"]
          or values["full_output_match_count"] < gate[
              "full_output_match_count_min"]):
        outcome = "NE"
    else:
        outcome = "PASS"
    return _dimension("END_TO_END_COVERAGE", outcome, values)


def _source_conflict_result(
        program: NormalizationSuccessorCandidateProgram | None,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationSuccessorDimensionResult:
    """从 TRAIN pack 重放每个 policy 输出并阻止无 scope 冲突执行。"""
    values = {
        "declared_conflict_count": 0,
        "declared_observed_conflict_mismatch_count": 0,
        "indexed_reference_mismatch_count": 0,
        "observed_training_source_conflict_count": 0,
        "policy_specific_replay_mismatch_count": 0,
        "training_source_policy_count": 0,
        "unscoped_conflict_execution_count": 0,
    }
    conflicts = outputs.get("conflict-ledger.jsonl", ())
    values["observed_training_source_conflict_count"] = len(conflicts)
    policies = {
        item["source_policy_scope"]
        for record in conflicts for item in record["source_policy_outputs"]}
    values["training_source_policy_count"] = len(policies)
    if program is None:
        return _dimension("SOURCE_POLICY_CONFLICT", "NE", values)
    values["declared_conflict_count"] = len(program.declared_conflict_ids)
    observed_ids = {
        str(item["conflict_id"]) for item in conflicts
        if item.get("record_kind") == NORMALIZATION_SUCCESSOR_CONFLICT_KIND}
    values["declared_observed_conflict_mismatch_count"] = len(
        observed_ids.symmetric_difference(program.declared_conflict_ids))
    for conflict in conflicts:
        for policy_output in conflict["source_policy_outputs"]:
            policy = str(policy_output["source_policy_scope"])
            indexed = execute_normalization_successor_candidate(
                program, str(conflict["input_text"]), policy_scope=policy)
            reference = reference_normalization_successor_candidate(
                program, str(conflict["input_text"]), policy_scope=policy)
            if indexed != reference:
                values["indexed_reference_mismatch_count"] += 1
            if (indexed.output_text != policy_output["expected_output"]
                    or str(conflict["conflict_id"])
                    not in indexed.conflict_ids):
                values["policy_specific_replay_mismatch_count"] += 1
        unscoped = execute_normalization_successor_candidate(
            program, str(conflict["input_text"]), policy_scope="")
        if (unscoped.output_text != conflict["input_text"]
                or unscoped.unscoped_conflict_blocked != 1):
            values["unscoped_conflict_execution_count"] += 1
    gate = NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
        "SOURCE_POLICY_CONFLICT"]
    if (values["training_source_policy_count"] < gate[
            "training_source_policy_count_min"]
            or values["observed_training_source_conflict_count"] < gate[
                "observed_training_source_conflict_count_min"]):
        outcome = "NE"
    elif (values["declared_observed_conflict_mismatch_count"] > 0
          or values["policy_specific_replay_mismatch_count"] > 0
          or values["unscoped_conflict_execution_count"] > 0
          or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("SOURCE_POLICY_CONFLICT", outcome, values)


def _defeater_result(
        program: NormalizationSuccessorCandidateProgram | None,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationSuccessorDimensionResult:
    """核对 declared context override 是否全部形成非 identity executable。"""
    values = {
        "declared_defeater_count": 0,
        "executable_defeater_count": 0,
        "identity_only_defeater_count": 0,
        "malformed_defeater_count": 0,
    }
    declared = {
        str(item["context_rule_id"])
        for item in outputs.get("context-rules.jsonl", ())
        if item.get("record_kind") == NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND}
    values["declared_defeater_count"] = len(declared)
    if program is None:
        return _dimension(
            "DEFEATER_REPRESENTATION_EXECUTABILITY", "NE", values)
    executable = {item.context_rule_id for item in program.context_rules}
    values["executable_defeater_count"] = len(executable)
    values["malformed_defeater_count"] = len(
        declared.symmetric_difference(executable))
    values["identity_only_defeater_count"] = sum(
        item.base_output == item.observed_output for item in program.context_rules)
    gate = NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
        "DEFEATER_REPRESENTATION_EXECUTABILITY"]
    if not declared:
        outcome = "NE"
    elif (values["identity_only_defeater_count"] > gate[
            "identity_only_defeater_count_max"]
          or values["malformed_defeater_count"] > gate[
              "malformed_defeater_count_max"]
          or declared != executable):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension(
        "DEFEATER_REPRESENTATION_EXECUTABILITY", outcome, values)


def _context_transfer_result(
        program: NormalizationSuccessorCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> NormalizationSuccessorDimensionResult:
    """在独立 context-sensitive phrase 上核对目标 policy exact 输出。"""
    values = {
        "applicable_context_count": 0,
        "context_exact_support_count": 0,
        "context_record_inventory_count": 0,
        "context_wrong_output_count": 0,
        "indexed_reference_mismatch_count": 0,
    }
    contexts = tuple(
        item for item in records
        if "INDEPENDENT_CONTEXT_TRANSFER" in item["family_keys"])
    values["context_record_inventory_count"] = len(contexts)
    if program is None:
        return _dimension("INDEPENDENT_CONTEXT_TRANSFER", "NE", values)
    for record in contexts:
        indexed = execute_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        reference = reference_normalization_successor_candidate(
            program, record["input_text"],
            policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if indexed.target_rule_ids:
            values["applicable_context_count"] += 1
            if indexed.output_text == record["expected_output"]:
                values["context_exact_support_count"] += 1
            else:
                values["context_wrong_output_count"] += 1
    gate = NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
        "INDEPENDENT_CONTEXT_TRANSFER"]
    if (values["context_wrong_output_count"] > gate[
            "context_wrong_output_count_max"]
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif (values["context_record_inventory_count"] < gate[
            "context_record_inventory_min"]
          or values["applicable_context_count"] < gate[
              "applicable_context_count_min"]
          or values["context_exact_support_count"] < gate[
              "context_exact_support_count_min"]):
        outcome = "NE"
    else:
        outcome = "PASS"
    return _dimension("INDEPENDENT_CONTEXT_TRANSFER", outcome, values)


def _runtime_result(
        program: NormalizationSuccessorCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> tuple[NormalizationSuccessorDimensionResult, str]:
    """对全部 evaluation input 双遍执行并绑定规范 replay digest。"""
    values = {
        "canonical_replay_mismatch_count": 0,
        "indexed_reference_mismatch_count": 0,
        "production_enabled": 0,
        "runtime_exception_count": 0,
        "runtime_executed_input_count": 0,
        "target_policy_scope_mismatch_count": 0,
    }
    digest = hashlib.sha256()
    if program is None:
        return (
            _dimension("RUNTIME_PRODUCTION_BEHAVIOR", "NE", values),
            digest.hexdigest(),
        )
    values["production_enabled"] = program.production_enabled
    for record in records:
        try:
            first = execute_normalization_successor_candidate(
                program, record["input_text"],
                policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
            second = execute_normalization_successor_candidate(
                program, record["input_text"],
                policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
            reference = reference_normalization_successor_candidate(
                program, record["input_text"],
                policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
        except (BroadQaExternalDataError, TypeError, ValueError):
            values["runtime_exception_count"] += 1
            continue
        values["runtime_executed_input_count"] += 1
        digest.update(bytes.fromhex(record["evaluation_id"]))
        digest.update(bytes.fromhex(first.sha256()))
        if first != second:
            values["canonical_replay_mismatch_count"] += 1
        if first != reference:
            values["indexed_reference_mismatch_count"] += 1
        if first.requested_policy_scope != NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE:
            values["target_policy_scope_mismatch_count"] += 1
    if (values["production_enabled"] != 0
            or values["runtime_exception_count"] > 0
            or values["canonical_replay_mismatch_count"] > 0
            or values["indexed_reference_mismatch_count"] > 0
            or values["target_policy_scope_mismatch_count"] > 0
            or values["runtime_executed_input_count"] != len(records)):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return (
        _dimension("RUNTIME_PRODUCTION_BEHAVIOR", outcome, values),
        digest.hexdigest(),
    )


def evaluate_normalization_successor_candidate(
        *,
        protocol_manifest: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        rule_pack_manifest: dict[str, object],
        pack_outputs: dict[str, tuple[dict[str, object], ...]],
        program: NormalizationSuccessorCandidateProgram | None,
        ) -> NormalizationSuccessorEvaluationReport:
    """按冻结门生成六维报告，不读取 reserve 或改变 pack 状态。"""
    protocol_sha, source_sha, inventory_sha, expected_count = _validate_protocol(
        protocol_manifest)
    _validate_inventory(
        evaluation_records,
        source_pack_manifest_sha256=source_sha,
        expected_count=expected_count,
    )
    pack_sha = _sha_value(
        rule_pack_manifest.get("manifest_sha256"),
        label="successor evaluator rule pack")
    if (rule_pack_manifest.get("production_enabled") != 0
            or rule_pack_manifest.get("mastery_claimed") != 0
            or rule_pack_manifest.get("runtime_state")
            != "LEARNED_PACK_DISABLED"):
        raise BroadQaExternalDataError(
            "successor evaluator pack 已启用或漂移")
    if program is not None and program.rule_pack_manifest_sha256 != pack_sha:
        raise BroadQaExternalDataError(
            "successor evaluator candidate/pack identity 漂移")
    runtime, runtime_sha = _runtime_result(program, evaluation_records)
    dimensions = (
        _local_mapping_result(program, evaluation_records),
        _coverage_result(program, evaluation_records),
        _source_conflict_result(program, pack_outputs),
        _defeater_result(program, pack_outputs),
        _context_transfer_result(program, evaluation_records),
        runtime,
    )
    outcomes = tuple(item.outcome for item in dimensions)
    overall = (
        "FAIL" if "FAIL" in outcomes else
        "NE" if "NE" in outcomes else "PASS")
    clone_sha = "0" * 64 if program is None else program.sha256()
    return NormalizationSuccessorEvaluationReport(
        protocol_sha,
        source_sha,
        pack_sha,
        clone_sha,
        inventory_sha,
        len(evaluation_records),
        dimensions,
        overall,
        runtime_sha,
        1,
        1,
        0,
        0,
        0,
    )


__all__ = [
    "NORMALIZATION_SUCCESSOR_DIMENSION_ORDER",
    "NORMALIZATION_SUCCESSOR_EVALUATION_OUTCOMES",
    "NORMALIZATION_SUCCESSOR_EVALUATION_REPORT_KIND",
    "NormalizationSuccessorDimensionResult",
    "NormalizationSuccessorEvaluationReport",
    "evaluate_normalization_successor_candidate",
]
