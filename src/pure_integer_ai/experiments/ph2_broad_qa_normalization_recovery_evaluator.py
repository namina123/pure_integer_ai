"""按冻结 recovery 六维协议纯评估 transfer candidate。

本模块不读路径、不写 publication、不读取 reserve。evaluation 输入先统一执行
indexed/replay/reference，六个 bearing dimension 再复用同一批结果，避免重复线性扫描。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_clone import (
    NormalizationRecoveryCandidateProgram,
    execute_normalization_recovery_candidate,
    reference_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_records import (
    NormalizationRecoveryExecutionResult,
    RECOVERY_TRANSFER_REGION_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS,
    NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    NORMALIZATION_RECOVERY_CONFLICT_KIND,
    NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
    NORMALIZATION_RECOVERY_PHRASE_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FIREFOX_L10N_COMMIT,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    SOURCE_POLICY_SCOPES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_EVALUATION_REPORT_V2")
NORMALIZATION_RECOVERY_EVALUATION_OUTCOMES = ("FAIL", "NE", "PASS")
NORMALIZATION_RECOVERY_DIMENSION_ORDER = (
    "LOCAL_MAPPING_TRANSFER",
    "END_TO_END_COVERAGE",
    "SOURCE_POLICY_CONFLICT",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "INDEPENDENT_CONTEXT_TRANSFER",
    "RUNTIME_PRODUCTION_BEHAVIOR",
)


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
    if isinstance(expected, (list, tuple)):
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
        raise BroadQaExternalDataError("recovery evaluation metrics 非法")
    return tuple(sorted(values.items()))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryDimensionResult:
    """一个 recovery bearing dimension 的整数计数和结论。"""

    dimension_key: str
    outcome: str
    metrics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """要求维度、结论与 metrics 完整规范。"""
        if (self.dimension_key not in NORMALIZATION_RECOVERY_DIMENSION_ORDER
                or self.outcome not in NORMALIZATION_RECOVERY_EVALUATION_OUTCOMES
                or self.metrics != tuple(sorted(set(self.metrics)))
                or any(not isinstance(key, str) or not key
                       or type(value) is not int or value < 0
                       for key, value in self.metrics)):
            raise BroadQaExternalDataError(
                "recovery evaluation dimension result 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范维度结果。"""
        return {
            "dimension_key": self.dimension_key,
            "metrics": {key: value for key, value in self.metrics},
            "outcome": self.outcome,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationRecoveryEvaluationReport:
    """绑定 protocol、TRAIN pack、transfer profile 与六维结果的报告。"""

    protocol_manifest_sha256: str
    evaluation_source_pack_manifest_sha256: str
    training_protocol_manifest_sha256: str
    rule_pack_manifest_sha256: str
    candidate_program_sha256: str
    transfer_profile_sha256: str
    evaluation_inventory_sha256: str
    evaluation_record_count: int
    dimensions: tuple[NormalizationRecoveryDimensionResult, ...]
    overall_outcome: str
    runtime_result_sha256: str
    learned_pack_read_count: int
    evaluation_run_count: int
    production_enabled: int
    mastery_claimed: int
    reserve_payload_read_count: int
    teacher_api_llm_call_count: int

    def __post_init__(self) -> None:
        """核验身份、六维顺序、总结果与正式零边界。"""
        for label, value in (
                ("protocol", self.protocol_manifest_sha256),
                ("evaluation source", self.evaluation_source_pack_manifest_sha256),
                ("training protocol", self.training_protocol_manifest_sha256),
                ("rule pack", self.rule_pack_manifest_sha256),
                ("candidate program", self.candidate_program_sha256),
                ("transfer profile", self.transfer_profile_sha256),
                ("evaluation inventory", self.evaluation_inventory_sha256),
                ("runtime result", self.runtime_result_sha256)):
            _sha_value(value, label=f"recovery evaluation {label}")
        outcomes = tuple(item.outcome for item in self.dimensions)
        expected_overall = (
            "FAIL" if "FAIL" in outcomes else
            "NE" if "NE" in outcomes else "PASS")
        if (type(self.evaluation_record_count) is not int
                or self.evaluation_record_count <= 0
                or tuple(item.dimension_key for item in self.dimensions)
                != NORMALIZATION_RECOVERY_DIMENSION_ORDER
                or self.overall_outcome != expected_overall
                or type(self.learned_pack_read_count) is not int
                or self.learned_pack_read_count != 1
                or type(self.evaluation_run_count) is not int
                or self.evaluation_run_count != 1
                or any(type(value) is not int or value != 0 for value in (
                    self.production_enabled, self.mastery_claimed,
                    self.reserve_payload_read_count,
                    self.teacher_api_llm_call_count))):
            raise BroadQaExternalDataError(
                "recovery evaluation report 边界漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范正式或 synthetic 六维报告。"""
        return {
            "artifact_kind": NORMALIZATION_RECOVERY_EVALUATION_REPORT_KIND,
            "candidate_program_sha256": self.candidate_program_sha256,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "evaluation_inventory_sha256": self.evaluation_inventory_sha256,
            "evaluation_record_count": self.evaluation_record_count,
            "evaluation_run_count": self.evaluation_run_count,
            "evaluation_source_pack_manifest_sha256": (
                self.evaluation_source_pack_manifest_sha256),
            "format_version": 2,
            "learned_pack_read_count": self.learned_pack_read_count,
            "mastery_claimed": self.mastery_claimed,
            "overall_outcome": self.overall_outcome,
            "production_enabled": self.production_enabled,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "reserve_payload_read_count": self.reserve_payload_read_count,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "runtime_result_sha256": self.runtime_result_sha256,
            "teacher_api_llm_call_count": self.teacher_api_llm_call_count,
            "training_protocol_manifest_sha256": (
                self.training_protocol_manifest_sha256),
            "transfer_profile_sha256": self.transfer_profile_sha256,
        }

    def sha256(self) -> str:
        """返回规范报告摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _dimension(
        key: str,
        outcome: str,
        values: dict[str, int],
        ) -> NormalizationRecoveryDimensionResult:
    """构造一个规范 bearing dimension。"""
    return NormalizationRecoveryDimensionResult(key, outcome, _metrics(values))


def _validate_protocol(manifest: dict[str, object]) -> tuple[str, str, str, int]:
    """核验学习前冻结且仍未消费的 recovery evaluation 协议。"""
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND
            or manifest.get("status") != NORMALIZATION_RECOVERY_EVALUATION_STATUS
            or manifest.get("format_version") != 2
            or manifest.get("target_policy_scope")
            != NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE
            or not _strict_equal(
                manifest.get("dimensions"),
                NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS)
            or not _strict_equal(
                manifest.get("metric_contract"),
                NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT)
            or any(manifest.get(name) != 0 for name in (
                "candidate_pack_read_count", "evaluation_run_count",
                "learned_pack_read_count", "mastery_claimed",
                "prior_formal_item_read_count", "production_enabled",
                "recovery_training_source_read_count",
                "reserve_payload_read_count", "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError(
            "recovery evaluator protocol 已消费或冻结字段漂移")
    inventory = manifest.get("evaluation_inventory")
    summary = manifest.get("inventory_summary")
    if (not isinstance(inventory, dict) or not isinstance(summary, dict)
            or type(summary.get("evaluation_count")) is not int
            or summary["evaluation_count"] <= 0):
        raise BroadQaExternalDataError(
            "recovery evaluator protocol inventory 漂移")
    return (
        _sha_value(manifest.get("manifest_sha256"), label="recovery protocol"),
        _sha_value(
            manifest.get("source_pack_manifest_sha256"),
            label="recovery evaluation source"),
        _sha_value(inventory.get("sha256"), label="recovery inventory"),
        summary["evaluation_count"],
    )


def _validate_inventory(
        records: tuple[dict[str, object], ...],
        *,
        source_pack_manifest_sha256: str,
        expected_count: int,
        ) -> None:
    """拒绝 reserve、未知字段、错来源、错 scope 或非规范次序。"""
    if (not isinstance(records, tuple) or len(records) != expected_count
            or not records):
        raise BroadQaExternalDataError("recovery evaluator inventory 数量漂移")
    common = {
        "content_cluster_id", "context_sensitive", "evaluation_id",
        "expected_output", "family_keys", "format_version",
        "identity_preservation", "input_scalar_count", "input_text",
        "output_scalar_count", "record_kind", "source_commit",
        "source_occurrence_count", "source_pack_manifest_sha256",
        "source_pair_id", "source_policy_scope", "split",
        "split_group_sha256", "target_policy_scope",
    }
    mapping = common | {
        "mapping_expected_character", "mapping_input_character",
        "mapping_offset",
    }
    identities = []
    for record in records:
        if (not isinstance(record, dict)
                or frozenset(record) not in {
                    frozenset(common), frozenset(mapping)}
                or record["split"] != "EVALUATION"
                or record["format_version"] != 2
                or record["record_kind"]
                != NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND
                or record["source_pack_manifest_sha256"]
                != source_pack_manifest_sha256
                or record["target_policy_scope"]
                != NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE
                or record["source_policy_scope"]
                != "MOZILLA_FIREFOX_L10N_ZH_TW_TO_ZH_CN"
                or record["source_commit"] != FIREFOX_L10N_COMMIT
                or not isinstance(record["input_text"], str)
                or not record["input_text"]
                or not isinstance(record["expected_output"], str)
                or not record["expected_output"]
                or record["input_scalar_count"] != len(record["input_text"])
                or record["output_scalar_count"]
                != len(record["expected_output"])
                or type(record["source_occurrence_count"]) is not int
                or record["source_occurrence_count"] <= 0
                or type(record["context_sensitive"]) is not int
                or record["context_sensitive"] not in {0, 1}
                or type(record["identity_preservation"]) is not int
                or record["identity_preservation"] not in {0, 1}
                or not isinstance(record["family_keys"], list)
                or not record["family_keys"]
                or record["family_keys"] != [
                    key for key in NORMALIZATION_RECOVERY_DIMENSION_ORDER
                    if key in record["family_keys"]]
                or any(key not in {
                    "LOCAL_MAPPING_TRANSFER", "END_TO_END_COVERAGE",
                    "INDEPENDENT_CONTEXT_TRANSFER"}
                    for key in record["family_keys"])):
            raise BroadQaExternalDataError(
                "recovery evaluator inventory 含 reserve/未知字段或来源漂移")
        _sha_value(record["content_cluster_id"], label="recovery content cluster")
        _sha_value(record["evaluation_id"], label="recovery evaluation id")
        _sha_value(record["source_pair_id"], label="recovery source pair")
        _sha_value(record["split_group_sha256"], label="recovery split group")
        has_mapping = frozenset(record) == frozenset(mapping)
        mapped_family = (
            "LOCAL_MAPPING_TRANSFER" in record["family_keys"]
            or "INDEPENDENT_CONTEXT_TRANSFER" in record["family_keys"])
        if has_mapping != mapped_family:
            raise BroadQaExternalDataError(
                "recovery evaluator mapping 字段/family 漂移")
        if has_mapping:
            offset = record["mapping_offset"]
            if (type(offset) is not int or offset < 0
                    or offset >= len(record["input_text"])
                    or offset >= len(record["expected_output"])
                    or record["mapping_input_character"]
                    != record["input_text"][offset]
                    or record["mapping_expected_character"]
                    != record["expected_output"][offset]
                    or len(record["mapping_input_character"]) != 1
                    or len(record["mapping_expected_character"]) != 1):
                raise BroadQaExternalDataError(
                    "recovery evaluator mapping offset 漂移")
        if ((record["context_sensitive"] == 1)
                != ("INDEPENDENT_CONTEXT_TRANSFER" in record["family_keys"])
                or (record["identity_preservation"] == 1)
                != (record["input_text"] == record["expected_output"])):
            raise BroadQaExternalDataError(
                "recovery evaluator context/identity 标记漂移")
        identities.append(record["evaluation_id"])
    if identities != sorted(set(identities)):
        raise BroadQaExternalDataError(
            "recovery evaluator evaluation identity/次序漂移")


def _prepare_evaluation_executions(
        program: NormalizationRecoveryCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, tuple[
                NormalizationRecoveryExecutionResult,
                NormalizationRecoveryExecutionResult,
                NormalizationRecoveryExecutionResult]],
            int,
        ]:
    """对每个 evaluation 输入一次性形成 replay/reference 三元组。"""
    if program is None:
        return {}, 0
    values = {}
    exception_count = 0
    for record in records:
        try:
            first = execute_normalization_recovery_candidate(
                program, str(record["input_text"]),
                policy_scope=NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
                regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
            )
            second = execute_normalization_recovery_candidate(
                program, str(record["input_text"]),
                policy_scope=NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
                regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
            )
            reference = reference_normalization_recovery_candidate(
                program, str(record["input_text"]),
                policy_scope=NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
                regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
            )
        except (BroadQaExternalDataError, TypeError, ValueError):
            exception_count += 1
            continue
        values[str(record["evaluation_id"])] = (first, second, reference)
    return values, exception_count


def _applicable(result: NormalizationRecoveryExecutionResult) -> bool:
    """要求 target projection、scope 与 transfer identity 全部成立。"""
    return (result.projection_used == 1
            and result.scope_mismatch == 0
            and result.requested_policy_scope
            == NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE
            and result.regional_scope == RECOVERY_TRANSFER_REGION_SCOPE
            and bool(result.transfer_profile_id))


def _local_mapping_result(
        program: NormalizationRecoveryCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        executions: dict[str, tuple[
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult]],
        ) -> NormalizationRecoveryDimensionResult:
    """核对稳定单字符映射的完整 applicability 与精确输出。"""
    values = {
        "applicable_mapping_count": 0,
        "indexed_reference_mismatch_count": 0,
        "local_mapping_inventory_count": 0,
        "mapping_false_accept_count": 0,
        "mapping_false_reject_count": 0,
        "supported_mapping_count": 0,
        "unscoped_rule_count": 0,
    }
    local = tuple(item for item in records
                  if "LOCAL_MAPPING_TRANSFER" in item["family_keys"])
    values["local_mapping_inventory_count"] = len(local)
    if program is None:
        return _dimension("LOCAL_MAPPING_TRANSFER", "NE", values)
    for record in local:
        triple = executions.get(str(record["evaluation_id"]))
        if triple is None:
            continue
        indexed, _second, reference = triple
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if _applicable(indexed):
            values["applicable_mapping_count"] += 1
        if indexed.output_text == record["expected_output"]:
            values["supported_mapping_count"] += 1
        elif indexed.output_text == record["input_text"]:
            values["mapping_false_reject_count"] += 1
        else:
            values["mapping_false_accept_count"] += 1
        unscoped = execute_normalization_recovery_candidate(
            program, str(record["input_text"]), policy_scope="")
        if unscoped.output_text != record["input_text"] or unscoped.target_rule_ids:
            values["unscoped_rule_count"] += 1
    gate = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS[
        "LOCAL_MAPPING_TRANSFER"]
    if (values["mapping_false_accept_count"] > gate[
            "mapping_false_accept_count_max"]
            or values["mapping_false_reject_count"] > gate[
                "mapping_false_reject_count_max"]
            or values["unscoped_rule_count"] > gate["unscoped_rule_count_max"]
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif (values["local_mapping_inventory_count"] < gate[
            "local_mapping_inventory_min"]
          or values["applicable_mapping_count"]
          != values["local_mapping_inventory_count"]):
        outcome = "NE"
    elif values["supported_mapping_count"] != values[
            "applicable_mapping_count"]:
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("LOCAL_MAPPING_TRANSFER", outcome, values)


def _coverage_result(
        program: NormalizationRecoveryCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        executions: dict[str, tuple[
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult]],
        ) -> NormalizationRecoveryDimensionResult:
    """逐项核对非恒等 phrase、identity preservation 与错误改写。"""
    values = {
        "applicable_phrase_count": 0,
        "coverage_inventory_count": 0,
        "false_accept_count": 0,
        "false_reject_count": 0,
        "full_output_match_count": 0,
        "identity_false_accept_count": 0,
        "identity_inventory_count": 0,
        "indexed_reference_mismatch_count": 0,
        "output_length_mismatch_count": 0,
        "phrase_inventory_count": 0,
        "wrong_changed_position_count": 0,
    }
    coverage = tuple(item for item in records
                     if "END_TO_END_COVERAGE" in item["family_keys"])
    values["coverage_inventory_count"] = len(coverage)
    values["phrase_inventory_count"] = sum(
        item["input_text"] != item["expected_output"] for item in coverage)
    values["identity_inventory_count"] = sum(
        item["identity_preservation"] for item in coverage)
    if program is None:
        return _dimension("END_TO_END_COVERAGE", "NE", values)
    for record in coverage:
        triple = executions.get(str(record["evaluation_id"]))
        if triple is None:
            continue
        indexed, _second, reference = triple
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if _applicable(indexed):
            values["applicable_phrase_count"] += 1
        input_text = str(record["input_text"])
        expected = str(record["expected_output"])
        actual = indexed.output_text
        if actual == expected:
            values["full_output_match_count"] += 1
        elif record["identity_preservation"] == 1:
            values["identity_false_accept_count"] += 1
        elif actual == input_text:
            values["false_reject_count"] += 1
        else:
            values["false_accept_count"] += 1
        if len(actual) != len(expected):
            values["output_length_mismatch_count"] += 1
            values["wrong_changed_position_count"] += 1
            continue
        for input_item, expected_item, actual_item in zip(
                input_text, expected, actual):
            if input_item == expected_item:
                if actual_item != input_item:
                    values["wrong_changed_position_count"] += 1
            elif actual_item not in {input_item, expected_item}:
                values["wrong_changed_position_count"] += 1
    gate = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS[
        "END_TO_END_COVERAGE"]
    if (values["false_accept_count"] > gate["false_accept_count_max"]
            or values["false_reject_count"] > gate["false_reject_count_max"]
            or values["identity_false_accept_count"] > gate[
                "identity_false_accept_count_max"]
            or values["wrong_changed_position_count"] > gate[
                "wrong_changed_position_count_max"]
            or values["output_length_mismatch_count"] > 0
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif (values["phrase_inventory_count"] < gate["phrase_inventory_min"]
          or values["identity_inventory_count"] < gate[
              "identity_inventory_min"]
          or values["applicable_phrase_count"]
          != values["coverage_inventory_count"]):
        outcome = "NE"
    elif values["full_output_match_count"] != values[
            "applicable_phrase_count"]:
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("END_TO_END_COVERAGE", outcome, values)


def _source_conflict_result(
        program: NormalizationRecoveryCandidateProgram | None,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationRecoveryDimensionResult:
    """从 SUPPORT Evidence 重放每个冲突的真实 source-policy 输出。"""
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
    supports = {
        str(item["observation_id"]): item
        for item in outputs.get("evidence.jsonl", ())
        if item.get("stance") == "SUPPORT"
    }
    observed = {
        str(item["conflict_id"]): item for item in conflicts
        if item.get("record_kind") == NORMALIZATION_RECOVERY_CONFLICT_KIND}
    values["observed_training_source_conflict_count"] = len(observed)
    if program is None:
        return _dimension("SOURCE_POLICY_CONFLICT", "NE", values)
    declared = {item.conflict_id: item for item in program.conflicts}
    values["declared_conflict_count"] = len(declared)
    values["declared_observed_conflict_mismatch_count"] = len(
        set(declared).symmetric_difference(observed))
    policies = set()
    for conflict_id, record in observed.items():
        observation_ids = sorted({
            str(value)
            for family in record.get("family_outputs", ())
            if isinstance(family, dict)
            for value in family.get("observation_ids", ())
        })
        if not observation_ids:
            values["policy_specific_replay_mismatch_count"] += 1
            continue
        for observation_id in observation_ids:
            evidence = supports.get(observation_id)
            if evidence is None:
                values["policy_specific_replay_mismatch_count"] += 1
                continue
            policy = str(evidence["source_policy_scope"])
            policies.add(policy)
            indexed = execute_normalization_recovery_candidate(
                program, str(record["input_text"]), policy_scope=policy)
            reference = reference_normalization_recovery_candidate(
                program, str(record["input_text"]), policy_scope=policy)
            if indexed != reference:
                values["indexed_reference_mismatch_count"] += 1
            if (indexed.output_text != evidence["hypothesis_output"]
                    or conflict_id not in indexed.conflict_ids):
                values["policy_specific_replay_mismatch_count"] += 1
        unscoped = execute_normalization_recovery_candidate(
            program, str(record["input_text"]), policy_scope="")
        if (unscoped.output_text != record["input_text"]
                or unscoped.unscoped_conflict_blocked != 1
                or conflict_id not in unscoped.conflict_ids):
            values["unscoped_conflict_execution_count"] += 1
    values["training_source_policy_count"] = len(policies)
    gate = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS[
        "SOURCE_POLICY_CONFLICT"]
    if (values["declared_conflict_count"] < gate[
            "declared_conflict_count_min"]
            or values["observed_training_source_conflict_count"] < gate[
                "declared_conflict_count_min"]
            or values["training_source_policy_count"] < gate[
                "training_source_policy_count_min"]):
        outcome = "NE"
    elif (values["declared_observed_conflict_mismatch_count"] > 0
          or values["policy_specific_replay_mismatch_count"] > 0
          or values["unscoped_conflict_execution_count"] > gate[
              "unscoped_conflict_execution_count_max"]
          or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("SOURCE_POLICY_CONFLICT", outcome, values)


def _defeater_result(
        program: NormalizationRecoveryCandidateProgram | None,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationRecoveryDimensionResult:
    """核对 phrase override 是否全部形成非空、非 base identity 的执行结构。"""
    values = {
        "declared_defeater_count": 0,
        "executable_defeater_count": 0,
        "identity_only_defeater_count": 0,
        "malformed_defeater_count": 0,
    }
    declared = {
        str(item["rule_id"])
        for item in outputs.get("source-phrase-rules.jsonl", ())
        if item.get("record_kind") == NORMALIZATION_RECOVERY_PHRASE_RULE_KIND}
    values["declared_defeater_count"] = len(declared)
    if program is None:
        return _dimension(
            "DEFEATER_REPRESENTATION_EXECUTABILITY", "NE", values)
    executable = {item.rule_id for item in program.phrase_overrides}
    values["executable_defeater_count"] = len(executable)
    values["malformed_defeater_count"] = len(
        declared.symmetric_difference(executable))
    values["identity_only_defeater_count"] = sum(
        item.base_output == item.output_text for item in program.phrase_overrides)
    gate = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS[
        "DEFEATER_REPRESENTATION_EXECUTABILITY"]
    if values["declared_defeater_count"] < gate[
            "declared_defeater_count_min"]:
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
        program: NormalizationRecoveryCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        executions: dict[str, tuple[
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult]],
        ) -> NormalizationRecoveryDimensionResult:
    """核对独立 context inventory 的完整 applicability 与精确输出。"""
    values = {
        "applicable_context_count": 0,
        "context_exact_support_count": 0,
        "context_record_inventory_count": 0,
        "context_wrong_output_count": 0,
        "indexed_reference_mismatch_count": 0,
    }
    contexts = tuple(item for item in records
                     if "INDEPENDENT_CONTEXT_TRANSFER" in item["family_keys"])
    values["context_record_inventory_count"] = len(contexts)
    if program is None:
        return _dimension("INDEPENDENT_CONTEXT_TRANSFER", "NE", values)
    for record in contexts:
        triple = executions.get(str(record["evaluation_id"]))
        if triple is None:
            continue
        indexed, _second, reference = triple
        if indexed != reference:
            values["indexed_reference_mismatch_count"] += 1
        if _applicable(indexed):
            values["applicable_context_count"] += 1
        if indexed.output_text == record["expected_output"]:
            values["context_exact_support_count"] += 1
        else:
            values["context_wrong_output_count"] += 1
    gate = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS[
        "INDEPENDENT_CONTEXT_TRANSFER"]
    if (values["context_wrong_output_count"] > gate[
            "context_wrong_output_count_max"]
            or values["indexed_reference_mismatch_count"] > 0):
        outcome = "FAIL"
    elif (values["context_record_inventory_count"] < gate[
            "context_inventory_min"]
          or values["applicable_context_count"]
          != values["context_record_inventory_count"]):
        outcome = "NE"
    elif values["context_exact_support_count"] != values[
            "applicable_context_count"]:
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("INDEPENDENT_CONTEXT_TRANSFER", outcome, values)


def _runtime_result(
        program: NormalizationRecoveryCandidateProgram | None,
        records: tuple[dict[str, object], ...],
        executions: dict[str, tuple[
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult,
            NormalizationRecoveryExecutionResult]],
        exception_count: int,
        ) -> tuple[NormalizationRecoveryDimensionResult, str]:
    """绑定全部 evaluation 输入的 replay/reference 与规范结果摘要。"""
    values = {
        "canonical_replay_mismatch_count": 0,
        "indexed_reference_mismatch_count": 0,
        "production_enabled": 0,
        "runtime_exception_count": exception_count,
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
    profile_sha = program.transfer_profile.sha256()
    for record in records:
        triple = executions.get(str(record["evaluation_id"]))
        if triple is None:
            continue
        first, second, reference = triple
        values["runtime_executed_input_count"] += 1
        digest.update(bytes.fromhex(str(record["evaluation_id"])))
        digest.update(bytes.fromhex(first.sha256()))
        if first != second:
            values["canonical_replay_mismatch_count"] += 1
        if first != reference:
            values["indexed_reference_mismatch_count"] += 1
        if (not _applicable(first)
                or first.transfer_profile_id != profile_sha):
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


def evaluate_normalization_recovery_candidate(
        *,
        protocol_manifest: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        rule_pack_manifest: dict[str, object],
        pack_outputs: dict[str, tuple[dict[str, object], ...]],
        program: NormalizationRecoveryCandidateProgram | None,
        ) -> NormalizationRecoveryEvaluationReport:
    """按冻结六维门生成报告，不读取 reserve 或改变 pack 状态。"""
    protocol_sha, source_sha, inventory_sha, expected_count = _validate_protocol(
        protocol_manifest)
    _validate_inventory(
        evaluation_records,
        source_pack_manifest_sha256=source_sha,
        expected_count=expected_count,
    )
    pack_sha = _sha_value(
        rule_pack_manifest.get("manifest_sha256"),
        label="recovery evaluator rule pack")
    training_sha = _sha_value(
        rule_pack_manifest.get("protocol_manifest_sha256"),
        label="recovery evaluator training protocol")
    expected_outputs = {
        name for name, _role, _identity
        in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES}
    if (rule_pack_manifest.get("production_enabled") != 0
            or rule_pack_manifest.get("mastery_claimed") != 0
            or rule_pack_manifest.get("runtime_state")
            != "LEARNED_PACK_DISABLED"
            or not isinstance(pack_outputs, dict)
            or set(pack_outputs) != expected_outputs):
        raise BroadQaExternalDataError(
            "recovery evaluator pack 已启用或 inventory 漂移")
    if program is not None and (
            program.transfer_profile.rule_pack_manifest_sha256 != pack_sha
            or program.transfer_profile.evaluation_protocol_manifest_sha256
            != protocol_sha):
        raise BroadQaExternalDataError(
            "recovery evaluator candidate/pack/protocol identity 漂移")
    executions, exception_count = _prepare_evaluation_executions(
        program, evaluation_records)
    runtime, runtime_sha = _runtime_result(
        program, evaluation_records, executions, exception_count)
    dimensions = (
        _local_mapping_result(program, evaluation_records, executions),
        _coverage_result(program, evaluation_records, executions),
        _source_conflict_result(program, pack_outputs),
        _defeater_result(program, pack_outputs),
        _context_transfer_result(program, evaluation_records, executions),
        runtime,
    )
    outcomes = tuple(item.outcome for item in dimensions)
    overall = (
        "FAIL" if "FAIL" in outcomes else
        "NE" if "NE" in outcomes else "PASS")
    candidate_sha = "0" * 64 if program is None else program.sha256()
    transfer_sha = (
        "0" * 64 if program is None else program.transfer_profile.sha256())
    return NormalizationRecoveryEvaluationReport(
        protocol_sha,
        source_sha,
        training_sha,
        pack_sha,
        candidate_sha,
        transfer_sha,
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
        0,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_DIMENSION_ORDER",
    "NORMALIZATION_RECOVERY_EVALUATION_OUTCOMES",
    "NORMALIZATION_RECOVERY_EVALUATION_REPORT_KIND",
    "NormalizationRecoveryDimensionResult",
    "NormalizationRecoveryEvaluationReport",
    "evaluate_normalization_recovery_candidate",
]
