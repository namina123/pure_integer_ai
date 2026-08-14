"""纯派生 recovery-v3 TRAIN-only runtime、defeater 与 LOSO 审计记录。

full-pack 审计核对长度桶、双解释器和每个 REFUTE occurrence 的可执行
defeater。LOSO 分别移除一个 source family 后重新归纳规则，再执行被留出来源；
不会把全量选择出的规则回放冒充独立泛化。
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_learning_records import (
    derive_normalization_recovery_v3_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_phrase_runtime import (
    compile_normalization_recovery_v3_phrase_program,
    execute_normalization_recovery_v3_phrase_batch,
    normalization_recovery_v3_defeater_matches,
    reference_normalization_recovery_v3_phrase_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    GODOT_SOURCE_FAMILY,
    RECOVERY_V3_TARGET_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    derive_normalization_recovery_v3_groups,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V3_RUNTIME_AUDIT_CASE_KIND = (
    "NORMALIZATION_RECOVERY_V3_RUNTIME_AUDIT_CASE_V1")
NORMALIZATION_RECOVERY_V3_LOSO_AUDIT_KIND = (
    "NORMALIZATION_RECOVERY_V3_LOSO_AUDIT_V1")

LENGTH_BUCKETS = ("L02", "L03_04", "L05_08", "L09_16", "L17_40")
RULE_BARE_OUTCOMES = ("DEFEATER_BACKOFF", "RULE_OUTPUT", "UNEXPECTED")
LOSO_OUTCOMES = ("EXACT", "UNKNOWN", "WRONG")
SOURCE_FAMILIES = (GODOT_SOURCE_FAMILY, THUNDERBIRD_SOURCE_FAMILY)


def _sha256(payload: bytes) -> str:
    """返回规范 audit identity。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _length_bucket(length: int) -> str:
    """按冻结 phrase scalar 长度返回互斥桶。"""
    if length == 2:
        return "L02"
    if 3 <= length <= 4:
        return "L03_04"
    if 5 <= length <= 8:
        return "L05_08"
    if 9 <= length <= 16:
        return "L09_16"
    if 17 <= length <= 40:
        return "L17_40"
    raise BroadQaExternalDataError("v3 audit phrase length 越界")


def _work_records(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """为 LOSO 子集重建与正式协议同构的确定 ordered work。"""
    sources = (
        ("PAIR_OBSERVATION_INGEST", "PAIR_OBSERVATION",
         observations, "observation_id"),
        ("PHRASE_FRAGMENT_INGEST", "PHRASE_FRAGMENT",
         fragments, "fragment_id"),
        ("PHRASE_GROUP_RESOLUTION", "PHRASE_GROUP",
         groups, "group_id"),
    )
    values = []
    ordinal = 0
    for phase, work_kind, records, identity_key in sources:
        for record in records:
            identity = {
                "phase": phase,
                "record_id": record[identity_key],
                "work_kind": work_kind,
            }
            values.append({
                **identity,
                "format_version": 1,
                "record_kind": "NORMALIZATION_RECOVERY_V3_WORK_ITEM_V1",
                "work_id": _record_id(identity),
                "work_ordinal": ordinal,
            })
            ordinal += 1
    return tuple(values)


def _program(
        *,
        pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """从 learned outputs 编译禁用态 phrase program。"""
    return compile_normalization_recovery_v3_phrase_program(
        rule_pack_manifest_sha256=pack_manifest_sha256,
        phrase_rules=outputs["phrase-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        overlap_index=outputs["overlap-index.jsonl"],
    )


def _rule_runtime_cases(
        *,
        pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        program: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """对每条 rule 的 bare input 形成长度桶与双解释器记录。"""
    rules = outputs["phrase-rules.jsonl"]
    if not rules:
        return ()
    texts = tuple(str(item["input_text"]) for item in rules)
    indexed = execute_normalization_recovery_v3_phrase_batch(program, texts)
    reference = reference_normalization_recovery_v3_phrase_batch(program, texts)
    values = []
    for rule, indexed_result, reference_result in zip(
            rules, indexed, reference):
        blocked_ids = sorted({
            str(value)
            for step in indexed_result["steps"]
            for value in step["blocked_defeater_ids"]
        })
        if indexed_result["output_text"] == rule["output_text"]:
            outcome = "RULE_OUTPUT"
        elif blocked_ids:
            outcome = "DEFEATER_BACKOFF"
        else:
            outcome = "UNEXPECTED"
        identity = {
            "case_kind": "RULE_BARE_INPUT",
            "pack_manifest_sha256": pack_manifest_sha256,
            "rule_id": rule["rule_id"],
        }
        values.append({
            **identity,
            "blocked_defeater_ids": blocked_ids,
            "case_id": _record_id(identity),
            "format_version": 1,
            "indexed_reference_equal": int(indexed_result == reference_result),
            "indexed_result_sha256": indexed_result["result_sha256"],
            "input_scalar_length": len(str(rule["input_text"])),
            "length_bucket": _length_bucket(len(str(rule["input_text"]))),
            "outcome": outcome,
            "output_scalar_length": len(str(rule["output_text"])),
            "record_kind": NORMALIZATION_RECOVERY_V3_RUNTIME_AUDIT_CASE_KIND,
            "reference_result_sha256": reference_result["result_sha256"],
        })
    return tuple(values)


def _defeater_runtime_cases(
        *,
        pack_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """逐 defeater 核对其全部 REFUTE occurrence 的 literal predicate。"""
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in outputs["evidence.jsonl"]}
    rule_by_id = {
        str(item["rule_id"]): item for item in outputs["phrase-rules.jsonl"]}
    if (len(observation_by_id) != len(observations)
            or len(evidence_by_id) != len(outputs["evidence.jsonl"])
            or len(rule_by_id) != len(outputs["phrase-rules.jsonl"])):
        raise BroadQaExternalDataError(
            "v3 audit observation/Evidence/rule identity 重复")
    values = []
    for defeater in outputs["defeaters.jsonl"]:
        rule = rule_by_id.get(str(defeater["rule_id"]))
        if rule is None:
            raise BroadQaExternalDataError("v3 audit defeater rule 缺失")
        matched_count = 0
        mismatch_count = 0
        observation_ids = []
        for evidence_id in defeater["refute_evidence_ids"]:
            evidence = evidence_by_id.get(str(evidence_id))
            if (evidence is None or evidence.get("stance") != "REFUTE"
                    or evidence.get("family_id") != rule["family_id"]
                    or evidence.get("hypothesis_input") != rule["input_text"]
                    or evidence.get("candidate_output") != rule["output_text"]):
                raise BroadQaExternalDataError(
                    "v3 audit defeater/REFUTE Evidence 未闭合")
            observation = observation_by_id.get(str(evidence["observation_id"]))
            if observation is None:
                raise BroadQaExternalDataError(
                    "v3 audit REFUTE observation 缺失")
            start = int(evidence["occurrence_start"])
            end = int(evidence["occurrence_end"])
            surface = str(observation["input_text"])
            signature = evidence["context_signature"]
            signature_equal = all(defeater[key] == signature[key] for key in (
                "left_boundary", "left_context",
                "right_boundary", "right_context"))
            predicate = normalization_recovery_v3_defeater_matches(
                defeater, surface, start, end)
            occurrence_equal = surface[start:end] == rule["input_text"]
            matched_count += int(signature_equal and predicate and occurrence_equal)
            mismatch_count += int(
                not (signature_equal and predicate and occurrence_equal))
            observation_ids.append(str(evidence["observation_id"]))
        identity = {
            "case_kind": "DEFEATER_OCCURRENCE",
            "defeater_id": defeater["defeater_id"],
            "pack_manifest_sha256": pack_manifest_sha256,
        }
        values.append({
            **identity,
            "case_id": _record_id(identity),
            "executable_match_count": matched_count,
            "format_version": 1,
            "mismatch_count": mismatch_count,
            "observation_ids_sha256": _sha256(canonical_json_bytes(
                sorted(observation_ids))),
            "record_kind": NORMALIZATION_RECOVERY_V3_RUNTIME_AUDIT_CASE_KIND,
            "refute_evidence_count": len(defeater["refute_evidence_ids"]),
            "rule_id": defeater["rule_id"],
        })
    return tuple(values)


def _context_interpreter_summary(
        *,
        observations: tuple[dict[str, object], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        program: dict[str, object],
        ) -> dict[str, object]:
    """对全部含 REFUTE 的真实 TRAIN observation 双解释器重放。"""
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    observation_ids = sorted({
        str(item["observation_id"])
        for item in outputs["evidence.jsonl"] if item["stance"] == "REFUTE"})
    texts = tuple(str(observation_by_id[value]["input_text"])
                  for value in observation_ids)
    if not texts:
        return {
            "case_count": 0,
            "indexed_reference_mismatch_count": 0,
            "result_pairs_sha256": _sha256(canonical_json_bytes([])),
        }
    indexed = execute_normalization_recovery_v3_phrase_batch(program, texts)
    reference = reference_normalization_recovery_v3_phrase_batch(program, texts)
    pairs = [{
        "indexed_result_sha256": left["result_sha256"],
        "observation_id": observation_id,
        "reference_result_sha256": right["result_sha256"],
    } for observation_id, left, right in zip(
        observation_ids, indexed, reference)]
    return {
        "case_count": len(pairs),
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "result_pairs_sha256": _sha256(canonical_json_bytes(pairs)),
    }


def _loso_record(
        *,
        protocol_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        held_out_source_family: str,
        ) -> dict[str, object]:
    """移除一个 family 重新学习，再执行 held-out observations。"""
    training_observations = tuple(
        item for item in observations
        if item["source_family"] != held_out_source_family)
    held_out_observations = tuple(
        item for item in observations
        if item["source_family"] == held_out_source_family)
    training_ids = {
        str(item["observation_id"]) for item in training_observations}
    training_fragments = tuple(
        item for item in fragments
        if str(item["observation_id"]) in training_ids)
    if (not training_observations or not held_out_observations
            or not training_fragments):
        raise BroadQaExternalDataError("v3 LOSO source partition 为空")
    training_groups = derive_normalization_recovery_v3_groups(
        training_fragments)
    work = _work_records(
        training_observations, training_fragments, training_groups)
    subset_identity = {
        "held_out_source_family": held_out_source_family,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "training_observation_ids": sorted(training_ids),
    }
    subset_sha = _sha256(canonical_json_bytes(subset_identity))
    subset_manifest = {
        "manifest_sha256": subset_sha,
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
    }
    outputs, learning_summary, _emission_counts = (
        derive_normalization_recovery_v3_learning_outputs(
            protocol_manifest=subset_manifest,
            observations=training_observations,
            fragments=training_fragments,
            groups=training_groups,
            work=work,
        ))
    subset_pack_sha = _sha256(canonical_json_bytes({
        "learning_summary": learning_summary,
        "subset_protocol_manifest_sha256": subset_sha,
    }))
    program = _program(
        pack_manifest_sha256=subset_pack_sha,
        outputs=outputs,
    )
    texts = tuple(str(item["input_text"]) for item in held_out_observations)
    indexed = execute_normalization_recovery_v3_phrase_batch(program, texts)
    reference = reference_normalization_recovery_v3_phrase_batch(program, texts)
    outcomes = Counter()
    result_rows = []
    for observation, indexed_result, reference_result in zip(
            held_out_observations, indexed, reference):
        expected = str(observation["output_text"])
        actual = str(indexed_result["output_text"])
        if actual == expected:
            outcome = "EXACT"
        elif actual == observation["input_text"]:
            outcome = "UNKNOWN"
        else:
            outcome = "WRONG"
        outcomes[outcome] += 1
        result_rows.append({
            "actual_output_sha256": _sha256(actual.encode("utf-8")),
            "expected_output_sha256": _sha256(expected.encode("utf-8")),
            "held_out_observation_id": observation["observation_id"],
            "indexed_result_sha256": indexed_result["result_sha256"],
            "outcome": outcome,
            "reference_result_sha256": reference_result["result_sha256"],
        })
    identity = {
        "held_out_source_family": held_out_source_family,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "subset_protocol_manifest_sha256": subset_sha,
    }
    return {
        **identity,
        "format_version": 1,
        "held_out_observation_count": len(held_out_observations),
        "held_out_observation_read_for_learning_count": 0,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "learning_summary": learning_summary,
        "loso_id": _record_id(identity),
        "outcome_counts": {
            key: outcomes[key] for key in LOSO_OUTCOMES},
        "program_sha256": program["program_sha256"],
        "record_kind": NORMALIZATION_RECOVERY_V3_LOSO_AUDIT_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "selection_leakage_count": 0,
        "training_fragment_count": len(training_fragments),
        "training_group_count": len(training_groups),
        "training_observation_count": len(training_observations),
        "training_phrase_rule_count": learning_summary["phrase_rule_count"],
        "training_source_families": sorted({
            str(item["source_family"]) for item in training_observations}),
    }


def derive_normalization_recovery_v3_training_audit(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        pack_manifest: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生完整 TRAIN-only runtime cases、LOSO records 与摘要。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v3 audit protocol manifest")
    pack_sha = _sha_value(
        pack_manifest.get("manifest_sha256"),
        label="v3 audit pack manifest")
    if (not observations or not fragments or not groups
            or pack_manifest.get("production_enabled") != 0
            or pack_manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v3 audit input material 漂移")
    program = _program(pack_manifest_sha256=pack_sha, outputs=outputs)
    rule_cases = _rule_runtime_cases(
        pack_manifest_sha256=pack_sha,
        outputs=outputs,
        program=program,
    )
    defeater_cases = _defeater_runtime_cases(
        pack_manifest_sha256=pack_sha,
        observations=observations,
        outputs=outputs,
    )
    runtime_cases = tuple(sorted(
        rule_cases + defeater_cases, key=lambda item: str(item["case_id"])))
    loso = tuple(sorted((
        _loso_record(
            protocol_manifest_sha256=protocol_sha,
            observations=observations,
            fragments=fragments,
            held_out_source_family=family,
        ) for family in SOURCE_FAMILIES), key=lambda item: str(item["loso_id"])))
    rule_outcomes = Counter(str(item["outcome"]) for item in rule_cases)
    bucket_counts = Counter(str(item["length_bucket"]) for item in rule_cases)
    context_summary = _context_interpreter_summary(
        observations=observations,
        outputs=outputs,
        program=program,
    )
    loso_outcomes = Counter()
    for item in loso:
        for outcome, count in item["outcome_counts"].items():
            loso_outcomes[(str(item["held_out_source_family"]), outcome)] += count
    facility_failures = (
        sum(item["indexed_reference_equal"] == 0 for item in rule_cases)
        + sum(int(item["mismatch_count"]) for item in defeater_cases)
        + int(context_summary["indexed_reference_mismatch_count"])
        + sum(int(item["indexed_reference_mismatch_count"]) for item in loso)
        + rule_outcomes["UNEXPECTED"]
    )
    summary = {
        "audit_outcome": (
            "FACILITY_PASS_CAPABILITY_DIAGNOSTIC"
            if facility_failures == 0 else "FACILITY_FAIL"),
        "context_interpreter": context_summary,
        "defeater_case_count": len(defeater_cases),
        "defeater_mismatch_count": sum(
            int(item["mismatch_count"]) for item in defeater_cases),
        "facility_failure_count": facility_failures,
        "length_bucket_counts": {
            key: bucket_counts[key] for key in LENGTH_BUCKETS},
        "loso_counts": {
            f"{family}:{outcome}": loso_outcomes[(family, outcome)]
            for family in SOURCE_FAMILIES for outcome in LOSO_OUTCOMES},
        "loso_family_count": len(loso),
        "phrase_program_sha256": program["program_sha256"],
        "rule_bare_outcome_counts": {
            key: rule_outcomes[key] for key in RULE_BARE_OUTCOMES},
        "rule_case_count": len(rule_cases),
        "runtime_case_count": len(runtime_cases),
    }
    return runtime_cases, loso, summary


__all__ = [
    "LENGTH_BUCKETS",
    "LOSO_OUTCOMES",
    "NORMALIZATION_RECOVERY_V3_LOSO_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V3_RUNTIME_AUDIT_CASE_KIND",
    "RULE_BARE_OUTCOMES",
    "SOURCE_FAMILIES",
    "derive_normalization_recovery_v3_training_audit",
]
