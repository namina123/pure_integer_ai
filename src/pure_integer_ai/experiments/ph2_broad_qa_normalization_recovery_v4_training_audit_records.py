"""纯派生 recovery-v4 scoped runtime、defeater 与三来源 LOSO 审计。

LOSO 每次完整移除一个 source family，重新构造 group/work 并重新学习；执行
held-out 时只能使用 target rule，因为该 family 的 source rule 不存在。正式门为
每个方向 WRONG=0、EXACT>0，且 indexed/reference 与 defeater 审计全通过。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_records import (
    derive_normalization_recovery_v4_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_phrase_runtime import (
    compile_normalization_recovery_v4_phrase_program,
    execute_normalization_recovery_v4_phrase_batch,
    normalization_recovery_v4_defeater_matches,
    reference_normalization_recovery_v4_phrase_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    V4_SOURCE_FAMILIES,
    derive_normalization_recovery_v4_groups,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V4_RUNTIME_AUDIT_CASE_KIND = (
    "NORMALIZATION_RECOVERY_V4_RUNTIME_AUDIT_CASE_V1")
NORMALIZATION_RECOVERY_V4_LOSO_AUDIT_KIND = (
    "NORMALIZATION_RECOVERY_V4_LOSO_AUDIT_V1")

LENGTH_BUCKETS = ("L02", "L03_04", "L05_08", "L09_16", "L17_40")
RULE_BARE_OUTCOMES = ("DEFEATER_BACKOFF", "RULE_OUTPUT", "UNEXPECTED")
LOSO_OUTCOMES = ("EXACT", "UNKNOWN", "WRONG")


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
    """按 phrase scalar 长度返回互斥桶。"""
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
    raise BroadQaExternalDataError("v4 audit phrase length 越界")


def _work_records(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """为 LOSO 子集重建与正式 v4 protocol 同构的 ordered work。"""
    sources = (
        ("PAIR_OBSERVATION_INGEST", "PAIR_OBSERVATION",
         observations, "observation_id"),
        ("PHRASE_FRAGMENT_INGEST", "PHRASE_FRAGMENT",
         fragments, "fragment_id"),
        ("PHRASE_GROUP_RESOLUTION", "PHRASE_GROUP", groups, "group_id"),
    )
    values = []
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
                "record_kind": "NORMALIZATION_RECOVERY_V4_WORK_ITEM_V1",
                "work_id": _record_id(identity),
                "work_ordinal": len(values),
            })
    return tuple(values)


def _program(
        *,
        pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """从 learned outputs 编译禁用态 scoped phrase program。"""
    return compile_normalization_recovery_v4_phrase_program(
        rule_pack_manifest_sha256=pack_manifest_sha256,
        target_phrase_rules=outputs["target-phrase-rules.jsonl"],
        source_phrase_rules=outputs["source-phrase-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        target_overlap_index=outputs["target-overlap-index.jsonl"],
        source_overlap_index=outputs["source-overlap-index.jsonl"],
    )


def _all_rules(outputs: dict[str, tuple[dict[str, object], ...]]):
    """按 identity 返回 target/source rule 的稳定联合序。"""
    return tuple(sorted(
        outputs["target-phrase-rules.jsonl"]
        + outputs["source-phrase-rules.jsonl"],
        key=lambda item: str(item["rule_id"]),
    ))


def _rule_runtime_cases(
        *,
        pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        program: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """对每条 scoped rule 的 bare input 形成双解释器记录。"""
    values = []
    rules_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rule in _all_rules(outputs):
        rules_by_source[str(rule["source_execution_family"])].append(rule)
    for source_family, rules in sorted(rules_by_source.items()):
        texts = tuple(str(item["input_text"]) for item in rules)
        indexed_results = execute_normalization_recovery_v4_phrase_batch(
            program, texts, source_family=source_family)
        reference_results = reference_normalization_recovery_v4_phrase_batch(
            program, texts, source_family=source_family)
        for rule, indexed, reference in zip(
                rules, indexed_results, reference_results):
            blocked_ids = sorted({
                str(value) for step in indexed["steps"]
                for value in step["blocked_defeater_ids"]})
            if indexed["output_text"] == rule["output_text"]:
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
                "candidate_scope_kind": rule["candidate_scope_kind"],
                "case_id": _record_id(identity),
                "format_version": 1,
                "indexed_reference_equal": int(indexed == reference),
                "indexed_result_sha256": indexed["result_sha256"],
                "input_scalar_length": len(str(rule["input_text"])),
                "length_bucket": _length_bucket(len(str(rule["input_text"]))),
                "outcome": outcome,
                "output_scalar_length": len(str(rule["output_text"])),
                "record_kind": (
                    NORMALIZATION_RECOVERY_V4_RUNTIME_AUDIT_CASE_KIND),
                "reference_result_sha256": reference["result_sha256"],
                "source_execution_family": source_family,
            })
    return tuple(values)


def _defeater_runtime_cases(
        *,
        pack_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """逐 defeater 核对其全部 scoped REFUTE occurrence。"""
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    evidence_by_id = {
        str(item["evidence_id"]): item for item in outputs["evidence.jsonl"]}
    rule_by_id = {str(item["rule_id"]): item for item in _all_rules(outputs)}
    if (len(observation_by_id) != len(observations)
            or len(evidence_by_id) != len(outputs["evidence.jsonl"])
            or len(rule_by_id) != len(_all_rules(outputs))):
        raise BroadQaExternalDataError(
            "v4 audit observation/Evidence/rule identity 重复")
    values = []
    for defeater in outputs["defeaters.jsonl"]:
        rule = rule_by_id.get(str(defeater["rule_id"]))
        if rule is None:
            raise BroadQaExternalDataError("v4 audit defeater rule 缺失")
        matched_count = 0
        mismatch_count = 0
        observation_ids = []
        for evidence_id in defeater["refute_evidence_ids"]:
            evidence = evidence_by_id.get(str(evidence_id))
            scope_keys = (
                "candidate_scope_kind", "source_execution_family",
                "source_execution_policy_scope", "target_policy_scope")
            if (evidence is None or evidence.get("stance") != "REFUTE"
                    or evidence.get("candidate_id") != rule["candidate_id"]
                    or evidence.get("hypothesis_input") != rule["input_text"]
                    or evidence.get("candidate_output") != rule["output_text"]
                    or any(evidence.get(key) != rule[key]
                           for key in scope_keys)):
                raise BroadQaExternalDataError(
                    "v4 audit defeater/REFUTE Evidence 未闭合")
            observation = observation_by_id.get(str(evidence["observation_id"]))
            if observation is None:
                raise BroadQaExternalDataError(
                    "v4 audit REFUTE observation 缺失")
            start = int(evidence["occurrence_start"])
            end = int(evidence["occurrence_end"])
            surface = str(observation["input_text"])
            signature = evidence["context_signature"]
            signature_equal = all(defeater[key] == signature[key] for key in (
                "left_boundary", "left_context",
                "right_boundary", "right_context"))
            predicate = normalization_recovery_v4_defeater_matches(
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
            "record_kind": NORMALIZATION_RECOVERY_V4_RUNTIME_AUDIT_CASE_KIND,
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
    """对全部含 REFUTE 的 observation 执行 scoped 双解释器。"""
    refute_ids = {
        str(item["observation_id"]) for item in outputs["evidence.jsonl"]
        if item["stance"] == "REFUTE"}
    selected = tuple(item for item in observations
                     if str(item["observation_id"]) in refute_ids)
    rows = []
    mismatch_count = 0
    for family in V4_SOURCE_FAMILIES:
        family_observations = tuple(
            item for item in selected if item["source_family"] == family)
        texts = tuple(str(item["input_text"]) for item in family_observations)
        indexed = execute_normalization_recovery_v4_phrase_batch(
            program, texts, source_family=family)
        reference = reference_normalization_recovery_v4_phrase_batch(
            program, texts, source_family=family)
        mismatch_count += sum(left != right for left, right in zip(
            indexed, reference))
        rows.extend({
            "indexed_result_sha256": left["result_sha256"],
            "observation_id": observation["observation_id"],
            "reference_result_sha256": right["result_sha256"],
        } for observation, left, right in zip(
            family_observations, indexed, reference))
    rows.sort(key=lambda item: str(item["observation_id"]))
    return {
        "case_count": len(rows),
        "indexed_reference_mismatch_count": mismatch_count,
        "result_pairs_sha256": _sha256(canonical_json_bytes(rows)),
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
            or not training_fragments
            or len({str(item["source_family"])
                    for item in training_observations}) != 2):
        raise BroadQaExternalDataError("v4 LOSO source partition 漂移")
    training_groups = derive_normalization_recovery_v4_groups(
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
        "learner_contract": {
            "source_scoped_candidate_target_upgrade_allowed": 0,
            "target_candidate_min_distinct_ui_source_family_count": 2,
        },
        "manifest_sha256": subset_sha,
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
    }
    outputs, learning_summary, _emission_counts = (
        derive_normalization_recovery_v4_learning_outputs(
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
    indexed = execute_normalization_recovery_v4_phrase_batch(
        program, texts, source_family=held_out_source_family)
    reference = reference_normalization_recovery_v4_phrase_batch(
        program, texts, source_family=held_out_source_family)
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
        "outcome_counts": {key: outcomes[key] for key in LOSO_OUTCOMES},
        "program_sha256": program["program_sha256"],
        "record_kind": NORMALIZATION_RECOVERY_V4_LOSO_AUDIT_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "selection_leakage_count": 0,
        "training_fragment_count": len(training_fragments),
        "training_group_count": len(training_groups),
        "training_observation_count": len(training_observations),
        "training_source_families": sorted({
            str(item["source_family"]) for item in training_observations}),
    }


def derive_normalization_recovery_v4_training_audit(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        pack_manifest: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], dict[str, object]]:
    """派生完整 TRAIN-only runtime cases、三方向 LOSO 与摘要。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v4 audit protocol manifest")
    pack_sha = _sha_value(
        pack_manifest.get("manifest_sha256"),
        label="v4 audit pack manifest")
    if (not observations or not fragments or not groups
            or pack_manifest.get("production_enabled") != 0
            or pack_manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v4 audit input material 漂移")
    program = _program(pack_manifest_sha256=pack_sha, outputs=outputs)
    rule_cases = _rule_runtime_cases(
        pack_manifest_sha256=pack_sha, outputs=outputs, program=program)
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
        ) for family in V4_SOURCE_FAMILIES),
        key=lambda item: str(item["loso_id"])))
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
    capability_pass = int(all(
        item["outcome_counts"]["WRONG"] == 0
        and item["outcome_counts"]["EXACT"] > 0 for item in loso))
    summary = {
        "audit_outcome": (
            "FACILITY_FAIL" if facility_failures
            else "FACILITY_PASS_CAPABILITY_PASS" if capability_pass
            else "FACILITY_PASS_CAPABILITY_FAIL"),
        "capability_gate_pass": capability_pass,
        "context_interpreter": context_summary,
        "defeater_case_count": len(defeater_cases),
        "defeater_mismatch_count": sum(
            int(item["mismatch_count"]) for item in defeater_cases),
        "facility_failure_count": facility_failures,
        "full_pack_source_rule_count": len(
            outputs["source-phrase-rules.jsonl"]),
        "full_pack_target_rule_count": len(
            outputs["target-phrase-rules.jsonl"]),
        "length_bucket_counts": {
            key: bucket_counts[key] for key in LENGTH_BUCKETS},
        "loso_counts": {
            f"{family}:{outcome}": loso_outcomes[(family, outcome)]
            for family in V4_SOURCE_FAMILIES for outcome in LOSO_OUTCOMES},
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
    "NORMALIZATION_RECOVERY_V4_LOSO_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V4_RUNTIME_AUDIT_CASE_KIND",
    "RULE_BARE_OUTCOMES",
    "derive_normalization_recovery_v4_training_audit",
]
