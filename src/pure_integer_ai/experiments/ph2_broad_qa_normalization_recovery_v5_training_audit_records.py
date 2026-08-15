"""派生 recovery-v5 TRAIN-only runtime、identity 与四方向 LOSO audit。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    compile_normalization_recovery_v5_phrase_program,
    execute_normalization_recovery_v5_phrase_batch,
    normalization_recovery_v5_defeater_matches,
    reference_normalization_recovery_v5_phrase_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
    V5_SOURCE_FAMILIES,
    derive_normalization_recovery_v5_groups,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND = (
    "NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_V1")
NORMALIZATION_RECOVERY_V5_LOSO_AUDIT_KIND = (
    "NORMALIZATION_RECOVERY_V5_FOUR_SOURCE_LOSO_AUDIT_V1")
AUDIT_BUCKETS = (
    "IDENTITY",
    "CHARACTER_LOCAL",
    "WHOLE_INPUT_EQUAL_LENGTH",
    "WHOLE_INPUT_VARIABLE_LENGTH",
    "CONTEXT_HUNK",
)
LOSO_OUTCOMES = ("EXACT", "UNKNOWN", "WRONG")
RULE_BARE_OUTCOMES = (
    "DEFEATER_BACKOFF",
    "PRIORITY_SHADOWED",
    "RULE_OUTPUT",
    "UNEXPECTED",
)


def _sha256(payload: bytes) -> str:
    """返回规范 audit record 摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 record id。"""
    return _sha256(canonical_json_bytes(identity))


def _work_records(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """为 LOSO 子集重建 v5 protocol 同构的 ordered work。"""
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
                "record_kind": "NORMALIZATION_RECOVERY_V5_WORK_ITEM_V1",
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
    return compile_normalization_recovery_v5_phrase_program(
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


def _support_structure_tokens(
        rule: dict[str, object],
        evidence_by_id: dict[str, dict[str, object]],
        ) -> tuple[str, ...]:
    """从 rule 的第一条 SUPPORT 恢复运行所需结构 token。"""
    values = [evidence_by_id[str(value)] for value in rule[
        "positive_evidence_ids"]]
    if not values:
        raise BroadQaExternalDataError("v5 audit rule SUPPORT 缺失")
    tokens = values[0].get("structure_tokens")
    if not isinstance(tokens, list) or any(not isinstance(item, str)
                                            for item in tokens):
        raise BroadQaExternalDataError("v5 audit SUPPORT structure 漂移")
    return tuple(tokens)


def normalization_recovery_v5_result_bucket(
        observation: dict[str, object],
        result: dict[str, object],
        ) -> str:
    """按实际 program 路径把一次留出结果归入五个互斥 audit bucket。"""
    if observation["identity_preservation"] == 1:
        return "IDENTITY"
    for step in result["steps"]:
        if step["mode"] == "WHOLE_INPUT_EXACT":
            return str(step["rule_class"])
    for step in result["steps"]:
        if step["mode"] == "LONGEST_LOCAL_MATCH":
            rule_class = str(step["rule_class"])
            if rule_class == "EDIT_CORE":
                return "CHARACTER_LOCAL"
            return rule_class
    return "CHARACTER_LOCAL"


def _rule_runtime_cases(
        *,
        pack_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        program: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """对每条 scoped rule 的 bare input 形成双解释器记录。"""
    evidence_by_id = {str(item["evidence_id"]): item
                      for item in outputs["evidence.jsonl"]}
    rules_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for rule in _all_rules(outputs):
        rules_by_source[str(rule["source_execution_family"])].append(rule)
    values = []
    for source_family, rules in sorted(rules_by_source.items()):
        texts = tuple(str(item["input_text"]) for item in rules)
        structures = tuple(_support_structure_tokens(item, evidence_by_id)
                           for item in rules)
        indexed_results = execute_normalization_recovery_v5_phrase_batch(
            program, texts, source_family=source_family,
            structure_tokens=structures)
        reference_results = reference_normalization_recovery_v5_phrase_batch(
            program, texts, source_family=source_family,
            structure_tokens=structures)
        for rule, structure, indexed, reference in zip(
                rules, structures, indexed_results, reference_results):
            blocked_ids = sorted({
                str(value) for step in indexed["steps"]
                for value in step["blocked_defeater_ids"]})
            selected_rule_ids = [
                str(step["rule_id"]) for step in indexed["steps"]
                if step["rule_id"]]
            rule_id = str(rule["rule_id"])
            rule_blocked = bool(
                set(blocked_ids).intersection(str(value) for value in rule[
                    "defeater_ids"]))
            if rule_id in selected_rule_ids:
                outcome = (
                    "RULE_OUTPUT"
                    if indexed["output_text"] == rule["output_text"]
                    else "UNEXPECTED")
            elif rule_blocked:
                outcome = "DEFEATER_BACKOFF"
            elif selected_rule_ids:
                outcome = "PRIORITY_SHADOWED"
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
                "outcome": outcome,
                "output_scalar_length": len(str(rule["output_text"])),
                "record_kind": NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND,
                "reference_result_sha256": reference["result_sha256"],
                "rule_class": rule["rule_class"],
                "selected_rule_ids": selected_rule_ids,
                "source_execution_family": source_family,
                "structure_token_count": len(structure),
            })
    return tuple(values)


def _defeater_runtime_cases(
        *,
        pack_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """逐 defeater 核对其全部 scoped REFUTE occurrence。"""
    observation_by_id = {str(item["observation_id"]): item
                         for item in observations}
    evidence_by_id = {str(item["evidence_id"]): item
                      for item in outputs["evidence.jsonl"]}
    rule_by_id = {str(item["rule_id"]): item for item in _all_rules(outputs)}
    values = []
    for defeater in outputs["defeaters.jsonl"]:
        rule = rule_by_id.get(str(defeater["rule_id"]))
        if rule is None:
            raise BroadQaExternalDataError("v5 audit defeater rule 缺失")
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
                    or evidence.get("rule_class") != rule["rule_class"]
                    or any(evidence.get(key) != rule[key]
                           for key in scope_keys)):
                raise BroadQaExternalDataError(
                    "v5 audit defeater/REFUTE Evidence 未闭合")
            observation = observation_by_id.get(str(evidence["observation_id"]))
            if observation is None:
                raise BroadQaExternalDataError(
                    "v5 audit REFUTE observation 缺失")
            start = int(evidence["occurrence_start"])
            end = int(evidence["occurrence_end"])
            surface = str(observation["input_text"])
            signature = evidence["context_signature"]
            signature_equal = all(defeater[key] == signature[key] for key in (
                "left_boundary", "left_context", "right_boundary",
                "right_context"))
            predicate = normalization_recovery_v5_defeater_matches(
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
            "record_kind": NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND,
            "refute_evidence_count": len(defeater["refute_evidence_ids"]),
            "rule_class": rule["rule_class"],
            "rule_id": defeater["rule_id"],
        })
    return tuple(values)


def _identity_runtime_cases(
        *,
        pack_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        program: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """对全部 identity observation 执行无 source scope 的 hard audit。"""
    selected = tuple(item for item in observations
                     if item["identity_preservation"] == 1)
    texts = tuple(str(item["input_text"]) for item in selected)
    structures = tuple(tuple(item["structure_tokens"]) for item in selected)
    indexed = execute_normalization_recovery_v5_phrase_batch(
        program, texts, structure_tokens=structures)
    reference = reference_normalization_recovery_v5_phrase_batch(
        program, texts, structure_tokens=structures)
    values = []
    for observation, left, right in zip(selected, indexed, reference):
        identity = {
            "case_kind": "IDENTITY_PRESERVATION",
            "observation_id": observation["observation_id"],
            "pack_manifest_sha256": pack_manifest_sha256,
        }
        values.append({
            **identity,
            "actual_output_sha256": _sha256(
                str(left["output_text"]).encode("utf-8")),
            "case_id": _record_id(identity),
            "expected_output_sha256": _sha256(
                str(observation["input_text"]).encode("utf-8")),
            "format_version": 1,
            "identity_false_change": int(
                left["output_text"] != observation["input_text"]),
            "indexed_reference_equal": int(left == right),
            "record_kind": NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND,
            "reference_result_sha256": right["result_sha256"],
            "source_family": observation["source_family"],
        })
    return tuple(values)


def _context_interpreter_summary(
        *,
        observations: tuple[dict[str, object], ...],
        program: dict[str, object],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """对全部 TRAIN observation 执行 scoped 双解释器并保留摘要行。"""
    rows = []
    mismatch_count = 0
    for family in V5_SOURCE_FAMILIES:
        selected = tuple(item for item in observations
                         if item["source_family"] == family)
        texts = tuple(str(item["input_text"]) for item in selected)
        structures = tuple(tuple(item["structure_tokens"]) for item in selected)
        indexed = execute_normalization_recovery_v5_phrase_batch(
            program, texts, source_family=family, structure_tokens=structures)
        reference = reference_normalization_recovery_v5_phrase_batch(
            program, texts, source_family=family, structure_tokens=structures)
        mismatch_count += sum(left != right for left, right in zip(
            indexed, reference))
        rows.extend({
            "indexed_result_sha256": left["result_sha256"],
            "observation_id": observation["observation_id"],
            "reference_result_sha256": right["result_sha256"],
        } for observation, left, right in zip(selected, indexed, reference))
    rows.sort(key=lambda item: str(item["observation_id"]))
    summary = {
        "case_count": len(rows),
        "indexed_reference_mismatch_count": mismatch_count,
        "result_pairs_sha256": _sha256(canonical_json_bytes(rows)),
    }
    return summary, tuple(rows)


def derive_normalization_recovery_v5_loso_execution(
        *,
        protocol_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        held_out_source_family: str,
        include_reference: bool,
        ) -> dict[str, object]:
    """移除一个 family 重学，并返回 held-out 内存执行材料。"""
    protocol_sha = _sha_value(
        protocol_manifest_sha256, label="v5 LOSO protocol manifest")
    if (held_out_source_family not in V5_SOURCE_FAMILIES
            or type(include_reference) is not bool):
        raise BroadQaExternalDataError("v5 LOSO execution 参数漂移")
    training_observations = tuple(
        item for item in observations
        if item["source_family"] != held_out_source_family)
    held_out_observations = tuple(
        item for item in observations
        if item["source_family"] == held_out_source_family)
    training_ids = {str(item["observation_id"])
                    for item in training_observations}
    training_fragments = tuple(
        item for item in fragments
        if str(item["observation_id"]) in training_ids)
    if (not training_observations or not held_out_observations
            or not training_fragments
            or len({str(item["source_family"])
                    for item in training_observations}) != 3):
        raise BroadQaExternalDataError("v5 LOSO source partition 漂移")
    training_groups = derive_normalization_recovery_v5_groups(
        training_fragments)
    work = _work_records(
        training_observations, training_fragments, training_groups)
    subset_identity = {
        "held_out_source_family": held_out_source_family,
        "protocol_manifest_sha256": protocol_sha,
        "training_observation_ids": sorted(training_ids),
    }
    subset_sha = _sha256(canonical_json_bytes(subset_identity))
    subset_manifest = {
        "learner_contract": {
            "identity_preservation_hard_gate_required": 1,
            "negative_evidence_required_before_execution": 1,
            "source_scoped_candidate_target_upgrade_allowed": 0,
            "target_equal_length_min_distinct_source_family_count": 2,
            "target_variable_length_min_distinct_source_family_count": 3,
            "target_variable_length_two_family_replicated_support_allowed": 1,
            "whole_input_exact_precedes_phrase_lexicon": 1,
        },
        "manifest_sha256": subset_sha,
        "target_policy_scope": RECOVERY_V5_TARGET_POLICY_SCOPE,
    }
    outputs, learning_summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
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
    program = _program(pack_manifest_sha256=subset_pack_sha, outputs=outputs)
    texts = tuple(str(item["input_text"]) for item in held_out_observations)
    structures = tuple(tuple(item["structure_tokens"])
                       for item in held_out_observations)
    indexed = execute_normalization_recovery_v5_phrase_batch(
        program, texts, source_family=held_out_source_family,
        structure_tokens=structures)
    reference = (
        reference_normalization_recovery_v5_phrase_batch(
            program, texts, source_family=held_out_source_family,
            structure_tokens=structures)
        if include_reference else ())
    return {
        "held_out_observations": held_out_observations,
        "indexed_results": indexed,
        "learning_summary": learning_summary,
        "outputs": outputs,
        "program": program,
        "reference_results": reference,
        "subset_pack_manifest_sha256": subset_pack_sha,
        "subset_protocol_manifest_sha256": subset_sha,
        "training_fragments": training_fragments,
        "training_groups": training_groups,
        "training_observations": training_observations,
    }


def _loso_record(
        *,
        protocol_manifest_sha256: str,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        held_out_source_family: str,
        ) -> dict[str, object]:
    """移除一个 family 重新学习，再执行 held-out observations。"""
    material = derive_normalization_recovery_v5_loso_execution(
        protocol_manifest_sha256=protocol_manifest_sha256,
        observations=observations,
        fragments=fragments,
        held_out_source_family=held_out_source_family,
        include_reference=True,
    )
    training_observations = material["training_observations"]
    held_out_observations = material["held_out_observations"]
    training_fragments = material["training_fragments"]
    training_groups = material["training_groups"]
    outputs = material["outputs"]
    learning_summary = material["learning_summary"]
    subset_sha = material["subset_protocol_manifest_sha256"]
    subset_pack_sha = material["subset_pack_manifest_sha256"]
    program = material["program"]
    indexed = material["indexed_results"]
    reference = material["reference_results"]
    outcomes = Counter()
    bucket_outcomes = {
        bucket: Counter() for bucket in AUDIT_BUCKETS}
    result_rows = []
    identity_false_change = 0
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
        bucket = normalization_recovery_v5_result_bucket(
            observation, indexed_result)
        if bucket not in bucket_outcomes:
            raise BroadQaExternalDataError("v5 LOSO bucket 漂移")
        outcomes[outcome] += 1
        bucket_outcomes[bucket][outcome] += 1
        identity_false_change += int(
            observation["identity_preservation"] == 1
            and actual != observation["input_text"])
        result_rows.append({
            "actual_output_sha256": _sha256(actual.encode("utf-8")),
            "bucket": bucket,
            "expected_output_sha256": _sha256(expected.encode("utf-8")),
            "held_out_observation_id": observation["observation_id"],
            "indexed_result_sha256": indexed_result["result_sha256"],
            "outcome": outcome,
            "reference_result_sha256": reference_result["result_sha256"],
        })
    result_rows.sort(key=lambda item: str(item["held_out_observation_id"]))
    subset_defeaters = _defeater_runtime_cases(
        pack_manifest_sha256=subset_pack_sha,
        observations=training_observations,
        outputs=outputs,
    )
    source_leak_count = sum(
        item["candidate_scope_kind"] != "SOURCE_ONLY"
        or item["source_execution_family"] not in {
            str(value["source_family"]) for value in training_observations}
        for item in outputs["source-phrase-rules.jsonl"])
    identity = {
        "held_out_source_family": held_out_source_family,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "subset_protocol_manifest_sha256": subset_sha,
    }
    return {
        **identity,
        "bucket_outcome_counts": {
            bucket: {key: bucket_outcomes[bucket][key]
                     for key in LOSO_OUTCOMES}
            for bucket in AUDIT_BUCKETS},
        "format_version": 1,
        "held_out_observation_count": len(held_out_observations),
        "held_out_observation_read_for_learning_count": 0,
        "identity_false_change_count": identity_false_change,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "learning_summary": learning_summary,
        "loso_id": _record_id(identity),
        "outcome_counts": {key: outcomes[key] for key in LOSO_OUTCOMES},
        "program_sha256": program["program_sha256"],
        "record_kind": NORMALIZATION_RECOVERY_V5_LOSO_AUDIT_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(result_rows)),
        "selection_leakage_count": 0,
        "source_leak_count": source_leak_count,
        "subset_defeater_mismatch_count": sum(
            int(item["mismatch_count"]) for item in subset_defeaters),
        "training_fragment_count": len(training_fragments),
        "training_group_count": len(training_groups),
        "training_observation_count": len(training_observations),
        "training_source_families": sorted({
            str(item["source_family"]) for item in training_observations}),
    }


def derive_normalization_recovery_v5_training_audit(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        pack_manifest: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], dict[str, object]]:
    """派生完整 TRAIN-only runtime cases、identity、四方向 LOSO 与摘要。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v5 audit protocol manifest")
    pack_sha = _sha_value(
        pack_manifest.get("manifest_sha256"),
        label="v5 audit pack manifest")
    if (not observations or not fragments or not groups
            or pack_manifest.get("production_enabled") != 0
            or pack_manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v5 audit input material 漂移")
    program = _program(pack_manifest_sha256=pack_sha, outputs=outputs)
    rule_cases = _rule_runtime_cases(
        pack_manifest_sha256=pack_sha,
        observations=observations,
        outputs=outputs,
        program=program,
    )
    defeater_cases = _defeater_runtime_cases(
        pack_manifest_sha256=pack_sha,
        observations=observations,
        outputs=outputs,
    )
    identity_cases = _identity_runtime_cases(
        pack_manifest_sha256=pack_sha,
        observations=observations,
        program=program,
    )
    context_summary, context_rows = _context_interpreter_summary(
        observations=observations,
        program=program,
    )
    context_cases = tuple({
        "case_kind": "CONTEXT_INTERPRETER",
        "case_id": _record_id({
            "case_kind": "CONTEXT_INTERPRETER",
            "observation_id": row["observation_id"],
            "pack_manifest_sha256": pack_sha,
        }),
        "format_version": 1,
        "indexed_reference_equal": int(
            row["indexed_result_sha256"] == row["reference_result_sha256"]),
        "indexed_result_sha256": row["indexed_result_sha256"],
        "observation_id": row["observation_id"],
        "pack_manifest_sha256": pack_sha,
        "record_kind": NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND,
        "reference_result_sha256": row["reference_result_sha256"],
    } for row in context_rows)
    runtime_cases = tuple(sorted(
        rule_cases + defeater_cases + identity_cases + context_cases,
        key=lambda item: str(item["case_id"])))
    loso = tuple(sorted((
        _loso_record(
            protocol_manifest_sha256=protocol_sha,
            observations=observations,
            fragments=fragments,
            held_out_source_family=family,
        ) for family in V5_SOURCE_FAMILIES),
        key=lambda item: str(item["loso_id"])))
    rule_outcomes = Counter(str(item["outcome"]) for item in rule_cases)
    loso_outcomes = Counter()
    for item in loso:
        for outcome, count in item["outcome_counts"].items():
            loso_outcomes[(str(item["held_out_source_family"]), outcome)] += count
    bucket_outcomes = Counter()
    for item in loso:
        for bucket, counts in item["bucket_outcome_counts"].items():
            for outcome, count in counts.items():
                bucket_outcomes[(bucket, outcome)] += count
    facility_failures = (
        sum(item["indexed_reference_equal"] == 0 for item in rule_cases)
        + sum(int(item["mismatch_count"]) for item in defeater_cases)
        + sum(int(item["identity_false_change"]) for item in identity_cases)
        + int(context_summary["indexed_reference_mismatch_count"])
        + sum(int(item["indexed_reference_mismatch_count"]) for item in loso)
        + sum(int(item["subset_defeater_mismatch_count"]) for item in loso)
        + sum(int(item["source_leak_count"]) for item in loso)
        + sum(int(item["selection_leakage_count"]) for item in loso)
        + rule_outcomes["UNEXPECTED"])
    capability_pass = int(
        all(item["outcome_counts"]["WRONG"] == 0
            and item["outcome_counts"]["EXACT"] > 0 for item in loso)
        and all(bucket_outcomes[(bucket, "WRONG")] == 0
                for bucket in AUDIT_BUCKETS)
        and all(bucket_outcomes[(bucket, "EXACT")] > 0
                for bucket in ("WHOLE_INPUT_EQUAL_LENGTH",
                               "WHOLE_INPUT_VARIABLE_LENGTH",
                               "CONTEXT_HUNK")))
    summary = {
        "audit_outcome": (
            "FACILITY_FAIL" if facility_failures
            else "FACILITY_PASS_CAPABILITY_PASS" if capability_pass
            else "FACILITY_PASS_CAPABILITY_FAIL"),
        "bucket_outcome_counts": {
            f"{bucket}:{outcome}": bucket_outcomes[(bucket, outcome)]
            for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES},
        "capability_gate_pass": capability_pass,
        "context_interpreter": context_summary,
        "defeater_case_count": len(defeater_cases),
        "defeater_mismatch_count": sum(
            int(item["mismatch_count"]) for item in defeater_cases),
        "facility_failure_count": facility_failures,
        "full_pack_identity_case_count": len(identity_cases),
        "full_pack_source_rule_count": len(
            outputs["source-phrase-rules.jsonl"]),
        "full_pack_target_rule_count": len(
            outputs["target-phrase-rules.jsonl"]),
        "loso_counts": {
            f"{family}:{outcome}": loso_outcomes[(family, outcome)]
            for family in V5_SOURCE_FAMILIES for outcome in LOSO_OUTCOMES},
        "loso_family_count": len(loso),
        "phrase_program_sha256": program["program_sha256"],
        "rule_bare_outcome_counts": {
            key: rule_outcomes[key] for key in RULE_BARE_OUTCOMES},
        "rule_case_count": len(rule_cases),
        "runtime_case_count": len(runtime_cases),
        "source_leak_count": sum(
            int(item["source_leak_count"]) for item in loso),
        "subset_defeater_mismatch_count": sum(
            int(item["subset_defeater_mismatch_count"]) for item in loso),
    }
    return runtime_cases, loso, summary


__all__ = [
    "AUDIT_BUCKETS",
    "LOSO_OUTCOMES",
    "NORMALIZATION_RECOVERY_V5_LOSO_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V5_RUNTIME_AUDIT_CASE_KIND",
    "RULE_BARE_OUTCOMES",
    "derive_normalization_recovery_v5_loso_execution",
    "derive_normalization_recovery_v5_training_audit",
    "normalization_recovery_v5_result_bucket",
]
