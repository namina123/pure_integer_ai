"""派生 recovery-v6 full-pack facility 与四方向 TRAIN-only audit。"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit import (
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    AUDIT_BUCKETS,
    LOSO_OUTCOMES,
    derive_normalization_recovery_v5_loso_execution,
    normalization_recovery_v5_result_bucket,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation import (
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_KIND,
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation_records import (
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_KIND,
    NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_records import (
    derive_normalization_recovery_v6_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    compile_normalization_recovery_v6_phrase_program,
    execute_normalization_recovery_v6_phrase_batch,
    reference_normalization_recovery_v6_phrase_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_rule_pack import (
    NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND,
    NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V6_RUNTIME_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_RUNTIME_AUDIT_V1")
NORMALIZATION_RECOVERY_V6_LOSO_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_LOSO_AUDIT_V1")


def _sha256(payload: bytes) -> str:
    """返回规范 audit 记录或集合摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 audit record id。"""
    return _sha256(canonical_json_bytes(identity))


def _outcome(observation: dict[str, object], actual: str) -> str:
    """按 sealed v5 denominator 合同评分 v6 输出。"""
    if actual == observation["output_text"]:
        return "EXACT"
    if actual == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _program(
        *,
        pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """从 v6 pack/projection outputs 编译 disabled whole program。"""
    return compile_normalization_recovery_v6_phrase_program(
        rule_pack_manifest_sha256=pack_manifest_sha256,
        target_whole_rules=outputs["target-whole-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        identity_vetoes=outputs["identity-vetoes.jsonl"],
        conflict_vetoes=outputs["conflict-vetoes.jsonl"],
        target_index=outputs["target-index.jsonl"],
    )


def _runtime_case(
        *,
        pack_sha: str,
        case_kind: str,
        case_count: int,
        expected_count: int,
        mismatch_count: int,
        rows: list[dict[str, object]],
        ) -> dict[str, object]:
    """冻结一个 full-pack facility aggregate。"""
    identity = {
        "case_kind": case_kind,
        "pack_manifest_sha256": pack_sha,
    }
    return {
        **identity,
        "case_count": case_count,
        "case_id": _record_id(identity),
        "expected_count": expected_count,
        "format_version": 1,
        "mismatch_count": mismatch_count,
        "record_kind": NORMALIZATION_RECOVERY_V6_RUNTIME_AUDIT_KIND,
        "rows_sha256": _sha256(canonical_json_bytes(rows)),
    }


def _execution_rows(
        *,
        program: dict[str, object],
        values: tuple[dict[str, object], ...],
        input_key: str,
        output_key: str,
        structure_builder,
        identity_key: str,
        ) -> tuple[list[dict[str, object]], int]:
    """执行一组 exact/identity cases 并比较 indexed/reference/expected。"""
    texts = tuple(str(item[input_key]) for item in values)
    structures = tuple(structure_builder(item) for item in values)
    indexed = execute_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    reference = reference_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    rows = []
    mismatch = 0
    for source, item, left, right in zip(texts, values, indexed, reference):
        expected = str(item[output_key])
        failed = int(left != right or left["output_text"] != expected)
        mismatch += failed
        rows.append({
            "actual_output_sha256": _sha256(
                str(left["output_text"]).encode("utf-8")),
            "expected_output_sha256": _sha256(expected.encode("utf-8")),
            "input_sha256": _sha256(source.encode("utf-8")),
            "item_id": item[identity_key],
            "mismatch": failed,
            "result_sha256": left["result_sha256"],
        })
    rows.sort(key=lambda item: str(item["item_id"]))
    return rows, mismatch


def _rehash(program: dict[str, object]) -> None:
    """只重算篡改 program 的外层摘要。"""
    payload = {key: value for key, value in program.items()
               if key != "program_sha256"}
    program["program_sha256"] = _sha256(canonical_json_bytes(payload))


def _rejected(callable_value) -> int:
    """执行一个预期 fail-closed 的篡改路径。"""
    try:
        callable_value()
    except BroadQaExternalDataError:
        return 1
    return 0


def _runtime_cases(
        *,
        pack_sha: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """派生 approved/veto/interpreter/tamper 五类 full-pack facility。"""
    program = _program(pack_manifest_sha256=pack_sha, outputs=outputs)
    cases = []
    rules = outputs["target-whole-rules.jsonl"]
    rows, mismatch = _execution_rows(
        program=program,
        values=rules,
        input_key="input_text",
        output_key="output_text",
        structure_builder=lambda item: tuple(
            item["structure_token_variants"][0]),
        identity_key="rule_id",
    )
    cases.append(_runtime_case(
        pack_sha=pack_sha,
        case_kind="APPROVED_RULE_EXACT",
        case_count=len(rules),
        expected_count=len(rules),
        mismatch_count=mismatch,
        rows=rows,
    ))
    for name, kind in (
            ("identity-vetoes.jsonl", "IDENTITY_VETO_BACKOFF"),
            ("conflict-vetoes.jsonl", "CONFLICT_VETO_BACKOFF")):
        values = outputs[name]
        rows, mismatch = _execution_rows(
            program=program,
            values=values,
            input_key="input_text",
            output_key="input_text",
            structure_builder=lambda _item: (),
            identity_key="veto_id",
        )
        cases.append(_runtime_case(
            pack_sha=pack_sha,
            case_kind=kind,
            case_count=len(values),
            expected_count=len(values),
            mismatch_count=mismatch,
            rows=rows,
        ))
    texts = tuple(str(item["input_text"]) for item in observations)
    structures = tuple(tuple(item["structure_tokens"]) for item in observations)
    indexed = execute_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    reference = reference_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    interpreter_rows = [{
        "indexed_result_sha256": left["result_sha256"],
        "observation_id": observation["observation_id"],
        "reference_result_sha256": right["result_sha256"],
    } for observation, left, right in zip(observations, indexed, reference)]
    interpreter_rows.sort(key=lambda item: str(item["observation_id"]))
    cases.append(_runtime_case(
        pack_sha=pack_sha,
        case_kind="FULL_TRAIN_INDEXED_REFERENCE",
        case_count=len(observations),
        expected_count=len(observations),
        mismatch_count=sum(left != right for left, right in zip(indexed, reference)),
        rows=interpreter_rows,
    ))
    tamper_rows = []
    outer = {**program, "whole_input_exact_only": 0}
    tamper_rows.append({"kind": "OUTER_SHA", "rejected": _rejected(
        lambda: execute_normalization_recovery_v6_phrase_batch(
            outer, ("probe",)))})
    if rules:
        rule_tampered = copy.deepcopy(program)
        rule_tampered["target_buckets"][0]["rules"][0]["output_text"] += "X"
        _rehash(rule_tampered)
        tamper_rows.append({"kind": "RULE_REHASH", "rejected": _rejected(
            lambda: execute_normalization_recovery_v6_phrase_batch(
                rule_tampered, (str(rules[0]["input_text"]),)))})
    if outputs["identity-vetoes.jsonl"]:
        veto_tampered = copy.deepcopy(program)
        veto_tampered["identity_vetoes"][0]["input_text"] += "X"
        _rehash(veto_tampered)
        tamper_rows.append({"kind": "VETO_REHASH", "rejected": _rejected(
            lambda: execute_normalization_recovery_v6_phrase_batch(
                veto_tampered, ("probe",)))})
    bad_index = list(outputs["target-index.jsonl"])
    if bad_index:
        bad_index[0] = {**bad_index[0], "candidate_input": "BROKEN"}
        tamper_rows.append({"kind": "INDEX_REHASH", "rejected": _rejected(
            lambda: compile_normalization_recovery_v6_phrase_program(
                rule_pack_manifest_sha256=pack_sha,
                target_whole_rules=outputs["target-whole-rules.jsonl"],
                defeaters=outputs["defeaters.jsonl"],
                identity_vetoes=outputs["identity-vetoes.jsonl"],
                conflict_vetoes=outputs["conflict-vetoes.jsonl"],
                target_index=tuple(bad_index)))})
    cases.append(_runtime_case(
        pack_sha=pack_sha,
        case_kind="REHASHED_TAMPER_GUARDS",
        case_count=len(tamper_rows),
        expected_count=len(tamper_rows),
        mismatch_count=sum(item["rejected"] != 1 for item in tamper_rows),
        rows=tamper_rows,
    ))
    return tuple(sorted(cases, key=lambda item: str(item["case_id"])))


def _contracts(
        *,
        protocol_sha: str,
        pack_sha: str,
        pack_manifest: dict[str, object],
        audit_sha: str,
        audit_manifest: dict[str, object],
        simulation_sha: str,
        simulation_manifest: dict[str, object],
        simulation_families: tuple[dict[str, object], ...],
        simulation_strategies: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], dict[str, dict[str, object]],
                   dict[str, object]]:
    """核验 full pack、sealed denominator 与 WHOLE_STRONG 参照。"""
    audit_summary = audit_manifest.get("summary")
    if (pack_manifest.get("artifact_kind") != NORMALIZATION_RECOVERY_V6_RULE_PACK_KIND
            or pack_manifest.get("status") != NORMALIZATION_RECOVERY_V6_RULE_PACK_STATUS
            or pack_manifest.get("manifest_sha256") != pack_sha
            or pack_manifest.get("protocol_manifest_sha256") != protocol_sha
            or pack_manifest.get("production_enabled") != 0
            or pack_manifest.get("mastery_claimed") != 0
            or audit_manifest.get("pack_manifest_sha256")
            != pack_manifest.get("predecessor_rule_pack_manifest_sha256")
            or audit_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND
            or audit_manifest.get("status")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS
            or audit_manifest.get("protocol_manifest_sha256") != protocol_sha
            or not isinstance(audit_summary, dict)
            or audit_summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_FAIL"
            or audit_summary.get("facility_failure_count") != 0
            or simulation_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_KIND
            or simulation_manifest.get("status")
            != NORMALIZATION_RECOVERY_V5_SUCCESSOR_SIMULATION_STATUS
            or simulation_manifest.get("manifest_sha256") != simulation_sha
            or simulation_manifest.get("protocol_manifest_sha256") != protocol_sha
            or simulation_manifest.get("training_audit_manifest_sha256")
            != audit_sha):
        raise BroadQaExternalDataError("v6 audit predecessor contract 漂移")
    family_by_source = {}
    for record in simulation_families:
        if (record.get("record_kind")
                != NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_KIND
                or record.get("strategy") != "WHOLE_STRONG"):
            continue
        family = str(record.get("held_out_source_family"))
        if family in family_by_source:
            raise BroadQaExternalDataError("v6 audit simulation family 重复")
        family_by_source[family] = record
    strategies = [item for item in simulation_strategies
                  if item.get("record_kind")
                  == NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_KIND
                  and item.get("strategy") == "WHOLE_STRONG"]
    if set(family_by_source) != set(V5_SOURCE_FAMILIES) or len(strategies) != 1:
        raise BroadQaExternalDataError("v6 audit WHOLE_STRONG reference 缺失")
    return audit_summary, family_by_source, strategies[0]


def _loso_record(
        *,
        protocol_sha: str,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        family: str,
        expected: dict[str, object],
        ) -> dict[str, object]:
    """重学一个三来源子集、投影 v6 并核对固定 family 参照。"""
    material = derive_normalization_recovery_v5_loso_execution(
        protocol_manifest_sha256=protocol_sha,
        observations=observations,
        fragments=fragments,
        held_out_source_family=family,
        include_reference=False,
    )
    outputs, projection_summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=protocol_sha,
        predecessor_pack_manifest_sha256=str(
            material["subset_pack_manifest_sha256"]),
        predecessor_outputs=material["outputs"],
    )
    subset_v6_pack_sha = _sha256(canonical_json_bytes({
        "projection_summary": projection_summary,
        "subset_predecessor_pack_manifest_sha256": material[
            "subset_pack_manifest_sha256"],
    }))
    program = _program(
        pack_manifest_sha256=subset_v6_pack_sha,
        outputs=outputs,
    )
    held_out = material["held_out_observations"]
    texts = tuple(str(item["input_text"]) for item in held_out)
    structures = tuple(tuple(item["structure_tokens"]) for item in held_out)
    indexed = execute_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    reference = reference_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures)
    outcomes = Counter()
    buckets = Counter()
    identity_false_change = 0
    rows = []
    for observation, baseline, left, right in zip(
            held_out, material["indexed_results"], indexed, reference):
        outcome = _outcome(observation, str(left["output_text"]))
        bucket = normalization_recovery_v5_result_bucket(observation, baseline)
        outcomes[outcome] += 1
        buckets[(bucket, outcome)] += 1
        identity_false_change += int(
            observation["identity_preservation"] == 1
            and left["output_text"] != observation["input_text"])
        rows.append({
            "actual_output_sha256": _sha256(
                str(left["output_text"]).encode("utf-8")),
            "denominator_bucket": bucket,
            "held_out_observation_id": observation["observation_id"],
            "indexed_result_sha256": left["result_sha256"],
            "outcome": outcome,
            "reference_result_sha256": right["result_sha256"],
        })
    outcome_counts = {key: outcomes[key] for key in LOSO_OUTCOMES}
    bucket_counts = {
        f"{bucket}:{outcome}": buckets[(bucket, outcome)]
        for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES}
    reference_equal = int(
        expected.get("case_count") == len(held_out)
        and expected.get("outcome_counts") == outcome_counts
        and expected.get("bucket_outcome_counts") == bucket_counts
        and expected.get("identity_false_change_count")
        == identity_false_change
        and expected.get("subset_protocol_manifest_sha256")
        == material["subset_protocol_manifest_sha256"])
    identity = {
        "held_out_source_family": family,
        "protocol_manifest_sha256": protocol_sha,
        "subset_protocol_manifest_sha256": material[
            "subset_protocol_manifest_sha256"],
    }
    rows.sort(key=lambda item: str(item["held_out_observation_id"]))
    return {
        **identity,
        "bucket_outcome_counts": bucket_counts,
        "format_version": 1,
        "held_out_observation_count": len(held_out),
        "identity_false_change_count": identity_false_change,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "loso_id": _record_id(identity),
        "outcome_counts": outcome_counts,
        "projection_summary": projection_summary,
        "record_kind": NORMALIZATION_RECOVERY_V6_LOSO_AUDIT_KIND,
        "result_rows_sha256": _sha256(canonical_json_bytes(rows)),
        "simulation_family_reference_equal": reference_equal,
        "subset_predecessor_pack_manifest_sha256": material[
            "subset_pack_manifest_sha256"],
        "subset_v6_pack_manifest_sha256": subset_v6_pack_sha,
        "training_observation_count": len(material["training_observations"]),
        "training_source_families": sorted({
            str(item["source_family"])
            for item in material["training_observations"]}),
    }


def derive_normalization_recovery_v6_training_audit(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        pack_manifest: dict[str, object],
        pack_outputs: dict[str, tuple[dict[str, object], ...]],
        audit_manifest_sha256: str,
        audit_manifest: dict[str, object],
        simulation_manifest_sha256: str,
        simulation_manifest: dict[str, object],
        simulation_family_records: tuple[dict[str, object], ...],
        simulation_strategy_records: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], dict[str, object]]:
    """派生 full-pack facility、四方向 v6 LOSO 与 capability 摘要。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"), label="v6 audit protocol")
    pack_sha = _sha_value(
        pack_manifest.get("manifest_sha256"), label="v6 audit pack")
    audit_sha = _sha_value(
        audit_manifest_sha256, label="v6 audit v5 denominator")
    simulation_sha = _sha_value(
        simulation_manifest_sha256, label="v6 audit simulation")
    _audit_summary, expected_families, expected_strategy = _contracts(
        protocol_sha=protocol_sha,
        pack_sha=pack_sha,
        pack_manifest=pack_manifest,
        audit_sha=audit_sha,
        audit_manifest=audit_manifest,
        simulation_sha=simulation_sha,
        simulation_manifest=simulation_manifest,
        simulation_families=simulation_family_records,
        simulation_strategies=simulation_strategy_records,
    )
    runtime = _runtime_cases(
        pack_sha=pack_sha, outputs=pack_outputs, observations=observations)
    loso = tuple(sorted((
        _loso_record(
            protocol_sha=protocol_sha,
            observations=observations,
            fragments=fragments,
            family=family,
            expected=expected_families[family],
        ) for family in V5_SOURCE_FAMILIES),
        key=lambda item: str(item["loso_id"])))
    outcomes = Counter()
    buckets = Counter()
    for record in loso:
        outcomes.update(record["outcome_counts"])
        buckets.update(record["bucket_outcome_counts"])
    identity_false_change = sum(
        int(item["identity_false_change_count"]) for item in loso)
    non_identity_exact = sum(
        buckets[f"{bucket}:EXACT"] for bucket in AUDIT_BUCKETS
        if bucket != "IDENTITY")
    aggregate_equal = int(
        expected_strategy.get("outcome_counts")
        == {key: outcomes[key] for key in LOSO_OUTCOMES}
        and expected_strategy.get("bucket_outcome_counts")
        == {key: buckets[key] for key in sorted(buckets)}
        and expected_strategy.get("identity_false_change_count")
        == identity_false_change
        and expected_strategy.get("non_identity_exact_count")
        == non_identity_exact)
    facility_failure_count = (
        sum(int(item["mismatch_count"]) for item in runtime)
        + sum(int(item["indexed_reference_mismatch_count"]) for item in loso)
        + sum(item["simulation_family_reference_equal"] != 1 for item in loso)
        + (aggregate_equal != 1))
    capability_pass = int(
        facility_failure_count == 0
        and outcomes["WRONG"] == 0
        and identity_false_change == 0
        and non_identity_exact > 0)
    summary = {
        "audit_outcome": (
            "FACILITY_FAIL" if facility_failure_count
            else "FACILITY_PASS_CAPABILITY_PASS" if capability_pass
            else "FACILITY_PASS_CAPABILITY_FAIL"),
        "bucket_outcome_counts": {
            key: buckets[key] for key in sorted(buckets)},
        "capability_gate_pass": capability_pass,
        "facility_failure_count": facility_failure_count,
        "identity_false_change_count": identity_false_change,
        "loso_family_count": len(loso),
        "non_identity_exact_count": non_identity_exact,
        "outcome_counts": {key: outcomes[key] for key in LOSO_OUTCOMES},
        "runtime_case_count": len(runtime),
        "simulation_strategy_reference_equal": aggregate_equal,
        "simulation_strategy_result_id": expected_strategy[
            "strategy_result_id"],
    }
    return runtime, loso, summary


__all__ = [
    "NORMALIZATION_RECOVERY_V6_LOSO_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V6_RUNTIME_AUDIT_KIND",
    "derive_normalization_recovery_v6_training_audit",
]
