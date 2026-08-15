"""编译、执行并预检 recovery-v6 whole-only candidate。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND,
    NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS,
    execute_normalization_recovery_v6_phrase_batch,
    reference_normalization_recovery_v6_phrase_batch,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V6_CANDIDATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_CANDIDATE_V1")
NORMALIZATION_RECOVERY_V6_CANDIDATE_STATUS = (
    "FROZEN_WHOLE_ONLY_PRODUCTION_DISABLED_FORMAL_NOT_RUN")

_CANDIDATE_KEYS = {
    "applicability_contract",
    "artifact_kind",
    "candidate_program_sha256",
    "evaluation_commitment_manifest_sha256",
    "format_version",
    "local_execution_allowed",
    "mastery_claimed",
    "phrase_program",
    "phrase_program_sha256",
    "production_enabled",
    "source_execution_allowed",
    "status",
    "target_policy_scope",
    "v6_rule_pack_manifest_sha256",
    "v6_training_audit_manifest_sha256",
    "whole_input_exact_only",
}


def _sha256(payload: bytes) -> str:
    """返回 candidate、query 或结果摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _candidate_payload(program: dict[str, object]) -> dict[str, object]:
    """移除 candidate 自摘要字段后返回完整 identity payload。"""
    return {key: value for key, value in program.items()
            if key != "candidate_program_sha256"}


def _probe(program: dict[str, object]) -> tuple[str, tuple[str, ...]]:
    """从 candidate 内部记录选择一个只用于 schema 核验的输入。"""
    phrase = program.get("phrase_program")
    if not isinstance(phrase, dict):
        raise BroadQaExternalDataError("v6 candidate phrase program 缺失")
    buckets = phrase.get("target_buckets")
    if isinstance(buckets, list):
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            rules = bucket.get("rules")
            if not isinstance(rules, list) or not rules:
                continue
            rule = rules[0]
            if not isinstance(rule, dict):
                continue
            variants = rule.get("structure_token_variants")
            tokens = ()
            if isinstance(variants, list) and variants:
                tokens = tuple(str(item) for item in variants[0])
            return str(rule.get("input_text")), tokens
    for name in ("identity_vetoes", "conflict_vetoes"):
        records = phrase.get(name)
        if isinstance(records, list) and records and isinstance(records[0], dict):
            return str(records[0].get("input_text")), ()
    raise BroadQaExternalDataError("v6 candidate 无可核验运行记录")


def _validate_candidate(program: dict[str, object]) -> dict[str, object]:
    """核验 candidate 外层身份、固定 scope 与内嵌 phrase program。"""
    if (not isinstance(program, dict) or set(program) != _CANDIDATE_KEYS
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V6_CANDIDATE_KIND
            or program.get("status")
            != NORMALIZATION_RECOVERY_V6_CANDIDATE_STATUS
            or program.get("format_version") != 1
            or program.get("production_enabled") != 0
            or program.get("mastery_claimed") != 0
            or program.get("local_execution_allowed") != 0
            or program.get("source_execution_allowed") != 0
            or program.get("whole_input_exact_only") != 1
            or program.get("target_policy_scope")
            != RECOVERY_V6_TARGET_POLICY_SCOPE
            or program.get("candidate_program_sha256") != _sha256(
                canonical_json_bytes(_candidate_payload(program)))):
        raise BroadQaExternalDataError("v6 candidate identity/status 漂移")
    pack_sha = _sha_value(
        program.get("v6_rule_pack_manifest_sha256"),
        label="v6 candidate rule pack")
    _sha_value(
        program.get("v6_training_audit_manifest_sha256"),
        label="v6 candidate training audit")
    _sha_value(
        program.get("evaluation_commitment_manifest_sha256"),
        label="v6 candidate evaluation commitment")
    phrase = program.get("phrase_program")
    applicability = program.get("applicability_contract")
    if (not isinstance(phrase, dict)
            or phrase.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND
            or phrase.get("status")
            != NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS
            or phrase.get("rule_pack_manifest_sha256") != pack_sha
            or phrase.get("program_sha256")
            != program.get("phrase_program_sha256")
            or applicability != {
                "invalid_scope_applicable": 0,
                "invalid_scope_returns_identity": 1,
                "valid_scope_applicable_for_full_denominator": 1,
                "valid_scope_identity_backoff_is_not_not_applicable": 1,
            }):
        raise BroadQaExternalDataError("v6 candidate phrase/scope 合同漂移")
    text, tokens = _probe(program)
    reference_normalization_recovery_v6_phrase_batch(
        phrase, (text,), structure_tokens=(tokens,))
    return phrase


def compile_normalization_recovery_v6_candidate(
        *,
        phrase_program: dict[str, object],
        v6_training_audit_manifest_sha256: str,
        evaluation_commitment_manifest_sha256: str,
        ) -> dict[str, object]:
    """把 v6 phrase program 封装为全分母适用的禁用态 candidate。"""
    pack_sha = _sha_value(
        phrase_program.get("rule_pack_manifest_sha256")
        if isinstance(phrase_program, dict) else None,
        label="v6 candidate pack manifest")
    audit_sha = _sha_value(
        v6_training_audit_manifest_sha256,
        label="v6 candidate audit manifest")
    commitment_sha = _sha_value(
        evaluation_commitment_manifest_sha256,
        label="v6 candidate commitment manifest")
    program = {
        "applicability_contract": {
            "invalid_scope_applicable": 0,
            "invalid_scope_returns_identity": 1,
            "valid_scope_applicable_for_full_denominator": 1,
            "valid_scope_identity_backoff_is_not_not_applicable": 1,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V6_CANDIDATE_KIND,
        "evaluation_commitment_manifest_sha256": commitment_sha,
        "format_version": 1,
        "local_execution_allowed": 0,
        "mastery_claimed": 0,
        "phrase_program": phrase_program,
        "phrase_program_sha256": phrase_program.get("program_sha256"),
        "production_enabled": 0,
        "source_execution_allowed": 0,
        "status": NORMALIZATION_RECOVERY_V6_CANDIDATE_STATUS,
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
        "v6_rule_pack_manifest_sha256": pack_sha,
        "v6_training_audit_manifest_sha256": audit_sha,
        "whole_input_exact_only": 1,
    }
    candidate = {
        **program,
        "candidate_program_sha256": _sha256(canonical_json_bytes(program)),
    }
    _validate_candidate(candidate)
    return candidate


def _result(
        *,
        program: dict[str, object],
        text: str,
        structure_tokens: tuple[str, ...],
        policy_scope: str,
        phrase_result: dict[str, object] | None,
        ) -> dict[str, object]:
    """封装一次 candidate 输出并保留 applicability 与 phrase trace。"""
    mismatch = int(policy_scope != RECOVERY_V6_TARGET_POLICY_SCOPE)
    output = text if phrase_result is None else str(phrase_result["output_text"])
    payload = {
        "applicable": 1 - mismatch,
        "candidate_program_sha256": program["candidate_program_sha256"],
        "decision_reasons": (
            ["SCOPE_MISMATCH", "IDENTITY_BACKOFF"]
            if phrase_result is None
            else phrase_result["decision_reasons"]),
        "input_text": text,
        "mastery_claimed": 0,
        "output_text": output,
        "phrase_result_sha256": (
            "" if phrase_result is None else phrase_result["result_sha256"]),
        "production_enabled": 0,
        "requested_policy_scope": policy_scope,
        "scope_mismatch": mismatch,
        "steps": ([] if phrase_result is None else phrase_result["steps"]),
        "structure_tokens": list(structure_tokens),
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    return {**payload, "result_sha256": _sha256(
        canonical_json_bytes(payload))}


def _execute_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        structure_tokens: tuple[tuple[str, ...], ...] | None,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """核验 candidate 后批量执行合法 scope 或一致拒绝错误 scope。"""
    phrase = _validate_candidate(program)
    tokens = (((),) * len(texts)
              if structure_tokens is None else structure_tokens)
    if (not isinstance(texts, tuple) or not texts
            or any(not isinstance(item, str) or not item for item in texts)
            or not isinstance(policy_scope, str)
            or not isinstance(tokens, tuple) or len(tokens) != len(texts)
            or any(not isinstance(row, tuple)
                   or any(not isinstance(item, str) for item in row)
                   for row in tokens)):
        raise BroadQaExternalDataError("v6 candidate batch 输入非法")
    if policy_scope != RECOVERY_V6_TARGET_POLICY_SCOPE:
        return tuple(_result(
            program=program,
            text=text,
            structure_tokens=row,
            policy_scope=policy_scope,
            phrase_result=None,
        ) for text, row in zip(texts, tokens))
    executor = (execute_normalization_recovery_v6_phrase_batch
                if indexed else reference_normalization_recovery_v6_phrase_batch)
    phrase_results = executor(
        phrase, texts, structure_tokens=tokens)
    return tuple(_result(
        program=program,
        text=text,
        structure_tokens=row,
        policy_scope=policy_scope,
        phrase_result=phrase_result,
    ) for text, row, phrase_result in zip(texts, tokens, phrase_results))


def execute_normalization_recovery_v6_candidate_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 indexed v6 candidate。"""
    return _execute_batch(
        program,
        texts,
        policy_scope=policy_scope,
        structure_tokens=structure_tokens,
        indexed=True,
    )


def reference_normalization_recovery_v6_candidate_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 linear-reference v6 candidate。"""
    return _execute_batch(
        program,
        texts,
        policy_scope=policy_scope,
        structure_tokens=structure_tokens,
        indexed=False,
    )


def derive_normalization_recovery_v6_candidate_preflight(
        program: dict[str, object],
        ) -> dict[str, object]:
    """用 candidate 自身 rule/veto roster 派生标签盲确定性 preflight。"""
    phrase = _validate_candidate(program)
    cases = []
    for bucket in phrase["target_buckets"]:
        for rule in bucket["rules"]:
            variants = rule["structure_token_variants"]
            tokens = tuple(variants[0]) if variants else ()
            cases.append({
                "case_id": str(rule["rule_id"]),
                "case_kind": "APPROVED_RULE_EXACT",
                "expected_output": str(rule["output_text"]),
                "input_text": str(rule["input_text"]),
                "structure_tokens": tokens,
            })
    for name, kind in (
            ("identity_vetoes", "IDENTITY_VETO_BACKOFF"),
            ("conflict_vetoes", "CONFLICT_VETO_BACKOFF")):
        for record in phrase[name]:
            cases.append({
                "case_id": str(record["veto_id"]),
                "case_kind": kind,
                "expected_output": str(record["input_text"]),
                "input_text": str(record["input_text"]),
                "structure_tokens": (),
            })
    cases.sort(key=lambda item: (
        str(item["case_kind"]), str(item["case_id"])))
    if not cases:
        raise BroadQaExternalDataError("v6 candidate preflight roster 为空")
    texts = tuple(str(item["input_text"]) for item in cases)
    structures = tuple(item["structure_tokens"] for item in cases)
    indexed = execute_normalization_recovery_v6_candidate_batch(
        program,
        texts,
        policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
        structure_tokens=structures,
    )
    reference = reference_normalization_recovery_v6_candidate_batch(
        program,
        texts,
        policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
        structure_tokens=structures,
    )
    rows = []
    failure_count = 0
    for case, left, right in zip(cases, indexed, reference):
        failed = int(
            left != right
            or left["output_text"] != case["expected_output"]
            or left["applicable"] != 1
            or left["scope_mismatch"] != 0
            or left["production_enabled"] != 0
            or left["mastery_claimed"] != 0)
        failure_count += failed
        rows.append({
            "case_id": case["case_id"],
            "case_kind": case["case_kind"],
            "expected_output_sha256": _sha256(
                str(case["expected_output"]).encode("utf-8")),
            "failed": failed,
            "indexed_result_sha256": left["result_sha256"],
            "reference_result_sha256": right["result_sha256"],
        })
    negative_indexed = execute_normalization_recovery_v6_candidate_batch(
        program,
        (texts[0],),
        policy_scope="",
        structure_tokens=(structures[0],),
    )[0]
    negative_reference = reference_normalization_recovery_v6_candidate_batch(
        program,
        (texts[0],),
        policy_scope="",
        structure_tokens=(structures[0],),
    )[0]
    negative_failed = int(
        negative_indexed != negative_reference
        or negative_indexed["output_text"] != texts[0]
        or negative_indexed["applicable"] != 0
        or negative_indexed["scope_mismatch"] != 1)
    failure_count += negative_failed
    rows.append({
        "case_id": _sha256(b"INVALID_SCOPE"),
        "case_kind": "INVALID_SCOPE_REJECTED",
        "expected_output_sha256": _sha256(texts[0].encode("utf-8")),
        "failed": negative_failed,
        "indexed_result_sha256": negative_indexed["result_sha256"],
        "reference_result_sha256": negative_reference["result_sha256"],
    })
    counts = Counter(str(item["case_kind"]) for item in cases)
    counts["INVALID_SCOPE_REJECTED"] += 1
    return {
        "candidate_program_sha256": program["candidate_program_sha256"],
        "case_count": len(cases) + 1,
        "case_kind_counts": dict(sorted(counts.items())),
        "evaluation_payload_read_count": 0,
        "failure_count": failure_count,
        "formal_run_count": 0,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference))
            + int(negative_indexed != negative_reference),
        "invalid_scope_rejected": 1 - negative_failed,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "result_rows_sha256": _sha256(canonical_json_bytes(rows)),
        "valid_scope_all_applicable": int(
            all(item["applicable"] == 1 for item in indexed)),
    }


__all__ = [
    "NORMALIZATION_RECOVERY_V6_CANDIDATE_KIND",
    "NORMALIZATION_RECOVERY_V6_CANDIDATE_STATUS",
    "compile_normalization_recovery_v6_candidate",
    "derive_normalization_recovery_v6_candidate_preflight",
    "execute_normalization_recovery_v6_candidate_batch",
    "reference_normalization_recovery_v6_candidate_batch",
]
