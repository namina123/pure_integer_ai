"""发布 recovery candidate 的 TRAIN-only indexed/reference 性能证据。"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import time

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_clone import (
    NormalizationRecoveryCandidateProgram,
    compile_normalization_recovery_candidate,
    execute_normalization_recovery_candidate,
    reference_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_rule_pack import (
    read_normalization_recovery_rule_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.train_execution import (
    process_working_set_bytes,
)


NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_CANDIDATE_TRAIN_PROFILE_V1")
NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_STATUS = (
    "TRAIN_ONLY_INDEXED_REFERENCE_EQUIVALENT_NOT_EVALUATED")


def _sha256(payload: bytes) -> str:
    """返回 query roster、代码或 profile 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[int], percent: int) -> int:
    """按 nearest-rank 计算严格整数纳秒分位数。"""
    if not values or percent not in {50, 95}:
        raise BroadQaExternalDataError(
            "recovery candidate profile percentile 输入非法")
    ordered = sorted(values)
    index = max(0, (len(ordered) * percent + 99) // 100 - 1)
    return ordered[index]


def _query(
        *,
        kind: str,
        input_text: str,
        policy_scope: str,
        regional_scope: str,
        expected_output: str,
        expected_trace_id: str,
        expected_trace_kind: str,
        ) -> dict[str, object]:
    """构造绑定 TRAIN identity 与预期 trace 的确定性 query。"""
    identity = {
        "expected_output": expected_output,
        "expected_trace_id": expected_trace_id,
        "expected_trace_kind": expected_trace_kind,
        "input_text": input_text,
        "kind": kind,
        "policy_scope": policy_scope,
        "regional_scope": regional_scope,
    }
    return {**identity, "query_id": _sha256(canonical_json_bytes(identity))}


def derive_normalization_recovery_training_queries(
        program: NormalizationRecoveryCandidateProgram,
        ) -> tuple[dict[str, object], ...]:
    """覆盖 authority、transfer、五 source policy 与 unscoped conflict。"""
    if not isinstance(program, NormalizationRecoveryCandidateProgram):
        raise BroadQaExternalDataError(
            "recovery candidate profile program 非法")
    profile = program.transfer_profile
    values = []
    for rule in program.generic_rules:
        values.append(_query(
            kind="AUTHORITY_GENERIC",
            input_text=rule.input_text,
            policy_scope=profile.authority_policy_scope,
            regional_scope="",
            expected_output=rule.output_text,
            expected_trace_id=rule.rule_id,
            expected_trace_kind="TARGET_RULE",
        ))
    for rule in program.regional_rules:
        values.append(_query(
            kind="AUTHORITY_REGIONAL",
            input_text=rule.input_text,
            policy_scope=profile.authority_policy_scope,
            regional_scope=profile.regional_scope,
            expected_output=rule.output_text,
            expected_trace_id=rule.rule_id,
            expected_trace_kind="TARGET_RULE",
        ))
    generic_by_input = {item.input_text: item for item in program.generic_rules}
    regional_by_input = {
        item.input_text: item for item in program.regional_rules}
    for input_text in sorted(set(generic_by_input) | set(regional_by_input)):
        rule = regional_by_input.get(input_text, generic_by_input.get(input_text))
        values.append(_query(
            kind="TRANSFER_TARGET",
            input_text=input_text,
            policy_scope=profile.candidate_target_policy_scope,
            regional_scope=profile.regional_scope,
            expected_output=rule.output_text,
            expected_trace_id=rule.rule_id,
            expected_trace_kind="TARGET_RULE",
        ))
    for replay in program.source_replays:
        values.append(_query(
            kind="SOURCE_REPLAY",
            input_text=replay.input_text,
            policy_scope=replay.source_policy_scope,
            regional_scope="",
            expected_output=replay.output_text,
            expected_trace_id=replay.evidence_id,
            expected_trace_kind="SOURCE_EVIDENCE",
        ))
    for conflict in program.conflicts:
        values.append(_query(
            kind="UNSCOPED_CONFLICT",
            input_text=conflict.input_text,
            policy_scope="",
            regional_scope="",
            expected_output=conflict.input_text,
            expected_trace_id=conflict.conflict_id,
            expected_trace_kind="CONFLICT",
        ))
    result = tuple(sorted(values, key=lambda item: str(item["query_id"])))
    if (not result
            or len({item["query_id"] for item in result}) != len(result)):
        raise BroadQaExternalDataError(
            "recovery candidate profile query identity 漂移")
    return result


def _result_failure_count(
        *,
        query: dict[str, object],
        result,
        program: NormalizationRecoveryCandidateProgram,
        phrase_rule_by_key: dict[tuple[str, str], str],
        ) -> int:
    """核对 output、scope、trace、projection 与禁用态。"""
    failure = int(result.output_text != query["expected_output"])
    failure += int(result.production_enabled != 0)
    kind = query["kind"]
    trace_id = str(query["expected_trace_id"])
    trace_kind = query["expected_trace_kind"]
    if trace_kind == "TARGET_RULE":
        failure += int(trace_id not in result.target_rule_ids)
    elif trace_kind == "SOURCE_EVIDENCE":
        failure += int(trace_id not in result.source_evidence_ids)
        phrase_id = phrase_rule_by_key.get((
            str(query["policy_scope"]), str(query["input_text"])))
        if phrase_id is not None:
            failure += int(phrase_id not in result.phrase_rule_ids)
    else:
        failure += int(trace_id not in result.conflict_ids)
    if kind == "TRANSFER_TARGET":
        failure += int(result.projection_used != 1)
        failure += int(
            result.transfer_profile_id != program.transfer_profile.sha256())
        failure += int(result.scope_mismatch != 0)
    elif kind == "UNSCOPED_CONFLICT":
        failure += int(result.unscoped_conflict_blocked != 1)
        failure += int(result.scope_mismatch != 1)
    else:
        failure += int(result.projection_used != 0)
        failure += int(result.scope_mismatch != 0)
    return failure


def _run_executor(
        *,
        program: NormalizationRecoveryCandidateProgram,
        queries: tuple[dict[str, object], ...],
        executor,
        expected_result_shas: tuple[str, ...] | None,
        ) -> tuple[dict[str, object], tuple[str, ...]]:
    """执行完整 roster，形成性能、正确性和规范结果摘要。"""
    durations = []
    result_shas = []
    result_digest = hashlib.sha256()
    failure_count = 0
    mismatch_count = 0
    input_bytes = 0
    output_bytes = 0
    phrase_by_key = {
        (item.source_policy_scope, item.input_text): item.rule_id
        for item in program.phrase_overrides}
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for ordinal, query in enumerate(queries):
        started = time.perf_counter_ns()
        result = executor(
            program,
            str(query["input_text"]),
            policy_scope=str(query["policy_scope"]),
            regional_scope=str(query["regional_scope"]),
        )
        durations.append(time.perf_counter_ns() - started)
        failure_count += _result_failure_count(
            query=query,
            result=result,
            program=program,
            phrase_rule_by_key=phrase_by_key,
        )
        result_sha = result.sha256()
        result_shas.append(result_sha)
        result_digest.update(canonical_json_bytes({
            "query_id": query["query_id"],
            "result_sha256": result_sha,
        }))
        if (expected_result_shas is not None
                and result_sha != expected_result_shas[ordinal]):
            mismatch_count += 1
        input_bytes += len(str(query["input_text"]).encode("utf-8"))
        output_bytes += len(result.output_text.encode("utf-8"))
    metrics = {
        "cpu_ns": time.process_time_ns() - cpu_started,
        "failure_count": failure_count,
        "input_bytes": input_bytes,
        "mismatch_count": mismatch_count,
        "output_bytes": output_bytes,
        "p50_ns": _percentile(durations, 50),
        "p95_ns": _percentile(durations, 95),
        "query_count": len(queries),
        "result_sha256": result_digest.hexdigest(),
        "wall_ns": time.perf_counter_ns() - wall_started,
    }
    return metrics, tuple(result_shas)


def profile_normalization_recovery_candidate(
        program: NormalizationRecoveryCandidateProgram,
        ) -> dict[str, object]:
    """在同一冻结 TRAIN roster 上比较 indexed 与线性 reference。"""
    queries = derive_normalization_recovery_training_queries(program)
    indexed, indexed_shas = _run_executor(
        program=program,
        queries=queries,
        executor=execute_normalization_recovery_candidate,
        expected_result_shas=None,
    )
    reference, _reference_shas = _run_executor(
        program=program,
        queries=queries,
        executor=reference_normalization_recovery_candidate,
        expected_result_shas=indexed_shas,
    )
    query_kinds = Counter(str(item["kind"]) for item in queries)
    query_payload = b"".join(canonical_json_line(item) for item in queries)
    return {
        "indexed": indexed,
        "indexed_reference_result_bytes_equal": int(
            indexed["result_sha256"] == reference["result_sha256"]),
        "peak_working_set_bytes": process_working_set_bytes(),
        "query_count": len(queries),
        "query_kind_counts": dict(sorted(query_kinds.items())),
        "query_roster_bytes": len(query_payload),
        "query_roster_sha256": _sha256(query_payload),
        "reference": reference,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求 profile run root 是显式已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "recovery candidate profile run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析 profile 输入输出并拒绝逃出 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def normalization_recovery_candidate_code_files() -> list[dict[str, object]]:
    """承诺 candidate/profile 七个公开源码文件的当前字节。"""
    directory = Path(__file__).resolve().parent
    names = (
        "ph2_broad_qa_normalization_recovery_candidate_clone.py",
        "ph2_broad_qa_normalization_recovery_candidate_compile.py",
        "ph2_broad_qa_normalization_recovery_candidate_execution.py",
        "ph2_broad_qa_normalization_recovery_candidate_profile.py",
        "ph2_broad_qa_normalization_recovery_candidate_profile_reader.py",
        "ph2_broad_qa_normalization_recovery_candidate_records.py",
        "train_execution.py",
    )
    values = []
    for name in names:
        payload = (directory / name).read_bytes()
        values.append({
            "bytes": len(payload),
            "relative_path": f"src/pure_integer_ai/experiments/{name}",
            "sha256": _sha256(payload),
        })
    return values


def publish_normalization_recovery_candidate_profile(
        *,
        run_root: str | Path,
        evaluation_protocol_dir: str | Path,
        expected_evaluation_manifest_sha256: str,
        training_protocol_dir: str | Path,
        expected_training_manifest_sha256: str,
        rule_pack_dir: str | Path,
        expected_rule_pack_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """只读 manifest/TRAIN pack，不可覆盖发布 TRAIN-only profile。"""
    root = _require_k_root(run_root)
    evaluation_root = _within(
        root, evaluation_protocol_dir, label="evaluation_protocol_dir")
    training_root = _within(
        root, training_protocol_dir, label="training_protocol_dir")
    pack_root = _within(root, rule_pack_dir, label="rule_pack_dir")
    target = _within(root, target_dir, label="target_dir")
    if (target.exists() or not evaluation_root.is_dir()
            or not training_root.is_dir() or not pack_root.is_dir()):
        raise BroadQaExternalDataError(
            "recovery candidate profile 输入缺失或 target 已存在")
    evaluation = read_normalization_recovery_evaluation_manifest_only(
        evaluation_root,
        expected_manifest_sha256=expected_evaluation_manifest_sha256,
    )
    pack, outputs = read_normalization_recovery_rule_pack(
        pack_root,
        protocol_dir=training_root,
        expected_protocol_manifest_sha256=expected_training_manifest_sha256,
        expected_pack_manifest_sha256=expected_rule_pack_manifest_sha256,
    )
    compile_started = time.perf_counter_ns()
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    compile_ns = time.perf_counter_ns() - compile_started
    profile = profile_normalization_recovery_candidate(program)
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_KIND,
        "candidate_compile_wall_ns": compile_ns,
        "candidate_program_sha256": program.sha256(),
        "candidate_rule_pack_read_count": 1,
        "code_files": normalization_recovery_candidate_code_files(),
        "evaluation_manifest_read_count": 1,
        "evaluation_payload_read_count": 0,
        "evaluation_protocol_manifest_sha256": evaluation["manifest_sha256"],
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "profile": profile,
        "reserve_payload_read_count": 0,
        "rule_pack_manifest_sha256": pack["manifest_sha256"],
        "status": NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_protocol_manifest_sha256": (
            expected_training_manifest_sha256),
        "transfer_profile_sha256": program.transfer_profile.sha256(),
    }
    if (profile["indexed"]["failure_count"] != 0
            or profile["reference"]["failure_count"] != 0
            or profile["reference"]["mismatch_count"] != 0
            or profile["indexed_reference_result_bytes_equal"] != 1):
        raise BroadQaExternalDataError(
            "recovery candidate TRAIN-only profile 未闭合")
    target.mkdir(parents=True)
    profile_path = target / "profile.json"
    with profile_path.open("xb") as handle:
        handle.write(canonical_json_line(report))
    return {
        **report,
        "profile_sha256": _sha256(profile_path.read_bytes()),
    }


__all__ = [
    "NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_KIND",
    "NORMALIZATION_RECOVERY_CANDIDATE_PROFILE_STATUS",
    "derive_normalization_recovery_training_queries",
    "normalization_recovery_candidate_code_files",
    "profile_normalization_recovery_candidate",
    "publish_normalization_recovery_candidate_profile",
]
