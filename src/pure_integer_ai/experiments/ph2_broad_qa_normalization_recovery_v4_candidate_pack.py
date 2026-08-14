"""发布 recovery-v4 composite candidate 与 TRAIN-only 性能证据。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_compile import (
    compile_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_rule_pack import (
    read_normalization_recovery_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    read_normalization_recovery_v3_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_candidate import (
    NORMALIZATION_RECOVERY_V4_CANDIDATE_KIND,
    NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
    compile_normalization_recovery_v4_candidate,
    execute_normalization_recovery_v4_candidate_batch,
    reference_normalization_recovery_v4_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_rule_pack import (
    read_normalization_recovery_v4_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_audit import (
    NORMALIZATION_RECOVERY_V4_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V4_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.train_execution import (
    process_working_set_bytes,
)


NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_V1")
NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_STATUS = (
    "TRAIN_PROFILE_PASS_PRODUCTION_DISABLED_FORMAL_NOT_RUN")

_FILES = (
    ("candidate-program.json", "FROZEN_COMPOSITE_CANDIDATE"),
    ("profile.json", "TRAIN_ONLY_INDEXED_REFERENCE_PROFILE"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact、query roster 或结果流 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


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


def _require_k_root(value: str | Path) -> Path:
    """要求 candidate 工作根是显式存在的 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "recovery v4 candidate run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _artifact(path: Path, *, role: str) -> dict[str, object]:
    """返回一个规范 candidate artifact identity。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _read_audit_manifest(
        audit_dir: Path,
        *,
        expected_manifest_sha256: str,
        expected_protocol_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """只读 v4 audit manifest/file identity，不重复执行 TRAIN audit。"""
    expected_sha = _sha_value(
        expected_manifest_sha256, label="recovery v4 audit manifest")
    try:
        encoded = (audit_dir / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("recovery v4 audit manifest 不可读") from error
    if (_sha256(encoded) != expected_sha or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V4_TRAINING_AUDIT_KIND
            or manifest.get("status")
            != NORMALIZATION_RECOVERY_V4_TRAINING_AUDIT_STATUS
            or manifest.get("protocol_manifest_sha256")
            != expected_protocol_manifest_sha256
            or manifest.get("pack_manifest_sha256")
            != expected_pack_manifest_sha256
            or manifest.get("formal_run_count") != 0
            or manifest.get("production_enabled") != 0
            or manifest.get("mastery_claimed") != 0
            or manifest.get("reserve_payload_read_count") != 0):
        raise BroadQaExternalDataError("recovery v4 audit 冻结边界漂移")
    summary = manifest.get("summary")
    if (not isinstance(summary, dict)
            or summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_PASS"
            or summary.get("capability_gate_pass") != 1
            or summary.get("facility_failure_count") != 0
            or summary.get("defeater_mismatch_count") != 0):
        raise BroadQaExternalDataError("recovery v4 audit 未通过冻结门")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise BroadQaExternalDataError("recovery v4 audit file roster 漂移")
    for record in files:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError("recovery v4 audit file identity 非法")
        path = audit_dir / str(record.get("relative_path"))
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                "recovery v4 audit file 不可读") from error
        if (record.get("bytes") != len(payload)
                or record.get("sha256") != _sha256(payload)):
            raise BroadQaExternalDataError("recovery v4 audit file 漂移")
    return {**manifest, "manifest_sha256": expected_sha}


def _phrase_rules(
        program: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...],
                   dict[str, tuple[dict[str, object], ...]]]:
    """从 candidate 内嵌 phrase program 取出 target/source rule roster。"""
    phrase = program["phrase_program"]
    target = tuple(
        rule for bucket in phrase["target_buckets"]
        for rule in bucket["rules"])
    source = {}
    for item in phrase["source_programs"]:
        source[str(item["source_family"])] = tuple(
            rule for bucket in item["buckets"] for rule in bucket["rules"])
    return target, source


def derive_normalization_recovery_v4_candidate_queries(
        program: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """冻结 target、source isolation 与 unscoped conflict 的 TRAIN roster。"""
    target_rules, source_rules = _phrase_rules(program)
    values = []

    def add(kind: str, policy: str, region: str, text: str) -> None:
        identity = {
            "input_text": text,
            "kind": kind,
            "policy_scope": policy,
            "regional_scope": region,
        }
        values.append({
            **identity,
            "query_id": _sha256(canonical_json_bytes(identity)),
        })

    for record in program["base_character_rules"]:
        add("TRANSFER_CHARACTER", (
            NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE),
            "ZH_CN", str(record["input_text"]))
    for record in target_rules:
        text = str(record["input_text"])
        add("TRANSFER_TARGET_PHRASE", (
            NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE),
            "ZH_CN", text)
        add("AUTHORITY_TARGET_PHRASE", RECOVERY_V4_TARGET_POLICY_SCOPE,
            "ZH_CN", text)
    source_policy_by_family = {
        str(item["source_family"]): str(item["source_policy_scope"])
        for item in program["transfer_profile"]["source_policy_to_family"]
    }
    for family, records in sorted(source_rules.items()):
        policy = source_policy_by_family[family]
        for record in records:
            text = str(record["input_text"])
            add("SOURCE_SCOPED_PHRASE", policy, "", text)
            add("SOURCE_RULE_TARGET_ISOLATION", (
                NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE),
                "ZH_CN", text)
    for record in program["conflicts"]:
        add("UNSCOPED_CONFLICT", "", "", str(record["input_text"]))
    result = tuple(sorted(values, key=lambda item: str(item["query_id"])))
    if (not result or len({item["query_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError("recovery v4 candidate query identity 重复")
    return result


def _percentile(values: list[int], percent: int) -> int:
    """按 nearest-rank 返回非空整数耗时分位数。"""
    if not values or percent not in {50, 95}:
        raise BroadQaExternalDataError("recovery v4 profile percentile 非法")
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) * percent + 99) // 100 - 1)]


def _profile_executor(
        *,
        program: dict[str, object],
        queries: tuple[dict[str, object], ...],
        executor,
        expected: dict[str, str] | None,
        ) -> tuple[dict[str, object], dict[str, str]]:
    """按 scope 分批执行 roster，并形成摊销 query 性能和结果摘要。"""
    groups = {}
    for query in queries:
        key = (str(query["policy_scope"]), str(query["regional_scope"]))
        groups.setdefault(key, []).append(query)
    digest = hashlib.sha256()
    result_shas = {}
    amortized = []
    failure_count = 0
    mismatch_count = 0
    input_bytes = 0
    output_bytes = 0
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for (policy, region), group in sorted(groups.items()):
        texts = tuple(str(item["input_text"]) for item in group)
        started = time.perf_counter_ns()
        results = executor(
            program, texts, policy_scope=policy, regional_scope=region)
        elapsed = time.perf_counter_ns() - started
        per_query = max(1, elapsed // len(group))
        amortized.extend([per_query] * len(group))
        for query, result in zip(group, results):
            query_id = str(query["query_id"])
            result_sha = str(result["result_sha256"])
            result_shas[query_id] = result_sha
            digest.update(canonical_json_bytes({
                "query_id": query_id,
                "result_sha256": result_sha,
            }))
            if expected is not None and expected.get(query_id) != result_sha:
                mismatch_count += 1
            kind = query["kind"]
            failure_count += int(result["production_enabled"] != 0)
            failure_count += int(result["mastery_claimed"] != 0)
            if kind in {"TRANSFER_CHARACTER", "TRANSFER_TARGET_PHRASE",
                        "SOURCE_RULE_TARGET_ISOLATION"}:
                failure_count += int(result["projection_used"] != 1)
                failure_count += int(result["scope_mismatch"] != 0)
            elif kind == "UNSCOPED_CONFLICT":
                failure_count += int(result["applicable"] != 0)
                failure_count += int(
                    result["unscoped_conflict_blocked"] != 1)
            else:
                failure_count += int(result["scope_mismatch"] != 0)
            input_bytes += len(str(query["input_text"]).encode("utf-8"))
            output_bytes += len(str(result["output_text"]).encode("utf-8"))
    metrics = {
        "cpu_ns": time.process_time_ns() - cpu_started,
        "failure_count": failure_count,
        "input_bytes": input_bytes,
        "mismatch_count": mismatch_count,
        "output_bytes": output_bytes,
        "p50_ns": _percentile(amortized, 50),
        "p95_ns": _percentile(amortized, 95),
        "query_count": len(queries),
        "result_sha256": digest.hexdigest(),
        "wall_ns": time.perf_counter_ns() - wall_started,
    }
    return metrics, result_shas


def profile_normalization_recovery_v4_candidate(
        program: dict[str, object],
        ) -> dict[str, object]:
    """在固定 TRAIN roster 上比较 indexed/reference composite runtime。"""
    queries = derive_normalization_recovery_v4_candidate_queries(program)
    indexed, result_shas = _profile_executor(
        program=program,
        queries=queries,
        executor=execute_normalization_recovery_v4_candidate_batch,
        expected=None,
    )
    reference, _ = _profile_executor(
        program=program,
        queries=queries,
        executor=reference_normalization_recovery_v4_candidate_batch,
        expected=result_shas,
    )
    roster_payload = b"".join(canonical_json_line(item) for item in queries)
    return {
        "indexed": indexed,
        "indexed_reference_result_bytes_equal": int(
            indexed["result_sha256"] == reference["result_sha256"]),
        "peak_working_set_bytes": process_working_set_bytes(),
        "query_count": len(queries),
        "query_kind_counts": dict(sorted(Counter(
            str(item["kind"]) for item in queries).items())),
        "query_roster_bytes": len(roster_payload),
        "query_roster_sha256": _sha256(roster_payload),
        "reference": reference,
        "timing_contract": "FIXED_SCOPE_BATCH_AMORTIZED_PER_QUERY_NS",
    }


def _materialize_candidate(
        *,
        prior_evaluation_protocol_dir: Path,
        expected_prior_evaluation_manifest_sha256: str,
        base_training_protocol_dir: Path,
        expected_base_training_manifest_sha256: str,
        base_rule_pack_dir: Path,
        expected_base_rule_pack_manifest_sha256: str,
        v4_training_protocol_dir: Path,
        expected_v4_training_manifest_sha256: str,
        v4_rule_pack_dir: Path,
        expected_v4_rule_pack_manifest_sha256: str,
        v4_training_audit_dir: Path,
        expected_v4_training_audit_manifest_sha256: str,
        evaluation_commitment_dir: Path,
        expected_evaluation_commitment_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """严格回读 base/v4/commitment，并返回 composite program 与 audit。"""
    prior_manifest = read_normalization_recovery_evaluation_manifest_only(
        prior_evaluation_protocol_dir,
        expected_manifest_sha256=expected_prior_evaluation_manifest_sha256,
    )
    commitment = read_normalization_recovery_v3_evaluation_commitment(
        evaluation_commitment_dir,
        prior_evaluation_protocol_dir=prior_evaluation_protocol_dir,
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256),
    )
    if commitment["prior_evaluation_protocol_manifest_sha256"] != (
            prior_manifest["manifest_sha256"]):
        raise BroadQaExternalDataError(
            "recovery v4 commitment/prior evaluation 漂移")
    base_pack, base_outputs = read_normalization_recovery_rule_pack(
        base_rule_pack_dir,
        protocol_dir=base_training_protocol_dir,
        expected_protocol_manifest_sha256=(
            expected_base_training_manifest_sha256),
        expected_pack_manifest_sha256=(
            expected_base_rule_pack_manifest_sha256),
    )
    base_program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=prior_manifest,
        rule_pack_manifest=base_pack,
        outputs=base_outputs,
    )
    v4_pack, v4_outputs = read_normalization_recovery_v4_rule_pack(
        v4_rule_pack_dir,
        protocol_dir=v4_training_protocol_dir,
        expected_protocol_manifest_sha256=expected_v4_training_manifest_sha256,
        expected_pack_manifest_sha256=(
            expected_v4_rule_pack_manifest_sha256),
    )
    audit = _read_audit_manifest(
        v4_training_audit_dir,
        expected_manifest_sha256=(
            expected_v4_training_audit_manifest_sha256),
        expected_protocol_manifest_sha256=(
            expected_v4_training_manifest_sha256),
        expected_pack_manifest_sha256=(
            expected_v4_rule_pack_manifest_sha256),
    )
    program = compile_normalization_recovery_v4_candidate(
        base_program=base_program,
        base_rule_pack_manifest_sha256=base_pack["manifest_sha256"],
        v4_protocol_manifest_sha256=expected_v4_training_manifest_sha256,
        v4_rule_pack_manifest_sha256=v4_pack["manifest_sha256"],
        v4_training_audit_manifest_sha256=audit["manifest_sha256"],
        evaluation_commitment_manifest_sha256=commitment["manifest_sha256"],
        v4_outputs=v4_outputs,
    )
    return program, audit


def _paths(
        root: Path,
        arguments: dict[str, object],
        ) -> dict[str, Path]:
    """核验 candidate publisher 的全部物理输入目录。"""
    names = (
        "prior_evaluation_protocol_dir", "base_training_protocol_dir",
        "base_rule_pack_dir", "v4_training_protocol_dir",
        "v4_rule_pack_dir", "v4_training_audit_dir",
        "evaluation_commitment_dir",
    )
    result = {}
    for name in names:
        path = _within(root, arguments[name], label=name)
        if not path.is_dir():
            raise BroadQaExternalDataError(
                f"recovery v4 candidate 输入目录不存在: {name}")
        result[name] = path
    return result


def publish_normalization_recovery_v4_candidate_pack(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 composite candidate、profile 与 manifest-last artifact。"""
    root = _require_k_root(run_root)
    paths = _paths(root, arguments)
    target = _within(root, target_dir, label="recovery v4 candidate target")
    if target.exists():
        raise BroadQaExternalDataError("recovery v4 candidate target 已存在")
    program, audit = _materialize_candidate(**paths, **{
        key: arguments[key] for key in arguments if key not in paths})
    profile = profile_normalization_recovery_v4_candidate(program)
    if (profile["indexed"]["failure_count"] != 0
            or profile["reference"]["failure_count"] != 0
            or profile["reference"]["mismatch_count"] != 0
            or profile["indexed_reference_result_bytes_equal"] != 1):
        raise BroadQaExternalDataError(
            "recovery v4 candidate TRAIN profile 未闭合")
    target.mkdir()
    program_path = target / "candidate-program.json"
    profile_path = target / "profile.json"
    with program_path.open("xb") as handle:
        handle.write(canonical_json_line(program))
    with profile_path.open("xb") as handle:
        handle.write(canonical_json_line(profile))
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_KIND,
        "base_rule_pack_manifest_sha256": program[
            "base_rule_pack_manifest_sha256"],
        "candidate_program_sha256": program["candidate_program_sha256"],
        "evaluation_commitment_manifest_sha256": program[
            "evaluation_commitment_manifest_sha256"],
        "evaluation_or_reserve_payload_read_count": 0,
        "files": [_artifact(target / name, role=role)
                  for name, role in _FILES],
        "formal_run_count": 0,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_STATUS,
        "summary": {
            "base_character_rule_count": len(program["base_character_rules"]),
            "conflict_count": len(program["conflicts"]),
            "profile_query_count": profile["query_count"],
            "source_rule_count": audit["summary"][
                "full_pack_source_rule_count"],
            "target_rule_count": audit["summary"][
                "full_pack_target_rule_count"],
        },
        "teacher_api_llm_call_count": 0,
        "transfer_profile_sha256": program["transfer_profile_sha256"],
        "v4_protocol_manifest_sha256": program[
            "v4_protocol_manifest_sha256"],
        "v4_rule_pack_manifest_sha256": program[
            "v4_rule_pack_manifest_sha256"],
        "v4_training_audit_manifest_sha256": program[
            "v4_training_audit_manifest_sha256"],
    }
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(
        manifest_path.read_bytes())}


def read_normalization_recovery_v4_candidate_pack(
        candidate_dir: str | Path,
        *,
        expected_candidate_manifest_sha256: str,
        **arguments: object,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """重编译 candidate 并严格回读 program/profile/manifest。"""
    root = Path(candidate_dir).resolve()
    expected_sha = _sha_value(
        expected_candidate_manifest_sha256,
        label="recovery v4 candidate expected manifest")
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        program_bytes = (root / "candidate-program.json").read_bytes()
        profile_bytes = (root / "profile.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        stored_program = json.loads(program_bytes)
        profile = json.loads(profile_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("recovery v4 candidate artifact 不可读") from error
    if (_sha256(manifest_bytes) != expected_sha
            or not all(isinstance(item, dict)
                       for item in (manifest, stored_program, profile))
            or canonical_json_line(manifest) != manifest_bytes
            or canonical_json_line(stored_program) != program_bytes
            or canonical_json_line(profile) != profile_bytes):
        raise BroadQaExternalDataError(
            "recovery v4 candidate encoding/identity 漂移")
    program, audit = _materialize_candidate(**arguments)
    expected_files = [_artifact(root / name, role=role)
                      for name, role in _FILES]
    if (not _strict_equal(stored_program, program)
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_KIND
            or manifest.get("status")
            != NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_STATUS
            or manifest.get("candidate_program_sha256")
            != program["candidate_program_sha256"]
            or manifest.get("transfer_profile_sha256")
            != program["transfer_profile_sha256"]
            or not _strict_equal(manifest.get("files"), expected_files)
            or any(manifest.get(name) != 0 for name in (
                "evaluation_or_reserve_payload_read_count",
                "formal_run_count", "mastery_claimed",
                "production_enabled", "teacher_api_llm_call_count"))):
        raise BroadQaExternalDataError("recovery v4 candidate material 漂移")
    queries = derive_normalization_recovery_v4_candidate_queries(program)
    roster_payload = b"".join(canonical_json_line(item) for item in queries)
    if (profile.get("query_count") != len(queries)
            or profile.get("query_roster_sha256") != _sha256(roster_payload)
            or profile.get("indexed_reference_result_bytes_equal") != 1
            or not isinstance(profile.get("indexed"), dict)
            or not isinstance(profile.get("reference"), dict)
            or any(profile[side].get("failure_count") != 0
                   or profile[side].get("mismatch_count") != 0
                   for side in ("indexed", "reference"))
            or manifest.get("summary", {}).get("target_rule_count")
            != audit["summary"]["full_pack_target_rule_count"]):
        raise BroadQaExternalDataError("recovery v4 candidate profile 漂移")
    return ({**manifest, "manifest_sha256": expected_sha}, program, profile)


__all__ = [
    "NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V4_CANDIDATE_PACK_STATUS",
    "derive_normalization_recovery_v4_candidate_queries",
    "profile_normalization_recovery_v4_candidate",
    "publish_normalization_recovery_v4_candidate_pack",
    "read_normalization_recovery_v4_candidate_pack",
]
