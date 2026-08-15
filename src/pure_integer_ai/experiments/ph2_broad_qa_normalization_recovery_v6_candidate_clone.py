"""构造 recovery-v6 formal evaluator 使用的只读 candidate clone。"""
from __future__ import annotations

import hashlib
import json

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate import (
    derive_normalization_recovery_v6_candidate_preflight,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _sha256(payload: bytes) -> str:
    """返回 clone 或外部 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def clone_normalization_recovery_v6_candidate(
        candidate: dict[str, object],
        *,
        candidate_manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """规范 round-trip candidate，并重跑标签盲 preflight 验证 clone。"""
    manifest_sha = _sha_value(
        candidate_manifest_sha256, label="v6 clone candidate manifest")
    if not isinstance(candidate, dict):
        raise BroadQaExternalDataError("v6 candidate clone 输入非对象")
    encoded = canonical_json_line(candidate)
    try:
        clone = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v6 candidate clone 编码失败") from error
    if not isinstance(clone, dict) or canonical_json_line(clone) != encoded:
        raise BroadQaExternalDataError("v6 candidate clone round-trip 漂移")
    preflight = derive_normalization_recovery_v6_candidate_preflight(clone)
    if (preflight.get("failure_count") != 0
            or preflight.get("indexed_reference_mismatch_count") != 0
            or preflight.get("valid_scope_all_applicable") != 1
            or preflight.get("invalid_scope_rejected") != 1):
        raise BroadQaExternalDataError("v6 candidate clone preflight 未闭合")
    identity = {
        "candidate_clone_sha256": _sha256(canonical_json_bytes(clone)),
        "candidate_manifest_sha256": manifest_sha,
        "candidate_program_sha256": clone["candidate_program_sha256"],
        "clone_write_count": 0,
        "formal_run_count": 0,
        "mastery_claimed": 0,
        "preflight_result_rows_sha256": preflight["result_rows_sha256"],
        "production_enabled": 0,
    }
    return clone, identity


__all__ = ["clone_normalization_recovery_v6_candidate"]
