"""W-02 v2 candidate 的公开 base fence 与 Git 外不可覆盖 freeze builder。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w02_learning_v2 import (
    W02_MORPHOLOGY_ADAPTER_VERSION,
)
from pure_integer_ai.experiments.ph2_w02_runtime import (
    W02RunOutcome,
    W02RuntimeConfig,
)
from pure_integer_ai.experiments.ph2_w02_runtime_v2 import (
    W02_FORMAL_RUNTIME_VERSION,
)


W02_V2_BASE_FENCE_KIND = "PH2_W02_V2_BASE_FENCE"
W02_V2_CANDIDATE_FREEZE_KIND = "PH2_W02_V2_CANDIDATE_HOST_FREEZE"
W02_V2_CANDIDATE_FREEZE_NAME = "candidate_host_freeze_v2.json"
W02_V2_HOST_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w02_candidate_v2.py",
    "src/pure_integer_ai/experiments/ph2_w02_contract.py",
    "src/pure_integer_ai/experiments/ph2_w02_faults.py",
    "src/pure_integer_ai/experiments/ph2_w02_learning.py",
    "src/pure_integer_ai/experiments/ph2_w02_learning_v2.py",
    "src/pure_integer_ai/experiments/ph2_w02_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w02_runtime_v2.py",
    "src/pure_integer_ai/experiments/ph2_w02_shards.py",
    "src/pure_integer_ai/experiments/ph2_w02_transaction.py",
    "src/pure_integer_ai/experiments/ph2_w02_use.py",
)
W02_V2_HOST_TEST_PATHS = (
    "tests/test_w02_language_stage1_candidate_v2.py",
    "tests/test_w02_language_stage1_contract.py",
    "tests/test_w02_language_stage1_learning.py",
    "tests/test_w02_language_stage1_morphology_v2.py",
    "tests/test_w02_language_stage1_runtime.py",
    "tests/test_w02_language_stage1_runtime_v2.py",
)


def _sha256(path: Path) -> str:
    """返回普通文件 SHA-256，并拒绝缺失或目录输入。"""
    if not path.is_file():
        raise RuntimeError(f"W-02 v2 freeze 文件不存在: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(
        repository_root: Path,
        paths: tuple[str, ...],
        ) -> list[dict[str, str]]:
    """按冻结相对路径形成代码或测试 SHA inventory。"""
    result = []
    for relative in paths:
        normalized = Path(relative).as_posix()
        full = (repository_root / normalized).resolve()
        try:
            full.relative_to(repository_root)
        except ValueError as exc:
            raise RuntimeError("W-02 v2 freeze 路径逃逸 repository") from exc
        result.append({"path": normalized, "sha256": _sha256(full)})
    return result


def _read_v1_public_freeze(path: Path) -> dict[str, Any]:
    """只读 v1 公开 freeze 摘要字段，不打开首轮 private report 内容。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("W-02 v1 public freeze 无法回读") from exc
    required = {
        "ablation_order",
        "artifact_kind",
        "base_fence_key",
        "d03_thresholds",
        "evaluation_order",
        "format_version",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise RuntimeError("W-02 v1 public freeze 字段不完整")
    return value


def build_w02_v2_base_fence(
        repository_root: str | Path,
        *,
        remote_commit_sha1: str,
        v1_freeze_path: str | Path,
        v1_report_path: str | Path,
        ) -> dict[str, Any]:
    """绑定公开实现、测试、v1 负结果身份和不变评价门，形成 v2 base fence。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise RuntimeError("W-02 v2 repository_root 不存在")
    if (not isinstance(remote_commit_sha1, str)
            or len(remote_commit_sha1) != 40
            or any(item not in "0123456789abcdef"
                   for item in remote_commit_sha1)):
        raise RuntimeError("W-02 v2 remote commit 必须是小写 SHA-1")
    prior_freeze = Path(v1_freeze_path).resolve()
    prior_report = Path(v1_report_path).resolve()
    prior = _read_v1_public_freeze(prior_freeze)
    return {
        "ablation_order": list(prior["ablation_order"]),
        "adapter_version": W02_MORPHOLOGY_ADAPTER_VERSION,
        "artifact_kind": W02_V2_BASE_FENCE_KIND,
        "code_inventory": _inventory(repository, W02_V2_HOST_CODE_PATHS),
        "d03_thresholds": prior["d03_thresholds"],
        "evaluation_order": list(prior["evaluation_order"]),
        "format_version": 2,
        "prior_v1_failure": {
            "candidate_freeze_sha256": _sha256(prior_freeze),
            "first_report_sha256": _sha256(prior_report),
        },
        "remote_commit_sha1": remote_commit_sha1,
        "runtime_version": W02_FORMAL_RUNTIME_VERSION,
        "test_inventory": _inventory(repository, W02_V2_HOST_TEST_PATHS),
    }


def w02_v2_base_fence_key(value: dict[str, Any]) -> tuple[int, ...]:
    """把完整 base fence canonical bytes 摘要为 32 个整数分量。"""
    if not isinstance(value, dict) or value.get("artifact_kind") != (
            W02_V2_BASE_FENCE_KIND):
        raise RuntimeError("W-02 v2 base fence 类型非法")
    return tuple(hashlib.sha256(canonical_json_bytes(value)).digest())


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    """封存 candidate root 中除 freeze 自身外的全部文件 identity。"""
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W02_V2_CANDIDATE_FREEZE_NAME:
            continue
        result.append({
            "path": relative,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    if not result:
        raise RuntimeError("W-02 v2 candidate root 没有可冻结 artifact")
    return result


def publish_w02_v2_candidate_freeze(
        repository_root: str | Path,
        artifact_root: str | Path,
        *,
        config: W02RuntimeConfig,
        outcome: W02RunOutcome,
        base_fence: dict[str, Any],
        ) -> tuple[Path, str]:
    """核验正式 v2 outcome 和 base fence 后，不可覆盖发布 candidate freeze。"""
    if not isinstance(config, W02RuntimeConfig):
        raise TypeError("W-02 v2 freeze config 类型错误")
    if not isinstance(outcome, W02RunOutcome):
        raise TypeError("W-02 v2 freeze outcome 类型错误")
    repository = Path(repository_root).resolve()
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise RuntimeError("W-02 v2 artifact_root 不存在")
    expected = build_w02_v2_base_fence(
        repository,
        remote_commit_sha1=config.current_remote_commit_sha1,
        v1_freeze_path=(root.parent / "formal_candidate_v1"
                        / "candidate_host_freeze.json"),
        v1_report_path=(root.parent / "formal_candidate_v1"
                        / "w02_private_evaluation_first_run.json"),
    )
    if expected != base_fence:
        raise RuntimeError("W-02 v2 base fence 在正式运行后发生漂移")
    base_key = w02_v2_base_fence_key(base_fence)
    if config.base_fence_key != base_key:
        raise RuntimeError("W-02 v2 config 未绑定完整 base fence")
    if (outcome.execution_state.get("W02_STARTED") != 1
            or outcome.execution_state.get("W03_STARTED") != 0
            or outcome.execution_state.get("LANGUAGE_CAPABILITY_MASTERED") != 0
            or outcome.execution_state.get("LANGUAGE_READINESS") != 0):
        raise RuntimeError("W-02 v2 candidate outcome 越界声明能力或 W-03")
    freeze = {
        "ablation_order": base_fence["ablation_order"],
        "adapter_version": W02_MORPHOLOGY_ADAPTER_VERSION,
        "artifact_inventory": _artifact_inventory(root),
        "artifact_kind": W02_V2_CANDIDATE_FREEZE_KIND,
        "base_fence_key": list(base_key),
        "base_fence_sha256": hashlib.sha256(
            canonical_json_bytes(base_fence)).hexdigest(),
        "code_inventory": base_fence["code_inventory"],
        "d03_thresholds": base_fence["d03_thresholds"],
        "evaluation_order": base_fence["evaluation_order"],
        "execution_state": outcome.execution_state,
        "format_version": 2,
        "host_digests": {
            "artifact": outcome.artifact_digest,
            "core": outcome.core_digest,
            "cursor": outcome.cursor_digest,
            "logical": outcome.logical_state_digest,
            "manifest": outcome.dump_manifest_sha256,
            "memory": outcome.memory_digest,
            "use": outcome.use_digest,
        },
        "owner_write_counts": {
            "core_learning_writes": outcome.resource_report[
                "core_learning_writes"],
            "memory_learning_writes": outcome.resource_report[
                "memory_learning_writes"],
            "use_learning_writes": outcome.resource_report[
                "use_learning_writes"],
            "word_form_writes": outcome.resource_report["word_form_writes"],
        },
        "prior_v1_failure": base_fence["prior_v1_failure"],
        "publication_counts": {
            "adopted_manifest_count": outcome.adopted_manifest_count,
            "merge_publication_count": outcome.merge_publication_count,
            "transaction_event_count": outcome.transaction_event_count,
        },
        "remote_commit_sha1": config.current_remote_commit_sha1,
        "resource_actual": {
            key.removeprefix("actual_"): value
            for key, value in sorted(outcome.resource_report.items())
            if key.startswith("actual_")
        },
        "run_id": config.run_id,
        "runtime_version": W02_FORMAL_RUNTIME_VERSION,
        "self_excluded": 1,
        "test_inventory": base_fence["test_inventory"],
    }
    target = root / W02_V2_CANDIDATE_FREEZE_NAME
    payload = canonical_json_bytes(freeze)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError("W-02 v2 candidate freeze 不可覆盖") from exc
    return target, hashlib.sha256(payload).hexdigest()


__all__ = [
    "W02_V2_BASE_FENCE_KIND",
    "W02_V2_CANDIDATE_FREEZE_KIND",
    "W02_V2_CANDIDATE_FREEZE_NAME",
    "W02_V2_HOST_CODE_PATHS",
    "W02_V2_HOST_TEST_PATHS",
    "build_w02_v2_base_fence",
    "publish_w02_v2_candidate_freeze",
    "w02_v2_base_fence_key",
]
