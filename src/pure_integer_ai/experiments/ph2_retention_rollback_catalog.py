"""构建 LC-10 retention、回滚与范围收缩的正式零执行账。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_retention_rollback_contract import (
    ARTIFACT_STATUS,
    EXECUTION_STATE,
    FORMAT_VERSION,
    FUTURE_PREREQUISITE_STATES,
    OUTCOME_CLASSES,
    RETENTION_PHASE_KEYS,
    RUNTIME_STATUS,
    VERIFIER_DIMENSIONS,
    VERIFIER_NE_CONDITIONS,
    RetentionCheckpoint,
    RetentionDimensionResult,
    RetentionEvidenceFile,
    RetentionProtocolFixture,
    RetentionRollbackManifest,
    RetentionRuntimeBinding,
    evaluate_retention_fixture,
)


LC10_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc10_retention_rollback_manifest_v1.json")
LC10_ARTIFACT_VERSION = "LC-10-retention-rollback-manifest-v1"

_EVIDENCE_PATHS = (
    "src/pure_integer_ai/cognition/shared/candidate_runtime.py",
    "src/pure_integer_ai/cognition/shared/memory_batch.py",
    "src/pure_integer_ai/cognition/shared/memory_event_log.py",
    "src/pure_integer_ai/cognition/shared/scoped_persistence.py",
    "src/pure_integer_ai/cognition/shared/situation_state.py",
    "src/pure_integer_ai/experiments/evaluation_isolation.py",
    "src/pure_integer_ai/storage/memory_recovery.py",
    "src/pure_integer_ai/training/cursor.py",
)


class RetentionRollbackCatalogError(RuntimeError):
    """LC-10 fixture 或现有设施文件身份不满足正式账要求。"""


def _digest(label: str) -> str:
    """为纯协议 fixture 生成稳定、可区分的状态摘要。"""
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sha256_path(path: Path) -> str:
    """以固定块大小计算现有设施文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _checkpoints(*, contracted: bool) -> tuple[RetentionCheckpoint, ...]:
    """构造严格按 LC-10 顺序的零写 append-only 状态摘要。"""
    source_visibility = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    event_counts = (0, 1, 1, 1, 1, 2, 3, 3, 4, 4)
    dependent = (
        "DEPENDENT_A",
        "DEPENDENT_A_B",
        "DEPENDENT_A_B",
        "DEPENDENT_A_B",
        "DEPENDENT_A_B",
        "DEPENDENT_AFTER_WITHDRAW_B",
        "DEPENDENT_A",
        "DEPENDENT_A",
        "DEPENDENT_A",
        "DEPENDENT_A",
    )
    matrices = (
        "MATRIX_A",
        "MATRIX_A_B",
        "MATRIX_A_B",
        "MATRIX_A_B",
        "MATRIX_A_B",
        "MATRIX_AFTER_WITHDRAW_B",
        "MATRIX_A",
        "MATRIX_A",
        "MATRIX_CONTRACTED" if contracted else "MATRIX_A",
        "MATRIX_CONTRACTED" if contracted else "MATRIX_A",
    )
    scopes = (
        ("SCOPE_CONTRACTED" if contracted and index >= 8 else "SCOPE_FULL")
        for index in range(len(RETENTION_PHASE_KEYS))
    )
    scope_values = tuple(scopes)
    dump_digest = _digest("DUMP_AFTER_B_V1")
    return tuple(
        RetentionCheckpoint(
            phase_key,
            _digest("CORE_A_BIT_IDENTITY"),
            _digest("UNAFFECTED_CAPABILITIES_BIT_IDENTITY"),
            _digest(dependent[index]),
            _digest(matrices[index]),
            _digest(scope_values[index]),
            dump_digest if index in {3, 4} else "NONE",
            source_visibility[index],
            event_counts[index],
            0,
        )
        for index, phase_key in enumerate(RETENTION_PHASE_KEYS)
    )


def build_retention_protocol_fixtures() -> tuple[RetentionProtocolFixture, ...]:
    """构建可接受、范围收缩和遗忘拒绝三组直接证据。"""
    accepted_dimensions = tuple(sorted((
        RetentionDimensionResult(
            "CAPABILITY_A_INTERFERENCE", "PASS", "PASS", -1,
            1, 1, 1, 0, "INTERFERENCE"),
        RetentionDimensionResult(
            "CAPABILITY_A_NO_CHANGE", "PASS", "PASS", 0,
            1, 1, 0, 0, "NO_CHANGE"),
        RetentionDimensionResult(
            "CAPABILITY_A_POSITIVE_TRANSFER", "PASS", "PASS", 1,
            1, 1, 1, 0, "POSITIVE_TRANSFER"),
    ), key=lambda item: item.dimension_key))
    contracted_dimensions = tuple(sorted((
        RetentionDimensionResult(
            "CAPABILITY_A_CONTRACTED", "PASS", "NE", 0,
            1, 0, 1, 1, "SCOPE_CONTRACTION"),
        RetentionDimensionResult(
            "CAPABILITY_A_UNAFFECTED", "PASS", "PASS", 0,
            1, 1, 0, 0, "NO_CHANGE"),
    ), key=lambda item: item.dimension_key))
    forgotten_dimensions = (
        RetentionDimensionResult(
            "CAPABILITY_A_FORGOTTEN", "PASS", "FAIL", -1,
            1, 1, 1, 0, "FORGETTING"),
    )
    fixtures = (
        RetentionProtocolFixture(
            "LC10_FORGETTING_REJECT_V1",
            forgotten_dimensions,
            _checkpoints(contracted=False),
            "REJECT",
            "OLD_CAPABILITY_FORGOTTEN",
        ),
        RetentionProtocolFixture(
            "LC10_NO_CHANGE_ACCEPT_V1",
            accepted_dimensions,
            _checkpoints(contracted=False),
            "PASS",
            "NONE",
        ),
        RetentionProtocolFixture(
            "LC10_SCOPE_CONTRACTION_ACCEPT_V1",
            contracted_dimensions,
            _checkpoints(contracted=True),
            "PASS",
            "NONE",
        ),
    )
    for fixture in fixtures:
        if evaluate_retention_fixture(
                fixture.dimension_results, fixture.checkpoints) != (
                fixture.expected_verdict, fixture.expected_failure_code):
            raise RetentionRollbackCatalogError("LC-10 fixture 直接结果漂移")
    return fixtures


def _runtime_bindings() -> tuple[RetentionRuntimeBinding, ...]:
    """登记可复用设施及仍未执行或仍缺通用 caller 的边界。"""
    return (
        RetentionRuntimeBinding(
            "APPEND_ONLY_MEMORY_EVENT_LOG",
            "AVAILABLE_NOT_EXECUTED",
            ("APPEND_ONLY_EVENT_SEQUENCE", "VISIBLE_ACTIVE_PROJECTION"),
            ("src/pure_integer_ai/cognition/shared/memory_event_log.py",),
            ("LC10_RETENTION_EPISODE_NOT_EXECUTED",),
        ),
        RetentionRuntimeBinding(
            "CANDIDATE_LIFECYCLE_RUNTIME",
            "AVAILABLE_NOT_EXECUTED",
            ("CANDIDATE_DECISION_REPLAY", "PROJECTION_STATE_REBUILD"),
            ("src/pure_integer_ai/cognition/shared/candidate_runtime.py",),
            ("LC10_RETENTION_EPISODE_NOT_EXECUTED",),
        ),
        RetentionRuntimeBinding(
            "CURSOR_DUMP_RESUME",
            "AVAILABLE_NOT_EXECUTED",
            ("CURSOR_DUMP", "CURSOR_EXACT_RESUME"),
            ("src/pure_integer_ai/training/cursor.py",),
            ("D03_CURSOR_NOT_FROZEN", "LC10_DUMP_RESUME_NOT_EXECUTED"),
        ),
        RetentionRuntimeBinding(
            "GENERAL_SOURCE_WITHDRAWAL",
            "PROTOCOL_ONLY_NE",
            ("DEPENDENCY_SCOPED_WITHDRAWAL", "LOCAL_VISIBILITY_REMOVAL"),
            (
                "src/pure_integer_ai/cognition/shared/scoped_persistence.py",
                "src/pure_integer_ai/cognition/shared/situation_state.py",
            ),
            (
                "GENERAL_SOURCE_DEPENDENCY_MAP_MISSING",
                "GENERAL_SOURCE_WITHDRAWAL_CALLER_MISSING",
            ),
        ),
        RetentionRuntimeBinding(
            "M10_MEMORY_BATCH_ROLLBACK",
            "AVAILABLE_NOT_EXECUTED",
            ("BATCH_LOCAL_ROLLBACK", "RECOVERY_PACKAGE_REPLAY"),
            (
                "src/pure_integer_ai/cognition/shared/memory_batch.py",
                "src/pure_integer_ai/storage/memory_recovery.py",
            ),
            ("LC10_ROLLBACK_EPISODE_NOT_EXECUTED",),
        ),
        RetentionRuntimeBinding(
            "SITUATION_DEPENDENCY_INVALIDATION",
            "AVAILABLE_NOT_EXECUTED",
            ("DEPENDENCY_INDEX_REBUILD", "LOCAL_INVALIDATION_RECEIPT"),
            ("src/pure_integer_ai/cognition/shared/situation_state.py",),
            ("CROSS_CAPABILITY_WITHDRAWAL_NOT_CONNECTED",),
        ),
        RetentionRuntimeBinding(
            "V06_RETENTION_ISOLATION_CLONE",
            "FUTURE_REQUIRED",
            ("CLONE_ONLY_MUTATION", "HOST_BIT_IDENTITY"),
            ("src/pure_integer_ai/experiments/evaluation_isolation.py",),
            ("V06_RETENTION_CLONE_NOT_EXECUTED",),
        ),
    )


def _evidence_inventory(repository_root: Path) -> tuple[RetentionEvidenceFile, ...]:
    """回读所有 binding 文件，拒绝绝对路径和失效身份。"""
    result = []
    for relative_path in _EVIDENCE_PATHS:
        path = repository_root / Path(*relative_path.split("/"))
        if not path.is_file():
            raise RetentionRollbackCatalogError(
                f"LC-10 evidence 文件缺失: {relative_path}")
        result.append(RetentionEvidenceFile(
            relative_path, path.stat().st_size, _sha256_path(path)))
    return tuple(result)


def build_retention_rollback_manifest(
        repository_root: str | Path,
        ) -> RetentionRollbackManifest:
    """从当前仓库文件身份构建 LC-10 正式零执行 manifest。"""
    repository = Path(repository_root).resolve()
    if not (repository / "src" / "pure_integer_ai").is_dir():
        raise RetentionRollbackCatalogError("repository_root 非当前源码仓库")
    return RetentionRollbackManifest(
        FORMAT_VERSION,
        LC10_ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        RUNTIME_STATUS,
        "LC-10",
        RETENTION_PHASE_KEYS,
        OUTCOME_CLASSES,
        build_retention_protocol_fixtures(),
        _runtime_bindings(),
        _evidence_inventory(repository),
        VERIFIER_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        CanonicalJsonObject.from_value(FUTURE_PREREQUISITE_STATES),
        0,
        0,
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "LC10_ARTIFACT_VERSION",
    "LC10_MANIFEST_PATH",
    "RetentionRollbackCatalogError",
    "build_retention_protocol_fixtures",
    "build_retention_rollback_manifest",
]
