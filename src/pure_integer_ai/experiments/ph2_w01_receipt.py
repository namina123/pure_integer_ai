"""正式中文 PH2 W-01 报告 bundle 的不可覆盖 receipt 与规范回读。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_w01_contract import W01_ALLOWED_MODES
from pure_integer_ai.experiments.ph2_w01_faults import W01FaultPoint
from pure_integer_ai.experiments.ph2_w01_report import (
    W01_RUN_CURSOR_NAME,
    W01_RUN_EXECUTION_NAME,
    W01_RUN_MANIFEST_NAME,
    W01_RUN_REPORT_NAME,
    W01_RUN_RESOURCE_NAME,
    W01_RUN_SEAL_NAME,
    W01RunOutcome,
    read_w01_run,
    run_directory,
)
from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes


W01_FORMAL_ROOT = "data/ph2/manifests/w01_v1"
W01_FORMAL_RUN_ID = 1
W01_FORMAL_RECEIPT_V1_PATH = (
    f"{W01_FORMAL_ROOT}/ph2_w01_stage0_receipt_v1.json")
W01_FORMAL_RECEIPT_PATH = (
    f"{W01_FORMAL_ROOT}/ph2_w01_stage0_receipt_v2.json")
W01_FORMAL_ARTIFACT_KIND = "PH2_W01_STAGE0_FORMAL_RECEIPT"
W01_FORMAL_ARTIFACT_VERSION = (
    "PH2-W01-stage0-formal-receipt-v2-supersedes-v1")

W01_IMPLEMENTATION_PATHS = (
    "src/pure_integer_ai/experiments/ph2_w01_contract.py",
    "src/pure_integer_ai/experiments/ph2_w01_faults.py",
    "src/pure_integer_ai/experiments/ph2_w01_receipt.py",
    "src/pure_integer_ai/experiments/ph2_w01_report.py",
    "src/pure_integer_ai/experiments/ph2_w01_runtime.py",
    "src/pure_integer_ai/experiments/ph2_w01_shards.py",
    "src/pure_integer_ai/experiments/ph2_w01_transaction.py",
    "src/pure_integer_ai/experiments/run_ph2_language_stage0.py",
    "src/pure_integer_ai/experiments/run_ph2_w01_formal_receipt.py",
    "src/pure_integer_ai/experiments/training_shard_runtime.py",
)
W01_TEST_PATHS = (
    "tests/test_w01_language_stage0_contract.py",
    "tests/test_w01_language_stage0_process.py",
    "tests/test_w01_language_stage0_receipt.py",
    "tests/test_w01_language_stage0_runtime.py",
)
W01_RUN_BUNDLE_NAMES = (
    W01_RUN_EXECUTION_NAME,
    W01_RUN_MANIFEST_NAME,
    W01_RUN_SEAL_NAME,
    W01_RUN_RESOURCE_NAME,
    W01_RUN_CURSOR_NAME,
    W01_RUN_REPORT_NAME,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class W01ReceiptError(RuntimeError):
    """正式 receipt 非规范、证据漂移、状态越级或覆盖冲突。"""


def _relative_path(value: object) -> str:
    """核验仓库相对 POSIX 路径并拒绝逃逸。"""
    if not isinstance(value, str) or not value:
        raise W01ReceiptError("receipt 文件路径不能为空")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise W01ReceiptError("receipt 文件路径必须是规范仓库相对路径")
    return value


def _sha256(value: object) -> str:
    """核验小写 SHA-256 文本。"""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise W01ReceiptError("receipt SHA-256 非法")
    return value


def _exact_dict(value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 字段精确匹配合同。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise W01ReceiptError(f"{where} 字段集合漂移")
    return value


@dataclass(frozen=True, order=True)
class W01FileIdentity:
    """正式 W-01 run、实现或测试文件的稳定字节身份。"""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """核验安全路径、非负字节数和 SHA-256。"""
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path))
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise W01ReceiptError("receipt 文件大小必须是非负严格整数")
        object.__setattr__(self, "sha256", _sha256(self.sha256))

    def to_dict(self) -> dict[str, object]:
        """导出规范文件身份。"""
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "W01FileIdentity":
        """从精确 JSON object 恢复文件身份。"""
        raw = _exact_dict(
            value,
            {"relative_path", "sha256", "size_bytes"},
            where="W01FileIdentity",
        )
        return cls(
            str(raw["relative_path"]),
            raw["size_bytes"],
            str(raw["sha256"]),
        )


def _identity(repository: Path, relative_path: str) -> W01FileIdentity:
    """从仓库内安全文件构造一次读取的字节身份。"""
    relative = _relative_path(relative_path)
    target = (repository / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise W01ReceiptError(f"正式 W-01 证据文件缺失: {relative}")
    payload = target.read_bytes()
    return W01FileIdentity(
        relative,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _identity_tuple(value: object, *, where: str) -> tuple[W01FileIdentity, ...]:
    """恢复排序、非空且路径唯一的文件身份集合。"""
    if not isinstance(value, list) or not value:
        raise W01ReceiptError(f"{where} 必须是非空数组")
    identities = tuple(W01FileIdentity.from_dict(item) for item in value)
    if identities != tuple(sorted(identities)):
        raise W01ReceiptError(f"{where} 必须按路径排序")
    paths = tuple(item.relative_path for item in identities)
    if len(paths) != len(set(paths)):
        raise W01ReceiptError(f"{where} 路径重复")
    return identities


def _expected_execution_state() -> dict[str, int]:
    """返回协议通过且训练、teacher、mastery 和学习写全零的终态。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W01_PROTOCOL_VERIFIED": 1,
        "W02_STARTED": 0,
        "assessment_updates": 0,
        "companion_writes": 0,
        "core_learning_writes": 0,
        "evaluator_label_writes": 0,
        "formal_training_runs": 0,
        "mastered_claims": 0,
        "memory_learning_writes": 0,
        "protocol_execution_runs": 1,
        "readiness_claims": 0,
        "teacher_calls": 0,
        "use_learning_writes": 0,
        "w02_semantic_writes": 0,
    }


def _verification_contract() -> dict[str, object]:
    """冻结 1/2/4、三恢复态、七故障和双 hash seed 的验收矩阵。"""
    return {
        "fault_coverage_points": list(W01FaultPoint.coverage_points()),
        "fresh_restart_resume_distinct": 1,
        "mode_keys": list(W01_ALLOWED_MODES),
        "python_hash_seeds": [0, 1],
        "sqlite_cross_process": 1,
        "worker_count_is_scheduling_only": 1,
        "worker_counts": [1, 2, 4],
    }


def _honest_boundary() -> dict[str, object]:
    """冻结 W-01 仅验证协议且下一阶段尚未开始的诚实边界。"""
    return {
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_unique_stage": "W-02",
        "protocol_only": 1,
        "teacher_or_llm_used": 0,
        "w02_semantic_learning_writes": 0,
        "w02_started": 0,
    }


@dataclass(frozen=True)
class W01FormalReceipt:
    """绑定正式 run bundle、全部新实现/测试和零学习终态的 receipt。"""

    d03_identity: dict[str, Any]
    execution_state: dict[str, int]
    formal_run: dict[str, Any]
    formal_run_inventory: tuple[W01FileIdentity, ...]
    implementation_inventory: tuple[W01FileIdentity, ...]
    test_inventory: tuple[W01FileIdentity, ...]
    verification_contract: dict[str, Any]
    honest_boundary: dict[str, Any]
    superseded_receipt_identity: W01FileIdentity
    receipt_relative_path: str = W01_FORMAL_RECEIPT_PATH

    def __post_init__(self) -> None:
        """核验 receipt 的固定状态、证据集合和不可自包含边界。"""
        _exact_dict(self.d03_identity, {
            "content_commit_sha1", "context_key", "global_manifest_path",
            "global_manifest_sha256", "receipt_sha256", "release_key",
            "stage_manifest_path", "stage_manifest_sha256",
        }, where="formal receipt D-03 identity")
        if self.execution_state != _expected_execution_state():
            raise W01ReceiptError("正式 W-01 execution state 非零或语义混合")
        _exact_dict(self.formal_run, {
            "adopted_manifest_count", "artifact_digest", "cursor_digest",
            "execution_identity_key", "logical_state_digest",
            "merge_publication_count", "report_digest", "run_id",
            "transaction_event_count",
        }, where="formal receipt run")
        if (self.formal_run.get("run_id") != W01_FORMAL_RUN_ID
                or self.formal_run.get("adopted_manifest_count") != 1
                or self.formal_run.get("merge_publication_count") != 1
                or self.formal_run.get("transaction_event_count") != 4):
            raise W01ReceiptError("正式 W-01 run 未单次 adopted/merge/transaction 闭合")
        for name in (
                "artifact_digest", "cursor_digest", "logical_state_digest",
                "report_digest"):
            _sha256(self.formal_run.get(name))
        execution_key = self.formal_run.get("execution_identity_key")
        if (not isinstance(execution_key, list) or len(execution_key) != 32
                or any(type(item) is not int or not 0 <= item <= 255
                       for item in execution_key)):
            raise W01ReceiptError("正式 W-01 execution identity key 非法")
        expected_run_paths = tuple(sorted(
            f"{W01_FORMAL_ROOT}/{run_directory('.', W01_FORMAL_RUN_ID).name}/{name}"
            for name in W01_RUN_BUNDLE_NAMES
        ))
        if tuple(item.relative_path for item in self.formal_run_inventory) != (
                expected_run_paths):
            raise W01ReceiptError("正式 W-01 run bundle inventory 不完整")
        if tuple(item.relative_path for item in self.implementation_inventory) != (
                tuple(sorted(W01_IMPLEMENTATION_PATHS))):
            raise W01ReceiptError("正式 W-01 implementation inventory 不完整")
        if tuple(item.relative_path for item in self.test_inventory) != tuple(
                sorted(W01_TEST_PATHS)):
            raise W01ReceiptError("正式 W-01 test inventory 不完整")
        if self.verification_contract != _verification_contract():
            raise W01ReceiptError("正式 W-01 verification contract 漂移")
        if self.honest_boundary != _honest_boundary():
            raise W01ReceiptError("正式 W-01 诚实边界漂移")
        if (not isinstance(self.superseded_receipt_identity, W01FileIdentity)
                or self.superseded_receipt_identity.relative_path
                != W01_FORMAL_RECEIPT_V1_PATH):
            raise W01ReceiptError("正式 W-01 superseded receipt identity 漂移")
        if _relative_path(self.receipt_relative_path) != W01_FORMAL_RECEIPT_PATH:
            raise W01ReceiptError("正式 W-01 receipt path 漂移")
        all_paths = {
            item.relative_path
            for inventory in (
                self.formal_run_inventory,
                self.implementation_inventory,
                self.test_inventory,
            )
            for item in inventory
        }
        if self.receipt_relative_path in all_paths:
            raise W01ReceiptError("正式 W-01 receipt 必须 self-excluded")

    def to_dict(self) -> dict[str, Any]:
        """导出规范正式 receipt。"""
        return {
            "artifact_kind": W01_FORMAL_ARTIFACT_KIND,
            "artifact_version": W01_FORMAL_ARTIFACT_VERSION,
            "d03_identity": dict(self.d03_identity),
            "execution_state": dict(self.execution_state),
            "formal_run": dict(self.formal_run),
            "formal_run_inventory": [
                item.to_dict() for item in self.formal_run_inventory],
            "format_version": 1,
            "honest_boundary": dict(self.honest_boundary),
            "implementation_inventory": [
                item.to_dict() for item in self.implementation_inventory],
            "receipt_relative_path": self.receipt_relative_path,
            "receipt_self_excluded": 1,
            "stage": {
                "next_stage_key": "W-02",
                "stage_key": "W-01",
                "status": "W01_PROTOCOL_VERIFIED",
            },
            "status": "W01_PROTOCOL_VERIFIED",
            "superseded_receipt_identity": (
                self.superseded_receipt_identity.to_dict()),
            "test_inventory": [item.to_dict() for item in self.test_inventory],
            "verification_contract": dict(self.verification_contract),
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的纯整数规范 receipt 字节。"""
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        """返回正式 receipt 的 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "W01FormalReceipt":
        """从字段精确的 JSON object 恢复正式 receipt。"""
        raw = _exact_dict(value, {
            "artifact_kind", "artifact_version", "d03_identity",
            "execution_state", "formal_run", "formal_run_inventory",
            "format_version", "honest_boundary", "implementation_inventory",
            "receipt_relative_path", "receipt_self_excluded", "stage",
            "status", "superseded_receipt_identity", "test_inventory",
            "verification_contract",
        }, where="W01FormalReceipt")
        if (raw["format_version"] != 1
                or raw["artifact_kind"] != W01_FORMAL_ARTIFACT_KIND
                or raw["artifact_version"] != W01_FORMAL_ARTIFACT_VERSION
                or raw["status"] != "W01_PROTOCOL_VERIFIED"
                or raw["receipt_self_excluded"] != 1
                or raw["stage"] != {
                    "next_stage_key": "W-02",
                    "stage_key": "W-01",
                    "status": "W01_PROTOCOL_VERIFIED",
                }):
            raise W01ReceiptError("正式 W-01 receipt identity/status 非法")
        return cls(
            raw["d03_identity"],
            raw["execution_state"],
            raw["formal_run"],
            _identity_tuple(raw["formal_run_inventory"], where="run inventory"),
            _identity_tuple(
                raw["implementation_inventory"], where="implementation inventory"),
            _identity_tuple(raw["test_inventory"], where="test inventory"),
            raw["verification_contract"],
            raw["honest_boundary"],
            W01FileIdentity.from_dict(raw["superseded_receipt_identity"]),
            str(raw["receipt_relative_path"]),
        )


def _run_summary(outcome: W01RunOutcome) -> dict[str, Any]:
    """投影 receipt 必须绑定的正式 run 逻辑和事务身份。"""
    return {
        "adopted_manifest_count": outcome.adopted_manifest_count,
        "artifact_digest": outcome.artifact_digest,
        "cursor_digest": outcome.cursor_digest,
        "execution_identity_key": outcome.execution_identity[
            "execution_identity_key"],
        "logical_state_digest": outcome.logical_state_digest,
        "merge_publication_count": outcome.merge_publication_count,
        "report_digest": outcome.report_digest,
        "run_id": outcome.report["execution_identity"]["run_id"],
        "transaction_event_count": outcome.transaction_event_count,
    }


def build_w01_formal_receipt(repository_root: str | Path) -> W01FormalReceipt:
    """从已 adopted 正式 run 和当前实现/测试字节构造 self-excluded receipt。"""
    repository = Path(repository_root).resolve()
    run_dir = repository / W01_FORMAL_ROOT / run_directory(
        ".", W01_FORMAL_RUN_ID).name
    outcome = read_w01_run(run_dir)
    run_prefix = f"{W01_FORMAL_ROOT}/{run_dir.name}"
    return W01FormalReceipt(
        d03_identity=dict(outcome.report["d03_identity"]),
        execution_state=dict(outcome.report["execution_state"]),
        formal_run=_run_summary(outcome),
        formal_run_inventory=tuple(sorted(
            _identity(repository, f"{run_prefix}/{name}")
            for name in W01_RUN_BUNDLE_NAMES
        )),
        implementation_inventory=tuple(sorted(
            _identity(repository, path) for path in W01_IMPLEMENTATION_PATHS)),
        test_inventory=tuple(sorted(
            _identity(repository, path) for path in W01_TEST_PATHS)),
        verification_contract=_verification_contract(),
        honest_boundary=_honest_boundary(),
        superseded_receipt_identity=_identity(
            repository, W01_FORMAL_RECEIPT_V1_PATH),
    )


def write_w01_formal_receipt(
        receipt: W01FormalReceipt,
        target: str | Path,
        ) -> Path:
    """独占或同字节幂等写 receipt，绝不覆盖同路径异字节。"""
    if not isinstance(receipt, W01FormalReceipt):
        raise W01ReceiptError("正式 W-01 receipt 类型错误")
    path = Path(target)
    payload = receipt.canonical_bytes()
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise W01ReceiptError("正式 W-01 receipt 不可覆盖")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except OSError as exc:
        raise W01ReceiptError("正式 W-01 receipt 无法独占写入") from exc
    return path


def _verify_identity(repository: Path, identity: W01FileIdentity) -> None:
    """逐字节回验一个 receipt 文件证据且拒绝路径逃逸。"""
    actual = _identity(repository, identity.relative_path)
    if actual != identity:
        raise W01ReceiptError(f"正式 W-01 文件身份漂移: {identity.relative_path}")


def _read_canonical_receipt(path: Path) -> dict[str, Any]:
    """读取并要求与 W-01 canonical JSON 编码逐字节一致。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise W01ReceiptError("正式 W-01 receipt 无法解析") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W01ReceiptError("正式 W-01 receipt 不是规范 JSON")
    return value


def read_w01_formal_receipt(
        repository_root: str | Path,
        relative_path: str = W01_FORMAL_RECEIPT_PATH,
        ) -> W01FormalReceipt:
    """规范回读 receipt、全部文件身份和正式 run bundle 的交叉绑定。"""
    repository = Path(repository_root).resolve()
    relative = _relative_path(relative_path)
    target = (repository / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise W01ReceiptError("正式 W-01 receipt 路径缺失或逃逸")
    receipt = W01FormalReceipt.from_dict(_read_canonical_receipt(target))
    if receipt.receipt_relative_path != relative:
        raise W01ReceiptError("正式 W-01 receipt 请求路径与内嵌路径不一致")
    for inventory in (
            receipt.formal_run_inventory,
            receipt.implementation_inventory,
            receipt.test_inventory):
        for identity in inventory:
            _verify_identity(repository, identity)
    _verify_identity(repository, receipt.superseded_receipt_identity)
    run_dir = repository / W01_FORMAL_ROOT / run_directory(
        ".", W01_FORMAL_RUN_ID).name
    outcome = read_w01_run(run_dir)
    if (_run_summary(outcome) != receipt.formal_run
            or outcome.report["d03_identity"] != receipt.d03_identity
            or outcome.report["execution_state"] != receipt.execution_state):
        raise W01ReceiptError("正式 W-01 receipt 与 run report/cursor/事务漂移")
    report = outcome.report
    if (report.get("status") != "W01_PROTOCOL_VERIFIED"
            or report.get("stage", {}).get("next_stage_key") != "W-02"
            or report.get("visibility", {}).get("payload_reads") != 0
            or report.get("visibility", {}).get("payload_bytes") != 0
            or report.get("visibility", {}).get("train_pack_count") != 0
            or report.get("visibility", {}).get("held_out_visible_count") != 0
            or report.get("visibility", {}).get("evaluator_visible_count") != 0):
        raise W01ReceiptError("正式 W-01 report 可见性或状态越级")
    if (tuple(report.get("fault_contract", {}).get("injectable_points", ()))
            != W01FaultPoint.injectable_points()):
        raise W01ReceiptError("正式 W-01 report 故障合同漂移")
    return receipt


__all__ = [
    "W01_FORMAL_RECEIPT_PATH",
    "W01_FORMAL_RECEIPT_V1_PATH",
    "W01_FORMAL_ROOT",
    "W01FileIdentity",
    "W01FormalReceipt",
    "W01ReceiptError",
    "build_w01_formal_receipt",
    "read_w01_formal_receipt",
    "write_w01_formal_receipt",
]
