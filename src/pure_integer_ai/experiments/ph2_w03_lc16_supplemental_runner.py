"""W-03 LC-16 supplemental 安全结果 pack 的只读执行桥。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_contract import (
    CASE_COUNT,
    DIRECTION_EVALUATION_COUNT,
    HOST_DIGEST_KEYS,
    MANIFEST_PATH,
    MAX_LOGIC_OPERATIONS,
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_READS,
    OVERLAY_PATH,
    OVERLAY_SHA256,
    SupplementalDirectionResult,
    W02_SUPPLEMENTAL_RECEIPT_SHA256,
    W03Lc16SupplementalError,
    W03Lc16SupplementalReport,
    W03_PARENT_RECEIPT_SHA256,
    read_w03_lc16_supplemental_manifest,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_evaluator import (
    aggregate_w03_lc16_supplemental,
)
from pure_integer_ai.experiments.ph2_w03_lc16_supplemental_catalog import (
    verify_w03_lc16_supplemental_files,
)


RESULT_PACK_FORMAT_VERSION = 1
RESULT_PACK_KIND = "PH2_W03_LC16_SUPPLEMENTAL_SAFE_RESULT_PACK"
RESULT_PACK_VERSION = "PH2-W03-LC16-SAFE-RESULT-PACK-20260801-A"
PRODUCER_KEY = "PH2-W03-LC16-INDEPENDENT-PRIVATE-EVALUATOR"
PRODUCER_REVISION = "20260801-A"

_PRIVATE_KEYS = frozenset({
    "accepted_surfaces", "candidate_sense", "expected_concept",
    "evaluator_label_payload",
    "expected_payload", "expected_sense", "expected_surface",
    "private_payload",
    "raw_observation",
})


class W03Lc16SupplementalRunnerError(RuntimeError):
    """安全 pack 身份、内容或公开依赖不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求 object 字段集合精确匹配。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise W03Lc16SupplementalRunnerError(f"{where} 字段不精确")
    return value


def _strict_int(value: Any, *, where: str, positive: bool = False) -> int:
    """校验不接受 bool 的非负或正整数。"""
    if type(value) is not int or value < int(positive):
        raise W03Lc16SupplementalRunnerError(f"{where} 整数非法")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """校验规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03Lc16SupplementalRunnerError(f"{where} 必须是小写 SHA-256")
    return value


def _reject_private(value: Any) -> None:
    """递归拒绝 private Observation、label 和 expected surface。"""
    if isinstance(value, dict):
        if _PRIVATE_KEYS.intersection(value):
            raise W03Lc16SupplementalRunnerError("安全 pack 含 private 字段")
        for item in value.values():
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)


def _host_digests(value: Any, *, where: str) -> dict[str, str]:
    """恢复精确五类 host 摘要。"""
    raw = _exact(value, set(HOST_DIGEST_KEYS), where=where)
    return {key: _sha256(raw[key], where=f"{where}.{key}")
            for key in HOST_DIGEST_KEYS}


@dataclass(frozen=True)
class W03Lc16SupplementalSafeResultPack:
    """private evaluator 唯一允许交给公开 runner 的投影。"""

    private_bundle_commitment_sha256: str
    direction_results: tuple[SupplementalDirectionResult, ...]
    host_digests_before: dict[str, str]
    host_digests_after: dict[str, str]
    private_path_count: int
    private_path_reads: int
    private_payload_bytes: int
    private_payload_reads: int
    evaluator_label_reads: int
    logic_operations: int
    evaluator_label_writes: int = 0
    host_write_count: int = 0
    independent_evaluator_module_separate: int = 1
    consumer_result_builder_reused: int = 0
    runtime_observed: int = 1

    def __post_init__(self) -> None:
        _sha256(
            self.private_bundle_commitment_sha256,
            where="private bundle commitment",
        )
        if (not isinstance(self.direction_results, tuple)
                or len(self.direction_results) != DIRECTION_EVALUATION_COUNT
                or any(not isinstance(item, SupplementalDirectionResult)
                       for item in self.direction_results)):
            raise W03Lc16SupplementalRunnerError(
                "安全 pack 必须精确包含 189 条方向结果")
        before = _host_digests(self.host_digests_before, where="host before")
        after = _host_digests(self.host_digests_after, where="host after")
        if before != after:
            raise W03Lc16SupplementalRunnerError("安全 pack 改变 host digest")
        for name in (
                "private_path_count", "private_path_reads",
                "private_payload_bytes", "private_payload_reads",
                "evaluator_label_reads", "logic_operations"):
            _strict_int(getattr(self, name), where=name, positive=True)
        if self.private_path_reads < self.private_path_count:
            raise W03Lc16SupplementalRunnerError("private path read 计数不闭合")
        if (self.private_payload_bytes > MAX_PAYLOAD_BYTES
                or self.private_payload_reads > MAX_PAYLOAD_READS
                or self.logic_operations > MAX_LOGIC_OPERATIONS):
            raise W03Lc16SupplementalRunnerError("安全 pack 资源预算超限")
        if self.evaluator_label_writes != 0 or self.host_write_count != 0:
            raise W03Lc16SupplementalRunnerError("安全 pack 必须零写")
        if (self.independent_evaluator_module_separate != 1
                or self.consumer_result_builder_reused != 0):
            raise W03Lc16SupplementalRunnerError("安全 pack evaluator 不独立")
        if self.runtime_observed != 1:
            raise W03Lc16SupplementalRunnerError("安全 pack 必须来自已观察 runtime")

    def to_public_dict(self) -> dict[str, Any]:
        """导出不含路径或 private payload 的 canonical object。"""
        value = {
            "artifact_kind": RESULT_PACK_KIND,
            "artifact_version": RESULT_PACK_VERSION,
            "case_count": CASE_COUNT,
            "direction_evaluations": DIRECTION_EVALUATION_COUNT,
            "direction_results": [
                item.to_public_dict() for item in self.direction_results],
            "format_version": RESULT_PACK_FORMAT_VERSION,
            "host_digests_after": dict(sorted(self.host_digests_after.items())),
            "host_digests_before": dict(sorted(self.host_digests_before.items())),
            "independence": {
                "consumer_result_builder_reused": self.consumer_result_builder_reused,
                "independent_evaluator_module_separate": self.independent_evaluator_module_separate,
            },
            "parent_overlay_sha256": OVERLAY_SHA256,
            "private_bundle_commitment_sha256": self.private_bundle_commitment_sha256,
            "private_reads": {
                "evaluator_label_reads": self.evaluator_label_reads,
                "private_path_count": self.private_path_count,
                "private_path_reads": self.private_path_reads,
                "private_payload_bytes": self.private_payload_bytes,
                "private_payload_reads": self.private_payload_reads,
            },
            "producer_key": PRODUCER_KEY,
            "producer_revision": PRODUCER_REVISION,
            "resource_use": {"logic_operations": self.logic_operations},
            "runtime_observed": self.runtime_observed,
            "w02_supplemental_receipt_sha256": (
                W02_SUPPLEMENTAL_RECEIPT_SHA256),
            "w03_parent_receipt_sha256": W03_PARENT_RECEIPT_SHA256,
            "writes": {
                "evaluator_label_writes": self.evaluator_label_writes,
                "host_write_count": self.host_write_count,
            },
        }
        _reject_private(value)
        return value

    def canonical_bytes(self) -> bytes:
        """返回单尾换行 canonical bytes。"""
        return canonical_json_bytes(self.to_public_dict()) + b"\n"

    @classmethod
    def from_public_dict(
            cls, value: Any,
            ) -> "W03Lc16SupplementalSafeResultPack":
        """严格恢复新 LC-16 family 的安全 pack。"""
        if isinstance(value, dict) and (
                value.get("case_count") == 5
                or "PH2-W03-PRIVATE" in str(
                    value.get("artifact_version", "")).upper()
                or value.get("artifact_kind") == "PH2_W03_PRIVATE_AGGREGATE"):
            raise W03Lc16SupplementalRunnerError(
                "原 W-03/5-case pack 不属于 LC-16 supplemental")
        _reject_private(value)
        raw = _exact(value, {
            "artifact_kind", "artifact_version", "case_count",
            "direction_evaluations", "direction_results", "format_version",
            "host_digests_after", "host_digests_before", "independence",
            "parent_overlay_sha256", "private_bundle_commitment_sha256",
            "private_reads", "producer_key", "producer_revision",
            "resource_use", "runtime_observed",
            "w02_supplemental_receipt_sha256", "w03_parent_receipt_sha256",
            "writes",
        }, where="supplemental safe result pack")
        if (raw["format_version"] != RESULT_PACK_FORMAT_VERSION
                or raw["artifact_kind"] != RESULT_PACK_KIND
                or raw["artifact_version"] != RESULT_PACK_VERSION
                or raw["producer_key"] != PRODUCER_KEY
                or raw["producer_revision"] != PRODUCER_REVISION):
            raise W03Lc16SupplementalRunnerError("安全 pack producer/revision 非法")
        if (raw["parent_overlay_sha256"] != OVERLAY_SHA256
                or raw["w03_parent_receipt_sha256"]
                != W03_PARENT_RECEIPT_SHA256
                or raw["w02_supplemental_receipt_sha256"]
                != W02_SUPPLEMENTAL_RECEIPT_SHA256):
            raise W03Lc16SupplementalRunnerError("安全 pack parent SHA 漂移")
        if (raw["case_count"] != CASE_COUNT
                or raw["direction_evaluations"]
                != DIRECTION_EVALUATION_COUNT):
            raise W03Lc16SupplementalRunnerError("安全 pack 63/189 覆盖漂移")
        reads = _exact(raw["private_reads"], {
            "evaluator_label_reads", "private_path_count",
            "private_path_reads", "private_payload_bytes",
            "private_payload_reads",
        }, where="private_reads")
        writes = _exact(raw["writes"], {
            "evaluator_label_writes", "host_write_count",
        }, where="writes")
        independence = _exact(raw["independence"], {
            "consumer_result_builder_reused",
            "independent_evaluator_module_separate",
        }, where="independence")
        resources = _exact(
            raw["resource_use"], {"logic_operations"}, where="resource_use")
        try:
            results = tuple(
                SupplementalDirectionResult.from_public_dict(item)
                for item in raw["direction_results"])
        except (TypeError, ValueError, W03Lc16SupplementalError) as error:
            raise W03Lc16SupplementalRunnerError(
                "安全 pack 方向结果回读失败") from error
        return cls(
            _sha256(
                raw["private_bundle_commitment_sha256"],
                where="private bundle commitment"),
            results,
            _host_digests(raw["host_digests_before"], where="host before"),
            _host_digests(raw["host_digests_after"], where="host after"),
            reads["private_path_count"],
            reads["private_path_reads"],
            reads["private_payload_bytes"],
            reads["private_payload_reads"],
            reads["evaluator_label_reads"],
            resources["logic_operations"],
            writes["evaluator_label_writes"],
            writes["host_write_count"],
            independence["independent_evaluator_module_separate"],
            independence["consumer_result_builder_reused"],
            raw["runtime_observed"],
        )


@dataclass(frozen=True)
class W03Lc16SupplementalRunOutcome:
    """只读 runner 的资格状态或缺 pack 阻断状态。"""

    status: str
    blocker_code: str | None
    safe_pack_sha256: str | None
    report: W03Lc16SupplementalReport | None


def read_w03_lc16_supplemental_safe_result_pack(
        path: str | Path,
        ) -> tuple[W03Lc16SupplementalSafeResultPack, str]:
    """严格回读 canonical 安全 pack 并返回其逐字节摘要。"""
    target = Path(path)
    try:
        payload = target.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise W03Lc16SupplementalRunnerError("安全 pack newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        pack = W03Lc16SupplementalSafeResultPack.from_public_dict(value)
    except W03Lc16SupplementalRunnerError:
        raise
    except Exception as error:
        raise W03Lc16SupplementalRunnerError("安全 pack 损坏") from error
    if pack.canonical_bytes() != payload:
        raise W03Lc16SupplementalRunnerError("安全 pack 非 canonical bytes")
    return pack, hashlib.sha256(payload).hexdigest()


def run_w03_lc16_supplemental_safe_pack(
        safe_pack_path: str | Path,
        *,
        repository_root: str | Path,
        ) -> W03Lc16SupplementalRunOutcome:
    """零写读取公开依赖和安全 pack，并交给独立聚合器。"""
    pack_path = Path(safe_pack_path)
    if not pack_path.is_file():
        return W03Lc16SupplementalRunOutcome(
            "BLOCKED", "SAFE_RESULT_PACK_MISSING", None, None)
    root = Path(repository_root).resolve()
    try:
        manifest = read_w03_lc16_supplemental_manifest(
            root / Path(*MANIFEST_PATH.split("/")))
        if (manifest.parent_overlay_sha256 != OVERLAY_SHA256
                or manifest.w03_parent_receipt_sha256
                != W03_PARENT_RECEIPT_SHA256
                or manifest.w02_supplemental_receipt_sha256
                != W02_SUPPLEMENTAL_RECEIPT_SHA256):
            raise W03Lc16SupplementalRunnerError(
                "supplemental manifest parent 漂移")
        verify_w03_lc16_supplemental_files(
            manifest, repository_root=root)
        overlay = read_d03_lc16_successor_overlay(
            root / Path(*OVERLAY_PATH.split("/")))
        if overlay.sha256() != OVERLAY_SHA256:
            raise W03Lc16SupplementalRunnerError("overlay SHA 漂移")
        pack, pack_sha256 = read_w03_lc16_supplemental_safe_result_pack(
            pack_path)
        report = aggregate_w03_lc16_supplemental(
            overlay,
            pack.direction_results,
            safe_result_pack_sha256=pack_sha256,
            private_bundle_commitment_sha256=(
                pack.private_bundle_commitment_sha256),
            host_digests_before=pack.host_digests_before,
            host_digests_after=pack.host_digests_after,
            private_path_reads=pack.private_path_reads,
            private_payload_bytes=pack.private_payload_bytes,
            private_payload_reads=pack.private_payload_reads,
            evaluator_label_reads=pack.evaluator_label_reads,
            evaluator_label_writes=pack.evaluator_label_writes,
            host_write_count=pack.host_write_count,
            independent_evaluator_module_separate=(
                pack.independent_evaluator_module_separate),
            consumer_result_builder_reused=pack.consumer_result_builder_reused,
            runtime_observed=pack.runtime_observed,
        )
    except W03Lc16SupplementalRunnerError:
        raise
    except Exception as error:
        raise W03Lc16SupplementalRunnerError(
            "supplemental safe pack 无法聚合") from error
    return W03Lc16SupplementalRunOutcome(
        report.status, None, pack_sha256, report)


__all__ = [
    "PRODUCER_KEY", "PRODUCER_REVISION", "RESULT_PACK_FORMAT_VERSION",
    "RESULT_PACK_KIND", "RESULT_PACK_VERSION",
    "W03Lc16SupplementalRunOutcome", "W03Lc16SupplementalRunnerError",
    "W03Lc16SupplementalSafeResultPack",
    "read_w03_lc16_supplemental_safe_result_pack",
    "run_w03_lc16_supplemental_safe_pack",
]
