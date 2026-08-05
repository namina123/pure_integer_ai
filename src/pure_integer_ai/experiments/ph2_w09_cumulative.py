"""W-09-03 累计 typed language runtime 与 public parent evidence ledger。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09FrozenContract,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09TrainingPayload
from pure_integer_ai.experiments.ph2_w09_registry import (
    W09Registry,
    W09RegistryAuditReport,
    W09RegistryError,
    audit_w09_registry_payload,
    build_w09_registry,
)
from pure_integer_ai.experiments.ph2_w09_types import W09DirectionalResult


W09_PUBLIC_PARENT_RECEIPTS = (
    ("data/ph2/manifests/w01_v1/ph2_w01_stage0_receipt_v2.json", "W01_PROTOCOL_VERIFIED"),
    ("data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json", "PASS"),
    ("data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
    ("data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
    ("data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
    ("data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
    ("data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
    ("data/ph2/manifests/d03_v1/w08_runtime_evidence_receipt_v1.json", "RUNTIME_EVIDENCED"),
)
W09_CUMULATIVE_TRAIN_SOURCE_COUNT = 535
W09_CUMULATIVE_TRAIN_OBSERVATION_COUNT = 309
W09_CUMULATIVE_TRAIN_EVIDENCE_COUNT = 309


class W09CumulativeError(RuntimeError):
    """累计 runtime 的 parent、payload 或 consumer identity 不闭合。"""


def _safe_relative(value: object) -> str:
    """校验只读 parent path 是安全 POSIX 相对路径。"""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W09CumulativeError("W-09 parent path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise W09CumulativeError("W-09 parent path escapes repository")
    return path.as_posix()


def _validate_parent_receipts(
        receipts: object,
        ) -> tuple["W09PublicParentReceipt", ...]:
    """要求 public parent tuple 的类型、顺序、路径和状态精确冻结。"""
    if (
        not isinstance(receipts, tuple)
        or any(not isinstance(item, W09PublicParentReceipt) for item in receipts)
        or tuple((item.relative_path, item.expected_status) for item in receipts)
        != W09_PUBLIC_PARENT_RECEIPTS
    ):
        raise W09CumulativeError("W-09 parent receipt coverage is incomplete")
    return receipts


@dataclass(frozen=True)
class W09PublicParentReceipt:
    """一个只读 public receipt 的路径、状态、大小和 canonical SHA。"""

    relative_path: str
    expected_status: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """校验 parent receipt 身份和最低状态，不打开 Git 外 artifact。"""
        _safe_relative(self.relative_path)
        if not isinstance(self.expected_status, str) or not self.expected_status:
            raise W09CumulativeError("W-09 parent status is invalid")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise W09CumulativeError("W-09 parent size is invalid")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise W09CumulativeError("W-09 parent SHA is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回不包含 surface、label 或绝对路径的 parent identity。"""
        return digest_value({
            "expected_status": self.expected_status,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        })


@dataclass(frozen=True)
class W09CumulativeTrainingDelta:
    """W09 train-only 累计输入及其 registry 审计计数。"""

    registry_audit: W09RegistryAuditReport
    source_ref_count: int
    observation_count: int
    evidence_count: int
    new_pack_count: int
    retention_parent_count: int
    delta_commitment: str

    def __post_init__(self) -> None:
        """要求 34-pack 全量累计输入，防止只跑最后 authored 子集。"""
        if not isinstance(self.registry_audit, W09RegistryAuditReport):
            raise W09CumulativeError("W-09 cumulative registry audit is invalid")
        values = (
            self.source_ref_count,
            self.observation_count,
            self.evidence_count,
            self.new_pack_count,
            self.retention_parent_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise W09CumulativeError("W-09 cumulative training count is invalid")
        if (
            self.source_ref_count != W09_CUMULATIVE_TRAIN_SOURCE_COUNT
            or self.observation_count != W09_CUMULATIVE_TRAIN_OBSERVATION_COUNT
            or self.evidence_count != W09_CUMULATIVE_TRAIN_EVIDENCE_COUNT
            or self.new_pack_count != 34
            or self.retention_parent_count != len(W09_PUBLIC_PARENT_RECEIPTS)
        ):
            raise W09CumulativeError("W-09 cumulative train delta is incomplete")
        if (
            not isinstance(self.delta_commitment, str)
            or len(self.delta_commitment) != 64
            or any(char not in "0123456789abcdef" for char in self.delta_commitment)
        ):
            raise W09CumulativeError("W-09 cumulative delta commitment is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回累计训练 delta 的稳定身份。"""
        return digest_value({
            "delta_commitment": self.delta_commitment,
            "evidence_count": self.evidence_count,
            "new_pack_count": self.new_pack_count,
            "observation_count": self.observation_count,
            "registry_audit": list(self.registry_audit.pack_counts),
            "retention_parent_count": self.retention_parent_count,
            "source_ref_count": self.source_ref_count,
        })


@dataclass(frozen=True)
class W09CumulativeReport:
    """累计 runtime 的可回读 bounded report，绝不发布 mastered/readiness。"""

    parent_receipts: tuple[W09PublicParentReceipt, ...]
    training_delta: W09CumulativeTrainingDelta | None
    consumer_cells: tuple[tuple[str, str, str], ...]
    shared_engine_count: int
    runtime_connected: int
    formal_evidenced: int
    language_capability_mastered: int
    language_readiness: int

    def __post_init__(self) -> None:
        """校验 parent/consumer coverage 计数和未发布状态。"""
        _validate_parent_receipts(self.parent_receipts)
        if self.training_delta is not None and not isinstance(
                self.training_delta, W09CumulativeTrainingDelta):
            raise W09CumulativeError("W-09 cumulative training delta is invalid")
        if self.shared_engine_count != 1 or self.runtime_connected != 1:
            raise W09CumulativeError("W-09 cumulative runtime is not connected")
        if any(value != 0 for value in (
                self.formal_evidenced,
                self.language_capability_mastered,
                self.language_readiness,
        )):
            raise W09CumulativeError("W-09-03 cannot publish formal/mastered/readiness")
        expected_cells = tuple(
            (carrier, consumer)
            for carrier in W09_CARRIER_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        actual_cells = tuple(
            (carrier, consumer)
            for carrier, consumer, _ in self.consumer_cells
        )
        if actual_cells != expected_cells:
            raise W09CumulativeError("W-09 cumulative consumer coverage is incomplete")
        for carrier, consumer, status in self.consumer_cells:
            if carrier not in W09_CARRIER_KEYS or consumer not in W09_CONSUMER_KEYS:
                raise W09CumulativeError("W-09 cumulative consumer cell is invalid")
            if status not in {"CONNECTED", "PENDING"}:
                raise W09CumulativeError("W-09 cumulative consumer status is invalid")

    @property
    def complete(self) -> bool:
        """W09-03 只表示 runtime connected；正式 complete 留给 B/C。"""
        return self.training_delta is not None and all(
            status == "CONNECTED" for _, _, status in self.consumer_cells
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回不含私有 payload 的累计报告键。"""
        return digest_value({
            "consumer_cells": [list(item) for item in self.consumer_cells],
            "formal_evidenced": self.formal_evidenced,
            "language_capability_mastered": self.language_capability_mastered,
            "language_readiness": self.language_readiness,
            "parents": [item.stable_key() for item in self.parent_receipts],
            "shared_engine_count": self.shared_engine_count,
            "training_delta": (
                list(self.training_delta.stable_key())
                if self.training_delta is not None else []
            ),
            "runtime_connected": self.runtime_connected,
        })


def read_w09_public_parent_receipts(
        repository_root: str | Path,
        ) -> tuple[W09PublicParentReceipt, ...]:
    """只读回读 W01-W08 public receipts 并校验 canonical status/bytes。"""
    result = [
        read_w09_public_receipt(repository_root, relative, expected_status)
        for relative, expected_status in W09_PUBLIC_PARENT_RECEIPTS
    ]
    return _validate_parent_receipts(tuple(result))


def read_w09_public_receipt(
        repository_root: str | Path,
        relative_path: str,
        expected_status: str,
        ) -> W09PublicParentReceipt:
    """严格回读一个冻结路径和状态的 canonical public receipt。"""
    root = Path(repository_root).resolve()
    relative = _safe_relative(relative_path)
    path = root / PurePosixPath(relative)
    if not path.is_file() or path.is_symlink():
        raise W09CumulativeError("W-09 public parent receipt is missing")
    try:
        payload = path.read_bytes()
        body = payload[:-1] if payload.endswith(b"\n") else payload
        value = parse_canonical_json_bytes(body, require_object=True)
    except (OSError, DatasetContractError) as error:
        raise W09CumulativeError("W-09 public parent receipt is invalid") from error
    if not isinstance(value, dict) or value.get("status") != expected_status:
        raise W09CumulativeError("W-09 public parent receipt status drifted")
    return W09PublicParentReceipt(
        relative,
        expected_status,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


class W09CumulativeRuntime:
    """把 public parent evidence、34 train pack 和共享 typed engine 组合成一个 runtime。"""

    def __init__(
            self,
            repository_root: str | Path,
            context: W09FrozenContract,
            *,
            parent_receipts: tuple[W09PublicParentReceipt, ...] | None = None,
            ) -> None:
        """绑定只读 parent ledger、W09 registry 和九 carrier 单一 engine。"""
        if not isinstance(context, W09FrozenContract):
            raise W09CumulativeError("W-09 cumulative context is invalid")
        self.repository_root = Path(repository_root).resolve()
        self.context = context
        self.registry: W09Registry = build_w09_registry(context)
        self.parent_receipts = (
            read_w09_public_parent_receipts(self.repository_root)
            if parent_receipts is None else parent_receipts
        )
        _validate_parent_receipts(self.parent_receipts)
        self.training_delta: W09CumulativeTrainingDelta | None = None
        self._consumer_cells: dict[tuple[str, str], str] = {}

    def ingest_training_payload(
            self,
            payload: W09TrainingPayload,
            ) -> W09CumulativeTrainingDelta:
        """审计并一次性吸收全部 34 train pack，不读取 dev/held-out/evaluator。"""
        if not isinstance(payload, W09TrainingPayload):
            raise W09CumulativeError("W-09 cumulative payload type is invalid")
        try:
            audit = audit_w09_registry_payload(payload, self.context)
        except W09RegistryError as error:
            raise W09CumulativeError("W-09 cumulative payload registry audit failed") from error
        if self.training_delta is not None:
            raise W09CumulativeError("W-09 cumulative training payload replayed")
        delta = W09CumulativeTrainingDelta(
            audit,
            audit.source_ref_count,
            audit.observation_count,
            audit.training_evidence_count,
            len(self.context.candidate_pack_keys),
            len(self.parent_receipts),
            _payload_commitment(payload),
        )
        self.training_delta = delta
        return delta

    def consume_directional(
            self,
            carrier_key: str,
            consumer_key: str,
            result: W09DirectionalResult,
            ) -> None:
        """登记一个载体/consumer 的独立 typed U/R/G result。"""
        if carrier_key not in W09_CARRIER_KEYS or consumer_key not in W09_CONSUMER_KEYS:
            raise W09CumulativeError("W-09 cumulative carrier or consumer is invalid")
        if not isinstance(result, W09DirectionalResult):
            raise W09CumulativeError("W-09 cumulative directional result is invalid")
        if result.request.consumer_key != consumer_key:
            raise W09CumulativeError("W-09 cumulative result direction drifted")
        key = (carrier_key, consumer_key)
        if key in self._consumer_cells:
            raise W09CumulativeError("W-09 cumulative consumer cell already consumed")
        self._consumer_cells[key] = "CONNECTED"

    def state_key(self) -> tuple[int, ...]:
        """返回 parent、registry、training delta 和 consumer coverage 的稳定键。"""
        return digest_value({
            "consumer_cells": [
                [carrier, consumer, status]
                for (carrier, consumer), status in sorted(self._consumer_cells.items())
            ],
            "context": list(self.context.stable_key()),
            "parents": [item.stable_key() for item in self.parent_receipts],
            "registry": [
                item.carrier_adapter_key for item in self.registry.carrier_bindings
            ],
            "training_delta": (
                list(self.training_delta.stable_key())
                if self.training_delta is not None else []
            ),
        })

    def report(self) -> W09CumulativeReport:
        """返回累计 runtime bounded report，缺失 consumer 明确保持 pending。"""
        if not isinstance(self.registry, W09Registry):
            raise W09CumulativeError("W-09 cumulative registry is invalid")
        bindings = self.registry.carrier_bindings
        shared_engine_count = len({item.semantic_engine_key for item in bindings})
        runtime_connected = int(
            tuple(item.carrier_key for item in bindings) == W09_CARRIER_KEYS
            and len({item.carrier_adapter_key for item in bindings})
            == len(W09_CARRIER_KEYS)
            and shared_engine_count == 1
        )
        cells = tuple(
            (carrier, consumer, self._consumer_cells.get((carrier, consumer), "PENDING"))
            for carrier in W09_CARRIER_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        return W09CumulativeReport(
            self.parent_receipts,
            self.training_delta,
            cells,
            shared_engine_count,
            runtime_connected,
            0,
            0,
            0,
        )


def _payload_commitment(payload: W09TrainingPayload) -> str:
    """只用公开 stable key 和 typed payload 摘要生成累计输入 commitment。"""
    value = {
        "evidence": [item.stable_key.stable_key() for item in payload.training_evidence],
        "observations": [item.stable_key.stable_key() for item in payload.observations],
        "sources": [item.stable_key.stable_key() for item in payload.source_refs],
    }
    return hashlib.sha256(bytes(digest_value(value))).hexdigest()


def open_w09_cumulative_runtime(
        repository_root: str | Path,
        context: W09FrozenContract | None = None,
        ) -> W09CumulativeRuntime:
    """现场构建 W09-03 cumulative runtime，不消费 formal Candidate/private guard。"""
    frozen = (
        open_w09_frozen_contract(repository_root)
        if context is None else context
    )
    return W09CumulativeRuntime(repository_root, frozen)


__all__ = [
    "W09CUMULATIVE_TRAIN_EVIDENCE_COUNT",
    "W09CUMULATIVE_TRAIN_OBSERVATION_COUNT",
    "W09CUMULATIVE_TRAIN_SOURCE_COUNT",
    "W09_PUBLIC_PARENT_RECEIPTS",
    "W09CumulativeError",
    "W09CumulativeReport",
    "W09CumulativeTrainingDelta",
    "W09CumulativeRuntime",
    "W09PublicParentReceipt",
    "open_w09_cumulative_runtime",
    "read_w09_public_receipt",
    "read_w09_public_parent_receipts",
]
