"""W09-04 累计维度与 J-LC-W09 public bounded coverage runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    D03Lc16OverlayError,
    read_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import SCOPE_KEYS
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    TASK_KEYS,
    LanguageCoverageV2Error,
    read_language_capability_coverage_v2,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_WALL_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_cumulative import (
    W09CumulativeError,
    W09CumulativeRuntime,
    W09PublicParentReceipt,
    read_w09_public_receipt,
)
from pure_integer_ai.experiments.ph2_w09_types import (
    W09DirectionalResult,
    W09ResultState,
)


W09_LC16_OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
W09_LC_COVERAGE_PATH = "data/ph2/manifests/language_capability_coverage_v2.json"
W09_RETENTION_SCOPE_KEY = "RETENTION_CONTINUAL_LEARNING"
W09_DIMENSIONAL_PUBLIC_STATES = (
    "PUBLIC_BOUNDED_PASS",
    "PUBLIC_BOUNDED_FAIL",
    "PUBLIC_BOUNDED_NE",
)
W09_J_LC_PUBLIC_STATES = (
    "PUBLIC_BOUNDED_NOT_FORMAL",
    "PUBLIC_BOUNDED_FAIL",
    "PUBLIC_BOUNDED_NE",
)
W09_SCOPE_PARENT_RECEIPTS = (
    (
        "BOUNDARY_OOV",
        "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json",
        "PASS",
    ),
    (
        "SENSE_CONCEPT",
        "data/ph2/manifests/w03_lc16_supplemental_runtime_receipt_v1.json",
        "PASS",
    ),
    (
        "PRIMITIVE_STRUCTURE",
        "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json",
        "RUNTIME_EVIDENCED",
    ),
    (
        "ROLE_PROPOSITION_SCOPE",
        "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json",
        "RUNTIME_EVIDENCED",
    ),
    (
        "RELATION_FAMILIES",
        "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json",
        "RUNTIME_EVIDENCED",
    ),
    (
        "LOGIC_MODAL_NESTED_SCOPE",
        "data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json",
        "RUNTIME_EVIDENCED",
    ),
    (
        "DISCOURSE_REFERENCE_GENERATION",
        "data/ph2/manifests/d03_v1/w08_runtime_evidence_receipt_v1.json",
        "RUNTIME_EVIDENCED",
    ),
)


class W09DimensionalError(RuntimeError):
    """W09 coverage、当前 result、学习 delta 或硬合取不闭合。"""


def _strict_key(value: object, *, where: str) -> tuple[int, ...]:
    """要求值是 W09 typed result 使用的 32-byte 稳定整数键。"""
    if (
        not isinstance(value, tuple)
        or len(value) != 32
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09DimensionalError(f"{where} identity is invalid")
    return value


def _strict_sha256(value: object, *, where: str) -> str:
    """要求值是小写 SHA-256。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise W09DimensionalError(f"{where} SHA is invalid")
    return value


def _directional_identity(result: W09DirectionalResult) -> tuple[int, ...]:
    """把 request、choice、Use/outcome 和 verifier 合成当前 cell 身份。"""
    if not isinstance(result, W09DirectionalResult):
        raise W09DimensionalError("W-09 directional result type is invalid")
    return digest_value({
        "choice": list(result.choice.choice_key),
        "consumer": result.request.consumer_key,
        "failure_kind": result.verifier.failure_kind,
        "input": result.request.input_commitment,
        "outcome": list(result.use_outcome.outcome_key),
        "request": list(result.request.request_key),
        "selected": list(result.choice.selected_candidate_key),
        "status": result.verifier.status.value,
        "use": list(result.use_outcome.use_key),
        "verifier": list(result.verifier.verifier_key),
    })


def _directional_component_identities(
        result: W09DirectionalResult,
        ) -> tuple[tuple[int, ...], ...]:
    """返回一个 current result 的五类独立身份。"""
    identities = (
        result.request.request_key,
        result.choice.choice_key,
        result.use_outcome.use_key,
        result.use_outcome.outcome_key,
        result.verifier.verifier_key,
    )
    if len(set(identities)) != len(identities):
        raise W09DimensionalError("W-09 directional result shares component identity")
    return identities


@dataclass(frozen=True)
class W09ContinualLearningEvidence:
    """W09 retention cell 的非 teacher 状态变化、Use/outcome 与隔离审计。"""

    before_state_key: tuple[int, ...]
    after_state_key: tuple[int, ...]
    source_evidence_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    teacher_call_count: int
    host_write_count: int

    def __post_init__(self) -> None:
        """要求状态真实变化，并保持 teacher 与正式 host 零写。"""
        for name in (
            "before_state_key",
            "after_state_key",
            "source_evidence_key",
            "use_key",
            "outcome_key",
        ):
            _strict_key(getattr(self, name), where=name)
        if self.before_state_key == self.after_state_key:
            raise W09DimensionalError("W-09 continual state did not change")
        if (
            type(self.teacher_call_count) is not int
            or type(self.host_write_count) is not int
            or self.teacher_call_count != 0
            or self.host_write_count != 0
        ):
            raise W09DimensionalError("W-09 continual evidence crossed isolation")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 payload surface 的学习变化身份。"""
        return digest_value({
            "after": list(self.after_state_key),
            "before": list(self.before_state_key),
            "host_writes": self.host_write_count,
            "outcome": list(self.outcome_key),
            "source": list(self.source_evidence_key),
            "teacher_calls": self.teacher_call_count,
            "use": list(self.use_key),
        })


@dataclass(frozen=True)
class W09CoverageCell:
    """一个 scope×carrier×direction 的当前 typed 运行证据。"""

    scope_key: str
    carrier_key: str
    consumer_key: str
    evidence_kind: str
    parent_receipt_sha256: str | None
    result: W09DirectionalResult
    continual_learning: W09ContinualLearningEvidence | None

    def __post_init__(self) -> None:
        """绑定方向、父 receipt 或 W09 学习 delta，不允许共享旧布尔位。"""
        if self.scope_key not in SCOPE_KEYS:
            raise W09DimensionalError("W-09 coverage scope is invalid")
        if self.carrier_key not in W09_CARRIER_KEYS:
            raise W09DimensionalError("W-09 coverage carrier is invalid")
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09DimensionalError("W-09 coverage consumer is invalid")
        if not isinstance(self.result, W09DirectionalResult):
            raise W09DimensionalError("W-09 coverage result is invalid")
        if self.result.request.consumer_key != self.consumer_key:
            raise W09DimensionalError("W-09 coverage result direction drifted")
        _directional_identity(self.result)
        _directional_component_identities(self.result)
        if self.scope_key == W09_RETENTION_SCOPE_KEY:
            if (
                self.evidence_kind != "CONTINUAL_LEARNING_EVIDENCE"
                or self.parent_receipt_sha256 is not None
                or not isinstance(
                    self.continual_learning, W09ContinualLearningEvidence
                )
            ):
                raise W09DimensionalError("W-09 continual coverage evidence is incomplete")
            if (
                self.continual_learning.use_key != self.result.use_outcome.use_key
                or self.continual_learning.outcome_key
                != self.result.use_outcome.outcome_key
            ):
                raise W09DimensionalError("W-09 continual delta is not bound to Use/outcome")
        else:
            if (
                self.evidence_kind != "RETENTION_EVIDENCE"
                or self.continual_learning is not None
                or self.parent_receipt_sha256 is None
            ):
                raise W09DimensionalError("W-09 retention coverage evidence is incomplete")
            _strict_sha256(self.parent_receipt_sha256, where="retention parent")

    @property
    def key(self) -> tuple[str, str, str]:
        """返回 overlay 规定的 cell key。"""
        return self.scope_key, self.carrier_key, self.consumer_key

    def stable_key(self) -> tuple[int, ...]:
        """返回 parent/current result/learning delta 的联合身份。"""
        return digest_value({
            "carrier": self.carrier_key,
            "consumer": self.consumer_key,
            "continual": (
                []
                if self.continual_learning is None
                else list(self.continual_learning.stable_key())
            ),
            "evidence_kind": self.evidence_kind,
            "parent": self.parent_receipt_sha256 or "",
            "result": list(_directional_identity(self.result)),
            "scope": self.scope_key,
        })


@dataclass(frozen=True)
class W09CapabilityAudit:
    """LC-01..16 的历史前沿与当前接口审计状态。"""

    task_key: str
    prior_state: str
    current_state: str

    def __post_init__(self) -> None:
        if self.task_key not in TASK_KEYS:
            raise W09DimensionalError("W-09 LC task key is invalid")
        if not isinstance(self.prior_state, str) or not self.prior_state:
            raise W09DimensionalError("W-09 LC prior state is invalid")
        if self.current_state != "CURRENT_INTERFACE_AUDITED":
            raise W09DimensionalError("W-09 LC current audit state is invalid")


@dataclass(frozen=True)
class W09DimensionalAblation:
    """aggregator 或单 cell 关闭后的正交击穿结果。"""

    component_key: str
    target_dimension_key: str
    target_status: str
    affected_cell_count: int
    preserved_cell_count: int
    unrelated_failure_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.component_key, str) or not self.component_key:
            raise W09DimensionalError("W-09 ablation component is invalid")
        if self.target_dimension_key != "W-09-DIMENSIONAL_PASS":
            raise W09DimensionalError("W-09 dimensional ablation target drifted")
        if self.target_status != "FAIL" or self.unrelated_failure_count != 0:
            raise W09DimensionalError("W-09 dimensional ablation is not orthogonal")
        if (
            type(self.affected_cell_count) is not int
            or type(self.preserved_cell_count) is not int
            or self.affected_cell_count < 0
            or self.preserved_cell_count < 0
            or self.affected_cell_count + self.preserved_cell_count != 216
        ):
            raise W09DimensionalError("W-09 ablation cell accounting drifted")


@dataclass(frozen=True)
class W09DimensionalReport:
    """216-cell public bounded 总验；正式 J-LC 与 mastered 仍禁止发布。"""

    task_audits: tuple[W09CapabilityAudit, ...]
    parent_receipts: tuple[tuple[str, W09PublicParentReceipt], ...]
    cells: tuple[W09CoverageCell, ...]
    cumulative_state_key: tuple[int, ...]
    dimensional_status: str
    j_lc_w09_state: str
    wall_states: tuple[tuple[str, str], ...]
    formal_evidenced: int
    language_capability_mastered: int
    language_readiness: int

    def __post_init__(self) -> None:
        """核验任务、父身份、216 cells、独立链和未正式发布状态。"""
        if tuple(item.task_key for item in self.task_audits) != TASK_KEYS:
            raise W09DimensionalError("W-09 LC-01..16 audit is incomplete")
        actual_parent = tuple(
            (scope, receipt.relative_path, receipt.expected_status)
            for scope, receipt in self.parent_receipts
            if isinstance(receipt, W09PublicParentReceipt)
        )
        if actual_parent != W09_SCOPE_PARENT_RECEIPTS:
            raise W09DimensionalError("W-09 scope parent coverage is incomplete")
        if any(
            not isinstance(receipt, W09PublicParentReceipt)
            for _, receipt in self.parent_receipts
        ):
            raise W09DimensionalError("W-09 scope parent type is invalid")
        expected_cells = tuple(
            (scope, carrier, consumer)
            for scope in SCOPE_KEYS
            for carrier in W09_CARRIER_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        if tuple(item.key for item in self.cells) != expected_cells:
            raise W09DimensionalError("W-09 216-cell coverage is incomplete")
        _strict_key(self.cumulative_state_key, where="cumulative runtime")
        if self.dimensional_status not in W09_DIMENSIONAL_PUBLIC_STATES:
            raise W09DimensionalError("W-09 dimensional public status is invalid")
        if self.j_lc_w09_state not in W09_J_LC_PUBLIC_STATES:
            raise W09DimensionalError("J-LC-W09 public status is invalid")
        if self.wall_states != tuple(
            (key, "NE_WALL") for key in W09_WALL_DIMENSION_KEYS
        ):
            raise W09DimensionalError("W-09 wall dimensions were promoted")
        if any(value != 0 for value in (
            self.formal_evidenced,
            self.language_capability_mastered,
            self.language_readiness,
        )):
            raise W09DimensionalError("W-09-04 published a formal/mastered claim")
        self._validate_result_conjunction()

    def _validate_result_conjunction(self) -> None:
        """要求各类 result identity 独立，并重算 public hard conjunction。"""
        result_identities = tuple(
            identity
            for item in self.cells
            for identity in _directional_component_identities(item.result)
        )
        if len(set(result_identities)) != 216 * 5:
            raise W09DimensionalError("W-09 coverage cells share current result identity")
        continual = tuple(
            item.continual_learning
            for item in self.cells
            if item.continual_learning is not None
        )
        continual_identities = tuple(
            identity
            for item in continual
            for identity in (
                item.before_state_key,
                item.after_state_key,
                item.source_evidence_key,
            )
        )
        if len(set(continual_identities)) != 27 * 3:
            raise W09DimensionalError("W-09 continual cells share learning identity")
        states = tuple(item.result.verifier.status for item in self.cells)
        expected = (
            "PUBLIC_BOUNDED_FAIL"
            if any(item is W09ResultState.FAIL for item in states)
            else "PUBLIC_BOUNDED_NE"
            if any(item is W09ResultState.NE for item in states)
            else "PUBLIC_BOUNDED_PASS"
        )
        j_lc = {
            "PUBLIC_BOUNDED_PASS": "PUBLIC_BOUNDED_NOT_FORMAL",
            "PUBLIC_BOUNDED_FAIL": "PUBLIC_BOUNDED_FAIL",
            "PUBLIC_BOUNDED_NE": "PUBLIC_BOUNDED_NE",
        }[expected]
        if self.dimensional_status != expected or self.j_lc_w09_state != j_lc:
            raise W09DimensionalError("W-09 dimensional status is not a hard conjunction")

    @property
    def retention_cell_count(self) -> int:
        """返回七个历史 scope 的当前 retention cell 数。"""
        return sum(item.evidence_kind == "RETENTION_EVIDENCE" for item in self.cells)

    @property
    def continual_learning_cell_count(self) -> int:
        """返回 W09 本阶段新填的 continual-learning cell 数。"""
        return sum(
            item.evidence_kind == "CONTINUAL_LEARNING_EVIDENCE"
            for item in self.cells
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 J-LC public bounded report 的 canonical identity。"""
        return digest_value({
            "cells": [list(item.stable_key()) for item in self.cells],
            "cumulative": list(self.cumulative_state_key),
            "dimensional_status": self.dimensional_status,
            "formal_evidenced": self.formal_evidenced,
            "j_lc_w09": self.j_lc_w09_state,
            "mastered": self.language_capability_mastered,
            "parents": [
                [scope, list(receipt.stable_key())]
                for scope, receipt in self.parent_receipts
            ],
            "readiness": self.language_readiness,
            "tasks": [
                [item.task_key, item.prior_state, item.current_state]
                for item in self.task_audits
            ],
            "walls": [list(item) for item in self.wall_states],
        })


class W09DimensionalRuntime:
    """登记当前 W09 result，合取 LC-01..16 与八 scope 的 216 cells。"""

    def __init__(
            self,
            repository_root: str | Path,
            cumulative_runtime: W09CumulativeRuntime,
            ) -> None:
        """绑定完整 cumulative runtime、旧 coverage、overlay 和七个 public parent。"""
        if not isinstance(cumulative_runtime, W09CumulativeRuntime):
            raise W09DimensionalError("W-09 cumulative runtime type is invalid")
        cumulative_report = cumulative_runtime.report()
        if not cumulative_report.complete:
            raise W09DimensionalError("W-09 cumulative runtime is incomplete")
        self.repository_root = Path(repository_root).resolve()
        self.cumulative_runtime = cumulative_runtime
        self._overlay, self._coverage = self._read_authorities()
        self.parent_receipts = tuple(
            (
                scope,
                read_w09_public_receipt(
                    self.repository_root,
                    relative_path,
                    expected_status,
                ),
            )
            for scope, relative_path, expected_status in W09_SCOPE_PARENT_RECEIPTS
        )
        self._parent_by_scope = dict(self.parent_receipts)
        self._cells: dict[tuple[str, str, str], W09CoverageCell] = {}
        self._result_identities: set[tuple[int, ...]] = set()

    def _read_authorities(self):
        """严格回读 coverage v2 与 LC-16 overlay，不打开 case payload。"""
        coverage_path = self.repository_root / W09_LC_COVERAGE_PATH
        overlay_path = self.repository_root / W09_LC16_OVERLAY_PATH
        if any(
            not path.is_file() or path.is_symlink()
            for path in (coverage_path, overlay_path)
        ):
            raise W09DimensionalError("W-09 LC authority is missing or linked")
        try:
            coverage = read_language_capability_coverage_v2(coverage_path)
            overlay = read_d03_lc16_successor_overlay(overlay_path)
        except (LanguageCoverageV2Error, D03Lc16OverlayError) as error:
            raise W09DimensionalError("W-09 LC authority is invalid") from error
        if tuple(item.scope_key for item in overlay.scope_records) != SCOPE_KEYS:
            raise W09DimensionalError("W-09 LC scope order drifted")
        if tuple(item.task_key for item in coverage.task_records) != TASK_KEYS:
            raise W09DimensionalError("W-09 LC task order drifted")
        return overlay, coverage

    def record_cell(
            self,
            scope_key: str,
            carrier_key: str,
            consumer_key: str,
            result: W09DirectionalResult,
            *,
            continual_learning: W09ContinualLearningEvidence | None = None,
            ) -> W09CoverageCell:
        """登记一个独立当前 result，并绑定 parent retention 或 W09 学习 delta。"""
        key = (scope_key, carrier_key, consumer_key)
        if key in self._cells:
            raise W09DimensionalError("W-09 coverage cell was replayed")
        parent = self._parent_by_scope.get(scope_key)
        cell = W09CoverageCell(
            scope_key,
            carrier_key,
            consumer_key,
            (
                "CONTINUAL_LEARNING_EVIDENCE"
                if scope_key == W09_RETENTION_SCOPE_KEY
                else "RETENTION_EVIDENCE"
            ),
            None if parent is None else parent.sha256,
            result,
            continual_learning,
        )
        identities = _directional_component_identities(result)
        if any(value in self._result_identities for value in identities):
            raise W09DimensionalError("W-09 coverage result identity was reused")
        self._result_identities.update(identities)
        self._cells[key] = cell
        return cell

    def report(self) -> W09DimensionalReport:
        """完成 216-cell hard conjunction，并保持 J-LC 仅为 public bounded。"""
        expected = tuple(
            (scope, carrier, consumer)
            for scope in SCOPE_KEYS
            for carrier in W09_CARRIER_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        if len(self._cells) != 216 or set(self._cells) != set(expected):
            raise W09DimensionalError("W-09 216-cell runtime is incomplete")
        cells = tuple(self._cells[key] for key in expected)
        states = tuple(item.result.verifier.status for item in cells)
        dimensional_status = (
            "PUBLIC_BOUNDED_FAIL"
            if any(item is W09ResultState.FAIL for item in states)
            else "PUBLIC_BOUNDED_NE"
            if any(item is W09ResultState.NE for item in states)
            else "PUBLIC_BOUNDED_PASS"
        )
        j_lc_state = {
            "PUBLIC_BOUNDED_PASS": "PUBLIC_BOUNDED_NOT_FORMAL",
            "PUBLIC_BOUNDED_FAIL": "PUBLIC_BOUNDED_FAIL",
            "PUBLIC_BOUNDED_NE": "PUBLIC_BOUNDED_NE",
        }[dimensional_status]
        task_audits = tuple(
            W09CapabilityAudit(
                item.task_key,
                item.baseline_state,
                "CURRENT_INTERFACE_AUDITED",
            )
            for item in self._coverage.task_records
        )
        return W09DimensionalReport(
            task_audits,
            self.parent_receipts,
            cells,
            self.cumulative_runtime.state_key(),
            dimensional_status,
            j_lc_state,
            tuple((key, "NE_WALL") for key in W09_WALL_DIMENSION_KEYS),
            0,
            0,
            0,
        )

    @staticmethod
    def ablate_aggregator(report: W09DimensionalReport) -> W09DimensionalAblation:
        """关闭累计 aggregator，只击穿 DIMENSIONAL_PASS 并保留 216 cells。"""
        if not isinstance(report, W09DimensionalReport):
            raise W09DimensionalError("W-09 dimensional report type is invalid")
        return W09DimensionalAblation(
            "DIMENSIONAL_AGGREGATOR",
            W09_DIMENSION_KEYS[0],
            "FAIL",
            0,
            216,
            0,
        )

    @staticmethod
    def ablate_cell(
            report: W09DimensionalReport,
            scope_key: str,
            carrier_key: str,
            consumer_key: str,
            ) -> W09DimensionalAblation:
        """关闭一个指定 consumer/verifier cell，只计该格为承重击穿。"""
        if not isinstance(report, W09DimensionalReport):
            raise W09DimensionalError("W-09 dimensional report type is invalid")
        key = (scope_key, carrier_key, consumer_key)
        if key not in {item.key for item in report.cells}:
            raise W09DimensionalError("W-09 ablation cell is not registered")
        return W09DimensionalAblation(
            ":".join(key),
            W09_DIMENSION_KEYS[0],
            "FAIL",
            1,
            215,
            0,
        )


def open_w09_dimensional_runtime(
        repository_root: str | Path,
        cumulative_runtime: W09CumulativeRuntime,
        ) -> W09DimensionalRuntime:
    """打开不消费 formal Candidate/private guard 的 W09-04 runtime。"""
    try:
        return W09DimensionalRuntime(repository_root, cumulative_runtime)
    except W09CumulativeError as error:
        raise W09DimensionalError("W-09 cumulative parent is invalid") from error


__all__ = [
    "W09_DIMENSIONAL_PUBLIC_STATES",
    "W09_J_LC_PUBLIC_STATES",
    "W09_RETENTION_SCOPE_KEY",
    "W09_SCOPE_PARENT_RECEIPTS",
    "W09CapabilityAudit",
    "W09ContinualLearningEvidence",
    "W09CoverageCell",
    "W09DimensionalAblation",
    "W09DimensionalError",
    "W09DimensionalReport",
    "W09DimensionalRuntime",
    "open_w09_dimensional_runtime",
]
