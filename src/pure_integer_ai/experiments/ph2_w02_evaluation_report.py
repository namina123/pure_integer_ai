"""W-02 首轮 private evaluator 公开摘要的不可覆盖持久化。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w02_evaluation_contract import (
    W02PrivateEvaluationReport,
)


_FORBIDDEN_PRIVATE_KEYS = frozenset({
    "accepted_surfaces",
    "evaluator_label_payload",
    "expected_payload",
    "expected_surface",
    "raw_observation",
})


class W02EvaluationPublicationError(RuntimeError):
    """W-02 evaluator 摘要泄漏、非规范或覆盖发布。"""


@dataclass(frozen=True)
class W02EvaluationPublication:
    """首轮 evaluator 公开摘要的物理身份。"""

    path: Path
    sha256: str
    size_bytes: int
    status: str


def _reject_private_keys(value: Any) -> None:
    """递归拒绝可携带 private expected、surface 或 Observation 的字段。"""
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_PRIVATE_KEYS.intersection(value)
        if forbidden:
            raise W02EvaluationPublicationError(
                f"公开 evaluator 摘要含 private 字段: {sorted(forbidden)!r}")
        for item in value.values():
            _reject_private_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_keys(item)


def _publication(path: Path, payload: bytes, value: dict[str, Any]) -> W02EvaluationPublication:
    """由已核 canonical bytes 构造公开摘要身份。"""
    status = value.get("status")
    if status not in {"PASS", "FAIL", "NE"}:
        raise W02EvaluationPublicationError("公开 evaluator 摘要状态非法")
    return W02EvaluationPublication(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        status=status,
    )


def publish_w02_private_evaluation_report(
        report_path: str | Path,
        report: W02PrivateEvaluationReport,
        ) -> W02EvaluationPublication:
    """排他创建并刷盘首轮 public dict；既存目标无论内容如何均拒绝覆盖。"""
    if not isinstance(report, W02PrivateEvaluationReport):
        raise TypeError("report 必须是 W02PrivateEvaluationReport")
    path = Path(report_path).resolve()
    if not path.parent.is_dir():
        raise W02EvaluationPublicationError("evaluator report 父目录不存在")
    value = report.to_public_dict()
    _reject_private_keys(value)
    payload = canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise W02EvaluationPublicationError(
            "首轮 evaluator report 已存在，禁止覆盖或重跑") from exc
    return _publication(path, payload, value)


def read_w02_private_evaluation_report(
        report_path: str | Path,
        ) -> tuple[dict[str, Any], W02EvaluationPublication]:
    """严格回读 canonical public dict，并重验私有字段边界与物理身份。"""
    path = Path(report_path).resolve()
    try:
        payload = path.read_bytes()
        value = parse_canonical_json_bytes(payload, require_object=True)
    except (OSError, TypeError, ValueError) as exc:
        raise W02EvaluationPublicationError(
            "首轮 evaluator report 无法规范回读") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W02EvaluationPublicationError("首轮 evaluator report 非 canonical bytes")
    _reject_private_keys(value)
    return value, _publication(path, payload, value)


__all__ = [
    "W02EvaluationPublication",
    "W02EvaluationPublicationError",
    "publish_w02_private_evaluation_report",
    "read_w02_private_evaluation_report",
]
