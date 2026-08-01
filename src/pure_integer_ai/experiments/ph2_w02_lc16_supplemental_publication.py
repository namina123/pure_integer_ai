"""W-02 LC-16 supplemental receipt 的公开、不可覆盖发布接口。"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_contract import (
    ARTIFACT_KIND,
    W02Lc16SupplementalReport,
)


RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json")


class W02Lc16SupplementalPublicationError(RuntimeError):
    """supplemental receipt 泄漏、损坏或覆盖发布。"""


@dataclass(frozen=True)
class W02Lc16SupplementalPublication:
    """已发布 receipt 的物理路径、摘要、大小和状态。"""

    path: Path
    sha256: str
    size_bytes: int
    status: str


def _reject_private(value: Any) -> None:
    """递归拒绝 expected/surface/Observation 等 private 字段。"""
    forbidden = {
        "accepted_surfaces", "expected_payload", "expected_surface",
        "evaluator_label_payload", "raw_observation", "private_payload",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise W02Lc16SupplementalPublicationError(
                "supplemental receipt 含 private 字段")
        for item in value.values():
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)


def _publication(path: Path, payload: bytes, value: dict[str, Any]) -> W02Lc16SupplementalPublication:
    """由 canonical bytes 构造公开 publication identity。"""
    status = value.get("status")
    if status not in {"PASS", "FAIL", "NE", "BLOCKED"}:
        raise W02Lc16SupplementalPublicationError("supplemental receipt status 非法")
    return W02Lc16SupplementalPublication(
        path,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        status,
    )


def publish_w02_lc16_supplemental_report(
        path: str | Path,
        report: W02Lc16SupplementalReport,
        ) -> W02Lc16SupplementalPublication:
    """排他创建 supplemental receipt，既存目标一律拒绝覆盖。"""
    if not isinstance(report, W02Lc16SupplementalReport):
        raise TypeError("report 类型非法")
    target = Path(path).resolve()
    if not target.parent.is_dir():
        raise W02Lc16SupplementalPublicationError("receipt 父目录不存在")
    value = report.to_public_dict()
    _reject_private(value)
    if value.get("artifact_kind") != ARTIFACT_KIND:
        raise W02Lc16SupplementalPublicationError("receipt artifact kind 非法")
    payload = report.canonical_bytes()
    try:
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise W02Lc16SupplementalPublicationError(
            "supplemental receipt 已存在，禁止覆盖或重跑") from error
    except OSError as error:
        raise W02Lc16SupplementalPublicationError(
            "supplemental receipt 无法发布") from error
    return _publication(target, payload, value)


def read_w02_lc16_supplemental_report(
        path: str | Path,
        ) -> tuple[dict[str, Any], W02Lc16SupplementalPublication]:
    """严格回读 canonical supplemental receipt 并重验私有边界。"""
    target = Path(path).resolve()
    try:
        payload = target.read_bytes()
        value = parse_canonical_json_bytes(payload[:-1], require_object=True) \
            if payload.endswith(b"\n") else None
    except (OSError, TypeError, ValueError) as error:
        raise W02Lc16SupplementalPublicationError(
            "supplemental receipt 无法回读") from error
    if value is None or canonical_json_bytes(value) + b"\n" != payload:
        raise W02Lc16SupplementalPublicationError(
            "supplemental receipt 非 canonical bytes")
    _reject_private(value)
    if value.get("artifact_kind") != ARTIFACT_KIND:
        raise W02Lc16SupplementalPublicationError("supplemental receipt kind 非法")
    return value, _publication(target, payload, value)


__all__ = [
    "RECEIPT_RELATIVE_PATH", "W02Lc16SupplementalPublication",
    "W02Lc16SupplementalPublicationError",
    "publish_w02_lc16_supplemental_report",
    "read_w02_lc16_supplemental_report",
]
