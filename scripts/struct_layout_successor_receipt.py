"""发布首批结构体布局优化的非 readiness successor receipt。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_STRUCT_LAYOUT_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "STRUCT-LAYOUT-SUCCESSOR-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/struct_layout_successor_receipt_v1.json"
PARENT_COMMIT = "5ce6874ebcf3498454d395be7fd3d8633c66d047"
CHANGE_COMMIT = "c6799cbd4db694ac4c1a8c23d33e1fe0e04abbdd"
PRIOR_RECEIPT_PATH = "data/ph2/manifests/source_successor_receipt_v1.json"
PRIOR_RECEIPT_SHA256 = (
    "6b4d042bb82f5ec5f467a0d253254bb13e252d339b2294c3d234e38b9ff5977e"
)
SOURCE_PATH = "src/pure_integer_ai/crosscut/integer/valtypes.py"
PARENT_SIZE = 2562
PARENT_SHA256 = "fb195983c9a2de91a037cfc2019d9758a643f4f2a76da4271fa67b8333cfb9f4"
CURRENT_SIZE = 2710
CURRENT_SHA256 = "81ea5665fb31f68dd99efbd76129adce6fb22dda03f03939eddc1e65d3b6ec4a"

_STRUCTURE_CHANGES = [
    {
        "class_id": "crosscut/integer/valtypes.py::FixedQuotient",
        "field_order_changed": 0,
        "frozen_after": 1,
        "frozen_before": 1,
        "instance_dict_after": 0,
        "instance_dict_before": 1,
        "interop": "pending",
        "representation": "struct",
        "slots_after": 1,
        "slots_before": 0,
        "validation_changed": 0,
    },
    {
        "class_id": "crosscut/integer/valtypes.py::Rational",
        "field_order_changed": 0,
        "frozen_after": 1,
        "frozen_before": 1,
        "instance_dict_after": 0,
        "instance_dict_before": 1,
        "interop": "pending",
        "representation": "struct",
        "slots_after": 1,
        "slots_before": 0,
        "validation_changed": 0,
    },
]
_BENCHMARK_EVIDENCE = {
    "construction_count_per_round": 300000,
    "construction_repeats": 11,
    "environment_scope": "CPYTHON-3.14-WINDOWS-LOCAL-DIRECTIONAL",
    "live_instance_count": 100000,
    "objects": [
        {
            "class_name": "FixedQuotient",
            "construction_delta_per_mille": -15,
            "live_bytes_after": 7201392,
            "live_bytes_before": 11201392,
            "live_bytes_delta_per_mille": -357,
        },
        {
            "class_name": "Rational",
            "construction_delta_per_mille": -26,
            "live_bytes_after": 5601264,
            "live_bytes_before": 9601264,
            "live_bytes_delta_per_mille": -417,
        },
    ],
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
}
_VERIFICATION = {
    "canonical_encoder_changed": 0,
    "compatibility_tests_passed": 12,
    "direct_test_files": 19,
    "direct_tests_passed": 617,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "pickle_bit_identical": 0,
    "pickle_roundtrip_cases": 2,
    "python311_object_model_guard_clean": 1,
    "python314_object_model_guard_clean": 1,
    "source_guards_clean": 1,
}


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} 字段不精确")
    return value


def _digest(value: str, *, length: int, where: str) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{where} 摘要非法")
    return value


def _relative(value: str, *, where: str) -> str:
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise ValueError(f"{where} 相对路径非法")
    return value


def _identity(root: Path, relative_path: str) -> tuple[int, str]:
    _relative(relative_path, where="文件 identity")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"文件缺失: {relative_path}")
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def build_struct_layout_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造已验证布局优化的固定 receipt，不写文件。"""
    root = Path(repository_root).resolve()
    current_size, current_sha = _identity(root, SOURCE_PATH)
    if (current_size, current_sha) != (CURRENT_SIZE, CURRENT_SHA256):
        raise ValueError("结构体后继源码 identity 漂移")
    prior_size, prior_sha = _identity(root, PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 source successor receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "benchmark_evidence": deepcopy(_BENCHMARK_EVIDENCE),
        "change_commit": CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "HISTORICAL_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_binding": {
            "current_sha256": CURRENT_SHA256,
            "current_size_bytes": CURRENT_SIZE,
            "parent_sha256": PARENT_SHA256,
            "parent_size_bytes": PARENT_SIZE,
            "relative_path": SOURCE_PATH,
        },
        "status": "STRUCT_LAYOUT_SUCCESSOR_EVIDENCED",
        "structure_changes": deepcopy(_STRUCTURE_CHANGES),
        "verification": deepcopy(_VERIFICATION),
    }


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "benchmark_evidence",
        "change_commit", "format_version", "parent_commit",
        "prior_successor_receipt", "readiness_transition",
        "receipt_relative_path", "receipt_self_excluded", "source_binding",
        "status", "structure_changes", "verification",
    }, where="struct layout successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["format_version"] != FORMAT_VERSION
            or raw["change_commit"] != CHANGE_COMMIT
            or raw["parent_commit"] != PARENT_COMMIT
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "STRUCT_LAYOUT_SUCCESSOR_EVIDENCED"):
        raise ValueError("struct layout successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="change_commit")
    _digest(raw["parent_commit"], length=40, where="parent_commit")

    prior = _exact(raw["prior_successor_receipt"], {
        "relative_path", "sha256", "status",
    }, where="prior_successor_receipt")
    if prior != {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "HISTORICAL_PREDECESSOR",
    }:
        raise ValueError("前序 source successor receipt 声明漂移")
    _digest(prior["sha256"], length=64, where="prior receipt")

    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("struct layout successor 不得转移 readiness")

    binding = _exact(raw["source_binding"], {
        "current_sha256", "current_size_bytes", "parent_sha256",
        "parent_size_bytes", "relative_path",
    }, where="source_binding")
    if binding != {
            "current_sha256": CURRENT_SHA256,
            "current_size_bytes": CURRENT_SIZE,
            "parent_sha256": PARENT_SHA256,
            "parent_size_bytes": PARENT_SIZE,
            "relative_path": SOURCE_PATH,
    }:
        raise ValueError("struct layout source binding 漂移")
    _digest(binding["current_sha256"], length=64, where="current source")
    _digest(binding["parent_sha256"], length=64, where="parent source")
    if raw["structure_changes"] != _STRUCTURE_CHANGES:
        raise ValueError("结构体变更台账漂移")
    if raw["benchmark_evidence"] != _BENCHMARK_EVIDENCE:
        raise ValueError("结构体性能证据漂移")
    if raw["verification"] != _VERIFICATION:
        raise ValueError("结构体验证证据漂移")


def read_struct_layout_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("struct layout receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("struct layout receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("struct layout receipt canonical bytes 漂移")
    _validate(value)
    if _identity(root, SOURCE_PATH) != (CURRENT_SIZE, CURRENT_SHA256):
        raise ValueError("结构体后继源码当前 identity 漂移")
    if _identity(root, PRIOR_RECEIPT_PATH)[1] != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 source successor receipt 当前 identity 漂移")
    return value


def publish_struct_layout_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("struct layout successor receipt 已发布，禁止覆盖")
    value = build_struct_layout_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("struct layout successor receipt 已发布，禁止覆盖") from error
    restored = read_struct_layout_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("struct layout successor receipt 发布回读不一致")
    return restored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布结构体布局 successor receipt。")
    parser.add_argument("--publish", action="store_true", help="独占发布正式 receipt")
    parser.add_argument("--target", type=Path, default=Path(RECEIPT_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    try:
        if args.publish:
            value = publish_struct_layout_successor_receipt(
                REPOSITORY_ROOT, target=args.target)
        else:
            value = build_struct_layout_successor_receipt(REPOSITORY_ROOT)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"struct_layout_successor_receipt: ERROR: {error}")
        return 1
    payload = canonical_json_bytes(value) + b"\n"
    print(
        "struct_layout_successor_receipt: ready "
        f"(sha256={hashlib.sha256(payload).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "CHANGE_COMMIT", "FORMAT_VERSION",
    "PARENT_COMMIT", "RECEIPT_PATH", "build_struct_layout_successor_receipt",
    "publish_struct_layout_successor_receipt",
    "read_struct_layout_successor_receipt",
]
