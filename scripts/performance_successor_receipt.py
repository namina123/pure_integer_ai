"""发布规范整数流长度直算的非 readiness 性能 successor receipt。"""
from __future__ import annotations

import argparse
import hashlib
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
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-INTEGER-SIZE-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v1.json"
PARENT_COMMIT = "ef3f38ce67d709dcce4546733e8515c71aff7750"
CHANGE_COMMIT = "0bb9a43256ed429a61001c3861c17260028675e8"
PRIOR_RECEIPT_PATH = (
    "data/ph2/manifests/struct_layout_successor_receipt_v2.json"
)
PRIOR_RECEIPT_SHA256 = (
    "1047e03de245d89ce07a8e8b35119b1a38ebc79490ddb2b2e7d2239920e6733f"
)
_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "78adea2f14dd55284a4302599927bfb4ba1dd83e0562550a7f7506fe2a4a1dcb"
        ),
        "current_size_bytes": 7461,
        "parent_sha256": (
            "1b70cbb9e4e84a19d3bd30c695b1a9a7b55e0c95208098687413b94baa1a0c8f"
        ),
        "parent_size_bytes": 5677,
        "relative_path": "src/pure_integer_ai/storage/integer_codec.py",
    },
    {
        "current_sha256": (
            "9398daf7661e2546bd7286a5b625872a8b9dfd1583c7296e4dc3f75b87434b34"
        ),
        "current_size_bytes": 11321,
        "parent_sha256": (
            "77f3b3390ba48a07b2524573e67442838e0de2939b4fd0709d88157868b854d5"
        ),
        "parent_size_bytes": 11229,
        "relative_path": "src/pure_integer_ai/storage/sealed_segment.py",
    },
]
_TRANSFORMATION = {
    "algorithm": "ENCODE_THEN_LEN_TO_DIRECT_VARINT_SIZE",
    "cache_cycle": {
        "delta_per_mille": -74,
        "new_median_ns": 118303200,
        "old_median_ns": 127691900,
        "record_count": 5000,
        "repeats": 11,
    },
    "cached_segment_record_slots_decision": {
        "cache_cycle_delta_per_mille": 31,
        "decision": "REJECTED_AND_REVERTED",
        "live_bytes_after_probe": 8001608,
        "live_bytes_before": 12001608,
    },
    "canonical_decoder_changed": 0,
    "canonical_encoder_changed": 0,
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
    "segment_record_fields_changed": 0,
}
_VERIFICATION = {
    "bounded_test_runs": [
        {"passed": 66, "scope": "CODEC_K02_R02"},
        {"passed": 83, "scope": "CODEC_K03_K04_W08_LONG_CONTEXT"},
    ],
    "codec_size_tests_passed": 17,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
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


def build_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造固定性能后继证据，不写 readiness 或文件。"""
    root = Path(repository_root).resolve()
    for binding in _SOURCE_BINDINGS:
        current = _identity(root, binding["relative_path"])
        if current != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 struct layout receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "change_commit": CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "STRUCT_LAYOUT_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_TRANSFORMATION),
        "verification": deepcopy(_VERIFICATION),
    }


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["change_commit"] != CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != PARENT_COMMIT
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="change_commit")
    _digest(raw["parent_commit"], length=40, where="parent_commit")

    if raw["prior_successor_receipt"] != {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "STRUCT_LAYOUT_PREDECESSOR",
    }:
        raise ValueError("前序 struct layout receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("performance successor 不得转移 readiness")
    if raw["source_bindings"] != _SOURCE_BINDINGS:
        raise ValueError("性能后继源码绑定漂移")
    previous = ""
    for binding in raw["source_bindings"]:
        path = binding["relative_path"]
        if not isinstance(path, str) or path <= previous:
            raise ValueError("性能后继源码路径未严格排序")
        _digest(binding["current_sha256"], length=64, where=path)
        _digest(binding["parent_sha256"], length=64, where=path)
        previous = path
    if raw["transformation"] != _TRANSFORMATION:
        raise ValueError("性能 transformation 证据漂移")
    if raw["verification"] != _VERIFICATION:
        raise ValueError("性能 verification 证据漂移")


def read_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("performance receipt canonical bytes 漂移")
    _validate(value)
    for binding in value["source_bindings"]:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"性能后继源码当前 identity 漂移: {binding['relative_path']}")
    if _identity(root, PRIOR_RECEIPT_PATH)[1] != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 struct layout receipt 当前 identity 漂移")
    return value


def publish_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("performance successor receipt 已发布，禁止覆盖")
    value = build_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("performance successor receipt 已发布，禁止覆盖") from error
    restored = read_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("performance successor receipt 发布回读不一致")
    return restored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布性能 successor receipt。")
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
            value = publish_performance_successor_receipt(
                REPOSITORY_ROOT, target=args.target)
        else:
            value = build_performance_successor_receipt(REPOSITORY_ROOT)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"performance_successor_receipt: ERROR: {error}")
        return 1
    payload = canonical_json_bytes(value) + b"\n"
    print(
        "performance_successor_receipt: ready "
        f"(sha256={hashlib.sha256(payload).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "CHANGE_COMMIT", "FORMAT_VERSION",
    "PARENT_COMMIT", "RECEIPT_PATH", "build_performance_successor_receipt",
    "publish_performance_successor_receipt",
    "read_performance_successor_receipt",
]
