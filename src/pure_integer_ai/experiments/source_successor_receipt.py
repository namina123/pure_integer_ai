"""固定 seed Hasher 后继源码线的非 readiness successor receipt。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_SOURCE_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "SOURCE-SUCCESSOR-HASHER-20260806-A"
RECEIPT_PATH = "data/ph2/manifests/source_successor_receipt_v1.json"
PARENT_COMMIT = "e8c86f996da1983eb92844558159e49fd4a30135"
PARENT_SEAL_SHA256 = "88f03b9c4b7a110ac031e99ae6f0d1bcc54b0523d6f35903628c65190c7d516f"
PARENT_CORE_SHA256 = "d68e8e27f3d0cfe0632f3d51ff56adfe0087b1546a1a58fc5fe6f5062e5e6759"

_PRODUCTION_FILES = (
    ("src/pure_integer_ai/cognition/understanding/observe.py", 44833,
     "17ea60754dbd47ff71bf6e5aa4ec7245c2d0f62995f25ea4d6dc353a9cb1cf4d"),
    ("src/pure_integer_ai/experiments/arithmetic_structure_runtime.py", 14480,
     "04a39143c337a34d1a8b3416ac71d8c003f9b329cc85f438b99d0c4b56be9fad"),
    ("src/pure_integer_ai/experiments/collection.py", 102706,
     "bfee7d3a153358c6204337bf2e544346b50b0b7be5e83a3209ea73ab6e585701"),
    ("src/pure_integer_ai/experiments/evaluation_runtime.py", 54593,
     "68ce94a3ef7dca547a632b6d6be2d6fb7b50aa861290555e1cfaad7d62ba1cec"),
    ("src/pure_integer_ai/experiments/language_structure_runtime.py", 31643,
     "e54c6bd4bc596e6134104bec7d3321f25c25f39895670571ce4733a755aec059"),
    ("src/pure_integer_ai/experiments/round_runtime.py", 96798,
     "58f1a4f68a0f36fdd3027771e33a2e91597484796c80e76fa8ab2b6723e84783"),
)
_SOURCE_PATHS = tuple(item[0] for item in _PRODUCTION_FILES)


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} 字段不精确")
    return value


def _sha256(value: str, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{where} SHA-256 非法")
    return value


def _sha1(value: str, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{where} SHA-1 非法")
    return value


def _relative(value: str, *, where: str) -> str:
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise ValueError(f"{where} 相对路径非法")
    return value


def _identity(root: Path, relative_path: str) -> tuple[int, str]:
    _relative(relative_path, where="源码 identity")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"源码文件缺失: {relative_path}")
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _parent_binding(path: str, size: int, digest: str) -> dict[str, Any]:
    return {
        "current_sha256": "",
        "current_size_bytes": 0,
        "parent_sha256": digest,
        "parent_size_bytes": size,
        "relative_path": path,
    }


def build_source_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """形成当前后继源码身份，不写 readiness 或覆盖任何历史 artifact。"""
    root = Path(repository_root).resolve()
    bindings = []
    for path, parent_size, parent_sha in _PRODUCTION_FILES:
        current_size, current_sha = _identity(root, path)
        item = _parent_binding(path, parent_size, parent_sha)
        item["current_size_bytes"] = current_size
        item["current_sha256"] = current_sha
        bindings.append(item)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "format_version": FORMAT_VERSION,
        "parent_commit": PARENT_COMMIT,
        "parent_core_manifest": {
            "relative_path": "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json",
            "sha256": PARENT_CORE_SHA256,
            "status": "HISTORICAL_ONLY",
        },
        "parent_seal": {
            "relative_path": "data/ph2/manifests/j_f2_final_joint_seal_v1.json",
            "sha256": PARENT_SEAL_SHA256,
            "status": "HISTORICAL_ONLY",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": bindings,
        "status": "SOURCE_SUCCESSOR_EVIDENCED",
        "transformation": {
            "algorithm_changed": 0,
            "bit_identical_hasher_cases": 11,
            "canonical_encoder_changed": 0,
            "fixed_seed_hot_loop_calls_after": 0,
            "fixed_seed_hot_loop_calls_before": 13,
            "seed_changed": 0,
            "seed_encode_count_after": 1,
            "seed_encode_count_before": 32,
        },
    }


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "format_version",
        "parent_commit", "parent_core_manifest", "parent_seal",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation",
    }, where="source successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != PARENT_COMMIT
            or raw["status"] != "SOURCE_SUCCESSOR_EVIDENCED"
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1):
        raise ValueError("source successor receipt 固定身份漂移")
    for key, expected_path, expected_sha in (
            ("parent_seal", "data/ph2/manifests/j_f2_final_joint_seal_v1.json",
             PARENT_SEAL_SHA256),
            ("parent_core_manifest", "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json",
             PARENT_CORE_SHA256)):
        item = _exact(raw[key], {"relative_path", "sha256", "status"}, where=key)
        if (item["relative_path"] != expected_path
                or item["sha256"] != expected_sha
                or item["status"] != "HISTORICAL_ONLY"):
            raise ValueError(f"{key} historical identity 漂移")
        _sha256(item["sha256"], where=key)
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="readiness_transition")
    if transition != {"LANGUAGE_READINESS_REPUBLISHED": 0, "PW00A_STARTED": 0}:
        raise ValueError("successor receipt 不得转移 readiness")
    bindings = raw["source_bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(_PRODUCTION_FILES):
        raise ValueError("source successor 文件数量漂移")
    expected = {path: (size, digest) for path, size, digest in _PRODUCTION_FILES}
    previous = ""
    for item in bindings:
        entry = _exact(item, {
            "current_sha256", "current_size_bytes", "parent_sha256",
            "parent_size_bytes", "relative_path",
        }, where="source binding")
        path = entry["relative_path"]
        if not isinstance(path, str) or path <= previous or path not in expected:
            raise ValueError("source successor 路径未排序或不在 allowlist")
        parent_size, parent_sha = expected[path]
        if (entry["parent_size_bytes"] != parent_size
                or entry["parent_sha256"] != parent_sha
                or type(entry["current_size_bytes"]) is not int
                or entry["current_size_bytes"] < 1):
            raise ValueError(f"source parent identity 漂移: {path}")
        _sha256(entry["current_sha256"], where=path)
        _sha256(entry["parent_sha256"], where=path)
        previous = path
    transform = _exact(raw["transformation"], {
        "algorithm_changed", "bit_identical_hasher_cases",
        "canonical_encoder_changed", "fixed_seed_hot_loop_calls_after",
        "fixed_seed_hot_loop_calls_before", "seed_changed",
        "seed_encode_count_after", "seed_encode_count_before",
    }, where="transformation")
    if transform != {
            "algorithm_changed": 0,
            "bit_identical_hasher_cases": 11,
            "canonical_encoder_changed": 0,
            "fixed_seed_hot_loop_calls_after": 0,
            "fixed_seed_hot_loop_calls_before": 13,
            "seed_changed": 0,
            "seed_encode_count_after": 1,
            "seed_encode_count_before": 32,
    }:
        raise ValueError("Hasher successor transformation 不是已验证的等价替换")


def read_source_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("source successor receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("source successor receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("source successor receipt canonical bytes 漂移")
    _validate(value)
    for item in value["source_bindings"]:
        current = _identity(root, item["relative_path"])
        if current != (item["current_size_bytes"], item["current_sha256"]):
            raise ValueError(f"source successor 当前 identity 漂移: {item['relative_path']}")
    return value


def publish_source_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("source successor receipt 已发布，禁止覆盖")
    value = build_source_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("source successor receipt 已发布，禁止覆盖") from error
    restored = read_source_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("source successor receipt 发布回读不一致")
    return restored


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "FORMAT_VERSION", "PARENT_COMMIT",
    "PARENT_CORE_SHA256", "PARENT_SEAL_SHA256", "RECEIPT_PATH",
    "build_source_successor_receipt", "publish_source_successor_receipt",
    "read_source_successor_receipt",
]
