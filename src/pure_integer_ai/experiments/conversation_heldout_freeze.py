"""DLG-05 公开 preflight 冻结清单与无标签 receipt 序列化。

该模块生成的是公开审计/report artifact，不是 Core、Memory 或会话存储。
冻结内容只保留完整对象的稳定整数键、CanonicalIdentity 的 hash/长度以及
文件 SHA-256；不写入 evaluator label、private 数据或问题表面文本。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutInputCatalog,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    ConversationHeldOutManifest,
)
from pure_integer_ai.experiments.conversation_heldout_qualification import (
    ConversationHeldOutQualificationReceipt,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)


class ConversationHeldOutFreezeError(RuntimeError):
    """DLG-05 冻结清单或不可覆盖写入不闭合。"""


_TEST_IMPORT_RE = re.compile(
    r"\b(?:from|import)\s+(test_[A-Za-z0-9_]+)",
)
_DLG05_TEST_NAMES = (
    "test_ph2_conversation_heldout_family.py",
    "test_ph2_conversation_heldout_protocol.py",
    "test_ph2_conversation_heldout_qualification.py",
    "test_ph2_conversation_heldout_runtime.py",
)


def _key(value: ProtocolKey | tuple[int, ...]) -> list[int]:
    """把协议键或稳定整数 tuple 转为 JSON 整数数组。"""
    components = value.components if isinstance(value, ProtocolKey) else value
    if (not isinstance(components, tuple) or not components
            or any(type(item) is not int or item < 0 for item in components)):
        raise ConversationHeldOutFreezeError("冻结对象键必须是非空非负整数 tuple")
    return list(components)


def _identity(value: CanonicalIdentity) -> dict[str, Any]:
    """保留 CanonicalIdentity 的内容身份，不复制表面载荷。"""
    if not isinstance(value, CanonicalIdentity):
        raise TypeError("冻结 identity 类型错误")
    return {
        "payload_sha256": value.sha256,
        "payload_size": len(value.payload),
        "index": value.index,
    }


def _sha256_bytes(payload: bytes) -> str:
    """计算文件/规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _manifest_dict(manifest: ConversationHeldOutManifest) -> dict[str, Any]:
    """序列化无标签 manifest 的完整输入身份。"""
    return {
        "version": manifest.version,
        "family_key": _key(manifest.family_key),
        "train_contents": [_identity(item) for item in manifest.train_contents],
        "train_dedup_clusters": [
            _identity(item) for item in manifest.train_dedup_clusters],
        "train_provenance_clusters": [
            _identity(item) for item in manifest.train_provenance_clusters],
        "required_axes": [_key(item) for item in manifest.required_axes],
        "required_memory_modes": [
            _key(item) for item in manifest.required_memory_modes],
        "cases": [
            {
                "case_key": _key(case.case_key),
                "axis_keys": [_key(item) for item in case.axis_keys],
                "dedup_cluster": _identity(case.dedup_cluster),
                "provenance_cluster": _identity(case.provenance_cluster),
                "turns": [
                    {
                        "turn_key": _key(turn.turn_key),
                        "ordinal": turn.ordinal,
                        "content": _identity(turn.content),
                        "source_key": _key(turn.source_key),
                        "scope_key": _key(turn.scope_key),
                        "context_mode": _key(turn.context_mode),
                        "reference_mode": _key(turn.reference_mode),
                        "memory_mode": _key(turn.memory_mode),
                        "rollback_mode": _key(turn.rollback_mode),
                    }
                    for turn in case.turns
                ],
            }
            for case in manifest.cases
        ],
    }


def _catalog_dict(catalog: ConversationHeldOutInputCatalog) -> dict[str, Any]:
    """序列化 typed catalog 的可重建身份和 request 键。"""
    return {
        "version": catalog.version,
        "family_key": _key(catalog.family_key),
        "train_contents": [_identity(item) for item in catalog.train_contents],
        "train_dedup_clusters": [
            _identity(item) for item in catalog.train_dedup_clusters],
        "train_provenance_clusters": [
            _identity(item) for item in catalog.train_provenance_clusters],
        "cases": [
            {
                "case_key": _key(case.case_key),
                "axis_keys": [_key(item) for item in case.axis_keys],
                "dedup_cluster": _identity(case.dedup_cluster),
                "provenance_cluster": _identity(case.provenance_cluster),
                "turns": [
                    {
                        "turn_key": _key(turn.turn_key),
                        "ordinal": turn.ordinal,
                        "request_key": list(turn.request.stable_key()),
                        "content": _identity(turn.content),
                        "source_key": _key(turn.source_key),
                        "scope_key": _key(turn.scope_key),
                        "context_mode": _key(turn.context_mode),
                        "reference_mode": _key(turn.reference_mode),
                        "memory_mode": _key(turn.memory_mode),
                        "rollback_mode": _key(turn.rollback_mode),
                    }
                    for turn in case.turns
                ],
            }
            for case in catalog.cases
        ],
    }


def _execution_dict(value: Any) -> dict[str, Any]:
    """序列化无标签 selection-first execution receipt。"""
    return {
        "manifest_key": list(value.manifest_key),
        "contract_key": list(value.contract_key),
        "stable_key": list(value.stable_key()),
        "observation_keys": [
            list(item.stable_key()) for item in value.observations],
    }


def _qualification_dict(
        receipt: ConversationHeldOutQualificationReceipt,
        ) -> dict[str, Any]:
    """序列化 qualification 的无标签证据与派生状态。"""
    rollback = receipt.rollback_recovery
    return {
        "manifest_key": list(receipt.manifest_key),
        "catalog_key": list(receipt.catalog_key),
        "stable_key": list(receipt.stable_key()),
        "execution": _execution_dict(receipt.execution),
        "fresh_execution": _execution_dict(receipt.fresh_execution),
        "clone_execution": _execution_dict(receipt.clone_execution),
        "resumed_execution": _execution_dict(receipt.resumed_execution),
        "rollback_recovery": {
            "fault_key": list(rollback.fault_key),
            "before_snapshot_key": list(rollback.before_snapshot_key),
            "after_fault_snapshot_key": list(rollback.after_fault_snapshot_key),
            "after_recovery_snapshot_key": list(
                rollback.after_recovery_snapshot_key),
            "recovered_execution": _execution_dict(
                rollback.recovered_execution),
            "recovered_clean": int(rollback.recovered_clean),
        },
        "replay_stable": int(receipt.replay_stable),
        "storage_stable": int(receipt.storage_stable),
        "axis_audit": [
            {
                "axis": _key(item.axis),
                "case_keys": [_key(case_key) for case_key in item.case_keys],
                "typed_input_bound": item.typed_input_bound,
                "semantic_runtime_bound": item.semantic_runtime_bound,
                "manifest_only": item.manifest_only,
            }
            for item in receipt.axis_audit
        ],
    }


def _relative_file(root: Path, path: Path) -> str:
    """返回以仓库 root 为边界的 POSIX 相对路径。"""
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ConversationHeldOutFreezeError(
            f"冻结文件越出仓库 root: {resolved}") from error
    return relative.as_posix()


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    """读取单个公开文件并记录相对路径、字节数和 hash。"""
    if not path.is_file():
        raise ConversationHeldOutFreezeError(f"冻结文件不存在: {path}")
    payload = path.read_bytes()
    return {
        "path": _relative_file(root, path),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _test_dependency_paths(root: Path) -> tuple[Path, ...]:
    """递归收集 DLG-05 测试及其直接 test_ fixture 依赖。"""
    tests_root = root / "tests"
    pending = [tests_root / name for name in _DLG05_TEST_NAMES]
    seen: set[Path] = set()
    result: list[Path] = []
    while pending:
        path = pending.pop()
        path = path.resolve()
        if path in seen:
            continue
        if not path.is_file() or path.parent != tests_root.resolve():
            raise ConversationHeldOutFreezeError(
                f"DLG-05 test fixture 不在 tests 根目录: {path}")
        seen.add(path)
        result.append(path)
        text = path.read_text(encoding="utf-8")
        for module_name in _TEST_IMPORT_RE.findall(text):
            dependency = tests_root / f"{module_name}.py"
            if dependency.exists():
                pending.append(dependency)
    return tuple(sorted(result))


def build_dlg05_public_freeze_document(
        repository_root: str | Path,
        catalog: ConversationHeldOutInputCatalog,
        manifest: ConversationHeldOutManifest,
        qualification: ConversationHeldOutQualificationReceipt,
        ) -> dict[str, Any]:
    """生成 DLG-05 公开冻结清单；不包含 label 或运行时文本。"""
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ConversationHeldOutFreezeError("repository root 不存在")
    if catalog.stable_key() != qualification.catalog_key:
        raise ConversationHeldOutFreezeError("freeze catalog key 漂移")
    if manifest.stable_key() != qualification.manifest_key:
        raise ConversationHeldOutFreezeError("freeze manifest key 漂移")
    source_paths = tuple(sorted((root / "src" / "pure_integer_ai").rglob("*.py")))
    test_paths = _test_dependency_paths(root)
    sample_path = root / "data" / "ph2" / "grounded_answer_train_v1.jsonl.sample"
    harness_paths = (
        root / "conftest.py",
        root / "scripts" / "freeze_dlg05_public_preflight.py",
    )
    files = {
        "source": [_file_record(root, path) for path in source_paths],
        "dlg05_tests_and_fixtures": [
            _file_record(root, path) for path in test_paths],
        "freeze_harness": [_file_record(root, path) for path in harness_paths],
        "training_sample": [_file_record(root, sample_path)],
    }
    inventory_payload = canonical_json_bytes(files)
    document: dict[str, Any] = {
        "schema": "dlg05-public-preflight-freeze-v1",
        "authority": "public-preflight-only",
        "labels_included": 0,
        "formal_run": 0,
        "manifest": _manifest_dict(manifest),
        "catalog": _catalog_dict(catalog),
        "qualification": _qualification_dict(qualification),
        "file_inventory": {
            "inventory_sha256": _sha256_bytes(inventory_payload),
            **files,
        },
    }
    document["document_sha256"] = _sha256_bytes(canonical_json_bytes(document))
    return document


def write_dlg05_public_freeze_document(
        target: str | Path,
        repository_root: str | Path,
        catalog: ConversationHeldOutInputCatalog,
        manifest: ConversationHeldOutManifest,
        qualification: ConversationHeldOutQualificationReceipt,
        ) -> Path:
    """不可覆盖写出公开冻结 JSON；已有内容必须逐字节相同。"""
    path = Path(target).resolve()
    root = Path(repository_root).resolve()
    expected_parent = (root / "data" / "ph2" / "manifests").resolve()
    if path.parent != expected_parent:
        raise ConversationHeldOutFreezeError(
            "freeze target 必须位于 data/ph2/manifests")
    payload = canonical_json_bytes(build_dlg05_public_freeze_document(
        root, catalog, manifest, qualification)) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ConversationHeldOutFreezeError(
                "freeze target 已存在且内容不同，不允许覆盖")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def verify_dlg05_public_freeze_document(
        path: str | Path,
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """只读复算冻结文档、分组 inventory 与每个文件 SHA-256。"""
    root = Path(repository_root).resolve()
    target = Path(path).resolve()
    expected_parent = (root / "data" / "ph2" / "manifests").resolve()
    if target.parent != expected_parent or not target.is_file():
        raise ConversationHeldOutFreezeError(
            "verify target 必须是 data/ph2/manifests 中的既有文件")
    payload = target.read_bytes()
    if not payload.endswith(b"\n"):
        raise ConversationHeldOutFreezeError("freeze document 缺少规范换行")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ConversationHeldOutFreezeError("freeze document 不是规范 JSON")
    if (value.get("schema") != "dlg05-public-preflight-freeze-v1"
            or value.get("labels_included") != 0
            or value.get("formal_run") != 0):
        raise ConversationHeldOutFreezeError(
            "freeze document 越过 public/label-free 边界")
    declared_document_sha = value.get("document_sha256")
    without_document_sha = dict(value)
    without_document_sha.pop("document_sha256", None)
    actual_document_sha = _sha256_bytes(
        canonical_json_bytes(without_document_sha))
    if declared_document_sha != actual_document_sha:
        raise ConversationHeldOutFreezeError("freeze document SHA-256 漂移")
    inventory = value.get("file_inventory")
    if not isinstance(inventory, dict):
        raise ConversationHeldOutFreezeError("freeze file inventory 缺失")
    inventory_groups = {
        name: records
        for name, records in inventory.items()
        if name != "inventory_sha256"
    }
    actual_inventory_sha = _sha256_bytes(
        canonical_json_bytes(inventory_groups))
    if inventory.get("inventory_sha256") != actual_inventory_sha:
        raise ConversationHeldOutFreezeError("freeze inventory SHA-256 漂移")
    file_count = 0
    for group, records in inventory_groups.items():
        if not isinstance(records, list) or not records:
            raise ConversationHeldOutFreezeError(
                f"freeze inventory group 非法: {group}")
        for record in records:
            if not isinstance(record, dict):
                raise ConversationHeldOutFreezeError(
                    f"freeze inventory record 非法: {group}")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative:
                raise ConversationHeldOutFreezeError(
                    f"freeze inventory path 非法: {group}")
            resolved = (root / Path(*relative.split("/"))).resolve()
            current = _file_record(root, resolved)
            if current != record:
                raise ConversationHeldOutFreezeError(
                    f"freeze file identity 漂移: {relative}")
            file_count += 1
    return {
        "path": _relative_file(root, target),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "document_sha256": actual_document_sha,
        "inventory_sha256": actual_inventory_sha,
        "file_count": file_count,
        "verified": 1,
    }


__all__ = [
    "ConversationHeldOutFreezeError",
    "build_dlg05_public_freeze_document",
    "verify_dlg05_public_freeze_document",
    "write_dlg05_public_freeze_document",
]
