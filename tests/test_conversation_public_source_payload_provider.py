"""DLG-RAW-07 公开 source payload closure 的有界专项。"""
from __future__ import annotations

import ast
from pathlib import Path
from shutil import copyfile

import pytest

from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    PublicSourcePayloadHostError,
    load_public_source_payload_closure_from_root,
    read_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1,
    PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1,
    PublicSourcePayloadProviderError,
    PublicSourcePayloadRecordV1,
    build_public_source_payload_closure_v1,
    portable_integer_record_bytes_v1,
    portable_sha256_v1,
    public_source_payload_record_from_u8_v1,
    require_public_source_payload_closure_identity_v1,
)


_ROOT = Path(__file__).resolve().parents[1]
_PROVIDER_SOURCE = (
    _ROOT / "src/pure_integer_ai/experiments/"
    "conversation_public_source_payload_provider.py"
)


def _copy_public_dialogue_resources(target: Path) -> Path:
    """将冻结的 53 个 raw payload 复制到独立物理根，不保留 mtime 语义。"""
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_ROOT / relative_path, destination)
    return target


def test_frozen_registry_is_exactly_56_ascii_resources_in_unsigned_order() -> None:
    """logical registry 必须固定为现有 public dialogue 的完整 56 项闭包。"""
    assert len(PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1) == 56
    assert all(type(key) is bytes and key.isascii()
               for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)
    assert tuple(sorted(PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)) == (
        PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)
    assert all((_ROOT / key.decode("ascii")).is_file()
               for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)


def test_portable_sha_framing_has_fixed_cross_language_golden_vector() -> None:
    """port 只能使用这个 u64 framing、domain 和逐 byte SHA 结果。"""
    record = (0, 1, 255, 256)
    assert portable_integer_record_bytes_v1(record, label="golden") == bytes.fromhex(
        "0000000000000004000000000000000100000000000000000101"
        "0000000000000001ff00000000000000020100")
    assert portable_sha256_v1(
        b"PURE-INTEGER-AI/DLG-RAW-07/GOLDEN/V1",
        ((), record, (65001, 60, 1)),
    ).hex() == "71cd8b152de7eb60671f57ae7defd669cd85a7ed21db1ff19caf431fdbff3d1b"


def test_two_physical_roots_with_same_bytes_have_same_closure_and_trace_identity(
        tmp_path: Path,
        ) -> None:
    """root、inode、mtime 和读取位置均不进入 closure 或纯读取 trace。"""
    first = read_public_source_payload_closure_from_root(_ROOT)
    copied_root = _copy_public_dialogue_resources(tmp_path / "copied-public")
    second = read_public_source_payload_closure_from_root(copied_root)

    assert first.closure.closure_identity.hex() == (
        "2e8793222e2dc105ba6079220748fa9d291aa928321b1045c9809804caf6b2da")
    assert first.closure.closure_identity == second.closure.closure_identity
    assert first.closure.canonical_record() == second.closure.canonical_record()
    assert first.read_trace == second.read_trace
    assert tuple(record.logical_key for record in first.closure.records) == (
        PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1)
    assert all(entry.result_code == PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1
               and entry.write_effect_code
               == PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1
               for entry in first.read_trace)


def test_builder_canonicalizes_input_order_without_leaking_request_order() -> None:
    """closure identity 只绑定规范 key 顺序，不能绑定 host 读取顺序。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    rebuilt = build_public_source_payload_closure_v1(tuple(reversed(closure.records)))

    assert rebuilt.records == closure.records
    assert rebuilt.closure_identity == closure.closure_identity


def test_missing_duplicate_and_unregistered_or_non_ascii_keys_fail_closed(
        tmp_path: Path,
        ) -> None:
    """registry 不允许缺失、重复、额外或非 ASCII logical key。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    with pytest.raises(PublicSourcePayloadProviderError):
        build_public_source_payload_closure_v1(closure.records[:-1])
    with pytest.raises(PublicSourcePayloadProviderError):
        build_public_source_payload_closure_v1(
            (*closure.records[:-1], closure.records[0]))
    with pytest.raises(PublicSourcePayloadProviderError):
        public_source_payload_record_from_u8_v1(
            b"data/ph2/not-registered.txt",
            b"payload",
        )
    with pytest.raises(PublicSourcePayloadProviderError):
        public_source_payload_record_from_u8_v1(
            b"data/ph2/\xff-invalid.txt",
            b"payload",
        )

    missing_root = _copy_public_dialogue_resources(tmp_path / "missing-public")
    target = missing_root / PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1[0].decode("ascii")
    target.unlink()
    with pytest.raises(PublicSourcePayloadHostError) as failure:
        read_public_source_payload_closure_from_root(missing_root)
    assert failure.value.result_code == PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1


def test_host_rejects_direct_resource_symlink_when_platform_allows_creation(
        tmp_path: Path,
        ) -> None:
    """物理 adapter 不得沿 public closure 内的链接读取 root 外内容。"""
    linked_root = _copy_public_dialogue_resources(tmp_path / "linked-public")
    target = linked_root / PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1[0].decode("ascii")
    external = tmp_path / "outside-payload.txt"
    copyfile(target, external)
    target.unlink()
    try:
        target.symlink_to(external)
    except (NotImplementedError, OSError):
        pytest.skip("当前 Windows 权限不允许建立文件符号链接")

    with pytest.raises(PublicSourcePayloadHostError) as failure:
        read_public_source_payload_closure_from_root(linked_root)
    assert failure.value.result_code == PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1


def test_payload_length_digest_and_u64_drift_are_rejected_before_closure_use() -> None:
    """payload record 的显式 length/SHA 是可验证数据，不能由 host 猜测补足。"""
    closure = load_public_source_payload_closure_from_root(_ROOT)
    record = closure.records[0]
    with pytest.raises(PublicSourcePayloadProviderError):
        PublicSourcePayloadRecordV1(
            record.logical_key,
            record.raw_payload,
            record.payload_length + 1,
            record.raw_sha256,
            PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
        )
    with pytest.raises(PublicSourcePayloadProviderError):
        PublicSourcePayloadRecordV1(
            record.logical_key,
            record.raw_payload,
            record.payload_length,
            b"\x00" * 32,
            PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
        )
    with pytest.raises(PublicSourcePayloadProviderError):
        PublicSourcePayloadRecordV1(
            record.logical_key,
            record.raw_payload,
            1 << 64,
            record.raw_sha256,
            PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
        )


def test_expected_identity_rejects_one_byte_content_drift_at_host_boundary(
        tmp_path: Path,
        ) -> None:
    """恢复绑定后的任一 source 内容漂移必须在 runtime 前 fail closed。"""
    baseline = read_public_source_payload_closure_from_root(_ROOT)
    changed_root = _copy_public_dialogue_resources(tmp_path / "changed-public")
    target = changed_root / PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1[-1].decode("ascii")
    raw = target.read_bytes()
    target.write_bytes(raw + b"\n")

    changed = read_public_source_payload_closure_from_root(changed_root)
    assert changed.closure.closure_identity != baseline.closure.closure_identity
    with pytest.raises(PublicSourcePayloadProviderError):
        require_public_source_payload_closure_identity_v1(
            changed.closure,
            baseline.closure.closure_identity,
        )
    with pytest.raises(PublicSourcePayloadHostError) as failure:
        read_public_source_payload_closure_from_root(
            changed_root,
            expected_closure_identity=baseline.closure.closure_identity,
        )
    assert failure.value.result_code == PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1


def test_pure_provider_ast_has_no_physical_filesystem_boundary() -> None:
    """纯 core 不能导入路径模块、读取文件或接收旧 repository-root 语义。"""
    source = _PROVIDER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ) + tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "pathlib" not in imported_modules
    assert "repository_root" not in source
    assert "read_bytes" not in source
    assert "PurePosixPath" not in source
    assert "Path(" not in source
