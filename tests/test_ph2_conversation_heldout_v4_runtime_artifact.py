"""DLG-05 v4 external-capsule runtime artifact 的无标签专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_runtime_artifact as artifact_module
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    V4_RUNTIME_CODE_CLOSURE_SCHEMA,
    V4_RUNTIME_CODE_RELATIVE_PATH,
    V4_RUNTIME_FAMILY_KEY,
    build_v4_synthetic_runtime_fixture,
    read_v4_runtime_inventory,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalInputCapsule,
    ConversationHeldOutV4ExternalProducer,
    read_v4_external_input_capsule,
    write_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_artifact import (
    ConversationHeldOutV4RuntimeArtifactError,
    V4_RUNTIME_ARTIFACT_UNQUALIFIED,
    audit_v4_runtime_artifact,
    write_v4_runtime_artifact,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import (
    parse_canonical_json_bytes,
)
from pure_integer_ai.storage.integer_codec import decode_integer_tuple


_NAMESPACE = (20260821, 405, 503)


def _sha256(value: str) -> tuple[int, ...]:
    """为 test-only transport 原文重算完整内容 SHA。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _test_transport_draft(
        *, producer_suffix: int = 1,
        ) -> ConversationHeldOutV4ExternalInputCapsule:
    """只构造一个 synthetic-derived transport fixture，绝不宣称独立来源。"""
    fixture = build_v4_synthetic_runtime_fixture()
    source_input = fixture.inputs[0]
    source_record = source_input.source_records[0]
    # synthetic fixture 的 anchor 位置为 runtime 回归而设；加长原文后才满足外部输入范围合同。
    raw_text = source_record.raw_text + "。" * 96
    external_record = replace(
        source_record,
        raw_text_scalars=unicode_scalars(raw_text),
        content_sha256=_sha256(raw_text),
    )
    external_input = replace(
        source_input,
        source_records=(external_record,),
        evidence_plans=tuple(sorted(
            source_input.evidence_plans,
            key=lambda item: item.target.stable_key(),
        )),
    )
    return ConversationHeldOutV4ExternalInputCapsule(
        V4_RUNTIME_FAMILY_KEY,
        ConversationHeldOutV4ExternalProducer(
            ProtocolKey((*_NAMESPACE, producer_suffix))),
        ConversationHeldOutV4DependencyBinding(
            fixture.dependencies.artifact_sha256,
            fixture.dependencies.inventory_sha256,
            fixture.dependencies.document_sha256,
        ),
        (external_input,),
    )


def _external_runtime(tmp_path, *, producer_suffix: int = 1):
    """发布 test-only transport；writer/audit 自己运行无标签 candidate runtime。"""
    source_root = write_v4_external_input_capsule(
        tmp_path / f"external-source-{producer_suffix}",
        _test_transport_draft(producer_suffix=producer_suffix),
        require_k_drive=False,
    )
    return source_root


def _snapshot(root):
    """记录 artifact 当前全部相对文件字节，供 audit 零写断言使用。"""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def test_runtime_artifact_binds_external_transport_and_audits_read_only(
        tmp_path, monkeypatch):
    """新闭包必须绑定同一 capsule/runtime，并由只读无标签重跑逐字节复核。"""
    source_root = _external_runtime(tmp_path)
    calls = []
    original_runtime = artifact_module.run_v4_candidate_runtime

    def counted_runtime(capsule):
        calls.append(capsule)
        return original_runtime(capsule)

    monkeypatch.setattr(
        artifact_module, "run_v4_candidate_runtime", counted_runtime)
    paths = write_v4_runtime_artifact(
        tmp_path / "runtime-artifact",
        source_root,
        require_k_drive=False,
    )
    monkeypatch.setattr(
        artifact_module, "run_v4_candidate_runtime", original_runtime)
    assert len(calls) == 1

    assert set(_snapshot(paths.root)) == {
        "artifact_manifest.json",
        "bundle.canonical.ints",
        "freeze.json",
        "projection/dlg05_v4_reading.html",
        "projection/dlg05_v4_reading.md",
        "runtime_receipt.canonical.ints",
    }
    assert decode_integer_tuple(paths.bundle.read_bytes())
    assert decode_integer_tuple(paths.runtime_receipt.read_bytes())
    manifest = parse_canonical_json_bytes(
        paths.manifest.read_bytes(), require_object=True)
    capsule = read_v4_external_input_capsule(
        source_root, require_k_drive=False)
    assert manifest["qualification_status"] == V4_RUNTIME_ARTIFACT_UNQUALIFIED
    assert manifest["external_capsule"]["origin"] == "EXTERNAL_SOURCE_CAPSULE"
    assert manifest["external_capsule"]["producer"]["producer_key"] == list(
        capsule.external_producer_key.components)
    inventory = read_v4_runtime_inventory()
    execution_code = manifest["runtime_inventory"]["execution_code"]
    assert execution_code["closure_schema"] == V4_RUNTIME_CODE_CLOSURE_SCHEMA
    assert execution_code["closure_sha256"] == bytes(
        inventory.execution_code_closure_sha256).hex()
    assert execution_code["file_count"] == len(inventory.execution_code)
    assert execution_code["total_size"] == inventory.execution_code_total_size
    assert execution_code["files"] == [
        {
            "path": item.relative_path,
            "sha256": bytes(item.sha256).hex(),
            "size": item.size,
        }
        for item in inventory.execution_code
    ]
    assert V4_RUNTIME_CODE_RELATIVE_PATH in {
        item["path"] for item in execution_code["files"]}
    assert manifest["runtime_inventory"]["surface_sample"]["sha256"] == bytes(
        inventory.surface_sample_sha256).hex()
    assert manifest["files_scope"] == "PRE_MANIFEST_PAYLOAD_FILES_ONLY"
    assert set(manifest["files"]) == {
        "bundle.canonical.ints",
        "freeze.json",
        "projection/dlg05_v4_reading.html",
        "projection/dlg05_v4_reading.md",
        "runtime_receipt.canonical.ints",
    }

    before = _snapshot(paths.root)
    audit = audit_v4_runtime_artifact(
        paths.root, source_root, require_k_drive=False)
    assert audit.qualification_status == V4_RUNTIME_ARTIFACT_UNQUALIFIED
    assert audit.external_capsule_bound == 1
    assert audit.frame_count == 1
    assert audit.executor_call_count == 1
    assert _snapshot(paths.root) == before


def test_runtime_artifact_rejects_overlap_existing_and_non_k_root(
        tmp_path):
    """writer 不得复用、嵌套或绕过 K 盘生产根。"""
    source_root = _external_runtime(tmp_path)
    rejected = tmp_path / "must-not-exist"
    with pytest.raises(ConversationHeldOutV4RuntimeArtifactError, match="K 盘"):
        write_v4_runtime_artifact(rejected, source_root)
    assert not rejected.exists()

    with pytest.raises(ConversationHeldOutV4RuntimeArtifactError, match="不得重叠"):
        write_v4_runtime_artifact(
            source_root / "nested-artifact",
            source_root,
            require_k_drive=False,
        )
    paths = write_v4_runtime_artifact(
        tmp_path / "published", source_root, require_k_drive=False)
    with pytest.raises(ConversationHeldOutV4RuntimeArtifactError, match="此前不存在|已存在"):
        write_v4_runtime_artifact(
            paths.root, source_root, require_k_drive=False)


def test_runtime_artifact_keeps_manifest_last_and_audit_rejects_tamper(tmp_path, monkeypatch):
    """写入失败不发布 manifest；任一 artifact/source 闭包漂移均 fail closed。"""
    source_root = _external_runtime(tmp_path)
    original = artifact_module._write_exclusive

    def fail_before_manifest(root, relative, payload):
        if relative.name == "runtime_receipt.canonical.ints":
            raise ConversationHeldOutV4RuntimeArtifactError("injected write failure")
        return original(root, relative, payload)

    monkeypatch.setattr(artifact_module, "_write_exclusive", fail_before_manifest)
    partial = tmp_path / "partial"
    with pytest.raises(ConversationHeldOutV4RuntimeArtifactError, match="injected"):
        write_v4_runtime_artifact(
            partial, source_root, require_k_drive=False)
    assert (partial / "bundle.canonical.ints").is_file()
    assert not (partial / "artifact_manifest.json").exists()

    monkeypatch.setattr(artifact_module, "_write_exclusive", original)
    paths = write_v4_runtime_artifact(
        tmp_path / "published", source_root, require_k_drive=False)
    for path in (paths.bundle, paths.manifest):
        original_payload = path.read_bytes()
        path.write_bytes(original_payload + b" ")
        with pytest.raises(ConversationHeldOutV4RuntimeArtifactError):
            audit_v4_runtime_artifact(
                paths.root, source_root, require_k_drive=False)
        path.write_bytes(original_payload)


def test_runtime_artifact_audit_rejects_runtime_inventory_drift(
        tmp_path, monkeypatch):
    """审计结束前仍须复核 runtime code/sample identity，不能只复核 source。"""
    source_root = _external_runtime(tmp_path)
    paths = write_v4_runtime_artifact(
        tmp_path / "published", source_root, require_k_drive=False)
    baseline = artifact_module.read_v4_runtime_inventory()
    calls = []

    def inventory_with_late_drift():
        calls.append(None)
        return baseline if len(calls) == 1 else object()

    monkeypatch.setattr(
        artifact_module, "read_v4_runtime_inventory", inventory_with_late_drift)
    with pytest.raises(ConversationHeldOutV4RuntimeArtifactError, match="漂移"):
        audit_v4_runtime_artifact(
            paths.root, source_root, require_k_drive=False)
    assert len(calls) == 2
