"""P3-C3 bound runtime artifact writer 的窄专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_bound_runtime_artifact_writer as writer_module
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_UNKNOWN
from pure_integer_ai.experiments.conversation_heldout_v4_bound_capsule_consumer_gate import (
    ConversationHeldOutV4BoundCapsuleConsumerGateInput,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    read_v4_runtime_static_assets,
    run_v4_candidate_runtime,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bound_runtime_artifact_writer import (
    ConversationHeldOutV4BoundRuntimeArtifactBudget,
    ConversationHeldOutV4BoundRuntimeArtifactWriterError,
    ConversationHeldOutV4BoundRuntimeArtifactWriterInput,
    ConversationHeldOutV4RuntimeStaticAssetReadBudget,
    V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED,
    write_v4_bound_runtime_artifact,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    read_budgeted_v4_external_input_capsule,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple

from test_ph2_conversation_heldout_v4_bound_capsule_consumer_gate import _bound_input  # noqa: E402


@pytest.fixture(scope="module")
def _shared_c1c_source(tmp_path_factory):
    """复用只读 C1c capability；每个 C3 用例仍独占 C2 caller 和 artifact root。"""
    return _bound_input(tmp_path_factory.mktemp("p3-c3-shared-c1c"))


def _input(
        tmp_path: Path,
        shared_c1c_source,
        ) -> ConversationHeldOutV4BoundRuntimeArtifactWriterInput:
    """从只读 C1c capability 创建本例独占 C2 future artifact root。"""
    adapter, template_caller = shared_c1c_source
    caller = replace(
        template_caller,
        future_artifact_root=tmp_path / "p3-c2-future-artifact",
    )
    return ConversationHeldOutV4BoundRuntimeArtifactWriterInput(
        ConversationHeldOutV4BoundCapsuleConsumerGateInput(adapter, caller),
        ConversationHeldOutV4RuntimeStaticAssetReadBudget(
            512, 4_000_000, 64_000_000, 4_000_000),
        ConversationHeldOutV4BoundRuntimeArtifactBudget(
            8_192, 8_192, 8_192, 8_192, 2_000_000, 4_000_000, 4_000_000),
        caller.caller_code_identity,
        "p3-c3-test",
    )


def _install_small_runtime(monkeypatch):
    """隔离 C3 边界测试中的 runtime payload；C2/source 均保持真实物理回读。"""
    inventory = SimpleNamespace(
        execution_code=(SimpleNamespace(),),
        execution_code_total_size=11,
        surface_sample_sha256=(0,) * 32,
        surface_sample_size=7,
        stable_key=lambda: (71, 72, 73),
    )
    assets = SimpleNamespace(
        inventory=inventory,
        stable_key=lambda: (74, 75, 76),
    )
    static_calls = []
    runtime_calls = []

    def static_loader(budget, *, test_transport):
        static_calls.append((budget, test_transport))
        return assets

    def runtime(capsule, *, static_assets):
        runtime_calls.append((capsule, static_assets))
        return SimpleNamespace(
            capsule=capsule,
            identity=SimpleNamespace(runtime_inventory=inventory),
        )

    payloads = {
        Path("bundle.canonical.ints"): b"bundle",
        Path("runtime_receipt.canonical.ints"): b"receipt",
        Path("freeze.json"): b"freeze",
        Path("projection/dlg05_v4_reading.md"): b"markdown",
        Path("projection/dlg05_v4_reading.html"): b"html",
    }
    monkeypatch.setattr(writer_module, "read_v4_runtime_static_assets", static_loader)
    monkeypatch.setattr(writer_module, "run_v4_candidate_runtime", runtime)
    monkeypatch.setattr(writer_module, "ConversationHeldOutV4CandidateRuntimeResult", object)
    monkeypatch.setattr(writer_module, "_payloads", lambda _result: payloads)
    return static_calls, runtime_calls


def test_c3_revalidates_c2_twice_reads_bound_source_and_publishes_manifest_last(
        tmp_path, monkeypatch, _shared_c1c_source):
    """真实 C2/C1c 只作为 typed source capability，C3 result 不暴露路径或正文。"""
    value = _input(tmp_path, _shared_c1c_source)
    static_calls, runtime_calls = _install_small_runtime(monkeypatch)
    original_gate = writer_module.run_v4_bound_capsule_consumer_zero_write_gate
    gate_calls = []

    def counted_gate(gate_input):
        gate_calls.append(gate_input)
        return original_gate(gate_input)

    monkeypatch.setattr(writer_module, "run_v4_bound_capsule_consumer_zero_write_gate",
                        counted_gate)
    original_read = writer_module.read_budgeted_v4_external_input_capsule
    source_reads = []

    def counted_read(*args, **kwargs):
        source_reads.append((args, kwargs))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(writer_module, "read_budgeted_v4_external_input_capsule", counted_read)
    result = write_v4_bound_runtime_artifact(value)
    root = value.bound_capsule_consumer_gate_input.active_caller_gate_input.future_artifact_root

    assert result.status == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert len(source_reads) == 2
    assert gate_calls == [value.bound_capsule_consumer_gate_input]
    assert len(static_calls) == 2
    assert len(runtime_calls) == 1
    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == {
        "artifact_manifest.json",
        "bundle.canonical.ints",
        "freeze.json",
        "projection/dlg05_v4_reading.html",
        "projection/dlg05_v4_reading.md",
        "runtime_receipt.canonical.ints",
    }
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert manifest["c2_stable_key"]
    assert manifest["caller_code_identity"] == list(value.bridge_code_identity.components)
    assert manifest["c1c_capsule_budget_identity_sha256"]
    assert manifest["runtime_asset_budget_identity_sha256"]
    assert manifest["artifact_budget_identity_sha256"]
    assert manifest["bridge_code_identity"] == list(value.bridge_code_identity.components)
    assert not hasattr(result, "root")
    assert not hasattr(result, "capsule")
    assert not hasattr(result, "runtime_result")


def test_c3_runs_real_c1c_c2_runtime_bridge_without_runtime_mock(
        tmp_path, _shared_c1c_source):
    """真实 C1c -> C2 -> runtime 必须在 D: test transport 内形成可回读 C3 闭包。"""
    value = _input(tmp_path, _shared_c1c_source)
    value = replace(
        value,
        artifact_budget=ConversationHeldOutV4BoundRuntimeArtifactBudget(
            8_192, 65_536, 8_192, 16_384, 131_072, 262_144, 262_144),
    )
    result = write_v4_bound_runtime_artifact(value)
    root = value.bound_capsule_consumer_gate_input.active_caller_gate_input.future_artifact_root

    assert result.status == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert result.counts.integer_stream()[:5] == (1, 2, 2, 1, 6)
    assert (root / "artifact_manifest.json").is_file()
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert manifest["test_transport"] is True
    assert manifest["runtime_static_assets"]["code_file_count"] > 1
    assert manifest["runtime_static_assets"]["surface_sample_size"] > 0


def test_c3_publishes_source_scoped_unknown_without_runtime_mock(tmp_path):
    """真实 C1c 的 UNKNOWN Evidence 必须经 C2/C3 保持原 scope 与测试限定状态。"""
    def unknown_turn(turn):
        return replace(
            turn,
            evidence_plans=tuple(
                replace(plan, stances=(EVIDENCE_UNKNOWN,))
                for plan in turn.evidence_plans),
        )

    adapter, caller = _bound_input(tmp_path, turn_transform=unknown_turn)
    value = ConversationHeldOutV4BoundRuntimeArtifactWriterInput(
        ConversationHeldOutV4BoundCapsuleConsumerGateInput(adapter, caller),
        ConversationHeldOutV4RuntimeStaticAssetReadBudget(
            512, 4_000_000, 64_000_000, 4_000_000),
        ConversationHeldOutV4BoundRuntimeArtifactBudget(
            8_192, 65_536, 8_192, 16_384, 131_072, 262_144, 262_144),
        caller.caller_code_identity,
        "p3-c3-source-scoped-unknown-test",
    )
    capsule = read_budgeted_v4_external_input_capsule(
        caller.source_capsule_root,
        budget=adapter.receipt.budget.capsule_budget,
        require_k_drive=False,
    )
    expected = run_v4_candidate_runtime(
        capsule,
        static_assets=read_v4_runtime_static_assets(
            value.runtime_static_asset_budget,
            test_transport=True,
        ),
    )

    assert capsule.is_external
    assert all(
        frame.selection.stance == frame.selection_protocol.content.unknown
        and frame.selection.selected_candidate_keys == ()
        and frame.selection_protocol.scope == frame.input.request.response_scope
        and frame.selection_protocol.scope.owner == frame.input.request.source.owner
        and frame.selection_protocol.scope.versions == frame.input.request.source.versions
        for frame in expected.frames)

    result = write_v4_bound_runtime_artifact(value)
    root = caller.future_artifact_root
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert result.status == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert result.active_provenance_coverage == "NE"
    assert manifest["status"] == V4_BOUND_RUNTIME_ARTIFACT_STATUS_TEST_TRANSPORT_UNQUALIFIED
    assert manifest["test_transport"] is True
    assert manifest["active_provenance_coverage"] == "NE"
    assert (root / "bundle.canonical.ints").read_bytes() == encode_integer_tuple(
        expected.bundle.canonical_payload)
    assert (root / "runtime_receipt.canonical.ints").read_bytes() == encode_integer_tuple(
        expected.receipt.canonical_payload)


def test_c3_checks_generated_payload_budget_before_creating_future_root(
        tmp_path, monkeypatch, _shared_c1c_source):
    """运行后的 payload 超限不会创建、清理或复用 C0 future root。"""
    value = _input(tmp_path, _shared_c1c_source)
    _install_small_runtime(monkeypatch)
    tiny_budget = ConversationHeldOutV4BoundRuntimeArtifactBudget(
        1, 8_192, 8_192, 8_192, 2_000_000, 4_000_000, 4_000_000)
    value = ConversationHeldOutV4BoundRuntimeArtifactWriterInput(
        value.bound_capsule_consumer_gate_input,
        value.runtime_static_asset_budget,
        tiny_budget,
        value.bridge_code_identity,
        value.logical_stage_name,
    )
    root = value.bound_capsule_consumer_gate_input.active_caller_gate_input.future_artifact_root

    with pytest.raises(ConversationHeldOutV4BoundRuntimeArtifactWriterError,
                       match="output budget"):
        write_v4_bound_runtime_artifact(value)

    assert not root.exists()


def test_c3_rejects_second_capsule_read_drift_without_publishing_manifest(
        tmp_path, monkeypatch, _shared_c1c_source):
    """source 的第二次受预算读取发生在 manifest 前，漂移仅留下未发布 partial。"""
    value = _input(tmp_path, _shared_c1c_source)
    _install_small_runtime(monkeypatch)
    original_read = writer_module.read_budgeted_v4_external_input_capsule
    reads = []

    def drift_on_second(*args, **kwargs):
        capsule = original_read(*args, **kwargs)
        reads.append(None)
        if len(reads) == 2:
            return replace(capsule, manifest_sha256=(0,) * 32)
        return capsule

    monkeypatch.setattr(writer_module, "read_budgeted_v4_external_input_capsule",
                        drift_on_second)
    root = value.bound_capsule_consumer_gate_input.active_caller_gate_input.future_artifact_root
    with pytest.raises(ConversationHeldOutV4BoundRuntimeArtifactWriterError):
        write_v4_bound_runtime_artifact(value)

    assert len(reads) == 2
    assert root.is_dir()
    assert not (root / "artifact_manifest.json").exists()


def test_c3_write_failure_keeps_unpublished_partial_root(
        tmp_path, monkeypatch, _shared_c1c_source):
    """任一 fixed payload 的排他写失败都不能发布 manifest 或执行清理。"""
    value = _input(tmp_path, _shared_c1c_source)
    _install_small_runtime(monkeypatch)
    original_write = writer_module.write_exclusive_bytes

    def fail_receipt(root, relative, payload, **kwargs):
        if relative == Path("runtime_receipt.canonical.ints"):
            raise writer_module.KRunBoundaryError("injected write failure")
        return original_write(root, relative, payload, **kwargs)

    monkeypatch.setattr(writer_module, "write_exclusive_bytes", fail_receipt)
    root = value.bound_capsule_consumer_gate_input.active_caller_gate_input.future_artifact_root
    with pytest.raises(ConversationHeldOutV4BoundRuntimeArtifactWriterError,
                       match="artifact boundary"):
        write_v4_bound_runtime_artifact(value)

    assert root.is_dir()
    assert not (root / "artifact_manifest.json").exists()


def test_c3_rejects_untyped_or_bare_entry(tmp_path, _shared_c1c_source):
    """C3 只接受封装 C2 input，不能直接接收 C2 result、C1c result 或 root。"""
    value = _input(tmp_path, _shared_c1c_source)
    with pytest.raises(TypeError):
        write_v4_bound_runtime_artifact(value.bound_capsule_consumer_gate_input)
    with pytest.raises(TypeError):
        ConversationHeldOutV4BoundRuntimeArtifactWriterInput(
            value.bound_capsule_consumer_gate_input.adapter_result,
            value.runtime_static_asset_budget,
            value.artifact_budget,
            value.bridge_code_identity,
            value.logical_stage_name,
        )


def test_c3_stays_outside_legacy_writer_reader_and_forbidden_subsystems():
    """bridge 只使用 bounded reader、现役 runtime 和 K-run capability writer。"""
    source = Path(writer_module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert "pure_integer_ai.experiments.conversation_heldout_v4_runtime_artifact" not in imported
    assert "read_v4_external_input_capsule(" not in source
    assert "write_v4_runtime_artifact(" not in source
    assert all(token not in source for token in (
        "sqlite3", "urllib", "requests", "socket", "teacher", "trainer",
        "private", "formal", "owner",
    ))
