"""P3-C2 bound C1c capsule consumer gate 的窄专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_bound_capsule_consumer_gate as gate_module
from pure_integer_ai.experiments.conversation_heldout_v4_active_caller_gate import (
    ConversationHeldOutV4ActiveCallerGateInput,
    run_v4_active_caller_zero_write_dry_run,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bound_capsule_consumer_gate import (
    ConversationHeldOutV4BoundCapsuleConsumerGateError,
    ConversationHeldOutV4BoundCapsuleConsumerGateInput,
    V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY,
    run_v4_bound_capsule_consumer_zero_write_gate,
)

from test_ph2_conversation_heldout_v4_runtime_task_capsule_adapter import (  # noqa: E402
    _chain,
    _input as _adapter_input,
    _protocol,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_capsule_adapter import (  # noqa: E402
    materialize_v4_runtime_task_capsule_adapter,
)


def _bound_input(tmp_path: Path, *, turn_transform=None):
    """构造真实 C1c test transport result 和严格绑定的 P3-C0 typed input。"""
    active, readback, ledger, blueprint, replay = _chain(
        tmp_path, turn_transform=turn_transform)
    adapter_result = materialize_v4_runtime_task_capsule_adapter(
        _adapter_input(tmp_path, active, readback, ledger, blueprint, replay))
    future_artifact_root = tmp_path / "p3-c2-future-artifact"
    caller_input = ConversationHeldOutV4ActiveCallerGateInput(
        adapter_result.readback,
        adapter_result.capsule_descriptor.parent_run_root.path
        / adapter_result.capsule_descriptor.relative_root,
        future_artifact_root,
        _protocol(8_001),
        True,
        "p3-c2-test",
    )
    return adapter_result, caller_input


@pytest.fixture(scope="module")
def _clean_bound_input(tmp_path_factory):
    """复用一份未篡改 C1c result，避免每个 P3-C2 拒绝分支重复物化上游链。"""
    return _bound_input(tmp_path_factory.mktemp("p3-c2-clean"))


def test_bound_capsule_consumer_gate_binds_real_c1c_to_p3c0_without_exposing_root(
        _clean_bound_input):
    """C2 只在 C1c descriptor child 与同一 P3-B evidence 均成立后调用 P3-C0。"""
    adapter_result, caller_input = _clean_bound_input
    source_before = tuple(caller_input.source_capsule_root.iterdir())
    parent_before = tuple(caller_input.future_artifact_root.parent.iterdir())

    result = run_v4_bound_capsule_consumer_zero_write_gate(
        ConversationHeldOutV4BoundCapsuleConsumerGateInput(
            adapter_result, caller_input))

    assert result.status == V4_BOUND_CAPSULE_CONSUMER_GATE_STATUS_TEST_ONLY
    assert result.adapter_stable_key == adapter_result.stable_key()
    assert (result.capsule_descriptor_stable_key
            == adapter_result.capsule_descriptor.stable_key())
    assert result.active_caller_gate_result.readback_stable_key == (
        adapter_result.readback.stable_key())
    assert result.counts.integer_stream() == (1, 1, 1, 1, 1, 1)
    assert not hasattr(result, "source_capsule_root")
    assert not hasattr(result, "capsule")
    assert tuple(caller_input.source_capsule_root.iterdir()) == source_before
    assert tuple(caller_input.future_artifact_root.parent.iterdir()) == parent_before
    assert not caller_input.future_artifact_root.exists()


def test_bound_capsule_consumer_gate_rejects_bare_or_untyped_entry(
        _clean_bound_input):
    """C2 API 不接受裸 P3-C0 input、bare root 或裸 C1c result。"""
    adapter_result, caller_input = _clean_bound_input

    with pytest.raises(TypeError):
        run_v4_bound_capsule_consumer_zero_write_gate(caller_input)
    with pytest.raises(TypeError):
        ConversationHeldOutV4BoundCapsuleConsumerGateInput(
            adapter_result, caller_input.source_capsule_root)
    with pytest.raises(TypeError):
        ConversationHeldOutV4BoundCapsuleConsumerGateInput(
            caller_input.source_capsule_root, caller_input)


def test_bound_capsule_consumer_gate_rejects_unbound_source_root_before_p3c0(
        _clean_bound_input, monkeypatch):
    """即使 root 是普通存在目录，也必须精确等于 C1c descriptor 的固定 child。"""
    adapter_result, caller_input = _clean_bound_input
    alternative_source = caller_input.future_artifact_root.parent / "alternative-capsule-root"
    alternative_source.mkdir()
    unbound = replace(caller_input, source_capsule_root=alternative_source)
    monkeypatch.setattr(gate_module, "revalidate_v4_runtime_task_capsule_adapter",
                        lambda value: value)
    monkeypatch.setattr(
        gate_module, "run_v4_active_caller_zero_write_dry_run",
        lambda value: (_ for _ in ()).throw(
            AssertionError("unbound source root must fail before P3-C0")),
    )

    with pytest.raises(ConversationHeldOutV4BoundCapsuleConsumerGateError,
                       match="descriptor child"):
        run_v4_bound_capsule_consumer_zero_write_gate(
            ConversationHeldOutV4BoundCapsuleConsumerGateInput(
                adapter_result, unbound))

    assert not caller_input.future_artifact_root.exists()


def test_bound_capsule_consumer_gate_rejects_readback_or_transport_drift(
        _clean_bound_input, monkeypatch):
    """caller 的 P3-B evidence 与 transport 都必须与 C1c capability 完全相同。"""
    adapter_result, caller_input = _clean_bound_input
    monkeypatch.setattr(gate_module, "revalidate_v4_runtime_task_capsule_adapter",
                        lambda value: value)
    drifted_readback = replace(
        caller_input.readback,
        manifest_sha256=tuple(reversed(caller_input.readback.manifest_sha256)),
    )
    with pytest.raises(ConversationHeldOutV4BoundCapsuleConsumerGateError,
                       match="P3-B readback"):
        run_v4_bound_capsule_consumer_zero_write_gate(
            ConversationHeldOutV4BoundCapsuleConsumerGateInput(
                adapter_result,
                replace(caller_input, readback=drifted_readback)))
    with pytest.raises(ConversationHeldOutV4BoundCapsuleConsumerGateError,
                       match="transport"):
        run_v4_bound_capsule_consumer_zero_write_gate(
            ConversationHeldOutV4BoundCapsuleConsumerGateInput(
                adapter_result, replace(caller_input, test_transport=False)))
    assert not caller_input.future_artifact_root.exists()


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("caller_code_identity", _protocol(8_002)),
        ("logical_stage_name", "p3-c2-other-stage"),
    ),
)
def test_bound_capsule_consumer_gate_rejects_p3c0_result_identity_drift(
        _clean_bound_input, monkeypatch, field_name, replacement):
    """P3-C0 result 的 caller 与 logical stage 也必须回绑到同一次 input。"""
    adapter_result, caller_input = _clean_bound_input
    expected = run_v4_active_caller_zero_write_dry_run(caller_input)
    monkeypatch.setattr(gate_module, "revalidate_v4_runtime_task_capsule_adapter",
                        lambda value: value)
    monkeypatch.setattr(
        gate_module, "run_v4_active_caller_zero_write_dry_run",
        lambda value: replace(expected, **{field_name: replacement}),
    )

    with pytest.raises(ConversationHeldOutV4BoundCapsuleConsumerGateError,
                       match="P3-C0 result"):
        run_v4_bound_capsule_consumer_zero_write_gate(
            ConversationHeldOutV4BoundCapsuleConsumerGateInput(
                adapter_result, caller_input))


def test_bound_capsule_consumer_gate_revalidates_c1c_before_p3c0(tmp_path, monkeypatch):
    """被篡改的 C1c capsule 必须在进入 P3-C0 前被封闭重验拒绝。"""
    adapter_result, caller_input = _bound_input(tmp_path)
    manifest = (adapter_result.capsule_descriptor.parent_run_root.path
                / adapter_result.capsule_descriptor.relative_root / "manifest.json")
    payload = manifest.read_bytes()
    manifest.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))

    def unexpected_p3c0(_value):
        raise AssertionError("tampered C1c must fail before P3-C0")

    monkeypatch.setattr(gate_module, "run_v4_active_caller_zero_write_dry_run",
                        unexpected_p3c0)
    with pytest.raises(ConversationHeldOutV4BoundCapsuleConsumerGateError,
                       match="C1c adapter revalidation"):
        run_v4_bound_capsule_consumer_zero_write_gate(
            ConversationHeldOutV4BoundCapsuleConsumerGateInput(
                adapter_result, caller_input))


def test_bound_capsule_consumer_gate_stays_outside_runtime_writer_and_legacy_reader():
    """C2 只能编排两个 typed gate，不能自行导入或调用被禁止的边界。"""
    source = Path(gate_module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert imported == {
        "__future__",
        "dataclasses",
        "pure_integer_ai.experiments.conversation_heldout_v4_active_caller_gate",
        "pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_capsule_adapter",
        "pure_integer_ai.storage.integer_codec",
    }
    assert all(token not in source for token in (
        "read_v4_external_input_capsule(",
        "read_budgeted_v4_external_input_capsule(",
        "run_v4_candidate_runtime(",
        "materialize_v4_runtime_artifact(",
        "sqlite3", "tempfile", "urllib", "requests", "socket",
    ))
