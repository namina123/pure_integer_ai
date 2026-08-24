"""P3-C1c 同次 payload replay 与 typed capsule 的窄专项。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_capsule_adapter as adapter_module
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    materialize_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleBudget,
    ConversationHeldOutV4ExternalProducer,
    read_budgeted_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    materialize_v4_payload_witness_ledger,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation import (
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_blueprint import (
    materialize_v4_runtime_task_blueprint,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_capsule_adapter import (
    ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget,
    ConversationHeldOutV4RuntimeTaskCapsuleAdapterError,
    ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput,
    ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration,
    ConversationHeldOutV4RuntimeTaskPayloadReplayItem,
    V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY,
    load_v4_runtime_task_capsule_adapter_publication,
    materialize_v4_runtime_task_capsule_adapter,
    revalidate_v4_runtime_task_capsule_adapter,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.k_run_boundary import create_new_run_root

from test_ph2_conversation_heldout_v4_active_roster_readback import (  # noqa: E402
    _request as _p3b_request,
)
from test_ph2_conversation_heldout_v4_payload_witness_ledger import (  # noqa: E402
    _input as _p3a2_input,
    _witnesses,
)
from test_ph2_conversation_heldout_v4_runtime_task_blueprint import (  # noqa: E402
    _input as _blueprint_input,
    _selected_witnesses,
    _turn,
)


_NAMESPACE = (20260822, 405, 322)


def _digest(value: str) -> tuple[int, ...]:
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _protocol(value: int) -> ProtocolKey:
    return ProtocolKey((*_NAMESPACE, value))


def _budget() -> ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget:
    return ConversationHeldOutV4RuntimeTaskCapsuleAdapterBudget(
        16,
        16,
        256 * 1024,
        512 * 1024,
        8,
        256 * 1024,
        256 * 1024,
        64 * 1024,
        256 * 1024,
        512 * 1024,
        128 * 1024,
        ConversationHeldOutV4ExternalCapsuleBudget(
            512 * 1024, 512 * 1024, 128 * 1024, 1024 * 1024),
    )


def _chain(tmp_path: Path, *, turn_transform=None):
    """物化真实 P3-A/P3-B/P3-A2/C1a/C1b test-transport 链。"""
    request, *_ = _p3b_request(tmp_path)
    readback = materialize_v4_active_roster_bidirectional_readback(request)
    active = request.active_intake
    ledger = materialize_v4_payload_witness_ledger(
        _p3a2_input(tmp_path, active, readback, lambda: _witnesses(active)))
    selected = _selected_witnesses(ledger)
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        selected, start=1))
    if turn_transform is not None:
        turns = tuple(turn_transform(item) for item in turns)
    blueprint_parent = tmp_path / "blueprint"
    blueprint_parent.mkdir()
    blueprint = materialize_v4_runtime_task_blueprint(_blueprint_input(
        blueprint_parent, active, readback, ledger, lambda: turns,
        annotation_turns=turns))
    witnesses = {item.active_record_key: item for item in _witnesses(active)}
    replay = tuple(ConversationHeldOutV4RuntimeTaskPayloadReplayItem(
        witnesses[record.witness_record_key],
        unicode_scalars(witnesses[record.witness_record_key].license_partition),
        unicode_scalars("P3-C1c test transport attribution"),
        ConversationHeldOutV4RuntimeTaskCapsuleMetadataDeclaration(
            record.source_ref,
            V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
            _protocol(100 + ordinal),
            _protocol(200 + ordinal),
            witnesses[record.witness_record_key].frozen_source_manifest_identity,
            _protocol(300 + ordinal),
            _protocol(400 + ordinal),
            _protocol(500 + ordinal),
        ),
    ) for ordinal, record in enumerate(selected, start=1))
    return active, readback, ledger, blueprint, replay


def _input(tmp_path: Path, active, readback, ledger, blueprint, replay):
    """构造四个 fresh、disjoint C1c test transport roots。"""
    parent = tmp_path / "c1c"
    parent.mkdir()
    root = lambda name: create_new_run_root(  # noqa: E731
        parent / name, require_k_drive=False, label=f"P3-C1c {name}")
    return ConversationHeldOutV4RuntimeTaskCapsuleAdapterInput(
        active,
        readback,
        ledger,
        blueprint,
        lambda: replay,
        ConversationHeldOutV4ExternalProducer(_protocol(600)),
        ConversationHeldOutV4DependencyBinding(
            _digest("artifact"), _digest("inventory"), _digest("document")),
        _protocol(700),
        root("staging"),
        root("work"),
        root("publication"),
        root("capsule-parent"),
        _budget(),
        "p3-c1c-test",
    )


@pytest.fixture(scope="module")
def _shared_upstream_chain(tmp_path_factory):
    """一次物化只读上游闭包，C1c 各反例仍各自创建新输出根。"""
    return _chain(tmp_path_factory.mktemp("p3-c1c-upstream"))


def test_runtime_task_capsule_adapter_replays_c1b_subset_without_retaining_raw_capsule(
        tmp_path, _shared_upstream_chain):
    """C1c 应只接受 C1b 选择的 P3-A2 子序列，并把正文限制在 external child。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)

    result = materialize_v4_runtime_task_capsule_adapter(request)

    assert result.receipt.status == V4_RUNTIME_TASK_CAPSULE_ADAPTER_STATUS_TEST_ONLY
    assert result.receipt.counts.integer_tuple()[:5] == (2, 1, 1, 2, 2)
    assert not hasattr(result, "runtime_capsule")
    assert result.capsule_descriptor.turn_count == 2
    assert revalidate_v4_runtime_task_capsule_adapter(result) == result
    publication = load_v4_runtime_task_capsule_adapter_publication(result)
    assert publication.result == result
    assert len(publication.records) == 2
    assert {path.relative_to(result.publication_run_root.path).as_posix()
            for path in result.publication_run_root.path.rglob("*") if path.is_file()} == {
                "adapter/receipt.pifrs", "manifest.pii"}
    loaded = read_budgeted_v4_external_input_capsule(
        result.capsule_descriptor.parent_run_root.path
        / result.capsule_descriptor.relative_root,
        budget=request.budget.capsule_budget,
        require_k_drive=False,
    )
    assert len(loaded.inputs) == 2
    for root in (request.staging_run_root, request.work_run_root,
                 result.publication_run_root):
        stored = b"".join(
            path.read_bytes() for path in root.path.rglob("*") if path.is_file())
        assert b"p3b-source-101" not in stored
        assert b"blueprint-question-1" not in stored
        assert str(tmp_path).encode() not in stored


@pytest.mark.parametrize("variant", ("missing", "extra", "reversed", "license"))
def test_runtime_task_capsule_adapter_rejects_replay_closure_before_capsule_creation(
        tmp_path, variant, _shared_upstream_chain):
    """factory 缺、余、错序或许可声明漂移均不得创建 C1c capsule child。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    if variant == "missing":
        replay = replay[:-1]
    elif variant == "extra":
        replay = (*replay, replay[-1])
    elif variant == "reversed":
        replay = tuple(reversed(replay))
    else:
        replay = (replace(replay[0], license_scalars=unicode_scalars(
            "INVALID-LICENSE")), *replay[1:])
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)

    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        materialize_v4_runtime_task_capsule_adapter(request)

    assert not any(request.capsule_parent_run_root.path.iterdir())
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_capsule_adapter_module_uses_budgeted_reader_only():
    """C1c 不得回退到 legacy unbounded v1 reader 或调用 runtime execution。"""
    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    assert "read_budgeted_v4_external_input_capsule" in source
    assert "read_v4_external_input_capsule(" not in source
    assert "run_v4_candidate_runtime(" not in source


def test_runtime_task_capsule_adapter_revalidation_rejects_tampered_capsule(
        tmp_path, _shared_upstream_chain):
    """已发布 C1c receipt 仍必须拒绝后续 external child 篡改。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)
    result = materialize_v4_runtime_task_capsule_adapter(request)
    manifest = (result.capsule_descriptor.parent_run_root.path
                / result.capsule_descriptor.relative_root / "manifest.json")
    payload = manifest.read_bytes()
    manifest.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))

    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        revalidate_v4_runtime_task_capsule_adapter(result)


def test_runtime_task_capsule_adapter_revalidation_rejects_metadata_binding_drift(
        tmp_path, _shared_upstream_chain):
    """P0 metadata digest 必须由 result 保存的无 payload declaration 重新导出。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    result = materialize_v4_runtime_task_capsule_adapter(
        _input(tmp_path, active, readback, ledger, blueprint, replay))
    first = result.replay_bindings[0]
    changed = replace(
        first,
        metadata_declaration=replace(
            first.metadata_declaration,
            constructing_code_identity=_protocol(9_999),
        ),
    )
    drifted = replace(result, replay_bindings=(changed, *result.replay_bindings[1:]))

    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        revalidate_v4_runtime_task_capsule_adapter(drifted)


@pytest.mark.parametrize("root_name", ("staging", "work"))
def test_runtime_task_capsule_adapter_rejects_intermediate_receipt_residue(
        tmp_path, monkeypatch, root_name, _shared_upstream_chain):
    """两个 relay root 的写后闭包都必须拒绝残留，且不得发布最终 receipt。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)
    original = adapter_module._write_receipt

    def write_with_residue(root, relative, records, **kwargs):
        stream = original(root, relative, records, **kwargs)
        if root == getattr(request, f"{root_name}_run_root"):
            (root.path / "unexpected-residue.bin").write_bytes(b"residue")
        return stream

    monkeypatch.setattr(adapter_module, "_write_receipt", write_with_residue)
    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        materialize_v4_runtime_task_capsule_adapter(request)

    contaminated = getattr(request, f"{root_name}_run_root")
    assert (contaminated.path / "unexpected-residue.bin").is_file()
    if root_name == "staging":
        assert not any(request.work_run_root.path.iterdir())
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_capsule_adapter_rejects_full_upstream_ledger_before_replay(
        tmp_path, _shared_upstream_chain):
    """C1b 小子集不能绕开 C1c 对整个顺序 P3-A2 ledger 的读取上限。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)
    request = replace(
        request,
        budget=replace(
            request.budget,
            max_upstream_witness_count=3,
            max_upstream_receipt_record_count=3,
        ),
    )
    request = replace(
        request,
        payload_replay_factory=lambda: (_ for _ in ()).throw(
            AssertionError("upstream preflight must run before replay factory")),
    )

    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        materialize_v4_runtime_task_capsule_adapter(request)

    for root in (request.staging_run_root, request.work_run_root,
                 request.publication_run_root, request.capsule_parent_run_root):
        assert not any(root.path.iterdir())


def test_runtime_task_capsule_adapter_revalidation_rejects_replay_binding_order_drift(
        tmp_path, _shared_upstream_chain):
    """已发布 result 仍必须保存 P3-A2 sealed-order 的严格过滤子序列。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    result = materialize_v4_runtime_task_capsule_adapter(
        _input(tmp_path, active, readback, ledger, blueprint, replay))
    drifted = replace(result, replay_bindings=tuple(reversed(result.replay_bindings)))

    with pytest.raises(ConversationHeldOutV4RuntimeTaskCapsuleAdapterError):
        revalidate_v4_runtime_task_capsule_adapter(drifted)


def test_runtime_task_capsule_adapter_passes_all_full_ledger_caps_to_direct_loader(
        tmp_path, monkeypatch, _shared_upstream_chain):
    """C1c 的每次直接 P3-A2 回读都必须携带同一组 caller budget。"""
    active, readback, ledger, blueprint, replay = _shared_upstream_chain
    request = _input(tmp_path, active, readback, ledger, blueprint, replay)
    original = adapter_module.load_v4_payload_witness_publication
    calls = []

    def bounded_loader(value, **kwargs):
        calls.append(kwargs)
        return original(value, **kwargs)

    monkeypatch.setattr(adapter_module, "load_v4_payload_witness_publication", bounded_loader)
    materialize_v4_runtime_task_capsule_adapter(request)

    expected = {
        "max_record_count": request.budget.max_upstream_receipt_record_count,
        "max_total_payload_bytes": request.budget.max_upstream_stream_payload_bytes,
        "max_physical_bytes": request.budget.max_upstream_stream_physical_bytes,
    }
    assert len(calls) >= 3
    assert all(call == expected for call in calls)
