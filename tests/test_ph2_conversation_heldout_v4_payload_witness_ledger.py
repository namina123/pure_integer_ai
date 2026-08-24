"""P3-A2 payload witness ledger 的 test-transport 合同。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger as ledger_module
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeCounts,
)
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    ConversationHeldOutV4PayloadWitness,
    ConversationHeldOutV4PayloadWitnessBudget,
    ConversationHeldOutV4PayloadWitnessInput,
    ConversationHeldOutV4PayloadWitnessLedgerError,
    V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES,
    V4_PAYLOAD_WITNESS_ENCODING_UTF8_SCALARS,
    V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY,
    load_v4_payload_witness_publication,
    materialize_v4_payload_witness_ledger,
    revalidate_v4_payload_witness_ledger,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.k_run_boundary import (
    create_new_run_root,
    write_exclusive_bytes,
)

# 复用既有的真实 P3-A -> P2 -> P3-B fixture builder，使 P3-A2 专项聚焦，且每个
# 负例不必重新运行 P2。
from test_ph2_conversation_heldout_v4_active_roster_readback import (  # noqa: E402
    _read_active_mappings,
    _request as _p3b_request,
)


_FROZEN_SOURCE_MANIFEST_IDENTITY = ProtocolKey((3001, 23_001))


def _protocol(value: int) -> ProtocolKey:
    return ProtocolKey((value, value + 20_000))


def _budget() -> ConversationHeldOutV4PayloadWitnessBudget:
    return ConversationHeldOutV4PayloadWitnessBudget(
        32, 32 * 1024, 128 * 1024, 256 * 1024, 32 * 1024,
        128 * 1024, 32 * 1024, 256 * 1024, 512 * 1024, 128 * 1024,
    )


@pytest.fixture(scope="module")
def p3_chain(tmp_path_factory):
    """本 module 只物化一次真实 test-only P3-A/P3-B pair。"""
    root = tmp_path_factory.mktemp("p3a2-chain")
    request, *_ = _p3b_request(root)
    from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
        materialize_v4_active_roster_bidirectional_readback,
    )
    return request.active_intake, materialize_v4_active_roster_bidirectional_readback(request)


def _payload_for_mapping(mapping) -> bytes:
    values = {
        hashlib.sha256(b"p3b-source-101").digest(): b"p3b-source-101",
        hashlib.sha256(b"p3b-source-102").digest(): b"p3b-source-102",
        hashlib.sha256(b"train-observation").digest(): b"train-observation",
        hashlib.sha256(b"held-observation").digest(): b"held-observation",
    }
    return values[bytes(mapping.content_sha256)]


def _witnesses(active) -> tuple[ConversationHeldOutV4PayloadWitness, ...]:
    """按 exact sealed mapping 顺序构造 witness，并交替两种支持编码。"""
    result = []
    for index, mapping in enumerate(_read_active_mappings(active)):
        payload = _payload_for_mapping(mapping)
        scalars = tuple(ord(item) for item in payload.decode("utf-8"))
        kwargs = dict(
            active_record_key=mapping.record_key,
            record_kind=mapping.record_kind,
            source_ref=mapping.source_ref,
            leaf=mapping.leaf,
            snapshot=mapping.snapshot,
            split=mapping.split,
            license_partition=mapping.license_partition,
            consumed_input_identity=mapping.leaf.consumed_input_identity,
            source_uri_scalars=mapping.snapshot.official_uri_scalars,
            source_span_byte_start=0,
            source_span_byte_end=len(payload),
            source_span_scalar_start=0,
            source_span_scalar_end=len(scalars),
            frozen_source_manifest_identity=_FROZEN_SOURCE_MANIFEST_IDENTITY,
            source_payload_identity=_protocol(1000 + index),
        )
        if index % 2:
            result.append(ConversationHeldOutV4PayloadWitness(
                encoding=V4_PAYLOAD_WITNESS_ENCODING_UTF8_SCALARS,
                content_scalars=scalars, **kwargs))
        else:
            result.append(ConversationHeldOutV4PayloadWitness(
                encoding=V4_PAYLOAD_WITNESS_ENCODING_UTF8_BYTES,
                content_bytes=payload, **kwargs))
    return tuple(result)


def _input(tmp_path: Path, active, readback, factory) -> ConversationHeldOutV4PayloadWitnessInput:
    root = lambda name: create_new_run_root(  # noqa: E731
        tmp_path / name, require_k_drive=False, label=f"P3-A2 {name}")
    return ConversationHeldOutV4PayloadWitnessInput(
        active, readback, _FROZEN_SOURCE_MANIFEST_IDENTITY, _protocol(3002), factory,
        root("staging"), root("work"), root("publication"), _budget(), "p3-a2-test",
    )


def test_payload_witness_ledger_closes_real_p3a_p3b_without_payload_publication(
        tmp_path, p3_chain):
    active, readback = p3_chain
    witnesses = _witnesses(active)

    request = _input(tmp_path, active, readback, lambda: witnesses)
    result = materialize_v4_payload_witness_ledger(request)

    assert result.receipt.status == V4_PAYLOAD_WITNESS_STATUS_TEST_ONLY
    assert result.receipt.active_intake_stable_key == active.stable_key()
    assert result.receipt.readback_stable_key == readback.stable_key()
    assert result.receipt.counts.mapping_count == len(witnesses)
    assert revalidate_v4_payload_witness_ledger(result) == result
    publications = {
        path.relative_to(result.publication_run_root.path).as_posix(): path.read_bytes()
        for path in result.publication_run_root.path.rglob("*") if path.is_file()
    }
    assert set(publications) == {"witness/receipt.pifrs", "manifest.pii"}
    for root in (request.staging_run_root, request.work_run_root,
                 result.publication_run_root):
        stored = b"".join(
            path.read_bytes() for path in root.path.rglob("*") if path.is_file())
        assert b"p3b-source-101" not in stored
        assert b"train-observation" not in stored
        assert str(tmp_path).encode() not in stored


def test_payload_witness_publication_loader_recovers_only_receipt_records_without_payload(
        tmp_path, p3_chain, monkeypatch):
    active, readback = p3_chain
    result = materialize_v4_payload_witness_ledger(
        _input(tmp_path, active, readback, lambda: _witnesses(active)))

    def must_not_access_payload(_self):
        raise AssertionError("P3-A2 publication loader accessed raw payload")

    monkeypatch.setattr(
        ledger_module.ConversationHeldOutV4PayloadWitness,
        "canonical_bytes_and_scalars",
        must_not_access_payload,
    )
    publication = load_v4_payload_witness_publication(result)

    assert publication.result == result
    assert len(publication.records) == result.receipt.counts.witness_count
    assert all(not hasattr(record, "content_bytes") for record in publication.records)
    assert all(not hasattr(record, "content_scalars") for record in publication.records)
    assert revalidate_v4_payload_witness_ledger(result) == result


def test_payload_witness_publication_loader_rejects_tampered_receipt_before_exposure(
        tmp_path, p3_chain):
    active, readback = p3_chain
    result = materialize_v4_payload_witness_ledger(
        _input(tmp_path, active, readback, lambda: _witnesses(active)))
    receipt_path = result.publication_run_root.path / "witness" / "receipt.pifrs"
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.write_bytes(receipt_bytes[:-1] + bytes((receipt_bytes[-1] ^ 1,)))

    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError):
        load_v4_payload_witness_publication(result)


def test_payload_witness_loader_rejects_caller_budget_before_opening_receipt(
        tmp_path, p3_chain, monkeypatch):
    """下游较小的 full-ledger 上限必须在打开 P0 文件前 fail closed。"""
    active, readback = p3_chain
    result = materialize_v4_payload_witness_ledger(
        _input(tmp_path, active, readback, lambda: _witnesses(active)))

    def must_not_open(*_args, **_kwargs):
        raise AssertionError("caller record preflight must run before receipt open")

    monkeypatch.setattr(ledger_module, "open_plain_binary", must_not_open)
    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError,
                       match="caller record budget"):
        load_v4_payload_witness_publication(result, max_record_count=3)


def test_payload_witness_loader_rejects_caller_physical_budget_before_opening(
        tmp_path, p3_chain, monkeypatch):
    """真实文件长度也必须在 reader 接管前服从调用方物理字节预算。"""
    active, readback = p3_chain
    result = materialize_v4_payload_witness_ledger(
        _input(tmp_path, active, readback, lambda: _witnesses(active)))

    def must_not_open(*_args, **_kwargs):
        raise AssertionError("caller physical preflight must run before receipt open")

    monkeypatch.setattr(ledger_module, "open_plain_binary", must_not_open)
    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError,
                       match="caller physical budget"):
        load_v4_payload_witness_publication(
            result,
            max_physical_bytes=result.receipt_stream.physical.byte_count - 1,
        )


@pytest.mark.parametrize("variant", (
    "missing", "extra", "out_of_order", "hash", "uri", "span", "manifest", "total",
))
def test_payload_witness_ledger_rejects_compact_binding_failures(
        tmp_path, p3_chain, variant):
    active, readback = p3_chain
    witnesses = list(_witnesses(active))
    if variant == "missing":
        candidate = lambda: tuple(witnesses[:-1])
    elif variant == "extra":
        candidate = lambda: tuple((*witnesses, witnesses[-1]))
    elif variant == "out_of_order":
        candidate = lambda: tuple((witnesses[1], witnesses[0], *witnesses[2:]))
    elif variant == "hash":
        original = witnesses[0]
        witnesses[0] = replace(original, content_bytes=b"different-content")
        candidate = lambda: tuple(witnesses)
    elif variant == "uri":
        original = witnesses[0]
        candidate = lambda: tuple((replace(
            original, source_uri_scalars=tuple(map(ord, "https://invalid.example/"))),
            *witnesses[1:]))
    else:
        if variant == "manifest":
            original = witnesses[0]
            candidate = lambda: tuple((replace(
                original, frozen_source_manifest_identity=_protocol(3999)), *witnesses[1:]))
        elif variant == "total":
            candidate = lambda: tuple(witnesses)
        else:
            original = witnesses[0]
            candidate = lambda: tuple((replace(
                original, source_span_byte_end=original.source_span_byte_end - 1),
                *witnesses[1:]))

    request = _input(tmp_path, active, readback, candidate)
    if variant == "total":
        request = replace(request, budget=replace(request.budget, max_total_content_bytes=1))
    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError):
        materialize_v4_payload_witness_ledger(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_payload_witness_ledger_rejects_teacher_before_factory_or_payload_reads(
        tmp_path, p3_chain, monkeypatch):
    active, readback = p3_chain
    teacher_active = replace(
        active,
        counts=ConversationHeldOutV4ActiveIntakeCounts(
            active.counts.lineage_node_count, active.counts.source_record_count,
            active.counts.observation_record_count, 1),
    )
    called = False

    def must_not_call():
        nonlocal called
        called = True
        raise AssertionError("teacher guard called the witness factory")

    # 这里验证 P3-A2 在 upstream readback 成功后的先后顺序；伪造的 teacher count
    # 本身不能通过真实 P3-A manifest 回验。
    monkeypatch.setattr(ledger_module, "revalidate_v4_active_intake", lambda _: teacher_active)
    teacher_readback = replace(
        readback,
        receipt=replace(readback.receipt,
                        active_intake_stable_key=teacher_active.stable_key()),
    )
    monkeypatch.setattr(ledger_module, "revalidate_v4_active_roster_bidirectional_readback",
                        lambda _: teacher_readback)
    request = _input(tmp_path, teacher_active, teacher_readback, must_not_call)

    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError, match="teacher"):
        materialize_v4_payload_witness_ledger(request)
    assert not called
    assert not any(request.publication_run_root.path.iterdir())


def test_payload_witness_ledger_requires_fresh_output_root(tmp_path, p3_chain):
    active, readback = p3_chain
    request = _input(tmp_path, active, readback, lambda: _witnesses(active))
    write_exclusive_bytes(request.publication_run_root, "residue.pii", b"residue",
                          label="P3-A2 test residue")

    with pytest.raises(ConversationHeldOutV4PayloadWitnessLedgerError, match="root"):
        materialize_v4_payload_witness_ledger(request)


def test_payload_witness_ledger_module_has_no_runtime_capsule_or_private_boundary():
    source_path = Path(ledger_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    imports = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name for name in imports for token in (
        "runtime", "candidate", "external_input_capsule", "runtime_artifact",
        "private", "formal", "owner", "trainer", "sqlite"))
    assert "tempfile" not in source
