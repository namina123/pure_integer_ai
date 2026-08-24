"""P3-C1a sealed annotation witness manifest 的有界测试。"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation_provenance as provenance_module
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation import (
    ConversationHeldOutV4RuntimeTaskAnnotationDeclaration,
    V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation_provenance import (
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget,
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput,
    ConversationHeldOutV4RuntimeTaskAnnotationWitness,
    V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY,
    load_v4_runtime_task_annotation_provenance_publication,
    materialize_v4_runtime_task_annotation_provenance,
    revalidate_v4_runtime_task_annotation_provenance,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    materialize_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeMapping,
)
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    load_v4_payload_witness_publication,
    materialize_v4_payload_witness_ledger,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.k_run_boundary import create_new_run_root, write_exclusive_bytes

from test_ph2_conversation_heldout_v4_active_roster_readback import (  # noqa: E402
    _request as _p3b_request,
)
from test_ph2_conversation_heldout_v4_payload_witness_ledger import (  # noqa: E402
    _input as _p3a2_input,
    _witnesses as _p3a2_witnesses,
)


_NAMESPACE = (20260822, 405, 320)


def _key(value: int) -> ProtocolKey:
    """构造本专项独立的严格整数 identity。"""
    return ProtocolKey((*_NAMESPACE, value))


def _declaration() -> ConversationHeldOutV4RuntimeTaskAnnotationDeclaration:
    """产生完整 test-authored annotation declaration。"""
    return ConversationHeldOutV4RuntimeTaskAnnotationDeclaration(
        _key(1), V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
        _key(2), _key(3), _key(4), _key(5), _key(6), _key(7),
    )


def _budget() -> ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget:
    """固定小型无正文 annotation manifest 的资源上限。"""
    return ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget(
        16, 8, 32 * 1024, 128 * 1024, 256 * 1024, 64 * 1024,
    )


@pytest.fixture(scope="module")
def p3a2_ledger(tmp_path_factory):
    """物化一次真实 test-only P3-A -> P3-B -> P3-A2 public 链。"""
    root = tmp_path_factory.mktemp("p3c1a-chain")
    request, *_ = _p3b_request(root)
    readback = materialize_v4_active_roster_bidirectional_readback(request)
    ledger = materialize_v4_payload_witness_ledger(_p3a2_input(
        root, request.active_intake, readback, lambda: _p3a2_witnesses(request.active_intake)))
    return ledger


def _annotation_witnesses(ledger, declaration=None):
    """从 P3-A2 public receipt 取 train/held-out 各一条，独立生成 manifest witness。"""
    declaration = _declaration() if declaration is None else declaration
    records = load_v4_payload_witness_publication(ledger).records
    selected = []
    for split in ("train", "held_out"):
        selected.append(next(
            record for record in records
            if ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
                record.mapping_identity).split == split))
    return tuple(
        ConversationHeldOutV4RuntimeTaskAnnotationWitness(
            _key(20 + ordinal), _key(40 + ordinal), ordinal, record.integer_stream(),
            (*_NAMESPACE, 100 + ordinal), declaration,
        )
        for ordinal, record in enumerate(selected, start=1)
    )


def _input(tmp_path: Path, ledger, factory):
    """构造 annotation stage 的 fresh、disjoint test transport root。"""
    root = lambda name: create_new_run_root(  # noqa: E731
        tmp_path / name, require_k_drive=False, label=f"P3-C1a {name}")
    return ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput(
        ledger, factory, root("staging"), root("work"), root("publication"),
        _budget(), "p3-c1a-test",
    )


def test_annotation_provenance_seals_independent_p3a2_bound_witnesses(p3a2_ledger, tmp_path):
    """每条 annotation 必须独立绑定 P3-A2 witness 与 typed-task identity。"""
    witnesses = _annotation_witnesses(p3a2_ledger)
    result = materialize_v4_runtime_task_annotation_provenance(
        _input(tmp_path, p3a2_ledger, lambda: witnesses))
    publication = load_v4_runtime_task_annotation_provenance_publication(result)

    assert result.receipt.status == V4_RUNTIME_TASK_ANNOTATION_PROVENANCE_STATUS_TEST_ONLY
    assert len(publication.records) == 2
    assert revalidate_v4_runtime_task_annotation_provenance(result) == result
    assert {
        path.relative_to(result.publication_run_root.path).as_posix()
        for path in result.publication_run_root.path.rglob("*") if path.is_file()
    } == {"annotation/receipt.pifrs", "manifest.pii"}


def test_annotation_provenance_hides_adversarial_protocol_key_scalars(p3a2_ledger, tmp_path):
    """即使 key/task identity 可读，三级 output 也只能持有摘要。"""
    case_leak = b"annotation-case-leak"
    task_leak = b"annotation-task-leak"
    declaration_leak = b"annotation-declaration-leak"
    witnesses = list(_annotation_witnesses(p3a2_ledger))
    witnesses[0] = replace(
        witnesses[0],
        case_key=ProtocolKey(tuple(ord(item) for item in case_leak.decode())),
        typed_task_identity=tuple(ord(item) for item in task_leak.decode()),
        declaration=replace(
            witnesses[0].declaration,
            declaration_identity=ProtocolKey(
                tuple(ord(item) for item in declaration_leak.decode()))),
    )
    request = _input(tmp_path, p3a2_ledger, lambda: tuple(witnesses))
    result = materialize_v4_runtime_task_annotation_provenance(request)
    stored = b"".join(
        path.read_bytes()
        for root in (request.staging_run_root.path, request.work_run_root.path,
                     result.publication_run_root.path)
        for path in root.rglob("*") if path.is_file())
    assert case_leak not in stored
    assert task_leak not in stored
    assert declaration_leak not in stored


def test_annotation_provenance_rejects_replaced_witness_or_p3a2_record(p3a2_ledger, tmp_path):
    """事后替换 typed task 或指向不存在的 P3-A2 record 都必须失败。"""
    witnesses = _annotation_witnesses(p3a2_ledger)
    result = materialize_v4_runtime_task_annotation_provenance(
        _input(tmp_path, p3a2_ledger, lambda: witnesses))
    drifted = replace(result, witnesses=(
        replace(result.witnesses[0], typed_task_identity=(*_NAMESPACE, 999)),
        result.witnesses[1],
    ))
    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
                       match="annotation witness"):
        revalidate_v4_runtime_task_annotation_provenance(drifted)

    orphan_parent = tmp_path / "orphan"
    orphan_parent.mkdir()
    request = _input(orphan_parent, p3a2_ledger, lambda: (
        replace(witnesses[0], witness_record_identity=(*_NAMESPACE, 998)), witnesses[1]))
    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
                       match="current P3-A2"):
        materialize_v4_runtime_task_annotation_provenance(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_annotation_provenance_rejects_published_or_staging_receipt_replacement(
        p3a2_ledger, tmp_path, monkeypatch):
    """physical identity 必须在 relay 前和 loader 时阻断替换。"""
    witnesses = _annotation_witnesses(p3a2_ledger)
    result = materialize_v4_runtime_task_annotation_provenance(
        _input(tmp_path, p3a2_ledger, lambda: witnesses))
    path = result.publication_run_root.path / "annotation" / "receipt.pifrs"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
                       match="readback|identity"):
        load_v4_runtime_task_annotation_provenance_publication(result)

    relay_parent = tmp_path / "relay"
    relay_parent.mkdir()
    request = _input(relay_parent, p3a2_ledger, lambda: witnesses)
    original = provenance_module._read

    def corrupt_staging(root, identity, **kwargs):
        if root == request.staging_run_root:
            receipt = root.path / identity.relative_path
            receipt.write_bytes(receipt.read_bytes() + b"\x00")
        return original(root, identity, **kwargs)

    monkeypatch.setattr(provenance_module, "_read", corrupt_staging)
    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
                       match="readback|identity"):
        materialize_v4_runtime_task_annotation_provenance(request)
    assert not any(request.work_run_root.path.iterdir())
    assert not any(request.publication_run_root.path.iterdir())


def test_annotation_provenance_rejects_preexisting_output_and_raw_key_input(p3a2_ledger, tmp_path):
    """factory 前要求 fresh root，入口不接受裸 ProtocolKey。"""
    witnesses = _annotation_witnesses(p3a2_ledger)
    request = _input(tmp_path, p3a2_ledger, lambda: witnesses)
    write_exclusive_bytes(request.publication_run_root, "residue.pii", b"residue",
                          label="P3-C1a residue")
    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError,
                       match="root"):
        materialize_v4_runtime_task_annotation_provenance(request)
    with pytest.raises(TypeError, match="input types"):
        ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput(
            _key(99), lambda: witnesses, request.staging_run_root,
            request.work_run_root, request.publication_run_root, _budget(), "p3-c1a-test")


def test_annotation_provenance_rejects_full_p3a2_ledger_before_factory(
        p3a2_ledger, tmp_path):
    """C1a output 可以是子集，但顺序 P3-A2 输入账本必须先满足独立读取上限。"""
    request = _input(
        tmp_path,
        p3a2_ledger,
        lambda: (_ for _ in ()).throw(
            AssertionError("P3-A2 preflight must run before annotation factory")),
    )
    request = replace(
        request,
        budget=replace(
            request.budget,
            max_upstream_payload_witness_record_count=3,
        ),
    )

    with pytest.raises(ConversationHeldOutV4RuntimeTaskAnnotationProvenanceError):
        materialize_v4_runtime_task_annotation_provenance(request)
    for root in (request.staging_run_root, request.work_run_root,
                 request.publication_run_root):
        assert not any(root.path.iterdir())


def test_annotation_provenance_module_stays_outside_payload_capsule_runtime_and_private_boundaries():
    """P3-C1a 只能消费 P3-A2 public receipt 和整数/K-run 基础设施。"""
    source = Path(provenance_module.__file__).read_text(encoding="utf-8")
    imports = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name for name in imports for token in (
        "external_input_capsule", "candidate_runtime", "runtime_artifact", "trainer",
        "sqlite", "tempfile", "private", "formal", "owner", "canonical_bytes_and_scalars",
    ))
