"""DLG-05 v4 P3-C1b 无载荷运行时任务蓝图专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_blueprint as blueprint_module
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_intake import (
    ConversationHeldOutV4ActiveIntakeMapping,
)
from pure_integer_ai.experiments.conversation_heldout_v4_active_roster_readback import (
    materialize_v4_active_roster_bidirectional_readback,
)
from pure_integer_ai.experiments.conversation_heldout_v4_payload_witness_ledger import (
    ConversationHeldOutV4PayloadWitness,
    ConversationHeldOutV4PayloadWitnessPublication,
    load_v4_payload_witness_publication,
    materialize_v4_payload_witness_ledger,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_blueprint import (
    ConversationHeldOutV4RuntimeTaskBlueprintBudget,
    ConversationHeldOutV4RuntimeTaskBlueprintError,
    ConversationHeldOutV4RuntimeTaskBlueprintInput,
    ConversationHeldOutV4RuntimeTaskBlueprintTurn,
    V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
    V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY,
    load_v4_runtime_task_blueprint_publication,
    materialize_v4_runtime_task_blueprint,
    revalidate_v4_runtime_task_blueprint,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation import (
    ConversationHeldOutV4RuntimeTaskAnnotationDeclaration,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_annotation_provenance import (
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget,
    ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput,
    ConversationHeldOutV4RuntimeTaskAnnotationWitness,
    materialize_v4_runtime_task_annotation_provenance,
)
from pure_integer_ai.experiments.conversation_heldout_v4_runtime_task_structure import (
    ConversationHeldOutV4BlueprintEvidencePlan,
    ConversationHeldOutV4BlueprintRepresentation,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4Representation,
    unicode_scalars,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.k_run_boundary import create_new_run_root, write_exclusive_bytes

# 复用真实 P3-A -> P2 -> P3-B fixture builder；P3-C1b 测试只负责将它的公开
# P3-A2 witness 连接到显式 authored typed task，绝不复用 runtime/capsule sample。
from test_ph2_conversation_heldout_v4_active_roster_readback import (  # noqa: E402
    _request as _p3b_request,
)
from test_ph2_conversation_heldout_v4_payload_witness_ledger import (  # noqa: E402
    _FROZEN_SOURCE_MANIFEST_IDENTITY,
    _input as _p3a2_input,
    _witnesses,
)


_NAMESPACE = (20260822, 405, 321)
_ANNOTATION_MANIFEST = ProtocolKey((*_NAMESPACE, 1))
_CODE_IDENTITY = ProtocolKey((*_NAMESPACE, 2))


def _protocol(value: int) -> ProtocolKey:
    """构造本专项的显式 annotation/domain identity。"""
    return ProtocolKey((*_NAMESPACE, value))


_ANNOTATION_DECLARATION = ConversationHeldOutV4RuntimeTaskAnnotationDeclaration(
    _protocol(3),
    V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
    _protocol(20),
    _protocol(21),
    _ANNOTATION_MANIFEST,
    _CODE_IDENTITY,
    _protocol(22),
    _protocol(23),
)


def _budget() -> ConversationHeldOutV4RuntimeTaskBlueprintBudget:
    """为双 split、两 turn 的无正文 receipt 固定有限资源上限。"""
    return ConversationHeldOutV4RuntimeTaskBlueprintBudget(
        16, 8, 8, 8, 64 * 1024, 256 * 1024, 512 * 1024, 128 * 1024,
    )


def _annotation_provenance(tmp_path: Path, ledger, turns):
    """独立封存每个 typed turn 的 annotation witness，P3-C1b 不能直接接收裸对象。"""
    parent = tmp_path / "annotation-provenance"
    parent.mkdir()
    root = lambda name: create_new_run_root(  # noqa: E731
        parent / name, require_k_drive=False, label=f"P3-C1a {name}")
    budget = ConversationHeldOutV4RuntimeTaskAnnotationProvenanceBudget(
        16, 16, 32 * 1024, 64 * 1024, 128 * 1024, 64 * 1024,
    )
    annotation_witnesses = tuple(
        ConversationHeldOutV4RuntimeTaskAnnotationWitness(
            turn.case_key, turn.turn_key, turn.ordinal,
            turn.witness_record_identity, turn.stable_key(), _ANNOTATION_DECLARATION)
        for turn in turns)
    return materialize_v4_runtime_task_annotation_provenance(
        ConversationHeldOutV4RuntimeTaskAnnotationProvenanceInput(
            ledger, lambda: annotation_witnesses,
            root("staging"), root("work"), root("publication"), budget,
            "p3-c1a-test",
        ))


@pytest.fixture(scope="module")
def p3a2_chain(tmp_path_factory):
    """只物化一次真实 test-only P3-A/P3-B/P3-A2 连续公开结果。"""
    root = tmp_path_factory.mktemp("p3c1b-chain")
    request, *_ = _p3b_request(root)
    readback = materialize_v4_active_roster_bidirectional_readback(request)
    active = request.active_intake
    ledger = materialize_v4_payload_witness_ledger(_p3a2_input(
        root, active, readback, lambda: _witnesses(active)))
    return active, readback, ledger


def _request(source, token: int) -> QuestionRequest:
    """按 P3-A2 SourceRef 构造 test-authored typed request，anchor 固定在完整 witness 首 scalar。"""
    boundary = {"owner": source.owner, "versions": source.versions}
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (*_NAMESPACE, 10, token)),
        concept_identity((*_NAMESPACE, 11, token), **boundary),
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (*_NAMESPACE, 12, token)),
        (),
    )
    graph = PropositionTemplateGraph((ScopedPropositionTemplate(
        definition,
        structure_concept_identity((*_NAMESPACE, 13, token), **boundary),
    ),))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity(
            (*_NAMESPACE, 14, token, index), **boundary)
        for index in range(1, 10)
    ))
    target = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((*_NAMESPACE, 15, token), **boundary),
        failures,
    )).substitute(definition.proposition, graph, BindingEnvironment())
    evidence_scope = document_scope(source)
    return QuestionRequest(
        minimal_instruction_identity((*_NAMESPACE, 16, token, 1), **boundary),
        minimal_instruction_identity((*_NAMESPACE, 16, token, 2), **boundary),
        minimal_instruction_identity((*_NAMESPACE, 16, token, 3), **boundary),
        target,
        LogicEvidenceState(True, False),
        evidence_scope,
        query_scope(token, parent=episode_scope(token, parent=evidence_scope)),
        (*_NAMESPACE, 17, token),
        language_branch_identity((*_NAMESPACE, 18, token), **boundary),
    )


def _turn(witness, ordinal: int, token: int) -> ConversationHeldOutV4RuntimeTaskBlueprintTurn:
    """由一条公开无正文 witness 与明确 test annotation 构造内存 blueprint turn。"""
    mapping = ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
        witness.mapping_identity)
    request = _request(witness.source_ref, token)
    question_scalars = unicode_scalars(f"blueprint-question-{token}", allow_empty=False)
    representation = ConversationHeldOutV4BlueprintRepresentation(
        ConversationHeldOutV4Representation(
            representation_identity(
                (*_NAMESPACE, 19, token),
                question_scalars,
                owner=witness.source_ref.owner,
                versions=witness.source_ref.versions,
            ),
            0,
            question_scalars,
        ),
        witness.source_span_scalar_start,
        witness.source_span_scalar_end,
        request.response_scope,
        _protocol(20),
        _protocol(21),
        _protocol(22),
    )
    plan = ConversationHeldOutV4BlueprintEvidencePlan(
        request.target,
        (*_NAMESPACE, 23, token),
        (EVIDENCE_SUPPORT,),
        witness.source_ref,
        witness.source_span_scalar_start,
        witness.source_span_scalar_end,
    )
    return ConversationHeldOutV4RuntimeTaskBlueprintTurn(
        _protocol(24 + token),
        _protocol(40 + token),
        ordinal,
        witness.integer_stream(),
        witness.source_ref,
        witness.content_sha256,
        witness.content_byte_count,
        witness.content_scalar_count,
        witness.source_span_byte_start,
        witness.source_span_byte_end,
        witness.source_span_scalar_start,
        witness.source_span_scalar_end,
        mapping.split,
        mapping.license_partition,
        V4_RUNTIME_TASK_BLUEPRINT_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED,
        _ANNOTATION_MANIFEST,
        _CODE_IDENTITY,
        _protocol(20),
        _protocol(21),
        _protocol(22),
        _protocol(23),
        request,
        (representation,),
        (plan,),
    )


def _selected_witnesses(ledger):
    """从公开 P3-A2 receipt 取一个 train 和一个 held-out witness，而非读取正文 payload。"""
    records = load_v4_payload_witness_publication(ledger).records
    selected = []
    for split in ("train", "held_out"):
        selected.append(next(
            record for record in records
            if ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
                record.mapping_identity).split == split
        ))
    return tuple(selected)


def _input(tmp_path: Path, active, readback, ledger, factory, *, annotation_turns=None):
    """构造三个 fresh、disjoint test-transport P3-C1b roots。"""
    if annotation_turns is None:
        annotation_turns = tuple(
            _turn(item, ordinal, ordinal) for ordinal, item in enumerate(
                _selected_witnesses(ledger), start=1))
    root = lambda name: create_new_run_root(  # noqa: E731
        tmp_path / name, require_k_drive=False, label=f"P3-C1b {name}")
    return ConversationHeldOutV4RuntimeTaskBlueprintInput(
        active, readback, ledger,
        _annotation_provenance(tmp_path, ledger, annotation_turns), factory,
        root("staging"), root("work"), root("publication"), _budget(),
        "p3-c1b-test",
    )


def test_runtime_task_blueprint_closes_train_and_held_out_without_public_typed_content(
        tmp_path, p3a2_chain, monkeypatch):
    """P3-C1b 只能发布摘要 receipt，且 P3-A2 回读期间不允许触碰 raw payload。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))

    def no_payload(_self):
        raise AssertionError("P3-C1b attempted to access raw payload")

    monkeypatch.setattr(ConversationHeldOutV4PayloadWitness,
                        "canonical_bytes_and_scalars", no_payload)
    request = _input(tmp_path, active, readback, ledger, lambda: turns)
    result = materialize_v4_runtime_task_blueprint(request)

    assert result.receipt.status == V4_RUNTIME_TASK_BLUEPRINT_STATUS_TEST_ONLY
    assert result.receipt.counts.integer_stream() == (2, 1, 1, 2, 2)
    assert result.turns == turns
    assert revalidate_v4_runtime_task_blueprint(result) == result
    publications = {
        path.relative_to(result.publication_run_root.path).as_posix(): path.read_bytes()
        for path in result.publication_run_root.path.rglob("*") if path.is_file()
    }
    assert set(publications) == {"blueprint/receipt.pifrs", "manifest.pii"}
    for root in (request.staging_run_root, request.work_run_root,
                 result.publication_run_root):
        stored = b"".join(
            path.read_bytes() for path in root.path.rglob("*") if path.is_file())
        assert b"p3b-source-101" not in stored
        assert b"blueprint-question-1" not in stored
        assert str(tmp_path).encode() not in stored


@pytest.mark.parametrize("variant", (
    "witness", "split", "license", "ordinal", "duplicate", "annotation", "anchor",
))
def test_runtime_task_blueprint_rejects_compact_identity_and_structure_drift(
        tmp_path, p3a2_chain, variant):
    """P3-A2 binding、split/license、annotation、case/turn 和 anchor 漂移必须零发布拒绝。"""
    active, readback, ledger = p3a2_chain
    train, held_out = _selected_witnesses(ledger)
    first, second = _turn(train, 1, 1), _turn(held_out, 2, 2)
    if variant == "witness":
        first = replace(first, witness_record_identity=held_out.integer_stream())
    elif variant == "split":
        first = replace(first, split="held_out")
    elif variant == "license":
        first = replace(first, license_partition="INVALID-LICENSE")
    elif variant == "ordinal":
        second = replace(second, ordinal=3)
    elif variant == "duplicate":
        second = replace(second, case_key=first.case_key, turn_key=first.turn_key)
    elif variant == "annotation":
        first = replace(first, applicable_domain_identity=_protocol(999))
    else:
        bad_anchor = occurrence_identity(
            first.source_ref,
            start=first.source_span_scalar_start,
            end=first.source_span_scalar_end + 1,
            ordinal=0,
        )
        bad_target = replace(first.request.target, source_anchor=bad_anchor)
        first = replace(first, request=replace(first.request, target=bad_target))
        first = replace(first, evidence_plans=(replace(
            first.evidence_plans[0], target=bad_target),))
    request = _input(tmp_path, active, readback, ledger, lambda: (first, second))
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError):
        materialize_v4_runtime_task_blueprint(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_blueprint_rejects_factory_turn_not_bound_to_sealed_annotation_witness(
        tmp_path, p3a2_chain):
    """factory turn 必须与预封存 annotation witness 的来源/修订/任务身份逐字段闭合。"""
    active, readback, ledger = p3a2_chain
    train, held_out = _selected_witnesses(ledger)
    first = replace(_turn(train, 1, 1), annotation_revision_identity=_protocol(999))
    request = _input(tmp_path, active, readback, ledger,
                     lambda: (first, _turn(held_out, 2, 2)))
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError,
                       match="annotation witness"):
        materialize_v4_runtime_task_blueprint(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_blueprint_limits_full_p3a2_ledger_before_turn_factory(
        tmp_path, p3a2_chain):
    """C1b 的 witness 上限必须传入 P3-A2 loader，而非在全量读取后才检查。"""
    active, readback, ledger = p3a2_chain
    request = _input(
        tmp_path,
        active,
        readback,
        ledger,
        lambda: (_ for _ in ()).throw(
            AssertionError("P3-A2 preflight must run before blueprint factory")),
    )
    request = replace(
        request,
        budget=replace(request.budget, max_witness_record_count=3),
    )

    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError):
        materialize_v4_runtime_task_blueprint(request)
    for root in (request.staging_run_root, request.work_run_root,
                 request.publication_run_root):
        assert not any(root.path.iterdir())


def test_runtime_task_blueprint_requires_exact_sealed_annotation_witness_inventory(
        tmp_path, p3a2_chain):
    """C1b 不得遗漏已封存 annotation witness，也不能以部分匹配代替完整 inventory 闭合。"""
    active, readback, ledger = p3a2_chain
    first, second = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    extra = replace(first, case_key=_protocol(901), turn_key=_protocol(902), ordinal=3)
    request = _input(
        tmp_path, active, readback, ledger, lambda: (first, second),
        annotation_turns=(first, second, extra))
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError,
                       match="annotation witness inventory"):
        materialize_v4_runtime_task_blueprint(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_blueprint_requires_fresh_publication_root(tmp_path, p3a2_chain):
    """已有 output residue 必须在 factory 调用和任何 receipt 写入前拒绝。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    request = _input(tmp_path, active, readback, ledger, lambda: turns)
    write_exclusive_bytes(request.publication_run_root, "residue.pii", b"residue",
                          label="P3-C1b test residue")
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError, match="root"):
        materialize_v4_runtime_task_blueprint(request)


def test_runtime_task_blueprint_rejects_witness_mapping_not_in_current_p3a_publication(
        tmp_path, p3a2_chain, monkeypatch):
    """可解码但不属于当前 P3-A sealed stream 的 mapping identity 不能借 P3-A2 混入。"""
    active, readback, ledger = p3a2_chain
    publication = load_v4_payload_witness_publication(ledger)
    original = publication.records[0]
    mapping = ConversationHeldOutV4ActiveIntakeMapping.from_integer_stream(
        original.mapping_identity)
    orphan_mapping = replace(mapping, owner_key=_protocol(998))
    orphan_record = replace(original, mapping_identity=orphan_mapping.integer_stream())
    forged = ConversationHeldOutV4PayloadWitnessPublication(
        publication.result, (orphan_record, *publication.records[1:]))
    turns = (_turn(orphan_record, 1, 1), _turn(_selected_witnesses(ledger)[1], 2, 2))
    monkeypatch.setattr(blueprint_module, "load_v4_payload_witness_publication",
                        lambda _, **_kwargs: forged)
    request = _input(tmp_path, active, readback, ledger, lambda: turns)
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError, match="current P3-A"):
        materialize_v4_runtime_task_blueprint(request)
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_blueprint_rejects_duplicate_p3a2_mapping_before_factory(
        tmp_path, p3a2_chain, monkeypatch):
    """P3-A2 witness inventory 必须与当前 P3-A mapping 做完整集合闭合，不能只比数量。"""
    active, readback, ledger = p3a2_chain
    publication = load_v4_payload_witness_publication(ledger)
    same_mapping_different_record = replace(
        publication.records[0], source_payload_identity=_protocol(997))
    forged = ConversationHeldOutV4PayloadWitnessPublication(
        publication.result,
        (publication.records[0], same_mapping_different_record, *publication.records[2:]),
    )
    factory_called = False

    def must_not_call():
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run after inventory mismatch")

    monkeypatch.setattr(blueprint_module, "load_v4_payload_witness_publication",
                        lambda _, **_kwargs: forged)
    request = _input(tmp_path, active, readback, ledger, must_not_call)
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError, match="inventory set"):
        materialize_v4_runtime_task_blueprint(request)
    assert not factory_called
    for root in (request.staging_run_root, request.work_run_root,
                 request.publication_run_root):
        assert not any(root.path.iterdir())


def test_runtime_task_blueprint_verifies_staging_receipt_before_work_relay(
        tmp_path, p3a2_chain, monkeypatch):
    """staging P0 被替换后，work/publication 不得接收任何未完成 physical 回读的 record。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    request = _input(tmp_path, active, readback, ledger, lambda: turns)
    original = blueprint_module._iter_receipt_records

    def corrupt_staging(root, identity, **kwargs):
        if root == request.staging_run_root:
            path = root.path / identity.relative_path
            path.write_bytes(path.read_bytes() + b"\x00")
        return original(root, identity, **kwargs)

    monkeypatch.setattr(blueprint_module, "_iter_receipt_records", corrupt_staging)
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError, match="readback|identity"):
        materialize_v4_runtime_task_blueprint(request)
    assert not any(request.work_run_root.path.iterdir())
    assert not any(request.publication_run_root.path.iterdir())


def test_runtime_task_blueprint_hashes_adversarial_case_and_turn_protocol_keys(
        tmp_path, p3a2_chain):
    """即使 factory 把可读 scalar 塞进 ProtocolKey，公开 receipt 也只能保留域分离摘要。"""
    active, readback, ledger = p3a2_chain
    train, held_out = _selected_witnesses(ledger)
    first = replace(
        _turn(train, 1, 1),
        case_key=ProtocolKey(tuple(ord(item) for item in "case-key-leak")),
        turn_key=ProtocolKey(tuple(ord(item) for item in "turn-key-leak")),
    )
    result = materialize_v4_runtime_task_blueprint(
        _input(tmp_path, active, readback, ledger, lambda: (first, _turn(held_out, 2, 2)),
               annotation_turns=(first, _turn(held_out, 2, 2))))
    stored = b"".join(
        path.read_bytes() for path in result.publication_run_root.path.rglob("*")
        if path.is_file())
    assert b"case-key-leak" not in stored
    assert b"turn-key-leak" not in stored


def test_runtime_task_blueprint_revalidation_rejects_replaced_in_memory_turn(
        tmp_path, p3a2_chain):
    """未来 P3-C1c 消费的内存 turn 也必须仍精确绑定已发布 receipt。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    result = materialize_v4_runtime_task_blueprint(
        _input(tmp_path, active, readback, ledger, lambda: turns))
    drifted = replace(
        result,
        turns=(replace(result.turns[0], applicable_domain_identity=_protocol(999)),
               result.turns[1]),
    )
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError, match="in-memory turn"):
        revalidate_v4_runtime_task_blueprint(drifted)


def test_runtime_task_blueprint_revalidation_rejects_replaced_sealed_annotation_witness(
        tmp_path, p3a2_chain):
    """下游不能把 P3-C1a receipt 已封存的 annotation witness 替换为另一组 identity。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    result = materialize_v4_runtime_task_blueprint(
        _input(tmp_path, active, readback, ledger, lambda: turns))
    drifted = replace(
        result,
        annotation_provenance=replace(
            result.annotation_provenance,
            witnesses=(
                replace(
                    result.annotation_provenance.witnesses[0],
                    declaration=replace(
                        result.annotation_provenance.witnesses[0].declaration,
                        annotation_revision_identity=_protocol(999))),
                result.annotation_provenance.witnesses[1],
            )),
    )
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError,
                       match="annotation provenance"):
        revalidate_v4_runtime_task_blueprint(drifted)


def test_runtime_task_blueprint_public_loader_returns_only_revalidated_receipt_records(
        tmp_path, p3a2_chain):
    """C1c 只能通过公开 loader 获取 C1b P0 records，且 loader 先重验内存 turn 绑定。"""
    active, readback, ledger = p3a2_chain
    turns = tuple(_turn(item, ordinal, ordinal) for ordinal, item in enumerate(
        _selected_witnesses(ledger), start=1))
    result = materialize_v4_runtime_task_blueprint(
        _input(tmp_path, active, readback, ledger, lambda: turns))

    publication = load_v4_runtime_task_blueprint_publication(result)

    assert publication.result == result
    assert len(publication.records) == len(result.turns)
    assert tuple(item.ordinal for item in publication.records) == (1, 2)

    drifted = replace(
        result,
        turns=(replace(result.turns[0], applicable_domain_identity=_protocol(999)),
               result.turns[1]),
    )
    with pytest.raises(ConversationHeldOutV4RuntimeTaskBlueprintError,
                       match="in-memory turn"):
        load_v4_runtime_task_blueprint_publication(drifted)


def test_runtime_task_blueprint_module_stays_outside_payload_capsule_runtime_and_private_boundaries():
    """P3-C1b 只能消费公开 P3-A/P3-B/P3-A2 与共享结构内核。"""
    source = Path(blueprint_module.__file__).read_text(encoding="utf-8")
    imports = {
        alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.ImportFrom) and node.module)
    assert not any(token in name for name in imports for token in (
        "external_input_capsule", "candidate_runtime", "runtime_artifact", "trainer",
        "sqlite", "tempfile", "private", "formal", "owner"))
    assert "canonical_bytes_and_scalars" not in source
