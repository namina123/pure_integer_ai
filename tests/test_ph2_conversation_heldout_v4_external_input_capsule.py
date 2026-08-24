"""DLG-05 v4 external input capsule 的无标签 transport 专项。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule as capsule_module
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
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
    binder_identity,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    BoundRoleBinding,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4Representation,
    ConversationHeldOutV4SourceRecord,
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    ConversationHeldOutV4RuntimeEvidencePlan,
    ConversationHeldOutV4RuntimeInput,
    V4_RUNTIME_FAMILY_KEY,
    run_v4_candidate_runtime,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalCapsuleBudget,
    ConversationHeldOutV4ExternalCapsuleError,
    ConversationHeldOutV4ExternalInputCapsule,
    ConversationHeldOutV4ExternalProducer,
    prepare_v4_external_input_capsule,
    read_budgeted_v4_external_input_capsule,
    read_v4_external_input_capsule,
    write_prepared_v4_external_input_capsule,
    write_v4_external_input_capsule,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import decode_integer_tuple


_NAMESPACE = (20260821, 405, 502)


def _digest(value: str) -> tuple[int, ...]:
    """为测试输入构造确定性 provenance digest。"""
    return tuple(hashlib.sha256(value.encode("utf-8")).digest())


def _external_draft() -> ConversationHeldOutV4ExternalInputCapsule:
    """构造一个锚点范围真实落入原文的单 turn transport 测试输入。"""
    source = SourceRef(
        _NAMESPACE[0], 50, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    raw_text = "外部事实。"
    record = ConversationHeldOutV4SourceRecord(
        source,
        unicode_scalars(raw_text),
        _digest(raw_text),
        unicode_scalars("CC-BY-4.0"),
        unicode_scalars("external capsule transport test"),
        unicode_scalars("https://example.invalid/dlg05-v4-source-1"),
    )
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (*_NAMESPACE, 1)),
        concept_identity((*_NAMESPACE, 2)),
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (*_NAMESPACE, 3)),
        (),
    )
    graph = PropositionTemplateGraph((ScopedPropositionTemplate(
        definition,
        structure_concept_identity((*_NAMESPACE, 4)),
    ),))
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((*_NAMESPACE, 5, index))
        for index in range(1, 10)
    ))
    target = PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((*_NAMESPACE, 6)), failures,
    )).substitute(definition.proposition, graph, BindingEnvironment())
    evidence_scope = document_scope(source)
    request = QuestionRequest(
        minimal_instruction_identity((*_NAMESPACE, 7, 1)),
        minimal_instruction_identity((*_NAMESPACE, 7, 2)),
        minimal_instruction_identity((*_NAMESPACE, 7, 3)),
        target,
        LogicEvidenceState(True, False),
        evidence_scope,
        query_scope(
            1,
            parent=episode_scope(1, parent=evidence_scope),
        ),
        (*_NAMESPACE, 8),
        language_branch_identity((*_NAMESPACE, 9)),
    )
    question_scalars = unicode_scalars("请核对外部事实。", allow_empty=False)
    representation = ConversationHeldOutV4Representation(
        representation_identity((*_NAMESPACE, 10), question_scalars),
        0,
        question_scalars,
    )
    evidence = ConversationHeldOutV4RuntimeEvidencePlan(
        target,
        (*_NAMESPACE, 11),
        (EVIDENCE_SUPPORT,),
        source,
        0,
        1,
    )
    runtime_input = ConversationHeldOutV4RuntimeInput(
        ProtocolKey((*_NAMESPACE, 12)),
        ProtocolKey((*_NAMESPACE, 12, 1)),
        1,
        request,
        (representation,),
        (record,),
        (evidence,),
    )
    return ConversationHeldOutV4ExternalInputCapsule(
        V4_RUNTIME_FAMILY_KEY,
        ConversationHeldOutV4ExternalProducer(ProtocolKey((*_NAMESPACE, 13))),
        ConversationHeldOutV4DependencyBinding(
            _digest("artifact"), _digest("inventory"), _digest("document")),
        (runtime_input,),
    )


def test_external_capsule_round_trips_full_typed_input_and_enters_runtime(tmp_path):
    """JSON/ints 必须完整恢复 typed input，且 read 后可进入真实无标签 runtime。"""
    draft = _external_draft()
    root = write_v4_external_input_capsule(
        tmp_path / "capsule", draft, require_k_drive=False)

    loaded = read_v4_external_input_capsule(root, require_k_drive=False)

    assert loaded.is_external
    assert loaded.dependencies == draft.dependencies
    assert loaded.inputs == draft.inputs
    assert loaded.external_producer_key == draft.producer.producer_key
    assert loaded.external_producer_declaration == draft.producer.declaration
    assert loaded.manifest_sha256 == tuple(hashlib.sha256(
        (root / "manifest.json").read_bytes()).digest())
    assert decode_integer_tuple(
        (root / "input.canonical.ints").read_bytes()) == draft.stable_key()

    document = draft.document()
    target = document["turns"][0]["request"]["target"]
    assert {"template_key", "bindings", "source_anchor"} <= set(target)
    assert target["source_anchor"]["kind"] == "occurrence"

    result = run_v4_candidate_runtime(loaded)
    assert len(result.frames) == 1
    assert result.frames[0].execution.candidates


def test_external_capsule_requires_new_k_root_and_manifest_last(tmp_path, monkeypatch):
    """生产拒绝 D 盘；失败时 manifest 不出现，成功根也永不覆盖。"""
    draft = _external_draft()
    rejected = tmp_path / "must-not-exist"
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="K 盘"):
        write_v4_external_input_capsule(rejected, draft)
    assert not rejected.exists()

    partial = tmp_path / "partial"
    original = capsule_module._write_exclusive

    def fail_before_manifest(path, payload):
        if path.name == "input.canonical.ints":
            raise ConversationHeldOutV4ExternalCapsuleError("injected write failure")
        return original(path, payload)

    monkeypatch.setattr(capsule_module, "_write_exclusive", fail_before_manifest)
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="injected"):
        write_v4_external_input_capsule(
            partial, draft, require_k_drive=False)
    assert (partial / "input_capsule.json").is_file()
    assert not (partial / "manifest.json").exists()

    root = tmp_path / "published"
    monkeypatch.setattr(capsule_module, "_write_exclusive", original)
    write_v4_external_input_capsule(root, draft, require_k_drive=False)
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="此前不存在|已存在"):
        write_v4_external_input_capsule(root, draft, require_k_drive=False)


def test_external_capsule_prepare_enforces_budget_before_new_root_creation(tmp_path):
    """C1c 可先冻结受预算的 canonical payload，超预算时不创建任何 capsule root。"""
    draft = _external_draft()
    rejected = tmp_path / "budget-rejected"
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="exceeds budget"):
        prepare_v4_external_input_capsule(
            draft, ConversationHeldOutV4ExternalCapsuleBudget(
                1, 1_000_000, 1_000_000, 2_000_000))
    assert not rejected.exists()

    prepared = prepare_v4_external_input_capsule(
        draft, ConversationHeldOutV4ExternalCapsuleBudget(
            1_000_000, 1_000_000, 1_000_000, 3_000_000))
    root = write_prepared_v4_external_input_capsule(
        tmp_path / "prepared", prepared, require_k_drive=False)

    assert read_v4_external_input_capsule(root, require_k_drive=False).inputs == draft.inputs
    assert (root / "input_capsule.json").read_bytes() == prepared.input_payload
    assert (root / "input.canonical.ints").read_bytes() == prepared.integer_payload
    assert (root / "manifest.json").read_bytes() == prepared.manifest_payload


def test_external_capsule_budgeted_reader_replays_v1_without_read_bytes(tmp_path, monkeypatch):
    """C1c reader 必须以预算内显式长度读取既有 v1 闭包，而非沿用 legacy read_bytes。"""
    draft = _external_draft()
    root = write_v4_external_input_capsule(
        tmp_path / "capsule", draft, require_k_drive=False)
    budget = ConversationHeldOutV4ExternalCapsuleBudget(
        1_000_000, 1_000_000, 1_000_000, 3_000_000)

    def deny_read_bytes(self):
        raise AssertionError("budgeted reader must not call Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", deny_read_bytes)
    loaded = read_budgeted_v4_external_input_capsule(
        root, budget=budget, require_k_drive=False)

    assert loaded.dependencies == draft.dependencies
    assert loaded.inputs == draft.inputs
    assert loaded.external_producer_key == draft.producer.producer_key


def test_prepared_external_capsule_writeback_avoids_unbounded_read_bytes(
        tmp_path, monkeypatch):
    """prepare/write 的写后内容复核同样必须固定长度读取，而非 Path.read_bytes。"""
    draft = _external_draft()
    prepared = prepare_v4_external_input_capsule(
        draft, ConversationHeldOutV4ExternalCapsuleBudget(
            1_000_000, 1_000_000, 1_000_000, 3_000_000))

    def deny_read_bytes(self):
        raise AssertionError("prepared capsule writer must not call Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", deny_read_bytes)
    root = write_prepared_v4_external_input_capsule(
        tmp_path / "prepared-no-read-bytes", prepared, require_k_drive=False)
    assert root.is_dir()


def test_external_capsule_budgeted_reader_stops_after_manifest_budget_rejection(
        tmp_path, monkeypatch):
    """data 文件声明超预算时，只允许读取 manifest，不能先读取输入正文。"""
    draft = _external_draft()
    root = write_v4_external_input_capsule(
        tmp_path / "capsule", draft, require_k_drive=False)
    calls = []
    original = capsule_module._read_bounded_plain_bytes

    def tracked(root_path, relative, **kwargs):
        calls.append(relative)
        return original(root_path, relative, **kwargs)

    monkeypatch.setattr(capsule_module, "_read_bounded_plain_bytes", tracked)
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="input.*预算"):
        read_budgeted_v4_external_input_capsule(
            root,
            budget=ConversationHeldOutV4ExternalCapsuleBudget(
                1, 1_000_000, 1_000_000, 2_000_000),
            require_k_drive=False,
        )
    assert calls == [capsule_module._MANIFEST_FILE]


def test_external_capsule_budgeted_reader_rechecks_captured_data_identity(
        tmp_path, monkeypatch):
    """manifest 读取后替换 data 文件必须在其读取前被身份重验拦截。"""
    draft = _external_draft()
    root = write_v4_external_input_capsule(
        tmp_path / "capsule", draft, require_k_drive=False)
    original = capsule_module._read_bounded_plain_bytes

    def mutate_after_manifest(root_path, relative, **kwargs):
        payload = original(root_path, relative, **kwargs)
        if relative == capsule_module._MANIFEST_FILE:
            input_path = root_path / capsule_module._INPUT_FILE
            input_path.write_bytes(input_path.read_bytes() + b" ")
        return payload

    monkeypatch.setattr(capsule_module, "_read_bounded_plain_bytes", mutate_after_manifest)
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="文件身份漂移|文件大小漂移"):
        read_budgeted_v4_external_input_capsule(
            root,
            budget=ConversationHeldOutV4ExternalCapsuleBudget(
                1_000_000, 1_000_000, 1_000_000, 3_000_000),
            require_k_drive=False,
        )


def test_external_capsule_reader_rejects_tamper_and_stable_key_only_bound(tmp_path):
    """文件 hash、整数 payload 和完整 BoundProposition 结构任一漂移均 fail closed。"""
    draft = _external_draft()
    root = write_v4_external_input_capsule(
        tmp_path / "capsule", draft, require_k_drive=False)
    input_path = root / "input_capsule.json"
    original = input_path.read_bytes()
    input_path.write_bytes(original + b" ")
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="文件身份漂移"):
        read_v4_external_input_capsule(root, require_k_drive=False)
    input_path.write_bytes(original)

    document = deepcopy(draft.document())
    del document["turns"][0]["request"]["target"]["source_anchor"]
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="字段集合"):
        ConversationHeldOutV4ExternalInputCapsule.from_document(document)

    ints_path = root / "input.canonical.ints"
    ints_path.write_bytes(b"\x00")
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="文件身份漂移"):
        read_v4_external_input_capsule(root, require_k_drive=False)


def test_external_capsule_rejects_anchor_outside_source_scalars():
    """来源 anchor 必须可回指到保存的 raw source scalar 范围。"""
    draft = _external_draft()
    input_item = draft.inputs[0]
    source = input_item.request.source
    raw_length = len(input_item.source_records[0].raw_text_scalars)
    bad_anchor = occurrence_identity(
        source, start=0, end=raw_length + 1, ordinal=0)
    bad_target = replace(input_item.request.target, source_anchor=bad_anchor)
    bad_request = replace(input_item.request, target=bad_target)
    bad_plan = replace(input_item.evidence_plans[0], target=bad_target)
    bad_input = replace(
        input_item,
        request=bad_request,
        evidence_plans=(bad_plan,),
    )
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="anchor 越过"):
        ConversationHeldOutV4ExternalInputCapsule(
            draft.family_key, draft.producer, draft.dependencies, (bad_input,))


def test_external_capsule_rejects_cross_source_role_anchor_and_duplicate_binder():
    """角色 filler 内的 Occurrence/Span 和 Binder 也必须闭合到当前来源。"""
    draft = _external_draft()
    item = draft.inputs[0]
    source = item.request.source
    foreign_source = SourceRef(
        _NAMESPACE[0], 51, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    cross_source_target = replace(
        item.request.target,
        bindings=(BoundRoleBinding(
            role_identity((*_NAMESPACE, 14)),
            occurrence_identity(foreign_source, start=0, end=1, ordinal=0),
        ),),
    )
    cross_source_request = replace(item.request, target=cross_source_target)
    cross_source_plan = replace(
        item.evidence_plans[0], target=cross_source_target)
    cross_source_input = replace(
        item,
        request=cross_source_request,
        evidence_plans=(cross_source_plan,),
    )
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="role filler anchor 跨"):
        ConversationHeldOutV4ExternalInputCapsule(
            draft.family_key,
            draft.producer,
            draft.dependencies,
            (cross_source_input,),
        )

    duplicate_binder = binder_identity(source, (*_NAMESPACE, 15))
    duplicate_target = replace(
        item.request.target,
        introduced_binders=(duplicate_binder, duplicate_binder),
    )
    duplicate_request = replace(item.request, target=duplicate_target)
    duplicate_plan = replace(item.evidence_plans[0], target=duplicate_target)
    duplicate_input = replace(
        item,
        request=duplicate_request,
        evidence_plans=(duplicate_plan,),
    )
    with pytest.raises(ConversationHeldOutV4ExternalCapsuleError, match="Binder 不得重复"):
        ConversationHeldOutV4ExternalInputCapsule(
            draft.family_key,
            draft.producer,
            draft.dependencies,
            (duplicate_input,),
        )
