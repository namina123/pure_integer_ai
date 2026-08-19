"""DLG-05 typed input catalog 的重建与泄漏边界专项。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.experiments.conversation_heldout_family import (
    ConversationHeldOutCatalogCase,
    ConversationHeldOutCatalogError,
    ConversationHeldOutCatalogTurn,
    ConversationHeldOutInputCatalog,
    build_conversation_heldout_manifest,
    manifest_case_turn,
)
from pure_integer_ai.experiments.conversation_heldout_preflight import (
    ConversationHeldOutPreflightQuestionExecutor,
    audit_dlg05_preflight_axis_inputs,
    build_dlg05_preflight_evidence_plans,
    build_dlg05_preflight_language_compiler,
    build_dlg05_typed_preflight_catalog,
    build_dlg05_typed_preflight_manifest,
    build_dlg05_unseen_relation_compiler,
    build_dlg05_unseen_source_compiler,
)
from pure_integer_ai.experiments.conversation_heldout_language_input import (
    ConversationLanguageInputError,
    ConversationQuestionInputCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_EXPLICIT_REPEAT,
    AXIS_MEMORY_MISS,
    AXIS_EVENT_REFERENCE,
    AXIS_OMISSION,
    AXIS_ORDER,
    AXIS_PROPOSITION_REFERENCE,
    AXIS_ROLLBACK,
    AXIS_SCOPE_DRIFT,
    AXIS_SYNONYM,
    AXIS_UNSEEN_RELATION,
    AXIS_UNSEEN_SOURCE,
    CONTEXT_EXPLICIT_REPEAT,
    CONTEXT_FRESH,
    MEMORY_OFF,
    MEMORY_ON,
    ConversationHeldOutManifest,
    ConversationHeldOutTurn,
)
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionQuery
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_EVENT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.experiments.conversation_heldout_relation_input import (
    ConversationRelationInputError,
    ConversationRelationStructure,
    ConversationRelationTrainingInventory,
    ConversationUnseenRelationCompiler,
)
from pure_integer_ai.experiments.conversation_heldout_source_input import (
    ConversationSourceInputError,
    ConversationSourceTrainingInventory,
    ConversationUnseenSourceCompiler,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationContextRead,
    ConversationTurnState,
)
from test_f00_question_answer_runtime import _fixture


_FAMILY = ProtocolKey((20260819, 5, 21))
_CASE = ProtocolKey((20260819, 521, 1))
_TURN_1 = ProtocolKey((20260819, 521, 1, 1))
_TURN_2 = ProtocolKey((20260819, 521, 1, 2))


def _catalog(request, *, train_contents=()):
    """建立一组不含答案、表面的两回合 typed catalog。"""
    second = replace(request, trace=(request.trace[0], request.trace[1] + 1))
    case = ConversationHeldOutCatalogCase(
        _CASE,
        _FAMILY,
        (AXIS_MEMORY_MISS, AXIS_EXPLICIT_REPEAT),
        CanonicalIdentity.from_value(("heldout", "dedup", 1)),
        CanonicalIdentity.from_value(("heldout", "provenance", 1)),
        (
            ConversationHeldOutCatalogTurn(
                _TURN_1, 1, request, CONTEXT_FRESH, MEMORY_OFF,
                ProtocolKey((6, 0)), ProtocolKey((5, 0))),
            ConversationHeldOutCatalogTurn(
                _TURN_2, 2, second, CONTEXT_EXPLICIT_REPEAT, MEMORY_ON,
                ProtocolKey((6, 0)), ProtocolKey((5, 0))),
        ),
    )
    return ConversationHeldOutInputCatalog(
        1,
        _FAMILY,
        tuple(train_contents),
        (CanonicalIdentity.from_value(("train", "dedup", 1)),),
        (CanonicalIdentity.from_value(("train", "provenance", 1)),),
        (case,),
    )


def test_typed_catalog_rebuilds_manifest_content_source_scope_without_labels():
    """manifest 的三类身份必须逐回合来自完整 QuestionRequest。"""
    fixture = _fixture()
    try:
        catalog = _catalog(fixture.request)
        manifest = build_conversation_heldout_manifest(
            catalog,
            (AXIS_MEMORY_MISS, AXIS_EXPLICIT_REPEAT),
        )
        catalog.assert_manifest_rebuildable(manifest)
        assert catalog.request_for(_CASE, _TURN_1) is fixture.request
        assert catalog.turn_for(_CASE, _TURN_1).request is fixture.request
        assert catalog.request_for(_CASE, _TURN_2).stable_key() != fixture.request.stable_key()
        turn = manifest_case_turn(manifest, _CASE, _TURN_1)
        assert turn.content == catalog.cases[0].turns[0].content
        assert turn.source_key == catalog.cases[0].turns[0].source_key
        assert turn.scope_key == catalog.cases[0].turns[0].scope_key
        assert catalog.stable_key() == catalog.stable_key()
        assert catalog.cases[0].stable_key() == catalog.cases[0].stable_key()
        assert not hasattr(manifest, "labels")
        assert not hasattr(manifest.cases[0].turns[0], "expected")
    finally:
        fixture.close()


def test_typed_catalog_rejects_train_content_leakage():
    """完整 QuestionRequest content 进入 TRAIN 集合时必须 fail closed。"""
    fixture = _fixture()
    try:
        with pytest.raises(ConversationHeldOutCatalogError, match="content 泄漏"):
            _catalog(
                fixture.request,
                train_contents=(
                    # 先构造无泄漏 catalog，取其完整 typed content identity。
                    _catalog(fixture.request).cases[0].turns[0].content,
                ),
            )
    finally:
        fixture.close()


def test_typed_catalog_rejects_same_content_across_cases():
    """同一完整请求不能伪装成两个独立 held-out case。"""
    fixture = _fixture()
    try:
        first = _catalog(fixture.request).cases[0]
        second = replace(first, case_key=ProtocolKey((20260819, 521, 2)))
        with pytest.raises(ConversationHeldOutCatalogError, match="跨 case"):
            ConversationHeldOutInputCatalog(
                1,
                _FAMILY,
                (),
                (CanonicalIdentity.from_value(("train", "dedup", 1)),),
                (CanonicalIdentity.from_value(("train", "provenance", 1)),),
                (first, second),
            )
    finally:
        fixture.close()


def test_typed_catalog_rejects_manifest_identity_drift():
    """manifest 即使仍满足协议，也不能替换 catalog 的 request 身份。"""
    fixture = _fixture()
    try:
        catalog = _catalog(fixture.request)
        manifest = build_conversation_heldout_manifest(
            catalog,
            (AXIS_MEMORY_MISS, AXIS_EXPLICIT_REPEAT),
        )
        original = manifest.cases[0].turns[0]
        drifted_turn = ConversationHeldOutTurn(
            original.turn_key,
            original.ordinal,
            catalog.cases[0].turns[1].content,
            original.source_key,
            original.scope_key,
            original.context_mode,
            original.memory_mode,
            original.reference_mode,
            original.rollback_mode,
        )
        drifted_case = replace(manifest.cases[0], turns=(
            drifted_turn,
            manifest.cases[0].turns[1],
        ))
        drifted = ConversationHeldOutManifest(
            manifest.version,
            manifest.family_key,
            manifest.train_contents,
            manifest.train_dedup_clusters,
            manifest.train_provenance_clusters,
            (drifted_case,),
            manifest.required_axes,
            manifest.required_memory_modes,
        )
        with pytest.raises(ConversationHeldOutCatalogError, match="不是由当前"):
            catalog.assert_manifest_rebuildable(drifted)
    finally:
        fixture.close()


def test_dlg05_preflight_catalog_has_six_new_typed_cases_without_labels():
    """六 case preflight 只冻结新 typed 输入和轴，不注入 response-act/答案。"""
    catalog = build_dlg05_typed_preflight_catalog()
    # 以明确的 catalog 轴集合构造，避免测试中的推导逻辑成为生产分母。
    axes = []
    for case in catalog.cases:
        for axis in case.axis_keys:
            if axis not in axes:
                axes.append(axis)
    manifest = build_conversation_heldout_manifest(catalog, tuple(axes))
    assert len(catalog.cases) == len(manifest.cases) == 6
    assert all(len(case.turns) == 2 for case in catalog.cases)
    assert len({
        turn.request.source.stable_key()
        for case in catalog.cases
        for turn in case.turns
    }) == 6
    assert not hasattr(manifest, "labels")
    rebuilt = build_dlg05_typed_preflight_manifest()
    assert rebuilt.stable_key() == manifest.stable_key()


def test_dlg05_preflight_query_provider_leaves_stance_to_g01_policy():
    """候选 Evidence 由 provider 给出，四态由真实 G-01 selector 推导。"""
    catalog = build_dlg05_typed_preflight_catalog()
    plans = build_dlg05_preflight_evidence_plans(catalog)
    assert all(plan.stable_key() == plan.stable_key() for plan in plans)
    route = minimal_instruction_identity((31005, 900, 1))
    reason = minimal_instruction_identity((31005, 900, 2))
    executor = ConversationHeldOutPreflightQuestionExecutor(
        route, reason, plans)
    fixture = _fixture()
    try:
        observed = []
        for case in catalog.cases:
            for turn in case.turns:
                query = QuestionQuery(
                    turn.request,
                    route,
                    (31005, 901, case.case_key.components[-1], turn.ordinal),
                )
                result = executor.execute(query)
                selection = fixture.runtime.selector.select(
                    result.planning_request())
                observed.append(selection.stance)
                assert all(candidate.evidence for candidate in result.candidates)
        assert observed == [
            fixture.content.answer, fixture.content.answer,
            fixture.content.clarify, fixture.content.clarify,
            fixture.content.unknown, fixture.content.unknown,
            fixture.content.conflict, fixture.content.conflict,
            fixture.content.unknown, fixture.content.unknown,
            fixture.content.answer, fixture.content.answer,
        ]
    finally:
        fixture.close()


def test_dlg05_axis_audit_rejects_manifest_only_language_axes():
    """synonym/order/omission 等不能仅凭 manifest key 进入 formal。"""
    catalog = build_dlg05_typed_preflight_catalog()
    compiler = build_dlg05_preflight_language_compiler(catalog)
    relation_compiler = build_dlg05_unseen_relation_compiler(catalog)
    source_compiler = build_dlg05_unseen_source_compiler(catalog)
    audit = audit_dlg05_preflight_axis_inputs(
        catalog,
        compiler=compiler,
        relation_compiler=relation_compiler,
        source_compiler=source_compiler,
    )
    by_axis = {item.axis: item for item in audit}
    assert by_axis[AXIS_PROPOSITION_REFERENCE].typed_input_bound == 1
    assert by_axis[AXIS_SCOPE_DRIFT].typed_input_bound == 1
    assert by_axis[AXIS_ROLLBACK].typed_input_bound == 1
    assert by_axis[AXIS_SYNONYM].typed_input_bound == 1
    assert by_axis[AXIS_SYNONYM].manifest_only == 0
    assert by_axis[AXIS_OMISSION].manifest_only == 0
    assert by_axis[AXIS_EVENT_REFERENCE].typed_input_bound == 1
    assert by_axis[AXIS_EVENT_REFERENCE].manifest_only == 0
    assert by_axis[AXIS_UNSEEN_SOURCE].typed_input_bound == 1
    assert by_axis[AXIS_UNSEEN_SOURCE].manifest_only == 0
    assert by_axis[AXIS_UNSEEN_RELATION].typed_input_bound == 1
    assert by_axis[AXIS_UNSEEN_RELATION].manifest_only == 0
    assert by_axis[AXIS_SYNONYM].semantic_runtime_bound == 1
    assert by_axis[AXIS_ORDER].typed_input_bound == 1
    assert by_axis[AXIS_ORDER].semantic_runtime_bound == 1
    assert by_axis[AXIS_ORDER].manifest_only == 0
    assert by_axis[AXIS_OMISSION].typed_input_bound == 1
    assert by_axis[AXIS_OMISSION].semantic_runtime_bound == 0
    assert by_axis[AXIS_OMISSION].manifest_only == 0
    assert by_axis[AXIS_EXPLICIT_REPEAT].typed_input_bound == 1
    assert by_axis[AXIS_EXPLICIT_REPEAT].manifest_only == 0
    assert all(
        item.semantic_runtime_bound == 0
        for axis, item in by_axis.items()
        if axis not in {AXIS_SYNONYM, AXIS_ORDER})


def test_dlg05_synonym_uses_distinct_typed_forms_and_same_semantic_frame():
    """case 1 的两个不同 Representation 都经 compiler 进入同一构式。"""
    catalog = build_dlg05_typed_preflight_catalog()
    compiler = build_dlg05_preflight_language_compiler(catalog)
    assert isinstance(compiler, ConversationQuestionInputCompiler)
    turns = catalog.cases[0].turns
    assert turns[0].language_input is not None
    assert turns[1].language_input is not None
    first = turns[0].language_input
    second = turns[1].language_input
    assert first.stable_key() != second.stable_key()
    assert first.utterance.visible_forms != second.utterance.visible_forms
    assert compiler.compile(first) == turns[0].request
    assert compiler.compile(second) == turns[1].request
    first_normalized = compiler.normalizer.normalize(first.utterance)
    second_normalized = compiler.normalizer.normalize(second.utterance)
    assert tuple(sorted(
        item.stable_key() for item in first_normalized.semantic_atoms)) == tuple(
            sorted(item.stable_key()
                   for item in second_normalized.semantic_atoms))
    assert first_normalized.semantic_atoms == tuple(
        reversed(second_normalized.semantic_atoms))
    assert turns[0].request.target != turns[1].request.target


def test_dlg05_omission_requires_context_target_anchor():
    """省略输入不能脱离 ConversationContextRead 独立编译。"""
    catalog = build_dlg05_typed_preflight_catalog()
    compiler = build_dlg05_preflight_language_compiler(catalog)
    omitted = catalog.cases[1].turns[1].language_input
    assert omitted is not None and omitted.provided_positions
    with pytest.raises(ConversationLanguageInputError, match="省略输入需要非空"):
        compiler.compile(omitted)
    stance = minimal_instruction_identity((31005, 399, 1))
    anchor_turn = ConversationTurnState(
        turn_ordinal=0,
        request_key=(1,),
        target_key=catalog.cases[1].turns[0].request.target.stable_key(),
        query_key=(2,),
        planning_key=(3,),
        response_stance=stance,
        selected_candidate_keys=((4,),),
        cited_sources=(SourceRef(
            31005, 398, 1, GLOBAL_OWNER_SCOPE, VersionBundle()),),
        discourse_sentence_keys=((5,),),
        parser_revision=(6,),
        readback_key=(7,),
    )
    read = ConversationContextRead(
        (31005, 398, 1),
        1,
        tuple(range(32)),
        (anchor_turn,),
    )
    restored = compiler.compile(omitted, context_read=read)
    assert restored == catalog.cases[1].turns[1].request


def test_dlg05_explicit_repeat_reuses_complete_typed_request():
    """explicit-repeat 不得用 trace 差异冒充第二种输入。"""
    catalog = build_dlg05_typed_preflight_catalog()
    case = catalog.cases[5]
    assert case.turns[1].context_mode == CONTEXT_EXPLICIT_REPEAT
    assert case.turns[1].request == case.turns[0].request
    assert case.turns[1].request.stable_key() == case.turns[0].request.stable_key()


def test_dlg05_event_reference_reuses_real_typed_event_filler():
    """REFERENCE_EVENT 必须指向命题中的一等 Event，不只是协议键。"""
    catalog = build_dlg05_typed_preflight_catalog()
    case = catalog.cases[2]
    event_keys = []
    for turn in case.turns:
        keys = tuple(
            binding.filler.stable_key()
            for binding in turn.request.target.bindings
            if isinstance(binding.filler, ObjectIdentity)
            and binding.filler.object_kind == OBJECT_EVENT
        )
        assert len(keys) == 1
        event_keys.append(keys)
    assert event_keys[1] == event_keys[0]
    assert case.turns[1].request.target != case.turns[0].request.target


def test_dlg05_unseen_relation_is_new_combination_of_seen_typed_primitives():
    """held-out relation 完整组合未见，但承重原语必须都在 TRAIN 分母。"""
    catalog = build_dlg05_typed_preflight_catalog()
    compiler = build_dlg05_unseen_relation_compiler(catalog)
    case = catalog.cases[3]
    structures = tuple(
        compiler.compile(turn.request) for turn in case.turns)
    assert structures[1] == structures[0]
    assert structures[0].stable_key() not in {
        item.stable_key() for item in compiler.inventory.structures}
    assert structures[0].predicate in {
        item.predicate for item in compiler.inventory.structures}
    assert structures[0].construction in {
        item.construction for item in compiler.inventory.structures}
    train_slots = {
        slot for item in compiler.inventory.structures for slot in item.slots}
    assert set(structures[0].slots).issubset(train_slots)


def test_dlg05_unseen_relation_rejects_train_replay_and_unseen_primitive():
    """已见完整组合与未见承重原语都不能冒充可组合关系泛化。"""
    catalog = build_dlg05_typed_preflight_catalog()
    request = catalog.cases[3].turns[0].request
    held_out = ConversationRelationStructure.from_target(request.target)
    replay = ConversationUnseenRelationCompiler(
        ConversationRelationTrainingInventory((held_out,)))
    with pytest.raises(
            ConversationRelationInputError,
            match="完整结构已出现在 TRAIN"):
        replay.compile(request)

    compiler = build_dlg05_unseen_relation_compiler(catalog)
    unknown_predicate = replace(
        request,
        target=replace(
            request.target,
            predicate=concept_identity((31005, 903, 1)),
        ),
    )
    with pytest.raises(
            ConversationRelationInputError,
            match="predicate 未在 TRAIN 出现"):
        compiler.compile(unknown_predicate)


def test_dlg05_unseen_source_requires_new_exact_ref_in_seen_source_domain():
    """来源必须精确未见，但 kind/owner/version 域不能凭空出现。"""
    catalog = build_dlg05_typed_preflight_catalog()
    request = catalog.cases[2].turns[0].request
    compiler = build_dlg05_unseen_source_compiler(catalog)
    assert compiler.compile(request) == request.source

    replay = ConversationUnseenSourceCompiler(
        ConversationSourceTrainingInventory((request.source,)))
    with pytest.raises(
            ConversationSourceInputError,
            match="SourceRef 已出现在 TRAIN"):
        replay.compile(request)

    foreign = SourceRef(
        request.source.source_kind + 1,
        request.source.source_id,
        request.source.document_id,
        request.source.owner,
        request.source.versions,
    )
    with pytest.raises(
            ConversationSourceInputError,
            match="来源域未在 TRAIN 出现"):
        compiler.compile_source(foreign)
