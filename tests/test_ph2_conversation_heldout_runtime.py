"""DLG-05 小规模真实 selection-first 两回合纵切。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import hashlib

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    SourceRef,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.question_answer import QuestionQuery
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.conversation_heldout_protocol import (
    AXIS_CONFLICT,
    AXIS_EXPLICIT_REPEAT,
    AXIS_MEMORY_CAUSAL,
    AXIS_MEMORY_MISS,
    AXIS_OMISSION,
    CONTEXT_CARRY,
    CONTEXT_FRESH,
    ConversationHeldOutCase,
    ConversationHeldOutLabel,
    ConversationHeldOutLabelSet,
    ConversationHeldOutManifest,
    ConversationHeldOutTurn,
    MEMORY_OFF,
    MEMORY_ON,
    RESPONSE_UNKNOWN,
    RESPONSE_CLARIFY,
    RESPONSE_CONFLICT,
    RESPONSE_ANSWER,
    evaluate_label_late,
)
from pure_integer_ai.experiments.conversation_context_runtime import (
    start_conversation_context,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    ConversationHeldOutMemoryPlan,
    ConversationHeldOutRuntimeError,
    ConversationHeldOutTurnPlan,
    conversation_turn_content_identity,
    conversation_turn_scope_key,
    conversation_turn_source_key,
    run_real_selection_first,
    run_real_selection_first_receipt,
)
from pure_integer_ai.experiments.conversation_heldout_runtime import (
    run_real_selection_first_case,
)
from pure_integer_ai.experiments.conversation_heldout_preflight_runtime import (
    ConversationHeldOutCatalogTurnFactory,
    MappedConversationHeldOutResponseActResolver,
)
from pure_integer_ai.experiments.conversation_memory_demand_runtime import (
    ConversationMemoryDemandConsumer,
    ConversationMemoryQuestionExecutor,
)
from pure_integer_ai.experiments.conversation_heldout_answer_runtime import (
    ConversationHeldOutAnswerInputError,
    claim_input_from_candidate,
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
from pure_integer_ai.experiments.conversation_heldout_qualification import (
    conversation_heldout_rollback_fault_key,
    ConversationHeldOutQualificationReceipt,
    ConversationHeldOutRollbackRecoveryReceipt,
    qualify_dlg05_preflight,
)
from pure_integer_ai.experiments.conversation_heldout_freeze import (
    write_dlg05_public_freeze_document,
)
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    learn_grounded_answer_surface_model,
    surface_pattern_structure_id,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorTarget,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalBuild,
    GroundedAnswerRunLocalComponents,
    GroundedAnswerRunLocalFactory,
)
from pure_integer_ai.experiments.ph2_grounded_answer_verification import (
    GroundedAnswerEvidenceSourceVerifier,
    GroundedAnswerStructureVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
    GroundedResponseActQuestionInput,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    compile_grounded_response_act_planning,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.verification_orchestration import VERDICT_SUPPORT
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    ProtocolKey,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)

from test_ph2_conversation_memory_demand import (
    _close_query,
    _fixture as _memory_fixture,
    _current as _memory_current,
    _profile_for_understanding,
)
from test_a10_attractor_state import _instruction, _setup
from test_d02_md03_directional_center_adapter import _adapter
from test_g05_memory_generation_evidence import (
    _EmptyQuestionExecutor,
    _complete_source,
)
from test_f00_generation_postcheck import _postcheck_owners
from test_f00_question_answer_runtime import _fixture as _question_fixture
from pure_integer_ai.cognition.shared.memory_resolver import (
    MemoryAggregateFilter,
    RESOLUTION_ORIGIN_MEMORY,
)
from test_ph2_grounded_answer_response_act_runtime import (
    _AliasFactory,
    _BASE,
)
from test_ph2_grounded_answer_connector_runtime import (
    _AliasFactory as _AnswerAliasFactory,
)
from test_ph2_grounded_answer_course import _connector_question_and_candidate
from test_g02_generation_structure_plan import _plan_protocol, _selection
from test_g03_generation_surface import _surface_protocol
from test_g04_generation_postcheck import (
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from test_s07_structure_order import _graphs


_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


def _snapshot_key(snapshot) -> tuple[int, ...]:
    """把 canonical backend snapshot 压成固定纯整数资格键。"""
    return tuple(hashlib.sha256(canonical_json_bytes(snapshot)).digest())


def _manifest(
        requests: tuple[QuestionRequest, QuestionRequest],
        ) -> tuple[ConversationHeldOutManifest, ConversationHeldOutCase]:
    """建立仅覆盖 Memory miss 的独立两回合 family。"""
    family = ProtocolKey((20260819, 5, 99))

    def identity(kind: str, value: int) -> CanonicalIdentity:
        return CanonicalIdentity.from_value(("dlg05-runtime", kind, value))

    case_key = ProtocolKey((20260819, 599, 1))
    turns = (
        ConversationHeldOutTurn(
            ProtocolKey((20260819, 599, 1, 1)), 1,
            conversation_turn_content_identity(requests[0]),
            conversation_turn_source_key(requests[0]),
            conversation_turn_scope_key(requests[0]),
            CONTEXT_FRESH, MEMORY_OFF,
            ProtocolKey((6, 0)), ProtocolKey((5, 0))),
        ConversationHeldOutTurn(
            ProtocolKey((20260819, 599, 1, 2)), 2,
            conversation_turn_content_identity(requests[1]),
            conversation_turn_source_key(requests[1]),
            conversation_turn_scope_key(requests[1]),
            CONTEXT_CARRY, MEMORY_ON,
            ProtocolKey((6, 0)), ProtocolKey((5, 0))),
    )
    case = ConversationHeldOutCase(
        case_key, family, (AXIS_MEMORY_MISS,), identity("dedup", 1),
        identity("provenance", 1), turns)
    manifest = ConversationHeldOutManifest(
        1, family,
        (identity("train-content", 1),),
        (identity("train-dedup", 1),),
        (identity("train-provenance", 1),),
        (case,),
        (AXIS_MEMORY_MISS,),
        (MEMORY_OFF, MEMORY_ON),
    )
    return manifest, case


def _unknown_runtime():
    """装配生产 response-act factory 的实际 UNKNOWN runtime。"""
    episode = next(
        item for item in read_grounded_answer_episodes(_SAMPLE)
        if item.episode_id == "train-grounded-unknown-budget-v1")
    model, _ = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    branch = language_branch_identity((_BASE, 901))
    planning_build = compile_grounded_response_act_planning(episode, branch)
    content = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 902, index))
        for index in range(1, 6)))
    selector = AnswerContentSelector(
        content,
        EvidenceAnswerPolicy(
            content,
            EvidenceAnswerPolicyProtocol(*tuple(
                minimal_instruction_identity((_BASE, 903, index))
                for index in range(1, 5))),
        ),
    )
    target = GroundedResponseActCompileTarget(
        "UNKNOWN", content.unknown, branch, (_BASE, 904))
    selected = compile_grounded_response_act_patterns(
        model, target).variants[0]
    backend = DictBackend()
    aliases = _AliasFactory(branch)
    graphs = _graphs(backend)
    components = GroundedResponseActRunLocalComponents(
        selector,
        _plan_protocol(_BASE + 910),
        GenerationStructureLayerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 911, index))
            for index in range(1, 4))),
        _surface_protocol(_BASE + 920),
        aliases,
        UnicodeRepresentationRenderer(
            target.representation_family,
            minimal_instruction_identity((_BASE, 921))),
        minimal_instruction_identity((_BASE, 921)),
        _postcheck_protocol(),
        GroundedResponseActStructureVerifier(
            minimal_instruction_identity((_BASE, 922, 1)),
            minimal_instruction_identity((_BASE, 922, 2))),
        _StaticVerifier(VERDICT_SUPPORT, 923),
        GroundedResponseActTaskVerifier(
            minimal_instruction_identity((_BASE, 924, 1)),
            minimal_instruction_identity((_BASE, 924, 2))),
        QuestionAnswerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 925, index))
            for index in range(1, 4))),
    )
    query_kind = minimal_instruction_identity((_BASE, 926, 1))
    installation = GroundedResponseActRunLocalFactory(
        graphs.lifecycle, components).build(
            GroundedResponseActRunLocalBuild(
                model,
                episode.question,
                target,
                planning_build.planning,
                selected.pattern_id,
                GroundedResponseActParserProtocol(*tuple(
                    minimal_instruction_identity((_BASE, 927, index))
                    for index in range(1, 4))),
                query_kind,
                minimal_instruction_identity((_BASE, 926, 2)),
                minimal_instruction_identity((_BASE, 928, 1)),
                (_BASE, 928, 2),
            ))
    request = QuestionRequest(
        query_kind,
        minimal_instruction_identity((_BASE, 929, 1)),
        planning_build.planning.goal.goal_kind,
        planning_build.planning.goal.proposition,
        planning_build.planning.goal.required,
        planning_build.planning.goal.scope,
        planning_build.planning.goal.scope,
        (_BASE, 929, 2),
        branch,
        (),
    )
    return backend, aliases, installation.runtime, request, content.unknown


def _response_act_runtime(
        response_act: str,
        episode_id: str,
        offset: int,
        *,
        planning_override=None,
        question_input=None,
        branch_override=None,
        query_kind_override=None,
        selector_override=None,
        content_override=None,
        ):
    """装配一个真实 learned CLARIFY/CONFLICT runtime，保留 owner 供两回合复用。"""
    episodes = read_grounded_answer_episodes(_SAMPLE)
    episode = next(item for item in episodes if item.episode_id == episode_id)
    model, _ = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    branch = (
        language_branch_identity((_BASE, 700, offset))
        if branch_override is None else branch_override)
    planning_build = compile_grounded_response_act_planning(episode, branch)
    planning = (
        planning_build.planning
        if planning_override is None else planning_override)
    content = (
        AnswerContentProtocol(*tuple(
            minimal_instruction_identity((_BASE, 701 + offset, index))
            for index in range(1, 6)))
        if content_override is None else content_override)
    selector = (
        AnswerContentSelector(
            content,
            EvidenceAnswerPolicy(
                content,
                EvidenceAnswerPolicyProtocol(*tuple(
                    minimal_instruction_identity((_BASE, 711 + offset, index))
                    for index in range(1, 5))),
            ),
        )
        if selector_override is None else selector_override)
    stance = getattr(content, response_act.lower())
    target = GroundedResponseActCompileTarget(
        response_act, stance, branch, (_BASE, 720, offset))
    selected = compile_grounded_response_act_patterns(
        model, target).variants[0]
    backend = DictBackend()
    aliases = _AliasFactory(branch)
    graphs = _graphs(backend)
    renderer_identity = minimal_instruction_identity((_BASE, 721, offset))
    components = GroundedResponseActRunLocalComponents(
        selector,
        _plan_protocol(_BASE + 730 + offset),
        GenerationStructureLayerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 740 + offset, index))
            for index in range(1, 4))),
        _surface_protocol(_BASE + 750 + offset),
        aliases,
        UnicodeRepresentationRenderer(
            target.representation_family, renderer_identity),
        renderer_identity,
        _postcheck_protocol(),
        GroundedResponseActStructureVerifier(
            minimal_instruction_identity((_BASE, 760 + offset, 1)),
            minimal_instruction_identity((_BASE, 760 + offset, 2))),
        _StaticVerifier(VERDICT_SUPPORT, 770 + offset),
        GroundedResponseActTaskVerifier(
            minimal_instruction_identity((_BASE, 780 + offset, 1)),
            minimal_instruction_identity((_BASE, 780 + offset, 2))),
        QuestionAnswerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 790 + offset, index))
            for index in range(1, 4))),
    )
    query_kind = (
        minimal_instruction_identity((_BASE, 800 + offset, 1))
        if query_kind_override is None else query_kind_override)
    installation = GroundedResponseActRunLocalFactory(
        graphs.lifecycle, components).build(
            GroundedResponseActRunLocalBuild(
                model,
                (episode.question if question_input is None else question_input),
                target,
                planning,
                selected.pattern_id,
                GroundedResponseActParserProtocol(*tuple(
                    minimal_instruction_identity((_BASE, 810 + offset, index))
                    for index in range(1, 4))),
                query_kind,
                minimal_instruction_identity((_BASE, 800 + offset, 2)),
                minimal_instruction_identity((_BASE, 820 + offset, 1)),
                (_BASE, 820 + offset, 2),
            ))
    request = QuestionRequest(
        query_kind,
        minimal_instruction_identity((_BASE, 830 + offset, 1)),
        planning.goal.goal_kind,
        planning.goal.proposition,
        planning.goal.required,
        planning.goal.scope,
        planning.goal.scope,
        (_BASE, 830 + offset, 2),
        branch,
        tuple(item.proposition for item in planning.candidates),
    )
    return (
        backend,
        aliases,
        installation.runtime,
        request,
        stance,
        planning,
        selector,
        content,
    )


def _answer_runtime(
        offset: int,
        *,
        detached_claim: bool = False,
        planning_override=None,
        candidate_override=None,
        branch_override=None,
        question_input_override=None,
        request_override=None,
        selector_override=None,
        content_override=None,
        ):
    """装配真实单命题 ANSWER connector、parser、citation 与 G-04 runtime。"""
    model, question, base_planning, base_candidate, base_branch = (
        _connector_question_and_candidate())
    planning = base_planning if planning_override is None else planning_override
    candidate = base_candidate if candidate_override is None else candidate_override
    branch = base_branch if branch_override is None else branch_override
    selected_pattern = next(
        pattern for pattern in model.patterns
        if any(part.literal == "档案显示，" for part in pattern.parts))
    _selected, default_selector, default_content = _selection(planning)
    selector = (
        default_selector if selector_override is None else selector_override)
    content = default_content if content_override is None else content_override
    surface_protocol = _surface_protocol(_BASE + 900 + offset)
    family = (_BASE, 910, offset)
    target = GroundedAnswerConnectorTarget(
        candidate.proposition, branch, family)
    backend = DictBackend()
    aliases = _AnswerAliasFactory(branch)
    graphs = _graphs(backend)
    renderer_identity = minimal_instruction_identity((_BASE, 911, offset))
    query_kind = (
        request_override.query_kind
        if request_override is not None
        else minimal_instruction_identity((_BASE, 912 + offset, 1))
    )
    components = GroundedAnswerRunLocalComponents(
        selector,
        _plan_protocol(_BASE + 920 + offset),
        GenerationStructureLayerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 930 + offset, index))
            for index in range(1, 4))),
        aliases,
        UnicodeRepresentationRenderer(family, renderer_identity),
        renderer_identity,
        _postcheck_protocol(),
        GroundedAnswerStructureVerifier(
            minimal_instruction_identity((_BASE, 940 + offset, 1)),
            minimal_instruction_identity((_BASE, 940 + offset, 2))),
        GroundedAnswerEvidenceSourceVerifier(
            minimal_instruction_identity((_BASE, 950 + offset, 1)),
            minimal_instruction_identity((_BASE, 950 + offset, 2))),
        QuestionAnswerProtocol(*tuple(
            minimal_instruction_identity((_BASE, 960 + offset, index))
            for index in range(1, 4))),
        EvidenceQuestionPostcheckMapper(
            (_BASE, 970 + offset, 1),
            citation_required=True,
            trust_required=True,
        ),
    )
    question_input = (
        question
        if question_input_override is None
        else question_input_override
    )
    if detached_claim and question_input_override is None:
        question_input = GroundedAnswerClaimInput(
            next(
                item.claim_text
                for item in question.evidence
                if item.proposition_id
                == question.answer_plan.ordered_claim_ids[0]
            ),
        )
    installation = GroundedAnswerRunLocalFactory(
        surface_protocol, graphs.lifecycle, components).build(
            GroundedAnswerRunLocalBuild(
                model,
                question_input,
                target,
                planning,
                candidate,
                surface_pattern_structure_id(selected_pattern),
                selected_pattern.pattern_id,
                GroundedAnswerParserProtocol(
                    *tuple(minimal_instruction_identity(
                        (_BASE, 980 + offset, index))
                           for index in range(1, 6)),
                    content.answer,
                ),
                query_kind,
                minimal_instruction_identity((_BASE, 912 + offset, 2)),
                minimal_instruction_identity((_BASE, 990 + offset, 1)),
                (_BASE, 990 + offset, 2),
            ))
    request = (
        QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 995 + offset, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            candidate.scope,
            candidate.scope,
            (_BASE, 995 + offset, 2),
            branch,
        )
        if request_override is None else request_override
    )
    return (
        backend,
        aliases,
        installation.runtime,
        request,
        content.answer,
        planning,
    )


def test_dlg05_real_runner_executes_context_memory_and_g04_before_labels():
    """两回合真实 runtime 只产 observation，最后才允许 label-late。"""
    qa_backend, aliases, runtime, request, unknown = _unknown_runtime()
    requests = tuple(
        replace(request, trace=(*request.trace, 500 + ordinal))
        for ordinal in (1, 2)
    )
    manifest, case = _manifest(requests)
    memory_backend, memory_ctx, current, center, consumer = _memory_fixture()
    profile = _profile_for_understanding()

    def prepare(_case, turn, _context_read):
        memory = None
        if turn.memory_mode == MEMORY_ON:
            memory = ConversationHeldOutMemoryPlan(
                consumer, current, center, profile,
                MemoryAccessContext(1, 2, 3))
        return ConversationHeldOutTurnPlan(
            runtime,
            requests[turn.ordinal - 1],
            unknown,
            RESPONSE_UNKNOWN,
            memory,
        )

    try:
        observations = run_real_selection_first(manifest, prepare)
        assert len(observations) == 1
        observation = observations[0]
        assert observation.response_act == RESPONSE_UNKNOWN
        assert observation.turn_response_acts == (RESPONSE_UNKNOWN,) * 2
        assert observation.selected_candidate_keys == ()
        assert observation.cited_source_keys == ()
        assert observation.context_revision == 2
        assert len(observation.memory_receipt_keys) == 1
        # 当前 fixture 的 Memory read 命中 Core 候选，而非生成候选；runner
        # 必须诚实标为“已读取、未形成因果回答”而不是伪造 Memory 学习。
        assert observation.memory_causal_proven == 0

        # label owner 在真实 selection-first 完成后才建立，不能进入 runtime。
        from pure_integer_ai.experiments.conversation_heldout_protocol import (
            ConversationHeldOutLabel,
            ConversationHeldOutLabelSet,
        )
        labels = ConversationHeldOutLabelSet(
            manifest.stable_key(),
            (ConversationHeldOutLabel(
                case.case_key,
                RESPONSE_UNKNOWN,
                (RESPONSE_UNKNOWN, RESPONSE_UNKNOWN),
                (),
                (),
            ),),
        )
        result = evaluate_label_late(manifest, labels, observations)
        assert result.complete
        assert result.passed == 1

        wrong_turn = replace(
            case.turns[0],
            content=CanonicalIdentity.from_value(("wrong-request", 1)),
        )
        wrong_case = replace(
            case, turns=(wrong_turn, case.turns[1]))
        wrong_manifest = replace(manifest, cases=(wrong_case,))
        with pytest.raises(
                ConversationHeldOutRuntimeError,
                match="content 与实际 QuestionRequest"):
            run_real_selection_first(wrong_manifest, prepare)
    finally:
        _close_query(memory_ctx)
        memory_backend.close()
        if aliases.fixture is not None:
            aliases.fixture.close()
        qa_backend.close()


def test_dlg04_read_can_be_projected_into_generation_memory_evidence():
    """同次 demand read 可无损形成带来源 Memory generation candidate。"""
    backend, ctx, source, _attractor, _strategy, compilation, goals = _setup()
    try:
        current = compilation.current
        context_read = start_conversation_context((599, 2)).read(0)
        center = _adapter().from_understanding(
            current, current.occurrences[0], strength="CONDITIONAL")
        read = ConversationMemoryDemandConsumer(
            ctx,
            ctx.memory_query_runtime,
            ctx.memory_resolver_runtime,
        ).read(
            current,
            center,
            _profile_for_understanding(),
            access=MemoryAccessContext(1, 2, 3),
            context_read=context_read,
        )
        repository = SourceRecordRepository(backend)
        memory_keys = tuple(
            candidate.stable_key()
            for candidate_set in read.resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
        )
        traces = {
            trace.source.stable_key(): trace
            for candidate_set in read.resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
            for trace in candidate.memory_source_traces
        }
        for ordinal, trace in enumerate(traces.values(), start=1):
            _complete_source(repository, trace, ordinal)
        request = QuestionRequest(
            _instruction(source, 59901),
            _instruction(source, 59902),
            _instruction(source, 59903),
            goals[0].proposition,
            goals[0].required,
            goals[0].scope,
            goals[0].scope,
            (59904, 1),
        )
        executor = ConversationMemoryQuestionExecutor(
            read,
            request.target,
            authorized_candidate_keys=memory_keys,
            executed_reason=_instruction(source, 59905),
            binding_reason=_instruction(source, 59906),
            trace_prefix=(59907, 1),
            source_records=repository,
        )
        result = executor.execute(QuestionQuery(
            request,
            _instruction(source, 59908),
            (59909, 1),
        ))
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.memory_evidence
        assert {
            item.candidate.stable_key()
            for item in candidate.memory_evidence
        } == set(memory_keys)
        assert set(candidate.citation_sources)
        assert read.receipt.status == "HIT"
        with pytest.raises(
                ConversationHeldOutAnswerInputError,
                match="多来源原文不一致"):
            claim_input_from_candidate(candidate, repository)
        single_key = next(
            candidate.stable_key()
            for candidate_set in read.resolution.sets
            for candidate in candidate_set.candidates
            if candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
            and len(candidate.memory_source_traces) == 1
        )
        single_executor = ConversationMemoryQuestionExecutor(
            read,
            request.target,
            authorized_candidate_keys=(single_key,),
            executed_reason=_instruction(source, 59910),
            binding_reason=_instruction(source, 59911),
            trace_prefix=(59912, 1),
            source_records=repository,
        )
        single = single_executor.execute(QuestionQuery(
            request,
            _instruction(source, 59913),
            (59914, 1),
        )).candidates[0]
        claim_input = claim_input_from_candidate(single, repository)
        assert claim_input.claim_text.startswith("来源")
    finally:
        _close_query(ctx)
        backend.close()


def _response_case(
        family: ProtocolKey,
        case_number: int,
        requests: tuple[QuestionRequest, QuestionRequest],
        axis: ProtocolKey,
        modes: tuple[ProtocolKey, ProtocolKey],
        ) -> ConversationHeldOutCase:
    """把两个真实基础 request 冻结为一个无标签 response-act case。"""
    identity = lambda kind: CanonicalIdentity.from_value(
        ("dlg05-response", kind, case_number))
    turns = tuple(
        ConversationHeldOutTurn(
            ProtocolKey((20260819, 700 + case_number, ordinal)),
            ordinal,
            conversation_turn_content_identity(request),
            conversation_turn_source_key(request),
            conversation_turn_scope_key(request),
            CONTEXT_FRESH if ordinal == 1 else CONTEXT_CARRY,
            modes[ordinal - 1],
            ProtocolKey((6, 0)),
            ProtocolKey((5, 0)),
        )
        for ordinal, request in enumerate(requests, start=1)
    )
    return ConversationHeldOutCase(
        ProtocolKey((20260819, 799, case_number)),
        family,
        (axis,),
        identity("dedup"),
        identity("provenance"),
        turns,
    )


def test_dlg05_real_response_act_cases_cover_clarify_and_conflict():
    """两个 learned non-answer case 通过同一 runner 完成两回合 readback。"""
    clarify_owner = _response_act_runtime(
        "CLARIFY", "train-grounded-clarify-site-v1", 51)
    conflict_owner = _response_act_runtime(
        "CONFLICT", "train-grounded-conflict-date-v1", 52)
    memory_backend, memory_ctx, current, center, consumer = _memory_fixture()
    profile = _profile_for_understanding()
    family = ProtocolKey((20260819, 5, 700))
    owners = (clarify_owner, conflict_owner)
    base_requests = []
    for owner in owners:
        base_requests.append(tuple(
            replace(owner[3], trace=(*owner[3].trace, 900 + ordinal))
            for ordinal in (1, 2)
        ))
    cases = (
        _response_case(
            family, 1, base_requests[0], AXIS_OMISSION,
            (MEMORY_OFF, MEMORY_OFF)),
        _response_case(
            family, 2, base_requests[1], AXIS_CONFLICT,
            (MEMORY_ON, MEMORY_OFF)),
    )
    manifest = ConversationHeldOutManifest(
        1,
        family,
        (),
        (),
        (),
        cases,
        (AXIS_OMISSION, AXIS_CONFLICT),
        (MEMORY_OFF, MEMORY_ON),
    )
    expected_candidate_keys = tuple(
        owner[2].selector.select(owner[5]).selected_candidate_keys
        for owner in owners
    )

    def prepare(case, turn, _context_read):
        index = 0 if case.case_key.components[-1] == 1 else 1
        owner = owners[index]
        memory = None
        if turn.memory_mode == MEMORY_ON:
            memory = ConversationHeldOutMemoryPlan(
                consumer,
                current,
                center,
                profile,
                MemoryAccessContext(1, 2, 3),
            )
        return ConversationHeldOutTurnPlan(
            owner[2],
            base_requests[index][turn.ordinal - 1],
            owner[4],
            RESPONSE_CLARIFY if index == 0 else RESPONSE_CONFLICT,
            memory,
        )

    labels = ConversationHeldOutLabelSet(
        manifest.stable_key(),
        tuple(
            ConversationHeldOutLabel(
                case.case_key,
                RESPONSE_CLARIFY if index == 0 else RESPONSE_CONFLICT,
                ((RESPONSE_CLARIFY,) if index == 0
                 else (RESPONSE_CONFLICT,)) * 2,
                expected_candidate_keys[index],
                (),
            )
            for index, case in enumerate(cases)
        ),
    )
    try:
        observations = run_real_selection_first(manifest, prepare)
        assert tuple(item.response_act for item in observations) == (
            RESPONSE_CLARIFY, RESPONSE_CONFLICT)
        assert all(item.turn_response_acts == (item.response_act,) * 2
                   for item in observations)
        assert tuple(item.selected_candidate_keys for item in observations) == (
            expected_candidate_keys)
        assert all(expected_candidate_keys)
        assert all(item.cited_source_keys == () for item in observations)
        assert observations[0].memory_receipt_keys == ()
        assert len(observations[1].memory_receipt_keys) == 1
        assert all(item.memory_causal_proven == 0 for item in observations)
        result = evaluate_label_late(manifest, labels, observations)
        assert result.complete and result.passed == 2
    finally:
        _close_query(memory_ctx)
        memory_backend.close()
        for backend, aliases, *_ in owners:
            if aliases.fixture is not None:
                aliases.fixture.close()
            backend.close()


def test_dlg05_preflight_g01_stance_installs_label_free_response_act_g04():
    """新 typed preflight request 用实际 G-01 stance 装配 learned response-act/G-04。"""
    catalog = build_dlg05_typed_preflight_catalog()
    plans = build_dlg05_preflight_evidence_plans(catalog)
    route = minimal_instruction_identity((31005, 990, 1))
    reason = minimal_instruction_identity((31005, 990, 2))
    query_executor = ConversationHeldOutPreflightQuestionExecutor(
        route, reason, plans)
    base = _question_fixture()
    try:
        cases = (
            (1, "CLARIFY"),
            (2, "UNKNOWN"),
            (3, "CONFLICT"),
        )
        # ANSWER 仍由 grounded connector 负责；本 factory 只接受 non-answer。
        for case_number, response_act in cases:
            turn = catalog.cases[case_number].turns[0]
            result = query_executor.execute(QuestionQuery(
                turn.request,
                route,
                (31005, 991, case_number + 1, 1),
            ))
            selection = base.runtime.selector.select(result.planning_request())
            stance = getattr(base.content, response_act.lower())
            assert selection.stance == stance
            owner = _response_act_runtime(
                response_act,
                (
                    "train-grounded-clarify-site-v1"
                    if response_act == "CLARIFY"
                    else (
                        "train-grounded-conflict-date-v1"
                        if response_act == "CONFLICT"
                        else "train-grounded-unknown-budget-v1"
                    )
                ),
                87 + case_number,
                planning_override=result.planning_request(),
                question_input=GroundedResponseActQuestionInput(response_act),
                branch_override=turn.request.target_branch,
                query_kind_override=turn.request.query_kind,
                selector_override=base.runtime.selector,
                content_override=base.content,
            )
            try:
                run = owner[2].run(turn.request)
                assert run.complete
                assert run.status == stance
                assert run.postcheck is not None
                assert run.postcheck.parsed.observation is not None
            finally:
                if owner[1].fixture is not None:
                    owner[1].fixture.close()
                owner[0].close()
    finally:
        base.close()


def test_dlg05_catalog_turn_factory_runs_two_fresh_typed_turns_selection_first():
    """catalog factory 将新 case 的两回合真实接入 context runner。"""
    catalog = build_dlg05_typed_preflight_catalog()
    plans = build_dlg05_preflight_evidence_plans(catalog)
    route = minimal_instruction_identity((31005, 992, 1))
    reason = minimal_instruction_identity((31005, 992, 2))
    executor = ConversationHeldOutPreflightQuestionExecutor(
        route, reason, plans)
    base = _question_fixture()
    owners = []

    class _Builder:
        """把实际 query result/selection 接到 learned runtime factory。"""

        def build(self, request, context_read, memory_read):
            """从当前 request 形成 planning，再由 G-01 选择 response-act。"""
            del context_read
            assert memory_read is None
            query = QuestionQuery(request, route, (31005, 993, request.trace[-2]))
            result = executor.execute(query)
            planning = result.planning_request()
            selection = base.runtime.selector.select(planning)
            stance_map = {
                base.content.unknown: (
                    "UNKNOWN", "train-grounded-unknown-budget-v1"),
                base.content.clarify: (
                    "CLARIFY", "train-grounded-clarify-site-v1"),
                base.content.conflict: (
                    "CONFLICT", "train-grounded-conflict-date-v1"),
            }
            response_act, episode_id = stance_map[selection.stance]
            owner = _response_act_runtime(
                response_act,
                episode_id,
                100 + request.trace[-2],
                planning_override=planning,
                question_input=GroundedResponseActQuestionInput(response_act),
                branch_override=request.target_branch,
                query_kind_override=request.query_kind,
                selector_override=base.runtime.selector,
                content_override=base.content,
            )
            owners.append(owner)
            return owner[2]

    resolver = MappedConversationHeldOutResponseActResolver((
        (base.content.unknown, RESPONSE_UNKNOWN),
        (base.content.clarify, RESPONSE_CLARIFY),
        (base.content.conflict, RESPONSE_CONFLICT),
    ))
    factory = ConversationHeldOutCatalogTurnFactory(
        catalog,
        _Builder(),
        response_act_resolver=resolver,
        question_input_compiler=build_dlg05_preflight_language_compiler(catalog),
    )
    # Case 2 is the independent two-candidate CLARIFY family and uses Memory OFF.
    case = catalog.cases[1]
    manifest = build_dlg05_typed_preflight_manifest()
    manifest_case = next(item for item in manifest.cases
                         if item.case_key == case.case_key)
    try:
        # 这里直接调用 runner 的单 case 核心，避免为窄 factory 测试伪造
        # Memory ON 评测分母；正式 family 仍必须经完整 manifest runner。
        observation = run_real_selection_first_case(
            manifest_case, factory, visible_limit=8)
        assert observation.response_act == RESPONSE_CLARIFY
        assert observation.turn_response_acts == (RESPONSE_CLARIFY,) * 2
        assert observation.context_revision == 2
        assert observation.memory_receipt_keys == ()
        assert all(owner[2] is not None for owner in owners)
        assert manifest_case.case_key == observation.case_key
    finally:
        base.close()
        for owner in owners:
            if owner[1].fixture is not None:
                owner[1].fixture.close()
            owner[0].close()


def test_dlg05_catalog_turn_factory_memory_on_requires_real_demand_plan():
    """Memory ON 没有真实 demand plan 时必须在 factory 边界 fail closed。"""
    catalog = build_dlg05_typed_preflight_catalog()
    manifest = build_dlg05_typed_preflight_manifest()
    case = catalog.cases[0]
    manifest_case = next(item for item in manifest.cases
                         if item.case_key == case.case_key)
    turn = manifest_case.turns[1]
    resolver = MappedConversationHeldOutResponseActResolver((
        (minimal_instruction_identity((31005, 994, 1)), RESPONSE_UNKNOWN),
    ))

    class _NeverBuilder:
        """Memory plan 缺失时不得被调用。"""

        def build(self, request, context_read, memory_read):
            del request, context_read, memory_read
            raise AssertionError("Memory ON 缺 plan 不得构造 runtime")

    factory = ConversationHeldOutCatalogTurnFactory(
        catalog,
        _NeverBuilder(),
        response_act_resolver=resolver,
    )
    with pytest.raises(
            ConversationHeldOutRuntimeError,
            match="缺少 typed memory plan"):
        factory(
            manifest_case,
            turn,
            start_conversation_context((31005, 995)).read(0),
        )


def test_dlg05_unseen_relation_turn_requires_train_structure_compiler():
    """case 4 不能绕过 TRAIN 结构分母直接进入问答 runtime。"""
    catalog = build_dlg05_typed_preflight_catalog()
    manifest = build_dlg05_typed_preflight_manifest()
    case = catalog.cases[3]
    manifest_case = next(item for item in manifest.cases
                         if item.case_key == case.case_key)
    resolver = MappedConversationHeldOutResponseActResolver((
        (minimal_instruction_identity((31005, 996, 1)), RESPONSE_UNKNOWN),
    ))

    class _NeverBuilder:
        """关系结构未鉴权时不得进入 runtime builder。"""

        def build(self, request, context_read, memory_read):
            del request, context_read, memory_read
            raise AssertionError("unseen relation 缺 compiler 不得构造 runtime")

    factory = ConversationHeldOutCatalogTurnFactory(
        catalog,
        _NeverBuilder(),
        memory_plan_factory=lambda *_args: None,
        response_act_resolver=resolver,
    )
    with pytest.raises(
            ConversationHeldOutRuntimeError,
            match="缺少 TRAIN structure compiler"):
        factory(
            manifest_case,
            manifest_case.turns[0],
            start_conversation_context((31005, 997)).read(0),
        )


def test_dlg05_unseen_source_turn_requires_train_source_compiler():
    """case 3 不能把 case 间不同 SourceRef 当作未见来源证明。"""
    catalog = build_dlg05_typed_preflight_catalog()
    manifest = build_dlg05_typed_preflight_manifest()
    case = catalog.cases[2]
    manifest_case = next(item for item in manifest.cases
                         if item.case_key == case.case_key)
    resolver = MappedConversationHeldOutResponseActResolver((
        (minimal_instruction_identity((31005, 998, 1)), RESPONSE_UNKNOWN),
    ))

    class _NeverBuilder:
        """来源分母未鉴权时不得进入 runtime builder。"""

        def build(self, request, context_read, memory_read):
            del request, context_read, memory_read
            raise AssertionError("unseen source 缺 compiler 不得构造 runtime")

    factory = ConversationHeldOutCatalogTurnFactory(
        catalog,
        _NeverBuilder(),
        response_act_resolver=resolver,
    )
    with pytest.raises(
            ConversationHeldOutRuntimeError,
            match="缺少 TRAIN source compiler"):
        factory(
            manifest_case,
            manifest_case.turns[0],
            start_conversation_context((31005, 999)).read(0),
        )


def test_dlg05_preflight_memory_reads_bind_each_new_source_scope_and_release():
    """六 case 的每个 Memory ON turn 都走真实 DLG-04 source/scope read。"""
    catalog = build_dlg05_typed_preflight_catalog()
    backend, ctx, _old_current, _old_center, consumer = _memory_fixture()
    profile = _profile_for_understanding()
    try:
        # _memory_fixture 为旧窄切片预开了一个 query；新 family 必须逐来源
        # 重新安装 M-06/M-07，并在每个 turn 结束时关闭完整生命周期。
        _close_query(ctx)
        seen = 0
        for case in catalog.cases:
            for turn in case.turns:
                if turn.memory_mode != MEMORY_ON:
                    continue
                request = turn.request
                document = document_scope(request.source)
                episode = request.response_scope.parent
                assert episode is not None
                assert episode.parent == document
                session = session_scope(
                    1,
                    owner=request.source.owner,
                    versions=request.source.versions,
                    source=request.source,
                )
                ctx.work_memory.begin_session(session)
                ctx.work_memory.begin_document(document)
                ctx.work_memory.begin_episode(episode)
                ctx.work_memory.begin_query(request.response_scope)
                current = _memory_current(
                    ctx, request.source, request.response_scope)
                center = _adapter().from_understanding(
                    current,
                    current.occurrences[0],
                    strength="CONDITIONAL",
                )
                released = []
                plan = ConversationHeldOutMemoryPlan(
                    consumer,
                    current,
                    center,
                    profile,
                    MemoryAccessContext(1, 2, 3),
                    release=lambda: released.append(1) or _close_query(ctx),
                )
                read = plan.consumer.read(
                    plan.current,
                    plan.center,
                    plan.profile,
                    access=plan.access,
                    context_read=start_conversation_context(
                        (31005, 1100, case.case_key.components[-1], turn.ordinal),
                    ).read(0),
                )
                assert read.receipt.source == request.source
                assert read.receipt.scope == request.response_scope
                assert read.receipt.status in {"HIT", "UNKNOWN"}
                plan.release_resources()
                assert released == [1]
                assert ctx.work_memory.active_query_scope is None
                seen += 1
        assert seen == 5
    finally:
        if ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_dlg05_catalog_selection_first_runs_all_six_cases_without_labels(
        tmp_path):
    """六 case 统一执行，并在文件 SQLite 重启后保持完整结果。"""
    catalog = build_dlg05_typed_preflight_catalog()
    language_compiler = build_dlg05_preflight_language_compiler(catalog)
    manifest = build_dlg05_typed_preflight_manifest()
    plans = build_dlg05_preflight_evidence_plans(catalog)
    route = minimal_instruction_identity((31005, 1200, 1))
    reason = minimal_instruction_identity((31005, 1200, 2))
    executor = ConversationHeldOutPreflightQuestionExecutor(
        route, reason, plans)
    base = _question_fixture()
    database = tmp_path / "dlg05-six-case-preflight.sqlite3"
    memory_backend, memory_ctx, _old_current, _old_center, consumer = (
        _memory_fixture(SQLiteBackend(str(database))))
    profile = _profile_for_understanding()
    source_records = SourceRecordRepository(memory_backend)
    owners = []
    cloned_backend = None
    try:
        _close_query(memory_ctx)
        # 每个独立 typed source 都有可回查原文；这属于 preflight fixture，
        # 不是 evaluator label 或 expected answer。
        for case in catalog.cases:
            for turn in case.turns:
                source_records.put_complete(
                    turn.request.source.stable_key(),
                    f"held-out-source-{case.case_key.components[-1]}",
                    metadata=SourceRecordMetadata(
                        "fixture-license",
                        case.case_key.components[-1],
                        400 + case.case_key.components[-1],
                        500 + case.case_key.components[-1],
                        600 + case.case_key.components[-1],
                    ),
                )

        class _EmptyCoreBaseline:
            """为 memory-miss 轴关闭 Core fallback，不改变 Memory 存储。"""

            def candidates(self, request):
                del request
                return ()

            def state_key(self):
                return (31005, 1198, 1)

        class _NoMatchMemoryFilter:
            """把 miss 查询限制到不存在的精确来源。"""

            def __init__(self, source):
                self.source = source

            def filters(self, request):
                del request
                return (MemoryAggregateFilter(source=self.source),)

            def state_key(self):
                return (31005, 1198, 2, *self.source.stable_key())

        def memory_plan_factory(_case, turn, _context_read):
            """为当前 request 打开精确 source/scope 的真实 DLG-04 read。"""
            request = catalog.turn_for(
                _case.case_key, turn.turn_key).request
            document = document_scope(request.source)
            episode = request.response_scope.parent
            assert episode is not None and episode.parent == document
            session = session_scope(
                1,
                owner=request.source.owner,
                versions=request.source.versions,
                source=request.source,
            )
            memory_ctx.work_memory.begin_session(session)
            memory_ctx.work_memory.begin_document(document)
            memory_ctx.work_memory.begin_episode(episode)
            memory_ctx.work_memory.begin_query(request.response_scope)
            current = _memory_current(
                memory_ctx, request.source, request.response_scope)
            center = _adapter().from_understanding(
                current, current.occurrences[0], strength="CONDITIONAL")
            resolver = memory_ctx.memory_resolver_runtime.resolver
            original_baseline = resolver.baseline_provider
            original_filter = resolver.index_filter_provider
            is_miss = AXIS_MEMORY_MISS in _case.axis_keys
            if is_miss:
                resolver.baseline_provider = _EmptyCoreBaseline()
                resolver.index_filter_provider = _NoMatchMemoryFilter(SourceRef(
                    request.source.source_kind,
                    request.source.source_id + 90_000,
                    request.source.document_id + 90_000,
                    request.source.owner,
                    request.source.versions,
                ))

            def release():
                """恢复本回合只读 resolver profile 并关闭 query 生命周期。"""
                if is_miss:
                    resolver.baseline_provider = original_baseline
                    resolver.index_filter_provider = original_filter
                _close_query(memory_ctx)

            return ConversationHeldOutMemoryPlan(
                consumer,
                current,
                center,
                profile,
                MemoryAccessContext(1, 2, 3),
                release=release,
            )

        class _Builder:
            """把实际 query result/receipt 接入现有 learned runtime factory。"""

            def build(self, request, context_read, memory_read):
                """从同次 query/Memory read 重新计算 G-01 stance。"""
                del context_read
                query = QuestionQuery(
                    request,
                    route,
                    (31005, 1201, request.trace[-2]),
                )
                if memory_read is None:
                    result = executor.execute(query)
                else:
                    for candidate_set in memory_read.resolution.sets:
                        for resolved in candidate_set.candidates:
                            if resolved.origin_kind != RESOLUTION_ORIGIN_MEMORY:
                                continue
                            for trace in resolved.memory_source_traces:
                                if source_records.find(trace.source.stable_key()) is None:
                                    _complete_source(
                                        source_records,
                                        trace,
                                        700 + source_records.source_count(),
                                    )
                    eligible = tuple(
                        resolved.stable_key()
                        for candidate_set in memory_read.resolution.sets
                        for resolved in candidate_set.candidates
                        if resolved.origin_kind == RESOLUTION_ORIGIN_MEMORY
                    )
                    result = ConversationMemoryQuestionExecutor(
                        memory_read,
                        request.target,
                        authorized_candidate_keys=eligible,
                        executed_reason=minimal_instruction_identity(
                            (31005, 1202, request.trace[-2])),
                        binding_reason=minimal_instruction_identity(
                            (31005, 1203, request.trace[-2])),
                        trace_prefix=(31005, 1204, request.trace[-2]),
                        source_records=source_records,
                    ).execute(query)
                planning = result.planning_request()
                selection = base.runtime.selector.select(planning)
                if selection.stance == base.content.answer:
                    selected = next(
                        item for item in planning.candidates
                        if item.stable_key()
                        == selection.selected_candidate_keys[0]
                    )
                    claim = claim_input_from_candidate(
                        selected, source_records)
                    owner = _answer_runtime(
                        1205 + request.trace[-2],
                        planning_override=planning,
                        candidate_override=selected,
                        branch_override=request.target_branch,
                        question_input_override=claim,
                        request_override=request,
                        selector_override=base.runtime.selector,
                        content_override=base.content,
                    )
                else:
                    response_act = {
                        base.content.unknown: "UNKNOWN",
                        base.content.clarify: "CLARIFY",
                        base.content.conflict: "CONFLICT",
                    }[selection.stance]
                    episode_id = {
                        "UNKNOWN": "train-grounded-unknown-budget-v1",
                        "CLARIFY": "train-grounded-clarify-site-v1",
                        "CONFLICT": "train-grounded-conflict-date-v1",
                    }[response_act]
                    owner = _response_act_runtime(
                        response_act,
                        episode_id,
                        1210 + request.trace[-2],
                        planning_override=planning,
                        question_input=GroundedResponseActQuestionInput(
                            response_act),
                        branch_override=request.target_branch,
                        query_kind_override=request.query_kind,
                        selector_override=base.runtime.selector,
                        content_override=base.content,
                    )
                owners.append(owner)
                return owner[2]

        resolver = MappedConversationHeldOutResponseActResolver((
            (base.content.unknown, RESPONSE_UNKNOWN),
            (base.content.clarify, RESPONSE_CLARIFY),
            (base.content.conflict, RESPONSE_CONFLICT),
            (base.content.answer, RESPONSE_ANSWER),
        ))
        factory = ConversationHeldOutCatalogTurnFactory(
            catalog,
            _Builder(),
            memory_plan_factory=memory_plan_factory,
            response_act_resolver=resolver,
            question_input_compiler=language_compiler,
            relation_input_compiler=build_dlg05_unseen_relation_compiler(
                catalog),
            source_input_compiler=build_dlg05_unseen_source_compiler(catalog),
        )
        execution = run_real_selection_first_receipt(manifest, factory)
        observations = execution.observations
        assert len(observations) == 6
        assert all(item.context_revision == 2 for item in observations)
        assert all(item.turn_response_acts for item in observations)
        assert any(
            RESPONSE_ANSWER in item.turn_response_acts
            for item in observations
        )
        assert sum(bool(item.memory_receipt_keys) for item in observations) == 5
        assert {
            axis
            for item in observations
            for axis in item.proven_axis_keys
        } == set(manifest.required_axes)
        # 首次公开 preflight 还负责物化新 source 的 typed query fixture 与
        # Memory SourceRecord；冻结后，fresh 重跑必须完全只读且逐项一致。
        frozen_backend = memory_backend.snapshot()
        fresh_execution = run_real_selection_first_receipt(manifest, factory)
        fresh = fresh_execution.observations
        assert fresh == observations
        assert memory_backend.snapshot() == frozen_backend

        host_ctx = memory_ctx
        host_consumer = consumer
        host_source_records = source_records
        cloned_backend = clone_backend(memory_backend)
        cloned_ctx = clone_train_context(
            host_ctx, cloned_backend, label="dlg05-six-case-preflight")
        memory_ctx = cloned_ctx
        consumer = ConversationMemoryDemandConsumer(
            cloned_ctx,
            cloned_ctx.memory_query_runtime,
            cloned_ctx.memory_resolver_runtime,
        )
        source_records = SourceRecordRepository(cloned_backend)
        cloned_before = cloned_backend.snapshot()
        clone_execution = run_real_selection_first_receipt(manifest, factory)
        cloned_observations = clone_execution.observations
        assert cloned_observations == observations
        assert cloned_backend.snapshot() == cloned_before
        memory_ctx = host_ctx
        consumer = host_consumer
        source_records = host_source_records

        # 将首次公开 fixture 提交到文件 SQLite，再真实关闭并重新打开。
        # 恢复只重建 TrainContext/M-06/M-07 的运行对象，不重复播种 Memory。
        memory_backend.commit()
        persisted_backend = memory_backend.snapshot()
        assert persisted_backend == frozen_backend
        memory_backend.close()
        reopened_backend = SQLiteBackend(str(database))
        memory_backend, memory_ctx, _old_current, _old_center, consumer = (
            _memory_fixture(reopened_backend, seed=False))
        _close_query(memory_ctx)
        source_records = SourceRecordRepository(memory_backend)
        assert memory_backend.snapshot() == persisted_backend

        resumed_before = memory_backend.snapshot()
        resumed_execution = run_real_selection_first_receipt(manifest, factory)
        resumed_observations = resumed_execution.observations
        assert resumed_observations == observations
        assert memory_backend.snapshot() == resumed_before

        # 在 rollback case 的 Memory read 之后注入真实持久化污染并抛错。
        # runner 必须关闭 query、恢复 SQLite 快照，且允许同输入确定重试。
        rollback_case = manifest.cases[-1]

        def faulting_factory(case, turn, context_read):
            """只在 rollback case 第二回合注入一次持久化写故障。"""
            plan = factory(case, turn, context_read)
            if case.case_key != rollback_case.case_key or turn.ordinal != 2:
                return plan

            def fail_after_read(_memory_read):
                """模拟 read 后、runtime 装配前发生半写并中断。"""
                memory_backend.insert("source_record", {
                    "source_hash": 9_876_543_210,
                    "text_hash": 1,
                    "codepoint_count": 5,
                    "source_kind": 9,
                    "source_id": 9,
                    "document_id": 9,
                    "corpus_version": 0,
                    "parser_version": 0,
                    "license_id": "",
                    "batch_id": 0,
                    "companion_type_hash": 0,
                    "companion_name_hash": 0,
                    "companion_assoc_id": 0,
                    "raw_text": "fault",
                })
                raise RuntimeError("DLG05_INJECTED_ROLLBACK_FAULT")

            return replace(plan, runtime_factory=fail_after_read)

        fault_before = memory_backend.snapshot()
        with pytest.raises(
                RuntimeError, match="DLG05_INJECTED_ROLLBACK_FAULT") as caught:
            run_real_selection_first(manifest, faulting_factory)
        assert memory_ctx.work_memory.active_query_scope is None
        assert memory_backend.snapshot() == fault_before
        fault_after = memory_backend.snapshot()
        recovered_execution = run_real_selection_first_receipt(
            manifest, factory)
        recovered_observations = recovered_execution.observations
        assert recovered_observations == observations
        assert memory_backend.snapshot() == fault_before
        recovered_after = memory_backend.snapshot()

        axis_audit = audit_dlg05_preflight_axis_inputs(
            catalog,
            compiler=language_compiler,
            relation_compiler=build_dlg05_unseen_relation_compiler(catalog),
            source_compiler=build_dlg05_unseen_source_compiler(catalog),
        )
        qualification = ConversationHeldOutQualificationReceipt(
            manifest.stable_key(),
            catalog.stable_key(),
            execution,
            fresh_execution,
            clone_execution,
            resumed_execution,
            ConversationHeldOutRollbackRecoveryReceipt(
                conversation_heldout_rollback_fault_key(caught.value),
                _snapshot_key(fault_before),
                _snapshot_key(fault_after),
                _snapshot_key(recovered_after),
                recovered_execution,
            ),
            (
                _snapshot_key(frozen_backend),
                _snapshot_key(memory_backend.snapshot()),
                _snapshot_key(cloned_before),
                _snapshot_key(persisted_backend),
                _snapshot_key(resumed_before),
                _snapshot_key(fault_before),
            ),
            axis_audit,
        )
        assert qualify_dlg05_preflight(
            catalog, manifest, qualification) is qualification
        freeze_target = os.environ.get("DLG05_PUBLIC_FREEZE_TARGET")
        if freeze_target:
            write_dlg05_public_freeze_document(
                freeze_target,
                Path(__file__).resolve().parents[1],
                catalog,
                manifest,
                qualification,
            )
    finally:
        if memory_ctx.work_memory.active_query_scope is not None:
            _close_query(memory_ctx)
        if cloned_backend is not None:
            cloned_backend.close()
        memory_backend.close()
        base.close()
        for owner in owners:
            if owner[1].fixture is not None:
                owner[1].fixture.close()
            owner[0].close()


def test_dlg05_real_answer_case_preserves_candidate_and_citation_receipt():
    """真实 ANSWER connector 两回合保留 G-01 candidate 与 G-04 citation。"""
    owner = _answer_runtime(61)
    memory_backend, memory_ctx, current, center, consumer = _memory_fixture()
    profile = _profile_for_understanding()
    requests = tuple(
        replace(owner[3], trace=(*owner[3].trace, 950 + ordinal))
        for ordinal in (1, 2)
    )
    family = ProtocolKey((20260819, 5, 701))
    case = _response_case(
        family,
        3,
        requests,
        AXIS_EXPLICIT_REPEAT,
        (MEMORY_OFF, MEMORY_ON),
    )
    manifest = ConversationHeldOutManifest(
        1,
        family,
        (),
        (),
        (),
        (case,),
        (AXIS_EXPLICIT_REPEAT,),
        (MEMORY_OFF, MEMORY_ON),
    )
    expected_selection = owner[2].selector.select(owner[5])
    expected_candidates = expected_selection.selected_candidate_keys
    expected_citations = tuple(
        source.stable_key()
        for source in owner[5].candidates[0].citation_sources
    )

    def prepare(_case, turn, _context_read):
        memory = None
        if turn.memory_mode == MEMORY_ON:
            memory = ConversationHeldOutMemoryPlan(
                consumer, current, center, profile,
                MemoryAccessContext(1, 2, 3),
            )
        return ConversationHeldOutTurnPlan(
            owner[2],
            requests[turn.ordinal - 1],
            owner[4],
            RESPONSE_ANSWER,
            memory,
        )

    labels = ConversationHeldOutLabelSet(
        manifest.stable_key(),
        (ConversationHeldOutLabel(
            case.case_key,
            RESPONSE_ANSWER,
            (RESPONSE_ANSWER, RESPONSE_ANSWER),
            expected_candidates,
            expected_citations,
        ),),
    )
    try:
        observations = run_real_selection_first(manifest, prepare)
        observation = observations[0]
        assert observation.response_act == RESPONSE_ANSWER
        assert observation.turn_response_acts == (RESPONSE_ANSWER,) * 2
        assert observation.selected_candidate_keys == expected_candidates
        assert observation.cited_source_keys == expected_citations
        assert len(observation.memory_receipt_keys) == 1
        assert observation.memory_causal_proven == 0
        result = evaluate_label_late(manifest, labels, observations)
        assert result.complete and result.passed == 1
    finally:
        _close_query(memory_ctx)
        memory_backend.close()
        if owner[1].fixture is not None:
            owner[1].fixture.close()
        owner[0].close()


def test_dlg05_detached_answer_factory_runs_without_course_episode():
    """独立 claim 输入可完整执行 ANSWER/G-04，不回读课程 episode。"""
    owner = _answer_runtime(64, detached_claim=True)
    try:
        run = owner[2].run(owner[3])
        assert run.complete
        assert run.selection is not None
        assert run.selection.selected_candidate_keys
        assert run.postcheck is not None and run.postcheck.complete
        assert run.postcheck.parsed.observation is not None
        assert run.postcheck.parsed.observation.cited_sources
    finally:
        if owner[1].fixture is not None:
            owner[1].fixture.close()
        owner[0].close()


def test_dlg05_memory_causal_factory_produces_answer_from_same_read():
    """同 source/scope 的 DLG-04 read 经过 factory 实际驱动 ANSWER/G-04。"""
    backend, ctx, source, _attractor, _strategy, compilation, goals = _setup()
    template_fixture = None
    materialized = []
    try:
        current = compilation.current
        center = _adapter().from_understanding(
            current, current.occurrences[0], strength="CONDITIONAL")
        consumer = ConversationMemoryDemandConsumer(
            ctx,
            ctx.memory_query_runtime,
            ctx.memory_resolver_runtime,
        )
        profile = _profile_for_understanding()
        resolution = ctx.memory_resolver_runtime.resolve(compilation)
        eligible = tuple(
            candidate
            for candidate_set in resolution.sets
            for candidate in candidate_set.candidates
            if (candidate.origin_kind == RESOLUTION_ORIGIN_MEMORY
                and candidate.memory_source_traces
                and all(trace.stance == EVIDENCE_SUPPORT
                        for trace in candidate.memory_source_traces))
        )
        assert eligible
        authorized = (eligible[0].stable_key(),)
        repository = SourceRecordRepository(backend)
        traces = {
            trace.source.stable_key(): trace
            for candidate in eligible[:1]
            for trace in candidate.memory_source_traces
        }
        for ordinal, trace in enumerate(traces.values(), start=1):
            _complete_source(repository, trace, ordinal)

        mapper, postchecker, _, _, _ = _postcheck_owners()
        template_fixture = _question_fixture(
            executor_factory=lambda route: _EmptyQuestionExecutor(
                _instruction(source, 61001)),
            world=(source, current.scope, goals[0].proposition),
            postcheck_mapper=mapper,
            postchecker=postchecker,
        )
        requests = tuple(
            replace(template_fixture.request, trace=(
                *template_fixture.request.trace, 61010 + ordinal))
            for ordinal in (1, 2)
        )
        family = ProtocolKey((20260819, 5, 702))
        case = _response_case(
            family,
            4,
            requests,
            AXIS_MEMORY_CAUSAL,
            (MEMORY_OFF, MEMORY_ON),
        )
        manifest = ConversationHeldOutManifest(
            1,
            family,
            (),
            (),
            (),
            (case,),
            (AXIS_MEMORY_CAUSAL,),
            (MEMORY_OFF, MEMORY_ON),
        )

        def materialize(read):
            assert read.receipt.status == "HIT"
            executor = ConversationMemoryQuestionExecutor(
                read,
                requests[1].target,
                authorized_candidate_keys=authorized,
                executed_reason=_instruction(source, 61002),
                binding_reason=_instruction(source, 61003),
                trace_prefix=(61004, 1),
                source_records=repository,
            )
            local_mapper, local_postchecker, _, _, _ = _postcheck_owners()
            fixture = _question_fixture(
                executor_factory=lambda route: executor,
                world=(source, current.scope, goals[0].proposition),
                postcheck_mapper=local_mapper,
                postchecker=local_postchecker,
            )
            assert fixture.request.query_kind == requests[1].query_kind
            assert fixture.request.source == requests[1].source
            assert fixture.request.response_scope == requests[1].response_scope
            materialized.append(fixture)
            return fixture.runtime

        def prepare(_case, turn, _context_read):
            if turn.memory_mode == MEMORY_OFF:
                return ConversationHeldOutTurnPlan(
                    template_fixture.runtime,
                    requests[0],
                    template_fixture.content.unknown,
                    RESPONSE_UNKNOWN,
                )
            return ConversationHeldOutTurnPlan(
                None,
                requests[1],
                template_fixture.content.answer,
                RESPONSE_ANSWER,
                ConversationHeldOutMemoryPlan(
                    consumer,
                    current,
                    center,
                    profile,
                    MemoryAccessContext(1, 2, 3),
                ),
                materialize,
            )

        observations = run_real_selection_first(manifest, prepare)
        observation = observations[0]
        assert observation.turn_response_acts == (
            RESPONSE_UNKNOWN, RESPONSE_ANSWER)
        assert observation.response_act == RESPONSE_ANSWER
        assert observation.selected_candidate_keys
        assert observation.cited_source_keys
        assert len(observation.memory_receipt_keys) == 1
        assert observation.memory_causal_proven == 1
    finally:
        for fixture in materialized:
            fixture.close()
        if template_fixture is not None:
            template_fixture.close()
        _close_query(ctx)
        backend.close()
