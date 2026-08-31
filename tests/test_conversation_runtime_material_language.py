import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import RuntimeMemoryState
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    ingest_runtime_material,
)
from pure_integer_ai.experiments.conversation_runtime_material_language import (
    RUNTIME_MATERIAL_RELATION_PRECEDES,
    RuntimeMaterialLanguageError,
    observe_runtime_material_language,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_runtime_material_response import (
    build_runtime_material_answer_provider,
    build_runtime_material_response_provider,
    organize_runtime_material_response,
    RuntimeMaterialResponseSpec,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)


def test_runtime_material_uses_existing_language_observation_and_relation_surface():
    backend = DictBackend()
    bootstrap(backend)
    source = SourceRef(91, 7301, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(7301, source=source)
    ctx = make_train_context(backend, companion=True)
    companion_identity = ctx.memory_read_intake.source_intake.companion.identity
    ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()),
        source=source,
        scope=scope,
        raw_text="夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        source_records=ctx.memory_read_intake.source_intake.repository,
        metadata=SourceRecordMetadata(
            "CC0-1.0", 1, companion_identity.type_hash,
            companion_identity.name_hash, 4),
        source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 7301),
        authority_key=(7, 7301),
    )
    result = observe_runtime_material_language(
        ctx,
        ingest,
        observation_id="obs-language-7301",
        context_id="ctx-manual",
        family_id="family-manual",
        source_namespace="runtime-language-test",
    )

    assert result.runtime_space_only
    assert result.input_payload.stage == 3
    assert len(result.raw_observation.units) == 2
    assert len(result.segments) == 2
    assert result.observation.built_concepts > 0
    assert result.observation.built_edges > 0
    assert len(result.lexical_evidence) == 2
    assert len(result.relation_candidates) == 1
    relation = result.relation_candidates[0]
    assert relation.proposition.relation_kind == RUNTIME_MATERIAL_RELATION_PRECEDES
    assert tuple(item.evidence_id for item in relation.evidence) == tuple(
        item.evidence_id for item in relation.proposition.arguments)
    qualification = RawPropositionQualification(
        "qualification-runtime-7301",
        relation.proposition.proposition_id,
        result.raw_observation.observation_id,
        result.raw_observation.source_id,
        result.raw_observation.context_id,
        result.raw_observation.family_id,
        result.raw_observation.source_namespace,
        result.raw_observation.split,
        "SUPPORTED",
        "explicit-runtime-source-qualification",
        tuple(item.evidence_id for item in relation.evidence),
        "runtime-owner",
    )
    gate = relation.qualification_gate(qualification)
    assert gate.response_act == "ANSWER"
    provider = build_runtime_material_answer_provider(
        result,
        qualification,
        source_records=ctx.memory_read_intake.source_intake.repository,
        question="夜间模式与设置的先后关系是什么？",
        source_title="Runtime 手册",
    )
    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((7, 73)),
        "夜间模式与设置的先后关系是什么？",
        object(),
        runtime_material_answer=provider.answer,
    )
    assert turn.status == "ANSWER"
    assert turn.answer == "夜间模式会降低屏幕亮度。长按菜单键可打开设置。"
    assert turn.source_title == "Runtime 手册"
    organized = organize_runtime_material_response(
        ingest, gate, support_surfaces=("来源：Runtime 资料。",))
    assert organized.response_act == "ANSWER"
    assert "Night mode" not in organized.output_surface
    assert result.stable_key() == result.stable_key()


def test_runtime_material_generation_context_carries_structure_and_proposition():
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    source = SourceRef(91, 7310, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(7310, source=source)
    ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
        raw_text="模式会降低亮度。按键可打开设置。",
        source_records=repository,
        metadata=SourceRecordMetadata(
            "CC0-1.0", 1, companion.identity.type_hash,
            companion.identity.name_hash, 4),
        source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 7310), authority_key=(7, 7310),
    )
    observed = observe_runtime_material_language(
        ctx, ingest, observation_id="obs-language-7310",
        context_id="ctx-manual", family_id="family-manual",
        source_namespace="runtime-language-test",
    )
    relation = observed.relation_candidates[0]
    qualification = RawPropositionQualification(
        "qualification-runtime-7310",
        relation.proposition.proposition_id,
        observed.raw_observation.observation_id,
        observed.raw_observation.source_id,
        observed.raw_observation.context_id,
        observed.raw_observation.family_id,
        observed.raw_observation.source_namespace,
        observed.raw_observation.split,
        "SUPPORTED", "explicit-runtime-source-qualification",
        tuple(item.evidence_id for item in relation.evidence),
        "runtime-owner",
    )
    provider = build_runtime_material_response_provider(
        (RuntimeMaterialResponseSpec(
            observed, qualification, "模式与设置的先后关系是什么？",
            source_title="Runtime 手册"),),
        source_records=repository,
    )
    result = provider.response_with_generation_context(
        "模式与设置的先后关系是什么？")
    assert result is not None
    status, answer, _title, _url, citations, generation = result
    assert status == "ANSWER"
    assert answer is not None and citations
    assert generation.response_act == "ANSWER"
    assert generation.evidence[0].structure_refs == tuple(
        tuple(ref) for ref in observed.observation.struct_refs)
    assert generation.evidence[0].proposition_keys == (
        relation.proposition.canonical_record(),)
    assert generation.identity_key == generation.identity_key


def test_runtime_material_mechanical_boundaries_project_through_split_winners():
    """句界按来源 scalar 端点投影，不要求机械 unit 等于 token。"""
    from pure_integer_ai.experiments.conversation_runtime_material_language import (
        _MechanicalBoundary,
    )

    boundary = _MechanicalBoundary((3, 7))
    assert boundary.token_cuts(
        ((0, 1, 0), (1, 3, 0), (3, 5, 0), (5, 7, 0))) == (2, 4)
    with pytest.raises(RuntimeMaterialLanguageError):
        boundary.token_cuts(((0, 1, 0), (1, 2, 0), (2, 4, 0)))


def test_runtime_material_multi_source_and_conflict_block_qa_fallback():
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion

    def observed(source_id: int, text: str):
        source = SourceRef(91, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        scope = session_scope(source_id, source=source)
        ingest = ingest_runtime_material(
            RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
            raw_text=text, source_records=repository,
            metadata=SourceRecordMetadata(
                "CC0-1.0", 1, companion.identity.type_hash,
                companion.identity.name_hash, 4),
            source_intake=ctx.memory_read_intake.source_intake,
            version_key=(1, source_id), authority_key=(7, source_id),
        )
        return observe_runtime_material_language(
            ctx, ingest, observation_id=f"obs-{source_id}",
            context_id=f"ctx-{source_id}", family_id="family-manual",
            source_namespace="runtime-multi-source",
        )

    first = observed(7401, "夜间模式会降低屏幕亮度。长按菜单键可打开设置。")
    second = observed(7402, "夜间模式会降低屏幕亮度。按住菜单键即可打开设置。")
    question = "夜间模式与设置的先后关系是什么？"

    def qualification(item, identifier: str, state: str):
        relation = item.relation_candidates[0]
        return RawPropositionQualification(
            identifier, relation.proposition.proposition_id,
            item.raw_observation.observation_id, item.raw_observation.source_id,
            item.raw_observation.context_id, item.raw_observation.family_id,
            item.raw_observation.source_namespace, item.raw_observation.split,
            state, "independent-runtime-source", tuple(
                evidence.evidence_id for evidence in relation.evidence),
            "runtime-owner",
        )

    provider = build_runtime_material_response_provider(
        (
            RuntimeMaterialResponseSpec(
                first, qualification(first, "qual-7401", "SUPPORTED"),
                question, source_title="手册A"),
            RuntimeMaterialResponseSpec(
                second, qualification(second, "qual-7402", "SUPPORTED"),
                question, source_title="手册B"),
        ),
        source_records=repository,
    )
    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((7, 74)), question, object(),
        runtime_material_response=provider.response_with_citations,
    )
    assert turn.status == "ANSWER"
    assert "长按菜单键可打开设置" in turn.answer
    assert "按住菜单键即可打开设置" in turn.answer
    assert "手册A" in turn.source_title
    assert "手册B" in turn.source_title
    assert tuple(item.source_title for item in turn.citations) == ("手册A", "手册B")
    assert tuple(item.surface for item in turn.citations) == tuple(
        item for item in turn.answer.split("\n"))

    conflict = build_runtime_material_response_provider(
        (
            RuntimeMaterialResponseSpec(
                first, qualification(first, "qual-7401-c", "CONFLICT"),
                question, source_title="手册A"),
            RuntimeMaterialResponseSpec(
                second, qualification(second, "qual-7402-c", "SUPPORTED"),
                question, source_title="手册B"),
        ),
        source_records=repository,
    )
    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((7, 75)), question, object(),
        narrow_answer=lambda _question: ("不应绕过资格的答案。", "ANSWER"),
        runtime_material_response=conflict.response,
    )
    assert turn.status == "CLARIFY"
    assert turn.answer is None
