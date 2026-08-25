import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    ADMISSION_DUPLICATE,
    RuntimeMemoryState,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    RUNTIME_MATERIAL_READ_UNKNOWN,
    RuntimeMaterialAnswerBinding,
    RuntimeMaterialAnswerProvider,
    RuntimeMaterialIngestError,
    RuntimeMaterialReadIndex,
    RuntimeMaterialQualificationGate,
    compile_runtime_material_observation,
    compile_runtime_material_structure_observation,
    ingest_runtime_material,
    persist_runtime_material_checkpoint,
)
from pure_integer_ai.experiments.conversation_runtime_material_response import (
    organize_runtime_material_response,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import BroadDialogueState
from pure_integer_ai.experiments.conversation_broad_qa_runtime import answer_broad_dialogue_turn
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import RawLexicalEvidence, RawLexicalEvidenceBinding
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import RawPropositionQualification
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import RawPropositionRelationBinding, RawPropositionRelationEvidence, RawRelationArgument
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.k_run_boundary import create_new_run_root
from pure_integer_ai.storage.source_record import SourceRecordMetadata
from pure_integer_ai.storage.source_record import SourceRecordRepository


def _setup():
    backend = DictBackend()
    bootstrap(backend)
    repository = SourceRecordRepository(backend)
    source = SourceRef(91, 1001, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(1001, source=source)
    memory = RuntimeMemoryState(scope.stable_key())
    metadata = SourceRecordMetadata("CC0-1.0", 1, 11, 12, 13)
    return repository, source, scope, memory, metadata


def _gate(state: str = "SUPPORTED") -> RuntimeMaterialQualificationGate:
    lexical = RawLexicalEvidenceBinding(
        "obs-1", "material-evidence", "unit-1", "surface",
        (1,), (1,), (1, 2, 3),
    )
    binding = RawPropositionRelationBinding(
        "prop-1", "ASSERTS", (lexical,), (4, 5, 6),
    )
    qualification = RawPropositionQualification(
        "qual-1", "prop-1", "obs-1", "source-1", "context-1",
        "family-1", "runtime-material-test", "train", state,
        "fixture", ("material-evidence",), "authority",
    )
    return RuntimeMaterialQualificationGate.from_relation(
        "material-evidence", binding, qualification)


def _annotated_gate(ingest, state: str = "SUPPORTED"):
    observation = compile_runtime_material_observation(
        ingest,
        observation_id="obs-material-1", context_id="context-1",
        family_id="family-1", source_namespace="runtime-material-test",
        split="train", unit_id="unit-1", unit_role="material",
    )
    scalar_count = len(observation.scalars)
    evidence = RawLexicalEvidence(
        "material-evidence", observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "unit-1", "material",
        "fixture-authority", 0, scalar_count, 0, len(observation.raw_bytes),
    )
    proposition = RawPropositionRelationEvidence(
        "prop-material-1", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id, observation.source_namespace,
        observation.split, "ASSERTS", "fixture-authority",
        (RawRelationArgument("material-evidence", "unit-1", "claim", 1),),
    )
    qualification = RawPropositionQualification(
        "qual-material-1", proposition.proposition_id,
        observation.observation_id, observation.source_id, observation.context_id,
        observation.family_id, observation.source_namespace, observation.split,
        state, "fixture", ("material-evidence",), "fixture-authority",
    )
    return RuntimeMaterialQualificationGate.from_annotation(
        "material-evidence", observation, (evidence,), proposition, qualification)


def test_user_material_runtime_ingest_read_and_explicit_promotion(tmp_path) -> None:
    repository, source, scope, memory, metadata = _setup()
    first = ingest_runtime_material(
        memory,
        source=source,
        scope=scope,
        raw_text="玩家手册：夜间模式会降低屏幕亮度。",
        source_records=repository,
        metadata=metadata,
        version_key=(1, 1001),
        authority_key=(7, 1001),
    )
    assert first.admission_status == ADMISSION_ACCEPTED
    request = first.promotion_request(
        authority_key=(8, 1001), consent_key=(9, 1001))
    assert request.runtime_event_key == first.event.event_key
    assert request.evidence_keys

    replay = ingest_runtime_material(
        first.memory_after,
        source=source,
        scope=scope,
        raw_text="玩家手册：夜间模式会降低屏幕亮度。",
        source_records=repository,
        metadata=metadata,
        version_key=(1, 1001),
        authority_key=(7, 1001),
    )
    assert replay.admission_status == ADMISSION_DUPLICATE
    index = RuntimeMaterialReadIndex.build(replay.memory_after)
    read = index.read(first.event.memory_item_key, repository)
    assert read.event == first.event
    assert read.source_record.raw_text.startswith("玩家手册")
    assert read.physical_read_count == 1

    root = create_new_run_root(tmp_path / "runtime-material-session",
                               require_k_drive=False)
    persist_runtime_material_checkpoint(
        root, BroadDialogueState((3, 4), 0, ()), first)

    provider = RuntimeMaterialAnswerProvider(
        RuntimeMaterialReadIndex.build(first.memory_after), repository,
        (RuntimeMaterialAnswerBinding(
            "夜间模式是什么？", first.event.memory_item_key,
            _annotated_gate(first),
            "游戏手册", "https://example.invalid/manual"),),
    )
    _, turn = answer_broad_dialogue_turn(
        BroadDialogueState((3, 4)), "夜间模式是什么？", object(),
        runtime_material_answer=provider.answer)
    assert turn.status == "ANSWER"
    assert "降低屏幕亮度" in turn.answer
    assert turn.source_title == "游戏手册"

    organized = organize_runtime_material_response(
        first, _annotated_gate(first), support_surfaces=("来源：游戏手册。",))
    assert organized.response_act == "ANSWER"
    assert organized.output_surface == (
        "玩家手册：夜间模式会降低屏幕亮度。\n来源：游戏手册。")

    unknown_provider = RuntimeMaterialAnswerProvider(
        RuntimeMaterialReadIndex.build(first.memory_after), repository,
        (RuntimeMaterialAnswerBinding(
            "夜间模式未知？", first.event.memory_item_key,
            _annotated_gate(first, "UNKNOWN"), "游戏手册", None),),
    )
    assert unknown_provider.answer("夜间模式未知？") is None
    unknown_plan = organize_runtime_material_response(
        first, _annotated_gate(first, "UNKNOWN"),
        fallback_surfaces=("该资料目前不足以确认。",))
    assert unknown_plan.response_act == "UNKNOWN"
    assert unknown_plan.output_surface == "该资料目前不足以确认。"


def test_unknown_and_missing_metadata_fail_closed() -> None:
    repository, source, scope, memory, metadata = _setup()
    with pytest.raises(RuntimeMaterialIngestError, match="完整"):
        ingest_runtime_material(
            memory, source=source, scope=scope, raw_text="资料",
            source_records=repository,
            metadata=SourceRecordMetadata(),
            version_key=(1, 1001), authority_key=(7, 1001),
        )
    first = ingest_runtime_material(
        memory, source=source, scope=scope, raw_text="资料",
        source_records=repository, metadata=metadata,
        version_key=(1, 1001), authority_key=(7, 1001),
    )
    read = RuntimeMaterialReadIndex.build(first.memory_after).read(
        (999, 1), repository)
    assert read.status == RUNTIME_MATERIAL_READ_UNKNOWN


def test_source_intake_rejects_companion_identity_drift() -> None:
    repository, source, scope, memory, metadata = _setup()
    from pure_integer_ai.cognition.understanding.source_intake import SourceIntake
    from pure_integer_ai.storage.spaces.registry import SpaceRegistry
    from pure_integer_ai.storage.spaces.companion import CompanionSpace
    registry = SpaceRegistry(repository.backend)
    companion = CompanionSpace.create(registry, "runtime-material-companion")
    intake = SourceIntake(repository, companion)
    with pytest.raises(RuntimeMaterialIngestError, match="Companion identity"):
        ingest_runtime_material(
            memory,
            source=source,
            scope=scope,
            raw_text="资料",
            source_records=repository,
            metadata=metadata,
            source_intake=intake,
            version_key=(1, 1001),
            authority_key=(7, 1001),
        )


def test_runtime_material_uses_mechanical_structure_then_multiunit_qualification():
    repository, source, scope, memory, metadata = _setup()
    ingest = ingest_runtime_material(
        memory,
        source=source,
        scope=scope,
        raw_text="夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        source_records=repository,
        metadata=metadata,
        version_key=(1, 2001),
        authority_key=(7, 2001),
    )
    observation, extraction, coverage = (
        compile_runtime_material_structure_observation(
            ingest,
            observation_id="obs-structured-material",
            context_id="context-structured",
            family_id="family-manual",
            source_namespace="runtime-material-structure",
            split="train",
        ))
    assert len(extraction.candidates) == 2
    assert len(observation.units) == 2
    assert coverage.complete

    evidence = tuple(
        RawLexicalEvidence(
            f"material-evidence-{unit.unit_id}",
            observation.observation_id,
            observation.source_id,
            observation.context_id,
            observation.family_id,
            observation.source_namespace,
            observation.split,
            unit.unit_id,
            unit.role,
            "manual-authority",
            unit.start_scalar,
            unit.end_scalar,
            unit.start_byte,
            unit.end_byte,
        )
        for unit in observation.units
    )
    proposition = RawPropositionRelationEvidence(
        "prop-structured-material", observation.observation_id,
        observation.source_id, observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "ASSERTS",
        "manual-authority",
        tuple(
            RawRelationArgument(item.evidence_id, item.unit_id, "claim", index)
            for index, item in enumerate(evidence, start=1)
        ),
    )
    qualification = RawPropositionQualification(
        "qual-structured-material", proposition.proposition_id,
        observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "SUPPORTED",
        "manual-qualification",
        tuple(item.evidence_id for item in evidence),
        "manual-authority",
    )
    gate = RuntimeMaterialQualificationGate.from_annotation(
        "material-evidence-unit-0", observation, evidence,
        proposition, qualification,
    )
    assert gate.response_act == "ANSWER"
    plan = organize_runtime_material_response(
        ingest, gate, support_surfaces=("来源：用户手册。",))
    assert plan.response_act == "ANSWER"
    assert "夜间模式会降低屏幕亮度" in plan.output_surface
    assert "长按菜单键可打开设置" in plan.output_surface
