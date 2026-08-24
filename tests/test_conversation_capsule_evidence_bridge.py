"""M3 capsule -> raw evidence -> qualification -> response-act 纵切。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import (
    CoreDelta,
    CoreLearningState,
    LearningInputCapsule,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_capsule_evidence_bridge import (
    CapsuleEvidenceBridgeError,
    run_capsule_evidence_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    start_public_frame_dialogue,
)
from pure_integer_ai.experiments.conversation_raw_lexical_evidence import (
    RawLexicalEvidence,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_raw_proposition_evidence import (
    RawPropositionRelationEvidence,
    RawRelationArgument,
)
from pure_integer_ai.experiments.conversation_raw_text_observation import (
    RawTextSpanUnit,
    compile_raw_text_observation,
)


_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    closure = load_public_source_payload_closure_from_root(_ROOT)
    return build_public_dialogue_runtime_v1(closure)


def _fixture(runtime, *, qualification_state: str = "SUPPORTED"):
    frame = next(
        item for item in runtime.base_catalog.frames
        if item.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    raw = frame.surface_bytes
    source = SourceRef(88003, 1, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
    capsule = LearningInputCapsule(
        source=source,
        scope=session_scope(88003, source=source),
        version_key=(1, 88003),
        parent_version_key=(),
        language=1,
        modality=1,
        raw_content_digest=digest_bytes(bytes(raw)),
        structural_units=((1, len(raw)),),
        authority_key=(1, 88003),
        license_id="CC0-1.0",
        split=1,
        delta_sequence=1,
    )
    scalar_count = len(tuple(frame.surface_scalars))
    observation = compile_raw_text_observation(
        raw,
        observation_id="obs-88003-1",
        source_id="source-88003",
        context_id="context-1",
        family_id="family-1",
        source_namespace="capsule-evidence-test",
        split="train",
        units=(RawTextSpanUnit(
            "unit-1", "surface", 0, scalar_count, 0, len(raw)),),
    )
    evidence = RawLexicalEvidence(
        "evidence-1", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "unit-1", "surface",
        "fixture-authority", 0, scalar_count, 0, len(raw),
    )
    proposition = RawPropositionRelationEvidence(
        "proposition-1", observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id,
        observation.source_namespace, observation.split, "ASSERTS",
        "fixture-authority",
        (RawRelationArgument("evidence-1", "unit-1", "claim", 1),),
    )
    qualification = RawPropositionQualification(
        "qualification-1", proposition.proposition_id,
        observation.observation_id, observation.source_id,
        observation.context_id, observation.family_id,
        observation.source_namespace, observation.split,
        qualification_state, "fixture-state", ("evidence-1",),
        "fixture-authority",
    )
    core = CoreLearningState(capsule.scope.stable_key(), (701,))
    delta = CoreDelta((701,), capsule, graph_diff=(13, 14))
    memory = RuntimeMemoryState(capsule.scope.stable_key())
    dialogue = start_public_frame_dialogue((88003, 1))
    return (capsule, raw, observation, (evidence,), proposition,
            qualification, delta, core, dialogue, memory)


def test_qualified_evidence_produces_answer_obligation(runtime) -> None:
    values = _fixture(runtime)
    result = run_capsule_evidence_dialogue_turn(*values, runtime)

    assert result.response_act == "ANSWER"
    assert result.consumer_result.state == "SUPPORTED"
    assert result.dual_plane.core_after.deltas == (values[6],)
    assert result.dual_plane.dialogue.dialogue_turn.after.next_operation_ordinal == 2
    assert result.canonical_record()


def test_unknown_qualification_stays_unknown(runtime) -> None:
    values = _fixture(runtime, qualification_state="UNKNOWN")
    result = run_capsule_evidence_dialogue_turn(*values, runtime)

    assert result.response_act == "UNKNOWN"
    assert result.consumer_result.state == "UNKNOWN"


def test_observation_digest_mismatch_fails_before_learning(runtime) -> None:
    values = list(_fixture(runtime))
    values[1] = (*values[1], 0)
    with pytest.raises(CapsuleEvidenceBridgeError, match="observation"):
        run_capsule_evidence_dialogue_turn(*values, runtime)
