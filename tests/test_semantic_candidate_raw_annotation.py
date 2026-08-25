from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import RuntimeMemoryState
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    context_scope_identity,
    proposition_identity,
    proposition_hypothesis_key,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticPropositionCandidate,
    SemanticPropositionSpec,
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
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    compile_runtime_material_observation,
    ingest_runtime_material,
)
from pure_integer_ai.experiments.semantic_candidate_raw_annotation import (
    SemanticCandidateRawAnnotation,
    SemanticRawAnnotationError,
    semantic_candidate_proposition_id,
)
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.source_record import (
    SourceRecordMetadata,
    SourceRecordRepository,
)


def _candidate(source: SourceRef, scope):
    anchor = span_identity(source, members=((0, 3),))
    upstream = HypothesisKey((7001, 1), (7001, 2), (7001, 3), scope, source)
    proposition = proposition_identity(source, (7001, 4))
    spec = SemanticPropositionSpec(
        (7001, 5), (7001, 6), concept_identity((7001, 7)),
        structure_concept_identity((7001, 8)), (), None,
    )
    definition = AtomicPropositionDefinition(
        proposition, spec.predicate, anchor,
        context_scope_identity(source, (7001, 9)),
        (),
    )
    hypothesis = proposition_hypothesis_key(
        proposition, hypothesis_kind=(7001, 10),
        competition_key=spec.competition_key, scope=scope,
    )
    return SemanticPropositionCandidate(
        spec, definition, hypothesis,
        minimal_instruction_identity((7001, 11)), upstream,
    )


def test_semantic_candidate_closes_runtime_material_g0_g3() -> None:
    backend = DictBackend()
    try:
        bootstrap(backend)
        records = SourceRecordRepository(backend)
        source = SourceRef(91, 7001, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        scope = document_scope(source)
        ingest = ingest_runtime_material(
            RuntimeMemoryState(scope.stable_key()),
            source=source, scope=scope, raw_text="甲乙丙",
            source_records=records,
            metadata=SourceRecordMetadata("CC0-1.0", 1, 2, 3, 4),
            version_key=(1, 1), authority_key=(1, 2),
        )
        candidate = _candidate(source, scope)
        observation = compile_runtime_material_observation(
            ingest, observation_id="obs-semantic-1", context_id="ctx-1",
            family_id="family-1", source_namespace="semantic-raw-test",
            split="train", unit_id="unit-1", unit_role="claim",
        )
        lexical = RawLexicalEvidence(
            "evidence-1", observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "unit-1", "claim",
            "annotator", 0, 3, 0, len(observation.raw_bytes),
        )
        proposition = RawPropositionRelationEvidence(
            semantic_candidate_proposition_id(candidate),
            observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "ASSERTS", "annotator",
            (RawRelationArgument("evidence-1", "unit-1", "claim", 1),),
        )
        qualification = RawPropositionQualification(
            "qualification-1", proposition.proposition_id,
            observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "SUPPORTED", "source",
            ("evidence-1",), "annotator",
        )
        annotation = SemanticCandidateRawAnnotation.from_runtime_material(
            ingest, candidate, observation, (lexical,), proposition,
            qualification, material_evidence_id="evidence-1",
            anchor_unit_id="unit-1",
        )
        assert annotation.qualification_gate().response_act == "ANSWER"
        assert annotation.stable_key()[0] == 1
        assert all(type(item) is int and item >= 0 for item in annotation.stable_key())
    finally:
        backend.close()


def test_semantic_candidate_identity_or_anchor_mismatch_fails_closed() -> None:
    backend = DictBackend()
    try:
        bootstrap(backend)
        records = SourceRecordRepository(backend)
        source = SourceRef(91, 7002, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        scope = document_scope(source)
        ingest = ingest_runtime_material(
            RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
            raw_text="甲乙丙", source_records=records,
            metadata=SourceRecordMetadata("CC0-1.0", 1, 2, 3, 4),
            version_key=(1, 1), authority_key=(1, 2),
        )
        candidate = _candidate(source, scope)
        observation = compile_runtime_material_observation(
            ingest, observation_id="obs-semantic-2", context_id="ctx-1",
            family_id="family-1", source_namespace="semantic-raw-test",
            split="train", unit_id="unit-1", unit_role="claim",
        )
        lexical = RawLexicalEvidence(
            "evidence-2", observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "unit-1", "claim",
            "annotator", 0, 3, 0, len(observation.raw_bytes),
        )
        proposition = RawPropositionRelationEvidence(
            "wrong-proposition-id", observation.observation_id,
            observation.source_id, observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "ASSERTS", "annotator",
            (RawRelationArgument("evidence-2", "unit-1", "claim", 1),),
        )
        qualification = RawPropositionQualification(
            "qualification-2", proposition.proposition_id,
            observation.observation_id, observation.source_id,
            observation.context_id, observation.family_id,
            observation.source_namespace, observation.split, "SUPPORTED", "source",
            ("evidence-2",), "annotator",
        )
        with pytest.raises(SemanticRawAnnotationError, match="proposition_id"):
            SemanticCandidateRawAnnotation.from_runtime_material(
                ingest, candidate, observation, (lexical,), proposition,
                qualification, material_evidence_id="evidence-2",
                anchor_unit_id="unit-1",
            )
    finally:
        backend.close()
