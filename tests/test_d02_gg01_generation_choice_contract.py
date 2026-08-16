"""GG-01 生成上下文、分层选择与零写装载测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    OwnerScope,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.scope_identity import query_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GG01ContractManifest,
    GenerationAddresseeContext,
    GenerationChoiceCandidateMapper,
    GenerationChoiceCandidateProtocol,
    GenerationChoiceCondition,
    GenerationChoiceContractError,
    GenerationChoiceHypothesis,
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationContentObligation,
    GenerationContextContract,
    GenerationDiscourseState,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
    build_gg01_contract_manifest,
    build_gg01_contract_manifest_v2,
    read_gg01_contract_manifest,
    write_gg01_contract_manifest,
)
from pure_integer_ai.storage.backend import DictBackend
from tests.test_r00_relation_closure import (
    _candidate_runtime,
    _projection_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v23.json"
MANIFEST_PATH = REPO_ROOT / "data/ph2/manifests/gg01_generation_choice_contract_v1.json"
MANIFEST_V2_PATH = REPO_ROOT / "data/ph2/manifests/gg01_generation_choice_contract_v2.json"


def _source(source_id: int) -> SourceRef:
    return SourceRef(
        771, source_id, 0, GLOBAL_OWNER_SCOPE, VersionBundle())


def _context(seed: int = 1) -> GenerationContextContract:
    source = _source(seed)
    scope = query_scope(seed, source=source)
    proposition = proposition_identity(source, (19000, seed, 1))
    visible = concept_identity((19000, seed, 2))
    return GenerationContextContract(
        context_scope_identity(source, (19000, seed, 3)),
        minimal_instruction_identity((19000, seed, 4)),
        concept_identity((19000, seed, 5)),
        scope,
        GenerationAddresseeContext(
            concept_identity((19000, seed, 6)),
            (visible,),
            (),
            (visible,),
        ),
        (GenerationContentObligation(
            minimal_instruction_identity((19000, seed, 7)),
            "REQUIRED",
            proposition,
            (source,),
            concept_identity((19000, seed, 8)),
        ),),
        GenerationExpressionConstraints(
            language_branch_identity((19000, seed, 9)),
            (structure_concept_identity((19000, seed, 10)),),
            (language_branch_identity((19000, seed, 11)),),
            1,
            0,
            1,
            64,
        ),
        GenerationDiscourseState(
            (concept_identity((19000, seed, 12)),),
            concept_identity((19000, seed, 13)),
            (proposition,),
            (concept_identity((19000, seed, 14)),),
        ),
    )


def _choice(
        choice_kind: str = "CONTENT_CHOICE", *, seed: int = 1,
        with_outcome: bool = False,
        ) -> GenerationChoiceHypothesis:
    context = _context(seed)
    obligation = context.content_obligations[0].obligation
    condition = GenerationChoiceCondition(
        concept_identity((19100, seed, 1)),
        context.context,
        (context.addressee.recoverable_references[0],),
        (concept_identity((19100, seed, 2)),),
        context.scope,
    )
    uses = ()
    outcomes = ()
    if with_outcome:
        use = GenerationChoiceUseRef(
            "CORE_USE", LosslessIntegerKey((seed, 0, 1)),
            LosslessIntegerKey((seed, 0, 2)), context.scope)
        uses = (use,)
        outcomes = (GenerationChoiceOutcomeRef(
            LosslessIntegerKey((seed, 0, 3)), use.use_key,
            LosslessIntegerKey((seed, 0, 4)), LosslessIntegerKey((seed, 0, 5)),
            LosslessIntegerKey((seed, 0, 6))),)
    return GenerationChoiceHypothesis(
        ObjectIdentity(OBJECT_HYPOTHESIS, (19100, seed, 3)),
        choice_kind,
        obligation,
        condition,
        context.content_obligations[0].proposition,
        (_source(seed + 100), _source(seed + 200)),
        (19100, seed, 4),
        context.scope,
        uses,
        outcomes,
    )


def _candidate_protocol(seed: int = 1) -> GenerationChoiceCandidateProtocol:
    predicates = tuple(
        concept_identity((19200, seed, ordinal)) for ordinal in range(1, 7))
    return GenerationChoiceCandidateProtocol(
        *predicates,
        concept_identity((19200, seed, 7)),
        tuple(concept_identity((19200, seed, 20 + ordinal))
              for ordinal in range(len(CHOICE_KINDS))),
        (19200, seed, 30),
    )


def test_generation_context_round_trip_binds_goal_and_all_required_fields():
    context = _context()
    restored = GenerationContextContract.from_dict(context.to_dict())
    assert restored == context
    assert restored.stable_key() == context.stable_key()
    assert restored.goal != restored.context
    assert restored.addressee.recoverable_references
    assert restored.content_obligations[0].requirement == "REQUIRED"
    assert restored.expression_constraints.max_output_units == 64
    assert restored.discourse_state.revision_dependencies


def test_context_rejects_owner_drift_conflicting_obligation_and_hidden_reference():
    context = _context()
    foreign_owner = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
    with pytest.raises(GenerationChoiceContractError, match="owner or version"):
        replace(
            context,
            communicative_goal=concept_identity(
                (19300, 1), owner=foreign_owner),
        )
    forbidden = replace(
        context.content_obligations[0],
        obligation=minimal_instruction_identity((19300, 2)),
        requirement="FORBIDDEN",
    )
    with pytest.raises(GenerationChoiceContractError, match="competing"):
        replace(context, content_obligations=(
            context.content_obligations[0], forbidden))
    with pytest.raises(GenerationChoiceContractError, match="shared-visible"):
        replace(
            context.addressee,
            recoverable_references=(concept_identity((19300, 3)),),
        )


def test_five_choice_layers_round_trip_without_complete_answer_template():
    choices = tuple(_choice(kind, seed=index + 1, with_outcome=True)
                    for index, kind in enumerate(CHOICE_KINDS))
    assert tuple(item.choice_kind for item in choices) == CHOICE_KINDS
    for choice in choices:
        restored = GenerationChoiceHypothesis.from_dict(choice.to_dict())
        assert restored == choice
        assert restored.stable_key() == choice.stable_key()
        assert restored.typed_outcomes[0].use_key == restored.exact_uses[0].use_key
        assert "answer" not in restored.to_dict()
        assert "surface" not in restored.to_dict()


def test_choice_condition_scope_and_exact_outcome_fail_closed():
    choice = _choice(with_outcome=True)
    with pytest.raises(GenerationChoiceContractError, match="require and forbid"):
        replace(
            choice.condition,
            forbidden_context_objects=(
                choice.condition.required_context_objects),
        )
    other_scope = query_scope(999, source=_source(1))
    with pytest.raises(GenerationChoiceContractError, match="scopes differ"):
        replace(choice, authorized_scope=other_scope)
    detached = replace(
        choice.typed_outcomes[0], use_key=LosslessIntegerKey((999, 0, 1)))
    with pytest.raises(GenerationChoiceContractError, match="exact Use"):
        replace(choice, typed_outcomes=(detached,))


def test_generic_candidate_preflight_accepts_choice_without_any_write():
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        graph = CandidateProjectionGraph(
            ctx.graph_ontology, _projection_protocol())
        learning = _candidate_runtime(graph)
        mapper = GenerationChoiceCandidateMapper(_candidate_protocol())
        choice = _choice()
        before_runtime = learning.state_key()
        before_backend = backend.snapshot()

        hypothesis = mapper.preflight_load(choice, learning)

        assert hypothesis.object_identity().object_kind == OBJECT_HYPOTHESIS
        assert learning.state_key() == before_runtime
        assert backend.snapshot() == before_backend
        definition = mapper.definition(choice)
        mapped = HypothesisKey.from_stable_key(
            definition.candidate.components)
        assert mapped.candidate_key == choice.candidate.stable_key()
        assert definition.candidate == mapper.candidate_identity(choice)
        assert definition.forming_sources == choice.forming_sources
        assert len(definition.bindings) == 6
    finally:
        backend.close()


def test_candidate_protocol_rejects_duplicate_predicate_and_missing_layer():
    protocol = _candidate_protocol()
    with pytest.raises(GenerationChoiceContractError, match="distinct"):
        replace(
            protocol,
            choice_kind_predicate=protocol.candidate_kind_predicate,
        )
    with pytest.raises(GenerationChoiceContractError, match="five layers"):
        replace(protocol, choice_kind_objects=protocol.choice_kind_objects[:-1])


def test_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    digest = hashlib.sha256(b"baseline").hexdigest()
    manifest = build_gg01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v23.json"),
        prerequisite_manifest_sha256=digest,
    )
    output = tmp_path / "gg01.json"
    write_gg01_contract_manifest(manifest, output)
    assert read_gg01_contract_manifest(output) == manifest
    write_gg01_contract_manifest(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(GenerationChoiceContractError, match="different content"):
        write_gg01_contract_manifest(manifest, output)
    with pytest.raises(GenerationChoiceContractError, match="runtime"):
        replace(manifest, runtime_status="CONNECTED")
    with pytest.raises(GenerationChoiceContractError, match="execution"):
        replace(
            manifest,
            execution_state=type(manifest.execution_state).from_value({
                **manifest.execution_state.to_value(), "teacher_calls": 1}),
        )


def test_lossless_integer_key_keeps_zero_bearing_identity_without_digest():
    key = LosslessIntegerKey((1, 0, 2, -3))
    assert key.stable_key() == (1, 0, 2, -3)
    assert LosslessIntegerKey.from_value(key.to_list(), where="exact") == key
    with pytest.raises(GenerationChoiceContractError, match="strict integers"):
        LosslessIntegerKey((1, False))
    with pytest.raises(GenerationChoiceContractError, match="non-empty"):
        LosslessIntegerKey(())


def test_repository_v2_supersedes_v1_and_freezes_zero_bearing_exact_keys():
    v1_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    manifest = read_gg01_contract_manifest(MANIFEST_V2_PATH)
    assert manifest == build_gg01_contract_manifest_v2(
        superseded_manifest_sha256=v1_hash)
    assert manifest.prerequisite_manifest_relative_path.endswith(
        "gg01_generation_choice_contract_v1.json")
    assert "EXACT_KEYS_PRESERVE_ZERO_BEARING_CORE_MEMORY_IDENTITIES" in (
        manifest.invariant_keys)


def test_manifest_rejects_incomplete_fields_and_direction_overclaim():
    digest = hashlib.sha256(b"baseline").hexdigest()
    manifest = build_gg01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v23.json"),
        prerequisite_manifest_sha256=digest,
    )
    with pytest.raises(GenerationChoiceContractError, match="choice kinds"):
        replace(manifest, choice_kind_keys=manifest.choice_kind_keys[:-1])
    directions = manifest.lc13_directional_consumption.to_value()
    directions["REASONING"]["fact_state"] = "CONTRACT_FROZEN"
    with pytest.raises(GenerationChoiceContractError, match="reasoning"):
        replace(
            manifest,
            lc13_directional_consumption=(
                type(manifest.lc13_directional_consumption).from_value(
                    directions)),
        )
    value = manifest.to_dict()
    value["answer_template"] = [1]
    with pytest.raises(GenerationChoiceContractError, match="not exact"):
        GG01ContractManifest.from_dict(value)


def test_repository_manifest_binds_v23_and_stays_not_connected():
    baseline_hash = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    manifest = read_gg01_contract_manifest(MANIFEST_PATH)
    expected = build_gg01_contract_manifest(
        prerequisite_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v23.json"),
        prerequisite_manifest_sha256=baseline_hash,
    )
    assert manifest == expected
    assert manifest.runtime_status == "NOT_CONNECTED"
    assert manifest.results_observed == 0
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
