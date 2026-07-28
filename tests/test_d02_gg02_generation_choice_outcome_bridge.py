"""GG-02 分层 exact Use/outcome 归因和零写 assessment 输入测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_USE,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseContext,
    RelationUseDefinition,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_choice_outcome_bridge import (
    GG02BridgeManifest,
    GenerationChoiceEpisodeAttribution,
    GenerationChoiceOutcomeBridge,
    GenerationChoiceOutcomeBridgeError,
    GenerationChoiceUseAttribution,
    GenerationVerifierLayerRoute,
    build_assessment_inputs,
    build_gg02_bridge_manifest,
    read_gg02_bridge_manifest,
    write_gg02_bridge_manifest,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationReport,
    VerificationResult,
)
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_MEMORY,
    SpaceIdentity,
)
from tests.test_d02_gg01_generation_choice_contract import _choice, _context


REPO_ROOT = Path(__file__).resolve().parents[1]
GG01_V2_PATH = REPO_ROOT / "data/ph2/manifests/gg01_generation_choice_contract_v2.json"
BASELINE_V25_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v25.json"
MANIFEST_PATH = REPO_ROOT / "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json"


def _core_use(context, ordinal: int) -> RelationUseDefinition:
    source = context.scope.source
    assert source is not None
    hypothesis = HypothesisKey(
        (81000, ordinal, 1),
        (81000, ordinal, 2),
        (81000, ordinal, 3),
        context.scope,
        source,
    )
    return RelationUseDefinition(
        (81000, ordinal, 4),
        RelationUseContext(
            source,
            context.scope,
            concept_identity((81000, ordinal, 5)),
            minimal_instruction_identity((81000, ordinal, 6)),
        ),
        context.content_obligations[0].proposition,
        hypothesis,
        ((81000, ordinal, 7, 0),),
        (81000, ordinal, 8),
        False,
    )


def _memory_use(context, ordinal: int) -> MemoryObjectRef:
    return MemoryObjectRef(
        SpaceIdentity(SPACE_TYPE_MEMORY, 0, ordinal),
        context.scope.owner,
        context.scope.versions,
        MEMORY_OBJECT_USE,
        (82000, ordinal, 0),
    )


def _episode():
    context = _context(41)
    source = context.scope.source
    assert source is not None
    query_key = LosslessIntegerKey(context.scope.stable_key())
    generation_key = LosslessIntegerKey((83000, 41, 0))
    actual_uses = tuple(
        _core_use(context, index + 1) if index < 3
        else _memory_use(context, index + 1)
        for index in range(len(CHOICE_KINDS)))
    attributions = []
    for index, (choice_kind, actual_use) in enumerate(
            zip(CHOICE_KINDS, actual_uses, strict=True), start=1):
        use_kind = "CORE_USE" if isinstance(
            actual_use, RelationUseDefinition) else "MEMORY_USE"
        use = GenerationChoiceUseRef(
            use_kind,
            LosslessIntegerKey(actual_use.stable_key()),
            LosslessIntegerKey((83000, index, 0)),
            context.scope,
        )
        choice = replace(
            _choice(choice_kind, seed=41),
            candidate=ObjectIdentity(OBJECT_HYPOTHESIS, (83100, index)),
            competition_key=(83101, index),
            exact_uses=(use,),
        )
        attributions.append(GenerationChoiceUseAttribution(
            choice,
            use,
            query_key,
            generation_key,
            (LosslessIntegerKey((83200, index, 0)),),
            source,
            context.scope,
        ))
    episode = GenerationChoiceEpisodeAttribution(
        LosslessIntegerKey(context.context.stable_key()),
        query_key,
        generation_key,
        source,
        context.scope,
        tuple(attributions),
    )
    return episode, actual_uses


def _routes() -> tuple[GenerationVerifierLayerRoute, ...]:
    return tuple(GenerationVerifierLayerRoute(
        ProtocolKey((index, 0)),
        ProtocolKey((index, 1)),
        (choice_kind,),
    ) for index, choice_kind in enumerate(CHOICE_KINDS, start=1))


def _report(episode, routes=None) -> VerificationReport:
    routes = _routes() if routes is None else routes
    by_kind = {item.choice.choice_kind: item for item in episode.choices}
    results = []
    for index, route in enumerate(routes, start=1):
        target = by_kind[CHOICE_KINDS[index - 1]]
        results.append(VerificationResult(
            route.dimension,
            route.verifier,
            APPLICABILITY_APPLICABLE,
            VERDICT_SUPPORT if index % 2 else VERDICT_REFUTE,
            tuple(item.components for item in target.verification_claim_keys),
            detail=() if index == 1 else (83300, index, 0),
            source=episode.source,
            scope=episode.scope,
        ))
    return VerificationReport(True, tuple(results))


def test_five_layers_bind_lossless_actual_core_and_memory_use_keys():
    episode, actual_uses = _episode()
    report = GenerationChoiceOutcomeBridge(_routes()).compile(
        episode, _report(episode))

    assert tuple(item.choice.choice_kind for item in episode.choices) == CHOICE_KINDS
    for attribution, actual in zip(episode.choices, actual_uses, strict=True):
        assert attribution.use.use_key.components == actual.stable_key()
        assert 0 in attribution.use.use_key.components
        assert attribution.query_key == episode.query_key
        assert attribution.source == episode.source
        assert attribution.scope == episode.scope
    assert len(report.outcomes) == len(CHOICE_KINDS)
    assert {item.choice_kind for item in report.outcomes} == set(CHOICE_KINDS)
    assert report.outcomes[0].detail.components == (0,)
    assert report.host_learning_write_count == 0
    assert report.teacher_call_count == 0
    assert report.assessment_consumer_status == "REQUIRED_NOT_CONNECTED"


def test_episode_rejects_foreign_use_duplicate_claim_and_query_drift():
    episode, _ = _episode()
    first, second, *rest = episode.choices
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="exact Use"):
        replace(first, use=second.use)
    duplicate = replace(
        second, verification_claim_keys=first.verification_claim_keys)
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="重复归因"):
        replace(episode, choices=(first, duplicate, *rest))
    drifted = replace(first, query_key=LosslessIntegerKey((999, 0)))
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="同一 query"):
        replace(episode, choices=(drifted, second, *rest))


def test_bridge_rejects_undeclared_foreign_missing_and_source_drift_claims():
    episode, _ = _episode()
    routes = list(_routes())
    routes[0] = replace(routes[0], choice_kinds=(CHOICE_KINDS[1],))
    routes[1] = replace(routes[1], choice_kinds=(CHOICE_KINDS[0],))
    swapped = tuple(routes)
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="越过声明"):
        GenerationChoiceOutcomeBridge(swapped).compile(
            episode, _report(episode, swapped))

    report = _report(episode)
    foreign = replace(report.results[0], claim_keys=((999, 0),))
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="不属于"):
        GenerationChoiceOutcomeBridge(_routes()).compile(
            episode, replace(report, results=(foreign, *report.results[1:])))
    missing = replace(report.results[0], claim_keys=())
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="没有 exact"):
        GenerationChoiceOutcomeBridge(_routes()).compile(
            episode, replace(report, results=(missing, *report.results[1:])))
    drifted = replace(report.results[0], source=episode.choices[0].choice.forming_sources[0])
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="同一 query"):
        GenerationChoiceOutcomeBridge(_routes()).compile(
            episode, replace(report, results=(drifted, *report.results[1:])))


def test_sentence_wide_reward_or_punishment_route_is_rejected():
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="整句五层广播"):
        GenerationVerifierLayerRoute(
            ProtocolKey((1, 0)), ProtocolKey((1, 1)), CHOICE_KINDS)


def test_read_only_and_effect_boundaries_fail_closed():
    episode, _ = _episode()
    report = _report(episode)
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="只读"):
        GenerationChoiceOutcomeBridge(_routes()).compile(
            episode, replace(report, read_only=False))
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="未逐 route"):
        GenerationChoiceOutcomeBridge(_routes()).compile(
            episode, replace(report, results=report.results[:-1]))


def test_disabled_layer_changes_only_its_assessment_and_executes_no_update():
    episode, _ = _episode()
    layered = GenerationChoiceOutcomeBridge(_routes()).compile(
        episode, _report(episode))
    enabled = build_assessment_inputs(layered)
    disabled = build_assessment_inputs(
        layered, disabled_choice_kinds=(CHOICE_KINDS[2],))

    assert enabled.assessment_updates_executed == 0
    assert disabled.assessment_updates_executed == 0
    assert enabled.host_learning_write_count == 0
    assert disabled.host_learning_write_count == 0
    for before, after in zip(enabled.inputs, disabled.inputs, strict=True):
        if before.choice_kind == CHOICE_KINDS[2]:
            assert before.assessment_state == "READY"
            assert after.assessment_state == "NE_LAYER_DISABLED"
        else:
            assert before == after


def test_outcome_ledger_never_implies_candidate_assessment_update():
    episode, _ = _episode()
    layered = GenerationChoiceOutcomeBridge(_routes()).compile(
        episode, _report(episode))
    assessment = build_assessment_inputs(layered)
    assert layered.assessment_consumer_status == "REQUIRED_NOT_CONNECTED"
    assert all(item.assessment_state == "READY" for item in assessment.inputs)
    assert assessment.assessment_updates_executed == 0
    assert assessment.host_learning_write_count == 0


def test_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    manifest = build_gg02_bridge_manifest(
        gg01_sha256="1" * 64,
        baseline_sha256="2" * 64,
    )
    output = tmp_path / "gg02.json"
    write_gg02_bridge_manifest(manifest, output)
    assert read_gg02_bridge_manifest(output) == manifest
    write_gg02_bridge_manifest(manifest, output)
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="内容不同"):
        write_gg02_bridge_manifest(manifest, output)
    with pytest.raises(GenerationChoiceOutcomeBridgeError, match="assessment"):
        replace(manifest, assessment_consumer_status="CONNECTED")
    assert all(value == 0 for value in manifest.execution_state.to_value().values())


def test_repository_manifest_binds_gg01_v2_and_baseline_v25():
    expected = build_gg02_bridge_manifest(
        gg01_sha256=hashlib.sha256(GG01_V2_PATH.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(BASELINE_V25_PATH.read_bytes()).hexdigest(),
    )
    manifest = read_gg02_bridge_manifest(MANIFEST_PATH)
    assert isinstance(manifest, GG02BridgeManifest)
    assert manifest == expected
    assert manifest.runtime_status == "NOT_CONNECTED"
    assert manifest.results_observed == 0
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
