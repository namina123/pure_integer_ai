"""W08-06 P3-Ia, bounded open generation, and LC-16 integration."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_SESSION,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.question_answer import QuestionQuery
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseResult,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.authorized_generation_delivery import (
    AuthorizedGenerationClaim,
)
from pure_integer_ai.experiments.free_text_hierarchy_runtime import (
    MechanicalTextHierarchyFormer,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    AclFirstExactRecallReader,
    FreeTextRecallRuntime,
    LearnedEvidenceCenterFormer,
    LearnedSurfaceFeatureMatcher,
    RecallIndexEntry,
    TypedRecallPayload,
    encode_surface_feature_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    RecallBudget,
    SourceDocument,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_discourse import (
    W08DiscourseClaim,
    W08DiscourseRequest,
    W08DiscourseScene,
    W08GenerationFormCandidate,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_lc16 import (
    W08LC16Error,
    W08LC16Qualifier,
    W08_LC16_SCOPE_KEY,
    compile_w08_lc16_projection_inventory,
)
from pure_integer_ai.experiments.ph2_w08_long_context_adapters import (
    materialize_w08_long_input,
)
from pure_integer_ai.experiments.ph2_w08_open_generation import (
    W08OpenGenerationRuntime,
    W08OpenGenerationSegment,
)
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08OpenClaimBinding,
    W08OpenGenerationCandidate,
    W08OpenGenerationRequest,
    assess_w08_open_generation_ablation,
)
from pure_integer_ai.experiments.ph2_w08_p3ia import (
    W08P3IaExecution,
    W08P3IaFacade,
    W08P3IaGenerationOwner,
    W08P3IaOwners,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_contract import (
    W08P3IaRequest,
    W08_P3IA_COMPONENT_KEYS,
    assess_w08_p3ia_stage_ablation,
    assess_w08_p3ia_supporting_ablation,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_training import (
    W08_P3IA_PARAPHRASE_REASON_KEY,
    compile_w08_p3ia_training,
)
from pure_integer_ai.experiments.ph2_w08_stage6 import W08Stage6Facade
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import _fixture as question_fixture
from tests.test_w08_03_discourse import _facade as discourse_facade
from tests.test_w08_04_recompute import _local_case
from tests.test_d02_md02_situation_state_adapter import _close as close_situation
from tests.test_w08_05_long_context import (
    _NeverPrefetch,
    _current,
    _md03,
    _storage,
)


ROOT = Path(__file__).resolve().parents[1]
OWNER = OwnerScope(80806, 1, 1, VISIBILITY_SESSION)


def _bound(
    source,
    *,
    ordinal: int,
    start: int,
    end: int,
    with_role: bool = True,
) -> BoundProposition:
    role = role_identity((80806, 10, ordinal))
    filler = concept_identity((80806, 11, ordinal))
    return BoundProposition(
        proposition_identity(source, (80806, 12, ordinal)),
        minimal_instruction_identity((80806, 13, ordinal)),
        concept_identity((80806, 14, ordinal)),
        structure_concept_identity((80806, 15, ordinal)),
        occurrence_identity(source, start=start, end=end, ordinal=ordinal),
        context_scope_identity(source, (80806, 16, ordinal)),
        (),
        (BoundRoleBinding(role, filler),) if with_role else (),
        (),
    )


def _qa(target, source, text):
    mapper, postchecker, _parser, _structure, _source = _postcheck_owners()
    fixture = question_fixture(
        EVIDENCE_SUPPORT,
        world=(source, document_scope(source), target),
        answer_text=text,
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    return fixture, fixture.runtime.run(fixture.request)


def _authorization(run, receipt_key):
    assert run.planning_request is not None
    assert run.selection is not None
    selected = set(run.selection.selected_candidate_keys)
    candidate = next(
        item for item in run.planning_request.candidates if item.stable_key() in selected
    )
    return AuthorizedGenerationClaim(
        candidate.stable_key(),
        candidate.proposition.stable_key(),
        candidate.source,
        candidate.scope,
        candidate.citation_sources,
        receipt_key,
    )


@pytest.fixture(scope="module")
def stage6_fixture():
    contract = open_w08_frozen_contract(ROOT)
    payload = W08PayloadFirewall.open(
        ROOT,
        contract,
        make_w08_request(contract),
    ).read_training_payload()
    training = compile_w08_p3ia_training(payload)
    case = training.cases[0]
    material = materialize_w08_long_input(
        training.long_context,
        owner=OWNER,
        chunk_width=19,
    )
    document = SourceDocument(
        material.source,
        StableRecordKey((80806, 20, 1)),
        StableRecordKey((80806, 20, 2)),
        StableRecordKey((80806, 20, 3)),
        training.long_context.document_text,
        hashlib.sha256(training.long_context.document_text.encode("utf-8")).hexdigest(),
    )
    graph_backend = DictBackend()
    ctx = make_train_context(graph_backend)
    current = _current(ctx, material.source)

    local_fixture, local_request, local_runtime, _old, _replacement = _local_case()
    revision = local_runtime.execute(local_request)
    dependency = revision.free_text.invalidated_keys[0]

    target = _bound(
        material.source,
        ordinal=1,
        start=case.citation_start,
        end=case.citation_end,
        with_role=False,
    )
    qa_one, qa_run_one = _qa(target, material.source, case.exact_surface)
    recall_payload = TypedRecallPayload(
        target,
        (qa_one.evidence[0],),
        case.citation_start,
        case.citation_end,
        (dependency,),
    )
    record_key = (80806, 30, 1)
    store, repository = _storage(
        recall_payload,
        material.source,
        document_scope(material.source),
        record_key=record_key,
    )
    policy = QueryHotSetPolicy(
        SegmentBudget(4, 1_000_000),
        SegmentBudget(4, 1_000_000),
        _NeverPrefetch(),
        8,
    )
    reader = AclFirstExactRecallReader(
        store,
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        policy,
    )
    hypothesis = HypothesisKey(
        (80806, 31, 1),
        case.paraphrase_quote_key.components,
        (80806, 31, 2),
        document_scope(material.source),
        material.source,
    )
    paraphrase_evidence = EvidenceRecord(
        80806001,
        hypothesis,
        EVIDENCE_SUPPORT,
        W08_P3IA_PARAPHRASE_REASON_KEY,
        material.source,
        1,
        encode_surface_feature_payload(case.paraphrase_surface, case.feature_key),
    )
    entry = RecallIndexEntry(
        record_key,
        material.source,
        document_scope(material.source),
        (case.feature_key,),
        (dependency,),
    )
    recall = FreeTextRecallRuntime(
        LearnedEvidenceCenterFormer(
            LearnedSurfaceFeatureMatcher(W08_P3IA_PARAPHRASE_REASON_KEY),
            _md03(),
        ),
        reader,
    )
    route = minimal_instruction_identity((80806, 32, 1))
    question = QuestionQuery(qa_one.request, route, (80806, 32, 2))
    execution = W08P3IaExecution(
        training,
        case,
        document,
        case.paraphrase_surface,
        (paraphrase_evidence,),
        current,
        (entry,),
        MemoryAccessContext(80806, 1, 1),
        RecallBudget(2, 2, 1_000_000, 1),
        question,
        minimal_instruction_identity((80806, 32, 3)),
        revision,
    )
    p3ia_facade = W08P3IaFacade(
        W08P3IaOwners(
            MechanicalTextHierarchyFormer(),
            recall,
            W08P3IaGenerationOwner(lambda _recall, _qa: qa_run_one),
        )
    )
    p3ia_request = W08P3IaRequest(
        (80806, 33, 1),
        case.case_key,
        W08_P3IA_COMPONENT_KEYS,
        (80806, 33, 2),
        1,
    )
    p3ia = p3ia_facade.execute(p3ia_request, execution)

    target_open_one = _bound(
        material.source,
        ordinal=2,
        start=case.citation_start,
        end=case.citation_end,
    )
    qa_open_one, qa_run_open_one = _qa(
        target_open_one,
        material.source,
        case.exact_surface,
    )
    target_two = _bound(
        material.source,
        ordinal=3,
        start=case.paraphrase_start,
        end=case.paraphrase_end,
    )
    qa_two, qa_run_two = _qa(
        target_two,
        material.source,
        case.paraphrase_surface,
    )
    assert p3ia.trace is not None
    receipt_key = p3ia.trace.authorization_receipt_key
    authorization_one = _authorization(qa_run_open_one, receipt_key)
    authorization_two = _authorization(qa_run_two, receipt_key)

    source_key = material.source.stable_key()
    scope_key = document_scope(material.source).stable_key()
    claim_keys = ((80806, 40, 1), (80806, 40, 2))
    holders = ((80806, 41, 1), (80806, 41, 2))
    scene = W08DiscourseScene(
        source_key,
        scope_key,
        ((80806, 42, 1), (80806, 42, 2)),
        ((80806, 43, 1), (80806, 43, 2)),
        ((80806, 44, 1),),
        holders,
        ((80806, 45, 1),),
        ((80806, 46, 1),),
        ((80806, 47, 1),),
        ((80806, 48, 1),),
        ((80806, 49, 1),),
        ((80806, 49, 2),),
        ((80806, 50, 1),),
    )
    discourse_claims = (
        W08DiscourseClaim(
            claim_keys[0],
            target_open_one.stable_key(),
            "DIRECT",
            "SUPPORTED",
            "ACTIVE",
            1,
            holder_key=holders[0],
            evidence_keys=((80806, 51, 1),),
        ),
        W08DiscourseClaim(
            claim_keys[1],
            target_two.stable_key(),
            "DIRECT",
            "SUPPORTED",
            "ACTIVE",
            1,
            holder_key=holders[1],
            evidence_keys=((80806, 51, 2),),
        ),
    )
    discourse_request = W08DiscourseRequest(
        (80806, 52, 1),
        scene,
        discourse_claims,
        "SOURCE_CONFLICT",
        2,
        ((80806, 52, 2),),
        1,
        ((80806, 53, 1), (80806, 53, 2)),
        (80806, 53, 1),
        1,
        (
            W08GenerationFormCandidate((80806, 54, 1), "ELLIPSIS", (80806, 53, 1)),
            W08GenerationFormCandidate((80806, 54, 2), "PRONOUN", (80806, 53, 1)),
        ),
    )
    discourse, _discourse_context = discourse_facade()
    discourse_audit = discourse.execute(discourse_request)
    claim_bindings = (
        W08OpenClaimBinding(
            claim_keys[0],
            target_open_one.stable_key(),
            source_key,
            scope_key,
            tuple(sorted(item.role.stable_key() for item in target_open_one.bindings)),
            "SUPPORTED",
            holders[0],
            receipt_key,
        ),
        W08OpenClaimBinding(
            claim_keys[1],
            target_two.stable_key(),
            source_key,
            scope_key,
            tuple(sorted(item.role.stable_key() for item in target_two.bindings)),
            "SUPPORTED",
            holders[1],
            receipt_key,
        ),
    )
    combination = digest_value(
        {"claims": [list(item.proposition_key) for item in claim_bindings]}
    )
    candidate_one = W08OpenGenerationCandidate(
        (80806, 60, 1),
        combination,
        claim_keys,
        digest_value({"order": [list(item) for item in claim_keys]}),
        "ELLIPSIS",
        digest_value(
            {
                "segments": [
                    list(qa_run_open_one.generation.representations[0].stable_key()),
                    list(qa_run_two.generation.representations[0].stable_key()),
                ],
                "reference": "ELLIPSIS",
            }
        ),
        (80806, 60, 2),
        (80806, 60, 3),
    )
    candidate_two = W08OpenGenerationCandidate(
        (80806, 61, 1),
        combination,
        tuple(reversed(claim_keys)),
        digest_value({"order": [list(item) for item in reversed(claim_keys)]}),
        "PRONOUN",
        digest_value(
            {
                "segments": [
                    list(qa_run_two.generation.representations[0].stable_key()),
                    list(qa_run_open_one.generation.representations[0].stable_key()),
                ],
                "reference": "PRONOUN",
            }
        ),
        (80806, 61, 2),
        (80806, 61, 3),
    )
    candidates = tuple(sorted((candidate_one, candidate_two)))
    open_request = W08OpenGenerationRequest(
        (80806, 62, 1),
        discourse_request,
        discourse_audit,
        claim_bindings,
        candidates,
        (),
        (),
        (),
        (80806, 62, 2),
    )
    segments = (
        W08OpenGenerationSegment(claim_keys[0], qa_run_open_one, authorization_one),
        W08OpenGenerationSegment(claim_keys[1], qa_run_two, authorization_two),
    )
    runtime = W08OpenGenerationRuntime(
        lambda request: request.candidates[0].candidate_key,
        lambda _request, _candidate: segments,
    )
    open_generation = runtime.execute(open_request)
    template_request = replace(
        open_request,
        known_complete_template_keys=(candidate_one.complete_template_key,),
    )
    template_ablation_receipt = runtime.execute(template_request)
    open_ablation = assess_w08_open_generation_ablation(
        open_generation,
        template_ablation_receipt,
    )

    inventory = compile_w08_lc16_projection_inventory(
        ROOT,
        semantic_engine_key=(80806, 70, 1),
        discourse_projection_key=discourse_request.scene.stable_key(),
        logic_projection_key=digest_value(
            {"claims": [list(item.proposition_key) for item in claim_bindings]}
        ),
        generation_projection_key=open_generation.canonical_key(),
    )
    lc16 = W08LC16Qualifier().qualify(
        inventory,
        p3ia=p3ia,
        generation=open_generation,
    )
    yield {
        "training": training,
        "case": case,
        "p3ia_facade": p3ia_facade,
        "p3ia_request": p3ia_request,
        "p3ia_execution": execution,
        "p3ia": p3ia,
        "open_request": open_request,
        "open_runtime": runtime,
        "open": open_generation,
        "template_receipt": template_ablation_receipt,
        "open_ablation": open_ablation,
        "segments": segments,
        "inventory": inventory,
        "lc16": lc16,
    }

    qa_two.close()
    qa_open_one.close()
    qa_one.close()
    close_situation(local_fixture)
    ctx.work_memory.end_query()
    ctx.work_memory.end_episode()
    ctx.work_memory.end_document()
    ctx.work_memory.end_session()
    graph_backend.close()
    del repository, store


def test_w08_p3ia_training_is_mechanical_and_future_free(stage6_fixture):
    training = stage6_fixture["training"]
    case = stage6_fixture["case"]
    assert training.audit.observation_count == 63
    assert training.audit.evidence_binding_count == 63
    assert training.audit.paraphrase_link_count == 1
    assert training.audit.case_count == 1
    assert training.audit.authored_answer_read_count == 0
    assert training.audit.future_pack_read_count == 0
    assert case.exact_surface_sha256 != case.paraphrase_surface_sha256
    assert case.citation_end - case.citation_start == len(case.exact_surface)


def test_w08_p3ia_runs_full_owner_chain_with_exact_cold_citation(stage6_fixture):
    receipt = stage6_fixture["p3ia"]
    assert receipt.state == "RESOLVED"
    assert receipt.trace is not None
    assert receipt.trace.acl_checked_before_payload == 1
    assert receipt.trace.citation_exact == 1
    assert receipt.trace.full_document_reparse_count == 0
    assert receipt.resources.payload_gets == 1
    assert receipt.resources.recalled_records == 1
    assert tuple(item.consumer_key for item in receipt.uses) == (
        "UNDERSTANDING",
        "REASONING",
        "GENERATION",
    )


@pytest.mark.parametrize("component", W08_P3IA_COMPONENT_KEYS)
def test_w08_p3ia_supporting_ablations_fail_closed(stage6_fixture, component):
    request = stage6_fixture["p3ia_request"]
    ablated = replace(
        request,
        enabled_components=tuple(
            item for item in W08_P3IA_COMPONENT_KEYS if item != component
        ),
    )
    receipt = stage6_fixture["p3ia_facade"].execute(
        ablated,
        stage6_fixture["p3ia_execution"],
    )
    report = assess_w08_p3ia_supporting_ablation(
        stage6_fixture["p3ia"],
        receipt,
        component_key=component,
    )
    assert report.ablated_state != "RESOLVED"
    assert receipt.trace is None
    assert receipt.uses == ()


def test_w08_open_generation_has_multiple_surfaces_five_ledgers_and_atomic_publish(
    stage6_fixture,
):
    request = stage6_fixture["open_request"]
    receipt = stage6_fixture["open"]
    assert len(request.candidates) == 2
    assert len({item.surface_family_key for item in request.candidates}) == 2
    assert receipt.state == "RESOLVED"
    assert len(receipt.publication_units) == 2
    assert tuple(item.layer_key for item in receipt.layers) == (
        "CONTENT",
        "STRUCTURE_LOGIC",
        "DISCOURSE_REFERENCE",
        "SURFACE_MORPHOLOGY",
        "TASK_COMMUNICATIVE",
    )
    assert all(item.state == "PASS" for item in receipt.layers)
    assert len({item.use_key for item in receipt.uses}) == len(receipt.uses)


def test_closed_renderer_template_replay_ablation_only_breaks_surface_layer(
    stage6_fixture,
):
    receipt = stage6_fixture["template_receipt"]
    report = stage6_fixture["open_ablation"]
    assert receipt.state == "GROUNDING_BLOCKED"
    assert receipt.publication_units == ()
    assert report.affected_layers == ("SURFACE_MORPHOLOGY",)


def test_unauthorized_generation_has_zero_publication(stage6_fixture):
    segments = stage6_fixture["segments"]
    bad_claim = replace(
        segments[0].authorization_claim,
        candidate_key=(80806, 999, 1),
    )
    bad_segments = (
        replace(segments[0], authorization_claim=bad_claim),
        segments[1],
    )
    runtime = W08OpenGenerationRuntime(
        lambda request: request.candidates[0].candidate_key,
        lambda _request, _candidate: bad_segments,
    )
    receipt = runtime.execute(stage6_fixture["open_request"])
    assert receipt.state == "ACCESS_BLOCKED"
    assert receipt.publication_units == ()
    assert any(item.state == "FAIL" for item in receipt.layers)
    assert any(item.state == "PASS" for item in receipt.layers)
    assert len({item.outcome_key for item in receipt.uses}) == len(receipt.uses)
    content_uses = tuple(item for item in receipt.uses if item.layer_key == "CONTENT")
    assert tuple(item.outcome_state for item in content_uses) == ("FAIL", "PASS")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_key", (80806, 999, 1)),
        ("scope_key", (80806, 999, 2)),
        ("holder_key", (80806, 999, 3)),
    ),
)
def test_source_scope_or_holder_drift_has_zero_publication(
    stage6_fixture,
    field,
    value,
):
    request = stage6_fixture["open_request"]
    bad_claims = (
        replace(request.claims[0], **{field: value}),
        request.claims[1],
    )
    receipt = stage6_fixture["open_runtime"].execute(
        replace(request, claims=bad_claims)
    )
    assert receipt.state == "GROUNDING_BLOCKED"
    assert receipt.publication_units == ()


def test_reference_postcheck_failure_stops_before_generation(stage6_fixture):
    request = stage6_fixture["open_request"]
    facade, context = discourse_facade()
    context.generation_blocked = True
    blocked_discourse = facade.execute(request.discourse_request)
    assert blocked_discourse.stop_state == "GROUNDING_BLOCKED"
    calls = []

    def forbidden_generator(_request, _candidate):
        calls.append("generation")
        return stage6_fixture["segments"]

    runtime = W08OpenGenerationRuntime(
        lambda current: current.candidates[0].candidate_key,
        forbidden_generator,
    )
    receipt = runtime.execute(replace(request, discourse_audit=blocked_discourse))
    assert calls == []
    assert receipt.state == "ACCESS_BLOCKED"
    assert receipt.publication_units == ()


def test_g04_failure_has_zero_publication(stage6_fixture):
    segments = stage6_fixture["segments"]
    run = segments[0].run
    assert run.postcheck is not None
    failed_parse = GenerationSurfaceParseResult(
        run.postcheck.parsed.reason,
        (80806, 999, 4),
        None,
    )
    failed_postcheck = replace(run.postcheck, parsed=failed_parse)
    failed_run = replace(run, postcheck=failed_postcheck)
    failed_segments = (replace(segments[0], run=failed_run), segments[1])
    runtime = W08OpenGenerationRuntime(
        lambda request: request.candidates[0].candidate_key,
        lambda _request, _candidate: failed_segments,
    )
    receipt = runtime.execute(stage6_fixture["open_request"])
    assert receipt.state == "ACCESS_BLOCKED"
    assert receipt.publication_units == ()
    assert dict((item.layer_key, item.state) for item in receipt.layers)[
        "TASK_COMMUNICATIVE"
    ] == "FAIL"


def test_lc16_uses_one_engine_for_nine_carriers_and_27_direction_cells(
    stage6_fixture,
):
    inventory = stage6_fixture["inventory"]
    receipt = stage6_fixture["lc16"]
    assert len(inventory.carriers) == 9
    assert len({item.semantic_engine_key for item in inventory.carriers}) == 1
    assert inventory.sample_payload_read_count == 0
    assert receipt.state == "PASS"
    assert len(receipt.cells) == 27
    assert {item.scope_key for item in receipt.cells} == {W08_LC16_SCOPE_KEY}
    assert all(item.state == "PASS" for item in receipt.cells)


def test_lc16_schema_ne_and_language_unknown_are_separate():
    qualifier = W08LC16Qualifier()
    assert qualifier.classify_boundary("UNKNOWN") == "LANGUAGE_UNKNOWN"
    assert qualifier.classify_boundary("SCHEMA_REQUIRED") == "SCHEMA_REQUIRED"
    assert qualifier.classify_boundary("UNREPRESENTABLE") == "UNREPRESENTABLE"
    assert qualifier.classify_boundary("NE") == "NE"
    with pytest.raises(W08LC16Error):
        qualifier.classify_boundary("ACCESS_BLOCKED")


def test_stage6_public_bounded_aggregate_keeps_formal_state_zero(stage6_fixture):
    full = {key: "PASS" for key in W08_DIMENSION_KEYS}
    ablated = dict(full)
    ablated["W-08-P3IA"] = "FAIL"
    supporting = tuple(
        assess_w08_p3ia_supporting_ablation(
            stage6_fixture["p3ia"],
            stage6_fixture["p3ia_facade"].execute(
                replace(
                    stage6_fixture["p3ia_request"],
                    enabled_components=tuple(
                        item for item in W08_P3IA_COMPONENT_KEYS if item != component
                    ),
                ),
                stage6_fixture["p3ia_execution"],
            ),
            component_key=component,
        )
        for component in W08_P3IA_COMPONENT_KEYS
    )
    stage = W08Stage6Facade().close(
        p3ia=stage6_fixture["p3ia"],
        p3ia_supporting_ablations=supporting,
        p3ia_stage_ablation=assess_w08_p3ia_stage_ablation(
            full_dimension_outcomes=full,
            ablated_dimension_outcomes=ablated,
        ),
        open_generation=stage6_fixture["open"],
        open_generation_ablation=stage6_fixture["open_ablation"],
        lc16=stage6_fixture["lc16"],
        dimension_outcomes=full,
    )
    assert stage.state == "PASS"
    assert stage.W08_STARTED == 0
    assert stage.formal_w08_training_runs == 0
    assert stage.OPEN_GENERATION == "NE_NOT_YET_EVALUABLE"
    assert stage.W09_STARTED == 0
