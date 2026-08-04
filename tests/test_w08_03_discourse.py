"""W08-03 篇章 owner 编排、投影隔离、修正与 U/R/G 专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_discourse import (
    W08AgendaReceipt,
    W08CenterReceipt,
    W08DiscourseClaim,
    W08DiscourseError,
    W08DiscourseFacade,
    W08DiscourseOwners,
    W08DiscourseRequest,
    W08DiscourseScene,
    W08DiscourseUse,
    W08EventReceipt,
    W08GenerationFormCandidate,
    W08GenerationReceipt,
    W08LifecycleReceipt,
    W08ProjectionReceipt,
    W08ReferenceReceipt,
    W08_DISCOURSE_OWNER_KEYS,
    assess_w08_discourse_ablation,
)
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_discourse_training import (
    audit_w08_discourse_training,
)
from pure_integer_ai.experiments.ph2_w08_discourse_adapters import (
    W08A01ReferenceOwner,
    W08AgendaOwner,
    W08ConsumerOwner,
    W08CurrentProjectionOwner,
    W08GenerationOwner,
    W08MD03CenterOwner,
    W08SituationEventOwner,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload


ROOT = Path(__file__).resolve().parents[1]


def _k(value: int) -> tuple[int, ...]:
    return (value,)


def _scene(*, close_qud: bool = False) -> W08DiscourseScene:
    return W08DiscourseScene(
        _k(1),
        _k(2),
        (_k(10), _k(11)),
        (_k(20), _k(21)),
        (_k(30), _k(31)),
        (_k(40), _k(41)),
        (_k(50), _k(51)),
        (_k(60), _k(61)),
        (_k(70),),
        (_k(71),),
        (_k(80),),
        (_k(81),),
        () if close_qud else (_k(90),),
        (_k(90),) if close_qud else (),
    )


def _claims(*, revision: bool = False) -> tuple[W08DiscourseClaim, ...]:
    direct = W08DiscourseClaim(
        _k(101),
        _k(201),
        "DIRECT",
        "REFUTED" if revision else "SUPPORTED",
        "INACTIVE" if revision else "ACTIVE",
        0 if revision else 1,
        evidence_keys=(_k(301),),
    )
    reported = W08DiscourseClaim(
        _k(102),
        _k(202),
        "ATTRIBUTION",
        "SUPPORTED",
        "ACTIVE",
        0,
        speaker_key=_k(30),
        holder_key=_k(40),
        evidence_keys=(_k(302),),
    )
    quoted = W08DiscourseClaim(
        _k(103),
        _k(203),
        "QUOTATION",
        "SUPPORTED",
        "ACTIVE",
        0,
        speaker_key=_k(31),
        evidence_keys=(_k(303),),
    )
    hypothesis = W08DiscourseClaim(
        _k(104),
        _k(204),
        "HYPOTHESIS",
        "UNKNOWN",
        "FORMING",
        0,
        evidence_keys=(_k(304),),
    )
    return direct, reported, quoted, hypothesis


def _request(
    *,
    change_kind: str = "NEW_OBSERVATION",
    reference: bool = True,
    generation: bool = True,
    clarification: tuple[int, ...] = (),
) -> W08DiscourseRequest:
    revision = change_kind != "NEW_OBSERVATION"
    return W08DiscourseRequest(
        _k(500),
        _scene(close_qud=change_kind == "OPEN_QUESTION_CLOSE"),
        _claims(revision=revision),
        change_kind,
        12,
        (_k(601),) if revision else (),
        int(reference),
        (_k(701), _k(702)) if reference else (),
        clarification,
        int(generation),
        (
            W08GenerationFormCandidate(_k(801), "PRONOUN", _k(701)),
            W08GenerationFormCandidate(_k(802), "PROPER_NAME", _k(701)),
            W08GenerationFormCandidate(_k(803), "DESCRIPTION", _k(701)),
            W08GenerationFormCandidate(_k(804), "ELLIPSIS", _k(701)),
        ) if generation else (),
    )


class _OwnerContext:
    def __init__(self):
        self.calls: list[str] = []
        self.ambiguous_reference = False
        self.include_reported_fact = False
        self.generation_blocked = False
        self.selected_form_index = 0


class _CenterOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[0]

    def __init__(self, context):
        self.context = context

    def form(self, request):
        self.context.calls.append("center")
        return W08CenterReceipt(
            self.owner_key,
            request.request_key,
            _k(901),
            (_k(902),),
            _k(903),
            (_k(904), _k(905)),
        )


class _EventOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[1]

    def __init__(self, context):
        self.context = context

    def append(self, request, center):
        assert center.center_key == _k(901)
        self.context.calls.append("event")
        return W08EventReceipt(
            self.owner_key,
            request.request_key,
            (_k(910), _k(911)),
            _k(912),
        )


class _LifecycleOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[2]

    def __init__(self, context):
        self.context = context

    def resolve(self, request, event):
        assert event.appended_event_key == _k(912)
        self.context.calls.append("lifecycle")
        return W08LifecycleReceipt(
            self.owner_key,
            request.request_key,
            tuple(item.claim_key for item in request.claims),
            tuple(key for item in request.claims for key in item.evidence_keys),
            tuple(item.claim_key for item in request.claims),
            (_k(920),),
        )


class _ProjectionOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[3]

    def __init__(self, context):
        self.context = context

    def project(self, request, event, lifecycle):
        self.context.calls.append("projection")
        active = tuple(
            item.proposition_key
            for item in request.claims
            if item.claim_key in lifecycle.adopted_claim_keys
            and item.current_projection_allowed
        )
        if self.context.include_reported_fact:
            active = (*active, _k(202))
        changed = (_k(930),) if request.change_kind != "NEW_OBSERVATION" else ()
        return W08ProjectionReceipt(
            self.owner_key,
            request.request_key,
            active,
            _k(931),
            _k(932),
            changed,
            changed,
            (_k(933),),
        )


class _AgendaOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[4]

    def __init__(self, context):
        self.context = context

    def plan(self, request, projection):
        self.context.calls.append("agenda")
        return W08AgendaReceipt(
            self.owner_key,
            request.request_key,
            request.changed_dependency_keys,
            projection.rebuilt_keys,
        )


class _ReferenceOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[5]

    def __init__(self, context):
        self.context = context

    def resolve(self, request, projection):
        self.context.calls.append("reference")
        adopted = () if self.context.ambiguous_reference else (
            (request.clarification_candidate_key,)
            if request.clarification_candidate_key else (request.reference_candidate_keys[0],)
        )
        return W08ReferenceReceipt(
            self.owner_key,
            request.request_key,
            request.reference_candidate_keys,
            adopted,
            (_k(940),),
            _k(941),
            "RESOLVED" if adopted else "CLARIFY",
        )


class _GenerationOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[6]

    def __init__(self, context):
        self.context = context

    def choose(self, request, projection, reference):
        self.context.calls.append("generation")
        selected = request.generation_form_candidates[
            self.context.selected_form_index
        ]
        recovered = _k(999) if self.context.generation_blocked else selected.antecedent_key
        return W08GenerationReceipt(
            self.owner_key,
            request.request_key,
            selected.form_key,
            selected.form_kind,
            selected.antecedent_key,
            recovered,
            _k(950),
            _k(951),
            "GROUNDING_BLOCKED" if self.context.generation_blocked else "RESOLVED",
            0 if self.context.generation_blocked else 1,
        )


class _ConsumerOwner:
    owner_key = W08_DISCOURSE_OWNER_KEYS[7]

    def __init__(self, context):
        self.context = context

    def consume(
        self,
        request,
        consumer_key,
        selected_candidate_key,
        evidence_keys,
        outcome_state,
    ):
        self.context.calls.append("use:" + consumer_key)
        directional = digest_value({"consumer": consumer_key, "kind": "choice"})
        use_key = digest_value({"consumer": consumer_key, "kind": "use"})
        outcome = digest_value({"consumer": consumer_key, "kind": "outcome"})
        return W08DiscourseUse(
            self.owner_key,
            consumer_key,
            request.request_key,
            selected_candidate_key,
            evidence_keys,
            directional,
            use_key,
            outcome_state,
            outcome,
        )


def _facade():
    context = _OwnerContext()
    owners = W08DiscourseOwners(
        _CenterOwner(context),
        _EventOwner(context),
        _LifecycleOwner(context),
        _ProjectionOwner(context),
        _AgendaOwner(context),
        _ReferenceOwner(context),
        _GenerationOwner(context),
        _ConsumerOwner(context),
    )
    return W08DiscourseFacade(owners), context


@pytest.fixture(scope="module")
def payload():
    contract = open_w08_frozen_contract(ROOT)
    return W08PayloadFirewall.open(
        ROOT, contract, make_w08_request(contract)
    ).read_training_payload()


def test_w08_discourse_train_schema_covers_all_public_typed_operations(payload):
    report = audit_w08_discourse_training(payload)
    assert report.observation_count == report.evidence_binding_count == 63
    assert (
        report.discourse_revision_count,
        report.discourse_information_count,
        report.open_set_clarification_count,
        report.attribution_quotation_count,
        report.raw_source_count,
    ) == (9, 16, 17, 17, 4)
    assert report.reference_plan_count == report.parser_revision_count == 2
    assert report.proposition_candidate_count == 49
    assert report.discourse_relation_count == 16
    assert report.information_structure_count == report.qud_candidate_count == 16
    assert report.open_obligation_count == 20
    assert report.attribution_candidate_count == 22
    assert report.quotation_span_count == 4
    assert report.reported_projection_violation_count == 0
    assert report.raw_truth_claim_count == report.authored_answer_read_count == 0


def test_w08_discourse_training_never_reads_authored_answers(payload):
    class GuardedEvidence:
        def __init__(self, value):
            self.value = value

        def __getattr__(self, name):
            if name == "typed_evidence":
                raise AssertionError("authored answer was read")
            return getattr(self.value, name)

    evidence = tuple(
        item if item.evidence_kind == "SOURCE_PARSER_RECEIPT_V1"
        else GuardedEvidence(item)
        for item in payload.teacher_evidence
    )
    report = audit_w08_discourse_training(W08TrainingPayload(
        payload.source_refs, payload.observations, evidence
    ))
    assert report.authored_answer_read_count == 0


def test_discourse_facade_delegates_complete_flow_and_emits_exact_urg_use():
    facade, context = _facade()
    receipt = facade.execute(_request())
    assert receipt.stop_state == "RESOLVED"
    assert context.calls == [
        "center",
        "event",
        "lifecycle",
        "projection",
        "agenda",
        "reference",
        "generation",
        "use:UNDERSTANDING",
        "use:REASONING",
        "use:GENERATION",
    ]
    assert receipt.owner_call_order == W08_DISCOURSE_OWNER_KEYS
    assert tuple(item.consumer_key for item in receipt.uses) == W08_CONSUMER_KEYS
    assert len({item.directional_choice_key for item in receipt.uses}) == 3
    assert len({item.use_key for item in receipt.uses}) == 3
    assert len({item.outcome_key for item in receipt.uses}) == 3
    assert receipt.event.prior_event_keys == (_k(910), _k(911))
    assert receipt.event.appended_event_key == _k(912)
    assert receipt.projection.active_proposition_keys == (_k(201),)
    assert receipt.reference.adopted_candidate_keys == (_k(701),)
    assert receipt.generation.selected_form_kind == "PRONOUN"
    assert receipt.generation.audience_recoverable == 1
    assert receipt.center.ring_receipt_key == _k(903)
    assert receipt.center.expansion_ring_keys == (_k(904), _k(905))
    assert len(receipt.center.obligation_keys) == 1
    assert len(receipt.event.prior_event_keys) == 2
    assert len(receipt.lifecycle.decision_keys) == 1
    assert len(receipt.uses) == 3
    assert len(_request().scene.entity_keys) == len(_request().scene.event_keys) == 2
    assert _request().scene.topic_keys and _request().scene.focus_keys
    assert _request().scene.given_keys and _request().scene.new_keys
    assert _request().scene.time_keys and _request().scene.location_keys


def test_reported_quoted_and_hypothetical_content_never_becomes_current_fact():
    for mode in ("ATTRIBUTION", "QUOTATION", "HYPOTHESIS"):
        with pytest.raises(W08DiscourseError, match="current fact"):
            replace(_claims()[1], mode=mode, current_projection_allowed=1)
    facade, context = _facade()
    context.include_reported_fact = True
    with pytest.raises(W08DiscourseError, match="promoted"):
        facade.execute(_request())


def test_reference_ambiguity_stops_before_generation_or_partial_use():
    facade, context = _facade()
    context.ambiguous_reference = True
    receipt = facade.execute(_request())
    assert receipt.stop_state == "CLARIFY"
    assert receipt.reference.candidate_keys == (_k(701), _k(702))
    assert receipt.reference.adopted_candidate_keys == ()
    assert receipt.generation is None and receipt.uses == ()
    assert not any(item.startswith("use:") for item in context.calls)
    assert "generation" not in context.calls


def test_reference_without_any_candidate_returns_unknown_without_partial_use():
    facade, context = _facade()
    request = replace(_request(), reference_candidate_keys=())

    def unknown(_request, _projection):
        context.calls.append("reference")
        return W08ReferenceReceipt(
            W08_DISCOURSE_OWNER_KEYS[5],
            request.request_key,
            (),
            (),
            (),
            _k(942),
            "UNKNOWN",
        )

    facade.owners.reference.resolve = unknown
    receipt = facade.execute(request)
    assert receipt.stop_state == "UNKNOWN"
    assert receipt.uses == () and receipt.generation is None


def test_explicit_clarification_selects_only_the_named_reference_candidate():
    facade, _ = _facade()
    receipt = facade.execute(_request(clarification=_k(702)))
    assert receipt.reference.adopted_candidate_keys == (_k(702),)
    assert receipt.uses[0].selected_candidate_key == _k(702)


def test_generation_audience_postcheck_blocks_unrecoverable_reference_form():
    facade, context = _facade()
    context.generation_blocked = True
    receipt = facade.execute(_request())
    assert receipt.stop_state == "GROUNDING_BLOCKED"
    assert receipt.generation.audience_recoverable == 0
    assert receipt.uses == ()
    assert not any(item.startswith("use:") for item in context.calls)


@pytest.mark.parametrize(
    "index,kind",
    tuple(enumerate(("PRONOUN", "PROPER_NAME", "DESCRIPTION", "ELLIPSIS"))),
)
def test_all_generation_reference_forms_use_the_same_audience_postcheck(index, kind):
    facade, context = _facade()
    context.selected_form_index = index
    receipt = facade.execute(_request())
    assert receipt.generation.selected_form_kind == kind
    assert receipt.generation.audience_recovered_key == receipt.generation.antecedent_key
    assert receipt.generation.stop_state == "RESOLVED"


@pytest.mark.parametrize(
    "change_kind,reference",
    [
        ("DENIAL", False),
        ("REFERENCE_REDIRECT", True),
        ("SOURCE_CONFLICT", False),
        ("OPEN_QUESTION_CLOSE", False),
    ],
)
def test_later_denial_redirect_conflict_and_qud_close_use_dependency_recompute(
    change_kind,
    reference,
):
    facade, _ = _facade()
    receipt = facade.execute(_request(
        change_kind=change_kind,
        reference=reference,
        generation=False,
    ))
    assert receipt.stop_state == "RESOLVED"
    assert receipt.projection.active_proposition_keys == ()
    assert receipt.projection.invalidated_keys == receipt.projection.rebuilt_keys == (
        _k(930),
    )
    assert receipt.projection.unaffected_keys == (_k(933),)
    assert receipt.agenda.changed_dependency_keys == (_k(601),)
    assert receipt.agenda.agenda_target_keys == (_k(930),)
    assert (
        receipt.projection.old_events_preserved,
        receipt.projection.old_observations_preserved,
        receipt.projection.old_evidence_preserved,
        receipt.projection.old_decisions_preserved,
    ) == (1, 1, 1, 1)


def test_discourse_contract_rejects_fifo_frequency_and_history_rewrite_shortcuts():
    with pytest.raises(W08DiscourseError, match="overwrote|append-only"):
        W08EventReceipt(
            W08_DISCOURSE_OWNER_KEYS[1],
            _k(1),
            (_k(2),),
            _k(2),
        )
    with pytest.raises(W08DiscourseError, match="history"):
        W08ProjectionReceipt(
            W08_DISCOURSE_OWNER_KEYS[3],
            _k(1),
            (),
            _k(2),
            _k(3),
            (_k(4),),
            (_k(4),),
            (_k(5),),
            old_decisions_preserved=0,
        )
    source = (ROOT / "src/pure_integer_ai/experiments/ph2_w08_discourse.py").read_text(
        encoding="utf-8"
    ).lower()
    assert "last_occurrence" not in source
    assert "global_frequency" not in source
    assert "surface_cue" not in source


def test_discourse_concrete_adapters_require_the_existing_owner_types():
    with pytest.raises(TypeError):
        W08MD03CenterOwner(object(), lambda *_: None)
    with pytest.raises(TypeError):
        W08SituationEventOwner(object(), lambda *_: None)
    with pytest.raises(TypeError):
        W08CurrentProjectionOwner(object(), lambda *_: None)
    with pytest.raises(TypeError):
        W08A01ReferenceOwner(object(), lambda *_: None, lambda *_: None)
    with pytest.raises(TypeError):
        W08GenerationOwner(object(), lambda *_: None)
    with pytest.raises(TypeError):
        W08AgendaOwner(object(), lambda *_: None)
    with pytest.raises(TypeError):
        W08ConsumerOwner(None)


def test_discourse_ablation_is_orthogonal_to_other_w08_bearings():
    full = {key: "PASS" for key in W08_DIMENSION_KEYS}
    ablated = dict(full)
    ablated["W-08-DISCOURSE"] = "FAIL"
    report = assess_w08_discourse_ablation(
        full_dimension_outcomes=full,
        ablated_dimension_outcomes=ablated,
    )
    assert report.affected_dimensions == ("W-08-DISCOURSE",)
    assert set(report.unaffected_dimensions) == set(W08_DIMENSION_KEYS) - {
        "W-08-DISCOURSE"
    }
    ablated["W-08-LOCAL_RECOMPUTE"] = "FAIL"
    with pytest.raises(W08DiscourseError, match="orthogonal"):
        assess_w08_discourse_ablation(
            full_dimension_outcomes=full,
            ablated_dimension_outcomes=ablated,
        )
