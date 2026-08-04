"""W-08 篇章 facade：只编排既有 owner，不复制事件、投影或真值状态。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS
from pure_integer_ai.experiments.ph2_w08_discourse_contract import (
    W08AgendaReceipt,
    W08CenterReceipt,
    W08DiscourseAblationReport,
    W08DiscourseAuditReceipt,
    W08DiscourseClaim,
    W08DiscourseError,
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
    W08_DISCOURSE_CHANGE_KINDS,
    W08_DISCOURSE_CLAIM_MODES,
    W08_DISCOURSE_OWNER_KEYS,
    W08_GENERATION_REFERENCE_FORMS,
    assess_w08_discourse_ablation,
)


class W08DiscourseFacade:
    """依次委托 center/event/lifecycle/projection/agenda/reference/G/Use owner。"""

    def __init__(self, owners: W08DiscourseOwners) -> None:
        if not isinstance(owners, W08DiscourseOwners):
            raise TypeError("W08DiscourseFacade requires W08DiscourseOwners")
        self.owners = owners

    @staticmethod
    def _require_request(
        receipt_request: tuple[int, ...],
        request: W08DiscourseRequest,
    ) -> None:
        if receipt_request != request.request_key:
            raise W08DiscourseError("owner receipt belongs to another request")

    def execute(self, request: W08DiscourseRequest) -> W08DiscourseAuditReceipt:
        if not isinstance(request, W08DiscourseRequest):
            raise TypeError("discourse facade request type is invalid")
        calls: list[str] = []

        center = self.owners.center.form(request)
        calls.append(self.owners.center.owner_key)
        self._require_request(center.request_key, request)
        if center.stop_state != "RESOLVED":
            return self._stopped(request, center.stop_state, center, calls)

        event = self.owners.events.append(request, center)
        calls.append(self.owners.events.owner_key)
        self._require_request(event.request_key, request)
        if event.stop_state != "RESOLVED":
            return self._stopped(request, event.stop_state, center, calls, event=event)

        lifecycle = self.owners.lifecycle.resolve(request, event)
        calls.append(self.owners.lifecycle.owner_key)
        self._require_request(lifecycle.request_key, request)
        expected_claims = {item.claim_key for item in request.claims}
        if set(lifecycle.candidate_keys) != expected_claims:
            raise W08DiscourseError("lifecycle receipt does not cover request claims")
        if lifecycle.stop_state != "RESOLVED":
            return self._stopped(
                request,
                lifecycle.stop_state,
                center,
                calls,
                event=event,
                lifecycle=lifecycle,
            )

        projection = self.owners.projection.project(request, event, lifecycle)
        calls.append(self.owners.projection.owner_key)
        self._require_request(projection.request_key, request)
        claim_by_key = {item.claim_key: item for item in request.claims}
        expected_active = {
            claim_by_key[key].proposition_key
            for key in lifecycle.adopted_claim_keys
            if claim_by_key[key].current_projection_allowed == 1
        }
        if set(projection.active_proposition_keys) != expected_active:
            raise W08DiscourseError("projection promoted or dropped a typed claim")
        if request.change_kind != "NEW_OBSERVATION" and not projection.invalidated_keys:
            raise W08DiscourseError("discourse revision did not invalidate dependencies")
        if projection.stop_state != "RESOLVED":
            return self._stopped(
                request,
                projection.stop_state,
                center,
                calls,
                event=event,
                lifecycle=lifecycle,
                projection=projection,
            )

        agenda = self.owners.agenda.plan(request, projection)
        calls.append(self.owners.agenda.owner_key)
        self._require_request(agenda.request_key, request)
        if agenda.changed_dependency_keys != request.changed_dependency_keys:
            raise W08DiscourseError("agenda changed-dependency input drifted")
        if request.change_kind != "NEW_OBSERVATION" and (
            set(agenda.agenda_target_keys) != set(projection.rebuilt_keys)
        ):
            raise W08DiscourseError("agenda did not target exact rebuilt projection slots")
        if agenda.stop_state != "RESOLVED":
            return self._stopped(
                request,
                agenda.stop_state,
                center,
                calls,
                event=event,
                lifecycle=lifecycle,
                projection=projection,
                agenda=agenda,
            )

        reference = None
        if request.reference_required:
            reference = self.owners.reference.resolve(request, projection)
            calls.append(self.owners.reference.owner_key)
            self._require_request(reference.request_key, request)
            if set(reference.candidate_keys) != set(request.reference_candidate_keys):
                raise W08DiscourseError("reference owner changed the candidate set")
            if request.clarification_candidate_key and (
                reference.adopted_candidate_keys != (request.clarification_candidate_key,)
            ):
                raise W08DiscourseError("clarification result did not select its candidate")
            if reference.stop_state != "RESOLVED":
                return self._stopped(
                    request,
                    reference.stop_state,
                    center,
                    calls,
                    event=event,
                    lifecycle=lifecycle,
                    projection=projection,
                    agenda=agenda,
                    reference=reference,
                )

        generation = None
        if request.generation_required:
            generation = self.owners.generation.choose(request, projection, reference)
            calls.append(self.owners.generation.owner_key)
            self._require_request(generation.request_key, request)
            candidates = {
                item.form_key: item for item in request.generation_form_candidates
            }
            selected = candidates.get(generation.selected_form_key)
            if (
                selected is None
                or selected.form_kind != generation.selected_form_kind
                or selected.antecedent_key != generation.antecedent_key
            ):
                raise W08DiscourseError(
                    "generation owner selected outside typed form candidates"
                )
            if generation.stop_state != "RESOLVED":
                return self._stopped(
                    request,
                    generation.stop_state,
                    center,
                    calls,
                    event=event,
                    lifecycle=lifecycle,
                    projection=projection,
                    agenda=agenda,
                    reference=reference,
                    generation=generation,
                )

        selected_by_consumer = {
            "UNDERSTANDING": (
                reference.adopted_candidate_keys[0]
                if reference is not None
                else projection.after_projection_ref
            ),
            "REASONING": projection.after_projection_ref,
            "GENERATION": (
                generation.selected_form_key
                if generation is not None
                else projection.after_projection_ref
            ),
        }
        evidence = tuple(
            sorted(
                {
                    *lifecycle.evidence_keys,
                    *(() if reference is None else reference.evidence_keys),
                }
            )
        )
        uses = []
        for consumer in W08_CONSUMER_KEYS:
            use = self.owners.consumers.consume(
                request,
                consumer,
                selected_by_consumer[consumer],
                evidence,
                "RESOLVED",
            )
            self._require_request(use.request_key, request)
            if (
                use.consumer_key != consumer
                or use.selected_candidate_key != selected_by_consumer[consumer]
                or use.evidence_keys != evidence
                or use.outcome_state != "RESOLVED"
            ):
                raise W08DiscourseError("consumer Use/outcome drifted")
            uses.append(use)
        calls.append(self.owners.consumers.owner_key)
        return W08DiscourseAuditReceipt(
            request.request_key,
            "RESOLVED",
            center,
            event,
            lifecycle,
            projection,
            agenda,
            reference,
            generation,
            tuple(uses),
            tuple(calls),
            0,
        )

    @staticmethod
    def _stopped(
        request: W08DiscourseRequest,
        state: str,
        center: W08CenterReceipt,
        calls: list[str],
        *,
        event: W08EventReceipt | None = None,
        lifecycle: W08LifecycleReceipt | None = None,
        projection: W08ProjectionReceipt | None = None,
        agenda: W08AgendaReceipt | None = None,
        reference: W08ReferenceReceipt | None = None,
        generation: W08GenerationReceipt | None = None,
    ) -> W08DiscourseAuditReceipt:
        return W08DiscourseAuditReceipt(
            request.request_key,
            state,
            center,
            event,
            lifecycle,
            projection,
            agenda,
            reference,
            generation,
            (),
            tuple(calls),
            0,
        )


__all__ = [
    "W08AgendaReceipt",
    "W08CenterReceipt",
    "W08DiscourseAuditReceipt",
    "W08DiscourseAblationReport",
    "W08DiscourseClaim",
    "W08DiscourseError",
    "W08DiscourseFacade",
    "W08DiscourseOwners",
    "W08DiscourseRequest",
    "W08DiscourseScene",
    "W08DiscourseUse",
    "W08EventReceipt",
    "W08GenerationFormCandidate",
    "W08GenerationReceipt",
    "W08LifecycleReceipt",
    "W08ProjectionReceipt",
    "W08ReferenceReceipt",
    "W08_DISCOURSE_CHANGE_KINDS",
    "W08_DISCOURSE_CLAIM_MODES",
    "W08_DISCOURSE_OWNER_KEYS",
    "W08_GENERATION_REFERENCE_FORMS",
    "assess_w08_discourse_ablation",
]
