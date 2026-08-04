"""基于真实 typed generation 与 G-04 运行执行开放生成评估。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.experiments.authorized_generation_delivery import (
    AuthorizedGenerationClaim,
    AuthorizedGenerationDeliveryAuthority,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08OpenGenerationAuditReceipt,
    W08OpenGenerationCandidate,
    W08OpenGenerationError,
    W08OpenGenerationLayerOutcome,
    W08OpenGenerationRequest,
    W08OpenGenerationResourceReceipt,
    W08OpenGenerationUse,
    W08_OPEN_GENERATION_COVERAGE_KEYS,
    W08_OPEN_GENERATION_LAYER_KEYS,
    W08_OPEN_GENERATION_OWNER_KEYS,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


@dataclass(frozen=True)
class W08OpenGenerationSegment:
    claim_key: tuple[int, ...]
    run: QuestionAnswerRun
    authorization_claim: AuthorizedGenerationClaim

    def __post_init__(self) -> None:
        if (
            not isinstance(self.claim_key, tuple)
            or not self.claim_key
            or any(type(item) is not int for item in self.claim_key)
        ):
            raise W08OpenGenerationError("open-generation segment claim key is invalid")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("open-generation segment run type is invalid")
        if not isinstance(self.authorization_claim, AuthorizedGenerationClaim):
            raise TypeError("open-generation authorization claim type is invalid")


class W08OpenGenerationRuntime:
    """先独立选择，再生成所选内容，最后原子授权发布。"""

    def __init__(
        self,
        selector: Callable[[W08OpenGenerationRequest], tuple[int, ...]],
        generator: Callable[
            [W08OpenGenerationRequest, W08OpenGenerationCandidate],
            tuple[W08OpenGenerationSegment, ...],
        ],
        authority: AuthorizedGenerationDeliveryAuthority | None = None,
    ) -> None:
        if not callable(selector) or not callable(generator):
            raise TypeError("open-generation selector/generator must be callable")
        self.selector = selector
        self.generator = generator
        self.authority = authority or AuthorizedGenerationDeliveryAuthority()

    @staticmethod
    def _coverage(
        request: W08OpenGenerationRequest,
        candidate: W08OpenGenerationCandidate,
    ) -> tuple[str, ...]:
        scene = request.discourse_request.scene
        facts = {
            "MULTI_PROPOSITION_ORDER": len(candidate.ordered_claim_keys) >= 2,
            "REFERENCE_OR_ELLIPSIS": candidate.reference_form in {"PRONOUN", "ELLIPSIS"},
            "INFORMATION_STRUCTURE": bool(
                scene.topic_keys and scene.focus_keys and scene.given_keys and scene.new_keys
            ),
            "LATER_CORRECTION": request.discourse_request.change_kind
            in {"DENIAL", "REFERENCE_REDIRECT", "SOURCE_CONFLICT"},
            "OPEN_QUESTION": bool(scene.open_question_keys),
            "DIRECTED_CLARIFICATION": bool(
                request.discourse_request.clarification_candidate_key
            ),
            "MULTIPLE_LEGAL_SURFACES": len(
                {item.surface_family_key for item in request.candidates}
            )
            >= 2,
            "UNSEEN_CONTENT_COMBINATION": candidate.content_combination_key
            not in request.seen_content_combination_keys,
            "UNSEEN_SURFACE_FAMILY": candidate.surface_family_key
            not in request.seen_surface_family_keys,
        }
        return tuple(key for key in W08_OPEN_GENERATION_COVERAGE_KEYS if facts[key])

    @staticmethod
    def _selected_generation_candidate(run: QuestionAnswerRun):
        if run.planning_request is None or run.selection is None:
            return None
        selected = set(run.selection.selected_candidate_keys)
        candidates = tuple(
            item
            for item in run.planning_request.candidates
            if item.stable_key() in selected
        )
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _layer_outcome(
        request: W08OpenGenerationRequest,
        layer: str,
        passed: bool,
    ) -> W08OpenGenerationLayerOutcome:
        claims = tuple(item.claim_key for item in request.claims)
        return W08OpenGenerationLayerOutcome(
            layer,
            "PASS" if passed else "FAIL",
            claims,
            digest_value(
                {
                    "request": list(request.request_key),
                    "layer": layer,
                    "kind": "verifier",
                }
            ),
            digest_value(
                {
                    "request": list(request.request_key),
                    "layer": layer,
                    "passed": int(passed),
                    "kind": "outcome",
                }
            ),
        )

    @staticmethod
    def _uses(
        request: W08OpenGenerationRequest,
        candidate: W08OpenGenerationCandidate,
        layer_states: dict[str, bool],
        claim_states: dict[tuple[int, ...], bool],
    ) -> tuple[W08OpenGenerationUse, ...]:
        selection = {
            "CONTENT": ("CONTENT_SELECTION", candidate.content_combination_key),
            "STRUCTURE_LOGIC": ("STRUCTURE_ORDER", candidate.structure_key),
            "DISCOURSE_REFERENCE": (
                "REFERENCE_FORM",
                tuple(ord(char) + 1 for char in candidate.reference_form),
            ),
            "SURFACE_MORPHOLOGY": ("SURFACE_FAMILY", candidate.surface_family_key),
            "TASK_COMMUNICATIVE": ("TASK_DELIVERY", candidate.selection_constraint_key),
        }
        content_uses = tuple(
            W08OpenGenerationUse(
                "CONTENT_SELECTION",
                "CONTENT",
                request.request_key,
                candidate.candidate_key,
                binding.proposition_key,
                digest_value(
                    {
                        "request": list(request.request_key),
                        "candidate": list(candidate.candidate_key),
                        "claim": list(binding.claim_key),
                        "kind": "claim-use",
                    }
                ),
                digest_value(
                    {
                        "request": list(request.request_key),
                        "candidate": list(candidate.candidate_key),
                        "claim": list(binding.claim_key),
                        "kind": "claim-outcome",
                        "passed": int(claim_states[binding.claim_key]),
                    }
                ),
                "PASS" if claim_states[binding.claim_key] else "FAIL",
            )
            for binding in request.claims
        )
        layer_uses = tuple(
            W08OpenGenerationUse(
                selection[layer][0],
                layer,
                request.request_key,
                candidate.candidate_key,
                selection[layer][1],
                digest_value(
                    {
                        "request": list(request.request_key),
                        "candidate": list(candidate.candidate_key),
                        "layer": layer,
                        "kind": "use",
                    }
                ),
                digest_value(
                    {
                        "request": list(request.request_key),
                        "candidate": list(candidate.candidate_key),
                        "layer": layer,
                        "kind": "choice-outcome",
                        "passed": int(layer_states[layer]),
                    }
                ),
                "PASS" if layer_states[layer] else "FAIL",
            )
            for layer in W08_OPEN_GENERATION_LAYER_KEYS
            if layer != "CONTENT"
        )
        return (*content_uses, *layer_uses)

    def execute(
        self,
        request: W08OpenGenerationRequest,
    ) -> W08OpenGenerationAuditReceipt:
        if not isinstance(request, W08OpenGenerationRequest):
            raise TypeError("open-generation runtime requires W08OpenGenerationRequest")
        selected_key = self.selector(request)
        candidate = request.candidate(selected_key)
        calls = [W08_OPEN_GENERATION_OWNER_KEYS[0]]
        discourse_ready = bool(
            request.discourse_audit.stop_state == "RESOLVED"
            and request.discourse_audit.generation is not None
            and request.discourse_audit.generation.stop_state == "RESOLVED"
            and request.discourse_audit.generation.audience_recoverable == 1
        )
        segments = self.generator(request, candidate) if discourse_ready else ()
        if not isinstance(segments, tuple) or any(
            not isinstance(item, W08OpenGenerationSegment) for item in segments
        ):
            raise TypeError("open-generation generator returned invalid segments")
        if discourse_ready:
            calls.append(W08_OPEN_GENERATION_OWNER_KEYS[1])

        claim_by_key = {item.claim_key: item for item in request.claims}
        discourse_claims = {
            item.claim_key: item for item in request.discourse_request.claims
        }
        claim_states = {item.claim_key: False for item in request.claims}
        content_ok = len(segments) == len(candidate.ordered_claim_keys)
        structure_ok = tuple(item.claim_key for item in segments) == candidate.ordered_claim_keys
        discourse_ok = bool(
            discourse_ready
            and request.discourse_audit.generation.selected_form_kind
            == candidate.reference_form
        )
        surface_ok = bool(
            len(request.candidates) >= 2
            and candidate.surface_family_key not in request.seen_surface_family_keys
            and candidate.content_combination_key
            not in request.seen_content_combination_keys
            and (
                not candidate.complete_template_key
                or candidate.complete_template_key
                not in request.known_complete_template_keys
            )
        )
        task_ok = bool(segments)
        publications: list[tuple[int, ...]] = []
        delivery_keys: list[tuple[int, ...]] = []
        postcheck_runs = 0

        for segment in segments:
            binding = claim_by_key.get(segment.claim_key)
            discourse_claim = discourse_claims.get(segment.claim_key)
            actual = self._selected_generation_candidate(segment.run)
            if binding is None or discourse_claim is None or actual is None:
                content_ok = False
                task_ok = False
                continue
            actual_roles = tuple(
                sorted(item.role.stable_key() for item in actual.proposition.bindings)
            )
            source = semantic_source(actual.proposition.template)
            binding_ok = bool(
                actual.proposition.stable_key() == binding.proposition_key
                and source.stable_key() == binding.source_key
                and actual.scope.stable_key() == binding.scope_key
                and actual_roles == binding.role_keys
                and discourse_claim.proposition_key == binding.proposition_key
                and discourse_claim.holder_key == binding.holder_key
                and discourse_claim.evidence_state == binding.polarity
                and discourse_claim.mode == "DIRECT"
                and request.discourse_request.scene.source_key == binding.source_key
                and request.discourse_request.scene.scope_key == binding.scope_key
            )
            auth = segment.authorization_claim
            authorization_binding_ok = bool(
                auth.candidate_key == actual.stable_key()
                and auth.proposition_key == binding.proposition_key
                and auth.source.stable_key() == binding.source_key
                and auth.scope.stable_key() == binding.scope_key
                and auth.authorization_receipt_key == binding.authorization_receipt_key
            )
            claim_states[segment.claim_key] = binding_ok and authorization_binding_ok
            run_complete = bool(
                segment.run.complete
                and segment.run.postcheck is not None
                and segment.run.postcheck.complete
            )
            surface_ok &= bool(segment.run.generation is not None and segment.run.generation.complete)
            task_ok &= run_complete
            if segment.run.postcheck is not None:
                postcheck_runs += 1
            decision = self.authority.authorize(
                segment.run,
                (segment.authorization_claim,),
            )
            delivery_keys.append(decision.audit_key)
            task_ok &= decision.deliverable
            if decision.envelope is not None:
                publications.append(decision.envelope.units)
        if discourse_ready:
            calls.append(W08_OPEN_GENERATION_OWNER_KEYS[2])

        if tuple(segment.claim_key for segment in segments) != candidate.ordered_claim_keys:
            structure_ok = False
        if set(candidate.ordered_claim_keys) != set(claim_by_key):
            structure_ok = False
        content_ok &= all(claim_states.values())
        coverage = self._coverage(request, candidate)
        layer_states = {
            "CONTENT": content_ok,
            "STRUCTURE_LOGIC": structure_ok,
            "DISCOURSE_REFERENCE": discourse_ok,
            "SURFACE_MORPHOLOGY": surface_ok,
            "TASK_COMMUNICATIVE": task_ok,
        }
        layers = tuple(
            self._layer_outcome(request, layer, layer_states[layer])
            for layer in W08_OPEN_GENERATION_LAYER_KEYS
        )
        uses = self._uses(request, candidate, layer_states, claim_states)
        calls.append(W08_OPEN_GENERATION_OWNER_KEYS[3])
        resolved = all(layer_states.values()) and coverage == W08_OPEN_GENERATION_COVERAGE_KEYS
        publication = tuple(publications) if resolved else ()
        state = "RESOLVED" if resolved else (
            "ACCESS_BLOCKED" if not task_ok else "GROUNDING_BLOCKED"
        )
        resources = W08OpenGenerationResourceReceipt(
            len(request.candidates),
            len(candidate.ordered_claim_keys),
            len(segments),
            postcheck_runs,
            len(delivery_keys),
            len(publication),
            sum(len(item) for item in publication),
            len(request.candidates)
            + len(segments) * 3
            + len(layers)
            + len(uses),
        )
        return W08OpenGenerationAuditReceipt(
            request.request_key,
            state,
            candidate.candidate_key,
            coverage,
            uses,
            layers,
            tuple(delivery_keys),
            publication,
            resources,
            tuple(calls),
        )


__all__ = [
    "W08OpenGenerationRuntime",
    "W08OpenGenerationSegment",
]
