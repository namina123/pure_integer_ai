"""带五层独立记账的 W08 有界开放生成合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_contract import W08_RESOURCE_BUDGET, W08_STOP_STATES
from pure_integer_ai.experiments.ph2_w08_discourse_contract import (
    W08DiscourseAuditReceipt,
    W08DiscourseRequest,
    W08_GENERATION_REFERENCE_FORMS,
)


W08_OPEN_GENERATION_LAYER_KEYS = (
    "CONTENT",
    "STRUCTURE_LOGIC",
    "DISCOURSE_REFERENCE",
    "SURFACE_MORPHOLOGY",
    "TASK_COMMUNICATIVE",
)
W08_OPEN_GENERATION_COVERAGE_KEYS = (
    "MULTI_PROPOSITION_ORDER",
    "REFERENCE_OR_ELLIPSIS",
    "INFORMATION_STRUCTURE",
    "LATER_CORRECTION",
    "OPEN_QUESTION",
    "DIRECTED_CLARIFICATION",
    "MULTIPLE_LEGAL_SURFACES",
    "UNSEEN_CONTENT_COMBINATION",
    "UNSEEN_SURFACE_FAMILY",
)
W08_OPEN_GENERATION_OWNER_KEYS = (
    "W08_OPEN_CANDIDATE_SELECTOR",
    "W08_TYPED_GENERATION_POSTCHECK_OWNER",
    "W08_AUTHORIZED_DELIVERY_OWNER",
    "W08_OPEN_LAYER_OUTCOME_OWNER",
)
W08_OPEN_GENERATION_LAYER_STATES = ("FAIL", "NE", "PASS")


class W08OpenGenerationError(ValueError):
    """开放生成身份、层级或发布合同发生漂移。"""


def _key(value: object, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(item) is not int for item in value):
        raise W08OpenGenerationError(f"{where} is not a strict integer key")
    if not allow_empty and not value:
        raise W08OpenGenerationError(f"{where} is empty")
    return value


def _keys(
    values: object, *, where: str, minimum: int = 0, canonical: bool = False
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(item, tuple)
        or not item
        or any(type(value) is not int for value in item)
        for item in values
    ):
        raise W08OpenGenerationError(f"{where} is not an integer-key tuple")
    if len(values) < minimum or len(set(values)) != len(values):
        raise W08OpenGenerationError(f"{where} count or identity is invalid")
    if canonical and values != tuple(sorted(values)):
        raise W08OpenGenerationError(f"{where} is not canonical")
    return values


@dataclass(frozen=True, order=True)
class W08OpenClaimBinding:
    claim_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    role_keys: tuple[tuple[int, ...], ...]
    polarity: str
    holder_key: tuple[int, ...]
    authorization_receipt_key: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "claim_key",
            "proposition_key",
            "source_key",
            "scope_key",
            "authorization_receipt_key",
        ):
            _key(getattr(self, name), where=f"open claim {name}")
        _keys(self.role_keys, where="open claim roles", minimum=1, canonical=True)
        if self.polarity not in {"SUPPORTED", "REFUTED"}:
            raise W08OpenGenerationError("open claim polarity is invalid")
        _key(self.holder_key, where="open claim holder")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "claim": list(self.claim_key),
                "proposition": list(self.proposition_key),
                "source": list(self.source_key),
                "scope": list(self.scope_key),
                "roles": [list(item) for item in self.role_keys],
                "polarity": self.polarity,
                "holder": list(self.holder_key),
                "authorization": list(self.authorization_receipt_key),
            }
        )


@dataclass(frozen=True, order=True)
class W08OpenGenerationCandidate:
    candidate_key: tuple[int, ...]
    content_combination_key: tuple[int, ...]
    ordered_claim_keys: tuple[tuple[int, ...], ...]
    structure_key: tuple[int, ...]
    reference_form: str
    surface_family_key: tuple[int, ...]
    selection_constraint_key: tuple[int, ...]
    complete_template_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_key",
            "content_combination_key",
            "structure_key",
            "surface_family_key",
            "selection_constraint_key",
        ):
            _key(getattr(self, name), where=f"open candidate {name}")
        _keys(self.ordered_claim_keys, where="open candidate claim order", minimum=2)
        if self.reference_form not in W08_GENERATION_REFERENCE_FORMS:
            raise W08OpenGenerationError("open candidate reference form is invalid")
        _key(
            self.complete_template_key,
            where="open candidate template",
            allow_empty=True,
        )

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "candidate": list(self.candidate_key),
                "combination": list(self.content_combination_key),
                "order": [list(item) for item in self.ordered_claim_keys],
                "structure": list(self.structure_key),
                "reference": self.reference_form,
                "surface_family": list(self.surface_family_key),
                "constraint": list(self.selection_constraint_key),
                "template": list(self.complete_template_key),
            }
        )


@dataclass(frozen=True)
class W08OpenGenerationRequest:
    request_key: tuple[int, ...]
    discourse_request: W08DiscourseRequest
    discourse_audit: W08DiscourseAuditReceipt
    claims: tuple[W08OpenClaimBinding, ...]
    candidates: tuple[W08OpenGenerationCandidate, ...]
    seen_content_combination_keys: tuple[tuple[int, ...], ...]
    seen_surface_family_keys: tuple[tuple[int, ...], ...]
    known_complete_template_keys: tuple[tuple[int, ...], ...]
    selector_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _key(self.request_key, where="open-generation request")
        if not isinstance(self.discourse_request, W08DiscourseRequest):
            raise TypeError("open-generation discourse request type is invalid")
        if not isinstance(self.discourse_audit, W08DiscourseAuditReceipt):
            raise TypeError("open-generation discourse audit type is invalid")
        if (
            self.discourse_request.request_key != self.discourse_audit.request_key
        ):
            raise W08OpenGenerationError("open-generation discourse identity drifted")
        if (
            not isinstance(self.claims, tuple)
            or len(self.claims) < 2
            or any(not isinstance(item, W08OpenClaimBinding) for item in self.claims)
        ):
            raise W08OpenGenerationError("open-generation requires multiple typed claims")
        claim_keys = tuple(item.claim_key for item in self.claims)
        if len(set(claim_keys)) != len(claim_keys):
            raise W08OpenGenerationError("open-generation claim is duplicated")
        if (
            not isinstance(self.candidates, tuple)
            or len(self.candidates) < 2
            or any(not isinstance(item, W08OpenGenerationCandidate) for item in self.candidates)
            or self.candidates != tuple(sorted(self.candidates))
        ):
            raise W08OpenGenerationError("open-generation candidates are not canonical")
        if len({item.candidate_key for item in self.candidates}) != len(self.candidates):
            raise W08OpenGenerationError("open-generation candidate identity is duplicated")
        if any(set(item.ordered_claim_keys) != set(claim_keys) for item in self.candidates):
            raise W08OpenGenerationError("open-generation candidate changed claim coverage")
        if len({item.content_combination_key for item in self.candidates}) != 1:
            raise W08OpenGenerationError("legal surface candidates changed content")
        if len({item.surface_family_key for item in self.candidates}) < 2:
            raise W08OpenGenerationError("multiple legal surface families are missing")
        for name in (
            "seen_content_combination_keys",
            "seen_surface_family_keys",
            "known_complete_template_keys",
        ):
            _keys(getattr(self, name), where=f"open-generation {name}", canonical=True)
        _key(self.selector_key, where="open-generation selector")

    def candidate(self, key: tuple[int, ...]) -> W08OpenGenerationCandidate:
        selected = tuple(item for item in self.candidates if item.candidate_key == key)
        if len(selected) != 1:
            raise W08OpenGenerationError("selector chose outside candidate inventory")
        return selected[0]

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "request": list(self.request_key),
                "discourse": list(self.discourse_request.request_key),
                "claims": [list(item.stable_key()) for item in self.claims],
                "candidates": [list(item.stable_key()) for item in self.candidates],
                "seen_combinations": [
                    list(item) for item in self.seen_content_combination_keys
                ],
                "seen_surface_families": [
                    list(item) for item in self.seen_surface_family_keys
                ],
                "templates": [list(item) for item in self.known_complete_template_keys],
                "selector": list(self.selector_key),
            }
        )


@dataclass(frozen=True, order=True)
class W08OpenGenerationUse:
    choice_kind: str
    layer_key: str
    request_key: tuple[int, ...]
    candidate_key: tuple[int, ...]
    selected_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]
    outcome_state: str

    def __post_init__(self) -> None:
        if self.choice_kind not in {
            "CONTENT_SELECTION",
            "STRUCTURE_ORDER",
            "REFERENCE_FORM",
            "SURFACE_FAMILY",
            "TASK_DELIVERY",
        }:
            raise W08OpenGenerationError("open-generation choice kind is invalid")
        if self.layer_key not in W08_OPEN_GENERATION_LAYER_KEYS:
            raise W08OpenGenerationError("open-generation Use layer is invalid")
        for name in (
            "request_key",
            "candidate_key",
            "selected_key",
            "use_key",
            "outcome_key",
        ):
            _key(getattr(self, name), where=f"open-generation Use {name}")
        if self.outcome_state not in W08_OPEN_GENERATION_LAYER_STATES:
            raise W08OpenGenerationError("open-generation Use outcome is invalid")


@dataclass(frozen=True, order=True)
class W08OpenGenerationLayerOutcome:
    layer_key: str
    state: str
    claim_keys: tuple[tuple[int, ...], ...]
    verifier_key: tuple[int, ...]
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer_key not in W08_OPEN_GENERATION_LAYER_KEYS:
            raise W08OpenGenerationError("open-generation layer is invalid")
        if self.state not in W08_OPEN_GENERATION_LAYER_STATES:
            raise W08OpenGenerationError("open-generation layer state is invalid")
        _keys(self.claim_keys, where="open-generation layer claims", minimum=1)
        _key(self.verifier_key, where="open-generation verifier")
        _key(self.outcome_key, where="open-generation layer outcome")


@dataclass(frozen=True)
class W08OpenGenerationResourceReceipt:
    candidate_count: int
    selected_claim_count: int
    generation_runs: int
    postcheck_runs: int
    delivery_checks: int
    published_surfaces: int
    published_units: int
    logic_operations: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise W08OpenGenerationError("open-generation resource count is invalid")
        if (
            self.candidate_count > W08_RESOURCE_BUDGET["max_records"]
            or self.selected_claim_count > W08_RESOURCE_BUDGET["max_records"]
            or self.generation_runs > W08_RESOURCE_BUDGET["max_records"]
            or self.postcheck_runs > W08_RESOURCE_BUDGET["max_records"]
            or self.delivery_checks > W08_RESOURCE_BUDGET["max_records"]
            or self.published_surfaces > W08_RESOURCE_BUDGET["max_records"]
            or self.published_units > W08_RESOURCE_BUDGET["max_payload_bytes"]
            or self.logic_operations > W08_RESOURCE_BUDGET["max_logic_operations"]
        ):
            raise W08OpenGenerationError("open-generation resource budget was exceeded")


@dataclass(frozen=True)
class W08OpenGenerationAuditReceipt:
    request_key: tuple[int, ...]
    state: str
    selected_candidate_key: tuple[int, ...]
    coverage_keys: tuple[str, ...]
    uses: tuple[W08OpenGenerationUse, ...]
    layers: tuple[W08OpenGenerationLayerOutcome, ...]
    delivery_audit_keys: tuple[tuple[int, ...], ...]
    publication_units: tuple[tuple[int, ...], ...]
    resources: W08OpenGenerationResourceReceipt
    owner_calls: tuple[str, ...]
    public_open_generation_state: str = "NE_NOT_YET_EVALUABLE"
    private_label_read_count: int = 0
    host_learning_write_count: int = 0
    memory_learning_write_count: int = 0

    def __post_init__(self) -> None:
        _key(self.request_key, where="open-generation audit request")
        if self.state not in W08_STOP_STATES:
            raise W08OpenGenerationError("open-generation audit state is invalid")
        _key(self.selected_candidate_key, where="open-generation selected candidate")
        if tuple(self.coverage_keys) != tuple(
            key for key in W08_OPEN_GENERATION_COVERAGE_KEYS if key in self.coverage_keys
        ) or len(set(self.coverage_keys)) != len(self.coverage_keys):
            raise W08OpenGenerationError("open-generation coverage inventory drifted")
        if tuple(item.layer_key for item in self.layers) != W08_OPEN_GENERATION_LAYER_KEYS:
            raise W08OpenGenerationError("open-generation five-layer ledger drifted")
        if len({item.use_key for item in self.uses}) != len(self.uses):
            raise W08OpenGenerationError("open-generation reused one Use")
        _keys(self.delivery_audit_keys, where="open-generation delivery audits")
        if any(
            not isinstance(units, tuple)
            or not units
            or any(type(item) is not int or item <= 0 for item in units)
            for units in self.publication_units
        ):
            raise W08OpenGenerationError("open-generation publication units are invalid")
        if not isinstance(self.resources, W08OpenGenerationResourceReceipt):
            raise TypeError("open-generation resource receipt type is invalid")
        if self.owner_calls != tuple(
            key for key in W08_OPEN_GENERATION_OWNER_KEYS if key in self.owner_calls
        ):
            raise W08OpenGenerationError("open-generation owner order drifted")
        if self.public_open_generation_state != "NE_NOT_YET_EVALUABLE":
            raise W08OpenGenerationError("public OPEN_GENERATION changed before formal run")
        if any(
            (
                self.private_label_read_count,
                self.host_learning_write_count,
                self.memory_learning_write_count,
            )
        ):
            raise W08OpenGenerationError("open-generation crossed a forbidden boundary")
        passed = all(item.state == "PASS" for item in self.layers)
        if self.state == "RESOLVED":
            if not passed or self.coverage_keys != W08_OPEN_GENERATION_COVERAGE_KEYS:
                raise W08OpenGenerationError("resolved open generation lacks hard conjuncts")
            if not self.publication_units:
                raise W08OpenGenerationError("resolved open generation lacks publication")
        elif self.publication_units:
            raise W08OpenGenerationError("failed open generation exposed publication")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "request": list(self.request_key),
                "state": self.state,
                "selected": list(self.selected_candidate_key),
                "coverage": list(self.coverage_keys),
                "uses": [
                    [item.choice_kind, item.layer_key, list(item.use_key), item.outcome_state]
                    for item in self.uses
                ],
                "layers": [
                    [item.layer_key, item.state, list(item.outcome_key)]
                    for item in self.layers
                ],
                "delivery": [list(item) for item in self.delivery_audit_keys],
                "publication_sha": [list(digest_value(list(item))) for item in self.publication_units],
            }
        )


@dataclass(frozen=True)
class W08OpenGenerationAblationReport:
    ablation_key: str
    full_state: str
    ablated_state: str
    affected_layers: tuple[str, ...]
    zero_publication: int

    def __post_init__(self) -> None:
        if self.ablation_key != "CLOSED_RENDERER_TEMPLATE_REPLAY":
            raise W08OpenGenerationError("open-generation ablation is not registered")
        if (
            self.full_state != "RESOLVED"
            or self.ablated_state == "RESOLVED"
            or self.affected_layers != ("SURFACE_MORPHOLOGY",)
            or self.zero_publication != 1
        ):
            raise W08OpenGenerationError("template replay ablation did not bear")


def assess_w08_open_generation_ablation(
    full: W08OpenGenerationAuditReceipt,
    ablated: W08OpenGenerationAuditReceipt,
) -> W08OpenGenerationAblationReport:
    if not isinstance(full, W08OpenGenerationAuditReceipt) or not isinstance(
        ablated, W08OpenGenerationAuditReceipt
    ):
        raise TypeError("open-generation ablation receipt type is invalid")
    if full.request_key != ablated.request_key:
        raise W08OpenGenerationError("open-generation ablation changed request identity")
    full_layers = {item.layer_key: item.state for item in full.layers}
    ablated_layers = {item.layer_key: item.state for item in ablated.layers}
    affected = tuple(
        key
        for key in W08_OPEN_GENERATION_LAYER_KEYS
        if full_layers[key] != ablated_layers[key]
    )
    return W08OpenGenerationAblationReport(
        "CLOSED_RENDERER_TEMPLATE_REPLAY",
        full.state,
        ablated.state,
        affected,
        int(not ablated.publication_units),
    )


__all__ = [
    "W08OpenClaimBinding",
    "W08OpenGenerationAblationReport",
    "W08OpenGenerationAuditReceipt",
    "W08OpenGenerationCandidate",
    "W08OpenGenerationError",
    "W08OpenGenerationLayerOutcome",
    "W08OpenGenerationRequest",
    "W08OpenGenerationResourceReceipt",
    "W08OpenGenerationUse",
    "W08_OPEN_GENERATION_COVERAGE_KEYS",
    "W08_OPEN_GENERATION_LAYER_KEYS",
    "W08_OPEN_GENERATION_OWNER_KEYS",
    "assess_w08_open_generation_ablation",
]
