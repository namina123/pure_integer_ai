"""W-08 篇章 typed contract、owner protocol 与审计 receipt。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_STOP_STATES,
)


W08_DISCOURSE_OWNER_KEYS = (
    "MD-03_DIRECTIONAL_CENTER_OWNER",
    "MD-02_SITUATION_EVENT_OWNER",
    "H-04_H-05_CANDIDATE_LIFECYCLE_OWNER",
    "MD-02_CURRENT_PROJECTION_OWNER",
    "A-10_DEPENDENCY_AGENDA_OWNER",
    "A-01_OCCURRENCE_REFERENCE_OWNER",
    "G-02_G-03_G-04_GENERATION_OWNER",
    "W-06_W-07_URG_CONSUMER_OWNER",
)
W08_DISCOURSE_CHANGE_KINDS = (
    "NEW_OBSERVATION",
    "DENIAL",
    "REFERENCE_REDIRECT",
    "SOURCE_CONFLICT",
    "OPEN_QUESTION_CLOSE",
)
W08_DISCOURSE_CLAIM_MODES = (
    "DIRECT",
    "ATTRIBUTION",
    "QUOTATION",
    "HYPOTHESIS",
)
W08_DISCOURSE_EVIDENCE_STATES = (
    "SUPPORTED",
    "REFUTED",
    "COMPETING",
    "UNKNOWN",
)
W08_DISCOURSE_LIFECYCLES = (
    "FORMING",
    "ACTIVE",
    "INACTIVE",
    "SUPERSEDED",
)
W08_GENERATION_REFERENCE_FORMS = (
    "PRONOUN",
    "PROPER_NAME",
    "DESCRIPTION",
    "ELLIPSIS",
)


class W08DiscourseError(ValueError):
    """W-08 篇章 owner、请求、投影、Use 或审计 receipt 不闭合。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise W08DiscourseError(f"{where} must be a non-empty strict integer tuple")
    return value


def _keys(
    value: object,
    *,
    where: str,
    allow_empty: bool = True,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise W08DiscourseError(f"{where} must be a tuple of keys")
    for item in value:
        _key(item, where=where)
    if len(set(value)) != len(value):
        raise W08DiscourseError(f"{where} contains duplicate keys")
    return value


def _bit(value: object, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise W08DiscourseError(f"{where} must be a strict bit")
    return value


def _stop(value: str, *, where: str) -> str:
    if value not in W08_STOP_STATES:
        raise W08DiscourseError(f"{where} is not a W-08 stop state")
    return value


@dataclass(frozen=True)
class W08DiscourseScene:
    """一个 typed 篇章现场；所有槽位都是既有对象引用而非 surface cue。"""

    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    entity_keys: tuple[tuple[int, ...], ...]
    event_keys: tuple[tuple[int, ...], ...]
    speaker_keys: tuple[tuple[int, ...], ...]
    holder_keys: tuple[tuple[int, ...], ...]
    time_keys: tuple[tuple[int, ...], ...]
    location_keys: tuple[tuple[int, ...], ...]
    topic_keys: tuple[tuple[int, ...], ...]
    focus_keys: tuple[tuple[int, ...], ...]
    given_keys: tuple[tuple[int, ...], ...]
    new_keys: tuple[tuple[int, ...], ...]
    open_question_keys: tuple[tuple[int, ...], ...]
    closed_question_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        _key(self.source_key, where="scene source_key")
        _key(self.scope_key, where="scene scope_key")
        groups = (
            ("entities", self.entity_keys, False),
            ("events", self.event_keys, False),
            ("speakers", self.speaker_keys, True),
            ("holders", self.holder_keys, True),
            ("times", self.time_keys, True),
            ("locations", self.location_keys, True),
            ("topics", self.topic_keys, True),
            ("foci", self.focus_keys, True),
            ("given", self.given_keys, True),
            ("new", self.new_keys, True),
            ("open questions", self.open_question_keys, True),
            ("closed questions", self.closed_question_keys, True),
        )
        for name, values, allow_empty in groups:
            _keys(values, where=f"scene {name}", allow_empty=allow_empty)
        if set(self.given_keys) & set(self.new_keys):
            raise W08DiscourseError("given/new keys must remain distinct")
        if set(self.open_question_keys) & set(self.closed_question_keys):
            raise W08DiscourseError("open/closed QUD keys must remain distinct")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "source": list(self.source_key),
            "scope": list(self.scope_key),
            "entities": [list(item) for item in self.entity_keys],
            "events": [list(item) for item in self.event_keys],
            "speakers": [list(item) for item in self.speaker_keys],
            "holders": [list(item) for item in self.holder_keys],
            "times": [list(item) for item in self.time_keys],
            "locations": [list(item) for item in self.location_keys],
            "topics": [list(item) for item in self.topic_keys],
            "foci": [list(item) for item in self.focus_keys],
            "given": [list(item) for item in self.given_keys],
            "new": [list(item) for item in self.new_keys],
            "open_questions": [list(item) for item in self.open_question_keys],
            "closed_questions": [list(item) for item in self.closed_question_keys],
        })


@dataclass(frozen=True)
class W08DiscourseClaim:
    """一个来源化命题候选及其 holder/speaker、Evidence 和生命周期。"""

    claim_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    mode: str
    evidence_state: str
    lifecycle: str
    current_projection_allowed: int
    speaker_key: tuple[int, ...] = ()
    holder_key: tuple[int, ...] = ()
    evidence_keys: tuple[tuple[int, ...], ...] = ()
    supersedes_claim_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _key(self.claim_key, where="claim key")
        _key(self.proposition_key, where="claim proposition")
        if self.mode not in W08_DISCOURSE_CLAIM_MODES:
            raise W08DiscourseError("claim mode is not registered")
        if self.evidence_state not in W08_DISCOURSE_EVIDENCE_STATES:
            raise W08DiscourseError("claim Evidence state is not registered")
        if self.lifecycle not in W08_DISCOURSE_LIFECYCLES:
            raise W08DiscourseError("claim lifecycle is not registered")
        _bit(self.current_projection_allowed, where="claim current projection")
        for name, value in (
            ("speaker", self.speaker_key),
            ("holder", self.holder_key),
            ("supersedes", self.supersedes_claim_key),
        ):
            if value:
                _key(value, where=f"claim {name}")
        _keys(self.evidence_keys, where="claim Evidence keys")
        if self.mode != "DIRECT" and self.current_projection_allowed:
            raise W08DiscourseError(
                "attribution/quotation/hypothesis cannot become current fact"
            )
        if self.mode in {"ATTRIBUTION", "QUOTATION"} and not (
            self.speaker_key or self.holder_key
        ):
            raise W08DiscourseError("reported claim requires speaker or holder")
        allowed = (
            self.mode == "DIRECT"
            and self.evidence_state == "SUPPORTED"
            and self.lifecycle == "ACTIVE"
        )
        if bool(self.current_projection_allowed) != allowed:
            raise W08DiscourseError("claim projection flag disagrees with typed state")
        if (self.lifecycle == "SUPERSEDED") != bool(self.supersedes_claim_key):
            raise W08DiscourseError("superseded claim must identify its predecessor")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "claim": list(self.claim_key),
            "proposition": list(self.proposition_key),
            "mode": self.mode,
            "evidence_state": self.evidence_state,
            "lifecycle": self.lifecycle,
            "current_projection_allowed": self.current_projection_allowed,
            "speaker": list(self.speaker_key),
            "holder": list(self.holder_key),
            "evidence": [list(item) for item in self.evidence_keys],
            "supersedes": list(self.supersedes_claim_key),
        })


@dataclass(frozen=True)
class W08GenerationFormCandidate:
    form_key: tuple[int, ...]
    form_kind: str
    antecedent_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _key(self.form_key, where="generation form key")
        _key(self.antecedent_key, where="generation antecedent key")
        if self.form_kind not in W08_GENERATION_REFERENCE_FORMS:
            raise W08DiscourseError("generation reference form is not registered")


@dataclass(frozen=True)
class W08DiscourseRequest:
    request_key: tuple[int, ...]
    scene: W08DiscourseScene
    claims: tuple[W08DiscourseClaim, ...]
    change_kind: str
    logical_clock: int
    changed_dependency_keys: tuple[tuple[int, ...], ...]
    reference_required: int
    reference_candidate_keys: tuple[tuple[int, ...], ...]
    clarification_candidate_key: tuple[int, ...] = ()
    generation_required: int = 0
    generation_form_candidates: tuple[W08GenerationFormCandidate, ...] = ()

    def __post_init__(self) -> None:
        _key(self.request_key, where="discourse request key")
        if not isinstance(self.scene, W08DiscourseScene):
            raise TypeError("discourse request scene type is invalid")
        if (
            not isinstance(self.claims, tuple)
            or not self.claims
            or any(not isinstance(item, W08DiscourseClaim) for item in self.claims)
        ):
            raise W08DiscourseError("discourse request requires typed claims")
        if len({item.claim_key for item in self.claims}) != len(self.claims):
            raise W08DiscourseError("discourse claims are duplicated")
        if self.change_kind not in W08_DISCOURSE_CHANGE_KINDS:
            raise W08DiscourseError("discourse change kind is not registered")
        if type(self.logical_clock) is not int or self.logical_clock < 0:
            raise W08DiscourseError("discourse logical clock is invalid")
        _keys(self.changed_dependency_keys, where="changed dependency keys")
        _bit(self.reference_required, where="reference_required")
        _keys(self.reference_candidate_keys, where="reference candidate keys")
        if not self.reference_required and self.reference_candidate_keys:
            raise W08DiscourseError("non-reference request cannot carry candidates")
        if self.clarification_candidate_key:
            _key(self.clarification_candidate_key, where="clarification candidate")
            if self.clarification_candidate_key not in self.reference_candidate_keys:
                raise W08DiscourseError("clarification selected an unknown reference")
        _bit(self.generation_required, where="generation_required")
        if (
            not isinstance(self.generation_form_candidates, tuple)
            or any(
                not isinstance(item, W08GenerationFormCandidate)
                for item in self.generation_form_candidates
            )
        ):
            raise TypeError("generation form candidates type is invalid")
        form_keys = tuple(item.form_key for item in self.generation_form_candidates)
        if len(set(form_keys)) != len(form_keys):
            raise W08DiscourseError("generation form candidates are duplicated")
        if bool(self.generation_form_candidates) != bool(self.generation_required):
            raise W08DiscourseError("generation requirement/forms disagree")
        if self.change_kind != "NEW_OBSERVATION" and not self.changed_dependency_keys:
            raise W08DiscourseError("discourse revision requires changed dependencies")
        if self.change_kind == "OPEN_QUESTION_CLOSE" and not self.scene.closed_question_keys:
            raise W08DiscourseError("QUD close requires a closed question")


@dataclass(frozen=True)
class W08CenterReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    center_key: tuple[int, ...]
    obligation_keys: tuple[tuple[int, ...], ...]
    ring_receipt_key: tuple[int, ...]
    expansion_ring_keys: tuple[tuple[int, ...], ...]
    stop_state: str = "RESOLVED"
    activation_authorizes_adoption: int = 0
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[0]:
            raise W08DiscourseError("typed center owner drifted")
        _key(self.request_key, where="center request key")
        _key(self.center_key, where="center key")
        _keys(self.obligation_keys, where="center obligations")
        _key(self.ring_receipt_key, where="center ring receipt")
        _keys(
            self.expansion_ring_keys,
            where="center expansion rings",
            allow_empty=False,
        )
        _stop(self.stop_state, where="center stop state")
        if self.activation_authorizes_adoption != 0 or self.host_learning_write_count != 0:
            raise W08DiscourseError("center owner wrote or authorized adoption")


@dataclass(frozen=True)
class W08EventReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    prior_event_keys: tuple[tuple[int, ...], ...]
    appended_event_key: tuple[int, ...]
    stop_state: str = "RESOLVED"
    append_only: int = 1
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[1]:
            raise W08DiscourseError("situation event owner drifted")
        _key(self.request_key, where="event request key")
        _keys(self.prior_event_keys, where="prior event keys")
        _key(self.appended_event_key, where="appended event key")
        if self.appended_event_key in self.prior_event_keys:
            raise W08DiscourseError("new event overwrote an old event")
        _stop(self.stop_state, where="event stop state")
        if self.append_only != 1 or self.host_learning_write_count != 0:
            raise W08DiscourseError("event log is not append-only")


@dataclass(frozen=True)
class W08LifecycleReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    candidate_keys: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    adopted_claim_keys: tuple[tuple[int, ...], ...]
    decision_keys: tuple[tuple[int, ...], ...]
    stop_state: str = "RESOLVED"
    prior_history_preserved: int = 1
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[2]:
            raise W08DiscourseError("candidate lifecycle owner drifted")
        _key(self.request_key, where="lifecycle request key")
        _keys(self.candidate_keys, where="lifecycle candidates", allow_empty=False)
        _keys(self.evidence_keys, where="lifecycle Evidence keys")
        _keys(self.adopted_claim_keys, where="adopted claim keys")
        _keys(self.decision_keys, where="decision keys")
        if not set(self.adopted_claim_keys) <= set(self.candidate_keys):
            raise W08DiscourseError("lifecycle adopted an unknown claim")
        _stop(self.stop_state, where="lifecycle stop state")
        if self.prior_history_preserved != 1 or self.host_learning_write_count != 0:
            raise W08DiscourseError("lifecycle history was rewritten")


@dataclass(frozen=True)
class W08ProjectionReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    active_proposition_keys: tuple[tuple[int, ...], ...]
    before_projection_ref: tuple[int, ...]
    after_projection_ref: tuple[int, ...]
    invalidated_keys: tuple[tuple[int, ...], ...]
    rebuilt_keys: tuple[tuple[int, ...], ...]
    unaffected_keys: tuple[tuple[int, ...], ...]
    stop_state: str = "RESOLVED"
    old_events_preserved: int = 1
    old_observations_preserved: int = 1
    old_evidence_preserved: int = 1
    old_decisions_preserved: int = 1
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[3]:
            raise W08DiscourseError("current projection owner drifted")
        _key(self.request_key, where="projection request key")
        _keys(self.active_proposition_keys, where="active propositions")
        _key(self.before_projection_ref, where="before projection ref")
        _key(self.after_projection_ref, where="after projection ref")
        for name, values in (
            ("invalidated", self.invalidated_keys),
            ("rebuilt", self.rebuilt_keys),
            ("unaffected", self.unaffected_keys),
        ):
            _keys(values, where=f"projection {name}")
        if self.invalidated_keys != self.rebuilt_keys:
            raise W08DiscourseError("projection invalidation/rebuild is not exact")
        if set(self.invalidated_keys) & set(self.unaffected_keys):
            raise W08DiscourseError("projection affected/unaffected sets overlap")
        _stop(self.stop_state, where="projection stop state")
        if (
            self.old_events_preserved,
            self.old_observations_preserved,
            self.old_evidence_preserved,
            self.old_decisions_preserved,
            self.host_learning_write_count,
        ) != (1, 1, 1, 1, 0):
            raise W08DiscourseError("projection revision rewrote append-only history")


@dataclass(frozen=True)
class W08AgendaReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    changed_dependency_keys: tuple[tuple[int, ...], ...]
    agenda_target_keys: tuple[tuple[int, ...], ...]
    stop_state: str = "RESOLVED"
    exact_dependency_match: int = 1
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[4]:
            raise W08DiscourseError("dependency agenda owner drifted")
        _key(self.request_key, where="agenda request key")
        _keys(self.changed_dependency_keys, where="agenda changed dependencies")
        _keys(self.agenda_target_keys, where="agenda targets")
        _stop(self.stop_state, where="agenda stop state")
        if self.exact_dependency_match != 1 or self.host_learning_write_count != 0:
            raise W08DiscourseError("agenda used a non-dependency heuristic")


@dataclass(frozen=True)
class W08ReferenceReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    candidate_keys: tuple[tuple[int, ...], ...]
    adopted_candidate_keys: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    decision_key: tuple[int, ...]
    stop_state: str
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[5]:
            raise W08DiscourseError("occurrence reference owner drifted")
        _key(self.request_key, where="reference request key")
        _keys(self.candidate_keys, where="reference candidates")
        _keys(self.adopted_candidate_keys, where="reference adopted candidates")
        _keys(self.evidence_keys, where="reference Evidence keys")
        _key(self.decision_key, where="reference decision key")
        if not set(self.adopted_candidate_keys) <= set(self.candidate_keys):
            raise W08DiscourseError("reference adopted an unknown candidate")
        if len(self.adopted_candidate_keys) > 1:
            raise W08DiscourseError("reference owner exposed multiple winners")
        _stop(self.stop_state, where="reference stop state")
        if len(self.adopted_candidate_keys) == 1:
            expected = "RESOLVED"
        elif self.candidate_keys:
            expected = "CLARIFY"
        else:
            expected = "UNKNOWN"
        if self.stop_state != expected:
            raise W08DiscourseError("reference stop state disagrees with adopted singleton")
        if self.host_learning_write_count != 0:
            raise W08DiscourseError("reference facade wrote host learning state")


@dataclass(frozen=True)
class W08GenerationReceipt:
    owner_key: str
    request_key: tuple[int, ...]
    selected_form_key: tuple[int, ...]
    selected_form_kind: str
    antecedent_key: tuple[int, ...]
    audience_recovered_key: tuple[int, ...]
    choice_key: tuple[int, ...]
    postcheck_key: tuple[int, ...]
    stop_state: str
    audience_recoverable: int
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[6]:
            raise W08DiscourseError("generation owner drifted")
        for name, value in (
            ("request", self.request_key),
            ("selected form", self.selected_form_key),
            ("antecedent", self.antecedent_key),
            ("audience recovered", self.audience_recovered_key),
            ("choice", self.choice_key),
            ("postcheck", self.postcheck_key),
        ):
            _key(value, where=f"generation {name} key")
        if self.selected_form_kind not in W08_GENERATION_REFERENCE_FORMS:
            raise W08DiscourseError("generation selected an unknown form")
        _stop(self.stop_state, where="generation stop state")
        _bit(self.audience_recoverable, where="audience recoverable")
        resolved = (
            self.audience_recoverable == 1
            and self.audience_recovered_key == self.antecedent_key
        )
        expected = "RESOLVED" if resolved else "GROUNDING_BLOCKED"
        if self.stop_state != expected:
            raise W08DiscourseError("generation stop state disagrees with audience postcheck")
        if self.host_learning_write_count != 0:
            raise W08DiscourseError("generation facade wrote host learning state")


@dataclass(frozen=True)
class W08DiscourseUse:
    owner_key: str
    consumer_key: str
    request_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    directional_choice_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_state: str
    outcome_key: tuple[int, ...]
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.owner_key != W08_DISCOURSE_OWNER_KEYS[7]:
            raise W08DiscourseError("U/R/G consumer owner drifted")
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08DiscourseError("discourse consumer is not registered")
        for name, value in (
            ("request", self.request_key),
            ("selected", self.selected_candidate_key),
            ("directional choice", self.directional_choice_key),
            ("use", self.use_key),
            ("outcome", self.outcome_key),
        ):
            _key(value, where=f"discourse Use {name}")
        _keys(self.evidence_keys, where="discourse Use Evidence")
        _stop(self.outcome_state, where="discourse Use outcome")
        if self.host_learning_write_count != 0:
            raise W08DiscourseError("discourse consumer wrote host learning state")


class W08CenterOwner(Protocol):
    owner_key: str

    def form(self, request: W08DiscourseRequest) -> W08CenterReceipt: ...


class W08EventOwner(Protocol):
    owner_key: str

    def append(self, request: W08DiscourseRequest, center: W08CenterReceipt) -> W08EventReceipt: ...


class W08LifecycleOwner(Protocol):
    owner_key: str

    def resolve(self, request: W08DiscourseRequest, event: W08EventReceipt) -> W08LifecycleReceipt: ...


class W08ProjectionOwner(Protocol):
    owner_key: str

    def project(
        self,
        request: W08DiscourseRequest,
        event: W08EventReceipt,
        lifecycle: W08LifecycleReceipt,
    ) -> W08ProjectionReceipt: ...


class W08AgendaOwner(Protocol):
    owner_key: str

    def plan(self, request: W08DiscourseRequest, projection: W08ProjectionReceipt) -> W08AgendaReceipt: ...


class W08ReferenceOwner(Protocol):
    owner_key: str

    def resolve(self, request: W08DiscourseRequest, projection: W08ProjectionReceipt) -> W08ReferenceReceipt: ...


class W08GenerationOwner(Protocol):
    owner_key: str

    def choose(
        self,
        request: W08DiscourseRequest,
        projection: W08ProjectionReceipt,
        reference: W08ReferenceReceipt | None,
    ) -> W08GenerationReceipt: ...


class W08ConsumerOwner(Protocol):
    owner_key: str

    def consume(
        self,
        request: W08DiscourseRequest,
        consumer_key: str,
        selected_candidate_key: tuple[int, ...],
        evidence_keys: tuple[tuple[int, ...], ...],
        outcome_state: str,
    ) -> W08DiscourseUse: ...


@dataclass(frozen=True)
class W08DiscourseOwners:
    center: W08CenterOwner
    events: W08EventOwner
    lifecycle: W08LifecycleOwner
    projection: W08ProjectionOwner
    agenda: W08AgendaOwner
    reference: W08ReferenceOwner
    generation: W08GenerationOwner
    consumers: W08ConsumerOwner

    def __post_init__(self) -> None:
        owners = (
            (self.center, "form"),
            (self.events, "append"),
            (self.lifecycle, "resolve"),
            (self.projection, "project"),
            (self.agenda, "plan"),
            (self.reference, "resolve"),
            (self.generation, "choose"),
            (self.consumers, "consume"),
        )
        if tuple(getattr(owner, "owner_key", None) for owner, _ in owners) != (
            W08_DISCOURSE_OWNER_KEYS
        ):
            raise W08DiscourseError("discourse owner inventory/order drifted")
        if any(not callable(getattr(owner, method, None)) for owner, method in owners):
            raise W08DiscourseError("discourse owner method is missing")


@dataclass(frozen=True)
class W08DiscourseAuditReceipt:
    request_key: tuple[int, ...]
    stop_state: str
    center: W08CenterReceipt
    event: W08EventReceipt | None
    lifecycle: W08LifecycleReceipt | None
    projection: W08ProjectionReceipt | None
    agenda: W08AgendaReceipt | None
    reference: W08ReferenceReceipt | None
    generation: W08GenerationReceipt | None
    uses: tuple[W08DiscourseUse, ...]
    owner_call_order: tuple[str, ...]
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        _key(self.request_key, where="discourse audit request")
        _stop(self.stop_state, where="discourse audit stop state")
        if not isinstance(self.center, W08CenterReceipt):
            raise TypeError("discourse audit center type is invalid")
        optional = (
            (self.event, W08EventReceipt),
            (self.lifecycle, W08LifecycleReceipt),
            (self.projection, W08ProjectionReceipt),
            (self.agenda, W08AgendaReceipt),
            (self.reference, W08ReferenceReceipt),
            (self.generation, W08GenerationReceipt),
        )
        if any(value is not None and not isinstance(value, kind) for value, kind in optional):
            raise TypeError("discourse audit contains an invalid owner receipt")
        if self.stop_state == "RESOLVED":
            if tuple(item.consumer_key for item in self.uses) != W08_CONSUMER_KEYS:
                raise W08DiscourseError("resolved discourse audit lacks exact U/R/G Use")
        elif self.uses:
            raise W08DiscourseError("stopped discourse request must not leak partial Use")
        if len(set(self.owner_call_order)) != len(self.owner_call_order):
            raise W08DiscourseError("discourse owner was called twice")
        if self.host_learning_write_count != 0:
            raise W08DiscourseError("discourse facade wrote host learning state")


@dataclass(frozen=True)
class W08DiscourseAblationReport:
    affected_dimensions: tuple[str, ...]
    unaffected_dimensions: tuple[str, ...]


def assess_w08_discourse_ablation(
    *,
    full_dimension_outcomes: dict[str, str],
    ablated_dimension_outcomes: dict[str, str],
) -> W08DiscourseAblationReport:
    """要求删除 discourse facade/Evidence 只改变篇章 bearing。"""
    expected = set(W08_DIMENSION_KEYS)
    if set(full_dimension_outcomes) != expected or set(ablated_dimension_outcomes) != expected:
        raise W08DiscourseError("discourse ablation dimension inventory drifted")
    discourse = "W-08-DISCOURSE"
    if full_dimension_outcomes[discourse] != "PASS":
        raise W08DiscourseError("full discourse assessment did not pass")
    changed = tuple(
        key for key in W08_DIMENSION_KEYS
        if full_dimension_outcomes[key] != ablated_dimension_outcomes[key]
    )
    if changed != (discourse,) or ablated_dimension_outcomes[discourse] == "PASS":
        raise W08DiscourseError("discourse ablation is not orthogonal")
    return W08DiscourseAblationReport(
        changed,
        tuple(key for key in W08_DIMENSION_KEYS if key != discourse),
    )


__all__ = [
    "W08AgendaReceipt",
    "W08CenterReceipt",
    "W08DiscourseAuditReceipt",
    "W08DiscourseAblationReport",
    "W08DiscourseClaim",
    "W08DiscourseError",
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
