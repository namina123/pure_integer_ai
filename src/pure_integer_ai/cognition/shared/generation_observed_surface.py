"""Source-local Memory span realization alongside trained Core R-01 routes."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import AliasResolutionProposal
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT, OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION, OBJECT_PROPOSITION, OBJECT_ROLE, OBJECT_SPAN,
    ObjectIdentity, SourceRef, representation_identity, span_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_OBJECT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS, MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity


DELIVERED_GRAPH_SURFACE_ROLE = 91572


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    if type(key) is not tuple or any(type(value) is not int for value in key):
        raise ValueError("observed surface keys must contain strict integers")
    return len(key), *key


def _evidence(keys: tuple[tuple[int, ...], ...]) -> None:
    if type(keys) is not tuple or not keys or keys != tuple(sorted(set(keys))):
        raise ValueError("observed surface requires complete canonical evidence")
    for key in keys:
        if not key:
            raise ValueError("observed surface evidence cannot be empty")
        _pack(key)


def delivered_graph_surface_role(
        category: int,
        proof_ref: ObjectIdentity,
        ) -> tuple[int, ...]:
    """Name one graph category projected from an acknowledged delivery."""
    if type(category) is not int or category not in range(1, 6):
        raise ValueError("delivered graph surface category is not registered")
    if not isinstance(proof_ref, ObjectIdentity):
        raise TypeError("delivered graph surface proof must be an identity")
    return DELIVERED_GRAPH_SURFACE_ROLE, category, *_pack(proof_ref.stable_key())


def delivered_graph_surface_category(
        role_key: tuple[int, ...],
        ) -> int | None:
    """Decode only the explicit delivered-slot role; ordinary roles return None."""
    if (type(role_key) is not tuple or len(role_key) < 4
            or role_key[0] != DELIVERED_GRAPH_SURFACE_ROLE):
        return None
    category = role_key[1]
    size = role_key[2]
    proof_key = role_key[3:]
    if (type(category) is not int or category not in range(1, 6)
            or type(size) is not int or size != len(proof_key)):
        raise ValueError("delivered graph surface role is truncated")
    ObjectIdentity.from_stable_key(proof_key)
    return category


@dataclass(frozen=True, slots=True)
class ObservedGenerationSpan:
    """An observed role name, not an asserted entity or proposition."""

    source: SourceRef
    scope: ScopeIdentity
    observation: MemoryObjectRef
    hypothesis: MemoryObjectRef
    role: ObjectIdentity
    ordinal: int
    start: int
    end: int
    values: tuple[int, ...]
    context_key: tuple[int, ...]
    evidence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef) or not isinstance(self.scope, ScopeIdentity):
            raise TypeError("observed generation source/scope types differ")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("observed generation scope differs from its source")
        for ref, kind in ((self.observation, MEMORY_OBJECT_OBSERVATION),
                          (self.hypothesis, MEMORY_OBJECT_HYPOTHESIS)):
            if (not isinstance(ref, MemoryObjectRef) or ref.object_kind != kind
                    or ref.owner != self.source.owner or ref.versions != self.source.versions):
                raise ValueError("observed generation Memory reference differs")
        if self.observation.memory_space != self.hypothesis.memory_space:
            raise ValueError("observed generation Memory spaces differ")
        if not isinstance(self.role, ObjectIdentity) or self.role.object_kind != OBJECT_ROLE:
            raise ValueError("observed generation requires a trained Role")
        if any(type(v) is not int or v < 0 for v in (self.ordinal, self.start, self.end)):
            raise ValueError("observed generation positions must be nonnegative integers")
        if not self.values or self.end - self.start != len(self.values):
            raise ValueError("observed generation values do not cover the role span")
        if not self.context_key:
            raise ValueError("observed generation requires its full query binding")
        _pack(self.values)
        _pack(self.context_key)
        _evidence(self.evidence)

    @property
    def origin(self) -> ObjectIdentity:
        return span_identity(self.source, members=((self.start, self.end),))

    def stable_key(self) -> tuple[int, ...]:
        return (
            91520, *_pack(self.source.stable_key()), *_pack(self.scope.stable_key()),
            *_pack(self.observation.stable_key()), *_pack(self.hypothesis.stable_key()),
            *_pack(self.role.stable_key()), self.ordinal, self.start, self.end,
            *_pack(self.values), *_pack(self.context_key), len(self.evidence),
            *(v for key in self.evidence for v in _pack(key)),
        )


@dataclass(frozen=True, slots=True)
class ObservedSpanRealizationRule:
    """A trained role reader and its actual source-span/R-01 language evidence."""

    branch: ObjectIdentity
    role: ObjectIdentity
    ordinal: int
    instruction: ObjectIdentity
    example_span: ObjectIdentity
    example_route: AliasResolutionProposal
    connector_key: tuple[int, ...]
    evidence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for value, kind in ((self.branch, OBJECT_LANGUAGE_BRANCH), (self.role, OBJECT_ROLE),
                            (self.instruction, OBJECT_MINIMAL_INSTRUCTION),
                            (self.example_span, OBJECT_SPAN)):
            if not isinstance(value, ObjectIdentity) or value.object_kind != kind:
                raise ValueError("observed surface rule identity kind differs")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("observed surface rule ordinal must be nonnegative")
        if (not isinstance(self.example_route, AliasResolutionProposal)
                or self.example_route.result.branch != self.branch
                or self.example_route.result.selected is None):
            raise ValueError("observed surface rule requires a unique trained R-01 route")
        if not self.connector_key:
            raise ValueError("observed surface rule requires the complete trained connector")
        _pack(self.connector_key)
        _evidence(self.evidence)

    def stable_key(self) -> tuple[int, ...]:
        return (
            91521, *_pack(self.branch.stable_key()), *_pack(self.role.stable_key()), self.ordinal,
            *_pack(self.instruction.stable_key()), *_pack(self.example_span.stable_key()),
            *_pack(self.example_route.stable_key()), *_pack(self.connector_key), len(self.evidence),
            *(v for key in self.evidence for v in _pack(key)),
        )


@dataclass(frozen=True, slots=True)
class ObservedSurfaceProposal:
    """A Memory realization and all its explicitly applicable trained role rules."""

    value: ObservedGenerationSpan
    rules: tuple[ObservedSpanRealizationRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.value, ObservedGenerationSpan):
            raise TypeError("observed surface requires a source-local role value")
        if (type(self.rules) is not tuple or not self.rules
                or any(not isinstance(item, ObservedSpanRealizationRule) for item in self.rules)):
            raise TypeError("observed surface requires trained realization rules")
        if any((item.role, item.ordinal) != (self.value.role, self.value.ordinal)
               for item in self.rules):
            raise ValueError("observed surface rule does not bind the input role")
        if len({item.branch for item in self.rules}) != 1:
            raise ValueError("observed surface rule branches compete")
        if len({item.stable_key() for item in self.rules}) != len(self.rules):
            raise ValueError("observed surface rules cannot be repeated")
        object.__setattr__(self, "rules", tuple(sorted(self.rules, key=lambda item: item.stable_key())))
        self.representation

    @property
    def branch(self) -> ObjectIdentity:
        return self.rules[0].branch

    @property
    def origin(self) -> ObjectIdentity:
        return self.value.origin

    @property
    def source(self) -> SourceRef:
        return self.value.source

    @property
    def scope(self) -> ScopeIdentity:
        return self.value.scope

    @property
    def representation(self) -> ObjectIdentity:
        # The renderer depends on G-03 contracts; defer this schema read until
        # the module graph is loaded rather than duplicating its decoder.
        from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
        families = {representation_parts(item.example_route.result.selected.value)[0]
                    for item in self.rules}
        if len(families) != 1:
            raise ValueError("observed surface rules have ambiguous representation families")
        return representation_identity(next(iter(families)), self.value.values,
                                       owner=self.value.source.owner,
                                       versions=self.value.source.versions)

    def stable_key(self) -> tuple[int, ...]:
        return (91522, *_pack(self.value.stable_key()), len(self.rules),
                *(v for rule in self.rules for v in _pack(rule.stable_key())),
                *_pack(self.representation.stable_key()))

    def use_key(self, use_key: tuple[int, ...], source: SourceRef,
                scope: ScopeIdentity) -> tuple[int, ...]:
        """A read-only adoption trace; never a fabricated Core relation Use."""
        if source != self.value.source or scope != self.value.scope:
            raise ValueError("observed surface adoption source/scope differs")
        if not use_key:
            raise ValueError("observed surface adoption requires an exact use key")
        return (91523, *_pack(use_key), *_pack(source.stable_key()),
                *_pack(scope.stable_key()), *_pack(self.stable_key()))

    @property
    def value_key(self) -> tuple[int, ...]:
        """Return the source-local value key consumed by G-03."""
        return self.value.stable_key()


@dataclass(frozen=True, slots=True)
class ObservedGraphSurfaceProposal:
    """A graph object expressed by its unique exact span in this query.

    This is not an alias fallback.  The target identity, trained input
    projection, current Observation, QueryState proof and integer source span
    are all required before G-03 may emit the span as one dynamic slot.
    """

    target: ObjectIdentity
    branch: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    observation: MemoryObjectRef
    input_source_ref: tuple[int, ...]
    start: int
    end: int
    values: tuple[int, ...]
    family: tuple[int, ...]
    projection_keys: tuple[tuple[int, ...], ...]
    evidence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.target, ObjectIdentity)
                or self.target.object_kind not in {
                    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT,
                    OBJECT_PROPOSITION,
                }):
            raise ValueError("observed graph surface target kind is not semantic")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise ValueError("observed graph surface requires a language branch")
        if not isinstance(self.source, SourceRef) or not isinstance(
                self.scope, ScopeIdentity):
            raise TypeError("observed graph surface source/scope types differ")
        if (self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("observed graph surface scope differs from source")
        if (not isinstance(self.observation, MemoryObjectRef)
                or self.observation.object_kind != MEMORY_OBJECT_OBSERVATION
                or self.observation.owner != self.source.owner
                or self.observation.versions != self.source.versions):
            raise ValueError("observed graph surface Observation differs from source")
        for key in (self.input_source_ref, self.values, self.family):
            if type(key) is not tuple or not key or any(
                    type(value) is not int or value < 0 for value in key):
                raise ValueError("observed graph surface keys must be non-empty integers")
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start
                or self.end - self.start != len(self.values)):
            raise ValueError("observed graph surface span does not cover its values")
        _evidence(self.projection_keys)
        _evidence(self.evidence)
        self.representation

    @property
    def origin(self) -> ObjectIdentity:
        return self.target

    @property
    def representation(self) -> ObjectIdentity:
        return representation_identity(
            self.family,
            self.values,
            owner=self.source.owner,
            versions=self.source.versions,
        )

    @property
    def value_key(self) -> tuple[int, ...]:
        return self.stable_key()

    def stable_key(self) -> tuple[int, ...]:
        return (
            91570,
            *_pack(self.target.stable_key()),
            *_pack(self.branch.stable_key()),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.observation.stable_key()),
            *_pack(self.input_source_ref),
            self.start,
            self.end,
            *_pack(self.values),
            *_pack(self.family),
            len(self.projection_keys),
            *(value for key in self.projection_keys for value in _pack(key)),
            len(self.evidence),
            *(value for key in self.evidence for value in _pack(key)),
        )

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "ObservedGraphSurfaceProposal":
        """Restore the complete integer proposal kept by a delivery proof."""
        if (type(key) is not tuple or not key or key[0] != 91570
                or any(type(value) is not int or value < 0 for value in key)):
            raise ValueError("observed graph surface key is invalid")
        cursor = 1

        def take() -> tuple[int, ...]:
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError("observed graph surface key is truncated")
            size = key[cursor]
            end = cursor + 1 + size
            if end > len(key):
                raise ValueError("observed graph surface field is truncated")
            value = key[cursor + 1:end]
            cursor = end
            return value

        target = ObjectIdentity.from_stable_key(take())
        branch = ObjectIdentity.from_stable_key(take())
        source = SourceRef.from_stable_key(take())
        scope = ScopeIdentity.from_stable_key(take())
        observation = MemoryObjectRef.from_stable_key(take())
        input_source_ref = take()
        if cursor + 2 > len(key):
            raise ValueError("observed graph surface span is truncated")
        start, end = key[cursor:cursor + 2]
        cursor += 2
        values = take()
        family = take()

        def take_many() -> tuple[tuple[int, ...], ...]:
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError("observed graph surface list is truncated")
            count = key[cursor]
            cursor += 1
            return tuple(take() for _ in range(count))

        projection_keys = take_many()
        evidence = take_many()
        if cursor != len(key):
            raise ValueError("observed graph surface key has trailing integers")
        return cls(
            target,
            branch,
            source,
            scope,
            observation,
            input_source_ref,
            start,
            end,
            values,
            family,
            projection_keys,
            evidence,
        )

    def use_key(self, use_key: tuple[int, ...], source: SourceRef,
                scope: ScopeIdentity) -> tuple[int, ...]:
        if source != self.source or scope != self.scope or not use_key:
            raise ValueError("observed graph surface adoption context differs")
        return (
            91571,
            *_pack(use_key),
            *_pack(source.stable_key()),
            *_pack(scope.stable_key()),
            *_pack(self.stable_key()),
        )


@dataclass(frozen=True, slots=True)
class DeliveredGraphSurface:
    """Bounded graph/span provenance retained after an acknowledged output.

    Observation and QueryState evidence remain in the surrounding Memory
    proof.  Keeping them out of this member prevents recursive identity growth
    while preserving every semantic and realization coordinate.
    """

    target: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    input_source_ref: tuple[int, ...]
    start: int
    end: int
    values: tuple[int, ...]
    family: tuple[int, ...]
    projection_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.target, ObjectIdentity)
                or self.target.object_kind not in {
                    OBJECT_CONCEPT, OBJECT_ENTITY, OBJECT_EVENT,
                    OBJECT_PROPOSITION,
                }
                or not isinstance(self.source, SourceRef)
                or not isinstance(self.scope, ScopeIdentity)
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise ValueError("delivered graph surface identity/source differs")
        for key in (self.input_source_ref, self.values, self.family):
            if type(key) is not tuple or not key or any(
                    type(value) is not int or value < 0 for value in key):
                raise ValueError(
                    "delivered graph surface keys must be non-empty integers")
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start
                or self.end - self.start != len(self.values)):
            raise ValueError("delivered graph surface span differs")
        _evidence(self.projection_keys)

    @classmethod
    def from_proposal(
            cls,
            proposal: ObservedGraphSurfaceProposal,
            ) -> "DeliveredGraphSurface":
        if not isinstance(proposal, ObservedGraphSurfaceProposal):
            raise TypeError("delivered graph surface requires an observed proposal")
        return cls(
            proposal.target,
            proposal.source,
            proposal.scope,
            proposal.input_source_ref,
            proposal.start,
            proposal.end,
            proposal.values,
            proposal.family,
            proposal.projection_keys,
        )

    def stable_key(self) -> tuple[int, ...]:
        return (
            91574,
            *_pack(self.target.stable_key()),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            *_pack(self.input_source_ref),
            self.start,
            self.end,
            *_pack(self.values),
            *_pack(self.family),
            len(self.projection_keys),
            *(value for key in self.projection_keys for value in _pack(key)),
        )

    @classmethod
    def from_stable_key(
            cls,
            key: tuple[int, ...],
            ) -> "DeliveredGraphSurface":
        if (type(key) is not tuple or not key or key[0] != 91574
                or any(type(value) is not int or value < 0 for value in key)):
            raise ValueError("delivered graph surface key is invalid")
        cursor = 1

        def take() -> tuple[int, ...]:
            nonlocal cursor
            if cursor >= len(key):
                raise ValueError("delivered graph surface key is truncated")
            size = key[cursor]
            end = cursor + 1 + size
            if end > len(key):
                raise ValueError("delivered graph surface field is truncated")
            value = key[cursor + 1:end]
            cursor = end
            return value

        target = ObjectIdentity.from_stable_key(take())
        source = SourceRef.from_stable_key(take())
        scope = ScopeIdentity.from_stable_key(take())
        input_source_ref = take()
        if cursor + 2 > len(key):
            raise ValueError("delivered graph surface span is truncated")
        start, end = key[cursor:cursor + 2]
        cursor += 2
        values = take()
        family = take()
        if cursor >= len(key):
            raise ValueError("delivered graph surface projections are truncated")
        count = key[cursor]
        cursor += 1
        projection_keys = tuple(take() for _ in range(count))
        if cursor != len(key):
            raise ValueError("delivered graph surface key has trailing integers")
        return cls(
            target,
            source,
            scope,
            input_source_ref,
            start,
            end,
            values,
            family,
            projection_keys,
        )


__all__ = [
    "DELIVERED_GRAPH_SURFACE_ROLE",
    "DeliveredGraphSurface",
    "ObservedGenerationSpan",
    "ObservedGraphSurfaceProposal",
    "ObservedSpanRealizationRule",
    "ObservedSurfaceProposal",
    "delivered_graph_surface_category",
    "delivered_graph_surface_role",
]
