"""Build non-factual unknown-input ResponsePlan objects from integer course data.

The course contains only project-owned, licensed integer surface variants.  A
variant is selected from the current QueryState and generic Memory candidate
identities; no input text, successor sentence, vocabulary, or character-near
lookup participates in the decision.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryObjectRef
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_PENDING,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    QueryState,
)
from pure_integer_ai.cognition.shared.response_plan import (
    ResponsePlan,
    ResponseRealization,
    ResponseSlot,
    response_act_identity,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_GENERIC_CANDIDATE_VERSION,
    frame_key,
)


UNKNOWN_RESPONSE_COURSE_VERSION = 91560
UNKNOWN_RESPONSE_COURSE_SOURCE = (91560, 1, 1)
GENERIC_RESPONSE_CONDITION_VERSION = 91561
STRUCTURAL_RESPONSE_COURSE_VERSION = 91562
RESPONSE_VARIANT_IDENTITY_VERSION = 91570
RESPONSE_PART_LITERAL = 1
RESPONSE_PART_GRAPH_ROLE = 2
RESPONSE_FEATURE_UNKNOWN = 1
RESPONSE_FEATURE_HOT_CONTEXT = 2
RESPONSE_FEATURE_COLD_CONTEXT = 4
RESPONSE_FEATURE_TOPIC = 8
RESPONSE_FEATURE_TOPIC_GRAPH = 16
RESPONSE_FEATURE_REFERENCE_RESOLVED = 32
RESPONSE_FEATURE_REFERENCE_OPEN = 64
RESPONSE_FEATURE_MULTI_HYPOTHESIS = 128
RESPONSE_FEATURE_CONFLICT = 256
RESPONSE_FEATURE_DISCOURSE_RELATION = 512
RESPONSE_FEATURE_RELATION_CHAIN = 1024
RESPONSE_FEATURE_QUD = 2048
RESPONSE_FEATURE_REVISION = 4096
RESPONSE_FEATURE_RELATION_CONFLICT = 8192
RESPONSE_FEATURE_EVENT_TIME = 16384
# Structural input conditions.  These bits are set only from trained graph
# candidate identities in the shared QueryState; they never inspect text
# similarity or a language vocabulary.
RESPONSE_FEATURE_ENTITY = 32768
RESPONSE_FEATURE_EVENT = 65536
RESPONSE_FEATURE_PROPERTY = 131072

@dataclass(frozen=True, slots=True)
class UnknownResponseSegment:
    """One graph-materialized response segment.

    The course owns integer units only; this view gives the existing response
    planner explicit role/atom/order material instead of treating a whole
    sentence as one literal surface.
    """

    ordinal: int
    units: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("unknown response segment ordinal must be positive")
        _strict_units(self.units)

    def stable_key(self) -> tuple[int, ...]:
        return frame_key(UNKNOWN_RESPONSE_COURSE_VERSION, (7,), (self.ordinal,), self.units)


@dataclass(frozen=True, slots=True)
class UnknownResponsePart:
    """One course-owned literal chunk or one graph-derived filler slot."""

    kind: int
    units: tuple[int, ...] = ()
    category: int = 0
    ordinal: int = 0

    def __post_init__(self) -> None:
        if self.kind == RESPONSE_PART_LITERAL:
            _strict_units(self.units)
            if self.category != 0 or self.ordinal != 0:
                raise ValueError("literal response part must not declare a graph role")
        elif self.kind == RESPONSE_PART_GRAPH_ROLE:
            if (self.units or type(self.category) is not int or self.category <= 0
                    or type(self.ordinal) is not int or self.ordinal <= 0):
                raise ValueError("graph response part requires category and ordinal")
        else:
            raise ValueError("response part kind is not registered")

    def stable_key(self) -> tuple[int, ...]:
        if self.kind == RESPONSE_PART_LITERAL:
            return frame_key(
                STRUCTURAL_RESPONSE_COURSE_VERSION,
                (RESPONSE_PART_LITERAL,),
                self.units,
            )
        return frame_key(
            STRUCTURAL_RESPONSE_COURSE_VERSION,
            (RESPONSE_PART_GRAPH_ROLE, self.category, self.ordinal),
        )


def _strict_units(units: tuple[int, ...]) -> tuple[int, ...]:
    if (type(units) is not tuple or not units
            or any(type(value) is not int
                   or value < 0 or value > 0x10FFFF
                   or 0xD800 <= value <= 0xDFFF
                   for value in units)):
        raise ValueError("unknown response course units must be Unicode scalar integers")
    return units


def response_condition_matches(condition: tuple[int, ...], feature_mask: int) -> bool:
    """Match a licensed integer response condition against QueryState features."""
    if condition == ():
        return True
    if (type(condition) is not tuple or len(condition) != 4
            or condition[0] != GENERIC_RESPONSE_CONDITION_VERSION
            or any(type(value) is not int or value < 0 for value in condition)
            or type(feature_mask) is not int or feature_mask < 0):
        raise ValueError("response condition identity is invalid")
    required, forbidden = condition[1], condition[2]
    if required & forbidden:
        raise ValueError("response condition required/forbidden masks overlap")
    return (feature_mask & required) == required and not (feature_mask & forbidden)


def response_condition_priority(condition: tuple[int, ...], feature_mask: int) -> int:
    """Return a deterministic frontier/style priority for a matching condition."""
    if not response_condition_matches(condition, feature_mask):
        return 0
    if not condition:
        return 1
    required, forbidden, _style = condition[1:]
    # Specificity dominates the baseline; style is only a deterministic tie
    # breaker among equally specific graph conditions.
    return 100 + (required | forbidden).bit_count() * 100


@dataclass(frozen=True, slots=True)
class UnknownResponseCourse:
    """A licensed set of structural unknown-response realizations."""

    course_key: tuple[int, ...]
    variants: tuple[tuple[int, ...], ...]
    source_key: tuple[int, ...] = UNKNOWN_RESPONSE_COURSE_SOURCE
    conditions: tuple[tuple[int, ...], ...] = ()
    parts: tuple[tuple[UnknownResponsePart, ...], ...] = ()

    def __post_init__(self) -> None:
        if (type(self.course_key) is not tuple or not self.course_key
                or any(type(value) is not int or value < 0 for value in self.course_key)):
            raise ValueError("unknown response course key must be integer-only")
        if (type(self.variants) is not tuple or not self.variants
                or any(type(item) is not tuple for item in self.variants)):
            raise ValueError("unknown response course variants are incomplete")
        for item in self.variants:
            _strict_units(item)
        if self.conditions:
            if (type(self.conditions) is not tuple
                    or len(self.conditions) != len(self.variants)):
                raise ValueError("unknown response course conditions must align with variants")
            for condition in self.conditions:
                if (type(condition) is not tuple
                        or any(type(value) is not int or value < 0 for value in condition)):
                    raise ValueError("unknown response course conditions must be integer-only")
                if condition:
                    response_condition_matches(condition, 0)
        if (type(self.source_key) is not tuple or not self.source_key
                or any(type(value) is not int or value < 0 for value in self.source_key)):
            raise ValueError("unknown response source key must be integer-only")
        if self.parts:
            if (type(self.parts) is not tuple
                    or len(self.parts) != len(self.variants)
                    or any(type(items) is not tuple or not items
                           for items in self.parts)
                    or any(not isinstance(part, UnknownResponsePart)
                           for items in self.parts for part in items)):
                raise ValueError("structural response parts must align with variants")
            for items in self.parts:
                roles = tuple(
                    (part.category, part.ordinal)
                    for part in items
                    if part.kind == RESPONSE_PART_GRAPH_ROLE
                )
                if not roles or len(set(roles)) != len(roles):
                    raise ValueError(
                        "structural response variant requires unique graph roles")
                if not any(part.kind == RESPONSE_PART_LITERAL for part in items):
                    raise ValueError(
                        "structural response variant requires a licensed literal part")

    def stable_key(self) -> tuple[int, ...]:
        if self.parts:
            return frame_key(
                STRUCTURAL_RESPONSE_COURSE_VERSION,
                self.course_key,
                self.source_key,
                *(frame_key(1, item) for item in self.variants),
                *(frame_key(2, item) for item in self.conditions),
                *(frame_key(
                    3, *(part.stable_key() for part in items))
                  for items in self.parts),
            )
        if not self.conditions:
            return frame_key(
                UNKNOWN_RESPONSE_COURSE_VERSION,
                self.course_key,
                self.source_key,
                *(frame_key(1, item) for item in self.variants),
            )
        return frame_key(
            UNKNOWN_RESPONSE_COURSE_VERSION,
            self.course_key,
            self.source_key,
            *(frame_key(1, item) for item in self.variants),
            *(frame_key(2, item) for item in self.conditions),
        )

    def condition(self, variant: int) -> tuple[int, ...]:
        """Return the integer graph-state condition for one variant."""
        if type(variant) is not int or not 1 <= variant <= len(self.variants):
            raise ValueError("unknown response variant ordinal is out of range")
        if not self.conditions:
            return ()
        return self.conditions[variant - 1]

    def select(self, query_key: tuple[int, ...], candidates: tuple[tuple[int, ...], ...]) -> tuple[int, tuple[int, ...]]:
        """Select one licensed variant using only integer graph identities."""
        if (type(query_key) is not tuple or not query_key
                or any(type(value) is not int or value < 0 for value in query_key)):
            raise ValueError("unknown response query key must be integer-only")
        if (type(candidates) is not tuple or not candidates
                or any(type(item) is not tuple or not item for item in candidates)):
            raise ValueError("unknown response candidates must be non-empty integer keys")
        seed = sum(query_key) + sum(sum(item) for item in candidates)
        index = seed % len(self.variants)
        return index, self.variants[index]

    def parts_for(self, variant: int) -> tuple[UnknownResponsePart, ...]:
        """Return explicit structural parts, preserving legacy atom courses."""
        if type(variant) is not int or not 1 <= variant <= len(self.variants):
            raise ValueError("unknown response variant ordinal is out of range")
        if self.parts:
            return self.parts[variant - 1]
        return tuple(
            UnknownResponsePart(RESPONSE_PART_LITERAL, (value,))
            for value in self.variants[variant - 1]
        )

    def semantic_variant_key(self, variant: int) -> tuple[int, ...]:
        """Return a package-independent identity for duplicate prevention."""
        parts = self.parts_for(variant)
        return frame_key(
            RESPONSE_VARIANT_IDENTITY_VERSION,
            self.condition(variant),
            *(part.stable_key() for part in parts),
        )

    @staticmethod
    def segments(units: tuple[int, ...]) -> tuple[UnknownResponseSegment, ...]:
        """Materialize ordered integer atoms from a selected course variant.

        The runtime does not classify punctuation, words or language-specific
        boundaries.  Each course-provided representation unit is a graph atom
        and its ordinal is the only linearization relation.
        """
        _strict_units(units)
        return tuple(UnknownResponseSegment(index, (value,))
                     for index, value in enumerate(units, 1))


def course_from_integer_payload(payload: bytes) -> UnknownResponseCourse:
    """Decode the external integer course without interpreting surface text."""
    if type(payload) is not bytes or not payload:
        raise ValueError("unknown response course payload must be bytes")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unknown response course is not ASCII integer JSON") from exc
    if (type(value) is not list or len(value) != 3
            or type(value[0]) is not int
            or value[0] not in {
                UNKNOWN_RESPONSE_COURSE_VERSION,
                GENERIC_RESPONSE_CONDITION_VERSION,
                STRUCTURAL_RESPONSE_COURSE_VERSION,
            }
            or value[1] not in {1, 2, 3}
            or type(value[2]) is not list or not value[2]):
        raise ValueError("unknown response course envelope is invalid")
    variants = []
    seen = set()
    structured = (
        value[0] == STRUCTURAL_RESPONSE_COURSE_VERSION or value[1] == 3)
    conditional = structured or value[0] == GENERIC_RESPONSE_CONDITION_VERSION or value[1] == 2
    for row in value[2]:
        if (type(row) is not list or len(row) not in ({2, 3} if conditional else {2})
                or type(row[0]) is not int or row[0] <= 0
                or type(row[1]) is not list):
            raise ValueError("unknown response course variant is invalid")
        if structured:
            if len(row) != 3 or type(row[2]) is not list or not row[2]:
                raise ValueError("structural response course parts are invalid")
            if any(type(v) is not int or v < 0 for v in row[1]):
                raise ValueError("unknown response course condition is invalid")
            condition = tuple(row[1])
            parts = []
            literal_units = []
            for part in row[2]:
                if (type(part) is not list or not part
                        or type(part[0]) is not int):
                    raise ValueError("structural response part is invalid")
                if part[0] == RESPONSE_PART_LITERAL:
                    if len(part) != 2 or type(part[1]) is not list:
                        raise ValueError("literal response part is invalid")
                    item = UnknownResponsePart(
                        RESPONSE_PART_LITERAL, _strict_units(tuple(part[1])))
                    literal_units.extend(item.units)
                elif part[0] == RESPONSE_PART_GRAPH_ROLE:
                    if len(part) != 3:
                        raise ValueError("graph response part is invalid")
                    item = UnknownResponsePart(
                        RESPONSE_PART_GRAPH_ROLE,
                        category=part[1],
                        ordinal=part[2],
                    )
                else:
                    raise ValueError("structural response part kind is unknown")
                parts.append(item)
            units = _strict_units(tuple(literal_units))
        elif conditional:
            if type(row[1]) is not list or any(type(v) is not int or v < 0 for v in row[1]):
                raise ValueError("unknown response course condition is invalid")
            condition = tuple(row[1])
            units = _strict_units(tuple(row[2]))
            parts = ()
        else:
            condition = ()
            units = _strict_units(tuple(row[1]))
            parts = ()
        if row[0] in seen:
            raise ValueError("unknown response course variant ordinal is duplicated")
        seen.add(row[0])
        variants.append((row[0], units, condition, tuple(parts)))
    variants.sort(key=lambda item: item[0])
    if tuple(item[0] for item in variants) != tuple(range(1, len(variants) + 1)):
        raise ValueError("unknown response course variant ordinals are not contiguous")
    return UnknownResponseCourse(
        (value[0], 2, len(variants)),
        tuple(item[1] for item in variants),
        conditions=tuple(item[2] for item in variants) if conditional else (),
        parts=tuple(item[3] for item in variants) if structured else (),
    )


def load_integer_course(course_path: str | Path,
                        manifest_path: str | Path) -> UnknownResponseCourse:
    """Load a licensed external course and verify its content commitment."""
    course = Path(course_path).resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)
    payload = course.read_bytes()
    metadata = json.loads(manifest.read_text(encoding="ascii"))
    if (metadata.get("format") not in {
            "UNKNOWN_RESPONSE_COURSE_LEDGER_V1",
            "CONDITIONAL_RESPONSE_COURSE_LEDGER_V1",
            "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
            }
            or metadata.get("free_dialogue_claim") != 0
            or metadata.get("course_sha256") != hashlib.sha256(payload).hexdigest()
            or not metadata.get("license_ids") or not metadata.get("rights_basis")):
        raise ValueError("unknown response course manifest is not licensed or committed")
    return course_from_integer_payload(payload)


def build_unknown_response_plan(
        state: QueryState,
        hypotheses: tuple[object, ...],
        course: UnknownResponseCourse,
        ) -> tuple[ResponsePlan, tuple[int, ...], int]:
    """Build a non-claim ResponsePlan from generic Memory evidence.

    The result remains an UNKNOWN response act.  It cannot be interpreted as
    a Core proposition because claim_refs is empty and every selected Memory
    evidence entry must have UNKNOWN polarity.
    """
    if not isinstance(state, QueryState) or not isinstance(course, UnknownResponseCourse):
        raise TypeError("unknown response plan requires QueryState and course")
    if set(state.active_spaces) != {SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}:
        raise ValueError("unknown response requires all three active graph spaces")
    if any(root.status == ROOT_PENDING for root in state.roots):
        raise ValueError("unknown response requires a completed shared frontier")
    generic = tuple(sorted(
        (item for item in hypotheses
         if (getattr(item, "candidate_key", ())
             and item.candidate_key[0] == MEMORY_GENERIC_CANDIDATE_VERSION)),
        key=lambda item: item.hypothesis_key,
    ))
    if not generic:
        raise ValueError("unknown response requires generic Memory hypotheses")
    if any(type(item.hypothesis_key) is not tuple or not item.hypothesis_key
           for item in generic):
        raise ValueError("unknown response hypothesis identity is incomplete")
    evidence = tuple(sorted(
        (item for item in state.evidence
         if item.space == SPACE_MEMORY
         and item.hypothesis_key in {candidate.hypothesis_key for candidate in generic}),
        key=lambda item: item.stable_key(),
    ))
    if not evidence or any(item.polarity != 3 for item in evidence):
        raise ValueError("unknown response requires UNKNOWN Memory evidence")
    candidate_keys = tuple(item.candidate_key for item in generic)
    variant, units = course.select(state.query_key, candidate_keys)
    segments = course.segments(units)
    surface = "".join(chr(value) for segment in segments for value in segment.units)
    response_act = response_act_identity((UNKNOWN_RESPONSE_COURSE_VERSION, 3, variant))
    source_hash = max(1, sum(course.source_key) + variant)
    slots = []
    for segment in segments:
        role = structure_concept_identity(
            (UNKNOWN_RESPONSE_COURSE_VERSION, 4, variant, segment.ordinal))
        filler = structure_concept_identity(
            (UNKNOWN_RESPONSE_COURSE_VERSION, 5, variant, segment.ordinal))
        slots.append(ResponseSlot(
            role, filler, "".join(chr(value) for value in segment.units), True,
            source_hash + segment.ordinal,
            allowed_node_kinds=(OBJECT_STRUCTURE_CONCEPT,),
        ))
    frame = structure_concept_identity((UNKNOWN_RESPONSE_COURSE_VERSION, 6, variant,
                                        len(segments)))
    realization = ResponseRealization(surface, frame, source_hash, len(slots))
    memory_refs = tuple(sorted({
        MemoryObjectRef.from_stable_key(item.hypothesis_key)
        for item in generic
    } | {
        MemoryObjectRef.from_stable_key(item.evidence_key)
        for item in evidence if item.evidence_key
    }, key=lambda item: item.stable_key()))
    evidence_refs = tuple(sorted({
        item.stable_key() for item in evidence
    } | {frame_key(UNKNOWN_RESPONSE_COURSE_VERSION, course.stable_key())}))
    plan = ResponsePlan(
        response_act,
        (),
        tuple(slots),
        (realization,),
        evidence_refs,
        scope_and_time=frame_key(
            UNKNOWN_RESPONSE_COURSE_VERSION,
            state.query_key,
            (len(generic),),
        ),
        discourse_links=(frame, *(slot.role for slot in slots)),
        memory_refs=memory_refs,
        style_ref=None,
        carrier_ref=None,
    )
    trace = frame_key(
        UNKNOWN_RESPONSE_COURSE_VERSION,
        course.stable_key(),
        state.query_key,
        (variant,),
        *(item.hypothesis_key for item in generic),
        *(item.evidence_key for item in evidence),
        frame_key(2, *(segment.stable_key() for segment in segments)),
    )
    return plan, trace, variant


__all__ = [
    "GENERIC_RESPONSE_CONDITION_VERSION",
    "STRUCTURAL_RESPONSE_COURSE_VERSION",
    "RESPONSE_PART_LITERAL",
    "RESPONSE_PART_GRAPH_ROLE",
    "RESPONSE_VARIANT_IDENTITY_VERSION",
    "RESPONSE_FEATURE_UNKNOWN",
    "RESPONSE_FEATURE_HOT_CONTEXT",
    "RESPONSE_FEATURE_COLD_CONTEXT",
    "RESPONSE_FEATURE_TOPIC",
    "RESPONSE_FEATURE_TOPIC_GRAPH",
    "RESPONSE_FEATURE_REFERENCE_RESOLVED",
    "RESPONSE_FEATURE_REFERENCE_OPEN",
    "RESPONSE_FEATURE_MULTI_HYPOTHESIS",
    "RESPONSE_FEATURE_CONFLICT",
    "RESPONSE_FEATURE_DISCOURSE_RELATION",
    "RESPONSE_FEATURE_RELATION_CHAIN",
    "RESPONSE_FEATURE_QUD",
    "RESPONSE_FEATURE_REVISION",
    "RESPONSE_FEATURE_RELATION_CONFLICT",
    "RESPONSE_FEATURE_EVENT_TIME",
    "RESPONSE_FEATURE_ENTITY",
    "RESPONSE_FEATURE_EVENT",
    "RESPONSE_FEATURE_PROPERTY",
    "UNKNOWN_RESPONSE_COURSE_SOURCE",
    "UNKNOWN_RESPONSE_COURSE_VERSION",
    "UnknownResponseCourse",
    "UnknownResponsePart",
    "UnknownResponseSegment",
    "build_unknown_response_plan",
    "course_from_integer_payload",
    "load_integer_course",
    "response_condition_matches",
    "response_condition_priority",
]
