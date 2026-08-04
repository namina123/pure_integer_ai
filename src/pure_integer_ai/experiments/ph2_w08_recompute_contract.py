"""W-08 局部重算的 typed mapping、快照、Use 与审计合同。"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorContextUpdate,
    AttractorDependency,
)
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.parser_revision import ParserRevisionRequest
from pure_integer_ai.cognition.shared.situation_state import (
    SituationProjectionReplacement,
    SituationRebuildReceipt,
)
from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextRevisionInvalidationReceipt,
)
from pure_integer_ai.experiments.memory_reparse_runtime import (
    MemoryParserRevisionResult,
)
from pure_integer_ai.experiments.parser_revision_runtime import ParserRevisionResult
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_STOP_STATES,
)
from pure_integer_ai.experiments.ph2_w08_discourse import W08DiscourseAuditReceipt


W08_LOCAL_REVISION_KINDS = (
    "PARSER_REVISION",
    "SENSE_BOUNDARY_SPLIT_MERGE",
    "REFERENCE_BACKTRACK",
    "SOURCE_WITHDRAWAL",
    "SOURCE_CONFLICT",
    "LATER_CORRECTION",
)
W08_REVISION_MAPPING_SHAPES = (
    "OLD_TO_ZERO",
    "OLD_TO_ONE",
    "ONE_TO_MANY",
    "MANY_TO_ONE",
)
W08_LOCAL_OBJECT_KINDS = (
    "HIERARCHY",
    "CENTER",
    "CLAIM",
    "REFERENCE",
    "SOURCE",
    "PARSER_BOUNDARY",
)
W08_LOCAL_STATE_CHANNELS = (
    "PROJECTION",
    "EVIDENCE",
    "UNDERSTANDING_USE",
    "REASONING_USE",
    "GENERATION_USE",
    "SOURCE_CITATION",
    "AGENDA",
    "CHECKPOINT",
    "GENERATION_OUTPUT",
)
W08_RECOMPUTE_OWNER_KEYS = (
    "W08-03_DISCOURSE_STATE_OWNER",
    "A-03_A-08_R-03_REVISION_OWNER",
    "MD-02_A-10_DEPENDENCY_PREVIEW_OWNER",
    "FREE_TEXT_DERIVED_INVALIDATION_OWNER",
    "W06_W07_U_R_G_REVALIDATION_OWNER",
    "MD-02_LOCAL_PROJECTION_COMMIT_OWNER",
)

_CHANNEL_FIELDS = {
    "PROJECTION": "projection_keys",
    "EVIDENCE": "evidence_keys",
    "UNDERSTANDING_USE": "understanding_use_keys",
    "REASONING_USE": "reasoning_use_keys",
    "GENERATION_USE": "generation_use_keys",
    "SOURCE_CITATION": "source_citation_keys",
    "AGENDA": "agenda_keys",
    "CHECKPOINT": "checkpoint_keys",
    "GENERATION_OUTPUT": "generation_output_keys",
}


class W08LocalRecomputeError(ValueError):
    """W-08 局部重算的影响集、映射、状态或审计不闭合。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise W08LocalRecomputeError(f"{where} must be a non-empty integer tuple")
    return value


def _keys(
    value: object,
    *,
    where: str,
    allow_empty: bool = False,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple) or (not allow_empty and not value):
        raise W08LocalRecomputeError(f"{where} must be a tuple of keys")
    for item in value:
        _key(item, where=where)
    if value != tuple(sorted(set(value))):
        raise W08LocalRecomputeError(f"{where} must be sorted and unique")
    return value


def _dependencies(
    value: object,
    *,
    where: str,
) -> tuple[AttractorDependency, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, AttractorDependency) for item in value)
    ):
        raise W08LocalRecomputeError(f"{where} must contain typed dependencies")
    keys = tuple(item.stable_key() for item in value)
    if keys != tuple(sorted(set(keys))):
        raise W08LocalRecomputeError(f"{where} must be sorted and unique")
    return value


@dataclass(frozen=True)
class W08RevisionMapping:
    """显式表达 old-to-zero/one、one-to-many 或 many-to-one。"""

    old_keys: tuple[tuple[int, ...], ...]
    new_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _keys(self.old_keys, where="revision mapping old keys")
        _keys(
            self.new_keys,
            where="revision mapping new keys",
            allow_empty=True,
        )
        if set(self.old_keys) & set(self.new_keys):
            raise W08LocalRecomputeError("revision mapping cannot select an old key")
        if len(self.old_keys) > 1 and len(self.new_keys) != 1:
            raise W08LocalRecomputeError("many old keys require exactly one new key")

    @property
    def shape(self) -> str:
        if not self.new_keys:
            return "OLD_TO_ZERO"
        if len(self.old_keys) == len(self.new_keys) == 1:
            return "OLD_TO_ONE"
        if len(self.old_keys) == 1:
            return "ONE_TO_MANY"
        return "MANY_TO_ONE"

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "shape": self.shape,
                "old": [list(item) for item in self.old_keys],
                "new": [list(item) for item in self.new_keys],
            }
        )


@dataclass(frozen=True)
class W08LocalObjectState:
    """一个依赖对象在既有 projection/Evidence/Use 等 owner 中的身份摘要。"""

    object_key: tuple[int, ...]
    object_kind: str
    dependencies: tuple[AttractorDependency, ...]
    projection_keys: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    understanding_use_keys: tuple[tuple[int, ...], ...]
    reasoning_use_keys: tuple[tuple[int, ...], ...]
    generation_use_keys: tuple[tuple[int, ...], ...]
    source_citation_keys: tuple[tuple[int, ...], ...]
    agenda_keys: tuple[tuple[int, ...], ...]
    checkpoint_keys: tuple[tuple[int, ...], ...]
    generation_output_keys: tuple[tuple[int, ...], ...]
    canonical_state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _key(self.object_key, where="local object key")
        if self.object_kind not in W08_LOCAL_OBJECT_KINDS:
            raise W08LocalRecomputeError("local object kind is not registered")
        _dependencies(self.dependencies, where="local object dependencies")
        for channel, field in _CHANNEL_FIELDS.items():
            _keys(getattr(self, field), where=f"local object {channel}")
        _key(self.canonical_state_key, where="local object canonical state")

    def channel_keys(self, channel: str) -> tuple[tuple[int, ...], ...]:
        field = _CHANNEL_FIELDS.get(channel)
        if field is None:
            raise W08LocalRecomputeError("local state channel is not registered")
        return getattr(self, field)

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "object": list(self.object_key),
                "kind": self.object_kind,
                "dependencies": [list(item.stable_key()) for item in self.dependencies],
                "channels": {
                    channel: [list(item) for item in self.channel_keys(channel)]
                    for channel in W08_LOCAL_STATE_CHANNELS
                },
                "state": list(self.canonical_state_key),
            }
        )


@dataclass(frozen=True)
class W08LocalSnapshot:
    """由 typed dependency 可求交的只读现役对象快照。"""

    objects: tuple[W08LocalObjectState, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.objects, tuple)
            or not self.objects
            or any(not isinstance(item, W08LocalObjectState) for item in self.objects)
        ):
            raise TypeError("local snapshot objects type is invalid")
        keys = tuple(item.object_key for item in self.objects)
        if keys != tuple(sorted(set(keys))):
            raise W08LocalRecomputeError("local snapshot object keys drifted")
        projection_keys = tuple(
            key for item in self.objects for key in item.projection_keys
        )
        if len(projection_keys) != len(set(projection_keys)):
            raise W08LocalRecomputeError("projection key belongs to multiple local objects")

    def affected(
        self,
        changed_dependencies: tuple[AttractorDependency, ...],
    ) -> tuple[W08LocalObjectState, ...]:
        _dependencies(changed_dependencies, where="snapshot changed dependencies")
        changed = {item.stable_key() for item in changed_dependencies}
        return tuple(
            item
            for item in self.objects
            if changed & {dependency.stable_key() for dependency in item.dependencies}
        )

    def state_ref(self) -> tuple[int, ...]:
        return digest_value(
            {"objects": [list(item.stable_key()) for item in self.objects]}
        )

    def channel_ref(
        self,
        channel: str,
        object_keys: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        _keys(object_keys, where="channel object keys", allow_empty=True)
        selected = tuple(
            item for item in self.objects if item.object_key in set(object_keys)
        )
        if tuple(item.object_key for item in selected) != object_keys:
            raise W08LocalRecomputeError("channel object keys are not exact")
        return digest_value(
            {
                "channel": channel,
                "objects": [
                    {
                        "object": list(item.object_key),
                        "values": [list(value) for value in item.channel_keys(channel)],
                        "state": list(item.canonical_state_key),
                    }
                    for item in selected
                ],
            }
        )


@dataclass(frozen=True)
class W08LocalRevisionRequest:
    """绑定 W08-03 状态入口、revision event、A-10 update 与显式映射。"""

    request_key: tuple[int, ...]
    discourse_audit: W08DiscourseAuditReceipt
    revision_kind: str
    changed_dependencies: tuple[AttractorDependency, ...]
    mappings: tuple[W08RevisionMapping, ...]
    before_snapshot: W08LocalSnapshot
    projection_update: AttractorContextUpdate
    parser_request: ParserRevisionRequest | None
    parser_result: ParserRevisionResult | None
    memory_parser_result: MemoryParserRevisionResult | None
    revision_event: MaterializedMemoryEvent
    projection_replacements: tuple[SituationProjectionReplacement, ...]
    preserved_event_hashes: tuple[int, ...]

    def __post_init__(self) -> None:
        _key(self.request_key, where="local revision request")
        if not isinstance(self.discourse_audit, W08DiscourseAuditReceipt):
            raise TypeError("local revision discourse audit type is invalid")
        if self.discourse_audit.stop_state != "RESOLVED":
            raise W08LocalRecomputeError("local revision requires resolved W08-03 state")
        if self.revision_kind not in W08_LOCAL_REVISION_KINDS:
            raise W08LocalRecomputeError("local revision kind is not registered")
        _dependencies(self.changed_dependencies, where="local changed dependencies")
        if (
            not isinstance(self.mappings, tuple)
            or not self.mappings
            or any(not isinstance(item, W08RevisionMapping) for item in self.mappings)
        ):
            raise TypeError("local revision mappings type is invalid")
        mapping_keys = tuple(item.stable_key() for item in self.mappings)
        if mapping_keys != tuple(sorted(set(mapping_keys))):
            raise W08LocalRecomputeError("local revision mappings drifted")
        if not isinstance(self.before_snapshot, W08LocalSnapshot):
            raise TypeError("local revision before snapshot type is invalid")
        if not isinstance(self.projection_update, AttractorContextUpdate):
            raise TypeError("local revision projection update type is invalid")
        if tuple(
            item.stable_key() for item in self.projection_update.changed_dependencies
        ) != tuple(item.stable_key() for item in self.changed_dependencies):
            raise W08LocalRecomputeError("projection update dependencies drifted")
        affected = tuple(
            item.object_key
            for item in self.before_snapshot.affected(self.changed_dependencies)
        )
        mapped_old = tuple(sorted(key for item in self.mappings for key in item.old_keys))
        if mapped_old != affected:
            raise W08LocalRecomputeError("revision mappings do not cover exact dependency hits")
        mapped_new = tuple(key for item in self.mappings for key in item.new_keys)
        if len(mapped_new) != len(set(mapped_new)):
            raise W08LocalRecomputeError("revision target is selected by multiple mappings")
        unaffected = {
            item.object_key for item in self.before_snapshot.objects
        } - set(affected)
        if set(mapped_new) & unaffected:
            raise W08LocalRecomputeError("revision target overwrites an unaffected object")
        parser_bound = self.revision_kind in {
            "PARSER_REVISION",
            "SENSE_BOUNDARY_SPLIT_MERGE",
        }
        if parser_bound != (
            isinstance(self.parser_request, ParserRevisionRequest)
            and isinstance(self.parser_result, ParserRevisionResult)
            and isinstance(self.memory_parser_result, MemoryParserRevisionResult)
        ):
            raise W08LocalRecomputeError(
                "parser-bound revision lacks exact A-03/A-08 receipt"
            )
        if not parser_bound and (
            self.parser_request is not None
            or self.parser_result is not None
            or self.memory_parser_result is not None
        ):
            raise W08LocalRecomputeError("non-parser revision carries A-03/A-08 state")
        if parser_bound and self.memory_parser_result.revision != self.parser_result.materialized:
            raise W08LocalRecomputeError("A-03 and A-08 materialized revisions drifted")
        if not isinstance(self.revision_event, MaterializedMemoryEvent):
            raise TypeError("local revision event type is invalid")
        if (
            not isinstance(self.projection_replacements, tuple)
            or any(
                not isinstance(item, SituationProjectionReplacement)
                for item in self.projection_replacements
            )
        ):
            raise TypeError("local projection replacements type is invalid")
        if (
            not isinstance(self.preserved_event_hashes, tuple)
            or not self.preserved_event_hashes
            or any(type(item) is not int or item <= 0 for item in self.preserved_event_hashes)
            or self.preserved_event_hashes
            != tuple(sorted(set(self.preserved_event_hashes)))
        ):
            raise W08LocalRecomputeError("preserved event hashes drifted")

    @property
    def target_keys(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(key for item in self.mappings for key in item.new_keys))


@dataclass(frozen=True)
class W08ConsumerRevalidation:
    """revision 后独立追加的一次 U/R/G Use/outcome。"""

    consumer_key: str
    prior_use_keys: tuple[tuple[int, ...], ...]
    appended_use_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    outcome_state: str
    prior_uses_preserved: int = 1
    prior_generation_output_rewritten: int = 0
    reference_recoverable: int = 1
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08LocalRecomputeError("local consumer is not registered")
        _keys(self.prior_use_keys, where="prior Use keys")
        _key(self.appended_use_key, where="appended Use key")
        if self.appended_use_key in self.prior_use_keys:
            raise W08LocalRecomputeError("consumer revalidation overwrote an old Use")
        _key(self.selected_candidate_key, where="revalidated candidate")
        _keys(self.evidence_keys, where="revalidated Evidence")
        if self.outcome_state not in W08_STOP_STATES:
            raise W08LocalRecomputeError("revalidated outcome state is invalid")
        if (
            self.prior_uses_preserved,
            self.prior_generation_output_rewritten,
            self.reference_recoverable,
            self.host_learning_write_count,
        ) != (1, 0, 1, 0):
            raise W08LocalRecomputeError("consumer revalidation rewrote history")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "consumer": self.consumer_key,
                "prior_uses": [list(item) for item in self.prior_use_keys],
                "appended_use": list(self.appended_use_key),
                "candidate": list(self.selected_candidate_key),
                "evidence": [list(item) for item in self.evidence_keys],
                "outcome": self.outcome_state,
                "prior_uses_preserved": self.prior_uses_preserved,
                "prior_generation_output_rewritten": (
                    self.prior_generation_output_rewritten
                ),
                "reference_recoverable": self.reference_recoverable,
                "host_learning_write_count": self.host_learning_write_count,
            }
        )


@dataclass(frozen=True)
class W08ChannelPreservation:
    channel: str
    before_ref: tuple[int, ...]
    after_ref: tuple[int, ...]
    bit_identical: int = 1

    def __post_init__(self) -> None:
        if self.channel not in W08_LOCAL_STATE_CHANNELS:
            raise W08LocalRecomputeError("preservation channel is not registered")
        _key(self.before_ref, where="preservation before ref")
        _key(self.after_ref, where="preservation after ref")
        if self.before_ref != self.after_ref or self.bit_identical != 1:
            raise W08LocalRecomputeError("unaffected channel is not bit-identical")


@dataclass(frozen=True)
class W08LocalRecomputeAuditReceipt:
    """一次局部 preview/commit 的完整影响集、Use 与资源收据。"""

    request_key: tuple[int, ...]
    revision_kind: str
    mapping_shapes: tuple[str, ...]
    before_snapshot_ref: tuple[int, ...]
    after_snapshot_ref: tuple[int, ...]
    projection: SituationRebuildReceipt
    free_text: FreeTextRevisionInvalidationReceipt
    consumers: tuple[W08ConsumerRevalidation, ...]
    preservations: tuple[W08ChannelPreservation, ...]
    affected_object_keys: tuple[tuple[int, ...], ...]
    owner_call_order: tuple[str, ...]
    recompute_object_count: int
    full_document_reparse_count: int
    additional_payload_get_count: int
    host_learning_write_count: int
    stop_state: str = "RESOLVED"

    def __post_init__(self) -> None:
        _key(self.request_key, where="local audit request")
        if self.revision_kind not in W08_LOCAL_REVISION_KINDS:
            raise W08LocalRecomputeError("local audit revision kind drifted")
        if (
            not isinstance(self.mapping_shapes, tuple)
            or not self.mapping_shapes
            or any(item not in W08_REVISION_MAPPING_SHAPES for item in self.mapping_shapes)
        ):
            raise W08LocalRecomputeError("local audit mapping shapes drifted")
        _key(self.before_snapshot_ref, where="local audit before snapshot")
        _key(self.after_snapshot_ref, where="local audit after snapshot")
        if not isinstance(self.projection, SituationRebuildReceipt):
            raise TypeError("local audit projection receipt type is invalid")
        if not isinstance(self.free_text, FreeTextRevisionInvalidationReceipt):
            raise TypeError("local audit free-text receipt type is invalid")
        if tuple(item.consumer_key for item in self.consumers) != W08_CONSUMER_KEYS:
            raise W08LocalRecomputeError("local audit lacks exact U/R/G revalidation")
        appended_uses = tuple(item.appended_use_key for item in self.consumers)
        if len(set(appended_uses)) != len(appended_uses):
            raise W08LocalRecomputeError("local audit reused one Use across consumers")
        if tuple(item.channel for item in self.preservations) != W08_LOCAL_STATE_CHANNELS:
            raise W08LocalRecomputeError("local audit preservation inventory drifted")
        _keys(self.affected_object_keys, where="local audit affected objects")
        parser_bound = self.revision_kind in {
            "PARSER_REVISION",
            "SENSE_BOUNDARY_SPLIT_MERGE",
        }
        expected_owners = (
            W08_RECOMPUTE_OWNER_KEYS
            if parser_bound
            else (
                W08_RECOMPUTE_OWNER_KEYS[0],
                *W08_RECOMPUTE_OWNER_KEYS[2:],
            )
        )
        if self.owner_call_order != expected_owners:
            raise W08LocalRecomputeError("local audit owner order drifted")
        for value in (
            self.recompute_object_count,
            self.full_document_reparse_count,
            self.additional_payload_get_count,
            self.host_learning_write_count,
        ):
            if type(value) is not int or value < 0:
                raise W08LocalRecomputeError("local audit resource count is invalid")
        if self.recompute_object_count != len(self.affected_object_keys):
            raise W08LocalRecomputeError("local audit recompute count is not exact")
        if not {
            item.components for item in self.free_text.invalidated_keys
        } <= set(self.affected_object_keys):
            raise W08LocalRecomputeError("free-text invalidation escaped the affected set")
        if (
            self.full_document_reparse_count,
            self.additional_payload_get_count,
            self.host_learning_write_count,
        ) != (0, 0, 0):
            raise W08LocalRecomputeError("local audit used full reparse or hidden writes")
        if self.stop_state != "RESOLVED":
            raise W08LocalRecomputeError("local recompute audit is not resolved")

    def result_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "request": list(self.request_key),
                "revision_kind": self.revision_kind,
                "mapping_shapes": list(self.mapping_shapes),
                "before": list(self.before_snapshot_ref),
                "after": list(self.after_snapshot_ref),
                "projection": list(self.projection.stable_key()),
                "free_text_invalidated": [
                    list(item.components) for item in self.free_text.invalidated_keys
                ],
                "free_text_preserved": [
                    list(item.components) for item in self.free_text.preserved_keys
                ],
                "consumers": [list(item.stable_key()) for item in self.consumers],
                "preservations": [
                    {
                        "channel": item.channel,
                        "ref": list(item.before_ref),
                    }
                    for item in self.preservations
                ],
                "affected_objects": [list(item) for item in self.affected_object_keys],
                "owners": list(self.owner_call_order),
                "recompute_objects": self.recompute_object_count,
                "stop": self.stop_state,
            }
        )

    def dump(self) -> "W08LocalRecomputeDump":
        return W08LocalRecomputeDump(
            self.request_key,
            self.before_snapshot_ref,
            self.after_snapshot_ref,
            self.projection.after_projection_ref,
            self.result_key(),
            self.recompute_object_count,
        )


@dataclass(frozen=True)
class W08LocalRecomputeDump:
    """metadata-only dump；不携带原文、Evidence payload 或完整答案。"""

    request_key: tuple[int, ...]
    before_snapshot_ref: tuple[int, ...]
    after_snapshot_ref: tuple[int, ...]
    after_projection_ref: tuple[int, ...]
    result_key: tuple[int, ...]
    recompute_object_count: int

    def __post_init__(self) -> None:
        for name in (
            "request_key",
            "before_snapshot_ref",
            "after_snapshot_ref",
            "after_projection_ref",
            "result_key",
        ):
            _key(getattr(self, name), where=f"local dump {name}")
        if type(self.recompute_object_count) is not int or self.recompute_object_count <= 0:
            raise W08LocalRecomputeError("local dump recompute count is invalid")

    def to_bytes(self) -> bytes:
        value = {
            "after_projection_ref": list(self.after_projection_ref),
            "after_snapshot_ref": list(self.after_snapshot_ref),
            "before_snapshot_ref": list(self.before_snapshot_ref),
            "recompute_object_count": self.recompute_object_count,
            "request_key": list(self.request_key),
            "result_key": list(self.result_key),
            "version": 1,
        }
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "ascii"
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "W08LocalRecomputeDump":
        if not isinstance(payload, bytes):
            raise TypeError("local dump payload must be bytes")
        try:
            value = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise W08LocalRecomputeError("local dump payload is invalid") from error
        expected = {
            "after_projection_ref",
            "after_snapshot_ref",
            "before_snapshot_ref",
            "recompute_object_count",
            "request_key",
            "result_key",
            "version",
        }
        if not isinstance(value, dict) or set(value) != expected or value["version"] != 1:
            raise W08LocalRecomputeError("local dump fields drifted")

        def key(name: str) -> tuple[int, ...]:
            raw = value[name]
            if not isinstance(raw, list) or any(type(item) is not int for item in raw):
                raise W08LocalRecomputeError(f"local dump {name} is invalid")
            return tuple(raw)

        restored = cls(
            key("request_key"),
            key("before_snapshot_ref"),
            key("after_snapshot_ref"),
            key("after_projection_ref"),
            key("result_key"),
            value["recompute_object_count"],
        )
        if restored.to_bytes() != payload:
            raise W08LocalRecomputeError("local dump is not canonical")
        return restored


@dataclass(frozen=True)
class W08LocalReplayReceipt:
    request_key: tuple[int, ...]
    result_key: tuple[int, ...]
    current_projection_ref: tuple[int, ...]
    dump_equal: int = 1
    additional_write_count: int = 0
    additional_payload_get_count: int = 0

    def __post_init__(self) -> None:
        _key(self.request_key, where="local replay request")
        _key(self.result_key, where="local replay result")
        _key(self.current_projection_ref, where="local replay projection")
        if (
            self.dump_equal,
            self.additional_write_count,
            self.additional_payload_get_count,
        ) != (1, 0, 0):
            raise W08LocalRecomputeError("local replay performed new work")


@dataclass(frozen=True)
class W08LocalRecomputeAblationReport:
    affected_dimensions: tuple[str, ...]
    unaffected_dimensions: tuple[str, ...]


def assess_w08_local_recompute_ablation(
    *,
    full_dimension_outcomes: dict[str, str],
    ablated_dimension_outcomes: dict[str, str],
) -> W08LocalRecomputeAblationReport:
    expected = set(W08_DIMENSION_KEYS)
    if set(full_dimension_outcomes) != expected or set(ablated_dimension_outcomes) != expected:
        raise W08LocalRecomputeError("local recompute ablation inventory drifted")
    target = "W-08-LOCAL_RECOMPUTE"
    changed = tuple(
        key
        for key in W08_DIMENSION_KEYS
        if full_dimension_outcomes[key] != ablated_dimension_outcomes[key]
    )
    if (
        full_dimension_outcomes[target] != "PASS"
        or changed != (target,)
        or ablated_dimension_outcomes[target] == "PASS"
    ):
        raise W08LocalRecomputeError("local recompute ablation is not orthogonal")
    return W08LocalRecomputeAblationReport(
        changed,
        tuple(key for key in W08_DIMENSION_KEYS if key != target),
    )


__all__ = [
    "W08ChannelPreservation",
    "W08ConsumerRevalidation",
    "W08LocalObjectState",
    "W08LocalRecomputeAblationReport",
    "W08LocalRecomputeAuditReceipt",
    "W08LocalRecomputeDump",
    "W08LocalRecomputeError",
    "W08LocalReplayReceipt",
    "W08LocalRevisionRequest",
    "W08LocalSnapshot",
    "W08RevisionMapping",
    "W08_LOCAL_OBJECT_KINDS",
    "W08_LOCAL_REVISION_KINDS",
    "W08_LOCAL_STATE_CHANNELS",
    "W08_RECOMPUTE_OWNER_KEYS",
    "W08_REVISION_MAPPING_SHAPES",
    "assess_w08_local_recompute_ablation",
]
