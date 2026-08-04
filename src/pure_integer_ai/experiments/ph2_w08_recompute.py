"""W-08 局部重算 runtime：preview 全部 owner 后唯一提交 MD-02 投影。"""
from __future__ import annotations

from collections.abc import Callable

from pure_integer_ai.cognition.shared.situation_state import CurrentSituationProjection
from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextRevisionInvalidator,
)
from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS
from pure_integer_ai.experiments.ph2_w08_recompute_contract import (
    W08ChannelPreservation,
    W08ConsumerRevalidation,
    W08LocalObjectState,
    W08LocalRecomputeAuditReceipt,
    W08LocalRecomputeDump,
    W08LocalRecomputeError,
    W08LocalReplayReceipt,
    W08LocalRevisionRequest,
    W08LocalSnapshot,
    W08_LOCAL_STATE_CHANNELS,
    W08_RECOMPUTE_OWNER_KEYS,
)


W08_LOCAL_RECOMPUTE_FAULT_POINTS = (
    "BEFORE_OWNER_PREVIEW",
    "AFTER_OWNER_PREVIEW_BEFORE_COMMIT",
)


class W08LocalRecomputeInjectedFailure(RuntimeError):
    """W08-04 在首个可变写之前注入的确定性故障。"""


class W08LocalRecomputeRuntime:
    """组合现役 MD-02/A-10 与 free-text invalidator，不持有第二份真值。"""

    def __init__(
        self,
        projection: CurrentSituationProjection,
        invalidator: FreeTextRevisionInvalidator,
        state_rebuilder: Callable[
            [W08LocalRevisionRequest, tuple[W08LocalObjectState, ...]],
            tuple[W08LocalObjectState, ...],
        ],
        consumer_revalidator: Callable[
            [
                W08LocalRevisionRequest,
                tuple[W08LocalObjectState, ...],
                W08LocalSnapshot,
            ],
            tuple[W08ConsumerRevalidation, ...],
        ],
    ) -> None:
        if not isinstance(projection, CurrentSituationProjection):
            raise TypeError("W08 local runtime requires CurrentSituationProjection")
        if not isinstance(invalidator, FreeTextRevisionInvalidator):
            raise TypeError("W08 local runtime requires FreeTextRevisionInvalidator")
        if not callable(state_rebuilder) or not callable(consumer_revalidator):
            raise TypeError("W08 local runtime callbacks must be callable")
        self.projection = projection
        self.invalidator = invalidator
        self.state_rebuilder = state_rebuilder
        self.consumer_revalidator = consumer_revalidator

    def _require_preview_unchanged(
        self,
        before_state: tuple[int, ...],
        before_work_memory: tuple[int, ...],
    ) -> None:
        if (
            self.projection.state_key() != before_state
            or self.projection.work_memory.state_key() != before_work_memory
        ):
            raise W08LocalRecomputeError("owner preview wrote current state")

    def execute(
        self,
        request: W08LocalRevisionRequest,
        *,
        fault_point: str | None = None,
    ) -> W08LocalRecomputeAuditReceipt:
        if not isinstance(request, W08LocalRevisionRequest):
            raise TypeError("W08 local runtime request type is invalid")
        if fault_point is not None and fault_point not in W08_LOCAL_RECOMPUTE_FAULT_POINTS:
            raise W08LocalRecomputeError("W08 local recompute fault point is invalid")
        discourse_projection = request.discourse_audit.projection
        if discourse_projection is None:
            raise W08LocalRecomputeError("W08-03 audit lacks current projection")
        before_projection_ref = self.projection.state_ref()
        if discourse_projection.after_projection_ref != before_projection_ref:
            raise W08LocalRecomputeError("W08-03 and MD-02 current projections drifted")

        affected = request.before_snapshot.affected(request.changed_dependencies)
        affected_keys = tuple(item.object_key for item in affected)
        unaffected = tuple(
            item
            for item in request.before_snapshot.objects
            if item.object_key not in set(affected_keys)
        )
        if not unaffected:
            raise W08LocalRecomputeError("local recompute lacks an unaffected control")

        expected_projection_keys = tuple(
            sorted(key for item in affected for key in item.projection_keys)
        )
        actual_projection_keys = self.projection.dependency_index.affected(
            request.changed_dependencies
        )
        if actual_projection_keys != expected_projection_keys:
            raise W08LocalRecomputeError("MD-02 dependency impact set drifted")
        replacement_keys = tuple(
            sorted(item.entry.projection_key for item in request.projection_replacements)
        )
        if replacement_keys != expected_projection_keys:
            raise W08LocalRecomputeError("MD-02 replacement set is not exact")

        free_text = self.invalidator.invalidate(request.changed_dependencies)
        expected_free_text = tuple(
            sorted(
                item.object_key
                for item in affected
                if item.object_kind in {"HIERARCHY", "CENTER", "CLAIM"}
            )
        )
        actual_free_text = tuple(item.components for item in free_text.invalidated_keys)
        if actual_free_text != expected_free_text:
            raise W08LocalRecomputeError("free-text invalidation impact set drifted")

        before_state = self.projection.state_key()
        before_work_memory = self.projection.work_memory.state_key()
        if fault_point == "BEFORE_OWNER_PREVIEW":
            raise W08LocalRecomputeInjectedFailure(fault_point)

        try:
            rebuilt = self.state_rebuilder(request, affected)
        except BaseException as error:
            try:
                self._require_preview_unchanged(before_state, before_work_memory)
            except W08LocalRecomputeError as drift:
                raise drift from error
            raise
        self._require_preview_unchanged(before_state, before_work_memory)
        if (
            not isinstance(rebuilt, tuple)
            or any(not isinstance(item, W08LocalObjectState) for item in rebuilt)
        ):
            raise TypeError("W08 local state rebuilder returned an invalid batch")
        rebuilt_keys = tuple(item.object_key for item in rebuilt)
        if rebuilt_keys != request.target_keys:
            raise W08LocalRecomputeError("local state rebuild targets are not exact")
        after_snapshot = W08LocalSnapshot(tuple(sorted(
            (*unaffected, *rebuilt), key=lambda item: item.object_key
        )))

        try:
            consumers = self.consumer_revalidator(request, affected, after_snapshot)
        except BaseException as error:
            try:
                self._require_preview_unchanged(before_state, before_work_memory)
            except W08LocalRecomputeError as drift:
                raise drift from error
            raise
        self._require_preview_unchanged(before_state, before_work_memory)
        if (
            not isinstance(consumers, tuple)
            or any(not isinstance(item, W08ConsumerRevalidation) for item in consumers)
            or tuple(item.consumer_key for item in consumers) != W08_CONSUMER_KEYS
        ):
            raise W08LocalRecomputeError("local consumer revalidation inventory drifted")
        use_channel = {
            "UNDERSTANDING": "UNDERSTANDING_USE",
            "REASONING": "REASONING_USE",
            "GENERATION": "GENERATION_USE",
        }
        for consumer in consumers:
            expected_prior = tuple(sorted(
                key
                for item in affected
                for key in item.channel_keys(use_channel[consumer.consumer_key])
            ))
            if consumer.prior_use_keys != expected_prior:
                raise W08LocalRecomputeError("consumer did not preserve exact prior Uses")

        unaffected_keys = tuple(item.object_key for item in unaffected)
        preservations = tuple(
            W08ChannelPreservation(
                channel,
                request.before_snapshot.channel_ref(channel, unaffected_keys),
                after_snapshot.channel_ref(channel, unaffected_keys),
            )
            for channel in W08_LOCAL_STATE_CHANNELS
        )
        if fault_point == "AFTER_OWNER_PREVIEW_BEFORE_COMMIT":
            self._require_preview_unchanged(before_state, before_work_memory)
            raise W08LocalRecomputeInjectedFailure(fault_point)

        try:
            projection = self.projection.apply_revision(
                request.projection_update,
                request.revision_event,
                request.projection_replacements,
                preserved_event_hashes=request.preserved_event_hashes,
            )
        except BaseException as error:
            if (
                self.projection.state_key() != before_state
                or self.projection.work_memory.state_key() != before_work_memory
            ):
                raise W08LocalRecomputeError(
                    "MD-02 local transaction did not restore its call state"
                ) from error
            raise
        if (
            projection.before_projection_ref != before_projection_ref
            or projection.invalidated_projection_keys != expected_projection_keys
            or projection.rebuilt_projection_keys != expected_projection_keys
        ):
            raise W08LocalRecomputeError("MD-02 commit receipt drifted from preview")

        owners = [W08_RECOMPUTE_OWNER_KEYS[0]]
        if request.parser_result is not None:
            owners.append(W08_RECOMPUTE_OWNER_KEYS[1])
        owners.extend(W08_RECOMPUTE_OWNER_KEYS[2:])
        return W08LocalRecomputeAuditReceipt(
            request.request_key,
            request.revision_kind,
            tuple(item.shape for item in request.mappings),
            request.before_snapshot.state_ref(),
            after_snapshot.state_ref(),
            projection,
            free_text,
            consumers,
            preservations,
            affected_keys,
            tuple(owners),
            len(affected),
            0,
            0,
            0,
        )

    def replay(
        self,
        audit: W08LocalRecomputeAuditReceipt,
        dump_payload: bytes,
    ) -> W08LocalReplayReceipt:
        """只回读 canonical dump 与 owner current，不重复 revision 或 payload read。"""
        if not isinstance(audit, W08LocalRecomputeAuditReceipt):
            raise TypeError("W08 local replay audit type is invalid")
        restored = W08LocalRecomputeDump.from_bytes(dump_payload)
        expected = audit.dump()
        if restored != expected or dump_payload != expected.to_bytes():
            raise W08LocalRecomputeError("local replay dump drifted")
        current = self.projection.state_ref()
        if current != restored.after_projection_ref:
            raise W08LocalRecomputeError("local replay current projection drifted")
        return W08LocalReplayReceipt(
            restored.request_key,
            restored.result_key,
            current,
        )

    def resume(
        self,
        audit: W08LocalRecomputeAuditReceipt,
        dump_payload: bytes,
    ) -> W08LocalReplayReceipt:
        """resume 与 replay 共用同一 metadata-only、零写校验路径。"""
        return self.replay(audit, dump_payload)


__all__ = [
    "W08LocalRecomputeInjectedFailure",
    "W08LocalRecomputeRuntime",
    "W08_LOCAL_RECOMPUTE_FAULT_POINTS",
]
