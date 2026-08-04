"""W-08 facade 到现役 A/H/MD/G owner 的小型边界适配。"""
from __future__ import annotations

from collections.abc import Callable

from pure_integer_ai.cognition.shared.memory_event import MemoryEvent
from pure_integer_ai.cognition.shared.memory_event_log import MaterializedMemoryEvent
from pure_integer_ai.cognition.shared.situation_state import (
    CurrentSituationProjection,
    SituationEventLog,
)
from pure_integer_ai.cognition.understanding.occurrence_reference import (
    OccurrenceReferenceRequest,
    OccurrenceReferenceResolution,
    OccurrenceReferenceRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import GenerationSurfaceRuntime
from pure_integer_ai.experiments.ph2_md03_center_adapter import DirectionalMemoryCenterAdapter
from pure_integer_ai.experiments.ph2_w08_discourse import (
    W08AgendaReceipt,
    W08CenterReceipt,
    W08DiscourseRequest,
    W08DiscourseUse,
    W08EventReceipt,
    W08GenerationReceipt,
    W08LifecycleReceipt,
    W08ProjectionReceipt,
    W08ReferenceReceipt,
    W08_DISCOURSE_OWNER_KEYS,
)


class W08DiscourseAdapterError(ValueError):
    """现役 owner 与 W-08 receipt 之间的边界不一致。"""


class W08MD03CenterOwner:
    """把 MD-03 ``DirectionalMemoryCenterAdapter`` 绑定为 W-08 center owner。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[0]

    def __init__(
        self,
        adapter: DirectionalMemoryCenterAdapter,
        mapper: Callable[[DirectionalMemoryCenterAdapter, W08DiscourseRequest], W08CenterReceipt],
    ) -> None:
        if not isinstance(adapter, DirectionalMemoryCenterAdapter):
            raise TypeError("W08 MD-03 owner requires DirectionalMemoryCenterAdapter")
        if not callable(mapper):
            raise TypeError("W08 MD-03 center mapper must be callable")
        self.adapter = adapter
        self.mapper = mapper

    def form(self, request: W08DiscourseRequest) -> W08CenterReceipt:
        receipt = self.mapper(self.adapter, request)
        if not isinstance(receipt, W08CenterReceipt):
            raise W08DiscourseAdapterError("MD-03 mapper returned an invalid center receipt")
        return receipt


class W08SituationEventOwner:
    """把 MD-02 ``SituationEventLog`` 的 append-only 事件写入接到 facade。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[1]

    def __init__(
        self,
        situation: SituationEventLog,
        event_builder: Callable[[W08DiscourseRequest, W08CenterReceipt], MemoryEvent],
    ) -> None:
        if not isinstance(situation, SituationEventLog):
            raise TypeError("W08 event owner requires SituationEventLog")
        if not callable(event_builder):
            raise TypeError("W08 event builder must be callable")
        self.situation = situation
        self.event_builder = event_builder

    def append(
        self,
        request: W08DiscourseRequest,
        center: W08CenterReceipt,
    ) -> W08EventReceipt:
        before = self.situation.event_log.query(access=self.situation.access)
        event = self.event_builder(request, center)
        if not isinstance(event, MemoryEvent):
            raise W08DiscourseAdapterError("event builder returned a non-MemoryEvent")
        materialized = self.situation.append(event)
        if not isinstance(materialized, MaterializedMemoryEvent):
            raise W08DiscourseAdapterError("SituationEventLog returned an invalid event")
        return W08EventReceipt(
            self.owner_key,
            request.request_key,
            tuple((item.event_hash,) for item in before),
            (materialized.event_hash,),
        )


class W08CurrentProjectionOwner:
    """把 MD-02 current projection/recompute mapper 绑定为唯一 projection owner。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[3]

    def __init__(
        self,
        projection: CurrentSituationProjection,
        materializer: Callable[
            [CurrentSituationProjection, W08DiscourseRequest, W08EventReceipt, W08LifecycleReceipt],
            W08ProjectionReceipt,
        ],
    ) -> None:
        if not isinstance(projection, CurrentSituationProjection):
            raise TypeError("W08 projection owner requires CurrentSituationProjection")
        if not callable(materializer):
            raise TypeError("W08 projection materializer must be callable")
        self.projection = projection
        self.materializer = materializer

    def project(
        self,
        request: W08DiscourseRequest,
        event: W08EventReceipt,
        lifecycle: W08LifecycleReceipt,
    ) -> W08ProjectionReceipt:
        receipt = self.materializer(self.projection, request, event, lifecycle)
        if not isinstance(receipt, W08ProjectionReceipt):
            raise W08DiscourseAdapterError("projection materializer returned an invalid receipt")
        return receipt


class W08A01ReferenceOwner:
    """把 A-01 ``OccurrenceReferenceRuntime`` 的真实 H-04 resolution 接入 facade。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[5]

    def __init__(
        self,
        runtime: OccurrenceReferenceRuntime,
        request_builder: Callable[[W08DiscourseRequest], OccurrenceReferenceRequest],
        receipt_builder: Callable[
            [W08DiscourseRequest, OccurrenceReferenceResolution],
            W08ReferenceReceipt,
        ],
    ) -> None:
        if not isinstance(runtime, OccurrenceReferenceRuntime):
            raise TypeError("W08 reference owner requires OccurrenceReferenceRuntime")
        if not callable(request_builder) or not callable(receipt_builder):
            raise TypeError("W08 reference builders must be callable")
        self.runtime = runtime
        self.request_builder = request_builder
        self.receipt_builder = receipt_builder

    def resolve(
        self,
        request: W08DiscourseRequest,
        projection: W08ProjectionReceipt,
    ) -> W08ReferenceReceipt:
        runtime_request = self.request_builder(request)
        if not isinstance(runtime_request, OccurrenceReferenceRequest):
            raise W08DiscourseAdapterError("A-01 request builder returned an invalid request")
        resolution = self.runtime.resolve(runtime_request)
        if not isinstance(resolution, OccurrenceReferenceResolution):
            raise W08DiscourseAdapterError("A-01 runtime returned an invalid resolution")
        receipt = self.receipt_builder(request, resolution)
        if not isinstance(receipt, W08ReferenceReceipt):
            raise W08DiscourseAdapterError("A-01 receipt builder returned an invalid receipt")
        return receipt


class W08GenerationOwner:
    """把 G-03 ``GenerationSurfaceRuntime``/G-04 postcheck 接入 facade。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[6]

    def __init__(
        self,
        runtime: GenerationSurfaceRuntime,
        chooser: Callable[
            [GenerationSurfaceRuntime, W08DiscourseRequest, W08ProjectionReceipt],
            W08GenerationReceipt,
        ],
    ) -> None:
        if not isinstance(runtime, GenerationSurfaceRuntime):
            raise TypeError("W08 generation owner requires GenerationSurfaceRuntime")
        if not callable(chooser):
            raise TypeError("W08 generation chooser must be callable")
        self.runtime = runtime
        self.chooser = chooser

    def choose(
        self,
        request: W08DiscourseRequest,
        projection: W08ProjectionReceipt,
        reference: W08ReferenceReceipt | None,
    ) -> W08GenerationReceipt:
        receipt = self.chooser(self.runtime, request, projection)
        if not isinstance(receipt, W08GenerationReceipt):
            raise W08DiscourseAdapterError("G-03/G-04 chooser returned an invalid receipt")
        return receipt


class W08AgendaOwner:
    """从现役 MD-02 dependency index 生成 A-10 局部 agenda receipt。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[4]

    def __init__(
        self,
        projection: CurrentSituationProjection,
        planner: Callable[
            [object, W08DiscourseRequest, W08ProjectionReceipt],
            W08AgendaReceipt,
        ],
    ) -> None:
        if not isinstance(projection, CurrentSituationProjection):
            raise TypeError("W08 agenda owner requires CurrentSituationProjection")
        if not callable(planner):
            raise TypeError("W08 agenda planner must be callable")
        self.projection = projection
        self.planner = planner

    def plan(self, request: W08DiscourseRequest, projection: W08ProjectionReceipt) -> W08AgendaReceipt:
        receipt = self.planner(self.projection.dependency_index, request, projection)
        if not isinstance(receipt, W08AgendaReceipt):
            raise W08DiscourseAdapterError("A-10 planner returned an invalid receipt")
        return receipt


class W08LifecycleOwner:
    """绑定 H-04/H-05 candidate lifecycle 的现役 mapper，不自持候选账本。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[2]

    def __init__(
        self,
        lifecycle_mapper: Callable[[W08DiscourseRequest, W08EventReceipt], W08LifecycleReceipt],
    ) -> None:
        if not callable(lifecycle_mapper):
            raise TypeError("W08 lifecycle mapper must be callable")
        self.lifecycle_mapper = lifecycle_mapper

    def resolve(self, request: W08DiscourseRequest, event: W08EventReceipt) -> W08LifecycleReceipt:
        receipt = self.lifecycle_mapper(request, event)
        if not isinstance(receipt, W08LifecycleReceipt):
            raise W08DiscourseAdapterError("H-04/H-05 mapper returned an invalid receipt")
        return receipt


class W08ConsumerOwner:
    """绑定 W06/W07 typed projection 的 U/R/G consumer mapper。"""

    owner_key = W08_DISCOURSE_OWNER_KEYS[7]

    def __init__(
        self,
        consumer_mapper: Callable[
            [W08DiscourseRequest, str, tuple[int, ...], tuple[tuple[int, ...], ...], str],
            W08DiscourseUse,
        ],
    ) -> None:
        if not callable(consumer_mapper):
            raise TypeError("W06/W07 consumer mapper must be callable")
        self.consumer_mapper = consumer_mapper

    def consume(
        self,
        request: W08DiscourseRequest,
        consumer_key: str,
        selected_candidate_key: tuple[int, ...],
        evidence_keys: tuple[tuple[int, ...], ...],
        outcome_state: str,
    ) -> W08DiscourseUse:
        receipt = self.consumer_mapper(
            request,
            consumer_key,
            selected_candidate_key,
            evidence_keys,
            outcome_state,
        )
        if not isinstance(receipt, W08DiscourseUse):
            raise W08DiscourseAdapterError("W06/W07 consumer mapper returned an invalid Use")
        return receipt


__all__ = [
    "W08A01ReferenceOwner",
    "W08AgendaOwner",
    "W08ConsumerOwner",
    "W08CurrentProjectionOwner",
    "W08DiscourseAdapterError",
    "W08GenerationOwner",
    "W08LifecycleOwner",
    "W08MD03CenterOwner",
    "W08SituationEventOwner",
]
