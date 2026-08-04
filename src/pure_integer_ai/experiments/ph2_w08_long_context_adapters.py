"""现役 R-04/R-06 生产 owner 上的 W08-05 薄适配层。"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity, document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.understanding.span_index import SpanIndex
from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterAgendaRun,
    AuthorizedCenterAgendaRuntime,
    CenterAuthorizationProjection,
)
from pure_integer_ai.experiments.free_text_recall_runtime import EvidenceFormedCenter
from pure_integer_ai.experiments.long_generation_checkpoint import (
    LONG_GENERATION_COMPLETE,
    LongGenerationCheckpoint,
    LongGenerationCheckpointStore,
    LongGenerationPageBudget,
    LongGenerationPageCommit,
    LongGenerationPlan,
)
from pure_integer_ai.experiments.long_input_hierarchy import (
    LongInputChunk,
    LongInputHierarchy,
    LongInputHierarchyBuilder,
    LongInputHierarchyError,
    LongInputHierarchyProtocol,
    LongInputHierarchySeed,
)
from pure_integer_ai.experiments.persistent_conversation_agenda import (
    AGENDA_OPEN,
    AGENDA_RESOLVED,
    PersistentAgendaCenter,
    PersistentConversationAgenda,
    PersistentConversationAgendaRuntime,
    PersistentConversationAgendaStore,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import RecallBudget
from pure_integer_ai.experiments.ph2_w08_long_context_contract import (
    W08LongContextError,
    W08LongContextRequest,
    W08LongContextUse,
    W08_LONG_CONTEXT_OWNER_KEYS,
)
from pure_integer_ai.experiments.ph2_w08_contract import W08_RESOURCE_BUDGET
from pure_integer_ai.experiments.ph2_w08_long_context_training import (
    W08LongContextTrainingBundle,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT


_NAMESPACE = 80805


def _positive_id(values: tuple[int, ...]) -> int:
    value = int.from_bytes(bytes(item & 0xFF for item in values[:8]), "big")
    return value if value > 0 else 1


@dataclass(frozen=True)
class W08LongInputMaterialization:
    training_material_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    document_digest: tuple[int, ...]
    chunks: tuple[LongInputChunk, ...]
    seeds: tuple[LongInputHierarchySeed, ...]

    def __post_init__(self) -> None:
        if not self.training_material_key:
            raise W08LongContextError("long-input material key is empty")
        if not isinstance(self.source, SourceRef):
            raise TypeError("long-input material source type is invalid")
        if not isinstance(self.scope, ScopeIdentity) or self.scope.source != self.source:
            raise W08LongContextError("long-input material scope drifted")
        if not self.chunks or not self.seeds:
            raise W08LongContextError("long-input material lacks chunks or seeds")
        if (
            not isinstance(self.document_digest, tuple)
            or len(self.document_digest) != 32
            or any(type(item) is not int or not 0 <= item <= 255 for item in self.document_digest)
        ):
            raise W08LongContextError("long-input document digest is invalid")
        try:
            source, _, digest = LongInputHierarchyBuilder.assemble(self.chunks)
        except LongInputHierarchyError as error:
            raise W08LongContextError("long-input chunk identity drifted") from error
        if source != self.source or digest != self.document_digest:
            raise W08LongContextError("long-input source/content digest drifted")


def materialize_w08_long_input(
    bundle: W08LongContextTrainingBundle,
    *,
    owner: OwnerScope,
    chunk_width: int,
    parser_version: int = 1,
) -> W08LongInputMaterialization:
    """不解析文本，把可见 typed 记录映射到 R-06 绝对 span。"""
    if not isinstance(bundle, W08LongContextTrainingBundle):
        raise TypeError("W08 long-input bundle type is invalid")
    if not isinstance(owner, OwnerScope):
        raise TypeError("W08 long-input owner type is invalid")
    if type(chunk_width) is not int or chunk_width <= 0:
        raise W08LongContextError("chunk width must be positive")
    if type(parser_version) is not int or parser_version <= 0:
        raise W08LongContextError("parser version must be positive")
    source = SourceRef(
        SOURCE_BARE_TEXT,
        _NAMESPACE,
        _positive_id(bundle.material_key),
        owner,
        VersionBundle(parser=ParserVersion(parser_version)),
    )
    scope = document_scope(source)
    chunks = tuple(reversed(tuple(
        LongInputChunk.from_text(
            source,
            start,
            bundle.document_text[start:start + chunk_width],
        )
        for start in range(0, len(bundle.document_text), chunk_width)
    )))
    _, _, document_digest = LongInputHierarchyBuilder.assemble(chunks)
    section_ranges: dict[int, tuple[int, int]] = {}
    for item in bundle.material:
        current = section_ranges.get(item.section_ordinal)
        section_ranges[item.section_ordinal] = (
            item.start if current is None else min(current[0], item.start),
            item.end if current is None else max(current[1], item.end),
        )
    document = ((0, len(bundle.document_text)),)
    seeds = tuple(
        LongInputHierarchySeed(
            proposition_identity(source, (_NAMESPACE, 1, item.proposition_ordinal + 1)),
            context_scope_identity(source, (_NAMESPACE, 2, item.proposition_ordinal + 1)),
            item.proposition_ordinal,
            document,
            (section_ranges[item.section_ordinal],),
            ((item.start, item.end),),
            ((item.start, item.end),),
            1,
            100 + item.section_ordinal,
            1000 + item.paragraph_ordinal,
            10000 + item.proposition_ordinal,
        )
        for item in bundle.material
    )
    return W08LongInputMaterialization(
        bundle.material_key,
        source,
        scope,
        document_digest,
        chunks,
        seeds,
    )


@dataclass(frozen=True)
class W08LongContextExecution:
    material: W08LongInputMaterialization
    centers: tuple[EvidenceFormedCenter, ...]
    current: MemoryCurrentQuery
    authorization: CenterAuthorizationProjection
    recall_budget: RecallBudget
    agenda_key: StableRecordKey
    generation_plan: LongGenerationPlan
    page_budget: LongGenerationPageBudget
    recompute_objects: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.material, W08LongInputMaterialization):
            raise TypeError("long-context execution material type is invalid")
        if not isinstance(self.centers, tuple) or any(
            not isinstance(item, EvidenceFormedCenter) for item in self.centers
        ):
            raise TypeError("long-context execution centers type is invalid")
        if len({item.center_key for item in self.centers}) != len(self.centers):
            raise W08LongContextError("long-context execution center is duplicated")
        if not isinstance(self.current, MemoryCurrentQuery):
            raise TypeError("long-context current query type is invalid")
        if not isinstance(self.authorization, CenterAuthorizationProjection):
            raise TypeError("long-context authorization type is invalid")
        if not isinstance(self.recall_budget, RecallBudget):
            raise TypeError("long-context recall budget type is invalid")
        if not isinstance(self.agenda_key, StableRecordKey):
            raise TypeError("long-context agenda key type is invalid")
        if not isinstance(self.generation_plan, LongGenerationPlan):
            raise TypeError("long-context generation plan type is invalid")
        if not isinstance(self.page_budget, LongGenerationPageBudget):
            raise TypeError("long-context page budget type is invalid")
        if type(self.recompute_objects) is not int or self.recompute_objects < 0:
            raise W08LongContextError("long-context recompute count is invalid")
        limits = W08_RESOURCE_BUDGET
        if (
            len(self.material.chunks) > limits["max_segments"]
            or len(self.material.seeds) > limits["max_records"]
            or len(self.centers) > limits["max_records"]
            or self.recall_budget.max_index_gets > limits["max_payload_gets"]
            or self.recall_budget.max_segment_payload_gets
            > limits["max_payload_gets"]
            or self.recall_budget.max_segment_payload_bytes
            > limits["max_payload_bytes"]
            or self.recall_budget.max_results > limits["max_records"]
            or len(self.generation_plan.items) >= limits["max_checkpoint_count"]
            or self.page_budget.max_units > limits["max_payload_bytes"]
            or self.page_budget.max_claims > limits["max_records"]
            or self.recompute_objects > limits["max_recompute_objects"]
        ):
            raise W08LongContextError("long-context execution budget exceeds W08 manifest")


class W08R06HierarchyOwner:
    owner_key = W08_LONG_CONTEXT_OWNER_KEYS[0]

    def __init__(
        self,
        spans: SpanIndex,
        protocol: LongInputHierarchyProtocol,
        builder: LongInputHierarchyBuilder | None = None,
    ) -> None:
        if not isinstance(spans, SpanIndex):
            raise TypeError("W08 hierarchy owner requires SpanIndex")
        if not isinstance(protocol, LongInputHierarchyProtocol):
            raise TypeError("W08 hierarchy owner requires LongInputHierarchyProtocol")
        self.spans = spans
        self.protocol = protocol
        self.builder = builder or LongInputHierarchyBuilder()

    def build(
        self, material: W08LongInputMaterialization
    ) -> tuple[LongInputHierarchy, tuple[tuple[object, object], ...]]:
        hierarchy = self.builder.build(
            material.chunks,
            material.scope,
            material.seeds,
            expected_document_digest=material.document_digest,
        )
        refs = self.builder.materialize(
            hierarchy,
            material.chunks,
            self.spans,
            self.protocol,
        )
        return hierarchy, refs


class W08AuthorizedCenterOwner:
    owner_key = W08_LONG_CONTEXT_OWNER_KEYS[2]

    def __init__(self, runtime: AuthorizedCenterAgendaRuntime) -> None:
        if not isinstance(runtime, AuthorizedCenterAgendaRuntime):
            raise TypeError("W08 page-in owner requires AuthorizedCenterAgendaRuntime")
        self.runtime = runtime

    def run(
        self,
        request: W08LongContextRequest,
        execution: W08LongContextExecution,
        centers: tuple[EvidenceFormedCenter, ...],
    ) -> AuthorizedCenterAgendaRun:
        by_record: dict[tuple[int, ...], list[EvidenceFormedCenter]] = {}
        for center in centers:
            by_record.setdefault(center.index_entry.record_key, []).append(center)
        lanes: list[list[EvidenceFormedCenter]] = [
            [] for _ in range(request.worker_count)
        ]
        for ordinal, record_key in enumerate(sorted(by_record)):
            lanes[ordinal % request.worker_count].extend(by_record[record_key])
        runs = tuple(
            self.runtime.run(
                tuple(lane),
                execution.current,
                execution.authorization,
                execution.recall_budget,
                reader_key_prefix=(
                    _NAMESPACE,
                    *request.request_key,
                    request.worker_count,
                    lane_index,
                ),
                current_policy_epoch=execution.authorization.policy_epoch,
            )
            for lane_index, lane in enumerate(lanes, start=1)
            if lane
        )
        if not runs:
            raise W08LongContextError("long-context worker schedule is empty")
        if len(runs) == 1:
            return runs[0]
        return AuthorizedCenterAgendaRun(
            execution.authorization.projection_key,
            tuple(state for run in runs for state in run.states),
            tuple(read for run in runs for read in run.record_reads),
        )


class W08PersistentAgendaOwner:
    owner_key = W08_LONG_CONTEXT_OWNER_KEYS[1]

    def __init__(self, store: PersistentConversationAgendaStore) -> None:
        if not isinstance(store, PersistentConversationAgendaStore):
            raise TypeError("W08 agenda owner requires PersistentConversationAgendaStore")
        self.store = store

    @staticmethod
    def _references(
        request: W08LongContextRequest,
        centers: tuple[EvidenceFormedCenter, ...],
    ) -> tuple[PersistentAgendaCenter, ...]:
        references = []
        previous: StableRecordKey | None = None
        for ordinal, center in enumerate(centers, start=1):
            dependencies = () if previous is None else (previous,)
            references.append(PersistentAgendaCenter.from_formed_center(
                center,
                StableRecordKey((_NAMESPACE, 10, ordinal, *request.request_key)),
                dependencies=dependencies,
                logical_seq=request.logical_seq,
            ))
            previous = center.center_key
        return tuple(sorted(references, key=lambda item: item.center_key))

    def open(
        self,
        request: W08LongContextRequest,
        agenda_key: StableRecordKey,
        centers: tuple[EvidenceFormedCenter, ...],
    ) -> tuple[PersistentConversationAgenda, tuple[EvidenceFormedCenter, ...]]:
        if request.mode == "fresh":
            agenda = self.store.create(
                agenda_key,
                self._references(request, centers),
            )
        else:
            agenda = self.store.load(agenda_key)
        binding = PersistentConversationAgendaRuntime.bind(agenda, centers)
        return agenda, binding.centers

    def record(
        self,
        request: W08LongContextRequest,
        agenda: PersistentConversationAgenda,
        run: AuthorizedCenterAgendaRun,
        checkpoint: LongGenerationCheckpoint,
    ) -> PersistentConversationAgenda:
        lifecycle = AGENDA_RESOLVED if checkpoint.status == LONG_GENERATION_COMPLETE else AGENDA_OPEN
        logical_seq = request.logical_seq + agenda.revision + 1
        return self.store.record_authorized_run(
            agenda,
            run,
            lifecycle_by_center=tuple(
                (state.center.center_key, lifecycle, logical_seq)
                for state in run.states
            ),
        )


LongContextPageBuilder = Callable[
    [LongGenerationCheckpoint, object, AuthorizedCenterAgendaRun, int],
    LongGenerationPageCommit,
]


class W08GenerationCheckpointOwner:
    owner_key = W08_LONG_CONTEXT_OWNER_KEYS[3]

    def __init__(
        self,
        store: LongGenerationCheckpointStore,
        page_builder: LongContextPageBuilder,
    ) -> None:
        if not isinstance(store, LongGenerationCheckpointStore):
            raise TypeError("W08 checkpoint owner requires LongGenerationCheckpointStore")
        if not callable(page_builder):
            raise TypeError("W08 checkpoint page builder must be callable")
        self.store = store
        self.page_builder = page_builder

    def run(
        self,
        request: W08LongContextRequest,
        execution: W08LongContextExecution,
        centers: AuthorizedCenterAgendaRun,
    ) -> LongGenerationCheckpoint:
        if request.mode == "fresh":
            checkpoint = self.store.create(execution.generation_plan)
        else:
            checkpoint = self.store.load(execution.generation_plan.answer_key)
        remaining = len(execution.generation_plan.items) - checkpoint.next_cursor
        commit_count = min(request.page_limit, remaining)
        if (
            checkpoint.revision + commit_count + 1
            > W08_RESOURCE_BUDGET["max_checkpoint_count"]
        ):
            raise W08LongContextError("long-context checkpoint budget was exhausted")
        committed = 0
        while (
            checkpoint.status != LONG_GENERATION_COMPLETE
            and committed < request.page_limit
        ):
            item = execution.generation_plan.items[checkpoint.next_cursor]
            page = self.page_builder(checkpoint, item, centers, committed)
            if not isinstance(page, LongGenerationPageCommit):
                raise TypeError("W08 checkpoint page builder returned an invalid page")
            checkpoint = self.store.commit_page(
                execution.generation_plan,
                page,
                execution.page_budget,
            ).checkpoint
            committed += 1
        return checkpoint


class W08LongContextConsumerOwner:
    owner_key = W08_LONG_CONTEXT_OWNER_KEYS[4]

    def __init__(
        self,
        mapper: Callable[
            [W08LongContextRequest, str, AuthorizedCenterAgendaRun, LongGenerationCheckpoint],
            W08LongContextUse,
        ],
    ) -> None:
        if not callable(mapper):
            raise TypeError("W08 long-context consumer mapper must be callable")
        self.mapper = mapper

    def consume(
        self,
        request: W08LongContextRequest,
        consumer_key: str,
        centers: AuthorizedCenterAgendaRun,
        checkpoint: LongGenerationCheckpoint,
    ) -> W08LongContextUse:
        result = self.mapper(request, consumer_key, centers, checkpoint)
        if not isinstance(result, W08LongContextUse):
            raise TypeError("W08 long-context consumer returned an invalid Use")
        return result


@dataclass(frozen=True)
class W08LongContextOwners:
    hierarchy: W08R06HierarchyOwner
    agenda: W08PersistentAgendaOwner
    centers: W08AuthorizedCenterOwner
    checkpoint: W08GenerationCheckpointOwner
    consumers: W08LongContextConsumerOwner

    def __post_init__(self) -> None:
        expected = (
            W08R06HierarchyOwner,
            W08PersistentAgendaOwner,
            W08AuthorizedCenterOwner,
            W08GenerationCheckpointOwner,
            W08LongContextConsumerOwner,
        )
        if any(not isinstance(value, kind) for value, kind in zip(
            (self.hierarchy, self.agenda, self.centers, self.checkpoint, self.consumers),
            expected,
        )):
            raise TypeError("W08 long-context owner inventory is invalid")


__all__ = [
    "W08AuthorizedCenterOwner",
    "W08GenerationCheckpointOwner",
    "W08LongContextConsumerOwner",
    "W08LongContextExecution",
    "W08LongContextOwners",
    "W08LongInputMaterialization",
    "W08PersistentAgendaOwner",
    "W08R06HierarchyOwner",
    "materialize_w08_long_input",
]
