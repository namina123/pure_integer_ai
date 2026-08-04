"""W08-05 long-context integration over W08 train material and public owners."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OwnerScope,
    ParserVersion,
    VISIBILITY_SESSION,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_QUERY,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.understanding.span_index import SpanIndex, SpanProtocol
from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterAgendaRuntime,
    CenterAuthorizationBinding,
    CenterAuthorizationProjection,
)
from pure_integer_ai.experiments.authorized_generation_delivery import (
    DELIVERY_POSTCHECK_FAILED,
    AuthorizedGenerationClaim,
    AuthorizedGenerationDeliveryAuthority,
    AuthorizedGenerationDeliveryDecision,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    EvidenceFormedCenter,
    RecallIndexEntry,
    TypedRecallPayload,
    TypedRecallRecordCodec,
)
from pure_integer_ai.experiments.long_generation_checkpoint import (
    OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
    LongGenerationCheckpointError,
    LongGenerationCheckpointStore,
    LongGenerationPageBudget,
    LongGenerationPageCommit,
    LongGenerationPlan,
    LongGenerationPlanItem,
)
from pure_integer_ai.experiments.long_input_hierarchy import (
    LongInputChunk,
    LongInputHierarchyError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import RecallBudget
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DIRECTIONS,
    DirectionalCenterAdapterConfig,
    DirectionalCenterProfile,
    DirectionalMemoryCenterAdapter,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_RESOURCE_BUDGET,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_long_context import W08LongContextFacade
from pure_integer_ai.experiments.ph2_w08_long_context_adapters import (
    W08AuthorizedCenterOwner,
    W08GenerationCheckpointOwner,
    W08LongContextConsumerOwner,
    W08LongContextExecution,
    W08LongContextOwners,
    W08PersistentAgendaOwner,
    W08R06HierarchyOwner,
    materialize_w08_long_input,
)
from pure_integer_ai.experiments.ph2_w08_long_context_contract import (
    W08LongContextError,
    W08LongContextRequest,
    W08LongContextResourceReceipt,
    W08LongContextUse,
    assess_w08_long_context_ablation,
)
from pure_integer_ai.experiments.persistent_conversation_agenda import (
    OBJECT_KIND_CONVERSATION_AGENDA,
    PersistentConversationAgendaStore,
)
from pure_integer_ai.experiments.ph2_w08_long_context_training import (
    W08LongContextTrainingBundle,
    compile_w08_long_context_training,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import OpenHotDelta, SegmentBudget, SegmentRecord
from pure_integer_ai.storage.segment_repository import (
    BackendObjectRepository,
    InMemoryObjectRepository,
    OBJECT_KIND_SEGMENT,
)
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore
from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.integer_codec import decode_integer_tuple
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_question_answer_runtime import _fixture as question_fixture


ROOT = Path(__file__).resolve().parents[1]
_OWNER = OwnerScope(80805, 1, 1, VISIBILITY_SESSION)
_DESCRIPTOR = MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY
_HOT = (80805, 21, 1)
_COLD = (80805, 21, 2)
_PROFILE = TemperatureProfile(
    (80805, 22, 1),
    (TemperatureTier(_HOT, 0), TemperatureTier(_COLD, 1)),
)


class _NeverPrefetch:
    def should_prefetch(self, context):
        del context
        return False

    def state_key(self):
        return (80805, 23, 1)


class _CountingRepository(InMemoryObjectRepository):
    def __init__(self):
        super().__init__()
        self.segment_gets = 0

    def get(self, object_kind, identity_key):
        if object_kind == OBJECT_KIND_SEGMENT:
            self.segment_gets += 1
        return super().get(object_kind, identity_key)


@pytest.fixture(scope="module")
def training_bundle():
    contract = open_w08_frozen_contract(ROOT)
    payload = W08PayloadFirewall.open(
        ROOT,
        contract,
        make_w08_request(contract),
    ).read_training_payload()
    return compile_w08_long_context_training(payload)


def _current(ctx, source, *, ordinal=0):
    session = session_scope(1, owner=source.owner, versions=source.versions, source=source)
    document = document_scope(source, parent=session)
    episode = episode_scope(1, parent=document)
    query = query_scope(1, parent=episode)
    ctx.work_memory.begin_session(session)
    ctx.work_memory.begin_document(document)
    ctx.work_memory.begin_episode(episode)
    ctx.work_memory.begin_query(query)
    ontology = ctx.graph_ontology
    occurrence = ontology.materialize(occurrence_identity(source, start=ordinal, end=ordinal + 1, ordinal=ordinal))
    span = ontology.materialize(span_identity(
        source, members=((ordinal, ordinal + 1),), ordinal=ordinal))
    semantic = ontology.materialize(proposition_identity(source, (80805, 30, ordinal + 1)))
    structure = ontology.materialize(structure_concept_identity(
        (80805, 31, ordinal + 1), owner=source.owner, versions=source.versions))
    timestamp = LogicalClock(LogicalClockIdentity(query, CLOCK_QUERY)).advance()
    return MemoryCurrentQuery(
        query,
        source,
        timestamp,
        (occurrence,),
        (span,),
        (semantic,),
        (structure,),
        concept_identity((80805, 32, 1), owner=source.owner, versions=source.versions),
        concept_identity((80805, 32, 2), owner=source.owner, versions=source.versions),
        concept_identity((80805, 32, 3), owner=source.owner, versions=source.versions),
        concept_identity((80805, 32, 4), owner=source.owner, versions=source.versions),
    )


def _bound(source, *, start, end):
    return BoundProposition(
        proposition_identity(source, (80805, 40, 1)),
        minimal_instruction_identity((80805, 40, 2)),
        concept_identity((80805, 40, 3)),
        structure_concept_identity((80805, 40, 4)),
        occurrence_identity(source, start=start, end=end, ordinal=0),
        context_scope_identity(source, (80805, 40, 5)),
        (),
        (),
        (),
    )


def _md03():
    profiles = tuple(
        DirectionalCenterProfile(
            direction,
            StableRecordKey((80805, 50, ordinal, 1)),
            StableRecordKey((80805, 50, ordinal, 2)),
            StableRecordKey((80805, 50, ordinal, 3)),
        )
        for ordinal, direction in enumerate(DIRECTIONS, start=1)
    )
    return DirectionalMemoryCenterAdapter(DirectionalCenterAdapterConfig(profiles))


def _storage(payload, source, scope, *, record_key):
    repository = _CountingRepository()
    store = TieredSegmentStore(repository, build_storage_role_registry(), _PROFILE)
    delta = OpenHotDelta(
        _DESCRIPTOR,
        (80805, 60, 1),
        (),
        SegmentBudget(1, 1_000_000),
    )
    delta.append(SegmentRecord(
        record_key,
        TypedRecallRecordCodec.encode(record_key, payload).payload,
    ))
    store.publish_delta(
        delta,
        segment_key=(80805, 61, record_key[-1]),
        tier_key=_COLD,
        read_fence=record_key[-1],
        manifest_key=(80805, 62, record_key[-1]),
        migration_key=(80805, 63, record_key[-1]),
    )
    return store, repository


def _fixture_bundle(training_bundle: W08LongContextTrainingBundle, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    material = materialize_w08_long_input(
        training_bundle,
        owner=_OWNER,
        chunk_width=19,
    )
    graph_backend = DictBackend()
    ctx = make_train_context(graph_backend)
    current = _current(ctx, material.source)
    md03 = _md03()
    md03_center = md03.from_understanding(current, current.spans[0], strength="CONDITIONAL")
    target = _bound(material.source, start=0, end=len(training_bundle.material[0].surface))
    mapper, postchecker, _, _, _ = _postcheck_owners()
    qa = question_fixture(
        EVIDENCE_SUPPORT,
        world=(material.source, document_scope(material.source), target),
        answer_text=training_bundle.material[0].surface,
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    evidence = qa.evidence[0]
    recall_payload = TypedRecallPayload(
        target,
        (evidence,),
        0,
        len(training_bundle.material[0].surface),
        (StableRecordKey((80805, 72, 2)),),
    )
    record_key = (80805, 71, 1)
    store, repository = _storage(recall_payload, material.source, document_scope(material.source), record_key=record_key)
    from pure_integer_ai.storage.query_hot_set import QueryHotSetPolicy
    policy = QueryHotSetPolicy(
        SegmentBudget(4, 1_000_000),
        SegmentBudget(4, 1_000_000),
        _NeverPrefetch(),
        8,
    )
    from pure_integer_ai.experiments.free_text_recall_runtime import AclFirstExactRecallReader
    reader = AclFirstExactRecallReader(store, _DESCRIPTOR, policy)
    scope = document_scope(material.source)
    entry = RecallIndexEntry(
        record_key,
        material.source,
        scope,
        (StableRecordKey((80805, 72, 1)),),
        (StableRecordKey((80805, 72, 2)),),
    )
    center_one = EvidenceFormedCenter(
        StableRecordKey((80805, 73, 1)),
        entry,
        (evidence.evidence_id,),
        md03_center,
    )
    center_two = replace(center_one, center_key=StableRecordKey((80805, 73, 2)))
    center_distractor = replace(center_one, center_key=StableRecordKey((80805, 73, 3)))
    manifest = store.current_manifest()
    assert manifest is not None
    location = next(item for item in manifest.entries if item.segment_key == (80805, 61, 1))
    bindings = tuple(
        CenterAuthorizationBinding(
            center.center_key,
            _DESCRIPTOR,
            record_key,
            material.source,
            scope,
            location.version_key,
            material.source.owner,
            location.segment_key,
        )
        for center in (center_one, center_two, center_distractor)
    )
    authorization = CenterAuthorizationProjection(
        StableRecordKey((80805, 74, 1)),
        (80805, 75, 1),
        1,
        manifest.manifest_key,
        manifest.publish_epoch,
        MemoryAccessContext(80805, 1, 1),
        bindings,
    )
    span_index = SpanIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        SpanProtocol((80805, 76, 1), (80805, 76, 2), (80805, 76, 3), (80805, 76, 4)),
    )
    from pure_integer_ai.experiments.long_input_hierarchy import LongInputHierarchyProtocol
    hierarchy_protocol = LongInputHierarchyProtocol(
        structure_concept_identity((80805, 77, 1)),
        structure_concept_identity((80805, 77, 2)),
        structure_concept_identity((80805, 77, 3)),
        structure_concept_identity((80805, 77, 4)),
    )
    plan = LongGenerationPlan(
        StableRecordKey((80805, 78, 1)),
        (
            LongGenerationPlanItem(StableRecordKey((80805, 78, 2)), target.stable_key()),
            LongGenerationPlanItem(StableRecordKey((80805, 78, 3)), target.stable_key()),
        ),
    )
    execution = W08LongContextExecution(
        material,
        (center_one, center_two, center_distractor),
        current,
        authorization,
        RecallBudget(2, 2, 1_000_000, 1),
        StableRecordKey((80805, 79, 1)),
        plan,
        LongGenerationPageBudget(1_000, 1),
    )
    agenda_path = tmp_path / "agenda.sqlite3"
    checkpoint_path = tmp_path / "checkpoint.sqlite3"
    agenda_backend = SQLiteBackend(str(agenda_path))
    checkpoint_backend = SQLiteBackend(str(checkpoint_path))
    agenda_store = PersistentConversationAgendaStore(
        BackendObjectRepository(agenda_backend), agenda_backend.commit)
    checkpoint_store = LongGenerationCheckpointStore(
        BackendObjectRepository(checkpoint_backend), checkpoint_backend.commit)
    qa_run = qa.runtime.run(qa.request)
    authority = AuthorizedGenerationDeliveryAuthority()

    def page_builder(checkpoint, item, centers, page_index):
        del page_index
        claim = AuthorizedGenerationClaim.from_authorized_center(
            qa_run.planning_request.candidates[0], centers.states[0])
        decision = authority.authorize(qa_run, (claim,))
        return LongGenerationPageCommit(
            StableRecordKey((80805, 80, checkpoint.next_cursor + 1)),
            (item.item_key,),
            StableRecordKey((80805, 81, 1)),
            StableRecordKey((80805, 81, 2)),
            checkpoint.revision,
            checkpoint.next_cursor,
            checkpoint.prefix_digest,
            (),
            (),
            decision,
            (centers.states[0],),
        )

    def consumer(request, consumer_key, centers, checkpoint):
        del checkpoint
        state = centers.states[0]
        return W08LongContextUse(
            consumer_key,
            request.request_key,
            tuple(sorted(item.center.center_key for item in centers.states)),
            tuple(sorted(citation.record_key for read in centers.record_reads for citation in read.exact.receipt.citations)),
            (80805, 82, W08_CONSUMER_KEYS.index(consumer_key) + 1),
            (80805, 83, W08_CONSUMER_KEYS.index(consumer_key) + 1),
            "RESOLVED",
        )

    owners = W08LongContextOwners(
        W08R06HierarchyOwner(span_index, hierarchy_protocol),
        W08PersistentAgendaOwner(agenda_store),
        W08AuthorizedCenterOwner(AuthorizedCenterAgendaRuntime(reader)),
        W08GenerationCheckpointOwner(checkpoint_store, page_builder),
        W08LongContextConsumerOwner(consumer),
    )
    return {
        "backend": graph_backend,
        "ctx": ctx,
        "qa": qa,
        "store": store,
        "repository": repository,
        "agenda_backend": agenda_backend,
        "checkpoint_backend": checkpoint_backend,
        "agenda_path": agenda_path,
        "checkpoint_path": checkpoint_path,
        "execution": execution,
        "owners": owners,
        "required": (center_one.center_key, center_two.center_key),
        "distractor": center_distractor.center_key,
    }


def _close_fixture(fixture):
    fixture["ctx"].work_memory.end_query()
    fixture["ctx"].work_memory.end_episode()
    fixture["ctx"].work_memory.end_document()
    fixture["ctx"].work_memory.end_session()
    fixture["qa"].close()
    fixture["agenda_backend"].close()
    fixture["checkpoint_backend"].close()
    fixture["backend"].close()


def _reopen_metadata_owners(fixture):
    fixture["agenda_backend"].close()
    fixture["checkpoint_backend"].close()
    agenda_backend = SQLiteBackend(str(fixture["agenda_path"]))
    checkpoint_backend = SQLiteBackend(str(fixture["checkpoint_path"]))
    agenda_store = PersistentConversationAgendaStore(
        BackendObjectRepository(agenda_backend), agenda_backend.commit)
    checkpoint_store = LongGenerationCheckpointStore(
        BackendObjectRepository(checkpoint_backend), checkpoint_backend.commit)
    previous = fixture["owners"]
    owners = W08LongContextOwners(
        previous.hierarchy,
        W08PersistentAgendaOwner(agenda_store),
        previous.centers,
        W08GenerationCheckpointOwner(
            checkpoint_store,
            previous.checkpoint.page_builder,
        ),
        previous.consumers,
    )
    fixture["agenda_backend"] = agenda_backend
    fixture["checkpoint_backend"] = checkpoint_backend
    fixture["owners"] = owners
    return owners


def _request(
    bundle,
    required,
    *,
    worker_count=1,
    mode="fresh",
    page_limit=2,
    **kwargs,
):
    return W08LongContextRequest(
        (80805, 90, 1),
        bundle.material_key,
        worker_count,
        mode,
        tuple(sorted({
            "max_checkpoint_count": 2048,
            "max_logic_operations": 8000000,
            "max_payload_bytes": 536870912,
            "max_payload_gets": 524288,
            "max_recompute_objects": 800000,
            "max_records": 800000,
            "max_segments": 32768,
            "max_workers": 4,
        }.items())),
        tuple(sorted(required)),
        page_limit=page_limit,
        **kwargs,
    )


def test_w08_long_context_training_material_is_visible_and_typed(training_bundle):
    assert training_bundle.audit.observation_count == 63
    assert training_bundle.audit.evidence_binding_count == 63
    assert training_bundle.audit.material_item_count == 59
    assert training_bundle.audit.source_surface_count == 4
    assert training_bundle.audit.multi_center_signal == 1
    assert training_bundle.audit.cold_relevant_signal == 1
    assert training_bundle.audit.conflict_signal == 1
    assert training_bundle.audit.clarification_signal == 1
    assert training_bundle.audit.authored_answer_read_count == 0


def test_w08_long_context_full_run_has_shared_cold_pagein_agenda_checkpoint_and_urg(
    training_bundle, tmp_path
):
    fixture = _fixture_bundle(training_bundle, tmp_path)
    try:
        request = _request(
            training_bundle,
            fixture["required"],
            distractor_center_keys=(fixture["distractor"],),
        )
        result = W08LongContextFacade(fixture["owners"]).execute(
            request, fixture["execution"])
        assert result.state == "RESOLVED"
        assert result.resources.real_consumers == 3
        assert result.resources.agenda_entries == 2
        assert result.resources.opened_segments == 1
        assert result.resources.opened_pages == 1
        assert result.resources.page_in_records == 1
        assert result.resources.payload_gets == 1
        assert result.resources.checkpoint_count == 3
        assert result.trace is not None
        assert result.trace.center_keys == tuple(sorted(fixture["required"]))
        assert len(result.trace.prefix_content_digests) == 59
        assert result.trace.checkpoint_cursor == 2
        assert result.owner_calls[-1] == "PH2-W08-TYPED-CONSUMERS"
    finally:
        _close_fixture(fixture)


@pytest.mark.parametrize("component", ("HIERARCHY", "PERSISTENT_AGENDA", "COLD_PAGE_IN", "GENERATION_CHECKPOINT"))
def test_w08_long_context_internal_ablations_fail_closed(training_bundle, tmp_path, component):
    fixture = _fixture_bundle(training_bundle, tmp_path / component)
    try:
        flags = tuple(
            (key, int(key != component))
            for key in ("HIERARCHY", "PERSISTENT_AGENDA", "COLD_PAGE_IN", "GENERATION_CHECKPOINT")
        )
        request = _request(training_bundle, fixture["required"], component_flags=flags)
        result = W08LongContextFacade(fixture["owners"]).execute(request, fixture["execution"])
        assert result.state != "RESOLVED"
        assert result.blocked_component == component
        assert result.resources.payload_gets == 0
    finally:
        _close_fixture(fixture)


def test_w08_long_context_unknown_clarify_and_budget_are_bounded(training_bundle, tmp_path):
    fixture = _fixture_bundle(training_bundle, tmp_path / "cases")
    try:
        facade = W08LongContextFacade(fixture["owners"])
        unknown = facade.execute(_request(training_bundle, ()), fixture["execution"])
        assert unknown.state == "UNKNOWN"
        clarify = facade.execute(
            _request(
                training_bundle,
                fixture["required"],
                clarification_candidate_keys=(
                    StableRecordKey((80805, 91, 1)),
                    StableRecordKey((80805, 91, 2)),
                ),
            ),
            fixture["execution"],
        )
        assert clarify.state == "CLARIFY"
        budget = facade.execute(
            _request(training_bundle, fixture["required"], page_limit=1),
            replace(fixture["execution"], agenda_key=StableRecordKey((80805, 79, 2)), generation_plan=replace(
                fixture["execution"].generation_plan,
                answer_key=StableRecordKey((80805, 78, 4)),
            )),
        )
        assert budget.state == "BUDGET_EXHAUSTED"
        assert budget.resources.opened_pages == 1
    finally:
        _close_fixture(fixture)


def test_w08_long_context_acl_denial_precedes_payload_get(training_bundle, tmp_path):
    fixture = _fixture_bundle(training_bundle, tmp_path / "acl-denied")
    try:
        fixture["repository"].segment_gets = 0
        denied = replace(
            fixture["execution"].authorization,
            access=MemoryAccessContext(80805, 1, 2),
        )
        result = W08LongContextFacade(fixture["owners"]).execute(
            _request(training_bundle, fixture["required"]),
            replace(fixture["execution"], authorization=denied),
        )
        assert result.state == "ACCESS_BLOCKED"
        assert result.resources.payload_gets == 0
        assert result.resources.opened_pages == 0
        assert fixture["repository"].segment_gets == 0
        assert result.owner_calls == (
            "R-06-LONG-INPUT-HIERARCHY",
            "R-06-PERSISTENT-CONVERSATION-AGENDA",
            "R-04-AUTHORIZED-CENTER-K04-PAGE-IN",
        )
    finally:
        _close_fixture(fixture)


def test_w08_long_context_conflict_requires_explicit_resolution(
    training_bundle,
    tmp_path,
):
    fixture = _fixture_bundle(training_bundle, tmp_path / "conflict")
    try:
        conflict_key = fixture["required"][0]
        unresolved = W08LongContextFacade(fixture["owners"]).execute(
            _request(
                training_bundle,
                fixture["required"],
                conflict_center_keys=(conflict_key,),
            ),
            fixture["execution"],
        )
        assert unresolved.state == "CLARIFY"
        assert unresolved.owner_calls == ()
        resolved = W08LongContextFacade(fixture["owners"]).execute(
            _request(
                training_bundle,
                fixture["required"],
                conflict_center_keys=(conflict_key,),
                resolved_conflict_keys=(conflict_key,),
            ),
            fixture["execution"],
        )
        assert resolved.state == "RESOLVED"
        assert resolved.trace is not None
        assert resolved.trace.center_keys == tuple(sorted(fixture["required"]))
    finally:
        _close_fixture(fixture)


def test_w08_long_context_workers_are_canonically_equivalent(
    training_bundle,
    tmp_path,
):
    keys = []
    scheduling_keys = []
    for worker_count in (1, 2, 4):
        fixture = _fixture_bundle(
            training_bundle,
            tmp_path / f"worker-{worker_count}",
        )
        try:
            request = _request(
                training_bundle,
                fixture["required"],
                worker_count=worker_count,
            )
            scheduling_keys.append(request.scheduling_key())
            result = W08LongContextFacade(fixture["owners"]).execute(
                request,
                fixture["execution"],
            )
            assert result.state == "RESOLVED"
            keys.append(result.canonical_key())
        finally:
            _close_fixture(fixture)
    assert len(set(keys)) == 1
    assert len(set(scheduling_keys)) == 3


@pytest.mark.parametrize("resume_mode", ("restart", "resume"))
def test_w08_long_context_reopened_resume_matches_fresh_canonical_outcome(
    training_bundle,
    tmp_path,
    resume_mode,
):
    full_fixture = _fixture_bundle(
        training_bundle,
        tmp_path / resume_mode / "full",
    )
    try:
        full = W08LongContextFacade(full_fixture["owners"]).execute(
            _request(training_bundle, full_fixture["required"]),
            full_fixture["execution"],
        )
    finally:
        _close_fixture(full_fixture)

    partial_fixture = _fixture_bundle(
        training_bundle,
        tmp_path / resume_mode / "partial",
    )
    try:
        partial = W08LongContextFacade(partial_fixture["owners"]).execute(
            _request(
                training_bundle,
                partial_fixture["required"],
                page_limit=1,
            ),
            partial_fixture["execution"],
        )
        assert partial.state == "BUDGET_EXHAUSTED"
        owners = _reopen_metadata_owners(partial_fixture)
        agenda = owners.agenda.store.load(
            partial_fixture["execution"].agenda_key,
        )
        checkpoint = owners.checkpoint.store.load(
            partial_fixture["execution"].generation_plan.answer_key,
        )
        assert agenda.revision == 1
        assert checkpoint.revision == 1
        assert checkpoint.next_cursor == 1
        assert len(owners.agenda.store.repository.list_kind(
            OBJECT_KIND_CONVERSATION_AGENDA,
        )) == 2
        assert len(owners.checkpoint.store.repository.list_kind(
            OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
        )) == 2
        resumed = W08LongContextFacade(owners).execute(
            _request(
                training_bundle,
                partial_fixture["required"],
                mode=resume_mode,
            ),
            partial_fixture["execution"],
        )
        assert resumed.state == "RESOLVED"
        assert resumed.canonical_key() == full.canonical_key()
        assert resumed.trace is not None
        assert full.trace is not None
        assert resumed.trace.canonical_key() != full.trace.canonical_key()
        assert resumed.trace.checkpoint_prefix_digest == (
            full.trace.checkpoint_prefix_digest
        )
    finally:
        _close_fixture(partial_fixture)


def test_w08_long_context_agenda_repository_contains_metadata_only(
    training_bundle,
    tmp_path,
):
    fixture = _fixture_bundle(training_bundle, tmp_path / "metadata-only")
    try:
        result = W08LongContextFacade(fixture["owners"]).execute(
            _request(training_bundle, fixture["required"]),
            fixture["execution"],
        )
        assert result.state == "RESOLVED"
        store = fixture["owners"].agenda.store
        agenda = store.load(fixture["execution"].agenda_key)
        assert agenda.revision == 1
        assert all(not hasattr(center, "payload") for center in agenda.centers)
        assert set(agenda.centers[0].__dataclass_fields__) == {
            "center_key",
            "query_key",
            "record_key",
            "dependencies",
            "lifecycle",
            "last_logical_seq",
            "consumer_receipt_keys",
        }
        surface = tuple(map(ord, training_bundle.material[0].surface))
        for descriptor in store.repository.list_kind(
            OBJECT_KIND_CONVERSATION_AGENDA,
        ):
            values = decode_integer_tuple(store.repository.get(
                OBJECT_KIND_CONVERSATION_AGENDA,
                descriptor.identity_key,
            ))
            assert not any(
                values[index:index + len(surface)] == surface
                for index in range(len(values) - len(surface) + 1)
            )
    finally:
        _close_fixture(fixture)


def test_w08_long_context_hierarchy_and_material_identity_drift_fail_closed(
    training_bundle,
    tmp_path,
):
    fixture = _fixture_bundle(training_bundle, tmp_path / "hierarchy-drift")
    try:
        fixture["repository"].segment_gets = 0
        material = fixture["execution"].material
        first = material.chunks[0]
        replacement_text = (
            ("X" if first.text[0] != "X" else "Y") + first.text[1:]
        )
        content_drift = LongInputChunk.from_text(
            first.source,
            first.absolute_start,
            replacement_text,
        )
        with pytest.raises(W08LongContextError, match="source/content digest"):
            replace(
                material,
                chunks=(content_drift, *material.chunks[1:]),
            )

        source_drift = replace(
            material.source,
            document_id=material.source.document_id + 1,
        )
        source_chunk = LongInputChunk.from_text(
            source_drift,
            first.absolute_start,
            first.text,
        )
        with pytest.raises(W08LongContextError, match="chunk identity"):
            replace(material, chunks=(source_chunk, *material.chunks[1:]))

        with pytest.raises(LongInputHierarchyError, match="content/prefix digest"):
            fixture["owners"].hierarchy.builder.build(
                material.chunks,
                material.scope,
                material.seeds,
                expected_document_digest=(0,) * 32,
            )

        parser_source = replace(
            material.source,
            versions=replace(
                material.source.versions,
                parser=ParserVersion(material.source.versions.parser.value + 1),
            ),
        )
        parser_material = replace(
            material,
            source=parser_source,
            scope=document_scope(parser_source),
            chunks=tuple(
                LongInputChunk.from_text(
                    parser_source,
                    chunk.absolute_start,
                    chunk.text,
                )
                for chunk in material.chunks
            ),
        )
        with pytest.raises(LongInputHierarchyError, match="seed.*source"):
            fixture["owners"].hierarchy.build(parser_material)

        request = replace(
            _request(training_bundle, fixture["required"]),
            training_material_key=(80805, 999, 1),
        )
        with pytest.raises(W08LongContextError, match="request/material"):
            W08LongContextFacade(fixture["owners"]).execute(
                request,
                fixture["execution"],
            )
        assert fixture["repository"].segment_gets == 0
    finally:
        _close_fixture(fixture)


def test_w08_long_context_failed_postcheck_does_not_advance_checkpoint(
    training_bundle,
    tmp_path,
):
    fixture = _fixture_bundle(training_bundle, tmp_path / "postcheck-failure")
    try:
        checkpoint_owner = fixture["owners"].checkpoint

        def failed_postcheck(checkpoint, item, centers, page_index):
            page = checkpoint_owner.page_builder(
                checkpoint,
                item,
                centers,
                page_index,
            )
            return replace(
                page,
                delivery=AuthorizedGenerationDeliveryDecision(
                    DELIVERY_POSTCHECK_FAILED,
                    (80805, 998, 1),
                ),
            )

        failing_owner = W08GenerationCheckpointOwner(
            checkpoint_owner.store,
            failed_postcheck,
        )
        owners = replace(fixture["owners"], checkpoint=failing_owner)
        with pytest.raises(LongGenerationCheckpointError, match="postcheck|envelope"):
            W08LongContextFacade(owners).execute(
                _request(training_bundle, fixture["required"]),
                fixture["execution"],
            )
        checkpoint = failing_owner.store.load(
            fixture["execution"].generation_plan.answer_key,
        )
        agenda = owners.agenda.store.load(fixture["execution"].agenda_key)
        assert checkpoint.revision == 0
        assert checkpoint.next_cursor == 0
        assert agenda.revision == 0
        assert len(failing_owner.store.repository.list_kind(
            OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
        )) == 1
        assert len(owners.agenda.store.repository.list_kind(
            OBJECT_KIND_CONVERSATION_AGENDA,
        )) == 1
    finally:
        _close_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "limit"),
    (
        ("opened_segments", W08_RESOURCE_BUDGET["max_segments"]),
        ("opened_pages", W08_RESOURCE_BUDGET["max_payload_gets"]),
        ("page_in_records", W08_RESOURCE_BUDGET["max_records"]),
        ("payload_gets", W08_RESOURCE_BUDGET["max_payload_gets"]),
        ("payload_bytes", W08_RESOURCE_BUDGET["max_payload_bytes"]),
        ("agenda_entries", W08_RESOURCE_BUDGET["max_records"]),
        ("real_consumers", len(W08_CONSUMER_KEYS)),
        ("recompute_objects", W08_RESOURCE_BUDGET["max_recompute_objects"]),
        ("logic_operations", W08_RESOURCE_BUDGET["max_logic_operations"]),
        ("checkpoint_count", W08_RESOURCE_BUDGET["max_checkpoint_count"]),
    ),
)
def test_w08_long_context_resource_ceiling_counterexamples(field, limit):
    values = {
        "opened_segments": 0,
        "opened_pages": 0,
        "page_in_records": 0,
        "payload_gets": 0,
        "payload_bytes": 0,
        "agenda_entries": 0,
        "real_consumers": 0,
        "recompute_objects": 0,
        "logic_operations": 0,
        "checkpoint_count": 0,
        "stop_reason": "BUDGET_EXHAUSTED",
    }
    values[field] = limit + 1
    with pytest.raises(W08LongContextError, match="resource budget"):
        W08LongContextResourceReceipt(**values)


def test_w08_long_context_page_limit_is_bounded_by_manifest(training_bundle):
    with pytest.raises(W08LongContextError, match="checkpoint budget"):
        _request(
            training_bundle,
            (),
            page_limit=W08_RESOURCE_BUDGET["max_checkpoint_count"] + 1,
        )


def test_w08_long_context_execution_budget_fails_before_owner_calls(
    training_bundle,
    tmp_path,
):
    fixture = _fixture_bundle(training_bundle, tmp_path / "execution-budget")
    try:
        fixture["repository"].segment_gets = 0
        with pytest.raises(W08LongContextError, match="execution budget"):
            replace(
                fixture["execution"],
                recompute_objects=(
                    W08_RESOURCE_BUDGET["max_recompute_objects"] + 1
                ),
            )
        assert fixture["repository"].segment_gets == 0
        assert fixture["owners"].agenda.store.repository.list_kind(
            OBJECT_KIND_CONVERSATION_AGENDA,
        ) == ()
        assert fixture["owners"].checkpoint.store.repository.list_kind(
            OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
        ) == ()
    finally:
        _close_fixture(fixture)


def test_w08_long_context_ablation_is_orthogonal():
    full = {
        "W-08-CHINESE_VARIATION": "PASS",
        "W-08-DISCOURSE": "PASS",
        "W-08-LOCAL_RECOMPUTE": "PASS",
        "W-08-LONG_CONTEXT": "PASS",
        "W-08-P3IA": "PASS",
    }
    ablated = dict(full, **{"W-08-LONG_CONTEXT": "NE"})
    report = assess_w08_long_context_ablation(
        internal_component="COLD_PAGE_IN",
        full_dimension_outcomes=full,
        ablated_dimension_outcomes=ablated,
    )
    assert report.affected_dimensions == ("W-08-LONG_CONTEXT",)
    assert report.unaffected_dimensions == tuple(
        key for key in full if key != "W-08-LONG_CONTEXT"
    )
