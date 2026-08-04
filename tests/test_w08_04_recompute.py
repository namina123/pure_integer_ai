"""W08-04 typed dependency 局部重算、A-03/A-08 与恢复专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.free_text_revision_runtime import (
    FreeTextDerivedDependency,
    FreeTextRevisionInvalidator,
)
from pure_integer_ai.experiments.parser_revision_runtime import ParserRevisionResult
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    make_w08_request,
    open_w08_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w08_firewall import W08PayloadFirewall
from pure_integer_ai.experiments.ph2_w08_recompute import (
    W08LocalRecomputeInjectedFailure,
    W08LocalRecomputeRuntime,
    W08_LOCAL_RECOMPUTE_FAULT_POINTS,
)
from pure_integer_ai.experiments.ph2_w08_recompute_adapters import (
    W08A03ParserRevisionOwner,
    W08A08MemoryRevisionOwner,
    W08FreeTextRevisionOwner,
)
from pure_integer_ai.experiments.ph2_w08_recompute_contract import (
    W08ConsumerRevalidation,
    W08LocalObjectState,
    W08LocalRecomputeDump,
    W08LocalRecomputeError,
    W08LocalRevisionRequest,
    W08LocalSnapshot,
    W08RevisionMapping,
    W08_LOCAL_REVISION_KINDS,
    W08_LOCAL_STATE_CHANNELS,
    W08_REVISION_MAPPING_SHAPES,
    assess_w08_local_recompute_ablation,
)
from pure_integer_ai.experiments.ph2_w08_recompute_training import (
    audit_w08_recompute_training,
)
from pure_integer_ai.cognition.shared.situation_state import (
    SituationProjectionReplacement,
)
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentError,
)
from pure_integer_ai.storage.backend import DictBackend
from tests.test_a03_parser_revision import _world as _a03_world
from tests.test_a08_memory_reparse import (
    _LICENSE as _A08_LICENSE,
    _TEXT as _A08_TEXT,
    _NewParser as _A08NewParser,
    _RejectParser as _A08RejectParser,
    _world as _a08_world,
)
from tests.test_d02_md02_situation_state_adapter import (
    _close as _close_situation,
    _fixture as _situation_fixture,
    _observation_event,
    _replacement,
    _update,
)
from tests.test_w08_03_discourse import _facade, _request


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def training_payload():
    contract = open_w08_frozen_contract(ROOT)
    return W08PayloadFirewall.open(
        ROOT,
        contract,
        make_w08_request(contract),
    ).read_training_payload()


def _key(seed: int, ordinal: int) -> tuple[int, ...]:
    return seed, ordinal


def _channel(seed: int, ordinal: int) -> tuple[tuple[int, ...], ...]:
    return (_key(seed, ordinal),)


def _object_state(entry, *, ordinal: int) -> W08LocalObjectState:
    kinds = ("HIERARCHY", "CENTER", "CLAIM")
    return W08LocalObjectState(
        _key(29100, ordinal),
        kinds[ordinal - 1],
        entry.dependencies,
        (entry.projection_key,),
        _channel(29110, ordinal),
        _channel(29120, ordinal),
        _channel(29130, ordinal),
        _channel(29140, ordinal),
        _channel(29150, ordinal),
        _channel(29160, ordinal),
        _channel(29170, ordinal),
        _channel(29180, ordinal),
        entry.stable_key(),
    )


def _rebuilt_state(request, affected, replacement) -> tuple[W08LocalObjectState, ...]:
    old = affected[0]
    return (
        W08LocalObjectState(
            request.target_keys[0],
            old.object_kind,
            replacement.entry.dependencies,
            old.projection_keys,
            _channel(29210, 1),
            _channel(29220, 1),
            _channel(29230, 1),
            _channel(29240, 1),
            _channel(29250, 1),
            _channel(29260, 1),
            _channel(29270, 1),
            _channel(29280, 1),
            replacement.entry.stable_key(),
        ),
    )


def _revalidated_consumers(request, affected, after_snapshot):
    del after_snapshot
    channel = {
        "UNDERSTANDING": "UNDERSTANDING_USE",
        "REASONING": "REASONING_USE",
        "GENERATION": "GENERATION_USE",
    }
    return tuple(
        W08ConsumerRevalidation(
            consumer,
            tuple(sorted(
                key
                for item in affected
                for key in item.channel_keys(channel[consumer])
            )),
            _key(29300, ordinal),
            request.target_keys[0],
            (_key(29310, ordinal),),
            "RESOLVED",
        )
        for ordinal, consumer in enumerate(W08_CONSUMER_KEYS, start=1)
    )


def _resolved_discourse_audit(projection_ref):
    discourse, _context = _facade()
    audit = discourse.execute(_request())
    return replace(
        audit,
        projection=replace(
            audit.projection,
            before_projection_ref=projection_ref,
            after_projection_ref=projection_ref,
        ),
    )


def _local_case(*, revision_kind: str = "LATER_CORRECTION"):
    fixture = _situation_fixture()
    original = fixture["facade"].append(
        _observation_event(fixture["ctx"], fixture["source"], seq=1)
    )
    revision = fixture["facade"].append(
        _observation_event(fixture["ctx"], fixture["source"], seq=2)
    )
    old_entry, replacement = _replacement(fixture)
    objects = tuple(
        _object_state(entry, ordinal=ordinal)
        for ordinal, entry in enumerate(fixture["projection"].entries(), start=1)
    )
    snapshot = W08LocalSnapshot(objects)
    affected = snapshot.affected(old_entry.dependencies)
    mapping = W08RevisionMapping(
        (affected[0].object_key,),
        (_key(29200, 1),),
    )
    invalidator = FreeTextRevisionInvalidator(tuple(
        FreeTextDerivedDependency(
            StableRecordKey(item.object_key),
            item.object_kind,
            item.dependencies,
        )
        for item in objects
    ))
    request = W08LocalRevisionRequest(
        request_key=_key(29400, 1),
        discourse_audit=_resolved_discourse_audit(
            fixture["projection"].state_ref()
        ),
        revision_kind=revision_kind,
        changed_dependencies=old_entry.dependencies,
        mappings=(mapping,),
        before_snapshot=snapshot,
        projection_update=_update(fixture, old_entry.dependencies),
        parser_request=None,
        parser_result=None,
        memory_parser_result=None,
        revision_event=revision,
        projection_replacements=(replacement,),
        preserved_event_hashes=(original.event_hash,),
    )
    runtime = W08LocalRecomputeRuntime(
        fixture["projection"],
        invalidator,
        lambda current, impacted: _rebuilt_state(
            current, impacted, replacement
        ),
        _revalidated_consumers,
    )
    return fixture, request, runtime, original, replacement


@pytest.mark.parametrize(
    ("old_keys", "new_keys", "shape"),
    (
        (((1,),), (), "OLD_TO_ZERO"),
        (((1,),), ((2,),), "OLD_TO_ONE"),
        (((1,),), ((2,), (3,)), "ONE_TO_MANY"),
        (((1,), (2,)), ((3,),), "MANY_TO_ONE"),
    ),
)
def test_revision_mapping_cardinality_is_explicit(old_keys, new_keys, shape):
    mapping = W08RevisionMapping(old_keys, new_keys)
    assert mapping.shape == shape
    assert shape in W08_REVISION_MAPPING_SHAPES


def test_revision_mapping_rejects_private_winner_and_unstable_order():
    with pytest.raises(W08LocalRecomputeError, match="many old"):
        W08RevisionMapping(((1,), (2,)), ((3,), (4,)))
    with pytest.raises(W08LocalRecomputeError, match="sorted"):
        W08RevisionMapping(((2,), (1,)), ((3,),))
    with pytest.raises(W08LocalRecomputeError, match="select an old"):
        W08RevisionMapping(((1,),), ((1,),))


@pytest.mark.parametrize(
    "revision_kind",
    (
        "REFERENCE_BACKTRACK",
        "SOURCE_WITHDRAWAL",
        "SOURCE_CONFLICT",
        "LATER_CORRECTION",
    ),
)
def test_non_parser_revisions_commit_only_typed_dependency_hits(revision_kind):
    fixture, request, runtime, original, replacement = _local_case(
        revision_kind=revision_kind
    )
    try:
        before = request.before_snapshot
        audit = runtime.execute(request)
        assert audit.revision_kind == revision_kind
        assert audit.mapping_shapes == ("OLD_TO_ONE",)
        assert audit.affected_object_keys == (
            request.mappings[0].old_keys[0],
        )
        assert audit.projection.invalidated_projection_keys == (
            replacement.entry.projection_key,
        )
        assert tuple(
            item.components for item in audit.free_text.invalidated_keys
        ) == audit.affected_object_keys
        assert audit.recompute_object_count == 1
        assert audit.full_document_reparse_count == 0
        assert audit.additional_payload_get_count == 0
        assert tuple(item.consumer_key for item in audit.consumers) == (
            "UNDERSTANDING",
            "REASONING",
            "GENERATION",
        )
        assert all(item.prior_uses_preserved == 1 for item in audit.consumers)
        assert all(
            item.prior_generation_output_rewritten == 0
            for item in audit.consumers
        )
        assert tuple(item.channel for item in audit.preservations) == (
            W08_LOCAL_STATE_CHANNELS
        )
        assert all(item.before_ref == item.after_ref for item in audit.preservations)
        assert fixture["facade"].read(original.event_hash) == original
        history_refs = {
            item.content_ref() for item in fixture["work_memory"].content_history()
        }
        assert replacement.entry.content_ref in history_refs
        assert before.objects[0].generation_output_keys == _channel(29180, 1)
    finally:
        _close_situation(fixture)


@pytest.mark.parametrize("fault_point", W08_LOCAL_RECOMPUTE_FAULT_POINTS)
def test_faults_before_commit_restore_exact_projection_and_work_memory(fault_point):
    fixture, request, runtime, _original, _replacement_value = _local_case()
    try:
        projection_state = fixture["projection"].state_key()
        work_memory_state = fixture["projection"].work_memory.state_key()
        history = fixture["work_memory"].content_history()
        with pytest.raises(W08LocalRecomputeInjectedFailure, match=fault_point):
            runtime.execute(request, fault_point=fault_point)
        assert fixture["projection"].state_key() == projection_state
        assert fixture["projection"].work_memory.state_key() == work_memory_state
        assert fixture["work_memory"].content_history() == history
    finally:
        _close_situation(fixture)


def test_md02_multi_slot_revision_uses_one_preview_commit_and_preserves_history():
    fixture = _situation_fixture()
    try:
        original = fixture["facade"].append(
            _observation_event(fixture["ctx"], fixture["source"], seq=1)
        )
        revision = fixture["facade"].append(
            _observation_event(fixture["ctx"], fixture["source"], seq=2)
        )
        entries = fixture["projection"].entries()[:2]
        replacements = []
        for ordinal, entry in enumerate(entries, start=1):
            old_item = next(
                item
                for item in fixture["items"]
                if item.content_ref() == entry.content_ref
            )
            new_item = replace(
                old_item,
                logical_seq=20 + ordinal,
                supersedes=(old_item.content_ref(),),
            )
            replacements.append(SituationProjectionReplacement(
                replace(
                    entry,
                    content_ref=new_item.content_ref(),
                    revision=entry.revision + 1,
                ),
                new_item,
            ))
        changed = tuple(sorted(
            (entry.dependencies[0] for entry in entries),
            key=lambda item: item.stable_key(),
        ))
        before_history = fixture["work_memory"].content_history()
        receipt = fixture["projection"].apply_revision(
            _update(fixture, changed),
            revision,
            tuple(sorted(
                replacements,
                key=lambda item: item.entry.projection_key,
            )),
            preserved_event_hashes=(original.event_hash,),
        )
        assert receipt.work_memory_write_count == 2
        assert len(receipt.invalidated_projection_keys) == 2
        history = fixture["work_memory"].content_history()
        assert set(before_history) < set(history)
        assert fixture["facade"].read(original.event_hash) == original
    finally:
        _close_situation(fixture)


def test_work_memory_preview_base_drift_fails_before_state_switch():
    fixture = _situation_fixture()
    try:
        store = fixture["projection"].work_memory
        preview = store.clone()
        before = store.state_key()
        with pytest.raises(WorkMemoryContentError, match="base state"):
            store.commit_preview(preview, expected_state_key=(1,))
        assert store.state_key() == before
    finally:
        _close_situation(fixture)


def test_dump_resume_replay_are_canonical_equivalent_and_zero_write():
    fixture, request, runtime, _original, _replacement_value = _local_case()
    try:
        audit = runtime.execute(request)
        payload = audit.dump().to_bytes()
        assert W08LocalRecomputeDump.from_bytes(payload) == audit.dump()
        state = fixture["projection"].state_key()
        work_memory = fixture["projection"].work_memory.state_key()
        replay = runtime.replay(audit, payload)
        resume = runtime.resume(audit, payload)
        assert replay.result_key == resume.result_key == audit.result_key()
        assert replay.additional_write_count == resume.additional_write_count == 0
        assert replay.additional_payload_get_count == 0
        assert fixture["projection"].state_key() == state
        assert fixture["projection"].work_memory.state_key() == work_memory
        assert b"raw_text" not in payload
        assert b"evidence_payload" not in payload
        with pytest.raises(W08LocalRecomputeError, match="fields drifted"):
            W08LocalRecomputeDump.from_bytes(
                payload.replace(b'"version":1', b'"extra":0,"version":1')
            )
    finally:
        _close_situation(fixture)


def test_request_rejects_raw_dependencies_full_set_drift_and_missing_parser_receipts():
    fixture, request, _runtime, _original, _replacement_value = _local_case()
    try:
        with pytest.raises(W08LocalRecomputeError, match="typed dependencies"):
            replace(request, changed_dependencies=((1,),))
        with pytest.raises(W08LocalRecomputeError, match="exact dependency hits"):
            replace(
                request,
                mappings=(W08RevisionMapping(((999,),), ((1000,),)),),
            )
        with pytest.raises(W08LocalRecomputeError, match="A-03/A-08"):
            replace(request, revision_kind="PARSER_REVISION")
    finally:
        _close_situation(fixture)


def test_a03_adapter_uses_real_atomic_revision_and_exact_replay():
    world = _a03_world()
    try:
        owner = W08A03ParserRevisionOwner(world.runtime)
        first = owner.apply(world.request)
        backend_state = world.backend.recovery_state_snapshot()
        ledger_state = world.ledger.state_key()
        resolver_state = world.resolver.state_key()
        replay = owner.apply(world.request)
        assert first.replayed is False
        assert replay.replayed is True
        assert replay.materialized == first.materialized
        assert world.backend.recovery_state_snapshot() == backend_state
        assert world.ledger.state_key() == ledger_state
        assert world.resolver.state_key() == resolver_state
    finally:
        world.backend.close()


def test_parser_and_boundary_revisions_bind_same_a03_a08_materialized_revision():
    world = _a08_world(DictBackend)
    try:
        memory_owner = W08A08MemoryRevisionOwner(world.runtime)
        memory_result = memory_owner.apply(
            world.request,
            raw_text=_A08_TEXT,
            license_id=_A08_LICENSE,
            batch_id=102,
            parser=_A08NewParser(world.new_source),
        )
        parser_result = ParserRevisionResult(
            memory_result.revision,
            (),
            True,
        )
        assert memory_result.preserved_use_refs == (world.use_ref,)
        replay = memory_owner.apply(
            world.request,
            raw_text=_A08_TEXT,
            license_id=_A08_LICENSE,
            batch_id=102,
            parser=_A08RejectParser(),
        )
        assert replay.replayed is True
        assert replay.revision == parser_result.materialized

        for revision_kind in (
            "PARSER_REVISION",
            "SENSE_BOUNDARY_SPLIT_MERGE",
        ):
            fixture, request, runtime, _original, _replacement_value = _local_case()
            try:
                parser_bound = replace(
                    request,
                    revision_kind=revision_kind,
                    parser_request=world.request,
                    parser_result=parser_result,
                    memory_parser_result=memory_result,
                )
                audit = runtime.execute(parser_bound)
                assert audit.revision_kind == revision_kind
                assert audit.owner_call_order[1] == (
                    "A-03_A-08_R-03_REVISION_OWNER"
                )
            finally:
                _close_situation(fixture)
    finally:
        world.backend.close()


def test_concrete_adapters_require_existing_owner_types():
    fixture, request, _runtime, _original, _replacement_value = _local_case()
    try:
        invalidator = FreeTextRevisionInvalidator(tuple(
            FreeTextDerivedDependency(
                StableRecordKey(item.object_key),
                item.object_kind,
                item.dependencies,
            )
            for item in request.before_snapshot.objects
        ))
        owner = W08FreeTextRevisionOwner(invalidator)
        assert owner.invalidate(request.changed_dependencies).invalidated_keys
        with pytest.raises(TypeError, match="ParserRevisionRuntime"):
            W08A03ParserRevisionOwner(object())
        with pytest.raises(TypeError, match="MemoryParserRevisionRuntime"):
            W08A08MemoryRevisionOwner(object())
        with pytest.raises(TypeError, match="FreeTextRevisionInvalidator"):
            W08FreeTextRevisionOwner(object())
    finally:
        _close_situation(fixture)


def test_local_recompute_ablation_is_orthogonal():
    full = {key: "PASS" for key in W08_DIMENSION_KEYS}
    ablated = dict(full)
    ablated["W-08-LOCAL_RECOMPUTE"] = "FAIL"
    report = assess_w08_local_recompute_ablation(
        full_dimension_outcomes=full,
        ablated_dimension_outcomes=ablated,
    )
    assert report.affected_dimensions == ("W-08-LOCAL_RECOMPUTE",)
    assert set(report.unaffected_dimensions) == set(W08_DIMENSION_KEYS) - {
        "W-08-LOCAL_RECOMPUTE"
    }
    drift = dict(ablated)
    drift["W-08-DISCOURSE"] = "FAIL"
    with pytest.raises(W08LocalRecomputeError, match="orthogonal"):
        assess_w08_local_recompute_ablation(
            full_dimension_outcomes=full,
            ablated_dimension_outcomes=drift,
        )


def test_revision_kind_and_mapping_inventories_are_frozen():
    assert W08_LOCAL_REVISION_KINDS == (
        "PARSER_REVISION",
        "SENSE_BOUNDARY_SPLIT_MERGE",
        "REFERENCE_BACKTRACK",
        "SOURCE_WITHDRAWAL",
        "SOURCE_CONFLICT",
        "LATER_CORRECTION",
    )
    assert W08_REVISION_MAPPING_SHAPES == (
        "OLD_TO_ZERO",
        "OLD_TO_ONE",
        "ONE_TO_MANY",
        "MANY_TO_ONE",
    )


def test_recompute_training_binds_visible_revision_evidence_without_shortcuts(
    training_payload,
):
    audit = audit_w08_recompute_training(training_payload)
    assert audit.observation_count == audit.evidence_binding_count == 63
    assert audit.discourse_revision_count == 9
    assert audit.parser_revision_plan_count == 2
    assert audit.reference_plan_count == 2
    assert audit.impacted_reference_query_count == 1
    assert audit.source_conflict_count == 1
    assert audit.later_correction_count == 1
    assert audit.affected_occurrence_count == 2
    assert audit.unaffected_occurrence_count == 2
    assert audit.recompute_query_count == 2
    assert audit.observed_mapping_shapes == ("OLD_TO_ONE",)
    assert audit.parent_mapping_shapes == W08_REVISION_MAPPING_SHAPES
    assert (
        audit.old_occurrence_rewrite_count,
        audit.unaffected_recompute_count,
        audit.whole_document_recompute_count,
        audit.fifo_authority_count,
        audit.surface_cue_authority_count,
        audit.authored_answer_read_count,
    ) == (0, 0, 0, 0, 0, 0)
