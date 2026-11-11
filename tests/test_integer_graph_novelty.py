"""Novelty must follow real typed topology, never fresh IDs or grouping labels."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import occurrence_identity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicRoleBinding, context_scope_identity, event_identity, proposition_identity,
)
from pure_integer_ai.experiments.integer_graph_novelty import (
    IntegerGraphNoveltyCase, IntegerGraphNoveltyError,
    consumed_integer_graph_topology, integer_graph_topology, validate_integer_graph_novelty_splits,
)
from pure_integer_ai.experiments.integer_relation_graph_course import IntegerRelationGraphRecord
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
    authored_relation_schema_identity, authored_semantic_source,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES


def _event(group: int, local_id: int):
    return event_identity(authored_semantic_source((9, group)), (local_id,))


def _record(serial: int, left, right, *, family: str = "EVENT_BEFORE"):
    profile = W06_RELATION_PROFILES[family]
    source = authored_semantic_source((1, serial))
    proposition = proposition_identity(source, (serial,))
    predicate = authored_relation_identity(profile.relation_kind)
    roles = tuple(authored_relation_role_identity(kind) for kind, _ in profile.role_object_kinds)
    assert len(roles) == 2
    bindings = tuple(AtomicRoleBinding(role, filler, 0)
                     for role, filler in zip(roles, (left, right), strict=True))
    return IntegerRelationGraphRecord(
        source_ref_key=(1, serial), dataset_key=(2, serial), artifact_key=(3, serial),
        observation_key=(4, serial), semantic_key=(5, serial),
        proposition_key=proposition.stable_key(), predicate_key=predicate.stable_key(),
        context_key=context_scope_identity(source, (1,)).stable_key(),
        source_anchor_key=occurrence_identity(source, start=0, end=4, ordinal=0).stable_key(),
        schema_key=authored_relation_schema_identity(profile.relation_kind).stable_key(),
        relation_key=predicate.stable_key(),
        role_bindings=tuple({
            "binding_key": list(binding.identity_for(proposition).stable_key()),
            "role_key": list(binding.role.stable_key()),
            "filler_key": list(binding.filler.stable_key()), "ordinal": 0,
        } for binding in bindings),
        slots=tuple({
            "role_key": list(role.stable_key()), "allowed_object_kinds": sorted(allowed),
            "min_count": 1, "max_count": 1,
        } for role, (_, allowed) in zip(roles, profile.role_object_kinds, strict=True)),
        endpoints=tuple({
            "endpoint_key": list(filler.stable_key()), "object_kind": filler.object_kind,
            "ordinal": 0, "start": index * 3, "end": index * 3 + 1,
        } for index, filler in enumerate((left, right))),
        surface_values=tuple(map(ord, "甲先乙后")), anchor_span=(0, 4, 0),
        relation_kind=profile.relation_kind, directionality=profile.directionality,
        sample_role_code=1, perturbation_code=0, logical_order=serial,
        source_cluster_key=(6, serial), dedup_cluster_key=(7, serial),
        content_group_key=(8, serial), template_group_key=(10, serial),
        shape_group_key=(11, serial),
    )


def _chain(serial: int, group: int, *, family: str = "EVENT_BEFORE"):
    first, middle, last = (_event(group, value) for value in (1, 2, 3))
    return (_record(serial, first, middle, family=family),
            _record(serial + 1, middle, last, family=family))


def _fork(serial: int, group: int):
    first, middle, last = (_event(group, value) for value in (1, 2, 3))
    return (_record(serial, first, middle), _record(serial + 1, first, last))


def _case(key: int, records, text: str):
    return IntegerGraphNoveltyCase((1, key), tuple(map(ord, text)), records, ((12, key),))


def _splits():
    return {
        "train": [_case(1, _chain(100, 100), "训练关系")],
        "held_out": [_case(2, _fork(200, 200), "未见关系")],
        "negative": [_case(3, (_record(300, _event(300, 1), _event(300, 2)),), "无证据")],
        "conflict": [_case(4, (_record(400, _event(400, 1), _event(400, 2)),), "冲突")],
    }


def _validate(splits):
    return validate_integer_graph_novelty_splits(
        splits, minimum_novel_held_out=1, minimum_relation_families=1)


def test_identity_and_provenance_renumbering_does_not_create_new_topology():
    first = integer_graph_topology(_chain(100, 100))
    renamed = integer_graph_topology(_chain(800, 800))
    assert first == renamed
    assert first.component_count == 1
    assert first.endpoint_count == 3


def test_consumed_topology_excludes_unselected_facts_and_deduplicates_premise_uses():
    """执行可多次引用同一前提，但未消费的课程分支不得计作新颖性。"""
    chain = _chain(100, 100)
    extra = _fork(200, 200)
    keys = tuple(row.proposition_key for row in chain)
    assert consumed_integer_graph_topology(chain + extra, keys + keys) == integer_graph_topology(chain)
    assert consumed_integer_graph_topology(_chain(800, 800),
        tuple(row.proposition_key for row in _chain(800, 800))) == integer_graph_topology(chain)
    with pytest.raises(IntegerGraphNoveltyError, match="missing"):
        consumed_integer_graph_topology(chain, (extra[0].proposition_key,))


def test_record_binding_and_slot_order_do_not_change_signature():
    records = _chain(100, 100)
    reordered = tuple(replace(row, role_bindings=tuple(reversed(row.role_bindings)),
                              slots=tuple(reversed(row.slots))) for row in reversed(records))
    assert integer_graph_topology(reordered) == integer_graph_topology(records)


def test_shape_and_template_labels_do_not_change_signature():
    records = _chain(100, 100)
    renamed = tuple(replace(row, shape_group_key=(99, index + 1),
                            template_group_key=(98, index + 1))
                    for index, row in enumerate(records))
    assert integer_graph_topology(renamed) == integer_graph_topology(records)


def test_shared_role_endpoint_direction_changes_actual_structure():
    chain = integer_graph_topology(_chain(100, 100))
    fork = integer_graph_topology(_fork(200, 200))
    assert chain.structure_sha256 != fork.structure_sha256
    assert chain.relation_combo_sha256 != fork.relation_combo_sha256
    assert fork.component_count == 1


def test_new_relation_labels_do_not_also_claim_new_structure():
    temporal = integer_graph_topology(_chain(100, 100))
    causal = integer_graph_topology(_chain(200, 200, family="CAUSES"))
    assert temporal.structure_sha256 == causal.structure_sha256
    assert temporal.relation_combo_sha256 != causal.relation_combo_sha256


def test_same_local_endpoint_id_in_another_source_does_not_join_edges():
    chain = _chain(100, 100)
    disconnected = (chain[0], _record(102, _event(101, 2), _event(101, 3)))
    original = integer_graph_topology(chain)
    separate = integer_graph_topology(disconnected)
    assert separate.component_count == 2
    assert original.structure_sha256 != separate.structure_sha256


def test_proposition_as_endpoint_preserves_its_real_edge_link():
    first = _record(100, _event(100, 1), _event(100, 2))
    from pure_integer_ai.cognition.shared.identity import ObjectIdentity
    proposition = ObjectIdentity.from_stable_key(first.proposition_key)
    linked = _record(101, proposition, _event(100, 3), family="CAUSES")
    renamed = proposition_identity(authored_semantic_source((1, 999)), (100,))
    separate = _record(101, renamed, _event(100, 3), family="CAUSES")
    assert integer_graph_topology((first, linked)).component_count == 1
    assert integer_graph_topology((first, separate)).component_count == 2


def test_repeated_proof_sources_do_not_count_as_new_topology():
    left, right = _event(100, 1), _event(100, 2)
    with pytest.raises(IntegerGraphNoveltyError, match="evidence copies"):
        integer_graph_topology((_record(100, left, right), _record(101, left, right)))
    first = _record(100, left, right)
    with pytest.raises(IntegerGraphNoveltyError, match="repeated proposition"):
        integer_graph_topology((first, first))


def test_missing_graph_records_and_unregistered_roles_fail_closed():
    with pytest.raises(IntegerGraphNoveltyError, match="actual integer graph records"):
        integer_graph_topology([{"shape_group_key": [1, 99]}])
    first = _record(100, _event(100, 1), _event(100, 2))
    wrong_profile = replace(first, relation_kind=W06_RELATION_PROFILES["CAUSES"].relation_kind)
    with pytest.raises(IntegerGraphNoveltyError, match="unregistered"):
        integer_graph_topology([wrong_profile])


def test_exact_labeling_and_case_size_budgets_fail_closed():
    records = tuple(_record(100 + index, _event(100, index + 1), _event(100, index + 2))
                    for index in range(5))
    with pytest.raises(IntegerGraphNoveltyError, match="budget exceeded"):
        integer_graph_topology(records, max_labelings=10)
    with pytest.raises(IntegerGraphNoveltyError, match="1..8"):
        integer_graph_topology(records + records)


def test_connected_novel_graph_passes_admission_without_behavior_claim():
    report = _validate(_splits())
    assert report["status"] == "PASS_STRUCTURAL_ADMISSION_ONLY"
    assert report["novel_held_out_case_count"] == 1
    assert report["independent_novel_held_out_graph_count"] == 1
    assert report["operation_qualification"].startswith("UNKNOWN")
    assert report["free_dialogue_complete"] == 0


def test_fresh_surface_and_groups_cannot_hide_an_old_structure():
    splits = _splits()
    splits["held_out"] = [_case(2, _chain(200, 200), "另一个全新表面")]
    report = _validate(splits)
    held = next(row for row in report["rows"] if row["split"] == "held_out")
    assert held["surface_novel"] == 1
    assert held["structure_novel"] == held["relation_combo_novel"] == 0
    assert report["status"] == "BLOCKED"
    assert report["novel_held_out_case_count"] == 0


@pytest.mark.parametrize("field,label", [
    ("source_ref_key", "source"), ("source_cluster_key", "source_cluster"),
    ("content_group_key", "document"), ("dedup_cluster_key", "concept_cluster"),
    ("observation_key", "observation"),
])
def test_all_case_splits_reject_provenance_leakage(field, label):
    splits = _splits()
    train_record = splits["train"][0].records[0]
    negative = splits["negative"][0]
    contaminated = replace(negative.records[0], **{field: getattr(train_record, field)})
    splits["negative"] = [replace(negative, records=(contaminated,))]
    report = _validate(splits)
    assert f"SPLIT_OVERLAP:{label}:train:negative" in report["issues"]
    assert report["status"] == "BLOCKED"


def test_dialogue_chain_overlap_and_repeated_held_out_graphs_are_rejected():
    splits = _splits()
    held = splits["held_out"][0]
    splits["held_out"] = [held, _case(5, _fork(500, 500), "另一个未见表面")]
    report = validate_integer_graph_novelty_splits(
        splits, minimum_novel_held_out=2, minimum_relation_families=1)
    assert report["novel_held_out_case_count"] == 2
    assert report["independent_novel_held_out_graph_count"] == 1
    assert "INSUFFICIENT_INDEPENDENT_NOVEL_HELD_OUT_GRAPHS" in report["issues"]
    splits["conflict"][0] = replace(splits["conflict"][0],
                                     dialogue_chain_keys=held.dialogue_chain_keys)
    assert "SPLIT_OVERLAP:dialogue_chain:held_out:conflict" in _validate(splits)["issues"]


def test_relation_kinds_in_the_same_registered_family_are_not_extra_families():
    splits = _splits()
    held = splits["held_out"][0]
    extra = _record(202, _event(200, 2), _event(200, 3), family="EVENT_AFTER")
    splits["held_out"] = [replace(held, records=held.records + (extra,))]
    report = validate_integer_graph_novelty_splits(
        splits, minimum_novel_held_out=1, minimum_relation_families=2)
    assert report["held_out_relation_family_count"] == 1
    assert "INSUFFICIENT_HELD_OUT_RELATION_FAMILIES" in report["issues"]


def test_missing_split_chain_and_boolean_scalar_are_contract_errors():
    splits = _splits()
    del splits["conflict"]
    with pytest.raises(IntegerGraphNoveltyError, match="all four"):
        _validate(splits)
    with pytest.raises(IntegerGraphNoveltyError, match="dialogue-chain"):
        IntegerGraphNoveltyCase((1, 1), (123,), _chain(100, 100), ())
    with pytest.raises(IntegerGraphNoveltyError, match="integer key"):
        IntegerGraphNoveltyCase((1, True), (123,), _chain(100, 100), ((2, 1),))
