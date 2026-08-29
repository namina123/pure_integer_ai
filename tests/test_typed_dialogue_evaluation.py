"""公开 typed 对话 V-00/H2/floor 装配回归。"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.corpus_identity import assign_corpus_source_refs
from pure_integer_ai.experiments.run_conversation_training import default_course_paths
from pure_integer_ai.experiments.typed_dialogue_evaluation import (
    build_typed_dialogue_evaluation_bundle,
)


def test_public_typed_v00_bundle_has_five_isolated_splits():
    """公开 authored source 显式形成五类 ledger，标签不由运行结果推导。"""
    pack = load_dialogue_training_pack((*default_course_paths("."),
                                       "data/ph2/dialogue_relation_causes_scale_v1.course.jsonl.sample"))
    items = pack.items_for_split(split=None)
    assign_corpus_source_refs(items, source_namespace=pack.pack_sha256)
    by_case = {
        case.case_id: item for case, item in zip(pack.cases, items)
    }
    bundle = build_typed_dialogue_evaluation_bundle(pack, by_case)
    protocol = bundle.evaluation_plan.protocol
    assert len(protocol.split_keys()) == 5
    assert len(bundle.evaluation_plan.assignments) == len(bundle.corpus)
    assert bundle.h2_protocol.cases[0].identity == next(
        item.identity for item in bundle.evaluation_plan.assignments
        if item.split == protocol.development_split)
    assert bundle.floor_protocol.cases[0].identity == next(
        item.identity for item in bundle.evaluation_plan.assignments
        if item.split == protocol.held_out_split)
    assert bundle.floor_protocol.requirements
    assert all(item.minimum_match_permille == 1000
               for item in bundle.floor_protocol.requirements)
    assert sum(
        item.split != protocol.training_split
        for item in bundle.evaluation_plan.assignments
    ) == 4


def test_postcheck_bridge_remains_training_after_h2_development_split():
    """H2 隔离原正例时，独立 bridge 仍向训练图提供 POSTCHECK。"""
    bridge_path = (
        "data/ph2/dialogue_postcheck_bridge_train_v1.course.jsonl.sample")
    pack = load_dialogue_training_pack(
        (*default_course_paths("."), bridge_path))
    items = pack.items_for_split(split=None)
    assign_corpus_source_refs(items, source_namespace=pack.pack_sha256)
    bundle = build_typed_dialogue_evaluation_bundle(
        pack, {case.case_id: item for case, item in zip(pack.cases, items)})
    protocol = bundle.evaluation_plan.protocol

    assert bundle.development_item.raw_text.startswith("小舟靠岸")
    development = next(
        assignment for assignment in bundle.evaluation_plan.assignments
        if assignment.identity == bundle.h2_protocol.cases[0].identity)
    assert development.split == protocol.development_split

    bridge = [
        (item, assignment)
        for item, assignment in zip(
            bundle.corpus, bundle.evaluation_plan.assignments)
        if item.raw_text is not None and item.raw_text.startswith("木筏靠岸")
    ]
    assert len(bridge) == 1
    bridge_item, bridge_assignment = bridge[0]
    assert bridge_assignment.split == protocol.training_split
    assert bridge_item.payload_kind == "GenerationAdoptionPostcheckQuery"
    assert bridge_item.typed_payload is not None
    assert bridge_item.typed_payload.to_value()["task_kind"] == "POSTCHECK"
