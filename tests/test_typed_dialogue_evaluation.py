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
