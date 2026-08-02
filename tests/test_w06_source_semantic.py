"""W06-00 来源独立性、关系 profile 和稳定 REFERS 防火墙测试。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_OCCURRENCE
from pure_integer_ai.experiments.ph2_authored_alias_refers_course import (
    read_authored_alias_refers_seeds,
)
from pure_integer_ai.experiments.ph2_authored_alias_refers_w06_course import (
    PACK_NAME,
    compile_authored_alias_refers_w06_course,
    read_authored_alias_refers_w06_seeds,
)
from pure_integer_ai.experiments.ph2_authored_causes_course import (
    read_authored_causes_seeds,
)
from pure_integer_ai.experiments.ph2_authored_mereology_course import (
    read_authored_mereology_seeds,
)
from pure_integer_ai.experiments.ph2_authored_precedes_course import (
    read_authored_precedes_seeds,
)
from pure_integer_ai.experiments.ph2_authored_property_course import (
    read_authored_property_seeds,
)
from pure_integer_ai.experiments.ph2_authored_semantic_pair_course import (
    read_authored_semantic_pair_seeds,
)
from pure_integer_ai.experiments.ph2_authored_subset_member_course import (
    read_authored_subset_member_seeds,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_PROFILES,
    W06_RELATION_SUBSTAGE_ORDER,
    W06SourceIsolationReport,
    W06SourceSemanticError,
    audit_w06_authored_source_isolation,
    validate_w06_relation_seed,
)


V1_ALIAS_PATH = Path(
    "data/ph2/authored_relation_alias_refers_seed_v1.jsonl.sample")
V2_ALIAS_PATH = Path(
    "data/ph2/authored_relation_alias_refers_w06_seed_v2.jsonl.sample")
RELATION_READERS = (
    (
        read_authored_subset_member_seeds,
        Path("data/ph2/authored_relation_subset_member_seed_v1.jsonl.sample"),
    ),
    (
        read_authored_property_seeds,
        Path("data/ph2/authored_relation_property_seed_v1.jsonl.sample"),
    ),
    (
        read_authored_mereology_seeds,
        Path("data/ph2/authored_relation_mereology_seed_v1.jsonl.sample"),
    ),
    (
        read_authored_semantic_pair_seeds,
        Path("data/ph2/authored_relation_similar_antonym_seed_v1.jsonl.sample"),
    ),
    (
        read_authored_precedes_seeds,
        Path("data/ph2/authored_relation_precedes_seed_v1.jsonl.sample"),
    ),
    (
        read_authored_causes_seeds,
        Path("data/ph2/authored_relation_causes_seed_v1.jsonl.sample"),
    ),
)


def test_v1_occurrence_refers_is_rejected_but_v2_is_stable():
    """旧篇章代词保持历史，W-06 v2 的 REFERS 只接受稳定一等对象。"""
    old = read_authored_alias_refers_seeds(V1_ALIAS_PATH)
    occurrence_refers = next(
        seed for seed in old
        if seed.relation_family == "REFERS"
        and any(item.object_kind == OBJECT_OCCURRENCE for item in seed.endpoints)
    )
    with pytest.raises(W06SourceSemanticError, match="篇章指代"):
        validate_w06_relation_seed(occurrence_refers)

    current = read_authored_alias_refers_w06_seeds(V2_ALIAS_PATH)
    assert len(current) == 9
    assert {seed.relation_family for seed in current} == {
        "PURE_ALIAS", "REFERS"}
    assert all(
        item.object_kind != OBJECT_OCCURRENCE
        for seed in current if seed.relation_family == "REFERS"
        for item in seed.endpoints
    )
    assert {
        validate_w06_relation_seed(seed).substage_key for seed in current
    } == {"PURE_ALIAS_REFERS"}


def test_all_seven_relation_courses_match_w06_profiles():
    """七个冻结 substage 的全部公开 seed 均通过方向、Role 和端点防火墙。"""
    seeds = list(read_authored_alias_refers_w06_seeds(V2_ALIAS_PATH))
    for reader, path in RELATION_READERS:
        seeds.extend(
            item.relation if hasattr(item, "relation") else item
            for item in reader(path)
        )
    observed_substages = {
        validate_w06_relation_seed(seed).substage_key for seed in seeds
    }
    assert observed_substages == set(W06_RELATION_SUBSTAGE_ORDER)
    assert set(W06_RELATION_PROFILES) == {
        seed.relation_family for seed in seeds}
    assert W06_GENERATION_HARD_CONJUNCT == (
        "W-06-GENERATION-RELATION-STRUCTURE-HARD-CONJUNCT")


def test_alias_v2_pack_is_bit_identical_and_four_way_source_isolated(tmp_path):
    """v2 pack 双编译一致，并由 cluster、owner、family、template 四重隔离。"""
    first = compile_authored_alias_refers_w06_course(
        V2_ALIAS_PATH, tmp_path / "first")
    second = compile_authored_alias_refers_w06_course(
        V2_ALIAS_PATH, tmp_path / "second")
    assert first.pack_root.name == PACK_NAME
    assert first.manifest == second.manifest
    assert (first.pack_root / "manifest.json").read_bytes() == (
        second.pack_root / "manifest.json").read_bytes()
    for left, right in zip(first.manifest.files, second.manifest.files):
        assert left == right
        assert (first.pack_root / left.relative_path).read_bytes() == (
            second.pack_root / right.relative_path).read_bytes()

    seeds = read_authored_alias_refers_w06_seeds(V2_ALIAS_PATH)
    report = audit_w06_authored_source_isolation(first.pack_root, seeds)
    assert report.source_key == "AUTHORED_CC0_V1"
    assert (
        report.train_observation_count,
        report.held_out_observation_count,
        report.teacher_evidence_count,
        report.evaluator_label_count,
    ) == (5, 4, 5, 4)
    assert set(report.train_cluster_keys).isdisjoint(report.held_out_cluster_keys)
    assert set(report.teacher_owner_keys).isdisjoint(report.evaluator_owner_keys)
    assert set(report.teacher_families).isdisjoint(report.evaluator_families)
    assert set(report.teacher_templates).isdisjoint(report.evaluator_templates)


def test_semantic_and_source_leakage_fail_closed(tmp_path):
    """关系 kind 漂移与任一来源隔离交集均在 W-06 payload 前失败。"""
    seed = read_authored_alias_refers_w06_seeds(V2_ALIAS_PATH)[0]
    with pytest.raises(W06SourceSemanticError, match="kind 或方向"):
        validate_w06_relation_seed(replace(seed, relation_kind=2))

    common = dict(
        pack_key="pack",
        source_key="AUTHORED_CC0_V1",
        train_observation_count=1,
        held_out_observation_count=1,
        teacher_evidence_count=1,
        evaluator_label_count=1,
        train_cluster_keys=((1, 1),),
        held_out_cluster_keys=((1, 2),),
        teacher_owner_keys=((1, 3),),
        evaluator_owner_keys=((1, 4),),
        teacher_families=("teacher",),
        evaluator_families=("evaluator",),
        teacher_templates=("teacher-template",),
        evaluator_templates=("evaluator-template",),
    )
    for field, leaked in (
            ("held_out_cluster_keys", ((1, 1),)),
            ("evaluator_owner_keys", ((1, 3),)),
            ("evaluator_families", ("teacher",)),
            ("evaluator_templates", ("teacher-template",))):
        values = dict(common)
        values[field] = leaked
        with pytest.raises(W06SourceSemanticError, match="泄漏"):
            W06SourceIsolationReport(**values)
