"""W06-00 来源/语义 overlay 的 canonical、parent 和 append-only 测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic_overlay import (
    W06_EXPECTED_PARENT_SHA256,
    W06SourceOverlayError,
    build_w06_source_semantic_overlay,
    canonical_w06_source_semantic_overlay_bytes,
    publish_w06_source_semantic_overlay,
    read_w06_source_semantic_overlay,
)


REPO_ROOT = Path(".")


def test_overlay_rebuild_is_canonical_and_binds_current_parent():
    """两次重建逐字节一致，parent、七阶段和生成补充门均绑定当前文件。"""
    first = canonical_w06_source_semantic_overlay_bytes(REPO_ROOT)
    second = canonical_w06_source_semantic_overlay_bytes(REPO_ROOT)
    assert first == second
    value = build_w06_source_semantic_overlay(REPO_ROOT)
    assert value["status"] == "W06_SOURCE_AND_SEMANTIC_PREREQUISITE_PASS"
    assert value["relation_substage_order"] == list(W06_RELATION_SUBSTAGE_ORDER)
    assert value["generation_supplement"]["hard_conjunct"] == (
        W06_GENERATION_HARD_CONJUNCT)
    assert value["legacy_v1_boundary"]["occurrence_refers_rejected_count"] > 0
    for relative, expected in W06_EXPECTED_PARENT_SHA256.items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected
        assert value["parent_identities"][relative] == expected


def test_overlay_proves_stable_v2_four_way_independence():
    """同一 CC0 来源键不替代 cluster、owner、family 和 template 四重互斥。"""
    course = build_w06_source_semantic_overlay(REPO_ROOT)["stable_v2_course"]
    assert course["source_key"] == "AUTHORED_CC0_V1"
    assert course["source_independence_policy"] == (
        "DISTINCT_CLUSTER_OWNER_SEED_FAMILY_TEMPLATE")
    assert course["train_observation_count"] == course["teacher_evidence_count"] == 5
    assert course["held_out_observation_count"] == course["evaluator_label_count"] == 4
    assert course["train_cluster_count"] == course["held_out_cluster_count"] == 1
    assert course["teacher_owner_count"] == course["evaluator_owner_count"] == 1
    assert course["teacher_family_count"] == course["evaluator_family_count"] == 1
    assert course["teacher_template_count"] == course["evaluator_template_count"] == 1


def test_overlay_publish_and_read_are_append_only(tmp_path):
    """发布后可规范回读，重复发布和非规范字节均 fail closed。"""
    target = tmp_path / "overlay.json"
    publish_w06_source_semantic_overlay(REPO_ROOT, target)
    value = read_w06_source_semantic_overlay(target)
    assert value == build_w06_source_semantic_overlay(REPO_ROOT)
    with pytest.raises(W06SourceOverlayError, match="禁止覆盖"):
        publish_w06_source_semantic_overlay(REPO_ROOT, target)

    bad = tmp_path / "bad.json"
    bad.write_text('{"status": "x"}\n', encoding="utf-8")
    with pytest.raises(W06SourceOverlayError, match="规范"):
        read_w06_source_semantic_overlay(bad)
