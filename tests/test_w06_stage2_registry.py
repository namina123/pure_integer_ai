"""W06-01 relation registry、train 证据覆盖与外部映射专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_FORMAL_RUN_ID,
    W06_RESOURCE_BUDGET,
    W06_RUNNER_KEY,
    W06_STAGE_KEY,
    W06_W05_BASE_RUN_ID,
    W06RunRequest,
    open_w06_frozen_context,
)
from pure_integer_ai.experiments.ph2_w06_firewall import W06PayloadFirewall
from pure_integer_ai.experiments.ph2_w06_registry import (
    W06_CONSUMER_KEYS,
    W06_RELATION_REGISTRY,
    W06RegistryError,
    audit_w06_registry_payload,
    validate_w06_external_relation,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.storage.backend import SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]
HEAD = "4d57305bc4474081c9304a05287ab4783f49a849"


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    """经 public firewall 一次性交付真实 W-06 train payload。"""
    path = tmp_path_factory.mktemp("w06-registry") / "probe.sqlite"
    backend = SQLiteBackend(str(path))
    try:
        context = open_w06_frozen_context(
            ROOT,
            current_remote_commit_sha1=HEAD,
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        request = W06RunRequest(
            W06_FORMAL_RUN_ID,
            W06_W05_BASE_RUN_ID,
            W06_W05_BASE_RUN_ID,
            W06_STAGE_KEY,
            context.owner_key,
            W06_RUNNER_KEY,
            context.current_remote_commit_sha1,
            context.source_overlay_sha256,
            context.stable_key(),
            context.backend_profile_key,
            context.base_fence_key,
            1,
            "fresh",
            tuple(sorted(W06_RESOURCE_BUDGET.items())),
            tuple(item.relative_path for item in context.candidate_payload_bindings),
            tuple(item.relative_path for item in context.teacher_evidence_bindings),
        )
        return W06PayloadFirewall.open(
            ROOT, context, request).read_training_payload()
    finally:
        backend.close()


def test_w06_registry_binds_seven_substages_sources_evidence_and_consumers():
    """七子阶段逐项声明关系族、forming/oracle 来源、Evidence 和 U/R/G。"""
    assert tuple(W06_RELATION_REGISTRY) == W06_RELATION_SUBSTAGE_ORDER
    relation_families = []
    for entry in W06_RELATION_REGISTRY.values():
        relation_families.extend(entry.relation_families)
        assert entry.forming_source_keys
        assert entry.independent_oracle_source_keys
        assert entry.teacher_evidence_kinds
        assert entry.required_perturbations
        assert entry.consumer_keys == W06_CONSUMER_KEYS
        assert entry.teacher_withdrawal_level == 0
        assert entry.minimum_independent_oracle_count == 1
    assert len(relation_families) == len(set(relation_families)) == 14


def test_w06_registry_audits_current_train_payload(payload):
    """当前有效 18-pack 中七类 relation train/Evidence 全部命中 registry。"""
    report = audit_w06_registry_payload(payload)
    assert report.observation_count == report.teacher_evidence_count == 51
    assert report.substage_counts == (
        ("PURE_ALIAS_REFERS", 5),
        ("SUBSET_MEMBER", 6),
        ("PROPERTY", 7),
        ("MEREOLOGY", 7),
        ("SIMILAR_ANTONYM", 7),
        ("PRECEDES", 9),
        ("CAUSES", 10),
    )
    assert len(report.relation_counts) == 14
    assert report.source_keys == ("AUTHORED_CC0_V1",)


def test_w06_external_relation_mapping_fails_closed():
    """外部 property/relation 必须逐项注册，PRECEDES 还必须带已登记 qualifier。"""
    assert validate_w06_external_relation(
        "WIKIDATA_REVISION_V1", "MEMBER", "P31").substage_key == "SUBSET_MEMBER"
    assert validate_w06_external_relation(
        "WIKIDATA_REVISION_V1", "PROPERTY", "P17").substage_key == "PROPERTY"
    assert validate_w06_external_relation(
        "WIKIDATA_REVISION_V1",
        "EVENT_BEFORE",
        "P156",
        qualifier_keys=("P580",),
    ).substage_key == "PRECEDES"
    assert validate_w06_external_relation(
        "CONCEPTNET_5_7_0", "CAUSES", "Causes").substage_key == "CAUSES"

    invalid = (
        ("WIKIDATA_REVISION_V1", "UNKNOWN", "P31", (), "未注册"),
        ("WIKIDATA_REVISION_V1", "MEMBER", "P999", (), "property"),
        ("WIKIDATA_REVISION_V1", "SUBSET", "P31", (), "错配"),
        ("WIKIDATA_REVISION_V1", "EVENT_BEFORE", "P156", (), "qualifier"),
        ("WIKIDATA_REVISION_V1", "EVENT_BEFORE", "P156", ("P999",), "qualifier"),
        ("CONCEPTNET_5_7_0", "CAUSES", "IsA", (), "错配"),
        ("AUTHORED_CC0_V1", "CAUSES", "Causes", (), "不接受"),
    )
    for source, family, relation, qualifiers, message in invalid:
        with pytest.raises(W06RegistryError, match=message):
            validate_w06_external_relation(
                source, family, relation, qualifier_keys=qualifiers)
