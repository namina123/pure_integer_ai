"""R-01.4 P3-Ia 四个 LC 账与能力基线的诚实修订测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_p3ia_ledger_revision_catalog import (
    P3IA_COURSE_PATH,
    P3IA_EVALUATOR_PATH,
    P3IA_PACK_PATH,
    P3IA_RUNTIME_PATHS,
    P3IA_SAMPLE_PATH,
    P3IA_TEST_PATH,
    REVISION_PATHS,
    SUPERSEDES_PATHS,
    build_p3ia_ledger_revisions,
)
from pure_integer_ai.experiments.ph2_p3ia_ledger_revision_contract import (
    ARTIFACT_VERSIONS,
    EXECUTION_STATE,
    LEDGER_FACTS,
    P3IaLedgerRevisionError,
    read_p3ia_ledger_revision,
    verify_p3ia_ledger_revision_files,
    write_p3ia_ledger_revision,
)
from pure_integer_ai.experiments.ph2_transfer_axis_catalog import (
    build_repository_transfer_axis_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
OLD_SHA256 = {
    "LC-00": "7c96579c900e9ca25390abd097d7f330d949fcf9e4288b7a311367d03e7f18f4",
    "LC-07": "791d8ee6f658fb0a7c3291ac6a9754fd1df414d27d99912c53be062f08a60ef5",
    "LC-09": "b1f010d9c2e761828be4e350f15245f6ab3c9e25cd8b31431afc8f02e972179c",
    "LC-13": "81fba102d31776518c2a34180ebdf3f90bdf759bd4add549bafcd69bb6f49765",
    "LC-15": "d69815ea6acd9e068ab88dd321993b95b3d6e0fad02a6c9c5101bdca9a40bd44",
}


@pytest.fixture(scope="module")
def built_revisions():
    return build_p3ia_ledger_revisions(REPOSITORY, WORKSPACE)


def test_five_revisions_build_deterministically_and_match_formal_files(
        built_revisions):
    """四个 LC v2 与 baseline v40 必须逐字节等于当前规范构建。"""
    rebuilt = build_p3ia_ledger_revisions(REPOSITORY, WORKSPACE)
    assert tuple(item.ledger_key for item in built_revisions) == (
        "LC-07", "LC-09", "LC-13", "LC-15", "LC-00")
    assert tuple(item.canonical_bytes() for item in rebuilt) == tuple(
        item.canonical_bytes() for item in built_revisions)
    for revision in built_revisions:
        path = REPOSITORY / REVISION_PATHS[revision.ledger_key]
        assert path.read_bytes() == revision.canonical_bytes()
        assert read_p3ia_ledger_revision(path) == revision


def test_old_manifests_remain_byte_identical_and_are_bound_as_superseded(
        built_revisions):
    """历史 v1/v39 不得覆盖，且每个新版本必须绑定其真实 SHA。"""
    for revision in built_revisions:
        old_path = REPOSITORY / SUPERSEDES_PATHS[revision.ledger_key]
        digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
        assert digest == OLD_SHA256[revision.ledger_key]
        assert revision.supersedes_sha256 == digest
        assert revision.artifact_version == ARTIFACT_VERSIONS[revision.ledger_key]


def test_common_course_pack_runtime_evaluator_and_test_evidence_are_exact(
        built_revisions):
    """每份修订必须直接绑定 R-01.3 的样本、pack、runtime 与 evaluator。"""
    required_repository = {
        P3IA_COURSE_PATH,
        P3IA_SAMPLE_PATH,
        P3IA_EVALUATOR_PATH,
        P3IA_TEST_PATH,
        *P3IA_RUNTIME_PATHS,
    }
    for revision in built_revisions:
        repository_paths = {
            item.relative_path for item in revision.evidence_files
            if item.root_key == "REPOSITORY"}
        workspace_paths = {
            item.relative_path for item in revision.evidence_files
            if item.root_key == "WORKSPACE"}
        assert required_repository <= repository_paths
        assert P3IA_PACK_PATH in workspace_paths
        verify_p3ia_ledger_revision_files(
            revision, repository_root=REPOSITORY, workspace_root=WORKSPACE)


def test_statuses_are_honest_and_p3ib_stays_ne_ph3(built_revisions):
    """中文窄纵切只到 COURSE_FROZEN/CONTRACT_READY，不能扩张为 mastered。"""
    for revision in built_revisions:
        assert revision.p3ia_course_status == "COURSE_FROZEN"
        assert revision.p3ia_production_contract_status == "CONTRACT_READY"
        assert revision.p3ib_status == "NE"
        assert revision.p3ib_phase == "PH3"
        assert revision.language_scope == ("zh",)
        assert revision.code_switch_status == "NE"
        assert revision.cross_language_pass_authority == 0
        assert revision.formal_runtime_status == "NOT_STARTED"
        assert revision.focused_runtime_evidence == "PASS"
        assert revision.execution_state.to_value() == EXECUTION_STATE


def test_ledger_specific_facts_do_not_invent_transfer_or_execution(
        built_revisions):
    """四本账只吸收实际新增事实，迁移、消融和候选淘汰保持未执行。"""
    by_key = {item.ledger_key: item for item in built_revisions}
    assert all(
        by_key[key].ledger_facts.to_value() == LEDGER_FACTS[key]
        for key in LEDGER_FACTS)
    assert by_key["LC-09"].ledger_facts.to_value()[
        "p3ia_transfer_claim_state"] == "NE"
    assert by_key["LC-15"].ledger_facts.to_value()[
        "candidate_eliminations_executed"] == 0
    assert by_key["LC-13"].ledger_facts.to_value()[
        "formal_directional_verdict"] == "NE"


def test_baseline_v40_binds_exactly_four_upstream_revisions(built_revisions):
    """v40 必须逐文件绑定四个 v2，不能只写聚合 PASS。"""
    baseline = built_revisions[-1]
    upstream = tuple(
        item for item in baseline.evidence_files
        if item.role == "UPSTREAM_REVISION")
    assert tuple(item.relative_path for item in upstream) == tuple(sorted(
        REVISION_PATHS[key].as_posix()
        for key in ("LC-07", "LC-09", "LC-13", "LC-15")))
    for item in upstream:
        path = REPOSITORY / item.relative_path
        payload = path.read_bytes()
        assert item.byte_count == len(payload)
        assert item.sha256 == hashlib.sha256(payload).hexdigest()


def test_contract_rejects_cross_language_mastered_and_fact_inflation(
        built_revisions):
    """跨语言权限、mastered 写和账目数字膨胀必须 fail closed。"""
    revision = built_revisions[0]
    with pytest.raises(P3IaLedgerRevisionError, match="跨语言 PASS"):
        replace(revision, cross_language_pass_authority=1)
    execution = dict(EXECUTION_STATE)
    execution["mastered_claims"] = 1
    with pytest.raises(P3IaLedgerRevisionError, match="execution_state"):
        replace(
            revision,
            execution_state=CanonicalJsonObject.from_value(execution),
        )
    facts = revision.ledger_facts.to_value()
    facts["base_course_preserved"] = 0
    with pytest.raises(P3IaLedgerRevisionError, match="ledger_facts"):
        replace(
            revision,
            ledger_facts=CanonicalJsonObject.from_value(facts),
        )


def test_nonoverwrite_writer_is_idempotent_and_rejects_corruption(
        tmp_path, built_revisions):
    """新版本可幂等核对，但绝不覆盖同路径的不同内容。"""
    revision = built_revisions[0]
    path = tmp_path / REVISION_PATHS[revision.ledger_key].name
    write_p3ia_ledger_revision(revision, path)
    write_p3ia_ledger_revision(revision, path)
    assert read_p3ia_ledger_revision(path) == revision
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(P3IaLedgerRevisionError, match="内容不同"):
        write_p3ia_ledger_revision(revision, path)


def test_lc09_v1_builder_remains_on_its_frozen_sixteen_pack_view():
    """新增 P3-Ia pack 不得让历史 LC-09 v1 builder 偷改 inventory。"""
    manifest = build_repository_transfer_axis_manifest(REPOSITORY, WORKSPACE)
    assert manifest.artifact_version == "LC-09-transfer-axis-manifest-v1"
    assert manifest.pack_inventory_count == 16
    assert all(P3IA_PACK_PATH != item.pack_manifest_relative_path
               for item in manifest.pack_audits)
    assert {item.transfer_claim_state for item in manifest.pack_audits} == {"NE"}
