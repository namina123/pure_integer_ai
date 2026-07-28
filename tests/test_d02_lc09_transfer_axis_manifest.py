"""LC-09 十轴迁移账、组合 split 与零运行权限测试。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments import ph2_transfer_axis_catalog as catalog
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_transfer_axis_catalog import (
    LC09_MANIFEST_PATH,
    TransferAxisCatalogError,
    build_repository_transfer_axis_manifest,
    build_transfer_split_probes,
)
from pure_integer_ai.experiments.ph2_transfer_axis_contract import (
    EXECUTION_STATE,
    SPLIT_PROBE_KINDS,
    TRANSFER_AXIS_KEYS,
    LanguageTransferAxisManifest,
    TransferAxisContractError,
    evaluate_transfer_split_probe,
    read_transfer_axis_manifest,
    write_transfer_axis_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
FORMAL_MANIFEST_SHA256 = (
    "b1f010d9c2e761828be4e350f15245f6ab3c9e25cd8b31431afc8f02e972179c")


@pytest.fixture(scope="module")
def formal_manifest() -> LanguageTransferAxisManifest:
    """只读构建一次当前正式 16-pack LC-09 账。"""
    return build_repository_transfer_axis_manifest(REPOSITORY, WORKSPACE)


def _combination(prefix: str) -> CanonicalJsonObject:
    """构造列全十轴的测试组合。"""
    return CanonicalJsonObject.from_value({
        axis: f"{prefix}_{axis}" for axis in TRANSFER_AXIS_KEYS
    })


def test_formal_inventory_is_complete_and_all_transfer_claims_remain_ne(
        formal_manifest):
    """LC-09 必须覆盖全部正式 pack，但不得签发迁移能力 PASS。"""
    assert formal_manifest.artifact_status == "CONTRACT_FROZEN"
    assert formal_manifest.runtime_status == "NOT_STARTED"
    assert formal_manifest.axis_keys == TRANSFER_AXIS_KEYS
    assert formal_manifest.pack_inventory_count == 16
    assert len(formal_manifest.pack_audits) == 16
    assert {item.pack_kind for item in formal_manifest.pack_audits} == {
        "AUTHORED_COURSE", "SOURCE_PACK"}
    assert sum(item.pack_kind == "AUTHORED_COURSE"
               for item in formal_manifest.pack_audits) == 10
    assert sum(item.pack_kind == "SOURCE_PACK"
               for item in formal_manifest.pack_audits) == 6
    assert {item.transfer_claim_state
            for item in formal_manifest.pack_audits} == {"NE"}
    assert formal_manifest.runtime_transfer_pass_authority == 0
    assert formal_manifest.execution_state.to_value() == EXECUTION_STATE


def test_source_axes_are_explicit_and_authored_missing_axes_are_ne(
        formal_manifest):
    """来源 pack 读显式十轴，authored pack 对未声明轴必须诚实 NE。"""
    source_packs = tuple(
        item for item in formal_manifest.pack_audits
        if item.pack_kind == "SOURCE_PACK")
    authored = tuple(
        item for item in formal_manifest.pack_audits
        if item.pack_kind == "AUTHORED_COURSE")
    for item in source_packs:
        assert "UNDECLARED" not in item.axis_states.to_value().values()
    for item in authored:
        states = item.axis_states.to_value()
        assert states["LANGUAGE"] == "BASELINE_ONLY"
        assert states["SOURCE"] == "BASELINE_ONLY"
        assert all(states[axis] == "UNDECLARED"
                   for axis in TRANSFER_AXIS_KEYS
                   if axis not in {"LANGUAGE", "SOURCE"})
        assert item.combination_split_state == "NE_AXIS_UNDECLARED"


def test_wikidata_language_scope_comes_from_explicit_source_axis(
        formal_manifest):
    """Wikidata 的多语言来源范围不得被 Observation.language=zh 覆盖。"""
    wikidata = next(
        item for item in formal_manifest.pack_audits
        if "WIKIDATA_REVISION_V1" in item.source_keys)
    assert wikidata.axis_values.to_value()["LANGUAGE"] == [
        "multilingual_with_zh_allowlist"]
    assert wikidata.axis_states.to_value()["LANGUAGE"] == "BASELINE_ONLY"


def test_three_preregistered_split_probes_are_direct_pass_evidence(
        formal_manifest):
    """单轴、双轴和完整组合 fixture 必须列全且由规则直接判 PASS。"""
    assert tuple(item.probe_kind for item in formal_manifest.split_probes) == (
        SPLIT_PROBE_KINDS)
    assert {item.verdict for item in formal_manifest.split_probes} == {"PASS"}
    assert {item.failure_code for item in formal_manifest.split_probes} == {
        "NONE"}
    assert all(item.host_learning_writes == 0
               for item in formal_manifest.split_probes)
    assert build_transfer_split_probes() == formal_manifest.split_probes


def test_split_probe_rejects_leak_and_unseen_components():
    """split 判定不得从字段存在或完整组合不同直接推导泛化通过。"""
    base = _combination("BASE")
    assert evaluate_transfer_split_probe(
        "FULL_COMBINATION", TRANSFER_AXIS_KEYS, (base,), (base,)) == (
            "REJECT", "COMPLETE_COMBINATION_LEAK")

    unseen = base.to_value()
    unseen["DOMAIN"] = "UNSEEN_DOMAIN"
    assert evaluate_transfer_split_probe(
        "FULL_COMBINATION", TRANSFER_AXIS_KEYS, (base,),
        (CanonicalJsonObject.from_value(unseen),)) == (
            "REJECT", "COMPONENT_VALUE_UNSEEN")

    held = base.to_value()
    held["SOURCE"] = "UNSEEN_SOURCE"
    assert evaluate_transfer_split_probe(
        "SINGLE_AXIS", ("DOMAIN",), (base,),
        (CanonicalJsonObject.from_value(held),)) == (
            "REJECT", "SINGLE_AXIS_VALUE_NOT_HELD_OUT")


def test_double_axis_pair_leak_is_rejected_without_exact_row_leak():
    """双轴组合见过时，即使完整行未见也必须 REJECT。"""
    first = _combination("PAIR").to_value()
    first["REGISTER"] = "R1"
    first["GENRE"] = "G1"
    first["SOURCE"] = "S1"
    second = dict(first)
    second["REGISTER"] = "R2"
    second["SOURCE"] = "S1"
    third = dict(first)
    third["SOURCE"] = "S2"
    held = dict(second)
    held["SOURCE"] = "S2"
    train = tuple(CanonicalJsonObject.from_value(item)
                  for item in (first, second, third))
    verdict = evaluate_transfer_split_probe(
        "DOUBLE_AXIS", ("GENRE", "REGISTER"), train,
        (CanonicalJsonObject.from_value(held),))
    assert verdict == ("REJECT", "DOUBLE_AXIS_PAIR_LEAK")


def test_contract_rejects_runtime_authority_host_writes_and_transfer_pass(
        formal_manifest):
    """合同必须拒绝 runtime 权限、宿主写和 pack 级迁移 PASS。"""
    with pytest.raises(TransferAxisContractError, match="runtime transfer PASS"):
        replace(formal_manifest, runtime_transfer_pass_authority=1)
    execution = dict(EXECUTION_STATE)
    execution["teacher_calls"] = 1
    with pytest.raises(TransferAxisContractError, match="execution_state"):
        replace(
            formal_manifest,
            execution_state=CanonicalJsonObject.from_value(execution),
        )
    with pytest.raises(TransferAxisContractError, match="transfer PASS"):
        replace(formal_manifest.pack_audits[0], transfer_claim_state="PASS")


def test_catalog_rejects_bad_hash_and_inventory_drift(monkeypatch):
    """正式 pack hash 或 inventory 任一漂移都必须 fail closed。"""
    references, _ = catalog._repository_pack_references(REPOSITORY)
    path = sorted(references)[0]
    with pytest.raises(TransferAxisCatalogError, match="manifest hash"):
        catalog._pack_audit(WORKSPACE, path, "0" * 64, references[path][1])

    inventory = catalog._formal_pack_inventory(WORKSPACE)
    monkeypatch.setattr(
        catalog, "_formal_pack_inventory",
        lambda workspace: inventory + ("ph2_dataset_artifacts/missing/manifest.json",),
    )
    with pytest.raises(TransferAxisCatalogError, match="inventory 未闭合"):
        build_repository_transfer_axis_manifest(REPOSITORY, WORKSPACE)


def test_manifest_round_trip_nonoverwrite_and_corruption_fail_closed(
        tmp_path, formal_manifest):
    """LC-09 artifact 必须规范回读、幂等写入并拒绝覆盖或损坏。"""
    path = tmp_path / LC09_MANIFEST_PATH.name
    write_transfer_axis_manifest(formal_manifest, path)
    assert read_transfer_axis_manifest(path) == formal_manifest
    write_transfer_axis_manifest(formal_manifest, path)
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(TransferAxisContractError):
        read_transfer_axis_manifest(path)
    with pytest.raises(TransferAxisContractError, match="内容不同"):
        write_transfer_axis_manifest(formal_manifest, path)


def test_manifest_build_is_bit_identical_and_has_no_private_paths(
        formal_manifest):
    """重复构建必须逐字节一致，artifact 不得携带本机绝对路径。"""
    rebuilt = build_repository_transfer_axis_manifest(REPOSITORY, WORKSPACE)
    assert rebuilt.canonical_bytes() == formal_manifest.canonical_bytes()
    payload = formal_manifest.canonical_bytes()
    assert b"D:\\" not in payload
    assert b"127.0.0.1" not in payload
    assert b"proxy" not in payload.lower()


def test_formal_repository_manifest_is_exact(formal_manifest):
    """仓库正式 LC-09 manifest 必须绑定当前构建和固定 SHA-256。"""
    path = REPOSITORY / LC09_MANIFEST_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        FORMAL_MANIFEST_SHA256)
    restored = read_transfer_axis_manifest(path)
    assert restored == formal_manifest
    assert restored.sha256() == FORMAL_MANIFEST_SHA256
