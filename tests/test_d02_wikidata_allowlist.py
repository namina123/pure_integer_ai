"""D-02 Wikidata bounded QID/property allowlist 与 claim 保留合同 T0。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_wikidata_allowlist import (
    REQUIRED_CLAIM_CONTRACT,
    WikidataAllowlistError,
    read_wikidata_allowlist,
    wikidata_entity_url,
)


ALLOWLIST_PATH = Path("data/ph2/wikidata_revision_v1_allowlist.json")
CURRENT_ALLOWLIST_PATH = Path(
    "data/ph2/wikidata_revision_v1_allowlist_v2.json")


def _value() -> dict:
    """读取规范 allowlist 为可复制测试 object。"""
    payload = CURRENT_ALLOWLIST_PATH.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict) -> None:
    """写一个规范 JSON 反例。"""
    path.write_bytes(canonical_json_line(value))


def test_repository_allowlist_freezes_bounded_entities_properties_and_hash():
    """正式 v2 只含 11 QID/11 property，并有唯一规范 hash。"""
    allowlist = read_wikidata_allowlist(CURRENT_ALLOWLIST_PATH)
    assert allowlist.sha256() == (
        "4dcdff27653e38544f02aea38b5e3623b27d632bb134c42b6b4a65b682d986f6")
    assert allowlist.allowlist_revision == 2
    assert len(allowlist.entities) == 11
    assert len(allowlist.properties) == 11
    assert [item.qid for item in allowlist.entities] == [
        "Q89", "Q312", "Q313", "Q361", "Q362", "Q446", "Q1420",
        "Q5113", "Q25364", "Q82069695", "Q84263196",
    ]
    assert {item.split for item in allowlist.entities} == {
        "train", "dev", "held_out",
    }


def test_relation_directions_are_individually_registered_not_name_guessed():
    """八类关系 property 的方向逐项固定，typed property 另走 value。"""
    allowlist = read_wikidata_allowlist(CURRENT_ALLOWLIST_PATH)
    rules = {item.property_id: item for item in allowlist.properties}
    assert rules["P155"].direction == "OBJECT_TO_SUBJECT"
    assert rules["P156"].direction == "SUBJECT_TO_OBJECT"
    assert rules["P828"].direction == "OBJECT_TO_SUBJECT"
    assert rules["P1542"].direction == "SUBJECT_TO_OBJECT"
    assert rules["P361"].project_relation_family == "PART_OF"
    assert rules["P527"].project_relation_family == "HAS_PART"
    assert rules["P17"].direction == "TYPED_VALUE"
    assert rules["P571"].allowed_datatypes == ("time",)
    assert rules["P2048"].allowed_datatypes == ("quantity",)


def test_claim_contract_retains_qualifiers_rank_references_and_unknown_values():
    """qualifier/rank/reference/deprecated/somevalue/novalue 均不得静默丢失。"""
    allowlist = read_wikidata_allowlist(CURRENT_ALLOWLIST_PATH)
    assert allowlist.claim_contract == REQUIRED_CLAIM_CONTRACT
    assert allowlist.claim_contract["preserve_qualifiers"] == 1
    assert allowlist.claim_contract["preserve_rank"] == 1
    assert allowlist.claim_contract["preserve_references"] == 1
    assert allowlist.claim_contract["deprecated_statement_policy"] == (
        "RETAIN_EXCLUDE_POSITIVE_EVIDENCE")
    assert allowlist.claim_contract["somevalue_policy"] == "RETAIN_NONPOSITIVE"
    assert allowlist.claim_contract["novalue_policy"] == "RETAIN_NONPOSITIVE"


def test_cluster_never_crosses_split_and_overclustering_is_conservative():
    """同一来源簇只属一个 split；多义/配对实体保守同簇。"""
    allowlist = read_wikidata_allowlist(CURRENT_ALLOWLIST_PATH)
    by_cluster: dict[str, set[str]] = {}
    qids_by_cluster: dict[str, set[str]] = {}
    for item in allowlist.entities:
        by_cluster.setdefault(item.cluster_id, set()).add(item.split)
        qids_by_cluster.setdefault(item.cluster_id, set()).add(item.qid)
    assert all(len(splits) == 1 for splits in by_cluster.values())
    assert qids_by_cluster["apple-polysemy"] == {"Q89", "Q312"}
    assert qids_by_cluster["vehicle-mereology"] == {"Q446", "Q1420"}
    assert qids_by_cluster["war-sequence"] == {"Q361", "Q362"}
    assert qids_by_cluster["animal-taxonomy"] == {"Q5113", "Q25364"}
    assert qids_by_cluster["pandemic-cause"] == {"Q82069695", "Q84263196"}


def test_v2_supersedes_v1_and_only_repairs_the_inaccurate_cluster_name():
    """v1 字节保留；v2 只修正动物分类簇名并绑定前版 hash。"""
    previous = read_wikidata_allowlist(ALLOWLIST_PATH)
    current = read_wikidata_allowlist(CURRENT_ALLOWLIST_PATH)
    assert previous.allowlist_revision == 1
    assert previous.sha256() == (
        "c052a162ef203dda770e48524a150ef8da2446de20b6b9995de972c85aabae48")
    assert current.supersedes_sha256 == previous.sha256()
    previous_by_qid = {item.qid: item for item in previous.entities}
    current_by_qid = {item.qid: item for item in current.entities}
    assert previous_by_qid.keys() == current_by_qid.keys()
    changed = {
        qid for qid in previous_by_qid
        if previous_by_qid[qid] != current_by_qid[qid]
    }
    assert changed == {"Q5113", "Q25364"}
    for qid in changed:
        assert previous_by_qid[qid].cluster_id == "bird-taxonomy"
        assert current_by_qid[qid].cluster_id == "animal-taxonomy"
        assert previous_by_qid[qid].split == current_by_qid[qid].split
        assert previous_by_qid[qid].purpose_keys == current_by_qid[qid].purpose_keys
    assert previous.properties == current.properties
    assert previous.claim_contract == current.claim_contract


def test_entity_urls_are_official_and_revision_pinning_is_explicit():
    """发现 URL 与正式 revision URL 分离，坏 QID/revision fail closed。"""
    assert wikidata_entity_url("Q313") == (
        "https://www.wikidata.org/wiki/Special:EntityData/Q313.json")
    assert wikidata_entity_url("Q313", revision=2400000000) == (
        "https://www.wikidata.org/wiki/Special:EntityData/Q313.json?"
        "revision=2400000000")
    with pytest.raises(WikidataAllowlistError, match="QID"):
        wikidata_entity_url("Q0313")
    with pytest.raises(WikidataAllowlistError, match="revision"):
        wikidata_entity_url("Q313", revision=0)


def test_bad_qid_duplicate_and_cluster_split_are_rejected(tmp_path):
    """坏 QID、重复 QID 和同 cluster 跨 split 均拒绝。"""
    value = _value()
    value["entity_allowlist"][0]["qid"] = "Q089"
    path = tmp_path / "bad-qid.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    value["entity_allowlist"][1]["qid"] = value["entity_allowlist"][0]["qid"]
    path = tmp_path / "duplicate.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    value["entity_allowlist"][1]["split"] = "dev"
    path = tmp_path / "cross-split.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)


def test_missing_property_wrong_direction_and_contract_relaxation_are_rejected(
        tmp_path,
        ):
    """property 注册缺失、方向错和 claim 字段放宽均 fail closed。"""
    value = _value()
    value["property_allowlist"] = value["property_allowlist"][:-1]
    path = tmp_path / "missing-property.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    for rule in value["property_allowlist"]:
        if rule["property_id"] == "P828":
            rule["direction"] = "SUBJECT_TO_OBJECT"
    path = tmp_path / "wrong-direction.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    value["claim_contract"]["preserve_qualifiers"] = 0
    path = tmp_path / "relaxed-contract.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)


def test_noncanonical_float_and_outer_whitespace_are_rejected(tmp_path):
    """float、非规范 key 顺序和多余换行不能进入 allowlist。"""
    float_path = tmp_path / "float.json"
    float_path.write_bytes(CURRENT_ALLOWLIST_PATH.read_bytes().replace(
        b'"format_version":1', b'"format_version":1.0'))
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(float_path)

    whitespace_path = tmp_path / "whitespace.json"
    whitespace_path.write_bytes(CURRENT_ALLOWLIST_PATH.read_bytes() + b"\n")
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(whitespace_path)


def test_bad_revision_supersede_and_unregistered_entity_drift_are_rejected(
        tmp_path,
        ):
    """未知版本、坏前版 hash 和未注册实体改写均不得进入正式获取。"""
    value = _value()
    value["allowlist_revision"] = 3
    path = tmp_path / "bad-revision.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    value["supersedes_sha256"] = "0" * 64
    path = tmp_path / "bad-supersede.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)

    value = _value()
    value["entity_allowlist"][0]["cluster_id"] = "search-picked"
    path = tmp_path / "entity-drift.json"
    _write(path, value)
    with pytest.raises(WikidataAllowlistError):
        read_wikidata_allowlist(path)
