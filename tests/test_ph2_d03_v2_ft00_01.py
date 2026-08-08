"""FT00-01 successor schema、owner、身份、失效和独占发布专项。"""
from __future__ import annotations

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import D03ContractError
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V1_PUBLIC_RECEIPT_PATH,
    V1_PUBLIC_RECEIPT_SHA256,
    V1_PUBLIC_RECEIPT_SIZE_BYTES,
    V2_EXECUTION_STAGES,
    V2_INITIAL_EXECUTION_STATE,
    V2_LOGICAL_SHARD_COUNT,
    V2_RELEASE_KEY,
    V2RunIdentity,
    build_v2_invalidation_rules,
    invalidation_suffix,
)
from pure_integer_ai.experiments.ph2_d03_v2_catalog import (
    build_v2_successor_contract,
    publish_v2_successor_contract,
    read_v2_successor_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    V2_CARRIER_KINDS,
    record_schema_bindings,
    validate_v2_record,
    validate_v2_record_set,
)
from pure_integer_ai.experiments.ph2_dataset_core import DatasetContractError


ROOT = Path(__file__).resolve().parents[1]
SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def _source(
        stable: int,
        source_cluster: int,
        document_cluster: int,
        entity_cluster: int,
        ) -> dict:
    """形成一个满足 v2 来源、许可和双簇定位的最小 SourceRef。"""
    return {
        "artifact_key": [2],
        "attribution": "FT00-01 authored fixture",
        "course_version": 2,
        "dataset_key": [1],
        "format_version": 2,
        "license_id": "CC0-1.0",
        "local_sha256": SHA1,
        "official_url": f"urn:ft00-01:{stable}",
        "parser_version": 2,
        "record_kind": "source_ref",
        "record_ordinal": stable,
        "redistribution_policy": "PUBLIC",
        "revision_id": "",
        "schema_version": 2,
        "snapshot_id": "FT00-01-V1",
        "source_cluster_key": [source_cluster],
        "source_identity": f"AUTHORED_CC0:{stable}",
        "source_key": "AUTHORED_CC0",
        "source_span": {
            "document_cluster_key": [document_cluster],
            "entity_graph_cluster_key": [entity_cluster],
            "locator_kind": "record",
            "locator_value": str(stable),
            "span_end": 1,
            "span_start": 0,
        },
        "stable_key": [stable],
        "upstream_checksum": "sha256:" + SHA0,
    }


def _carrier(kind: str, node: int) -> dict:
    """形成保留节点、root、span 和载体种类的最小 typed payload。"""
    return {
        "carrier": {
            "carrier_kind": kind,
            "edges": [],
            "nodes": [{
                "attributes": {},
                "node_key": [node],
                "node_kind": "content",
                "parent_node_key": None,
                "span_end": 1,
                "span_start": 0,
            }],
            "raw_text_sha256": SHA0,
            "root_node_keys": [[node]],
        },
        "language_payload": {"text": "甲"},
    }


def _observation(
        stable: int,
        source_ref: int,
        split: str,
        cluster_base: int,
        *,
        carrier_kind: str = "plain_text",
        ) -> dict:
    """形成一个 cluster 完整且保留载体的最小 Observation。"""
    return {
        "artifact_key": [2],
        "content_group_key": [cluster_base + 1],
        "course_version": 2,
        "dataset_key": [1],
        "dedup_cluster_key": [cluster_base],
        "epistemic_role": "forming",
        "format_version": 2,
        "language": "zh",
        "license_partition": "CC0-1.0",
        "logical_order": stable,
        "payload_kind": "typed_carrier",
        "perturbation_kind": "NONE",
        "prerequisite_keys": [],
        "record_kind": "observation",
        "representation": carrier_kind,
        "sample_role": "support",
        "schema_version": 2,
        "shape_group_key": [cluster_base + 4],
        "source_ref_key": [source_ref],
        "split": split,
        "stable_key": [stable],
        "substage": "FT01_FORMAL_FOUNDATION",
        "supersedes_key": None,
        "template_group_key": [cluster_base + 3],
        "typed_payload": _carrier(carrier_kind, cluster_base + 5),
        "w_stage": "W-02",
    }


def _teacher(stable: int, observation: int, source_ref: int) -> dict:
    """形成只绑定 train Observation 的 teacher Evidence。"""
    return {
        "artifact_key": [2],
        "course_version": 2,
        "dataset_key": [1],
        "evidence_kind": "AUTHORED_FORM",
        "format_version": 2,
        "observation_key": [observation],
        "owner_key": [9001],
        "record_kind": "teacher_evidence",
        "schema_version": 2,
        "source_ref_key": [source_ref],
        "stable_key": [stable],
        "typed_evidence": {"accepted": 1},
        "visible_from_stage": "W-02",
        "withdrawal_level": 0,
    }


def _label(stable: int, observation: int) -> dict:
    """形成只绑定非 train Observation 的 evaluator label。"""
    return {
        "artifact_key": [2],
        "budget_units": 10,
        "course_version": 2,
        "dataset_key": [1],
        "dimension_key": [7001],
        "evaluator_version": 2,
        "expected_payload": {"accepted": 1},
        "expected_state": "TRUE",
        "format_version": 2,
        "observation_key": [observation],
        "owner_key": [9002],
        "owner_mode": "read_only",
        "record_kind": "evaluator_label",
        "schema_version": 2,
        "stable_key": [stable],
        "visible_stage": "W-02",
    }


def _file(
        kind: str,
        owner: str,
        path: str,
        split: str | None,
        stable: int,
        clusters: list[list[int]],
        *,
        count: int = 1,
        ) -> dict:
    """形成带 canonical/transport 双 hash 的 manifest 文件身份。"""
    key = [stable] if count else None
    return {
        "content_sha256": SHA1,
        "content_size_bytes": count * 100,
        "first_record_key": key,
        "last_record_key": key,
        "license_partition": "CC0-1.0",
        "owner_kind": owner,
        "record_count": count,
        "record_kind": kind,
        "relative_path": path,
        "source_cluster_keys": clusters,
        "split": split,
        "transport_sha256": SHA2,
        "transport_size_bytes": count * 80,
    }


def _manifest(*, empty: bool = False) -> dict:
    """形成四类 owner 物理分离、split 闭合的 v2 ArtifactManifest。"""
    count = 0 if empty else 1
    files = [
        _file("source_ref", "source", "source_refs.jsonl.gz", None, 3,
              [[100], [110]], count=count),
        _file("observation", "observation", "observations/train.jsonl.gz",
              "train", 4, [[100]], count=count),
        _file("observation", "observation", "observations/held_out.jsonl.gz",
              "held_out", 14, [[110]], count=count),
        _file("teacher_evidence", "teacher",
              "owners/teacher/train.evidence.jsonl.gz", "train", 5,
              [[100]], count=count),
        _file("evaluator_label", "evaluator",
              "owners/evaluator/held_out.labels.jsonl.gz", "held_out", 15,
              [[110]], count=count),
    ]
    files.sort(key=lambda item: item["relative_path"])
    return {
        "adapter_version": 2,
        "artifact_version": 2,
        "course_version": 2,
        "dataset_key": [1],
        "earliest_invalidated_stage": "W-02",
        "files": files,
        "format_version": 2,
        "generator_version": 2,
        "license_partition": "CC0-1.0",
        "parser_version": 2,
        "prerequisite_manifest_keys": [],
        "record_count": sum(item["record_count"] for item in files),
        "record_kind": "artifact_manifest",
        "redistribution_policy": "PUBLIC",
        "schema_version": 2,
        "source_cluster_keys": [[100], [110]],
        "source_key": "AUTHORED_CC0",
        "splits": ["train", "held_out"],
        "stable_key": [20],
        "w_stages": ["W-02"],
    }


def _record_set() -> list[dict]:
    """形成 train/held-out 来源簇完全隔离的最小跨记录集合。"""
    return [
        _source(3, 100, 101, 102),
        _observation(4, 3, "train", 200),
        _teacher(5, 4, 3),
        _source(13, 110, 111, 112),
        _observation(14, 13, "held_out", 300),
        _label(15, 14),
        _manifest(),
    ]


def test_v2_schema_accepts_all_carriers_and_owner_separated_record_set() -> None:
    """九种载体共用 schema，且 teacher/evaluator 保持物理 split 隔离。"""
    for index, kind in enumerate(V2_CARRIER_KINDS):
        validate_v2_record(_observation(
            1000 + index, 3, "train", 2000 + index * 10,
            carrier_kind=kind,
        ))
    records = validate_v2_record_set(
        _record_set(), teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    assert len(records) == 7
    assert len(record_schema_bindings()) == 5


def test_v2_schema_rejects_flattening_unknown_fields_and_wrong_license() -> None:
    """拒绝展平载体、v1/未知字段和来源许可错配。"""
    flattened = _observation(4, 3, "train", 200)
    flattened["typed_payload"] = {"language_payload": {"text": "甲"}}
    with pytest.raises(DatasetContractError):
        validate_v2_record(flattened)
    unknown = _source(3, 100, 101, 102)
    unknown["legacy_field"] = 1
    with pytest.raises(DatasetContractError):
        validate_v2_record(unknown)
    wrong_license = _source(3, 100, 101, 102)
    wrong_license["license_id"] = "CC-BY-4.0"
    with pytest.raises(DatasetContractError):
        validate_v2_record(wrong_license)


def test_v2_source_license_partition_keeps_both_conceptnet_public_licenses() -> None:
    """ConceptNet 的两份公开许可分包都合法，但其他来源仍按自身许可闭集。"""
    for license_id in ("CC-BY-4.0", "CC-BY-SA-4.0"):
        source = _source(3, 100, 101, 102)
        source["source_key"] = "CONCEPTNET_5_7_0"
        source["license_id"] = license_id
        validate_v2_record(source)


def test_v2_record_set_rejects_cluster_and_owner_leakage() -> None:
    """同一 cluster 不得跨 split，teacher/evaluator 不得交换训练边界。"""
    cluster_leak = _record_set()
    cluster_leak[4]["content_group_key"] = cluster_leak[1]["content_group_key"]
    with pytest.raises(DatasetContractError):
        validate_v2_record_set(
            cluster_leak, teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    teacher_leak = _record_set()
    teacher_leak[2]["observation_key"] = [14]
    teacher_leak[2]["source_ref_key"] = [13]
    with pytest.raises(DatasetContractError):
        validate_v2_record_set(
            teacher_leak, teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    evaluator_leak = _record_set()
    evaluator_leak[5]["observation_key"] = [4]
    with pytest.raises(DatasetContractError):
        validate_v2_record_set(
            evaluator_leak, teacher_owner_key=(9001,), evaluator_owner_key=(9002,))


def test_v2_manifest_rejects_empty_pack_owner_overlap_and_path_drift() -> None:
    """pack 必须非空、双 hash 完整且四类 owner 路径不能漂移或重叠。"""
    manifest = validate_v2_record(_manifest())
    assert manifest.record_count == 5
    assert manifest.files[0].content_sha256 != manifest.files[0].transport_sha256
    with pytest.raises(DatasetContractError):
        validate_v2_record(_manifest(empty=True))
    wrong_path = _manifest()
    wrong_path["files"][3]["relative_path"] = "observations/train.jsonl.gz"
    with pytest.raises(DatasetContractError):
        validate_v2_record(wrong_path)
    traversal = _manifest()
    traversal["files"][0]["relative_path"] = "../source_refs.jsonl.gz"
    with pytest.raises(DatasetContractError):
        validate_v2_record(traversal)
    unordered = _manifest()
    unordered["files"] = list(reversed(unordered["files"]))
    with pytest.raises(DatasetContractError):
        validate_v2_record(unordered)


def test_v2_invalidation_suffixes_are_complete_and_unknown_fails_closed() -> None:
    """全局、pack 和 evaluator 变化都返回从最早阶段开始的完整后缀。"""
    rules = build_v2_invalidation_rules((
        ("PACK-W02", "W-02"),
        ("PACK-W08", "W-08"),
    ))
    assert invalidation_suffix(rules, "SCHEMA_VERSION", "GLOBAL") == V2_EXECUTION_STAGES
    assert invalidation_suffix(rules, "PACK_CONTENT", "PACK-W08") == (
        "W-08", "W-09", "PW")
    assert invalidation_suffix(rules, "EVALUATOR_VERSION", "PW") == ("PW",)
    with pytest.raises(D03ContractError):
        invalidation_suffix(rules, "PACK_CONTENT", "UNKNOWN")
    with pytest.raises(D03ContractError):
        build_v2_invalidation_rules((("PACK", "FT00"),))


def test_v2_run_identity_is_canonical_across_worker_configuration() -> None:
    """worker 数不进入语义 run identity，1/2/4 仅由执行策略授权。"""
    run = V2RunIdentity(
        V2_RELEASE_KEY, "W-02", "P0", 1, V2_LOGICAL_SHARD_COUNT, SHA0, "")
    assert V2RunIdentity.from_dict(run.to_dict()) == run
    assert run.sha256() == V2RunIdentity.from_dict(run.to_dict()).sha256()
    with pytest.raises(D03ContractError):
        V2RunIdentity(
            V2_RELEASE_KEY, "W-01", "P0", 1,
            V2_LOGICAL_SHARD_COUNT, SHA0, "")
    with pytest.raises(D03ContractError):
        V2RunIdentity(
            V2_RELEASE_KEY, "W-02", "P3", 1,
            V2_LOGICAL_SHARD_COUNT, SHA0, "")


def test_v2_contract_binds_only_public_v1_receipt_and_zero_state() -> None:
    """successor 只绑定公开 v1 canonical receipt，且不继承 mastery/readiness。"""
    contract = build_v2_successor_contract(ROOT)
    assert contract.release_key == "PH2-D03-V2"
    assert contract.prior_release_receipt.relative_path == V1_PUBLIC_RECEIPT_PATH
    assert contract.prior_release_receipt.size_bytes == V1_PUBLIC_RECEIPT_SIZE_BYTES
    assert contract.prior_release_receipt.sha256 == V1_PUBLIC_RECEIPT_SHA256
    assert contract.initial_state == V2_INITIAL_EXECUTION_STATE
    assert contract.run_policy["allowed_worker_counts"] == [1, 2, 4]
    assert contract.run_policy["p3_activation_policy"] == (
        "FREEZE_ONLY_AFTER_P2_SLOPE_PASS")
    assert contract.pack_invalidation_kinds == (
        "PACK_CONTENT", "SOURCE_SET", "LICENSE")
    assert contract.unknown_invalidation_policy == "FAIL_CLOSED"
    assert contract.owner_policies[3].allowed_splits == ("train", "dev")
    assert contract.source_licenses[1] == (
        "CONCEPTNET_5_7_0", ("CC-BY-4.0", "CC-BY-SA-4.0"))
    assert contract.to_dict() == type(contract).from_dict(contract.to_dict()).to_dict()


def test_v2_contract_parent_drift_fails_before_publish(tmp_path: Path) -> None:
    """v1 receipt 字节漂移时不得构建或发布 successor authority。"""
    parent = tmp_path / Path(*Path(V1_PUBLIC_RECEIPT_PATH).parts)
    parent.parent.mkdir(parents=True)
    shutil.copy2(ROOT / V1_PUBLIC_RECEIPT_PATH, parent)
    assert build_v2_successor_contract(tmp_path).prior_release_receipt.sha256 == (
        V1_PUBLIC_RECEIPT_SHA256)
    parent.write_bytes(parent.read_bytes() + b" ")
    with pytest.raises(D03ContractError):
        build_v2_successor_contract(tmp_path)


def test_v2_contract_publish_is_canonical_idempotent_and_non_overwriting(
        tmp_path: Path,
        ) -> None:
    """首次独占发布、同字节幂等和异字节拒绝同时成立。"""
    target = tmp_path / "v2-contract.json"
    first = publish_v2_successor_contract(ROOT, target)
    payload = first.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert hashlib.sha256(payload[:-1]).hexdigest() == read_v2_successor_contract(
        ROOT, target).sha256()
    assert publish_v2_successor_contract(ROOT, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(D03ContractError):
        publish_v2_successor_contract(ROOT, target)
