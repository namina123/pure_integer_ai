"""FT00-02 v2 variable-count registry、budget 与 generic trainer preflight。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2GenericTrainer,
    V2PackEntry,
    V2PackRegistry,
    V2RegistryError,
    V2TrainPlan,
)


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64


def _range_keys(base: int, count: int) -> tuple[list[int] | None, list[int] | None]:
    """形成非空文件的首末稳定键，空文件不声明范围。"""
    if count == 0:
        return None, None
    return [base], [base + count - 1]


def _file(
        kind: str,
        owner: str,
        relative_path: str,
        split: str | None,
        count: int,
        cluster: int,
        base: int,
        ) -> dict:
    """形成只含文件身份的公开 manifest entry。"""
    first, last = _range_keys(base, count)
    return {
        "content_sha256": SHA1,
        "content_size_bytes": count * 10,
        "first_record_key": first,
        "last_record_key": last,
        "license_partition": "CC0-1.0",
        "owner_kind": owner,
        "record_count": count,
        "record_kind": kind,
        "relative_path": relative_path,
        "source_cluster_keys": [[cluster]],
        "split": split,
        "transport_sha256": SHA2,
        "transport_size_bytes": count * 8,
    }


def _manifest(
        pack_key: int,
        cluster: int,
        *,
        source_count: int = 1,
        train_count: int = 1,
        dev_count: int = 1,
        teacher_count: int = 1,
        evaluator_count: int = 1,
        earliest: str = "W-02",
        format_version: int = 2,
        ) -> dict:
    """形成变量计数、四 owner 分离且没有 payload 的 manifest。"""
    files = [
        _file("source_ref", "source", "source_refs.jsonl.gz", None,
              source_count, cluster, pack_key * 1000 + 1),
        _file("observation", "observation", "observations/train.jsonl.gz",
              "train", train_count, cluster, pack_key * 1000 + 100),
        _file("observation", "observation", "observations/dev.jsonl.gz",
              "dev", dev_count, cluster, pack_key * 1000 + 200),
        _file("teacher_evidence", "teacher",
              "owners/teacher/train.evidence.jsonl.gz", "train",
              teacher_count, cluster, pack_key * 1000 + 300),
        _file("evaluator_label", "evaluator",
              "owners/evaluator/dev.labels.jsonl.gz", "dev",
              evaluator_count, cluster, pack_key * 1000 + 400),
    ]
    files.sort(key=lambda item: item["relative_path"])
    total = sum(item["record_count"] for item in files)
    return {
        "adapter_version": format_version,
        "artifact_version": format_version,
        "course_version": format_version,
        "dataset_key": [pack_key],
        "earliest_invalidated_stage": earliest,
        "files": files,
        "format_version": format_version,
        "generator_version": format_version,
        "license_partition": "CC0-1.0",
        "parser_version": format_version,
        "prerequisite_manifest_keys": [],
        "record_count": total,
        "record_kind": "artifact_manifest",
        "redistribution_policy": "PUBLIC",
        "schema_version": format_version,
        "source_cluster_keys": [[cluster]],
        "source_key": "AUTHORED_CC0",
        "splits": ["train", "dev"],
        "stable_key": [pack_key],
        "w_stages": [earliest],
    }


def _write_manifest(root: Path, relative: str, value: dict) -> str:
    """以规范 JSON 写入测试 manifest，刻意不创建其 payload 文件。"""
    path = root / Path(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_line(value))
    return relative


def _source(stable: int = 1, cluster: int = 100) -> dict:
    """形成 generic trainer 使用的最小公开 SourceRef。"""
    return {
        "artifact_key": [2],
        "attribution": "FT00-02 authored fixture",
        "course_version": 2,
        "dataset_key": [1],
        "format_version": 2,
        "license_id": "CC0-1.0",
        "local_sha256": SHA1,
        "official_url": f"urn:ft00-02:{stable}",
        "parser_version": 2,
        "record_kind": "source_ref",
        "record_ordinal": stable,
        "redistribution_policy": "PUBLIC",
        "revision_id": "",
        "schema_version": 2,
        "snapshot_id": "FT00-02-V1",
        "source_cluster_key": [cluster],
        "source_identity": f"AUTHORED_CC0:{stable}",
        "source_key": "AUTHORED_CC0",
        "source_span": {
            "document_cluster_key": [cluster + 1],
            "entity_graph_cluster_key": [cluster + 2],
            "locator_kind": "record",
            "locator_value": str(stable),
            "span_end": 1,
            "span_start": 0,
        },
        "stable_key": [stable],
        "upstream_checksum": "sha256:" + SHA0,
    }


def _observation(stable: int = 2, source_ref: int = 1,
                 cluster: int = 200, split: str = "train") -> dict:
    """形成保留 carrier 结构的最小 Observation。"""
    return {
        "artifact_key": [2],
        "content_group_key": [cluster + 1],
        "course_version": 2,
        "dataset_key": [1],
        "dedup_cluster_key": [cluster],
        "epistemic_role": "forming",
        "format_version": 2,
        "language": "zh",
        "license_partition": "CC0-1.0",
        "logical_order": stable,
        "payload_kind": "typed_carrier",
        "perturbation_kind": "NONE",
        "prerequisite_keys": [],
        "record_kind": "observation",
        "representation": "plain_text",
        "sample_role": "support",
        "schema_version": 2,
        "shape_group_key": [cluster + 4],
        "source_ref_key": [source_ref],
        "split": split,
        "stable_key": [stable],
        "substage": "FT01_FORMAL_FOUNDATION",
        "supersedes_key": None,
        "template_group_key": [cluster + 3],
        "typed_payload": {
            "carrier": {
                "carrier_kind": "plain_text",
                "edges": [],
                "nodes": [{
                    "attributes": {},
                    "node_key": [cluster + 5],
                    "node_kind": "content",
                    "parent_node_key": None,
                    "span_end": 1,
                    "span_start": 0,
                }],
                "raw_text_sha256": SHA0,
                "root_node_keys": [[cluster + 5]],
            },
            "language_payload": {"text": "甲"},
        },
        "w_stage": "W-02",
    }


def _teacher(stable: int = 3, observation: int = 2, source_ref: int = 1) -> dict:
    """形成绑定 train observation 的 teacher evidence。"""
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


def _label(stable: int = 4, observation: int = 2) -> dict:
    """形成 evaluator label，仅用于 trainer 拒绝路径。"""
    return {
        "artifact_key": [2],
        "budget_units": 1,
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


def test_variable_count_registry_is_manifest_only_and_order_independent(tmp_path: Path) -> None:
    """1/2 pack 计数来自 manifest，payload 缺失也不影响 registry 建立。"""
    first = _write_manifest(
        tmp_path, "data/packs/a.json", _manifest(1, 100, source_count=2,
                                                   train_count=3, dev_count=1))
    second = _write_manifest(
        tmp_path, "data/packs/b.json", _manifest(2, 200, source_count=1,
                                                   train_count=2, dev_count=2,
                                                   earliest="W-03"))
    registry = V2PackRegistry.from_manifest_paths(tmp_path, [second, first])
    assert tuple(item.pack_key for item in registry.entries) == ((1,), (2,))
    snapshot = registry.snapshot()
    assert snapshot.pack_count == 2
    assert snapshot.total_record_count == 15
    assert snapshot.source_ref_count == 3
    assert dict(snapshot.observation_counts) == {"train": 5, "dev": 3}
    assert snapshot.teacher_evidence_count == 2
    assert dict(snapshot.evaluator_label_counts) == {"dev": 2}
    assert snapshot.source_cluster_count == 2
    w02 = registry.train_plan("W-02", scale_key="P0")
    assert w02.pack_keys == ((1,),)
    assert (w02.source_ref_count, w02.observation_count,
            w02.teacher_evidence_count, w02.total_input_count) == (2, 3, 1, 6)
    w03 = registry.train_plan("W-03", scale_key="P0")
    assert w03.pack_keys == ((1,), (2,))
    assert w03.total_input_count == 10
    assert w03 == V2PackRegistry.from_manifest_paths(
        tmp_path, [first, second]).train_plan("W-03", scale_key="P0")


def test_registry_rejects_legacy_private_escape_and_duplicate_identity(tmp_path: Path) -> None:
    """旧格式、旧/private 路径、逃逸路径和重复身份均 fail closed。"""
    valid = _write_manifest(tmp_path, "data/packs/a.json", _manifest(1, 100))
    legacy = _write_manifest(
        tmp_path, "data/packs/legacy.json", _manifest(3, 300, format_version=1))
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, [legacy])
    private = _write_manifest(tmp_path, "private/pack.json", _manifest(4, 400))
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, [private])
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, ["../outside.json"])
    old = _write_manifest(tmp_path, "data/ph2/manifests/d03_v1/old.json",
                          _manifest(5, 500))
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, [old])
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, [valid, valid])
    duplicate_key = _write_manifest(tmp_path, "data/packs/duplicate.json",
                                    _manifest(1, 600))
    with pytest.raises(V2RegistryError):
        V2PackRegistry.from_manifest_paths(tmp_path, [valid, duplicate_key])


def test_registry_rejects_source_cluster_overlap_and_enforces_single_plan_budget(
        tmp_path: Path,
        ) -> None:
    """source cluster 不能跨 pack，P0 约束单次 train plan 而非整个 release。"""
    first = _write_manifest(tmp_path, "data/packs/a.json", _manifest(1, 100))
    overlap = _write_manifest(tmp_path, "data/packs/overlap.json", _manifest(2, 100))
    with pytest.raises(V2RegistryError, match="cluster"):
        V2PackRegistry.from_manifest_paths(tmp_path, [first, overlap])

    pass_path = _write_manifest(
        tmp_path, "data/packs/p0-pass.json",
        _manifest(3, 300, source_count=1, train_count=3198,
                   dev_count=0, teacher_count=1, evaluator_count=0))
    registry = V2PackRegistry.from_manifest_paths(tmp_path, [pass_path])
    assert registry.train_plan("W-02", scale_key="P0").total_input_count == 3200
    with pytest.raises(V2RegistryError, match="budget"):
        registry.train_plan("W-02", scale_key="P0", max_records=3199)
    over_path = _write_manifest(
        tmp_path, "data/packs/p0-over.json",
        _manifest(4, 400, source_count=1, train_count=3199,
                   dev_count=0, teacher_count=1, evaluator_count=0))
    over = V2PackRegistry.from_manifest_paths(tmp_path, [over_path])
    with pytest.raises(V2RegistryError, match="budget"):
        over.train_plan("W-02", scale_key="P0")
    with pytest.raises(V2RegistryError):
        registry.train_plan("W-02", scale_key="P3")


def test_generic_trainer_accepts_only_train_owner_records_and_is_read_only(
        tmp_path: Path,
        ) -> None:
    """generic preflight 只接受三类 train 输入且顺序变化不改变 commitment。"""
    path = _write_manifest(tmp_path, "data/packs/a.json",
                           _manifest(1, 100, dev_count=0, evaluator_count=1))
    registry = V2PackRegistry.from_manifest_paths(tmp_path, [path])
    plan = V2GenericTrainer().prepare(
        registry, registry.train_plan("W-02", scale_key="P0"))
    values = [_source(), _observation(), _teacher()]
    trainer = V2GenericTrainer()
    result = trainer.validate_train_records(
        plan, values, teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    reversed_result = trainer.validate_train_records(
        plan, list(reversed(values)), teacher_owner_key=(9001,),
        evaluator_owner_key=(9002,))
    assert result.input_commitment == reversed_result.input_commitment
    assert (result.candidate_writes, result.core_writes, result.teacher_calls) == (0, 0, 0)
    with pytest.raises(V2RegistryError, match="只接受 train"):
        trainer.validate_train_records(
            plan, [_source(), _observation(split="held_out"), _teacher()],
            teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    with pytest.raises(V2RegistryError, match="只接受"):
        trainer.validate_train_records(
            plan, [*_values_without_label(values), _label()],
            teacher_owner_key=(9001,), evaluator_owner_key=(9002,))
    with pytest.raises(V2RegistryError, match="只接受"):
        trainer.validate_train_records(
            plan, [*_values_without_label(values), _manifest(9, 900)],
            teacher_owner_key=(9001,), evaluator_owner_key=(9002,))


def _values_without_label(values: list[dict]) -> list[dict]:
    """测试中返回独立副本，避免后续分支共享可变字典。"""
    return [dict(item) for item in values]


def test_registry_value_guards_reject_bool_and_bad_sha(tmp_path: Path) -> None:
    """内部 immutable value 也保持严格整数、排序和 SHA 合同。"""
    path = _write_manifest(tmp_path, "data/packs/a.json", _manifest(1, 100))
    registry = V2PackRegistry.from_manifest_paths(tmp_path, [path])
    entry = registry.entries[0]
    with pytest.raises(V2RegistryError):
        replace(entry, pack_key=(True,))
    plan = registry.train_plan("W-02", scale_key="P0")
    with pytest.raises(V2RegistryError):
        replace(plan, source_ref_count=True)
    with pytest.raises(V2RegistryError):
        replace(plan, manifest_commitment="A" * 64)
