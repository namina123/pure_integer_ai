"""D-02F 全资料包小批 pilot、恢复、隔离和确定性 T0。"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    JSONL_RECORD_KINDS,
    SPLITS,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_dataset_pilot import (
    FAULT_CODES,
    DatasetPilotError,
    DatasetPilotInterrupted,
    DatasetPilotRunResult,
    run_dataset_pilot,
)
from pure_integer_ai.experiments.ph2_dataset_pilot_registry import (
    PILOT_PACK_SPECS,
)
from pure_integer_ai.experiments.ph2_dataset_pilot_state import (
    register_pilot_tables,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATOMIC_PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--atomic-v1"
EXPECTED_NORMATIVE_SHA256 = (
    "5864003d99518a78d80c1a486db9813196931600c96076df92b1e4c89f0da9aa")


@dataclass(frozen=True)
class _BaselinePilot:
    """保存一次真实 21 包 Dict fresh pilot 及其可复用 release。"""

    release_root: Path
    result: DatasetPilotRunResult


@pytest.fixture(scope="module")
def baseline_pilot(tmp_path_factory) -> _BaselinePilot:
    """只运行一次基线，后续反向破坏复用未受影响 pack。"""
    release_root = tmp_path_factory.mktemp("d02f-baseline")
    backend = DictBackend()
    try:
        result = run_dataset_pilot(
            REPOSITORY_ROOT, release_root, backend, worker_count=1)
    finally:
        backend.close()
    return _BaselinePilot(release_root, result)


def _report(result: DatasetPilotRunResult) -> dict:
    """恢复独立规范 report object。"""
    return result.report.to_value()


def _seed_unaffected_packs(source: Path, target: Path) -> None:
    """复制除 atomic 外的基线 pack，使每个坏输入只重新执行目标包。"""
    source_packs = source / "packs"
    target_packs = target / "packs"
    target_packs.mkdir(parents=True)
    for pack in source_packs.iterdir():
        if pack.name == ATOMIC_PACK_NAME:
            continue
        shutil.copytree(pack, target_packs / pack.name)


def test_registry_is_exactly_twenty_authored_packs_plus_split_probe():
    """固定输入只含当前 D-02 资料和 probe，不含 WuDao/归档/工程根遗留源。"""
    assert len(PILOT_PACK_SPECS) == 21
    assert [spec.pack_id for spec in PILOT_PACK_SPECS] == list(range(1, 22))
    assert sum(spec.synthetic for spec in PILOT_PACK_SPECS) == 1
    samples = [
        spec.sample_relative_path for spec in PILOT_PACK_SPECS
        if spec.sample_relative_path is not None]
    assert len(samples) == 20
    assert all(path.startswith("data/ph2/") for path in samples)
    assert all("wudao" not in path.casefold() for path in samples)
    assert all("archive" not in path.casefold() for path in samples)


def test_full_pilot_reads_all_records_hashes_counts_and_never_forms_mastery(
        baseline_pilot):
    """21 包重读、四 record kind、四 split、未来负例和零训练写全部有报告证据。"""
    result = baseline_pilot.result
    report = _report(result)
    assert result.normative_sha256 == EXPECTED_NORMATIVE_SHA256
    assert result.resumed_pack_count == 0
    assert result.published_pack_count == 21
    assert report["pipeline_usable"] == 1
    assert report["pack_count"] == 21
    assert report["successful_pack_count"] == 21
    assert report["anomaly_count"] == 0
    assert set(report["record_counts"]) == set(JSONL_RECORD_KINDS)
    assert all(report["record_counts"][kind] > 0 for kind in JSONL_RECORD_KINDS)
    assert report["record_counts"] == {
        "evaluator_label": 73,
        "observation": 232,
        "source_ref": 232,
        "teacher_evidence": 159,
    }
    assert report["split_observation_counts"] == {
        "adversarial": 1,
        "dev": 1,
        "held_out": 71,
        "train": 159,
    }
    assert report["stage_visibility_valid_view_count"] == 18
    assert report["future_stage_leak_rejection_count"] == 16
    assert report["physical_split_audit"]["physical_split_isolation"] == 1
    assert report["physical_split_audit"]["source_cluster_disjoint"] == 1
    assert report["training_state_write_count"] == 0
    assert report["held_out_training_state_write_count"] == 0
    assert report["evaluator_training_state_write_count"] == 0
    assert report["v06_clone_host_write_count"] == 0
    assert report["v06_clone_training_state_write_count"] == 0
    assert report["formal_training_started"] == 0
    assert report["teacher_call_count"] == 0
    assert report["mastered"] == 0
    assert report["memory_enabled"] == 0
    assert report["companion_enabled"] == 0


def test_report_and_every_manifest_are_canonical_and_directly_rereadable(
        baseline_pilot):
    """报告及全部 manifest/gzip 文件由磁盘事实直接复核，不信任 backend 缓存。"""
    report_path = baseline_pilot.result.report_path
    payload = report_path.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    restored = parse_canonical_json_bytes(payload[:-1], require_object=True)
    assert restored == _report(baseline_pilot.result)
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_NORMATIVE_SHA256

    pack_count = 0
    record_count = 0
    for pack_root in sorted((baseline_pilot.release_root / "packs").iterdir()):
        manifest = read_artifact_manifest(pack_root / "manifest.json")
        assert manifest.sha256() == hashlib.sha256(
            (pack_root / "manifest.json").read_bytes()).hexdigest()
        reread = tuple(
            record
            for identity in manifest.files
            for record in read_record_artifact(pack_root, identity)
        )
        assert len(reread) == manifest.record_count
        record_count += len(reread)
        pack_count += 1
    assert pack_count == 21
    assert record_count == sum(_report(
        baseline_pilot.result)["record_counts"].values())


def test_split_probe_uses_four_observation_files_and_disjoint_owner_files(
        baseline_pilot):
    """dev/adversarial 不是报告中的虚构位，而是独立 gzip、cluster 和 evaluator owner。"""
    probe_spec = next(spec for spec in PILOT_PACK_SPECS if spec.synthetic)
    pack_root = baseline_pilot.release_root / "packs" / probe_spec.pack_name
    manifest = read_artifact_manifest(pack_root / "manifest.json")
    assert manifest.splits == SPLITS[:4]
    observation_files = {
        identity.split: identity for identity in manifest.files
        if identity.record_kind == "observation"}
    assert set(observation_files) == set(SPLITS[:4])
    assert {
        identity.relative_path for identity in observation_files.values()
    } == {
        f"observations/{split}.jsonl.gz" for split in SPLITS[:4]}
    assert len({
        identity.source_cluster_keys[0]
        for identity in observation_files.values()
    }) == 4
    teacher_splits = {
        identity.split for identity in manifest.files
        if identity.record_kind == "teacher_evidence"}
    evaluator_splits = {
        identity.split for identity in manifest.files
        if identity.record_kind == "evaluator_label"}
    assert teacher_splits == {"train"}
    assert evaluator_splits == set(SPLITS[1:4])


def test_dict_sqlite_and_one_two_four_workers_are_normatively_identical(
        tmp_path, baseline_pilot):
    """全部支持的 backend/worker 组合从 fresh 编译得到同一规范产物。"""
    expected = baseline_pilot.result.normative_sha256
    variants = (
        ("dict", 2),
        ("dict", 4),
        ("sqlite", 1),
        ("sqlite", 2),
        ("sqlite", 4),
    )
    for backend_kind, worker_count in variants:
        backend = (
            DictBackend() if backend_kind == "dict" else SQLiteBackend())
        try:
            result = run_dataset_pilot(
                REPOSITORY_ROOT,
                tmp_path / f"{backend_kind}-{worker_count}",
                backend,
                worker_count=worker_count,
            )
        finally:
            backend.close()
        assert result.normative_sha256 == expected
        assert result.report.to_value() == baseline_pilot.result.report.to_value()


def test_dict_and_sqlite_resume_adopt_orphan_pack_and_match_fresh(
        tmp_path, baseline_pilot):
    """中断点位于 pack 发布后/cursor 提交前，两个 backend 均精确恢复。"""
    expected = baseline_pilot.result.normative_sha256

    dict_release = tmp_path / "dict-release"
    first_dict = DictBackend()
    try:
        with pytest.raises(DatasetPilotInterrupted, match="cursor 尚未提交"):
            run_dataset_pilot(
                REPOSITORY_ROOT,
                dict_release,
                first_dict,
                interrupt_after_publish_pack_id=6,
            )
        dict_state = first_dict.recovery_state_snapshot()
    finally:
        first_dict.close()
    assert (dict_release / "packs" / PILOT_PACK_SPECS[5].pack_name
            / "manifest.json").is_file()
    resumed_dict = DictBackend()
    try:
        register_pilot_tables(resumed_dict)
        resumed_dict.restore_recovery_state(dict_state)
        dict_result = run_dataset_pilot(
            REPOSITORY_ROOT,
            dict_release,
            resumed_dict,
            worker_count=4,
        )
        repeated = run_dataset_pilot(
            REPOSITORY_ROOT,
            dict_release,
            resumed_dict,
            worker_count=2,
        )
    finally:
        resumed_dict.close()
    assert dict_result.normative_sha256 == expected
    assert dict_result.resumed_pack_count == 5
    assert dict_result.published_pack_count == 15
    assert repeated.normative_sha256 == expected
    assert repeated.resumed_pack_count == 21
    assert repeated.published_pack_count == 0

    sqlite_release = tmp_path / "sqlite-release"
    sqlite_path = tmp_path / "pilot.sqlite3"
    first_sqlite = SQLiteBackend(str(sqlite_path))
    try:
        with pytest.raises(DatasetPilotInterrupted, match="cursor 尚未提交"):
            run_dataset_pilot(
                REPOSITORY_ROOT,
                sqlite_release,
                first_sqlite,
                interrupt_after_publish_pack_id=6,
            )
    finally:
        first_sqlite.close()
    resumed_sqlite = SQLiteBackend(str(sqlite_path))
    try:
        sqlite_result = run_dataset_pilot(
            REPOSITORY_ROOT,
            sqlite_release,
            resumed_sqlite,
            worker_count=2,
        )
    finally:
        resumed_sqlite.close()
    assert sqlite_result.normative_sha256 == expected
    assert sqlite_result.resumed_pack_count == 5
    assert sqlite_result.published_pack_count == 15


@pytest.mark.parametrize("fault_code", FAULT_CODES)
def test_each_bad_input_is_exactly_one_anomaly_and_other_packs_are_unchanged(
        fault_code, tmp_path, baseline_pilot):
    """坏样本/来源/许可/supersede/扰动只隔离 atomic，不破坏其余 20 包。"""
    release_root = tmp_path / fault_code.casefold()
    _seed_unaffected_packs(baseline_pilot.release_root, release_root)
    backend = DictBackend()
    try:
        result = run_dataset_pilot(
            REPOSITORY_ROOT,
            release_root,
            backend,
            faults={1: fault_code},
        )
    finally:
        backend.close()
    report = _report(result)
    assert report["pipeline_usable"] == 0
    assert report["anomaly_count"] == 1
    assert report["successful_pack_count"] == 20
    assert report["anomalies"][0]["pack_id"] == 1
    assert report["anomalies"][0]["anomaly_code"] == fault_code
    baseline_by_id = {
        item["pack_id"]: item for item in _report(
            baseline_pilot.result)["packs"]}
    fault_by_id = {item["pack_id"]: item for item in report["packs"]}
    for pack_id in range(2, 22):
        assert fault_by_id[pack_id] == baseline_by_id[pack_id]
    assert report["training_state_write_count"] == 0
    assert report["mastered"] == 0


def test_invalid_execution_dimensions_fail_before_pack_publish(tmp_path):
    """未知 worker、多 fault 和仓库根 release 均 fail-closed。"""
    backend = DictBackend()
    try:
        with pytest.raises(DatasetPilotError, match="worker_count"):
            run_dataset_pilot(
                REPOSITORY_ROOT, tmp_path / "bad-worker", backend,
                worker_count=3)
    finally:
        backend.close()
    backend = DictBackend()
    try:
        with pytest.raises(DatasetPilotError, match="一次只允许"):
            run_dataset_pilot(
                REPOSITORY_ROOT, tmp_path / "bad-faults", backend,
                faults={1: "BAD_LICENSE", 2: "BAD_SOURCE"})
    finally:
        backend.close()
    backend = DictBackend()
    try:
        with pytest.raises(DatasetPilotError, match="不得等于"):
            run_dataset_pilot(REPOSITORY_ROOT, REPOSITORY_ROOT, backend)
    finally:
        backend.close()
