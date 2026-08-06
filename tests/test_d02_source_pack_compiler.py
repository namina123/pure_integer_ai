"""D-02 外部来源统一 pack、组合 cluster、owner grant 和恢复 T0。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackCompilerError,
    compile_or_resume_source_pack,
    read_source_pack_coverage_manifest,
    read_source_pack_view,
    validate_source_pack_payloads,
    write_source_pack_coverage_manifest,
)
from pure_integer_ai.experiments.ph2_source_pack_catalog import (
    SOURCE_PACK_COVERAGE_PATH,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
    SourcePackContractError,
    SourcePackCoverageEntry,
    SourcePackCoverageManifest,
    SourcePackSpec,
)
from pure_integer_ai.experiments.ph2_source_pack_runtime import (
    SOURCE_PACK_FAULT_CODES,
    SourcePackRuntimeError,
    SourcePackTask,
    run_source_pack_batch,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


def _spec(pack_id: int, *, license_id: str = "CC-BY-SA-4.0") -> SourcePackSpec:
    """构造不含网络或私有路径的 synthetic 来源 pack spec。"""
    return SourcePackSpec(
        f"SYNTHETIC_SOURCE_{pack_id}",
        license_id,
        "PUBLIC",
        f"snapshot-{pack_id}",
        f"https://example.invalid/source/{pack_id}",
        f"Synthetic contributors {pack_id}",
        f"data/ph2/manifests/synthetic_{pack_id}.raw_snapshot.json",
        f"{pack_id:x}"[-1] * 64,
        1,
        1,
        1,
        1,
        1,
        f"synthetic-source-{pack_id}-pack-v1",
        "W-03",
        "D-02-SOURCE-PACK-V1",
        "W-03",
    )


def _seed(pack_id: int, ordinal: int, split: str) -> SourceObservationSeed:
    """构造 raw 字节、来源和六类 cluster 均可独立审计的 seed。"""
    token = f"{pack_id}-{ordinal}-{split}"
    return SourceObservationSeed(
        f"seed-{token}",
        split,
        "zh",
        "raw-source-text",
        f"synthetic/{token}.txt#line=1",
        "sha256:" + f"{ordinal:x}"[-1] * 64,
        f"{pack_id + ordinal:x}"[-1] * 64,
        CanonicalJsonObject.from_value({
            "line_end": 1,
            "line_start": 1,
            "relative_path": f"synthetic/{token}.txt",
        }),
        CanonicalJsonObject.from_value({
            "raw_text": f" 原文 {token}\n第二行。 ",
            "title": f"标题-{token}",
        }),
        CanonicalJsonObject.from_value({
            "code_switch": "NONE",
            "domain": f"domain-{ordinal}",
            "genre": "fixture",
            "language": "zh",
            "length": "SHORT",
            "register": "neutral",
            "source": f"SYNTHETIC_SOURCE_{pack_id}",
        }),
        ("document", token),
        ("dedup", token),
        ("content", token),
        ("template", split, ordinal),
        ("shape", split, ordinal),
        ("combination", split, ordinal),
        "support" if split == "train" else "read_only_probe",
        "NONE",
        ordinal,
    )


def _task(pack_id: int) -> SourcePackTask:
    """构造同时覆盖 train/held-out 的来源批次任务。"""
    return SourcePackTask(
        pack_id,
        _spec(pack_id),
        (_seed(pack_id, 1, "train"), _seed(pack_id, 2, "held_out")),
    )


def test_source_pack_preserves_raw_and_enforces_owner_read_grants(tmp_path):
    """student 只读 Observation；raw、组合轴和四 owner 均有直接证据。"""
    task = _task(1)
    build = compile_or_resume_source_pack(
        task.spec, task.seeds, tmp_path / "release")
    assert build.published
    assert build.bundle.manifest.splits == ("train", "held_out")
    assert build.bundle.combination_audit.to_value()[
        "combination_cluster_count"] == 2

    student = read_source_pack_view(
        build.pack_root, reader_kind="student", split="train")
    teacher = read_source_pack_view(
        build.pack_root, reader_kind="teacher", split="train")
    evaluator = read_source_pack_view(
        build.pack_root, reader_kind="evaluator", split="train")
    sources = read_source_pack_view(
        build.pack_root, reader_kind="source_audit")
    assert all(isinstance(item, ObservationRecord) for item in student)
    assert all(isinstance(item, TeacherEvidenceRecord) for item in teacher)
    assert all(isinstance(item, EvaluatorLabelRecord) for item in evaluator)
    assert all(isinstance(item, SourceRefRecord) for item in sources)
    payload = student[0].typed_payload.to_value()
    assert payload["raw_observation"] == task.seeds[0].raw_observation.to_value()
    assert payload["raw_observation_append_only"] == 1
    assert "expected" not in payload
    with pytest.raises(SourcePackCompilerError, match="不接受 split"):
        read_source_pack_view(
            build.pack_root, reader_kind="source_audit", split="train")
    with pytest.raises(SourcePackCompilerError, match="split"):
        read_source_pack_view(
            build.pack_root, reader_kind="student", split="wall")


def test_source_pack_resume_is_idempotent_and_rejects_contract_drift(tmp_path):
    """fresh/resume 记录一致，输入漂移不能领养已有目录。"""
    task = _task(2)
    release = tmp_path / "release"
    first = compile_or_resume_source_pack(task.spec, task.seeds, release)
    resumed = compile_or_resume_source_pack(task.spec, task.seeds, release)
    assert first.published
    assert not resumed.published
    assert resumed.bundle == first.bundle
    changed = replace(
        task.seeds[0],
        raw_observation=CanonicalJsonObject.from_value({"raw_text": "changed"}),
    )
    with pytest.raises(SourcePackCompilerError, match="漂移"):
        compile_or_resume_source_pack(
            task.spec, (changed,) + task.seeds[1:], release)


def test_bad_private_payload_and_combination_cross_split_fail_atomically(tmp_path):
    """私有 expected 与完整组合 cluster 泄漏都不得留下半 pack。"""
    task = _task(3)
    raw = task.seeds[0].raw_observation.to_value()
    raw["expected"] = "private"
    bad_private = replace(
        task.seeds[0], raw_observation=CanonicalJsonObject.from_value(raw))
    release = tmp_path / "private"
    with pytest.raises(Exception, match="私有字段"):
        compile_or_resume_source_pack(
            task.spec, (bad_private,) + task.seeds[1:], release)
    assert not (release / "packs" / task.spec.pack_name).exists()

    bad_combo = replace(
        task.seeds[1], combination_parts=task.seeds[0].combination_parts)
    release = tmp_path / "combination"
    with pytest.raises(SourcePackCompilerError, match="combination cluster 跨 split"):
        compile_or_resume_source_pack(
            task.spec, (task.seeds[0], bad_combo), release)
    assert not (release / "packs" / task.spec.pack_name).exists()


def test_raw_observation_hash_damage_is_rejected_without_expected_cue():
    """raw receipt 必须重算正文，不能只信非空输出或 fixture 字段。"""
    task = _task(4)
    from pure_integer_ai.experiments.ph2_source_pack_compiler import (
        _records_from_inputs,
    )
    observation = _records_from_inputs(task.spec, task.seeds)[1][0]
    payload = observation.typed_payload.to_value()
    payload["raw_observation"]["raw_text"] = "damaged"
    damaged = replace(
        observation, typed_payload=CanonicalJsonObject.from_value(payload))
    with pytest.raises(SourcePackCompilerError, match="raw Observation hash"):
        validate_source_pack_payloads((damaged,))


def test_dict_sqlite_and_workers_are_normatively_identical(tmp_path):
    """统一来源批次在 Dict/SQLite、1/2/4 worker 下规范结果一致。"""
    tasks = (_task(11), _task(12))
    digests: set[str] = set()
    for backend_kind, worker_count in (
            ("dict", 1), ("dict", 2), ("dict", 4),
            ("sqlite", 1), ("sqlite", 2), ("sqlite", 4)):
        root = tmp_path / f"{backend_kind}-{worker_count}"
        if backend_kind == "dict":
            backend = DictBackend()
        else:
            backend = SQLiteBackend(str(tmp_path / f"{worker_count}.sqlite3"))
        try:
            result = run_source_pack_batch(
                tasks, root, backend, worker_count=worker_count)
        finally:
            backend.close()
        report = result.report.to_value()
        assert report["successful_pack_count"] == 2
        assert report["anomaly_count"] == 0
        assert report["teacher_call_count"] == 0
        assert report["training_state_write_count"] == 0
        assert report["owner_read_isolation"] == 1
        digests.add(result.normative_sha256)
    assert len(digests) == 1


def test_dict_and_sqlite_resume_match_fresh(tmp_path):
    """同 backend/release 恢复不重发 pack，规范报告与 fresh 相同。"""
    tasks = (_task(21), _task(22))
    for backend_kind in ("dict", "sqlite"):
        release = tmp_path / backend_kind
        backend = (
            DictBackend() if backend_kind == "dict"
            else SQLiteBackend(str(tmp_path / "resume.sqlite3")))
        try:
            fresh = run_source_pack_batch(
                tasks, release, backend, worker_count=1)
            resumed = run_source_pack_batch(
                tasks, release, backend, worker_count=4)
        finally:
            backend.close()
        assert resumed.normative_sha256 == fresh.normative_sha256
        assert resumed.resumed_pack_count == 2
        assert resumed.published_pack_count == 0


def test_one_bad_record_is_precisely_isolated_and_other_pack_survives(tmp_path):
    """一个坏记录只形成一个 anomaly，另一来源仍发布且可恢复。"""
    tasks = (_task(31), _task(32))
    backend = DictBackend()
    try:
        result = run_source_pack_batch(
            tasks,
            tmp_path / "release",
            backend,
            worker_count=2,
            faults={31: "BAD_RECORD"},
        )
    finally:
        backend.close()
    report = result.report.to_value()
    assert report["anomaly_count"] == 1
    assert report["successful_pack_count"] == 1
    assert report["anomalies"][0]["anomaly_code"] == "BAD_RECORD"
    assert not (tmp_path / "release" / "packs"
                / tasks[0].spec.pack_name).exists()
    assert (tmp_path / "release" / "packs"
            / tasks[1].spec.pack_name / "manifest.json").is_file()


def test_fault_and_worker_contracts_fail_closed(tmp_path):
    """未知 worker、多 fault 和不安全 pack 名在入口停止。"""
    assert SOURCE_PACK_FAULT_CODES == (
        "BAD_COMBINATION", "BAD_LICENSE", "BAD_RECORD", "BAD_SOURCE")
    backend = DictBackend()
    try:
        with pytest.raises(SourcePackRuntimeError, match="worker_count"):
            run_source_pack_batch(
                (_task(41),), tmp_path / "worker", backend, worker_count=3)
        with pytest.raises(SourcePackRuntimeError, match="一次"):
            run_source_pack_batch(
                (_task(41), _task(42)),
                tmp_path / "faults",
                backend,
                faults={41: "BAD_SOURCE", 42: "BAD_LICENSE"},
            )
    finally:
        backend.close()
    bad_spec = replace(_spec(43), pack_name="../escape")
    with pytest.raises(SourcePackCompilerError, match="pack_name"):
        compile_or_resume_source_pack(
            bad_spec, (_seed(43, 1, "train"),), tmp_path / "escape")


def _coverage_entry(status: str) -> SourcePackCoverageEntry:
    """构造正式 pack 或 blocker 覆盖记录。"""
    if status == "PACK_FROZEN":
        return SourcePackCoverageEntry(
            "SYNTHETIC_SOURCE",
            "CC-BY-SA-4.0",
            status,
            "data/ph2/manifests/source.raw_snapshot.json",
            "a" * 64,
            "ph2_dataset_artifacts/source/manifest.json",
            "b" * 64,
            8,
            ("train", "held_out"),
            2,
            2,
            "",
            ("tests/test_d02_source_pack_compiler.py",),
        )
    return SourcePackCoverageEntry(
        "SYNTHETIC_SOURCE",
        "UNRESOLVED/BLOCKED",
        status,
        "data/ph2/manifests/source.raw_snapshot.json",
        "a" * 64,
        "",
        "",
        0,
        (),
        0,
        0,
        "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
        ("data/ph2/manifests/source.license_reconciliation.json",),
    )


def test_coverage_manifest_round_trip_nonoverwrite_and_zero_execution(tmp_path):
    """来源覆盖账可恢复、不可覆盖，且不夹带 D-03/训练事实。"""
    manifest = SourcePackCoverageManifest(
        1, "D02-source-pack-coverage-test-v1", (_coverage_entry("PACK_FROZEN"),))
    path = tmp_path / "coverage.json"
    write_source_pack_coverage_manifest(manifest, path)
    restored = read_source_pack_coverage_manifest(path)
    assert restored == manifest
    write_source_pack_coverage_manifest(manifest, path)
    path.write_bytes(b'{"damaged":1}\n')
    with pytest.raises(SourcePackCompilerError, match="内容不同"):
        write_source_pack_coverage_manifest(manifest, path)
    with pytest.raises(SourcePackContractError, match="必须为 0"):
        replace(manifest, d03_published=1)


def test_blocker_cannot_forge_pack_and_extra_coverage_fields_fail_closed(tmp_path):
    """许可 blocker 不得伪造 PASS pack；环境字段不能进入覆盖 artifact。"""
    blocked = _coverage_entry("BLOCKED")
    with pytest.raises(SourcePackContractError, match="不得伪造"):
        replace(
            blocked,
            pack_manifest_relative_path="artifact/manifest.json",
            pack_manifest_sha256="b" * 64,
            pack_record_count=1,
            source_cluster_count=1,
            combination_cluster_count=1,
        )
    value = SourcePackCoverageManifest(
        1, "D02-source-pack-coverage-test-v1", (blocked,)).to_dict()
    value["proxy"] = "http://proxy.invalid:8080"
    path = tmp_path / "bad.json"
    from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(SourcePackCompilerError, match="损坏"):
        read_source_pack_coverage_manifest(path)


def test_repository_source_pack_coverage_has_exact_frozen_identities():
    """正式覆盖账逐来源绑定 snapshot/pack hash、计数和诚实 blocker。"""
    repository = Path(__file__).resolve().parents[1]
    path = repository / SOURCE_PACK_COVERAGE_PATH
    manifest = read_source_pack_coverage_manifest(path)
    assert manifest.sha256() == (
        "0ccd7cb0699c18ed777764075a2c544000fff37f0b57962a83df13acb239ba28")
    assert (
        manifest.d03_published,
        manifest.w01_started,
        manifest.formal_training_runs,
        manifest.teacher_calls,
        manifest.learning_state_writes,
        manifest.mastered_claims,
        manifest.readiness_claims,
    ) == (0, 0, 0, 0, 0, 0, 0)

    actual = {
        (item.source_key, item.license_partition): (
            item.status,
            item.pack_manifest_sha256,
            item.pack_record_count,
            item.source_cluster_count,
            item.combination_cluster_count,
            item.splits,
            item.blocker_code,
        )
        for item in manifest.entries
    }
    expected = {
        ("CC_CEDICT_20260725", "UNRESOLVED/BLOCKED"): (
            "BLOCKED", "", 0, 0, 0, (),
            "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE",
        ),
        ("CONCEPTNET_5_7_0", "CC-BY-4.0"): (
            "PACK_FROZEN",
            "274aacc64db608a00f5d6b808a5ebea31c69af27ad9b0d8161e6a1e5d075c607",
            8, 2, 2, ("held_out",), "",
        ),
        ("CONCEPTNET_5_7_0", "CC-BY-SA-4.0"): (
            "PACK_FROZEN",
            "e4f239d5396413509edcbb62f4778957b4916eb420d61cf8d374b559793cfbd0",
            8, 1, 1, ("held_out",), "",
        ),
        ("UD_ZH_GSDSIMP_R2_18", "CC-BY-SA-4.0"): (
            "PACK_FROZEN",
            "8802732c9a2adf69a10412975e10f1f7d4cf6bad3ded24e249153147cc2d3b79",
            4, 1, 1, ("held_out",), "",
        ),
        ("WIKIDATA_REVISION_V1", "CC0-1.0"): (
            "PACK_FROZEN",
            "d3b0a87930b72513f20554c7aab75085a157d108e4eca8e521f5138c5f9a7b46",
            44, 6, 6, ("train", "dev", "held_out"), "",
        ),
        ("ZHWIKIPEDIA_20260701", "CC-BY-SA-4.0"): (
            "PACK_FROZEN",
            "d74453a85e6732048630a5a5b018d0eb97e3a23d4f3d04acd2fd42f839aa2bbb",
            16, 4, 4, ("train", "held_out"), "",
        ),
        ("ZHWIKTIONARY_20260701", "CC-BY-SA-4.0"): (
            "PACK_FROZEN",
            "9c947242afd3c00a0dbe35f5e62ba28d3446475f2223e6bd8cd82437b85098e6",
            16, 4, 4, ("train", "held_out"), "",
        ),
    }
    assert actual == expected
    frozen = tuple(item for item in manifest.entries
                   if item.status == "PACK_FROZEN")
    assert len(frozen) == 6
    assert sum(item.pack_record_count for item in frozen) == 96
    assert sum(item.source_cluster_count for item in frozen) == 18
    assert sum(item.combination_cluster_count for item in frozen) == 18

    for item in manifest.entries:
        snapshot = repository / Path(
            *item.raw_snapshot_manifest_relative_path.split("/"))
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == (
            item.raw_snapshot_manifest_sha256)
        if item.status == "PACK_FROZEN":
            assert item.pack_manifest_relative_path.startswith(
                "ph2_dataset_artifacts/d02_source_pack_v1/packs/")
            assert item.blocker_code == ""
        else:
            assert item.pack_manifest_relative_path == ""
            assert item.pack_manifest_sha256 == ""
