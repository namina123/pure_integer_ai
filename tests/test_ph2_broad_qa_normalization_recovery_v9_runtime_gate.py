"""覆盖 recovery-v9 标签盲runtime性能门。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_record_id,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _clock(step: int = 100):
    """返回严格单调的整数测试时钟。"""
    value = 0

    def read() -> int:
        nonlocal value
        value += step
        return value

    return read


def _result(query: dict[str, object]) -> dict[str, object]:
    """构造无提交、无异常、无结构破坏的测试结果。"""
    return {
        "behavior": "UNKNOWN",
        "exception_count": 0,
        "input_text": query["input_text"],
        "official_source_text": query["official_source_text"],
        "output_text": "",
        "partial_commit_count": 0,
        "production_enabled": 0,
        "route_kind": "UNKNOWN",
        "structure_mismatch_count": 0,
    }


def _pass_profile(count: int) -> dict[str, object]:
    """形成strict reader可核验的小分母PASS profile。"""
    result_sha = "a" * 64
    runs = []
    for ordinal, kind in enumerate(("INDEXED", "INDEXED", "REFERENCE"), 1):
        wall_ns = 100
        runs.append({
            "cpu_ns": 80,
            "exception_count": 0,
            "executor_kind": kind,
            "input_bytes": 20,
            "max_ns": 20,
            "ordinal": ordinal,
            "output_bytes": 40,
            "p50_ns": 10,
            "p95_ns": 20,
            "partial_commit_count": 0,
            "production_enabled_count": 0,
            "query_count": count,
            "queries_per_second": count * 1_000_000_000 // wall_ns,
            "result_sha256": result_sha,
            "route_counts": {"UNKNOWN": count},
            "structure_mismatch_count": 0,
            "wall_ns": wall_ns,
        })
    return {
        "aggregate": {
            "exception_count": 0,
            "indexed_reference_mismatch_count": 0,
            "indexed_repeat_mismatch_count": 0,
            "partial_commit_count": 0,
            "peak_working_set_bytes": 1_024,
            "production_enabled_count": 0,
            "query_count": count,
            "result_sha256": result_sha,
            "structure_mismatch_count": 0,
            "total_cpu_ns": 300,
            "total_wall_ns": 400,
        },
        "budget": {
            "execution_order": ["INDEXED", "INDEXED", "REFERENCE"],
            "query_count": count,
            "total_wall_ns_max_exclusive": 10_000,
        },
        "gate_outcome": "PASS",
        "runs": runs,
    }


def test_runtime_profile_executes_two_indexed_and_one_reference(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """三次batch必须同结果且全部失败计数为零。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate as module

    shapes = tuple({
        "query": {
            "input_text": f"測{index}",
            "official_source_text": f"S{index}",
            "structure_tokens": [],
        },
    } for index in range(2))
    calls = []

    def profiled(candidate, queries, *, indexed, clock_ns):
        calls.append(indexed)
        return tuple(_result(item) for item in queries), (10, 20)

    monkeypatch.setattr(module, "V9_RUNTIME_GATE_QUERY_COUNT", 2)
    monkeypatch.setattr(
        module, "V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE", 10_000)
    monkeypatch.setattr(
        module, "profile_normalization_recovery_v8_candidate_batch", profiled)
    profile = module.derive_normalization_recovery_v9_runtime_profile(
        candidate={}, shapes=shapes,
        wall_clock_ns=_clock(), cpu_clock_ns=_clock(80),
        working_set_bytes=lambda: 2_048)
    assert calls == [True, True, False]
    assert profile["gate_outcome"] == "PASS"
    assert profile["aggregate"]["total_wall_ns"] < 10_000
    assert profile["aggregate"]["indexed_reference_mismatch_count"] == 0
    assert profile["aggregate"]["partial_commit_count"] == 0
    assert profile["aggregate"]["structure_mismatch_count"] == 0


def _write_inputs(tmp_path: Path, count: int):
    """建立不含真实translation的最小source/candidate pack。"""
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    for name in (
            "gimp-78fc57122afa94d3-zh-raw-v1.zip",
            "pair-identities.jsonl", "source-census.jsonl",
            "source-files.jsonl"):
        (source / name).write_bytes(b"")
    shapes = []
    for ordinal in range(count):
        query = {
            "input_text": f"測{ordinal}",
            "official_source_text": f"S{ordinal}",
            "structure_tokens": [],
        }
        identity = {
            "ordinal": ordinal,
            "query": query,
            "record_kind": "V9_GIMP_LABEL_BLIND_RUNTIME_SHAPE_V1",
        }
        shapes.append({
            **identity,
            "format_version": 1,
            "input_scalar_count": len(query["input_text"]),
            "official_source_scalar_count": len(query["official_source_text"]),
            "shape_id": localization_record_id(identity),
            "structure_category_sequence": [],
            "structure_token_count": 0,
            "synthetic_surface_only": 1,
        })
    shape_payload = b"".join(canonical_json_line(item) for item in shapes)
    (source / "runtime-shapes.jsonl").write_bytes(shape_payload)
    shape_sha = hashlib.sha256(shape_payload).hexdigest()
    source_manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"),
        "files": [{
            "bytes": len(shape_payload),
            "record_count": count,
            "relative_path": "runtime-shapes.jsonl",
            "sha256": shape_sha,
        }],
        "label_or_translation_surface_published": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": "GIMP_RAW_AND_LABEL_FREE_IDENTITY_FROZEN_NOT_FORMAL",
        "summary": {
            "runtime_shape_count": count,
            "synthetic_runtime_surface_only": 1,
        },
    }
    source_manifest_payload = canonical_json_line(source_manifest)
    (source / "manifest.json").write_bytes(source_manifest_payload)

    program_sha = "b" * 64
    program = {
        "candidate_program_sha256": program_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
    }
    program_payload = canonical_json_line(program)
    (candidate / "candidate-program.json").write_bytes(program_payload)
    (candidate / "preflight.json").write_bytes(b"{}\n")
    candidate_manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_V1"),
        "candidate_program_sha256": program_sha,
        "files": [{
            "bytes": len(program_payload),
            "relative_path": "candidate-program.json",
            "sha256": hashlib.sha256(program_payload).hexdigest(),
        }],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": "LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN",
        "vlc_final_read_count": 0,
    }
    candidate_manifest_payload = canonical_json_line(candidate_manifest)
    (candidate / "manifest.json").write_bytes(candidate_manifest_payload)
    return {
        "candidate": candidate,
        "candidate_manifest_sha": hashlib.sha256(
            candidate_manifest_payload).hexdigest(),
        "candidate_program_file_sha": hashlib.sha256(program_payload).hexdigest(),
        "candidate_program_sha": program_sha,
        "shape_sha": shape_sha,
        "source": source,
        "source_manifest_sha": hashlib.sha256(
            source_manifest_payload).hexdigest(),
    }


def test_runtime_gate_publication_and_strict_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime gate必须不可覆盖、aggregate-only且strict reader不重跑。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate as module
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_runtime_gate_reader as reader

    count = 2
    material = _write_inputs(tmp_path, count)
    values = {
        "V9_RUNTIME_GATE_QUERY_COUNT": count,
        "V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE": 10_000,
        "V9_SOURCE_PACK_MANIFEST_SHA256": material["source_manifest_sha"],
        "V9_RUNTIME_SHAPES_SHA256": material["shape_sha"],
        "V8_CANDIDATE_PACK_MANIFEST_SHA256": material["candidate_manifest_sha"],
        "V8_CANDIDATE_PROGRAM_FILE_SHA256": material[
            "candidate_program_file_sha"],
        "V8_CANDIDATE_PROGRAM_SHA256": material["candidate_program_sha"],
    }
    for name, value in values.items():
        monkeypatch.setattr(module, name, value)
        if hasattr(reader, name):
            monkeypatch.setattr(reader, name, value)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "derive_normalization_recovery_v9_runtime_profile",
        lambda **kwargs: _pass_profile(count))
    target = tmp_path / "runtime-gate"
    published = module.publish_normalization_recovery_v9_runtime_gate(
        run_root=tmp_path,
        source_pack_dir=material["source"],
        candidate_pack_dir=material["candidate"],
        target_dir=target,
    )
    reread = reader.read_normalization_recovery_v9_runtime_gate(
        target,
        source_pack_dir=material["source"],
        candidate_pack_dir=material["candidate"],
        expected_runtime_gate_sha256=published["runtime_gate_sha256"],
    )
    assert reread == published
    assert reread["formal_label_read_count"] == 0
    assert reread["source_raw_archive_read_count"] == 0
    assert reread["profile"]["gate_outcome"] == "PASS"
    with pytest.raises(BroadQaExternalDataError):
        module.publish_normalization_recovery_v9_runtime_gate(
            run_root=tmp_path,
            source_pack_dir=material["source"],
            candidate_pack_dir=material["candidate"],
            target_dir=target,
        )
