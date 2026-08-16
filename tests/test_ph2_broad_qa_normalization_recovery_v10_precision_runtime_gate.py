"""覆盖 recovery-v10 precision candidate pack 与 TRAIN runtime gate。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha(label: str) -> str:
    """返回稳定测试 SHA。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _clock(step: int = 100):
    """返回严格单调的整数测试时钟。"""
    value = 0

    def read() -> int:
        nonlocal value
        value += step
        return value

    return read


def _query(input_text: str, source: str) -> dict[str, object]:
    """构造结构自洽的 runtime query。"""
    return {
        "input_text": input_text,
        "official_source_text": source,
        "structure_tokens": list(localization_structure_tokens(input_text)),
    }


def _shape(ordinal: int, query: dict[str, object]) -> dict[str, object]:
    """构造 runtime profile 所需的最小 shape。"""
    return {"ordinal": ordinal, "query": query}


def _result(
        query: dict[str, object], *, committed: bool,
        ) -> dict[str, object]:
    """构造 source-only commit 或干净 UNKNOWN 结果。"""
    return {
        "behavior": "EXACT" if committed else "UNKNOWN",
        "exception_count": 0,
        "input_text": query["input_text"],
        "official_source_text": query["official_source_text"],
        "output_text": "输出" if committed else "",
        "partial_commit_count": 0,
        "production_enabled": 0,
        "route_kind": (
            "SOURCE_CONDITIONED_LEXICAL_ATOM" if committed else "UNKNOWN"),
        "structure_mismatch_count": 0,
    }


def _pass_profile(count: int, committed: int) -> dict[str, object]:
    """形成 strict reader 可核验的小分母 PASS profile。"""
    result_sha = _sha("runtime-result")
    runs = []
    for ordinal, kind in enumerate(("INDEXED", "INDEXED", "REFERENCE"), 1):
        wall_ns = 100
        runs.append({
            "behavior_counts": {
                "EXACT": committed,
                "UNKNOWN": count - committed,
            },
            "committed_output_count": committed,
            "cpu_ns": 80,
            "exception_count": 0,
            "executor_kind": kind,
            "input_bytes": 20,
            "max_ns": 20,
            "non_source_commit_count": 0,
            "ordinal": ordinal,
            "output_bytes": 40,
            "p50_ns": 10,
            "p95_ns": 20,
            "partial_commit_count": 0,
            "production_enabled_count": 0,
            "queries_per_second": count * 1_000_000_000 // wall_ns,
            "query_count": count,
            "result_sha256": result_sha,
            "route_counts": {
                "SOURCE_CONDITIONED_LEXICAL_ATOM": committed,
                "UNKNOWN": count - committed,
            },
            "structure_mismatch_count": 0,
            "unknown_output_count": 0,
            "wall_ns": wall_ns,
        })
    return {
        "aggregate": {
            "committed_output_count": committed * 3,
            "exception_count": 0,
            "indexed_reference_mismatch_count": 0,
            "indexed_repeat_mismatch_count": 0,
            "non_source_commit_count": 0,
            "partial_commit_count": 0,
            "peak_working_set_bytes": 1_024,
            "production_enabled_count": 0,
            "query_count": count,
            "result_sha256": result_sha,
            "structure_mismatch_count": 0,
            "total_cpu_ns": 300,
            "total_wall_ns": 400,
            "unknown_output_count": 0,
        },
        "budget": {
            "execution_order": ["INDEXED", "INDEXED", "REFERENCE"],
            "peak_working_set_bytes_max_exclusive": 10_000,
            "query_count": count,
            "total_wall_ns_max_exclusive": 10_000,
        },
        "gate_outcome": "PASS",
        "runs": runs,
    }


def test_runtime_shapes_exclude_expected_output_and_source_family(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """pack roster 只能携带 query，不得夹带 TRAIN answer 或 family。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack as pack

    monkeypatch.setattr(pack, "V10_PRECISION_RUNTIME_QUERY_COUNT", 2)
    cases = tuple({
        **_query(f"输入{index}", f"Source {index}"),
        "expected_output": f"输出{index}",
        "source_family": f"family-{index}",
    } for index in range(2))
    shapes = pack.derive_normalization_recovery_v10_precision_runtime_shapes(
        cases)
    assert len(shapes) == 2
    assert all(set(item["query"]) == {
        "input_text", "official_source_text", "structure_tokens"}
        for item in shapes)
    assert all(item["expected_output_included"] == 0 for item in shapes)
    assert all(item["source_family_included"] == 0 for item in shapes)
    assert all("expected_output" not in item["query"] for item in shapes)


def test_candidate_pack_round_trip_and_duplicate_rejection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """candidate pack 必须不可覆盖且 strict reader 逐字节重派生。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack as pack

    inputs = []
    for name in ("feasibility", "predecessor", "base", "protocol",
                 "observations", "opencc"):
        path = tmp_path / name
        path.mkdir()
        inputs.append(path)
    candidate = {"candidate_program_sha256": _sha("candidate")}
    preflight = {"failure_count": 0}
    shapes = (_shape(0, _query("输入", "Source")),)
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
        "runtime-shapes.jsonl": canonical_json_line(shapes[0]),
    }
    manifest = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_V1"),
        "files": [{
            "bytes": len(payload),
            "relative_path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        } for name, payload in payloads.items()],
    }
    monkeypatch.setattr(pack, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        pack, "_derive",
        lambda **kwargs: (manifest, candidate, preflight, shapes, payloads))
    target = tmp_path / "candidate-pack"
    published = pack.publish_normalization_recovery_v10_precision_candidate_pack(
        run_root=tmp_path,
        feasibility_dir=inputs[0],
        predecessor_feasibility_dir=inputs[1],
        base_candidate_dir=inputs[2],
        protocol_dir=inputs[3],
        observation_dir=inputs[4],
        opencc_source_pack_dir=inputs[5],
        target_dir=target,
    )
    reread = pack.read_normalization_recovery_v10_precision_candidate_pack(
        target,
        feasibility_dir=inputs[0],
        predecessor_feasibility_dir=inputs[1],
        base_candidate_dir=inputs[2],
        protocol_dir=inputs[3],
        observation_dir=inputs[4],
        opencc_source_pack_dir=inputs[5],
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert reread == (published, candidate, preflight, shapes)
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        pack.publish_normalization_recovery_v10_precision_candidate_pack(
            run_root=tmp_path,
            feasibility_dir=inputs[0],
            predecessor_feasibility_dir=inputs[1],
            base_candidate_dir=inputs[2],
            protocol_dir=inputs[3],
            observation_dir=inputs[4],
            opencc_source_pack_dir=inputs[5],
            target_dir=target,
        )


def test_candidate_pack_runtime_payload_reads_only_self_contained_pack(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime reader 必须核验自包含 pack，且不要求再次提供上游目录。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack as pack

    program_sha = _sha("direct-candidate")
    preflight_sha = _sha("direct-preflight")
    monkeypatch.setattr(pack, "V10_PRECISION_RUNTIME_QUERY_COUNT", 1)
    monkeypatch.setattr(
        pack, "V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT", 0)
    monkeypatch.setattr(
        pack, "V10_PRECISION_CANDIDATE_PROGRAM_SHA256", program_sha)
    monkeypatch.setattr(pack, "V10_PRECISION_PREFLIGHT_SHA256", preflight_sha)
    candidate = {
        "artifact_kind": pack.NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND,
        "candidate_program_sha256": program_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": pack.NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS,
    }
    preflight = {"failure_count": 0, "preflight_sha256": preflight_sha}
    monkeypatch.setattr(
        pack, "derive_normalization_recovery_v10_precision_v2_preflight",
        lambda value: preflight)
    cases = ({
        **_query("输入", "Source"),
        "expected_output": "输出",
        "source_family": "family",
    },)
    shapes = pack.derive_normalization_recovery_v10_precision_runtime_shapes(
        cases)
    payloads = {
        "candidate-program.json": canonical_json_line(candidate),
        "preflight.json": canonical_json_line(preflight),
        "runtime-shapes.jsonl": canonical_json_line(shapes[0]),
    }
    query_payload = canonical_json_line(shapes[0]["query"])
    manifest = {
        "artifact_kind": pack.NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND,
        "candidate_program_sha256": program_sha,
        "expected_source_commit_count": 0,
        "files": [{
            "bytes": len(payload),
            "record_count": 1 if name == "runtime-shapes.jsonl" else None,
            "relative_path": name,
            "role": "TEST",
            "sha256": hashlib.sha256(payload).hexdigest(),
        } for name, payload in payloads.items()],
        "formal_or_evaluation_payload_read_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "query_roster_bytes": len(query_payload),
        "query_roster_sha256": hashlib.sha256(query_payload).hexdigest(),
        "runtime_shape_count": 1,
        "runtime_shapes_sha256": hashlib.sha256(
            payloads["runtime-shapes.jsonl"]).hexdigest(),
        "status": pack.NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS,
        "teacher_api_llm_call_count": 0,
        "v2_feasibility_manifest_sha256": (
            pack.V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
    }
    root = tmp_path / "direct-pack"
    root.mkdir()
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    manifest_payload = canonical_json_line(manifest)
    (root / "manifest.json").write_bytes(manifest_payload)
    reread = (
        pack.read_normalization_recovery_v10_precision_candidate_pack_runtime_payload(
            root,
            expected_manifest_sha256=hashlib.sha256(
                manifest_payload).hexdigest(),
        ))
    assert reread[0]["runtime_shape_count"] == 1
    assert reread[1] == candidate
    assert reread[2] == preflight
    assert reread[3] == shapes


def test_runtime_profile_executes_two_indexed_and_one_reference(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """三次执行必须逐结果相等、source-only commit 且通过资源门。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate as gate

    shapes = (
        _shape(0, _query("输入0", "Source 0")),
        _shape(1, _query("输入1", "Source 1")),
    )
    calls = []

    def profiled(candidate, queries, *, indexed, clock_ns):
        calls.append(indexed)
        return (
            tuple(_result(query, committed=index == 0)
                  for index, query in enumerate(queries)),
            (10, 20),
        )

    monkeypatch.setattr(gate, "V10_PRECISION_RUNTIME_QUERY_COUNT", 2)
    monkeypatch.setattr(
        gate, "V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT", 1)
    monkeypatch.setattr(
        gate, "V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE",
        10_000)
    monkeypatch.setattr(
        gate,
        "V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE",
        10_000)
    monkeypatch.setattr(
        gate,
        "profile_normalization_recovery_v10_precision_candidate_v2_batch",
        profiled)
    profile = gate.derive_normalization_recovery_v10_precision_runtime_profile(
        candidate={}, shapes=shapes,
        wall_clock_ns=_clock(), cpu_clock_ns=_clock(80),
        working_set_bytes=lambda: 2_048)
    assert calls == [True, True, False]
    assert profile["gate_outcome"] == "PASS"
    assert profile["aggregate"]["indexed_reference_mismatch_count"] == 0
    assert profile["aggregate"]["non_source_commit_count"] == 0
    assert profile["aggregate"]["committed_output_count"] == 3


def test_runtime_gate_publication_and_strict_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runtime gate 必须 aggregate-only、不可覆盖且 reader 不重跑。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack as pack
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate as gate
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate_reader as reader

    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    count = 2
    committed = 1
    pack_sha = _sha("candidate-pack")
    shapes = (
        _shape(0, _query("输入0", "Source 0")),
        _shape(1, _query("输入1", "Source 1")),
    )
    query_payload = b"".join(
        canonical_json_line(item["query"]) for item in shapes)
    metadata = {
        "query_roster_bytes": len(query_payload),
        "query_roster_sha256": hashlib.sha256(query_payload).hexdigest(),
        "runtime_shapes_sha256": _sha("runtime-shapes"),
    }
    manifest = {
        "manifest_sha256": pack_sha,
        "rule_counts": {
            "identity_veto_rules": 2,
            "orthographic_whole_input_rules": 1,
            "source_conditioned_rules": 1,
        },
    }
    candidate = {
        "candidate_program_sha256": pack.V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
    }
    material = (manifest, candidate, shapes, metadata)
    for module in (gate, reader):
        monkeypatch.setattr(
            module, "V10_PRECISION_RUNTIME_QUERY_COUNT", count)
        monkeypatch.setattr(
            module, "V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT",
            committed)
        monkeypatch.setattr(
            module, "V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE",
            10_000)
        monkeypatch.setattr(
            module,
            "V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE",
            10_000)
        monkeypatch.setattr(
            module,
            "read_normalization_recovery_v10_precision_runtime_inputs",
            lambda **kwargs: material)
    monkeypatch.setattr(gate, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        gate, "derive_normalization_recovery_v10_precision_runtime_profile",
        lambda **kwargs: _pass_profile(count, committed))
    target = tmp_path / "runtime-gate"
    published = gate.publish_normalization_recovery_v10_precision_runtime_gate(
        run_root=tmp_path,
        candidate_pack_dir=candidate_path,
        expected_candidate_pack_manifest_sha256=pack_sha,
        target_dir=target,
    )
    reread = reader.read_normalization_recovery_v10_precision_runtime_gate(
        target,
        candidate_pack_dir=candidate_path,
        expected_candidate_pack_manifest_sha256=pack_sha,
        expected_runtime_gate_sha256=published["runtime_gate_sha256"],
    )
    assert reread == published
    assert reread["formal_or_evaluation_payload_read_count"] == 0
    assert reread["profile"]["gate_outcome"] == "PASS"
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        gate.publish_normalization_recovery_v10_precision_runtime_gate(
            run_root=tmp_path,
            candidate_pack_dir=candidate_path,
            expected_candidate_pack_manifest_sha256=pack_sha,
            target_dir=target,
        )
