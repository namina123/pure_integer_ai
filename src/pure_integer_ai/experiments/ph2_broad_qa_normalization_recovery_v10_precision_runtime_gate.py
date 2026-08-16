"""发布 recovery-v10 source-only 候选的完整 TRAIN runtime 性能门。"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import time

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate import (
    profile_normalization_recovery_v10_precision_candidate_v2_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack import (
    NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND,
    NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS,
    V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
    V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256,
    V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT,
    V10_PRECISION_RUNTIME_QUERY_COUNT,
    read_normalization_recovery_v10_precision_candidate_pack_runtime_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.train_execution import (
    process_working_set_bytes,
)


NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_V1")
NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_STATUS = (
    "TRAIN_SOURCE_ONLY_RUNTIME_BUDGET_PASS_NOT_FORMAL_NOT_DEPLOYED")

V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT = 3
V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE = 30_000_000_000
V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE = 536_870_912


def _sha256(payload: bytes) -> str:
    """返回输入、结果、源码或 runtime gate 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 precision runtime gate {label} 非法")
    return value


def _percentile(values: tuple[int, ...], percent: int) -> int:
    """按 nearest-rank 形成整数纳秒 p50/p95。"""
    if not values or percent not in {50, 95}:
        raise BroadQaExternalDataError(
            "v10 precision runtime gate percentile 非法")
    ordered = sorted(values)
    index = max(0, (len(ordered) * percent + 99) // 100 - 1)
    return ordered[index]


def _require_k_root(value: str | Path) -> Path:
    """要求显式、已存在的 K 盘 run root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 precision runtime gate run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析路径并拒绝逃出唯一 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(
            f"v10 precision runtime gate {label} 越界")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def read_normalization_recovery_v10_precision_runtime_inputs(
        *, candidate_pack_dir: str | Path,
        expected_candidate_pack_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, object],
            tuple[dict[str, object], ...], dict[str, object],
        ]:
    """严格回读 candidate pack，并返回无 expected output 的 query shape。"""
    expected_sha = _sha_value(
        expected_candidate_pack_manifest_sha256,
        label="candidate pack manifest SHA")
    manifest, candidate, _preflight, shapes = (
        read_normalization_recovery_v10_precision_candidate_pack_runtime_payload(
            candidate_pack_dir,
            expected_manifest_sha256=expected_sha,
        ))
    query_payload = b"".join(
        canonical_json_line(item["query"]) for item in shapes)
    if (manifest.get("manifest_sha256") != expected_sha
            or manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_KIND
            or manifest.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_PACK_STATUS
            or manifest.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or manifest.get("v2_feasibility_manifest_sha256")
            != V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256
            or manifest.get("runtime_shape_count")
            != V10_PRECISION_RUNTIME_QUERY_COUNT
            or manifest.get("expected_source_commit_count")
            != V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
            or manifest.get("query_roster_bytes") != len(query_payload)
            or manifest.get("query_roster_sha256") != _sha256(query_payload)
            or candidate.get("candidate_program_sha256")
            != V10_PRECISION_CANDIDATE_PROGRAM_SHA256
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate candidate pack state 漂移")
    metadata = {
        "query_roster_bytes": len(query_payload),
        "query_roster_sha256": _sha256(query_payload),
        "runtime_shapes_sha256": manifest["runtime_shapes_sha256"],
    }
    return manifest, candidate, shapes, metadata


def _run_executor(
        *, candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        indexed: bool,
        ordinal: int,
        wall_clock_ns,
        cpu_clock_ns,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """执行一次 batch，并形成不含个体输出的整数 aggregate。"""
    wall_started = wall_clock_ns()
    cpu_started = cpu_clock_ns()
    results, durations = (
        profile_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, queries, indexed=indexed, clock_ns=wall_clock_ns))
    cpu_ns = cpu_clock_ns() - cpu_started
    wall_ns = wall_clock_ns() - wall_started
    if (type(cpu_ns) is not int or cpu_ns < 0
            or type(wall_ns) is not int or wall_ns <= 0):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate clock 漂移")
    output_payload = canonical_json_bytes(results)
    route_counts = Counter(str(item.get("route_kind")) for item in results)
    behavior_counts = Counter(str(item.get("behavior")) for item in results)
    metrics = {
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "committed_output_count": sum(
            item.get("behavior") != "UNKNOWN" for item in results),
        "cpu_ns": cpu_ns,
        "exception_count": sum(
            int(item.get("exception_count", 1)) for item in results),
        "executor_kind": "INDEXED" if indexed else "REFERENCE",
        "input_bytes": sum(
            len(canonical_json_bytes(item)) for item in queries),
        "max_ns": max(durations),
        "non_source_commit_count": sum(
            item.get("behavior") != "UNKNOWN"
            and item.get("route_kind") != "SOURCE_CONDITIONED_LEXICAL_ATOM"
            for item in results),
        "ordinal": ordinal,
        "output_bytes": len(output_payload),
        "p50_ns": _percentile(durations, 50),
        "p95_ns": _percentile(durations, 95),
        "partial_commit_count": sum(
            int(item.get("partial_commit_count", 1)) for item in results),
        "production_enabled_count": sum(
            int(item.get("production_enabled", 1) != 0) for item in results),
        "queries_per_second": (
            len(queries) * 1_000_000_000 // wall_ns),
        "query_count": len(queries),
        "result_sha256": _sha256(output_payload),
        "route_counts": dict(sorted(route_counts.items())),
        "structure_mismatch_count": sum(
            int(item.get("structure_mismatch_count", 1))
            for item in results),
        "unknown_output_count": sum(
            item.get("behavior") == "UNKNOWN" and bool(item.get("output_text"))
            for item in results),
        "wall_ns": wall_ns,
    }
    return metrics, results


def derive_normalization_recovery_v10_precision_runtime_profile(
        *, candidate: dict[str, object],
        shapes: tuple[dict[str, object], ...],
        wall_clock_ns=time.perf_counter_ns,
        cpu_clock_ns=time.process_time_ns,
        working_set_bytes=process_working_set_bytes,
        ) -> dict[str, object]:
    """执行两次 indexed 与一次 reference，形成固定性能与安全门。"""
    if (not isinstance(shapes, tuple)
            or len(shapes) != V10_PRECISION_RUNTIME_QUERY_COUNT
            or not callable(wall_clock_ns)
            or not callable(cpu_clock_ns)
            or not callable(working_set_bytes)):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate profile input 漂移")
    queries = tuple(item["query"] for item in shapes)
    total_wall_started = wall_clock_ns()
    total_cpu_started = cpu_clock_ns()
    peak_working_set = working_set_bytes()

    first_metrics, first = _run_executor(
        candidate=candidate, queries=queries, indexed=True, ordinal=1,
        wall_clock_ns=wall_clock_ns, cpu_clock_ns=cpu_clock_ns)
    peak_working_set = max(peak_working_set, working_set_bytes())
    second_metrics, second = _run_executor(
        candidate=candidate, queries=queries, indexed=True, ordinal=2,
        wall_clock_ns=wall_clock_ns, cpu_clock_ns=cpu_clock_ns)
    peak_working_set = max(peak_working_set, working_set_bytes())
    indexed_repeat_mismatch = sum(
        left != right for left, right in zip(first, second))
    second = ()
    reference_metrics, reference = _run_executor(
        candidate=candidate, queries=queries, indexed=False, ordinal=3,
        wall_clock_ns=wall_clock_ns, cpu_clock_ns=cpu_clock_ns)
    peak_working_set = max(peak_working_set, working_set_bytes())
    indexed_reference_mismatch = sum(
        left != right for left, right in zip(first, reference))

    runs = [first_metrics, second_metrics, reference_metrics]
    total_cpu_ns = cpu_clock_ns() - total_cpu_started
    total_wall_ns = wall_clock_ns() - total_wall_started
    aggregate = {
        "committed_output_count": sum(
            int(item["committed_output_count"]) for item in runs),
        "exception_count": sum(int(item["exception_count"]) for item in runs),
        "indexed_reference_mismatch_count": indexed_reference_mismatch,
        "indexed_repeat_mismatch_count": indexed_repeat_mismatch,
        "non_source_commit_count": sum(
            int(item["non_source_commit_count"]) for item in runs),
        "partial_commit_count": sum(
            int(item["partial_commit_count"]) for item in runs),
        "peak_working_set_bytes": peak_working_set,
        "production_enabled_count": sum(
            int(item["production_enabled_count"]) for item in runs),
        "query_count": len(queries),
        "result_sha256": first_metrics["result_sha256"],
        "structure_mismatch_count": sum(
            int(item["structure_mismatch_count"]) for item in runs),
        "total_cpu_ns": total_cpu_ns,
        "total_wall_ns": total_wall_ns,
        "unknown_output_count": sum(
            int(item["unknown_output_count"]) for item in runs),
    }
    zero_fields = (
        "exception_count", "indexed_reference_mismatch_count",
        "indexed_repeat_mismatch_count", "non_source_commit_count",
        "partial_commit_count", "production_enabled_count",
        "structure_mismatch_count", "unknown_output_count",
    )
    passed = (
        len(runs) == V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT
        and total_wall_ns
        < V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE
        and peak_working_set
        < V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE
        and all(aggregate[name] == 0 for name in zero_fields)
        and aggregate["committed_output_count"]
        == (V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
            * V10_PRECISION_RUNTIME_GATE_EXECUTION_COUNT)
        and len({item["result_sha256"] for item in runs}) == 1
        and all(item["query_count"] == V10_PRECISION_RUNTIME_QUERY_COUNT
                for item in runs)
        and all(item["committed_output_count"]
                == V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT
                for item in runs)
        and all(item["behavior_counts"] == {
            "EXACT": V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT,
            "UNKNOWN": (V10_PRECISION_RUNTIME_QUERY_COUNT
                        - V10_PRECISION_RUNTIME_EXPECTED_SOURCE_COMMIT_COUNT),
        } for item in runs)
    )
    return {
        "aggregate": aggregate,
        "budget": {
            "execution_order": ["INDEXED", "INDEXED", "REFERENCE"],
            "peak_working_set_bytes_max_exclusive": (
                V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE),
            "query_count": V10_PRECISION_RUNTIME_QUERY_COUNT,
            "total_wall_ns_max_exclusive": (
                V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE),
        },
        "gate_outcome": "PASS" if passed else "FAIL",
        "runs": runs,
    }


def normalization_recovery_v10_precision_runtime_gate_code_files(
        ) -> list[dict[str, object]]:
    """承诺 candidate、pack、runtime 与 strict reader 的公开源码字节。"""
    directory = Path(__file__).resolve().parent
    names = (
        "ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_candidate.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_feasibility.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_candidate_pack.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate.py",
        "ph2_broad_qa_normalization_recovery_v10_precision_runtime_gate_reader.py",
    )
    return [{
        "bytes": len(payload),
        "relative_path": f"src/pure_integer_ai/experiments/{name}",
        "sha256": _sha256(payload),
    } for name in names for payload in [(directory / name).read_bytes()]]


def publish_normalization_recovery_v10_precision_runtime_gate(
        *, run_root: str | Path,
        candidate_pack_dir: str | Path,
        expected_candidate_pack_manifest_sha256: str,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 33,179 分母的 TRAIN-only runtime gate。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=label) for value, label in (
        (candidate_pack_dir, "candidate_pack_dir"),
        (target_dir, "target_dir"),
    ))
    if (any(not path.is_dir() for path in paths[:-1])
            or paths[-1].exists()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError(
            "v10 precision runtime gate path 非法")
    expected_sha = _sha_value(
        expected_candidate_pack_manifest_sha256,
        label="candidate pack manifest SHA")
    manifest, candidate, shapes, metadata = (
        read_normalization_recovery_v10_precision_runtime_inputs(
            candidate_pack_dir=paths[0],
            expected_candidate_pack_manifest_sha256=expected_sha,
        ))
    profile = derive_normalization_recovery_v10_precision_runtime_profile(
        candidate=candidate, shapes=shapes)
    if profile["gate_outcome"] != "PASS":
        raise BroadQaExternalDataError(
            "v10 precision runtime gate 未通过")
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_KIND,
        "candidate_pack_manifest_sha256": expected_sha,
        "candidate_program_read_count": 1,
        "candidate_program_sha256": V10_PRECISION_CANDIDATE_PROGRAM_SHA256,
        "candidate_rule_counts": manifest["rule_counts"],
        "code_files": (
            normalization_recovery_v10_precision_runtime_gate_code_files()),
        "formal_guard_write_count": 0,
        "formal_or_evaluation_payload_read_count": 0,
        "format_version": 1,
        "individual_candidate_output_publication_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "profile": profile,
        "query_roster_bytes": metadata["query_roster_bytes"],
        "query_roster_sha256": metadata["query_roster_sha256"],
        "runtime_shape_read_count": len(shapes),
        "runtime_shapes_sha256": metadata["runtime_shapes_sha256"],
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_STATUS,
        "teacher_api_llm_call_count": 0,
        "v2_feasibility_manifest_sha256": (
            V10_PRECISION_FEASIBILITY_V2_MANIFEST_SHA256),
    }
    target = paths[1]
    target.mkdir()
    path = target / "runtime-gate.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(report))
    return {**report, "runtime_gate_sha256": _sha256(path.read_bytes())}


__all__ = [
    "NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_RUNTIME_GATE_STATUS",
    "V10_PRECISION_RUNTIME_GATE_PEAK_WORKING_SET_BYTES_MAX_EXCLUSIVE",
    "V10_PRECISION_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE",
    "derive_normalization_recovery_v10_precision_runtime_profile",
    "normalization_recovery_v10_precision_runtime_gate_code_files",
    "publish_normalization_recovery_v10_precision_runtime_gate",
    "read_normalization_recovery_v10_precision_runtime_inputs",
]
