"""发布 recovery-v9 GIMP 标签盲 candidate runtime 性能门。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_record_id,
    localization_structure_token_category,
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    V8_CANDIDATE_RULE_COUNTS,
    profile_normalization_recovery_v8_candidate_batch,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from pure_integer_ai.experiments.train_execution import (
    process_working_set_bytes,
)


NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_V1")
NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS = (
    "LABEL_BLIND_RUNTIME_BUDGET_PASS_NOT_FORMAL_NOT_DEPLOYED")

V9_SOURCE_PACK_MANIFEST_SHA256 = (
    "eb3be7e7a4b7482f20416549fce4d9a8cdf8f82db1fce385b290793aee6a0e09")
V9_RUNTIME_SHAPES_SHA256 = (
    "806085e6e00458c18d7dfc5233b34b498434d9dcdd2782b77f5bfe8c4423458a")
V8_CANDIDATE_PACK_MANIFEST_SHA256 = (
    "f3c1a011a05afbb3307e7a9c308077a5c990e093800df7fc1a292a221cfc02f6")
V8_CANDIDATE_PROGRAM_FILE_SHA256 = (
    "e4018f674935e486eb6e1eff4e3bb705057faf5c0edcf135a250b6815c674643")
V8_CANDIDATE_PROGRAM_SHA256 = (
    "d83971fc5a1bf511295172cafd2ad64c8f2b107fb5f652793825cd076ef4677f")
V8_BATCH_FIX_GIT_COMMIT = "23123025c3acec15bef91d9ecccd07cffa68728d"

V9_RUNTIME_GATE_QUERY_COUNT = 9_264
V9_RUNTIME_GATE_EXECUTION_COUNT = 3
V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE = 10_000_000_000

_SOURCE_PACK_NAMES = {
    "gimp-78fc57122afa94d3-zh-raw-v1.zip",
    "manifest.json",
    "pair-identities.jsonl",
    "runtime-shapes.jsonl",
    "source-census.jsonl",
    "source-files.jsonl",
}
_CANDIDATE_PACK_NAMES = {
    "candidate-program.json", "manifest.json", "preflight.json"}
_SHAPE_FIELDS = {
    "format_version", "input_scalar_count", "official_source_scalar_count",
    "ordinal", "query", "record_kind", "shape_id",
    "structure_category_sequence", "structure_token_count",
    "synthetic_surface_only",
}


def _sha256(payload: bytes) -> str:
    """返回输入、结果、源码或runtime gate的SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: tuple[int, ...], percent: int) -> int:
    """按nearest-rank形成整数纳秒p50/p95。"""
    if not values or percent not in {50, 95}:
        raise BroadQaExternalDataError("v9 runtime gate percentile 非法")
    ordered = sorted(values)
    index = max(0, (len(ordered) * percent + 99) // 100 - 1)
    return ordered[index]


def _require_k_root(value: str | Path) -> Path:
    """要求显式runtime gate根位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v9 runtime gate run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析输入输出并限制其仍位于本次K盘run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"v9 runtime gate {label} 越出run root")
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个artifact根是否相同或互为祖先。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def _canonical_json(path: Path, *, expected_sha256: str,
                    label: str) -> dict[str, object]:
    """读取一份规范单行JSON并核对固定SHA。"""
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v9 runtime gate {label} 不可读") from error
    if (_sha256(encoded) != expected_sha256 or not isinstance(value, dict)
            or canonical_json_line(value) != encoded):
        raise BroadQaExternalDataError(
            f"v9 runtime gate {label} identity漂移")
    return value


def _file_record(
        manifest: dict[str, object], *, relative_path: str,
        ) -> dict[str, object]:
    """从manifest中取得唯一文件承诺。"""
    files = manifest.get("files")
    matches = [item for item in files if isinstance(item, dict)
               and item.get("relative_path") == relative_path] if isinstance(
                   files, list) else []
    if len(matches) != 1:
        raise BroadQaExternalDataError("v9 runtime gate file commitment 漂移")
    return matches[0]


def _runtime_shapes(
        source_dir: Path, manifest: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...], bytes]:
    """只读取synthetic runtime shapes，不读取raw、identity或translation。"""
    path = source_dir / "runtime-shapes.jsonl"
    try:
        payload = path.read_bytes()
        shapes = tuple(json.loads(line) for line in payload.splitlines())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v9 runtime gate shapes 不可读") from error
    commitment = _file_record(manifest, relative_path=path.name)
    if (_sha256(payload) != V9_RUNTIME_SHAPES_SHA256
            or commitment.get("sha256") != V9_RUNTIME_SHAPES_SHA256
            or commitment.get("bytes") != len(payload)
            or commitment.get("record_count") != V9_RUNTIME_GATE_QUERY_COUNT
            or len(shapes) != V9_RUNTIME_GATE_QUERY_COUNT
            or b"".join(canonical_json_line(item) for item in shapes) != payload):
        raise BroadQaExternalDataError("v9 runtime gate shapes identity漂移")
    for ordinal, item in enumerate(shapes):
        query = item.get("query") if isinstance(item, dict) else None
        if (set(item) != _SHAPE_FIELDS
                or item.get("format_version") != 1
                or item.get("record_kind")
                != "V9_GIMP_LABEL_BLIND_RUNTIME_SHAPE_V1"
                or item.get("ordinal") != ordinal
                or item.get("synthetic_surface_only") != 1
                or not isinstance(query, dict)
                or set(query) != {
                    "input_text", "official_source_text", "structure_tokens"}
                or not isinstance(query.get("input_text"), str)
                or not query["input_text"]
                or not isinstance(query.get("official_source_text"), str)
                or not query["official_source_text"]
                or not isinstance(query.get("structure_tokens"), list)
                or tuple(query["structure_tokens"])
                != localization_structure_tokens(str(query["input_text"]))
                or item.get("input_scalar_count") != len(query["input_text"])
                or item.get("official_source_scalar_count")
                != len(query["official_source_text"])
                or item.get("structure_token_count")
                != len(query["structure_tokens"])
                or item.get("structure_category_sequence") != [
                    localization_structure_token_category(str(token))
                    for token in query["structure_tokens"]]
                or item.get("shape_id") != localization_record_id({
                    "ordinal": ordinal,
                    "query": query,
                    "record_kind": item["record_kind"],
                })):
            raise BroadQaExternalDataError("v9 runtime gate shape schema漂移")
    return shapes, payload


def read_normalization_recovery_v9_runtime_probe_inputs(
        *, source_pack_dir: str | Path,
        candidate_pack_dir: str | Path,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   dict[str, object], dict[str, object]]:
    """读取固定synthetic分母与旧候选，不触及GIMP/VLC个体标签。"""
    source_dir = Path(source_pack_dir).resolve()
    candidate_dir = Path(candidate_pack_dir).resolve()
    if (_overlap(source_dir, candidate_dir)
            or not source_dir.is_dir() or not candidate_dir.is_dir()
            or {item.name for item in source_dir.iterdir()}
            != _SOURCE_PACK_NAMES
            or {item.name for item in candidate_dir.iterdir()}
            != _CANDIDATE_PACK_NAMES):
        raise BroadQaExternalDataError("v9 runtime gate input root漂移")
    source_manifest = _canonical_json(
        source_dir / "manifest.json",
        expected_sha256=V9_SOURCE_PACK_MANIFEST_SHA256,
        label="source manifest",
    )
    source_summary = source_manifest.get("summary")
    if (source_manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"
            or source_manifest.get("status")
            != "GIMP_RAW_AND_LABEL_FREE_IDENTITY_FROZEN_NOT_FORMAL"
            or source_manifest.get("production_enabled") != 0
            or source_manifest.get("mastery_claimed") != 0
            or source_manifest.get("label_or_translation_surface_published")
            != 0
            or not isinstance(source_summary, dict)
            or source_summary.get("runtime_shape_count")
            != V9_RUNTIME_GATE_QUERY_COUNT
            or source_summary.get("synthetic_runtime_surface_only") != 1):
        raise BroadQaExternalDataError("v9 runtime gate source state漂移")
    shapes, shape_payload = _runtime_shapes(source_dir, source_manifest)

    candidate_manifest = _canonical_json(
        candidate_dir / "manifest.json",
        expected_sha256=V8_CANDIDATE_PACK_MANIFEST_SHA256,
        label="candidate manifest",
    )
    candidate_file = _file_record(
        candidate_manifest, relative_path="candidate-program.json")
    candidate = _canonical_json(
        candidate_dir / "candidate-program.json",
        expected_sha256=V8_CANDIDATE_PROGRAM_FILE_SHA256,
        label="candidate program",
    )
    if (candidate_manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_CANDIDATE_PACK_V1"
            or candidate_manifest.get("status")
            != "LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN"
            or candidate_manifest.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or candidate_manifest.get("production_enabled") != 0
            or candidate_manifest.get("mastery_claimed") != 0
            or candidate_manifest.get("vlc_final_read_count") != 0
            or candidate_file.get("sha256")
            != V8_CANDIDATE_PROGRAM_FILE_SHA256
            or candidate.get("candidate_program_sha256")
            != V8_CANDIDATE_PROGRAM_SHA256
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v9 runtime gate candidate state漂移")
    metadata = {
        "query_roster_bytes": len(b"".join(
            canonical_json_line(item["query"]) for item in shapes)),
        "query_roster_sha256": _sha256(b"".join(
            canonical_json_line(item["query"]) for item in shapes)),
        "runtime_shape_file_bytes": len(shape_payload),
        "runtime_shape_file_sha256": _sha256(shape_payload),
    }
    return source_manifest, shapes, candidate_manifest, {
        "candidate": candidate,
        "metadata": metadata,
    }


def _run_executor(
        *, candidate: dict[str, object],
        queries: tuple[dict[str, object], ...], indexed: bool,
        ordinal: int, wall_clock_ns, cpu_clock_ns,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """执行一次profiled batch并形成不含个体输出的整数aggregate。"""
    wall_started = wall_clock_ns()
    cpu_started = cpu_clock_ns()
    results, durations = profile_normalization_recovery_v8_candidate_batch(
        candidate, queries, indexed=indexed, clock_ns=wall_clock_ns)
    cpu_ns = cpu_clock_ns() - cpu_started
    wall_ns = wall_clock_ns() - wall_started
    if (type(cpu_ns) is not int or cpu_ns < 0
            or type(wall_ns) is not int or wall_ns <= 0):
        raise BroadQaExternalDataError("v9 runtime gate clock漂移")
    output_payload = canonical_json_bytes(results)
    route_counts = Counter(str(item.get("route_kind")) for item in results)
    metrics = {
        "cpu_ns": cpu_ns,
        "exception_count": sum(int(item.get("exception_count", 1))
                               for item in results),
        "executor_kind": "INDEXED" if indexed else "REFERENCE",
        "input_bytes": sum(len(canonical_json_bytes(item)) for item in queries),
        "max_ns": max(durations),
        "ordinal": ordinal,
        "output_bytes": len(output_payload),
        "p50_ns": _percentile(durations, 50),
        "p95_ns": _percentile(durations, 95),
        "partial_commit_count": sum(
            int(item.get("partial_commit_count", 1)) for item in results),
        "production_enabled_count": sum(
            int(item.get("production_enabled", 1) != 0) for item in results),
        "query_count": len(queries),
        "queries_per_second": len(queries) * 1_000_000_000 // wall_ns,
        "result_sha256": _sha256(output_payload),
        "route_counts": dict(sorted(route_counts.items())),
        "structure_mismatch_count": sum(
            int(item.get("structure_mismatch_count", 1)) for item in results),
        "wall_ns": wall_ns,
    }
    return metrics, results


def derive_normalization_recovery_v9_runtime_profile(
        *, candidate: dict[str, object],
        shapes: tuple[dict[str, object], ...],
        wall_clock_ns=time.perf_counter_ns,
        cpu_clock_ns=time.process_time_ns,
        working_set_bytes=process_working_set_bytes,
        ) -> dict[str, object]:
    """执行两次indexed和一次reference，形成正式前标签盲性能门。"""
    if len(shapes) != V9_RUNTIME_GATE_QUERY_COUNT:
        raise BroadQaExternalDataError("v9 runtime gate denominator漂移")
    queries = tuple(item["query"] for item in shapes)
    total_wall_started = wall_clock_ns()
    total_cpu_started = cpu_clock_ns()
    runs = []
    results = []
    peak_working_set = working_set_bytes()
    for ordinal, indexed in ((1, True), (2, True), (3, False)):
        metrics, result = _run_executor(
            candidate=candidate, queries=queries, indexed=indexed,
            ordinal=ordinal, wall_clock_ns=wall_clock_ns,
            cpu_clock_ns=cpu_clock_ns)
        runs.append(metrics)
        results.append(result)
        peak_working_set = max(peak_working_set, working_set_bytes())
    total_cpu_ns = cpu_clock_ns() - total_cpu_started
    total_wall_ns = wall_clock_ns() - total_wall_started
    first, second, reference = results
    aggregate = {
        "exception_count": sum(int(item["exception_count"]) for item in runs),
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(first, reference)),
        "indexed_repeat_mismatch_count": sum(
            left != right for left, right in zip(first, second)),
        "partial_commit_count": sum(
            int(item["partial_commit_count"]) for item in runs),
        "peak_working_set_bytes": peak_working_set,
        "production_enabled_count": sum(
            int(item["production_enabled_count"]) for item in runs),
        "query_count": len(queries),
        "result_sha256": runs[0]["result_sha256"],
        "structure_mismatch_count": sum(
            int(item["structure_mismatch_count"]) for item in runs),
        "total_cpu_ns": total_cpu_ns,
        "total_wall_ns": total_wall_ns,
    }
    passed = (
        len(runs) == V9_RUNTIME_GATE_EXECUTION_COUNT
        and total_wall_ns < V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE
        and aggregate["exception_count"] == 0
        and aggregate["indexed_reference_mismatch_count"] == 0
        and aggregate["indexed_repeat_mismatch_count"] == 0
        and aggregate["partial_commit_count"] == 0
        and aggregate["production_enabled_count"] == 0
        and aggregate["structure_mismatch_count"] == 0
        and len({item["result_sha256"] for item in runs}) == 1
        and all(item["query_count"] == V9_RUNTIME_GATE_QUERY_COUNT
                for item in runs)
    )
    return {
        "aggregate": aggregate,
        "budget": {
            "execution_order": ["INDEXED", "INDEXED", "REFERENCE"],
            "query_count": V9_RUNTIME_GATE_QUERY_COUNT,
            "total_wall_ns_max_exclusive": (
                V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE),
        },
        "gate_outcome": "PASS" if passed else "FAIL",
        "runs": runs,
    }


def normalization_recovery_v9_runtime_gate_code_files(
        ) -> list[dict[str, object]]:
    """承诺runtime、shape合同与strict reader的公开源码字节。"""
    directory = Path(__file__).resolve().parent
    names = (
        "ph2_broad_qa_normalization_recovery_v5_localization_structure.py",
        "ph2_broad_qa_normalization_recovery_v8_candidate.py",
        "ph2_broad_qa_normalization_recovery_v9_runtime_gate.py",
        "ph2_broad_qa_normalization_recovery_v9_runtime_gate_reader.py",
        "ph2_broad_qa_normalization_recovery_v9_source_pack.py",
    )
    return [{
        "bytes": len(payload),
        "relative_path": f"src/pure_integer_ai/experiments/{name}",
        "sha256": _sha256(payload),
    } for name in names for payload in [(directory / name).read_bytes()]]


def publish_normalization_recovery_v9_runtime_gate(
        *, run_root: str | Path,
        source_pack_dir: str | Path,
        candidate_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布9,264分母的正式前aggregate-only性能门。"""
    root = _require_k_root(run_root)
    source_dir = _within(root, source_pack_dir, label="source pack")
    candidate_dir = _within(root, candidate_pack_dir, label="candidate pack")
    target = _within(root, target_dir, label="target")
    paths = (source_dir, candidate_dir, target)
    if (target.exists() or not source_dir.is_dir()
            or not candidate_dir.is_dir()
            or any(_overlap(left, right)
                   for index, left in enumerate(paths)
                   for right in paths[index + 1:])):
        raise BroadQaExternalDataError("v9 runtime gate path非法")
    source_manifest, shapes, candidate_manifest, material = (
        read_normalization_recovery_v9_runtime_probe_inputs(
            source_pack_dir=source_dir, candidate_pack_dir=candidate_dir))
    profile = derive_normalization_recovery_v9_runtime_profile(
        candidate=material["candidate"], shapes=shapes)
    if profile["gate_outcome"] != "PASS":
        raise BroadQaExternalDataError("v9 runtime gate 未通过")
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND,
        "batch_fix_git_commit": V8_BATCH_FIX_GIT_COMMIT,
        "candidate_pack_manifest_sha256": V8_CANDIDATE_PACK_MANIFEST_SHA256,
        "candidate_program_read_count": 1,
        "candidate_program_sha256": V8_CANDIDATE_PROGRAM_SHA256,
        "candidate_rule_counts": V8_CANDIDATE_RULE_COUNTS,
        "code_files": normalization_recovery_v9_runtime_gate_code_files(),
        "formal_guard_write_count": 0,
        "formal_label_read_count": 0,
        "format_version": 1,
        "individual_candidate_output_publication_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "profile": profile,
        "query_roster_bytes": material["metadata"]["query_roster_bytes"],
        "query_roster_sha256": material["metadata"]["query_roster_sha256"],
        "source_pair_identity_read_count": 0,
        "source_pack_manifest_sha256": V9_SOURCE_PACK_MANIFEST_SHA256,
        "source_raw_archive_read_count": 0,
        "source_runtime_shape_read_count": len(shapes),
        "source_runtime_shapes_sha256": V9_RUNTIME_SHAPES_SHA256,
        "source_translation_surface_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    if (source_manifest.get("manifest_sha256") is not None
            or candidate_manifest.get("manifest_sha256") is not None):
        raise BroadQaExternalDataError("v9 runtime gate manifest内存形态漂移")
    target.mkdir()
    path = target / "runtime-gate.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(report))
    return {**report, "runtime_gate_sha256": _sha256(path.read_bytes())}


__all__ = [
    "NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_KIND",
    "NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_STATUS",
    "V8_BATCH_FIX_GIT_COMMIT",
    "V8_CANDIDATE_PACK_MANIFEST_SHA256",
    "V8_CANDIDATE_PROGRAM_SHA256",
    "V9_RUNTIME_GATE_QUERY_COUNT",
    "V9_RUNTIME_GATE_TOTAL_WALL_NS_MAX_EXCLUSIVE",
    "V9_RUNTIME_SHAPES_SHA256",
    "V9_SOURCE_PACK_MANIFEST_SHA256",
    "derive_normalization_recovery_v9_runtime_profile",
    "normalization_recovery_v9_runtime_gate_code_files",
    "publish_normalization_recovery_v9_runtime_gate",
    "read_normalization_recovery_v9_runtime_probe_inputs",
]
