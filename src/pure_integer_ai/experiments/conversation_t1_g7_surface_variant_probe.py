"""T1-G7 表层结构变体的最小真实运行 probe。

该 probe 只观察已学习的 literal gap 变体是否能在新 typed slot 上重组。
输入是调用方提供的探针句，不是答案目录；运行时不会读取 gold、不会注入
事实，也不会改变默认 v6 表层消费者。报告明确不是广域问答、事实生成或
断奶 readiness 证据。
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_organization_compile import (
    load_surface_organization_jsonl,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    SurfaceRenderResult,
    load_trained_surface_runtime,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


G7_PROBE_PROTOCOL_V1 = 1
G7_PROBE_OBSERVED = "OBSERVED"
G7_PROBE_NE = "NE"
_G7_DOMAIN = "pure_integer_ai.t1.g7.surface-variant-probe.v1"


class G7ProbeError(ValueError):
    """G7 probe 输入、运行身份或证据合同无效。"""


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise G7ProbeError(f"{where} 必须是规范字符串")
    if not allow_empty and not value:
        raise G7ProbeError(f"{where} 不能为空")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise G7ProbeError(f"{where} 含非 Unicode scalar")
    return value


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    value = _text(value, where)
    scalars = tuple(ord(item) for item in value)
    return (len(scalars), *scalars)


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    if any(type(item) is not int or item < 0 for item in values):
        raise G7ProbeError("probe 整数记录非法")
    return (len(values), *values)


def _sha256_integer_record(values: tuple[int, ...]) -> str:
    """对规范整数流做跨语言可复现的 SHA-256。"""
    return hashlib.sha256(encode_integer_tuple(values)).hexdigest()


@dataclass(frozen=True, slots=True)
class G7ProbeObservation:
    """一次表层变体运行的可回放纯值投影。"""

    input_surface: str
    output_surface: str
    used: bool
    reason: str
    pattern_id: int
    run_id: str
    graph_size: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.input_surface, "observation.input_surface")
        _text(self.output_surface, "observation.output_surface")
        _text(self.reason, "observation.reason")
        _text(self.run_id, "observation.run_id")
        if type(self.used) is not bool:
            raise G7ProbeError("observation.used 必须是 bool")
        if type(self.pattern_id) is not int or self.pattern_id < 0:
            raise G7ProbeError("observation.pattern_id 必须是非负整数")
        if type(self.graph_size) is not int or self.graph_size <= 0:
            raise G7ProbeError("observation.graph_size 必须是正整数")
        if (not self.trace or
                any(type(item) is not int or item < 0 for item in self.trace)):
            raise G7ProbeError("observation.trace 必须是非空非负整数 tuple")

    def canonical_record(self) -> tuple[int, ...]:
        result = [G7_PROBE_PROTOCOL_V1, int(self.used), self.pattern_id,
                  self.graph_size]
        for value, label in (
                (self.input_surface, "input_surface"),
                (self.output_surface, "output_surface"),
                (self.reason, "reason"),
                (self.run_id, "run_id")):
            result.extend(_pack_text(value, f"observation.{label}"))
        result.extend(_pack(self.trace))
        return tuple(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_size": self.graph_size,
            "input_surface": self.input_surface,
            "output_surface": self.output_surface,
            "pattern_id": self.pattern_id,
            "reason": self.reason,
            "run_id": self.run_id,
            "trace_u": list(self.trace),
            "used": self.used,
        }


def _observation(result: SurfaceRenderResult,
                 input_surface: str) -> G7ProbeObservation:
    return G7ProbeObservation(
        input_surface, result.surface, result.used, result.reason,
        result.pattern_id, result.run_id, result.graph_size, result.trace,
    )


def build_g7_surface_variant_probe(
        *,
        project_root: str | Path,
        training_run_root: str | Path,
        expected_pack_sha256: str,
        variant_course_path: str | Path,
        variant_evidence_path: str | Path,
        input_surface: str,
        source_title: str = "g7-surface-variant-probe",
        ) -> dict[str, object]:
    """执行两遍 opt-in G7 runtime，并返回未宣称能力的观察摘要。"""
    root = Path(project_root).resolve()
    input_surface = _text(input_surface, "probe.input_surface")
    source_title = _text(source_title, "probe.source_title")
    course_path = Path(variant_course_path).resolve()
    evidence_path = Path(variant_evidence_path).resolve()
    if not course_path.is_file() or not evidence_path.is_file():
        raise G7ProbeError("G7 variant course/evidence 缺失")
    records = tuple(item.record for item in
                    load_surface_organization_jsonl(course_path.read_bytes()))
    accepted_surfaces = {
        variant.surface for record in records for variant in record.accepted
    }
    if input_surface in accepted_surfaces:
        raise G7ProbeError("probe input 不得是 G7 course 的原句回放")
    runtime = load_trained_surface_runtime(
        project_root=root,
        training_run_root=training_run_root,
        expected_pack_sha256=expected_pack_sha256,
        extra_variant_course_paths=(course_path,),
        extra_variant_evidence_paths=(evidence_path,),
    )
    first = _observation(
        runtime.render(input_surface, response_act="ANSWER",
                       source_title=source_title, ordinal=0), input_surface)
    replay = _observation(
        runtime.render(input_surface, response_act="ANSWER",
                       source_title=source_title, ordinal=0), input_surface)
    guard_results: list[dict[str, object]] = []
    for response_act in ("UNKNOWN", "CLARIFY", "REPAIR"):
        guarded = runtime.render(input_surface, response_act=response_act,
                                 source_title=source_title, ordinal=0)
        guard_results.append({
            "response_act": response_act,
            "surface": guarded.surface,
            "used": guarded.used,
            "reason": guarded.reason,
        })
    guard_ok = all(
        item["used"] is False and item["surface"] == input_surface
        for item in guard_results)
    observed = (
        first.used
        and first.output_surface != first.input_surface
        and first.reason == "variant_selected"
        and first.pattern_id > 0
        and first.output_surface not in accepted_surfaces
        and first == replay
        and guard_ok
    )
    replay_record_sha256 = _sha256_integer_record(first.canonical_record())
    replay_record_sha256_replay = _sha256_integer_record(replay.canonical_record())
    return {
        "artifact_kind": "T1_G7_SURFACE_VARIANT_RUNTIME_PROBE_V1",
        "capability_scope": "surface_structure_variant_only",
        "format_version": G7_PROBE_PROTOCOL_V1,
        "not_claimed": ["broad_qa", "fact_generation", "weaning_readiness"],
        "probe_status": G7_PROBE_OBSERVED if observed else G7_PROBE_NE,
        "input_not_course_surface": input_surface not in accepted_surfaces,
        "output_not_course_surface": first.output_surface not in accepted_surfaces,
        "replay_bit_identical": first == replay,
        "replay_record_sha256": replay_record_sha256,
        "replay_record_sha256_replay": replay_record_sha256_replay,
        "training": {
            "run_id": first.run_id,
            "graph_size": first.graph_size,
            "pack_sha256": expected_pack_sha256,
        },
        "observation": first.to_dict(),
        "replay_observation": replay.to_dict(),
        "non_answer_guards": guard_results,
    }


def write_g7_surface_variant_probe(value: dict[str, object],
                                   output_path: str | Path) -> str:
    """排他写入 K 盘 canonical JSON，不覆盖既有证据。"""
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise G7ProbeError("probe output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    return str(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run T1-G7 surface variant probe")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--expected-pack-sha256", required=True)
    parser.add_argument("--variant-course", required=True)
    parser.add_argument("--variant-evidence", required=True)
    parser.add_argument("--input-surface", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build_g7_surface_variant_probe(
        project_root=args.project_root,
        training_run_root=args.training_run_root,
        expected_pack_sha256=args.expected_pack_sha256,
        variant_course_path=args.variant_course,
        variant_evidence_path=args.variant_evidence,
        input_surface=args.input_surface,
    )
    write_g7_surface_variant_probe(value, args.output)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


__all__ = [
    "G7_PROBE_NE", "G7_PROBE_OBSERVED", "G7_PROBE_PROTOCOL_V1",
    "G7ProbeError", "G7ProbeObservation", "build_g7_surface_variant_probe",
    "main", "write_g7_surface_variant_probe",
]


if __name__ == "__main__":
    raise SystemExit(main())
