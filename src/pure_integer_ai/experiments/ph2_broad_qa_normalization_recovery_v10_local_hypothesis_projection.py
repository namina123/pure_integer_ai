"""从 TRAIN Observation 投影 recovery-v10 局部正字 hypothesis evidence。"""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_contract import (
    LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY,
    build_normalization_recovery_v10_local_span_hypothesis,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_V1")
NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_STATUS = (
    "TRAIN_ONLY_HYPOTHESIS_TRACE_NO_AUTHORIZATION_NOT_FORMAL")
NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT = (
    "PYTHON_SEQUENCE_MATCHER_AUTOJUNK_FALSE_STRUCTURE_SEGMENT_V1")


def _sha256(value: object) -> str:
    """返回 evidence、route、record 或 summary 的规范 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis projection {label} 非法")
    return value


def _surface(observation: dict[str, object], role: str) -> str:
    """从 Qt/gettext 统一 Observation 提取唯一 locale 表面。"""
    value = observation.get(role)
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection locale record 漂移")
    strings = [item for item in (
        value.get("translation"), value.get("msgstr"))
        if isinstance(item, str)]
    if len(strings) != 1 or not strings[0]:
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection surface 漂移")
    return strings[0]


def _character_routes(routes: dict[str, str]) -> dict[str, str]:
    """只保留一对一 changed scalar OpenCC 路由。"""
    if (not isinstance(routes, dict) or not routes
            or any(not isinstance(left, str) or not left
                   or not isinstance(right, str) or not right
                   for left, right in routes.items())):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection OpenCC routes 非法")
    values = {
        left: right for left, right in routes.items()
        if len(left) == 1 and len(right) == 1 and left != right
    }
    if not values:
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection character routes 为空")
    return values


def _segment_starts(layout: dict[str, tuple[str, ...]]) -> tuple[int, ...]:
    """把结构 layout 的 text segment 投影为完整 surface 绝对起点。"""
    segments = layout.get("segments")
    raw_tokens = layout.get("raw_tokens")
    if (not isinstance(segments, tuple) or not isinstance(raw_tokens, tuple)
            or len(segments) != len(raw_tokens) + 1):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection layout 漂移")
    starts = []
    position = 0
    for ordinal, segment in enumerate(segments):
        if not isinstance(segment, str):
            raise BroadQaExternalDataError(
                "v10 local hypothesis projection segment 非字符串")
        starts.append(position)
        position += len(segment)
        if ordinal < len(raw_tokens):
            token = raw_tokens[ordinal]
            if not isinstance(token, str) or not token:
                raise BroadQaExternalDataError(
                    "v10 local hypothesis projection raw token 漂移")
            position += len(token)
    return tuple(starts)


def _opencc_route_id(
        input_atom: str, output_atom: str, *,
        opencc_source_pack_manifest_sha256: str,
        ) -> str:
    """形成与 v10 precision candidate 一致的 OpenCC route identity。"""
    return _sha256({
        "input_atom": input_atom,
        "opencc_source_pack_manifest_sha256": (
            opencc_source_pack_manifest_sha256),
        "output_atom": output_atom,
        "rule_kind": "OPENCC_CHARACTER_ROUTE",
    })


def _changed_runs(
        *, source: str, target: str,
        source_start: int, target_start: int,
        routes: dict[str, str],
        ) -> tuple[tuple[int, int], ...] | None:
    """要求等长replace中每个changed scalar均被OpenCC支持，并返回连续run。"""
    if len(source) != len(target) or not source:
        return None
    changed = []
    for offset, (left, right) in enumerate(zip(source, target)):
        if left == right:
            continue
        if routes.get(left) != right:
            return None
        changed.append(offset)
    if not changed:
        return ()
    runs = []
    run_start = changed[0]
    previous = changed[0]
    for offset in changed[1:]:
        if offset != previous + 1:
            runs.append((run_start, previous + 1))
            run_start = offset
        previous = offset
    runs.append((run_start, previous + 1))
    if any(not source_start <= source_start + start < source_start + end
           or not target_start <= target_start + start < target_start + end
           for start, end in runs):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection changed run 漂移")
    return tuple(runs)


def _observation_projection(
        observation: dict[str, object], *,
        routes: dict[str, str],
        opencc_source_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """把一条 TRAIN Observation 投影为零授权的局部span trace。"""
    if not isinstance(observation, dict):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection observation 非对象")
    observation_id = _sha_value(
        observation.get("observation_id"), label="observation id")
    source_identity_sha = _sha_value(
        observation.get("source_identity_sha256"),
        label="source identity SHA")
    family = observation.get("source_family")
    official_source = observation.get("official_source_text")
    if (family not in V8_TRAIN_FAMILIES
            or not isinstance(official_source, str) or not official_source):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection source/family 漂移")
    input_text = _surface(observation, "zh_hant")
    output_text = _surface(observation, "zh_hans")
    input_layout = localization_structure_layout(input_text)
    output_layout = localization_structure_layout(output_text)
    input_ledger = observation.get("zh_hant_structure_tokens")
    output_ledger = observation.get("zh_hans_structure_tokens")
    if (not isinstance(input_ledger, list)
            or not isinstance(output_ledger, list)
            or tuple(input_ledger) != input_layout["structure_tokens"]
            or tuple(output_ledger) != output_layout["structure_tokens"]):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection structure ledger 漂移")
    structure_equal = int(
        input_layout["structure_tokens"] == output_layout["structure_tokens"])
    spans = []
    unsupported_opcode_count = 0
    supported_opcode_count = 0
    if structure_equal:
        input_segments = input_layout["segments"]
        output_segments = output_layout["segments"]
        if len(input_segments) != len(output_segments):
            raise BroadQaExternalDataError(
                "v10 local hypothesis projection segment count 漂移")
        input_starts = _segment_starts(input_layout)
        for segment_ordinal, (source_segment, target_segment) in enumerate(
                zip(input_segments, output_segments)):
            matcher = SequenceMatcher(
                None, source_segment, target_segment, autojunk=False)
            for opcode_ordinal, (tag, i1, i2, j1, j2) in enumerate(
                    matcher.get_opcodes()):
                if tag == "equal":
                    continue
                if tag != "replace" or i2 - i1 != j2 - j1:
                    unsupported_opcode_count += 1
                    continue
                runs = _changed_runs(
                    source=source_segment[i1:i2],
                    target=target_segment[j1:j2],
                    source_start=i1,
                    target_start=j1,
                    routes=routes,
                )
                if runs is None:
                    unsupported_opcode_count += 1
                    continue
                supported_opcode_count += 1
                for run_ordinal, (run_start, run_end) in enumerate(runs):
                    absolute_start = input_starts[segment_ordinal] + i1 + run_start
                    absolute_end = input_starts[segment_ordinal] + i1 + run_end
                    left = source_segment[i1 + run_start:i1 + run_end]
                    right = target_segment[j1 + run_start:j1 + run_end]
                    route_ids = sorted({
                        _opencc_route_id(
                            input_atom,
                            output_atom,
                            opencc_source_pack_manifest_sha256=(
                                opencc_source_pack_manifest_sha256),
                        )
                        for input_atom, output_atom in zip(left, right)
                        if input_atom != output_atom
                    })
                    evidence_id = _sha256({
                        "alignment_algorithm": (
                            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
                        "input_end": absolute_end,
                        "input_start": absolute_start,
                        "input_text": left,
                        "observation_id": observation_id,
                        "opcode_ordinal": opcode_ordinal,
                        "output_text": right,
                        "run_ordinal": run_ordinal,
                        "segment_ordinal": segment_ordinal,
                        "source_identity_sha256": source_identity_sha,
                    })
                    spans.append(
                        build_normalization_recovery_v10_local_span_hypothesis(
                            input_start=absolute_start,
                            input_end=absolute_end,
                            input_text=left,
                            output_text=right,
                            authorization_kind=(
                                LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY),
                            evidence_ids=[evidence_id],
                            opencc_route_ids=route_ids,
                            training_support_families=[str(family)],
                            source_context_authorization_id="",
                            authorized_official_source_text="",
                        ))
    status = (
        "HYPOTHESIS_PROJECTED" if spans else
        "STRUCTURE_MISMATCH" if not structure_equal else
        "NO_SUPPORTED_LOCAL_HYPOTHESIS")
    payload = {
        "alignment_algorithm": (
            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
        "authorization_count": 0,
        "hypothesis_count": len(spans),
        "input_text": input_text,
        "observation_id": observation_id,
        "official_source_text": official_source,
        "source_family": family,
        "source_identity_sha256": source_identity_sha,
        "span_hypotheses": spans,
        "status": status,
        "structure_equal": structure_equal,
        "structure_tokens": list(input_layout["structure_tokens"]),
        "supported_opcode_count": supported_opcode_count,
        "unsupported_opcode_count": unsupported_opcode_count,
    }
    return {**payload, "projection_record_sha256": _sha256(payload)}


def derive_normalization_recovery_v10_local_hypothesis_projection(
        *, observations: tuple[dict[str, object], ...],
        opencc_routes: dict[str, str],
        opencc_source_pack_manifest_sha256: str,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """投影全量TRAIN记录并汇总零授权、family与拒绝原因。"""
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection observations 为空")
    opencc_sha = _sha_value(
        opencc_source_pack_manifest_sha256, label="OpenCC source manifest")
    routes = _character_routes(opencc_routes)
    records = tuple(_observation_projection(
        item,
        routes=routes,
        opencc_source_pack_manifest_sha256=opencc_sha,
    ) for item in observations)
    if len({item["observation_id"] for item in records}) != len(records):
        raise BroadQaExternalDataError(
            "v10 local hypothesis projection observation identity 重复")
    status_counts = Counter(str(item["status"]) for item in records)
    family_record_counts = Counter(
        str(item["source_family"]) for item in records)
    family_hypothesis_counts = Counter()
    for item in records:
        family_hypothesis_counts[str(item["source_family"])] += int(
            item["hypothesis_count"])
    payload = {
        "alignment_algorithm": (
            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_KIND),
        "authorization_count": 0,
        "family_hypothesis_counts": dict(sorted(
            family_hypothesis_counts.items())),
        "family_record_counts": dict(sorted(family_record_counts.items())),
        "formal_or_evaluation_payload_read_count": 0,
        "hypothesis_count": sum(
            int(item["hypothesis_count"]) for item in records),
        "mastery_claimed": 0,
        "observation_count": len(records),
        "opencc_source_pack_manifest_sha256": opencc_sha,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_STATUS,
        "status_counts": dict(sorted(status_counts.items())),
        "supported_opcode_count": sum(
            int(item["supported_opcode_count"]) for item in records),
        "teacher_api_llm_call_count": 0,
        "unsupported_opcode_count": sum(
            int(item["unsupported_opcode_count"]) for item in records),
    }
    return records, {**payload, "projection_summary_sha256": _sha256(payload)}


__all__ = [
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_KIND",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_PROJECTION_STATUS",
    "derive_normalization_recovery_v10_local_hypothesis_projection",
]
