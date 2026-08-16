"""为 recovery-v10 五 family TRAIN 投影局部正字 hypothesis。"""
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
import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_projection as _base
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES = (
    "KEEPASSXC_PROJECT",
    "MIXXX_PROJECT",
    "MUMBLE_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)
V10_SOURCE_EXPANSION_LOCAL_PROJECTION_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_LOCAL_PROJECTION_V1")
V10_SOURCE_EXPANSION_LOCAL_PROJECTION_STATUS = (
    "FIVE_FAMILY_TRAIN_ONLY_HYPOTHESIS_TRACE_NO_AUTHORIZATION_NOT_FORMAL")


def _sha256(value: object) -> str:
    """返回evidence、span、record或summary的规范SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 expanded local projection {label} 非法")
    return value


def _sha_list(value: list[str], *, label: str) -> list[str]:
    """核验非空、有序、唯一的SHA列表。"""
    if (not isinstance(value, list) or not value
            or value != sorted(set(value))):
        raise BroadQaExternalDataError(
            f"v10 expanded local projection {label} 漂移")
    return [_sha_value(item, label=label) for item in value]


def _hypothesis_span(
        *,
        input_start: int,
        input_end: int,
        input_text: str,
        output_text: str,
        evidence_ids: list[str],
        opencc_route_ids: list[str],
        family: str,
        ) -> dict[str, object]:
    """构造五family可用、永不携带整句授权的局部span trace。"""
    if (type(input_start) is not int or input_start < 0
            or type(input_end) is not int or input_end <= input_start
            or not isinstance(input_text, str) or not input_text
            or not isinstance(output_text, str) or not output_text
            or input_text == output_text
            or family not in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 expanded local projection span semantic 非法")
    payload = {
        "authorization_kind": "LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY",
        "authorized_official_source_text": "",
        "conflict_count": 0,
        "evidence_ids": _sha_list(evidence_ids, label="evidence ids"),
        "identity_veto_count": 0,
        "input_end": input_end,
        "input_start": input_start,
        "input_text": input_text,
        "offset_unit": "UNICODE_SCALAR_INDEX",
        "opencc_route_ids": _sha_list(
            opencc_route_ids, label="OpenCC route ids"),
        "output_text": output_text,
        "source_context_authorization_id": "",
        "training_support_families": [family],
    }
    return {**payload, "span_hypothesis_id": _sha256(payload)}


def _observation_projection(
        observation: dict[str, object],
        *,
        routes: dict[str, str],
        opencc_source_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """把一条五family TRAIN Observation投影为零授权局部span trace。"""
    if not isinstance(observation, dict):
        raise BroadQaExternalDataError(
            "v10 expanded local projection observation 非对象")
    observation_id = _sha_value(
        observation.get("observation_id"), label="observation id")
    source_identity_sha = _sha_value(
        observation.get("source_identity_sha256"),
        label="source identity SHA")
    family = observation.get("source_family")
    official_source = observation.get("official_source_text")
    if (family not in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES
            or not isinstance(official_source, str) or not official_source):
        raise BroadQaExternalDataError(
            "v10 expanded local projection source/family 漂移")
    input_text = _base._surface(observation, "zh_hant")
    output_text = _base._surface(observation, "zh_hans")
    input_layout = localization_structure_layout(input_text)
    output_layout = localization_structure_layout(output_text)
    input_ledger = observation.get("zh_hant_structure_tokens")
    output_ledger = observation.get("zh_hans_structure_tokens")
    if (not isinstance(input_ledger, list)
            or not isinstance(output_ledger, list)
            or tuple(input_ledger) != input_layout["structure_tokens"]
            or tuple(output_ledger) != output_layout["structure_tokens"]):
        raise BroadQaExternalDataError(
            "v10 expanded local projection structure ledger 漂移")
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
                "v10 expanded local projection segment count 漂移")
        input_starts = _base._segment_starts(input_layout)
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
                runs = _base._changed_runs(
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
                        _base._opencc_route_id(
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
                            _base.NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
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
                    spans.append(_hypothesis_span(
                        input_start=absolute_start,
                        input_end=absolute_end,
                        input_text=left,
                        output_text=right,
                        evidence_ids=[evidence_id],
                        opencc_route_ids=route_ids,
                        family=str(family),
                    ))
    status = (
        "HYPOTHESIS_PROJECTED" if spans else
        "STRUCTURE_MISMATCH" if not structure_equal else
        "NO_SUPPORTED_LOCAL_HYPOTHESIS")
    payload = {
        "alignment_algorithm": (
            _base.NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
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


def derive_normalization_recovery_v10_source_expansion_local_projection(
        *,
        observations: tuple[dict[str, object], ...],
        opencc_routes: dict[str, str],
        opencc_source_pack_manifest_sha256: str,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """投影五family全量TRAIN记录并汇总零授权与拒绝原因。"""
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError(
            "v10 expanded local projection observations 为空")
    opencc_sha = _sha_value(
        opencc_source_pack_manifest_sha256, label="OpenCC source manifest")
    routes = _base._character_routes(opencc_routes)
    records = tuple(_observation_projection(
        item,
        routes=routes,
        opencc_source_pack_manifest_sha256=opencc_sha,
    ) for item in observations)
    if (len({item["observation_id"] for item in records}) != len(records)
            or set(str(item["source_family"]) for item in records)
            != set(V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES)):
        raise BroadQaExternalDataError(
            "v10 expanded local projection identity/family 漂移")
    status_counts = Counter(str(item["status"]) for item in records)
    family_record_counts = Counter(
        str(item["source_family"]) for item in records)
    family_hypothesis_counts = Counter()
    for item in records:
        family_hypothesis_counts[str(item["source_family"])] += int(
            item["hypothesis_count"])
    payload = {
        "alignment_algorithm": (
            _base.NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_ALIGNMENT),
        "artifact_kind": V10_SOURCE_EXPANSION_LOCAL_PROJECTION_KIND,
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
        "status": V10_SOURCE_EXPANSION_LOCAL_PROJECTION_STATUS,
        "status_counts": dict(sorted(status_counts.items())),
        "supported_opcode_count": sum(
            int(item["supported_opcode_count"]) for item in records),
        "teacher_api_llm_call_count": 0,
        "unsupported_opcode_count": sum(
            int(item["unsupported_opcode_count"]) for item in records),
    }
    return records, {**payload, "projection_summary_sha256": _sha256(payload)}


__all__ = [
    "V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES",
    "derive_normalization_recovery_v10_source_expansion_local_projection",
]
