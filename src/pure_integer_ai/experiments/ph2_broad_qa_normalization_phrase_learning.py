"""Normalization phrase learner 的无版本共享纯函数。

本模块只承载不含 family policy 的对齐、literal occurrence、ordered-work
计数与规范 JSONL 编码。版本化 learner 仍负责 Evidence、rule 与 scope 语义。
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def normalization_phrase_alignment_boundary_map(
        input_text: str,
        output_text: str,
        ) -> dict[int, set[int]]:
    """建立只含确定边界的 input-to-output 对齐表。"""
    result: dict[int, set[int]] = defaultdict(set)
    matcher = SequenceMatcher(None, input_text, output_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        result[i1].add(j1)
        result[i2].add(j2)
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for offset in range(i2 - i1 + 1):
                result[i1 + offset].add(j1 + offset)
        elif tag == "insert":
            result[i1].add(j2)
    return result


def normalization_phrase_context_signature(
        text: str,
        start: int,
        end: int,
        *,
        identity_builder,
        ) -> dict[str, object]:
    """截取 occurrence 两侧至多四个 scalar，并绑定稳定 identity。"""
    identity = {
        "left_boundary": int(start == 0),
        "left_context": text[max(0, start - 4):start],
        "right_boundary": int(end == len(text)),
        "right_context": text[end:min(len(text), end + 4)],
    }
    return {
        **identity,
        "context_signature_id": identity_builder(identity),
    }


def normalization_phrase_observed_output(
        observation: dict[str, object],
        start: int,
        end: int,
        boundaries: dict[int, set[int]],
        *,
        label: str,
        ) -> str | None:
    """只在 occurrence 两端均有唯一确定对齐时返回实际输出 span。"""
    input_text = observation.get("input_text")
    output_text = observation.get("output_text")
    if not isinstance(input_text, str) or not isinstance(output_text, str):
        raise BroadQaExternalDataError(f"{label} observation surface 漂移")
    starts = boundaries.get(start, set())
    ends = boundaries.get(end, set())
    if len(starts) != 1 or len(ends) != 1:
        return None
    output_start = next(iter(starts))
    output_end = next(iter(ends))
    if not 0 <= output_start <= output_end <= len(output_text):
        return None
    return output_text[output_start:output_end]


def normalization_phrase_occurrences(text: str, phrase: str):
    """按 scalar 起点产生允许重叠的全部 literal occurrence。"""
    start = text.find(phrase)
    while start >= 0:
        yield start, start + len(phrase)
        start = text.find(phrase, start + 1)


def require_normalization_phrase_work_alignment(
        *,
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        label: str,
        ) -> None:
    """要求 ordered work 精确覆盖 observation、fragment 与 group。"""
    expected = (
        [("PAIR_OBSERVATION_INGEST", "PAIR_OBSERVATION",
          item["observation_id"]) for item in observations]
        + [("PHRASE_FRAGMENT_INGEST", "PHRASE_FRAGMENT",
            item["fragment_id"]) for item in fragments]
        + [("PHRASE_GROUP_RESOLUTION", "PHRASE_GROUP",
            item["group_id"]) for item in groups]
    )
    observed = [(item.get("phase"), item.get("work_kind"),
                 item.get("record_id")) for item in work]
    if (observed != expected
            or [item.get("work_ordinal") for item in work]
            != list(range(len(work)))):
        raise BroadQaExternalDataError(
            f"{label} ordered work/material 漂移")


def normalization_phrase_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        processed_item_count: int,
        label: str,
        ) -> tuple[int, int]:
    """机械计算任意 ordered-work 前缀的 Evidence/result 数。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError(f"{label} processed prefix 非法")
    by_group = {str(item["group_id"]): item for item in emission_counts}
    if len(by_group) != len(emission_counts):
        raise BroadQaExternalDataError(
            f"{label} emission group identity 重复")
    evidence_count = 0
    result_count = 0
    seen_emissions = set()
    for item in work[:processed_item_count]:
        if item["work_kind"] != "PHRASE_GROUP":
            if item["work_kind"] not in {"PAIR_OBSERVATION", "PHRASE_FRAGMENT"}:
                raise BroadQaExternalDataError(f"{label} work kind 非法")
            continue
        counts = by_group.get(str(item["record_id"]))
        if counts is None:
            continue
        seen_emissions.add(str(item["record_id"]))
        evidence_count += int(counts["evidence_increment"])
        result_count += int(counts["result_increment"])
    if processed_item_count == len(work) and seen_emissions != set(by_group):
        raise BroadQaExternalDataError(
            f"{label} emission group 未被 work 覆盖")
    return evidence_count, result_count


def normalization_phrase_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        label: str,
        ) -> dict[str, bytes]:
    """把完整输出转为 manifest-last writer 的规范 JSONL。"""
    expected = {name for name, _role, _identity in output_file_roles}
    if set(outputs) != expected:
        raise BroadQaExternalDataError(f"{label} output inventory 漂移")
    return {
        name: b"".join(canonical_json_line(item) for item in outputs[name])
        for name, _role, _identity in output_file_roles
    }


__all__ = [
    "normalization_phrase_alignment_boundary_map",
    "normalization_phrase_context_signature",
    "normalization_phrase_observed_output",
    "normalization_phrase_occurrences",
    "normalization_phrase_output_payloads",
    "normalization_phrase_prefix_output_counts",
    "require_normalization_phrase_work_alignment",
]
