"""在唯一 formal guard 之后物化旧 Firefox reserve 的标签。

Family freeze 与 candidate 发布只读取标签盲 commitment。只有 runner 已不可逆写入
guard 后，本模块才允许读取 Firefox source pack 与 reserve identity，并从冻结来源
重建完整 expected output；不会按 candidate 结果重选或缩减分母。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
    NORMALIZATION_RECOVERY_RESERVE_RECORD_KIND,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
    _content_cluster_id,
    _has_han,
    _single_han_difference,
    normalization_recovery_evaluation_split,
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FIREFOX_L10N_COMMIT,
    read_normalization_recovery_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    read_normalization_recovery_v3_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha256(payload: bytes) -> str:
    """返回 reserve identity 或来源记录的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """严格读取规范 UTF-8 JSONL，不接受空行或非对象。"""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError("recovery v4 reserve identity 不可读") from error
    values = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                "recovery v4 reserve identity 非规范") from error
        if not isinstance(value, dict) or canonical_json_line(value) != line:
            raise BroadQaExternalDataError(
                "recovery v4 reserve identity encoding 漂移")
        values.append(value)
    if not values or b"".join(canonical_json_line(item) for item in values) != payload:
        raise BroadQaExternalDataError("recovery v4 reserve identity 为空或漂移")
    return tuple(values)


def _full_records(
        *,
        source_pack_manifest_sha256: str,
        pair_records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """以旧冻结算法重建 evaluation+reserve 的完整带标签记录。"""
    clusters: dict[str, dict[str, object]] = {}
    for record in pair_records:
        if (not isinstance(record, dict)
                or record.get("plain_pair_eligible") != 1
                or record.get("structure_equal") != 1):
            continue
        cn = record.get("zh_cn")
        tw = record.get("zh_tw")
        if not isinstance(cn, dict) or not isinstance(tw, dict):
            raise BroadQaExternalDataError(
                "recovery v4 Firefox pair provenance 非法")
        input_text = tw.get("surface_text")
        expected_output = cn.get("surface_text")
        skeleton = cn.get("pattern_skeleton_sha256")
        pair_id = record.get("pair_id")
        if (not isinstance(input_text, str)
                or not isinstance(expected_output, str)
                or not isinstance(skeleton, str)
                or not isinstance(pair_id, str)):
            raise BroadQaExternalDataError(
                "recovery v4 Firefox plain pair 字段非法")
        cluster_id = _content_cluster_id(
            skeleton, input_text, expected_output)
        cluster = clusters.setdefault(cluster_id, {
            "expected_output": expected_output,
            "input_text": input_text,
            "pair_ids": [],
            "skeleton_sha256": skeleton,
        })
        if (cluster["expected_output"] != expected_output
                or cluster["input_text"] != input_text
                or cluster["skeleton_sha256"] != skeleton):
            raise BroadQaExternalDataError(
                "recovery v4 Firefox content cluster 碰撞")
        cluster["pair_ids"].append(pair_id)
    local_by_input: dict[str, list[str]] = defaultdict(list)
    local_difference = {}
    for cluster_id, cluster in clusters.items():
        difference = _single_han_difference(
            str(cluster["input_text"]), str(cluster["expected_output"]))
        if difference is not None:
            local_difference[cluster_id] = difference
            local_by_input[difference[0]].append(cluster_id)
    local_outputs = {
        source: {local_difference[cluster_id][1]
                 for cluster_id in cluster_ids}
        for source, cluster_ids in local_by_input.items()
    }
    dimension_order = (
        "LOCAL_MAPPING_TRANSFER",
        "END_TO_END_COVERAGE",
        "INDEPENDENT_CONTEXT_TRANSFER",
    )
    full = []
    for cluster_id in sorted(clusters):
        cluster = clusters[cluster_id]
        input_text = str(cluster["input_text"])
        expected_output = str(cluster["expected_output"])
        difference = local_difference.get(cluster_id)
        family_keys = []
        context_sensitive = 0
        if difference is not None:
            if len(local_outputs[difference[0]]) == 1:
                family_keys.append("LOCAL_MAPPING_TRANSFER")
            else:
                family_keys.append("INDEPENDENT_CONTEXT_TRANSFER")
                context_sensitive = 1
        if (input_text != expected_output and _has_han(input_text + expected_output)
                and 2 <= len(input_text) <= 160
                and 2 <= len(expected_output) <= 160):
            family_keys.append("END_TO_END_COVERAGE")
        identity_preservation = int(
            input_text == expected_output and _has_han(input_text)
            and 1 <= len(input_text) <= 160)
        if identity_preservation:
            family_keys.append("END_TO_END_COVERAGE")
        if not family_keys:
            continue
        pair_ids = sorted(cluster["pair_ids"])
        if len(pair_ids) != len(set(pair_ids)):
            raise BroadQaExternalDataError(
                "recovery v4 Firefox pair identity 重复")
        split_group = (("LOCAL_INPUT\0" + difference[0])
                       if difference is not None
                       else ("CONTENT\0" + cluster_id))
        item = {
            "content_cluster_id": cluster_id,
            "context_sensitive": context_sensitive,
            "evaluation_id": _sha256((
                NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND + "\0"
                + NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE + "\0"
                + cluster_id).encode("utf-8")),
            "expected_output": expected_output,
            "family_keys": [key for key in dimension_order
                            if key in family_keys],
            "format_version": 2,
            "identity_preservation": identity_preservation,
            "input_scalar_count": len(input_text),
            "input_text": input_text,
            "output_scalar_count": len(expected_output),
            "record_kind": NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
            "source_commit": FIREFOX_L10N_COMMIT,
            "source_occurrence_count": len(pair_ids),
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "source_pair_id": pair_ids[0],
            "source_policy_scope": "MOZILLA_FIREFOX_L10N_ZH_TW_TO_ZH_CN",
            "split": normalization_recovery_evaluation_split(split_group),
            "split_group_sha256": _sha256(split_group.encode("utf-8")),
            "target_policy_scope": NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        }
        if difference is not None:
            item.update({
                "mapping_expected_character": difference[1],
                "mapping_input_character": difference[0],
                "mapping_offset": difference[2],
            })
        full.append(item)
    result = tuple(sorted(full, key=lambda item: str(item["evaluation_id"])))
    if not result or len({item["evaluation_id"] for item in result}) != len(result):
        raise BroadQaExternalDataError(
            "recovery v4 full evaluation identity 漂移")
    return result


def materialize_normalization_recovery_v4_reserve_after_guard(
        *,
        guard_consumed: int,
        prior_evaluation_protocol_dir: str | Path,
        expected_prior_evaluation_manifest_sha256: str,
        firefox_source_pack_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        expected_evaluation_commitment_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """在 guard 后重建完整 reserve label，并核对整分母 identity。"""
    if type(guard_consumed) is not int or guard_consumed != 1:
        raise BroadQaExternalDataError(
            "recovery v4 reserve label 只能在 formal guard 后物化")
    protocol_root = Path(prior_evaluation_protocol_dir).resolve()
    protocol = read_normalization_recovery_evaluation_manifest_only(
        protocol_root,
        expected_manifest_sha256=expected_prior_evaluation_manifest_sha256,
    )
    commitment = read_normalization_recovery_v3_evaluation_commitment(
        evaluation_commitment_dir,
        prior_evaluation_protocol_dir=protocol_root,
        expected_manifest_sha256=(
            expected_evaluation_commitment_manifest_sha256),
    )
    source_manifest, _sources, pairs = read_normalization_recovery_source_pack(
        firefox_source_pack_dir)
    if (source_manifest["manifest_sha256"]
            != commitment["source_exclusion"][
                "excluded_source_pack_manifest_sha256"]
            or protocol["source_pack_manifest_sha256"]
            != source_manifest["manifest_sha256"]):
        raise BroadQaExternalDataError(
            "recovery v4 Firefox source commitment 漂移")
    reserve_path = protocol_root / str(
        protocol["reserve_identity"]["relative_path"])
    reserve_identity = _read_jsonl(reserve_path)
    encoded = reserve_path.read_bytes()
    if (len(encoded) != protocol["reserve_identity"]["bytes"]
            or _sha256(encoded) != protocol["reserve_identity"]["sha256"]
            or len(reserve_identity)
            != protocol["reserve_identity"]["record_count"]):
        raise BroadQaExternalDataError(
            "recovery v4 reserve identity artifact 漂移")
    full = _full_records(
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        pair_records=pairs,
    )
    reserve = tuple(item for item in full if item["split"] == "RESERVE")
    derived_identity = tuple({
        "evaluation_id": item["evaluation_id"],
        "format_version": 2,
        "record_kind": NORMALIZATION_RECOVERY_RESERVE_RECORD_KIND,
        "split": "RESERVE",
        "split_group_sha256": item["split_group_sha256"],
    } for item in reserve)
    denominator = commitment["denominator"]
    if (not _strict_equal(derived_identity, reserve_identity)
            or len(reserve) != denominator["record_count"]
            or sum("END_TO_END_COVERAGE" in item["family_keys"]
                   for item in reserve) != denominator["coverage_count"]
            or sum("LOCAL_MAPPING_TRANSFER" in item["family_keys"]
                   for item in reserve) != denominator["local_mapping_count"]
            or sum(item["context_sensitive"] for item in reserve)
            != denominator["context_count"]
            or sum(item["identity_preservation"] for item in reserve)
            != denominator["identity_count"]):
        raise BroadQaExternalDataError(
            "recovery v4 reserve label/identity/denominator 漂移")
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "firefox_source_pack_manifest_sha256": source_manifest[
            "manifest_sha256"],
        "label_materialization_count": len(reserve),
        "prior_evaluation_protocol_manifest_sha256": protocol[
            "manifest_sha256"],
        "reserve_identity_sha256": protocol["reserve_identity"]["sha256"],
        "reserve_payload_read_count": 1,
    }
    return materialization, reserve


__all__ = [
    "materialize_normalization_recovery_v4_reserve_after_guard",
]
