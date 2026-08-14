"""冻结 normalization recovery v2 的独立 evaluation split 与六维门。

协议只消费尚未用于训练的 Firefox ``zh-TW``/``zh-CN`` 对齐来源。
它在 Unihan/MediaWiki 转入 recovery TRAIN 前冻结 label、reserve、metric 与阈值。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FIREFOX_L10N_COMMIT,
    read_normalization_recovery_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_V2")
NORMALIZATION_RECOVERY_EVALUATION_STATUS = (
    "FROZEN_BEFORE_RECOVERY_TRAINING_NOT_EVALUATED")
NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_EVALUATION_ITEM_V2")
NORMALIZATION_RECOVERY_RESERVE_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_RESERVE_IDENTITY_V2")
NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE = (
    "ZH_CN_FIREFOX_LOCALIZATION_TRANSFER_V2")
NORMALIZATION_RECOVERY_SPLIT_SEED = (
    "NORMALIZATION_RECOVERY_FIREFOX_SPLIT_V2")

NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS = {
    "DEFEATER_REPRESENTATION_EXECUTABILITY": {
        "bearing": 1,
        "declared_defeater_count_min": 1,
        "declared_must_equal_executable": 1,
        "identity_only_defeater_count_max": 0,
        "malformed_defeater_count_max": 0,
        "missing_candidate_outcome": "NE",
    },
    "END_TO_END_COVERAGE": {
        "applicable_phrase_must_equal_inventory": 1,
        "bearing": 1,
        "false_accept_count_max": 0,
        "false_reject_count_max": 0,
        "full_output_match_must_equal_applicable": 1,
        "identity_false_accept_count_max": 0,
        "identity_inventory_min": 192,
        "phrase_inventory_min": 4_096,
        "wrong_changed_position_count_max": 0,
    },
    "INDEPENDENT_CONTEXT_TRANSFER": {
        "applicable_context_must_equal_inventory": 1,
        "bearing": 1,
        "context_exact_support_must_equal_applicable": 1,
        "context_inventory_min": 64,
        "context_wrong_output_count_max": 0,
        "missing_applicable_context_outcome": "NE",
    },
    "LOCAL_MAPPING_TRANSFER": {
        "applicable_mapping_must_equal_inventory": 1,
        "bearing": 1,
        "local_mapping_inventory_min": 320,
        "mapping_false_accept_count_max": 0,
        "mapping_false_reject_count_max": 0,
        "supported_must_equal_applicable": 1,
        "unscoped_rule_count_max": 0,
    },
    "RUNTIME_PRODUCTION_BEHAVIOR": {
        "bearing": 1,
        "canonical_replay_mismatch_count_max": 0,
        "evaluation_input_execution_required": 1,
        "exception_count_max": 0,
        "indexed_reference_mismatch_count_max": 0,
        "production_enabled_must_equal": 0,
        "target_policy_scope_required": 1,
    },
    "SOURCE_POLICY_CONFLICT": {
        "bearing": 1,
        "declared_conflict_count_min": 1,
        "declared_conflict_must_equal_observed": 1,
        "missing_conflict_facility_outcome": "NE",
        "policy_specific_replay_must_equal_observation": 1,
        "training_source_policy_count_min": 4,
        "unscoped_conflict_execution_count_max": 0,
    },
}

NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT = {
    "bearing_dimension_count": 6,
    "candidate_applicability_cannot_remove_denominator": 1,
    "evaluation_run_count_max": 1,
    "formal_source": "MOZILLA_FIREFOX_L10N_FIXED_COMMIT",
    "identity_preservation_is_end_to_end_hard_conjunct": 1,
    "local_mapping_unit": (
        "one plain aligned message with exactly one Han-to-Han difference; "
        "input character has one output across full frozen source"),
    "mastery_claimed_before_all_bearing_pass": 0,
    "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
    "phrase_unit": (
        "deduplicated structure-equal plain Fluent pattern; exact zh-CN output"),
    "production_enablement_during_evaluation": 0,
    "reserve_payload_read_count_max": 0,
    "target_policy_is_source_scoped_not_absolute_truth": 1,
    "teacher_api_llm_call_count_max": 0,
}

_DIMENSION_ORDER = (
    "LOCAL_MAPPING_TRANSFER",
    "END_TO_END_COVERAGE",
    "SOURCE_POLICY_CONFLICT",
    "DEFEATER_REPRESENTATION_EXECUTABILITY",
    "INDEPENDENT_CONTEXT_TRANSFER",
    "RUNTIME_PRODUCTION_BEHAVIOR",
)


def _sha256(payload: bytes) -> str:
    """返回协议输入或规范 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


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


def _is_han(item: str) -> bool:
    """判断单个 Unicode scalar 是否位于汉字区段。"""
    if not isinstance(item, str) or len(item) != 1:
        return False
    codepoint = ord(item)
    return (0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2FA1F)


def _has_han(value: str) -> bool:
    """判断文本是否含至少一个汉字 scalar。"""
    return any(_is_han(item) for item in value)


def _content_cluster_id(
        skeleton_sha256: str,
        input_text: str,
        expected_output: str,
        ) -> str:
    """把重复 message/attribute 对齐聚为一个内容分母。"""
    return _sha256((
        "NORMALIZATION_RECOVERY_CONTENT_CLUSTER_V2\0"
        + skeleton_sha256 + "\0" + input_text + "\0" + expected_output
    ).encode("utf-8"))


def normalization_recovery_evaluation_split(split_group: str) -> str:
    """按冻结 group 做 4:1 evaluation/reserve 划分。"""
    if not isinstance(split_group, str) or not split_group:
        raise BroadQaExternalDataError("recovery evaluation split group 非法")
    digest = hashlib.sha256((
        NORMALIZATION_RECOVERY_SPLIT_SEED + "\0" + split_group
    ).encode("utf-8")).digest()
    return "EVALUATION" if digest[0] % 5 < 4 else "RESERVE"


def _single_han_difference(
        input_text: str,
        expected_output: str,
        ) -> tuple[str, str, int] | None:
    """返回唯一 Han-to-Han 差异；词序或多差异不得伪装为局部映射。"""
    if (len(input_text) != len(expected_output)
            or not 1 <= len(input_text) <= 160):
        return None
    differences = [
        (source, target, offset)
        for offset, (source, target) in enumerate(
            zip(input_text, expected_output))
        if source != target
    ]
    if (len(differences) != 1 or not _is_han(differences[0][0])
            or not _is_han(differences[0][1])):
        return None
    return differences[0]


def derive_normalization_recovery_evaluation_inventory(
        *,
        source_pack_manifest_sha256: str,
        pair_records: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从 Firefox source pack 确定性派生 evaluation 与无 label reserve。"""
    source_sha = _sha_value(
        source_pack_manifest_sha256, label="recovery source pack manifest")
    if not isinstance(pair_records, tuple) or not pair_records:
        raise BroadQaExternalDataError("recovery evaluation source inventory 为空")
    clusters: dict[str, dict[str, object]] = {}
    for record in pair_records:
        if (not isinstance(record, dict)
                or record.get("plain_pair_eligible") != 1
                or record.get("structure_equal") != 1):
            continue
        cn = record.get("zh_cn")
        tw = record.get("zh_tw")
        if not isinstance(cn, dict) or not isinstance(tw, dict):
            raise BroadQaExternalDataError("Firefox pair provenance 非法")
        input_text = tw.get("surface_text")
        expected_output = cn.get("surface_text")
        skeleton = cn.get("pattern_skeleton_sha256")
        pair_id = record.get("pair_id")
        if (not isinstance(input_text, str)
                or not isinstance(expected_output, str)
                or not isinstance(skeleton, str)
                or not isinstance(pair_id, str)):
            raise BroadQaExternalDataError("Firefox plain pair 字段非法")
        cluster_id = _content_cluster_id(
            skeleton, input_text, expected_output)
        cluster = clusters.setdefault(cluster_id, {
            "expected_output": expected_output,
            "input_text": input_text,
            "pair_ids": [],
            "skeleton_sha256": skeleton,
        })
        if (cluster["input_text"] != input_text
                or cluster["expected_output"] != expected_output
                or cluster["skeleton_sha256"] != skeleton):
            raise BroadQaExternalDataError("Firefox content cluster 碰撞")
        cluster["pair_ids"].append(pair_id)
    local_by_input: dict[str, list[str]] = defaultdict(list)
    local_difference: dict[str, tuple[str, str, int]] = {}
    for cluster_id, cluster in clusters.items():
        difference = _single_han_difference(
            str(cluster["input_text"]), str(cluster["expected_output"]))
        if difference is not None:
            local_difference[cluster_id] = difference
            local_by_input[difference[0]].append(cluster_id)
    local_outputs = {
        source: {local_difference[cluster_id][1] for cluster_id in cluster_ids}
        for source, cluster_ids in local_by_input.items()
    }
    full = []
    for cluster_id in sorted(clusters):
        cluster = clusters[cluster_id]
        input_text = str(cluster["input_text"])
        expected_output = str(cluster["expected_output"])
        family_keys = []
        difference = local_difference.get(cluster_id)
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
        if len(set(pair_ids)) != len(pair_ids):
            raise BroadQaExternalDataError("Firefox cluster pair identity 重复")
        if difference is not None:
            split_group = "LOCAL_INPUT\0" + difference[0]
        else:
            split_group = "CONTENT\0" + cluster_id
        evaluation_id = _sha256((
            NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND + "\0"
            + NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE + "\0" + cluster_id
        ).encode("utf-8"))
        item = {
            "content_cluster_id": cluster_id,
            "context_sensitive": context_sensitive,
            "evaluation_id": evaluation_id,
            "expected_output": expected_output,
            "family_keys": [key for key in _DIMENSION_ORDER
                            if key in family_keys],
            "format_version": 2,
            "identity_preservation": identity_preservation,
            "input_scalar_count": len(input_text),
            "input_text": input_text,
            "output_scalar_count": len(expected_output),
            "record_kind": NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
            "source_commit": FIREFOX_L10N_COMMIT,
            "source_occurrence_count": len(pair_ids),
            "source_pack_manifest_sha256": source_sha,
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
    identities = [item["evaluation_id"] for item in full]
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError("recovery evaluation identity 重复")
    evaluation = tuple(sorted(
        (item for item in full if item["split"] == "EVALUATION"),
        key=lambda item: str(item["evaluation_id"])))
    reserve_source = tuple(sorted(
        (item for item in full if item["split"] == "RESERVE"),
        key=lambda item: str(item["evaluation_id"])))
    reserve = tuple({
        "evaluation_id": item["evaluation_id"],
        "format_version": 2,
        "record_kind": NORMALIZATION_RECOVERY_RESERVE_RECORD_KIND,
        "split": "RESERVE",
        "split_group_sha256": item["split_group_sha256"],
    } for item in reserve_source)
    evaluation_families = Counter(
        key for item in evaluation for key in item["family_keys"])
    reserve_families = Counter(
        key for item in reserve_source for key in item["family_keys"])
    summary = {
        "content_cluster_count": len(clusters),
        "context_evaluation_count": sum(
            item["context_sensitive"] for item in evaluation),
        "context_reserve_count": sum(
            item["context_sensitive"] for item in reserve_source),
        "evaluation_count": len(evaluation),
        "evaluation_family_counts": dict(sorted(evaluation_families.items())),
        "identity_evaluation_count": sum(
            item["identity_preservation"] for item in evaluation),
        "identity_reserve_count": sum(
            item["identity_preservation"] for item in reserve_source),
        "local_mapping_evaluation_count": evaluation_families[
            "LOCAL_MAPPING_TRANSFER"],
        "local_mapping_reserve_count": reserve_families[
            "LOCAL_MAPPING_TRANSFER"],
        "phrase_evaluation_count": sum(
            item["input_text"] != item["expected_output"]
            and "END_TO_END_COVERAGE" in item["family_keys"]
            for item in evaluation),
        "phrase_reserve_count": sum(
            item["input_text"] != item["expected_output"]
            and "END_TO_END_COVERAGE" in item["family_keys"]
            for item in reserve_source),
        "reserve_count": len(reserve),
        "reserve_family_counts": dict(sorted(reserve_families.items())),
        "selected_inventory_count": len(full),
        "split_overlap_count": len(
            {item["evaluation_id"] for item in evaluation}.intersection(
                item["evaluation_id"] for item in reserve)),
    }
    dimensions = NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS
    if (not evaluation or not reserve or summary["split_overlap_count"] != 0
            or summary["local_mapping_evaluation_count"]
            < dimensions["LOCAL_MAPPING_TRANSFER"][
                "local_mapping_inventory_min"]
            or summary["phrase_evaluation_count"]
            < dimensions["END_TO_END_COVERAGE"]["phrase_inventory_min"]
            or summary["identity_evaluation_count"]
            < dimensions["END_TO_END_COVERAGE"]["identity_inventory_min"]
            or summary["context_evaluation_count"]
            < dimensions["INDEPENDENT_CONTEXT_TRANSFER"][
                "context_inventory_min"]):
        raise BroadQaExternalDataError(
            "normalization recovery evaluation family 库存不足")
    return evaluation, reserve, summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(f"{label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} JSONL 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """返回冻结 protocol 文件身份。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _validate_artifact_identity(
        value: object,
        *,
        relative_path: str,
        role: str,
        ) -> dict[str, object]:
    """核验 manifest-only reader 可验证的 artifact identity。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("recovery evaluation artifact 非法")
    expected_keys = {"bytes", "record_count", "relative_path", "role", "sha256"}
    if (set(value) != expected_keys
            or value.get("relative_path") != relative_path
            or value.get("role") != role
            or type(value.get("bytes")) is not int or value["bytes"] < 0
            or type(value.get("record_count")) is not int
            or value["record_count"] < 0):
        raise BroadQaExternalDataError("recovery evaluation artifact 漂移")
    _sha_value(value.get("sha256"), label="recovery evaluation artifact SHA")
    return value


def _manifest(
        *,
        source_pack_manifest_sha256: str,
        evaluation_artifact: dict[str, object],
        reserve_artifact: dict[str, object],
        inventory_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造在 recovery TRAIN 读取前冻结的 protocol manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
        "candidate_pack_read_count": 0,
        "dimensions": NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS,
        "evaluation_inventory": evaluation_artifact,
        "evaluation_run_count": 0,
        "format_version": 2,
        "inventory_summary": inventory_summary,
        "learned_pack_read_count": 0,
        "mastery_claimed": 0,
        "metric_contract": NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "recovery_training_source_read_count": 0,
        "reserve_identity": reserve_artifact,
        "reserve_payload_read_count": 0,
        "selection_rule": {
            "content_cluster": (
                "sha256(pattern skeleton + NUL + zh-TW + NUL + zh-CN); "
                "duplicate occurrences count once"),
            "context": (
                "single Han difference whose input character has multiple "
                "outputs across the full frozen source"),
            "identity": (
                "structure-equal plain pair; identical; contains Han; "
                "1..160 scalars"),
            "local": (
                "structure-equal plain pair; equal scalar length; exactly one "
                "Han-to-Han difference; input character has one source output"),
            "phrase": (
                "structure-equal plain pair; non-identity; contains Han; "
                "both surfaces 2..160 scalars"),
            "split": (
                "local/context group by input character; other records by "
                "content cluster; sha256(seed + NUL + group)[0] mod 5; "
                "0..3=EVALUATION, 4=RESERVE"),
        },
        "source_pack_manifest_sha256": source_pack_manifest_sha256,
        "status": NORMALIZATION_RECOVERY_EVALUATION_STATUS,
        "target_policy_scope": NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery evaluation run root 必须是 K 盘目录")
    return root


def publish_normalization_recovery_evaluation_protocol(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 recovery learner 读取训练来源前不可覆盖发布评测协议。"""
    root = _require_k_root(run_root)
    source = Path(source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_dir() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization recovery evaluation path 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery evaluation target 已存在")
    source_manifest, _, pair_records = (
        read_normalization_recovery_source_pack(source))
    evaluation, reserve, summary = (
        derive_normalization_recovery_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            pair_records=pair_records,
        ))
    target.mkdir(parents=True)
    evaluation_path = target / "evaluation.inventory.jsonl"
    reserve_path = target / "reserve.identity.jsonl"
    _write_jsonl(evaluation_path, evaluation)
    _write_jsonl(reserve_path, reserve)
    manifest = _manifest(
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        evaluation_artifact=_artifact(
            evaluation_path, role="EVALUATION_WITH_LABELS",
            count=len(evaluation)),
        reserve_artifact=_artifact(
            reserve_path, role="RESERVE_IDENTITY_WITHOUT_LABELS",
            count=len(reserve)),
        inventory_summary=summary,
    )
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_recovery_evaluation_protocol(
        protocol_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从冻结 source pack 重派生 split/label，并严格回读完整协议。"""
    root = Path(protocol_dir).resolve()
    source_manifest, _, pair_records = (
        read_normalization_recovery_source_pack(source_pack_dir))
    try:
        encoded_manifest = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery evaluation manifest 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization recovery evaluation manifest 非规范")
    derived_evaluation, derived_reserve, summary = (
        derive_normalization_recovery_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            pair_records=pair_records,
        ))
    stored_evaluation = _read_jsonl(
        root / "evaluation.inventory.jsonl", label="recovery evaluation")
    stored_reserve = _read_jsonl(
        root / "reserve.identity.jsonl", label="recovery reserve")
    if (not _strict_equal(stored_evaluation, derived_evaluation)
            or not _strict_equal(stored_reserve, derived_reserve)):
        raise BroadQaExternalDataError(
            "normalization recovery inventory/source 漂移")
    expected = _manifest(
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        evaluation_artifact=_artifact(
            root / "evaluation.inventory.jsonl",
            role="EVALUATION_WITH_LABELS", count=len(derived_evaluation)),
        reserve_artifact=_artifact(
            root / "reserve.identity.jsonl",
            role="RESERVE_IDENTITY_WITHOUT_LABELS", count=len(derived_reserve)),
        inventory_summary=summary,
    )
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "normalization recovery evaluation manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_evaluation,
        derived_reserve,
    )


def read_normalization_recovery_evaluation_manifest_only(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只打开 manifest 并核验冻结 identity，绝不读取 evaluation/reserve。"""
    root = Path(protocol_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery evaluation manifest 不可读") from error
    expected_sha = _sha_value(
        expected_manifest_sha256,
        label="recovery evaluation expected manifest")
    if (_sha256(encoded) != expected_sha or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "recovery evaluation manifest identity/encoding 漂移")
    evaluation_artifact = _validate_artifact_identity(
        stored.get("evaluation_inventory"),
        relative_path="evaluation.inventory.jsonl",
        role="EVALUATION_WITH_LABELS")
    reserve_artifact = _validate_artifact_identity(
        stored.get("reserve_identity"),
        relative_path="reserve.identity.jsonl",
        role="RESERVE_IDENTITY_WITHOUT_LABELS")
    summary = stored.get("inventory_summary")
    source_sha = _sha_value(
        stored.get("source_pack_manifest_sha256"),
        label="recovery source manifest")
    if (not isinstance(summary, dict)
            or summary.get("evaluation_count")
            != evaluation_artifact["record_count"]
            or summary.get("reserve_count")
            != reserve_artifact["record_count"]):
        raise BroadQaExternalDataError(
            "recovery evaluation manifest inventory summary 漂移")
    expected = _manifest(
        source_pack_manifest_sha256=source_sha,
        evaluation_artifact=evaluation_artifact,
        reserve_artifact=reserve_artifact,
        inventory_summary=summary,
    )
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "recovery evaluation manifest 冻结字段漂移")
    return {**stored, "manifest_sha256": expected_sha}


def read_normalization_recovery_evaluation_inventory_only(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """只按冻结 manifest 打开 evaluation，不读 source/reserve payload。"""
    root = Path(protocol_dir).resolve()
    manifest = read_normalization_recovery_evaluation_manifest_only(
        root, expected_manifest_sha256=expected_manifest_sha256)
    stored = _read_jsonl(
        root / "evaluation.inventory.jsonl", label="recovery evaluation")
    artifact = _artifact(
        root / "evaluation.inventory.jsonl",
        role="EVALUATION_WITH_LABELS", count=len(stored))
    if (not _strict_equal(artifact, manifest["evaluation_inventory"])
            or len(stored) != manifest["inventory_summary"]["evaluation_count"]):
        raise BroadQaExternalDataError(
            "recovery evaluation-only inventory/manifest 漂移")
    return manifest, stored


__all__ = [
    "NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS",
    "NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT",
    "NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND",
    "NORMALIZATION_RECOVERY_EVALUATION_STATUS",
    "NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE",
    "derive_normalization_recovery_evaluation_inventory",
    "normalization_recovery_evaluation_split",
    "publish_normalization_recovery_evaluation_protocol",
    "read_normalization_recovery_evaluation_inventory_only",
    "read_normalization_recovery_evaluation_manifest_only",
    "read_normalization_recovery_evaluation_protocol",
]
