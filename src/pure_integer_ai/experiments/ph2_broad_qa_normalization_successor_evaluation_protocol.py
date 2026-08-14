"""冻结 normalization successor 的独立 evaluation family 与六维门。

本模块只消费已经冻结的 Unihan/MediaWiki source pack。split、label、metric
与阈值均在 successor learner 读取 OpenCC/ICU 全量来源前发布。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_source_pack import (
    MEDIAWIKI_COMMIT,
    UNIHAN_VERSION,
    read_normalization_successor_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_V1")
NORMALIZATION_SUCCESSOR_EVALUATION_STATUS = (
    "FROZEN_BEFORE_SUCCESSOR_LEARNING_NOT_EVALUATED")
NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND = (
    "NORMALIZATION_SUCCESSOR_EVALUATION_ITEM_V1")
NORMALIZATION_SUCCESSOR_RESERVE_RECORD_KIND = (
    "NORMALIZATION_SUCCESSOR_RESERVE_IDENTITY_V1")
NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE = (
    "ZH_HANS_CROSS_SOURCE_CONSENSUS_V1")
NORMALIZATION_SUCCESSOR_SPLIT_SEED = (
    "NORMALIZATION_SUCCESSOR_EVALUATION_SPLIT_V1")

NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS = {
    "DEFEATER_REPRESENTATION_EXECUTABILITY": {
        "bearing": 1,
        "declared_defeater_count_min": 1,
        "declared_must_equal_executable": 1,
        "identity_only_defeater_count_max": 0,
        "malformed_defeater_count_max": 0,
        "missing_candidate_outcome": "NE",
    },
    "END_TO_END_COVERAGE": {
        "bearing": 1,
        "equal_length_phrase_inventory_required": 1,
        "evaluation_phrase_count_min": 128,
        "full_output_match_count_min": 128,
        "output_length_mismatch_count_max": 0,
        "partial_coverage_is_false_accept": 0,
        "uncovered_position_report_required": 1,
        "wrong_changed_position_count_max": 0,
    },
    "INDEPENDENT_CONTEXT_TRANSFER": {
        "applicable_context_count_min": 16,
        "bearing": 1,
        "context_exact_support_count_min": 16,
        "context_record_inventory_min": 24,
        "context_wrong_output_count_max": 0,
        "missing_applicable_context_outcome": "NE",
    },
    "LOCAL_MAPPING_TRANSFER": {
        "applicable_mapping_count_min": 256,
        "bearing": 1,
        "conflicting_mapping_count_max": 0,
        "missing_applicable_mapping_outcome": "NE",
        "scope_mismatch_execution_count_max": 0,
        "supported_must_equal_applicable": 1,
        "unscoped_rule_count_max": 0,
    },
    "RUNTIME_PRODUCTION_BEHAVIOR": {
        "bearing": 1,
        "canonical_replay_mismatch_count_max": 0,
        "evaluation_input_execution_required": 1,
        "exception_count_max": 0,
        "production_enabled_must_equal": 0,
        "target_policy_scope_required": 1,
    },
    "SOURCE_POLICY_CONFLICT": {
        "bearing": 1,
        "declared_conflict_must_equal_observed": 1,
        "missing_conflict_facility_outcome": "NE",
        "observed_training_source_conflict_count_min": 1,
        "policy_specific_replay_must_equal_observation": 1,
        "training_source_policy_count_min": 2,
        "unscoped_conflict_execution_count_max": 0,
    },
}

NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT = {
    "bearing_dimension_count": 6,
    "coverage_alignment": (
        "equal Unicode-scalar length; position-wise INPUT/EXPECTED/CANDIDATE"),
    "evaluation_run_count_max": 1,
    "failed_icu_v2_records_are_formal_denominator": 0,
    "local_mapping_unit": "one unambiguous Unihan kSimplifiedVariant edge",
    "mastery_claimed_before_all_bearing_pass": 0,
    "overall_rule": "FAIL_DOMINATES_NE_DOMINATES_PASS",
    "phrase_partial_output_is_local_false_accept": 0,
    "production_enablement_during_evaluation": 0,
    "reserve_label_read_count_max": 0,
    "source_behavior_scope_is_target_policy_scope": 0,
    "source_policy_conflict_inventory": (
        "successor TRAIN OpenCC/ICU source replay, not evaluation labels"),
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
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _record_identity(*values: str) -> str:
    """从来源身份和原始内容构造稳定 evaluation identity。"""
    if any(not isinstance(value, str) or not value for value in values):
        raise BroadQaExternalDataError("successor evaluation identity 输入非法")
    return _sha256("\0".join(values).encode("utf-8"))


def normalization_successor_evaluation_split(evaluation_id: str) -> str:
    """仅按冻结 seed 与 evaluation identity 产生 80/20 split。"""
    identity = _sha_value(evaluation_id, label="successor evaluation id")
    digest = hashlib.sha256(
        f"{NORMALIZATION_SUCCESSOR_SPLIT_SEED}\0{identity}".encode("ascii")
    ).digest()
    return "EVALUATION" if digest[0] % 5 < 4 else "RESERVE"


def _source_record_id(source_key: str, record: dict[str, object]) -> str:
    """把来源、表/属性和物理行绑定为不泄漏 label 的身份。"""
    qualifier = str(record.get("property_name", record.get("table_name", "")))
    return _record_identity(
        source_key, qualifier, str(record["line_ordinal"]),
        str(record["line_sha256"]))


def _unihan_items(
        *,
        source_pack_manifest_sha256: str,
        records: tuple[dict[str, object], ...],
        ) -> list[dict[str, object]]:
    """从唯一非恒等 kSimplifiedVariant 生成局部 transfer items。"""
    result = []
    seen_inputs = set()
    for record in records:
        if record.get("t2s_unambiguous_eligible") != 1:
            continue
        input_text = record.get("t2s_input")
        expected_output = record.get("t2s_expected_output")
        if (not isinstance(input_text, str) or len(input_text) != 1
                or not isinstance(expected_output, str)
                or len(expected_output) != 1
                or input_text == expected_output or input_text in seen_inputs):
            raise BroadQaExternalDataError(
                "successor Unihan eligible record 漂移")
        seen_inputs.add(input_text)
        source_record_id = _source_record_id("UNIHAN", record)
        evaluation_id = _record_identity(
            "UNIHAN_LOCAL_MAPPING", source_pack_manifest_sha256,
            source_record_id, input_text, expected_output)
        result.append({
            "evaluation_id": evaluation_id,
            "expected_output": expected_output,
            "family_keys": ["LOCAL_MAPPING_TRANSFER"],
            "format_version": 1,
            "input_text": input_text,
            "record_kind": NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
            "source_key": "UNICODE_UNIHAN",
            "source_license_id": "Unicode-3.0",
            "source_line_ordinal": record["line_ordinal"],
            "source_line_sha256": record["line_sha256"],
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "source_property": "kSimplifiedVariant",
            "source_record_id": source_record_id,
            "source_revision": UNIHAN_VERSION,
            "split": normalization_successor_evaluation_split(evaluation_id),
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
    return result


def _mediawiki_phrase_records(
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """选择 HANS 等长非恒等短语，确保逐位置覆盖账可判定。"""
    values = []
    for record in records:
        if (record.get("table_name") != "ZH_TO_HANS"
                or record.get("is_identity") != 0
                or type(record.get("input_scalar_count")) is not int
                or type(record.get("output_scalar_count")) is not int
                or record["input_scalar_count"] < 2
                or record["input_scalar_count"]
                != record["output_scalar_count"]):
            continue
        input_text = record.get("input_text")
        expected_output = record.get("expected_output")
        if (not isinstance(input_text, str)
                or not isinstance(expected_output, str)
                or len(input_text) != record["input_scalar_count"]
                or len(expected_output) != record["output_scalar_count"]):
            raise BroadQaExternalDataError(
                "successor MediaWiki phrase record 漂移")
        values.append(record)
    if len({item["input_text"] for item in values}) != len(values):
        raise BroadQaExternalDataError(
            "successor MediaWiki phrase input 重复")
    return tuple(values)


def _mediawiki_items(
        *,
        source_pack_manifest_sha256: str,
        records: tuple[dict[str, object], ...],
        ) -> list[dict[str, object]]:
    """生成等长 phrase coverage 与机械 context-sensitive items。"""
    phrases = _mediawiki_phrase_records(records)
    outputs_by_input: dict[str, set[str]] = defaultdict(set)
    for record in phrases:
        for input_item, output_item in zip(
                str(record["input_text"]), str(record["expected_output"])):
            outputs_by_input[input_item].add(output_item)
    context_inputs = {
        input_item for input_item, outputs in outputs_by_input.items()
        if len(outputs) > 1
    }
    result = []
    for record in phrases:
        input_text = str(record["input_text"])
        expected_output = str(record["expected_output"])
        positions = [{
            "expected_output": output_item,
            "input_text": input_item,
            "scalar_offset": offset,
        } for offset, (input_item, output_item) in enumerate(
            zip(input_text, expected_output)) if input_item != output_item]
        if not positions:
            raise BroadQaExternalDataError(
                "successor MediaWiki phrase 无变化位置")
        context_sensitive = int(any(
            item["input_text"] in context_inputs for item in positions))
        families = ["END_TO_END_COVERAGE"]
        if context_sensitive:
            families.append("INDEPENDENT_CONTEXT_TRANSFER")
        source_record_id = _source_record_id("MEDIAWIKI", record)
        evaluation_id = _record_identity(
            "MEDIAWIKI_PHRASE", source_pack_manifest_sha256,
            source_record_id, input_text, expected_output)
        result.append({
            "context_sensitive": context_sensitive,
            "evaluation_id": evaluation_id,
            "expected_output": expected_output,
            "family_keys": families,
            "format_version": 1,
            "input_text": input_text,
            "position_expectations": positions,
            "record_kind": NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
            "source_key": "MEDIAWIKI_CORE",
            "source_license_id": "GPL-2.0-or-later",
            "source_line_ordinal": record["line_ordinal"],
            "source_line_sha256": record["line_sha256"],
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "source_record_id": source_record_id,
            "source_revision": MEDIAWIKI_COMMIT,
            "source_table": "ZH_TO_HANS",
            "split": normalization_successor_evaluation_split(evaluation_id),
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
    return result


def derive_normalization_successor_evaluation_inventory(
        *,
        source_pack_manifest_sha256: str,
        unihan_records: tuple[dict[str, object], ...],
        mediawiki_records: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从独立 source pack 确定性派生 evaluation 与无 label reserve。"""
    source_sha = _sha_value(
        source_pack_manifest_sha256, label="successor source pack manifest")
    if (not isinstance(unihan_records, tuple) or not unihan_records
            or not isinstance(mediawiki_records, tuple)
            or not mediawiki_records):
        raise BroadQaExternalDataError(
            "successor evaluation source inventory 为空")
    full = _unihan_items(
        source_pack_manifest_sha256=source_sha, records=unihan_records)
    full.extend(_mediawiki_items(
        source_pack_manifest_sha256=source_sha, records=mediawiki_records))
    identities = [item["evaluation_id"] for item in full]
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError(
            "successor evaluation identity 重复")
    evaluation = tuple(sorted(
        (item for item in full if item["split"] == "EVALUATION"),
        key=lambda item: str(item["evaluation_id"])))
    reserve_source = tuple(sorted(
        (item for item in full if item["split"] == "RESERVE"),
        key=lambda item: str(item["evaluation_id"])))
    reserve = tuple({
        "evaluation_id": item["evaluation_id"],
        "format_version": 1,
        "record_kind": NORMALIZATION_SUCCESSOR_RESERVE_RECORD_KIND,
        "source_record_id": item["source_record_id"],
        "split": "RESERVE",
    } for item in reserve_source)
    evaluation_sources = Counter(item["source_key"] for item in evaluation)
    reserve_sources = Counter(item["source_key"] for item in reserve_source)
    summary = {
        "context_evaluation_count": sum(
            item.get("context_sensitive") == 1 for item in evaluation),
        "context_reserve_count": sum(
            item.get("context_sensitive") == 1 for item in reserve_source),
        "evaluation_count": len(evaluation),
        "evaluation_source_counts": dict(sorted(evaluation_sources.items())),
        "full_inventory_count": len(full),
        "local_mapping_evaluation_count": sum(
            "LOCAL_MAPPING_TRANSFER" in item["family_keys"]
            for item in evaluation),
        "phrase_evaluation_count": sum(
            "END_TO_END_COVERAGE" in item["family_keys"]
            for item in evaluation),
        "reserve_count": len(reserve),
        "reserve_source_counts": dict(sorted(reserve_sources.items())),
        "split_overlap_count": len(
            {item["evaluation_id"] for item in evaluation}.intersection(
                item["evaluation_id"] for item in reserve)),
    }
    if (not evaluation or not reserve
            or summary["split_overlap_count"] != 0
            or summary["local_mapping_evaluation_count"]
            < NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
                "LOCAL_MAPPING_TRANSFER"]["applicable_mapping_count_min"]
            or summary["phrase_evaluation_count"]
            < NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
                "END_TO_END_COVERAGE"]["evaluation_phrase_count_min"]
            or summary["context_evaluation_count"]
            < NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS[
                "INDEPENDENT_CONTEXT_TRANSFER"][
                    "context_record_inventory_min"]):
        raise BroadQaExternalDataError(
            "successor evaluation family 库存不足")
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
    """构造协议物理文件承诺。"""
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
    """核验 manifest 内一个未打开 payload 的冻结文件身份。"""
    expected_fields = {
        "bytes", "record_count", "relative_path", "role", "sha256"}
    if (not isinstance(value, dict) or set(value) != expected_fields
            or type(value["bytes"]) is not int or value["bytes"] <= 0
            or type(value["record_count"]) is not int
            or value["record_count"] <= 0
            or value["relative_path"] != relative_path
            or value["role"] != role):
        raise BroadQaExternalDataError(
            f"successor evaluation {relative_path} identity 漂移")
    _sha_value(value["sha256"], label=f"successor {relative_path} SHA")
    return value


def _manifest(
        *,
        source_pack_manifest_sha256: str,
        evaluation_artifact: dict[str, object],
        reserve_artifact: dict[str, object],
        inventory_summary: dict[str, object],
        ) -> dict[str, object]:
    """构造冻结在学习前的 evaluation protocol manifest。"""
    return {
        "artifact_kind": NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_KIND,
        "candidate_pack_read_count": 0,
        "dimensions": NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS,
        "evaluation_inventory": evaluation_artifact,
        "evaluation_run_count": 0,
        "failed_icu_evaluation_read_count": 0,
        "format_version": 1,
        "inventory_summary": inventory_summary,
        "learned_pack_read_count": 0,
        "mastery_claimed": 0,
        "metric_contract": NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT,
        "production_enabled": 0,
        "reserve_identity": reserve_artifact,
        "reserve_label_read_count": 0,
        "selection_rule": {
            "mediawiki": (
                "ZH_TO_HANS; non-identity; input scalar count >= 2; "
                "input/output scalar counts equal"),
            "split": (
                "sha256(split_seed + NUL + evaluation_id)[0] mod 5; "
                "0..3=EVALUATION, 4=RESERVE"),
            "unihan": (
                "kSimplifiedVariant; exactly one unsourced target; non-identity"),
        },
        "source_pack_manifest_sha256": source_pack_manifest_sha256,
        "status": NORMALIZATION_SUCCESSOR_EVALUATION_STATUS,
        "successor_training_source_read_count": 0,
        "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization successor evaluation run root 必须是 K 盘目录")
    return root


def publish_normalization_successor_evaluation_protocol(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 successor learner 读取训练来源前不可覆盖发布评测协议。"""
    root = _require_k_root(run_root)
    source = Path(source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not source.is_dir() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization successor evaluation path 越出 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization successor evaluation target 已存在")
    source_manifest, unihan_records, mediawiki_records = (
        read_normalization_successor_source_pack(source))
    evaluation, reserve, summary = (
        derive_normalization_successor_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            unihan_records=unihan_records,
            mediawiki_records=mediawiki_records,
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


def read_normalization_successor_evaluation_protocol(
        protocol_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """从 source pack 重派生并严格回读 evaluation 与 reserve。"""
    root = Path(protocol_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        encoded_manifest = manifest_path.read_bytes()
        stored = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization successor evaluation manifest 不可读") from error
    if (not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded_manifest):
        raise BroadQaExternalDataError(
            "normalization successor evaluation manifest 非规范")
    source_manifest, unihan_records, mediawiki_records = (
        read_normalization_successor_source_pack(source_pack_dir))
    derived_evaluation, derived_reserve, summary = (
        derive_normalization_successor_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            unihan_records=unihan_records,
            mediawiki_records=mediawiki_records,
        ))
    stored_evaluation = _read_jsonl(
        root / "evaluation.inventory.jsonl", label="successor evaluation")
    stored_reserve = _read_jsonl(
        root / "reserve.identity.jsonl", label="successor reserve")
    if (not _strict_equal(stored_evaluation, derived_evaluation)
            or not _strict_equal(stored_reserve, derived_reserve)):
        raise BroadQaExternalDataError(
            "normalization successor inventory/source 漂移")
    evaluation_artifact = _artifact(
        root / "evaluation.inventory.jsonl",
        role="EVALUATION_WITH_LABELS", count=len(derived_evaluation))
    reserve_artifact = _artifact(
        root / "reserve.identity.jsonl",
        role="RESERVE_IDENTITY_WITHOUT_LABELS", count=len(derived_reserve))
    expected_manifest = _manifest(
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        evaluation_artifact=evaluation_artifact,
        reserve_artifact=reserve_artifact,
        inventory_summary=summary,
    )
    if not _strict_equal(stored, expected_manifest):
        raise BroadQaExternalDataError(
            "normalization successor evaluation manifest 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded_manifest)},
        derived_evaluation,
        derived_reserve,
    )


def read_normalization_successor_evaluation_manifest_only(
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
            "normalization successor evaluation manifest 不可读") from error
    expected_sha = _sha_value(
        expected_manifest_sha256,
        label="successor evaluation expected manifest")
    if (_sha256(encoded) != expected_sha
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "successor evaluation manifest identity/encoding 漂移")
    evaluation_artifact = _validate_artifact_identity(
        stored.get("evaluation_inventory"),
        relative_path="evaluation.inventory.jsonl",
        role="EVALUATION_WITH_LABELS",
    )
    reserve_artifact = _validate_artifact_identity(
        stored.get("reserve_identity"),
        relative_path="reserve.identity.jsonl",
        role="RESERVE_IDENTITY_WITHOUT_LABELS",
    )
    summary = stored.get("inventory_summary")
    source_sha = _sha_value(
        stored.get("source_pack_manifest_sha256"),
        label="successor evaluation source manifest")
    if (not isinstance(summary, dict)
            or summary.get("evaluation_count")
            != evaluation_artifact["record_count"]
            or summary.get("reserve_count")
            != reserve_artifact["record_count"]):
        raise BroadQaExternalDataError(
            "successor evaluation manifest inventory summary 漂移")
    expected = _manifest(
        source_pack_manifest_sha256=source_sha,
        evaluation_artifact=evaluation_artifact,
        reserve_artifact=reserve_artifact,
        inventory_summary=summary,
    )
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError(
            "successor evaluation manifest 冻结字段漂移")
    return {**stored, "manifest_sha256": expected_sha}


def read_normalization_successor_evaluation_inventory_only(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """只按冻结 manifest 打开正式 evaluation，不读 source/reserve payload。"""
    root = Path(protocol_dir).resolve()
    manifest = read_normalization_successor_evaluation_manifest_only(
        root, expected_manifest_sha256=expected_manifest_sha256)
    stored_evaluation = _read_jsonl(
        root / "evaluation.inventory.jsonl", label="successor evaluation")
    artifact = _artifact(
        root / "evaluation.inventory.jsonl",
        role="EVALUATION_WITH_LABELS", count=len(stored_evaluation))
    if (not _strict_equal(artifact, manifest["evaluation_inventory"])
            or len(stored_evaluation)
            != manifest["inventory_summary"]["evaluation_count"]):
        raise BroadQaExternalDataError(
            "successor evaluation-only inventory/manifest 漂移")
    return manifest, stored_evaluation


__all__ = [
    "NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS",
    "NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT",
    "NORMALIZATION_SUCCESSOR_EVALUATION_PROTOCOL_KIND",
    "NORMALIZATION_SUCCESSOR_EVALUATION_STATUS",
    "NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE",
    "derive_normalization_successor_evaluation_inventory",
    "normalization_successor_evaluation_split",
    "publish_normalization_successor_evaluation_protocol",
    "read_normalization_successor_evaluation_inventory_only",
    "read_normalization_successor_evaluation_manifest_only",
    "read_normalization_successor_evaluation_protocol",
]
