"""冻结未消费的 ICU reverse/T2S normalization evaluation family。

协议只从独立 ICU source pack 派生 inventory 和 reserve。它不读取 learned pack，
不运行 evaluator，并把直接映射、defeater、context application 与 runtime 分账。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    NORMALIZATION_ICU_SOURCE_PACK_KIND,
    NORMALIZATION_ICU_SOURCE_STATUS,
    read_normalization_icu_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_EVALUATION_PROTOCOL_V2")
NORMALIZATION_ICU_EVALUATION_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_T2S_EVALUATION_RECORD_V1")
NORMALIZATION_ICU_RESERVE_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_T2S_RESERVE_IDENTITY_V1")
NORMALIZATION_ICU_EVALUATION_STATUS = "FROZEN_UNCONSUMED_NOT_RUN_V2"
NORMALIZATION_ICU_EVALUATION_SEED = (
    "NORMALIZATION_ICU_T2S_REVERSE_EVALUATION_V1")
NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT = {
    "APPLICABLE_INDEPENDENT_RULE_COUNT": (
        "COUNT_EVALUATION_RECORDS_WITH_ACCEPTED_INPUT_OCCURRENCE"),
    "INDEPENDENT_SUPPORT_COUNT": (
        "COUNT_APPLICABLE_RECORDS_WHOSE_EXPECTED_OUTPUT_EQUALS_"
        "DEFEATER_AWARE_CANDIDATE_REWRITE"),
    "UNRESOLVED_INDEPENDENT_CONFLICT_COUNT": (
        "COUNT_APPLICABLE_RECORDS_WHOSE_EXPECTED_OUTPUT_DIFFERS_AND_HAS_NO_"
        "EXECUTABLE_MATCHING_DEFEATER"),
    "INDEPENDENT_FALSE_ACCEPT_COUNT": (
        "COUNT_APPLICABLE_RECORDS_CHANGED_BY_CLONE_TO_NON_EXPECTED_OUTPUT"),
    "INDEPENDENT_FALSE_REJECT_COUNT": (
        "COUNT_SUPPORTED_RECORDS_NOT_EMITTED_AS_EXPECTED_OUTPUT_BY_CLONE"),
    "DECLARED_DEFEATER_COUNT": (
        "COUNT_UNIQUE_DEFEATERS_REFERENCED_BY_ACCEPTED_RULES"),
    "EXECUTABLE_DEFEATER_COUNT": (
        "COUNT_DECLARED_DEFEATERS_PARSED_AS_NON_IDENTITY_CONTEXT_PREDICATES"),
    "POSITIVE_CONTEXT_CASE_COUNT": (
        "COUNT_ACCEPTED_RULE_SUPPORT_EVIDENCE_CASES_REPLAYED_BY_CLONE"),
    "NEGATIVE_CONTEXT_CASE_COUNT": (
        "COUNT_REJECTED_TRIAL_CONTEXT_CASES_REPLAYED_BY_CLONE"),
    "DEFEATER_HIT_COUNT": (
        "COUNT_NEGATIVE_CONTEXT_CASES_WITH_EXPECTED_DEFEATER_ID_HIT"),
    "CONTEXT_FALSE_ACCEPT_COUNT": (
        "COUNT_NEGATIVE_CONTEXT_CASES_WHERE_BASE_REWRITE_FIRES"),
    "CONTEXT_FALSE_REJECT_COUNT": (
        "COUNT_POSITIVE_CONTEXT_CASES_WHERE_EXPECTED_REWRITE_DOES_NOT_FIRE"),
    "RUNTIME_EXECUTED_INPUT_COUNT": (
        "COUNT_EVALUATION_INPUTS_EXECUTED_BY_CANDIDATE_CLONE"),
    "RUNTIME_EXCEPTION_COUNT": (
        "COUNT_CANDIDATE_CLONE_EXECUTIONS_ENDING_IN_EXCEPTION"),
    "RUNTIME_REPLAY_MISMATCH_COUNT": (
        "COUNT_IDENTICAL_INPUT_REPLAYS_WITH_DIFFERENT_CANONICAL_OUTPUT"),
}
NORMALIZATION_ICU_EVALUATION_DIMENSIONS = {
    "DIRECT_MAPPING_CONSISTENCY": {
        "bearing": 1,
        "applicable_independent_rule_count_min": 1,
        "independent_false_accept_count_max": 0,
        "independent_false_reject_count_max": 0,
        "independent_support_count_min": 1,
        "no_applicable_rule_outcome": "NE",
        "unresolved_independent_conflict_count_max": 0,
    },
    "DEFEATER_REPRESENTATION_EXECUTABILITY": {
        "bearing": 1,
        "declared_defeater_count_min": 1,
        "executable_defeater_count_must_equal_declared_count": 1,
        "identity_only_defeater_outcome": "FAIL",
        "malformed_defeater_count_max": 0,
        "no_declared_defeater_outcome": "NE",
    },
    "CONTEXT_APPLICATION_OUTCOME": {
        "bearing": 1,
        "consumer_execution_required": 1,
        "context_false_accept_count_max": 0,
        "context_false_reject_count_max": 0,
        "defeater_hit_count_must_equal_negative_context_case_count": 1,
        "missing_consumer_outcome": "NE",
        "negative_context_case_count_min": 1,
        "positive_context_case_count_min": 1,
    },
    "RUNTIME_PRODUCTION_BEHAVIOR": {
        "all_evaluation_inputs_must_execute": 1,
        "bearing": 1,
        "candidate_clone_consumer_required": 1,
        "missing_candidate_clone_outcome": "NE",
        "public_production_gate_must_remain_disabled": 1,
        "runtime_exception_count_max": 0,
        "runtime_replay_mismatch_count_max": 0,
    },
}
NORMALIZATION_ICU_EVALUATION_RUN_CONTRACT = {
    "formal_run_count_max": 1,
    "learned_pack_read_before_protocol_freeze_allowed": 0,
    "overwrite_allowed": 0,
    "production_enable_allowed": 0,
    "reserve_label_read_allowed": 0,
    "teacher_api_llm_calls_allowed": 0,
}


def _sha256(payload: bytes) -> str:
    """返回规范记录或文件摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 合同并区分 bool 与 int。"""
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


def normalization_icu_evaluation_split(record_sha256: str) -> str:
    """按来源 record SHA 固定四比一 evaluation/reserve 分割。"""
    if (not isinstance(record_sha256, str) or len(record_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in record_sha256)):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation record SHA 非法")
    digest = hashlib.sha256((
        NORMALIZATION_ICU_EVALUATION_SEED + "\0" + record_sha256
    ).encode("utf-8")).digest()
    return "RESERVE" if int.from_bytes(digest, "big") % 5 == 4 else "EVALUATION"


def derive_normalization_icu_evaluation_inventory(
        *,
        source_pack_manifest_sha256: str,
        rules: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """从全部 reverse-eligible ICU rules 派生 inventory，不读 learned pack。"""
    if (not isinstance(source_pack_manifest_sha256, str)
            or len(source_pack_manifest_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in source_pack_manifest_sha256)
            or not isinstance(rules, tuple)):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation source identity 非法")
    evaluation = []
    reserve = []
    seen_inputs = set()
    for rule in rules:
        if rule.get("t2s_reverse_eligible") != 1:
            continue
        source_rule_sha = rule.get("statement_sha256")
        input_text = rule.get("t2s_input")
        expected = rule.get("t2s_expected_output")
        if (not isinstance(source_rule_sha, str)
                or not isinstance(input_text, str) or not input_text
                or not isinstance(expected, str) or not expected
                or input_text in seen_inputs
                or rule.get("has_context") != 0
                or rule.get("arrow") not in ("←", "↔")):
            raise BroadQaExternalDataError(
                "ICU normalization evaluation eligible rule 漂移")
        seen_inputs.add(input_text)
        split = normalization_icu_evaluation_split(source_rule_sha)
        mapping_kind = (
            "IDENTITY" if input_text == expected else
            "CHARACTER" if len(input_text) == 1 and len(expected) == 1 else
            "PHRASE")
        core = {
            "expected_output": expected,
            "input_text": input_text,
            "mapping_kind": mapping_kind,
            "source_arrow": rule["arrow"],
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "source_rule_byte_end": rule["byte_end"],
            "source_rule_byte_start": rule["byte_start"],
            "source_rule_line_end_ordinal": rule["line_end_ordinal"],
            "source_rule_line_start_ordinal": rule["line_start_ordinal"],
            "source_rule_sha256": source_rule_sha,
        }
        evaluation_id = _sha256(canonical_json_bytes(core))
        if split == "EVALUATION":
            evaluation.append({
                **core,
                "evaluation_id": evaluation_id,
                "format_version": 1,
                "record_kind": NORMALIZATION_ICU_EVALUATION_RECORD_KIND,
                "split": split,
            })
        else:
            reserve.append({
                "evaluation_id": evaluation_id,
                "format_version": 1,
                "record_kind": NORMALIZATION_ICU_RESERVE_RECORD_KIND,
                "source_rule_sha256": source_rule_sha,
                "split": split,
            })
    evaluation.sort(key=lambda value: value["evaluation_id"])
    reserve.sort(key=lambda value: value["evaluation_id"])
    all_ids = [value["evaluation_id"] for value in evaluation + reserve]
    if not evaluation or not reserve or len(set(all_ids)) != len(all_ids):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation split 为空或 identity 重复")
    counts = Counter(value["mapping_kind"] for value in evaluation)
    summary = {
        "evaluation_mapping_kind_counts": {
            kind: counts[kind] for kind in ("CHARACTER", "IDENTITY", "PHRASE")
        },
        "evaluation_record_count": len(evaluation),
        "reserve_identity_count": len(reserve),
        "source_reverse_eligible_count": len(evaluation) + len(reserve),
    }
    return tuple(evaluation), tuple(reserve), summary


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    records = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        "ICU normalization evaluation JSONL 非规范")
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "ICU normalization evaluation JSONL 不可读") from error
    return tuple(records)


def publish_normalization_icu_evaluation_protocol(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 learned pack 读取前不可覆盖冻结 evaluation family。"""
    root = Path(run_root).resolve()
    source = Path(source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not root.is_dir() or not source.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation 路径必须位于有效 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "ICU normalization evaluation target 已存在")
    source_manifest, _, rules = read_normalization_icu_source_pack(source)
    if (source_manifest["artifact_kind"] != NORMALIZATION_ICU_SOURCE_PACK_KIND
            or source_manifest["status"] != NORMALIZATION_ICU_SOURCE_STATUS
            or source_manifest["learned_pack_read_count"] != 0
            or source_manifest["evaluation_run_count"] != 0):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation source 已消费或漂移")
    evaluation, reserve, summary = (
        derive_normalization_icu_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            rules=rules,
        ))
    target.mkdir(parents=True)
    evaluation_path = target / "evaluation.inventory.jsonl"
    reserve_path = target / "reserve.identity.jsonl"
    _write_jsonl(evaluation_path, evaluation)
    _write_jsonl(reserve_path, reserve)
    manifest = {
        "artifact_kind": NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND,
        "dimensions": NORMALIZATION_ICU_EVALUATION_DIMENSIONS,
        "evaluation_inventory": {
            "bytes": evaluation_path.stat().st_size,
            "record_count": len(evaluation),
            "relative_path": evaluation_path.name,
            "sha256": _sha256(evaluation_path.read_bytes()),
        },
        "evaluation_run_count": 0,
        "format_version": 2,
        "learned_pack_read_count": 0,
        "mastery_claimed": 0,
        "metric_contract": NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT,
        "overall_pass_contract": (
            "PASS_ONLY_IF_ALL_BEARING_DIMENSIONS_PASS_OTHERWISE_FAIL_OR_NE"),
        "production_enabled": 0,
        "reserve_identity": {
            "bytes": reserve_path.stat().st_size,
            "record_count": len(reserve),
            "relative_path": reserve_path.name,
            "sha256": _sha256(reserve_path.read_bytes()),
        },
        "reserve_labels_published": 0,
        "run_contract": NORMALIZATION_ICU_EVALUATION_RUN_CONTRACT,
        "selection_rule": (
            "SHA256_SEED_AND_SOURCE_RULE_SHA_MOD_5_RESERVE_BUCKET_4_V1"),
        "source_pack_manifest_sha256": source_manifest["manifest_sha256"],
        "status": NORMALIZATION_ICU_EVALUATION_STATUS,
        "summary": summary,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_normalization_icu_evaluation_protocol(
        target_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """严格回读协议并从独立 ICU source 重派生完整 split。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "ICU normalization evaluation manifest 不可读") from error
    expected = {
        "artifact_kind", "dimensions", "evaluation_inventory",
        "evaluation_run_count", "format_version", "learned_pack_read_count",
        "mastery_claimed", "metric_contract", "overall_pass_contract",
        "production_enabled", "reserve_identity", "reserve_labels_published",
        "run_contract", "selection_rule", "source_pack_manifest_sha256",
        "status", "summary",
    }
    fixed = {
        "artifact_kind": NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND,
        "dimensions": NORMALIZATION_ICU_EVALUATION_DIMENSIONS,
        "evaluation_run_count": 0,
        "format_version": 2,
        "learned_pack_read_count": 0,
        "mastery_claimed": 0,
        "metric_contract": NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT,
        "overall_pass_contract": (
            "PASS_ONLY_IF_ALL_BEARING_DIMENSIONS_PASS_OTHERWISE_FAIL_OR_NE"),
        "production_enabled": 0,
        "reserve_labels_published": 0,
        "run_contract": NORMALIZATION_ICU_EVALUATION_RUN_CONTRACT,
        "selection_rule": (
            "SHA256_SEED_AND_SOURCE_RULE_SHA_MOD_5_RESERVE_BUCKET_4_V1"),
        "status": NORMALIZATION_ICU_EVALUATION_STATUS,
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != payload
            or any(not _strict_equal(manifest[key], value)
                   for key, value in fixed.items())):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation manifest 漂移")
    source_manifest, _, rules = read_normalization_icu_source_pack(
        source_pack_dir)
    if source_manifest["manifest_sha256"] != manifest[
            "source_pack_manifest_sha256"]:
        raise BroadQaExternalDataError(
            "ICU normalization evaluation source manifest 漂移")
    expected_evaluation, expected_reserve, expected_summary = (
        derive_normalization_icu_evaluation_inventory(
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            rules=rules,
        ))
    evaluation = _read_jsonl(root / "evaluation.inventory.jsonl")
    reserve = _read_jsonl(root / "reserve.identity.jsonl")
    for key, path, records in (
            ("evaluation_inventory", root / "evaluation.inventory.jsonl",
             evaluation),
            ("reserve_identity", root / "reserve.identity.jsonl", reserve)):
        identity = manifest[key]
        if (not isinstance(identity, dict)
                or set(identity) != {"bytes", "record_count", "relative_path",
                                     "sha256"}
                or identity["relative_path"] != path.name
                or type(identity["bytes"]) is not int
                or type(identity["record_count"]) is not int
                or identity["bytes"] != path.stat().st_size
                or identity["record_count"] != len(records)
                or identity["sha256"] != _sha256(path.read_bytes())):
            raise BroadQaExternalDataError(
                "ICU normalization evaluation artifact commitment 漂移")
    if (evaluation != expected_evaluation or reserve != expected_reserve
            or not _strict_equal(manifest["summary"], expected_summary)):
        raise BroadQaExternalDataError(
            "ICU normalization evaluation inventory/source 漂移")
    return ({**manifest, "manifest_sha256": _sha256(payload)},
            evaluation, reserve)


def main(argv: list[str] | None = None) -> int:
    """发布或回读未消费 ICU normalization evaluation protocol。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-root", required=True)
    publish.add_argument("--source-pack-dir", required=True)
    publish.add_argument("--target-dir", required=True)
    read = subparsers.add_parser("read")
    read.add_argument("--source-pack-dir", required=True)
    read.add_argument("--target-dir", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "publish":
        report = publish_normalization_icu_evaluation_protocol(
            run_root=arguments.run_root,
            source_pack_dir=arguments.source_pack_dir,
            target_dir=arguments.target_dir,
        )
    else:
        report, _, _ = read_normalization_icu_evaluation_protocol(
            arguments.target_dir,
            source_pack_dir=arguments.source_pack_dir,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_ICU_EVALUATION_DIMENSIONS",
    "NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT",
    "NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND",
    "NORMALIZATION_ICU_EVALUATION_STATUS",
    "derive_normalization_icu_evaluation_inventory",
    "normalization_icu_evaluation_split",
    "publish_normalization_icu_evaluation_protocol",
    "read_normalization_icu_evaluation_protocol",
]
