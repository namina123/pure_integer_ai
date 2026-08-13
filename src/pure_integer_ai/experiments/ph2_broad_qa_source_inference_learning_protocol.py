"""来源归纳 learner 的固定切分、物理隔离和协议来源身份。

本模块只把已冻结 training dossier 与机械 census 切成 learner 可读 TRAIN、
evaluator 可读 VALIDATION 以及双方均不可读 payload 的 RESERVE 身份清单。它不
产生语义标签、Evidence、规则或 mastery，也不接受调用方自选 item 列表。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from pure_integer_ai.cognition.shared.identity import (
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_census import (
    MECHANICAL_SIGNAL_STATES,
    SOURCE_INFERENCE_TRAINING_CENSUS_KIND,
    SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_dossier import (
    SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
    read_source_inference_training_dossier,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_INFERENCE_LEARNING_PROTOCOL_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNING_PROTOCOL_V2")
SOURCE_INFERENCE_LEARNING_SPLIT_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNING_SPLIT_RECORD_V1")
SOURCE_INFERENCE_LEARNING_RESERVE_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNING_RESERVE_IDENTITY_V1")
SOURCE_INFERENCE_LEARNING_PROTOCOL_SEED = (
    "SOURCE_INFERENCE_LEARNING_PROTOCOL_V1")
SOURCE_INFERENCE_LEARNING_FAMILIES = (
    "NORMALIZATION_EQUIVALENCE",
    "SOURCE_SPAN_SELECTION",
)
SOURCE_INFERENCE_BLOCKED_FAMILIES = (
    "PARENTHETICAL_EXPANSION",
    "ENUMERATION_MEMBER_SELECTION",
    "EXPLICIT_UNIT_ERA_FORMAT_MAPPING",
    "FINITE_ROLE_COMPOSITION",
)
SOURCE_INFERENCE_LEARNING_SPLITS = ("TRAIN", "VALIDATION", "RESERVE")
SOURCE_INFERENCE_LEARNING_ACCESS_ROLES = ("LEARNER", "EVALUATOR")
SOURCE_INFERENCE_PROTOCOL_SOURCE_KIND = 817031
SOURCE_INFERENCE_MECHANICAL_SIGNAL_USAGE = {
    "candidate_routing_allowed": 1,
    "evidence_stance_assignment_allowed": 0,
    "mastery_or_acceptance_allowed": 0,
    "semantic_label_interpretation_allowed": 0,
}
SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_CONTRACTS = {
    "NORMALIZATION_EQUIVALENCE": {
        "candidate_application_must_be_replayable": 1,
        "expected_output": "NORMALIZED_FROZEN_GOLD_ANSWER",
        "input": "QUESTION_AND_TERMINAL_SOURCE_WITHOUT_CENSUS_STANCE",
        "refute_condition": "REPLAYED_OUTPUT_DIFFERS_FROM_EXPECTED",
        "support_condition": "REPLAYED_OUTPUT_EQUALS_EXPECTED",
    },
    "SOURCE_SPAN_SELECTION": {
        "candidate_application_must_be_replayable": 1,
        "expected_output": "NORMALIZED_FROZEN_GOLD_ANSWER",
        "input": "QUESTION_AND_TERMINAL_PASSAGES_WITHOUT_CENSUS_STANCE",
        "refute_condition": "REPLAYED_SELECTED_OUTPUT_DIFFERS_FROM_EXPECTED",
        "support_condition": "REPLAYED_SELECTED_OUTPUT_EQUALS_EXPECTED",
    },
}
SOURCE_INFERENCE_LEARNING_STOPPING_CONTRACT = {
    "all_train_items_processed_once_before_completion": 1,
    "early_success_allowed": 0,
    "fresh_resume_byte_equivalence_required": 1,
    "identity_dispatch_allowed": 0,
    "maximum_evidence_candidates_per_item": 2,
    "maximum_rule_candidates_per_family": 64,
    "teacher_api_llm_calls_allowed": 0,
}
_SPLIT_BUCKETS = {
    "TRAIN": (0, 1, 2, 3, 4),
    "VALIDATION": (5, 6),
    "RESERVE": (7,),
}
_ACCESS_SPLIT = {"LEARNER": "TRAIN", "EVALUATOR": "VALIDATION"}


def _sha256(value: object, *, label: str) -> str:
    """要求协议承诺使用小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


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


def _sha256_file(path: Path) -> str:
    """流式计算输入或协议文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """要求协议输入输出位于显式 run root 内。"""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _read_manifest(
        path: Path,
        *,
        artifact_kind: str,
        status: str,
        ) -> dict[str, object]:
    """严格读取既有冻结输入 manifest。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("source inference manifest 不可读") from error
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or value.get("artifact_kind") != artifact_kind
            or value.get("format_version") != 1
            or value.get("status") != status):
        raise BroadQaExternalDataError("source inference manifest 漂移")
    return value


def _read_census_records(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读 v2 机械 census，拒绝语义标签或规则写入。"""
    expected = {
        "format_version", "item_id", "mechanical_reason",
        "mechanical_signal_state", "operator_family", "record_kind",
        "rules_written", "semantic_label_written", "source_key",
        "training_assignment",
    }
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                identity = (
                    value.get("item_id"), value.get("operator_family"))
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or set(value) != expected or value["format_version"] != 1
                        or type(value["format_version"]) is not int
                        or value["record_kind"]
                        != SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND
                        or value["mechanical_signal_state"]
                        not in MECHANICAL_SIGNAL_STATES
                        or type(value["rules_written"]) is not int
                        or value["rules_written"] != 0
                        or type(value["semantic_label_written"]) is not int
                        or value["semantic_label_written"] != 0
                        or identity in identities):
                    raise BroadQaExternalDataError(
                        f"source inference census record 漂移: {line_number}")
                identities.add(identity)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference census records 不可读") from error
    if not values:
        raise BroadQaExternalDataError("source inference census records 为空")
    return tuple(values)


def source_inference_learning_bucket(item_id: str) -> int:
    """按冻结 seed 和完整 item identity 计算全局八桶。"""
    _sha256(item_id, label="learning item_id")
    payload = (
        SOURCE_INFERENCE_LEARNING_PROTOCOL_SEED + "\0" + item_id
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % 8


def source_inference_learning_split(item_id: str) -> str:
    """把固定八桶映射为 TRAIN、VALIDATION 或 RESERVE。"""
    bucket = source_inference_learning_bucket(item_id)
    for split, buckets in _SPLIT_BUCKETS.items():
        if bucket in buckets:
            return split
    raise AssertionError("source inference split bucket 未覆盖")


def source_inference_protocol_source(
        protocol_manifest_sha256: str,
        ) -> SourceRef:
    """把协议 manifest 本身映射为来源，而不是伪装成训练文档来源。"""
    digest = _sha256(
        protocol_manifest_sha256, label="learning protocol manifest")
    return SourceRef(
        SOURCE_INFERENCE_PROTOCOL_SOURCE_KIND,
        int(digest, 16) + 1,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(curriculum=CurriculumVersion(1)),
    )


def source_inference_protocol_scope(
        protocol_manifest_sha256: str,
        ) -> ScopeIdentity:
    """返回只指向协议 artifact 的 document scope。"""
    return document_scope(
        source_inference_protocol_source(protocol_manifest_sha256))


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    """以不可覆盖方式写入规范 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))


def _file_identity(path: Path, root: Path, count: int) -> dict[str, object]:
    """返回协议内相对路径、字节、条数和摘要。"""
    return {
        "bytes": path.stat().st_size,
        "record_count": count,
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _state_counts(
        records: tuple[dict[str, object], ...],
        *,
        split_by_item: dict[str, str],
        ) -> dict[str, dict[str, dict[str, int]]]:
    """按 family、split 和三态冻结机械库存计数。"""
    counts: dict[str, dict[str, Counter[str]]] = {
        family: {split: Counter() for split in SOURCE_INFERENCE_LEARNING_SPLITS}
        for family in SOURCE_INFERENCE_LEARNING_FAMILIES
    }
    for record in records:
        family = str(record["operator_family"])
        if family not in counts:
            continue
        split = split_by_item[str(record["item_id"])]
        counts[family][split][str(record["mechanical_signal_state"])] += 1
    return {
        family: {
            split: {
                state: counts[family][split][state]
                for state in MECHANICAL_SIGNAL_STATES
            }
            for split in SOURCE_INFERENCE_LEARNING_SPLITS
        }
        for family in SOURCE_INFERENCE_LEARNING_FAMILIES
    }


def publish_source_inference_learning_protocol(
        *,
        run_root: str | Path,
        dossier_manifest_path: str | Path,
        dossier_path: str | Path,
        census_manifest_path: str | Path,
        census_records_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """冻结协议、全局切分和两个角色互不混用的物理切片。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError(
            "source inference learning run root 不存在")
    dossier_manifest_file = _within(
        root, dossier_manifest_path, label="dossier_manifest_path")
    dossier_file = _within(root, dossier_path, label="dossier_path")
    census_manifest_file = _within(
        root, census_manifest_path, label="census_manifest_path")
    census_file = _within(
        root, census_records_path, label="census_records_path")
    target = _within(root, target_dir, label="target_dir")
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference learning protocol target 已存在")

    dossier_manifest = _read_manifest(
        dossier_manifest_file,
        artifact_kind=SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
        status="MATERIALIZED_UNREAD_UNLEARNED",
    )
    census_manifest = _read_manifest(
        census_manifest_file,
        artifact_kind=SOURCE_INFERENCE_TRAINING_CENSUS_KIND,
        status="MECHANICAL_CENSUS_ONLY_NOT_LEARNED",
    )
    if (_sha256_file(dossier_file) != dossier_manifest.get("dossier_sha256")
            or _sha256_file(census_file)
            != census_manifest.get("records_sha256")
            or census_manifest.get("dossier_sha256")
            != dossier_manifest.get("dossier_sha256")
            or census_manifest.get("dossier_manifest_sha256")
            != _sha256_file(dossier_manifest_file)):
        raise BroadQaExternalDataError(
            "source inference learning protocol 输入承诺漂移")
    dossier = read_source_inference_training_dossier(dossier_file)
    census = _read_census_records(census_file)
    dossier_by_id = {str(value["item_id"]): value for value in dossier}
    if (len(dossier_by_id) != len(dossier)
            or len(dossier) != dossier_manifest.get("dossier_record_count")
            or len(census) != census_manifest.get("record_count")):
        raise BroadQaExternalDataError(
            "source inference learning protocol 输入计数漂移")
    census_identities = {
        (str(value["item_id"]), str(value["operator_family"]))
        for value in census
    }
    expected_identities = {
        (item_id, family)
        for item_id in dossier_by_id
        for family in SOURCE_INFERENCE_LEARNING_FAMILIES
        + SOURCE_INFERENCE_BLOCKED_FAMILIES
    }
    if census_identities != expected_identities:
        raise BroadQaExternalDataError(
            "source inference learning protocol census 覆盖漂移")

    split_by_item = {
        item_id: source_inference_learning_split(item_id)
        for item_id in dossier_by_id
    }
    bucket_by_item = {
        item_id: source_inference_learning_bucket(item_id)
        for item_id in dossier_by_id
    }
    inventory = tuple({
        "bucket": bucket_by_item[item_id],
        "format_version": 1,
        "item_id": item_id,
        "record_kind": SOURCE_INFERENCE_LEARNING_SPLIT_RECORD_KIND,
        "source_key": dossier_by_id[item_id]["training_source"]["source_key"],
        "split": split_by_item[item_id],
        "training_assignment": dossier_by_id[item_id]["training_assignment"],
    } for item_id in sorted(dossier_by_id))
    active_census = tuple(
        value for value in census
        if value["operator_family"] in SOURCE_INFERENCE_LEARNING_FAMILIES)
    train_ids = {
        item_id for item_id, split in split_by_item.items()
        if split == "TRAIN"
    }
    validation_ids = {
        item_id for item_id, split in split_by_item.items()
        if split == "VALIDATION"
    }
    reserve_ids = {
        item_id for item_id, split in split_by_item.items()
        if split == "RESERVE"
    }
    train_dossier = tuple(
        dossier_by_id[item_id] for item_id in sorted(train_ids))
    validation_dossier = tuple(
        dossier_by_id[item_id] for item_id in sorted(validation_ids))
    train_census = tuple(sorted(
        (value for value in active_census if value["item_id"] in train_ids),
        key=lambda value: (value["item_id"], value["operator_family"]),
    ))
    validation_census = tuple(sorted(
        (value for value in active_census
         if value["item_id"] in validation_ids),
        key=lambda value: (value["item_id"], value["operator_family"]),
    ))
    reserve_inventory = tuple({
        "format_version": 1,
        "item_id": item_id,
        "record_kind": SOURCE_INFERENCE_LEARNING_RESERVE_RECORD_KIND,
        "split": "RESERVE",
    } for item_id in sorted(reserve_ids))

    target.mkdir(parents=True)
    inventory_path = target / "item-split.inventory.jsonl"
    train_dossier_path = target / "learner" / "train.dossier.jsonl"
    train_census_path = target / "learner" / "train.census.jsonl"
    validation_dossier_path = (
        target / "evaluator" / "validation.dossier.jsonl")
    validation_census_path = (
        target / "evaluator" / "validation.census.jsonl")
    reserve_path = target / "reserve" / "reserve.identity.jsonl"
    for path, records in (
            (inventory_path, inventory),
            (train_dossier_path, train_dossier),
            (train_census_path, train_census),
            (validation_dossier_path, validation_dossier),
            (validation_census_path, validation_census),
            (reserve_path, reserve_inventory)):
        _write_jsonl(path, records)

    split_counts = Counter(split_by_item.values())
    manifest = {
        "access_artifacts": {
            "EVALUATOR": {
                "census": _file_identity(
                    validation_census_path, target, len(validation_census)),
                "dossier": _file_identity(
                    validation_dossier_path, target, len(validation_dossier)),
                "split": "VALIDATION",
            },
            "LEARNER": {
                "census": _file_identity(
                    train_census_path, target, len(train_census)),
                "dossier": _file_identity(
                    train_dossier_path, target, len(train_dossier)),
                "split": "TRAIN",
            },
        },
        "artifact_kind": SOURCE_INFERENCE_LEARNING_PROTOCOL_KIND,
        "blocked_operator_families": list(SOURCE_INFERENCE_BLOCKED_FAMILIES),
        "bucket_modulus": 8,
        "census_manifest_sha256": _sha256_file(census_manifest_file),
        "census_records_sha256": _sha256_file(census_file),
        "dossier_manifest_sha256": _sha256_file(dossier_manifest_file),
        "dossier_sha256": _sha256_file(dossier_file),
        "enabled_operator_families": list(SOURCE_INFERENCE_LEARNING_FAMILIES),
        "format_version": 1,
        "item_split_inventory": _file_identity(
            inventory_path, target, len(inventory)),
        "learner_payload_split": "TRAIN",
        "learning_stopping_contract": dict(
            SOURCE_INFERENCE_LEARNING_STOPPING_CONTRACT),
        "mechanical_signal_usage": dict(
            SOURCE_INFERENCE_MECHANICAL_SIGNAL_USAGE),
        "operator_family_concurrency_limit": 1,
        "evidence_qualification_contracts": {
            family: dict(SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_CONTRACTS[
                family])
            for family in SOURCE_INFERENCE_LEARNING_FAMILIES
        },
        "protocol_seed": SOURCE_INFERENCE_LEARNING_PROTOCOL_SEED,
        "reserve_artifact": _file_identity(
            reserve_path, target, len(reserve_inventory)),
        "reserve_payload_published": 0,
        "split_buckets": {
            split: list(_SPLIT_BUCKETS[split])
            for split in SOURCE_INFERENCE_LEARNING_SPLITS
        },
        "split_item_counts": {
            split: split_counts[split]
            for split in SOURCE_INFERENCE_LEARNING_SPLITS
        },
        "split_state_counts": _state_counts(
            active_census, split_by_item=split_by_item),
        "status": "FROZEN_NOT_LEARNED",
        "validation_acceptance_gates": {
            family: {
                "evaluated_item_count_required": split_counts["VALIDATION"],
                "false_acceptance_max": 0,
                "false_rejection_max": 0,
                "forced_undetermined_max": 0,
                "fresh_resume_byte_equivalence_required": 1,
                "identity_dispatch_allowed": 0,
                "independent_replay_required": 1,
            }
            for family in SOURCE_INFERENCE_LEARNING_FAMILIES
        },
        "validation_payload_split": "VALIDATION",
        "validation_read_count_at_freeze": 0,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def _strict_file_identity(
        value: object,
        *,
        label: str,
        relative_path: str,
        ) -> dict[str, object]:
    """严格核验协议内文件 identity。"""
    expected = {"bytes", "record_count", "relative_path", "sha256"}
    if (not isinstance(value, dict) or set(value) != expected
            or type(value["bytes"]) is not int or value["bytes"] < 0
            or type(value["record_count"]) is not int
            or value["record_count"] < 0
            or not isinstance(value["relative_path"], str)
            or value["relative_path"] != relative_path):
        raise BroadQaExternalDataError(f"{label} file identity 漂移")
    _sha256(value["sha256"], label=f"{label} file")
    return value


def _strict_split_counts(
        value: object,
        *,
        file_identities: dict[str, dict[str, object]],
        ) -> dict[str, int]:
    """核验 split 总数与全部物理 dossier/census identity 一致。"""
    if (not isinstance(value, dict)
            or set(value) != set(SOURCE_INFERENCE_LEARNING_SPLITS)
            or any(type(count) is not int or count < 0
                   for count in value.values())):
        raise BroadQaExternalDataError("learning split item counts 漂移")
    counts = {split: value[split]
              for split in SOURCE_INFERENCE_LEARNING_SPLITS}
    family_count = len(SOURCE_INFERENCE_LEARNING_FAMILIES)
    if (file_identities["inventory"]["record_count"] != sum(counts.values())
            or file_identities["reserve"]["record_count"]
            != counts["RESERVE"]
            or file_identities["learner_dossier"]["record_count"]
            != counts["TRAIN"]
            or file_identities["learner_census"]["record_count"]
            != counts["TRAIN"] * family_count
            or file_identities["evaluator_dossier"]["record_count"]
            != counts["VALIDATION"]
            or file_identities["evaluator_census"]["record_count"]
            != counts["VALIDATION"] * family_count):
        raise BroadQaExternalDataError(
            "learning split 与物理 artifact 计数漂移")
    return counts


def _strict_split_state_counts(
        value: object,
        *,
        split_counts: dict[str, int],
        ) -> None:
    """核验每个启用 family 的机械三态完整覆盖对应 split。"""
    if (not isinstance(value, dict)
            or set(value) != set(SOURCE_INFERENCE_LEARNING_FAMILIES)):
        raise BroadQaExternalDataError("learning split state counts 漂移")
    for family in SOURCE_INFERENCE_LEARNING_FAMILIES:
        family_value = value[family]
        if (not isinstance(family_value, dict)
                or set(family_value) != set(SOURCE_INFERENCE_LEARNING_SPLITS)):
            raise BroadQaExternalDataError(
                "learning split state counts 漂移")
        for split in SOURCE_INFERENCE_LEARNING_SPLITS:
            state_value = family_value[split]
            if (not isinstance(state_value, dict)
                    or set(state_value) != set(MECHANICAL_SIGNAL_STATES)
                    or any(type(count) is not int or count < 0
                           for count in state_value.values())
                    or sum(state_value.values()) != split_counts[split]):
                raise BroadQaExternalDataError(
                    "learning split state counts 漂移")


def read_source_inference_learning_protocol(
        path: str | Path,
        ) -> dict[str, object]:
    """严格回读协议 manifest，拒绝未知字段和边界漂移。"""
    file = Path(path).resolve()
    try:
        payload = file.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference learning protocol 不可读") from error
    expected = {
        "access_artifacts", "artifact_kind", "blocked_operator_families",
        "bucket_modulus", "census_manifest_sha256", "census_records_sha256",
        "dossier_manifest_sha256", "dossier_sha256",
        "enabled_operator_families", "evidence_qualification_contracts",
        "format_version", "item_split_inventory", "learner_payload_split",
        "learning_stopping_contract", "mechanical_signal_usage",
        "operator_family_concurrency_limit",
        "protocol_seed", "reserve_artifact", "reserve_payload_published",
        "split_buckets", "split_item_counts", "split_state_counts", "status",
        "validation_acceptance_gates", "validation_payload_split",
        "validation_read_count_at_freeze",
    }
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or set(value) != expected
            or value["artifact_kind"] != SOURCE_INFERENCE_LEARNING_PROTOCOL_KIND
            or type(value["format_version"]) is not int
            or value["format_version"] != 1
            or value["protocol_seed"]
            != SOURCE_INFERENCE_LEARNING_PROTOCOL_SEED
            or type(value["bucket_modulus"]) is not int
            or value["bucket_modulus"] != 8
            or value["enabled_operator_families"]
            != list(SOURCE_INFERENCE_LEARNING_FAMILIES)
            or value["blocked_operator_families"]
            != list(SOURCE_INFERENCE_BLOCKED_FAMILIES)
            or type(value["operator_family_concurrency_limit"]) is not int
            or value["operator_family_concurrency_limit"] != 1
            or value["learner_payload_split"] != "TRAIN"
            or value["validation_payload_split"] != "VALIDATION"
            or type(value["reserve_payload_published"]) is not int
            or value["reserve_payload_published"] != 0
            or type(value["validation_read_count_at_freeze"]) is not int
            or value["validation_read_count_at_freeze"] != 0
            or value["status"] != "FROZEN_NOT_LEARNED"
            or not _strict_equal(
                value["mechanical_signal_usage"],
                SOURCE_INFERENCE_MECHANICAL_SIGNAL_USAGE)
            or not _strict_equal(
                value["evidence_qualification_contracts"],
                SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_CONTRACTS)
            or not _strict_equal(
                value["learning_stopping_contract"],
                SOURCE_INFERENCE_LEARNING_STOPPING_CONTRACT)
            or not _strict_equal(value["split_buckets"], {
                split: list(_SPLIT_BUCKETS[split])
                for split in SOURCE_INFERENCE_LEARNING_SPLITS
            })):
        raise BroadQaExternalDataError(
            "source inference learning protocol manifest 漂移")
    for name in (
            "census_manifest_sha256", "census_records_sha256",
            "dossier_manifest_sha256", "dossier_sha256"):
        _sha256(value[name], label=name)
    inventory_identity = _strict_file_identity(
        value["item_split_inventory"],
        label="item split inventory",
        relative_path="item-split.inventory.jsonl",
    )
    reserve_identity = _strict_file_identity(
        value["reserve_artifact"],
        label="reserve",
        relative_path="reserve/reserve.identity.jsonl",
    )
    access = value["access_artifacts"]
    if (not isinstance(access, dict)
            or set(access) != set(SOURCE_INFERENCE_LEARNING_ACCESS_ROLES)):
        raise BroadQaExternalDataError("learning access artifacts 漂移")
    access_identities = {}
    for role, split in _ACCESS_SPLIT.items():
        entry = access[role]
        if (not isinstance(entry, dict)
                or set(entry) != {"census", "dossier", "split"}
                or entry["split"] != split):
            raise BroadQaExternalDataError(
                f"learning {role} access artifact 漂移")
        access_identities[(role, "census")] = _strict_file_identity(
            entry["census"],
            label=f"{role} census",
            relative_path=("learner/train.census.jsonl" if role == "LEARNER"
                           else "evaluator/validation.census.jsonl"),
        )
        access_identities[(role, "dossier")] = _strict_file_identity(
            entry["dossier"],
            label=f"{role} dossier",
            relative_path=("learner/train.dossier.jsonl" if role == "LEARNER"
                           else "evaluator/validation.dossier.jsonl"),
        )
    split_counts = _strict_split_counts(value["split_item_counts"],
        file_identities={
            "inventory": inventory_identity,
            "reserve": reserve_identity,
            "learner_census": access_identities[("LEARNER", "census")],
            "learner_dossier": access_identities[("LEARNER", "dossier")],
            "evaluator_census": access_identities[("EVALUATOR", "census")],
            "evaluator_dossier": access_identities[("EVALUATOR", "dossier")],
        })
    _strict_split_state_counts(
        value["split_state_counts"], split_counts=split_counts)
    expected_gates = {
        family: {
            "evaluated_item_count_required": split_counts["VALIDATION"],
            "false_acceptance_max": 0,
            "false_rejection_max": 0,
            "forced_undetermined_max": 0,
            "fresh_resume_byte_equivalence_required": 1,
            "identity_dispatch_allowed": 0,
            "independent_replay_required": 1,
        }
        for family in SOURCE_INFERENCE_LEARNING_FAMILIES
    }
    if not _strict_equal(value["validation_acceptance_gates"], expected_gates):
        raise BroadQaExternalDataError(
            "learning validation acceptance gates 漂移")
    return {**value, "manifest_sha256": hashlib.sha256(payload).hexdigest()}


def _committed_file(
        root: Path,
        identity: dict[str, object],
        *,
        label: str,
        ) -> Path:
    """解析协议相对路径并重验字节、条数之外的内容承诺。"""
    path = (root / str(identity["relative_path"])).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or _sha256_file(path) != identity["sha256"]):
        raise BroadQaExternalDataError(f"learning {label} commitment 漂移")
    return path


def read_source_inference_learning_slice(
        *,
        protocol_dir: str | Path,
        access_role: str,
        operator_family: str,
        ) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """按单一角色和 family 回读一个物理切片，禁止 RESERVE/跨族读取。"""
    root = Path(protocol_dir).resolve()
    if access_role not in SOURCE_INFERENCE_LEARNING_ACCESS_ROLES:
        raise BroadQaExternalDataError("learning access role 未注册")
    if operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
        raise BroadQaExternalDataError("learning operator family 未启用")
    manifest = read_source_inference_learning_protocol(root / "manifest.json")
    access = manifest["access_artifacts"][access_role]
    dossier_identity = access["dossier"]
    census_identity = access["census"]
    dossier_path = _committed_file(
        root, dossier_identity, label=f"{access_role} dossier")
    census_path = _committed_file(
        root, census_identity, label=f"{access_role} census")
    dossier = read_source_inference_training_dossier(dossier_path)
    census_all = _read_census_records(census_path)
    census = tuple(
        value for value in census_all
        if value["operator_family"] == operator_family)
    item_ids = {str(value["item_id"]) for value in dossier}
    census_ids = {str(value["item_id"]) for value in census}
    expected_split = _ACCESS_SPLIT[access_role]
    if (len(dossier) != dossier_identity["record_count"]
            or len(census_all) != census_identity["record_count"]
            or item_ids != census_ids
            or any(source_inference_learning_split(item_id) != expected_split
                   for item_id in item_ids)):
        raise BroadQaExternalDataError(
            "source inference learning slice 跨 split 或计数漂移")
    return dossier, census


def _work_path(value: str) -> Path:
    """要求正式协议路径为显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从冻结 dossier/census 发布不可覆盖 learning protocol。"""
    parser = argparse.ArgumentParser(
        description="Freeze source-inference learning protocol slices.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--dossier-manifest", type=_work_path, required=True)
    parser.add_argument("--dossier", type=_work_path, required=True)
    parser.add_argument("--census-manifest", type=_work_path, required=True)
    parser.add_argument("--census-records", type=_work_path, required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    args = parser.parse_args(argv)
    report = publish_source_inference_learning_protocol(
        run_root=args.run_root,
        dossier_manifest_path=args.dossier_manifest,
        dossier_path=args.dossier,
        census_manifest_path=args.census_manifest,
        census_records_path=args.census_records,
        target_dir=args.target_dir,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_INFERENCE_BLOCKED_FAMILIES",
    "SOURCE_INFERENCE_LEARNING_ACCESS_ROLES",
    "SOURCE_INFERENCE_LEARNING_FAMILIES",
    "SOURCE_INFERENCE_LEARNING_PROTOCOL_KIND",
    "SOURCE_INFERENCE_LEARNING_PROTOCOL_SEED",
    "SOURCE_INFERENCE_LEARNING_SPLITS",
    "SOURCE_INFERENCE_EVIDENCE_QUALIFICATION_CONTRACTS",
    "SOURCE_INFERENCE_LEARNING_STOPPING_CONTRACT",
    "SOURCE_INFERENCE_MECHANICAL_SIGNAL_USAGE",
    "SOURCE_INFERENCE_PROTOCOL_SOURCE_KIND",
    "main",
    "publish_source_inference_learning_protocol",
    "read_source_inference_learning_protocol",
    "read_source_inference_learning_slice",
    "source_inference_learning_bucket",
    "source_inference_learning_split",
    "source_inference_protocol_scope",
    "source_inference_protocol_source",
]
