"""冻结 normalization ICU evaluation 的候选、代码和运行边界。

family freeze 只读取 evaluation protocol manifest 的 identity，不读取 evaluation
inventory 或 reserve payload。它严格重放 development pack 与 OpenCC TRAIN_SOURCE，
并绑定 candidate clone 和全部 evaluator/runner 代码文件。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_candidate_clone import (
    compile_normalization_candidate_clone,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    read_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_development_learner_v1 import (
    read_normalization_development_learner_v1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_protocol import (
    NORMALIZATION_ICU_EVALUATION_DIMENSIONS,
    NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_ICU_EVALUATION_RUN_CONTRACT,
    NORMALIZATION_ICU_EVALUATION_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    read_normalization_icu_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    read_normalization_rule_pack_v3,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_ICU_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_EVALUATION_FAMILY_FREEZE_V1")
NORMALIZATION_ICU_EVALUATION_FAMILY_STATUS = (
    "FROZEN_CANDIDATE_AND_CODE_UNCONSUMED")
NORMALIZATION_ICU_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_icu_evaluation_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_candidate_clone.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_icu_evaluator.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_icu_evaluation_family.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_icu_evaluation_runner.py",
)
NORMALIZATION_ICU_EVALUATION_DATA_ARGUMENTS = (
    "icu_source_pack_dir",
    "evaluation_protocol_dir",
    "normalization_source_pack_dir",
    "contrastive_protocol_dir",
    "rule_pack_dir",
    "fresh_learner_dir",
    "resumed_learner_dir",
)


def _sha256(payload: bytes) -> str:
    """返回文件或规范对象的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_sha(value: object, *, label: str) -> str:
    """核验小写 SHA-256。"""
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


def require_normalization_k_run_root(value: str | Path) -> Path:
    """解析唯一工作盘根并拒绝任何非 K 盘回退。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization evaluation run root 必须是已存在的 K 盘目录")
    return root


def normalization_path_within(
        root: Path,
        value: str | Path,
        *,
        label: str,
        ) -> Path:
    """解析路径并要求始终位于显式 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def validate_normalization_evaluation_data_paths(
        root: Path,
        arguments: dict[str, object],
        ) -> dict[str, Path]:
    """要求全部评测/候选物理输入位于显式 K 盘 run root。"""
    if (not isinstance(root, Path) or not root.is_dir()
            or not isinstance(arguments, dict)):
        raise BroadQaExternalDataError(
            "normalization evaluation data path 参数非法")
    paths = {}
    for name in NORMALIZATION_ICU_EVALUATION_DATA_ARGUMENTS:
        if name not in arguments:
            raise BroadQaExternalDataError(
                f"normalization evaluation 缺少路径参数: {name}")
        path = normalization_path_within(root, arguments[name], label=name)
        if not path.is_dir():
            raise BroadQaExternalDataError(
                f"normalization evaluation 输入目录不存在: {name}")
        paths[name] = path
    return paths


def _read_protocol_manifest_only(
        protocol_dir: Path,
        ) -> tuple[dict[str, object], bytes]:
    """只读取 protocol manifest，不打开 evaluation/reserve payload。"""
    try:
        payload = (protocol_dir / "manifest.json").read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization evaluation protocol manifest 不可读") from error
    required = {
        "artifact_kind", "dimensions", "evaluation_inventory",
        "evaluation_run_count", "format_version", "learned_pack_read_count",
        "mastery_claimed", "metric_contract", "overall_pass_contract",
        "production_enabled", "reserve_identity", "reserve_labels_published",
        "run_contract", "selection_rule", "source_pack_manifest_sha256",
        "status", "summary",
    }
    if (not isinstance(manifest, dict) or set(manifest) != required
            or canonical_json_line(manifest) != payload
            or manifest["artifact_kind"]
            != NORMALIZATION_ICU_EVALUATION_PROTOCOL_KIND
            or manifest["status"] != NORMALIZATION_ICU_EVALUATION_STATUS
            or manifest["format_version"] != 2
            or not _strict_equal(
                manifest["dimensions"], NORMALIZATION_ICU_EVALUATION_DIMENSIONS)
            or not _strict_equal(
                manifest["metric_contract"],
                NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT)
            or not _strict_equal(
                manifest["run_contract"],
                NORMALIZATION_ICU_EVALUATION_RUN_CONTRACT)
            or any(type(manifest[name]) is not int or manifest[name] != 0
                   for name in (
                       "evaluation_run_count", "learned_pack_read_count",
                       "mastery_claimed", "production_enabled",
                       "reserve_labels_published"))):
        raise BroadQaExternalDataError(
            "normalization evaluation protocol manifest 漂移")
    for name in ("evaluation_inventory", "reserve_identity"):
        identity = manifest[name]
        if (not isinstance(identity, dict)
                or set(identity) != {
                    "bytes", "record_count", "relative_path", "sha256"}
                or type(identity["bytes"]) is not int
                or identity["bytes"] <= 0
                or type(identity["record_count"]) is not int
                or identity["record_count"] <= 0):
            raise BroadQaExternalDataError(
                "normalization evaluation payload identity 漂移")
        _strict_sha(
            identity["sha256"],
            label=f"normalization evaluation {name} SHA",
        )
    return manifest, payload


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结 evaluator family 的完整 live code 文件身份。"""
    files = []
    for relative in NORMALIZATION_ICU_EVALUATION_CODE_FILES:
        path = (repository / Path(*relative.split("/"))).resolve()
        if (not path.is_relative_to(repository) or not path.is_file()):
            raise BroadQaExternalDataError(
                f"normalization evaluation code file 缺失: {relative}")
        payload = path.read_bytes()
        files.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    code_sha = _sha256(canonical_json_bytes(files))
    return files, code_sha


def build_normalization_icu_evaluation_family_freeze(
        *,
        repository_root: str | Path,
        icu_source_pack_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        normalization_source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        rule_pack_dir: str | Path,
        fresh_learner_dir: str | Path,
        resumed_learner_dir: str | Path,
        ) -> dict[str, object]:
    """重放候选和 live code，构造零 evaluation payload read 的 freeze。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise BroadQaExternalDataError(
            "normalization evaluation repository root 非法")
    protocol_dir = Path(evaluation_protocol_dir).resolve()
    protocol, protocol_payload = _read_protocol_manifest_only(protocol_dir)
    icu_manifest, _, _ = read_normalization_icu_source_pack(
        icu_source_pack_dir)
    if (icu_manifest["manifest_sha256"]
            != protocol["source_pack_manifest_sha256"]):
        raise BroadQaExternalDataError(
            "normalization evaluation ICU source/protocol 漂移")
    fresh_manifest, fresh_accepted, fresh_rejected = (
        read_normalization_development_learner_v1(
            fresh_learner_dir,
            source_pack_dir=normalization_source_pack_dir,
            contrastive_protocol_dir=contrastive_protocol_dir,
        ))
    resumed_manifest, resumed_accepted, resumed_rejected = (
        read_normalization_development_learner_v1(
            resumed_learner_dir,
            source_pack_dir=normalization_source_pack_dir,
            contrastive_protocol_dir=contrastive_protocol_dir,
        ))
    pack_manifest, pack_accepted, pack_rejected = read_normalization_rule_pack_v3(
        rule_pack_dir,
        source_pack_dir=normalization_source_pack_dir,
        contrastive_protocol_dir=contrastive_protocol_dir,
        fresh_checkpoint_chain_path=(
            Path(fresh_learner_dir).resolve() / "checkpoints.jsonl"),
        resumed_checkpoint_chain_path=(
            Path(resumed_learner_dir).resolve() / "checkpoints.jsonl"),
    )
    if (fresh_accepted != resumed_accepted
            or fresh_rejected != resumed_rejected
            or fresh_accepted != pack_accepted
            or fresh_rejected != pack_rejected
            or fresh_manifest["result_sha256"]
            != resumed_manifest["result_sha256"]
            or fresh_manifest["result_sha256"]
            != pack_manifest["fresh_result_sha256"]):
        raise BroadQaExternalDataError(
            "normalization evaluation candidate lineage 漂移")
    _, _, contrastive_trials = read_normalization_contrastive_protocol(
        contrastive_protocol_dir,
        source_pack_dir=normalization_source_pack_dir,
    )
    clone = compile_normalization_candidate_clone(
        rule_pack_manifest_sha256=pack_manifest["manifest_sha256"],
        accepted_rules=pack_accepted,
        rejected_trials=pack_rejected,
        contrastive_trials=contrastive_trials,
    )
    code_files, code_sha = _code_identity(repository)
    candidate_identity = {
        "candidate_clone_sha256": clone.sha256(),
        "fresh_checkpoint_chain_sha256": (
            fresh_manifest["checkpoint_chain_sha256"]),
        "fresh_learner_manifest_sha256": fresh_manifest["manifest_sha256"],
        "resumed_checkpoint_chain_sha256": (
            resumed_manifest["checkpoint_chain_sha256"]),
        "resumed_learner_manifest_sha256": resumed_manifest["manifest_sha256"],
        "rule_pack_manifest_sha256": pack_manifest["manifest_sha256"],
    }
    candidate_sha = _sha256(canonical_json_bytes(candidate_identity))
    freeze_core = {
        "artifact_kind": NORMALIZATION_ICU_EVALUATION_FAMILY_KIND,
        "candidate_freeze": candidate_identity,
        "candidate_freeze_sha256": candidate_sha,
        "candidate_pack_read_count": 1,
        "code_files": code_files,
        "code_freeze_sha256": code_sha,
        "evaluation_inventory_identity": protocol["evaluation_inventory"],
        "evaluation_payload_read_count": 0,
        "evaluation_protocol_manifest_sha256": _sha256(protocol_payload),
        "evaluation_run_count": 0,
        "format_version": 1,
        "icu_source_pack_manifest_sha256": icu_manifest["manifest_sha256"],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "reserve_identity": protocol["reserve_identity"],
        "reserve_payload_read_count": 0,
        "status": NORMALIZATION_ICU_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    return {
        **freeze_core,
        "family_commitment_sha256": _sha256(
            canonical_json_bytes(freeze_core)),
    }


def publish_normalization_icu_evaluation_family_freeze(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **build_arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 K 盘 family freeze，不读取 evaluation payload。"""
    root = require_normalization_k_run_root(run_root)
    validate_normalization_evaluation_data_paths(root, build_arguments)
    target = normalization_path_within(
        root, target_dir, label="normalization evaluation family target")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization evaluation family target 已存在")
    freeze = build_normalization_icu_evaluation_family_freeze(
        **build_arguments)
    target.mkdir(parents=True)
    path = target / "family-freeze.json"
    path.write_bytes(canonical_json_line(freeze))
    return {
        **freeze,
        "manifest_sha256": _sha256(path.read_bytes()),
    }


def read_normalization_icu_evaluation_family_freeze(
        target_dir: str | Path,
        **build_arguments: object,
        ) -> dict[str, object]:
    """严格回读 freeze 并重算 live candidate、code 和 protocol identity。"""
    root = Path(target_dir).resolve()
    try:
        payload = (root / "family-freeze.json").read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization evaluation family freeze 不可读") from error
    expected = build_normalization_icu_evaluation_family_freeze(
        **build_arguments)
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or not _strict_equal(value, expected)):
        raise BroadQaExternalDataError(
            "normalization evaluation family freeze 与 live identity 漂移")
    return {
        **value,
        "manifest_sha256": _sha256(payload),
    }


def main(argv: list[str] | None = None) -> int:
    """不可覆盖发布或严格回读 normalization evaluation family freeze。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("publish", "read"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--target-dir", required=True)
    for name in NORMALIZATION_ICU_EVALUATION_DATA_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    run_root = arguments.pop("run_root")
    target = arguments.pop("target_dir")
    root = require_normalization_k_run_root(run_root)
    validate_normalization_evaluation_data_paths(root, arguments)
    target = normalization_path_within(
        root, target, label="normalization evaluation family target")
    if command == "publish":
        report = publish_normalization_icu_evaluation_family_freeze(
            run_root=root,
            target_dir=target,
            **arguments,
        )
    else:
        report = read_normalization_icu_evaluation_family_freeze(
            target, **arguments)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_ICU_EVALUATION_CODE_FILES",
    "NORMALIZATION_ICU_EVALUATION_DATA_ARGUMENTS",
    "NORMALIZATION_ICU_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_ICU_EVALUATION_FAMILY_STATUS",
    "build_normalization_icu_evaluation_family_freeze",
    "normalization_path_within",
    "publish_normalization_icu_evaluation_family_freeze",
    "read_normalization_icu_evaluation_family_freeze",
    "require_normalization_k_run_root",
    "validate_normalization_evaluation_data_paths",
]
