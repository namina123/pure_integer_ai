"""冻结 normalization recovery evaluation 的候选、代码与读取边界。

family freeze 只打开 Firefox evaluation manifest，不读取 evaluation inventory
或 reserve。候选身份由 recovery TRAIN、fresh/resume lineage、禁用态 pack 和
live code 共同决定。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_clone import (
    compile_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learner import (
    read_normalization_recovery_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_rule_pack import (
    read_normalization_recovery_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_protocol import (
    read_normalization_recovery_learner_input,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_EVALUATION_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_EVALUATION_FAMILY_FREEZE_V2")
NORMALIZATION_RECOVERY_EVALUATION_FAMILY_STATUS = (
    "FROZEN_CANDIDATE_AND_CODE_UNCONSUMED")
NORMALIZATION_RECOVERY_EVALUATION_CODE_FILES = (
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_source_inference_learning_checkpoint.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_source_inference_learning_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_materialized_learner_runtime.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_materialized_rule_pack.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_contrastive_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_source_pack.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_evaluation_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_training_records.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_training_protocol.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_learning_records.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_learner.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_rule_pack.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_candidate_records.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_candidate_compile.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_candidate_execution.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_candidate_clone.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_evaluator.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_evaluation_family.py",
    "src/pure_integer_ai/experiments/"
    "ph2_broad_qa_normalization_recovery_evaluation_runner.py",
)
NORMALIZATION_RECOVERY_EVALUATION_DATA_ARGUMENTS = (
    "evaluation_protocol_dir",
    "training_protocol_dir",
    "fresh_learner_dir",
    "resumed_learner_dir",
    "rule_pack_dir",
)
NORMALIZATION_RECOVERY_EVALUATION_IDENTITY_ARGUMENTS = (
    "expected_evaluation_protocol_manifest_sha256",
    "expected_training_protocol_manifest_sha256",
    "expected_fresh_learner_manifest_sha256",
    "expected_resumed_learner_manifest_sha256",
    "expected_rule_pack_manifest_sha256",
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
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def require_normalization_recovery_k_run_root(value: str | Path) -> Path:
    """解析唯一工作盘根并拒绝任何非 K 盘回退。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery evaluation run root 必须是 K 盘目录")
    return root


def normalization_recovery_path_within(
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


def validate_normalization_recovery_evaluation_paths(
        root: Path,
        arguments: dict[str, object],
        ) -> dict[str, Path]:
    """要求全部候选与评测物理输入位于显式 K 盘 run root。"""
    if (not isinstance(root, Path) or not root.is_dir()
            or not isinstance(arguments, dict)):
        raise BroadQaExternalDataError(
            "normalization recovery evaluation path 参数非法")
    paths = {}
    for name in NORMALIZATION_RECOVERY_EVALUATION_DATA_ARGUMENTS:
        if name not in arguments:
            raise BroadQaExternalDataError(
                f"normalization recovery evaluation 缺少路径参数: {name}")
        path = normalization_recovery_path_within(
            root, arguments[name], label=name)
        if not path.is_dir():
            raise BroadQaExternalDataError(
                f"normalization recovery evaluation 输入目录不存在: {name}")
        paths[name] = path
    for name in NORMALIZATION_RECOVERY_EVALUATION_IDENTITY_ARGUMENTS:
        if name not in arguments:
            raise BroadQaExternalDataError(
                f"normalization recovery evaluation 缺少 identity: {name}")
        _strict_sha(arguments[name], label=name)
    return paths


def _code_identity(repository: Path) -> tuple[list[dict[str, object]], str]:
    """冻结 recovery TRAIN、candidate、evaluator 与共享运行时 live code。"""
    files = []
    for relative in NORMALIZATION_RECOVERY_EVALUATION_CODE_FILES:
        path = (repository / Path(*relative.split("/"))).resolve()
        if (not path.is_relative_to(repository) or not path.is_file()):
            raise BroadQaExternalDataError(
                f"normalization recovery code file 缺失: {relative}")
        payload = path.read_bytes()
        files.append({
            "bytes": len(payload),
            "relative_path": relative,
            "sha256": _sha256(payload),
        })
    return files, _sha256(canonical_json_bytes(files))


def _validate_disabled_lineage(
        *,
        training_manifest: dict[str, object],
        fresh_manifest: dict[str, object],
        fresh_outputs: dict[str, tuple[dict[str, object], ...]],
        resumed_manifest: dict[str, object],
        resumed_outputs: dict[str, tuple[dict[str, object], ...]],
        pack_manifest: dict[str, object],
        pack_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> None:
    """核验 fresh/resume/pack 同语义且始终处于生产禁用态。"""
    protocol_sha = training_manifest["manifest_sha256"]
    manifests = (fresh_manifest, resumed_manifest, pack_manifest)
    if (any(item.get("protocol_manifest_sha256") != protocol_sha
            for item in manifests)
            or any(item.get("runtime_state") != "LEARNED_PACK_DISABLED"
                   or item.get("production_enabled") != 0
                   or item.get("mastery_claimed") != 0
                   for item in manifests)
            or fresh_manifest.get("semantic_result_sha256")
            != resumed_manifest.get("semantic_result_sha256")
            or fresh_manifest.get("semantic_result_sha256")
            != pack_manifest.get("semantic_result_sha256")
            or not _strict_equal(fresh_outputs, resumed_outputs)
            or not _strict_equal(fresh_outputs, pack_outputs)):
        raise BroadQaExternalDataError(
            "normalization recovery candidate lineage 漂移")
    lineages = pack_manifest.get("learner_lineages")
    if not isinstance(lineages, list) or len(lineages) != 2:
        raise BroadQaExternalDataError(
            "normalization recovery pack learner lineage 漂移")
    by_role = {item.get("role"): item for item in lineages
               if isinstance(item, dict)}
    if set(by_role) != {"FRESH", "RESUMED"}:
        raise BroadQaExternalDataError(
            "normalization recovery pack learner role 漂移")
    for role, manifest in (
            ("FRESH", fresh_manifest), ("RESUMED", resumed_manifest)):
        expected = {
            "checkpoint_chain_sha256": manifest["checkpoint_chain_sha256"],
            "checkpoint_terminal_sha256": manifest[
                "checkpoint_terminal_sha256"],
            "learner_manifest_sha256": manifest["manifest_sha256"],
            "resume_marker_count": manifest["resume_markers"]["record_count"],
            "role": role,
            "run_id": manifest["run_id"],
        }
        if not _strict_equal(by_role[role], expected):
            raise BroadQaExternalDataError(
                "normalization recovery pack/learner lineage 漂移")


def build_normalization_recovery_evaluation_family_freeze(
        *,
        repository_root: str | Path,
        evaluation_protocol_dir: str | Path,
        expected_evaluation_protocol_manifest_sha256: str,
        training_protocol_dir: str | Path,
        expected_training_protocol_manifest_sha256: str,
        fresh_learner_dir: str | Path,
        expected_fresh_learner_manifest_sha256: str,
        resumed_learner_dir: str | Path,
        expected_resumed_learner_manifest_sha256: str,
        rule_pack_dir: str | Path,
        expected_rule_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """重放候选与 live code，构造零 evaluation/reserve read 的 freeze。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise BroadQaExternalDataError(
            "normalization recovery repository root 非法")
    expected_evaluation_sha = _strict_sha(
        expected_evaluation_protocol_manifest_sha256,
        label="recovery expected evaluation manifest")
    expected_training_sha = _strict_sha(
        expected_training_protocol_manifest_sha256,
        label="recovery expected training manifest")
    expected_fresh_sha = _strict_sha(
        expected_fresh_learner_manifest_sha256,
        label="recovery expected fresh learner manifest")
    expected_resumed_sha = _strict_sha(
        expected_resumed_learner_manifest_sha256,
        label="recovery expected resumed learner manifest")
    expected_pack_sha = _strict_sha(
        expected_rule_pack_manifest_sha256,
        label="recovery expected rule pack manifest")
    evaluation_manifest = read_normalization_recovery_evaluation_manifest_only(
        evaluation_protocol_dir,
        expected_manifest_sha256=expected_evaluation_sha,
    )
    training_values = read_normalization_recovery_learner_input(
        training_protocol_dir,
        expected_manifest_sha256=expected_training_sha,
    )
    training_manifest = training_values[0]
    fresh_manifest, fresh_outputs = read_normalization_recovery_learner(
        fresh_learner_dir,
        protocol_dir=training_protocol_dir,
        expected_protocol_manifest_sha256=expected_training_sha,
    )
    resumed_manifest, resumed_outputs = read_normalization_recovery_learner(
        resumed_learner_dir,
        protocol_dir=training_protocol_dir,
        expected_protocol_manifest_sha256=expected_training_sha,
    )
    pack_manifest, pack_outputs = read_normalization_recovery_rule_pack(
        rule_pack_dir,
        protocol_dir=training_protocol_dir,
        expected_protocol_manifest_sha256=expected_training_sha,
        expected_pack_manifest_sha256=expected_pack_sha,
    )
    if (training_manifest["manifest_sha256"] != expected_training_sha
            or fresh_manifest["manifest_sha256"] != expected_fresh_sha
            or resumed_manifest["manifest_sha256"] != expected_resumed_sha
            or pack_manifest["manifest_sha256"] != expected_pack_sha):
        raise BroadQaExternalDataError(
            "normalization recovery external manifest identity 漂移")
    _validate_disabled_lineage(
        training_manifest=training_manifest,
        fresh_manifest=fresh_manifest,
        fresh_outputs=fresh_outputs,
        resumed_manifest=resumed_manifest,
        resumed_outputs=resumed_outputs,
        pack_manifest=pack_manifest,
        pack_outputs=pack_outputs,
    )
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation_manifest,
        rule_pack_manifest=pack_manifest,
        outputs=pack_outputs,
    )
    code_files, code_sha = _code_identity(repository)
    candidate_identity = {
        "candidate_program_sha256": program.sha256(),
        "conflict_count": len(program.conflicts),
        "fresh_checkpoint_chain_sha256": fresh_manifest[
            "checkpoint_chain_sha256"],
        "fresh_learner_manifest_sha256": fresh_manifest["manifest_sha256"],
        "generic_rule_count": len(program.generic_rules),
        "phrase_override_count": len(program.phrase_overrides),
        "regional_rule_count": len(program.regional_rules),
        "resumed_checkpoint_chain_sha256": resumed_manifest[
            "checkpoint_chain_sha256"],
        "resumed_learner_manifest_sha256": resumed_manifest[
            "manifest_sha256"],
        "rule_pack_manifest_sha256": pack_manifest["manifest_sha256"],
        "semantic_result_sha256": pack_manifest["semantic_result_sha256"],
        "source_replay_count": len(program.source_replays),
        "training_protocol_manifest_sha256": training_manifest[
            "manifest_sha256"],
        "training_work_identity_sha256": training_manifest[
            "learner_contract"]["work_identity_sha256"],
        "transfer_profile_sha256": program.transfer_profile.sha256(),
    }
    freeze_core = {
        "artifact_kind": NORMALIZATION_RECOVERY_EVALUATION_FAMILY_KIND,
        "candidate_freeze": candidate_identity,
        "candidate_freeze_sha256": _sha256(
            canonical_json_bytes(candidate_identity)),
        "candidate_pack_read_count": 1,
        "code_files": code_files,
        "code_freeze_sha256": code_sha,
        "evaluation_inventory_identity": evaluation_manifest[
            "evaluation_inventory"],
        "evaluation_payload_read_count": 0,
        "evaluation_protocol_manifest_sha256": evaluation_manifest[
            "manifest_sha256"],
        "evaluation_run_count": 0,
        "evaluation_source_pack_manifest_sha256": evaluation_manifest[
            "source_pack_manifest_sha256"],
        "format_version": 2,
        "mastery_claimed": 0,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "reserve_identity": evaluation_manifest["reserve_identity"],
        "reserve_payload_read_count": 0,
        "status": NORMALIZATION_RECOVERY_EVALUATION_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
    }
    return {
        **freeze_core,
        "family_commitment_sha256": _sha256(
            canonical_json_bytes(freeze_core)),
    }


def publish_normalization_recovery_evaluation_family_freeze(
        *,
        run_root: str | Path,
        target_dir: str | Path,
        **build_arguments: object,
        ) -> dict[str, object]:
    """不可覆盖发布 K 盘 family freeze，且不读取 evaluation payload。"""
    root = require_normalization_recovery_k_run_root(run_root)
    validate_normalization_recovery_evaluation_paths(root, build_arguments)
    target = normalization_recovery_path_within(
        root, target_dir, label="normalization recovery family target")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery evaluation family target 已存在")
    freeze = build_normalization_recovery_evaluation_family_freeze(
        **build_arguments)
    target.mkdir(parents=True)
    path = target / "family-freeze.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(freeze))
    return {**freeze, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_evaluation_family_freeze(
        target_dir: str | Path,
        **build_arguments: object,
        ) -> dict[str, object]:
    """严格回读 freeze 并重算 live candidate、code 与 protocol identity。"""
    root = Path(target_dir).resolve()
    try:
        payload = (root / "family-freeze.json").read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization recovery evaluation family freeze 不可读") from error
    expected = build_normalization_recovery_evaluation_family_freeze(
        **build_arguments)
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or not _strict_equal(value, expected)):
        raise BroadQaExternalDataError(
            "normalization recovery family freeze 与 live identity 漂移")
    return {**value, "manifest_sha256": _sha256(payload)}


def main(argv: list[str] | None = None) -> int:
    """不可覆盖发布或严格回读 recovery evaluation family freeze。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("publish", "read"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--target-dir", required=True)
    for name in NORMALIZATION_RECOVERY_EVALUATION_DATA_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in NORMALIZATION_RECOVERY_EVALUATION_IDENTITY_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    run_root = arguments.pop("run_root")
    target = arguments.pop("target_dir")
    root = require_normalization_recovery_k_run_root(run_root)
    validate_normalization_recovery_evaluation_paths(root, arguments)
    target = normalization_recovery_path_within(
        root, target, label="normalization recovery family target")
    if command == "publish":
        report = publish_normalization_recovery_evaluation_family_freeze(
            run_root=root, target_dir=target, **arguments)
    else:
        report = read_normalization_recovery_evaluation_family_freeze(
            target, **arguments)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_RECOVERY_EVALUATION_CODE_FILES",
    "NORMALIZATION_RECOVERY_EVALUATION_DATA_ARGUMENTS",
    "NORMALIZATION_RECOVERY_EVALUATION_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_EVALUATION_FAMILY_STATUS",
    "NORMALIZATION_RECOVERY_EVALUATION_IDENTITY_ARGUMENTS",
    "build_normalization_recovery_evaluation_family_freeze",
    "normalization_recovery_path_within",
    "publish_normalization_recovery_evaluation_family_freeze",
    "read_normalization_recovery_evaluation_family_freeze",
    "require_normalization_recovery_k_run_root",
    "validate_normalization_recovery_evaluation_paths",
]
