"""封存来源对齐广域问答算法，并约束唯一一次正式评测。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


FORMAL_FREEZE_NAME = "formal-algorithm-freeze.json"
FORMAL_INTENT_NAME = "formal-run-intent.json"
FORMAL_OUTCOME_NAME = "formal-run-outcome.json"
FORMAL_FREEZE_KIND = "PH2_BROAD_QA_FORMAL_ALGORITHM_FREEZE_V1"
FORMAL_INTENT_KIND = "PH2_BROAD_QA_FORMAL_RUN_INTENT_V1"
FORMAL_OUTCOME_KIND = "PH2_BROAD_QA_FORMAL_RUN_OUTCOME_V1"

_CODE_RELATIVE_PATHS = (
    "data/ph2/broad_qa_question_slots_v1.json",
    "src/pure_integer_ai/experiments/ph2_broad_qa_contract.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_external_data.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_formal_protocol.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_index.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_joint_eval.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_query.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_question_slots.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_selection.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_aligned_family.py",
    "src/pure_integer_ai/experiments/ph2_broad_qa_source_alignment.py",
    "src/pure_integer_ai/experiments/run_ph2_broad_qa_joint_eval.py",
    "src/pure_integer_ai/storage/integer_codec.py",
)
_ARTIFACT_ROLES = (
    "alias_ledger",
    "candidate_manifest",
    "census",
    "census_manifest",
    "database",
    "dev_aggregate",
    "dev_labels",
    "dev_questions",
    "family_manifest",
    "held_out_labels",
    "held_out_questions",
    "runtime_source_manifest",
    "source_targets",
    "terminal_selection",
)


def _sha256_file(path: Path) -> str:
    """流式计算 formal 输入、代码或结果的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, object]:
    """读取单行 canonical JSON object，拒绝宽松或重复编码。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"formal JSON 非法: {path.name}") from error
    if (not isinstance(value, dict)
            or canonical_json_line(value) != payload):
        raise BroadQaExternalDataError(
            f"formal JSON 非 canonical: {path.name}")
    return value


def _identity(run_root: Path, role: str, path: Path) -> dict[str, object]:
    """形成一个严格位于 run root 内的文件身份。"""
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(run_root):
        raise BroadQaExternalDataError(f"formal {role} 路径非法")
    return {
        "bytes": resolved.stat().st_size,
        "role": role,
        "run_root_relative_path": resolved.relative_to(run_root).as_posix(),
        "sha256": _sha256_file(resolved),
    }


def _resolve_relative(run_root: Path, value: object) -> Path:
    """把 receipt 中的 POSIX 相对路径安全恢复到显式 run root。"""
    if (not isinstance(value, str) or not value
            or Path(value).is_absolute() or "\\" in value):
        raise BroadQaExternalDataError("formal run-root 相对路径非法")
    result = (run_root / Path(*value.split("/"))).resolve()
    if not result.is_relative_to(run_root):
        raise BroadQaExternalDataError("formal artifact 越出 run root")
    return result


def _git_value(repository_root: Path, arguments: Iterable[str]) -> str:
    """读取本地 Git 身份，不调用网络且不写配置。"""
    try:
        result = subprocess.run(
            ("git", *arguments), cwd=repository_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise BroadQaExternalDataError("formal Git 身份不可用") from error
    return result.stdout.strip()


def _repository_identity(repository_root: Path) -> dict[str, object]:
    """返回 HEAD、公开远端跟踪身份和工作树清洁状态。"""
    root = repository_root.resolve()
    head = _git_value(root, ("rev-parse", "HEAD"))
    origin = _git_value(root, ("rev-parse", "origin/master"))
    dirty = _git_value(root, ("status", "--porcelain=v1"))
    if (len(head) != 40 or len(origin) != 40
            or any(character not in "0123456789abcdef" for character in head)
            or any(character not in "0123456789abcdef" for character in origin)):
        raise BroadQaExternalDataError("formal Git commit 身份非法")
    return {"clean": int(not dirty), "head": head, "origin_master": origin}


def _code_bindings(repository_root: Path) -> list[dict[str, object]]:
    """冻结会影响检索、选择、评分和协议的公开代码闭包。"""
    result = []
    for relative_path in _CODE_RELATIVE_PATHS:
        path = repository_root / Path(*relative_path.split("/"))
        if not path.is_file():
            raise BroadQaExternalDataError(
                f"formal code binding 缺失: {relative_path}")
        result.append({
            "bytes": path.stat().st_size,
            "relative_path": relative_path,
            "sha256": _sha256_file(path),
        })
    return result


def _artifact_map(receipt: dict[str, object]) -> dict[str, dict[str, object]]:
    """把严格排序且角色完整的 artifact binding 恢复为映射。"""
    raw = receipt.get("artifacts")
    if not isinstance(raw, list):
        raise BroadQaExternalDataError("formal artifact bindings 缺失")
    result = {}
    for item in raw:
        if (not isinstance(item, dict)
                or set(item) != {
                    "bytes", "role", "run_root_relative_path", "sha256"}
                or not isinstance(item.get("role"), str)
                or item["role"] in result):
            raise BroadQaExternalDataError("formal artifact binding 漂移")
        result[item["role"]] = item
    if tuple(sorted(result)) != _ARTIFACT_ROLES:
        raise BroadQaExternalDataError("formal artifact role inventory 漂移")
    return result


def _validate_freeze_schema(value: dict[str, object]) -> None:
    """核验 formal freeze 顶层字段和不可变状态。"""
    if (set(value) != {
            "algorithm_commit", "artifact_kind", "artifacts",
            "code_bindings", "formal_outputs", "format_version", "status",
            "unique_formal_run_limit"}
            or value.get("artifact_kind") != FORMAL_FREEZE_KIND
            or value.get("format_version") != 1
            or value.get("status") != "FROZEN_NOT_RUN"
            or value.get("unique_formal_run_limit") != 1):
        raise BroadQaExternalDataError("formal freeze receipt schema 漂移")


def _validate_freeze_code(
        value: dict[str, object], repository: Path) -> None:
    """回验算法文件闭包及本地、远端提交身份。"""
    if value.get("code_bindings") != _code_bindings(repository):
        raise BroadQaExternalDataError("formal code binding 漂移")
    repository_state = _repository_identity(repository)
    if (repository_state["head"] != value.get("algorithm_commit")
            or repository_state["origin_master"] != value.get(
                "algorithm_commit")):
        raise BroadQaExternalDataError("formal algorithm commit 漂移")


def _validate_outputs(
        value: dict[str, object], run_root: Path) -> dict[str, object]:
    """恢复并核验冻结的 predictions/aggregate 唯一路径。"""
    outputs = value.get("formal_outputs")
    if (not isinstance(outputs, dict) or set(outputs) != {
            "aggregate_run_root_relative_path",
            "predictions_run_root_relative_path"}):
        raise BroadQaExternalDataError("formal output binding 漂移")
    aggregate = _resolve_relative(
        run_root, outputs["aggregate_run_root_relative_path"])
    predictions = _resolve_relative(
        run_root, outputs["predictions_run_root_relative_path"])
    if aggregate == predictions:
        raise BroadQaExternalDataError("formal output identity 冲突")
    return outputs


def _validate_family_chain(
        artifacts: dict[str, dict[str, object]], run_root: Path) -> None:
    """交叉核对 family、来源、开发门和实际运行 artifact 身份。"""
    paths = {
        role: _resolve_relative(run_root, item["run_root_relative_path"])
        for role, item in artifacts.items()
    }
    family = _canonical_object(paths["family_manifest"])
    runtime = _canonical_object(paths["runtime_source_manifest"])
    dev = _canonical_object(paths["dev_aggregate"])
    family_roles = {
        item.get("role"): item
        for item in family.get("artifacts", [])
        if isinstance(item, dict) and isinstance(item.get("role"), str)
    }
    expected_family_roles = {
        "dev_labels", "dev_questions", "held_out_labels",
        "held_out_questions", "source_targets",
    }
    if set(family_roles) != expected_family_roles:
        raise BroadQaExternalDataError("formal family artifact inventory 漂移")
    for role in expected_family_roles:
        if family_roles[role].get("sha256") != artifacts[role]["sha256"]:
            raise BroadQaExternalDataError(
                f"formal family {role} binding 漂移")
    if (family.get("status") != "FROZEN_NOT_RUN"
            or family.get("census_sha256") != artifacts["census"]["sha256"]
            or family.get("candidate_manifest_sha256")
            != artifacts["candidate_manifest"]["sha256"]
            or family.get("census_manifest_sha256")
            != artifacts["census_manifest"]["sha256"]):
        raise BroadQaExternalDataError("formal family 来源冻结链漂移")
    if (runtime.get("alias_sha256") != artifacts["alias_ledger"]["sha256"]
            or runtime.get("source_targets_sha256")
            != artifacts["source_targets"]["sha256"]
            or runtime.get("terminal_selection_sha256")
            != artifacts["terminal_selection"]["sha256"]):
        raise BroadQaExternalDataError("formal runtime source binding 漂移")
    if (dev.get("scope") != "DEVELOPMENT" or dev.get("status") != "PASS"
            or dev.get("questions_sha256")
            != artifacts["dev_questions"]["sha256"]
            or dev.get("labels_sha256") != artifacts["dev_labels"]["sha256"]
            or dev.get("database_sha256") != artifacts["database"]["sha256"]
            or dev.get("alias_sha256") != artifacts["alias_ledger"]["sha256"]
            or dev.get("target_selection_sha256")
            != artifacts["terminal_selection"]["sha256"]
            or dev.get("thresholds") != family.get("thresholds")):
        raise BroadQaExternalDataError("formal development gate binding 漂移")


def publish_formal_algorithm_freeze(
        run_root: str | Path,
        family_root: str | Path,
        *,
        candidate_manifest_path: str | Path,
        census_path: str | Path,
        census_manifest_path: str | Path,
        dev_aggregate_path: str | Path,
        database_path: str | Path,
        alias_path: str | Path,
        terminal_selection_path: str | Path,
        runtime_source_manifest_path: str | Path,
        predictions_path: str | Path,
        aggregate_path: str | Path,
        repository_root: str | Path,
        ) -> dict[str, object]:
    """在 clean、已推送提交上首次发布完整算法与输入冻结 receipt。"""
    root = Path(run_root).resolve()
    family_dir = Path(family_root).resolve()
    repository = Path(repository_root).resolve()
    if (not root.is_dir() or not family_dir.is_dir()
            or not family_dir.is_relative_to(root)):
        raise BroadQaExternalDataError("formal family/run root 非法")
    target = family_dir / FORMAL_FREEZE_NAME
    predictions = Path(predictions_path).resolve()
    aggregate = Path(aggregate_path).resolve()
    if (target.exists() or predictions.exists() or aggregate.exists()
            or not predictions.is_relative_to(root)
            or not aggregate.is_relative_to(root)
            or predictions == aggregate):
        raise BroadQaExternalDataError("formal freeze 输出边界非法")
    artifact_paths = {
        "alias_ledger": Path(alias_path),
        "candidate_manifest": Path(candidate_manifest_path),
        "census": Path(census_path),
        "census_manifest": Path(census_manifest_path),
        "database": Path(database_path),
        "dev_aggregate": Path(dev_aggregate_path),
        "dev_labels": family_dir / "dev.labels.jsonl",
        "dev_questions": family_dir / "dev.questions.jsonl",
        "family_manifest": family_dir / "manifest.json",
        "held_out_labels": family_dir / "held_out.labels.jsonl",
        "held_out_questions": family_dir / "held_out.questions.jsonl",
        "runtime_source_manifest": Path(runtime_source_manifest_path),
        "source_targets": family_dir / "source_targets.jsonl",
        "terminal_selection": Path(terminal_selection_path),
    }
    artifacts = [
        _identity(root, role, artifact_paths[role])
        for role in _ARTIFACT_ROLES
    ]
    artifact_by_role = {item["role"]: item for item in artifacts}
    _validate_family_chain(artifact_by_role, root)
    repository_state = _repository_identity(repository)
    if (repository_state["clean"] != 1
            or repository_state["head"] != repository_state["origin_master"]):
        raise BroadQaExternalDataError(
            "formal algorithm freeze 要求 clean 且 HEAD=origin/master")
    value = {
        "algorithm_commit": repository_state["head"],
        "artifact_kind": FORMAL_FREEZE_KIND,
        "artifacts": artifacts,
        "code_bindings": _code_bindings(repository),
        "formal_outputs": {
            "aggregate_run_root_relative_path": (
                aggregate.relative_to(root).as_posix()),
            "predictions_run_root_relative_path": (
                predictions.relative_to(root).as_posix()),
        },
        "format_version": 1,
        "status": "FROZEN_NOT_RUN",
        "unique_formal_run_limit": 1,
    }
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_line(value))
    except FileExistsError as error:
        raise BroadQaExternalDataError(
            "formal algorithm freeze 禁止重复发布") from error
    return {**value, "freeze_sha256": _sha256_file(target)}


def verify_formal_algorithm_freeze(
        run_root: str | Path,
        freeze_path: str | Path,
        *, repository_root: str | Path) -> dict[str, object]:
    """回验算法提交、代码闭包、全部输入及开发门均未漂移。"""
    root = Path(run_root).resolve()
    target = Path(freeze_path).resolve()
    repository = Path(repository_root).resolve()
    if (not root.is_dir() or target.name != FORMAL_FREEZE_NAME
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError("formal freeze receipt 路径非法")
    value = _canonical_object(target)
    _validate_freeze_schema(value)
    artifacts = _artifact_map(value)
    for item in artifacts.values():
        path = _resolve_relative(root, item["run_root_relative_path"])
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or _sha256_file(path) != item["sha256"]):
            raise BroadQaExternalDataError(
                f"formal artifact 漂移: {item['role']}")
    _validate_family_chain(artifacts, root)
    _validate_freeze_code(value, repository)
    _validate_outputs(value, root)
    return {**value, "freeze_sha256": _sha256_file(target)}


def _verify_prediction_freeze(
        run_root: Path,
        freeze_path: Path,
        repository_root: Path,
        ) -> dict[str, object]:
    """预测前回验算法和必要输入，但绝不读取 held-out labels。"""
    if (not run_root.is_dir() or freeze_path.name != FORMAL_FREEZE_NAME
            or not freeze_path.is_relative_to(run_root)):
        raise BroadQaExternalDataError("formal freeze receipt 路径非法")
    value = _canonical_object(freeze_path)
    _validate_freeze_schema(value)
    artifacts = _artifact_map(value)
    for role in ("database", "family_manifest", "held_out_questions"):
        item = artifacts[role]
        path = _resolve_relative(run_root, item["run_root_relative_path"])
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or _sha256_file(path) != item["sha256"]):
            raise BroadQaExternalDataError(f"formal artifact 漂移: {role}")
    _validate_freeze_code(value, repository_root)
    _validate_outputs(value, run_root)
    return {**value, "freeze_sha256": _sha256_file(freeze_path)}


def publish_formal_run_intent(
        run_root: str | Path,
        freeze_path: str | Path,
        *, repository_root: str | Path) -> dict[str, object]:
    """以固定 sibling 路径原子占用 family 的唯一 formal run。"""
    root = Path(run_root).resolve()
    freeze = Path(freeze_path).resolve()
    verified = verify_formal_algorithm_freeze(
        root, freeze, repository_root=repository_root)
    target = freeze.parent / FORMAL_INTENT_NAME
    outcome = freeze.parent / FORMAL_OUTCOME_NAME
    outputs = verified["formal_outputs"]
    if (target.exists() or outcome.exists()
            or _resolve_relative(
                root, outputs["predictions_run_root_relative_path"]).exists()
            or _resolve_relative(
                root, outputs["aggregate_run_root_relative_path"]).exists()):
        raise BroadQaExternalDataError("formal run 已占用或已有输出")
    artifacts = _artifact_map(verified)
    value = {
        "algorithm_commit": verified["algorithm_commit"],
        "artifact_kind": FORMAL_INTENT_KIND,
        "family_manifest_sha256": artifacts["family_manifest"]["sha256"],
        "format_version": 1,
        "freeze_sha256": verified["freeze_sha256"],
        "held_out_questions_sha256": (
            artifacts["held_out_questions"]["sha256"]),
        "status": "OUTCOME_PENDING",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_line(value))
    except FileExistsError as error:
        raise BroadQaExternalDataError("formal run intent 禁止重复") from error
    return {**value, "intent_sha256": _sha256_file(target)}


def _verified_intent(
        run_root: Path,
        freeze_path: Path,
        intent_path: Path,
        repository_root: Path,
        *, prediction_only: bool = False,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """共同回验 immutable freeze 与 OUTCOME_PENDING intent。"""
    freeze = (
        _verify_prediction_freeze(run_root, freeze_path, repository_root)
        if prediction_only else verify_formal_algorithm_freeze(
            run_root, freeze_path, repository_root=repository_root)
    )
    if (intent_path.resolve() != freeze_path.resolve().parent / FORMAL_INTENT_NAME
            or not intent_path.is_file()):
        raise BroadQaExternalDataError("formal intent 路径非法")
    intent = _canonical_object(intent_path)
    artifacts = _artifact_map(freeze)
    expected = {
        "algorithm_commit": freeze["algorithm_commit"],
        "artifact_kind": FORMAL_INTENT_KIND,
        "family_manifest_sha256": artifacts["family_manifest"]["sha256"],
        "format_version": 1,
        "freeze_sha256": freeze["freeze_sha256"],
        "held_out_questions_sha256": (
            artifacts["held_out_questions"]["sha256"]),
        "status": "OUTCOME_PENDING",
    }
    if intent != expected:
        raise BroadQaExternalDataError("formal intent 漂移")
    return freeze, {**intent, "intent_sha256": _sha256_file(intent_path)}


def verify_formal_prediction_authorization(
        run_root: str | Path,
        freeze_path: str | Path,
        intent_path: str | Path,
        *,
        questions_path: str | Path,
        database_path: str | Path,
        predictions_path: str | Path,
        repository_root: str | Path,
        ) -> dict[str, object]:
    """在读取数据库前验证 held-out 预测确属唯一 formal run。"""
    root = Path(run_root).resolve()
    freeze, intent = _verified_intent(
        root, Path(freeze_path).resolve(), Path(intent_path).resolve(),
        Path(repository_root).resolve(), prediction_only=True)
    artifacts = _artifact_map(freeze)
    outputs = freeze["formal_outputs"]
    questions = Path(questions_path).resolve()
    database = Path(database_path).resolve()
    predictions = Path(predictions_path).resolve()
    if (questions != _resolve_relative(
                root, artifacts["held_out_questions"][
                    "run_root_relative_path"])
            or database != _resolve_relative(
                root, artifacts["database"]["run_root_relative_path"])
            or predictions != _resolve_relative(
                root, outputs["predictions_run_root_relative_path"])
            or predictions.exists()
            or _resolve_relative(
                root, outputs["aggregate_run_root_relative_path"]).exists()
            or (Path(freeze_path).resolve().parent
                / FORMAL_OUTCOME_NAME).exists()):
        raise BroadQaExternalDataError("formal prediction authorization 不匹配")
    return {
        "freeze_sha256": freeze["freeze_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "status": "AUTHORIZED",
    }


def verify_formal_score_authorization(
        run_root: str | Path,
        freeze_path: str | Path,
        intent_path: str | Path,
        *,
        questions_path: str | Path,
        predictions_path: str | Path,
        labels_path: str | Path,
        database_path: str | Path,
        alias_path: str | Path,
        terminal_selection_path: str | Path,
        aggregate_path: str | Path,
        repository_root: str | Path,
        ) -> dict[str, object]:
    """在解析任何正式标签前回验评分所需的完整冻结链。"""
    root = Path(run_root).resolve()
    freeze, intent = _verified_intent(
        root, Path(freeze_path).resolve(), Path(intent_path).resolve(),
        Path(repository_root).resolve())
    artifacts = _artifact_map(freeze)
    outputs = freeze["formal_outputs"]
    expected = {
        "alias_ledger": Path(alias_path).resolve(),
        "database": Path(database_path).resolve(),
        "held_out_labels": Path(labels_path).resolve(),
        "held_out_questions": Path(questions_path).resolve(),
        "terminal_selection": Path(terminal_selection_path).resolve(),
    }
    if any(path != _resolve_relative(
            root, artifacts[role]["run_root_relative_path"])
            for role, path in expected.items()):
        raise BroadQaExternalDataError("formal score input authorization 不匹配")
    predictions = Path(predictions_path).resolve()
    aggregate = Path(aggregate_path).resolve()
    if (predictions != _resolve_relative(
                root, outputs["predictions_run_root_relative_path"])
            or not predictions.is_file()
            or aggregate != _resolve_relative(
                root, outputs["aggregate_run_root_relative_path"])
            or aggregate.exists()
            or (Path(freeze_path).resolve().parent
                / FORMAL_OUTCOME_NAME).exists()):
        raise BroadQaExternalDataError("formal score output authorization 不匹配")
    return {
        "freeze_sha256": freeze["freeze_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "predictions_sha256": _sha256_file(predictions),
        "status": "AUTHORIZED",
    }


def publish_formal_run_outcome(
        run_root: str | Path,
        freeze_path: str | Path,
        intent_path: str | Path,
        aggregate_path: str | Path,
        *, repository_root: str | Path) -> dict[str, object]:
    """从已写 aggregate 发布不可覆盖的 PASS/FAIL formal outcome。"""
    root = Path(run_root).resolve()
    freeze_path_value = Path(freeze_path).resolve()
    freeze, intent = _verified_intent(
        root, freeze_path_value, Path(intent_path).resolve(),
        Path(repository_root).resolve())
    outputs = freeze["formal_outputs"]
    aggregate_file = Path(aggregate_path).resolve()
    if aggregate_file != _resolve_relative(
            root, outputs["aggregate_run_root_relative_path"]):
        raise BroadQaExternalDataError("formal aggregate 路径漂移")
    aggregate = _canonical_object(aggregate_file)
    artifacts = _artifact_map(freeze)
    if (aggregate.get("scope") != "FORMAL_HELD_OUT"
            or aggregate.get("status") not in {"PASS", "FAIL"}
            or aggregate.get("questions_sha256")
            != artifacts["held_out_questions"]["sha256"]
            or aggregate.get("labels_sha256")
            != artifacts["held_out_labels"]["sha256"]
            or aggregate.get("database_sha256")
            != artifacts["database"]["sha256"]
            or aggregate.get("alias_sha256")
            != artifacts["alias_ledger"]["sha256"]
            or aggregate.get("target_selection_sha256")
            != artifacts["terminal_selection"]["sha256"]):
        raise BroadQaExternalDataError("formal aggregate binding 漂移")
    target = freeze_path_value.parent / FORMAL_OUTCOME_NAME
    value = {
        "aggregate_sha256": _sha256_file(aggregate_file),
        "algorithm_commit": freeze["algorithm_commit"],
        "artifact_kind": FORMAL_OUTCOME_KIND,
        "evidence_hit_ppm": aggregate.get("evidence_hit_ppm"),
        "format_version": 1,
        "freeze_sha256": freeze["freeze_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "question_count": aggregate.get("question_count"),
        "recall_at_20_ppm": aggregate.get("recall_at_20_ppm"),
        "status": aggregate["status"],
        "top1_source_hit_ppm": aggregate.get("top1_source_hit_ppm"),
    }
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_line(value))
    except FileExistsError as error:
        raise BroadQaExternalDataError("formal outcome 禁止重复发布") from error
    return {**value, "outcome_sha256": _sha256_file(target)}


__all__ = [
    "FORMAL_FREEZE_KIND",
    "FORMAL_FREEZE_NAME",
    "FORMAL_INTENT_KIND",
    "FORMAL_INTENT_NAME",
    "FORMAL_OUTCOME_KIND",
    "FORMAL_OUTCOME_NAME",
    "publish_formal_algorithm_freeze",
    "publish_formal_run_intent",
    "publish_formal_run_outcome",
    "verify_formal_algorithm_freeze",
    "verify_formal_prediction_authorization",
    "verify_formal_score_authorization",
]
