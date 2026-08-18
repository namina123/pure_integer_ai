"""``CONFLICT_SET`` family freeze、prediction seal 与唯一 formal runner。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_family import (
    FAMILY_FREEZE_MANIFEST_NAME,
    ConflictSetFamilyFreeze,
    assert_conflict_set_family_freeze_matches_live_public_code,
    parse_conflict_set_family_freeze_bytes,
    read_conflict_set_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_private import (
    ConflictSetPrivateLabelRead,
    read_conflict_set_semantic_labels_after_prediction_seal,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
    ConflictSetSemanticFormalAggregate,
    ConflictSetSemanticPredictionRecord,
    ConflictSetSemanticPredictionSeal,
    build_conflict_set_semantic_formal_aggregate,
    build_conflict_set_semantic_prediction_record,
    build_conflict_set_semantic_prediction_seal,
    build_conflict_set_semantic_unavailable_aggregate,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_private_protocol import (
    TRANSPORT_ROOT_NAMESPACE,
    ConflictSetPrivateArtifact,
    ConflictSetRunGuard,
    ConflictSetRunIntent,
    consume_conflict_set_run_guard,
    strict_conflict_set_relative_path,
)


CONFLICT_SET_GUARD_AVAILABLE_NAME = "guard.available.json"
CONFLICT_SET_GUARD_CONSUMED_NAME = "guard.consumed.json"
CONFLICT_SET_RUN_INTENT_NAME = "run.intent.json"
CONFLICT_SET_PREDICTION_SEAL_NAME = "predictions.seal.json"
CONFLICT_SET_PUBLICATION_NAME = "publication"
CONFLICT_SET_OUTCOME_NAME = "run.outcome.json"
CONFLICT_SET_AGGREGATE_NAME = "aggregate.json"
CONFLICT_SET_DECISION_NAME = "decision.json"
CONFLICT_SET_RUNTIME_RECEIPT_NAME = "runtime_receipt.json"
CONFLICT_SET_FAILURE_SEAL_NAME = "failure_seal.json"


# object-model: exception
class ConflictSetFormalRunnerError(ValueError):
    """CONFLICT_SET formal family 运行或 publication 违反不可覆盖合同。"""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_k_run_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise ConflictSetFormalRunnerError("formal run root 必须位于 K 盘")
    return root


def _family_path(run_root: Path, family_root: str | Path) -> Path:
    family = Path(family_root).resolve()
    if (not family.is_relative_to(run_root)
            or family == run_root or family.is_symlink()):
        raise ConflictSetFormalRunnerError(
            "formal family root 必须位于 run root 内且不可为 symlink")
    return family


def _immutable_jsonl(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ConflictSetFormalRunnerError("immutable artifact 已存在")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ConflictSetFormalRunnerError("artifact 必须是单条 canonical JSONL")
    with path.open("xb") as handle:
        handle.write(payload)


def _artifact(freeze: ConflictSetFamilyFreeze, role: str) -> ConflictSetPrivateArtifact:
    matches = tuple(item for item in freeze.transport.artifacts
                    if item.role == role)
    if len(matches) != 1:
        raise ConflictSetFormalRunnerError(f"family freeze 缺唯一 {role} artifact")
    return matches[0]


def _transport_path(
        run_root: Path, artifact: ConflictSetPrivateArtifact,
        ) -> Path:
    relative = strict_conflict_set_relative_path(
        artifact.relative_path, where=f"{artifact.role}.relative_path")
    target = (run_root / Path(*relative.split("/"))).resolve()
    namespace = (run_root / TRANSPORT_ROOT_NAMESPACE).resolve()
    if (not target.is_relative_to(namespace)
            or target.is_symlink()):
        raise ConflictSetFormalRunnerError(
            f"{artifact.role} artifact path 越界或为 symlink")
    return target


def publish_conflict_set_family_freeze(
        *,
        run_root: str | Path,
        family_root: str | Path,
        freeze: ConflictSetFamilyFreeze,
        ) -> Path:
    """在新 formal family 目录不可覆盖发布 freeze 与 AVAILABLE guard。"""
    if not isinstance(freeze, ConflictSetFamilyFreeze):
        raise TypeError("family freeze 类型错误")
    run = _require_k_run_root(run_root)
    family = _family_path(run, family_root)
    if family.exists():
        raise ConflictSetFormalRunnerError("formal family target 已存在")
    family.parent.mkdir(parents=True, exist_ok=True)
    family.mkdir()
    try:
        _immutable_jsonl(
            family / FAMILY_FREEZE_MANIFEST_NAME,
            freeze.canonical_bytes(),
        )
        write_immutable_json(
            freeze.available_guard.to_dict(),
            family / CONFLICT_SET_GUARD_AVAILABLE_NAME,
        )
        parsed = read_conflict_set_family_freeze(
            family / FAMILY_FREEZE_MANIFEST_NAME)
        if parsed != freeze or read_canonical_object(
                family / CONFLICT_SET_GUARD_AVAILABLE_NAME
        ) != freeze.available_guard.to_dict():
            raise ConflictSetFormalRunnerError("family freeze/guard 回读漂移")
    except Exception:
        shutil.rmtree(family)
        raise
    return family


def _consume_guard(
        family: Path, freeze: ConflictSetFamilyFreeze,
        ) -> tuple[ConflictSetRunGuard, ConflictSetRunIntent]:
    available_path = family / CONFLICT_SET_GUARD_AVAILABLE_NAME
    consumed_path = family / CONFLICT_SET_GUARD_CONSUMED_NAME
    intent_path = family / CONFLICT_SET_RUN_INTENT_NAME
    if (not available_path.is_file() or consumed_path.exists()
            or intent_path.exists()):
        raise ConflictSetFormalRunnerError("formal guard 已消费或缺失")
    try:
        available = ConflictSetRunGuard.from_dict(
            read_canonical_object(available_path))
    except Exception as error:
        raise ConflictSetFormalRunnerError("AVAILABLE guard 无法回读") from error
    if available != freeze.available_guard:
        raise ConflictSetFormalRunnerError("AVAILABLE guard identity 漂移")
    consumed, intent = consume_conflict_set_run_guard(available)
    write_immutable_json(consumed.to_dict(), consumed_path)
    write_immutable_json(intent.to_dict(), intent_path)
    available_path.unlink()
    if (ConflictSetRunGuard.from_dict(read_canonical_object(consumed_path))
            != consumed
            or ConflictSetRunIntent.from_dict(read_canonical_object(intent_path))
            != intent):
        raise ConflictSetFormalRunnerError("consumed guard lineage 回读漂移")
    return consumed, intent


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetActualRun:
    """一个 label-free actual executor 返回的安全语义结果。"""

    observation_stable_key_sha256: str
    candidate_identity_sha256: str
    status: str
    projection: object | None
    teacher_call_count: int = 0
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
            ConflictSetSemanticProjection,
        )
        from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_formal_protocol import (
            CONFLICT_SET_FORMAL_LABEL_STATUSES,
        )
        if self.status not in CONFLICT_SET_FORMAL_LABEL_STATUSES:
            raise ConflictSetFormalRunnerError("actual status 非法")
        if self.status == "NE" and self.projection is not None:
            raise ConflictSetFormalRunnerError("NE actual 不得携 projection")
        if self.status != "NE" and not isinstance(
                self.projection, ConflictSetSemanticProjection):
            raise ConflictSetFormalRunnerError(
                "PASS/FAIL actual 必须携 typed projection")
        if type(self.teacher_call_count) is not int or self.teacher_call_count != 0:
            raise ConflictSetFormalRunnerError("teacher call count 必须为零")
        if (type(self.host_learning_write_count) is not int
                or self.host_learning_write_count != 0):
            raise ConflictSetFormalRunnerError("host learning write count 必须为零")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConflictSetFormalPublication:
    """唯一 formal run 的安全内存 publication 投影。"""

    aggregate: ConflictSetSemanticFormalAggregate
    prediction: ConflictSetSemanticPredictionSeal
    decision: dict[str, object]
    runtime_receipt: dict[str, object] | None
    failure_seal: dict[str, object] | None
    label_read: ConflictSetPrivateLabelRead | None

    def __post_init__(self) -> None:
        if self.aggregate.status == "PASS":
            if self.runtime_receipt is None or self.failure_seal is not None:
                raise ConflictSetFormalRunnerError(
                    "PASS publication receipt/seal 投影漂移")
        elif self.runtime_receipt is not None or self.failure_seal is None:
            raise ConflictSetFormalRunnerError(
                "non-PASS publication receipt/seal 投影漂移")


def _safe_publication(
        aggregate: ConflictSetSemanticFormalAggregate,
        prediction: ConflictSetSemanticPredictionSeal,
        ) -> tuple[dict[str, object], dict[str, object] | None,
                   dict[str, object] | None]:
    decision = {
        "aggregate_sha256": aggregate.sha256(),
        "artifact_kind": "PH2_GG03_CONFLICT_SET_FORMAL_DECISION_V1",
        "family_commitment_sha256": aggregate.family_commitment_sha256,
        "format_version": 1,
        "prediction_seal_sha256": prediction.sha256(),
        "status": aggregate.status,
    }
    if aggregate.status == "PASS":
        return decision, {
            "aggregate_sha256": aggregate.sha256(),
            "artifact_kind": "PH2_GG03_CONFLICT_SET_RUNTIME_RECEIPT_V1",
            "family_commitment_sha256": aggregate.family_commitment_sha256,
            "format_version": 1,
            "formal_run_count": 1,
            "status": "RUNTIME_EVIDENCED",
        }, None
    return decision, None, {
        "aggregate_sha256": aggregate.sha256(),
        "artifact_kind": "PH2_GG03_CONFLICT_SET_FAILURE_SEAL_V1",
        "failure_phase": aggregate.failure_phase,
        "format_version": 1,
        "formal_run_count": 1,
        "status": aggregate.status,
    }


def _publish_result(
        family: Path,
        aggregate: ConflictSetSemanticFormalAggregate,
        prediction: ConflictSetSemanticPredictionSeal,
        label_read: ConflictSetPrivateLabelRead | None,
        ) -> ConflictSetFormalPublication:
    publication = family / CONFLICT_SET_PUBLICATION_NAME
    outcome_path = family / CONFLICT_SET_OUTCOME_NAME
    if publication.exists() or outcome_path.exists():
        raise ConflictSetFormalRunnerError("formal publication 已存在")
    staging = Path(tempfile.mkdtemp(
        prefix=".conflict-set-publication-building-", dir=family))
    decision, receipt, failure = _safe_publication(aggregate, prediction)
    try:
        write_immutable_json(aggregate.to_dict(), staging / CONFLICT_SET_AGGREGATE_NAME)
        write_immutable_json(decision, staging / CONFLICT_SET_DECISION_NAME)
        if receipt is not None:
            write_immutable_json(
                receipt, staging / CONFLICT_SET_RUNTIME_RECEIPT_NAME)
        if failure is not None:
            write_immutable_json(
                failure, staging / CONFLICT_SET_FAILURE_SEAL_NAME)
        outcome = {
            "aggregate_sha256": aggregate.sha256(),
            "artifact_kind": "PH2_GG03_CONFLICT_SET_RUN_OUTCOME_V1",
            "failure_phase": aggregate.failure_phase,
            "format_version": 1,
            "prediction_seal_sha256": prediction.sha256(),
            "status": aggregate.status,
        }
        _immutable_jsonl(outcome_path, canonical_json_bytes(outcome) + b"\n")
        os.replace(staging, publication)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if outcome_path.exists() and not publication.exists():
            outcome_path.unlink()
        raise
    if read_canonical_object(
            publication / CONFLICT_SET_AGGREGATE_NAME) != aggregate.to_dict():
        raise ConflictSetFormalRunnerError("aggregate publication 回读漂移")
    return ConflictSetFormalPublication(
        aggregate, prediction, decision, receipt, failure, label_read)


def run_conflict_set_formal_evaluation_once(
        *,
        repository_root: str | Path,
        run_root: str | Path,
        family_root: str | Path,
        freeze: ConflictSetFamilyFreeze,
        observation_keys: tuple[str, ...],
        actual_executor: Callable[[str], ConflictSetActualRun],
        ) -> ConflictSetFormalPublication:
    """消费唯一 guard，seal label-free prediction，随后读取 label 并发布三态。"""
    if not isinstance(freeze, ConflictSetFamilyFreeze):
        raise TypeError("family freeze 类型错误")
    if not isinstance(observation_keys, tuple) or not observation_keys:
        raise TypeError("observation keys 必须为非空 tuple")
    if observation_keys != tuple(sorted(set(observation_keys))):
        raise ConflictSetFormalRunnerError("observation keys 必须排序且唯一")
    if not callable(actual_executor):
        raise TypeError("actual executor 必须可调用")
    run = _require_k_run_root(run_root)
    family = _family_path(run, family_root)
    freeze_path = family / FAMILY_FREEZE_MANIFEST_NAME
    if (read_conflict_set_family_freeze(freeze_path) != freeze):
        raise ConflictSetFormalRunnerError("family freeze 回读与输入漂移")
    assert_conflict_set_family_freeze_matches_live_public_code(
        freeze, repository_root,
    )
    if any((family / name).exists() for name in (
            CONFLICT_SET_GUARD_CONSUMED_NAME,
            CONFLICT_SET_RUN_INTENT_NAME,
            CONFLICT_SET_PUBLICATION_NAME,
            CONFLICT_SET_OUTCOME_NAME)):
        raise ConflictSetFormalRunnerError("formal family 已消费")
    _consume_guard(family, freeze)

    phase = "GUARD_CONSUMED"
    records: tuple[ConflictSetSemanticPredictionRecord, ...]
    actual_error = False
    try:
        phase = "CANDIDATE_RUN"
        actuals = []
        for key in observation_keys:
            try:
                result = actual_executor(key)
                if (not isinstance(result, ConflictSetActualRun)
                        or result.observation_stable_key_sha256 != key):
                    raise ConflictSetFormalRunnerError(
                        "actual executor 返回 identity 漂移")
            except Exception:
                actual_error = True
                result = ConflictSetActualRun(
                    key,
                    freeze.transport.candidate_manifest_sha256,
                    "NE",
                    None,
                )
            actuals.append(result)
        records = tuple(
            build_conflict_set_semantic_prediction_record(
                item.observation_stable_key_sha256,
                item.candidate_identity_sha256,
                item.status,
                item.projection,
            )
            for item in actuals
        )
        prediction = build_conflict_set_semantic_prediction_seal(
            records,
            family_manifest_sha256=freeze.sha256(),
            family_commitment_sha256=freeze.family_commitment_sha256,
            candidate_manifest_sha256=freeze.transport.candidate_manifest_sha256,
        )
        prediction_path = _transport_path(
            run, _artifact(freeze, "prediction_seal"))
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        write_immutable_json(prediction.to_dict(), prediction_path)
        if read_canonical_object(prediction_path) != prediction.to_dict():
            raise ConflictSetFormalRunnerError("prediction seal 回读漂移")
        phase = "PREDICTION_SEALED"
    except Exception:
        actual_error = True
        records = tuple(
            build_conflict_set_semantic_prediction_record(
                key,
                freeze.transport.candidate_manifest_sha256,
                "NE",
                None,
            ) for key in observation_keys)
        prediction = build_conflict_set_semantic_prediction_seal(
            records,
            family_manifest_sha256=freeze.sha256(),
            family_commitment_sha256=freeze.family_commitment_sha256,
            candidate_manifest_sha256=freeze.transport.candidate_manifest_sha256,
        )
        prediction_path = _transport_path(
            run, _artifact(freeze, "prediction_seal"))
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        if not prediction_path.exists():
            write_immutable_json(prediction.to_dict(), prediction_path)
        phase = "PREDICTION_SEALED"

    label_read: ConflictSetPrivateLabelRead | None = None
    try:
        phase = "PRIVATE_LABEL_READ"
        label_read = read_conflict_set_semantic_labels_after_prediction_seal(
            run_root=run,
            family_root=family,
            freeze=freeze,
            prediction=prediction,
        )
        phase = "SEMANTIC_SCORING"
        aggregate = build_conflict_set_semantic_formal_aggregate(
            prediction,
            label_read.records,
            label_commitment_sha256=label_read.label_commitment_sha256,
            label_transport_bytes=label_read.transport_size_bytes,
        )
        if actual_error and aggregate.failure_phase == "NONE":
            raise ConflictSetFormalRunnerError(
                "actual executor failure 未进入 prediction status")
        phase = "COMPLETE"
    except Exception:
        aggregate = build_conflict_set_semantic_unavailable_aggregate(
            prediction,
            label_commitment_sha256=_artifact(
                freeze, "private_labels").content_sha256 or "0" * 64,
            prediction_seal_sha256=prediction.sha256(),
            failure_phase=phase,
            label_read_count=0 if label_read is None else label_read.read_count,
            label_record_count=0 if label_read is None else len(label_read.records),
            label_transport_bytes=(
                0 if label_read is None else label_read.transport_size_bytes),
        )
    return _publish_result(family, aggregate, prediction, label_read)


__all__ = [
    "CONFLICT_SET_AGGREGATE_NAME",
    "CONFLICT_SET_FAILURE_SEAL_NAME",
    "CONFLICT_SET_GUARD_AVAILABLE_NAME",
    "CONFLICT_SET_GUARD_CONSUMED_NAME",
    "CONFLICT_SET_OUTCOME_NAME",
    "CONFLICT_SET_PREDICTION_SEAL_NAME",
    "CONFLICT_SET_PUBLICATION_NAME",
    "CONFLICT_SET_RUN_INTENT_NAME",
    "ConflictSetActualRun",
    "ConflictSetFormalPublication",
    "ConflictSetFormalRunnerError",
    "publish_conflict_set_family_freeze",
    "run_conflict_set_formal_evaluation_once",
]
