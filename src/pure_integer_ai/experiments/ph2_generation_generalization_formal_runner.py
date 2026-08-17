"""GG-03 prediction-first、label-after-seal 的唯一 formal evaluation runner。"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
import traceback

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.runtime import (
    consume_evaluation_guard_once,
    verify_evaluation_guard_consumed,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    read_generation_generalization_evaluation_family_freeze,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    generation_generalization_sha256_bytes,
    read_generation_generalization_private_label_owner_receipt,
    scan_generation_generalization_observation_inventory,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    read_generation_generalization_evaluation_observations,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationBatchRunError,
    GenerationGeneralizationEvaluationPolicy,
    run_generation_generalization_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_labels import (
    read_generation_generalization_private_labels_after_guard,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_protocol import (
    FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND,
    GenerationGeneralizationFormalAggregate,
    GenerationGeneralizationPredictionSeal,
    build_generation_generalization_formal_aggregate,
    build_generation_generalization_prediction_seal,
    build_generation_generalization_publication,
    build_generation_generalization_unavailable_aggregate,
)
from pure_integer_ai.experiments.train_context import TrainContext


PREDICTION_SEAL_NAME = "predictions.seal.json"
FORMAL_OUTCOME_NAME = "run.outcome.json"
PUBLICATION_DIRECTORY_NAME = "publication"
FORMAL_AGGREGATE_NAME = "aggregate.json"
FORMAL_DECISION_NAME = "decision.json"
FORMAL_RUNTIME_RECEIPT_NAME = "runtime_receipt.json"
FORMAL_FAILURE_SEAL_NAME = "failure_seal.json"
FORMAL_FAILURE_DIAGNOSTIC_NAME = "failure_diagnostic.json"


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationFormalPublication:
    """唯一 formal run 的内存安全投影，不包含 surface 或 label。"""

    aggregate: GenerationGeneralizationFormalAggregate
    decision: dict[str, object]
    runtime_receipt: dict[str, object] | None
    failure_seal: dict[str, object] | None
    prediction_seal: GenerationGeneralizationPredictionSeal | None
    failure_diagnostic: dict[str, object] | None

    def __post_init__(self) -> None:
        if not isinstance(self.aggregate, GenerationGeneralizationFormalAggregate):
            raise TypeError("GG-03 formal publication aggregate 类型错误")
        if (self.aggregate.status == "PASS"
                and (self.runtime_receipt is None or self.failure_seal is not None)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 PASS publication 投影漂移")
        if (self.aggregate.status != "PASS"
                and (self.runtime_receipt is not None or self.failure_seal is None)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 non-PASS publication 投影漂移")
        if ((self.aggregate.failure_phase != "NONE")
                != (self.failure_diagnostic is not None)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 operational diagnostic 投影漂移")


def _safe_failure_diagnostic(
        error: Exception, *, phase: str,
        ) -> dict[str, object]:
    """剥离异常消息和输入，只保留公开代码位置与 batch 边界。"""
    root = error
    while root.__cause__ is not None:
        root = root.__cause__
    frames = tuple({
        "file": Path(item.filename).name,
        "function": item.name,
        "line": item.lineno,
    } for item in traceback.extract_tb(root.__traceback__))
    leaf = frames[-1] if frames else {
        "file": "UNAVAILABLE",
        "function": "UNAVAILABLE",
        "line": 0,
    }
    batch = (
        error if isinstance(
            error, GenerationGeneralizationEvaluationBatchRunError)
        else None)
    return {
        "artifact_kind": FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND,
        "evaluation_path": (
            "UNAVAILABLE" if batch is None else batch.evaluation_path),
        "exception_type": (
            f"{type(root).__module__}.{type(root).__qualname__}"),
        "failure_phase": phase,
        "format_version": 1,
        "leaf_file": leaf["file"],
        "leaf_function": leaf["function"],
        "leaf_line": leaf["line"],
        "message_or_input_field_count": 0,
        "observation_ordinal": (
            0 if batch is None else batch.observation_ordinal),
        "traceback_shape_sha256": generation_generalization_sha256_bytes(
            canonical_json_bytes(frames)),
    }


def _publication_paths(family: Path) -> tuple[Path, ...]:
    """返回 guard 消费前必须全部不存在的 formal output 路径。"""
    publication = family / PUBLICATION_DIRECTORY_NAME
    return (
        family / PREDICTION_SEAL_NAME,
        family / FORMAL_OUTCOME_NAME,
        publication,
    )


def _publish_formal_result(
        family: Path,
        aggregate: GenerationGeneralizationFormalAggregate,
        *,
        phase: str,
        prediction: GenerationGeneralizationPredictionSeal | None,
        diagnostic: dict[str, object] | None,
        ) -> GenerationGeneralizationFormalPublication:
    """以 publication 目录事务发布 aggregate 与唯一 receipt/seal。"""
    if any(path.exists() for path in (
            family / PUBLICATION_DIRECTORY_NAME,
            family / FORMAL_OUTCOME_NAME)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal publication 已存在")
    temporary = Path(tempfile.mkdtemp(
        prefix=".gg03-publication-building-", dir=family)).resolve()
    outcome_temporary = family / ".run.outcome.building.json"
    try:
        write_immutable_json(
            aggregate.to_dict(), temporary / FORMAL_AGGREGATE_NAME)
        diagnostic_sha = None
        if diagnostic is not None:
            diagnostic_path = temporary / FORMAL_FAILURE_DIAGNOSTIC_NAME
            write_immutable_json(diagnostic, diagnostic_path)
            diagnostic_sha = generation_generalization_sha256_bytes(
                diagnostic_path.read_bytes())
        decision, receipt, failure = (
            build_generation_generalization_publication(
                aggregate,
                failure_diagnostic_sha256=diagnostic_sha,
            ))
        write_immutable_json(
            decision, temporary / FORMAL_DECISION_NAME)
        if receipt is not None:
            write_immutable_json(
                receipt, temporary / FORMAL_RUNTIME_RECEIPT_NAME)
        if failure is not None:
            write_immutable_json(
                failure, temporary / FORMAL_FAILURE_SEAL_NAME)
        outcome = {
            "aggregate_sha256": aggregate.sha256(),
            "artifact_kind": "PH2_GG03_FORMAL_RUN_OUTCOME_V1",
            "failure_phase": aggregate.failure_phase,
            "failure_diagnostic_sha256": (
                "0" * 64 if diagnostic_sha is None else diagnostic_sha),
            "format_version": 1,
            "phase": phase,
            "prediction_seal_sha256": (
                "0" * 64 if prediction is None else prediction.sha256()),
            "status": aggregate.status,
        }
        with outcome_temporary.open("xb") as handle:
            handle.write(canonical_json_bytes(outcome) + b"\n")
        os.replace(temporary, family / PUBLICATION_DIRECTORY_NAME)
        os.replace(outcome_temporary, family / FORMAL_OUTCOME_NAME)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if outcome_temporary.exists():
            outcome_temporary.unlink()
        raise
    if (read_canonical_object(
            family / PUBLICATION_DIRECTORY_NAME / FORMAL_AGGREGATE_NAME)
            != aggregate.to_dict()
            or read_canonical_object(family / FORMAL_OUTCOME_NAME)["status"]
            != aggregate.status):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal publication 回读漂移")
    return GenerationGeneralizationFormalPublication(
        aggregate, decision, receipt, failure, prediction, diagnostic)


def run_generation_generalization_formal_evaluation_once(
        host_ctx: TrainContext,
        *,
        repository_root: str | Path,
        run_root: str | Path,
        family_dir: str | Path,
        candidate_visible_root: str | Path,
        private_label_root: str | Path,
        loaded_candidate: LoadedGenerationCandidatePack,
        observation_inventory_path: str | Path,
        private_owner_receipt_path: str | Path,
        public_dry_run_receipt_path: str | Path,
        policy: GenerationGeneralizationEvaluationPolicy,
        resource_ceiling: GenerationGeneralizationEvaluationBudget,
        ) -> GenerationGeneralizationFormalPublication:
    """消费唯一 guard，先封存全部输出，再读取 label 并发布三态结果。"""
    if not isinstance(host_ctx, TrainContext):
        raise TypeError("GG-03 formal host context 类型错误")
    family_root = Path(family_dir).resolve()
    freeze_arguments = {
        "repository_root": repository_root,
        "run_root": run_root,
        "candidate_visible_root": candidate_visible_root,
        "private_label_root": private_label_root,
        "loaded_candidate": loaded_candidate,
        "observation_inventory_path": observation_inventory_path,
        "private_owner_receipt_path": private_owner_receipt_path,
        "public_dry_run_receipt_path": public_dry_run_receipt_path,
        "policy": policy,
        "resource_ceiling": resource_ceiling,
    }
    freeze = read_generation_generalization_evaluation_family_freeze(
        family_root, **freeze_arguments)
    if any(path.exists() for path in _publication_paths(family_root)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal outputs 已存在")
    owner, owner_sha = read_generation_generalization_private_label_owner_receipt(
        private_owner_receipt_path)
    inventory = scan_generation_generalization_observation_inventory(
        observation_inventory_path, resource_ceiling=resource_ceiling)
    frozen_inventory = freeze.get("observation_inventory")
    if (not isinstance(frozen_inventory, dict)
            or set(frozen_inventory) != {
                *inventory.to_dict(), "double_pass_equal", "relative_path"}
            or frozen_inventory.get("double_pass_equal") != 1
            or {key: value for key, value in frozen_inventory.items()
                if key not in {"double_pass_equal", "relative_path"}}
            != inventory.to_dict()):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 formal Observation inventory 与 freeze 漂移")
    expected_guard = build_available_guard_for_identity(
        str(freeze["manifest_sha256"]),
        str(freeze["family_commitment_sha256"]),
        owner_sha,
        loaded_candidate.pack.sha256(),
        str(freeze["code_identity"]["aggregate_sha256"]),
    )
    consume_evaluation_guard_once(family_root, expected_guard)

    phase = "GUARD_CONSUMED"
    prediction = None
    diagnostic = None
    label_read_count = 0
    label_record_count = 0
    label_transport_bytes = 0
    try:
        phase = "OBSERVATIONS_READ"
        observations = read_generation_generalization_evaluation_observations(
            observation_inventory_path)
        observation_payload = b"".join(
            canonical_json_bytes(item.to_dict()) + b"\n"
            for item in observations)
        if (len(observations) != inventory.record_count
                or len(observation_payload) != inventory.transport_size_bytes
                or generation_generalization_sha256_bytes(observation_payload)
                != inventory.transport_sha256):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 formal materialized Observation 与 freeze 漂移")
        phase = "CANDIDATE_RUN"
        batch = run_generation_generalization_evaluation_batch(
            host_ctx, loaded_candidate, observations, policy)
        prediction = build_generation_generalization_prediction_seal(
            batch,
            family_manifest_sha256=str(freeze["manifest_sha256"]),
            family_commitment_sha256=str(
                freeze["family_commitment_sha256"]),
            candidate_payload_sha256=loaded_candidate.pack.sha256(),
            policy_sha256=str(freeze["policy_sha256"]),
            observation_inventory=inventory,
        )
        phase = "PREDICTIONS_SEALED"
        prediction_path = family_root / PREDICTION_SEAL_NAME
        write_immutable_json(prediction.to_dict(), prediction_path)
        if read_canonical_object(prediction_path) != prediction.to_dict():
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 prediction seal 回读漂移")
        verify_evaluation_guard_consumed(family_root, expected_guard)

        phase = "PRIVATE_LABEL_READ"
        label_read_count = 1
        label_transport_bytes = owner.label_transport_size_bytes
        labels = read_generation_generalization_private_labels_after_guard(
            family_root=family_root,
            expected_guard=expected_guard,
            prediction_seal_path=prediction_path,
            prediction_seal_sha256=prediction.sha256(),
            private_label_root=private_label_root,
            owner_receipt=owner,
            observation_inventory=inventory,
        )
        label_record_count = len(labels)
        phase = "SCORING"
        aggregate = build_generation_generalization_formal_aggregate(
            prediction,
            labels,
            label_commitment_sha256=owner.label_commitment_sha256,
            label_transport_bytes=label_transport_bytes,
        )
        phase = "COMPLETE"
    except Exception as error:
        diagnostic = _safe_failure_diagnostic(error, phase=phase)
        verify_evaluation_guard_consumed(family_root, expected_guard)
        aggregate = build_generation_generalization_unavailable_aggregate(
            inventory,
            family_manifest_sha256=str(freeze["manifest_sha256"]),
            family_commitment_sha256=str(
                freeze["family_commitment_sha256"]),
            label_commitment_sha256=owner.label_commitment_sha256,
            prediction_seal_sha256=(
                "0" * 64 if prediction is None else prediction.sha256()),
            failure_phase=phase,
            label_read_count=label_read_count,
            label_record_count=label_record_count,
            label_transport_bytes=label_transport_bytes,
        )
        phase = "SEALED_NE"
    return _publish_formal_result(
        family_root,
        aggregate,
        phase=phase,
        prediction=prediction,
        diagnostic=diagnostic,
    )


__all__ = [
    "FORMAL_AGGREGATE_NAME",
    "FORMAL_DECISION_NAME",
    "FORMAL_FAILURE_SEAL_NAME",
    "FORMAL_FAILURE_DIAGNOSTIC_NAME",
    "FORMAL_OUTCOME_NAME",
    "FORMAL_RUNTIME_RECEIPT_NAME",
    "PREDICTION_SEAL_NAME",
    "PUBLICATION_DIRECTORY_NAME",
    "GenerationGeneralizationFormalPublication",
    "run_generation_generalization_formal_evaluation_once",
]
