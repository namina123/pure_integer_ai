"""Normalization recovery 的 K 盘 deterministic learner 与恢复 reader。

learner 只读取物化 TRAIN protocol 和调用方提供的冻结 manifest SHA；它不打开
source、Firefox evaluation/reserve、LOSO audit、旧 formal，也不启用生产规则。
"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_materialized_learner_runtime import (
    read_materialized_learner,
    run_materialized_learner,
    validate_materialized_checkpoint_chain,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
    derive_normalization_recovery_learning_outputs,
    normalization_recovery_output_payloads,
    normalization_recovery_prefix_output_counts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_protocol import (
    read_normalization_recovery_learner_input,
)


NORMALIZATION_RECOVERY_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_LEARNER_V2")
NORMALIZATION_RECOVERY_LEARNER_STATUS = (
    "DEVELOPMENT_COMPLETE_PACK_DISABLED")
NORMALIZATION_RECOVERY_CHECKPOINT_OPEN = "RECOVERY_CHECKPOINT_OPEN"
NORMALIZATION_RECOVERY_RESUME_MARKER_KIND = (
    "NORMALIZATION_RECOVERY_RESUME_MARKER_V2")

_FIXED_MANIFEST_FIELDS = {
    "candidate_pack_read_count": 0,
    "evaluation_payload_read_count": 0,
    "evaluation_protocol_manifest_read_count": 0,
    "loso_audit_read_count": 0,
    "prior_formal_item_read_count": 0,
    "reserve_identity_read_count": 0,
    "reserve_payload_read_count": 0,
    "source_pack_read_count": 0,
    "teacher_api_llm_call_count": 0,
    "training_protocol_read_count": 1,
}


def normalization_recovery_learning_material(
        protocol_dir: Path,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只从 recovery protocol 读取 learner 文件，并纯派生唯一输出。"""
    values = read_normalization_recovery_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest, roster, observations, groups, compositions, work = values
    outputs, summary = derive_normalization_recovery_learning_outputs(
        protocol_manifest=manifest,
        roster=roster,
        observations=observations,
        groups=groups,
        compositions=compositions,
        work=work,
    )
    return {
        "manifest": manifest,
        "outputs": outputs,
        "payloads": normalization_recovery_output_payloads(outputs),
        "prefix_context": {
            "compositions": compositions,
            "groups": groups,
        },
        "summary": summary,
        "work": work,
    }


def _prefix_counter(
        work: tuple[dict[str, object], ...],
        context: object,
        processed_item_count: int,
        ) -> tuple[int, int]:
    """把 recovery 前缀输出计数映射到通用 checkpoint 双计数。"""
    if (not isinstance(context, dict)
            or set(context) != {"compositions", "groups"}
            or not isinstance(context["compositions"], tuple)
            or not isinstance(context["groups"], tuple)):
        raise BroadQaExternalDataError(
            "recovery learner prefix context 漂移")
    return normalization_recovery_prefix_output_counts(
        work=work,
        groups=context["groups"],
        compositions=context["compositions"],
        processed_item_count=processed_item_count,
    )


def run_normalization_recovery_learner(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        run_dir: str | Path,
        run_id: str,
        mode: str,
        checkpoint_interval: int = 512,
        stop_after: int | None = None,
        ) -> dict[str, object]:
    """在 K 盘 fresh/resume 完整扫描 recovery TRAIN 并 manifest-last 封口。"""
    return run_materialized_learner(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        run_dir=run_dir,
        run_id=run_id,
        mode=mode,
        checkpoint_interval=checkpoint_interval,
        stop_after=stop_after,
        label="normalization recovery learner",
        artifact_kind=NORMALIZATION_RECOVERY_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_LEARNER_STATUS,
        checkpoint_open_status=NORMALIZATION_RECOVERY_CHECKPOINT_OPEN,
        resume_marker_kind=NORMALIZATION_RECOVERY_RESUME_MARKER_KIND,
        format_version=2,
        marker_format_version=2,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_learning_material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_learner(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 protocol 重派生并严格回读完整 recovery learner artifact。"""
    return read_materialized_learner(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        label="normalization recovery learner",
        artifact_kind=NORMALIZATION_RECOVERY_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_LEARNER_STATUS,
        resume_marker_kind=NORMALIZATION_RECOVERY_RESUME_MARKER_KIND,
        format_version=2,
        marker_format_version=2,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_learning_material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def validate_normalization_recovery_checkpoint_chain(
        *,
        chain_path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        work: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        require_complete: bool,
        ):
    """供审计和测试逐 revision 重算 recovery checkpoint。"""
    return validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        work=work,
        prefix_context={"compositions": compositions, "groups": groups},
        prefix_counter=_prefix_counter,
        require_complete=require_complete,
        label="normalization recovery learner",
    )


__all__ = [
    "NORMALIZATION_RECOVERY_CHECKPOINT_OPEN",
    "NORMALIZATION_RECOVERY_LEARNER_KIND",
    "NORMALIZATION_RECOVERY_LEARNER_STATUS",
    "normalization_recovery_learning_material",
    "read_normalization_recovery_learner",
    "run_normalization_recovery_learner",
    "validate_normalization_recovery_checkpoint_chain",
]
