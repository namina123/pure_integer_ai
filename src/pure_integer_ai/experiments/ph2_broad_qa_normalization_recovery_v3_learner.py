"""Recovery-v3 的 K 盘 deterministic learner 与严格恢复 reader。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_materialized_learner_runtime import (
    read_materialized_learner,
    run_materialized_learner,
    validate_materialized_checkpoint_chain,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_learning_records import (
    NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES,
    derive_normalization_recovery_v3_learning_outputs,
    normalization_recovery_v3_output_payloads,
    normalization_recovery_v3_prefix_output_counts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_protocol import (
    read_normalization_recovery_v3_learner_input,
)


NORMALIZATION_RECOVERY_V3_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V3_LEARNER_V1")
NORMALIZATION_RECOVERY_V3_LEARNER_STATUS = (
    "DEVELOPMENT_COMPLETE_PACK_DISABLED")
NORMALIZATION_RECOVERY_V3_CHECKPOINT_OPEN = (
    "RECOVERY_V3_CHECKPOINT_OPEN")
NORMALIZATION_RECOVERY_V3_RESUME_MARKER_KIND = (
    "NORMALIZATION_RECOVERY_V3_RESUME_MARKER_V1")
NORMALIZATION_RECOVERY_V3_OPERATOR_FAMILY = (
    NORMALIZATION_CONTRASTIVE_FAMILY)

_FIXED_MANIFEST_FIELDS = {
    "base_rule_pack_read_count": 0,
    "candidate_pack_read_count": 0,
    "evaluation_commitment_read_count": 0,
    "evaluation_payload_read_count": 0,
    "prior_formal_item_read_count": 0,
    "reserve_identity_read_count": 0,
    "reserve_payload_read_count": 0,
    "source_pack_read_count": 0,
    "teacher_api_llm_call_count": 0,
    "training_protocol_read_count": 1,
}


def normalization_recovery_v3_learning_material(
        protocol_dir: Path,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只从 v3 protocol 读取四份 TRAIN 文件并纯派生唯一输出。"""
    values = read_normalization_recovery_v3_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest, observations, fragments, groups, work = values
    outputs, summary, emission_counts = (
        derive_normalization_recovery_v3_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=fragments,
            groups=groups,
            work=work,
        ))
    return {
        "manifest": manifest,
        "outputs": outputs,
        "payloads": normalization_recovery_v3_output_payloads(outputs),
        "prefix_context": {"emission_counts": emission_counts},
        "summary": summary,
        "work": work,
    }


def _prefix_counter(
        work: tuple[dict[str, object], ...],
        context: object,
        processed_item_count: int,
        ) -> tuple[int, int]:
    """把 v3 family emission 映射到共享 checkpoint 双计数。"""
    if (not isinstance(context, dict)
            or set(context) != {"emission_counts"}
            or not isinstance(context["emission_counts"], tuple)):
        raise BroadQaExternalDataError(
            "v3 learner prefix context 漂移")
    return normalization_recovery_v3_prefix_output_counts(
        work=work,
        emission_counts=context["emission_counts"],
        processed_item_count=processed_item_count,
    )


def run_normalization_recovery_v3_learner(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        run_dir: str | Path,
        run_id: str,
        mode: str,
        checkpoint_interval: int = 8_192,
        stop_after: int | None = None,
        ) -> dict[str, object]:
    """在 K 盘 fresh/resume 完整扫描 v3 TRAIN 并 manifest-last 封口。"""
    return run_materialized_learner(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        run_dir=run_dir,
        run_id=run_id,
        mode=mode,
        checkpoint_interval=checkpoint_interval,
        stop_after=stop_after,
        label="normalization recovery v3 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V3_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V3_LEARNER_STATUS,
        checkpoint_open_status=NORMALIZATION_RECOVERY_V3_CHECKPOINT_OPEN,
        resume_marker_kind=NORMALIZATION_RECOVERY_V3_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V3_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_v3_learning_material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_v3_learner(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """从 protocol 重派生并严格回读完整 v3 learner artifact。"""
    return read_materialized_learner(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        label="normalization recovery v3 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V3_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V3_LEARNER_STATUS,
        resume_marker_kind=NORMALIZATION_RECOVERY_V3_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V3_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V3_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_v3_learning_material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def validate_normalization_recovery_v3_checkpoint_chain(
        *,
        chain_path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        require_complete: bool,
        ):
    """供审计和测试逐 revision 重算 v3 checkpoint。"""
    return validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=NORMALIZATION_RECOVERY_V3_OPERATOR_FAMILY,
        work=work,
        prefix_context={"emission_counts": emission_counts},
        prefix_counter=_prefix_counter,
        require_complete=require_complete,
        label="normalization recovery v3 learner",
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V3_CHECKPOINT_OPEN",
    "NORMALIZATION_RECOVERY_V3_LEARNER_KIND",
    "NORMALIZATION_RECOVERY_V3_LEARNER_STATUS",
    "normalization_recovery_v3_learning_material",
    "read_normalization_recovery_v3_learner",
    "run_normalization_recovery_v3_learner",
    "validate_normalization_recovery_v3_checkpoint_chain",
]
