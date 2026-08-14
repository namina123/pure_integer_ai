"""Recovery-v5 的 K 盘 deterministic learner 与严格恢复 reader。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
    normalization_recovery_v5_checkpoint_prefix_context,
    normalization_recovery_v5_checkpoint_prefix_counts,
    normalization_recovery_v5_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_protocol import (
    read_normalization_recovery_v5_learner_input,
)


NORMALIZATION_RECOVERY_V5_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_LEARNER_V1")
NORMALIZATION_RECOVERY_V5_LEARNER_STATUS = (
    "DEVELOPMENT_COMPLETE_PACK_DISABLED")
NORMALIZATION_RECOVERY_V5_CHECKPOINT_OPEN = "RECOVERY_V5_CHECKPOINT_OPEN"
NORMALIZATION_RECOVERY_V5_RESUME_MARKER_KIND = (
    "NORMALIZATION_RECOVERY_V5_RESUME_MARKER_V1")
NORMALIZATION_RECOVERY_V5_OPERATOR_FAMILY = NORMALIZATION_CONTRASTIVE_FAMILY

_FIXED_MANIFEST_FIELDS = {
    "base_rule_pack_read_count": 0,
    "candidate_pack_read_count": 0,
    "evaluation_commitment_read_count": 0,
    "evaluation_or_held_out_payload_read_count": 0,
    "evaluation_payload_read_count": 0,
    "held_out_source_non_manifest_read_count": 0,
    "predecessor_rule_pack_read_count": 0,
    "prior_formal_item_read_count": 0,
    "reserve_identity_read_count": 0,
    "reserve_payload_read_count": 0,
    "source_pack_read_count": 0,
    "teacher_api_llm_call_count": 0,
    "training_protocol_read_count": 1,
}


def normalization_recovery_v5_learning_material(
        protocol_dir: Path,
        expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """只从 v5 protocol 读取四份 TRAIN 文件并纯派生唯一输出。"""
    values = read_normalization_recovery_v5_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest, observations, fragments, groups, work = values
    outputs, summary, emission_counts = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=fragments,
            groups=groups,
            work=work,
        ))
    return {
        "manifest": manifest,
        "outputs": outputs,
        "payloads": normalization_recovery_v5_output_payloads(outputs),
        "prefix_context": normalization_recovery_v5_checkpoint_prefix_context(
            work=work,
            emission_counts=emission_counts,
        ),
        "summary": summary,
        "work": work,
    }


def _prefix_counter(
        work: tuple[dict[str, object], ...],
        context: object,
        processed_item_count: int,
        ) -> tuple[int, int]:
    """把 v5 observation/group emission 映射到共享 checkpoint 双计数。"""
    if not isinstance(context, dict):
        raise BroadQaExternalDataError("v5 learner prefix context 漂移")
    return normalization_recovery_v5_checkpoint_prefix_counts(
        work_item_count=len(work),
        prefix_context=context,
        processed_item_count=processed_item_count,
    )


def run_normalization_recovery_v5_learner(
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
    """在 K 盘 fresh/resume 完整扫描 v5 TRAIN 并 manifest-last 封口。"""
    return run_materialized_learner(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        run_dir=run_dir,
        run_id=run_id,
        mode=mode,
        checkpoint_interval=checkpoint_interval,
        stop_after=stop_after,
        label="normalization recovery v5 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V5_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V5_LEARNER_STATUS,
        checkpoint_open_status=NORMALIZATION_RECOVERY_V5_CHECKPOINT_OPEN,
        resume_marker_kind=NORMALIZATION_RECOVERY_V5_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V5_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_v5_learning_material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_v5_learner(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """从 protocol 重派生并严格回读完整 v5 learner artifact。"""
    material = normalization_recovery_v5_learning_material(
        Path(protocol_dir), expected_protocol_manifest_sha256)
    return read_normalization_recovery_v5_learner_with_material(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        material=material,
    )


def read_normalization_recovery_v5_learner_with_material(
        run_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        material: dict[str, object],
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """用同一份已派生 material 严格回读一条 v5 learner lineage。"""
    return read_materialized_learner(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        label="normalization recovery v5 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V5_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V5_LEARNER_STATUS,
        resume_marker_kind=NORMALIZATION_RECOVERY_V5_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V5_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
        material_loader=lambda _protocol_dir, _protocol_sha: material,
        prefix_counter=_prefix_counter,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def validate_normalization_recovery_v5_checkpoint_chain(
        *,
        chain_path: Path,
        run_id: str,
        protocol_manifest_sha256: str,
        work: tuple[dict[str, object], ...],
        emission_counts: tuple[dict[str, object], ...],
        require_complete: bool,
        ):
    """供审计和测试逐 revision 重算 v5 checkpoint。"""
    return validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=NORMALIZATION_RECOVERY_V5_OPERATOR_FAMILY,
        work=work,
        prefix_context=normalization_recovery_v5_checkpoint_prefix_context(
            work=work,
            emission_counts=emission_counts,
        ),
        prefix_counter=_prefix_counter,
        require_complete=require_complete,
        label="normalization recovery v5 learner",
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V5_CHECKPOINT_OPEN",
    "NORMALIZATION_RECOVERY_V5_LEARNER_KIND",
    "NORMALIZATION_RECOVERY_V5_LEARNER_STATUS",
    "normalization_recovery_v5_learning_material",
    "read_normalization_recovery_v5_learner",
    "read_normalization_recovery_v5_learner_with_material",
    "run_normalization_recovery_v5_learner",
    "validate_normalization_recovery_v5_checkpoint_chain",
]
