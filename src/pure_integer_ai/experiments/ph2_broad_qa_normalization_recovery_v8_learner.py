"""Recovery-v8 deterministic family-LOSO learner on the K-drive work disk."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_materialized_learner_runtime import (
    read_materialized_learner,
    run_materialized_learner,
    validate_materialized_checkpoint_chain,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_learning_contract import (
    NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_learning_records import (
    derive_normalization_recovery_v8_learning_outputs,
    normalization_recovery_v8_output_payloads,
    normalization_recovery_v8_prefix_counts,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_protocol import (
    read_normalization_recovery_v8_learner_input,
)


NORMALIZATION_RECOVERY_V8_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_LEARNER_V1")
NORMALIZATION_RECOVERY_V8_LEARNER_STATUS = (
    "FAMILY_LOSO_LEARNED_PACK_DISABLED")
NORMALIZATION_RECOVERY_V8_CHECKPOINT_OPEN = "RECOVERY_V8_CHECKPOINT_OPEN"
NORMALIZATION_RECOVERY_V8_RESUME_MARKER_KIND = (
    "NORMALIZATION_RECOVERY_V8_RESUME_MARKER_V1")
NORMALIZATION_RECOVERY_V8_OPERATOR_FAMILY = NORMALIZATION_CONTRASTIVE_FAMILY

_FIXED_MANIFEST_FIELDS = {
    "candidate_pack_read_count": 0,
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
    "vlc_final_read_count": 0,
}


def normalization_recovery_v8_learning_material(
        protocol_dir: Path, expected_manifest_sha256: str,
        ) -> dict[str, object]:
    """Read only the sealed protocol and derive the unique LOSO outputs."""
    manifest, protocol_outputs = read_normalization_recovery_v8_learner_input(
        protocol_dir, expected_manifest_sha256=expected_manifest_sha256)
    outputs, summary, work, increments = (
        derive_normalization_recovery_v8_learning_outputs(
            protocol_manifest=manifest,
            protocol_outputs=protocol_outputs,
        ))
    return {
        "manifest": manifest,
        "outputs": outputs,
        "payloads": normalization_recovery_v8_output_payloads(outputs),
        "prefix_context": increments,
        "summary": summary,
        "work": work,
    }


def run_normalization_recovery_v8_learner(
        *, run_root: str | Path, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str, run_dir: str | Path,
        run_id: str, mode: str, checkpoint_interval: int = 128,
        stop_after: int | None = None,
        ) -> dict[str, object]:
    """Run or resume the v8 learner with append-only checkpoints."""
    return run_materialized_learner(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        run_dir=run_dir,
        run_id=run_id,
        mode=mode,
        checkpoint_interval=checkpoint_interval,
        stop_after=stop_after,
        label="normalization recovery v8 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V8_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V8_LEARNER_STATUS,
        checkpoint_open_status=NORMALIZATION_RECOVERY_V8_CHECKPOINT_OPEN,
        resume_marker_kind=NORMALIZATION_RECOVERY_V8_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V8_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_v8_learning_material,
        prefix_counter=normalization_recovery_v8_prefix_counts,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_v8_learner(
        run_dir: str | Path, *, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """Strictly reread a learner lineage by rederiving from the protocol."""
    material = normalization_recovery_v8_learning_material(
        Path(protocol_dir), expected_protocol_manifest_sha256)
    return read_normalization_recovery_v8_learner_with_material(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        material=material,
    )


def read_normalization_recovery_v8_learner_with_material(
        run_dir: str | Path, *, protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        material: dict[str, object],
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """Strictly reread one lineage using a shared derived material object."""
    return read_materialized_learner(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        label="normalization recovery v8 learner",
        artifact_kind=NORMALIZATION_RECOVERY_V8_LEARNER_KIND,
        status=NORMALIZATION_RECOVERY_V8_LEARNER_STATUS,
        resume_marker_kind=NORMALIZATION_RECOVERY_V8_RESUME_MARKER_KIND,
        format_version=1,
        marker_format_version=1,
        operator_family=NORMALIZATION_RECOVERY_V8_OPERATOR_FAMILY,
        output_file_roles=NORMALIZATION_RECOVERY_V8_OUTPUT_FILE_ROLES,
        material_loader=lambda _protocol_dir, _protocol_sha: material,
        prefix_counter=normalization_recovery_v8_prefix_counts,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def validate_normalization_recovery_v8_checkpoint_chain(
        *, chain_path: Path, run_id: str,
        protocol_manifest_sha256: str,
        work: tuple[dict[str, object], ...],
        increments: tuple[dict[str, int], ...],
        require_complete: bool,
        ):
    """Recompute every v8 checkpoint revision for tests and audit."""
    return validate_materialized_checkpoint_chain(
        chain_path=chain_path,
        run_id=run_id,
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=NORMALIZATION_RECOVERY_V8_OPERATOR_FAMILY,
        work=work,
        prefix_context=increments,
        prefix_counter=normalization_recovery_v8_prefix_counts,
        require_complete=require_complete,
        label="normalization recovery v8 learner",
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V8_CHECKPOINT_OPEN",
    "NORMALIZATION_RECOVERY_V8_LEARNER_KIND",
    "NORMALIZATION_RECOVERY_V8_LEARNER_STATUS",
    "normalization_recovery_v8_learning_material",
    "read_normalization_recovery_v8_learner",
    "read_normalization_recovery_v8_learner_with_material",
    "run_normalization_recovery_v8_learner",
    "validate_normalization_recovery_v8_checkpoint_chain",
]
