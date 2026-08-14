"""发布 recovery 双运行等价后的禁用态 rule pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_materialized_rule_pack import (
    publish_materialized_rule_pack,
    read_materialized_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learner import (
    normalization_recovery_learning_material,
    read_normalization_recovery_learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
    normalization_recovery_output_payloads,
)


NORMALIZATION_RECOVERY_RULE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_RULE_PACK_V2")
NORMALIZATION_RECOVERY_RULE_PACK_STATUS = (
    "FROZEN_NOT_EVALUATED_NOT_DEPLOYED")

_FIXED_MANIFEST_FIELDS = {
    "candidate_pack_read_count": 0,
    "evaluation_payload_read_count": 0,
    "evaluation_protocol_manifest_read_count": 0,
    "learner_run_read_count": 2,
    "loso_audit_read_count": 0,
    "prior_formal_item_read_count": 0,
    "reserve_identity_read_count": 0,
    "reserve_payload_read_count": 0,
    "source_pack_read_count": 0,
    "teacher_api_llm_call_count": 0,
}


def _learner_reader(
        run_dir: Path,
        protocol_dir: Path,
        protocol_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """把共享 pack runtime 的位置参数适配到 recovery reader。"""
    return read_normalization_recovery_learner(
        run_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_manifest_sha256,
    )


def publish_normalization_recovery_rule_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        fresh_run_dir: str | Path,
        resumed_run_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """验证双运行语义字节相等后不可覆盖发布禁用态 recovery pack。"""
    return publish_materialized_rule_pack(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        fresh_run_dir=fresh_run_dir,
        resumed_run_dir=resumed_run_dir,
        target_dir=target_dir,
        label="normalization recovery rule pack",
        artifact_kind=NORMALIZATION_RECOVERY_RULE_PACK_KIND,
        status=NORMALIZATION_RECOVERY_RULE_PACK_STATUS,
        format_version=2,
        output_file_roles=NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
        learner_reader=_learner_reader,
        payload_builder=normalization_recovery_output_payloads,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_rule_pack(
        pack_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """以双外部 SHA 与 protocol 重派生严格回读禁用态 recovery pack。"""
    return read_materialized_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
        label="normalization recovery rule pack",
        artifact_kind=NORMALIZATION_RECOVERY_RULE_PACK_KIND,
        status=NORMALIZATION_RECOVERY_RULE_PACK_STATUS,
        format_version=2,
        output_file_roles=NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_learning_material,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_RULE_PACK_KIND",
    "NORMALIZATION_RECOVERY_RULE_PACK_STATUS",
    "publish_normalization_recovery_rule_pack",
    "read_normalization_recovery_rule_pack",
]
