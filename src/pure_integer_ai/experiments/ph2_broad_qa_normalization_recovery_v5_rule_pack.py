"""发布 recovery-v5 双运行等价后的禁用态 scoped phrase rule pack。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_rule_pack as materialized_pack_runtime,
)
from pure_integer_ai.experiments.ph2_broad_qa_materialized_rule_pack import (
    publish_materialized_rule_pack,
    read_materialized_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learner import (
    normalization_recovery_v5_learning_material,
    read_normalization_recovery_v5_learner_with_material,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    normalization_recovery_v5_output_payloads,
)


NORMALIZATION_RECOVERY_V5_RULE_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_RULE_PACK_V1")
NORMALIZATION_RECOVERY_V5_RULE_PACK_STATUS = (
    "FROZEN_NOT_EVALUATED_NOT_DEPLOYED")

_FIXED_MANIFEST_FIELDS = {
    "base_rule_pack_read_count": 0,
    "candidate_pack_read_count": 0,
    "evaluation_commitment_read_count": 0,
    "evaluation_or_held_out_payload_read_count": 0,
    "evaluation_payload_read_count": 0,
    "held_out_source_non_manifest_read_count": 0,
    "learner_run_read_count": 2,
    "predecessor_rule_pack_read_count": 0,
    "prior_formal_item_read_count": 0,
    "reserve_identity_read_count": 0,
    "reserve_payload_read_count": 0,
    "source_pack_read_count": 0,
    "teacher_api_llm_call_count": 0,
}


def publish_normalization_recovery_v5_rule_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        fresh_run_dir: str | Path,
        resumed_run_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """验证 fresh/resume 语义字节相等后不可覆盖发布禁用态 pack。"""
    root = materialized_pack_runtime.require_k_run_root(
        run_root, label="normalization recovery v5 rule pack")
    paths = tuple(Path(value).resolve() for value in (
        protocol_dir, fresh_run_dir, resumed_run_dir, target_dir))
    protocol_root, fresh_root, resumed_root, target = paths
    if (any(not path.is_relative_to(root) for path in paths)
            or target.exists()
            or not protocol_root.is_dir()
            or not fresh_root.is_dir()
            or not resumed_root.is_dir()):
        raise BroadQaExternalDataError(
            "normalization recovery v5 rule pack path 越界、缺失或已存在")
    material = normalization_recovery_v5_learning_material(
        protocol_root, expected_protocol_manifest_sha256)

    def _cached_reader(
            run_dir: Path,
            cached_protocol_dir: Path,
            protocol_manifest_sha256: str,
            ) -> tuple[dict[str, object],
                       dict[str, tuple[dict[str, object], ...]]]:
        """让 fresh/resume 共用一次 protocol 派生，同时各自严格回读。"""
        return read_normalization_recovery_v5_learner_with_material(
            run_dir,
            protocol_dir=cached_protocol_dir,
            expected_protocol_manifest_sha256=protocol_manifest_sha256,
            material=material,
        )

    return publish_materialized_rule_pack(
        run_root=run_root,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        fresh_run_dir=fresh_run_dir,
        resumed_run_dir=resumed_run_dir,
        target_dir=target_dir,
        label="normalization recovery v5 rule pack",
        artifact_kind=NORMALIZATION_RECOVERY_V5_RULE_PACK_KIND,
        status=NORMALIZATION_RECOVERY_V5_RULE_PACK_STATUS,
        format_version=1,
        output_file_roles=NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
        learner_reader=_cached_reader,
        payload_builder=normalization_recovery_v5_output_payloads,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


def read_normalization_recovery_v5_rule_pack(
        pack_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        ) -> tuple[dict[str, object],
                   dict[str, tuple[dict[str, object], ...]]]:
    """以双外部 SHA 与 protocol 重派生严格回读 v5 disabled pack。"""
    return read_materialized_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=expected_protocol_manifest_sha256,
        expected_pack_manifest_sha256=expected_pack_manifest_sha256,
        label="normalization recovery v5 rule pack",
        artifact_kind=NORMALIZATION_RECOVERY_V5_RULE_PACK_KIND,
        status=NORMALIZATION_RECOVERY_V5_RULE_PACK_STATUS,
        format_version=1,
        output_file_roles=NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
        material_loader=normalization_recovery_v5_learning_material,
        fixed_manifest_fields=_FIXED_MANIFEST_FIELDS,
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V5_RULE_PACK_KIND",
    "NORMALIZATION_RECOVERY_V5_RULE_PACK_STATUS",
    "publish_normalization_recovery_v5_rule_pack",
    "read_normalization_recovery_v5_rule_pack",
]
