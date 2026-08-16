"""覆盖 recovery-v9 GIMP 标签盲正式commitment。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER,
    NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_KIND,
    publish_normalization_recovery_v9_evaluation_commitment,
    read_normalization_recovery_v9_evaluation_commitment,
)


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """形成不含任何surface的最小aggregate前驱。"""
    return (
        {
            "artifact_kind": (
                "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_GIMP_SOURCE_PACK_V1"),
        },
        {"record_kind": "V9_GIMP_SOURCE_PACK_CENSUS_V1"},
        {
            "artifact_kind": (
                "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_RUNTIME_GATE_V1"),
        },
    )


def test_v9_commitment_freezes_full_denominator_and_is_strict(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """9,264全分母不得被8,924 eligible子集替代。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment as module

    source = tmp_path / "source"
    gate = tmp_path / "gate"
    target = tmp_path / "commitment"
    source.mkdir()
    gate.mkdir()
    source_manifest, census, runtime_gate = _inputs()
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_source_inputs", lambda value: (source_manifest, census))
    monkeypatch.setattr(module, "_runtime_gate", lambda value: runtime_gate)
    published = publish_normalization_recovery_v9_evaluation_commitment(
        run_root=tmp_path,
        source_pack_dir=source,
        runtime_gate_dir=gate,
        target_dir=target,
    )
    reread = read_normalization_recovery_v9_evaluation_commitment(
        target,
        source_pack_dir=source,
        runtime_gate_dir=gate,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert reread == published
    assert reread["artifact_kind"] == (
        NORMALIZATION_RECOVERY_V9_EVALUATION_COMMITMENT_KIND)
    assert reread["denominator"]["record_count"] == 9_264
    assert reread["denominator"]["aggregate_buckets"][
        "evaluation_eligible_count"] == 8_924
    assert reread["denominator"][
        "eligible_subset_cannot_replace_full_denominator"] == 1
    assert tuple(reread["dimension_order"]) == (
        NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER)
    assert reread["candidate_or_code_read_count"] == 0
    assert reread["gimp_identity_raw_or_translation_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError):
        publish_normalization_recovery_v9_evaluation_commitment(
            run_root=tmp_path,
            source_pack_dir=source,
            runtime_gate_dir=gate,
            target_dir=target,
        )
