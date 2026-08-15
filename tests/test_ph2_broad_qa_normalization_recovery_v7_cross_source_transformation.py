"""覆盖 recovery-v7 cross-source transformation feasibility。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_audit
    as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_cross_source_transformation_records import (
    derive_cross_source_transformation_unscored_proposals,
    derive_cross_source_transformation_feasibility,
    derive_external_cross_source_optional_rewrite_proposals,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic SHA identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(
        family: str,
        suffix: str,
        input_text: str,
        output_text: str,
        *,
        tokens: list[str] | None = None,
        ) -> dict[str, object]:
    """构造 records 层所需的最小 TRAIN observation。"""
    structure_tokens = [] if tokens is None else tokens
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": int(input_text == output_text),
        "input_text": input_text,
        "observation_id": _id(f"observation-{family}-{suffix}"),
        "output_text": output_text,
        "source_family": family,
        "source_pair_id": _id(f"pair-{family}-{suffix}"),
        "source_policy_scope": V5_SOURCE_POLICY_BY_FAMILY[family],
        "structure_tokens": structure_tokens,
    }


def _fragment(
        observation: dict[str, object],
        suffix: str,
        ) -> dict[str, object]:
    """构造一个 frozen EDIT_CORE evidence。"""
    return {
        "fragment_id": _id(f"fragment-{suffix}"),
        "fragment_kind": "EDIT_CORE",
        "input_text": "舊",
        "observation_id": observation["observation_id"],
        "output_text": "新詞",
        "source_family": observation["source_family"],
    }


def _projection(
        observation: dict[str, object],
        *,
        surface_sha256: str,
        output_sha256: str,
        ) -> dict[str, object]:
    """构造无 raw surface 的 neutral projection。"""
    return {
        "neutral_surface_sha256": surface_sha256,
        "output_sha256": output_sha256,
        "pair_id": observation["source_pair_id"],
        "source_family": observation["source_family"],
    }


def _material(
        *,
        held_output_sha256: str | None = None,
        conflict_authority: bool = False,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """构造两独立 family model 可生成同一 held variable output。"""
    godot = _observation(
        GODOT_SOURCE_FAMILY, "godot", "舊", "新詞")
    libreoffice = _observation(
        LIBREOFFICE_SOURCE_FAMILY, "libreoffice", "舊", "新詞")
    held = _observation(
        VSCODE_SOURCE_FAMILY, "held", "舊{0}", "新詞{0}",
        tokens=["{0}"],
    )
    plans, _summary = derive_variable_structure_plans((held,))
    surface_sha256 = _id("neutral-source")
    expected_output_sha256 = _id("新詞{0}")
    projections = (
        _projection(
            godot,
            surface_sha256=surface_sha256,
            output_sha256=expected_output_sha256,
        ),
        _projection(
            libreoffice,
            surface_sha256=surface_sha256,
            output_sha256=(
                _id("冲突") if conflict_authority
                else expected_output_sha256),
        ),
        _projection(
            held,
            surface_sha256=surface_sha256,
            output_sha256=(
                held_output_sha256 or expected_output_sha256),
        ),
    )
    return (
        (godot, libreoffice, held),
        (_fragment(godot, "godot"),
         _fragment(libreoffice, "libreoffice")),
        plans,
        projections,
    )


def test_independent_models_generate_and_neutral_route_only_authorizes() -> None:
    """两个 family 独立生成后，TRAIN neutral 共识只负责授权 output SHA。"""
    observations, fragments, plans, projections = _material()
    models, stages, loso, summary = (
        derive_cross_source_transformation_feasibility(
            observations=observations,
            fragments=fragments,
            plans=plans,
            neutral_projections=projections,
        ))
    assert len(models) == 4
    assert stages[0]["outcome_counts"] == {
        "EXACT": 1, "UNKNOWN": 0, "WRONG": 0}
    assert stages[1]["outcome_counts"] == {
        "EXACT": 1, "UNKNOWN": 0, "WRONG": 0}
    assert summary["capability_outcome"] == (
        "PASS_NONZERO_AUTHORIZED_EXACT")
    assert summary["indexed_reference_mismatch_count"] == 0
    assert summary["partial_commit_count"] == 0
    held = next(item for item in loso
                if item["held_out_source_family"] == VSCODE_SOURCE_FAMILY)
    assert held["neutral_authorized_count"] == 1
    encoded = canonical_json_bytes({
        "models": models, "stages": stages, "loso": loso})
    assert "舊".encode("utf-8") not in encoded
    assert "新詞".encode("utf-8") not in encoded


def test_held_projection_output_sha_is_label_blind_and_conflict_defers() -> None:
    """held output SHA 不参与授权；TRAIN authority 冲突则完整退回 UNKNOWN。"""
    material = _material(held_output_sha256=_id("not-the-label"))
    left = derive_cross_source_transformation_feasibility(
        observations=material[0], fragments=material[1],
        plans=material[2], neutral_projections=material[3])
    baseline = _material()
    right = derive_cross_source_transformation_feasibility(
        observations=baseline[0], fragments=baseline[1],
        plans=baseline[2], neutral_projections=baseline[3])
    assert left[0] == right[0]
    left_vscode = next(
        item for item in left[2]
        if item["held_out_source_family"] == VSCODE_SOURCE_FAMILY)
    right_vscode = next(
        item for item in right[2]
        if item["held_out_source_family"] == VSCODE_SOURCE_FAMILY)
    assert left_vscode == right_vscode

    conflict = _material(conflict_authority=True)
    _models, stages, _loso, summary = (
        derive_cross_source_transformation_feasibility(
            observations=conflict[0], fragments=conflict[1],
            plans=conflict[2], neutral_projections=conflict[3]))
    assert stages[0]["outcome_counts"]["EXACT"] == 1
    assert stages[1]["outcome_counts"] == {
        "EXACT": 0, "UNKNOWN": 1, "WRONG": 0}
    assert summary["capability_outcome"] == "NE_ZERO_AUTHORIZED_EXACT"


def test_unscored_proposals_do_not_publish_or_depend_on_held_outcome() -> None:
    """无标签 proposal 接口不含 outcome，held label 变化也不改变 proposal。"""
    observations, fragments, plans, _projections = _material()
    baseline = derive_cross_source_transformation_unscored_proposals(
        observations=observations, fragments=fragments, plans=plans)
    changed = tuple({
        **item,
        "output_text": (
            "另一标签{0}" if item["source_family"] == VSCODE_SOURCE_FAMILY
            else item["output_text"]),
    } for item in observations)
    repeated = derive_cross_source_transformation_unscored_proposals(
        observations=changed, fragments=fragments, plans=plans)
    assert baseline == repeated
    assert baseline
    assert all("pre_authorization_outcome" not in item for item in baseline)


def test_external_optional_rewrite_uses_input_only_and_two_family_consensus(
        ) -> None:
    """外部输入不提供 output/plan，仍可由 TRAIN family 形成唯一共识。"""
    observations, fragments, plans, _projections = _material()
    held_inputs = ({
        "format_version": 1,
        "input_text": "舊{0}",
        "official_source_text": "Old{0}",
        "pair_id": _id("audacity-external-pair"),
        "record_kind": "NORMALIZATION_RECOVERY_V7_EXTERNAL_HELD_INPUT_V1",
        "source_family": "AUDACITY_PROJECT",
        "source_policy_scope": "AUDACITY_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
        "structure_tokens": ["{0}"],
    },)
    proposals, census = (
        derive_external_cross_source_optional_rewrite_proposals(
            observations=observations,
            fragments=fragments,
            plans=plans,
            held_inputs=held_inputs,
        ))
    assert census == {
        "held_input_count": 1,
        "proposed_count": 1,
        "deferred_count": 0,
        "indexed_reference_mismatch_count": 0,
        "partial_commit_count": 0,
        "structure_token_mismatch_count": 0,
    }
    assert proposals[0]["proposal_decision"] == (
        "PROPOSED_UNIQUE_MULTI_FAMILY_CONSENSUS")
    assert proposals[0]["proposal_output_text"] == "新詞{0}"
    assert proposals[0]["family_consensus_support_count"] >= 2
    assert proposals[0]["held_label_read_count"] == 0


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 audit publisher/reader 的小型稳定输出。"""
    outputs = {
        "model-representations.jsonl": ({
            "model_id": _id("model"), "record_kind": "MODEL"},),
        "stage-audit.jsonl": ({
            "record_kind": "STAGE", "stage_id": _id("stage")},),
        "loso-audit.jsonl": ({
            "loso_id": _id("loso"), "record_kind": "LOSO"},),
    }
    return outputs, {
        "audit_outcome": "FACILITY_PASS_CAPABILITY_NE",
        "transformation": {"capability_outcome": "NE_ZERO_AUTHORIZED_EXACT"},
    }


def _patch_audit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 synthetic audit test 与真实 K 盘 predecessor。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        audit, "_input_state",
        lambda **_kwargs: ({}, {}, {}, {}, {}))
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: _fake_outputs())


def _input_dirs(tmp_path: Path) -> list[Path]:
    """创建 publisher 所需的五个 synthetic input 目录。"""
    paths = [tmp_path / name for name in (
        "protocol", "variable", "context", "replay", "neutral")]
    for path in paths:
        path.mkdir()
    return paths


def _publish(
        tmp_path: Path,
        inputs: list[Path],
        ) -> tuple[Path, dict[str, object]]:
    """发布小型 synthetic transformation artifact。"""
    target = tmp_path / "audit"
    manifest = audit.publish_normalization_recovery_v7_cross_source_transformation_audit(
        run_root=tmp_path,
        training_protocol_dir=inputs[0],
        variable_structure_audit_dir=inputs[1],
        context_local_audit_dir=inputs[2],
        source_replay_audit_dir=inputs[3],
        neutral_source_projection_dir=inputs[4],
        target_dir=target,
    )
    return target, manifest


def _read(
        target: Path,
        inputs: list[Path],
        manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格回读小型 synthetic transformation artifact。"""
    return audit.read_normalization_recovery_v7_cross_source_transformation_audit(
        target,
        training_protocol_dir=inputs[0],
        variable_structure_audit_dir=inputs[1],
        context_local_audit_dir=inputs[2],
        source_replay_audit_dir=inputs[3],
        neutral_source_projection_dir=inputs[4],
        expected_manifest_sha256=manifest_sha256,
    )


def test_transformation_audit_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit 往返一致、拒绝覆盖，并拒绝 record 篡改。"""
    _patch_audit_inputs(monkeypatch)
    inputs = _input_dirs(tmp_path)
    target, published = _publish(tmp_path, inputs)
    manifest, outputs = _read(
        target, inputs, str(published["manifest_sha256"]))
    assert manifest == published
    assert manifest["status"] == (
        "TRAIN_ONLY_CROSS_SOURCE_TRANSFORMATION_NE_NOT_RUNTIME")
    assert outputs == _fake_outputs()[0]
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish(tmp_path, inputs)

    path = target / "loso-audit.jsonl"
    path.write_bytes(canonical_json_line({
        "loso_id": _id("tampered"), "record_kind": "LOSO"}))
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read(target, inputs, str(published["manifest_sha256"]))

    path.write_bytes(canonical_json_line(_fake_outputs()[0][
        "loso-audit.jsonl"][0]))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="manifest 字段漂移"):
        _read(target, inputs, hashlib.sha256(encoded).hexdigest())
