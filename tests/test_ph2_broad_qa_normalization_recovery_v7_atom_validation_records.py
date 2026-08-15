"""覆盖 Audacity external atom-validation 标签盲授权与 v2 评分。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_validation_family as family,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_validation_runner as runner,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit import (
    NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment_v2 import (
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS,
)

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_records import (
    derive_audacity_atom_validation_authorizations,
    score_audacity_atom_validation_authorizations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    AUDACITY_SOURCE_FAMILY,
    AUDACITY_SOURCE_POLICY_SCOPE,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND,
    NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS,
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
    """构造最小 TRAIN observation。"""
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": int(input_text == output_text),
        "input_text": input_text,
        "observation_id": _id(f"observation-{family}-{suffix}"),
        "output_text": output_text,
        "source_family": family,
        "source_pair_id": _id(f"pair-{family}-{suffix}"),
        "source_policy_scope": V5_SOURCE_POLICY_BY_FAMILY[family],
        "structure_tokens": [] if tokens is None else tokens,
    }


def _fragment(observation: dict[str, object], suffix: str) -> dict[str, object]:
    """构造 frozen multi-scalar lexical EDIT_CORE。"""
    return {
        "fragment_id": _id(f"fragment-{suffix}"),
        "fragment_kind": "EDIT_CORE",
        "input_text": "外掛程式",
        "observation_id": observation["observation_id"],
        "output_text": "插件",
        "source_family": observation["source_family"],
    }


def _training() -> tuple[
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        ]:
    """构造两家、三 occurrence 的 TRAIN lexical route。"""
    godot_a = _observation(
        GODOT_SOURCE_FAMILY, "a", "外掛程式{0}", "插件{0}",
        tokens=["{0}"])
    godot_b = _observation(
        GODOT_SOURCE_FAMILY, "b", "外掛程式", "插件")
    libreoffice = _observation(
        LIBREOFFICE_SOURCE_FAMILY, "a", "外掛程式{0}", "插件{0}",
        tokens=["{0}"])
    vscode = _observation(
        VSCODE_SOURCE_FAMILY, "a", "外掛程式{0}", "插件{0}",
        tokens=["{0}"])
    plans, _summary = derive_variable_structure_plans((vscode,))
    return (
        (godot_a, godot_b, libreoffice, vscode),
        (_fragment(godot_a, "godot"),
         _fragment(libreoffice, "libreoffice")),
        plans,
    )


def _held(input_text: str = "外掛程式{0}") -> tuple[dict[str, object], ...]:
    """构造不含 zh-CN/output 的 Audacity external held input。"""
    return ({
        "format_version": 1,
        "input_text": input_text,
        "official_source_text": "Plugin{0}",
        "pair_id": _id(f"audacity-{input_text}"),
        "record_kind": "NORMALIZATION_RECOVERY_V7_EXTERNAL_HELD_INPUT_V1",
        "source_family": AUDACITY_SOURCE_FAMILY,
        "source_policy_scope": AUDACITY_SOURCE_POLICY_SCOPE,
        "structure_tokens": ["{0}"],
    },)


def test_external_authorization_is_label_blind_then_scores_exact() -> None:
    """授权阶段零 label read，评分阶段才形成 changed EXACT。"""
    observations, fragments, plans = _training()
    authorizations, census = derive_audacity_atom_validation_authorizations(
        observations=observations,
        fragments=fragments,
        plans=plans,
        held_inputs=_held(),
        opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)},
    )
    assert census["held_label_read_count"] == 0
    assert census["authorized_count"] == 1
    assert authorizations[0]["authorization_decision"] == "AUTHORIZED"
    assert authorizations[0]["held_label_read_count"] == 0
    assert authorizations[0]["stable_lexical_atom_count"] == 1

    pair_id = str(authorizations[0]["pair_id"])
    records, scoring = score_audacity_atom_validation_authorizations(
        authorizations,
        labels_by_pair={pair_id: ("外掛程式{0}", "插件{0}")},
        expected_denominator_count=1,
    )
    assert scoring["outcome"] == (
        "PASS_NONZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG")
    assert scoring["authorized_changed_exact_output_count"] == 1
    assert scoring["outcome_counts"] == {
        "EXACT": 1, "UNKNOWN": 0, "WRONG": 0}
    assert records[0]["surface_published"] == 0


def test_wrong_label_changes_only_scoring_and_identity_exact_is_not_pass() -> None:
    """label 不反向改变授权；identity-only EXACT 只能得到 NE。"""
    observations, fragments, plans = _training()
    authorizations, _census = derive_audacity_atom_validation_authorizations(
        observations=observations,
        fragments=fragments,
        plans=plans,
        held_inputs=_held(),
        opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)},
    )
    pair_id = str(authorizations[0]["pair_id"])
    _records, wrong = score_audacity_atom_validation_authorizations(
        authorizations,
        labels_by_pair={pair_id: ("外掛程式{0}", "其他{0}")},
        expected_denominator_count=1,
    )
    assert wrong["outcome"] == "FAIL_HARD_CONJUNCT"
    assert wrong["outcome_counts"]["WRONG"] == 1

    deferred = ({
        **authorizations[0],
        "authorization_decision": "DEFERRED",
        "proposal_output_sha256": _id("identity-output"),
        "proposal_output_text": "外掛程式{0}",
    },)
    _records, identity = score_audacity_atom_validation_authorizations(
        deferred,
        labels_by_pair={pair_id: ("外掛程式{0}", "外掛程式{0}")},
        expected_denominator_count=1,
    )
    assert identity["identity_exact_output_count"] == 1
    assert identity["authorized_changed_exact_output_count"] == 0
    assert identity["outcome"] == (
        "NE_ZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG")


def test_marked_morphology_defers_external_proposal() -> None:
    """官方 source 的 marked morphology 必须在授权前完整 defer。"""
    observations, fragments, plans = _training()
    held = ({**_held()[0], "official_source_text": "Plugins{0}"},)
    authorizations, census = derive_audacity_atom_validation_authorizations(
        observations=observations,
        fragments=fragments,
        plans=plans,
        held_inputs=held,
        opencc_routes={},
        morphology_by_form={"plugins": (("plugin", "N;PL"),)},
    )
    assert census["authorized_count"] == 0
    assert authorizations[0]["reason_counts"] == {
        "MARKED_MORPHOLOGY_UNRESOLVED": 1}


def _write_manifest(path: Path, value: dict[str, object]) -> str:
    """写入规范 synthetic manifest 并返回 SHA。"""
    path.mkdir()
    encoded = canonical_json_line(value)
    (path / "manifest.json").write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_family_freeze_binds_manifests_code_and_pushed_git(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """family 只读 predecessor manifest，并重算 code/public Git identity。"""
    source_dir = tmp_path / "source"
    source_sha = _write_manifest(source_dir, {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_PACK_KIND,
        "parser_summary": {"plain_pair_count": 4404},
        "source_family": AUDACITY_SOURCE_FAMILY,
        "source_policy_scope": AUDACITY_SOURCE_POLICY_SCOPE,
        "status": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_SOURCE_STATUS,
        "validation_state": {
            "candidate_or_runtime_read_count": 0,
            "formal_label_jsonl_materialized": 0,
            "validation_run_count": 0,
        },
    })
    atom_dir = tmp_path / "atom"
    atom_sha = _write_manifest(atom_dir, {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
        "candidate_family_formal_run_count": 0,
        "status": "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME",
        "summary": {"identifiability": {"scoring": {
            "outcome_counts": {"EXACT": 2, "UNKNOWN": 12, "WRONG": 0}}}},
    })
    commitment_dir = tmp_path / "commitment"
    commitment_sha = _write_manifest(commitment_dir, {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_KIND,
        "denominator": {
            "record_count": 4404,
            "source_family": AUDACITY_SOURCE_FAMILY,
        },
        "gates": {
            "authorized_changed_exact_output_count_min": 1,
            "wrong_output_count_max": 0,
        },
        "inputs": {
            "atom_identifiability_manifest_sha256": atom_sha,
            "audacity_source_pack_manifest_sha256": source_sha,
        },
        "status": NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_COMMITMENT_V2_STATUS,
    })
    family_v1_dir = tmp_path / "family-v1"
    family_v1_sha = _write_manifest(family_v1_dir, {
        "artifact_kind": family.NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_KIND,
        "format_version": 1,
        "inputs": {
            "atom_identifiability_manifest_sha256": atom_sha,
            "audacity_source_pack_manifest_sha256": source_sha,
            "commitment_v2_manifest_sha256": commitment_sha,
        },
        "status": family.NORMALIZATION_RECOVERY_V7_ATOM_VALIDATION_FAMILY_V1_STATUS,
        "validation_reads": {
            "audacity_identity_raw_or_translation_read_count": 0,
            "validation_run_count": 0,
            "zh_cn_label_read_count": 0,
        },
    })
    monkeypatch.setattr(
        family, "AUDACITY_ATOM_VALIDATION_SOURCE_MANIFEST_SHA256", source_sha)
    monkeypatch.setattr(
        family, "AUDACITY_ATOM_IDENTIFIABILITY_MANIFEST_SHA256", atom_sha)
    monkeypatch.setattr(
        family, "AUDACITY_ATOM_VALIDATION_COMMITMENT_V2_MANIFEST_SHA256",
        commitment_sha)
    monkeypatch.setattr(
        family, "AUDACITY_ATOM_VALIDATION_FAMILY_V1_MANIFEST_SHA256",
        family_v1_sha)
    monkeypatch.setattr(family, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(family, "_repository_identity", lambda _root: {
        "head_commit_sha1": "a" * 40,
        "origin_master_commit_sha1": "a" * 40,
        "remote_origin_url": "https://example.invalid/repository.git",
        "tracked_worktree_clean": 1,
    })
    monkeypatch.setattr(family, "_code_identity", lambda _root: ([{
        "bytes": 1,
        "relative_path": "bearing.py",
        "sha256": "b" * 64,
    }], "c" * 64))
    target = tmp_path / "family"
    arguments = {
        "repository_root": tmp_path,
        "source_pack_dir": source_dir,
        "expected_source_manifest_sha256": source_sha,
        "atom_audit_dir": atom_dir,
        "expected_atom_manifest_sha256": atom_sha,
        "commitment_v2_dir": commitment_dir,
        "expected_commitment_v2_manifest_sha256": commitment_sha,
        "family_v1_dir": family_v1_dir,
        "expected_family_v1_manifest_sha256": family_v1_sha,
    }
    published = family.publish_audacity_atom_validation_family_freeze(
        run_root=tmp_path, target_dir=target, **arguments)
    restored = family.read_audacity_atom_validation_family_freeze(
        target,
        expected_manifest_sha256=str(published["manifest_sha256"]),
        **arguments)
    assert restored == published
    assert restored["validation_reads"]["zh_cn_label_read_count"] == 0
    assert restored["authorization_protocol"][
        "held_output_or_held_derived_plan_allowed"] == 0
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        family.publish_audacity_atom_validation_family_freeze(
            run_root=tmp_path, target_dir=target, **arguments)

    stored = json.loads((target / "manifest.json").read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="fields/code 漂移"):
        family.read_audacity_atom_validation_family_freeze(
            target,
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            **arguments)


def test_runner_writes_authorization_before_label_and_consumes_failure_identity(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """runner guard/authorization 先于 label，且同一 publication 不可重跑。"""
    roots = {}
    for name in ("family", "family-v1", "source", "atom", "commitment"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    training_roots = []
    for ordinal in range(11):
        path = tmp_path / f"training-{ordinal}"
        path.mkdir()
        training_roots.append(path)
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    publication = tmp_path / "publication"
    monkeypatch.setattr(runner, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        runner, "read_audacity_atom_validation_family_freeze",
        lambda *args, **kwargs: {
            "denominator": {"record_count": 1},
            "family_commitment_sha256": "f" * 64,
            "publication_contract": {"relative_path": "publication"},
        })
    monkeypatch.setattr(
        runner, "read_normalization_recovery_v7_atom_identifiability_audit_state",
        lambda *args, **kwargs: (
            {"manifest_sha256": "a" * 64}, {}, {
                "observations": (), "fragments": (), "plans": (),
                "opencc_routes": {}, "morphology": {},
            }))
    held = ({"pair_id": "1" * 64},)
    monkeypatch.setattr(
        runner, "read_audacity_atom_validation_held_inputs_after_family_freeze",
        lambda *args, **kwargs: (
            {"manifest_sha256": "s" * 64}, held, {
                "zh_hans_translation_file_read_count": 0}))
    authorization = ({
        "authorization_id": "2" * 64,
        "pair_id": "1" * 64,
        "surface_published": 0,
    },)
    monkeypatch.setattr(
        runner, "derive_audacity_atom_validation_authorizations",
        lambda **kwargs: (authorization, {
            "authorization_count": 1,
            "held_label_read_count": 0,
        }))

    def materialize(*args, **kwargs):
        assert (publication / "run-000001.authorization.json").is_file()
        return {"1" * 64: ("input", "output")}, {
            "zh_hans_translation_file_read_count": 1}

    monkeypatch.setattr(
        runner,
        "materialize_audacity_atom_validation_labels_after_authorization_freeze",
        materialize)
    monkeypatch.setattr(
        runner, "score_audacity_atom_validation_authorizations",
        lambda *args, **kwargs: (({
            "outcome": "UNKNOWN", "surface_published": 0},), {
                "denominator_count": 1,
                "outcome": "NE_ZERO_AUTHORIZED_CHANGED_EXACT_ZERO_WRONG",
            }))
    arguments = {
        "run_root": tmp_path,
        "repository_root": tmp_path,
        "family_freeze_dir": roots["family"],
        "expected_family_manifest_sha256": "3" * 64,
        "source_pack_dir": roots["source"],
        "expected_source_manifest_sha256": "4" * 64,
        "atom_audit_dir": roots["atom"],
        "expected_atom_manifest_sha256": "5" * 64,
        "commitment_v2_dir": roots["commitment"],
        "expected_commitment_v2_manifest_sha256": "6" * 64,
        "family_v1_dir": roots["family-v1"],
        "expected_family_v1_manifest_sha256": "7" * 64,
        "training_protocol_dir": training_roots[0],
        "variable_structure_audit_dir": training_roots[1],
        "neutral_semantic_source_audit_dir": training_roots[2],
        "godot_source_pack_dir": training_roots[3],
        "libreoffice_source_pack_dir": training_roots[4],
        "vscode_source_pack_dir": training_roots[5],
        "thunderbird_source_pack_dir": training_roots[6],
        "vscode_source_archive_path": archive,
        "vscode_source_root": training_roots[7],
        "typescript_parser_root": training_roots[8],
        "opencc_source_pack_dir": training_roots[9],
        "unimorph_english_dir": training_roots[10],
        "publication_dir": publication,
    }
    aggregate = runner.run_audacity_atom_validation_once(**arguments)
    assert aggregate["validation_run_count"] == 1
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.authorization.json").is_file()
    assert (publication / "run-000001.aggregate.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        runner.run_audacity_atom_validation_once(**arguments)
