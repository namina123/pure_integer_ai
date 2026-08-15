"""Tests for the recovery-v8 three-ledger TRAIN protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_training_protocol as protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    derive_normalization_recovery_v8_training_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


_FAMILIES = (
    "KEEPASSXC_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _mapping(
        label: str, *, kind: str, input_text: str, outputs: tuple[
            tuple[str, tuple[str, ...]], ...],
        status: str, official_source: str = "",
        ) -> dict[str, object]:
    """Build a synthetic coverage mapping candidate."""
    families = sorted({family for _output, support in outputs for family in support})
    return {
        "candidate_id": _identity(label),
        "candidate_kind": kind,
        "candidate_status": status,
        "format_version": 1,
        "input_text": input_text,
        "official_source_text": official_source,
        "outputs": [{
            "family_record_counts": {family: 1 for family in support},
            "identity_record_count": int(input_text == output),
            "nonidentity_record_count": int(input_text != output),
            "output_text": output,
            "support_families": list(support),
            "support_family_count": len(support),
        } for output, support in outputs],
        "record_kind": "NORMALIZATION_RECOVERY_V8_MAPPING_CANDIDATE_V1",
        "support_families": families,
    }


def _atom(
        label: str, *, source: str,
        outputs: tuple[tuple[str, tuple[str, ...]], ...], status: str,
        ) -> dict[str, object]:
    """Build a synthetic orthographic atom candidate."""
    families = sorted({family for _output, support in outputs for family in support})
    return {
        "candidate_id": _identity(label),
        "candidate_kind": "ORTHOGRAPHIC_ATOM",
        "candidate_status": status,
        "format_version": 1,
        "input_atom": source,
        "outputs": [{
            "family_record_counts": {family: 1 for family in support},
            "output_atom": output,
            "support_families": list(support),
            "support_family_count": len(support),
        } for output, support in outputs],
        "record_kind": "NORMALIZATION_RECOVERY_V8_ORTHOGRAPHIC_ATOM_CANDIDATE_V1",
        "support_families": families,
    }


def _structure(
        label: str, *, tokens: list[str], families: tuple[str, ...], status: str,
        ) -> dict[str, object]:
    """Build a synthetic structure obligation candidate."""
    return {
        "candidate_id": _identity(label),
        "candidate_kind": "STRUCTURE_OBLIGATION",
        "candidate_status": status,
        "family_record_counts": {family: 1 for family in families},
        "format_version": 1,
        "record_kind": "NORMALIZATION_RECOVERY_V8_STRUCTURE_OBLIGATION_CANDIDATE_V1",
        "structure_tokens": tokens,
        "support_families": list(families),
        "support_family_count": len(families),
    }


def _observation(
        label: str, *, family: str, eligible: int, tokens: list[str],
        ) -> dict[str, object]:
    """Build a synthetic observation used only by the defer ledger."""
    return {
        "eligibility": {
            "exclusion_reasons": [] if eligible else ["NO_HAN_BOTH"],
            "pair_features": {"v8_training_eligible": eligible},
        },
        "observation_id": _identity(f"observation-{label}"),
        "source_family": family,
        "source_pair_id": _identity(f"pair-{label}"),
        "zh_hans_structure_tokens": tokens,
        "zh_hant_structure_tokens": tokens,
    }


def _synthetic_sources() -> tuple[
        dict[str, tuple[dict[str, object], ...]],
        dict[str, tuple[dict[str, object], ...]],
        ]:
    """Return coverage with support-2/support-3, conflict and defer cases."""
    support_2 = _FAMILIES[:2]
    coverage = {
        "exact-input-mappings.jsonl": (
            _mapping(
                "exact-control", kind="EXACT_INPUT_MAPPING",
                input_text="檔", outputs=(("档", support_2),),
                status="MULTI_FAMILY_UNIQUE_OUTPUT"),
        ),
        "source-conditioned-mappings.jsonl": (
            _mapping(
                "lexical-support-3", kind="SOURCE_CONDITIONED_MAPPING",
                input_text="檔案", outputs=(("文件", _FAMILIES),),
                status="MULTI_FAMILY_UNIQUE_OUTPUT", official_source="File"),
            _mapping(
                "lexical-support-2", kind="SOURCE_CONDITIONED_MAPPING",
                input_text="資料夾", outputs=(("文件夹", support_2),),
                status="MULTI_FAMILY_UNIQUE_OUTPUT", official_source="Folder"),
            _mapping(
                "lexical-identity", kind="SOURCE_CONDITIONED_MAPPING",
                input_text="地址", outputs=(("地址", _FAMILIES),),
                status="MULTI_FAMILY_UNIQUE_OUTPUT", official_source="Address"),
            _mapping(
                "lexical-conflict", kind="SOURCE_CONDITIONED_MAPPING",
                input_text="佇列", outputs=(
                    ("队列", support_2), ("队伍", (_FAMILIES[2],))),
                status="MULTI_FAMILY_CONFLICT", official_source="Queue"),
        ),
        "orthographic-atoms.jsonl": (
            _atom(
                "atom-authorized", source="檔", outputs=(("档", support_2),),
                status="MULTI_FAMILY_UNIQUE_OUTPUT"),
            _atom(
                "atom-conflict", source="著", outputs=(
                    ("着", support_2), ("著", (_FAMILIES[2],))),
                status="MULTI_FAMILY_CONFLICT"),
        ),
        "structure-obligations.jsonl": (
            _structure(
                "structure-authorized", tokens=["%1"], families=_FAMILIES,
                status="MULTI_FAMILY_OBSERVED"),
            _structure(
                "structure-single", tokens=["HTML_OPEN:a", "HTML_CLOSE:a"],
                families=(_FAMILIES[0],), status="SINGLE_FAMILY_OBSERVED"),
        ),
        "coverage-census.jsonl": ({"record_kind": "SYNTHETIC_CENSUS"},),
    }
    observations = {
        "qbittorrent-observations.jsonl": (
            _observation("qbit", family=_FAMILIES[1], eligible=1, tokens=["%1"]),
        ),
        "stellarium-observations.jsonl": (
            _observation("stell", family=_FAMILIES[2], eligible=0, tokens=[]),
        ),
        "keepassxc-observations.jsonl": (
            _observation(
                "keep", family=_FAMILIES[0], eligible=1,
                tokens=["HTML_OPEN:a", "HTML_CLOSE:a"]),
        ),
    }
    return coverage, observations


def test_three_ledgers_defer_every_unsafe_candidate_and_freeze_loso() -> None:
    """Support-3 transfers; support-2 disappears on a supporter holdout."""
    coverage, observations = _synthetic_sources()
    outputs, summary = derive_normalization_recovery_v8_training_records(
        coverage, observations)
    assert summary["authorization_ledger_counts"] == {
        "LAYOUT_MORPHOLOGY_OBLIGATION": 1,
        "ORTHOGRAPHIC_ATOM": 1,
        "SOURCE_CONDITIONED_LEXICAL_ATOM": 2,
    }
    assert summary["authorization_count"] == 4
    assert summary["exact_input_control_count"] == 1
    assert summary["deferred_candidate_reason_counts"] == {
        "EXACT_INPUT_CONTROL_ONLY": 1,
        "IDENTITY_ONLY_CHANGED_RULE_VETO": 1,
        "OUTPUT_CONFLICT": 2,
        "SINGLE_FAMILY_AUTHORITY": 1,
    }
    assert summary["deferred_observation_reason_counts"] == {
        "PER_OBSERVATION_LAYOUT_MORPHOLOGY_UNRESOLVED": 2,
        "V8_TRAINING_INELIGIBLE": 1,
    }

    lexical = outputs["authorized-source-conditioned-lexical-atoms.jsonl"]
    support_3 = next(item for item in lexical if item["support_family_count"] == 3)
    support_2 = next(item for item in lexical if item["support_family_count"] == 2)
    plans = outputs["family-loso-plan.jsonl"]
    assert {item["expected_behavior"] for item in plans
            if item["authorization_id"] == support_3["authorization_id"]} == {"EXACT"}
    support_2_plans = [item for item in plans
                       if item["authorization_id"] == support_2["authorization_id"]]
    assert all(item["expected_behavior"] == "UNKNOWN" for item in support_2_plans)
    assert sum(item["expected_rule_present_after_holdout"]
               for item in support_2_plans) == 1
    assert all(item["held_out_output_read_count"] == 0 for item in plans)


def _input_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create the publisher's ten required sealed input directories."""
    values = {}
    for name in (
            "coverage_dir", "observation_dir", "v2_roster_dir", "v1_roster_dir",
            "v1_content_audit_dir", "v2_content_audit_dir", "source_overlap_dir",
            "qbittorrent_source_pack_dir", "stellarium_source_pack_dir",
            "keepassxc_source_pack_dir"):
        path = tmp_path / name
        path.mkdir()
        values[name] = path
    return values


def test_publish_learner_read_strict_read_duplicate_and_tamper_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """Publisher and both readers preserve the frozen protocol boundary."""
    coverage, observations = _synthetic_sources()
    monkeypatch.setattr(protocol, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        protocol, "_source_state", lambda **kwargs: (coverage, observations))
    paths = _input_dirs(tmp_path)
    target = tmp_path / "protocol"
    published = protocol.publish_normalization_recovery_v8_training_protocol(
        run_root=tmp_path, target_dir=target, **paths)
    assert published["status"] == "THREE_LEDGER_PROTOCOL_FROZEN_NOT_TRAINED"
    assert published["summary"]["authorization_count"] == 4
    assert published["execution_contract"]["wrong_count_required"] == 0
    learner_manifest, learner_outputs = (
        protocol.read_normalization_recovery_v8_learner_input(
            target, expected_manifest_sha256=published["manifest_sha256"]))
    assert learner_manifest == published
    assert learner_outputs["family-loso-plan.jsonl"]
    strict_manifest, strict_outputs = (
        protocol.read_normalization_recovery_v8_training_protocol(
            target, expected_manifest_sha256=published["manifest_sha256"],
            **paths))
    assert strict_manifest == published
    assert strict_outputs == learner_outputs
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        protocol.publish_normalization_recovery_v8_training_protocol(
            run_root=tmp_path, target_dir=target, **paths)

    deferred = target / "deferred-candidates.jsonl"
    lines = deferred.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["authorization_allowed"] = 1
    lines[0] = canonical_json_line(value)
    deferred.write_bytes(b"".join(lines))
    with pytest.raises(BroadQaExternalDataError, match="learner material"):
        protocol.read_normalization_recovery_v8_learner_input(
            target, expected_manifest_sha256=published["manifest_sha256"])


def test_protocol_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """Formal publication may never fall back from the K-drive work disk."""
    target = tmp_path / "protocol"
    with pytest.raises(BroadQaExternalDataError, match="K盘"):
        protocol.publish_normalization_recovery_v8_training_protocol(
            run_root=tmp_path,
            coverage_dir=tmp_path / "coverage",
            observation_dir=tmp_path / "observation",
            v2_roster_dir=tmp_path / "roster-v2",
            v1_roster_dir=tmp_path / "roster-v1",
            v1_content_audit_dir=tmp_path / "content-v1",
            v2_content_audit_dir=tmp_path / "content-v2",
            source_overlap_dir=tmp_path / "overlap",
            qbittorrent_source_pack_dir=tmp_path / "qbit",
            stellarium_source_pack_dir=tmp_path / "stell",
            keepassxc_source_pack_dir=tmp_path / "keep",
            target_dir=target,
        )
    assert not target.exists()
