"""Publish and strictly read the recovery-v8 three-ledger TRAIN protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_coverage import (
    V8_OBSERVATION_PACK_MANIFEST_SHA256,
    read_normalization_recovery_v8_observation_coverage_with_observations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
    V8_TRAINING_RECORD_FILES,
    derive_normalization_recovery_v8_training_records,
    summarize_normalization_recovery_v8_training_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_V1")
NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_STATUS = (
    "THREE_LEDGER_PROTOCOL_FROZEN_NOT_TRAINED")
V8_OBSERVATION_COVERAGE_MANIFEST_SHA256 = (
    "e5a73ae895794a6b4a556ec5e9209ba28d97cc674996e056e9d48a8de46634a5")


def _sha256(payload: bytes) -> str:
    """Return a SHA-256 hex digest."""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """Require an explicit, existing K-drive work root."""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 TRAIN protocol run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """Resolve a protocol path and keep it inside the run root."""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 TRAIN protocol {label} 越出run root") from error
    return path


def _overlap(left: Path, right: Path) -> bool:
    """Return whether two resolved paths are equal or ancestor-related."""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """Write canonical JSONL without overwrite."""
    if not values:
        raise BroadQaExternalDataError("v8 TRAIN protocol output 为空")
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """Strictly read nonempty canonical JSONL."""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 TRAIN protocol {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 TRAIN protocol {label} 不可读") from error
    if not values:
        raise BroadQaExternalDataError(f"v8 TRAIN protocol {label} 为空")
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """Commit one materialized protocol file."""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _manifest(
        *, files: list[dict[str, object]], summary: dict[str, object],
        ) -> dict[str, object]:
    """Build the frozen pre-learner protocol manifest."""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_KIND,
        "authorization_is_not_learning": 1,
        "deferred_contract": {
            "all_conflicts_deferred": 1,
            "all_single_family_candidates_deferred": 1,
            "exact_mapping_is_control_only": 1,
            "identity_only_changed_rule_veto_required": 1,
            "per_observation_layout_morphology_defer_required": 1,
        },
        "execution_contract": {
            "exception_count_required": 0,
            "indexed_interpreter_required": 1,
            "indexed_reference_mismatch_count_required": 0,
            "partial_commit_count_required": 0,
            "reference_interpreter_required": 1,
            "structure_mismatch_count_required": 0,
            "wrong_count_required": 0,
        },
        "files": files,
        "format_version": 1,
        "inputs": {
            "observation_coverage_manifest_sha256": (
                V8_OBSERVATION_COVERAGE_MANIFEST_SHA256),
            "observation_pack_manifest_sha256": (
                V8_OBSERVATION_PACK_MANIFEST_SHA256),
        },
        "loso_contract": {
            "direction_count": len(V8_TRAIN_FAMILIES),
            "family_consensus_minimum": 2,
            "held_out_output_may_influence_rule_construction": 0,
            "support_2_supporter_holdout_expected": "UNKNOWN",
            "support_2_non_supporter_holdout_expected": "UNKNOWN_NO_CASE",
            "support_3_any_holdout_expected": "EXACT",
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_started": 0,
        "source_families": list(V8_TRAIN_FAMILIES),
        "status": NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "three_ledger_contract": {
            "LAYOUT_MORPHOLOGY_OBLIGATION": (
                "MULTI_FAMILY_STRUCTURE_PRESERVATION_ONLY"),
            "ORTHOGRAPHIC_ATOM": "MULTI_FAMILY_UNIQUE_OUTPUT_ONLY",
            "SOURCE_CONDITIONED_LEXICAL_ATOM": (
                "MULTI_FAMILY_UNIQUE_CHANGED_OUTPUT_ONLY"),
        },
        "training_executed": 0,
        "vlc_final_read_count": 0,
    }


def _source_state(
        *, coverage_dir: Path, observation_dir: Path,
        v2_roster_dir: Path, v1_roster_dir: Path,
        v1_content_audit_dir: Path, v2_content_audit_dir: Path,
        source_overlap_dir: Path, qbittorrent_source_pack_dir: Path,
        stellarium_source_pack_dir: Path, keepassxc_source_pack_dir: Path,
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """Strictly reread both sealed inputs and their full predecessor chain."""
    _coverage_manifest, coverage_outputs, observation_outputs = (
        read_normalization_recovery_v8_observation_coverage_with_observations(
            coverage_dir,
            observation_dir=observation_dir,
            v2_roster_dir=v2_roster_dir,
            v1_roster_dir=v1_roster_dir,
            v1_content_audit_dir=v1_content_audit_dir,
            v2_content_audit_dir=v2_content_audit_dir,
            source_overlap_dir=source_overlap_dir,
            qbittorrent_source_pack_dir=qbittorrent_source_pack_dir,
            stellarium_source_pack_dir=stellarium_source_pack_dir,
            keepassxc_source_pack_dir=keepassxc_source_pack_dir,
            expected_manifest_sha256=(
                V8_OBSERVATION_COVERAGE_MANIFEST_SHA256),
        ))
    return coverage_outputs, observation_outputs


def _derive_from_sources(**paths: Path) -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """Rederive the complete protocol from sealed K-drive sources."""
    coverage, observations = _source_state(**paths)
    return derive_normalization_recovery_v8_training_records(
        coverage, observations)


def publish_normalization_recovery_v8_training_protocol(
        *, run_root: str | Path, coverage_dir: str | Path,
        observation_dir: str | Path, v2_roster_dir: str | Path,
        v1_roster_dir: str | Path, v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path, source_overlap_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """Publish the v8 TRAIN protocol once, before any learner execution."""
    root = _require_k_root(run_root)
    names = (
        "coverage_dir", "observation_dir", "v2_roster_dir", "v1_roster_dir",
        "v1_content_audit_dir", "v2_content_audit_dir", "source_overlap_dir",
        "qbittorrent_source_pack_dir", "stellarium_source_pack_dir",
        "keepassxc_source_pack_dir", "target_dir",
    )
    raw = (
        coverage_dir, observation_dir, v2_roster_dir, v1_roster_dir,
        v1_content_audit_dir, v2_content_audit_dir, source_overlap_dir,
        qbittorrent_source_pack_dir, stellarium_source_pack_dir,
        keepassxc_source_pack_dir, target_dir,
    )
    resolved = tuple(_within(root, value, label=name)
                     for name, value in zip(names, raw))
    *inputs, target = resolved
    if (target.exists() or any(not path.is_dir() for path in inputs)
            or any(_overlap(target, path) for path in inputs)):
        raise BroadQaExternalDataError("v8 TRAIN protocol input/target path 非法")
    source_paths = dict(zip(names[:-1], inputs))
    outputs, summary = _derive_from_sources(**source_paths)
    target.mkdir()
    files = []
    for name, role in V8_TRAINING_RECORD_FILES:
        values = outputs[name]
        path = target / name
        _write_jsonl(path, values)
        files.append(_artifact(path, role=role, count=len(values)))
    manifest = _manifest(files=files, summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def _read_manifest(
        protocol_dir: str | Path, *, expected_manifest_sha256: str,
        ) -> tuple[Path, dict[str, object], bytes]:
    """Read a canonical manifest bound by an external SHA."""
    root = Path(protocol_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 TRAIN protocol manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in expected_manifest_sha256)
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError("v8 TRAIN protocol manifest identity 漂移")
    return root, stored, encoded


def read_normalization_recovery_v8_learner_input(
        protocol_dir: str | Path, *, expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """Read only sealed protocol files; do not open source or held-out data."""
    root, stored, encoded = _read_manifest(
        protocol_dir, expected_manifest_sha256=expected_manifest_sha256)
    outputs = {}
    files = []
    for name, role in V8_TRAINING_RECORD_FILES:
        values = _read_jsonl(root / name, label=role)
        outputs[name] = values
        files.append(_artifact(root / name, role=role, count=len(values)))
    summary = summarize_normalization_recovery_v8_training_records(outputs)
    if stored != _manifest(files=files, summary=summary):
        raise BroadQaExternalDataError("v8 TRAIN protocol learner material 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


def read_normalization_recovery_v8_training_protocol(
        protocol_dir: str | Path, *, expected_manifest_sha256: str,
        coverage_dir: str | Path, observation_dir: str | Path,
        v2_roster_dir: str | Path, v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path, v2_content_audit_dir: str | Path,
        source_overlap_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """Strictly rederive all protocol fields from the sealed source chain."""
    manifest, stored_outputs = read_normalization_recovery_v8_learner_input(
        protocol_dir, expected_manifest_sha256=expected_manifest_sha256)
    source_paths = {
        "coverage_dir": Path(coverage_dir).resolve(),
        "observation_dir": Path(observation_dir).resolve(),
        "v2_roster_dir": Path(v2_roster_dir).resolve(),
        "v1_roster_dir": Path(v1_roster_dir).resolve(),
        "v1_content_audit_dir": Path(v1_content_audit_dir).resolve(),
        "v2_content_audit_dir": Path(v2_content_audit_dir).resolve(),
        "source_overlap_dir": Path(source_overlap_dir).resolve(),
        "qbittorrent_source_pack_dir": Path(qbittorrent_source_pack_dir).resolve(),
        "stellarium_source_pack_dir": Path(stellarium_source_pack_dir).resolve(),
        "keepassxc_source_pack_dir": Path(keepassxc_source_pack_dir).resolve(),
    }
    expected_outputs, summary = _derive_from_sources(**source_paths)
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v8 TRAIN protocol records 漂移")
    stored_without_sha = {key: value for key, value in manifest.items()
                          if key != "manifest_sha256"}
    files = [_artifact(Path(protocol_dir).resolve() / name,
                       role=role, count=len(expected_outputs[name]))
             for name, role in V8_TRAINING_RECORD_FILES]
    if stored_without_sha != _manifest(files=files, summary=summary):
        raise BroadQaExternalDataError("v8 TRAIN protocol fields 漂移")
    return manifest, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_KIND",
    "NORMALIZATION_RECOVERY_V8_TRAINING_PROTOCOL_STATUS",
    "V8_OBSERVATION_COVERAGE_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_training_protocol",
    "read_normalization_recovery_v8_learner_input",
    "read_normalization_recovery_v8_training_protocol",
]
