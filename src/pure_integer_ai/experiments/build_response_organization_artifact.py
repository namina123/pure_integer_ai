"""Build and validate a training-bound OASST1 response organization artifact."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.conversation_response_organization import (
    ResponseOrganizationError,
    ResponseOrganizationModel,
    learn_response_organization_model,
    profile_response_surface,
    response_feature_counts,
)
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    create_new_run_root,
    open_existing_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)


ARTIFACT_KIND = "OASST1_RESPONSE_ORGANIZATION_V1"
MODEL_FILE = "response_organization_model.int"
HELDOUT_FILE = "response_organization_heldout.json"
MANIFEST_FILE = "response_organization_manifest.json"
_COURSE_FORMATS = frozenset({
    "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2",
    "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2",
})
_REQUIRED_FEATURES = (
    "code", "heading", "html", "list", "mixed", "paragraph", "table")


# object-model: exception; interop=response-organization-artifact-v1
class ResponseOrganizationArtifactError(ValueError):
    """Artifact source, binding, publication, or validation failed."""


# object-model: value; representation=struct; interop=response-organization-artifact-v1
@dataclass(frozen=True, slots=True)
class ResponseOrganizationArtifact:
    """Validated model and its compact publication identity."""

    model: ResponseOrganizationModel
    run_id: str
    status: str
    capability_status: str
    model_sha256: str
    heldout_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseOrganizationArtifactError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise ResponseOrganizationArtifactError(f"{label} must be an object")
    return value


def _sha_bytes(value: str, *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ResponseOrganizationArtifactError(f"{label} is not canonical SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise ResponseOrganizationArtifactError(f"{label} is not SHA-256") from error
    if len(raw) != 32:
        raise ResponseOrganizationArtifactError(f"{label} is not canonical SHA-256")
    return tuple(raw)


def _course_rows(
        course_path: Path,
        ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    rows: list[tuple[str, str]] = []
    sample_ids: set[str] = set()
    source_shas: set[str] = set()
    try:
        with course_path.open("r", encoding="utf-8", newline="") as stream:
            for ordinal, line in enumerate(stream, 1):
                if not line.endswith("\n") or not line.strip():
                    raise ResponseOrganizationArtifactError(
                        f"course line {ordinal} is empty or unterminated")
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or value.get("format") not in _COURSE_FORMATS
                        or value.get("license_id") != "Apache-2.0"
                        or value.get("human_generated") != 1):
                    raise ResponseOrganizationArtifactError(
                        f"course line {ordinal} identity drifted")
                split = value.get("split")
                surface = value.get("response_surface")
                sample_id = value.get("sample_id")
                source_sha = value.get("source_sha256")
                if (split not in {"train", "heldout"}
                        or not isinstance(surface, str) or not surface.strip()
                        or not isinstance(sample_id, str) or not sample_id
                        or not isinstance(source_sha, str)):
                    raise ResponseOrganizationArtifactError(
                        f"course line {ordinal} response binding is invalid")
                if sample_id in sample_ids:
                    raise ResponseOrganizationArtifactError(
                        "course contains duplicate sample_id")
                sample_ids.add(sample_id)
                source_shas.add(source_sha)
                rows.append((split, surface))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseOrganizationArtifactError("course cannot be streamed") from error
    if not rows or len(source_shas) != 1:
        raise ResponseOrganizationArtifactError(
            "course is empty or mixes source snapshots")
    return tuple(rows), tuple(sorted(source_shas))


def _training_binding(
        run_root: Path, *, expected_course_sha256: str,
        expected_pack_sha256: str,
        ) -> tuple[str, str, str]:
    summary_path = run_root / "training_summary.json"
    cursor_path = run_root / "training_cursor.int"
    pack_path = run_root / "dialogue_pack_manifest.json"
    if not all(path.is_file() for path in (summary_path, cursor_path, pack_path)):
        raise ResponseOrganizationArtifactError(
            "training run is not a completed checkpoint")
    summary = _read_json_object(summary_path, label="training summary")
    pack = _read_json_object(pack_path, label="dialogue pack manifest")
    stages = summary.get("stages_completed")
    if (summary.get("pack_sha256") != expected_pack_sha256
            or not isinstance(stages, list)
            or not {1, 2}.issubset(set(stages))):
        raise ResponseOrganizationArtifactError(
            "training summary does not bind completed stages 1 and 2")
    if pack.get("pack_sha256") != expected_pack_sha256:
        raise ResponseOrganizationArtifactError("training pack SHA drifted")
    sources = pack.get("source_files")
    if (not isinstance(sources, list)
            or sum(1 for row in sources if isinstance(row, list)
                   and len(row) >= 2 and row[1] == expected_course_sha256) != 1):
        raise ResponseOrganizationArtifactError(
            "training pack does not uniquely commit the response course")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ResponseOrganizationArtifactError("training run_id is invalid")
    return run_id, _sha256_file(cursor_path), _sha256_file(summary_path)


def _heldout_report(
        rows: tuple[tuple[str, str], ...], model: ResponseOrganizationModel,
        ) -> dict[str, object]:
    train_profiles = tuple(profile_response_surface(surface)
                           for split, surface in rows if split == "train")
    heldout_profiles = tuple(profile_response_surface(surface)
                             for split, surface in rows if split == "heldout")
    train_features = dict(response_feature_counts(train_profiles))
    heldout_features = dict(response_feature_counts(heldout_profiles))
    train_sequences = {item.signature for item in train_profiles}
    heldout_kinds = {kind for item in heldout_profiles for kind in item.signature}
    supported = set(model.supported_kinds)
    exact = sum(item.signature in train_sequences for item in heldout_profiles)
    required_status = {
        name: ("PASS" if train_features[name] > 0 else "NE")
        for name in _REQUIRED_FEATURES
    }
    capability_status = (
        "PASS" if all(value == "PASS" for value in required_status.values())
        else "NE")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "capability_status": capability_status,
        "exact_sequence_seen_count": exact,
        "heldout_count": len(heldout_profiles),
        "heldout_feature_counts": heldout_features,
        "heldout_observed_kind_coverage": (
            "PASS" if heldout_kinds.issubset(supported) else "NE"),
        "required_feature_status": required_status,
        "schema_version": 1,
        "status": "PASS" if heldout_kinds.issubset(supported) else "NE",
        "train_count": len(train_profiles),
        "train_feature_counts": train_features,
    }


def build_response_organization_artifact(
        *, course_path: str | Path, training_run_root: str | Path,
        artifact_root: str | Path, expected_course_sha256: str,
        expected_pack_sha256: str, require_k_drive: bool = True,
        ) -> ResponseOrganizationArtifact:
    """Build a new non-overwritable successor artifact root."""
    course = Path(course_path).resolve()
    training = Path(training_run_root).resolve()
    if (require_k_drive and (course.drive.upper() != "K:"
                            or training.drive.upper() != "K:")):
        raise ResponseOrganizationArtifactError(
            "course and training run must be on K drive")
    if not course.is_file() or not training.is_dir():
        raise ResponseOrganizationArtifactError(
            "course or training run does not exist")
    actual_course_sha = _sha256_file(course)
    if actual_course_sha != expected_course_sha256:
        raise ResponseOrganizationArtifactError("course SHA drifted")
    _sha_bytes(expected_pack_sha256, label="expected pack SHA")
    rows, source_shas = _course_rows(course)
    run_id, cursor_sha, summary_sha = _training_binding(
        training,
        expected_course_sha256=expected_course_sha256,
        expected_pack_sha256=expected_pack_sha256,
    )
    model = learn_response_organization_model(
        rows,
        course_sha256=_sha_bytes(actual_course_sha, label="course SHA"),
        pack_sha256=_sha_bytes(expected_pack_sha256, label="pack SHA"),
        cursor_sha256=_sha_bytes(cursor_sha, label="cursor SHA"),
        summary_sha256=_sha_bytes(summary_sha, label="summary SHA"),
    )
    heldout = _heldout_report(rows, model)
    model_payload = encode_integer_tuple(model.integer_stream())
    heldout_payload = _canonical_json_line(heldout)
    model_sha = hashlib.sha256(model_payload).hexdigest()
    heldout_sha = hashlib.sha256(heldout_payload).hexdigest()
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "capability_status": heldout["capability_status"],
        "course_sha256": actual_course_sha,
        "cursor_sha256": cursor_sha,
        "files": [
            {"bytes": len(model_payload), "name": MODEL_FILE,
             "sha256": model_sha},
            {"bytes": len(heldout_payload), "name": HELDOUT_FILE,
             "sha256": heldout_sha},
        ],
        "license_id": "Apache-2.0",
        "pack_sha256": expected_pack_sha256,
        "run_id": run_id,
        "schema_version": 1,
        "source_snapshot_sha256": list(source_shas),
        "status": heldout["status"],
        "summary_sha256": summary_sha,
    }
    root = create_new_run_root(
        artifact_root, require_k_drive=require_k_drive,
        label="response organization artifact root")
    write_exclusive_bytes(root, MODEL_FILE, model_payload,
                          label="response organization model")
    write_exclusive_bytes(root, HELDOUT_FILE, heldout_payload,
                          label="response organization heldout")
    write_exclusive_bytes(root, MANIFEST_FILE, _canonical_json_line(manifest),
                          label="response organization manifest")
    return load_response_organization_artifact(
        artifact_root, expected_run_id=run_id,
        expected_pack_sha256=expected_pack_sha256,
        require_k_drive=require_k_drive,
    )


def load_response_organization_artifact(
        artifact_root: str | Path, *, expected_run_id: str | None = None,
        expected_pack_sha256: str | None = None,
        require_k_drive: bool = True,
        ) -> ResponseOrganizationArtifact:
    """Load only a closed manifest and verify both payload files."""
    root = open_existing_run_root(
        artifact_root, require_k_drive=require_k_drive,
        label="response organization artifact root")
    with open_plain_binary(root, MANIFEST_FILE,
                           label="response organization manifest") as stream:
        manifest_payload = stream.read()
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseOrganizationArtifactError("artifact manifest is invalid") from error
    if (not isinstance(manifest, dict)
            or _canonical_json_line(manifest) != manifest_payload
            or manifest.get("artifact_kind") != ARTIFACT_KIND
            or manifest.get("schema_version") != 1):
        raise ResponseOrganizationArtifactError("artifact manifest drifted")
    run_id = manifest.get("run_id")
    pack_sha = manifest.get("pack_sha256")
    if (not isinstance(run_id, str) or not isinstance(pack_sha, str)
            or (expected_run_id is not None and run_id != expected_run_id)
            or (expected_pack_sha256 is not None
                and pack_sha != expected_pack_sha256)):
        raise ResponseOrganizationArtifactError("artifact training binding drifted")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ResponseOrganizationArtifactError("artifact file inventory drifted")
    inventory = {row.get("name"): row for row in files
                 if isinstance(row, dict)}
    if set(inventory) != {MODEL_FILE, HELDOUT_FILE}:
        raise ResponseOrganizationArtifactError("artifact file names drifted")
    payloads: dict[str, bytes] = {}
    for name in (MODEL_FILE, HELDOUT_FILE):
        row = inventory[name]
        with open_plain_binary(root, name,
                               label=f"response organization {name}") as stream:
            payload = stream.read()
        if (row.get("bytes") != len(payload)
                or row.get("sha256") != hashlib.sha256(payload).hexdigest()):
            raise ResponseOrganizationArtifactError(
                f"artifact payload {name} drifted")
        payloads[name] = payload
    try:
        model = ResponseOrganizationModel.from_integer_stream(
            decode_integer_tuple(payloads[MODEL_FILE]))
    except (ResponseOrganizationError, TypeError, ValueError) as error:
        raise ResponseOrganizationArtifactError("model payload is invalid") from error
    if (bytes(model.pack_sha256).hex() != pack_sha
            or bytes(model.course_sha256).hex() != manifest.get("course_sha256")
            or bytes(model.cursor_sha256).hex() != manifest.get("cursor_sha256")
            or bytes(model.summary_sha256).hex() != manifest.get("summary_sha256")):
        raise ResponseOrganizationArtifactError("model commitment drifted")
    try:
        heldout = json.loads(payloads[HELDOUT_FILE].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseOrganizationArtifactError("heldout payload is invalid") from error
    if (_canonical_json_line(heldout) != payloads[HELDOUT_FILE]
            or not isinstance(heldout, dict)
            or heldout.get("status") != manifest.get("status")
            or heldout.get("capability_status")
            != manifest.get("capability_status")):
        raise ResponseOrganizationArtifactError("heldout commitment drifted")
    return ResponseOrganizationArtifact(
        model, run_id, str(manifest["status"]),
        str(manifest["capability_status"]),
        str(inventory[MODEL_FILE]["sha256"]),
        str(inventory[HELDOUT_FILE]["sha256"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build OASST1 response organization artifact")
    parser.add_argument("--course", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-course-sha256", required=True)
    parser.add_argument("--expected-pack-sha256", required=True)
    args = parser.parse_args(argv)
    value = build_response_organization_artifact(
        course_path=args.course,
        training_run_root=args.training_run,
        artifact_root=args.artifact_root,
        expected_course_sha256=args.expected_course_sha256,
        expected_pack_sha256=args.expected_pack_sha256,
    )
    print(json.dumps({
        "capability_status": value.capability_status,
        "heldout_sha256": value.heldout_sha256,
        "model_sha256": value.model_sha256,
        "run_id": value.run_id,
        "status": value.status,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ResponseOrganizationArtifact", "ResponseOrganizationArtifactError",
    "build_response_organization_artifact",
    "load_response_organization_artifact", "main",
]
