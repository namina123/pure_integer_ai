"""Deterministic Git-external artifact for the public W-02 V4 overlay."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_language_overlay import (
    W02MorphologySuccessorV4Index,
    build_w02_morphology_successor_v4_from_counts,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_public_probe import (
    W02MorphologySuccessorV4PublicTraining,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


W02_MORPH_V4_ARTIFACT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-ARTIFACT-V1")
W02_MORPH_V4_ARTIFACT_STORE = "morphology-v4-language-overlay-store"
W02_MORPH_V4_ARTIFACT_MANIFEST = "morphology-v4-language-overlay.artifact.json"
W02_MORPH_V4_ARTIFACT_ROWS = "morphology-v4-language-overlay.rows.jsonl.gz"


# object-model: exception
class W02MorphologySuccessorV4ArtifactError(RuntimeError):
    """V4 artifact layout, content, semantic identity, or run guard drifted."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_rows(path: Path, rows: tuple[dict[str, object], ...]) -> tuple[int, str, int, str]:
    content_digest = hashlib.sha256()
    content_size = 0
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for row in rows:
                line = canonical_json_bytes(row) + b"\n"
                stream.write(line)
                content_digest.update(line)
                content_size += len(line)
    transport_size, transport_sha = _sha256_file(path)
    return content_size, content_digest.hexdigest(), transport_size, transport_sha


def _read_rows(path: Path) -> Iterator[dict[str, object]]:
    try:
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        raise W02MorphologySuccessorV4ArtifactError(
                            "V4 artifact row newline drifted")
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    yield value
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02MorphologySuccessorV4ArtifactError):
            raise
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact gzip/JSONL is damaged") from error


def _index_from_rows(rows: tuple[dict[str, object], ...]) -> W02MorphologySuccessorV4Index:
    exact_rows = []
    for row in rows:
        kind = row.get("row_kind")
        if kind in {"LANGUAGE_ROUTE", "LANGUAGE_BACKOFF"}:
            continue
        if kind != "EXACT_LEXEME":
            raise W02MorphologySuccessorV4ArtifactError(
                "V4 artifact row kind is not registered")
        try:
            exact_rows.append((
                str(row["language"]), str(row["form"]), str(row["lemma"]),
                str(row["upos"]), str(row["feats_json"]), int(row["count"]),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise W02MorphologySuccessorV4ArtifactError(
                "V4 artifact exact row is invalid") from error
    index = build_w02_morphology_successor_v4_from_counts(tuple(exact_rows))
    if index.semantic_rows() != rows:
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact derived backoff identity drifted")
    return index


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4ArtifactResult:
    """Verified immutable V4 artifact and its reconstructed index."""

    artifact_path: Path
    manifest_sha256: str
    rows_transport_sha256: str
    semantic_sha256: str
    row_count: int
    index: W02MorphologySuccessorV4Index

    def __post_init__(self) -> None:
        if (not isinstance(self.artifact_path, Path)
                or not isinstance(self.index, W02MorphologySuccessorV4Index)):
            raise W02MorphologySuccessorV4ArtifactError(
                "V4 artifact result type drifted")
        if any(not isinstance(value, str) or len(value) != 64 for value in (
                self.manifest_sha256, self.rows_transport_sha256,
                self.semantic_sha256)):
            raise W02MorphologySuccessorV4ArtifactError(
                "V4 artifact result SHA is invalid")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise W02MorphologySuccessorV4ArtifactError(
                "V4 artifact result row count is invalid")


def publish_w02_morphology_successor_v4_artifact(
        training: W02MorphologySuccessorV4PublicTraining,
        artifact_root: str | Path,
        *,
        run_id: int = 1,
        ) -> W02MorphologySuccessorV4ArtifactResult:
    """Publish one fresh deterministic artifact from a verified public index."""
    if not isinstance(training, W02MorphologySuccessorV4PublicTraining):
        raise TypeError("V4 artifact training type drifted")
    if type(run_id) is not int or run_id <= 0:
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact run id is invalid")
    root = Path(artifact_root).resolve()
    store = root / W02_MORPH_V4_ARTIFACT_STORE
    staging = store / f".run-{run_id:06d}.staging"
    final = store / f"run-{run_id:06d}"
    if staging.exists() or final.exists():
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact run root is not fresh")
    staging.mkdir(parents=True)
    rows = training.index.semantic_rows()
    rows_path = staging / W02_MORPH_V4_ARTIFACT_ROWS
    content_size, content_sha, transport_size, transport_sha = _write_rows(
        rows_path, rows)
    manifest = {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V4_ARTIFACT",
        "artifact_version": W02_MORPH_V4_ARTIFACT_VERSION,
        "candidate_v1_v2_v3_writes": 0,
        "formal_private_evaluation_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "languages": list(training.index.languages),
        "row_file": {
            "content_sha256": content_sha,
            "content_size_bytes": content_size,
            "relative_path": W02_MORPH_V4_ARTIFACT_ROWS,
            "row_count": len(rows),
            "transport_sha256": transport_sha,
            "transport_size_bytes": transport_size,
        },
        "run_id": run_id,
        "semantic_sha256": training.index.semantic_sha256,
        "source_files": list(training.source_files),
        "status": "W02_MORPHOLOGY_SUCCESSOR_V4_ARTIFACT_FROZEN",
        "training_counts": {
            "backoff_lexeme_row_count":
                training.index.backoff_lexeme_row_count,
            "exact_lexeme_row_count": training.index.exact_lexeme_row_count,
            "logic_operations": training.index.logic_operations,
            "sentence_count": training.sentence_count,
            "token_count": training.token_count,
            "unique_form_count": training.unique_form_count,
            "unique_tuple_count": training.unique_tuple_count,
        },
        "tree_commitment": _hash_value({
            "content_sha256": content_sha,
            "content_size_bytes": content_size,
            "row_count": len(rows),
            "transport_sha256": transport_sha,
            "transport_size_bytes": transport_size,
        }),
    }
    write_immutable_json(manifest, staging / W02_MORPH_V4_ARTIFACT_MANIFEST)
    staging.rename(final)
    return read_w02_morphology_successor_v4_artifact(final)


def read_w02_morphology_successor_v4_artifact(
        artifact_path: str | Path,
        ) -> W02MorphologySuccessorV4ArtifactResult:
    """Verify transport/content/semantic identities and reconstruct the index."""
    root = Path(artifact_path).resolve()
    manifest_path = root / W02_MORPH_V4_ARTIFACT_MANIFEST
    rows_path = root / W02_MORPH_V4_ARTIFACT_ROWS
    if not manifest_path.is_file() or not rows_path.is_file():
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact layout is incomplete")
    manifest = read_canonical_object(manifest_path)
    expected_fields = {
        "artifact_kind", "artifact_version", "candidate_v1_v2_v3_writes",
        "formal_private_evaluation_runs", "language_capability_mastered",
        "language_readiness", "languages", "row_file", "run_id",
        "semantic_sha256", "source_files", "status", "training_counts",
        "tree_commitment",
    }
    if set(manifest) != expected_fields:
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact manifest fields drifted")
    if (manifest.get("artifact_version") != W02_MORPH_V4_ARTIFACT_VERSION
            or manifest.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V4_ARTIFACT_FROZEN"
            or manifest.get("candidate_v1_v2_v3_writes") != 0
            or manifest.get("formal_private_evaluation_runs") != 0):
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact manifest state drifted")
    row_file = manifest.get("row_file")
    if not isinstance(row_file, dict) or set(row_file) != {
            "content_sha256", "content_size_bytes", "relative_path",
            "row_count", "transport_sha256", "transport_size_bytes"}:
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact row identity is invalid")
    if row_file.get("relative_path") != W02_MORPH_V4_ARTIFACT_ROWS:
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact row path drifted")
    transport_size, transport_sha = _sha256_file(rows_path)
    if (transport_size != row_file.get("transport_size_bytes")
            or transport_sha != row_file.get("transport_sha256")):
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact transport identity drifted")
    rows = tuple(_read_rows(rows_path))
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if (len(rows) != row_file.get("row_count")
            or len(content) != row_file.get("content_size_bytes")
            or hashlib.sha256(content).hexdigest()
            != row_file.get("content_sha256")):
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact content identity drifted")
    tree = _hash_value({
        "content_sha256": row_file["content_sha256"],
        "content_size_bytes": row_file["content_size_bytes"],
        "row_count": row_file["row_count"],
        "transport_sha256": row_file["transport_sha256"],
        "transport_size_bytes": row_file["transport_size_bytes"],
    })
    if tree != manifest.get("tree_commitment"):
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact tree commitment drifted")
    index = _index_from_rows(rows)
    if (index.semantic_sha256 != manifest.get("semantic_sha256")
            or list(index.languages) != manifest.get("languages")):
        raise W02MorphologySuccessorV4ArtifactError(
            "V4 artifact semantic identity drifted")
    _, manifest_sha = _sha256_file(manifest_path)
    return W02MorphologySuccessorV4ArtifactResult(
        root, manifest_sha, transport_sha, index.semantic_sha256,
        len(rows), index)


__all__ = [
    "W02_MORPH_V4_ARTIFACT_MANIFEST",
    "W02_MORPH_V4_ARTIFACT_ROWS",
    "W02_MORPH_V4_ARTIFACT_STORE",
    "W02_MORPH_V4_ARTIFACT_VERSION",
    "W02MorphologySuccessorV4ArtifactError",
    "W02MorphologySuccessorV4ArtifactResult",
    "publish_w02_morphology_successor_v4_artifact",
    "read_w02_morphology_successor_v4_artifact",
]
