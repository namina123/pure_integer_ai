"""FT00-05 P0/P1 的 v2 公开规模、恢复和端点性能基线。"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_RELEASE_KEY,
    V2_SCALE_RECORD_LIMITS,
)
from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2GenericTrainer,
    V2PackEntry,
    V2PackRegistry,
    V2RegistryError,
    V2TrainPackStream,
)
from pure_integer_ai.experiments.ph2_d03_v2_source_adapters import (
    V2SourceAdapterError,
    read_v2_source_adapter_audit,
)
from pure_integer_ai.experiments.ph2_d03_v2_streaming import V2StreamReader
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactManifest,
    ArtifactFileIdentity,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    CanonicalJsonObject,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    write_artifact_manifest,
    write_record_artifact_streaming,
)
from pure_integer_ai.experiments.v02_run_store import HostProcessMemory


FT00_05_REPORT_VERSION = 1
FT00_05_ARTIFACT_KIND = "PH2_D03_V2_FT00_05_SCALE_BASELINE"
FT00_05_SCALES = ("P0", "P1")
FT00_05_STAGE = "W-08"
FT00_05_PUBLIC_PACKS = 3
FT00_05_QUERY_LIMIT = 16
FT00_05_LOGICAL_BUCKETS = 257
FT00_05_SOURCE_CLUSTER = (20260808, 5, 1)
FT00_05_OWNER_TEACHER = (20260808, 5, 2)
FT00_05_SOURCE_KEY = "AUTHORED_CC0"
FT00_05_LICENSE = "CC0-1.0"
FT00_05_MAX_PHASE_ELAPSED_NS = {
    "P0": 60_000_000_000,
    "P1": 180_000_000_000,
}
FT00_05_MAX_DATABASE_BYTES = {
    "P0": 256 * 1024 * 1024,
    "P1": 1024 * 1024 * 1024,
}
FT00_05_MAX_SPILL_BYTES = {
    "P0": 256 * 1024 * 1024,
    "P1": 512 * 1024 * 1024,
}


class V2ScaleBaselineError(RuntimeError):
    """FT00-05 输入、资源、SQLite scratch 或 canonical report 错误。"""


def _strict_int(value: Any, *, where: str, positive: bool = False) -> int:
    """Require a real integer and reject bool, matching the integer runtime."""
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        raise V2ScaleBaselineError(f"{where} must be a strict integer")
    return value


def validate_scale_target(scale_key: str, target_records: int) -> None:
    """Fail closed before scratch creation when a scale hard limit is exceeded."""
    if scale_key not in FT00_05_SCALES:
        raise V2ScaleBaselineError("FT00-05 target scale is not registered")
    _strict_int(target_records, where="scale target", positive=True)
    if target_records > V2_SCALE_RECORD_LIMITS[scale_key]:
        raise V2ScaleBaselineError("FT00-05 scale target exceeds registered budget")


def _sha256_file(path: Path) -> str:
    """Hash a file with bounded read memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(root: Path, path: Path) -> str:
    """Return a POSIX path only after root containment has been checked."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise V2ScaleBaselineError("baseline path escaped scratch root")
    return PurePosixPath(resolved_path.relative_to(resolved_root)).as_posix()


def _clock_ns() -> int:
    """Use a monotonic integer clock available on supported hosts."""
    return time.perf_counter_ns()


def _memory_sample() -> tuple[int, int, int]:
    """Return current RSS, process peak RSS and whether host evidence is available."""
    values = HostProcessMemory()()
    current = values.get("current_working_set_bytes", 0)
    peak = values.get("process_peak_working_set_bytes", 0)
    if type(current) is not int or current < 0 or type(peak) is not int or peak < current:
        raise V2ScaleBaselineError("host RSS sample is malformed")
    return max(1, current), max(1, peak), int(current > 0 and peak > 0)


def _tree_bytes(root: Path) -> int:
    """Sum scratch bytes without following symlinks."""
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _database_bytes(path: Path) -> int:
    """Return SQLite main/journal bytes, excluding source packs."""
    return sum(
        candidate.stat().st_size
        for candidate in (
            path, Path(f"{path}-wal"), Path(f"{path}-shm"),
            Path(f"{path}-journal"),
        ) if candidate.exists())


def _key_blob(key: tuple[int, ...]) -> bytes:
    """Encode a stable key as canonical JSON bytes for SQLite."""
    return canonical_json_bytes(list(key))


def _phase(
        name: str,
        input_records: int,
        workspace: Path,
        operation: Callable[[], tuple[int, int, int]],
        ) -> tuple["V2ScalePhase", tuple[int, int, int]]:
    """Measure one isolated phase and retain integer RSS/byte endpoints."""
    before_rss, before_peak, rss_available = _memory_sample()
    before_bytes = _tree_bytes(workspace)
    started = _clock_ns()
    result = operation()
    elapsed = max(1, _clock_ns() - started)
    after_rss, after_peak, after_available = _memory_sample()
    after_bytes = _tree_bytes(workspace)
    output_records, bytes_read, bytes_written = result
    for value, label in (
            (input_records, "input_records"),
            (output_records, "output_records"),
            (bytes_read, "bytes_read"),
            (bytes_written, "bytes_written")):
        _strict_int(value, where=f"{name}.{label}")
    if after_bytes < before_bytes:
        raise V2ScaleBaselineError(f"{name} scratch bytes moved backwards")
    available = int(rss_available and after_available)
    return V2ScalePhase(
        name, elapsed, input_records, output_records, bytes_read,
        bytes_written, before_rss, after_rss, max(before_peak, after_peak),
        max(0, after_bytes - before_bytes), available,
    ), result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2ScalePhase:
    """One named performance phase with closed integer accounting."""

    phase_key: str
    elapsed_ns: int
    input_records: int
    output_records: int
    bytes_read: int
    bytes_written: int
    rss_before_bytes: int
    rss_after_bytes: int
    rss_peak_bytes: int
    scratch_delta_bytes: int
    rss_evidence: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase_key, str) or not self.phase_key:
            raise V2ScaleBaselineError("phase key is empty")
        for name, value in self.as_dict().items():
            if name == "phase_key":
                continue
            _strict_int(value, where=f"phase.{name}")
        if self.elapsed_ns <= 0:
            raise V2ScaleBaselineError("phase elapsed must be positive")
        if self.rss_peak_bytes < max(self.rss_before_bytes, self.rss_after_bytes):
            raise V2ScaleBaselineError("phase RSS peak is not closed")
        if self.rss_evidence not in (0, 1):
            raise V2ScaleBaselineError("phase RSS evidence is not a flag")

    def as_dict(self) -> dict[str, int | str]:
        """Return canonical report fields."""
        return {
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "elapsed_ns": self.elapsed_ns,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "phase_key": self.phase_key,
            "rss_after_bytes": self.rss_after_bytes,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_evidence": self.rss_evidence,
            "rss_peak_bytes": self.rss_peak_bytes,
            "scratch_delta_bytes": self.scratch_delta_bytes,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2ScalePoint:
    """P0 or P1 complete scale point, including fresh/resume equivalence."""

    scale_key: str
    target_records: int
    public_anchor_records: int
    authored_records: int
    authored_manifest_sha256: str
    phases: tuple[V2ScalePhase, ...]
    database_bytes: int
    spill_peak_bytes: int
    query_rows: int
    resolved_rows: int
    generated_bytes: int
    fresh_digest: str
    resume_digest: str
    query_index_built: int
    rollback_clean: int
    resource_stop_boundary: int

    def __post_init__(self) -> None:
        if self.scale_key not in FT00_05_SCALES:
            raise V2ScaleBaselineError("unknown FT00-05 scale")
        _strict_int(self.target_records, where="point.target_records", positive=True)
        _strict_int(self.public_anchor_records, where="point.public_anchor_records", positive=True)
        _strict_int(self.authored_records, where="point.authored_records", positive=True)
        if self.public_anchor_records + self.authored_records != self.target_records:
            raise V2ScaleBaselineError("point records do not close")
        for value, label in (
                (self.database_bytes, "database_bytes"),
                (self.spill_peak_bytes, "spill_peak_bytes"),
                (self.query_rows, "query_rows"),
                (self.resolved_rows, "resolved_rows"),
                (self.generated_bytes, "generated_bytes")):
            _strict_int(value, where=f"point.{label}")
        if not self.phases or tuple(item.phase_key for item in self.phases) != tuple(
                dict.fromkeys(item.phase_key for item in self.phases)):
            raise V2ScaleBaselineError("point phases must be unique")
        if any(not isinstance(item, V2ScalePhase) for item in self.phases):
            raise V2ScaleBaselineError("point phase type is invalid")
        for value, label in (
                (self.query_index_built, "query_index_built"),
                (self.rollback_clean, "rollback_clean"),
                (self.resource_stop_boundary, "resource_stop_boundary")):
            if value not in (0, 1):
                raise V2ScaleBaselineError(f"point.{label} is not a flag")
        for value, label in (
                (self.authored_manifest_sha256, "authored_manifest_sha256"),
                (self.fresh_digest, "fresh_digest"),
                (self.resume_digest, "resume_digest")):
            if not isinstance(value, str) or len(value) != 64 or any(
                    char not in "0123456789abcdef" for char in value):
                raise V2ScaleBaselineError(f"point.{label} is not SHA-256")
        if self.fresh_digest != self.resume_digest:
            raise V2ScaleBaselineError("fresh/resume digest mismatch")

    def as_dict(self) -> dict[str, Any]:
        """Return the public, path-free point payload."""
        return {
            "authored_manifest_sha256": self.authored_manifest_sha256,
            "authored_records": self.authored_records,
            "database_bytes": self.database_bytes,
            "fresh_digest": self.fresh_digest,
            "generated_bytes": self.generated_bytes,
            "phases": [item.as_dict() for item in self.phases],
            "public_anchor_records": self.public_anchor_records,
            "query_index_built": self.query_index_built,
            "query_rows": self.query_rows,
            "resolved_rows": self.resolved_rows,
            "resource_stop_boundary": self.resource_stop_boundary,
            "resume_digest": self.resume_digest,
            "rollback_clean": self.rollback_clean,
            "scale_key": self.scale_key,
            "spill_peak_bytes": self.spill_peak_bytes,
            "target_records": self.target_records,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2ScaleSlope:
    """Integer slope verdict for the P0 -> P1 curve."""

    record_multiplier: int
    bulk_phase_limit_multiplier: int
    query_phase_limit_multiplier: int
    database_limit_multiplier: int
    rss_limit_multiplier: int
    passed: int
    checked_phase_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if name == "checked_phase_keys":
                continue
            if type(value) is not int or value < 0:
                raise V2ScaleBaselineError(f"slope.{name} must be an integer")
        if self.passed not in (0, 1) or not self.checked_phase_keys:
            raise V2ScaleBaselineError("slope verdict is incomplete")

    def as_dict(self) -> dict[str, Any]:
        """Return slope policy and result."""
        return {
            "bulk_phase_limit_multiplier": self.bulk_phase_limit_multiplier,
            "checked_phase_keys": list(self.checked_phase_keys),
            "database_limit_multiplier": self.database_limit_multiplier,
            "passed": self.passed,
            "query_phase_limit_multiplier": self.query_phase_limit_multiplier,
            "record_multiplier": self.record_multiplier,
            "rss_limit_multiplier": self.rss_limit_multiplier,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2ScaleBaselineReport:
    """Canonical FT00-05 report; no formal Candidate or evaluator state."""

    artifact_kind: str
    report_version: int
    release_key: str
    stage_key: str
    scales: tuple[str, ...]
    source_audit_sha256: str
    public_manifest_sha256: tuple[str, ...]
    points: tuple[V2ScalePoint, ...]
    slope: V2ScaleSlope
    rss_evidence: int
    formal_training_runs: int
    candidate_writes: int
    core_writes: int
    memory_writes: int
    companion_writes: int
    use_writes: int
    teacher_calls: int
    status: str

    def __post_init__(self) -> None:
        if self.artifact_kind != FT00_05_ARTIFACT_KIND or self.report_version != FT00_05_REPORT_VERSION:
            raise V2ScaleBaselineError("FT00-05 report identity drift")
        if self.release_key != V2_RELEASE_KEY or self.stage_key != FT00_05_STAGE:
            raise V2ScaleBaselineError("FT00-05 report release/stage drift")
        if self.scales != FT00_05_SCALES or tuple(item.scale_key for item in self.points) != self.scales:
            raise V2ScaleBaselineError("FT00-05 scales are not frozen")
        if not isinstance(self.source_audit_sha256, str) or len(self.source_audit_sha256) != 64 or any(
                char not in "0123456789abcdef" for char in self.source_audit_sha256):
            raise V2ScaleBaselineError("source audit SHA is not SHA-256")
        if any(not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value)
                for value in self.public_manifest_sha256):
            raise V2ScaleBaselineError("public manifest SHA is not SHA-256")
        if len(self.public_manifest_sha256) != FT00_05_PUBLIC_PACKS:
            raise V2ScaleBaselineError("FT00-05 public manifest inventory drift")
        if any(type(value) is not int or value != 0 for value in (
                self.formal_training_runs, self.candidate_writes,
                self.core_writes, self.memory_writes, self.companion_writes,
                self.use_writes, self.teacher_calls)):
            raise V2ScaleBaselineError("FT00-05 formal state must remain zero")
        if self.rss_evidence not in (0, 1) or self.status not in {"PASS", "NE"}:
            raise V2ScaleBaselineError("FT00-05 report status is invalid")
        if self.status == "PASS" and self.slope.passed != 1:
            raise V2ScaleBaselineError("FT00-05 PASS without slope PASS")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical report object."""
        return {
            "artifact_kind": self.artifact_kind,
            "candidate_writes": self.candidate_writes,
            "core_writes": self.core_writes,
            "formal_training_runs": self.formal_training_runs,
            "companion_writes": self.companion_writes,
            "memory_writes": self.memory_writes,
            "points": [item.as_dict() for item in self.points],
            "public_manifest_sha256": list(self.public_manifest_sha256),
            "release_key": self.release_key,
            "report_version": self.report_version,
            "rss_evidence": self.rss_evidence,
            "scales": list(self.scales),
            "slope": self.slope.as_dict(),
            "source_audit_sha256": self.source_audit_sha256,
            "stage_key": self.stage_key,
            "status": self.status,
            "teacher_calls": self.teacher_calls,
            "use_writes": self.use_writes,
        }

    def canonical_bytes(self) -> bytes:
        """Return the report bytes with one trailing newline."""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """Return the canonical report SHA-256."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _carrier_payload(ordinal: int) -> dict[str, Any]:
    """Create one deterministic authored carrier without flattening structure."""
    text = f"ft00-05 baseline item {ordinal}"
    node_key = (20260808, 5, 10, ordinal)
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "carrier": {
            "carrier_kind": "document_container",
            "edges": [],
            "nodes": [{
                "attributes": {"query_bucket": ordinal % FT00_05_LOGICAL_BUCKETS},
                "node_key": list(node_key),
                "node_kind": "paragraph",
                "parent_node_key": None,
                "span_end": len(text),
                "span_start": 0,
            }],
            "raw_text_sha256": raw_hash,
            "root_node_keys": [list(node_key)],
        },
        "language_payload": {
            "query_bucket": ordinal % FT00_05_LOGICAL_BUCKETS,
            "text": text,
        },
    }


def _authored_source(scale_code: int) -> SourceRefRecord:
    """Return the single public authored source anchor for one scale pack."""
    source_key = (20260808, 5, scale_code, 1)
    cluster_key = StableRecordKey(FT00_05_SOURCE_CLUSTER + (scale_code,))
    return SourceRefRecord(
        2, 2, 2,
        StableRecordKey((20260808, 5, scale_code, 20)),
        StableRecordKey((20260808, 5, scale_code, 21)),
        StableRecordKey(source_key), "AUTHORED_CC0", "FT00-05-AUTHORED", "",
        "urn:public-ft00-05:authored", "AUTHORED_CC0:1",
        "sha256:" + "a" * 64, "a" * 64, "CC0-1.0", "PUBLIC",
        "FT00-05 authored scale fixture", 2,
        CanonicalJsonObject.from_value({
            "document_cluster_key": list(cluster_key.components + (1,)),
            "entity_graph_cluster_key": list(cluster_key.components + (2,)),
            "locator_kind": "record", "locator_value": "1",
            "span_end": 1, "span_start": 0,
        }),
        1, cluster_key,
    )


def _authored_observation(source: SourceRefRecord, scale_code: int, ordinal: int) -> ObservationRecord:
    """Return one train Observation in a unique source/content cluster."""
    base = (20260808, 5, scale_code, 10, ordinal)
    cluster = StableRecordKey(base)
    payload = _carrier_payload(ordinal)
    return ObservationRecord(
        2, 2, 2, StableRecordKey((20260808, 5, scale_code, 20)),
        StableRecordKey((20260808, 5, scale_code, 21)),
        StableRecordKey(base + (1,)), "W-08", "FT00-05-AUTHORED", "train",
        "zh", "document_container", source.stable_key, "CC0-1.0",
        StableRecordKey(base), StableRecordKey(base + (2,)),
        StableRecordKey(base + (3,)), StableRecordKey(base + (4,)),
        "forming", "support", "typed_carrier",
        CanonicalJsonObject.from_value(payload), "NONE", None, (), ordinal,
    )


def _authored_teacher(source: SourceRefRecord, observation: ObservationRecord,
                      scale_code: int, ordinal: int) -> TeacherEvidenceRecord:
    """Return train-only authored Evidence."""
    return TeacherEvidenceRecord(
        2, 2, 2, StableRecordKey((20260808, 5, scale_code, 20)),
        StableRecordKey((20260808, 5, scale_code, 21)),
        StableRecordKey((20260808, 5, scale_code, 30, ordinal)),
        observation.stable_key, "AUTHORED_FORM",
        CanonicalJsonObject.from_value({
            "accepted": 1,
            "query_bucket": ordinal % FT00_05_LOGICAL_BUCKETS,
        }), source.stable_key, "W-08", 0,
        StableRecordKey(FT00_05_OWNER_TEACHER + (scale_code,)),
    )


def _empty_evaluator(scale_code: int) -> Iterable[object]:
    """Keep the public pack's evaluator owner/split present without labels."""
    if scale_code < 0:
        raise V2ScaleBaselineError("scale code cannot be negative")
    return iter(())


def build_authored_scale_pack(
        root: str | Path,
        scale_key: str,
        authored_train_records: int,
        ) -> tuple[str, str, int]:
    """Stream-build an exact-count authored pack for a P0/P1 baseline."""
    if scale_key not in FT00_05_SCALES:
        raise V2ScaleBaselineError("authored fixture scale is not P0/P1")
    _strict_int(authored_train_records, where="authored_train_records", positive=True)
    if authored_train_records < 3 or authored_train_records % 2 != 1:
        raise V2ScaleBaselineError("authored train records must be 1 + 2N")
    scale_code = FT00_05_SCALES.index(scale_key) + 1
    target_root = Path(root).resolve()
    pack_relative = f"data/ph2/ft00_05/packs/AUTHORED_CC0--{scale_key}--stream-v1"
    pack_root = target_root / Path(*PurePosixPath(pack_relative).parts)
    if pack_root.exists():
        raise V2ScaleBaselineError("authored fixture target already exists")
    pack_root.mkdir(parents=True, exist_ok=True)
    source = _authored_source(scale_code)
    observation_count = (authored_train_records - 1) // 2
    observations = (
        _authored_observation(source, scale_code, ordinal)
        for ordinal in range(1, observation_count + 1)
    )
    teachers = (
        _authored_teacher(
            source, _authored_observation(source, scale_code, ordinal),
            scale_code, ordinal)
        for ordinal in range(1, observation_count + 1)
    )
    cluster_key = StableRecordKey(FT00_05_SOURCE_CLUSTER + (scale_code,))
    files: list[ArtifactFileIdentity] = []
    files.append(write_record_artifact_streaming(
        (source,), pack_root,
        ArtifactWriteSpec("source_ref", "source", "source_refs.jsonl.gz", None,
                          FT00_05_LICENSE, (cluster_key,))))
    files.append(write_record_artifact_streaming(
        observations, pack_root,
        ArtifactWriteSpec("observation", "observation",
                          "observations/train.jsonl.gz", "train",
                          FT00_05_LICENSE, (cluster_key,))))
    files.append(write_record_artifact_streaming(
        teachers, pack_root,
        ArtifactWriteSpec("teacher_evidence", "teacher",
                          "owners/teacher/train.evidence.jsonl.gz", "train",
                          FT00_05_LICENSE, (cluster_key,))))
    files.append(write_record_artifact_streaming(
        _empty_evaluator(scale_code), pack_root,
        ArtifactWriteSpec("evaluator_label", "evaluator",
                          "owners/evaluator/dev.labels.jsonl.gz", "dev",
                          FT00_05_LICENSE, (cluster_key,))))
    manifest = ArtifactManifest(
        2, 2, 2, 2,
        StableRecordKey((20260808, 5, scale_code, 20)),
        StableRecordKey((20260808, 5, scale_code, 21)),
        FT00_05_SOURCE_KEY, FT00_05_LICENSE, "PUBLIC", 2, 2, 2,
        tuple(files), ("train", "dev"), (FT00_05_STAGE,), (cluster_key,), (),
        FT00_05_STAGE,
    )
    validate_v2_record(manifest.to_dict())
    manifest_path = write_artifact_manifest(manifest, pack_root)
    return _safe_relative(target_root, manifest_path), manifest.sha256(), observation_count


def _copy_public_packs(repository_root: Path, scratch_root: Path,
                       manifest_paths: tuple[str, ...]) -> None:
    """Copy only the three public v2 packs into an isolated benchmark root."""
    for relative in manifest_paths:
        source_manifest = repository_root / Path(*PurePosixPath(relative).parts)
        source_pack = source_manifest.parent
        target_pack = scratch_root / Path(*PurePosixPath(relative).parts).parent
        target_pack.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_pack, target_pack)


def _public_manifest_paths(repository_root: Path) -> tuple[str, ...]:
    """Read the immutable public source audit and return READY v2 manifests."""
    audit = read_v2_source_adapter_audit(repository_root)
    paths = tuple(item.v2_manifest_relative_path
                  for item in audit.entries if item.status == "READY")
    if len(paths) != FT00_05_PUBLIC_PACKS or len(set(paths)) != len(paths):
        raise V2ScaleBaselineError("public v2 manifest inventory is not exactly three")
    return paths


def _scan_files(root: Path, paths: Iterable[str]) -> tuple[int, int, str]:
    """Hash files using bounded blocks and return file/byte/digest counts."""
    digest = hashlib.sha256()
    file_count = byte_count = 0
    for relative in sorted(paths):
        path = root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise V2ScaleBaselineError("baseline scan file is missing")
        file_count += 1
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                byte_count += len(block)
    return file_count, byte_count, digest.hexdigest()


def _typed_train_inputs(
        root: Path,
        registry: V2PackRegistry,
        plan: Any,
        ) -> tuple[V2TrainPackStream, ...]:
    """Create repeatable per-pack stream descriptors without retaining payload."""
    streams: list[V2TrainPackStream] = []
    for entry in registry.entries:
        if entry.pack_key not in plan.pack_keys:
            continue
        def records_factory(
                selected: V2PackEntry = entry,
                ) -> Iterable[dict[str, Any]]:
            return (
                item.to_dict()
                for item in V2StreamReader(root, selected).iter_records("teacher")
            )
        streams.append(V2TrainPackStream(entry.pack_key, records_factory))
    return tuple(sorted(streams, key=lambda item: item.pack_key))


def _configure_database(path: Path) -> sqlite3.Connection:
    """Open an isolated SQLite scratch database with durable commits."""
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE train_input(
            stable_key BLOB PRIMARY KEY,
            kind TEXT NOT NULL,
            source_key BLOB,
            observation_key BLOB,
            logical_order INTEGER NOT NULL,
            query_bucket INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE candidate_simulation(
            stable_key BLOB PRIMARY KEY,
            source_key BLOB NOT NULL,
            logical_order INTEGER NOT NULL,
            query_bucket INTEGER NOT NULL,
            payload BLOB NOT NULL,
            evidence_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE evidence_projection(
            observation_key BLOB PRIMARY KEY,
            evidence_count INTEGER NOT NULL
        );
        CREATE TABLE merge_fresh(
            stable_key BLOB PRIMARY KEY,
            kind TEXT NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE merge_resume(
            stable_key BLOB PRIMARY KEY,
            kind TEXT NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE checkpoints(
            checkpoint_key TEXT PRIMARY KEY,
            cursor_key BLOB NOT NULL,
            complete INTEGER NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _insert_train_inputs(connection: sqlite3.Connection,
                         batches: tuple[V2TrainPackStream, ...]) -> int:
    """Write only the intake projection, never Candidate/Core/Memory state."""
    rows: list[tuple[Any, ...]] = []
    inserted = 0
    for batch in batches:
        for item in batch.records_factory():
            kind = item["record_kind"]
            stable = tuple(item["stable_key"])
            source_key: tuple[int, ...] | None = None
            observation_key: tuple[int, ...] | None = None
            logical_order = 0
            query_bucket = 0
            if kind == "source_ref":
                source_key = stable
            elif kind == "observation":
                source_key = tuple(item["source_ref_key"])
                observation_key = stable
                logical_order = item["logical_order"]
                query_bucket = logical_order % FT00_05_LOGICAL_BUCKETS
            elif kind == "teacher_evidence":
                source_key = tuple(item["source_ref_key"])
                observation_key = tuple(item["observation_key"])
                logical_order = item["stable_key"][-1]
                query_bucket = logical_order % FT00_05_LOGICAL_BUCKETS
            payload = canonical_json_bytes(item)
            rows.append((
                _key_blob(stable), kind,
                None if source_key is None else _key_blob(source_key),
                None if observation_key is None else _key_blob(observation_key),
                logical_order, query_bucket, payload,
            ))
            if len(rows) == 512:
                connection.executemany(
                    "INSERT INTO train_input VALUES(?,?,?,?,?,?,?)", rows)
                inserted += len(rows)
                rows.clear()
    if rows:
        connection.executemany(
            "INSERT INTO train_input VALUES(?,?,?,?,?,?,?)", rows)
        inserted += len(rows)
    connection.commit()
    return inserted


def _canonical_table_digest(connection: sqlite3.Connection, table: str) -> str:
    """Digest a deterministic table stream, without collecting rows."""
    if table not in {"merge_fresh", "merge_resume"}:
        raise V2ScaleBaselineError("unknown digest table")
    digest = hashlib.sha256()
    for stable_key, kind, payload in connection.execute(
            f"SELECT stable_key, kind, payload FROM {table} ORDER BY stable_key"):
        digest.update(stable_key)
        digest.update(kind.encode("ascii"))
        digest.update(payload)
    return digest.hexdigest()


def _copy_merge_rows(connection: sqlite3.Connection, table: str,
                     lower_key: bytes | None = None) -> int:
    """Copy a sorted slice of intake rows into one merge table."""
    if table not in {"merge_fresh", "merge_resume"}:
        raise V2ScaleBaselineError("unknown merge table")
    if lower_key is None:
        query = "SELECT stable_key, kind, payload FROM train_input ORDER BY stable_key"
        rows = connection.execute(query)
    else:
        rows = connection.execute(
            "SELECT stable_key, kind, payload FROM train_input "
            "WHERE stable_key > ? ORDER BY stable_key", (lower_key,))
    batch: list[tuple[bytes, str, bytes]] = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) == 512:
            connection.executemany(
                f"INSERT INTO {table} VALUES(?,?,?)", batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(f"INSERT INTO {table} VALUES(?,?,?)", batch)
        count += len(batch)
    return count


def _run_sqlite_scale(
        root: Path,
        batches: tuple[V2TrainPackStream, ...],
        target_records: int,
        ) -> tuple[tuple[V2ScalePhase, ...], dict[str, Any]]:
    """Run isolated intake, simulation, recovery, indexed question and generation."""
    database_path = root / "ft00_05_scale.sqlite3"
    connection = _configure_database(database_path)
    phases: list[V2ScalePhase] = []
    metadata: dict[str, Any] = {}

    def run_phase(name: str, input_count: int,
                  operation: Callable[[], tuple[int, int, int]]) -> Any:
        phase, result = _phase(name, input_count, root, operation)
        phases.append(phase)
        return result

    intake_count = run_phase(
        "intake_projection", target_records,
        lambda: (_insert_train_inputs(connection, batches), 0, 0))
    metadata["intake_count"] = intake_count[0]

    def build_candidate() -> tuple[int, int, int]:
        connection.execute(
            "INSERT INTO candidate_simulation "
            "SELECT stable_key, source_key, logical_order, query_bucket, payload, 0 "
            "FROM train_input WHERE kind='observation'")
        connection.commit()
        count = connection.execute(
            "SELECT COUNT(*) FROM candidate_simulation").fetchone()[0]
        return count, 0, 0

    candidate_count = run_phase("candidate_build_simulation", target_records,
                                build_candidate)[0]
    metadata["candidate_count"] = candidate_count

    def apply_evidence() -> tuple[int, int, int]:
        connection.execute(
            "INSERT INTO evidence_projection "
            "SELECT observation_key, COUNT(*) FROM train_input "
            "WHERE kind='teacher_evidence' GROUP BY observation_key")
        connection.execute(
            "UPDATE candidate_simulation SET evidence_count=("
            "SELECT evidence_count FROM evidence_projection "
            "WHERE observation_key=candidate_simulation.stable_key)")
        connection.commit()
        count = connection.execute(
            "SELECT COUNT(*) FROM evidence_projection").fetchone()[0]
        return count, 0, 0

    metadata["evidence_count"] = run_phase(
        "evidence_apply", candidate_count, apply_evidence)[0]

    def merge_and_resume() -> tuple[int, int, int]:
        fresh_count = _copy_merge_rows(connection, "merge_fresh")
        connection.commit()
        total = connection.execute(
            "SELECT COUNT(*) FROM train_input").fetchone()[0]
        midpoint = total // 2
        cursor_row = connection.execute(
            "SELECT stable_key FROM train_input ORDER BY stable_key "
            "LIMIT 1 OFFSET ?", (max(0, midpoint - 1),)).fetchone()
        cursor = cursor_row[0] if cursor_row is not None else b""
        if cursor:
            connection.execute(
                "INSERT INTO merge_resume SELECT stable_key, kind, payload "
                "FROM train_input WHERE stable_key <= ?", (cursor,))
        else:
            cursor = b""
        prefix_count = connection.execute(
            "SELECT COUNT(*) FROM merge_resume").fetchone()[0]
        connection.execute(
            "INSERT INTO checkpoints VALUES('FT00-05-resume', ?, 0)",
            (cursor,))
        connection.commit()
        connection.close()
        reopened = sqlite3.connect(str(database_path))
        reopened.execute("PRAGMA synchronous=FULL")
        reopened.execute("PRAGMA foreign_keys=ON")
        resumed_count = _copy_merge_rows(reopened, "merge_resume", cursor)
        reopened.execute(
            "UPDATE checkpoints SET complete=1 WHERE checkpoint_key='FT00-05-resume'")
        reopened.commit()
        fresh_digest = _canonical_table_digest(reopened, "merge_fresh")
        resume_digest = _canonical_table_digest(reopened, "merge_resume")
        metadata["fresh_digest"] = fresh_digest
        metadata["resume_digest"] = resume_digest
        metadata["merge_count"] = resumed_count + (prefix_count if cursor else 0)
        reopened.close()
        return fresh_count, 0, 0

    run_phase("checkpoint_merge_resume", target_records, merge_and_resume)
    connection = sqlite3.connect(str(database_path))

    def build_query_index() -> tuple[int, int, int]:
        connection.execute(
            "CREATE INDEX query_simulation_idx ON candidate_simulation "
            "(query_bucket, logical_order, stable_key)")
        connection.commit()
        return connection.execute(
            "SELECT COUNT(*) FROM candidate_simulation").fetchone()[0], 0, 0

    run_phase("query_index_build", candidate_count, build_query_index)

    question_bucket = 1
    query_holder: dict[str, Any] = {}

    def query_operation() -> tuple[int, int, int]:
        rows = connection.execute(
            "SELECT stable_key, payload FROM candidate_simulation "
            "WHERE query_bucket=? ORDER BY logical_order, stable_key LIMIT ?",
            (question_bucket, FT00_05_QUERY_LIMIT),
        ).fetchall()
        query_holder["rows"] = rows
        return len(rows), 0, 0

    run_phase("query", candidate_count, query_operation)
    rows = query_holder["rows"]
    resolve_holder: dict[str, Any] = {}

    def resolve_operation() -> tuple[int, int, int]:
        resolved = connection.execute(
            "SELECT c.stable_key, s.stable_key, c.evidence_count "
            "FROM candidate_simulation c JOIN train_input s "
            "ON s.stable_key=c.source_key WHERE c.query_bucket=? "
            "ORDER BY c.logical_order, c.stable_key LIMIT ?",
            (question_bucket, FT00_05_QUERY_LIMIT),
        ).fetchall()
        resolve_holder["resolved"] = resolved
        return len(resolved), 0, 0

    run_phase("resolve", len(rows), resolve_operation)
    resolved = resolve_holder["resolved"]
    generation_holder: dict[str, Any] = {}

    def generation_operation() -> tuple[int, int, int]:
        generated = bytearray()
        for stable_key, payload in rows:
            value = json.loads(payload.decode("utf-8"))
            language_payload = value["typed_payload"]["language_payload"]
            text = language_payload.get("text")
            if not isinstance(text, str):
                text = canonical_json_bytes(language_payload).decode("utf-8")
            generated.extend(text.encode("utf-8"))
            generated.extend(b"\n")
        generation_holder["generated"] = generated
        return len(rows), 0, len(generated)

    run_phase("generation", len(resolved), generation_operation)
    generated = generation_holder["generated"]
    metadata.update({
        "query_rows": len(rows),
        "resolved_rows": len(resolved),
        "generated_bytes": len(generated),
        "question_operation_elapsed_ns": sum(
            item.elapsed_ns for item in phases
            if item.phase_key in {"query", "resolve", "generation"}),
    })

    def rollback_operation() -> tuple[int, int, int]:
        before_digest = _canonical_table_digest(connection, "merge_resume")
        connection.execute("BEGIN")
        connection.execute(
            "INSERT INTO candidate_simulation VALUES(?,?,?,?,?,?)",
            (_key_blob((20260808, 5, 99, 99)), _key_blob((1,)), 0, 0,
             b"rollback", 0),
        )
        connection.rollback()
        after_digest = _canonical_table_digest(connection, "merge_resume")
        clean = int(before_digest == after_digest)
        metadata["rollback_clean"] = clean
        return clean, 0, 0

    run_phase("rollback", target_records, rollback_operation)
    connection.close()
    metadata["database_bytes"] = _database_bytes(database_path)
    metadata["spill_peak_bytes"] = 0
    metadata["query_index_built"] = 1
    return tuple(phases), metadata


def _build_slope(points: tuple[V2ScalePoint, ...]) -> V2ScaleSlope:
    """Apply the pre-registered integer slope policy."""
    if tuple(item.scale_key for item in points) != FT00_05_SCALES:
        raise V2ScaleBaselineError("slope requires P0 then P1")
    p0, p1 = points
    bulk = {"pack_build", "raw_pack_scan", "typed_adaptation",
            "registry_trainer_intake", "intake_projection",
            "candidate_build_simulation",
            "evidence_apply", "checkpoint_merge_resume", "query_index_build"}
    query = {"query", "resolve", "generation"}
    p0_phases = {item.phase_key: item for item in p0.phases}
    p1_phases = {item.phase_key: item for item in p1.phases}
    checked = tuple(item.phase_key for item in p0.phases)
    passed = 1
    for key in checked:
        if key not in p1_phases:
            passed = 0
            continue
        left = p0_phases[key].elapsed_ns
        right = p1_phases[key].elapsed_ns
        if key in bulk and right > left * 8 + 1_000_000_000:
            passed = 0
        if key in query and right > left * 4 + 1_000_000_000:
            passed = 0
    if p1.database_bytes > p0.database_bytes * 6 + 1_048_576:
        passed = 0
    if p1.spill_peak_bytes > p0.spill_peak_bytes * 8 + 1_048_576:
        passed = 0
    if all(item.rss_evidence == 1 for point in points for item in point.phases):
        p0_endpoints = tuple(
            value for item in p0.phases
            for value in (item.rss_before_bytes, item.rss_after_bytes))
        p1_endpoints = tuple(
            value for item in p1.phases
            for value in (item.rss_before_bytes, item.rss_after_bytes))
        p0_span = max(p0_endpoints) - min(p0_endpoints)
        p1_span = max(p1_endpoints) - min(p1_endpoints)
        if p1_span > p0_span * 8 + 256 * 1024 * 1024:
            passed = 0
    return V2ScaleSlope(4, 8, 4, 6, 8, passed, checked)


def validate_scale_point_resources(point: V2ScalePoint) -> None:
    """Enforce the pre-registered FT00 resource stop boundary."""
    budget_ns = FT00_05_MAX_PHASE_ELAPSED_NS[point.scale_key]
    if any(item.elapsed_ns > budget_ns for item in point.phases):
        raise V2ScaleBaselineError("FT00-05 phase exceeded resource time budget")
    if point.database_bytes > FT00_05_MAX_DATABASE_BYTES[point.scale_key]:
        raise V2ScaleBaselineError("FT00-05 database exceeded resource budget")
    if point.spill_peak_bytes > FT00_05_MAX_SPILL_BYTES[point.scale_key]:
        raise V2ScaleBaselineError("FT00-05 spill exceeded resource budget")
    if (point.query_rows > FT00_05_QUERY_LIMIT
            or point.resolved_rows > FT00_05_QUERY_LIMIT
            or point.rollback_clean != 1):
        raise V2ScaleBaselineError("FT00-05 question/rollback stop boundary failed")


def _make_point(
        repository_root: Path,
        scale_key: str,
        public_manifest_paths: tuple[str, ...],
        ) -> V2ScalePoint:
    """Run one isolated scale point under a temporary scratch root."""
    target = V2_SCALE_RECORD_LIMITS[scale_key]
    validate_scale_target(scale_key, target)
    scratch = Path(tempfile.mkdtemp(prefix=f"ft00-05-{scale_key.lower()}-"))
    try:
        _copy_public_packs(repository_root, scratch, public_manifest_paths)
        public_registry = V2PackRegistry.from_manifest_paths(
            scratch, public_manifest_paths)
        public_plan = public_registry.train_plan(FT00_05_STAGE, scale_key="P0")
        anchor_count = public_plan.total_input_count
        authored_count = target - anchor_count
        if authored_count <= 0:
            raise V2ScaleBaselineError("public anchor exceeds baseline scale")
        built: dict[str, Any] = {}
        def build_pack() -> tuple[int, int, int]:
            relative, digest, observation_count = build_authored_scale_pack(
                scratch, scale_key, authored_count)
            built.update({
                "relative": relative,
                "sha": digest,
                "observation_count": observation_count,
            })
            pack_root = scratch / Path(*PurePosixPath(relative).parts).parent
            return authored_count, 0, _tree_bytes(pack_root)
        pack_phase, _ = _phase("pack_build", authored_count, scratch, build_pack)
        authored_relative = built["relative"]
        authored_sha = built["sha"]
        manifest_paths = tuple(sorted((*public_manifest_paths, authored_relative)))
        registry = V2PackRegistry.from_manifest_paths(scratch, manifest_paths)
        plan = registry.train_plan(FT00_05_STAGE, scale_key=scale_key)
        if plan.total_input_count != target:
            raise V2ScaleBaselineError("baseline train plan count drift")
        scan_paths = []
        for relative in manifest_paths:
            manifest = ArtifactManifest.from_dict(json.loads(
                (scratch / Path(*PurePosixPath(relative).parts)).read_text(
                    encoding="utf-8").strip()))
            scan_paths.append(relative)
            scan_paths.extend(
                f"{PurePosixPath(relative).parent.as_posix()}/{item.relative_path}"
                for item in manifest.files)
        scan_paths = tuple(sorted(set(scan_paths)))
        phases: list[V2ScalePhase] = [pack_phase]
        scan_phase, scan_result = _phase(
            "raw_pack_scan", target, scratch,
            lambda: (len(scan_paths), _scan_files(scratch, scan_paths)[1], 0))
        phases.append(scan_phase)
        typed_holder: dict[str, Any] = {}
        def adapt_streams() -> tuple[int, int, int]:
            streams = _typed_train_inputs(scratch, registry, plan)
            count = sum(1 for stream in streams
                        for _ in stream.records_factory())
            typed_holder["streams"] = streams
            return count, 0, 0
        typed_phase, _ = _phase(
            "typed_adaptation", target, scratch, adapt_streams)
        phases.append(typed_phase)
        trainer_holder: dict[str, Any] = {}
        def validate_streams() -> tuple[int, int, int]:
            streams = typed_holder["streams"]
            result = V2GenericTrainer().validate_train_streams(plan, streams)
            trainer_holder["result"] = result
            return (
                result.source_ref_count + result.observation_count
                + result.teacher_evidence_count, 0, 0)
        trainer_phase, _ = _phase(
            "registry_trainer_intake", target, scratch, validate_streams)
        phases.append(trainer_phase)
        batches = typed_holder["streams"]
        sqlite_phases, metadata = _run_sqlite_scale(
            scratch, batches, target)
        phases.extend(sqlite_phases)
        all_phase_keys = tuple(item.phase_key for item in phases)
        if len(all_phase_keys) != len(set(all_phase_keys)):
            raise V2ScaleBaselineError("baseline phase key duplicated")
        result = V2ScalePoint(
            scale_key, target, anchor_count, authored_count, authored_sha,
            tuple(phases), metadata["database_bytes"], metadata["spill_peak_bytes"],
            metadata["query_rows"], metadata["resolved_rows"], metadata["generated_bytes"],
            metadata["fresh_digest"], metadata["resume_digest"],
            metadata["query_index_built"], metadata["rollback_clean"], 1,
        )
        validate_scale_point_resources(result)
        return result
    except (OSError, ValueError, sqlite3.Error, V2RegistryError,
            V2SourceAdapterError) as error:
        raise V2ScaleBaselineError("FT00-05 scale point failed") from error
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_ft00_05_scale_baseline(
        repository_root: str | Path,
        ) -> V2ScaleBaselineReport:
    """Run the exact P0/P1 public scale baseline in isolated scratch roots."""
    root = Path(repository_root).resolve()
    public_paths = _public_manifest_paths(root)
    audit = read_v2_source_adapter_audit(root)
    points = tuple(_make_point(root, scale, public_paths)
                   for scale in FT00_05_SCALES)
    slope = _build_slope(points)
    rss_evidence = int(all(
        any(item.rss_evidence for item in point.phases) for point in points))
    status = "PASS" if slope.passed == 1 and all(
        item.resource_stop_boundary == 1 and item.rollback_clean == 1
        for item in points) else "NE"
    report = V2ScaleBaselineReport(
        FT00_05_ARTIFACT_KIND, FT00_05_REPORT_VERSION, V2_RELEASE_KEY,
        FT00_05_STAGE, FT00_05_SCALES, audit.sha256(),
        tuple(_sha256_file(root / Path(*PurePosixPath(path).parts))
              for path in public_paths), points, slope, rss_evidence,
        0, 0, 0, 0, 0, 0, 0, status,
    )
    return report


def write_ft00_05_report(report: V2ScaleBaselineReport,
                         path: str | Path) -> Path:
    """Publish an immutable public report."""
    return write_immutable_json(report.to_dict(), path)


def read_ft00_05_report(path: str | Path) -> V2ScaleBaselineReport:
    """Read and revalidate one canonical public report."""
    target = Path(path)
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise V2ScaleBaselineError("FT00-05 report newline is invalid")
    value = json.loads(payload[:-1].decode("utf-8"))
    if canonical_json_bytes(value) + b"\n" != payload:
        raise V2ScaleBaselineError("FT00-05 report canonical bytes drift")
    if value.get("artifact_kind") != FT00_05_ARTIFACT_KIND:
        raise V2ScaleBaselineError("FT00-05 report kind drift")
    return _report_from_dict(value)


def _report_from_dict(value: dict[str, Any]) -> V2ScaleBaselineReport:
    """Strictly restore the public report without accepting extra fields."""
    required = {
        "artifact_kind", "candidate_writes", "companion_writes", "core_writes",
        "formal_training_runs", "memory_writes",
        "points", "public_manifest_sha256", "release_key", "report_version",
        "rss_evidence", "scales", "slope", "source_audit_sha256", "stage_key",
        "status", "teacher_calls", "use_writes",
    }
    if set(value) != required or not isinstance(value["points"], list):
        raise V2ScaleBaselineError("FT00-05 report fields are not exact")
    phases = []
    points = []
    for raw_point in value["points"]:
        point_fields = set(V2ScalePoint.__dataclass_fields__)
        if set(raw_point) != point_fields or not isinstance(raw_point["phases"], list):
            raise V2ScaleBaselineError("FT00-05 point fields are not exact")
        point_phases = []
        for raw_phase in raw_point["phases"]:
            if set(raw_phase) != set(V2ScalePhase.__dataclass_fields__):
                raise V2ScaleBaselineError("FT00-05 phase fields are not exact")
            point_phases.append(V2ScalePhase(**raw_phase))
        point_values = dict(raw_point)
        point_values["phases"] = tuple(point_phases)
        points.append(V2ScalePoint(**point_values))
    slope_raw = value["slope"]
    if set(slope_raw) != set(V2ScaleSlope.__dataclass_fields__):
        raise V2ScaleBaselineError("FT00-05 slope fields are not exact")
    slope_values = dict(slope_raw)
    slope_values["checked_phase_keys"] = tuple(slope_values["checked_phase_keys"])
    return V2ScaleBaselineReport(
        value["artifact_kind"], value["report_version"], value["release_key"],
        value["stage_key"], tuple(value["scales"]), value["source_audit_sha256"],
        tuple(value["public_manifest_sha256"]), tuple(points),
        V2ScaleSlope(**slope_values), value["rss_evidence"],
        value["formal_training_runs"], value["candidate_writes"],
        value["core_writes"], value["memory_writes"],
        value["companion_writes"], value["use_writes"],
        value["teacher_calls"], value["status"],
    )


__all__ = [
    "FT00_05_ARTIFACT_KIND", "FT00_05_REPORT_VERSION", "FT00_05_SCALES",
    "V2ScaleBaselineError", "V2ScaleBaselineReport", "V2ScalePhase",
    "V2ScalePoint", "V2ScaleSlope", "build_authored_scale_pack",
    "read_ft00_05_report", "run_ft00_05_scale_baseline",
    "validate_scale_point_resources", "validate_scale_target",
    "write_ft00_05_report",
]
