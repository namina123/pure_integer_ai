"""Successor source adapter audit and v1-to-v2 public pack compiler.

This module only consumes the public D-02 source-pack surface.  It never
opens evaluator roots and never mutates an existing artifact.  A v1 pack is
either projected into a new v2 pack or reported as audit-only when its frozen
sample does not contain both a forming and an evaluation split.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_ADAPTER_VERSION,
    V2_COURSE_VERSION,
    V2_GENERATOR_VERSION,
    V2_PARSER_VERSION,
    V2_RELEASE_KEY,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    V2_SOURCE_KEYS,
    V2_SOURCE_LICENSES,
    validate_v2_record,
    validate_v2_record_set,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    ArtifactManifest,
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    read_artifact_manifest,
    read_record_artifact,
    write_artifact_manifest,
    write_record_artifact,
)
from pure_integer_ai.experiments.ph2_source_pack_catalog import (
    SOURCE_PACK_COVERAGE_PATH,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack,
    read_source_pack_coverage_manifest,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourcePackCoverageEntry,
)


V2_SOURCE_ADAPTER_FORMAT_VERSION = 1
V2_SOURCE_ARTIFACT_RELATIVE_ROOT = (
    "ph2_d03_v2_dataset_artifacts/source_adapter_v1")
V2_SOURCE_AUDIT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_source_adapter_audit_v1.json")
V2_SOURCE_STATUSES = ("READY", "SOURCE_ONLY", "BLOCKED")


class V2SourceAdapterError(RuntimeError):
    """A public source pack cannot satisfy the successor contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _safe_relative(value: Any, *, where: str) -> str:
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str) or not value:
        raise V2SourceAdapterError(f"{where} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    lowered = tuple(part.casefold() for part in path.parts)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value or "private" in lowered):
        raise V2SourceAdapterError(f"{where} is not a public relative path")
    return value


def _resolve(root: Path, relative: str, *, where: str) -> Path:
    safe = _safe_relative(relative, where=where)
    target = (root.resolve() / Path(*PurePosixPath(safe).parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise V2SourceAdapterError(f"{where} escapes repository root")
    return target


def _sha256(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise V2SourceAdapterError(f"{where} must be lowercase SHA-256")
    return value


def _exact(value: Any, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise V2SourceAdapterError(f"{where} fields are not exact")
    return value


def _counts_from_value(value: Any, *, where: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise V2SourceAdapterError(f"{where} must be an array")
    rows: list[tuple[str, int]] = []
    for raw in value:
        item = _exact(raw, {"count", "split"}, where=f"{where} item")
        rows.append((str(item["split"]), item["count"]))
    return tuple(rows)


def _validate_counts(value: tuple[tuple[str, int], ...], *, where: str) -> None:
    order = ("train", "dev", "held_out", "adversarial", "wall")
    if (not isinstance(value, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2 for item in value)
            or any(split not in order or type(count) is not int or count <= 0
                   for split, count in value)
            or tuple(sorted(value, key=lambda item: order.index(item[0]))) != value
            or len({split for split, _ in value}) != len(value)):
        raise V2SourceAdapterError(f"{where} is not canonical")


def _split_counts(values: Iterable[ObservationRecord]) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for item in values:
        counts[item.split] = counts.get(item.split, 0) + 1
    order = ("train", "dev", "held_out", "adversarial", "wall")
    return tuple((split, counts[split]) for split in order if counts.get(split, 0))


def _projected_label_counts(
        observations: Iterable[ObservationRecord],
        ) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for item in observations:
        if item.split != "train":
            counts[item.split] = counts.get(item.split, 0) + 1
    order = ("dev", "held_out", "adversarial", "wall")
    return tuple((split, counts[split]) for split in order if counts.get(split, 0))


def _key(namespace: str, *parts: Any) -> StableRecordKey:
    from pure_integer_ai.experiments.ph2_source_pack_contract import (
        stable_source_pack_key,
    )
    return stable_source_pack_key(namespace, V2_RELEASE_KEY, *parts)


def _assert_public_raw(value: Any, *, where: str) -> None:
    forbidden = {
        "expected", "expected_output", "expected_state", "teacher_output",
        "teacher_answer", "evaluator_label", "held_out_label",
        "fixture_label",
    }
    if isinstance(value, dict):
        for name, child in value.items():
            normalized = name.casefold().replace("-", "_").replace(" ", "_")
            if normalized in forbidden:
                raise V2SourceAdapterError(f"{where} contains evaluator payload")
            _assert_public_raw(child, where=f"{where}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_raw(child, where=f"{where}[{index}]")


def _locator(source: SourceRefRecord) -> tuple[str, str]:
    span = source.source_span.to_value()
    for key, kind in (
            ("page_id", "page"), ("qid", "entity"),
            ("sent_id", "sentence"), ("line_number", "record"),
            ("assertion_uri", "record")):
        if key in span:
            return kind, str(span[key])
    return "record", source.source_identity


@dataclass(frozen=True)
class V2SourcePackAuditEntry:
    """Public audit row for one source/license partition."""

    source_key: str
    license_partition: str
    d02_status: str
    raw_snapshot_manifest_relative_path: str
    raw_snapshot_manifest_sha256: str
    d02_manifest_relative_path: str
    d02_manifest_sha256: str
    d02_record_count: int
    splits: tuple[str, ...]
    source_cluster_count: int
    combination_cluster_count: int
    source_ref_count: int
    observation_counts: tuple[tuple[str, int], ...]
    teacher_evidence_count: int
    evaluator_label_counts: tuple[tuple[str, int], ...]
    status: str
    blocker_code: str
    limitation_code: str
    v2_manifest_relative_path: str
    v2_manifest_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.source_key, str) or not self.source_key
                or not isinstance(self.license_partition, str)
                or not self.license_partition):
            raise V2SourceAdapterError("source audit source/license is invalid")
        if self.d02_status not in {"PACK_FROZEN", "BLOCKED"}:
            raise V2SourceAdapterError("source audit D-02 status is invalid")
        if self.status not in V2_SOURCE_STATUSES:
            raise V2SourceAdapterError("unknown v2 source audit status")
        if self.status == "BLOCKED" and not self.blocker_code:
            raise V2SourceAdapterError("blocked source audit needs blocker code")
        if self.status != "BLOCKED" and self.blocker_code:
            raise V2SourceAdapterError("ready/source-only audit cannot have blocker")
        if self.status == "SOURCE_ONLY" and self.limitation_code != "SOURCE_SPLIT_INCOMPLETE":
            raise V2SourceAdapterError("source-only audit needs split limitation")
        if self.status != "SOURCE_ONLY" and self.limitation_code:
            raise V2SourceAdapterError("only source-only audit can have limitation")
        _safe_relative(
            self.raw_snapshot_manifest_relative_path,
            where="raw snapshot manifest",
        )
        _sha256(
            self.raw_snapshot_manifest_sha256,
            where="raw snapshot manifest sha256",
        )
        if self.d02_manifest_relative_path:
            _safe_relative(self.d02_manifest_relative_path, where="d02 manifest")
        if self.v2_manifest_relative_path:
            _safe_relative(self.v2_manifest_relative_path, where="v2 manifest")
        _sha256(self.d02_manifest_sha256, where="d02 manifest sha256")
        if self.v2_manifest_sha256:
            _sha256(self.v2_manifest_sha256, where="v2 manifest sha256")
        if type(self.d02_record_count) is not int or self.d02_record_count < 0:
            raise V2SourceAdapterError("d02 record count must be nonnegative")
        split_order = ("train", "dev", "held_out", "adversarial", "wall")
        if (not isinstance(self.splits, tuple)
                or any(split not in split_order for split in self.splits)
                or tuple(sorted(self.splits, key=split_order.index)) != self.splits
                or len(set(self.splits)) != len(self.splits)):
            raise V2SourceAdapterError("source audit splits are not canonical")
        _validate_counts(self.observation_counts, where="observation counts")
        _validate_counts(self.evaluator_label_counts, where="evaluator counts")
        for name in ("source_cluster_count", "combination_cluster_count",
                     "source_ref_count", "teacher_evidence_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise V2SourceAdapterError(f"source audit {name} is invalid")
        if self.d02_status == "PACK_FROZEN":
            if not self.d02_manifest_relative_path or self.d02_record_count <= 0:
                raise V2SourceAdapterError("frozen D-02 audit lacks manifest identity")
        elif (self.d02_manifest_relative_path or self.d02_record_count
              or self.d02_manifest_sha256 != "0" * 64):
            raise V2SourceAdapterError("blocked D-02 audit forges manifest identity")
        if self.status == "READY":
            if bool(self.v2_manifest_relative_path) != bool(self.v2_manifest_sha256):
                raise V2SourceAdapterError("ready audit v2 manifest identity is partial")
        elif self.v2_manifest_relative_path or self.v2_manifest_sha256:
            raise V2SourceAdapterError("non-ready audit cannot publish v2 manifest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocker_code": self.blocker_code,
            "combination_cluster_count": self.combination_cluster_count,
            "d02_manifest_relative_path": self.d02_manifest_relative_path,
            "d02_manifest_sha256": self.d02_manifest_sha256,
            "d02_record_count": self.d02_record_count,
            "d02_status": self.d02_status,
            "evaluator_label_counts": [
                {"count": count, "split": split}
                for split, count in self.evaluator_label_counts
            ],
            "license_partition": self.license_partition,
            "limitation_code": self.limitation_code,
            "observation_counts": [
                {"count": count, "split": split}
                for split, count in self.observation_counts
            ],
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "source_ref_count": self.source_ref_count,
            "raw_snapshot_manifest_relative_path": (
                self.raw_snapshot_manifest_relative_path),
            "raw_snapshot_manifest_sha256": self.raw_snapshot_manifest_sha256,
            "splits": list(self.splits),
            "status": self.status,
            "teacher_evidence_count": self.teacher_evidence_count,
            "v2_manifest_relative_path": self.v2_manifest_relative_path,
            "v2_manifest_sha256": self.v2_manifest_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2SourcePackAuditEntry":
        raw = _exact(value, {
            "blocker_code", "combination_cluster_count",
            "d02_manifest_relative_path", "d02_manifest_sha256",
            "d02_record_count", "d02_status", "evaluator_label_counts",
            "license_partition", "limitation_code", "observation_counts",
            "raw_snapshot_manifest_relative_path",
            "raw_snapshot_manifest_sha256", "source_cluster_count",
            "source_key", "source_ref_count", "splits", "status",
            "teacher_evidence_count", "v2_manifest_relative_path",
            "v2_manifest_sha256",
        }, where="V2SourcePackAuditEntry")
        if not isinstance(raw["splits"], list):
            raise V2SourceAdapterError("source audit splits must be an array")
        return cls(
            str(raw["source_key"]), str(raw["license_partition"]),
            str(raw["d02_status"]),
            str(raw["raw_snapshot_manifest_relative_path"]),
            str(raw["raw_snapshot_manifest_sha256"]),
            str(raw["d02_manifest_relative_path"]),
            str(raw["d02_manifest_sha256"]), raw["d02_record_count"],
            tuple(str(item) for item in raw["splits"]),
            raw["source_cluster_count"], raw["combination_cluster_count"],
            raw["source_ref_count"],
            _counts_from_value(raw["observation_counts"], where="observation counts"),
            raw["teacher_evidence_count"],
            _counts_from_value(raw["evaluator_label_counts"], where="evaluator counts"),
            str(raw["status"]), str(raw["blocker_code"]),
            str(raw["limitation_code"]),
            str(raw["v2_manifest_relative_path"]),
            str(raw["v2_manifest_sha256"]),
        )


@dataclass(frozen=True)
class V2SourceAdapterAudit:
    """Append-only audit covering all public D-02 source partitions."""

    format_version: int
    release_key: str
    parent_coverage_manifest_relative_path: str
    parent_coverage_manifest_sha256: str
    entries: tuple[V2SourcePackAuditEntry, ...]
    ready_pack_count: int
    source_only_count: int
    blocked_count: int
    formal_training_runs: int = 0
    teacher_calls: int = 0
    candidate_writes: int = 0
    core_writes: int = 0
    memory_writes: int = 0

    def __post_init__(self) -> None:
        if self.format_version != V2_SOURCE_ADAPTER_FORMAT_VERSION:
            raise V2SourceAdapterError("source audit format version drift")
        if self.release_key != V2_RELEASE_KEY:
            raise V2SourceAdapterError("source audit release drift")
        _safe_relative(
            self.parent_coverage_manifest_relative_path,
            where="parent coverage manifest",
        )
        _sha256(
            self.parent_coverage_manifest_sha256,
            where="parent coverage manifest sha256",
        )
        if not self.entries:
            raise V2SourceAdapterError("source audit entries cannot be empty")
        identities = tuple((item.source_key, item.license_partition)
                          for item in self.entries)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise V2SourceAdapterError("source audit entry order or identity drift")
        actual = {
            "READY": sum(item.status == "READY" for item in self.entries),
            "SOURCE_ONLY": sum(item.status == "SOURCE_ONLY" for item in self.entries),
            "BLOCKED": sum(item.status == "BLOCKED" for item in self.entries),
        }
        if (self.ready_pack_count, self.source_only_count, self.blocked_count) != (
                actual["READY"], actual["SOURCE_ONLY"], actual["BLOCKED"]):
            raise V2SourceAdapterError("source audit status counts drift")
        for name in ("formal_training_runs", "teacher_calls", "candidate_writes",
                     "core_writes", "memory_writes"):
            if type(getattr(self, name)) is not int or getattr(self, name) != 0:
                raise V2SourceAdapterError(f"source audit {name} must remain zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_D03_V2_SOURCE_ADAPTER_AUDIT",
            "blocked_count": self.blocked_count,
            "candidate_writes": self.candidate_writes,
            "core_writes": self.core_writes,
            "entries": [item.to_dict() for item in self.entries],
            "formal_training_runs": self.formal_training_runs,
            "format_version": self.format_version,
            "memory_writes": self.memory_writes,
            "parent_coverage_manifest_relative_path": (
                self.parent_coverage_manifest_relative_path),
            "parent_coverage_manifest_sha256": self.parent_coverage_manifest_sha256,
            "ready_pack_count": self.ready_pack_count,
            "release_key": self.release_key,
            "source_only_count": self.source_only_count,
            "teacher_calls": self.teacher_calls,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "V2SourceAdapterAudit":
        raw = _exact(value, {
            "artifact_kind", "blocked_count", "candidate_writes",
            "core_writes", "entries", "formal_training_runs",
            "format_version", "memory_writes",
            "parent_coverage_manifest_relative_path",
            "parent_coverage_manifest_sha256", "ready_pack_count",
            "release_key", "source_only_count", "teacher_calls",
        }, where="V2SourceAdapterAudit")
        if (raw["artifact_kind"] != "PH2_D03_V2_SOURCE_ADAPTER_AUDIT"
                or not isinstance(raw["entries"], list)):
            raise V2SourceAdapterError("source adapter audit kind/entries drift")
        return cls(
            raw["format_version"], str(raw["release_key"]),
            str(raw["parent_coverage_manifest_relative_path"]),
            str(raw["parent_coverage_manifest_sha256"]),
            tuple(V2SourcePackAuditEntry.from_dict(item)
                  for item in raw["entries"]),
            raw["ready_pack_count"], raw["source_only_count"],
            raw["blocked_count"], raw["formal_training_runs"],
            raw["teacher_calls"], raw["candidate_writes"],
            raw["core_writes"], raw["memory_writes"],
        )


@dataclass(frozen=True)
class V2SourcePackBuild:
    """A newly published or idempotently resumed v2 public pack."""

    pack_root: Path
    manifest: ArtifactManifest
    audit: V2SourcePackAuditEntry
    published: bool


def _audit_from_bundle(
        entry: SourcePackCoverageEntry,
        *,
        bundle: Any,
        v2_manifest_relative_path: str = "",
        v2_manifest_sha256: str = "",
        ) -> V2SourcePackAuditEntry:
    observations = bundle.observations
    nontrain = tuple(item for item in observations if item.split != "train")
    status = "READY" if any(item.split == "train" for item in observations) and nontrain else "SOURCE_ONLY"
    return V2SourcePackAuditEntry(
        entry.source_key,
        entry.license_partition,
        entry.status,
        entry.raw_snapshot_manifest_relative_path,
        entry.raw_snapshot_manifest_sha256,
        entry.pack_manifest_relative_path,
        entry.pack_manifest_sha256,
        bundle.manifest.record_count,
        bundle.manifest.splits,
        bundle.validation.source_cluster_count,
        bundle.combination_audit.to_value()["combination_cluster_count"],
        len(bundle.sources),
        _split_counts(observations),
        sum(item.split == "train" for item in observations),
        _projected_label_counts(observations),
        status,
        "",
        "" if status == "READY" else "SOURCE_SPLIT_INCOMPLETE",
        v2_manifest_relative_path,
        v2_manifest_sha256,
    )


def audit_d02_source_pack(
        repository_root: str | Path,
        entry: SourcePackCoverageEntry,
        ) -> V2SourcePackAuditEntry:
    """Audit one public D-02 pack without opening any private surface."""
    root = Path(repository_root).resolve()
    raw_manifest = _resolve(
        root, entry.raw_snapshot_manifest_relative_path,
        where="raw snapshot manifest",
    )
    if _sha256_file(raw_manifest) != entry.raw_snapshot_manifest_sha256:
        raise V2SourceAdapterError("raw snapshot manifest SHA drift")
    if entry.status == "BLOCKED":
        return V2SourcePackAuditEntry(
            entry.source_key, entry.license_partition, entry.status,
            entry.raw_snapshot_manifest_relative_path,
            entry.raw_snapshot_manifest_sha256, "",
            "0" * 64, 0, (), 0, 0, 0, (), 0, (), "BLOCKED",
            entry.blocker_code, "", "", "",
        )
    if entry.status != "PACK_FROZEN":
        raise V2SourceAdapterError("unsupported D-02 coverage status")
    if entry.source_key not in V2_SOURCE_KEYS:
        raise V2SourceAdapterError("source is not in v2 allowlist")
    if entry.license_partition not in V2_SOURCE_LICENSES[entry.source_key]:
        raise V2SourceAdapterError("source license is not in v2 allowlist")
    manifest_path = _resolve(
        root, entry.pack_manifest_relative_path, where="D-02 pack manifest")
    if _sha256_file(manifest_path) != entry.pack_manifest_sha256:
        raise V2SourceAdapterError("D-02 pack manifest SHA drift")
    bundle = read_source_pack(manifest_path.parent)
    if (bundle.manifest.record_count != entry.pack_record_count
            or bundle.manifest.splits != entry.splits
            or bundle.validation.source_cluster_count != entry.source_cluster_count
            or bundle.combination_audit.to_value()["combination_cluster_count"] != entry.combination_cluster_count):
        raise V2SourceAdapterError("D-02 coverage counts drift")
    return _audit_from_bundle(entry, bundle=bundle)


def audit_d02_source_coverage(
        repository_root: str | Path,
        *,
        coverage_relative_path: str = SOURCE_PACK_COVERAGE_PATH,
        ) -> V2SourceAdapterAudit:
    """Audit all public D-02 source partitions and preserve blockers."""
    root = Path(repository_root).resolve()
    if isinstance(coverage_relative_path, Path):
        coverage_relative_path = coverage_relative_path.as_posix()
    coverage_path = _resolve(root, coverage_relative_path, where="D-02 coverage")
    coverage = read_source_pack_coverage_manifest(coverage_path)
    coverage_sha = _sha256_file(coverage_path)
    rows: list[V2SourcePackAuditEntry] = []
    for entry in coverage.entries:
        try:
            rows.append(audit_d02_source_pack(root, entry))
        except V2SourceAdapterError as error:
            rows.append(V2SourcePackAuditEntry(
                entry.source_key, entry.license_partition, entry.status,
                entry.raw_snapshot_manifest_relative_path,
                entry.raw_snapshot_manifest_sha256,
                entry.pack_manifest_relative_path, entry.pack_manifest_sha256,
                entry.pack_record_count, entry.splits,
                entry.source_cluster_count, entry.combination_cluster_count,
                0, (), 0, (), "BLOCKED", "V2_ADAPTER_INVALID: " + str(error),
                "", "", "",
            ))
    rows.sort(key=lambda item: (item.source_key, item.license_partition))
    return V2SourceAdapterAudit(
        V2_SOURCE_ADAPTER_FORMAT_VERSION,
        V2_RELEASE_KEY,
        coverage_relative_path,
        coverage_sha,
        tuple(rows),
        sum(item.status == "READY" for item in rows),
        sum(item.status == "SOURCE_ONLY" for item in rows),
        sum(item.status == "BLOCKED" for item in rows),
    )


def _v2_records_from_bundle(
        entry: SourcePackCoverageEntry,
        bundle: Any,
        ) -> tuple[tuple[SourceRefRecord, ...], tuple[ObservationRecord, ...],
                   tuple[TeacherEvidenceRecord, ...], tuple[EvaluatorLabelRecord, ...],
                   StableRecordKey, StableRecordKey, StableRecordKey, StableRecordKey]:
    observations_v1 = tuple(sorted(bundle.observations, key=lambda item: item.stable_key))
    if not any(item.split == "train" for item in observations_v1) or not any(
            item.split != "train" for item in observations_v1):
        raise V2SourceAdapterError("SOURCE_SPLIT_INCOMPLETE")
    source_by_key = {item.stable_key: item for item in bundle.sources}
    source_clusters = {
        old.source_cluster_key: _key("source-cluster", entry.source_key, old.source_cluster_key.components)
        for old in bundle.sources
    }
    dataset_key = _key("dataset", entry.source_key, entry.license_partition)
    artifact_key = _key(
        "artifact", entry.source_key, entry.license_partition,
        entry.pack_manifest_sha256,
    )
    sources: list[SourceRefRecord] = []
    source_map: dict[StableRecordKey, StableRecordKey] = {}
    for ordinal, old in enumerate(sorted(bundle.sources, key=lambda item: item.stable_key), start=1):
        source_key = _key("source-ref", entry.source_key, old.stable_key.components,
                          old.source_identity, old.local_sha256)
        source_map[old.stable_key] = source_key
        locator_kind, locator_value = _locator(old)
        span = old.source_span.to_value()
        source_span = {
            "document_cluster_key": _key(
                "document-cluster", entry.source_key,
                old.source_cluster_key.components).to_list(),
            "entity_graph_cluster_key": _key(
                "entity-cluster", entry.source_key,
                old.source_cluster_key.components).to_list(),
            "locator_kind": locator_kind,
            "locator_value": locator_value,
            "span_end": max(1, len(canonical_json_bytes(span))),
            "span_start": 0,
        }
        sources.append(SourceRefRecord(
            2, 2, V2_COURSE_VERSION, dataset_key, artifact_key, source_key,
            old.source_key, old.snapshot_id, old.revision_id,
            old.official_url, old.source_identity, old.upstream_checksum,
            old.local_sha256, old.license_id, "PUBLIC", old.attribution,
            V2_PARSER_VERSION, CanonicalJsonObject.from_value(source_span),
            ordinal, source_clusters[old.source_cluster_key],
        ))
    obs_map: dict[StableRecordKey, ObservationRecord] = {}
    observations: list[ObservationRecord] = []
    for old in observations_v1:
        source = source_by_key.get(old.source_ref_key)
        if source is None:
            raise V2SourceAdapterError("Observation source_ref is missing")
        payload = old.typed_payload.to_value()
        raw = payload.get("raw_observation")
        if not isinstance(raw, dict):
            raise V2SourceAdapterError("Observation raw payload is missing")
        _assert_public_raw(raw, where="D-02 raw observation")
        raw_sha = payload.get("raw_observation_sha256")
        if not isinstance(raw_sha, str):
            raw_sha = hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
        _sha256(raw_sha, where="D-02 raw observation")
        observation_key = _key(
            "observation", entry.source_key, old.stable_key.components, raw_sha)
        old_clusters = (
            old.dedup_cluster_key, old.content_group_key,
            old.template_group_key, old.shape_group_key,
        )
        carrier_node = _key("carrier-node", entry.source_key, old.stable_key.components)
        typed = CanonicalJsonObject.from_value({
            "carrier": {
                "carrier_kind": "document_container",
                "edges": [],
                "nodes": [{
                    "attributes": {
                        "raw_observation_sha256": raw_sha,
                        "source_identity": source.source_identity,
                        "source_representation": old.representation,
                    },
                    "node_key": carrier_node.to_list(),
                    "node_kind": "source_record",
                    "parent_node_key": None,
                    "span_end": max(1, len(canonical_json_bytes(raw))),
                    "span_start": 0,
                }],
                "raw_text_sha256": raw_sha,
                "root_node_keys": [carrier_node.to_list()],
            },
            "language_payload": {
                "combination_axes": payload.get("combination_axes", {}),
                "raw_observation": raw,
                "source_adapter": {
                    "d02_manifest_relative_path": entry.pack_manifest_relative_path,
                    "d02_manifest_sha256": entry.pack_manifest_sha256,
                    "source_span": source.source_span.to_value(),
                    "source_ref_key": source.stable_key.to_list(),
                },
            },
        })
        converted = ObservationRecord(
            2, 2, V2_COURSE_VERSION, dataset_key, artifact_key,
            observation_key, old.w_stage,
            "FT00-04-" + old.substage, old.split, old.language,
            "document_container", source_map[source.stable_key],
            old.license_partition,
            _key("dedup", entry.source_key, old_clusters[0].components),
            _key("content", entry.source_key, old_clusters[1].components),
            _key("template", entry.source_key, old_clusters[2].components),
            _key("shape", entry.source_key, old_clusters[3].components),
            "forming" if old.split == "train" else "evaluator",
            old.sample_role, "typed_carrier", typed, old.perturbation_kind,
            None, (), old.logical_order,
        )
        observations.append(converted)
        obs_map[old.stable_key] = converted
    teacher_owner = _key("owner", entry.source_key, entry.license_partition, "teacher")
    evaluator_owner = _key("owner", entry.source_key, entry.license_partition, "evaluator")
    teachers: list[TeacherEvidenceRecord] = []
    labels: list[EvaluatorLabelRecord] = []
    for old in observations_v1:
        converted = obs_map[old.stable_key]
        source = source_by_key[old.source_ref_key]
        if old.split == "train":
            teachers.append(TeacherEvidenceRecord(
                2, 2, V2_COURSE_VERSION, dataset_key, artifact_key,
                _key("teacher", entry.source_key, old.stable_key.components),
                converted.stable_key, "SOURCE_ADAPTER_RECEIPT_V2",
                CanonicalJsonObject.from_value({
                    "definitive_truth_authoritative": 0,
                    "raw_observation_sha256": converted.typed_payload.to_value()[
                        "carrier"]["raw_text_sha256"],
                    "source_ref_key": source_map[source.stable_key].to_list(),
                }), source_map[source.stable_key], old.w_stage, 3,
                teacher_owner,
            ))
        else:
            labels.append(EvaluatorLabelRecord(
                2, 2, V2_COURSE_VERSION, dataset_key, artifact_key,
                _key("integrity-label", entry.source_key, old.stable_key.components),
                converted.stable_key,
                _key("dimension", "SOURCE_ADAPTER_INTEGRITY_V2"), "TRUE",
                CanonicalJsonObject.from_value({
                    "definitive_truth_authoritative": 0,
                    "raw_observation_sha256": converted.typed_payload.to_value()[
                        "carrier"]["raw_text_sha256"],
                    "source_binding_required": 1,
                }), 1, V2_PARSER_VERSION, old.w_stage, evaluator_owner,
            ))
    validate_v2_record_set(
        [item.to_dict() for item in sources + observations + teachers + labels],
        teacher_owner_key=teacher_owner.components,
        evaluator_owner_key=evaluator_owner.components,
    )
    return (
        tuple(sources), tuple(observations), tuple(teachers), tuple(labels),
        dataset_key, artifact_key, teacher_owner, evaluator_owner,
    )


def _manifest_for_records(
        root: Path,
        relative_pack: str,
        sources: tuple[SourceRefRecord, ...],
        observations: tuple[ObservationRecord, ...],
        teachers: tuple[TeacherEvidenceRecord, ...],
        labels: tuple[EvaluatorLabelRecord, ...],
        dataset_key: StableRecordKey,
        artifact_key: StableRecordKey,
        stage: str,
        ) -> ArtifactManifest:
    target = root / Path(*PurePosixPath(relative_pack).parts)
    clusters = tuple(sorted({item.source_cluster_key for item in sources}))
    files: list[ArtifactFileIdentity] = [write_record_artifact(
        sources, target,
        ArtifactWriteSpec("source_ref", "source", "source_refs.jsonl.gz",
                          None, sources[0].license_id, clusters),
    )]
    split_order = ("train", "dev", "held_out", "adversarial", "wall")
    splits = tuple(split for split in split_order
                   if any(item.split == split for item in observations))
    for split in splits:
        split_obs = tuple(item for item in observations if item.split == split)
        split_clusters = tuple(sorted({
            source.source_cluster_key for item in split_obs
            for source in sources if source.stable_key == item.source_ref_key
        }))
        files.append(write_record_artifact(
            split_obs, target,
            ArtifactWriteSpec("observation", "observation",
                              f"observations/{split}.jsonl.gz", split,
                              sources[0].license_id, split_clusters),
        ))
        if split == "train":
            owner_records = tuple(item for item in teachers if item.observation_key in {
                observation.stable_key for observation in split_obs})
            files.append(write_record_artifact(
                owner_records, target,
                ArtifactWriteSpec("teacher_evidence", "teacher",
                                  "owners/teacher/train.evidence.jsonl.gz",
                                  "train", sources[0].license_id, split_clusters),
            ))
        else:
            owner_records = tuple(item for item in labels if item.observation_key in {
                observation.stable_key for observation in split_obs})
            files.append(write_record_artifact(
                owner_records, target,
                ArtifactWriteSpec("evaluator_label", "evaluator",
                                  f"owners/evaluator/{split}.labels.jsonl.gz",
                                  split, sources[0].license_id, split_clusters),
            ))
    manifest = ArtifactManifest(
        2, 2, V2_COURSE_VERSION, 2, dataset_key, artifact_key,
        sources[0].source_key, sources[0].license_id, "PUBLIC",
        V2_ADAPTER_VERSION, V2_GENERATOR_VERSION, V2_PARSER_VERSION,
        tuple(files), splits, (stage,), clusters, (), stage,
    )
    validate_v2_record(manifest.to_dict())
    return manifest


def compile_v2_source_pack(
        repository_root: str | Path,
        d02_manifest_relative_path: str,
        *,
        output_relative_root: str = V2_SOURCE_ARTIFACT_RELATIVE_ROOT,
        ) -> V2SourcePackBuild:
    """Compile one compatible public D-02 pack into an immutable v2 pack."""
    root = Path(repository_root).resolve()
    manifest_path = _resolve(root, d02_manifest_relative_path, where="D-02 pack")
    if _sha256_file(manifest_path) == "":
        raise V2SourceAdapterError("D-02 pack manifest is empty")
    bundle = read_source_pack(manifest_path.parent)
    first_span = bundle.sources[0].source_span.to_value()
    raw_snapshot_relative = first_span.get(
        "raw_snapshot_manifest_relative_path",
        "data/ph2/manifests/unknown.raw_snapshot.json",
    )
    raw_snapshot_sha256 = first_span.get(
        "raw_snapshot_manifest_sha256", "0" * 64)
    if not isinstance(raw_snapshot_relative, str) or not isinstance(
            raw_snapshot_sha256, str):
        raise V2SourceAdapterError("D-02 source snapshot binding is malformed")
    entry = SourcePackCoverageEntry(
        bundle.manifest.source_key, bundle.manifest.license_partition,
        "PACK_FROZEN", raw_snapshot_relative, raw_snapshot_sha256,
        d02_manifest_relative_path,
        _sha256_file(manifest_path), bundle.manifest.record_count,
        bundle.manifest.splits, len(bundle.manifest.source_cluster_keys),
        bundle.combination_audit.to_value()["combination_cluster_count"], "",
        (d02_manifest_relative_path,),
    )
    audit = _audit_from_bundle(entry, bundle=bundle)
    if audit.status != "READY":
        raise V2SourceAdapterError("SOURCE_SPLIT_INCOMPLETE")
    sources, observations, teachers, labels, dataset_key, artifact_key, _, _ = (
        _v2_records_from_bundle(entry, bundle))
    old_stage = bundle.manifest.w_stages[0]
    relative_root = _safe_relative(output_relative_root, where="v2 artifact root")
    pack_name = PurePosixPath(d02_manifest_relative_path).parent.name + "--v2"
    relative_pack = f"{relative_root}/packs/{pack_name}"
    target = _resolve(root, relative_pack, where="v2 pack")
    published = False
    if target.exists():
        if not target.is_dir():
            raise V2SourceAdapterError("v2 pack target is not a directory")
        existing_manifest = read_artifact_manifest(target / "manifest.json")
        if existing_manifest.source_key != bundle.manifest.source_key:
            raise V2SourceAdapterError("v2 pack source identity drift")
        actual_records = tuple(
            record
            for identity in existing_manifest.files
            for record in read_record_artifact(target, identity)
        )
        expected_records = tuple(sources + observations + teachers + labels)
        if {item.stable_key: item.to_dict() for item in actual_records} != {
                item.stable_key: item.to_dict() for item in expected_records}:
            raise V2SourceAdapterError("v2 pack resume records drift")
        validate_v2_record(existing_manifest.to_dict())
        validate_v2_record_set(
            [item.to_dict() for item in actual_records],
            teacher_owner_key=_key("owner", bundle.manifest.source_key,
                                   bundle.manifest.license_partition, "teacher").components,
            evaluator_owner_key=_key("owner", bundle.manifest.source_key,
                                     bundle.manifest.license_partition, "evaluator").components,
        )
        manifest = existing_manifest
    else:
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{pack_name}.building-", dir=parent))
        try:
            manifest = _manifest_for_records(
                staging, relative_pack=".", sources=sources,
                observations=observations, teachers=teachers, labels=labels,
                dataset_key=dataset_key, artifact_key=artifact_key,
                stage=old_stage)
            write_artifact_manifest(manifest, staging)
            os.replace(staging, target)
            published = True
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    v2_manifest_relative = f"{relative_pack}/manifest.json"
    final_audit = replace(
        audit, v2_manifest_relative_path=v2_manifest_relative,
        v2_manifest_sha256=_sha256_file(target / "manifest.json"),
    )
    return V2SourcePackBuild(target, manifest, final_audit, published)


def build_v2_source_adapter_audit(
        repository_root: str | Path,
        *,
        compile_ready: bool = False,
        output_relative_root: str = V2_SOURCE_ARTIFACT_RELATIVE_ROOT,
        ) -> tuple[V2SourceAdapterAudit, tuple[V2SourcePackBuild, ...]]:
    """Audit D-02 coverage and optionally compile all ready public packs."""
    root = Path(repository_root).resolve()
    report = audit_d02_source_coverage(root)
    builds: list[V2SourcePackBuild] = []
    rows = list(report.entries)
    if compile_ready:
        for index, row in enumerate(rows):
            if row.status != "READY":
                continue
            build = compile_v2_source_pack(
                root, row.d02_manifest_relative_path,
                output_relative_root=output_relative_root,
            )
            builds.append(build)
            rows[index] = build.audit
        report = V2SourceAdapterAudit(
            report.format_version, report.release_key,
            report.parent_coverage_manifest_relative_path,
            report.parent_coverage_manifest_sha256, tuple(rows),
            sum(item.status == "READY" for item in rows),
            sum(item.status == "SOURCE_ONLY" for item in rows),
            sum(item.status == "BLOCKED" for item in rows),
        )
    return report, tuple(builds)


def write_v2_source_adapter_audit(
        audit: V2SourceAdapterAudit,
        repository_root: str | Path,
        *,
        relative_path: str = V2_SOURCE_AUDIT_RELATIVE_PATH,
        ) -> Path:
    """Publish the audit report exclusively or idempotently."""
    root = Path(repository_root).resolve()
    target = _resolve(root, relative_path, where="v2 source audit")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = audit.canonical_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise V2SourceAdapterError("v2 source audit cannot overwrite bytes")
        return target
    with target.open("xb") as handle:
        handle.write(payload)
    return target


def read_v2_source_adapter_audit(
        repository_root: str | Path,
        *,
        relative_path: str = V2_SOURCE_AUDIT_RELATIVE_PATH,
        ) -> V2SourceAdapterAudit:
    """Read and revalidate one canonical public audit report."""
    root = Path(repository_root).resolve()
    target = _resolve(root, relative_path, where="v2 source audit")
    try:
        payload = target.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise V2SourceAdapterError("v2 source audit newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        audit = V2SourceAdapterAudit.from_dict(value)
    except V2SourceAdapterError:
        raise
    except (OSError, ValueError) as error:
        raise V2SourceAdapterError("v2 source audit cannot be read") from error
    if audit.canonical_bytes() != payload:
        raise V2SourceAdapterError("v2 source audit canonical bytes drift")
    return audit


__all__ = [
    "V2_SOURCE_ADAPTER_FORMAT_VERSION",
    "V2_SOURCE_STATUSES",
    "V2_SOURCE_ARTIFACT_RELATIVE_ROOT",
    "V2_SOURCE_AUDIT_RELATIVE_PATH",
    "V2SourceAdapterAudit",
    "V2SourceAdapterError",
    "V2SourcePackAuditEntry",
    "V2SourcePackBuild",
    "audit_d02_source_coverage",
    "audit_d02_source_pack",
    "build_v2_source_adapter_audit",
    "compile_v2_source_pack",
    "read_v2_source_adapter_audit",
    "write_v2_source_adapter_audit",
]
