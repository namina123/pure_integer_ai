"""Unlabeled shadow identity-transfer gate for W-02 V3 source routing."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _sha256_file,
    _tree_sha256,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_shadow_audit import (
    W02_SHADOW_LAYOUTS,
    W02ShadowInputRoot,
    _audit_prediction,
    _gate,
    _select_shadow_spans,
    _shadow_identity,
    iter_w02_shadow_records,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN,
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_dev_calibration import (
    _assert_v1_retained,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_audit import (
    W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_dev_probe import (
    W02_MORPH_V3_DEV_FREEZE_PATH,
    read_w02_morphology_successor_v3_dev_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySourceCapability,
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
)


W02_MORPH_V3_SHADOW_PROBE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-ROUTE-SHADOW-PROBE-V1"
)
W02_MORPH_V3_SHADOW_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-ROUTE-SHADOW-FREEZE-V1"
)
W02_MORPH_V3_SHADOW_REPORT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-ROUTE-SHADOW-REPORT-V1"
)
W02_MORPH_V3_SHADOW_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_route_shadow_freeze_v1.json"
)
W02_MORPH_V3_SHADOW_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_route_shadow_report_v1.json"
)
W02_MORPH_V3_DEV_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_route_dev_report_v1.json"
)
W02_MORPH_V3_SNAPSHOT_PATH = (
    "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json"
)
W02_MORPH_V3_PARENT_SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18"
W02_MORPH_V3_TRAIN_SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18_ROUTE_PROBE_TRAIN"
W02_MORPH_V3_DEV_SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18_ROUTE_PROBE_DEV"
W02_MORPH_V3_TRAIN_DATASET_KEY = StableRecordKey((2, 2, 201, 1))
W02_MORPH_V3_TRAIN_ARTIFACT_KEY = StableRecordKey((2, 2, 201, 2))
W02_MORPH_V3_DEV_DATASET_KEY = StableRecordKey((2, 2, 202, 1))
W02_MORPH_V3_DEV_ARTIFACT_KEY = StableRecordKey((2, 2, 202, 2))
W02_MORPH_V3_EXPECTED_SOURCE_COUNT = 50_322
W02_MORPH_V3_EXPECTED_OBSERVATION_COUNT = 58_506
W02_MORPH_V3_EXPECTED_ROUTED_COUNT = 4_497
W02_MORPH_V3_EXPECTED_TRAIN_ROUTED_COUNT = 3_997
W02_MORPH_V3_EXPECTED_DEV_ROUTED_COUNT = 500
W02_MORPH_V3_MAX_LOGIC_OPERATIONS = 9_000_000
W02_MORPH_V3_SHADOW_GATES = (
    "W02-SHADOW-V3-ROUTE-AUTHORIZED",
    "W02-SHADOW-V3-OLD-ROUTE-ZERO",
    "W02-SHADOW-V3-CANDIDATE-PROJECTION",
    "W02-SHADOW-V3-CARRIER-BOUNDARY-UNICODE",
    "W02-SHADOW-V3-FINAL-PROJECTION",
    "W02-SHADOW-V3-V1-V2-BOUNDED-RETENTION",
)
W02_MORPH_V3_SHADOW_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_shadow_probe.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v3_shadow_probe.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v3_shadow_probe.py",
)
W02_MORPH_V3_SHADOW_PARENT_PUBLIC_PATHS = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_compile_freeze_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_candidate_artifact_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_artifact_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_overlay_artifact_v1.json",
    W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH,
    W02_MORPH_V3_DEV_FREEZE_PATH,
    W02_MORPH_V3_DEV_REPORT_PATH,
)


# object-model: exception
class W02MorphologySuccessorV3ShadowProbeError(RuntimeError):
    """The V3 shadow identity-transfer probe or a public parent drifted."""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow repository path is invalid")
    current = repository
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02MorphologySuccessorV3ShadowProbeError(
                "V3 shadow repository path crosses a symlink")
    target = (repository / Path(*pure.parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow repository file is missing")
    return target


def _file_rows(
        repository: Path,
        paths: tuple[str, ...],
        ) -> list[dict[str, object]]:
    rows = []
    for relative in paths:
        size, sha256 = _sha256_file(_repository_file(repository, relative))
        rows.append({
            "repository_path": relative,
            "sha256": sha256,
            "size_bytes": size,
        })
    return rows


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyShadowRouteSpec:
    """One synthetic route identity for one immutable upstream blob."""

    split: str
    source_key: str
    dataset_key: StableRecordKey
    artifact_key: StableRecordKey
    key_marker: int
    upstream_checksum: str
    capability: W02MorphologySourceCapability

    def __post_init__(self) -> None:
        if (self.split not in {"train", "dev"}
                or not self.source_key
                or not isinstance(self.dataset_key, StableRecordKey)
                or not isinstance(self.artifact_key, StableRecordKey)
                or type(self.key_marker) is not int or self.key_marker <= 0
                or not self.upstream_checksum.startswith("sha1:")
                or not isinstance(
                    self.capability, W02MorphologySourceCapability)
                or self.capability.source_key != self.source_key
                or self.capability.upstream_checksum
                != self.upstream_checksum):
            raise W02MorphologySuccessorV3ShadowProbeError(
                "V3 shadow route spec drifted")


def _key(marker: int, kind: int, ordinal: int) -> StableRecordKey:
    if (type(marker) is not int or marker <= 0
            or type(kind) is not int or kind <= 0
            or type(ordinal) is not int or ordinal <= 0):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow stable key component is invalid")
    return StableRecordKey((2, 2, marker, kind, ordinal))


def _shadow_route_specs(
        repository: Path,
        ) -> tuple[W02MorphologyShadowRouteSpec, ...]:
    snapshot = read_canonical_object(
        _repository_file(repository, W02_MORPH_V3_SNAPSHOT_PATH))
    if (snapshot.get("source_key") != W02_MORPH_V3_PARENT_SOURCE_KEY
            or snapshot.get("license_id") != "CC-BY-SA-4.0"):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow public snapshot identity drifted")
    files = snapshot.get("files")
    if not isinstance(files, list):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow public snapshot files drifted")
    by_split = {
        str(row.get("split")): row for row in files
        if isinstance(row, dict) and row.get("file_kind") == "conllu"
        and row.get("split") in {"train", "dev"}
    }
    if set(by_split) != {"train", "dev"}:
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow train/dev blob inventory drifted")
    definitions = (
        ("train", W02_MORPH_V3_TRAIN_SOURCE_KEY,
         W02_MORPH_V3_TRAIN_DATASET_KEY, W02_MORPH_V3_TRAIN_ARTIFACT_KEY,
         201),
        ("dev", W02_MORPH_V3_DEV_SOURCE_KEY,
         W02_MORPH_V3_DEV_DATASET_KEY, W02_MORPH_V3_DEV_ARTIFACT_KEY,
         202),
    )
    specs = []
    for split, source_key, dataset_key, artifact_key, marker in definitions:
        row = by_split[split]
        capability = w02_ud_morphology_source_capability({
            "annotation_provenance": (
                "Universal Dependencies documented treebank annotation"),
            "commit_sha1": snapshot["commit_sha1"],
            "data_file": {"git_blob_sha1": row["git_blob_sha1"]},
            "language": "zh",
            "license_id": snapshot["license_id"],
            "repository_url": snapshot["repository_url"],
            "snapshot_id": "ud-zh-gsdsimp-r2.18",
            "source_key": source_key,
        })
        specs.append(W02MorphologyShadowRouteSpec(
            split, source_key, dataset_key, artifact_key, marker,
            "sha1:" + str(row["git_blob_sha1"]), capability))
    return tuple(specs)


def _remap_source(
        source: SourceRefRecord,
        ordinal: int,
        spec: W02MorphologyShadowRouteSpec,
        ) -> SourceRefRecord:
    if (not isinstance(source, SourceRefRecord)
            or not isinstance(spec, W02MorphologyShadowRouteSpec)
            or source.source_key != W02_MORPH_V3_PARENT_SOURCE_KEY
            or source.upstream_checksum != spec.upstream_checksum):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow parent SourceRef drifted")
    return replace(
        source,
        dataset_key=spec.dataset_key,
        artifact_key=spec.artifact_key,
        stable_key=_key(spec.key_marker, 10, ordinal),
        source_key=spec.source_key,
        source_identity=f"public-route-probe:{spec.split}:{ordinal}",
        source_cluster_key=_key(spec.key_marker, 50, ordinal),
    )


def _remap_observation(
        observation: ObservationRecord,
        source: SourceRefRecord,
        ordinal: int,
        spec: W02MorphologyShadowRouteSpec,
        ) -> ObservationRecord:
    if (not isinstance(observation, ObservationRecord)
            or not isinstance(source, SourceRefRecord)
            or not isinstance(spec, W02MorphologyShadowRouteSpec)):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow observation identity drifted")
    return replace(
        observation,
        dataset_key=spec.dataset_key,
        artifact_key=spec.artifact_key,
        stable_key=_key(spec.key_marker, 20, ordinal),
        source_ref_key=source.stable_key,
        dedup_cluster_key=_key(spec.key_marker, 60, ordinal),
        content_group_key=_key(spec.key_marker, 61, ordinal),
        template_group_key=_key(spec.key_marker, 62, ordinal),
        shape_group_key=_key(spec.key_marker, 63, ordinal),
    )


def _prediction_projection(prediction: object) -> dict[str, object]:
    return {
        "boundary_lattice": list(prediction.boundary_lattice),
        "capabilities": list(prediction.capabilities),
        "generation": prediction.generation.to_dict(),
        "morphology_candidates": [
            row.to_dict() for row in prediction.morphology_candidates
        ],
        "status": prediction.status,
        "unicode_units": [row.to_dict() for row in prediction.unicode_units],
    }


def _dependency_state(repository: Path) -> dict[str, object]:
    parent = read_w02_compile_freeze(repository)
    dev_freeze = read_w02_morphology_successor_v3_dev_freeze(repository)
    dev_freeze_path = _repository_file(repository, W02_MORPH_V3_DEV_FREEZE_PATH)
    dev_report_path = _repository_file(repository, W02_MORPH_V3_DEV_REPORT_PATH)
    dev_report = read_canonical_object(dev_report_path)
    v2_shadow_path = _repository_file(
        repository, W02_MORPH_SUCCESSOR_V2_SHADOW_REPORT_PATH)
    v2_shadow = read_canonical_object(v2_shadow_path)
    validate_v2_safe_report(dev_report)
    validate_v2_safe_report(v2_shadow)
    if (dev_report.get("status") != "PASS"
            or dev_report.get("run_scope") != "FORMAL"
            or dev_report.get("run_id") != 1
            or dev_report.get("formal_dev_calibration_runs") != 1
            or dev_report.get("formal_private_evaluation_runs") != 0
            or dev_report.get("private_payload_reads") != 0
            or dev_report.get("teacher_calls") != 0
            or dev_report.get("code_freeze_sha256")
            != dev_freeze.get("code_freeze_sha256")
            or dev_report.get("freeze_file_sha256")
            != _sha256_file(dev_freeze_path)[1]):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow parent dev PASS drifted")
    if (v2_shadow.get("status") != "PASS"
            or v2_shadow.get("run_scope") != "FORMAL"
            or v2_shadow.get("formal_shadow_audit_runs") != 1
            or v2_shadow.get("formal_private_evaluation_runs") != 0
            or v2_shadow.get("private_payload_reads") != 0
            or v2_shadow.get("label_reads") != 0
            or v2_shadow.get("teacher_calls") != 0
            or v2_shadow.get("observation_reads")
            != W02_MORPH_V3_EXPECTED_OBSERVATION_COUNT
            or v2_shadow.get("full_route_observation_count")
            != W02_MORPH_V3_EXPECTED_ROUTED_COUNT):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow parent V2 shadow PASS drifted")
    shadow_files = [
        _shadow_identity(parent, key).to_dict() for key in W02_SHADOW_LAYOUTS
    ]
    return {
        "candidate_artifact_manifest_sha256":
            v2_shadow["candidate_artifact_manifest_sha256"],
        "candidate_semantic_sha256": v2_shadow["candidate_semantic_sha256"],
        "compile_freeze_sha256": parent.sha256(),
        "parent_public_files": _file_rows(
            repository, W02_MORPH_V3_SHADOW_PARENT_PUBLIC_PATHS),
        "shadow_input_commitment": _hash_value(shadow_files),
        "shadow_input_files": shadow_files,
        "v1_overlay_artifact_manifest_sha256":
            v2_shadow["v1_overlay_artifact_manifest_sha256"],
        "v1_overlay_semantic_sha256":
            v2_shadow["v1_overlay_semantic_sha256"],
        "v2_overlay_artifact_manifest_sha256":
            v2_shadow["v2_overlay_artifact_manifest_sha256"],
        "v2_overlay_semantic_sha256":
            v2_shadow["v2_overlay_semantic_sha256"],
        "v2_shadow_report_sha256": _sha256_file(v2_shadow_path)[1],
        "v3_dev_report_sha256": _sha256_file(dev_report_path)[1],
    }


def _development_identity(repository: Path) -> dict[str, object]:
    dependency = _dependency_state(repository)
    code_files = _file_rows(repository, W02_MORPH_V3_SHADOW_CODE_PATHS)
    return {
        **dependency,
        "code_files": code_files,
        "code_freeze_sha256": _hash_value(code_files),
        "resource_budget": {
            "max_logic_operations": W02_MORPH_V3_MAX_LOGIC_OPERATIONS,
            "max_records": 100_000,
        },
    }


def _artifact_indexes(
        identity: dict[str, object],
        candidate_root: Path,
        v1_root: Path,
        v2_root: Path,
        ) -> tuple[object, object, object]:
    candidate_result = read_w02_candidate_artifact(candidate_root)
    v1_result = read_w02_morphology_overlay_artifact(v1_root)
    v2_result = read_w02_morphology_successor_v2_overlay_artifact(v2_root)
    if (candidate_result.artifact_manifest_sha256
            != identity["candidate_artifact_manifest_sha256"]
            or candidate_result.candidate_semantic_sha256
            != identity["candidate_semantic_sha256"]
            or v1_result.artifact_manifest_sha256
            != identity["v1_overlay_artifact_manifest_sha256"]
            or v1_result.overlay_semantic_sha256
            != identity["v1_overlay_semantic_sha256"]
            or v2_result.artifact_manifest_sha256
            != identity["v2_overlay_artifact_manifest_sha256"]
            or v2_result.semantic_sha256
            != identity["v2_overlay_semantic_sha256"]):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow artifact identity drifted")
    return (
        candidate_result,
        load_w02_morphology_overlay_index(v1_root),
        load_w02_morphology_successor_v2_overlay_index(v2_root),
    )


def run_w02_morphology_successor_v3_shadow_preflight(
        repository_root: str | Path,
        shadow_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    """Compare every routed shadow row under disjoint train/dev identities."""
    repository = Path(repository_root).resolve()
    identity = _development_identity(repository)
    parent = read_w02_compile_freeze(repository)
    shadow = W02ShadowInputRoot(Path(shadow_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before = tuple(_tree_sha256(root) for root in (
        shadow.root, candidate_root, v1_root, v2_root))
    _candidate_result, v1_index, v2_index = _artifact_indexes(
        identity, candidate_root, v1_root, v2_root)
    if v1_index.dataset_keys != v2_index.dataset_keys:
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow V1/V2 parent route identity drifted")
    specs = _shadow_route_specs(repository)
    by_checksum = {row.upstream_checksum: row for row in specs}
    if len(by_checksum) != len(specs):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow upstream route identity is duplicated")
    source_digests = []
    source_map: dict[
        tuple[int, ...],
        tuple[SourceRefRecord, W02MorphologyShadowRouteSpec, int],
    ] = {}
    mapped_sources = []
    source_count_by_split = {"train": 0, "dev": 0}
    source_count = 0
    for source in iter_w02_shadow_records(parent, shadow, "SHADOW_SOURCE"):
        if not isinstance(source, SourceRefRecord):
            raise W02MorphologySuccessorV3ShadowProbeError(
                "V3 shadow SourceRef type drifted")
        source_count += 1
        source_digests.append(_hash_value(source.stable_key.to_list()))
        spec = by_checksum.get(source.upstream_checksum)
        if source.source_key != W02_MORPH_V3_PARENT_SOURCE_KEY:
            if spec is not None:
                raise W02MorphologySuccessorV3ShadowProbeError(
                    "V3 shadow capability matched an unrelated source")
            continue
        if spec is None:
            raise W02MorphologySuccessorV3ShadowProbeError(
                "V3 shadow UD source has an unauthorized upstream blob")
        source_count_by_split[spec.split] += 1
        ordinal = source_count_by_split[spec.split]
        mapped = _remap_source(source, ordinal, spec)
        source_map[source.stable_key.components] = (mapped, spec, ordinal)
        mapped_sources.append(mapped)
    if (source_count != W02_MORPH_V3_EXPECTED_SOURCE_COUNT
            or source_count_by_split != {
                "train": W02_MORPH_V3_EXPECTED_TRAIN_ROUTED_COUNT,
                "dev": W02_MORPH_V3_EXPECTED_DEV_ROUTED_COUNT,
            }
            or len(mapped_sources) != W02_MORPH_V3_EXPECTED_ROUTED_COUNT):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow source inventory drifted")
    routes = authorize_w02_morphology_source_routes(
        mapped_sources, tuple(row.capability for row in specs))
    indexes = build_w02_morphology_routed_indexes(v1_index, v2_index, routes)
    counts = {
        "base_logic_operations": 0,
        "blocked_v1_inference_logic_operations": 0,
        "blocked_v2_inference_logic_operations": 0,
        "candidate_exact_projection_count": 0,
        "candidate_identity_rebind_count": 0,
        "candidate_runtime_prediction_count": 0,
        "carrier_boundary_unicode_count": 0,
        "exact_final_projection_count": 0,
        "light_observation_count": 0,
        "max_v1_generalized_candidates_per_observation": 0,
        "max_v2_edge_candidates_per_observation": 0,
        "max_v2_edge_candidates_per_requested_span": 0,
        "observation_reads": 0,
        "old_route_zero_count": 0,
        "original_v1_candidate_count": 0,
        "original_v1_inference_logic_operations": 0,
        "original_v2_candidate_count": 0,
        "original_v2_inference_logic_operations": 0,
        "queried_span_count": 0,
        "route_authorized_count": 0,
        "route_logic_operations": routes.logic_operations,
        "routed_observation_count": 0,
        "routed_v1_candidate_count": 0,
        "routed_v1_inference_logic_operations": 0,
        "routed_v2_candidate_count": 0,
        "routed_v2_inference_logic_operations": 0,
        "source_reads": source_count,
        "v1_v2_bounded_retention_count": 0,
    }
    passed = {name: 0 for name in W02_MORPH_V3_SHADOW_GATES}
    evidence = {name: [] for name in W02_MORPH_V3_SHADOW_GATES}
    old_v1_cache = W02MorphologyRankingCache.empty()
    old_v2_cache = W02MorphologySuccessorV2Cache.empty()
    routed_v1_cache = W02MorphologyRankingCache.empty()
    routed_v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
        for layout_key in (
                "SHADOW_TRAIN_OBSERVATION", "SHADOW_DEV_OBSERVATION"):
            for observation in iter_w02_shadow_records(
                    parent, shadow, layout_key):
                if not isinstance(observation, ObservationRecord):
                    raise W02MorphologySuccessorV3ShadowProbeError(
                        "V3 shadow Observation type drifted")
                counts["observation_reads"] += 1
                mapped_identity = source_map.get(
                    observation.source_ref_key.components)
                if mapped_identity is None:
                    counts["light_observation_count"] += 1
                    continue
                mapped_source, spec, ordinal = mapped_identity
                mapped_observation = _remap_observation(
                    observation, mapped_source, ordinal, spec)
                old_base, old_base_operations = predict_w02_dev_observation(
                    candidate_index, observation)
                if mapped_observation.typed_payload != observation.typed_payload:
                    raise W02MorphologySuccessorV3ShadowProbeError(
                        "V3 shadow identity transfer changed the typed payload")
                mapped_base = replace(
                    old_base,
                    observation_key=mapped_observation.stable_key.components,
                )
                counts["base_logic_operations"] += (
                    old_base_operations + 8)
                counts["candidate_identity_rebind_count"] += 1
                counts["candidate_runtime_prediction_count"] += 1
                spans = _select_shadow_spans(old_base, v1_index.max_form_length)
                old_v1 = predict_w02_morphology_successor(
                    v1_index, observation, old_base, requested_spans=spans,
                    ranking_cache=old_v1_cache)
                old_v2 = predict_w02_morphology_successor_v2(
                    v2_index, observation, old_v1, requested_spans=spans,
                    cache=old_v2_cache)
                blocked_v1 = predict_w02_morphology_successor(
                    v1_index, mapped_observation, mapped_base,
                    requested_spans=spans)
                blocked_v2 = predict_w02_morphology_successor_v2(
                    v2_index, mapped_observation, blocked_v1,
                    requested_spans=spans)
                routed = predict_w02_morphology_successor_v3(
                    indexes, mapped_observation, mapped_base,
                    requested_spans=spans,
                    v1_cache=routed_v1_cache, v2_cache=routed_v2_cache)
                maximum = _assert_v1_retained(routed.v1, routed.v2, spans)
                old_projection = _prediction_projection(old_v2.prediction)
                routed_projection = _prediction_projection(routed.v2.prediction)
                old_base_projection = _prediction_projection(old_base)
                mapped_base_projection = _prediction_projection(mapped_base)
                old_carrier, old_boundary, _old_morph, old_digest = (
                    _audit_prediction(
                        observation, old_base, require_morphology=True))
                mapped_carrier, mapped_boundary, _mapped_morph, mapped_digest = (
                    _audit_prediction(
                        mapped_observation, mapped_base,
                        require_morphology=True))
                gate_values = {
                    W02_MORPH_V3_SHADOW_GATES[0]: routed.route_authorized == 1,
                    W02_MORPH_V3_SHADOW_GATES[1]: (
                        blocked_v1.generalized_candidate_count == 0
                        and blocked_v2.edge_candidate_count == 0),
                    W02_MORPH_V3_SHADOW_GATES[2]: (
                        old_base_projection == mapped_base_projection),
                    W02_MORPH_V3_SHADOW_GATES[3]: (
                        old_carrier and old_boundary
                        and mapped_carrier and mapped_boundary),
                    W02_MORPH_V3_SHADOW_GATES[4]: (
                        old_projection == routed_projection),
                    W02_MORPH_V3_SHADOW_GATES[5]: (
                        maximum <= W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN),
                }
                shared = {
                    "mapped_observation_key":
                        mapped_observation.stable_key.to_list(),
                    "observation_key": observation.stable_key.to_list(),
                    "requested_spans": [list(span) for span in spans],
                    "route_source_key": spec.source_key,
                }
                evidence_rows = {
                    W02_MORPH_V3_SHADOW_GATES[0]: {
                        **shared,
                        "route_authorized": routed.route_authorized,
                        "route_semantic_sha256": routes.semantic_sha256,
                    },
                    W02_MORPH_V3_SHADOW_GATES[1]: {
                        **shared,
                        "blocked_v1_candidates":
                            blocked_v1.generalized_candidate_count,
                        "blocked_v2_candidates": blocked_v2.edge_candidate_count,
                    },
                    W02_MORPH_V3_SHADOW_GATES[2]: {
                        **shared,
                        "mapped_base_sha256": _hash_value(mapped_base_projection),
                        "old_base_sha256": _hash_value(old_base_projection),
                    },
                    W02_MORPH_V3_SHADOW_GATES[3]: {
                        **shared,
                        "mapped_audit_sha256": mapped_digest,
                        "old_audit_sha256": old_digest,
                    },
                    W02_MORPH_V3_SHADOW_GATES[4]: {
                        **shared,
                        "old_final_sha256": _hash_value(old_projection),
                        "routed_final_sha256": _hash_value(routed_projection),
                    },
                    W02_MORPH_V3_SHADOW_GATES[5]: {
                        **shared,
                        "max_v2_edge_candidates_per_span": maximum,
                        "routed_v1_candidates":
                            routed.v1.generalized_candidate_count,
                        "routed_v2_candidates": routed.v2.edge_candidate_count,
                    },
                }
                for gate_name in W02_MORPH_V3_SHADOW_GATES:
                    passed[gate_name] += int(gate_values[gate_name])
                    evidence[gate_name].append(
                        _hash_value(evidence_rows[gate_name]))
                counts["candidate_exact_projection_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[2]])
                counts["carrier_boundary_unicode_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[3]])
                counts["exact_final_projection_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[4]])
                counts["old_route_zero_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[1]])
                counts["route_authorized_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[0]])
                counts["v1_v2_bounded_retention_count"] += int(
                    gate_values[W02_MORPH_V3_SHADOW_GATES[5]])
                counts["queried_span_count"] += len(spans)
                counts["routed_observation_count"] += 1
                counts["route_logic_operations"] += (
                    routed.route_logic_operations)
                counts["original_v1_candidate_count"] += (
                    old_v1.generalized_candidate_count)
                counts["original_v2_candidate_count"] += (
                    old_v2.edge_candidate_count)
                counts["routed_v1_candidate_count"] += (
                    routed.v1.generalized_candidate_count)
                counts["routed_v2_candidate_count"] += (
                    routed.v2.edge_candidate_count)
                counts["original_v1_inference_logic_operations"] += (
                    old_v1.logic_operations)
                counts["original_v2_inference_logic_operations"] += (
                    old_v2.logic_operations)
                counts["blocked_v1_inference_logic_operations"] += (
                    blocked_v1.logic_operations)
                counts["blocked_v2_inference_logic_operations"] += (
                    blocked_v2.logic_operations)
                counts["routed_v1_inference_logic_operations"] += (
                    routed.v1.logic_operations)
                counts["routed_v2_inference_logic_operations"] += (
                    routed.v2.logic_operations)
                counts["max_v1_generalized_candidates_per_observation"] = max(
                    counts["max_v1_generalized_candidates_per_observation"],
                    routed.v1.generalized_candidate_count)
                counts["max_v2_edge_candidates_per_observation"] = max(
                    counts["max_v2_edge_candidates_per_observation"],
                    routed.v2.edge_candidate_count)
                counts["max_v2_edge_candidates_per_requested_span"] = max(
                    counts["max_v2_edge_candidates_per_requested_span"],
                    maximum)
    finally:
        old_v1_cache.close()
        old_v2_cache.close()
        routed_v1_cache.close()
        routed_v2_cache.close()
    audit_results = [
        _gate(name, W02_MORPH_V3_EXPECTED_ROUTED_COUNT,
              passed[name], evidence[name])
        for name in W02_MORPH_V3_SHADOW_GATES
    ]
    logic_operations = (
        v1_index.logic_operations + v2_index.logic_operations
        + counts["source_reads"] + counts["observation_reads"]
        + counts["base_logic_operations"] + counts["route_logic_operations"]
        + counts["original_v1_inference_logic_operations"]
        + counts["original_v2_inference_logic_operations"]
        + counts["blocked_v1_inference_logic_operations"]
        + counts["blocked_v2_inference_logic_operations"]
        + counts["routed_v1_inference_logic_operations"]
        + counts["routed_v2_inference_logic_operations"])
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_SHADOW_REPORT"),
        "artifact_version": W02_MORPH_V3_SHADOW_REPORT_VERSION,
        "audit_results": audit_results,
        **counts,
        "code_freeze_sha256": identity["code_freeze_sha256"],
        "candidate_identity_policy": (
            "TYPED_PAYLOAD_PRESERVED_OBSERVATION_KEY_REBOUND"),
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 0,
        "formal_training_runs": 1,
        "label_reads": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": logic_operations,
        "next_action": "W02_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE",
        "parent_v1_semantic_sha256": v1_index.semantic_sha256,
        "parent_v2_semantic_sha256": v2_index.semantic_sha256,
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "route_capability_sha256s": [
            row.capability.sha256() for row in specs
        ],
        "route_semantic_sha256": routes.semantic_sha256,
        "routed_index_semantic_sha256": indexes.semantic_sha256,
        "run_id": 0,
        "run_scope": "DEVELOPMENT_PREFLIGHT",
        "shadow_input_commitment": identity["shadow_input_commitment"],
        "shadow_started": 0,
        "source_count": source_count,
        "source_identity_sha256": _hash_value(source_digests),
        "stage_key": "W-02",
        "status": (
            "PASS" if all(row["status"] == "PASS" for row in audit_results)
            and logic_operations <= W02_MORPH_V3_MAX_LOGIC_OPERATIONS
            else "FAIL"),
        "teacher_calls": 0,
        "train_routed_source_count": source_count_by_split["train"],
        "dev_routed_source_count": source_count_by_split["dev"],
        "transport_bytes_read": sum(
            _shadow_identity(parent, key).transport_size_bytes
            for key in W02_SHADOW_LAYOUTS),
        "v1_overlay_semantic_sha256": v1_index.semantic_sha256,
        "v2_overlay_semantic_sha256": v2_index.semantic_sha256,
        "zero_write_audit": {
            "candidate_writes": 0,
            "companion_writes": 0,
            "core_writes": 0,
            "evidence_writes": 0,
            "memory_writes": 0,
            "shadow_owner_writes": 0,
            "use_writes": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
        },
    }
    after = tuple(_tree_sha256(root) for root in (
        shadow.root, candidate_root, v1_root, v2_root))
    if after != before:
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow probe wrote to a frozen parent")
    _validate_preflight(report)
    validate_v2_safe_report(report)
    return report


def _validate_preflight(report: dict[str, object]) -> None:
    if (not isinstance(report, dict)
            or report.get("artifact_kind") != (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_SHADOW_REPORT")
            or report.get("artifact_version")
            != W02_MORPH_V3_SHADOW_REPORT_VERSION
            or report.get("status") != "PASS"
            or report.get("run_scope") != "DEVELOPMENT_PREFLIGHT"
            or report.get("run_id") != 0
            or report.get("source_count")
            != W02_MORPH_V3_EXPECTED_SOURCE_COUNT
            or report.get("observation_reads")
            != W02_MORPH_V3_EXPECTED_OBSERVATION_COUNT
            or report.get("routed_observation_count")
            != W02_MORPH_V3_EXPECTED_ROUTED_COUNT
            or report.get("candidate_identity_rebind_count")
            != W02_MORPH_V3_EXPECTED_ROUTED_COUNT
            or report.get("candidate_runtime_prediction_count")
            != W02_MORPH_V3_EXPECTED_ROUTED_COUNT
            or report.get("light_observation_count")
            != (W02_MORPH_V3_EXPECTED_OBSERVATION_COUNT
                - W02_MORPH_V3_EXPECTED_ROUTED_COUNT)
            or report.get("train_routed_source_count")
            != W02_MORPH_V3_EXPECTED_TRAIN_ROUTED_COUNT
            or report.get("dev_routed_source_count")
            != W02_MORPH_V3_EXPECTED_DEV_ROUTED_COUNT
            or report.get("private_payload_reads") != 0
            or report.get("label_reads") != 0
            or report.get("teacher_calls") != 0
            or type(report.get("logic_operations")) is not int
            or report["logic_operations"] > W02_MORPH_V3_MAX_LOGIC_OPERATIONS):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow preflight status, inventory, or resources drifted")
    audits = report.get("audit_results")
    zero_write = report.get("zero_write_audit")
    if (not isinstance(audits, list)
            or [row.get("gate_key") for row in audits
                if isinstance(row, dict)] != list(W02_MORPH_V3_SHADOW_GATES)
            or any(row.get("status") != "PASS" for row in audits)
            or any(row.get("numerator") != W02_MORPH_V3_EXPECTED_ROUTED_COUNT
                   or row.get("denominator")
                   != W02_MORPH_V3_EXPECTED_ROUTED_COUNT
                   for row in audits)
            or not isinstance(zero_write, dict)
            or any(value != 0 for value in zero_write.values())):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow gates or zero-write audit drifted")


W02_MORPH_V3_SHADOW_EXPECTED_COUNT_KEYS = (
    "base_logic_operations",
    "blocked_v1_inference_logic_operations",
    "blocked_v2_inference_logic_operations",
    "candidate_exact_projection_count",
    "candidate_identity_rebind_count",
    "candidate_runtime_prediction_count",
    "carrier_boundary_unicode_count",
    "dev_routed_source_count",
    "exact_final_projection_count",
    "light_observation_count",
    "logic_operations",
    "max_v1_generalized_candidates_per_observation",
    "max_v2_edge_candidates_per_observation",
    "max_v2_edge_candidates_per_requested_span",
    "observation_reads",
    "old_route_zero_count",
    "original_v1_candidate_count",
    "original_v1_inference_logic_operations",
    "original_v2_candidate_count",
    "original_v2_inference_logic_operations",
    "queried_span_count",
    "route_authorized_count",
    "route_logic_operations",
    "routed_observation_count",
    "routed_v1_candidate_count",
    "routed_v1_inference_logic_operations",
    "routed_v2_candidate_count",
    "routed_v2_inference_logic_operations",
    "source_count",
    "source_reads",
    "train_routed_source_count",
    "transport_bytes_read",
    "v1_v2_bounded_retention_count",
)


def build_w02_morphology_successor_v3_shadow_freeze(
        repository_root: str | Path,
        preflight: dict[str, object],
        ) -> dict[str, object]:
    """Build the public freeze for one formal V3 shadow identity transfer."""
    repository = Path(repository_root).resolve()
    _validate_preflight(preflight)
    identity = _development_identity(repository)
    if preflight.get("code_freeze_sha256") != identity["code_freeze_sha256"]:
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow preflight live code identity drifted")
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE"),
        "artifact_version": W02_MORPH_V3_SHADOW_FREEZE_VERSION,
        **identity,
        "expected_audit_results": preflight["audit_results"],
        "expected_counts": {
            key: preflight[key]
            for key in W02_MORPH_V3_SHADOW_EXPECTED_COUNT_KEYS
        },
        "expected_preflight_sha256": _hash_value(preflight),
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_SUCCESSOR_V3_ROUTE_SHADOW_PROBE",
        "private_family_registered": 0,
        "private_payload_reads": 0,
        "route_capability_sha256s": preflight["route_capability_sha256s"],
        "route_semantic_sha256": preflight["route_semantic_sha256"],
        "routed_index_semantic_sha256":
            preflight["routed_index_semantic_sha256"],
        "run_id": 1,
        "stage_key": "W-02",
        "status": "W02_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE_COMPLETE",
        "teacher_calls": 0,
    }


def publish_w02_morphology_successor_v3_shadow_freeze(
        repository_root: str | Path,
        preflight: dict[str, object],
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v3_shadow_freeze(
        repository, preflight)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_V3_SHADOW_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_morphology_successor_v3_shadow_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    repository = Path(repository_root).resolve()
    freeze = read_canonical_object(
        _repository_file(repository, W02_MORPH_V3_SHADOW_FREEZE_PATH))
    if (freeze.get("artifact_kind") != (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE")
            or freeze.get("artifact_version")
            != W02_MORPH_V3_SHADOW_FREEZE_VERSION
            or freeze.get("status")
            != "W02_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE_COMPLETE"
            or freeze.get("run_id") != 1
            or freeze.get("formal_shadow_audit_runs") != 0
            or freeze.get("formal_private_evaluation_runs") != 0
            or freeze.get("private_payload_reads") != 0):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow freeze status drifted")
    identity = _development_identity(repository)
    for key, value in identity.items():
        if freeze.get(key) != value:
            raise W02MorphologySuccessorV3ShadowProbeError(
                f"V3 shadow freeze live identity drifted: {key}")
    for name in (
            "expected_preflight_sha256", "route_semantic_sha256",
            "routed_index_semantic_sha256"):
        value = freeze.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise W02MorphologySuccessorV3ShadowProbeError(
                f"V3 shadow freeze {name} is not a SHA-256")
    return freeze


def assert_w02_morphology_successor_v3_shadow_preflight(
        preflight: dict[str, object],
        freeze: dict[str, object],
        ) -> None:
    _validate_preflight(preflight)
    if (_hash_value(preflight) != freeze.get("expected_preflight_sha256")
            or preflight.get("audit_results")
            != freeze.get("expected_audit_results")
            or preflight.get("route_capability_sha256s")
            != freeze.get("route_capability_sha256s")
            or preflight.get("route_semantic_sha256")
            != freeze.get("route_semantic_sha256")
            or preflight.get("routed_index_semantic_sha256")
            != freeze.get("routed_index_semantic_sha256")
            or any(preflight.get(key) != value for key, value in (
                freeze.get("expected_counts") or {}).items())):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow formal result differs from the frozen preflight")


def publish_w02_morphology_successor_v3_shadow_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze = read_w02_morphology_successor_v3_shadow_freeze(repository)
    freeze_path = _repository_file(repository, W02_MORPH_V3_SHADOW_FREEZE_PATH)
    if (value.get("artifact_version")
            != W02_MORPH_V3_SHADOW_REPORT_VERSION
            or value.get("status") != "PASS"
            or value.get("run_scope") != "FORMAL"
            or value.get("run_id") != 1
            or value.get("formal_shadow_audit_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("label_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256")
            != freeze.get("code_freeze_sha256")
            or value.get("freeze_file_sha256")
            != _sha256_file(freeze_path)[1]
            or value.get("audit_results")
            != freeze.get("expected_audit_results")
            or any(value.get(key) != expected for key, expected in (
                freeze.get("expected_counts") or {}).items())):
        raise W02MorphologySuccessorV3ShadowProbeError(
            "V3 shadow formal report does not match its freeze")
    target = repository / Path(*PurePosixPath(
        W02_MORPH_V3_SHADOW_REPORT_PATH).parts)
    write_immutable_json(value, target)
    return target


__all__ = [
    "W02_MORPH_V3_SHADOW_CODE_PATHS",
    "W02_MORPH_V3_SHADOW_FREEZE_PATH",
    "W02_MORPH_V3_SHADOW_REPORT_PATH",
    "W02MorphologyShadowRouteSpec",
    "W02MorphologySuccessorV3ShadowProbeError",
    "assert_w02_morphology_successor_v3_shadow_preflight",
    "build_w02_morphology_successor_v3_shadow_freeze",
    "publish_w02_morphology_successor_v3_shadow_freeze",
    "publish_w02_morphology_successor_v3_shadow_report",
    "read_w02_morphology_successor_v3_shadow_freeze",
    "run_w02_morphology_successor_v3_shadow_preflight",
]
