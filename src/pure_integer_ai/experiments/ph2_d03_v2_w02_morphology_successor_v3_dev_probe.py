"""Public dev metamorphic gate for W-02 morphology source routing."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    W02DevInputRoot,
    _dimension_key,
    _dimension_report,
    _evaluate_pair,
    iter_w02_dev_pairs,
    iter_w02_dev_records,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
    predict_w02_morphology_successor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Cache,
    predict_w02_morphology_successor_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_dev_calibration import (
    _assert_v1_retained,
    _requested_spans,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
)


W02_MORPH_V3_DEV_PROBE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-ROUTE-DEV-PROBE-V1"
)
W02_MORPH_V3_DEV_SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18_ROUTE_PROBE"
W02_MORPH_V3_DEV_PARENT_SOURCE_KEY = "UD_ZH_GSDSIMP_R2_18"
W02_MORPH_V3_DEV_DATASET_KEY = StableRecordKey((2, 2, 200, 1))
W02_MORPH_V3_DEV_ARTIFACT_KEY = StableRecordKey((2, 2, 200, 2))
W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT = 500
W02_MORPH_V3_DEV_MAX_LOGIC_OPERATIONS = 9_000_000
W02_MORPH_V3_DEV_SNAPSHOT_PATH = (
    "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json"
)
W02_MORPH_V3_DEV_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-ROUTE-DEV-FREEZE-V1"
)
W02_MORPH_V3_DEV_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_route_dev_freeze_v1.json"
)
W02_MORPH_V3_PARENT_FAILED_AGGREGATE_SHA256 = (
    "d61e836e5a757925f178bf0600f018f66997403ca7a6bc94f69bb972ce3de3f6"
)
W02_MORPH_V3_PARENT_FAILURE_SEAL_SHA256 = (
    "9f091f949fdd18ce5905754b1e02e93fd252675f0b90ca58dd9247da93c8d67b"
)
W02_MORPH_V3_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_route.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v3_dev_probe.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v3_dev_probe.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v3_route.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v3_dev_probe.py",
)
W02_MORPH_V3_PARENT_PUBLIC_PATHS = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_compile_freeze_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_candidate_artifact_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_artifact_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_overlay_artifact_v1.json",
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_private_family_freeze_v1.json",
)


# object-model: exception
class W02MorphologySuccessorV3DevProbeError(RuntimeError):
    """The public identity-transfer dev probe or a frozen parent drifted."""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tree_sha256(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            payload = path.read_bytes()
            rows.append((
                path.relative_to(root).as_posix(),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ))
    return _hash_value(rows)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe repository path is invalid")
    current = repository
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02MorphologySuccessorV3DevProbeError(
                "route probe repository path crosses a symlink")
    target = (repository / Path(*pure.parts)).resolve()
    if not target.is_relative_to(repository) or not target.is_file():
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe repository file is missing")
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


def _key(kind: int, ordinal: int) -> StableRecordKey:
    if type(kind) is not int or kind <= 0 or type(ordinal) is not int or ordinal <= 0:
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe stable key component is invalid")
    return StableRecordKey((2, 2, 200, kind, 2, ordinal))


def _remap_source(source: SourceRefRecord, ordinal: int) -> SourceRefRecord:
    if (not isinstance(source, SourceRefRecord)
            or source.source_key != W02_MORPH_V3_DEV_PARENT_SOURCE_KEY):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe parent SourceRef drifted")
    return replace(
        source,
        dataset_key=W02_MORPH_V3_DEV_DATASET_KEY,
        artifact_key=W02_MORPH_V3_DEV_ARTIFACT_KEY,
        stable_key=_key(10, ordinal),
        source_key=W02_MORPH_V3_DEV_SOURCE_KEY,
        source_identity=f"public-route-probe:sentence:{ordinal}",
        source_cluster_key=_key(50, ordinal),
    )


def _remap_pair(
        observation: ObservationRecord,
        evaluation: EvaluatorLabelRecord,
        source: SourceRefRecord,
        ordinal: int,
        ) -> tuple[ObservationRecord, EvaluatorLabelRecord]:
    if (not isinstance(observation, ObservationRecord)
            or not isinstance(evaluation, EvaluatorLabelRecord)
            or evaluation.observation_key != observation.stable_key):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe dev pair identity drifted")
    mapped_observation = replace(
        observation,
        dataset_key=W02_MORPH_V3_DEV_DATASET_KEY,
        artifact_key=W02_MORPH_V3_DEV_ARTIFACT_KEY,
        stable_key=_key(20, ordinal),
        source_ref_key=source.stable_key,
        dedup_cluster_key=_key(60, ordinal),
        content_group_key=_key(61, ordinal),
        template_group_key=_key(62, ordinal),
        shape_group_key=_key(63, ordinal),
    )
    mapped_evaluation = replace(
        evaluation,
        dataset_key=W02_MORPH_V3_DEV_DATASET_KEY,
        artifact_key=W02_MORPH_V3_DEV_ARTIFACT_KEY,
        stable_key=_key(40, ordinal),
        observation_key=mapped_observation.stable_key,
    )
    return mapped_observation, mapped_evaluation


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


def _dev_source_capability(repository: Path, source: SourceRefRecord):
    snapshot = read_canonical_object(repository / W02_MORPH_V3_DEV_SNAPSHOT_PATH)
    if (snapshot.get("source_key") != W02_MORPH_V3_DEV_PARENT_SOURCE_KEY
            or snapshot.get("repository_url") != source.official_url
            or snapshot.get("commit_sha1") != source.revision_id
            or snapshot.get("license_id") != source.license_id):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe public snapshot provenance drifted")
    files = snapshot.get("files")
    if not isinstance(files, list):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe public snapshot files drifted")
    dev_files = [
        row for row in files
        if isinstance(row, dict)
        and row.get("file_kind") == "conllu"
        and row.get("split") == "dev"
    ]
    if (len(dev_files) != 1
            or "sha1:" + str(dev_files[0].get("git_blob_sha1"))
            != source.upstream_checksum):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe public dev blob identity drifted")
    return w02_ud_morphology_source_capability({
        "annotation_provenance": (
            "Universal Dependencies documented treebank annotation"),
        "commit_sha1": source.revision_id,
        "data_file": {"git_blob_sha1": dev_files[0]["git_blob_sha1"]},
        "language": "zh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": W02_MORPH_V3_DEV_SOURCE_KEY,
    })


def run_w02_morphology_successor_v3_dev_preflight(
        repository_root: str | Path,
        dev_root: str | Path,
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    """Run all public UD dev rows under a disjoint synthetic source identity."""
    repository = Path(repository_root).resolve()
    dev = W02DevInputRoot(Path(dev_root))
    candidate_root = Path(candidate_artifact_root).resolve()
    v1_root = Path(v1_overlay_artifact_root).resolve()
    v2_root = Path(v2_overlay_artifact_root).resolve()
    before = tuple(_tree_sha256(root) for root in (
        dev.root, candidate_root, v1_root, v2_root))
    parent = read_w02_compile_freeze(repository)
    old_sources = tuple(
        row for row in iter_w02_dev_records(parent, dev, "DEV_SOURCE")
        if isinstance(row, SourceRefRecord)
        and row.source_key == W02_MORPH_V3_DEV_PARENT_SOURCE_KEY
    )
    if len(old_sources) != W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT:
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe public dev source count drifted")
    mapped_sources = tuple(
        _remap_source(source, ordinal)
        for ordinal, source in enumerate(old_sources, start=1)
    )
    source_map = {
        old.stable_key.components: mapped
        for old, mapped in zip(old_sources, mapped_sources, strict=True)
    }
    capability = _dev_source_capability(repository, old_sources[0])
    routes = authorize_w02_morphology_source_routes(
        mapped_sources, (capability,))
    v1_index = load_w02_morphology_overlay_index(v1_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(v2_root)
    indexes = build_w02_morphology_routed_indexes(v1_index, v2_index, routes)
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    dimension_rows: dict[str, list[tuple[bool | None, str]]] = {
        name: [] for name in W02_DEV_DIMENSIONS
    }
    counts = {
        "base_logic_operations": 0,
        "exact_prediction_projection_count": 0,
        "observation_count": 0,
        "old_route_zero_count": 0,
        "route_logic_operations": routes.logic_operations,
        "v1_candidate_count": 0,
        "v1_inference_logic_operations": 0,
        "v2_candidate_count": 0,
        "v2_inference_logic_operations": 0,
    }
    max_v1 = 0
    max_v2 = 0
    max_v2_per_span = 0
    old_v1_cache = W02MorphologyRankingCache.empty()
    old_v2_cache = W02MorphologySuccessorV2Cache.empty()
    routed_v1_cache = W02MorphologyRankingCache.empty()
    routed_v2_cache = W02MorphologySuccessorV2Cache.empty()
    try:
        with open_w02_candidate_predictor(candidate_root) as predictor:
            candidate_index = load_w02_dev_candidate_index(predictor)
        for observation, evaluation in iter_w02_dev_pairs(parent, dev):
            mapped_source = source_map.get(observation.source_ref_key.components)
            if mapped_source is None:
                continue
            ordinal = counts["observation_count"] + 1
            mapped_observation, mapped_evaluation = _remap_pair(
                observation, evaluation, mapped_source, ordinal)
            old_base, old_base_operations = predict_w02_dev_observation(
                candidate_index, observation)
            mapped_base, mapped_base_operations = predict_w02_dev_observation(
                candidate_index, mapped_observation)
            spans = _requested_spans(evaluation)
            old_v1 = predict_w02_morphology_successor(
                v1_index, observation, old_base,
                requested_spans=spans, ranking_cache=old_v1_cache)
            old_v2 = predict_w02_morphology_successor_v2(
                v2_index, observation, old_v1,
                requested_spans=spans, cache=old_v2_cache)
            blocked_v1 = predict_w02_morphology_successor(
                v1_index, mapped_observation, mapped_base,
                requested_spans=spans)
            blocked_v2 = predict_w02_morphology_successor_v2(
                v2_index, mapped_observation, blocked_v1,
                requested_spans=spans)
            if (blocked_v1.generalized_candidate_count == 0
                    and blocked_v2.edge_candidate_count == 0):
                counts["old_route_zero_count"] += 1
            routed = predict_w02_morphology_successor_v3(
                indexes, mapped_observation, mapped_base,
                requested_spans=spans,
                v1_cache=routed_v1_cache,
                v2_cache=routed_v2_cache)
            per_span = _assert_v1_retained(routed.v1, routed.v2, spans)
            if (_prediction_projection(old_v2.prediction)
                    == _prediction_projection(routed.v2.prediction)):
                counts["exact_prediction_projection_count"] += 1
            dimension, _family, passed, evidence_sha = _evaluate_pair(
                mapped_observation, mapped_evaluation,
                routed.v2.prediction, dimension_by_key)
            dimension_rows[dimension].append((passed, evidence_sha))
            counts["observation_count"] += 1
            counts["base_logic_operations"] += (
                old_base_operations + mapped_base_operations + 16)
            counts["route_logic_operations"] += routed.route_logic_operations
            counts["v1_candidate_count"] += routed.v1.generalized_candidate_count
            counts["v2_candidate_count"] += routed.v2.edge_candidate_count
            counts["v1_inference_logic_operations"] += (
                old_v1.logic_operations + routed.v1.logic_operations)
            counts["v2_inference_logic_operations"] += (
                old_v2.logic_operations + routed.v2.logic_operations)
            max_v1 = max(max_v1, routed.v1.generalized_candidate_count)
            max_v2 = max(max_v2, routed.v2.edge_candidate_count)
            max_v2_per_span = max(max_v2_per_span, per_span)
    finally:
        old_v1_cache.close()
        old_v2_cache.close()
        routed_v1_cache.close()
        routed_v2_cache.close()
    dimensions = [
        _dimension_report(name, dimension_rows[name])
        for name in W02_DEV_DIMENSIONS
    ]
    observation_count = counts["observation_count"]
    logic_operations = (
        v1_index.logic_operations + v2_index.logic_operations
        + counts["base_logic_operations"]
        + counts["route_logic_operations"]
        + counts["v1_inference_logic_operations"]
        + counts["v2_inference_logic_operations"]
    )
    gates = {
        "all_dimensions_pass": int(all(
            row["status"] == "PASS" for row in dimensions)),
        "exact_prediction_projection": int(
            counts["exact_prediction_projection_count"] == observation_count),
        "old_route_was_zero": int(
            counts["old_route_zero_count"] == observation_count),
        "route_authorized_all_sources": int(
            routes.source_count == len(mapped_sources)),
        "resource_within_budget": int(
            logic_operations <= W02_MORPH_V3_DEV_MAX_LOGIC_OPERATIONS),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_DEV_REPORT"),
        "artifact_version": W02_MORPH_V3_DEV_PROBE_VERSION,
        **counts,
        "dimension_results": dimensions,
        "formal_dev_calibration_runs": 0,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1,
        "gates": gates,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": logic_operations,
        "max_v1_generalized_candidates_per_observation": max_v1,
        "max_v2_edge_candidates_per_observation": max_v2,
        "max_v2_edge_candidates_per_requested_span": max_v2_per_span,
        "next_action": (
            "W02_SUCCESSOR_V3_ROUTE_DEV_FREEZE"
            if status == "PASS" else "W02_SUCCESSOR_V3_ROUTE_DEV_FAILED_STOP"),
        "parent_v1_semantic_sha256": v1_index.semantic_sha256,
        "parent_v2_semantic_sha256": v2_index.semantic_sha256,
        "private_payload_reads": 0,
        "route_capability_sha256": capability.sha256(),
        "route_semantic_sha256": routes.semantic_sha256,
        "routed_index_semantic_sha256": indexes.semantic_sha256,
        "run_scope": "DEVELOPMENT_PREFLIGHT",
        "source_count": len(mapped_sources),
        "stage_key": "W-02",
        "status": status,
        "teacher_calls": 0,
        "zero_write_audit": {
            "candidate_writes": 0,
            "core_writes": 0,
            "dev_owner_writes": 0,
            "memory_writes": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
        },
    }
    after = tuple(_tree_sha256(root) for root in (
        dev.root, candidate_root, v1_root, v2_root))
    if after != before:
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe wrote to a frozen parent")
    return report


def _validate_preflight(report: dict[str, object]) -> None:
    if (not isinstance(report, dict)
            or report.get("artifact_kind") != (
                "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_DEV_REPORT")
            or report.get("artifact_version") != W02_MORPH_V3_DEV_PROBE_VERSION
            or report.get("status") != "PASS"
            or report.get("run_scope") != "DEVELOPMENT_PREFLIGHT"
            or report.get("source_count") != W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT
            or report.get("observation_count")
            != W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT
            or report.get("old_route_zero_count")
            != W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT
            or report.get("exact_prediction_projection_count")
            != W02_MORPH_V3_DEV_EXPECTED_SOURCE_COUNT
            or report.get("private_payload_reads") != 0
            or report.get("teacher_calls") != 0):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe preflight status or exact counts drifted")
    dimensions = report.get("dimension_results")
    gates = report.get("gates")
    if (not isinstance(dimensions, list)
            or len(dimensions) != len(W02_DEV_DIMENSIONS)
            or [row.get("dimension_key") for row in dimensions
                if isinstance(row, dict)] != list(W02_DEV_DIMENSIONS)
            or any(row.get("status") != "PASS" for row in dimensions)
            or not isinstance(gates, dict)
            or not gates
            or any(value != 1 for value in gates.values())
            or type(report.get("logic_operations")) is not int
            or report["logic_operations"] > W02_MORPH_V3_DEV_MAX_LOGIC_OPERATIONS):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe preflight dimensions, gates, or resources drifted")


def build_w02_morphology_successor_v3_dev_freeze(
        repository_root: str | Path,
        preflight: dict[str, object],
        ) -> dict[str, object]:
    """Build the public metadata-only freeze for one formal dev route run."""
    repository = Path(repository_root).resolve()
    _validate_preflight(preflight)
    code_files = _file_rows(repository, W02_MORPH_V3_CODE_PATHS)
    parent_files = _file_rows(repository, W02_MORPH_V3_PARENT_PUBLIC_PATHS)
    expected_counts = {
        key: preflight[key] for key in (
            "base_logic_operations",
            "exact_prediction_projection_count",
            "logic_operations",
            "max_v1_generalized_candidates_per_observation",
            "max_v2_edge_candidates_per_observation",
            "max_v2_edge_candidates_per_requested_span",
            "observation_count",
            "old_route_zero_count",
            "route_logic_operations",
            "source_count",
            "v1_candidate_count",
            "v1_inference_logic_operations",
            "v2_candidate_count",
            "v2_inference_logic_operations",
        )
    }
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_DEV_FREEZE"),
        "artifact_version": W02_MORPH_V3_DEV_FREEZE_VERSION,
        "code_files": code_files,
        "code_freeze_sha256": _hash_value(code_files),
        "expected_counts": expected_counts,
        "expected_dimensions": preflight["dimension_results"],
        "expected_gates": preflight["gates"],
        "expected_preflight_sha256": _hash_value(preflight),
        "formal_dev_calibration_runs": 0,
        "formal_private_evaluation_runs": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_SUCCESSOR_V3_ROUTE_DEV_PROBE",
        "parent_failed_aggregate_sha256": (
            W02_MORPH_V3_PARENT_FAILED_AGGREGATE_SHA256),
        "parent_failure_seal_sha256": W02_MORPH_V3_PARENT_FAILURE_SEAL_SHA256,
        "parent_public_files": parent_files,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "route_capability_sha256": preflight["route_capability_sha256"],
        "route_semantic_sha256": preflight["route_semantic_sha256"],
        "routed_index_semantic_sha256": (
            preflight["routed_index_semantic_sha256"]),
        "run_id": 1,
        "stage_key": "W-02",
        "status": "W02_SUCCESSOR_V3_ROUTE_DEV_FREEZE_COMPLETE",
        "teacher_calls": 0,
    }


def read_w02_morphology_successor_v3_dev_freeze(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """Read the canonical freeze and verify every live public dependency."""
    repository = Path(repository_root).resolve()
    freeze = read_canonical_object(
        _repository_file(repository, W02_MORPH_V3_DEV_FREEZE_PATH))
    if (freeze.get("artifact_kind") != (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_DEV_FREEZE")
            or freeze.get("artifact_version") != W02_MORPH_V3_DEV_FREEZE_VERSION
            or freeze.get("status")
            != "W02_SUCCESSOR_V3_ROUTE_DEV_FREEZE_COMPLETE"
            or freeze.get("run_id") != 1
            or freeze.get("private_payload_reads") != 0
            or freeze.get("parent_failed_aggregate_sha256")
            != W02_MORPH_V3_PARENT_FAILED_AGGREGATE_SHA256
            or freeze.get("parent_failure_seal_sha256")
            != W02_MORPH_V3_PARENT_FAILURE_SEAL_SHA256):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe freeze status or failed-parent binding drifted")
    code_files = _file_rows(repository, W02_MORPH_V3_CODE_PATHS)
    parent_files = _file_rows(repository, W02_MORPH_V3_PARENT_PUBLIC_PATHS)
    if (freeze.get("code_files") != code_files
            or freeze.get("code_freeze_sha256") != _hash_value(code_files)
            or freeze.get("parent_public_files") != parent_files):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe freeze live code or parent identity drifted")
    for name in (
            "expected_preflight_sha256", "route_capability_sha256",
            "route_semantic_sha256", "routed_index_semantic_sha256"):
        value = freeze.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise W02MorphologySuccessorV3DevProbeError(
                f"route probe freeze {name} is not a SHA-256")
    return freeze


def assert_w02_morphology_successor_v3_dev_preflight(
        preflight: dict[str, object],
        freeze: dict[str, object],
        ) -> None:
    """Require a formal rerun to equal the last pre-freeze development state."""
    _validate_preflight(preflight)
    if (_hash_value(preflight) != freeze.get("expected_preflight_sha256")
            or preflight.get("dimension_results")
            != freeze.get("expected_dimensions")
            or preflight.get("gates") != freeze.get("expected_gates")
            or any(preflight.get(key) != value for key, value in (
                freeze.get("expected_counts") or {}).items())):
        raise W02MorphologySuccessorV3DevProbeError(
            "route probe formal result differs from the frozen preflight")


__all__ = [
    "W02_MORPH_V3_CODE_PATHS",
    "W02_MORPH_V3_DEV_FREEZE_PATH",
    "W02_MORPH_V3_DEV_FREEZE_VERSION",
    "W02_MORPH_V3_DEV_PROBE_VERSION",
    "W02MorphologySuccessorV3DevProbeError",
    "assert_w02_morphology_successor_v3_dev_preflight",
    "build_w02_morphology_successor_v3_dev_freeze",
    "read_w02_morphology_successor_v3_dev_freeze",
    "run_w02_morphology_successor_v3_dev_preflight",
]
