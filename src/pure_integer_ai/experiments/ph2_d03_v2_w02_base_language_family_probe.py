"""Public metamorphic probe for the W-02 lzh base-language adapter."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_base_language_family_adapter import (
    W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION,
    adapt_w02_observation_for_base_candidate,
    predict_w02_dev_observation_language_family,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import (
    _observation_record,
    _source_record,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _hash_value,
    load_w02_dev_candidate_index,
    predict_w02_dev_observation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02MorphologyRankingCache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02MorphologySuccessorV2Cache,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    load_w02_morphology_successor_v2_overlay_index,
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    authorize_w02_morphology_source_routes,
    build_w02_morphology_routed_indexes,
    predict_w02_morphology_successor_v3,
    w02_ud_morphology_source_capability,
)


W02_BASE_LANGUAGE_FAMILY_PROBE_VERSION = (
    "PH2-D03-V2-W02-BASE-LANGUAGE-FAMILY-PROBE-V1"
)
W02_BASE_LANGUAGE_FAMILY_PROBE_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_base_language_family_probe_report_v1.json"
)


# object-model: exception
class W02BaseLanguageFamilyProbeError(RuntimeError):
    """The public lzh/zh metamorphic probe or artifact identity drifted."""


def _public_fixture():
    surface = "新化"
    parent = _source_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1,
        snapshot_id="public-language-family-parent",
        revision_id="public-parent-revision",
        official_url=(
            "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp"),
        source_identity="public-language-family-parent:1",
        upstream_checksum="sha256:" + "1" * 64,
        local_sha256="2" * 64,
        license_id="CC-BY-SA-4.0",
        attribution="public synthetic language-family fixture",
        locator_kind="record", locator_value="1", span_end=len(surface),
    )
    dataset_key = StableRecordKey((9, 9, 8, 1))
    source_key = StableRecordKey((9, 9, 8, 10, 1))
    source = replace(
        parent,
        dataset_key=dataset_key,
        artifact_key=StableRecordKey((9, 9, 8, 2)),
        stable_key=source_key,
        source_cluster_key=StableRecordKey((9, 9, 8, 50, 1)),
        source_key="UD_LZH_PUBLIC_LANGUAGE_FAMILY_FIXTURE",
        snapshot_id="ud-lzh-public-language-family-r1",
        revision_id="public-lzh-route-revision",
        official_url=(
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-PublicFixture"),
        source_identity="public-lzh-language-family:sentence:1",
    )
    parent_observation = _observation_record(
        "UD_ZH_GSDSIMP_R2_18", "held_out", 1, parent,
        carrier_kind="plain_text", surface=surface, family_ordinal=1,
        sample_role="read_only_probe", perturbation_kind="HELD_OUT_DOCUMENT",
    )
    zh_observation = replace(
        parent_observation,
        dataset_key=dataset_key,
        artifact_key=source.artifact_key,
        stable_key=StableRecordKey((9, 9, 8, 20, 1)),
        source_ref_key=source_key,
    )
    lzh_observation = replace(zh_observation, language="lzh")
    capability = w02_ud_morphology_source_capability({
        "annotation_provenance": "public synthetic manual UD annotation",
        "commit_sha1": source.revision_id,
        "language": "lzh",
        "license_id": source.license_id,
        "repository_url": source.official_url,
        "snapshot_id": source.snapshot_id,
        "source_key": source.source_key,
        "upstream_checksum": source.upstream_checksum,
    })
    return source, zh_observation, lzh_observation, capability


def run_w02_base_language_family_probe(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> dict[str, object]:
    """Prove base equivalence while retaining original lzh route identity."""
    candidate = read_w02_candidate_artifact(candidate_artifact_root)
    v1 = read_w02_morphology_overlay_artifact(v1_overlay_artifact_root)
    v2 = read_w02_morphology_successor_v2_overlay_artifact(
        v2_overlay_artifact_root)
    source, zh_observation, lzh_observation, capability = _public_fixture()
    original_before = lzh_observation.to_dict()
    adapted = adapt_w02_observation_for_base_candidate(lzh_observation)
    if (adapted.language != "zh"
            or lzh_observation.to_dict() != original_before
            or any(adapted.to_dict()[key] != original_before[key]
                   for key in original_before if key != "language")):
        raise W02BaseLanguageFamilyProbeError(
            "base language adapter did not preserve public fixture identity")
    with open_w02_candidate_predictor(candidate_artifact_root) as predictor:
        index = load_w02_dev_candidate_index(predictor)
        baseline, baseline_operations = predict_w02_dev_observation(
            index, zh_observation)
        adapted_prediction, adapted_operations = (
            predict_w02_dev_observation_language_family(
                index, lzh_observation))
    if (baseline.to_dict() != adapted_prediction.to_dict()
            or baseline_operations != adapted_operations):
        raise W02BaseLanguageFamilyProbeError(
            "lzh adapter changed the unchanged base Candidate prediction")

    routes = authorize_w02_morphology_source_routes(
        (source,), (capability,))
    if not routes.permits(lzh_observation) or routes.permits(adapted):
        raise W02BaseLanguageFamilyProbeError(
            "language-family route did not retain original lzh identity")
    v1_index = load_w02_morphology_overlay_index(v1_overlay_artifact_root)
    v2_index = load_w02_morphology_successor_v2_overlay_index(
        v2_overlay_artifact_root)
    routed = build_w02_morphology_routed_indexes(v1_index, v2_index, routes)
    prediction = predict_w02_morphology_successor_v3(
        routed,
        lzh_observation,
        adapted_prediction,
        requested_spans=((0, 2),),
        v1_cache=W02MorphologyRankingCache.empty(),
        v2_cache=W02MorphologySuccessorV2Cache.empty(),
    )
    if prediction.route_authorized != 1:
        raise W02BaseLanguageFamilyProbeError(
            "original lzh observation was not route-authorized")

    report = {
        "adapter_changes_only_language": 1,
        "adapter_version": W02_BASE_LANGUAGE_FAMILY_ADAPTER_VERSION,
        "artifact_kind": "PH2_D03_V2_W02_BASE_LANGUAGE_FAMILY_PROBE_REPORT",
        "artifact_version": W02_BASE_LANGUAGE_FAMILY_PROBE_VERSION,
        "base_prediction_metamorphic_equal": 1,
        "base_prediction_sha256": _hash_value(baseline.to_dict()),
        "base_scope_language": "zh",
        "candidate_artifact_manifest_sha256":
            candidate.artifact_manifest_sha256,
        "candidate_semantic_sha256": candidate.candidate_semantic_sha256,
        "formal_private_evaluation_runs": 0,
        "logic_operations": baseline_operations + prediction.route_logic_operations,
        "next_action": "AUTHORIZE_UNUSED_KYOTO_R5_OWNER",
        "original_observation_unchanged": 1,
        "route_adapted_zh_authorized": 0,
        "route_original_lzh_authorized": 1,
        "route_semantic_sha256": prediction.route_semantic_sha256,
        "source_language": "lzh",
        "status": "PASS",
        "teacher_calls": 0,
        "v1_overlay_artifact_manifest_sha256": v1.artifact_manifest_sha256,
        "v1_overlay_semantic_sha256": v1.overlay_semantic_sha256,
        "v2_overlay_artifact_manifest_sha256": v2.artifact_manifest_sha256,
        "v2_overlay_semantic_sha256": v2.semantic_sha256,
        "zero_write_audit": {
            "candidate_writes": 0,
            "host_writes": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
            "v3_route_writes": 0,
        },
    }
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_BASE_LANGUAGE_FAMILY_PROBE_REPORT_PATH",
    "W02_BASE_LANGUAGE_FAMILY_PROBE_VERSION",
    "W02BaseLanguageFamilyProbeError",
    "run_w02_base_language_family_probe",
]
