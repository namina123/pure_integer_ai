"""Append-only unused PUD-Wikipedia source for the R3 blind owner."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    SourceRefRecord,
    StableRecordKey,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v3 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH,
    PUD_NEWS_SOURCE_KEY,
    blind_private_source_specs_v3,
    build_blind_private_source_extension_v3_manifest,
    read_blind_private_source_extension_v3_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v4.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V4_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V4"
)
PARENT_EXTENSION_V3_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v3.py"
)
PARENT_EXTENSION_V3_CODE_SIZE_BYTES = 7_731
PARENT_EXTENSION_V3_CODE_SHA256 = (
    "294427c06b66da453e66a16bba57b6ca333b82df74792c1f4fdd5c4c719ecada"
)
PARENT_EXTENSION_V3_MANIFEST_SIZE_BYTES = 3_753
PARENT_EXTENSION_V3_MANIFEST_SHA256 = (
    "54962c192f0d49b135646badbbb2ac81bea1b245bbacf87f27560eb3f6ebd1c2"
)
CONSUMED_R2_FAMILY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_family_freeze_v1.json"
)
CONSUMED_R2_FAMILY_FREEZE_SIZE_BYTES = 13_681
CONSUMED_R2_FAMILY_FREEZE_SHA256 = (
    "26266ae7068ace1fdf9b4efd51071eeb12485cbb983768d7aaab474803135afb"
)
CONSUMED_R2_PUBLIC_COMMIT = "8cae8807f9a669dcb668734c345d4f63d8716b25"
CONSUMED_R2_FAILURE_SEAL_SIZE_BYTES = 532
CONSUMED_R2_FAILURE_SEAL_SHA256 = (
    "2aa1d2f8378cc862400747ef7c3a08fb0c9f158596880d023b543245547aaa10"
)
CONSUMED_R2_FAILURE_ERROR_EVIDENCE_SHA256 = (
    "aaca72dfd87268404a399e4888d7dfd4dae5e64d69307fe94204abca7e48f3d8"
)
CONSUMED_R2_GUARD_SHA256 = (
    "6f6664890c49e4c5d57c28747923d3851fa0869fbbf70dfe4f31e8b201ceeda1"
)
CONSUMED_R2_RUN_INTENT_SHA256 = (
    "459abd534976a2cbf770c97a1f0ff32e86913ac8ca60ccfb9b2ceb80a645add2"
)
PUD_WIKIPEDIA_SOURCE_KEY = "UD_ZH_PUD_R2_18_WIKIPEDIA_BLIND_PRIVATE"
FIXED_BLOB_NEWS_SENTENCE_COUNT = 500
FIXED_BLOB_WIKIPEDIA_SENTENCE_COUNT = 500


# object-model: exception
class BlindPrivateSourceExtensionV4Error(DatasetContractError):
    """The R3 PUD-Wikipedia extension or a SourceRef drifted."""


def _source_specs() -> tuple[dict[str, object], ...]:
    source = deepcopy(blind_private_source_specs_v3()[0])
    source.update({
        "genre": "wikipedia",
        "snapshot_id": "ud-zh-pud-r2.18-wikipedia-only",
        "source_key": PUD_WIKIPEDIA_SOURCE_KEY,
        "source_origin": (
            "PUD Wikipedia-domain sentences with the news subset excluded"),
    })
    data_file = source["data_file"]
    assert isinstance(data_file, dict)
    data_file["owner_filter"] = {
        "count_basis": "ISOLATED_OWNER_SENT_ID_PREFIX_SCAN_OF_FIXED_R2_18_BLOB",
        "excluded_sentence_id_prefixes": ["n"],
        "fixed_blob_news_sentence_count": FIXED_BLOB_NEWS_SENTENCE_COUNT,
        "fixed_blob_wikipedia_sentence_count":
            FIXED_BLOB_WIKIPEDIA_SENTENCE_COUNT,
        "required_sentence_id_prefix": "w",
        "selection_policy": "PUD_WIKIPEDIA_ONLY_NO_NEWS",
    }
    return (source,)


def blind_private_source_specs_v4() -> tuple[dict[str, object], ...]:
    """Return detached, payload-free specs for the unused w* domain."""
    return tuple(deepcopy(row) for row in _source_specs())


def build_blind_private_source_extension_v4_manifest() -> dict[str, object]:
    """Build the R3 source manifest without changing V1/V2/V3 bytes."""
    manifest = deepcopy(build_blind_private_source_extension_v3_manifest())
    excluded = [
        row for row in manifest["excluded_sources"]
        if row.get("source") != "UD_CHINESE_PUD_WIKIPEDIA_SUBSET"
    ]
    excluded.append({
        "reason": "PREVIOUS_BLIND_OWNER_PERMANENTLY_CONSUMED",
        "source": "UD_CHINESE_PUD_NEWS_SUBSET",
    })
    consumed = list(manifest["consumed_private_source_keys"])
    if PUD_NEWS_SOURCE_KEY not in consumed:
        consumed.append(PUD_NEWS_SOURCE_KEY)
    manifest.update({
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V4_VERSION,
        "consumed_private_source_keys": consumed,
        "consumed_r2_owner_reuse_authorized": 0,
        "excluded_sources": excluded,
        "next_action": "START_FRESH_ISOLATED_V3_BLIND_PRIVATE_OWNER_R3",
        "parent_extension_v3_code_sha256": PARENT_EXTENSION_V3_CODE_SHA256,
        "parent_extension_v3_manifest_sha256":
            PARENT_EXTENSION_V3_MANIFEST_SHA256,
        "previous_consumed_r2_family": {
            "aggregate_report_published": 0,
            "error_evidence_sha256":
                CONSUMED_R2_FAILURE_ERROR_EVIDENCE_SHA256,
            "error_type": "W02MorphologySuccessorV3PrivateEvaluationError",
            "failure_phase": "PRIVATE_AUTHORIZATION_OR_EVALUATION",
            "failure_seal_sha256": CONSUMED_R2_FAILURE_SEAL_SHA256,
            "failure_seal_size_bytes": CONSUMED_R2_FAILURE_SEAL_SIZE_BYTES,
            "family_freeze_sha256": CONSUMED_R2_FAMILY_FREEZE_SHA256,
            "formal_private_evaluation_runs": 1,
            "guard_sha256": CONSUMED_R2_GUARD_SHA256,
            "public_commit": CONSUMED_R2_PUBLIC_COMMIT,
            "reuse_authorized": 0,
            "run_intent_sha256": CONSUMED_R2_RUN_INTENT_SHA256,
            "status": "NE_NO_RECEIPT",
        },
        "scope": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R3-ONLY"),
        "source_nonoverlap_basis": (
            "UNUSED_PUD_WIKIPEDIA_SENT_ID_DOMAIN_WITH_FRESH_OWNER_AUDIT"),
        "sources": [deepcopy(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V4_APPROVED",
    })
    return manifest


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV4Error(
            "source extension V4 repository file is invalid")
    return target


def _assert_file(
        root: Path, relative: str, size: int, sha256: str) -> Path:
    target = _repository_file(root, relative)
    payload = target.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
        raise BlindPrivateSourceExtensionV4Error(
            "source extension V4 public dependency drifted")
    return target


def read_blind_private_source_extension_v4_manifest(
        repository_root: str | Path) -> dict[str, object]:
    """Read V4 and verify the immutable V3 and consumed-family lineage."""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH))
    if value != build_blind_private_source_extension_v4_manifest():
        raise BlindPrivateSourceExtensionV4Error(
            "source extension V4 manifest drifted")
    read_blind_private_source_extension_v3_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_V3_CODE_PATH,
        PARENT_EXTENSION_V3_CODE_SIZE_BYTES, PARENT_EXTENSION_V3_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH,
        PARENT_EXTENSION_V3_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_V3_MANIFEST_SHA256)
    freeze_path = _assert_file(
        root, CONSUMED_R2_FAMILY_FREEZE_PATH,
        CONSUMED_R2_FAMILY_FREEZE_SIZE_BYTES,
        CONSUMED_R2_FAMILY_FREEZE_SHA256)
    freeze = read_canonical_object(freeze_path)
    if (freeze.get("status")
            != "W02_SUCCESSOR_V3_BLIND_PRIVATE_FAMILY_FROZEN"
            or freeze.get("owner_source_count") != 500
            or freeze.get("owner_pair_count") != 500
            or freeze.get("formal_private_evaluation_runs") != 0
            or freeze.get("private_payload_reads") != 0):
        raise BlindPrivateSourceExtensionV4Error(
            "source extension V4 consumed-family freeze drifted")
    return value


def _validate_source_span(value: object) -> None:
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionV4Error(
            "PUD Wikipedia SourceRef span fields drifted")
    locator = value.get("locator_value")
    if (value.get("locator_kind") != "sentence"
            or not isinstance(locator, str) or not locator.startswith("w")
            or type(value.get("span_start")) is not int
            or type(value.get("span_end")) is not int
            or value["span_start"] < 0
            or value["span_end"] < value["span_start"]):
        raise BlindPrivateSourceExtensionV4Error(
            "PUD R3 owner must use a Wikipedia sentence locator")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="PUD Wikipedia document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="PUD Wikipedia entity cluster")


def validate_blind_private_source_ref_v4(
        value: dict[str, Any]) -> SourceRefRecord:
    """Validate one w* SourceRef and reject news or consumed source keys."""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionV4Error(
            "PUD Wikipedia SourceRef fields drifted")
    spec = _source_specs()[0]
    if value.get("source_key") != PUD_WIKIPEDIA_SOURCE_KEY:
        raise BlindPrivateSourceExtensionV4Error(
            "source is not authorized for the R3 owner")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionV4Error(
            "PUD Wikipedia SourceRef is not canonical")
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    if (record.format_version != V2_FORMAT_VERSION
            or record.schema_version != V2_SCHEMA_VERSION
            or record.course_version != V2_COURSE_VERSION
            or record.snapshot_id != spec["snapshot_id"]
            or record.revision_id != spec["commit_sha1"]
            or record.official_url != spec["repository_url"]
            or record.license_id != spec["license_id"]
            or record.redistribution_policy != "PUBLIC"
            or record.upstream_checksum
            != "sha1:" + str(data_file["git_blob_sha1"])
            or not record.source_identity.startswith(
                f"{PUD_WIKIPEDIA_SOURCE_KEY}:sentence:w")):
        raise BlindPrivateSourceExtensionV4Error(
            "PUD Wikipedia SourceRef provenance drifted")
    _validate_source_span(value["source_span"])
    return record


def validate_blind_private_owner_record_v4(value: dict[str, Any]) -> object:
    """Use V4 authority only for the new w* SourceRef domain."""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v4(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V4_VERSION",
    "CONSUMED_R2_FAILURE_SEAL_SHA256",
    "CONSUMED_R2_FAMILY_FREEZE_SHA256",
    "BlindPrivateSourceExtensionV4Error",
    "PUD_WIKIPEDIA_SOURCE_KEY",
    "blind_private_source_specs_v4",
    "build_blind_private_source_extension_v4_manifest",
    "read_blind_private_source_extension_v4_manifest",
    "validate_blind_private_owner_record_v4",
    "validate_blind_private_source_ref_v4",
]
