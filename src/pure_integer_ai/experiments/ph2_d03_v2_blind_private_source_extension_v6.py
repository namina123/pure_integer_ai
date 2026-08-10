"""Append-only Kyoto Classical Chinese source for the R5 blind owner."""
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
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH,
    KYOTO_COMMIT_SHA1,
    KYOTO_DATA_BLOB_SHA1,
    KYOTO_DATA_SIZE_BYTES,
    KYOTO_LICENSE_BLOB_SHA1,
    KYOTO_LICENSE_SHA256,
    KYOTO_README_BLOB_SHA1,
    KYOTO_README_SHA256,
    KYOTO_SOURCE_KEY as CONSUMED_R4_KYOTO_SOURCE_KEY,
    KYOTO_TEST_SENTENCE_COUNT,
    build_blind_private_source_extension_v5_manifest,
    read_blind_private_source_extension_v5_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v6.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V6"
)
PARENT_EXTENSION_V5_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v5.py"
)
PARENT_EXTENSION_V5_CODE_SIZE_BYTES = 13_913
PARENT_EXTENSION_V5_CODE_SHA256 = (
    "da799df8a505e01dcc74c969c350dfb33b99b85e683d3ca5c1088133ac43169e"
)
PARENT_EXTENSION_V5_MANIFEST_SIZE_BYTES = 6_287
PARENT_EXTENSION_V5_MANIFEST_SHA256 = (
    "76003a6531df489ee1ec91772ca8b4be449d628418de387bcc0f67c9b3c2be11"
)
CONSUMED_R4_FAMILY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_r4_family_freeze_v1.json"
)
CONSUMED_R4_FAMILY_FREEZE_SIZE_BYTES = 14_947
CONSUMED_R4_FAMILY_FREEZE_SHA256 = (
    "bbebc7df2a4b80a79ed6b0aba33206be81f0c27f8a2b8864e1de1ed367a6a43a"
)
CONSUMED_R4_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_receipt_v1.json"
)
CONSUMED_R4_OWNER_RECEIPT_SIZE_BYTES = 10_029
CONSUMED_R4_OWNER_RECEIPT_SHA256 = (
    "3145abfb35f9388e316f2e05612d3ebc6df87202d02345d89ff6b354533e1a8d"
)
CONSUMED_R4_PUBLIC_COMMIT = "f9d270b78e853012b5d3809d6346bb82904f5276"
CONSUMED_R4_FAILURE_SEAL_SIZE_BYTES = 511
CONSUMED_R4_FAILURE_SEAL_SHA256 = (
    "3f428e696543cd822ff6877e70df194535f0c5f73c7cb86fc5c49c3e3b22aead"
)
CONSUMED_R4_FAILURE_ERROR_EVIDENCE_SHA256 = (
    "4a6d455cd094856e686c265880514686d0167274ffbffe044baec0d182b9337d"
)
CONSUMED_R4_GUARD_SHA256 = (
    "ff3065b2fc78c941e7d61ee01bfb77f0464826a60035b69a4b862fa944ac0ff0"
)
CONSUMED_R4_RUN_INTENT_SHA256 = (
    "ce68b28e04a13ef2c49cf44548d006ca4af7e4c847d2ca0a9ec76ade63e1f7db"
)
BASE_LANGUAGE_ADAPTER_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_base_language_family_adapter.py"
)
BASE_LANGUAGE_ADAPTER_CODE_SIZE_BYTES = 2_492
BASE_LANGUAGE_ADAPTER_CODE_SHA256 = (
    "a71dd56a68e94eea8bb7d87e569d96a98a029c488a4ea02d4de11d3c4cf58a69"
)
BASE_LANGUAGE_PROBE_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_base_language_family_probe.py"
)
BASE_LANGUAGE_PROBE_CODE_SIZE_BYTES = 9_058
BASE_LANGUAGE_PROBE_CODE_SHA256 = (
    "e997947a1e72fe0b9ac9afd84cb2738ccf3a2c2c15d5ee179db9d6e4d24ee867"
)
BASE_LANGUAGE_PROBE_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_base_language_family_probe_report_v1.json"
)
BASE_LANGUAGE_PROBE_REPORT_SIZE_BYTES = 1_471
BASE_LANGUAGE_PROBE_REPORT_SHA256 = (
    "41fd0bd004c2c6fe5c1f8b6f6d6c8251af8e3fc7be0778975edbaabaaf979a31"
)
KYOTO_REMAINDER_SOURCE_KEY = (
    "UD_LZH_KYOTO_R2_18_TEST_REMAINDER_BLIND_PRIVATE"
)
KYOTO_REMAINDER_MINIMUM_ORDINAL = 1_001
KYOTO_REMAINDER_AVAILABLE_SENTENCES = 4_528
CONSUMED_R4_SELECTION_SCAN_UPPER_BOUND = 592


# object-model: exception
class BlindPrivateSourceExtensionV6Error(DatasetContractError):
    """The R5 Kyoto extension or one authorized SourceRef drifted."""


def _source_specs() -> tuple[dict[str, object], ...]:
    return ({
        "annotation_provenance": (
            "manual native Classical Chinese UD annotation converted by "
            "Kyoto University"),
        "commit_sha1": KYOTO_COMMIT_SHA1,
        "data_file": {
            "git_blob_sha1": KYOTO_DATA_BLOB_SHA1,
            "main_session_content_reads": 0,
            "owner_filter": {
                "available_sentence_count": KYOTO_REMAINDER_AVAILABLE_SENTENCES,
                "contamination_and_duplicate_audit_required": 1,
                "max_private_source_count": 500,
                "minimum_sentence_ordinal":
                    KYOTO_REMAINDER_MINIMUM_ORDINAL,
                "previous_r4_selection_scan_upper_bound":
                    CONSUMED_R4_SELECTION_SCAN_UPPER_BOUND,
                "selection_policy": (
                    "DETERMINISTIC_TEST_ORDER_AT_OR_AFTER_ORDINAL_1001_"
                    "AFTER_PUBLIC_AND_PRIVATE_CONTAMINATION_FILTERS"),
            },
            "relative_path": "lzh_kyoto-ud-test.conllu",
            "size_bytes": KYOTO_DATA_SIZE_BYTES,
            "upstream_split": "test",
        },
        "genre": "classical-literature",
        "language": "lzh",
        "license_evidence": {
            "git_blob_sha1": KYOTO_LICENSE_BLOB_SHA1,
            "relative_path": "LICENSE.txt",
            "sha256": KYOTO_LICENSE_SHA256,
            "size_bytes": 202,
        },
        "license_id": "CC-BY-SA-4.0",
        "parallel": 0,
        "readme_evidence": {
            "git_blob_sha1": KYOTO_README_BLOB_SHA1,
            "relative_path": "README.md",
            "sha256": KYOTO_README_SHA256,
            "size_bytes": 4_742,
        },
        "repository_url": (
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-Kyoto"),
        "script": "Hant",
        "snapshot_id": "ud-lzh-kyoto-r2.18-test-remainder-r5",
        "source_key": KYOTO_REMAINDER_SOURCE_KEY,
        "source_origin": (
            "unused Kyoto r2.18 test sentence ordinal 1001 or later"),
        "tag": "r2.18",
    },)


def blind_private_source_specs_v6() -> tuple[dict[str, object], ...]:
    """Return detached specs for the unconsumed Kyoto remainder domain."""
    return tuple(deepcopy(row) for row in _source_specs())


def build_blind_private_source_extension_v6_manifest() -> dict[str, object]:
    """Bind the consumed R4 failure and authorize only a fresh R5 owner."""
    manifest = deepcopy(build_blind_private_source_extension_v5_manifest())
    consumed = list(manifest["consumed_private_source_keys"])
    if CONSUMED_R4_KYOTO_SOURCE_KEY not in consumed:
        consumed.append(CONSUMED_R4_KYOTO_SOURCE_KEY)
    excluded = list(manifest["excluded_sources"])
    excluded.append({
        "reason": "R4_SELECTED_KYOTO_PREFIX_DOMAIN_PERMANENTLY_CONSUMED",
        "source": "UD_CLASSICAL_CHINESE_KYOTO_TEST_ORDINALS_BELOW_1001",
    })
    manifest.update({
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION,
        "consumed_private_source_keys": consumed,
        "consumed_r4_owner_reuse_authorized": 0,
        "excluded_sources": excluded,
        "next_action": "START_FRESH_ISOLATED_V3_BLIND_PRIVATE_OWNER_R5",
        "parent_extension_v5_code_sha256": PARENT_EXTENSION_V5_CODE_SHA256,
        "parent_extension_v5_manifest_sha256":
            PARENT_EXTENSION_V5_MANIFEST_SHA256,
        "previous_consumed_r4_family": {
            "aggregate_report_published": 0,
            "error_evidence_sha256":
                CONSUMED_R4_FAILURE_ERROR_EVIDENCE_SHA256,
            "error_type": "W02CandidateModelError",
            "failure_phase": "PRIVATE_AUTHORIZATION_OR_EVALUATION",
            "failure_seal_sha256": CONSUMED_R4_FAILURE_SEAL_SHA256,
            "failure_seal_size_bytes": CONSUMED_R4_FAILURE_SEAL_SIZE_BYTES,
            "family_freeze_sha256": CONSUMED_R4_FAMILY_FREEZE_SHA256,
            "formal_private_evaluation_runs": 1,
            "guard_sha256": CONSUMED_R4_GUARD_SHA256,
            "owner_pair_and_label_stream_consumed": 1,
            "owner_receipt_sha256": CONSUMED_R4_OWNER_RECEIPT_SHA256,
            "public_commit": CONSUMED_R4_PUBLIC_COMMIT,
            "reuse_authorized": 0,
            "run_intent_sha256": CONSUMED_R4_RUN_INTENT_SHA256,
            "source_ref_records_closed_before_pair_stream": 500,
            "base_candidate_failure": "OBSERVATION_LANGUAGE_LZH_OUT_OF_SCOPE",
            "status": "NE_NO_RECEIPT",
        },
        "public_base_language_family_adapter": {
            "adapter_changes_only_language": 1,
            "adapter_code_sha256": BASE_LANGUAGE_ADAPTER_CODE_SHA256,
            "adapter_version":
                "PH2-D03-V2-W02-BASE-LANGUAGE-FAMILY-ADAPTER-V1",
            "base_prediction_metamorphic_equal": 1,
            "base_scope_language": "zh",
            "probe_code_sha256": BASE_LANGUAGE_PROBE_CODE_SHA256,
            "probe_report_sha256": BASE_LANGUAGE_PROBE_REPORT_SHA256,
            "route_adapted_zh_authorized": 0,
            "route_original_lzh_authorized": 1,
            "source_language": "lzh",
            "status": "PASS",
        },
        "private_owner_authorized": 1,
        "scope": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R5-ONLY"),
        "source_nonoverlap_basis": (
            "KYOTO_TEST_ORDINAL_AT_OR_AFTER_1001_EXCLUDES_R4_SCAN_UPPER_"
            "BOUND_592_AND_REQUIRES_FULL_CONTAMINATION_AUDIT"),
        "sources": [deepcopy(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V6_APPROVED",
    })
    return manifest


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV6Error(
            "source extension V6 repository file is invalid")
    return target


def _assert_file(
        root: Path, relative: str, size: int, sha256: str) -> Path:
    target = _repository_file(root, relative)
    payload = target.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
        raise BlindPrivateSourceExtensionV6Error(
            "source extension V6 public dependency drifted")
    return target


def read_blind_private_source_extension_v6_manifest(
        repository_root: str | Path) -> dict[str, object]:
    """Read V6 and verify V5, R4 sealing, and the public adapter probe."""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH))
    if value != build_blind_private_source_extension_v6_manifest():
        raise BlindPrivateSourceExtensionV6Error(
            "source extension V6 manifest drifted")
    read_blind_private_source_extension_v5_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_V5_CODE_PATH,
        PARENT_EXTENSION_V5_CODE_SIZE_BYTES, PARENT_EXTENSION_V5_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH,
        PARENT_EXTENSION_V5_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_V5_MANIFEST_SHA256)
    freeze_path = _assert_file(
        root, CONSUMED_R4_FAMILY_FREEZE_PATH,
        CONSUMED_R4_FAMILY_FREEZE_SIZE_BYTES,
        CONSUMED_R4_FAMILY_FREEZE_SHA256)
    owner_path = _assert_file(
        root, CONSUMED_R4_OWNER_RECEIPT_PATH,
        CONSUMED_R4_OWNER_RECEIPT_SIZE_BYTES,
        CONSUMED_R4_OWNER_RECEIPT_SHA256)
    _assert_file(
        root, BASE_LANGUAGE_ADAPTER_CODE_PATH,
        BASE_LANGUAGE_ADAPTER_CODE_SIZE_BYTES,
        BASE_LANGUAGE_ADAPTER_CODE_SHA256)
    _assert_file(
        root, BASE_LANGUAGE_PROBE_CODE_PATH,
        BASE_LANGUAGE_PROBE_CODE_SIZE_BYTES,
        BASE_LANGUAGE_PROBE_CODE_SHA256)
    report_path = _assert_file(
        root, BASE_LANGUAGE_PROBE_REPORT_PATH,
        BASE_LANGUAGE_PROBE_REPORT_SIZE_BYTES,
        BASE_LANGUAGE_PROBE_REPORT_SHA256)
    freeze = read_canonical_object(freeze_path)
    if (freeze.get("status")
            != "W02_SUCCESSOR_V3_R4_BLIND_PRIVATE_FAMILY_FROZEN"
            or freeze.get("owner_source_count") != 500
            or freeze.get("owner_pair_count") != 500
            or freeze.get("formal_private_evaluation_runs") != 0
            or freeze.get("private_payload_reads") != 0):
        raise BlindPrivateSourceExtensionV6Error(
            "source extension V6 consumed R4 freeze drifted")
    owner = read_canonical_object(owner_path)
    report = read_canonical_object(report_path)
    if (owner.get("status")
            != "OWNER_METADATA_INGESTED_SOURCE_V5_PAYLOAD_UNREAD"
            or owner.get("source_key") != CONSUMED_R4_KYOTO_SOURCE_KEY
            or owner.get("source_count") != 500
            or report.get("status") != "PASS"
            or report.get("adapter_changes_only_language") != 1
            or report.get("base_prediction_metamorphic_equal") != 1
            or report.get("source_language") != "lzh"
            or report.get("base_scope_language") != "zh"
            or report.get("route_original_lzh_authorized") != 1
            or report.get("route_adapted_zh_authorized") != 0
            or report.get("formal_private_evaluation_runs") != 0
            or report.get("teacher_calls") != 0):
        raise BlindPrivateSourceExtensionV6Error(
            "source extension V6 adapter or consumed owner evidence drifted")
    return value


def _validate_source_span(value: object) -> int:
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionV6Error(
            "Kyoto SourceRef span fields drifted")
    locator = value.get("locator_value")
    parts = locator.split(":", 2) if isinstance(locator, str) else []
    try:
        ordinal = int(parts[1]) if len(parts) == 3 else 0
    except ValueError:
        ordinal = 0
    if (value.get("locator_kind") != "sentence"
            or len(parts) != 3 or parts[0] != "test"
            or ordinal < KYOTO_REMAINDER_MINIMUM_ORDINAL
            or ordinal > KYOTO_TEST_SENTENCE_COUNT
            or type(value.get("span_start")) is not int
            or type(value.get("span_end")) is not int
            or value["span_start"] < 0
            or value["span_end"] < value["span_start"]):
        raise BlindPrivateSourceExtensionV6Error(
            "R5 owner must use a Kyoto test sentence locator")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="Kyoto document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="Kyoto entity cluster")
    return ordinal


def validate_blind_private_source_ref_v6(
        value: dict[str, Any]) -> SourceRefRecord:
    """Validate one Kyoto test SourceRef and reject every consumed source."""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionV6Error(
            "Kyoto SourceRef fields drifted")
    spec = _source_specs()[0]
    if value.get("source_key") != KYOTO_REMAINDER_SOURCE_KEY:
        raise BlindPrivateSourceExtensionV6Error(
            "source is not authorized for the R5 owner")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionV6Error(
            "Kyoto SourceRef is not canonical")
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
            != "sha1:" + str(data_file["git_blob_sha1"])):
        raise BlindPrivateSourceExtensionV6Error(
            "Kyoto SourceRef provenance drifted")
    ordinal = _validate_source_span(value["source_span"])
    locator = value["source_span"]["locator_value"]
    if record.source_identity != (
            f"{KYOTO_REMAINDER_SOURCE_KEY}:sentence:{locator}"):
        raise BlindPrivateSourceExtensionV6Error(
            "Kyoto remainder SourceRef ordinal identity drifted")
    return record


def validate_blind_private_owner_record_v6(value: dict[str, Any]) -> object:
    """Use V6 authority for Kyoto SourceRef and the frozen V2 schema otherwise."""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v6(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION",
    "BlindPrivateSourceExtensionV6Error",
    "BASE_LANGUAGE_PROBE_REPORT_SHA256",
    "CONSUMED_R4_FAILURE_SEAL_SHA256",
    "CONSUMED_R4_FAMILY_FREEZE_SHA256",
    "KYOTO_REMAINDER_MINIMUM_ORDINAL",
    "KYOTO_REMAINDER_SOURCE_KEY",
    "blind_private_source_specs_v6",
    "build_blind_private_source_extension_v6_manifest",
    "read_blind_private_source_extension_v6_manifest",
    "validate_blind_private_owner_record_v6",
    "validate_blind_private_source_ref_v6",
]
