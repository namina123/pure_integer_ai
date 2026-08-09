"""Append-only Kyoto Classical Chinese source for the R4 blind owner."""
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
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v4 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH,
    PUD_WIKIPEDIA_SOURCE_KEY,
    build_blind_private_source_extension_v4_manifest,
    read_blind_private_source_extension_v4_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v5.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V5"
)
PARENT_EXTENSION_V4_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v4.py"
)
PARENT_EXTENSION_V4_CODE_SIZE_BYTES = 12_141
PARENT_EXTENSION_V4_CODE_SHA256 = (
    "c2e2d4e4542d26a42e61d465c7b4b350e2873ae3121a12decaa7181e541befbe"
)
PARENT_EXTENSION_V4_MANIFEST_SIZE_BYTES = 4_826
PARENT_EXTENSION_V4_MANIFEST_SHA256 = (
    "ea3d48e518e6ca9e3c7cc90a4d391ad4304ae19a66d28c28ccab87bc613165a8"
)
CONSUMED_R3_FAMILY_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_r3_family_freeze_v1.json"
)
CONSUMED_R3_FAMILY_FREEZE_SIZE_BYTES = 14_580
CONSUMED_R3_FAMILY_FREEZE_SHA256 = (
    "ad52b8abd11859e45de46f0ae7287f91b72bc8329cf965fa4fc5a1bddcd2be2a"
)
CONSUMED_R3_PUBLIC_COMMIT = "33af00a2de42f44d47be0198bc0736af6f05f911"
CONSUMED_R3_FAILURE_SEAL_SIZE_BYTES = 527
CONSUMED_R3_FAILURE_SEAL_SHA256 = (
    "a22ef161c67c09f1387a30fb7eaa7e1fa5253044bea2360cc229330b6d56db3b"
)
CONSUMED_R3_FAILURE_ERROR_EVIDENCE_SHA256 = (
    "ee3149fe9e9f5925591947d902d8d12147352bdf33456f3dfca55f94b4e9e063"
)
CONSUMED_R3_GUARD_SHA256 = (
    "ceed3fee57de29b804b9a4926aa7ef464651541b61d2df70ebfa66df3df2b2e3"
)
CONSUMED_R3_RUN_INTENT_SHA256 = (
    "c45a34287b3f3d04fdea38281072be868d80d2c4648cd3c233ce1af91821a794"
)
KYOTO_SOURCE_KEY = "UD_LZH_KYOTO_R2_18_TEST_BLIND_PRIVATE"
KYOTO_COMMIT_SHA1 = "2f5ff2e1ac5df5315cbe547283cca80fb69224e0"
KYOTO_DATA_BLOB_SHA1 = "226c48f1148c8ae8b69dcd527b2b07c306ac4301"
KYOTO_DATA_SIZE_BYTES = 2_698_869
KYOTO_TEST_SENTENCE_COUNT = 5_528
KYOTO_LICENSE_BLOB_SHA1 = "9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846"
KYOTO_LICENSE_SHA256 = (
    "899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79"
)
KYOTO_README_BLOB_SHA1 = "db3dbc8b51283ed32878b609e5f6b8b9feeb52e6"
KYOTO_README_SHA256 = (
    "2067b50bc8f23189d60bfed740e3438c8186b9a0297533362a1c6a0504d415e6"
)


# object-model: exception
class BlindPrivateSourceExtensionV5Error(DatasetContractError):
    """The R4 Kyoto extension or one authorized SourceRef drifted."""


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
                "available_sentence_count": KYOTO_TEST_SENTENCE_COUNT,
                "contamination_and_duplicate_audit_required": 1,
                "max_private_source_count": 500,
                "selection_policy": (
                    "DETERMINISTIC_TEST_ORDER_AFTER_PUBLIC_AND_PRIVATE_"
                    "CONTAMINATION_FILTERS"),
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
        "snapshot_id": "ud-lzh-kyoto-r2.18-test",
        "source_key": KYOTO_SOURCE_KEY,
        "source_origin": (
            "unused non-parallel Kyoto Classical Chinese r2.18 test split"),
        "tag": "r2.18",
    },)


def blind_private_source_specs_v5() -> tuple[dict[str, object], ...]:
    """Return detached metadata-only specs for the unused Kyoto domain."""
    return tuple(deepcopy(row) for row in _source_specs())


def build_blind_private_source_extension_v5_manifest() -> dict[str, object]:
    """Bind the consumed R3 failure and authorize only a fresh R4 owner."""
    manifest = deepcopy(build_blind_private_source_extension_v4_manifest())
    consumed = list(manifest["consumed_private_source_keys"])
    if PUD_WIKIPEDIA_SOURCE_KEY not in consumed:
        consumed.append(PUD_WIKIPEDIA_SOURCE_KEY)
    excluded = list(manifest["excluded_sources"])
    excluded.extend((
        {
            "reason": "PREVIOUS_BLIND_OWNER_PERMANENTLY_CONSUMED",
            "source": "UD_CHINESE_PUD_WIKIPEDIA_SUBSET",
        },
        {
            "reason": "PARALLEL_TO_PREVIOUSLY_CONSUMED_CHINESE_HK_PRIVATE",
            "source": "UD_CANTONESE_HK",
        },
        {
            "reason": "ONLY_100_SENTENCES_BELOW_R4_TARGET",
            "source": "UD_CLASSICAL_CHINESE_TUECL",
        },
    ))
    manifest.update({
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION,
        "consumed_private_source_keys": consumed,
        "consumed_r3_owner_reuse_authorized": 0,
        "excluded_sources": excluded,
        "next_action": "START_FRESH_ISOLATED_V3_BLIND_PRIVATE_OWNER_R4",
        "parent_extension_v4_code_sha256": PARENT_EXTENSION_V4_CODE_SHA256,
        "parent_extension_v4_manifest_sha256":
            PARENT_EXTENSION_V4_MANIFEST_SHA256,
        "previous_consumed_r3_family": {
            "aggregate_report_published": 0,
            "error_evidence_sha256":
                CONSUMED_R3_FAILURE_ERROR_EVIDENCE_SHA256,
            "error_type": "W02MorphologySuccessorV3PrivateIOError",
            "failure_phase": "PRIVATE_AUTHORIZATION_OR_EVALUATION",
            "failure_seal_sha256": CONSUMED_R3_FAILURE_SEAL_SHA256,
            "failure_seal_size_bytes": CONSUMED_R3_FAILURE_SEAL_SIZE_BYTES,
            "family_freeze_sha256": CONSUMED_R3_FAMILY_FREEZE_SHA256,
            "formal_private_evaluation_runs": 1,
            "guard_sha256": CONSUMED_R3_GUARD_SHA256,
            "owner_pair_and_label_stream_consumed": 1,
            "public_commit": CONSUMED_R3_PUBLIC_COMMIT,
            "reuse_authorized": 0,
            "run_intent_sha256": CONSUMED_R3_RUN_INTENT_SHA256,
            "source_validator_failure": (
                "V3_IO_REJECTED_V4_SOURCE_BEFORE_AGGREGATE"),
            "status": "NE_NO_RECEIPT",
        },
        "private_owner_authorized": 1,
        "scope": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R4-ONLY"),
        "source_nonoverlap_basis": (
            "UNUSED_NON_PARALLEL_KYOTO_CLASSICAL_CHINESE_TEST_WITH_FULL_"
            "PUBLIC_AND_PRIVATE_CONTAMINATION_AUDIT"),
        "sources": [deepcopy(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V5_APPROVED",
    })
    return manifest


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV5Error(
            "source extension V5 repository file is invalid")
    return target


def _assert_file(
        root: Path, relative: str, size: int, sha256: str) -> Path:
    target = _repository_file(root, relative)
    payload = target.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
        raise BlindPrivateSourceExtensionV5Error(
            "source extension V5 public dependency drifted")
    return target


def read_blind_private_source_extension_v5_manifest(
        repository_root: str | Path) -> dict[str, object]:
    """Read V5 and verify V4 plus the consumed R3 public freeze."""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH))
    if value != build_blind_private_source_extension_v5_manifest():
        raise BlindPrivateSourceExtensionV5Error(
            "source extension V5 manifest drifted")
    read_blind_private_source_extension_v4_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_V4_CODE_PATH,
        PARENT_EXTENSION_V4_CODE_SIZE_BYTES, PARENT_EXTENSION_V4_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_V4_PATH,
        PARENT_EXTENSION_V4_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_V4_MANIFEST_SHA256)
    freeze_path = _assert_file(
        root, CONSUMED_R3_FAMILY_FREEZE_PATH,
        CONSUMED_R3_FAMILY_FREEZE_SIZE_BYTES,
        CONSUMED_R3_FAMILY_FREEZE_SHA256)
    freeze = read_canonical_object(freeze_path)
    if (freeze.get("status")
            != "W02_SUCCESSOR_V3_R3_BLIND_PRIVATE_FAMILY_FROZEN"
            or freeze.get("owner_source_count") != 500
            or freeze.get("owner_pair_count") != 500
            or freeze.get("formal_private_evaluation_runs") != 0
            or freeze.get("private_payload_reads") != 0):
        raise BlindPrivateSourceExtensionV5Error(
            "source extension V5 consumed R3 freeze drifted")
    return value


def _validate_source_span(value: object) -> None:
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionV5Error(
            "Kyoto SourceRef span fields drifted")
    locator = value.get("locator_value")
    if (value.get("locator_kind") != "sentence"
            or not isinstance(locator, str) or not locator.startswith("test:")
            or type(value.get("span_start")) is not int
            or type(value.get("span_end")) is not int
            or value["span_start"] < 0
            or value["span_end"] < value["span_start"]):
        raise BlindPrivateSourceExtensionV5Error(
            "R4 owner must use a Kyoto test sentence locator")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="Kyoto document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="Kyoto entity cluster")


def validate_blind_private_source_ref_v5(
        value: dict[str, Any]) -> SourceRefRecord:
    """Validate one Kyoto test SourceRef and reject every consumed source."""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionV5Error(
            "Kyoto SourceRef fields drifted")
    spec = _source_specs()[0]
    if value.get("source_key") != KYOTO_SOURCE_KEY:
        raise BlindPrivateSourceExtensionV5Error(
            "source is not authorized for the R4 owner")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionV5Error(
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
            != "sha1:" + str(data_file["git_blob_sha1"])
            or not record.source_identity.startswith(
                f"{KYOTO_SOURCE_KEY}:sentence:test:")):
        raise BlindPrivateSourceExtensionV5Error(
            "Kyoto SourceRef provenance drifted")
    _validate_source_span(value["source_span"])
    return record


def validate_blind_private_owner_record_v5(value: dict[str, Any]) -> object:
    """Use V5 authority for Kyoto SourceRef and the frozen V2 schema otherwise."""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v5(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V5_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V5_VERSION",
    "BlindPrivateSourceExtensionV5Error",
    "CONSUMED_R3_FAILURE_SEAL_SHA256",
    "CONSUMED_R3_FAMILY_FREEZE_SHA256",
    "KYOTO_SOURCE_KEY",
    "blind_private_source_specs_v5",
    "build_blind_private_source_extension_v5_manifest",
    "read_blind_private_source_extension_v5_manifest",
    "validate_blind_private_owner_record_v5",
    "validate_blind_private_source_ref_v5",
]
