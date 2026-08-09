"""Owner-only source extension for the post-V3 blind private family."""
from __future__ import annotations

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
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension import (
    BLIND_PRIVATE_SOURCE_EXTENSION_PATH,
    read_blind_private_source_extension_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_V2_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v2.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V2_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V2"
)
PARENT_EXTENSION_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension.py"
)
PARENT_EXTENSION_CODE_SIZE_BYTES = 10_911
PARENT_EXTENSION_CODE_SHA256 = (
    "b9dcfe90a4a9511ceda0ece80c1a9108f4f00ea6237bdf371c1cdea100c6def6"
)
PARENT_EXTENSION_MANIFEST_SIZE_BYTES = 3_111
PARENT_EXTENSION_MANIFEST_SHA256 = (
    "1a41419bb9406c7d20846085a17ca9a40b778f2554cdf8905d4e4fd2ee699d64"
)
CONSUMED_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_private_owner_receipt_v1.json"
)
CONSUMED_OWNER_RECEIPT_SIZE_BYTES = 7_309
CONSUMED_OWNER_RECEIPT_SHA256 = (
    "d5d97d5ddc7cf155fdebf492c287f9fe5d68e0e688d21f5d20230b1afb493829"
)
V3_SHADOW_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_route_shadow_report_v1.json"
)
V3_SHADOW_REPORT_SIZE_BYTES = 4_375
V3_SHADOW_REPORT_SHA256 = (
    "ca9ff9e5969ba944a611ea554f9870c65e8d9dc0da90d34c25f613a6ba304f33"
)
PUD_NEWS_SOURCE_KEY = "UD_ZH_PUD_R2_18_NEWS_BLIND_PRIVATE"
CONSUMED_PRIVATE_SOURCE_KEYS = (
    "UD_ZH_CFL_R2_18_BLIND_PRIVATE",
    "UD_ZH_HK_R2_18_BLIND_PRIVATE",
)


# object-model: exception
class BlindPrivateSourceExtensionV2Error(DatasetContractError):
    """The post-V3 owner source extension or SourceRef drifted."""


def _source_specs() -> tuple[dict[str, object], ...]:
    return ({
        "annotation_provenance": (
            "manual and converted under the documented PUD process"),
        "commit_sha1": "7b54411fb8cec041c4f8794413e08878605792f3",
        "data_file": {
            "git_blob_sha1": "126a653c0bc55838f07e86183bae7842001d06f5",
            "main_session_content_reads": 0,
            "owner_filter": {
                "documented_news_sentence_count": 750,
                "excluded_sentence_id_prefixes": ["w"],
                "required_sentence_id_prefix": "n",
                "selection_policy": "PUD_NEWS_ONLY_NO_WIKIPEDIA",
            },
            "relative_path": "zh_pud-ud-test.conllu",
            "size_bytes": 2_199_911,
            "upstream_split": "test",
        },
        "genre": "news",
        "language": "zh",
        "license_evidence": {
            "git_blob_sha1": "a0bd8baeb667daae1407869f3152489fab3001de",
            "relative_path": "LICENSE.txt",
            "sha256": (
                "b278eb53fe50b8bb7fa0d90fb8536c35fdcaa80f9d63812cb51db539555d2a89"),
            "size_bytes": 19_556,
        },
        "license_id": "CC-BY-SA-3.0",
        "parallel": 1,
        "readme_evidence": {
            "git_blob_sha1": "031dd74262c8e03f16e5b8109d2fff21a0ce203b",
            "relative_path": "README.md",
            "sha256": (
                "4abc2a54f3b38f8da61c9a751f26d8e3292c99461e9e7fe28b6ad545d3fbd4e7"),
            "size_bytes": 5_743,
        },
        "repository_url": (
            "https://github.com/UniversalDependencies/UD_Chinese-PUD"),
        "script": "Chinese",
        "snapshot_id": "ud-zh-pud-r2.18-news-only",
        "source_key": PUD_NEWS_SOURCE_KEY,
        "source_origin": (
            "PUD news-domain sentences with the Wikipedia subset excluded"),
        "tag": "r2.18",
    },)


def blind_private_source_specs_v2() -> tuple[dict[str, object], ...]:
    """Return detached payload-free source specs for isolated owner tooling."""
    return tuple({**row} for row in _source_specs())


def build_blind_private_source_extension_v2_manifest() -> dict[str, object]:
    """Build the post-V3 owner-only, payload-free source manifest."""
    return {
        "artifact_kind": "PH2_D03_V2_BLIND_PRIVATE_SOURCE_EXTENSION",
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V2_VERSION,
        "candidate_evaluation_runs": 0,
        "consumed_private_source_keys": list(CONSUMED_PRIVATE_SOURCE_KEYS),
        "development_authorized": 0,
        "excluded_sources": [
            {
                "reason": "CC_BY_NC_SA_IS_NOT_OPEN_FOR_UNRESTRICTED_USE",
                "source": "UD_CHINESE_BEGINNER",
            },
            {
                "reason": "CC_BY_NC_SA_IS_NOT_OPEN_FOR_UNRESTRICTED_USE",
                "source": "UD_CHINESE_PATENTCHAR",
            },
            {
                "reason": "CONTENT_EQUIVALENT_TO_USED_GSDSIMP_TREEBANK",
                "source": "UD_CHINESE_GSD",
            },
            {
                "reason": "PREVIOUS_BLIND_OWNER_PERMANENTLY_CONSUMED",
                "source": "UD_CHINESE_CFL_AND_HK",
            },
            {
                "reason": "TRAIN_DEV_SHADOW_SOURCE_ALREADY_USED",
                "source": "UD_CHINESE_GSDSIMP",
            },
            {
                "reason": "WIKIPEDIA_DOMAIN_EXCLUDED_BY_SENTENCE_ID_PREFIX",
                "source": "UD_CHINESE_PUD_WIKIPEDIA_SUBSET",
            },
        ],
        "formal_private_evaluation_runs": 0,
        "format_version": 1,
        "main_session_conllu_payload_reads": 0,
        "next_action": "START_ISOLATED_V3_BLIND_PRIVATE_OWNER",
        "parent_extension_code_sha256": PARENT_EXTENSION_CODE_SHA256,
        "parent_extension_manifest_sha256": PARENT_EXTENSION_MANIFEST_SHA256,
        "previous_consumed_owner_receipt_sha256":
            CONSUMED_OWNER_RECEIPT_SHA256,
        "private_owner_authorized": 1,
        "scope": "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-ONLY",
        "shadow_authorized": 0,
        "source_nonoverlap_basis": (
            "NEW_PUD_NEWS_SOURCE_IDENTITY_WITH_OWNER_EXACT_CONTENT_AUDIT"),
        "sources": [dict(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V2_APPROVED",
        "training_authorized": 0,
        "v3_formal_shadow_report_sha256": V3_SHADOW_REPORT_SHA256,
    }


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV2Error(
            "source extension V2 repository file is invalid")
    return target


def _assert_file(
        root: Path,
        relative: str,
        size: int,
        sha256: str,
        ) -> Path:
    target = _repository_file(root, relative)
    payload = target.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
        raise BlindPrivateSourceExtensionV2Error(
            "source extension V2 public dependency drifted")
    return target


def read_blind_private_source_extension_v2_manifest(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """Read the V2 manifest and verify every live public dependency."""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V2_PATH))
    if value != build_blind_private_source_extension_v2_manifest():
        raise BlindPrivateSourceExtensionV2Error(
            "source extension V2 manifest drifted")
    read_blind_private_source_extension_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_CODE_PATH,
        PARENT_EXTENSION_CODE_SIZE_BYTES, PARENT_EXTENSION_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_PATH,
        PARENT_EXTENSION_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_MANIFEST_SHA256)
    _assert_file(
        root, CONSUMED_OWNER_RECEIPT_PATH,
        CONSUMED_OWNER_RECEIPT_SIZE_BYTES, CONSUMED_OWNER_RECEIPT_SHA256)
    shadow_path = _assert_file(
        root, V3_SHADOW_REPORT_PATH,
        V3_SHADOW_REPORT_SIZE_BYTES, V3_SHADOW_REPORT_SHA256)
    shadow = read_canonical_object(shadow_path)
    if (shadow.get("status") != "PASS"
            or shadow.get("run_scope") != "FORMAL"
            or shadow.get("formal_shadow_audit_runs") != 1
            or shadow.get("formal_private_evaluation_runs") != 0
            or shadow.get("private_payload_reads") != 0
            or shadow.get("label_reads") != 0
            or shadow.get("teacher_calls") != 0):
        raise BlindPrivateSourceExtensionV2Error(
            "source extension V2 parent shadow PASS drifted")
    return value


def _validate_source_span(value: object) -> None:
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionV2Error(
            "PUD news SourceRef span fields drifted")
    locator = value.get("locator_value")
    if (value.get("locator_kind") != "sentence"
            or not isinstance(locator, str) or not locator.startswith("n")
            or type(value.get("span_start")) is not int
            or type(value.get("span_end")) is not int
            or value["span_start"] < 0
            or value["span_end"] < value["span_start"]):
        raise BlindPrivateSourceExtensionV2Error(
            "PUD owner must use a news sentence locator")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="PUD news document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="PUD news entity cluster")


def validate_blind_private_source_ref_v2(
        value: dict[str, Any],
        ) -> SourceRefRecord:
    """Validate one PUD-news SourceRef and reject the Wikipedia subset."""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionV2Error(
            "PUD news SourceRef fields drifted")
    spec = _source_specs()[0]
    if value.get("source_key") != PUD_NEWS_SOURCE_KEY:
        raise BlindPrivateSourceExtensionV2Error(
            "source is not authorized for the post-V3 owner")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionV2Error(
            "PUD news SourceRef is not canonical")
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
                f"{PUD_NEWS_SOURCE_KEY}:sentence:n")):
        raise BlindPrivateSourceExtensionV2Error(
            "PUD news SourceRef provenance drifted")
    _validate_source_span(value["source_span"])
    return record


def validate_blind_private_owner_record_v2(value: dict[str, Any]) -> object:
    """Allow only PUD-news SourceRefs; preserve old validation for other records."""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v2(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V2_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V2_VERSION",
    "BlindPrivateSourceExtensionV2Error",
    "PUD_NEWS_SOURCE_KEY",
    "blind_private_source_specs_v2",
    "build_blind_private_source_extension_v2_manifest",
    "read_blind_private_source_extension_v2_manifest",
    "validate_blind_private_owner_record_v2",
    "validate_blind_private_source_ref_v2",
]
