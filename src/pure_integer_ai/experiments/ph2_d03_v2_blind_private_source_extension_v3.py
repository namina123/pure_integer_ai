"""Append-only count correction for the PUD-news blind owner source."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    SourceRefRecord,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import read_canonical_object
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v2 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V2_PATH,
    BlindPrivateSourceExtensionV2Error,
    PUD_NEWS_SOURCE_KEY,
    blind_private_source_specs_v2,
    build_blind_private_source_extension_v2_manifest,
    read_blind_private_source_extension_v2_manifest,
    validate_blind_private_source_ref_v2,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record


BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v3.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V3_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V3"
)
PARENT_EXTENSION_V2_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v2.py"
)
PARENT_EXTENSION_V2_CODE_SIZE_BYTES = 12_714
PARENT_EXTENSION_V2_CODE_SHA256 = (
    "f86dc0aae2af7958f9f5544b98a6cd55bcf99f3058770954f7daaa59d4bfed5d"
)
PARENT_EXTENSION_V2_MANIFEST_SIZE_BYTES = 2_942
PARENT_EXTENSION_V2_MANIFEST_SHA256 = (
    "3f7e2dd99fe7856472ecd907a595ee5df1973a4f2d59f1f8ae98e7b006385ea5"
)
BLOCKED_OWNER_METADATA_SHA256 = (
    "9fecbad2bf1cfc04e9997ee5824d25e626e7fc34c59a2148094fa0abf5bd3689"
)
BLOCKED_OWNER_METADATA_SIZE_BYTES = 473
BLOCKED_OWNER_OPAQUE_ID = "8158afe9f3994c7d"
BLOCKED_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V3-PUD-NEWS-BLIND-8158afe9f3994c7d"
)
BLOCKED_OWNER_CODE = "PUD_SOURCE_DOMAIN_COUNT_MISMATCH"
FIXED_BLOB_NEWS_SENTENCE_COUNT = 500
FIXED_BLOB_WIKIPEDIA_SENTENCE_COUNT = 500


# object-model: exception
class BlindPrivateSourceExtensionV3Error(DatasetContractError):
    """The count-corrected owner extension or a public dependency drifted."""


def _source_specs() -> tuple[dict[str, object], ...]:
    source = deepcopy(blind_private_source_specs_v2()[0])
    data_file = source["data_file"]
    assert isinstance(data_file, dict)
    data_file["owner_filter"] = {
        "count_basis": (
            "ISOLATED_OWNER_SENT_ID_PREFIX_SCAN_OF_FIXED_R2_18_BLOB"),
        "excluded_sentence_id_prefixes": ["w"],
        "fixed_blob_news_sentence_count": FIXED_BLOB_NEWS_SENTENCE_COUNT,
        "fixed_blob_wikipedia_sentence_count":
            FIXED_BLOB_WIKIPEDIA_SENTENCE_COUNT,
        "required_sentence_id_prefix": "n",
        "selection_policy": "PUD_NEWS_ONLY_NO_WIKIPEDIA",
    }
    return (source,)


def blind_private_source_specs_v3() -> tuple[dict[str, object], ...]:
    """Return detached, payload-free specs with corrected fixed-blob counts."""
    return tuple(deepcopy(row) for row in _source_specs())


def build_blind_private_source_extension_v3_manifest() -> dict[str, object]:
    """Build the owner-only V3 manifest without changing published V2 bytes."""
    manifest = deepcopy(build_blind_private_source_extension_v2_manifest())
    manifest.update({
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V3_VERSION,
        "blocked_owner_reuse_authorized": 0,
        "next_action": "START_FRESH_ISOLATED_V3_BLIND_PRIVATE_OWNER_R2",
        "parent_extension_v2_code_sha256": PARENT_EXTENSION_V2_CODE_SHA256,
        "parent_extension_v2_manifest_sha256":
            PARENT_EXTENSION_V2_MANIFEST_SHA256,
        "previous_blocked_owner": {
            "blocker_code": BLOCKED_OWNER_CODE,
            "candidate_evaluation_runs": 0,
            "formal_artifact_count": 0,
            "formal_private_evaluation_runs": 0,
            "metadata_sha256": BLOCKED_OWNER_METADATA_SHA256,
            "metadata_size_bytes": BLOCKED_OWNER_METADATA_SIZE_BYTES,
            "opaque_owner_id": BLOCKED_OWNER_OPAQUE_ID,
            "owner_family_key": BLOCKED_OWNER_FAMILY_KEY,
            "status": "BLOCKED",
            "wikipedia_accepted_count": 0,
        },
        "scope": (
            "PH2-D03-V2-W02-SUCCESSOR-V3-BLIND-PRIVATE-OWNER-R2-ONLY"),
        "sources": [deepcopy(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V3_APPROVED",
    })
    return manifest


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV3Error(
            "source extension V3 repository file is invalid")
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
        raise BlindPrivateSourceExtensionV3Error(
            "source extension V3 public dependency drifted")
    return target


def read_blind_private_source_extension_v3_manifest(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """Read V3 and verify the immutable V2 lineage and correction boundary."""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH))
    if value != build_blind_private_source_extension_v3_manifest():
        raise BlindPrivateSourceExtensionV3Error(
            "source extension V3 manifest drifted")

    parent = read_blind_private_source_extension_v2_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_V2_CODE_PATH,
        PARENT_EXTENSION_V2_CODE_SIZE_BYTES, PARENT_EXTENSION_V2_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_V2_PATH,
        PARENT_EXTENSION_V2_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_V2_MANIFEST_SHA256)
    parent_filter = parent["sources"][0]["data_file"]["owner_filter"]
    if parent_filter.get("documented_news_sentence_count") != 750:
        raise BlindPrivateSourceExtensionV3Error(
            "source extension V3 correction parent drifted")
    return value


def validate_blind_private_source_ref_v3(
        value: dict[str, Any],
        ) -> SourceRefRecord:
    """Validate only a PUD-news SourceRef and continue rejecting ``w*``."""
    try:
        return validate_blind_private_source_ref_v2(value)
    except BlindPrivateSourceExtensionV2Error as error:
        raise BlindPrivateSourceExtensionV3Error(str(error)) from error


def validate_blind_private_owner_record_v3(value: dict[str, Any]) -> object:
    """Use corrected owner authority only for the authorized PUD SourceRef."""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v3(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V3_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V3_VERSION",
    "BLOCKED_OWNER_CODE",
    "BLOCKED_OWNER_METADATA_SHA256",
    "BLOCKED_OWNER_METADATA_SIZE_BYTES",
    "BlindPrivateSourceExtensionV3Error",
    "FIXED_BLOB_NEWS_SENTENCE_COUNT",
    "FIXED_BLOB_WIKIPEDIA_SENTENCE_COUNT",
    "PUD_NEWS_SOURCE_KEY",
    "blind_private_source_specs_v3",
    "build_blind_private_source_extension_v3_manifest",
    "read_blind_private_source_extension_v3_manifest",
    "validate_blind_private_owner_record_v3",
    "validate_blind_private_source_ref_v3",
]
