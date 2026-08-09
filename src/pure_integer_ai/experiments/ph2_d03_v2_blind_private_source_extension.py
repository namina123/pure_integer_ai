"""Append-only source extension for a new PH2-D03-V2 blind private owner."""
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
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    V2_SOURCE_KEYS,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v1.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V1"
)
PARENT_SCHEMA_SHA256 = (
    "a903a1a52ccc69e05098c3480f62f6db5d6643b2bfb433e45d8212a99c687fdc"
)
PARENT_SCHEMA_SIZE_BYTES = 21856
PARENT_SCHEMA_PATH = "src/pure_integer_ai/experiments/ph2_d03_v2_schema.py"
OLD_PRIVATE_FREEZE_COMMIT = "bb7a3e235e82f38243f18fee0e1eaaad1b186a71"
PARENT_SOURCE_KEYS = (
    "AUTHORED_CC0",
    "CONCEPTNET_5_7_0",
    "UD_ZH_GSDSIMP_R2_18",
    "WIKIDATA_REVISION_V1",
    "ZHWIKIPEDIA_20260701",
    "ZHWIKTIONARY_20260701",
)


class BlindPrivateSourceExtensionError(DatasetContractError):
    """The append-only private source extension or a SourceRef drifted."""


def _source_specs() -> tuple[dict[str, object], ...]:
    """Return immutable-by-construction upstream metadata, without payload."""
    return (
        {
            "annotation_provenance": "manual with documented automatic corrections",
            "commit_sha1": "ad71b068e4581343dab897ef2d54abf102580897",
            "data_file": {
                "git_blob_sha1": "029ae40a23928c11d09a8745ef92acea1b494d5b",
                "main_session_content_reads": 0,
                "relative_path": "zh_cfl-ud-test.conllu",
                "size_bytes": 651965,
                "upstream_split": "test",
            },
            "genre": "learner-essays",
            "language": "zh",
            "license_evidence": {
                "git_blob_sha1": "9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846",
                "relative_path": "LICENSE.txt",
                "sha256": "899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79",
                "size_bytes": 202,
            },
            "license_id": "CC-BY-SA-4.0",
            "parallel": 0,
            "readme_evidence": {
                "git_blob_sha1": "6096a6df4d7c23a8c635443a679fee361a2ad8c4",
                "relative_path": "README.md",
                "sha256": "987630d654b7ec57e164518f5a1083ea9f3f490320d28ad23959d88b853c11af",
                "size_bytes": 7817,
            },
            "repository_url": "https://github.com/UniversalDependencies/UD_Chinese-CFL",
            "script": "Simplified Chinese",
            "snapshot_id": "ud-zh-cfl-r2.18",
            "source_key": "UD_ZH_CFL_R2_18_BLIND_PRIVATE",
            "source_origin": "independently authored Mandarin learner essays",
            "tag": "r2.18",
        },
        {
            "annotation_provenance": "direct manual annotation",
            "commit_sha1": "72dbb27668c13daa1fb456a14af823d445dc4c3a",
            "data_file": {
                "git_blob_sha1": "7eec8e5620e9233a7df331668a4a9823d23fffc0",
                "main_session_content_reads": 0,
                "relative_path": "zh_hk-ud-test.conllu",
                "size_bytes": 942232,
                "upstream_split": "test",
            },
            "genre": "film-subtitles-and-legislative-proceedings",
            "language": "zh",
            "license_evidence": {
                "git_blob_sha1": "9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846",
                "relative_path": "LICENSE.txt",
                "sha256": "899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79",
                "size_bytes": 202,
            },
            "license_id": "CC-BY-SA-4.0",
            "parallel": 1,
            "readme_evidence": {
                "git_blob_sha1": "39d3f8cdd87079bb3ab6733f4e685f839325d6be",
                "relative_path": "README.md",
                "sha256": "9cc01e2a1cdfc9baa9a7a34a8ce7ce68a7fc2a407c59ad7d2505064aee3c72b3",
                "size_bytes": 5228,
            },
            "repository_url": "https://github.com/UniversalDependencies/UD_Chinese-HK",
            "script": "Traditional Chinese",
            "snapshot_id": "ud-zh-hk-r2.18",
            "source_key": "UD_ZH_HK_R2_18_BLIND_PRIVATE",
            "source_origin": "independent student films and Hong Kong legislative proceedings",
            "tag": "r2.18",
        },
    )


def blind_private_source_specs() -> tuple[dict[str, object], ...]:
    """Return detached source spec dictionaries for owner tooling."""
    return tuple({**row} for row in _source_specs())


def build_blind_private_source_extension_manifest() -> dict[str, object]:
    """Build the payload-free, private-owner-only extension manifest."""
    return {
        "artifact_kind": "PH2_D03_V2_BLIND_PRIVATE_SOURCE_EXTENSION",
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_VERSION,
        "candidate_evaluation_runs": 0,
        "development_authorized": 0,
        "formal_private_evaluation_runs": 0,
        "format_version": 1,
        "main_session_conllu_payload_reads": 0,
        "next_action": "RESTART_ISOLATED_BLIND_PRIVATE_OWNER",
        "old_private_freeze_commit": OLD_PRIVATE_FREEZE_COMMIT,
        "old_private_nonoverlap_basis": (
            "NEW_SOURCE_IDENTITIES_ABSENT_FROM_FROZEN_PARENT_SCHEMA"
        ),
        "old_private_payload_reads": 0,
        "parent_schema_sha256": PARENT_SCHEMA_SHA256,
        "parent_source_keys": list(PARENT_SOURCE_KEYS),
        "private_owner_authorized": 1,
        "scope": "PH2-D03-V2-W02-SUCCESSOR-V2-BLIND-PRIVATE-OWNER-ONLY",
        "shadow_authorized": 0,
        "sources": [dict(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_APPROVED",
        "training_authorized": 0,
    }


def read_blind_private_source_extension_manifest(
        repository_root: str | Path,
        ) -> dict[str, object]:
    """Read the canonical public manifest and reject any metadata drift."""
    root = Path(repository_root).resolve()
    relative = Path(*PurePosixPath(BLIND_PRIVATE_SOURCE_EXTENSION_PATH).parts)
    value = read_canonical_object(root / relative)
    expected = build_blind_private_source_extension_manifest()
    if value != expected:
        raise BlindPrivateSourceExtensionError("blind private source manifest drift")
    schema_path = root / Path(*PurePosixPath(PARENT_SCHEMA_PATH).parts)
    schema_payload = schema_path.read_bytes()
    if (len(schema_payload) != PARENT_SCHEMA_SIZE_BYTES
            or hashlib.sha256(schema_payload).hexdigest() != PARENT_SCHEMA_SHA256
            or tuple(V2_SOURCE_KEYS) != PARENT_SOURCE_KEYS):
        raise BlindPrivateSourceExtensionError("frozen parent schema drift")
    return value


def _validate_source_span(value: object) -> None:
    """Validate the inherited v2 SourceRef span without changing old schema."""
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionError("private SourceRef span fields drift")
    if value["locator_kind"] != "sentence":
        raise BlindPrivateSourceExtensionError("private SourceRef must use sentence locator")
    if not isinstance(value["locator_value"], str) or not value["locator_value"]:
        raise BlindPrivateSourceExtensionError("private SourceRef locator is empty")
    if (type(value["span_start"]) is not int
            or type(value["span_end"]) is not int
            or value["span_start"] < 0
            or value["span_end"] < value["span_start"]):
        raise BlindPrivateSourceExtensionError("private SourceRef span is invalid")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="private document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="private entity cluster")


def validate_blind_private_source_ref(value: dict[str, Any]) -> SourceRefRecord:
    """Validate an extension SourceRef while leaving the frozen v2 schema intact."""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionError("private SourceRef fields drift")
    source_key = value.get("source_key")
    specs = {row["source_key"]: row for row in _source_specs()}
    spec = specs.get(source_key)
    if spec is None:
        raise BlindPrivateSourceExtensionError("source is not in private extension")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionError("private SourceRef is not canonical")
    if (record.format_version != V2_FORMAT_VERSION
            or record.schema_version != V2_SCHEMA_VERSION
            or record.course_version != V2_COURSE_VERSION):
        raise BlindPrivateSourceExtensionError("private SourceRef version drift")
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    expected_checksum = "sha1:" + str(data_file["git_blob_sha1"])
    if (record.snapshot_id != spec["snapshot_id"]
            or record.revision_id != spec["commit_sha1"]
            or record.official_url != spec["repository_url"]
            or record.license_id != spec["license_id"]
            or record.redistribution_policy != "PUBLIC"
            or record.upstream_checksum != expected_checksum
            or not record.source_identity.startswith(f"{source_key}:sentence:")):
        raise BlindPrivateSourceExtensionError("private SourceRef provenance drift")
    _validate_source_span(value["source_span"])
    return record


def validate_blind_private_owner_record(value: dict[str, Any]) -> object:
    """Use the extension only for its two SourceRefs; preserve old validation otherwise."""
    if (isinstance(value, dict)
            and value.get("record_kind") == "source_ref"
            and value.get("source_key") in {
                row["source_key"] for row in _source_specs()
            }):
        return validate_blind_private_source_ref(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_VERSION",
    "BlindPrivateSourceExtensionError",
    "blind_private_source_specs",
    "build_blind_private_source_extension_manifest",
    "read_blind_private_source_extension_manifest",
    "validate_blind_private_owner_record",
    "validate_blind_private_source_ref",
]
