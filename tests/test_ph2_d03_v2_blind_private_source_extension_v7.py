"""V7 TueCL token-span 来源合同的专项测试。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v7 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V7_VERSION,
    TUECL_DATA_BLOB_SHA1,
    TUECL_SOURCE_KEY,
    TUECL_SNAPSHOT_ID,
    BlindPrivateSourceExtensionV7Error,
    blind_private_source_specs_v7,
    read_blind_private_source_extension_v7_manifest,
    validate_blind_private_source_ref_v7,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _source_ref() -> dict[str, object]:
    """构造不含真实私有内容的最小 canonical token-span SourceRef。"""
    locator = "test:1:1:synthetic-case"
    return {
        "artifact_key": [81, 1],
        "attribution": "Universal Dependencies TueCL attribution retained",
        "course_version": 2,
        "dataset_key": [81],
        "format_version": 2,
        "license_id": "CC-BY-SA-4.0",
        "local_sha256": "1" * 64,
        "official_url": (
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-TueCL"),
        "parser_version": 1,
        "record_kind": "source_ref",
        "record_ordinal": 1,
        "redistribution_policy": "PUBLIC",
        "revision_id": "0d35ec4b78bba618ff621b63c57fe9542ab61240",
        "schema_version": 2,
        "snapshot_id": TUECL_SNAPSHOT_ID,
        "source_cluster_key": [81, 2, 1],
        "source_identity": f"{TUECL_SOURCE_KEY}:token_span:{locator}",
        "source_key": TUECL_SOURCE_KEY,
        "source_span": {
            "document_cluster_key": [81, 3, 1],
            "entity_graph_cluster_key": [81, 4, 1],
            "locator_kind": "token_span",
            "locator_value": locator,
            "span_end": 1,
            "span_start": 0,
        },
        "stable_key": [81, 10, 1],
        "upstream_checksum": "sha1:" + TUECL_DATA_BLOB_SHA1,
    }


def test_v7_manifest_and_source_spec_are_frozen() -> None:
    """公开 manifest 必须闭合 V6、V4 与 feasibility 身份。"""
    manifest = read_blind_private_source_extension_v7_manifest(REPOSITORY)
    specs = blind_private_source_specs_v7()
    assert manifest["artifact_version"] == BLIND_PRIVATE_SOURCE_EXTENSION_V7_VERSION
    assert manifest["status"] == "BLIND_PRIVATE_SOURCE_EXTENSION_V7_APPROVED"
    assert manifest["formal_private_evaluation_runs"] == 0
    assert manifest["main_session_tuecl_content_reads"] == 0
    assert len(specs) == 1
    assert specs[0]["source_key"] == TUECL_SOURCE_KEY
    assert specs[0]["owner_filter"]["case_count"] == 500


def test_v7_accepts_one_synthetic_token_span() -> None:
    """合法 token-span 必须保持来源、句簇与字符 span。"""
    record = validate_blind_private_source_ref_v7(_source_ref())
    assert record.source_key == TUECL_SOURCE_KEY
    assert record.source_span.to_value()["locator_kind"] == "token_span"


@pytest.mark.parametrize("mutation", ("source", "kind", "sentence", "identity"))
def test_v7_rejects_old_or_malformed_source_refs(mutation: str) -> None:
    """旧来源、伪句数、错误 locator 和身份错配均须 fail closed。"""
    value = deepcopy(_source_ref())
    span = value["source_span"]
    assert isinstance(span, dict)
    if mutation == "source":
        value["source_key"] = "UD_LZH_KYOTO_R2_18_TEST_REMAINDER_BLIND_PRIVATE"
    elif mutation == "kind":
        span["locator_kind"] = "sentence"
    elif mutation == "sentence":
        span["locator_value"] = "test:101:1:synthetic-case"
    else:
        value["source_identity"] = "mismatch"
    with pytest.raises(BlindPrivateSourceExtensionV7Error):
        validate_blind_private_source_ref_v7(value)
