"""为 W02 R6 冻结 TueCL 古汉语 token-span 盲测来源。"""
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
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH,
    BLIND_PRIVATE_SOURCE_EXTENSION_V6_VERSION,
    KYOTO_REMAINDER_SOURCE_KEY,
    build_blind_private_source_extension_v6_manifest,
    read_blind_private_source_extension_v6_manifest,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import (
    SOURCE_REF_FIELDS,
    V2_COURSE_VERSION,
    V2_FORMAT_VERSION,
    V2_SCHEMA_VERSION,
    validate_v2_record,
)


BLIND_PRIVATE_SOURCE_EXTENSION_V7_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_blind_private_source_extension_v7.json"
)
BLIND_PRIVATE_SOURCE_EXTENSION_V7_VERSION = (
    "PH2-D03-V2-BLIND-PRIVATE-SOURCE-EXTENSION-V7"
)
PARENT_EXTENSION_V6_CODE_PATH = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_blind_private_source_extension_v6.py"
)
PARENT_EXTENSION_V6_CODE_SIZE_BYTES = 17_890
PARENT_EXTENSION_V6_CODE_SHA256 = (
    "c931a57db70d60456290d1e0457b4b1aa215cecefa89e6c0616987afc1e96876"
)
PARENT_EXTENSION_V6_MANIFEST_SIZE_BYTES = 8_417
PARENT_EXTENSION_V6_MANIFEST_SHA256 = (
    "55853ce32be99ca955209957b86eab569a58ec5dc0f18c9c38e7d9a8357b6535"
)
TUECL_SOURCE_KEY = "UD_LZH_TUECL_R2_18_TOKEN_SPAN_BLIND_PRIVATE"
TUECL_SNAPSHOT_ID = "ud-lzh-tuecl-r2.18-test-token-span-r6"
TUECL_COMMIT_SHA1 = "0d35ec4b78bba618ff621b63c57fe9542ab61240"
TUECL_DATA_BLOB_SHA1 = "9b93e591c7747758badff15051de31fb465a2cd0"
TUECL_DATA_SHA256 = (
    "596d6b22837e0e5bc72471dc6d10d80029da276c4e3298d89bfd4a1c1727fa2a"
)
TUECL_DATA_SIZE_BYTES = 59_420
TUECL_SENTENCE_COUNT = 100
TUECL_TOKEN_COUNT = 648
TUECL_LICENSE_SHA256 = (
    "899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79"
)
TUECL_README_SHA256 = (
    "b2571e77f61f480ea5ba938d63dd5c9cea7730521541dea251a8a25c3fb3bb9b"
)
TUECL_STATS_SHA256 = (
    "b181d7d125048342ad7faab02ab92b81526d259c83a06a6bc4d29d28be777457"
)
TUECL_FEASIBILITY_METADATA_SHA256 = (
    "5c027695c9c46e3763a1a91dc6ff126b69730c670650459b0f8a0e4c4f4c37e8"
)
TUECL_FEASIBILITY_SELECTION_COMMITMENT = (
    "812658e2b8fa1eaf925d4dfd2a101ee80dda2e6c1854fc0c4238cf200504cf3a"
)
TUECL_FEASIBILITY_CASE_TRANSPORT_SHA256 = (
    "90d9bf4cb41959ed123bcec24ea2f86a99f30167baa7dd2ca0fe11a2c10e0d25"
)
TUECL_FEASIBILITY_CASE_CONTENT_SHA256 = (
    "4337a727368351d5e2f63cbb94c4759f9e0286bfe118e41c84ed685f19b342a3"
)
R5_CAPABILITY_FAIL_AGGREGATE_SHA256 = (
    "cc5b25e3c8c9f35fca20efc882638f53c0ab0c80713b50dab2d60c76cb7c80d1"
)
V4_ARTIFACT_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_artifact_receipt_v1.json"
)
V4_ARTIFACT_RECEIPT_SIZE_BYTES = 2_574
V4_ARTIFACT_RECEIPT_SHA256 = (
    "12fb6b42e3e293b6639996996a59fe09bf8368c85f4e3c372426b017e72007d8"
)
V4_PUBLIC_PROBE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v4_public_probe_v1.json"
)
V4_PUBLIC_PROBE_SIZE_BYTES = 2_444
V4_PUBLIC_PROBE_SHA256 = (
    "7c64cdf6d9b25c1fb45488445018218eab69c2e4da62fae862b3eb97a519266c"
)


# object-model: exception
class BlindPrivateSourceExtensionV7Error(DatasetContractError):
    """V7 来源、冻结依赖或 token-span SourceRef 漂移。"""


def _source_specs() -> tuple[dict[str, object], ...]:
    """返回唯一获准用于 R6 的 TueCL 来源声明。"""
    return ({
        "annotation_provenance": (
            "manual native Classical Chinese lemma, UPOS and FEATS in "
            "Universal Dependencies TueCL"),
        "commit_sha1": TUECL_COMMIT_SHA1,
        "data_file": {
            "git_blob_sha1": TUECL_DATA_BLOB_SHA1,
            "main_session_content_reads": 0,
            "relative_path": "lzh_tuecl-ud-test.conllu",
            "sha256": TUECL_DATA_SHA256,
            "size_bytes": TUECL_DATA_SIZE_BYTES,
            "upstream_split": "test",
        },
        "genre": "classical-literature",
        "language": "lzh",
        "license_evidence": {
            "relative_path": "LICENSE.txt",
            "sha256": TUECL_LICENSE_SHA256,
            "size_bytes": 202,
        },
        "license_id": "CC-BY-SA-4.0",
        "owner_filter": {
            "case_count": 500,
            "label_imputation_authorized": 0,
            "locator_kind": "token_span",
            "real_sentence_cluster_count": 84,
            "selection_commitment": TUECL_FEASIBILITY_SELECTION_COMMITMENT,
            "sentence_count": TUECL_SENTENCE_COUNT,
            "token_count": TUECL_TOKEN_COUNT,
        },
        "parallel": 0,
        "readme_evidence": {
            "relative_path": "README.md",
            "sha256": TUECL_README_SHA256,
            "size_bytes": 1_060,
        },
        "repository_url": (
            "https://github.com/UniversalDependencies/"
            "UD_Classical_Chinese-TueCL"),
        "script": "Hant",
        "snapshot_id": TUECL_SNAPSHOT_ID,
        "source_key": TUECL_SOURCE_KEY,
        "source_origin": "TueCL r2.18 test token spans frozen for R6 only",
        "stats_evidence": {
            "relative_path": "stats.xml",
            "sha256": TUECL_STATS_SHA256,
            "size_bytes": 4_815,
        },
        "tag": "r2.18",
    },)


def blind_private_source_specs_v7() -> tuple[dict[str, object], ...]:
    """返回与内部常量解耦的 R6 来源副本。"""
    return tuple(deepcopy(row) for row in _source_specs())


def build_blind_private_source_extension_v7_manifest() -> dict[str, object]:
    """在 V6 后追加 TueCL，并冻结 R5 FAIL 与 feasibility 证据。"""
    parent = deepcopy(build_blind_private_source_extension_v6_manifest())
    consumed = list(parent["consumed_private_source_keys"])
    if KYOTO_REMAINDER_SOURCE_KEY not in consumed:
        consumed.append(KYOTO_REMAINDER_SOURCE_KEY)
    parent.update({
        "artifact_version": BLIND_PRIVATE_SOURCE_EXTENSION_V7_VERSION,
        "consumed_private_source_keys": consumed,
        "feasibility_case_content_sha256":
            TUECL_FEASIBILITY_CASE_CONTENT_SHA256,
        "feasibility_case_transport_sha256":
            TUECL_FEASIBILITY_CASE_TRANSPORT_SHA256,
        "feasibility_metadata_sha256": TUECL_FEASIBILITY_METADATA_SHA256,
        "feasibility_selection_commitment":
            TUECL_FEASIBILITY_SELECTION_COMMITMENT,
        "formal_private_evaluation_runs": 0,
        "main_session_tuecl_content_reads": 0,
        "next_action": "PUBLISH_R6_PUBLIC_FAMILY_PROTOCOL_FREEZE",
        "parent_extension_v6_code_sha256": PARENT_EXTENSION_V6_CODE_SHA256,
        "parent_extension_v6_manifest_sha256":
            PARENT_EXTENSION_V6_MANIFEST_SHA256,
        "previous_r5_capability_result": {
            "aggregate_sha256": R5_CAPABILITY_FAIL_AGGREGATE_SHA256,
            "failed_dimension": "W-02-V2-NEW-CONTENT-MORPHOLOGY",
            "numerator": 13,
            "denominator": 100,
            "status": "FAIL",
        },
        "private_owner_authorized": 1,
        "revision_a_status": "BLOCKED_CONLLU_MORPHOLOGY_MISSING",
        "revision_b_status": "PASS_500_TOKEN_SPAN_CASES_FEASIBLE",
        "scope": "PH2-D03-V2-W02-V4-FIRST-R6-BLIND-PRIVATE-ONLY",
        "source_nonoverlap_basis": "TUECL_REPOSITORY_DISJOINT_FROM_KYOTO",
        "sources": [deepcopy(row) for row in _source_specs()],
        "status": "BLIND_PRIVATE_SOURCE_EXTENSION_V7_APPROVED",
        "v4_artifact_receipt_sha256": V4_ARTIFACT_RECEIPT_SHA256,
        "v4_public_probe_sha256": V4_PUBLIC_PROBE_SHA256,
    })
    return parent


def _repository_file(root: Path, relative: str) -> Path:
    """把公开相对路径约束在仓库真实普通文件内。"""
    pure = PurePosixPath(relative)
    target = (root / Path(*pure.parts)).resolve()
    if (pure.is_absolute() or "\\" in relative or target.is_symlink()
            or not target.is_relative_to(root) or not target.is_file()):
        raise BlindPrivateSourceExtensionV7Error("V7 公开依赖路径非法")
    return target


def _assert_file(
        root: Path, relative: str, size: int, sha256: str) -> Path:
    """核对一个冻结公开依赖的字节身份。"""
    target = _repository_file(root, relative)
    payload = target.read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
        raise BlindPrivateSourceExtensionV7Error("V7 公开依赖发生漂移")
    return target


def read_blind_private_source_extension_v7_manifest(
        repository_root: str | Path) -> dict[str, object]:
    """回读 V7，并逐项复核 V6 与 V4 公共证据。"""
    root = Path(repository_root).resolve()
    value = read_canonical_object(
        _repository_file(root, BLIND_PRIVATE_SOURCE_EXTENSION_V7_PATH))
    if value != build_blind_private_source_extension_v7_manifest():
        raise BlindPrivateSourceExtensionV7Error("V7 manifest 漂移")
    read_blind_private_source_extension_v6_manifest(root)
    _assert_file(
        root, PARENT_EXTENSION_V6_CODE_PATH,
        PARENT_EXTENSION_V6_CODE_SIZE_BYTES, PARENT_EXTENSION_V6_CODE_SHA256)
    _assert_file(
        root, BLIND_PRIVATE_SOURCE_EXTENSION_V6_PATH,
        PARENT_EXTENSION_V6_MANIFEST_SIZE_BYTES,
        PARENT_EXTENSION_V6_MANIFEST_SHA256)
    artifact = read_canonical_object(_assert_file(
        root, V4_ARTIFACT_RECEIPT_PATH, V4_ARTIFACT_RECEIPT_SIZE_BYTES,
        V4_ARTIFACT_RECEIPT_SHA256))
    probe = read_canonical_object(_assert_file(
        root, V4_PUBLIC_PROBE_PATH, V4_PUBLIC_PROBE_SIZE_BYTES,
        V4_PUBLIC_PROBE_SHA256))
    artifact_identity = artifact.get("git_external_artifact")
    metamorphic = probe.get("metamorphic")
    if (artifact.get("status")
            != "W02_MORPHOLOGY_SUCCESSOR_V4_PUBLIC_ARTIFACT_FROZEN"
            or not isinstance(artifact_identity, dict)
            or artifact_identity.get("semantic_sha256")
            != "55a64c12007aaa5b8fc625c0b477d1dd539bb6eabb9bacf90408c624bbc7f332"
            or probe.get("status") != "PASS"
            or not isinstance(metamorphic, dict)
            or metamorphic.get("language_isolation_candidate_count") != 0
            or probe.get("test_split_content_reads") != 0):
        raise BlindPrivateSourceExtensionV7Error("V4 公共证据状态漂移")
    return value


def _validate_source_span(value: object) -> tuple[int, int]:
    """校验 TueCL 句簇内的唯一 token-span locator。"""
    fields = {
        "document_cluster_key", "entity_graph_cluster_key", "locator_kind",
        "locator_value", "span_end", "span_start",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise BlindPrivateSourceExtensionV7Error("TueCL SourceRef span 字段漂移")
    locator = value.get("locator_value")
    parts = locator.split(":", 3) if isinstance(locator, str) else []
    try:
        sentence = int(parts[1]) if len(parts) == 4 else 0
        token = int(parts[2]) if len(parts) == 4 else 0
    except ValueError:
        sentence = token = 0
    if (value.get("locator_kind") != "token_span"
            or len(parts) != 4 or parts[0] != "test" or not parts[3]
            or sentence < 1 or sentence > TUECL_SENTENCE_COUNT or token < 1
            or type(value.get("span_start")) is not int
            or type(value.get("span_end")) is not int
            or value["span_start"] < 0
            or value["span_end"] <= value["span_start"]):
        raise BlindPrivateSourceExtensionV7Error("TueCL token-span locator 非法")
    StableRecordKey.from_value(
        value["document_cluster_key"], where="TueCL document cluster")
    StableRecordKey.from_value(
        value["entity_graph_cluster_key"], where="TueCL entity cluster")
    return sentence, token


def validate_blind_private_source_ref_v7(
        value: dict[str, Any]) -> SourceRefRecord:
    """只接受冻结 TueCL 来源，拒绝所有旧来源与伪造句数。"""
    if not isinstance(value, dict) or set(value) != set(SOURCE_REF_FIELDS):
        raise BlindPrivateSourceExtensionV7Error("TueCL SourceRef 字段漂移")
    if value.get("source_key") != TUECL_SOURCE_KEY:
        raise BlindPrivateSourceExtensionV7Error("来源未获 R6 授权")
    record = record_from_dict(value)
    if not isinstance(record, SourceRefRecord) or record.to_dict() != value:
        raise BlindPrivateSourceExtensionV7Error("TueCL SourceRef 非 canonical")
    spec = _source_specs()[0]
    data_file = spec["data_file"]
    assert isinstance(data_file, dict)
    if (record.format_version != V2_FORMAT_VERSION
            or record.schema_version != V2_SCHEMA_VERSION
            or record.course_version != V2_COURSE_VERSION
            or record.snapshot_id != TUECL_SNAPSHOT_ID
            or record.revision_id != TUECL_COMMIT_SHA1
            or record.official_url != spec["repository_url"]
            or record.license_id != "CC-BY-SA-4.0"
            or record.redistribution_policy != "PUBLIC"
            or record.upstream_checksum != "sha1:" + TUECL_DATA_BLOB_SHA1):
        raise BlindPrivateSourceExtensionV7Error("TueCL SourceRef 来源身份漂移")
    _validate_source_span(value["source_span"])
    locator = value["source_span"]["locator_value"]
    if record.source_identity != f"{TUECL_SOURCE_KEY}:token_span:{locator}":
        raise BlindPrivateSourceExtensionV7Error("TueCL token-span 身份漂移")
    return record


def validate_blind_private_owner_record_v7(value: dict[str, Any]) -> object:
    """SourceRef 走 V7，其余记录继续走冻结 V2 schema。"""
    if isinstance(value, dict) and value.get("record_kind") == "source_ref":
        return validate_blind_private_source_ref_v7(value)
    return validate_v2_record(value)


__all__ = [
    "BLIND_PRIVATE_SOURCE_EXTENSION_V7_PATH",
    "BLIND_PRIVATE_SOURCE_EXTENSION_V7_VERSION",
    "BlindPrivateSourceExtensionV7Error",
    "TUECL_FEASIBILITY_CASE_CONTENT_SHA256",
    "TUECL_FEASIBILITY_CASE_TRANSPORT_SHA256",
    "TUECL_FEASIBILITY_METADATA_SHA256",
    "TUECL_FEASIBILITY_SELECTION_COMMITMENT",
    "TUECL_SOURCE_KEY",
    "TUECL_SNAPSHOT_ID",
    "blind_private_source_specs_v7",
    "build_blind_private_source_extension_v7_manifest",
    "read_blind_private_source_extension_v7_manifest",
    "validate_blind_private_owner_record_v7",
    "validate_blind_private_source_ref_v7",
]
