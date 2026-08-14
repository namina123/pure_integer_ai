"""从四个冻结 source pack 派生 normalization recovery TRAIN 记录。

本模块只做纯记录变换。它把 source pack、source family、source policy、
目标 policy、phrase composition 与 context override 分账，不读取路径、
evaluation、reserve、formal report、candidate 或生产 consumer。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    NORMALIZATION_ICU_LICENSE_ID,
    NORMALIZATION_ICU_SOURCE_PACK_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
    NORMALIZATION_SOURCE_LICENSE_ID,
    NORMALIZATION_SOURCE_PACK_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_source_pack import (
    MEDIAWIKI_COMMIT,
    MEDIAWIKI_CONVERSION_RECORD_KIND,
    MEDIAWIKI_LICENSE_ID,
    NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
    UNIHAN_LICENSE_ID,
    UNIHAN_VARIANT_RECORD_KIND,
    UNIHAN_VERSION,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    derive_icu_successor_observations,
    derive_opencc_successor_observations,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_SOURCE_ROSTER_KIND = (
    "NORMALIZATION_RECOVERY_SOURCE_POLICY_ROSTER_V2")
NORMALIZATION_RECOVERY_OBSERVATION_KIND = (
    "NORMALIZATION_RECOVERY_TRAIN_OBSERVATION_V2")
NORMALIZATION_RECOVERY_GROUP_KIND = (
    "NORMALIZATION_RECOVERY_SOURCE_FAMILY_GROUP_V2")
NORMALIZATION_RECOVERY_COMPOSITION_KIND = (
    "NORMALIZATION_RECOVERY_PHRASE_COMPOSITION_V2")
NORMALIZATION_RECOVERY_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_LEAVE_ONE_POLICY_OUT_V2")

OPENCC_SOURCE_KEY = "OPENCC_T2S"
ICU_SOURCE_KEY = "UNICODE_ICU_HANS_HANT"
UNIHAN_SOURCE_KEY = "UNICODE_UNIHAN_VARIANTS"
MEDIAWIKI_HANS_SOURCE_KEY = "MEDIAWIKI_ZH_TO_HANS"
MEDIAWIKI_CN_SOURCE_KEY = "MEDIAWIKI_ZH_TO_CN"

OPENCC_SOURCE_FAMILY = "OPENCC_PROJECT"
UNICODE_SOURCE_FAMILY = "UNICODE_CONSORTIUM"
WIKIMEDIA_SOURCE_FAMILY = "WIKIMEDIA_PROJECTS"

UNIHAN_SOURCE_POLICY_SCOPE = "UNIHAN_17_0_0_T2S_SOURCE_BEHAVIOR_V1"
MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE = (
    "MEDIAWIKI_CORE_ZH_TO_HANS_SOURCE_BEHAVIOR_V1")
MEDIAWIKI_CN_SOURCE_POLICY_SCOPE = (
    "MEDIAWIKI_CORE_ZH_TO_CN_SOURCE_BEHAVIOR_V1")
RECOVERY_TARGET_POLICY_SCOPE = "ZH_CN_MULTI_SOURCE_AUTHORITY_V2"

GENERIC_T2S_AUTHORITY = "GENERIC_T2S_EVIDENCE"
REGIONAL_ZH_CN_AUTHORITY = "REGIONAL_ZH_CN_EXACT_AUTHORITY"

SOURCE_POLICY_DEFINITIONS = {
    OPENCC_SOURCE_POLICY_SCOPE: {
        "authority_role": GENERIC_T2S_AUTHORITY,
        "license_id": NORMALIZATION_SOURCE_LICENSE_ID,
        "source_family": OPENCC_SOURCE_FAMILY,
        "source_key": OPENCC_SOURCE_KEY,
    },
    ICU_SOURCE_POLICY_SCOPE: {
        "authority_role": GENERIC_T2S_AUTHORITY,
        "license_id": NORMALIZATION_ICU_LICENSE_ID,
        "source_family": UNICODE_SOURCE_FAMILY,
        "source_key": ICU_SOURCE_KEY,
    },
    UNIHAN_SOURCE_POLICY_SCOPE: {
        "authority_role": GENERIC_T2S_AUTHORITY,
        "license_id": UNIHAN_LICENSE_ID,
        "source_family": UNICODE_SOURCE_FAMILY,
        "source_key": UNIHAN_SOURCE_KEY,
    },
    MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE: {
        "authority_role": GENERIC_T2S_AUTHORITY,
        "license_id": MEDIAWIKI_LICENSE_ID,
        "source_family": WIKIMEDIA_SOURCE_FAMILY,
        "source_key": MEDIAWIKI_HANS_SOURCE_KEY,
    },
    MEDIAWIKI_CN_SOURCE_POLICY_SCOPE: {
        "authority_role": REGIONAL_ZH_CN_AUTHORITY,
        "license_id": MEDIAWIKI_LICENSE_ID,
        "source_family": WIKIMEDIA_SOURCE_FAMILY,
        "source_key": MEDIAWIKI_CN_SOURCE_KEY,
    },
}
SOURCE_POLICY_SCOPES = tuple(sorted(SOURCE_POLICY_DEFINITIONS))
SOURCE_FAMILIES = tuple(sorted({
    str(item["source_family"]) for item in SOURCE_POLICY_DEFINITIONS.values()
}))

GENERIC_RESOLUTION_KINDS = (
    "CROSS_FAMILY_CONSENSUS",
    "INTRA_FAMILY_CONFLICT",
    "NO_GENERIC_AUTHORITY",
    "SINGLE_FAMILY_DEFER",
    "SOURCE_FAMILY_CONFLICT",
)
TARGET_RESOLUTION_KINDS = (
    "CROSS_FAMILY_CONSENSUS",
    "NO_TARGET_AUTHORITY",
    "REGIONAL_EXACT_AUTHORITY",
)
COMPOSITION_QUALIFICATIONS = (
    "COMPOSITION_SUPPORT",
    "EXPLICIT_OVERRIDE",
    "NO_COMPOSITION_EVIDENCE",
    "PARTIAL_COMPOSITION",
)


def _sha256(payload: bytes) -> str:
    """返回规范 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _integer(value: object, *, minimum: int = 0) -> bool:
    """只接受不小于下界的真整数，拒绝 bool 冒充计数。"""
    return type(value) is int and value >= minimum


def _manifest_identity(
        manifest: dict[str, object],
        *,
        expected_kind: str,
        label: str,
        ) -> str:
    """核验 source manifest 的 kind 与外部 SHA identity。"""
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind") != expected_kind):
        raise BroadQaExternalDataError(f"{label} source manifest kind 漂移")
    return _sha_value(manifest.get("manifest_sha256"), label=f"{label} manifest")


def _physical_range(commitment: dict[str, object], *, label: str) -> None:
    """核验单条来源记录的物理字节、行号与行摘要。"""
    if (not _integer(commitment.get("byte_start"))
            or not _integer(commitment.get("byte_end"), minimum=1)
            or commitment["byte_end"] <= commitment["byte_start"]
            or not _integer(commitment.get("line_ordinal"), minimum=1)):
        raise BroadQaExternalDataError(f"{label} 物理范围非法")
    _sha_value(commitment.get("line_sha256"), label=f"{label} line SHA")


def _validate_source_commitment(
        commitment: object,
        *,
        source_policy_scope: str,
        input_text: str,
        expected_output: str,
        ) -> None:
    """按 policy 严格核验来源承诺，不把来源名当作证据本身。"""
    if not isinstance(commitment, dict):
        raise BroadQaExternalDataError("recovery source commitment 非对象")
    if source_policy_scope == OPENCC_SOURCE_POLICY_SCOPE:
        expected_fields = {
            "byte_end", "byte_start", "file_sha256", "line_ordinal",
            "line_sha256", "relative_path",
        }
        relative_path = commitment.get("relative_path")
        if (set(commitment) != expected_fields
                or not isinstance(relative_path, str)
                or relative_path not in {
                    "dictionary/TSCharacters.txt",
                    "dictionary/TSPhrases.txt",
                }
                or commitment.get("file_sha256")
                != NORMALIZATION_SOURCE_FILES[str(relative_path)]["sha256"]):
            raise BroadQaExternalDataError(
                "recovery OpenCC source commitment 漂移")
        _physical_range(commitment, label="recovery OpenCC")
        return
    if source_policy_scope == ICU_SOURCE_POLICY_SCOPE:
        expected_fields = {
            "byte_end", "byte_start", "line_end_ordinal",
            "line_start_ordinal", "physical_lines", "statement_sha256",
        }
        physical_lines = commitment.get("physical_lines")
        if (set(commitment) != expected_fields
                or not _integer(commitment.get("byte_start"))
                or not _integer(commitment.get("byte_end"), minimum=1)
                or commitment["byte_end"] <= commitment["byte_start"]
                or not _integer(
                    commitment.get("line_start_ordinal"), minimum=1)
                or not _integer(
                    commitment.get("line_end_ordinal"), minimum=1)
                or commitment["line_end_ordinal"]
                < commitment["line_start_ordinal"]
                or not isinstance(physical_lines, list)
                or not physical_lines):
            raise BroadQaExternalDataError(
                "recovery ICU source commitment 漂移")
        expected_line_fields = {
            "byte_end", "byte_start", "line_ordinal", "line_sha256"}
        for line in physical_lines:
            if not isinstance(line, dict) or set(line) != expected_line_fields:
                raise BroadQaExternalDataError(
                    "recovery ICU physical line schema 漂移")
            _physical_range(line, label="recovery ICU physical line")
        if (physical_lines[0]["byte_start"] != commitment["byte_start"]
                or physical_lines[-1]["byte_end"] != commitment["byte_end"]
                or physical_lines[0]["line_ordinal"]
                != commitment["line_start_ordinal"]
                or physical_lines[-1]["line_ordinal"]
                != commitment["line_end_ordinal"]):
            raise BroadQaExternalDataError(
                "recovery ICU source commitment 外层范围漂移")
        _sha_value(
            commitment.get("statement_sha256"),
            label="recovery ICU statement SHA")
        return
    if source_policy_scope == UNIHAN_SOURCE_POLICY_SCOPE:
        expected_fields = {
            "byte_end", "byte_start", "line_ordinal", "line_sha256",
            "property_name", "source_codepoint", "source_uplus", "targets",
        }
        targets = commitment.get("targets")
        if (set(commitment) != expected_fields
                or len(input_text) != 1 or len(expected_output) != 1
                or commitment.get("property_name") != "kSimplifiedVariant"
                or commitment.get("source_codepoint") != ord(input_text)
                or commitment.get("source_uplus")
                != f"U+{ord(input_text):04X}"
                or not isinstance(targets, list) or len(targets) != 1):
            raise BroadQaExternalDataError(
                "recovery Unihan source commitment 漂移")
        target = targets[0]
        if (not isinstance(target, dict)
                or set(target) != {"codepoint", "source_tags", "text", "uplus"}
                or target.get("codepoint") != ord(expected_output)
                or target.get("source_tags") != []
                or target.get("text") != expected_output
                or target.get("uplus") != f"U+{ord(expected_output):04X}"):
            raise BroadQaExternalDataError(
                "recovery Unihan target commitment 漂移")
        _physical_range(commitment, label="recovery Unihan")
        return
    expected_table = {
        MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE: "ZH_TO_HANS",
        MEDIAWIKI_CN_SOURCE_POLICY_SCOPE: "ZH_TO_CN",
    }.get(source_policy_scope)
    expected_fields = {
        "byte_end", "byte_start", "line_ordinal", "line_sha256",
        "table_name",
    }
    if (expected_table is None or set(commitment) != expected_fields
            or commitment.get("table_name") != expected_table):
        raise BroadQaExternalDataError(
            "recovery MediaWiki source commitment 漂移")
    _physical_range(commitment, label="recovery MediaWiki")


def derive_normalization_recovery_source_roster(
        *,
        opencc_manifest: dict[str, object],
        icu_manifest: dict[str, object],
        successor_manifest: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """冻结三个 source pack、三个 family 与五个 policy 的对应关系。"""
    opencc_sha = _manifest_identity(
        opencc_manifest, expected_kind=NORMALIZATION_SOURCE_PACK_KIND,
        label="OpenCC")
    icu_sha = _manifest_identity(
        icu_manifest, expected_kind=NORMALIZATION_ICU_SOURCE_PACK_KIND,
        label="ICU")
    successor_sha = _manifest_identity(
        successor_manifest,
        expected_kind=NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
        label="Unihan/MediaWiki")
    source_values = {
        OPENCC_SOURCE_POLICY_SCOPE: {
            "source_pack_manifest_sha256": opencc_sha,
            "source_revision": str(opencc_manifest.get("package_version", "")),
        },
        ICU_SOURCE_POLICY_SCOPE: {
            "source_pack_manifest_sha256": icu_sha,
            "source_revision": str(icu_manifest.get("repository_commit", "")),
        },
        UNIHAN_SOURCE_POLICY_SCOPE: {
            "source_pack_manifest_sha256": successor_sha,
            "source_revision": UNIHAN_VERSION,
        },
        MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE: {
            "source_pack_manifest_sha256": successor_sha,
            "source_revision": MEDIAWIKI_COMMIT,
        },
        MEDIAWIKI_CN_SOURCE_POLICY_SCOPE: {
            "source_pack_manifest_sha256": successor_sha,
            "source_revision": MEDIAWIKI_COMMIT,
        },
    }
    records = []
    for policy in SOURCE_POLICY_SCOPES:
        definition = SOURCE_POLICY_DEFINITIONS[policy]
        identity = {
            "authority_role": definition["authority_role"],
            "license_id": definition["license_id"],
            "source_family": definition["source_family"],
            "source_key": definition["source_key"],
            "source_pack_manifest_sha256": source_values[policy][
                "source_pack_manifest_sha256"],
            "source_policy_scope": policy,
            "source_revision": source_values[policy]["source_revision"],
        }
        records.append({
            **identity,
            "format_version": 2,
            "record_kind": NORMALIZATION_RECOVERY_SOURCE_ROSTER_KIND,
            "roster_id": _sha256(canonical_json_bytes(identity)),
            "split": "TRAIN_SOURCE",
        })
    result = tuple(records)
    if (len(result) != 5
            or len({item["roster_id"] for item in result}) != len(result)
            or len({item["source_policy_scope"] for item in result}) != 5
            or len({item["source_family"] for item in result}) != 3):
        raise BroadQaExternalDataError("normalization recovery source roster 漂移")
    return result


def _roster_by_policy(
        roster: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """核验 roster schema 并按 policy 建立只读索引。"""
    result = {}
    expected_fields = {
        "authority_role", "format_version", "license_id", "record_kind",
        "roster_id", "source_family", "source_key",
        "source_pack_manifest_sha256", "source_policy_scope",
        "source_revision", "split",
    }
    for record in roster:
        if (not isinstance(record, dict) or set(record) != expected_fields
                or record.get("format_version") != 2
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_SOURCE_ROSTER_KIND
                or record.get("split") != "TRAIN_SOURCE"):
            raise BroadQaExternalDataError("recovery roster schema 漂移")
        policy = record.get("source_policy_scope")
        if policy not in SOURCE_POLICY_DEFINITIONS or policy in result:
            raise BroadQaExternalDataError("recovery roster policy 漂移")
        definition = SOURCE_POLICY_DEFINITIONS[str(policy)]
        identity = {key: record[key] for key in (
            "authority_role", "license_id", "source_family", "source_key",
            "source_pack_manifest_sha256", "source_policy_scope",
            "source_revision")}
        if (any(record[key] != definition[key] for key in (
                "authority_role", "license_id", "source_family", "source_key"))
                or not isinstance(record["source_revision"], str)
                or not record["source_revision"]
                or record["roster_id"] != _sha256(canonical_json_bytes(identity))):
            raise BroadQaExternalDataError("recovery roster identity 漂移")
        _sha_value(record["source_pack_manifest_sha256"], label="roster source")
        result[str(policy)] = record
    if set(result) != set(SOURCE_POLICY_SCOPES):
        raise BroadQaExternalDataError("recovery roster policy inventory 漂移")
    return result


def _observation(
        *,
        roster_record: dict[str, object],
        input_text: str,
        expected_output: str,
        source_commitment: dict[str, object],
        target_variant_count: int,
        selected_target_variant_ordinal: int,
        ) -> dict[str, object]:
    """构造绑定 roster 与物理来源承诺的统一 TRAIN observation。"""
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(expected_output, str) or not expected_output
            or not isinstance(source_commitment, dict)
            or not _integer(target_variant_count, minimum=1)
            or not _integer(selected_target_variant_ordinal)
            or selected_target_variant_ordinal >= target_variant_count):
        raise BroadQaExternalDataError("recovery observation 输入非法")
    identity = {
        "expected_output": expected_output,
        "input_text": input_text,
        "source_commitment": source_commitment,
        "source_pack_manifest_sha256": roster_record[
            "source_pack_manifest_sha256"],
        "source_policy_scope": roster_record["source_policy_scope"],
    }
    return {
        **identity,
        "authority_role": roster_record["authority_role"],
        "evidence_source_scope": (
            f"{roster_record['source_key']}_SOURCE_PACK_SHA256:"
            f"{roster_record['source_pack_manifest_sha256']}"),
        "format_version": 2,
        "input_scalar_count": len(input_text),
        "mapping_kind": (
            "CHARACTER_INPUT" if len(input_text) == 1 else "PHRASE_INPUT"),
        "observation_id": _sha256(canonical_json_bytes(identity)),
        "output_scalar_count": len(expected_output),
        "record_kind": NORMALIZATION_RECOVERY_OBSERVATION_KIND,
        "selected_target_variant_ordinal": selected_target_variant_ordinal,
        "source_family": roster_record["source_family"],
        "source_key": roster_record["source_key"],
        "source_roster_id": roster_record["roster_id"],
        "split": "TRAIN_SOURCE",
        "target_variant_count": target_variant_count,
    }


def _adapt_successor_observations(
        values: tuple[dict[str, object], ...],
        *,
        roster_record: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """把既有 OpenCC/ICU source adapter 输出提升到 recovery v2 schema。"""
    result = tuple(_observation(
        roster_record=roster_record,
        input_text=str(item["input_text"]),
        expected_output=str(item["expected_output"]),
        source_commitment=dict(item["source_commitment"]),
        target_variant_count=int(item["target_variant_count"]),
        selected_target_variant_ordinal=int(
            item["selected_target_variant_ordinal"]),
    ) for item in values)
    return tuple(sorted(result, key=lambda item: str(item["observation_id"])))


def derive_opencc_recovery_observations(
        *,
        roster: tuple[dict[str, object], ...],
        character_payload: bytes,
        phrase_payload: bytes,
        ) -> tuple[dict[str, object], ...]:
    """从冻结 OpenCC 字典派生统一 recovery observations。"""
    by_policy = _roster_by_policy(roster)
    record = by_policy[OPENCC_SOURCE_POLICY_SCOPE]
    values = derive_opencc_successor_observations(
        source_pack_manifest_sha256=str(
            record["source_pack_manifest_sha256"]),
        character_payload=character_payload,
        phrase_payload=phrase_payload,
    )
    return _adapt_successor_observations(values, roster_record=record)


def derive_icu_recovery_observations(
        *,
        roster: tuple[dict[str, object], ...],
        rules: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从冻结 ICU reverse-eligible rules 派生统一 observations。"""
    by_policy = _roster_by_policy(roster)
    record = by_policy[ICU_SOURCE_POLICY_SCOPE]
    values = derive_icu_successor_observations(
        source_pack_manifest_sha256=str(
            record["source_pack_manifest_sha256"]),
        rules=rules,
    )
    return _adapt_successor_observations(values, roster_record=record)


def derive_unihan_recovery_observations(
        *,
        roster: tuple[dict[str, object], ...],
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从单 target、无来源限定的 Unihan 简化边派生 observations。"""
    by_policy = _roster_by_policy(roster)
    roster_record = by_policy[UNIHAN_SOURCE_POLICY_SCOPE]
    values = []
    for item in records:
        if (not isinstance(item, dict)
                or item.get("record_kind") != UNIHAN_VARIANT_RECORD_KIND):
            raise BroadQaExternalDataError("Unihan recovery record schema 漂移")
        if item.get("t2s_unambiguous_eligible") != 1:
            continue
        commitment = {key: item[key] for key in (
            "byte_end", "byte_start", "line_ordinal", "line_sha256",
            "property_name", "source_codepoint", "source_uplus", "targets")}
        targets = item.get("targets")
        if not isinstance(targets, list) or len(targets) != 1:
            raise BroadQaExternalDataError("Unihan recovery target 漂移")
        values.append(_observation(
            roster_record=roster_record,
            input_text=str(item["t2s_input"]),
            expected_output=str(item["t2s_expected_output"]),
            source_commitment=commitment,
            target_variant_count=1,
            selected_target_variant_ordinal=0,
        ))
    result = tuple(sorted(values, key=lambda item: str(item["observation_id"])))
    if not result or len({item["observation_id"] for item in result}) != len(result):
        raise BroadQaExternalDataError("Unihan recovery observation identity 漂移")
    return result


def derive_mediawiki_recovery_observations(
        *,
        roster: tuple[dict[str, object], ...],
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从 MediaWiki HANS 与 CN 两张表派生 generic/regional observations。"""
    by_policy = _roster_by_policy(roster)
    table_policy = {
        "ZH_TO_HANS": MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE,
        "ZH_TO_CN": MEDIAWIKI_CN_SOURCE_POLICY_SCOPE,
    }
    values = []
    for item in records:
        if (not isinstance(item, dict)
                or item.get("record_kind")
                != MEDIAWIKI_CONVERSION_RECORD_KIND):
            raise BroadQaExternalDataError(
                "MediaWiki recovery record schema 漂移")
        policy = table_policy.get(item.get("table_name"))
        if policy is None:
            continue
        commitment = {key: item[key] for key in (
            "byte_end", "byte_start", "line_ordinal", "line_sha256",
            "table_name")}
        values.append(_observation(
            roster_record=by_policy[policy],
            input_text=str(item["input_text"]),
            expected_output=str(item["expected_output"]),
            source_commitment=commitment,
            target_variant_count=1,
            selected_target_variant_ordinal=0,
        ))
    result = tuple(sorted(values, key=lambda item: str(item["observation_id"])))
    if not result or len({item["observation_id"] for item in result}) != len(result):
        raise BroadQaExternalDataError(
            "MediaWiki recovery observation identity 漂移")
    return result


def _validate_observation(
        observation: dict[str, object],
        roster_by_policy: dict[str, dict[str, object]],
        ) -> None:
    """核验统一 observation 的 roster binding、schema 与 identity。"""
    expected_fields = {
        "authority_role", "evidence_source_scope", "expected_output",
        "format_version", "input_scalar_count", "input_text", "mapping_kind",
        "observation_id", "output_scalar_count", "record_kind",
        "selected_target_variant_ordinal", "source_commitment", "source_family",
        "source_key", "source_pack_manifest_sha256", "source_policy_scope",
        "source_roster_id", "split", "target_variant_count",
    }
    if not isinstance(observation, dict) or set(observation) != expected_fields:
        raise BroadQaExternalDataError("recovery observation schema 漂移")
    policy = observation.get("source_policy_scope")
    roster = roster_by_policy.get(str(policy))
    input_text = observation.get("input_text")
    expected_output = observation.get("expected_output")
    if (roster is None or not isinstance(input_text, str) or not input_text
            or not isinstance(expected_output, str) or not expected_output
            or any(0xD800 <= ord(item) <= 0xDFFF
                   for item in input_text + expected_output)
            or observation.get("format_version") != 2
            or observation.get("record_kind")
            != NORMALIZATION_RECOVERY_OBSERVATION_KIND
            or observation.get("split") != "TRAIN_SOURCE"
            or not _integer(
                observation.get("target_variant_count"), minimum=1)
            or not _integer(
                observation.get("selected_target_variant_ordinal"))
            or observation["selected_target_variant_ordinal"]
            >= observation["target_variant_count"]
            or observation.get("input_scalar_count") != len(input_text)
            or observation.get("output_scalar_count") != len(expected_output)
            or observation.get("mapping_kind") != (
                "CHARACTER_INPUT" if len(input_text) == 1 else "PHRASE_INPUT")
            or any(observation[key] != roster[key] for key in (
                "authority_role", "source_family", "source_key",
                "source_pack_manifest_sha256"))
            or observation.get("source_roster_id") != roster["roster_id"]):
        raise BroadQaExternalDataError("recovery observation 字段漂移")
    expected_evidence_scope = (
        f"{roster['source_key']}_SOURCE_PACK_SHA256:"
        f"{roster['source_pack_manifest_sha256']}")
    if observation.get("evidence_source_scope") != expected_evidence_scope:
        raise BroadQaExternalDataError(
            "recovery observation evidence source scope 漂移")
    _validate_source_commitment(
        observation.get("source_commitment"),
        source_policy_scope=str(policy),
        input_text=input_text,
        expected_output=expected_output,
    )
    identity = {key: observation[key] for key in (
        "expected_output", "input_text", "source_commitment",
        "source_pack_manifest_sha256", "source_policy_scope")}
    if observation.get("observation_id") != _sha256(canonical_json_bytes(identity)):
        raise BroadQaExternalDataError("recovery observation identity 漂移")


def _source_policy_output(observation: dict[str, object]) -> dict[str, object]:
    """投影 group 内的 policy/family/authority 输出。"""
    return {key: observation[key] for key in (
        "authority_role", "expected_output", "observation_id", "source_family",
        "source_key", "source_policy_scope")}


def resolve_normalization_recovery_group_authority(
        ordered: list[dict[str, object]],
        ) -> dict[str, object]:
    """按 family 一票与区域精确 authority 解析一个 exact-input 集合。"""
    policies = [str(item["source_policy_scope"]) for item in ordered]
    if len(set(policies)) != len(policies):
        raise BroadQaExternalDataError("recovery group 同 policy 重复 input")
    generic = [item for item in ordered
               if item["authority_role"] == GENERIC_T2S_AUTHORITY]
    regional = [item for item in ordered
                if item["authority_role"] == REGIONAL_ZH_CN_AUTHORITY]
    if len(regional) > 1:
        raise BroadQaExternalDataError("recovery regional authority 重复")
    generic_by_family: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for item in generic:
        generic_by_family[str(item["source_family"])].append(item)
    family_outputs = []
    family_votes = []
    intra_family_conflict = False
    for family in sorted(generic_by_family):
        values = generic_by_family[family]
        outputs = sorted({str(item["expected_output"]) for item in values})
        conflicted = int(len(outputs) != 1)
        intra_family_conflict |= bool(conflicted)
        record = {
            "family_conflicted": conflicted,
            "observation_ids": sorted(
                str(item["observation_id"]) for item in values),
            "outputs": outputs,
            "source_family": family,
            "source_policy_scopes": sorted(
                str(item["source_policy_scope"]) for item in values),
        }
        family_outputs.append(record)
        if not conflicted:
            family_votes.append((family, outputs[0], record["observation_ids"]))
    vote_outputs = sorted({item[1] for item in family_votes})
    if intra_family_conflict:
        generic_kind = "INTRA_FAMILY_CONFLICT"
    elif not family_votes:
        generic_kind = "NO_GENERIC_AUTHORITY"
    elif len(family_votes) == 1:
        generic_kind = "SINGLE_FAMILY_DEFER"
    elif len(vote_outputs) == 1:
        generic_kind = "CROSS_FAMILY_CONSENSUS"
    else:
        generic_kind = "SOURCE_FAMILY_CONFLICT"
    generic_output = (
        vote_outputs[0] if generic_kind == "CROSS_FAMILY_CONSENSUS" else "")
    generic_ids = sorted(
        observation_id for _, output, ids in family_votes
        if output == generic_output for observation_id in ids)
    if regional:
        target_kind = "REGIONAL_EXACT_AUTHORITY"
        target_output = str(regional[0]["expected_output"])
        target_ids = [str(regional[0]["observation_id"])]
    elif generic_kind == "CROSS_FAMILY_CONSENSUS":
        target_kind = "CROSS_FAMILY_CONSENSUS"
        target_output = generic_output
        target_ids = generic_ids
    else:
        target_kind = "NO_TARGET_AUTHORITY"
        target_output = ""
        target_ids = []
    return {
        "family_outputs": family_outputs,
        "generic_consensus_observation_ids": generic_ids,
        "generic_consensus_output": generic_output,
        "generic_resolution_kind": generic_kind,
        "regional": regional,
        "target_authority_observation_ids": target_ids,
        "target_output": target_output,
        "target_resolution_kind": target_kind,
    }


def _group_record(
        input_text: str,
        observations: list[dict[str, object]],
        ) -> dict[str, object]:
    """按 source family 计 generic 共识，并单列 zh-CN 精确 authority。"""
    ordered = sorted(observations, key=lambda item: (
        str(item["source_policy_scope"]), str(item["observation_id"])))
    resolution = resolve_normalization_recovery_group_authority(ordered)
    regional = resolution["regional"]
    target_kind = str(resolution["target_resolution_kind"])
    target_output = str(resolution["target_output"])
    identity = {
        "input_text": input_text,
        "observation_ids": sorted(
            str(item["observation_id"]) for item in ordered),
        "source_policy_outputs": [_source_policy_output(item) for item in ordered],
    }
    return {
        **identity,
        "eligible_target_policy_scope": (
            RECOVERY_TARGET_POLICY_SCOPE
            if target_kind != "NO_TARGET_AUTHORITY" else ""),
        "family_outputs": resolution["family_outputs"],
        "format_version": 2,
        "generic_consensus_observation_ids": resolution[
            "generic_consensus_observation_ids"],
        "generic_consensus_output": resolution["generic_consensus_output"],
        "generic_resolution_kind": resolution["generic_resolution_kind"],
        "group_id": _sha256(canonical_json_bytes(identity)),
        "input_scalar_count": len(input_text),
        "mapping_kind": (
            "CHARACTER_INPUT" if len(input_text) == 1 else "PHRASE_INPUT"),
        "record_kind": NORMALIZATION_RECOVERY_GROUP_KIND,
        "regional_authority_observation_ids": [
            str(item["observation_id"]) for item in regional],
        "source_family_count": len({
            str(item["source_family"]) for item in ordered}),
        "source_policy_count": len(ordered),
        "target_authority_observation_ids": resolution[
            "target_authority_observation_ids"],
        "target_output": target_output,
        "target_resolution_kind": target_kind,
        "target_rule_is_identity": int(
            bool(target_output) and input_text == target_output),
    }


def derive_normalization_recovery_groups(
        *,
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从五个 policy observation 派生 family-aware exact-input groups。"""
    roster_by_policy = _roster_by_policy(roster)
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError("recovery observations 为空")
    identities = []
    by_input: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for observation in observations:
        _validate_observation(observation, roster_by_policy)
        identities.append(str(observation["observation_id"]))
        by_input[str(observation["input_text"])].append(observation)
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError("recovery observation identity 重复")
    groups = tuple(sorted(
        (_group_record(input_text, values)
         for input_text, values in by_input.items()),
        key=lambda item: str(item["group_id"])))
    if not groups or len({item["group_id"] for item in groups}) != len(groups):
        raise BroadQaExternalDataError("recovery group identity 漂移")
    return groups


def validate_normalization_recovery_observation_inventory(
        *,
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> None:
    """一次性核验 observation inventory，供多次 TRAIN-only 投影复用。"""
    roster_by_policy = _roster_by_policy(roster)
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError("recovery observations 为空")
    identities = []
    for observation in observations:
        _validate_observation(observation, roster_by_policy)
        identities.append(str(observation["observation_id"]))
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError("recovery observation identity 重复")


def _target_character_map(
        groups: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[str, str]]:
    """投影已获 target authority 的单字符规则，用于 phrase composition。"""
    result = {}
    for group in groups:
        if (group["mapping_kind"] != "CHARACTER_INPUT"
                or group["target_resolution_kind"] == "NO_TARGET_AUTHORITY"):
            continue
        input_text = str(group["input_text"])
        if input_text in result:
            raise BroadQaExternalDataError("recovery target character 重复")
        result[input_text] = (str(group["target_output"]), str(group["group_id"]))
    return result


def _compose(
        input_text: str,
        character_map: dict[str, tuple[str, str]],
        ) -> tuple[str, list[int], list[str]]:
    """只改写已有 target authority 的字符，未知位置保持原文。"""
    output = []
    covered_positions = []
    group_ids = []
    for offset, character in enumerate(input_text):
        mapped = character_map.get(character)
        if mapped is None:
            output.append(character)
            continue
        output.append(mapped[0])
        covered_positions.append(offset)
        group_ids.append(mapped[1])
    return "".join(output), covered_positions, sorted(set(group_ids))


def derive_normalization_recovery_compositions(
        *,
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """逐 phrase observation 比较 target 字符组合与来源显式输出。"""
    character_map = _target_character_map(groups)
    group_by_input = {str(item["input_text"]): item for item in groups}
    values = []
    for observation in observations:
        if observation["mapping_kind"] != "PHRASE_INPUT":
            continue
        input_text = str(observation["input_text"])
        observed_output = str(observation["expected_output"])
        base_output, covered_positions, base_group_ids = _compose(
            input_text, character_map)
        if not covered_positions:
            qualification = "NO_COMPOSITION_EVIDENCE"
        elif base_output == observed_output:
            qualification = "COMPOSITION_SUPPORT"
        elif len(covered_positions) == len(input_text):
            qualification = "EXPLICIT_OVERRIDE"
        else:
            qualification = "PARTIAL_COMPOSITION"
        target_group = group_by_input[input_text]
        identity = {
            "base_character_group_ids": base_group_ids,
            "base_output": base_output,
            "input_text": input_text,
            "observed_output": observed_output,
            "phrase_observation_id": observation["observation_id"],
            "source_policy_scope": observation["source_policy_scope"],
            "target_group_id": target_group["group_id"],
        }
        values.append({
            **identity,
            "composition_id": _sha256(canonical_json_bytes(identity)),
            "covered_position_count": len(covered_positions),
            "covered_positions": covered_positions,
            "format_version": 2,
            "qualification_kind": qualification,
            "record_kind": NORMALIZATION_RECOVERY_COMPOSITION_KIND,
            "source_family": observation["source_family"],
            "target_output": target_group["target_output"],
            "target_resolution_kind": target_group["target_resolution_kind"],
            "unknown_position_count": len(input_text) - len(covered_positions),
        })
    result = tuple(sorted(values, key=lambda item: str(item["composition_id"])))
    if not result or len({item["composition_id"] for item in result}) != len(result):
        raise BroadQaExternalDataError("recovery composition identity 漂移")
    return result


__all__ = [
    "COMPOSITION_QUALIFICATIONS",
    "ICU_SOURCE_KEY",
    "MEDIAWIKI_CN_SOURCE_KEY",
    "MEDIAWIKI_CN_SOURCE_POLICY_SCOPE",
    "MEDIAWIKI_HANS_SOURCE_KEY",
    "MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE",
    "NORMALIZATION_RECOVERY_COMPOSITION_KIND",
    "NORMALIZATION_RECOVERY_GROUP_KIND",
    "NORMALIZATION_RECOVERY_LOSO_KIND",
    "NORMALIZATION_RECOVERY_OBSERVATION_KIND",
    "NORMALIZATION_RECOVERY_SOURCE_ROSTER_KIND",
    "OPENCC_SOURCE_KEY",
    "RECOVERY_TARGET_POLICY_SCOPE",
    "SOURCE_FAMILIES",
    "SOURCE_POLICY_DEFINITIONS",
    "SOURCE_POLICY_SCOPES",
    "UNIHAN_SOURCE_KEY",
    "UNIHAN_SOURCE_POLICY_SCOPE",
    "derive_icu_recovery_observations",
    "derive_mediawiki_recovery_observations",
    "derive_normalization_recovery_compositions",
    "derive_normalization_recovery_groups",
    "derive_normalization_recovery_source_roster",
    "derive_opencc_recovery_observations",
    "derive_unihan_recovery_observations",
    "resolve_normalization_recovery_group_authority",
    "validate_normalization_recovery_observation_inventory",
]
