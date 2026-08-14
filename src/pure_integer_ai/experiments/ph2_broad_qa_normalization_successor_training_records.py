"""从 OpenCC/ICU 官方 source pack 派生 successor 训练记录。

本模块只做纯记录变换，不读取路径、不写 artifact，也不接触任何 evaluation、
reserve、formal report、candidate 或生产 consumer。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_SUCCESSOR_OBSERVATION_KIND = (
    "NORMALIZATION_SUCCESSOR_TRAIN_OBSERVATION_V1")
NORMALIZATION_SUCCESSOR_GROUP_KIND = (
    "NORMALIZATION_SUCCESSOR_TRAIN_GROUP_V1")
NORMALIZATION_SUCCESSOR_CONTEXT_KIND = (
    "NORMALIZATION_SUCCESSOR_TRAIN_CONTEXT_REPLAY_V1")
OPENCC_SOURCE_POLICY_SCOPE = "OPENCC_T2S_SOURCE_BEHAVIOR_V1"
ICU_SOURCE_POLICY_SCOPE = "ICU_RELEASE_77_1_T2S_SOURCE_BEHAVIOR_V1"
SUCCESSOR_TARGET_POLICY_SCOPE = "ZH_HANS_CROSS_SOURCE_CONSENSUS_V1"
SOURCE_POLICY_SCOPES = (OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE)
GROUP_KINDS = (
    "CROSS_SOURCE_CONSENSUS", "SINGLE_SOURCE", "SOURCE_POLICY_CONFLICT")
CONTEXT_QUALIFICATIONS = ("SOURCE_REPLAY_OVERRIDE", "SOURCE_REPLAY_SUPPORT")
OBSERVATION_FIELDS = {
    "evidence_source_scope", "expected_output", "format_version",
    "input_scalar_count", "input_text", "mapping_kind", "observation_id",
    "output_scalar_count", "record_kind", "selected_target_variant_ordinal",
    "source_commitment", "source_pack_manifest_sha256",
    "source_policy_scope", "split", "target_variant_count",
}


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
    """只接受不小于下界的真整数，拒绝 JSON bool 冒充计数。"""
    return type(value) is int and value >= minimum


def _validate_source_commitment(
        commitment: object,
        *,
        source_policy_scope: str,
        ) -> None:
    """严格核验两类来源承诺的字段、物理范围与摘要格式。"""
    if not isinstance(commitment, dict):
        raise BroadQaExternalDataError(
            "successor observation source commitment 非对象")
    if source_policy_scope == OPENCC_SOURCE_POLICY_SCOPE:
        expected_fields = {
            "byte_end", "byte_start", "file_sha256", "line_ordinal",
            "line_sha256", "relative_path",
        }
        if set(commitment) != expected_fields:
            raise BroadQaExternalDataError(
                "successor OpenCC source commitment schema 漂移")
        relative_path = commitment["relative_path"]
        if (relative_path not in {
                "dictionary/TSCharacters.txt", "dictionary/TSPhrases.txt"}
                or commitment["file_sha256"]
                != NORMALIZATION_SOURCE_FILES[relative_path]["sha256"]):
            raise BroadQaExternalDataError(
                "successor OpenCC source commitment 文件身份漂移")
        if (not _integer(commitment["byte_start"])
                or not _integer(commitment["byte_end"], minimum=1)
                or commitment["byte_end"] <= commitment["byte_start"]
                or not _integer(commitment["line_ordinal"], minimum=1)):
            raise BroadQaExternalDataError(
                "successor OpenCC source commitment 物理范围非法")
        _sha_value(commitment["line_sha256"], label="OpenCC line SHA")
        return
    if source_policy_scope != ICU_SOURCE_POLICY_SCOPE:
        raise BroadQaExternalDataError(
            "successor observation source policy 非法")
    expected_fields = {
        "byte_end", "byte_start", "line_end_ordinal",
        "line_start_ordinal", "physical_lines", "statement_sha256",
    }
    if set(commitment) != expected_fields:
        raise BroadQaExternalDataError(
            "successor ICU source commitment schema 漂移")
    physical_lines = commitment["physical_lines"]
    if (not _integer(commitment["byte_start"])
            or not _integer(commitment["byte_end"], minimum=1)
            or commitment["byte_end"] <= commitment["byte_start"]
            or not _integer(commitment["line_start_ordinal"], minimum=1)
            or not _integer(commitment["line_end_ordinal"], minimum=1)
            or commitment["line_end_ordinal"]
            < commitment["line_start_ordinal"]
            or not isinstance(physical_lines, list) or not physical_lines):
        raise BroadQaExternalDataError(
            "successor ICU source commitment 物理范围非法")
    expected_line_fields = {
        "byte_end", "byte_start", "line_ordinal", "line_sha256"}
    for line in physical_lines:
        if (not isinstance(line, dict) or set(line) != expected_line_fields
                or not _integer(line["byte_start"])
                or not _integer(line["byte_end"], minimum=1)
                or line["byte_end"] <= line["byte_start"]
                or not _integer(line["line_ordinal"], minimum=1)):
            raise BroadQaExternalDataError(
                "successor ICU source commitment physical line 非法")
        _sha_value(line["line_sha256"], label="ICU line SHA")
    if (physical_lines[0]["byte_start"] != commitment["byte_start"]
            or physical_lines[-1]["byte_end"] != commitment["byte_end"]
            or physical_lines[0]["line_ordinal"]
            != commitment["line_start_ordinal"]
            or physical_lines[-1]["line_ordinal"]
            != commitment["line_end_ordinal"]):
        raise BroadQaExternalDataError(
            "successor ICU source commitment 外层范围漂移")
    _sha_value(commitment["statement_sha256"], label="ICU statement SHA")


def _validate_observation(observation: dict[str, object]) -> None:
    """核验 learner 物化 observation 的完整 schema 与自绑定 identity。"""
    if set(observation) != OBSERVATION_FIELDS:
        raise BroadQaExternalDataError(
            "successor training observation schema 漂移")
    input_text = observation["input_text"]
    expected_output = observation["expected_output"]
    policy = observation["source_policy_scope"]
    source_sha = _sha_value(
        observation["source_pack_manifest_sha256"],
        label="successor observation source manifest")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(expected_output, str) or not expected_output
            or policy not in SOURCE_POLICY_SCOPES
            or observation["format_version"] != 1
            or observation["record_kind"]
            != NORMALIZATION_SUCCESSOR_OBSERVATION_KIND
            or observation["split"] != "TRAIN_SOURCE"
            or observation["selected_target_variant_ordinal"] != 0
            or not _integer(observation["target_variant_count"], minimum=1)
            or observation["input_scalar_count"] != len(input_text)
            or observation["output_scalar_count"] != len(expected_output)
            or observation["mapping_kind"] != (
                "CHARACTER_INPUT" if len(input_text) == 1
                else "PHRASE_INPUT")):
        raise BroadQaExternalDataError(
            "successor training observation 字段值漂移")
    prefix = (
        "OPENCC_SOURCE_PACK_SHA256:"
        if policy == OPENCC_SOURCE_POLICY_SCOPE
        else "ICU_SOURCE_PACK_SHA256:")
    if observation["evidence_source_scope"] != prefix + source_sha:
        raise BroadQaExternalDataError(
            "successor observation evidence source scope 漂移")
    _validate_source_commitment(
        observation["source_commitment"], source_policy_scope=str(policy))
    identity = {
        "expected_output": expected_output,
        "input_text": input_text,
        "source_commitment": observation["source_commitment"],
        "source_pack_manifest_sha256": source_sha,
        "source_policy_scope": policy,
    }
    if observation["observation_id"] != _sha256(
            canonical_json_bytes(identity)):
        raise BroadQaExternalDataError(
            "successor training observation identity 漂移")


def _dictionary_lines(
        payload: bytes,
        *,
        relative_path: str,
        file_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """解析 OpenCC 单 tab 词典并保存完整物理行承诺。"""
    if not isinstance(payload, bytes):
        raise BroadQaExternalDataError("successor OpenCC dictionary 非 bytes")
    lines = tuple(payload.splitlines(keepends=True))
    if (not lines or any(not line.endswith(b"\n") for line in lines)
            or any(line.endswith(b"\r\n") for line in lines)):
        raise BroadQaExternalDataError(
            "successor OpenCC dictionary 必须是完整 LF 物理行")
    records = []
    byte_start = 0
    keys = set()
    for line_ordinal, encoded in enumerate(lines, start=1):
        try:
            text = encoded[:-1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise BroadQaExternalDataError(
                "successor OpenCC dictionary 非 UTF-8") from error
        if text.count("\t") != 1:
            raise BroadQaExternalDataError(
                "successor OpenCC dictionary 非单 tab")
        source, raw_targets = text.split("\t")
        targets = raw_targets.split(" ")
        if (not source or source in keys or not targets
                or any(not value for value in targets)
                or any(0xD800 <= ord(value) <= 0xDFFF
                       for value in source + raw_targets)):
            raise BroadQaExternalDataError(
                "successor OpenCC dictionary key/value 漂移")
        keys.add(source)
        byte_end = byte_start + len(encoded)
        records.append({
            "byte_end": byte_end,
            "byte_start": byte_start,
            "file_sha256": file_sha256,
            "line_ordinal": line_ordinal,
            "line_sha256": _sha256(encoded),
            "relative_path": relative_path,
            "source": source,
            "target_variants": targets,
        })
        byte_start = byte_end
    if byte_start != len(payload):
        raise BroadQaExternalDataError(
            "successor OpenCC dictionary 字节覆盖漂移")
    return tuple(records)


def _opencc_observation(
        *,
        source_pack_manifest_sha256: str,
        line: dict[str, object],
        ) -> dict[str, object]:
    """把一条 OpenCC 物理词典行投影为来源行为观察。"""
    input_text = str(line["source"])
    target_variants = list(line["target_variants"])
    expected_output = str(target_variants[0])
    commitment = {
        key: line[key] for key in (
            "byte_end", "byte_start", "file_sha256", "line_ordinal",
            "line_sha256", "relative_path")
    }
    identity = {
        "expected_output": expected_output,
        "input_text": input_text,
        "source_commitment": commitment,
        "source_pack_manifest_sha256": source_pack_manifest_sha256,
        "source_policy_scope": OPENCC_SOURCE_POLICY_SCOPE,
    }
    return {
        **identity,
        "evidence_source_scope": (
            f"OPENCC_SOURCE_PACK_SHA256:{source_pack_manifest_sha256}"),
        "format_version": 1,
        "input_scalar_count": len(input_text),
        "mapping_kind": (
            "CHARACTER_INPUT" if len(input_text) == 1 else "PHRASE_INPUT"),
        "observation_id": _sha256(canonical_json_bytes(identity)),
        "output_scalar_count": len(expected_output),
        "record_kind": NORMALIZATION_SUCCESSOR_OBSERVATION_KIND,
        "selected_target_variant_ordinal": 0,
        "split": "TRAIN_SOURCE",
        "target_variant_count": len(target_variants),
    }


def derive_opencc_successor_observations(
        *,
        source_pack_manifest_sha256: str,
        character_payload: bytes,
        phrase_payload: bytes,
        ) -> tuple[dict[str, object], ...]:
    """从两份 OpenCC T2S 词典派生字符与短语训练观察。"""
    source_sha = _sha_value(
        source_pack_manifest_sha256, label="OpenCC source pack manifest")
    values = []
    for relative_path, payload in (
            ("dictionary/TSCharacters.txt", character_payload),
            ("dictionary/TSPhrases.txt", phrase_payload)):
        lines = _dictionary_lines(
            payload,
            relative_path=relative_path,
            file_sha256=NORMALIZATION_SOURCE_FILES[relative_path]["sha256"],
        )
        values.extend(_opencc_observation(
            source_pack_manifest_sha256=source_sha, line=line)
            for line in lines)
    result = tuple(sorted(values, key=lambda item: str(item["observation_id"])))
    if (not result or len({item["observation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError(
            "successor OpenCC observation identity 漂移")
    return result


def _icu_source_commitment(rule: dict[str, object]) -> dict[str, object]:
    """投影一条 ICU statement 的物理来源承诺。"""
    keys = (
        "byte_end", "byte_start", "line_end_ordinal",
        "line_start_ordinal", "physical_lines", "statement_sha256")
    if any(key not in rule for key in keys):
        raise BroadQaExternalDataError(
            "successor ICU source commitment 缺字段")
    return {key: rule[key] for key in keys}


def derive_icu_successor_observations(
        *,
        source_pack_manifest_sha256: str,
        rules: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从 ICU reverse-eligible rules 派生来源行为观察。"""
    source_sha = _sha_value(
        source_pack_manifest_sha256, label="ICU source pack manifest")
    if not isinstance(rules, tuple) or not rules:
        raise BroadQaExternalDataError("successor ICU rules 为空")
    values = []
    inputs = set()
    for rule in rules:
        if rule.get("t2s_reverse_eligible") != 1:
            continue
        input_text = rule.get("t2s_input")
        expected_output = rule.get("t2s_expected_output")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(expected_output, str) or not expected_output
                or input_text in inputs):
            raise BroadQaExternalDataError(
                "successor ICU eligible input 重复或非法")
        inputs.add(input_text)
        commitment = _icu_source_commitment(rule)
        identity = {
            "expected_output": expected_output,
            "input_text": input_text,
            "source_commitment": commitment,
            "source_pack_manifest_sha256": source_sha,
            "source_policy_scope": ICU_SOURCE_POLICY_SCOPE,
        }
        values.append({
            **identity,
            "evidence_source_scope": (
                f"ICU_SOURCE_PACK_SHA256:{source_sha}"),
            "format_version": 1,
            "input_scalar_count": len(input_text),
            "mapping_kind": (
                "CHARACTER_INPUT" if len(input_text) == 1
                else "PHRASE_INPUT"),
            "observation_id": _sha256(canonical_json_bytes(identity)),
            "output_scalar_count": len(expected_output),
            "record_kind": NORMALIZATION_SUCCESSOR_OBSERVATION_KIND,
            "selected_target_variant_ordinal": 0,
            "split": "TRAIN_SOURCE",
            "target_variant_count": 1,
        })
    result = tuple(sorted(values, key=lambda item: str(item["observation_id"])))
    if (not result or len({item["observation_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError(
            "successor ICU observation identity 漂移")
    return result


def _group_record(
        input_text: str,
        observations: list[dict[str, object]],
        ) -> dict[str, object]:
    """把同 input 的多来源观察分为共识、冲突或单来源。"""
    ordered = sorted(
        observations,
        key=lambda item: (str(item["source_policy_scope"]),
                          str(item["observation_id"])))
    policies = [str(item["source_policy_scope"]) for item in ordered]
    if len(set(policies)) != len(policies):
        raise BroadQaExternalDataError(
            "successor group 同 policy 重复 input")
    outputs = [str(item["expected_output"]) for item in ordered]
    unique_outputs = sorted(set(outputs))
    if len(policies) == 1:
        group_kind = "SINGLE_SOURCE"
    elif len(unique_outputs) == 1:
        group_kind = "CROSS_SOURCE_CONSENSUS"
    else:
        group_kind = "SOURCE_POLICY_CONFLICT"
    identity = {
        "input_text": input_text,
        "observation_ids": [item["observation_id"] for item in ordered],
        "source_policy_outputs": [{
            "expected_output": item["expected_output"],
            "observation_id": item["observation_id"],
            "source_policy_scope": item["source_policy_scope"],
        } for item in ordered],
    }
    return {
        **identity,
        "consensus_is_identity": int(
            group_kind == "CROSS_SOURCE_CONSENSUS"
            and input_text == unique_outputs[0]),
        "consensus_output": (
            unique_outputs[0]
            if group_kind == "CROSS_SOURCE_CONSENSUS" else ""),
        "eligible_target_policy_scope": (
            SUCCESSOR_TARGET_POLICY_SCOPE
            if group_kind == "CROSS_SOURCE_CONSENSUS" else ""),
        "format_version": 1,
        "group_id": _sha256(canonical_json_bytes(identity)),
        "group_kind": group_kind,
        "input_scalar_count": len(input_text),
        "mapping_kind": (
            "CHARACTER_INPUT" if len(input_text) == 1 else "PHRASE_INPUT"),
        "record_kind": NORMALIZATION_SUCCESSOR_GROUP_KIND,
        "source_policy_count": len(policies),
        "unique_output_count": len(unique_outputs),
    }


def _context_records(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """逐 policy 用字符观察重放短语，形成 support/override inventory。"""
    character_maps: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    phrases = []
    for observation in observations:
        policy = str(observation["source_policy_scope"])
        input_text = str(observation["input_text"])
        expected = str(observation["expected_output"])
        if len(input_text) == 1:
            if input_text in character_maps[policy]:
                raise BroadQaExternalDataError(
                    "successor context base character 重复")
            character_maps[policy][input_text] = (
                expected, str(observation["observation_id"]))
        else:
            phrases.append(observation)
    values = []
    for phrase in phrases:
        policy = str(phrase["source_policy_scope"])
        input_text = str(phrase["input_text"])
        observed_output = str(phrase["expected_output"])
        base_parts = []
        base_observation_ids = []
        for character in input_text:
            mapped = character_maps[policy].get(character)
            if mapped is None:
                base_parts.append(character)
            else:
                base_parts.append(mapped[0])
                base_observation_ids.append(mapped[1])
        base_output = "".join(base_parts)
        qualification = (
            "SOURCE_REPLAY_SUPPORT" if base_output == observed_output
            else "SOURCE_REPLAY_OVERRIDE")
        identity = {
            "base_observation_ids": sorted(set(base_observation_ids)),
            "base_output": base_output,
            "input_text": input_text,
            "observed_output": observed_output,
            "phrase_observation_id": phrase["observation_id"],
            "source_policy_scope": policy,
        }
        values.append({
            **identity,
            "context_id": _sha256(canonical_json_bytes(identity)),
            "exact_context_required": int(
                qualification == "SOURCE_REPLAY_OVERRIDE"),
            "format_version": 1,
            "qualification_kind": qualification,
            "record_kind": NORMALIZATION_SUCCESSOR_CONTEXT_KIND,
        })
    result = tuple(sorted(values, key=lambda item: str(item["context_id"])))
    if (not result or len({item["context_id"] for item in result})
            != len(result)):
        raise BroadQaExternalDataError(
            "successor context identity 漂移")
    return result


def derive_normalization_successor_training_records(
        *,
        opencc_observations: tuple[dict[str, object], ...],
        icu_observations: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """合并两源观察并派生 exact-input group 与 context replay。"""
    if (not isinstance(opencc_observations, tuple) or not opencc_observations
            or not isinstance(icu_observations, tuple)
            or not icu_observations):
        raise BroadQaExternalDataError(
            "successor training observation source 为空")
    observations = tuple(sorted(
        opencc_observations + icu_observations,
        key=lambda item: str(item["observation_id"])))
    identities = [str(item["observation_id"]) for item in observations]
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError(
            "successor training observation identity 重复")
    by_input: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for observation in observations:
        _validate_observation(observation)
        by_input[str(observation["input_text"])].append(observation)
    groups = tuple(sorted(
        (_group_record(input_text, values)
         for input_text, values in by_input.items()),
        key=lambda item: str(item["group_id"])))
    contexts = _context_records(observations)
    group_counts = Counter(str(item["group_kind"]) for item in groups)
    consensus_identity_count = sum(
        item["consensus_is_identity"] == 1 for item in groups)
    mapping_group_counts = Counter(
        (str(item["mapping_kind"]), str(item["group_kind"]))
        for item in groups)
    context_counts = Counter(
        (str(item["source_policy_scope"]),
         str(item["qualification_kind"])) for item in contexts)
    source_counts = Counter(
        str(item["source_policy_scope"]) for item in observations)
    summary = {
        "context_count": len(contexts),
        "context_qualification_counts": {
            f"{policy}:{qualification}": context_counts[(policy, qualification)]
            for policy in SOURCE_POLICY_SCOPES
            for qualification in CONTEXT_QUALIFICATIONS
        },
        "group_count": len(groups),
        "group_kind_counts": {
            kind: group_counts[kind] for kind in GROUP_KINDS},
        "identity_consensus_count": consensus_identity_count,
        "mapping_group_counts": {
            f"{mapping}:{kind}": mapping_group_counts[(mapping, kind)]
            for mapping in ("CHARACTER_INPUT", "PHRASE_INPUT")
            for kind in GROUP_KINDS
        },
        "observation_count": len(observations),
        "nonidentity_consensus_count": (
            group_counts["CROSS_SOURCE_CONSENSUS"]
            - consensus_identity_count),
        "source_observation_counts": {
            policy: source_counts[policy] for policy in SOURCE_POLICY_SCOPES},
    }
    if (group_counts["CROSS_SOURCE_CONSENSUS"] < 1
            or group_counts["SOURCE_POLICY_CONFLICT"] < 1
            or sum(context_counts[(policy, "SOURCE_REPLAY_OVERRIDE")]
                   for policy in SOURCE_POLICY_SCOPES) < 1
            or sum(context_counts[(policy, "SOURCE_REPLAY_SUPPORT")]
                   for policy in SOURCE_POLICY_SCOPES) < 1):
        raise BroadQaExternalDataError(
            "successor training 共识/冲突/context 库存不足")
    return observations, groups, contexts, summary


__all__ = [
    "CONTEXT_QUALIFICATIONS",
    "GROUP_KINDS",
    "ICU_SOURCE_POLICY_SCOPE",
    "NORMALIZATION_SUCCESSOR_CONTEXT_KIND",
    "NORMALIZATION_SUCCESSOR_GROUP_KIND",
    "NORMALIZATION_SUCCESSOR_OBSERVATION_KIND",
    "OPENCC_SOURCE_POLICY_SCOPE",
    "SOURCE_POLICY_SCOPES",
    "SUCCESSOR_TARGET_POLICY_SCOPE",
    "derive_icu_successor_observations",
    "derive_normalization_successor_training_records",
    "derive_opencc_successor_observations",
]
