"""冻结 OpenCC/ICU normalization successor TRAIN 协议与 learner 读边界。

publisher/auditor 可严格读取两个官方 source pack；learner 只能读取本协议物化的
TRAIN 文件。本模块不导入或接受任何 evaluation、reserve、formal publication 路径。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    read_normalization_icu_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    SUCCESSOR_TARGET_POLICY_SCOPE,
    derive_icu_successor_observations,
    derive_normalization_successor_training_records,
    derive_opencc_successor_observations,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_SUCCESSOR_TRAINING_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_TRAINING_PROTOCOL_V1")
NORMALIZATION_SUCCESSOR_TRAINING_STATUS = "FROZEN_NOT_READ_NOT_LEARNED"
NORMALIZATION_SUCCESSOR_WORK_KIND = "NORMALIZATION_SUCCESSOR_WORK_ITEM_V1"
OPENCC_TRAIN_SOURCE_MANIFEST_SHA256 = (
    "189f42097dc059218be337231d340a4265d2783b64c2fb884892db0caf8af94c")
ICU_TRAIN_SOURCE_MANIFEST_SHA256 = (
    "26ef0c1f566030a611dd534d170598430944785d112d765d5737cda45e6ce747")
TRAINING_PHASES = (
    "SOURCE_OBSERVATION_INGEST",
    "CROSS_SOURCE_GROUP_RESOLUTION",
    "CONTEXT_REPLAY_RESOLUTION",
)
TRAINING_FILE_ROLES = (
    ("train.observations.jsonl", "TRAIN_OBSERVATIONS"),
    ("train.groups.jsonl", "TRAIN_CROSS_SOURCE_GROUPS"),
    ("train.contexts.jsonl", "TRAIN_CONTEXT_REPLAYS"),
    ("train.work.jsonl", "TRAIN_ORDERED_WORK"),
)

NORMALIZATION_SUCCESSOR_EVIDENCE_CONTRACT = {
    "allowed_stances": ["REFUTE", "SUPPORT"],
    "context_evidence_requires_all_available_base_observation_ids": 1,
    "context_evidence_requires_phrase_observation_id": 1,
    "cross_source_rule_support_per_source_policy_min": 1,
    "evidence_id_must_bind_protocol_and_source_commitment": 1,
    "evidence_record_kind": "NORMALIZATION_SUCCESSOR_EVIDENCE_V1",
    "evaluation_or_reserve_evidence_allowed": 0,
    "mechanical_group_kind_is_mastery_label": 0,
    "source_commitment_required": 1,
    "source_policy_scope_required": 1,
    "target_policy_scope_required_for_consensus_rule": 1,
}

NORMALIZATION_SUCCESSOR_RULE_PACK_CONTRACT = {
    "candidate_state": "LEARNED_PACK_DISABLED",
    "conflict_global_rewrite_allowed": 0,
    "conflict_group_must_preserve_all_policy_outputs": 1,
    "context_override_requires_exact_input_and_source_policy": 1,
    "context_override_replay_must_equal_observation": 1,
    "hardcoded_item_title_page_or_evaluation_id_dispatch_allowed": 0,
    "nonidentity_consensus_group_rule_required": 1,
    "production_enabled": 0,
    "rule_application_domain_required": 1,
    "rule_evidence_source_scopes_must_equal_group_policies": 1,
    "rule_pack_kind": "NORMALIZATION_SUCCESSOR_RULE_PACK_V1",
    "single_source_group_global_rule_allowed": 0,
    "target_policy_scope": SUCCESSOR_TARGET_POLICY_SCOPE,
    "teacher_api_llm_call_count": 0,
}

NORMALIZATION_SUCCESSOR_CHECKPOINT_CONTRACT = {
    "append_only_hash_chain_required": 1,
    "checkpoint_record_kind": "NORMALIZATION_SUCCESSOR_CHECKPOINT_V1",
    "complete_requires_full_ordered_work_prefix": 1,
    "fresh_and_resume_run_identity_must_differ": 1,
    "fresh_resume_canonical_output_bytes_must_equal": 1,
    "logical_cursor_must_equal_processed_prefix_count": 1,
    "manifest_last_required": 1,
    "protocol_manifest_sha256_required": 1,
    "wall_clock_or_random_identity_allowed": 0,
    "work_identity_sha256_required": 1,
}


def _sha256(payload: bytes) -> str:
    """返回来源、规范记录或 artifact 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, list):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _work_records(
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        contexts: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按固定三阶段构造 learner 必须完整消费的有序 work。"""
    sources = (
        (TRAINING_PHASES[0], "OBSERVATION", observations, "observation_id"),
        (TRAINING_PHASES[1], "GROUP", groups, "group_id"),
        (TRAINING_PHASES[2], "CONTEXT", contexts, "context_id"),
    )
    values = []
    ordinal = 0
    for phase, work_kind, records, identity_key in sources:
        for record in records:
            record_id = str(record[identity_key])
            identity = {
                "phase": phase,
                "record_id": record_id,
                "work_kind": work_kind,
            }
            values.append({
                **identity,
                "format_version": 1,
                "record_kind": NORMALIZATION_SUCCESSOR_WORK_KIND,
                "work_id": _sha256(canonical_json_bytes(identity)),
                "work_ordinal": ordinal,
            })
            ordinal += 1
    result = tuple(values)
    if (not result or [item["work_ordinal"] for item in result]
            != list(range(len(result)))
            or len({item["work_id"] for item in result}) != len(result)):
        raise BroadQaExternalDataError(
            "successor training ordered work 漂移")
    return result


def _work_identity(values: tuple[dict[str, object], ...]) -> str:
    """绑定完整有序 work identity，不依赖文件路径或墙钟。"""
    return _sha256(canonical_json_bytes([{
        "record_id": item["record_id"],
        "work_id": item["work_id"],
        "work_ordinal": item["work_ordinal"],
    } for item in values]))


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范 JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(f"{label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} JSONL 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """构造一个 learner 文件的物理身份。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _derived_artifact(
        *,
        name: str,
        role: str,
        values: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """不读取物化目录，直接计算一份规范 JSONL 的预期物理身份。"""
    digest = hashlib.sha256()
    byte_count = 0
    for value in values:
        encoded = canonical_json_line(value)
        digest.update(encoded)
        byte_count += len(encoded)
    return {
        "bytes": byte_count,
        "record_count": len(values),
        "relative_path": name,
        "role": role,
        "sha256": digest.hexdigest(),
    }


def _contract(
        *,
        summary: dict[str, object],
        work: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """按冻结库存构造停止、输出和恢复门。"""
    group_counts = summary["group_kind_counts"]
    context_counts = summary["context_qualification_counts"]
    override_count = sum(
        int(value) for key, value in context_counts.items()
        if key.endswith(":SOURCE_REPLAY_OVERRIDE"))
    return {
        "checkpoint_contract": NORMALIZATION_SUCCESSOR_CHECKPOINT_CONTRACT,
        "complete_work_item_count": len(work),
        "evidence_contract": NORMALIZATION_SUCCESSOR_EVIDENCE_CONTRACT,
        "exact_output_counts": {
            "conflict_ledger_count": group_counts["SOURCE_POLICY_CONFLICT"],
            "context_override_count": override_count,
            "nonidentity_consensus_rule_count": summary[
                "nonidentity_consensus_count"],
            "single_source_defer_count": group_counts["SINGLE_SOURCE"],
        },
        "learner_source_pack_read_count": 0,
        "no_early_success": 1,
        "operator_family_concurrency_max": 1,
        "phase_order": list(TRAINING_PHASES),
        "process_every_work_item_at_least_once": 1,
        "rule_pack_contract": NORMALIZATION_SUCCESSOR_RULE_PACK_CONTRACT,
        "work_identity_sha256": _work_identity(work),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        learner_contract: dict[str, object],
        ) -> dict[str, object]:
    """构造来源冻结、评测隔离且尚未学习的协议 manifest。"""
    return {
        "artifact_kind": NORMALIZATION_SUCCESSOR_TRAINING_PROTOCOL_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_or_reserve_artifact_read_count": 0,
        "failed_icu_v2_artifact_read_count": 0,
        "files": files,
        "format_version": 1,
        "learner_contract": learner_contract,
        "learner_read_count": 0,
        "mastery_claimed": 0,
        "opencc_source_manifest_sha256": (
            OPENCC_TRAIN_SOURCE_MANIFEST_SHA256),
        "icu_source_manifest_sha256": ICU_TRAIN_SOURCE_MANIFEST_SHA256,
        "production_enabled": 0,
        "source_read_contract": {
            "allowed_artifact_kinds": [
                "PH2_BROAD_QA_NORMALIZATION_DEPENDENCY_SOURCE_PACK_V1",
                "PH2_BROAD_QA_NORMALIZATION_ICU_EVALUATION_SOURCE_PACK_V1",
            ],
            "allowed_source_policy_scopes": [
                OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE],
            "learner_reads_materialized_protocol_only": 1,
            "publisher_source_pack_count": 2,
            "unihan_or_mediawiki_source_allowed": 0,
        },
        "status": NORMALIZATION_SUCCESSOR_TRAINING_STATUS,
        "summary": summary,
        "target_policy_scope": SUCCESSOR_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization successor training run root 必须是 K 盘目录")
    return root


def _derive_from_sources(
        *,
        opencc_source_pack_dir: Path,
        icu_source_pack_dir: Path,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """严格回读两个 source pack 并派生全部 learner material。"""
    opencc_manifest = read_normalization_source_pack(opencc_source_pack_dir)
    icu_manifest, _variables, icu_rules = read_normalization_icu_source_pack(
        icu_source_pack_dir)
    if (opencc_manifest["manifest_sha256"]
            != OPENCC_TRAIN_SOURCE_MANIFEST_SHA256
            or icu_manifest["manifest_sha256"]
            != ICU_TRAIN_SOURCE_MANIFEST_SHA256):
        raise BroadQaExternalDataError(
            "successor training source manifest identity 漂移")
    try:
        character_payload = (
            opencc_source_pack_dir / "dictionary/TSCharacters.txt").read_bytes()
        phrase_payload = (
            opencc_source_pack_dir / "dictionary/TSPhrases.txt").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "successor OpenCC training source 不可读") from error
    opencc = derive_opencc_successor_observations(
        source_pack_manifest_sha256=opencc_manifest["manifest_sha256"],
        character_payload=character_payload,
        phrase_payload=phrase_payload,
    )
    icu = derive_icu_successor_observations(
        source_pack_manifest_sha256=icu_manifest["manifest_sha256"],
        rules=icu_rules,
    )
    observations, groups, contexts, summary = (
        derive_normalization_successor_training_records(
            opencc_observations=opencc, icu_observations=icu))
    work = _work_records(observations, groups, contexts)
    return observations, groups, contexts, work, summary


def publish_normalization_successor_training_protocol(
        *,
        run_root: str | Path,
        opencc_source_pack_dir: str | Path,
        icu_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 learner 唯一可读的 successor TRAIN protocol。"""
    root = _require_k_root(run_root)
    opencc = Path(opencc_source_pack_dir).resolve()
    icu = Path(icu_source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not opencc.is_dir() or not icu.is_dir() or opencc == icu
            or any(not path.is_relative_to(root)
                   for path in (opencc, icu, target))):
        raise BroadQaExternalDataError(
            "successor training source/target 越出或混淆 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization successor training target 已存在")
    observations, groups, contexts, work, summary = _derive_from_sources(
        opencc_source_pack_dir=opencc, icu_source_pack_dir=icu)
    target.mkdir(parents=True)
    values_by_name = {
        "train.observations.jsonl": observations,
        "train.groups.jsonl": groups,
        "train.contexts.jsonl": contexts,
        "train.work.jsonl": work,
    }
    for name, _role in TRAINING_FILE_ROLES:
        _write_jsonl(target / name, values_by_name[name])
    files = [
        _artifact(target / name, role=role,
                  count=len(values_by_name[name]))
        for name, role in TRAINING_FILE_ROLES
    ]
    learner_contract = _contract(summary=summary, work=work)
    manifest = _manifest(
        files=files, summary=summary, learner_contract=learner_contract)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def _read_manifest_and_material(
        protocol_dir: Path,
        ) -> tuple[
            dict[str, object],
            bytes,
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """只在 protocol 目录内读取 manifest 与四份 learner material。"""
    manifest_path = protocol_dir / "manifest.json"
    try:
        encoded_manifest = manifest_path.read_bytes()
        manifest = json.loads(encoded_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "successor training protocol manifest 不可读") from error
    if (not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded_manifest):
        raise BroadQaExternalDataError(
            "successor training protocol manifest 非规范")
    values = tuple(_read_jsonl(
        protocol_dir / name, label=role)
        for name, role in TRAINING_FILE_ROLES)
    return manifest, encoded_manifest, values[0], values[1], values[2], values[3]


def read_normalization_successor_learner_input(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """按外部冻结 manifest 身份回读 learner 输入，不打开 source pack。"""
    root = Path(protocol_dir).resolve()
    manifest, encoded, observations, groups, contexts, work = (
        _read_manifest_and_material(root))
    expected_manifest_sha = _sha_value(
        expected_manifest_sha256,
        label="successor training expected manifest SHA")
    if _sha256(encoded) != expected_manifest_sha:
        raise BroadQaExternalDataError(
            "successor training protocol manifest identity 漂移")
    opencc = tuple(
        item for item in observations
        if item.get("source_policy_scope") == OPENCC_SOURCE_POLICY_SCOPE)
    icu = tuple(
        item for item in observations
        if item.get("source_policy_scope") == ICU_SOURCE_POLICY_SCOPE)
    derived_observations, derived_groups, derived_contexts, summary = (
        derive_normalization_successor_training_records(
            opencc_observations=opencc, icu_observations=icu))
    derived_work = _work_records(
        derived_observations, derived_groups, derived_contexts)
    if (not _strict_equal(observations, derived_observations)
            or not _strict_equal(groups, derived_groups)
            or not _strict_equal(contexts, derived_contexts)
            or not _strict_equal(work, derived_work)):
        raise BroadQaExternalDataError(
            "successor learner material 内部派生漂移")
    files = [
        _artifact(root / name, role=role, count=len(values))
        for (name, role), values in zip(
            TRAINING_FILE_ROLES,
            (observations, groups, contexts, work))
    ]
    expected = _manifest(
        files=files, summary=summary,
        learner_contract=_contract(summary=summary, work=work))
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(
            "successor training protocol manifest 漂移")
    return (
        {**manifest, "manifest_sha256": _sha256(encoded)},
        observations, groups, contexts, work,
    )


def read_normalization_successor_training_protocol(
        protocol_dir: str | Path,
        *,
        opencc_source_pack_dir: str | Path,
        icu_source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """auditor 从两源重派生并核对已物化 TRAIN protocol。"""
    derived = _derive_from_sources(
        opencc_source_pack_dir=Path(opencc_source_pack_dir).resolve(),
        icu_source_pack_dir=Path(icu_source_pack_dir).resolve(),
    )
    values = derived[:4]
    files = [
        _derived_artifact(name=name, role=role, values=records)
        for (name, role), records in zip(TRAINING_FILE_ROLES, values)
    ]
    expected_manifest = _manifest(
        files=files,
        summary=derived[4],
        learner_contract=_contract(summary=derived[4], work=derived[3]),
    )
    expected_manifest_sha = _sha256(canonical_json_line(expected_manifest))
    stored = read_normalization_successor_learner_input(
        protocol_dir,
        expected_manifest_sha256=expected_manifest_sha,
    )
    if any(not _strict_equal(left, right)
           for left, right in zip(stored[1:], derived[:4])):
        raise BroadQaExternalDataError(
            "successor training protocol/source 漂移")
    return stored


__all__ = [
    "ICU_TRAIN_SOURCE_MANIFEST_SHA256",
    "NORMALIZATION_SUCCESSOR_CHECKPOINT_CONTRACT",
    "NORMALIZATION_SUCCESSOR_EVIDENCE_CONTRACT",
    "NORMALIZATION_SUCCESSOR_RULE_PACK_CONTRACT",
    "NORMALIZATION_SUCCESSOR_TRAINING_PROTOCOL_KIND",
    "NORMALIZATION_SUCCESSOR_TRAINING_STATUS",
    "OPENCC_TRAIN_SOURCE_MANIFEST_SHA256",
    "publish_normalization_successor_training_protocol",
    "read_normalization_successor_learner_input",
    "read_normalization_successor_training_protocol",
]
