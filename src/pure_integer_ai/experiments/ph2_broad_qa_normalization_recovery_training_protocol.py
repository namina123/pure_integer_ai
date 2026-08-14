"""冻结多来源 normalization recovery TRAIN 协议与读取边界。

publisher/auditor 先用 manifest-only reader 证明 Firefox evaluation v2 已冻结，
再严格读取 OpenCC、ICU、Unihan/MediaWiki source pack。learner 只读取本协议
物化的许可分区 TRAIN 文件，不打开来源、evaluation、reserve 或 LOSO 审计文件。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    NORMALIZATION_ICU_LICENSE_ID,
    NORMALIZATION_ICU_SOURCE_PACK_KIND,
    read_normalization_icu_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND,
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
    read_normalization_recovery_evaluation_manifest_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_audit import (
    derive_normalization_recovery_loso,
    normalization_recovery_training_summary,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    MEDIAWIKI_CN_SOURCE_POLICY_SCOPE,
    MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    RECOVERY_TARGET_POLICY_SCOPE,
    SOURCE_POLICY_SCOPES,
    UNIHAN_SOURCE_POLICY_SCOPE,
    derive_icu_recovery_observations,
    derive_mediawiki_recovery_observations,
    derive_normalization_recovery_compositions,
    derive_normalization_recovery_groups,
    derive_normalization_recovery_source_roster,
    derive_opencc_recovery_observations,
    derive_unihan_recovery_observations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_LICENSE_ID,
    NORMALIZATION_SOURCE_PACK_KIND,
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_source_pack import (
    MEDIAWIKI_LICENSE_ID,
    NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
    UNIHAN_LICENSE_ID,
    read_normalization_successor_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_TRAINING_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_TRAINING_PROTOCOL_V2")
NORMALIZATION_RECOVERY_TRAINING_STATUS = "FROZEN_NOT_READ_NOT_LEARNED"
NORMALIZATION_RECOVERY_WORK_KIND = "NORMALIZATION_RECOVERY_WORK_ITEM_V2"

OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256 = (
    "189f42097dc059218be337231d340a4265d2783b64c2fb884892db0caf8af94c")
ICU_RECOVERY_SOURCE_MANIFEST_SHA256 = (
    "26ef0c1f566030a611dd534d170598430944785d112d765d5737cda45e6ce747")
SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256 = (
    "16944c9e42e954dbe5496235e9c796aa7d3816ccd5dc8c7826b16570ca2e8e56")
RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256 = (
    "9a1aa10f2b4285e74e62a8a265967caeefbb31779faf7af2bf8c6c29f15dfb70")

TRAINING_PHASES = (
    "SOURCE_ROSTER_INGEST",
    "LICENSE_PARTITIONED_OBSERVATION_INGEST",
    "SOURCE_FAMILY_GROUP_RESOLUTION",
    "PHRASE_COMPOSITION_RESOLUTION",
)
OBSERVATION_FILE_ROLES = (
    ("train.opencc.observations.jsonl", "TRAIN_OPENCC_OBSERVATIONS",
     (OPENCC_SOURCE_POLICY_SCOPE,), NORMALIZATION_SOURCE_LICENSE_ID),
    ("train.icu.observations.jsonl", "TRAIN_ICU_OBSERVATIONS",
     (ICU_SOURCE_POLICY_SCOPE,), NORMALIZATION_ICU_LICENSE_ID),
    ("train.unihan.observations.jsonl", "TRAIN_UNIHAN_OBSERVATIONS",
     (UNIHAN_SOURCE_POLICY_SCOPE,), UNIHAN_LICENSE_ID),
    ("train.mediawiki.observations.jsonl", "TRAIN_MEDIAWIKI_OBSERVATIONS",
     (MEDIAWIKI_HANS_SOURCE_POLICY_SCOPE, MEDIAWIKI_CN_SOURCE_POLICY_SCOPE),
     MEDIAWIKI_LICENSE_ID),
)
TRAINING_FILE_ROLES = (
    ("train.roster.jsonl", "TRAIN_SOURCE_POLICY_ROSTER"),
    *((name, role) for name, role, _policies, _license in OBSERVATION_FILE_ROLES),
    ("train.groups.jsonl", "TRAIN_SOURCE_FAMILY_GROUPS"),
    ("train.compositions.jsonl", "TRAIN_PHRASE_COMPOSITIONS"),
    ("train.audit.loso.jsonl", "TRAIN_ONLY_LOSO_AUDIT"),
    ("train.work.jsonl", "TRAIN_ORDERED_WORK"),
)

NORMALIZATION_RECOVERY_AUTHORITY_CONTRACT = {
    "generic_authority_requires_distinct_source_family_count_min": 2,
    "intra_family_policy_count_is_one_family_vote": 1,
    "mediawiki_cn_exact_input_only": 1,
    "mediawiki_cn_is_generic_family_vote": 0,
    "regional_authority_global_upgrade_allowed": 0,
    "source_family_conflict_global_rule_allowed": 0,
    "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
    "unknown_character_preserves_input": 1,
}

NORMALIZATION_RECOVERY_LEARNER_CONTRACT = {
    "evaluation_inventory_read_allowed": 0,
    "evaluation_manifest_read_allowed": 0,
    "hardcoded_evaluation_or_formal_dispatch_allowed": 0,
    "leave_one_policy_out_audit_read_allowed": 0,
    "learner_reads_materialized_protocol_only": 1,
    "phrase_override_requires_exact_input": 1,
    "production_enabled": 0,
    "reserve_identity_or_payload_read_allowed": 0,
    "source_pack_read_allowed": 0,
    "teacher_api_llm_call_count": 0,
}


def _sha256(payload: bytes) -> str:
    """返回规范记录或 artifact 的 SHA-256。"""
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
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按四阶段构造 learner 必须完整消费的确定序 work。"""
    sources = (
        (TRAINING_PHASES[0], "ROSTER", roster, "roster_id"),
        (TRAINING_PHASES[1], "OBSERVATION", observations, "observation_id"),
        (TRAINING_PHASES[2], "GROUP", groups, "group_id"),
        (TRAINING_PHASES[3], "COMPOSITION", compositions, "composition_id"),
    )
    values = []
    ordinal = 0
    for phase, work_kind, records, identity_key in sources:
        for record in records:
            identity = {
                "phase": phase,
                "record_id": record[identity_key],
                "work_kind": work_kind,
            }
            values.append({
                **identity,
                "format_version": 2,
                "record_kind": NORMALIZATION_RECOVERY_WORK_KIND,
                "work_id": _sha256(canonical_json_bytes(identity)),
                "work_ordinal": ordinal,
            })
            ordinal += 1
    result = tuple(values)
    if (not result or [item["work_ordinal"] for item in result]
            != list(range(len(result)))
            or len({item["work_id"] for item in result}) != len(result)):
        raise BroadQaExternalDataError("recovery training ordered work 漂移")
    return result


def _work_identity(values: tuple[dict[str, object], ...]) -> str:
    """绑定完整 work 序，不依赖墙钟、路径或 worker 数。"""
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
    """返回一个已物化文件的物理身份。"""
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
    """不打开目录，直接计算规范 JSONL 的预期身份。"""
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


def _license_partitions() -> list[dict[str, object]]:
    """冻结 observation 文件、许可与 policy 的一一对应。"""
    return [{
        "license_id": license_id,
        "relative_path": name,
        "source_policy_scopes": list(policies),
    } for name, _role, policies, license_id in OBSERVATION_FILE_ROLES]


def _contract(
        *,
        summary: dict[str, object],
        work: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """按冻结库存构造 learner 完整消费与输出边界。"""
    return {
        "authority_contract": NORMALIZATION_RECOVERY_AUTHORITY_CONTRACT,
        "complete_work_item_count": len(work),
        "learner_contract": NORMALIZATION_RECOVERY_LEARNER_CONTRACT,
        "license_partitions": _license_partitions(),
        "no_early_success": 1,
        "phase_order": list(TRAINING_PHASES),
        "process_every_work_item_at_least_once": 1,
        "train_only_loso_record_count": summary["loso_count"],
        "work_identity_sha256": _work_identity(work),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        learner_contract: dict[str, object],
        ) -> dict[str, object]:
    """构造来源、评测隔离、family authority 与许可分区 manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_TRAINING_PROTOCOL_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_payload_read_count": 0,
        "evaluation_protocol_artifact_kind": (
            NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND),
        "evaluation_protocol_manifest_only_read_count": 1,
        "evaluation_protocol_manifest_sha256": (
            RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256),
        "evaluation_protocol_status": NORMALIZATION_RECOVERY_EVALUATION_STATUS,
        "files": files,
        "format_version": 2,
        "learner_contract": learner_contract,
        "learner_read_count": 0,
        "mastery_claimed": 0,
        "opencc_source_manifest_sha256": (
            OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256),
        "icu_source_manifest_sha256": ICU_RECOVERY_SOURCE_MANIFEST_SHA256,
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "source_read_contract": {
            "allowed_artifact_kinds": [
                NORMALIZATION_SOURCE_PACK_KIND,
                NORMALIZATION_ICU_SOURCE_PACK_KIND,
                NORMALIZATION_SUCCESSOR_SOURCE_PACK_KIND,
            ],
            "allowed_source_policy_scopes": list(SOURCE_POLICY_SCOPES),
            "publisher_source_pack_count": 3,
            "source_family_count": 3,
            "source_policy_count": 5,
        },
        "status": NORMALIZATION_RECOVERY_TRAINING_STATUS,
        "successor_source_manifest_sha256": (
            SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256),
        "summary": summary,
        "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
    }


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery training run root 必须是 K 盘目录")
    return root


def _partition_observations(
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, tuple[dict[str, object], ...]]:
    """按许可文件冻结 observation，不允许 policy 跨分区或落空。"""
    result = {}
    covered = []
    for name, _role, policies, _license in OBSERVATION_FILE_ROLES:
        values = tuple(item for item in observations
                       if item.get("source_policy_scope") in policies)
        if not values or {item["source_policy_scope"] for item in values} != set(
                policies):
            raise BroadQaExternalDataError(
                "recovery training observation license partition 漂移")
        result[name] = values
        covered.extend(str(item["observation_id"]) for item in values)
    if (len(covered) != len(observations)
            or len(set(covered)) != len(observations)):
        raise BroadQaExternalDataError(
            "recovery training observation partition 覆盖漂移")
    return result


def _derive_from_sources(
        *,
        opencc_source_pack_dir: Path,
        icu_source_pack_dir: Path,
        successor_source_pack_dir: Path,
        evaluation_protocol_dir: Path,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """先证明 evaluation freeze，再从三份来源派生全部 TRAIN material。"""
    evaluation_manifest = read_normalization_recovery_evaluation_manifest_only(
        evaluation_protocol_dir,
        expected_manifest_sha256=(
            RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256),
    )
    if (evaluation_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_EVALUATION_PROTOCOL_KIND
            or evaluation_manifest.get("status")
            != NORMALIZATION_RECOVERY_EVALUATION_STATUS):
        raise BroadQaExternalDataError(
            "recovery evaluation protocol freeze 漂移")

    opencc_manifest = read_normalization_source_pack(opencc_source_pack_dir)
    icu_manifest, _variables, icu_rules = read_normalization_icu_source_pack(
        icu_source_pack_dir)
    successor_manifest, unihan_records, mediawiki_records = (
        read_normalization_successor_source_pack(successor_source_pack_dir))
    actual_shas = (
        opencc_manifest.get("manifest_sha256"),
        icu_manifest.get("manifest_sha256"),
        successor_manifest.get("manifest_sha256"),
    )
    expected_shas = (
        OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256,
        ICU_RECOVERY_SOURCE_MANIFEST_SHA256,
        SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256,
    )
    if actual_shas != expected_shas:
        raise BroadQaExternalDataError(
            "recovery training source manifest identity 漂移")
    try:
        character_payload = (
            opencc_source_pack_dir / "dictionary/TSCharacters.txt").read_bytes()
        phrase_payload = (
            opencc_source_pack_dir / "dictionary/TSPhrases.txt").read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "recovery OpenCC training source 不可读") from error

    roster = derive_normalization_recovery_source_roster(
        opencc_manifest=opencc_manifest,
        icu_manifest=icu_manifest,
        successor_manifest=successor_manifest,
    )
    observation_partitions = (
        derive_opencc_recovery_observations(
            roster=roster,
            character_payload=character_payload,
            phrase_payload=phrase_payload,
        ),
        derive_icu_recovery_observations(roster=roster, rules=icu_rules),
        derive_unihan_recovery_observations(
            roster=roster, records=unihan_records),
        derive_mediawiki_recovery_observations(
            roster=roster, records=mediawiki_records),
    )
    observations = tuple(sorted(
        (item for partition in observation_partitions for item in partition),
        key=lambda item: str(item["observation_id"])))
    groups = derive_normalization_recovery_groups(
        roster=roster, observations=observations)
    compositions = derive_normalization_recovery_compositions(
        observations=observations, groups=groups)
    loso = derive_normalization_recovery_loso(
        roster=roster, observations=observations)
    summary = normalization_recovery_training_summary(
        roster=roster,
        observations=observations,
        groups=groups,
        compositions=compositions,
        loso=loso,
    )
    work = _work_records(roster, observations, groups, compositions)
    _partition_observations(observations)
    return roster, observations, groups, compositions, loso, work, summary


def publish_normalization_recovery_training_protocol(
        *,
        run_root: str | Path,
        opencc_source_pack_dir: str | Path,
        icu_source_pack_dir: str | Path,
        successor_source_pack_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 Firefox payload 零读取前提下不可覆盖发布 recovery TRAIN。"""
    root = _require_k_root(run_root)
    sources = tuple(Path(value).resolve() for value in (
        opencc_source_pack_dir,
        icu_source_pack_dir,
        successor_source_pack_dir,
        evaluation_protocol_dir,
    ))
    target = Path(target_dir).resolve()
    if (any(not path.is_dir() or not path.is_relative_to(root)
            for path in sources)
            or not target.is_relative_to(root)
            or len(set(sources)) != len(sources)
            or target in sources):
        raise BroadQaExternalDataError(
            "recovery training source/evaluation/target 越出或混淆 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization recovery training target 已存在")
    derived = _derive_from_sources(
        opencc_source_pack_dir=sources[0],
        icu_source_pack_dir=sources[1],
        successor_source_pack_dir=sources[2],
        evaluation_protocol_dir=sources[3],
    )
    roster, observations, groups, compositions, loso, work, summary = derived
    observation_partitions = _partition_observations(observations)
    values_by_name = {
        "train.roster.jsonl": roster,
        **observation_partitions,
        "train.groups.jsonl": groups,
        "train.compositions.jsonl": compositions,
        "train.audit.loso.jsonl": loso,
        "train.work.jsonl": work,
    }
    target.mkdir(parents=True)
    for name, _role in TRAINING_FILE_ROLES:
        _write_jsonl(target / name, values_by_name[name])
    files = [
        _artifact(target / name, role=role, count=len(values_by_name[name]))
        for name, role in TRAINING_FILE_ROLES
    ]
    learner_contract = _contract(summary=summary, work=work)
    manifest = _manifest(
        files=files, summary=summary, learner_contract=learner_contract)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def _read_manifest(protocol_dir: Path) -> tuple[dict[str, object], bytes]:
    """只读取并核验 protocol manifest 的规范编码。"""
    try:
        encoded = (protocol_dir / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "recovery training protocol manifest 不可读") from error
    if (not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded):
        raise BroadQaExternalDataError(
            "recovery training protocol manifest 非规范")
    return manifest, encoded


def _read_learner_material(
        protocol_dir: Path,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """只打开 learner 获准读取的本地文件，排除 LOSO 审计文件。"""
    roster = _read_jsonl(
        protocol_dir / "train.roster.jsonl", label="recovery roster")
    partitions = tuple(_read_jsonl(
        protocol_dir / name, label=role)
        for name, role, _policies, _license in OBSERVATION_FILE_ROLES)
    for partition, (_name, role, policies, _license) in zip(
            partitions, OBSERVATION_FILE_ROLES):
        if (not partition
                or {item.get("source_policy_scope") for item in partition}
                != set(policies)):
            raise BroadQaExternalDataError(
                f"{role} license/policy partition 漂移")
    observations = tuple(sorted(
        (item for partition in partitions for item in partition),
        key=lambda item: str(item.get("observation_id", ""))))
    groups = _read_jsonl(
        protocol_dir / "train.groups.jsonl", label="recovery groups")
    compositions = _read_jsonl(
        protocol_dir / "train.compositions.jsonl",
        label="recovery compositions")
    work = _read_jsonl(
        protocol_dir / "train.work.jsonl", label="recovery work")
    return roster, observations, groups, compositions, work


def read_normalization_recovery_learner_input(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """按外部 manifest 身份回读 learner 输入，不打开 source/eval/LOSO。"""
    root = Path(protocol_dir).resolve()
    manifest, encoded = _read_manifest(root)
    expected_sha = _sha_value(
        expected_manifest_sha256,
        label="recovery training expected manifest SHA")
    if _sha256(encoded) != expected_sha:
        raise BroadQaExternalDataError(
            "recovery training protocol manifest identity 漂移")
    roster, observations, groups, compositions, work = (
        _read_learner_material(root))
    derived_groups = derive_normalization_recovery_groups(
        roster=roster, observations=observations)
    derived_compositions = derive_normalization_recovery_compositions(
        observations=observations, groups=derived_groups)
    derived_loso = derive_normalization_recovery_loso(
        roster=roster, observations=observations)
    summary = normalization_recovery_training_summary(
        roster=roster,
        observations=observations,
        groups=derived_groups,
        compositions=derived_compositions,
        loso=derived_loso,
    )
    derived_work = _work_records(
        roster, observations, derived_groups, derived_compositions)
    _partition_observations(observations)
    if (not _strict_equal(groups, derived_groups)
            or not _strict_equal(compositions, derived_compositions)
            or not _strict_equal(work, derived_work)):
        raise BroadQaExternalDataError(
            "recovery learner material 内部派生漂移")
    values_by_name = {
        "train.roster.jsonl": roster,
        **_partition_observations(observations),
        "train.groups.jsonl": groups,
        "train.compositions.jsonl": compositions,
        "train.work.jsonl": work,
    }
    files = []
    for name, role in TRAINING_FILE_ROLES:
        if name == "train.audit.loso.jsonl":
            files.append(_derived_artifact(
                name=name, role=role, values=derived_loso))
        else:
            files.append(_artifact(
                root / name, role=role, count=len(values_by_name[name])))
    expected = _manifest(
        files=files,
        summary=summary,
        learner_contract=_contract(summary=summary, work=work),
    )
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(
            "recovery training protocol manifest 漂移")
    return (
        {**manifest, "manifest_sha256": expected_sha},
        roster, observations, groups, compositions, work,
    )


def read_normalization_recovery_training_protocol(
        protocol_dir: str | Path,
        *,
        opencc_source_pack_dir: str | Path,
        icu_source_pack_dir: str | Path,
        successor_source_pack_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """auditor 从 evaluation freeze 与三份来源重派生完整协议。"""
    root = Path(protocol_dir).resolve()
    derived = _derive_from_sources(
        opencc_source_pack_dir=Path(opencc_source_pack_dir).resolve(),
        icu_source_pack_dir=Path(icu_source_pack_dir).resolve(),
        successor_source_pack_dir=Path(successor_source_pack_dir).resolve(),
        evaluation_protocol_dir=Path(evaluation_protocol_dir).resolve(),
    )
    roster, observations, groups, compositions, loso, work, summary = derived
    partitions = _partition_observations(observations)
    values_by_name = {
        "train.roster.jsonl": roster,
        **partitions,
        "train.groups.jsonl": groups,
        "train.compositions.jsonl": compositions,
        "train.audit.loso.jsonl": loso,
        "train.work.jsonl": work,
    }
    files = [
        _derived_artifact(name=name, role=role, values=values_by_name[name])
        for name, role in TRAINING_FILE_ROLES
    ]
    expected_manifest = _manifest(
        files=files,
        summary=summary,
        learner_contract=_contract(summary=summary, work=work),
    )
    stored = read_normalization_recovery_learner_input(
        root,
        expected_manifest_sha256=_sha256(
            canonical_json_line(expected_manifest)),
    )
    stored_loso = _read_jsonl(
        root / "train.audit.loso.jsonl", label="recovery LOSO audit")
    expected_loso_artifact = next(
        item for item in files
        if item["relative_path"] == "train.audit.loso.jsonl")
    if (any(not _strict_equal(left, right)
            for left, right in zip(stored[1:], (
                roster, observations, groups, compositions, work)))
            or not _strict_equal(stored_loso, loso)
            or not _strict_equal(
                _artifact(
                    root / "train.audit.loso.jsonl",
                    role="TRAIN_ONLY_LOSO_AUDIT", count=len(stored_loso)),
                expected_loso_artifact)):
        raise BroadQaExternalDataError(
            "recovery training protocol/source 或 LOSO 漂移")
    return (
        stored[0], roster, observations, groups, compositions, loso, work)


__all__ = [
    "ICU_RECOVERY_SOURCE_MANIFEST_SHA256",
    "NORMALIZATION_RECOVERY_AUTHORITY_CONTRACT",
    "NORMALIZATION_RECOVERY_LEARNER_CONTRACT",
    "NORMALIZATION_RECOVERY_TRAINING_PROTOCOL_KIND",
    "NORMALIZATION_RECOVERY_TRAINING_STATUS",
    "OPENCC_RECOVERY_SOURCE_MANIFEST_SHA256",
    "RECOVERY_EVALUATION_PROTOCOL_MANIFEST_SHA256",
    "SUCCESSOR_RECOVERY_SOURCE_MANIFEST_SHA256",
    "publish_normalization_recovery_training_protocol",
    "read_normalization_recovery_learner_input",
    "read_normalization_recovery_training_protocol",
]
