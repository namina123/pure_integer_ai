"""冻结 recovery-v4 三来源 phrase/context TRAIN protocol。

publisher 先严格回读标签盲 evaluation commitment，再读取 Godot、Thunderbird、
VS Code 三套独立 source pack 并物化 learner material。learner-only reader 只读取
本目录四份 TRAIN 文件，不能打开来源、Firefox inventory/reserve 或历史 rule pack。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_evaluation_commitment import (
    read_normalization_recovery_v3_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_godot_source_pack import (
    read_normalization_recovery_v3_godot_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_thunderbird_source_pack import (
    read_normalization_recovery_v3_thunderbird_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    derive_normalization_recovery_v4_fragments,
    derive_normalization_recovery_v4_groups,
    derive_normalization_recovery_v4_pair_observations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    read_normalization_recovery_v4_vscode_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V4_TRAINING_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_TRAINING_PROTOCOL_V1")
NORMALIZATION_RECOVERY_V4_TRAINING_STATUS = "FROZEN_NOT_READ_NOT_LEARNED"
NORMALIZATION_RECOVERY_V4_WORK_KIND = (
    "NORMALIZATION_RECOVERY_V4_WORK_ITEM_V1")

V4_EVALUATION_COMMITMENT_MANIFEST_SHA256 = (
    "a02e561e823d17e76cee57334903bf3ff0c49191b0b741f2a9497c500e365e36")
V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256 = (
    "aa481d0f1f0c4f5fe9b57826f2f81e5af31542c78831b04e926508c0e4cbca54")
V4_GODOT_SOURCE_PACK_MANIFEST_SHA256 = (
    "54a6308cf3dafc79cd9db336cf4ea983c6fcbb1d0d0f9c9225956992c13d05c2")
V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256 = (
    "10fcdfd37503e3c2058b28dbd7a2d3cfa13ef4e301e4d1c64b1dbf563995b27c")
V4_BASE_RECOVERY_RULE_PACK_MANIFEST_SHA256 = (
    "a676340af3717c078069c4c80535df66f13af43b675cdb69000d281101fcf21c")
V4_PREDECESSOR_RULE_PACK_MANIFEST_SHA256 = (
    "afb87e80b68169a6ff66949c74a1cb350f9d10d20171107aff475eb809cbb665")

TRAINING_FILES = (
    ("train.pair-observations.jsonl", "TRAIN_PAIR_OBSERVATIONS", "observation_id"),
    ("train.phrase-fragments.jsonl", "TRAIN_PHRASE_FRAGMENTS", "fragment_id"),
    ("train.phrase-groups.jsonl", "TRAIN_PHRASE_GROUPS", "group_id"),
    ("train.work.jsonl", "TRAIN_ORDERED_WORK", "work_id"),
)
TRAINING_PHASES = (
    "PAIR_OBSERVATION_INGEST",
    "PHRASE_FRAGMENT_INGEST",
    "PHRASE_GROUP_RESOLUTION",
)


def _sha256(payload: bytes) -> str:
    """返回规范 artifact identity。"""
    return hashlib.sha256(payload).hexdigest()


def _strict_equal(value: object, expected: object) -> bool:
    """递归比较 JSON 值并区分 bool 与 int。"""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(value) == set(expected)
                and all(_strict_equal(value[key], expected[key])
                        for key in expected))
    if isinstance(expected, (list, tuple)):
        return (len(value) == len(expected)
                and all(_strict_equal(item, expected_item)
                        for item, expected_item in zip(value, expected)))
    return value == expected


def _work_records(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按三阶段构造 learner 必须完整消费的确定序 work。"""
    sources = (
        (TRAINING_PHASES[0], "PAIR_OBSERVATION", observations, "observation_id"),
        (TRAINING_PHASES[1], "PHRASE_FRAGMENT", fragments, "fragment_id"),
        (TRAINING_PHASES[2], "PHRASE_GROUP", groups, "group_id"),
    )
    values = []
    ordinal = 0
    for phase, kind, records, identity_key in sources:
        for record in records:
            identity = {
                "phase": phase,
                "record_id": record[identity_key],
                "work_kind": kind,
            }
            values.append({
                **identity,
                "format_version": 1,
                "record_kind": NORMALIZATION_RECOVERY_V4_WORK_KIND,
                "work_id": _sha256(canonical_json_bytes(identity)),
                "work_ordinal": ordinal,
            })
            ordinal += 1
    return tuple(values)


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
    if not values:
        raise BroadQaExternalDataError(f"{label} JSONL 为空")
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """返回物化文件 identity。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _summary(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """返回来源、长度、冲突与 candidate scope 分账。"""
    return {
        "candidate_scope_counts": dict(sorted(Counter(
            str(item["candidate_scope_kind"]) for item in groups).items())),
        "cross_family_target_candidate_count": sum(
            item["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
            for item in groups),
        "disposition_counts": dict(sorted(Counter(
            str(item["disposition"]) for item in groups).items())),
        "equal_length_fragment_count": sum(
            item["equal_length"] == 1 for item in fragments),
        "fragment_count": len(fragments),
        "fragment_kind_counts": dict(sorted(Counter(
            str(item["fragment_kind"]) for item in fragments).items())),
        "group_count": len(groups),
        "identity_observation_count": sum(
            item["identity_preservation"] == 1 for item in observations),
        "observation_count": len(observations),
        "source_family_count": len({
            str(item["source_family"]) for item in observations}),
        "source_family_counts": dict(sorted(Counter(
            str(item["source_family"]) for item in observations).items())),
        "source_scoped_candidate_count": sum(
            item["candidate_scope_kind"] == "SOURCE_ONLY"
            for item in groups),
        "variable_length_fragment_count": sum(
            item["equal_length"] == 0 for item in fragments),
        "work_count": len(work),
        "work_identity_sha256": _work_identity(work),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造标签盲分母之后冻结的 v4 TRAIN manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_TRAINING_PROTOCOL_KIND,
        "base_rule_pack_manifest_sha256": (
            V4_BASE_RECOVERY_RULE_PACK_MANIFEST_SHA256),
        "base_rule_pack_read_count": 0,
        "evaluation_commitment_manifest_sha256": (
            V4_EVALUATION_COMMITMENT_MANIFEST_SHA256),
        "evaluation_or_reserve_payload_read_count": 0,
        "files": files,
        "format_version": 1,
        "learner_contract": {
            "base_rule_pack_ui_target_execution_allowed": 0,
            "candidate_group_is_not_executable_rule": 1,
            "evaluation_or_reserve_read_allowed": 0,
            "fresh_resume_byte_equivalence_required": 1,
            "longest_match_and_overlap_resolution_required": 1,
            "negative_evidence_required_before_execution": 1,
            "source_pack_read_allowed": 0,
            "source_scoped_candidate_target_upgrade_allowed": 0,
            "target_candidate_min_distinct_ui_source_family_count": 2,
            "three_source_leave_one_out_required": 1,
            "whole_input_exact_precedes_phrase_lexicon": 1,
        },
        "license_partitions": [
            {"license_id": "MIT", "source_family": "GODOT_ENGINE_PROJECT"},
            {"license_id": "MIT", "source_family": VSCODE_SOURCE_FAMILY},
            {"license_id": "MPL-2.0", "source_family": "THUNDERBIRD_PROJECT"},
        ],
        "mastery_claimed": 0,
        "predecessor_rule_pack_manifest_sha256": (
            V4_PREDECESSOR_RULE_PACK_MANIFEST_SHA256),
        "predecessor_rule_pack_read_count": 0,
        "production_enabled": 0,
        "source_pack_manifests": {
            "GODOT_ENGINE_PROJECT": V4_GODOT_SOURCE_PACK_MANIFEST_SHA256,
            "MICROSOFT_VSCODE_PROJECT": V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256,
            "THUNDERBIRD_PROJECT": V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256,
        },
        "status": NORMALIZATION_RECOVERY_V4_TRAINING_STATUS,
        "summary": summary,
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
    }


def _derive_from_sources(
        *,
        thunderbird_source_pack_dir: str | Path,
        godot_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """严格回读三源并派生全部 learner material。"""
    thunderbird_manifest, _files, thunderbird_pairs = (
        read_normalization_recovery_v3_thunderbird_source_pack(
            thunderbird_source_pack_dir))
    godot_manifest, _files, godot_pairs = (
        read_normalization_recovery_v3_godot_source_pack(
            godot_source_pack_dir))
    vscode_manifest, _files, vscode_pairs = (
        read_normalization_recovery_v4_vscode_source_pack(
            vscode_source_pack_dir))
    if (thunderbird_manifest["manifest_sha256"]
            != V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256
            or godot_manifest["manifest_sha256"]
            != V4_GODOT_SOURCE_PACK_MANIFEST_SHA256
            or vscode_manifest["manifest_sha256"]
            != V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256):
        raise BroadQaExternalDataError("v4 TRAIN source manifest 漂移")
    observations = derive_normalization_recovery_v4_pair_observations(
        thunderbird_manifest_sha256=V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256,
        thunderbird_pairs=thunderbird_pairs,
        godot_manifest_sha256=V4_GODOT_SOURCE_PACK_MANIFEST_SHA256,
        godot_pairs=godot_pairs,
        vscode_manifest_sha256=V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256,
        vscode_pairs=vscode_pairs,
    )
    fragments = derive_normalization_recovery_v4_fragments(observations)
    groups = derive_normalization_recovery_v4_groups(fragments)
    work = _work_records(observations, fragments, groups)
    return observations, fragments, groups, work


def _require_k_root(value: str | Path) -> Path:
    """要求显式工作根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization recovery v4 training run root 必须是 K 盘目录")
    return root


def publish_normalization_recovery_v4_training_protocol(
        *,
        run_root: str | Path,
        prior_evaluation_protocol_dir: str | Path,
        evaluation_commitment_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        godot_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """在 learner 运行前不可覆盖发布 v4 TRAIN protocol。"""
    root = _require_k_root(run_root)
    paths = [Path(value).resolve() for value in (
        prior_evaluation_protocol_dir,
        evaluation_commitment_dir,
        thunderbird_source_pack_dir,
        godot_source_pack_dir,
        vscode_source_pack_dir,
        target_dir,
    )]
    prior, commitment, thunderbird, godot, vscode, target = paths
    if (any(not path.is_relative_to(root) for path in paths)
            or any(not path.is_dir() for path in (
                prior, commitment, thunderbird, godot, vscode))):
        raise BroadQaExternalDataError("v4 TRAIN protocol path 越界或缺失")
    if target.exists():
        raise BroadQaExternalDataError("v4 TRAIN protocol target 已存在")
    read_normalization_recovery_v3_evaluation_commitment(
        commitment,
        prior_evaluation_protocol_dir=prior,
        expected_manifest_sha256=V4_EVALUATION_COMMITMENT_MANIFEST_SHA256,
    )
    observations, fragments, groups, work = _derive_from_sources(
        thunderbird_source_pack_dir=thunderbird,
        godot_source_pack_dir=godot,
        vscode_source_pack_dir=vscode,
    )
    target.mkdir()
    records = (observations, fragments, groups, work)
    files = []
    for (name, role, _identity), values in zip(TRAINING_FILES, records):
        path = target / name
        _write_jsonl(path, values)
        files.append(_artifact(path, role=role, count=len(values)))
    summary = _summary(observations, fragments, groups, work)
    manifest = _manifest(files=files, summary=summary)
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def _read_manifest(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[Path, dict[str, object], bytes]:
    """严格读取规范 manifest 与外部 SHA。"""
    root = Path(protocol_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v4 TRAIN manifest 不可读") from error
    if (not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(item not in "0123456789abcdef"
                   for item in expected_manifest_sha256)
            or _sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError("v4 TRAIN manifest identity/encoding 漂移")
    return root, stored, encoded


def read_normalization_recovery_v4_learner_input(
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
    """只读取物化 TRAIN 文件，不打开 source/evaluation/reserve。"""
    root, stored, encoded = _read_manifest(
        protocol_dir, expected_manifest_sha256=expected_manifest_sha256)
    values = []
    expected_files = []
    for name, role, _identity in TRAINING_FILES:
        records = _read_jsonl(root / name, label=role)
        values.append(records)
        expected_files.append(
            _artifact(root / name, role=role, count=len(records)))
    observations, fragments, groups, work = values
    expected = _manifest(
        files=expected_files,
        summary=_summary(observations, fragments, groups, work),
    )
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError("v4 TRAIN learner material 漂移")
    return (
        {**stored, "manifest_sha256": _sha256(encoded)},
        observations,
        fragments,
        groups,
        work,
    )


def read_normalization_recovery_v4_training_protocol(
        protocol_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        thunderbird_source_pack_dir: str | Path,
        godot_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        ) -> dict[str, object]:
    """auditor 从三套 source pack 重派生并逐字段核验协议。"""
    root, stored, encoded = _read_manifest(
        protocol_dir, expected_manifest_sha256=expected_manifest_sha256)
    observations, fragments, groups, work = _derive_from_sources(
        thunderbird_source_pack_dir=thunderbird_source_pack_dir,
        godot_source_pack_dir=godot_source_pack_dir,
        vscode_source_pack_dir=vscode_source_pack_dir,
    )
    records = (observations, fragments, groups, work)
    files = []
    for (name, role, _identity), derived in zip(TRAINING_FILES, records):
        stored_records = _read_jsonl(root / name, label=role)
        if not _strict_equal(stored_records, derived):
            raise BroadQaExternalDataError("v4 TRAIN records/source 漂移")
        files.append(_artifact(root / name, role=role, count=len(derived)))
    expected = _manifest(
        files=files, summary=_summary(observations, fragments, groups, work))
    if not _strict_equal(stored, expected):
        raise BroadQaExternalDataError("v4 TRAIN manifest/source 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}


__all__ = [
    "NORMALIZATION_RECOVERY_V4_TRAINING_PROTOCOL_KIND",
    "NORMALIZATION_RECOVERY_V4_TRAINING_STATUS",
    "V4_BASE_RECOVERY_RULE_PACK_MANIFEST_SHA256",
    "V4_EVALUATION_COMMITMENT_MANIFEST_SHA256",
    "V4_GODOT_SOURCE_PACK_MANIFEST_SHA256",
    "V4_PREDECESSOR_RULE_PACK_MANIFEST_SHA256",
    "V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256",
    "V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256",
    "publish_normalization_recovery_v4_training_protocol",
    "read_normalization_recovery_v4_learner_input",
    "read_normalization_recovery_v4_training_protocol",
]
