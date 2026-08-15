"""发布并严格回读 recovery-v8 三家统一 Observation pack。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_overlap_audit import (
    KEEPASSXC_SOURCE_PACK_MANIFEST_SHA256,
    QBITTORRENT_SOURCE_PACK_MANIFEST_SHA256,
    STELLARIUM_SOURCE_PACK_MANIFEST_SHA256,
    read_normalization_recovery_v8_source_overlap_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack import (
    read_normalization_recovery_v8_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_V1")
NORMALIZATION_RECOVERY_V8_OBSERVATION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_RECORD_V1")
NORMALIZATION_RECOVERY_V8_OBSERVATION_SOURCE_FILE_KIND = (
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_SOURCE_FILE_V1")
NORMALIZATION_RECOVERY_V8_OBSERVATION_FAMILY_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_FAMILY_CENSUS_V1")
NORMALIZATION_RECOVERY_V8_OBSERVATION_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_CENSUS_V1")
NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_STATUS = (
    "THREE_FAMILY_OBSERVATIONS_FROZEN_NOT_TRAINED")

V8_SOURCE_OVERLAP_MANIFEST_SHA256 = (
    "21dcc689ca8f3016d2ff0ab54b8476e0ac89425e1e6f8e25cd2b32a75d4cb122")

_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)
_PACK_SHA = {
    "QBITTORRENT_PROJECT": QBITTORRENT_SOURCE_PACK_MANIFEST_SHA256,
    "STELLARIUM_PROJECT": STELLARIUM_SOURCE_PACK_MANIFEST_SHA256,
    "KEEPASSXC_PROJECT": KEEPASSXC_SOURCE_PACK_MANIFEST_SHA256,
}
_OBSERVATION_FILES = (
    ("QBITTORRENT_PROJECT", "qbittorrent-observations.jsonl"),
    ("STELLARIUM_PROJECT", "stellarium-observations.jsonl"),
    ("KEEPASSXC_PROJECT", "keepassxc-observations.jsonl"),
)
_OUTPUT_FILES = (
    ("source-files.jsonl", "V8_OBSERVATION_SOURCE_FILES"),
    *((name, "V8_FAMILY_OBSERVATIONS")
      for _family, name in _OBSERVATION_FILES),
)
_OUTPUT_FILES = tuple(_OUTPUT_FILES) + (
    ("family-census.jsonl", "V8_OBSERVATION_FAMILY_CENSUS"),
    ("observation-census.jsonl", "V8_OBSERVATION_CENSUS"),
)
_PAIR_FEATURES = (
    "contains_han_both",
    "equal_length",
    "identity_preservation",
    "single_han_difference",
    "structure_equal",
    "training_eligible",
    "within_scalar_limit",
    "v8_training_eligible",
)


def _sha256(payload: bytes) -> str:
    """返回artifact、record或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v8 observation run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 observation {label} 越出run root") from error
    return path


def _overlap(left: Path, right: Path) -> bool:
    """判断两个已解析路径是否互为祖先。"""
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _record_id(value: dict[str, object]) -> str:
    """形成Observation或source-file wrapper的规范identity。"""
    return _sha256(canonical_json_line(value))


def _source_file_commitment(value: dict[str, object]) -> dict[str, object]:
    """从locale record提取显式source-file回链。"""
    file_id = value.get("source_file_id")
    sha = value.get("source_file_sha256")
    if (not isinstance(file_id, str) or len(file_id) != 64
            or not isinstance(sha, str) or len(sha) != 64):
        raise BroadQaExternalDataError(
            "v8 observation source-file commitment 漂移")
    return {"file_id": file_id, "sha256": sha}


def _exclusion_reasons(features: dict[str, int]) -> tuple[str, ...]:
    """从冻结pair facts形成不互相覆盖的显式资格原因。"""
    reasons = []
    if features["structure_equal"] == 0:
        reasons.append("STRUCTURE_UNEQUAL")
    if features["within_scalar_limit"] == 0:
        reasons.append("SCALAR_LIMIT")
    if features["contains_han_both"] == 0:
        reasons.append("NO_HAN_BOTH")
    if (features["training_eligible"]
            != int(features["structure_equal"] == 1
                   and features["within_scalar_limit"] == 1)
            or features["v8_training_eligible"]
            != int(features["training_eligible"] == 1
                   and features["contains_han_both"] == 1)):
        raise BroadQaExternalDataError(
            "v8 observation eligibility facts 漂移")
    return tuple(reasons)


def _observation(pair: dict[str, object]) -> dict[str, object]:
    """把完整source pair提升为不丢字段的统一Observation。"""
    features = {}
    for name in _PAIR_FEATURES:
        value = pair.get(name)
        if type(value) is not int or value not in (0, 1):
            raise BroadQaExternalDataError(
                "v8 observation pair feature 漂移")
        features[name] = value
    family = pair.get("source_family")
    pair_id = pair.get("pair_id")
    pair_kind = pair.get("record_kind")
    identity = pair.get("source_identity")
    official_source = pair.get("official_source_text")
    policy = pair.get("source_policy_scope")
    license_expression = pair.get("license_expression")
    hans = pair.get("zh_hans")
    hant = pair.get("zh_hant")
    hans_tokens = pair.get("zh_hans_structure_tokens")
    hant_tokens = pair.get("zh_hant_structure_tokens")
    if (family not in _FAMILIES
            or not isinstance(pair_id, str) or len(pair_id) != 64
            or not isinstance(pair_kind, str) or not pair_kind
            or not isinstance(identity, dict)
            or not isinstance(official_source, str)
            or not isinstance(policy, str) or not policy
            or not isinstance(license_expression, str)
            or not isinstance(hans, dict) or not isinstance(hant, dict)
            or not isinstance(hans_tokens, list)
            or not isinstance(hant_tokens, list)
            or any(not isinstance(item, str)
                   for item in hans_tokens + hant_tokens)):
        raise BroadQaExternalDataError(
            "v8 observation source pair schema 漂移")
    reasons = _exclusion_reasons(features)
    observation_identity = {
        "source_family": family,
        "source_pair_id": pair_id,
        "source_pair_record_kind": pair_kind,
    }
    return {
        "eligibility": {
            "exclusion_reasons": list(reasons),
            "pair_features": features,
            "status": (
                "V8_TRAINING_ELIGIBLE" if not reasons
                else "V8_TRAINING_EXCLUDED"),
        },
        "format_version": 1,
        "license_expression": license_expression,
        "observation_id": _record_id(observation_identity),
        "official_source_text": official_source,
        "record_kind": NORMALIZATION_RECOVERY_V8_OBSERVATION_RECORD_KIND,
        "source_family": family,
        "source_file_commitments": {
            "zh_Hans": _source_file_commitment(hans),
            "zh_Hant": _source_file_commitment(hant),
        },
        "source_identity": identity,
        "source_identity_sha256": pair.get("source_identity_sha256"),
        "source_pair_id": pair_id,
        "source_pair_record_kind": pair_kind,
        "source_policy_scope": policy,
        "zh_hans": hans,
        "zh_hans_structure_tokens": hans_tokens,
        "zh_hant": hant,
        "zh_hant_structure_tokens": hant_tokens,
    }


def _parser_exclusion_counts(
        parser_summary: dict[str, object],
        ) -> dict[str, int]:
    """汇总未形成pair的locale-entry parser排除账。"""
    locale_summaries = parser_summary.get("locale_summaries")
    if not isinstance(locale_summaries, dict):
        raise BroadQaExternalDataError(
            "v8 observation parser locale census 漂移")
    mapping = {
        "empty_active_translation_count": "EMPTY_TRANSLATION",
        "empty_translation_count": "EMPTY_TRANSLATION",
        "fuzzy_count": "FUZZY",
        "numerus_count": "NUMERUS_OR_PLURAL",
        "plural_count": "NUMERUS_OR_PLURAL",
        "obsolete_count": "OBSOLETE",
        "unfinished_count": "UNFINISHED",
        "vanished_count": "VANISHED",
    }
    counts = Counter()
    for domains in locale_summaries.values():
        if not isinstance(domains, dict):
            raise BroadQaExternalDataError(
                "v8 observation parser domain census 漂移")
        for census in domains.values():
            if not isinstance(census, dict):
                raise BroadQaExternalDataError(
                    "v8 observation parser source census 漂移")
            for source_key, target_key in mapping.items():
                value = census.get(source_key, 0)
                if type(value) is not int or value < 0:
                    raise BroadQaExternalDataError(
                        "v8 observation parser exclusion count 漂移")
                counts[target_key] += value
    return {key: counts[key] for key in sorted(set(mapping.values()))}


def _source_file_observations(
        family: str,
        source_files: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """为source-file records增加family与source-pack回链而不改原记录。"""
    values = []
    for item in source_files:
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or len(file_id) != 64:
            raise BroadQaExternalDataError(
                "v8 observation source-file record 漂移")
        identity = {"source_family": family, "source_file_id": file_id}
        values.append({
            "format_version": 1,
            "observation_source_file_id": _record_id(identity),
            "record_kind": NORMALIZATION_RECOVERY_V8_OBSERVATION_SOURCE_FILE_KIND,
            "source_family": family,
            "source_file": item,
            "source_pack_manifest_sha256": _PACK_SHA[family],
        })
    return tuple(values)


def _family_census(
        family: str,
        *,
        manifest: dict[str, object],
        source_files: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """形成pair排除与parser排除分账的family census。"""
    reasons = Counter()
    identity_count = 0
    eligible_count = 0
    for item in observations:
        eligibility = item["eligibility"]
        for reason in eligibility["exclusion_reasons"]:
            reasons[reason] += 1
        features = eligibility["pair_features"]
        identity_count += int(features["identity_preservation"])
        eligible_count += int(features["v8_training_eligible"])
    parser_summary = manifest.get("parser_summary")
    if not isinstance(parser_summary, dict):
        raise BroadQaExternalDataError(
            "v8 observation source-pack parser summary 漂移")
    return {
        "family_vote_count": 1,
        "format_version": 1,
        "identity_pair_count": identity_count,
        "observation_count": len(observations),
        "pair_exclusion_reason_counts": {
            key: reasons[key] for key in (
                "NO_HAN_BOTH", "SCALAR_LIMIT", "STRUCTURE_UNEQUAL")
        },
        "parser_stage_exclusion_locale_entry_counts": (
            _parser_exclusion_counts(parser_summary)),
        "parser_stage_locale_summaries": parser_summary["locale_summaries"],
        "record_kind": NORMALIZATION_RECOVERY_V8_OBSERVATION_FAMILY_CENSUS_KIND,
        "source_family": family,
        "source_file_record_count": len(source_files),
        "source_format_policy": parser_summary["source_format_policy"],
        "source_pack_manifest_sha256": _PACK_SHA[family],
        "v8_training_eligible_count": eligible_count,
        "v8_training_excluded_count": len(observations) - eligible_count,
    }


def _derive(
        manifests: dict[str, dict[str, object]],
        source_files: dict[str, tuple[dict[str, object], ...]],
        pairs: dict[str, tuple[dict[str, object], ...]],
        overlap_manifest: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """从三份source pack全量重派生Observation与两级census。"""
    if (set(manifests) != set(_FAMILIES)
            or set(source_files) != set(_FAMILIES)
            or set(pairs) != set(_FAMILIES)
            or overlap_manifest.get("status")
            != "THREE_INDEPENDENT_FAMILIES_NO_LOCALE_OR_SUBSET_COPY"):
        raise BroadQaExternalDataError(
            "v8 observation predecessor inventory 漂移")
    observations = {
        family: tuple(_observation(pair) for pair in pairs[family])
        for family in _FAMILIES
    }
    all_source_files = tuple(
        item
        for family in _FAMILIES
        for item in _source_file_observations(
            family, source_files[family]))
    family_census = tuple(_family_census(
        family,
        manifest=manifests[family],
        source_files=source_files[family],
        observations=observations[family],
    ) for family in _FAMILIES)
    global_summary = {
        "family_count": 3,
        "family_vote_count": 3,
        "identity_pair_count": sum(
            int(item["identity_pair_count"]) for item in family_census),
        "observation_count": sum(
            int(item["observation_count"]) for item in family_census),
        "pair_surface_public_git_count": 0,
        "source_file_record_count": len(all_source_files),
        "train_protocol_published": 0,
        "v8_training_eligible_count": sum(
            int(item["v8_training_eligible_count"])
            for item in family_census),
        "v8_training_excluded_count": sum(
            int(item["v8_training_excluded_count"])
            for item in family_census),
    }
    global_census = ({
        **global_summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_OBSERVATION_CENSUS_KIND,
    },)
    outputs = {"source-files.jsonl": all_source_files}
    for family, name in _OBSERVATION_FILES:
        outputs[name] = observations[family]
    outputs["family-census.jsonl"] = family_census
    outputs["observation-census.jsonl"] = global_census
    return outputs, global_summary


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范JSONL。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != line):
                    raise BroadQaExternalDataError(
                        f"v8 observation {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 observation {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成Observation输出文件commitment。"""
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": count,
        "relative_path": path.name,
        "role": role,
        "sha256": _sha256(payload),
    }


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造全量Observation冻结、尚未训练的manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {
            "keepassxc_source_pack_manifest_sha256": (
                KEEPASSXC_SOURCE_PACK_MANIFEST_SHA256),
            "qbittorrent_source_pack_manifest_sha256": (
                QBITTORRENT_SOURCE_PACK_MANIFEST_SHA256),
            "source_overlap_manifest_sha256": (
                V8_SOURCE_OVERLAP_MANIFEST_SHA256),
            "stellarium_source_pack_manifest_sha256": (
                STELLARIUM_SOURCE_PACK_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _state(
        *,
        source_pack_dirs: dict[str, Path],
        source_overlap_dir: Path,
        v2_roster_dir: Path,
        v1_roster_dir: Path,
        v1_content_audit_dir: Path,
        v2_content_audit_dir: Path,
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
        ]:
    """严格回读三份source pack与sealed overlap aggregate。"""
    if set(source_pack_dirs) != set(_FAMILIES):
        raise BroadQaExternalDataError(
            "v8 observation source-pack path inventory 漂移")
    overlap_manifest, _overlap_outputs = (
        read_normalization_recovery_v8_source_overlap_aggregate(
            source_overlap_dir,
            expected_manifest_sha256=V8_SOURCE_OVERLAP_MANIFEST_SHA256,
        ))
    manifests = {}
    source_files = {}
    pairs = {}
    for family in _FAMILIES:
        manifest, file_records, pair_records, _census = (
            read_normalization_recovery_v8_source_pack(
                source_pack_dirs[family],
                v2_roster_dir=v2_roster_dir,
                v1_roster_dir=v1_roster_dir,
                v1_content_audit_dir=v1_content_audit_dir,
                v2_content_audit_dir=v2_content_audit_dir,
                expected_manifest_sha256=_PACK_SHA[family],
            ))
        manifests[family] = manifest
        source_files[family] = file_records
        pairs[family] = pair_records
    return manifests, source_files, pairs, overlap_manifest


def publish_normalization_recovery_v8_observation_pack(
        *,
        run_root: str | Path,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        source_overlap_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布三家全量、按family分区的Observation pack。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      v2_roster_dir,
                      v1_roster_dir,
                      v1_content_audit_dir,
                      v2_content_audit_dir,
                      source_overlap_dir,
                      qbittorrent_source_pack_dir,
                      stellarium_source_pack_dir,
                      keepassxc_source_pack_dir,
                      target_dir,
                  )))
    (v2_roster, v1_roster, v1_content, v2_content, overlap,
     qbit, stellarium, keepassxc, target) = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v8 observation input/target path 非法")
    state = _state(
        source_pack_dirs={
            "QBITTORRENT_PROJECT": qbit,
            "STELLARIUM_PROJECT": stellarium,
            "KEEPASSXC_PROJECT": keepassxc,
        },
        source_overlap_dir=overlap,
        v2_roster_dir=v2_roster,
        v1_roster_dir=v1_roster,
        v1_content_audit_dir=v1_content,
        v2_content_audit_dir=v2_content,
    )
    outputs, summary = _derive(*state)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_observation_pack(
        observation_dir: str | Path,
        *,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        source_overlap_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生全量Observation并拒绝records/manifest同步篡改。"""
    root = Path(observation_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 observation manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 observation manifest identity 漂移")
    state = _state(
        source_pack_dirs={
            "QBITTORRENT_PROJECT": Path(
                qbittorrent_source_pack_dir).resolve(),
            "STELLARIUM_PROJECT": Path(
                stellarium_source_pack_dir).resolve(),
            "KEEPASSXC_PROJECT": Path(
                keepassxc_source_pack_dir).resolve(),
        },
        source_overlap_dir=Path(source_overlap_dir).resolve(),
        v2_roster_dir=Path(v2_roster_dir).resolve(),
        v1_roster_dir=Path(v1_roster_dir).resolve(),
        v1_content_audit_dir=Path(v1_content_audit_dir).resolve(),
        v2_content_audit_dir=Path(v2_content_audit_dir).resolve(),
    )
    expected_outputs, summary = _derive(*state)
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v8 observation records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v8 observation fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_KIND",
    "NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_STATUS",
    "V8_SOURCE_OVERLAP_MANIFEST_SHA256",
    "publish_normalization_recovery_v8_observation_pack",
    "read_normalization_recovery_v8_observation_pack",
]
