"""派生 recovery-v10 五 family、按family分区的统一Observation。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_SOURCE_EXPANSION_OBSERVATION_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_OBSERVATION_RECORD_V1")
V10_SOURCE_EXPANSION_OBSERVATION_SOURCE_FILE_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_OBSERVATION_SOURCE_FILE_V1")
V10_SOURCE_EXPANSION_OBSERVATION_FAMILY_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_OBSERVATION_FAMILY_CENSUS_V1")
V10_SOURCE_EXPANSION_OBSERVATION_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_OBSERVATION_CENSUS_V1")

V10_SOURCE_EXPANSION_OBSERVATION_FILES = (
    ("QBITTORRENT_PROJECT", "qbittorrent-observations.jsonl"),
    ("STELLARIUM_PROJECT", "stellarium-observations.jsonl"),
    ("KEEPASSXC_PROJECT", "keepassxc-observations.jsonl"),
    ("MIXXX_PROJECT", "mixxx-observations.jsonl"),
    ("MUMBLE_PROJECT", "mumble-observations.jsonl"),
)
_OLD_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)
_NEW_FAMILIES = ("MIXXX_PROJECT", "MUMBLE_PROJECT")
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
    """返回Observation或source-file wrapper的规范identity。"""
    return hashlib.sha256(payload).hexdigest()


def _record_id(value: dict[str, object]) -> str:
    """形成不依赖物理写盘位置的规范记录identity。"""
    return _sha256(canonical_json_line(value))


def _source_file_commitment(value: dict[str, object]) -> dict[str, object]:
    """从locale record提取显式source-file回链。"""
    file_id = value.get("source_file_id")
    sha = value.get("source_file_sha256")
    if (not isinstance(file_id, str) or len(file_id) != 64
            or not isinstance(sha, str) or len(sha) != 64):
        raise BroadQaExternalDataError(
            "v10 expanded observation source-file commitment 漂移")
    return {"file_id": file_id, "sha256": sha}


def _exclusion_reasons(features: dict[str, int]) -> tuple[str, ...]:
    """从冻结pair facts形成互不覆盖的显式资格原因。"""
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
            "v10 expanded observation eligibility facts 漂移")
    return tuple(reasons)


def _observation(pair: dict[str, object], *, family: str) -> dict[str, object]:
    """把新source pair提升为不丢字段的统一Observation。"""
    features = {}
    for name in _PAIR_FEATURES:
        value = pair.get(name)
        if type(value) is not int or value not in (0, 1):
            raise BroadQaExternalDataError(
                "v10 expanded observation pair feature 漂移")
        features[name] = value
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
    if (pair.get("source_family") != family
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
            "v10 expanded observation source pair schema 漂移")
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
        "record_kind": V10_SOURCE_EXPANSION_OBSERVATION_RECORD_KIND,
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
            "v10 expanded observation parser locale census 漂移")
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
                "v10 expanded observation parser domain census 漂移")
        for census in domains.values():
            if not isinstance(census, dict):
                raise BroadQaExternalDataError(
                    "v10 expanded observation parser source census 漂移")
            for source_key, target_key in mapping.items():
                value = census.get(source_key, 0)
                if type(value) is not int or value < 0:
                    raise BroadQaExternalDataError(
                        "v10 expanded observation parser exclusion 漂移")
                counts[target_key] += value
    return {key: counts[key] for key in sorted(set(mapping.values()))}


def _source_file_observations(
        family: str,
        *,
        manifest: dict[str, object],
        source_files: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """为新source-file records增加family与source-pack回链。"""
    values = []
    for item in source_files:
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or len(file_id) != 64:
            raise BroadQaExternalDataError(
                "v10 expanded observation source-file record 漂移")
        identity = {"source_family": family, "source_file_id": file_id}
        values.append({
            "format_version": 1,
            "observation_source_file_id": _record_id(identity),
            "record_kind": V10_SOURCE_EXPANSION_OBSERVATION_SOURCE_FILE_KIND,
            "source_family": family,
            "source_file": item,
            "source_pack_manifest_sha256": manifest["manifest_sha256"],
        })
    return tuple(values)


def _family_census(
        family: str,
        *,
        manifest: dict[str, object],
        source_files: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """形成新family的pair排除与parser排除分账。"""
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
            "v10 expanded observation parser summary 漂移")
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
        "record_kind": V10_SOURCE_EXPANSION_OBSERVATION_FAMILY_CENSUS_KIND,
        "source_family": family,
        "source_file_record_count": len(source_files),
        "source_format_policy": parser_summary["source_format_policy"],
        "source_pack_manifest_sha256": manifest["manifest_sha256"],
        "v8_training_eligible_count": eligible_count,
        "v8_training_excluded_count": len(observations) - eligible_count,
    }


def derive_normalization_recovery_v10_source_expansion_observations(
        predecessor_outputs: dict[str, tuple[dict[str, object], ...]],
        new_manifests: dict[str, dict[str, object]],
        new_source_files: dict[str, tuple[dict[str, object], ...]],
        new_pairs: dict[str, tuple[dict[str, object], ...]],
        audit_manifest: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """继承旧三家identity并提升新两家，形成五family Observation分区。"""
    old_observation_names = {
        name for family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES
        if family in _OLD_FAMILIES
    }
    required = {
        "source-files.jsonl",
        "family-census.jsonl",
        "observation-census.jsonl",
        *old_observation_names,
    }
    if (not required.issubset(predecessor_outputs)
            or set(new_manifests) != set(_NEW_FAMILIES)
            or set(new_source_files) != set(_NEW_FAMILIES)
            or set(new_pairs) != set(_NEW_FAMILIES)
            or audit_manifest.get("status")
            != "FIVE_INDEPENDENT_TRAIN_FAMILIES_COLLISIONS_FROZEN_NOT_OBSERVED"):
        raise BroadQaExternalDataError(
            "v10 expanded observation predecessor inventory 漂移")
    old_census = predecessor_outputs["observation-census.jsonl"]
    old_family_census = predecessor_outputs["family-census.jsonl"]
    if (len(old_census) != 1
            or old_census[0].get("observation_count") != 33_179
            or len(old_family_census) != 3
            or tuple(item.get("source_family") for item in old_family_census)
            != _OLD_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 expanded observation predecessor census 漂移")
    new_observations = {
        family: tuple(_observation(pair, family=family)
                      for pair in new_pairs[family])
        for family in _NEW_FAMILIES
    }
    wrapped_files = {
        family: _source_file_observations(
            family,
            manifest=new_manifests[family],
            source_files=new_source_files[family],
        ) for family in _NEW_FAMILIES
    }
    new_family_census = tuple(_family_census(
        family,
        manifest=new_manifests[family],
        source_files=new_source_files[family],
        observations=new_observations[family],
    ) for family in _NEW_FAMILIES)
    all_source_files = (
        predecessor_outputs["source-files.jsonl"]
        + tuple(item for family in _NEW_FAMILIES
                for item in wrapped_files[family]))
    family_census = old_family_census + new_family_census
    summary = {
        "cross_family_output_conflict_key_count": int(
            audit_manifest["summary"]["cross_family_output_conflict_key_count"]),
        "family_count": 5,
        "family_vote_count": 5,
        "identity_changed_mixed_key_count": int(
            audit_manifest["summary"]["identity_changed_mixed_key_count"]),
        "identity_pair_count": sum(
            int(item["identity_pair_count"]) for item in family_census),
        "observation_count": sum(
            int(item["observation_count"]) for item in family_census),
        "pair_surface_public_git_count": 0,
        "source_file_record_count": len(all_source_files),
        "source_input_collision_record_count": int(
            audit_manifest["summary"]["source_input_collision_record_count"]),
        "train_protocol_published": 0,
        "v8_training_eligible_count": sum(
            int(item["v8_training_eligible_count"]) for item in family_census),
        "v8_training_excluded_count": sum(
            int(item["v8_training_excluded_count"]) for item in family_census),
    }
    audit_summary = audit_manifest.get("summary")
    if (not isinstance(audit_summary, dict)
            or summary["observation_count"] != audit_summary.get(
                "pair_record_count")
            or summary["identity_pair_count"] != audit_summary.get(
                "identity_pair_count")
            or summary["v8_training_eligible_count"] != audit_summary.get(
                "training_eligible_pair_count")
            or summary["v8_training_excluded_count"] != audit_summary.get(
                "training_excluded_pair_count")):
        raise BroadQaExternalDataError(
            "v10 expanded observation audit denominator 漂移")
    outputs = {"source-files.jsonl": all_source_files}
    for family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES:
        outputs[name] = (
            predecessor_outputs[name] if family in _OLD_FAMILIES
            else new_observations[family])
    outputs["family-census.jsonl"] = family_census
    outputs["observation-census.jsonl"] = ({
        **summary,
        "format_version": 1,
        "record_kind": V10_SOURCE_EXPANSION_OBSERVATION_CENSUS_KIND,
    },)
    return outputs, summary


__all__ = [
    "V10_SOURCE_EXPANSION_OBSERVATION_CENSUS_KIND",
    "V10_SOURCE_EXPANSION_OBSERVATION_FILES",
    "derive_normalization_recovery_v10_source_expansion_observations",
]
