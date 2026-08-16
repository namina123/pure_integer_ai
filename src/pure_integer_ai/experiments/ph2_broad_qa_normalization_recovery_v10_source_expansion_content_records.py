"""派生 recovery-v10 新 TRAIN 来源内容与重叠可行性 records。

本模块只消费 v10 名册选中的 Mixxx、Mumble 固定 blob，并用既有 v8
Observation 的 sealed aggregate inventory 计算来源、繁中输入和完整映射重叠。
输出只有 aggregate census，不发布任何新旧 translation surface，也不形成
source pack、TRAIN rule、candidate 或 production caller。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_structured_source_records import (
    derive_normalization_recovery_v8_qt_ts_source_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    V8_OBSERVATION_PACK_MANIFEST_SHA256,
    read_normalization_recovery_v10_source_expansion_roster,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256 = (
    "274852ba4f93b5d1db9fad3ee5268f7fdbbafcf649ba77a84b1ce4cfb322c3be")

_FAMILIES = ("MIXXX_PROJECT", "MUMBLE_PROJECT")
_PREDECESSOR_FILES = (
    "qbittorrent-observations.jsonl",
    "stellarium-observations.jsonl",
    "keepassxc-observations.jsonl",
)
V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES = (
    ("source-content.jsonl", "V10_SOURCE_EXPANSION_CONTENT"),
    ("source-cross-overlap.jsonl", "V10_SOURCE_EXPANSION_CROSS_OVERLAP"),
    ("source-census.jsonl", "V10_SOURCE_EXPANSION_CONTENT_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回 artifact 或 manifest 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _translation(value: object, *, label: str) -> str:
    """从 Qt/gettext locale record 读取唯一 translation 字段。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            f"v10 source content {label} locale record 漂移")
    candidates = [value.get("translation"), value.get("msgstr")]
    strings = [item for item in candidates if isinstance(item, str)]
    if len(strings) != 1:
        raise BroadQaExternalDataError(
            f"v10 source content {label} translation schema 漂移")
    return strings[0]


def _read_candidate_payloads(
        record: dict[str, object],
        source_root: Path,
        ) -> dict[str, bytes]:
    """按名册逐 blob 读取并核对物理 inventory、Git SHA 与许可 SHA。"""
    license_value = record.get("license")
    locale_files = record.get("locale_files")
    if (not source_root.is_dir()
            or not isinstance(license_value, dict)
            or not isinstance(license_value.get("files"), list)
            or not isinstance(locale_files, list)):
        raise BroadQaExternalDataError(
            "v10 source content candidate file inventory 漂移")
    items = tuple(license_value["files"] + locale_files)
    expected_paths = {str(item.get("relative_path")) for item in items
                      if isinstance(item, dict)}
    physical_paths = {
        item.relative_to(source_root).as_posix()
        for item in source_root.rglob("*") if item.is_file()
    }
    if (len(expected_paths) != len(items)
            or physical_paths != expected_paths):
        raise BroadQaExternalDataError(
            "v10 source content candidate physical inventory 漂移")
    payloads = {}
    for item in items:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v10 source content candidate blob 非对象")
        relative = str(item["relative_path"])
        path = (source_root / Path(relative)).resolve()
        try:
            path.relative_to(source_root)
            payload = path.read_bytes()
        except (ValueError, OSError) as error:
            raise BroadQaExternalDataError(
                "v10 source content candidate blob 不可读") from error
        if (len(payload) != item.get("bytes")
                or git_blob_sha1(payload) != item.get("git_blob_sha1")):
            raise BroadQaExternalDataError(
                "v10 source content candidate blob identity 漂移")
        if (relative in {
                str(value.get("relative_path"))
                for value in license_value["files"]
                if isinstance(value, dict)}
                and sha256_hex(payload) != item.get("sha256")):
            raise BroadQaExternalDataError(
                "v10 source content candidate license SHA 漂移")
        payloads[relative] = payload
    return payloads


def _parse_candidate(
        record: dict[str, object],
        payloads: dict[str, bytes],
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """按固定 family spec 调共享 Qt TS parser。"""
    family = str(record.get("source_family"))
    license_value = record.get("license")
    if not isinstance(license_value, dict):
        raise BroadQaExternalDataError("v10 source content license 漂移")
    if family == "MIXXX_PROJECT":
        domain = "mixxx"
        hans = "res/translations/mixxx_zh_CN.ts"
        hant = "res/translations/mixxx_zh_TW.ts"
        source_language = "en"
    elif family == "MUMBLE_PROJECT":
        domain = "mumble"
        hans = "src/mumble/mumble_zh_CN.ts"
        hant = "src/mumble/mumble_zh_TW.ts"
        source_language = ""
    else:
        raise BroadQaExternalDataError(
            "v10 source content candidate family 未支持")
    hans_spec: dict[str, object] = {
        "expected_language": "zh_CN",
        "relative_path": hans,
    }
    hant_spec: dict[str, object] = {
        "expected_language": "zh_TW",
        "relative_path": hant,
    }
    if source_language:
        hans_spec["expected_source_language"] = source_language
        hant_spec["expected_source_language"] = source_language
    return derive_normalization_recovery_v8_qt_ts_source_records(
        source_family=family,
        source_policy_scope=(
            f"{family}_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
        license_expression=str(license_value.get("expression")),
        pair_specs=({
            "domain": domain,
            "zh_Hans": hans_spec,
            "zh_Hant": hant_spec,
        },),
        files={hans: payloads[hans], hant: payloads[hant]},
    )


def _validate_predecessor_manifest(
        observation_root: Path,
        ) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """核验 sealed v8 Observation manifest 与物理文件清单。"""
    path = observation_root / "manifest.json"
    try:
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 source content predecessor manifest 不可读") from error
    if (_sha256(encoded) != V8_OBSERVATION_PACK_MANIFEST_SHA256
            or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded):
        raise BroadQaExternalDataError(
            "v10 source content predecessor manifest identity 漂移")
    files = {
        str(item.get("relative_path")): item
        for item in manifest.get("files", []) if isinstance(item, dict)
    }
    expected_names = {"manifest.json", *files}
    try:
        physical = tuple(observation_root.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 source content predecessor artifact 不可读") from error
    if ({item.name for item in physical} != expected_names
            or any(item.is_dir() for item in physical)
            or not set(_PREDECESSOR_FILES).issubset(files)):
        raise BroadQaExternalDataError(
            "v10 source content predecessor physical inventory 漂移")
    return manifest, files


def _hash_non_observation_file(
        path: Path,
        commitment: dict[str, object],
        ) -> None:
    """流式核验无需解析的 predecessor 文件。"""
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 source content predecessor file 不可读") from error
    if (size != commitment.get("bytes")
            or hasher.hexdigest() != commitment.get("sha256")):
        raise BroadQaExternalDataError(
            "v10 source content predecessor file identity 漂移")


def _read_predecessor_index(observation_root: Path) -> dict[str, object]:
    """严格流读旧三家 Observation 并建立瞬时 exact overlap 索引。"""
    manifest, files = _validate_predecessor_manifest(observation_root)
    for name, commitment in files.items():
        if name not in _PREDECESSOR_FILES:
            _hash_non_observation_file(observation_root / name, commitment)
    source_families: dict[str, set[str]] = defaultdict(set)
    source_input_families: dict[tuple[str, str], set[str]] = defaultdict(set)
    mapping_families: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    family_counts = Counter()
    total = 0
    for name in _PREDECESSOR_FILES:
        commitment = files[name]
        hasher = hashlib.sha256()
        size = 0
        count = 0
        try:
            with (observation_root / name).open("rb") as handle:
                for line in handle:
                    hasher.update(line)
                    size += len(line)
                    value = json.loads(line)
                    if (not isinstance(value, dict)
                            or canonical_json_line(value) != line):
                        raise BroadQaExternalDataError(
                            "v10 source content predecessor JSONL 非规范")
                    family = value.get("source_family")
                    source = value.get("official_source_text")
                    if (not isinstance(family, str) or not family
                            or not isinstance(source, str) or not source):
                        raise BroadQaExternalDataError(
                            "v10 source content predecessor schema 漂移")
                    hant = _translation(value.get("zh_hant"), label="predecessor")
                    hans = _translation(value.get("zh_hans"), label="predecessor")
                    source_families[source].add(family)
                    source_input_families[(source, hant)].add(family)
                    mapping_families[(source, hant, hans)].add(family)
                    family_counts[family] += 1
                    count += 1
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                "v10 source content predecessor Observation 不可读") from error
        if (size != commitment.get("bytes")
                or hasher.hexdigest() != commitment.get("sha256")
                or count != commitment.get("record_count")):
            raise BroadQaExternalDataError(
                "v10 source content predecessor Observation identity 漂移")
        total += count
    if (set(family_counts) != {
            "KEEPASSXC_PROJECT", "QBITTORRENT_PROJECT", "STELLARIUM_PROJECT"}
            or total != manifest.get("summary", {}).get("observation_count")
            or total != 33_179):
        raise BroadQaExternalDataError(
            "v10 source content predecessor denominator 漂移")
    return {
        "family_counts": dict(sorted(family_counts.items())),
        "mapping_families": mapping_families,
        "observation_count": total,
        "source_families": source_families,
        "source_input_families": source_input_families,
    }


def _family_record(
        *,
        family: str,
        roster: dict[str, object],
        pairs: tuple[dict[str, object], ...],
        parser_summary: dict[str, object],
        predecessor: dict[str, object],
        ) -> dict[str, object]:
    """形成单个新 family 的 aggregate 内容与旧 TRAIN 重叠记录。"""
    source_families = predecessor["source_families"]
    source_input_families = predecessor["source_input_families"]
    mapping_families = predecessor["mapping_families"]
    counts = Counter()
    sources = set()
    source_inputs = set()
    mappings = set()
    for pair in pairs:
        source = pair.get("official_source_text")
        if not isinstance(source, str) or not source:
            raise BroadQaExternalDataError(
                "v10 source content new pair source 漂移")
        hant = _translation(pair.get("zh_hant"), label="new")
        hans = _translation(pair.get("zh_hans"), label="new")
        source_input = (source, hant)
        mapping = (source, hant, hans)
        sources.add(source)
        source_inputs.add(source_input)
        mappings.add(mapping)
        counts["changed_pair_count"] += int(hant != hans)
        counts["predecessor_source_overlap_pair_count"] += int(
            source in source_families)
        counts["predecessor_source_input_overlap_pair_count"] += int(
            source_input in source_input_families)
        counts["predecessor_exact_mapping_overlap_pair_count"] += int(
            mapping in mapping_families)
        counts["training_eligible_pair_count"] += int(
            pair.get("v8_training_eligible") == 1)
    overlap_pass = (
        len(pairs) > 0
        and counts["training_eligible_pair_count"] > 0
        and counts["predecessor_source_input_overlap_pair_count"] > 0
    )
    return {
        "changed_pair_count": counts["changed_pair_count"],
        "content_outcome": parser_summary.get("content_outcome"),
        "format_version": 1,
        "license_expression": roster["license"]["expression"],
        "locale_file_read_count": len(roster["locale_files"]),
        "pair_surface_published": 0,
        "parser_summary": parser_summary,
        "predecessor_exact_mapping_overlap_pair_count": counts[
            "predecessor_exact_mapping_overlap_pair_count"],
        "predecessor_source_input_overlap_pair_count": counts[
            "predecessor_source_input_overlap_pair_count"],
        "predecessor_source_overlap_pair_count": counts[
            "predecessor_source_overlap_pair_count"],
        "record_kind": (
            "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CONTENT_RECORD_V1"),
        "selection_outcome": (
            "PASS_CONTENT_AND_PREDECESSOR_SOURCE_INPUT_OVERLAP"
            if overlap_pass else "REJECTED_INSUFFICIENT_CONTENT_OR_OVERLAP"),
        "source_family": family,
        "source_pack_published": 0,
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "transient_pair_count": len(pairs),
        "unique_exact_mapping_count": len(mappings),
        "unique_exact_mapping_overlap_count": sum(
            value in mapping_families for value in mappings),
        "unique_source_count": len(sources),
        "unique_source_input_count": len(source_inputs),
        "unique_source_input_overlap_count": sum(
            value in source_input_families for value in source_inputs),
        "unique_source_overlap_count": sum(
            value in source_families for value in sources),
    }


def _derive(
        parsed: dict[str, dict[str, object]],
        predecessor: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """派生两家 aggregate 内容、彼此重叠和全局 census。"""
    if set(parsed) != set(_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 source content parsed family inventory 漂移")
    records = []
    sets = {}
    for family in _FAMILIES:
        state = parsed[family]
        roster = state.get("roster")
        pairs = state.get("pairs")
        summary = state.get("summary")
        if (not isinstance(roster, dict)
                or not isinstance(pairs, tuple)
                or not isinstance(summary, dict)):
            raise BroadQaExternalDataError(
                "v10 source content parsed state 漂移")
        record = _family_record(
            family=family,
            roster=roster,
            pairs=pairs,
            parser_summary=summary,
            predecessor=predecessor,
        )
        records.append(record)
        sets[family] = {
            "mappings": {
                (str(item["official_source_text"]),
                 _translation(item.get("zh_hant"), label="cross"),
                 _translation(item.get("zh_hans"), label="cross"))
                for item in pairs},
            "source_inputs": {
                (str(item["official_source_text"]),
                 _translation(item.get("zh_hant"), label="cross"))
                for item in pairs},
            "sources": {str(item["official_source_text"]) for item in pairs},
        }
    left = sets[_FAMILIES[0]]
    right = sets[_FAMILIES[1]]
    cross = {
        "exact_mapping_intersection_count": len(
            left["mappings"].intersection(right["mappings"])),
        "format_version": 1,
        "left_source_family": _FAMILIES[0],
        "pair_surface_published": 0,
        "record_kind": (
            "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CROSS_OVERLAP_V1"),
        "right_source_family": _FAMILIES[1],
        "source_input_intersection_count": len(
            left["source_inputs"].intersection(right["source_inputs"])),
        "source_intersection_count": len(
            left["sources"].intersection(right["sources"])),
    }
    counts = Counter()
    for record in records:
        counts["changed_pair_count"] += int(record["changed_pair_count"])
        counts["predecessor_exact_mapping_overlap_pair_count"] += int(
            record["predecessor_exact_mapping_overlap_pair_count"])
        counts["predecessor_source_input_overlap_pair_count"] += int(
            record["predecessor_source_input_overlap_pair_count"])
        counts["predecessor_source_overlap_pair_count"] += int(
            record["predecessor_source_overlap_pair_count"])
        counts["selected_content_pass_count"] += int(
            record["selection_outcome"]
            == "PASS_CONTENT_AND_PREDECESSOR_SOURCE_INPUT_OVERLAP")
        counts["training_eligible_pair_count"] += int(
            record["training_eligible_pair_count"])
        counts["transient_pair_count"] += int(record["transient_pair_count"])
    census = {
        "changed_pair_count": counts["changed_pair_count"],
        "cross_exact_mapping_intersection_count": cross[
            "exact_mapping_intersection_count"],
        "cross_source_input_intersection_count": cross[
            "source_input_intersection_count"],
        "cross_source_intersection_count": cross[
            "source_intersection_count"],
        "formal_reserved_locale_read_count": 0,
        "pair_surface_published": 0,
        "predecessor_exact_mapping_overlap_pair_count": counts[
            "predecessor_exact_mapping_overlap_pair_count"],
        "predecessor_observation_count": predecessor["observation_count"],
        "predecessor_source_input_overlap_pair_count": counts[
            "predecessor_source_input_overlap_pair_count"],
        "predecessor_source_overlap_pair_count": counts[
            "predecessor_source_overlap_pair_count"],
        "selected_content_pass_count": counts["selected_content_pass_count"],
        "selected_source_family_count": len(records),
        "source_pack_published_count": 0,
        "training_eligible_pair_count": counts[
            "training_eligible_pair_count"],
        "transient_pair_count": counts["transient_pair_count"],
    }
    if (census["selected_content_pass_count"] != 2
            or census["cross_source_input_intersection_count"] <= 0
            or census["predecessor_source_input_overlap_pair_count"] <= 0):
        raise BroadQaExternalDataError(
            "v10 source content selection hard gate 未通过")
    census_record = {
        **census,
        "format_version": 1,
        "record_kind": (
            "NORMALIZATION_RECOVERY_V10_SOURCE_EXPANSION_CONTENT_CENSUS_V1"),
    }
    records.sort(key=lambda item: str(item["source_family"]))
    return {
        V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES[0][0]: tuple(records),
        V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES[1][0]: (cross,),
        V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES[2][0]: (census_record,),
    }, census


def _state(
        *,
        roster_dir: Path,
        predecessor_observation_dir: Path,
        source_roots: dict[str, Path],
        ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """严格读取名册、两家 blob 与旧 Observation，并返回瞬时状态。"""
    _manifest, roster_outputs = (
        read_normalization_recovery_v10_source_expansion_roster(
            roster_dir,
            expected_manifest_sha256=(
                V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256),
        ))
    candidates = {
        str(item["source_family"]): item
        for item in roster_outputs["source-candidates.jsonl"]
        if item.get("selection_status")
        == "SELECTED_TRAIN_CONTENT_FEASIBILITY_PENDING"
    }
    if set(candidates) != set(_FAMILIES) or set(source_roots) != set(_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 source content selected family inventory 漂移")
    parsed = {}
    for family in _FAMILIES:
        payloads = _read_candidate_payloads(
            candidates[family], source_roots[family])
        _files, pairs, summary = _parse_candidate(
            candidates[family], payloads)
        parsed[family] = {
            "pairs": pairs,
            "roster": candidates[family],
            "summary": summary,
        }
    predecessor = _read_predecessor_index(predecessor_observation_dir)
    return parsed, predecessor


def derive_normalization_recovery_v10_source_expansion_content(
        parsed: dict[str, dict[str, object]],
        predecessor: dict[str, object],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """公开纯派生入口，供 publisher 与专项共享。"""
    return _derive(parsed, predecessor)


def read_normalization_recovery_v10_source_expansion_content_state(
        *,
        roster_dir: Path,
        predecessor_observation_dir: Path,
        source_roots: dict[str, Path],
        ) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """公开严格输入入口，publisher 不持有 parser 与重叠算法。"""
    return _state(
        roster_dir=roster_dir,
        predecessor_observation_dir=predecessor_observation_dir,
        source_roots=source_roots,
    )


__all__ = [
    "V10_SOURCE_EXPANSION_CONTENT_OUTPUT_FILES",
    "V10_SOURCE_EXPANSION_ROSTER_MANIFEST_SHA256",
    "derive_normalization_recovery_v10_source_expansion_content",
    "read_normalization_recovery_v10_source_expansion_content_state",
]
