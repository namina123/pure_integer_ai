"""发布 recovery-v8 三家 TRAIN source 的aggregate overlap/copy audit。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack import (
    read_normalization_recovery_v8_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_AUDIT_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_RECORD_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_PAIRWISE_RECORD_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_CENSUS_KIND = (
    "NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_CENSUS_V1")
NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_PASS_STATUS = (
    "THREE_INDEPENDENT_FAMILIES_NO_LOCALE_OR_SUBSET_COPY")
NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_REJECTED_STATUS = (
    "SOURCE_INDEPENDENCE_OR_COPY_GATE_REJECTED")

QBITTORRENT_SOURCE_PACK_MANIFEST_SHA256 = (
    "0a0d29bbcbb6d3470a458f1762f5be82963a1173fb5c66795e4637b29a1dad36")
STELLARIUM_SOURCE_PACK_MANIFEST_SHA256 = (
    "459adcadb7000c232ee7e8004a03aca9ba31971a3690a529f9c51bcc66917212")
KEEPASSXC_SOURCE_PACK_MANIFEST_SHA256 = (
    "ad1223f64bd09706a48cd1855000a8c6437873696097be56e1c26af45753ade3")

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
_OUTPUT_FILES = (
    ("source-overlap.jsonl", "V8_SOURCE_OVERLAP_PAIRWISE"),
    ("source-overlap-census.jsonl", "V8_SOURCE_OVERLAP_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v8 source overlap run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 source overlap {label} 越出run root") from error
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


def _translation(pair: dict[str, object], role: str) -> str:
    """从Qt/gettext统一pair schema提取指定locale表面。"""
    value = pair.get(role)
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "v8 source overlap locale record 漂移")
    candidates = [value.get("translation"), value.get("msgstr")]
    strings = [item for item in candidates if isinstance(item, str)]
    if len(strings) != 1:
        raise BroadQaExternalDataError(
            "v8 source overlap translation schema 漂移")
    return strings[0]


def _pair_indexes(
        pairs: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """建立仅用于精确overlap统计的family presence集合。"""
    source_texts = set()
    inputs = set()
    outputs = set()
    input_output = set()
    full_pairs = set()
    source_input_outputs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in pairs:
        source = pair.get("official_source_text")
        hans_tokens = pair.get("zh_hans_structure_tokens")
        hant_tokens = pair.get("zh_hant_structure_tokens")
        if (not isinstance(source, str)
                or not isinstance(hans_tokens, list)
                or not isinstance(hant_tokens, list)
                or any(not isinstance(item, str)
                       for item in hans_tokens + hant_tokens)):
            raise BroadQaExternalDataError(
                "v8 source overlap pair schema 漂移")
        input_text = _translation(pair, "zh_hant")
        output_text = _translation(pair, "zh_hans")
        source_texts.add(source)
        inputs.add(input_text)
        outputs.add(output_text)
        input_output.add((input_text, output_text))
        full_pairs.add((
            source,
            input_text,
            output_text,
            tuple(hant_tokens),
            tuple(hans_tokens),
        ))
        source_input_outputs[(source, input_text)].add(output_text)
    return {
        "full_pairs": full_pairs,
        "input_output": input_output,
        "inputs": inputs,
        "outputs": outputs,
        "source_input_outputs": source_input_outputs,
        "source_texts": source_texts,
    }


def _raw_blob_sets(manifest: dict[str, object]) -> dict[str, set[str]]:
    """从source-pack manifest分离license与locale raw blob身份。"""
    values = {"license": set(), "locale": set()}
    for item in manifest.get("files", []):
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v8 source overlap source-pack file record 漂移")
        role = item.get("role")
        sha = item.get("sha256")
        if role == "V8_SOURCE_RAW_LICENSE_BLOB":
            label = "license"
        elif role == "V8_SOURCE_RAW_LOCALE_BLOB":
            label = "locale"
        else:
            continue
        if not isinstance(sha, str) or len(sha) != 64:
            raise BroadQaExternalDataError(
                "v8 source overlap raw blob identity 漂移")
        values[label].add(sha)
    if not values["license"] or not values["locale"]:
        raise BroadQaExternalDataError(
            "v8 source overlap raw blob inventory 漂移")
    return values


def _pairwise_record(
        left_family: str,
        right_family: str,
        *,
        manifests: dict[str, dict[str, object]],
        indexes: dict[str, dict[str, object]],
        blobs: dict[str, dict[str, set[str]]],
        pair_counts: dict[str, int],
        ) -> dict[str, object]:
    """形成两个family之间的精确presence overlap与复制门。"""
    left = indexes[left_family]
    right = indexes[right_family]
    full_overlap = len(left["full_pairs"].intersection(right["full_pairs"]))
    smaller = min(pair_counts[left_family], pair_counts[right_family])
    shared_source_input = set(left["source_input_outputs"]).intersection(
        right["source_input_outputs"])
    conflicts = sum(
        left["source_input_outputs"][key]
        != right["source_input_outputs"][key]
        for key in shared_source_input)
    left_raw = manifests[left_family]["raw_source"]
    right_raw = manifests[right_family]["raw_source"]
    return {
        "commit_equal": int(left_raw["commit"] == right_raw["commit"]),
        "exact_full_subset_copy": int(full_overlap == smaller),
        "format_version": 1,
        "full_pair_semantic_overlap_count": full_overlap,
        "full_pair_semantic_overlap_denominator": smaller,
        "input_output_mapping_overlap_count": len(
            left["input_output"].intersection(right["input_output"])),
        "left_family": left_family,
        "left_pair_count": pair_counts[left_family],
        "license_blob_overlap_count": len(
            blobs[left_family]["license"].intersection(
                blobs[right_family]["license"])),
        "locale_blob_overlap_count": len(
            blobs[left_family]["locale"].intersection(
                blobs[right_family]["locale"])),
        "official_source_text_overlap_count": len(
            left["source_texts"].intersection(right["source_texts"])),
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_RECORD_KIND,
        "repository_equal": int(
            left_raw["repository"] == right_raw["repository"]),
        "right_family": right_family,
        "right_pair_count": pair_counts[right_family],
        "root_tree_equal": int(
            left_raw["root_tree"] == right_raw["root_tree"]),
        "source_input_conflicting_output_count": conflicts,
        "source_input_key_overlap_count": len(shared_source_input),
        "zh_hans_output_overlap_count": len(
            left["outputs"].intersection(right["outputs"])),
        "zh_hant_input_overlap_count": len(
            left["inputs"].intersection(right["inputs"])),
    }


def _presence_count(
        indexes: dict[str, dict[str, object]],
        key: str,
        ) -> int:
    """计算至少出现在两个独立family中的unique identity数量。"""
    presence = defaultdict(int)
    for family in _FAMILIES:
        for value in indexes[family][key]:
            presence[value] += 1
    return sum(count >= 2 for count in presence.values())


def _derive(
        manifests: dict[str, dict[str, object]],
        pairs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """重算三家pair/raw/lineage overlap并执行独立性硬门。"""
    if set(manifests) != set(_FAMILIES) or set(pairs) != set(_FAMILIES):
        raise BroadQaExternalDataError(
            "v8 source overlap family inventory 漂移")
    for family in _FAMILIES:
        if (manifests[family].get("source_family") != family
                or manifests[family].get("source_family_vote_count") != 1
                or not pairs[family]):
            raise BroadQaExternalDataError(
                "v8 source overlap source-pack fields 漂移")
    indexes = {family: _pair_indexes(pairs[family]) for family in _FAMILIES}
    blobs = {family: _raw_blob_sets(manifests[family])
             for family in _FAMILIES}
    pair_counts = {family: len(pairs[family]) for family in _FAMILIES}
    records = []
    for index, left in enumerate(_FAMILIES):
        for right in _FAMILIES[index + 1:]:
            records.append(_pairwise_record(
                left,
                right,
                manifests=manifests,
                indexes=indexes,
                blobs=blobs,
                pair_counts=pair_counts,
            ))
    records.sort(key=lambda item: (item["left_family"], item["right_family"]))
    independence_fail_count = sum(
        int(item[key])
        for item in records
        for key in (
            "commit_equal",
            "exact_full_subset_copy",
            "locale_blob_overlap_count",
            "repository_equal",
            "root_tree_equal",
        ))
    summary = {
        "cross_family_vote_count": 3,
        "exact_full_subset_copy_pair_count": sum(
            int(item["exact_full_subset_copy"]) for item in records),
        "family_count": 3,
        "full_pair_semantic_shared_identity_count": _presence_count(
            indexes, "full_pairs"),
        "hard_independence_failure_count": independence_fail_count,
        "input_output_shared_identity_count": _presence_count(
            indexes, "input_output"),
        "license_blob_pairwise_overlap_count": sum(
            int(item["license_blob_overlap_count"]) for item in records),
        "locale_blob_pairwise_overlap_count": sum(
            int(item["locale_blob_overlap_count"]) for item in records),
        "official_source_shared_identity_count": _presence_count(
            indexes, "source_texts"),
        "pair_surface_published": 0,
        "pairwise_record_count": len(records),
        "source_input_conflicting_output_pairwise_count": sum(
            int(item["source_input_conflicting_output_count"])
            for item in records),
        "total_pair_record_count": sum(pair_counts.values()),
    }
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_CENSUS_KIND,
    },)
    return {
        _OUTPUT_FILES[0][0]: tuple(records),
        _OUTPUT_FILES[1][0]: census,
    }, summary


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
                        f"v8 source overlap {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v8 source overlap {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成aggregate输出文件commitment。"""
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
    """构造不含pair surface的cross-family audit manifest。"""
    status = (
        NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_PASS_STATUS
        if summary["hard_independence_failure_count"] == 0
        else NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_REJECTED_STATUS)
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_AUDIT_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {
            "keepassxc_source_pack_manifest_sha256": (
                KEEPASSXC_SOURCE_PACK_MANIFEST_SHA256),
            "qbittorrent_source_pack_manifest_sha256": (
                QBITTORRENT_SOURCE_PACK_MANIFEST_SHA256),
            "stellarium_source_pack_manifest_sha256": (
                STELLARIUM_SOURCE_PACK_MANIFEST_SHA256),
        },
        "mastery_claimed": 0,
        "observation_pack_published": 0,
        "production_enabled": 0,
        "status": status,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _state(
        *,
        source_pack_dirs: dict[str, Path],
        v2_roster_dir: Path,
        v1_roster_dir: Path,
        v1_content_audit_dir: Path,
        v2_content_audit_dir: Path,
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """严格回读三份source pack并返回manifest/pair records。"""
    if set(source_pack_dirs) != set(_FAMILIES):
        raise BroadQaExternalDataError(
            "v8 source overlap source-pack path inventory 漂移")
    manifests = {}
    pairs = {}
    for family in _FAMILIES:
        manifest, _files, pair_records, _census = (
            read_normalization_recovery_v8_source_pack(
                source_pack_dirs[family],
                v2_roster_dir=v2_roster_dir,
                v1_roster_dir=v1_roster_dir,
                v1_content_audit_dir=v1_content_audit_dir,
                v2_content_audit_dir=v2_content_audit_dir,
                expected_manifest_sha256=_PACK_SHA[family],
            ))
        manifests[family] = manifest
        pairs[family] = pair_records
    return manifests, pairs


def publish_normalization_recovery_v8_source_overlap_audit(
        *,
        run_root: str | Path,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布三家source的aggregate overlap/copy audit。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      v2_roster_dir,
                      v1_roster_dir,
                      v1_content_audit_dir,
                      v2_content_audit_dir,
                      qbittorrent_source_pack_dir,
                      stellarium_source_pack_dir,
                      keepassxc_source_pack_dir,
                      target_dir,
                  )))
    (v2_roster, v1_roster, v1_content, v2_content,
     qbit, stellarium, keepassxc, target) = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v8 source overlap input/target path 非法")
    source_pack_dirs = {
        "QBITTORRENT_PROJECT": qbit,
        "STELLARIUM_PROJECT": stellarium,
        "KEEPASSXC_PROJECT": keepassxc,
    }
    state = _state(
        source_pack_dirs=source_pack_dirs,
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


def read_normalization_recovery_v8_source_overlap_audit(
        audit_dir: str | Path,
        *,
        v2_roster_dir: str | Path,
        v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path,
        v2_content_audit_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生overlap aggregate，并拒绝records/manifest同步篡改。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v8 source overlap manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError(
            "v8 source overlap manifest identity 漂移")
    state = _state(
        source_pack_dirs={
            "QBITTORRENT_PROJECT": Path(
                qbittorrent_source_pack_dir).resolve(),
            "STELLARIUM_PROJECT": Path(
                stellarium_source_pack_dir).resolve(),
            "KEEPASSXC_PROJECT": Path(
                keepassxc_source_pack_dir).resolve(),
        },
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
            "v8 source overlap records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v8 source overlap fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_AUDIT_KIND",
    "NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_PASS_STATUS",
    "NORMALIZATION_RECOVERY_V8_SOURCE_OVERLAP_REJECTED_STATUS",
    "publish_normalization_recovery_v8_source_overlap_audit",
    "read_normalization_recovery_v8_source_overlap_audit",
]
