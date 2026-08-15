"""发布 recovery-v8 Observation 的TRAIN前coverage/collision候选审计。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import BroadQaExternalDataError
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_pack import (
    read_normalization_recovery_v8_observation_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_COVERAGE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_OBSERVATION_COVERAGE_V1")
NORMALIZATION_RECOVERY_V8_COVERAGE_STATUS = (
    "CANDIDATES_FROZEN_NOT_AUTHORIZED_NOT_TRAINED")
V8_OBSERVATION_PACK_MANIFEST_SHA256 = (
    "99ab49c0605be76b2206746330969a071d8b6deed83f3aa454610a99546ddf65")

_FAMILY_FILES = (
    "qbittorrent-observations.jsonl",
    "stellarium-observations.jsonl",
    "keepassxc-observations.jsonl",
)
_OUTPUT_FILES = (
    ("exact-input-mappings.jsonl", "V8_EXACT_INPUT_MAPPING_CANDIDATES"),
    ("source-conditioned-mappings.jsonl", "V8_SOURCE_CONDITIONED_MAPPING_CANDIDATES"),
    ("orthographic-atoms.jsonl", "V8_ORTHOGRAPHIC_ATOM_CANDIDATES"),
    ("structure-obligations.jsonl", "V8_STRUCTURE_OBLIGATION_CANDIDATES"),
    ("coverage-census.jsonl", "V8_OBSERVATION_COVERAGE_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact、candidate或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError("v8 coverage run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v8 coverage {label} 越出run root") from error
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


def _surface(observation: dict[str, object], role: str) -> str:
    """从Qt/gettext统一Observation提取locale表面。"""
    value = observation.get(role)
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("v8 coverage locale record 漂移")
    strings = [item for item in (
        value.get("translation"), value.get("msgstr"))
        if isinstance(item, str)]
    if len(strings) != 1:
        raise BroadQaExternalDataError("v8 coverage surface schema 漂移")
    return strings[0]


def _eligible(observation: dict[str, object]) -> bool:
    """严格读取Observation冻结的v8 eligibility。"""
    eligibility = observation.get("eligibility")
    if not isinstance(eligibility, dict):
        raise BroadQaExternalDataError("v8 coverage eligibility 漂移")
    features = eligibility.get("pair_features")
    if (not isinstance(features, dict)
            or features.get("v8_training_eligible") not in (0, 1)):
        raise BroadQaExternalDataError("v8 coverage eligibility facts 漂移")
    return features["v8_training_eligible"] == 1


def _status(outputs: dict[str, dict[str, set[str]]]) -> str:
    """按family presence与output collision形成候选状态。"""
    families = set().union(*(
        values["families"] for values in outputs.values()))
    if len(families) >= 2:
        return ("MULTI_FAMILY_UNIQUE_OUTPUT" if len(outputs) == 1
                else "MULTI_FAMILY_CONFLICT")
    return ("SINGLE_FAMILY_UNIQUE_OUTPUT" if len(outputs) == 1
            else "SINGLE_FAMILY_INTERNAL_CONFLICT")


def _mapping_candidates(
        observations: tuple[dict[str, object], ...],
        *,
        conditioned: bool,
        ) -> tuple[dict[str, object], ...]:
    """按input或official-source+input聚合完整output/family presence。"""
    groups: dict[tuple[str, ...], dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "families": set(), "family_counts": Counter(),
            "identity_count": 0, "nonidentity_count": 0,
        }))
    for item in observations:
        if not _eligible(item):
            continue
        family = str(item["source_family"])
        source = str(item["official_source_text"])
        input_text = _surface(item, "zh_hant")
        output_text = _surface(item, "zh_hans")
        key = (source, input_text) if conditioned else (input_text,)
        bucket = groups[key][output_text]
        bucket["families"].add(family)
        bucket["family_counts"][family] += 1
        identity = int(input_text == output_text)
        bucket["identity_count"] += identity
        bucket["nonidentity_count"] += 1 - identity
    kind = ("SOURCE_CONDITIONED_MAPPING" if conditioned
            else "EXACT_INPUT_MAPPING")
    records = []
    for key in sorted(groups):
        outputs = groups[key]
        identity = {"candidate_kind": kind, "key": list(key)}
        records.append({
            "candidate_id": _sha256(canonical_json_line(identity)),
            "candidate_kind": kind,
            "candidate_status": _status(outputs),
            "format_version": 1,
            "input_text": key[-1],
            "official_source_text": key[0] if conditioned else "",
            "outputs": [{
                "family_record_counts": dict(sorted(value["family_counts"].items())),
                "identity_record_count": value["identity_count"],
                "nonidentity_record_count": value["nonidentity_count"],
                "output_text": output,
                "support_families": sorted(value["families"]),
                "support_family_count": len(value["families"]),
            } for output, value in sorted(outputs.items())],
            "record_kind": "NORMALIZATION_RECOVERY_V8_MAPPING_CANDIDATE_V1",
            "support_families": sorted(set().union(*(
                value["families"] for value in outputs.values()))),
        })
    return tuple(records)


def _orthographic_candidates(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """从eligible single-Han pairs聚合一字符映射及collision。"""
    groups: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "families": set(), "family_counts": Counter(),
        }))
    for item in observations:
        features = item["eligibility"]["pair_features"]
        if not _eligible(item) or features["single_han_difference"] != 1:
            continue
        left = _surface(item, "zh_hant")
        right = _surface(item, "zh_hans")
        differences = [(a, b) for a, b in zip(left, right) if a != b]
        if len(left) != len(right) or len(differences) != 1:
            raise BroadQaExternalDataError("v8 coverage single-Han facts 漂移")
        source, target = differences[0]
        bucket = groups[source][target]
        family = str(item["source_family"])
        bucket["families"].add(family)
        bucket["family_counts"][family] += 1
    records = []
    for source in sorted(groups):
        outputs = groups[source]
        identity = {"candidate_kind": "ORTHOGRAPHIC_ATOM", "input": source}
        records.append({
            "candidate_id": _sha256(canonical_json_line(identity)),
            "candidate_kind": "ORTHOGRAPHIC_ATOM",
            "candidate_status": _status(outputs),
            "format_version": 1,
            "input_atom": source,
            "outputs": [{
                "family_record_counts": dict(sorted(value["family_counts"].items())),
                "output_atom": output,
                "support_families": sorted(value["families"]),
                "support_family_count": len(value["families"]),
            } for output, value in sorted(outputs.items())],
            "record_kind": "NORMALIZATION_RECOVERY_V8_ORTHOGRAPHIC_ATOM_CANDIDATE_V1",
            "support_families": sorted(set().union(*(
                value["families"] for value in outputs.values()))),
        })
    return tuple(records)


def _structure_candidates(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按非空规范token序聚合structure obligation family presence。"""
    groups: dict[tuple[str, ...], dict[str, object]] = defaultdict(
        lambda: {"families": set(), "family_counts": Counter()})
    for item in observations:
        if not _eligible(item):
            continue
        tokens = item.get("zh_hant_structure_tokens")
        if not isinstance(tokens, list) or any(not isinstance(v, str) for v in tokens):
            raise BroadQaExternalDataError("v8 coverage structure tokens 漂移")
        if not tokens:
            continue
        key = tuple(tokens)
        family = str(item["source_family"])
        groups[key]["families"].add(family)
        groups[key]["family_counts"][family] += 1
    return tuple({
        "candidate_id": _sha256(canonical_json_line({
            "candidate_kind": "STRUCTURE_OBLIGATION", "tokens": list(key)})),
        "candidate_kind": "STRUCTURE_OBLIGATION",
        "candidate_status": ("MULTI_FAMILY_OBSERVED" if len(value["families"]) >= 2
                             else "SINGLE_FAMILY_OBSERVED"),
        "family_record_counts": dict(sorted(value["family_counts"].items())),
        "format_version": 1,
        "record_kind": "NORMALIZATION_RECOVERY_V8_STRUCTURE_OBLIGATION_CANDIDATE_V1",
        "structure_tokens": list(key),
        "support_families": sorted(value["families"]),
        "support_family_count": len(value["families"]),
    } for key, value in sorted(groups.items()))


def _candidate_counts(records: tuple[dict[str, object], ...]) -> dict[str, int]:
    """汇总候选状态，不把candidate existence解释为授权。"""
    counts = Counter(str(item["candidate_status"]) for item in records)
    return {key: counts[key] for key in sorted(counts)}


def _derive(outputs: dict[str, tuple[dict[str, object], ...]]) -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """从全量Observation形成四类候选与collision census。"""
    if any(name not in outputs for name in _FAMILY_FILES):
        raise BroadQaExternalDataError("v8 coverage observation inventory 漂移")
    observations = tuple(item for name in _FAMILY_FILES for item in outputs[name])
    exact = _mapping_candidates(observations, conditioned=False)
    source = _mapping_candidates(observations, conditioned=True)
    atoms = _orthographic_candidates(observations)
    structures = _structure_candidates(observations)
    eligible_count = sum(_eligible(item) for item in observations)
    summary = {
        "authorization_count": 0,
        "eligible_observation_count": eligible_count,
        "exact_input_candidate_count": len(exact),
        "exact_input_status_counts": _candidate_counts(exact),
        "observation_count": len(observations),
        "orthographic_atom_candidate_count": len(atoms),
        "orthographic_atom_status_counts": _candidate_counts(atoms),
        "source_conditioned_candidate_count": len(source),
        "source_conditioned_status_counts": _candidate_counts(source),
        "structure_obligation_candidate_count": len(structures),
        "structure_obligation_status_counts": _candidate_counts(structures),
        "train_protocol_published": 0,
    }
    census = ({**summary, "format_version": 1,
               "record_kind": "NORMALIZATION_RECOVERY_V8_COVERAGE_CENSUS_V1"},)
    return {
        _OUTPUT_FILES[0][0]: exact,
        _OUTPUT_FILES[1][0]: source,
        _OUTPUT_FILES[2][0]: atoms,
        _OUTPUT_FILES[3][0]: structures,
        _OUTPUT_FILES[4][0]: census,
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
                if not isinstance(value, dict) or canonical_json_line(value) != line:
                    raise BroadQaExternalDataError(f"v8 coverage {label} JSONL 非规范")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"v8 coverage {label} 不可读") from error
    return tuple(values)


def _artifact(path: Path, *, role: str, count: int) -> dict[str, object]:
    """形成coverage输出文件commitment。"""
    payload = path.read_bytes()
    return {"bytes": len(payload), "record_count": count,
            "relative_path": path.name, "role": role, "sha256": _sha256(payload)}


def _manifest(files: list[dict[str, object]], summary: dict[str, object]) -> dict[str, object]:
    """构造候选已冻结、未授权未训练的manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_COVERAGE_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {"observation_pack_manifest_sha256": V8_OBSERVATION_PACK_MANIFEST_SHA256},
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V8_COVERAGE_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _state(observation_dir: Path, **paths: Path) -> dict[str, tuple[dict[str, object], ...]]:
    """严格回读Observation pack及其全部sealed predecessors。"""
    _manifest_value, outputs = read_normalization_recovery_v8_observation_pack(
        observation_dir, expected_manifest_sha256=V8_OBSERVATION_PACK_MANIFEST_SHA256,
        **paths)
    return outputs


def publish_normalization_recovery_v8_observation_coverage(
        *, run_root: str | Path, observation_dir: str | Path,
        v2_roster_dir: str | Path, v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path, v2_content_audit_dir: str | Path,
        source_overlap_dir: str | Path, qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path, keepassxc_source_pack_dir: str | Path,
        target_dir: str | Path) -> dict[str, object]:
    """不可覆盖发布Observation coverage/collision候选审计。"""
    root = _require_k_root(run_root)
    values = (observation_dir, v2_roster_dir, v1_roster_dir, v1_content_audit_dir,
              v2_content_audit_dir, source_overlap_dir, qbittorrent_source_pack_dir,
              stellarium_source_pack_dir, keepassxc_source_pack_dir, target_dir)
    paths = tuple(_within(root, value, label=str(i)) for i, value in enumerate(values))
    observation, v2_roster, v1_roster, v1_content, v2_content, overlap, qbit, stell, keep, target = paths
    if (target.exists() or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError("v8 coverage input/target path 非法")
    state = _state(observation, v2_roster_dir=v2_roster, v1_roster_dir=v1_roster,
                   v1_content_audit_dir=v1_content, v2_content_audit_dir=v2_content,
                   source_overlap_dir=overlap, qbittorrent_source_pack_dir=qbit,
                   stellarium_source_pack_dir=stell, keepassxc_source_pack_dir=keep)
    outputs, summary = _derive(state)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files, summary)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v8_observation_coverage(
        coverage_dir: str | Path, *, observation_dir: str | Path,
        v2_roster_dir: str | Path, v1_roster_dir: str | Path,
        v1_content_audit_dir: str | Path, v2_content_audit_dir: str | Path,
        source_overlap_dir: str | Path, qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path, keepassxc_source_pack_dir: str | Path,
        expected_manifest_sha256: str) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生coverage候选并拒绝records/manifest同步篡改。"""
    root = Path(coverage_dir).resolve()
    try:
        encoded = (root / "manifest.json").read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("v8 coverage manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256 or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded):
        raise BroadQaExternalDataError("v8 coverage manifest identity 漂移")
    state = _state(Path(observation_dir).resolve(),
        v2_roster_dir=Path(v2_roster_dir).resolve(), v1_roster_dir=Path(v1_roster_dir).resolve(),
        v1_content_audit_dir=Path(v1_content_audit_dir).resolve(),
        v2_content_audit_dir=Path(v2_content_audit_dir).resolve(),
        source_overlap_dir=Path(source_overlap_dir).resolve(),
        qbittorrent_source_pack_dir=Path(qbittorrent_source_pack_dir).resolve(),
        stellarium_source_pack_dir=Path(stellarium_source_pack_dir).resolve(),
        keepassxc_source_pack_dir=Path(keepassxc_source_pack_dir).resolve())
    expected_outputs, summary = _derive(state)
    stored_outputs = {name: _read_jsonl(root / name, label=role)
                      for name, role in _OUTPUT_FILES}
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError("v8 coverage records 漂移")
    files = [_artifact(root / name, role=role, count=len(expected_outputs[name]))
             for name, role in _OUTPUT_FILES]
    if stored != _manifest(files, summary):
        raise BroadQaExternalDataError("v8 coverage fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = ["NORMALIZATION_RECOVERY_V8_COVERAGE_KIND",
           "NORMALIZATION_RECOVERY_V8_COVERAGE_STATUS",
           "V8_OBSERVATION_PACK_MANIFEST_SHA256",
           "publish_normalization_recovery_v8_observation_coverage",
           "read_normalization_recovery_v8_observation_coverage"]
