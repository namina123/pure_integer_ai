"""发布并严格回读 recovery-v10 五 family Observation pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit import (
    read_normalization_recovery_v10_five_family_audit_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_records import (
    V10_SOURCE_EXPANSION_OBSERVATION_FILES,
    derive_normalization_recovery_v10_source_expansion_observations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    V8_OBSERVATION_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_source_pack import (
    read_normalization_recovery_v10_source_expansion_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_SOURCE_EXPANSION_OBSERVATION_PACK_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_OBSERVATION_PACK_V1")
V10_SOURCE_EXPANSION_OBSERVATION_PACK_STATUS = (
    "FIVE_FAMILY_OBSERVATIONS_FROZEN_NOT_TRAINED")
V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256 = (
    "b4abb7edec79408c0e643bdcb6058e704f6e951054aee8eec93566110d31a671")
_NEW_PACK_SHA = {
    "MIXXX_PROJECT": (
        "fd9eed181a2c551966e1792777186de10051458d9ce21a187b80c82055715d9c"),
    "MUMBLE_PROJECT": (
        "c336cb36a9eeeda51282130ab20f75136f9cddabb7f5ef60b57dbc8532c03681"),
}
_PREDECESSOR_OUTPUT_FILES = (
    ("source-files.jsonl", "V8_OBSERVATION_SOURCE_FILES"),
    *((name, "V8_FAMILY_OBSERVATIONS")
      for family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES
      if family not in _NEW_PACK_SHA),
    ("family-census.jsonl", "V8_OBSERVATION_FAMILY_CENSUS"),
    ("observation-census.jsonl", "V8_OBSERVATION_CENSUS"),
)
_OUTPUT_FILES = (
    ("source-files.jsonl", "V10_EXPANDED_OBSERVATION_SOURCE_FILES"),
    *((name, "V10_EXPANDED_FAMILY_OBSERVATIONS")
      for _family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES),
    ("family-census.jsonl", "V10_EXPANDED_OBSERVATION_FAMILY_CENSUS"),
    ("observation-census.jsonl", "V10_EXPANDED_OBSERVATION_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 expanded observation run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v10 expanded observation {label} 越出run root") from error
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


def _read_jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    """严格读取规范JSONL，并允许显式空ledger。"""
    values = []
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BroadQaExternalDataError(
                    f"v10 expanded observation {label} record 非对象")
            values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 expanded observation {label} 不可读") from error
    if (b"".join(lines) != payload
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(
            f"v10 expanded observation {label} JSONL 非规范")
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


def _read_predecessor_observation_pack(
        observation_dir: Path,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """按fixed SHA严格回读旧三家Observation aggregate。"""
    path = observation_dir / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 expanded predecessor Observation manifest 不可读") from error
    if (_sha256(encoded) != V8_OBSERVATION_PACK_MANIFEST_SHA256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_V1"):
        raise BroadQaExternalDataError(
            "v10 expanded predecessor Observation identity 漂移")
    files = {
        str(item.get("relative_path")): item
        for item in stored.get("files", []) if isinstance(item, dict)
    }
    expected_names = {name for name, _role in _PREDECESSOR_OUTPUT_FILES}
    try:
        physical = tuple(observation_dir.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 expanded predecessor Observation inventory 不可读") from error
    if ({item.name for item in physical} != {"manifest.json", *expected_names}
            or any(item.is_dir() for item in physical)
            or set(files) != expected_names):
        raise BroadQaExternalDataError(
            "v10 expanded predecessor Observation inventory 漂移")
    outputs = {}
    for name, role in _PREDECESSOR_OUTPUT_FILES:
        values = _read_jsonl(observation_dir / name, label=role)
        if files[name] != _artifact(
                observation_dir / name, role=role, count=len(values)):
            raise BroadQaExternalDataError(
                "v10 expanded predecessor Observation file 漂移")
        outputs[name] = values
    if (len(outputs["family-census.jsonl"]) != 3
            or len(outputs["observation-census.jsonl"]) != 1
            or stored.get("summary", {}).get("observation_count") != 33_179):
        raise BroadQaExternalDataError(
            "v10 expanded predecessor Observation census 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


def _state(
        *,
        observation_dir: Path,
        audit_dir: Path,
        roster_dir: Path,
        content_dir: Path,
        new_source_pack_dirs: dict[str, Path],
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, dict[str, object]],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """读取旧Observation、五family audit并严格重派生新两家source pack。"""
    if set(new_source_pack_dirs) != set(_NEW_PACK_SHA):
        raise BroadQaExternalDataError(
            "v10 expanded observation new source path inventory 漂移")
    predecessor_manifest, predecessor_outputs = (
        _read_predecessor_observation_pack(observation_dir))
    audit_manifest, audit_outputs = (
        read_normalization_recovery_v10_five_family_audit_aggregate(
            audit_dir,
            expected_manifest_sha256=V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256,
        ))
    manifests = {}
    source_files = {}
    pairs = {}
    for family in _NEW_PACK_SHA:
        manifest, file_records, pair_records, _census = (
            read_normalization_recovery_v10_source_expansion_source_pack(
                new_source_pack_dirs[family],
                roster_dir=roster_dir,
                content_dir=content_dir,
                expected_manifest_sha256=_NEW_PACK_SHA[family],
            ))
        manifests[family] = manifest
        source_files[family] = file_records
        pairs[family] = pair_records
    return (
        predecessor_manifest,
        predecessor_outputs,
        manifests,
        source_files,
        pairs,
        audit_manifest,
        audit_outputs,
    )


def _outputs(
        state: tuple[object, ...],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """从state派生五family Observation outputs与summary。"""
    (_predecessor_manifest, predecessor_outputs, manifests,
     source_files, pairs, audit_manifest, _audit_outputs) = state
    return derive_normalization_recovery_v10_source_expansion_observations(
        predecessor_outputs,
        manifests,
        source_files,
        pairs,
        audit_manifest,
    )


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        state: tuple[object, ...],
        ) -> dict[str, object]:
    """构造旧identity保留、新两家未训练的Observation manifest。"""
    predecessor_manifest, _outputs_value, _manifests, _files, _pairs, audit, audit_outputs = state
    predecessor_files = {
        str(item["relative_path"]): item
        for item in predecessor_manifest["files"]
    }
    collision_file = next(
        item for item in audit["files"]
        if item["relative_path"] == "source-input-collisions.jsonl")
    collision_count = audit.get("summary", {}).get(
        "source_input_collision_record_count")
    if (type(collision_count) is not int or collision_count < 0
            or len(audit_outputs["source-input-collisions.jsonl"])
            != collision_count):
        raise BroadQaExternalDataError(
            "v10 expanded observation collision ledger 漂移")
    return {
        "artifact_kind": V10_SOURCE_EXPANSION_OBSERVATION_PACK_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {
            "five_family_audit_manifest_sha256": (
                V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256),
            "mixxx_source_pack_manifest_sha256": _NEW_PACK_SHA["MIXXX_PROJECT"],
            "mumble_source_pack_manifest_sha256": _NEW_PACK_SHA["MUMBLE_PROJECT"],
            "predecessor_family_observation_sha256s": {
                family: predecessor_files[name]["sha256"]
                for family, name in V10_SOURCE_EXPANSION_OBSERVATION_FILES
                if family not in _NEW_PACK_SHA
            },
            "predecessor_observation_pack_manifest_sha256": (
                V8_OBSERVATION_PACK_MANIFEST_SHA256),
            "source_input_collision_ledger_sha256": collision_file["sha256"],
        },
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": V10_SOURCE_EXPANSION_OBSERVATION_PACK_STATUS,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def publish_normalization_recovery_v10_source_expansion_observation_pack(
        *,
        run_root: str | Path,
        predecessor_observation_dir: str | Path,
        five_family_audit_dir: str | Path,
        roster_dir: str | Path,
        content_dir: str | Path,
        mixxx_source_pack_dir: str | Path,
        mumble_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布五family、按family分区的Observation pack。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      predecessor_observation_dir,
                      five_family_audit_dir,
                      roster_dir,
                      content_dir,
                      mixxx_source_pack_dir,
                      mumble_source_pack_dir,
                      target_dir,
                  )))
    predecessor, audit, roster, content, mixxx, mumble, target = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v10 expanded observation input/target path 非法")
    state = _state(
        observation_dir=predecessor,
        audit_dir=audit,
        roster_dir=roster,
        content_dir=content,
        new_source_pack_dirs={
            "MIXXX_PROJECT": mixxx,
            "MUMBLE_PROJECT": mumble,
        },
    )
    outputs, summary = _outputs(state)
    target.mkdir()
    files = []
    for name, role in _OUTPUT_FILES:
        path = target / name
        _write_jsonl(path, outputs[name])
        files.append(_artifact(path, role=role, count=len(outputs[name])))
    manifest = _manifest(files=files, summary=summary, state=state)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_source_expansion_observation_pack(
        observation_dir: str | Path,
        *,
        predecessor_observation_dir: str | Path,
        five_family_audit_dir: str | Path,
        roster_dir: str | Path,
        content_dir: str | Path,
        mixxx_source_pack_dir: str | Path,
        mumble_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格重派生五family Observation并拒绝records/manifest同步篡改。"""
    root = Path(observation_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 expanded observation manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != V10_SOURCE_EXPANSION_OBSERVATION_PACK_KIND):
        raise BroadQaExternalDataError(
            "v10 expanded observation manifest identity 漂移")
    state = _state(
        observation_dir=Path(predecessor_observation_dir).resolve(),
        audit_dir=Path(five_family_audit_dir).resolve(),
        roster_dir=Path(roster_dir).resolve(),
        content_dir=Path(content_dir).resolve(),
        new_source_pack_dirs={
            "MIXXX_PROJECT": Path(mixxx_source_pack_dir).resolve(),
            "MUMBLE_PROJECT": Path(mumble_source_pack_dir).resolve(),
        },
    )
    expected_outputs, summary = _outputs(state)
    stored_outputs = {
        name: _read_jsonl(root / name, label=role)
        for name, role in _OUTPUT_FILES
    }
    if stored_outputs != expected_outputs:
        raise BroadQaExternalDataError(
            "v10 expanded observation records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary, state=state)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v10 expanded observation fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


def read_normalization_recovery_v10_source_expansion_observation_aggregate(
        observation_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """只按sealed commitments回读五family Observation，不重开source pack。"""
    root = Path(observation_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 expanded observation aggregate manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != V10_SOURCE_EXPANSION_OBSERVATION_PACK_KIND):
        raise BroadQaExternalDataError(
            "v10 expanded observation aggregate identity 漂移")
    files = {
        str(item.get("relative_path")): item
        for item in stored.get("files", []) if isinstance(item, dict)
    }
    expected_names = {name for name, _role in _OUTPUT_FILES}
    try:
        physical = tuple(root.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 expanded observation aggregate inventory 不可读") from error
    if ({item.name for item in physical} != {"manifest.json", *expected_names}
            or any(item.is_dir() for item in physical)
            or set(files) != expected_names):
        raise BroadQaExternalDataError(
            "v10 expanded observation aggregate inventory 漂移")
    outputs = {}
    for name, role in _OUTPUT_FILES:
        values = _read_jsonl(root / name, label=role)
        if files[name] != _artifact(root / name, role=role, count=len(values)):
            raise BroadQaExternalDataError(
                "v10 expanded observation aggregate file 漂移")
        outputs[name] = values
    census = outputs["observation-census.jsonl"]
    if (len(outputs["family-census.jsonl"]) != 5
            or len(census) != 1
            or stored.get("summary") != {
                key: value for key, value in census[0].items()
                if key not in {"format_version", "record_kind"}
            }):
        raise BroadQaExternalDataError(
            "v10 expanded observation aggregate census 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, outputs


__all__ = [
    "V10_FIVE_FAMILY_AUDIT_MANIFEST_SHA256",
    "V10_SOURCE_EXPANSION_OBSERVATION_PACK_KIND",
    "V10_SOURCE_EXPANSION_OBSERVATION_PACK_STATUS",
    "publish_normalization_recovery_v10_source_expansion_observation_pack",
    "read_normalization_recovery_v10_source_expansion_observation_aggregate",
    "read_normalization_recovery_v10_source_expansion_observation_pack",
]
