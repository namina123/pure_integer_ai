"""发布并严格回读 recovery-v10 五个 TRAIN family 审计。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack import (
    NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit_records import (
    V10_FIVE_FAMILY_AUDIT_CENSUS_KIND,
    V10_FIVE_FAMILY_AUDIT_FAMILIES,
    derive_normalization_recovery_v10_five_family_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_roster import (
    V8_OBSERVATION_PACK_MANIFEST_SHA256,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_source_pack import (
    V10_SOURCE_EXPANSION_SOURCE_PACK_KIND,
    read_normalization_recovery_v10_source_expansion_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


V10_FIVE_FAMILY_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_AUDIT_V1")
V10_FIVE_FAMILY_AUDIT_PASS_STATUS = (
    "FIVE_INDEPENDENT_TRAIN_FAMILIES_COLLISIONS_FROZEN_NOT_OBSERVED")
V10_FIVE_FAMILY_AUDIT_REJECTED_STATUS = (
    "FIVE_FAMILY_SOURCE_INDEPENDENCE_OR_COPY_GATE_REJECTED")

_OLD_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)
_NEW_FAMILIES = ("MIXXX_PROJECT", "MUMBLE_PROJECT")
_OLD_OBSERVATION_FILES = {
    "QBITTORRENT_PROJECT": "qbittorrent-observations.jsonl",
    "STELLARIUM_PROJECT": "stellarium-observations.jsonl",
    "KEEPASSXC_PROJECT": "keepassxc-observations.jsonl",
}
_PACK_SHA = {
    "QBITTORRENT_PROJECT": (
        "0a0d29bbcbb6d3470a458f1762f5be82963a1173fb5c66795e4637b29a1dad36"),
    "STELLARIUM_PROJECT": (
        "459adcadb7000c232ee7e8004a03aca9ba31971a3690a529f9c51bcc66917212"),
    "KEEPASSXC_PROJECT": (
        "ad1223f64bd09706a48cd1855000a8c6437873696097be56e1c26af45753ade3"),
    "MIXXX_PROJECT": (
        "fd9eed181a2c551966e1792777186de10051458d9ce21a187b80c82055715d9c"),
    "MUMBLE_PROJECT": (
        "c336cb36a9eeeda51282130ab20f75136f9cddabb7f5ef60b57dbc8532c03681"),
}
_OUTPUT_FILES = (
    ("family-census.jsonl", "V10_FIVE_FAMILY_CENSUS"),
    ("pairwise-overlap.jsonl", "V10_FIVE_FAMILY_PAIRWISE_OVERLAP"),
    ("source-input-collisions.jsonl", "V10_FIVE_FAMILY_COLLISION_LEDGER"),
    ("audit-census.jsonl", "V10_FIVE_FAMILY_AUDIT_CENSUS"),
)


def _sha256(payload: bytes) -> str:
    """返回artifact、record或manifest SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _require_k_root(value: str | Path) -> Path:
    """要求显式run root位于已存在K盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "v10 five-family audit run root 必须在K盘")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析并限制输入/输出仍位于本次K盘run root。"""
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise BroadQaExternalDataError(
            f"v10 five-family audit {label} 越出run root") from error
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
    """严格读取规范JSONL。"""
    values = []
    try:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        for line in lines:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BroadQaExternalDataError(
                    f"v10 five-family audit {label} record 非对象")
            values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            f"v10 five-family audit {label} 不可读") from error
    if (b"".join(lines) != payload
            or b"".join(canonical_json_line(item) for item in values)
            != payload):
        raise BroadQaExternalDataError(
            f"v10 five-family audit {label} JSONL 非规范")
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


def _validate_artifact_file(
        path: Path,
        commitment: dict[str, object],
        ) -> None:
    """流式核验无需解析的sealed predecessor文件。"""
    hasher = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 five-family predecessor file 不可读") from error
    if (size != commitment.get("bytes")
            or hasher.hexdigest() != commitment.get("sha256")):
        raise BroadQaExternalDataError(
            "v10 five-family predecessor file identity 漂移")


def _read_old_observations(
        observation_dir: Path,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """严格核验sealed旧Observation并读取三份family surface分区。"""
    path = observation_dir / "manifest.json"
    try:
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 five-family predecessor Observation manifest 不可读") from error
    if (_sha256(encoded) != V8_OBSERVATION_PACK_MANIFEST_SHA256
            or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded
            or manifest.get("artifact_kind")
            != "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_OBSERVATION_PACK_V1"):
        raise BroadQaExternalDataError(
            "v10 five-family predecessor Observation identity 漂移")
    files = {
        str(item.get("relative_path")): item
        for item in manifest.get("files", []) if isinstance(item, dict)
    }
    expected_names = {"manifest.json", *files}
    try:
        physical = tuple(observation_dir.iterdir())
    except OSError as error:
        raise BroadQaExternalDataError(
            "v10 five-family predecessor Observation inventory 不可读") from error
    if ({item.name for item in physical} != expected_names
            or any(item.is_dir() for item in physical)
            or not set(_OLD_OBSERVATION_FILES.values()).issubset(files)):
        raise BroadQaExternalDataError(
            "v10 five-family predecessor Observation inventory 漂移")
    pairs = {}
    for family, name in _OLD_OBSERVATION_FILES.items():
        commitment = files[name]
        values = _read_jsonl(observation_dir / name, label=family)
        if (len(values) != commitment.get("record_count")
                or _artifact(
                    observation_dir / name,
                    role=str(commitment.get("role")),
                    count=len(values),
                ) != commitment
                or any(item.get("source_family") != family for item in values)):
            raise BroadQaExternalDataError(
                "v10 five-family predecessor Observation records 漂移")
        pairs[family] = values
    for name, commitment in files.items():
        if name not in _OLD_OBSERVATION_FILES.values():
            _validate_artifact_file(observation_dir / name, commitment)
    summary = manifest.get("summary")
    if (not isinstance(summary, dict)
            or summary.get("family_count") != 3
            or summary.get("observation_count") != 33_179
            or sum(len(value) for value in pairs.values()) != 33_179):
        raise BroadQaExternalDataError(
            "v10 five-family predecessor denominator 漂移")
    return manifest, pairs


def _read_old_source_pack_manifest(
        source_pack_dir: Path,
        *,
        family: str,
        ) -> dict[str, object]:
    """按固定SHA读取旧source-pack manifest，不重扫其raw。"""
    path = source_pack_dir / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 five-family old source-pack manifest 不可读") from error
    if (_sha256(encoded) != _PACK_SHA[family]
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V8_SOURCE_PACK_KIND
            or stored.get("source_family") != family
            or stored.get("source_family_vote_count") != 1):
        raise BroadQaExternalDataError(
            "v10 five-family old source-pack identity 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}


def _state(
        *,
        observation_dir: Path,
        roster_dir: Path,
        content_dir: Path,
        source_pack_dirs: dict[str, Path],
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """读取旧三家sealed Observation并严格重派生新两家source pack。"""
    if set(source_pack_dirs) != set(V10_FIVE_FAMILY_AUDIT_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 five-family source-pack path inventory 漂移")
    observation_manifest, pairs = _read_old_observations(observation_dir)
    inputs = observation_manifest.get("inputs")
    if (not isinstance(inputs, dict)
            or inputs.get("qbittorrent_source_pack_manifest_sha256")
            != _PACK_SHA["QBITTORRENT_PROJECT"]
            or inputs.get("stellarium_source_pack_manifest_sha256")
            != _PACK_SHA["STELLARIUM_PROJECT"]
            or inputs.get("keepassxc_source_pack_manifest_sha256")
            != _PACK_SHA["KEEPASSXC_PROJECT"]):
        raise BroadQaExternalDataError(
            "v10 five-family predecessor lineage 漂移")
    manifests = {
        family: _read_old_source_pack_manifest(
            source_pack_dirs[family], family=family)
        for family in _OLD_FAMILIES
    }
    for family in _NEW_FAMILIES:
        manifest, _files, pair_records, _census = (
            read_normalization_recovery_v10_source_expansion_source_pack(
                source_pack_dirs[family],
                roster_dir=roster_dir,
                content_dir=content_dir,
                expected_manifest_sha256=_PACK_SHA[family],
            ))
        if (manifest.get("artifact_kind") != V10_SOURCE_EXPANSION_SOURCE_PACK_KIND
                or manifest.get("source_family") != family):
            raise BroadQaExternalDataError(
                "v10 five-family new source-pack identity 漂移")
        manifests[family] = manifest
        pairs[family] = pair_records
    return manifests, pairs


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> None:
    """不可覆盖写入规范JSONL。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))


def _manifest(
        *,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> dict[str, object]:
    """构造只含aggregate与哈希化collision ledger的manifest。"""
    status = (
        V10_FIVE_FAMILY_AUDIT_PASS_STATUS
        if summary["hard_independence_failure_count"] == 0
        else V10_FIVE_FAMILY_AUDIT_REJECTED_STATUS)
    return {
        "artifact_kind": V10_FIVE_FAMILY_AUDIT_KIND,
        "files": files,
        "format_version": 1,
        "inputs": {
            "mixxx_source_pack_manifest_sha256": _PACK_SHA["MIXXX_PROJECT"],
            "mumble_source_pack_manifest_sha256": _PACK_SHA["MUMBLE_PROJECT"],
            "predecessor_observation_pack_manifest_sha256": (
                V8_OBSERVATION_PACK_MANIFEST_SHA256),
            "predecessor_source_pack_manifest_sha256s": {
                family: _PACK_SHA[family] for family in _OLD_FAMILIES
            },
        },
        "mastery_claimed": 0,
        "observation_pack_published": 0,
        "pair_surface_public_git_count": 0,
        "production_enabled": 0,
        "status": status,
        "summary": summary,
        "teacher_api_llm_call_count": 0,
        "train_protocol_published": 0,
    }


def _outputs(
        state: tuple[
            dict[str, dict[str, object]],
            dict[str, tuple[dict[str, object], ...]],
        ],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """把纯派生结果映射到固定输出文件。"""
    family, pairwise, collisions, summary = (
        derive_normalization_recovery_v10_five_family_audit(*state))
    return {
        "family-census.jsonl": family,
        "pairwise-overlap.jsonl": pairwise,
        "source-input-collisions.jsonl": collisions,
        "audit-census.jsonl": ({
            **summary,
            "format_version": 1,
            "record_kind": V10_FIVE_FAMILY_AUDIT_CENSUS_KIND,
        },),
    }, summary


def publish_normalization_recovery_v10_five_family_audit(
        *,
        run_root: str | Path,
        observation_dir: str | Path,
        roster_dir: str | Path,
        content_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        mixxx_source_pack_dir: str | Path,
        mumble_source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布五family完整分母、overlap与collision audit。"""
    root = _require_k_root(run_root)
    paths = tuple(_within(root, value, label=str(index)) for index, value in
                  enumerate((
                      observation_dir,
                      roster_dir,
                      content_dir,
                      qbittorrent_source_pack_dir,
                      stellarium_source_pack_dir,
                      keepassxc_source_pack_dir,
                      mixxx_source_pack_dir,
                      mumble_source_pack_dir,
                      target_dir,
                  )))
    (observation, roster, content, qbit, stellarium, keepassxc,
     mixxx, mumble, target) = paths
    if (target.exists()
            or any(not path.is_dir() for path in paths[:-1])
            or any(_overlap(target, path) for path in paths[:-1])):
        raise BroadQaExternalDataError(
            "v10 five-family audit input/target path 非法")
    state = _state(
        observation_dir=observation,
        roster_dir=roster,
        content_dir=content,
        source_pack_dirs={
            "QBITTORRENT_PROJECT": qbit,
            "STELLARIUM_PROJECT": stellarium,
            "KEEPASSXC_PROJECT": keepassxc,
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
    manifest = _manifest(files=files, summary=summary)
    path = target / "manifest.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(path.read_bytes())}


def read_normalization_recovery_v10_five_family_audit(
        audit_dir: str | Path,
        *,
        observation_dir: str | Path,
        roster_dir: str | Path,
        content_dir: str | Path,
        qbittorrent_source_pack_dir: str | Path,
        stellarium_source_pack_dir: str | Path,
        keepassxc_source_pack_dir: str | Path,
        mixxx_source_pack_dir: str | Path,
        mumble_source_pack_dir: str | Path,
        expected_manifest_sha256: str,
        ) -> tuple[
            dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """从五份固定来源重派生audit并拒绝同步篡改。"""
    root = Path(audit_dir).resolve()
    path = root / "manifest.json"
    try:
        encoded = path.read_bytes()
        stored = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "v10 five-family audit manifest 不可读") from error
    if (_sha256(encoded) != expected_manifest_sha256
            or not isinstance(stored, dict)
            or canonical_json_line(stored) != encoded
            or stored.get("artifact_kind") != V10_FIVE_FAMILY_AUDIT_KIND):
        raise BroadQaExternalDataError(
            "v10 five-family audit manifest identity 漂移")
    state = _state(
        observation_dir=Path(observation_dir).resolve(),
        roster_dir=Path(roster_dir).resolve(),
        content_dir=Path(content_dir).resolve(),
        source_pack_dirs={
            "QBITTORRENT_PROJECT": Path(qbittorrent_source_pack_dir).resolve(),
            "STELLARIUM_PROJECT": Path(stellarium_source_pack_dir).resolve(),
            "KEEPASSXC_PROJECT": Path(keepassxc_source_pack_dir).resolve(),
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
            "v10 five-family audit records 漂移")
    files = [
        _artifact(root / name, role=role, count=len(expected_outputs[name]))
        for name, role in _OUTPUT_FILES
    ]
    expected = _manifest(files=files, summary=summary)
    if stored != expected:
        raise BroadQaExternalDataError(
            "v10 five-family audit fields 漂移")
    return {**stored, "manifest_sha256": _sha256(encoded)}, stored_outputs


__all__ = [
    "V10_FIVE_FAMILY_AUDIT_KIND",
    "V10_FIVE_FAMILY_AUDIT_PASS_STATUS",
    "V10_FIVE_FAMILY_AUDIT_REJECTED_STATUS",
    "publish_normalization_recovery_v10_five_family_audit",
    "read_normalization_recovery_v10_five_family_audit",
]
