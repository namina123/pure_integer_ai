"""共享的双 learner 运行等价与禁用态 materialized rule-pack runtime。"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


LearnerReader = Callable[
    [Path, Path, str],
    tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]],
]
MaterialLoader = Callable[[Path, str], dict[str, object]]
PayloadBuilder = Callable[[dict[str, tuple[dict[str, object], ...]]], dict[str, bytes]]
_RESERVED_ARTIFACT_FILES = {
    "checkpoints.jsonl", "manifest.json", "resume-markers.jsonl"}


def _sha256(payload: bytes) -> str:
    """返回规范 artifact、文件或结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


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


def require_k_run_root(value: str | Path, *, label: str) -> Path:
    """要求 pack 发布根是已存在 K 盘目录。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(f"{label} run root 必须是 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析 publisher 输入输出并拒绝逃出 K 盘 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    """判断两个 artifact 根是否相同或存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _validate_output_file_roles(
        output_file_roles: object,
        *,
        label: str,
        ) -> tuple[tuple[str, str, str], ...]:
    """核验 pack 语义输出只能使用互异单层文件名。"""
    if (not isinstance(output_file_roles, tuple) or not output_file_roles
            or any(type(item) is not tuple or len(item) != 3
                   for item in output_file_roles)):
        raise BroadQaExternalDataError(
            f"{label} output file roles schema 漂移")
    names = []
    for name, role, identity_key in output_file_roles:
        if (not all(isinstance(value, str) and value
                    for value in (name, role, identity_key))
                or name in _RESERVED_ARTIFACT_FILES):
            raise BroadQaExternalDataError(
                f"{label} output file role 非法")
        path = Path(name)
        if (path.is_absolute() or path.drive or path.name != name
                or name in {".", ".."}):
            raise BroadQaExternalDataError(
                f"{label} output path 非法")
        names.append(name)
    if len(set(names)) != len(names):
        raise BroadQaExternalDataError(f"{label} output path 重复")
    return output_file_roles


def _validate_output_material(
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        label: str,
        ) -> None:
    """要求 pack 的记录 identity、规范 JSONL 与 payload 一致。"""
    names = {name for name, _role, _identity in output_file_roles}
    if (not isinstance(outputs, dict) or not isinstance(payloads, dict)
            or set(outputs) != names or set(payloads) != names):
        raise BroadQaExternalDataError(f"{label} output inventory 漂移")
    for name, _role, identity_key in output_file_roles:
        values = outputs[name]
        payload = payloads[name]
        if not isinstance(values, tuple) or not isinstance(payload, bytes):
            raise BroadQaExternalDataError(f"{label} output type 漂移")
        identities = []
        encoded = []
        for value in values:
            if (not isinstance(value, dict)
                    or not isinstance(value.get(identity_key), str)
                    or not value[identity_key]):
                raise BroadQaExternalDataError(
                    f"{label} {name} record identity 漂移")
            identities.append(value[identity_key])
            encoded.append(canonical_json_line(value))
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"{label} {name} record identity 重复")
        if b"".join(encoded) != payload:
            raise BroadQaExternalDataError(
                f"{label} {name} payload/record 漂移")


def _files(
        *,
        output_file_roles: tuple[tuple[str, str, str], ...],
        outputs: dict[str, tuple[dict[str, object], ...]],
        payloads: dict[str, bytes],
        ) -> list[dict[str, object]]:
    """构造 pack 内全部语义输出的物理承诺。"""
    return [{
        "bytes": len(payloads[name]),
        "record_count": len(outputs[name]),
        "relative_path": name,
        "role": role,
        "sha256": _sha256(payloads[name]),
    } for name, role, _identity in output_file_roles]


def _semantic_result_sha256(
        *,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        ) -> str:
    """重算与两个 learner run identity 无关的唯一语义结果。"""
    return _sha256(canonical_json_bytes({
        "files": files,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "summary": summary,
    }))


def _lineage(manifest: dict[str, object], *, role: str) -> dict[str, object]:
    """从一个完整 learner manifest 投影 pack 所需运行血缘。"""
    return {
        "checkpoint_chain_sha256": manifest["checkpoint_chain_sha256"],
        "checkpoint_terminal_sha256": manifest["checkpoint_terminal_sha256"],
        "learner_manifest_sha256": manifest["manifest_sha256"],
        "resume_marker_count": manifest["resume_markers"]["record_count"],
        "role": role,
        "run_id": manifest["run_id"],
    }


def _validate_lineages(
        lineages: object,
        *,
        label: str,
        ) -> list[dict[str, object]]:
    """核验 fresh/resumed 两条独立运行血缘及 resume 证据。"""
    expected_fields = {
        "checkpoint_chain_sha256", "checkpoint_terminal_sha256",
        "learner_manifest_sha256", "resume_marker_count", "role", "run_id",
    }
    if (not isinstance(lineages, list) or len(lineages) != 2
            or any(not isinstance(item, dict) or set(item) != expected_fields
                   for item in lineages)
            or [item["role"] for item in lineages] != ["FRESH", "RESUMED"]
            or lineages[0]["run_id"] == lineages[1]["run_id"]
            or type(lineages[0]["resume_marker_count"]) is not int
            or lineages[0]["resume_marker_count"] != 0
            or type(lineages[1]["resume_marker_count"]) is not int
            or lineages[1]["resume_marker_count"] < 1):
        raise BroadQaExternalDataError(
            f"{label} fresh/resumed lineage 漂移")
    for lineage in lineages:
        for name in (
                "checkpoint_chain_sha256", "checkpoint_terminal_sha256",
                "learner_manifest_sha256", "run_id"):
            _sha_value(lineage[name], label=f"{label} lineage {name}")
    return lineages


def _manifest(
        *,
        artifact_kind: str,
        status: str,
        format_version: int,
        protocol_manifest_sha256: str,
        files: list[dict[str, object]],
        summary: dict[str, object],
        lineages: list[dict[str, object]],
        fixed_manifest_fields: dict[str, object],
        ) -> dict[str, object]:
    """构造未评测、未部署、生产禁用的 pack manifest。"""
    base = {
        "artifact_kind": artifact_kind,
        "files": files,
        "format_version": format_version,
        "fresh_resume_output_bytes_equal": 1,
        "learner_lineages": lineages,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": _semantic_result_sha256(
            protocol_manifest_sha256=protocol_manifest_sha256,
            files=files,
            summary=summary,
        ),
        "status": status,
        "summary": summary,
    }
    if set(base).intersection(fixed_manifest_fields):
        raise BroadQaExternalDataError(
            "materialized rule pack fixed manifest field 冲突")
    return {**base, **fixed_manifest_fields}


def _material(
        material_loader: MaterialLoader,
        protocol_root: Path,
        protocol_sha: str,
        output_file_roles: tuple[tuple[str, str, str], ...],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]],
            dict[str, object],
            dict[str, bytes],
        ]:
    """核验 pack reader 所需 protocol 派生输出。"""
    value = material_loader(protocol_root, protocol_sha)
    if (not isinstance(value, dict)
            or not isinstance(value.get("outputs"), dict)
            or not isinstance(value.get("summary"), dict)
            or not isinstance(value.get("payloads"), dict)):
        raise BroadQaExternalDataError(
            "materialized rule pack adapter schema 漂移")
    outputs = value["outputs"]
    summary = value["summary"]
    payloads = value["payloads"]
    _validate_output_material(
        output_file_roles=_validate_output_file_roles(
            output_file_roles, label="materialized rule pack"),
        outputs=outputs,
        payloads=payloads,
        label="materialized rule pack",
    )
    return outputs, summary, payloads


def publish_materialized_rule_pack(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        fresh_run_dir: str | Path,
        resumed_run_dir: str | Path,
        target_dir: str | Path,
        label: str,
        artifact_kind: str,
        status: str,
        format_version: int,
        output_file_roles: tuple[tuple[str, str, str], ...],
        learner_reader: LearnerReader,
        payload_builder: PayloadBuilder,
        fixed_manifest_fields: dict[str, object],
        ) -> dict[str, object]:
    """验证双运行语义字节相等后，不可覆盖发布禁用态 pack。"""
    root = require_k_run_root(run_root, label=label)
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    fresh_root = _within(root, fresh_run_dir, label="fresh_run_dir")
    resumed_root = _within(root, resumed_run_dir, label="resumed_run_dir")
    target = _within(root, target_dir, label="target_dir")
    roles = _validate_output_file_roles(output_file_roles, label=label)
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label=f"{label} protocol manifest")
    roots = (protocol_root, fresh_root, resumed_root, target)
    if (any(_paths_overlap(left, right)
            for ordinal, left in enumerate(roots)
            for right in roots[ordinal + 1:])
            or target.exists()
            or not fresh_root.is_dir() or not resumed_root.is_dir()):
        raise BroadQaExternalDataError(
            f"{label} run 混淆、缺失或 target 已存在")
    fresh_manifest, fresh_outputs = learner_reader(
        fresh_root, protocol_root, protocol_sha)
    resumed_manifest, resumed_outputs = learner_reader(
        resumed_root, protocol_root, protocol_sha)
    fresh_payloads = payload_builder(fresh_outputs)
    resumed_payloads = payload_builder(resumed_outputs)
    _validate_output_material(
        output_file_roles=roles,
        outputs=fresh_outputs,
        payloads=fresh_payloads,
        label=label,
    )
    _validate_output_material(
        output_file_roles=roles,
        outputs=resumed_outputs,
        payloads=resumed_payloads,
        label=label,
    )
    if (fresh_payloads != resumed_payloads
            or fresh_manifest["semantic_result_sha256"]
            != resumed_manifest["semantic_result_sha256"]
            or not _strict_equal(
                fresh_manifest["summary"], resumed_manifest["summary"])):
        raise BroadQaExternalDataError(
            f"{label} learner fresh/resume 语义输出不等价")
    lineages = _validate_lineages([
        _lineage(fresh_manifest, role="FRESH"),
        _lineage(resumed_manifest, role="RESUMED"),
    ], label=label)
    files = _files(
        output_file_roles=output_file_roles,
        outputs=fresh_outputs,
        payloads=fresh_payloads,
    )
    manifest = _manifest(
        artifact_kind=artifact_kind,
        status=status,
        format_version=format_version,
        protocol_manifest_sha256=protocol_sha,
        files=files,
        summary=fresh_manifest["summary"],
        lineages=lineages,
        fixed_manifest_fields=fixed_manifest_fields,
    )
    if manifest["semantic_result_sha256"] != fresh_manifest[
            "semantic_result_sha256"]:
        raise BroadQaExternalDataError(
            f"{label} pack/learner semantic result 漂移")
    target.mkdir(parents=True)
    for name, _role, _identity in output_file_roles:
        with (target / name).open("xb") as handle:
            handle.write(fresh_payloads[name])
    manifest_path = target / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256(manifest_path.read_bytes())}


def read_materialized_rule_pack(
        pack_dir: str | Path,
        *,
        protocol_dir: str | Path,
        expected_protocol_manifest_sha256: str,
        expected_pack_manifest_sha256: str,
        label: str,
        artifact_kind: str,
        status: str,
        format_version: int,
        output_file_roles: tuple[tuple[str, str, str], ...],
        material_loader: MaterialLoader,
        fixed_manifest_fields: dict[str, object],
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """以双外部 SHA 和 protocol 重派生严格回读禁用态 pack。"""
    root = Path(pack_dir).resolve()
    protocol_root = Path(protocol_dir).resolve()
    if _paths_overlap(protocol_root, root):
        raise BroadQaExternalDataError(
            f"{label} protocol/pack_dir 不得重叠")
    protocol_sha = _sha_value(
        expected_protocol_manifest_sha256,
        label=f"{label} expected protocol manifest")
    pack_sha = _sha_value(
        expected_pack_manifest_sha256,
        label=f"{label} expected manifest")
    outputs, summary, payloads = _material(
        material_loader, protocol_root, protocol_sha, output_file_roles)
    try:
        encoded = (root / "manifest.json").read_bytes()
        manifest = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} manifest 不可读") from error
    if (_sha256(encoded) != pack_sha or not isinstance(manifest, dict)
            or canonical_json_line(manifest) != encoded):
        raise BroadQaExternalDataError(
            f"{label} manifest identity/encoding 漂移")
    lineages = _validate_lineages(
        manifest.get("learner_lineages"), label=label)
    for name, _role, _identity in output_file_roles:
        try:
            stored = (root / name).read_bytes()
        except OSError as error:
            raise BroadQaExternalDataError(
                f"{label} {name} 不可读") from error
        if stored != payloads[name]:
            raise BroadQaExternalDataError(
                f"{label} {name} 与 protocol 派生漂移")
    expected = _manifest(
        artifact_kind=artifact_kind,
        status=status,
        format_version=format_version,
        protocol_manifest_sha256=protocol_sha,
        files=_files(
            output_file_roles=output_file_roles,
            outputs=outputs,
            payloads=payloads,
        ),
        summary=summary,
        lineages=lineages,
        fixed_manifest_fields=fixed_manifest_fields,
    )
    if not _strict_equal(manifest, expected):
        raise BroadQaExternalDataError(f"{label} manifest 漂移")
    return ({**manifest, "manifest_sha256": pack_sha}, outputs)


__all__ = [
    "publish_materialized_rule_pack",
    "read_materialized_rule_pack",
    "require_k_run_root",
]
