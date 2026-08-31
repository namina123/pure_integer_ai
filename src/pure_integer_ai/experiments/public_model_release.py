"""可搬运的公开模型 release root 契约。

发布根同时承载训练状态、公开课程、窄域快照、广域 SQLite 索引和来源
许可账本。所有入口只接受相对路径；manifest 之外不依赖发布机上的绝对
路径。数据文件仍是 SQLite/JSONL 等开发期载体，永久身份由字节 hash 和
整数训练状态绑定，便于后续迁移到其他语言或存储后端。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
from typing import Iterable

from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
)


PUBLIC_MODEL_RELEASE_FORMAT = "PURE_INTEGER_AI_PUBLIC_MODEL_RELEASE"
PUBLIC_MODEL_RELEASE_SCHEMA_VERSION = 1
PUBLIC_MODEL_RELEASE_MANIFEST = "public_model_release.json"
PUBLIC_MODEL_RELEASE_DIGEST = "public_model_release.sha256"
PUBLIC_DIALOGUE_PROTOCOL_CONFIG = "model/dialogue_protocol.json"
PUBLIC_DIALOGUE_RESPONSE_ARTIFACT = "model/artifacts/dialogue_response"
PUBLIC_DIALOGUE_DOMAIN_EXPERT_ROOT = "model/artifacts/dialogue_domain_experts"
PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE = "model/dialogue_expert_router.int"
PUBLIC_DIALOGUE_SOURCE_MANIFEST_ROOT = "model/dialogue_source_manifests"
PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT = "model/artifacts/response_organization"
PUBLIC_SCIENCE_PASSAGE_ARTIFACT = "knowledge/artifacts/science_passage"

_DIALOGUE_RESPONSE_MANIFEST = "learned_dialogue_response_manifest.json"
_RESPONSE_ORGANIZATION_MANIFEST = "response_organization_manifest.json"
_SCIENCE_PASSAGE_MANIFEST = "scidb_csq_passage_index_manifest.json"


class PublicModelReleaseError(ValueError):
    """发布根缺失、越界、哈希漂移或携带绝对路径。"""


# These are transport/path fields used by the release and its nested public
# manifests.  URLs and source identities are intentionally not included.
_MANIFEST_PATH_KEYS = frozenset({
    "path", "source_manifest", "qa_database", "training_root",
    "sparse_snapshot", "protocol_config", "pack_manifest", "database",
    "dialogue_expert_routing_artifact",
    "training_cursor", "snapshot", "token_index_file", "aggregate_index_file",
})
_MANIFEST_PATH_LIST_KEYS = frozenset({
    "course_files", "course_sidecars", "surface_courses", "source_files",
    "extra_course_paths", "surface_evidence_files", "source_artifacts",
    "dialogue_domain_expert_artifacts",
    "dialogue_source_manifests",
})
_PRIVATE_ARTIFACT_RE = re.compile(
    r"(?:^|[_-])(private(?:[_-](?:evaluator|eval|label|payload|formal))|"
    r"formal[_-]private|w\d+[_-]private)(?:$|[_./-])",
    re.IGNORECASE,
)
_PRIVATE_DECLARATION_KEYS = frozenset({
    "private_evaluator", "private_eval", "private_labels",
    "private_payload", "private_formal_data",
})


def _reject_local_path_identity(value: object, *, label: str) -> None:
    """Source identities are portable names, never host filesystem paths."""
    if not isinstance(value, str) or not value:
        raise PublicModelReleaseError(f"{label} identity 非法")
    if (value.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:[/\\]", value)
            or "\\" in value):
        raise PublicModelReleaseError(f"{label} 不得包含本机绝对路径")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PublicModelReleaseError(f"{label} 必须是规范相对 POSIX 路径")
    path = PurePosixPath(value)
    # ``PurePosixPath('C:/x').parts`` is ('C:', 'x'), so checking for a
    # literal ':' component misses Windows drive-qualified paths.
    if (path.is_absolute() or ".." in path.parts
            or any(":" in part for part in path.parts)):
        raise PublicModelReleaseError(f"{label} 越出 release root")
    return path


def _resolve(root: Path, relative: object, *, label: str) -> Path:
    path = _relative_path(relative, label=label)
    target = (root / Path(*path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise PublicModelReleaseError(f"{label} 越出 release root") from error
    return target


def _validate_nested_manifest_paths(value: object, *, label: str,
                                    key: str | None = None) -> None:
    """Reject absolute/escaping paths in nested JSON manifests.

    Only fields whose schema denotes a file path are interpreted as paths;
    ordinary text, source URLs and hashes remain unrestricted.  List records
    such as ``source_files: [[path, sha256, count], ...]`` validate their first
    element only.
    """
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in _MANIFEST_PATH_KEYS:
                if isinstance(child, str):
                    _relative_path(child, label=f"{label}.{child_key}")
                elif child_key == "database" and isinstance(child, dict):
                    _validate_nested_manifest_paths(
                        child, label=f"{label}.{child_key}", key=child_key)
                elif child is not None:
                    raise PublicModelReleaseError(
                        f"{label}.{child_key} 必须是相对路径")
                continue
            if child_key in _MANIFEST_PATH_LIST_KEYS:
                if not isinstance(child, list):
                    raise PublicModelReleaseError(
                        f"{label}.{child_key} 必须是路径列表")
                for ordinal, item in enumerate(child):
                    candidate = item[0] if isinstance(item, list) else item
                    if not isinstance(candidate, str):
                        raise PublicModelReleaseError(
                            f"{label}.{child_key}[{ordinal}] 缺少相对路径")
                    _relative_path(
                        candidate,
                        label=f"{label}.{child_key}[{ordinal}]")
                continue
            if child_key == "source_identities":
                if not isinstance(child, list):
                    raise PublicModelReleaseError(
                        f"{label}.source_identities 必须是身份列表")
                for ordinal, item in enumerate(child):
                    if (not isinstance(item, list) or len(item) != 2
                            or not isinstance(item[0], str)
                            or not isinstance(item[1], str)):
                        raise PublicModelReleaseError(
                            f"{label}.source_identities[{ordinal}] 非法")
                    _relative_path(
                        item[0],
                        label=f"{label}.source_identities[{ordinal}].path")
                    _reject_local_path_identity(
                        item[1],
                        label=f"{label}.source_identities[{ordinal}]")
                continue
            _validate_nested_manifest_paths(
                child, label=f"{label}.{child_key}", key=child_key)
    elif isinstance(value, list):
        for ordinal, child in enumerate(value):
            _validate_nested_manifest_paths(
                child, label=f"{label}[{ordinal}]", key=key)


def _validate_no_private_artifacts(root: Path, *, manifest_values: tuple[object, ...]
                                   ) -> None:
    """Reject actual private evaluator payloads while allowing boundary flags.

    A public snapshot may legitimately say ``private_formal_data=FORBIDDEN``;
    that declaration is not private data and is therefore explicitly allowed.
    """
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if _PRIVATE_ARTIFACT_RE.search(relative):
            raise PublicModelReleaseError(
                f"release 不得携带 private evaluator artifact: {relative}")
    def walk(value: object, label: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in _PRIVATE_DECLARATION_KEYS:
                    # The sole public form is an explicit boundary declaration.
                    if child != "FORBIDDEN":
                        raise PublicModelReleaseError(
                            f"release 携带 private evaluator 数据: {label}.{key}")
                    continue
                walk(child, f"{label}.{key}")
        elif isinstance(value, list):
            for ordinal, child in enumerate(value):
                walk(child, f"{label}[{ordinal}]")
    for ordinal, value in enumerate(manifest_values):
        walk(value, f"manifest[{ordinal}]")


def _require_root(root: str | Path, *, require_k_drive: bool) -> Path:
    path = Path(root).resolve()
    if not path.is_dir():
        raise PublicModelReleaseError("release root 必须是已存在目录")
    if require_k_drive and path.drive.upper() != "K:":
        raise PublicModelReleaseError("release root 必须位于 K 盘")
    return path


@dataclass(frozen=True, slots=True)
class PublicModelRelease:
    """已验证发布根的只读路径投影。"""

    root: Path
    release_id: str
    qa_database: Path
    training_root: Path
    sparse_snapshot: Path
    source_manifest: Path
    protocol_config: Path
    dialogue_response_artifact: Path | None
    dialogue_domain_expert_artifacts: tuple[Path, ...]
    dialogue_expert_routing_artifact: Path | None
    response_organization_artifact: Path | None
    science_passage_artifact: Path | None
    manifest: dict[str, object]


def _optional_entry_directory(
        root: Path, entry: dict[str, object], *, field: str,
        label: str,
        ) -> Path | None:
    """解析可选 release 目录入口，存在声明时必须指向闭合目录。"""
    value = entry.get(field)
    if value is None:
        return None
    path = _resolve(root, value, label=f"entry.{field}")
    if not path.is_dir():
        raise PublicModelReleaseError(f"release entry 缺失: {label}")
    return path


def _entry_directories(
        root: Path, entry: dict[str, object], *, field: str, label: str,
        ) -> tuple[Path, ...]:
    """Resolve one ordered, unique list of embedded artifact directories."""
    value = entry.get(field, [])
    if not isinstance(value, list) or any(not isinstance(item, str)
                                          for item in value):
        raise PublicModelReleaseError(f"entry.{field} 必须是路径列表")
    paths = tuple(_resolve(root, item, label=f"entry.{field}")
                  for item in value)
    if len(set(paths)) != len(paths) or any(not path.is_dir() for path in paths):
        raise PublicModelReleaseError(f"release entry 缺失或重复: {label}")
    return paths


def _embedded_artifact_manifest(
        artifact_root: Path | None, *, manifest_name: str, label: str,
        ) -> dict[str, object] | None:
    """核验内置 artifact 的平面文件集合并回读其规范 manifest。"""
    if artifact_root is None:
        return None
    manifest_path = artifact_root / manifest_name
    if not manifest_path.is_file():
        raise PublicModelReleaseError(f"{label} 缺少 {manifest_name}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError(f"{label} manifest 不可回读") from error
    if not isinstance(manifest, dict):
        raise PublicModelReleaseError(f"{label} manifest 必须是对象")
    declared: set[str] = {manifest_name}
    files = manifest.get("files")
    if files is not None:
        if not isinstance(files, list) or not files:
            raise PublicModelReleaseError(f"{label} files 非法")
        for ordinal, item in enumerate(files):
            if not isinstance(item, dict):
                raise PublicModelReleaseError(
                    f"{label}.files[{ordinal}] 非法")
            relative = _relative_path(
                item.get("name"), label=f"{label}.files[{ordinal}].name")
            if len(relative.parts) != 1:
                raise PublicModelReleaseError(f"{label} 只允许平面 payload")
            declared.add(relative.as_posix())
    else:
        database = manifest.get("database")
        if not isinstance(database, dict):
            raise PublicModelReleaseError(f"{label} 缺少 payload inventory")
        relative = _relative_path(
            database.get("path"), label=f"{label}.database.path")
        if len(relative.parts) != 1:
            raise PublicModelReleaseError(f"{label} 只允许平面 payload")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.iterdir() if path.is_file()
    }
    if any(path.is_dir() or path.is_symlink()
           for path in artifact_root.iterdir()) or actual != declared:
        raise PublicModelReleaseError(
            f"{label} 文件集合不闭合: expected={sorted(declared)}, "
            f"actual={sorted(actual)}")
    _validate_nested_manifest_paths(manifest, label=label)
    return manifest


def load_public_model_release(
        root: str | Path, *, require_k_drive: bool = True,
        verify_payload_hashes: bool = True,
        ) -> PublicModelRelease:
    """验证 release root 并返回只读入口路径。

    ``verify_payload_hashes=False`` 是明确的启动性能档位：仍验证 manifest
    的闭合集合、文件大小、路径边界和 manifest digest，但跳过逐文件内容
    SHA-256。默认严格档位保持逐文件 hash；发布校验和篡改审计必须使用默认值。
    """
    if type(verify_payload_hashes) is not bool:
        raise TypeError("verify_payload_hashes 必须是严格 bool")
    target = _require_root(root, require_k_drive=require_k_drive)
    manifest_path = target / PUBLIC_MODEL_RELEASE_MANIFEST
    if not manifest_path.is_file():
        raise PublicModelReleaseError("release root 缺少 public_model_release.json")
    raw = manifest_path.read_bytes()
    if b"\x00" in raw or any(
            marker in raw for marker in (b"D:/", b"D:\\", b"K:/", b"K:\\")):
        raise PublicModelReleaseError("release manifest 不得含本机绝对路径")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError("release manifest 不可回读") from error
    if not isinstance(value, dict):
        raise PublicModelReleaseError("release manifest 必须是对象")
    if (value.get("format") != PUBLIC_MODEL_RELEASE_FORMAT
            or value.get("schema_version") != PUBLIC_MODEL_RELEASE_SCHEMA_VERSION):
        raise PublicModelReleaseError("release manifest 格式不兼容")
    release_id = value.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise PublicModelReleaseError("release_id 非法")
    entries = value.get("files")
    if not isinstance(entries, list) or not entries:
        raise PublicModelReleaseError("release manifest 缺少 files")
    seen: set[str] = set()
    for ordinal, item in enumerate(entries):
        if not isinstance(item, dict):
            raise PublicModelReleaseError(f"files[{ordinal}] 非法")
        relative = _relative_path(item.get("path"), label=f"files[{ordinal}].path")
        key = relative.as_posix()
        if key in seen or key in {PUBLIC_MODEL_RELEASE_MANIFEST,
                                  PUBLIC_MODEL_RELEASE_DIGEST}:
            raise PublicModelReleaseError("release files 路径重复或保留名冲突")
        seen.add(key)
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if type(size) is not int or size < 0 or (
                not isinstance(digest, str) or len(digest) != 64):
            raise PublicModelReleaseError(f"files[{ordinal}] 描述非法")
        path = _resolve(target, key, label=f"files[{ordinal}].path")
        if not path.is_file() or path.stat().st_size != size:
            raise PublicModelReleaseError(f"release file 缺失或大小漂移: {key}")
        if verify_payload_hashes and _sha256(path) != digest:
            raise PublicModelReleaseError(f"release file hash 漂移: {key}")
    # A manifest is a closed set, not merely an allow-list.  Unlisted files
    # would otherwise be silently shipped beside a valid model and could
    # contain private evaluator material or alter runtime behavior.
    actual: set[str] = set()
    for path in target.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise PublicModelReleaseError("release 不得包含符号链接")
        relative = path.relative_to(target).as_posix()
        if relative in {PUBLIC_MODEL_RELEASE_MANIFEST,
                        PUBLIC_MODEL_RELEASE_DIGEST}:
            continue
        actual.add(relative)
    if actual != seen:
        missing = sorted(seen - actual)
        extra = sorted(actual - seen)
        raise PublicModelReleaseError(
            f"release manifest 文件集合不闭合: missing={missing}, extra={extra}")
    digest_path = target / PUBLIC_MODEL_RELEASE_DIGEST
    if digest_path.is_file():
        expected = digest_path.read_text(encoding="ascii").strip()
        if expected != _sha256(manifest_path):
            raise PublicModelReleaseError("release manifest digest 漂移")
    entry = value.get("entry")
    if not isinstance(entry, dict):
        raise PublicModelReleaseError("release manifest 缺少 entry")
    qa_database = _resolve(target, entry.get("qa_database"), label="entry.qa_database")
    training_root = _resolve(target, entry.get("training_root"), label="entry.training_root")
    sparse_snapshot = _resolve(target, entry.get("sparse_snapshot"), label="entry.sparse_snapshot")
    source_manifest = _resolve(target, value.get("source_manifest"), label="source_manifest")
    protocol_config = _resolve(
        target, entry.get("protocol_config"), label="entry.protocol_config")
    dialogue_response_artifact = _optional_entry_directory(
        target, entry, field="dialogue_response_artifact",
        label="dialogue response artifact")
    dialogue_domain_expert_artifacts = _entry_directories(
        target, entry, field="dialogue_domain_expert_artifacts",
        label="dialogue domain expert artifacts")
    routing_value = entry.get("dialogue_expert_routing_artifact")
    dialogue_expert_routing_artifact = (
        _resolve(target, routing_value, label="entry.dialogue_expert_routing_artifact")
        if routing_value is not None else None)
    if dialogue_expert_routing_artifact is not None:
        if not dialogue_expert_routing_artifact.is_file():
            raise PublicModelReleaseError("release entry 缺失: dialogue expert router")
        if not dialogue_domain_expert_artifacts:
            raise PublicModelReleaseError("expert router 不得没有领域专家")
        from pure_integer_ai.experiments.dialogue_expert_routing_artifact import (
            load_dialogue_expert_routing_model,
        )
        routing_model = load_dialogue_expert_routing_model(
            dialogue_expert_routing_artifact)
        if len(routing_model.domain_activation_features) != len(
                dialogue_domain_expert_artifacts):
            raise PublicModelReleaseError("expert router 与领域目录数量不一致")
    response_organization_artifact = _optional_entry_directory(
        target, entry, field="response_organization_artifact",
        label="response organization artifact")
    science_passage_artifact = _optional_entry_directory(
        target, entry, field="science_passage_artifact",
        label="science passage artifact")
    for path in (qa_database, sparse_snapshot, source_manifest, protocol_config):
        if not path.is_file():
            raise PublicModelReleaseError(f"release entry 缺失: {path.name}")
    if not training_root.is_dir():
        raise PublicModelReleaseError("release training_root 缺失")
    try:
        sources = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError("source manifest 不可回读") from error
    if not isinstance(sources, dict) or sources.get("format") != "PUBLIC_SOURCE_MANIFEST_V1":
        raise PublicModelReleaseError("source manifest 格式不兼容")
    _validate_nested_manifest_paths(sources, label="source_manifest")
    training_manifest_path = target / "model" / "dialogue_pack_manifest.json"
    summary_path = target / "model" / "training_summary.json"
    try:
        training_manifest = json.loads(
            training_manifest_path.read_text(encoding="utf-8"))
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError("嵌套训练 manifest 不可回读") from error
    if not isinstance(training_manifest, dict):
        raise PublicModelReleaseError("嵌套 training pack manifest 必须是对象")
    if not isinstance(training_summary, dict):
        raise PublicModelReleaseError("嵌套 training summary 必须是对象")
    _validate_nested_manifest_paths(
        training_manifest, label="training_pack_manifest")
    _validate_nested_manifest_paths(training_summary, label="training_summary")
    try:
        protocol = json.loads(protocol_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError("dialogue protocol config 不可回读") from error
    if (not isinstance(protocol, dict)
            or protocol.get("format") != "PURE_INTEGER_AI_DIALOGUE_PROTOCOL_CONFIG"
            or protocol.get("schema_version") != 1
            or protocol.get("transport") != "jsonl"
            or protocol.get("encoding") != "utf-8"
            or protocol.get("operations") != ["turn", "quit", "exit"]
            or protocol.get("response") != {
                "type": "response",
                "text_field": "text",
                "internal_status_field": None,
            }):
        raise PublicModelReleaseError("dialogue protocol config 格式不兼容")
    _validate_nested_manifest_paths(protocol, label="dialogue_protocol")
    artifact_manifests = tuple(item for item in (
        _embedded_artifact_manifest(
            dialogue_response_artifact,
            manifest_name=_DIALOGUE_RESPONSE_MANIFEST,
            label="dialogue_response_artifact"),
        _embedded_artifact_manifest(
            response_organization_artifact,
            manifest_name=_RESPONSE_ORGANIZATION_MANIFEST,
            label="response_organization_artifact"),
        _embedded_artifact_manifest(
            science_passage_artifact,
            manifest_name=_SCIENCE_PASSAGE_MANIFEST,
            label="science_passage_artifact"),
        *(
            _embedded_artifact_manifest(
                path, manifest_name=_DIALOGUE_RESPONSE_MANIFEST,
                label=f"dialogue_domain_expert_artifact[{ordinal}]")
            for ordinal, path in enumerate(
                dialogue_domain_expert_artifacts)
        ),
    ) if item is not None)
    _validate_no_private_artifacts(
        target, manifest_values=(value, sources, training_manifest,
                                 training_summary, protocol,
                                 *artifact_manifests))
    return PublicModelRelease(
        target, release_id, qa_database, training_root, sparse_snapshot,
        source_manifest, protocol_config, dialogue_response_artifact,
        dialogue_domain_expert_artifacts,
        dialogue_expert_routing_artifact,
        response_organization_artifact, science_passage_artifact, value,
    )


def _source_path(project_root: Path, training_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise PublicModelReleaseError("训练 manifest source path 非法")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        fallback = (training_root / Path(value)).resolve()
        if fallback.is_file():
            candidate = fallback
    if not candidate.is_file():
        raise PublicModelReleaseError(f"训练来源缺失: {value}")
    return candidate


def _has_materialized_training_source_closure(
        training_root: Path, training_manifest: dict[str, object],
        summary: dict[str, object],
        ) -> bool:
    """Return whether this run already carries every graph source.

    Additive checkpoints usually need their ancestor manifests because their
    current pack contains only the new shard.  A portable recovery may instead
    replay the complete course pack while preserving the ancestor graph
    namespace.  In that case every declared source is materialized below the
    run root and the pack's train surface count equals the database source
    record count; following archived ancestors would add no identity and would
    make an otherwise closed run depend on deleted training backups.
    """
    source_count = summary.get("source_record_count")
    train_count = training_manifest.get("train_surface_count")
    if (type(source_count) is not int or source_count <= 0
            or train_count != source_count):
        return False
    path_values: list[object] = []
    for row in training_manifest.get("source_files", ()):
        if not isinstance(row, list) or len(row) != 3:
            return False
        path_values.append(row[0])
    path_values.extend(training_manifest.get("extra_course_paths", ()))
    for row in training_manifest.get("surface_evidence_files", ()):
        if not isinstance(row, list) or not row:
            return False
        path_values.append(row[0])
    if not path_values:
        return False
    root = training_root.resolve()
    for value in path_values:
        if not isinstance(value, str) or not value:
            return False
        relative = PurePosixPath(value)
        if (relative.is_absolute() or ".." in relative.parts
                or any(":" in part for part in relative.parts)):
            return False
        candidate = (root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
    return True


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise PublicModelReleaseError(f"拒绝覆盖发布文件: {target}")
    shutil.copyfile(source, target)


def _copy_closed_artifact(
        source_root: Path, target_root: Path, *, manifest_name: str,
        label: str,
        ) -> dict[str, object]:
    """复制已经验证过的平面 artifact，禁止额外文件随包进入 release。"""
    manifest = _embedded_artifact_manifest(
        source_root, manifest_name=manifest_name, label=label)
    assert manifest is not None
    target_root.mkdir(parents=True, exist_ok=False)
    for source in sorted(source_root.iterdir(), key=lambda item: item.name):
        _copy_file(source, target_root / source.name)
    return manifest


def _course_sidecars(source: Path) -> tuple[Path, ...]:
    """Resolve all declared integer sidecars beside an indexed course file."""
    found: list[Path] = []
    fields = ("token_index_file", "aggregate_index_file")
    try:
        for raw in source.read_bytes().splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                return ()
            for field in fields:
                name = value.get(field)
                if name is None:
                    continue
                if not isinstance(name, str) or not name:
                    return ()
                candidate = (source.parent / Path(name)).resolve()
                candidate.relative_to(source.parent.resolve())
                if not candidate.is_file():
                    return ()
                if candidate not in found:
                    found.append(candidate)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return ()
    return tuple(found)


def _course_sidecar(source: Path) -> Path | None:
    """Backward-compatible token sidecar resolver."""
    return next(iter(_course_sidecars(source)), None)


def _license_ids(paths: Iterable[Path]) -> tuple[str, ...]:
    values: set[str] = set()
    for path in paths:
        for raw in path.read_bytes().splitlines():
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("license_id"), str):
                values.add(value["license_id"])
    return tuple(sorted(values))


def _qa_metadata(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("select key,value from metadata order by key").fetchall()
    finally:
        connection.close()
    return {str(key): str(value) for key, value in rows}


def _dialogue_source_manifest(path: Path) -> dict[str, object]:
    """Validate one compact public source/license ledger before release copy."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicModelReleaseError("dialogue source manifest 不可回读") from error
    source = value.get("source") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or not isinstance(source, dict)
            or not isinstance(source.get("license_id"), str)
            or not source["license_id"].strip()
            or not isinstance(value.get("course"), dict)
            or not isinstance(value["course"].get("sha256"), str)):
        raise PublicModelReleaseError("dialogue source manifest 身份非法")
    _validate_nested_manifest_paths(value, label="dialogue_source_manifest")
    return value


def build_public_model_release(
        *, project_root: str | Path, qa_database: str | Path,
        training_run_root: str | Path, release_root: str | Path,
        release_id: str,
        dialogue_response_artifact_root: str | Path | None = None,
        dialogue_domain_expert_artifact_roots: tuple[str | Path, ...] = (),
        dialogue_expert_routing_artifact_root: str | Path | None = None,
        dialogue_source_manifest_paths: tuple[str | Path, ...] = (),
        response_organization_artifact_root: str | Path | None = None,
        science_passage_artifact_root: str | Path | None = None,
        require_k_drive: bool = True,
        ) -> PublicModelRelease:
    """从现有训练 run 和广域索引构造一次性公开 release root。"""
    if not isinstance(release_id, str) or not release_id.strip():
        raise PublicModelReleaseError("release_id 非法")
    project = Path(project_root).resolve()
    qa = Path(qa_database).resolve()
    training = _require_root(training_run_root, require_k_drive=require_k_drive)
    source_release = None
    if (project / PUBLIC_MODEL_RELEASE_MANIFEST).is_file():
        source_release = load_public_model_release(
            project, require_k_drive=require_k_drive,
            verify_payload_hashes=False)
        if (training != source_release.training_root
                or qa != source_release.qa_database):
            raise PublicModelReleaseError(
                "release 派生构建必须使用其自身 training/QA 入口")
    target = Path(release_root).resolve()
    if require_k_drive and target.drive.upper() != "K:":
        raise PublicModelReleaseError("release root 必须位于 K 盘")
    if target.exists():
        raise PublicModelReleaseError("release root 已存在，拒绝覆盖")
    if not qa.is_file() or (require_k_drive and qa.drive.upper() != "K:"):
        raise PublicModelReleaseError("qa_database 必须是 K 盘已存在文件")
    optional_artifact_roots: dict[str, Path] = {}
    for name, value in (
            ("dialogue_response", dialogue_response_artifact_root),
            ("response_organization", response_organization_artifact_root),
            ("science_passage", science_passage_artifact_root)):
        if value is None:
            continue
        optional_artifact_roots[name] = _require_root(
            value, require_k_drive=require_k_drive)
    domain_artifact_roots = tuple(
        _require_root(value, require_k_drive=require_k_drive)
        for value in dialogue_domain_expert_artifact_roots)
    if len(set(domain_artifact_roots)) != len(domain_artifact_roots):
        raise PublicModelReleaseError("领域对话专家 root 重复")
    if domain_artifact_roots and "dialogue_response" not in optional_artifact_roots:
        raise PublicModelReleaseError("领域对话专家必须绑定通用对话专家")
    routing_source = None
    if dialogue_expert_routing_artifact_root is not None:
        routing_source = Path(dialogue_expert_routing_artifact_root).resolve()
        if not routing_source.is_file():
            raise PublicModelReleaseError("dialogue expert router 必须是已存在文件")
        if require_k_drive and routing_source.drive.upper() != "K:":
            raise PublicModelReleaseError("dialogue expert router 必须位于 K 盘")
    if domain_artifact_roots and routing_source is None:
        raise PublicModelReleaseError("领域专家必须绑定 dialogue expert router")
    routing_model = None
    if routing_source is not None:
        from pure_integer_ai.experiments.dialogue_expert_routing_artifact import (
            load_dialogue_expert_routing_model,
        )
        routing_model = load_dialogue_expert_routing_model(routing_source)
        if len(routing_model.domain_activation_features) != len(
                domain_artifact_roots):
            raise PublicModelReleaseError("expert router 与领域 root 数量不一致")
    dialogue_source_manifests = tuple(
        Path(value).resolve() for value in dialogue_source_manifest_paths)
    has_dialogue_artifacts = "dialogue_response" in optional_artifact_roots
    if has_dialogue_artifacts and not dialogue_source_manifests:
        raise PublicModelReleaseError(
            "含对话 artifact 的 release 必须提供来源 manifest")
    if (dialogue_source_manifests
            and len(set(dialogue_source_manifests))
            != len(dialogue_source_manifests)):
        raise PublicModelReleaseError(
            "dialogue source manifest 集合必须唯一")
    dialogue_source_values = []
    for source in dialogue_source_manifests:
        if (not source.is_file()
                or require_k_drive and source.drive.upper() != "K:"):
            raise PublicModelReleaseError(
                "dialogue source manifest 必须位于 K 盘")
        dialogue_source_values.append(_dialogue_source_manifest(source))
    if "dialogue_response" in optional_artifact_roots:
        from pure_integer_ai.experiments.build_learned_dialogue_response_artifact import (
            load_learned_dialogue_response_artifact,
        )
        general_dialogue_artifact = load_learned_dialogue_response_artifact(
            optional_artifact_roots["dialogue_response"],
            require_k_drive=require_k_drive)
        domain_dialogue_artifacts = []
        for domain_root in domain_artifact_roots:
            domain_dialogue_artifacts.append(
                load_learned_dialogue_response_artifact(
                    domain_root, require_k_drive=require_k_drive))
        if routing_model is not None:
            if (routing_model.general_course_sha256
                    != general_dialogue_artifact.model.course_sha256):
                raise PublicModelReleaseError(
                    "expert router 与通用模型 course SHA 不一致")
            domain_course_shas = tuple(
                artifact.model.course_sha256
                for artifact in domain_dialogue_artifacts)
            if routing_model.domain_course_sha256s != domain_course_shas:
                raise PublicModelReleaseError(
                    "expert router 与领域模型 course SHA 不一致")
    if "response_organization" in optional_artifact_roots:
        from pure_integer_ai.experiments.build_response_organization_artifact import (
            load_response_organization_artifact,
        )
        load_response_organization_artifact(
            optional_artifact_roots["response_organization"],
            require_k_drive=require_k_drive)
    if "science_passage" in optional_artifact_roots:
        from pure_integer_ai.experiments.scidb_csq_passage_index import (
            ScidbCsqPassageRuntime,
        )
        with ScidbCsqPassageRuntime(
                optional_artifact_roots["science_passage"],
                require_k_drive=require_k_drive):
            pass
    manifest_path = training / "dialogue_pack_manifest.json"
    summary_path = training / "training_summary.json"
    cursor_path = training / "training_cursor.int"
    database_path = training / "training.sqlite3"
    for path in (manifest_path, summary_path, cursor_path, database_path):
        if not path.is_file():
            raise PublicModelReleaseError(f"training run 缺少 {path.name}")
    training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(training_manifest, dict):
        raise PublicModelReleaseError("training pack manifest 非法")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise PublicModelReleaseError("training summary 非法")
    source_rows = training_manifest.get("source_files")
    if not isinstance(source_rows, list) or not source_rows:
        raise PublicModelReleaseError("training pack manifest 缺少 source_files")
    source_files: list[tuple[Path, str]] = []
    inherited_files: list[tuple[Path, str]] = []
    source_seen: set[Path] = set()
    source_counts: dict[str, int] = {}
    for row in source_rows:
        if not isinstance(row, list) or len(row) != 3:
            raise PublicModelReleaseError("training source_files 记录非法")
        source = _source_path(project, training, row[0])
        if source not in source_seen:
            source_seen.add(source)
            source_files.append((source, str(row[0])))
        source_counts[str(row[0])] = int(row[2])
    extra_paths = training_manifest.get("extra_course_paths", ())
    if not isinstance(extra_paths, list):
        raise PublicModelReleaseError("extra_course_paths 非法")
    for value in extra_paths:
        source = _source_path(project, training, value)
        if source not in source_seen:
            source_seen.add(source)
            source_files.append((source, str(value)))
    # A resumed training run's graph contains the base run state, so its
    # public release must carry every ancestor course identity as well.  A
    # resumed run may itself resume another run (for example v13 -> portable
    # CAUSES -> Wikipedia shard); stopping after one hop produces a release
    # that validates structurally but cannot rebuild the training pack.
    # A validated release has already flattened every ancestor course into its
    # closed data inventory.  Its historical resume_from is provenance, not a
    # live host path and must never be followed outside that release root.
    materialized_source_closure = _has_materialized_training_source_closure(
        training, training_manifest, summary)
    resume_from = (None if source_release is not None
                   or materialized_source_closure else
                   summary.get("resume_from")
                   if isinstance(summary, dict) else None)
    visited_runs: set[Path] = {training}
    while isinstance(resume_from, str) and resume_from.strip():
        base_dir = (training.parent / resume_from).resolve()
        if base_dir in visited_runs:
            raise PublicModelReleaseError("resume lineage 出现循环")
        visited_runs.add(base_dir)
        base_manifest_path = base_dir / "dialogue_pack_manifest.json"
        base_summary_path = base_dir / "training_summary.json"
        if not base_manifest_path.is_file() or not base_summary_path.is_file():
            raise PublicModelReleaseError("resume 基座课程 manifest/summary 缺失")
        try:
            base_manifest = json.loads(
                base_manifest_path.read_text(encoding="utf-8"))
            base_summary = json.loads(
                base_summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicModelReleaseError(
                "resume 基座课程 manifest/summary 不可回读") from error
        if not isinstance(base_manifest, dict) or not isinstance(base_summary, dict):
            raise PublicModelReleaseError("resume 基座课程 manifest/summary 非法")
        for row in base_manifest.get("source_files", ()):
            if not isinstance(row, list) or len(row) != 3:
                raise PublicModelReleaseError("resume 基座 source_files 记录非法")
            source = _source_path(project, base_dir, row[0])
            if source not in source_seen:
                source_seen.add(source)
                inherited_files.append((source, str(row[0])))
        for value in base_manifest.get("extra_course_paths", ()):
            if not isinstance(value, str):
                raise PublicModelReleaseError("resume 基座 extra_course_paths 非法")
            source = _source_path(project, base_dir, value)
            if source not in source_seen:
                source_seen.add(source)
                inherited_files.append((source, value))
        resume_from = base_summary.get("resume_from")
    evidence_files: list[Path] = []
    for row in training_manifest.get("surface_evidence_files", ()):
        if not isinstance(row, list) or not row:
            raise PublicModelReleaseError("surface_evidence_files 记录非法")
        evidence_files.append(_source_path(project, training, row[0]))
    # DLG-RAW-16 is a runtime surface consumer dependency, not necessarily a
    # training source (shard-only runs intentionally use --no-default-courses).
    # Keep it outside training_manifest.source_files so pack identity remains
    # bound to the trained shard while the release remains self-contained.
    surface_course_source = (
        project / "data" / "ph2"
        / "dlg_raw16_surface_organization_v1.jsonl.sample")
    surface_evidence_source = (
        project / "data" / "ph2"
        / "dlg_raw16_surface_slot_evidence_v1.jsonl.sample")
    if not surface_course_source.is_file() or not surface_evidence_source.is_file():
        raise PublicModelReleaseError("DLG-RAW-16 公开课程或 evidence 缺失")
    # Include the exact 14 public snapshot parents and the snapshot itself.
    sparse_snapshot_source = project / "data/ph2/sparse_qa_runtime_snapshot_v1.json"
    if not sparse_snapshot_source.is_file():
        raise PublicModelReleaseError("窄域公开 snapshot 缺失")
    sparse_sources = [project / relative for relative in PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS]
    if any(not item.is_file() for item in sparse_sources):
        raise PublicModelReleaseError("窄域公开 snapshot parent 缺失")
    target.mkdir(parents=True)
    _copy_file(qa, target / "knowledge/broad_qa.sqlite3")
    _copy_file(database_path, target / "model/training.sqlite3")
    _copy_file(cursor_path, target / "model/training_cursor.int")
    # summary paths are resolved relative to entry.training_root (model/).
    summary["database"] = "training.sqlite3"
    summary["training_cursor"] = "training_cursor.int"
    (target / "model").mkdir(parents=True, exist_ok=True)
    (target / "model/training_summary.json").write_bytes(_canonical_json(summary))
    copied_courses: list[Path] = []
    copied_course_sidecars: list[Path] = []
    seen_names: set[str] = set()
    copied_sidecar_digests: dict[str, str] = {}
    rewritten_rows = []
    rewritten_identities = []
    for source, original in source_files:
        name = source.name
        if name in seen_names:
            raise PublicModelReleaseError(f"训练来源 basename 冲突: {name}")
        seen_names.add(name)
        relative = Path("data/ph2") / name
        _copy_file(source, target / relative)
        copied_courses.append(source)
        for sidecar in _course_sidecars(source):
            sidecar_relative = Path("data/ph2") / sidecar.name
            if sidecar.name in seen_names:
                digest = _sha256(sidecar)
                if copied_sidecar_digests.get(sidecar.name) != digest:
                    raise PublicModelReleaseError(
                        f"训练 sidecar basename 冲突: {sidecar.name}")
                # Multiple shards intentionally share one immutable sidecar;
                # publish it once and keep a single manifest entry.
                continue
            _copy_file(sidecar, target / sidecar_relative)
            copied_course_sidecars.append(sidecar)
            seen_names.add(sidecar.name)
            copied_sidecar_digests[sidecar.name] = _sha256(sidecar)
        digest = _sha256(source)
        count = source_counts.get(original, 0)
        rewritten_rows.append([relative.as_posix(), digest, count])
        rewritten_identities.append([relative.as_posix(), original])
    # Base-run courses are copied for release-wide held-out inspection and
    # citation provenance, but remain outside the resumed run's pack identity.
    for source, _original in inherited_files:
        name = source.name
        if name in seen_names:
            continue
        seen_names.add(name)
        relative = Path("data/ph2") / name
        _copy_file(source, target / relative)
        copied_courses.append(source)
        for sidecar in _course_sidecars(source):
            sidecar_relative = Path("data/ph2") / sidecar.name
            if sidecar.name in seen_names:
                digest = _sha256(sidecar)
                if copied_sidecar_digests.get(sidecar.name) != digest:
                    raise PublicModelReleaseError(
                        f"训练 sidecar basename 冲突: {sidecar.name}")
                continue
            _copy_file(sidecar, target / sidecar_relative)
            copied_course_sidecars.append(sidecar)
            seen_names.add(sidecar.name)
            copied_sidecar_digests[sidecar.name] = _sha256(sidecar)
    for source in evidence_files:
        if source.name not in seen_names:
            _copy_file(source, target / "data/ph2" / source.name)
            seen_names.add(source.name)
    for source in (surface_course_source, surface_evidence_source):
        if source.name not in seen_names:
            _copy_file(source, target / "data/ph2" / source.name)
            seen_names.add(source.name)
    for source in (sparse_snapshot_source, *sparse_sources):
        relative = Path("data/ph2") / source.name
        if source.name not in seen_names:
            _copy_file(source, target / relative)
            seen_names.add(source.name)
    if routing_source is not None:
        _copy_file(routing_source, target / PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE)
        seen_names.add(PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE)
    copied_dialogue_source_manifests = []
    for ordinal, source in enumerate(dialogue_source_manifests):
        relative = (
            f"{PUBLIC_DIALOGUE_SOURCE_MANIFEST_ROOT}/{ordinal:04d}.json")
        _copy_file(source, target / relative)
        copied_dialogue_source_manifests.append(relative)
    rewritten_manifest = dict(training_manifest)
    rewritten_manifest["source_files"] = rewritten_rows
    rewritten_manifest["source_identities"] = rewritten_identities
    rewritten_manifest["extra_course_paths"] = [
        (Path("data/ph2") / _source_path(project, training, value).name).as_posix()
        for value in extra_paths
    ]
    rewritten_manifest["surface_evidence_files"] = [
        [(Path("data/ph2") / item.name).as_posix(), _sha256(item)]
        for item in evidence_files
    ]
    (target / "model/dialogue_pack_manifest.json").write_bytes(
        _canonical_json(rewritten_manifest))
    embedded_artifacts: dict[str, dict[str, object]] = {}
    if "dialogue_response" in optional_artifact_roots:
        manifest = _copy_closed_artifact(
            optional_artifact_roots["dialogue_response"],
            target / PUBLIC_DIALOGUE_RESPONSE_ARTIFACT,
            manifest_name=_DIALOGUE_RESPONSE_MANIFEST,
            label="dialogue response artifact")
        embedded_artifacts["dialogue_response"] = {
            "artifact_kind": manifest.get("artifact_kind"),
            "license_id": manifest.get("license_id"),
            "manifest_sha256": _sha256(
                target / PUBLIC_DIALOGUE_RESPONSE_ARTIFACT
                / _DIALOGUE_RESPONSE_MANIFEST),
            "path": PUBLIC_DIALOGUE_RESPONSE_ARTIFACT,
        }
        if domain_artifact_roots:
            domain_entries = []
            for ordinal, domain_root in enumerate(domain_artifact_roots):
                relative = (
                    f"{PUBLIC_DIALOGUE_DOMAIN_EXPERT_ROOT}/{ordinal:04d}")
                manifest = _copy_closed_artifact(
                    domain_root, target / relative,
                    manifest_name=_DIALOGUE_RESPONSE_MANIFEST,
                    label=f"dialogue domain expert artifact {ordinal}")
                domain_entries.append({
                    "artifact_kind": manifest.get("artifact_kind"),
                    "license_id": manifest.get("license_id"),
                    "manifest_sha256": _sha256(
                        target / relative / _DIALOGUE_RESPONSE_MANIFEST),
                    "path": relative,
                })
            embedded_artifacts["dialogue_domain_experts"] = domain_entries
    if "response_organization" in optional_artifact_roots:
        manifest = _copy_closed_artifact(
            optional_artifact_roots["response_organization"],
            target / PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT,
            manifest_name=_RESPONSE_ORGANIZATION_MANIFEST,
            label="response organization artifact")
        embedded_artifacts["response_organization"] = {
            "artifact_kind": manifest.get("artifact_kind"),
            "license_id": manifest.get("license_id"),
            "manifest_sha256": _sha256(
                target / PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT
                / _RESPONSE_ORGANIZATION_MANIFEST),
            "path": PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT,
        }
    if "science_passage" in optional_artifact_roots:
        manifest = _copy_closed_artifact(
            optional_artifact_roots["science_passage"],
            target / PUBLIC_SCIENCE_PASSAGE_ARTIFACT,
            manifest_name=_SCIENCE_PASSAGE_MANIFEST,
            label="science passage artifact")
        embedded_artifacts["science_passage"] = {
            "artifact_kind": manifest.get("artifact_kind"),
            "attribution": manifest.get("attribution"),
            "license_id": manifest.get("license_id"),
            "manifest_sha256": _sha256(
                target / PUBLIC_SCIENCE_PASSAGE_ARTIFACT
                / _SCIENCE_PASSAGE_MANIFEST),
            "path": PUBLIC_SCIENCE_PASSAGE_ARTIFACT,
        }
    source_manifest = {
        "format": "PUBLIC_SOURCE_MANIFEST_V1",
        "schema_version": 1,
        "qa_index": {
            "path": "knowledge/broad_qa.sqlite3",
            "license_id": _qa_metadata(qa).get("license_id", "UNKNOWN"),
            "metadata": _qa_metadata(qa),
        },
        "training": {
            "pack_manifest": "model/dialogue_pack_manifest.json",
            "license_ids": list(_license_ids(copied_courses)),
            "course_files": [
                (Path("data/ph2") / item.name).as_posix()
                for item in copied_courses
            ],
            "course_sidecars": [
                (Path("data/ph2") / item.name).as_posix()
                for item in copied_course_sidecars
            ],
            "surface_courses": [
                (Path("data/ph2") / surface_course_source.name).as_posix(),
                (Path("data/ph2") / surface_evidence_source.name).as_posix(),
            ],
        },
        "sparse_runtime": {
            "snapshot": "data/ph2/sparse_qa_runtime_snapshot_v1.json",
            "license_id": "CC0-1.0",
            "source_files": list(PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS),
        },
        "artifacts": embedded_artifacts,
        "dialogue_source_manifests": copied_dialogue_source_manifests,
        **({"dialogue_expert_routing": {
            "path": PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE,
            "sha256": _sha256(target / PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE),
        }} if routing_source is not None else {}),
    }
    (target / "source_manifest.json").write_bytes(_canonical_json(source_manifest))
    protocol_config = {
        "format": "PURE_INTEGER_AI_DIALOGUE_PROTOCOL_CONFIG",
        "schema_version": 1,
        "transport": "jsonl",
        "encoding": "utf-8",
        "byte_order": "big",
        "operations": ["turn", "quit", "exit"],
        "request": {"required": ["op", "text"], "id_optional": True},
        # Internal ANSWER/UNKNOWN/CLARIFY/REPAIR state is deliberately absent
        # from the public wire contract.  It remains available in the
        # in-process DialogueTurn used by evaluation and checkpointing.
        "response": {
            "type": "response",
            "text_field": "text",
            "internal_status_field": None,
        },
        "checkpoint": {"optional": True, "integer_only": True},
    }
    (target / PUBLIC_DIALOGUE_PROTOCOL_CONFIG).write_bytes(
        _canonical_json(protocol_config))
    payload_files = sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.name not in {
            PUBLIC_MODEL_RELEASE_MANIFEST, PUBLIC_MODEL_RELEASE_DIGEST,
        }
    )
    entries = []
    for path in payload_files:
        relative = path.relative_to(target).as_posix()
        role = ("knowledge_index" if relative.startswith("knowledge/") else
                "training_state" if relative.startswith("model/") else
                "runtime_resource" if relative in {
                    "data/ph2/sparse_qa_runtime_snapshot_v1.json",
                    *PUBLIC_SPARSE_QA_SOURCE_ARTIFACTS,
                } else "training_course")
        entries.append({"path": relative, "role": role,
                        "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    release_manifest = {
        "format": PUBLIC_MODEL_RELEASE_FORMAT,
        "schema_version": PUBLIC_MODEL_RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "source_manifest": "source_manifest.json",
        "entry": {
            "qa_database": "knowledge/broad_qa.sqlite3",
            "training_root": "model",
            "sparse_snapshot": "data/ph2/sparse_qa_runtime_snapshot_v1.json",
            "protocol_config": PUBLIC_DIALOGUE_PROTOCOL_CONFIG,
            "protocol": "jsonl",
            **({"dialogue_response_artifact":
                PUBLIC_DIALOGUE_RESPONSE_ARTIFACT}
               if "dialogue_response" in optional_artifact_roots else {}),
            **({"dialogue_domain_expert_artifacts": [
                    f"{PUBLIC_DIALOGUE_DOMAIN_EXPERT_ROOT}/{ordinal:04d}"
                    for ordinal in range(len(domain_artifact_roots))]}
               if domain_artifact_roots else {}),
            **({"dialogue_expert_routing_artifact":
                PUBLIC_DIALOGUE_EXPERT_ROUTING_FILE}
               if routing_source is not None else {}),
            **({"response_organization_artifact":
                PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT}
               if "response_organization" in optional_artifact_roots else {}),
            **({"science_passage_artifact":
                PUBLIC_SCIENCE_PASSAGE_ARTIFACT}
               if "science_passage" in optional_artifact_roots else {}),
        },
        "files": entries,
    }
    manifest_path = target / PUBLIC_MODEL_RELEASE_MANIFEST
    manifest_path.write_bytes(_canonical_json(release_manifest))
    (target / PUBLIC_MODEL_RELEASE_DIGEST).write_text(
        _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
    return load_public_model_release(target, require_k_drive=require_k_drive)


def main(argv: list[str] | None = None) -> int:
    """构造公开 release root 的命令行包装。"""
    import argparse
    parser = argparse.ArgumentParser(description="build self-contained public model release")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--qa-database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--dialogue-response-artifact-root", default=None)
    parser.add_argument(
        "--dialogue-domain-expert-artifact-root", action="append", default=[])
    parser.add_argument("--dialogue-expert-routing-artifact-root", default=None)
    parser.add_argument(
        "--dialogue-source-manifest", action="append", default=[])
    parser.add_argument("--response-organization-artifact-root", default=None)
    parser.add_argument("--science-passage-artifact-root", default=None)
    args = parser.parse_args(argv)
    release = build_public_model_release(
        project_root=args.project_root,
        qa_database=args.qa_database,
        training_run_root=args.training_run_root,
        release_root=args.release_root,
        release_id=args.release_id,
        dialogue_response_artifact_root=args.dialogue_response_artifact_root,
        dialogue_domain_expert_artifact_roots=tuple(
            args.dialogue_domain_expert_artifact_root),
        dialogue_expert_routing_artifact_root=(
            args.dialogue_expert_routing_artifact_root),
        dialogue_source_manifest_paths=tuple(args.dialogue_source_manifest),
        response_organization_artifact_root=(
            args.response_organization_artifact_root),
        science_passage_artifact_root=args.science_passage_artifact_root,
    )
    print(json.dumps({
        "release_id": release.release_id,
        "root": str(release.root),
        "qa_database": release.qa_database.relative_to(release.root).as_posix(),
        "training_root": release.training_root.relative_to(release.root).as_posix(),
        "sparse_snapshot": release.sparse_snapshot.relative_to(release.root).as_posix(),
        "dialogue_response_artifact": (
            release.dialogue_response_artifact.relative_to(release.root).as_posix()
            if release.dialogue_response_artifact is not None else None),
        "dialogue_domain_expert_artifacts": [
            path.relative_to(release.root).as_posix()
            for path in release.dialogue_domain_expert_artifacts
        ],
        "dialogue_expert_routing_artifact": (
            release.dialogue_expert_routing_artifact.relative_to(release.root).as_posix()
            if release.dialogue_expert_routing_artifact is not None else None),
        "response_organization_artifact": (
            release.response_organization_artifact.relative_to(release.root).as_posix()
            if release.response_organization_artifact is not None else None),
        "science_passage_artifact": (
            release.science_passage_artifact.relative_to(release.root).as_posix()
            if release.science_passage_artifact is not None else None),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "PUBLIC_MODEL_RELEASE_DIGEST", "PUBLIC_MODEL_RELEASE_FORMAT",
    "PUBLIC_MODEL_RELEASE_MANIFEST", "PUBLIC_MODEL_RELEASE_SCHEMA_VERSION",
    "PUBLIC_DIALOGUE_PROTOCOL_CONFIG", "PUBLIC_DIALOGUE_RESPONSE_ARTIFACT",
    "PUBLIC_RESPONSE_ORGANIZATION_ARTIFACT",
    "PUBLIC_SCIENCE_PASSAGE_ARTIFACT",
    "PublicModelRelease", "PublicModelReleaseError",
    "build_public_model_release", "load_public_model_release", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
