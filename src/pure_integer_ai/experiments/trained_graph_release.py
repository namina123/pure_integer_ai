"""构建并验证不携带课程或外部 QA 的训练图发布根。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil


TRAINED_GRAPH_RELEASE_FORMAT = "PURE_INTEGER_TRAINED_GRAPH_RELEASE_V1"
TRAINED_GRAPH_RELEASE_MANIFEST = "trained_graph_release.json"
TRAINED_GRAPH_RELEASE_DIGEST = "trained_graph_release.sha256"
_REQUIRED_TRAINING_FILES = (
    "training.sqlite3",
    "training_cursor.int",
    "training_summary.json",
    "sqlite_resume_manifest.json",
    "dialogue_pack_manifest.json",
)
_STATE_FIELDS = (
    "run_id",
    "pack_sha256",
    "source_namespace",
    "active_stages",
    "stages_completed",
    "cumulative_stages_completed",
    "campaign_required_stages",
    "case_count",
    "training_item_count",
    "source_record_count",
    "occurrence_count",
    "occurrence_order_fact_count",
    "dialogue_successor_count",
    "dialogue_successor_feature_count",
    "split_counts",
    "typed_course",
    "typed_language_floor",
    "stage_weaning_ready",
    "weaning_ready",
    "weaning_blockers",
)


class TrainedGraphReleaseError(ValueError):
    """训练图发布根缺失、漂移或包含不允许的宿主状态。"""


@dataclass(frozen=True, slots=True)
class TrainedGraphRelease:
    """已验证发布根的路径结构体。"""

    root: Path
    release_id: str
    training_database: Path
    training_cursor: Path
    fallback_surfaces: Path
    source_manifest: Path
    protocol_config: Path
    manifest: dict[str, object]


def _canonical_json(value: object) -> bytes:
    """返回跨语言可复现的 UTF-8 JSON bytes。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，不把训练 SQLite 整体载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    """读取一个严格 JSON object。"""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainedGraphReleaseError(f"{label} 不可回读") from error
    if not isinstance(value, dict):
        raise TrainedGraphReleaseError(f"{label} 必须是 JSON object")
    return value


def _require_relative(value: object, *, label: str) -> Path:
    """校验发布 manifest 中的正向相对路径。"""
    if type(value) is not str or not value or "\\" in value:
        raise TrainedGraphReleaseError(f"{label} 必须是正向相对路径")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainedGraphReleaseError(f"{label} 越出 release root")
    return path


def _resolve_source(project: Path, training: Path, value: object) -> Path:
    """仅在构建期解析训练来源；返回路径不会写入发布 manifest。"""
    if type(value) is not str or not value:
        raise TrainedGraphReleaseError("训练来源路径非法")
    source = Path(value)
    candidates = (
        (source,) if source.is_absolute()
        else (training / source, project / source)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise TrainedGraphReleaseError(f"训练来源不可回读: {source.name}")


def _source_license(
        path: Path, *, project: Path, declared_source: str,
        ) -> tuple[tuple[str, ...], dict[str, object]]:
    """从记录或同目录 authored 声明恢复许可及其可核验证据。"""
    values = set()

    def collect(value: object) -> None:
        """递归收集结构化来源中任意层级的显式 license_id。"""
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "license_id" and type(item) is str and item.strip():
                    values.add(item.strip())
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    try:
        with path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                row = json.loads(raw.decode("utf-8"))
                collect(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainedGraphReleaseError(
            f"训练来源许可不可回读: {path.name}") from error
    if values:
        return tuple(sorted(values)), {"kind": "EMBEDDED_RECORD_FIELD"}
    declarations = [path.parent / "DATA_LICENSE.md"]
    declared_path = Path(declared_source)
    if not declared_path.is_absolute() and ".." not in declared_path.parts:
        project_declaration = (
            project / declared_path.parent / "DATA_LICENSE.md").resolve()
        try:
            project_declaration.relative_to(project)
        except ValueError:
            pass
        else:
            declarations.append(project_declaration)
    for declaration in declarations:
        try:
            declaration_text = declaration.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        authored_declaration = (
            "Files in this directory whose source key is `AUTHORED_CC0_V1`"
            in declaration_text
            and "This declaration applies only to the authored data samples"
            in declaration_text
            and "SPDX-License-Identifier: CC0-1.0" in declaration_text
        )
        if path.name.startswith("authored_") and authored_declaration:
            return ("CC0-1.0",), {
                "kind": "DIRECTORY_AUTHORED_DECLARATION",
                "name": declaration.name,
                "sha256": _sha256(declaration),
            }
    raise TrainedGraphReleaseError(f"训练来源缺少许可: {path.name}")


def _contains_text(value: object, target: str) -> bool:
    """在结构化训练记录中查找完全相同的表层值。"""
    if value == target:
        return True
    if isinstance(value, dict):
        return any(_contains_text(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, target) for item in value)
    return False


def _verify_fallback_source(
        path: Path, surfaces: tuple[str, ...]) -> tuple[str, ...]:
    """核验每条边界表层都来自指定训练记录，而不是发布期手写。"""
    missing = set(surfaces)
    try:
        with path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                value = json.loads(raw.decode("utf-8"))
                missing = {
                    surface for surface in missing
                    if not _contains_text(value, surface)
                }
                if not missing:
                    break
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainedGraphReleaseError("边界表层训练来源不可回读") from error
    if missing:
        raise TrainedGraphReleaseError(
            f"边界表层未见于训练来源: {len(missing)}")
    return surfaces


def _source_ledger(
        project: Path,
        lineage: tuple[
            tuple[Path, dict[str, object], dict[str, object]], ...],
        ) -> dict[str, object]:
    """把课程 manifest 压缩为无正文、无本机路径的来源/许可账本。"""
    sources = []
    seen = set()
    for training, pack, _summary in lineage:
        rows = pack.get("source_files")
        if not isinstance(rows, list) or not rows:
            raise TrainedGraphReleaseError("训练 pack 缺少 source_files")
        for ordinal, row in enumerate(rows):
            if (not isinstance(row, list) or len(row) != 3
                    or type(row[0]) is not str
                    or type(row[1]) is not str or len(row[1]) != 64
                    or type(row[2]) is not int or row[2] < 0):
                raise TrainedGraphReleaseError(
                    f"source_files[{ordinal}] 非规范")
            source = _resolve_source(project, training, row[0])
            digest = _sha256(source)
            if digest != row[1]:
                raise TrainedGraphReleaseError(
                    f"训练来源 SHA 漂移: {source.name}")
            identity = (source.name, digest)
            if identity in seen:
                continue
            seen.add(identity)
            license_ids, license_evidence = _source_license(
                source, project=project, declared_source=row[0])
            sources.append({
                "name": source.name,
                "sha256": digest,
                "record_count": row[2],
                "license_ids": list(license_ids),
                "license_evidence": license_evidence,
            })
    current_pack = lineage[0][1]
    return {
        "format": "PURE_INTEGER_TRAINED_GRAPH_SOURCE_LEDGER_V1",
        "schema_version": 1,
        "pack_sha256": current_pack.get("pack_sha256"),
        "source_namespace": current_pack.get("source_namespace"),
        "lineage": [
            {
                "run_id": summary.get("run_id"),
                "pack_sha256": pack.get("pack_sha256"),
            }
            for _root, pack, summary in lineage
        ],
        "sources": sorted(
            sources, key=lambda item: (item["name"], item["sha256"])),
    }


def _training_lineage(
        training: Path,
        pack: dict[str, object],
        summary: dict[str, object],
        ) -> tuple[tuple[Path, dict[str, object], dict[str, object]], ...]:
    """沿同一 run parent 恢复全部祖先 pack，不接受路径跳转。"""
    result = [(training, pack, summary)]
    seen = {training}
    current_summary = summary
    current_pack = pack
    current_root = training
    while True:
        if _has_materialized_source_closure(
                current_root, current_pack, current_summary):
            break
        resume_from = current_summary.get("resume_from")
        if resume_from is None or resume_from == "":
            break
        if (type(resume_from) is not str
                or Path(resume_from).name != resume_from):
            raise TrainedGraphReleaseError("resume lineage 必须使用同级 run id")
        base = (training.parent / resume_from).resolve()
        try:
            base.relative_to(training.parent)
        except ValueError as error:
            raise TrainedGraphReleaseError("resume lineage 越出 campaign") from error
        if base in seen:
            raise TrainedGraphReleaseError("resume lineage 出现循环")
        seen.add(base)
        base_pack_path = base / "dialogue_pack_manifest.json"
        base_summary_path = base / "training_summary.json"
        if not base_pack_path.is_file() or not base_summary_path.is_file():
            raise TrainedGraphReleaseError(
                f"resume lineage 缺失: {resume_from}")
        base_pack = _read_object(base_pack_path, label="ancestor dialogue pack")
        current_summary = _read_object(
            base_summary_path, label="ancestor training summary")
        if current_summary.get("pack_sha256") != base_pack.get("pack_sha256"):
            raise TrainedGraphReleaseError("ancestor summary/pack 身份漂移")
        result.append((base, base_pack, current_summary))
        current_root = base
        current_pack = base_pack
    return tuple(result)


def _has_materialized_source_closure(
        training: Path,
        pack: dict[str, object],
        summary: dict[str, object],
        ) -> bool:
    """判断当前 run 是否已物化其训练图的完整来源闭包。"""
    if (type(summary.get("source_record_count")) is not int
            or summary["source_record_count"] <= 0
            or pack.get("train_surface_count")
            != summary["source_record_count"]):
        return False
    values: list[object] = []
    for row in pack.get("source_files", ()):
        if not isinstance(row, list) or len(row) != 3:
            return False
        values.append(row[0])
    values.extend(pack.get("extra_course_paths", ()))
    for row in pack.get("surface_evidence_files", ()):
        if not isinstance(row, list) or not row:
            return False
        values.append(row[0])
    if not values:
        return False
    for value in values:
        if type(value) is not str or not value:
            return False
        relative = PurePosixPath(value)
        if (relative.is_absolute() or ".." in relative.parts
                or any(":" in part for part in relative.parts)):
            return False
        path = (training / Path(*relative.parts)).resolve()
        try:
            path.relative_to(training)
        except ValueError:
            return False
        if not path.is_file():
            return False
    return True


def build_trained_graph_release(
        *,
        project_root: str | Path,
        training_run_root: str | Path,
        release_root: str | Path,
        release_id: str,
        fallback_surfaces: tuple[str, ...],
        fallback_surface_source: str | Path,
        require_k_drive: bool = True,
        ) -> TrainedGraphRelease:
    """构造仅含训练后图状态的闭合发布根，不复制课程或 QA。"""
    if type(release_id) is not str or not release_id.strip():
        raise TrainedGraphReleaseError("release_id 必须是非空文本")
    if (not isinstance(fallback_surfaces, tuple) or not fallback_surfaces
            or any(type(item) is not str or not item.strip()
                   or "\n" in item or "\r" in item
                   for item in fallback_surfaces)):
        raise TrainedGraphReleaseError("fallback_surfaces 必须是单行非空文本 tuple")
    project = Path(project_root).resolve()
    training = Path(training_run_root).resolve()
    target = Path(release_root).resolve()
    if not project.is_dir() or not training.is_dir():
        raise TrainedGraphReleaseError("project/training root 必须存在")
    if require_k_drive and (
            training.drive.upper() != "K:" or target.drive.upper() != "K:"):
        raise TrainedGraphReleaseError("训练和发布根必须位于 K 盘")
    if target.exists():
        raise TrainedGraphReleaseError("release root 已存在，拒绝覆盖")
    required = {name: training / name for name in _REQUIRED_TRAINING_FILES}
    if any(not path.is_file() for path in required.values()):
        missing = sorted(name for name, path in required.items()
                         if not path.is_file())
        raise TrainedGraphReleaseError(f"训练 run 缺少文件: {missing}")
    summary = _read_object(required["training_summary.json"], label="training summary")
    resume = _read_object(
        required["sqlite_resume_manifest.json"], label="SQLite resume manifest")
    pack = _read_object(
        required["dialogue_pack_manifest.json"], label="dialogue pack manifest")
    database_sha = _sha256(required["training.sqlite3"])
    if (resume.get("status") != "PASS"
            or resume.get("database_sha256") != database_sha
            or summary.get("pack_sha256") != resume.get("pack_sha256")
            or summary.get("pack_sha256") != pack.get("pack_sha256")):
        raise TrainedGraphReleaseError("训练 summary/resume/pack 身份不闭合")
    state = {key: summary[key] for key in _STATE_FIELDS if key in summary}
    state.update({
        "format": "PURE_INTEGER_TRAINED_GRAPH_STATE_V1",
        "schema_version": 1,
        "database_sha256": database_sha,
        "database_bytes": required["training.sqlite3"].stat().st_size,
        "schema_sha256": resume.get("schema_sha256"),
        "table_counts": resume.get("table_counts"),
        "table_counts_sha256": resume.get("table_counts_sha256"),
    })
    from pure_integer_ai.experiments.trained_relation_graph_runtime import (
        TrainedRelationGraphRuntime,
    )
    from pure_integer_ai.experiments.dialogue_successor_graph import (
        SqliteDialogueSuccessorRuntime,
    )
    with TrainedRelationGraphRuntime(
            required["training.sqlite3"]) as relation_runtime:
        relation_count = len(relation_runtime.active_propositions())
        relation_frame_count = len(relation_runtime.active_surface_frames())
    dialogue_runtime = SqliteDialogueSuccessorRuntime(
        required["training.sqlite3"])
    try:
        dialogue_count = dialogue_runtime.count()
    finally:
        dialogue_runtime.close()
    if min(relation_count, relation_frame_count, dialogue_count) <= 0:
        raise TrainedGraphReleaseError("训练图缺少 relation/dialogue 承重状态")
    state["runtime_capability_counts"] = {
        "active_relation_propositions": relation_count,
        "relation_surface_frames": relation_frame_count,
        "dialogue_successor_projections": dialogue_count,
    }
    source_ledger = _source_ledger(
        project, _training_lineage(training, pack, summary))
    fallback_source = Path(fallback_surface_source).resolve()
    if not fallback_source.is_file():
        raise TrainedGraphReleaseError("边界表层训练来源不存在")
    fallback_digest = _sha256(fallback_source)
    source_digests = {
        item["sha256"] for item in source_ledger["sources"]
        if isinstance(item, dict) and type(item.get("sha256")) is str
    }
    if fallback_digest not in source_digests:
        raise TrainedGraphReleaseError("边界表层来源未进入训练 lineage")
    _verify_fallback_source(fallback_source, fallback_surfaces)
    source_ledger["fallback_surface_source"] = {
        "name": fallback_source.name,
        "sha256": fallback_digest,
    }
    protocol = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_DIALOGUE_PROTOCOL_V1",
        "schema_version": 1,
        "transport": "jsonl",
        "encoding": "utf-8",
        "operations": ["turn", "quit", "exit"],
        "request": {"required": ["op", "text"], "id_optional": True},
        "response": {"type": "turn", "text_field": "text"},
        "memory": {"optional": True, "storage": "sqlite", "integer_graph": True},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise TrainedGraphReleaseError("release staging 已存在，拒绝覆盖")
    (staging / "model").mkdir(parents=True)
    shutil.copyfile(required["training.sqlite3"], staging / "model/training.sqlite3")
    shutil.copyfile(
        required["training_cursor.int"], staging / "model/training_cursor.int")
    (staging / "model/training_state.json").write_bytes(_canonical_json(state))
    (staging / "model/fallback_surfaces.txt").write_text(
        "\n".join(fallback_surfaces) + "\n", encoding="utf-8", newline="\n")
    (staging / "source_manifest.json").write_bytes(
        _canonical_json(source_ledger))
    (staging / "dialogue_protocol.json").write_bytes(_canonical_json(protocol))
    payloads = tuple(sorted(
        path for path in staging.rglob("*") if path.is_file()))
    files = [{
        "path": path.relative_to(staging).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in payloads]
    manifest = {
        "format": TRAINED_GRAPH_RELEASE_FORMAT,
        "schema_version": 1,
        "release_id": release_id.strip(),
        "entry": {
            "training_database": "model/training.sqlite3",
            "training_cursor": "model/training_cursor.int",
            "training_state": "model/training_state.json",
            "fallback_surfaces": "model/fallback_surfaces.txt",
            "source_manifest": "source_manifest.json",
            "protocol_config": "dialogue_protocol.json",
        },
        "files": files,
    }
    manifest_path = staging / TRAINED_GRAPH_RELEASE_MANIFEST
    manifest_path.write_bytes(_canonical_json(manifest))
    (staging / TRAINED_GRAPH_RELEASE_DIGEST).write_text(
        _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
    staging.rename(target)
    return load_trained_graph_release(
        target, require_k_drive=require_k_drive)


def load_trained_graph_release(
        root: str | Path,
        *,
        require_k_drive: bool = False,
        verify_payload_hashes: bool = True,
        ) -> TrainedGraphRelease:
    """验证闭合文件集合、逐文件 SHA 与无课程/QA边界。"""
    if type(require_k_drive) is not bool or type(verify_payload_hashes) is not bool:
        raise TypeError("release 校验开关必须是严格 bool")
    target = Path(root).resolve()
    if not target.is_dir() or require_k_drive and target.drive.upper() != "K:":
        raise TrainedGraphReleaseError("trained graph release root 非法")
    manifest_path = target / TRAINED_GRAPH_RELEASE_MANIFEST
    digest_path = target / TRAINED_GRAPH_RELEASE_DIGEST
    if not manifest_path.is_file() or not digest_path.is_file():
        raise TrainedGraphReleaseError("trained graph release 缺少根 manifest")
    if digest_path.read_text(encoding="ascii").strip() != _sha256(manifest_path):
        raise TrainedGraphReleaseError("trained graph release manifest SHA 漂移")
    manifest = _read_object(manifest_path, label="trained graph release manifest")
    if (manifest.get("format") != TRAINED_GRAPH_RELEASE_FORMAT
            or manifest.get("schema_version") != 1
            or type(manifest.get("release_id")) is not str):
        raise TrainedGraphReleaseError("trained graph release 格式不兼容")
    entry = manifest.get("entry")
    rows = manifest.get("files")
    if not isinstance(entry, dict) or not isinstance(rows, list) or not rows:
        raise TrainedGraphReleaseError("trained graph release inventory 非法")
    declared = {TRAINED_GRAPH_RELEASE_MANIFEST, TRAINED_GRAPH_RELEASE_DIGEST}
    for ordinal, row in enumerate(rows):
        if (not isinstance(row, dict)
                or type(row.get("size_bytes")) is not int
                or row["size_bytes"] < 0
                or type(row.get("sha256")) is not str
                or len(row["sha256"]) != 64):
            raise TrainedGraphReleaseError(f"files[{ordinal}] 非规范")
        relative = _require_relative(row.get("path"), label=f"files[{ordinal}].path")
        path = (target / relative).resolve()
        try:
            path.relative_to(target)
        except ValueError as error:
            raise TrainedGraphReleaseError("release payload 越界") from error
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != row["size_bytes"]
                or verify_payload_hashes and _sha256(path) != row["sha256"]):
            raise TrainedGraphReleaseError(f"release payload 漂移: {relative}")
        if path.suffix in {".json", ".txt"}:
            raw = path.read_bytes()
            if any(marker in raw for marker in (
                    b"D:\\", b"D:/", b"K:\\", b"K:/")):
                raise TrainedGraphReleaseError(
                    f"release payload 含本机绝对路径: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*") if path.is_file()
    }
    if actual != declared or any(path.is_symlink() for path in target.rglob("*")):
        raise TrainedGraphReleaseError("trained graph release 文件集合不闭合")
    if any(path.startswith("data/") or "course" in path.lower()
           or "qa" in Path(path).name.lower() for path in actual):
        raise TrainedGraphReleaseError("trained graph release 不得携带课程或 QA")
    raw_manifest = manifest_path.read_bytes()
    if any(marker in raw_manifest for marker in (
            b"D:\\", b"D:/", b"K:\\", b"K:/")):
        raise TrainedGraphReleaseError("release manifest 含本机绝对路径")

    def entry_path(field: str) -> Path:
        relative = _require_relative(entry.get(field), label=f"entry.{field}")
        path = (target / relative).resolve()
        if not path.is_file():
            raise TrainedGraphReleaseError(f"release entry 缺失: {field}")
        return path

    return TrainedGraphRelease(
        target,
        manifest["release_id"],
        entry_path("training_database"),
        entry_path("training_cursor"),
        entry_path("fallback_surfaces"),
        entry_path("source_manifest"),
        entry_path("protocol_config"),
        manifest,
    )


__all__ = [
    "TRAINED_GRAPH_RELEASE_DIGEST",
    "TRAINED_GRAPH_RELEASE_FORMAT",
    "TRAINED_GRAPH_RELEASE_MANIFEST",
    "TrainedGraphRelease",
    "TrainedGraphReleaseError",
    "build_trained_graph_release",
    "load_trained_graph_release",
]
