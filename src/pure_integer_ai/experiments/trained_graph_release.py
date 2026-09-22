"""构建并验证不携带课程或外部 QA 的训练图发布根。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
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
    "typed_relation_generation",
    "grounded_answer_graph_training",
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
            # A materialization child intentionally carries the parent's pack
            # manifest but no course files.  Resolve each declared source
            # against every lineage root, newest first, before giving up.
            source = None
            for candidate_training, _candidate_pack, _candidate_summary in lineage:
                try:
                    source = _resolve_source(project, candidate_training, row[0])
                except TrainedGraphReleaseError:
                    continue
                break
            if source is None:
                raise TrainedGraphReleaseError(
                    f"训练来源不可回读: {Path(row[0]).name}")
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
    # R-01 materialization runs deliberately copy only the pack manifest and
    # SQLite checkpoint; course files remain owned by the parent training run.
    # Treating the copied manifest as a closed source set makes release
    # construction fail on otherwise valid additive materialization runs.
    if (training / "trained_relation_generation_materialization.json").is_file():
        return False
    if (type(summary.get("source_record_count")) is not int
            or summary["source_record_count"] <= 0
            or type(pack.get("train_surface_count")) is not int
            or pack["train_surface_count"] <= 0):
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
        require_k_drive: bool = True,
        require_dialogue_successor: bool = True,
        qualification_audit: str | Path | None = None,
        ) -> TrainedGraphRelease:
    """构造仅含训练后图状态的闭合发布根，不复制课程或 QA。

    关系生成增量允许先发布明确标记为非自由对话完成的模型；这只放宽
    successor 数量门，不会伪造 successor、回放正文或改变能力声明。
    """
    if (type(require_k_drive) is not bool
            or type(require_dialogue_successor) is not bool):
        raise TypeError("发布开关必须是严格 bool")
    if type(release_id) is not str or not release_id.strip():
        raise TrainedGraphReleaseError("release_id 必须是非空文本")
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
    qualification_path = None
    if qualification_audit is not None:
        qualification_path = Path(qualification_audit).resolve(strict=True)
        qualification = _read_object(
            qualification_path, label="free-dialogue qualification audit")
        if (qualification.get("format")
                != "PURE_INTEGER_FREE_DIALOGUE_QUALIFICATION_AUDIT_V1"
                or qualification.get("qualification_status")
                != "PASS_HELDOUT_CONSUMPTION_ONLY"
                or qualification.get("model_sha256") != database_sha
                or qualification.get("free_dialogue_complete") != 0
                or qualification.get("independent_release_required") != 1):
            raise TrainedGraphReleaseError(
                "free-dialogue qualification audit 与训练图身份/资格不闭合")
    relation_generation = summary.get("typed_relation_generation")
    if relation_generation is not None:
        materialization_path = training / (
            "trained_relation_generation_materialization.json")
        materialization = _read_object(
            materialization_path,
            label="relation generation materialization",
        )
        if (not isinstance(relation_generation, dict)
                or materialization.get("format")
                != "TRAINED_RELATION_GENERATION_MATERIALIZATION_V1"
                or materialization.get("schema_version") != 1
                or materialization.get("run_id") != training.name
                or materialization.get("database_sha256") != database_sha
                or materialization.get("alias_manifest_sha256")
                != relation_generation.get("alias_manifest_sha256")
                or materialization.get("active_core_count")
                != relation_generation.get("active_core_count")
                or materialization.get("materialized_realization_count")
                != relation_generation.get("materialized_realization_count")
                or materialization.get("active_realization_count")
                != relation_generation.get("active_realization_count")
                or resume.get("materialization_manifest_sha256")
                != _sha256(materialization_path)):
            raise TrainedGraphReleaseError(
                "relation generation materialization 身份不闭合")
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
    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        TrainedGenerationConnectorRuntime,
    )
    with TrainedGenerationConnectorRuntime(
            required["training.sqlite3"]) as generation_runtime:
        generation_connector_count = generation_runtime.template_count
    dialogue_runtime = SqliteDialogueSuccessorRuntime(
        required["training.sqlite3"])
    try:
        dialogue_count = dialogue_runtime.count()
    finally:
        dialogue_runtime.close()
    if (min(relation_count, relation_frame_count) <= 0
            or require_dialogue_successor and dialogue_count <= 0):
        raise TrainedGraphReleaseError("训练图缺少 relation/dialogue 承重状态")
    state["runtime_capability_counts"] = {
        "active_relation_propositions": relation_count,
        "relation_surface_frames": relation_frame_count,
        "dialogue_successor_projections": dialogue_count,
        "response_connectors": generation_connector_count,
    }
    state["dialogue_successor_required"] = int(require_dialogue_successor)
    state["free_dialogue_complete"] = 0
    source_ledger = _source_ledger(
        project, _training_lineage(training, pack, summary))
    protocol = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_DIALOGUE_PROTOCOL_V1",
        "schema_version": 1,
        "transport": "jsonl",
        "encoding": "utf-8",
        "operations": ["turn", "quit", "exit"],
        "request": {"required": ["op", "text"], "id_optional": True,
                    "graph_object_keys_optional": True,
                    "graph_object_key_encoding": "integer_arrays",
                    "core_filler_graph_inputs": True,
                    "same_query_state_graph_consumption": True},
        "response": {"type": "turn", "text_field": "text"},
        "memory": {"optional": True, "storage": "sqlite", "integer_graph": True},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise TrainedGraphReleaseError("release staging 已存在，拒绝覆盖")
    (staging / "model").mkdir(parents=True)
    linked_database = staging / "model/training.sqlite3"
    try:
        os.link(required["training.sqlite3"], linked_database)
    except OSError as error:
        raise TrainedGraphReleaseError(
            "同盘模型硬链接失败；拒绝静默复制大模型") from error
    if _sha256(linked_database) != database_sha:
        raise TrainedGraphReleaseError("硬链接后的模型身份漂移")
    shutil.copyfile(
        required["training_cursor.int"], staging / "model/training_cursor.int")
    (staging / "model/training_state.json").write_bytes(_canonical_json(state))
    (staging / "source_manifest.json").write_bytes(
        _canonical_json(source_ledger))
    (staging / "dialogue_protocol.json").write_bytes(_canonical_json(protocol))
    if qualification_path is not None:
        (staging / "qualification_audit.json").write_bytes(
            qualification_path.read_bytes())
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
            "source_manifest": "source_manifest.json",
            "protocol_config": "dialogue_protocol.json",
        },
        "files": files,
    }
    if qualification_path is not None:
        manifest["entry"]["qualification_audit"] = "qualification_audit.json"
    manifest_path = staging / TRAINED_GRAPH_RELEASE_MANIFEST
    manifest_path.write_bytes(_canonical_json(manifest))
    (staging / TRAINED_GRAPH_RELEASE_DIGEST).write_text(
        _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
    staging.rename(target)
    return load_trained_graph_release(
        target, require_k_drive=require_k_drive)


def _closed_runtime_boundary(path: Path) -> dict[str, object]:
    """读取主线审计并拒绝任何生产可达的禁用语言路径。"""
    report = _read_object(path, label="mainline integration audit")
    modules = report.get("modules")
    scopes = report.get("reachable_scope_counts")
    if (report.get("parse_error_count") != 0
            or type(report.get("module_count")) is not int
            or not isinstance(modules, list)
            or not isinstance(scopes, dict)):
        raise TrainedGraphReleaseError("mainline audit 不完整")
    forbidden = [
        item.get("module")
        for item in modules
        if (isinstance(item, dict)
            and item.get("production_reachable") == 1
            and isinstance(item.get("forbidden_markers"), dict)
            and item["forbidden_markers"])
    ]
    if forbidden:
        raise TrainedGraphReleaseError(
            f"发布入口仍可达禁用路径: {sorted(forbidden)}")
    required_scopes = ("query", "generation", "terminal")
    if any(type(scopes.get(name)) is not int or scopes[name] <= 0
           for name in required_scopes):
        raise TrainedGraphReleaseError("mainline audit 缺少生产入口覆盖")
    return {
        "audit_sha256": _sha256(path),
        "module_count": report["module_count"],
        "parse_error_count": 0,
        "production_forbidden_count": 0,
        "reachable_scope_counts": {
            name: scopes[name] for name in required_scopes
        },
    }


def build_event_time_graph_release(
        *,
        materialized_run_root: str | Path,
        mainline_audit: str | Path,
        release_root: str | Path,
        release_id: str,
        require_k_drive: bool = True,
        ) -> TrainedGraphRelease:
    """把已封存 Event/Time 绑定图装成现有 strict graph 发布根。

    物化 run 没有 formal_train 的五件套，不能伪造 training summary。本入口
    改为核验其单调 cursor、最终 receipt、父来源/许可 ledger 和实际运行图。
    SQLite 在同一 K 卷使用硬链接，避免为候选发布重复占用大模型空间；manifest
    仍逐文件锁定内容，任一路径修改都会使发布加载失败。
    """
    if type(release_id) is not str or not release_id.strip():
        raise TrainedGraphReleaseError("release_id 必须是非空文本")
    run = Path(materialized_run_root).resolve()
    audit_path = Path(mainline_audit).resolve()
    target = Path(release_root).resolve()
    if not run.is_dir() or not audit_path.is_file():
        raise TrainedGraphReleaseError("materialized run 或 mainline audit 不存在")
    if require_k_drive and (
            run.drive.upper() != "K:" or target.drive.upper() != "K:"):
        raise TrainedGraphReleaseError("物化模型和发布根必须位于 K 盘")
    if target.exists():
        raise TrainedGraphReleaseError("release root 已存在，拒绝覆盖")
    # 原地物化阶段把多个生成 owner 连续追加到训练 run 的唯一 SQLite；
    # cursor 的完整绝对路径只在构建期读取，发布 manifest 不携带宿主路径。
    cursor_path = run / "event_time_generation_binding_cursor.json"
    cursor_hint: dict[str, object] = {}
    if cursor_path.is_file():
        cursor_hint = _read_object(cursor_path, label="Event/Time binding cursor")
    database_value = cursor_hint.get("database_path")
    database = (
        Path(database_value).resolve(strict=True)
        if type(database_value) is str and database_value
        else run / "training.sqlite3"
    )
    receipt_path = run / "event_time_generation_binding_receipt.json"
    ledger_path = run / "parent_manifest.json"
    if any(not path.is_file() for path in (
            database, cursor_path, receipt_path, ledger_path)):
        raise TrainedGraphReleaseError("Event/Time 物化 run 不完整")
    cursor = _read_object(cursor_path, label="Event/Time binding cursor")
    receipt = _read_object(receipt_path, label="Event/Time binding receipt")
    ledger = _read_object(ledger_path, label="Event/Time source ledger")
    if cursor.get("in_place") == 1:
        model_root = run.parents[1]
        if (database.drive.upper() != "K:"
                or database.parent != model_root
                or receipt.get("in_place") != 1
                or cursor.get("database_copy_count") != 0
                or receipt.get("database_copy_count") != 0
                or receipt.get("database_path") != str(database)):
            raise TrainedGraphReleaseError(
                "Event/Time 原地物化数据库所有权不闭合")
    elif database != run / "training.sqlite3":
        raise TrainedGraphReleaseError("Event/Time 复制物化数据库路径漂移")
    expected_format = "PURE_INTEGER_EVENT_TIME_GENERATION_BINDING_V1"
    database_sha = _sha256(database)
    ledger_sha = _sha256(ledger_path)
    if (cursor.get("format") != expected_format
            or receipt.get("format") != expected_format
            or cursor.get("stage") != 3 or receipt.get("stage") != 3
            or receipt.get("database_sha256") != database_sha
            or receipt.get("database_bytes") != database.stat().st_size
            or cursor.get("source_manifest_sha256") != ledger_sha
            or receipt.get("source_manifest_sha256") != ledger_sha):
        raise TrainedGraphReleaseError("Event/Time cursor/receipt/模型身份不闭合")
    for field in (
            "generation_binding_count",
            "generation_binding_statement_count",
            "reused_object_count"):
        if (type(receipt.get(field)) is not int or receipt[field] <= 0
                or cursor.get(field) != receipt[field]):
            raise TrainedGraphReleaseError(f"Event/Time {field} 不闭合")
    if (receipt.get("semantic_object_copy_count") != 0
            or receipt.get("source_body_read_for_binding") != 0
            or receipt.get("successor_answer_route") != 0
            or receipt.get("free_dialogue_complete") != 0):
        raise TrainedGraphReleaseError("Event/Time 物化边界违反 strict graph 约束")
    sources = ledger.get("sources")
    if (ledger.get("format") != "PURE_INTEGER_TRAINED_GRAPH_SOURCE_LEDGER_V1"
            or not isinstance(sources, list) or not sources
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("license_ids"), list)
                   or not item["license_ids"]
                   or type(item.get("sha256")) is not str
                   or len(item["sha256"]) != 64
                   for item in sources)):
        raise TrainedGraphReleaseError("Event/Time 来源/许可 ledger 不完整")
    boundary = _closed_runtime_boundary(audit_path)

    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        TrainedGenerationConnectorRuntime,
    )
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )
    from pure_integer_ai.experiments.trained_relation_graph_runtime import (
        TrainedRelationGraphRuntime,
    )
    with TrainedRelationGraphRuntime(database) as relation_runtime:
        proposition_count = len(relation_runtime.active_propositions())
        surface_frame_count = len(relation_runtime.active_surface_frames())
    with TrainedGenerationConnectorRuntime(database) as generation_runtime:
        connector_count = generation_runtime.template_count
    with TrainedGraphQueryBridge(database, successor_evidence=False) as query_runtime:
        binding_count = len(query_runtime.event_time_generation_bindings)
        topology_edge_count = len(query_runtime.event_time_topology.edges)
    if (min(proposition_count, surface_frame_count, connector_count,
            binding_count, topology_edge_count) <= 0
            or binding_count != receipt["generation_binding_count"]):
        raise TrainedGraphReleaseError("Event/Time strict runtime 承重状态不闭合")

    state = {
        "format": "PURE_INTEGER_EVENT_TIME_GRAPH_STATE_V1",
        "schema_version": 1,
        "release_id": release_id.strip(),
        "database_sha256": database_sha,
        "database_bytes": database.stat().st_size,
        "source_manifest_sha256": ledger_sha,
        "binding_receipt_sha256": _sha256(receipt_path),
        "runtime_boundary": boundary,
        "runtime_capability_counts": {
            "active_relation_propositions": proposition_count,
            "relation_surface_frames": surface_frame_count,
            "generation_connectors": connector_count,
            "event_time_generation_bindings": binding_count,
            "event_time_topology_edges": topology_edge_count,
        },
        "semantic_object_copy_count": 0,
        "successor_answer_route": 0,
        "source_body_answer_route": 0,
        "shared_model_file_identity": 1,
        "weaning_ready": False,
        "free_dialogue_complete": 0,
    }
    protocol = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_DIALOGUE_PROTOCOL_V1",
        "schema_version": 1,
        "transport": "jsonl",
        "encoding": "utf-8",
        "operations": ["turn", "quit", "exit"],
        "request": {"required": ["op", "text"], "id_optional": True,
                    "graph_object_keys_optional": True,
                    "graph_object_key_encoding": "integer_arrays",
                    "core_filler_graph_inputs": True,
                    "same_query_state_graph_consumption": True},
        "response": {"type": "turn", "text_field": "text"},
        "memory": {"optional": True, "storage": "sqlite", "integer_graph": True},
    }
    from pure_integer_ai.storage.integer_codec import encode_integer_tuple
    cursor_record = (
        1,
        *tuple(bytes.fromhex(database_sha)),
        *tuple(bytes.fromhex(ledger_sha)),
        *tuple(bytes.fromhex(_sha256(receipt_path))),
        receipt["generation_binding_count"],
        receipt["generation_binding_statement_count"],
        receipt["reused_object_count"],
        0,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise TrainedGraphReleaseError("release staging 已存在，拒绝覆盖")
    (staging / "model").mkdir(parents=True)
    linked_database = staging / "model/training.sqlite3"
    try:
        os.link(database, linked_database)
    except OSError as error:
        raise TrainedGraphReleaseError(
            "同盘模型硬链接失败；拒绝静默复制大模型") from error
    if _sha256(linked_database) != database_sha:
        raise TrainedGraphReleaseError("硬链接后的模型身份漂移")
    (staging / "model/training_cursor.int").write_bytes(
        encode_integer_tuple(cursor_record))
    (staging / "model/training_state.json").write_bytes(_canonical_json(state))
    (staging / "source_manifest.json").write_bytes(_canonical_json(ledger))
    (staging / "dialogue_protocol.json").write_bytes(_canonical_json(protocol))
    payloads = tuple(sorted(path for path in staging.rglob("*") if path.is_file()))
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
    return load_trained_graph_release(target, require_k_drive=require_k_drive)


def build_materialized_graph_release(
        *,
        project_root: str | Path,
        materialized_run_root: str | Path,
        materialization_receipt: str | Path | None = None,
        qualification_audit: str | Path,
        mainline_audit: str | Path,
        parallel_query_receipt: str | Path | None = None,
        release_root: str | Path,
        release_id: str,
        require_k_drive: bool = True,
        ) -> TrainedGraphRelease:
    """发布原地物化图，不把它伪装成 formal training run。

    物化运行只携带 cursor/receipt/source ledger；SQLite 在同一 K 卷使用
    硬链接，课程正文和会话状态永不进入 release root。
    """
    if type(release_id) is not str or not release_id.strip():
        raise TrainedGraphReleaseError("release_id 必须是非空文本")
    project = Path(project_root).resolve()
    run = Path(materialized_run_root).resolve()
    target = Path(release_root).resolve()
    qualification_path = Path(qualification_audit).resolve(strict=True)
    audit_path = Path(mainline_audit).resolve(strict=True)
    if not project.is_dir() or not run.is_dir():
        raise TrainedGraphReleaseError("project/materialized run root 不存在")
    if require_k_drive and (
            run.drive.upper() != "K:" or target.drive.upper() != "K:"):
        raise TrainedGraphReleaseError("物化模型和发布根必须位于 K 盘")
    # 资格/主线审计属于 F 盘审计域；只读消费不应把审计副本写回 K 盘。
    if qualification_path.drive.upper() not in {"F:", "K:"}:
        raise TrainedGraphReleaseError("资格收据必须位于 F: 或 K: 审计域")
    if target.exists():
        raise TrainedGraphReleaseError("release root 已存在，拒绝覆盖")
    cursor_path = run / "generic_response_cursor.json"
    receipt_path = Path(materialization_receipt).resolve(strict=True) if (
        materialization_receipt is not None) else run / "generic_response_receipt.json"
    source_path = run / "source_manifest.json"
    parent_source_path = run / "parent_source_manifest.json"
    if any(not path.is_file() for path in (
            cursor_path, receipt_path, source_path, parent_source_path)):
        raise TrainedGraphReleaseError("物化 run 缺少闭合 cursor/receipt/source ledger")
    cursor = _read_object(cursor_path, label="generic response cursor")
    database_value = cursor.get("database_path")
    if cursor.get("in_place") == 1:
        if type(database_value) is not str or not database_value:
            raise TrainedGraphReleaseError("原地物化 cursor 缺少数据库路径")
        database = Path(database_value).resolve(strict=True)
        if database.drive.upper() != "K:" or database == Path(database.anchor):
            raise TrainedGraphReleaseError("原地物化数据库必须是 K 盘显式文件")
    else:
        database = run / "training.sqlite3"
        if not database.is_file() or database_value != str(database):
            raise TrainedGraphReleaseError("复制物化数据库路径漂移")
    receipt = _read_object(receipt_path, label="generic response receipt")
    source = _read_object(source_path, label="materialized source manifest")
    parent_source = _read_object(
        parent_source_path, label="parent source manifest")
    database_sha = _sha256(database)
    broad_receipt = receipt.get("format") == "PURE_INTEGER_BROAD_STRUCTURAL_INCREMENT_V1"
    development_categories = source.get("graph_role_categories")
    heldout_categories = parent_source.get("graph_role_categories")
    structural_categories_closed = (
        isinstance(development_categories, list)
        and isinstance(heldout_categories, list)
        and bool(development_categories) and bool(heldout_categories)
        and development_categories == sorted(set(development_categories))
        and heldout_categories == sorted(set(heldout_categories))
        and all(type(value) is int and 1 <= value <= 5
                for value in (*development_categories, *heldout_categories))
    )
    cursor_closed = (
        cursor.get("stage") == 4 and cursor.get("in_place") == 1
        and cursor.get("database_path") == str(database)
        and cursor.get("database_copy_count") == 0
        and cursor.get("source_manifest_sha256") == _sha256(source_path)
        and cursor.get("parent_source_manifest_sha256")
        == _sha256(parent_source_path))
    if broad_receipt:
        receipt_closed = (
            receipt.get("database") == str(database)
            and receipt.get("database_copy_count") == 0
            and receipt.get("database_link_count_before") == 1
            and receipt.get("database_sha256_after") == database_sha
            and receipt.get("database_sha256_before")
            == cursor.get("parent_database_sha256")
            and receipt.get("course_sha256") == cursor.get("course_sha256")
            and receipt.get("connector_count") == cursor.get("connector_count")
            and receipt.get("realization_count") == cursor.get("realization_count")
            and receipt.get("semantic_variant_count")
            == cursor.get("semantic_variant_count")
            and receipt.get("recovered_semantic_variant_count")
            == cursor.get("semantic_variant_count")
            and receipt.get("source_text_rows_added") == 0
            and receipt.get("semantic_variant_duplicate_count") == 0
            and receipt.get("development_graph_role_categories")
            == development_categories
            and receipt.get("heldout_graph_role_categories")
            == heldout_categories
            and structural_categories_closed
            and receipt.get(
                "development_whole_sentence_representation_count") == 0
            and receipt.get(
                "heldout_whole_sentence_representation_count") == 0
            and receipt.get("duplicate_literal_chunk_count") == 0
            and receipt.get(
                "cross_split_semantic_variant_duplicate_count") == 0
            and receipt.get("recovered_semantic_variant_count")
            == cursor.get("semantic_variant_count")
            and receipt.get("semantic_variant_registry_normalized") == 1
            and receipt.get("successor_answer_route") == 0
            and receipt.get("free_dialogue_complete") == 0)
    else:
        receipt_closed = (
            receipt.get("stage") == 4
            and receipt.get("in_place") == 1
            and receipt.get("database_path") == str(database)
            and receipt.get("model_sha256") == database_sha
            and receipt.get("database_bytes") == database.stat().st_size
            and receipt.get("database_copy_count") == 0
            and receipt.get("source_text_rows_added") == 0
            and receipt.get("semantic_variant_duplicate_count") == 0
            and receipt.get("semantic_variant_registry_normalized") == 1
            and receipt.get("free_dialogue_complete") == 0
            and cursor.get("course_sha256") == receipt.get("course_sha256")
            and cursor.get("source_manifest_sha256")
            == receipt.get("source_manifest_sha256")
            and cursor.get("parent_source_manifest_sha256")
            == receipt.get("parent_source_manifest_sha256"))
    if not cursor_closed or not receipt_closed:
        raise TrainedGraphReleaseError("物化 cursor/receipt 身份或 strict 边界不闭合")
    if int(database.stat().st_nlink) != 1:
        raise TrainedGraphReleaseError(
            "物化模型已与其他发布共享硬链接；拒绝创建重复 active release")
    response_course_formats = {
        "CONDITIONAL_RESPONSE_COURSE_LEDGER_V1",
        "STRUCTURAL_RESPONSE_COURSE_LEDGER_V2",
    }
    if (source.get("format") not in response_course_formats
            or parent_source.get("format") != source.get("format")
            or {source.get("split"), parent_source.get("split")}
            != {"development", "heldout"}
            or source.get("free_dialogue_claim", 0) != 0
            or parent_source.get("free_dialogue_claim", 0) != 0):
        raise TrainedGraphReleaseError("来源账本必须是 development + heldout 且不宣称自由对话")
    qualification = _read_object(
        qualification_path, label="free-dialogue qualification audit")
    if (qualification.get("format")
            != "PURE_INTEGER_FREE_DIALOGUE_QUALIFICATION_AUDIT_V1"
            or qualification.get("qualification_status")
            != "PASS_HELDOUT_CONSUMPTION_ONLY"
            or qualification.get("model_sha256") != database_sha
            or qualification.get("free_dialogue_complete") != 0
            or qualification.get("independent_release_required") != 1):
        raise TrainedGraphReleaseError("资格收据与当前物化模型不闭合")
    boundary = _closed_runtime_boundary(audit_path)

    from pure_integer_ai.experiments.trained_relation_graph_runtime import (
        TrainedRelationGraphRuntime,
    )
    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        TrainedGenerationConnectorRuntime,
    )
    from pure_integer_ai.experiments.trained_graph_query_bridge import (
        TrainedGraphQueryBridge,
    )
    with TrainedRelationGraphRuntime(database) as relation_runtime:
        proposition_count = len(relation_runtime.active_propositions())
        surface_frame_count = len(relation_runtime.active_surface_frames())
    with TrainedGenerationConnectorRuntime(database) as generation_runtime:
        connector_count = generation_runtime.template_count
    with TrainedGraphQueryBridge(database, successor_evidence=False) as query_runtime:
        bridge_binding_count = len(getattr(
            query_runtime, "_artifact_semantic_bindings", ()))
        relation_fact_count = len(
            query_runtime.core_runtime.active_surface_facts())
        relation_routes = query_runtime.relation_routes.all()
        relation_kind_counts: dict[int, int] = {}
        relation_generation_kind_counts: dict[int, int] = {}
        for route in relation_routes:
            relation_kind_counts[route.relation_kind] = (
                relation_kind_counts.get(route.relation_kind, 0) + 1)
            if route.generation_registered:
                relation_generation_kind_counts[route.relation_kind] = (
                    relation_generation_kind_counts.get(
                        route.relation_kind, 0) + 1)
        core_filler_graph_input_count = len(query_runtime.filler_edges)
    if min(proposition_count, surface_frame_count, connector_count) <= 0:
        raise TrainedGraphReleaseError("物化图缺少 relation/surface/connector 承重状态")
    if (len(relation_routes) != relation_fact_count
            or not relation_kind_counts
            or core_filler_graph_input_count <= 0):
        raise TrainedGraphReleaseError(
            "物化图的 relation capability 或 Core filler 输入索引不闭合")
    bridge_receipt_sha = None
    if parallel_query_receipt is not None:
        bridge_path = Path(parallel_query_receipt).resolve(strict=True)
        bridge = _read_object(bridge_path, label="parallel query receipt")
        if bridge.get("protocol") == 1:
            query_count = bridge.get("queried_user_count")
            after_ohe = bridge.get("after_ohe")
            graph_input_count = bridge.get("graph_input_count", 0)
            graph_inputs_closed = (
                graph_input_count == 0
                or (type(graph_input_count) is int
                    and graph_input_count > 0
                    and bridge.get("consumed_graph_input_count")
                    == graph_input_count
                    and bridge.get("generation_consumed_graph_input_count")
                    == graph_input_count))
            modern_closed = (
                bridge_path.name == "receipt.json"
                and qualification.get("heldout_root") == bridge_path.parent.name
                and type(query_count) is int and query_count >= 1
                and bridge.get("canonical_model_sha256_before") == database_sha
                and bridge.get("canonical_model_sha256_after") == database_sha
                and bridge.get("active_spaces") == [1, 2, 3]
                and bridge.get("three_graph_query_count") == query_count
                and type(bridge.get("delivery_count")) is int
                and 0 < bridge["delivery_count"] <= query_count
                and type(bridge.get("heldout_match_count")) is int
                and bridge["heldout_match_count"] >= 1
                and bridge.get("heldout_consumption") == 1
                and bridge.get("heldout_owner_verified") == 1
                and graph_inputs_closed
                and bridge.get("cold_restore_equal") == 1
                and bridge.get("model_read_only") == 1
                and isinstance(after_ohe, list) and len(after_ohe) == 3
                and all(type(value) is int and value > 0
                        for value in after_ohe)
                and bridge.get("database_copy_count") == 0
                and bridge.get("source_text_rows_added") == 0
                and bridge.get("successor_answer_route") == 0
                and bridge.get("source_body_answer_route") == 0
                and bridge.get("character_nearest_route") == 0
                and bridge.get("opencc_route") == 0
                and bridge.get("language_vocabulary_route") == 0
                and bridge.get("free_dialogue_complete") == 0)
            if not modern_closed:
                raise TrainedGraphReleaseError(
                    "现代三图 session 收据与资格审计/当前模型不闭合")
        else:
            query = bridge.get("query")
            if (bridge.get("database_sha256_after") != database_sha
                    or bridge.get("source_text_selected") != 0
                    or bridge.get("successor_answer_route") != 0
                    or bridge.get("source_body_answer_route") != 0
                    or not isinstance(query, dict)
                    or type(query.get("three_graph_query_count")) is not int
                    or query["three_graph_query_count"] <= 0):
                raise TrainedGraphReleaseError(
                    "三图并行收据与当前模型不闭合")
        bridge_receipt_sha = _sha256(bridge_path)

    state = {
        "format": "PURE_INTEGER_MATERIALIZED_GRAPH_STATE_V1",
        "schema_version": 1,
        "release_id": release_id.strip(),
        "database_sha256": database_sha,
        "database_bytes": database.stat().st_size,
        "source_manifest_sha256": _sha256(source_path),
        "parent_source_manifest_sha256": _sha256(parent_source_path),
        "materialization_receipt_sha256": _sha256(receipt_path),
        "qualification_audit_sha256": _sha256(qualification_path),
        "runtime_boundary": boundary,
        "runtime_capability_counts": {
            "active_relation_propositions": proposition_count,
            "active_relation_surface_facts": relation_fact_count,
            "relation_surface_frames": surface_frame_count,
            "response_connectors": connector_count,
            "artifact_bridge_bindings": bridge_binding_count,
            "relation_capability_routes": len(relation_routes),
            "relation_generation_routes": sum(
                relation_generation_kind_counts.values()),
            "core_filler_graph_inputs": core_filler_graph_input_count,
            "semantic_response_variants": int(
                receipt.get("semantic_variant_count", 0)),
        },
        "relation_capability_kind_counts": [
            [kind, relation_kind_counts[kind]]
            for kind in sorted(relation_kind_counts)
        ],
        "relation_generation_kind_counts": [
            [kind, relation_generation_kind_counts[kind]]
            for kind in sorted(relation_generation_kind_counts)
        ],
        "parallel_query_receipt_sha256": bridge_receipt_sha,
        "database_copy_count": 0,
        "source_body_answer_route": 0,
        "successor_answer_route": 0,
        "weaning_ready": False,
        "stage_weaning_ready": False,
        "free_dialogue_complete": 0,
    }
    protocol = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_DIALOGUE_PROTOCOL_V1",
        "schema_version": 1,
        "transport": "jsonl",
        "encoding": "utf-8",
        "operations": ["turn", "quit", "exit"],
        "request": {"required": ["op", "text"], "id_optional": True,
                    "graph_object_keys_optional": True,
                    "graph_object_key_encoding": "integer_arrays",
                    "core_filler_graph_inputs": True,
                    "same_query_state_graph_consumption": True},
        "response": {"type": "turn", "text_field": "text"},
        "memory": {"optional": True, "storage": "sqlite", "integer_graph": True},
    }
    from pure_integer_ai.storage.integer_codec import encode_integer_tuple
    cursor_record = (
        1, *tuple(bytes.fromhex(database_sha)),
        *tuple(bytes.fromhex(_sha256(source_path))),
        int(receipt.get("connector_count", 0)),
        int(receipt.get("realization_count", 0)),
        bridge_binding_count, 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise TrainedGraphReleaseError("release staging 已存在，拒绝覆盖")
    (staging / "model").mkdir(parents=True)
    linked = staging / "model/training.sqlite3"
    try:
        os.link(database, linked)
    except OSError as error:
        raise TrainedGraphReleaseError("同盘模型硬链接失败；拒绝静默复制大模型") from error
    if _sha256(linked) != database_sha:
        raise TrainedGraphReleaseError("硬链接后的模型身份漂移")
    (staging / "model/training_cursor.int").write_bytes(
        encode_integer_tuple(cursor_record))
    (staging / "model/training_state.json").write_bytes(_canonical_json(state))
    (staging / "source_manifest.json").write_bytes(_canonical_json(source))
    (staging / "dialogue_protocol.json").write_bytes(_canonical_json(protocol))
    # 资格收据已验证；发布副本只保留可搬运身份，不携带 K 盘宿主路径。
    qualification_public = dict(qualification)
    qualification_public["model_path"] = "model/training.sqlite3"
    qualification_public["heldout_root"] = "qualification-heldout"
    (staging / "qualification_audit.json").write_bytes(
        _canonical_json(qualification_public))
    files = [{
        "path": path.relative_to(staging).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in sorted(path for path in staging.rglob("*") if path.is_file())]
    manifest = {
        "format": TRAINED_GRAPH_RELEASE_FORMAT,
        "schema_version": 1,
        "release_id": release_id.strip(),
        "entry": {
            "training_database": "model/training.sqlite3",
            "training_cursor": "model/training_cursor.int",
            "training_state": "model/training_state.json",
            "source_manifest": "source_manifest.json",
            "protocol_config": "dialogue_protocol.json",
            "qualification_audit": "qualification_audit.json",
        },
        "files": files,
    }
    manifest_path = staging / TRAINED_GRAPH_RELEASE_MANIFEST
    manifest_path.write_bytes(_canonical_json(manifest))
    (staging / TRAINED_GRAPH_RELEASE_DIGEST).write_text(
        _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
    staging.rename(target)
    return load_trained_graph_release(target, require_k_drive=require_k_drive)


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
    if ("fallback_surfaces" in entry
            or any(isinstance(row, dict)
                   and Path(str(row.get("path", ""))).name
                   == "fallback_surfaces.txt" for row in rows)):
        raise TrainedGraphReleaseError(
            "trained graph release 禁止 fallback 表层")
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
    "build_materialized_graph_release",
    "build_event_time_graph_release",
    "build_trained_graph_release",
    "load_trained_graph_release",
]
