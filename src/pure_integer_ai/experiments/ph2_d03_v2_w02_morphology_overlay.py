"""PH2-D03-V2 W-02 morphology successor 的隔离 overlay artifact。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    W02_CANDIDATE_DB_NAME,
    W02_CANDIDATE_MANIFEST_NAME,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02_MORPH_FEATURE_KINDS,
    W02_MORPH_SUCCESSOR_VERSION,
    W02MorphologySuccessorIndex,
    build_w02_morphology_successor_from_counts,
    w02_morphology_features,
    w02_morphology_lemma_rule,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


W02_MORPH_OVERLAY_VERSION = "PH2-D03-V2-W02-MORPHOLOGY-OVERLAY-V1"
W02_MORPH_OVERLAY_OPERATION = "CANDIDATE_DERIVED_SUCCESSOR_TRANSFORM"
W02_MORPH_OVERLAY_SHARD_COUNT = 128
W02_MORPH_OVERLAY_DB_NAME = "morphology-overlay.sqlite3"
W02_MORPH_OVERLAY_MANIFEST_NAME = "morphology-overlay.artifact.json"
W02_MORPH_OVERLAY_SPOOL_NAME = "input.candidate-rows.sqlite3"
W02_MORPH_OVERLAY_CHECKPOINT_NAME = "checkpoints.jsonl"
W02_MORPH_OVERLAY_SHARD_DIR = "shards"
W02_MORPH_OVERLAY_STATE_DIR = "run-state"
W02_MORPH_OVERLAY_EVENTS = ("BEGIN", "PREVIEW", "COMMIT")


# object-model: exception
class W02MorphologyOverlayError(RuntimeError):
    """overlay 的来源、事务、资源或封存合同被违反。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyOverlayBudget:
    """successor transform 的纯整数资源硬上界。"""

    max_input_rows: int = 100_000
    max_logic_operations: int = 9_000_000
    max_rule_rows: int = 100_000
    max_payload_bytes: int = 536_870_912
    max_shard_delta_bytes: int = 536_870_912
    max_checkpoint_count: int = W02_MORPH_OVERLAY_SHARD_COUNT

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (
                self.max_input_rows, self.max_logic_operations,
                self.max_rule_rows, self.max_payload_bytes,
                self.max_shard_delta_bytes, self.max_checkpoint_count)):
            raise W02MorphologyOverlayError("overlay budget 必须是正整数")
        if self.max_checkpoint_count < W02_MORPH_OVERLAY_SHARD_COUNT:
            raise W02MorphologyOverlayError("overlay checkpoint budget 不足")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyOverlayRunIdentity:
    """不含物理 worker 数和绝对路径的 transform 身份。"""

    release_key: str
    stage_key: str
    run_scope: str
    operation_kind: str
    successor_version: str
    runtime_freeze_sha256: str
    parent_candidate_manifest_sha256: str
    parent_candidate_semantic_sha256: str
    run_id: int
    logical_shard_count: int = W02_MORPH_OVERLAY_SHARD_COUNT

    def __post_init__(self) -> None:
        if (self.release_key != "PH2-D03-V2" or self.stage_key != "W-02"
                or self.run_scope not in {
                    "PUBLIC_SYNTHETIC_FIXTURE", "DEVELOPMENT_PREFLIGHT"}
                or self.operation_kind != W02_MORPH_OVERLAY_OPERATION
                or self.successor_version != W02_MORPH_SUCCESSOR_VERSION):
            raise W02MorphologyOverlayError("overlay run identity 枚举漂移")
        for value in (
                self.runtime_freeze_sha256,
                self.parent_candidate_manifest_sha256,
                self.parent_candidate_semantic_sha256):
            _sha256_text(value, where="overlay run commitment")
        if type(self.run_id) is not int or self.run_id <= 0:
            raise W02MorphologyOverlayError("overlay run_id 必须为正整数")
        if self.logical_shard_count != W02_MORPH_OVERLAY_SHARD_COUNT:
            raise W02MorphologyOverlayError("overlay logical shard count 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_shard_count": self.logical_shard_count,
            "operation_kind": self.operation_kind,
            "parent_candidate_manifest_sha256": (
                self.parent_candidate_manifest_sha256),
            "parent_candidate_semantic_sha256": (
                self.parent_candidate_semantic_sha256),
            "release_key": self.release_key,
            "run_id": self.run_id,
            "run_scope": self.run_scope,
            "runtime_freeze_sha256": self.runtime_freeze_sha256,
            "stage_key": self.stage_key,
            "successor_version": self.successor_version,
        }

    def sha256(self) -> str:
        return _canonical_sha(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologyOverlayRunResult:
    """一个 overlay fresh/restart/resume 的只读结果。"""

    mode: str
    run_identity_sha256: str
    artifact_path: Path
    artifact_manifest_sha256: str
    overlay_semantic_sha256: str
    parent_candidate_semantic_sha256: str
    training_pair_count: int
    morphology_observation_count: int
    morphology_token_count: int
    logic_operations: int
    rule_row_count: int
    shard_count: int
    requested_workers: int
    private_payload_reads: int
    teacher_calls: int
    candidate_writes: int
    overlay_writes: int

    def __post_init__(self) -> None:
        if self.mode not in {"fresh", "restart", "resume"}:
            raise W02MorphologyOverlayError("overlay result mode 未注册")
        for value in (
                self.run_identity_sha256, self.artifact_manifest_sha256,
                self.overlay_semantic_sha256,
                self.parent_candidate_semantic_sha256):
            _sha256_text(value, where="overlay result SHA")
        if not isinstance(self.artifact_path, Path):
            raise W02MorphologyOverlayError("overlay artifact path 类型错误")
        if any(type(value) is not int or value < 0 for value in (
                self.training_pair_count, self.morphology_observation_count,
                self.morphology_token_count, self.logic_operations,
                self.rule_row_count, self.shard_count,
                self.requested_workers, self.private_payload_reads,
                self.teacher_calls, self.candidate_writes,
                self.overlay_writes)):
            raise W02MorphologyOverlayError("overlay result 计数非法")


def _sha256_text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologyOverlayError(f"{where} 非法")
    return value


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _tree_inventory(root: Path, *, skip_manifest: bool) -> list[dict[str, object]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if skip_manifest and relative == W02_MORPH_OVERLAY_MANIFEST_NAME:
            continue
        size, digest = _sha256_file(path)
        rows.append({"path": relative, "sha256": digest, "size_bytes": size})
    return rows


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True)
    else:
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA page_size=4096")
    connection.row_factory = sqlite3.Row
    return connection


def _safe_child(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root) or target.is_symlink():
        raise W02MorphologyOverlayError("overlay path 逃逸或为 symlink")
    return target


def _publish_directory(staging: Path, final: Path) -> None:
    """有界处理 Windows 已关闭句柄的延迟释放，不改变发布身份。"""
    last_error: PermissionError | None = None
    for _attempt in range(8):
        try:
            os.replace(staging, final)
            return
        except PermissionError as error:
            last_error = error
            if final.exists() or not staging.is_dir():
                raise W02MorphologyOverlayError(
                    "overlay final publish 路径状态漂移") from error
            gc.collect()
    raise W02MorphologyOverlayError(
        "overlay final publish Windows handle resource stop") from last_error


def _run_dirs(root: Path, run_id: int) -> tuple[Path, Path, Path]:
    store = _safe_child(root, "morphology-overlay-store")
    staging = _safe_child(store, f"run-{run_id:06d}.staging")
    final = _safe_child(store, f"run-{run_id:06d}")
    return store, staging, final


def _dataset_key_from_source_ref(value: object) -> tuple[int, ...]:
    """按冻结 W-02 v2 完整整数键投影来源对应的 dataset route。"""
    if (not isinstance(value, list) or len(value) < 6
            or any(type(item) is not int or item <= 0 for item in value)
            or value[0:2] != [2, 2] or value[3] != 10 or value[4] != 1):
        raise W02MorphologyOverlayError("overlay source_ref_key 不是 train SourceRef")
    return (value[0], value[1], value[2], 1)


def _validate_dataset_key(value: object) -> tuple[int, ...]:
    if (not isinstance(value, list) or len(value) != 4
            or any(type(item) is not int or item <= 0 for item in value)
            or value[0:2] != [2, 2] or value[3] != 1):
        raise W02MorphologyOverlayError("overlay dataset route 非法")
    return tuple(value)


def _candidate_tree(root: Path) -> tuple[tuple[str, tuple[int, str]], ...]:
    return tuple(
        (item.relative_to(root).as_posix(), _sha256_file(item))
        for item in sorted(root.rglob("*")) if item.is_file()
    )


def _candidate_counts(
        connection: sqlite3.Connection,
        ) -> tuple[
            tuple[tuple[int, ...], ...],
            tuple[tuple[str, str, str, str, int], ...],
            int,
            int,
        ]:
    """只读 Candidate 的 train 聚合，不接触 Observation 表层或 teacher。"""
    lexemes = tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]))
        for row in connection.execute(
            "SELECT form,lemma,upos,feats_json,support_count FROM lexemes "
            "ORDER BY form,lemma,upos,feats_json")
    )
    pair_count = int(connection.execute(
        "SELECT COUNT(*) FROM evidence_applications").fetchone()[0])
    morphology_count = 0
    routes: set[tuple[int, ...]] = set()
    for row in connection.execute(
            "SELECT source_ref_key,evidence_mode FROM evidence_applications "
            "ORDER BY observation_key"):
        if str(row[1]) != "UD_ANNOTATION":
            continue
        source = parse_canonical_json_bytes(bytes(row[0]), require_object=False)
        routes.add(_dataset_key_from_source_ref(source))
        morphology_count += 1
    return tuple(sorted(routes)), lexemes, pair_count, morphology_count


def derive_w02_morphology_successor_from_candidate(
        candidate_artifact_root: str | Path,
        ) -> W02MorphologySuccessorIndex:
    """从 sealed Candidate 只读重建索引，并证明没有写回父 artifact。"""
    result = read_w02_candidate_artifact(candidate_artifact_root)
    root = result.artifact_path
    before = _candidate_tree(root)
    connection = _connect(root / W02_CANDIDATE_DB_NAME, read_only=True)
    try:
        routes, lexemes, pair_count, morphology_count = _candidate_counts(connection)
    finally:
        connection.close()
    index = build_w02_morphology_successor_from_counts(
        dataset_keys=routes,
        lexeme_counts=lexemes,
        training_pair_count=pair_count,
        morphology_observation_count=morphology_count,
    )
    if _candidate_tree(root) != before:
        raise W02MorphologyOverlayError("overlay derivation 写回了父 Candidate")
    return index


def _shard_for(kind: str, payload: dict[str, object]) -> int:
    digest = hashlib.sha256(canonical_json_bytes([kind, payload])).digest()
    return int.from_bytes(digest[:8], "big") % W02_MORPH_OVERLAY_SHARD_COUNT


def _create_spool(path: Path) -> sqlite3.Connection:
    connection = _connect(path)
    connection.executescript(
        """
        CREATE TABLE input_meta(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
        CREATE TABLE input_rows(
            sequence INTEGER PRIMARY KEY,
            shard_index INTEGER NOT NULL,
            row_kind TEXT NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE INDEX input_rows_shard_order
            ON input_rows(shard_index,sequence);
        """
    )
    connection.commit()
    return connection


def _write_input_spool(
        path: Path,
        candidate_root: Path,
        identity: W02MorphologyOverlayRunIdentity,
        budget: W02MorphologyOverlayBudget,
        ) -> dict[str, int]:
    """将父 Candidate 的必要聚合行单遍写入受控临时 spool。"""
    if path.exists():
        raise W02MorphologyOverlayError("overlay input spool 不得覆盖")
    parent = read_w02_candidate_artifact(candidate_root)
    if (parent.artifact_manifest_sha256
            != identity.parent_candidate_manifest_sha256
            or parent.candidate_semantic_sha256
            != identity.parent_candidate_semantic_sha256):
        raise W02MorphologyOverlayError("overlay parent Candidate identity 漂移")
    before = _candidate_tree(parent.artifact_path)
    spool = _create_spool(path)
    source = _connect(
        parent.artifact_path / W02_CANDIDATE_DB_NAME, read_only=True)
    sequence = 0
    payload_bytes = 0
    lexeme_row_count = 0
    application_row_count = 0
    digest = hashlib.sha256()

    def append(kind: str, payload: dict[str, object]) -> None:
        nonlocal sequence, payload_bytes
        if sequence >= budget.max_input_rows:
            raise W02MorphologyOverlayError("overlay input row resource stop")
        encoded = canonical_json_bytes(payload)
        payload_bytes += len(encoded)
        if payload_bytes > budget.max_payload_bytes:
            raise W02MorphologyOverlayError("overlay input payload resource stop")
        spool.execute(
            "INSERT INTO input_rows VALUES(?,?,?,?)",
            (sequence, _shard_for(kind, payload), kind, encoded),
        )
        digest.update(hashlib.sha256(
            canonical_json_bytes([kind, payload])).digest())
        sequence += 1

    try:
        for row in source.execute(
                "SELECT form,lemma,upos,feats_json,support_count FROM lexemes "
                "ORDER BY form,lemma,upos,feats_json"):
            append("LEXEME", {
                "feats_json": str(row[3]),
                "form": str(row[0]),
                "lemma": str(row[1]),
                "support_count": int(row[4]),
                "upos": str(row[2]),
            })
            lexeme_row_count += 1
        for row in source.execute(
                "SELECT source_ref_key,evidence_mode FROM evidence_applications "
                "ORDER BY observation_key"):
            source_key = parse_canonical_json_bytes(
                bytes(row[0]), require_object=False)
            if not isinstance(source_key, list):
                raise W02MorphologyOverlayError("overlay application source key 非法")
            append("APPLICATION", {
                "evidence_mode": str(row[1]),
                "source_ref_key": source_key,
            })
            application_row_count += 1
        if application_row_count != parent.pair_count:
            raise W02MorphologyOverlayError("overlay application count 漂移")
        meta: dict[str, object] = {
            "application_row_count": application_row_count,
            "input_digest_sha256": digest.hexdigest(),
            "input_row_count": sequence,
            "lexeme_row_count": lexeme_row_count,
            "parent_candidate_manifest_sha256": (
                identity.parent_candidate_manifest_sha256),
            "parent_candidate_semantic_sha256": (
                identity.parent_candidate_semantic_sha256),
            "payload_bytes": payload_bytes,
            "run_identity_sha256": identity.sha256(),
            "run_identity": identity.to_dict(),
        }
        for key, value in sorted(meta.items()):
            spool.execute(
                "INSERT INTO input_meta VALUES(?,?)",
                (key, canonical_json_bytes(value).decode("utf-8")),
            )
        spool.commit()
    except BaseException:
        spool.close()
        source.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    spool.close()
    source.close()
    if _candidate_tree(parent.artifact_path) != before:
        raise W02MorphologyOverlayError("overlay spool 写回了父 Candidate")
    return {
        "application_row_count": application_row_count,
        "input_row_count": sequence,
        "lexeme_row_count": lexeme_row_count,
        "payload_bytes": payload_bytes,
    }


def _read_spool_meta(
        path: Path,
        identity: W02MorphologyOverlayRunIdentity,
        ) -> dict[str, Any]:
    connection = _connect(path, read_only=True)
    values = {
        str(row[0]): json.loads(str(row[1]))
        for row in connection.execute(
            "SELECT key,value_json FROM input_meta ORDER BY key")
    }
    count = int(connection.execute(
        "SELECT COUNT(*) FROM input_rows").fetchone()[0])
    connection.close()
    if (values.get("run_identity_sha256") != identity.sha256()
            or values.get("input_row_count") != count):
        raise W02MorphologyOverlayError("overlay input spool identity 漂移")
    return values


def _write_gzip_rows(
        path: Path,
        rows: list[dict[str, object]],
        ) -> tuple[int, str, int, str]:
    partial = path.with_suffix(path.suffix + ".partial")
    content_digest = hashlib.sha256()
    content_size = 0
    with partial.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for row in rows:
                line = canonical_json_bytes(row) + b"\n"
                stream.write(line)
                content_digest.update(line)
                content_size += len(line)
    transport_size, transport_sha = _sha256_file(partial)
    os.replace(partial, path)
    return content_size, content_digest.hexdigest(), transport_size, transport_sha


def _read_gzip_rows(path: Path) -> Iterator[dict[str, object]]:
    try:
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        raise W02MorphologyOverlayError(
                            "overlay shard delta 换行非法")
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    if not isinstance(value, dict):
                        raise W02MorphologyOverlayError(
                            "overlay shard delta row 非 object")
                    yield value
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02MorphologyOverlayError):
            raise
        raise W02MorphologyOverlayError(
            "overlay shard delta gzip/JSONL 损坏") from error


def _aggregate_row(
        values: dict[bytes, dict[str, object]],
        row: dict[str, object],
        ) -> None:
    identity = dict(row)
    identity.pop("count", None)
    key = canonical_json_bytes(identity)
    prior = values.get(key)
    if prior is None:
        values[key] = row
        return
    if row["row_kind"] == "DATASET_ROUTE":
        return
    updated = dict(prior)
    updated["count"] = int(prior["count"]) + int(row["count"])
    values[key] = updated


def _shard_worker(
        spool_path: str,
        shard_dir: str,
        identity: W02MorphologyOverlayRunIdentity,
        shard_index: int,
        budget: W02MorphologyOverlayBudget,
        ) -> dict[str, object]:
    """只读一个 logical shard，并输出可规范归并的规则 delta。"""
    output = Path(shard_dir) / f"{shard_index:03d}.delta.jsonl.gz"
    manifest_path = Path(shard_dir) / f"{shard_index:03d}.delta.manifest.json"
    if output.is_file() and manifest_path.is_file():
        value = read_canonical_object(manifest_path)
        if (value.get("run_identity_sha256") != identity.sha256()
                or value.get("shard_index") != shard_index):
            raise W02MorphologyOverlayError("已有 overlay shard identity 漂移")
        return value
    if output.exists() or manifest_path.exists():
        raise W02MorphologyOverlayError("overlay shard partial 不可恢复")
    connection = _connect(Path(spool_path), read_only=True)
    aggregates: dict[bytes, dict[str, object]] = {}
    input_count = 0
    pair_count = 0
    morphology_count = 0
    token_count = 0
    operations = 0
    max_form_length = 0
    try:
        for source in connection.execute(
                "SELECT row_kind,payload FROM input_rows "
                "WHERE shard_index=? ORDER BY sequence", (shard_index,)):
            kind = str(source[0])
            payload = parse_canonical_json_bytes(
                bytes(source[1]), require_object=True)
            if not isinstance(payload, dict):
                raise W02MorphologyOverlayError("overlay spool payload 非 object")
            input_count += 1
            if kind == "APPLICATION":
                pair_count += 1
                mode = str(payload.get("evidence_mode"))
                if mode == "UD_ANNOTATION":
                    route = _dataset_key_from_source_ref(
                        payload.get("source_ref_key"))
                    morphology_count += 1
                    _aggregate_row(aggregates, {
                        "dataset_key": list(route),
                        "row_kind": "DATASET_ROUTE",
                    })
                else:
                    operations += 1
            elif kind == "LEXEME":
                form = str(payload.get("form"))
                lemma = str(payload.get("lemma"))
                upos = str(payload.get("upos"))
                feats_json = str(payload.get("feats_json"))
                support = payload.get("support_count")
                if (not form or not upos or not feats_json
                        or type(support) is not int or support <= 0):
                    raise W02MorphologyOverlayError("overlay lexeme payload 非法")
                rule = w02_morphology_lemma_rule(form, lemma)
                if rule is None:
                    operations += support
                    continue
                token_count += support
                max_form_length = max(max_form_length, len(form))
                operations += support * (
                    len(form) + len(W02_MORPH_FEATURE_KINDS) + 8)
                _aggregate_row(aggregates, {
                    "count": support,
                    "feats_json": feats_json,
                    "lemma_rule": rule,
                    "row_kind": "GLOBAL_COMBO",
                    "upos": upos,
                })
                for feature_kind, feature_value in w02_morphology_features(form):
                    _aggregate_row(aggregates, {
                        "count": support,
                        "feats_json": feats_json,
                        "feature_kind": feature_kind,
                        "feature_value": feature_value,
                        "lemma_rule": rule,
                        "row_kind": "LOCAL_COMBO",
                        "upos": upos,
                    })
            else:
                raise W02MorphologyOverlayError("overlay spool row kind 未注册")
            if operations > budget.max_logic_operations:
                raise W02MorphologyOverlayError(
                    "overlay shard logic resource stop")
    finally:
        connection.close()
    rows = [aggregates[key] for key in sorted(aggregates)]
    if len(rows) > budget.max_rule_rows:
        raise W02MorphologyOverlayError("overlay shard rule resource stop")
    content_size, content_sha, transport_size, transport_sha = _write_gzip_rows(
        output, rows)
    if transport_size > budget.max_shard_delta_bytes:
        raise W02MorphologyOverlayError("overlay shard payload resource stop")
    value = {
        "artifact_version": W02_MORPH_OVERLAY_VERSION,
        "content_sha256": content_sha,
        "content_size_bytes": content_size,
        "input_row_count": input_count,
        "logic_operations": operations,
        "max_form_length": max_form_length,
        "morphology_observation_count": morphology_count,
        "morphology_token_count": token_count,
        "rule_row_count": len(rows),
        "run_identity_sha256": identity.sha256(),
        "shard_index": shard_index,
        "training_pair_count": pair_count,
        "transport_sha256": transport_sha,
        "transport_size_bytes": transport_size,
    }
    write_immutable_json(value, manifest_path)
    return value


def _content_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for row in _read_gzip_rows(path):
        payload = canonical_json_bytes(row) + b"\n"
        digest.update(payload)
        size += len(payload)
    return size, digest.hexdigest()


def _create_overlay_db(path: Path) -> sqlite3.Connection:
    connection = _connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
        CREATE TABLE run_events(
            event_seq INTEGER PRIMARY KEY,
            event_kind TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE dataset_routes(
            dataset_key BLOB PRIMARY KEY
        );
        CREATE TABLE global_combos(
            lemma_rule TEXT NOT NULL,
            upos TEXT NOT NULL,
            feats_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(lemma_rule,upos,feats_json)
        );
        CREATE TABLE local_combos(
            feature_kind TEXT NOT NULL,
            feature_value TEXT NOT NULL,
            lemma_rule TEXT NOT NULL,
            upos TEXT NOT NULL,
            feats_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(feature_kind,feature_value,lemma_rule,upos,feats_json)
        );
        CREATE TABLE checkpoints(
            checkpoint_seq INTEGER PRIMARY KEY,
            shard_index INTEGER NOT NULL UNIQUE,
            shard_manifest_sha256 TEXT NOT NULL,
            input_row_count INTEGER NOT NULL,
            logic_operations INTEGER NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _event(
        connection: sqlite3.Connection,
        kind: str,
        payload: dict[str, object],
        ) -> None:
    position = W02_MORPH_OVERLAY_EVENTS.index(kind)
    encoded = canonical_json_bytes(payload).decode("utf-8")
    connection.execute(
        "INSERT INTO run_events VALUES(?,?,?,?)",
        (position + 1, kind,
         hashlib.sha256(encoded.encode("utf-8")).hexdigest(), encoded),
    )


def _merge_delta_row(
        connection: sqlite3.Connection,
        value: dict[str, object],
        ) -> None:
    kind = value.get("row_kind")
    if kind == "DATASET_ROUTE":
        if set(value) != {"dataset_key", "row_kind"}:
            raise W02MorphologyOverlayError("overlay route row 字段漂移")
        key = _validate_dataset_key(value["dataset_key"])
        connection.execute(
            "INSERT OR IGNORE INTO dataset_routes VALUES(?)",
            (canonical_json_bytes(list(key)),))
        return
    count = value.get("count")
    if type(count) is not int or count <= 0:
        raise W02MorphologyOverlayError("overlay combo count 非法")
    if kind == "GLOBAL_COMBO":
        if set(value) != {
                "count", "feats_json", "lemma_rule", "row_kind", "upos"}:
            raise W02MorphologyOverlayError("overlay global row 字段漂移")
        connection.execute(
            "INSERT INTO global_combos VALUES(?,?,?,?) "
            "ON CONFLICT DO UPDATE SET "
            "support_count=support_count+excluded.support_count",
            (value["lemma_rule"], value["upos"], value["feats_json"], count),
        )
    elif kind == "LOCAL_COMBO":
        if set(value) != {
                "count", "feats_json", "feature_kind", "feature_value",
                "lemma_rule", "row_kind", "upos"}:
            raise W02MorphologyOverlayError("overlay local row 字段漂移")
        if value["feature_kind"] not in W02_MORPH_FEATURE_KINDS:
            raise W02MorphologyOverlayError("overlay feature kind 漂移")
        connection.execute(
            "INSERT INTO local_combos VALUES(?,?,?,?,?,?) "
            "ON CONFLICT DO UPDATE SET "
            "support_count=support_count+excluded.support_count",
            (value["feature_kind"], value["feature_value"],
             value["lemma_rule"], value["upos"], value["feats_json"], count),
        )
    else:
        raise W02MorphologyOverlayError("overlay delta row kind 未注册")


def _index_from_combo_tables(
        connection: sqlite3.Connection,
        *,
        max_form_length: int,
        training_pair_count: int,
        morphology_observation_count: int,
        morphology_token_count: int,
        logic_operations: int,
        ) -> W02MorphologySuccessorIndex:
    """从已归并 combo 表恢复索引，不尝试反推原始词形。"""
    routes = tuple(sorted(
        tuple(parse_canonical_json_bytes(bytes(row[0]), require_object=False))
        for row in connection.execute(
            "SELECT dataset_key FROM dataset_routes ORDER BY dataset_key")
    ))
    global_counts = {
        (str(row[0]), str(row[1]), str(row[2])): int(row[3])
        for row in connection.execute(
            "SELECT lemma_rule,upos,feats_json,support_count "
            "FROM global_combos ORDER BY lemma_rule,upos,feats_json")
    }
    feature_counts: dict[
        tuple[str, str], dict[tuple[str, str, str], int]
    ] = {}
    for row in connection.execute(
            "SELECT feature_kind,feature_value,lemma_rule,upos,feats_json,"
            "support_count FROM local_combos ORDER BY feature_kind,feature_value,"
            "lemma_rule,upos,feats_json"):
        feature = (str(row[0]), str(row[1]))
        combo = (str(row[2]), str(row[3]), str(row[4]))
        feature_counts.setdefault(feature, {})[combo] = int(row[5])
    semantic_rows: list[dict[str, object]] = [
        {"dataset_key": list(key), "row_kind": "DATASET_ROUTE"}
        for key in routes
    ]
    semantic_rows.extend({
        "count": count,
        "feats_json": combo[2],
        "lemma_rule": combo[0],
        "row_kind": "GLOBAL_COMBO",
        "upos": combo[1],
    } for combo, count in sorted(global_counts.items()))
    for feature, counts in sorted(feature_counts.items()):
        semantic_rows.extend({
            "count": count,
            "feats_json": combo[2],
            "feature_kind": feature[0],
            "feature_value": feature[1],
            "lemma_rule": combo[0],
            "row_kind": "LOCAL_COMBO",
            "upos": combo[1],
        } for combo, count in sorted(counts.items()))
    return W02MorphologySuccessorIndex(
        routes, global_counts, feature_counts, max_form_length,
        training_pair_count, morphology_observation_count,
        morphology_token_count, logic_operations,
        _canonical_sha(semantic_rows), len(semantic_rows),
    )


def _merge_artifact(
        staging: Path,
        identity: W02MorphologyOverlayRunIdentity,
        expected_input_rows: int,
        budget: W02MorphologyOverlayBudget,
        ) -> W02MorphologySuccessorIndex:
    partial = staging / (W02_MORPH_OVERLAY_DB_NAME + ".partial")
    if partial.exists():
        raise W02MorphologyOverlayError("overlay DB partial 不得覆盖")
    connection = _create_overlay_db(partial)
    _event(connection, "BEGIN", {
        "operation_kind": identity.operation_kind,
        "run_identity_sha256": identity.sha256(),
    })
    totals = {
        "input_row_count": 0,
        "logic_operations": 0,
        "morphology_observation_count": 0,
        "morphology_token_count": 0,
        "training_pair_count": 0,
    }
    max_form_length = 0
    try:
        for shard_index in range(W02_MORPH_OVERLAY_SHARD_COUNT):
            manifest_path = (
                staging / W02_MORPH_OVERLAY_SHARD_DIR
                / f"{shard_index:03d}.delta.manifest.json")
            delta_path = (
                staging / W02_MORPH_OVERLAY_SHARD_DIR
                / f"{shard_index:03d}.delta.jsonl.gz")
            manifest = read_canonical_object(manifest_path)
            if (manifest.get("run_identity_sha256") != identity.sha256()
                    or manifest.get("shard_index") != shard_index):
                raise W02MorphologyOverlayError("overlay shard manifest 漂移")
            content_size, content_sha = _content_identity(delta_path)
            transport_size, transport_sha = _sha256_file(delta_path)
            if (content_size != manifest.get("content_size_bytes")
                    or content_sha != manifest.get("content_sha256")
                    or transport_size != manifest.get("transport_size_bytes")
                    or transport_sha != manifest.get("transport_sha256")):
                raise W02MorphologyOverlayError("overlay shard content 漂移")
            for value in _read_gzip_rows(delta_path):
                _merge_delta_row(connection, value)
            for key in totals:
                totals[key] += int(manifest[key])
            max_form_length = max(
                max_form_length, int(manifest["max_form_length"]))
            connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?,?)",
                (shard_index + 1, shard_index,
                 _canonical_sha(manifest), int(manifest["input_row_count"]),
                 int(manifest["logic_operations"])),
            )
        rule_row_count = sum(int(connection.execute(query).fetchone()[0]) for query in (
            "SELECT COUNT(*) FROM dataset_routes",
            "SELECT COUNT(*) FROM global_combos",
            "SELECT COUNT(*) FROM local_combos",
        ))
        if totals["input_row_count"] != expected_input_rows:
            raise W02MorphologyOverlayError("overlay merged input count 不闭合")
        if (totals["logic_operations"] > budget.max_logic_operations
                or rule_row_count > budget.max_rule_rows):
            raise W02MorphologyOverlayError("overlay merged resource stop")
        index = _index_from_combo_tables(
            connection,
            max_form_length=max_form_length,
            training_pair_count=totals["training_pair_count"],
            morphology_observation_count=(
                totals["morphology_observation_count"]),
            morphology_token_count=totals["morphology_token_count"],
            logic_operations=totals["logic_operations"],
        )
        for key, value in sorted({
                "artifact_version": W02_MORPH_OVERLAY_VERSION,
                "logic_operations": index.logic_operations,
                "max_form_length": index.max_form_length,
                "morphology_observation_count": (
                    index.morphology_observation_count),
                "morphology_token_count": index.morphology_token_count,
                "overlay_semantic_sha256": index.semantic_sha256,
                "parent_candidate_manifest_sha256": (
                    identity.parent_candidate_manifest_sha256),
                "parent_candidate_semantic_sha256": (
                    identity.parent_candidate_semantic_sha256),
                "rule_row_count": index.row_count,
                "run_identity_sha256": identity.sha256(),
                "training_pair_count": index.training_pair_count,
                }.items()):
            connection.execute(
                "INSERT INTO meta VALUES(?,?)",
                (key, canonical_json_bytes(value).decode("utf-8")),
            )
        _event(connection, "PREVIEW", {
            "logic_operations": index.logic_operations,
            "overlay_semantic_sha256": index.semantic_sha256,
            "rule_row_count": index.row_count,
        })
        _event(connection, "COMMIT", {
            "overlay_semantic_sha256": index.semantic_sha256,
            "training_pair_count": index.training_pair_count,
        })
        connection.commit()
    finally:
        connection.close()
    os.replace(partial, staging / W02_MORPH_OVERLAY_DB_NAME)
    return index


def _checkpoint_payload(value: dict[str, object]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _append_checkpoint(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as handle:
        handle.write(_checkpoint_payload(value))


def _read_checkpoints(
        path: Path,
        identity: W02MorphologyOverlayRunIdentity,
        ) -> dict[int, dict[str, object]]:
    result = {}
    if not path.is_file():
        return result
    with path.open("rb") as handle:
        for line in handle:
            value = parse_canonical_json_bytes(
                line.rstrip(b"\n"), require_object=True)
            if (not isinstance(value, dict)
                    or value.get("run_identity_sha256") != identity.sha256()):
                raise W02MorphologyOverlayError("overlay checkpoint identity 漂移")
            shard = value.get("shard_index")
            if (type(shard) is not int or shard in result
                    or value.get("checkpoint_seq") != shard + 1):
                raise W02MorphologyOverlayError("overlay checkpoint 顺序非法")
            result[shard] = value
    return result


def _run_state(
        staging: Path,
        identity: W02MorphologyOverlayRunIdentity,
        status: str,
        ) -> None:
    if status not in {"RUNNING", "FAILED_RECOVERABLE", "SEALED"}:
        raise W02MorphologyOverlayError("overlay run state 未注册")
    root = staging / W02_MORPH_OVERLAY_STATE_DIR
    root.mkdir(exist_ok=True)
    prior = tuple(sorted(root.glob("*.json")))
    value = {
        "artifact_version": W02_MORPH_OVERLAY_VERSION,
        "run_identity_sha256": identity.sha256(),
        "status": status,
    }
    if prior and read_canonical_object(prior[-1]) == value:
        return
    write_immutable_json(
        value, root / f"{len(prior) + 1:03d}-{status.casefold()}.json")


def _publish_manifest(
        staging: Path,
        identity: W02MorphologyOverlayRunIdentity,
        index: W02MorphologySuccessorIndex,
        input_row_count: int,
        ) -> str:
    path = staging / W02_MORPH_OVERLAY_MANIFEST_NAME
    if path.exists():
        raise W02MorphologyOverlayError("overlay artifact manifest 不得覆盖")
    value = {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_OVERLAY",
        "artifact_version": W02_MORPH_OVERLAY_VERSION,
        "candidate_writes": 0,
        "formal_private_evaluation_runs": 0,
        "formal_successor_transform_runs": 0,
        "formal_training_runs": 0,
        "input_row_count": input_row_count,
        "input_roots": ["PARENT_CANDIDATE_ARTIFACT"],
        "logic_operations": index.logic_operations,
        "logical_shard_count": W02_MORPH_OVERLAY_SHARD_COUNT,
        "morphology_observation_count": index.morphology_observation_count,
        "morphology_token_count": index.morphology_token_count,
        "operation_kind": identity.operation_kind,
        "overlay_semantic_sha256": index.semantic_sha256,
        "overlay_writes": 1,
        "parent_candidate_manifest_sha256": (
            identity.parent_candidate_manifest_sha256),
        "parent_candidate_semantic_sha256": (
            identity.parent_candidate_semantic_sha256),
        "private_payload_reads": 0,
        "rule_row_count": index.row_count,
        "run_identity": identity.to_dict(),
        "run_identity_sha256": identity.sha256(),
        "run_scope": identity.run_scope,
        "status": "MORPHOLOGY_OVERLAY_SEALED",
        "teacher_calls": 0,
        "training_pair_count": index.training_pair_count,
        "tree": _tree_inventory(staging, skip_manifest=True),
        "visible_splits": ["train-derived-candidate-statistics"],
        "worker_counts_supported": [1, 2, 4],
    }
    write_immutable_json(value, path)
    expected = canonical_json_bytes(value) + b"\n"
    if path.read_bytes() != expected:
        raise W02MorphologyOverlayError("overlay artifact manifest 字节漂移")
    return hashlib.sha256(expected).hexdigest()


def _execute_pipeline(
        *,
        staging: Path,
        final: Path,
        identity: W02MorphologyOverlayRunIdentity,
        expected_input_rows: int,
        requested_workers: int,
        mode: str,
        budget: W02MorphologyOverlayBudget,
        fault_after_shard: int | None,
        ) -> W02MorphologyOverlayRunResult:
    completed = _read_checkpoints(
        staging / W02_MORPH_OVERLAY_CHECKPOINT_NAME, identity)
    shard_dir = staging / W02_MORPH_OVERLAY_SHARD_DIR
    futures = {}
    with ThreadPoolExecutor(max_workers=requested_workers) as executor:
        for shard_index in range(W02_MORPH_OVERLAY_SHARD_COUNT):
            if shard_index not in completed:
                futures[shard_index] = executor.submit(
                    _shard_worker,
                    str(staging / W02_MORPH_OVERLAY_SPOOL_NAME),
                    str(shard_dir), identity, shard_index, budget)
        for shard_index in range(W02_MORPH_OVERLAY_SHARD_COUNT):
            if shard_index in completed:
                continue
            manifest = futures[shard_index].result()
            checkpoint = {
                "checkpoint_seq": shard_index + 1,
                "input_row_count": int(manifest["input_row_count"]),
                "logic_operations": int(manifest["logic_operations"]),
                "run_identity_sha256": identity.sha256(),
                "shard_index": shard_index,
                "shard_manifest_sha256": _canonical_sha(manifest),
            }
            _append_checkpoint(
                staging / W02_MORPH_OVERLAY_CHECKPOINT_NAME, checkpoint)
            completed[shard_index] = checkpoint
            if fault_after_shard == shard_index:
                _run_state(staging, identity, "FAILED_RECOVERABLE")
                raise W02MorphologyOverlayError("injected overlay shard fault")
    if len(completed) != W02_MORPH_OVERLAY_SHARD_COUNT:
        raise W02MorphologyOverlayError("overlay shard completion 不闭合")
    index = _merge_artifact(staging, identity, expected_input_rows, budget)
    (staging / W02_MORPH_OVERLAY_SPOOL_NAME).unlink()
    _run_state(staging, identity, "SEALED")
    manifest_sha = _publish_manifest(
        staging, identity, index, expected_input_rows)
    _publish_directory(staging, final)
    return W02MorphologyOverlayRunResult(
        mode, identity.sha256(), final, manifest_sha, index.semantic_sha256,
        identity.parent_candidate_semantic_sha256,
        index.training_pair_count, index.morphology_observation_count,
        index.morphology_token_count, index.logic_operations,
        index.row_count, W02_MORPH_OVERLAY_SHARD_COUNT, requested_workers,
        0, 0, 0, 1,
    )


def run_w02_morphology_overlay_fixture(
        *,
        fixture_root: str | Path,
        candidate_artifact_root: str | Path | None,
        run_id: int,
        requested_workers: int,
        mode: str,
        budget: W02MorphologyOverlayBudget | None = None,
        fault_after_shard: int | None = None,
        ) -> W02MorphologyOverlayRunResult:
    """运行公开 fixture transform；该入口不构成正式训练或正式 successor。"""
    return _run_w02_morphology_overlay_nonformal(
        fixture_root=fixture_root,
        candidate_artifact_root=candidate_artifact_root,
        run_id=run_id,
        requested_workers=requested_workers,
        mode=mode,
        run_scope="PUBLIC_SYNTHETIC_FIXTURE",
        budget=budget,
        fault_after_shard=fault_after_shard,
    )


def run_w02_morphology_overlay_development(
        *,
        development_root: str | Path,
        candidate_artifact_root: str | Path | None,
        run_id: int,
        requested_workers: int,
        mode: str,
        budget: W02MorphologyOverlayBudget | None = None,
        ) -> W02MorphologyOverlayRunResult:
    """运行全规模开发预演；正式 transform 计数和 guard 均保持为零。"""
    return _run_w02_morphology_overlay_nonformal(
        fixture_root=development_root,
        candidate_artifact_root=candidate_artifact_root,
        run_id=run_id,
        requested_workers=requested_workers,
        mode=mode,
        run_scope="DEVELOPMENT_PREFLIGHT",
        budget=budget,
        fault_after_shard=None,
    )


def _run_w02_morphology_overlay_nonformal(
        *,
        fixture_root: str | Path,
        candidate_artifact_root: str | Path | None,
        run_id: int,
        requested_workers: int,
        mode: str,
        run_scope: str,
        budget: W02MorphologyOverlayBudget | None,
        fault_after_shard: int | None,
        ) -> W02MorphologyOverlayRunResult:
    """复用 fixture/dev 的非正式事务，运行域进入 identity 和 manifest。"""
    if type(requested_workers) is not int or requested_workers not in (1, 2, 4):
        raise W02MorphologyOverlayError("overlay requested_workers 必须是 1/2/4")
    if mode not in {"fresh", "restart", "resume"}:
        raise W02MorphologyOverlayError("overlay fixture mode 未注册")
    if run_scope not in {"PUBLIC_SYNTHETIC_FIXTURE", "DEVELOPMENT_PREFLIGHT"}:
        raise W02MorphologyOverlayError("overlay nonformal run scope 非法")
    budget = W02MorphologyOverlayBudget() if budget is None else budget
    root = Path(fixture_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    store, staging, final = _run_dirs(root, run_id)
    store.mkdir(exist_ok=True)
    if mode == "fresh":
        if candidate_artifact_root is None or staging.exists() or final.exists():
            raise W02MorphologyOverlayError("fresh overlay fixture 输入或路径非法")
        candidate = read_w02_candidate_artifact(candidate_artifact_root)
        identity = W02MorphologyOverlayRunIdentity(
            "PH2-D03-V2", "W-02", run_scope, W02_MORPH_OVERLAY_OPERATION,
            W02_MORPH_SUCCESSOR_VERSION,
            hashlib.sha256(
                (run_scope + "-MORPHOLOGY-OVERLAY-RUNTIME").encode("ascii")
            ).hexdigest(),
            candidate.artifact_manifest_sha256,
            candidate.candidate_semantic_sha256,
            run_id,
        )
        staging.mkdir()
        (staging / W02_MORPH_OVERLAY_SHARD_DIR).mkdir()
        _run_state(staging, identity, "RUNNING")
        meta = _write_input_spool(
            staging / W02_MORPH_OVERLAY_SPOOL_NAME,
            candidate.artifact_path, identity, budget)
    else:
        identity = _identity_from_existing(staging if mode == "restart" else final)
        if identity.run_scope != run_scope:
            raise W02MorphologyOverlayError("overlay existing run scope 漂移")
        if mode == "restart":
            if candidate_artifact_root is not None or not staging.is_dir() or final.exists():
                raise W02MorphologyOverlayError("restart overlay fixture 路径非法")
            meta = _read_spool_meta(
                staging / W02_MORPH_OVERLAY_SPOOL_NAME, identity)
        else:
            if candidate_artifact_root is not None or not final.is_dir() or staging.exists():
                raise W02MorphologyOverlayError("resume overlay fixture 路径非法")
            return read_w02_morphology_overlay_artifact(
                final, requested_workers=requested_workers)
    return _execute_pipeline(
        staging=staging, final=final, identity=identity,
        expected_input_rows=int(meta["input_row_count"]),
        requested_workers=requested_workers, mode=mode, budget=budget,
        fault_after_shard=fault_after_shard)


def _identity_from_value(value: object) -> W02MorphologyOverlayRunIdentity:
    if not isinstance(value, dict):
        raise W02MorphologyOverlayError("overlay run identity 非 object")
    try:
        return W02MorphologyOverlayRunIdentity(
            str(value["release_key"]), str(value["stage_key"]),
            str(value["run_scope"]), str(value["operation_kind"]),
            str(value["successor_version"]),
            str(value["runtime_freeze_sha256"]),
            str(value["parent_candidate_manifest_sha256"]),
            str(value["parent_candidate_semantic_sha256"]),
            int(value["run_id"]), int(value["logical_shard_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise W02MorphologyOverlayError(
            "overlay run identity 无法回读") from error


def _identity_from_existing(root: Path) -> W02MorphologyOverlayRunIdentity:
    if not root.is_dir():
        raise W02MorphologyOverlayError("overlay existing root 缺失")
    manifest = root / W02_MORPH_OVERLAY_MANIFEST_NAME
    if manifest.is_file():
        value = read_canonical_object(manifest)
        return _identity_from_value(value.get("run_identity"))
    states = tuple(sorted((root / W02_MORPH_OVERLAY_STATE_DIR).glob("*.json")))
    if not states:
        raise W02MorphologyOverlayError("overlay run state 缺失")
    spool = _connect(root / W02_MORPH_OVERLAY_SPOOL_NAME, read_only=True)
    meta = {
        str(row[0]): json.loads(str(row[1]))
        for row in spool.execute("SELECT key,value_json FROM input_meta")
    }
    spool.close()
    identity = _identity_from_value(meta.get("run_identity"))
    if identity.sha256() != meta.get("run_identity_sha256"):
        raise W02MorphologyOverlayError("overlay existing identity 漂移")
    return identity


def _read_overlay_index(
        db_path: Path,
        meta: dict[str, Any],
        ) -> W02MorphologySuccessorIndex:
    connection = _connect(db_path, read_only=True)
    try:
        index = _index_from_combo_tables(
            connection,
            max_form_length=int(meta["max_form_length"]),
            training_pair_count=int(meta["training_pair_count"]),
            morphology_observation_count=int(
                meta["morphology_observation_count"]),
            morphology_token_count=int(meta["morphology_token_count"]),
            logic_operations=int(meta["logic_operations"]),
        )
        events = tuple(str(row[0]) for row in connection.execute(
            "SELECT event_kind FROM run_events ORDER BY event_seq"))
        checkpoints = int(connection.execute(
            "SELECT COUNT(*) FROM checkpoints").fetchone()[0])
    finally:
        connection.close()
    if events != W02_MORPH_OVERLAY_EVENTS:
        raise W02MorphologyOverlayError("overlay DB event sequence 漂移")
    if checkpoints != W02_MORPH_OVERLAY_SHARD_COUNT:
        raise W02MorphologyOverlayError("overlay DB checkpoint count 漂移")
    return index


def read_w02_morphology_overlay_artifact(
        artifact_root: str | Path,
        *,
        requested_workers: int = 1,
        ) -> W02MorphologyOverlayRunResult:
    """严格回读 sealed overlay，并证明回读过程零写。"""
    if type(requested_workers) is not int or requested_workers not in (1, 2, 4):
        raise W02MorphologyOverlayError("overlay readback worker 非法")
    root = Path(artifact_root).resolve()
    manifest_path = root / W02_MORPH_OVERLAY_MANIFEST_NAME
    if not manifest_path.is_file():
        raise W02MorphologyOverlayError("overlay artifact manifest 缺失")
    before = _tree_inventory(root, skip_manifest=False)
    value = read_canonical_object(manifest_path)
    required = {
        "artifact_kind", "artifact_version", "candidate_writes",
        "formal_private_evaluation_runs", "formal_successor_transform_runs",
        "formal_training_runs", "input_roots", "input_row_count",
        "logic_operations", "logical_shard_count",
        "morphology_observation_count", "morphology_token_count",
        "operation_kind", "overlay_semantic_sha256", "overlay_writes",
        "parent_candidate_manifest_sha256",
        "parent_candidate_semantic_sha256", "private_payload_reads",
        "rule_row_count", "run_identity", "run_identity_sha256",
        "run_scope", "status", "teacher_calls", "training_pair_count",
        "tree", "visible_splits", "worker_counts_supported",
    }
    if set(value) != required:
        raise W02MorphologyOverlayError("overlay artifact manifest 字段漂移")
    if (value["artifact_kind"] != "PH2_D03_V2_W02_MORPHOLOGY_OVERLAY"
            or value["artifact_version"] != W02_MORPH_OVERLAY_VERSION
            or value["operation_kind"] != W02_MORPH_OVERLAY_OPERATION
            or value["status"] != "MORPHOLOGY_OVERLAY_SEALED"
            or value["run_scope"] not in {
                "PUBLIC_SYNTHETIC_FIXTURE", "DEVELOPMENT_PREFLIGHT"}
            or value["worker_counts_supported"] != [1, 2, 4]
            or value["logical_shard_count"] != W02_MORPH_OVERLAY_SHARD_COUNT
            or value["input_roots"] != ["PARENT_CANDIDATE_ARTIFACT"]
            or value["visible_splits"] != [
                "train-derived-candidate-statistics"]
            or any(value[key] != 0 for key in (
                "candidate_writes", "formal_private_evaluation_runs",
                "formal_successor_transform_runs", "formal_training_runs",
                "private_payload_reads", "teacher_calls"))
            or value["overlay_writes"] != 1):
        raise W02MorphologyOverlayError("overlay artifact 状态非法")
    identity = _identity_from_value(value["run_identity"])
    if (value["run_identity"] != identity.to_dict()
            or value["run_identity_sha256"] != identity.sha256()):
        raise W02MorphologyOverlayError("overlay artifact run identity 漂移")
    if value["tree"] != _tree_inventory(root, skip_manifest=True):
        raise W02MorphologyOverlayError("overlay artifact tree 漂移")
    db_path = root / W02_MORPH_OVERLAY_DB_NAME
    connection = _connect(db_path, read_only=True)
    meta = {
        str(row[0]): json.loads(str(row[1]))
        for row in connection.execute(
            "SELECT key,value_json FROM meta ORDER BY key")
    }
    connection.close()
    required_meta = {
        "artifact_version", "logic_operations", "max_form_length",
        "morphology_observation_count", "morphology_token_count",
        "overlay_semantic_sha256", "parent_candidate_manifest_sha256",
        "parent_candidate_semantic_sha256", "rule_row_count",
        "run_identity_sha256", "training_pair_count",
    }
    if set(meta) != required_meta:
        raise W02MorphologyOverlayError("overlay artifact DB meta 字段漂移")
    index = _read_overlay_index(db_path, meta)
    if (index.semantic_sha256 != value["overlay_semantic_sha256"]
            or meta.get("overlay_semantic_sha256") != index.semantic_sha256
            or index.row_count != value["rule_row_count"]
            or index.logic_operations != value["logic_operations"]
            or index.training_pair_count != value["training_pair_count"]
            or identity.parent_candidate_semantic_sha256
            != value["parent_candidate_semantic_sha256"]
            or identity.parent_candidate_manifest_sha256
            != value["parent_candidate_manifest_sha256"]
            or meta["parent_candidate_manifest_sha256"]
            != identity.parent_candidate_manifest_sha256
            or meta["parent_candidate_semantic_sha256"]
            != identity.parent_candidate_semantic_sha256
            or meta["run_identity_sha256"] != identity.sha256()):
        raise W02MorphologyOverlayError("overlay artifact semantic 漂移")
    if _tree_inventory(root, skip_manifest=False) != before:
        raise W02MorphologyOverlayError("overlay artifact readback 产生写入")
    return W02MorphologyOverlayRunResult(
        "resume", identity.sha256(), root,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        index.semantic_sha256, identity.parent_candidate_semantic_sha256,
        index.training_pair_count, index.morphology_observation_count,
        index.morphology_token_count, index.logic_operations,
        index.row_count, W02_MORPH_OVERLAY_SHARD_COUNT, requested_workers,
        0, 0, 0, 1,
    )


def load_w02_morphology_overlay_index(
        artifact_root: str | Path,
        ) -> W02MorphologySuccessorIndex:
    """在严格 artifact 回读后加载不可变 morphology 索引。"""
    result = read_w02_morphology_overlay_artifact(artifact_root)
    connection = _connect(
        result.artifact_path / W02_MORPH_OVERLAY_DB_NAME, read_only=True)
    meta = {
        str(row[0]): json.loads(str(row[1]))
        for row in connection.execute("SELECT key,value_json FROM meta ORDER BY key")
    }
    connection.close()
    return _read_overlay_index(
        result.artifact_path / W02_MORPH_OVERLAY_DB_NAME, meta)


__all__ = [
    "W02MorphologyOverlayBudget",
    "W02MorphologyOverlayError",
    "W02MorphologyOverlayRunIdentity",
    "W02MorphologyOverlayRunResult",
    "derive_w02_morphology_successor_from_candidate",
    "load_w02_morphology_overlay_index",
    "read_w02_morphology_overlay_artifact",
    "run_w02_morphology_overlay_development",
    "run_w02_morphology_overlay_fixture",
]
