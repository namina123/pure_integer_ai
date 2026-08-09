"""W-02 morphology successor V2 edge-edit overlay artifact runtime。"""
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
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    load_w02_morphology_overlay_index,
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_SUCCESSOR_V2_VERSION,
    W02MorphologySuccessorV2Index,
    derive_w02_morphology_successor_v2_from_candidate,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


W02_MORPH_V2_OVERLAY_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-OVERLAY-V1")
W02_MORPH_V2_OVERLAY_SHARD_COUNT = 128
W02_MORPH_V2_OVERLAY_STORE = "morphology-v2-overlay-store"
W02_MORPH_V2_OVERLAY_MANIFEST = "morphology-v2-overlay.artifact.json"
W02_MORPH_V2_OVERLAY_DB = "morphology-v2-overlay.sqlite3"
W02_MORPH_V2_OVERLAY_SPOOL = "input-spool.jsonl.gz"
W02_MORPH_V2_OVERLAY_SHARDS = "shards"
W02_MORPH_V2_OVERLAY_CHECKPOINTS = "checkpoints.jsonl"
W02_MORPH_V2_OVERLAY_STATE = "run-state.json"


# object-model: exception
class W02MorphologySuccessorV2OverlayError(RuntimeError):
    """V2 overlay transaction、artifact、资源或父身份发生漂移。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2OverlayBudget:
    """小型 edge overlay 的纯整数资源上界。"""

    max_input_rows: int = 20_000
    max_rule_rows: int = 2_000
    max_logic_operations: int = 9_000_000
    max_payload_bytes: int = 16_777_216
    max_shard_delta_bytes: int = 1_048_576
    max_checkpoint_count: int = W02_MORPH_V2_OVERLAY_SHARD_COUNT

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (
                self.max_input_rows, self.max_rule_rows,
                self.max_logic_operations, self.max_payload_bytes,
                self.max_shard_delta_bytes, self.max_checkpoint_count)):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay budget 必须为正整数")
        if self.max_checkpoint_count < W02_MORPH_V2_OVERLAY_SHARD_COUNT:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay checkpoint budget 不足")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2OverlayIdentity:
    """不含绝对路径和物理 worker 数的 V2 transform identity。"""

    run_scope: str
    run_id: int
    parent_candidate_manifest_sha256: str
    parent_candidate_semantic_sha256: str
    parent_v1_overlay_manifest_sha256: str
    parent_v1_overlay_semantic_sha256: str
    logical_shard_count: int = W02_MORPH_V2_OVERLAY_SHARD_COUNT

    def __post_init__(self) -> None:
        if self.run_scope not in {"PUBLIC_SYNTHETIC_FIXTURE", "DEVELOPMENT_PREFLIGHT"}:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay run scope 未注册")
        if type(self.run_id) is not int or self.run_id <= 0:
            raise W02MorphologySuccessorV2OverlayError("V2 overlay run_id 非法")
        for value in (
                self.parent_candidate_manifest_sha256,
                self.parent_candidate_semantic_sha256,
                self.parent_v1_overlay_manifest_sha256,
                self.parent_v1_overlay_semantic_sha256):
            _strict_sha(value)
        if self.logical_shard_count != W02_MORPH_V2_OVERLAY_SHARD_COUNT:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay shard count 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_shard_count": self.logical_shard_count,
            "operation_kind": "CANDIDATE_DERIVED_EDGE_LEMMA_TRANSFORM",
            "parent_candidate_manifest_sha256":
                self.parent_candidate_manifest_sha256,
            "parent_candidate_semantic_sha256":
                self.parent_candidate_semantic_sha256,
            "parent_v1_overlay_manifest_sha256":
                self.parent_v1_overlay_manifest_sha256,
            "parent_v1_overlay_semantic_sha256":
                self.parent_v1_overlay_semantic_sha256,
            "release_key": "PH2-D03-V2",
            "run_id": self.run_id,
            "run_scope": self.run_scope,
            "stage_key": "W-02",
            "successor_version": W02_MORPH_SUCCESSOR_V2_VERSION,
        }

    def sha256(self) -> str:
        return _hash_value(self.to_dict())


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2OverlayResult:
    """fresh/restart/resume 的严格只读结果。"""

    mode: str
    artifact_path: Path
    artifact_manifest_sha256: str
    run_identity_sha256: str
    semantic_sha256: str
    rule_row_count: int
    logic_operations: int
    accepted_lexeme_rows: int
    accepted_support_count: int
    unsupported_lexeme_rows: int
    unsupported_support_count: int
    shard_count: int
    requested_workers: int

    def __post_init__(self) -> None:
        if self.mode not in {"fresh", "restart", "resume"}:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay result mode 未注册")
        if not isinstance(self.artifact_path, Path):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay artifact path 类型错误")
        for value in (
                self.artifact_manifest_sha256, self.run_identity_sha256,
                self.semantic_sha256):
            _strict_sha(value)
        if any(type(value) is not int or value < 0 for value in (
                self.rule_row_count, self.logic_operations,
                self.accepted_lexeme_rows, self.accepted_support_count,
                self.unsupported_lexeme_rows, self.unsupported_support_count,
                self.shard_count, self.requested_workers)):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay result 计数非法")


def _strict_sha(value: object) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV2OverlayError("V2 overlay SHA 非法")
    return value


def _hash_value(value: object) -> str:
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
    for target in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = target.relative_to(root).as_posix()
        if skip_manifest and relative == W02_MORPH_V2_OVERLAY_MANIFEST:
            continue
        size, digest = _sha256_file(target)
        rows.append({"path": relative, "sha256": digest, "size_bytes": size})
    return rows


def _write_gzip(path: Path, rows: list[dict[str, object]]) -> tuple[int, str]:
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for row in rows:
                stream.write(canonical_json_bytes(row) + b"\n")
    return _sha256_file(path)


def _read_gzip(path: Path) -> Iterator[dict[str, object]]:
    try:
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        raise W02MorphologySuccessorV2OverlayError(
                            "V2 overlay gzip 换行非法")
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    yield value
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02MorphologySuccessorV2OverlayError):
            raise
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay gzip/JSONL 损坏") from error


def _run_dirs(root: Path, run_id: int) -> tuple[Path, Path, Path]:
    store = root / W02_MORPH_V2_OVERLAY_STORE
    staging = store / f".run-{run_id:06d}.staging"
    final = store / f"run-{run_id:06d}"
    return store, staging, final


def _state(staging: Path, identity: W02MorphologySuccessorV2OverlayIdentity,
           status: str) -> None:
    target = staging / W02_MORPH_V2_OVERLAY_STATE
    value = {
        "run_identity_sha256": identity.sha256(),
        "status": status,
    }
    if target.exists():
        current = read_canonical_object(target)
        if (current.get("run_identity_sha256") != identity.sha256()
                or current.get("status") not in {"RUNNING", "FAILED_RECOVERABLE"}):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay run state 漂移")
        target.unlink()
    write_immutable_json(value, target)


def _spool_value(index: W02MorphologySuccessorV2Index,
                 identity: W02MorphologySuccessorV2OverlayIdentity) -> list[dict[str, object]]:
    return [{
        "accepted_lexeme_rows": index.accepted_lexeme_rows,
        "accepted_support_count": index.accepted_support_count,
        "logic_operations": index.logic_operations,
        "max_form_length": index.max_form_length,
        "row_count": index.row_count,
        "row_kind": "V2_OVERLAY_META",
        "run_identity": identity.to_dict(),
        "run_identity_sha256": identity.sha256(),
        "semantic_sha256": index.semantic_sha256,
        "unsupported_lexeme_rows": index.unsupported_lexeme_rows,
        "unsupported_support_count": index.unsupported_support_count,
    }, *index.semantic_rows()]


def _index_from_spool(rows: list[dict[str, object]]) -> W02MorphologySuccessorV2Index:
    if not rows or rows[0].get("row_kind") != "V2_OVERLAY_META":
        raise W02MorphologySuccessorV2OverlayError("V2 overlay spool meta 缺失")
    meta = rows[0]
    routes = []
    global_counts = {}
    local_counts: dict[tuple[str, str], dict[tuple[str, str, str], int]] = {}
    for row in rows[1:]:
        kind = row.get("row_kind")
        if kind == "DATASET_ROUTE":
            routes.append(tuple(int(value) for value in row["dataset_key"]))
        elif kind == "GLOBAL_EDGE_COMBO":
            global_counts[(str(row["lemma_rule"]), str(row["upos"]),
                           str(row["feats_json"]))] = int(row["count"])
        elif kind == "LOCAL_EDGE_COMBO":
            feature = (str(row["feature_kind"]), str(row["feature_value"]))
            combo = (str(row["lemma_rule"]), str(row["upos"]),
                     str(row["feats_json"]))
            local_counts.setdefault(feature, {})[combo] = int(row["count"])
        else:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay spool row kind 漂移")
    index = W02MorphologySuccessorV2Index(
        tuple(routes), global_counts, local_counts,
        int(meta["max_form_length"]), int(meta["accepted_lexeme_rows"]),
        int(meta["accepted_support_count"]),
        int(meta["unsupported_lexeme_rows"]),
        int(meta["unsupported_support_count"]), int(meta["logic_operations"]),
        str(meta["semantic_sha256"]), int(meta["row_count"]))
    if index.semantic_rows() != tuple(rows[1:]):
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay spool semantic rows 漂移")
    return index


def _shard_index(row: dict[str, object]) -> int:
    return int(_hash_value(row)[:16], 16) % W02_MORPH_V2_OVERLAY_SHARD_COUNT


def _validate_shard(directory: Path, shard_index: int,
                    expected_rows: list[dict[str, object]],
                    identity_sha: str) -> dict[str, object]:
    delta = directory / f"{shard_index:03d}.delta.jsonl.gz"
    manifest = directory / f"{shard_index:03d}.delta.manifest.json"
    if not delta.is_file() or not manifest.is_file():
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay shard pair 缺失")
    value = read_canonical_object(manifest)
    size, digest = _sha256_file(delta)
    ordered = sorted(expected_rows, key=canonical_json_bytes)
    if (value.get("shard_index") != shard_index
            or value.get("run_identity_sha256") != identity_sha
            or value.get("row_count") != len(ordered)
            or value.get("transport_size_bytes") != size
            or value.get("transport_sha256") != digest):
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay shard transport 漂移")
    if list(_read_gzip(delta)) != ordered:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay shard semantic 漂移")
    return value


def _write_shard(directory: Path, shard_index: int,
                 rows: list[dict[str, object]], identity_sha: str,
                 budget: W02MorphologySuccessorV2OverlayBudget) -> dict[str, object]:
    delta = directory / f"{shard_index:03d}.delta.jsonl.gz"
    manifest = directory / f"{shard_index:03d}.delta.manifest.json"
    if delta.is_file() and manifest.is_file():
        return _validate_shard(directory, shard_index, rows, identity_sha)
    if delta.exists() or manifest.exists():
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay partial shard 不可恢复")
    ordered = sorted(rows, key=canonical_json_bytes)
    size, digest = _write_gzip(delta, ordered)
    if size > budget.max_shard_delta_bytes:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay shard payload resource stop")
    value = {
        "row_count": len(ordered),
        "run_identity_sha256": identity_sha,
        "shard_index": shard_index,
        "transport_sha256": digest,
        "transport_size_bytes": size,
    }
    write_immutable_json(value, manifest)
    return value


def _read_checkpoints(path: Path, identity_sha: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_bytes().splitlines():
        value = parse_canonical_json_bytes(line, require_object=True)
        assert isinstance(value, dict)
        index = int(value.get("shard_index", -1))
        if (value.get("run_identity_sha256") != identity_sha
                or index != len(result)):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay checkpoint 顺序漂移")
        result[index] = value
    return result


def _append_checkpoint(path: Path, value: dict[str, object]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _create_db(path: Path, index: W02MorphologySuccessorV2Index,
               identity: W02MorphologySuccessorV2OverlayIdentity,
               checkpoints: dict[int, dict[str, object]]) -> None:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
        CREATE TABLE dataset_routes(dataset_key BLOB PRIMARY KEY);
        CREATE TABLE global_edge_combos(
            lemma_rule TEXT NOT NULL,upos TEXT NOT NULL,feats_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(lemma_rule,upos,feats_json));
        CREATE TABLE local_edge_combos(
            feature_kind TEXT NOT NULL,feature_value TEXT NOT NULL,
            lemma_rule TEXT NOT NULL,upos TEXT NOT NULL,feats_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(feature_kind,feature_value,lemma_rule,upos,feats_json));
        CREATE TABLE checkpoints(
            checkpoint_seq INTEGER PRIMARY KEY,shard_index INTEGER NOT NULL UNIQUE,
            checkpoint_sha256 TEXT NOT NULL);
    """)
    meta = {
        "accepted_lexeme_rows": index.accepted_lexeme_rows,
        "accepted_support_count": index.accepted_support_count,
        "artifact_version": W02_MORPH_V2_OVERLAY_VERSION,
        "logic_operations": index.logic_operations,
        "max_form_length": index.max_form_length,
        "row_count": index.row_count,
        "run_identity_sha256": identity.sha256(),
        "semantic_sha256": index.semantic_sha256,
        "unsupported_lexeme_rows": index.unsupported_lexeme_rows,
        "unsupported_support_count": index.unsupported_support_count,
    }
    connection.executemany(
        "INSERT INTO meta VALUES(?,?)",
        ((key, canonical_json_bytes(value).decode("utf-8"))
         for key, value in sorted(meta.items())))
    connection.executemany(
        "INSERT INTO dataset_routes VALUES(?)",
        ((canonical_json_bytes(list(key)),) for key in index.dataset_keys))
    connection.executemany(
        "INSERT INTO global_edge_combos VALUES(?,?,?,?)",
        ((*combo, count) for combo, count in sorted(index.global_counts.items())))
    connection.executemany(
        "INSERT INTO local_edge_combos VALUES(?,?,?,?,?,?)",
        ((feature[0], feature[1], combo[0], combo[1], combo[2], count)
         for feature, values in sorted(index.feature_counts.items())
         for combo, count in sorted(values.items())))
    connection.executemany(
        "INSERT INTO checkpoints VALUES(?,?,?)",
        ((position + 1, position, _hash_value(checkpoints[position]))
         for position in range(W02_MORPH_V2_OVERLAY_SHARD_COUNT)))
    connection.commit()
    connection.close()


def _publish_directory(staging: Path, final: Path) -> None:
    last: PermissionError | None = None
    for _attempt in range(8):
        try:
            os.replace(staging, final)
            return
        except PermissionError as error:
            last = error
            gc.collect()
    raise W02MorphologySuccessorV2OverlayError(
        "V2 overlay final directory publish failed") from last


def _identity_from_manifest(value: dict[str, object]) -> W02MorphologySuccessorV2OverlayIdentity:
    raw = value.get("run_identity")
    if not isinstance(raw, dict):
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay run identity 缺失")
    return W02MorphologySuccessorV2OverlayIdentity(
        str(raw.get("run_scope")), int(raw.get("run_id", 0)),
        str(raw.get("parent_candidate_manifest_sha256")),
        str(raw.get("parent_candidate_semantic_sha256")),
        str(raw.get("parent_v1_overlay_manifest_sha256")),
        str(raw.get("parent_v1_overlay_semantic_sha256")))


def _index_from_db(path: Path) -> W02MorphologySuccessorV2Index:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    meta = {str(key): json.loads(str(value)) for key, value in
            connection.execute("SELECT key,value_json FROM meta ORDER BY key")}
    routes = tuple(tuple(parse_canonical_json_bytes(
        bytes(row[0]), require_object=False)) for row in
                   connection.execute("SELECT dataset_key FROM dataset_routes ORDER BY dataset_key"))
    global_counts = {(str(a), str(b), str(c)): int(d) for a, b, c, d in
                     connection.execute("SELECT * FROM global_edge_combos ORDER BY lemma_rule,upos,feats_json")}
    local: dict[tuple[str, str], dict[tuple[str, str, str], int]] = {}
    for fk, fv, rule, upos, feats, count in connection.execute(
            "SELECT * FROM local_edge_combos ORDER BY feature_kind,feature_value,lemma_rule,upos,feats_json"):
        local.setdefault((str(fk), str(fv)), {})[(str(rule), str(upos), str(feats))] = int(count)
    checkpoint_count = int(connection.execute(
        "SELECT COUNT(*) FROM checkpoints").fetchone()[0])
    connection.close()
    if checkpoint_count != W02_MORPH_V2_OVERLAY_SHARD_COUNT:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay DB checkpoints 漂移")
    return W02MorphologySuccessorV2Index(
        routes, global_counts, local, int(meta["max_form_length"]),
        int(meta["accepted_lexeme_rows"]), int(meta["accepted_support_count"]),
        int(meta["unsupported_lexeme_rows"]),
        int(meta["unsupported_support_count"]), int(meta["logic_operations"]),
        str(meta["semantic_sha256"]), int(meta["row_count"]))


def read_w02_morphology_successor_v2_overlay_artifact(
        artifact_root: str | Path, *, requested_workers: int = 1,
        ) -> W02MorphologySuccessorV2OverlayResult:
    """严格回读 sealed V2 overlay，并证明回读零写。"""
    if requested_workers not in {1, 2, 4}:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay readback workers 非法")
    root = Path(artifact_root).resolve()
    manifest_path = root / W02_MORPH_V2_OVERLAY_MANIFEST
    if not manifest_path.is_file():
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay manifest 缺失")
    before = _tree_inventory(root, skip_manifest=False)
    value = read_canonical_object(manifest_path)
    identity = _identity_from_manifest(value)
    if (value.get("artifact_version") != W02_MORPH_V2_OVERLAY_VERSION
            or value.get("status") != "MORPHOLOGY_V2_OVERLAY_SEALED"
            or value.get("run_identity") != identity.to_dict()
            or value.get("run_identity_sha256") != identity.sha256()
            or value.get("logical_shard_count")
            != W02_MORPH_V2_OVERLAY_SHARD_COUNT
            or value.get("worker_counts_supported") != [1, 2, 4]
            or value.get("formal_training_runs") != 0
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("candidate_writes") != 0
            or value.get("v1_overlay_writes") != 0
            or value.get("v2_overlay_writes") != 1
            or value.get("tree") != _tree_inventory(root, skip_manifest=True)):
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay manifest 状态或 tree 漂移")
    index = _index_from_db(root / W02_MORPH_V2_OVERLAY_DB)
    if (value.get("semantic_sha256") != index.semantic_sha256
            or value.get("rule_row_count") != index.row_count
            or value.get("logic_operations") != index.logic_operations):
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay DB semantic 漂移")
    if _tree_inventory(root, skip_manifest=False) != before:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay readback 产生写入")
    return W02MorphologySuccessorV2OverlayResult(
        "resume", root, hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        identity.sha256(), index.semantic_sha256, index.row_count,
        index.logic_operations, index.accepted_lexeme_rows,
        index.accepted_support_count, index.unsupported_lexeme_rows,
        index.unsupported_support_count, W02_MORPH_V2_OVERLAY_SHARD_COUNT,
        requested_workers)


def load_w02_morphology_successor_v2_overlay_index(
        artifact_root: str | Path) -> W02MorphologySuccessorV2Index:
    """严格 artifact 回读后加载 edge index。"""
    result = read_w02_morphology_successor_v2_overlay_artifact(artifact_root)
    return _index_from_db(result.artifact_path / W02_MORPH_V2_OVERLAY_DB)


def _run(
        *, root: Path, candidate_artifact_root: str | Path | None,
        v1_overlay_artifact_root: str | Path | None, run_id: int,
        requested_workers: int, mode: str, run_scope: str,
        budget: W02MorphologySuccessorV2OverlayBudget,
        fault_after_shard: int | None,
        ) -> W02MorphologySuccessorV2OverlayResult:
    if requested_workers not in {1, 2, 4} or mode not in {"fresh", "restart", "resume"}:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay workers/mode 非法")
    store, staging, final = _run_dirs(root.resolve(), run_id)
    if final.is_dir():
        if mode != "resume":
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay sealed artifact 不得重建")
        return read_w02_morphology_successor_v2_overlay_artifact(
            final, requested_workers=requested_workers)
    if mode == "fresh":
        if store.exists() or staging.exists():
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay fresh root 已存在")
        if candidate_artifact_root is None or v1_overlay_artifact_root is None:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay fresh parent 缺失")
        candidate_root = Path(candidate_artifact_root).resolve()
        v1_root = Path(v1_overlay_artifact_root).resolve()
        before_candidate = _tree_inventory(candidate_root, skip_manifest=False)
        before_v1 = _tree_inventory(v1_root, skip_manifest=False)
        candidate = read_w02_candidate_artifact(candidate_root)
        v1 = read_w02_morphology_overlay_artifact(v1_root)
        v1_index = load_w02_morphology_overlay_index(v1_root)
        index = derive_w02_morphology_successor_v2_from_candidate(
            candidate_root, v1_index)
        identity = W02MorphologySuccessorV2OverlayIdentity(
            run_scope, run_id, candidate.artifact_manifest_sha256,
            candidate.candidate_semantic_sha256, v1.artifact_manifest_sha256,
            v1.overlay_semantic_sha256)
        if (index.row_count > budget.max_rule_rows
                or index.logic_operations > budget.max_logic_operations):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay formation resource stop")
        store.mkdir(parents=True)
        staging.mkdir()
        (staging / W02_MORPH_V2_OVERLAY_SHARDS).mkdir()
        _state(staging, identity, "RUNNING")
        spool_rows = _spool_value(index, identity)
        if len(spool_rows) - 1 > budget.max_input_rows:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay input row resource stop")
        spool_size, _spool_sha = _write_gzip(
            staging / W02_MORPH_V2_OVERLAY_SPOOL, spool_rows)
        if spool_size > budget.max_payload_bytes:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay spool resource stop")
        if (_tree_inventory(candidate_root, skip_manifest=False) != before_candidate
                or _tree_inventory(v1_root, skip_manifest=False) != before_v1):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay fresh 写入父 artifact")
    else:
        if not staging.is_dir() or candidate_artifact_root is not None or v1_overlay_artifact_root is not None:
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay restart/resume state 非法")
        state = read_canonical_object(staging / W02_MORPH_V2_OVERLAY_STATE)
        spool_rows = list(_read_gzip(staging / W02_MORPH_V2_OVERLAY_SPOOL))
        index = _index_from_spool(spool_rows)
        identity = _identity_from_manifest(spool_rows[0])
        if (spool_rows[0].get("run_identity") != identity.to_dict()
                or spool_rows[0].get("run_identity_sha256") != identity.sha256()
                or state.get("run_identity_sha256") != identity.sha256()
                or identity.run_id != run_id
                or identity.run_scope != run_scope):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay restart state identity 漂移")
        _state(staging, identity, "RUNNING")
    spool_rows = list(_read_gzip(staging / W02_MORPH_V2_OVERLAY_SPOOL))
    index = _index_from_spool(spool_rows)
    grouped = {number: [] for number in range(W02_MORPH_V2_OVERLAY_SHARD_COUNT)}
    for row in index.semantic_rows():
        grouped[_shard_index(row)].append(row)
    checkpoint_path = staging / W02_MORPH_V2_OVERLAY_CHECKPOINTS
    completed = _read_checkpoints(checkpoint_path, identity.sha256())
    directory = staging / W02_MORPH_V2_OVERLAY_SHARDS
    for number, checkpoint in completed.items():
        shard = _validate_shard(
            directory, number, grouped[number], identity.sha256())
        if checkpoint.get("shard_manifest_sha256") != _hash_value(shard):
            raise W02MorphologySuccessorV2OverlayError(
                "V2 overlay checkpoint shard 漂移")
    futures = {}
    with ThreadPoolExecutor(max_workers=requested_workers) as executor:
        for number in range(W02_MORPH_V2_OVERLAY_SHARD_COUNT):
            if number not in completed:
                futures[number] = executor.submit(
                    _write_shard, directory, number, grouped[number],
                    identity.sha256(), budget)
        for number in range(W02_MORPH_V2_OVERLAY_SHARD_COUNT):
            if number in completed:
                continue
            shard = futures[number].result()
            checkpoint = {
                "run_identity_sha256": identity.sha256(),
                "shard_index": number,
                "shard_manifest_sha256": _hash_value(shard),
            }
            _append_checkpoint(checkpoint_path, checkpoint)
            completed[number] = checkpoint
            if fault_after_shard == number:
                _state(staging, identity, "FAILED_RECOVERABLE")
                raise W02MorphologySuccessorV2OverlayError(
                    "injected V2 overlay shard fault")
    if len(completed) != W02_MORPH_V2_OVERLAY_SHARD_COUNT:
        raise W02MorphologySuccessorV2OverlayError(
            "V2 overlay checkpoint completion 不闭合")
    _create_db(staging / W02_MORPH_V2_OVERLAY_DB, index, identity, completed)
    (staging / W02_MORPH_V2_OVERLAY_SPOOL).unlink()
    _state(staging, identity, "SEALED")
    manifest = {
        "accepted_lexeme_rows": index.accepted_lexeme_rows,
        "accepted_support_count": index.accepted_support_count,
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_OVERLAY",
        "artifact_version": W02_MORPH_V2_OVERLAY_VERSION,
        "candidate_writes": 0,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 0,
        "logic_operations": index.logic_operations,
        "logical_shard_count": W02_MORPH_V2_OVERLAY_SHARD_COUNT,
        "private_payload_reads": 0,
        "rule_row_count": index.row_count,
        "run_identity": identity.to_dict(),
        "run_identity_sha256": identity.sha256(),
        "semantic_sha256": index.semantic_sha256,
        "status": "MORPHOLOGY_V2_OVERLAY_SEALED",
        "teacher_calls": 0,
        "tree": _tree_inventory(staging, skip_manifest=True),
        "unsupported_lexeme_rows": index.unsupported_lexeme_rows,
        "unsupported_support_count": index.unsupported_support_count,
        "v1_overlay_writes": 0,
        "v2_overlay_writes": 1,
        "worker_counts_supported": [1, 2, 4],
    }
    write_immutable_json(manifest, staging / W02_MORPH_V2_OVERLAY_MANIFEST)
    _publish_directory(staging, final)
    result = read_w02_morphology_successor_v2_overlay_artifact(
        final, requested_workers=requested_workers)
    return W02MorphologySuccessorV2OverlayResult(
        mode, result.artifact_path, result.artifact_manifest_sha256,
        result.run_identity_sha256, result.semantic_sha256,
        result.rule_row_count, result.logic_operations,
        result.accepted_lexeme_rows, result.accepted_support_count,
        result.unsupported_lexeme_rows, result.unsupported_support_count,
        result.shard_count, requested_workers)


def run_w02_morphology_successor_v2_overlay_fixture(
        *, fixture_root: str | Path,
        candidate_artifact_root: str | Path | None,
        v1_overlay_artifact_root: str | Path | None,
        run_id: int, requested_workers: int, mode: str,
        budget: W02MorphologySuccessorV2OverlayBudget | None = None,
        fault_after_shard: int | None = None,
        ) -> W02MorphologySuccessorV2OverlayResult:
    return _run(
        root=Path(fixture_root),
        candidate_artifact_root=candidate_artifact_root,
        v1_overlay_artifact_root=v1_overlay_artifact_root,
        run_id=run_id, requested_workers=requested_workers, mode=mode,
        run_scope="PUBLIC_SYNTHETIC_FIXTURE",
        budget=budget or W02MorphologySuccessorV2OverlayBudget(),
        fault_after_shard=fault_after_shard)


def run_w02_morphology_successor_v2_overlay_development(
        *, development_root: str | Path,
        candidate_artifact_root: str | Path | None,
        v1_overlay_artifact_root: str | Path | None,
        run_id: int, requested_workers: int, mode: str,
        budget: W02MorphologySuccessorV2OverlayBudget | None = None,
        ) -> W02MorphologySuccessorV2OverlayResult:
    return _run(
        root=Path(development_root),
        candidate_artifact_root=candidate_artifact_root,
        v1_overlay_artifact_root=v1_overlay_artifact_root,
        run_id=run_id, requested_workers=requested_workers, mode=mode,
        run_scope="DEVELOPMENT_PREFLIGHT",
        budget=budget or W02MorphologySuccessorV2OverlayBudget(),
        fault_after_shard=None)


__all__ = [
    "W02_MORPH_V2_OVERLAY_MANIFEST", "W02_MORPH_V2_OVERLAY_SHARD_COUNT",
    "W02MorphologySuccessorV2OverlayBudget",
    "W02MorphologySuccessorV2OverlayError",
    "W02MorphologySuccessorV2OverlayIdentity",
    "W02MorphologySuccessorV2OverlayResult",
    "load_w02_morphology_successor_v2_overlay_index",
    "read_w02_morphology_successor_v2_overlay_artifact",
    "run_w02_morphology_successor_v2_overlay_development",
    "run_w02_morphology_successor_v2_overlay_fixture",
]
