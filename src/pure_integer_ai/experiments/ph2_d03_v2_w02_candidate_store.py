"""PH2-D03-V2 W-02 Candidate 的分片、事务、checkpoint 和封存存储。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import unicodedata

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_streaming import V2LogicalShardPlan
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_io import (
    W02FormalInputRoots,
    iter_w02_training_pairs,
    scan_w02_train_sources,
    verify_w02_visible_transport,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02_CAPABILITY_OOV_BOUNDARY_LATTICE,
    W02_CAPABILITY_UD_MORPHOLOGY,
    W02_CAPABILITY_UNICODE_ANALYSIS,
    W02CandidatePrediction,
    W02CarrierRule,
    W02LearningDelta,
    W02MorphologyCandidate,
    W02UnicodeUnit,
    boundary_lattice,
    generate_with_carrier_rules,
    learn_w02_training_pair,
    observe_w02_carrier,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02CompileFreeze,
    W02_LOGICAL_SHARD_COUNT,
    consume_w02_first_run_guard,
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    TeacherEvidenceRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record


W02_CANDIDATE_ARTIFACT_VERSION = "PH2-D03-V2-W02-CANDIDATE-ARTIFACT-V1"
W02_CANDIDATE_DB_NAME = "candidate.sqlite3"
W02_CANDIDATE_MANIFEST_NAME = "candidate.artifact.json"
W02_CHECKPOINT_NAME = "checkpoints.jsonl"
W02_INPUT_SPOOL_NAME = "input.pairs.sqlite3"
W02_SHARD_DIR_NAME = "shards"
W02_RUN_STATE_NAME = "run-state"
W02_EVENT_BEGIN = "BEGIN"
W02_EVENT_PREVIEW = "PREVIEW"
W02_EVENT_COMMIT = "COMMIT"
W02_EVENT_PUBLISHED = "PUBLISHED"
W02_EVENT_SEQUENCE = (W02_EVENT_BEGIN, W02_EVENT_PREVIEW, W02_EVENT_COMMIT,
                      W02_EVENT_PUBLISHED)


# object-model: exception
class W02CandidateStoreError(RuntimeError):
    """Candidate 运行、资源、恢复或封存边界错误。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CandidateRuntimeBudget:
    """Formal Candidate 的纯整数资源上界。"""

    max_pairs: int = 51_200
    max_logic_operations: int = 9_000_000
    max_checkpoint_count: int = 512
    max_payload_bytes: int = 536_870_912
    max_shard_delta_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        values = (
            self.max_pairs, self.max_logic_operations, self.max_checkpoint_count,
            self.max_payload_bytes, self.max_shard_delta_bytes,
        )
        if any(type(item) is not int or item <= 0 for item in values):
            raise W02CandidateStoreError("Candidate runtime budget 必须是正整数")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_checkpoint_count": self.max_checkpoint_count,
            "max_logic_operations": self.max_logic_operations,
            "max_pairs": self.max_pairs,
            "max_payload_bytes": self.max_payload_bytes,
            "max_shard_delta_bytes": self.max_shard_delta_bytes,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CandidateRunIdentity:
    """与 worker 数无关的 Candidate run 身份。"""

    release_key: str
    stage_key: str
    compile_freeze_sha256: str
    runtime_freeze_sha256: str
    pack_commitment: str
    run_id: int
    logical_shard_count: int = W02_LOGICAL_SHARD_COUNT

    def __post_init__(self) -> None:
        if self.release_key != "PH2-D03-V2" or self.stage_key != "W-02":
            raise W02CandidateStoreError("Candidate run release/stage 身份非法")
        for value in (
                self.compile_freeze_sha256, self.runtime_freeze_sha256,
                self.pack_commitment):
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise W02CandidateStoreError("Candidate run commitment 非法")
        if type(self.run_id) is not int or self.run_id <= 0:
            raise W02CandidateStoreError("Candidate run_id 必须为正整数")
        if self.logical_shard_count != W02_LOGICAL_SHARD_COUNT:
            raise W02CandidateStoreError("Candidate logical shard count 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            "compile_freeze_sha256": self.compile_freeze_sha256,
            "logical_shard_count": self.logical_shard_count,
            "pack_commitment": self.pack_commitment,
            "release_key": self.release_key,
            "run_id": self.run_id,
            "runtime_freeze_sha256": self.runtime_freeze_sha256,
            "stage_key": self.stage_key,
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CandidateRunResult:
    """一个 fresh/restart/resume Candidate 运行的只读结果。"""

    mode: str
    run_identity_sha256: str
    artifact_path: Path
    artifact_manifest_sha256: str
    candidate_semantic_sha256: str
    pair_count: int
    source_count: int
    logic_operations: int
    shard_count: int
    requested_workers: int
    private_payload_reads: int
    teacher_calls: int
    candidate_writes: int
    generated_probe_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"fresh", "restart", "resume"}:
            raise W02CandidateStoreError("Candidate result mode 未注册")
        for value in (
                self.run_identity_sha256, self.artifact_manifest_sha256,
                self.candidate_semantic_sha256, self.generated_probe_sha256):
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise W02CandidateStoreError("Candidate result SHA 非法")
        if not isinstance(self.artifact_path, Path):
            raise W02CandidateStoreError("Candidate artifact path 类型错误")
        if any(type(value) is not int or value < 0 for value in (
                self.pair_count, self.source_count, self.logic_operations,
                self.shard_count, self.requested_workers, self.private_payload_reads,
                self.teacher_calls, self.candidate_writes)):
            raise W02CandidateStoreError("Candidate result 计数非法")


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_child(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root) or target.is_symlink():
        raise W02CandidateStoreError("Candidate store 路径逃逸或 symlink")
    return target


def _run_dirs(candidate_root: Path, run_id: int) -> tuple[Path, Path, Path]:
    store = _safe_child(candidate_root, "candidate-store")
    staging = _safe_child(store, f"run-{run_id:06d}.staging")
    final = _safe_child(store, f"run-{run_id:06d}")
    return store, staging, final


def _sqlite_connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        connection = sqlite3.connect(str(path))
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA page_size=4096")
    connection.row_factory = sqlite3.Row
    return connection


def _create_spool(path: Path) -> None:
    connection = _sqlite_connect(path)
    connection.executescript(
        """
        CREATE TABLE input_meta(
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE input_pairs(
            sequence INTEGER PRIMARY KEY,
            shard_index INTEGER NOT NULL,
            observation_json BLOB NOT NULL,
            evidence_json BLOB NOT NULL
        );
        CREATE INDEX input_pairs_shard_order
            ON input_pairs(shard_index, sequence);
        """
    )
    connection.commit()
    connection.close()


def _write_input_spool(
        path: Path,
        freeze: W02CompileFreeze,
        roots: W02FormalInputRoots,
        identity: W02CandidateRunIdentity,
        budget: W02CandidateRuntimeBudget,
        ) -> tuple[int, int, str]:
    """单遍读取 train pair，物化仅供本次分片的受控临时 spool。"""
    if path.exists():
        raise W02CandidateStoreError("Candidate input spool 不得覆盖")
    _create_spool(path)
    connection = _sqlite_connect(path)
    digest = hashlib.sha256()
    pair_count = 0
    payload_bytes = 0
    plan = V2LogicalShardPlan()
    try:
        source_count, source_digest = scan_w02_train_sources(freeze, roots)
        transport = verify_w02_visible_transport(freeze, roots)
        for observation, evidence in iter_w02_training_pairs(freeze, roots):
            if pair_count >= budget.max_pairs:
                raise W02CandidateStoreError("Candidate runtime pair resource stop")
            observation_payload = canonical_json_bytes(observation.to_dict())
            evidence_payload = canonical_json_bytes(evidence.to_dict())
            shard = plan.shard_for(observation.stable_key.components)
            connection.execute(
                "INSERT INTO input_pairs VALUES(?,?,?,?)",
                (pair_count, shard, observation_payload, evidence_payload),
            )
            pair_digest = hashlib.sha256(observation_payload + evidence_payload).digest()
            digest.update(pair_digest)
            payload_bytes += len(observation_payload) + len(evidence_payload)
            pair_count += 1
            if payload_bytes > budget.max_payload_bytes:
                raise W02CandidateStoreError("Candidate runtime payload resource stop")
        if pair_count != freeze.plan.split_total("train"):
            raise W02CandidateStoreError("Candidate input pair count 不闭合")
        meta = {
            "identity_sha256": identity.sha256(),
            "pair_count": pair_count,
            "pair_digest_sha256": digest.hexdigest(),
            "payload_bytes": payload_bytes,
            "source_count": source_count,
            "source_digest_sha256": source_digest,
            "transport": transport,
        }
        for key, value in sorted(meta.items()):
            connection.execute(
                "INSERT INTO input_meta VALUES(?,?)",
                (key, canonical_json_bytes(value).decode("utf-8")),
            )
        connection.commit()
    except BaseException:
        connection.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    connection.close()
    return pair_count, source_count, digest.hexdigest()


def _write_fixture_spool(
        path: Path,
        pairs: tuple[tuple[ObservationRecord, TeacherEvidenceRecord], ...],
        identity: W02CandidateRunIdentity,
        budget: W02CandidateRuntimeBudget,
        *,
        source_count: int,
        ) -> tuple[int, int, str]:
    """为公开合成 fixture 建 spool；该入口不产生正式运行证据。"""
    if path.exists() or not isinstance(pairs, tuple) or not pairs:
        raise W02CandidateStoreError("Candidate fixture spool 输入非法")
    if type(source_count) is not int or source_count <= 0:
        raise W02CandidateStoreError("Candidate fixture source count 非法")
    _create_spool(path)
    connection = _sqlite_connect(path)
    digest = hashlib.sha256()
    payload_bytes = 0
    plan = V2LogicalShardPlan()
    try:
        for sequence, pair in enumerate(pairs):
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or not isinstance(pair[0], ObservationRecord)
                    or not isinstance(pair[1], TeacherEvidenceRecord)):
                raise W02CandidateStoreError("Candidate fixture pair 类型非法")
            if sequence >= budget.max_pairs:
                raise W02CandidateStoreError("Candidate fixture pair resource stop")
            observation, evidence = pair
            observation_payload = canonical_json_bytes(observation.to_dict())
            evidence_payload = canonical_json_bytes(evidence.to_dict())
            shard = plan.shard_for(observation.stable_key.components)
            connection.execute(
                "INSERT INTO input_pairs VALUES(?,?,?,?)",
                (sequence, shard, observation_payload, evidence_payload),
            )
            digest.update(hashlib.sha256(
                observation_payload + evidence_payload).digest())
            payload_bytes += len(observation_payload) + len(evidence_payload)
            if payload_bytes > budget.max_payload_bytes:
                raise W02CandidateStoreError("Candidate fixture payload resource stop")
        meta = {
            "identity_sha256": identity.sha256(),
            "pair_count": len(pairs),
            "pair_digest_sha256": digest.hexdigest(),
            "payload_bytes": payload_bytes,
            "source_count": source_count,
            "source_digest_sha256": _canonical_sha({"source_count": source_count}),
            "transport": {"scope": "PUBLIC_SYNTHETIC_FIXTURE"},
        }
        for key, value in sorted(meta.items()):
            connection.execute(
                "INSERT INTO input_meta VALUES(?,?)",
                (key, canonical_json_bytes(value).decode("utf-8")),
            )
        connection.commit()
    except BaseException:
        connection.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    connection.close()
    return len(pairs), source_count, digest.hexdigest()


def _read_spool_meta(path: Path, identity: W02CandidateRunIdentity) -> dict[str, Any]:
    connection = _sqlite_connect(path, read_only=True)
    raw = {
        row["key"]: json.loads(row["value_json"])
        for row in connection.execute("SELECT key,value_json FROM input_meta ORDER BY key")
    }
    count = connection.execute("SELECT COUNT(*) FROM input_pairs").fetchone()[0]
    connection.close()
    if raw.get("identity_sha256") != identity.sha256() or raw.get("pair_count") != count:
        raise W02CandidateStoreError("Candidate input spool identity 漂移")
    return raw


def _row_key(value: object) -> bytes:
    return canonical_json_bytes(value)


def _row(kind: str, key: object, payload: object, count: int = 1) -> dict[str, object]:
    return {"count": count, "key": key, "payload": payload, "row_kind": kind}


def _delta_rows(delta: W02LearningDelta) -> list[dict[str, object]]:
    rows = [_row(
        "carrier_rule", list(delta.carrier_rule.key()), delta.carrier_rule.to_dict())]
    for item in delta.unicode_units:
        rows.append(_row(
            "unicode_unit",
            [item.code_point, item.category, item.combining_class],
            item.to_dict(),
        ))
    for item in delta.lexemes:
        rows.append(_row("lexeme", list(item.key()), item.to_dict()))
    for item in delta.oov_units:
        rows.append(_row("oov_unit", list(item.key()), item.to_dict()))
    for capability in delta.capabilities:
        rows.append(_row("capability", capability, {"capability": capability}))
    rows.append(_row(
        "application",
        list(delta.observation_key),
        {
            "boundary_points": list(delta.boundary_points),
            "delta_sha256": delta.delta_sha256,
            "evidence_mode": delta.evidence_mode,
            "evidence_sha256": delta.evidence_sha256,
            "logic_operations": delta.logic_operations,
            "source_ref_key": list(delta.source_ref_key),
            "use_outcome_sha256": delta.use_outcome_sha256,
        },
    ))
    return rows


def _write_gzip_lines(path: Path, values: list[dict[str, object]]) -> tuple[int, str, int, str]:
    partial = path.with_suffix(path.suffix + ".partial")
    content_digest = hashlib.sha256()
    content_size = 0
    line_count = 0
    with partial.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for value in values:
                line = canonical_json_bytes(value) + b"\n"
                stream.write(line)
                content_digest.update(line)
                content_size += len(line)
                line_count += 1
    transport_size, transport_sha = _sha256_file(partial)
    os.replace(partial, path)
    return content_size, content_digest.hexdigest(), transport_size, transport_sha


def _read_gzip_lines(path: Path) -> Iterator[dict[str, object]]:
    try:
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02CandidateStoreError("shard delta 换行非法")
                    value = parse_canonical_json_bytes(line[:-1], require_object=True)
                    if not isinstance(value, dict):
                        raise W02CandidateStoreError("shard delta row 不是 object")
                    yield value
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02CandidateStoreError):
            raise
        raise W02CandidateStoreError("shard delta gzip/JSONL 损坏") from error


def _shard_worker(
        spool_path: str,
        shard_dir: str,
        identity: W02CandidateRunIdentity,
        shard_index: int,
        ) -> dict[str, object]:
    """一个 worker 只读 spool 的一个 deterministic logical shard。"""
    output = Path(shard_dir) / f"{shard_index:03d}.delta.jsonl.gz"
    manifest = Path(shard_dir) / f"{shard_index:03d}.delta.manifest.json"
    if output.is_file() and manifest.is_file():
        value = read_canonical_object(manifest)
        if value.get("run_identity_sha256") != identity.sha256() or value.get("shard_index") != shard_index:
            raise W02CandidateStoreError("已有 shard delta identity 漂移")
        return value
    if output.exists() or manifest.exists():
        raise W02CandidateStoreError("shard delta partial artifact 不可恢复")
    connection = _sqlite_connect(Path(spool_path), read_only=True)
    aggregates: dict[bytes, tuple[dict[str, object], int]] = {}
    pair_count = 0
    logic_operations = 0
    try:
        rows = connection.execute(
            "SELECT observation_json,evidence_json FROM input_pairs "
            "WHERE shard_index=? ORDER BY sequence", (shard_index,))
        for row in rows:
            observation_value = parse_canonical_json_bytes(
                row["observation_json"], require_object=True)
            evidence_value = parse_canonical_json_bytes(
                row["evidence_json"], require_object=True)
            observation = validate_v2_record(observation_value)
            evidence = validate_v2_record(evidence_value)
            if not isinstance(observation, ObservationRecord) or not isinstance(
                    evidence, TeacherEvidenceRecord):
                raise W02CandidateStoreError("spool typed pair 类型错误")
            delta = learn_w02_training_pair(observation, evidence)
            pair_count += 1
            logic_operations += delta.logic_operations
            for item in _delta_rows(delta):
                key = _row_key((item["row_kind"], item["key"]))
                prior = aggregates.get(key)
                if prior is None:
                    aggregates[key] = (item, int(item["count"]))
                else:
                    prior_item, prior_count = prior
                    if prior_item["payload"] != item["payload"]:
                        raise W02CandidateStoreError("同 shard row key payload 冲突")
                    aggregates[key] = (prior_item, prior_count + int(item["count"]))
    finally:
        connection.close()
    values = []
    for key in sorted(aggregates):
        value, count = aggregates[key]
        value = dict(value)
        value["count"] = count
        values.append(value)
    content_size, content_sha, transport_size, transport_sha = _write_gzip_lines(
        output, values)
    value = {
        "artifact_version": W02_CANDIDATE_ARTIFACT_VERSION,
        "content_sha256": content_sha,
        "content_size_bytes": content_size,
        "logic_operations": logic_operations,
        "pair_count": pair_count,
        "row_count": len(values),
        "run_identity_sha256": identity.sha256(),
        "shard_index": shard_index,
        "transport_sha256": transport_sha,
        "transport_size_bytes": transport_size,
    }
    write_immutable_json(value, manifest)
    return value


def _create_candidate_db(path: Path) -> sqlite3.Connection:
    connection = _sqlite_connect(path)
    connection.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE run_events(
            event_seq INTEGER PRIMARY KEY,
            event_kind TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE carrier_rules(
            carrier_kind TEXT NOT NULL,
            prefix TEXT NOT NULL,
            suffix TEXT NOT NULL,
            root_node_kind TEXT NOT NULL,
            content_node_kind TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(carrier_kind,prefix,suffix,root_node_kind,content_node_kind)
        );
        CREATE TABLE unicode_units(
            code_point INTEGER NOT NULL,
            category TEXT NOT NULL,
            combining_class INTEGER NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(code_point,category,combining_class)
        );
        CREATE TABLE lexemes(
            form TEXT NOT NULL,
            lemma TEXT NOT NULL,
            upos TEXT NOT NULL,
            feats_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(form,lemma,upos,feats_json)
        );
        CREATE TABLE oov_units(
            surface TEXT NOT NULL,
            class_signature TEXT NOT NULL,
            unit_length INTEGER NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY(surface,class_signature,unit_length)
        );
        CREATE TABLE capabilities(
            capability TEXT PRIMARY KEY,
            support_count INTEGER NOT NULL
        );
        CREATE TABLE evidence_applications(
            observation_key BLOB PRIMARY KEY,
            source_ref_key BLOB NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            delta_sha256 TEXT NOT NULL,
            evidence_mode TEXT NOT NULL,
            boundary_points_json TEXT NOT NULL,
            use_outcome_sha256 TEXT NOT NULL,
            logic_operations INTEGER NOT NULL
        );
        CREATE TABLE checkpoints(
            checkpoint_seq INTEGER PRIMARY KEY,
            shard_index INTEGER NOT NULL UNIQUE,
            shard_manifest_sha256 TEXT NOT NULL,
            pair_count INTEGER NOT NULL,
            logic_operations INTEGER NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _insert_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT INTO meta VALUES(?,?)",
        (key, canonical_json_bytes(value).decode("utf-8")),
    )


def _append_event(connection: sqlite3.Connection, event_kind: str, payload: dict[str, object]) -> None:
    if event_kind not in W02_EVENT_SEQUENCE:
        raise W02CandidateStoreError("Candidate event 未注册")
    prior = connection.execute(
        "SELECT event_seq,event_kind,payload_sha256,payload_json FROM run_events ORDER BY event_seq"
    ).fetchall()
    position = W02_EVENT_SEQUENCE.index(event_kind)
    payload_json = canonical_json_bytes(payload).decode("utf-8")
    payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    if len(prior) > position:
        row = prior[position]
        if (row["event_kind"] != event_kind or row["payload_sha256"] != payload_sha
                or row["payload_json"] != payload_json):
            raise W02CandidateStoreError("Candidate event identity 漂移")
        return
    if len(prior) != position:
        raise W02CandidateStoreError("Candidate event 不得跳级")
    connection.execute(
        "INSERT INTO run_events VALUES(?,?,?,?)",
        (position + 1, event_kind, payload_sha, payload_json),
    )


def _upsert_count(
        connection: sqlite3.Connection,
        table: str,
        columns: tuple[str, ...],
        values: tuple[object, ...],
        count: int,
        ) -> None:
    placeholders = ",".join("?" for _ in values)
    connection.execute(
        f"INSERT INTO {table} ({','.join(columns)},support_count) VALUES({placeholders},?) "
        "ON CONFLICT DO UPDATE SET support_count=support_count+excluded.support_count",
        (*values, count),
    )


def _merge_row(connection: sqlite3.Connection, value: dict[str, object]) -> None:
    kind = value.get("row_kind")
    payload = value.get("payload")
    count = value.get("count")
    if type(count) is not int or count <= 0 or not isinstance(payload, dict):
        raise W02CandidateStoreError("shard delta row count/payload 非法")
    if kind == "carrier_rule":
        rule = W02CarrierRule(
            str(payload["carrier_kind"]), str(payload["prefix"]),
            str(payload["suffix"]), str(payload["root_node_kind"]),
            str(payload["content_node_kind"]),
        )
        _upsert_count(connection, "carrier_rules", (
            "carrier_kind", "prefix", "suffix", "root_node_kind", "content_node_kind"),
            rule.key(), count)
    elif kind == "unicode_unit":
        _upsert_count(connection, "unicode_units",
                      ("code_point", "category", "combining_class"),
                      (int(payload["code_point"]), str(payload["category"]),
                       int(payload["combining_class"])), count)
    elif kind == "lexeme":
        _upsert_count(connection, "lexemes", ("form", "lemma", "upos", "feats_json"),
                      (str(payload["form"]), str(payload["lemma"]),
                       str(payload["upos"]), str(payload["feats_json"])), count)
    elif kind == "oov_unit":
        _upsert_count(connection, "oov_units",
                      ("surface", "class_signature", "unit_length"),
                      (str(payload["surface"]), str(payload["class_signature"]),
                       int(payload["length"])), count)
    elif kind == "capability":
        _upsert_count(connection, "capabilities", ("capability",),
                      (str(payload["capability"]),), count)
    elif kind == "application":
        key = canonical_json_bytes(value["key"])
        prior = connection.execute(
            "SELECT source_ref_key,evidence_sha256,delta_sha256,evidence_mode,"
            "boundary_points_json,use_outcome_sha256,logic_operations "
            "FROM evidence_applications WHERE observation_key=?", (key,)).fetchone()
        boundary_json = canonical_json_bytes(payload["boundary_points"]).decode("utf-8")
        row = (
            canonical_json_bytes(payload["source_ref_key"]),
            str(payload["evidence_sha256"]), str(payload["delta_sha256"]),
            str(payload["evidence_mode"]), boundary_json,
            str(payload["use_outcome_sha256"]), int(payload["logic_operations"]),
        )
        if prior is not None and tuple(prior) != row:
            raise W02CandidateStoreError("同 observation application 漂移")
        if prior is None:
            connection.execute(
                "INSERT INTO evidence_applications VALUES(?,?,?,?,?,?,?,?)",
                (key, *row),
            )
    else:
        raise W02CandidateStoreError("未知 shard delta row kind")


def _semantic_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    queries = (
        "SELECT carrier_kind,prefix,suffix,root_node_kind,content_node_kind,support_count "
        "FROM carrier_rules ORDER BY carrier_kind,prefix,suffix,root_node_kind,content_node_kind",
        "SELECT code_point,category,combining_class,support_count FROM unicode_units "
        "ORDER BY code_point,category,combining_class",
        "SELECT form,lemma,upos,feats_json,support_count FROM lexemes "
        "ORDER BY form,lemma,upos,feats_json",
        "SELECT surface,class_signature,unit_length,support_count FROM oov_units "
        "ORDER BY surface,class_signature,unit_length",
        "SELECT capability,support_count FROM capabilities ORDER BY capability",
        "SELECT observation_key,source_ref_key,evidence_sha256,delta_sha256,evidence_mode,"
        "boundary_points_json,use_outcome_sha256,logic_operations "
        "FROM evidence_applications ORDER BY observation_key",
    )
    for query in queries:
        for row in connection.execute(query):
            projected = [
                {"blob_hex": value.hex()} if isinstance(value, bytes) else value
                for value in row
            ]
            digest.update(canonical_json_bytes(projected))
    return digest.hexdigest()


def _checkpoint_payload(value: dict[str, object]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _append_checkpoint(path: Path, value: dict[str, object]) -> str:
    payload = _checkpoint_payload(value)
    with path.open("ab") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _read_checkpoint(path: Path, identity: W02CandidateRunIdentity) -> dict[int, dict[str, object]]:
    result = {}
    if not path.is_file():
        return result
    with path.open("rb") as handle:
        for line in handle:
            value = parse_canonical_json_bytes(line.rstrip(b"\n"), require_object=True)
            if (not isinstance(value, dict)
                    or value.get("run_identity_sha256") != identity.sha256()):
                raise W02CandidateStoreError("Candidate checkpoint identity 漂移")
            shard = value.get("shard_index")
            if type(shard) is not int or shard in result:
                raise W02CandidateStoreError("Candidate checkpoint shard 重复/非法")
            result[shard] = value
    return result


def _merge_artifact(
        staging: Path,
        identity: W02CandidateRunIdentity,
        expected_pair_count: int,
        budget: W02CandidateRuntimeBudget,
        ) -> tuple[str, int, int]:
    shard_dir = staging / W02_SHARD_DIR_NAME
    db_partial = staging / (W02_CANDIDATE_DB_NAME + ".partial")
    if db_partial.exists():
        raise W02CandidateStoreError("Candidate DB partial 不可覆盖")
    connection = _create_candidate_db(db_partial)
    _append_event(connection, W02_EVENT_BEGIN, {
        "identity_sha256": identity.sha256(),
        "stage_key": identity.stage_key,
    })
    pair_count = 0
    logic_operations = 0
    shard_manifests = []
    try:
        for shard_index in range(identity.logical_shard_count):
            manifest_path = shard_dir / f"{shard_index:03d}.delta.manifest.json"
            delta_path = shard_dir / f"{shard_index:03d}.delta.jsonl.gz"
            manifest = read_canonical_object(manifest_path)
            if (manifest.get("run_identity_sha256") != identity.sha256()
                    or manifest.get("shard_index") != shard_index):
                raise W02CandidateStoreError("shard manifest identity 漂移")
            content_size, content_sha = _content_identity(delta_path)
            transport_size, transport_sha = _sha256_file(delta_path)
            if (content_size != manifest["content_size_bytes"]
                    or content_sha != manifest["content_sha256"]
                    or transport_size != manifest["transport_size_bytes"]
                    or transport_sha != manifest["transport_sha256"]):
                raise W02CandidateStoreError("shard delta content identity 漂移")
            for value in _read_gzip_lines(delta_path):
                _merge_row(connection, value)
            pair_count += int(manifest["pair_count"])
            logic_operations += int(manifest["logic_operations"])
            shard_manifests.append(manifest)
            if len(shard_manifests) > budget.max_checkpoint_count:
                raise W02CandidateStoreError("Candidate checkpoint resource stop")
            connection.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?,?)",
                (shard_index + 1, shard_index,
                 hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
                 int(manifest["pair_count"]), int(manifest["logic_operations"])),
            )
        if pair_count != expected_pair_count:
            raise W02CandidateStoreError("Candidate merged pair count 不闭合")
        if logic_operations > budget.max_logic_operations:
            raise W02CandidateStoreError("Candidate logic resource stop")
        semantic = _semantic_digest(connection)
        _insert_meta(connection, "artifact_version", W02_CANDIDATE_ARTIFACT_VERSION)
        _insert_meta(connection, "candidate_semantic_sha256", semantic)
        _insert_meta(connection, "compile_freeze_sha256", identity.compile_freeze_sha256)
        _insert_meta(connection, "logical_shard_count", identity.logical_shard_count)
        _insert_meta(connection, "pack_commitment", identity.pack_commitment)
        _insert_meta(connection, "pair_count", pair_count)
        _insert_meta(connection, "run_identity_sha256", identity.sha256())
        _insert_meta(connection, "runtime_freeze_sha256", identity.runtime_freeze_sha256)
        _insert_meta(connection, "stage_key", identity.stage_key)
        _append_event(connection, W02_EVENT_PREVIEW, {
            "candidate_semantic_sha256": semantic,
            "logic_operations": logic_operations,
            "pair_count": pair_count,
            "shard_count": len(shard_manifests),
        })
        _append_event(connection, W02_EVENT_COMMIT, {
            "candidate_semantic_sha256": semantic,
            "logic_operations": logic_operations,
            "pair_count": pair_count,
        })
        connection.commit()
    finally:
        connection.close()
    os.replace(db_partial, staging / W02_CANDIDATE_DB_NAME)
    db_size, _ = _sha256_file(staging / W02_CANDIDATE_DB_NAME)
    if db_size > budget.max_payload_bytes:
        raise W02CandidateStoreError("Candidate DB resource stop")
    return semantic, pair_count, logic_operations


def _content_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for line in _read_gzip_lines(path):
        payload = canonical_json_bytes(line) + b"\n"
        digest.update(payload)
        size += len(payload)
    return size, digest.hexdigest()


def _tree_inventory(root: Path) -> list[dict[str, object]]:
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == W02_CANDIDATE_MANIFEST_NAME:
            continue
        size, digest = _sha256_file(path)
        values.append({"path": relative, "sha256": digest, "size_bytes": size})
    return values


def _publish_manifest(
        staging: Path,
        identity: W02CandidateRunIdentity,
        source_count: int,
        semantic: str,
        pair_count: int,
        logic_operations: int,
        generated_probe_sha256: str,
        run_scope: str,
        ) -> tuple[Path, str]:
    manifest_path = staging / W02_CANDIDATE_MANIFEST_NAME
    if manifest_path.exists():
        raise W02CandidateStoreError("Candidate artifact manifest 不得覆盖")
    if run_scope not in {"FORMAL", "PUBLIC_SYNTHETIC_FIXTURE"}:
        raise W02CandidateStoreError("Candidate run scope 未注册")
    value = {
        "artifact_kind": "PH2_D03_V2_W02_CANDIDATE_ARTIFACT",
        "artifact_version": W02_CANDIDATE_ARTIFACT_VERSION,
        "candidate_semantic_sha256": semantic,
        "compile_freeze_sha256": identity.compile_freeze_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1 if run_scope == "FORMAL" else 0,
        "generated_probe_sha256": generated_probe_sha256,
        "input_roots": (
            ["CANDIDATE_TRAIN_ROOT", "TEACHER_TRAIN_ROOT"]
            if run_scope == "FORMAL" else ["PUBLIC_SYNTHETIC_FIXTURE"]),
        "logic_operations": logic_operations,
        "logical_shard_count": identity.logical_shard_count,
        "pack_commitment": identity.pack_commitment,
        "pair_count": pair_count,
        "private_payload_reads": 0,
        "run_identity": identity.to_dict(),
        "run_identity_sha256": identity.sha256(),
        "runtime_freeze_sha256": identity.runtime_freeze_sha256,
        "run_scope": run_scope,
        "source_count": source_count,
        "stage_key": identity.stage_key,
        "status": "CANDIDATE_ARTIFACT_SEALED",
        "teacher_calls": 0,
        "tree": _tree_inventory(staging),
        "visible_splits": ["train"],
        "worker_counts_supported": [1, 2, 4],
        "candidate_writes": 1,
    }
    write_immutable_json(value, manifest_path)
    payload = manifest_path.read_bytes()
    expected = canonical_json_bytes(value) + b"\n"
    if payload != expected:
        raise W02CandidateStoreError("Candidate artifact manifest 字节漂移")
    return manifest_path, hashlib.sha256(payload).hexdigest()


def _run_state(path: Path, identity: W02CandidateRunIdentity, status: str) -> None:
    if status not in {"RUNNING", "FAILED_RECOVERABLE", "SEALED"}:
        raise W02CandidateStoreError("Candidate run state 未注册")
    root = path / W02_RUN_STATE_NAME
    root.mkdir(exist_ok=True)
    value = {
        "artifact_version": W02_CANDIDATE_ARTIFACT_VERSION,
        "run_identity_sha256": identity.sha256(),
        "status": status,
    }
    prior_paths = tuple(sorted(root.glob("*.json")))
    for prior_path in prior_paths:
        prior = read_canonical_object(prior_path)
        if prior.get("run_identity_sha256") != identity.sha256():
            raise W02CandidateStoreError("Candidate run state identity 漂移")
    if prior_paths and read_canonical_object(prior_paths[-1]) == value:
        return
    target = root / f"{len(prior_paths) + 1:03d}-{status.casefold()}.json"
    write_immutable_json(value, target)


def _build_probe_sha256(connection: sqlite3.Connection) -> str:
    rows = tuple(connection.execute(
        "SELECT carrier_kind,prefix,suffix,root_node_kind,content_node_kind,"
        "support_count FROM carrier_rules ORDER BY carrier_kind,prefix,suffix,"
        "root_node_kind,content_node_kind"))
    if not rows:
        raise W02CandidateStoreError("Candidate 没有 carrier hypothesis")
    rules = tuple((W02CarrierRule(
        str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])),
        int(row[5])) for row in rows)
    probe_surface = "probe-" + hashlib.sha256(
        canonical_json_bytes([list(row) for row in rows])).hexdigest()[:12]
    carrier_kinds = tuple(sorted({rule.carrier_kind for rule, _ in rules}))
    generated = tuple(generate_with_carrier_rules(
        rules, carrier_kind=carrier_kind, surface=probe_surface).to_dict()
        for carrier_kind in carrier_kinds)
    if any(item["status"] != "GENERATED" for item in generated):
        raise W02CandidateStoreError("Candidate learned carrier probe 未唯一生成")
    payload = canonical_json_bytes({
        "generated": list(generated),
        "probe_surface": probe_surface,
    })
    return hashlib.sha256(payload).hexdigest()


def _shard_manifest_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _execute_candidate_pipeline(
        *,
        staging: Path,
        final: Path,
        identity: W02CandidateRunIdentity,
        expected_pair_count: int,
        source_count: int,
        requested_workers: int,
        mode: str,
        budget: W02CandidateRuntimeBudget,
        fault_after_shard: int | None,
        run_scope: str,
        ) -> W02CandidateRunResult:
    """fixture/formal 共用的 shard、checkpoint、merge 和 manifest-last 管线。"""
    if expected_pair_count > budget.max_pairs:
        raise W02CandidateStoreError("Candidate pair resource stop")
    spool = staging / W02_INPUT_SPOOL_NAME
    checkpoint_path = staging / W02_CHECKPOINT_NAME
    completed = _read_checkpoint(checkpoint_path, identity)
    shard_dir = staging / W02_SHARD_DIR_NAME
    shard_indices = tuple(range(identity.logical_shard_count))
    futures = {}
    with ThreadPoolExecutor(max_workers=requested_workers) as executor:
        for shard_index in shard_indices:
            if shard_index in completed:
                continue
            futures[shard_index] = executor.submit(
                _shard_worker, str(spool), str(shard_dir), identity, shard_index)
        for shard_index in shard_indices:
            if shard_index in completed:
                continue
            manifest = futures[shard_index].result()
            checkpoint = {
                "checkpoint_seq": shard_index + 1,
                "logic_operations": int(manifest["logic_operations"]),
                "pair_count": int(manifest["pair_count"]),
                "run_identity_sha256": identity.sha256(),
                "shard_index": shard_index,
                "shard_manifest_sha256": _shard_manifest_sha(manifest),
            }
            _append_checkpoint(checkpoint_path, checkpoint)
            completed[shard_index] = checkpoint
            if fault_after_shard is not None and shard_index == fault_after_shard:
                _run_state(staging, identity, "FAILED_RECOVERABLE")
                raise W02CandidateStoreError("injected Candidate shard fault")
    if len(completed) != identity.logical_shard_count:
        raise W02CandidateStoreError("Candidate shard completion 不闭合")
    semantic, pair_count, logic_operations = _merge_artifact(
        staging, identity, expected_pair_count, budget)
    if fault_after_shard == identity.logical_shard_count:
        _run_state(staging, identity, "FAILED_RECOVERABLE")
        raise W02CandidateStoreError("injected Candidate merge fault")
    db = _sqlite_connect(staging / W02_CANDIDATE_DB_NAME, read_only=True)
    generated_probe_sha = _build_probe_sha256(db)
    db.close()
    spool.unlink()
    _run_state(staging, identity, "SEALED")
    _, manifest_sha = _publish_manifest(
        staging, identity, source_count, semantic, pair_count,
        logic_operations, generated_probe_sha, run_scope)
    os.replace(staging, final)
    return W02CandidateRunResult(
        mode, identity.sha256(), final, manifest_sha, semantic, pair_count,
        source_count, logic_operations, identity.logical_shard_count,
        requested_workers, 0, 0, 1, generated_probe_sha)


def run_w02_candidate(
        *,
        repository_root: str | Path,
        candidate_root: str | Path,
        teacher_root: str | Path,
        compile_freeze_sha256: str,
        runtime_freeze_sha256: str,
        pack_commitment: str,
        expected_guard_sha256: str,
        run_id: int,
        requested_workers: int,
        mode: str,
        budget: W02CandidateRuntimeBudget | None = None,
        fault_after_shard: int | None = None,
        ) -> W02CandidateRunResult:
    """执行 Candidate fresh/restart/resume；formal caller 负责唯一 guard 语义。"""
    if type(requested_workers) is not int or requested_workers not in (1, 2, 4):
        raise W02CandidateStoreError("Candidate requested_workers 必须是 1/2/4")
    if mode not in {"fresh", "resume"}:
        raise W02CandidateStoreError(
            "正式 Candidate 只允许唯一 fresh 或封存后的只读 resume")
    if fault_after_shard is not None:
        raise W02CandidateStoreError("正式 Candidate 禁止注入 fixture fault")
    budget = W02CandidateRuntimeBudget() if budget is None else budget
    candidate = Path(candidate_root).resolve()
    teacher = Path(teacher_root).resolve()
    roots = W02FormalInputRoots(candidate, teacher)
    repository = Path(repository_root).resolve()
    freeze = read_w02_compile_freeze(repository)
    if freeze.sha256() != compile_freeze_sha256:
        raise W02CandidateStoreError("Candidate compile freeze file SHA 漂移")
    if freeze.pack_commitment != pack_commitment:
        raise W02CandidateStoreError("Candidate pack commitment 漂移")
    identity = W02CandidateRunIdentity(
        "PH2-D03-V2", "W-02", compile_freeze_sha256,
        runtime_freeze_sha256, pack_commitment, run_id)
    store, staging, final = _run_dirs(candidate, run_id)
    if mode == "fresh":
        if staging.exists() or final.exists():
            raise W02CandidateStoreError("fresh Candidate run 不能复用 staging/final")
        consume_w02_first_run_guard(
            candidate, expected_guard_sha256=expected_guard_sha256,
            run_id=run_id, run_identity_sha256=identity.sha256())
        store.mkdir(parents=False, exist_ok=True) if not store.exists() else None
        staging.mkdir()
        shard_dir = staging / W02_SHARD_DIR_NAME
        shard_dir.mkdir()
        _run_state(staging, identity, "RUNNING")
        spool = staging / W02_INPUT_SPOOL_NAME
        pair_count, source_count, _ = _write_input_spool(
            spool, freeze, roots, identity, budget)
    else:
        if not final.is_dir() or staging.exists():
            raise W02CandidateStoreError("resume 只接受已封存 final")
        return read_w02_candidate_artifact(final, requested_workers=requested_workers)
    return _execute_candidate_pipeline(
        staging=staging, final=final, identity=identity,
        expected_pair_count=pair_count, source_count=source_count,
        requested_workers=requested_workers, mode=mode, budget=budget,
        fault_after_shard=None, run_scope="FORMAL")


def run_w02_candidate_fixture(
        *,
        fixture_root: str | Path,
        pairs: tuple[tuple[ObservationRecord, TeacherEvidenceRecord], ...] | None,
        run_id: int,
        requested_workers: int,
        mode: str,
        budget: W02CandidateRuntimeBudget | None = None,
        fault_after_shard: int | None = None,
        ) -> W02CandidateRunResult:
    """运行公开合成 fixture；不得被引用为正式 Candidate 或 mastery 证据。"""
    if type(requested_workers) is not int or requested_workers not in (1, 2, 4):
        raise W02CandidateStoreError("fixture requested_workers 必须是 1/2/4")
    if mode not in {"fresh", "restart", "resume"}:
        raise W02CandidateStoreError("fixture mode 未注册")
    budget = W02CandidateRuntimeBudget() if budget is None else budget
    root = Path(fixture_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = W02CandidateRunIdentity(
        "PH2-D03-V2", "W-02",
        hashlib.sha256(b"PUBLIC-FIXTURE-COMPILE").hexdigest(),
        hashlib.sha256(b"PUBLIC-FIXTURE-RUNTIME").hexdigest(),
        hashlib.sha256(b"PUBLIC-FIXTURE-PACK").hexdigest(),
        run_id,
    )
    store, staging, final = _run_dirs(root, run_id)
    store.mkdir(exist_ok=True)
    if mode == "fresh":
        if pairs is None or staging.exists() or final.exists():
            raise W02CandidateStoreError("fresh fixture 输入或路径非法")
        staging.mkdir()
        (staging / W02_SHARD_DIR_NAME).mkdir()
        _run_state(staging, identity, "RUNNING")
        pair_count, source_count, _ = _write_fixture_spool(
            staging / W02_INPUT_SPOOL_NAME, pairs, identity, budget,
            source_count=len({pair[0].source_ref_key.components for pair in pairs}))
    elif mode == "restart":
        if pairs is not None or not staging.is_dir() or final.exists():
            raise W02CandidateStoreError("restart fixture 输入或路径非法")
        meta = _read_spool_meta(staging / W02_INPUT_SPOOL_NAME, identity)
        pair_count = int(meta["pair_count"])
        source_count = int(meta["source_count"])
    else:
        if pairs is not None or not final.is_dir() or staging.exists():
            raise W02CandidateStoreError("resume fixture 输入或路径非法")
        return read_w02_candidate_artifact(final, requested_workers=requested_workers)
    return _execute_candidate_pipeline(
        staging=staging, final=final, identity=identity,
        expected_pair_count=pair_count, source_count=source_count,
        requested_workers=requested_workers, mode=mode, budget=budget,
        fault_after_shard=fault_after_shard,
        run_scope="PUBLIC_SYNTHETIC_FIXTURE")


def read_w02_candidate_artifact(
        artifact_root: str | Path,
        *,
        requested_workers: int = 1,
        ) -> W02CandidateRunResult:
    """只读回读封存 Candidate，验证 manifest、DB semantic digest 和零写入。"""
    root = Path(artifact_root).resolve()
    manifest_path = root / W02_CANDIDATE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise W02CandidateStoreError("Candidate artifact manifest 缺失")
    before = tuple((item.relative_to(root).as_posix(), _sha256_file(item))
                   for item in sorted(root.rglob("*")) if item.is_file())
    value = read_canonical_object(manifest_path)
    required = {
        "artifact_kind", "artifact_version", "candidate_semantic_sha256",
        "compile_freeze_sha256", "formal_private_evaluation_runs",
        "formal_training_runs",
        "generated_probe_sha256", "input_roots", "logic_operations",
        "logical_shard_count", "pack_commitment", "pair_count",
        "private_payload_reads", "run_identity",
        "run_identity_sha256", "runtime_freeze_sha256", "run_scope", "source_count",
        "stage_key", "status", "teacher_calls", "tree", "visible_splits",
        "worker_counts_supported", "candidate_writes",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise W02CandidateStoreError("Candidate artifact manifest 字段漂移")
    if (value["artifact_kind"] != "PH2_D03_V2_W02_CANDIDATE_ARTIFACT"
            or value["artifact_version"] != W02_CANDIDATE_ARTIFACT_VERSION
            or value["status"] != "CANDIDATE_ARTIFACT_SEALED"
            or value["private_payload_reads"] != 0 or value["teacher_calls"] != 0
            or value["formal_private_evaluation_runs"] != 0
            or value["candidate_writes"] != 1):
        raise W02CandidateStoreError("Candidate artifact manifest 状态非法")
    if (value["worker_counts_supported"] != [1, 2, 4]
            or value["visible_splits"] != ["train"]
            or value["run_scope"] not in {"FORMAL", "PUBLIC_SYNTHETIC_FIXTURE"}
            or value["formal_training_runs"] != (
                1 if value["run_scope"] == "FORMAL" else 0)):
        raise W02CandidateStoreError("Candidate artifact scope/worker 状态非法")
    run_raw = value["run_identity"]
    if not isinstance(run_raw, dict):
        raise W02CandidateStoreError("Candidate run identity 类型错误")
    try:
        identity = W02CandidateRunIdentity(
            str(run_raw["release_key"]), str(run_raw["stage_key"]),
            str(run_raw["compile_freeze_sha256"]),
            str(run_raw["runtime_freeze_sha256"]),
            str(run_raw["pack_commitment"]), int(run_raw["run_id"]),
            int(run_raw["logical_shard_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise W02CandidateStoreError("Candidate run identity 无法回读") from error
    if (run_raw != identity.to_dict()
            or value["run_identity_sha256"] != identity.sha256()):
        raise W02CandidateStoreError("Candidate run identity 摘要漂移")
    expected_tree = _tree_inventory(root)
    if value["tree"] != expected_tree:
        raise W02CandidateStoreError("Candidate artifact tree identity 漂移")
    db_path = root / W02_CANDIDATE_DB_NAME
    if not db_path.is_file():
        raise W02CandidateStoreError("Candidate DB 缺失")
    connection = _sqlite_connect(db_path, read_only=True)
    meta = {
        row["key"]: json.loads(row["value_json"])
        for row in connection.execute("SELECT key,value_json FROM meta ORDER BY key")
    }
    semantic = _semantic_digest(connection)
    event_kinds = tuple(row[0] for row in connection.execute(
        "SELECT event_kind FROM run_events ORDER BY event_seq"))
    pair_count = connection.execute(
        "SELECT COUNT(*) FROM evidence_applications").fetchone()[0]
    logic_operations = connection.execute(
        "SELECT COALESCE(SUM(logic_operations),0) FROM evidence_applications").fetchone()[0]
    connection.close()
    if (semantic != value["candidate_semantic_sha256"]
            or meta.get("candidate_semantic_sha256") != semantic
            or pair_count != value["pair_count"]
            or logic_operations != value["logic_operations"]
            or event_kinds != W02_EVENT_SEQUENCE[:3]):
        raise W02CandidateStoreError("Candidate artifact DB semantic/readback 漂移")
    after = tuple((item.relative_to(root).as_posix(), _sha256_file(item))
                  for item in sorted(root.rglob("*")) if item.is_file())
    if before != after:
        raise W02CandidateStoreError("Candidate readback 产生写入")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    identity_sha = str(value["run_identity_sha256"])
    return W02CandidateRunResult(
        "resume", identity_sha, root, manifest_sha, semantic,
        int(value["pair_count"]), int(value["source_count"]),
        int(value["logic_operations"]), int(value["logical_shard_count"]),
        requested_workers, 0, 0, 1, str(value["generated_probe_sha256"]))


# object-model: lifecycle; owner=evaluator; cleanup=context-manager-close
class W02CandidatePredictor:
    """只读持有 Candidate SQLite 连接的 evaluator 边缘资源。"""

    def __init__(self, artifact_root: str | Path) -> None:
        result = read_w02_candidate_artifact(artifact_root)
        self.artifact_root = result.artifact_path
        self.connection = _sqlite_connect(
            self.artifact_root / W02_CANDIDATE_DB_NAME, read_only=True)
        self.capabilities = tuple(row[0] for row in self.connection.execute(
            "SELECT capability FROM capabilities ORDER BY capability"))
        max_length = self.connection.execute(
            "SELECT COALESCE(MAX(LENGTH(form)),0) FROM lexemes").fetchone()[0]
        self.max_lexeme_length = int(max_length)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "W02CandidatePredictor":
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.close()

    def predict(self, observation: ObservationRecord) -> W02CandidatePrediction:
        """对无 label Observation 形成载体、边界、Unicode 和词形候选。"""
        observed = observe_w02_carrier(observation)
        rule_rows = tuple(self.connection.execute(
            "SELECT carrier_kind,prefix,suffix,root_node_kind,content_node_kind,"
            "support_count FROM carrier_rules WHERE carrier_kind=? "
            "ORDER BY prefix,suffix,root_node_kind,content_node_kind",
            (observed.carrier_kind,)))
        rules = tuple((W02CarrierRule(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])),
            int(row[5])) for row in rule_rows)
        generation = generate_with_carrier_rules(
            rules, carrier_kind=observed.carrier_kind, surface=observed.surface)
        lengths = tuple(row[0] for row in self.connection.execute(
            "SELECT DISTINCT unit_length FROM oov_units ORDER BY unit_length"))
        if W02_CAPABILITY_OOV_BOUNDARY_LATTICE in self.capabilities and lengths:
            points = boundary_lattice(observed.surface, observed_unit_lengths=lengths)
        else:
            points = (0,) if not observed.surface else (0, len(observed.surface))
        unicode_units = ()
        if W02_CAPABILITY_UNICODE_ANALYSIS in self.capabilities:
            unicode_units = tuple(W02UnicodeUnit(
                ord(char), unicodedata.category(char), unicodedata.combining(char))
                for char in observed.surface)
        morphology = []
        if (W02_CAPABILITY_UD_MORPHOLOGY in self.capabilities
                and self.max_lexeme_length > 0):
            for start in range(len(observed.surface)):
                limit = min(len(observed.surface), start + self.max_lexeme_length)
                for end in range(start + 1, limit + 1):
                    form = observed.surface[start:end]
                    for row in self.connection.execute(
                            "SELECT lemma,upos,feats_json,support_count FROM lexemes "
                            "WHERE form=? ORDER BY lemma,upos,feats_json", (form,)):
                        morphology.append(W02MorphologyCandidate(
                            start, end, form, str(row[0]), str(row[1]),
                            str(row[2]), int(row[3])))
        status = {
            "GENERATED": "PREDICTED",
            "AMBIGUOUS": "AMBIGUOUS",
            "UNKNOWN": "UNKNOWN",
        }[generation.status]
        return W02CandidatePrediction(
            observed.observation_key, status, generation, points,
            unicode_units, tuple(morphology), self.capabilities)


def open_w02_candidate_predictor(
        artifact_root: str | Path,
        ) -> W02CandidatePredictor:
    """打开只读 Candidate predictor，并先完成 artifact 全身份回读。"""
    return W02CandidatePredictor(artifact_root)


__all__ = [
    "W02CandidateRunIdentity",
    "W02CandidateRunResult",
    "W02CandidateRuntimeBudget",
    "W02CandidatePredictor",
    "W02CandidateStoreError",
    "open_w02_candidate_predictor",
    "read_w02_candidate_artifact",
    "run_w02_candidate",
    "run_w02_candidate_fixture",
]
