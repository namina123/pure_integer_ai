"""Runtime 资料 event/observation 的纯整数、可重建账本。

资料 response binding 本身只保存路由和 qualification；本模块补上它依赖的
两类运行时事实：累计 Runtime Memory state，以及语言 observation/evidence 的
稳定身份。文件采用 append-only 编号，回读时要求 event、SourceRecord、raw
observation、lexical evidence、relation candidate 和 stable key 全部闭合。

账本不保存 Python 对象、SQLite 句柄或自然语言推断结果。回读需要一个已经
装配好的 ``TrainContext``，但不再需要调用方逐条传入 observation；语言观察
管线只用于按已记录的元数据重放，并与账本身份逐字段比较。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    ADMISSION_ACCEPTED,
    LearningReplayReceipt,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.experiments.conversation_persistent_run import (
    PersistentDialogueRunError,
    decode_runtime_memory_state,
    encode_runtime_memory_state,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    RuntimeMaterialIngest,
    _replay_key,
)
from pure_integer_ai.experiments.conversation_runtime_material_language import (
    RuntimeMaterialLanguageObservation,
    observe_runtime_material_language,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunRoot,
    ensure_normal_relative_directory,
    open_existing_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)
from pure_integer_ai.storage.source_record import (
    SOURCE_RECORD_TABLE,
    SourceRecordRepository,
)
from pure_integer_ai.storage.backend import SQLiteBackend


RUNTIME_MATERIAL_PERSISTENCE_VERSION = 1
RUNTIME_MATERIAL_EVENT_LEDGER_VERSION = 2
RUNTIME_MATERIAL_OBSERVATION_LEDGER_VERSION = 1
RUNTIME_MATERIAL_EVENT_RELATIVE = "runtime_material/events"
RUNTIME_MATERIAL_OBSERVATION_RELATIVE = "runtime_material/observations"
RUNTIME_MATERIAL_MANIFEST_RELATIVE = "runtime_material_manifest.json"
RUNTIME_MATERIAL_MANIFEST_FORMAT = "PURE_INTEGER_AI_RUNTIME_MATERIAL_MANIFEST"
RUNTIME_MATERIAL_MANIFEST_SCHEMA_VERSION = 1


class RuntimeMaterialPersistenceError(ValueError):
    """Runtime 资料账本缺失、损坏或跨层 identity 不闭合。"""


def _key(value: tuple[int, ...], *, label: str, empty: bool = True) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or (not empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise RuntimeMaterialPersistenceError(f"{label} 必须是非负整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str,
          empty: bool = True) -> None:
    value = _key(value, label=label, empty=empty)
    result.extend((len(value), *value))


def _read_key(reader: IntegerStreamReader, *, label: str,
              empty: bool = True) -> tuple[int, ...]:
    try:
        return _key(reader.read_key(label=label, empty=empty), label=label,
                    empty=empty)
    except (IntegerCodecError, ValueError) as error:
        raise RuntimeMaterialPersistenceError(f"{label} 不可读取") from error


def _text(value: str, *, label: str) -> tuple[int, ...]:
    if type(value) is not str or not value.strip() or value.strip() != value:
        raise RuntimeMaterialPersistenceError(f"{label} 必须是无首尾空白文本")
    return tuple(ord(item) for item in value)


def _decode_text(value: tuple[int, ...], *, label: str) -> str:
    if not value or any(item < 0 or item > 0x10FFFF
                        or 0xD800 <= item <= 0xDFFF for item in value):
        raise RuntimeMaterialPersistenceError(f"{label} scalar 非法")
    try:
        return "".join(chr(item) for item in value)
    except (TypeError, ValueError) as error:
        raise RuntimeMaterialPersistenceError(f"{label} 无法恢复") from error


def _pack_nested(result: list[int], values: tuple[tuple[int, ...], ...], *,
                 label: str) -> None:
    if not isinstance(values, tuple):
        raise RuntimeMaterialPersistenceError(f"{label} 必须是 tuple")
    result.append(len(values))
    for index, value in enumerate(values):
        _pack(result, value, label=f"{label}[{index}]")


def _read_nested(reader: IntegerStreamReader, *, label: str,
                 allow_empty: bool = True) -> tuple[tuple[int, ...], ...]:
    try:
        count = reader.read_nonnegative(label=f"{label}.count")
    except (IntegerCodecError, ValueError) as error:
        raise RuntimeMaterialPersistenceError(f"{label}.count 不可读取") from error
    if not allow_empty and count == 0:
        raise RuntimeMaterialPersistenceError(f"{label} 不得为空")
    return tuple(_read_key(reader, label=f"{label}[{index}]")
                 for index in range(count))


def _scope_directory_name(scope_key: tuple[int, ...]) -> str:
    """Return a deterministic path component for one Runtime scope."""
    checked = _key(scope_key, label="runtime scope", empty=False)
    digest = digest_bytes(encode_integer_tuple(checked))
    return "scope-" + "".join(f"{item:02x}" for item in digest)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialObservationRecord:
    """一条 observation 的可跨语言元数据和证据身份。"""

    ordinal: int
    event_key: tuple[int, ...]
    memory_item_key: tuple[int, ...]
    source_key: tuple[int, ...]
    observation_id: str
    context_id: str
    family_id: str
    source_namespace: str
    split: str
    raw_observation_record: tuple[int, ...]
    lexical_records: tuple[tuple[int, ...], ...]
    relation_records: tuple[tuple[int, ...], ...]
    observation_stable_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise RuntimeMaterialPersistenceError("observation ordinal 必须为正整数")
        for value, label in (
                (self.event_key, "event_key"),
                (self.memory_item_key, "memory_item_key"),
                (self.source_key, "source_key"),
                (self.raw_observation_record, "raw_observation_record"),
                (self.observation_stable_key, "observation_stable_key")):
            _key(value, label=label, empty=False)
        for value, label in ((self.observation_id, "observation_id"),
                             (self.context_id, "context_id"),
                             (self.family_id, "family_id"),
                             (self.source_namespace, "source_namespace"),
                             (self.split, "split")):
            _text(value, label=label)
        if not isinstance(self.lexical_records, tuple) or not self.lexical_records:
            raise RuntimeMaterialPersistenceError("lexical_records 不能为空")
        if not isinstance(self.relation_records, tuple):
            raise RuntimeMaterialPersistenceError("relation_records 必须是 tuple")

    @classmethod
    def from_observation(cls, observation: RuntimeMaterialLanguageObservation,
                         *, ordinal: int) -> "RuntimeMaterialObservationRecord":
        if not isinstance(observation, RuntimeMaterialLanguageObservation):
            raise TypeError("observation 类型错误")
        raw = observation.raw_observation
        return cls(
            ordinal,
            observation.ingest.event.event_key,
            observation.ingest.event.memory_item_key,
            observation.ingest.event.capsule.source.stable_key(),
            raw.observation_id, raw.context_id, raw.family_id,
            raw.source_namespace, raw.split,
            raw.canonical_record(),
            tuple(item.canonical_record() for item in observation.lexical_evidence),
            tuple(item.proposition.canonical_record()
                  for item in observation.relation_candidates),
            observation.stable_key(),
        )

    def canonical_record(self) -> tuple[int, ...]:
        result = [RUNTIME_MATERIAL_OBSERVATION_LEDGER_VERSION, self.ordinal]
        for value, label in ((self.event_key, "event_key"),
                             (self.memory_item_key, "memory_item_key"),
                             (self.source_key, "source_key")):
            _pack(result, value, label=label, empty=False)
        for value, label in ((self.observation_id, "observation_id"),
                             (self.context_id, "context_id"),
                             (self.family_id, "family_id"),
                             (self.source_namespace, "source_namespace"),
                             (self.split, "split")):
            _pack(result, _text(value, label=label), label=label, empty=False)
        _pack(result, self.raw_observation_record,
              label="raw_observation_record", empty=False)
        _pack_nested(result, self.lexical_records, label="lexical_records")
        _pack_nested(result, self.relation_records, label="relation_records")
        _pack(result, self.observation_stable_key,
              label="observation_stable_key", empty=False)
        return tuple(result)


def _decode_observation_record(record: tuple[int, ...]) -> RuntimeMaterialObservationRecord:
    reader = IntegerStreamReader(record)
    try:
        if reader.read_positive(label="observation protocol") != RUNTIME_MATERIAL_OBSERVATION_LEDGER_VERSION:
            raise RuntimeMaterialPersistenceError("observation protocol 未注册")
        ordinal = reader.read_positive(label="observation ordinal")
    except (IntegerCodecError, ValueError) as error:
        raise RuntimeMaterialPersistenceError("observation header 损坏") from error
    event_key = _read_key(reader, label="observation event", empty=False)
    item_key = _read_key(reader, label="observation memory item", empty=False)
    source_key = _read_key(reader, label="observation source", empty=False)
    texts = tuple(_decode_text(_read_key(reader, label=f"observation text[{i}]"),
                               label=f"observation text[{i}]") for i in range(5))
    raw = _read_key(reader, label="raw observation", empty=False)
    lexical = _read_nested(reader, label="lexical records", allow_empty=False)
    relation = _read_nested(reader, label="relation records")
    stable = _read_key(reader, label="observation stable key", empty=False)
    try:
        reader.finish()
    except IntegerCodecError as error:
        raise RuntimeMaterialPersistenceError("observation record 含尾随整数") from error
    result = RuntimeMaterialObservationRecord(
        ordinal, event_key, item_key, source_key,
        texts[0], texts[1], texts[2], texts[3], texts[4],
        raw, lexical, relation, stable,
    )
    if result.canonical_record() != record:
        raise RuntimeMaterialPersistenceError("observation canonical record 漂移")
    return result


@dataclass(frozen=True, slots=True)
class RuntimeMaterialRuntimeRecovery:
    """回读后的各 Runtime scope state 与严格排序的 observation descriptors."""

    runtime_states: tuple[RuntimeMemoryState, ...]
    observations: tuple[RuntimeMaterialObservationRecord, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.runtime_states, tuple)
                or not self.runtime_states
                or any(not isinstance(item, RuntimeMemoryState)
                       for item in self.runtime_states)):
            raise RuntimeMaterialPersistenceError("runtime_states 不能为空")
        keys = tuple(item.scope_key for item in self.runtime_states)
        if len(set(keys)) != len(keys):
            raise RuntimeMaterialPersistenceError("runtime scope 不得重复")
        if tuple(sorted(self.runtime_states, key=lambda item: item.scope_key)) != self.runtime_states:
            raise RuntimeMaterialPersistenceError("runtime scope 未规范排序")
        if (not isinstance(self.observations, tuple)
                or any(not isinstance(item, RuntimeMaterialObservationRecord)
                       for item in self.observations)):
            raise RuntimeMaterialPersistenceError("observations 类型错误")

    @property
    def runtime_state(self) -> RuntimeMemoryState:
        """兼容单 scope 调用方；多 scope 必须按 runtime_states 消费。"""
        if len(self.runtime_states) != 1:
            raise RuntimeMaterialPersistenceError(
                "多 scope recovery 不存在唯一 runtime_state")
        return self.runtime_states[0]


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class RuntimeMaterialSQLiteRuntime:
    """一个 Runtime SQLite 打开句柄及其已装配的 Companion 上下文。"""

    backend: SQLiteBackend
    context: object
    source_records: SourceRecordRepository

    def close(self) -> None:
        """关闭本次终端专用 SQLite 句柄。"""
        self.backend.close()


def _canonical_manifest_json(value: object) -> bytes:
    """Encode a manifest with the same deterministic JSON contract everywhere."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _manifest_relative_path(value: object, *, label: str) -> str:
    """Accept only release-relative POSIX paths (never host paths)."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeMaterialPersistenceError(f"{label} 路径非法")
    path = Path(value)
    if (path.is_absolute() or ":" in value
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise RuntimeMaterialPersistenceError(f"{label} 必须是相对 POSIX 路径")
    return value


def _manifest_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeMaterialPersistenceError(
            f"Runtime manifest 无法读取文件: {path.name}") from error


def _runtime_manifest_files(root: Path) -> tuple[dict[str, object], ...]:
    """Return the closed payload file inventory, excluding the manifest itself."""
    manifest_name = Path(RUNTIME_MATERIAL_MANIFEST_RELATIVE).as_posix()
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeMaterialPersistenceError(
                "Runtime manifest 不允许符号链接")
        relative = path.relative_to(root).as_posix()
        if relative == manifest_name:
            continue
        rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _manifest_sha256(path),
        })
    if not rows:
        raise RuntimeMaterialPersistenceError("Runtime manifest payload 为空")
    return tuple(rows)


def _runtime_manifest_source_rows(
        source_records: SourceRecordRepository,
        ) -> tuple[dict[str, object], ...]:
    """Project SourceRecord rows without copying user text into the manifest."""
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    try:
        raw_rows = source_records.backend.select(
            SOURCE_RECORD_TABLE, where=None)
    except Exception as error:
        raise RuntimeMaterialPersistenceError(
            "Runtime manifest 无法读取 SourceRecord 表") from error
    result: list[dict[str, object]] = []
    seen: set[tuple[int, ...]] = set()
    for row in raw_rows:
        try:
            record = source_records.read(int(row["source_hash"]))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise RuntimeMaterialPersistenceError(
                "Runtime manifest SourceRecord 无法回读") from error
        if record.source_key in seen:
            raise RuntimeMaterialPersistenceError(
                "Runtime manifest SourceRef identity 重复")
        seen.add(record.source_key)
        result.append({
            "source_key": list(record.source_key),
            "source_hash": record.source_hash,
            "text_hash": record.text_hash,
            "codepoint_count": record.codepoint_count,
            "license_id": record.license_id,
            "batch_id": record.batch_id,
            "companion_type_hash": record.companion_type_hash,
            "companion_name_hash": record.companion_name_hash,
            "companion_assoc_id": record.companion_assoc_id,
        })
    return tuple(sorted(result, key=lambda item: tuple(item["source_key"])))


def _read_runtime_manifest(root: Path) -> dict[str, object] | None:
    path = root / RUNTIME_MATERIAL_MANIFEST_RELATIVE
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeMaterialPersistenceError(
            "Runtime manifest 必须是普通文件")
    try:
        payload = path.read_bytes()
        if payload[:3] == b"\xef\xbb\xbf":
            raise RuntimeMaterialPersistenceError(
                "Runtime manifest 不得包含 UTF-8 BOM")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeMaterialPersistenceError(
            "Runtime manifest 不可回读") from error
    if not isinstance(value, dict):
        raise RuntimeMaterialPersistenceError("Runtime manifest 根必须是对象")
    if (value.get("format") != RUNTIME_MATERIAL_MANIFEST_FORMAT
            or value.get("schema_version") != RUNTIME_MATERIAL_MANIFEST_SCHEMA_VERSION):
        raise RuntimeMaterialPersistenceError("Runtime manifest 格式不兼容")
    return value


def validate_runtime_material_manifest(
        root_path: str | Path,
        *,
        source_records: SourceRecordRepository,
        database_path: str | Path | None = None,
        require_k_drive: bool = True,
        ) -> bool:
    """Validate an optional Runtime ledger manifest and its SourceRef closure.

    A missing manifest is accepted for backwards compatibility.  Once present,
    the manifest is authoritative: every payload file must be listed exactly
    once, hashes and sizes must match, and every listed SourceRecord must be
    present with identical integer identity and license metadata.
    """
    root = open_existing_run_root(
        root_path, require_k_drive=require_k_drive,
        label="runtime material manifest root")
    value = _read_runtime_manifest(root.path)
    if value is None:
        return False
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeMaterialPersistenceError("Runtime manifest files 不能为空")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeMaterialPersistenceError("Runtime manifest file entry 非法")
        relative = _manifest_relative_path(item.get("path"), label="Runtime manifest")
        if relative in seen or relative == RUNTIME_MATERIAL_MANIFEST_RELATIVE:
            raise RuntimeMaterialPersistenceError("Runtime manifest 文件路径重复")
        seen.add(relative)
        path = root.path / Path(relative)
        try:
            path.relative_to(root.path)
        except ValueError as error:
            raise RuntimeMaterialPersistenceError("Runtime manifest 路径越界") from error
        if path.is_symlink() or not path.is_file():
            raise RuntimeMaterialPersistenceError("Runtime manifest 文件缺失或为链接")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (type(size) is not int or size < 0
                or type(digest) is not str or len(digest) != 64):
            raise RuntimeMaterialPersistenceError("Runtime manifest 文件摘要非法")
        if path.stat().st_size != size or _manifest_sha256(path) != digest:
            raise RuntimeMaterialPersistenceError(
                f"Runtime manifest 文件摘要漂移: {relative}")
    actual = {
        path.relative_to(root.path).as_posix()
        for path in root.path.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.relative_to(root.path).as_posix()
        != RUNTIME_MATERIAL_MANIFEST_RELATIVE
    }
    if actual != seen:
        raise RuntimeMaterialPersistenceError(
            "Runtime manifest 文件集合不闭合")
    sqlite = value.get("sqlite")
    if not isinstance(sqlite, dict):
        raise RuntimeMaterialPersistenceError("Runtime manifest 缺少 sqlite")
    sqlite_relative = _manifest_relative_path(
        sqlite.get("path"), label="Runtime manifest sqlite")
    if sqlite_relative != "runtime.sqlite3":
        raise RuntimeMaterialPersistenceError("Runtime manifest sqlite 路径漂移")
    sqlite_path = root.path / sqlite_relative
    if database_path is not None:
        candidate = Path(database_path).resolve()
        if candidate != sqlite_path.resolve():
            raise RuntimeMaterialPersistenceError("Runtime manifest sqlite identity 漂移")
    entry = next((item for item in files if item.get("path") == sqlite_relative), None)
    if entry is None or sqlite.get("sha256") != entry.get("sha256"):
        raise RuntimeMaterialPersistenceError("Runtime manifest sqlite 摘要缺失")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise RuntimeMaterialPersistenceError("Runtime manifest sources 非法")
    actual_rows = _runtime_manifest_source_rows(source_records)
    expected_by_key = {tuple(item["source_key"]): item for item in sources
                       if isinstance(item, dict)
                       and isinstance(item.get("source_key"), list)}
    if len(expected_by_key) != len(sources) or len(expected_by_key) != len(actual_rows):
        raise RuntimeMaterialPersistenceError("Runtime manifest SourceRef 集合不闭合")
    for actual_row in actual_rows:
        expected = expected_by_key.get(tuple(actual_row["source_key"]))
        if expected != actual_row:
            raise RuntimeMaterialPersistenceError(
                "Runtime manifest SourceRef/许可 identity 漂移")
    return True


def write_runtime_material_manifest(
        root: KRunRoot,
        *,
        source_records: SourceRecordRepository,
        database_path: str | Path | None = None,
        ) -> Path:
    """Publish the closed Runtime ledger manifest after all append writes commit."""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 必须是 KRunRoot")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    database = (root.path / "runtime.sqlite3" if database_path is None
                else Path(database_path).resolve())
    if database != (root.path / "runtime.sqlite3").resolve() or not database.is_file():
        raise RuntimeMaterialPersistenceError("Runtime manifest sqlite 必须位于 root/runtime.sqlite3")
    # Validate all event/observation/source closures before publishing identity.
    recovery = load_runtime_material_runtime(
        root.path, source_records=source_records,
        require_k_drive=not root.test_transport)
    files = _runtime_manifest_files(root.path)
    sqlite_entry = next(item for item in files if item["path"] == "runtime.sqlite3")
    payload = {
        "format": RUNTIME_MATERIAL_MANIFEST_FORMAT,
        "schema_version": RUNTIME_MATERIAL_MANIFEST_SCHEMA_VERSION,
        "sqlite": {
            "path": "runtime.sqlite3",
            "size_bytes": sqlite_entry["size_bytes"],
            "sha256": sqlite_entry["sha256"],
        },
        "files": list(files),
        "sources": list(_runtime_manifest_source_rows(source_records)),
        "scopes": [
            {"scope_key": list(state.scope_key), "event_count": len(state.events)}
            for state in recovery.runtime_states
        ],
        "observation_count": len(recovery.observations),
    }
    return write_exclusive_bytes(
        root, RUNTIME_MATERIAL_MANIFEST_RELATIVE,
        _canonical_manifest_json(payload), label="runtime material manifest")


def open_runtime_material_sqlite(
        database_path: str | Path,
        *,
        require_k_drive: bool = True,
        ) -> RuntimeMaterialSQLiteRuntime:
    """从现有 K 盘 Runtime SQLite 自动装配 observation 重放上下文。

    ``make_train_context`` 只注册/核验既有表和 Companion，不读取训练课程，
    因此该入口不会把 Runtime 资料提升到 Core；资料正文仍只由
    ``SourceRecordRepository`` 按 source identity 回读。
    """
    path = Path(database_path).resolve()
    if type(require_k_drive) is not bool:
        raise TypeError("require_k_drive 必须是 bool")
    if (require_k_drive and path.drive.upper() != "K:") or not path.is_file():
        raise RuntimeMaterialPersistenceError(
            "runtime material SQLite 必须是 K 盘已存在文件")
    try:
        from pure_integer_ai.experiments.train_context import make_train_context
        backend = SQLiteBackend(str(path))
        context = make_train_context(backend, companion=True)
        source_records = SourceRecordRepository(backend)
        validate_runtime_material_manifest(
            path.parent,
            source_records=source_records,
            database_path=path,
            require_k_drive=require_k_drive,
        )
    except Exception:
        try:
            backend.close()
        except (UnboundLocalError, AttributeError):
            pass
        raise
    return RuntimeMaterialSQLiteRuntime(backend, context, source_records)


def _event_record(state: RuntimeMemoryState, ordinal: int) -> tuple[int, ...]:
    if type(ordinal) is not int or ordinal < 1 or ordinal > len(state.events):
        raise RuntimeMaterialPersistenceError("event ordinal 越界")
    event = state.events[ordinal - 1]
    # Store one event and its scope, never a cumulative state snapshot.  This
    # keeps append cost and ledger size linear for long-lived Runtime Memory.
    state_record = tuple(encode_runtime_memory_state(
        RuntimeMemoryState(state.scope_key, (event,))))
    result = [RUNTIME_MATERIAL_EVENT_LEDGER_VERSION, ordinal]
    _pack(result, state.scope_key, label="runtime scope", empty=False)
    _pack(result, event.event_key, label="event key", empty=False)
    _pack(result, state_record, label="runtime state", empty=False)
    _pack(result, digest_bytes(encode_integer_tuple(state_record)),
          label="runtime state digest", empty=False)
    return tuple(result)


def _decode_event_record(record: tuple[int, ...]) -> tuple[int, tuple[int, ...], RuntimeMemoryEvent]:
    reader = IntegerStreamReader(record)
    try:
        if reader.read_positive(label="event protocol") != RUNTIME_MATERIAL_EVENT_LEDGER_VERSION:
            raise RuntimeMaterialPersistenceError("event protocol 未注册")
        ordinal = reader.read_positive(label="event ordinal")
    except (IntegerCodecError, ValueError) as error:
        raise RuntimeMaterialPersistenceError("event header 损坏") from error
    scope_key = _read_key(reader, label="runtime scope", empty=False)
    event_key = _read_key(reader, label="event key", empty=False)
    state_record = _read_key(reader, label="runtime state", empty=False)
    digest = _read_key(reader, label="runtime state digest", empty=False)
    try:
        reader.finish()
        singleton_state = decode_runtime_memory_state(state_record)
    except (IntegerCodecError, ValueError, PersistentDialogueRunError) as error:
        raise RuntimeMaterialPersistenceError("runtime state 不可恢复") from error
    if digest_bytes(encode_integer_tuple(state_record)) != digest:
        raise RuntimeMaterialPersistenceError("runtime state digest 漂移")
    if len(singleton_state.events) != 1:
        raise RuntimeMaterialPersistenceError("event record 必须只含一个 event")
    event = singleton_state.events[0]
    if singleton_state.scope_key != scope_key or event.event_key != event_key:
        raise RuntimeMaterialPersistenceError("event/state identity 漂移")
    return ordinal, scope_key, event


def persist_runtime_material_observation(
        root: KRunRoot,
        observation: RuntimeMaterialLanguageObservation,
        *,
        relative_events: str | Path = RUNTIME_MATERIAL_EVENT_RELATIVE,
        relative_observations: str | Path = RUNTIME_MATERIAL_OBSERVATION_RELATIVE,
        ) -> tuple[Path, Path]:
    """排他追加一份 Runtime state 和对应 observation descriptor。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 必须是 KRunRoot")
    if not isinstance(observation, RuntimeMaterialLanguageObservation):
        raise TypeError("observation 类型错误")
    scope_name = _scope_directory_name(observation.ingest.memory_after.scope_key)
    event_dir = ensure_normal_relative_directory(
        root, Path(relative_events) / scope_name,
        label="runtime material event directory")
    observation_dir = ensure_normal_relative_directory(
        root, Path(relative_observations) / scope_name,
        label="runtime material observation directory")
    ordinal = len(observation.ingest.memory_after.events)
    descriptor = RuntimeMaterialObservationRecord.from_observation(
        observation, ordinal=ordinal)
    event_relative = Path(relative_events) / scope_name / f"event-{ordinal:020d}.int"
    observation_relative = (
        Path(relative_observations) / scope_name
        / f"observation-{ordinal:020d}.int")
    event_path = write_exclusive_bytes(
        root, event_relative,
        encode_integer_tuple(_event_record(observation.ingest.memory_after, ordinal)),
        label="runtime material event ledger")
    observation_path = write_exclusive_bytes(
        root, observation_relative,
        encode_integer_tuple(descriptor.canonical_record()),
        label="runtime material observation ledger")
    if event_path.parent != event_dir.path or observation_path.parent != observation_dir.path:
        raise RuntimeMaterialPersistenceError("runtime material ledger directory 漂移")
    return event_path, observation_path


def load_runtime_material_runtime(
        root_path: str | Path,
        *,
        source_records: SourceRecordRepository,
        relative_events: str | Path = RUNTIME_MATERIAL_EVENT_RELATIVE,
        relative_observations: str | Path = RUNTIME_MATERIAL_OBSERVATION_RELATIVE,
        require_k_drive: bool = True,
        ) -> RuntimeMaterialRuntimeRecovery:
    """严格回读 event/observation ledger，不读取或修改 Core。"""
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    root = open_existing_run_root(root_path, require_k_drive=require_k_drive,
                                  label="runtime material ledger root")
    validate_runtime_material_manifest(
        root.path, source_records=source_records,
        require_k_drive=require_k_drive,
    )
    event_base = ensure_normal_relative_directory(
        root, relative_events, label="runtime material event directory")
    observation_base = ensure_normal_relative_directory(
        root, relative_observations,
        label="runtime material observation directory")

    # A flat directory is accepted for the original single-scope v2 ledger;
    # new writes use one deterministic child directory per Runtime scope.
    event_groups: dict[str, list[tuple[int, Path, Path]]] = {"": []}
    event_directories = [("", event_base.path, Path(relative_events))]
    for child in event_base.path.iterdir():
        if child.is_dir() and child.name.startswith("scope-"):
            event_groups[child.name] = []
            event_directories.append((
                child.name, child,
                Path(relative_events) / child.name))
    for group_name, directory, relative_directory in event_directories:
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix != ".int" or not path.name.startswith("event-"):
                continue
            token = path.stem.removeprefix("event-")
            if not token.isdigit() or len(token) != 20:
                raise RuntimeMaterialPersistenceError("event 文件名非法")
            event_groups[group_name].append((
                int(token), path, relative_directory / path.name))
    if not any(event_groups.values()):
        raise RuntimeMaterialPersistenceError("Runtime event ledger 为空")

    states: list[RuntimeMemoryState] = []
    event_by_key: dict[tuple[int, ...], tuple[RuntimeMemoryState, int, RuntimeMemoryEvent]] = {}
    for group_name in sorted(event_groups):
        files = sorted(event_groups[group_name], key=lambda item: item[0])
        if not files:
            continue
        events: list[RuntimeMemoryEvent] = []
        expected = 1
        scope_key: tuple[int, ...] | None = None
        for ordinal, path, relative in files:
            if ordinal != expected:
                raise RuntimeMaterialPersistenceError("event ledger 序号不连续")
            with open_plain_binary(root, relative,
                                   label="runtime material event ledger") as stream:
                payload = stream.read()
            try:
                decoded = decode_integer_tuple(payload)
            except (TypeError, ValueError) as error:
                raise RuntimeMaterialPersistenceError(
                    "event ledger 整数流损坏") from error
            decoded_ordinal, decoded_scope, event = _decode_event_record(decoded)
            if decoded_ordinal != ordinal:
                raise RuntimeMaterialPersistenceError("event 文件序号漂移")
            if scope_key is None:
                scope_key = decoded_scope
            elif decoded_scope != scope_key:
                raise RuntimeMaterialPersistenceError("event group scope 漂移")
            events.append(event)
            expected += 1
        if scope_key is None:
            raise RuntimeMaterialPersistenceError("event group scope 缺失")
        state = RuntimeMemoryState(scope_key, tuple(events))
        states.append(state)
        for index, event in enumerate(events):
            if event.event_key in event_by_key:
                raise RuntimeMaterialPersistenceError("Runtime event identity 重复")
            event_by_key[event.event_key] = (state, index + 1, event)
    states_tuple = tuple(sorted(states, key=lambda item: item.scope_key))

    # Every Runtime event must remain source-backed, even when a conflict or
    # revision has no language observation descriptor of its own.
    for state in states_tuple:
        for event in state.events:
            source_record = source_records.find(event.capsule.source.stable_key())
            if source_record is None:
                raise RuntimeMaterialPersistenceError(
                    "Runtime event 缺少 SourceRecord")
            if not source_record.metadata_complete:
                raise RuntimeMaterialPersistenceError(
                    "Runtime event SourceRecord metadata 不完整")
            if digest_bytes(source_record.raw_text.encode("utf-8")) != (
                    event.capsule.raw_content_digest):
                raise RuntimeMaterialPersistenceError(
                    "Runtime event SourceRecord raw digest 漂移")

    observation_files: list[tuple[int, Path, Path]] = []
    observation_directories = [(observation_base.path, Path(relative_observations))]
    for child in observation_base.path.iterdir():
        if child.is_dir() and child.name.startswith("scope-"):
            observation_directories.append((
                child, Path(relative_observations) / child.name))
    for directory, relative_directory in observation_directories:
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix != ".int" or not path.name.startswith("observation-"):
                continue
            token = path.stem.removeprefix("observation-")
            if not token.isdigit() or len(token) != 20:
                raise RuntimeMaterialPersistenceError("observation 文件名非法")
            observation_files.append((int(token), path, relative_directory / path.name))
    if len(observation_files) > len(event_by_key):
        raise RuntimeMaterialPersistenceError("observation 超出 Runtime event 数量")
    descriptors: list[RuntimeMaterialObservationRecord] = []
    seen_observation_events: set[tuple[int, ...]] = set()
    for ordinal, path, relative in sorted(observation_files, key=lambda item: (item[0], str(item[2]))):
        with open_plain_binary(root, relative,
                               label="runtime material observation ledger") as stream:
            payload = stream.read()
        try:
            descriptor = _decode_observation_record(decode_integer_tuple(payload))
        except (TypeError, ValueError, IntegerCodecError) as error:
            raise RuntimeMaterialPersistenceError("observation ledger 不可读取") from error
        if descriptor.ordinal != ordinal:
            raise RuntimeMaterialPersistenceError("observation 序号漂移")
        event_entry = event_by_key.get(descriptor.event_key)
        if event_entry is None:
            raise RuntimeMaterialPersistenceError("observation 指向未知 Runtime event")
        state, event_ordinal, event = event_entry
        if ordinal != event_ordinal:
            raise RuntimeMaterialPersistenceError("observation/event ordinal 漂移")
        if descriptor.event_key in seen_observation_events:
            raise RuntimeMaterialPersistenceError("observation event 重复")
        seen_observation_events.add(descriptor.event_key)
        if (descriptor.event_key != event.event_key
                or descriptor.memory_item_key != event.memory_item_key
                or descriptor.source_key != event.capsule.source.stable_key()):
            raise RuntimeMaterialPersistenceError("observation/event identity 漂移")
        source_record = source_records.find(descriptor.source_key)
        if source_record is None:
            raise RuntimeMaterialPersistenceError("observation 缺少 SourceRecord")
        if digest_bytes(source_record.raw_text.encode("utf-8")) != event.capsule.raw_content_digest:
            raise RuntimeMaterialPersistenceError("SourceRecord 与 Runtime capsule 漂移")
        descriptors.append(descriptor)
    descriptors.sort(key=lambda item: (item.source_key, item.ordinal,
                                       item.event_key))
    return RuntimeMaterialRuntimeRecovery(states_tuple, tuple(descriptors))


def rebuild_runtime_material_observations(
        ctx,
        recovery: RuntimeMaterialRuntimeRecovery,
        *,
        source_records: SourceRecordRepository,
        ) -> tuple[RuntimeMaterialLanguageObservation, ...]:
    """由账本元数据自动重放 observation，并逐项核验 canonical identity。"""
    if not isinstance(recovery, RuntimeMaterialRuntimeRecovery):
        raise TypeError("recovery 类型错误")
    if not isinstance(source_records, SourceRecordRepository):
        raise TypeError("source_records 类型错误")
    event_lookup: dict[tuple[int, ...], tuple[RuntimeMemoryState, int, RuntimeMemoryEvent]] = {}
    for state in recovery.runtime_states:
        for ordinal, event in enumerate(state.events, start=1):
            event_lookup[event.event_key] = (state, ordinal, event)
    result: list[RuntimeMaterialLanguageObservation] = []
    for descriptor in recovery.observations:
        event_entry = event_lookup.get(descriptor.event_key)
        if event_entry is None:
            raise RuntimeMaterialPersistenceError("重建 observation 指向未知 event")
        state, ordinal, event = event_entry
        if ordinal != descriptor.ordinal:
            raise RuntimeMaterialPersistenceError("重建 observation ordinal 漂移")
        source_record = source_records.find(event.capsule.source.stable_key())
        if source_record is None:
            raise RuntimeMaterialPersistenceError("重建缺少 SourceRecord")
        before = RuntimeMemoryState(
            state.scope_key,
            state.events[:ordinal - 1],
        )
        replay_key = _replay_key(capsule=event.capsule, event=event,
                                 source_record=source_record)
        ingest = RuntimeMaterialIngest(
            event.capsule, event, source_record, before,
            state, ADMISSION_ACCEPTED,
            LearningReplayReceipt.from_runtime_event(event, replay_key=replay_key),
        )
        observed = observe_runtime_material_language(
            ctx, ingest,
            observation_id=descriptor.observation_id,
            context_id=descriptor.context_id,
            family_id=descriptor.family_id,
            source_namespace=descriptor.source_namespace,
            split=descriptor.split,
        )
        current = RuntimeMaterialObservationRecord.from_observation(
            observed, ordinal=descriptor.ordinal)
        if current != descriptor:
            raise RuntimeMaterialPersistenceError(
                "重放 observation/evidence identity 与账本不一致")
        result.append(observed)
    return tuple(result)


__all__ = [
    "RUNTIME_MATERIAL_EVENT_LEDGER_VERSION",
    "RUNTIME_MATERIAL_EVENT_RELATIVE",
    "RUNTIME_MATERIAL_OBSERVATION_LEDGER_VERSION",
    "RUNTIME_MATERIAL_OBSERVATION_RELATIVE",
    "RUNTIME_MATERIAL_PERSISTENCE_VERSION",
    "RuntimeMaterialObservationRecord",
    "RuntimeMaterialPersistenceError",
    "RuntimeMaterialRuntimeRecovery",
    "RuntimeMaterialSQLiteRuntime",
    "load_runtime_material_runtime",
    "open_runtime_material_sqlite",
    "persist_runtime_material_observation",
    "rebuild_runtime_material_observations",
    "validate_runtime_material_manifest",
    "write_runtime_material_manifest",
]
