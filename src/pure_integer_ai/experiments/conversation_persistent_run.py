"""M5 可恢复的对话/Core/Runtime 增量运行边界。

本模块把已经完成的双平面 transition 写入 K 盘上的不可变整数检查点。
检查点不保存 Python 对象、路径或表层文本；所有字段都通过 canonical integer
record 表达，RAW-04 会话复用既有 typed snapshot codec。每个检查点排他创建，
通过前一检查点身份形成连续链；恢复时一次建立 memory-item 索引，查询只读取
索引命中的事件，不在每一轮重新扫描完整 Runtime ledger。
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    CoreDelta,
    CoreLearningState,
    LearningInputCapsule,
    RuntimeMemoryEvent,
    RuntimeMemoryState,
)
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_session import (
    ConversationRawDialogueState,
)
from pure_integer_ai.experiments.conversation_raw_dialogue_snapshot import (
    restore_public_frame_dialogue_state,
    snapshot_public_frame_dialogue_state,
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
    require_plain_file,
    write_exclusive_bytes,
)


PERSISTENT_DIALOGUE_CHECKPOINT_V1 = 1
PERSISTENT_DIALOGUE_CAPSULE_V1 = 1
PERSISTENT_DIALOGUE_CORE_DELTA_V1 = 1
PERSISTENT_DIALOGUE_RUNTIME_EVENT_V1 = 1
PERSISTENT_DIALOGUE_CORE_STATE_V1 = 1
PERSISTENT_DIALOGUE_RUNTIME_STATE_V1 = 1
PERSISTENT_DIALOGUE_DEFAULT_DIR = "m5_dialogue_checkpoints"


# object-model: exception; interop=portable
class PersistentDialogueRunError(ValueError):
    """持久化检查点、链、K 盘边界或恢复索引不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    if not isinstance(value, tuple) or any(type(item) is not int for item in value):
        raise PersistentDialogueRunError("检查点字段必须是整数 tuple")
    result.extend((len(value), *value))


def _read_key(reader: IntegerStreamReader, *, label: str) -> tuple[int, ...]:
    try:
        return reader.read_key(label=label, empty=True)
    except (IntegerCodecError, ValueError) as error:
        raise PersistentDialogueRunError(f"{label} 不可读取") from error


def _read_count(reader: IntegerStreamReader, *, label: str) -> int:
    try:
        return reader.read_nonnegative(label=label)
    except (IntegerCodecError, ValueError) as error:
        raise PersistentDialogueRunError(f"{label} 不可读取") from error


def _encode_capsule(capsule: LearningInputCapsule) -> tuple[int, ...]:
    if not isinstance(capsule, LearningInputCapsule):
        raise TypeError("capsule 类型错误")
    result = [PERSISTENT_DIALOGUE_CAPSULE_V1]
    for value in (
            capsule.source.stable_key(), capsule.scope.stable_key(),
            capsule.version_key, capsule.parent_version_key):
        _pack(result, value)
    result.extend((capsule.language, capsule.modality))
    _pack(result, capsule.raw_content_digest)
    result.append(len(capsule.structural_units))
    for unit in capsule.structural_units:
        _pack(result, unit)
    _pack(result, capsule.authority_key)
    _pack(result, tuple(capsule.license_id.encode("utf-8")))
    result.extend((capsule.split, capsule.delta_sequence))
    _pack(result, capsule.canonical_record)
    return tuple(result)


def _decode_capsule(record: tuple[int, ...]) -> LearningInputCapsule:
    reader = IntegerStreamReader(record)
    version = reader.read_positive(label="capsule codec version")
    if version != PERSISTENT_DIALOGUE_CAPSULE_V1:
        raise PersistentDialogueRunError("capsule codec version 未注册")
    source_key = _read_key(reader, label="capsule source")
    scope_key = _read_key(reader, label="capsule scope")
    version_key = _read_key(reader, label="capsule version")
    parent_key = _read_key(reader, label="capsule parent version")
    language = reader.read_positive(label="capsule language")
    modality = reader.read_positive(label="capsule modality")
    raw_digest = _read_key(reader, label="capsule raw digest")
    unit_count = _read_count(reader, label="capsule structural unit count")
    units = tuple(_read_key(reader, label=f"capsule structural unit[{i}]")
                  for i in range(unit_count))
    authority_key = _read_key(reader, label="capsule authority")
    license_bytes = _read_key(reader, label="capsule license bytes")
    if any(item < 0 or item > 255 for item in license_bytes):
        raise PersistentDialogueRunError("capsule license bytes 越界")
    try:
        license_id = bytes(license_bytes).decode("utf-8")
        source = SourceRef.from_stable_key(source_key)
        scope = ScopeIdentity.from_stable_key(scope_key)
    except (UnicodeDecodeError, TypeError, ValueError) as error:
        raise PersistentDialogueRunError("capsule identity 或 license 不可恢复") from error
    split = reader.read_positive(label="capsule split")
    delta_sequence = reader.read_positive(label="capsule delta sequence")
    expected_canonical = _read_key(reader, label="capsule canonical")
    try:
        capsule = LearningInputCapsule(
            source, scope, version_key, parent_key, language, modality,
            raw_digest, units, authority_key, license_id, split, delta_sequence,
        )
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("capsule 字段无法恢复") from error
    if capsule.canonical_record != expected_canonical:
        raise PersistentDialogueRunError("capsule canonical record 漂移")
    reader.finish()
    return capsule


def _encode_core_delta(delta: CoreDelta) -> tuple[int, ...]:
    if not isinstance(delta, CoreDelta):
        raise TypeError("core delta 类型错误")
    result = [PERSISTENT_DIALOGUE_CORE_DELTA_V1, delta.status]
    _pack(result, delta.base_state_identity)
    _pack(result, delta.graph_diff)
    _pack(result, _encode_capsule(delta.capsule))
    _pack(result, delta.stable_key())
    return tuple(result)


def _decode_core_delta(record: tuple[int, ...]) -> CoreDelta:
    reader = IntegerStreamReader(record)
    if reader.read_positive(label="core delta codec version") != PERSISTENT_DIALOGUE_CORE_DELTA_V1:
        raise PersistentDialogueRunError("core delta codec version 未注册")
    status = reader.read(label="core delta status")
    base = _read_key(reader, label="core delta base")
    graph_diff = _read_key(reader, label="core delta graph diff")
    capsule = _decode_capsule(_read_key(reader, label="core delta capsule"))
    expected = _read_key(reader, label="core delta stable key")
    try:
        delta = CoreDelta(base, capsule, status=status, graph_diff=graph_diff)
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("core delta 字段无法恢复") from error
    if delta.stable_key() != expected:
        raise PersistentDialogueRunError("core delta stable key 漂移")
    reader.finish()
    return delta


def _encode_runtime_event(event: RuntimeMemoryEvent) -> tuple[int, ...]:
    if not isinstance(event, RuntimeMemoryEvent):
        raise TypeError("runtime event 类型错误")
    result = [PERSISTENT_DIALOGUE_RUNTIME_EVENT_V1, event.event_kind,
              event.revision, 1 if event.tombstone else 0]
    for value in (event.memory_item_key, event.supersedes_event_key,
                  event.conflict_key, _encode_capsule(event.capsule),
                  event.event_key):
        _pack(result, value)
    return tuple(result)


def _decode_runtime_event(record: tuple[int, ...]) -> RuntimeMemoryEvent:
    reader = IntegerStreamReader(record)
    if reader.read_positive(label="runtime event codec version") != PERSISTENT_DIALOGUE_RUNTIME_EVENT_V1:
        raise PersistentDialogueRunError("runtime event codec version 未注册")
    event_kind = reader.read_positive(label="runtime event kind")
    revision = reader.read_positive(label="runtime event revision")
    tombstone = reader.read(label="runtime event tombstone")
    if tombstone not in (0, 1):
        raise PersistentDialogueRunError("runtime event tombstone 非法")
    memory_item_key = _read_key(reader, label="runtime event item")
    supersedes = _read_key(reader, label="runtime event supersedes")
    conflict_key = _read_key(reader, label="runtime event conflict")
    capsule = _decode_capsule(_read_key(reader, label="runtime event capsule"))
    expected = _read_key(reader, label="runtime event key")
    try:
        event = RuntimeMemoryEvent(
            capsule, memory_item_key, event_kind, revision, supersedes,
            bool(tombstone), conflict_key,
        )
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("runtime event 字段无法恢复") from error
    if event.event_key != expected:
        raise PersistentDialogueRunError("runtime event key 漂移")
    reader.finish()
    return event


def _encode_core_state(state: CoreLearningState) -> tuple[int, ...]:
    if not isinstance(state, CoreLearningState):
        raise TypeError("core state 类型错误")
    result = [PERSISTENT_DIALOGUE_CORE_STATE_V1]
    _pack(result, state.scope_key)
    _pack(result, state.base_state_identity)
    result.append(len(state.consumed_item_ledger))
    for key in state.consumed_item_ledger:
        _pack(result, key)
    result.append(len(state.deltas))
    for delta in state.deltas:
        _pack(result, _encode_core_delta(delta))
    return tuple(result)


def _decode_core_state(record: tuple[int, ...]) -> CoreLearningState:
    reader = IntegerStreamReader(record)
    if reader.read_positive(label="core state codec version") != PERSISTENT_DIALOGUE_CORE_STATE_V1:
        raise PersistentDialogueRunError("core state codec version 未注册")
    scope_key = _read_key(reader, label="core state scope")
    base = _read_key(reader, label="core state base")
    ledger = tuple(_read_key(reader, label=f"core ledger[{i}]")
                   for i in range(_read_count(reader, label="core ledger count")))
    deltas = tuple(
        _decode_core_delta(_read_key(reader, label=f"core delta[{i}]"))
        for i in range(_read_count(reader, label="core delta count"))
    )
    reader.finish()
    try:
        return CoreLearningState(scope_key, base, ledger, deltas)
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("core state 无法恢复") from error


def _encode_runtime_state(state: RuntimeMemoryState) -> tuple[int, ...]:
    if not isinstance(state, RuntimeMemoryState):
        raise TypeError("runtime state 类型错误")
    result = [PERSISTENT_DIALOGUE_RUNTIME_STATE_V1]
    _pack(result, state.scope_key)
    result.append(len(state.events))
    for event in state.events:
        _pack(result, _encode_runtime_event(event))
    return tuple(result)


def _decode_runtime_state(record: tuple[int, ...]) -> RuntimeMemoryState:
    reader = IntegerStreamReader(record)
    if reader.read_positive(label="runtime state codec version") != PERSISTENT_DIALOGUE_RUNTIME_STATE_V1:
        raise PersistentDialogueRunError("runtime state codec version 未注册")
    scope_key = _read_key(reader, label="runtime state scope")
    events = tuple(
        _decode_runtime_event(_read_key(reader, label=f"runtime event[{i}]"))
        for i in range(_read_count(reader, label="runtime event count"))
    )
    reader.finish()
    try:
        return RuntimeMemoryState(scope_key, events)
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("runtime state 无法恢复") from error


def _checkpoint_record(
        ordinal: int,
        core_state: CoreLearningState,
        runtime_state: RuntimeMemoryState,
        dialogue_state_record: tuple[int, ...],
        previous_identity: tuple[int, ...],
        ) -> tuple[int, ...]:
    result = [PERSISTENT_DIALOGUE_CHECKPOINT_V1, ordinal]
    for value in (
            _encode_core_state(core_state), _encode_runtime_state(runtime_state),
            dialogue_state_record, previous_identity):
        _pack(result, value)
    return tuple(result)


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class PersistentDialogueCheckpoint:
    """一个可跨进程恢复的双平面状态检查点。"""

    ordinal: int
    core_state: CoreLearningState
    runtime_state: RuntimeMemoryState
    dialogue_state: ConversationRawDialogueState
    previous_identity: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise PersistentDialogueRunError("checkpoint ordinal 必须为正整数")
        if not isinstance(self.core_state, CoreLearningState):
            raise TypeError("checkpoint core_state 类型错误")
        if not isinstance(self.runtime_state, RuntimeMemoryState):
            raise TypeError("checkpoint runtime_state 类型错误")
        if not isinstance(self.dialogue_state, ConversationRawDialogueState):
            raise TypeError("checkpoint dialogue_state 类型错误")
        if self.core_state.scope_key != self.runtime_state.scope_key:
            raise PersistentDialogueRunError("Core/Runtime scope 不一致")
        if not isinstance(self.previous_identity, tuple) or any(
                type(item) is not int or item < 0 for item in self.previous_identity):
            raise PersistentDialogueRunError("previous checkpoint identity 非法")

    def canonical_record(self, runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
        if type(runtime) is not PublicDialogueRuntimeV1:
            raise TypeError("runtime 类型错误")
        return _checkpoint_record(
            self.ordinal, self.core_state, self.runtime_state,
            snapshot_public_frame_dialogue_state(self.dialogue_state, runtime),
            self.previous_identity,
        )

    def identity(self, runtime: PublicDialogueRuntimeV1) -> tuple[int, ...]:
        from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
        return digest_bytes(encode_integer_tuple(self.canonical_record(runtime)))


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class PersistentDialogueRecovery:
    """重启后恢复的最新状态及一次性 memory-item 索引。"""

    checkpoint: PersistentDialogueCheckpoint
    checkpoint_identity: tuple[int, ...]
    indexed_event_count: int
    event_index: tuple[tuple[tuple[int, ...], int], ...]

    def __post_init__(self) -> None:
        if type(self.indexed_event_count) is not int or self.indexed_event_count < 0:
            raise PersistentDialogueRunError("indexed_event_count 非法")
        if len(self.event_index) != self.indexed_event_count:
            raise PersistentDialogueRunError("event index count 漂移")

    def query_event(self, memory_item_key: tuple[int, ...]) -> tuple[RuntimeMemoryEvent | None, int]:
        """按已建索引读取一个事件，返回 (event, physical_read_count)。"""
        position = bisect_left(self.event_index, (memory_item_key, -1))
        if (position < len(self.event_index)
                and self.event_index[position][0] == memory_item_key):
            ordinal = self.event_index[position][1]
            return self.checkpoint.runtime_state.events[ordinal], 1
        return None, 1


def _decode_checkpoint(record: tuple[int, ...], runtime: PublicDialogueRuntimeV1) -> PersistentDialogueCheckpoint:
    reader = IntegerStreamReader(record)
    if reader.read_positive(label="checkpoint version") != PERSISTENT_DIALOGUE_CHECKPOINT_V1:
        raise PersistentDialogueRunError("checkpoint version 未注册")
    ordinal = reader.read_positive(label="checkpoint ordinal")
    core_state = _decode_core_state(_read_key(reader, label="checkpoint core state"))
    runtime_state = _decode_runtime_state(_read_key(reader, label="checkpoint runtime state"))
    dialogue_record = _read_key(reader, label="checkpoint dialogue state")
    previous = _read_key(reader, label="checkpoint previous identity")
    reader.finish()
    try:
        dialogue_state = restore_public_frame_dialogue_state(dialogue_record, runtime)
    except (TypeError, ValueError) as error:
        raise PersistentDialogueRunError("checkpoint dialogue state 无法恢复") from error
    return PersistentDialogueCheckpoint(
        ordinal, core_state, runtime_state, dialogue_state, previous)


def write_dialogue_checkpoint(
        root: KRunRoot,
        checkpoint: PersistentDialogueCheckpoint,
        runtime: PublicDialogueRuntimeV1,
        *,
        relative_dir: str | Path = PERSISTENT_DIALOGUE_DEFAULT_DIR,
        ) -> Path:
    """在 K run 内排他发布一个不可覆盖的检查点文件。"""
    directory = ensure_normal_relative_directory(root, relative_dir,
                                                  label="M5 checkpoint directory")
    existing = tuple(
        item for item in directory.path.iterdir()
        if item.is_file() and item.suffix == ".int"
        and item.name.startswith("checkpoint-")
    )
    if checkpoint.ordinal == 1:
        if existing:
            raise PersistentDialogueRunError(
                "M5 首个 checkpoint 不能覆盖既有链")
    else:
        if not existing:
            raise PersistentDialogueRunError(
                "M5 非首个 checkpoint 缺少前驱链")
        recovery = recover_dialogue_checkpoint(
            root.path, runtime, relative_dir=relative_dir,
            require_k_drive=not root.test_transport,
        )
        if (checkpoint.ordinal != recovery.checkpoint.ordinal + 1
                or checkpoint.previous_identity != recovery.checkpoint_identity):
            raise PersistentDialogueRunError("M5 checkpoint 前驱 identity 或序号不匹配")
    record = checkpoint.canonical_record(runtime)
    filename = f"checkpoint-{checkpoint.ordinal:020d}.int"
    relative = Path(relative_dir) / filename
    try:
        path = write_exclusive_bytes(root, relative, encode_integer_tuple(record),
                                     label="M5 checkpoint")
    except (TypeError, ValueError, OSError) as error:
        raise PersistentDialogueRunError("M5 checkpoint 写入失败") from error
    # directory capability is intentionally retained in the call path; this also
    # catches a caller that accidentally passed a sibling path to the writer.
    if path.parent != directory.path:
        raise PersistentDialogueRunError("M5 checkpoint directory 漂移")
    return path


def recover_dialogue_checkpoint(
        root_path: str | Path,
        runtime: PublicDialogueRuntimeV1,
        *,
        relative_dir: str | Path = PERSISTENT_DIALOGUE_DEFAULT_DIR,
        require_k_drive: bool = True,
        ) -> PersistentDialogueRecovery:
    """关闭进程后从最新连续检查点恢复，并建立一次性 Runtime 索引。"""
    root = open_existing_run_root(root_path, require_k_drive=require_k_drive,
                                  label="M5 dialogue run root")
    directory = ensure_normal_relative_directory(root, relative_dir,
                                                  label="M5 checkpoint directory")
    candidates: list[tuple[int, Path]] = []
    for path in directory.path.iterdir():
        if not path.is_file() or path.suffix != ".int" or not path.name.startswith("checkpoint-"):
            continue
        token = path.stem.removeprefix("checkpoint-")
        if not token.isdigit() or len(token) != 20:
            raise PersistentDialogueRunError("M5 checkpoint 文件名非法")
        candidates.append((int(token), path))
    if not candidates:
        raise PersistentDialogueRunError("M5 没有可恢复 checkpoint")
    candidates.sort()
    previous = ()
    latest: PersistentDialogueCheckpoint | None = None
    latest_identity: tuple[int, ...] = ()
    expected_ordinal = 1
    for ordinal, path in candidates:
        if ordinal != expected_ordinal:
            raise PersistentDialogueRunError("M5 checkpoint 序号不连续")
        relative = Path(relative_dir) / path.name
        require_plain_file(root, relative, label="M5 checkpoint")
        try:
            with open_plain_binary(root, relative, label="M5 checkpoint") as stream:
                payload = stream.read()
        except OSError as error:
            raise PersistentDialogueRunError("M5 checkpoint 读取失败") from error
        try:
            record = decode_integer_tuple(payload)
        except (TypeError, ValueError, IntegerCodecError) as error:
            raise PersistentDialogueRunError("M5 checkpoint 整数流损坏") from error
        checkpoint = _decode_checkpoint(record, runtime)
        if checkpoint.ordinal != ordinal or checkpoint.previous_identity != previous:
            raise PersistentDialogueRunError("M5 checkpoint 链漂移")
        identity = checkpoint.identity(runtime)
        previous = identity
        latest, latest_identity = checkpoint, identity
        expected_ordinal += 1
    if latest is None:
        raise PersistentDialogueRunError("M5 checkpoint 恢复为空")
    # A memory item may have an append-only revision chain.  The hot index keeps
    # one deterministic latest row per item; competing same-revision branches are
    # deliberately not made queryable.
    by_item: dict[tuple[int, ...], tuple[int, RuntimeMemoryEvent]] = {}
    for ordinal, event in enumerate(latest.runtime_state.events):
        previous_item = by_item.get(event.memory_item_key)
        if previous_item is None or event.revision > previous_item[1].revision:
            by_item[event.memory_item_key] = (ordinal, event)
        elif event.revision == previous_item[1].revision:
            if event.event_key != previous_item[1].event_key:
                raise PersistentDialogueRunError(
                    "M5 Runtime memory-item 索引存在竞争 revision")
    index = tuple(sorted(
        ((key, ordinal) for key, (ordinal, _event) in by_item.items()),
        key=lambda item: item[0],
    ))
    return PersistentDialogueRecovery(latest, latest_identity, len(index), index)


__all__ = [
    "PERSISTENT_DIALOGUE_CHECKPOINT_V1",
    "PERSISTENT_DIALOGUE_DEFAULT_DIR",
    "PersistentDialogueCheckpoint",
    "PersistentDialogueRecovery",
    "PersistentDialogueRunError",
    "recover_dialogue_checkpoint",
    "write_dialogue_checkpoint",
]
