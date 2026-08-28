"""广域对话 hot history 的纯整数、可恢复会话检查点。

该模块只持久化 ``BroadDialogueState``，不把广域检索结果写回 Core、训练
SQLite 或来源索引。每一轮生成一个排他整数检查点，恢复时一次读取完整链并
建立有限 hot history 索引；查询阶段不扫描历史文件。
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.learning_input_capsule import (
    RuntimeMemoryEvent,
    RuntimeMemoryState,
    digest_bytes,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueCitation,
    DialogueTurn,
)
from pure_integer_ai.experiments.conversation_persistent_run import (
    PersistentDialogueRunError,
    decode_runtime_memory_state,
    encode_runtime_memory_state,
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


BROAD_DIALOGUE_PERSISTENCE_V1 = 1
BROAD_DIALOGUE_PERSISTENCE_V2 = 2
BROAD_DIALOGUE_PERSISTENCE_V3 = 3
BROAD_DIALOGUE_CHECKPOINT_V1 = 1
BROAD_DIALOGUE_CHECKPOINT_V2 = 2
BROAD_DIALOGUE_DEFAULT_DIR = "broad_dialogue_checkpoints"
BROAD_DIALOGUE_HOT_HISTORY_LIMIT = 8


class BroadDialoguePersistenceError(ValueError):
    """广域会话检查点、编码或恢复链不闭合。"""


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 for item in value)):
        raise BroadDialoguePersistenceError(f"{label} 必须是非负整数 tuple")
    result.extend((len(value), *value))


def _read_key(reader: IntegerStreamReader, *, label: str,
              empty: bool = True) -> tuple[int, ...]:
    try:
        return reader.read_key(label=label, empty=empty)
    except (IntegerCodecError, ValueError) as error:
        raise BroadDialoguePersistenceError(f"{label} 不可读取") from error


def _text(value: str, *, label: str, ascii_only: bool = False) -> tuple[int, ...]:
    if type(value) is not str:
        raise BroadDialoguePersistenceError(f"{label} 必须是 str")
    try:
        encoded = value.encode("ascii" if ascii_only else "utf-8")
    except UnicodeEncodeError as error:
        raise BroadDialoguePersistenceError(f"{label} 编码失败") from error
    return tuple(encoded)


def _decode_text(value: tuple[int, ...], *, label: str,
                 ascii_only: bool = False) -> str:
    if any(item > 255 for item in value):
        raise BroadDialoguePersistenceError(f"{label} 字节越界")
    try:
        return bytes(value).decode("ascii" if ascii_only else "utf-8")
    except UnicodeDecodeError as error:
        raise BroadDialoguePersistenceError(f"{label} 编码损坏") from error


def _pack_optional_text(result: list[int], value: str | None, *, label: str,
                        ascii_only: bool = False) -> None:
    if value is None:
        result.append(0)
        return
    result.append(1)
    _pack(result, _text(value, label=label, ascii_only=ascii_only), label=label)


def _read_optional_text(reader: IntegerStreamReader, *, label: str,
                        ascii_only: bool = False) -> str | None:
    try:
        present = reader.read(label=f"{label}.present")
    except IntegerCodecError as error:
        raise BroadDialoguePersistenceError(f"{label} 标记缺失") from error
    if present == 0:
        return None
    if present != 1:
        raise BroadDialoguePersistenceError(f"{label} 标记非法")
    return _decode_text(_read_key(reader, label=label), label=label,
                        ascii_only=ascii_only)


def _encode_citation(citation: DialogueCitation) -> tuple[int, ...]:
    """编码逐 evidence 来源记录，不把合并标题当成唯一 citation。"""
    if not isinstance(citation, DialogueCitation):
        raise TypeError("dialogue citation 类型错误")
    result: list[int] = []
    _pack(result, _text(citation.surface, label="citation surface"),
          label="citation surface")
    _pack_optional_text(result, citation.source_title,
                        label="citation source title")
    _pack_optional_text(result, citation.source_url,
                        label="citation source url")
    _pack_optional_text(result, citation.license_id,
                        label="citation license id", ascii_only=True)
    _pack_optional_text(result, citation.attribution,
                        label="citation attribution")
    if citation.source_ref is None:
        result.append(0)
    else:
        result.append(1)
        _pack(result, citation.source_ref, label="citation source ref")
    return tuple(result)


def _decode_citation(record: tuple[int, ...], *, version: int) -> DialogueCitation:
    reader = IntegerStreamReader(record)
    surface = _decode_text(
        _read_key(reader, label="citation surface", empty=False),
        label="citation surface")
    title = _read_optional_text(reader, label="citation source title")
    url = _read_optional_text(reader, label="citation source url")
    license_id = None
    attribution = None
    source_ref = None
    if version >= BROAD_DIALOGUE_PERSISTENCE_V3:
        license_id = _read_optional_text(
            reader, label="citation license id", ascii_only=True)
        attribution = _read_optional_text(reader, label="citation attribution")
        try:
            source_ref_present = reader.read(
                label="citation source ref.present")
        except IntegerCodecError as error:
            raise BroadDialoguePersistenceError(
                "citation source ref 标记缺失") from error
        if source_ref_present == 1:
            source_ref = _read_key(
                reader, label="citation source ref", empty=False)
        elif source_ref_present != 0:
            raise BroadDialoguePersistenceError("citation source ref 标记非法")
    try:
        reader.finish()
        return DialogueCitation(
            surface, title, url, license_id, attribution, source_ref)
    except (IntegerCodecError, TypeError, ValueError) as error:
        raise BroadDialoguePersistenceError("citation 无法恢复") from error


def _encode_turn(turn: DialogueTurn) -> tuple[int, ...]:
    if not isinstance(turn, DialogueTurn):
        raise TypeError("dialogue turn 类型错误")
    if type(turn.ordinal) is not int or turn.ordinal < 0:
        raise BroadDialoguePersistenceError("turn ordinal 非法")
    if type(turn.status) is not str or not turn.status:
        raise BroadDialoguePersistenceError("turn status 非法")
    if (not isinstance(turn.citations, tuple)
            or any(not isinstance(item, DialogueCitation)
                   for item in turn.citations)):
        raise BroadDialoguePersistenceError("turn citations 非法")
    result = [BROAD_DIALOGUE_PERSISTENCE_V3, turn.ordinal]
    _pack(result, _text(turn.status, label="turn status", ascii_only=True),
          label="turn status")
    _pack(result, _text(turn.question, label="turn question"),
          label="turn question")
    _pack_optional_text(result, turn.answer, label="turn answer")
    _pack_optional_text(result, turn.display_answer, label="turn display")
    _pack_optional_text(result, turn.source_title, label="turn source title")
    _pack_optional_text(result, turn.source_url, label="turn source url")
    _pack(result, turn.turn_key, label="turn key")
    _pack_optional_text(result, turn.retrieval_question,
                        label="turn retrieval question")
    result.append(len(turn.citations))
    for index, citation in enumerate(turn.citations):
        _pack(result, _encode_citation(citation),
              label=f"turn citation[{index}]")
    return tuple(result)


def _decode_turn(record: tuple[int, ...]) -> DialogueTurn:
    reader = IntegerStreamReader(record)
    try:
        version = reader.read_positive(label="turn codec version")
        ordinal = reader.read_nonnegative(label="turn ordinal")
    except (IntegerCodecError, ValueError) as error:
        raise BroadDialoguePersistenceError("turn header 损坏") from error
    if version not in (BROAD_DIALOGUE_PERSISTENCE_V1,
                       BROAD_DIALOGUE_PERSISTENCE_V2,
                       BROAD_DIALOGUE_PERSISTENCE_V3):
        raise BroadDialoguePersistenceError("turn codec version 未注册")
    status = _decode_text(_read_key(reader, label="turn status"),
                          label="turn status", ascii_only=True)
    question = _decode_text(_read_key(reader, label="turn question"),
                            label="turn question")
    if not question.strip():
        raise BroadDialoguePersistenceError("turn question 不能为空")
    answer = _read_optional_text(reader, label="turn answer")
    display = _read_optional_text(reader, label="turn display")
    source_title = _read_optional_text(reader, label="turn source title")
    source_url = _read_optional_text(reader, label="turn source url")
    turn_key = _read_key(reader, label="turn key", empty=False)
    retrieval = _read_optional_text(reader, label="turn retrieval question")
    citations = ()
    if version >= BROAD_DIALOGUE_PERSISTENCE_V2:
        try:
            citation_count = reader.read_nonnegative(label="turn citation count")
        except IntegerCodecError as error:
            raise BroadDialoguePersistenceError("turn citation count 损坏") from error
        citations = tuple(_decode_citation(
            _read_key(reader, label=f"turn citation[{index}]", empty=False),
            version=version)
            for index in range(citation_count))
    try:
        reader.finish()
        return DialogueTurn(
            ordinal, question, answer, display, status, source_title,
            source_url, turn_key, retrieval, citations,
        )
    except (TypeError, ValueError) as error:
        raise BroadDialoguePersistenceError("turn 无法恢复") from error


def _encode_state(state: BroadDialogueState) -> tuple[int, ...]:
    if not isinstance(state, BroadDialogueState):
        raise TypeError("broad dialogue state 类型错误")
    if (not isinstance(state.conversation_key, tuple)
            or not state.conversation_key
            or any(type(item) is not int or item < 0
                   for item in state.conversation_key)):
        raise BroadDialoguePersistenceError("conversation key 非法")
    if type(state.next_ordinal) is not int or state.next_ordinal < 0:
        raise BroadDialoguePersistenceError("next ordinal 非法")
    if len(state.turns) > BROAD_DIALOGUE_HOT_HISTORY_LIMIT:
        raise BroadDialoguePersistenceError("hot history 超过 8 轮上限")
    result = [BROAD_DIALOGUE_PERSISTENCE_V1]
    _pack(result, state.conversation_key, label="conversation key")
    result.append(state.next_ordinal)
    result.append(len(state.turns))
    previous = state.next_ordinal - len(state.turns)
    if tuple(turn.ordinal for turn in state.turns) != tuple(
            range(previous, state.next_ordinal)):
        raise BroadDialoguePersistenceError("hot history ordinal 不连续")
    for index, turn in enumerate(state.turns):
        if turn.ordinal < previous or turn.ordinal >= state.next_ordinal:
            raise BroadDialoguePersistenceError("hot history ordinal 不在状态范围")
        _pack(result, _encode_turn(turn), label=f"turn[{index}]")
    return tuple(result)


def _decode_state(record: tuple[int, ...]) -> BroadDialogueState:
    reader = IntegerStreamReader(record)
    try:
        version = reader.read_positive(label="state codec version")
    except (IntegerCodecError, ValueError) as error:
        raise BroadDialoguePersistenceError("state header 损坏") from error
    if version != BROAD_DIALOGUE_PERSISTENCE_V1:
        raise BroadDialoguePersistenceError("state codec version 未注册")
    key = _read_key(reader, label="conversation key", empty=False)
    try:
        next_ordinal = reader.read_nonnegative(label="next ordinal")
        count = reader.read_nonnegative(label="turn count")
    except (IntegerCodecError, ValueError) as error:
        raise BroadDialoguePersistenceError("state ordinal/count 损坏") from error
    if count > BROAD_DIALOGUE_HOT_HISTORY_LIMIT:
        raise BroadDialoguePersistenceError("state turn count 超过 hot history 上限")
    turns = tuple(_decode_turn(_read_key(reader, label=f"turn[{i}]", empty=False))
                  for i in range(count))
    try:
        reader.finish()
    except IntegerCodecError as error:
        raise BroadDialoguePersistenceError("state 含尾随整数") from error
    expected_first = next_ordinal - count
    if tuple(turn.ordinal for turn in turns) != tuple(range(expected_first, next_ordinal)):
        raise BroadDialoguePersistenceError("state turn ordinal 不连续")
    return BroadDialogueState(key, next_ordinal, turns)


def _checkpoint_record(checkpoint: "PersistentBroadDialogueCheckpoint") -> tuple[int, ...]:
    version = (BROAD_DIALOGUE_CHECKPOINT_V2
               if checkpoint.runtime_memory_state is not None
               else BROAD_DIALOGUE_CHECKPOINT_V1)
    result = [version, checkpoint.ordinal]
    _pack(result, _encode_state(checkpoint.state), label="checkpoint state")
    if checkpoint.runtime_memory_state is not None:
        try:
            runtime_record = encode_runtime_memory_state(
                checkpoint.runtime_memory_state)
        except PersistentDialogueRunError as error:
            raise BroadDialoguePersistenceError(
                "checkpoint Runtime Memory 编码失败") from error
        _pack(result, runtime_record, label="checkpoint runtime memory")
    _pack(result, checkpoint.previous_identity, label="checkpoint previous identity")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PersistentBroadDialogueCheckpoint:
    """一个 append-only 广域对话状态检查点。"""

    ordinal: int
    state: BroadDialogueState
    previous_identity: tuple[int, ...] = ()
    runtime_memory_state: RuntimeMemoryState | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise BroadDialoguePersistenceError("checkpoint ordinal 必须为正整数")
        if type(self.state.next_ordinal) is not int or self.state.next_ordinal < 0:
            raise BroadDialoguePersistenceError("dialogue next ordinal 非法")
        if (not isinstance(self.previous_identity, tuple)
                or any(type(item) is not int or item < 0
                       for item in self.previous_identity)):
            raise BroadDialoguePersistenceError("previous identity 非法")
        if (self.runtime_memory_state is not None
                and not isinstance(self.runtime_memory_state, RuntimeMemoryState)):
            raise TypeError("runtime memory state 类型错误")

    def canonical_record(self) -> tuple[int, ...]:
        return _checkpoint_record(self)

    def identity(self) -> tuple[int, ...]:
        return digest_bytes(encode_integer_tuple(self.canonical_record()))


@dataclass(frozen=True, slots=True)
class PersistentBroadDialogueRecovery:
    """恢复后的最新状态和一次建立的 hot turn 索引。"""

    checkpoint: PersistentBroadDialogueCheckpoint
    checkpoint_identity: tuple[int, ...]
    turn_index: tuple[tuple[int, int], ...]
    runtime_event_index: tuple[tuple[tuple[int, ...], int], ...] = ()
    # 一次恢复期间建立的冷轮次索引。索引是派生缓存，不参与 checkpoint
    # 身份；每个 ordinal 只保留最新可回读的完整轮次。
    cold_turns: tuple[DialogueTurn, ...] = ()
    cold_feature_index: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    cold_turn_ordinals: tuple[int, ...] = ()

    @property
    def indexed_turn_count(self) -> int:
        return len(self.turn_index)

    def query_turn(self, ordinal: int) -> tuple[DialogueTurn | None, int]:
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("turn ordinal 必须是非负整数")
        position = bisect_left(self.turn_index, (ordinal, -1))
        if position < len(self.turn_index) and self.turn_index[position][0] == ordinal:
            return self.checkpoint.state.turns[self.turn_index[position][1]], 1
        return None, 1

    def query_runtime_memory_item(
            self, memory_item_key: tuple[int, ...],
            ) -> tuple[RuntimeMemoryEvent | None, int]:
        """按一次建立的 Runtime item 索引读取事件。"""
        if (not isinstance(memory_item_key, tuple)
                or any(type(item) is not int or item < 0
                       for item in memory_item_key)):
            raise ValueError("memory item key 必须是非负整数 tuple")
        position = bisect_left(self.runtime_event_index,
                               (memory_item_key, -1))
        runtime_state = self.checkpoint.runtime_memory_state
        if (runtime_state is not None
                and position < len(self.runtime_event_index)
                and self.runtime_event_index[position][0] == memory_item_key):
            ordinal = self.runtime_event_index[position][1]
            return runtime_state.events[ordinal], 1
        return None, 1

    @staticmethod
    def _recall_features(value: str) -> frozenset[tuple[int, ...]]:
        """形成与具体语言无关的有界码点 n-gram 特征。"""
        if type(value) is not str or not value.strip():
            raise ValueError("memory recall query 必须是非空文本")
        normalized = " ".join(value.split())
        codepoints = tuple(ord(item) for item in normalized)
        features: set[tuple[int, ...]] = set()
        for width in (1, 2, 3):
            features.update(
                codepoints[index:index + width]
                for index in range(max(0, len(codepoints) - width + 1))
                if codepoints[index:index + width]
            )
        # 空白和标点只在组合中有意义；限制特征数避免长输入放大召回。
        return frozenset(item for item in features if any(
            codepoint not in {9, 10, 13, 32} for codepoint in item))

    def query_relevant_turns(
            self, question: str, *, limit: int = 4,
            minimum_similarity_permille: int = 500,
            ) -> tuple[DialogueTurn, ...]:
        """按整数特征相似度返回有界冷记忆轮次，不读取 checkpoint 文件。"""
        if type(limit) is not int or limit <= 0:
            raise ValueError("memory recall limit 必须为正整数")
        if (type(minimum_similarity_permille) is not int
                or not 0 <= minimum_similarity_permille <= 1000):
            raise ValueError("memory recall similarity 必须是 0..1000 整数")
        query = self._recall_features(question)
        if not query:
            return ()
        feature_keys = tuple(item[0] for item in self.cold_feature_index)
        candidates: set[int] = set()
        for feature in query:
            position = bisect_left(feature_keys, feature)
            if (position < len(feature_keys)
                    and feature_keys[position] == feature):
                candidates.update(self.cold_feature_index[position][1])
        ranked: list[tuple[int, int, int, DialogueTurn]] = []
        for ordinal in candidates:
            position = bisect_left(self.cold_turn_ordinals, ordinal)
            if (position >= len(self.cold_turn_ordinals)
                    or self.cold_turn_ordinals[position] != ordinal):
                continue
            turn = self.cold_turns[position]
            candidate = self._recall_features(turn.question)
            overlap = len(query.intersection(candidate))
            if overlap <= 0:
                continue
            # Dice 型整数分数，时间较近者只作为确定性次级排序。
            score = (2000 * overlap) // (len(query) + len(candidate))
            if score < minimum_similarity_permille:
                continue
            ranked.append((score, overlap, turn.ordinal, turn))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return tuple(item[3] for item in ranked[:limit])

    def with_turn(self, turn: DialogueTurn) -> "PersistentBroadDialogueRecovery":
        """把新完成的一轮加入 run-local 冷索引，不重读 checkpoint 链。"""
        if not isinstance(turn, DialogueTurn):
            raise TypeError("memory recall turn 类型错误")
        if self.cold_turn_ordinals and turn.ordinal < self.cold_turn_ordinals[-1]:
            # This path is only for a repaired/replayed checkpoint; normal
            # dialogue appends are monotonic and remain O(features of turn).
            by_ordinal = {item.ordinal: item for item in self.cold_turns}
            by_ordinal[turn.ordinal] = turn
            cold_turns = tuple(by_ordinal[key] for key in sorted(by_ordinal))
            feature_ordinals: dict[tuple[int, ...], set[int]] = {}
            for item in cold_turns:
                for feature in self._recall_features(item.question):
                    feature_ordinals.setdefault(feature, set()).add(item.ordinal)
            cold_feature_index = tuple(
                (feature, tuple(sorted(ordinals)))
                for feature, ordinals in sorted(feature_ordinals.items())
            )
        else:
            if self.cold_turns and turn.ordinal == self.cold_turns[-1].ordinal:
                if turn.turn_key == self.cold_turns[-1].turn_key:
                    return self
                raise ValueError("memory recall ordinal identity 漂移")
            cold_turns = (*self.cold_turns, turn)
            feature_map = {
                feature: ordinals
                for feature, ordinals in self.cold_feature_index
            }
            for feature in self._recall_features(turn.question):
                feature_map[feature] = (*feature_map.get(feature, ()), turn.ordinal)
            cold_feature_index = tuple(
                (feature, tuple(ordinals))
                for feature, ordinals in sorted(feature_map.items())
            )
        return PersistentBroadDialogueRecovery(
            self.checkpoint, self.checkpoint_identity, self.turn_index,
            self.runtime_event_index, cold_turns, cold_feature_index,
            tuple(item.ordinal for item in cold_turns),
        )


def _checkpoint_candidates(root: KRunRoot, directory: KRunRoot,
                           relative_dir: str | Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    for path in directory.path.iterdir():
        if not path.is_file() or path.suffix != ".int" or not path.name.startswith("checkpoint-"):
            continue
        token = path.stem.removeprefix("checkpoint-")
        if not token.isdigit() or len(token) != 20:
            raise BroadDialoguePersistenceError("广域 checkpoint 文件名非法")
        relative = Path(relative_dir) / path.name
        require_plain_file(root, relative, label="broad dialogue checkpoint")
        candidates.append((int(token), path))
    return sorted(candidates)


def write_broad_dialogue_checkpoint(
        root: KRunRoot,
        state: BroadDialogueState,
        *,
        runtime_memory_state: RuntimeMemoryState | None = None,
        relative_dir: str | Path = BROAD_DIALOGUE_DEFAULT_DIR,
        ) -> Path:
    """在 K run 内排他追加一轮广域 hot history。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("root 必须是 KRunRoot")
    directory = ensure_normal_relative_directory(root, relative_dir,
                                                  label="broad dialogue checkpoint directory")
    ordinal = 1
    existing = tuple(item for item in directory.path.iterdir()
                     if item.is_file() and item.suffix == ".int"
                     and item.name.startswith("checkpoint-"))
    previous = ()
    if existing:
        recovery = recover_broad_dialogue_checkpoint(
            root.path, relative_dir=relative_dir,
            require_k_drive=not root.test_transport,
        )
        ordinal = recovery.checkpoint.ordinal + 1
        if (recovery.checkpoint.runtime_memory_state is not None
                and runtime_memory_state is None):
            raise BroadDialoguePersistenceError(
                "既有会话已启用 Runtime Memory，后续 checkpoint 不得省略")
        previous = recovery.checkpoint_identity
    elif ordinal != 1:
        raise BroadDialoguePersistenceError("非首个 checkpoint 缺少前驱")
    checkpoint = PersistentBroadDialogueCheckpoint(
        ordinal, state, previous, runtime_memory_state)
    relative = Path(relative_dir) / f"checkpoint-{ordinal:020d}.int"
    try:
        return write_exclusive_bytes(root, relative,
                                    encode_integer_tuple(checkpoint.canonical_record()),
                                    label="broad dialogue checkpoint")
    except (TypeError, ValueError, OSError) as error:
        raise BroadDialoguePersistenceError("broad dialogue checkpoint 写入失败") from error


def recover_broad_dialogue_checkpoint(
        root_path: str | Path,
        *,
        relative_dir: str | Path = BROAD_DIALOGUE_DEFAULT_DIR,
        require_k_drive: bool = True,
        ) -> PersistentBroadDialogueRecovery:
    """关闭进程后恢复最新状态，并一次建立 bounded hot history 索引。"""
    root = open_existing_run_root(root_path, require_k_drive=require_k_drive,
                                  label="broad dialogue session root")
    directory = ensure_normal_relative_directory(root, relative_dir,
                                                  label="broad dialogue checkpoint directory")
    candidates = _checkpoint_candidates(root, directory, relative_dir)
    if not candidates:
        raise BroadDialoguePersistenceError("没有可恢复的 broad dialogue checkpoint")
    previous: tuple[int, ...] = ()
    latest: PersistentBroadDialogueCheckpoint | None = None
    latest_identity: tuple[int, ...] = ()
    turns_by_ordinal: dict[int, DialogueTurn] = {}
    expected = 1
    for ordinal, path in candidates:
        if ordinal != expected:
            raise BroadDialoguePersistenceError("checkpoint 序号不连续")
        relative = Path(relative_dir) / path.name
        try:
            with open_plain_binary(root, relative, label="broad dialogue checkpoint") as stream:
                payload = stream.read()
        except OSError as error:
            raise BroadDialoguePersistenceError("checkpoint 读取失败") from error
        try:
            record = decode_integer_tuple(payload)
        except (TypeError, ValueError, IntegerCodecError) as error:
            raise BroadDialoguePersistenceError("checkpoint 整数流损坏") from error
        reader = IntegerStreamReader(record)
        try:
            version = reader.read_positive(label="checkpoint version")
            item_ordinal = reader.read_positive(label="checkpoint ordinal")
        except (IntegerCodecError, ValueError) as error:
            raise BroadDialoguePersistenceError("checkpoint header 损坏") from error
        if version not in (BROAD_DIALOGUE_CHECKPOINT_V1,
                           BROAD_DIALOGUE_CHECKPOINT_V2) or item_ordinal != ordinal:
            raise BroadDialoguePersistenceError("checkpoint version/ordinal 漂移")
        state = _decode_state(_read_key(reader, label="checkpoint state", empty=False))
        runtime_memory_state = None
        if version == BROAD_DIALOGUE_CHECKPOINT_V2:
            runtime_record = _read_key(
                reader, label="checkpoint runtime memory", empty=False)
            try:
                runtime_memory_state = decode_runtime_memory_state(runtime_record)
            except PersistentDialogueRunError as error:
                raise BroadDialoguePersistenceError(
                    "checkpoint Runtime Memory 无法恢复") from error
        checkpoint_previous = _read_key(reader, label="checkpoint previous identity")
        try:
            reader.finish()
        except IntegerCodecError as error:
            raise BroadDialoguePersistenceError("checkpoint 含尾随整数") from error
        checkpoint = PersistentBroadDialogueCheckpoint(
            ordinal, state, checkpoint_previous, runtime_memory_state)
        if checkpoint_previous != previous:
            raise BroadDialoguePersistenceError("checkpoint 前驱 identity 漂移")
        identity = checkpoint.identity()
        previous = identity
        latest, latest_identity = checkpoint, identity
        for turn in state.turns:
            prior = turns_by_ordinal.get(turn.ordinal)
            if prior is None or turn.turn_key != prior.turn_key:
                turns_by_ordinal[turn.ordinal] = turn
        expected += 1
    if latest is None:
        raise BroadDialoguePersistenceError("checkpoint 恢复为空")
    index = tuple((turn.ordinal, index)
                  for index, turn in enumerate(latest.state.turns))
    runtime_index = ()
    if latest.runtime_memory_state is not None:
        by_item: dict[tuple[int, ...], tuple[int, RuntimeMemoryEvent]] = {}
        for event_index, event in enumerate(latest.runtime_memory_state.events):
            previous_item = by_item.get(event.memory_item_key)
            if previous_item is None or event.revision > previous_item[1].revision:
                by_item[event.memory_item_key] = (event_index, event)
            elif (event.revision == previous_item[1].revision
                  and event.event_key != previous_item[1].event_key):
                raise BroadDialoguePersistenceError(
                    "Runtime memory-item 索引存在竞争 revision")
        runtime_index = tuple(sorted(
            (key, index) for key, (index, _event) in by_item.items()))
    cold_turns = tuple(turns_by_ordinal[key] for key in sorted(turns_by_ordinal))
    feature_ordinals: dict[tuple[int, ...], set[int]] = {}
    for turn in cold_turns:
        for feature in PersistentBroadDialogueRecovery._recall_features(
                turn.question):
            feature_ordinals.setdefault(feature, set()).add(turn.ordinal)
    cold_feature_index = tuple(
        (feature, tuple(sorted(ordinals)))
        for feature, ordinals in sorted(feature_ordinals.items())
    )
    return PersistentBroadDialogueRecovery(
        latest, latest_identity, index, runtime_index,
        cold_turns, cold_feature_index,
        tuple(item.ordinal for item in cold_turns),
    )


__all__ = [
    "BROAD_DIALOGUE_CHECKPOINT_V1",
    "BROAD_DIALOGUE_CHECKPOINT_V2",
    "BROAD_DIALOGUE_DEFAULT_DIR",
    "BROAD_DIALOGUE_HOT_HISTORY_LIMIT",
    "BROAD_DIALOGUE_PERSISTENCE_V1",
    "BROAD_DIALOGUE_PERSISTENCE_V2",
    "BroadDialoguePersistenceError",
    "PersistentBroadDialogueCheckpoint",
    "PersistentBroadDialogueRecovery",
    "recover_broad_dialogue_checkpoint",
    "write_broad_dialogue_checkpoint",
]
