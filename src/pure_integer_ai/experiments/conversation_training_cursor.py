"""公开对话训练 run 的纯整数恢复游标。

游标只描述一次已经结束的 formal_train run 的输入锁、阶段边界和计数；它不
保存表层文本、Python 对象或 SQLite 路径。文件位于 K 盘 run 内，可由其他
语言按同一整数流重建。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    KRunRoot,
    KRunBoundaryError,
    open_existing_run_root,
    open_plain_binary,
    require_plain_file,
    write_exclusive_bytes,
)


TRAINING_CURSOR_V1 = 1
TRAINING_CURSOR_FILE = "training_cursor.int"


class DialogueTrainingCursorError(ValueError):
    """训练游标不满足整数协议、身份锁或 K 盘边界。"""


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 for item in value)):
        raise DialogueTrainingCursorError(f"{label} 必须是非负整数 tuple")
    result.extend((len(value), *value))


def _read_key(reader: IntegerStreamReader, *, label: str) -> tuple[int, ...]:
    try:
        return reader.read_key(label=label, empty=True)
    except (IntegerCodecError, ValueError) as error:
        raise DialogueTrainingCursorError(f"{label} 不可读取") from error


def _read_count(reader: IntegerStreamReader, *, label: str) -> int:
    try:
        return reader.read_nonnegative(label=label)
    except (IntegerCodecError, ValueError) as error:
        raise DialogueTrainingCursorError(f"{label} 不可读取") from error


@dataclass(frozen=True, slots=True)
class DialogueTrainingCursor:
    """一个已封存训练 run 的可恢复整数游标。"""

    pack_sha256_u8: tuple[int, ...]
    run_id_u8: tuple[int, ...]
    requested_stages: tuple[int, ...]
    completed_stages: tuple[int, ...]
    training_item_count: int
    heldout_probe_count: int
    graph_size: int
    weaning_ready: bool
    previous_identity: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.pack_sha256_u8) != 32 or any(
                type(item) is not int or not 0 <= item <= 255
                for item in self.pack_sha256_u8):
            raise DialogueTrainingCursorError("pack SHA 必须是 32 字节整数摘要")
        if (not self.run_id_u8
                or any(type(item) is not int or not 0 <= item <= 255
                       for item in self.run_id_u8)):
            raise DialogueTrainingCursorError("run_id 必须是非空 UTF-8 字节整数")
        for label, value in (("requested_stages", self.requested_stages),
                             ("completed_stages", self.completed_stages)):
            if (not isinstance(value, tuple)
                    or any(type(item) is not int or item < 1 for item in value)
                    or tuple(sorted(set(value))) != value):
                raise DialogueTrainingCursorError(f"{label} 必须是升序阶段 tuple")
        if not set(self.completed_stages).issubset(self.requested_stages):
            raise DialogueTrainingCursorError("completed stages 越出 requested stages")
        for label, value in (("training_item_count", self.training_item_count),
                             ("heldout_probe_count", self.heldout_probe_count),
                             ("graph_size", self.graph_size)):
            if type(value) is not int or value < 0:
                raise DialogueTrainingCursorError(f"{label} 必须是非负严格整数")
        if type(self.weaning_ready) is not bool:
            raise DialogueTrainingCursorError("weaning_ready 必须是 bool")
        if (not isinstance(self.previous_identity, tuple)
                or any(type(item) is not int or item < 0
                       for item in self.previous_identity)):
            raise DialogueTrainingCursorError("previous identity 非法")

    def canonical_record(self) -> tuple[int, ...]:
        result = [TRAINING_CURSOR_V1]
        _pack(result, self.pack_sha256_u8, label="pack SHA")
        _pack(result, self.run_id_u8, label="run id")
        _pack(result, self.requested_stages, label="requested stages")
        _pack(result, self.completed_stages, label="completed stages")
        result.extend((self.training_item_count, self.heldout_probe_count,
                       self.graph_size, int(self.weaning_ready)))
        _pack(result, self.previous_identity, label="previous identity")
        return tuple(result)

    def identity(self) -> tuple[int, ...]:
        return digest_bytes(encode_integer_tuple(self.canonical_record()))


def write_training_cursor(root: KRunRoot, cursor: DialogueTrainingCursor) -> Path:
    """在指定 K run 内排他写入唯一 training cursor。"""
    if not isinstance(root, KRunRoot):
        raise TypeError("training cursor root 必须是 KRunRoot")
    if not isinstance(cursor, DialogueTrainingCursor):
        raise TypeError("training cursor 类型错误")
    relative = Path(TRAINING_CURSOR_FILE)
    try:
        path = write_exclusive_bytes(
            root,
            relative,
            encode_integer_tuple(cursor.canonical_record()),
            label="dialogue training cursor",
        )
    except (KRunBoundaryError, TypeError, ValueError, OSError) as error:
        raise DialogueTrainingCursorError("training cursor 写入失败") from error
    if path.name != TRAINING_CURSOR_FILE:
        raise DialogueTrainingCursorError("training cursor 路径漂移")
    return path


def recover_training_cursor(
        root_path: str | Path,
        *,
        require_k_drive: bool = True,
        ) -> DialogueTrainingCursor:
    """从已完成 run 恢复并核验唯一整数 cursor。"""
    root = open_existing_run_root(
        root_path, require_k_drive=require_k_drive,
        label="dialogue training cursor root",
    )
    relative = Path(TRAINING_CURSOR_FILE)
    require_plain_file(root, relative, label="dialogue training cursor")
    try:
        with open_plain_binary(
                root, relative, label="dialogue training cursor") as stream:
            record = decode_integer_tuple(stream.read())
    except (OSError, TypeError, ValueError, IntegerCodecError) as error:
        raise DialogueTrainingCursorError("training cursor 整数流损坏") from error
    reader = IntegerStreamReader(record)
    try:
        if reader.read_positive(label="training cursor version") != TRAINING_CURSOR_V1:
            raise DialogueTrainingCursorError("training cursor version 未注册")
        pack_sha = _read_key(reader, label="pack SHA")
        run_id_u8 = _read_key(reader, label="run id")
        requested = _read_key(reader, label="requested stages")
        completed = _read_key(reader, label="completed stages")
        training_count = reader.read_nonnegative(label="training item count")
        heldout_count = reader.read_nonnegative(label="heldout probe count")
        graph_size = reader.read_nonnegative(label="graph size")
        ready = reader.read(label="weaning ready")
        previous = _read_key(reader, label="previous identity")
        reader.finish()
        if ready not in (0, 1):
            raise DialogueTrainingCursorError("weaning_ready 非法")
        try:
            run_id = bytes(run_id_u8).decode("utf-8")
        except UnicodeDecodeError as error:
            raise DialogueTrainingCursorError("run_id UTF-8 不可恢复") from error
        if not run_id:
            raise DialogueTrainingCursorError("run_id 不能为空")
        return DialogueTrainingCursor(
            pack_sha, run_id_u8, requested, completed,
            training_count, heldout_count, graph_size, bool(ready), previous,
        )
    except (IntegerCodecError, ValueError, TypeError) as error:
        if isinstance(error, DialogueTrainingCursorError):
            raise
        raise DialogueTrainingCursorError("training cursor 字段不可恢复") from error


__all__ = [
    "DialogueTrainingCursor",
    "DialogueTrainingCursorError",
    "TRAINING_CURSOR_FILE",
    "TRAINING_CURSOR_V1",
    "recover_training_cursor",
    "write_training_cursor",
]
