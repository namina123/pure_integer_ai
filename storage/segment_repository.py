"""K-02 sealed object 的 append-only 介质协议和 backend 适配器。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT
from pure_integer_ai.storage.backend_capability import (
    BackendCapabilityRequirement,
    BackendNegotiationReport,
    CAPABILITY_ATOMIC_BATCH,
    CAPABILITY_ATOMIC_MANIFEST_PUBLISH,
    CAPABILITY_BULK_READ,
    CAPABILITY_BULK_WRITE,
    CAPABILITY_MODE_NATIVE,
    CAPABILITY_RANGE_SCAN,
    CAPABILITY_RECLAMATION,
    CAPABILITY_STABLE_ORDER_SCAN,
    negotiate_backend_capabilities,
)
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
    strict_integer_tuple,
)


SEGMENT_OBJECT_FORMAT_VERSION = 1

OBJECT_KIND_SEGMENT = 1
OBJECT_KIND_LOCATION_MANIFEST = 2
OBJECT_KIND_MIGRATION_COMMIT = 3
OBJECT_KIND_SEGMENT_RELEASE = 4

FAULT_OBJECT_AFTER_RESERVE = 1
FAULT_OBJECT_AFTER_PART = 2
FAULT_OBJECT_AFTER_SEAL = 3
FAULT_OBJECT_BEFORE_RECLAIM = 4
FAULT_OBJECT_AFTER_TOMBSTONE = 5
FAULT_OBJECT_AFTER_RECLAIM = 6

K02_FALLBACK_ATOMIC_BATCH_KEY = (20260723, 2, 1)
K02_FALLBACK_MANIFEST_PUBLISH_KEY = (20260723, 2, 2)
K02_FALLBACK_BULK_WRITE_KEY = (20260723, 2, 3)
K02_FALLBACK_RECLAMATION_KEY = (20260723, 2, 4)

SEGMENT_OBJECT_RESERVATION_TABLE = "segment_object_reservation"
SEGMENT_OBJECT_PART_TABLE = "segment_object_part"
SEGMENT_OBJECT_SEAL_TABLE = "segment_object_seal"
SEGMENT_OBJECT_TOMBSTONE_TABLE = "segment_object_tombstone"

_CHECKSUM_WORD_COUNT = 5
_WORD_BYTES = 7
_PART_WORD_COUNT = 16


class SegmentRepositoryError(RuntimeError):
    """介质对象缺失、碰撞、半写、校验或回收状态不一致。"""


@runtime_checkable
class SegmentRepositoryFaultInjector(Protocol):
    """在 append-only 对象写入和回收边界注入故障。"""

    def hit(self, point: int, context: dict[str, Any]) -> None:
        """观察指定故障点；需要模拟失败时直接抛异常。"""
        ...


def hit_repository_fault(
        injector: SegmentRepositoryFaultInjector | None,
        point: int,
        context: dict[str, Any],
        ) -> None:
    """调用可选故障注入器，并复制上下文避免外部修改内部状态。"""
    if injector is None:
        return
    if not isinstance(injector, SegmentRepositoryFaultInjector):
        raise TypeError("segment repository fault injector 协议错误")
    injector.hit(point, dict(context))


def register_segment_repository_tables(backend: StorageBackend) -> None:
    """注册通用对象预留、part、seal 和回收墓碑表。"""
    backend.register_table(
        SEGMENT_OBJECT_RESERVATION_TABLE,
        [
            ("object_id", TYPE_INT),
            ("object_kind", TYPE_INT),
            ("index_key", TYPE_INT),
        ],
        disc.DISC_APPEND_ONLY,
        [("object_id",), ("object_kind", "index_key")],
        recovery_key=("object_id",),
    )
    backend.register_table(
        SEGMENT_OBJECT_PART_TABLE,
        [
            ("object_id", TYPE_INT),
            ("part_index", TYPE_INT),
            ("part_size", TYPE_INT),
            ("word_count", TYPE_INT),
            *((f"part_{index:02d}", TYPE_INT)
              for index in range(_PART_WORD_COUNT)),
        ],
        disc.DISC_NONE,
        [("object_id",), ("object_id", "part_index")],
        recovery_key=("object_id", "part_index"),
    )
    backend.register_table(
        SEGMENT_OBJECT_SEAL_TABLE,
        [
            ("object_id", TYPE_INT),
            ("object_kind", TYPE_INT),
            ("index_key", TYPE_INT),
            ("word_count", TYPE_INT),
            ("part_count", TYPE_INT),
            ("size_bytes", TYPE_INT),
            *((f"checksum_{index:02d}", TYPE_INT)
              for index in range(_CHECKSUM_WORD_COUNT)),
        ],
        disc.DISC_NONE,
        [("object_id",), ("object_kind", "index_key")],
        recovery_key=("object_id",),
    )
    backend.register_table(
        SEGMENT_OBJECT_TOMBSTONE_TABLE,
        [("object_id", TYPE_INT), ("reclaim_seq", TYPE_INT)],
        disc.DISC_APPEND_ONLY,
        [("object_id",), ("reclaim_seq",)],
        recovery_key=("object_id",),
    )


@dataclass(frozen=True)
class StoredObjectDescriptor:
    """一个已 seal 且尚未回收的完整介质对象描述。"""

    object_id: int
    object_kind: int
    identity_key: tuple[int, ...]
    checksum_key: tuple[int, ...]
    size_bytes: int

    def __post_init__(self) -> None:
        """核验物理序号、对象类型、完整身份、校验和和尺寸。"""
        if type(self.object_id) is not int or self.object_id <= 0:
            raise ValueError("stored object_id 必须是正严格整数")
        if type(self.object_kind) is not int or self.object_kind <= 0:
            raise ValueError("stored object_kind 必须是正严格整数")
        strict_integer_tuple(
            self.identity_key, label="stored object identity_key")
        strict_integer_tuple(
            self.checksum_key, label="stored object checksum_key")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("stored object size_bytes 必须是正严格整数")


@runtime_checkable
class AppendOnlyObjectRepository(Protocol):
    """sealed segment、manifest 和 commit 共用的最小介质协议。"""

    def put(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            payload: bytes,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> StoredObjectDescriptor:
        """幂等写入一个 seal-last 对象，身份漂移必须失败。"""
        ...

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        """按完整身份读取并核验一个未回收对象。"""
        ...

    def list_kind(self, object_kind: int) -> tuple[StoredObjectDescriptor, ...]:
        """按物理发布序返回一种对象的完整描述。"""
        ...

    def reclaim(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> bool:
        """先发布墓碑，再按能力执行物理或逻辑回收。"""
        ...


def _validate_object_request(
        object_kind: int,
        identity_key: tuple[int, ...],
        payload: bytes | None = None,
        ) -> None:
    """核验对象类型、完整身份和可选非空载荷。"""
    if type(object_kind) is not int or object_kind <= 0:
        raise ValueError("object_kind 必须是正严格整数")
    strict_integer_tuple(identity_key, label="object identity_key")
    if payload is not None and (not isinstance(payload, bytes) or not payload):
        raise ValueError("object payload 必须是非空 bytes")


def _object_bytes(identity_key: tuple[int, ...], payload: bytes) -> bytes:
    """把完整身份和原始 payload 封装为规范整数流字节。"""
    values: list[int] = [SEGMENT_OBJECT_FORMAT_VERSION]
    pack_key(values, identity_key)
    values.extend((len(payload), *payload))
    return encode_integer_tuple(tuple(values))


def _parse_object_bytes(data: bytes) -> tuple[tuple[int, ...], bytes]:
    """恢复对象完整身份和原始 payload，并拒绝未知格式或尾字段。"""
    try:
        reader = IntegerStreamReader(decode_integer_tuple(data))
        version = reader.read_positive(label="segment object format")
        if version != SEGMENT_OBJECT_FORMAT_VERSION:
            raise SegmentRepositoryError("segment object format 不兼容")
        identity_key = reader.read_key(label="segment object identity_key")
        payload_size = reader.read_positive(label="segment object payload_size")
        payload_values = []
        for _ in range(payload_size):
            value = reader.read_nonnegative(label="segment object payload byte")
            if value > 255:
                raise SegmentRepositoryError("segment object payload byte 越界")
            payload_values.append(value)
        reader.finish()
        return identity_key, bytes(payload_values)
    except (IntegerCodecError, TypeError, ValueError) as exc:
        if isinstance(exc, SegmentRepositoryError):
            raise
        raise SegmentRepositoryError("segment object 编码损坏") from exc


def _default_index_key(
        object_kind: int, identity_key: tuple[int, ...],
        ) -> int:
    """生成仅用于候选缩小的 56-bit 索引值，不承担对象身份。"""
    values = (object_kind, len(identity_key), *identity_key)
    digest = hashlib.sha256(encode_integer_tuple(values)).digest()
    return int.from_bytes(digest[:_WORD_BYTES], "little") + 1


def _bytes_to_words(data: bytes) -> tuple[tuple[int, int], ...]:
    """把字节流拆成 SQLite 安全的 56-bit word 和实际字节数。"""
    if not isinstance(data, bytes) or not data:
        raise ValueError("word source 必须是非空 bytes")
    result = []
    for start in range(0, len(data), _WORD_BYTES):
        chunk = data[start:start + _WORD_BYTES]
        result.append((int.from_bytes(chunk, "little"), len(chunk)))
    return tuple(result)


def _word_parts(
        words: tuple[tuple[int, int], ...],
        ) -> tuple[tuple[tuple[int, int], ...], ...]:
    """把 56-bit word 按物理格式宽度分组，运行时容量仍由外部预算控制。"""
    if not isinstance(words, tuple) or not words:
        raise ValueError("segment object words 必须是非空 tuple")
    return tuple(
        words[start:start + _PART_WORD_COUNT]
        for start in range(0, len(words), _PART_WORD_COUNT)
    )


def _words_to_bytes(
        words: tuple[tuple[int, int], ...], *, expected_size: int,
        ) -> bytes:
    """从连续 56-bit word 恢复字节流并核验每个尾部尺寸。"""
    if type(expected_size) is not int or expected_size <= 0:
        raise SegmentRepositoryError("word expected_size 非法")
    data = bytearray()
    for index, (word, byte_size) in enumerate(words):
        if (type(word) is not int or not 0 <= word < (1 << 56)
                or type(byte_size) is not int
                or not 1 <= byte_size <= _WORD_BYTES):
            raise SegmentRepositoryError("segment object word 字段非法")
        if index + 1 < len(words) and byte_size != _WORD_BYTES:
            raise SegmentRepositoryError("segment object 非尾 word 不完整")
        raw = word.to_bytes(_WORD_BYTES, "little")
        if any(raw[byte_size:]):
            raise SegmentRepositoryError("segment object word 填充位非零")
        data.extend(raw[:byte_size])
    if len(data) != expected_size:
        raise SegmentRepositoryError("segment object word 总字节数不一致")
    return bytes(data)


class InMemoryObjectRepository:
    """不依赖 StorageBackend 的最小 append-only 介质适配器。"""

    def __init__(self) -> None:
        """创建空对象日志、发布序和回收墓碑集合。"""
        self._next_id = 1
        self._objects: dict[
            tuple[int, tuple[int, ...]], tuple[StoredObjectDescriptor, bytes]
        ] = {}
        self._reclaimed: set[tuple[int, tuple[int, ...]]] = set()

    def put(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            payload: bytes,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> StoredObjectDescriptor:
        """按 seal-last 语义写入内存对象，供最小 adapter 和故障验收。"""
        _validate_object_request(object_kind, identity_key, payload)
        key = (object_kind, identity_key)
        previous = self._objects.get(key)
        if previous is not None and key not in self._reclaimed:
            if previous[1] != payload:
                raise SegmentRepositoryError("同一对象完整身份载荷漂移")
            return previous[0]
        object_id = self._next_id
        self._next_id += 1
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_RESERVE, {
            "object_id": object_id,
            "object_kind": object_kind,
        })
        envelope = _object_bytes(identity_key, payload)
        for part_index, _ in enumerate(_word_parts(_bytes_to_words(envelope))):
            hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_PART, {
                "object_id": object_id,
                "part_index": part_index,
            })
        descriptor = StoredObjectDescriptor(
            object_id,
            object_kind,
            identity_key,
            tuple(hashlib.sha256(envelope).digest()),
            len(envelope),
        )
        self._objects[key] = (descriptor, payload)
        self._reclaimed.discard(key)
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_SEAL, {
            "object_id": object_id,
            "object_kind": object_kind,
        })
        return descriptor

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        """按完整身份读取未回收内存对象。"""
        _validate_object_request(object_kind, identity_key)
        key = (object_kind, identity_key)
        if key in self._reclaimed or key not in self._objects:
            raise KeyError(f"sealed object 不存在: {key}")
        return self._objects[key][1]

    def list_kind(self, object_kind: int) -> tuple[StoredObjectDescriptor, ...]:
        """按发布序列出一种未回收内存对象。"""
        _validate_object_request(object_kind, (1,))
        descriptors = [
            descriptor
            for key, (descriptor, _) in self._objects.items()
            if key[0] == object_kind and key not in self._reclaimed
        ]
        return tuple(sorted(descriptors, key=lambda item: item.object_id))

    def reclaim(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> bool:
        """发布内存墓碑，使对象立即退出可见集合。"""
        _validate_object_request(object_kind, identity_key)
        key = (object_kind, identity_key)
        if key not in self._objects or key in self._reclaimed:
            return False
        hit_repository_fault(fault_injector, FAULT_OBJECT_BEFORE_RECLAIM, {
            "object_id": self._objects[key][0].object_id,
        })
        self._reclaimed.add(key)
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_TOMBSTONE, {
            "object_id": self._objects[key][0].object_id,
        })
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_RECLAIM, {
            "object_id": self._objects[key][0].object_id,
        })
        return True


class BackendObjectRepository:
    """仅经 StorageBackend CRUD 保存 sealed object 的通用整数适配器。"""

    def __init__(
            self,
            backend: StorageBackend,
            *,
            index_key_fn: Callable[[int, tuple[int, ...]], int] | None = None,
            ) -> None:
        """注册表、协商能力，并绑定只作候选索引的可注入函数。"""
        if not isinstance(backend, StorageBackend):
            raise TypeError("segment repository backend 协议错误")
        self.backend = backend
        register_segment_repository_tables(backend)
        self.negotiation = negotiate_backend_capabilities(
            backend,
            (
                BackendCapabilityRequirement(
                    CAPABILITY_ATOMIC_BATCH,
                    K02_FALLBACK_ATOMIC_BATCH_KEY,
                ),
                BackendCapabilityRequirement(
                    CAPABILITY_ATOMIC_MANIFEST_PUBLISH,
                    K02_FALLBACK_MANIFEST_PUBLISH_KEY,
                ),
                BackendCapabilityRequirement(CAPABILITY_BULK_READ),
                BackendCapabilityRequirement(
                    CAPABILITY_BULK_WRITE,
                    K02_FALLBACK_BULK_WRITE_KEY,
                ),
                BackendCapabilityRequirement(CAPABILITY_RANGE_SCAN),
                BackendCapabilityRequirement(
                    CAPABILITY_RECLAMATION,
                    K02_FALLBACK_RECLAMATION_KEY,
                ),
                BackendCapabilityRequirement(CAPABILITY_STABLE_ORDER_SCAN),
            ),
        )
        self._physical_reclaim = self._negotiated_mode(
            CAPABILITY_RECLAMATION) == CAPABILITY_MODE_NATIVE
        self._index_key_fn = index_key_fn or _default_index_key
        if not callable(self._index_key_fn):
            raise TypeError("segment repository index_key_fn 必须可调用")
        self._writing = False
        self.cleanup_reclaimed()
        self.cleanup_unsealed()

    def capability_report(self) -> BackendNegotiationReport:
        """返回本适配器真实采用的 native/fallback 稳定协商结果。"""
        return self.negotiation

    def put(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            payload: bytes,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> StoredObjectDescriptor:
        """以预留、word 追加和 seal-last 顺序幂等发布一个对象。"""
        _validate_object_request(object_kind, identity_key, payload)
        previous = self._find(object_kind, identity_key)
        if previous is not None:
            if self.get(object_kind, identity_key) != payload:
                raise SegmentRepositoryError("同一对象完整身份载荷漂移")
            return previous
        if self._writing:
            raise SegmentRepositoryError("当前 adapter 不允许重入并发 writer")
        self._writing = True
        try:
            return self._put_new(
                object_kind,
                identity_key,
                payload,
                fault_injector=fault_injector,
            )
        finally:
            self._writing = False

    def get(self, object_kind: int, identity_key: tuple[int, ...]) -> bytes:
        """按完整身份读取唯一可见对象并核验全部 word、seal 和校验和。"""
        _validate_object_request(object_kind, identity_key)
        descriptor = self._find(object_kind, identity_key)
        if descriptor is None:
            raise KeyError(
                f"sealed object 不存在: {(object_kind, identity_key)}")
        _, payload = self._read_object_id(descriptor.object_id)
        return payload

    def list_kind(self, object_kind: int) -> tuple[StoredObjectDescriptor, ...]:
        """扫描一种 seal 对象，过滤墓碑并按物理发布序完整核验。"""
        _validate_object_request(object_kind, (1,))
        result = []
        identities: dict[tuple[int, ...], StoredObjectDescriptor] = {}
        for row in self.backend.select(
                SEGMENT_OBJECT_SEAL_TABLE,
                where={"object_kind": object_kind},
                order_by="object_id"):
            object_id = row["object_id"]
            if self._is_reclaimed(object_id):
                continue
            descriptor, _ = self._read_object_id(object_id)
            previous = identities.get(descriptor.identity_key)
            if previous is not None:
                raise SegmentRepositoryError(
                    "同一完整对象身份存在多个可见 seal")
            identities[descriptor.identity_key] = descriptor
            result.append(descriptor)
        return tuple(result)

    def reclaim(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            *,
            fault_injector: SegmentRepositoryFaultInjector | None = None,
            ) -> bool:
        """先耐久发布墓碑，再按协商结果执行物理删除或逻辑隐藏。"""
        _validate_object_request(object_kind, identity_key)
        descriptor = self._find(object_kind, identity_key)
        if descriptor is None:
            return False
        hit_repository_fault(fault_injector, FAULT_OBJECT_BEFORE_RECLAIM, {
            "object_id": descriptor.object_id,
            "object_kind": object_kind,
        })
        reclaim_seq = self._next_reclaim_seq()
        self.backend.insert(SEGMENT_OBJECT_TOMBSTONE_TABLE, {
            "object_id": descriptor.object_id,
            "reclaim_seq": reclaim_seq,
        })
        self.backend.commit()
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_TOMBSTONE, {
            "object_id": descriptor.object_id,
            "reclaim_seq": reclaim_seq,
        })
        if self._physical_reclaim:
            self._delete_object_rows(descriptor.object_id)
            self.backend.commit()
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_RECLAIM, {
            "object_id": descriptor.object_id,
            "physical": int(self._physical_reclaim),
        })
        return True

    def cleanup_reclaimed(self) -> int:
        """重启时完成已发布墓碑但尚未结束的原生物理回收。"""
        if not self._physical_reclaim:
            return 0
        rows = self.backend.select(
            SEGMENT_OBJECT_TOMBSTONE_TABLE,
            order_by="reclaim_seq",
        )
        cleaned = 0
        seen: set[int] = set()
        for row in rows:
            object_id = row["object_id"]
            if object_id in seen:
                raise SegmentRepositoryError("同一 object_id 重复发布回收墓碑")
            seen.add(object_id)
            has_payload = bool(self.backend.select(
                SEGMENT_OBJECT_SEAL_TABLE,
                where={"object_id": object_id},
            ) or self.backend.select(
                SEGMENT_OBJECT_PART_TABLE,
                where={"object_id": object_id},
            ))
            if has_payload:
                self._delete_object_rows(object_id)
                cleaned += 1
        if cleaned:
            self.backend.commit()
        return cleaned

    def cleanup_unsealed(self) -> int:
        """启动时给无 seal 的中断写入发布墓碑，并按能力清理孤儿 part。"""
        reclaimed_ids = {
            row["object_id"] for row in self.backend.select(
                SEGMENT_OBJECT_TOMBSTONE_TABLE)
        }
        sealed_ids = {
            row["object_id"] for row in self.backend.select(
                SEGMENT_OBJECT_SEAL_TABLE)
        }
        reservation_ids = [
            row["object_id"]
            for row in self.backend.select(SEGMENT_OBJECT_RESERVATION_TABLE)
        ]
        if len(set(reservation_ids)) != len(reservation_ids):
            raise SegmentRepositoryError("segment object reservation id 重复")
        orphan_ids = sorted(
            object_id for object_id in reservation_ids
            if object_id not in sealed_ids and object_id not in reclaimed_ids
        )
        for object_id in orphan_ids:
            self.backend.insert(SEGMENT_OBJECT_TOMBSTONE_TABLE, {
                "object_id": object_id,
                "reclaim_seq": self._next_reclaim_seq(),
            })
            if self._physical_reclaim:
                self.backend.delete(
                    SEGMENT_OBJECT_PART_TABLE, {"object_id": object_id})
        if orphan_ids:
            self.backend.commit()
        return len(orphan_ids)

    def _put_new(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            payload: bytes,
            *,
            fault_injector: SegmentRepositoryFaultInjector | None,
            ) -> StoredObjectDescriptor:
        """执行一个新对象的 append-only 物理发布过程。"""
        index_key = self._index_key(object_kind, identity_key)
        object_id = self._next_object_id()
        self.backend.insert(SEGMENT_OBJECT_RESERVATION_TABLE, {
            "object_id": object_id,
            "object_kind": object_kind,
            "index_key": index_key,
        })
        self.backend.commit()
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_RESERVE, {
            "object_id": object_id,
            "object_kind": object_kind,
        })
        envelope = _object_bytes(identity_key, payload)
        words = _bytes_to_words(envelope)
        parts = _word_parts(words)
        for part_index, part in enumerate(parts):
            row = {
                "object_id": object_id,
                "part_index": part_index,
                "part_size": sum(item[1] for item in part),
                "word_count": len(part),
            }
            for index in range(_PART_WORD_COUNT):
                row[f"part_{index:02d}"] = (
                    part[index][0] if index < len(part) else 0)
            self.backend.insert(SEGMENT_OBJECT_PART_TABLE, row)
            hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_PART, {
                "object_id": object_id,
                "part_index": part_index,
            })
        checksum_key = tuple(hashlib.sha256(envelope).digest())
        checksum_words = _bytes_to_words(bytes(checksum_key))
        if len(checksum_words) != _CHECKSUM_WORD_COUNT:
            raise AssertionError("SHA-256 checksum word 数量漂移")
        row = {
            "object_id": object_id,
            "object_kind": object_kind,
            "index_key": index_key,
            "word_count": len(words),
            "part_count": len(parts),
            "size_bytes": len(envelope),
        }
        for index, (word, _) in enumerate(checksum_words):
            row[f"checksum_{index:02d}"] = word
        self.backend.insert(SEGMENT_OBJECT_SEAL_TABLE, row)
        self.backend.commit()
        hit_repository_fault(fault_injector, FAULT_OBJECT_AFTER_SEAL, {
            "object_id": object_id,
            "object_kind": object_kind,
        })
        descriptor, restored = self._read_object_id(object_id)
        if descriptor.identity_key != identity_key or restored != payload:
            raise SegmentRepositoryError("sealed object 写后核验失败")
        return descriptor

    def _read_object_id(
            self, object_id: int,
            ) -> tuple[StoredObjectDescriptor, bytes]:
        """按物理序号完整核验预留、word、seal、checksum 和 identity。"""
        reservations = self.backend.select(
            SEGMENT_OBJECT_RESERVATION_TABLE,
            where={"object_id": object_id},
        )
        seals = self.backend.select(
            SEGMENT_OBJECT_SEAL_TABLE,
            where={"object_id": object_id},
        )
        if len(reservations) != 1 or len(seals) != 1:
            raise SegmentRepositoryError("sealed object 预留或 seal 不唯一")
        reservation = reservations[0]
        seal = seals[0]
        if (reservation["object_kind"] != seal["object_kind"]
                or reservation["index_key"] != seal["index_key"]):
            raise SegmentRepositoryError("sealed object 预留与 seal 漂移")
        part_rows = self.backend.select(
            SEGMENT_OBJECT_PART_TABLE,
            where={"object_id": object_id},
            order_by="part_index",
        )
        if len(part_rows) != seal["part_count"]:
            raise SegmentRepositoryError("sealed object part 数量不完整")
        words = []
        remaining_bytes = seal["size_bytes"]
        for part_index, row in enumerate(part_rows):
            if row["part_index"] != part_index:
                raise SegmentRepositoryError("sealed object part_index 不连续")
            word_count = row["word_count"]
            if (type(word_count) is not int or not 1 <= word_count <= _PART_WORD_COUNT
                    or (part_index + 1 < len(part_rows)
                        and word_count != _PART_WORD_COUNT)):
                raise SegmentRepositoryError("sealed object part word_count 非法")
            part_words = []
            for word_index in range(_PART_WORD_COUNT):
                word = row[f"part_{word_index:02d}"]
                if word_index >= word_count:
                    if word != 0:
                        raise SegmentRepositoryError(
                            "sealed object part 未用 word 非零")
                    continue
                byte_size = min(_WORD_BYTES, remaining_bytes)
                part_words.append((word, byte_size))
                remaining_bytes -= byte_size
            if sum(item[1] for item in part_words) != row["part_size"]:
                raise SegmentRepositoryError("sealed object part_size 不匹配")
            words.extend(part_words)
        if len(words) != seal["word_count"] or remaining_bytes != 0:
            raise SegmentRepositoryError("sealed object word 数量不完整")
        envelope = _words_to_bytes(
            tuple(words), expected_size=seal["size_bytes"])
        checksum_words = tuple(
            (seal[f"checksum_{index:02d}"],
             _WORD_BYTES if index + 1 < _CHECKSUM_WORD_COUNT else 4)
            for index in range(_CHECKSUM_WORD_COUNT)
        )
        checksum_key = tuple(_words_to_bytes(
            checksum_words, expected_size=32))
        if tuple(hashlib.sha256(envelope).digest()) != checksum_key:
            raise SegmentRepositoryError("sealed object checksum 不匹配")
        identity_key, payload = _parse_object_bytes(envelope)
        object_kind = seal["object_kind"]
        if self._index_key(object_kind, identity_key) != seal["index_key"]:
            raise SegmentRepositoryError("sealed object 候选索引漂移")
        return StoredObjectDescriptor(
            object_id,
            object_kind,
            identity_key,
            checksum_key,
            seal["size_bytes"],
        ), payload

    def _find(
            self,
            object_kind: int,
            identity_key: tuple[int, ...],
            ) -> StoredObjectDescriptor | None:
        """用可碰撞索引缩小候选，再按完整身份决定唯一对象。"""
        index_key = self._index_key(object_kind, identity_key)
        matches = []
        for row in self.backend.select(
                SEGMENT_OBJECT_SEAL_TABLE,
                where={"object_kind": object_kind, "index_key": index_key},
                order_by="object_id"):
            if self._is_reclaimed(row["object_id"]):
                continue
            descriptor, _ = self._read_object_id(row["object_id"])
            if descriptor.identity_key == identity_key:
                matches.append(descriptor)
        if len(matches) > 1:
            raise SegmentRepositoryError("同一完整对象身份存在多个可见 seal")
        return None if not matches else matches[0]

    def _index_key(
            self, object_kind: int, identity_key: tuple[int, ...],
            ) -> int:
        """执行可注入候选索引函数，并限制到 SQLite 安全正整数。"""
        value = self._index_key_fn(object_kind, identity_key)
        if type(value) is not int or not 0 < value < (1 << 63):
            raise ValueError("segment object index_key 必须是 63-bit 正严格整数")
        return value

    def _next_object_id(self) -> int:
        """从 append-only 预留日志确定下一个物理序号。"""
        rows = self.backend.select(
            SEGMENT_OBJECT_RESERVATION_TABLE,
            order_by="object_id",
            descending=True,
            limit=1,
        )
        return 1 if not rows else rows[0]["object_id"] + 1

    def _next_reclaim_seq(self) -> int:
        """从 append-only 墓碑日志确定下一个回收序号。"""
        rows = self.backend.select(
            SEGMENT_OBJECT_TOMBSTONE_TABLE,
            order_by="reclaim_seq",
            descending=True,
            limit=1,
        )
        return 1 if not rows else rows[0]["reclaim_seq"] + 1

    def _is_reclaimed(self, object_id: int) -> bool:
        """判断对象是否已有唯一可见回收墓碑。"""
        rows = self.backend.select(
            SEGMENT_OBJECT_TOMBSTONE_TABLE,
            where={"object_id": object_id},
        )
        if len(rows) > 1:
            raise SegmentRepositoryError("同一 object_id 重复发布回收墓碑")
        return bool(rows)

    def _delete_object_rows(self, object_id: int) -> None:
        """墓碑可见后删除载荷和 seal，保留预留序号防止 id 重用。"""
        self.backend.delete(
            SEGMENT_OBJECT_PART_TABLE, {"object_id": object_id})
        self.backend.delete(
            SEGMENT_OBJECT_SEAL_TABLE, {"object_id": object_id})

    def _negotiated_mode(self, capability: int) -> int:
        """读取一项已经完成的能力协商结果。"""
        for item in self.negotiation.capabilities:
            if item.capability == capability:
                return item.mode
        raise SegmentRepositoryError(f"缺少已协商 capability: {capability}")


__all__ = [
    "AppendOnlyObjectRepository",
    "BackendObjectRepository",
    "FAULT_OBJECT_AFTER_RECLAIM",
    "FAULT_OBJECT_AFTER_RESERVE",
    "FAULT_OBJECT_AFTER_SEAL",
    "FAULT_OBJECT_AFTER_TOMBSTONE",
    "FAULT_OBJECT_AFTER_PART",
    "FAULT_OBJECT_BEFORE_RECLAIM",
    "InMemoryObjectRepository",
    "K02_FALLBACK_ATOMIC_BATCH_KEY",
    "K02_FALLBACK_BULK_WRITE_KEY",
    "K02_FALLBACK_MANIFEST_PUBLISH_KEY",
    "K02_FALLBACK_RECLAMATION_KEY",
    "OBJECT_KIND_LOCATION_MANIFEST",
    "OBJECT_KIND_MIGRATION_COMMIT",
    "OBJECT_KIND_SEGMENT_RELEASE",
    "OBJECT_KIND_SEGMENT",
    "SEGMENT_OBJECT_RESERVATION_TABLE",
    "SEGMENT_OBJECT_SEAL_TABLE",
    "SEGMENT_OBJECT_TOMBSTONE_TABLE",
    "SEGMENT_OBJECT_PART_TABLE",
    "SegmentRepositoryError",
    "SegmentRepositoryFaultInjector",
    "StoredObjectDescriptor",
    "hit_repository_fault",
    "register_segment_repository_tables",
]
