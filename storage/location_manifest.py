"""逻辑对象到物理 segment 的版本化 location manifest 协议。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
)
from pure_integer_ai.storage.placement import TemperatureProfile
from pure_integer_ai.storage.segment_dependency import (
    SegmentDependency as ManifestDependency,
    canonical_dependencies,
)


LOCATION_MANIFEST_FORMAT_VERSION = 1
from pure_integer_ai.storage.storage_role import (
    STORAGE_ROLE_AUTHORITATIVE,
    STORAGE_ROLE_EPHEMERAL,
    STORAGE_ROLE_REBUILDABLE,
    StorageRoleRegistry,
)


def _key(value: tuple[int, ...], *, label: str, empty: bool = False) -> tuple[int, ...]:
    """核验 manifest 中的完整整数键。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是非空整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


def _nonnegative(value: int, *, label: str) -> int:
    """核验 manifest 计数和序号为非负严格整数。"""
    assert_int(value, _where=label)
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} 必须是非负严格整数")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长度整数键按长度分帧写入稳定键。"""
    result.extend((len(value), *value))


@dataclass(frozen=True, order=True)
class ManifestKeyRange:
    """一个 segment 覆盖的完整稳定键闭区间。"""

    lower_key: tuple[int, ...]
    upper_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验范围端点完整且按字典序不逆行。"""
        _key(self.lower_key, label="manifest lower_key")
        _key(self.upper_key, label="manifest upper_key")
        if self.lower_key > self.upper_key:
            raise ValueError("manifest key range 不能反向")

    def overlaps(self, other: "ManifestKeyRange") -> bool:
        """判断两个闭区间是否重叠。"""
        return not (
            self.upper_key < other.lower_key
            or other.upper_key < self.lower_key
        )


@dataclass(frozen=True, order=True)
class LocationManifestEntry:
    """一个发布 epoch 内唯一可见的 segment 位置记录。"""

    descriptor_key: tuple[int, ...]
    segment_key: tuple[int, ...]
    tier_key: tuple[int, ...]
    key_range: ManifestKeyRange
    version_key: tuple[int, ...]
    checksum_key: tuple[int, ...]
    dependencies: tuple[ManifestDependency, ...]
    read_fence: int
    publish_epoch: int

    def __post_init__(self) -> None:
        """核验 segment 身份、版本、依赖、read fence 和发布 epoch。"""
        _key(self.descriptor_key, label="manifest entry descriptor_key")
        _key(self.segment_key, label="manifest entry segment_key")
        _key(self.tier_key, label="manifest entry tier_key")
        if not isinstance(self.key_range, ManifestKeyRange):
            raise TypeError("manifest entry key_range 类型错误")
        _key(self.version_key, label="manifest entry version_key")
        _key(self.checksum_key, label="manifest entry checksum_key")
        if not isinstance(self.dependencies, tuple) or any(
                not isinstance(item, ManifestDependency)
                for item in self.dependencies):
            raise TypeError("manifest entry dependencies 类型错误")
        dependencies = canonical_dependencies(self.dependencies)
        object.__setattr__(self, "dependencies", dependencies)
        _nonnegative(self.read_fence, label="manifest entry read_fence")
        assert_int(self.publish_epoch, _where="manifest entry publish_epoch")
        if type(self.publish_epoch) is not int or self.publish_epoch <= 0:
            raise ValueError("manifest entry publish_epoch 必须是正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回位置记录的完整稳定整数键。"""
        result: list[int] = []
        for value in (
                self.descriptor_key,
                self.segment_key,
                self.tier_key,
                self.key_range.lower_key,
                self.key_range.upper_key,
                self.version_key,
                self.checksum_key):
            _pack(result, value)
        result.extend((len(self.dependencies),))
        for dependency in self.dependencies:
            key = dependency.stable_key()
            result.extend((len(key), *key))
        result.extend((self.read_fence, self.publish_epoch))
        return tuple(result)


@dataclass(frozen=True, order=True)
class LocationManifest:
    """一个已发布 epoch 的完整 canonical location manifest。"""

    manifest_key: tuple[int, ...]
    profile_key: tuple[int, ...]
    publish_epoch: int
    previous_epoch: int | None
    entries: tuple[LocationManifestEntry, ...]

    def __post_init__(self) -> None:
        """规范化 entry 顺序并拒绝同 store 重叠、epoch 漂移和重复 segment。"""
        _key(self.manifest_key, label="location manifest_key")
        _key(self.profile_key, label="location profile_key")
        assert_int(self.publish_epoch, _where="location publish_epoch")
        if type(self.publish_epoch) is not int or self.publish_epoch <= 0:
            raise ValueError("location publish_epoch 必须是正严格整数")
        if self.previous_epoch is not None:
            assert_int(self.previous_epoch, _where="location previous_epoch")
            if type(self.previous_epoch) is not int or self.previous_epoch <= 0:
                raise ValueError("location previous_epoch 必须是正严格整数")
            if self.previous_epoch >= self.publish_epoch:
                raise ValueError("previous_epoch 必须早于 publish_epoch")
        if not isinstance(self.entries, tuple) or any(
                not isinstance(item, LocationManifestEntry)
                for item in self.entries):
            raise TypeError("location manifest entries 类型错误")
        if any(item.publish_epoch != self.publish_epoch for item in self.entries):
            raise ValueError("manifest entry publish_epoch 漂移")
        if len({item.segment_key for item in self.entries}) != len(self.entries):
            raise ValueError("一个 manifest 不得重复发布同一 segment")
        entries = tuple(sorted(self.entries, key=lambda item: (
            item.descriptor_key,
            item.key_range.lower_key,
            item.key_range.upper_key,
            item.segment_key,
        )))
        previous_descriptor: tuple[int, ...] | None = None
        previous_upper: tuple[int, ...] | None = None
        for entry in entries:
            if entry.descriptor_key != previous_descriptor:
                previous_descriptor = entry.descriptor_key
                previous_upper = entry.key_range.upper_key
                continue
            if (previous_upper is not None
                    and entry.key_range.lower_key <= previous_upper):
                raise ValueError("同一 descriptor 的 canonical key range 重叠")
            previous_upper = entry.key_range.upper_key
        object.__setattr__(self, "entries", entries)

    def stable_key(self) -> tuple[int, ...]:
        """返回 manifest 的完整稳定整数键。"""
        result: list[int] = []
        _pack(result, self.manifest_key)
        _pack(result, self.profile_key)
        result.extend((self.publish_epoch, 0 if self.previous_epoch is None
                       else self.previous_epoch, len(self.entries)))
        for entry in self.entries:
            key = entry.stable_key()
            result.extend((len(key), *key))
        return tuple(result)

    def integer_stream(self) -> tuple[int, ...]:
        """返回可逆、版本化且完整保存 entry 与依赖的整数流。"""
        result: list[int] = [LOCATION_MANIFEST_FORMAT_VERSION]
        pack_key(result, self.manifest_key)
        pack_key(result, self.profile_key)
        result.extend((
            self.publish_epoch,
            0 if self.previous_epoch is None else self.previous_epoch,
            len(self.entries),
        ))
        for entry in self.entries:
            for value in (
                    entry.descriptor_key,
                    entry.segment_key,
                    entry.tier_key,
                    entry.key_range.lower_key,
                    entry.key_range.upper_key,
                    entry.version_key,
                    entry.checksum_key):
                pack_key(result, value)
            result.append(len(entry.dependencies))
            for dependency in entry.dependencies:
                pack_key(result, dependency.descriptor_key)
                pack_key(result, dependency.version_key)
                pack_key(result, dependency.checksum_key)
            result.append(entry.read_fence)
        return tuple(result)

    def to_bytes(self) -> bytes:
        """把完整 location manifest 编为确定性规范字节。"""
        return encode_integer_tuple(self.integer_stream())

    @classmethod
    def from_bytes(cls, data: bytes) -> "LocationManifest":
        """从规范字节恢复 manifest，并重新执行全部 K-01 不变量。"""
        try:
            reader = IntegerStreamReader(decode_integer_tuple(data))
            version = reader.read_positive(label="location manifest format")
            if version != LOCATION_MANIFEST_FORMAT_VERSION:
                raise ManifestIntegrityError("location manifest format 不兼容")
            manifest_key = reader.read_key(label="location manifest_key")
            profile_key = reader.read_key(label="location profile_key")
            publish_epoch = reader.read_positive(label="location publish_epoch")
            previous_raw = reader.read_nonnegative(
                label="location previous_epoch")
            previous_epoch = None if previous_raw == 0 else previous_raw
            entry_count = reader.read_nonnegative(label="location entry_count")
            entries = []
            for _ in range(entry_count):
                descriptor_key = reader.read_key(
                    label="location entry descriptor_key")
                segment_key = reader.read_key(
                    label="location entry segment_key")
                tier_key = reader.read_key(label="location entry tier_key")
                lower_key = reader.read_key(label="location entry lower_key")
                upper_key = reader.read_key(label="location entry upper_key")
                version_key = reader.read_key(
                    label="location entry version_key")
                checksum_key = reader.read_key(
                    label="location entry checksum_key")
                dependency_count = reader.read_nonnegative(
                    label="location entry dependency_count")
                dependencies = []
                for _ in range(dependency_count):
                    dependencies.append(ManifestDependency(
                        reader.read_key(
                            label="location dependency descriptor_key"),
                        reader.read_key(
                            label="location dependency version_key"),
                        reader.read_key(
                            label="location dependency checksum_key"),
                    ))
                read_fence = reader.read_nonnegative(
                    label="location entry read_fence")
                entries.append(LocationManifestEntry(
                    descriptor_key,
                    segment_key,
                    tier_key,
                    ManifestKeyRange(lower_key, upper_key),
                    version_key,
                    checksum_key,
                    tuple(dependencies),
                    read_fence,
                    publish_epoch,
                ))
            reader.finish()
            return cls(
                manifest_key,
                profile_key,
                publish_epoch,
                previous_epoch,
                tuple(entries),
            )
        except (IntegerCodecError, TypeError, ValueError) as exc:
            if isinstance(exc, ManifestIntegrityError):
                raise
            raise ManifestIntegrityError("location manifest 编码损坏") from exc

    def validate_roles(self, registry: StorageRoleRegistry) -> None:
        """核验 entry、依赖和角色声明的重建依赖完全一致。"""
        if not isinstance(registry, StorageRoleRegistry):
            raise TypeError("storage role registry 类型错误")
        for entry in self.entries:
            descriptor = registry.get(entry.descriptor_key)
            for dependency in entry.dependencies:
                registry.get(dependency.descriptor_key)
            actual_dependencies = tuple(
                item.descriptor_key for item in entry.dependencies)
            if actual_dependencies != descriptor.dependency_keys:
                raise ValueError("manifest entry 依赖与 storage role 声明不一致")


@dataclass(frozen=True, order=True)
class SegmentAvailability:
    """物理探针提供的 segment 完整版本、校验和位置身份。"""

    segment_key: tuple[int, ...]
    tier_key: tuple[int, ...]
    version_key: tuple[int, ...]
    checksum_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验可用 segment 的完整物理身份。"""
        _key(self.segment_key, label="available segment_key")
        _key(self.tier_key, label="available tier_key")
        _key(self.version_key, label="available version_key")
        _key(self.checksum_key, label="available checksum_key")


@dataclass(frozen=True, order=True)
class ManifestAvailabilityReport:
    """manifest 可用性核验结果和需要重建的派生 segment。"""

    available_segment_keys: tuple[tuple[int, ...], ...]
    rebuildable_segment_keys: tuple[tuple[int, ...], ...]


class ManifestIntegrityError(RuntimeError):
    """location manifest、segment 版本或权威物理副本不一致。"""


class LocationManifestLedger:
    """上下文内 append-only manifest epoch ledger，暂不执行物理迁移。"""

    def __init__(
            self,
            registry: StorageRoleRegistry,
            temperature_profile: TemperatureProfile,
            ) -> None:
        """绑定角色和温层 profile，并创建空的发布历史。"""
        if not isinstance(registry, StorageRoleRegistry):
            raise TypeError("manifest ledger registry 类型错误")
        if not isinstance(temperature_profile, TemperatureProfile):
            raise TypeError("manifest ledger temperature_profile 类型错误")
        self.registry = registry
        self.temperature_profile = temperature_profile
        self._manifests: dict[int, LocationManifest] = {}

    def append(self, manifest: LocationManifest) -> LocationManifest:
        """按严格 epoch 追加 manifest；精确重放幂等，漂移重放拒绝。"""
        if not isinstance(manifest, LocationManifest):
            raise TypeError("manifest 类型错误")
        self._validate_manifest(manifest)
        previous = self._manifests.get(manifest.publish_epoch)
        if previous is not None:
            if previous != manifest:
                raise ManifestIntegrityError("同一 publish epoch 内容漂移")
            return previous
        for existing in self._manifests.values():
            if existing.manifest_key == manifest.manifest_key:
                raise ManifestIntegrityError("manifest_key 不得跨 epoch 重用")
        current_epoch = max(self._manifests, default=0)
        if manifest.publish_epoch != current_epoch + 1:
            raise ManifestIntegrityError("manifest epoch 必须严格递增")
        expected_previous = None if current_epoch == 0 else current_epoch
        if manifest.previous_epoch != expected_previous:
            raise ManifestIntegrityError("manifest previous_epoch 不匹配")
        self._manifests[manifest.publish_epoch] = manifest
        return manifest

    def current(self) -> LocationManifest | None:
        """返回当前完整发布 epoch；没有发布时返回 None。"""
        if not self._manifests:
            return None
        return self._manifests[max(self._manifests)]

    def get(self, publish_epoch: int) -> LocationManifest:
        """按发布 epoch 读取历史 manifest，未知 epoch 失败。"""
        assert_int(publish_epoch, _where="manifest ledger epoch")
        try:
            return self._manifests[publish_epoch]
        except KeyError as exc:
            raise KeyError(f"未发布 manifest epoch: {publish_epoch}") from exc

    def verify_availability(
            self,
            manifest: LocationManifest,
            available: tuple[SegmentAvailability, ...],
            ) -> ManifestAvailabilityReport:
        """核验物理 segment；权威缺失失败，派生缺失只返回重建建议。"""
        if not isinstance(manifest, LocationManifest):
            raise TypeError("availability manifest 类型错误")
        self._validate_manifest(manifest)
        if not isinstance(available, tuple) or any(
                not isinstance(item, SegmentAvailability) for item in available):
            raise TypeError("available 必须是 SegmentAvailability tuple")
        available_map: dict[tuple[int, ...], SegmentAvailability] = {}
        for item in available:
            previous = available_map.get(item.segment_key)
            if previous is not None and previous != item:
                raise ManifestIntegrityError("同一 segment_key 的物理身份冲突")
            available_map[item.segment_key] = item
        present: list[tuple[int, ...]] = []
        rebuild: list[tuple[int, ...]] = []
        for entry in manifest.entries:
            actual = available_map.get(entry.segment_key)
            matches = actual is not None and (
                actual.tier_key == entry.tier_key
                and actual.version_key == entry.version_key
                and actual.checksum_key == entry.checksum_key
            )
            if matches:
                present.append(entry.segment_key)
                continue
            descriptor = self.registry.get(entry.descriptor_key)
            if descriptor.role == STORAGE_ROLE_AUTHORITATIVE:
                raise ManifestIntegrityError(
                    f"权威 segment 缺失或校验不匹配: {entry.segment_key}")
            if descriptor.role == STORAGE_ROLE_REBUILDABLE:
                rebuild.append(entry.segment_key)
                continue
            if descriptor.role == STORAGE_ROLE_EPHEMERAL and actual is None:
                continue
            if descriptor.role == STORAGE_ROLE_EPHEMERAL:
                raise ManifestIntegrityError(
                    f"临时 segment 物理身份冲突: {entry.segment_key}")
            raise ManifestIntegrityError(
                f"未知 storage role 无法裁决 segment: {entry.segment_key}")
        return ManifestAvailabilityReport(tuple(present), tuple(rebuild))

    def _validate_manifest(self, manifest: LocationManifest) -> None:
        """统一核验角色、温层 profile 和每个 entry 的物理温层。"""
        manifest.validate_roles(self.registry)
        if manifest.profile_key != self.temperature_profile.profile_key:
            raise ManifestIntegrityError("manifest temperature profile 漂移")
        for entry in manifest.entries:
            if not self.temperature_profile.has(entry.tier_key):
                raise ManifestIntegrityError("manifest entry 使用未注册温层")


__all__ = [
    "LOCATION_MANIFEST_FORMAT_VERSION",
    "LocationManifest",
    "LocationManifestEntry",
    "LocationManifestLedger",
    "ManifestAvailabilityReport",
    "ManifestDependency",
    "ManifestIntegrityError",
    "ManifestKeyRange",
    "SegmentAvailability",
]
