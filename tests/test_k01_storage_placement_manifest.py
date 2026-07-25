"""K-01 存储角色、注入式放置建议和 location manifest 对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.storage import build_storage_role_registry
from pure_integer_ai.storage.backend_capability import (
    BackendDeviceBudget,
    dict_backend_profile,
)
from pure_integer_ai.storage.location_manifest import (
    LocationManifest,
    LocationManifestEntry,
    LocationManifestLedger,
    ManifestDependency,
    ManifestIntegrityError,
    ManifestKeyRange,
    SegmentAvailability,
)
from pure_integer_ai.storage.memory_aggregate import (
    MEMORY_AGGREGATE_REBUILD_PROTOCOL_KEY,
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR,
    MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_STORAGE_DESCRIPTOR,
    MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
)
from pure_integer_ai.storage.placement import (
    PLACEMENT_MOVE,
    PLACEMENT_PREFETCH,
    PLACEMENT_RELEASE,
    PlacementAdvice,
    PlacementPlanner,
    PlacementPolicyError,
    PlacementRequest,
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.storage_role import (
    STORAGE_ACCESS_INDEXED_READ,
    STORAGE_ROLE_AUTHORITATIVE,
    STORAGE_ROLE_EPHEMERAL,
    StorageRoleDescriptor,
)


_TEMPERATURE = TemperatureProfile(
    (71, 1),
    (
        TemperatureTier((71, 1), 0),
        TemperatureTier((71, 2), 1),
        TemperatureTier((71, 3), 2),
    ),
)
_BUDGET = BackendDeviceBudget(4096, 512, 2, 1)
_PROFILE = dict_backend_profile(_BUDGET)


class _RecordingPolicy:
    """记录完整输入并返回一个需要 flush 的迁移建议。"""

    def __init__(self) -> None:
        """创建调用记录。"""
        self.observed: tuple[object, ...] | None = None

    def state_key(self) -> tuple[int, ...]:
        """返回测试策略的版本化稳定身份。"""
        return (991, 1)

    def advise(self, request, descriptor):
        """读取角色、身份、局部性、逻辑序、大小和设备 profile。"""
        self.observed = (
            request.object_key,
            descriptor.descriptor_key,
            request.dirty,
            request.locality_score,
            request.logical_seq,
            request.size_bytes,
            request.backend_profile.profile_key,
            request.backend_profile.device_budget,
            request.temperature_profile.profile_key,
        )
        return PlacementAdvice(
            PLACEMENT_MOVE,
            (71, 2),
            True,
            False,
            (801, 1),
            self.state_key(),
        )


class _BadReleasePolicy:
    """返回未经 flush 的 dirty release，供 planner 拒绝。"""

    def state_key(self) -> tuple[int, ...]:
        """返回坏策略的稳定身份。"""
        return (992, 1)

    def advise(self, request, descriptor):
        """构造违反 dirty release 不变量的建议。"""
        return PlacementAdvice(
            PLACEMENT_RELEASE, None, False, False, (802, 1), self.state_key())


class _DriftPolicy:
    """返回与自身 state key 不一致的建议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回策略声明身份。"""
        return (994, 1)

    def advise(self, request, descriptor):
        """故意把另一策略身份写入 advice。"""
        return PlacementAdvice(
            PLACEMENT_PREFETCH,
            (71, 2),
            False,
            True,
            (804, 1),
            (994, 2),
        )


class _FalsePrefetchPolicy:
    """返回动作与 prefetch 标志矛盾的建议。"""

    def state_key(self) -> tuple[int, ...]:
        """返回策略稳定身份。"""
        return (995, 1)

    def advise(self, request, descriptor):
        """故意遗漏 prefetch 标志。"""
        return PlacementAdvice(
            PLACEMENT_PREFETCH,
            (71, 2),
            False,
            False,
            (805, 1),
            self.state_key(),
        )


def _entry(
        *,
        descriptor_key: tuple[int, ...],
        segment_key: tuple[int, ...],
        lower: tuple[int, ...],
        upper: tuple[int, ...],
        tier: tuple[int, ...] = (71, 1),
        version: tuple[int, ...] = (1, 1),
        checksum: tuple[int, ...] = (9, 1),
        dependencies=(),
        epoch: int = 1,
        ) -> LocationManifestEntry:
    """构造测试用完整 location entry。"""
    return LocationManifestEntry(
        descriptor_key,
        segment_key,
        tier,
        ManifestKeyRange(lower, upper),
        version,
        checksum,
        tuple(dependencies),
        read_fence=epoch * 10,
        publish_epoch=epoch,
    )


def _manifest(epoch: int = 1, manifest_key: tuple[int, ...] | None = None):
    """构造包含 M-03 真源和 M-04 派生段的完整 manifest。"""
    dependency = ManifestDependency(
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        (1, 1),
        (9, 1),
    )
    return LocationManifest(
        manifest_key or (700, epoch),
        _TEMPERATURE.profile_key,
        epoch,
        None if epoch == 1 else epoch - 1,
        (
            _entry(
                descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
                segment_key=(900, 1),
                lower=(1, 1),
                upper=(1, 9),
                checksum=(9, 1),
                epoch=epoch,
            ),
            _entry(
                descriptor_key=MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
                segment_key=(900, 2),
                lower=(2, 1),
                upper=(2, 9),
                checksum=(9, 2),
                dependencies=(dependency,),
                epoch=epoch,
            ),
        ),
    )


def _registry():
    """返回含 M-03/M-04 角色的上下文注册表。"""
    return build_storage_role_registry()


def test_memory_event_is_authoritative_and_aggregate_is_rebuildable():
    """M-03 事件是真源，M-04 aggregate 只能删除后按依赖重建。"""
    registry = _registry()
    event = registry.get(MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY)
    aggregate = registry.get(MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY)
    assert event == MEMORY_EVENT_STORAGE_DESCRIPTOR
    assert aggregate == MEMORY_AGGREGATE_STORAGE_DESCRIPTOR
    assert event.role == STORAGE_ROLE_AUTHORITATIVE
    assert event.can_discard() is False
    assert aggregate.can_discard() is True
    assert aggregate.dependency_keys == (MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,)
    assert aggregate.rebuild_protocol_key == MEMORY_AGGREGATE_REBUILD_PROTOCOL_KEY


def test_role_registry_rejects_definition_drift_and_unknown_lookup():
    """角色同键漂移和未知角色不能被后续策略静默吸收。"""
    registry = _registry()
    with pytest.raises(ValueError):
        registry.register(StorageRoleDescriptor(
            MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
            STORAGE_ROLE_EPHEMERAL,
            (STORAGE_ACCESS_INDEXED_READ,),
        ))
    with pytest.raises(KeyError):
        registry.get((999, 1))


def test_placement_policy_receives_all_physical_inputs():
    """策略接收完整对象、角色、dirty、局部性、逻辑序、大小和预算。"""
    policy = _RecordingPolicy()
    planner = PlacementPlanner(policy)
    request = PlacementRequest(
        object_key=(12, 44, 8, 1, 3),
        descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        current_tier_key=(71, 1),
        dirty=True,
        locality_score=27,
        logical_seq=91,
        size_bytes=800,
        backend_profile=_PROFILE,
        temperature_profile=_TEMPERATURE,
    )
    advice = planner.advise(request, MEMORY_EVENT_STORAGE_DESCRIPTOR)
    assert advice.action == PLACEMENT_MOVE
    assert policy.observed == (
        (12, 44, 8, 1, 3),
        MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        True,
        27,
        91,
        800,
        _PROFILE.profile_key,
        _BUDGET,
        _TEMPERATURE.profile_key,
    )


def test_dirty_release_requires_explicit_flush_and_authoritative_release_is_rejected():
    """dirty 派生对象只能建议先 flush 再释放，权威事件始终不可释放。"""
    request = PlacementRequest(
        object_key=(1, 2, 3),
        descriptor_key=MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
        current_tier_key=(71, 1),
        dirty=True,
        locality_score=0,
        logical_seq=1,
        size_bytes=1,
        backend_profile=_PROFILE,
        temperature_profile=_TEMPERATURE,
    )
    with pytest.raises(PlacementPolicyError):
        PlacementPlanner(_BadReleasePolicy()).advise(
            request, MEMORY_AGGREGATE_STORAGE_DESCRIPTOR)

    class _FlushRelease:
        """返回先刷写再释放的派生对象建议。"""

        def state_key(self) -> tuple[int, ...]:
            """返回 flush-release 策略的稳定身份。"""
            return (993, 1)

        def advise(self, request, descriptor):
            """要求调用方在物理释放前完成 flush。"""
            return PlacementAdvice(
                PLACEMENT_RELEASE, None, True, False,
                (803, 1), self.state_key())

    assert PlacementPlanner(_FlushRelease()).advise(
        request, MEMORY_AGGREGATE_STORAGE_DESCRIPTOR).requires_flush is True

    authoritative_request = PlacementRequest(
        object_key=(1, 2, 4),
        descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        current_tier_key=(71, 1),
        dirty=False,
        locality_score=0,
        logical_seq=1,
        size_bytes=1,
        backend_profile=_PROFILE,
        temperature_profile=_TEMPERATURE,
    )
    with pytest.raises(PlacementPolicyError):
        PlacementPlanner(_FlushRelease()).advise(
            authoritative_request, MEMORY_EVENT_STORAGE_DESCRIPTOR)


@pytest.mark.parametrize("policy", (_DriftPolicy(), _FalsePrefetchPolicy()))
def test_policy_identity_and_prefetch_flags_fail_closed(policy):
    """策略身份漂移和伪 prefetch 不得进入物理执行层。"""
    request = PlacementRequest(
        object_key=(1, 5, 8),
        descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        current_tier_key=(71, 1),
        dirty=False,
        locality_score=1,
        logical_seq=2,
        size_bytes=3,
        backend_profile=_PROFILE,
        temperature_profile=_TEMPERATURE,
    )
    with pytest.raises(PlacementPolicyError):
        PlacementPlanner(policy).advise(
            request, MEMORY_EVENT_STORAGE_DESCRIPTOR)


def test_manifest_epoch_is_append_only_and_exact_replay_is_idempotent():
    """manifest 只能严格递增发布，同 epoch 完整重放幂等。"""
    ledger = LocationManifestLedger(_registry(), _TEMPERATURE)
    first = _manifest()
    assert ledger.append(first) == first
    assert ledger.append(first) == first
    second = _manifest(2)
    assert ledger.append(second) == second
    assert ledger.current() == second
    with pytest.raises(ManifestIntegrityError):
        ledger.append(_manifest(2, manifest_key=(701, 2)))
    with pytest.raises(ManifestIntegrityError):
        ledger.append(_manifest(4))


def test_manifest_rejects_overlap_dependency_drift_and_profile_drift():
    """同 store 重叠、依赖漂移和温层 profile 漂移必须 fail closed。"""
    registry = _registry()
    first = _manifest()
    with pytest.raises(ValueError):
        LocationManifest(
            (702, 1),
            _TEMPERATURE.profile_key,
            1,
            None,
            (
                first.entries[0],
                _entry(
                    descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
                    segment_key=(900, 3),
                    lower=(1, 8),
                    upper=(1, 12),
                    checksum=(9, 3),
                ),
            ),
        )

    bad_dependency = _entry(
        descriptor_key=MEMORY_AGGREGATE_STORAGE_DESCRIPTOR_KEY,
        segment_key=(900, 4),
        lower=(2, 10),
        upper=(2, 12),
        checksum=(9, 4),
        dependencies=(),
    )
    invalid = LocationManifest(
        (703, 1), _TEMPERATURE.profile_key, 1, None, (bad_dependency,))
    with pytest.raises(ValueError):
        LocationManifestLedger(registry, _TEMPERATURE).append(invalid)

    other_temperature = TemperatureProfile(
        (71, 9), (TemperatureTier((71, 9), 0), TemperatureTier((71, 10), 1)))
    with pytest.raises(ManifestIntegrityError):
        LocationManifestLedger(registry, other_temperature).append(first)


def test_manifest_availability_only_rebuilds_derived_segments():
    """权威段缺失或损坏失败，aggregate 派生段缺失只返回重建集合。"""
    ledger = LocationManifestLedger(_registry(), _TEMPERATURE)
    manifest = _manifest()
    report = ledger.verify_availability(
        manifest,
        (
            SegmentAvailability((900, 1), (71, 1), (1, 1), (9, 1)),
        ),
    )
    assert report.available_segment_keys == ((900, 1),)
    assert report.rebuildable_segment_keys == ((900, 2),)

    with pytest.raises(ManifestIntegrityError):
        ledger.verify_availability(
            manifest,
            (SegmentAvailability((900, 2), (71, 1), (1, 1), (9, 2)),),
        )
    with pytest.raises(ManifestIntegrityError):
        ledger.verify_availability(
            manifest,
            (
                SegmentAvailability((900, 1), (71, 1), (1, 1), (9, 1)),
                SegmentAvailability((900, 1), (71, 2), (1, 1), (9, 1)),
            ),
        )


def test_manifest_uses_full_keys_not_checksum_as_identity():
    """同 checksum 但 segment 完整键不同仍可并存，不能由裸 hash 合并。"""
    registry = _registry()
    entry_one = _entry(
        descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        segment_key=(901, 1), lower=(10, 1), upper=(10, 2), checksum=(5, 5))
    entry_two = _entry(
        descriptor_key=MEMORY_EVENT_STORAGE_DESCRIPTOR_KEY,
        segment_key=(901, 2), lower=(10, 3), upper=(10, 4), checksum=(5, 5))
    manifest = LocationManifest(
        (704, 1), _TEMPERATURE.profile_key, 1, None, (entry_one, entry_two))
    assert len(manifest.entries) == 2
    manifest.validate_roles(registry)


def test_ephemeral_missing_is_allowed_but_unknown_role_fails_closed():
    """query/worker 临时段可自然消失，开放未知角色不能默认按临时态处理。"""
    ephemeral_key = (880, 1)
    registry = _registry()
    registry.register(StorageRoleDescriptor(
        ephemeral_key,
        STORAGE_ROLE_EPHEMERAL,
        (STORAGE_ACCESS_INDEXED_READ,),
    ))
    ephemeral_manifest = LocationManifest(
        (705, 1),
        _TEMPERATURE.profile_key,
        1,
        None,
        (_entry(
            descriptor_key=ephemeral_key,
            segment_key=(902, 1),
            lower=(20, 1),
            upper=(20, 2),
            checksum=(6, 1),
        ),),
    )
    report = LocationManifestLedger(
        registry, _TEMPERATURE).verify_availability(ephemeral_manifest, ())
    assert report.available_segment_keys == ()
    assert report.rebuildable_segment_keys == ()

    unknown_key = (880, 2)
    registry.register(StorageRoleDescriptor(
        unknown_key,
        99,
        (STORAGE_ACCESS_INDEXED_READ,),
    ))
    unknown_manifest = LocationManifest(
        (706, 1),
        _TEMPERATURE.profile_key,
        1,
        None,
        (_entry(
            descriptor_key=unknown_key,
            segment_key=(902, 2),
            lower=(21, 1),
            upper=(21, 2),
            checksum=(6, 2),
        ),),
    )
    with pytest.raises(ManifestIntegrityError):
        LocationManifestLedger(
            registry, _TEMPERATURE).verify_availability(unknown_manifest, ())
