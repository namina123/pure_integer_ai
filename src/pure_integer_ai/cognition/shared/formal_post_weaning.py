"""PW-00A 正式断奶装载、发布清单与启动报告结构体。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.post_weaning import (
    PostWeaningFacilityProbe,
    PostWeaningResourceBudget,
    PostWeaningRouteProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


PW00A_START_PUBLISHED = 1
PW00A_START_RESUMED = 2


def _strict_key(
        value: tuple[int, ...],
        *,
        label: str,
        empty: bool = False,
        ) -> tuple[int, ...]:
    """核验稳定键只含严格整数，并按字段决定是否允许空键。"""
    if (not isinstance(value, tuple)
            or (not empty and not value)
            or any(type(item) is not int for item in value)):
        raise ValueError(f"{label} 必须是严格整数 tuple")
    if value:
        assert_int(*value, _where=label)
    return value


def _digest_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """要求文件或状态摘要使用精确 32 字节整数键。"""
    _strict_key(value, label=label)
    if len(value) != 32 or any(not 0 <= item <= 255 for item in value):
        raise ValueError(f"{label} 必须是 32 字节整数键")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给嵌套稳定键增加长度边界。"""
    return len(value), *value


def _runtime_owner(value: ObjectIdentity) -> ObjectIdentity:
    """要求正式 runtime owner 是一等最小指令。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise ValueError("formal runtime owner 必须是 MinimalInstruction")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FormalPostWeaningLoadRequest:
    """调用方对一次正式装载所提交的固定依赖和设施请求。"""

    run_id: int
    publish_epoch: int
    runtime_owner: ObjectIdentity
    authority_receipt_key: tuple[int, ...]
    inference_artifact_key: tuple[int, ...]
    routes: PostWeaningRouteProtocol
    probe: PostWeaningFacilityProbe
    budget: PostWeaningResourceBudget
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝零运行、漂移依赖和不完整设施合同。"""
        assert_int(self.run_id, self.publish_epoch, _where="PW00A load request")
        if (type(self.run_id) is not int or self.run_id <= 0
                or type(self.publish_epoch) is not int
                or self.publish_epoch <= 0):
            raise ValueError("PW00A run_id/publish_epoch 必须为正严格整数")
        _runtime_owner(self.runtime_owner)
        _digest_key(self.authority_receipt_key, label="authority receipt")
        _digest_key(self.inference_artifact_key, label="inference artifact")
        if not isinstance(self.routes, PostWeaningRouteProtocol):
            raise TypeError("PW00A routes 类型错误")
        if not isinstance(self.probe, PostWeaningFacilityProbe):
            raise TypeError("PW00A probe 类型错误")
        if not isinstance(self.budget, PostWeaningResourceBudget):
            raise TypeError("PW00A budget 类型错误")
        _strict_key(self.trace, label="PW00A request trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回正式装载请求的完整稳定整数身份。"""
        result = [1, self.run_id, self.publish_epoch]
        for value in (
                self.runtime_owner.stable_key(),
                self.authority_receipt_key,
                self.inference_artifact_key,
                self.routes.stable_key(),
                self.probe.stable_key(),
                self.budget.stable_key(),
                self.trace):
            result.extend(_packed(value))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FormalPostWeaningManifest:
    """正式发布绑定的 artifact、Core、owner 与运行设施清单。"""

    request: FormalPostWeaningLoadRequest
    inference_state_key: tuple[int, ...]
    core_state_key: tuple[int, ...]
    schema_state_key: tuple[int, ...]
    backend_state_key: tuple[int, ...]
    component_state_key: tuple[int, ...]
    owner_identity_key: tuple[int, ...]
    owner_watermark_key: tuple[int, ...]
    trace: tuple[int, ...]
    pw00a_started: int = 1

    def __post_init__(self) -> None:
        """要求全部承重状态已有非空稳定身份且启动位只能为一。"""
        if not isinstance(self.request, FormalPostWeaningLoadRequest):
            raise TypeError("formal manifest request 类型错误")
        _digest_key(self.inference_state_key, label="inference state")
        for label, value in (
                ("Core state", self.core_state_key),
                ("schema state", self.schema_state_key),
                ("backend state", self.backend_state_key),
                ("component state", self.component_state_key),
                ("owner identity", self.owner_identity_key),
                ("owner watermark", self.owner_watermark_key),
                ("formal manifest trace", self.trace)):
            _strict_key(value, label=label)
        if self.pw00a_started != 1:
            raise ValueError("formal manifest PW00A_STARTED 必须为 1")

    @property
    def routes(self) -> PostWeaningRouteProtocol:
        """返回正式入口路由。"""
        return self.request.routes

    @property
    def probe(self) -> PostWeaningFacilityProbe:
        """返回正式设施探针。"""
        return self.request.probe

    @property
    def budget(self) -> PostWeaningResourceBudget:
        """返回正式运行资源预算。"""
        return self.request.budget

    def stable_key(self) -> tuple[int, ...]:
        """返回正式发布清单的完整稳定整数身份。"""
        result = [1]
        for value in (
                self.request.stable_key(),
                self.inference_state_key,
                self.core_state_key,
                self.schema_state_key,
                self.backend_state_key,
                self.component_state_key,
                self.owner_identity_key,
                self.owner_watermark_key,
                self.trace):
            result.extend(_packed(value))
        result.append(self.pw00a_started)
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class FormalPostWeaningStartupReport:
    """一次首次发布或重启恢复的正式启动边界报告。"""

    run_id: int
    publish_epoch: int
    status: int
    manifest_key: tuple[int, ...]
    previous_manifest_key: tuple[int, ...]
    authority_receipt_key: tuple[int, ...]
    inference_artifact_key: tuple[int, ...]
    core_state_key: tuple[int, ...]
    owner_state_key: tuple[int, ...]
    protected_owner_space_ids: tuple[int, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验发布状态、摘要、保护集合和报告 trace。"""
        assert_int(
            self.run_id,
            self.publish_epoch,
            self.status,
            *self.protected_owner_space_ids,
            _where="PW00A startup report",
        )
        if (type(self.run_id) is not int or self.run_id <= 0
                or type(self.publish_epoch) is not int
                or self.publish_epoch <= 0
                or self.status not in {
                    PW00A_START_PUBLISHED, PW00A_START_RESUMED}):
            raise ValueError("PW00A startup report 状态非法")
        for label, value, empty in (
                ("manifest", self.manifest_key, False),
                ("previous manifest", self.previous_manifest_key, True),
                ("authority receipt", self.authority_receipt_key, False),
                ("inference artifact", self.inference_artifact_key, False),
                ("Core state", self.core_state_key, False),
                ("owner state", self.owner_state_key, False),
                ("startup trace", self.trace, False)):
            _strict_key(value, label=label, empty=empty)
        if (tuple(sorted(set(self.protected_owner_space_ids))
                ) != self.protected_owner_space_ids):
            raise ValueError("PW00A protected owner 集合非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回可进入 append-only receipt 的启动报告身份。"""
        result = [1, self.run_id, self.publish_epoch, self.status]
        for value in (
                self.manifest_key,
                self.previous_manifest_key,
                self.authority_receipt_key,
                self.inference_artifact_key,
                self.core_state_key,
                self.owner_state_key,
                self.protected_owner_space_ids,
                self.trace):
            result.extend(_packed(value))
        return tuple(result)


__all__ = [
    "FormalPostWeaningLoadRequest",
    "FormalPostWeaningManifest",
    "FormalPostWeaningStartupReport",
    "PW00A_START_PUBLISHED",
    "PW00A_START_RESUMED",
]
