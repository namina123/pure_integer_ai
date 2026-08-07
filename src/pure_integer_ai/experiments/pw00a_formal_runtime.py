"""PW-00A authority 核验、双阶段发布、Core 保护与正式入口。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from pure_integer_ai.cognition.shared.formal_post_weaning import (
    FormalPostWeaningLoadRequest,
    FormalPostWeaningManifest,
    FormalPostWeaningStartupReport,
    PW00A_START_PUBLISHED,
    PW00A_START_RESUMED,
)
from pure_integer_ai.cognition.shared.types import WEANING_POST, WEANING_PRE
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w09_inference import (
    W09CandidateInferenceAdapter,
    W09InferenceState,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
    PostWeaningOperationRuntime,
    PostWeaningStartupError,
    _schema_state_key,
    post_weaning_component_state_key,
)
from pure_integer_ai.experiments.pw00a_formal_transaction import (
    PW00A_EVENT_ABORTED,
    PW00A_EVENT_PREPARED,
    PW00AFormalEventStore,
    PW00AFormalTransactionError,
)
from pure_integer_ai.experiments.pw00a_inference_artifact import (
    ARTIFACT_PATH as INFERENCE_ARTIFACT_PATH,
    read_pw00a_w09_inference_artifact,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.backend_capability import capability_profile
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE
from pure_integer_ai.storage.spaces.registry import SpaceRegistry


AUTHORITY_RECEIPT_PATH = "data/ph2/manifests/pw00a_formal_load_authority_v1.json"


# object-model: exception
class PW00AFormalStartupError(PostWeaningStartupError):
    """PW-00A artifact、owner、发布或恢复状态不满足。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """给嵌套稳定键增加长度边界。"""
    return len(value), *value


def _digest_key(payload: bytes) -> tuple[int, ...]:
    """返回文件或稳定结构的 SHA-256 字节整数键。"""
    return tuple(hashlib.sha256(payload).digest())


def _stable_digest(value: tuple[int, ...]) -> tuple[int, ...]:
    """把稳定整数键规范编码后形成发布摘要。"""
    return _digest_key(canonical_json_bytes(list(value)))


def _hex(value: tuple[int, ...]) -> str:
    """把 32 字节整数键转换为小写十六进制摘要。"""
    if len(value) != 32 or any(type(item) is not int or not 0 <= item <= 255
                               for item in value):
        raise PW00AFormalStartupError("PW00A digest key 非法")
    return bytes(value).hex()


def _artifact_key(root: Path, relative_path: str) -> tuple[int, ...]:
    """读取仓库内固定 artifact 字节并形成摘要键。"""
    supplied = Path(relative_path)
    target = (
        supplied.resolve()
        if supplied.is_absolute()
        else (root / Path(*relative_path.replace("\\", "/").split("/"))).resolve()
    )
    if not target.is_file():
        raise PW00AFormalStartupError(
            f"PW00A artifact 缺失: {relative_path}")
    return _digest_key(target.read_bytes())


def _owner_spaces(ctx: TrainContext) -> tuple[tuple[int, int], ...]:
    """返回 Core、双 Memory 与 Companion 的稳定顺序运行时编号。"""
    try:
        companion = ctx.memory_read_intake.source_intake.companion
    except AttributeError as error:
        raise PW00AFormalStartupError("PW00A 缺少 Companion owner") from error
    values = (
        (1, ctx.core_space.space_id),
        (2, ctx.memory_read.space_id),
        (3, ctx.memory_interact.space_id),
        (4, companion.space_id),
    )
    if len({space_id for _, space_id in values}) != 4:
        raise PW00AFormalStartupError("PW00A 四个空间 owner 未隔离")
    return values


def _owner_identity_key(ctx: TrainContext) -> tuple[int, ...]:
    """绑定四个运行时编号及其可跨重启稳定的空间身份。"""
    registry = SpaceRegistry(ctx.backend)
    result = [1, 4]
    for tag, space_id in _owner_spaces(ctx):
        identity = registry.identity(space_id).stable_key()
        result.extend((tag, space_id, *_packed(identity)))
    return tuple(result)


def _space_local_watermarks(ctx: TrainContext) -> dict[int, int]:
    """扫描全部同时含 space_id/local_id 的表并返回各 owner 最高水位。"""
    spaces = {space_id: 0 for _, space_id in _owner_spaces(ctx)}
    schema = ctx.backend.schema_snapshot()
    for table in sorted(schema):
        columns = set(schema[table]["columns"])
        if not {"space_id", "local_id"}.issubset(columns):
            continue
        for space_id in spaces:
            rows = ctx.backend.select(table, where={"space_id": space_id})
            values = tuple(
                row.get("local_id") for row in rows
                if type(row.get("local_id")) is int)
            if values:
                spaces[space_id] = max(spaces[space_id], max(values))
    return spaces


def _resume_owner_id_pools(ctx: TrainContext) -> dict[int, int]:
    """按全部直接 owner 表推高 ID pool，避免 restart 后复用既有 local_id。"""
    watermarks = _space_local_watermarks(ctx)
    for space_id, floor in watermarks.items():
        ctx.backend.advance_id_pool(space_id, floor)
    return watermarks


def _max_field(ctx: TrainContext, table: str, space_id: int, field: str) -> int:
    """返回一个 owner 表整数序字段的非负最高值。"""
    rows = ctx.backend.select(table, where={"space_id": space_id})
    values = tuple(
        row.get(field) for row in rows if type(row.get(field)) is int)
    return max(values, default=0)


def _owner_watermark_key(
        ctx: TrainContext,
        local_watermarks: dict[int, int],
        ) -> tuple[int, ...]:
    """冻结双 Memory 时序、Companion、来源计数和四空间 ID 水位。"""
    spaces = dict(_owner_spaces(ctx))
    result = [1]
    for tag in (1, 2, 3, 4):
        space_id = spaces[tag]
        result.extend((
            local_watermarks[space_id],
            ctx.backend.id_pool_floor(space_id),
        ))
    for tag in (2, 3):
        space_id = spaces[tag]
        result.extend((
            _max_field(ctx, MEMORY_EVENT_TABLE, space_id, "event_seq"),
            _max_field(ctx, MEMORY_EVENT_TABLE, space_id, "timeline_seq"),
        ))
    result.extend((
        _max_field(ctx, TEXT_ASSOC_TABLE, spaces[4], "assoc_id"),
        ctx.backend.count(SOURCE_RECORD_TABLE, where=None),
    ))
    return tuple(result)


def _watermark_not_older(
        current: tuple[int, ...],
        published: tuple[int, ...],
        ) -> bool:
    """要求恢复状态逐项不低于首次发布水位。"""
    return (
        len(current) == len(published)
        and current[:1] == published[:1] == (1,)
        and all(now >= prior for now, prior in zip(current[1:], published[1:]))
    )


def _authority_ready(value: object) -> None:
    """只接受已重新发布 sealed readiness 且尚未伪造启动位的 authority。"""
    if not isinstance(value, dict):
        raise PW00AFormalStartupError("PW00A authority 根类型错误")
    if value.get("status") != "PW00A_FORMAL_LOAD_AUTHORITY_EVIDENCED":
        raise PW00AFormalStartupError("PW00A authority 状态未闭合")
    if value.get("readiness_transition") != {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS_REPUBLISHED": 1,
            "PW00A_STARTED": 0}:
        raise PW00AFormalStartupError("PW00A authority readiness 漂移")


def _prepared_payload(
        manifest: FormalPostWeaningManifest,
        previous_manifest_key: tuple[int, ...],
        ) -> dict[str, Any]:
    """形成不含路径、文本或自由错误消息的 PREPARED 载荷。"""
    return {
        "authority_receipt_sha256": _hex(
            manifest.request.authority_receipt_key),
        "inference_artifact_sha256": _hex(
            manifest.request.inference_artifact_key),
        "manifest_key": list(manifest.stable_key()),
        "owner_watermark_key": list(manifest.owner_watermark_key),
        "previous_manifest_key": list(previous_manifest_key),
        "pw00a_started": 0,
    }


def _published_payload(
        prepared_sha256: str,
        report: FormalPostWeaningStartupReport,
        ) -> dict[str, Any]:
    """形成使正式状态唯一可见的 PUBLISHED 载荷。"""
    return {
        "prepared_payload_sha256": prepared_sha256,
        "pw00a_started": 1,
        "startup_report_key": list(report.stable_key()),
    }


def _aborted_payload(error: BaseException) -> dict[str, Any]:
    """只记录异常类型身份，不把自由消息写入正式历史。"""
    hasher = Hasher("pw00a.formal.abort.v1")
    return {
        "error_module_hash": hasher.h63(error.__class__.__module__),
        "error_type_hash": hasher.h63(error.__class__.__qualname__),
        "pw00a_started": 0,
    }


AuthorityReader = Callable[[str | Path, str | Path], dict[str, Any]]
InferenceReader = Callable[
    [str | Path, str | Path], tuple[dict[str, Any], W09InferenceState]]


# object-model: lifecycle; owner=formal-post-weaning-context; cleanup=backend-close
class PW00AFormalRuntime:
    """已由持久 PUBLISHED 事件授权的正式 post-weaning 入口。"""

    def __init__(
            self,
            operations: PostWeaningOperationRuntime,
            manifest: FormalPostWeaningManifest,
            startup_report: FormalPostWeaningStartupReport,
            inference_adapter: W09CandidateInferenceAdapter,
            ) -> None:
        """只接收完成发布后形成的共用入口与已装载推理状态。"""
        self._operations = operations
        self.manifest = manifest
        self.startup_report = startup_report
        self.inference_adapter = inference_adapter

    @property
    def ctx(self) -> TrainContext:
        """返回被正式状态保护的运行上下文。"""
        return self._operations.ctx

    def reports(self):
        """返回本实例已完成的 post-weaning 操作报告。"""
        return self._operations.reports()

    def run_intake(self, request):
        """委托共用入口执行受来源约束的正式摄入。"""
        return self._operations.run_intake(request)

    def run_question(self, dialogue, request):
        """委托共用入口执行正式问答并保持 Core 不变。"""
        return self._operations.run_question(dialogue, request)

    @classmethod
    def start(
            cls,
            ctx: TrainContext,
            request: FormalPostWeaningLoadRequest,
            *,
            repository_root: str | Path,
            authority_path: str | Path = AUTHORITY_RECEIPT_PATH,
            inference_path: str | Path = INFERENCE_ARTIFACT_PATH,
            authority_reader: AuthorityReader | None = None,
            inference_reader: InferenceReader | None = None,
            ) -> "PW00AFormalRuntime":
        """核验 authority 后首次发布，或从唯一 PUBLISHED 事件恢复。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("PW00A ctx 类型错误")
        if not isinstance(request, FormalPostWeaningLoadRequest):
            raise PW00AFormalStartupError("PW-00A formal load request 类型错误")
        root = Path(repository_root).resolve()
        authority_key = _artifact_key(root, str(authority_path))
        inference_key = _artifact_key(root, str(inference_path))
        if (authority_key != request.authority_receipt_key
                or inference_key != request.inference_artifact_key):
            raise PW00AFormalStartupError("PW00A request artifact identity 漂移")
        if authority_reader is None:
            from pure_integer_ai.experiments.pw00a_authority import (
                read_pw00a_formal_load_authority,
            )
            authority_reader = read_pw00a_formal_load_authority
        if inference_reader is None:
            inference_reader = read_pw00a_w09_inference_artifact
        authority = authority_reader(root, authority_path)
        _authority_ready(authority)
        _, inference_state = inference_reader(root, inference_path)
        inference_state_key = tuple(bytes.fromhex(inference_state.sha256()))
        inference_adapter = W09CandidateInferenceAdapter(inference_state)

        store = PW00AFormalEventStore(ctx.backend)
        local_watermarks = _resume_owner_id_pools(ctx)
        current_watermarks = _owner_watermark_key(ctx, local_watermarks)
        published = store.published_event()
        saved_watermarks = current_watermarks
        if published is not None:
            raw = published.payload.get("startup_report_key")
            prepared_events = store.events(published.run_id)
            prepared = prepared_events[0]
            saved = prepared.payload.get("owner_watermark_key")
            if (not isinstance(raw, list)
                    or not isinstance(saved, list)
                    or any(type(item) is not int for item in saved)):
                raise PW00AFormalStartupError("PW00A published payload 损坏")
            saved_watermarks = tuple(saved)
            if not _watermark_not_older(current_watermarks, saved_watermarks):
                raise PW00AFormalStartupError("PW00A resume owner 水位倒退")

        reader = CoreCanonicalStateReader(ctx)
        manifest = FormalPostWeaningManifest(
            request,
            inference_state_key,
            reader.read(),
            _schema_state_key(ctx),
            capability_profile(ctx.backend).stable_key(),
            post_weaning_component_state_key(ctx),
            _owner_identity_key(ctx),
            saved_watermarks,
            (*request.trace, 1),
        )
        manifest_key = manifest.stable_key()
        manifest_digest = _stable_digest(manifest_key)
        manifest_sha256 = _hex(manifest_digest)

        if published is not None:
            if (published.run_id != request.run_id
                    or published.publish_epoch != request.publish_epoch
                    or published.manifest_sha256 != manifest_sha256):
                raise PW00AFormalStartupError(
                    "PW00A 已发布 manifest 与恢复请求漂移")
            required_phase = ctx.weaning_phase
            if required_phase not in {WEANING_PRE, WEANING_POST}:
                raise PW00AFormalStartupError("PW00A resume phase 非法")
            ctx.backend.protect_owner_space(ctx.core_space.space_id)
            operations = PostWeaningOperationRuntime(
                ctx,
                manifest,
                core_reader=reader,
                required_phase=required_phase,
            )
            ctx.weaning_phase = WEANING_POST
            report = FormalPostWeaningStartupReport(
                request.run_id,
                request.publish_epoch,
                PW00A_START_RESUMED,
                manifest_key,
                tuple(published.payload.get("previous_manifest_key", ())),
                authority_key,
                inference_key,
                manifest.core_state_key,
                current_watermarks,
                ctx.backend.owner_write_protection_state(),
                (*request.trace, 2),
            )
            return cls(operations, manifest, report, inference_adapter)

        incomplete = tuple(
            event for event in store.all_events()
            if event.event_kind == PW00A_EVENT_PREPARED
            and len(store.events(event.run_id)) == 1)
        if incomplete:
            raise PW00AFormalStartupError(
                "PW00A 存在 prepared-only run，必须先显式封存 ABORTED")
        if ctx.weaning_phase != WEANING_PRE:
            raise PW00AFormalStartupError("PW00A 首次发布前 phase 非 WEANING_PRE")
        if request.publish_epoch != len({
                item.publish_epoch for item in store.all_events()}) + 1:
            raise PW00AFormalStartupError("PW00A publish_epoch 不是下一顺序值")

        operations = PostWeaningOperationRuntime(
            ctx,
            manifest,
            core_reader=reader,
            required_phase=WEANING_PRE,
        )
        previous_manifest_key: tuple[int, ...] = ()
        prepared_payload = _prepared_payload(manifest, previous_manifest_key)
        protection_before = ctx.backend.owner_write_protection_state()
        phase_before = ctx.weaning_phase
        prepared = store.prepared(
            run_id=request.run_id,
            publish_epoch=request.publish_epoch,
            manifest_sha256=manifest_sha256,
            payload=prepared_payload,
        )
        try:
            ctx.backend.protect_owner_space(ctx.core_space.space_id)
            report = FormalPostWeaningStartupReport(
                request.run_id,
                request.publish_epoch,
                PW00A_START_PUBLISHED,
                manifest_key,
                previous_manifest_key,
                authority_key,
                inference_key,
                manifest.core_state_key,
                current_watermarks,
                ctx.backend.owner_write_protection_state(),
                (*request.trace, 1),
            )
            store.published(
                run_id=request.run_id,
                publish_epoch=request.publish_epoch,
                manifest_sha256=manifest_sha256,
                payload=_published_payload(prepared.payload_sha256, report),
            )
        except BaseException as error:
            ctx.backend.restore_owner_write_protection(protection_before)
            ctx.weaning_phase = phase_before
            if store.published_event() is None:
                try:
                    store.aborted(
                        run_id=request.run_id,
                        publish_epoch=request.publish_epoch,
                        manifest_sha256=manifest_sha256,
                        payload=_aborted_payload(error),
                    )
                except BaseException as abort_error:
                    raise PW00AFormalTransactionError(
                        "PW00A 启动与 ABORTED 封存同时失败") from abort_error
            raise
        ctx.weaning_phase = WEANING_POST
        return cls(operations, manifest, report, inference_adapter)

    @classmethod
    def abort_prepared(
            cls,
            ctx: TrainContext,
            *,
            run_id: int,
            trace: tuple[int, ...],
            ) -> None:
        """显式封存 crash 留下的 prepared-only run，允许新 epoch 重试。"""
        del cls
        if not isinstance(ctx, TrainContext):
            raise TypeError("PW00A abort ctx 类型错误")
        if (not isinstance(trace, tuple) or not trace
                or any(type(item) is not int for item in trace)):
            raise PW00AFormalStartupError("PW00A abort trace 非法")
        store = PW00AFormalEventStore(ctx.backend)
        events = store.events(run_id)
        if len(events) != 1 or events[0].event_kind != PW00A_EVENT_PREPARED:
            raise PW00AFormalStartupError("PW00A run 不是 prepared-only")
        error = PW00AFormalStartupError(
            f"PW00A prepared-only recovery {Hasher('pw00a.abort.trace').h63(trace)}")
        store.aborted(
            run_id=events[0].run_id,
            publish_epoch=events[0].publish_epoch,
            manifest_sha256=events[0].manifest_sha256,
            payload=_aborted_payload(error),
        )


__all__ = [
    "AUTHORITY_RECEIPT_PATH",
    "PW00AFormalRuntime",
    "PW00AFormalStartupError",
]
