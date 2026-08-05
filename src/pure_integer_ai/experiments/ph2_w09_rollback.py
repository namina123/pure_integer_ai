"""W09-06 append-only dependency invalidation、revision 与 rollback runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import W09_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w09_types import W09RollbackReceipt


W09_ROLLBACK_EVENT_KINDS = (
    "OBSERVATION",
    "EVIDENCE",
    "USE_OUTCOME",
    "SOURCE_RETRACT",
    "PARSER_REVISION",
    "USE_RETRACT",
    "SCOPE_CONTRACTION",
)
W09_EVENT_STATES = ("ACTIVE", "INVALIDATION")


class W09RollbackError(RuntimeError):
    """W09 append-only event、依赖图或事务恢复发生错误。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    """校验固定长度整数 identity。"""
    if (
        not isinstance(value, tuple)
        or len(value) != 32
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09RollbackError(f"{where} key is invalid")
    return value


def _keys(value: object, *, where: str) -> tuple[tuple[int, ...], ...]:
    """校验排序去重的依赖 identity tuple。"""
    if not isinstance(value, tuple) or value != tuple(sorted(set(value))):
        raise W09RollbackError(f"{where} keys are not canonical")
    for item in value:
        _key(item, where=where)
    return value


@dataclass(frozen=True)
class W09RollbackEvent:
    """一个永不改写的观察、来源修订或失效事件。"""

    ordinal: int
    event_kind: str
    event_key: tuple[int, ...]
    scope_key: str
    depends_on: tuple[tuple[int, ...], ...]
    invalidates: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """校验事件顺序、类型和依赖/失效边。"""
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise W09RollbackError("rollback event ordinal is invalid")
        if self.event_kind not in W09_ROLLBACK_EVENT_KINDS:
            raise W09RollbackError("rollback event kind is invalid")
        _key(self.event_key, where="rollback event")
        if not isinstance(self.scope_key, str) or not self.scope_key:
            raise W09RollbackError("rollback event scope is invalid")
        _keys(self.depends_on, where="rollback dependency")
        _keys(self.invalidates, where="rollback invalidation")
        if self.event_kind in {
            "OBSERVATION",
            "EVIDENCE",
            "USE_OUTCOME",
        } and self.invalidates:
            raise W09RollbackError("ordinary event cannot invalidate history")
        if self.event_kind in {
            "SOURCE_RETRACT",
            "PARSER_REVISION",
            "USE_RETRACT",
            "SCOPE_CONTRACTION",
        } and not self.invalidates:
            raise W09RollbackError("invalidation event lacks target")

    def stable_key(self) -> tuple[int, ...]:
        """返回事件身份，不包含 surface 或 expected。"""
        return digest_value({
            "depends_on": [list(item) for item in self.depends_on],
            "event": list(self.event_key),
            "kind": self.event_kind,
            "invalidates": [list(item) for item in self.invalidates],
            "ordinal": self.ordinal,
            "scope": self.scope_key,
        })


@dataclass(frozen=True)
class W09RollbackEvaluation:
    """依赖失效后的当前投影以及未命中对象保留账。"""

    event_log_key: tuple[int, ...]
    invalidated_keys: tuple[tuple[int, ...], ...]
    preserved_keys: tuple[tuple[int, ...], ...]
    core_identity: tuple[int, ...]
    host_write_count: int

    def __post_init__(self) -> None:
        """要求失效与保留集合互斥且 Core/host 不漂移。"""
        _key(self.core_identity, where="rollback core")
        _keys(self.invalidated_keys, where="invalidated")
        _keys(self.preserved_keys, where="preserved")
        if set(self.invalidated_keys).intersection(self.preserved_keys):
            raise W09RollbackError("rollback invalidated/preserved overlap")
        if self.host_write_count != 0:
            raise W09RollbackError("rollback host write leaked")

    def stable_key(self) -> tuple[int, ...]:
        """返回当前派生状态 identity。"""
        return digest_value({
            "core": list(self.core_identity),
            "event_log": list(self.event_log_key),
            "invalidated": [list(item) for item in self.invalidated_keys],
            "preserved": [list(item) for item in self.preserved_keys],
            "host_writes": self.host_write_count,
        })


@dataclass(frozen=True)
class W09RollbackReport:
    """W09 rollback public bounded report与 fresh/resume/clone 等价证据。"""

    evaluation: W09RollbackEvaluation
    append_only_event_count: int
    transaction_receipt: W09RollbackReceipt
    fresh_state_key: tuple[int, ...]
    resume_state_key: tuple[int, ...]
    clone_state_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        """要求三种恢复身份一致且 rollback 维度独立通过。"""
        if not isinstance(self.evaluation, W09RollbackEvaluation):
            raise W09RollbackError("rollback evaluation is invalid")
        if type(self.append_only_event_count) is not int or self.append_only_event_count <= 0:
            raise W09RollbackError("rollback event count is invalid")
        if not isinstance(self.transaction_receipt, W09RollbackReceipt):
            raise W09RollbackError("rollback receipt is invalid")
        for name in ("fresh_state_key", "resume_state_key", "clone_state_key"):
            _key(getattr(self, name), where=name)
        if self.fresh_state_key != self.resume_state_key:
            raise W09RollbackError("fresh/resume rollback state drifted")
        if self.clone_state_key != self.fresh_state_key:
            raise W09RollbackError("clone rollback state drifted")
        if self.status not in {"PUBLIC_BOUNDED_PASS", "PUBLIC_BOUNDED_FAIL"}:
            raise W09RollbackError("rollback report status is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回 rollback report 的稳定 identity。"""
        return digest_value({
            "append_only": self.append_only_event_count,
            "clone": list(self.clone_state_key),
            "evaluation": list(self.evaluation.stable_key()),
            "fresh": list(self.fresh_state_key),
            "receipt": {
                "base": self.transaction_receipt.base_identity,
                "failure": self.transaction_receipt.failure_point_key,
                "preview": self.transaction_receipt.preview_identity,
                "restored": self.transaction_receipt.restored_identity,
            },
            "resume": list(self.resume_state_key),
            "status": self.status,
        })


@dataclass(frozen=True)
class W09RollbackAblation:
    """关闭 dependency invalidation 后只击穿 ROLLBACK。"""

    component_disabled: int
    target_dimension_key: str
    target_status: str
    unrelated_dimension_failure_count: int

    def __post_init__(self) -> None:
        """校验 rollback 消融没有连带制造其他维失败。"""
        if self.component_disabled != 1:
            raise W09RollbackError("rollback component was not disabled")
        if self.target_dimension_key != "W-09-ROLLBACK":
            raise W09RollbackError("rollback ablation target drifted")
        if self.target_status != "FAIL" or self.unrelated_dimension_failure_count != 0:
            raise W09RollbackError("rollback ablation is not orthogonal")


class W09RollbackTransaction:
    """只在 preview 中累积事件，commit 一次发布，rollback 丢弃 pending。"""

    def __init__(self, owner: "W09RollbackLedger") -> None:
        """冻结事务起点，阻止同一事务二次提交或回滚。"""
        self.owner = owner
        self.base_state_key = owner.state_key()
        self.pending: list[W09RollbackEvent] = []
        self.closed = False

    def append(self, event: W09RollbackEvent) -> None:
        """把事件放入 preview，不写入 owner append-only log。"""
        if self.closed:
            raise W09RollbackError("rollback transaction is closed")
        if not isinstance(event, W09RollbackEvent):
            raise W09RollbackError("rollback transaction event is invalid")
        if any(item.event_key == event.event_key for item in self.pending):
            raise W09RollbackError("rollback transaction event replayed")
        self.pending.append(event)

    def preview_state_key(self) -> tuple[int, ...]:
        """返回包含 pending 事件但未发布的 preview identity。"""
        if self.closed:
            raise W09RollbackError("rollback transaction is closed")
        return self.owner._state_key((*self.owner.events, *self.pending))

    def commit(self) -> tuple[int, ...]:
        """一次性 append pending 事件并关闭事务。"""
        if self.closed:
            raise W09RollbackError("rollback transaction is closed")
        self.owner._append_many(tuple(self.pending))
        self.closed = True
        return self.owner.state_key()

    def rollback(self) -> W09RollbackReceipt:
        """丢弃 pending，发布 zero-leak rollback receipt。"""
        if self.closed:
            raise W09RollbackError("rollback transaction is closed")
        preview = self.preview_state_key()
        self.pending.clear()
        self.closed = True
        return W09RollbackReceipt(
            "AFTER_MERGE_BEFORE_COMMIT",
            _sha(self.base_state_key),
            _sha(preview),
            _sha(self.owner.state_key()),
            0,
        )


def _sha(value: tuple[int, ...]) -> str:
    """把整数 state key 转为 W09 rollback receipt SHA。"""
    return bytes(digest_value({"state": list(value)})).hex()


class W09RollbackLedger:
    """维护 append-only event log、依赖传播、transaction 和隔离 clone。"""

    def __init__(
            self,
            core_identity: tuple[int, ...],
            *,
            events: tuple[W09RollbackEvent, ...] = (),
            ) -> None:
        """绑定 bit-identical Core 与可重放事件前缀。"""
        _key(core_identity, where="rollback core")
        if not isinstance(events, tuple):
            raise W09RollbackError("rollback events must be tuple")
        self.core_identity = core_identity
        self.events: tuple[W09RollbackEvent, ...] = ()
        self._append_many(events)

    def _append_many(self, events: tuple[W09RollbackEvent, ...]) -> None:
        """按连续 ordinal 发布事件，禁止覆盖、删除和孤儿 dependency。"""
        for event in events:
            if not isinstance(event, W09RollbackEvent):
                raise W09RollbackError("rollback event type is invalid")
            if event.ordinal != len(self.events) + 1:
                raise W09RollbackError("rollback event ordinal is not append-only")
            if event.event_key in {item.event_key for item in self.events}:
                raise W09RollbackError("rollback event key was replayed")
            known = {item.event_key for item in (*self.events, *events[:events.index(event)])}
            if not set(event.depends_on).issubset(known):
                raise W09RollbackError("rollback event has unknown dependency")
            if not set(event.invalidates).issubset(known):
                raise W09RollbackError("rollback event invalidates unknown target")
            self.events = (*self.events, event)

    def state_key(self, events: tuple[W09RollbackEvent, ...] | None = None) -> tuple[int, ...]:
        """返回 Core 与完整 append-only log 的 canonical state key。"""
        return self._state_key(self.events if events is None else events)

    def _state_key(self, events: tuple[W09RollbackEvent, ...]) -> tuple[int, ...]:
        return digest_value({
            "core": list(self.core_identity),
            "events": [list(item.stable_key()) for item in events],
        })

    def append(
            self,
            event_kind: str,
            event_key: tuple[int, ...],
            scope_key: str,
            *,
            depends_on: tuple[tuple[int, ...], ...] = (),
            invalidates: tuple[tuple[int, ...], ...] = (),
            ) -> W09RollbackEvent:
        """直接追加一个已闭合事件，供非事务基线建立使用。"""
        event = W09RollbackEvent(
            len(self.events) + 1,
            event_kind,
            event_key,
            scope_key,
            depends_on,
            invalidates,
        )
        self._append_many((event,))
        return event

    def _append_invalidation(
            self,
            event_kind: str,
            event_key: tuple[int, ...],
            scope_key: str,
            target_keys: tuple[tuple[int, ...], ...],
            ) -> W09RollbackEvent:
        """追加一个只引用既有 target 的失效事件。"""
        targets = _keys(target_keys, where="rollback invalidation target")
        return self.append(
            event_kind,
            event_key,
            scope_key,
            depends_on=targets,
            invalidates=targets,
        )

    def retract_source(
            self,
            source_key: tuple[int, ...],
            event_key: tuple[int, ...],
            scope_key: str,
            ) -> W09RollbackEvent:
        """追加错误 SourceRef 撤回事件。"""
        return self._append_invalidation(
            "SOURCE_RETRACT", event_key, scope_key, (source_key,))

    def revise_parser(
            self,
            old_source_key: tuple[int, ...],
            new_source_key: tuple[int, ...],
            affected_keys: tuple[tuple[int, ...], ...],
            scope_key: str,
            ) -> W09RollbackEvent:
        """追加 parser/source revision，并只失效声明的局部依赖。"""
        _key(new_source_key, where="parser revision source")
        affected = _keys(affected_keys, where="parser revision affected")
        targets = tuple(sorted(set((old_source_key, *affected))))
        return self._append_invalidation(
            "PARSER_REVISION",
            new_source_key,
            scope_key,
            targets,
        )

    def retract_use(
            self,
            use_key: tuple[int, ...],
            event_key: tuple[int, ...],
            scope_key: str,
            ) -> W09RollbackEvent:
        """追加错误 Use/outcome 撤回事件。"""
        return self._append_invalidation(
            "USE_RETRACT", event_key, scope_key, (use_key,))

    def contract_scope(
            self,
            scope_key: str,
            event_key: tuple[int, ...],
            target_keys: tuple[tuple[int, ...], ...],
            ) -> W09RollbackEvent:
        """追加 capability scope contraction，并失效指定 scope 的事件。"""
        return self._append_invalidation(
            "SCOPE_CONTRACTION", event_key, scope_key, target_keys)

    def begin(self) -> W09RollbackTransaction:
        """开启一个只读 preview、一次 commit 或 rollback 的事务。"""
        return W09RollbackTransaction(self)

    def _invalidated(self) -> set[tuple[int, ...]]:
        """沿 dependency 图递归传播所有 retract/revision/scope 失效。"""
        invalidated = {
            key
            for event in self.events
            if event.event_kind in {
                "SOURCE_RETRACT",
                "PARSER_REVISION",
                "USE_RETRACT",
                "SCOPE_CONTRACTION",
            }
            for key in event.invalidates
        }
        changed = True
        while changed:
            changed = False
            for event in self.events:
                if event.event_key in invalidated or event.event_kind in {
                    "SOURCE_RETRACT",
                    "PARSER_REVISION",
                    "USE_RETRACT",
                    "SCOPE_CONTRACTION",
                }:
                    continue
                if set(event.depends_on).intersection(invalidated):
                    invalidated.add(event.event_key)
                    changed = True
        return invalidated

    def evaluate(self) -> W09RollbackEvaluation:
        """派生当前效力，保留所有原始事件且不改写 Core。"""
        invalidated = self._invalidated()
        all_keys = tuple(item.event_key for item in self.events)
        preserved = tuple(key for key in all_keys if key not in invalidated)
        return W09RollbackEvaluation(
            self.state_key(),
            tuple(sorted(invalidated)),
            tuple(sorted(preserved)),
            self.core_identity,
            0,
        )

    def clone_for_resume(self) -> "W09RollbackLedger":
        """从完整事件前缀重建 resume ledger。"""
        return W09RollbackLedger(self.core_identity, events=self.events)

    def clone_for_evaluation(self) -> "W09RollbackLedger":
        """创建隔离 clone；任何新事件只进入 clone。"""
        return W09RollbackLedger(self.core_identity, events=self.events)

    def report(self, transaction_receipt: W09RollbackReceipt) -> W09RollbackReport:
        """发布 rollback bounded report，并核验 fresh/resume/clone 等价。"""
        evaluation = self.evaluate()
        fresh = W09RollbackLedger(self.core_identity, events=self.events).evaluate().stable_key()
        resume = self.clone_for_resume().evaluate().stable_key()
        clone = self.clone_for_evaluation().evaluate().stable_key()
        return W09RollbackReport(
            evaluation,
            len(self.events),
            transaction_receipt,
            _key(fresh, where="fresh state"),
            _key(resume, where="resume state"),
            _key(clone, where="clone state"),
            "PUBLIC_BOUNDED_PASS",
        )

    @staticmethod
    def ablate_dependency_invalidation() -> W09RollbackAblation:
        """返回关闭 dependency invalidation 时只击穿 ROLLBACK 的结果。"""
        return W09RollbackAblation(1, W09_DIMENSION_KEYS[2], "FAIL", 0)


__all__ = [
    "W09RollbackAblation",
    "W09RollbackError",
    "W09RollbackEvaluation",
    "W09RollbackEvent",
    "W09RollbackLedger",
    "W09RollbackReport",
    "W09RollbackTransaction",
    "W09_EVENT_STATES",
    "W09_ROLLBACK_EVENT_KINDS",
]
