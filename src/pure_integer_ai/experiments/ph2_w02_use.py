"""W-02 理解/生成实际采用对象的方向性 Use、outcome 与 assessment 事件账。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


DIRECTION_UNDERSTANDING = "UNDERSTANDING"
DIRECTION_GENERATION = "GENERATION"
OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_FAILURE = "FAILURE"
OUTCOME_UNKNOWN = "UNKNOWN"

_DIRECTIONS = (DIRECTION_GENERATION, DIRECTION_UNDERSTANDING)
_OUTCOMES = (OUTCOME_FAILURE, OUTCOME_SUCCESS, OUTCOME_UNKNOWN)
_EVENT_USE = 1
_EVENT_OUTCOME = 2
_EVENT_ASSESSMENT = 3
_EVENT_KINDS = (_EVENT_USE, _EVENT_OUTCOME, _EVENT_ASSESSMENT)
_TABLE = "ph2_w02_use_outcome_event"
_FORMAT_VERSION = 1
_CONSUMERS = {
    DIRECTION_UNDERSTANDING: (920301, 1),
    DIRECTION_GENERATION: (920301, 2),
}
_ROUTE_HASHER = Hasher("ph2.w02.use.route.v1")


class W02UseOutcomeError(RuntimeError):
    """W-02 Use/outcome 路由、事件序或 exact Candidate 归因损坏。"""


def _key(value: tuple[int, ...], *, where: str,
         allow_empty: bool = False) -> tuple[int, ...]:
    """核验开放整数键，禁止 bool 冒充整数。"""
    if (not isinstance(value, tuple)
            or (not value and not allow_empty)
            or any(type(item) is not int for item in value)):
        raise W02UseOutcomeError(f"{where} 必须是整数 tuple")
    return value


def _direction(value: str) -> str:
    """核验理解/生成方向枚举。"""
    if value not in _DIRECTIONS:
        raise W02UseOutcomeError("W-02 Use direction 未登记")
    return value


def _outcome(value: str) -> str:
    """核验 outcome 三态枚举。"""
    if value not in _OUTCOMES:
        raise W02UseOutcomeError("W-02 Use outcome 未登记")
    return value


def _route_id(event_kind: int, direction: str,
              use_key: tuple[int, ...]) -> int:
    """从完整 event kind/direction/use route 形成稳定短索引。"""
    route = canonical_json_bytes({
        "direction": direction,
        "event_kind": event_kind,
        "use_key": list(use_key),
    })
    return _ROUTE_HASHER.h63(tuple(route)) or 1


def _event_value(
        *,
        event_kind: int,
        direction: str,
        logical_seq: int,
        use_key: tuple[int, ...],
        request_key: tuple[int, ...],
        candidate_key: tuple[int, ...],
        outcome_kind: str,
        outcome_trace_key: tuple[int, ...],
        assessment_delta: int,
        ) -> dict[str, Any]:
    """构造字段精确的持久事件对象。"""
    return {
        "assessment_delta": assessment_delta,
        "candidate_key": list(candidate_key),
        "consumer_key": list(_CONSUMERS[direction]),
        "direction": direction,
        "event_kind": event_kind,
        "format_version": _FORMAT_VERSION,
        "logical_seq": logical_seq,
        "outcome_kind": outcome_kind,
        "outcome_trace_key": list(outcome_trace_key),
        "request_key": list(request_key),
        "use_key": list(use_key),
    }


@dataclass(frozen=True)
class W02DirectionalAttribution:
    """一个 actual Candidate 的 Use、结果和可选 assessment 更新。"""

    direction: str
    consumer_key: tuple[int, ...]
    use_key: tuple[int, ...]
    request_key: tuple[int, ...]
    candidate_key: tuple[int, ...]
    outcome_kind: str
    outcome_trace_key: tuple[int, ...]
    use_seq: int
    outcome_seq: int
    assessment_seq: int
    assessment_delta: int

    def __post_init__(self) -> None:
        """核方向、exact Candidate、逻辑序和 outcome-to-assessment 关系。"""
        _direction(self.direction)
        if self.consumer_key != _CONSUMERS[self.direction]:
            raise W02UseOutcomeError("W-02 consumer 与方向不一致")
        for label, value, allow_empty in (
                ("use_key", self.use_key, False),
                ("request_key", self.request_key, False),
                ("candidate_key", self.candidate_key, False),
                ("outcome_trace_key", self.outcome_trace_key, True)):
            _key(value, where=label, allow_empty=allow_empty)
        _outcome(self.outcome_kind)
        if (type(self.use_seq) is not int or self.use_seq <= 0
                or type(self.outcome_seq) is not int
                or self.outcome_seq != self.use_seq + 1
                or type(self.assessment_seq) is not int
                or self.assessment_seq < 0
                or type(self.assessment_delta) is not int
                or self.assessment_delta not in (-1, 0, 1)):
            raise W02UseOutcomeError("W-02 Use/outcome logical seq 非法")
        expected_delta = {
            OUTCOME_SUCCESS: 1,
            OUTCOME_FAILURE: -1,
            OUTCOME_UNKNOWN: 0,
        }[self.outcome_kind]
        if self.assessment_seq:
            if (self.assessment_seq != self.outcome_seq + 1
                    or self.assessment_delta != expected_delta
                    or self.assessment_delta == 0):
                raise W02UseOutcomeError("W-02 assessment 与 outcome 不一致")
        elif self.assessment_delta != 0:
            raise W02UseOutcomeError("关闭 assessment 时不得留下 effect")

    def stable_key(self) -> tuple:
        """返回方向、actual Candidate、结果及逻辑序的完整状态。"""
        return (
            self.direction,
            self.consumer_key,
            self.use_key,
            self.request_key,
            self.candidate_key,
            self.outcome_kind,
            self.outcome_trace_key,
            self.use_seq,
            self.outcome_seq,
            self.assessment_seq,
            self.assessment_delta,
        )


@dataclass(frozen=True)
class W02AttributionReport:
    """理解/生成分账后的持久 Use、outcome 和 assessment 计数。"""

    use_count_by_direction: tuple[tuple[str, int], ...]
    outcome_count: int
    assessment_count: int
    consumer_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """要求方向排序唯一，计数使用非负严格整数。"""
        if (not isinstance(self.use_count_by_direction, tuple)
                or tuple(item[0] for item in self.use_count_by_direction)
                != tuple(sorted(item[0] for item in self.use_count_by_direction))
                or any(item[0] not in _DIRECTIONS
                       or type(item[1]) is not int or item[1] < 0
                       for item in self.use_count_by_direction)
                or type(self.outcome_count) is not int
                or self.outcome_count < 0
                or type(self.assessment_count) is not int
                or self.assessment_count < 0):
            raise W02UseOutcomeError("W-02 attribution report 非法")


class W02UseOutcomeStore:
    """复用统一 backend 的窄 append-only W-02 方向事件 owner。"""

    def __init__(self, backend: StorageBackend) -> None:
        """注册唯一领域表；不创建第二套通用 Use 或 Memory。"""
        if not isinstance(backend, StorageBackend):
            raise TypeError("W-02 Use backend 类型错误")
        self.backend = backend
        register_extension_table(
            backend,
            _TABLE,
            [
                ("route_id", TYPE_INT),
                ("event_kind", TYPE_INT),
                ("direction_kind", TYPE_INT),
                ("logical_seq", TYPE_INT),
                ("payload_json", TYPE_TEXT),
            ],
            disc.DISC_APPEND_ONLY,
            indexes=[
                ("route_id",),
                ("event_kind",),
                ("direction_kind",),
                ("logical_seq",),
            ],
            recovery_key=("route_id",),
        )

    def _read_value(self, row: dict[str, Any]) -> dict[str, Any]:
        """回读一行规范 JSON，并核索引列没有与 payload 漂移。"""
        text = row.get("payload_json")
        if not isinstance(text, str):
            raise W02UseOutcomeError("W-02 Use payload 类型损坏")
        payload = text.encode("utf-8")
        try:
            value = parse_canonical_json_bytes(payload, require_object=True)
        except (TypeError, ValueError) as exc:
            raise W02UseOutcomeError("W-02 Use payload JSON 损坏") from exc
        assert isinstance(value, dict)
        if canonical_json_bytes(value) != payload:
            raise W02UseOutcomeError("W-02 Use payload 非规范字节")
        expected_fields = {
            "assessment_delta", "candidate_key", "consumer_key", "direction",
            "event_kind", "format_version", "logical_seq", "outcome_kind",
            "outcome_trace_key", "request_key", "use_key",
        }
        if set(value) != expected_fields or value["format_version"] != _FORMAT_VERSION:
            raise W02UseOutcomeError("W-02 Use payload 字段或版本漂移")
        direction = _direction(value["direction"])
        event_kind = value["event_kind"]
        if event_kind not in _EVENT_KINDS:
            raise W02UseOutcomeError("W-02 Use event kind 未登记")
        use_key = tuple(value["use_key"])
        _key(use_key, where="persisted use_key")
        direction_kind = _DIRECTIONS.index(direction) + 1
        if (row.get("route_id") != _route_id(event_kind, direction, use_key)
                or row.get("event_kind") != event_kind
                or row.get("direction_kind") != direction_kind
                or row.get("logical_seq") != value["logical_seq"]):
            raise W02UseOutcomeError("W-02 Use 索引列与 payload 漂移")
        return value

    def _route_value(
            self, event_kind: int, direction: str,
            use_key: tuple[int, ...],
            ) -> dict[str, Any] | None:
        """按完整 route 短索引读取唯一行，并检测 hash 碰撞。"""
        rows = self.backend.select(_TABLE, {
            "route_id": _route_id(event_kind, direction, use_key)})
        if not rows:
            return None
        if len(rows) != 1:
            raise W02UseOutcomeError("W-02 Use route 重复")
        value = self._read_value(rows[0])
        if (value["event_kind"] != event_kind
                or value["direction"] != direction
                or tuple(value["use_key"]) != use_key):
            raise W02UseOutcomeError("W-02 Use route hash 碰撞")
        return value

    def _insert(self, value: dict[str, Any]) -> None:
        """在整组预检后追加一条规范事件。"""
        direction = value["direction"]
        event_kind = value["event_kind"]
        use_key = tuple(value["use_key"])
        self.backend.insert(_TABLE, {
            "route_id": _route_id(event_kind, direction, use_key),
            "event_kind": event_kind,
            "direction_kind": _DIRECTIONS.index(direction) + 1,
            "logical_seq": value["logical_seq"],
            "payload_json": canonical_json_bytes(value).decode("utf-8"),
        })

    def _next_seq(self) -> int:
        """从 append-only 账中取得下一逻辑序，不读取墙钟。"""
        rows = self.backend.select(
            _TABLE, order_by="logical_seq", descending=True, limit=1)
        return 1 if not rows else rows[0]["logical_seq"] + 1

    def record(
            self,
            *,
            direction: str,
            use_key: tuple[int, ...],
            request_key: tuple[int, ...],
            candidate_key: tuple[int, ...],
            outcome_kind: str,
            outcome_trace_key: tuple[int, ...] = (),
            assessment_enabled: bool = True,
            commit: bool = True,
            ) -> W02DirectionalAttribution:
        """原子语义预检一组 Use/outcome/assessment，重放幂等且 partial fail closed。"""
        direction = _direction(direction)
        use_key = _key(use_key, where="use_key")
        request_key = _key(request_key, where="request_key")
        candidate_key = _key(candidate_key, where="candidate_key")
        outcome_trace_key = _key(
            outcome_trace_key, where="outcome_trace_key", allow_empty=True)
        outcome_kind = _outcome(outcome_kind)
        if type(assessment_enabled) is not bool:
            raise TypeError("assessment_enabled 必须是 bool")
        if type(commit) is not bool:
            raise TypeError("commit 必须是 bool")
        kinds = [_EVENT_USE, _EVENT_OUTCOME]
        delta = {
            OUTCOME_SUCCESS: 1,
            OUTCOME_FAILURE: -1,
            OUTCOME_UNKNOWN: 0,
        }[outcome_kind]
        if assessment_enabled and delta:
            kinds.append(_EVENT_ASSESSMENT)
        all_existing = {
            kind: self._route_value(kind, direction, use_key)
            for kind in _EVENT_KINDS
        }
        present = {kind for kind, value in all_existing.items()
                   if value is not None}
        expected = set(kinds)
        if present and present != expected:
            raise W02UseOutcomeError("W-02 Use/outcome 出现 partial adoption")
        if present:
            existing = tuple(all_existing[kind] for kind in kinds)
            attribution = self._attribution(existing)
            if (attribution.request_key != request_key
                    or attribution.candidate_key != candidate_key
                    or attribution.outcome_kind != outcome_kind
                    or attribution.outcome_trace_key != outcome_trace_key):
                raise W02UseOutcomeError("同一 W-02 Use route 绑定了不同归因")
            return attribution
        base = self._next_seq()
        values = []
        for offset, kind in enumerate(kinds):
            values.append(_event_value(
                event_kind=kind,
                direction=direction,
                logical_seq=base + offset,
                use_key=use_key,
                request_key=request_key,
                candidate_key=candidate_key,
                outcome_kind=outcome_kind,
                outcome_trace_key=outcome_trace_key,
                assessment_delta=delta if kind == _EVENT_ASSESSMENT else 0,
            ))
        for value in values:
            self._insert(value)
        attribution = self._attribution(tuple(values))
        if commit:
            self.backend.commit()
        return attribution

    @staticmethod
    def _attribution(values: tuple[dict[str, Any] | None, ...]) -> (
            W02DirectionalAttribution):
        """从一组已核事件恢复 exact Candidate 归因并验证逻辑顺序。"""
        if any(item is None for item in values):
            raise W02UseOutcomeError("W-02 attribution 缺事件")
        typed = tuple(item for item in values if item is not None)
        by_kind = {item["event_kind"]: item for item in typed}
        if set(by_kind) not in (
                {_EVENT_USE, _EVENT_OUTCOME},
                {_EVENT_USE, _EVENT_OUTCOME, _EVENT_ASSESSMENT}):
            raise W02UseOutcomeError("W-02 attribution event kind 不完整")
        use = by_kind[_EVENT_USE]
        outcome = by_kind[_EVENT_OUTCOME]
        shared_fields = (
            "direction", "consumer_key", "use_key", "request_key",
            "candidate_key", "outcome_kind", "outcome_trace_key",
        )
        if any(use[field] != outcome[field] for field in shared_fields):
            raise W02UseOutcomeError("W-02 outcome 未精确指向 Use Candidate")
        assessment = by_kind.get(_EVENT_ASSESSMENT)
        if assessment is not None and any(
                use[field] != assessment[field] for field in shared_fields):
            raise W02UseOutcomeError("W-02 assessment 未精确指向 Use Candidate")
        return W02DirectionalAttribution(
            use["direction"],
            tuple(use["consumer_key"]),
            tuple(use["use_key"]),
            tuple(use["request_key"]),
            tuple(use["candidate_key"]),
            use["outcome_kind"],
            tuple(use["outcome_trace_key"]),
            use["logical_seq"],
            outcome["logical_seq"],
            0 if assessment is None else assessment["logical_seq"],
            0 if assessment is None else assessment["assessment_delta"],
        )

    def attributions(self) -> tuple[W02DirectionalAttribution, ...]:
        """全表验链并按逻辑序恢复每个 Use 的方向性归因。"""
        rows = self.backend.select(_TABLE, order_by="logical_seq")
        values = tuple(self._read_value(row) for row in rows)
        groups: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
        for value in values:
            route = value["direction"], tuple(value["use_key"])
            groups.setdefault(route, []).append(value)
        result = tuple(self._attribution(tuple(group)) for group in groups.values())
        ordered = tuple(sorted(result, key=lambda item: item.use_seq))
        expected_seq = 1
        for item in ordered:
            if item.use_seq != expected_seq:
                raise W02UseOutcomeError("W-02 Use/outcome logical seq 非连续")
            expected_seq = item.assessment_seq + 1 if item.assessment_seq else (
                item.outcome_seq + 1)
        return ordered

    def score(self, direction: str, candidate_key: tuple[int, ...]) -> int:
        """只汇总该方向该 actual Candidate 的 assessment effect。"""
        direction = _direction(direction)
        candidate_key = _key(candidate_key, where="candidate_key")
        return sum(
            item.assessment_delta for item in self.attributions()
            if item.direction == direction and item.candidate_key == candidate_key)

    def report(self) -> W02AttributionReport:
        """返回方向 Use、outcome 和真实 assessment 写计数。"""
        values = self.attributions()
        counts = tuple(
            (direction, sum(item.direction == direction for item in values))
            for direction in _DIRECTIONS
            if any(item.direction == direction for item in values)
        )
        return W02AttributionReport(
            counts,
            len(values),
            sum(bool(item.assessment_seq) for item in values),
            tuple(sorted({item.consumer_key for item in values})),
        )

    def state_key(self) -> tuple:
        """返回全量 Use/outcome/assessment 规范逻辑状态。"""
        return tuple(item.stable_key() for item in self.attributions())


__all__ = [
    "DIRECTION_GENERATION",
    "DIRECTION_UNDERSTANDING",
    "OUTCOME_FAILURE",
    "OUTCOME_SUCCESS",
    "OUTCOME_UNKNOWN",
    "W02AttributionReport",
    "W02DirectionalAttribution",
    "W02UseOutcomeError",
    "W02UseOutcomeStore",
]
