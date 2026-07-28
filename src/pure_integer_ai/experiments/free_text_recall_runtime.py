"""Evidence 驱动的自由文本 center、ACL-first K-04 召回与 QA 适配。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import MemoryCurrentQuery
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_free_text_hierarchy_recall_contract import (
    AbsoluteSpan,
    RecallBudget,
    RecallCitation,
    RecallObligation,
    RecallReceipt,
)
from pure_integer_ai.experiments.ph2_md03_center_adapter import (
    DirectionalMemoryCenter,
    DirectionalMemoryCenterAdapter,
)
from pure_integer_ai.storage.query_hot_set import (
    QueryHotSetBudgetExceeded,
    QueryHotSetMetrics,
    QueryHotSetPolicy,
    QuerySegmentHotSet,
)
from pure_integer_ai.storage.sealed_segment import SegmentRecord
from pure_integer_ai.storage.tiered_segment_store import TieredSegmentStore


FEATURE_PAYLOAD_VERSION = 1
RECALL_PAYLOAD_VERSION = 1


class FreeTextRecallRuntimeError(RuntimeError):
    """自由文本 Evidence、center、冷记录或 QA 绑定不完整。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求运行期注入键非空且仅含正严格整数。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise FreeTextRecallRuntimeError(f"{where} 必须是正严格整数 tuple")
    return value


def _stable(domain: int, *parts: int) -> StableRecordKey:
    """从运行期完整整数输入形成稳定正整数记录键。"""
    if type(domain) is not int or domain <= 0:
        raise FreeTextRecallRuntimeError("stable key domain 非法")
    if any(type(item) is not int for item in parts):
        raise FreeTextRecallRuntimeError("stable key parts 非法")
    raw = ":".join(str(item) for item in (domain, *parts)).encode("ascii")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((domain, value if value else 1))


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """向纯整数 payload 写入一个长度前缀键。"""
    result.extend((len(value), *value))


class _Cursor:
    """对嵌套稳定键做有界纯整数读取。"""

    def __init__(self, values: tuple[int, ...]) -> None:
        if not isinstance(values, tuple) or any(type(item) is not int for item in values):
            raise FreeTextRecallRuntimeError("integer payload 类型非法")
        self.values = values
        self.index = 0

    def take(self, *, where: str) -> int:
        """读取一个整数并拒绝截断。"""
        if self.index >= len(self.values):
            raise FreeTextRecallRuntimeError(f"{where} 被截断")
        value = self.values[self.index]
        self.index += 1
        return value

    def packed(self, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
        """读取一个长度前缀整数键并推进游标。"""
        size = self.take(where=f"{where}.size")
        if size < 0 or (size == 0 and not allow_empty):
            raise FreeTextRecallRuntimeError(f"{where} 长度非法")
        end = self.index + size
        if end > len(self.values):
            raise FreeTextRecallRuntimeError(f"{where} 被截断")
        value = self.values[self.index:end]
        self.index = end
        return value

    def finish(self, *, where: str) -> None:
        """要求解码恰好消费全部整数。"""
        if self.index != len(self.values):
            raise FreeTextRecallRuntimeError(f"{where} 含尾随整数")


def encode_surface_feature_payload(
        surface: str, feature_key: StableRecordKey,
        ) -> tuple[int, ...]:
    """把学习所得 surface→feature 关联编码进 Evidence payload。"""
    if not isinstance(surface, str) or not surface or surface.strip() != surface:
        raise FreeTextRecallRuntimeError("feature surface 必须是规范非空文本")
    if not isinstance(feature_key, StableRecordKey):
        raise TypeError("feature key 必须是 StableRecordKey")
    encoded = surface.encode("utf-8")
    return (
        FEATURE_PAYLOAD_VERSION,
        len(encoded),
        *(value + 1 for value in encoded),
        len(feature_key.components),
        *feature_key.components,
    )


def decode_surface_feature_payload(
        payload: tuple[int, ...],
        ) -> tuple[str, StableRecordKey]:
    """从真实 Evidence payload 恢复 surface 与离散 feature 身份。"""
    cursor = _Cursor(payload)
    if cursor.take(where="feature version") != FEATURE_PAYLOAD_VERSION:
        raise FreeTextRecallRuntimeError("feature payload version 非法")
    size = cursor.take(where="feature surface size")
    if size <= 0:
        raise FreeTextRecallRuntimeError("feature surface 为空")
    raw = tuple(cursor.take(where="feature surface") for _ in range(size))
    if any(value <= 0 or value > 256 for value in raw):
        raise FreeTextRecallRuntimeError("feature surface 字节非法")
    try:
        surface = bytes(value - 1 for value in raw).decode("utf-8")
    except UnicodeDecodeError as error:
        raise FreeTextRecallRuntimeError("feature surface UTF-8 损坏") from error
    feature_key = StableRecordKey(cursor.packed(where="feature key"))
    cursor.finish(where="feature payload")
    if encode_surface_feature_payload(surface, feature_key) != payload:
        raise FreeTextRecallRuntimeError("feature payload 非规范 round-trip")
    return surface, feature_key


@dataclass(frozen=True)
class LearnedFeatureMatch:
    """当前 raw query 命中的一个历史 Evidence 与离散 feature。"""

    evidence_id: int
    feature_key: StableRecordKey
    start: int
    end: int

    def __post_init__(self) -> None:
        """核验 Evidence 身份和 query 半开范围。"""
        if type(self.evidence_id) is not int or self.evidence_id <= 0:
            raise FreeTextRecallRuntimeError("feature Evidence id 非法")
        if not isinstance(self.feature_key, StableRecordKey):
            raise TypeError("feature match key 类型错误")
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.start >= self.end):
            raise FreeTextRecallRuntimeError("feature match span 非法")


class LearnedSurfaceFeatureMatcher:
    """只解释注入 Evidence 格式，不在 Python 内写死任何词义或同义表。"""

    def __init__(self, feature_reason_key: tuple[int, ...]) -> None:
        """绑定课程/学习系统注入的一等 Evidence reason key。"""
        self.feature_reason_key = _strict_key(
            feature_reason_key, where="feature reason key")

    def match(
            self,
            raw_query: str,
            evidence: tuple[EvidenceRecord, ...],
            ) -> tuple[LearnedFeatureMatch, ...]:
        """在 raw query 中定位已学 surface，并返回 Evidence/feature 身份。"""
        if not isinstance(raw_query, str) or not raw_query:
            raise FreeTextRecallRuntimeError("raw query 必须非空")
        if (not isinstance(evidence, tuple)
                or any(not isinstance(item, EvidenceRecord) for item in evidence)):
            raise TypeError("surface matcher 需要 EvidenceRecord tuple")
        matches = []
        for item in evidence:
            if item.reason_key != self.feature_reason_key:
                continue
            surface, feature_key = decode_surface_feature_payload(item.payload)
            start = raw_query.find(surface)
            while start >= 0:
                matches.append(LearnedFeatureMatch(
                    item.evidence_id, feature_key, start, start + len(surface)))
                start = raw_query.find(surface, start + 1)
        return tuple(sorted(set(matches), key=lambda item: (
            item.feature_key, item.start, item.end, item.evidence_id)))


@dataclass(frozen=True, order=True)
class RecallIndexEntry:
    """不含 payload 的长期索引行；只携带来源边界和 feature Evidence 身份。"""

    record_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    required_feature_keys: tuple[StableRecordKey, ...]
    dependency_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        """核验 index 不跨 source/scope/owner/version 且 feature 非空。"""
        _strict_key(self.record_key, where="recall index record key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("recall index source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("recall index scope 类型错误")
        if (self.scope.source != self.source
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise FreeTextRecallRuntimeError("recall index source/scope 漂移")
        for name in ("required_feature_keys", "dependency_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(item, StableRecordKey) for item in values)
                    or values != tuple(sorted(set(values)))):
                raise FreeTextRecallRuntimeError(f"recall index {name} 非法")


@dataclass(frozen=True, order=True)
class EvidenceFormedCenter:
    """共享 feature Evidence 形成的 record center 与真实 MD-03 envelope。"""

    center_key: StableRecordKey
    index_entry: RecallIndexEntry
    evidence_ids: tuple[int, ...]
    md03_center: DirectionalMemoryCenter

    def __post_init__(self) -> None:
        """核验 center 仍是 activation-only 且证据身份稳定。"""
        if not isinstance(self.center_key, StableRecordKey):
            raise TypeError("formed center key 类型错误")
        if not isinstance(self.index_entry, RecallIndexEntry):
            raise TypeError("formed center index entry 类型错误")
        if (not isinstance(self.evidence_ids, tuple) or not self.evidence_ids
                or any(type(item) is not int or item <= 0
                       for item in self.evidence_ids)
                or self.evidence_ids != tuple(sorted(set(self.evidence_ids)))):
            raise FreeTextRecallRuntimeError("formed center Evidence ids 非法")
        if not isinstance(self.md03_center, DirectionalMemoryCenter):
            raise TypeError("formed center 缺少 MD-03 center")
        if self.md03_center.center.activation_only != 1:
            raise FreeTextRecallRuntimeError("MD-03 center 不得授权 adoption")


class LearnedEvidenceCenterFormer:
    """按共享离散 feature 形成 center，并委托 MD-03 建立真实方向 envelope。"""

    def __init__(
            self,
            matcher: LearnedSurfaceFeatureMatcher,
            md03: DirectionalMemoryCenterAdapter,
            ) -> None:
        """绑定 Evidence matcher 和真实 MD-03 adapter。"""
        if not isinstance(matcher, LearnedSurfaceFeatureMatcher):
            raise TypeError("center former matcher 类型错误")
        if not isinstance(md03, DirectionalMemoryCenterAdapter):
            raise TypeError("center former MD-03 adapter 类型错误")
        self.matcher = matcher
        self.md03 = md03

    def form(
            self,
            raw_query: str,
            history: tuple[EvidenceRecord, ...],
            current: MemoryCurrentQuery,
            index: tuple[RecallIndexEntry, ...],
            ) -> tuple[EvidenceFormedCenter, ...]:
        """从当前 raw query 和允许历史 Evidence 形成零或多个 activation center。"""
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("center former current 类型错误")
        if (not isinstance(index, tuple)
                or any(not isinstance(item, RecallIndexEntry) for item in index)):
            raise TypeError("center former index 类型错误")
        if not current.spans:
            raise FreeTextRecallRuntimeError("MD-03 center 需要当前 query Span anchor")
        matches = self.matcher.match(raw_query, history)
        by_feature: dict[StableRecordKey, set[int]] = {}
        for match in matches:
            by_feature.setdefault(match.feature_key, set()).add(match.evidence_id)
        centers = []
        anchor: TypedRef | ObjectIdentity = sorted(
            current.spans, key=lambda item: item.stable_key())[0]
        for entry in sorted(index):
            if any(feature not in by_feature
                   for feature in entry.required_feature_keys):
                continue
            evidence_ids = tuple(sorted({
                evidence_id
                for feature in entry.required_feature_keys
                for evidence_id in by_feature[feature]
            }))
            md03_center = self.md03.from_understanding(
                current, anchor, strength="CONDITIONAL")
            center_key = _stable(
                9201,
                *entry.record_key,
                *evidence_ids,
                *md03_center.center.center_key.components,
            )
            centers.append(EvidenceFormedCenter(
                center_key, entry, evidence_ids, md03_center))
        return tuple(sorted(centers))


def _decode_bound_binding(cursor: _Cursor) -> BoundRoleBinding:
    """递归解码一个 BoundRoleBinding 稳定键。"""
    role = ObjectIdentity.from_stable_key(cursor.packed(where="bound role"))
    ordinal = cursor.take(where="bound role ordinal")
    tag = cursor.take(where="bound filler tag")
    filler_key = cursor.packed(where="bound filler")
    if tag == 1:
        filler: ObjectIdentity | BoundProposition = ObjectIdentity.from_stable_key(
            filler_key)
    elif tag == 2:
        filler = _decode_bound_proposition(filler_key)
    else:
        raise FreeTextRecallRuntimeError("bound filler tag 非法")
    return BoundRoleBinding(role, filler, ordinal)


def _decode_bound_proposition(key: tuple[int, ...]) -> BoundProposition:
    """从完整运行期稳定键恢复 BoundProposition，不从文本猜语义。"""
    cursor = _Cursor(key)
    version = cursor.take(where="bound proposition version")
    if version != 1:
        raise FreeTextRecallRuntimeError("bound proposition version 非法")
    identities = tuple(ObjectIdentity.from_stable_key(
        cursor.packed(where="bound identity")) for _ in range(6))
    binder_count = cursor.take(where="bound binder count")
    if binder_count < 0:
        raise FreeTextRecallRuntimeError("bound binder count 非法")
    binders = tuple(ObjectIdentity.from_stable_key(
        cursor.packed(where="bound binder")) for _ in range(binder_count))
    binding_count = cursor.take(where="bound binding count")
    if binding_count < 0:
        raise FreeTextRecallRuntimeError("bound binding count 非法")
    bindings = tuple(_decode_bound_binding(cursor) for _ in range(binding_count))
    variable_count = cursor.take(where="bound variable count")
    if variable_count < 0:
        raise FreeTextRecallRuntimeError("bound variable count 非法")
    variables = tuple(ObjectIdentity.from_stable_key(
        cursor.packed(where="bound variable")) for _ in range(variable_count))
    cursor.finish(where="bound proposition")
    proposition = BoundProposition(
        identities[0], identities[1], identities[2], identities[3],
        identities[4], identities[5], binders, bindings, variables)
    if proposition.stable_key() != key:
        raise FreeTextRecallRuntimeError("BoundProposition 非规范 round-trip")
    return proposition


@dataclass(frozen=True)
class TypedRecallPayload:
    """K-04 冷记录中的真实命题、Evidence、精确引用和依赖身份。"""

    proposition: BoundProposition
    evidence: tuple[EvidenceRecord, ...]
    citation_start: int
    citation_end: int
    dependency_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        """核验 payload 来源一致、引用范围非空且 Evidence 可供 QA 消费。"""
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("recall payload proposition 类型错误")
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or any(not isinstance(item, EvidenceRecord) for item in self.evidence)):
            raise TypeError("recall payload Evidence 类型错误")
        source = semantic_source(self.proposition.template)
        if any(item.hypothesis.observation != source for item in self.evidence):
            raise FreeTextRecallRuntimeError("recall payload Evidence 跨命题来源")
        if (type(self.citation_start) is not int
                or type(self.citation_end) is not int
                or self.citation_start < 0
                or self.citation_start >= self.citation_end):
            raise FreeTextRecallRuntimeError("recall payload citation range 非法")
        if (not isinstance(self.dependency_keys, tuple)
                or not self.dependency_keys
                or any(not isinstance(item, StableRecordKey)
                       for item in self.dependency_keys)
                or self.dependency_keys != tuple(sorted(set(self.dependency_keys)))):
            raise FreeTextRecallRuntimeError("recall payload dependencies 非法")


class TypedRecallRecordCodec:
    """把 typed recall payload 编入 K-02/K-04 可封存纯整数记录。"""

    @staticmethod
    def encode(
            record_key: tuple[int, ...], payload: TypedRecallPayload,
            ) -> SegmentRecord:
        """编码命题、全部 Evidence、citation 和 dependency，不丢身份。"""
        _strict_key(record_key, where="typed recall record key")
        if not isinstance(payload, TypedRecallPayload):
            raise TypeError("typed recall payload 类型错误")
        values = [RECALL_PAYLOAD_VERSION]
        _pack(values, payload.proposition.stable_key())
        values.append(len(payload.evidence))
        for item in payload.evidence:
            _pack(values, item.stable_key())
        values.extend((payload.citation_start, payload.citation_end))
        values.append(len(payload.dependency_keys))
        for item in payload.dependency_keys:
            _pack(values, item.components)
        return SegmentRecord(record_key, tuple(values))

    @staticmethod
    def decode(record: SegmentRecord) -> TypedRecallPayload:
        """从 K-04 record 恢复 typed payload 并要求逐整数规范回环。"""
        if not isinstance(record, SegmentRecord):
            raise TypeError("typed recall codec 需要 SegmentRecord")
        cursor = _Cursor(record.payload)
        if cursor.take(where="recall payload version") != RECALL_PAYLOAD_VERSION:
            raise FreeTextRecallRuntimeError("recall payload version 非法")
        proposition = _decode_bound_proposition(
            cursor.packed(where="recall proposition"))
        evidence_count = cursor.take(where="recall Evidence count")
        if evidence_count <= 0:
            raise FreeTextRecallRuntimeError("recall Evidence 不能为空")
        evidence = tuple(EvidenceRecord.from_stable_key(
            cursor.packed(where="recall Evidence")) for _ in range(evidence_count))
        citation_start = cursor.take(where="recall citation start")
        citation_end = cursor.take(where="recall citation end")
        dependency_count = cursor.take(where="recall dependency count")
        if dependency_count <= 0:
            raise FreeTextRecallRuntimeError("recall dependencies 不能为空")
        dependencies = tuple(sorted(StableRecordKey(
            cursor.packed(where="recall dependency"))
            for _ in range(dependency_count)))
        cursor.finish(where="recall payload")
        payload = TypedRecallPayload(
            proposition, evidence, citation_start, citation_end, dependencies)
        if TypedRecallRecordCodec.encode(record.record_key, payload) != record:
            raise FreeTextRecallRuntimeError("recall record 非规范 round-trip")
        return payload


def _zero_metrics() -> QueryHotSetMetrics:
    """返回 ACL 拒绝、歧义或无匹配时的物理零读取计数。"""
    return QueryHotSetMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


@dataclass(frozen=True)
class ExactRecallResult:
    """ACL 检查、K-04 指标、合同收据和可选 typed payload。"""

    obligation: RecallObligation
    receipt: RecallReceipt
    metrics: QueryHotSetMetrics
    payload: TypedRecallPayload | None
    acl_checked_before_payload: int
    private_label_read_count: int

    def __post_init__(self) -> None:
        """核验授权顺序、物理指标和 payload/停止状态闭合。"""
        if not isinstance(self.obligation, RecallObligation):
            raise TypeError("exact recall obligation 类型错误")
        if not isinstance(self.receipt, RecallReceipt):
            raise TypeError("exact recall receipt 类型错误")
        if not isinstance(self.metrics, QueryHotSetMetrics):
            raise TypeError("exact recall metrics 类型错误")
        if self.receipt.obligation_key != self.obligation.obligation_key:
            raise FreeTextRecallRuntimeError("recall receipt 替换了 obligation")
        if self.acl_checked_before_payload != 1:
            raise FreeTextRecallRuntimeError("ACL 必须在 payload 前检查")
        if self.private_label_read_count != 0:
            raise FreeTextRecallRuntimeError("recall runtime 不得读取私有标签")
        if self.payload is None and self.receipt.stop_reason == "RESOLVED":
            raise FreeTextRecallRuntimeError("RESOLVED recall 缺 typed payload")
        if self.payload is not None and self.receipt.stop_reason != "RESOLVED":
            raise FreeTextRecallRuntimeError("非 RESOLVED recall 不得携带 payload")


class AclFirstExactRecallReader:
    """先检查 production ACL，再用 QuerySegmentHotSet exact page-in。"""

    def __init__(
            self,
            store: TieredSegmentStore,
            descriptor_key: tuple[int, ...],
            policy: QueryHotSetPolicy,
            codec: TypedRecallRecordCodec | None = None,
            ) -> None:
        """绑定真实 K-04 store/descriptor/policy 和 typed 冷记录 codec。"""
        if not isinstance(store, TieredSegmentStore):
            raise TypeError("exact recall store 类型错误")
        self.store = store
        self.descriptor_key = _strict_key(
            descriptor_key, where="exact recall descriptor key")
        if not isinstance(policy, QueryHotSetPolicy):
            raise TypeError("exact recall policy 类型错误")
        self.policy = policy
        self.codec = codec or TypedRecallRecordCodec()

    @staticmethod
    def _obligation(
            center: EvidenceFormedCenter,
            current: MemoryCurrentQuery,
            budget: RecallBudget,
            *,
            failure_state: str,
            ) -> RecallObligation:
        """从 center/current/source boundary 建立 T0 合同兼容 obligation。"""
        entry = center.index_entry
        query_key = _stable(9251, *current.stable_key())
        return RecallObligation(
            _stable(9252, *center.center_key.components, *query_key.components),
            center.center_key,
            query_key,
            "PROPOSITION",
            StableRecordKey(entry.record_key),
            entry.source,
            _stable(9253, *entry.scope.stable_key()),
            _stable(9254, *entry.source.owner.stable_key()),
            budget,
            failure_state,
        )

    def read(
            self,
            center: EvidenceFormedCenter,
            current: MemoryCurrentQuery,
            access: MemoryAccessContext,
            budget: RecallBudget,
            *,
            reader_key: tuple[int, ...],
            ) -> ExactRecallResult:
        """执行 ACL-first exact range；越权路径在 hot set 构造前返回零读取。"""
        if not isinstance(center, EvidenceFormedCenter):
            raise TypeError("exact recall center 类型错误")
        if not isinstance(current, MemoryCurrentQuery):
            raise TypeError("exact recall current 类型错误")
        if not isinstance(access, MemoryAccessContext):
            raise TypeError("exact recall access 类型错误")
        if not isinstance(budget, RecallBudget):
            raise TypeError("exact recall budget 类型错误")
        _strict_key(reader_key, where="exact recall reader key")
        entry = center.index_entry
        if not access.can_read(entry.source.owner):
            obligation = self._obligation(
                center, current, budget, failure_state="UNAUTHORIZED")
            receipt = RecallReceipt(
                obligation.obligation_key,
                (StableRecordKey(self.descriptor_key),),
                (),
                1,
                0,
                0,
                (),
                (),
                "UNAUTHORIZED",
                0,
            )
            return ExactRecallResult(
                obligation, receipt, _zero_metrics(), None, 1, 0)
        obligation = self._obligation(center, current, budget, failure_state="NONE")
        hot_set = QuerySegmentHotSet(
            self.store,
            reader_key=reader_key,
            descriptor_key=self.descriptor_key,
            policy=self.policy,
        )
        records = []
        exhausted = False
        try:
            iterator = hot_set.iter_range(
                lower_key=entry.record_key,
                upper_key=entry.record_key,
            )
            try:
                for cached in iterator:
                    records.append(cached.record)
                    if len(records) >= budget.max_results:
                        break
            finally:
                close_iterator = getattr(iterator, "close", None)
                if callable(close_iterator):
                    close_iterator()
        except QueryHotSetBudgetExceeded:
            exhausted = True
        finally:
            hot_set.close()
        metrics = hot_set.metrics()
        if (metrics.page_faults > budget.max_segment_payload_gets
                or metrics.cold_read_bytes > budget.max_segment_payload_bytes):
            exhausted = True
        if exhausted:
            receipt = RecallReceipt(
                obligation.obligation_key,
                (StableRecordKey(self.descriptor_key),),
                (),
                1,
                min(metrics.page_faults, budget.max_segment_payload_gets),
                min(metrics.cold_read_bytes, budget.max_segment_payload_bytes),
                (),
                (),
                "BUDGET_EXHAUSTED",
                0,
            )
            return ExactRecallResult(obligation, receipt, metrics, None, 1, 0)
        if not records:
            receipt = RecallReceipt(
                obligation.obligation_key,
                (StableRecordKey(self.descriptor_key),),
                (),
                1,
                metrics.page_faults,
                metrics.cold_read_bytes,
                (),
                (),
                "NO_MATCH",
                0,
            )
            return ExactRecallResult(obligation, receipt, metrics, None, 1, 0)
        if len(records) != 1 or records[0].record_key != entry.record_key:
            raise FreeTextRecallRuntimeError("exact K-04 range 返回非唯一目标")
        payload = self.codec.decode(records[0])
        source = semantic_source(payload.proposition.template)
        if source != entry.source:
            raise FreeTextRecallRuntimeError("cold proposition 与 index SourceRef 漂移")
        if payload.dependency_keys != entry.dependency_keys:
            raise FreeTextRecallRuntimeError("cold payload 与 index dependency 漂移")
        span = AbsoluteSpan(
            _stable(9255, *entry.record_key, payload.citation_start,
                    payload.citation_end),
            source,
            payload.citation_start,
            payload.citation_end,
        )
        citation = RecallCitation(
            _stable(9256, *entry.record_key),
            StableRecordKey(entry.record_key),
            source,
            span,
        )
        receipt = RecallReceipt(
            obligation.obligation_key,
            (StableRecordKey(self.descriptor_key),),
            (StableRecordKey(self.descriptor_key),),
            1,
            metrics.page_faults,
            metrics.cold_read_bytes,
            (StableRecordKey(entry.record_key),),
            (citation,),
            "RESOLVED",
            0,
        )
        return ExactRecallResult(obligation, receipt, metrics, payload, 1, 0)


@dataclass(frozen=True)
class FreeTextRecallRun:
    """feature、center、停止状态和可选 exact read 的完整零写报告。"""

    matched_features: tuple[LearnedFeatureMatch, ...]
    centers: tuple[EvidenceFormedCenter, ...]
    rejected_unauthorized_center_keys: tuple[StableRecordKey, ...]
    selected_center_key: StableRecordKey | None
    stop_reason: str
    exact_read: ExactRecallResult | None
    host_learning_write_count: int
    private_label_read_count: int

    def __post_init__(self) -> None:
        """核验歧义/unknown 不读 payload，resolved 恰有一个已选 center。"""
        if (not isinstance(self.matched_features, tuple)
                or any(not isinstance(item, LearnedFeatureMatch)
                       for item in self.matched_features)):
            raise TypeError("recall run matched features 类型错误")
        if (not isinstance(self.centers, tuple)
                or any(not isinstance(item, EvidenceFormedCenter)
                       for item in self.centers)):
            raise TypeError("recall run centers 类型错误")
        if (not isinstance(self.rejected_unauthorized_center_keys, tuple)
                or any(not isinstance(item, StableRecordKey)
                       for item in self.rejected_unauthorized_center_keys)
                or self.rejected_unauthorized_center_keys
                != tuple(sorted(set(self.rejected_unauthorized_center_keys)))):
            raise TypeError("recall run rejected center keys 类型错误")
        if self.stop_reason not in {
                "BUDGET_EXHAUSTED", "CLARIFY", "NO_MATCH", "RESOLVED",
                "UNAUTHORIZED", "UNKNOWN"}:
            raise FreeTextRecallRuntimeError("recall run stop reason 非法")
        if self.host_learning_write_count != 0 or self.private_label_read_count != 0:
            raise FreeTextRecallRuntimeError("recall run 不得产生 host 写或读私有标签")
        if self.stop_reason in {"CLARIFY", "UNKNOWN"} and self.exact_read is not None:
            raise FreeTextRecallRuntimeError("歧义/unknown 不得读取 payload")
        if self.stop_reason == "RESOLVED":
            if (self.selected_center_key is None or self.exact_read is None
                    or self.exact_read.payload is None):
                raise FreeTextRecallRuntimeError("resolved run 不完整")


class FreeTextRecallRuntime:
    """组合 Evidence center 和 ACL-first reader，不拥有 teacher/evaluator 数据。"""

    def __init__(
            self,
            center_former: LearnedEvidenceCenterFormer,
            reader: AclFirstExactRecallReader,
            ) -> None:
        """绑定可独立消融的 center former 与 exact reader。"""
        if not isinstance(center_former, LearnedEvidenceCenterFormer):
            raise TypeError("free text runtime center former 类型错误")
        if not isinstance(reader, AclFirstExactRecallReader):
            raise TypeError("free text runtime reader 类型错误")
        self.center_former = center_former
        self.reader = reader

    def resolve(
            self,
            raw_query: str,
            history: tuple[EvidenceRecord, ...],
            current: MemoryCurrentQuery,
            index: tuple[RecallIndexEntry, ...],
            access: MemoryAccessContext,
            budget: RecallBudget,
            *,
            reader_key: tuple[int, ...],
            ) -> FreeTextRecallRun:
        """形成 center；零中心返回 unknown，多中心 clarify，唯一中心才读取 K-04。"""
        matches = self.center_former.matcher.match(raw_query, history)
        formed = self.center_former.form(raw_query, history, current, index)
        if not formed:
            return FreeTextRecallRun(
                matches, (), (), None, "UNKNOWN", None, 0, 0)
        centers = tuple(
            item for item in formed if access.can_read(item.index_entry.source.owner))
        rejected = tuple(sorted(
            item.center_key for item in formed if item not in centers))
        if not centers:
            center = formed[0]
            exact = self.reader.read(
                center, current, access, budget, reader_key=reader_key)
            return FreeTextRecallRun(
                matches, (), rejected, center.center_key,
                exact.receipt.stop_reason, exact, 0, 0)
        if len(centers) != 1:
            return FreeTextRecallRun(
                matches, centers, rejected, None, "CLARIFY", None, 0, 0)
        center = centers[0]
        exact = self.reader.read(
            center, current, access, budget, reader_key=reader_key)
        return FreeTextRecallRun(
            matches,
            centers,
            rejected,
            center.center_key,
            exact.receipt.stop_reason,
            exact,
            0,
            0,
        )


class RecalledFactQuestionExecutor:
    """把一次成功 exact recall 薄投影为现有 QA 的真实 GenerationCandidate。"""

    def __init__(
            self,
            recall: FreeTextRecallRun,
            *,
            route: ObjectIdentity,
            executed_reason: ObjectIdentity,
            trace_prefix: tuple[int, ...],
            ) -> None:
        """绑定成功 recall、开放 route/reason 和 caller trace。"""
        if (not isinstance(recall, FreeTextRecallRun)
                or recall.stop_reason != "RESOLVED"
                or recall.exact_read is None
                or recall.exact_read.payload is None):
            raise FreeTextRecallRuntimeError("QA executor 只接受 resolved recall")
        if not isinstance(route, ObjectIdentity):
            raise TypeError("recalled QA route 类型错误")
        if not isinstance(executed_reason, ObjectIdentity):
            raise TypeError("recalled QA reason 类型错误")
        self.recall = recall
        self.route = route
        self.executed_reason = executed_reason
        self.trace_prefix = _strict_key(trace_prefix, where="recalled QA trace")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """只消费同次 recall payload，形成 QA/G-00 候选且不重复 page-in。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("recalled QA executor 需要 QuestionQuery")
        if query.route != self.route:
            raise FreeTextRecallRuntimeError("recalled QA route 漂移")
        assert self.recall.exact_read is not None
        payload = self.recall.exact_read.payload
        assert payload is not None
        request = query.request
        if request.target != payload.proposition:
            raise FreeTextRecallRuntimeError("QA target 与 recall proposition 漂移")
        source = semantic_source(payload.proposition.template)
        if request.source != source:
            raise FreeTextRecallRuntimeError("QA source 与 recall source 漂移")
        state = LogicEvidenceState(
            any(item.stance == EVIDENCE_SUPPORT for item in payload.evidence),
            any(item.stance == EVIDENCE_REFUTE for item in payload.evidence),
        )
        candidate = GenerationCandidate(
            payload.proposition,
            state,
            source,
            request.response_scope,
            payload.evidence,
        )
        trace = (
            *self.trace_prefix,
            *self.recall.selected_center_key.components,
            self.recall.exact_read.metrics.page_faults,
            self.recall.exact_read.metrics.page_in_records,
            self.recall.exact_read.metrics.cold_read_bytes,
        )
        return QuestionExecutionResult(
            query, self.executed_reason, (candidate,), trace)


__all__ = [
    "AclFirstExactRecallReader",
    "EvidenceFormedCenter",
    "ExactRecallResult",
    "FreeTextRecallRun",
    "FreeTextRecallRuntime",
    "FreeTextRecallRuntimeError",
    "LearnedEvidenceCenterFormer",
    "LearnedFeatureMatch",
    "LearnedSurfaceFeatureMatcher",
    "RecallIndexEntry",
    "RecalledFactQuestionExecutor",
    "TypedRecallPayload",
    "TypedRecallRecordCodec",
    "decode_surface_feature_payload",
    "encode_surface_feature_payload",
]
