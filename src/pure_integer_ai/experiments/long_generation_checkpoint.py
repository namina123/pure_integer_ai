"""授权逐页生成的 append-only checkpoint、cursor 和 prefix digest。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterState,
)
from pure_integer_ai.experiments.authorized_generation_delivery import (
    DELIVERY_AUTHORIZED,
    AuthorizedGenerationClaim,
    AuthorizedGenerationDeliveryDecision,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    IntegerStreamReader,
    decode_integer_tuple,
    encode_integer_tuple,
    pack_key,
)
from pure_integer_ai.storage.segment_repository import (
    AppendOnlyObjectRepository,
)


OBJECT_KIND_LONG_GENERATION_CHECKPOINT = 7
LONG_GENERATION_OPEN = 1
LONG_GENERATION_BUDGET_STOP = 2
LONG_GENERATION_COMPLETE = 3
_STATUSES = frozenset({
    LONG_GENERATION_OPEN,
    LONG_GENERATION_BUDGET_STOP,
    LONG_GENERATION_COMPLETE,
})
_FORMAT_VERSION = 1
_DIGEST_SIZE = 32


class LongGenerationCheckpointError(RuntimeError):
    """plan、cursor、prefix、授权 page-in 或 checkpoint revision 不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """按统一 integer codec 分帧追加稳定键。"""
    pack_key(result, value)


def _strict_key(
        value: tuple[int, ...], *, where: str, positive: bool = False,
        empty: bool = False,
        ) -> tuple[int, ...]:
    """核验调用方整数键，并按用途限制正数或空值。"""
    if not isinstance(value, tuple) or (not value and not empty):
        raise LongGenerationCheckpointError(f"{where} 必须是整数 tuple")
    for item in value:
        if type(item) is not int or item < 0 or (positive and item <= 0):
            raise LongGenerationCheckpointError(f"{where} 整数值非法")
    return value


def _digest_key(
        value: tuple[int, ...], *, empty: bool = False,
        ) -> tuple[int, ...]:
    """核验空初始链指针或完整 SHA-256 字节 tuple。"""
    if empty and value == ():
        return value
    if (not isinstance(value, tuple) or len(value) != _DIGEST_SIZE
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in value)):
        raise LongGenerationCheckpointError("checkpoint digest 非法")
    return value


def _digest(values: tuple[int, ...]) -> tuple[int, ...]:
    """对规范整数流计算 SHA-256。"""
    return tuple(hashlib.sha256(encode_integer_tuple(values)).digest())


def _identity_sort_key(value: ObjectIdentity) -> tuple[int, ...]:
    """返回一等对象的完整稳定排序键。"""
    return value.stable_key()


@dataclass(frozen=True, order=True)
class LongGenerationPlanItem:
    """长回答 cursor 上一个唯一 item 与预期 Proposition identity。"""

    item_key: StableRecordKey
    proposition_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 item key 和允许合法零位的 Proposition stable key。"""
        if not isinstance(self.item_key, StableRecordKey):
            raise TypeError("long generation item_key 类型错误")
        _strict_key(
            self.proposition_key,
            where="long generation proposition_key",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 item 和预期命题完整键。"""
        return (
            len(self.item_key.components),
            *self.item_key.components,
            len(self.proposition_key),
            *self.proposition_key,
        )


@dataclass(frozen=True)
class LongGenerationPlan:
    """由调用方冻结、无固定句数或 renderer 语义的 continuation plan。"""

    answer_key: StableRecordKey
    items: tuple[LongGenerationPlanItem, ...]

    def __post_init__(self) -> None:
        """核验 answer identity、非空有序 items 和 item key 唯一性。"""
        if not isinstance(self.answer_key, StableRecordKey):
            raise TypeError("long generation answer_key 类型错误")
        if (not isinstance(self.items, tuple) or not self.items
                or any(not isinstance(item, LongGenerationPlanItem)
                       for item in self.items)):
            raise LongGenerationCheckpointError("long generation plan items 非法")
        keys = tuple(item.item_key for item in self.items)
        if len(keys) != len(set(keys)):
            raise LongGenerationCheckpointError("long generation item key 重复")

    def stable_key(self) -> tuple[int, ...]:
        """返回 answer 和全部有序计划 item 的完整键。"""
        result = [
            _FORMAT_VERSION,
            len(self.answer_key.components),
            *self.answer_key.components,
            len(self.items),
        ]
        for item in self.items:
            _pack(result, item.stable_key())
        return tuple(result)

    def digest(self) -> tuple[int, ...]:
        """返回冻结计划摘要；摘要不替代 commit 时的完整 plan 输入。"""
        return _digest(self.stable_key())


@dataclass(frozen=True)
class LongGenerationCheckpoint:
    """跨进程 continuation 所需的小型元数据 snapshot。"""

    answer_key: StableRecordKey
    revision: int
    previous_digest: tuple[int, ...]
    plan_digest: tuple[int, ...]
    next_cursor: int
    prefix_digest: tuple[int, ...]
    current_section_key: StableRecordKey | None
    current_paragraph_key: StableRecordKey | None
    available_antecedents: tuple[ObjectIdentity, ...]
    committed_page_keys: tuple[StableRecordKey, ...]
    citation_record_keys: tuple[StableRecordKey, ...]
    authorization_receipt_keys: tuple[StableRecordKey, ...]
    status: int

    def __post_init__(self) -> None:
        """核验 revision/digest 链、cursor、上下文和 append-only 收据元数据。"""
        if not isinstance(self.answer_key, StableRecordKey):
            raise TypeError("checkpoint answer_key 类型错误")
        if type(self.revision) is not int or self.revision < 0:
            raise LongGenerationCheckpointError("checkpoint revision 非法")
        _digest_key(self.previous_digest, empty=self.revision == 0)
        if (self.revision == 0) != (self.previous_digest == ()):
            raise LongGenerationCheckpointError("checkpoint previous digest 链断裂")
        _digest_key(self.plan_digest)
        if type(self.next_cursor) is not int or self.next_cursor < 0:
            raise LongGenerationCheckpointError("checkpoint cursor 非法")
        _digest_key(self.prefix_digest)
        if (self.current_section_key is None) != (
                self.current_paragraph_key is None):
            raise LongGenerationCheckpointError("section/paragraph 必须同时存在或缺失")
        for value, where in (
                (self.current_section_key, "checkpoint section"),
                (self.current_paragraph_key, "checkpoint paragraph")):
            if value is not None and not isinstance(value, StableRecordKey):
                raise TypeError(f"{where} 类型错误")
        if (not isinstance(self.available_antecedents, tuple)
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.available_antecedents)):
            raise TypeError("checkpoint antecedents 类型错误")
        antecedents = tuple(sorted(
            set(self.available_antecedents), key=_identity_sort_key))
        if antecedents != self.available_antecedents:
            raise LongGenerationCheckpointError(
                "checkpoint antecedents 必须排序去重")
        for name in (
                "committed_page_keys", "citation_record_keys",
                "authorization_receipt_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, StableRecordKey)
                           for item in values)):
                raise TypeError(f"checkpoint {name} 类型错误")
        if len(set(self.committed_page_keys)) != len(self.committed_page_keys):
            raise LongGenerationCheckpointError("checkpoint page key 重复")
        if self.status not in _STATUSES:
            raise LongGenerationCheckpointError("checkpoint status 未注册")
        if self.revision == 0 and any((
                self.next_cursor,
                self.current_section_key is not None,
                len(self.available_antecedents),
                len(self.committed_page_keys),
                len(self.citation_record_keys),
                len(self.authorization_receipt_keys),
        )):
            raise LongGenerationCheckpointError("initial checkpoint 必须为空前缀")

    def stable_key(self) -> tuple[int, ...]:
        """返回 plan/cursor/prefix/context/receipt 的完整 snapshot key。"""
        result = [
            _FORMAT_VERSION,
            len(self.answer_key.components),
            *self.answer_key.components,
            self.revision,
        ]
        for value in (
                self.previous_digest,
                self.plan_digest,
                self.prefix_digest):
            _pack(result, value)
        result.append(self.next_cursor)
        for value in (self.current_section_key, self.current_paragraph_key):
            result.append(int(value is not None))
            if value is not None:
                _pack(result, value.components)
        result.append(len(self.available_antecedents))
        for item in self.available_antecedents:
            _pack(result, item.stable_key())
        for values in (
                self.committed_page_keys,
                self.citation_record_keys,
                self.authorization_receipt_keys):
            result.append(len(values))
            for item in values:
                _pack(result, item.components)
        result.append(self.status)
        return tuple(result)

    def digest(self) -> tuple[int, ...]:
        """返回当前完整 checkpoint 摘要，绑定下一 revision。"""
        return _digest(self.stable_key())


@dataclass(frozen=True)
class LongGenerationPageBudget:
    """一个局部 generation 页的 renderer unit 和 claim 硬预算。"""

    max_units: int
    max_claims: int

    def __post_init__(self) -> None:
        """预算必须为严格正整数。"""
        if (type(self.max_units) is not int or self.max_units <= 0
                or type(self.max_claims) is not int or self.max_claims <= 0):
            raise LongGenerationCheckpointError("generation page budget 非法")


@dataclass(frozen=True)
class LongGenerationPageCommit:
    """一次局部完整生成页的计划 slice、授权 page-in 和 continuation 预期。"""

    page_key: StableRecordKey
    plan_item_keys: tuple[StableRecordKey, ...]
    section_key: StableRecordKey
    paragraph_key: StableRecordKey
    expected_revision: int
    expected_cursor: int
    expected_prefix_digest: tuple[int, ...]
    antecedents_used: tuple[ObjectIdentity, ...]
    antecedents_introduced: tuple[ObjectIdentity, ...]
    delivery: AuthorizedGenerationDeliveryDecision
    authorized_states: tuple[AuthorizedCenterState, ...]

    def __post_init__(self) -> None:
        """核验页 identity、预期 cursor/prefix 和 typed 授权输入形状。"""
        for name in ("page_key", "section_key", "paragraph_key"):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"generation page {name} 类型错误")
        if (not isinstance(self.plan_item_keys, tuple)
                or not self.plan_item_keys
                or any(not isinstance(item, StableRecordKey)
                       for item in self.plan_item_keys)
                or len(set(self.plan_item_keys)) != len(self.plan_item_keys)):
            raise LongGenerationCheckpointError("page plan_item_keys 非法")
        if (type(self.expected_revision) is not int
                or self.expected_revision < 0
                or type(self.expected_cursor) is not int
                or self.expected_cursor < 0):
            raise LongGenerationCheckpointError("page expected cursor/revision 非法")
        _digest_key(self.expected_prefix_digest)
        for name in ("antecedents_used", "antecedents_introduced"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ObjectIdentity)
                           for item in values)):
                raise TypeError(f"page {name} 类型错误")
            ordered = tuple(sorted(set(values), key=_identity_sort_key))
            if ordered != values:
                raise LongGenerationCheckpointError(
                    f"page {name} 必须排序去重")
        if not isinstance(self.delivery, AuthorizedGenerationDeliveryDecision):
            raise TypeError("page delivery 类型错误")
        if (not isinstance(self.authorized_states, tuple)
                or any(not isinstance(item, AuthorizedCenterState)
                       for item in self.authorized_states)):
            raise TypeError("page authorized_states 类型错误")


@dataclass(frozen=True)
class LongGenerationContinuation:
    """成功原子提交后的新 checkpoint、page digest 和 continuation 状态。"""

    checkpoint: LongGenerationCheckpoint
    page_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 checkpoint 和 page digest 类型。"""
        if not isinstance(self.checkpoint, LongGenerationCheckpoint):
            raise TypeError("continuation checkpoint 类型错误")
        _digest_key(self.page_digest)

    @property
    def complete(self) -> bool:
        """只有精确覆盖整个冻结 plan 后才返回真。"""
        return self.checkpoint.status == LONG_GENERATION_COMPLETE


def _serialize(checkpoint: LongGenerationCheckpoint) -> bytes:
    """把 checkpoint snapshot 编为规范整数流。"""
    values: list[int] = [_FORMAT_VERSION]
    _pack(values, checkpoint.answer_key.components)
    values.append(checkpoint.revision)
    for value in (
            checkpoint.previous_digest,
            checkpoint.plan_digest,
            checkpoint.prefix_digest):
        _pack(values, value)
    values.append(checkpoint.next_cursor)
    for value in (
            checkpoint.current_section_key,
            checkpoint.current_paragraph_key):
        values.append(int(value is not None))
        if value is not None:
            _pack(values, value.components)
    values.append(len(checkpoint.available_antecedents))
    for item in checkpoint.available_antecedents:
        _pack(values, item.stable_key())
    for items in (
            checkpoint.committed_page_keys,
            checkpoint.citation_record_keys,
            checkpoint.authorization_receipt_keys):
        values.append(len(items))
        for item in items:
            _pack(values, item.components)
    values.append(checkpoint.status)
    return encode_integer_tuple(tuple(values))


def _deserialize(payload: bytes) -> LongGenerationCheckpoint:
    """从规范整数流恢复 checkpoint，并拒绝未知尾字段。"""
    try:
        reader = IntegerStreamReader(decode_integer_tuple(payload))
        version = reader.read_positive(label="checkpoint format")
        if version != _FORMAT_VERSION:
            raise LongGenerationCheckpointError("checkpoint format 不兼容")
        answer_key = StableRecordKey(reader.read_key(label="answer key"))
        revision = reader.read_nonnegative(label="checkpoint revision")
        previous = reader.read_key(label="checkpoint previous", empty=True)
        plan_digest = reader.read_key(label="checkpoint plan digest")
        prefix_digest = reader.read_key(label="checkpoint prefix digest")
        next_cursor = reader.read_nonnegative(label="checkpoint cursor")

        def optional_key(label: str) -> StableRecordKey | None:
            present = reader.read_nonnegative(label=f"{label}.present")
            if present not in (0, 1):
                raise LongGenerationCheckpointError(
                    f"{label} present 标志非法")
            return (
                StableRecordKey(reader.read_key(label=label))
                if present else None
            )

        section = optional_key("checkpoint section")
        paragraph = optional_key("checkpoint paragraph")
        antecedent_count = reader.read_nonnegative(
            label="checkpoint antecedent count")
        antecedents = tuple(
            ObjectIdentity.from_stable_key(reader.read_key(
                label=f"checkpoint antecedent[{index}]"))
            for index in range(antecedent_count)
        )

        def record_keys(label: str) -> tuple[StableRecordKey, ...]:
            count = reader.read_nonnegative(label=f"{label}.count")
            return tuple(
                StableRecordKey(reader.read_key(label=f"{label}[{index}]"))
                for index in range(count)
            )

        pages = record_keys("checkpoint pages")
        citations = record_keys("checkpoint citations")
        receipts = record_keys("checkpoint receipts")
        status = reader.read_positive(label="checkpoint status")
        reader.finish()
        return LongGenerationCheckpoint(
            answer_key,
            revision,
            previous,
            plan_digest,
            next_cursor,
            prefix_digest,
            section,
            paragraph,
            antecedents,
            pages,
            citations,
            receipts,
            status,
        )
    except LongGenerationCheckpointError:
        raise
    except (IntegerCodecError, TypeError, ValueError) as error:
        raise LongGenerationCheckpointError("checkpoint payload 损坏") from error


def _object_identity(
        answer_key: StableRecordKey, revision: int,
        ) -> tuple[int, ...]:
    """形成 repository 内完整 answer key 加 revision identity。"""
    return (
        _FORMAT_VERSION,
        len(answer_key.components),
        *answer_key.components,
        revision,
    )


def _identity_matches(
        identity: tuple[int, ...], answer_key: StableRecordKey,
        ) -> bool:
    """判断 descriptor 是否属于目标 answer checkpoint 链。"""
    size = len(answer_key.components)
    return (
        len(identity) == size + 3
        and identity[0] == _FORMAT_VERSION
        and identity[1] == size
        and identity[2:2 + size] == answer_key.components
        and type(identity[-1]) is int
        and identity[-1] >= 0
    )


def _authorized_page_receipts(
        decision: AuthorizedGenerationDeliveryDecision,
        states: tuple[AuthorizedCenterState, ...],
        ) -> tuple[
            tuple[AuthorizedGenerationClaim, ...],
            tuple[StableRecordKey, ...],
            tuple[StableRecordKey, ...],
        ]:
    """重验严格 G-04 envelope 与真实 READY page-in state 一一对应。"""
    if (decision.state != DELIVERY_AUTHORIZED
            or not decision.deliverable
            or decision.envelope is None):
        raise LongGenerationCheckpointError(
            "generation page 缺授权且完整 G-04 的 envelope")
    claims = decision.envelope.claims
    by_receipt = {
        item.receipt.receipt_key.components: item for item in states
    }
    if len(by_receipt) != len(states):
        raise LongGenerationCheckpointError("authorized page state receipt 重复")
    citation_keys = []
    receipt_keys = []
    for claim in claims:
        state = by_receipt.get(claim.authorization_receipt_key)
        if state is None:
            raise LongGenerationCheckpointError(
                "delivery claim 无对应 authorized page-in state")
        receipt = state.receipt
        if (receipt.state != "READY"
                or state.payload is None
                or receipt.physical_payload_gets + receipt.reused_payload != 1
                or not receipt.citations):
            raise LongGenerationCheckpointError(
                "generation page 未经 READY exact citation page-in")
        if claim.proposition_key != state.payload.proposition.stable_key():
            raise LongGenerationCheckpointError(
                "generation claim 与 page-in proposition 漂移")
        if not set(claim.evidence_sources).issubset({
                item.source_ref for item in receipt.citations}):
            raise LongGenerationCheckpointError(
                "generation claim 引用了 page-in 外来源")
        receipt_keys.append(receipt.receipt_key)
        citation_keys.extend(item.record_key for item in receipt.citations)
    if len(claims) != len(states):
        raise LongGenerationCheckpointError("authorized state 覆盖多于 delivery claim")
    return (
        claims,
        tuple(citation_keys),
        tuple(receipt_keys),
    )


def _page_digest(
        checkpoint: LongGenerationCheckpoint,
        page: LongGenerationPageCommit,
        claims: tuple[AuthorizedGenerationClaim, ...],
        citation_keys: tuple[StableRecordKey, ...],
        receipt_keys: tuple[StableRecordKey, ...],
        ) -> tuple[int, ...]:
    """绑定实际 renderer units、G-04、claims、citation 和 anaphora 的页摘要。"""
    envelope = page.delivery.envelope
    assert envelope is not None
    values: list[int] = [
        _FORMAT_VERSION,
        checkpoint.revision,
        checkpoint.next_cursor,
    ]
    for value in (
            page.page_key.components,
            page.section_key.components,
            page.paragraph_key.components,
            checkpoint.prefix_digest,
            envelope.run_key,
            envelope.units,
            envelope.postcheck_key):
        _pack(values, value)
    values.append(len(page.plan_item_keys))
    for item in page.plan_item_keys:
        _pack(values, item.components)
    values.append(len(claims))
    for claim in claims:
        _pack(values, claim.stable_key())
    for items in (citation_keys, receipt_keys):
        values.append(len(items))
        for item in items:
            _pack(values, item.components)
    for items in (page.antecedents_used, page.antecedents_introduced):
        values.append(len(items))
        for item in items:
            _pack(values, item.stable_key())
    return _digest(tuple(values))


class LongGenerationCheckpointStore:
    """在 seal-last repository 上先 postcheck、后原子追加 continuation snapshot。"""

    def __init__(
            self,
            repository: AppendOnlyObjectRepository,
            commit: Callable[[], None],
            ) -> None:
        """绑定 append-only repository 与显式事务 owner。"""
        if not isinstance(repository, AppendOnlyObjectRepository):
            raise TypeError("checkpoint repository 协议错误")
        if not callable(commit):
            raise TypeError("checkpoint commit 必须可调用")
        self.repository = repository
        self.commit = commit

    def create(self, plan: LongGenerationPlan) -> LongGenerationCheckpoint:
        """独占创建空 cursor；不读取事实、不生成 surface。"""
        if not isinstance(plan, LongGenerationPlan):
            raise TypeError("long generation plan 类型错误")
        if self._descriptors(plan.answer_key):
            raise LongGenerationCheckpointError("answer checkpoint 已存在")
        prefix = _digest((
            _FORMAT_VERSION,
            len(plan.answer_key.components),
            *plan.answer_key.components,
            0,
        ))
        checkpoint = LongGenerationCheckpoint(
            plan.answer_key,
            0,
            (),
            plan.digest(),
            0,
            prefix,
            None,
            None,
            (),
            (),
            (),
            (),
            LONG_GENERATION_OPEN,
        )
        self._put(checkpoint)
        return checkpoint

    def load(self, answer_key: StableRecordKey) -> LongGenerationCheckpoint:
        """只读取 checkpoint metadata，并重验连续 revision/digest 链。"""
        descriptors = self._descriptors(answer_key)
        if not descriptors:
            raise KeyError(f"generation checkpoint 不存在: {answer_key.components}")
        checkpoints = []
        for descriptor in descriptors:
            checkpoint = _deserialize(self.repository.get(
                OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
                descriptor.identity_key,
            ))
            if (checkpoint.answer_key != answer_key
                    or descriptor.identity_key != _object_identity(
                        answer_key, checkpoint.revision)):
                raise LongGenerationCheckpointError(
                    "checkpoint object identity 与 payload 漂移")
            checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda item: item.revision)
        if tuple(item.revision for item in checkpoints) != tuple(
                range(len(checkpoints))):
            raise LongGenerationCheckpointError("checkpoint revision 不连续")
        previous = None
        for checkpoint in checkpoints:
            expected = () if previous is None else previous.digest()
            if checkpoint.previous_digest != expected:
                raise LongGenerationCheckpointError("checkpoint digest 链断裂")
            previous = checkpoint
        assert previous is not None
        return previous

    def commit_page(
            self,
            plan: LongGenerationPlan,
            page: LongGenerationPageCommit,
            budget: LongGenerationPageBudget,
            ) -> LongGenerationContinuation:
        """严格校验 plan/page-in/G-04/anaphora/cursor 后追加一个完整 snapshot。"""
        if not isinstance(plan, LongGenerationPlan):
            raise TypeError("long generation plan 类型错误")
        if not isinstance(page, LongGenerationPageCommit):
            raise TypeError("long generation page 类型错误")
        if not isinstance(budget, LongGenerationPageBudget):
            raise TypeError("long generation page budget 类型错误")
        current = self.load(plan.answer_key)
        if current.status == LONG_GENERATION_COMPLETE:
            raise LongGenerationCheckpointError("long generation 已完成")
        if current.plan_digest != plan.digest():
            raise LongGenerationCheckpointError("generation plan digest 漂移")
        if (page.expected_revision != current.revision
                or page.expected_cursor != current.next_cursor
                or page.expected_prefix_digest != current.prefix_digest):
            raise LongGenerationCheckpointError(
                "generation continuation cursor/prefix 已漂移")
        start = current.next_cursor
        stop = start + len(page.plan_item_keys)
        if stop > len(plan.items):
            raise LongGenerationCheckpointError("generation page 越过 plan 尾部")
        expected_slice = plan.items[start:stop]
        if tuple(item.item_key for item in expected_slice) != page.plan_item_keys:
            raise LongGenerationCheckpointError("generation plan slice 漂移")
        if page.page_key in current.committed_page_keys:
            raise LongGenerationCheckpointError("generation page key 已提交")
        claims, citation_keys, receipt_keys = _authorized_page_receipts(
            page.delivery,
            page.authorized_states,
        )
        envelope = page.delivery.envelope
        assert envelope is not None
        if (len(envelope.units) > budget.max_units
                or len(claims) > budget.max_claims):
            raise LongGenerationCheckpointError("generation page 超出硬预算")
        expected_propositions = tuple(sorted(
            item.proposition_key for item in expected_slice))
        actual_propositions = tuple(sorted(
            claim.proposition_key for claim in claims))
        if expected_propositions != actual_propositions:
            raise LongGenerationCheckpointError(
                "generation page Proposition coverage 漂移")
        if not set(page.antecedents_used).issubset(
                current.available_antecedents):
            raise LongGenerationCheckpointError(
                "generation page 引用了未提交前缀 antecedent")
        if set(page.antecedents_introduced).intersection(
                current.available_antecedents):
            raise LongGenerationCheckpointError(
                "generation page 重复引入 antecedent")
        digest = _page_digest(
            current,
            page,
            claims,
            citation_keys,
            receipt_keys,
        )
        next_prefix = _digest((
            _FORMAT_VERSION,
            *current.prefix_digest,
            *digest,
        ))
        next_status = (
            LONG_GENERATION_COMPLETE
            if stop == len(plan.items)
            else LONG_GENERATION_BUDGET_STOP
        )
        next_checkpoint = LongGenerationCheckpoint(
            current.answer_key,
            current.revision + 1,
            current.digest(),
            current.plan_digest,
            stop,
            next_prefix,
            page.section_key,
            page.paragraph_key,
            tuple(sorted(set((
                *current.available_antecedents,
                *page.antecedents_introduced,
            )), key=_identity_sort_key)),
            (*current.committed_page_keys, page.page_key),
            (*current.citation_record_keys, *citation_keys),
            (*current.authorization_receipt_keys, *receipt_keys),
            next_status,
        )
        self._put(next_checkpoint)
        return LongGenerationContinuation(next_checkpoint, digest)

    def _put(self, checkpoint: LongGenerationCheckpoint) -> None:
        """把已完整核验的单 snapshot seal-last 发布并显式提交。"""
        self.repository.put(
            OBJECT_KIND_LONG_GENERATION_CHECKPOINT,
            _object_identity(checkpoint.answer_key, checkpoint.revision),
            _serialize(checkpoint),
        )
        self.commit()

    def _descriptors(self, answer_key: StableRecordKey):
        """按完整 answer key 过滤 checkpoint descriptor。"""
        if not isinstance(answer_key, StableRecordKey):
            raise TypeError("checkpoint answer_key 类型错误")
        return tuple(
            item
            for item in self.repository.list_kind(
                OBJECT_KIND_LONG_GENERATION_CHECKPOINT)
            if _identity_matches(item.identity_key, answer_key)
        )


__all__ = [
    "LONG_GENERATION_BUDGET_STOP",
    "LONG_GENERATION_COMPLETE",
    "LONG_GENERATION_OPEN",
    "LongGenerationCheckpoint",
    "LongGenerationCheckpointError",
    "LongGenerationCheckpointStore",
    "LongGenerationContinuation",
    "LongGenerationPageBudget",
    "LongGenerationPageCommit",
    "LongGenerationPlan",
    "LongGenerationPlanItem",
    "OBJECT_KIND_LONG_GENERATION_CHECKPOINT",
]
