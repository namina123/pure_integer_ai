"""DLG-RAW-11B：来源锚点与 Frame 问答的可迁移混合会话纯状态。

本模块不读取 terminal、路径、SQLite 或 provider runtime。它只消费已经完成的
``ConversationTurnState`` 与已验证的 ``ProviderOriginAnchorProjectionV1``，把两者
置于带类型的 append-only V2 snapshot 中。Python dataclass 只是一层当前宿主的
结构体便利；所有可观察状态都能导出为有序非负整数 record 和 raw u8 identity。

V2 故意不修改 DLG-RAW-04 的 V1 context/snapshot。未来的 V2 codec 必须以本模块
公开的 record version、turn kind、payload record 和 read witness 为唯一语义来源，
而不得猜测或升级 V1 bytes。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_context_runtime import (
    ConversationTurnState,
)
from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    ProviderOriginAnchorProjectionV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)


MIXED_CONTEXT_SCHEMA_V2 = 2
MIXED_CONTEXT_STATE_RECORD_V2 = 2
MIXED_CONTEXT_READ_WITNESS_RECORD_V2 = 2
MIXED_CONTEXT_READ_RECORD_V2 = 2
MIXED_CONTEXT_FRAME_TURN_RECORD_V2 = 2
MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1 = 1
MIXED_CONTEXT_APPEND_RESULT_RECORD_V1 = 1

MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN = 1
MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION = 2

MIXED_CONTEXT_WRITE_ORIGIN_NONE = 0
MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN = 1
MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION = 2

MIXED_CONTEXT_APPEND_ACCEPTED = 0
MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE = 1
MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS = 2

MIXED_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V2 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/MIXED-CONTEXT-SNAPSHOT/V2")
MIXED_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V2 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/MIXED-CONTEXT-READ-WITNESS/V2")
MIXED_CONTEXT_TURN_IDENTITY_DOMAIN_V2 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/MIXED-CONTEXT-TURN/V2")

_DIGEST_SIZE = 32


# object-model: exception; interop=DLG-RAW-11B
class ProviderOriginContextError(ValueError):
    """混合 context 的整数 record、read witness 或 append 链不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长非负整数段写为显式 count 加有序内容。"""
    result.extend((len(value), *value))


def _vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证不依赖 Python collection 语义的有限非负整数 vector。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ProviderOriginContextError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证不可为空的会话或 target 稳定整数 key。"""
    return _vector(value, label=label, allow_empty=False)


def _nonnegative(value: int, *, label: str) -> int:
    """拒绝 bool、整数子类与负数，固定协议整数边界。"""
    if type(value) is not int or value < 0:
        raise ProviderOriginContextError(f"{label} 必须是非负严格整数")
    return value


def _digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证以显式 raw u8[32] 保存的 identity，不接受 hex 文本。"""
    result = _vector(value, label=label, allow_empty=False)
    if len(result) != _DIGEST_SIZE or any(item > 255 for item in result):
        raise ProviderOriginContextError(f"{label} 必须是 raw u8[32]")
    return result


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """按已冻结 portable SHA framing 形成 raw u8[32] identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ProviderOriginContextError(f"{label} 无法形成") from error


def _read_witness_body(
        value: "MixedContextReadWitnessV2",
        ) -> tuple[int, ...]:
    """写出不含 witness self identity 的 canonical read record。"""
    result = [MIXED_CONTEXT_READ_WITNESS_RECORD_V2]
    _pack(result, value.conversation_key)
    result.append(value.revision)
    _pack(result, value.snapshot_digest_u8)
    result.extend((value.requested_limit, value.visible_start_ordinal))
    result.append(len(value.visible_turn_identities_u8))
    for identity in value.visible_turn_identities_u8:
        _pack(result, identity)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class MixedContextReadWitnessV2:
    """一次明确尾部读取的可持久化整数见证。

    ``requested_limit`` 与实际可见 turn identity 都被记录，避免未来 decoder 或
    caller 把“读取了多少”改写为只看最终 snapshot digest 的隐式行为。
    """

    conversation_key: tuple[int, ...]
    revision: int
    snapshot_digest_u8: tuple[int, ...]
    requested_limit: int
    visible_start_ordinal: int
    visible_turn_identities_u8: tuple[tuple[int, ...], ...]
    witness_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 read scope、尾部范围和 self identity。"""
        key = _key(self.conversation_key, label="mixed context read conversation key")
        revision = _nonnegative(self.revision, label="mixed context read revision")
        digest = _digest(
            self.snapshot_digest_u8,
            label="mixed context read snapshot digest",
        )
        limit = _nonnegative(
            self.requested_limit,
            label="mixed context read requested limit",
        )
        start = _nonnegative(
            self.visible_start_ordinal,
            label="mixed context read visible start ordinal",
        )
        identities = self.visible_turn_identities_u8
        if type(identities) is not tuple:
            raise ProviderOriginContextError(
                "mixed context read visible turn identities 必须是 tuple")
        identities = tuple(
            _digest(item, label=f"mixed context read turn identity[{ordinal}]")
            for ordinal, item in enumerate(identities)
        )
        expected_count = min(revision, limit)
        if (len(identities) != expected_count
                or start != revision - expected_count):
            raise ProviderOriginContextError(
                "mixed context read 可见尾部与 revision/limit 不一致")
        object.__setattr__(self, "conversation_key", key)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "snapshot_digest_u8", digest)
        object.__setattr__(self, "requested_limit", limit)
        object.__setattr__(self, "visible_start_ordinal", start)
        object.__setattr__(self, "visible_turn_identities_u8", identities)
        expected = _identity(
            MIXED_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V2,
            _read_witness_body(self),
            label="mixed context read witness identity",
        )
        supplied = self.witness_identity_u8
        if supplied and _digest(
                supplied,
                label="mixed context read witness identity") != expected:
            raise ProviderOriginContextError(
                "mixed context read witness identity 漂移")
        object.__setattr__(self, "witness_identity_u8", expected)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 future V2 decoder 可直接消费的 read witness record。"""
        result = list(_read_witness_body(self))
        _pack(result, self.witness_identity_u8)
        return tuple(result)


def _frame_turn_body(value: "FrameQuestionAnswerTurnV2") -> tuple[int, ...]:
    """写出 Frame tagged turn 的不含 self identity payload。"""
    result = [
        MIXED_CONTEXT_FRAME_TURN_RECORD_V2,
        MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN,
        value.append_ordinal,
    ]
    for segment in (
            value.previous_snapshot_digest_u8,
            value.prior_read_witness.canonical_record(),
            value.frame_turn.stable_key()):
        _pack(result, segment)
    result.append(value.context_write_origin)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class FrameQuestionAnswerTurnV2:
    """V2 中包装既有完整 Frame ``ConversationTurnState`` 的 tagged turn。"""

    append_ordinal: int
    previous_snapshot_digest_u8: tuple[int, ...]
    prior_read_witness: MixedContextReadWitnessV2
    frame_turn: ConversationTurnState
    context_write_origin: int = MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN
    turn_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 Frame payload、前驱 digest 与显式 prior read witness。"""
        ordinal = _nonnegative(
            self.append_ordinal,
            label="mixed frame append ordinal",
        )
        previous = _digest(
            self.previous_snapshot_digest_u8,
            label="mixed frame previous snapshot digest",
        )
        if type(self.prior_read_witness) is not MixedContextReadWitnessV2:
            raise TypeError("mixed frame prior read witness 类型错误")
        if type(self.frame_turn) is not ConversationTurnState:
            raise TypeError("mixed frame turn 必须包装 ConversationTurnState")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin
                != MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN):
            raise ProviderOriginContextError("mixed frame write origin 漂移")
        if (self.prior_read_witness.revision != ordinal
                or self.prior_read_witness.snapshot_digest_u8 != previous):
            raise ProviderOriginContextError(
                "mixed frame 前驱 digest 或 read witness revision 漂移")
        _vector(
            self.frame_turn.stable_key(),
            label="mixed frame legacy typed record",
            allow_empty=False,
        )
        object.__setattr__(self, "append_ordinal", ordinal)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        expected = _identity(
            MIXED_CONTEXT_TURN_IDENTITY_DOMAIN_V2,
            _frame_turn_body(self),
            label="mixed frame turn identity",
        )
        supplied = self.turn_identity_u8
        if supplied and _digest(
                supplied,
                label="mixed frame turn identity") != expected:
            raise ProviderOriginContextError("mixed frame turn identity 漂移")
        object.__setattr__(self, "turn_identity_u8", expected)

    @property
    def turn_kind(self) -> int:
        """返回 V2 tagged union 的冻结 Frame kind code。"""
        return MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN

    @property
    def target_key(self) -> tuple[int, ...]:
        """暴露 Frame payload 已有 target key，不从 surface 或 provider 推断。"""
        return self.frame_turn.target_key

    def payload_record(self) -> tuple[int, ...]:
        """返回包装的完整 legacy typed record，供 V2 decoder 的 Frame 分支使用。"""
        return self.frame_turn.stable_key()

    def canonical_record(self) -> tuple[int, ...]:
        """导出 ``record_version, turn_kind, payload`` 的 canonical V2 turn。"""
        result = list(_frame_turn_body(self))
        _pack(result, self.turn_identity_u8)
        return tuple(result)


def _provider_turn_body(
        value: "ProviderOriginContextTurnV1",
        ) -> tuple[int, ...]:
    """写出 provider-origin tagged turn 的不含 self identity payload。"""
    result = [
        MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
        value.append_ordinal,
    ]
    for segment in (
            value.previous_snapshot_digest_u8,
            value.prior_read_witness.canonical_record(),
            value.anchor_projection.canonical_record(),
            value.provider_result_identity_u8):
        _pack(result, segment)
    result.extend((value.context_write_origin, len(value.consumed_reference)))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class ProviderOriginContextTurnV1:
    """只承载已验证 anchor 的 provider-origin V2 union payload。

    它没有 request、target、query、planning 或 surface text。``consumed_reference``
    在 11B 冻结为空，因而该 turn 绝不能被当前 target-anchor 机制误认作 Frame。
    """

    append_ordinal: int
    previous_snapshot_digest_u8: tuple[int, ...]
    prior_read_witness: MixedContextReadWitnessV2
    anchor_projection: ProviderOriginAnchorProjectionV1
    provider_result_identity_u8: tuple[int, ...]
    consumed_reference: tuple[tuple[int, ...], ...] = ()
    context_write_origin: int = (
        MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)
    turn_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """只接纳完整 ``ANCHOR_ANSWER``，并禁止 provider 伪装成 target anchor。"""
        ordinal = _nonnegative(
            self.append_ordinal,
            label="mixed provider append ordinal",
        )
        previous = _digest(
            self.previous_snapshot_digest_u8,
            label="mixed provider previous snapshot digest",
        )
        if type(self.prior_read_witness) is not MixedContextReadWitnessV2:
            raise TypeError("mixed provider prior read witness 类型错误")
        if type(self.anchor_projection) is not ProviderOriginAnchorProjectionV1:
            raise TypeError("mixed provider anchor projection 类型错误")
        if not self.anchor_projection.accepted:
            raise ProviderOriginContextError(
                "mixed provider context turn 只能消费 ANCHOR_ANSWER")
        result_identity = _digest(
            self.provider_result_identity_u8,
            label="mixed provider result identity",
        )
        if result_identity != self.anchor_projection.provider_result_identity_u8:
            raise ProviderOriginContextError(
                "mixed provider result identity 与 anchor 漂移")
        if self.consumed_reference != ():
            raise ProviderOriginContextError(
                "mixed provider 11B consumed reference 必须为空")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin
                != MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION):
            raise ProviderOriginContextError("mixed provider write origin 漂移")
        if (self.prior_read_witness.revision != ordinal
                or self.prior_read_witness.snapshot_digest_u8 != previous):
            raise ProviderOriginContextError(
                "mixed provider 前驱 digest 或 read witness revision 漂移")
        object.__setattr__(self, "append_ordinal", ordinal)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        object.__setattr__(self, "provider_result_identity_u8", result_identity)
        expected = _identity(
            MIXED_CONTEXT_TURN_IDENTITY_DOMAIN_V2,
            _provider_turn_body(self),
            label="mixed provider turn identity",
        )
        supplied = self.turn_identity_u8
        if supplied and _digest(
                supplied,
                label="mixed provider turn identity") != expected:
            raise ProviderOriginContextError("mixed provider turn identity 漂移")
        object.__setattr__(self, "turn_identity_u8", expected)

    @property
    def turn_kind(self) -> int:
        """返回 V2 tagged union 的冻结 provider-origin kind code。"""
        return MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION

    def payload_record(self) -> tuple[int, ...]:
        """返回不含 turn envelope 的来源锚点和同次 result identity payload。"""
        result: list[int] = []
        _pack(result, self.anchor_projection.canonical_record())
        _pack(result, self.provider_result_identity_u8)
        result.append(len(self.consumed_reference))
        return tuple(result)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 ``record_version, turn_kind, payload`` 的 canonical provider turn。"""
        result = list(_provider_turn_body(self))
        _pack(result, self.turn_identity_u8)
        return tuple(result)


MixedContextTurnV2 = FrameQuestionAnswerTurnV2 | ProviderOriginContextTurnV1


def _turn(value: MixedContextTurnV2, *, label: str) -> MixedContextTurnV2:
    """验证 tagged union 只含两个冻结的 V2 turn 分支。"""
    if (type(value) is not FrameQuestionAnswerTurnV2
            and type(value) is not ProviderOriginContextTurnV1):
        raise TypeError(f"{label} 必须是已登记 mixed context turn")
    return value


def _state_record(
        conversation_key: tuple[int, ...],
        revision: int,
        previous_snapshot_digest_u8: tuple[int, ...],
        turns: tuple[MixedContextTurnV2, ...],
        ) -> tuple[int, ...]:
    """写出完整 V2 snapshot record；digest 以此 record 为唯一输入。"""
    result = [MIXED_CONTEXT_STATE_RECORD_V2, MIXED_CONTEXT_SCHEMA_V2]
    _pack(result, conversation_key)
    result.append(revision)
    _pack(result, previous_snapshot_digest_u8)
    result.append(len(turns))
    for turn in turns:
        _pack(result, turn.canonical_record())
    return tuple(result)


def _snapshot_digest(
        conversation_key: tuple[int, ...],
        revision: int,
        previous_snapshot_digest_u8: tuple[int, ...],
        turns: tuple[MixedContextTurnV2, ...],
        ) -> tuple[int, ...]:
    """从完整 V2 state record 形成 raw snapshot identity。"""
    return _identity(
        MIXED_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V2,
        _state_record(
            conversation_key,
            revision,
            previous_snapshot_digest_u8,
            turns,
        ),
        label="mixed context snapshot identity",
    )


def _read_matches_prefix(
        read: MixedContextReadV2,
        conversation_key: tuple[int, ...],
        revision: int,
        snapshot_digest_u8: tuple[int, ...],
        prefix: tuple[MixedContextTurnV2, ...],
        ) -> bool:
    """验证 read witness 精确描述给定 snapshot 的所请求可见尾部。"""
    witness = read.witness
    if (witness.conversation_key != conversation_key
            or witness.revision != revision
            or witness.snapshot_digest_u8 != snapshot_digest_u8):
        return False
    visible = prefix[-witness.requested_limit:] if witness.requested_limit else ()
    if tuple(turn.canonical_record() for turn in read.turns) != tuple(
            turn.canonical_record() for turn in visible):
        return False
    return witness.visible_turn_identities_u8 == tuple(
        turn.turn_identity_u8 for turn in visible)


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class MixedContextReadV2:
    """由 snapshot 产生的 explicit read，含见证及可消费的 tagged 尾轮。"""

    witness: MixedContextReadWitnessV2
    turns: tuple[MixedContextTurnV2, ...]

    def __post_init__(self) -> None:
        """确保 witness 与实际 typed tail 的 ordinal 和 identity 全部一致。"""
        if type(self.witness) is not MixedContextReadWitnessV2:
            raise TypeError("mixed context read witness 类型错误")
        if type(self.turns) is not tuple:
            raise ProviderOriginContextError("mixed context read turns 必须是 tuple")
        turns = tuple(_turn(item, label=f"mixed context read turn[{ordinal}]")
                      for ordinal, item in enumerate(self.turns))
        expected_ordinals = tuple(range(
            self.witness.visible_start_ordinal,
            self.witness.revision,
        ))
        if tuple(item.append_ordinal for item in turns) != expected_ordinals:
            raise ProviderOriginContextError(
                "mixed context read turns 不是 witness 所示连续尾部")
        identities = tuple(item.turn_identity_u8 for item in turns)
        if identities != self.witness.visible_turn_identities_u8:
            raise ProviderOriginContextError(
                "mixed context read turn identity 与 witness 漂移")
        object.__setattr__(self, "turns", turns)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 future V2 snapshot decoder 可审计的完整 read record。"""
        result = [MIXED_CONTEXT_READ_RECORD_V2]
        _pack(result, self.witness.canonical_record())
        result.append(len(self.turns))
        for turn in self.turns:
            _pack(result, turn.canonical_record())
        return tuple(result)

    def latest_frame_target_turn(
            self,
            target_key: tuple[int, ...],
            ) -> FrameQuestionAnswerTurnV2 | None:
        """只在可见尾轮是同 target 的 Frame 时返回，不跨 provider 向前跳。"""
        target = _key(target_key, label="mixed context target key")
        if not self.turns:
            return None
        last = self.turns[-1]
        if type(last) is not FrameQuestionAnswerTurnV2:
            return None
        return last if last.target_key == target else None


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class MixedConversationContextStateV2:
    """V2 append-only mixed context snapshot，Frame 与 provider turn 显式分型。"""

    conversation_key: tuple[int, ...]
    revision: int
    previous_snapshot_digest_u8: tuple[int, ...]
    turns: tuple[MixedContextTurnV2, ...] = ()

    def __post_init__(self) -> None:
        """验证全量 predecessor/read witness 链，不允许跳轮或可见范围漂移。"""
        key = _key(self.conversation_key, label="mixed context conversation key")
        revision = _nonnegative(self.revision, label="mixed context revision")
        if type(self.turns) is not tuple:
            raise ProviderOriginContextError("mixed context turns 必须是 tuple")
        turns = tuple(_turn(item, label=f"mixed context turn[{ordinal}]")
                      for ordinal, item in enumerate(self.turns))
        if revision != len(turns):
            raise ProviderOriginContextError(
                "mixed context revision 必须等于 turn 数")
        if revision == 0:
            if self.previous_snapshot_digest_u8 != ():
                raise ProviderOriginContextError(
                    "mixed context 初始 snapshot 不得携带 previous digest")
            previous: tuple[int, ...] = ()
        else:
            previous = _digest(
                self.previous_snapshot_digest_u8,
                label="mixed context previous snapshot digest",
            )
        object.__setattr__(self, "conversation_key", key)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "previous_snapshot_digest_u8", previous)
        object.__setattr__(self, "turns", turns)

        expected_prior = _snapshot_digest(key, 0, (), ())
        prefix: tuple[MixedContextTurnV2, ...] = ()
        for ordinal, turn in enumerate(turns):
            if (turn.append_ordinal != ordinal
                    or turn.previous_snapshot_digest_u8 != expected_prior):
                raise ProviderOriginContextError(
                    "mixed context turn ordinal 或 previous digest 链漂移")
            witness = turn.prior_read_witness
            read = MixedContextReadV2(
                witness,
                prefix[-witness.requested_limit:]
                if witness.requested_limit else (),
            )
            if not _read_matches_prefix(
                    read,
                    key,
                    ordinal,
                    expected_prior,
                    prefix,
            ):
                raise ProviderOriginContextError(
                    "mixed context turn prior read witness 漂移")
            prefix = (*prefix, turn)
            expected_prior = _snapshot_digest(
                key,
                ordinal + 1,
                turn.previous_snapshot_digest_u8,
                prefix,
            )
        if revision and previous != turns[-1].previous_snapshot_digest_u8:
            raise ProviderOriginContextError(
                "mixed context state previous digest 与末轮不一致")

    def canonical_record(self) -> tuple[int, ...]:
        """导出完整 V2 snapshot 的唯一 canonical integer record。"""
        return _state_record(
            self.conversation_key,
            self.revision,
            self.previous_snapshot_digest_u8,
            self.turns,
        )

    def digest(self) -> tuple[int, ...]:
        """返回当前完整 snapshot 的 raw u8[32] identity。"""
        return _snapshot_digest(
            self.conversation_key,
            self.revision,
            self.previous_snapshot_digest_u8,
            self.turns,
        )

    def turn_records(self) -> tuple[tuple[int, ...], ...]:
        """返回按 append ordinal 排列的 tagged turn canonical records。"""
        return tuple(turn.canonical_record() for turn in self.turns)

    def visible_turns(self, limit: int) -> tuple[MixedContextTurnV2, ...]:
        """只提供明确上限内的最近 tagged turns，零上限不读取任何 turn。"""
        bounded = _nonnegative(limit, label="mixed context visible limit")
        return self.turns[-bounded:] if bounded else ()

    def read(self, limit: int) -> MixedContextReadV2:
        """形成绑定当前 snapshot digest 的显式尾部 read witness。"""
        bounded = _nonnegative(limit, label="mixed context read limit")
        visible = self.visible_turns(bounded)
        witness = MixedContextReadWitnessV2(
            self.conversation_key,
            self.revision,
            self.digest(),
            bounded,
            self.revision - len(visible),
            tuple(turn.turn_identity_u8 for turn in visible),
        )
        return MixedContextReadV2(witness, visible)

    def latest_frame_target_turn(
            self,
            target_key: tuple[int, ...],
            ) -> FrameQuestionAnswerTurnV2 | None:
        """按一轮可见预算查 target anchor，provider 尾轮必定返回 ``None``。"""
        return self.read(1).latest_frame_target_turn(target_key)

    def _read_is_current(self, read: MixedContextReadV2) -> bool:
        """比较完整 canonical read，拒绝其他 revision、scope 或 physical replay。"""
        if type(read) is not MixedContextReadV2:
            return False
        expected = self.read(read.witness.requested_limit)
        return read.canonical_record() == expected.canonical_record()

    def admit_frame_qa_run(
            self,
            frame_turn: ConversationTurnState,
            prior_read: MixedContextReadV2 | None,
            ) -> "MixedContextAppendResultV1":
        """以当前 read witness 写入一个 Frame tagged turn，失败保持原 snapshot。"""
        if type(frame_turn) is not ConversationTurnState:
            raise TypeError("mixed context frame admission 需要 ConversationTurnState")
        if not self._read_is_current(prior_read):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        if prior_read is None:
            raise AssertionError("mixed context current read 不得为空")
        turn = FrameQuestionAnswerTurnV2(
            self.revision,
            self.digest(),
            prior_read.witness,
            frame_turn,
        )
        after = MixedConversationContextStateV2(
            self.conversation_key,
            self.revision + 1,
            self.digest(),
            (*self.turns, turn),
        )
        return MixedContextAppendResultV1(
            self,
            prior_read,
            MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
            MIXED_CONTEXT_APPEND_ACCEPTED,
            turn,
            after,
        )

    def admit_provider_origin_projection(
            self,
            anchor_projection: ProviderOriginAnchorProjectionV1,
            prior_read: MixedContextReadV2 | None = None,
            ) -> "MixedContextAppendResultV1":
        """独立入场 provider anchor；``ANCHOR_NONE`` 不读、不写且返回拒绝 record。"""
        if type(anchor_projection) is not ProviderOriginAnchorProjectionV1:
            raise TypeError(
                "mixed context provider admission 需要 ProviderOriginAnchorProjectionV1")
        if not anchor_projection.accepted:
            return _rejected_append_result(
                self,
                None,
                MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
            )
        if not self._read_is_current(prior_read):
            return _rejected_append_result(
                self,
                prior_read,
                MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
            )
        if prior_read is None:
            raise AssertionError("mixed context current read 不得为空")
        turn = ProviderOriginContextTurnV1(
            self.revision,
            self.digest(),
            prior_read.witness,
            anchor_projection,
            anchor_projection.provider_result_identity_u8,
        )
        after = MixedConversationContextStateV2(
            self.conversation_key,
            self.revision + 1,
            self.digest(),
            (*self.turns, turn),
        )
        return MixedContextAppendResultV1(
            self,
            prior_read,
            MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
            MIXED_CONTEXT_APPEND_ACCEPTED,
            turn,
            after,
        )


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class MixedContextAppendResultV1:
    """一次 V2 admission 的显式结果；拒绝永远保持 ``after == before``。"""

    before: MixedConversationContextStateV2
    prior_read: MixedContextReadV2 | None
    context_write_origin: int
    result_code: int
    appended_turn: MixedContextTurnV2 | None
    after: MixedConversationContextStateV2

    def __post_init__(self) -> None:
        """冻结 write origin、结果码、前后 snapshot 和 append/no-op 对应关系。"""
        if type(self.before) is not MixedConversationContextStateV2:
            raise TypeError("mixed context append before 类型错误")
        if (self.prior_read is not None
                and type(self.prior_read) is not MixedContextReadV2):
            raise TypeError("mixed context append prior read 类型错误")
        if type(self.after) is not MixedConversationContextStateV2:
            raise TypeError("mixed context append after 类型错误")
        if (type(self.result_code) is not int
                or self.result_code not in (
                MIXED_CONTEXT_APPEND_ACCEPTED,
                MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
                MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS)):
            raise ProviderOriginContextError("mixed context append result code 未注册")
        if (type(self.context_write_origin) is not int
                or self.context_write_origin not in (
                MIXED_CONTEXT_WRITE_ORIGIN_NONE,
                MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN,
                MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)):
            raise ProviderOriginContextError("mixed context append write origin 未注册")
        if self.result_code == MIXED_CONTEXT_APPEND_ACCEPTED:
            if self.prior_read is None or self.appended_turn is None:
                raise ProviderOriginContextError(
                    "mixed context accepted append 缺 prior read 或 turn")
            appended = _turn(self.appended_turn, label="mixed context appended turn")
            if (self.context_write_origin == MIXED_CONTEXT_WRITE_ORIGIN_NONE
                    or self.after.conversation_key != self.before.conversation_key
                    or self.after.revision != self.before.revision + 1
                    or self.after.turn_records() != (*self.before.turn_records(),
                                                     appended.canonical_record())
                    or appended.previous_snapshot_digest_u8
                    != self.before.digest()
                    or not self.before._read_is_current(self.prior_read)
                    or (appended.prior_read_witness.canonical_record()
                        != self.prior_read.witness.canonical_record())):
                raise ProviderOriginContextError(
                    "mixed context accepted append 前后状态或 witness 漂移")
            if ((type(appended) is FrameQuestionAnswerTurnV2
                 and self.context_write_origin
                 != MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN)
                    or (type(appended) is ProviderOriginContextTurnV1
                        and self.context_write_origin
                        != MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION)):
                raise ProviderOriginContextError(
                    "mixed context append write origin 与 tagged turn 不一致")
            return
        if (self.context_write_origin != MIXED_CONTEXT_WRITE_ORIGIN_NONE
                or self.appended_turn is not None
                or (self.after.canonical_record()
                    != self.before.canonical_record())):
            raise ProviderOriginContextError(
                "mixed context rejected append 必须为 NONE/no-op")
        if (self.result_code == MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE
                and self.prior_read is not None):
            raise ProviderOriginContextError(
                "mixed context anchor-none 拒绝不得消费 prior read")

    @property
    def accepted(self) -> bool:
        """返回该 admission 是否真的形成了一次 append。"""
        return self.result_code == MIXED_CONTEXT_APPEND_ACCEPTED

    def canonical_record(self) -> tuple[int, ...]:
        """导出 transition 的全部整数证据，供 future V2 runtime 审计。"""
        result = [
            MIXED_CONTEXT_APPEND_RESULT_RECORD_V1,
            self.result_code,
            self.context_write_origin,
        ]
        for segment in (
                self.before.canonical_record(),
                (() if self.prior_read is None
                 else self.prior_read.canonical_record()),
                (() if self.appended_turn is None
                 else self.appended_turn.canonical_record()),
                self.after.canonical_record()):
            _pack(result, segment)
        return tuple(result)


def _rejected_append_result(
        state: MixedConversationContextStateV2,
        prior_read: MixedContextReadV2 | None,
        result_code: int,
        ) -> MixedContextAppendResultV1:
    """统一构造写入来源为 NONE 的 immutable admission rejection。"""
    return MixedContextAppendResultV1(
        state,
        prior_read,
        MIXED_CONTEXT_WRITE_ORIGIN_NONE,
        result_code,
        None,
        state,
    )


def start_mixed_conversation_context_v2(
        conversation_key: tuple[int, ...],
        ) -> MixedConversationContextStateV2:
    """创建 revision 0 的 V2 mixed context；首次 append 也必须持有 read(0) 见证。"""
    return MixedConversationContextStateV2(conversation_key, 0, ())


__all__ = [
    "MIXED_CONTEXT_APPEND_ACCEPTED",
    "MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE",
    "MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS",
    "MIXED_CONTEXT_APPEND_RESULT_RECORD_V1",
    "MIXED_CONTEXT_FRAME_TURN_RECORD_V2",
    "MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1",
    "MIXED_CONTEXT_READ_RECORD_V2",
    "MIXED_CONTEXT_READ_WITNESS_IDENTITY_DOMAIN_V2",
    "MIXED_CONTEXT_READ_WITNESS_RECORD_V2",
    "MIXED_CONTEXT_SCHEMA_V2",
    "MIXED_CONTEXT_SNAPSHOT_IDENTITY_DOMAIN_V2",
    "MIXED_CONTEXT_STATE_RECORD_V2",
    "MIXED_CONTEXT_TURN_IDENTITY_DOMAIN_V2",
    "MIXED_CONTEXT_TURN_KIND_FRAME_QA_RUN",
    "MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION",
    "MIXED_CONTEXT_WRITE_ORIGIN_FRAME_QA_RUN",
    "MIXED_CONTEXT_WRITE_ORIGIN_NONE",
    "MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION",
    "FrameQuestionAnswerTurnV2",
    "MixedContextAppendResultV1",
    "MixedContextReadV2",
    "MixedContextReadWitnessV2",
    "MixedContextTurnV2",
    "MixedConversationContextStateV2",
    "ProviderOriginContextError",
    "ProviderOriginContextTurnV1",
    "start_mixed_conversation_context_v2",
]
