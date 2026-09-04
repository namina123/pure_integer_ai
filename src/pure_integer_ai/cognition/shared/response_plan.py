"""阶段 C：闭合命题到 ResponsePlan 的组织契约（纯整数值类型）。

ResponsePlan 是生成入口唯一接受的已查询组织结果：它把一次闭合回答分解为
response_act、claim_refs、event_refs、memory_refs、slot_sequence、
discourse_links、scope_and_time、evidence_refs、style_ref、carrier_ref 与
realization_candidates。所有引用均为一等 ObjectIdentity 或稳定整数键，
不携带语言字符串；表层只作为最终输出边界，由 realization 渲染。

每个 slot 记录角色、可允许节点类型、是否必填、填充者整数引用、来源与
token 验证摘要；渲染结果必须能由图内已学 literal 间隔与角色表层精确重建，
否则判定为不可验证并拒绝。代码不选择某语言的连接词——literal 间隔全部来自
训练后 Span 图。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_REPRESENTATION,
    OBJECT_ROLE,
    OBJECT_SET_EXPR,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)

# 可允许的槽位角色类型：结构概念（契约槽）或训练后关系图的 Role（RoleBinding
# 的 role 身份）。filler 允许的概念/事件/命题/集合节点由训练图实际产生。
ROLE_NODE_KINDS = frozenset({OBJECT_STRUCTURE_CONCEPT, OBJECT_ROLE})
FILLER_NODE_KINDS = frozenset({
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_REPRESENTATION,
})

_RESPONSE_PLAN_DOMAIN = "pure_integer_ai.response_plan.v1"
_RESPONSE_PLAN_ACT_DOMAIN = "pure_integer_ai.response_plan.act.v1"
_RESPONSE_PLAN_SLOT_TOKEN_DOMAIN = "pure_integer_ai.response_plan.slot_token.v1"


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数键加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str,
                allow_empty: bool = False) -> tuple[int, ...]:
    """核验开放整数键并拒绝 bool、str、float 混入。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if not allow_empty and not value:
        raise ValueError(f"{label} 不能为空")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 act/reason 均为一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _token_digest(surface: str) -> tuple[int, ...]:
    """按码点整数流给出 slot token 的确定性内容摘要。"""
    if type(surface) is not str or not surface:
        raise ValueError("response slot token 摘要需要非空表层")
    return integer_tuple_fingerprint(
        tuple(ord(value) for value in surface),
        domain=_RESPONSE_PLAN_SLOT_TOKEN_DOMAIN,
    )


def response_act_identity(act_key: tuple[int, ...], *, version: int = 1) -> ObjectIdentity:
    """构造 answer/explain/clarify 等 response act 的一等指令身份。"""
    key = _strict_key(act_key, label="response act key")
    if type(version) is not int or version <= 0:
        raise ValueError("response act version 必须为正整数")
    return ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION,
        (version, *_packed(key)),
    )


def _identity_key(identity: ObjectIdentity | None) -> tuple[int, ...]:
    """把可选一等身份编码为带空边界的稳定键。"""
    if identity is None:
        return ()
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("response plan identity 必须是 ObjectIdentity")
    return identity.stable_key()


def _identities_key(identities: tuple[ObjectIdentity, ...]) -> tuple[int, ...]:
    """把一组一等身份编码为按 stable_key 排序的长度限定序列。"""
    if not isinstance(identities, tuple) or any(
            not isinstance(item, ObjectIdentity) for item in identities):
        raise TypeError("response plan identities 必须是 ObjectIdentity tuple")
    ordered = tuple(sorted(identities, key=ObjectIdentity.stable_key))
    out: list[int] = []
    for item in ordered:
        out.extend(_packed(item.stable_key()))
    return len(ordered), *out


@dataclass(frozen=True)
class ResponseSlot:
    """一个有序输出槽位：角色、填充者、表层与来源、token 验证摘要。"""

    role: ObjectIdentity
    filler: ObjectIdentity
    filler_surface: str
    required: bool
    source_hash: int
    token_digest: tuple[int, ...] = ()
    allowed_node_kinds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, ObjectIdentity):
            raise TypeError("response slot role 必须是 ObjectIdentity")
        if self.role.object_kind not in ROLE_NODE_KINDS:
            raise ValueError("response slot role 必须是一等 Role/StructureConcept")
        if not isinstance(self.filler, ObjectIdentity):
            raise TypeError("response slot filler 必须是 ObjectIdentity")
        if type(self.filler_surface) is not str or not self.filler_surface.strip():
            raise ValueError("response slot filler 表层不能为空")
        if type(self.required) is not bool:
            raise TypeError("response slot required 必须是 bool")
        if type(self.source_hash) is not int or self.source_hash <= 0:
            raise ValueError("response slot source_hash 必须为正整数")
        expected = _token_digest(self.filler_surface)
        if not self.token_digest:
            object.__setattr__(self, "token_digest", expected)
        elif self.token_digest != expected:
            raise ValueError("response slot token 摘要与表层不一致")
        if not isinstance(self.allowed_node_kinds, tuple) or not (
                self.allowed_node_kinds) or any(
                type(item) is not int or item <= 0
                for item in self.allowed_node_kinds):
            raise ValueError("response slot allowed_node_kinds 必须是非空正整数 tuple")
        if self.filler.object_kind not in self.allowed_node_kinds:
            raise ValueError("response slot filler 类型不在 allowed_node_kinds 内")

    def stable_key(self) -> tuple[int, ...]:
        """返回角色、填充者、来源与 token 摘要的稳定键。"""
        return (
            *_packed(self.role.stable_key()),
            *_packed(self.filler.stable_key()),
            *_packed(tuple(ord(v) for v in self.filler_surface)),
            1 if self.required else 0,
            self.source_hash,
            *_packed(self.token_digest),
            len(self.allowed_node_kinds),
            *self.allowed_node_kinds,
        )


@dataclass(frozen=True)
class ResponseRealization:
    """同一次 ResponsePlan 的一个已学习 realization 候选。"""

    surface: str
    frame_proposition: ObjectIdentity
    frame_source_hash: int
    slot_count: int

    def __post_init__(self) -> None:
        if type(self.surface) is not str or not self.surface.strip():
            raise ValueError("response realization surface 不能为空")
        if not isinstance(self.frame_proposition, ObjectIdentity):
            raise TypeError("response realization frame_proposition 类型错误")
        if type(self.frame_source_hash) is not int or self.frame_source_hash <= 0:
            raise ValueError("response realization frame source_hash 必须为正整数")
        if type(self.slot_count) is not int or self.slot_count <= 0:
            raise ValueError("response realization slot_count 必须为正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回表层 token 内容引用、框架命题与来源。"""
        return (
            *_packed(integer_tuple_fingerprint(
                tuple(ord(v) for v in self.surface),
                domain=_RESPONSE_PLAN_ACT_DOMAIN,
            )),
            *_packed(self.frame_proposition.stable_key()),
            self.frame_source_hash,
            self.slot_count,
        )


@dataclass(frozen=True)
class ResponsePlan:
    """一次已查询回答的完整组织计划（值型、整数、无宿主对象）。"""

    response_act: ObjectIdentity
    claim_refs: tuple[ObjectIdentity, ...]
    slot_sequence: tuple[ResponseSlot, ...]
    realization_candidates: tuple[ResponseRealization, ...]
    evidence_refs: tuple[tuple[int, ...], ...]
    scope_and_time: tuple[int, ...] = ()
    discourse_links: tuple[ObjectIdentity, ...] = ()
    event_refs: tuple[ObjectIdentity, ...] = ()
    memory_refs: tuple[ObjectIdentity, ...] = ()
    style_ref: ObjectIdentity | None = None
    carrier_ref: ObjectIdentity | None = None
    _stable_key_cache: tuple[int, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        _require_instruction(self.response_act, label="response plan act")
        if not isinstance(self.claim_refs, tuple) or any(
                not isinstance(item, ObjectIdentity)
                for item in self.claim_refs):
            raise TypeError("response plan claim_refs 必须是 ObjectIdentity tuple")
        if not isinstance(self.slot_sequence, tuple) or not self.slot_sequence or any(
                not isinstance(item, ResponseSlot) for item in self.slot_sequence):
            raise TypeError("response plan slot_sequence 必须是非空 ResponseSlot tuple")
        if not isinstance(self.realization_candidates, tuple) or not (
                self.realization_candidates) or any(
                not isinstance(item, ResponseRealization)
                for item in self.realization_candidates):
            raise TypeError(
                "response plan realization_candidates 必须是非空 ResponseRealization tuple")
        if not isinstance(self.evidence_refs, tuple) or any(
                not isinstance(item, tuple)
                or not item
                or any(type(v) is not int for v in item)
                for item in self.evidence_refs):
            raise TypeError("response plan evidence_refs 必须是非空整数 tuple 集合")
        if len(set(item.stable_key() for item in self.slot_sequence)) != len(
                self.slot_sequence):
            raise ValueError("response plan slot 序列不得重复同一角色")
        self._validate_required_slots()
        object.__setattr__(self, "claim_refs", tuple(sorted(
            self.claim_refs, key=ObjectIdentity.stable_key)))
        object.__setattr__(self, "realization_candidates", tuple(sorted(
            self.realization_candidates,
            key=lambda item: item.stable_key())))

    def _validate_required_slots(self) -> None:
        """至少一个 realization 按序包含全部必填槽表层（token postcheck）。"""
        for slot in self.slot_sequence:
            if slot.required and not slot.filler_surface.strip():
                raise ValueError("response plan 必填槽位缺少填充者")
        # token postcheck：槽表层必须按 slot 顺序作为连续子串出现在同一个
        # realization 中；连接词/literal 间隔来自框架图而非宿主语言规则。
        verified = self._verified_realization()
        if not verified.surface.strip():
            raise ValueError("response plan 没有可核验 realization 表层")

    def _verified_realization(self) -> ResponseRealization:
        """选择按序包含全部必填槽表层、可被 token 验证的确定性 realization。"""
        required_surfaces = tuple(
            slot.filler_surface for slot in self.slot_sequence
            if slot.required)
        candidates = []
        for realization in self.realization_candidates:
            cursor = 0
            ok = True
            for surface in required_surfaces:
                position = realization.surface.find(surface, cursor)
                if position < 0:
                    ok = False
                    break
                cursor = position + len(surface)
            if ok:
                candidates.append(realization)
        if not candidates:
            raise ValueError("response plan realization 未按序覆盖全部必填槽 token")
        return min(candidates, key=lambda item: item.stable_key())

    def surface(self) -> str:
        """返回通过 token postcheck 的确定性 realization 表层（非整句回放）。"""
        return self._verified_realization().surface

    def render(self, realization: ResponseRealization) -> str:
        """核验一个 realization 与 slot 命题一致后返回其表层。"""
        if not isinstance(realization, ResponseRealization):
            raise TypeError("response plan render 需要 ResponseRealization")
        if realization not in self.realization_candidates:
            raise ValueError("response plan 只渲染自己的 realization 候选")
        # token postcheck：被渲染的 realization 也必须按序覆盖必填槽 token。
        selected = self._verified_realization()
        return selected.surface

    def stable_key(self) -> tuple[int, ...]:
        """返回完整计划的稳定整数键（含 act、slot、realization 与证据）。"""
        if self._stable_key_cache:
            return self._stable_key_cache
        out = [
            *_packed(self.response_act.stable_key()),
            *_identities_key(self.claim_refs),
            *_identities_key(self.discourse_links),
            *_identities_key(self.event_refs),
            *_identities_key(self.memory_refs),
            *_packed(self.scope_and_time),
        ]
        out.append(len(self.slot_sequence))
        for slot in self.slot_sequence:
            out.extend(_packed(slot.stable_key()))
        out.append(len(self.realization_candidates))
        for realization in self.realization_candidates:
            out.extend(_packed(realization.stable_key()))
        out.append(len(self.evidence_refs))
        for ref in self.evidence_refs:
            out.extend(_packed(ref))
        out.append(0 if self.style_ref is None else 1)
        if self.style_ref is not None:
            out.extend(_packed(self.style_ref.stable_key()))
        out.append(0 if self.carrier_ref is None else 1)
        if self.carrier_ref is not None:
            out.extend(_packed(self.carrier_ref.stable_key()))
        value = tuple(out)
        object.__setattr__(self, "_stable_key_cache", value)
        return value

    def to_dict(self) -> dict[str, object]:
        """序列化为纯整数可重放 trace（不含宿主对象与语言机制）。"""
        return {
            "response_act": list(self.response_act.stable_key()),
            "claim_refs": [list(item.stable_key())
                           for item in self.claim_refs],
            "event_refs": [list(item.stable_key())
                           for item in self.event_refs],
            "memory_refs": [list(item.stable_key())
                            for item in self.memory_refs],
            "slot_sequence": [
                {
                    "role": list(slot.role.stable_key()),
                    "filler": list(slot.filler.stable_key()),
                    "filler_surface": slot.filler_surface,
                    "required": slot.required,
                    "source_hash": slot.source_hash,
                    "token_digest": list(slot.token_digest),
                    "allowed_node_kinds": list(slot.allowed_node_kinds),
                }
                for slot in self.slot_sequence
            ],
            "discourse_links": [list(item.stable_key())
                                for item in self.discourse_links],
            "scope_and_time": list(self.scope_and_time),
            "evidence_refs": [list(item) for item in self.evidence_refs],
            "style_ref": (list(self.style_ref.stable_key())
                          if self.style_ref is not None else []),
            "carrier_ref": (list(self.carrier_ref.stable_key())
                            if self.carrier_ref is not None else []),
            "realization_candidates": [
                {
                    "surface": item.surface,
                    "frame_proposition": list(
                        item.frame_proposition.stable_key()),
                    "frame_source_hash": item.frame_source_hash,
                    "slot_count": item.slot_count,
                }
                for item in self.realization_candidates
            ],
            "stable_key": list(self.stable_key()),
        }


__all__ = [
    "FILLER_NODE_KINDS",
    "RESPONSE_PLAN_ACT_DOMAIN",
    "RESPONSE_PLAN_SLOT_TOKEN_DOMAIN",
    "ROLE_NODE_KINDS",
    "ResponsePlan",
    "ResponseRealization",
    "ResponseSlot",
    "response_act_identity",
]
