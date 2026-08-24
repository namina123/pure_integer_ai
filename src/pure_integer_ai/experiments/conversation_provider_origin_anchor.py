"""DLG-RAW-11A：同次 provider proof 的可迁移来源锚点。

本模块故意分成两个边界。``provider_origin_legacy_proof_from_same_dispatch_v1``
只在 Python host adapter 中读取同一次 W03-W05 dispatch 留下的 legacy
对象，并立即投影为有限整数 record。其余函数只消费明确的整数、u8[] 与
版本化 record；它们不读取路径、不查询 runtime、不从回答文本反推来源，也
不写会话状态。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1,
    PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1,
    PUBLIC_PROOF_SENTENCE_PROVIDER_RESULT_DOMAIN_V1,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
    PublicProofSentenceProviderResultV1,
    PublicProofSentenceProviderV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    ConversationRawIntakeError,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQASameDispatchProofProjection,
)


PROVIDER_ORIGIN_ANCHOR_RECORD_V1 = 1
PROVIDER_ORIGIN_LEGACY_PROOF_RECORD_V1 = 1
PROVIDER_ORIGIN_PROVIDER_BINDING_RECORD_V1 = 1
PROVIDER_ORIGIN_ROLE_BINDING_RECORD_V1 = 1
PROVIDER_ORIGIN_OCCURRENCE_RECORD_V1 = 1
PROVIDER_ORIGIN_RELATION_ENUM_RECORD_V1 = 1
PROVIDER_ORIGIN_ANCHOR_SCHEMA_RECORD_V1 = 1

PROVIDER_ORIGIN_ANCHOR_STATUS_NONE = 0
PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER = 1
PROVIDER_ORIGIN_PROVIDER_KIND_NONE = 0
PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05 = (
    PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1)

# 当前公开 W03-W05 proof 唯一可被投影的结构关系：一个 Proposition 中已
# 验证的 RoleBinding 指向对应 filler occurrence。它不是表层谓词文本枚举。
PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1 = 1

PROVIDER_ORIGIN_TYPED_STATUS_NONE = 0
PROVIDER_ORIGIN_TYPED_STATUS_ANSWER = 1
PROVIDER_ORIGIN_CANDIDATE_STATUS_UNKNOWN = 0
PROVIDER_ORIGIN_CANDIDATE_STATUS_UNIQUE = 1
PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_UNKNOWN = 0
PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_ACTIVE = 1
PROVIDER_ORIGIN_REASONING_UNKNOWN = 0
PROVIDER_ORIGIN_REASONING_AUTHORIZED = 1
PROVIDER_ORIGIN_GENERATION_UNKNOWN = 0
PROVIDER_ORIGIN_GENERATION_READY = 1

PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/CATALOG-RECORD/V1")
PROVIDER_ORIGIN_INPUT_INTAKE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/INPUT-INTAKE/V1")
PROVIDER_ORIGIN_OUTPUT_READBACK_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/OUTPUT-READBACK/V1")
PROVIDER_ORIGIN_ANCHOR_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/ANCHOR/V1")
PROVIDER_ORIGIN_RELATION_ENUM_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/RELATION-ENUM/V1")


# object-model: exception; interop=DLG-RAW-11
class ProviderOriginAnchorError(ValueError):
    """DLG-RAW-11 整数 record 或 host proof carrier 不满足闭锁合同。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以显式 count 写入一个有限非负整数段。"""
    result.extend((len(value), *value))


def _strict_vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证不依赖 Python 序列化的有序非负整数 vector。"""
    if (type(value) is not tuple
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise ProviderOriginAnchorError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证本合同中不可为空的稳定整数 key。"""
    return _strict_vector(value, label=label, allow_empty=False)


def _u8(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证显式 raw u8[]，不让 Python bytes 定义 record 语义。"""
    result = _strict_vector(value, label=label, allow_empty=allow_empty)
    if any(item > 255 for item in result):
        raise ProviderOriginAnchorError(f"{label} 含非 u8 整数")
    return result


def _u8_sha256(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证以 raw 32-byte 形式保存的 SHA-256 identity。"""
    result = _u8(value, label=label, allow_empty=False)
    if len(result) != 32:
        raise ProviderOriginAnchorError(f"{label} 必须为 raw 32-byte SHA-256")
    return result


def _strict_nonnegative(value: int, *, label: str) -> int:
    """拒绝 bool 与整数子类，固定协议整数的边界。"""
    if type(value) is not int or value < 0:
        raise ProviderOriginAnchorError(f"{label} 必须是非负严格整数")
    return value


def _sha256_raw_u8_from_ascii(value: str, *, label: str) -> tuple[int, ...]:
    """host adapter 将 legacy 小写 SHA 文本转为 raw 32-byte identity。"""
    if (type(value) is not str or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise ProviderOriginAnchorError(f"{label} 不是小写 SHA-256")
    result: list[int] = []
    for cursor in range(0, 64, 2):
        high = ord(value[cursor])
        low = ord(value[cursor + 1])
        high_value = high - 0x30 if high <= 0x39 else high - 0x61 + 10
        low_value = low - 0x30 if low <= 0x39 else low - 0x61 + 10
        result.append((high_value << 4) | low_value)
    return tuple(result)


def _host_text_to_scalars(value: str, *, label: str) -> tuple[int, ...]:
    """host 边缘把 legacy Python 文本复制为显式 Unicode scalar vector。"""
    if type(value) is not str or not value:
        raise ProviderOriginAnchorError(f"{label} 必须是非空 Python 文本")
    result = tuple(ord(item) for item in value)
    # ``encode_utf8_v1`` 同时拒绝 surrogate，确保 carrier 只持有可迁移 scalar。
    encode_utf8_v1(result)
    return result


def _identity(domain: bytes, record: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """使用已冻结 portable SHA framing 计算 raw identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise ProviderOriginAnchorError(f"{label} 无法形成") from error


def provider_origin_relation_enum_record_v1() -> tuple[int, ...]:
    """导出当前已登记 relation enum；未知关系不可由字符串隐式加入。"""
    return (
        PROVIDER_ORIGIN_RELATION_ENUM_RECORD_V1,
        PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1,
    )


def provider_origin_relation_enum_identity_v1() -> tuple[int, ...]:
    """返回 future runtime binding 可消费的 relation enum raw identity。"""
    return _identity(
        PROVIDER_ORIGIN_RELATION_ENUM_IDENTITY_DOMAIN_V1,
        provider_origin_relation_enum_record_v1(),
        label="provider-origin relation enum identity",
    )


def provider_origin_anchor_schema_record_v1() -> tuple[int, ...]:
    """导出 anchor payload 的冻结 record 布局与枚举，供 V3 runtime binding 固定。

    这个 schema record 不携带某次 provider 的数据。它只声明可被
    ``ProviderOriginAnchorProjectionV1`` 接受的 record revision、固定字段顺序
    和状态枚举，避免 runtime binding 仅依赖一条恰好通过的 anchor 样本。
    """
    return (
        PROVIDER_ORIGIN_ANCHOR_SCHEMA_RECORD_V1,
        PROVIDER_ORIGIN_ANCHOR_RECORD_V1,
        PROVIDER_ORIGIN_LEGACY_PROOF_RECORD_V1,
        PROVIDER_ORIGIN_PROVIDER_BINDING_RECORD_V1,
        PROVIDER_ORIGIN_ROLE_BINDING_RECORD_V1,
        PROVIDER_ORIGIN_OCCURRENCE_RECORD_V1,
        # anchor body：3 个头字段、14 个前段、relation、5 个后段、2 个
        # span scalar 与 4 个尾段；self identity 是最终一个长度前缀段。
        3,
        14,
        1,
        5,
        2,
        4,
        1,
        PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
        PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
        PROVIDER_ORIGIN_PROVIDER_KIND_NONE,
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        PROVIDER_ORIGIN_TYPED_STATUS_NONE,
        PROVIDER_ORIGIN_TYPED_STATUS_ANSWER,
        PROVIDER_ORIGIN_CANDIDATE_STATUS_UNKNOWN,
        PROVIDER_ORIGIN_CANDIDATE_STATUS_UNIQUE,
        PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_UNKNOWN,
        PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_ACTIVE,
        PROVIDER_ORIGIN_REASONING_UNKNOWN,
        PROVIDER_ORIGIN_REASONING_AUTHORIZED,
        PROVIDER_ORIGIN_GENERATION_UNKNOWN,
        PROVIDER_ORIGIN_GENERATION_READY,
    )


def _relation_registered(value: int) -> bool:
    """仅允许已有 W03-W05 typed proof 支持的冻结关系。"""
    return value == PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1


# object-model: value; representation=struct; interop=DLG-RAW-11
@dataclass(frozen=True, slots=True)
class ProviderOriginRoleBindingV1:
    """一个显式 RoleBinding，不借角色表层名称表达关系。"""

    binding_key: tuple[int, ...]
    role_key: tuple[int, ...]
    filler_key: tuple[int, ...]
    ordinal: int

    def __post_init__(self) -> None:
        """固定每个 key 和来自 candidate 的原始 ordinal。"""
        object.__setattr__(self, "binding_key", _key(
            self.binding_key, label="provider-origin binding key"))
        object.__setattr__(self, "role_key", _key(
            self.role_key, label="provider-origin role key"))
        object.__setattr__(self, "filler_key", _key(
            self.filler_key, label="provider-origin filler key"))
        object.__setattr__(self, "ordinal", _strict_nonnegative(
            self.ordinal, label="provider-origin binding ordinal"))

    def canonical_record(self) -> tuple[int, ...]:
        """导出单个 binding 的自描述整数 record。"""
        result = [PROVIDER_ORIGIN_ROLE_BINDING_RECORD_V1, self.ordinal]
        for value in (self.binding_key, self.role_key, self.filler_key):
            _pack(result, value)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11
@dataclass(frozen=True, slots=True)
class ProviderOriginOccurrenceV1:
    """一个可审计 occurrence，保留对象 key、原始顺序与整数 span。"""

    occurrence_key: tuple[int, ...]
    semantic_object_key: tuple[int, ...]
    ordinal: int
    start: int
    end: int

    def __post_init__(self) -> None:
        """拒绝 span、key 或 ordinal 非规范的 occurrence。"""
        object.__setattr__(self, "occurrence_key", _key(
            self.occurrence_key, label="provider-origin occurrence key"))
        object.__setattr__(self, "semantic_object_key", _key(
            self.semantic_object_key,
            label="provider-origin occurrence semantic object key"))
        ordinal = _strict_nonnegative(
            self.ordinal, label="provider-origin occurrence ordinal")
        start = _strict_nonnegative(
            self.start, label="provider-origin occurrence start")
        end = _strict_nonnegative(
            self.end, label="provider-origin occurrence end")
        if end <= start:
            raise ProviderOriginAnchorError("provider-origin occurrence span 非法")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def canonical_record(self) -> tuple[int, ...]:
        """导出单个 occurrence 的自描述整数 record。"""
        result = [
            PROVIDER_ORIGIN_OCCURRENCE_RECORD_V1,
            self.ordinal,
            self.start,
            self.end,
        ]
        for value in (self.occurrence_key, self.semantic_object_key):
            _pack(result, value)
        return tuple(result)


def _ordered_role_binding_record(
        bindings: tuple[ProviderOriginRoleBindingV1, ...],
        ) -> tuple[int, ...]:
    """保留 candidate 原始次序编码完整 RoleBinding 序列。"""
    result = [len(bindings)]
    for item in bindings:
        _pack(result, item.binding_key)
        _pack(result, item.role_key)
        _pack(result, item.filler_key)
        result.append(item.ordinal)
    return tuple(result)


def _ordered_occurrence_record(
        occurrences: tuple[ProviderOriginOccurrenceV1, ...],
        ) -> tuple[int, ...]:
    """保留 candidate 原始次序编码完整 occurrence 序列。"""
    result = [len(occurrences)]
    for item in occurrences:
        _pack(result, item.occurrence_key)
        _pack(result, item.semantic_object_key)
        result.extend((item.ordinal, item.start, item.end))
    return tuple(result)


def _validate_ordered_bindings(
        value: tuple[ProviderOriginRoleBindingV1, ...],
        *,
        label: str,
        ) -> tuple[ProviderOriginRoleBindingV1, ...]:
    """拒绝重复 binding/role identity 与空的 carrier binding 集。

    legacy W05 ``ordinal`` 是各来源对象的局部字段，并不保证在一个
    Proposition 内连续或唯一；可审计的全序由后续 generation role-key
    sequence 闭合，不能把 Python 方便的连续编号假设写进语义。
    """
    if (type(value) is not tuple or not value
            or any(type(item) is not ProviderOriginRoleBindingV1
                   for item in value)):
        raise ProviderOriginAnchorError(f"{label} 必须是非空 RoleBinding tuple")
    binding_keys = tuple(item.binding_key for item in value)
    role_keys = tuple(item.role_key for item in value)
    if len(set(binding_keys)) != len(binding_keys) or len(set(role_keys)) != len(role_keys):
        raise ProviderOriginAnchorError(f"{label} 含重复 binding 或 role key")
    return value


def _validate_ordered_occurrences(
        value: tuple[ProviderOriginOccurrenceV1, ...],
        *,
        label: str,
        ) -> tuple[ProviderOriginOccurrenceV1, ...]:
    """拒绝重复 occurrence identity 与空的 carrier occurrence 集。

    occurrence 的 source-local ordinal 同样不是 Proposition 内的全序；
    其顺序由 generation occurrence-key sequence 单独回链。
    """
    if (type(value) is not tuple or not value
            or any(type(item) is not ProviderOriginOccurrenceV1
                   for item in value)):
        raise ProviderOriginAnchorError(f"{label} 必须是非空 occurrence tuple")
    keys = tuple(item.occurrence_key for item in value)
    if len(set(keys)) != len(keys):
        raise ProviderOriginAnchorError(f"{label} 含重复 occurrence key")
    return value


def _key_sequence(
        value: tuple[tuple[int, ...], ...],
        *,
        label: str,
        ) -> tuple[tuple[int, ...], ...]:
    """验证保序 nested key sequence，禁止 host map 参与排序。"""
    if type(value) is not tuple or not value:
        raise ProviderOriginAnchorError(f"{label} 必须是非空 key tuple")
    result = tuple(_key(item, label=f"{label}[{ordinal}]")
                   for ordinal, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ProviderOriginAnchorError(f"{label} 含重复 key")
    return result


# object-model: value; representation=struct; interop=DLG-RAW-11
@dataclass(frozen=True, slots=True)
class ProviderOriginLegacyProofCarrierV1:
    """host adapter 已读取的同次 typed proof 的全整数投影。

    此 carrier 可表示漂移的宿主输入，以便纯核心统一返回 ``ANCHOR_NONE``；
    它不以自身构造成功代表 proof 已获准。所有跨字段链路均由
    :func:`project_provider_origin_anchor_v1` 重新闭合。
    """

    typed_status: int
    candidate_status: int
    candidate_active: int
    candidate_lifecycle: int
    candidate_reasoning: int
    generation_status: int
    source_record_key: tuple[int, ...]
    source_ref_stable_key: tuple[int, ...]
    source_commitment_u8: tuple[int, ...]
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    candidate_source_record_key: tuple[int, ...]
    candidate_source_ref_stable_key: tuple[int, ...]
    candidate_source_commitment_u8: tuple[int, ...]
    candidate_proposition_key: tuple[int, ...]
    candidate_predicate_key: tuple[int, ...]
    candidate_context_key: tuple[int, ...]
    relation_kind_code: int
    generation_construction_key: tuple[int, ...]
    generation_target_proposition_key: tuple[int, ...]
    generation_target_predicate_key: tuple[int, ...]
    generation_target_source_ref_stable_key: tuple[int, ...]
    generation_target_source_commitment_u8: tuple[int, ...]
    generation_context_key: tuple[int, ...]
    generation_role_binding_keys: tuple[tuple[int, ...], ...]
    generation_occurrence_keys: tuple[tuple[int, ...], ...]
    focus_role_binding_key: tuple[int, ...]
    focus_role_key: tuple[int, ...]
    focus_filler_key: tuple[int, ...]
    focus_occurrence_key: tuple[int, ...]
    focus_answer_start: int
    focus_answer_end: int
    ordered_role_bindings: tuple[ProviderOriginRoleBindingV1, ...]
    ordered_occurrences: tuple[ProviderOriginOccurrenceV1, ...]
    generated_proposition_scalars: tuple[int, ...]
    generated_proposition_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """冻结 carrier 的结构合法性，但把跨字段资格留给纯投影阶段。"""
        for name in (
                "typed_status", "candidate_status", "candidate_active",
                "candidate_lifecycle", "candidate_reasoning",
                "generation_status", "relation_kind_code",
                "focus_answer_start", "focus_answer_end"):
            object.__setattr__(self, name, _strict_nonnegative(
                getattr(self, name), label=f"provider-origin carrier {name}"))
        for name in (
                "source_record_key", "source_ref_stable_key",
                "w03_observation_key", "w04_observation_key",
                "w05_observation_key", "proposition_key", "predicate_key",
                "candidate_source_record_key",
                "candidate_source_ref_stable_key",
                "candidate_proposition_key", "candidate_predicate_key",
                "candidate_context_key", "generation_construction_key",
                "generation_target_proposition_key",
                "generation_target_predicate_key",
                "generation_target_source_ref_stable_key",
                "generation_context_key", "focus_role_binding_key",
                "focus_role_key", "focus_filler_key", "focus_occurrence_key"):
            object.__setattr__(self, name, _key(
                getattr(self, name), label=f"provider-origin carrier {name}"))
        for name in (
                "source_commitment_u8", "candidate_source_commitment_u8",
                "generation_target_source_commitment_u8"):
            object.__setattr__(self, name, _u8_sha256(
                getattr(self, name), label=f"provider-origin carrier {name}"))
        bindings = _validate_ordered_bindings(
            self.ordered_role_bindings,
            label="provider-origin carrier ordered role bindings",
        )
        occurrences = _validate_ordered_occurrences(
            self.ordered_occurrences,
            label="provider-origin carrier ordered occurrences",
        )
        scalars = _strict_vector(
            self.generated_proposition_scalars,
            label="provider-origin carrier generated scalars",
            allow_empty=False,
        )
        if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
               for item in scalars):
            raise ProviderOriginAnchorError(
                "provider-origin carrier generated scalar 非法")
        output = _u8(
            self.generated_proposition_u8,
            label="provider-origin carrier generated u8",
            allow_empty=False,
        )
        if encode_utf8_v1(scalars) != output:
            raise ProviderOriginAnchorError(
                "provider-origin carrier generated UTF-8 漂移")
        if self.focus_answer_end <= self.focus_answer_start:
            raise ProviderOriginAnchorError("provider-origin carrier focus span 非法")
        object.__setattr__(self, "ordered_role_bindings", bindings)
        object.__setattr__(self, "ordered_occurrences", occurrences)
        object.__setattr__(self, "generation_role_binding_keys", _key_sequence(
            self.generation_role_binding_keys,
            label="provider-origin carrier generation role binding keys"))
        object.__setattr__(self, "generation_occurrence_keys", _key_sequence(
            self.generation_occurrence_keys,
            label="provider-origin carrier generation occurrence keys"))
        object.__setattr__(self, "generated_proposition_scalars", scalars)
        object.__setattr__(self, "generated_proposition_u8", output)

    def canonical_record(self) -> tuple[int, ...]:
        """导出 host text 与 legacy object 均已排除的完整 proof record。"""
        result = [
            PROVIDER_ORIGIN_LEGACY_PROOF_RECORD_V1,
            self.typed_status,
            self.candidate_status,
            self.candidate_active,
            self.candidate_lifecycle,
            self.candidate_reasoning,
            self.generation_status,
            self.relation_kind_code,
            self.focus_answer_start,
            self.focus_answer_end,
        ]
        for value in (
                self.source_record_key,
                self.source_ref_stable_key,
                self.source_commitment_u8,
                self.w03_observation_key,
                self.w04_observation_key,
                self.w05_observation_key,
                self.proposition_key,
                self.predicate_key,
                self.candidate_source_record_key,
                self.candidate_source_ref_stable_key,
                self.candidate_source_commitment_u8,
                self.candidate_proposition_key,
                self.candidate_predicate_key,
                self.candidate_context_key,
                self.generation_construction_key,
                self.generation_target_proposition_key,
                self.generation_target_predicate_key,
                self.generation_target_source_ref_stable_key,
                self.generation_target_source_commitment_u8,
                self.generation_context_key,
                self.focus_role_binding_key,
                self.focus_role_key,
                self.focus_filler_key,
                self.focus_occurrence_key,
                _ordered_role_binding_record(self.ordered_role_bindings),
                _ordered_occurrence_record(self.ordered_occurrences),
                self.generated_proposition_scalars,
                self.generated_proposition_u8):
            _pack(result, value)
        for keys in (
                self.generation_role_binding_keys,
                self.generation_occurrence_keys):
            nested = [len(keys)]
            for key in keys:
                _pack(nested, key)
            _pack(result, tuple(nested))
        return tuple(result)


def _legacy_status_code(value: object, *, expected: str, yes: int) -> int:
    """host adapter 将已验证 legacy status 收敛为冻结整数，不透传文本。"""
    return yes if value == expected else 0


def _find_same_generation_option(raw_result, proof, candidate):
    """读取同次 typed result 的等价 generation construction。

    已验证的旧回答链允许多个 construction-source option 在目标 Proposition、
    结构角色和 occurrence 序上完全等价，并在回答侧按功能等价裁决。这里不把
    这种已验证等价误判为候选歧义；但只要它们在本锚点实际保留的任一 binding
    字段上分叉，就拒绝 carrier。
    """
    vertical = raw_result.typed_result.vertical_result
    w05 = vertical.w04_w05.w05_result
    options = tuple(
        item for item in w05.generation_options
        if item.construction_key == proof.generation_construction_key
        and item.target_proposition_key == proof.proposition_key
        and item.target_predicate_key == proof.predicate_key
        and item.target_source_ref_key == proof.source_ref_key
        and item.target_source_commitment == proof.source_commitment
        and item.context_key == candidate.context_key
        and item.occurrence_order == candidate.occurrence_order
        and item.role_binding_keys == tuple(
            binding.identity_key for binding in candidate.role_bindings)
    )
    if not options:
        return None
    retained = tuple(
        (item.construction_key, item.target_proposition_key,
         item.target_predicate_key, item.target_source_ref_key,
         item.target_source_commitment, item.context_key,
         item.occurrence_order, item.role_binding_keys)
        for item in options
    )
    return options[0] if all(item == retained[0] for item in retained) else None


def provider_origin_legacy_proof_from_same_dispatch_v1(
        projection: SparseQASameDispatchProofProjection,
        *,
        relation_kind_code: int = (
            PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1),
        ) -> ProviderOriginLegacyProofCarrierV1 | None:
    """把一次 dispatch 保留的 typed proof 降解为整数 carrier。

    本函数只读取 ``SparseQASameDispatchProofProjection`` 已持有的对象，不调用
    sparse runtime、provider 或任何文本匹配入口。不能证明同次链路时返回
    ``None``，交由后续调用者产出 ``ANCHOR_NONE``。
    """
    if type(projection) is not SparseQASameDispatchProofProjection:
        return None
    try:
        raw = projection.raw_result
        proof = projection.typed_proof
        if (projection.query_result.status != "ANSWER"
                or raw is None or raw.status != "ANSWER"
                or raw.typed_result is None
                or raw.typed_result.status != "ANSWER"
                or proof is None
                or raw.typed_result.proof is not proof
                or projection.generated_proposition_surface
                != proof.generated_proposition_surface):
            return None
        vertical = raw.typed_result.vertical_result
        if vertical.status != "BRIDGED" or vertical.link is None:
            return None
        link = vertical.link
        if (proof.source_record_key != link.source_ref_key
                or proof.source_commitment != link.source_commitment
                or proof.w03_observation_key != link.w03_observation_key
                or proof.w04_observation_key != link.w04_observation_key
                or proof.w05_observation_key != link.w05_observation_key
                or proof.proposition_key != link.proposition_key
                or proof.predicate_key != link.predicate_key):
            return None
        w05 = vertical.w04_w05.w05_result
        candidates = tuple(
            item for item in w05.candidates
            if item.proposition_key == proof.proposition_key)
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        if (candidate.source_record_key != proof.source_record_key
                or candidate.source_ref_key != proof.source_ref_key
                or candidate.source_commitment != proof.source_commitment
                or candidate.proposition_key != proof.proposition_key
                or candidate.predicate_key != proof.predicate_key):
            return None
        generation = _find_same_generation_option(raw, proof, candidate)
        if generation is None:
            return None
        scalars = _host_text_to_scalars(
            proof.generated_proposition_surface,
            label="same-dispatch generated proposition",
        )
        role_bindings = tuple(
            ProviderOriginRoleBindingV1(
                item.identity_key,
                item.role_key,
                item.filler_key,
                item.ordinal,
            )
            for item in candidate.role_bindings
        )
        occurrences = tuple(
            ProviderOriginOccurrenceV1(
                item.identity_key,
                item.semantic_object_key,
                item.ordinal,
                item.start,
                item.end,
            )
            for item in candidate.occurrences
        )
        return ProviderOriginLegacyProofCarrierV1(
            _legacy_status_code(raw.typed_result.status,
                                expected="ANSWER",
                                yes=PROVIDER_ORIGIN_TYPED_STATUS_ANSWER),
            _legacy_status_code(w05.status,
                                expected="UNIQUE",
                                yes=PROVIDER_ORIGIN_CANDIDATE_STATUS_UNIQUE),
            candidate.active,
            _legacy_status_code(candidate.lifecycle_status,
                                expected="ACTIVE",
                                yes=PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_ACTIVE),
            _legacy_status_code(candidate.reasoning_status,
                                expected="AUTHORIZED",
                                yes=PROVIDER_ORIGIN_REASONING_AUTHORIZED),
            _legacy_status_code(w05.generation_status,
                                expected="READY",
                                yes=PROVIDER_ORIGIN_GENERATION_READY),
            proof.source_record_key,
            proof.source_ref_key,
            _sha256_raw_u8_from_ascii(
                proof.source_commitment,
                label="same-dispatch source commitment",
            ),
            proof.w03_observation_key,
            proof.w04_observation_key,
            proof.w05_observation_key,
            proof.proposition_key,
            proof.predicate_key,
            candidate.source_record_key,
            candidate.source_ref_key,
            _sha256_raw_u8_from_ascii(
                candidate.source_commitment,
                label="same-dispatch candidate source commitment",
            ),
            candidate.proposition_key,
            candidate.predicate_key,
            candidate.context_key,
            relation_kind_code,
            generation.construction_key,
            generation.target_proposition_key,
            generation.target_predicate_key,
            generation.target_source_ref_key,
            _sha256_raw_u8_from_ascii(
                generation.target_source_commitment,
                label="same-dispatch generation target commitment",
            ),
            generation.context_key,
            generation.role_binding_keys,
            generation.occurrence_order,
            proof.role_binding_key,
            proof.role_key,
            proof.filler_key,
            proof.answer_occurrence_key,
            proof.answer_start,
            proof.answer_end,
            role_bindings,
            occurrences,
            scalars,
            encode_utf8_v1(scalars),
        )
    except (AttributeError, ConversationRawIntakeError,
            ProviderOriginAnchorError, TypeError, ValueError):
        return None


# object-model: value; representation=struct; interop=DLG-RAW-11
@dataclass(frozen=True, slots=True)
class ProviderOriginProviderBindingV1:
    """当前 provider/runtime/catalog 的不可替换整数 binding。"""

    provider_kind: int
    provider_identity_u8: tuple[int, ...]
    runtime_identity_u8: tuple[int, ...]
    catalog_record: tuple[int, ...]
    catalog_record_identity_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 catalog 完整 record 及其 raw SHA identity。"""
        if (type(self.provider_kind) is not int
                or self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05):
            raise ProviderOriginAnchorError("provider-origin provider kind 未注册")
        provider = _u8_sha256(
            self.provider_identity_u8, label="provider-origin provider identity")
        runtime = _u8_sha256(
            self.runtime_identity_u8, label="provider-origin runtime identity")
        catalog = _strict_vector(
            self.catalog_record,
            label="provider-origin catalog record",
            allow_empty=False,
        )
        catalog_identity = _u8_sha256(
            self.catalog_record_identity_u8,
            label="provider-origin catalog record identity",
        )
        expected = _identity(
            PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
            catalog,
            label="provider-origin catalog record identity",
        )
        if catalog_identity != expected:
            raise ProviderOriginAnchorError("provider-origin catalog identity 漂移")
        object.__setattr__(self, "provider_identity_u8", provider)
        object.__setattr__(self, "runtime_identity_u8", runtime)
        object.__setattr__(self, "catalog_record", catalog)
        object.__setattr__(self, "catalog_record_identity_u8", catalog_identity)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含 provider Python resource owner 的 binding record。"""
        result = [
            PROVIDER_ORIGIN_PROVIDER_BINDING_RECORD_V1,
            self.provider_kind,
        ]
        for value in (
                self.provider_identity_u8,
                self.runtime_identity_u8,
                self.catalog_record,
                self.catalog_record_identity_u8):
            _pack(result, value)
        return tuple(result)


def provider_origin_provider_binding_from_public_provider_v1(
        provider: PublicProofSentenceProviderV1,
        ) -> ProviderOriginProviderBindingV1:
    """host adapter 从已构造的 DLG-RAW-10 provider 读取冻结 binding。"""
    if type(provider) is not PublicProofSentenceProviderV1:
        raise TypeError("provider-origin binding 需要 PublicProofSentenceProviderV1")
    catalog = provider.catalog_record
    return ProviderOriginProviderBindingV1(
        PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05,
        provider.provider_identity,
        provider.runtime_identity,
        catalog,
        _identity(
            PROVIDER_ORIGIN_CATALOG_IDENTITY_DOMAIN_V1,
            catalog,
            label="provider-origin catalog record identity",
        ),
    )


def _anchor_body(value: "ProviderOriginAnchorProjectionV1") -> tuple[int, ...]:
    """写出不含 self identity 的 anchor canonical body。"""
    result = [
        PROVIDER_ORIGIN_ANCHOR_RECORD_V1,
        value.anchor_status,
        value.provider_kind,
    ]
    for item in (
            value.provider_identity_u8,
            value.runtime_identity_u8,
            value.catalog_record_identity_u8,
            value.provider_result_identity_u8,
            value.input_intake_identity_u8,
            value.output_readback_identity_u8,
            value.source_record_key,
            value.source_ref_stable_key,
            value.source_commitment_u8,
            value.w03_observation_key,
            value.w04_observation_key,
            value.w05_observation_key,
            value.proposition_key,
            value.predicate_key):
        _pack(result, item)
    result.append(value.relation_kind_code)
    for item in (
            value.generation_construction_key,
            value.focus_role_binding_key,
            value.focus_role_key,
            value.focus_filler_key,
            value.focus_occurrence_key):
        _pack(result, item)
    result.extend((value.focus_answer_start, value.focus_answer_end))
    for item in (
            _ordered_role_binding_record(value.ordered_role_bindings),
            _ordered_occurrence_record(value.ordered_occurrences),
            value.output_scalars,
            value.output_u8):
        _pack(result, item)
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-11
@dataclass(frozen=True, slots=True)
class ProviderOriginAnchorProjectionV1:
    """同次 provider proof 的来源锚点；仅有 ``ANSWER`` 才可被后续 admission 消费。"""

    anchor_status: int
    provider_kind: int = PROVIDER_ORIGIN_PROVIDER_KIND_NONE
    provider_identity_u8: tuple[int, ...] = ()
    runtime_identity_u8: tuple[int, ...] = ()
    catalog_record_identity_u8: tuple[int, ...] = ()
    provider_result_identity_u8: tuple[int, ...] = ()
    input_intake_identity_u8: tuple[int, ...] = ()
    output_readback_identity_u8: tuple[int, ...] = ()
    source_record_key: tuple[int, ...] = ()
    source_ref_stable_key: tuple[int, ...] = ()
    source_commitment_u8: tuple[int, ...] = ()
    w03_observation_key: tuple[int, ...] = ()
    w04_observation_key: tuple[int, ...] = ()
    w05_observation_key: tuple[int, ...] = ()
    proposition_key: tuple[int, ...] = ()
    predicate_key: tuple[int, ...] = ()
    relation_kind_code: int = 0
    generation_construction_key: tuple[int, ...] = ()
    focus_role_binding_key: tuple[int, ...] = ()
    focus_role_key: tuple[int, ...] = ()
    focus_filler_key: tuple[int, ...] = ()
    focus_occurrence_key: tuple[int, ...] = ()
    focus_answer_start: int = 0
    focus_answer_end: int = 0
    ordered_role_bindings: tuple[ProviderOriginRoleBindingV1, ...] = ()
    ordered_occurrences: tuple[ProviderOriginOccurrenceV1, ...] = ()
    output_scalars: tuple[int, ...] = ()
    output_u8: tuple[int, ...] = ()
    anchor_identity_u8: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """冻结 ANSWER 全字段或 NONE 零 payload，且 self identity 必须可重算。"""
        if type(self.anchor_status) is not int or self.anchor_status not in {
                PROVIDER_ORIGIN_ANCHOR_STATUS_NONE,
                PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER}:
            raise ProviderOriginAnchorError("provider-origin anchor status 未注册")
        if type(self.provider_kind) is not int or self.provider_kind not in {
                PROVIDER_ORIGIN_PROVIDER_KIND_NONE,
                PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05}:
            raise ProviderOriginAnchorError("provider-origin anchor provider kind 未注册")
        for name in ("relation_kind_code", "focus_answer_start", "focus_answer_end"):
            object.__setattr__(self, name, _strict_nonnegative(
                getattr(self, name), label=f"provider-origin anchor {name}"))
        if self.anchor_status == PROVIDER_ORIGIN_ANCHOR_STATUS_NONE:
            if self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_NONE:
                raise ProviderOriginAnchorError("ANCHOR_NONE 不得携带 provider kind")
            payload = (
                self.provider_identity_u8, self.runtime_identity_u8,
                self.catalog_record_identity_u8,
                self.provider_result_identity_u8,
                self.input_intake_identity_u8,
                self.output_readback_identity_u8, self.source_record_key,
                self.source_ref_stable_key, self.source_commitment_u8,
                self.w03_observation_key, self.w04_observation_key,
                self.w05_observation_key, self.proposition_key,
                self.predicate_key, self.generation_construction_key,
                self.focus_role_binding_key, self.focus_role_key,
                self.focus_filler_key, self.focus_occurrence_key,
                self.ordered_role_bindings, self.ordered_occurrences,
                self.output_scalars, self.output_u8)
            if (any(payload) or self.relation_kind_code != 0
                    or self.focus_answer_start != 0
                    or self.focus_answer_end != 0):
                raise ProviderOriginAnchorError("ANCHOR_NONE 不得携带可消费 payload")
        else:
            if self.provider_kind != PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05:
                raise ProviderOriginAnchorError("ANCHOR_ANSWER provider kind 漂移")
            for name in (
                    "provider_identity_u8", "runtime_identity_u8",
                    "catalog_record_identity_u8",
                    "provider_result_identity_u8",
                    "input_intake_identity_u8",
                    "output_readback_identity_u8", "source_commitment_u8"):
                object.__setattr__(self, name, _u8_sha256(
                    getattr(self, name), label=f"provider-origin anchor {name}"))
            for name in (
                    "source_record_key", "source_ref_stable_key",
                    "w03_observation_key", "w04_observation_key",
                    "w05_observation_key", "proposition_key", "predicate_key",
                    "generation_construction_key", "focus_role_binding_key",
                    "focus_role_key", "focus_filler_key", "focus_occurrence_key"):
                object.__setattr__(self, name, _key(
                    getattr(self, name), label=f"provider-origin anchor {name}"))
            if not _relation_registered(self.relation_kind_code):
                raise ProviderOriginAnchorError("ANCHOR_ANSWER relation 未注册")
            if self.focus_answer_end <= self.focus_answer_start:
                raise ProviderOriginAnchorError("ANCHOR_ANSWER focus span 非法")
            bindings = _validate_ordered_bindings(
                self.ordered_role_bindings,
                label="ANCHOR_ANSWER ordered role bindings",
            )
            occurrences = _validate_ordered_occurrences(
                self.ordered_occurrences,
                label="ANCHOR_ANSWER ordered occurrences",
            )
            scalars = _strict_vector(
                self.output_scalars,
                label="ANCHOR_ANSWER output scalars",
                allow_empty=False,
            )
            if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
                   for item in scalars):
                raise ProviderOriginAnchorError("ANCHOR_ANSWER output scalar 非法")
            output = _u8(
                self.output_u8,
                label="ANCHOR_ANSWER output u8",
                allow_empty=False,
            )
            if encode_utf8_v1(scalars) != output:
                raise ProviderOriginAnchorError("ANCHOR_ANSWER output UTF-8 漂移")
            object.__setattr__(self, "ordered_role_bindings", bindings)
            object.__setattr__(self, "ordered_occurrences", occurrences)
            object.__setattr__(self, "output_scalars", scalars)
            object.__setattr__(self, "output_u8", output)
        expected = _identity(
            PROVIDER_ORIGIN_ANCHOR_IDENTITY_DOMAIN_V1,
            _anchor_body(self),
            label="provider-origin anchor identity",
        )
        supplied = self.anchor_identity_u8
        if supplied:
            if _u8_sha256(
                    supplied,
                    label="provider-origin anchor identity") != expected:
                raise ProviderOriginAnchorError("provider-origin anchor identity 漂移")
        object.__setattr__(self, "anchor_identity_u8", expected)

    @property
    def accepted(self) -> bool:
        """只有可由后续 context admission 消费的来源锚点返回真。"""
        return self.anchor_status == PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER

    def canonical_record(self) -> tuple[int, ...]:
        """导出 complete anchor record；不存在 Python object 或文本语义。"""
        result = list(_anchor_body(self))
        _pack(result, self.anchor_identity_u8)
        return tuple(result)


def _none_anchor() -> ProviderOriginAnchorProjectionV1:
    """统一产生零 payload 的 canonical ``ANCHOR_NONE``。"""
    return ProviderOriginAnchorProjectionV1(PROVIDER_ORIGIN_ANCHOR_STATUS_NONE)


def _carrier_is_eligible(carrier: ProviderOriginLegacyProofCarrierV1) -> bool:
    """闭合 typed proof、candidate、generation 与 focus 的全部整数回链。"""
    if (carrier.typed_status != PROVIDER_ORIGIN_TYPED_STATUS_ANSWER
            or carrier.candidate_status != PROVIDER_ORIGIN_CANDIDATE_STATUS_UNIQUE
            or carrier.candidate_active != 1
            or carrier.candidate_lifecycle
            != PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_ACTIVE
            or carrier.candidate_reasoning
            != PROVIDER_ORIGIN_REASONING_AUTHORIZED
            or carrier.generation_status != PROVIDER_ORIGIN_GENERATION_READY
            or not _relation_registered(carrier.relation_kind_code)):
        return False
    if (carrier.candidate_source_record_key != carrier.source_record_key
            or carrier.candidate_source_ref_stable_key
            != carrier.source_ref_stable_key
            or carrier.candidate_source_commitment_u8
            != carrier.source_commitment_u8
            or carrier.candidate_proposition_key != carrier.proposition_key
            or carrier.candidate_predicate_key != carrier.predicate_key):
        return False
    if (carrier.generation_target_proposition_key != carrier.proposition_key
            or carrier.generation_target_predicate_key != carrier.predicate_key
            or carrier.generation_target_source_ref_stable_key
            != carrier.source_ref_stable_key
            or carrier.generation_target_source_commitment_u8
            != carrier.source_commitment_u8
            or carrier.generation_context_key != carrier.candidate_context_key):
        return False
    binding_keys = tuple(item.binding_key for item in carrier.ordered_role_bindings)
    occurrence_keys = tuple(item.occurrence_key for item in carrier.ordered_occurrences)
    if (carrier.generation_role_binding_keys != binding_keys
            or carrier.generation_occurrence_keys != occurrence_keys):
        return False
    bindings = tuple(
        item for item in carrier.ordered_role_bindings
        if item.binding_key == carrier.focus_role_binding_key
        and item.role_key == carrier.focus_role_key
        and item.filler_key == carrier.focus_filler_key)
    occurrences = tuple(
        item for item in carrier.ordered_occurrences
        if item.occurrence_key == carrier.focus_occurrence_key
        and item.semantic_object_key == carrier.focus_filler_key
        and item.start == carrier.focus_answer_start
        and item.end == carrier.focus_answer_end)
    return len(bindings) == 1 and len(occurrences) == 1


def project_provider_origin_anchor_v1(
        binding: ProviderOriginProviderBindingV1,
        verified_provider_result: PublicProofSentenceProviderResultV1,
        carrier: ProviderOriginLegacyProofCarrierV1 | None,
        ) -> ProviderOriginAnchorProjectionV1:
    """投影一条 provider ANSWER 到来源锚点，任何漂移均闭锁为 ``NONE``。

    ``verified_provider_result`` 的重放验证属于调用方的 DLG-RAW-10 边界；本函数
    明确不调用 provider/runtime，因而不会为锚点产生第二次 query。它只比较已
    验证 result、当前 binding 与同次 legacy proof carrier 的整数 record。
    """
    if (type(binding) is not ProviderOriginProviderBindingV1
            or type(verified_provider_result)
            is not PublicProofSentenceProviderResultV1
            or (carrier is not None
                and type(carrier) is not ProviderOriginLegacyProofCarrierV1)):
        return _none_anchor()
    if carrier is None:
        return _none_anchor()
    try:
        result = verified_provider_result
        if (result.provider_status != PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER
                or result.mapped_dlg_result_code != DLG_RAW_ACCEPT
                or result.context_policy
                != PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1
                or not result.intake.accepted
                or not result.demo_record
                or result.provider_identity != binding.provider_identity_u8
                or result.runtime_identity != binding.runtime_identity_u8
                or result.catalog_record != binding.catalog_record):
            return _none_anchor()
        if not _carrier_is_eligible(carrier):
            return _none_anchor()
        if (result.source_record_key != carrier.source_record_key
                or result.output_scalars != carrier.generated_proposition_scalars
                or result.output_bytes != carrier.generated_proposition_u8):
            return _none_anchor()
        output_readback = intake_raw_conversation_vector(result.output_bytes)
        if (not output_readback.accepted
                or output_readback.unicode_scalars != result.output_scalars):
            return _none_anchor()
        result_identity = _identity(
            PUBLIC_PROOF_SENTENCE_PROVIDER_RESULT_DOMAIN_V1,
            result.canonical_record(),
            label="provider-origin provider result identity",
        )
        input_identity = _identity(
            PROVIDER_ORIGIN_INPUT_INTAKE_IDENTITY_DOMAIN_V1,
            result.intake.canonical_record(),
            label="provider-origin input intake identity",
        )
        readback_identity = _identity(
            PROVIDER_ORIGIN_OUTPUT_READBACK_IDENTITY_DOMAIN_V1,
            output_readback.canonical_record(),
            label="provider-origin output readback identity",
        )
        return ProviderOriginAnchorProjectionV1(
            PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER,
            binding.provider_kind,
            binding.provider_identity_u8,
            binding.runtime_identity_u8,
            binding.catalog_record_identity_u8,
            result_identity,
            input_identity,
            readback_identity,
            carrier.source_record_key,
            carrier.source_ref_stable_key,
            carrier.source_commitment_u8,
            carrier.w03_observation_key,
            carrier.w04_observation_key,
            carrier.w05_observation_key,
            carrier.proposition_key,
            carrier.predicate_key,
            carrier.relation_kind_code,
            carrier.generation_construction_key,
            carrier.focus_role_binding_key,
            carrier.focus_role_key,
            carrier.focus_filler_key,
            carrier.focus_occurrence_key,
            carrier.focus_answer_start,
            carrier.focus_answer_end,
            carrier.ordered_role_bindings,
            carrier.ordered_occurrences,
            result.output_scalars,
            result.output_bytes,
        )
    except (ConversationRawIntakeError, ProviderOriginAnchorError,
            PublicSourcePayloadProviderError, TypeError, ValueError):
        return _none_anchor()


__all__ = [
    "PROVIDER_ORIGIN_ANCHOR_IDENTITY_DOMAIN_V1",
    "PROVIDER_ORIGIN_ANCHOR_RECORD_V1",
    "PROVIDER_ORIGIN_ANCHOR_SCHEMA_RECORD_V1",
    "PROVIDER_ORIGIN_ANCHOR_STATUS_ANSWER",
    "PROVIDER_ORIGIN_ANCHOR_STATUS_NONE",
    "PROVIDER_ORIGIN_CANDIDATE_LIFECYCLE_ACTIVE",
    "PROVIDER_ORIGIN_CANDIDATE_STATUS_UNIQUE",
    "PROVIDER_ORIGIN_GENERATION_READY",
    "PROVIDER_ORIGIN_LEGACY_PROOF_RECORD_V1",
    "PROVIDER_ORIGIN_PROVIDER_BINDING_RECORD_V1",
    "PROVIDER_ORIGIN_PROVIDER_KIND_W03_W05",
    "PROVIDER_ORIGIN_REASONING_AUTHORIZED",
    "PROVIDER_ORIGIN_RELATION_ENUM_RECORD_V1",
    "PROVIDER_ORIGIN_RELATION_PROPOSITION_ROLE_FILLER_V1",
    "PROVIDER_ORIGIN_TYPED_STATUS_ANSWER",
    "ProviderOriginAnchorError",
    "ProviderOriginAnchorProjectionV1",
    "ProviderOriginLegacyProofCarrierV1",
    "ProviderOriginOccurrenceV1",
    "ProviderOriginProviderBindingV1",
    "ProviderOriginRoleBindingV1",
    "project_provider_origin_anchor_v1",
    "provider_origin_legacy_proof_from_same_dispatch_v1",
    "provider_origin_anchor_schema_record_v1",
    "provider_origin_provider_binding_from_public_provider_v1",
    "provider_origin_relation_enum_identity_v1",
    "provider_origin_relation_enum_record_v1",
]
