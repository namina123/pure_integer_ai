"""Runtime 资料进入生成边界时使用的纯整数结构证据。

该模块只保存可由 Runtime observation、relation candidate、event 和
SourceRef 重建的结构身份。它不保存 digest 对应的表面文本，也不执行
事实推断；用户可见文字仍由 SourceRecord/既有回答组织路径提供。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.learning_input_capsule import digest_bytes
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


RUNTIME_MATERIAL_GENERATION_CONTEXT_PROTOCOL_V1 = 1
_RESPONSE_ACTS = frozenset({"ANSWER", "UNKNOWN", "CLARIFY"})


class RuntimeMaterialGenerationContextError(ValueError):
    """Runtime 结构证据上下文不满足生成边界。"""


def _key(value: tuple[int, ...], *, label: str, empty: bool = False) -> None:
    if (not isinstance(value, tuple) or (not empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise RuntimeMaterialGenerationContextError(
            f"{label} 必须是{'可空' if empty else '非空'}非负整数 tuple")


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    _key(value, label=label)
    result.extend((len(value), *value))


@dataclass(frozen=True, slots=True)
class RuntimeMaterialGenerationEvidence:
    """一个已资格化 Runtime candidate 的结构证据闭包。"""

    event_key: tuple[int, ...]
    memory_item_key: tuple[int, ...]
    source_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    structure_refs: tuple[tuple[int, int], ...]
    proposition_keys: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for value, label in (
                (self.event_key, "event_key"),
                (self.memory_item_key, "memory_item_key"),
                (self.source_key, "source_key"),
                (self.observation_key, "observation_key")):
            _key(value, label=label)
        if (not isinstance(self.structure_refs, tuple)
                or not self.structure_refs
                or any(not isinstance(item, tuple) or len(item) != 2
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in self.structure_refs)):
            raise RuntimeMaterialGenerationContextError(
                "structure_refs 必须是非空 ConceptRef tuple")
        for values, label in ((self.proposition_keys, "proposition_keys"),
                              (self.evidence_keys, "evidence_keys")):
            if not isinstance(values, tuple) or not values:
                raise RuntimeMaterialGenerationContextError(
                    f"{label} 必须是非空 tuple")
            for index, value in enumerate(values):
                _key(value, label=f"{label}[{index}]")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RUNTIME_MATERIAL_GENERATION_CONTEXT_PROTOCOL_V1]
        for value, label in (
                (self.event_key, "event_key"),
                (self.memory_item_key, "memory_item_key"),
                (self.source_key, "source_key"),
                (self.observation_key, "observation_key")):
            _pack(result, value, label=label)
        result.append(len(self.structure_refs))
        for ref in self.structure_refs:
            result.extend(ref)
        for values, label in ((self.proposition_keys, "proposition_keys"),
                              (self.evidence_keys, "evidence_keys")):
            result.append(len(values))
            for value in values:
                _pack(result, value, label=f"{label}[]")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeMaterialGenerationContext:
    """供回答生成侧消费的、来源守恒的 Runtime 结构上下文。"""

    response_act: str
    evidence: tuple[RuntimeMaterialGenerationEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.response_act) is not str or self.response_act not in _RESPONSE_ACTS:
            raise RuntimeMaterialGenerationContextError(
                "response_act 未注册")
        if (not isinstance(self.evidence, tuple) or not self.evidence
                or any(not isinstance(item, RuntimeMaterialGenerationEvidence)
                       for item in self.evidence)):
            raise RuntimeMaterialGenerationContextError(
                "generation context evidence 不能为空")
        identities = tuple((item.event_key, item.observation_key,
                            item.proposition_keys) for item in self.evidence)
        if identities != tuple(sorted(identities)):
            raise RuntimeMaterialGenerationContextError(
                "generation context evidence 未规范排序")

    def canonical_record(self) -> tuple[int, ...]:
        result = [RUNTIME_MATERIAL_GENERATION_CONTEXT_PROTOCOL_V1,
                  *map(ord, self.response_act), len(self.evidence)]
        for item in self.evidence:
            record = item.canonical_record()
            _pack(result, record, label="generation evidence")
        return tuple(result)

    @property
    def identity_key(self) -> tuple[int, ...]:
        return digest_bytes(encode_integer_tuple(self.canonical_record()))


def validate_runtime_material_generation_context(
        context: RuntimeMaterialGenerationContext,
        *,
        response_act: str,
        ) -> RuntimeMaterialGenerationContext:
    """在进入回答组织前核验 response-act 与结构证据一致。"""
    if not isinstance(context, RuntimeMaterialGenerationContext):
        raise RuntimeMaterialGenerationContextError(
            "generation context 类型错误")
    if type(response_act) is not str or response_act not in _RESPONSE_ACTS:
        raise RuntimeMaterialGenerationContextError(
            "response_act 未注册")
    if context.response_act != response_act:
        raise RuntimeMaterialGenerationContextError(
            "generation context response_act 漂移")
    # ANSWER 必须有完整的结构、命题和 evidence 闭包；digest 只作身份，
    # 不得被当作可读 claim 送入生成侧。
    if response_act == "ANSWER" and any(
            not item.structure_refs or not item.proposition_keys
            or not item.evidence_keys for item in context.evidence):
        raise RuntimeMaterialGenerationContextError(
            "ANSWER generation context 缺少结构/命题/evidence")
    return context


__all__ = [
    "RUNTIME_MATERIAL_GENERATION_CONTEXT_PROTOCOL_V1",
    "RuntimeMaterialGenerationContext",
    "RuntimeMaterialGenerationContextError",
    "RuntimeMaterialGenerationEvidence",
    "validate_runtime_material_generation_context",
]
