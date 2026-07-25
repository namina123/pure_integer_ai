"""cognition.understanding.refers_to — 模块3 REFERS_TO 归一链（resolve 原语·性质分发）。

normalize_to_concept：tok → ConceptRef（可能 1:N 多挂·§B12）。性质分发：
  tok 是代词?        → 性质B 语篇指代（模块5·resolve_pronoun_occurrence·落记忆）
  tok 有 QID/synset? → 多义挂概念 1:N（模块7·预处理产物·返 MultiRef·消歧在理解侧 recognize）
  tok 有形态归一源?  → 性质A 稳定同指（模块4·lemmatizer 结构化源·落核心）
  否则               → 直接 _ensure_concept（OOV·SHADOW 起步·§十一缺口#3）

归一链边类型纯净：仅性质A 纯同指边进闭包·喻称子类型不进·性质B 不进（occurrence-bound）。
activate_candidates 返全部邻居（禁取首·§H5/§B12）。

**消歧位置（刀6 修正）**：sense 消歧在**理解侧 recognize**（token→concept 时·结构拟合选 sense·IS_A 共祖
  ATTR_SLOT_ROLE）·**不在生成侧**——生成侧 dispatch_slot 是 concept→词形选词形（activate_candidates REFERS_TO
  反向·非选 sense）。observe MultiRef 进 observe 立即塌缩 refs[0]（PRECEDES 结构序）·**observe MultiRef 是 theater
  死码**·刀6 sense_candidates 表才是真摄入侧（observe record_sense_token_seen 写 sc_tn·recognize clone 选 sense）。

外部资源（lemmatizer / QID-synset / pronoun_features）是预处理离线产物·首版注入式 hook：
  默认 no-op（返 None/[]）→ 落 OOV SHADOW 起步（自足 fallback）·生产接线随预处理层。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED
from pure_integer_ai.storage.abstract_mark import set_mark, MARK_LANG
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.telemetry import record_diagnostic_event
from pure_integer_ai.cognition.shared.hub_detect import HubDegreeState
from pure_integer_ai.cognition.shared.types import LANG_NONE
from pure_integer_ai.cognition.shared.concept_index import ConceptIndex
from pure_integer_ai.cognition.shared.types import MultiRef, ConceptRef
from pure_integer_ai.cognition.understanding.refers_stable import build_refers_stable_edge
from pure_integer_ai.cognition.understanding.refers_occurrence import resolve_pronoun_occurrence

# 外部资源 hook（注入式·默认 no-op·首版自足 fallback 落 OOV SHADOW）
PronounResolver = Callable[[str, ConceptIndex, Any], ConceptRef | None]
Lemmatizer = Callable[[str], str | None]      # tok → lemma（None=无形态归一源）
SenseLookup = Callable[[str], list[ConceptRef]]  # tok → 1:N 概念（预处理离线·空=未覆盖）
PronounFeatureLookup = Callable[[str], int | tuple[int, ...] | None]


@dataclass(frozen=True)
class PronounSignalResolution:
    """一次代词图解析结果，保留身份、特征键和图证据状态。"""

    has_evidence: bool
    is_pronoun: bool
    feature_keys: tuple[int, ...]
    instruction_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """校验代词结果的三态不变量和纯整数特征键。"""
        if not self.has_evidence and self.instruction_key is not None:
            raise ValueError("无图证据的代词结果不得携带 instruction_key")
        if not self.is_pronoun and self.feature_keys:
            raise ValueError("非代词结果不得携带 feature_keys")
        if any(type(value) is not int for value in self.feature_keys):
            raise TypeError("代词 feature_keys 只能包含严格整数")


def _legacy_is_pronoun(tok: str) -> bool:
    """读取迁移期旧代词字面表，不参与来源化图主读。"""
    return tok in ("他", "她", "它", "他们", "她们", "它们",
                   "he", "she", "it", "they", "them", "him", "her")


def _normalize_feature_keys(value: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """把兼容 feature lookup 归一为去重且稳定的整数键序列。"""
    if value is None:
        return ()
    values = (value,) if type(value) is int else value
    if (not isinstance(values, tuple)
            or any(type(item) is not int for item in values)):
        raise TypeError("代词 feature lookup 必须返回整数或整数 tuple")
    return tuple(dict.fromkeys(values))


def resolve_pronoun_signal(
        tok: str, *, lang: int, language_signal_runtime=None,
        pronoun_instruction_bindings: tuple[
            tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
        language_signal_compatibility_enabled: bool = True,
        pronoun_feature_lookup: PronounFeatureLookup | None = None,
        ) -> PronounSignalResolution:
    """按图优先解析代词身份和多特征键，兼容源只在图无证据时生效。

    一个 MinimalInstruction 绑定一个完整的代词 profile；profile 可包含零个或
    多个特征概念键，避免把人称、数、有生性和性别压成单布尔或单整数。
    """
    if language_signal_runtime is not None:
        graph_result = language_signal_runtime.resolve_instruction(
            tok, language=lang)
        if graph_result.has_evidence:
            if graph_result.instruction_key is None:
                return PronounSignalResolution(True, False, ())
            binding_map: dict[tuple[int, ...], tuple[int, ...]] = {}
            for instruction_key, feature_keys in pronoun_instruction_bindings:
                if (not isinstance(instruction_key, tuple)
                        or not instruction_key
                        or any(type(item) is not int for item in instruction_key)):
                    raise TypeError("代词 instruction_key 必须是非空整数 tuple")
                normalized = _normalize_feature_keys(feature_keys)
                if instruction_key in binding_map:
                    raise ValueError("代词 instruction_key 绑定重复")
                binding_map[instruction_key] = normalized
            feature_keys = binding_map.get(graph_result.instruction_key)
            if feature_keys is None:
                return PronounSignalResolution(
                    True, False, (), graph_result.instruction_key)
            return PronounSignalResolution(
                True, True, feature_keys, graph_result.instruction_key)
    if not language_signal_compatibility_enabled:
        return PronounSignalResolution(False, False, ())
    if not _legacy_is_pronoun(tok):
        return PronounSignalResolution(False, False, ())
    feature_keys = _normalize_feature_keys(
        pronoun_feature_lookup(tok) if pronoun_feature_lookup else None)
    return PronounSignalResolution(False, True, feature_keys)


def is_pronoun(tok: str, *, lang: int | None = None,
               language_signal_runtime=None,
               pronoun_instruction_bindings: tuple[
                   tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
               language_signal_compatibility_enabled: bool = True) -> bool:
    """判断代词身份，优先读取来源化图，兼容旧字面表仅作迁移回退。"""
    if language_signal_runtime is None:
        return (_legacy_is_pronoun(tok)
                if language_signal_compatibility_enabled else False)
    if lang is None:
        raise ValueError("启用语言信号 runtime 时 is_pronoun 必须提供 lang")
    return resolve_pronoun_signal(
        tok,
        lang=lang,
        language_signal_runtime=language_signal_runtime,
        pronoun_instruction_bindings=pronoun_instruction_bindings,
        language_signal_compatibility_enabled=(
            language_signal_compatibility_enabled),
    ).is_pronoun


def normalize_to_concept(tok: str, *,
                         concept_index: ConceptIndex,
                         edge_store: Any,
                         space_id: int,
                         source: int,
                         work_memory: Any = None,
                         memory_space_id: int | None = None,
                         timestamp_seq: int = 0,
                         lemmatizer: Lemmatizer | None = None,
                         sense_lookup: SenseLookup | None = None,
                         pronoun_feature_lookup: PronounFeatureLookup | None = None,
                         backend: StorageBackend | None = None,
                         lang: int = LANG_NONE,
                         language_signal_runtime=None,
                         pronoun_instruction_bindings: tuple[
                             tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
                         language_signal_compatibility_enabled: bool = True,
                         pronoun_signal: PronounSignalResolution | None = None,
                         hub_degree_state: HubDegreeState | None = None,
                         ) -> ConceptRef | MultiRef:
    """归一链：tok → ConceptRef / MultiRef（性质分发）。

    space_id：概念点落点（按 stage·训练期 CORE / 训练后阅读 MEMORY）。
    memory_space_id：性质B pronoun 落记忆（必传·pronoun 时用）。
    backend + lang：词形 NODE_WORD 挂 abstract_mark MARK_LANG（§7.7.1 路径 B·解 target_lang 缺口·
      observe 透传 raw.lang·backend=None 时 skip·bare fixture 向后兼容）。
    """
    record_diagnostic_event("hotspot.normalize")
    signal = pronoun_signal or resolve_pronoun_signal(
        tok,
        lang=lang,
        language_signal_runtime=language_signal_runtime,
        pronoun_instruction_bindings=pronoun_instruction_bindings,
        language_signal_compatibility_enabled=(
            language_signal_compatibility_enabled),
        pronoun_feature_lookup=pronoun_feature_lookup,
    )
    # —— 代词走性质B（occurrence-bound·落记忆） ——
    if signal.is_pronoun:
        if memory_space_id is None or work_memory is None:
            # 无记忆空间/工作记忆 → 悬空代词概念点 SHADOW（待 defer 机制）
            # #733 ② fix：此 bare-fixture 路径不记 _segment_dangling（无 work_memory 可记）·
            # 生产 observe 始终提供 work_memory（observe.py:71 兜底 WorkMemory()）+ memory_space_id·永不进此分支。
            return concept_index.ensure(tok, space_id=space_id, tier=TIER_SHADOW)
        antecedent = resolve_pronoun_occurrence(
            edge_store, concept_index, tok,
            work_memory=work_memory, memory_space_id=memory_space_id,
            timestamp_seq=timestamp_seq,
            pronoun_features=signal.feature_keys,
            backend=backend,   # B6 方案3 tn+fn 路 pronoun_resolution_count 读写（gate 守·None skip）
            hub_degree_state=hub_degree_state,
        )
        if antecedent is None:
            # 悬空 → 代词概念点 SHADOW（J4=0 真碎句由卷三判）
            return concept_index.ensure(tok, space_id=space_id, tier=TIER_SHADOW)
        return antecedent

    # —— 多义 sense：预处理已挂 1:N（模块7·消歧在理解侧 recognize·刀6 修正·非生成侧） ——
    if sense_lookup is not None:
        senses = sense_lookup(tok)
        if senses:
            return MultiRef(tuple(senses))   # activate_candidates 返全部·禁取首

    # —— 形态归一：性质A 稳定同指（lemmatizer 结构化源·模块4） ——
    if lemmatizer is not None:
        lemma = lemmatizer(tok)
        if lemma is not None and lemma != tok:
            lemma_ref = concept_index.ensure(lemma, space_id=space_id, tier=TIER_PRIMARY)
            tok_ref = concept_index.ensure(tok, space_id=space_id, tier=TIER_PRIMARY)
            # 性质A 边随概念点同 space（守决策1·核心训练后不增长）
            build_refers_stable_edge(edge_store, concept_index, tok_ref, lemma_ref,
                                     epistemic=EPI_STRUCTURED, space_id=space_id)
            return lemma_ref   # 归一到原形

    # —— OOV：SHADOW 起步（§十一缺口#3） ——
    word_ref = concept_index.ensure(tok, space_id=space_id, tier=TIER_SHADOW,
                                    node_type=2)  # NODE_WORD
    # §七片3：词形 NODE_WORD 挂 abstract_mark MARK_LANG（§7.7.1 路径 B·解 target_lang 缺口·
    # lang_of 从注入式 None → 读 MARK_LANG·dispatch_slot target_lang 偏好对词形候选生效）。
    # backend=None（bare fixture）→ set_mark 内部 KeyError skip·向后兼容·守 bit-identical。
    # 非语言模态 lang=LANG_NONE 不挂（observe 不调 normalize_to_concept·守门在 caller）。
    # **gap（defer）**：仅 OOV 分支写 MARK_LANG。早返分支——sense_lookup（生产 SENSE_LOOKUP_MODE ON·:92）
    # 返 MultiRef 指既有 sense 概念点（非 NODE_WORD·按设计概念点无 lang mark ≡ None·graph_view lang_of）故不须挂；
    # lemmatizer（生产 dormant·无 caller 注入·:98）早返 lemma_ref/tok_ref 是 NODE_WORD·接入后须补挂 lang。
    # 当前 OOV 主路径已活·bit-identical。
    if backend is not None and lang != LANG_NONE:
        set_mark(backend, ref=word_ref, mark_kind=MARK_LANG, mark_value=lang)
    return word_ref
