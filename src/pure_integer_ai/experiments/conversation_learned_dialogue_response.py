"""从人工对话课程学习可迁移的提示特征到回答片段模型。

模型不保存原始 prompt，也不保存 prompt 到完整 response 的键值映射。训练先把
assistant 表层切成短片段，再累计稀疏提示特征到首片段的关联和片段间转移。运行时
只有在多个特征共同支持、最高候选唯一且相似度过门时才产生回答；事实检索和拒答门
仍由上层负责。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms


DIALOGUE_RESPONSE_SCHEMA = 1
DIALOGUE_RESPONSE_MAGIC = (21402, 260827, 72)
DIALOGUE_INTENT_SCHEMA = 2
DIALOGUE_INTENT_MAGIC = (21403, 260827, 73)
MIN_SHARED_FEATURES = 2
MIN_SIMILARITY_PERMILLE = 180
PRODUCTION_MIN_SIMILARITY_PERMILLE = 500
MAX_FEATURES_PER_PROMPT = 512
MAX_INTENT_FEATURES_PER_PROMPT = 1024
MAX_INTENT_HISTORY_TURNS = 8
MIN_INTENT_FEATURE_DOCUMENTS = 3
MAX_INTENT_FEATURE_WEIGHT = 128
MIN_INTENT_SHARED_FEATURES = 3
MAX_FRAGMENTS_PER_RESPONSE = 4
MAX_FRAGMENT_CHARS = 160
MAX_GENERATED_CHARS = 240

# Mechanical sentence-terminal registry shared by the raw-text boundary
# contract.  Values are Unicode scalar numbers, not language or vocabulary
# rules; other languages can add terminals through their own boundary course.
_TERMINAL_SCALARS = frozenset({33, 46, 59, 63, 12290, 65307, 65281, 65311})
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_DIGIT_RE = re.compile(r"[0-9]+")
_PROVIDER_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")


# object-model: exception; interop=learned-dialogue-response-v1
class LearnedDialogueResponseError(ValueError):
    """课程、模型或整数流不满足对话回答学习合同。"""


def _strict_codepoints(value: tuple[int, ...], *, label: str) -> None:
    """核验可由 Unicode scalar sequence 跨语言重建的文本记录。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 or item > 0x10FFFF
                   or 0xD800 <= item <= 0xDFFF for item in value)):
        raise LearnedDialogueResponseError(f"{label} 不是规范 Unicode scalar tuple")


def _codepoints(value: str) -> tuple[int, ...]:
    result = tuple(ord(item) for item in value)
    _strict_codepoints(result, label="text")
    return result


def _surface(values: tuple[int, ...]) -> str:
    _strict_codepoints(values, label="surface")
    return "".join(chr(item) for item in values)


def dialogue_prompt_features(surface: str) -> tuple[tuple[int, ...], ...]:
    """复用广域词项并补充脚本无关 Unicode 词元特征。

    ``broad_qa_terms`` 仍保留旧 artifact 使用的 ``c:``/``w:`` 特征。
    对话层另外从 Unicode scalar 的字母/数字连续段形成 ``u:`` 单元、二元
    和三元特征；它不判断语言名称、脚本名称或词义，因此旧模型可以继续
    消费旧特征，新模型则能覆盖非拉丁文字。
    """
    if type(surface) is not str or not surface.strip():
        raise LearnedDialogueResponseError("dialogue prompt 必须是非空文本")
    base_terms = tuple(sorted(set(broad_qa_terms(surface))))
    # 响应匹配保持已学习的词元通道稳定；只有当前语言没有可用词元时，
    # 才使用开放 Unicode scalar n-gram。意图层独立消费 Unicode 特征，
    # 因而跨脚本输入仍能进入新模型，而不会让既有长句相似度分母膨胀。
    extra_terms = (() if base_terms else tuple(sorted(
        set(_unicode_lexical_terms(surface)).difference(base_terms))))
    values = tuple(_codepoints(item)
                   for item in (*base_terms, *extra_terms))
    return values[:MAX_FEATURES_PER_PROMPT]


def _unicode_lexical_terms(surface: str) -> tuple[str, ...]:
    """由 Unicode 类别形成开放词元，不内置任何语言或脚本表。"""
    if type(surface) is not str:
        raise TypeError("unicode lexical surface 必须是字符串")
    values: set[str] = set()
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        sequence = "".join(current)
        # ASCII 已由既有 w: 特征覆盖；保留 u: 给含非 ASCII scalar 的
        # 连续段，避免旧模型的特征集合发生无必要的同义重复。
        if all(ord(item) < 128 for item in sequence):
            current.clear()
            return
        for width in (1, 2, 3):
            if len(sequence) < width:
                continue
            values.update(
                "u:" + sequence[index:index + width]
                for index in range(len(sequence) - width + 1)
            )
        current.clear()

    for item in surface:
        # 仅使用 scalar 数值和 ASCII 分隔符，不查询宿主 UCD 版本、语言或
        # 脚本名称。所有非 ASCII scalar 都可进入同一可迁移词元通道；具体
        # 词义和语言归属仍由图中 Representation 与 LanguageBranch 证据决定。
        codepoint = ord(item)
        lexical = codepoint >= 128 or "0" <= item <= "9" \
            or "A" <= item <= "Z" or "a" <= item <= "z"
        if lexical:
            current.append(item)
        else:
            flush()
    flush()
    return tuple(sorted(values))


def _lexical_scalar(value: str) -> bool:
    """识别可进入词元窗口的 scalar，不把语言或文字脚本写死。"""
    if type(value) is not str or len(value) != 1:
        raise TypeError("lexical scalar 必须是单一字符串标量")
    codepoint = ord(value)
    return codepoint >= 128 or "0" <= value <= "9" \
        or "A" <= value <= "Z" or "a" <= value <= "z"


def _intent_surface_features(surface: str) -> tuple[str, ...]:
    """形成 n-gram 与单码点两层开放词形证据，不内置意图词表。"""
    values = {"q:" + item for item in broad_qa_terms(surface)}
    values.update("q:" + item for item in _unicode_lexical_terms(surface))
    lexical = tuple(item for item in surface if _lexical_scalar(item))
    values.update("q:u:" + item for item in lexical)
    for width in range(1, min(3, len(lexical)) + 1):
        values.add("q:b:" + "".join(lexical[:width]))
        values.add("q:e:" + "".join(lexical[-width:]))
    return tuple(sorted(values))


def dialogue_intent_features(
        surface: str, *, history: tuple[tuple[int, str], ...] = (),
        ) -> tuple[tuple[int, ...], ...]:
    """把当前输入及有界历史编码为带角色、距离的离散意图证据。"""
    if type(surface) is not str or not surface.strip():
        raise LearnedDialogueResponseError("dialogue intent prompt 必须是非空文本")
    if (not isinstance(history, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2
                   or item[0] not in {1, 2}
                   or type(item[1]) is not str or not item[1].strip()
                   for item in history)):
        raise LearnedDialogueResponseError("dialogue intent history 非法")
    values: list[str] = []
    seen: set[str] = set()

    def append_features(features: Iterable[str]) -> None:
        for feature in features:
            if feature not in seen:
                seen.add(feature)
                values.append(feature)

    # 当前输入永远先于历史进入有界特征表，避免长历史挤掉本轮证据。
    append_features(_intent_surface_features(surface))
    recent = history[-MAX_INTENT_HISTORY_TURNS:]
    for distance, (role, prior_surface) in enumerate(reversed(recent), start=1):
        prefix = f"h:{distance}:{role}:"
        append_features(prefix + item[2:]
                        for item in _intent_surface_features(prior_surface))
    return tuple(_codepoints(item)
                 for item in values[:MAX_INTENT_FEATURES_PER_PROMPT])


def _safe_fragment(surface: str) -> bool:
    """只允许有限的单段自然语言片段进入低优先级对话生成。"""
    value = surface.strip()
    if not 2 <= len(value) <= MAX_FRAGMENT_CHARS or "\n" in value:
        return False
    if value.startswith(("```", "~~~", "# ", "## ", "> ")):
        return False
    if value.startswith(("(", "（", "[", "【")):
        return False
    if any(ord(item) < 32 and item not in "\t" for item in value):
        return False
    return True


def response_fragments(surface: str) -> tuple[str, ...]:
    """按显式句界切出最多四个短片段，不跨代码/长块强行截断。"""
    if type(surface) is not str or not surface.strip():
        raise LearnedDialogueResponseError("assistant response 必须是非空文本")
    result: list[str] = []
    current: list[str] = []
    parts: list[str] = []
    text = surface.strip()
    index = 0
    while index < len(text):
        item = text[index]
        if ord(item) in _TERMINAL_SCALARS:
            current.append(item)
            parts.append("".join(current))
            current = []
            index += 1
            continue
        if item == "\n":
            end = index
            while end < len(text) and text[end] == "\n":
                end += 1
            if end - index >= 2:
                if current:
                    parts.append("".join(current))
                    current = []
            elif current:
                # A single newline is content unless it follows a terminal;
                # an empty current buffer means the terminal already closed
                # the previous fragment and this is only inter-fragment space.
                current.append("\n")
            index = end
            continue
        current.append(item)
        index += 1
    if current:
        parts.append("".join(current))
    for part in parts:
        value = part.strip()
        if not value:
            continue
        if _safe_fragment(value):
            result.append(value)
        elif result:
            break
        if len(result) >= MAX_FRAGMENTS_PER_RESPONSE:
            break
    return tuple(result)


def provider_identity_markers(source_title: str) -> tuple[str, ...]:
    """从来源标题推导 provider 标记，避免把上游助手身份学成当前系统身份。"""
    if type(source_title) is not str or not source_title.strip():
        raise LearnedDialogueResponseError("source title 必须是非空文本")
    return tuple(sorted({item.casefold() for item in
                         _PROVIDER_TOKEN_RE.findall(source_title)
                         if len(item) >= 4}))


def _fold_ascii(value: str) -> str:
    return "".join(item.casefold() for item in value if item.isascii()
                   and item.isalnum())


def _contains_provider_identity(surface: str,
                                markers: tuple[str, ...]) -> bool:
    folded = _fold_ascii(surface)
    return any(_fold_ascii(marker) in folded for marker in markers)


def _response_surface_allowed(surface: str, prompt: str) -> bool:
    """只执行语言无关的表层安全门；事实约束由上层 Evidence 门负责。"""
    if not _safe_fragment(surface):
        return False
    if type(prompt) is not str or not prompt.strip():
        return False
    # 对话回答可以自然地引入组织性词汇、礼貌语和解释性内容。仅保留
    # 数字守恒，避免把安全门误用成“回答只能复述问题”的模板门；带来源
    # 的事实回答不经过此低优先级模型，仍由广域 Evidence/来源门约束。
    prompt_digits = set(_DIGIT_RE.findall(prompt))
    return set(_DIGIT_RE.findall(surface)) == prompt_digits


@dataclass(frozen=True, slots=True)
class DialogueResponseTrainingRow:
    """一条只在训练时存在的 prompt/response 投影。"""

    split: str
    prompt: str
    response: str
    source_title: str
    history: tuple[tuple[int, str], ...] = ()
    intent_support: bool = True

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout"}:
            raise LearnedDialogueResponseError("dialogue response split 非法")
        for label, value in (("prompt", self.prompt),
                             ("response", self.response),
                             ("source_title", self.source_title)):
            if type(value) is not str or not value.strip():
                raise LearnedDialogueResponseError(f"{label} 必须是非空文本")
        if (not isinstance(self.history, tuple)
                or any(not isinstance(item, tuple) or len(item) != 2
                       or item[0] not in {1, 2}
                       or type(item[1]) is not str or not item[1].strip()
                       for item in self.history)):
            raise LearnedDialogueResponseError("dialogue response history 非法")
        if type(self.intent_support) is not bool:
            raise LearnedDialogueResponseError("dialogue intent support 非法")


@dataclass(frozen=True, slots=True)
class LearnedDialogueResponseModel:
    """只含聚合整数特征、回答片段和转移的规范模型。"""

    train_count: int
    heldout_count: int
    excluded_provider_identity_count: int
    course_sha256: tuple[int, ...]
    source_sha256s: tuple[tuple[int, ...], ...]
    features: tuple[tuple[int, ...], ...]
    fragments: tuple[tuple[int, ...], ...]
    fragment_occurrence_counts: tuple[int, ...]
    fragment_start_counts: tuple[int, ...]
    fragment_feature_counts: tuple[int, ...]
    feature_fragment_counts: tuple[tuple[int, int, int], ...]
    transition_counts: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if (type(self.train_count) is not int or self.train_count <= 0
                or type(self.heldout_count) is not int or self.heldout_count <= 0
                or type(self.excluded_provider_identity_count) is not int
                or self.excluded_provider_identity_count < 0):
            raise LearnedDialogueResponseError("model row counts 非法")
        if (len(self.course_sha256) != 32
                or any(type(item) is not int or not 0 <= item <= 255
                       for item in self.course_sha256)):
            raise LearnedDialogueResponseError("course SHA-256 非法")
        if (not self.source_sha256s
                or self.source_sha256s != tuple(sorted(set(self.source_sha256s)))
                or any(len(item) != 32 or any(type(value) is not int
                       or not 0 <= value <= 255 for value in item)
                       for item in self.source_sha256s)):
            raise LearnedDialogueResponseError("source SHA-256 集合非法")
        if (not self.features or self.features != tuple(sorted(set(self.features)))):
            raise LearnedDialogueResponseError("feature table 必须非空、唯一且有序")
        if (not self.fragments
                or self.fragments != tuple(sorted(set(self.fragments)))):
            raise LearnedDialogueResponseError("fragment table 必须非空、唯一且有序")
        for value in self.features:
            _strict_codepoints(value, label="feature")
        for value in self.fragments:
            _strict_codepoints(value, label="fragment")
        widths = (self.fragment_occurrence_counts, self.fragment_start_counts,
                  self.fragment_feature_counts)
        if (any(len(item) != len(self.fragments) for item in widths)
                or any(type(value) is not int or value <= 0
                       for value in self.fragment_occurrence_counts)
                or any(type(value) is not int or value < 0
                       for values in widths[1:] for value in values)
                or not any(self.fragment_start_counts)):
            raise LearnedDialogueResponseError("fragment count vectors 非法")
        prior = None
        for feature, fragment, count in self.feature_fragment_counts:
            key = (feature, fragment)
            if (type(feature) is not int or not 0 <= feature < len(self.features)
                    or type(fragment) is not int
                    or not 0 <= fragment < len(self.fragments)
                    or type(count) is not int or count <= 0
                    or prior is not None and key <= prior):
                raise LearnedDialogueResponseError("feature-fragment table 非法")
            prior = key
        prior = None
        for source, target, count in self.transition_counts:
            key = (source, target)
            if (type(source) is not int or not 0 <= source < len(self.fragments)
                    or type(target) is not int
                    or not 0 <= target < len(self.fragments)
                    or type(count) is not int or count <= 0
                    or source == target or prior is not None and key <= prior):
                raise LearnedDialogueResponseError("fragment transition table 非法")
            prior = key

    def integer_stream(self) -> tuple[int, ...]:
        """返回完整规范整数流；所有变长记录均带显式长度。"""
        result = [
            *DIALOGUE_RESPONSE_MAGIC, DIALOGUE_RESPONSE_SCHEMA,
            self.train_count, self.heldout_count,
            self.excluded_provider_identity_count,
            *self.course_sha256, len(self.source_sha256s),
        ]
        for value in self.source_sha256s:
            result.extend(value)
        result.append(len(self.features))
        for value in self.features:
            result.extend((len(value), *value))
        result.append(len(self.fragments))
        for ordinal, value in enumerate(self.fragments):
            result.extend((len(value), *value,
                           self.fragment_occurrence_counts[ordinal],
                           self.fragment_start_counts[ordinal],
                           self.fragment_feature_counts[ordinal]))
        result.append(len(self.feature_fragment_counts))
        for value in self.feature_fragment_counts:
            result.extend(value)
        result.append(len(self.transition_counts))
        for value in self.transition_counts:
            result.extend(value)
        return tuple(result)

    @classmethod
    def from_integer_stream(cls, values: tuple[int, ...]
                            ) -> "LearnedDialogueResponseModel":
        """严格回读完整整数流，拒绝截断、扩展字段和非规范重编码。"""
        if (not isinstance(values, tuple) or not values
                or any(type(item) is not int for item in values)):
            raise LearnedDialogueResponseError("model stream 必须是整数 tuple")
        cursor = 0

        def take(count: int) -> tuple[int, ...]:
            nonlocal cursor
            if type(count) is not int or count < 0 or cursor + count > len(values):
                raise LearnedDialogueResponseError("model stream 被截断")
            result = values[cursor:cursor + count]
            cursor += count
            return result

        if take(len(DIALOGUE_RESPONSE_MAGIC)) != DIALOGUE_RESPONSE_MAGIC:
            raise LearnedDialogueResponseError("model magic 不兼容")
        if take(1) != (DIALOGUE_RESPONSE_SCHEMA,):
            raise LearnedDialogueResponseError("model schema 不兼容")
        train_count, heldout_count, excluded_count = take(3)
        course_sha = take(32)
        source_count = take(1)[0]
        source_shas = tuple(take(32) for _ in range(source_count))
        feature_count = take(1)[0]
        features = tuple(take(take(1)[0]) for _ in range(feature_count))
        fragment_count = take(1)[0]
        fragments = []
        occurrences = []
        starts = []
        feature_widths = []
        for _ in range(fragment_count):
            fragments.append(take(take(1)[0]))
            occurrence, start, feature_width = take(3)
            occurrences.append(occurrence)
            starts.append(start)
            feature_widths.append(feature_width)
        association_count = take(1)[0]
        associations = tuple(take(3) for _ in range(association_count))
        transition_count = take(1)[0]
        transitions = tuple(take(3) for _ in range(transition_count))
        if cursor != len(values):
            raise LearnedDialogueResponseError("model stream 存在尾随整数")
        model = cls(
            train_count, heldout_count, excluded_count, course_sha, source_shas,
            features, tuple(fragments), tuple(occurrences), tuple(starts),
            tuple(feature_widths), associations, transitions,
        )
        if model.integer_stream() != values:
            raise LearnedDialogueResponseError("model stream 非规范")
        return model


def learn_dialogue_response_model(
        rows: Iterable[DialogueResponseTrainingRow], *,
        course_sha256: tuple[int, ...],
        source_sha256s: tuple[tuple[int, ...], ...],
        ) -> LearnedDialogueResponseModel:
    """从显式 train/heldout 分账学习聚合特征、首片段和片段转移。"""
    row_values = tuple(rows)
    heldout_count = sum(item.split == "heldout" for item in row_values)
    train_rows = tuple(item for item in row_values if item.split == "train")
    if not train_rows or heldout_count <= 0:
        raise LearnedDialogueResponseError("训练和 heldout 均必须非空")
    fragment_occurrences: dict[tuple[int, ...], int] = {}
    fragment_starts: dict[tuple[int, ...], int] = {}
    fragment_features: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    association_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    transition_counts: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    excluded = 0
    admitted_train_count = 0
    for row in train_rows:
        markers = provider_identity_markers(row.source_title)
        if _contains_provider_identity(row.response, markers):
            excluded += 1
            continue
        fragments = tuple(_codepoints(item)
                          for item in response_fragments(row.response))
        features = dialogue_prompt_features(row.prompt)
        # 两字中文短问候只有一个二元特征。允许它进入模型；运行时仅对单特征
        # query 启用短输入裁决，不能用一个通用二元组接管更长问题。
        if not fragments or not features:
            continue
        admitted_train_count += 1
        first = fragments[0]
        fragment_starts[first] = fragment_starts.get(first, 0) + 1
        feature_set = fragment_features.setdefault(first, set())
        feature_set.update(features)
        for feature in features:
            key = (feature, first)
            association_counts[key] = association_counts.get(key, 0) + 1
        for fragment in fragments:
            fragment_occurrences[fragment] = (
                fragment_occurrences.get(fragment, 0) + 1)
        for source, target in zip(fragments, fragments[1:]):
            if source == target:
                continue
            key = (source, target)
            transition_counts[key] = transition_counts.get(key, 0) + 1
    if admitted_train_count <= 0 or not association_counts:
        raise LearnedDialogueResponseError("没有可学习的安全回答片段")
    features = tuple(sorted({key[0] for key in association_counts}))
    fragments = tuple(sorted(fragment_occurrences))
    feature_ordinals = {value: index for index, value in enumerate(features)}
    fragment_ordinals = {value: index for index, value in enumerate(fragments)}
    associations = tuple(sorted((
        feature_ordinals[feature], fragment_ordinals[fragment], count)
        for (feature, fragment), count in association_counts.items()
    ))
    transitions = tuple(sorted((
        fragment_ordinals[source], fragment_ordinals[target], count)
        for (source, target), count in transition_counts.items()
    ))
    return LearnedDialogueResponseModel(
        admitted_train_count, heldout_count, excluded,
        course_sha256, source_sha256s, features, fragments,
        tuple(fragment_occurrences[item] for item in fragments),
        tuple(fragment_starts.get(item, 0) for item in fragments),
        tuple(len(fragment_features.get(item, ())) for item in fragments),
        associations, transitions,
    )


@dataclass(frozen=True, slots=True)
class LearnedDialogueIntentModel:
    """离散意图特征、聚合关联及不含原文的稀疏提示原型。"""

    train_count: int
    heldout_count: int
    fragment_count: int
    features: tuple[tuple[int, ...], ...]
    feature_document_counts: tuple[int, ...]
    feature_fragment_counts: tuple[tuple[int, int, int], ...]
    prototype_features: tuple[tuple[int, ...], ...] = ()
    prototype_fragments: tuple[int, ...] = ()
    prototype_counts: tuple[int, ...] = ()
    serialization_schema: int = DIALOGUE_INTENT_SCHEMA

    def __post_init__(self) -> None:
        if (type(self.train_count) is not int or self.train_count <= 0
                or type(self.heldout_count) is not int or self.heldout_count <= 0
                or type(self.fragment_count) is not int or self.fragment_count <= 0):
            raise LearnedDialogueResponseError("intent model counts 非法")
        if (not self.features
                or self.features != tuple(sorted(set(self.features)))
                or len(self.feature_document_counts) != len(self.features)):
            raise LearnedDialogueResponseError("intent feature table 非法")
        for feature, document_count in zip(
                self.features, self.feature_document_counts):
            _strict_codepoints(feature, label="intent feature")
            if (type(document_count) is not int
                    or not 1 <= document_count <= self.train_count):
                raise LearnedDialogueResponseError("intent document count 非法")
        prior = None
        for feature, fragment, count in self.feature_fragment_counts:
            key = feature, fragment
            if (type(feature) is not int
                    or not 0 <= feature < len(self.features)
                    or type(fragment) is not int
                    or not 0 <= fragment < self.fragment_count
                    or type(count) is not int or count <= 0
                    or prior is not None and key <= prior):
                raise LearnedDialogueResponseError("intent association table 非法")
            prior = key
        if (self.serialization_schema not in {1, DIALOGUE_INTENT_SCHEMA}
                or self.serialization_schema == 1
                and (self.prototype_features or self.prototype_fragments
                     or self.prototype_counts)
                or not (len(self.prototype_features)
                        == len(self.prototype_fragments)
                        == len(self.prototype_counts))):
            raise LearnedDialogueResponseError("intent prototype schema 非法")
        prior_prototype = None
        for features, fragment, count in zip(
                self.prototype_features, self.prototype_fragments,
                self.prototype_counts):
            key = features, fragment
            if (not features or features != tuple(sorted(set(features)))
                    or any(type(item) is not int
                           or not 0 <= item < len(self.features)
                           for item in features)
                    or type(fragment) is not int
                    or not 0 <= fragment < self.fragment_count
                    or type(count) is not int or count <= 0
                    or prior_prototype is not None and key <= prior_prototype):
                raise LearnedDialogueResponseError("intent prototype table 非法")
            prior_prototype = key
        if (self.serialization_schema == DIALOGUE_INTENT_SCHEMA
                and not self.prototype_features):
            raise LearnedDialogueResponseError("intent prototype table 为空")

    def integer_stream(self) -> tuple[int, ...]:
        """返回可由其他整数实现逐字段重建的规范流。"""
        result = [
            *DIALOGUE_INTENT_MAGIC, self.serialization_schema,
            self.train_count, self.heldout_count, self.fragment_count,
            len(self.features),
        ]
        for feature, document_count in zip(
                self.features, self.feature_document_counts):
            result.extend((len(feature), *feature, document_count))
        result.append(len(self.feature_fragment_counts))
        for value in self.feature_fragment_counts:
            result.extend(value)
        if self.serialization_schema >= 2:
            result.append(len(self.prototype_features))
            for features, fragment, count in zip(
                    self.prototype_features, self.prototype_fragments,
                    self.prototype_counts):
                result.extend((len(features), *features, fragment, count))
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "LearnedDialogueIntentModel":
        """严格回读 intent 整数流并拒绝尾随字段。"""
        if (not isinstance(values, tuple) or not values
                or any(type(item) is not int for item in values)):
            raise LearnedDialogueResponseError("intent stream 必须是整数 tuple")
        cursor = 0

        def take(count: int) -> tuple[int, ...]:
            nonlocal cursor
            if type(count) is not int or count < 0 or cursor + count > len(values):
                raise LearnedDialogueResponseError("intent stream 被截断")
            result = values[cursor:cursor + count]
            cursor += count
            return result

        if take(len(DIALOGUE_INTENT_MAGIC)) != DIALOGUE_INTENT_MAGIC:
            raise LearnedDialogueResponseError("intent magic 不兼容")
        schema = take(1)[0]
        if schema not in {1, DIALOGUE_INTENT_SCHEMA}:
            raise LearnedDialogueResponseError("intent schema 不兼容")
        train_count, heldout_count, fragment_count, feature_count = take(4)
        features = []
        document_counts = []
        for _ in range(feature_count):
            features.append(take(take(1)[0]))
            document_counts.append(take(1)[0])
        association_count = take(1)[0]
        associations = tuple(take(3) for _ in range(association_count))
        prototype_features = []
        prototype_fragments = []
        prototype_counts = []
        if schema >= 2:
            prototype_count = take(1)[0]
            for _ in range(prototype_count):
                prototype_features.append(take(take(1)[0]))
                fragment, count = take(2)
                prototype_fragments.append(fragment)
                prototype_counts.append(count)
        if cursor != len(values):
            raise LearnedDialogueResponseError("intent stream 存在尾随整数")
        model = cls(
            train_count, heldout_count, fragment_count, tuple(features),
            tuple(document_counts), associations, tuple(prototype_features),
            tuple(prototype_fragments), tuple(prototype_counts), schema)
        if model.integer_stream() != values:
            raise LearnedDialogueResponseError("intent stream 非规范")
        return model


def learn_dialogue_intent_model(
        rows: Iterable[DialogueResponseTrainingRow],
        response_model: LearnedDialogueResponseModel,
        ) -> LearnedDialogueIntentModel:
    """学习聚合关联和不保存提示原文的当前句/历史稀疏原型。"""
    if not isinstance(response_model, LearnedDialogueResponseModel):
        raise TypeError("intent learning 需要 response model")
    fragment_ordinals = {
        value: ordinal for ordinal, value in enumerate(response_model.fragments)}
    admitted: list[tuple[tuple[tuple[int, ...], ...],
                         tuple[tuple[int, ...], ...], int]] = []
    heldout_count = 0
    for row in rows:
        if row.split == "heldout":
            heldout_count += 1
            continue
        if not row.intent_support:
            continue
        markers = provider_identity_markers(row.source_title)
        if _contains_provider_identity(row.response, markers):
            continue
        fragments = response_fragments(row.response)
        if not fragments:
            continue
        fragment = fragment_ordinals.get(_codepoints(fragments[0]))
        if fragment is None:
            continue
        current_features = dialogue_intent_features(row.prompt)
        contextual_features = dialogue_intent_features(
            row.prompt, history=row.history)
        if current_features and contextual_features:
            admitted.append((current_features, contextual_features, fragment))
    if not admitted or heldout_count <= 0:
        raise LearnedDialogueResponseError("intent train/heldout 不能为空")
    document_counts = Counter(
        feature for _, features, _ in admitted for feature in set(features))
    maximum_documents = max(
        MIN_INTENT_FEATURE_DOCUMENTS, len(admitted) // 2)
    selected_features = tuple(sorted(
        feature for feature, count in document_counts.items()
        if 1 <= count <= len(admitted)))
    if not selected_features:
        raise LearnedDialogueResponseError("没有可重复验证的 intent feature")
    ordinals = {value: ordinal for ordinal, value in enumerate(selected_features)}
    association_counts: Counter[tuple[int, int]] = Counter()
    prototype_counts: Counter[tuple[tuple[int, ...], int]] = Counter()
    for current_features, contextual_features, fragment in admitted:
        for prototype in {current_features, contextual_features}:
            feature_ordinals = tuple(sorted(
                ordinal for feature in set(prototype)
                if (ordinal := ordinals.get(feature)) is not None))
            if feature_ordinals:
                prototype_counts[(feature_ordinals, fragment)] += 1
        features = contextual_features
        for feature in set(features):
            ordinal = ordinals.get(feature)
            if (ordinal is not None
                    and MIN_INTENT_FEATURE_DOCUMENTS
                    <= document_counts[feature] <= maximum_documents):
                association_counts[(ordinal, fragment)] += 1
    associations = tuple(sorted((feature, fragment, count)
                                for (feature, fragment), count
                                in association_counts.items()))
    if not associations:
        raise LearnedDialogueResponseError("intent association 为空")
    prototypes = tuple(sorted(
        (features, fragment, count)
        for (features, fragment), count in prototype_counts.items()))
    if not prototypes:
        raise LearnedDialogueResponseError("intent prototype 为空")
    return LearnedDialogueIntentModel(
        len(admitted), heldout_count, len(response_model.fragments),
        selected_features,
        tuple(document_counts[item] for item in selected_features),
        associations,
        tuple(item[0] for item in prototypes),
        tuple(item[1] for item in prototypes),
        tuple(item[2] for item in prototypes),
    )


@dataclass(frozen=True, slots=True)
class LearnedDialogueResponseResult:
    """一次低优先级对话回答决策的值记录。"""

    surface: str | None
    used: bool
    similarity_permille: int
    shared_feature_count: int
    fragment_ordinal: int
    reason: str
    trace: tuple[int, ...]


# object-model: derived_cache; representation=runtime; interop=learned-dialogue-intent-v1
class LearnedDialogueIntentRuntime:
    """以倒排原型为主、旧版聚合关联为兼容路径消费意图模型。"""

    __slots__ = (
        "model", "_feature_lookup", "_fragment_surfaces", "_postings",
        "_prototype_postings", "_prototype_weight_sums",
        "_prototype_current_weight_sums")

    def __init__(
            self, model: LearnedDialogueIntentModel,
            fragment_surfaces: tuple[tuple[int, ...], ...] = (),
            ) -> None:
        if not isinstance(model, LearnedDialogueIntentModel):
            raise TypeError("dialogue intent runtime 需要 intent model")
        if (fragment_surfaces
                and len(fragment_surfaces) != model.fragment_count):
            raise LearnedDialogueResponseError("intent fragment surface 漂移")
        self.model = model
        self._fragment_surfaces = fragment_surfaces
        self._feature_lookup = {
            value: ordinal for ordinal, value in enumerate(model.features)}
        postings: dict[int, list[tuple[int, int]]] = {}
        for feature, fragment, count in model.feature_fragment_counts:
            postings.setdefault(feature, []).append((fragment, count))
        self._postings = {key: tuple(value)
                          for key, value in postings.items()}
        prototype_postings: dict[int, list[int]] = {}
        prototype_weight_sums = []
        prototype_current_weight_sums = []
        for prototype, features in enumerate(model.prototype_features):
            weight_sum = 0
            current_weight_sum = 0
            for feature in features:
                prototype_postings.setdefault(feature, []).append(prototype)
                weight = self._feature_weight(feature)
                weight_sum += weight
                if model.features[feature][:2] == (ord("q"), ord(":")):
                    current_weight_sum += weight
            prototype_weight_sums.append(weight_sum)
            prototype_current_weight_sums.append(current_weight_sum)
        self._prototype_postings = {
            key: tuple(value) for key, value in prototype_postings.items()}
        self._prototype_weight_sums = tuple(prototype_weight_sums)
        self._prototype_current_weight_sums = tuple(
            prototype_current_weight_sums)

    def _feature_weight(self, feature: int) -> int:
        return min(
            MAX_INTENT_FEATURE_WEIGHT,
            max(1, self.model.train_count
                // self.model.feature_document_counts[feature]),
        )

    def rank(
            self, prompt: str, *, history: tuple[tuple[int, str], ...] = (),
            minimum_similarity_permille: int,
            ) -> tuple[int, int, int, tuple[int, ...]] | None:
        """返回唯一片段、置信度、共享数和不含词面的整数 trace。"""
        if self.model.prototype_features:
            return self._rank_prototypes(
                prompt, history=history,
                minimum_similarity_permille=minimum_similarity_permille)
        return self._rank_aggregate(
            prompt, history=history,
            minimum_similarity_permille=minimum_similarity_permille)

    def _rank_prototypes(
            self, prompt: str, *, history: tuple[tuple[int, str], ...],
            minimum_similarity_permille: int,
            ) -> tuple[int, int, int, tuple[int, ...]] | None:
        """只比较倒排命中的稀疏提示原型，并先排除不安全回答。"""
        query_features = dialogue_intent_features(prompt, history=history)
        current_features = dialogue_intent_features(prompt)
        current_ordinals = frozenset(
            feature for value in current_features
            if (feature := self._feature_lookup.get(value)) is not None)
        query_ordinals = tuple(dict.fromkeys(
            feature for value in query_features
            if (feature := self._feature_lookup.get(value)) is not None))
        if not query_ordinals or not current_ordinals:
            return None
        query_weight = sum(self._feature_weight(item)
                           for item in query_ordinals)
        current_query_weight = sum(self._feature_weight(item)
                                   for item in current_ordinals)
        overlaps: dict[int, int] = {}
        current_overlaps: dict[int, int] = {}
        shared_counts: dict[int, int] = {}
        current_shared_counts: dict[int, int] = {}
        rare_counts: dict[int, int] = {}
        current_rare_counts: dict[int, int] = {}
        rare_limit = max(16, self.model.train_count // 64)
        for feature in query_ordinals:
            weight = self._feature_weight(feature)
            is_rare = self.model.feature_document_counts[feature] <= rare_limit
            for prototype in self._prototype_postings.get(feature, ()):
                overlaps[prototype] = overlaps.get(prototype, 0) + weight
                shared_counts[prototype] = shared_counts.get(prototype, 0) + 1
                if feature in current_ordinals:
                    current_overlaps[prototype] = (
                        current_overlaps.get(prototype, 0) + weight)
                    current_shared_counts[prototype] = (
                        current_shared_counts.get(prototype, 0) + 1)
                if is_rare:
                    rare_counts[prototype] = rare_counts.get(prototype, 0) + 1
                    if feature in current_ordinals:
                        current_rare_counts[prototype] = (
                            current_rare_counts.get(prototype, 0) + 1)
        best_by_fragment: dict[int, tuple[int, ...]] = {}
        for prototype, overlap in overlaps.items():
            shared = shared_counts[prototype]
            rare = rare_counts.get(prototype, 0)
            current_shared = current_shared_counts.get(prototype, 0)
            current_rare = current_rare_counts.get(prototype, 0)
            fragment = self.model.prototype_fragments[prototype]
            if (shared < MIN_INTENT_SHARED_FEATURES or rare <= 0
                    or current_shared < MIN_INTENT_SHARED_FEATURES
                    or current_rare <= 0):
                continue
            if self._fragment_surfaces and not _response_surface_allowed(
                    _surface(self._fragment_surfaces[fragment]), prompt):
                continue
            width = self._prototype_weight_sums[prototype]
            current_width = self._prototype_current_weight_sums[prototype]
            if width <= 0 or current_width <= 0:
                continue
            current_overlap = current_overlaps[prototype]
            current_score = ((2000 * current_overlap)
                             // (current_query_weight + current_width))
            context_score = (2000 * overlap) // (query_weight + width)
            coverage = (1000 * overlap) // query_weight
            current_coverage = ((1000 * current_overlap)
                                // current_query_weight)
            rank = (
                current_score, context_score, current_coverage, coverage,
                current_overlap, overlap, current_shared, shared,
                current_rare, rare,
                self.model.prototype_counts[prototype], -prototype,
                fragment, prototype)
            prior = best_by_fragment.get(fragment)
            if prior is None or rank > prior:
                best_by_fragment[fragment] = rank
        ranked = sorted(best_by_fragment.values(), reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][:11] == best[:11]:
            return None
        runner_score = ranked[1][0] if len(ranked) > 1 else 0
        relative_confidence = (1000 if runner_score == 0 else
                               (1000 * best[0])
                               // (best[0] + runner_score))
        if (best[0] < minimum_similarity_permille
                or relative_confidence < minimum_similarity_permille):
            return None
        return (
            best[-2], best[0], best[6],
            (3, len(query_features), len(query_ordinals), query_weight,
             current_query_weight, best[0], runner_score,
             relative_confidence,
             best[1], best[2], best[3], best[4], best[5], best[6],
             best[7], best[8], best[9], best[10], best[-2], best[-1]),
        )

    def _rank_aggregate(
            self, prompt: str, *, history: tuple[tuple[int, str], ...],
            minimum_similarity_permille: int,
            ) -> tuple[int, int, int, tuple[int, ...]] | None:
        """兼容 v3 artifact 的聚合特征到片段差分算法。"""
        query_features = dialogue_intent_features(prompt, history=history)
        matched: dict[int, set[int]] = {}
        scores: dict[int, int] = {}
        rare_counts: dict[int, int] = {}
        rare_limit = max(16, self.model.train_count // 64)
        for value in query_features:
            feature = self._feature_lookup.get(value)
            if feature is None:
                continue
            document_count = self.model.feature_document_counts[feature]
            weight = self._feature_weight(feature)
            for fragment, association_count in self._postings.get(feature, ()):
                matched.setdefault(fragment, set()).add(feature)
                scores[fragment] = scores.get(fragment, 0) + (
                    weight * association_count)
                if document_count <= rare_limit:
                    rare_counts[fragment] = rare_counts.get(fragment, 0) + 1
        ranked = []
        for fragment, features in matched.items():
            shared = len(features)
            rare = rare_counts.get(fragment, 0)
            if shared < MIN_INTENT_SHARED_FEATURES or rare <= 0:
                continue
            ranked.append((scores[fragment], shared, rare, -fragment, fragment))
        ranked.sort(reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        if len(ranked) > 1 and ranked[1][:-2] == best[:-2]:
            return None
        runner_score = ranked[1][0] if len(ranked) > 1 else 0
        confidence = (1000 if runner_score == 0 else
                      (1000 * best[0]) // (best[0] + runner_score))
        if confidence < minimum_similarity_permille:
            return None
        return (
            best[-1], confidence, best[1],
            (2, len(query_features), best[0], runner_score,
             confidence, best[1], best[2], best[-1]),
        )


# object-model: derived_cache; representation=runtime; interop=learned-dialogue-response-v1
class LearnedDialogueResponseRuntime:
    """为规范模型建立不改变语义的 feature posting 与 transition 派生缓存。"""

    __slots__ = (
        "model", "intent", "_feature_lookup", "_postings", "_transitions")

    def __init__(
            self, model: LearnedDialogueResponseModel,
            intent_model: LearnedDialogueIntentModel | None = None,
            *, intent_runtime: object | None = None,
            ) -> None:
        if not isinstance(model, LearnedDialogueResponseModel):
            raise TypeError("dialogue response runtime 需要 learned model")
        if (intent_model is not None
                and intent_model.fragment_count != len(model.fragments)):
            raise LearnedDialogueResponseError("intent/response fragment table 漂移")
        if intent_model is not None and intent_runtime is not None:
            raise LearnedDialogueResponseError("intent model/runtime 不得同时指定")
        if (intent_runtime is not None
                and not callable(getattr(intent_runtime, "rank", None))):
            raise LearnedDialogueResponseError("intent runtime 缺少 rank")
        self.model = model
        self.intent = (
            intent_runtime if intent_runtime is not None else
            None if intent_model is None else
            LearnedDialogueIntentRuntime(intent_model, model.fragments))
        self._feature_lookup = {
            value: ordinal for ordinal, value in enumerate(model.features)}
        postings: dict[int, list[tuple[int, int]]] = {}
        for feature, fragment, count in model.feature_fragment_counts:
            postings.setdefault(feature, []).append((fragment, count))
        self._postings = {key: tuple(value)
                          for key, value in postings.items()}
        transitions: dict[int, list[tuple[int, int]]] = {}
        for source, target, count in model.transition_counts:
            transitions.setdefault(source, []).append((target, count))
        self._transitions = {key: tuple(value)
                             for key, value in transitions.items()}

    def close(self) -> None:
        """Close an optional persistent intent runtime."""
        close = getattr(self.intent, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _surface_allowed(surface: str, prompt: str) -> bool:
        return _response_surface_allowed(surface, prompt)

    def respond(
            self, prompt: str, *,
            history: tuple[tuple[int, str], ...] = (),
            minimum_fragment_occurrences: int = 1,
            minimum_similarity_permille: int = MIN_SIMILARITY_PERMILLE,
            ) -> LearnedDialogueResponseResult:
        """按 Dice 型纯整数相似度选择唯一首片段，并可追加共享转移。"""
        if (type(minimum_fragment_occurrences) is not int
                or minimum_fragment_occurrences <= 0
                or type(minimum_similarity_permille) is not int
                or not 0 <= minimum_similarity_permille <= 1000):
            raise ValueError("dialogue response runtime 门槛非法")
        query_features = dialogue_prompt_features(prompt)
        known_query_features = tuple(
            item for item in query_features if item in self._feature_lookup)
        matched: dict[int, set[int]] = {}
        support: dict[int, int] = {}
        for feature_value in query_features:
            feature = self._feature_lookup.get(feature_value)
            if feature is None:
                continue
            for fragment, count in self._postings.get(feature, ()):
                if self.model.fragment_start_counts[fragment] <= 0:
                    continue
                matched.setdefault(fragment, set()).add(feature)
                support[fragment] = support.get(fragment, 0) + count
        ranked = []
        for fragment, feature_set in matched.items():
            shared = len(feature_set)
            width = self.model.fragment_feature_counts[fragment]
            required_shared = (
                1 if len(known_query_features) == 1 else MIN_SHARED_FEATURES)
            if not known_query_features:
                continue
            if (shared < required_shared or width <= 0
                    or self.model.fragment_occurrence_counts[fragment]
                    < minimum_fragment_occurrences):
                continue
            similarity = (2000 * shared) // (
                len(known_query_features) + width)
            if similarity < minimum_similarity_permille:
                continue
            value = _surface(self.model.fragments[fragment])
            if not self._surface_allowed(value, prompt):
                continue
            ranked.append((similarity, shared, support[fragment],
                           self.model.fragment_start_counts[fragment],
                           self.model.fragment_occurrence_counts[fragment],
                           -fragment, fragment))
        ranked.sort(reverse=True)
        if not ranked:
            return self._intent_response(
                prompt, history=history,
                minimum_fragment_occurrences=minimum_fragment_occurrences,
                minimum_similarity_permille=minimum_similarity_permille,
                fallback_reason="insufficient_learned_support",
                fallback_trace=(0, len(known_query_features)))
        best = ranked[0]
        skipped_ambiguous = 0
        if len(known_query_features) == 1:
            # 单一短特征常对应多个只出现一次的问候变体。它们不能靠 fragment
            # ordinal 任意决胜；跳过整个并列层，允许下一层由跨样本重复支持的
            # 唯一片段接管。长输入仍在最高层歧义时直接拒答。
            cursor = 0
            while cursor < len(ranked):
                end = cursor + 1
                while (end < len(ranked)
                       and ranked[end][:-2] == ranked[cursor][:-2]):
                    end += 1
                if end == cursor + 1:
                    best = ranked[cursor]
                    break
                skipped_ambiguous += end - cursor
                cursor = end
            else:
                return self._intent_response(
                    prompt, history=history,
                    minimum_fragment_occurrences=minimum_fragment_occurrences,
                    minimum_similarity_permille=minimum_similarity_permille,
                    fallback_reason="ambiguous_learned_response",
                    fallback_trace=(
                        0, len(known_query_features), skipped_ambiguous))
        elif len(ranked) > 1 and ranked[1][:-2] == best[:-2]:
            return self._intent_response(
                prompt, history=history,
                minimum_fragment_occurrences=minimum_fragment_occurrences,
                minimum_similarity_permille=minimum_similarity_permille,
                fallback_reason="ambiguous_learned_response",
                fallback_trace=(0, len(known_query_features), 2))
        fragment = best[-1]
        parts = [_surface(self.model.fragments[fragment])]
        transitions = sorted(
            self._transitions.get(fragment, ()),
            key=lambda item: (-item[1], item[0]))
        if transitions and transitions[0][1] >= 2:
            target, count = transitions[0]
            value = _surface(self.model.fragments[target])
            if (sum(len(item) for item in parts) + len(value)
                    <= MAX_GENERATED_CHARS
                    and self._surface_allowed(value, prompt)):
                parts.append(value)
        result = "".join(parts)
        return LearnedDialogueResponseResult(
            result, True, best[0], best[1], fragment,
            "learned_fragment_selected",
            (1, len(known_query_features), best[0], best[1], best[2],
             best[3], best[4], fragment, len(parts), skipped_ambiguous))

    def _intent_response(
            self, prompt: str, *, history: tuple[tuple[int, str], ...],
            minimum_fragment_occurrences: int,
            minimum_similarity_permille: int,
            fallback_reason: str, fallback_trace: tuple[int, ...],
            ) -> LearnedDialogueResponseResult:
        """在直接 n-gram 失败后尝试语料统计意图；失败则保持原拒答。"""
        if self.intent is None:
            return LearnedDialogueResponseResult(
                None, False, 0, 0, -1, fallback_reason, fallback_trace)
        selected = self.intent.rank(
            prompt, history=history,
            minimum_similarity_permille=minimum_similarity_permille)
        if selected is None:
            return LearnedDialogueResponseResult(
                None, False, 0, 0, -1, fallback_reason, fallback_trace)
        fragment, confidence, shared, trace = selected
        if (self.model.fragment_occurrence_counts[fragment]
                < minimum_fragment_occurrences):
            return LearnedDialogueResponseResult(
                None, False, confidence, shared, -1,
                "insufficient_intent_fragment_occurrence", trace)
        first = _surface(self.model.fragments[fragment])
        if not self._surface_allowed(first, prompt):
            return LearnedDialogueResponseResult(
                None, False, confidence, shared, -1,
                "unsafe_intent_fragment", trace)
        parts = [first]
        transitions = sorted(
            self._transitions.get(fragment, ()),
            key=lambda item: (-item[1], item[0]))
        if transitions and transitions[0][1] >= 2:
            target, _ = transitions[0]
            value = _surface(self.model.fragments[target])
            if (len(first) + len(value) <= MAX_GENERATED_CHARS
                    and self._surface_allowed(value, prompt)):
                parts.append(value)
        return LearnedDialogueResponseResult(
            "".join(parts), True, confidence, shared, fragment,
            "learned_intent_fragment_selected",
            (*trace, len(parts)))


__all__ = [
    "DialogueResponseTrainingRow", "LearnedDialogueResponseError",
    "LearnedDialogueResponseModel", "LearnedDialogueResponseResult",
    "LearnedDialogueResponseRuntime", "dialogue_prompt_features",
    "learn_dialogue_response_model", "provider_identity_markers",
    "response_fragments", "PRODUCTION_MIN_SIMILARITY_PERMILLE",
]
