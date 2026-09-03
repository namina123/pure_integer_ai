"""自由对话三阶段的公共整数 trace 协议。

该协议只承载图查询的整数投影和稳定引用，供理解、过程、结果三卷以及
公开运行入口共享。它不承载语言词表、问答文本或宿主对象，因此可在其他
整数语言中按同一 stable key 重建。
"""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata


DIALOGUE_RESULT_EXACT = 1
DIALOGUE_RESULT_TRANSFER = 2
DIALOGUE_RESULT_RESPONSE_CLASS = 3
DIALOGUE_RESULT_CLARIFICATION = 4
_DIALOGUE_RESULT_MODES = frozenset({
    DIALOGUE_RESULT_EXACT,
    DIALOGUE_RESULT_TRANSFER,
    DIALOGUE_RESULT_RESPONSE_CLASS,
    DIALOGUE_RESULT_CLARIFICATION,
})
_MAX_TRANSFER_TOKENS = 128


def _int_tuple(value: tuple[int, ...], *, label: str,
               non_empty: bool = False) -> tuple[int, ...]:
    if type(value) is not tuple or (non_empty and not value):
        raise TypeError(f"{label} 必须是整数 tuple")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} 只能包含非负整数")
    return value


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class IntegerSurfaceToken:
    """一个 Unicode 整数 token 及其在原表层中的半开区间。"""

    codepoints: tuple[int, ...]
    start: int
    end: int

    def __post_init__(self) -> None:
        _int_tuple(self.codepoints, label="surface token", non_empty=True)
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start):
            raise ValueError("surface token 区间非法")


def integer_surface_tokens(surface: str) -> tuple[IntegerSurfaceToken, ...]:
    """按通用 Unicode 类别切分，不注入语言词表或文字转换规则。"""
    if type(surface) is not str:
        raise TypeError("surface 必须是文本")
    value = surface.strip()
    result: list[IntegerSurfaceToken] = []
    current: list[int] = []
    current_start = 0

    def flush(end: int) -> None:
        if current:
            result.append(IntegerSurfaceToken(
                tuple(current), current_start, end))
            current.clear()

    for ordinal, character in enumerate(value):
        category = unicodedata.category(character)
        if category.startswith("Z") or category in {"Cc", "Cf"}:
            flush(ordinal)
            continue
        if category.startswith("P") or category.startswith("S"):
            flush(ordinal)
            result.append(IntegerSurfaceToken(
                (ord(character),), ordinal, ordinal + 1))
            continue
        if ord(character) > 127 and category.startswith("L"):
            flush(ordinal)
            result.append(IntegerSurfaceToken(
                (ord(character),), ordinal, ordinal + 1))
            continue
        if not current:
            current_start = ordinal
        current.append(ord(character))
    flush(len(value))
    return tuple(result)


def integer_token_values(surface: str) -> tuple[tuple[int, ...], ...]:
    """返回不携带宿主字符串的 token 整数序列。"""
    return tuple(item.codepoints for item in integer_surface_tokens(surface))


def integer_token_features(
        tokens: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[int, ...], ...]:
    """从 token 序列派生一至三元、带长度边界的整数片段。"""
    if type(tokens) is not tuple:
        raise TypeError("tokens 必须是 tuple")
    for ordinal, token in enumerate(tokens):
        _int_tuple(token, label=f"tokens[{ordinal}]", non_empty=True)
    result: list[tuple[int, ...]] = []
    for width in (3, 2, 1):
        for start in range(max(0, len(tokens) - width + 1)):
            result.append((
                width,
                *(item for token in tokens[start:start + width]
                  for item in (len(token), *token)),
            ))
    return tuple(result)


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class DialogueSurfaceTransfer:
    """把输入变项代入已学 current->response 图路径的结果。"""

    surface: str
    result_tokens: tuple[tuple[int, ...], ...]
    anchor_count: int
    replacement_count: int

    def __post_init__(self) -> None:
        if type(self.surface) is not str or not self.surface.strip():
            raise ValueError("dialogue transfer surface 不能为空")
        if integer_token_values(self.surface) != self.result_tokens:
            raise ValueError("dialogue transfer token 与表层漂移")
        if (type(self.anchor_count) is not int or self.anchor_count <= 0
                or type(self.replacement_count) is not int
                or self.replacement_count <= 0):
            raise ValueError("dialogue transfer 计数非法")


def _lcs_pairs(
        left: tuple[tuple[int, ...], ...],
        right: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[int, int], ...]:
    """返回一个确定性的最长公共 token 子序列坐标。"""
    rows = len(left)
    columns = len(right)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for left_index in range(rows - 1, -1, -1):
        for right_index in range(columns - 1, -1, -1):
            if left[left_index] == right[right_index]:
                lengths[left_index][right_index] = (
                    lengths[left_index + 1][right_index + 1] + 1)
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index + 1][right_index],
                    lengths[left_index][right_index + 1],
                )
    result: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < rows and right_index < columns:
        if left[left_index] == right[right_index]:
            result.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif lengths[left_index + 1][right_index] >= (
                lengths[left_index][right_index + 1]):
            left_index += 1
        else:
            right_index += 1
    return tuple(result)


def _category_family(codepoints: tuple[int, ...]) -> frozenset[str]:
    """把码点投影为 Unicode 大类，不判断具体文字或语言。"""
    return frozenset(unicodedata.category(chr(value))[0]
                     for value in codepoints)


def _find_token_sequence(
        values: tuple[tuple[int, ...], ...],
        target: tuple[tuple[int, ...], ...],
        ) -> tuple[tuple[int, int], ...]:
    if not target or len(target) > len(values):
        return ()
    return tuple(
        (start, start + len(target))
        for start in range(len(values) - len(target) + 1)
        if values[start:start + len(target)] == target
    )


def transfer_dialogue_surface(
        query_surface: str,
        learned_current_surface: str,
        learned_response_surface: str,
        ) -> DialogueSurfaceTransfer | None:
    """从已学后继路径归纳变项并生成新表层；无可证代入时返回 None。

    唯一允许的改写是：查询和已学 current 的差异片段，也在已学 response
    中作为连续 token 出现。运行时将该 response 片段替换为查询同位置片段；
    其余输出、空白和标点完全沿用图中 response occurrence。算法只使用
    Unicode 整数、序和最长公共子序列，可由其他语言逐位复现。
    """
    for label, value in (
            ("query", query_surface),
            ("learned current", learned_current_surface),
            ("learned response", learned_response_surface)):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{label} surface 必须是非空文本")
    query = query_surface.strip()
    current = learned_current_surface.strip()
    response = learned_response_surface.strip()
    query_spans = integer_surface_tokens(query)
    current_spans = integer_surface_tokens(current)
    response_spans = integer_surface_tokens(response)
    query_tokens = tuple(item.codepoints for item in query_spans)
    current_tokens = tuple(item.codepoints for item in current_spans)
    response_tokens = tuple(item.codepoints for item in response_spans)
    if (query_tokens == current_tokens or not query_tokens
            or not current_tokens or not response_tokens
            or max(len(query_tokens), len(current_tokens))
            > _MAX_TRANSFER_TOKENS):
        return None
    anchors = _lcs_pairs(current_tokens, query_tokens)
    if (len(anchors) < 2
            or 1000 * len(anchors)
            < 500 * max(len(query_tokens), len(current_tokens))):
        return None
    boundaries = ((-1, -1), *anchors,
                  (len(current_tokens), len(query_tokens)))
    edits: list[tuple[int, int, int, int]] = []
    for previous, following in zip(boundaries, boundaries[1:]):
        current_start, query_start = previous[0] + 1, previous[1] + 1
        current_end, query_end = following
        if current_start >= current_end or query_start >= query_end:
            continue
        old_codepoints = tuple(
            value
            for token in current_tokens[current_start:current_end]
            for value in token)
        new_codepoints = tuple(
            value
            for token in query_tokens[query_start:query_end]
            for value in token)
        if (_category_family(old_codepoints) != _category_family(new_codepoints)
                or _category_family(old_codepoints).issubset({"P", "S", "Z", "C"})):
            continue
        edits.append((current_start, current_end, query_start, query_end))
    if not edits or len(edits) > 4:
        return None
    replacements: list[tuple[int, int, str]] = []
    occupied: set[int] = set()
    for current_start, current_end, query_start, query_end in sorted(
            edits, key=lambda item: (-(item[1] - item[0]), item)):
        old_tokens = current_tokens[current_start:current_end]
        occurrences = _find_token_sequence(response_tokens, old_tokens)
        if len(old_tokens) == 1 and len(occurrences) != 1:
            continue
        replacement = query[
            query_spans[query_start].start:query_spans[query_end - 1].end]
        for response_start, response_end in occurrences:
            indices = set(range(response_start, response_end))
            if indices & occupied:
                continue
            occupied.update(indices)
            replacements.append((response_start, response_end, replacement))
    if not replacements:
        return None
    ordered = tuple(sorted(replacements))
    cursor = 0
    pieces: list[str] = []
    for token_start, token_end, replacement in ordered:
        start = response_spans[token_start].start
        end = response_spans[token_end - 1].end
        if start < cursor:
            continue
        pieces.extend((response[cursor:start], replacement))
        cursor = end
    pieces.append(response[cursor:])
    surface = "".join(pieces)
    result_tokens = integer_token_values(surface)
    if (surface == response or surface == query or not result_tokens
            or len(surface) > 512):
        return None
    return DialogueSurfaceTransfer(
        surface,
        result_tokens,
        len(anchors),
        len(ordered),
    )


# object-model: value; representation=struct; interop=portable
@dataclass(frozen=True, slots=True)
class DialoguePipelineTrace:
    """理解→过程→结果的可跨语言整数 trace。"""

    understanding_tokens: tuple[tuple[int, ...], ...]
    process_candidate_count: int
    process_selected_key: tuple[int, int]
    input_exact: int
    result_tokens: tuple[tuple[int, ...], ...]
    confidence_permille: int
    result_mode: int = DIALOGUE_RESULT_EXACT
    transformation_count: int = 0
    support_count: int = 1

    def __post_init__(self) -> None:
        if type(self.understanding_tokens) is not tuple:
            raise TypeError("understanding_tokens 必须是 tuple")
        for ordinal, token in enumerate(self.understanding_tokens):
            _int_tuple(token, label=f"understanding_tokens[{ordinal}]",
                       non_empty=True)
        if type(self.process_candidate_count) is not int \
                or self.process_candidate_count < 0:
            raise ValueError("process_candidate_count 必须是非负整数")
        _int_tuple(self.process_selected_key,
                   label="process_selected_key", non_empty=True)
        if len(self.process_selected_key) != 2:
            raise ValueError("process_selected_key 必须是二元 key")
        if type(self.input_exact) is not int or self.input_exact not in {0, 1}:
            raise ValueError("input_exact 必须是 0/1 整数")
        if type(self.result_tokens) is not tuple or not self.result_tokens:
            raise ValueError("result_tokens 不能为空")
        for ordinal, token in enumerate(self.result_tokens):
            _int_tuple(token, label=f"result_tokens[{ordinal}]",
                       non_empty=True)
        if (type(self.confidence_permille) is not int
                or not 0 <= self.confidence_permille <= 1000):
            raise ValueError("confidence_permille 必须是 0..1000 整数")
        if self.result_mode not in _DIALOGUE_RESULT_MODES:
            raise ValueError("result_mode 未注册")
        if ((self.result_mode == DIALOGUE_RESULT_EXACT)
                != bool(self.input_exact)):
            raise ValueError("result_mode 与 input_exact 不一致")
        if (type(self.transformation_count) is not int
                or self.transformation_count < 0
                or type(self.support_count) is not int
                or self.support_count <= 0):
            raise ValueError("dialogue result 计数非法")
        if ((self.result_mode == DIALOGUE_RESULT_TRANSFER)
                != bool(self.transformation_count)):
            raise ValueError("transfer mode 与 transformation_count 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含表层文本的确定性协议键。"""
        flattened_input = tuple(
            value for token in self.understanding_tokens
            for value in (len(token), *token))
        flattened_result = tuple(
            value for token in self.result_tokens
            for value in (len(token), *token))
        return (
            2,
            len(self.understanding_tokens), *flattened_input,
            self.process_candidate_count, *self.process_selected_key,
            self.input_exact,
            len(self.result_tokens), *flattened_result,
            self.confidence_permille,
            self.result_mode,
            self.transformation_count,
            self.support_count,
        )


__all__ = [
    "DIALOGUE_RESULT_CLARIFICATION",
    "DIALOGUE_RESULT_EXACT",
    "DIALOGUE_RESULT_RESPONSE_CLASS",
    "DIALOGUE_RESULT_TRANSFER",
    "DialoguePipelineTrace",
    "DialogueSurfaceTransfer",
    "IntegerSurfaceToken",
    "integer_surface_tokens",
    "integer_token_features",
    "integer_token_values",
    "transfer_dialogue_surface",
]
