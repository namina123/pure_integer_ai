"""从词形命中 lattice 生成连续 OOV 和多边界分词候选。

本模块只处理 Unicode 码点位置和候选边界，不创建 Concept、Sense 或最终结构事实。
候选预算由调用方注入；词频不参与排序，避免高频单轴把长词直接固化为唯一答案。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


@dataclass(frozen=True, order=True)
class SegmentationPart:
    """一个候选中的半开码点 span、surface 和词形目录命中标记。"""

    start: int
    end: int
    surface: str
    known_word_form: bool

    def __post_init__(self) -> None:
        assert_int(self.start, self.end, _where="SegmentationPart")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("分词 part 的半开 span 非法")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("分词 part surface 必须是非空字符串")
        if type(self.known_word_form) is not bool:
            raise TypeError("known_word_form 必须是 bool")


@dataclass(frozen=True, order=True)
class SegmentationCandidate:
    """覆盖输入中全部非空白码点的一种边界方案。"""

    parts: tuple[SegmentationPart, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parts, tuple) or not self.parts:
            raise ValueError("分词候选必须包含至少一个 part")
        previous_end = -1
        for part in self.parts:
            if not isinstance(part, SegmentationPart):
                raise TypeError("分词候选只能包含 SegmentationPart")
            if part.start < previous_end:
                raise ValueError("分词候选 part 不得重叠或逆序")
            previous_end = part.end

    @property
    def tokens(self) -> tuple[str, ...]:
        """返回供兼容消费者使用的 token 序。"""
        return tuple(part.surface for part in self.parts)

    @property
    def known_codepoints(self) -> int:
        """统计由词形目录支持的码点覆盖量。"""
        return sum(
            part.end - part.start
            for part in self.parts
            if part.known_word_form)

    @property
    def unknown_spans(self) -> int:
        """统计连续未知 span 数，不把 span 长度混入同一 strength。"""
        return sum(not part.known_word_form for part in self.parts)

    def stable_key(self) -> tuple[int, ...]:
        """返回纯 span 边界键，词形是否已知只作为 Evidence 而不进入身份。"""
        values: list[int] = [len(self.parts)]
        for part in self.parts:
            values.extend((part.start, part.end))
        return tuple(values)


def _candidate_rank(candidate: SegmentationCandidate) -> tuple:
    """按分轴证据排序，不读取频率或单一综合 strength。"""
    return (
        -candidate.known_codepoints,
        candidate.unknown_spans,
        len(candidate.parts),
        candidate.stable_key(),
    )


def _unknown_run_end(
        text: str, lattice: tuple[tuple[str, ...], ...], start: int) -> int:
    """找到下一个空白或已有词形起点，形成最大连续 OOV span。"""
    end = start + 1
    while end < len(text):
        if text[end].isspace() or lattice[end]:
            break
        end += 1
    return end


def _fmm_candidate(
        text: str,
        lattice: tuple[tuple[str, ...], ...]) -> SegmentationCandidate:
    """构造原始最长匹配 + 单码点 OOV 回退基线，保证不会被 Top-K 裁掉。"""
    parts: list[SegmentationPart] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        matched = next((
            surface for surface in lattice[pos]
            if (not any(char.isspace() for char in surface)
                and text[pos:pos + len(surface)] == surface)
        ), None)
        if matched is None:
            matched = text[pos:pos + 1]
            known = False
        else:
            known = True
        end = pos + len(matched)
        parts.append(SegmentationPart(pos, end, matched, known))
        pos = end
    return SegmentationCandidate(tuple(parts))


def _character_candidate(text: str) -> SegmentationCandidate:
    """构造不相信任何长词边界的全字符回退基线。"""
    return SegmentationCandidate(tuple(
        SegmentationPart(index, index + 1, char, False)
        for index, char in enumerate(text)
        if not char.isspace()
    ))


def build_segmentation_candidates(
        text: str,
        lattice: tuple[tuple[str, ...], ...],
        *, candidate_limit: int,
        ) -> tuple[SegmentationCandidate, ...]:
    """在显式预算内生成词形组合、字符回退和连续 OOV 候选。"""
    if not isinstance(text, str):
        raise TypeError("分词输入必须是字符串")
    assert_int(candidate_limit, _where="candidate_limit")
    if candidate_limit < 3:
        raise ValueError("多候选预算必须至少为 3")
    if len(lattice) != len(text):
        raise ValueError("词形 lattice 长度必须等于输入码点数")
    if not text or not text.strip():
        return ()

    paths: dict[int, tuple[tuple[SegmentationPart, ...], ...]] = {
        len(text): ((),),
    }
    for pos in range(len(text) - 1, -1, -1):
        if text[pos].isspace():
            paths[pos] = paths[pos + 1]
            continue
        options: list[SegmentationPart] = []
        for surface in lattice[pos]:
            end = pos + len(surface)
            if end > len(text) or text[pos:end] != surface:
                raise ValueError("词形 lattice 与输入 surface 不一致")
            if any(char.isspace() for char in surface):
                continue
            options.append(SegmentationPart(pos, end, surface, True))

        has_single_known = any(part.end == pos + 1 for part in options)
        if not has_single_known:
            options.append(SegmentationPart(
                pos, pos + 1, text[pos:pos + 1], False))
        if not lattice[pos]:
            run_end = _unknown_run_end(text, lattice, pos)
            if run_end > pos + 1:
                options.append(SegmentationPart(
                    pos, run_end, text[pos:run_end], False))

        combined: dict[tuple[tuple[int, int], ...],
                       tuple[SegmentationPart, ...]] = {}
        for option in options:
            for tail in paths.get(option.end, ()):
                candidate_parts = (option, *tail)
                key = tuple(
                    (part.start, part.end)
                    for part in candidate_parts)
                existing = combined.get(key)
                if existing is None or _candidate_rank(
                        SegmentationCandidate(candidate_parts)) < _candidate_rank(
                            SegmentationCandidate(existing)):
                    combined[key] = candidate_parts
        ranked = sorted(
            (SegmentationCandidate(parts) for parts in combined.values()),
            key=_candidate_rank,
        )
        paths[pos] = tuple(
            candidate.parts for candidate in ranked[:candidate_limit])

    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    ranked_candidates = tuple(
        SegmentationCandidate(parts) for parts in paths.get(start, ()))
    required = (_fmm_candidate(text, lattice), _character_candidate(text))
    all_candidates: dict[tuple[int, ...], SegmentationCandidate] = {}
    for candidate in (*required, *ranked_candidates):
        key = candidate.stable_key()
        existing = all_candidates.get(key)
        if existing is None or _candidate_rank(candidate) < _candidate_rank(existing):
            all_candidates[key] = candidate
    selected: dict[tuple[int, ...], SegmentationCandidate] = {}
    for required_candidate in required:
        candidate = all_candidates[required_candidate.stable_key()]
        selected.setdefault(candidate.stable_key(), candidate)
    for candidate in sorted(all_candidates.values(), key=_candidate_rank):
        selected.setdefault(candidate.stable_key(), candidate)
        if len(selected) >= candidate_limit:
            break
    return tuple(sorted(selected.values(), key=_candidate_rank))


__all__ = [
    "SegmentationCandidate",
    "SegmentationPart",
    "build_segmentation_candidates",
]
