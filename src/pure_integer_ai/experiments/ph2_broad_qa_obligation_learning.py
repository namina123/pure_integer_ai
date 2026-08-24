"""从公开问答课程学习 typed obligation 的问式特征。

该模块只消费公开 JSONL 中显式登记的 ``question.typed_intent``。它不读取
评测 gold、来源页面或答案文本，也不把题目映射到任何固定答案。模型只把
足够稳定的问式特征投影为已有 broad-QA obligation ontology；冲突或支持不足
时返回空值，由原有问式/拒答门继续处理。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


OBLIGATION_LEARNING_PROTOCOL_V1 = 1
OBLIGATION_LEARNING_DOMAIN = "pure_integer_ai.broad_qa.typed_obligation.v1"
_ANSWER_KINDS = frozenset({
    "CAUSE", "ENTITY", "LOCATION", "MANNER", "QUANTITY", "TIME", "TYPE",
})
_INTENT_TO_KIND = {
    "ASK_EVENT_TIME": "TIME",
    "ASK_QUANTITY": "QUANTITY",
    "ASK_LOCATION": "LOCATION",
    "ASK_EVENT_AND_RECORD": "ENTITY",
    "ASK_ENTITY": "ENTITY",
    "ASK_CAUSE": "CAUSE",
    "ASK_MANNER": "MANNER",
    "ASK_TYPE": "TYPE",
}
_MIN_SUPPORT = 2
_MIN_MARGIN = 1
_QUESTION_CUE_RE = re.compile(
    r"(?:以|用)何(?:种|種)?方式|"
    r"为什么|為什麼|为何|為何|因何|何故|"
    r"何(?:时|時|年|地|处|處|人|种|種|项|項)|"
    r"哪(?:一|里|裏|裡|边|邊|位|类|類|种|種|本)|"
    r"(?:几|幾)(?:时|時|年|个|個|多|许|許)|"
    r"多少|多(?:大|高|宽|寬|远|遠|长|長)|"
    r"谁|誰|如何|怎么|怎麼|怎样|怎樣|有何(?:作用|用途)")
_CUE_TRANSLATION = str.maketrans({
    "為": "为", "麼": "么", "時": "时", "處": "处",
    "種": "种", "裏": "里", "裡": "里", "邊": "边",
    "類": "类", "幾": "几", "個": "个", "許": "许",
    "寬": "宽", "遠": "远", "長": "长", "誰": "谁",
    "樣": "样",
})


class BroadQaObligationLearningError(ValueError):
    """公开问式课程无法形成确定的 obligation 模型。"""


def _canonical_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise BroadQaObligationLearningError("typed obligation surface 非法")
    return value


def _question_cues(surface: str) -> tuple[str, ...]:
    """抽取问式功能词及其短上下文，不把实体内容作为模型特征。"""
    return tuple(sorted({
        match.group(0).translate(_CUE_TRANSLATION)
        for match in _QUESTION_CUE_RE.finditer(surface)
    }))


@dataclass(frozen=True, slots=True)
class LearnedTypedObligation:
    """可跨语言重建的问式 cue 到 obligation 类型整数模型。"""

    cue_kinds: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    source_sha256: str
    case_count: int

    def __post_init__(self) -> None:
        if (not self.cue_kinds
                or tuple(item[0] for item in self.cue_kinds)
                != tuple(sorted(item[0] for item in self.cue_kinds))
                or len({item[0] for item in self.cue_kinds}) != len(self.cue_kinds)
                or any(
                    not cue or not values
                    or tuple(kind for kind, _ in values)
                    != tuple(sorted(kind for kind, _ in values))
                    or any(kind not in _ANSWER_KINDS or type(count) is not int
                           or count <= 0 for kind, count in values)
                    for cue, values in self.cue_kinds)
                or len(self.source_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.source_sha256)
                or type(self.case_count) is not int or self.case_count <= 0):
            raise BroadQaObligationLearningError("typed obligation model 非法")

    def answer_kinds(self, question: str) -> tuple[str, ...]:
        """返回无冲突且达到支持阈值的 learned kinds；不确定则为空。"""
        if not isinstance(question, str) or not question.strip():
            raise TypeError("question 必须是非空字符串")
        terms = set(_question_cues(question))
        votes: Counter[str] = Counter()
        for cue, values in self.cue_kinds:
            if cue in terms:
                for kind, count in values:
                    votes[kind] += count
        if not votes:
            return ()
        ranked = votes.most_common()
        if ranked[0][1] < _MIN_SUPPORT:
            return ()
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < _MIN_MARGIN:
            return ()
        return (ranked[0][0],)

    def canonical_record(self) -> tuple[int, ...]:
        result = [OBLIGATION_LEARNING_PROTOCOL_V1, self.case_count,
                  len(self.source_sha256)]
        result.extend(ord(item) for item in self.source_sha256)
        result.append(len(self.cue_kinds))
        for cue, values in self.cue_kinds:
            result.extend((len(cue), *map(ord, cue), len(values)))
            for kind, count in values:
                result.extend((len(kind), *map(ord, kind), count))
        return tuple(result)


def _iter_records(paths: Iterable[str | Path]):
    for value in tuple(sorted(Path(item).resolve() for item in paths)):
        if not value.is_file():
            raise BroadQaObligationLearningError(f"课程文件缺失: {value}")
        for line_number, raw in enumerate(value.read_bytes().splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BroadQaObligationLearningError(
                    f"课程 JSONL 非法: {value}:{line_number}") from error
            if isinstance(record, dict):
                yield value, record


def learn_typed_obligations(
        paths: Iterable[str | Path],
        ) -> LearnedTypedObligation:
    """从 train split 的显式 typed_intent 学习问式 obligation。"""
    files = tuple(sorted(Path(item).resolve() for item in paths))
    if not files or len(files) != len(set(files)):
        raise BroadQaObligationLearningError("课程文件 inventory 重复或为空")
    support: dict[str, Counter[str]] = defaultdict(Counter)
    digest = hashlib.sha256()
    case_count = 0
    for path in files:
        if not path.is_file():
            raise BroadQaObligationLearningError(f"课程文件缺失: {path}")
        digest.update(path.as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    for path, record in _iter_records(files):
        question = record.get("question")
        if record.get("split") not in {"train", "course", None}:
            continue
        if not isinstance(question, dict):
            continue
        intent = question.get("typed_intent")
        surface = question.get("question_surface")
        kind = _INTENT_TO_KIND.get(intent)
        if kind is None or not isinstance(surface, str):
            continue
        cues = _question_cues(_canonical_text(surface))
        if not cues:
            continue
        case_count += 1
        for cue in cues:
            support[cue][kind] += 1
    if not case_count:
        raise BroadQaObligationLearningError("没有可学习的 typed_intent train case")
    cue_kinds = []
    for cue, values in support.items():
        # 保留低支持 cue 作为可审计模型证据，但 answer_kinds() 仍要求
        # _MIN_SUPPORT；这样稀疏公开课程不会阻断主线，也不会造成运行时
        # 把一次偶然表面当成确定 obligation。
        filtered = tuple(sorted(values.items()))
        if filtered:
            cue_kinds.append((cue, filtered))
    if not cue_kinds:
        raise BroadQaObligationLearningError("typed_intent cue 支持不足")
    source_sha = digest.hexdigest()
    return LearnedTypedObligation(tuple(sorted(cue_kinds)), source_sha, case_count)


def typed_obligation_sha256(model: LearnedTypedObligation) -> str:
    if not isinstance(model, LearnedTypedObligation):
        raise TypeError("model 必须是 LearnedTypedObligation")
    payload = b"".join(int(value).to_bytes(8, "big", signed=False)
                       for value in model.canonical_record())
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BroadQaObligationLearningError",
    "LearnedTypedObligation",
    "OBLIGATION_LEARNING_DOMAIN",
    "OBLIGATION_LEARNING_PROTOCOL_V1",
    "learn_typed_obligations",
    "typed_obligation_sha256",
]
