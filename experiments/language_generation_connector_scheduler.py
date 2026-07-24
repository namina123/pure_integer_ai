"""按启动恢复快照调度 active connector 与 exact forming trial。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnectorError,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorTemplate,
    LanguageConnectorValueProtocol,
)


ConnectorScheduleEntry = tuple[
    LanguageGenerationConnectorTemplate,
    HypothesisKey,
]


def _normalize_entries(
        entries: tuple[ConnectorScheduleEntry, ...],
        *,
        label: str,
        ) -> tuple[ConnectorScheduleEntry, ...]:
    """核验并规范化一组模板与 exact Hypothesis 调度绑定。"""
    if not isinstance(entries, tuple):
        raise TypeError(f"{label} 必须是 tuple")
    normalized = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{label} entry 必须是 template/Hypothesis 对")
        template, hypothesis = entry
        if not isinstance(template, LanguageGenerationConnectorTemplate):
            raise TypeError(f"{label} template 类型错误")
        if not isinstance(hypothesis, HypothesisKey):
            raise TypeError(f"{label} hypothesis 类型错误")
        normalized.append(entry)
    connectors = tuple(item[0].connector for item in normalized)
    hypotheses = tuple(item[1] for item in normalized)
    if len(set(connectors)) != len(connectors):
        raise ValueError(f"{label} 不得重复 connector")
    if len(set(hypotheses)) != len(hypotheses):
        raise ValueError(f"{label} 不得重复 Hypothesis")
    return tuple(sorted(
        normalized,
        key=lambda item: item[0].connector.stable_key(),
    ))


class ScheduledLanguageGenerationConnectorRegistry(
        LanguageGenerationConnectorRegistry):
    """active 优先；无 active 时只允许唯一 exact forming trial。"""

    def __init__(
            self,
            value_protocol: LanguageConnectorValueProtocol,
            active_entries: tuple[ConnectorScheduleEntry, ...],
            trial_entries: tuple[ConnectorScheduleEntry, ...],
            ) -> None:
        """从一次启动恢复快照建立按精确命题键查询的局部索引。"""
        active = _normalize_entries(active_entries, label="active schedule")
        trial = _normalize_entries(trial_entries, label="trial schedule")
        active_connectors = {item[0].connector for item in active}
        trial_connectors = {item[0].connector for item in trial}
        if active_connectors & trial_connectors:
            raise ValueError("同一 connector 不得同时进入 active 和 trial 调度")
        active_hypotheses = {item[1] for item in active}
        trial_hypotheses = {item[1] for item in trial}
        if active_hypotheses & trial_hypotheses:
            raise ValueError("同一 Hypothesis 不得同时进入 active 和 trial 调度")
        templates = tuple(item[0] for item in active + trial)
        if not templates:
            raise LanguageGenerationConnectorError(
                "当前启动快照没有可调度的 active 或 forming connector")
        super().__init__(value_protocol, templates)
        self._active_entries = active
        self._trial_entries = trial
        self._active_by_key = self._index(active)
        self._trial_by_key = self._index(trial)

    @staticmethod
    def _index(entries: tuple[ConnectorScheduleEntry, ...]) -> dict:
        """按模板精确匹配键建立局部多值索引，不在 round 内扫描历史。"""
        result = {}
        for entry in entries:
            result.setdefault(entry[0].match_key(), []).append(entry)
        return {
            key: tuple(sorted(
                values,
                key=lambda item: item[1].stable_key(),
            ))
            for key, values in result.items()
        }

    @property
    def active_entries(self) -> tuple[ConnectorScheduleEntry, ...]:
        """返回启动时恢复的 active 模板及其 exact Hypothesis。"""
        return self._active_entries

    @property
    def trial_entries(self) -> tuple[ConnectorScheduleEntry, ...]:
        """返回启动时恢复的 forming trial 模板及其 exact Hypothesis。"""
        return self._trial_entries

    def match(
            self,
            selection: AnswerContentSelection,
            ) -> tuple[LanguageGenerationConnectorTemplate, object]:
        """按 G-01 实际选择执行 active 优先和唯一 trial 的显式调度。"""
        selected = self.selected_candidates(selection)
        if len(selected) != 1:
            raise LanguageGenerationConnectorError(
                "单命题调度入口不得私选多命题 selection")
        return self.match_candidate(selection, selected[0])

    def match_candidate(
            self,
            selection: AnswerContentSelection,
            candidate: object,
            ) -> tuple[LanguageGenerationConnectorTemplate, object]:
        """为一个 selected candidate 执行 active 优先和唯一 forming trial 调度。"""
        selected = self.selected_candidates(selection)
        candidate_key = getattr(candidate, "stable_key", None)
        if not callable(candidate_key):
            raise TypeError("scheduled connector candidate 缺少稳定身份")
        key_value = candidate_key()
        matches = tuple(
            item for item in selected if item.stable_key() == key_value)
        if len(matches) != 1 or matches[0] != candidate:
            raise LanguageGenerationConnectorError(
                "scheduled connector candidate 不属于当前精确 selection")
        key = self.match_key_for_candidate(selection, candidate)
        active = self._active_by_key.get(key, ())
        if len(active) > 1:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 存在多个 active 模板")
        if active:
            return active[0][0], candidate
        trial = self._trial_by_key.get(key, ())
        if len(trial) > 1:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 存在多个 forming trial")
        if trial:
            return trial[0][0], candidate
        raise LanguageGenerationConnectorError(
            "当前 predicate/structure/LanguageBranch 没有可调度模板")

    def stable_key(self) -> tuple[int, ...]:
        """返回模板定义及 active/trial exact Hypothesis 启动快照。"""
        result = [*super().stable_key(), len(self._active_entries)]
        for template, hypothesis in self._active_entries:
            template_key = template.connector.stable_key()
            hypothesis_key = hypothesis.stable_key()
            result.extend((len(template_key), *template_key))
            result.extend((len(hypothesis_key), *hypothesis_key))
        result.append(len(self._trial_entries))
        for template, hypothesis in self._trial_entries:
            template_key = template.connector.stable_key()
            hypothesis_key = hypothesis.stable_key()
            result.extend((len(template_key), *template_key))
            result.extend((len(hypothesis_key), *hypothesis_key))
        return tuple(result)


__all__ = [
    "ConnectorScheduleEntry",
    "ScheduledLanguageGenerationConnectorRegistry",
]
