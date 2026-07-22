"""把 typed language 多维反馈转换为 connector H-00/H-04 状态变化。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorDiscourseMapper,
    LanguageConnectorPropositionMapper,
    LanguageConnectorSyntaxMapper,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.language_generation_connector_candidate import (
    LanguageConnectorCandidateRuntime,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
    TypedLanguageRewardSignal,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
)


_APPLICABILITIES = frozenset({
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_UNKNOWN,
})
_VERDICTS = frozenset({
    VERDICT_SUPPORT,
    VERDICT_REFUTE,
    VERDICT_UNKNOWN,
    VERDICT_CONFLICTED,
})


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键添加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _take(
        values: tuple[int, ...], cursor: int, *, label: str,
        ) -> tuple[tuple[int, ...], int]:
    """从长度前缀整数流读取一个非空键段。"""
    if cursor >= len(values):
        raise ValueError(f"connector stage4 历史缺少 {label} 长度")
    size = values[cursor]
    cursor += 1
    if (type(size) is not int or size <= 0
            or cursor + size > len(values)):
        raise ValueError(f"connector stage4 历史 {label} 长度非法")
    return values[cursor:cursor + size], cursor + size


def _outcomes(
        values: tuple[tuple[int, int], ...], *, label: str,
        ) -> tuple[tuple[int, int], ...]:
    """核验并规范化 applicability/verdict 组合。"""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{label} 必须是非空 outcome tuple")
    normalized = []
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{label} outcome 必须是二元 tuple")
        applicability, verdict = value
        assert_int(applicability, verdict, _where=label)
        if (type(applicability) is not int
                or applicability not in _APPLICABILITIES
                or type(verdict) is not int
                or verdict not in _VERDICTS):
            raise ValueError(f"{label} outcome 未注册")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} outcome 不得重复")
    return tuple(sorted(normalized))


def _episode_identity(episode: TypedLanguageEpisode) -> tuple[int, ...]:
    """用来源化 episode scope 和 round 建立非递归事件身份。"""
    return (
        1,
        episode.round_id,
        *_packed(episode.source.stable_key()),
        *_packed(episode.scope.stable_key()),
    )


def _signal_trace(signal: TypedLanguageRewardSignal) -> tuple[int, ...]:
    """保留 stage4 实际消费字段，避免把递归 execution claim 内联进图事件。"""
    source_key = () if signal.source is None else signal.source.stable_key()
    scope_key = () if signal.scope is None else signal.scope.stable_key()
    failure_key = (() if signal.operational_failure is None else tuple(
        ord(item) for item in signal.operational_failure))
    return (
        *_packed(signal.dimension.stable_key()),
        *_packed(signal.verifier.stable_key()),
        signal.applicability,
        signal.verdict,
        *_packed(signal.detail),
        *_packed(source_key),
        *_packed(scope_key),
        *_packed(failure_key),
    )


def _batch_identity(
        episode_keys: tuple[tuple[int, ...], ...],
        hypothesis: HypothesisKey,
        ) -> tuple[int, ...]:
    """用规范 episode 集和 exact Hypothesis 建立顺序无关批次身份。"""
    ordered = tuple(sorted(episode_keys))
    return (
        1,
        len(ordered),
        *(value for key in ordered for value in _packed(key)),
        *_packed(hypothesis.stable_key()),
    )


@dataclass(frozen=True)
class LanguageConnectorSignalRoute:
    """声明一个 dimension/verifier 的哪些结果支持或反驳 connector。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    support_outcomes: tuple[tuple[int, int], ...]
    refute_outcomes: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("connector feedback dimension 类型错误")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("connector feedback verifier 类型错误")
        support = _outcomes(
            self.support_outcomes,
            label="connector support outcomes",
        )
        refute = _outcomes(
            self.refute_outcomes,
            label="connector refute outcomes",
        )
        if set(support) & set(refute):
            raise ValueError("同一 signal outcome 不得同时支持和反驳")
        object.__setattr__(self, "support_outcomes", support)
        object.__setattr__(self, "refute_outcomes", refute)

    def stable_key(self) -> tuple[int, ...]:
        """返回维度、verifier 和两类 outcome 路由。"""
        return (
            *_packed(self.dimension.stable_key()),
            *_packed(self.verifier.stable_key()),
            len(self.support_outcomes),
            *(value for pair in self.support_outcomes for value in pair),
            len(self.refute_outcomes),
            *(value for pair in self.refute_outcomes for value in pair),
        )

    def stance(self, signal: TypedLanguageRewardSignal) -> int:
        """把精确匹配的 typed signal 解释为三态 Evidence stance。"""
        if (signal.dimension != self.dimension
                or signal.verifier != self.verifier):
            raise ValueError("connector signal 与 route 身份不一致")
        if signal.operational_failure is not None:
            return EVIDENCE_UNKNOWN
        outcome = (signal.applicability, signal.verdict)
        if outcome in self.support_outcomes:
            return EVIDENCE_SUPPORT
        if outcome in self.refute_outcomes:
            return EVIDENCE_REFUTE
        return EVIDENCE_UNKNOWN


@dataclass(frozen=True)
class LanguageConnectorStage4Policy:
    """注入反馈 route、独立 verifier manifest 来源和事件命名空间。"""

    routes: tuple[LanguageConnectorSignalRoute, ...]
    verifier_source: SourceRef
    event_namespace: tuple[int, ...]
    active_purpose: ObjectIdentity
    trial_purpose: ObjectIdentity

    def __post_init__(self) -> None:
        if (not isinstance(self.routes, tuple) or not self.routes
                or any(not isinstance(item, LanguageConnectorSignalRoute)
                       for item in self.routes)):
            raise TypeError("connector stage4 routes 类型错误")
        keys = tuple((item.dimension, item.verifier) for item in self.routes)
        if len(set(keys)) != len(keys):
            raise ValueError("connector stage4 route 不得重复")
        if not isinstance(self.verifier_source, SourceRef):
            raise TypeError("connector stage4 verifier source 类型错误")
        _strict_key(
            self.event_namespace,
            label="connector stage4 event namespace",
        )
        for label, purpose in (
                ("active", self.active_purpose),
                ("trial", self.trial_purpose)):
            if (not isinstance(purpose, ObjectIdentity)
                    or purpose.object_kind != OBJECT_MINIMAL_INSTRUCTION):
                raise TypeError(f"connector stage4 {label} purpose 类型错误")
        if self.active_purpose == self.trial_purpose:
            raise ValueError("connector stage4 active/trial purpose 必须互异")
        object.__setattr__(self, "routes", tuple(sorted(
            self.routes,
            key=lambda item: (
                item.dimension.stable_key(),
                item.verifier.stable_key(),
            ),
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 route、verifier 来源和事件命名空间。"""
        return (
            len(self.routes),
            *(value for route in self.routes
              for value in _packed(route.stable_key())),
            *_packed(self.verifier_source.stable_key()),
            *_packed(self.event_namespace),
            *_packed(self.active_purpose.stable_key()),
            *_packed(self.trial_purpose.stable_key()),
        )


@dataclass(frozen=True)
class LanguageConnectorStage4Outcome:
    """一次 episode 对唯一 connector 产生的聚合 stance 和学习结果。"""

    episode_key: tuple[int, ...]
    connector: LanguageGenerationConnectorTemplate
    stance: int
    learning: CandidateLearningOutcome

    def __post_init__(self) -> None:
        _strict_key(self.episode_key, label="connector stage4 episode key")
        if not isinstance(self.connector, LanguageGenerationConnectorTemplate):
            raise TypeError("connector stage4 template 类型错误")
        if self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("connector stage4 stance 未注册")
        if not isinstance(self.learning, CandidateLearningOutcome):
            raise TypeError("connector stage4 learning outcome 类型错误")
        if self.learning.verification.stance != self.stance:
            raise ValueError("connector stage4 聚合 stance 与 Evidence 不一致")


@dataclass(frozen=True)
class LanguageConnectorStage4Report:
    """保留每个 connector 的真实 prediction/Evidence/decision/projection。"""

    outcomes: tuple[LanguageConnectorStage4Outcome, ...]

    @property
    def complete(self) -> bool:
        """至少处理一个 episode 且每项都有 lifecycle 投影时才完成。"""
        return bool(self.outcomes) and all(
            item.learning.projection is not None for item in self.outcomes)

    @property
    def changed_count(self) -> int:
        """返回真实写入 lifecycle Event 的 connector 数量。"""
        return sum(
            item.learning.projection is not None for item in self.outcomes)


@dataclass(frozen=True)
class _PreparedFeedback:
    """批量零写预检后的唯一 connector、聚合 stance 和 trace。"""

    episode: TypedLanguageEpisode
    episode_key: tuple[int, ...]
    template: LanguageGenerationConnectorTemplate
    hypothesis: HypothesisKey
    stance: int
    trace: tuple[int, ...]


class LanguageConnectorStage4Runtime:
    """消费 typed episode，并通过通用 candidate runtime 更新 connector 状态。"""

    def __init__(
            self,
            candidates: LanguageConnectorCandidateRuntime,
            policy: LanguageConnectorStage4Policy,
            ) -> None:
        if not isinstance(candidates, LanguageConnectorCandidateRuntime):
            raise TypeError("connector stage4 candidates 类型错误")
        if not isinstance(policy, LanguageConnectorStage4Policy):
            raise TypeError("connector stage4 policy 类型错误")
        self.candidates = candidates
        self.policy = policy
        self._processed: dict[tuple[int, ...], LanguageConnectorStage4Outcome] = {}
        self._processed_traces: dict[
            tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
        self._restore_processed()

    def apply(
            self,
            episodes: tuple[TypedLanguageEpisode, ...],
            ) -> LanguageConnectorStage4Report:
        """批量预检 typed 反馈后逐 connector 形成可回溯 lifecycle 转换。"""
        if not isinstance(episodes, tuple) or not episodes:
            raise ValueError("connector stage4 episodes 必须是非空 tuple")
        if any(not isinstance(item, TypedLanguageEpisode) for item in episodes):
            raise TypeError("connector stage4 只能消费 TypedLanguageEpisode")
        replayed = self._replayed_report(episodes)
        if replayed is not None:
            return replayed
        prepared = tuple(self._prepare(item) for item in episodes)
        episode_keys = tuple(item.episode_key for item in prepared)
        if len(set(episode_keys)) != len(episode_keys):
            raise ValueError("同批 stage4 episode 不得重复")

        outcomes = []
        grouped: dict[HypothesisKey, list[_PreparedFeedback]] = {}
        for item in prepared:
            grouped.setdefault(item.hypothesis, []).append(item)
        for hypothesis in sorted(grouped, key=HypothesisKey.stable_key):
            items = tuple(sorted(
                grouped[hypothesis],
                key=lambda current: current.episode_key,
            ))
            template = items[0].template
            if any(item.template != template for item in items):
                raise ValueError("同一 connector Hypothesis 对应多个权威理论")
            stances = tuple(item.stance for item in items)
            if EVIDENCE_REFUTE in stances:
                stance = EVIDENCE_REFUTE
            elif all(item == EVIDENCE_SUPPORT for item in stances):
                stance = EVIDENCE_SUPPORT
            else:
                stance = EVIDENCE_UNKNOWN
            batch_key = _batch_identity(
                tuple(item.episode_key for item in items),
                hypothesis,
            )
            existing = self._processed.get(batch_key)
            if existing is not None:
                if (self._processed_traces[batch_key]
                        != tuple(item.trace for item in items)
                        or existing.connector != template
                        or existing.stance != stance):
                    raise RuntimeError("已处理 stage4 episode 的理论或 stance 漂移")
                outcomes.append(existing)
                continue
            timestamp_seq, resolve_seq, projection_seq = (
                self.candidates.learning.next_timestamps(3))
            event_key = (
                *_packed(self.policy.event_namespace),
                *_packed(batch_key),
                *_packed(template.connector.stable_key()),
            )
            supported = (
                (template.connector,)
                if stance == EVIDENCE_SUPPORT else ())
            refuted = (
                (template.connector,)
                if stance == EVIDENCE_REFUTE else ())
            trace = (
                *_packed(self.policy.stable_key()),
                len(items),
                *(value for item in items
                  for value in _packed(item.trace)),
            )
            learning = self.candidates.recognize(
                hypothesis,
                observation=self.policy.verifier_source,
                scope=document_scope(self.policy.verifier_source),
                event_key=event_key,
                visible_inputs=(
                    template.connector,
                    template.language_branch,
                    template.proposition_structure,
                    template.predicate,
                ),
                predicted=template.connector,
                revealed=RevealedObjectObservation(
                    self.policy.verifier_source,
                    document_scope(self.policy.verifier_source),
                    event_key,
                    self.policy.verifier_source,
                    supported,
                    refuted,
                    trace,
                ),
                timestamp_seq=timestamp_seq,
                resolve_timestamp_seq=resolve_seq,
                projection_timestamp_seq=projection_seq,
                archive_refuted=stance == EVIDENCE_REFUTE,
            )
            outcome = LanguageConnectorStage4Outcome(
                batch_key,
                template,
                stance,
                learning,
            )
            self._processed[batch_key] = outcome
            self._processed_traces[batch_key] = tuple(
                item.trace for item in items)
            outcomes.append(outcome)
        return LanguageConnectorStage4Report(tuple(outcomes))

    def _restore_processed(self) -> None:
        """从 H-00/H-04 与候选图重建 stage4 批次游标，不建立第二真源。"""
        engine = self.candidates.learning.engine
        for hypothesis in engine.ledger.hypotheses():
            for recognition in engine.recognition_history(hypothesis):
                parsed = self._stage4_event_key(
                    recognition.prediction.event_key)
                if parsed is None:
                    continue
                batch_key, connector = parsed
                if (recognition.prediction.observation
                        != self.policy.verifier_source
                        or recognition.verification.source
                        != self.policy.verifier_source
                        or recognition.prediction.scope
                        != document_scope(self.policy.verifier_source)
                        or recognition.prediction.predicted != connector):
                    raise RuntimeError(
                        "恢复 stage4 prediction 的来源、scope 或目标漂移")
                template = self.candidates.definition_graph.read(
                    connector).definition
                definition = engine.definition(hypothesis)
                if (definition.candidate != connector
                        or template.connector != connector):
                    raise RuntimeError(
                        "恢复 stage4 Evidence 未绑定同一 connector 理论")
                traces, episode_keys = self._batch_traces(
                    recognition.verification.trace)
                if _batch_identity(episode_keys, hypothesis) != batch_key:
                    raise RuntimeError(
                        "恢复 stage4 批次身份与 Evidence trace 不一致")
                decisions = tuple(
                    item for item in engine.resolver.decision_history(
                        hypothesis)
                    if item.timestamp_seq
                    == recognition.evidence.timestamp_seq + 1
                )
                if len(decisions) != 1:
                    raise RuntimeError(
                        "恢复 stage4 Evidence 缺少唯一相邻 H-04 决策")
                decision = decisions[0]
                trace = decision.candidate(hypothesis)
                active_ids = {
                    *trace.after.support_evidence_ids,
                    *trace.after.refute_evidence_ids,
                    *trace.after.unknown_evidence_ids,
                }
                if recognition.evidence.evidence_id not in active_ids:
                    raise RuntimeError(
                        "恢复 stage4 H-04 决策未消费对应 Evidence")
                projection = self._projection_for_decision(
                    connector,
                    decision,
                    recognition.evidence,
                )
                learning = CandidateLearningOutcome(
                    recognition.prediction,
                    recognition.verification,
                    recognition.evidence,
                    decision,
                    projection,
                )
                outcome = LanguageConnectorStage4Outcome(
                    batch_key,
                    template,
                    recognition.verification.stance,
                    learning,
                )
                existing = self._processed.get(batch_key)
                if (existing is not None
                        and (existing != outcome
                             or self._processed_traces[batch_key] != traces)):
                    raise RuntimeError("恢复 stage4 批次身份命中不同历史")
                self._processed[batch_key] = outcome
                self._processed_traces[batch_key] = traces

    def _stage4_event_key(
            self,
            event_key: tuple[int, ...],
            ) -> tuple[tuple[int, ...], ObjectIdentity] | None:
        """识别本 policy 的 stage4 事件键，并恢复批次和 connector 身份。"""
        if (not event_key
                or event_key[0] != len(self.policy.event_namespace)):
            return None
        namespace, cursor = _take(event_key, 0, label="event namespace")
        if namespace != self.policy.event_namespace:
            return None
        batch_key, cursor = _take(
            event_key, cursor, label="batch key")
        connector_key, cursor = _take(
            event_key, cursor, label="connector")
        if cursor != len(event_key):
            raise RuntimeError("恢复 stage4 event key 含尾随字段")
        return batch_key, ObjectIdentity.from_stable_key(connector_key)

    def _batch_traces(
            self,
            trace: tuple[int, ...],
            ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        """从 CandidateVerification trace 恢复逐 episode 消费键和 episode 身份。"""
        policy_key, cursor = _take(trace, 0, label="policy")
        if policy_key != self.policy.stable_key():
            raise RuntimeError("恢复 stage4 trace 的 policy 漂移")
        if cursor >= len(trace):
            raise RuntimeError("恢复 stage4 trace 缺少 episode 数量")
        count = trace[cursor]
        cursor += 1
        if type(count) is not int or count <= 0:
            raise RuntimeError("恢复 stage4 trace 的 episode 数量非法")
        traces = []
        episode_keys = []
        for _index in range(count):
            item, cursor = _take(trace, cursor, label="episode trace")
            episode_key = self._validate_episode_trace(item)
            traces.append(item)
            episode_keys.append(episode_key)
        if cursor != len(trace):
            raise RuntimeError("恢复 stage4 batch trace 含尾随字段")
        return tuple(traces), tuple(episode_keys)

    def _validate_episode_trace(
            self,
            trace: tuple[int, ...],
            ) -> tuple[int, ...]:
        """核验逐 episode trace 的 policy、purpose 和 routed signal 结构。"""
        policy_key, cursor = _take(trace, 0, label="episode policy")
        if policy_key != self.policy.stable_key():
            raise RuntimeError("恢复 stage4 episode trace 的 policy 漂移")
        episode_key, cursor = _take(
            trace, cursor, label="episode identity")
        purpose_key, cursor = _take(trace, cursor, label="surface purpose")
        purpose = ObjectIdentity.from_stable_key(purpose_key)
        if purpose not in {
                self.policy.active_purpose,
                self.policy.trial_purpose}:
            raise RuntimeError("恢复 stage4 episode trace 的 purpose 未注册")
        if cursor >= len(trace):
            raise RuntimeError("恢复 stage4 episode trace 缺 signal 数量")
        count = trace[cursor]
        cursor += 1
        if type(count) is not int or count != len(self.policy.routes):
            raise RuntimeError("恢复 stage4 episode trace 的 signal 数量漂移")
        for _index in range(count):
            _signal, cursor = _take(trace, cursor, label="signal trace")
        if cursor != len(trace):
            raise RuntimeError("恢复 stage4 episode trace 含尾随字段")
        return episode_key

    def _projection_for_decision(
            self,
            connector: ObjectIdentity,
            decision,
            evidence,
            ) -> CandidateGraphProjection | None:
        """恢复本次 decision 对应的图投影前缀；unknown 可以没有投影。"""
        candidate_ref = self.candidates.learning.graph.ontology.resolve(
            connector)
        if candidate_ref is None:
            raise RuntimeError("恢复 stage4 projection 缺少 connector 图对象")
        graph = self.candidates.learning.graph
        history = graph.history(candidate_ref)
        matches = tuple(
            index for index, item in enumerate(history)
            if item.definition.decision_key == decision.stable_key()
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("恢复 stage4 decision 对应多个图 Event")
        index = matches[0]
        event = history[index].definition
        if evidence.stable_key() not in event.evidence_keys:
            raise RuntimeError("恢复 stage4 图 Event 未引用对应 Evidence")
        replacement = None
        for item in history[:index + 1]:
            if item.definition.replacement is not None:
                replacement = item.definition.replacement
        materialized = graph.read_definition(event.hypothesis)
        return CandidateGraphProjection(
            materialized,
            event.to_state,
            history[:index + 1],
            replacement,
        )

    def _replayed_report(
            self,
            episodes: tuple[TypedLanguageEpisode, ...],
            ) -> LanguageConnectorStage4Report | None:
        """在候选状态变化前识别完整旧批次，并拒绝同身份内容漂移。"""
        episode_keys = tuple(_episode_identity(item) for item in episodes)
        if len(set(episode_keys)) != len(episode_keys):
            raise ValueError("同批 stage4 episode 不得重复")
        grouped: dict[HypothesisKey, list[TypedLanguageEpisode]] = {}
        for episode in episodes:
            execution = episode.production.execution
            if execution is None or execution.surface is None:
                return None
            attribution = execution.surface.preview.request.attribution
            if attribution is None:
                return None
            grouped.setdefault(attribution.hypothesis, []).append(episode)
        outcomes = []
        for hypothesis in sorted(grouped, key=HypothesisKey.stable_key):
            items = tuple(sorted(
                grouped[hypothesis],
                key=_episode_identity,
            ))
            batch_key = _batch_identity(
                tuple(_episode_identity(item) for item in items),
                hypothesis,
            )
            existing = self._processed.get(batch_key)
            if existing is None:
                return None
            prepared = tuple(
                self._prepare(item, restored=existing)
                for item in items
            )
            traces = tuple(item.trace for item in prepared)
            stances = tuple(item.stance for item in prepared)
            stance = (
                EVIDENCE_REFUTE
                if EVIDENCE_REFUTE in stances
                else EVIDENCE_SUPPORT
                if stances and all(
                    item == EVIDENCE_SUPPORT for item in stances)
                else EVIDENCE_UNKNOWN
            )
            if (self._processed_traces[batch_key] != traces
                    or existing.stance != stance
                    or any(item.template != existing.connector
                           or item.hypothesis != hypothesis
                           for item in prepared)):
                raise RuntimeError("已处理 stage4 episode 的消费内容漂移")
            outcomes.append(existing)
        return LanguageConnectorStage4Report(tuple(outcomes))

    def _prepare(
            self,
            episode: TypedLanguageEpisode,
            *,
            restored: LanguageConnectorStage4Outcome | None = None,
            ) -> _PreparedFeedback:
        """零写核验 episode、active connector、完整 route 和聚合三态。"""
        if episode.read_only:
            raise ValueError("read-only typed episode 不得写 connector Evidence")
        if episode.scope.source != episode.source:
            raise ValueError("typed episode scope 未绑定同一 observation source")
        execution = episode.production.execution
        if (execution is None or execution.surface is None
                or not episode.generation_complete):
            raise ValueError("connector stage4 只接受完整 typed generation")
        if episode.production.postcheck is None or not episode.signals:
            raise ValueError("connector stage4 缺少同次 G-04 typed signal")
        request = execution.surface.preview.request
        selection = request.structure.selection
        attribution = request.attribution
        if attribution is None:
            raise ValueError("connector stage4 surface 缺少显式理论归属")
        if restored is not None:
            if (attribution.theory != restored.connector.connector
                    or attribution.hypothesis
                    != restored.learning.prediction.hypothesis):
                raise RuntimeError("恢复 stage4 理论归属与重放 episode 漂移")
            if attribution.purpose not in {
                    self.policy.active_purpose,
                    self.policy.trial_purpose}:
                raise ValueError("恢复 stage4 surface purpose 未被 policy 注册")
            registry = LanguageGenerationConnectorRegistry(
                self.candidates.definition_graph.value_protocol,
                (restored.connector,),
            )
        elif attribution.purpose == self.policy.active_purpose:
            registry = self.candidates.active_registry()
        elif attribution.purpose == self.policy.trial_purpose:
            trial = self.candidates.trial_template(attribution.hypothesis)
            registry = LanguageGenerationConnectorRegistry(
                self.candidates.definition_graph.value_protocol,
                (trial,),
            )
        else:
            raise ValueError("connector surface purpose 未被 stage4 注册")
        template, candidate = registry.match(selection)
        self._validate_surface_binding(
            episode,
            registry,
            template,
            candidate,
        )
        if attribution.theory != template.connector:
            raise ValueError("connector stage4 surface 理论归属漂移")
        hypothesis = self.candidates.learning.hypothesis_for_candidate(
            template.connector)
        if attribution.hypothesis != hypothesis:
            raise ValueError("connector stage4 surface Hypothesis 归属漂移")
        definition = self.candidates.learning.engine.definition(hypothesis)
        if (episode.source in definition.forming_sources
                or self.policy.verifier_source in definition.forming_sources):
            raise ValueError("connector forming 来源不得冒充 stage4 recognition")

        signals = {
            (item.dimension, item.verifier): item
            for item in episode.signals
        }
        if len(signals) != len(episode.signals):
            raise ValueError("typed episode signal 身份重复")
        routed = []
        stances = []
        for route in self.policy.routes:
            signal = signals.get((route.dimension, route.verifier))
            if signal is None:
                raise ValueError("typed episode 缺少 connector stage4 必需 route")
            routed.append(signal)
            stances.append(route.stance(signal))
        if EVIDENCE_REFUTE in stances:
            stance = EVIDENCE_REFUTE
        elif stances and all(item == EVIDENCE_SUPPORT for item in stances):
            stance = EVIDENCE_SUPPORT
        else:
            stance = EVIDENCE_UNKNOWN
        trace = (
            *_packed(self.policy.stable_key()),
            *_packed(_episode_identity(episode)),
            *_packed(attribution.purpose.stable_key()),
            len(routed),
            *(value for signal in routed
              for value in _packed(_signal_trace(signal))),
        )
        return _PreparedFeedback(
            episode,
            _episode_identity(episode),
            template,
            hypothesis,
            stance,
            trace,
        )

    @staticmethod
    def _validate_surface_binding(
            episode: TypedLanguageEpisode,
            registry,
            template: LanguageGenerationConnectorTemplate,
            candidate,
            ) -> None:
        """双向证明本次完整 surface 确由当前 active connector 生成。"""
        execution = episode.production.execution
        if execution is None or execution.surface is None:
            raise ValueError("connector stage4 缺完整 surface")
        request = execution.surface.preview.request
        structure = request.structure
        selection = structure.selection

        discourse = LanguageConnectorDiscourseMapper(registry).plan(selection)
        propositions = LanguageConnectorPropositionMapper().plan(
            selection,
            discourse,
        )
        syntax = LanguageConnectorSyntaxMapper(registry).plan(
            selection,
            discourse,
            propositions,
        )
        if (structure.discourse != discourse
                or structure.propositions != propositions
                or structure.syntax != syntax):
            raise ValueError("typed episode structure 未由当前 active connector 产生")
        if request.branch != template.language_branch:
            raise ValueError("typed episode surface branch 与 connector 不一致")

        sentences = syntax.sentences
        executed = request.execution.sentences
        if len(sentences) != 1 or len(executed) != 1:
            raise ValueError("connector stage4 当前只接受单句完整执行")
        sentence = sentences[0]
        actual_execution = executed[0]
        expected_values = registry.values(template, candidate.proposition)
        if (sentence.sentence != template.sentence
                or sentence.structure != template.structure
                or sentence.slots != template.slots
                or sentence.values != expected_values
                or actual_execution.obligation != syntax.linearization[0]
                or actual_execution.graph_slots != template.slots):
            raise ValueError("typed episode sentence/slot/value 未绑定当前 connector")
        active_constraints = tuple(
            item.constraint.definition.constraint
            for item in actual_execution.active_constraints
        )
        if (syntax.linearization[0].constraints != template.constraints
                or syntax.linearization[0].context != template.context
                or syntax.linearization[0].reason
                != template.linearization_reason
                or active_constraints != template.constraints):
            raise ValueError("typed episode constraint/context 未绑定当前 connector")

        theory_by_slot = {item.slot: item for item in template.surface}
        directives = request.directive_map()
        expected_keys = {
            (template.sentence, slot.slot) for slot in template.slots}
        if set(directives) != expected_keys:
            raise ValueError("typed episode surface directive 未覆盖 connector slot")
        for key, directive in directives.items():
            theory = theory_by_slot[key[1]]
            if (directive.sentence != template.sentence
                    or directive.action != theory.action
                    or directive.instruction != theory.instruction
                    or directive.surface_prefix_steps
                    != theory.surface_prefix_steps):
                raise ValueError("typed episode surface directive 与 connector 理论漂移")

    def state_key(self) -> tuple:
        """返回 policy、候选 owner 和已处理 episode 的完整状态。"""
        return (
            self.policy.stable_key(),
            self.candidates.state_key(),
            tuple(sorted(
                (key, value.connector.connector.stable_key(), value.stance)
                for key, value in self._processed.items()
            )),
        )


__all__ = [
    "LanguageConnectorSignalRoute",
    "LanguageConnectorStage4Outcome",
    "LanguageConnectorStage4Policy",
    "LanguageConnectorStage4Report",
    "LanguageConnectorStage4Runtime",
]
