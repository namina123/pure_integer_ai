"""把来源化 occurrence 序列接到 H-01 通用条件预测协议。

本适配器只从调用方指定对象种类中接受唯一 typed 候选，并把其完整图身份作为预测
单元。预测总在当前 SourceRef 的观察写入模型之前执行，查询时还会整体排除该来源；
因此同一文档的目标、重放和后续位置都不能成为自身答案。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef, TypedRef
from pure_integer_ai.cognition.shared.prediction import (
    PredictionEngine,
    PredictionExample,
    PredictionProtocol,
    PredictionResult,
    build_masked_example,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalClock,
    LogicalClockIdentity,
    ScopeIdentity,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceRecord,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _protocol_key(value, *, where: str) -> tuple[int, ...]:
    """校验语言预测 adapter 的调用方注入键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_widths(value, *, where: str) -> tuple[int, ...]:
    """要求目标宽度严格递增，避免配置顺序改变预测事件集合。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{where} 必须是整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{where} 只能包含严格正整数")
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{where} 必须严格递增且唯一")
    return value


@dataclass(frozen=True)
class LanguagePredictionProtocol:
    """注入语言预测目标、单元种类、宽度、预算和逻辑时钟。"""

    prediction: PredictionProtocol
    masked_objective_key: tuple[int, ...]
    next_objective_key: tuple[int, ...]
    condition_namespace_key: tuple[int, ...]
    unit_object_kinds: tuple[int, ...]
    masked_widths: tuple[int, ...]
    next_widths: tuple[int, ...]
    candidate_limit: int
    clock_kind: int

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, PredictionProtocol):
            raise TypeError("prediction 必须是 PredictionProtocol")
        keys = tuple(
            _protocol_key(value, where=f"LanguagePredictionProtocol.{name}")
            for name, value in (
                ("masked_objective_key", self.masked_objective_key),
                ("next_objective_key", self.next_objective_key),
                ("condition_namespace_key", self.condition_namespace_key),
            )
        )
        if len(set(keys)) != len(keys):
            raise ValueError("masked、next 和条件命名空间键必须互不相同")
        if not isinstance(self.unit_object_kinds, tuple):
            raise TypeError("unit_object_kinds 必须是整数 tuple")
        assert_int(
            *self.unit_object_kinds,
            _where="LanguagePredictionProtocol.unit_object_kinds",
        )
        if (not self.unit_object_kinds
                or any(type(item) is not int or item <= 0
                       for item in self.unit_object_kinds)
                or tuple(sorted(set(self.unit_object_kinds)))
                != self.unit_object_kinds):
            raise ValueError("unit_object_kinds 必须严格递增、唯一且为正整数")
        _strict_widths(
            self.masked_widths,
            where="LanguagePredictionProtocol.masked_widths",
        )
        _strict_widths(
            self.next_widths,
            where="LanguagePredictionProtocol.next_widths",
        )
        if not self.masked_widths and not self.next_widths:
            raise ValueError("语言预测至少需要一种 masked 或 next 目标")
        assert_int(
            self.candidate_limit,
            self.clock_kind,
            _where="LanguagePredictionProtocol.budget",
        )
        if (type(self.candidate_limit) is not int
                or self.candidate_limit <= 0):
            raise ValueError("candidate_limit 必须为严格正整数")
        if type(self.clock_kind) is not int or self.clock_kind <= 0:
            raise ValueError("clock_kind 必须为严格正整数")


@dataclass(frozen=True)
class LanguagePredictionEvaluation:
    """一次先预测后揭示目标的结果及其真实目标。"""

    example: PredictionExample
    result: PredictionResult

    @property
    def selected_matches_target(self) -> bool:
        """判断至少一个并列最优候选是否与被遮蔽目标完整相同。"""
        selected = set(self.result.selected_hypotheses)
        return any(
            candidate.hypothesis in selected
            and candidate.target == self.example.target
            for candidate in self.result.candidates
        )


@dataclass(frozen=True)
class LanguagePredictionReport:
    """一个来源文档的单元解析、预测、Evidence 和跳过情况。"""

    source: SourceRef
    scope: ScopeIdentity
    unit_count: int
    unresolved_occurrences: tuple[TypedRef, ...]
    evaluations: tuple[LanguagePredictionEvaluation, ...]

    @property
    def prediction_count(self) -> int:
        """返回已形成的 masked 与 next 预测目标总数。"""
        return len(self.evaluations)

    @property
    def candidate_count(self) -> int:
        """返回所有目标实际形成的预测候选数。"""
        return sum(
            len(item.result.candidates) for item in self.evaluations)

    @property
    def matched_count(self) -> int:
        """返回并列最优候选命中被遮蔽目标的目标数。"""
        return sum(item.selected_matches_target for item in self.evaluations)


class LanguagePredictionRuntime:
    """按文档执行无泄漏预测、揭示 Evidence，再追加训练观察。"""

    def __init__(
            self, ontology: GraphOntology, occurrences: OccurrenceIndex,
            protocol: LanguagePredictionProtocol, *,
            engine: PredictionEngine | None = None,
            ) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(occurrences, OccurrenceIndex):
            raise TypeError("occurrences 必须是 OccurrenceIndex")
        if not isinstance(protocol, LanguagePredictionProtocol):
            raise TypeError("protocol 必须是 LanguagePredictionProtocol")
        self.ontology = ontology
        self.occurrences = occurrences
        self.protocol = protocol
        self.engine = engine or PredictionEngine(protocol.prediction)
        if self.engine.protocol != protocol.prediction:
            raise ValueError("PredictionEngine 与语言预测协议不一致")
        self._reports: dict[tuple, LanguagePredictionReport] = {}

    def observe_document(
            self, occurrence_refs: tuple[TypedRef, ...]
            ) -> LanguagePredictionReport:
        """对一个来源 occurrence 序列先预测后学习；歧义单元使整文档诚实跳过。"""
        if not isinstance(occurrence_refs, tuple) or not occurrence_refs:
            raise ValueError("语言预测需要非空 occurrence tuple")
        if any(not isinstance(ref, TypedRef) for ref in occurrence_refs):
            raise TypeError("occurrence_refs 只能包含 TypedRef")
        records = tuple(self.occurrences.read(ref) for ref in occurrence_refs)
        source, scope = self._document_identity(records)
        cache_key = tuple(ref.stable_key() for ref in occurrence_refs)
        cached = self._reports.get(cache_key)
        if cached is not None:
            return cached

        units: list[ObjectIdentity] = []
        unresolved: list[TypedRef] = []
        for record in records:
            unit = self._unit_identity(record)
            if unit is None:
                unresolved.append(record.occurrence)
            else:
                units.append(unit)
        if unresolved:
            report = LanguagePredictionReport(
                source,
                scope,
                len(units),
                tuple(unresolved),
                (),
            )
            self._reports[cache_key] = report
            return report

        examples = self._examples(tuple(units), source)
        clock = LogicalClock(LogicalClockIdentity(
            scope,
            self.protocol.clock_kind,
        ))
        evaluations: list[LanguagePredictionEvaluation] = []
        for example in examples:
            predicted = self.engine.predict(
                example.context,
                observation=source,
                scope=scope,
                candidate_limit=self.protocol.candidate_limit,
                excluded_sources=(source,),
            )
            timestamp = clock.advance()
            evaluated = self.engine.evaluate(
                predicted,
                expected=example.target,
                evidence_source=source,
                timestamp_seq=timestamp.seq,
            )
            evaluations.append(LanguagePredictionEvaluation(
                example,
                evaluated,
            ))

        # 同一来源全部目标评完后才写模型，防后序位置读取当前文档早先目标。
        for example in examples:
            self.engine.observe(example)
        report = LanguagePredictionReport(
            source,
            scope,
            len(units),
            (),
            tuple(evaluations),
        )
        self._reports[cache_key] = report
        return report

    def report_count(self) -> int:
        """返回已处理且按 occurrence 身份去重的文档数。"""
        return len(self._reports)

    def prediction_count(self) -> int:
        """返回所有文档形成的预测目标总数。"""
        return sum(report.prediction_count for report in self._reports.values())

    def evidence_count(self) -> int:
        """返回所有预测结果实际写入 H-00 的候选 Evidence 数。"""
        return sum(
            report.candidate_count for report in self._reports.values())

    def clone_for_context(
            self, ontology: GraphOntology,
            occurrences: OccurrenceIndex,
            ) -> "LanguagePredictionRuntime":
        """在评测图和 occurrence 索引上复制模型与 ledger，不共享报告缓存。"""
        return LanguagePredictionRuntime(
            ontology,
            occurrences,
            self.protocol,
            engine=self.engine.clone(),
        )

    def state_key(self) -> tuple:
        """返回模型、Evidence 与已处理文档的完整可比较状态。"""
        return (
            self.engine.state_key(),
            tuple(
                (
                    key,
                    report.source.stable_key(),
                    report.unit_count,
                    tuple(ref.stable_key()
                          for ref in report.unresolved_occurrences),
                    report.prediction_count,
                    report.candidate_count,
                    report.matched_count,
                )
                for key, report in sorted(self._reports.items())
            ),
        )

    def _unit_identity(
            self, record: OccurrenceRecord) -> ObjectIdentity | None:
        """读取唯一允许种类的 typed 候选；多义时不按 ordinal 偷选 winner。"""
        allowed = set(self.protocol.unit_object_kinds)
        identities = {
            self.ontology.identity_of(candidate.typed_ref)
            for candidate in record.candidates
            if candidate.typed_ref is not None
            and candidate.typed_ref.object_kind in allowed
        }
        if len(identities) != 1:
            return None
        return next(iter(identities))

    def _examples(
            self, units: tuple[ObjectIdentity, ...],
            source: SourceRef,
            ) -> tuple[PredictionExample, ...]:
        """按注入宽度构造 masked 与 next 目标，不在 adapter 写死 token/span 类别。"""
        examples: list[PredictionExample] = []
        for width in self.protocol.masked_widths:
            for start in range(0, len(units) - width + 1):
                examples.append(self._example(
                    units,
                    source,
                    objective_key=self.protocol.masked_objective_key,
                    start=start,
                    width=width,
                    reveal_suffix=True,
                ))
        for width in self.protocol.next_widths:
            for start in range(1, len(units) - width + 1):
                examples.append(self._example(
                    units,
                    source,
                    objective_key=self.protocol.next_objective_key,
                    start=start,
                    width=width,
                    reveal_suffix=False,
                ))
        return tuple(examples)

    def _example(
            self, units: tuple[ObjectIdentity, ...], source: SourceRef, *,
            objective_key: tuple[int, ...], start: int, width: int,
            reveal_suffix: bool,
            ) -> PredictionExample:
        """构造带来源无关结构回退键的单个预测目标。"""
        namespace = self.protocol.condition_namespace_key
        condition = (
            len(namespace),
            *namespace,
            len(units),
            start,
            width,
            1 if reveal_suffix else 0,
        )
        event_key = (
            len(objective_key),
            *objective_key,
            start,
            width,
            1 if reveal_suffix else 0,
        )
        return build_masked_example(
            units,
            target_start=start,
            target_width=width,
            objective_key=objective_key,
            source=source,
            reveal_suffix=reveal_suffix,
            condition_keys=(condition,),
            event_key=event_key,
        )

    @staticmethod
    def _document_identity(
            records: tuple[OccurrenceRecord, ...]
            ) -> tuple[SourceRef, ScopeIdentity]:
        """核验预测序列只属于同一 SourceRef 和 document scope。"""
        first = records[0]
        if any(
                record.source != first.source or record.scope != first.scope
                for record in records[1:]):
            raise ValueError("语言预测 occurrence 必须属于同一来源和 scope")
        return first.source, first.scope


__all__ = [
    "LanguagePredictionEvaluation",
    "LanguagePredictionProtocol",
    "LanguagePredictionReport",
    "LanguagePredictionRuntime",
]
