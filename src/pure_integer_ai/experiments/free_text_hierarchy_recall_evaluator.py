"""P3-Ia runtime 完成后读取私有标签的独立、零写 evaluator。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.free_text_hierarchy_runtime import (
    FormedTextHierarchy,
)
from pure_integer_ai.experiments.free_text_recall_runtime import (
    FreeTextRecallRun,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_authored_free_text_hierarchy_recall_course import (
    EVALUATOR_DIMENSIONS,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


class FreeTextHierarchyRecallEvaluatorError(RuntimeError):
    """runtime 报告、私有标签或分维裁决不闭合。"""


def _keys(value: Any, *, where: str) -> tuple[StableRecordKey, ...]:
    """从私有标签读取稳定排序、去重的正整数键数组。"""
    if not isinstance(value, list):
        raise FreeTextHierarchyRecallEvaluatorError(f"{where} 必须是数组")
    try:
        result = tuple(StableRecordKey(tuple(item)) for item in value)
    except (TypeError, ValueError) as error:
        raise FreeTextHierarchyRecallEvaluatorError(f"{where} key 非法") from error
    if result != tuple(sorted(set(result))):
        raise FreeTextHierarchyRecallEvaluatorError(f"{where} 必须排序去重")
    return result


@dataclass(frozen=True)
class FreeTextProductionReport:
    """候选链执行完毕后交给 evaluator 的只读投影。"""

    hierarchy: FormedTextHierarchy | None
    recall: FreeTextRecallRun
    question: QuestionAnswerRun | None
    invalidated_keys: tuple[StableRecordKey, ...]
    preserved_keys: tuple[StableRecordKey, ...]
    runtime_complete: int
    runtime_private_label_read_count: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        """核验报告只含运行产物，且 runtime 从未读取 evaluator 私有标签。"""
        if self.hierarchy is not None and not isinstance(
                self.hierarchy, FormedTextHierarchy):
            raise TypeError("production report hierarchy 类型错误")
        if not isinstance(self.recall, FreeTextRecallRun):
            raise TypeError("production report recall 类型错误")
        if self.question is not None and not isinstance(
                self.question, QuestionAnswerRun):
            raise TypeError("production report question 类型错误")
        for name in ("invalidated_keys", "preserved_keys"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, StableRecordKey) for item in values)
                    or values != tuple(sorted(set(values)))):
                raise FreeTextHierarchyRecallEvaluatorError(
                    f"production report {name} 非法")
        if self.runtime_complete != 1:
            raise FreeTextHierarchyRecallEvaluatorError("runtime 报告尚未冻结完成")
        if (self.runtime_private_label_read_count != 0
                or self.host_learning_write_count != 0):
            raise FreeTextHierarchyRecallEvaluatorError(
                "runtime 不得读私有标签或产生 host learning write")


@dataclass(frozen=True, order=True)
class FreeTextDimensionResult:
    """一个冻结 evaluator 维度的二分结果与实际计数。"""

    dimension: str
    passed: int
    observed_count: int

    def __post_init__(self) -> None:
        """要求维度已登记且结果/计数均为严格整数。"""
        if self.dimension not in EVALUATOR_DIMENSIONS:
            raise FreeTextHierarchyRecallEvaluatorError("evaluator dimension 未登记")
        if self.passed not in {0, 1}:
            raise FreeTextHierarchyRecallEvaluatorError("dimension passed 非二分")
        if type(self.observed_count) is not int or self.observed_count < 0:
            raise FreeTextHierarchyRecallEvaluatorError("observed count 非法")


@dataclass(frozen=True)
class FreeTextEvaluationReport:
    """全部七维裁决和 evaluator 自身零写事实。"""

    results: tuple[FreeTextDimensionResult, ...]
    evaluator_label_read_count: int
    evaluator_host_write_count: int

    def __post_init__(self) -> None:
        """核验维度列全、顺序稳定且 evaluator 没有宿主写。"""
        if (not isinstance(self.results, tuple)
                or tuple(item.dimension for item in self.results)
                != EVALUATOR_DIMENSIONS):
            raise FreeTextHierarchyRecallEvaluatorError("evaluator dimensions 未列全")
        if self.evaluator_label_read_count != 1:
            raise FreeTextHierarchyRecallEvaluatorError("evaluator 必须且只能读一次标签")
        if self.evaluator_host_write_count != 0:
            raise FreeTextHierarchyRecallEvaluatorError("evaluator 不得写宿主状态")

    @property
    def passed(self) -> bool:
        """仅当七维全部通过时返回 True。"""
        return all(item.passed == 1 for item in self.results)

    def result(self, dimension: str) -> FreeTextDimensionResult:
        """按冻结名称返回唯一分维结果。"""
        matches = tuple(item for item in self.results
                        if item.dimension == dimension)
        if len(matches) != 1:
            raise FreeTextHierarchyRecallEvaluatorError("dimension result 不唯一")
        return matches[0]


class IndependentFreeTextHierarchyRecallEvaluator:
    """只在 runtime 完成后比较私有边界、center、citation、QA 和局部修正。"""

    @staticmethod
    def _expected(value: CanonicalJsonObject) -> dict[str, Any]:
        """读取私有 expected payload 的精确字段集合。"""
        if not isinstance(value, CanonicalJsonObject):
            raise TypeError("evaluator expected payload 类型错误")
        raw = value.to_value()
        expected = {
            "answer_surface", "center_record_keys", "citation",
            "hierarchy_ranges", "invalidated_keys", "preserved_keys",
            "proposition_record_key", "required_stop_reason",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise FreeTextHierarchyRecallEvaluatorError(
                "expected payload 字段集合非法")
        return raw

    def evaluate(
            self,
            runtime: FreeTextProductionReport,
            expected_payload: CanonicalJsonObject,
            ) -> FreeTextEvaluationReport:
        """在 execution 完成后进行一次只读私有标签比较并返回七维结果。"""
        if not isinstance(runtime, FreeTextProductionReport):
            raise TypeError("evaluator runtime report 类型错误")
        expected = self._expected(expected_payload)
        expected_ranges_value = expected["hierarchy_ranges"]
        if (not isinstance(expected_ranges_value, list)
                or any(not isinstance(item, dict)
                       or set(item) != {"end", "start"}
                       or type(item["start"]) is not int
                       or type(item["end"]) is not int
                       or item["start"] < 0
                       or item["start"] >= item["end"]
                       for item in expected_ranges_value)):
            raise FreeTextHierarchyRecallEvaluatorError("expected hierarchy ranges 非法")
        expected_ranges = tuple(
            (item["start"], item["end"]) for item in expected_ranges_value)
        actual_ranges = ()
        if runtime.hierarchy is not None:
            actual_ranges = tuple(
                (candidate.span.start, candidate.span.end)
                for candidate in runtime.hierarchy.candidates
                if candidate.candidate_kind == "PARAGRAPH"
            )
        expected_centers = _keys(
            expected["center_record_keys"], where="expected centers")
        actual_centers = tuple(sorted(StableRecordKey(
            item.index_entry.record_key) for item in runtime.recall.centers))
        citation = expected["citation"]
        if not isinstance(citation, dict) or set(citation) != {
                "end", "record_key", "source_ref_key", "start"}:
            raise FreeTextHierarchyRecallEvaluatorError("expected citation 字段非法")
        exact = runtime.recall.exact_read
        actual_citation = None
        if exact is not None and exact.receipt.citations:
            item = exact.receipt.citations[0]
            actual_citation = (
                item.record_key,
                item.source_ref.stable_key(),
                item.span.start,
                item.span.end,
            )
        expected_citation = (
            StableRecordKey(tuple(citation["record_key"])),
            tuple(citation["source_ref_key"]),
            citation["start"],
            citation["end"],
        )
        should_resolve = expected["required_stop_reason"] == "RESOLVED"
        citation_pass = (
            actual_citation == expected_citation
            and exact is not None
            and exact.metrics.page_faults >= 1
            and exact.metrics.page_in_records >= 1
        ) if should_resolve else actual_citation is None
        proposition_key = StableRecordKey(tuple(expected["proposition_record_key"]))
        result_keys = () if exact is None else exact.receipt.result_keys
        recall_pass = (
            runtime.recall.stop_reason == expected["required_stop_reason"]
            and (not should_resolve or proposition_key in result_keys)
        )
        if should_resolve:
            qa_pass = (
                runtime.question is not None
                and runtime.question.complete
                and runtime.question.query_result is not None
                and len(runtime.question.query_result.candidates) == 1
            )
        else:
            qa_pass = runtime.question is None
        expected_invalidated = _keys(
            expected["invalidated_keys"], where="expected invalidated")
        expected_preserved = _keys(
            expected["preserved_keys"], where="expected preserved")
        values = {
            "CENTER_FORMATION": (
                actual_centers == expected_centers, len(actual_centers)),
            "CITATION_EXACTNESS": (
                citation_pass, 0 if actual_citation is None else 1),
            "HIERARCHY_FORMATION": (
                actual_ranges == expected_ranges, len(actual_ranges)),
            "LABEL_ISOLATION": (
                runtime.runtime_private_label_read_count == 0
                and runtime.recall.private_label_read_count == 0,
                runtime.runtime_private_label_read_count
                + runtime.recall.private_label_read_count,
            ),
            "QA_CONSUMER": (qa_pass, 0 if runtime.question is None else 1),
            "RECALL_SELECTION": (recall_pass, len(result_keys)),
            "REVISION_LOCALITY": (
                runtime.invalidated_keys == expected_invalidated
                and runtime.preserved_keys == expected_preserved
                and not set(runtime.invalidated_keys).intersection(
                    runtime.preserved_keys),
                len(runtime.invalidated_keys),
            ),
        }
        results = tuple(FreeTextDimensionResult(
            dimension,
            int(values[dimension][0]),
            values[dimension][1],
        ) for dimension in EVALUATOR_DIMENSIONS)
        return FreeTextEvaluationReport(results, 1, 0)


__all__ = [
    "FreeTextDimensionResult",
    "FreeTextEvaluationReport",
    "FreeTextHierarchyRecallEvaluatorError",
    "FreeTextProductionReport",
    "IndependentFreeTextHierarchyRecallEvaluator",
]
