"""独立的已训练表层组织 held-out 评估。

该评估只检验一个窄而真实的能力：公开 DLG-RAW-16 课程学到的
``literal gap + typed slots`` 是否能在新实体、新限定和新组合上生成完整、
可读的句子。它不把来源检索、事实正确性或原始文本理解混入分数。

每条 case 同时保存一个没有接入训练消费者的显式基线，以及真实
``TrainedSurfaceRuntime.render_typed`` 的结果。基线不是伪造的旧答案，而是
``NO_LEARNED_SURFACE``；因此报告中的 PASS 只表示训练后的表层结构成功消费
了新的 typed 输入，不表示通用问答、自由生成或断奶已经完成。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceSemantic,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
)


HELDOUT_SURFACE_PROTOCOL_V1 = 1
HELDOUT_PASS = "PASS"
HELDOUT_FAIL = "FAIL"
HELDOUT_NE = "NE"
BASELINE_NO_LEARNED_SURFACE = "NO_LEARNED_SURFACE"
_TRACE_DOMAIN = "pure_integer_ai.dialogue.surface.heldout.v1"


class HeldOutSurfaceEvaluationError(ValueError):
    """held-out case、独立性或报告合同无效。"""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise HeldOutSurfaceEvaluationError(
            f"{where} 必须是无首尾空白的非空字符串")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise HeldOutSurfaceEvaluationError(f"{where} 含非 Unicode scalar")
    return value


def _positive(value: Any, where: str) -> None:
    if type(value) is not int or value <= 0:
        raise HeldOutSurfaceEvaluationError(f"{where} 必须是正整数")


def _pack_text(value: str, where: str) -> tuple[int, ...]:
    value = _text(value, where)
    scalars = tuple(ord(item) for item in value)
    return len(scalars), *scalars


def _pack_texts(values: tuple[str, ...], where: str) -> tuple[int, ...]:
    result = [len(values)]
    for index, value in enumerate(values):
        result.extend(_pack_text(value, f"{where}[{index}]") )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class HeldOutSurfaceCase:
    """不含 evaluator label 的 typed 输入和独立 expected surface。"""

    case_id: str
    response_act: str
    register: str
    ordered_roles: tuple[str, ...]
    semantic: SurfaceSemantic
    slot_values: tuple[str, ...]
    expected_surface: str
    source_id: str
    context_id: str
    family_id: str
    min_chars: int = 1
    max_chars: int = 4096

    def __post_init__(self) -> None:
        _text(self.case_id, "case.case_id")
        _text(self.response_act, "case.response_act")
        _text(self.register, "case.register")
        if (not isinstance(self.ordered_roles, tuple)
                or not self.ordered_roles
                or any(not isinstance(item, str) or not item
                       for item in self.ordered_roles)):
            raise HeldOutSurfaceEvaluationError("case.ordered_roles 非法")
        if not isinstance(self.semantic, SurfaceSemantic):
            raise TypeError("case.semantic 类型错误")
        if (not isinstance(self.slot_values, tuple)
                or len(self.slot_values) != len(self.ordered_roles)):
            raise HeldOutSurfaceEvaluationError(
                "case.slot_values 必须与 ordered_roles 一一对应")
        for index, value in enumerate(self.slot_values):
            _text(value, f"case.slot_values[{index}]")
        _text(self.expected_surface, "case.expected_surface")
        for name in ("source_id", "context_id", "family_id"):
            _text(getattr(self, name), f"case.{name}")
        _positive(self.min_chars, "case.min_chars")
        _positive(self.max_chars, "case.max_chars")
        if self.max_chars < self.min_chars:
            raise HeldOutSurfaceEvaluationError("case 字符预算倒置")

    @property
    def is_long(self) -> bool:
        return len(self.expected_surface.encode("utf-8")) >= 48

    def canonical_record(self) -> tuple[int, ...]:
        result = [HELDOUT_SURFACE_PROTOCOL_V1]
        for value in (self.case_id, self.response_act, self.register):
            result.extend(_pack_text(value, "case.identity"))
        result.extend(_pack_texts(self.ordered_roles, "case.roles"))
        result.extend(_pack_texts(self.slot_values, "case.slot_values"))
        result.extend(_pack_text(self.expected_surface, "case.expected_surface"))
        for value in (self.source_id, self.context_id, self.family_id):
            result.extend(_pack_text(value, "case.binding"))
        result.extend((self.min_chars, self.max_chars))
        result.extend(self.semantic.canonical_record())
        return tuple(result)


@dataclass(frozen=True, slots=True)
class HeldOutSurfaceObservation:
    """一条 case 的基线、训练结果和整数 trace。"""

    case_id: str
    response_act: str
    is_long: int
    baseline_status: str
    trained_status: str
    trained_used: int
    pattern_id: int
    reason: str
    generated_surface: str
    expected_surface: str
    result_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _text(self.case_id, "observation.case_id")
        _text(self.response_act, "observation.response_act")
        if self.is_long not in (0, 1) or self.trained_used not in (0, 1):
            raise HeldOutSurfaceEvaluationError("observation bit 字段非法")
        _text(self.baseline_status, "observation.baseline_status")
        _text(self.trained_status, "observation.trained_status")
        _text(self.reason, "observation.reason")
        if type(self.pattern_id) is not int or self.pattern_id < 0:
            raise HeldOutSurfaceEvaluationError("observation.pattern_id 非法")
        _text(self.expected_surface, "observation.expected_surface")
        if self.generated_surface:
            _text(self.generated_surface, "observation.generated_surface")
        if (not isinstance(self.result_trace, tuple) or not self.result_trace
                or any(type(item) is not int or item < 0
                       for item in self.result_trace)):
            raise HeldOutSurfaceEvaluationError("observation.result_trace 非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_status": self.baseline_status,
            "case_id": self.case_id,
            "expected_surface": self.expected_surface,
            "generated_surface": self.generated_surface,
            "is_long": bool(self.is_long),
            "pattern_id": self.pattern_id,
            "reason": self.reason,
            "response_act": self.response_act,
            "result_trace_u": list(self.result_trace),
            "trained_status": self.trained_status,
            "trained_used": bool(self.trained_used),
        }

    def canonical_record(self) -> tuple[int, ...]:
        result = [HELDOUT_SURFACE_PROTOCOL_V1, self.is_long,
                  self.trained_used, self.pattern_id]
        for value in (self.case_id, self.response_act, self.baseline_status,
                      self.trained_status, self.reason, self.generated_surface,
                      self.expected_surface):
            result.extend(_pack_text(value, "observation.text")
                          if value else (0,))
        result.extend(self.result_trace)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class HeldOutSurfaceReport:
    """开发性报告；不会改变 formal_train 或 weaning gate。"""

    status: str
    run_id: str
    pack_sha256: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    long_cases: int
    baseline_no_consumer_cases: int
    observations: tuple[HeldOutSurfaceObservation, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in {HELDOUT_PASS, HELDOUT_FAIL, HELDOUT_NE}:
            raise HeldOutSurfaceEvaluationError("report.status 未注册")
        _text(self.run_id, "report.run_id")
        _text(self.pack_sha256, "report.pack_sha256")
        for name in ("total_cases", "passed_cases", "failed_cases",
                     "long_cases", "baseline_no_consumer_cases"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise HeldOutSurfaceEvaluationError(f"report.{name} 非法")
        if (not isinstance(self.observations, tuple)
                or len(self.observations) != self.total_cases
                or len({item.case_id for item in self.observations})
                != self.total_cases):
            raise HeldOutSurfaceEvaluationError("report observations 不闭合")
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise HeldOutSurfaceEvaluationError("report pass/fail 分母漂移")
        if self.long_cases != sum(item.is_long for item in self.observations):
            raise HeldOutSurfaceEvaluationError("report long_cases 漂移")
        if self.baseline_no_consumer_cases != sum(
                item.baseline_status == BASELINE_NO_LEARNED_SURFACE
                for item in self.observations):
            raise HeldOutSurfaceEvaluationError("report baseline 计数漂移")
        if not isinstance(self.trace, tuple) or not self.trace:
            raise HeldOutSurfaceEvaluationError("report.trace 非法")
        if self.status == HELDOUT_PASS and (
                self.total_cases == 0 or self.failed_cases != 0):
            raise HeldOutSurfaceEvaluationError("PASS 报告计数不闭合")

    @property
    def ready(self) -> int:
        """开发评估不改变断奶状态，故永远为 0。"""
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_no_consumer_cases": self.baseline_no_consumer_cases,
            "failed_cases": self.failed_cases,
            "format_version": HELDOUT_SURFACE_PROTOCOL_V1,
            "long_cases": self.long_cases,
            "observations": [item.to_dict() for item in self.observations],
            "pack_sha256": self.pack_sha256,
            "passed_cases": self.passed_cases,
            "ready": False,
            "run_id": self.run_id,
            "status": self.status,
            "total_cases": self.total_cases,
            "trace_u": list(self.trace),
        }

    def canonical_record(self) -> tuple[int, ...]:
        result = [HELDOUT_SURFACE_PROTOCOL_V1, self.total_cases,
                  self.passed_cases, self.failed_cases, self.long_cases,
                  self.baseline_no_consumer_cases]
        for value in (self.status, self.run_id, self.pack_sha256):
            result.extend(_pack_text(value, "report.identity"))
        for item in self.observations:
            record = item.canonical_record()
            result.extend((len(record), *record))
        result.extend(self.trace)
        return tuple(result)


def build_public_heldout_cases() -> tuple[HeldOutSurfaceCase, ...]:
    """返回固定的新实体/限定/组合集合。

    case 的 source/context/family 均使用本评估的独立命名空间；这些值不会
    被课程加载器或训练图写回。
    """
    return (
        HeldOutSurfaceCase(
            "surface-heldout-causal-01", "ANSWER", "neutral",
            ("cause", "relation", "effect"),
            SurfaceSemantic("heldout-causal-01", "causal",
                            "霁川观测舱连续三日记录潮汐异常",
                            "导致", "玄铜环线在夜间临时限速"),
            ("霁川观测舱连续三日记录潮汐异常", "导致",
             "玄铜环线在夜间临时限速"),
            "霁川观测舱连续三日记录潮汐异常导致玄铜环线在夜间临时限速。",
            "heldout-surface-v1-source-01", "heldout-surface-v1-context-01",
            "heldout-surface-v1-family-01"),
        HeldOutSurfaceCase(
            "surface-heldout-causal-02", "ANSWER", "neutral",
            ("cause", "relation", "effect"),
            SurfaceSemantic("heldout-causal-02", "causal",
                            "北岸储能站在寒潮期间连续启动备用机组",
                            "使得", "沿线社区的夜间供电保持稳定"),
            ("北岸储能站在寒潮期间连续启动备用机组", "使得",
             "沿线社区的夜间供电保持稳定"),
            "北岸储能站在寒潮期间连续启动备用机组使得沿线社区的夜间供电保持稳定。",
            "heldout-surface-v1-source-02", "heldout-surface-v1-context-02",
            "heldout-surface-v1-family-02"),
        HeldOutSurfaceCase(
            "surface-heldout-qualified-01", "ANSWER", "polite",
            ("subject", "predicate", "qualifier", "object"),
            SurfaceSemantic("heldout-qualified-01", "fact",
                            "霜谷数据中继站", "开放时间", "每周三至周五"),
            ("霜谷数据中继站", "开放时间", "维护窗口结束后",
             "每周三至周五"),
            "霜谷数据中继站的开放时间（维护窗口结束后）为每周三至周五。",
            "heldout-surface-v1-source-03", "heldout-surface-v1-context-03",
            "heldout-surface-v1-family-03"),
        HeldOutSurfaceCase(
            "surface-heldout-qualified-02", "ANSWER", "polite",
            ("subject", "predicate", "qualifier", "object"),
            SurfaceSemantic("heldout-qualified-02", "fact",
                            "青岚实验码头", "预约截止时间", "每月最后一个工作日"),
            ("青岚实验码头", "预约截止时间", "仅限下一季度航次",
             "每月最后一个工作日"),
            "青岚实验码头的预约截止时间（仅限下一季度航次）为每月最后一个工作日。",
            "heldout-surface-v1-source-04", "heldout-surface-v1-context-04",
            "heldout-surface-v1-family-04"),
        HeldOutSurfaceCase(
            "surface-heldout-unknown-01", "UNKNOWN", "neutral",
            ("source", "scope"),
            SurfaceSemantic("heldout-unknown-01", "unknown",
                            "unknown-subject", "unknown-predicate", "unknown-object"),
            ("当前公开档案", "霁川观测舱的维护预算与责任单位"),
            "当前公开档案资料没有提供霁川观测舱的维护预算与责任单位。",
            "heldout-surface-v1-source-05", "heldout-surface-v1-context-05",
            "heldout-surface-v1-family-05"),
        HeldOutSurfaceCase(
            "surface-heldout-unknown-02", "UNKNOWN", "neutral",
            ("source", "scope"),
            SurfaceSemantic("heldout-unknown-02", "unknown",
                            "unknown-subject", "unknown-predicate", "unknown-object"),
            ("已核对档案", "玄铜环线在极端天气下的备用调度规则"),
            "已核对档案资料没有提供玄铜环线在极端天气下的备用调度规则。",
            "heldout-surface-v1-source-06", "heldout-surface-v1-context-06",
            "heldout-surface-v1-family-06"),
        HeldOutSurfaceCase(
            "surface-heldout-clarify-01", "CLARIFY", "polite",
            ("choice", "target"),
            SurfaceSemantic("heldout-clarify-01", "scope",
                            "unknown-subject", "unknown-predicate", "unknown-object"),
            ("东侧蓄水区还是西侧蓄水区", "月度入流量"),
            "请先选择东侧蓄水区还是西侧蓄水区，再说明要查询的月度入流量。",
            "heldout-surface-v1-source-07", "heldout-surface-v1-context-07",
            "heldout-surface-v1-family-07"),
        HeldOutSurfaceCase(
            "surface-heldout-clarify-02", "CLARIFY", "polite",
            ("choice", "target"),
            SurfaceSemantic("heldout-clarify-02", "scope",
                            "unknown-subject", "unknown-predicate", "unknown-object"),
            ("工作日还是节假日", "夜间开放时段"),
            "请先选择工作日还是节假日，再说明要查询的夜间开放时段。",
            "heldout-surface-v1-source-08", "heldout-surface-v1-context-08",
            "heldout-surface-v1-family-08"),
        HeldOutSurfaceCase(
            "surface-heldout-repair-01", "REPAIR", "polite",
            ("acknowledge", "request"),
            SurfaceSemantic("heldout-repair-01", "repair",
                            "先前问题", "需要", "补充限定"),
            ("前面的条件没有说明统计口径", "补充按日还是按月计算"),
            "前面的条件没有说明统计口径，请说明补充按日还是按月计算。",
            "heldout-surface-v1-source-09", "heldout-surface-v1-context-09",
            "heldout-surface-v1-family-09"),
        HeldOutSurfaceCase(
            "surface-heldout-repair-02", "REPAIR", "polite",
            ("acknowledge", "request"),
            SurfaceSemantic("heldout-repair-02", "repair",
                            "先前问题", "需要", "补充限定"),
            ("前面的条件没有说明时间范围", "补充起止日期和时区"),
            "前面的条件没有说明时间范围，请说明补充起止日期和时区。",
            "heldout-surface-v1-source-10", "heldout-surface-v1-context-10",
            "heldout-surface-v1-family-10"),
    )


def _course_surface_texts(project_root: str | Path) -> tuple[str, ...]:
    path = Path(project_root).resolve() / "data" / "ph2"
    course = path / "dlg_raw16_surface_organization_v1.jsonl.sample"
    if not course.is_file():
        raise HeldOutSurfaceEvaluationError("公开表层课程缺失")
    try:
        lines = course.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise HeldOutSurfaceEvaluationError("公开表层课程不可回读") from error
    return tuple(lines)


def validate_cases_independent(project_root: str | Path,
                               cases: Iterable[HeldOutSurfaceCase]) -> tuple[HeldOutSurfaceCase, ...]:
    """拒绝把训练课程的完整 surface 或身份重新当成 held-out。"""
    values = tuple(cases)
    if not values:
        return ()
    if len({item.case_id for item in values}) != len(values):
        raise HeldOutSurfaceEvaluationError("held-out case_id 必须唯一")
    course_lines = _course_surface_texts(project_root)
    course_text = "\n".join(course_lines)
    for case in values:
        if case.expected_surface in course_text:
            raise HeldOutSurfaceEvaluationError(
                f"{case.case_id} expected surface 泄漏到课程")
        for value in (case.source_id, case.context_id, case.family_id):
            if value in course_text:
                raise HeldOutSurfaceEvaluationError(
                    f"{case.case_id} identity 泄漏到课程: {value}")
        if any(value in course_text for value in case.slot_values
               if len(value) >= 4):
            raise HeldOutSurfaceEvaluationError(
                f"{case.case_id} slot value 泄漏到课程")
    return values


def run_trained_surface_heldout(
        runtime: TrainedSurfaceRuntime,
        project_root: str | Path,
        cases: Iterable[HeldOutSurfaceCase] | None = None,
        ) -> HeldOutSurfaceReport:
    """运行独立 held-out 评估，不写训练状态。"""
    if not isinstance(runtime, TrainedSurfaceRuntime):
        raise TypeError("runtime 类型错误")
    selected = validate_cases_independent(
        project_root, build_public_heldout_cases() if cases is None else cases)
    if not selected:
        trace = integer_tuple_fingerprint(
            (HELDOUT_SURFACE_PROTOCOL_V1, 0), domain=_TRACE_DOMAIN)
        return HeldOutSurfaceReport(
            HELDOUT_NE, runtime.observation.run_id,
            runtime.observation.pack_sha256, 0, 0, 0, 0, 0, (), trace)
    observations: list[HeldOutSurfaceObservation] = []
    for case in selected:
        # Baseline is intentionally a value, not an old answer. It models the
        # same typed input with no learned surface consumer connected.
        baseline_status = BASELINE_NO_LEARNED_SURFACE
        rendered = runtime.render_typed(
            case.semantic,
            response_act=case.response_act,
            register=case.register,
            ordered_roles=case.ordered_roles,
            slot_values=case.slot_values,
            source_id=case.source_id,
            context_id=case.context_id,
            family_id=case.family_id,
            ordinal=0,
        )
        trained_used = int(rendered.used)
        exact = int(rendered.used and rendered.surface == case.expected_surface)
        trained_status = HELDOUT_PASS if exact else HELDOUT_FAIL
        result_trace = rendered.trace
        if not result_trace:
            result_trace = ((int(rendered.used), rendered.pattern_id)
                            + (tuple(rendered.surface.encode("utf-8"))
                               if rendered.surface else (0,)))
        observations.append(HeldOutSurfaceObservation(
            case.case_id, case.response_act, int(case.is_long),
            baseline_status, trained_status, trained_used,
            rendered.pattern_id, rendered.reason, rendered.surface,
            case.expected_surface, tuple(result_trace),
        ))
    passed = sum(item.trained_status == HELDOUT_PASS for item in observations)
    failed = len(observations) - passed
    status = HELDOUT_PASS if failed == 0 else HELDOUT_FAIL
    trace_values = [HELDOUT_SURFACE_PROTOCOL_V1, len(observations), passed, failed]
    for item in observations:
        trace_values.extend(item.canonical_record())
    trace = integer_tuple_fingerprint(tuple(trace_values), domain=_TRACE_DOMAIN)
    return HeldOutSurfaceReport(
        status, runtime.observation.run_id, runtime.observation.pack_sha256,
        len(observations), passed, failed,
        sum(item.is_long for item in observations),
        sum(item.baseline_status == BASELINE_NO_LEARNED_SURFACE
            for item in observations),
        tuple(observations), trace,
    )


def write_heldout_surface_report(report: HeldOutSurfaceReport,
                                 output_path: str | Path) -> str:
    """只创建 K 盘开发摘要，不覆盖既有 artifact。"""
    if not isinstance(report, HeldOutSurfaceReport):
        raise TypeError("report 类型错误")
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("held-out report 必须写入不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")) + "\n"
    output.write_text(payload, encoding="utf-8")
    return str(output)


__all__ = [
    "BASELINE_NO_LEARNED_SURFACE", "HELDOUT_FAIL", "HELDOUT_NE",
    "HELDOUT_PASS", "HELDOUT_SURFACE_PROTOCOL_V1", "HeldOutSurfaceCase",
    "HeldOutSurfaceEvaluationError", "HeldOutSurfaceObservation",
    "HeldOutSurfaceReport", "build_public_heldout_cases",
    "run_trained_surface_heldout", "validate_cases_independent",
    "write_heldout_surface_report",
]
