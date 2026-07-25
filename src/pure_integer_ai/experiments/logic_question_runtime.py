"""把 R-08 只读执行报告投影到 F-00 QuestionExecutionResult。

本模块不解释 operator、handler 或状态含义。它只接受 R-08 已交叉核验的
``LogicDerivedEvidenceBundle``，并要求 target、SourceRef、query scope 与实际
derivation 完整一致；原子直读或没有 operator adoption 的结果不能冒充有限逻辑。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.experiments.logic_closure_runtime import (
    LogicClosureCourseRuntime,
    LogicClosureRoundReport,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 route 和执行原因是注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _report_key(report: LogicClosureRoundReport) -> tuple[int, ...]:
    """编码只读轮的 scope 和全部 R-08 派生 bundle。"""
    if not isinstance(report, LogicClosureRoundReport):
        raise TypeError("logic question report 类型错误")
    result = [
        *_packed(report.scope.stable_key()),
        int(report.read_only),
        len(report.formations),
    ]
    for hypothesis in report.formations:
        result.extend(_packed(hypothesis.stable_key()))
    result.append(len(report.recognitions))
    for recognition in report.recognitions:
        result.extend(_packed(recognition.stable_key()))
    result.append(len(report.executions))
    for execution in report.executions:
        result.extend(_packed(execution.stable_key()))
    return tuple(result)


class LogicClosureQuestionExecutor:
    """只读执行一个 R-08 course round，并投影当前问题的有限逻辑结果。"""

    def __init__(
            self,
            runtime: LogicClosureCourseRuntime,
            *,
            route: ObjectIdentity,
            executed_reason: ObjectIdentity,
            ) -> None:
        """绑定既有 R-08 production owner、F-00 route 和执行原因。"""
        if not isinstance(runtime, LogicClosureCourseRuntime):
            raise TypeError("logic question runtime 类型错误")
        self.runtime = runtime
        self.route = _instruction(route, label="logic question route")
        self.executed_reason = _instruction(
            executed_reason, label="logic question executed reason")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """执行只读逻辑轮，只采用同一 target/source/scope 且有 derivation 的候选。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("logic question 需要 QuestionQuery")
        if query.route != self.route:
            raise ValueError("logic question 收到未注册 route")
        request = query.request
        before = self.runtime.state_key()
        report = self.runtime.process(
            request.response_scope,
            read_only=True,
        )
        if not isinstance(report, LogicClosureRoundReport):
            raise TypeError("logic question runtime 返回类型错误")
        if (not report.read_only
                or report.scope != request.response_scope
                or report.formations
                or report.recognitions):
            raise ValueError("logic question 必须返回同 scope 只读执行报告")
        if self.runtime.state_key() != before:
            raise RuntimeError("logic question 只读执行改变了 R-08 owner 状态")

        candidates = []
        matched = []
        for bundle in report.executions:
            execution = bundle.execution
            evaluation = execution.evaluation
            if (bundle.candidate.proposition != request.target
                    or bundle.candidate.source != request.source
                    or bundle.candidate.scope != request.response_scope):
                continue
            if not evaluation.derivation or not execution.adoptions:
                continue
            candidates.append(bundle.candidate)
            matched.append(bundle)
        keys = tuple(item.stable_key() for item in candidates)
        if len(set(keys)) != len(keys):
            raise ValueError("logic question 不得重复投影同一派生候选")
        trace = (
            1,
            *_packed(query.stable_key()),
            *_packed(_report_key(report)),
            len(matched),
            *(value for item in matched
              for value in _packed(item.stable_key())),
        )
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            tuple(candidates),
            trace,
        )


__all__ = ["LogicClosureQuestionExecutor"]
