"""把 R-05 typed 对称 pair 查询投影到 F-00 QuestionExecutionResult。

adapter 只调用 R-05 的只读 ``select_many``，不写 Relation Use。只有与问题目标
Proposition、原始方向、SourceRef 和 Evidence scope 全部一致的直接 Evidence 才能
形成 G-00 候选；只有反向事实或规则前提而缺少来源化派生 Evidence 时保持 unknown。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairEvaluation,
    SymmetricPairPattern,
    SymmetricPairSelection,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.symmetric_relation_runtime import (
    SymmetricPairQuery,
    SymmetricRelationChannelRuntime,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 route 和原因是调用方注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _role_filler(
        proposition: BoundProposition,
        role: ObjectIdentity,
        ) -> ObjectIdentity:
    """按完整 Role 身份读取唯一非嵌套 filler，拒绝缺失或竞争绑定。"""
    matches = tuple(
        item.filler
        for item in proposition.bindings
        if item.role == role
    )
    if len(matches) != 1:
        raise ValueError("symmetric question Role 必须恰有一个 filler")
    filler = matches[0]
    if not isinstance(filler, ObjectIdentity):
        raise ValueError("symmetric question endpoint 不接受嵌套 Proposition")
    return filler


def _evaluation_key(evaluation: SymmetricPairEvaluation) -> tuple[int, ...]:
    """编码 R-05 pair 四态、全部直接 Evidence 和规则反驳。"""
    if not isinstance(evaluation, SymmetricPairEvaluation):
        raise TypeError("symmetric question evaluation 类型错误")
    result = [
        *_packed(evaluation.pair.stable_key()),
        *evaluation.state.stable_key(),
        len(evaluation.evidence),
    ]
    for evidence in evaluation.evidence:
        result.extend(_packed(evidence.stable_key()))
    result.append(len(evaluation.rule_refutes))
    for refute in evaluation.rule_refutes:
        result.extend(_packed(refute.stable_key()))
    return tuple(result)


def _selection_key(selection: SymmetricPairSelection) -> tuple[int, ...]:
    """编码精确 R-05 查询模式和稳定有序的全部 evaluation。"""
    if not isinstance(selection, SymmetricPairSelection):
        raise TypeError("symmetric question selection 类型错误")
    result = [
        *_packed(selection.pattern.stable_key()),
        len(selection.evaluations),
    ]
    for evaluation in selection.evaluations:
        result.extend(_packed(_evaluation_key(evaluation)))
    return tuple(result)


class SymmetricPairQuestionExecutor:
    """执行一个精确 R-05 pair 查询并形成来源守恒的 G-00 候选。"""

    def __init__(
            self,
            owner: SymmetricRelationChannelRuntime,
            *,
            route: ObjectIdentity,
            executed_reason: ObjectIdentity,
            ) -> None:
        """绑定单个关系 channel、F-00 route 和来源化执行原因。"""
        if not isinstance(owner, SymmetricRelationChannelRuntime):
            raise TypeError("symmetric question owner 类型错误")
        self.owner = owner
        self.route = _instruction(route, label="symmetric question route")
        self.executed_reason = _instruction(
            executed_reason, label="symmetric question executed reason")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """从目标 Role 绑定构造精确 pair，只读查询并投影同源直接 Evidence。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("symmetric question 需要 QuestionQuery")
        if query.route != self.route:
            raise ValueError("symmetric question 收到未注册 route")
        request = query.request
        target = request.target
        protocol = self.owner.protocol
        if (target.predicate != protocol.relation
                or target.structure != protocol.schema.schema):
            raise ValueError("symmetric question target 与关系 schema 不一致")
        left = _role_filler(target, protocol.left_role)
        right = _role_filler(target, protocol.right_role)
        native_query = SymmetricPairQuery(SymmetricPairPattern(left, right))
        selection = self.owner.select_many((native_query,))[0]
        if selection.pattern != native_query.pattern:
            raise ValueError("symmetric question owner 替换了查询 pattern")
        if len(selection.evaluations) != 1:
            raise ValueError("symmetric exact question 必须返回唯一 pair evaluation")
        evaluation = selection.evaluations[0]

        direct = tuple(
            item for item in evaluation.evidence
            if (item.proposition == target.template
                and item.left == left
                and item.right == right
                and item.scope == request.evidence_scope)
        )
        records_by_id = {
            record.evidence_id: record
            for item in direct
            for record in item.evidence
        }
        records = tuple(
            records_by_id[key] for key in sorted(records_by_id)
        )
        candidates = ()
        if records:
            state = LogicEvidenceState(
                any(item.stance == EVIDENCE_SUPPORT for item in records),
                any(item.stance == EVIDENCE_REFUTE for item in records),
            )
            candidates = (GenerationCandidate(
                target,
                state,
                request.source,
                request.response_scope,
                records,
            ),)
        trace = (
            1,
            *_packed(query.stable_key()),
            *_packed(native_query.pattern.stable_key()),
            *_packed(_selection_key(selection)),
            len(direct),
            *(value for item in direct
              for value in _packed(item.stable_key())),
            len(candidates),
        )
        return QuestionExecutionResult(
            query,
            self.executed_reason,
            candidates,
            trace,
        )


__all__ = ["SymmetricPairQuestionExecutor"]
