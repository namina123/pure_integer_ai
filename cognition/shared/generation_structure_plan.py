"""G-02 篇章、命题和句法计划及 G-00 中间三层 resolver。

本模块复用 G-01 selected content、S-05 unresolved obligation 和 S-07 slot/value
类型，只保存待 surface 层消费的 typed 义务。它不读取 Attractor、PR、dag_path、
role_seq/token_seq 或旧 generate，也不在 Python 中写死语言角色与顺序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
    AnswerContentSelector,
    ContentArtifactAttachment,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerDecision,
    GenerationLayerResult,
    GenerationPlanProtocol,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.reasoning_planner import ReasoningObligation
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order import StructureSlotDefinition
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_int_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放整数 tuple，拒绝 bool、字符串和浮点混入 trace。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验计划动作或原因是注入的一等 MinimalInstruction。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return identity


def _require_structure(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验篇章 relation、句子、结构和约束均为一等 StructureConcept。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != OBJECT_STRUCTURE_CONCEPT:
        raise ValueError(f"{label} 必须是 StructureConcept")
    return identity


def _object_tuple(values: tuple[ObjectIdentity, ...], *, label: str) -> None:
    """核验开放一等对象 tuple，并拒绝重复身份。"""
    if not isinstance(values, tuple):
        raise TypeError(f"{label} 必须是 ObjectIdentity tuple")
    if any(not isinstance(item, ObjectIdentity) for item in values):
        raise TypeError(f"{label} 含非法项")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} 不得重复")


def _slot_key(slot: StructureSlotDefinition) -> tuple[int, ...]:
    """序列化 S-07 slot 定义，保持 structure/slot/Role/value type 全身份。"""
    if not isinstance(slot, StructureSlotDefinition):
        raise TypeError("slot 必须是 StructureSlotDefinition")
    return (
        *_packed(slot.structure.stable_key()),
        *_packed(slot.slot.stable_key()),
        *_packed(slot.role.stable_key()),
        *_packed(slot.value_type.stable_key()),
    )


def _slot_value_key(value: StructureSlotValue) -> tuple[int, ...]:
    """序列化 S-07 slot filler，不解释 filler 表面含义。"""
    if not isinstance(value, StructureSlotValue):
        raise TypeError("slot value 必须是 StructureSlotValue")
    return (
        *_packed(value.slot.stable_key()),
        *_packed(value.filler.stable_key()),
    )


def _topological_order(
        nodes: tuple[tuple[int, ...], ...],
        dependencies: tuple["DiscourseDependency", ...],
        ) -> tuple[tuple[int, ...], ...]:
    """对 typed 篇章依赖做确定性拓扑排序，环或端点漂移立即失败。"""
    node_set = set(nodes)
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for dependency in dependencies:
        if (dependency.before_candidate_key not in node_set
                or dependency.after_candidate_key not in node_set):
            raise ValueError("discourse dependency 端点不在 selected candidate")
        if dependency.after_candidate_key not in outgoing[
                dependency.before_candidate_key]:
            outgoing[dependency.before_candidate_key].add(
                dependency.after_candidate_key)
            indegree[dependency.after_candidate_key] += 1
    ready = sorted(node for node, count in indegree.items() if count == 0)
    ordered: list[tuple[int, ...]] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for target in sorted(outgoing[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(ordered) != len(nodes):
        raise ValueError("discourse dependency 含环")
    return tuple(ordered)


@dataclass(frozen=True)
class DiscourseDependency:
    """两个 selected Proposition 之间的一等篇章依赖。"""

    before_candidate_key: tuple[int, ...]
    after_candidate_key: tuple[int, ...]
    relation: ObjectIdentity
    reason: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        for label, key in (
                ("before", self.before_candidate_key),
                ("after", self.after_candidate_key)):
            _strict_int_tuple(key, label=f"discourse {label} candidate key")
            if not key:
                raise ValueError(f"discourse {label} candidate key 不能为空")
        if self.before_candidate_key == self.after_candidate_key:
            raise ValueError("discourse dependency 不得自环")
        _require_structure(self.relation, label="discourse relation")
        _require_instruction(self.reason, label="discourse dependency reason")
        _strict_int_tuple(self.trace, label="discourse dependency trace")
        if not self.trace:
            raise ValueError("discourse dependency trace 不能为空")

    def stable_key(self) -> tuple[int, ...]:
        """返回端点、relation、reason 和 trace 的完整依赖键。"""
        return (
            *_packed(self.before_candidate_key),
            *_packed(self.after_candidate_key),
            *_packed(self.relation.stable_key()),
            *_packed(self.reason.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class DiscoursePlan:
    """selected candidate 节点、无环依赖和 S-05 open question 集。"""

    selection_key: tuple[int, ...]
    candidate_keys: tuple[tuple[int, ...], ...]
    dependencies: tuple[DiscourseDependency, ...]
    open_questions: tuple[ReasoningObligation, ...]
    context: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        _strict_int_tuple(self.selection_key, label="discourse selection key")
        if not self.selection_key:
            raise ValueError("discourse selection key 不能为空")
        if not isinstance(self.candidate_keys, tuple):
            raise TypeError("discourse candidate_keys 必须是 tuple")
        for key in self.candidate_keys:
            _strict_int_tuple(key, label="discourse candidate key")
            if not key:
                raise ValueError("discourse candidate key 不能为空")
        if len(set(self.candidate_keys)) != len(self.candidate_keys):
            raise ValueError("discourse candidate key 不得重复")
        if not isinstance(self.dependencies, tuple):
            raise TypeError("discourse dependencies 必须是 tuple")
        if any(not isinstance(item, DiscourseDependency)
               for item in self.dependencies):
            raise TypeError("discourse dependencies 含非法项")
        if len({item.stable_key() for item in self.dependencies}) != len(
                self.dependencies):
            raise ValueError("discourse dependency 不得重复")
        if not isinstance(self.open_questions, tuple):
            raise TypeError("discourse open_questions 必须是 tuple")
        if any(not isinstance(item, ReasoningObligation)
               for item in self.open_questions):
            raise TypeError("discourse open_questions 含非法项")
        if len(set(self.open_questions)) != len(self.open_questions):
            raise ValueError("discourse open question 不得重复")
        _object_tuple(self.context, label="discourse context")
        candidate_keys = tuple(sorted(self.candidate_keys))
        dependencies = tuple(sorted(
            self.dependencies, key=lambda item: item.stable_key()))
        questions = tuple(sorted(
            self.open_questions, key=lambda item: item.stable_key()))
        order = _topological_order(candidate_keys, dependencies)
        object.__setattr__(self, "candidate_keys", candidate_keys)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "open_questions", questions)
        object.__setattr__(self, "_topological_order", order)

    @property
    def topological_order(self) -> tuple[tuple[int, ...], ...]:
        """返回依赖图的确定性顺序，不代表 surface 词序。"""
        return self._topological_order

    def stable_key(self) -> tuple[int, ...]:
        """返回 selection、节点、依赖、open question 和 context 完整键。"""
        result = [*_packed(self.selection_key), len(self.candidate_keys)]
        for key in self.candidate_keys:
            result.extend(_packed(key))
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        result.append(len(self.open_questions))
        for obligation in self.open_questions:
            result.extend(_packed(obligation.stable_key()))
        result.append(len(self.context))
        for identity in self.context:
            result.extend(_packed(identity.stable_key()))
        result.append(len(self.topological_order))
        for key in self.topological_order:
            result.extend(_packed(key))
        return tuple(result)


class DiscoursePlanner(Protocol):
    """从已核验 G-01 选择建立篇章依赖和开放问题。"""

    def plan(self, selection: AnswerContentSelection) -> DiscoursePlan:
        """返回不含 surface 排序的 typed discourse plan。"""
        ...


@dataclass(frozen=True)
class PlannedProposition:
    """一个 selected candidate 在命题层的完整 Evidence/Hypothesis 投影。"""

    candidate_key: tuple[int, ...]
    proposition: BoundProposition
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence: tuple[EvidenceRecord, ...]
    hypotheses: tuple[HypothesisKey, ...]
    qualifiers: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        _strict_int_tuple(self.candidate_key, label="planned candidate key")
        if not self.candidate_key:
            raise ValueError("planned candidate key 不能为空")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("planned proposition 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("planned proposition state 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("planned proposition source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("planned proposition scope 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("planned proposition evidence 类型错误")
        if not isinstance(self.hypotheses, tuple) or any(
                not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("planned proposition hypotheses 类型错误")
        _object_tuple(self.qualifiers, label="planned proposition qualifiers")

    def stable_key(self) -> tuple[int, ...]:
        """返回命题、四态、来源、Evidence/Hypothesis 和 qualifier 完整键。"""
        result = [
            *_packed(self.candidate_key),
            *_packed(self.proposition.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self.evidence),
        ]
        for item in self.evidence:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.hypotheses))
        for item in self.hypotheses:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.qualifiers))
        for item in self.qualifiers:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class PropositionPlan:
    """与 G-01 selected content 一一对应的来源化命题计划。"""

    selection_key: tuple[int, ...]
    propositions: tuple[PlannedProposition, ...]

    def __post_init__(self) -> None:
        _strict_int_tuple(self.selection_key, label="proposition selection key")
        if not self.selection_key:
            raise ValueError("proposition selection key 不能为空")
        if not isinstance(self.propositions, tuple):
            raise TypeError("proposition plan propositions 必须是 tuple")
        if any(not isinstance(item, PlannedProposition)
               for item in self.propositions):
            raise TypeError("proposition plan 含非法项")
        keys = tuple(item.candidate_key for item in self.propositions)
        if len(set(keys)) != len(keys):
            raise ValueError("proposition plan candidate 不得重复")
        object.__setattr__(self, "propositions", tuple(sorted(
            self.propositions, key=lambda item: item.candidate_key)))

    def stable_key(self) -> tuple[int, ...]:
        """返回 selection 和全部 planned proposition 完整键。"""
        result = [*_packed(self.selection_key), len(self.propositions)]
        for proposition in self.propositions:
            result.extend(_packed(proposition.stable_key()))
        return tuple(result)


class PropositionPlanner(Protocol):
    """把 G-01 selected candidate 映射为保留全部证据的命题计划。"""

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            ) -> PropositionPlan:
        """返回命题和 qualifier，不执行 surface 或排序。"""
        ...


@dataclass(frozen=True)
class PropositionSlotFiller:
    """把一个完整 BoundProposition 绑定到实际 S-07 slot filler。"""

    candidate_key: tuple[int, ...]
    proposition: BoundProposition
    value: StructureSlotValue

    def __post_init__(self) -> None:
        _strict_int_tuple(
            self.candidate_key, label="proposition slot candidate key")
        if not self.candidate_key:
            raise ValueError("proposition slot candidate key 不能为空")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("proposition slot proposition 类型错误")
        if not isinstance(self.value, StructureSlotValue):
            raise TypeError("proposition slot value 类型错误")
        if self.value.filler != self.proposition.template:
            raise ValueError("proposition slot filler 必须引用 bound template")

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、完整绑定命题和 S-07 value 的守恒键。"""
        return (
            *_packed(self.candidate_key),
            *_packed(self.proposition.stable_key()),
            *_packed(_slot_value_key(self.value)),
        )


@dataclass(frozen=True)
class PlannedSentence:
    """一个句子结构、成员命题、response act、slot/value 和显式边界。"""

    sentence: ObjectIdentity
    structure: ObjectIdentity
    ordinal: int
    proposition_keys: tuple[tuple[int, ...], ...]
    slots: tuple[StructureSlotDefinition, ...]
    values: tuple[StructureSlotValue, ...]
    proposition_fillers: tuple[PropositionSlotFiller, ...]
    boundary: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    response_act: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        _require_structure(self.sentence, label="planned sentence")
        _require_structure(self.structure, label="planned sentence structure")
        assert_int(self.ordinal, _where="planned sentence ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("planned sentence ordinal 必须是非负严格整数")
        if not isinstance(self.proposition_keys, tuple):
            raise TypeError("sentence proposition_keys 必须是 tuple")
        for key in self.proposition_keys:
            _strict_int_tuple(key, label="sentence proposition key")
            if not key:
                raise ValueError("sentence proposition key 不能为空")
        if len(set(self.proposition_keys)) != len(self.proposition_keys):
            raise ValueError("sentence proposition key 不得重复")
        if not isinstance(self.slots, tuple) or not self.slots:
            raise ValueError("planned sentence 必须携带非空 S-07 slot")
        if any(not isinstance(item, StructureSlotDefinition)
               for item in self.slots):
            raise TypeError("planned sentence slots 含非法项")
        if any(item.structure != self.structure for item in self.slots):
            raise ValueError("planned sentence slot 必须属于当前 structure")
        slot_ids = tuple(item.slot for item in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("planned sentence slot 不得重复")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("planned sentence 必须携带非空 slot value")
        if any(not isinstance(item, StructureSlotValue) for item in self.values):
            raise TypeError("planned sentence values 含非法项")
        value_slots = tuple(item.slot for item in self.values)
        if len(set(value_slots)) != len(value_slots):
            raise ValueError("planned sentence 同一 slot 不得重复赋值")
        if any(slot not in set(slot_ids) for slot in value_slots):
            raise ValueError("planned sentence value 引用了未声明 slot")
        if not isinstance(self.proposition_fillers, tuple):
            raise TypeError("planned sentence proposition_fillers 必须是 tuple")
        if any(not isinstance(item, PropositionSlotFiller)
               for item in self.proposition_fillers):
            raise TypeError("planned sentence proposition_fillers 含非法项")
        filler_keys = tuple(
            item.candidate_key for item in self.proposition_fillers)
        if len(set(filler_keys)) != len(filler_keys):
            raise ValueError("同一 Proposition 不得重复绑定 sentence slot")
        if set(filler_keys) != set(self.proposition_keys):
            raise ValueError("sentence slot binding 必须完整覆盖 proposition_keys")
        filler_values = tuple(
            item.value for item in self.proposition_fillers)
        if len(set(filler_values)) != len(filler_values):
            raise ValueError("不同 Proposition 不得共用同一 slot value")
        if not set(filler_values).issubset(set(self.values)):
            raise ValueError("sentence slot binding 必须对应实际 slot value")
        if self.response_act is not None:
            _require_instruction(
                self.response_act, label="planned sentence response act")
            act_values = tuple(
                item for item in self.values
                if item.filler == self.response_act
            )
            if len(act_values) != 1:
                raise ValueError("response act 必须恰绑定一个实际 slot value")
        if not self.proposition_keys and self.response_act is None:
            raise ValueError("无命题 planned sentence 必须显式绑定 response act")
        _require_instruction(self.boundary, label="sentence boundary")
        if not isinstance(self.source, SourceRef):
            raise TypeError("planned sentence source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("planned sentence scope 类型错误")
        object.__setattr__(self, "proposition_keys", tuple(sorted(
            self.proposition_keys)))
        object.__setattr__(self, "slots", tuple(sorted(
            self.slots, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "values", tuple(sorted(
            self.values, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "proposition_fillers", tuple(sorted(
            self.proposition_fillers,
            key=lambda item: item.candidate_key,
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回句子、结构、命题覆盖、slot/value、边界和归属完整键。"""
        result = [
            *_packed(self.sentence.stable_key()),
            *_packed(self.structure.stable_key()),
            self.ordinal,
            len(self.proposition_keys),
        ]
        for key in self.proposition_keys:
            result.extend(_packed(key))
        result.append(len(self.slots))
        for slot in self.slots:
            result.extend(_packed(_slot_key(slot)))
        result.append(len(self.values))
        for value in self.values:
            result.extend(_packed(_slot_value_key(value)))
        result.append(len(self.proposition_fillers))
        for filler in self.proposition_fillers:
            result.extend(_packed(filler.stable_key()))
        result.extend(_packed(self.boundary.stable_key()))
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.scope.stable_key()))
        result.append(0 if self.response_act is None else 1)
        if self.response_act is not None:
            result.extend(_packed(self.response_act.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class AnaphoraRequirement:
    """一个句内 slot 对 selected antecedent 命题的照应义务。"""

    sentence: ObjectIdentity
    slot: ObjectIdentity
    antecedent_candidate_key: tuple[int, ...]
    instruction: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_structure(self.sentence, label="anaphora sentence")
        _require_structure(self.slot, label="anaphora slot")
        _strict_int_tuple(
            self.antecedent_candidate_key,
            label="anaphora antecedent candidate key",
        )
        if not self.antecedent_candidate_key:
            raise ValueError("anaphora antecedent candidate key 不能为空")
        _require_instruction(self.instruction, label="anaphora instruction")
        _strict_int_tuple(self.trace, label="anaphora trace")
        if not self.trace:
            raise ValueError("anaphora trace 不能为空")

    def stable_key(self) -> tuple[int, ...]:
        """返回句子、slot、antecedent、指令和 trace 完整键。"""
        return (
            *_packed(self.sentence.stable_key()),
            *_packed(self.slot.stable_key()),
            *_packed(self.antecedent_candidate_key),
            *_packed(self.instruction.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class SyntaxLinearizationObligation:
    """交给 G-03/S-07 的句级 slot 值、约束身份和 query context。"""

    sentence: ObjectIdentity
    structure: ObjectIdentity
    values: tuple[StructureSlotValue, ...]
    constraints: tuple[ObjectIdentity, ...]
    context: tuple[ObjectIdentity, ...]
    reason: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        _require_structure(self.sentence, label="linearization sentence")
        _require_structure(self.structure, label="linearization structure")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("linearization obligation values 不能为空")
        if any(not isinstance(item, StructureSlotValue) for item in self.values):
            raise TypeError("linearization obligation values 含非法项")
        if len({item.slot for item in self.values}) != len(self.values):
            raise ValueError("linearization obligation slot 不得重复")
        if not isinstance(self.constraints, tuple):
            raise TypeError("linearization constraints 必须是 tuple")
        for constraint in self.constraints:
            _require_structure(constraint, label="linearization constraint")
        if len(set(self.constraints)) != len(self.constraints):
            raise ValueError("linearization constraint 不得重复")
        _object_tuple(self.context, label="linearization context")
        _require_instruction(self.reason, label="linearization reason")
        if not isinstance(self.source, SourceRef):
            raise TypeError("linearization source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("linearization scope 类型错误")
        object.__setattr__(self, "values", tuple(sorted(
            self.values, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "constraints", tuple(sorted(
            self.constraints, key=lambda item: item.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回句子、结构、slot 值、约束、context、reason 和归属完整键。"""
        result = [
            *_packed(self.sentence.stable_key()),
            *_packed(self.structure.stable_key()),
            len(self.values),
        ]
        for value in self.values:
            result.extend(_packed(_slot_value_key(value)))
        result.append(len(self.constraints))
        for constraint in self.constraints:
            result.extend(_packed(constraint.stable_key()))
        result.append(len(self.context))
        for identity in self.context:
            result.extend(_packed(identity.stable_key()))
        result.extend(_packed(self.reason.stable_key()))
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.scope.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SyntaxPlan:
    """句子、slot/value、照应和待线性化义务的完整句法计划。"""

    selection_key: tuple[int, ...]
    sentences: tuple[PlannedSentence, ...]
    anaphora: tuple[AnaphoraRequirement, ...]
    linearization: tuple[SyntaxLinearizationObligation, ...]

    def __post_init__(self) -> None:
        _strict_int_tuple(self.selection_key, label="syntax selection key")
        if not self.selection_key:
            raise ValueError("syntax selection key 不能为空")
        if not isinstance(self.sentences, tuple) or any(
                not isinstance(item, PlannedSentence) for item in self.sentences):
            raise TypeError("syntax sentences 类型错误")
        if len({item.sentence for item in self.sentences}) != len(self.sentences):
            raise ValueError("syntax sentence identity 不得重复")
        if len({item.ordinal for item in self.sentences}) != len(self.sentences):
            raise ValueError("syntax sentence ordinal 不得重复")
        if not isinstance(self.anaphora, tuple) or any(
                not isinstance(item, AnaphoraRequirement) for item in self.anaphora):
            raise TypeError("syntax anaphora 类型错误")
        anaphora_slots = tuple(
            (item.sentence, item.slot) for item in self.anaphora)
        if len(set(anaphora_slots)) != len(anaphora_slots):
            raise ValueError("同一 sentence/slot 不得重复声明 anaphora")
        if not isinstance(self.linearization, tuple) or any(
                not isinstance(item, SyntaxLinearizationObligation)
                for item in self.linearization):
            raise TypeError("syntax linearization 类型错误")
        sentence_map = {item.sentence: item for item in self.sentences}
        if set(item.sentence for item in self.linearization) != set(sentence_map):
            raise ValueError("每个 planned sentence 必须恰有一个 linearization obligation")
        if len(self.linearization) != len(self.sentences):
            raise ValueError("linearization obligation 不得重复覆盖 sentence")
        for obligation in self.linearization:
            sentence = sentence_map[obligation.sentence]
            if (obligation.structure != sentence.structure
                    or obligation.values != sentence.values
                    or obligation.source != sentence.source
                    or obligation.scope != sentence.scope):
                raise ValueError("linearization obligation 与 sentence 内容不一致")
        selected_keys = {
            key for sentence in self.sentences for key in sentence.proposition_keys}
        slot_by_sentence = {
            sentence.sentence: {slot.slot for slot in sentence.slots}
            for sentence in self.sentences
        }
        for requirement in self.anaphora:
            if requirement.sentence not in sentence_map:
                raise ValueError("anaphora 引用了未知 sentence")
            if requirement.slot not in slot_by_sentence[requirement.sentence]:
                raise ValueError("anaphora 引用了 sentence 未声明 slot")
            if requirement.antecedent_candidate_key not in selected_keys:
                raise ValueError("anaphora antecedent 不在 selected Proposition")
        object.__setattr__(self, "sentences", tuple(sorted(
            self.sentences, key=lambda item: item.ordinal)))
        object.__setattr__(self, "anaphora", tuple(sorted(
            self.anaphora, key=lambda item: item.stable_key())))
        ordinal_by_sentence = {
            item.sentence: item.ordinal for item in self.sentences}
        object.__setattr__(self, "linearization", tuple(sorted(
            self.linearization,
            key=lambda item: ordinal_by_sentence[item.sentence],
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回 selection、句子、照应和线性化义务完整键。"""
        result = [*_packed(self.selection_key), len(self.sentences)]
        for sentence in self.sentences:
            result.extend(_packed(sentence.stable_key()))
        result.append(len(self.anaphora))
        for requirement in self.anaphora:
            result.extend(_packed(requirement.stable_key()))
        result.append(len(self.linearization))
        for obligation in self.linearization:
            result.extend(_packed(obligation.stable_key()))
        return tuple(result)


class SyntaxPlanner(Protocol):
    """把命题计划映射为 S-07 slot 和待 G-03 线性化义务。"""

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
            ) -> SyntaxPlan:
        """返回句法结构，不选择词形、不执行顺序搜索。"""
        ...


def _selected_candidates(
        selection: AnswerContentSelection,
        ) -> dict[tuple[int, ...], object]:
    """按 G-01 已选键返回候选映射，拒绝 selection 内部漂移。"""
    selected_keys = set(selection.selected_candidate_keys)
    selected = {
        item.stable_key(): item
        for item in selection.request.candidates
        if item.stable_key() in selected_keys
    }
    if set(selected) != selected_keys:
        raise ValueError("G-02 selection 含请求外候选")
    return selected


def _validate_discourse(
        selection: AnswerContentSelection,
        discourse: DiscoursePlan,
        ) -> dict[tuple[int, ...], object]:
    """核验篇章节点与 open question 完整覆盖 G-01/S-05 输入。"""
    if discourse.selection_key != selection.stable_key():
        raise ValueError("discourse 必须绑定当前 G-01 selection")
    selected = _selected_candidates(selection)
    selected_keys = set(selection.selected_candidate_keys)
    if set(discourse.candidate_keys) != selected_keys:
        raise ValueError("discourse 节点必须完整覆盖 selected candidate")
    expected_questions = {
        obligation
        for candidate in selected.values()
        if candidate.reasoning is not None
        for obligation in candidate.reasoning.unresolved
    }
    if set(discourse.open_questions) != expected_questions:
        raise ValueError("discourse open question 必须完整保留 S-05 unresolved")
    return selected


def _validate_propositions(
        selection: AnswerContentSelection,
        discourse: DiscoursePlan,
        propositions: PropositionPlan,
        ) -> dict[tuple[int, ...], object]:
    """核验命题层逐候选保存原始四态、Evidence 和 Hypothesis。"""
    selected = _validate_discourse(selection, discourse)
    selection_key = selection.stable_key()
    if propositions.selection_key != selection_key:
        raise ValueError("proposition plan 必须绑定当前 G-01 selection")
    planned = {item.candidate_key: item for item in propositions.propositions}
    selected_keys = set(selection.selected_candidate_keys)
    if set(planned) != selected_keys:
        raise ValueError("proposition plan 必须完整覆盖 selected candidate")
    for key, item in planned.items():
        candidate = selected[key]
        if (item.proposition != candidate.proposition
                or item.state != candidate.state
                or item.source != candidate.source
                or item.scope != candidate.scope
                or item.evidence != candidate.evidence
                or item.hypotheses != candidate.hypotheses):
            raise ValueError("planned proposition 丢失或替换了候选 Evidence 身份")
    return selected


def _validate_syntax(
        selection: AnswerContentSelection,
        discourse: DiscoursePlan,
        propositions: PropositionPlan,
        syntax: SyntaxPlan,
        ) -> None:
    """核验句法覆盖、篇章顺序、照应方向和待线性化义务守恒。"""
    selected = _validate_propositions(selection, discourse, propositions)
    selection_key = selection.stable_key()
    if syntax.selection_key != selection_key:
        raise ValueError("syntax plan 必须绑定当前 G-01 selection")
    selected_keys = set(selection.selected_candidate_keys)
    coverage: list[tuple[int, ...]] = []
    sentence_map = {item.sentence: item for item in syntax.sentences}
    sentence_ordinal_by_candidate: dict[tuple[int, ...], int] = {}
    for sentence in syntax.sentences:
        coverage.extend(sentence.proposition_keys)
        if (sentence.response_act is not None
                and sentence.response_act != selection.stance):
            raise ValueError("syntax response act 与 G-01 selection stance 不一致")
        for key in sentence.proposition_keys:
            sentence_ordinal_by_candidate[key] = sentence.ordinal
        fillers = {
            item.candidate_key: item
            for item in sentence.proposition_fillers
        }
        for key in sentence.proposition_keys:
            candidate = selected.get(key)
            if candidate is None:
                raise ValueError("syntax sentence 引用了未选择候选")
            filler = fillers[key]
            if filler.proposition != candidate.proposition:
                raise ValueError("syntax slot 丢失 BoundProposition 绑定")
    if len(coverage) != len(set(coverage)) or set(coverage) != selected_keys:
        raise ValueError("每个 selected Proposition 必须恰被一个 sentence 覆盖")
    if not selected_keys and not sentence_map:
        raise ValueError("无 selected Proposition 时仍必须规划 response-act sentence")
    for dependency in discourse.dependencies:
        if (sentence_ordinal_by_candidate[dependency.before_candidate_key]
                > sentence_ordinal_by_candidate[
                    dependency.after_candidate_key]):
            raise ValueError("syntax sentence ordinal 违反 discourse dependency")
    sentence_ordinal_by_identity = {
        item.sentence: item.ordinal for item in syntax.sentences}
    for requirement in syntax.anaphora:
        if (sentence_ordinal_by_candidate[
                requirement.antecedent_candidate_key]
                > sentence_ordinal_by_identity[requirement.sentence]):
            raise ValueError("anaphora antecedent 不得位于未来 sentence")


@dataclass(frozen=True)
class GenerationStructurePlan:
    """G-01 selection 到 discourse/proposition/syntax 三层的守恒结果。"""

    selection: AnswerContentSelection
    discourse: DiscoursePlan
    propositions: PropositionPlan
    syntax: SyntaxPlan

    def __post_init__(self) -> None:
        if not isinstance(self.selection, AnswerContentSelection):
            raise TypeError("structure plan selection 类型错误")
        if not isinstance(self.discourse, DiscoursePlan):
            raise TypeError("structure plan discourse 类型错误")
        if not isinstance(self.propositions, PropositionPlan):
            raise TypeError("structure plan propositions 类型错误")
        if not isinstance(self.syntax, SyntaxPlan):
            raise TypeError("structure plan syntax 类型错误")
        _validate_syntax(
            self.selection,
            self.discourse,
            self.propositions,
            self.syntax,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 selection 及三层 typed 计划完整键。"""
        return (
            *_packed(self.selection.stable_key()),
            *_packed(self.discourse.stable_key()),
            *_packed(self.propositions.stable_key()),
            *_packed(self.syntax.stable_key()),
        )


class GenerationStructurePlanner:
    """依次调用三类 mapper，并由 GenerationStructurePlan 做内容守恒核验。"""

    def __init__(
            self,
            discourse: DiscoursePlanner,
            propositions: PropositionPlanner,
            syntax: SyntaxPlanner,
            ) -> None:
        for label, planner in (
                ("discourse", discourse),
                ("proposition", propositions),
                ("syntax", syntax)):
            if not hasattr(planner, "plan"):
                raise TypeError(f"{label} planner 必须实现 plan")
        self._discourse = discourse
        self._propositions = propositions
        self._syntax = syntax

    def plan(self, selection: AnswerContentSelection) -> GenerationStructurePlan:
        """建立局部三层计划；mapper 输出必须逐层绑定同一 selection。"""
        discourse = self.plan_discourse(selection)
        propositions = self.plan_propositions(selection, discourse)
        syntax = self.plan_syntax(selection, discourse, propositions)
        return GenerationStructurePlan(
            selection, discourse, propositions, syntax)

    def plan_discourse(
            self, selection: AnswerContentSelection,
            ) -> DiscoursePlan:
        """只执行并核验 discourse mapper，不提前触碰未来层。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("structure planner 只接受 AnswerContentSelection")
        discourse = self._discourse.plan(selection)
        if not isinstance(discourse, DiscoursePlan):
            raise TypeError("discourse planner 必须返回 DiscoursePlan")
        _validate_discourse(selection, discourse)
        return discourse

    def plan_propositions(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            ) -> PropositionPlan:
        """只执行并核验 proposition mapper，不提前触碰 syntax。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("structure planner 只接受 AnswerContentSelection")
        if not isinstance(discourse, DiscoursePlan):
            raise TypeError("discourse 类型错误")
        _validate_discourse(selection, discourse)
        propositions = self._propositions.plan(selection, discourse)
        if not isinstance(propositions, PropositionPlan):
            raise TypeError("proposition planner 必须返回 PropositionPlan")
        _validate_propositions(selection, discourse, propositions)
        return propositions

    def plan_syntax(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
            ) -> SyntaxPlan:
        """只执行并核验 syntax mapper，输出待 G-03 消费义务。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("structure planner 只接受 AnswerContentSelection")
        if not isinstance(discourse, DiscoursePlan):
            raise TypeError("discourse 类型错误")
        if not isinstance(propositions, PropositionPlan):
            raise TypeError("propositions 类型错误")
        _validate_propositions(selection, discourse, propositions)
        syntax = self._syntax.plan(selection, discourse, propositions)
        if not isinstance(syntax, SyntaxPlan):
            raise TypeError("syntax planner 必须返回 SyntaxPlan")
        _validate_syntax(selection, discourse, propositions, syntax)
        return syntax


@dataclass(frozen=True)
class GenerationStructureLayerProtocol:
    """注入 discourse/proposition/syntax 三层成功 reason。"""

    discourse_reason: ObjectIdentity
    proposition_reason: ObjectIdentity
    syntax_reason: ObjectIdentity

    def __post_init__(self) -> None:
        reasons = self.reasons()
        if len(set(reasons)) != len(reasons):
            raise ValueError("G-02 layer reason 必须互不相同")
        for reason in reasons:
            _require_instruction(reason, label="G-02 layer reason")

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """返回三层 reason，供 resolver 稳定接线。"""
        return self.discourse_reason, self.proposition_reason, self.syntax_reason


class _GenerationStructureLayerResolver:
    """为 G-00 中间三层独立重算 G-01/G-02 计划并核验上游 payload。"""

    def __init__(
            self,
            layer_index: int,
            planner_protocol: GenerationPlanProtocol,
            layer_protocol: GenerationStructureLayerProtocol,
            selector: AnswerContentSelector,
            planner: GenerationStructurePlanner,
            artifacts: Sequence[ContentArtifactAttachment] = (),
            ) -> None:
        assert_int(layer_index, _where="G-02 layer_index")
        if type(layer_index) is not int or layer_index not in (2, 3, 4):
            raise ValueError("G-02 layer_index 必须是 discourse/proposition/syntax")
        if not isinstance(planner_protocol, GenerationPlanProtocol):
            raise TypeError("G-02 resolver planner protocol 类型错误")
        if not isinstance(layer_protocol, GenerationStructureLayerProtocol):
            raise TypeError("G-02 resolver layer protocol 类型错误")
        if not isinstance(selector, AnswerContentSelector):
            raise TypeError("G-02 resolver selector 类型错误")
        if not isinstance(planner, GenerationStructurePlanner):
            raise TypeError("G-02 resolver planner 类型错误")
        if not isinstance(artifacts, Sequence):
            raise TypeError("G-02 resolver artifacts 必须是 Sequence")
        self._index = layer_index
        self._planner_protocol = planner_protocol
        self._layer_protocol = layer_protocol
        self._selector = selector
        self._planner = planner
        self._artifacts = tuple(artifacts)

    def resolve(
            self,
            request: GenerationPlanningRequest,
            prior: tuple[GenerationLayerResult, ...],
            ) -> GenerationLayerDecision:
        """独立重算 selection 和三层计划，并拒绝任一上游 payload 漂移。"""
        if len(prior) != self._index:
            raise ValueError("G-02 layer 上游数量不匹配")
        expected_layers = self._planner_protocol.layers()[:self._index]
        if tuple(item.layer for item in prior) != expected_layers:
            raise ValueError("G-02 layer 上游顺序不匹配")
        if any(item.outcome != self._planner_protocol.complete for item in prior):
            raise ValueError("G-02 layer 只接受 complete 上游")
        selection = self._selector.select(request, self._artifacts)
        selection_key = selection.stable_key()
        if prior[0].payload != selection_key or prior[1].payload != selection_key:
            raise ValueError("G-02 layer 与 stance/content selection 不一致")
        discourse = self._planner.plan_discourse(selection)
        payloads = [discourse.stable_key()]
        propositions = None
        if self._index >= 3:
            propositions = self._planner.plan_propositions(
                selection, discourse)
            payloads.append(propositions.stable_key())
        if self._index >= 4:
            if propositions is None:
                raise RuntimeError("syntax layer 缺少 proposition plan")
            syntax = self._planner.plan_syntax(
                selection, discourse, propositions)
            payloads.append(syntax.stable_key())
        for offset, payload in enumerate(payloads[:self._index - 2]):
            if prior[2 + offset].payload != payload:
                raise ValueError("G-02 三层独立重算结果与上游不一致")
        layer = self._planner_protocol.layers()[self._index]
        reason = self._layer_protocol.reasons()[self._index - 2]
        payload = payloads[self._index - 2]
        return GenerationLayerDecision(
            layer,
            self._planner_protocol.complete,
            reason,
            selection.selected_candidate_keys,
            payload,
            (*_packed(selection_key), *_packed(payload)),
        )


class GenerationDiscourseLayerResolver(_GenerationStructureLayerResolver):
    """G-00 discourse 层 resolver。"""

    def __init__(self, *args, **kwargs) -> None:
        """固定为 G-00 第三层，不暴露可错配索引。"""
        super().__init__(2, *args, **kwargs)


class GenerationPropositionLayerResolver(_GenerationStructureLayerResolver):
    """G-00 proposition 层 resolver。"""

    def __init__(self, *args, **kwargs) -> None:
        """固定为 G-00 第四层，不暴露可错配索引。"""
        super().__init__(3, *args, **kwargs)


class GenerationSyntaxLayerResolver(_GenerationStructureLayerResolver):
    """G-00 syntax 层 resolver。"""

    def __init__(self, *args, **kwargs) -> None:
        """固定为 G-00 第五层，不暴露可错配索引。"""
        super().__init__(4, *args, **kwargs)


__all__ = [
    "AnaphoraRequirement",
    "DiscourseDependency",
    "DiscoursePlan",
    "DiscoursePlanner",
    "GenerationDiscourseLayerResolver",
    "GenerationPropositionLayerResolver",
    "GenerationStructureLayerProtocol",
    "GenerationStructurePlan",
    "GenerationStructurePlanner",
    "GenerationSyntaxLayerResolver",
    "PlannedProposition",
    "PlannedSentence",
    "PropositionSlotFiller",
    "PropositionPlan",
    "PropositionPlanner",
    "SyntaxLinearizationObligation",
    "SyntaxPlan",
    "SyntaxPlanner",
]
