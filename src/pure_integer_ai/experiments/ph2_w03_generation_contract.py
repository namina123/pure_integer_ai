"""W03-G target Sense 生成选择、精确 Use 与 outcome 值对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_REPRESENTATION,
    OBJECT_SENSE,
    OBJECT_SPAN,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)


W03_GENERATION_READY = "READY"
W03_GENERATION_CLARIFY = "CLARIFY"
W03_GENERATION_UNKNOWN = "UNKNOWN"
W03_GENERATION_STATUSES = (
    W03_GENERATION_READY,
    W03_GENERATION_CLARIFY,
    W03_GENERATION_UNKNOWN,
)
W03_GENERATION_ADOPTED = "ADOPTED"
W03_GENERATION_REJECTED = "REJECTED"
W03_GENERATION_ACTIONS = (
    W03_GENERATION_ADOPTED,
    W03_GENERATION_REJECTED,
)
W03_GENERATION_OUTCOME_SUPPORT = "SUPPORT"
W03_GENERATION_OUTCOME_REFUTE = "REFUTE"
W03_GENERATION_OUTCOME_NEUTRAL = "NEUTRAL"
W03_GENERATION_OUTCOMES = (
    W03_GENERATION_OUTCOME_SUPPORT,
    W03_GENERATION_OUTCOME_REFUTE,
    W03_GENERATION_OUTCOME_NEUTRAL,
)
W03_GENERATION_HARD_CASES = (
    "TARGET_SENSE_LEGAL_SURFACE",
    "HOMOGRAPH_SENSE_ISOLATION",
    "AMBIGUITY_FAIL_CLOSED",
    "MULTIPLE_LEGAL_SURFACES",
    "WITHDRAWAL_SUPERSEDE_AVAILABILITY",
)

_NAMESPACE = 30304
_STATUS_CODE = {
    W03_GENERATION_READY: 1,
    W03_GENERATION_CLARIFY: 2,
    W03_GENERATION_UNKNOWN: 3,
}
_ACTION_CODE = {
    W03_GENERATION_ADOPTED: 1,
    W03_GENERATION_REJECTED: 2,
}
_OUTCOME_CODE = {
    W03_GENERATION_OUTCOME_SUPPORT: 1,
    W03_GENERATION_OUTCOME_REFUTE: 2,
    W03_GENERATION_OUTCOME_NEUTRAL: 3,
}


class W03GenerationError(RuntimeError):
    """W03-G 请求、授权、choice 或 append-only ledger 非法。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _identity(
        value: ObjectIdentity,
        kind: int,
        *,
        where: str,
        ) -> ObjectIdentity:
    if not isinstance(value, ObjectIdentity) or value.object_kind != kind:
        raise W03GenerationError(f"{where} identity 非法")
    return value


def _strict_bool(value: bool, *, where: str) -> bool:
    if type(value) is not bool:
        raise W03GenerationError(f"{where} 必须是严格 bool")
    return value


@dataclass(frozen=True)
class W03ExpressionConstraints:
    """不含具体答案文本的有界表层表达约束。"""

    allow_single_atom: bool
    allow_lexicalized_multiword: bool
    max_codepoints: int

    def __post_init__(self) -> None:
        _strict_bool(self.allow_single_atom, where="allow_single_atom")
        _strict_bool(
            self.allow_lexicalized_multiword,
            where="allow_lexicalized_multiword",
        )
        if not self.allow_single_atom and not self.allow_lexicalized_multiword:
            raise W03GenerationError("表达约束拒绝了全部 surface 类别")
        assert_int(self.max_codepoints, _where="max_codepoints")
        if type(self.max_codepoints) is not int or self.max_codepoints <= 0:
            raise W03GenerationError("max_codepoints 必须是严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            int(self.allow_single_atom),
            int(self.allow_lexicalized_multiword),
            self.max_codepoints,
        )


@dataclass(frozen=True)
class W03GenerationRequest:
    """由 target identity 和调用 scope 构成的生成请求。"""

    request_key: LosslessIntegerKey
    target_sense: ObjectIdentity
    target_concept: ObjectIdentity
    context: ObjectIdentity | None
    branch: ObjectIdentity
    constraints: W03ExpressionConstraints
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, LosslessIntegerKey):
            raise W03GenerationError("request_key 类型非法")
        _identity(self.target_sense, OBJECT_SENSE, where="target Sense")
        _identity(self.target_concept, OBJECT_CONCEPT, where="target Concept")
        if self.context is not None:
            _identity(self.context, OBJECT_CONTEXT_SCOPE, where="context")
        _identity(self.branch, OBJECT_LANGUAGE_BRANCH, where="LanguageBranch")
        if not isinstance(self.constraints, W03ExpressionConstraints):
            raise W03GenerationError("expression constraints 类型非法")
        if not isinstance(self.source, SourceRef):
            raise W03GenerationError("request source 类型非法")
        if not isinstance(self.scope, ScopeIdentity):
            raise W03GenerationError("request scope 类型非法")
        if (self.scope.source != self.source
                or self.scope.owner != self.source.owner
                or self.scope.versions != self.source.versions):
            raise W03GenerationError("request source/scope identity 漂移")

    def stable_key(self) -> tuple[int, ...]:
        context_key = () if self.context is None else self.context.stable_key()
        return (
            1,
            *_pack(self.request_key.components),
            *_pack(self.target_sense.stable_key()),
            *_pack(self.target_concept.stable_key()),
            *_pack(context_key),
            *_pack(self.branch.stable_key()),
            *_pack(self.constraints.stable_key()),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
        )


@dataclass(frozen=True)
class W03GenerationOption:
    """一个经当前 active projection 授权的精确 surface/Sense 选项。"""

    surface: str
    sense: ObjectIdentity
    concept: ObjectIdentity
    context: ObjectIdentity
    branch: ObjectIdentity
    atom: ObjectIdentity
    representation: ObjectIdentity
    span: ObjectIdentity
    forming_source: SourceRef
    lexicalized_multiword: bool
    authorization_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.surface, str) or not self.surface:
            raise W03GenerationError("surface 必须是非空文本")
        _identity(self.sense, OBJECT_SENSE, where="option Sense")
        _identity(self.concept, OBJECT_CONCEPT, where="option Concept")
        _identity(self.context, OBJECT_CONTEXT_SCOPE, where="option context")
        _identity(self.branch, OBJECT_LANGUAGE_BRANCH, where="option branch")
        _identity(self.atom, OBJECT_LANGUAGE_ATOM, where="option atom")
        _identity(
            self.representation,
            OBJECT_REPRESENTATION,
            where="option representation",
        )
        _identity(self.span, OBJECT_SPAN, where="option span")
        if not isinstance(self.forming_source, SourceRef):
            raise W03GenerationError("option forming source 类型非法")
        _strict_bool(
            self.lexicalized_multiword,
            where="option lexicalized_multiword",
        )
        if not isinstance(self.authorization_key, LosslessIntegerKey):
            raise W03GenerationError("option authorization key 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            *_pack(self.sense.stable_key()),
            *_pack(self.concept.stable_key()),
            *_pack(self.context.stable_key()),
            *_pack(self.branch.stable_key()),
            *_pack(self.atom.stable_key()),
            *_pack(self.representation.stable_key()),
            *_pack(self.span.stable_key()),
            *_pack(self.forming_source.stable_key()),
            int(self.lexicalized_multiword),
            len(self.surface),
            *(ord(character) for character in self.surface),
            *_pack(self.authorization_key.components),
        )


@dataclass(frozen=True)
class W03GenerationChoice:
    """不私选首项的完整合法 option 集。"""

    request: W03GenerationRequest
    status: str
    options: tuple[W03GenerationOption, ...]
    selected: W03GenerationOption | None
    reason_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.request, W03GenerationRequest):
            raise W03GenerationError("choice request 类型非法")
        if self.status not in W03_GENERATION_STATUSES:
            raise W03GenerationError("choice status 非法")
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, W03GenerationOption)
                       for item in self.options)):
            raise W03GenerationError("choice options 类型非法")
        normalized = tuple(sorted(self.options, key=lambda item: item.stable_key()))
        if self.options != normalized or len(set(self.options)) != len(self.options):
            raise W03GenerationError("choice options 必须规范排序且不重复")
        if self.status == W03_GENERATION_READY:
            if not self.options:
                raise W03GenerationError("READY choice 缺合法 option")
        elif self.options:
            raise W03GenerationError("非 READY choice 不得泄漏 option")
        if self.selected is not None:
            raise W03GenerationError("choice 不得按排序预选 surface")
        if not isinstance(self.reason_key, LosslessIntegerKey):
            raise W03GenerationError("choice reason key 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            _STATUS_CODE[self.status],
            *_pack(self.request.stable_key()),
            len(self.options),
        ]
        for item in self.options:
            values.extend(_pack(item.stable_key()))
        values.extend(_pack(self.reason_key.components))
        return tuple(values)


@dataclass(frozen=True)
class W03GenerationDecision:
    """一个 choice 内对单个 option 的采用或拒绝记录。"""

    choice: W03GenerationChoice
    option: W03GenerationOption
    action: str
    decision_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if not isinstance(self.choice, W03GenerationChoice):
            raise W03GenerationError("decision choice 类型非法")
        if self.option not in self.choice.options:
            raise W03GenerationError("decision option 不属于 choice")
        if self.action not in W03_GENERATION_ACTIONS:
            raise W03GenerationError("decision action 非法")
        if not isinstance(self.decision_key, LosslessIntegerKey):
            raise W03GenerationError("decision key 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            _ACTION_CODE[self.action],
            *_pack(self.choice.stable_key()),
            *_pack(self.option.stable_key()),
            *_pack(self.decision_key.components),
        )


@dataclass(frozen=True)
class W03GenerationUse:
    """采用或拒绝一个精确 option 的 lossless Use。"""

    decision: W03GenerationDecision
    ref: GenerationChoiceUseRef

    def __post_init__(self) -> None:
        if not isinstance(self.decision, W03GenerationDecision):
            raise W03GenerationError("generation Use decision 类型非法")
        if not isinstance(self.ref, GenerationChoiceUseRef):
            raise W03GenerationError("generation Use ref 类型非法")
        if self.ref.scope != self.decision.choice.request.scope:
            raise W03GenerationError("generation Use scope 漂移")
        if self.ref.selection_key.components != self.decision.option.stable_key():
            raise W03GenerationError("generation Use 未绑定精确 option")

    def stable_key(self) -> tuple[int, ...]:
        return 1, *_pack(self.decision.stable_key()), *_pack(self.ref.stable_key())


@dataclass(frozen=True)
class W03GenerationOutcome:
    """当前 active projection 对一个 exact Use 的分型 verifier 结果。"""

    use: W03GenerationUse
    verdict: str
    ref: GenerationChoiceOutcomeRef
    current_authorization_key: LosslessIntegerKey | None

    def __post_init__(self) -> None:
        if not isinstance(self.use, W03GenerationUse):
            raise W03GenerationError("generation outcome Use 类型非法")
        if self.verdict not in W03_GENERATION_OUTCOMES:
            raise W03GenerationError("generation outcome verdict 非法")
        if not isinstance(self.ref, GenerationChoiceOutcomeRef):
            raise W03GenerationError("generation outcome ref 类型非法")
        if self.ref.use_key != self.use.ref.use_key:
            raise W03GenerationError("generation outcome 未绑定 exact Use")
        if (self.current_authorization_key is not None
                and not isinstance(
                    self.current_authorization_key,
                    LosslessIntegerKey,
                )):
            raise W03GenerationError("current authorization key 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        authorization = (
            () if self.current_authorization_key is None
            else self.current_authorization_key.components
        )
        return (
            1,
            _OUTCOME_CODE[self.verdict],
            *_pack(self.use.stable_key()),
            *_pack(self.ref.stable_key()),
            *_pack(authorization),
        )


@dataclass(frozen=True)
class W03GenerationCaseResult:
    """一个公开 evaluator case 的逻辑真假；异常不在此对象内吞并。"""

    case_name: str
    passed: bool
    evidence_key: LosslessIntegerKey

    def __post_init__(self) -> None:
        if self.case_name not in W03_GENERATION_HARD_CASES:
            raise W03GenerationError("generation hard case 未注册")
        _strict_bool(self.passed, where="generation case passed")
        if not isinstance(self.evidence_key, LosslessIntegerKey):
            raise W03GenerationError("generation case evidence key 类型非法")

    def stable_key(self) -> tuple[int, ...]:
        return (
            1,
            W03_GENERATION_HARD_CASES.index(self.case_name) + 1,
            int(self.passed),
            *_pack(self.evidence_key.components),
        )


@dataclass(frozen=True)
class W03GenerationHardConjunctReport:
    """W-03 generation supplemental dimension 的 1/1 硬合取。"""

    status: str
    passed: int
    required: int
    fail_count: int
    ne_count: int
    cases: tuple[W03GenerationCaseResult, ...]
    sense_consumer_connected: bool
    choice_bridge_connected: bool

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "FAIL"}:
            raise W03GenerationError("hard conjunct status 非法")
        for name in ("passed", "required", "fail_count", "ne_count"):
            assert_int(getattr(self, name), _where=f"hard conjunct {name}")
        if self.required != 1 or self.passed not in {0, 1}:
            raise W03GenerationError("hard conjunct score 非法")
        if self.ne_count != 0:
            raise W03GenerationError("逻辑 generation case 不得伪装为 NE")
        expected_pass = int(
            all(item.passed for item in self.cases)
            and self.sense_consumer_connected
            and self.choice_bridge_connected
        )
        if (self.passed != expected_pass
                or self.fail_count != 1 - expected_pass
                or self.status != ("PASS" if expected_pass else "FAIL")):
            raise W03GenerationError("hard conjunct aggregate 漂移")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            1,
            self.passed,
            self.required,
            self.fail_count,
            self.ne_count,
            int(self.sense_consumer_connected),
            int(self.choice_bridge_connected),
            len(self.cases),
        ]
        for item in self.cases:
            values.extend(_pack(item.stable_key()))
        return tuple(values)


def choice_reason_key(status: str) -> LosslessIntegerKey:
    """按分型结果建立不含表层文本的公开原因键。"""
    try:
        return LosslessIntegerKey((_NAMESPACE, 100 + _STATUS_CODE[status]))
    except KeyError as exc:
        raise W03GenerationError("choice status 未注册") from exc


__all__ = [
    "W03_GENERATION_ACTIONS",
    "W03_GENERATION_ADOPTED",
    "W03_GENERATION_CLARIFY",
    "W03_GENERATION_HARD_CASES",
    "W03_GENERATION_OUTCOME_NEUTRAL",
    "W03_GENERATION_OUTCOME_REFUTE",
    "W03_GENERATION_OUTCOME_SUPPORT",
    "W03_GENERATION_OUTCOMES",
    "W03_GENERATION_READY",
    "W03_GENERATION_REJECTED",
    "W03_GENERATION_STATUSES",
    "W03_GENERATION_UNKNOWN",
    "W03ExpressionConstraints",
    "W03GenerationCaseResult",
    "W03GenerationChoice",
    "W03GenerationDecision",
    "W03GenerationError",
    "W03GenerationHardConjunctReport",
    "W03GenerationOption",
    "W03GenerationOutcome",
    "W03GenerationRequest",
    "W03GenerationUse",
    "choice_reason_key",
]
