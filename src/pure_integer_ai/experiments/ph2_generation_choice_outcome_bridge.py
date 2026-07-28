"""GG-02 分层生成选择、精确 Use、verifier claim 与 assessment 输入桥。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VerificationReport,
    VerificationResult,
)


FORMAT_VERSION = 1
ASSESSMENT_STATES = ("NE_LAYER_DISABLED", "NE_VERIFIER", "READY")
GG02_CONTRACT_TYPES = (
    "GENERATION_CHOICE_ASSESSMENT_INPUT",
    "GENERATION_CHOICE_EPISODE_ATTRIBUTION",
    "GENERATION_CHOICE_LAYER_OUTCOME",
    "GENERATION_CHOICE_USE_ATTRIBUTION",
    "GENERATION_LAYERED_OUTCOME_REPORT",
    "GENERATION_VERIFIER_LAYER_ROUTE",
)
GG02_INVARIANTS = (
    "ASSESSMENT_CONSUMER_REQUIRED_BUT_NOT_CONNECTED",
    "DISABLED_LAYER_ONLY_CHANGES_OWN_ASSESSMENT",
    "EPISODE_SUCCESS_NEVER_REWARDS_ALL_CHOICES",
    "EXACT_USE_REQUIRED_FOR_EVERY_CHOICE",
    "FIVE_LAYER_ATTRIBUTION_COMPLETE",
    "OUTCOME_LEDGER_IS_NOT_ASSESSMENT_UPDATE",
    "TEACHER_CALL_ZERO",
    "VERIFIER_CANNOT_CLAIM_UNDECLARED_LAYER",
    "ZERO_HOST_LEARNING_WRITE",
)
GG02_VERIFIER_DIMENSIONS = (
    "ASSESSMENT_LAYER_ISOLATION",
    "CLAIM_LAYER_AUTHORIZATION",
    "EXACT_USE_OUTCOME_LINK",
    "FIVE_LAYER_COMPLETENESS",
    "NO_SENTENCE_WIDE_BROADCAST",
    "OWNER_SCOPE_QUERY_BINDING",
    "READ_ONLY_VERIFIER_INPUT",
    "ZERO_HOST_LEARNING_WRITE",
)
GG02_NE_CONDITIONS = (
    "ASSESSMENT_CONSUMER_NOT_CONNECTED",
    "GG03_COMBINATION_COURSE_NOT_FROZEN",
    "RUNTIME_CHOICE_ADOPTION_NOT_EXECUTED",
    "W_TRAINING_NOT_EXECUTED",
)
EXECUTION_STATE_KEYS = (
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "formal_training_runs",
    "mastered_claims",
    "memory_learning_writes",
    "readiness_claims",
    "teacher_calls",
    "use_learning_writes",
    "w01_started",
)


class GenerationChoiceOutcomeBridgeError(RuntimeError):
    """GG-02 归因不完整、跨层、跨 query 或可能扩散奖惩。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _choice_order(value: str) -> int:
    if value not in CHOICE_KINDS:
        raise GenerationChoiceOutcomeBridgeError("choice kind 未注册")
    return CHOICE_KINDS.index(value)


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GenerationChoiceOutcomeBridgeError(f"{where} 必须是非空文本")
    return value


def _text_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise GenerationChoiceOutcomeBridgeError(f"{where} 必须是非空文本 tuple")
    if value != tuple(sorted(set(value))):
        raise GenerationChoiceOutcomeBridgeError(f"{where} 必须排序且去重")
    return value


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise GenerationChoiceOutcomeBridgeError(f"{where} 字段不精确")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise GenerationChoiceOutcomeBridgeError(f"{where} 不是安全 POSIX 路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise GenerationChoiceOutcomeBridgeError(f"{where} 不是 SHA-256")
    return text


def _zero(value: Any, *, where: str) -> int:
    assert_int(value, _where=where)
    if type(value) is not int or value != 0:
        raise GenerationChoiceOutcomeBridgeError(f"{where} 必须为严格整数 0")
    return value


@dataclass(frozen=True)
class GenerationVerifierLayerRoute:
    """声明一个 verifier 有权判断的 choice layer，不重解释 verdict。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    choice_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("GG-02 route dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("GG-02 route verifier 必须是 ProtocolKey")
        if (not isinstance(self.choice_kinds, tuple) or not self.choice_kinds
                or any(item not in CHOICE_KINDS for item in self.choice_kinds)):
            raise GenerationChoiceOutcomeBridgeError("GG-02 route choice layers 非法")
        normalized = tuple(sorted(set(self.choice_kinds), key=_choice_order))
        if normalized != self.choice_kinds:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 route choice layers 必须按层排序且去重")
        if len(normalized) == len(CHOICE_KINDS):
            raise GenerationChoiceOutcomeBridgeError(
                "单个 verifier 不得声明整句五层广播权限")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *_pack(self.dimension.stable_key()),
            *_pack(self.verifier.stable_key()),
            len(self.choice_kinds),
            *(_choice_order(item) for item in self.choice_kinds),
        )


@dataclass(frozen=True)
class GenerationChoiceUseAttribution:
    """一次实际 choice 到 exact Core/Memory Use 和 verifier claim 的链接。"""

    choice: GenerationChoiceHypothesis
    use: GenerationChoiceUseRef
    query_key: LosslessIntegerKey
    generation_key: LosslessIntegerKey
    verification_claim_keys: tuple[LosslessIntegerKey, ...]
    source: SourceRef
    scope: ScopeIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("GG-02 attribution choice 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("GG-02 attribution use 类型错误")
        if not isinstance(self.query_key, LosslessIntegerKey):
            raise TypeError("GG-02 attribution query key 类型错误")
        if not isinstance(self.generation_key, LosslessIntegerKey):
            raise TypeError("GG-02 attribution generation key 类型错误")
        if (not isinstance(self.verification_claim_keys, tuple)
                or not self.verification_claim_keys
                or any(not isinstance(item, LosslessIntegerKey)
                       for item in self.verification_claim_keys)):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 attribution claim keys 非法或为空")
        claims = tuple(sorted(
            set(self.verification_claim_keys), key=lambda item: item.components))
        if claims != self.verification_claim_keys:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 attribution claim keys 必须排序且去重")
        if not isinstance(self.source, SourceRef):
            raise TypeError("GG-02 attribution source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("GG-02 attribution scope 类型错误")
        if (self.choice.authorized_scope != self.scope
                or self.use.scope != self.scope):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 choice/use attribution scope 漂移")
        if self.use not in self.choice.exact_uses:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 attribution use 不是 choice 声明的 exact Use")
        if (self.source.owner != self.scope.owner
                or self.source.versions != self.scope.versions):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 attribution source 越过 owner/version")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *_pack(self.choice.stable_key()),
            *_pack(self.use.stable_key()),
            *_pack(self.query_key.components),
            *_pack(self.generation_key.components),
            len(self.verification_claim_keys),
        ]
        for item in self.verification_claim_keys:
            values.extend(_pack(item.components))
        values.extend(_pack(self.source.stable_key()))
        values.extend(_pack(self.scope.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class GenerationChoiceEpisodeAttribution:
    """同一 query 中五层 choice 与 exact Use 的完整实际选择链。"""

    context_key: LosslessIntegerKey
    query_key: LosslessIntegerKey
    generation_key: LosslessIntegerKey
    source: SourceRef
    scope: ScopeIdentity
    choices: tuple[GenerationChoiceUseAttribution, ...]

    def __post_init__(self) -> None:
        for name in ("context_key", "query_key", "generation_key"):
            if not isinstance(getattr(self, name), LosslessIntegerKey):
                raise TypeError(f"GG-02 episode {name} 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("GG-02 episode source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("GG-02 episode scope 类型错误")
        if (not isinstance(self.choices, tuple)
                or any(not isinstance(item, GenerationChoiceUseAttribution)
                       for item in self.choices)):
            raise TypeError("GG-02 episode choices 类型错误")
        normalized = tuple(sorted(
            self.choices, key=lambda item: _choice_order(item.choice.choice_kind)))
        object.__setattr__(self, "choices", normalized)
        if tuple(item.choice.choice_kind for item in normalized) != CHOICE_KINDS:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 episode 未逐层覆盖五类实际 choice")
        if len({item.choice.candidate for item in normalized}) != len(normalized):
            raise GenerationChoiceOutcomeBridgeError("GG-02 choice candidate 重复")
        if len({item.use.stable_key() for item in normalized}) != len(normalized):
            raise GenerationChoiceOutcomeBridgeError("GG-02 exact Use 被跨层复用")
        claim_keys = tuple(
            key.components
            for item in normalized
            for key in item.verification_claim_keys)
        if len(set(claim_keys)) != len(claim_keys):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 verifier claim 被跨层重复归因")
        for item in normalized:
            if (item.query_key != self.query_key
                    or item.generation_key != self.generation_key
                    or item.source != self.source
                    or item.scope != self.scope):
                raise GenerationChoiceOutcomeBridgeError(
                    "GG-02 五层 choice 未绑定同一 query/generation/source/scope")
            if LosslessIntegerKey(item.choice.condition.context.stable_key()) != (
                    self.context_key):
                raise GenerationChoiceOutcomeBridgeError(
                    "GG-02 choice context 与 episode context 漂移")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *_pack(self.context_key.components),
            *_pack(self.query_key.components),
            *_pack(self.generation_key.components),
            *_pack(self.source.stable_key()),
            *_pack(self.scope.stable_key()),
            len(self.choices),
        ]
        for item in self.choices:
            values.extend(_pack(item.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class GenerationChoiceLayerOutcome:
    """一个 verifier 只对声明 layer 的 exact choice/Use 产生的结果。"""

    choice_candidate_key: LosslessIntegerKey
    choice_kind: str
    use: GenerationChoiceUseRef
    dimension: ProtocolKey
    verifier: ProtocolKey
    applicability: int
    verdict: int
    detail: LosslessIntegerKey
    assessment_ready: int
    source: SourceRef | None
    scope: ScopeIdentity | None

    def __post_init__(self) -> None:
        if not isinstance(self.choice_candidate_key, LosslessIntegerKey):
            raise TypeError("GG-02 outcome choice key 类型错误")
        _choice_order(self.choice_kind)
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("GG-02 outcome use 类型错误")
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("GG-02 outcome dimension 类型错误")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("GG-02 outcome verifier 类型错误")
        assert_int(self.applicability, self.verdict, self.assessment_ready,
                   _where="GG-02 layer outcome")
        if self.assessment_ready not in (0, 1):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 outcome assessment_ready 必须为 0/1")
        if not isinstance(self.detail, LosslessIntegerKey):
            raise TypeError("GG-02 outcome detail 类型错误")
        if self.assessment_ready:
            if (self.applicability != APPLICABILITY_APPLICABLE
                    or self.source is None or self.scope is None):
                raise GenerationChoiceOutcomeBridgeError(
                    "可 assessment outcome 缺 applicable/source/scope")
            if self.scope != self.use.scope:
                raise GenerationChoiceOutcomeBridgeError(
                    "GG-02 outcome scope 与 exact Use 漂移")
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("GG-02 outcome source 类型错误")
        if self.scope is not None and not isinstance(self.scope, ScopeIdentity):
            raise TypeError("GG-02 outcome scope 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        return (
            *_pack(self.choice_candidate_key.components),
            _choice_order(self.choice_kind),
            *_pack(self.use.stable_key()),
            *_pack(self.dimension.stable_key()),
            *_pack(self.verifier.stable_key()),
            self.applicability,
            self.verdict,
            *_pack(self.detail.components),
            self.assessment_ready,
            *_pack(() if self.source is None else self.source.stable_key()),
            *_pack(() if self.scope is None else self.scope.stable_key()),
        )


@dataclass(frozen=True)
class GenerationLayeredOutcomeReport:
    """五层分维 outcome 落账计划；它本身不更新任何候选 assessment。"""

    episode: GenerationChoiceEpisodeAttribution
    outcomes: tuple[GenerationChoiceLayerOutcome, ...]
    assessment_consumer_status: str
    host_learning_write_count: int
    teacher_call_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode, GenerationChoiceEpisodeAttribution):
            raise TypeError("GG-02 outcome report episode 类型错误")
        if (not isinstance(self.outcomes, tuple) or not self.outcomes
                or any(not isinstance(item, GenerationChoiceLayerOutcome)
                       for item in self.outcomes)):
            raise TypeError("GG-02 outcome report outcomes 类型错误")
        normalized = tuple(sorted(
            self.outcomes,
            key=lambda item: (
                _choice_order(item.choice_kind), item.dimension.stable_key(),
                item.verifier.stable_key())))
        object.__setattr__(self, "outcomes", normalized)
        keys = tuple((
            item.choice_candidate_key, item.dimension, item.verifier)
            for item in normalized)
        if len(set(keys)) != len(keys):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 同 choice/dimension/verifier outcome 重复")
        expected = {
            LosslessIntegerKey(item.choice.candidate.stable_key())
            for item in self.episode.choices
        }
        if {item.choice_candidate_key for item in normalized} != expected:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 outcome 未覆盖全部五层 choice")
        if self.assessment_consumer_status != "REQUIRED_NOT_CONNECTED":
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 不得冒充 assessment consumer 已接通")
        _zero(self.host_learning_write_count, where="GG-02 host learning writes")
        _zero(self.teacher_call_count, where="GG-02 teacher calls")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *_pack(self.episode.stable_key()),
            len(self.outcomes),
        ]
        for item in self.outcomes:
            values.extend(_pack(item.stable_key()))
        values.extend((self.host_learning_write_count, self.teacher_call_count))
        return tuple(values)


@dataclass(frozen=True)
class GenerationChoiceAssessmentInput:
    """供未来 assessment consumer 使用的单层输入，不含更新动作。"""

    choice_candidate_key: LosslessIntegerKey
    choice_kind: str
    use: GenerationChoiceUseRef
    outcomes: tuple[GenerationChoiceLayerOutcome, ...]
    assessment_state: str

    def __post_init__(self) -> None:
        if not isinstance(self.choice_candidate_key, LosslessIntegerKey):
            raise TypeError("GG-02 assessment choice key 类型错误")
        _choice_order(self.choice_kind)
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("GG-02 assessment use 类型错误")
        if (not isinstance(self.outcomes, tuple) or not self.outcomes
                or any(not isinstance(item, GenerationChoiceLayerOutcome)
                       for item in self.outcomes)):
            raise TypeError("GG-02 assessment outcomes 类型错误")
        if any(
                item.choice_candidate_key != self.choice_candidate_key
                or item.choice_kind != self.choice_kind
                or item.use != self.use
                for item in self.outcomes):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 assessment input 混入其他 choice/use")
        if self.assessment_state not in ASSESSMENT_STATES:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 assessment state 未注册")
        if (self.assessment_state == "READY"
                and not any(item.assessment_ready for item in self.outcomes)):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 READY 输入没有可消费 outcome")

    def stable_key(self) -> tuple[int, ...]:
        values = [
            *_pack(self.choice_candidate_key.components),
            _choice_order(self.choice_kind),
            *_pack(self.use.stable_key()),
            ASSESSMENT_STATES.index(self.assessment_state),
            len(self.outcomes),
        ]
        for item in self.outcomes:
            values.extend(_pack(item.stable_key()))
        return tuple(values)


@dataclass(frozen=True)
class GenerationChoiceAssessmentReport:
    """五层 assessment 输入路由报告，只证明可归因而不执行学习。"""

    inputs: tuple[GenerationChoiceAssessmentInput, ...]
    assessment_updates_executed: int
    host_learning_write_count: int

    def __post_init__(self) -> None:
        if (not isinstance(self.inputs, tuple)
                or any(not isinstance(item, GenerationChoiceAssessmentInput)
                       for item in self.inputs)):
            raise TypeError("GG-02 assessment report inputs 类型错误")
        normalized = tuple(sorted(
            self.inputs, key=lambda item: _choice_order(item.choice_kind)))
        object.__setattr__(self, "inputs", normalized)
        if tuple(item.choice_kind for item in normalized) != CHOICE_KINDS:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 assessment report 未覆盖五层")
        _zero(self.assessment_updates_executed,
              where="GG-02 assessment updates executed")
        _zero(self.host_learning_write_count,
              where="GG-02 assessment host learning writes")


class GenerationChoiceOutcomeBridge:
    """把只读 G-04 report 精确归因到五层 actual choice/Use。"""

    def __init__(self, routes: tuple[GenerationVerifierLayerRoute, ...]) -> None:
        if (not isinstance(routes, tuple) or not routes
                or any(not isinstance(item, GenerationVerifierLayerRoute)
                       for item in routes)):
            raise TypeError("GG-02 routes 类型错误或为空")
        normalized = tuple(sorted(routes, key=GenerationVerifierLayerRoute.stable_key))
        keys = tuple((item.dimension, item.verifier) for item in normalized)
        if len(set(keys)) != len(keys):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 route dimension/verifier 重复")
        covered = {kind for item in normalized for kind in item.choice_kinds}
        if covered != set(CHOICE_KINDS):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 routes 未声明全部五层 verifier 能力")
        self.routes = normalized
        self._routes = dict(zip(keys, normalized, strict=True))

    def compile(
            self,
            episode: GenerationChoiceEpisodeAttribution,
            report: VerificationReport,
            ) -> GenerationLayeredOutcomeReport:
        """零写编译分层 outcome；非适用层只形成 NE，不形成奖惩。"""
        if not isinstance(episode, GenerationChoiceEpisodeAttribution):
            raise TypeError("GG-02 episode 类型错误")
        if not isinstance(report, VerificationReport) or not report.read_only:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 只接受只读 VerificationReport")
        if any(item.proposed_effects or item.committed_effects
               for item in report.results):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 verifier report 不得携带 effect")
        attributions = episode.choices
        all_claims = {
            key.components
            for item in attributions
            for key in item.verification_claim_keys
        }
        outcomes = []
        seen_routes = set()
        for result in report.results:
            route_key = (result.dimension, result.verifier)
            route = self._routes.get(route_key)
            if route is None:
                raise GenerationChoiceOutcomeBridgeError(
                    "GG-02 report 含未登记 dimension/verifier")
            seen_routes.add(route_key)
            allowed = set(route.choice_kinds)
            if result.applicability == APPLICABILITY_APPLICABLE:
                if result.operational_failure is not None:
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 applicable result 不接受 operational failure")
                if result.source != episode.source or result.scope != episode.scope:
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 claimed result 未绑定同一 query source/scope")
                foreign = set(result.claim_keys) - all_claims
                if foreign:
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 report claim 不属于当前 actual choice chain")
                targeted = tuple(
                    item for item in attributions
                    if set(result.claim_keys) & {
                        key.components for key in item.verification_claim_keys}
                )
                if not targeted:
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 applicable result 没有 exact choice claim")
                if any(item.choice.choice_kind not in allowed for item in targeted):
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 verifier claim 越过声明 choice layer")
                if len(targeted) == len(CHOICE_KINDS):
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 禁止整句 outcome 广播到五层")
                targets = targeted
                ready = 1
            else:
                if result.claim_keys:
                    raise GenerationChoiceOutcomeBridgeError(
                        "GG-02 非 applicable result 不得携带 claim")
                targets = tuple(
                    item for item in attributions
                    if item.choice.choice_kind in allowed)
                ready = 0
            detail = LosslessIntegerKey(result.detail or (0,))
            for item in targets:
                outcomes.append(GenerationChoiceLayerOutcome(
                    LosslessIntegerKey(item.choice.candidate.stable_key()),
                    item.choice.choice_kind,
                    item.use,
                    result.dimension,
                    result.verifier,
                    result.applicability,
                    result.verdict,
                    detail,
                    ready,
                    result.source,
                    result.scope,
                ))
        if seen_routes != set(self._routes):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 report 未逐 route 提供 result/NE")
        return GenerationLayeredOutcomeReport(
            episode, tuple(outcomes), "REQUIRED_NOT_CONNECTED", 0, 0)


def build_assessment_inputs(
        report: GenerationLayeredOutcomeReport,
        *, disabled_choice_kinds: tuple[str, ...] = (),
        ) -> GenerationChoiceAssessmentReport:
    """按层形成未来 assessment 输入；关闭一层不改变其他层对象。"""
    if not isinstance(report, GenerationLayeredOutcomeReport):
        raise TypeError("GG-02 layered outcome report 类型错误")
    if (not isinstance(disabled_choice_kinds, tuple)
            or any(item not in CHOICE_KINDS for item in disabled_choice_kinds)
            or len(set(disabled_choice_kinds)) != len(disabled_choice_kinds)):
        raise GenerationChoiceOutcomeBridgeError("GG-02 disabled layers 非法")
    disabled = set(disabled_choice_kinds)
    inputs = []
    for attribution in report.episode.choices:
        key = LosslessIntegerKey(attribution.choice.candidate.stable_key())
        outcomes = tuple(
            item for item in report.outcomes
            if item.choice_candidate_key == key)
        if attribution.choice.choice_kind in disabled:
            state = "NE_LAYER_DISABLED"
        elif any(item.assessment_ready for item in outcomes):
            state = "READY"
        else:
            state = "NE_VERIFIER"
        inputs.append(GenerationChoiceAssessmentInput(
            key, attribution.choice.choice_kind, attribution.use,
            outcomes, state))
    return GenerationChoiceAssessmentReport(tuple(inputs), 0, 0)


@dataclass(frozen=True)
class GG02BridgeManifest:
    """不可覆盖的 GG-02 分层 Use/outcome bridge 冻结证据。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_keys: tuple[str, ...]
    prerequisite_paths: tuple[str, ...]
    prerequisite_sha256: tuple[str, ...]
    contract_type_keys: tuple[str, ...]
    choice_kind_keys: tuple[str, ...]
    invariant_keys: tuple[str, ...]
    reused_component_refs: tuple[str, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    assessment_consumer_status: str
    runtime_status: str
    results_observed: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise GenerationChoiceOutcomeBridgeError("GG-02 format version 非法")
        _text(self.artifact_version, where="GG-02 artifact version")
        if self.artifact_status != "BRIDGE_FROZEN":
            raise GenerationChoiceOutcomeBridgeError("GG-02 artifact status 非法")
        if self.task_keys != ("GG-02", "LC-13"):
            raise GenerationChoiceOutcomeBridgeError("GG-02 task keys 非法")
        if (not isinstance(self.prerequisite_paths, tuple)
                or len(self.prerequisite_paths) != 2):
            raise GenerationChoiceOutcomeBridgeError("GG-02 prerequisites 未列全")
        paths = tuple(_relative_path(item, where="GG-02 prerequisite path")
                      for item in self.prerequisite_paths)
        if paths != tuple(sorted(set(paths))):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 prerequisite paths 必须排序且去重")
        if (not isinstance(self.prerequisite_sha256, tuple)
                or len(self.prerequisite_sha256) != len(paths)):
            raise GenerationChoiceOutcomeBridgeError("GG-02 prerequisite hash 未列全")
        tuple(_sha256(item, where="GG-02 prerequisite hash")
              for item in self.prerequisite_sha256)
        for actual, expected, label in (
                (self.contract_type_keys, GG02_CONTRACT_TYPES, "contract types"),
                (self.choice_kind_keys, CHOICE_KINDS, "choice kinds"),
                (self.invariant_keys, GG02_INVARIANTS, "invariants"),
                (self.verifier_dimensions, GG02_VERIFIER_DIMENSIONS,
                 "verifier dimensions"),
                (self.verifier_ne_conditions, GG02_NE_CONDITIONS,
                 "verifier NE conditions")):
            if actual != expected:
                raise GenerationChoiceOutcomeBridgeError(
                    f"GG-02 {label} 未列全")
        _text_tuple(self.reused_component_refs, where="GG-02 reused refs")
        if self.assessment_consumer_status != "REQUIRED_NOT_CONNECTED":
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 不得冒充 assessment consumer 已接通")
        if self.runtime_status != "NOT_CONNECTED":
            raise GenerationChoiceOutcomeBridgeError("GG-02 不得冒充 runtime 已接通")
        _zero(self.results_observed, where="GG-02 results observed")
        state = self.execution_state.to_value()
        if tuple(state) != EXECUTION_STATE_KEYS or any(state.values()):
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 execution state 必须字段精确且全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_GG02_GENERATION_CHOICE_OUTCOME_BRIDGE",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "assessment_consumer_status": self.assessment_consumer_status,
            "choice_kind_keys": list(self.choice_kind_keys),
            "contract_type_keys": list(self.contract_type_keys),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "invariant_keys": list(self.invariant_keys),
            "prerequisite_paths": list(self.prerequisite_paths),
            "prerequisite_sha256": list(self.prerequisite_sha256),
            "results_observed": self.results_observed,
            "reused_component_refs": list(self.reused_component_refs),
            "runtime_status": self.runtime_status,
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GG02BridgeManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "assessment_consumer_status", "choice_kind_keys",
            "contract_type_keys", "execution_state", "format_version",
            "invariant_keys", "prerequisite_paths", "prerequisite_sha256",
            "results_observed", "reused_component_refs", "runtime_status",
            "task_keys", "verifier_dimensions", "verifier_ne_conditions",
        }, where="GG02BridgeManifest")
        if raw["artifact_kind"] != "PH2_GG02_GENERATION_CHOICE_OUTCOME_BRIDGE":
            raise GenerationChoiceOutcomeBridgeError("GG-02 artifact kind 非法")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            tuple(str(item) for item in raw["prerequisite_paths"]),
            tuple(str(item) for item in raw["prerequisite_sha256"]),
            tuple(str(item) for item in raw["contract_type_keys"]),
            tuple(str(item) for item in raw["choice_kind_keys"]),
            tuple(str(item) for item in raw["invariant_keys"]),
            tuple(str(item) for item in raw["reused_component_refs"]),
            tuple(str(item) for item in raw["verifier_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            str(raw["assessment_consumer_status"]),
            str(raw["runtime_status"]), raw["results_observed"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())


def zero_execution_state() -> CanonicalJsonObject:
    return CanonicalJsonObject.from_value({key: 0 for key in EXECUTION_STATE_KEYS})


def build_gg02_bridge_manifest(
        *, gg01_sha256: str, baseline_sha256: str,
        ) -> GG02BridgeManifest:
    paths = (
        "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
        "data/ph2/manifests/language_capability_baseline_v25.json",
    )
    return GG02BridgeManifest(
        FORMAT_VERSION,
        "GG-02-layered-use-outcome-bridge-v1",
        "BRIDGE_FROZEN",
        ("GG-02", "LC-13"),
        paths,
        (gg01_sha256, baseline_sha256),
        GG02_CONTRACT_TYPES,
        CHOICE_KINDS,
        GG02_INVARIANTS,
        tuple(sorted((
            "src/pure_integer_ai/cognition/shared/generation_surface.py",
            "src/pure_integer_ai/cognition/shared/memory_event.py",
            "src/pure_integer_ai/cognition/shared/memory_generation.py",
            "src/pure_integer_ai/cognition/shared/relation_use.py",
            "src/pure_integer_ai/experiments/generation_verification_runtime.py",
            "src/pure_integer_ai/experiments/language_generation_connector_stage4.py",
            "src/pure_integer_ai/experiments/memory_generation_outcome_runtime.py",
            "src/pure_integer_ai/experiments/memory_use_runtime.py",
            "src/pure_integer_ai/experiments/ph2_generation_choice_contract.py",
        ))),
        GG02_VERIFIER_DIMENSIONS,
        GG02_NE_CONDITIONS,
        "REQUIRED_NOT_CONNECTED",
        "NOT_CONNECTED",
        0,
        zero_execution_state(),
    )


def write_gg02_bridge_manifest(
        manifest: GG02BridgeManifest, path: str | Path) -> Path:
    if not isinstance(manifest, GG02BridgeManifest):
        raise GenerationChoiceOutcomeBridgeError("GG-02 manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise GenerationChoiceOutcomeBridgeError(
                "GG-02 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise GenerationChoiceOutcomeBridgeError("GG-02 manifest 无法发布") from error
    return target


def read_gg02_bridge_manifest(path: str | Path) -> GG02BridgeManifest:
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise GenerationChoiceOutcomeBridgeError("GG-02 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = GG02BridgeManifest.from_dict(value)
    except GenerationChoiceOutcomeBridgeError:
        raise
    except Exception as error:
        raise GenerationChoiceOutcomeBridgeError("GG-02 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise GenerationChoiceOutcomeBridgeError("GG-02 manifest 非规范字节")
    return manifest


__all__ = [
    "ASSESSMENT_STATES",
    "GG02BridgeManifest",
    "GG02_CONTRACT_TYPES",
    "GG02_INVARIANTS",
    "GG02_NE_CONDITIONS",
    "GG02_VERIFIER_DIMENSIONS",
    "GenerationChoiceAssessmentInput",
    "GenerationChoiceAssessmentReport",
    "GenerationChoiceEpisodeAttribution",
    "GenerationChoiceLayerOutcome",
    "GenerationChoiceOutcomeBridge",
    "GenerationChoiceOutcomeBridgeError",
    "GenerationChoiceUseAttribution",
    "GenerationLayeredOutcomeReport",
    "GenerationVerifierLayerRoute",
    "build_assessment_inputs",
    "build_gg02_bridge_manifest",
    "read_gg02_bridge_manifest",
    "write_gg02_bridge_manifest",
    "zero_execution_state",
]
