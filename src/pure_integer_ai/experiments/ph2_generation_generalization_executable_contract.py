"""GG-03 实际生成、反解析和六路独立 verifier 的公开纯合同。"""
from __future__ import annotations

from dataclasses import dataclass, field

from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
)
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VerificationReport,
    VerificationResult,
)


GG03_EXECUTABLE_STATUSES = (
    "FAIL_INDEPENDENT_LAYER_CONJUNCTION",
    "NE_INDEPENDENT_LAYER_INPUT_INDETERMINATE",
    "NE_INDEPENDENT_LAYER_INPUT_MISSING",
    "NE_INDEPENDENT_UNDERSTANDING_READBACK",
    "PASS_EXECUTABLE_LAYER_CONJUNCTION",
)


# object-model: exception
class GenerationGeneralizationExecutableContractError(ValueError):
    """实际生成、readback 或独立 verifier 违反 E-00 边界。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 key。"""
    if not isinstance(value, tuple) or not value:
        raise GenerationGeneralizationExecutableContractError(
            f"{where} 必须为非空 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise GenerationGeneralizationExecutableContractError(
            f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """把开放整数键编码为有长度边界的片段。"""
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    """把可选运行失败文本编码为确定性整数片段。"""
    raw = value.encode("utf-8")
    return len(raw), *raw


def _result_key(result: VerificationResult) -> tuple[int, ...]:
    """返回不含临时 artifact 的完整只读 verifier 结果键。"""
    values = [
        *_pack(result.dimension.stable_key()),
        *_pack(result.verifier.stable_key()),
        result.applicability,
        result.verdict,
        len(result.claim_keys),
    ]
    for claim in result.claim_keys:
        values.extend(_pack(claim))
    values.append(len(result.proposed_effects))
    for effect in result.proposed_effects:
        values.extend(_pack(effect.stable_key()))
    values.append(len(result.committed_effects))
    for effect in result.committed_effects:
        values.extend(_pack(effect.stable_key()))
    values.extend(_pack(result.detail))
    for identity in (result.source, result.scope):
        values.append(0 if identity is None else 1)
        if identity is not None:
            values.extend(_pack(identity.stable_key()))
    values.append(0 if result.operational_failure is None else 1)
    if result.operational_failure is not None:
        values.extend(_text_key(result.operational_failure))
    return tuple(values)


def _report_key(report: VerificationReport) -> tuple[int, ...]:
    """返回只读 report 的确定性排序和内容键。"""
    values = [int(report.read_only), len(report.results)]
    for result in report.results:
        values.extend(_pack(_result_key(result)))
    return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationIndependentVerification:
    """一项独立要求及其 actual input identity 和只读结果。"""

    requirement: str
    input_key: LosslessIntegerKey
    result: VerificationResult

    def __post_init__(self) -> None:
        if self.requirement not in INDEPENDENT_VERIFIER_REQUIREMENTS:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verifier requirement 未注册")
        if not isinstance(self.input_key, LosslessIntegerKey):
            raise TypeError("GG-03 executable verifier input key 类型错误")
        if not isinstance(self.result, VerificationResult):
            raise TypeError("GG-03 executable verifier result 类型错误")
        if self.result.committed_effects:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verifier 不得提交 effect")
        if (self.result.applicability == APPLICABILITY_APPLICABLE
                and self.input_key.components not in self.result.claim_keys):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verifier input 未绑定 applicable claim")

    def stable_key(self) -> tuple[int, ...]:
        """返回要求 ordinal、actual input 和完整 verifier result。"""
        return (
            INDEPENDENT_VERIFIER_REQUIREMENTS.index(self.requirement) + 1,
            *_pack(self.input_key.components),
            *_pack(_result_key(self.result)),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationExecutableEvidence:
    """一次 actual choice/Use、surface、readback 与六路 verifier 证据。"""

    choice: GenerationChoiceHypothesis
    use: GenerationChoiceUseRef
    execution: TypedGenerationExecution
    parse_request: GenerationSurfaceParseRequest
    postcheck: GenerationPostcheckRun
    verification_reports: tuple[VerificationReport, ...]
    verifications: tuple[GenerationGeneralizationIndependentVerification, ...]
    trace: tuple[int, ...]
    _stable_key_cache: tuple[int, ...] = field(
        init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        """核验 actual caller、readback、report 归属和六路输入隔离。"""
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("GG-03 executable choice 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("GG-03 executable use 类型错误")
        if self.use not in self.choice.exact_uses:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable Use 未回填 exact choice")
        if not isinstance(self.execution, TypedGenerationExecution):
            raise TypeError("GG-03 executable generation 类型错误")
        if not self.execution.complete:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable generation 未形成 actual surface")
        if not isinstance(self.parse_request, GenerationSurfaceParseRequest):
            raise TypeError("GG-03 executable parse request 类型错误")
        expected_parse = GenerationSurfaceParseRequest.from_execution(
            self.execution)
        if self.parse_request != expected_parse:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable parse request 未由同次 execution 派生")
        if not isinstance(self.postcheck, GenerationPostcheckRun):
            raise TypeError("GG-03 executable postcheck 类型错误")
        if self.postcheck.request.execution != self.execution:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable postcheck 替换了 generation")
        goal = self.execution.plan.request.goal
        if (self.use.scope != self.choice.authorized_scope
                or goal.scope != self.choice.authorized_scope
                or self.parse_request.scope != self.choice.authorized_scope
                or goal.source != self.parse_request.source
                or goal.source not in self.choice.forming_sources):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable choice/generation/parser 归属漂移")
        parsed = self.postcheck.parsed
        if parsed.succeeded:
            observation = parsed.observation
            assert observation is not None
            if (observation.parse_request_key != self.parse_request.stable_key()
                    or observation.source != goal.source
                    or observation.scope != goal.scope
                    or observation.representations
                    != self.execution.representations):
                raise GenerationGeneralizationExecutableContractError(
                    "GG-03 executable parser readback 未绑定 actual output")
        if (not isinstance(self.verification_reports, tuple)
                or not self.verification_reports
                or any(not isinstance(item, VerificationReport)
                       for item in self.verification_reports)):
            raise TypeError("GG-03 executable verification reports 类型错误")
        if any(not item.read_only for item in self.verification_reports):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verification reports 必须全部只读")
        reports = tuple(sorted(
            self.verification_reports, key=_report_key))
        if len({_report_key(item) for item in reports}) != len(reports):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verification report 重复")
        if self.postcheck.report not in reports:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable reports 缺 actual G-04 report")
        object.__setattr__(self, "verification_reports", reports)
        if (not isinstance(self.verifications, tuple)
                or any(not isinstance(
                    item, GenerationGeneralizationIndependentVerification)
                    for item in self.verifications)):
            raise TypeError("GG-03 executable verifications 类型错误")
        expected_order = tuple(
            item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
            if any(value.requirement == item for value in self.verifications))
        if tuple(item.requirement for item in self.verifications) != expected_order:
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable verifier requirement 重复或顺序漂移")
        report_results = tuple(
            result for report in reports for result in report.results)
        result_keys = tuple(_result_key(item) for item in report_results)
        if len(set(result_keys)) != len(result_keys):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable reports 复用了 verifier result")
        inputs = tuple(item.input_key for item in self.verifications)
        bindings = tuple(
            (item.result.dimension, item.result.verifier)
            for item in self.verifications)
        if len(set(inputs)) != len(inputs) or len(set(bindings)) != len(bindings):
            raise GenerationGeneralizationExecutableContractError(
                "GG-03 executable 六路 verifier input 或 route 发生广播")
        for verification in self.verifications:
            if _result_key(verification.result) not in result_keys:
                raise GenerationGeneralizationExecutableContractError(
                    "GG-03 executable verifier result 不属于只读 report")
            result = verification.result
            if (result.source != goal.source or result.scope != goal.scope):
                raise GenerationGeneralizationExecutableContractError(
                    "GG-03 executable verifier result 跨 source/scope")
        _strict_key(self.trace, where="GG-03 executable evidence trace")
        object.__setattr__(self, "_stable_key_cache", self._build_stable_key())

    @property
    def runtime_status(self) -> str:
        """按 hard conjunction 返回 executable 层 PASS/FAIL/NE。"""
        if any(item.result.verdict == VERDICT_REFUTE
               and item.result.applicability == APPLICABILITY_APPLICABLE
               for item in self.verifications):
            return "FAIL_INDEPENDENT_LAYER_CONJUNCTION"
        if not self.postcheck.parsed.succeeded:
            return "NE_INDEPENDENT_UNDERSTANDING_READBACK"
        if len(self.verifications) != len(INDEPENDENT_VERIFIER_REQUIREMENTS):
            return "NE_INDEPENDENT_LAYER_INPUT_MISSING"
        if any(
                item.result.operational_failure is not None
                or item.result.applicability != APPLICABILITY_APPLICABLE
                or item.result.verdict != VERDICT_SUPPORT
                for item in self.verifications):
            return "NE_INDEPENDENT_LAYER_INPUT_INDETERMINATE"
        return "PASS_EXECUTABLE_LAYER_CONJUNCTION"

    @property
    def ready_for_label_comparison(self) -> int:
        """只在六路 actual executable conjunction 全 support 时返回一。"""
        return int(self.runtime_status == "PASS_EXECUTABLE_LAYER_CONJUNCTION")

    def stable_key(self) -> tuple[int, ...]:
        """返回 choice/Use、actual output、readback 和独立结果内容引用。"""
        if not self._stable_key_cache:
            raise RuntimeError("GG-03 executable stable key 尚未构造")
        return self._stable_key_cache

    def _build_stable_key(self) -> tuple[int, ...]:
        """在冻结构造完成时计算一次有界内容引用键。"""
        values = [
            *_pack(integer_tuple_fingerprint(
                self.choice.stable_key(),
                domain="gg03.executable.choice.v1",
            )),
            *_pack(integer_tuple_fingerprint(
                self.use.stable_key(),
                domain="gg03.executable.use.v1",
            )),
            *_pack(integer_tuple_fingerprint(
                self.execution.stable_key(),
                domain="gg03.executable.execution.v1",
            )),
            *_pack(integer_tuple_fingerprint(
                self.parse_request.stable_key(),
                domain="gg03.executable.parse.request.v1",
            )),
            *_pack(integer_tuple_fingerprint(
                self.postcheck.stable_key(),
                domain="gg03.executable.postcheck.v1",
            )),
            len(self.verification_reports),
        ]
        for report in self.verification_reports:
            values.extend(_pack(integer_tuple_fingerprint(
                _report_key(report),
                domain="gg03.executable.verification.report.v1",
            )))
        values.append(len(self.verifications))
        for verification in self.verifications:
            values.extend(_pack(verification.stable_key()))
        values.extend(_pack(self.trace))
        values.append(GG03_EXECUTABLE_STATUSES.index(self.runtime_status) + 1)
        return tuple(values)


__all__ = [
    "GG03_EXECUTABLE_STATUSES",
    "GenerationGeneralizationExecutableContractError",
    "GenerationGeneralizationExecutableEvidence",
    "GenerationGeneralizationIndependentVerification",
]
