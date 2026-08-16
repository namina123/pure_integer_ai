"""generation candidate alias 的纯值合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_PROPOSITION,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    RULE_CLAIM,
    RULE_LITERAL,
    RULE_REFERENCE,
    RULE_RESPONSE_ACT,
)


_RULE_IDS = {
    RULE_CLAIM: 1,
    RULE_LITERAL: 2,
    RULE_REFERENCE: 3,
    RULE_RESPONSE_ACT: 4,
}


# object-model: exception
class GenerationCandidateAliasRuntimeError(RuntimeError):
    """pack 规则、运行输入、R-01 manifest 或 owner 恢复不一致。"""


def normalize_forming_evidence_keys(
        values: tuple[tuple[int, ...], ...], *, where: str,
        ) -> tuple[tuple[int, ...], ...]:
    """规范化互异形成证据键，不允许运行期补空来源。"""
    if not isinstance(values, tuple) or not values:
        raise GenerationCandidateAliasRuntimeError(f"{where} 缺少形成证据")
    checked = []
    for value in values:
        if (not isinstance(value, tuple) or not value
                or any(type(item) is not int for item in value)):
            raise GenerationCandidateAliasRuntimeError(
                f"{where} 必须是非空严格整数 tuple")
        checked.append(value)
    result = tuple(sorted(set(checked)))
    if len(result) != len(checked):
        raise GenerationCandidateAliasRuntimeError(f"{where} 形成证据重复")
    return result


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(value), *value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationCandidateRealizationBinding:
    """一条 input/grammar 对象到 Unicode Representation 的 realizes 请求。"""

    origin: ObjectIdentity
    representation: ObjectIdentity
    rule: str
    forming_evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.origin, ObjectIdentity):
            raise TypeError("candidate realization origin 类型错误")
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise TypeError("candidate realization representation 类型错误")
        if self.rule not in _RULE_IDS:
            raise GenerationCandidateAliasRuntimeError(
                "candidate realization rule 未注册")
        object.__setattr__(
            self,
            "forming_evidence_keys",
            normalize_forming_evidence_keys(
                self.forming_evidence_keys,
                where="realization forming evidence",
            ),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 origin、Representation、规则和全部形成证据。"""
        result = [
            *_packed(self.origin.stable_key()),
            *_packed(self.representation.stable_key()),
            _RULE_IDS[self.rule],
            len(self.forming_evidence_keys),
        ]
        for key in self.forming_evidence_keys:
            result.extend(_packed(key))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationCandidateReferenceBinding:
    """一条 visible reference origin 到 antecedent Proposition 的 refers 请求。"""

    origin: ObjectIdentity
    target: ObjectIdentity
    forming_evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.origin, ObjectIdentity)
                or self.origin.object_kind != OBJECT_PROPOSITION
                or not isinstance(self.target, ObjectIdentity)
                or self.target.object_kind != OBJECT_PROPOSITION):
            raise GenerationCandidateAliasRuntimeError(
                "candidate reference 端点必须是 Proposition")
        object.__setattr__(
            self,
            "forming_evidence_keys",
            normalize_forming_evidence_keys(
                self.forming_evidence_keys,
                where="reference forming evidence",
            ),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回方向端点和全部形成证据。"""
        result = [
            *_packed(self.origin.stable_key()),
            *_packed(self.target.stable_key()),
            len(self.forming_evidence_keys),
        ]
        for key in self.forming_evidence_keys:
            result.extend(_packed(key))
        return tuple(result)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationCandidateAliasCourseRequest:
    """一个 run-local branch 的全部 realizes/refers 物化请求。"""

    branch: ObjectIdentity
    realizations: tuple[GenerationCandidateRealizationBinding, ...]
    references: tuple[GenerationCandidateReferenceBinding, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise GenerationCandidateAliasRuntimeError(
                "candidate alias branch 类型错误")
        if (not isinstance(self.realizations, tuple) or not self.realizations
                or any(not isinstance(
                    item, GenerationCandidateRealizationBinding)
                    for item in self.realizations)):
            raise GenerationCandidateAliasRuntimeError(
                "candidate alias realizations 不能为空")
        if (not isinstance(self.references, tuple)
                or any(not isinstance(
                    item, GenerationCandidateReferenceBinding)
                    for item in self.references)):
            raise TypeError("candidate alias references 类型错误")
        realization_routes = tuple(
            (item.origin, item.representation) for item in self.realizations)
        if len(set(realization_routes)) != len(realization_routes):
            raise GenerationCandidateAliasRuntimeError(
                "candidate alias realization 重复")
        reference_routes = tuple(
            (item.origin, item.target) for item in self.references)
        if len(set(reference_routes)) != len(reference_routes):
            raise GenerationCandidateAliasRuntimeError(
                "candidate alias reference 重复")
        object.__setattr__(self, "realizations", tuple(sorted(
            self.realizations, key=lambda item: item.stable_key())))
        object.__setattr__(self, "references", tuple(sorted(
            self.references, key=lambda item: item.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回 branch 与完整 relation 输入，供 manifest 内容锁。"""
        result = [*_packed(self.branch.stable_key()), len(self.realizations)]
        for item in self.realizations:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.references))
        for item in self.references:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


__all__ = [
    "GenerationCandidateAliasCourseRequest",
    "GenerationCandidateAliasRuntimeError",
    "GenerationCandidateRealizationBinding",
    "GenerationCandidateReferenceBinding",
    "normalize_forming_evidence_keys",
]
