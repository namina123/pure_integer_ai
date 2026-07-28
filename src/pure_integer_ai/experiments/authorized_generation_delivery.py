"""授权 claim 与同次 G-04 完整通过后才形成外部可交付 surface。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_plan import GenerationCandidate
from pure_integer_ai.cognition.shared.identity import ObjectIdentity, SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.authorized_center_runtime import (
    AuthorizedCenterState,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


DELIVERY_AUTHORIZED = "AUTHORIZED"
DELIVERY_GENERATION_INCOMPLETE = "GENERATION_INCOMPLETE"
DELIVERY_POSTCHECK_MISSING = "POSTCHECK_MISSING"
DELIVERY_POSTCHECK_FAILED = "POSTCHECK_FAILED"
DELIVERY_CLAIM_COVERAGE_MISMATCH = "CLAIM_COVERAGE_MISMATCH"
DELIVERY_CLAIM_BINDING_MISMATCH = "CLAIM_BINDING_MISMATCH"
DELIVERY_REQUIREMENT_MISMATCH = "REQUIREMENT_MISMATCH"
DELIVERY_REQUIREMENT_NOT_STRICT = "REQUIREMENT_NOT_STRICT"
DELIVERY_OBSERVATION_MISMATCH = "OBSERVATION_MISMATCH"
DELIVERY_CITATION_MISMATCH = "CITATION_MISMATCH"
_DELIVERY_STATES = {
    DELIVERY_AUTHORIZED,
    DELIVERY_GENERATION_INCOMPLETE,
    DELIVERY_POSTCHECK_MISSING,
    DELIVERY_POSTCHECK_FAILED,
    DELIVERY_CLAIM_COVERAGE_MISMATCH,
    DELIVERY_CLAIM_BINDING_MISMATCH,
    DELIVERY_REQUIREMENT_MISMATCH,
    DELIVERY_REQUIREMENT_NOT_STRICT,
    DELIVERY_OBSERVATION_MISMATCH,
    DELIVERY_CITATION_MISMATCH,
}


class AuthorizedGenerationDeliveryError(RuntimeError):
    """授权 claim、postcheck 或交付 envelope 合同不闭合。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """要求交付合同身份是非空正严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item <= 0 for item in value)):
        raise AuthorizedGenerationDeliveryError(
            f"{where} 必须是正严格整数 tuple")
    return value


def _integer_stream(
        value: tuple[int, ...], *, where: str,
        ) -> tuple[int, ...]:
    """要求内容引用流非空、仅含非负严格整数，并允许合法零位。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise AuthorizedGenerationDeliveryError(
            f"{where} 必须是非负严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长稳定键按长度分帧写入结果。"""
    result.extend((len(value), *value))


@dataclass(frozen=True, order=True)
class AuthorizedGenerationClaim:
    """一个 generation candidate 的前序授权、命题、来源和引用合同。"""

    candidate_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    source: SourceRef
    scope: ScopeIdentity
    evidence_sources: tuple[SourceRef, ...]
    authorization_receipt_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 claim 身份完整、引用来源非空且归属 source/scope 一致。"""
        _integer_stream(
            self.candidate_key, where="authorized claim candidate_key")
        _integer_stream(
            self.proposition_key, where="authorized claim proposition_key")
        if not isinstance(self.source, SourceRef):
            raise TypeError("authorized claim source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("authorized claim scope 类型错误")
        if (not isinstance(self.evidence_sources, tuple)
                or not self.evidence_sources
                or any(not isinstance(item, SourceRef)
                       for item in self.evidence_sources)):
            raise TypeError("authorized claim evidence_sources 类型错误")
        evidence_sources = tuple(sorted(
            set(self.evidence_sources), key=lambda item: item.stable_key()))
        if evidence_sources != self.evidence_sources:
            raise AuthorizedGenerationDeliveryError(
                "authorized claim evidence_sources 必须排序去重")
        _strict_key(
            self.authorization_receipt_key,
            where="authorized claim receipt key",
        )

    @classmethod
    def from_authorized_center(
            cls,
            candidate: GenerationCandidate,
            state: AuthorizedCenterState,
            ) -> "AuthorizedGenerationClaim":
        """把 READY center 的 typed payload 和 citation 绑定到实际候选。"""
        if not isinstance(candidate, GenerationCandidate):
            raise TypeError("authorized claim candidate 类型错误")
        if not isinstance(state, AuthorizedCenterState):
            raise TypeError("authorized claim center state 类型错误")
        if state.receipt.state != "READY" or state.payload is None:
            raise AuthorizedGenerationDeliveryError(
                "只有 READY center 可授权 generation claim")
        if candidate.proposition != state.payload.proposition:
            raise AuthorizedGenerationDeliveryError(
                "generation candidate 替换了授权 payload 命题")
        cited = tuple(sorted(
            {item.source_ref for item in state.receipt.citations},
            key=lambda item: item.stable_key(),
        ))
        if not set(candidate.citation_sources).issubset(set(cited)):
            raise AuthorizedGenerationDeliveryError(
                "generation candidate 引用了未授权来源")
        return cls(
            candidate.stable_key(),
            candidate.proposition.stable_key(),
            candidate.source,
            candidate.scope,
            candidate.citation_sources,
            state.receipt.receipt_key.components,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回候选、命题、归属、引用和授权收据完整键。"""
        result: list[int] = []
        for value in (
                self.candidate_key,
                self.proposition_key,
                self.source.stable_key(),
                self.scope.stable_key()):
            _pack(result, value)
        result.append(len(self.evidence_sources))
        for source in self.evidence_sources:
            _pack(result, source.stable_key())
        _pack(result, self.authorization_receipt_key)
        return tuple(result)


@dataclass(frozen=True)
class AuthorizedGenerationEnvelope:
    """唯一允许跨外部交付边界暴露的 renderer 输出和授权证明。"""

    run_key: tuple[int, ...]
    renderer: ObjectIdentity
    representations: tuple[ObjectIdentity, ...]
    units: tuple[int, ...]
    claims: tuple[AuthorizedGenerationClaim, ...]
    cited_sources: tuple[SourceRef, ...]
    postcheck_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验 envelope 含实际输出、精确 claim 集、引用和同次 G-04。"""
        _integer_stream(self.run_key, where="delivery envelope run_key")
        if not isinstance(self.renderer, ObjectIdentity):
            raise TypeError("delivery envelope renderer 类型错误")
        if (not isinstance(self.representations, tuple)
                or not self.representations
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.representations)):
            raise TypeError("delivery envelope representations 类型错误")
        _strict_key(self.units, where="delivery envelope units")
        if (not isinstance(self.claims, tuple)
                or any(not isinstance(item, AuthorizedGenerationClaim)
                       for item in self.claims)):
            raise TypeError("delivery envelope claims 类型错误")
        if self.claims != tuple(sorted(
                self.claims, key=lambda item: item.candidate_key)):
            raise AuthorizedGenerationDeliveryError(
                "delivery envelope claims 未规范排序")
        if (not isinstance(self.cited_sources, tuple)
                or any(not isinstance(item, SourceRef)
                       for item in self.cited_sources)):
            raise TypeError("delivery envelope cited_sources 类型错误")
        if self.cited_sources != tuple(sorted(
                set(self.cited_sources), key=lambda item: item.stable_key())):
            raise AuthorizedGenerationDeliveryError(
                "delivery envelope citations 必须排序去重")
        _integer_stream(
            self.postcheck_key, where="delivery envelope postcheck_key")


@dataclass(frozen=True)
class AuthorizedGenerationDeliveryDecision:
    """交付授权结果；拒绝态只保留审计键，不携带 surface。"""

    state: str
    audit_key: tuple[int, ...]
    envelope: AuthorizedGenerationEnvelope | None = None

    def __post_init__(self) -> None:
        """核验只有 AUTHORIZED 状态可暴露 envelope。"""
        if self.state not in _DELIVERY_STATES:
            raise AuthorizedGenerationDeliveryError("delivery decision state 非法")
        _integer_stream(self.audit_key, where="delivery decision audit_key")
        if self.state == DELIVERY_AUTHORIZED:
            if not isinstance(self.envelope, AuthorizedGenerationEnvelope):
                raise TypeError("AUTHORIZED decision 缺 envelope")
        elif self.envelope is not None:
            raise AuthorizedGenerationDeliveryError(
                "拒绝 decision 不得暴露 renderer surface")

    @property
    def deliverable(self) -> bool:
        """返回当前决断是否持有可交付 envelope。"""
        return self.envelope is not None


class AuthorizedGenerationDeliveryAuthority:
    """将授权 claim、planned Proposition、citation 与 G-04 逐点对齐。"""

    @staticmethod
    def _audit_key(
            run: QuestionAnswerRun,
            claims: tuple[AuthorizedGenerationClaim, ...],
            state: str,
            ) -> tuple[int, ...]:
        """形成不包含 renderer units 的拒绝或授权审计键。"""
        result = [*map(ord, state), len(claims)]
        for claim in claims:
            _pack(result, integer_tuple_fingerprint(
                claim.stable_key(), domain="authorized.delivery.claim.v1"))
        _pack(result, integer_tuple_fingerprint(
            run.stable_key(), domain="authorized.delivery.run.v1"))
        return integer_tuple_fingerprint(
            tuple(result), domain="authorized.delivery.audit.v1")

    @classmethod
    def _reject(
            cls,
            run: QuestionAnswerRun,
            claims: tuple[AuthorizedGenerationClaim, ...],
            state: str,
            ) -> AuthorizedGenerationDeliveryDecision:
        """返回只含内容引用审计键的 fail-closed 决断。"""
        return AuthorizedGenerationDeliveryDecision(
            state, cls._audit_key(run, claims, state))

    def authorize(
            self,
            run: QuestionAnswerRun,
            claims: tuple[AuthorizedGenerationClaim, ...],
            ) -> AuthorizedGenerationDeliveryDecision:
        """仅在授权 claim 与同次完整 G-04 精确闭合时暴露 surface。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("delivery authority run 类型错误")
        if (not isinstance(claims, tuple)
                or any(not isinstance(item, AuthorizedGenerationClaim)
                       for item in claims)):
            raise TypeError("delivery authority claims 类型错误")
        claims = tuple(sorted(claims, key=lambda item: item.candidate_key))
        claim_map = {item.candidate_key: item for item in claims}
        if len(claim_map) != len(claims):
            return self._reject(run, claims, DELIVERY_CLAIM_COVERAGE_MISMATCH)
        generation = run.generation
        if (generation is None or not generation.complete
                or generation.rendered is None):
            return self._reject(run, claims, DELIVERY_GENERATION_INCOMPLETE)
        if run.postcheck is None:
            return self._reject(run, claims, DELIVERY_POSTCHECK_MISSING)
        if not run.postcheck.complete:
            return self._reject(run, claims, DELIVERY_POSTCHECK_FAILED)
        planned_items = generation.surface.preview.request.structure \
            .propositions.propositions
        planned = {item.candidate_key: item for item in planned_items}
        if set(claim_map) != set(planned):
            return self._reject(run, claims, DELIVERY_CLAIM_COVERAGE_MISMATCH)
        candidates = {
            item.stable_key(): item for item in generation.plan.request.candidates}
        for key, claim in claim_map.items():
            candidate = candidates.get(key)
            proposition = planned[key]
            if (candidate is None
                    or candidate.proposition.stable_key() != claim.proposition_key
                    or proposition.proposition.stable_key() != claim.proposition_key
                    or candidate.source != claim.source
                    or proposition.source != claim.source
                    or candidate.scope != claim.scope
                    or proposition.scope != claim.scope
                    or candidate.citation_sources != claim.evidence_sources):
                return self._reject(
                    run, claims, DELIVERY_CLAIM_BINDING_MISMATCH)
        requirements = {
            item.candidate_key: item
            for item in run.postcheck.request.source_requirements
        }
        if set(requirements) != set(planned):
            return self._reject(run, claims, DELIVERY_REQUIREMENT_MISMATCH)
        for key, requirement in requirements.items():
            claim = claim_map[key]
            if not requirement.citation_required or not requirement.trust_required:
                return self._reject(
                    run, claims, DELIVERY_REQUIREMENT_NOT_STRICT)
            if (requirement.source != claim.source
                    or requirement.scope != claim.scope
                    or requirement.evidence_sources != claim.evidence_sources):
                return self._reject(run, claims, DELIVERY_REQUIREMENT_MISMATCH)
        observation = run.postcheck.parsed.observation
        if observation is None:
            return self._reject(run, claims, DELIVERY_OBSERVATION_MISMATCH)
        recovered = {
            item.candidate_key: item for item in observation.propositions}
        if set(recovered) != set(planned):
            return self._reject(run, claims, DELIVERY_OBSERVATION_MISMATCH)
        for key, item in recovered.items():
            claim = claim_map[key]
            if (item.proposition.stable_key() != claim.proposition_key
                    or item.source != claim.source
                    or item.scope != claim.scope):
                return self._reject(
                    run, claims, DELIVERY_OBSERVATION_MISMATCH)
        expected_citations = tuple(sorted(
            {source for item in claims for source in item.evidence_sources},
            key=lambda item: item.stable_key(),
        ))
        if observation.cited_sources != expected_citations:
            return self._reject(run, claims, DELIVERY_CITATION_MISMATCH)
        envelope = AuthorizedGenerationEnvelope(
            integer_tuple_fingerprint(
                run.stable_key(), domain="authorized.delivery.run.v1"),
            generation.rendered.renderer,
            generation.rendered.representations,
            generation.rendered.units,
            claims,
            expected_citations,
            integer_tuple_fingerprint(
                run.postcheck.stable_key(),
                domain="authorized.delivery.postcheck.v1",
            ),
        )
        return AuthorizedGenerationDeliveryDecision(
            DELIVERY_AUTHORIZED,
            self._audit_key(run, claims, DELIVERY_AUTHORIZED),
            envelope,
        )


__all__ = [
    "AuthorizedGenerationClaim",
    "AuthorizedGenerationDeliveryAuthority",
    "AuthorizedGenerationDeliveryDecision",
    "AuthorizedGenerationDeliveryError",
    "AuthorizedGenerationEnvelope",
    "DELIVERY_AUTHORIZED",
    "DELIVERY_CITATION_MISMATCH",
    "DELIVERY_CLAIM_BINDING_MISMATCH",
    "DELIVERY_CLAIM_COVERAGE_MISMATCH",
    "DELIVERY_GENERATION_INCOMPLETE",
    "DELIVERY_OBSERVATION_MISMATCH",
    "DELIVERY_POSTCHECK_FAILED",
    "DELIVERY_POSTCHECK_MISSING",
    "DELIVERY_REQUIREMENT_MISMATCH",
    "DELIVERY_REQUIREMENT_NOT_STRICT",
]
