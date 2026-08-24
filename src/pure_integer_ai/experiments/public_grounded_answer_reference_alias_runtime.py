"""DLG-RAW-05B 的无标签 R-01 alias/refers runtime factory。

本模块不读取完整课程、训练 pack、answer plan 或已接受 surface。它只把同轮
``GroundedAnswerReferenceCompilation`` 已携带的 Evidence 形成键、claim/reference
Representation 和显式 strategy 投影为一个通用 R-01 manifest，再由正式 Loader
在本轮 ``TrainContext`` 建立 active relation owner。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
    AliasRelationPreflightCache,
    LoadedAliasRelationCourse,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateAliasCourseRequest,
    GenerationCandidateAliasRuntimeError,
    GenerationCandidateRealizationBinding,
    GenerationCandidateReferenceBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    AliasRelationManifestProfile,
    build_alias_relation_manifest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    RULE_CLAIM,
    RULE_LITERAL,
    RULE_REFERENCE,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)
from pure_integer_ai.experiments.train_context import TrainContext


_PROFILE_VERSION = (3, 1)


# object-model: exception; interop=DLG-RAW-05B
class PublicGroundedAnswerReferenceAliasRuntimeError(RuntimeError):
    """公开 reference compilation 不能闭合为仅本轮有效的 R-01 owner。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长整数 record 加长度前缀，避免 profile fingerprint 拼接歧义。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise PublicGroundedAnswerReferenceAliasRuntimeError(
            "public reference alias record 非法")
    return len(value), *value


def _request_for(
        compilation: GroundedAnswerReferenceCompilation,
        ) -> GenerationCandidateAliasCourseRequest:
    """只从编译后的 slots 与 Evidence 形成键建立 alias/refers 请求。"""
    if not isinstance(compilation, GroundedAnswerReferenceCompilation):
        raise TypeError("public reference alias compilation 类型错误")
    if compilation.strategy not in {
            "ANTECEDENT_REFERENCE", "EXPLICIT_REPETITION"}:
        raise PublicGroundedAnswerReferenceAliasRuntimeError(
            "public reference alias strategy 未注册")
    branch = compilation.sentences[0].template.language_branch
    if any(item.template.language_branch != branch
           for item in compilation.sentences):
        raise PublicGroundedAnswerReferenceAliasRuntimeError(
            "public reference alias language branch 漂移")
    forming = compilation.forming_evidence_keys
    bindings = []
    for sentence in compilation.sentences:
        for alias in sentence.aliases:
            if alias.filler == compilation.reference_origin:
                rule = RULE_REFERENCE
            elif alias.part_kind == PATTERN_CLAIM:
                rule = RULE_CLAIM
            elif alias.part_kind == PATTERN_LITERAL:
                rule = RULE_LITERAL
            else:
                raise PublicGroundedAnswerReferenceAliasRuntimeError(
                    "public reference alias part kind 未注册")
            bindings.append(GenerationCandidateRealizationBinding(
                alias.filler,
                alias.representation,
                rule,
                forming,
            ))
    references = ()
    if compilation.strategy == "ANTECEDENT_REFERENCE":
        references = (GenerationCandidateReferenceBinding(
            compilation.reference_origin,
            compilation.claims[0].candidate.proposition.template,
            forming,
        ),)
    try:
        return GenerationCandidateAliasCourseRequest(
            branch, tuple(bindings), references)
    except (TypeError, ValueError, RuntimeError) as error:
        raise PublicGroundedAnswerReferenceAliasRuntimeError(
            "public reference alias request 无法闭合") from error


def _profile_for(
        compilation: GroundedAnswerReferenceCompilation,
        request: GenerationCandidateAliasCourseRequest,
        ) -> AliasRelationManifestProfile:
    """用公开 planning/connector/request record 建立不含训练标签的 profile。"""
    if not isinstance(compilation, GroundedAnswerReferenceCompilation):
        raise TypeError("public reference alias profile compilation 类型错误")
    if not isinstance(request, GenerationCandidateAliasCourseRequest):
        raise TypeError("public reference alias profile request 类型错误")
    goal = compilation.planning.goal
    if (request.branch != compilation.sentences[0].template.language_branch
            or goal.target_branch != request.branch
            or goal.scope.source != goal.source):
        raise PublicGroundedAnswerReferenceAliasRuntimeError(
            "public reference alias profile planning/request 漂移")
    values = [1]
    for record in (
            compilation.planning.stable_key(),
            compilation.connector.stable_key(),
            request.stable_key()):
        values.extend(_packed(record))
    values.append(len(compilation.forming_evidence_keys))
    for record in compilation.forming_evidence_keys:
        values.extend(_packed(record))
    digest = integer_tuple_fingerprint(
        tuple(values),
        domain="dlg.raw.public.reference.alias.profile.v1",
    )[2:]
    return AliasRelationManifestProfile(
        _PROFILE_VERSION,
        digest,
        goal.source,
        goal.scope,
        1,
    )


def build_public_grounded_answer_reference_alias_manifest(
        compilation: GroundedAnswerReferenceCompilation,
        ):
    """编译 V3 R-01 manifest；输入只来自本轮无标签 compilation。"""
    request = _request_for(compilation)
    profile = _profile_for(compilation, request)
    return build_alias_relation_manifest(profile, request)


# object-model: resource-owner; interop=DLG-RAW-05B; semantic-state=run-local
class PublicGroundedAnswerReferenceAliasRuntimeFactory:
    """把一份 V3 compilation 幂等装配为当前 TrainContext 独占 R-01 owner。"""

    def __init__(
            self,
            ctx: TrainContext,
            *,
            preflight_cache: AliasRelationPreflightCache | None = None,
            ) -> None:
        """绑定 fresh run context；不可跨 compilation 或跨 backend 复用。"""
        if not isinstance(ctx, TrainContext):
            raise TypeError("public reference alias ctx 类型错误")
        if (preflight_cache is not None
                and not isinstance(preflight_cache, AliasRelationPreflightCache)):
            raise TypeError("public reference alias preflight cache 类型错误")
        self._ctx = ctx
        self._preflight_cache = preflight_cache
        self._manifest = None
        self._loaded: LoadedAliasRelationCourse | None = None

    def build(
            self,
            compilation: GroundedAnswerReferenceCompilation,
            ) -> AliasRelationRuntime:
        """用正式内容锁 Loader 建立 active realizes/refers，不产生长期写入。"""
        manifest = build_public_grounded_answer_reference_alias_manifest(
            compilation)
        if self._manifest is not None and manifest != self._manifest:
            raise PublicGroundedAnswerReferenceAliasRuntimeError(
                "同一 public reference alias factory 不得装配第二份 compilation")
        if self._loaded is None:
            try:
                self._loaded = AliasRelationCourseLoader(
                    manifest,
                    manifest.sha256(),
                ).load(
                    self._ctx,
                    preflight_cache=self._preflight_cache,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise PublicGroundedAnswerReferenceAliasRuntimeError(
                    "public reference alias R-01 Loader 失败") from error
            self._manifest = manifest
        return self._loaded.alias

    def state_key(self) -> tuple[int, ...]:
        """返回仅由 manifest identity 决定的 run-local 可审计状态 record。"""
        if self._manifest is None:
            return ()
        return self._manifest.stable_key()


__all__ = [
    "PublicGroundedAnswerReferenceAliasRuntimeError",
    "PublicGroundedAnswerReferenceAliasRuntimeFactory",
    "build_public_grounded_answer_reference_alias_manifest",
]
