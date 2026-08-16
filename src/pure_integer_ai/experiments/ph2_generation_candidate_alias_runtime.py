"""把 generation candidate pack 适配为 run-local production R-01 owner。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseLoader,
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
    normalize_forming_evidence_keys,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_course import (
    build_generation_candidate_alias_manifest,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    GenerationCandidatePack,
    RULE_CLAIM,
    RULE_LITERAL,
    RULE_REFERENCE,
    RULE_RESPONSE_ACT,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorVariant,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActVariant,
)
from pure_integer_ai.experiments.train_context import TrainContext


def _text_content(representation: ObjectIdentity) -> str:
    """从 Unicode Representation 恢复文本，仅用于 pack 规则核验。"""
    _family, content = representation_parts(representation)
    try:
        return "".join(chr(item) for item in content)
    except (ValueError, OverflowError) as error:
        raise GenerationCandidateAliasRuntimeError(
            "candidate Representation 含非法 Unicode scalar") from error


def _pattern_support_keys(
        pack: GenerationCandidatePack, *, claim_count: int | None = None,
        ) -> tuple[tuple[int, ...], ...]:
    """汇总指定形状 TRAIN pattern 的形成 Evidence，不读取 episode surface。"""
    keys = {
        key
        for pattern in pack.model.patterns
        if claim_count is None or pattern.claim_count == claim_count
        for key in pattern.support_teacher_keys
    }
    if not keys:
        raise GenerationCandidateAliasRuntimeError(
            "candidate pack 缺少所需 TRAIN pattern 形成 Evidence")
    return tuple(sorted(keys))


def _request_for_grounded_variant(
        pack: GenerationCandidatePack,
        variant: GroundedAnswerConnectorVariant,
        visible_evidence_keys: tuple[tuple[int, ...], ...],
        ) -> GenerationCandidateAliasCourseRequest:
    """把单命题 ANSWER variant 转为 literal/claim 分账的 R-01 请求。"""
    pattern = pack.pattern(variant.option.pattern_id)
    if (variant.option.support_teacher_keys != pattern.support_teacher_keys
            or len(variant.aliases) != len(pattern.parts)):
        raise GenerationCandidateAliasRuntimeError(
            "grounded variant 与 candidate pack pattern 漂移")
    visible = () if not visible_evidence_keys else (
        normalize_forming_evidence_keys(
            visible_evidence_keys, where="visible answer evidence"))
    aliases = tuple(sorted(
        variant.aliases, key=lambda item: item.part_ordinal))
    bindings = []
    for ordinal, (alias, part) in enumerate(
            zip(aliases, pattern.parts, strict=True)):
        if alias.part_kind != part.kind or alias.part_ordinal != ordinal:
            raise GenerationCandidateAliasRuntimeError(
                "grounded alias 与 learned pattern part 漂移")
        text = _text_content(alias.representation)
        if part.kind == PATTERN_LITERAL:
            if text != part.literal:
                raise GenerationCandidateAliasRuntimeError(
                    "grounded literal Representation 未由 TRAIN pattern 形成")
            rule = RULE_LITERAL
            forming = pattern.support_teacher_keys
        elif part.kind == PATTERN_CLAIM:
            if not visible:
                raise GenerationCandidateAliasRuntimeError(
                    "grounded claim Representation 缺少 visible Evidence")
            rule = RULE_CLAIM
            forming = tuple(sorted(set(
                (*pattern.support_teacher_keys, *visible))))
        else:
            raise GenerationCandidateAliasRuntimeError(
                "grounded pattern part 未注册")
        bindings.append(GenerationCandidateRealizationBinding(
            alias.filler, alias.representation, rule, forming))
    return GenerationCandidateAliasCourseRequest(
        variant.template.language_branch, tuple(bindings))


def _request_for_response_act(
        pack: GenerationCandidatePack,
        variant: GroundedResponseActVariant,
        ) -> GenerationCandidateAliasCourseRequest:
    """把 learned response-act literal 转为精确 pack-owned realizes 请求。"""
    pattern = pack.pattern(variant.pattern_id)
    if (pattern.response_act != variant.response_act
            or pattern.claim_count != 0
            or len(pattern.parts) != 1
            or pattern.parts[0].kind != PATTERN_LITERAL
            or variant.support_teacher_keys != pattern.support_teacher_keys
            or _text_content(variant.representation)
            != pattern.parts[0].literal):
        raise GenerationCandidateAliasRuntimeError(
            "response-act variant 未由 candidate pack 精确形成")
    return GenerationCandidateAliasCourseRequest(
        variant.template.branch,
        (GenerationCandidateRealizationBinding(
            variant.template.stance,
            variant.representation,
            RULE_RESPONSE_ACT,
            pattern.support_teacher_keys,
        ),),
    )


def _request_for_reference(
        pack: GenerationCandidatePack,
        compilation: GroundedAnswerReferenceCompilation,
        ) -> GenerationCandidateAliasCourseRequest:
    """把双句 compilation 转为 visible claim/reference 与 TRAIN grammar 请求。"""
    if compilation.strategy not in pack.reference_strategies:
        raise GenerationCandidateAliasRuntimeError(
            "reference strategy 不属于 candidate pack")
    forming = normalize_forming_evidence_keys(
        compilation.forming_evidence_keys,
        where="reference compilation forming evidence",
    )
    grammar = _pattern_support_keys(pack, claim_count=2)
    literals = pack.literal_inventory()
    branch = compilation.sentences[0].template.language_branch
    if any(item.template.language_branch != branch
           for item in compilation.sentences):
        raise GenerationCandidateAliasRuntimeError(
            "reference compilation language branch 漂移")
    bindings = []
    for sentence in compilation.sentences:
        for alias in sentence.aliases:
            text = _text_content(alias.representation)
            if alias.filler == compilation.reference_origin:
                rule = RULE_REFERENCE
                evidence = tuple(sorted(set((*grammar, *forming))))
            elif alias.part_kind == PATTERN_CLAIM:
                rule = RULE_CLAIM
                evidence = tuple(sorted(set((*grammar, *forming))))
            else:
                if not any(text in literal for literal in literals):
                    raise GenerationCandidateAliasRuntimeError(
                        "reference grammar literal 未由 TRAIN pattern 覆盖")
                rule = RULE_LITERAL
                evidence = grammar
            bindings.append(GenerationCandidateRealizationBinding(
                alias.filler, alias.representation, rule, evidence))
    reference = GenerationCandidateReferenceBinding(
        compilation.reference_origin,
        compilation.claims[0].candidate.proposition.template,
        tuple(sorted(set((*grammar, *forming)))),
    )
    return GenerationCandidateAliasCourseRequest(
        branch, tuple(bindings), (reference,))


def generation_candidate_alias_request(
        pack: GenerationCandidatePack,
        value: (
            GroundedAnswerConnectorVariant
            | GroundedResponseActVariant
            | GroundedAnswerReferenceCompilation
        ),
        *, visible_evidence_keys: tuple[tuple[int, ...], ...] = (),
        ) -> GenerationCandidateAliasCourseRequest:
    """按现役三类 run-local 输入建立统一、无 label 学习的 alias 请求。"""
    if not isinstance(pack, GenerationCandidatePack):
        raise TypeError("generation alias request pack 类型错误")
    if isinstance(value, GroundedAnswerConnectorVariant):
        return _request_for_grounded_variant(
            pack, value, visible_evidence_keys)
    if isinstance(value, GroundedResponseActVariant):
        return _request_for_response_act(pack, value)
    if isinstance(value, GroundedAnswerReferenceCompilation):
        return _request_for_reference(pack, value)
    raise TypeError("generation alias request 输入类型未注册")


# object-model: factory; owner=run-local-r01
class ProductionGenerationAliasRuntimeFactory:
    """以一个 pack 和 TrainContext 为一次 generation run 建立独占 R-01 owner。"""

    def __init__(
            self,
            pack: GenerationCandidatePack,
            ctx: TrainContext,
            *,
            visible_evidence_keys: tuple[tuple[int, ...], ...] = (),
            ) -> None:
        if not isinstance(pack, GenerationCandidatePack):
            raise TypeError("production generation alias pack 类型错误")
        if not isinstance(ctx, TrainContext):
            raise TypeError("production generation alias ctx 类型错误")
        if visible_evidence_keys:
            visible_evidence_keys = normalize_forming_evidence_keys(
                visible_evidence_keys, where="factory visible evidence")
        elif not isinstance(visible_evidence_keys, tuple):
            raise TypeError("factory visible evidence 必须是 tuple")
        self.pack = pack
        self.ctx = ctx
        self.visible_evidence_keys = visible_evidence_keys
        self._request: GenerationCandidateAliasCourseRequest | None = None
        self._loaded: LoadedAliasRelationCourse | None = None

    def build(
            self,
            value: (
                GroundedAnswerConnectorVariant
                | GroundedResponseActVariant
                | GroundedAnswerReferenceCompilation
            ),
            ) -> AliasRelationRuntime:
        """核验输入只消费 pack/visible Evidence，再幂等加载正式 R-01 owner。"""
        request = generation_candidate_alias_request(
            self.pack,
            value,
            visible_evidence_keys=self.visible_evidence_keys,
        )
        if self._request is not None and request != self._request:
            raise GenerationCandidateAliasRuntimeError(
                "同一 run-local alias factory 不得装配第二个 request")
        if self._loaded is None:
            manifest = build_generation_candidate_alias_manifest(
                self.pack, request)
            self._loaded = AliasRelationCourseLoader(
                manifest, manifest.sha256()).load(self.ctx)
            self._request = request
        return self._loaded.alias

    @property
    def loaded_course(self) -> LoadedAliasRelationCourse | None:
        """返回已加载课程，未 build 时保持 None。"""
        return self._loaded

    def clone_for_evaluation(
            self, ctx: TrainContext,
            ) -> "ProductionGenerationAliasRuntimeFactory":
        """复制不可变 pack/policy，在 evaluation context 重建独立 owner。"""
        return ProductionGenerationAliasRuntimeFactory(
            self.pack,
            ctx,
            visible_evidence_keys=self.visible_evidence_keys,
        )

    def state_key(self) -> tuple:
        """返回 pack 内容锁、visible Evidence 与可选 request/loader 状态。"""
        request_key = () if self._request is None else self._request.stable_key()
        loader_key = (
            () if self._loaded is None else self._loaded.factory.state_key())
        return (
            self.pack.sha256(),
            self.visible_evidence_keys,
            request_key,
            loader_key,
        )


__all__ = [
    "GenerationCandidateAliasCourseRequest",
    "GenerationCandidateAliasRuntimeError",
    "GenerationCandidateRealizationBinding",
    "GenerationCandidateReferenceBinding",
    "ProductionGenerationAliasRuntimeFactory",
    "build_generation_candidate_alias_manifest",
    "generation_candidate_alias_request",
]
