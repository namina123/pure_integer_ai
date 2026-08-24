"""把 evidence-bound Proposition/event reference 输入编译为多句 connector。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    ObjectIdentity,
    concept_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderSearchBudget,
)
from pure_integer_ai.experiments.language_generation_connector import (
    BoundPropositionAnaphoraDeclaration,
    BoundPropositionAnaphoraDeclarations,
    BoundPropositionAnaphoraLink,
    BoundPropositionDiscourseDeclaration,
    BoundPropositionDiscourseDeclarations,
    BoundPropositionDiscourseDependency,
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective,
    LanguageConnectorSurfaceRuntimePolicy,
    LanguageConnectorTemplateRuntimePolicy,
    LanguageConnectorValueProtocol,
    LanguageGenerationConnector,
    LanguageGenerationConnectorRegistry,
    LanguageGenerationConnectorRuntimePolicy,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerAliasRequirement,
    GroundedAnswerOrderRequirement,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
    GroundedAnswerEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    PublicResponseActPlanningBuild,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    decode_utf8_v1,
    encode_utf8_v1,
)


_NAMESPACE = 20960
_ExecutableEpisode = (
    GroundedAnswerEpisode | GenerationGeneralizationEvaluationObservation)


def _is_executable_episode(value: object) -> bool:
    """只接受 TRAIN 完整 episode 或 held-out label-free Observation。"""
    return isinstance(value, (
        GroundedAnswerEpisode,
        GenerationGeneralizationEvaluationObservation,
    ))


# object-model: exception
class GroundedAnswerReferenceCompileError(ValueError):
    """reference 课程不能无损形成多句 connector。"""


def _stable_id(value: object) -> int:
    """从规范 JSON 值形成稳定正整数身份。"""
    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    result = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return result if result > 0 else 1


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerReferenceCompileError(
            f"{where} 必须是非空严格整数 tuple")
    return value


def _source_claim_scalars(value: str, *, where: str) -> tuple[int, ...]:
    """将 planning Evidence 的文本经显式 UTF-8 v1 往返为规范 scalar。

    ``GroundedResponseActPlanningInput`` 是课程 byte parser 的 transport 结果；这里
    不把 Python ``str`` 当作 claim identity，而是要求其能经已有整数 UTF-8 状态机无损
    重建。Public connector 的 claim surface 只能等于这份来源化 planning 输入。
    """
    if not isinstance(value, str) or not value:
        raise GroundedAnswerReferenceCompileError(
            f"{where} 必须是非空 Evidence claim text")
    transport_scalars = tuple(ord(character) for character in value)
    try:
        encoded = encode_utf8_v1(transport_scalars)
    except (TypeError, ValueError) as error:
        raise GroundedAnswerReferenceCompileError(
            f"{where} 不能编码为 UTF-8 v1") from error
    restored = decode_utf8_v1(encoded)
    if restored is None or restored != transport_scalars:
        raise GroundedAnswerReferenceCompileError(
            f"{where} UTF-8 v1 scalar 往返漂移")
    return restored


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerClaimCandidateBinding:
    """把 answer-plan claim id 显式绑定到一个 planning candidate。"""

    proposition_id: str
    candidate: GenerationCandidate

    def __post_init__(self) -> None:
        if (not isinstance(self.proposition_id, str)
                or not self.proposition_id
                or self.proposition_id.strip() != self.proposition_id):
            raise GroundedAnswerReferenceCompileError(
                "claim binding proposition_id 非法")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("claim binding candidate 类型错误")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceCompileRequest:
    """冻结一条 reference 输入、完整 planning 与显式 claim 映射。"""

    episode: _ExecutableEpisode
    planning: GenerationPlanningRequest
    claims: tuple[GroundedAnswerClaimCandidateBinding, ...]
    language_branch: ObjectIdentity
    representation_family: tuple[int, ...]
    strategy: str
    forming_evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not _is_executable_episode(self.episode):
            raise TypeError("reference compile episode 类型错误")
        if self.episode.reference_course is None:
            raise GroundedAnswerReferenceCompileError(
                "reference compile 缺少 reference 输入")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("reference compile planning 类型错误")
        if (not isinstance(self.claims, tuple)
                or len(self.claims) != 2
                or any(not isinstance(
                    item, GroundedAnswerClaimCandidateBinding)
                    for item in self.claims)):
            raise GroundedAnswerReferenceCompileError(
                "首个 reference compile 必须精确绑定两个 claim")
        if self.language_branch.object_kind != OBJECT_LANGUAGE_BRANCH:
            raise GroundedAnswerReferenceCompileError(
                "reference compile language branch 类型错误")
        _strict_key(
            self.representation_family,
            where="reference representation family")
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GroundedAnswerReferenceCompileError(
                "reference compile strategy 未注册")
        if (not isinstance(self.forming_evidence_keys, tuple)
                or not self.forming_evidence_keys):
            raise GroundedAnswerReferenceCompileError(
                "reference compile 缺少 forming evidence keys")
        for key in self.forming_evidence_keys:
            _strict_key(key, where="reference forming evidence key")
        if len(set(self.forming_evidence_keys)) != len(
                self.forming_evidence_keys):
            raise GroundedAnswerReferenceCompileError(
                "reference forming evidence key 重复")
        object.__setattr__(self, "forming_evidence_keys", tuple(sorted(
            self.forming_evidence_keys)))
        course = self.episode.reference_course
        claim_ids = tuple(item.proposition_id for item in self.claims)
        if claim_ids != course.ordered_proposition_ids:
            raise GroundedAnswerReferenceCompileError(
                "claim binding 必须遵循 reference course 命题顺序")
        candidates = tuple(item.candidate for item in self.claims)
        if (len(set(candidates)) != len(candidates)
                or set(candidates) != set(self.planning.candidates)):
            raise GroundedAnswerReferenceCompileError(
                "claim binding 必须精确覆盖 planning candidates")
        if self.planning.goal.target_branch != self.language_branch:
            raise GroundedAnswerReferenceCompileError(
                "planning goal branch 与 reference compile 漂移")
        sources = {item.source for item in candidates}
        scopes = {item.scope for item in candidates}
        if (len(sources) != 1 or len(scopes) != 1
                or self.planning.goal.source not in sources
                or self.planning.goal.scope not in scopes):
            raise GroundedAnswerReferenceCompileError(
                "reference candidates 必须同 source/scope")

    @property
    def episode_id(self) -> str:
        """返回旧训练适配器已持有的 episode identity。"""
        return self.episode.episode_id


# object-model: value; representation=struct; interop=DLG-RAW-05B
@dataclass(frozen=True, slots=True)
class PublicGroundedAnswerReferenceClaimSurface:
    """一个公开 Evidence claim 的 source-derived Unicode scalar，不含答案表面。"""

    proposition_id: str
    scalars: tuple[int, ...]

    def __post_init__(self) -> None:
        """拒绝宿主文本、空 claim、surrogate 或未界定的标量序列。"""
        if (not isinstance(self.proposition_id, str) or not self.proposition_id
                or self.proposition_id.strip() != self.proposition_id):
            raise GroundedAnswerReferenceCompileError(
                "public reference claim proposition id 非法")
        if (not isinstance(self.scalars, tuple) or not self.scalars
                or any(type(item) is not int or item < 0 or item > 0x10FFFF
                       or 0xD800 <= item <= 0xDFFF
                       for item in self.scalars)):
            raise GroundedAnswerReferenceCompileError(
                "public reference claim 必须是非空 Unicode scalar tuple")


# object-model: value; representation=struct; interop=DLG-RAW-05B
@dataclass(frozen=True, slots=True)
class PublicGroundedAnswerReferenceCompileRequest:
    """V3 production connector 的无标签、纯 record 编译输入。"""

    planning_build: PublicResponseActPlanningBuild
    ordered_claims: tuple[PublicGroundedAnswerReferenceClaimSurface, ...]
    representation_family: tuple[int, ...]
    strategy: str
    antecedent_reference_scalars: tuple[int, ...]
    explicit_repetition_scalars: tuple[int, ...]
    forming_evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """验证 planning、两条 claim、reference lexeme 与 Evidence 覆盖闭合。"""
        if not isinstance(self.planning_build, PublicResponseActPlanningBuild):
            raise TypeError("public reference compile planning build 类型错误")
        if (not isinstance(self.ordered_claims, tuple)
                or len(self.ordered_claims) != 2
                or any(not isinstance(
                    item, PublicGroundedAnswerReferenceClaimSurface)
                       for item in self.ordered_claims)):
            raise GroundedAnswerReferenceCompileError(
                "public reference compile 必须有两个 ordered claim")
        if (not isinstance(self.representation_family, tuple)
                or not self.representation_family
                or any(type(item) is not int
                       for item in self.representation_family)):
            raise GroundedAnswerReferenceCompileError(
                "public reference representation family 非法")
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GroundedAnswerReferenceCompileError(
                "public reference strategy 未注册")
        for label, scalars in (
                ("antecedent reference", self.antecedent_reference_scalars),
                ("explicit repetition", self.explicit_repetition_scalars)):
            if (not isinstance(scalars, tuple) or not scalars
                    or any(type(item) is not int or item < 0
                           or item > 0x10FFFF
                           or 0xD800 <= item <= 0xDFFF
                           for item in scalars)):
                raise GroundedAnswerReferenceCompileError(
                    f"public reference {label} 必须是非空 Unicode scalar tuple")
        if (not isinstance(self.forming_evidence_keys, tuple)
                or not self.forming_evidence_keys):
            raise GroundedAnswerReferenceCompileError(
                "public reference forming evidence 缺失")
        for key in self.forming_evidence_keys:
            _strict_key(key, where="public reference forming evidence")
        if self.forming_evidence_keys != tuple(sorted(set(
                self.forming_evidence_keys))):
            raise GroundedAnswerReferenceCompileError(
                "public reference forming evidence 未规范排序")
        planning = self.planning_build.planning
        candidates = planning.candidates
        claim_ids = tuple(item.proposition_id for item in self.ordered_claims)
        binding_ids = tuple(
            item.proposition_id for item
            in self.planning_build.candidate_bindings)
        if (len(candidates) != 2 or len(set(claim_ids)) != 2
                or set(claim_ids) != set(binding_ids)
                or candidates != tuple(
                    self.planning_build.candidate_for(item)
                    for item in claim_ids)
                or any(not item.state.support or item.state.refute
                       for item in candidates)):
            raise GroundedAnswerReferenceCompileError(
                "public reference claims/planning 未形成两个有序 support candidate")
        source_claims: dict[str, tuple[int, ...]] = {}
        for evidence in self.planning_build.planning_input.evidence:
            scalars = _source_claim_scalars(
                evidence.claim_text,
                where=("public reference planning Evidence "
                       + evidence.proposition_id),
            )
            previous = source_claims.get(evidence.proposition_id)
            if previous is not None and previous != scalars:
                raise GroundedAnswerReferenceCompileError(
                    "public reference 同一 Proposition 的 Evidence claim 不唯一")
            source_claims[evidence.proposition_id] = scalars
        if any(source_claims.get(item.proposition_id) != item.scalars
               for item in self.ordered_claims):
            raise GroundedAnswerReferenceCompileError(
                "public reference claim scalar 不等于来源 Evidence")
        if (len({item.source for item in candidates}) != 1
                or len({item.scope for item in candidates}) != 1):
            raise GroundedAnswerReferenceCompileError(
                "public reference candidates 必须同 source/scope")
        expected_evidence = tuple(sorted({
            evidence.stable_key()
            for candidate in candidates for evidence in candidate.evidence
        }))
        if self.forming_evidence_keys != expected_evidence:
            raise GroundedAnswerReferenceCompileError(
                "public reference forming Evidence 未精确覆盖 candidates")

    @property
    def episode_id(self) -> str:
        """从无标签 public planning input 取得公开课程 identity。"""
        return self.planning_build.planning_input.episode_id

    @property
    def planning(self) -> GenerationPlanningRequest:
        """返回只由公开 Evidence 形成的 typed planning。"""
        return self.planning_build.planning

    @property
    def language_branch(self) -> ObjectIdentity:
        """返回 planning 所绑定的公开语言 branch。"""
        return self.planning_build.language_branch

    @property
    def claims(self) -> tuple[GroundedAnswerClaimCandidateBinding, ...]:
        """按显式 public claim order 形成旧执行层所需的 candidate binding。"""
        return tuple(GroundedAnswerClaimCandidateBinding(
            item.proposition_id,
            self.planning_build.candidate_for(item.proposition_id),
        ) for item in self.ordered_claims)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceSentenceCompilation:
    """一个 claim 对应的单句 template、alias 与相邻顺序义务。"""

    proposition_id: str
    candidate: GenerationCandidate
    template: LanguageGenerationConnectorTemplate
    aliases: tuple[GroundedAnswerAliasRequirement, ...]
    orders: tuple[GroundedAnswerOrderRequirement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposition_id, str) or not self.proposition_id:
            raise GroundedAnswerReferenceCompileError(
                "sentence compilation proposition_id 非法")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("sentence compilation candidate 类型错误")
        if not isinstance(self.template, LanguageGenerationConnectorTemplate):
            raise TypeError("sentence compilation template 类型错误")
        if (self.template.proposition_structure
                != self.candidate.proposition.structure
                or self.template.predicate
                != self.candidate.proposition.predicate):
            raise GroundedAnswerReferenceCompileError(
                "sentence template 未绑定 candidate match key")
        if ({item.slot for item in self.aliases}
                != {item.slot for item in self.template.slots}):
            raise GroundedAnswerReferenceCompileError(
                "sentence aliases 未精确覆盖 template slots")
        expected = tuple(
            (before.slot, after.slot)
            for before, after in zip(
                sorted(self.aliases, key=lambda item: item.part_ordinal),
                sorted(self.aliases, key=lambda item: item.part_ordinal)[1:],
            )
        )
        if tuple(
                (item.before_slot, item.after_slot)
                for item in self.orders) != expected:
            raise GroundedAnswerReferenceCompileError(
                "sentence order requirements 与 alias part 顺序漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceCompilation:
    """多句 connector 及其 discourse/anaphora 来源课程。"""

    episode_id: str
    strategy: str
    claims: tuple[GroundedAnswerClaimCandidateBinding, ...]
    planning: GenerationPlanningRequest
    forming_evidence_keys: tuple[tuple[int, ...], ...]
    sentences: tuple[GroundedAnswerReferenceSentenceCompilation, ...]
    connector: LanguageGenerationConnector
    reference_origin: ObjectIdentity
    reference_slot: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise GroundedAnswerReferenceCompileError(
                "reference compilation episode id 非法")
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GroundedAnswerReferenceCompileError(
                "reference compilation strategy 未注册")
        if (not isinstance(self.claims, tuple) or len(self.claims) != 2
                or any(not isinstance(
                    item, GroundedAnswerClaimCandidateBinding)
                    for item in self.claims)):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation claims 非法")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("reference compilation planning 类型错误")
        if set(self.planning.candidates) != {
                item.candidate for item in self.claims}:
            raise GroundedAnswerReferenceCompileError(
                "reference compilation planning/candidates 漂移")
        if (not isinstance(self.forming_evidence_keys, tuple)
                or not self.forming_evidence_keys
                or self.forming_evidence_keys != tuple(sorted(
                    set(self.forming_evidence_keys)))):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation forming evidence keys 非规范")
        if (not isinstance(self.sentences, tuple)
                or len(self.sentences) != 2
                or any(not isinstance(
                    item, GroundedAnswerReferenceSentenceCompilation)
                    for item in self.sentences)):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation 必须精确包含两个句子")
        if tuple(item.proposition_id for item in self.sentences) != tuple(
                item.proposition_id for item in self.claims):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation 句序与 claim 顺序漂移")
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("reference compilation connector 类型错误")
        templates = tuple(item.template for item in self.sentences)
        if set(self.connector.registry.templates) != set(templates):
            raise GroundedAnswerReferenceCompileError(
                "reference connector templates 覆盖漂移")
        if not isinstance(self.reference_origin, ObjectIdentity):
            raise TypeError("reference origin 类型错误")
        if not isinstance(self.reference_slot, ObjectIdentity):
            raise TypeError("reference slot 类型错误")

    @property
    def ordered_candidates(self) -> tuple[GenerationCandidate, ...]:
        """按显式 claim 顺序返回候选，不借用无序 planning 容器顺序。"""
        return tuple(item.candidate for item in self.claims)


def _claim_texts(episode: _ExecutableEpisode) -> dict[str, str]:
    """恢复每个 required Proposition 的唯一 Evidence claim text。"""
    grouped: dict[str, set[str]] = {}
    for evidence in episode.question.evidence:
        grouped.setdefault(evidence.proposition_id, set()).add(
            evidence.claim_text)
    result = {}
    for proposition_id in episode.question.answer_plan.ordered_claim_ids:
        values = grouped.get(proposition_id, set())
        if len(values) != 1:
            raise GroundedAnswerReferenceCompileError(
                "reference claim 缺少唯一 Evidence text")
        result[proposition_id] = next(iter(values))
    return result


def _surface_label(
        request: GroundedAnswerReferenceCompileRequest,
        ) -> tuple[str, str]:
    """返回 selected strategy 的 reference surface 与可选训练 id。"""
    course = request.episode.reference_course
    if isinstance(
            request.episode, GenerationGeneralizationEvaluationObservation):
        assert course is not None
        return course.surface_for(request.strategy), ""
    assert course is not None
    matches = tuple(
        item for item in course.surface_labels
        if item.strategy == request.strategy
    )
    if len(matches) != 1:
        raise GroundedAnswerReferenceCompileError(
            "reference strategy 未唯一绑定训练资源")
    return matches[0].reference_surface, matches[0].realization_id


def _text_scalars(value: str, *, where: str) -> tuple[int, ...]:
    """训练适配器将已验证文本显式降为 Unicode scalar；production 不调用本函数。"""
    if not isinstance(value, str) or not value:
        raise GroundedAnswerReferenceCompileError(
            f"{where} 必须是非空训练文本")
    return tuple(ord(item) for item in value)


def _part_scalars(
        value: str | tuple[int, ...],
        *,
        where: str,
        ) -> tuple[int, ...]:
    """统一旧训练 text 与 public scalar 输入，输出固定 Unicode scalar record。"""
    if isinstance(value, str):
        return _text_scalars(value, where=where)
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int or item < 0 or item > 0x10FFFF
                   or 0xD800 <= item <= 0xDFFF for item in value)):
        raise GroundedAnswerReferenceCompileError(
            f"{where} 不是有效 Unicode scalar tuple")
    return value


def _part_identity(value: str | tuple[int, ...]) -> int:
    """保留旧 text identity，同时为 public scalar 形成明确、无宿主编码的 identity。"""
    if isinstance(value, str):
        return _stable_id(value)
    return _stable_id(list(_part_scalars(value, where="part identity")))


def _legacy_sentence_parts(
        request: GroundedAnswerReferenceCompileRequest,
        ) -> tuple[tuple[tuple[str, str], ...], ...]:
    """训练适配器从冻结课程标签核验双句形状，保留其既有 text identity。"""
    claim_texts = _claim_texts(request.episode)
    first_id, second_id = (
        item.proposition_id for item in request.claims)
    reference_surface, realization_id = _surface_label(request)
    expected = (
        claim_texts[first_id] + "。"
        + reference_surface + claim_texts[second_id] + "。"
    )
    if isinstance(request.episode, GroundedAnswerEpisode):
        accepted = {
            item.realization_id: item.surface
            for item in request.episode.surfaces.accepted
        }
        if accepted.get(realization_id) != expected:
            raise GroundedAnswerReferenceCompileError(
                "reference compiler 只接受冻结的双句 claim/reference 形状")
    return (
        ((PATTERN_CLAIM, claim_texts[first_id]),
         (PATTERN_LITERAL, "。")),
        (("REFERENCE", reference_surface),
         (PATTERN_CLAIM, claim_texts[second_id]),
         (PATTERN_LITERAL, "。")),
    )


def _public_sentence_parts(
        request: PublicGroundedAnswerReferenceCompileRequest,
        ) -> tuple[tuple[tuple[str, tuple[int, ...]], ...], ...]:
    """从无标签 public claim/reference scalar 形成两个结构 part。"""
    first, second = request.ordered_claims
    reference = (
        request.antecedent_reference_scalars
        if request.strategy == "ANTECEDENT_REFERENCE"
        else request.explicit_repetition_scalars)
    return (
        ((PATTERN_CLAIM, first.scalars),
         (PATTERN_LITERAL, (0x3002,))),
        (("REFERENCE", reference),
         (PATTERN_CLAIM, second.scalars),
         (PATTERN_LITERAL, (0x3002,))),
    )


def _sentence_parts(
        request: (
            GroundedAnswerReferenceCompileRequest
            | PublicGroundedAnswerReferenceCompileRequest
        ),
        ) -> tuple[
            tuple[tuple[str, str | tuple[int, ...]], ...], ...]:
    """按输入边界选择训练适配或 public scalar 结构投影。"""
    if isinstance(request, GroundedAnswerReferenceCompileRequest):
        return _legacy_sentence_parts(request)
    if isinstance(request, PublicGroundedAnswerReferenceCompileRequest):
        return _public_sentence_parts(request)
    raise TypeError("reference sentence parts request 类型错误")


def compile_grounded_answer_reference_connector(
        request: (
            GroundedAnswerReferenceCompileRequest
            | PublicGroundedAnswerReferenceCompileRequest
        ),
        surface_protocol: GenerationSurfaceProtocol,
        ) -> GroundedAnswerReferenceCompilation:
    """把训练适配或无标签 public record 编译为逐 Proposition connector。"""
    if not isinstance(request, (
            GroundedAnswerReferenceCompileRequest,
            PublicGroundedAnswerReferenceCompileRequest)):
        raise TypeError("reference connector request 类型错误")
    if not isinstance(surface_protocol, GenerationSurfaceProtocol):
        raise TypeError("reference connector surface protocol 类型错误")
    value_protocol = LanguageConnectorValueProtocol(*tuple(
        minimal_instruction_identity((_NAMESPACE, 1, index))
        for index in range(1, 5)
    ))
    sentence_parts = _sentence_parts(request)
    reference_origin = proposition_identity(
        request.claims[1].candidate.source,
        (_NAMESPACE, 2, _stable_id({
            "episode": request.episode_id,
            "strategy": request.strategy,
            "version": 1,
        })),
    )
    templates = []
    template_policies = []
    sentence_compilations = []
    reference_slot = None
    global_part_ordinal = 0
    for sentence_ordinal, (binding, parts) in enumerate(
            zip(request.claims, sentence_parts, strict=True), start=1):
        candidate = binding.candidate
        identity = _stable_id({
            "candidate": list(candidate.proposition.stable_key()),
            "episode": request.episode_id,
            "sentence_ordinal": sentence_ordinal,
            "strategy": request.strategy,
            "version": 1,
        })
        connector_id = structure_concept_identity(
            (_NAMESPACE, 3, identity, 1))
        sentence = structure_concept_identity(
            (_NAMESPACE, 3, identity, 2))
        structure = structure_concept_identity(
            (_NAMESPACE, 3, identity, 3))
        value_type = concept_identity((_NAMESPACE, 3, identity, 4))
        slots = tuple(
            StructureSlotDefinition(
                structure,
                structure_concept_identity(
                    (_NAMESPACE, 3, identity, 10, index)),
                role_identity((_NAMESPACE, 3, identity, 11, index)),
                value_type,
            )
            for index in range(1, len(parts) + 1)
        )
        bindings = []
        directives = []
        runtime = []
        aliases = []
        for part_ordinal, ((kind, text), slot) in enumerate(
                zip(parts, slots, strict=True), start=1):
            global_part_ordinal += 1
            if kind == PATTERN_CLAIM:
                source = value_protocol.proposition_source
                filler = candidate.proposition.template
                constant = None
                alias_kind = PATTERN_CLAIM
            elif kind == "REFERENCE":
                source = value_protocol.constant_source
                filler = reference_origin
                constant = filler
                alias_kind = PATTERN_LITERAL
                reference_slot = slot.slot
            else:
                source = value_protocol.constant_source
                filler = concept_identity((
                    _NAMESPACE, 3, identity, 20, part_ordinal,
                    _part_identity(text),
                ))
                constant = filler
                alias_kind = PATTERN_LITERAL
            bindings.append(LanguageConnectorSlotBinding(
                structure_concept_identity(
                    (_NAMESPACE, 3, identity, 30, part_ordinal)),
                slot.slot,
                source,
                constant=constant,
            ))
            directives.append(LanguageConnectorSurfaceDirective(
                structure_concept_identity(
                    (_NAMESPACE, 3, identity, 40, part_ordinal)),
                slot.slot,
                surface_protocol.emit_action,
                minimal_instruction_identity(
                    (_NAMESPACE, 3, identity, 41, part_ordinal)),
                structure_concept_identity(
                    (_NAMESPACE, 3, identity, 42, part_ordinal)),
                (),
            ))
            is_reference = kind == "REFERENCE" and (
                request.strategy == "ANTECEDENT_REFERENCE")
            runtime.append(LanguageConnectorSurfaceRuntimePolicy(
                slot.slot,
                (_NAMESPACE, 3, identity, 50, part_ordinal),
                AliasRouteSearchBudget(32, 32, 32),
                (_NAMESPACE, 3, identity, 51, part_ordinal),
                (AliasRouteSearchBudget(32, 32, 32)
                 if is_reference else None),
                ((_NAMESPACE, 3, identity, 52, part_ordinal)
                 if is_reference else ()),
            ))
            aliases.append(GroundedAnswerAliasRequirement(
                filler,
                slot.slot,
                representation_identity(
                    request.representation_family,
                    _part_scalars(text, where="reference surface part"),
                ),
                alias_kind,
                global_part_ordinal - 1,
            ))
        orders = tuple(
            GroundedAnswerOrderRequirement(
                structure_concept_identity(
                    (_NAMESPACE, 3, identity, 60, index)),
                before.slot,
                after.slot,
            )
            for index, (before, after) in enumerate(
                zip(slots, slots[1:]), start=1)
        )
        template = LanguageGenerationConnectorTemplate(
            connector_id,
            request.language_branch,
            candidate.proposition.structure,
            candidate.proposition.predicate,
            sentence,
            structure,
            slots,
            tuple(bindings),
            structure_concept_identity((_NAMESPACE, 3, identity, 70)),
            tuple(item.constraint for item in orders),
            structure_concept_identity((_NAMESPACE, 3, identity, 71)),
            (),
            minimal_instruction_identity((_NAMESPACE, 3, identity, 72)),
            minimal_instruction_identity((_NAMESPACE, 3, identity, 73)),
            tuple(directives),
        )
        templates.append(template)
        template_policies.append(LanguageConnectorTemplateRuntimePolicy(
            connector_id, tuple(runtime)))
        sentence_compilations.append(
            GroundedAnswerReferenceSentenceCompilation(
                binding.proposition_id,
                candidate,
                template,
                tuple(aliases),
                orders,
            )
        )
    if reference_slot is None:
        raise GroundedAnswerReferenceCompileError(
            "reference compiler 未形成 reference slot")
    first, second = request.claims
    discourse = BoundPropositionDiscourseDeclarations((
        BoundPropositionDiscourseDeclaration(
            (first.candidate.proposition, second.candidate.proposition),
            (BoundPropositionDiscourseDependency(
                first.candidate.proposition,
                second.candidate.proposition,
                structure_concept_identity((_NAMESPACE, 4, 1)),
                minimal_instruction_identity((_NAMESPACE, 4, 2)),
                (_NAMESPACE, 4, 3),
            ),),
            first.candidate.source,
            (_NAMESPACE, 4, 4),
        ),
    ))
    anaphora = None
    if request.strategy == "ANTECEDENT_REFERENCE":
        second_template = templates[1]
        anaphora = BoundPropositionAnaphoraDeclarations((
            BoundPropositionAnaphoraDeclaration(
                (first.candidate.proposition, second.candidate.proposition),
                (BoundPropositionAnaphoraLink(
                    first.candidate.proposition,
                    second.candidate.proposition,
                    second_template.sentence,
                    reference_slot,
                    minimal_instruction_identity((_NAMESPACE, 5, 1)),
                    (_NAMESPACE, 5, 2),
                ),),
                first.candidate.source,
                (_NAMESPACE, 5, 3),
            ),
        ))
    runtime_policy = LanguageGenerationConnectorRuntimePolicy(
        (_NAMESPACE, 6, _stable_id({
            "episode": request.episode_id,
            "strategy": request.strategy,
        })),
        StructureOrderSearchBudget(32),
        tuple(template_policies),
    )
    connector = LanguageGenerationConnector(
        LanguageGenerationConnectorRegistry(
            value_protocol, tuple(templates)),
        runtime_policy,
        surface_protocol,
        discourse_declarations=discourse,
        anaphora_declarations=anaphora,
    )
    return GroundedAnswerReferenceCompilation(
        request.episode_id,
        request.strategy,
        request.claims,
        request.planning,
        request.forming_evidence_keys,
        tuple(sentence_compilations),
        connector,
        reference_origin,
        reference_slot,
    )


def compile_public_grounded_answer_reference_connector(
        request: PublicGroundedAnswerReferenceCompileRequest,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> GroundedAnswerReferenceCompilation:
    """编译 V3 public scalar record，不允许调用方走训练 episode 适配器。"""
    if not isinstance(request, PublicGroundedAnswerReferenceCompileRequest):
        raise TypeError("public reference connector request 类型错误")
    return compile_grounded_answer_reference_connector(
        request,
        surface_protocol,
    )


__all__ = [
    "GroundedAnswerClaimCandidateBinding",
    "GroundedAnswerReferenceCompilation",
    "GroundedAnswerReferenceCompileError",
    "GroundedAnswerReferenceCompileRequest",
    "GroundedAnswerReferenceSentenceCompilation",
    "PublicGroundedAnswerReferenceClaimSurface",
    "PublicGroundedAnswerReferenceCompileRequest",
    "compile_grounded_answer_reference_connector",
    "compile_public_grounded_answer_reference_connector",
]
