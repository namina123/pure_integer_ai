"""把 teacher-frozen Proposition/event reference 课程编译为多句 connector。"""
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
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)


_NAMESPACE = 20960


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
    """冻结一条 schema v2 课程、完整 planning 与显式 claim 映射。"""

    episode: GroundedAnswerEpisode
    planning: GenerationPlanningRequest
    claims: tuple[GroundedAnswerClaimCandidateBinding, ...]
    language_branch: ObjectIdentity
    representation_family: tuple[int, ...]
    strategy: str
    forming_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, GroundedAnswerEpisode):
            raise TypeError("reference compile episode 类型错误")
        if self.episode.reference_course is None:
            raise GroundedAnswerReferenceCompileError(
                "reference compile 缺少 teacher-frozen 课程")
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
        if (not isinstance(self.forming_teacher_keys, tuple)
                or not self.forming_teacher_keys):
            raise GroundedAnswerReferenceCompileError(
                "reference compile 缺少 forming teacher keys")
        for key in self.forming_teacher_keys:
            _strict_key(key, where="reference forming teacher key")
        if len(set(self.forming_teacher_keys)) != len(
                self.forming_teacher_keys):
            raise GroundedAnswerReferenceCompileError(
                "reference forming teacher key 重复")
        object.__setattr__(self, "forming_teacher_keys", tuple(sorted(
            self.forming_teacher_keys)))
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
        if self.planning.candidates != candidates:
            raise GroundedAnswerReferenceCompileError(
                "planning candidates 必须使用显式 claim 顺序")
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
    forming_teacher_keys: tuple[tuple[int, ...], ...]
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
        if self.planning.candidates != tuple(
                item.candidate for item in self.claims):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation planning/candidates 漂移")
        if (not isinstance(self.forming_teacher_keys, tuple)
                or not self.forming_teacher_keys
                or self.forming_teacher_keys != tuple(sorted(
                    set(self.forming_teacher_keys)))):
            raise GroundedAnswerReferenceCompileError(
                "reference compilation forming teacher keys 非规范")
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


def _claim_texts(episode: GroundedAnswerEpisode) -> dict[str, str]:
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
    """返回 selected strategy 的 reference surface 与 accepted id。"""
    course = request.episode.reference_course
    matches = tuple(
        item for item in course.surface_labels
        if item.strategy == request.strategy
    )
    if len(matches) != 1:
        raise GroundedAnswerReferenceCompileError(
            "reference strategy 未唯一绑定 teacher label")
    return matches[0].reference_surface, matches[0].realization_id


def _sentence_parts(
        request: GroundedAnswerReferenceCompileRequest,
        ) -> tuple[tuple[tuple[str, str], ...], ...]:
    """按冻结双句课程分解 claim、reference 与边界字面。"""
    claim_texts = _claim_texts(request.episode)
    first_id, second_id = (
        item.proposition_id for item in request.claims)
    reference_surface, realization_id = _surface_label(request)
    accepted = {
        item.realization_id: item.surface
        for item in request.episode.surfaces.accepted
    }
    expected = (
        claim_texts[first_id] + "。"
        + reference_surface + claim_texts[second_id] + "。"
    )
    if accepted.get(realization_id) != expected:
        raise GroundedAnswerReferenceCompileError(
            "首个 reference compiler 只接受冻结的双句 claim/reference 形状")
    return (
        ((PATTERN_CLAIM, claim_texts[first_id]),
         (PATTERN_LITERAL, "。")),
        (("REFERENCE", reference_surface),
         (PATTERN_CLAIM, claim_texts[second_id]),
         (PATTERN_LITERAL, "。")),
    )


def compile_grounded_answer_reference_connector(
        request: GroundedAnswerReferenceCompileRequest,
        surface_protocol: GenerationSurfaceProtocol,
        ) -> GroundedAnswerReferenceCompilation:
    """把冻结双句课程编译为逐 Proposition templates 和来源 declarations。"""
    if not isinstance(request, GroundedAnswerReferenceCompileRequest):
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
            "episode": request.episode.episode_id,
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
            "episode": request.episode.episode_id,
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
                    _stable_id(text),
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
                    tuple(ord(char) for char in text),
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
            "episode": request.episode.episode_id,
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
        request.episode.episode_id,
        request.strategy,
        request.claims,
        request.planning,
        request.forming_teacher_keys,
        tuple(sentence_compilations),
        connector,
        reference_origin,
        reference_slot,
    )


__all__ = [
    "GroundedAnswerClaimCandidateBinding",
    "GroundedAnswerReferenceCompilation",
    "GroundedAnswerReferenceCompileError",
    "GroundedAnswerReferenceCompileRequest",
    "GroundedAnswerReferenceSentenceCompilation",
    "compile_grounded_answer_reference_connector",
]
