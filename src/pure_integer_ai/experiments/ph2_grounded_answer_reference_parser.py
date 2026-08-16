"""从真实多句 units 与执行实例恢复命题和 reference annotation。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationSentenceInstance,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePreview,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceObservation,
    GenerationSurfaceParseRequest,
    GenerationSurfaceParseResult,
    RecoveredGenerationProposition,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice import (
    GroundedAnswerReferenceStrategy,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)


# object-model: exception
class GroundedAnswerReferenceParserError(ValueError):
    """reference parser catalog 或实际执行不满足受限合同。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加边界。"""
    return len(key), *key


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 renderer 使用一等 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedAnswerReferenceParserError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerReferenceParserError(
            f"{where} 必须是非空严格整数 tuple")
    return value


def _occurrences(units: tuple[int, ...], needle: tuple[int, ...]) -> int:
    """计算一个非空 claim unit 片段的完整出现次数。"""
    if not needle or len(needle) > len(units):
        return 0
    return sum(
        units[index:index + len(needle)] == needle
        for index in range(len(units) - len(needle) + 1)
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceParseOption:
    """一个可从实际 units 和 sentence instance 唯一识别的 grammar 项。"""

    strategy_object: ObjectIdentity
    candidates: tuple[GenerationCandidate, ...]
    representations: tuple[ObjectIdentity, ...]
    units: tuple[int, ...]
    claim_units: tuple[tuple[int, ...], ...]
    structure_payload: tuple[int, ...]
    syntax_key: tuple[int, ...]
    reference_sentence: GenerationSentenceInstance
    reference_slot: ObjectIdentity
    reference_origin: ObjectIdentity
    antecedent: GenerationCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_object, ObjectIdentity):
            raise TypeError("reference parse strategy object 类型错误")
        if (not isinstance(self.candidates, tuple)
                or len(self.candidates) != 2
                or any(not isinstance(item, GenerationCandidate)
                       for item in self.candidates)):
            raise GroundedAnswerReferenceParserError(
                "reference parse option 必须保存两个 candidates")
        if (not isinstance(self.representations, tuple)
                or not self.representations
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.representations)):
            raise GroundedAnswerReferenceParserError(
                "reference parse representations 非法")
        _strict_key(self.units, where="reference parse units")
        if (not isinstance(self.claim_units, tuple)
                or len(self.claim_units) != 2
                or any(not item for item in self.claim_units)):
            raise GroundedAnswerReferenceParserError(
                "reference parse option 缺两个 claim unit 片段")
        _strict_key(
            self.structure_payload,
            where="reference parse structure payload")
        _strict_key(self.syntax_key, where="reference parse syntax key")
        if not isinstance(
                self.reference_sentence, GenerationSentenceInstance):
            raise TypeError("reference parse sentence instance 类型错误")
        if not isinstance(self.reference_slot, ObjectIdentity):
            raise TypeError("reference parse slot 类型错误")
        if not isinstance(self.reference_origin, ObjectIdentity):
            raise TypeError("reference parse origin 类型错误")
        if self.antecedent is not None:
            if (not isinstance(self.antecedent, GenerationCandidate)
                    or self.antecedent not in self.candidates[:-1]):
                raise GroundedAnswerReferenceParserError(
                    "reference parse antecedent 不属于前序候选")

    def stable_key(self) -> tuple[int, ...]:
        """返回本 grammar 项的实际结构、表示和 reference 形状。"""
        values = [*_packed(self.strategy_object.stable_key())]
        values.append(len(self.candidates))
        for candidate in self.candidates:
            values.extend(_packed(candidate.stable_key()))
        values.append(len(self.representations))
        for representation in self.representations:
            values.extend(_packed(representation.stable_key()))
        values.extend((
            *_packed(self.units),
            len(self.claim_units),
        ))
        for units in self.claim_units:
            values.extend(_packed(units))
        values.extend((
            *_packed(self.structure_payload),
            *_packed(self.syntax_key),
            *_packed(self.reference_sentence.stable_key()),
            *_packed(self.reference_slot.stable_key()),
            *_packed(self.reference_origin.stable_key()),
            0 if self.antecedent is None else 1,
        ))
        if self.antecedent is not None:
            values.extend(_packed(self.antecedent.stable_key()))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerRecoveredReference:
    """parser 从 actual units/preview 恢复的 reference 专用观察。"""

    parse_request_key: tuple[int, ...]
    strategy_object: ObjectIdentity
    sentence: GenerationSentenceInstance
    slot: ObjectIdentity
    origin: ObjectIdentity
    representation: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    antecedent_candidate_key: tuple[int, ...] = ()
    antecedent: ObjectIdentity | None = None
    reference_proposal_key: tuple[int, ...] = ()
    surface_proposal_key: tuple[int, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _strict_key(
            self.parse_request_key,
            where="recovered reference parse request key")
        for label, value in (
                ("strategy", self.strategy_object),
                ("slot", self.slot),
                ("origin", self.origin),
                ("representation", self.representation)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"recovered reference {label} 类型错误")
        if not isinstance(self.sentence, GenerationSentenceInstance):
            raise TypeError("recovered reference sentence 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("recovered reference source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("recovered reference scope 类型错误")
        if self.antecedent is None:
            if self.antecedent_candidate_key or self.reference_proposal_key:
                raise GroundedAnswerReferenceParserError(
                    "explicit recovery 不得携带 antecedent/reference proposal")
        else:
            if not isinstance(self.antecedent, ObjectIdentity):
                raise TypeError("recovered antecedent 类型错误")
            _strict_key(
                self.antecedent_candidate_key,
                where="recovered antecedent candidate key")
            _strict_key(
                self.reference_proposal_key,
                where="recovered reference proposal key")
        _strict_key(
            self.surface_proposal_key,
            where="recovered surface proposal key")
        _strict_key(self.trace, where="recovered reference trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回策略、执行实例、proposal 与归属完整键。"""
        values = [
            *_packed(self.parse_request_key),
            *_packed(self.strategy_object.stable_key()),
            *_packed(self.sentence.stable_key()),
            *_packed(self.slot.stable_key()),
            *_packed(self.origin.stable_key()),
            *_packed(self.representation.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            0 if self.antecedent is None else 1,
        ]
        if self.antecedent is not None:
            values.extend(_packed(self.antecedent_candidate_key))
            values.extend(_packed(self.antecedent.stable_key()))
            values.extend(_packed(self.reference_proposal_key))
        values.extend((
            *_packed(self.surface_proposal_key),
            *_packed(self.trace),
        ))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceParserCatalog:
    """保存完整策略 grammar，不保存哪一个策略已被选择。"""

    renderer: ObjectIdentity
    branch: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    candidates: tuple[GenerationCandidate, ...]
    options: tuple[GroundedAnswerReferenceParseOption, ...]

    def __post_init__(self) -> None:
        _instruction(self.renderer, where="reference parser renderer")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise TypeError("reference parser branch 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("reference parser source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("reference parser scope 类型错误")
        if (not isinstance(self.candidates, tuple)
                or len(self.candidates) != 2
                or any(not isinstance(item, GenerationCandidate)
                       for item in self.candidates)):
            raise GroundedAnswerReferenceParserError(
                "reference parser 必须保存两个 candidates")
        if (not isinstance(self.options, tuple) or len(self.options) != 2
                or any(not isinstance(
                    item, GroundedAnswerReferenceParseOption)
                    for item in self.options)):
            raise GroundedAnswerReferenceParserError(
                "reference parser 必须保存两个 strategy grammar")
        if len({item.strategy_object for item in self.options}) != 2:
            raise GroundedAnswerReferenceParserError(
                "reference parser strategy object 重复")
        if len({item.units for item in self.options}) != 2:
            raise GroundedAnswerReferenceParserError(
                "reference parser strategy units 不可区分")
        if any(item.candidates != self.candidates for item in self.options):
            raise GroundedAnswerReferenceParserError(
                "reference parser options 替换了公共 candidates")
        if any(
                candidate.source != self.source
                or candidate.scope != self.scope
                for candidate in self.candidates):
            raise GroundedAnswerReferenceParserError(
                "reference parser candidates 与 source/scope 漂移")
        if len({item.claim_units for item in self.options}) != 1:
            raise GroundedAnswerReferenceParserError(
                "reference parser strategy grammar 替换了公共 claims")


def _parse_option(
        strategy_object: ObjectIdentity,
        compilation: GroundedAnswerReferenceCompilation,
        selection: AnswerContentSelection,
        ) -> GroundedAnswerReferenceParseOption:
    """只把一份 compiled grammar 与实际 sentence instance 编入 catalog。"""
    aliases = tuple(sorted(
        (
            alias
            for sentence in compilation.sentences
            for alias in sentence.aliases
        ),
        key=lambda item: item.part_ordinal,
    ))
    if tuple(item.part_ordinal for item in aliases) != tuple(range(len(aliases))):
        raise GroundedAnswerReferenceParserError(
            "reference parser alias ordinal 不连续")
    representations = tuple(item.representation for item in aliases)
    parts = tuple(
        representation_parts(item.representation)[1]
        for item in aliases
    )
    units = tuple(value for part in parts for value in part)
    claim_units = tuple(
        parts[index]
        for index, item in enumerate(aliases)
        if item.part_kind == PATTERN_CLAIM
    )
    structure_payload = tuple(
        value
        for sentence in compilation.sentences
        for value in _packed(sentence.template.structure.stable_key())
    )
    syntax = compilation.connector.structure_planner().plan(selection).syntax
    matches = tuple(
        sentence for sentence in syntax.sentences
        if compilation.reference_slot in {item.slot for item in sentence.slots}
    )
    if len(matches) != 1 or matches[0].instance is None:
        raise GroundedAnswerReferenceParserError(
            "reference parser 未恢复唯一运行期 reference sentence")
    requirements = tuple(
        item for item in syntax.anaphora
        if (item.address == matches[0].address
            and item.slot == compilation.reference_slot)
    )
    if len(requirements) > 1:
        raise GroundedAnswerReferenceParserError(
            "reference parser grammar 重复声明 anaphora")
    candidates = tuple(item.candidate for item in compilation.claims)
    antecedent = None
    if requirements:
        by_key = {item.stable_key(): item for item in candidates[:-1]}
        antecedent = by_key.get(requirements[0].antecedent_candidate_key)
        if antecedent is None:
            raise GroundedAnswerReferenceParserError(
                "reference parser anaphora 未命中前序 candidate")
    return GroundedAnswerReferenceParseOption(
        strategy_object,
        candidates,
        representations,
        units,
        claim_units,
        structure_payload,
        syntax.stable_key(),
        matches[0].instance,
        compilation.reference_slot,
        compilation.reference_origin,
        antecedent,
    )


def build_grounded_answer_reference_parser_catalog(
        reference_options: tuple[GroundedAnswerReferenceStrategy, ...],
        content_selection: AnswerContentSelection,
        renderer: ObjectIdentity,
        ) -> GroundedAnswerReferenceParserCatalog:
    """从完整 compiled grammar 构造 catalog，不读取 selected strategy。"""
    if (not isinstance(reference_options, tuple)
            or len(reference_options) != 2
            or any(not isinstance(item, GroundedAnswerReferenceStrategy)
                   for item in reference_options)):
        raise TypeError("reference parser options 类型错误")
    if not isinstance(content_selection, AnswerContentSelection):
        raise TypeError("reference parser content selection 类型错误")
    _instruction(renderer, where="reference parser renderer")
    candidates = tuple(
        item.candidate
        for item in reference_options[0].compilation.claims)
    base = reference_options[0].compilation
    if (content_selection.request != base.planning
            or set(content_selection.selected_candidate_keys)
            != {item.stable_key() for item in candidates}):
        raise GroundedAnswerReferenceParserError(
            "reference parser content selection 未覆盖公共 candidates")
    options = tuple(
        _parse_option(
            option.declarative_object,
            option.compilation,
            content_selection,
        )
        for option in reference_options
    )
    branch = base.connector.registry.templates[0].language_branch
    return GroundedAnswerReferenceParserCatalog(
        renderer,
        branch,
        candidates[0].source,
        candidates[0].scope,
        candidates,
        options,
    )


# object-model: parser
class GroundedAnswerReferenceSurfaceParser:
    """按完整 grammar 匹配实际输出，并独立恢复 reference 执行。"""

    def __init__(
            self,
            protocol: GroundedAnswerParserProtocol,
            catalog: GroundedAnswerReferenceParserCatalog,
            ) -> None:
        if not isinstance(protocol, GroundedAnswerParserProtocol):
            raise TypeError("reference parser protocol 类型错误")
        if not isinstance(catalog, GroundedAnswerReferenceParserCatalog):
            raise TypeError("reference parser catalog 类型错误")
        self.protocol = protocol
        self.catalog = catalog

    def _failure(
            self,
            reason: ObjectIdentity,
            request: GenerationSurfaceParseRequest,
            *detail: int,
            ) -> GenerationSurfaceParseResult:
        """构造绑定同次 parse request 的 typed failure。"""
        return GenerationSurfaceParseResult(
            reason,
            (1, *request.stable_key(), *detail),
        )

    def _matched_option(
            self,
            request: GenerationSurfaceParseRequest,
            ) -> GroundedAnswerReferenceParseOption | None:
        """只按实际 units 与公开运行归属返回唯一 grammar 项。"""
        catalog = self.catalog
        if (request.renderer != catalog.renderer
                or request.branch != catalog.branch
                or request.source != catalog.source
                or request.scope != catalog.scope):
            return None
        matches = tuple(
            item for item in catalog.options if item.units == request.units)
        if len(matches) > 1:
            raise GroundedAnswerReferenceParserError(
                "reference parser units 命中多个 strategy grammar")
        return None if not matches else matches[0]

    def parse(
            self,
            request: GenerationSurfaceParseRequest,
            ) -> GenerationSurfaceParseResult:
        """只凭 units/branch/source/scope 恢复 compiled candidates。"""
        if not isinstance(request, GenerationSurfaceParseRequest):
            raise TypeError("reference parser request 类型错误")
        option = self._matched_option(request)
        if option is None:
            relevant = (
                request.renderer == self.catalog.renderer
                and request.branch == self.catalog.branch
                and request.source == self.catalog.source
                and request.scope == self.catalog.scope
            )
            claim_units = self.catalog.options[0].claim_units
            counts = tuple(
                _occurrences(request.units, units) for units in claim_units)
            if not relevant or any(count == 0 for count in counts):
                reason = self.protocol.missing_claim
            elif any(count > 1 for count in counts):
                reason = self.protocol.duplicate_claim
            else:
                reason = self.protocol.no_match
            return self._failure(reason, request, int(relevant))
        candidates = option.candidates
        recovered = tuple(
            RecoveredGenerationProposition(
                candidate.stable_key(),
                candidate.proposition,
                candidate.source,
                candidate.scope,
                (1, index, *candidate.stable_key()),
            )
            for index, candidate in enumerate(candidates, start=1)
        )
        cited_sources = tuple(sorted({
            source_ref
            for candidate in candidates
            for source_ref in candidate.citation_sources
        }, key=lambda item: item.stable_key()))
        observation = GenerationSurfaceObservation(
            request.stable_key(),
            option.representations,
            self.catalog.branch,
            self.protocol.answer_stance,
            self.catalog.source,
            self.catalog.scope,
            recovered,
            (),
            cited_sources,
            option.structure_payload,
            (),
            (2, len(candidates), len(option.representations)),
        )
        return GenerationSurfaceParseResult(
            self.protocol.succeeded,
            (2, *request.stable_key()),
            observation,
        )

    def recover_reference(
            self,
            request: GenerationSurfaceParseRequest,
            preview: GenerationSurfacePreview,
            ) -> GroundedAnswerRecoveredReference:
        """从 actual units、preview proposal 与 sentence instance 恢复引用。"""
        if not isinstance(request, GenerationSurfaceParseRequest):
            raise TypeError("reference recovery request 类型错误")
        if not isinstance(preview, GenerationSurfacePreview):
            raise TypeError("reference recovery preview 类型错误")
        option = self._matched_option(request)
        if option is None or not self.parse(request).succeeded:
            raise GroundedAnswerReferenceParserError(
                "reference recovery 缺唯一实际 grammar match")
        surface_request = preview.request
        if (not preview.complete
                or surface_request.branch != self.catalog.branch
                or surface_request.structure.syntax.stable_key()
                != option.syntax_key
                or preview.representations != option.representations):
            raise GroundedAnswerReferenceParserError(
                "reference recovery 的 units/syntax/Representation 漂移")
        matches = tuple(
            item for item in preview.slots
            if (item.directive.sentence == option.reference_sentence
                and item.value.slot == option.reference_slot)
        )
        if len(matches) != 1:
            raise GroundedAnswerReferenceParserError(
                "reference recovery 未命中唯一 sentence/slot")
        slot = matches[0]
        if (slot.value.filler != option.reference_origin
                or slot.representation is None
                or slot.surface is None):
            raise GroundedAnswerReferenceParserError(
                "reference recovery 缺 origin/direct surface proposal")
        if option.antecedent is None:
            if (slot.antecedent is not None
                    or slot.antecedent_candidate_key
                    or slot.reference is not None):
                raise GroundedAnswerReferenceParserError(
                    "explicit grammar 实际执行了 reference proposal")
            antecedent_key = ()
            antecedent = None
            reference_key = ()
        else:
            expected = option.antecedent.proposition.template
            if (slot.antecedent_candidate_key
                    != option.antecedent.stable_key()
                    or slot.antecedent != expected
                    or slot.reference is None
                    or slot.reference.result.selected is None
                    or slot.reference.result.selected.value != expected):
                raise GroundedAnswerReferenceParserError(
                    "antecedent grammar 未实际唯一命中前序命题")
            antecedent_key = option.antecedent.stable_key()
            antecedent = expected
            reference_key = slot.reference.stable_key()
        return GroundedAnswerRecoveredReference(
            request.stable_key(),
            option.strategy_object,
            option.reference_sentence,
            option.reference_slot,
            option.reference_origin,
            slot.representation,
            self.catalog.source,
            self.catalog.scope,
            antecedent_key,
            antecedent,
            reference_key,
            slot.surface.stable_key(),
            (3, *option.strategy_object.stable_key()),
        )


__all__ = [
    "GroundedAnswerRecoveredReference",
    "GroundedAnswerReferenceParseOption",
    "GroundedAnswerReferenceParserCatalog",
    "GroundedAnswerReferenceParserError",
    "GroundedAnswerReferenceSurfaceParser",
    "build_grounded_answer_reference_parser_catalog",
]
