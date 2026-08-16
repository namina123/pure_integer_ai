"""从真实多句 units 独立恢复两个 grounded Proposition。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceObservation,
    GenerationSurfaceParseRequest,
    GenerationSurfaceParseResult,
    RecoveredGenerationProposition,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)


# object-model: exception
class GroundedAnswerReferenceParserError(ValueError):
    """reference parser catalog 或实际 units 不满足受限合同。"""


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
class GroundedAnswerReferenceParserCatalog:
    """保存一个已编译 strategy 的实际 units 与独立候选恢复资料。"""

    compilation: GroundedAnswerReferenceCompilation
    renderer: ObjectIdentity
    representations: tuple[ObjectIdentity, ...]
    units: tuple[int, ...]
    claim_units: tuple[tuple[int, ...], ...]
    structure_payload: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(
                self.compilation, GroundedAnswerReferenceCompilation):
            raise TypeError("reference parser compilation 类型错误")
        _instruction(self.renderer, where="reference parser renderer")
        if (not isinstance(self.representations, tuple)
                or not self.representations):
            raise GroundedAnswerReferenceParserError(
                "reference parser representations 不能为空")
        if (not isinstance(self.units, tuple) or not self.units
                or any(type(item) is not int for item in self.units)):
            raise GroundedAnswerReferenceParserError(
                "reference parser units 非法")
        if (not isinstance(self.claim_units, tuple)
                or len(self.claim_units) != 2
                or any(not item for item in self.claim_units)):
            raise GroundedAnswerReferenceParserError(
                "reference parser 必须保存两个 claim unit 片段")
        if (not isinstance(self.structure_payload, tuple)
                or not self.structure_payload):
            raise GroundedAnswerReferenceParserError(
                "reference parser structure payload 不能为空")


def build_grounded_answer_reference_parser_catalog(
        compilation: GroundedAnswerReferenceCompilation,
        renderer: ObjectIdentity,
        ) -> GroundedAnswerReferenceParserCatalog:
    """只从 compiled aliases/candidates 构造 parser，不读取 teacher surface。"""
    if not isinstance(compilation, GroundedAnswerReferenceCompilation):
        raise TypeError("reference parser compilation 类型错误")
    _instruction(renderer, where="reference parser renderer")
    aliases = tuple(sorted(
        (
            alias
            for sentence in compilation.sentences
            for alias in sentence.aliases
        ),
        key=lambda item: item.part_ordinal,
    ))
    ordinals = tuple(item.part_ordinal for item in aliases)
    if ordinals != tuple(range(len(aliases))):
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
    return GroundedAnswerReferenceParserCatalog(
        compilation,
        renderer,
        representations,
        units,
        claim_units,
        structure_payload,
    )


# object-model: parser
class GroundedAnswerReferenceSurfaceParser:
    """完整匹配 selected strategy units，并恢复两个 actual candidates。"""

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

    def parse(
            self,
            request: GenerationSurfaceParseRequest,
            ) -> GenerationSurfaceParseResult:
        """只凭 units/branch/source/scope 恢复 compiled candidates。"""
        if not isinstance(request, GenerationSurfaceParseRequest):
            raise TypeError("reference parser request 类型错误")
        catalog = self.catalog
        compilation = catalog.compilation
        candidates = tuple(item.candidate for item in compilation.claims)
        source = candidates[0].source
        scope = candidates[0].scope
        branch = compilation.connector.registry.templates[0].language_branch
        relevant = (
            request.renderer == catalog.renderer
            and request.branch == branch
            and request.source == source
            and request.scope == scope
        )
        if not relevant or request.units != catalog.units:
            counts = tuple(
                _occurrences(request.units, units)
                for units in catalog.claim_units
            )
            if not relevant or any(count == 0 for count in counts):
                reason = self.protocol.missing_claim
            elif any(count > 1 for count in counts):
                reason = self.protocol.duplicate_claim
            else:
                reason = self.protocol.no_match
            return self._failure(reason, request, int(relevant))
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
            catalog.representations,
            branch,
            self.protocol.answer_stance,
            source,
            scope,
            recovered,
            (),
            cited_sources,
            catalog.structure_payload,
            (),
            (2, len(candidates), len(catalog.representations)),
        )
        return GenerationSurfaceParseResult(
            self.protocol.succeeded,
            (2, *request.stable_key()),
            observation,
        )


__all__ = [
    "GroundedAnswerReferenceParserCatalog",
    "GroundedAnswerReferenceParserError",
    "GroundedAnswerReferenceSurfaceParser",
    "build_grounded_answer_reference_parser_catalog",
]
