"""从实际 Unicode units 独立恢复单 claim grounded-answer surface。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
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
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.representation_rendering import (
    representation_parts,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorCompilation,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
    PATTERN_LITERAL,
)


# object-model: exception
class GroundedAnswerParserError(ValueError):
    """parser catalog 泄漏边界、结构或候选归属不完整。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerParserError(f"{where} 必须是非空严格整数 tuple")
    return value


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 parser reason/stance/renderer 使用最小指令。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedAnswerParserError(f"{where} 必须是 MinimalInstruction")
    return value


def _count_occurrences(
        units: tuple[int, ...], needle: tuple[int, ...],
        ) -> int:
    """计算完整整数片段出现次数，用于区分 claim 遗漏与重复。"""
    if len(needle) > len(units):
        return 0
    return sum(
        units[index:index + len(needle)] == needle
        for index in range(len(units) - len(needle) + 1)
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerParserProtocol:
    """注入成功、四类 parse 失败和 ANSWER stance 身份。"""

    succeeded: ObjectIdentity
    no_match: ObjectIdentity
    ambiguous: ObjectIdentity
    missing_claim: ObjectIdentity
    duplicate_claim: ObjectIdentity
    answer_stance: ObjectIdentity

    def __post_init__(self) -> None:
        values = (
            self.succeeded,
            self.no_match,
            self.ambiguous,
            self.missing_claim,
            self.duplicate_claim,
            self.answer_stance,
        )
        if len(set(values)) != len(values):
            raise GroundedAnswerParserError("parser protocol identity 不得重复")
        for value in values:
            _instruction(value, where="parser protocol identity")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerParserSlot:
    """一个 grammar slot 的 Representation、units 和 claim/literal 类型。"""

    representation: ObjectIdentity
    units: tuple[int, ...]
    part_kind: str

    def __post_init__(self) -> None:
        if (not isinstance(self.representation, ObjectIdentity)
                or self.representation.object_kind != OBJECT_REPRESENTATION):
            raise TypeError("parser slot representation 类型错误")
        _strict_key(self.units, where="parser slot units")
        if self.part_kind not in {PATTERN_LITERAL, PATTERN_CLAIM}:
            raise GroundedAnswerParserError("parser slot part kind 非法")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerParserGrammar:
    """独立 parser 使用的结构 grammar，不保存 answer plan 或 surface label。"""

    pattern_id: int
    renderer: ObjectIdentity
    branch: ObjectIdentity
    source: SourceRef
    scope: ScopeIdentity
    structure: ObjectIdentity
    slots: tuple[GroundedAnswerParserSlot, ...]
    candidate_key: tuple[int, ...]
    proposition: BoundProposition
    cited_sources: tuple[SourceRef, ...]
    support_teacher_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedAnswerParserError("parser pattern id 非法")
        _instruction(self.renderer, where="parser renderer")
        if (not isinstance(self.branch, ObjectIdentity)
                or self.branch.object_kind != OBJECT_LANGUAGE_BRANCH):
            raise TypeError("parser branch 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("parser source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("parser scope 类型错误")
        if (not isinstance(self.structure, ObjectIdentity)
                or self.structure.object_kind != OBJECT_STRUCTURE_CONCEPT):
            raise TypeError("parser structure 类型错误")
        if (not isinstance(self.slots, tuple) or not self.slots
                or any(not isinstance(item, GroundedAnswerParserSlot)
                       for item in self.slots)):
            raise GroundedAnswerParserError("parser grammar slots 不能为空")
        if sum(item.part_kind == PATTERN_CLAIM for item in self.slots) != 1:
            raise GroundedAnswerParserError("首轮 parser grammar 必须恰有一个 claim")
        _strict_key(self.candidate_key, where="parser candidate key")
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("parser proposition 类型错误")
        if (not isinstance(self.cited_sources, tuple)
                or any(not isinstance(item, SourceRef)
                       for item in self.cited_sources)
                or len(set(self.cited_sources)) != len(self.cited_sources)):
            raise GroundedAnswerParserError("parser cited source 非规范")
        if not self.cited_sources:
            raise GroundedAnswerParserError(
                "首轮 parser citation 必须保留 actual Evidence source")
        if (not isinstance(self.support_teacher_keys, tuple)
                or not self.support_teacher_keys
                or any(not isinstance(key, tuple) or not key
                       or any(type(value) is not int for value in key)
                       for key in self.support_teacher_keys)):
            raise GroundedAnswerParserError("parser grammar 缺 teacher Evidence 追溯")
        object.__setattr__(self, "cited_sources", tuple(sorted(
            self.cited_sources, key=lambda item: item.stable_key())))

    @property
    def units(self) -> tuple[int, ...]:
        """返回按 grammar slot 顺序连接的实际输出单元。"""
        return tuple(value for slot in self.slots for value in slot.units)

    @property
    def representations(self) -> tuple[ObjectIdentity, ...]:
        """返回与 units 同序的可恢复 Representation。"""
        return tuple(item.representation for item in self.slots)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerParserCatalog:
    """保存同一候选下全部合法 pattern grammar。"""

    grammars: tuple[GroundedAnswerParserGrammar, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.grammars, tuple) or not self.grammars
                or any(not isinstance(item, GroundedAnswerParserGrammar)
                       for item in self.grammars)):
            raise GroundedAnswerParserError("parser catalog 不能为空")
        ids = tuple(item.pattern_id for item in self.grammars)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise GroundedAnswerParserError("parser grammar id 非唯一递增")


def build_grounded_answer_parser_catalog(
        compilation: GroundedAnswerConnectorCompilation,
        candidate: GenerationCandidate,
        renderer: ObjectIdentity,
        ) -> GroundedAnswerParserCatalog:
    """从已学 connector grammar 和独立 typed 候选建立只读 parser catalog。"""
    if not isinstance(compilation, GroundedAnswerConnectorCompilation):
        raise TypeError("parser compilation 类型错误")
    if not isinstance(candidate, GenerationCandidate):
        raise TypeError("parser candidate 类型错误")
    _instruction(renderer, where="parser renderer")
    grammars = []
    candidate_key = candidate.stable_key()
    cited_sources = candidate.citation_sources
    for variant in compilation.variants:
        template = variant.template
        if (template.proposition_structure != candidate.proposition.structure
                or template.predicate != candidate.proposition.predicate):
            raise GroundedAnswerParserError(
                "parser candidate 与 compiled connector match key 漂移")
        alias_by_ordinal = {
            item.part_ordinal: item for item in variant.aliases}
        if len(alias_by_ordinal) != len(variant.aliases):
            raise GroundedAnswerParserError("parser alias ordinal 重复")
        slots = tuple(
            GroundedAnswerParserSlot(
                alias_by_ordinal[index].representation,
                representation_parts(
                    alias_by_ordinal[index].representation)[1],
                alias_by_ordinal[index].part_kind,
            )
            for index in range(len(variant.aliases))
        )
        claim_aliases = tuple(
            item for item in variant.aliases
            if item.part_kind == PATTERN_CLAIM)
        if (len(claim_aliases) != 1
                or claim_aliases[0].filler != candidate.proposition.template):
            raise GroundedAnswerParserError(
                "parser claim alias 未绑定独立候选 Proposition")
        grammars.append(GroundedAnswerParserGrammar(
            variant.option.pattern_id,
            renderer,
            template.language_branch,
            candidate.source,
            candidate.scope,
            template.structure,
            slots,
            candidate_key,
            candidate.proposition,
            cited_sources,
            variant.option.support_teacher_keys,
        ))
    return GroundedAnswerParserCatalog(tuple(sorted(
        grammars, key=lambda item: item.pattern_id)))


# object-model: parser
class GroundedAnswerSurfaceParser:
    """只按受限 parse request 和独立 grammar catalog 恢复 typed 观察。"""

    def __init__(
            self,
            protocol: GroundedAnswerParserProtocol,
            catalog: GroundedAnswerParserCatalog,
            ) -> None:
        if not isinstance(protocol, GroundedAnswerParserProtocol):
            raise TypeError("grounded parser protocol 类型错误")
        if not isinstance(catalog, GroundedAnswerParserCatalog):
            raise TypeError("grounded parser catalog 类型错误")
        self.protocol = protocol
        self.catalog = catalog

    def _failure(
            self,
            reason: ObjectIdentity,
            request: GenerationSurfaceParseRequest,
            *detail: int,
            ) -> GenerationSurfaceParseResult:
        """构造绑定实际受限请求的 typed parse failure。"""
        return GenerationSurfaceParseResult(
            reason,
            (1, *request.stable_key(), *detail),
        )

    def parse(
            self,
            request: GenerationSurfaceParseRequest,
            ) -> GenerationSurfaceParseResult:
        """完整匹配唯一 grammar；不读取 generation execution 或 answer plan。"""
        if not isinstance(request, GenerationSurfaceParseRequest):
            raise TypeError("grounded parser request 类型错误")
        relevant = tuple(
            item for item in self.catalog.grammars
            if (item.renderer == request.renderer
                and item.branch == request.branch
                # learned candidate source may differ from the current query
                # source; source attribution is verified from the recovered
                # proposition and the postcheck requirement below.
                and item.scope == request.scope)
        )
        matches = tuple(item for item in relevant if item.units == request.units)
        if len(matches) > 1:
            return self._failure(
                self.protocol.ambiguous, request, len(matches))
        if not matches:
            claim_units = {
                slot.units
                for grammar in relevant
                for slot in grammar.slots
                if slot.part_kind == PATTERN_CLAIM
            }
            counts = tuple(
                _count_occurrences(request.units, units)
                for units in claim_units)
            if not counts or max(counts, default=0) == 0:
                reason = self.protocol.missing_claim
            elif max(counts) > 1:
                reason = self.protocol.duplicate_claim
            else:
                reason = self.protocol.no_match
            return self._failure(reason, request, len(relevant))
        grammar = matches[0]
        recovered = RecoveredGenerationProposition(
            grammar.candidate_key,
            grammar.proposition,
            grammar.source,
            grammar.scope,
            (1, grammar.pattern_id),
        )
        observation = GenerationSurfaceObservation(
            request.stable_key(),
            grammar.representations,
            request.branch,
            self.protocol.answer_stance,
            request.source,
            request.scope,
            (recovered,),
            (),
            grammar.cited_sources,
            grammar.structure.stable_key(),
            (),
            (1, grammar.pattern_id, len(grammar.slots)),
        )
        return GenerationSurfaceParseResult(
            self.protocol.succeeded,
            (2, grammar.pattern_id, *request.stable_key()),
            observation,
        )


__all__ = [
    "GroundedAnswerParserCatalog",
    "GroundedAnswerParserError",
    "GroundedAnswerParserGrammar",
    "GroundedAnswerParserProtocol",
    "GroundedAnswerParserSlot",
    "GroundedAnswerSurfaceParser",
    "build_grounded_answer_parser_catalog",
]
