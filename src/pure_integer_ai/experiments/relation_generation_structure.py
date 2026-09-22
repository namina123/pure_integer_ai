"""将现役关系 Span 图编译为既有 connector 的角色、语言原子和顺序课程。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity, concept_identity, language_atom_identity, minimal_instruction_identity,
    representation_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import StructureSlotDefinition
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSlotBinding,
    LanguageConnectorSurfaceDirective, LanguageConnectorValueProtocol,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateRealizationBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_LITERAL
from pure_integer_ai.experiments.relation_generation_protocol import (
    RELATION_CONNECTOR_PROFILE, ordinal_definition,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationGenerationInput, ActiveRelationSurface, RelationSurfaceFrame,
)


_CONNECTOR_NAMESPACE = 20916
_FRAME_HASHER = Hasher("trained.relation.connector.structure.v1")


def _pack(values: tuple[int, ...]) -> tuple[int, ...]:
    """保留每个可变长图身份的边界。"""
    return len(values), *values


@dataclass(frozen=True, slots=True)
class RelationConnectorCourse:
    """纯角色课程；执行仍由既有 G-02/S-07/R-01/G-03 owner 完成。"""

    template: LanguageGenerationConnectorTemplate
    values: LanguageConnectorValueProtocol
    aliases: tuple[GenerationCandidateRealizationBinding, ...]
    positions: tuple[tuple[int, int], ...]
    source: ActiveRelationGenerationInput
    qualifiers: tuple[int, ...]


def compile_relation_connector(fact: ActiveRelationSurface, frame: RelationSurfaceFrame,
                               source: ActiveRelationGenerationInput, branch: ObjectIdentity,
                               family: tuple[int, ...], values: LanguageConnectorValueProtocol,
                               surface_protocol) -> RelationConnectorCourse:
    """用真实角色和间隔编译既有模板，不创建整句命题 Representation。

    课程保留来源中的成员顺序和真实区间；角色按 Role+ordinal 动态取 filler，
    连接结构成为有完整来源引用的 LanguageAtom。任何来源/成员漂移都拒绝物化。
    """
    if (frame.proposition != fact.proposition or frame.source_hash != fact.source_hash
            or source.proposition.definition.proposition != fact.proposition
            or tuple(item.role for item in fact.bindings) != frame.roles):
        raise ValueError("角色生成课程与训练后图成员不符")
    parts = []
    position = frame.envelope_start
    for index, gap in enumerate(frame.gaps):
        if gap:
            atom_values = tuple(map(ord, gap))
            atom = language_atom_identity(branch, (
                RELATION_CONNECTOR_PROFILE, 1, *_pack(fact.proposition.stable_key()),
                index, *_pack(atom_values)))
            parts.append((atom, None, atom_values, position, position + len(atom_values)))
            position += len(atom_values)
        if index < len(fact.bindings):
            binding = fact.bindings[index]
            candidates = tuple(item for item in source.proposition.definition.bindings
                               if item.role == binding.role and item.filler == binding.filler)
            if len(candidates) != 1 or binding.start != position:
                raise ValueError("角色生成课程没有唯一原始 RoleBinding/区间")
            parts.append((binding.filler, candidates[0], tuple(map(ord, binding.surface)),
                          binding.start, binding.end))
            position = binding.end
    if position != frame.envelope_end or not parts:
        raise ValueError("角色生成课程没有完整覆盖训练框架")
    full_key = (RELATION_CONNECTOR_PROFILE, 1, *_pack(fact.proposition.stable_key()),
                *_pack(fact.predicate.stable_key()), frame.source_hash,
                *(v for item in parts for v in _pack((item[3], item[4], *item[2]))))
    pattern_id = _FRAME_HASHER.h63(full_key) or 1
    pattern = (_CONNECTOR_NAMESPACE, 2, pattern_id, pattern_id)
    structure_key = (_CONNECTOR_NAMESPACE, 3, pattern_id, pattern_id)

    def structure(*suffix: int) -> ObjectIdentity:
        """在原分支 owner/version 下登记 S-07 结构身份。"""
        return structure_concept_identity((*structure_key, *suffix),
                                           owner=branch.owner, versions=branch.versions)

    def instruction(*suffix: int) -> ObjectIdentity:
        """在原分支 owner/version 下登记执行指令。"""
        return minimal_instruction_identity((*pattern, *suffix),
                                             owner=branch.owner, versions=branch.versions)

    def owned(*suffix: int) -> ObjectIdentity:
        """以现有 connector 身份布局登记其成员，不用摘要代替成员拓扑。"""
        return structure_concept_identity((*pattern, *suffix),
                                           owner=branch.owner, versions=branch.versions)

    ordinals = {item.ordinal: item for item in values.ordinals}
    for _filler, binding, _tokens, _start, _end in parts:
        if binding is not None:
            declared = ordinal_definition(branch, binding.ordinal)
            if binding.ordinal in ordinals and ordinals[binding.ordinal] != declared:
                raise ValueError("角色坐标协议发生竞争")
            ordinals[binding.ordinal] = declared
    values = replace(values, ordinals=tuple(ordinals.values()))
    slot_type = concept_identity((*structure_key, 2), owner=branch.owner, versions=branch.versions)
    slots = tuple(StructureSlotDefinition(
        structure(1), structure(10, ordinal),
        role_identity((*structure_key, 11, ordinal), owner=branch.owner, versions=branch.versions),
        slot_type) for ordinal in range(1, len(parts) + 1))
    bindings = []
    directives = []
    aliases = []
    forming = tuple(sorted(item.stable_key() for item in source.evidence))
    for ordinal, (slot, part) in enumerate(zip(slots, parts, strict=True), 1):
        filler, binding, tokens, start, end = part
        if fact.evidence_surface[start:end] != "".join(map(chr, tokens)):
            raise ValueError("角色生成课程表层不在真实训练 Span 中")
        bindings.append(LanguageConnectorSlotBinding(
            owned(30, ordinal), slot.slot,
            values.constant_source if binding is None else values.role_filler_source,
            role=None if binding is None else binding.role,
            ordinal=None if binding is None else ordinals[binding.ordinal].instruction,
            constant=filler if binding is None else None))
        directives.append(LanguageConnectorSurfaceDirective(
            owned(40, ordinal), slot.slot, surface_protocol.emit_action,
            instruction(41, ordinal), owned(42, ordinal), ()))
        aliases.append(GenerationCandidateRealizationBinding(
            filler, representation_identity(family, tokens, owner=branch.owner, versions=branch.versions),
            RULE_LITERAL, forming))
    family_structure = structure_concept_identity((
        RELATION_CONNECTOR_PROFILE, 2, *_pack(fact.predicate.stable_key()),
        *(v for item in source.proposition.definition.bindings
          for v in (item.ordinal, *_pack(item.role.stable_key())))),
        owner=branch.owner, versions=branch.versions)
    template = LanguageGenerationConnectorTemplate(
        owned(1), branch, family_structure, fact.predicate, structure(3), structure(1),
        slots, tuple(bindings), structure(4),
        tuple(structure(60, ordinal) for ordinal in range(1, len(parts))),
        structure(5), (), minimal_instruction_identity((*structure_key, 6),
            owner=branch.owner, versions=branch.versions),
        minimal_instruction_identity((*structure_key, 7), owner=branch.owner, versions=branch.versions),
        tuple(directives))
    return RelationConnectorCourse(template, values, tuple(aliases),
                                   tuple((item[3], item[4]) for item in parts), source, full_key)
