"""把冻结的整数对话观察编译为现有 connector、语言原子和顺序课程。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    SourceRef, concept_identity, language_atom_identity, minimal_instruction_identity,
    representation_identity, span_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import StructureSlotDefinition
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSlotBinding, LanguageConnectorSurfaceDirective,
    LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import (
    GenerationCandidateRealizationBinding,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_LITERAL
from pure_integer_ai.experiments.ph2_grounded_answer_order import SourceGenerationOrderCourse
from pure_integer_ai.experiments.relation_generation_protocol import ordinal_definition
from pure_integer_ai.experiments.response_generation_graph import (
    RESPONSE_CONNECTOR_PROFILE, RecoveredResponseConnector,
    generic_response_context_marker, generic_response_context_role,
    generic_response_context_type, generic_response_graph_role,
    generic_response_graph_type,
)
from pure_integer_ai.experiments.unknown_response_course import (
    RESPONSE_PART_GRAPH_ROLE,
    RESPONSE_PART_LITERAL,
    UnknownResponseCourse,
)
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_RESPONSE_VARIANT,
)


_SOURCE_HASHER = Hasher("dialogue.structure.observation.source.v1")


@dataclass(frozen=True, slots=True)
class ResponseConnectorCourse:
    """来源观察、语言原子 realization、顺序及可回读动作关系的同一份课程。"""

    response: RecoveredResponseConnector
    order: SourceGenerationOrderCourse
    aliases: tuple[GenerationCandidateRealizationBinding, ...]
    values: object
    tokens: tuple[int, ...]


def compile_response_course(record: tuple, *, digest: tuple[int, ...], fact,
                            parent_template, values, family, content, surface) -> ResponseConnectorCourse:
    """只按显式角色标注和来源顺序编译，不识别问式、不硬编码语言内容。

    record 为 (观察序号, 父框架序号, stance 协议序号, parts)。part 为
    (1, 码点序列) 或 (2, 来源框架角色坐标, 码点序列)。全量来源与标注进入证据。
    """
    if (type(record) is not tuple or len(record) != 4
            or any(type(v) is not int or v <= 0 for v in record[:3])
            or record[2] not in (2, 3, 4, 5)
            or type(record[3]) is not tuple or len(record[3]) < 3):
        raise ValueError("对话结构观察协议不完整或试图训练事实答案")
    if len(digest) != 32 or any(type(v) is not int or not 0 <= v <= 255 for v in digest):
        raise ValueError("对话结构来源必须具有完整 SHA-256 字节身份")
    branch = parent_template.language_branch
    owner = dict(owner=branch.owner, versions=branch.versions)
    source = SourceRef(RESPONSE_CONNECTOR_PROFILE, _SOURCE_HASHER.h63(digest) or 1,
                       record[0], branch.owner, branch.versions)
    parts = record[3]
    positions = []
    observed_roles = []
    tokens = []
    encoded_parts = []
    for part in parts:
        if (type(part) is not tuple or part[0] not in (1, 2)
                or len(part) != (2 if part[0] == 1 else 3)
                or type(part[-1]) is not tuple or not part[-1]
                or any(type(v) is not int or not 0 <= v <= 0x10FFFF or 0xD800 <= v <= 0xDFFF
                       for v in part[-1])):
            raise ValueError("对话成员必须为非空完整整数原子或角色观察")
        if part[0] == 2:
            index = part[1]
            if type(index) is not int or not 0 <= index < len(fact.bindings):
                raise ValueError("对话角色坐标超出来源框架")
            if tuple(map(ord, fact.bindings[index].surface)) != part[-1]:
                raise ValueError("对话角色标注与真实图内例值不一致")
            observed_roles.append(index)
        positions.append((len(tokens), len(tokens) + len(part[-1])))
        tokens.extend(part[-1])
        encoded_parts.append(pack_record(1, part[:-1], part[-1]))
    if sorted(observed_roles) != list(range(len(fact.bindings))):
        raise ValueError("对话结构必须覆盖全部角色，不得重复或遗失信息")
    stance = content.stances()[record[2] - 1]
    annotation = pack_record(RESPONSE_CONNECTOR_PROFILE, digest, record[:3],
                             source.stable_key(), stance.stable_key(),
                             fact.proposition.stable_key(), parent_template.stable_key(),
                             pack_record(1, *encoded_parts))
    key_id = _SOURCE_HASHER.h63(annotation) or 1
    base = (20916, 2, key_id, key_id)
    structure_base = (20916, 3, key_id, key_id)

    def node(*suffix):
        """保留现有 connector 布局，成员拓扑不由摘要替代。"""
        return structure_concept_identity((*base, *suffix), **owner)

    def structure(*suffix):
        """沿原 S-07 结构身份布局登记对话句式。"""
        return structure_concept_identity((*structure_base, *suffix), **owner)

    def instruction(*suffix):
        """来源确定的指令身份只承载执行类别，不包含语言分支规则。"""
        return minimal_instruction_identity((*base, *suffix), **owner)

    ordinals = {item.ordinal: item for item in values.ordinals}
    ordinals.update({index: ordinal_definition(branch, index) for index in observed_roles})
    values = replace(values, ordinals=tuple(ordinals.values()))
    slots, bindings, directives, aliases, examples = [], [], [], [], []
    marker_slot = None
    slot_type = concept_identity((*structure_base, 2), **owner)
    for ordinal, (part, position) in enumerate(zip(parts, positions, strict=True), 1):
        dynamic = part[0] == 2
        role = (fact.bindings[part[1]].role if dynamic
                else role_identity((*structure_base, 11, ordinal), **owner))
        slot = StructureSlotDefinition(structure(1), structure(10, ordinal), role, slot_type)
        span = span_identity(source, members=(position,))
        atom = None if dynamic else language_atom_identity(branch, (
            RESPONSE_CONNECTOR_PROFILE, 2, *source.stable_key(), *position))
        if atom is not None:
            if marker_slot is None:
                marker_slot = slot.slot
            aliases.append(GenerationCandidateRealizationBinding(
                atom, representation_identity(family, part[-1], **owner), RULE_LITERAL, (annotation,)))
        slots.append(slot)
        bindings.append(LanguageConnectorSlotBinding(
            node(30, ordinal), slot.slot, values.role_filler_source if dynamic else values.constant_source,
            role=role if dynamic else None,
            ordinal=ordinals[part[1]].instruction if dynamic else None, constant=atom))
        directives.append(LanguageConnectorSurfaceDirective(
            node(40, ordinal), slot.slot, surface.emit_action, instruction(41, ordinal), node(42, ordinal), ()))
        examples.append((slot.slot, span))
    if marker_slot is None:
        raise ValueError("对话动作缺少来源化语言成员")
    template = LanguageGenerationConnectorTemplate(
        node(1), branch, parent_template.proposition_structure, parent_template.predicate,
        structure(3), structure(1), tuple(slots), tuple(bindings), structure(4),
        tuple(structure(60, index) for index in range(1, len(slots))), structure(5), (),
        minimal_instruction_identity((*structure_base, 6), **owner),
        minimal_instruction_identity((*structure_base, 7), **owner), tuple(directives))
    response = RecoveredResponseConnector(template, stance, marker_slot, source, document_scope(source),
                                           tuple(sorted(examples, key=lambda pair: pair[0].stable_key())),
                                           (annotation,))
    return ResponseConnectorCourse(response, SourceGenerationOrderCourse(
        template, source, response.scope, tuple(positions), annotation, RESPONSE_CONNECTOR_PROFILE),
        tuple(aliases), values, tuple(tokens))


def compile_generic_response_course(
        course: UnknownResponseCourse,
        *,
        digest: tuple[int, ...],
        branch,
        values,
        family,
        content,
        surface,
        variant_identity_hashes: tuple[int, ...],
        ) -> tuple[ResponseConnectorCourse, ...]:
    """Compile licensed integer variants into non-proposition connector graphs."""
    if not isinstance(course, UnknownResponseCourse):
        raise TypeError("generic response course type differs")
    if len(digest) != 32 or any(type(value) is not int or not 0 <= value <= 255
                                for value in digest):
        raise ValueError("generic response course requires a complete SHA-256 identity")
    if (type(variant_identity_hashes) is not tuple
            or len(variant_identity_hashes) != len(course.variants)
            or any(type(value) is not int or value <= 0
                   for value in variant_identity_hashes)
            or len(set(variant_identity_hashes)) != len(variant_identity_hashes)):
        raise ValueError("generic response semantic variant identities differ")
    owner = dict(owner=branch.owner, versions=branch.versions)
    context_ordinal = 0
    ordinal = ordinal_definition(branch, context_ordinal)
    ordinals = {item.ordinal: item for item in values.ordinals}
    ordinals[context_ordinal] = ordinal
    for variant in range(1, len(course.variants) + 1):
        for part in course.parts_for(variant):
            if part.kind == RESPONSE_PART_GRAPH_ROLE:
                ordinals[part.ordinal] = ordinal_definition(
                    branch, part.ordinal)
    values = replace(values, ordinals=tuple(ordinals.values()))
    # 相同 literal 成员在整批课程中只物化一个 LanguageAtom/R-01 route。
    # connector 可以多次引用该对象，但不能因来源 variant 不同把同一整数
    # 表示重复录入成多个图对象。
    literal_atoms: dict[tuple[int, ...], object] = {}
    context_role = generic_response_context_role(branch)
    context_type = generic_response_context_type(branch)
    context_marker = generic_response_context_marker(branch)
    result = []
    for variant, units in enumerate(course.variants, 1):
        parts = course.parts_for(variant)
        semantic_variant_hash = variant_identity_hashes[variant - 1]
        source = SourceRef(
            RESPONSE_CONNECTOR_PROFILE,
            _SOURCE_HASHER.h63(pack_record(1, digest, (variant,), course.stable_key())) or 1,
            variant,
            branch.owner,
            branch.versions,
        )
        stance = content.unknown
        # The complete condition/parts identity is stored once in identity
        # kind 13.  Every downstream assertion cites this resolvable identity
        # instead of copying the whole course into every qualifier.
        annotation = pack_record(
            RESPONSE_CONNECTOR_PROFILE,
            digest,
            (variant,),
            source.stable_key(),
            (IDENTITY_RESPONSE_VARIANT, semantic_variant_hash),
        )
        key_id = _SOURCE_HASHER.h63(annotation) or 1
        base = (91563, 1, key_id, key_id)
        structure_base = (91563, 2, key_id, key_id)

        def node(*suffix):
            return structure_concept_identity((*base, *suffix), **owner)

        def structure(*suffix):
            return structure_concept_identity((*structure_base, *suffix), **owner)

        def instruction(*suffix):
            return minimal_instruction_identity((*base, *suffix), **owner)

        slots = []
        bindings = []
        directives = []
        aliases = []
        examples = []
        positions = []
        tokens = []
        marker_slot = None
        graph_slots = []
        slot_type = concept_identity((*structure_base, 2), **owner)
        for part_ordinal, part in enumerate(parts, 1):
            start = len(tokens)
            if part.kind == RESPONSE_PART_LITERAL:
                tokens.extend(part.units)
            position = (start, len(tokens))
            positions.append(position)
            if part.kind == RESPONSE_PART_LITERAL:
                role = role_identity(
                    (*structure_base, 11, part_ordinal), **owner)
                value_type = slot_type
            elif part.kind == RESPONSE_PART_GRAPH_ROLE:
                role = generic_response_graph_role(branch, part.category)
                value_type = generic_response_graph_type(branch, part.category)
            else:
                raise ValueError("generic response part kind is not registered")
            slot = StructureSlotDefinition(
                structure(1), structure(10, part_ordinal), role, value_type)
            slots.append(slot)
            if part.kind == RESPONSE_PART_LITERAL:
                atom = literal_atoms.get(part.units)
                new_literal = atom is None
                if atom is None:
                    atom = language_atom_identity(
                        branch,
                        (RESPONSE_CONNECTOR_PROFILE, 3,
                         *source.stable_key(), *position),
                    )
                    literal_atoms[part.units] = atom
                bindings.append(LanguageConnectorSlotBinding(
                    node(30, part_ordinal),
                    slot.slot,
                    values.constant_source,
                    constant=atom,
                ))
                if new_literal:
                    aliases.append(GenerationCandidateRealizationBinding(
                        atom,
                        representation_identity(
                            family, part.units, **owner),
                        RULE_LITERAL,
                        (annotation,),
                    ))
                if marker_slot is None:
                    marker_slot = slot.slot
            else:
                bindings.append(LanguageConnectorSlotBinding(
                    node(30, part_ordinal),
                    slot.slot,
                    values.role_filler_source,
                    role=role,
                    ordinal=ordinal_definition(
                        branch, part.ordinal).instruction,
                ))
                graph_slots.append(slot.slot)
            directives.append(LanguageConnectorSurfaceDirective(
                node(40, part_ordinal),
                slot.slot,
                surface.emit_action,
                instruction(41, part_ordinal),
                node(42, part_ordinal),
                (),
            ))
            examples.append((slot.slot, span_identity(source, members=(position,))))
        if marker_slot is None:
            raise ValueError("generic response requires a literal marker part")
        context_member = len(slots) + 1
        context_position = (len(tokens), len(tokens))
        context_slot = StructureSlotDefinition(
            structure(1),
            structure(10, context_member),
            context_role,
            context_type,
        )
        slots.append(context_slot)
        positions.append(context_position)
        bindings.append(LanguageConnectorSlotBinding(
            node(30, context_member),
            context_slot.slot,
            values.role_filler_source,
            role=context_role,
            ordinal=ordinal.instruction,
        ))
        directives.append(LanguageConnectorSurfaceDirective(
            node(40, context_member),
            context_slot.slot,
            surface.silent_action,
            instruction(41, context_member),
            node(42, context_member),
            (),
        ))
        examples.append((
            context_slot.slot,
            span_identity(source, members=(context_position,)),
        ))
        template = LanguageGenerationConnectorTemplate(
            node(1),
            branch,
            structure(8),
            concept_identity((*structure_base, 9), **owner),
            structure(3),
            structure(1),
            tuple(slots),
            tuple(bindings),
            structure(4),
            tuple(structure(60, index) for index in range(1, len(slots))),
            structure(5),
            (context_marker,),
            minimal_instruction_identity((*structure_base, 6), **owner),
            minimal_instruction_identity((*structure_base, 7), **owner),
            tuple(directives),
        )
        response = RecoveredResponseConnector(
            template,
            stance,
            marker_slot,
            source,
            document_scope(source),
            tuple(sorted(examples, key=lambda pair: pair[0].stable_key())),
            (annotation,),
            course.condition(variant),
        )
        result.append(ResponseConnectorCourse(
            response,
            SourceGenerationOrderCourse(
                template,
                source,
                response.scope,
                tuple(positions),
                annotation,
                RESPONSE_CONNECTOR_PROFILE,
                (context_slot.slot,),
                tuple(graph_slots),
            ),
            tuple(aliases),
            values,
            tuple(tokens),
        ))
    return tuple(result)


__all__ = [
    "ResponseConnectorCourse",
    "compile_generic_response_course",
    "compile_response_course",
]
