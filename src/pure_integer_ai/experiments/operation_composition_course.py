"""Compile licensed multi-claim syntax into ordinary native G/S-07/R-01 courses.

Training exemplars authorize predicate/role shapes and grammar intervals. The
recovered contract contains no exemplar claim, filler, whole sentence or answer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ROLE, ObjectIdentity, SourceRef, concept_identity, language_atom_identity, minimal_instruction_identity,
    representation_identity, structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.representation_rendering import representation_parts
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import StructureSlotDefinition
from pure_integer_ai.cognition.shared.unicode_representation import validate_unicode_scalars
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.operation_fact_source import verify_operation_fact_source
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageConnectorSlotBinding, LanguageConnectorSurfaceDirective, LanguageGenerationConnectorTemplate,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line, parse_canonical_json_bytes
from pure_integer_ai.experiments.ph2_generation_candidate_alias_contract import GenerationCandidateRealizationBinding
from pure_integer_ai.experiments.ph2_generation_candidate_pack import RULE_LITERAL
from pure_integer_ai.experiments.ph2_grounded_answer_order import SourceGenerationOrderCourse
from pure_integer_ai.experiments.relation_generation_protocol import ordinal_definition
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    MultiClaimCompositionVariant, PART_CLAIM_PREDICATE, PART_CLAIM_ROLE, PART_LITERAL,
)
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.experiments.operation_training_surface import verify_operation_training_surface
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry


PROFILE = 92300
WITNESS = 92301
READERS = 92302
READER = 92303


@dataclass(frozen=True, slots=True)
class OperationCompositionWitness:
    variant_id: int
    source: SourceRef
    source_sha256: tuple[int, ...]
    template_key: tuple[int, ...]
    # claim coordinate, original semantic role, scoped reader role, native ordinal
    readers: tuple[tuple[int, ObjectIdentity, ObjectIdentity, int], ...]

    def __post_init__(self):
        if type(self.variant_id) is not int or self.variant_id <= 0 or not isinstance(self.source, SourceRef):
            raise ValueError("native operation composition identity is incomplete")
        digest = self.source_sha256
        if (type(digest) is not tuple or len(digest) != 32
                or any(type(value) is not int or not 0 <= value <= 255 for value in digest)
                or self.source.source_id != ((int.from_bytes(bytes(digest[:8]), "big") & ((1 << 63) - 1)) or 1)
                or self.source.source_kind != PROFILE):
            raise ValueError("native composition source lacks its complete physical digest")
        if (type(self.template_key) is not tuple or not self.template_key
                or any(type(value) is not int or value < 0 for value in self.template_key)
                or len(self.template_key) > 65536 or type(self.readers) is not tuple
                or not self.readers or len(self.readers) > 64):
            raise ValueError("native composition template or reader budget exceeded")
        if (tuple((claim, role.stable_key()) for claim, role, _, _ in self.readers)
                != tuple(sorted({(claim, role.stable_key()) for claim, role, _, _ in self.readers}))
                or len({scoped for _, _, scoped, _ in self.readers}) != len(self.readers)):
            raise ValueError("native composition readers are ambiguous or not canonical")
        for claim, role, scoped, ordinal in self.readers:
            if (type(claim) is not int or claim <= 0 or type(ordinal) is not int or ordinal < 0
                    or role.object_kind != scoped.object_kind or role.object_kind != OBJECT_ROLE):
                raise ValueError("native composition has an invalid typed role reader")

    def stable_key(self):
        return pack_record(WITNESS, (self.variant_id,), self.source.stable_key(), self.source_sha256,
            self.template_key, pack_record(READERS, *(pack_record(READER, (claim, ordinal),
                role.stable_key(), scoped.stable_key()) for claim, role, scoped, ordinal in self.readers)))

    @classmethod
    def from_stable_key(cls, key):
        fields = unpack_record(key, WITNESS)
        if len(fields) != 5 or len(fields[0]) != 1:
            raise ValueError("native composition witness is incomplete")
        readers = []
        for row in unpack_record(fields[4], READERS):
            parts = unpack_record(row, READER)
            if len(parts) != 3 or len(parts[0]) != 2:
                raise ValueError("native composition role coordinates are incomplete")
            readers.append((parts[0][0], ObjectIdentity.from_stable_key(parts[1]),
                            ObjectIdentity.from_stable_key(parts[2]), parts[0][1]))
        result = cls(fields[0][0], SourceRef.from_stable_key(fields[1]), fields[2], fields[3], tuple(readers))
        if result.stable_key() != key:
            raise ValueError("native composition witness is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class OperationCompositionCourse:
    variant: MultiClaimCompositionVariant
    witness: OperationCompositionWitness
    template: LanguageGenerationConnectorTemplate
    order: SourceGenerationOrderCourse
    aliases: tuple[GenerationCandidateRealizationBinding, ...]
    values: object


def compile_train_operation_composition(source_path, facts, core, generator):
    """读取物理规范指令，复用来源与当前Core完整编译核。"""
    adapter, _seeds = verify_operation_fact_source(facts)
    raw = Path(source_path).resolve(strict=True).read_bytes()
    if len(raw) > 1024 * 1024 or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("native composition needs one bounded physical train record")
    row = parse_canonical_json_bytes(raw[:-1], require_object=True)
    return _compile_operation_composition_record(row, raw, facts, core, generator, adapter)


def compile_train_operation_composition_record(row, facts, core, generator):
    """直接编译typed课程指令；规范字节只计算来源身份，不落盘JSONL。"""
    adapter, _seeds = verify_operation_fact_source(facts)
    if type(row) is not dict:
        raise ValueError("native composition needs one bounded physical train record")
    raw = canonical_json_line(row)
    if len(raw) > 1024 * 1024:
        raise ValueError("native composition needs one bounded physical train record")
    return _compile_operation_composition_record(row, raw, facts, core, generator, adapter)


def _compile_operation_composition_record(row, raw, facts, core, generator, adapter):
    """核对实际许可前提、完整角色和原生生成组织，不增加事实。"""
    required = {"claim_keys", "separator_values", "variant_id", "owner", "split", "license_id"}
    if (set(row) not in (required, required | {"output_gaps"})
            or (row["owner"], row["split"], row["license_id"]) != ("teacher", "train", "CC0-1.0")
            or type(row["claim_keys"]) is not list or not 1 <= len(row["claim_keys"]) <= 8
            or type(row["variant_id"]) is not int or row["variant_id"] <= 0
            or type(row["separator_values"]) is not list or len(row["separator_values"]) > 8):
        raise ValueError("native composition requires a licensed train grammar instruction")
    keys = tuple(tuple(key) for key in row["claim_keys"])
    if len(set(keys)) != len(keys):
        raise ValueError("native composition needs distinct training exemplar claims")
    if len(keys) > 1 and not row["separator_values"]:
        raise ValueError("multi-claim composition requires a trained separator")
    output_gaps = row.get("output_gaps")
    if len(keys) == 1 and output_gaps is None:
        raise ValueError("single-claim realization requires an independently trained output frame")
    if output_gaps is not None:
        if (type(output_gaps) is not list or len(output_gaps) != len(keys)
                or any(type(gaps) is not list for gaps in output_gaps)):
            raise ValueError("output frame must provide one ordered gap sequence per claim")
        for gaps in output_gaps:
            if (not gaps or any(type(gap) is not list or len(gap) > 128
                    or any(type(value) is not int for value in gap) for gap in gaps)):
                raise ValueError("output frame gap exceeds its bounded integer contract")
            for gap in gaps:
                if gap:
                    validate_unicode_scalars(tuple(gap))
    separator = tuple(row["separator_values"])
    if separator:
        validate_unicode_scalars(separator)
    teacher = {record.proposition_key: record for record in facts.course.records}
    definitions = {candidate.proposition.proposition.stable_key(): candidate.proposition for candidate in adapter.candidates}
    observation_sources = {candidate.proposition.proposition.stable_key(): candidate.observation_source
                           for candidate in adapter.candidates}
    active = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    frames = {frame.proposition.stable_key(): frame for frame in core.active_surface_frames()}
    if any(key not in teacher or key not in active or key not in frames for key in keys):
        raise ValueError("native composition exemplar is outside the actual licensed active Core")
    chosen = tuple(active[key] for key in keys)
    for fact in chosen:
        record = teacher[fact.proposition.stable_key()]
        definition = core.generation_input(fact.proposition).proposition.definition
        if (record.sample_role_code != 1 or record.predicate_key != fact.predicate.stable_key()
                or definitions.get(fact.proposition.stable_key()) != definition):
            raise ValueError("native composition fact differs from its physical train teacher")
        source_hash = IntegerIdentityRegistry(core.backend).find(IDENTITY_SOURCE_RECORD,
            observation_sources[record.proposition_key].stable_key())
        verify_operation_training_surface(record,fact,frames[fact.proposition.stable_key()],source_hash)
    parent = generator.generation_template_for(chosen[0])
    branch, metadata = parent.language_branch, dict(owner=parent.language_branch.owner, versions=parent.language_branch.versions)
    from pure_integer_ai.experiments.trained_generation_connector_runtime import (
        _bound_relation_proposition, _generation_protocols, _surface_protocol,
    )
    surface = _surface_protocol(branch)
    profiles = tuple(owner for owner in generator._branches if owner.branch == branch)
    if len(profiles) != 1 or any(generator.generation_template_for(fact).language_branch != branch for fact in chosen):
        raise ValueError("native composition requires an explicit common language branch")
    values = profiles[0].definition_graph.value_protocol
    recovered = generator.connector_for(_bound_relation_proposition(core.generation_input(chosen[0].proposition),
        parent, _generation_protocols(generator.context, branch).content.answer),
        branch=branch, connector_identity=parent.connector)
    route = recovered.surface_for(chosen[0].bindings[0].filler)
    if route.result.selected is None or len(route.result.options) != 1:
        raise ValueError("native composition has no unique parent representation family")
    family = representation_parts(route.result.selected.value)[0]
    digest = tuple(hashlib.sha256(raw).digest())
    source = SourceRef(PROFILE, (int.from_bytes(bytes(digest[:8]), "big") & ((1 << 63) - 1)) or 1,
                       row["variant_id"], branch.owner, branch.versions)
    identity = _hash((PROFILE, row["variant_id"], *digest))
    base, structure_base = (20916, 2, identity, identity), (20916, 3, identity, identity)
    owned = lambda *suffix: structure_concept_identity((*base, *suffix), **metadata)
    structure = lambda *suffix: structure_concept_identity((*structure_base, *suffix), **metadata)
    instruction = lambda *suffix: minimal_instruction_identity((*base, *suffix), **metadata)
    parts, readers, grammar = [], [], []
    for claim, fact in enumerate(chosen, 1):
        parts.append((PART_CLAIM_PREDICATE, claim, fact.predicate.stable_key()))
        frame = frames[fact.proposition.stable_key()]
        if tuple(binding.role for binding in fact.bindings) != frame.roles:
            raise ValueError("native composition exemplar roles do not follow their trained frame")
        definition = core.generation_input(fact.proposition).proposition.definition
        gaps = (tuple(tuple(gap) for gap in output_gaps[claim - 1])
                if output_gaps is not None
                else tuple(tuple(map(ord, gap)) for gap in frame.gaps))
        if len(gaps) != len(fact.bindings) + 1:
            raise ValueError("output frame must leave exactly one gap around each ordered role")
        for index, units in enumerate(gaps):
            if units:
                if len(units) > 128:
                    raise ValueError("native composition grammar atom budget exceeded")
                if output_gaps is not None:
                    literal = "".join(map(chr, units))
                    if any(binding.surface and binding.surface in literal for binding in fact.bindings):
                        raise ValueError("output frame literal contains an exemplar filler")
                parts.append((PART_LITERAL, units)); grammar.append((units, None))
            if index < len(fact.bindings):
                role = fact.bindings[index].role
                typed = tuple(binding for binding in definition.bindings if binding.role == role)
                if len(typed) != 1:
                    raise ValueError("native composition requires single-valued exact typed roles")
                scoped = role_identity((PROFILE, row["variant_id"], claim, len(role.stable_key()), *role.stable_key()), **metadata)
                readers.append((claim, role, scoped, typed[0].ordinal))
                parts.append((PART_CLAIM_ROLE, claim, index + 1)); grammar.append(((), scoped))
        if claim < len(chosen):
            parts.append((PART_LITERAL, separator)); grammar.append((separator, None))
    if len(grammar) > 64:
        raise ValueError("native composition slot budget exceeded")
    variant = MultiClaimCompositionVariant(row["variant_id"], len(chosen), tuple(parts),
                                           facts.course.source_identity, facts.course.license_identity)
    ordinals = {item.ordinal: item for item in values.ordinals}
    ordinals[0] = ordinal_definition(branch, 0)
    values = replace(values, ordinals=tuple(ordinals.values()))
    slot_type = concept_identity((*structure_base, 2), **metadata)
    slots = tuple(StructureSlotDefinition(structure(1), structure(10, index),
        role_identity((*structure_base, 11, index), **metadata), slot_type) for index in range(1, len(grammar) + 1))
    bindings, directives, aliases, positions, external = [], [], [], [], []
    cursor = 0
    for index, (slot, (units, scoped)) in enumerate(zip(slots, grammar, strict=True), 1):
        atom = None if scoped is not None else language_atom_identity(branch, (PROFILE, *digest, index))
        bindings.append(LanguageConnectorSlotBinding(owned(30, index), slot.slot,
            values.role_filler_source if scoped is not None else values.constant_source,
            role=scoped, ordinal=ordinals[0].instruction if scoped is not None else None, constant=atom))
        directives.append(LanguageConnectorSurfaceDirective(owned(40, index), slot.slot, surface.emit_action,
                                                             instruction(41, index), owned(42, index), ()))
        positions.append((cursor, cursor + len(units))); cursor += len(units)
        if scoped is not None:
            external.append(slot.slot)
        else:
            aliases.append(GenerationCandidateRealizationBinding(atom, representation_identity(family, units, **metadata),
                RULE_LITERAL, ((PROFILE, *source.stable_key(), *digest, index),)))
    template = LanguageGenerationConnectorTemplate(owned(1), branch, structure(80),
        concept_identity((PROFILE, identity), **metadata), structure(3), structure(1), tuple(slots), tuple(bindings),
        structure(4), tuple(structure(60, index) for index in range(1, len(slots))), structure(5), (),
        minimal_instruction_identity((*structure_base, 6), **metadata),
        minimal_instruction_identity((*structure_base, 7), **metadata), tuple(directives))
    witness = OperationCompositionWitness(row["variant_id"], source, digest, template.stable_key(),
        tuple(sorted(readers, key=lambda item: (item[0], item[1].stable_key()))))
    order = SourceGenerationOrderCourse(template, source, document_scope(source), tuple(positions),
        (PROFILE, 1, *digest, row["variant_id"], len(grammar)), PROFILE, external_slots=tuple(external))
    return OperationCompositionCourse(variant, witness, template, order, tuple(aliases), values)
