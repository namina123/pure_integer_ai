"""U-00 语言、表示、结构和最小指令分型身份测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.crosscut.guards.int_blocker import IntViolation
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_OCCURRENCE,
    OBJECT_REPRESENTATION,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    language_atom_identity,
    language_branch_identity,
    minimal_instruction_identity,
    object_contracts_by_kind,
    occurrence_identity,
    representation_identity,
    span_identity,
    structure_concept_identity,
    validate_object_contracts,
    VersionBundle,
)


def test_language_atom_identity_uses_branch_but_never_surface():
    zh = language_branch_identity((101,))
    en = language_branch_identity((202,))
    zh_atom = language_atom_identity(zh, (1,))
    en_atom = language_atom_identity(en, (1,))

    assert zh.object_kind == OBJECT_LANGUAGE_BRANCH
    assert zh_atom.object_kind == OBJECT_LANGUAGE_ATOM
    assert zh_atom != en_atom
    with pytest.raises(TypeError):
        language_atom_identity(zh, (2,), surface="x")


def test_representation_identity_is_independent_of_language_atoms():
    family_key = (9001,)
    sequence = representation_identity(family_key, (20320, 22909))
    zh_atom = language_atom_identity(
        language_branch_identity((101,)), (11,))
    other_atom = language_atom_identity(
        language_branch_identity((202,)), (37,))

    candidate_links = {
        zh_atom.stable_key(): sequence.stable_key(),
        other_atom.stable_key(): sequence.stable_key(),
    }
    assert sequence.object_kind == OBJECT_REPRESENTATION
    assert len(set(candidate_links.values())) == 1
    with pytest.raises(TypeError):
        representation_identity(family_key, (65,), language=1)


def test_representation_family_is_open_injected_identity_not_host_enum():
    text_family = representation_identity((17, 23), (65,))
    future_family = representation_identity((88001, 99002, 3), (65,))

    assert text_family != future_family
    assert text_family.object_kind == future_family.object_kind


def test_occurrence_span_structure_and_instruction_remain_distinct():
    source = SourceRef(
        1, 2, 3, GLOBAL_OWNER_SCOPE, VersionBundle())
    occurrence = occurrence_identity(source, start=0, end=1, ordinal=0)
    span = span_identity(source, members=((0, 1),))
    structure = structure_concept_identity((77, 1))
    instruction = minimal_instruction_identity((77, 1))

    assert occurrence.object_kind == OBJECT_OCCURRENCE
    assert span.object_kind == OBJECT_SPAN
    assert structure.object_kind == OBJECT_STRUCTURE_CONCEPT
    assert instruction.object_kind == OBJECT_MINIMAL_INSTRUCTION
    assert len({item.stable_key() for item in (
        occurrence, span, structure, instruction)}) == 4


def test_language_atom_rejects_non_branch_and_identity_keys_fail_closed():
    representation = representation_identity((1,), (2,))
    with pytest.raises(ValueError, match="语言分支"):
        language_atom_identity(representation, (3,))
    with pytest.raises(ValueError, match="非空整数元组"):
        language_branch_identity(())
    with pytest.raises(IntViolation):
        representation_identity((1,), ("x",))
    with pytest.raises(ValueError, match="布尔值"):
        structure_concept_identity((True,))


def test_u00_object_contracts_have_unique_authoritative_owners():
    contracts = object_contracts_by_kind()
    for object_kind in (
            OBJECT_LANGUAGE_BRANCH,
            OBJECT_LANGUAGE_ATOM,
            OBJECT_REPRESENTATION,
            OBJECT_STRUCTURE_CONCEPT,
            OBJECT_MINIMAL_INSTRUCTION):
        contract = contracts[object_kind]
        assert contract.authoritative_identity
        assert contract.persistence_owner == "storage.graph_object"
    assert validate_object_contracts() == ()


def test_u00_object_identity_round_trip_rejects_corrupt_keys():
    identity = representation_identity((17, 23), (65, 66))
    assert ObjectIdentity.from_stable_key(identity.stable_key()) == identity
    with pytest.raises(ValueError, match="components 长度"):
        ObjectIdentity.from_stable_key(identity.stable_key()[:-1])
    corrupt = list(identity.stable_key())
    corrupt[9] += 1
    with pytest.raises(ValueError, match="components 长度"):
        ObjectIdentity.from_stable_key(tuple(corrupt))
