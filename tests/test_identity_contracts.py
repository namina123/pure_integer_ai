"""B-02 共享身份、版本、owner 和逻辑时序契约测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    LogicalTime,
    OBJECT_CONCEPT,
    OBJECT_SOURCE_RECORD,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    TypedRef,
    VersionBundle,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
    character_identity,
    legacy_character_identity,
    legacy_word_form_identity,
    object_contracts_by_kind,
    occurrence_identity,
    span_identity,
    validate_object_contracts,
    word_form_identity,
)
from pure_integer_ai.experiments.lang_curriculum import build_language_curriculum_plan
from pure_integer_ai.storage.spaces.registry import SPACE_TYPE_CORE, SpaceRegistry


def _versions(*, corpus: int = 1, parser: int = 2,
              primitive: int = 3, curriculum: int = 4) -> VersionBundle:
    return VersionBundle(
        CorpusVersion(corpus),
        ParserVersion(parser),
        PrimitiveVersion(primitive),
        CurriculumVersion(curriculum),
    )


def test_named_versions_are_not_interchangeable():
    assert CorpusVersion(1) != ParserVersion(1)
    assert _versions().stable_key() == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        CorpusVersion(-1)


def test_owner_scope_enforces_visibility_shape():
    tenant = OwnerScope(tenant_id=7, visibility=VISIBILITY_TENANT)
    user = OwnerScope(tenant_id=7, user_id=9, visibility=VISIBILITY_USER)
    session = OwnerScope(
        tenant_id=7, user_id=9, session_id=11,
        visibility=VISIBILITY_SESSION,
    )
    assert len({GLOBAL_OWNER_SCOPE, tenant, user, session}) == 4
    with pytest.raises(ValueError):
        OwnerScope(user_id=9, visibility=VISIBILITY_USER)


def test_source_ref_is_not_a_concept_ref_or_typed_ref():
    source = SourceRef(4, 12, 3, GLOBAL_OWNER_SCOPE, _versions())
    concept = TypedRef(OBJECT_CONCEPT, 1, 12, GLOBAL_OWNER_SCOPE, _versions())
    assert source != concept
    assert source != concept.node_ref()
    with pytest.raises(ValueError, match="SourceRef"):
        TypedRef(OBJECT_SOURCE_RECORD, 1, 12)


def test_legacy_surface_keys_are_compatible_but_not_authoritative():
    zh = legacy_word_form_identity(
        (20320, 22909), language=1, versions=_versions())
    en = legacy_word_form_identity(
        (20320, 22909), language=2, versions=_versions())
    changed_parser = legacy_word_form_identity(
        (20320, 22909), language=1, versions=_versions(parser=3))
    assert zh.stable_key() != en.stable_key()
    assert zh.stable_key() != changed_parser.stable_key()
    legacy_character = legacy_character_identity(65, language=1)
    assert legacy_character != legacy_character_identity(65, language=2)
    assert character_identity(65, language=1) == legacy_character
    assert word_form_identity(
        (20320, 22909), language=1, versions=_versions()) == zh
    contracts = object_contracts_by_kind()
    assert not contracts[legacy_character.object_kind].authoritative_identity
    assert not contracts[zh.object_kind].authoritative_identity


def test_occurrence_and_span_keys_include_source_and_position():
    source = SourceRef(4, 12, 3, GLOBAL_OWNER_SCOPE, _versions())
    first = occurrence_identity(source, start=0, end=2, ordinal=0)
    second = occurrence_identity(source, start=0, end=2, ordinal=1)
    span = span_identity(source, members=((0, 2),))
    assert first.stable_key() != second.stable_key()
    assert first.stable_key() != span.stable_key()


def test_logical_time_rejects_wall_clock_like_reverse_order():
    assert LogicalTime(3, 5, 8).stable_key() == (3, 5, 8)
    with pytest.raises(ValueError, match="observed_seq"):
        LogicalTime(5, 4, 0)
    with pytest.raises(ValueError, match="used_seq"):
        LogicalTime(5, 0, 4)


def test_object_contracts_cover_all_shared_object_kinds():
    assert validate_object_contracts() == ()


def test_space_identity_is_stable_without_runtime_allocation():
    first = SpaceRegistry.identity_for(SPACE_TYPE_CORE, "core")
    second = SpaceRegistry.identity_for(SPACE_TYPE_CORE, "core")
    other = SpaceRegistry.identity_for(SPACE_TYPE_CORE, "other")
    assert first == second
    assert first.stable_key() != other.stable_key()


def test_curriculum_plan_carries_typed_version():
    version = CurriculumVersion(9)
    plan = build_language_curriculum_plan(2, curriculum_version=version)
    assert plan
    assert {state.curriculum_version for state in plan} == {version}
