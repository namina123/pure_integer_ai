"""L-00 语言对象身份、词形迁移和恢复契约测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.concept_index import (
    ConceptIdentityConflict,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_REPRESENTATION,
    OBJECT_SENSE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.understanding.word_form_index import (
    WordFormIndex,
)
from pure_integer_ai.cognition.understanding.sense_index import (
    LegacySenseSpec,
    SenseIndex,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import (
    EPI_STRUCTURED,
    SOURCE_BARE_TEXT,
)
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.node_store import NODE_CONCEPT, NODE_WORD
from pure_integer_ai.storage.word_form_index import (
    load_legacy_word_form_bridges,
)
from pure_integer_ai.storage.sense_candidates import (
    SenseLegacyBridgeConflict,
    bootstrap_sense_candidates,
    load_legacy_sense_bridges,
    read_sense_candidates,
    sense_surface_hash,
)
from pure_integer_ai.training.cursor import dump_run, load_run

_UNICODE_FAMILY_KEY = (71001,)
_BRANCH_FORM_RELATION_KEY = (71002,)
_ATOM_FORM_RELATION_KEY = (71003,)
_ATOM_SENSE_RELATION_KEY = (71004,)
_OCCURRENCE_CONCEPT_RELATION_KEY = (71005,)
_SENSE_CONCEPT_RELATION_KEY = (71006,)


def _source(document_id: int = 1) -> SourceRef:
    """构造测试使用的稳定来源，不把 surface 放入对象键。"""
    return SourceRef(
        1,
        9001,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _word_forms(ctx) -> WordFormIndex:
    """创建使用注入表示族和 predicate 的权威词形入口。"""
    return WordFormIndex(
        ctx.backend,
        ctx.concept_index,
        ontology=ctx.graph_ontology,
        unicode_family_key=_UNICODE_FAMILY_KEY,
        inventory_relation_key=_BRANCH_FORM_RELATION_KEY,
    )


def test_same_representation_can_belong_to_different_language_branches():
    backend = DictBackend()
    ctx = make_train_context(backend)
    objects = LanguageObjectIndex(ctx.graph_ontology)
    forms = _word_forms(ctx)
    branch_a = objects.ensure_branch((101,))
    branch_b = objects.ensure_branch((202,))
    atom_a = objects.ensure_atom(branch_a, (1,))
    atom_b = objects.ensure_atom(branch_b, (1,))

    representation_a = forms.ensure(
        "A",
        branch=branch_a,
        scope=session_scope(1),
        provenance_kind=SOURCE_BARE_TEXT,
    )
    representation_b = forms.ensure(
        "A",
        branch=branch_b,
        scope=session_scope(2),
        provenance_kind=SOURCE_BARE_TEXT,
    )
    objects.relate(
        _ATOM_FORM_RELATION_KEY,
        atom_a,
        representation_a,
        scope=session_scope(3),
        provenance_kind=SOURCE_BARE_TEXT,
    )
    objects.relate(
        _ATOM_FORM_RELATION_KEY,
        atom_b,
        representation_b,
        scope=session_scope(4),
        provenance_kind=SOURCE_BARE_TEXT,
    )

    assert atom_a != atom_b
    assert representation_a == representation_b
    assert representation_a.object_kind == OBJECT_REPRESENTATION
    assert forms.lookup("A", branch=branch_a) == representation_a
    assert forms.lookup("A", branch=branch_b) == representation_a
    assert forms.forms(branch=branch_a)[(65,)] == representation_a


def test_same_atom_can_have_multiple_senses_and_occurrences_remain_distinct():
    backend = DictBackend()
    ctx = make_train_context(backend)
    objects = LanguageObjectIndex(ctx.graph_ontology)
    branch = objects.ensure_branch((303,))
    atom = objects.ensure_atom(branch, (11,))
    source = _source()
    sense_a = objects.ensure_sense(source, sense_key=(1,))
    sense_b = objects.ensure_sense(source, sense_key=(2,))
    concept = objects.ensure_concept((88001,))

    objects.relate(
        _ATOM_SENSE_RELATION_KEY,
        atom,
        sense_a,
        scope=session_scope(5),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    objects.relate(
        _ATOM_SENSE_RELATION_KEY,
        atom,
        sense_b,
        scope=session_scope(6),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    occurrence_a = objects.ensure_occurrence(
        source, start=0, end=1, ordinal=0)
    occurrence_b = objects.ensure_occurrence(
        source, start=0, end=1, ordinal=1)
    for scope_id, occurrence in enumerate(
            (occurrence_a, occurrence_b), start=7):
        objects.relate(
            _OCCURRENCE_CONCEPT_RELATION_KEY,
            occurrence,
            concept,
            scope=session_scope(scope_id),
            provenance_kind=SOURCE_BARE_TEXT,
        )

    predicate = ctx.graph_ontology.resolve(
        relation_concept_identity(_ATOM_SENSE_RELATION_KEY))
    assert predicate is not None
    assert {item.object for item in ctx.graph_ontology.statements(
        predicate=predicate, subject=atom)} == {sense_a, sense_b}
    assert sense_a.object_kind == sense_b.object_kind == OBJECT_SENSE
    assert occurrence_a != occurrence_b


def test_legacy_surface_and_word_nodes_require_explicit_complete_migration():
    backend = DictBackend()
    ctx = make_train_context(backend)
    objects = LanguageObjectIndex(ctx.graph_ontology)
    forms = _word_forms(ctx)
    branch = objects.ensure_branch((404,))
    legacy_concept = ctx.concept_index.ensure(
        "南京市", space_id=ctx.space_id, node_type=NODE_CONCEPT)
    legacy_word = forms.register(
        "南京市", language=1, space_id=ctx.space_id)

    migration = forms.migrate_legacy(
        "南京市",
        language=1,
        space_id=ctx.space_id,
        branch=branch,
        scope=session_scope(9),
        provenance_kind=SOURCE_BARE_TEXT,
    )

    assert set(migration.legacy_refs) == {legacy_concept, legacy_word}
    for ref, node_type in (
            (legacy_concept, NODE_CONCEPT), (legacy_word, NODE_WORD)):
        assert load_legacy_word_form_bridges(
            backend, legacy_ref=ref) == ((
                node_type,
                OBJECT_REPRESENTATION,
                migration.representation.space_id,
                migration.representation.local_id,
            ),)
    assert backend.count(
        GRAPH_OBJECT_TABLE,
        where={"object_kind": OBJECT_LANGUAGE_ATOM},
    ) == 0, "词形迁移不得顺带创造语言原子或语义概念"
    with pytest.raises(ConceptIdentityConflict):
        ctx.concept_index.ensure_typed(
            "南京市", space_id=ctx.space_id, node_type=NODE_WORD)


def test_legacy_sense_candidates_require_source_and_complete_key_mapping():
    backend = DictBackend()
    ctx = make_train_context(backend)
    objects = LanguageObjectIndex(ctx.graph_ontology)
    branch = objects.ensure_branch((450,))
    atom = objects.ensure_atom(branch, (1,))
    assert bootstrap_sense_candidates(
        backend,
        ctx.concept_index,
        [("老鼠", ["动物义", "设备义"])],
        space_id=ctx.space_id,
    ) == 2
    legacy_refs = tuple(item[0] for item in read_sense_candidates(
        backend, ctx.space_id, sense_surface_hash("老鼠")))
    senses = SenseIndex(
        backend,
        objects,
        atom_sense_relation_key=_ATOM_SENSE_RELATION_KEY,
        sense_concept_relation_key=_SENSE_CONCEPT_RELATION_KEY,
    )
    specs = tuple(
        LegacySenseSpec(ref, (index,), (90000 + index,))
        for index, ref in enumerate(legacy_refs, start=1)
    )

    with pytest.raises(SenseLegacyBridgeConflict):
        senses.migrate_legacy(
            "老鼠",
            atom,
            _source(2),
            specs=specs[:1],
            scope=session_scope(11),
            provenance_kind=SOURCE_BARE_TEXT,
        )
    bindings = senses.migrate_legacy(
        "老鼠",
        atom,
        _source(2),
        specs=specs,
        scope=session_scope(11),
        provenance_kind=SOURCE_BARE_TEXT,
    )

    assert senses.lookup(atom) == bindings
    for legacy_ref, binding in zip(legacy_refs, bindings, strict=True):
        assert load_legacy_sense_bridges(
            backend, legacy_ref=legacy_ref) == tuple(sorted((
                (
                    binding.sense.object_kind,
                    binding.sense.space_id,
                    binding.sense.local_id,
                ),
                (
                    binding.concept.object_kind,
                    binding.concept.space_id,
                    binding.concept.local_id,
                ),
            )))


def test_authoritative_language_objects_and_legacy_bridges_roundtrip(tmp_path):
    first = DictBackend()
    ctx1 = make_train_context(first)
    objects1 = LanguageObjectIndex(ctx1.graph_ontology)
    forms1 = _word_forms(ctx1)
    branch1 = objects1.ensure_branch((505,))
    legacy_concept = ctx1.concept_index.ensure(
        "逻辑", space_id=ctx1.space_id, node_type=NODE_CONCEPT)
    forms1.register("逻辑", language=1, space_id=ctx1.space_id)
    migrated = forms1.migrate_legacy(
        "逻辑",
        language=1,
        space_id=ctx1.space_id,
        branch=branch1,
        scope=session_scope(10),
        provenance_kind=SOURCE_BARE_TEXT,
    )
    atom1 = objects1.ensure_atom(branch1, (2,))
    bootstrap_sense_candidates(
        first,
        ctx1.concept_index,
        [("逻辑", ["逻辑词义"])],
        space_id=ctx1.space_id,
    )
    legacy_sense_ref = read_sense_candidates(
        first, ctx1.space_id, sense_surface_hash("逻辑"))[0][0]
    sense_index1 = SenseIndex(
        first,
        objects1,
        atom_sense_relation_key=_ATOM_SENSE_RELATION_KEY,
        sense_concept_relation_key=_SENSE_CONCEPT_RELATION_KEY,
    )
    sense_binding1 = sense_index1.migrate_legacy(
        "逻辑",
        atom1,
        _source(3),
        specs=(LegacySenseSpec(
            legacy_sense_ref, (31,), (93001,)),),
        scope=session_scope(12),
        provenance_kind=SOURCE_BARE_TEXT,
    )[0]
    config = FormalTrainConfig(run_dir=str(tmp_path), run_id="l00_identity")
    dump_run(
        first,
        config.run_dir,
        config.run_id,
        spaces=[ctx1.space_id],
        tables=config.dump_tables,
    )

    second = DictBackend()
    ctx2 = make_train_context(second)
    assert load_run(second, config.run_dir, config.run_id) == [ctx2.space_id]
    objects2 = LanguageObjectIndex(ctx2.graph_ontology)
    forms2 = _word_forms(ctx2)
    branch2 = objects2.lookup_branch((505,))

    assert branch2 == branch1
    assert forms2.lookup("逻辑", branch=branch2) == migrated.representation
    assert forms2.segment("逻辑结构", branch=branch2) == ["逻辑", "结", "构"]
    assert load_legacy_word_form_bridges(
        second, legacy_ref=legacy_concept)[0][1:] == (
            OBJECT_REPRESENTATION,
            migrated.representation.space_id,
            migrated.representation.local_id,
        )
    atom2 = objects2.lookup_atom(branch2, (2,))
    assert atom2 is not None
    sense_index2 = SenseIndex(
        second,
        objects2,
        atom_sense_relation_key=_ATOM_SENSE_RELATION_KEY,
        sense_concept_relation_key=_SENSE_CONCEPT_RELATION_KEY,
    )
    assert sense_index2.lookup(atom2)[0] == sense_binding1
    assert len(load_legacy_sense_bridges(
        second, legacy_ref=legacy_sense_ref)) == 2
