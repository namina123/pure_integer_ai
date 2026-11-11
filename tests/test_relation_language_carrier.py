"""无正文模型须保留来源化载体间隔，生产开放角色解析不能依赖旧来源句。"""
from dataclasses import replace
import sqlite3

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY, OBJECT_PROPOSITION, OBJECT_ROLE, ObjectIdentity,
)
from pure_integer_ai.experiments.relation_language_carrier import learn_carrier_frame
from pure_integer_ai.experiments.trained_open_role_frames import restore_open_role_frames
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, RelationSurfaceBinding, RelationSurfaceFrame,
)
from pure_integer_ai.experiments.trained_relation_surface_artifact import (
    RelationSurfaceArtifactError, _create_schema, _header, _pack_fact, _pack_frame,
    _unpack_fact, _unpack_frame, read_relation_surface_artifact, validate_language_members,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


def _members():
    """原来源仅用于训练边缘间隔；运行事实本体不携带此正文。"""
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (301,))
    predicate = ObjectIdentity(OBJECT_ENTITY, (302,))
    roles = tuple(ObjectIdentity(OBJECT_ROLE, (303, index)) for index in (1, 2))
    bindings = tuple(RelationSurfaceBinding(role, ObjectIdentity(OBJECT_ENTITY, (304, index)),
        surface, 101, start, end) for index, (role, surface, start, end) in enumerate(
            ((roles[0], '青石', 3, 5), (roles[1], '石类', 7, 9)), 1))
    fact = ActiveRelationSurface(proposition, predicate, '属于', bindings,
                                 '提示：青石属于石类。', 101, 5, 7)
    frame = RelationSurfaceFrame(proposition, predicate, roles, ('', '属于', ''), 101, 3, 9)
    return fact, frame


def test_body_free_open_parser_matches_complete_original_carrier_without_reconstructing_body():
    """真实开放角色消费者得到与父模型相同框架，正文始终为空。"""
    fact, frame = _members()
    learned = learn_carrier_frame(fact, frame)
    clean = replace(fact, evidence_surface='')
    assert (learned.carrier_prefix, learned.carrier_suffix) == ('提示：', '。')
    assert replace(learned, carrier_prefix=None, carrier_suffix=None) == frame
    validate_language_members((clean,), (learned,), require_carrier=True)
    assert restore_open_role_frames((learned,), (clean,)) == restore_open_role_frames((frame,), (fact,))
    assert clean.evidence_surface == ''
    assert learn_carrier_frame(clean, learned) == learned


def test_v3_artifact_cold_restore_keeps_carrier_members_and_no_source_sentence(tmp_path):
    """实际有类型持久化后版本、全部角色与边缘间隔原样恢复。"""
    fact, frame = _members()
    clean, learned = replace(fact, evidence_surface=''), learn_carrier_frame(fact, frame)
    path = tmp_path/'language.sqlite3'
    with sqlite3.connect(path) as db:
        _create_schema(db)
        db.execute('INSERT INTO artifact_meta VALUES(1,?)', (encode_integer_tuple(
            _header('ab'*32, 1, 1, schema_version=3)),))
        db.execute('INSERT INTO relation_surface_row VALUES(1,0,?)',
                   (encode_integer_tuple(_pack_fact(clean, schema_version=3)),))
        db.execute('INSERT INTO relation_surface_row VALUES(2,0,?)',
                   (encode_integer_tuple(_pack_frame(learned, schema_version=3)),))
    facts, frames, metadata = read_relation_surface_artifact(path, expected_core_sha256='ab'*32)
    assert facts == (clean,) and frames == (learned,) and metadata['schema_version'] == 3
    assert restore_open_role_frames(frames, facts) == restore_open_role_frames((frame,), (fact,))


@pytest.mark.parametrize('mutation', ('missing', 'prefix', 'suffix', 'gap', 'binding', 'cue', 'source'))
def test_carrier_migration_rejects_missing_or_forged_language_members(mutation):
    """不猜标点，不借跨度或来源漂移放行无正文模型。"""
    fact, frame = _members()
    learned, clean = learn_carrier_frame(fact, frame), replace(fact, evidence_surface='')
    if mutation == 'missing':
        learned = frame
    elif mutation == 'prefix':
        learned = replace(learned, carrier_prefix='')
    elif mutation == 'suffix':
        learned = replace(learned, carrier_suffix='石类。')
    elif mutation == 'gap':
        learned = replace(learned, gaps=('', '属于了', ''))
    elif mutation == 'binding':
        clean = replace(clean, bindings=(replace(clean.bindings[0], start=2), clean.bindings[1]))
    elif mutation == 'cue':
        clean = replace(clean, cue_end=10)
    else:
        learned = replace(learned, source_hash=102)
    with pytest.raises(ValueError):
        restore_open_role_frames((learned,), (clean,))


def test_old_body_free_component_cannot_guess_lost_carrier_language():
    """V2缺少来源正文且未登记carrier，迁移和生产解析均拒绝。"""
    fact, frame = _members()
    clean = replace(fact, evidence_surface='')
    with pytest.raises(ValueError):
        learn_carrier_frame(clean, frame)
    with pytest.raises(ValueError):
        restore_open_role_frames((frame,), (clean,))


def test_complete_language_frame_survives_a_filtered_active_fact_view():
    """生产只给出当前可用事实子集，完整已核验框架仍可解析未知事实的语言。"""
    fact, frame = _members()
    learned = learn_carrier_frame(fact, frame)
    assert restore_open_role_frames((learned,), ()) == restore_open_role_frames((frame,), (fact,))


@pytest.mark.parametrize('writer,reader', ((1, 3), (2, 3), (3, 1), (3, 2)))
def test_carrier_frame_and_fact_payloads_reject_cross_version_reads(writer, reader):
    """新旧payload不能靠header替换伪装，亦不能静默丢弃新载体成员。"""
    fact, frame = _members()
    if writer == 3:
        fact, frame = replace(fact, evidence_surface=''), learn_carrier_frame(fact, frame)
    with pytest.raises(RelationSurfaceArtifactError):
        _unpack_frame(_pack_frame(frame, schema_version=writer), schema_version=reader)
    with pytest.raises(RelationSurfaceArtifactError):
        _unpack_fact(_pack_fact(fact, schema_version=writer), schema_version=reader)
