"""V2语言组件移除完整来源句，版本与完整成员保持严格可核验。"""
from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_ENTITY, OBJECT_PROPOSITION, OBJECT_ROLE, ObjectIdentity
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, RelationSurfaceBinding, RelationSurfaceFrame,
)
from pure_integer_ai.experiments.trained_relation_surface_artifact import (
    RelationSurfaceArtifactError, _pack_fact, _unpack_fact, validate_language_members,
)


def _members():
    """构造带完整角色与来源跨度的语言成员，正文只用于检查旧版本兼容。"""
    entity = ObjectIdentity(OBJECT_ENTITY, (1,))
    role = ObjectIdentity(OBJECT_ROLE, (2,))
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (3,))
    predicate = ObjectIdentity(OBJECT_ENTITY, (4,))
    binding = RelationSurfaceBinding(role, entity, '青石', 101, 0, 2)
    fact = ActiveRelationSurface(proposition, predicate, '属于', (binding,), '青石属于石类。', 101, 2, 4)
    frame = RelationSurfaceFrame(proposition, predicate, (role,), ('', '属于石类。'), 101, 0, 7)
    return fact, frame


def test_language_record_has_no_source_sentence_field_or_read_time_reconstruction():
    """V2完整保留角色、cue与Span，读取后正文为空，旧版本行为不变。"""
    fact, frame = _members()
    old = _pack_fact(fact)
    new = _pack_fact(fact, schema_version=2)
    assert _unpack_fact(old) == fact
    assert len(old) - len(new) == len(fact.evidence_surface) + 1
    restored = _unpack_fact(new, schema_version=2)
    assert restored == replace(fact, evidence_surface='')
    validate_language_members((restored,), (frame,))


@pytest.mark.parametrize('writer,reader', ((1, 2), (2, 1), (2, 3)))
def test_language_record_rejects_cross_version_or_unregistered_payload(writer, reader):
    """不能把带正文的旧payload标成新版本，或默许未知版本。"""
    fact, _frame = _members()
    with pytest.raises(RelationSurfaceArtifactError):
        _unpack_fact(_pack_fact(fact, schema_version=writer), schema_version=reader)


@pytest.mark.parametrize('mutation', ('body', 'literal', 'role', 'source'))
def test_language_members_reject_source_body_and_cross_member_drift(mutation):
    """拒绝把示例藏进gap，以及伪造完整角色或来源归属。"""
    fact, frame = _members()
    fact = replace(fact, evidence_surface='')
    if mutation == 'body':
        fact = replace(fact, evidence_surface='青石属于石类。')
    elif mutation == 'literal':
        frame = replace(frame, gaps=('青石', '属于石类。'))
    elif mutation == 'role':
        frame = replace(frame, roles=(ObjectIdentity(OBJECT_ROLE, (99,)),))
    else:
        frame = replace(frame, source_hash=102)
    with pytest.raises(RelationSurfaceArtifactError):
        validate_language_members((fact,), (frame,))
