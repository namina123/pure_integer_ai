"""全部现役关系复用真实原型编译，端点变化不能污染其它整数或专属协议。"""
from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from pure_integer_ai.experiments.authored_relation_sqlite_course import (
    COURSE_FORMAT, COURSE_SCHEMA, _map_payload, build_sqlite_relation_adapter, file_sha256,
)
from pure_integer_ai.experiments.relation_sqlite_prototypes import (
    PROTOTYPE_LOADERS, compile_sqlite_relation_prototype, read_sqlite_relation_prototypes,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import W06_RELATION_PROFILES
from pure_integer_ai.storage.integer_codec import encode_integer_tuple

SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'data/ph2'


def _prototypes():
    """按来源及完整关系族冻结全部teacher/train/support原型，不读取开发答案。"""
    result = []
    for filename in sorted(PROTOTYPE_LOADERS):
        seen = set()
        for record in read_sqlite_relation_prototypes(SOURCE_ROOT / filename):
            seed = record.seed
            if (seed.label_owner == 'teacher' and seed.split == 'train'
                    and seed.sample_role == 'support' and seed.perturbation_kind == 'NONE'
                    and seed.relation_family not in seen):
                seen.add(seed.relation_family)
                result.append((filename, record))
    return tuple(result)


def test_all_registered_relations_reach_the_existing_typed_adapter(tmp_path):
    """七子阶段的十三种正向关系须真实接通，未知时序保留原负例资格。"""
    prototypes = _prototypes()
    assert {item.seed.relation_family for _, item in prototypes} == set(W06_RELATION_PROFILES) - {'EVENT_UNKNOWN'}
    path = tmp_path / 'all-relations.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript(COURSE_SCHEMA)
        db.executemany('INSERT INTO course_meta VALUES(?,?)', (
            ('format', COURSE_FORMAT), ('split', 'train'), ('label_owner', 'teacher'),
            ('license_id', 'CC0-1.0'), ('source_uri', 'urn:test:all-registered-relations'),
            ('attribution', 'Original CC0 test annotation'), ('record_count', str(len(prototypes)))))
        db.execute('INSERT INTO domain VALUES(1,?)', (b'cross-relation',))
        for case, (filename, record) in enumerate(prototypes, 1):
            seed = record.seed
            db.execute('INSERT INTO prototype VALUES(?,?,?,?)',
                (case, filename, seed.seed_id, file_sha256(SOURCE_ROOT / filename)))
            db.execute('INSERT INTO teacher_relation VALUES(?,1,?,?)', (case, case, case))
            for ordinal, endpoint in enumerate(seed.endpoints):
                surface = '训练端点' + str(case) + '编号' + str(ordinal)
                db.execute('INSERT INTO teacher_endpoint VALUES(?,?,?,?)',
                    (case, ordinal, encode_integer_tuple(tuple(map(ord, surface))), 9000000 + case*100 + ordinal))
    adapter, receipt = build_sqlite_relation_adapter(path, SOURCE_ROOT)
    assert not adapter.rejections
    assert len(adapter.candidates) == len(adapter.observations) == len(prototypes)
    assert set(receipt['relation_families']) == set(W06_RELATION_PROFILES) - {'EVENT_UNKNOWN'}
    assert len(receipt['relation_substages']) == 7
    assert receipt['evaluator_read_count'] == 0
    filename = 'authored_relation_precedes_seed_v1.jsonl.sample'
    unknown, = (item.seed for item in read_sqlite_relation_prototypes(SOURCE_ROOT / filename)
        if item.seed.relation_family == 'EVENT_UNKNOWN' and item.seed.label_owner == 'teacher')
    assert (unknown.split, unknown.sample_role, unknown.expected_state) == ('train', 'refute', 'UNKNOWN')
    case, = (index for index, (_filename, item) in enumerate(prototypes, 1)
             if item.seed.relation_family == 'EVENT_BEFORE')
    with sqlite3.connect(path) as db:
        db.execute('UPDATE prototype SET seed_id=? WHERE prototype_id=?', (unknown.seed_id, case))
    with pytest.raises(ValueError, match='train/support'):
        build_sqlite_relation_adapter(path, SOURCE_ROOT)


@pytest.mark.parametrize('substage,field', (('MEREOLOGY', 'mereology_protocol'),
    ('SIMILAR_ANTONYM', 'semantic_pair_protocol'), ('PRECEDES', 'event_time_protocol'),
    ('CAUSES', 'causal_protocol')))
def test_specialized_protocols_are_not_replaced_by_generic_compilation(substage, field):
    """关系专属证明必须保留在真实payload，不能只改来源标签使其过门。"""
    for _filename, record in _prototypes():
        if W06_RELATION_PROFILES[record.seed.relation_family].substage_key == substage:
            compiled, label = compile_sqlite_relation_prototype(record)
            assert field in compiled.observation_payload.to_value()
            assert label[0] == substage


def test_endpoint_remap_preserves_equal_nonidentity_integers_and_booleans():
    """相同数值的预算、方向和强度不随端点ID改变；嵌套local_id仍精确改写。"""
    value = dict(expected_endpoint_local_ids=[1, 2], expected_whole_local_id=2,
        direction=1, max_states=2, intensity_num=1, intensity_den=2, enabled=True,
        nested=dict(target_local_id=1, count=1))
    mapped = _map_payload(value, {1: 90001, 2: 90002})
    assert mapped == dict(expected_endpoint_local_ids=[90001, 90002], expected_whole_local_id=90002,
        direction=1, max_states=2, intensity_num=1, intensity_den=2, enabled=True,
        nested=dict(target_local_id=90001, count=1))
    assert value['expected_endpoint_local_ids'] == [1, 2]


def test_property_cannot_silently_drop_its_rational_member():
    """属性缺强度或其它关系混入强度时均关闭，不能依赖默认整数补齐。"""
    for _filename, record in _prototypes():
        if record.seed.relation_family == 'PROPERTY':
            with pytest.raises(ValueError, match='Rational'):
                compile_sqlite_relation_prototype(replace(record, rational_role_values=()))
        else:
            with pytest.raises(ValueError, match='Rational'):
                compile_sqlite_relation_prototype(replace(record, rational_role_values=((1, 1, 2),)))
