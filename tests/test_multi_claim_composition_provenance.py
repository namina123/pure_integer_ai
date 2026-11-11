"""多来源增量必须保留父成员与独立许可，冷恢复不能借用其它成员权限。"""
from dataclasses import replace
import sqlite3

import pytest

from pure_integer_ai.experiments.multi_claim_composition_provenance import TABLE
from pure_integer_ai.experiments.trained_multi_claim_composer import (
    IDENTITY_PART_TABLE, META_TABLE, VARIANT_TABLE,
    MultiClaimCompositionVariant, PART_CLAIM_PREDICATE, PART_CLAIM_ROLE,
    PART_LITERAL, materialize_multi_claim_composition_graph,
    read_multi_claim_composition_graph,
)


def _members():
    """两份独立权限的语法分别绑定谓词；示例不含事实或完整回答。"""
    first = MultiClaimCompositionVariant(1, 1,
        ((PART_CLAIM_PREDICATE, 1, (901, 1)), (PART_CLAIM_ROLE, 1, 1),
         (PART_LITERAL, (12290,))), (902, 1), (903, 1))
    return first, replace(first, variant_id=2, source_identity=(902, 2),
                          license_identity=(903, 2))


def test_mixed_graph_keeps_exact_parent_and_independent_member_permissions(tmp_path):
    """单来源V2父键原样继承，V3冷恢复不把新成员改成父来源。"""
    first, second = _members()
    parent = tmp_path / 'parent.sqlite3'
    child = tmp_path / 'child.sqlite3'
    original = materialize_multi_claim_composition_graph((first,), parent)
    parent_bytes = parent.read_bytes()
    written = materialize_multi_claim_composition_graph((second, first), child)
    restored, metadata = read_multi_claim_composition_graph(child)
    assert original['format_version'] == 2
    assert written['format_version'] == metadata['format_version'] == 3
    assert restored == (first, second)
    assert restored[0].stable_key() == read_multi_claim_composition_graph(parent)[0][0].stable_key()
    assert metadata['variant_permissions'] == (
        (1, first.source_identity, first.license_identity),
        (2, second.source_identity, second.license_identity))
    assert parent.read_bytes() == parent_bytes


@pytest.mark.parametrize('field', ('source_identity', 'license_identity'))
def test_any_distinct_permission_uses_complete_per_member_identities(tmp_path, field):
    """只改变来源或只改变许可也必须逐项保存，不能只比较其中一维。"""
    first, _second = _members()
    second = replace(first, variant_id=2, **{field: (904, 2)})
    graph = tmp_path / 'distinct.sqlite3'
    materialize_multi_claim_composition_graph((first, second), graph)
    restored, metadata = read_multi_claim_composition_graph(graph)
    assert restored == (first, second)
    assert metadata['format_version'] == 3


def test_mixed_graph_has_input_order_independent_physical_identity(tmp_path):
    """相同成员按规范序写入，调用方枚举顺序不改变物理artifact。"""
    members = _members()
    a, b = tmp_path / 'a.sqlite3', tmp_path / 'b.sqlite3'
    materialize_multi_claim_composition_graph(members, a)
    materialize_multi_claim_composition_graph(members[::-1], b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize('mutation', (
    'missing', 'ordinal', 'field', 'orphan', 'source', 'license',
    'variant_source_hash', 'variant_license_hash', 'envelope', 'graph_identity'))
def test_mixed_graph_rejects_incomplete_or_reassigned_permission(tmp_path, mutation):
    """逐成员权限、完整图封套和哈希须同时成立，任一漂移均拒绝恢复。"""
    graph = tmp_path / 'tampered.sqlite3'
    materialize_multi_claim_composition_graph(_members(), graph)
    with sqlite3.connect(graph) as db:
        if mutation == 'missing':
            db.execute(f'DELETE FROM {TABLE} WHERE variant_id=2 AND field_code=1')
        elif mutation == 'ordinal':
            db.execute(f'UPDATE {TABLE} SET part_ordinal=7 WHERE variant_id=2 AND field_code=1 AND part_ordinal=1')
        elif mutation == 'field':
            db.execute(f'UPDATE {TABLE} SET field_code=3 WHERE variant_id=2 AND field_code=1')
        elif mutation == 'orphan':
            db.execute(f'UPDATE {TABLE} SET variant_id=3 WHERE variant_id=2')
        elif mutation in ('source', 'license'):
            field = 1 if mutation == 'source' else 2
            db.execute(f'UPDATE {TABLE} SET part_value=1 WHERE variant_id=2 AND field_code=? AND part_ordinal=1', (field,))
        elif mutation in ('variant_source_hash', 'variant_license_hash'):
            column = 'source_hash' if mutation == 'variant_source_hash' else 'license_hash'
            db.execute(f'UPDATE {VARIANT_TABLE} SET {column}=1 WHERE variant_id=2')
        elif mutation == 'envelope':
            db.execute(f'UPDATE {IDENTITY_PART_TABLE} SET part_value=9 WHERE field_code=1 AND part_ordinal=0')
        else:
            db.execute(f'UPDATE {META_TABLE} SET graph_id=1')
    with pytest.raises(ValueError):
        read_multi_claim_composition_graph(graph)
