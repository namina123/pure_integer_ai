"""真实无正文父模型必须逐段匹配独立教师来源，缺成员或漂移不能被空正文掩盖。"""
from dataclasses import replace
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.operation_fact_source import read_sqlite_operation_fact_source
from pure_integer_ai.experiments.operation_training_surface import verify_operation_training_surface
from pure_integer_ai.experiments.integer_relation_graph_course import integer_relation_graph_course_to_w06_adapter
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphRuntime
from pure_integer_ai.storage.assertion_identity import IDENTITY_SOURCE_RECORD, IntegerIdentityRegistry


@pytest.fixture(scope='module')
def trained_source():
    """读取当前开发用V3模型与原物理train课程，所有父组件保持只读。"""
    parent = os.environ.get('PIA_OPERATION_BODYFREE_PARENT')
    source = os.environ.get('PIA_OPERATION_TEACHER_DATABASE')
    if parent is None or source is None:
        pytest.skip('requires explicitly supplied current body-free development parent and train source')
    parent, source = Path(parent).resolve(strict=True), Path(source).resolve(strict=True)
    if parent.drive.upper() != 'K:' or source.drive.upper() != 'K:':
        raise ValueError('explicit K training and parent roots required')
    course = read_sqlite_operation_fact_source(source, Path(__file__).resolve().parents[1] / 'data/ph2')
    adapter = integer_relation_graph_course_to_w06_adapter(course.course,require_full_curriculum=False)
    with TrainedRelationGraphRuntime(parent/'training.sqlite3',
            surface_artifact_database=parent/'relation_surfaces.sqlite3') as core:
        facts = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
        frames = {frame.proposition.stable_key(): frame for frame in core.active_surface_frames()}
        record = next(record for record in course.course.records if record.sample_role_code == 1)
        fact, frame = facts[record.proposition_key], frames[record.proposition_key]
        candidate, = (candidate for candidate in adapter.candidates
                       if candidate.proposition.proposition.stable_key() == record.proposition_key)
        source_hash = IntegerIdentityRegistry(core.backend).find(IDENTITY_SOURCE_RECORD,candidate.observation_source.stable_key())
        assert not fact.evidence_surface and frame.carrier_prefix is not None and frame.carrier_suffix is not None
    return record, fact, frame, source_hash


def test_verified_body_free_members_keep_legacy_teacher_contract_without_restoring_body(trained_source):
    """空正文模型与原兼容格式都须消费同一完整来源，而不是跳过来源检查。"""
    record, fact, frame, source_hash = trained_source
    verify_operation_training_surface(record,fact,frame,source_hash)
    legacy = replace(fact,evidence_surface=''.join(map(chr,record.surface_values)))
    verify_operation_training_surface(record,legacy,replace(frame,carrier_prefix=None,carrier_suffix=None),source_hash)
    assert fact.evidence_surface == ''


@pytest.mark.parametrize('change', ('source','span','roles','filler','cue','gap','carrier','missing_carrier','legacy_body'))
def test_current_teacher_rejects_span_member_and_source_drift(trained_source,change):
    """即使命题id相同，任何语言成员、来源或完整旧正文改变都不能授权新课程。"""
    record, fact, frame, source_hash = trained_source
    if change == 'source':
        source_hash += 1
    elif change == 'span':
        fact = replace(fact,bindings=(replace(fact.bindings[0],end=fact.bindings[0].end+1),*fact.bindings[1:]))
    elif change == 'roles':
        frame = replace(frame,roles=tuple(reversed(frame.roles)))
    elif change == 'filler':
        fact = replace(fact,bindings=(replace(fact.bindings[0],filler=fact.bindings[1].filler),*fact.bindings[1:]))
    elif change == 'cue':
        fact = replace(fact,cue_start=fact.cue_start+1)
    elif change == 'gap':
        frame = replace(frame,gaps=(frame.gaps[0]+'x',*frame.gaps[1:]))
    elif change == 'carrier':
        frame = replace(frame,carrier_suffix=frame.carrier_suffix+'x')
    elif change == 'missing_carrier':
        frame = replace(frame,carrier_prefix=None,carrier_suffix=None)
    else:
        fact = replace(fact,evidence_surface=''.join(map(chr,record.surface_values))+'x')
    with pytest.raises(ValueError):
        verify_operation_training_surface(record,fact,frame,source_hash)
