"""V3知识增量必须继承空正文父框架、真实学习新增载体且不再落盘旧格式。"""
from dataclasses import replace
from types import SimpleNamespace
import sqlite3

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY, OBJECT_PROPOSITION, OBJECT_ROLE, ObjectIdentity, concept_identity, language_branch_identity,
)
from pure_integer_ai.experiments import incremental_relation_model as extension
from pure_integer_ai.experiments import trained_relation_language_artifact as language
from pure_integer_ai.experiments.integer_relation_generation_bridge import _value_protocol, _surface_protocol
from pure_integer_ai.experiments.relation_generation_structure import compile_relation_connector
from pure_integer_ai.experiments.relation_language_carrier import learn_carrier_frame
from pure_integer_ai.experiments.relation_model_inheritance import (
    REQUIRED_COMPONENTS, inherit_parent_model_sidecars, parent_model_identity,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface, RelationSurfaceBinding, RelationSurfaceFrame, TrainedRelationGraphRuntime,
    TrainedRelationGraphError,
)
from pure_integer_ai.experiments.trained_relation_surface_artifact import (
    _FACT_ROW, _FRAME_ROW, _create_schema, _header, _pack_fact, _pack_frame,
    read_relation_surface_artifact, sha256_path,
)
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


def _members(number):
    """构造两个完整Span角色；训练正文仅存在本次内存与模型外测试来源。"""
    proposition = ObjectIdentity(OBJECT_PROPOSITION, (92610, number))
    predicate = concept_identity((92611,))
    roles = tuple(ObjectIdentity(OBJECT_ROLE, (92612, index)) for index in (1, 2))
    bindings = tuple(RelationSurfaceBinding(role, ObjectIdentity(OBJECT_ENTITY, (92613, number, index)),
        surface, 101+number, start, end) for index, (role, surface, start, end) in enumerate(
            ((roles[0], '青石', 3, 5), (roles[1], '石类', 7, 9)), 1))
    fact = ActiveRelationSurface(proposition, predicate, '属于', bindings,
        '提示：青石属于石类。', 101+number, 5, 7)
    frame = RelationSurfaceFrame(proposition, predicate, roles, ('', '属于', ''),
        101+number, 3, 9)
    return fact, frame


def test_incremental_frames_preserve_body_free_parent_and_learn_only_new_source(monkeypatch):
    """共享恢复器只看到新增事实，空正文父框架及其carrier原样保留。"""
    old, old_frame = _members(1)
    new, new_frame = _members(2)
    parent_frame = learn_carrier_frame(old, old_frame)
    owner = object.__new__(extension.IncrementalRelationTrainingRuntime)
    owner._parent_frame_contract = (parent_frame,)
    complete = (replace(old, evidence_surface=''), new)
    owner._surface_facts = complete
    original = TrainedRelationGraphRuntime._restore_surface_frames
    calls = []

    def restore(current):
        """保留真实Span切片逻辑，并确认父事实从未送入空正文恢复器。"""
        calls.append(current._surface_facts)
        return original(current)

    monkeypatch.setattr(TrainedRelationGraphRuntime, '_restore_surface_frames', restore)
    frames = owner._restore_surface_frames()
    assert frames == (parent_frame, learn_carrier_frame(new, new_frame))
    assert calls == [(new,)] and owner._surface_facts is complete
    owner._surface_facts = (complete[0], replace(new, evidence_surface=''))
    broken = owner._surface_facts
    with pytest.raises(TrainedRelationGraphError):
        owner._restore_surface_frames()
    assert owner._surface_facts is broken


def test_body_free_atomic_connector_uses_same_graph_slots_without_static_source_qualifier():
    """新增无正文语法保留同一角色/原子生成内容，但qualifier不重复编码示例句。"""
    fact, frame = _members(1)
    definition = SimpleNamespace(proposition=fact.proposition, predicate=fact.predicate,
        bindings=tuple(SimpleNamespace(role=b.role, filler=b.filler, ordinal=index)
            for index, b in enumerate(fact.bindings, 1)))
    source = SimpleNamespace(proposition=SimpleNamespace(definition=definition),
        evidence=(SimpleNamespace(stable_key=lambda: (92614, 1)),))
    branch = language_branch_identity((92615, 1))
    args = (source, branch, (92616, 1), _value_protocol(branch), _surface_protocol(branch))
    original = compile_relation_connector(fact, frame, *args)
    clean = replace(fact, evidence_surface='')
    learned = learn_carrier_frame(fact, frame)
    current = compile_relation_connector(clean, learned, *args)
    assert original.positions == current.positions
    assert original.aliases == current.aliases
    assert len(original.template.slots) == len(current.template.slots)
    assert current.qualifiers[1] == 2 and original.qualifiers[1] == 1
    for binding in fact.bindings:
        tokens = tuple(map(ord, binding.surface))
        assert not any(current.qualifiers[i:i+len(tokens)] == tokens
            for i in range(len(current.qualifiers)))
    with pytest.raises(ValueError):
        compile_relation_connector(clean, frame, *args)


def test_extended_language_first_physical_write_is_complete_v3(tmp_path, monkeypatch):
    """实际SQLite文件自首次写入就无整句且保留父/新全部成员，普通读者可恢复。"""
    core = tmp_path/'core.sqlite3'
    with sqlite3.connect(core) as db:
        db.execute('CREATE TABLE marker(value INTEGER NOT NULL) STRICT')
    originals = tuple(_members(i) for i in (1, 2))
    facts = tuple(replace(fact, evidence_surface='') for fact, _frame in originals)
    frames = tuple(learn_carrier_frame(fact, frame) for fact, frame in originals)
    propositions = tuple(fact.proposition for fact in facts)

    class Runtime:
        """只模拟Core资源owner，语言成员仍由真实SQLite编解码器核验。"""

        def __init__(self, _core, *, surface_artifact_database=None, **_kwargs):
            self.path = surface_artifact_database

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def active_surface_facts(self):
            if self.path is None:
                return tuple(fact for fact, _frame in originals)
            return read_relation_surface_artifact(self.path)[0]

        def active_surface_frames(self):
            return frames if self.path is None else read_relation_surface_artifact(self.path)[1]

        def active_propositions(self):
            return propositions

    monkeypatch.setattr(extension, 'IncrementalRelationTrainingRuntime', Runtime)
    monkeypatch.setattr(language, 'TrainedRelationGraphRuntime', Runtime)
    output = tmp_path/'language.sqlite3'
    receipt = extension.materialize_extended_relation_surfaces(core, core, core, output)
    restored, restored_frames, metadata = read_relation_surface_artifact(output)
    assert restored == facts and restored_frames == frames
    assert metadata['schema_version'] == receipt['schema_version'] == 3
    assert receipt['full_source_sentence_count'] == receipt['source_body_restored'] == 0
    assert not output.with_name(output.name+'.building').exists()


def _parent(tmp_path):
    """构建有类型的组件清单，验证复制不会漏掉新增可选组件或旧身份。"""
    parent = tmp_path/'parent'
    parent.mkdir()
    for name in (*REQUIRED_COMPONENTS, 'input_dialogue_act.sqlite3', 'outputs.sqlite3'):
        if name == 'relation_surfaces.sqlite3':
            continue
        with sqlite3.connect(parent/name) as db:
            db.execute('CREATE TABLE source_record(raw_text TEXT NOT NULL) STRICT')
    fact, frame = _members(1)
    with sqlite3.connect(parent/'relation_surfaces.sqlite3') as db:
        _create_schema(db)
        db.execute('INSERT INTO artifact_meta VALUES(1,?)', (encode_integer_tuple(
            _header(sha256_path(parent/'training.sqlite3'), 1, 1, schema_version=3)),))
        for kind, payload in ((_FACT_ROW, _pack_fact(replace(fact, evidence_surface=''), schema_version=3)),
                              (_FRAME_ROW, _pack_frame(learn_carrier_frame(fact, frame), schema_version=3))):
            db.execute('INSERT INTO relation_surface_row VALUES(?,0,?)', (kind, encode_integer_tuple(payload)))
    return parent


def test_inheritance_keeps_every_optional_component_and_exact_parent_identity(tmp_path):
    """实际文件复制保留全部11组件的冻结身份，Core与语言由增量阶段另行产生。"""
    parent = _parent(tmp_path)
    identity = parent_model_identity(parent)
    root = tmp_path/'run'
    (root/'model').mkdir(parents=True)
    copied = inherit_parent_model_sidecars(parent, root, identity)
    assert set(copied) == set(identity)-{'training.sqlite3', 'relation_surfaces.sqlite3'}
    assert 'input_dialogue_act.sqlite3' in copied and 'outputs.sqlite3' in copied
    assert identity == parent_model_identity(parent)
    assert all((root/'model'/name).read_bytes() == (parent/name).read_bytes() for name in copied)


@pytest.mark.parametrize('mutation', ('unknown_component', 'parent_body', 'sidecar_change'))
def test_inheritance_rejects_unregistered_data_source_body_or_changed_parent(tmp_path, mutation):
    """正文、额外资料和父身份漂移在任何新侧库写入前拒绝。"""
    parent = _parent(tmp_path)
    identity = parent_model_identity(parent)
    root = tmp_path/'run'
    (root/'model').mkdir(parents=True)
    if mutation == 'unknown_component':
        (parent/'answers.sqlite3').write_bytes(b'unregistered')
    else:
        name = 'training.sqlite3' if mutation == 'parent_body' else 'outputs.sqlite3'
        with sqlite3.connect(parent/name) as db:
            db.execute('INSERT INTO source_record VALUES(?)', ('来源正文',))
    with pytest.raises(ValueError):
        inherit_parent_model_sidecars(parent, root, identity)
    assert list((root/'model').iterdir()) == []
