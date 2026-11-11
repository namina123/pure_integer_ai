"""增量表面恢复必须核验完整父合同，且只读取新增命题的来源 Span。"""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import ObjectIdentity, OBJECT_PROPOSITION
from pure_integer_ai.experiments import incremental_relation_model as extension
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationGraphSnapshot, TrainedRelationGraphError, TrainedRelationGraphRuntime,
)


@dataclass(frozen=True)
class _Fact:
    """测试只保留身份；真实图绑定和证据由完整 generation_input 合同模拟。"""

    proposition: ObjectIdentity


def _runtime(monkeypatch, mode):
    """建立父完整内容合同与候选视图，观察共享恢复器实际收到的命题集合。"""
    old = ObjectIdentity(OBJECT_PROPOSITION, (92456, 101))
    new = ObjectIdentity(OBJECT_PROPOSITION, (92456, 102))
    old_fact, new_fact = _Fact(old), _Fact(new)
    atomic = lambda identity: SimpleNamespace(definition=SimpleNamespace(proposition=identity))
    owner = object.__new__(extension.IncrementalRelationTrainingRuntime)
    owner._parent_core, owner._parent_surfaces = "parent-core", "parent-surfaces"
    owner._snapshot = ActiveRelationGraphSnapshot(
        (atomic(new),) if mode == "missing" else (atomic(old), atomic(new)), ())
    owner.generation_input = lambda _identity: (1, 2, 4 if mode == "drift" else 3)
    observed = []

    class Parent:
        """测试父资源 owner，完整内容元组代表命题、角色、来源、证据合同。"""

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def active_surface_facts(self):
            return (old_fact,)

        def active_surface_frames(self):
            return ()

        def generation_input(self, _identity):
            return (1, 2, 3)

    def restore(current):
        """共享恢复器若读到父命题即失败，证明不会回填或读取旧正文。"""
        keys = tuple(item.definition.proposition for item in current._snapshot.propositions)
        observed.append(keys)
        assert keys == (new,)
        if mode == "restore_failure":
            raise ValueError("span incomplete")
        return (new_fact,)

    monkeypatch.setattr(extension, "TrainedRelationGraphRuntime", Parent)
    monkeypatch.setattr(TrainedRelationGraphRuntime, "_restore_surface_facts", restore)
    return owner, observed, (old_fact, new_fact)


def test_incremental_restore_preserves_parent_and_reads_only_new_spans(monkeypatch):
    """父模型表面完整保留，新增恢复仍调用原共享算法。"""
    owner, observed, facts = _runtime(monkeypatch, "valid")
    complete = owner._snapshot
    assert owner._restore_surface_facts() == facts
    assert len(observed) == 1
    assert owner._snapshot is complete


@pytest.mark.parametrize("mode", ("missing", "drift"))
def test_incremental_restore_rejects_parent_identity_or_content_drift(monkeypatch, mode):
    """父 active 丢失或相同身份的内容漂移，在任何新正文读取前拒绝。"""
    owner, observed, _facts = _runtime(monkeypatch, mode)
    with pytest.raises(TrainedRelationGraphError):
        owner._restore_surface_facts()
    assert observed == []


def test_incremental_restore_failure_does_not_leave_partial_snapshot(monkeypatch):
    """新增来源失败后完整只读视图仍恢复，不能把部分集冒充全部 Core。"""
    owner, _observed, _facts = _runtime(monkeypatch, "restore_failure")
    complete = owner._snapshot
    with pytest.raises(ValueError, match="span incomplete"):
        owner._restore_surface_facts()
    assert owner._snapshot is complete
