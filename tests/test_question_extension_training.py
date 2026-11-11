"""验证增量课程只贡献训练语法，且不能把非训练来源或答案带入图。"""
from pathlib import Path
import sqlite3

import pytest

from scripts.train_relation_question_extension import COURSE_FORMAT, train
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    SLOT_MODE_REFERENCE, TrainedRelationQuestionFrame,
)
from pure_integer_ai.cognition.understanding.query_slot_question_composition import (
    ComposedSlotQuestionFrame, available_slot_question_frames,
)
from pure_integer_ai.experiments.trained_relation_question_graph import (
    materialize_relation_question_graph, read_relation_question_graph,
)

SEEDS = Path(__file__).resolve().parents[1] / "data/ph2"


def _course(tmp_path, *, split=1, family=1, seed="teacher-switch-causes-light-v1",
            wh="什么", license_id="CC0-1.0", mode=0, text="灯亮的原因是什么？", answer=0):
    """创建独立 SQLite 教师课程与一帧父图，不读取任何评测资料。"""
    course = tmp_path / "course.sqlite3"
    with sqlite3.connect(course) as db:
        db.execute("CREATE TABLE course_meta(key TEXT PRIMARY KEY,value TEXT)")
        db.executemany("INSERT INTO course_meta VALUES(?,?)", {
            "format": COURSE_FORMAT, "split": "train", "license_id": license_id,
            "label_owner": "teacher", "source_uri": "urn:test:question-extension",
        }.items())
        db.execute("CREATE TABLE question_case(case_id INTEGER,split_code INTEGER,"
                   "seed_family INTEGER,seed_id TEXT,input_text TEXT,"
                   "answer_binding_index INTEGER,wh_surface TEXT,"
                   "operation_code INTEGER,slot_mode INTEGER)")
        db.execute("INSERT INTO question_case VALUES(1,?,?,?,?,?,?,?,?)",
                   (split, family, seed, text, answer, wh, 1, mode))
    parent = tmp_path / "parent.sqlite3"
    frame = TrainedRelationQuestionFrame(
        (1, 1), (92004, 1), (5, 1), ((8, 1), (8, 2)), ((), (33,), ()))
    materialize_relation_question_graph((frame,), parent,
                                        source_identity=(7, 1), license_identity=(7, 2))
    return course, parent, frame


def test_extension_preserves_parent_and_stores_syntax_without_answers(tmp_path):
    """真实编译后冷恢复父帧与新角色语法，整数图不保存示例端点。"""
    course, parent, old = _course(tmp_path)
    before = parent.read_bytes()
    output = tmp_path / "trained"
    receipt = train(course, parent, SEEDS, output)
    frames, meta = read_relation_question_graph(output / "relation_questions.sqlite3")
    added, = (frame for frame in frames if frame != old)
    assert parent.read_bytes() == before
    assert receipt["added_frame_count"] == 1
    assert receipt["frame_count"] == 2
    assert meta["source_identity"] == (7, 1)
    assert added.gaps == ((), tuple(map(ord, "的原因是什么？")))
    assert len(added.roles) == 1 and added.answer_role_key not in added.roles
    assert receipt["evaluator_read_count"] == 0
    with sqlite3.connect(output / "relation_questions.sqlite3") as db:
        assert db.execute(
            "SELECT DISTINCT typeof(part_value) FROM relation_question_frame_part"
        ).fetchall() == [("integer",)]


def test_slot_question_composition_keeps_each_gap_parent_and_contract():
    """新间隔组合须保留父帧，WH 序位和答案 Role，不能跨合同拼接。"""
    first = TrainedRelationQuestionFrame(
        (92002, 101), (92004, 1), (5, 60), ((8, 1),),
        ((), (100, 200)), (8, 2), (200,), 0)
    second = TrainedRelationQuestionFrame(
        (92002, 102), first.operation_key, first.predicate_key, first.roles,
        ((300,), (400, 200)), first.answer_role_key, first.wh_values, 0)
    frames = available_slot_question_frames(
        (first, second), (300, 55, 100, 200))
    composed = tuple(item for item in frames
                     if isinstance(item, ComposedSlotQuestionFrame))
    assert len(composed) == 1
    frame = composed[0]
    assert frame.gaps == ((300,), (100, 200))
    assert frame.gap_sources == ((second.stable_key(),), (first.stable_key(),))
    assert ComposedSlotQuestionFrame.from_stable_key(frame.stable_key()) == frame


def test_extension_trains_reference_marker_with_exact_predicate_and_role(tmp_path):
    """真实课程编译并冷恢复指代模式，运行授权只能来自新训练图。"""
    from pure_integer_ai.cognition.understanding.query_reference_context import reference_marker_sources
    course, parent, old = _course(tmp_path, mode=SLOT_MODE_REFERENCE,
                                  text="它是灯亮的原因吗？", wh="它")
    output = tmp_path / "reference-trained"
    receipt = train(course, parent, SEEDS, output)
    frames, _ = read_relation_question_graph(output / "relation_questions.sqlite3")
    added, = (item for item in frames if item != old)
    assert added.is_reference_slot
    assert added.wh_values == tuple(map(ord, "它"))
    assert reference_marker_sources(added.wh_values, added.answer_role_key,
                                    added.predicate_key, frames) == (added.stable_key(),)
    assert not reference_marker_sources(added.wh_values, added.answer_role_key,
                                        added.predicate_key, (old,))
    assert receipt["added_frame_count"] == 1 and receipt["evaluator_read_count"] == 0


@pytest.mark.parametrize(("family", "seed", "text", "middle_gap"), [
    (1, "teacher-heat-causes-melt-v1", "高温会使冰融化吗？", "会使"),
    (4, "teacher-pronoun-refers-v1", "它指向甲吗？", "指向"),
])
def test_extension_trains_complete_relation_question_surface_from_supported_seed(
        tmp_path, family, seed, text, middle_gap):
    """完整 yes/no 问式只从匹配的 train teacher/support 关系源编译。"""
    course, parent, old = _course(
        tmp_path, family=family, seed=seed, text=text, answer=-1, wh="")
    output = tmp_path / "complete-question-trained"
    receipt = train(course, parent, SEEDS, output)
    frames, _ = read_relation_question_graph(output / "relation_questions.sqlite3")
    added, = (frame for frame in frames if frame != old)
    assert added.gaps == (
        (), tuple(map(ord, middle_gap)), tuple(map(ord, "吗？")))
    assert len(added.roles) == 2 and not added.answer_role_key
    assert receipt["added_frame_count"] == 1
    assert receipt["evaluator_read_count"] == 0
    assert receipt["source_text_included"] == 0


def test_slot_question_composition_rejects_wh_gap_position_drift():
    """WH 位于不同间隔时保持独立语法，不能制造无 WH 的组合。"""
    first = TrainedRelationQuestionFrame(
        (92002, 111), (92004, 1), (5, 60), ((8, 1),),
        ((), (100, 200)), (8, 2), (200,), 0)
    second = TrainedRelationQuestionFrame(
        (92002, 112), first.operation_key, first.predicate_key, first.roles,
        ((200, 300), (400,)), (8, 2), (200,), 0)
    assert not any(isinstance(item, ComposedSlotQuestionFrame)
                   for item in available_slot_question_frames(
                       (first, second), (200, 55, 400)))


@pytest.mark.parametrize("family,seed,text,answer", [
    (5, "teacher-high-low-antonym-v1", "高和什么相反？", 1),
    (6, "teacher-open-before-enter-v1", "开门之后的事件是什么？", 1),
])
def test_extension_learns_semantic_pair_and_event_syntax(tmp_path, family, seed, text, answer):
    """新增来源族走同一教师编译及冷恢复，不能将答案端点写入语言间隔。"""
    course, parent, old = _course(tmp_path, family=family, seed=seed, text=text, answer=answer)
    output = tmp_path / "trained"
    receipt = train(course, parent, SEEDS, output)
    frames, _ = read_relation_question_graph(output / "relation_questions.sqlite3")
    added, = (frame for frame in frames if frame != old)
    assert receipt["added_frame_count"] == 1
    assert added.wh_values == tuple(map(ord, "什么"))
    assert added.answer_role_key not in added.roles
    assert added.gaps == ((), tuple(map(ord, text[1 if family == 5 else 2:])))


@pytest.mark.parametrize("arguments", [
    {"split": 2}, {"seed": "unqualified-evaluation-seed"}, {"wh": ""},
    {"license_id": "UNVERIFIED"},
    {"mode": 2},
    {"mode": SLOT_MODE_REFERENCE, "wh": "", "answer": -1, "text": "按开关导致灯亮"},
])
def test_extension_rejects_unqualified_courses_before_writing(tmp_path, arguments):
    """非训练 split、非教师来源、缺 WH 和无许可均不得生成新模型。"""
    course, parent, _ = _course(tmp_path, **arguments)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        train(course, parent, SEEDS, output)
    assert not output.exists()
