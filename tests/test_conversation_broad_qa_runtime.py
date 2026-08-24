from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pathlib import Path
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
)


_ROOT = Path(__file__).resolve().parents[1]
_RELATION_COURSES = (
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v1.jsonl.sample",
    _ROOT / "data/ph2/dlg_raw_public_relation_evidence_v2.jsonl.sample",
)
_MARKER_COURSE = _ROOT / (
    "data/ph2/dlg_raw_public_relation_marker_evidence_v1.jsonl.sample")
_ANSWER_FRAME_COURSE = _ROOT / (
    "data/ph2/dlg_raw_public_relation_answer_frame_v1.jsonl.sample")


class _FakeResult:
    status = "ANSWER"
    answer = "完整回答。"
    title = "公开来源"
    source_url = "https://example.invalid/source"
    evidence_chain = ()


def test_narrow_consumer_precedes_broad_query_and_keeps_bounded_history() -> None:
    calls = []

    class _Connection:
        pass

    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module
    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, question: calls.append(question) or _FakeResult()
    try:
        state = BroadDialogueState((1, 2, 3))
        state, turn = answer_broad_dialogue_turn(
            state, "窄域问题", _Connection(),
            narrow_answer=lambda _question: ("完整命题句。", "ANSWER"),
        )
        assert turn.answer == "完整命题句。"
        assert turn.display_answer == "完整命题句。"
        assert calls == []
        for index in range(10):
            state, _ = answer_broad_dialogue_turn(
                state, f"广域问题{index}", _Connection())
        assert len(state.turns) == 8
        assert state.turns[-1].turn_key
    finally:
        module.query_broad_qa = original


def test_broad_answer_keeps_full_evidence_but_projects_readable_primary_surface() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    class _Citation:
        selected_text = "主证据句。"

    class _Result:
        status = "ANSWER"
        answer = "主证据句。\n相邻证据句。"
        title = "公开来源"
        source_url = "https://example.invalid/source"
        evidence_chain = (_Citation(),)

    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, _question: _Result()
    try:
        state, turn = answer_broad_dialogue_turn(
            BroadDialogueState((9, 9, 1)), "来源问题", object())
        assert turn.answer == "主证据句。\n相邻证据句。"
        assert turn.display_answer == "主证据句。"
        assert state.turns[-1] == turn
    finally:
        module.query_broad_qa = original


def test_surface_consumer_only_changes_display_surface() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    class _Citation:
        selected_text = "主证据句。"

    class _Result:
        status = "ANSWER"
        answer = "主证据句。\n相邻证据句。"
        title = "公开来源"
        source_url = "https://example.invalid/source"
        evidence_chain = (_Citation(),)

    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, _question: _Result()
    try:
        state, turn = answer_broad_dialogue_turn(
            BroadDialogueState((3, 4, 5)), "来源问题", object(),
            surface_consumer=lambda surface, status, title: (
                "改写表面。" if status == "ANSWER" and title == "公开来源"
                else None),
        )
        assert turn.display_answer == "改写表面。"
        assert turn.answer == "主证据句。\n相邻证据句。"
        assert turn.source_title == "公开来源"
        assert state.turns[-1] == turn
    finally:
        module.query_broad_qa = original


def test_source_title_is_injected_for_immediate_reference_followup() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    calls = []

    class _Result:
        status = "ANSWER"
        answer = "来源证据句。"
        title = "矮寨大桥"
        source_url = "https://example.invalid/bridge"
        evidence_chain = ()

    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, question: (
        calls.append(question) or _Result())
    try:
        state = BroadDialogueState((4, 5, 6))
        state, _ = answer_broad_dialogue_turn(state, "首轮问题", object())
        state, turn = answer_broad_dialogue_turn(state, "它位于哪里？", object())
        assert turn.status == "ANSWER"
        assert calls == ["首轮问题", "矮寨大桥，它位于哪里？"]
    finally:
        module.query_broad_qa = original


def test_source_focus_does_not_cross_unknown_turn() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    calls = []

    class _Answer:
        status = "ANSWER"
        answer = "来源证据句。"
        title = "公开来源"
        source_url = "https://example.invalid/source"
        evidence_chain = ()

    class _Unknown:
        status = "UNKNOWN"
        answer = None
        title = None
        source_url = None
        evidence_chain = ()

    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, question: (
        calls.append(question) or (_Answer() if len(calls) == 1 else _Unknown()))
    try:
        state = BroadDialogueState((7, 8, 9))
        state, _ = answer_broad_dialogue_turn(state, "首轮问题", object())
        state, _ = answer_broad_dialogue_turn(state, "新话题", object())
        state, _ = answer_broad_dialogue_turn(state, "它是什么？", object())
        assert calls == ["首轮问题", "新话题", "它是什么？"]
    finally:
        module.query_broad_qa = original


def test_source_focus_does_not_match_pronoun_substring_in_direct_question() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    class _Result:
        status = "ANSWER"
        answer = "来源证据句。"
        title = "武穴酥糖"
        source_url = "https://example.invalid/source"
        evidence_chain = ()

    calls = []
    original = module.query_broad_qa
    module.query_broad_qa = lambda _connection, question: (
        calls.append(question) or _Result())
    try:
        state = BroadDialogueState((8, 8, 8))
        state, _ = answer_broad_dialogue_turn(state, "首轮问题", object())
        state, _ = answer_broad_dialogue_turn(
            state, "徐陵和庾信齐名，他们的文体合称为什么？", object())
        assert calls == [
            "首轮问题", "徐陵和庾信齐名，他们的文体合称为什么？"
        ]
    finally:
        module.query_broad_qa = original


def test_marker_projection_is_opt_in_and_keeps_full_evidence() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    class _Citation:
        selected_text = "档案说明，由某机构负责。"

    class _Result:
        status = "ANSWER"
        answer = "档案说明，由某机构负责。\n相邻证据句。"
        title = "公开来源"
        source_url = "https://example.invalid/source"
        evidence_chain = (_Citation(),)

    relation = learn_relation_evidence_model(_RELATION_COURSES)
    marker = learn_relation_marker_evidence_model((_MARKER_COURSE,))
    original = module.query_broad_qa
    module.query_broad_qa = lambda *_args, **_kwargs: _Result()
    try:
        state, turn = answer_broad_dialogue_turn(
            BroadDialogueState((11, 12, 13)), "该事项由谁负责？", object(),
            learned_relation_evidence_model=relation,
            learned_relation_marker_evidence_model=marker,
        )
        assert turn.display_answer == "某机构"
        assert turn.answer == "档案说明，由某机构负责。\n相邻证据句。"
        assert state.turns[-1] == turn
    finally:
        module.query_broad_qa = original


def test_relation_frame_wraps_only_a_unique_source_value() -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    class _Citation:
        selected_text = "档案说明，由某机构负责。"

    class _Result:
        status = "ANSWER"
        answer = "档案说明，由某机构负责。\n相邻证据句。"
        title = "公开事项"
        source_url = "https://example.invalid/source"
        evidence_chain = (_Citation(),)

    relation = learn_relation_evidence_model(_RELATION_COURSES)
    marker = learn_relation_marker_evidence_model((_MARKER_COURSE,))
    frame = learn_relation_answer_frame_model((_ANSWER_FRAME_COURSE,))
    original = module.query_broad_qa
    module.query_broad_qa = lambda *_args, **_kwargs: _Result()
    try:
        _, turn = answer_broad_dialogue_turn(
            BroadDialogueState((14, 15, 16)), "该事项由谁负责？", object(),
            learned_relation_evidence_model=relation,
            learned_relation_marker_evidence_model=marker,
            learned_relation_answer_frame_model=frame,
        )
        assert turn.display_answer == "公开事项由某机构负责。"
        assert turn.answer == "档案说明，由某机构负责。\n相邻证据句。"
    finally:
        module.query_broad_qa = original
