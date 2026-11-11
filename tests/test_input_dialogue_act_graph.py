from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.input_dialogue_act_course import (
    load_input_dialogue_act_course,
)
from pure_integer_ai.experiments.input_dialogue_act_graph import (
    TrainedInputDialogueActFrame,
    TrainedInputDialogueActGraph,
    TrainedInputDialogueActLexeme,
    materialize_input_dialogue_act_graph,
    read_input_dialogue_act_graph,
)
from pure_integer_ai.experiments.unknown_response_course import (
    GENERIC_RESPONSE_CONDITION_VERSION,
    RESPONSE_FEATURE_CONVERSATIONAL_OPENING,
    RESPONSE_FEATURE_DIALOGUE_CONTINUATION,
    RESPONSE_FEATURE_DIRECT_DIALOGUE_ACTS,
    RESPONSE_FEATURE_FAREWELL,
    RESPONSE_FEATURE_GRATITUDE,
    RESPONSE_FEATURE_HELP_REQUEST,
    RESPONSE_FEATURE_MULTI_HYPOTHESIS,
    RESPONSE_FEATURE_UNKNOWN,
    load_integer_course,
    response_condition_matches,
    response_condition_priority,
)


_PROJECT = Path(__file__).resolve().parents[1]
_COURSE = _PROJECT / "data/ph2/input_dialogue_act_course_v1.int.json"
_MANIFEST = _PROJECT / "data/ph2/manifests/input_dialogue_act_course_v1.json"
_COMBINED_COURSE = _PROJECT / "data/ph2/input_dialogue_act_course_v2.int.json"
_COMBINED_MANIFEST = _PROJECT / "data/ph2/manifests/input_dialogue_act_course_v2.json"
_HELP_COURSE = _PROJECT / "data/ph2/input_dialogue_act_course_v3.int.json"
_HELP_MANIFEST = _PROJECT / "data/ph2/manifests/input_dialogue_act_course_v3.json"
_RESPONSE_COURSE = _PROJECT / "data/ph2/conversational_opening_response_v1.int.json"
_RESPONSE_MANIFEST = _PROJECT / "data/ph2/manifests/conversational_opening_response_v1.json"
_HELP_RESPONSE_COURSE = _PROJECT / "data/ph2/conversational_help_response_v1.int.json"
_HELP_RESPONSE_MANIFEST = _PROJECT / "data/ph2/manifests/conversational_help_response_v1.json"
_SOCIAL_COURSE = _PROJECT / "data/ph2/input_dialogue_act_course_v4.int.json"
_SOCIAL_MANIFEST = _PROJECT / "data/ph2/manifests/input_dialogue_act_course_v4.json"
_SOCIAL_RESPONSE_COURSE = _PROJECT / "data/ph2/conversational_social_response_v1.int.json"
_SOCIAL_RESPONSE_MANIFEST = _PROJECT / "data/ph2/manifests/conversational_social_response_v1.json"
_GREETING = (91694, 1)
_OPENING_FEATURE = RESPONSE_FEATURE_CONVERSATIONAL_OPENING


def _values(value: str) -> tuple[int, ...]:
    return tuple(map(ord, value))


def test_input_dialogue_act_graph_cold_restores_and_composes_heldout_forms(
        tmp_path):
    course = load_input_dialogue_act_course(_COURSE, _MANIFEST)
    database = tmp_path / "input-dialogue-acts.sqlite3"
    receipt = materialize_input_dialogue_act_graph(
        course.lexemes, course.frames, database,
        source_identity=course.source_identity,
        license_identity=course.license_identity,
        course_sha256=course.course_sha256,
    )
    restored = read_input_dialogue_act_graph(database)

    assert receipt["lexeme_count"] == 13
    assert receipt["frame_count"] == 4
    assert restored.source_identity == course.source_identity
    assert restored.license_identity == course.license_identity
    assert restored.course_sha256 == course.course_sha256
    for surface in ("你好", "哈喽", "早上好", "您好"):
        match = restored.match(_values(surface))
        assert match is not None
        assert match.act_key == _GREETING
        assert match.response_feature_mask == _OPENING_FEATURE
    for heldout in ("你好呀", "您好啊。", "早上好啊！"):
        match = restored.match(_values(heldout))
        assert match is not None
        assert match.act_key == _GREETING
        assert match.response_feature_mask == _OPENING_FEATURE


def test_combined_act_graph_cold_restores_heldout_gratitude_compositions(
        tmp_path):
    course = load_input_dialogue_act_course(
        _COMBINED_COURSE, _COMBINED_MANIFEST)
    database = tmp_path / "combined-dialogue-acts.sqlite3"
    receipt = materialize_input_dialogue_act_graph(
        course.lexemes, course.frames, database,
        source_identity=course.source_identity,
        license_identity=course.license_identity,
        course_sha256=course.course_sha256,
    )
    restored = read_input_dialogue_act_graph(database)

    assert receipt["lexeme_count"] == 19
    assert receipt["frame_count"] == 7
    for surface in ("谢谢", "多谢你", "感谢啦"):
        match = restored.match(_values(surface))
        assert match is not None
        assert match.act_key == (91694, 2)
        assert match.response_feature_mask == RESPONSE_FEATURE_GRATITUDE
    for heldout in ("感谢你", "谢谢您", "多谢啦"):
        match = restored.match(_values(heldout))
        assert match is not None
        assert match.act_key == (91694, 2)
        assert match.response_feature_mask == RESPONSE_FEATURE_GRATITUDE
    for negative in (
            "谢谢是什么意思？", "不是谢谢", "你能帮我吗？", "未知甲表示什么？"):
        assert restored.match(_values(negative)) is None
    for greeting in ("你好", "哈喽", "你好呀"):
        match = restored.match(_values(greeting))
        assert match is not None
        assert match.response_feature_mask == _OPENING_FEATURE


def test_input_dialogue_act_graph_requires_complete_unique_parse():
    graph = TrainedInputDialogueActGraph(
        lexemes=(
            TrainedInputDialogueActLexeme((10,), _values("你好")),
            TrainedInputDialogueActLexeme((11,), _values("呀")),
        ),
        frames=(TrainedInputDialogueActFrame(
            (20,), _OPENING_FEATURE, ((10,),)),
            TrainedInputDialogueActFrame(
                (20,), _OPENING_FEATURE, ((10,), (11,))),
        ),
        source_identity=(1,), license_identity=(2,),
        course_sha256=(0,) * 32, graph_id=1,
    )

    assert graph.match(_values("你好呀")) is not None
    assert graph.match(_values("你好是什么意思")) is None
    assert graph.match(_values("你好呀吗")) is None
    assert graph.match(_values("未知甲表示什么？")) is None


def test_input_dialogue_act_graph_fails_closed_on_competing_acts():
    graph = TrainedInputDialogueActGraph(
        lexemes=(TrainedInputDialogueActLexeme((10,), _values("你好")),),
        frames=(
            TrainedInputDialogueActFrame((20,), _OPENING_FEATURE, ((10,),)),
            TrainedInputDialogueActFrame((21,), _OPENING_FEATURE, ((10,),)),
        ),
        source_identity=(1,), license_identity=(2,),
        course_sha256=(0,) * 32, graph_id=1,
    )

    assert graph.match(_values("你好")) is None


def test_input_dialogue_act_graph_fails_closed_on_ambiguous_lexical_paths():
    graph = TrainedInputDialogueActGraph(
        lexemes=(
            TrainedInputDialogueActLexeme((10,), _values("你好")),
            TrainedInputDialogueActLexeme((11,), _values("你")),
            TrainedInputDialogueActLexeme((12,), _values("好")),
        ),
        frames=(TrainedInputDialogueActFrame(
            (20,), _OPENING_FEATURE, ((10,),)),
            TrainedInputDialogueActFrame(
                (20,), _OPENING_FEATURE, ((11,), (12,))),
        ),
        source_identity=(1,), license_identity=(2,),
        course_sha256=(0,) * 32, graph_id=1,
    )

    assert graph.match(_values("你好")) is None


def test_opening_connector_course_requires_the_trained_act_feature():
    course = load_integer_course(_RESPONSE_COURSE, _RESPONSE_MANIFEST)
    feature_mask = (
        RESPONSE_FEATURE_UNKNOWN
        | RESPONSE_FEATURE_CONVERSATIONAL_OPENING)

    assert len(course.variants) == 4
    assert len(course.conditions) == 4
    assert all(response_condition_matches(condition, feature_mask)
               for condition in course.conditions)
    assert all(not response_condition_matches(
        condition, RESPONSE_FEATURE_UNKNOWN)
        for condition in course.conditions)
    assert all(response_condition_priority(condition, feature_mask)
               > response_condition_priority((), feature_mask)
               for condition in course.conditions)


def test_current_opening_act_outranks_incidental_memory_multiplicity():
    feature_mask = (
        RESPONSE_FEATURE_UNKNOWN
        | RESPONSE_FEATURE_CONVERSATIONAL_OPENING
        | RESPONSE_FEATURE_MULTI_HYPOTHESIS
    )
    opening = (
        GENERIC_RESPONSE_CONDITION_VERSION,
        feature_mask & ~RESPONSE_FEATURE_MULTI_HYPOTHESIS,
        0, 0)
    multi_hypothesis = (
        GENERIC_RESPONSE_CONDITION_VERSION,
        RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_MULTI_HYPOTHESIS,
        0, 0)
    gratitude = (
        GENERIC_RESPONSE_CONDITION_VERSION,
        RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_GRATITUDE,
        0, 0)
    help_request = (
        GENERIC_RESPONSE_CONDITION_VERSION,
        RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_HELP_REQUEST,
        0, 0)

    assert response_condition_matches(opening, feature_mask)
    assert response_condition_matches(multi_hypothesis, feature_mask)
    assert response_condition_priority(opening, feature_mask) > (
        response_condition_priority(multi_hypothesis, feature_mask))
    gratitude_mask = (
        RESPONSE_FEATURE_UNKNOWN
        | RESPONSE_FEATURE_GRATITUDE
        | RESPONSE_FEATURE_MULTI_HYPOTHESIS)
    assert RESPONSE_FEATURE_DIRECT_DIALOGUE_ACTS == (
        RESPONSE_FEATURE_CONVERSATIONAL_OPENING
        | RESPONSE_FEATURE_GRATITUDE
        | RESPONSE_FEATURE_HELP_REQUEST
        | RESPONSE_FEATURE_DIALOGUE_CONTINUATION
        | RESPONSE_FEATURE_FAREWELL)
    assert response_condition_priority(
        gratitude, gratitude_mask) > response_condition_priority(
            multi_hypothesis, gratitude_mask)
    help_mask = (
        RESPONSE_FEATURE_UNKNOWN
        | RESPONSE_FEATURE_HELP_REQUEST
        | RESPONSE_FEATURE_MULTI_HYPOTHESIS)
    assert response_condition_priority(
        help_request, help_mask) > response_condition_priority(
            multi_hypothesis, help_mask)
    for feature in (RESPONSE_FEATURE_DIALOGUE_CONTINUATION,
                    RESPONSE_FEATURE_FAREWELL):
        condition = (
            GENERIC_RESPONSE_CONDITION_VERSION,
            RESPONSE_FEATURE_UNKNOWN | feature,
            0, 0)
        mask = (RESPONSE_FEATURE_UNKNOWN | feature
                | RESPONSE_FEATURE_MULTI_HYPOTHESIS)
        assert response_condition_priority(condition, mask) > (
            response_condition_priority(multi_hypothesis, mask))


def test_social_act_graph_composes_continuation_and_farewell_frames(tmp_path):
    course = load_input_dialogue_act_course(_SOCIAL_COURSE, _SOCIAL_MANIFEST)
    database = tmp_path / "social-dialogue-acts.sqlite3"
    receipt = materialize_input_dialogue_act_graph(
        course.lexemes, course.frames, database,
        source_identity=course.source_identity,
        license_identity=course.license_identity,
        course_sha256=course.course_sha256,
    )
    restored = read_input_dialogue_act_graph(database)
    assert receipt["lexeme_count"] == 43
    assert receipt["frame_count"] == 32
    for heldout in ("再见啦", "我先走了！", "回头见", "下次聊"):
        match = restored.match(_values(heldout))
        assert match is not None, heldout
        assert match.act_key == (91694, 4)
        assert match.response_feature_mask == RESPONSE_FEATURE_FAREWELL
    for heldout in (
            "接着聊吧", "聊下去吧", "咱们继续聊吧", "我们继续聊吧"):
        match = restored.match(_values(heldout))
        assert match is not None, heldout
        assert match.act_key == (91694, 5)
        assert match.response_feature_mask == RESPONSE_FEATURE_DIALOGUE_CONTINUATION
    for negative in (
            "不是再见", "再见是什么意思？", "继续聊是什么意思？",
            "未知甲表示什么？", "帮我关灯"):
        assert restored.match(_values(negative)) is None
    greeting = restored.match(_values("你好"))
    gratitude = restored.match(_values("谢谢"))
    assert greeting is not None
    assert greeting.response_feature_mask == RESPONSE_FEATURE_CONVERSATIONAL_OPENING
    assert gratitude is not None
    assert gratitude.response_feature_mask == RESPONSE_FEATURE_GRATITUDE


def test_social_response_connectors_require_their_current_act():
    course = load_integer_course(
        _SOCIAL_RESPONSE_COURSE, _SOCIAL_RESPONSE_MANIFEST)
    continuation_mask = (
        RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_DIALOGUE_CONTINUATION)
    farewell_mask = RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_FAREWELL
    assert len(course.variants) == 8
    assert len(course.conditions) == 8
    assert all(response_condition_matches(condition, farewell_mask)
               for condition in course.conditions[:4])
    assert all(not response_condition_matches(condition, continuation_mask)
               for condition in course.conditions[:4])
    assert all(response_condition_matches(condition, continuation_mask)
               for condition in course.conditions[4:])
    assert all(not response_condition_matches(condition, farewell_mask)
               for condition in course.conditions[4:])
    assert all(not response_condition_matches(
        condition, RESPONSE_FEATURE_UNKNOWN)
        for condition in course.conditions)


def test_help_request_graph_composes_heldout_forms_and_rejects_lookalikes(
        tmp_path):
    course = load_input_dialogue_act_course(_HELP_COURSE, _HELP_MANIFEST)
    database = tmp_path / "help-request-acts.sqlite3"
    receipt = materialize_input_dialogue_act_graph(
        course.lexemes, course.frames, database,
        source_identity=course.source_identity,
        license_identity=course.license_identity,
        course_sha256=course.course_sha256,
    )
    restored = read_input_dialogue_act_graph(database)
    assert receipt["lexeme_count"] == 30
    assert receipt["frame_count"] == 23
    for surface in ("你能帮我吗", "可以帮我吗", "请帮我一下",
                    "你能不能帮我一下？", "可以帮我一下吗？",
                    "能帮我吗？"):
        match = restored.match(_values(surface))
        assert match is not None
        assert match.act_key == (91694, 3)
        assert match.response_feature_mask == RESPONSE_FEATURE_HELP_REQUEST
    for heldout in ("您能帮帮我吗？", "您可以帮我吗？",
                    "可以帮帮我吗？"):
        match = restored.match(_values(heldout))
        assert match is not None, heldout
        assert match.act_key == (91694, 3)
        assert match.response_feature_mask == RESPONSE_FEATURE_HELP_REQUEST
    for negative in ("你不能帮我吗？", "你能帮我吗是什么意思？",
                     "帮我关灯", "谢谢是什么意思？", "不是谢谢"):
        assert restored.match(_values(negative)) is None
    greeting = restored.match(_values("你好"))
    assert greeting is not None
    assert greeting.response_feature_mask == _OPENING_FEATURE


def test_help_response_connectors_require_the_trained_request_act():
    course = load_integer_course(
        _HELP_RESPONSE_COURSE, _HELP_RESPONSE_MANIFEST)
    help_mask = RESPONSE_FEATURE_UNKNOWN | RESPONSE_FEATURE_HELP_REQUEST
    assert len(course.variants) == 4
    assert all(response_condition_matches(condition, help_mask)
               for condition in course.conditions)
    assert all(not response_condition_matches(
        condition, RESPONSE_FEATURE_UNKNOWN)
        for condition in course.conditions)
    assert all(response_condition_priority(condition, help_mask)
               > response_condition_priority((), help_mask)
               for condition in course.conditions)
