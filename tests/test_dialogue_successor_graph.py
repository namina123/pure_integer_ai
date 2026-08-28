"""普通结构化对话后继图的训练、回源和终端查询专项。"""
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.experiments.collection import (
    CollectedItem,
    DialogueContentSpan,
    SpeakerSpan,
)
from pure_integer_ai.experiments.dialogue_successor_graph import (
    DialogueSuccessorProtocol,
    SqliteDialogueSuccessorRuntime,
    install_dialogue_successor_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


def _item(source_id: int, prompt: str, response: str,
          *, history: tuple[str, str] | None = None) -> CollectedItem:
    """构造带显式 speaker/content span 的公开训练项。"""
    turns = []
    if history is not None:
        turns.extend(((1, history[0]), (2, history[1])))
    turns.extend(((1, prompt), (2, response)))
    rendered = tuple(surface for _role, surface in turns)
    raw_text = "\n".join(rendered)
    speaker_spans = []
    content_spans = []
    cursor = 0
    for index, ((role, surface), value) in enumerate(
            zip(turns, rendered), start=1):
        content_start = cursor
        content_end = cursor + len(value)
        turn_end = content_end + (1 if index < len(turns) else 0)
        speaker_spans.append(SpeakerSpan(
            cursor, turn_end, index, role_identity((9091, role))))
        content_spans.append(DialogueContentSpan(
            content_start, content_end, index))
        cursor = turn_end
    return CollectedItem(
        tokens=list(raw_text),
        raw_text=raw_text,
        role_seq=[1] * len(raw_text),
        source=SOURCE_BARE_TEXT,
        source_ref=SourceRef(
            SOURCE_BARE_TEXT, source_id, 1,
            GLOBAL_OWNER_SCOPE, VersionBundle()),
        speaker_spans=tuple(speaker_spans),
        dialogue_content_spans=tuple(content_spans),
    )


def test_dialogue_successor_is_learned_in_core_and_recovered_from_occurrences(
        tmp_path):
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        ctx = make_train_context(backend)
        install_language_graph_protocols(
            ctx,
            occurrence_protocol=OccurrenceProtocol((9092, 1), (9092, 2)),
            occurrence_order_protocol=OccurrenceOrderProtocol((9092, 3)),
        )
        runtime = install_dialogue_successor_runtime(
            ctx, DialogueSuccessorProtocol((9092, 10), EPI_STRUCTURED))
        runner = DefaultRoundRunner()
        training_item = _item(
            1, "你好，请问可以聊天吗？", "当然可以，你想聊些什么？")
        result = runner.run_round_full(
            ctx,
            training_item,
            1,
            1,
        )
        assert result.episode is None
        assert runtime.counts()[0] == 1
        assert len(ctx.dialogue_successor_reports) == 1
        assert ctx.dialogue_successor_reports[0].response_codepoints > 0
        runner.run_round_full(ctx, training_item, 2, 2)
        assert runtime.counts()[0] == 1
        assert ctx.dialogue_successor_reports[-1].replayed
        backend.commit()
    finally:
        backend.close()

    query = SqliteDialogueSuccessorRuntime(database)
    try:
        learned = query.respond("你好，请问可以聊天吗？")
        assert learned is not None
        assert learned.surface == "当然可以，你想聊些什么？"
        assert learned.similarity_permille == 1000
        assert learned.source_hash > 0
        assert query.respond("火星轨道参数是什么？") is None
    finally:
        query.close()


def test_multiturn_history_breaks_same_prompt_tie_without_copying_answer_index(
        tmp_path):
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        ctx = make_train_context(backend)
        install_language_graph_protocols(
            ctx,
            occurrence_protocol=OccurrenceProtocol((9192, 1), (9192, 2)),
            occurrence_order_protocol=OccurrenceOrderProtocol((9192, 3)),
        )
        install_dialogue_successor_runtime(
            ctx, DialogueSuccessorProtocol((9192, 10), EPI_STRUCTURED))
        runner = DefaultRoundRunner()
        runner.run_round_full(
            ctx,
            _item(11, "那接下来呢？", "接下来检查电源连接。",
                  history=("设备无法启动。", "先确认电源指示灯。")),
            1, 1)
        runner.run_round_full(
            ctx,
            _item(12, "那接下来呢？", "接下来加入面粉并搅拌。",
                  history=("我正在做蛋糕。", "先把鸡蛋打散。")),
            1, 2)
        backend.commit()
    finally:
        backend.close()

    query = SqliteDialogueSuccessorRuntime(database)
    try:
        result = query.respond(
            "那接下来呢？",
            history=((1, "设备无法启动。"), (2, "先确认电源指示灯。")),
        )
        assert result is not None
        assert result.surface == "接下来检查电源连接。"
        assert result.history_similarity_permille > 0
    finally:
        query.close()
