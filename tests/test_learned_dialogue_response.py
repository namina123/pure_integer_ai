from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.build_learned_dialogue_response_artifact import (
    build_learned_dialogue_response_artifact,
    load_learned_dialogue_response_artifact,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    DialogueResponseTrainingRow,
    LearnedDialogueIntentRuntime,
    LearnedDialogueResponseModel,
    LearnedDialogueResponseRuntime,
    dialogue_prompt_features,
    dialogue_intent_features,
    learn_dialogue_intent_model,
    learn_dialogue_response_model,
)
from pure_integer_ai.experiments.conversation_dialogue_experts import (
    LearnedDialogueExpertRouter,
)
from pure_integer_ai.experiments.sqlite_learned_dialogue_intent import (
    SqliteLearnedDialogueIntentRuntime,
    build_sqlite_learned_dialogue_intent_index,
)


_SOURCE_SHA = "ab" * 32


def _row(sample_id: str, split: str, prompt: str, response: str) -> dict:
    return {
        "dialogue_turns": [
            {"speaker_role": 1, "surface": prompt, "turn_ordinal": 1},
            {"speaker_role": 2, "surface": response, "turn_ordinal": 2},
        ],
        "format": "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2",
        "human_generated": 1,
        "license_id": "Apache-2.0",
        "response_surface": response,
        "sample_id": sample_id,
        "sample_kind": "POSITIVE",
        "sample_role": "support",
        "source_sha256": _SOURCE_SHA,
        "source_title": "OpenAssistant OASST2",
        "split": split,
    }


def _course(path: Path) -> str:
    rows = (
        _row("a", "train", "你好，很高兴认识你",
             "你好！很高兴认识你。"),
        _row("b", "train", "你好，请问可以聊天吗",
             "你好！当然可以。"),
        _row("c", "train", "没什么，只想和你聊天一点",
             "好呀，我们可以聊聊。"),
        _row("d", "train", "我想和你聊一会儿",
             "好呀，我们可以聊聊。"),
        _row("g", "train", "你好",
             "您好，有什么可以帮助您的吗？"),
        # 来源模型身份必须被训练侧排除，不能迁移为当前系统身份。
        _row("e", "train", "请介绍一下你自己",
             "我是 Open Assistant 的助手。"),
        _row("f", "heldout", "你好，很高兴见到你",
             "你好！有什么可以帮助你的吗？"),
    )
    payload = b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")) + "\n").encode("utf-8")
        for item in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_artifact_roundtrip_aggregates_features_without_prompt_mapping(
        tmp_path: Path) -> None:
    course = tmp_path / "course.jsonl"
    course_sha = _course(course)
    root = tmp_path / "artifact"
    artifact = build_learned_dialogue_response_artifact(
        course_path=course,
        artifact_root=root,
        expected_course_sha256=course_sha,
        require_k_drive=False,
    )
    assert artifact.status == "PASS"
    assert artifact.model.excluded_provider_identity_count == 1
    assert artifact.model.train_count == 5
    assert LearnedDialogueResponseModel.from_integer_stream(
        artifact.model.integer_stream()) == artifact.model
    restored = load_learned_dialogue_response_artifact(
        root, expected_course_sha256=course_sha, require_k_drive=False)
    assert restored.model == artifact.model
    assert restored.intent_model is None
    assert restored.intent_index_path is not None
    sqlite_intent = SqliteLearnedDialogueIntentRuntime(
        restored.intent_index_path, restored.model.fragments)
    sqlite_runtime = LearnedDialogueResponseRuntime(
        restored.model, intent_runtime=sqlite_intent)
    try:
        sqlite_result = sqlite_runtime.respond(
            "你好，请问可以聊天吗", minimum_similarity_permille=500)
        assert sqlite_result.used
    finally:
        sqlite_runtime.close()
    # 模型只保留二/三元特征与回答片段，不登记完整训练 prompt。
    prompt = tuple(ord(item) for item in "你好，很高兴认识你")
    assert prompt not in restored.model.features
    assert prompt not in restored.model.fragments
    runtime = LearnedDialogueResponseRuntime(restored.model)
    result = runtime.respond("你好，很高兴认识你")
    assert result.used
    assert result.surface is not None
    assert "Open Assistant" not in result.surface
    short = runtime.respond("你好", minimum_similarity_permille=500)
    assert short.used
    assert short.surface == "您好，有什么可以帮助您的吗？"


def test_dialogue_surface_allows_learned_organization_and_non_latin_features() -> None:
    rows = (
        DialogueResponseTrainingRow(
            "train", "hola", "Claro, puedo ayudarte.", "Public dialogue"),
        DialogueResponseTrainingRow(
            "train", "bonjour", "Bien sûr, je peux aider.", "Public dialogue"),
        DialogueResponseTrainingRow(
            "heldout", "ciao", "Puedo conversar contigo.", "Public dialogue"),
    )
    model = _response_model(rows, 61)
    runtime = LearnedDialogueResponseRuntime(model)
    result = runtime.respond("hola", minimum_similarity_permille=500)
    assert result.used
    assert result.surface == "Claro, puedo ayudarte."
    # 意图层不得依赖固定 CJK 区间；非拉丁文字同样形成可迁移 scalar 特征。
    assert dialogue_intent_features("Привет")


def test_learned_response_is_only_used_after_broad_unknown(monkeypatch) -> None:
    class _Unknown:
        status = "UNKNOWN"
        answer = None
        title = None
        source_url = None
        evidence_chain = ()

    class _Answer:
        status = "ANSWER"
        answer = "有来源回答。"
        title = "公开来源"
        source_url = "https://example.invalid/source"
        evidence_chain = ()

    class _Clarify:
        status = "CLARIFY"
        answer = None
        title = None
        source_url = None
        evidence_chain = ()

    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    monkeypatch.setattr(module, "query_broad_qa", lambda *_args: _Unknown())
    state, learned = answer_broad_dialogue_turn(
        BroadDialogueState((1, 2, 3)), "你好", object(),
        learned_dialogue_answer=lambda _value: "你好！",
    )
    assert learned.status == "ANSWER"
    assert learned.answer == "你好！"
    assert learned.source_title is None

    monkeypatch.setattr(module, "query_broad_qa", lambda *_args: _Clarify())
    _, conversational = answer_broad_dialogue_turn(
        state, "你好", object(),
        learned_dialogue_clarify_answer=lambda _value: "你好！",
    )
    assert conversational.status == "ANSWER"
    assert conversational.answer == "你好！"

    monkeypatch.setattr(module, "query_broad_qa", lambda *_args: _Answer())
    _, sourced = answer_broad_dialogue_turn(
        state, "有来源的问题", object(),
        learned_dialogue_answer=lambda _value: "不应覆盖。",
    )
    assert sourced.answer == "有来源回答。"
    assert sourced.source_title == "公开来源"


def test_learned_response_cannot_override_knowledge_or_non_real_gate(
        monkeypatch) -> None:
    class _Unknown:
        status = "UNKNOWN"
        answer = None
        title = None
        source_url = None
        evidence_chain = ()

    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module
    monkeypatch.setattr(module, "query_broad_qa", lambda *_args: _Unknown())
    state = BroadDialogueState((7, 8, 9))
    _, knowledge = answer_broad_dialogue_turn(
        state, "天空为什么是蓝色的？", object(),
        learned_dialogue_answer=lambda _value: "不应成为无来源事实。",
    )
    _, fictional = answer_broad_dialogue_turn(
        state, "虚构的澄明子为什么会发光？", object(),
        learned_dialogue_answer=lambda _value: "不应覆盖虚构拒答。",
    )
    assert knowledge.status == "UNKNOWN"
    assert fictional.status == "UNKNOWN"


def test_fast_learned_priority_skips_broad_only_for_dialogue_eligible(
        monkeypatch) -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    calls = []

    def broad_unknown(*_args, **_kwargs):
        calls.append("broad")
        return type("Unknown", (), {
            "status": "UNKNOWN", "answer": None, "title": None,
            "source_url": None, "evidence_chain": (),
        })()

    monkeypatch.setattr(module, "query_broad_qa", broad_unknown)
    state = BroadDialogueState((7, 8, 9))
    _, greeting = answer_broad_dialogue_turn(
        state, "你好", object(),
        learned_dialogue_answer=lambda _value: "你好！",
        prefer_learned_dialogue=True,
    )
    assert greeting.status == "ANSWER"
    assert not calls
    _, knowledge = answer_broad_dialogue_turn(
        state, "天空为什么是蓝色的？", object(),
        learned_dialogue_answer=lambda _value: "不应接管。",
        prefer_learned_dialogue=True,
    )
    assert knowledge.status == "UNKNOWN"
    assert calls == ["broad"]


def test_fast_bounded_dialogue_miss_avoids_heavy_retrieval(monkeypatch) -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    def unexpected_broad(*_args, **_kwargs):
        raise AssertionError("bounded dialogue miss should not query broad QA")

    monkeypatch.setattr(module, "query_broad_qa", unexpected_broad)
    _, turn = answer_broad_dialogue_turn(
        BroadDialogueState((7, 8, 9)), "尚未覆盖的闲聊", object(),
        learned_dialogue_answer=lambda _value: None,
        source_passage_response=lambda _value: None,
        prefer_learned_dialogue=True,
        prefer_source_passage=True,
        fast_path=True,
        defer_narrow=True,
        narrow_answer=lambda _value: (_ for _ in ()).throw(
            AssertionError("bounded dialogue miss should not build narrow QA")),
    )
    assert turn.status == "UNKNOWN"
    assert turn.retrieval_question == "尚未覆盖的闲聊"


def _response_model(
        rows: tuple[DialogueResponseTrainingRow, ...], seed: int,
        ) -> LearnedDialogueResponseModel:
    return learn_dialogue_response_model(
        rows, course_sha256=tuple([seed] * 32),
        source_sha256s=(tuple([seed + 1] * 32),))


def test_unique_weak_intent_candidate_requires_absolute_similarity(
        tmp_path: Path) -> None:
    rows = (
        DialogueResponseTrainingRow(
            "train",
            "请帮我制定一个包含准备材料检查步骤执行顺序风险说明和最终复核的完整项目计划",
            "可以先明确目标，再列出步骤和检查点。", "Public dialogue"),
        DialogueResponseTrainingRow(
            "train",
            "请帮我制定一个包含需求访谈原型评审实施排期风险预案和验收标准的完整产品计划",
            "可以先明确目标，再列出步骤和检查点。", "Public dialogue"),
        DialogueResponseTrainingRow(
            "train",
            "请帮我制定一个包含资料检索证据整理交叉核对风险边界和结论复查的完整研究计划",
            "可以先明确目标，再列出步骤和检查点。", "Public dialogue"),
        DialogueResponseTrainingRow(
            "heldout", "请帮我安排一份完整计划", "先明确目标和约束。",
            "Public dialogue"),
    )
    model = _response_model(rows, 51)
    intent = learn_dialogue_intent_model(rows, model)
    memory_runtime = LearnedDialogueIntentRuntime(intent, model.fragments)
    weak_memory = memory_runtime.rank(
        "请帮我制定", minimum_similarity_permille=0)
    assert weak_memory is not None
    assert weak_memory[1] < 500
    assert memory_runtime.rank(
        "请帮我制定", minimum_similarity_permille=500) is None

    index = build_sqlite_learned_dialogue_intent_index(
        tmp_path / "intent.sqlite3", intent)
    sqlite_runtime = SqliteLearnedDialogueIntentRuntime(index, model.fragments)
    try:
        weak_sqlite = sqlite_runtime.rank(
            "请帮我制定", minimum_similarity_permille=0)
        assert weak_sqlite is not None
        assert weak_sqlite[1] == weak_memory[1]
        assert sqlite_runtime.rank(
            "请帮我制定", minimum_similarity_permille=500) is None
    finally:
        sqlite_runtime.close()


def test_general_expert_precedes_learned_domain_activation() -> None:
    general = _response_model((
        DialogueResponseTrainingRow(
            "train", "我今天有点累想聊聊", "可以，我在听。", "General"),
        DialogueResponseTrainingRow(
            "train", "今天有点累陪我聊天", "可以，我在听。", "General"),
        DialogueResponseTrainingRow(
            "train", "你好可以聊天吗", "当然可以。", "General"),
        DialogueResponseTrainingRow(
            "heldout", "你好想聊一会儿", "当然可以。", "General"),
    ), 11)
    domain = _response_model((
        DialogueResponseTrainingRow(
            "train", "星河号电影好看吗",
            "这部作品值得从剧情和人物聊起。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "你觉得星河号电影如何",
            "这部作品值得从剧情和人物聊起。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "山海录的主演是谁", "可以先谈谈作品本身。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "长风曲的情节怎样", "可以先谈谈作品本身。", "Domain"),
        DialogueResponseTrainingRow(
            "heldout", "星河号值得看吗",
            "这部作品值得从剧情和人物聊起。", "Domain"),
    ), 21)
    router = LearnedDialogueExpertRouter(
        LearnedDialogueResponseRuntime(general),
        (LearnedDialogueResponseRuntime(domain),))
    general_result = router.respond(
        "我今天有点累想聊聊", minimum_similarity_permille=500)
    assert general_result.surface == "可以，我在听。"
    domain_result = router.respond(
        "星河号电影好看吗", minimum_similarity_permille=500)
    assert domain_result.used
    assert domain_result.reason.startswith("learned_domain_expert_")


def test_loaded_lazy_domain_still_requires_current_activation() -> None:
    general = _response_model((
        DialogueResponseTrainingRow(
            "train", "我今天有点累想聊聊", "可以，我在听。", "General"),
        DialogueResponseTrainingRow(
            "train", "今天有点累陪我聊天", "可以，我在听。", "General"),
        DialogueResponseTrainingRow(
            "heldout", "你好想聊一会儿", "当然可以。", "General"),
    ), 31)
    domain = _response_model((
        DialogueResponseTrainingRow(
            "train", "星河号电影好看吗", "可以先聊作品本身。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "星河号电影谁导演的", "可以先聊主创。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "星河号电影编剧是谁", "可以先聊主创。", "Domain"),
        DialogueResponseTrainingRow(
            "heldout", "星河号电影值得看吗", "可以先聊作品本身。", "Domain"),
    ), 41)
    activation = frozenset(
        feature for feature in dialogue_prompt_features("星河号电影好看吗")
        if len(feature) >= 2)
    loaded = []

    def load_domain() -> LearnedDialogueResponseRuntime:
        loaded.append(1)
        return LearnedDialogueResponseRuntime(domain)

    router = LearnedDialogueExpertRouter(
        LearnedDialogueResponseRuntime(general), (),
        lazy_domains=((activation, load_domain),))
    first = router.respond(
        "星河号电影好看吗", minimum_similarity_permille=500)
    assert first.reason.startswith("learned_domain_expert_")
    assert loaded == [1]

    follow_up = router.respond(
        "那编剧是谁呀",
        history=((1, "星河号电影好看吗"), (2, first.surface)),
        minimum_similarity_permille=500)
    assert not follow_up.reason.startswith("learned_domain_expert_")
    assert loaded == [1]


def test_domain_activation_ignores_assistant_surface_history() -> None:
    general = _response_model((
        DialogueResponseTrainingRow(
            "train", "我今天有点累想聊聊", "可以，我在听。", "General"),
        DialogueResponseTrainingRow(
            "train", "你好可以聊天吗", "当然可以。", "General"),
        DialogueResponseTrainingRow(
            "heldout", "你好想聊一会儿", "当然可以。", "General"),
    ), 61)
    domain = _response_model((
        DialogueResponseTrainingRow(
            "train", "星河号电影好看吗", "这部作品值得从剧情和人物聊起。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "星河号电影谁导演的", "可以先聊主创。", "Domain"),
        DialogueResponseTrainingRow(
            "train", "星河号电影编剧是谁", "可以先聊主创。", "Domain"),
        DialogueResponseTrainingRow(
            "heldout", "星河号电影值得看吗", "可以先聊作品本身。", "Domain"),
    ), 71)
    activation = frozenset(
        feature for feature in dialogue_prompt_features("星河号电影好看吗")
        if len(feature) >= 2)
    router = LearnedDialogueExpertRouter(
        LearnedDialogueResponseRuntime(general), (), lazy_domains=(
            (activation, lambda: LearnedDialogueResponseRuntime(domain)),))
    result = router.respond(
        "电影", history=((1, "我想聊天"),
                          (2, "星河号电影值得从剧情和人物聊起。")),
        minimum_similarity_permille=500)
    assert not result.reason.startswith("learned_domain_expert_")
