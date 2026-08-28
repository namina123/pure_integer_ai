from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    DialogueCitation,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.scidb_csq_passage_index import (
    ScidbCsqPassageRuntime,
    build_scidb_csq_passage_index,
)


_SOURCE_SHA = "ab" * 32
_DOI = "10.57760/sciencedb.22816"
_URL = "https://www.scidb.cn/detail?dataSetId=test"


def _row(record_id: int, split: str, *, theme: str, category: str,
         question: str, knowledge: str) -> dict[str, object]:
    record_sha = hashlib.sha256(
        f"source:{record_id}".encode("utf-8")).hexdigest()
    return {
        "answer_surface": "解题过程：测试过程。\n答案：1",
        "context_surface": (
            f"学段：小学\n领域：{theme}\n主题：{category}\n"
            f"知识点：{knowledge}\n提示：测试提示\n科学技能：观察"),
        "course_version": 1,
        "family": "scidb-csq-v1",
        "format": "PURE_INTEGER_AI_SCIDB_CSQ_COURSE_V1",
        "license_id": "CC-BY-4.0",
        "question_surface": question + "\n选项：[甲,乙]",
        "sample_id": f"{_DOI}:{record_id}:{record_sha}",
        "sample_kind": "POSITIVE",
        "sample_role": "support",
        "source_dataset_doi": _DOI,
        "source_dataset_url": _URL,
        "source_id": 77,
        "source_kind": 205,
        "source_record_id": record_id,
        "source_ref_key": [205, 77, record_id, 0, 0, 0, 1, 0, 0, 0, 0],
        "source_sha256": _SOURCE_SHA,
        "split": split,
    }


def _course(path: Path) -> str:
    rows = (
        _row(
            1, "train", theme="地球与宇宙科学", category="地球系统",
            question="天空为什么呈现蓝色？",
            knowledge=("天空呈现蓝色是因为阳光通过大气层时，短波长的蓝光"
                       "更容易被空气分子散射到各个方向。"),
        ),
        _row(
            2, "train", theme="生命科学", category="植物",
            question="叶片为什么通常是绿色？",
            knowledge=("叶绿素吸收红光和蓝光并反射绿光，因此叶片通常呈现绿色。"),
        ),
        _row(
            4, "train", theme="地球与宇宙科学", category="恒星",
            question="北极星为什么看起来不动？",
            knowledge=("北极星位于北侧天空，接近地球自转轴，因此看起来"
                       "几乎不动，可以用于辨认方向。"),
        ),
        _row(
            5, "train", theme="物质科学", category="水",
            question="大海为什么看起来是蓝色的？",
            knowledge=("大海呈现蓝色与水对光的散射和吸收有关，短波长蓝光"
                       "更容易进入观察者眼睛。"),
        ),
        _row(
            3, "heldout", theme="技术与工程", category="机械",
            question="滑轮有什么作用？",
            knowledge="定滑轮可以改变力的方向。",
        ),
    )
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_build_query_excludes_questions_answers_and_heldout(
        tmp_path: Path) -> None:
    course = tmp_path / "course.jsonl"
    course_sha = _course(course)
    root = tmp_path / "artifact"
    report = build_scidb_csq_passage_index(
        course_path=course,
        artifact_root=root,
        expected_course_sha256=course_sha,
        expected_source_sha256=_SOURCE_SHA,
        require_k_drive=False,
    )
    assert report["train_count"] == 4
    assert report["heldout_count"] == 1
    connection = sqlite3.connect(str(
        root / "scidb_csq_passage_index.sqlite3"))
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM document").fetchone() == (4,)
        assert connection.execute(
            "SELECT COUNT(*) FROM passage").fetchone() == (4,)
        assert connection.execute(
            "SELECT COUNT(*) FROM passage WHERE text LIKE '%答案%'").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM passage WHERE text LIKE '%滑轮%'").fetchone() == (0,)
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(document)")}
        assert "question" not in columns
        assert "answer" not in columns
    finally:
        connection.close()
    with ScidbCsqPassageRuntime(
            root, require_k_drive=False) as runtime:
        result = runtime.query(
            "为什么天空通常看起来是蓝色的？",
            minimum_margin_permille=1000,
        )
        short_result = runtime.query(
            "天空为什么是蓝色的？", minimum_margin_permille=1000)
        second_subject = runtime.query(
            "大海为什么是蓝色的？", minimum_margin_permille=1000)
        unsupported_subject = runtime.query(
            "澄明子为什么是蓝色的？", minimum_margin_permille=1000)
        heldout = runtime.query(
            "滑轮有什么作用？", minimum_margin_permille=1000)
    assert result.status == "ANSWER", result
    assert result.surface is not None and "空气分子" in result.surface
    assert result.source_ref is not None and len(result.source_ref) == 11
    assert result.license_id == "CC-BY-4.0"
    assert result.attribution
    assert short_result.status == "ANSWER", short_result
    assert short_result.surface is not None and "空气分子" in short_result.surface
    assert second_subject.status == "ANSWER", second_subject
    assert second_subject.surface is not None and "水对光" in second_subject.surface
    assert unsupported_subject.status == "UNKNOWN"
    assert heldout.status == "UNKNOWN"


def test_source_passage_only_promotes_unanswered_broad_result(monkeypatch) -> None:
    class _Unknown:
        status = "UNKNOWN"
        answer = None
        title = None
        source_url = None
        evidence_chain = ()

    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module
    monkeypatch.setattr(module, "query_broad_qa", lambda *_args: _Unknown())
    citation = DialogueCitation(
        "可核验知识。", "CSQ 记录", "https://example.test/csq")
    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((1, 2, 3)), "知识问题", object(),
        source_passage_response=lambda _question: (
            "ANSWER", "可核验知识。", "CSQ 记录",
            "https://example.test/csq", (citation,),
        ),
        learned_dialogue_answer=lambda _question: "不应覆盖。",
    )
    assert turn.status == "ANSWER"
    assert turn.answer == "可核验知识。"
    assert turn.citations == (citation,)


def test_fast_source_passage_priority_skips_broad_when_cited(monkeypatch) -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    def unexpected_broad(*_args, **_kwargs):
        raise AssertionError("cited source passage should short-circuit broad QA")

    monkeypatch.setattr(module, "query_broad_qa", unexpected_broad)
    citation = DialogueCitation(
        "来源知识。", "CSQ 记录", "https://example.test/csq")
    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((1, 2, 3)), "科学问题", object(),
        source_passage_response=lambda _question: (
            "ANSWER", "来源知识。", "CSQ 记录",
            "https://example.test/csq", (citation,),
        ),
        prefer_source_passage=True,
    )
    assert turn.status == "ANSWER"
    assert turn.answer == "来源知识。"
    assert turn.citations == (citation,)


def test_fast_structured_question_prefers_exact_broad_source(monkeypatch) -> None:
    import pure_integer_ai.experiments.conversation_broad_qa_runtime as module

    calls = []

    class _BroadAnswer:
        status = "ANSWER"
        answer = "比赛资料。"
        title = "2012年俄羅斯羽毛球大獎賽"
        source_url = "https://example.test/wikipedia"
        evidence_chain = ()

    monkeypatch.setattr(
        module, "query_broad_qa",
        lambda _database, question, **_kwargs: (
            calls.append(("broad", question)) or _BroadAnswer()),
    )
    monkeypatch.setattr(
        module, "has_exact_broad_qa_title",
        lambda _database, _question: True,
    )
    citation = DialogueCitation(
        "无关科学资料。", "CSQ 记录", "https://example.test/csq")

    def source_passage(_question):
        calls.append(("source", _question))
        return (
            "ANSWER", "无关科学资料。", "CSQ 记录",
            "https://example.test/csq", (citation,),
        )

    _state, turn = answer_broad_dialogue_turn(
        BroadDialogueState((1, 2, 3)),
        "2012年俄羅斯羽毛球大獎賽是什么？", object(),
        source_passage_response=source_passage,
        prefer_source_passage=True,
        fast_path=True,
    )
    assert calls == [
        ("broad", "2012年俄羅斯羽毛球大獎賽是什么？"),
    ]
    assert turn.status == "ANSWER"
    assert turn.answer == "比赛资料。"
    assert turn.source_title == "2012年俄羅斯羽毛球大獎賽"
    assert turn.citations == ()
