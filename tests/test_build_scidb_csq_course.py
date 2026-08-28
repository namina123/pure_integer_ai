import hashlib
import json

from pure_integer_ai.experiments.build_scidb_csq_course import (
    build_scidb_csq_course,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)


def _row(*, theme="生命科学", with_options=True):
    row = {
        "年级": "小学三年级上册",
        "任务": "选择题",
        "题目": "哪一项是正确现象？",
        "答案": 0,
        "提示": "观察现象。",
        "知识点": "现象来自公开课程。",
        "解题思路": "比较选项后选择第一项。",
        "主题": theme,
        "类别": "观察",
        "技能": ["比较", "归纳"],
    }
    if with_options:
        row["选项"] = ["甲", "乙"]
    return row


def _source(tmp_path):
    value = {
        "1": _row(),
        "2": _row(theme="错位主题"),
        "3": _row(with_options=False),
        "9601": _row(theme="地球与宇宙"),
    }
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    path = tmp_path / "CSQ.json"
    path.write_bytes(payload)
    return path, payload


def test_filters_invalid_rows_and_preserves_official_holdout(tmp_path):
    source, payload = _source(tmp_path)
    output = tmp_path / "course.jsonl"
    result = build_scidb_csq_course(
        source, output,
        doi="10.57760/example",
        dataset_url="https://example.org/csq",
        source_md5=hashlib.md5(payload).hexdigest(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        require_k_drive=False,
    )
    assert result["record_count"] == 2
    assert result["split_counts"] == [["train", 1], ["heldout", 1]]
    assert dict(result["excluded_counts"]) == {
        "REQUIRED_FIELD_MISSING": 1,
        "THEME_INVALID": 1,
    }
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert rows[0]["split"] == "train"
    assert rows[1]["split"] == "heldout"
    assert all(len(row["source_ref_key"]) == 11 for row in rows)
    assert "解题过程" in rows[0]["answer_surface"]
    pack = load_dialogue_training_pack((output,))
    assert dict(pack.split_counts) == {
        "train": 1, "heldout": 1, "negative": 0,
    }
    assert all(case.source_ref is not None for case in pack.cases)


def test_replay_is_deterministic(tmp_path):
    source, payload = _source(tmp_path)
    first = build_scidb_csq_course(
        source, tmp_path / "first.jsonl",
        doi="10.57760/example", dataset_url="https://example.org/csq",
        source_md5=hashlib.md5(payload).hexdigest(), require_k_drive=False,
    )
    second = build_scidb_csq_course(
        source, tmp_path / "second.jsonl",
        doi="10.57760/example", dataset_url="https://example.org/csq",
        source_md5=hashlib.md5(payload).hexdigest(), require_k_drive=False,
    )
    assert first["course_sha256"] == second["course_sha256"]
