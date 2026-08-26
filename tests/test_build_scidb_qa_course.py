import hashlib
import json

import pytest

from pure_integer_ai.experiments.build_scidb_qa_course import (
    build_scidb_qa_course,
)


def _row(*, title=True, url=True):
    value = {
        "来源": "公开来源",
        "栏目": "测试",
        "段落标题": "分段",
        "正文": "上下文",
        "question": "问题是什么？",
        "answer": "这是答案。",
    }
    if title:
        value["标题"] = "标题"
    if url:
        value["链接"] = "https://example.org/item"
    return value


def _write_source(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "source.jsonl"
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    path.write_bytes(payload)
    return path, payload


def _build(tmp_path, source, payload, *, output_name="course.jsonl"):
    return build_scidb_qa_course(
        source,
        tmp_path / output_name,
        doi="10.0000/test",
        dataset_url="https://example.org/dataset",
        source_md5=hashlib.md5(payload).hexdigest(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        require_k_drive=False,
    )


def test_conversion_is_deterministic_and_records_fallbacks(tmp_path):
    source, payload = _write_source(tmp_path, [
        _row(title=False, url=False), _row(), _row(title=False, url=True),
    ])
    first = _build(tmp_path, source, payload)
    output = tmp_path / "course.jsonl"
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert first["record_count"] == 3
    assert first["source_url_fallback_count"] == 1
    assert sum(item["split"] == "train" for item in rows) == first["split_counts"][0][1]
    assert rows[0]["source_title"] == "分段"
    assert rows[0]["source_title_fallback"] is True
    assert rows[0]["source_url"] == "https://example.org/dataset"
    assert rows[0]["source_url_fallback"] is True

    source2, payload2 = _write_source(tmp_path / "second", [_row(title=False, url=False), _row(), _row(title=False, url=True)])
    second = _build(tmp_path / "second", source2, payload2)
    assert first["course_sha256"] == second["course_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_md5_mismatch_and_bom_are_rejected(tmp_path):
    source, payload = _write_source(tmp_path, [_row()])
    with pytest.raises(ValueError, match="MD5"):
        build_scidb_qa_course(
            source, tmp_path / "bad.jsonl", doi="10.0000/test",
            dataset_url="https://example.org/dataset", source_md5="0" * 32,
            require_k_drive=False,
        )

    bom = tmp_path / "bom.jsonl"
    bom.write_bytes(b"\xef\xbb\xbf" + payload)
    with pytest.raises(ValueError, match="BOM"):
        build_scidb_qa_course(
            bom, tmp_path / "bom-course.jsonl", doi="10.0000/test",
            dataset_url="https://example.org/dataset",
            source_md5=hashlib.md5(bom.read_bytes()).hexdigest(),
            require_k_drive=False,
        )


def test_negative_and_heldout_percentages_must_be_valid(tmp_path):
    source, payload = _write_source(tmp_path, [_row()])
    with pytest.raises(ValueError, match="split 百分比"):
        build_scidb_qa_course(
            source, tmp_path / "invalid.jsonl", doi="10.0000/test",
            dataset_url="https://example.org/dataset",
            source_md5=hashlib.md5(payload).hexdigest(),
            heldout_percent=100, require_k_drive=False,
        )
