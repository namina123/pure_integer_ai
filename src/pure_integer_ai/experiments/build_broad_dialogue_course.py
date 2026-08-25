"""从公开广域索引生成可重放的语言课程交换文件。

这不是把检索答案硬编码进回答器：课程只把已许可来源的原文段落作为
``Observation`` 输入训练图，问题仍由运行时检索和用户输入决定。抽样、
split、来源 URL 和 hash 均由整数/字节规则确定，可在其他语言重现。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from pure_integer_ai.experiments.integer_token_index import (
    build_integer_token_index, write_integer_token_index,
)


def _bucket(value: int) -> int:
    return int.from_bytes(hashlib.sha256(
        b"PURE-INTEGER-AI/BROAD-COURSE/V1" + value.to_bytes(8, "big")
    ).digest()[:4], "big") % 100


def build_broad_dialogue_course(
        database: str | Path, output: str | Path, *, max_rows: int = 20000,
        heldout_percent: int = 10, negative_percent: int = 5,
        ) -> dict[str, object]:
    """按 passage_id 顺序生成广域课程，返回公开摘要。"""
    if type(max_rows) is not int or not 1 <= max_rows <= 200000:
        raise ValueError("max_rows 必须在 1..200000")
    if type(heldout_percent) is not int or type(negative_percent) is not int:
        raise TypeError("split 百分比必须是整数")
    if heldout_percent < 1 or negative_percent < 0 \
            or heldout_percent + negative_percent >= 100:
        raise ValueError("split 百分比非法")
    source = Path(database).resolve()
    target = Path(output).resolve()
    if not source.is_file() or source.drive.upper() != "K:":
        raise ValueError("广域索引必须是 K 盘已存在 SQLite")
    if target.exists():
        raise FileExistsError(target)
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute(
            "select key,value from metadata order by key").fetchall())
        license_id = metadata.get("license_id")
        snapshot_id = metadata.get("snapshot_id")
        source_key = metadata.get("source_key")
        if not all(isinstance(value, str) and value
                   for value in (license_id, snapshot_id, source_key)):
            raise ValueError("广域索引 metadata 缺少来源身份")
        rows = connection.execute(
            "SELECT p.passage_id,p.text,d.title,d.page_id,d.revision_id "
            "FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id "
            "WHERE length(trim(p.text))>=16 "
            "ORDER BY p.passage_id LIMIT ?", (max_rows,)).fetchall()
    finally:
        connection.close()
    valid_rows = tuple(row for row in rows
                       if isinstance(row[1], str) and row[1].strip())
    texts = tuple(str(row[1]).strip() for row in valid_rows)
    keys = tuple(f"{source_key}:{int(row[0])}" for row in valid_rows)
    token_index = build_integer_token_index(texts, sequence_keys=keys)
    sidecar = target.with_name(target.name + ".tokens.int.json")
    records = []
    counts = {"train": 0, "heldout": 0, "negative": 0}
    for ordinal, (passage_id, text, title, page_id, revision_id) in enumerate(valid_rows):
        bucket = _bucket(int(passage_id))
        if bucket < negative_percent:
            split = "negative"
        elif bucket < negative_percent + heldout_percent:
            split = "heldout"
        else:
            split = "train"
        # Negative records are retained for open-set calibration but carry no
        # fabricated answer; the runtime never treats this course as facts.
        record = {
            "course_version": 1,
            "family": "broad-wikipedia-passage-v1",
            "license_id": license_id,
            "passage_id": int(passage_id),
            "sample_id": f"{source_key}:{int(passage_id)}",
            "sample_kind": "NEGATIVE" if split == "negative" else "POSITIVE",
            "snapshot_id": snapshot_id,
            "source_url": (
                "https://zh.wikipedia.org/w/index.php?curid="
                f"{int(page_id)}&oldid={int(revision_id)}"),
            "split": split,
            "token_index_file": sidecar.name,
            "token_index_ordinal": ordinal,
            "token_index_sha256": token_index.sha256,
            "title": str(title),
        }
        records.append(record)
        counts[split] += 1
    if not records:
        raise ValueError("广域索引没有可消费 passage")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8") + b"\n"
        for item in records)
    target.write_bytes(payload)
    write_integer_token_index(sidecar, token_index)
    return {
        "course_path": target.as_posix(),
        "course_sha256": hashlib.sha256(payload).hexdigest(),
        "record_count": len(records),
        "split_counts": [[key, counts[key]] for key in ("train", "heldout", "negative")],
        "max_rows": max_rows,
        "source_key": source_key,
        "snapshot_id": snapshot_id,
        "license_id": license_id,
        "token_index_path": sidecar.as_posix(),
        "token_vocabulary_count": len(token_index.vocabulary),
        "unique_sequence_count": len(token_index.sequences),
        "occurrence_sequence_count": len(token_index.occurrence_ordinals),
        "token_occurrence_count": token_index.token_count(),
        "raw_surface_bytes": sum(len(item.encode("utf-8")) for item in texts),
        "course_bytes": len(payload),
        "sidecar_bytes": sidecar.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build deterministic broad dialogue course")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=20000)
    args = parser.parse_args(argv)
    print(json.dumps(build_broad_dialogue_course(
        args.database, args.output, max_rows=args.max_rows),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_broad_dialogue_course", "main"]
