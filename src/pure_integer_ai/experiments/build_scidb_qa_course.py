"""把已下载的公开 SciDB QA JSONL 转成可训练的整数课程交换层。

转换器只处理公开数据文件，不调用模型、不生成答案。每条课程记录保留
原始上下文、问题、答案和来源 URL，并以确定性 SHA-256 桶划分 train/
heldout/negative；原始文件和转换 manifest 必须留在 K: 盘。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "PURE_INTEGER_AI_SCIDB_QA_COURSE_V1"
LICENSE_ID = "CC-BY-4.0"
SOURCE_KIND = 203


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _bucket(identity: str) -> int:
    payload = b"PURE-INTEGER-AI/SCIDB-QA/COURSE/V1" + identity.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 100


def _text(value: object, *, label: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 缺失或为空: line {line}")
    return value.strip()


def build_scidb_qa_course(
        source: str | Path,
        output: str | Path,
        *,
        doi: str,
        dataset_url: str,
        source_md5: str,
        source_sha256: str | None = None,
        heldout_percent: int = 10,
        negative_percent: int = 5,
        max_records: int | None = None,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """Convert one downloaded QA JSONL into a deterministic public course."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if require_k_drive and (source_path.drive.upper() != "K:"
                            or output_path.drive.upper() != "K:"):
        raise ValueError("SciDB source/output 必须位于 K 盘")
    if not source_path.is_file():
        raise ValueError(f"SciDB source 不存在: {source_path}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not isinstance(doi, str) or not doi.strip():
        raise ValueError("doi 不得为空")
    if not isinstance(dataset_url, str) or not dataset_url.startswith("https://"):
        raise ValueError("dataset_url 必须是 HTTPS")
    if not isinstance(source_md5, str) or len(source_md5) != 32:
        raise ValueError("source_md5 必须是 32 位十六进制")
    source_md5 = source_md5.lower()
    if any(char not in "0123456789abcdef" for char in source_md5):
        raise ValueError("source_md5 非法")
    source_payload = source_path.read_bytes()
    actual_md5 = _md5(source_payload)
    if actual_md5 != source_md5:
        raise ValueError(
            f"SciDB source MD5 不匹配: expected={source_md5}, actual={actual_md5}")
    if source_sha256 is None:
        source_sha256 = _sha256(source_payload)
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256.lower())):
        raise ValueError("source_sha256 非法")
    source_sha256 = source_sha256.lower()
    if type(heldout_percent) is not int or type(negative_percent) is not int:
        raise TypeError("split 百分比必须是整数")
    if (heldout_percent < 1 or negative_percent < 0
            or heldout_percent + negative_percent >= 100):
        raise ValueError("split 百分比非法")
    if (max_records is not None
            and (type(max_records) is not int or max_records <= 0)):
        raise ValueError("max_records 必须为正整数")
    records: list[dict[str, object]] = []
    counts = {"train": 0, "heldout": 0, "negative": 0}
    upstream_url_fallback_count = 0
    raw_lines = source_payload.splitlines()
    if raw_lines and raw_lines[0].startswith(b"\xef\xbb\xbf"):
        raise ValueError("SciDB source 不得含 UTF-8 BOM")
    source_id = int.from_bytes(
        hashlib.sha256((doi + "\n" + source_sha256).encode("utf-8")).digest()[:8],
        "big") & ((1 << 63) - 1)
    source_id = source_id or 1
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"SciDB JSONL 非法: line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"SciDB record 必须是对象: line {line_number}")
        question = _text(row.get("question"), label="question", line=line_number)
        answer = _text(row.get("answer"), label="answer", line=line_number)
        context = _text(row.get("正文"), label="正文", line=line_number)
        raw_title = row.get("标题")
        if isinstance(raw_title, str) and raw_title.strip():
            title = raw_title.strip()
            title_fallback = False
        else:
            # Some source blocks omit a page title but retain a stable section
            # title.  Keep that explicit fallback in the provenance record.
            title = _text(row.get("段落标题") or row.get("来源"),
                          label="段落标题/来源", line=line_number)
            title_fallback = True
        raw_upstream_url = row.get("链接")
        if isinstance(raw_upstream_url, str) and raw_upstream_url.strip():
            upstream_url = raw_upstream_url.strip()
            upstream_url_fallback = False
        else:
            # A missing row-level URL is not silently invented: the stable
            # dataset detail URL remains the explicit provenance fallback.
            upstream_url = dataset_url
            upstream_url_fallback = True
            upstream_url_fallback_count += 1
        identity = f"{doi}:{line_number}:{_sha256(raw)}"
        bucket = _bucket(identity)
        if bucket < negative_percent:
            split = "negative"
        elif bucket < negative_percent + heldout_percent:
            split = "heldout"
        else:
            split = "train"
        record: dict[str, object] = {
            "course_version": 1,
            "format": FORMAT,
            "family": "scidb-wushuqa-v1",
            "sample_id": identity,
            "sample_kind": "NEGATIVE" if split == "negative" else "POSITIVE",
            "sample_role": "negative" if split == "negative" else "support",
            "split": split,
            "license_id": LICENSE_ID,
            "source_kind": SOURCE_KIND,
            "source_id": source_id,
            "source_record_id": line_number,
            "source_ref_key": [SOURCE_KIND, source_id, line_number, 0, 0, 0, 0, 0],
            "source_dataset_doi": doi,
            "source_dataset_url": dataset_url,
            "source_url": upstream_url,
            "source_url_fallback": upstream_url_fallback,
            "source_title": title,
            "source_title_fallback": title_fallback,
            "source_sha256": source_sha256,
            "question_surface": question,
            "context_surface": context,
        }
        if split != "negative":
            record["answer_surface"] = answer
        records.append(record)
        counts[split] += 1
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise ValueError("SciDB source 没有可转换记录")
    payload = b"".join(_canonical(item) for item in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "source": {
            "path": output_path.name,
            "doi": doi,
            "dataset_url": dataset_url,
            "license_id": LICENSE_ID,
            "source_md5": source_md5,
            "source_sha256": source_sha256,
        },
        "course": {
            "path": output_path.name,
            "sha256": _sha256(payload),
            "record_count": len(records),
            "split_counts": [[key, counts[key]]
                             for key in ("train", "heldout", "negative")],
            "heldout_percent": heldout_percent,
            "negative_percent": negative_percent,
            "max_records": max_records,
            "source_kind": SOURCE_KIND,
            "source_id": source_id,
            "source_url_fallback_count": upstream_url_fallback_count,
        },
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest_payload = _canonical(manifest)
    manifest_path.write_bytes(manifest_payload)
    return {
        "course_path": output_path.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "course_sha256": _sha256(payload),
        "manifest_sha256": _sha256(manifest_payload),
        "record_count": len(records),
        "split_counts": manifest["course"]["split_counts"],
        "source_md5": source_md5,
        "source_sha256": source_sha256,
        "source_url_fallback_count": upstream_url_fallback_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert SciDB QA JSONL to public course")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--source-md5", required=True)
    parser.add_argument("--source-sha256", default=None)
    parser.add_argument("--heldout-percent", type=int, default=10)
    parser.add_argument("--negative-percent", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_scidb_qa_course(
        args.source, args.output, doi=args.doi, dataset_url=args.dataset_url,
        source_md5=args.source_md5, source_sha256=args.source_sha256,
        heldout_percent=args.heldout_percent,
        negative_percent=args.negative_percent, max_records=args.max_records,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_scidb_qa_course", "main"]
