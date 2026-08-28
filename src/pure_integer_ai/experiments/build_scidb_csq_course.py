"""Convert the public SciDB CSQ JSON object to a deterministic science course.

The converter preserves the upstream 80/10/10 benchmark boundary by admitting
only ids 1..9600 as training.  Later ids remain held out.  It never calls a
model, repairs answers, or invents missing fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "PURE_INTEGER_AI_SCIDB_CSQ_COURSE_V1"
LICENSE_ID = "CC-BY-4.0"
SOURCE_KIND = 205
DEFAULT_TRAIN_LAST_ID = 9600
_THEME_MAP = {
    "生命科学": "生命科学",
    "物质科学": "物质科学",
    "技术与工程": "技术与工程",
    "地球与宇宙": "地球与宇宙科学",
    "地球与宇宙科学": "地球与宇宙科学",
    "地球系统与宇宙科学": "地球与宇宙科学",
}
_TASKS = frozenset({"选择题", "判断题"})


def _digest(payload: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def _surface(value: object) -> str | None:
    """Render a scalar/list surface without parsing ad-hoc list strings."""
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, (str, int)) and not isinstance(item, bool):
                rendered = str(item).strip()
                if rendered:
                    parts.append(rendered)
            else:
                return None
        return "；".join(parts) if parts else None
    return None


def _answer(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    return _text(value)


def _positive_source_id(doi: str, source_sha256: str) -> int:
    value = int.from_bytes(hashlib.sha256(
        (doi + "\n" + source_sha256).encode("utf-8")).digest()[:8], "big")
    return (value & ((1 << 63) - 1)) or 1


def build_scidb_csq_course(
        source: str | Path,
        output: str | Path,
        *,
        doi: str,
        dataset_url: str,
        source_md5: str,
        source_sha256: str | None = None,
        train_last_id: int = DEFAULT_TRAIN_LAST_ID,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """Build a quality-filtered CSQ course while preserving benchmark holdout."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if require_k_drive and (source_path.drive.upper() != "K:"
                            or output_path.drive.upper() != "K:"):
        raise ValueError("CSQ source/output 必须位于 K 盘")
    if not source_path.is_file():
        raise ValueError(f"CSQ source 不存在: {source_path}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not isinstance(doi, str) or not doi.strip():
        raise ValueError("doi 不得为空")
    if not isinstance(dataset_url, str) or not dataset_url.startswith("https://"):
        raise ValueError("dataset_url 必须是 HTTPS")
    if type(train_last_id) is not int or train_last_id <= 0:
        raise ValueError("train_last_id 必须是正整数")
    if (not isinstance(source_md5, str) or len(source_md5) != 32
            or any(char not in "0123456789abcdef" for char in source_md5.lower())):
        raise ValueError("source_md5 非法")
    payload = source_path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("CSQ source 不得含 UTF-8 BOM")
    actual_md5 = _digest(payload, "md5")
    if actual_md5 != source_md5.lower():
        raise ValueError(
            f"CSQ source MD5 不匹配: expected={source_md5.lower()}, "
            f"actual={actual_md5}")
    actual_sha256 = _digest(payload, "sha256")
    if source_sha256 is not None and actual_sha256 != source_sha256.lower():
        raise ValueError(
            f"CSQ source SHA-256 不匹配: expected={source_sha256.lower()}, "
            f"actual={actual_sha256}")
    try:
        raw_data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CSQ source 不是规范 UTF-8 JSON") from error
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError("CSQ root 必须是非空对象")

    keyed: list[tuple[int, dict[str, object]]] = []
    for raw_id, row in raw_data.items():
        if (not isinstance(raw_id, str) or not raw_id.isascii()
                or not raw_id.isdigit() or int(raw_id) <= 0
                or not isinstance(row, dict)):
            raise ValueError("CSQ record id 或对象非法")
        keyed.append((int(raw_id), row))
    keyed.sort(key=lambda item: item[0])
    if len({identity for identity, _row in keyed}) != len(keyed):
        raise ValueError("CSQ record id 重复")

    source_id = _positive_source_id(doi, actual_sha256)
    records = []
    counts = {"train": 0, "heldout": 0}
    exclusions: dict[str, int] = {}
    themes: dict[str, int] = {}
    for record_id, row in keyed:
        task = _text(row.get("任务"))
        theme = _text(row.get("主题"))
        normalized_theme = None if theme is None else _THEME_MAP.get(theme)
        question = _text(row.get("题目"))
        options = _surface(row.get("选项"))
        answer = _answer(row.get("答案"))
        hint = _text(row.get("提示"))
        knowledge = _text(row.get("知识点"))
        reasoning = _text(row.get("解题思路"))
        grade = _text(row.get("年级"))
        category = _text(row.get("类别"))
        skills = _surface(row.get("技能"))
        reason = None
        if task not in _TASKS:
            reason = "TASK_INVALID"
        elif normalized_theme is None:
            reason = "THEME_INVALID"
        elif any(value is None for value in (
                question, options, answer, hint, knowledge, reasoning,
                grade, category, skills)):
            reason = "REQUIRED_FIELD_MISSING"
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        assert all(value is not None for value in (
            question, options, answer, hint, knowledge, reasoning,
            grade, category, skills, normalized_theme,
        ))
        split = "train" if record_id <= train_last_id else "heldout"
        raw_record = _canonical(row)
        identity = f"{doi}:{record_id}:{_digest(raw_record, 'sha256')}"
        context = (
            f"学段：{grade}\n领域：{normalized_theme}\n主题：{category}\n"
            f"知识点：{knowledge}\n提示：{hint}\n科学技能：{skills}"
        )
        response = f"解题过程：{reasoning}\n答案：{answer}"
        records.append({
            "answer_surface": response,
            "context_surface": context,
            "course_version": 1,
            "family": "scidb-csq-v1",
            "format": FORMAT,
            "license_id": LICENSE_ID,
            "question_surface": f"{question}\n选项：{options}",
            "sample_id": identity,
            "sample_kind": "POSITIVE",
            "sample_role": "support",
            "source_dataset_doi": doi,
            "source_dataset_url": dataset_url,
            "source_id": source_id,
            "source_kind": SOURCE_KIND,
            "source_record_id": record_id,
            "source_ref_key": [
                SOURCE_KIND, source_id, record_id,
                0, 0, 0, 1,
                0, 0, 0, 0,
            ],
            "source_sha256": actual_sha256,
            "split": split,
        })
        counts[split] += 1
        themes[normalized_theme] = themes.get(normalized_theme, 0) + 1
    if not records or counts["train"] == 0 or counts["heldout"] == 0:
        raise ValueError("CSQ 过滤后缺少 train 或 heldout")

    course_payload = b"".join(_canonical(record) for record in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(course_payload)
    manifest = {
        "course": {
            "excluded_counts": [[key, exclusions[key]]
                                for key in sorted(exclusions)],
            "path": output_path.name,
            "record_count": len(records),
            "sha256": _digest(course_payload, "sha256"),
            "source_record_count": len(keyed),
            "split_counts": [[key, counts[key]]
                             for key in ("train", "heldout")],
            "theme_counts": [[key, themes[key]] for key in sorted(themes)],
            "train_last_id": train_last_id,
        },
        "format": FORMAT,
        "schema_version": 1,
        "source": {
            "dataset_url": dataset_url,
            "doi": doi,
            "license_id": LICENSE_ID,
            "path": source_path.name,
            "source_id": source_id,
            "source_md5": actual_md5,
            "source_sha256": actual_sha256,
        },
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest_payload = _canonical(manifest)
    manifest_path.write_bytes(manifest_payload)
    return {
        "course_path": output_path.as_posix(),
        "course_sha256": manifest["course"]["sha256"],
        "excluded_counts": manifest["course"]["excluded_counts"],
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _digest(manifest_payload, "sha256"),
        "record_count": len(records),
        "source_md5": actual_md5,
        "source_sha256": actual_sha256,
        "source_record_count": len(keyed),
        "split_counts": manifest["course"]["split_counts"],
        "theme_counts": manifest["course"]["theme_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert SciDB CSQ JSON to course")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--source-md5", required=True)
    parser.add_argument("--source-sha256", default=None)
    parser.add_argument("--train-last-id", type=int, default=DEFAULT_TRAIN_LAST_ID)
    args = parser.parse_args(argv)
    result = build_scidb_csq_course(
        args.source, args.output,
        doi=args.doi, dataset_url=args.dataset_url,
        source_md5=args.source_md5, source_sha256=args.source_sha256,
        train_last_id=args.train_last_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_scidb_csq_course", "main"]
