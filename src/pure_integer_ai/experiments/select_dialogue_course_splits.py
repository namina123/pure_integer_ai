"""Select bounded dialogue-course splits without prefix bias.

Large public courses are often grouped by upstream split or domain.  Taking a
file prefix can therefore consume mostly held-out records and leave no useful
training body.  This module scans once and keeps the lowest deterministic
content ranks for each requested split.  Memory use is bounded by the selected
records; source lines and language surfaces are not interpreted or rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path


SELECTION_FORMAT_V1 = "PURE_INTEGER_AI_DIALOGUE_SPLIT_SELECTION_V1"
_RANK_DOMAIN = b"PURE-INTEGER-AI/DIALOGUE-SPLIT-SELECTION/V1\0"
_SPLITS = ("train", "heldout", "negative")


class DialogueCourseSelectionError(ValueError):
    """The source course or deterministic split selection is invalid."""


def _require_k_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).resolve()
    if path.drive.upper() != "K:":
        raise DialogueCourseSelectionError(f"{label} 必须位于 K 盘")
    return path


def _canonical_line(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _normalized_split(record: dict[str, object]) -> str | None:
    split = record.get("split", "train")
    if split == "held_out":
        split = "heldout"
    elif split == "course":
        split = "train"
    if record.get("sample_role") in {"refute", "negative"}:
        split = "negative"
    sample_kind = record.get("sample_kind")
    if sample_kind == "NEGATIVE":
        split = "negative"
    elif sample_kind == "AMBIGUOUS" and split == "train":
        split = "heldout"
    return split if split in _SPLITS else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_dialogue_course_splits(
        source: str | Path,
        output: str | Path,
        *,
        train_limit: int,
        heldout_limit: int,
        negative_limit: int = 0,
        ) -> dict[str, object]:
    """Write a deterministic, content-ranked subset of a public JSONL course."""
    source_path = _require_k_path(source, label="source course")
    output_path = _require_k_path(output, label="selected course")
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    if not source_path.is_file():
        raise DialogueCourseSelectionError("source course 不存在")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path)
    limits = {
        "train": train_limit,
        "heldout": heldout_limit,
        "negative": negative_limit,
    }
    if (type(train_limit) is not int or train_limit <= 0
            or any(type(value) is not int or value < 0
                   for value in (heldout_limit, negative_limit))):
        raise DialogueCourseSelectionError("split limit 非法")

    # Each heap root is the worst retained rank.  Negative rank/line values
    # let stdlib's min-heap behave as a bounded max-heap.
    retained: dict[str, list[tuple[int, int, bytes]]] = {
        split: [] for split in _SPLITS
    }
    seen = {split: 0 for split in _SPLITS}
    source_digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            source_digest.update(raw)
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DialogueCourseSelectionError(
                    f"source course JSONL 非法: line {line_number}") from error
            if not isinstance(value, dict):
                raise DialogueCourseSelectionError(
                    f"source course 记录不是对象: line {line_number}")
            split = _normalized_split(value)
            if split is None:
                continue
            seen[split] += 1
            limit = limits[split]
            if limit == 0:
                continue
            canonical = _canonical_line(value)
            rank = int.from_bytes(hashlib.sha256(
                _RANK_DOMAIN + canonical).digest(), "big")
            entry = (-rank, -line_number, canonical)
            heap = retained[split]
            if len(heap) < limit:
                heapq.heappush(heap, entry)
                continue
            worst_rank = -heap[0][0]
            worst_line = -heap[0][1]
            if (rank, line_number) < (worst_rank, worst_line):
                heapq.heapreplace(heap, entry)

    selected_counts = {split: len(retained[split]) for split in _SPLITS}
    shortages = {
        split: limits[split] - selected_counts[split]
        for split in _SPLITS if selected_counts[split] != limits[split]
    }
    if shortages:
        raise DialogueCourseSelectionError(
            f"source course split 数量不足: {shortages}; seen={seen}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    with output_path.open("xb") as stream:
        for split in _SPLITS:
            ordered = sorted(
                retained[split], key=lambda item: (-item[0], -item[1]))
            for _negative_rank, _negative_line, payload in ordered:
                stream.write(payload)
                output_digest.update(payload)

    manifest: dict[str, object] = {
        "format": SELECTION_FORMAT_V1,
        "schema_version": 1,
        "selection": {
            "algorithm": "lowest-content-sha256-rank",
            "rank_domain_sha256": hashlib.sha256(_RANK_DOMAIN).hexdigest(),
            "limits": [[split, limits[split]] for split in _SPLITS],
            "selected_counts": [
                [split, selected_counts[split]] for split in _SPLITS],
            "seen_counts": [[split, seen[split]] for split in _SPLITS],
        },
        "source": {
            "bytes": source_path.stat().st_size,
            "name": source_path.name,
            "sha256": source_digest.hexdigest(),
        },
        "output": {
            "bytes": output_path.stat().st_size,
            "name": output_path.name,
            "sha256": output_digest.hexdigest(),
        },
    }
    manifest_path.write_bytes(_canonical_line(manifest))
    return {
        "course_path": output_path.as_posix(),
        "course_sha256": output_digest.hexdigest(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "selected_counts": manifest["selection"]["selected_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="select deterministic dialogue-course split quotas")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--heldout-limit", type=int, default=0)
    parser.add_argument("--negative-limit", type=int, default=0)
    args = parser.parse_args(argv)
    result = select_dialogue_course_splits(
        args.source,
        args.output,
        train_limit=args.train_limit,
        heldout_limit=args.heldout_limit,
        negative_limit=args.negative_limit,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DialogueCourseSelectionError",
    "SELECTION_FORMAT_V1",
    "select_dialogue_course_splits",
]
