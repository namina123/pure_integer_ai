"""Convert the public OASST1 tree export to a deterministic dialogue course.

Only reviewed, human-authored Chinese paths are admitted.  JSONL/GZip stays
at the exchange boundary; the resulting course is consumed by the existing
integer training pipeline and retains an Apache-2.0 SourceRef per reply.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


FORMAT = "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2"
OPENASSISTANT_FORMAT = "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2"
LICENSE_ID = "Apache-2.0"
SOURCE_KIND = 204
SPEAKER_USER = 1
SPEAKER_ASSISTANT = 2
_RELEASE_IDENTITIES = {
    "oasst1": (FORMAT, "oasst1-human-zh-v1", "OpenAssistant OASST1"),
    "oasst2": (
        OPENASSISTANT_FORMAT,
        "oasst2-human-zh-v1",
        "OpenAssistant OASST2",
    ),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _stable_positive_int(label: str) -> int:
    value = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8],
                           "big") & ((1 << 63) - 1)
    return value or 1


def _bucket(tree_id: str) -> int:
    payload = b"PURE-INTEGER-AI/OASST1/COURSE/V1" + tree_id.encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 100


def _text(message: dict[str, object]) -> str | None:
    value = message.get("text")
    if not isinstance(value, str) or len(value.strip()) < 2:
        return None
    return value.strip()


def _message_admitted(message: dict[str, object]) -> bool:
    role = message.get("role")
    return (
        role in {"prompter", "assistant"}
        and message.get("lang") == "zh"
        and message.get("synthetic") is False
        and message.get("deleted") is False
        and message.get("review_result") is True
        and type(message.get("review_count")) is int
        and int(message["review_count"]) >= 1
        and _text(message) is not None
        and (role != "assistant" or message.get("rank") == 0)
    )


def _valid_path(path: tuple[dict[str, object], ...]) -> bool:
    if len(path) < 2 or path[0].get("role") != "prompter":
        return False
    if path[-1].get("role") != "assistant":
        return False
    for ordinal, message in enumerate(path):
        expected = "prompter" if ordinal % 2 == 0 else "assistant"
        if message.get("role") != expected or not _message_admitted(message):
            return False
        if ordinal:
            if message.get("parent_id") != path[ordinal - 1].get("message_id"):
                return False
    return True


def _input_surface(path: tuple[dict[str, object], ...]) -> str:
    return "\n".join(_text(message) for message in path[:-1])


def _dialogue_turns(
        path: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    """保留 turn/speaker 边界；整数 role 可由其他语言直接恢复。"""
    result = []
    for ordinal, message in enumerate(path, start=1):
        role = message["role"]
        result.append({
            "message_id": message["message_id"],
            "speaker_role": (
                SPEAKER_USER if role == "prompter" else SPEAKER_ASSISTANT),
            "surface": _text(message),
            "turn_ordinal": ordinal,
        })
    return result


def build_openassistant_dialogue_course(
        source: str | Path,
        output: str | Path,
        *,
        source_sha256: str,
        dataset_url: str,
        file_url: str,
        release_id: str,
        heldout_percent: int = 10,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """Build a Chinese human dialogue course from one registered ready export."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if require_k_drive and (source_path.drive.upper() != "K:"
                            or output_path.drive.upper() != "K:"):
        raise ValueError("OASST1 source/output must be on K drive")
    if not source_path.is_file():
        raise ValueError(f"OASST1 source does not exist: {source_path}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in source_sha256.lower())):
        raise ValueError("source_sha256 is invalid")
    if (not isinstance(dataset_url, str) or not dataset_url.startswith("https://")
            or not isinstance(file_url, str) or not file_url.startswith("https://")):
        raise ValueError("dataset_url and file_url must be HTTPS")
    if type(heldout_percent) is not int or not 1 <= heldout_percent <= 50:
        raise ValueError("heldout_percent must be an integer from 1 to 50")
    release_identity = _RELEASE_IDENTITIES.get(release_id)
    if release_identity is None:
        raise ValueError("release_id must be oasst1 or oasst2")
    course_format, family, source_title = release_identity

    source_payload = source_path.read_bytes()
    actual_sha256 = _sha256(source_payload)
    if actual_sha256 != source_sha256.lower():
        raise ValueError(
            f"OASST1 SHA-256 mismatch: expected={source_sha256.lower()}, "
            f"actual={actual_sha256}")

    try:
        stream = gzip.open(io.BytesIO(source_payload), "rt", encoding="utf-8")
    except OSError as error:
        raise ValueError("OASST1 source is not valid GZip") from error
    source_id = _stable_positive_int(dataset_url + "\n" + actual_sha256)
    records: list[dict[str, object]] = []
    message_ids: set[str] = set()
    source_record_ids: set[int] = set()
    tree_count = 0
    admitted_tree_ids: set[str] = set()
    multi_turn_count = 0
    try:
        with stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                try:
                    tree = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"OASST1 JSONL is invalid at line {line_number}") from error
                if not isinstance(tree, dict) or not isinstance(tree.get("prompt"), dict):
                    raise ValueError(f"OASST1 tree is invalid at line {line_number}")
                tree_count += 1
                root = tree["prompt"]
                tree_id = tree.get("message_tree_id") or root.get("message_id")
                if not isinstance(tree_id, str) or not tree_id:
                    raise ValueError(f"OASST1 tree identity is missing at line {line_number}")
                split = "heldout" if _bucket(tree_id) < heldout_percent else "train"
                stack: list[tuple[dict[str, object], tuple[dict[str, object], ...]]] = [
                    (root, ()),
                ]
                while stack:
                    message, ancestors = stack.pop()
                    path = (*ancestors, message)
                    if _valid_path(path):
                        message_id = message.get("message_id")
                        if not isinstance(message_id, str) or not message_id:
                            raise ValueError("OASST1 admitted reply has no message_id")
                        if message_id in message_ids:
                            raise ValueError(f"duplicate OASST1 message_id: {message_id}")
                        record_id = _stable_positive_int(message_id)
                        if record_id in source_record_ids:
                            raise ValueError("OASST1 source_record_id collision")
                        message_ids.add(message_id)
                        source_record_ids.add(record_id)
                        admitted_tree_ids.add(tree_id)
                        multi_turn_count += int(len(path) > 2)
                        records.append({
                            "course_version": 1,
                            "format": course_format,
                            "family": family,
                            "sample_id": message_id,
                            "sample_kind": "POSITIVE",
                            "sample_role": "support",
                            "split": split,
                            "license_id": LICENSE_ID,
                            "source_kind": SOURCE_KIND,
                            "source_id": source_id,
                            "source_record_id": record_id,
                            "source_ref_key": [
                                SOURCE_KIND, source_id, record_id,
                                0, 0, 0, 1,
                                0, 0, 0, 0,
                            ],
                            "source_dataset_url": dataset_url,
                            "source_url": file_url,
                            "source_sha256": actual_sha256,
                            "source_title": source_title,
                            "message_tree_id": tree_id,
                            "message_id": message_id,
                            "parent_id": message.get("parent_id"),
                            "path_message_ids": [item["message_id"] for item in path],
                            "path_turn_count": len(path),
                            "context_turn_count": len(path) - 1,
                            "prompt_turn_ordinal": len(path) - 1,
                            "response_turn_ordinal": len(path),
                            "dialogue_turns": _dialogue_turns(path),
                            "human_generated": 1,
                            "input_surface": _input_surface(path),
                            "response_surface": _text(message),
                        })
                    replies = message.get("replies")
                    if replies is None:
                        replies = []
                    if not isinstance(replies, list) or any(
                            not isinstance(item, dict) for item in replies):
                        raise ValueError("OASST1 replies must be a list of objects")
                    for child in reversed(replies):
                        stack.append((child, path))
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("OASST1 source cannot be decompressed as UTF-8") from error
    if not records:
        raise ValueError("OASST1 source has no admitted Chinese human dialogue")

    records.sort(key=lambda item: str(item["sample_id"]))
    counts = {
        "train": sum(item["split"] == "train" for item in records),
        "heldout": sum(item["split"] == "heldout" for item in records),
    }
    payload = b"".join(_canonical(item) for item in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    manifest = {
        "format": course_format,
        "schema_version": 1,
        "source": {
            "path": source_path.name,
            "dataset_url": dataset_url,
            "file_url": file_url,
            "license_id": LICENSE_ID,
            "source_sha256": actual_sha256,
        },
        "course": {
            "path": output_path.name,
            "sha256": _sha256(payload),
            "record_count": len(records),
            "split_counts": [[key, counts[key]] for key in ("train", "heldout")],
            "tree_count": tree_count,
            "admitted_tree_count": len(admitted_tree_ids),
            "multi_turn_record_count": multi_turn_count,
            "heldout_percent": heldout_percent,
            "source_kind": SOURCE_KIND,
            "source_id": source_id,
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
        "tree_count": tree_count,
        "admitted_tree_count": len(admitted_tree_ids),
        "multi_turn_record_count": multi_turn_count,
        "source_sha256": actual_sha256,
    }


def build_oasst1_dialogue_course(
        source: str | Path,
        output: str | Path,
        *,
        source_sha256: str,
        dataset_url: str,
        file_url: str,
        heldout_percent: int = 10,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """保持既有 OASST1 调用和逐字段课程身份。"""
    return build_openassistant_dialogue_course(
        source,
        output,
        source_sha256=source_sha256,
        dataset_url=dataset_url,
        file_url=file_url,
        release_id="oasst1",
        heldout_percent=heldout_percent,
        require_k_drive=require_k_drive,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="convert OASST1 Chinese human dialogue to public course")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--file-url", required=True)
    parser.add_argument(
        "--release-id", choices=tuple(sorted(_RELEASE_IDENTITIES)),
        default="oasst1")
    parser.add_argument("--heldout-percent", type=int, default=10)
    args = parser.parse_args(argv)
    result = build_openassistant_dialogue_course(
        args.source, args.output,
        source_sha256=args.source_sha256,
        dataset_url=args.dataset_url,
        file_url=args.file_url,
        release_id=args.release_id,
        heldout_percent=args.heldout_percent,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAT", "OPENASSISTANT_FORMAT",
    "build_oasst1_dialogue_course", "build_openassistant_dialogue_course",
    "main",
]
