"""Convert the Apache-2.0 KdConv snapshot into a deterministic dialogue course."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT = "PURE_INTEGER_AI_KDCONV_DIALOGUE_COURSE_V1"
LICENSE_ID = "Apache-2.0"
SOURCE_KIND = 206
SPEAKER_USER = 1
SPEAKER_ASSISTANT = 2
MAX_HISTORY_TURNS = 8
DATASET_URL = "https://github.com/thu-coai/KdConv"
_DOMAINS = ("film", "music", "travel")
_SPLITS = ("train", "dev", "test")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _stable_positive_int(label: str) -> int:
    value = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "big",
    ) & ((1 << 63) - 1)
    return value or 1


def _strict_surface(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value.strip()) < 1:
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _relative_role(response_ordinal: int, turn_ordinal: int) -> int:
    return (SPEAKER_ASSISTANT
            if (response_ordinal - turn_ordinal) % 2 == 0
            else SPEAKER_USER)


def _dialogue_turns(messages: list[dict[str, object]],
                    response_ordinal: int, *,
                    identity_prefix: str) -> list[dict[str, object]]:
    first = max(0, response_ordinal - MAX_HISTORY_TURNS - 1)
    if first % 2 == response_ordinal % 2:
        first += 1
    result = []
    for source_ordinal in range(first, response_ordinal + 1):
        result.append({
            "message_id": f"{identity_prefix}/{source_ordinal}",
            "speaker_role": _relative_role(response_ordinal, source_ordinal),
            "surface": _strict_surface(
                messages[source_ordinal].get("message"),
                label="KdConv message"),
            "turn_ordinal": source_ordinal - first + 1,
        })
    return result


def build_kdconv_dialogue_course(
        source_root: str | Path, output: str | Path, *,
        expected_commit: str,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """Build adjacent-turn samples from the frozen official train/dev/test files."""
    root = Path(source_root).resolve()
    output_path = Path(output).resolve()
    if require_k_drive and (root.drive.upper() != "K:"
                            or output_path.drive.upper() != "K:"):
        raise ValueError("KdConv source/output must be on K drive")
    if not root.is_dir() or not (root / "data").is_dir():
        raise ValueError("KdConv source root is incomplete")
    if output_path.exists() or output_path.with_name(
            output_path.name + ".manifest.json").exists():
        raise FileExistsError(output_path)
    if (not isinstance(expected_commit, str) or len(expected_commit) != 40
            or any(item not in "0123456789abcdef" for item in expected_commit)):
        raise ValueError("expected_commit must be a lowercase Git object id")
    head_path = root / ".git" / "refs" / "heads" / "master"
    if not head_path.is_file():
        head_path = root / ".git" / "refs" / "heads" / "main"
    if (not head_path.is_file()
            or head_path.read_text("ascii").strip() != expected_commit):
        raise ValueError("KdConv checked-out commit drifted")
    license_payload = (root / "LICENSE").read_bytes()
    if b"Apache License" not in license_payload or b"Version 2.0" not in license_payload:
        raise ValueError("KdConv Apache-2.0 license evidence is missing")

    records: list[dict[str, object]] = []
    source_inventory = []
    source_record_ids: set[int] = set()
    conversation_count = 0
    grounded_response_count = 0
    intent_support_count = 0
    for domain in _DOMAINS:
        for source_split in _SPLITS:
            relative = Path("data") / domain / f"{source_split}.json"
            path = root / relative
            payload = path.read_bytes()
            source_sha = _sha256(payload)
            source_inventory.append({
                "bytes": len(payload),
                "path": relative.as_posix(),
                "sha256": source_sha,
            })
            try:
                conversations = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid KdConv file: {relative}") from error
            if not isinstance(conversations, list):
                raise ValueError(f"KdConv split is not a list: {relative}")
            source_url = (
                f"{DATASET_URL}/blob/{expected_commit}/{relative.as_posix()}")
            source_id = _stable_positive_int(source_url + "\n" + source_sha)
            split = "train" if source_split == "train" else "heldout"
            for conversation_ordinal, conversation in enumerate(conversations):
                if (not isinstance(conversation, dict)
                        or not isinstance(conversation.get("messages"), list)
                        or len(conversation["messages"]) < 2):
                    raise ValueError("KdConv conversation is malformed")
                messages = conversation["messages"]
                if any(not isinstance(item, dict) for item in messages):
                    raise ValueError("KdConv message is malformed")
                conversation_count += 1
                for response_ordinal in range(1, len(messages)):
                    response_message = messages[response_ordinal]
                    attrs = response_message.get("attrs", [])
                    if attrs is None:
                        attrs = []
                    if not isinstance(attrs, list):
                        raise ValueError("KdConv attrs is malformed")
                    turns = _dialogue_turns(
                        messages, response_ordinal,
                        identity_prefix=(
                            f"kdconv/{domain}/{source_split}/"
                            f"{conversation_ordinal}"))
                    if (turns[-2]["speaker_role"] != SPEAKER_USER
                            or turns[-1]["speaker_role"] != SPEAKER_ASSISTANT):
                        raise AssertionError("KdConv role projection drifted")
                    sample_id = (
                        f"kdconv/{domain}/{source_split}/"
                        f"{conversation_ordinal}/{response_ordinal}")
                    record_id = _stable_positive_int(sample_id)
                    if record_id in source_record_ids:
                        raise ValueError("KdConv source_record_id collision")
                    source_record_ids.add(record_id)
                    intent_support = int(not attrs)
                    grounded_response_count += int(bool(attrs))
                    intent_support_count += intent_support
                    response = str(turns[-1]["surface"])
                    records.append({
                        "course_version": 1,
                        "format": FORMAT,
                        "family": f"kdconv-{domain}-human-v1",
                        "sample_id": sample_id,
                        "sample_kind": "POSITIVE",
                        "sample_role": "support",
                        "split": split,
                        "license_id": LICENSE_ID,
                        "source_kind": SOURCE_KIND,
                        "source_id": source_id,
                        "source_record_id": record_id,
                        "source_ref_key": [
                            SOURCE_KIND, source_id, record_id,
                            0, 0, 0, 1, 0, 0, 0, 0,
                        ],
                        "source_dataset_url": DATASET_URL,
                        "source_url": source_url,
                        "source_sha256": source_sha,
                        "source_title": f"KdConv {domain}",
                        "source_split": source_split,
                        "conversation_ordinal": conversation_ordinal,
                        "response_source_ordinal": response_ordinal,
                        "path_message_ids": [
                            str(turn["message_id"]) for turn in turns],
                        "path_turn_count": len(turns),
                        "context_turn_count": len(turns) - 1,
                        "prompt_turn_ordinal": len(turns) - 1,
                        "response_turn_ordinal": len(turns),
                        "dialogue_turns": turns,
                        "human_generated": 1,
                        "intent_support": intent_support,
                        "grounding_attr_count": len(attrs),
                        "input_surface": "\n".join(
                            str(turn["surface"]) for turn in turns[:-1]),
                        "response_surface": response,
                    })
    if not records or intent_support_count <= 0:
        raise ValueError("KdConv produced no usable dialogue records")
    records.sort(key=lambda item: str(item["sample_id"]))
    payload = b"".join(_canonical(item) for item in records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    split_counts = {
        name: sum(item["split"] == name for item in records)
        for name in ("train", "heldout")}
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "source": {
            "commit": expected_commit,
            "dataset_url": DATASET_URL,
            "license_file_sha256": _sha256(license_payload),
            "license_id": LICENSE_ID,
            "files": source_inventory,
        },
        "course": {
            "path": output_path.name,
            "sha256": _sha256(payload),
            "record_count": len(records),
            "conversation_count": conversation_count,
            "grounded_response_count": grounded_response_count,
            "intent_support_count": intent_support_count,
            "maximum_history_turns": MAX_HISTORY_TURNS,
            "split_counts": [
                [name, split_counts[name]] for name in ("train", "heldout")],
        },
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest_payload = _canonical(manifest)
    manifest_path.write_bytes(manifest_payload)
    return {
        "course_path": output_path.as_posix(),
        "course_sha256": _sha256(payload),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256(manifest_payload),
        "record_count": len(records),
        "conversation_count": conversation_count,
        "grounded_response_count": grounded_response_count,
        "intent_support_count": intent_support_count,
        "split_counts": manifest["course"]["split_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="convert KdConv to a deterministic human dialogue course")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_kdconv_dialogue_course(
        args.source_root, args.output, expected_commit=args.expected_commit),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FORMAT", "build_kdconv_dialogue_course", "main"]
