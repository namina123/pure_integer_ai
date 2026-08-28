"""把隔离 LLM 撰写的中文多轮来源转换为确定性公开对话课程。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_FORMAT = "PURE_INTEGER_AI_LLM_DIALOGUE_SOURCE_V1"
SOURCE_MANIFEST_FORMAT = "PURE_INTEGER_AI_LLM_DIALOGUE_SOURCE_MANIFEST_V1"
COURSE_FORMAT = "PURE_INTEGER_AI_LLM_ASSISTED_DIALOGUE_COURSE_V1"
LICENSE_ID = "CC0-1.0"
GENERATOR_MODEL = "gpt-5.6-sol"
SOURCE_KIND = 214
SPEAKER_USER = 1
SPEAKER_ASSISTANT = 2
MIN_EPISODES = 35
MIN_ASSISTANT_TURNS = 140
LONG_MEMORY_MIN_EPISODES = 32
LONG_MEMORY_MIN_ASSISTANT_TURNS = 160
DEFAULT_HELDOUT_PERCENT = 20
_SOURCE_FILES = frozenset({
    "GENERATION_CONTRACT.md", "TASK.md", "dataset.jsonl",
    "generation_report.json", "source_manifest.json", "DONE",
})
_SOURCE_FIELDS = frozenset({
    "format", "source_id", "episode_id", "family", "language",
    "generator_model", "human_generated", "turns", "quality_tags",
    "contains_external_fact",
})
_TURN_FIELDS = frozenset({"turn_ordinal", "speaker_role", "surface"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(item in "0123456789abcdef" for item in value))


def _stable_positive_int(label: str) -> int:
    value = int.from_bytes(
        hashlib.sha256(label.encode("utf-8")).digest()[:8], "big",
    ) & ((1 << 63) - 1)
    return value or 1


def _heldout(source_id: str, episode_id: str, percent: int) -> bool:
    payload = (
        b"PURE-INTEGER-AI/LLM-DIALOGUE/COURSE/V1\0"
        + source_id.encode("utf-8") + b"\0" + episode_id.encode("ascii"))
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 100 < percent


def _manifest_file_inventory(value: dict[str, object]) -> dict[str, dict[str, object]]:
    """读取冻结的 name/bytes/SHA inventory；不接受未绑定的路径文本。"""
    raw = value.get("files")
    if not isinstance(raw, list):
        raise ValueError("source manifest files 必须是 record 列表")
    inventory: dict[str, dict[str, object]] = {}
    for item in raw:
        if (not isinstance(item, dict)
                or set(item) != {"name", "bytes", "sha256"}
                or not isinstance(item.get("name"), str)
                or type(item.get("bytes")) is not int
                or int(item["bytes"]) < 0
                or not _valid_sha256(item.get("sha256"))):
            raise ValueError("source manifest file record 非法")
        name = str(item["name"])
        if name in inventory or name not in _SOURCE_FILES - {"DONE", "source_manifest.json"}:
            raise ValueError("source manifest file inventory 非法")
        inventory[name] = item
    expected = _SOURCE_FILES - {"DONE", "source_manifest.json"}
    if set(inventory) != expected:
        raise ValueError("source manifest file inventory 不闭合")
    return inventory


def _read_json(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是 UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON object")
    return value, payload


def _validate_source_root(
        root: Path, expected_manifest_sha256: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...], dict[str, object]]:
    """验证一个 manifest-last 来源并返回来源身份、episode 和统计。"""
    if set(item.name for item in root.iterdir()) != _SOURCE_FILES:
        raise ValueError(f"LLM dialogue source 文件集合不闭合: {root.name}")
    manifest_path = root / "source_manifest.json"
    manifest, manifest_payload = _read_json(
        manifest_path, label="source manifest")
    manifest_sha = _sha256(manifest_payload)
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("source manifest SHA-256 漂移")
    done = (root / "DONE").read_bytes()
    if done != (manifest_sha + "\n").encode("ascii"):
        raise ValueError("source DONE 不是 manifest-last 承诺")
    if (manifest.get("format") != SOURCE_MANIFEST_FORMAT
            or manifest.get("schema_version") != 1
            or manifest.get("generator_model") != GENERATOR_MODEL
            or manifest.get("human_generated") != 0
            or manifest.get("creation_method")
            != "isolated_codex_authored_dialogue"
            or manifest.get("license_id") != LICENSE_ID
            or not isinstance(manifest.get("source_id"), str)
            or not manifest["source_id"]
            or not isinstance(manifest.get("family"), str)
            or not manifest["family"]):
        raise ValueError("source manifest 身份漂移")
    inventory = _manifest_file_inventory(manifest)
    for name, item in inventory.items():
        path = root / name
        payload = path.read_bytes()
        if len(payload) != item["bytes"] or _sha256(payload) != item["sha256"]:
            raise ValueError(f"source payload 漂移: {name}")

    report, _ = _read_json(root / "generation_report.json", label="generation report")
    dataset_payload = (root / "dataset.jsonl").read_bytes()
    try:
        raw_lines = dataset_payload.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as error:
        raise ValueError("dataset 不是 UTF-8") from error
    if not raw_lines or any(not line.endswith("\n") or not line.strip()
                            for line in raw_lines):
        raise ValueError("dataset 含空行或缺少末尾换行")

    records = []
    episode_ids: set[str] = set()
    assistant_surfaces: set[str] = set()
    assistant_count = 0
    external_fact_count = 0
    source_id = str(manifest["source_id"])
    family = str(manifest["family"])
    for line_number, raw in enumerate(raw_lines, start=1):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset line {line_number} 非法") from error
        if not isinstance(record, dict) or set(record) != _SOURCE_FIELDS:
            raise ValueError(f"dataset line {line_number} schema 漂移")
        episode_id = record.get("episode_id")
        tags = record.get("quality_tags")
        turns = record.get("turns")
        if (record.get("format") != SOURCE_FORMAT
                or record.get("source_id") != source_id
                or record.get("family") != family
                or record.get("language") != "zh-CN"
                or record.get("generator_model") != GENERATOR_MODEL
                or record.get("human_generated") != 0
                or record.get("contains_external_fact") not in {0, 1}
                or not isinstance(episode_id, str)
                or episode_id in episode_ids
                or not isinstance(tags, list) or not tags
                or tags != sorted(set(tags))
                or any(not isinstance(tag, str) or not tag or not tag.isascii()
                       for tag in tags)
                or not isinstance(turns, list)):
            raise ValueError(f"dataset line {line_number} 身份漂移")
        # 来源合同冻结为 <family>-NNNN；family 本身可以含下划线。
        prefix = family + "-"
        suffix = episode_id.removeprefix(prefix)
        if (not episode_id.startswith(prefix) or len(suffix) != 4
                or not suffix.isascii() or not suffix.isdigit()
                or int(suffix) != line_number):
            raise ValueError("episode identity 不连续")
        minimum_turns = 10 if family == "long_memory" else 6
        if len(turns) < minimum_turns or len(turns) % 2:
            raise ValueError(f"dataset line {line_number} turn 数量不足")
        for ordinal, turn in enumerate(turns, start=1):
            expected_role = SPEAKER_USER if ordinal % 2 else SPEAKER_ASSISTANT
            if (not isinstance(turn, dict) or set(turn) != _TURN_FIELDS
                    or turn.get("turn_ordinal") != ordinal
                    or turn.get("speaker_role") != expected_role
                    or not isinstance(turn.get("surface"), str)
                    or len(turn["surface"].strip()) < 2
                    or turn["surface"] != turn["surface"].strip()):
                raise ValueError(
                    f"dataset line {line_number} turn {ordinal} 非法")
            if expected_role == SPEAKER_ASSISTANT:
                surface = str(turn["surface"])
                if surface in assistant_surfaces:
                    raise ValueError("assistant surface 出现完全重复")
                assistant_surfaces.add(surface)
                assistant_count += 1
        episode_ids.add(episode_id)
        external_fact_count += int(record["contains_external_fact"])
        records.append(record)

    minimum_episodes = (LONG_MEMORY_MIN_EPISODES
                        if family == "long_memory" else MIN_EPISODES)
    minimum_assistant = (LONG_MEMORY_MIN_ASSISTANT_TURNS
                         if family == "long_memory" else MIN_ASSISTANT_TURNS)
    if len(records) < minimum_episodes or assistant_count < minimum_assistant:
        raise ValueError("source 未达到 episode/assistant turn 下限")
    if (report.get("family") != family
            or report.get("episode_count") != len(records)
            or report.get("assistant_turn_count") != assistant_count
            or report.get("external_fact_episode_count") != external_fact_count
            or report.get("duplicate_assistant_surface_count") != 0
            or report.get("validation_status") != "PASS"):
        raise ValueError("generation report 与 dataset 不一致")
    identity = {
        "dataset_bytes": len(dataset_payload),
        "dataset_sha256": _sha256(dataset_payload),
        "family": family,
        "manifest_sha256": manifest_sha,
        "source_id": source_id,
    }
    stats = {
        "assistant_turn_count": assistant_count,
        "episode_count": len(records),
        "excluded_external_fact_episode_count": external_fact_count,
    }
    return identity, tuple(records), stats


def build_llm_assisted_dialogue_course(
        source_roots: tuple[str | Path, ...], output: str | Path, *,
        expected_source_manifest_sha256s: tuple[str, ...],
        heldout_percent: int = DEFAULT_HELDOUT_PERCENT,
        require_k_drive: bool = True,
        ) -> dict[str, object]:
    """验证多个隔离分片，并按 episode 投影所有无事实 assistant turn。"""
    roots = tuple(Path(value).resolve() for value in source_roots)
    output_path = Path(output).resolve()
    if (not roots or len(roots) != len(expected_source_manifest_sha256s)
            or len(set(roots)) != len(roots)
            or len(set(expected_source_manifest_sha256s))
            != len(expected_source_manifest_sha256s)):
        raise ValueError("source root/SHA 集合非法")
    if (type(heldout_percent) is not int
            or not 1 <= heldout_percent <= 50):
        raise ValueError("heldout_percent 必须是 1..50 的整数")
    if any(not _valid_sha256(value)
           for value in expected_source_manifest_sha256s):
        raise ValueError("expected source manifest SHA-256 非法")
    if require_k_drive and (output_path.drive.upper() != "K:"
                            or any(root.drive.upper() != "K:" for root in roots)):
        raise ValueError("LLM dialogue source/output 必须位于 K 盘")
    if any(not root.is_dir() for root in roots):
        raise ValueError("LLM dialogue source root 不存在")
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(output_path)

    validated = []
    source_ids: set[str] = set()
    families: set[str] = set()
    for root, expected_sha in sorted(
            zip(roots, expected_source_manifest_sha256s),
            key=lambda item: item[1]):
        identity, records, stats = _validate_source_root(root, expected_sha)
        source_id = str(identity["source_id"])
        family = str(identity["family"])
        if source_id in source_ids or family in families:
            raise ValueError("source_id/family 重复")
        source_ids.add(source_id)
        families.add(family)
        validated.append((identity, records, stats))

    course_records = []
    source_record_ids: set[int] = set()
    split_episode_counts = {"train": 0, "heldout": 0}
    excluded_external_facts = 0
    for identity, episodes, stats in validated:
        source_id_text = str(identity["source_id"])
        source_sha = str(identity["dataset_sha256"])
        family = str(identity["family"])
        source_id = _stable_positive_int(source_id_text + "\n" + source_sha)
        excluded_external_facts += int(
            stats["excluded_external_fact_episode_count"])
        for episode in episodes:
            if episode["contains_external_fact"] == 1:
                continue
            episode_id = str(episode["episode_id"])
            split = ("heldout" if _heldout(
                source_id_text, episode_id, heldout_percent) else "train")
            split_episode_counts[split] += 1
            turns = episode["turns"]
            for response_index in range(1, len(turns), 2):
                path = turns[:response_index + 1]
                response_ordinal = response_index + 1
                sample_id = f"{source_id_text}/{episode_id}/{response_ordinal}"
                source_record_id = _stable_positive_int(sample_id)
                if source_record_id in source_record_ids:
                    raise ValueError("source_record_id collision")
                source_record_ids.add(source_record_id)
                projected_turns = [{
                    "message_id": f"{episode_id}/{turn['turn_ordinal']}",
                    "speaker_role": turn["speaker_role"],
                    "surface": turn["surface"],
                    "turn_ordinal": turn["turn_ordinal"],
                } for turn in path]
                response = str(projected_turns[-1]["surface"])
                course_records.append({
                    "contains_external_fact": 0,
                    "context_turn_count": len(projected_turns) - 1,
                    "course_version": 1,
                    "dialogue_turns": projected_turns,
                    "family": "llm-dialogue-" + family + "-v1",
                    "format": COURSE_FORMAT,
                    "generator_model": GENERATOR_MODEL,
                    "human_generated": 0,
                    "input_surface": "\n".join(
                        str(turn["surface"]) for turn in projected_turns[:-1]),
                    "intent_support": 1,
                    "license_id": LICENSE_ID,
                    "path_message_ids": [
                        str(turn["message_id"]) for turn in projected_turns],
                    "path_turn_count": len(projected_turns),
                    "prompt_turn_ordinal": len(projected_turns) - 1,
                    "quality_tags": episode["quality_tags"],
                    "response_surface": response,
                    "response_turn_ordinal": len(projected_turns),
                    "sample_id": sample_id,
                    "sample_kind": "POSITIVE",
                    "sample_role": "support",
                    "source_id": source_id,
                    "source_kind": SOURCE_KIND,
                    "source_record_id": source_record_id,
                    "source_ref_key": [
                        SOURCE_KIND, source_id, source_record_id,
                        0, 0, 0, 1, 0, 0, 0, 0,
                    ],
                    "source_sha256": source_sha,
                    "source_title": "CC0 dialogue " + family,
                    "source_url": "urn:pure-integer-ai:" + source_id_text,
                    "split": split,
                })
    if not course_records or not split_episode_counts["train"] \
            or not split_episode_counts["heldout"]:
        raise ValueError("课程 train/heldout 为空")
    course_records.sort(key=lambda item: str(item["sample_id"]))
    course_payload = b"".join(_canonical(item) for item in course_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(course_payload)
    split_counts = {
        name: sum(record["split"] == name for record in course_records)
        for name in ("train", "heldout")}
    manifest = {
        "course": {
            "excluded_external_fact_episode_count": excluded_external_facts,
            "heldout_percent": heldout_percent,
            "path": output_path.name,
            "record_count": len(course_records),
            "sha256": _sha256(course_payload),
            "source_kind": SOURCE_KIND,
            "split_counts": [[name, split_counts[name]]
                             for name in ("train", "heldout")],
            "split_episode_counts": [[name, split_episode_counts[name]]
                                     for name in ("train", "heldout")],
        },
        "format": COURSE_FORMAT,
        "schema_version": 1,
        "source": {
            "creation_method": "isolated_codex_authored_dialogue",
            "generator_model": GENERATOR_MODEL,
            "human_generated": 0,
            "license_id": LICENSE_ID,
            "source_manifests": [identity for identity, _, _ in validated],
        },
    }
    manifest_payload = _canonical(manifest)
    manifest_path.write_bytes(manifest_payload)
    return {
        "course_path": output_path.as_posix(),
        "course_sha256": _sha256(course_payload),
        "excluded_external_fact_episode_count": excluded_external_facts,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256(manifest_payload),
        "record_count": len(course_records),
        "source_count": len(validated),
        "split_counts": manifest["course"]["split_counts"],
        "split_episode_counts": manifest["course"]["split_episode_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="convert isolated LLM dialogue shards to a CC0 course")
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-percent", type=int,
                        default=DEFAULT_HELDOUT_PERCENT)
    args = parser.parse_args(argv)
    result = build_llm_assisted_dialogue_course(
        tuple(args.source_root), args.output,
        expected_source_manifest_sha256s=tuple(
            args.expected_source_manifest_sha256),
        heldout_percent=args.heldout_percent)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COURSE_FORMAT", "GENERATOR_MODEL", "LICENSE_ID", "SOURCE_KIND",
    "build_llm_assisted_dialogue_course", "main",
]
