"""构建并验证 OASST 人工对话的聚合回答片段 artifact。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    DialogueResponseTrainingRow,
    LearnedDialogueIntentModel,
    LearnedDialogueResponseError,
    LearnedDialogueResponseModel,
    LearnedDialogueResponseRuntime,
    PRODUCTION_MIN_SIMILARITY_PERMILLE,
    learn_dialogue_intent_model,
    learn_dialogue_response_model,
    provider_identity_markers,
)
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.storage.k_run_boundary import (
    create_new_run_root,
    open_existing_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)
from pure_integer_ai.experiments.sqlite_learned_dialogue_intent import (
    SQLITE_INTENT_FILE,
    build_sqlite_learned_dialogue_intent_index,
    SqliteLearnedDialogueIntentRuntime,
    validate_sqlite_learned_dialogue_intent_index,
)


ARTIFACT_KIND = "MULTISOURCE_LEARNED_DIALOGUE_RESPONSE_V6"
ARTIFACT_SCHEMA_VERSION = 6
TRAINING_ALGORITHM = "SQLITE_BOUNDED_SPARSE_INTENT_CONTEXT_LEARNING_V6"
MODEL_FILE = "learned_dialogue_response_model.int"
INTENT_MODEL_FILE = "learned_dialogue_intent_model.int"
HELDOUT_FILE = "learned_dialogue_response_heldout.json"
MANIFEST_FILE = "learned_dialogue_response_manifest.json"
MIN_HELDOUT_GENERATED_PERMILLE = 100
_LEGACY_MANIFEST_IDENTITIES = {
    ("OPENASSISTANT_LEARNED_DIALOGUE_RESPONSE_V2", 2,
     "AGGREGATE_FEATURE_FRAGMENT_LEARNING_V2_SHORT_EXACT"),
    ("OPENASSISTANT_LEARNED_DIALOGUE_RESPONSE_V3", 3,
     "AGGREGATE_FEATURE_FRAGMENT_INTENT_CONTEXT_LEARNING_V3"),
    ("MULTISOURCE_LEARNED_DIALOGUE_RESPONSE_V4", 4,
     "SPARSE_PROMPT_PROTOTYPE_INTENT_CONTEXT_LEARNING_V4"),
    ("MULTISOURCE_LEARNED_DIALOGUE_RESPONSE_V5", 5,
     "SQLITE_BOUNDED_SPARSE_INTENT_CONTEXT_LEARNING_V5"),
}
_LEGACY_NO_INTENT_IDENTITIES = {
    ("OPENASSISTANT_LEARNED_DIALOGUE_RESPONSE_V2", 2,
     "AGGREGATE_FEATURE_FRAGMENT_LEARNING_V2_SHORT_EXACT"),
}
_LEGACY_SQLITE_IDENTITIES = {
    ("MULTISOURCE_LEARNED_DIALOGUE_RESPONSE_V5", 5,
     "SQLITE_BOUNDED_SPARSE_INTENT_CONTEXT_LEARNING_V5"),
}
_COURSE_IDENTITIES = {
    "PURE_INTEGER_AI_OASST1_DIALOGUE_COURSE_V2": ("Apache-2.0", 1),
    "PURE_INTEGER_AI_OPENASSISTANT_DIALOGUE_COURSE_V2": ("Apache-2.0", 1),
    "PURE_INTEGER_AI_KDCONV_DIALOGUE_COURSE_V1": ("Apache-2.0", 1),
    "PURE_INTEGER_AI_LLM_ASSISTED_DIALOGUE_COURSE_V1": ("CC0-1.0", 0),
}


# object-model: exception; interop=learned-dialogue-response-artifact-v1
class LearnedDialogueResponseArtifactError(ValueError):
    """Artifact 来源、发布边界或封存身份不闭合。"""


# object-model: value; representation=struct; interop=learned-dialogue-response-artifact-v1
@dataclass(frozen=True, slots=True)
class LearnedDialogueResponseArtifact:
    """已验证模型及其公开能力摘要。"""

    model: LearnedDialogueResponseModel
    intent_model: LearnedDialogueIntentModel | None
    status: str
    capability_status: str
    model_sha256: str
    heldout_sha256: str
    course_sha256: str
    intent_index_path: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha_bytes(value: str, *, label: str) -> tuple[int, ...]:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise LearnedDialogueResponseArtifactError(f"{label} 不是规范 SHA-256")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise LearnedDialogueResponseArtifactError(
            f"{label} 不是规范 SHA-256") from error
    if len(raw) != 32:
        raise LearnedDialogueResponseArtifactError(f"{label} 不是规范 SHA-256")
    return tuple(raw)


def _course_rows(course: Path) -> tuple[
        tuple[DialogueResponseTrainingRow, ...], tuple[str, ...], tuple[str, ...]]:
    """流式读取正式中文人工课程，保留末轮边界及此前完整角色历史。"""
    rows: list[DialogueResponseTrainingRow] = []
    sample_ids: set[str] = set()
    source_shas: set[str] = set()
    license_ids: set[str] = set()
    try:
        with course.open("r", encoding="utf-8", newline="") as stream:
            for ordinal, line in enumerate(stream, 1):
                if not line.endswith("\n") or not line.strip():
                    raise LearnedDialogueResponseArtifactError(
                        f"course line {ordinal} 为空或未以换行结束")
                value = json.loads(line)
                identity = (_COURSE_IDENTITIES.get(value.get("format"))
                            if isinstance(value, dict) else None)
                if (not isinstance(value, dict)
                        or identity is None
                        or (value.get("license_id"), value.get("human_generated"))
                        != identity
                        or value.get("sample_kind") != "POSITIVE"
                        or value.get("sample_role") != "support"):
                    raise LearnedDialogueResponseArtifactError(
                        f"course line {ordinal} 来源身份漂移")
                split = value.get("split")
                response = value.get("response_surface")
                source_title = value.get("source_title")
                sample_id = value.get("sample_id")
                source_sha = value.get("source_sha256")
                turns = value.get("dialogue_turns")
                intent_support = value.get("intent_support", 1)
                if (split not in {"train", "heldout"}
                        or not isinstance(response, str) or not response.strip()
                        or not isinstance(source_title, str) or not source_title.strip()
                        or not isinstance(sample_id, str) or not sample_id
                        or not isinstance(source_sha, str)
                        or intent_support not in {0, 1}
                        or not isinstance(turns, list) or len(turns) < 2):
                    raise LearnedDialogueResponseArtifactError(
                        f"course line {ordinal} 对话绑定非法")
                user = turns[-2]
                assistant = turns[-1]
                if (not isinstance(user, dict) or not isinstance(assistant, dict)
                        or user.get("speaker_role") != 1
                        or assistant.get("speaker_role") != 2
                        or assistant.get("surface") != response
                        or not isinstance(user.get("surface"), str)
                        or not user["surface"].strip()):
                    raise LearnedDialogueResponseArtifactError(
                        f"course line {ordinal} 末轮 user/assistant 漂移")
                history = []
                for turn in turns[:-2]:
                    if (not isinstance(turn, dict)
                            or turn.get("speaker_role") not in {1, 2}
                            or not isinstance(turn.get("surface"), str)
                            or not turn["surface"].strip()):
                        raise LearnedDialogueResponseArtifactError(
                            f"course line {ordinal} 历史 turn 非法")
                    history.append((int(turn["speaker_role"]), turn["surface"]))
                if sample_id in sample_ids:
                    raise LearnedDialogueResponseArtifactError(
                        "course 包含重复 sample_id")
                sample_ids.add(sample_id)
                source_shas.add(source_sha)
                license_ids.add(str(value["license_id"]))
                rows.append(DialogueResponseTrainingRow(
                    split, user["surface"], response, source_title,
                    tuple(history), bool(intent_support)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnedDialogueResponseArtifactError("course 无法流式读取") from error
    if not rows or not source_shas:
        raise LearnedDialogueResponseArtifactError("course 或来源集合为空")
    for value in source_shas:
        _sha_bytes(value, label="source SHA")
    return (tuple(rows), tuple(sorted(source_shas)),
            tuple(sorted(license_ids)))


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = (len(ordered) * percentile + 99) // 100
    return ordered[rank - 1]


def _heldout_report(
        rows: tuple[DialogueResponseTrainingRow, ...],
        model: LearnedDialogueResponseModel,
        intent_model: LearnedDialogueIntentModel,
        *,
        intent_runtime: object | None = None,
        ) -> dict[str, object]:
    """运行冻结 heldout；只判断覆盖与身份泄漏，不冒充语义质量人工评测。

    发布 artifact 的 intent 消费者是 SQLite 整数倒排 runtime。构建阶段若只
    使用内存实现，会让 held-out 覆盖与真正公开运行路径不一致；传入
    ``intent_runtime`` 时强制复用同一生产消费者。
    """
    runtime = (LearnedDialogueResponseRuntime(
        model, intent_runtime=intent_runtime)
               if intent_runtime is not None else
               LearnedDialogueResponseRuntime(model, intent_model))
    heldout = tuple(item for item in rows if item.split == "heldout")
    results = tuple((
        item,
        runtime.respond(
            item.prompt, history=item.history,
            minimum_similarity_permille=(
                PRODUCTION_MIN_SIMILARITY_PERMILLE)),
    ) for item in heldout)
    generated = tuple((item, result) for item, result in results if result.used)
    similarities = [result.similarity_permille for _, result in generated]
    provider_leakage = 0
    exact_response = 0
    intent_generated = 0
    for item, result in generated:
        assert result.surface is not None
        folded = "".join(value.casefold() for value in result.surface
                         if value.isascii() and value.isalnum())
        provider_leakage += int(any(
            "".join(value.casefold() for value in marker
                    if value.isascii() and value.isalnum()) in folded
            for marker in provider_identity_markers(item.source_title)))
        exact_response += int(result.surface.strip() == item.response.strip())
        intent_generated += int(
            result.reason == "learned_intent_fragment_selected")
    coverage = (len(generated) * 1000) // len(heldout)
    capability = (
        "PASS" if coverage >= MIN_HELDOUT_GENERATED_PERMILLE
        and provider_leakage == 0 else "NE")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "capability_status": capability,
        "exact_source_response_count": exact_response,
        "generated_count": len(generated),
        "generated_coverage_permille": coverage,
        "intent_generated_count": intent_generated,
        "heldout_count": len(heldout),
        "minimum_generated_coverage_permille": (
            MIN_HELDOUT_GENERATED_PERMILLE),
        "provider_identity_leakage_count": provider_leakage,
        "production_minimum_similarity_permille": (
            PRODUCTION_MIN_SIMILARITY_PERMILLE),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "semantic_quality_status": "NE",
        "similarity_p50_permille": _nearest_rank(similarities, 50),
        "similarity_p95_permille": _nearest_rank(similarities, 95),
        "status": "PASS" if provider_leakage == 0 else "FAIL",
        "train_count": model.train_count,
        "excluded_provider_identity_count": (
            model.excluded_provider_identity_count),
    }


def build_learned_dialogue_response_artifact(
        *, course_path: str | Path | tuple[str | Path, ...],
        artifact_root: str | Path,
        expected_course_sha256: str | tuple[str, ...],
        require_k_drive: bool = True,
        ) -> LearnedDialogueResponseArtifact:
    """合并一至多份人工课程并在新 K 盘根发布 v3 模型。"""
    raw_courses = ((course_path,) if isinstance(course_path, (str, Path))
                   else tuple(course_path))
    expected_shas = ((expected_course_sha256,)
                     if isinstance(expected_course_sha256, str)
                     else tuple(expected_course_sha256))
    if (not raw_courses or len(raw_courses) != len(expected_shas)
            or len(set(expected_shas)) != len(expected_shas)):
        raise LearnedDialogueResponseArtifactError("course 集合或 SHA 集合非法")
    course_records = []
    for raw_course, expected_sha in zip(raw_courses, expected_shas):
        course = Path(raw_course).resolve()
        if require_k_drive and course.drive.upper() != "K:":
            raise LearnedDialogueResponseArtifactError("course 必须位于 K 盘")
        if not course.is_file():
            raise LearnedDialogueResponseArtifactError("course 不存在")
        actual_sha = _sha256_file(course)
        if actual_sha != expected_sha:
            raise LearnedDialogueResponseArtifactError("course SHA 漂移")
        course_records.append((actual_sha, course))
    course_records.sort(key=lambda item: item[0])
    all_source_shas: set[str] = set()
    all_license_ids: set[str] = set()
    row_map: dict[tuple[object, ...], DialogueResponseTrainingRow] = {}
    split_by_content: dict[tuple[object, ...], str] = {}
    reconciled_split_collision_count = 0
    for _, course in course_records:
        course_rows, course_source_shas, course_license_ids = _course_rows(course)
        all_source_shas.update(course_source_shas)
        all_license_ids.update(course_license_ids)
        for row in course_rows:
            content_key = (
                row.prompt, row.response, row.history, row.intent_support)
            prior_split = split_by_content.get(content_key)
            if prior_split is not None and prior_split != row.split:
                reconciled_split_collision_count += 1
                split_by_content[content_key] = "heldout"
            elif prior_split is None:
                split_by_content[content_key] = row.split
            existing = row_map.get(content_key)
            if existing is None or row.source_title < existing.source_title:
                row_map[content_key] = row
    rows = tuple(sorted((
        DialogueResponseTrainingRow(
            split_by_content[key], row.prompt, row.response,
            row.source_title, row.history, row.intent_support)
        for key, row in row_map.items()),
        key=lambda item: (
            item.split, item.prompt, item.response, item.history,
            item.intent_support, item.source_title)))
    course_shas = tuple(item[0] for item in course_records)
    actual_course_sha = (
        course_shas[0] if len(course_shas) == 1 else hashlib.sha256(
            _canonical_json_line({"course_sha256s": list(course_shas)})).hexdigest()
    )
    source_shas = tuple(sorted(all_source_shas))
    model = learn_dialogue_response_model(
        rows,
        course_sha256=_sha_bytes(actual_course_sha, label="course SHA"),
        source_sha256s=tuple(_sha_bytes(value, label="source SHA")
                            for value in source_shas),
    )
    intent_model = learn_dialogue_intent_model(rows, model)
    model_payload = encode_integer_tuple(model.integer_stream())
    model_sha = hashlib.sha256(model_payload).hexdigest()
    root = create_new_run_root(
        artifact_root, require_k_drive=require_k_drive,
        label="learned dialogue response artifact root")
    write_exclusive_bytes(root, MODEL_FILE, model_payload,
                          label="learned dialogue response model")
    intent_path = build_sqlite_learned_dialogue_intent_index(
        root.path / SQLITE_INTENT_FILE, intent_model)
    intent_bytes = intent_path.stat().st_size
    intent_sha = _sha256_file(intent_path)
    sqlite_intent_runtime = SqliteLearnedDialogueIntentRuntime(
        intent_path, model.fragments)
    try:
        heldout = _heldout_report(
            rows, model, intent_model, intent_runtime=sqlite_intent_runtime)
    finally:
        sqlite_intent_runtime.close()
    heldout_payload = _canonical_json_line(heldout)
    heldout_sha = hashlib.sha256(heldout_payload).hexdigest()
    license_ids = sorted(all_license_ids)
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "capability_status": heldout["capability_status"],
        "course_sha256": actual_course_sha,
        "course_sha256s": list(course_shas),
        "files": [
            {"bytes": len(model_payload), "name": MODEL_FILE,
             "sha256": model_sha},
            {"bytes": intent_bytes, "name": SQLITE_INTENT_FILE,
             "sha256": intent_sha},
            {"bytes": len(heldout_payload), "name": HELDOUT_FILE,
             "sha256": heldout_sha},
        ],
        "license_id": license_ids[0] if len(license_ids) == 1 else "MULTIPLE",
        "license_ids": license_ids,
        "intent_storage": "SQLITE_INTEGER_POSTING_V1",
        "reconciled_split_collision_count": (
            reconciled_split_collision_count),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_snapshot_sha256": list(source_shas),
        "status": heldout["status"],
        "training_kind": TRAINING_ALGORITHM,
    }
    write_exclusive_bytes(root, HELDOUT_FILE, heldout_payload,
                          label="learned dialogue response heldout")
    write_exclusive_bytes(root, MANIFEST_FILE, _canonical_json_line(manifest),
                          label="learned dialogue response manifest")
    return load_learned_dialogue_response_artifact(
        artifact_root, expected_course_sha256=actual_course_sha,
        require_k_drive=require_k_drive)


def load_learned_dialogue_response_artifact(
        artifact_root: str | Path, *,
        expected_course_sha256: str | None = None,
        require_k_drive: bool = True,
        verify_payload_hashes: bool = True,
        ) -> LearnedDialogueResponseArtifact:
    """只加载闭合 manifest，并逐文件核验大小、SHA 和整数承诺。"""
    root = open_existing_run_root(
        artifact_root, require_k_drive=require_k_drive,
        label="learned dialogue response artifact root")
    with open_plain_binary(root, MANIFEST_FILE,
                           label="learned dialogue response manifest") as stream:
        manifest_payload = stream.read()
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnedDialogueResponseArtifactError("artifact manifest 非法") from error
    identity = ((manifest.get("artifact_kind"), manifest.get("schema_version"),
                 manifest.get("training_kind"))
                if isinstance(manifest, dict) else None)
    current_identity = (
        ARTIFACT_KIND, ARTIFACT_SCHEMA_VERSION, TRAINING_ALGORITHM)
    if (not isinstance(manifest, dict)
            or _canonical_json_line(manifest) != manifest_payload
            or identity not in {current_identity, *_LEGACY_MANIFEST_IDENTITIES}):
        raise LearnedDialogueResponseArtifactError("artifact manifest 漂移")
    is_current = identity == current_identity
    if is_current:
        license_ids = manifest.get("license_ids")
        if (not isinstance(license_ids, list) or not license_ids
                or license_ids != sorted(set(license_ids))
                or any(not isinstance(item, str) or not item
                       for item in license_ids)
                or manifest.get("license_id") != (
                    license_ids[0] if len(license_ids) == 1 else "MULTIPLE")):
            raise LearnedDialogueResponseArtifactError(
                "artifact license inventory 漂移")
    elif manifest.get("license_id") != "Apache-2.0":
        raise LearnedDialogueResponseArtifactError("legacy artifact license 漂移")
    has_legacy_intent = (
        identity in _LEGACY_MANIFEST_IDENTITIES
        and identity not in _LEGACY_NO_INTENT_IDENTITIES
        and identity not in _LEGACY_SQLITE_IDENTITIES)
    has_sqlite_intent = is_current or identity in _LEGACY_SQLITE_IDENTITIES
    course_sha = manifest.get("course_sha256")
    if (not isinstance(course_sha, str)
            or expected_course_sha256 is not None
            and course_sha != expected_course_sha256):
        raise LearnedDialogueResponseArtifactError("artifact course 绑定漂移")
    files = manifest.get("files")
    expected_names = (
        {MODEL_FILE, SQLITE_INTENT_FILE, HELDOUT_FILE} if has_sqlite_intent else
        {MODEL_FILE, INTENT_MODEL_FILE, HELDOUT_FILE}
        if has_legacy_intent else {MODEL_FILE, HELDOUT_FILE})
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise LearnedDialogueResponseArtifactError("artifact inventory 非法")
    inventory = {item.get("name"): item for item in files
                 if isinstance(item, dict)}
    if set(inventory) != expected_names:
        raise LearnedDialogueResponseArtifactError("artifact 文件集合漂移")
    payloads: dict[str, bytes] = {}
    for name in sorted(expected_names):
        item = inventory[name]
        path = root.path / name
        if (not path.is_file() or item.get("bytes") != path.stat().st_size
                or verify_payload_hashes
                and item.get("sha256") != _sha256_file(path)):
            raise LearnedDialogueResponseArtifactError(
                f"artifact payload {name} 漂移")
        if name != SQLITE_INTENT_FILE:
            with open_plain_binary(root, name, label=name) as stream:
                payloads[name] = stream.read()
    try:
        model = LearnedDialogueResponseModel.from_integer_stream(
            decode_integer_tuple(payloads[MODEL_FILE]))
    except (LearnedDialogueResponseError, TypeError, ValueError) as error:
        raise LearnedDialogueResponseArtifactError("model payload 非法") from error
    intent_model = None
    intent_index_path = None
    if has_legacy_intent:
        try:
            intent_model = LearnedDialogueIntentModel.from_integer_stream(
                decode_integer_tuple(payloads[INTENT_MODEL_FILE]))
        except (LearnedDialogueResponseError, TypeError, ValueError) as error:
            raise LearnedDialogueResponseArtifactError(
                "intent model payload 非法") from error
        if intent_model.fragment_count != len(model.fragments):
            raise LearnedDialogueResponseArtifactError(
                "intent/response fragment table 漂移")
    elif has_sqlite_intent:
        if manifest.get("intent_storage") != "SQLITE_INTEGER_POSTING_V1":
            raise LearnedDialogueResponseArtifactError(
                "SQLite intent storage 身份漂移")
        intent_index_path = root.path / SQLITE_INTENT_FILE
        try:
            validate_sqlite_learned_dialogue_intent_index(
                intent_index_path, expected_train_count=None,
                expected_fragment_count=len(model.fragments))
        except LearnedDialogueResponseError as error:
            raise LearnedDialogueResponseArtifactError(
                "SQLite intent index 非法") from error
    if bytes(model.course_sha256).hex() != course_sha:
        raise LearnedDialogueResponseArtifactError("model course SHA 漂移")
    source_shas = manifest.get("source_snapshot_sha256")
    if (not isinstance(source_shas, list)
            or tuple(bytes(item).hex() for item in model.source_sha256s)
            != tuple(source_shas)):
        raise LearnedDialogueResponseArtifactError("model source SHA 漂移")
    if is_current:
        course_shas = manifest.get("course_sha256s")
        if (not isinstance(course_shas, list) or not course_shas
                or course_shas != sorted(set(course_shas))
                or any(not isinstance(item, str) or len(item) != 64
                       or any(value not in "0123456789abcdef" for value in item)
                       for item in course_shas)):
            raise LearnedDialogueResponseArtifactError(
                "artifact course SHA inventory 漂移")
    try:
        heldout = json.loads(payloads[HELDOUT_FILE].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnedDialogueResponseArtifactError("heldout payload 非法") from error
    if (_canonical_json_line(heldout) != payloads[HELDOUT_FILE]
            or not isinstance(heldout, dict)
            or heldout.get("status") != manifest.get("status")
            or heldout.get("capability_status")
            != manifest.get("capability_status")):
        raise LearnedDialogueResponseArtifactError("heldout 承诺漂移")
    return LearnedDialogueResponseArtifact(
        model, intent_model, str(manifest["status"]),
        str(manifest["capability_status"]),
        str(inventory[MODEL_FILE]["sha256"]),
        str(inventory[HELDOUT_FILE]["sha256"]), str(course_sha),
        intent_index_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build learned OpenAssistant dialogue response artifact")
    parser.add_argument("--course", action="append", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--expected-course-sha256", action="append", required=True)
    args = parser.parse_args(argv)
    artifact = build_learned_dialogue_response_artifact(
        course_path=tuple(args.course), artifact_root=args.artifact_root,
        expected_course_sha256=tuple(args.expected_course_sha256))
    print(json.dumps({
        "capability_status": artifact.capability_status,
        "course_sha256": artifact.course_sha256,
        "heldout_sha256": artifact.heldout_sha256,
        "model_sha256": artifact.model_sha256,
        "status": artifact.status,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LearnedDialogueResponseArtifact",
    "LearnedDialogueResponseArtifactError",
    "build_learned_dialogue_response_artifact",
    "load_learned_dialogue_response_artifact", "main",
]
