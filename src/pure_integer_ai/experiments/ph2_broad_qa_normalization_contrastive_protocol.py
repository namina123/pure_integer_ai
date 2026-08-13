"""从冻结 OpenCC 来源形成 normalization 对比训练来源，不形成 learned rule。

字符词典提供候选映射；短语词典在相同位置明确给出目标字符时，才能形成该候选
在对应短语上下文中的 SOURCE_REPLAY_SUPPORT/REFUTE。这里的 REFUTE 只否决
该上下文中的字符级应用，不声明一般语义不等价，也不自动学习 defeater。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
    NORMALIZATION_SOURCE_PACK_STATUS,
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_CONTRASTIVE_TRAIN_SOURCE_V1")
NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_MAPPING_CANDIDATE_V1")
NORMALIZATION_CONTRASTIVE_TRIAL_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_CONTEXT_TRIAL_V1")
NORMALIZATION_CONTRASTIVE_STATUS = (
    "FROZEN_TRAIN_SOURCE_NOT_LEARNED_NO_EVALUATION")
NORMALIZATION_CONTRASTIVE_FAMILY = "NORMALIZATION_EQUIVALENCE"
NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN = (
    "OPENCC_T2S_SOURCE_BEHAVIOR_V1")
NORMALIZATION_CONTRASTIVE_QUALIFICATIONS = (
    "SOURCE_REPLAY_REFUTE",
    "SOURCE_REPLAY_SUPPORT",
)


def _sha256(payload: bytes) -> str:
    """返回规范记录或来源字节的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _source_lines(
        payload: bytes,
        *,
        relative_path: str,
        file_sha256: str,
        ) -> tuple[dict[str, object], ...]:
    """解析 UTF-8 单 tab 词典并保留逐行物理字节坐标。"""
    lines = payload.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise BroadQaExternalDataError(
            f"contrastive source {relative_path} 为空或截断")
    records = []
    byte_start = 0
    keys = set()
    for ordinal, encoded_line in enumerate(lines, start=1):
        raw_line = encoded_line[:-1]
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BroadQaExternalDataError(
                f"contrastive source {relative_path} 非 UTF-8") from error
        if text.count("\t") != 1:
            raise BroadQaExternalDataError(
                f"contrastive source {relative_path} 第 {ordinal} 行非法")
        source, targets = text.split("\t")
        target_variants = targets.split(" ")
        if (not source or source in keys
                or any(not target for target in target_variants)):
            raise BroadQaExternalDataError(
                f"contrastive source {relative_path} key/value 漂移")
        keys.add(source)
        byte_end = byte_start + len(encoded_line)
        records.append({
            "byte_end": byte_end,
            "byte_start": byte_start,
            "file_sha256": file_sha256,
            "line_ordinal": ordinal,
            "line_sha256": _sha256(encoded_line),
            "relative_path": relative_path,
            "source": source,
            "target_variants": target_variants,
        })
        byte_start = byte_end
    if byte_start != len(payload):
        raise BroadQaExternalDataError(
            f"contrastive source {relative_path} 字节覆盖漂移")
    return tuple(records)


def _line_commitment(line: dict[str, object]) -> dict[str, object]:
    """投影不含词典内容的物理行承诺。"""
    return {
        "byte_end": line["byte_end"],
        "byte_start": line["byte_start"],
        "file_sha256": line["file_sha256"],
        "line_ordinal": line["line_ordinal"],
        "line_sha256": line["line_sha256"],
        "relative_path": line["relative_path"],
    }


def derive_normalization_contrastive_records(
        *,
        source_pack_manifest_sha256: str,
        character_payload: bytes,
        phrase_payload: bytes,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """派生字符候选和短语上下文 trial，不进行规则采用或验收。"""
    if (not isinstance(source_pack_manifest_sha256, str)
            or len(source_pack_manifest_sha256) != 64
            or any(item not in "0123456789abcdef"
                   for item in source_pack_manifest_sha256)):
        raise BroadQaExternalDataError(
            "contrastive source pack manifest SHA 非法")
    if not isinstance(character_payload, bytes) or not isinstance(
            phrase_payload, bytes):
        raise BroadQaExternalDataError("contrastive dictionary payload 非法")
    character_path = "dictionary/TSCharacters.txt"
    phrase_path = "dictionary/TSPhrases.txt"
    character_lines = _source_lines(
        character_payload,
        relative_path=character_path,
        file_sha256=NORMALIZATION_SOURCE_FILES[character_path]["sha256"],
    )
    phrase_lines = _source_lines(
        phrase_payload,
        relative_path=phrase_path,
        file_sha256=NORMALIZATION_SOURCE_FILES[phrase_path]["sha256"],
    )
    candidate_by_source = {}
    candidates = []
    for line in character_lines:
        source = line["source"]
        target = line["target_variants"][0]
        if len(source) != 1 or len(target) != 1:
            raise BroadQaExternalDataError(
                "normalization character candidate 必须一对一码点")
        identity = {
            "input_codepoint": ord(source),
            "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
            "output_codepoint": ord(target),
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
        }
        candidate_id = _sha256(canonical_json_bytes(identity))
        candidate = {
            "accepted_rule_written": 0,
            "application_domain": NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
            "candidate_id": candidate_id,
            "direct_mapping_source": 1,
            "format_version": 1,
            "input_codepoint": ord(source),
            "input_surface": source,
            "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
            "output_codepoint": ord(target),
            "output_surface": target,
            "record_kind": NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND,
            "semantic_non_equivalence_label_written": 0,
            "source_commitment": _line_commitment(line),
            "source_pack_manifest_sha256": source_pack_manifest_sha256,
            "split": "TRAIN_SOURCE",
            "target_variant_count": len(line["target_variants"]),
        }
        candidate_by_source[source] = candidate
        candidates.append(candidate)

    trials = []
    qualification_counts: Counter[str] = Counter()
    candidate_qualifications: defaultdict[str, set[str]] = defaultdict(set)
    for line in phrase_lines:
        source_phrase = line["source"]
        target_phrase = line["target_variants"][0]
        if len(source_phrase) != len(target_phrase):
            raise BroadQaExternalDataError(
                "normalization phrase trial 不是位置保持映射")
        for offset, (source, observed) in enumerate(zip(
                source_phrase, target_phrase)):
            candidate = candidate_by_source.get(source)
            if candidate is None or candidate["output_surface"] == source:
                continue
            proposed = candidate["output_surface"]
            qualification = (
                "SOURCE_REPLAY_SUPPORT"
                if proposed == observed else "SOURCE_REPLAY_REFUTE")
            identity = {
                "candidate_id": candidate["candidate_id"],
                "phrase_line_sha256": line["line_sha256"],
                "source_codepoint_offset": offset,
                "source_pack_manifest_sha256": source_pack_manifest_sha256,
            }
            trial_id = _sha256(canonical_json_bytes(identity))
            trials.append({
                "candidate_id": candidate["candidate_id"],
                "candidate_output_codepoint": candidate["output_codepoint"],
                "candidate_output_surface": proposed,
                "defeater_written": 0,
                "format_version": 1,
                "observed_output_codepoint": ord(observed),
                "observed_output_surface": observed,
                "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
                "phrase_source": source_phrase,
                "phrase_target": target_phrase,
                "qualification_kind": qualification,
                "record_kind": NORMALIZATION_CONTRASTIVE_TRIAL_KIND,
                "rejection_record_written": 0,
                "semantic_non_equivalence_label_written": 0,
                "source_codepoint": ord(source),
                "source_codepoint_offset": offset,
                "source_commitment": _line_commitment(line),
                "source_pack_manifest_sha256": source_pack_manifest_sha256,
                "split": "TRAIN_SOURCE",
                "trial_id": trial_id,
            })
            qualification_counts[qualification] += 1
            candidate_qualifications[candidate["candidate_id"]].add(
                qualification)

    candidates.sort(key=lambda item: item["candidate_id"])
    trials.sort(key=lambda item: item["trial_id"])
    candidate_ids = tuple(item["candidate_id"] for item in candidates)
    trial_ids = tuple(item["trial_id"] for item in trials)
    if (candidate_ids != tuple(sorted(set(candidate_ids)))
            or trial_ids != tuple(sorted(set(trial_ids)))):
        raise BroadQaExternalDataError(
            "normalization contrastive identity 非唯一规范排序")
    summary = {
        "candidate_count": len(candidates),
        "candidate_with_both_qualifications_count": sum(
            states == set(NORMALIZATION_CONTRASTIVE_QUALIFICATIONS)
            for states in candidate_qualifications.values()),
        "candidate_with_refute_count": sum(
            "SOURCE_REPLAY_REFUTE" in states
            for states in candidate_qualifications.values()),
        "candidate_with_support_count": sum(
            "SOURCE_REPLAY_SUPPORT" in states
            for states in candidate_qualifications.values()),
        "source_replay_refute_count": qualification_counts[
            "SOURCE_REPLAY_REFUTE"],
        "source_replay_support_count": qualification_counts[
            "SOURCE_REPLAY_SUPPORT"],
        "trial_count": len(trials),
    }
    if not candidates or not trials or not all(summary.values()):
        raise BroadQaExternalDataError(
            "normalization contrastive 来源库存不足")
    return tuple(candidates), tuple(trials), summary


def _derive_from_source_pack(
        source_pack_dir: Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
        ]:
    """严格回读来源 pack 后重新派生全部训练来源记录。"""
    source_manifest = read_normalization_source_pack(source_pack_dir)
    if source_manifest["status"] != NORMALIZATION_SOURCE_PACK_STATUS:
        raise BroadQaExternalDataError(
            "normalization contrastive source pack 状态漂移")
    candidates, trials, summary = derive_normalization_contrastive_records(
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        character_payload=(
            source_pack_dir / "dictionary" / "TSCharacters.txt").read_bytes(),
        phrase_payload=(
            source_pack_dir / "dictionary" / "TSPhrases.txt").read_bytes(),
    )
    return source_manifest, candidates, trials, summary


def publish_normalization_contrastive_protocol(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 TRAIN_SOURCE 候选/trial，不发布 rule 或 evaluation。"""
    root = Path(run_root).resolve()
    source_root = Path(source_pack_dir).resolve()
    target = Path(target_dir).resolve()
    if (not root.is_dir() or not source_root.is_relative_to(root)
            or not target.is_relative_to(root)):
        raise BroadQaExternalDataError(
            "normalization contrastive source/target 必须位于 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization contrastive target 已存在")
    source_manifest, candidates, trials, summary = _derive_from_source_pack(
        source_root)
    candidate_payload = b"".join(
        canonical_json_line(item) for item in candidates)
    trial_payload = b"".join(canonical_json_line(item) for item in trials)
    target.mkdir(parents=True)
    candidates_path = target / "mapping-candidates.jsonl"
    trials_path = target / "context-trials.jsonl"
    candidates_path.write_bytes(candidate_payload)
    trials_path.write_bytes(trial_payload)
    manifest = {
        "accepted_rules_written": 0,
        "application_domain": NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
        "artifact_kind": NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND,
        "candidate_records_bytes": len(candidate_payload),
        "candidate_records_sha256": _sha256(candidate_payload),
        "defeaters_written": 0,
        "evaluation_record_count": 0,
        "format_version": 1,
        "learner_read_count": 0,
        "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
        "production_enabled": 0,
        "rejection_records_written": 0,
        "reserve_record_count": 0,
        "semantic_non_equivalence_label_count": 0,
        "source_pack_manifest_sha256": source_manifest["manifest_sha256"],
        "status": NORMALIZATION_CONTRASTIVE_STATUS,
        "summary": summary,
        "trial_records_bytes": len(trial_payload),
        "trial_records_sha256": _sha256(trial_payload),
        "validation_record_count": 0,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def _read_canonical_jsonl(
        payload: bytes,
        *,
        label: str,
        ) -> tuple[dict[str, object], ...]:
    """回读非空规范 JSONL，并拒绝尾随或非 object 记录。"""
    if not payload or not payload.endswith(b"\n"):
        raise BroadQaExternalDataError(f"{label} 为空或截断")
    records = []
    try:
        for line in payload.splitlines(keepends=True):
            value = json.loads(line)
            if not isinstance(value, dict) or canonical_json_line(value) != line:
                raise BroadQaExternalDataError(f"{label} 不是规范 JSONL")
            records.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(f"{label} 不可解析") from error
    return tuple(records)


def read_normalization_contrastive_protocol(
        target_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
        ]:
    """严格回读并从来源 pack 独立重派生逐字验证全部记录。"""
    root = Path(target_dir).resolve()
    source_root = Path(source_pack_dir).resolve()
    manifest_path = root / "manifest.json"
    candidate_path = root / "mapping-candidates.jsonl"
    trial_path = root / "context-trials.jsonl"
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        candidate_payload = candidate_path.read_bytes()
        trial_payload = trial_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization contrastive artifact 不可读") from error
    expected = {
        "accepted_rules_written", "application_domain", "artifact_kind",
        "candidate_records_bytes", "candidate_records_sha256",
        "defeaters_written", "evaluation_record_count", "format_version",
        "learner_read_count", "operator_family", "production_enabled",
        "rejection_records_written", "reserve_record_count",
        "semantic_non_equivalence_label_count",
        "source_pack_manifest_sha256", "status", "summary",
        "trial_records_bytes", "trial_records_sha256",
        "validation_record_count",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != manifest_payload
            or manifest["artifact_kind"]
            != NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"] != NORMALIZATION_CONTRASTIVE_FAMILY
            or manifest["application_domain"]
            != NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN
            or manifest["status"] != NORMALIZATION_CONTRASTIVE_STATUS
            or any(type(manifest[name]) is not int or manifest[name] != 0
                   for name in (
                       "accepted_rules_written", "defeaters_written",
                       "evaluation_record_count", "learner_read_count",
                       "production_enabled", "rejection_records_written",
                       "reserve_record_count",
                       "semantic_non_equivalence_label_count",
                       "validation_record_count"))
            or type(manifest["candidate_records_bytes"]) is not int
            or manifest["candidate_records_bytes"] != len(candidate_payload)
            or type(manifest["trial_records_bytes"]) is not int
            or manifest["trial_records_bytes"] != len(trial_payload)
            or manifest["candidate_records_sha256"]
            != _sha256(candidate_payload)
            or manifest["trial_records_sha256"] != _sha256(trial_payload)):
        raise BroadQaExternalDataError(
            "normalization contrastive manifest 漂移")
    for name in (
            "candidate_records_sha256", "source_pack_manifest_sha256",
            "trial_records_sha256"):
        value = manifest[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(item not in "0123456789abcdef" for item in value)):
            raise BroadQaExternalDataError(
                f"normalization contrastive {name} 非法")
    candidates = _read_canonical_jsonl(
        candidate_payload, label="normalization candidates")
    trials = _read_canonical_jsonl(
        trial_payload, label="normalization trials")
    source_manifest, expected_candidates, expected_trials, summary = (
        _derive_from_source_pack(source_root))
    if (manifest["source_pack_manifest_sha256"]
            != source_manifest["manifest_sha256"]
            or candidates != expected_candidates
            or trials != expected_trials
            or manifest["summary"] != summary):
        raise BroadQaExternalDataError(
            "normalization contrastive records/source 漂移")
    return ({
        **manifest,
        "manifest_sha256": _sha256(manifest_payload),
    }, candidates, trials)


def main(argv: list[str] | None = None) -> int:
    """发布或严格回读 normalization 对比训练来源。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-root", required=True)
    publish.add_argument("--source-pack-dir", required=True)
    publish.add_argument("--target-dir", required=True)
    read = subparsers.add_parser("read")
    read.add_argument("--source-pack-dir", required=True)
    read.add_argument("--target-dir", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "publish":
        report = publish_normalization_contrastive_protocol(
            run_root=arguments.run_root,
            source_pack_dir=arguments.source_pack_dir,
            target_dir=arguments.target_dir,
        )
    else:
        report, _, _ = read_normalization_contrastive_protocol(
            arguments.target_dir,
            source_pack_dir=arguments.source_pack_dir,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN",
    "NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND",
    "NORMALIZATION_CONTRASTIVE_FAMILY",
    "NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND",
    "NORMALIZATION_CONTRASTIVE_QUALIFICATIONS",
    "NORMALIZATION_CONTRASTIVE_STATUS",
    "NORMALIZATION_CONTRASTIVE_TRIAL_KIND",
    "derive_normalization_contrastive_records",
    "publish_normalization_contrastive_protocol",
    "read_normalization_contrastive_protocol",
]
