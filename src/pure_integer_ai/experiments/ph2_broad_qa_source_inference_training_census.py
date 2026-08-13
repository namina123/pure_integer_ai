"""来源归纳训练总体的只读 operator 可行性盘点。

盘点只报告由来源字节机械计算出的支持信号、反例信号和不可判定项。它不把
信号写成语义标签，不选择 operator，不生成 Evidence、规则或 mastery。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_feasibility import (
    minimum_source_segments,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training import (
    SOURCE_INFERENCE_OPERATOR_FAMILIES,
    SOURCE_INFERENCE_TRAINING_ASSIGNMENTS,
    SOURCE_INFERENCE_TRAINING_SOURCES,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_dossier import (
    SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
    read_source_inference_training_dossier,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_INFERENCE_TRAINING_CENSUS_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_OPERATOR_CENSUS_V2")
SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_OPERATOR_CENSUS_RECORD_V2")
MECHANICAL_SIGNAL_STATES = (
    "MECHANICAL_SUPPORT_SIGNAL",
    "MECHANICAL_COUNTER_SIGNAL",
    "UNDETERMINED",
)
_ENUMERATION_DELIMITERS = frozenset("、,，;；")
_ENUMERATION_BOUNDARIES = frozenset("。！？!?\n")
_ENUMERATION_MEMBER_TRIM = " \t\r\n：:()（）[]【】“”‘’\"'"
_OPEN_TO_CLOSE = {"(": ")", "（": "）"}
_UNREACHABLE = -1


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入和盘点 artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """要求盘点输入输出始终位于显式 K 盘 run root 内。"""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _read_dossier_manifest(path: Path) -> dict[str, object]:
    """严格回读未学习训练 dossier manifest。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference training dossier manifest 不可读") from error
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or value.get("artifact_kind")
            != SOURCE_INFERENCE_TRAINING_DOSSIER_KIND
            or value.get("format_version") != 1
            or value.get("status") != "MATERIALIZED_UNREAD_UNLEARNED"
            or value.get("learner_read_count") != 0
            or value.get("rules_written") != 0
            or value.get("semantic_labels_written") != 0):
        raise BroadQaExternalDataError(
            "source inference training dossier manifest 漂移")
    return value


def _parenthetical_contents(text: str) -> tuple[str, ...]:
    """机械提取中英文圆括号内文本，嵌套时保留每层内容。"""
    stack: list[tuple[str, int]] = []
    values = []
    for index, character in enumerate(text):
        if character in _OPEN_TO_CLOSE:
            stack.append((character, index + 1))
            continue
        if not stack or character != _OPEN_TO_CLOSE[stack[-1][0]]:
            continue
        _, start = stack.pop()
        if start < index:
            values.append(text[start:index])
    return tuple(values)


def _enumeration_members(text: str) -> tuple[str, ...]:
    """在句界内切分枚举，并只返回去边界后的机械成员。"""
    values = []
    clauses = []
    start = 0
    for index, character in enumerate(text):
        if character not in _ENUMERATION_BOUNDARIES:
            continue
        if start < index:
            clauses.append(text[start:index])
        start = index + 1
    if start < len(text):
        clauses.append(text[start:])
    for clause in clauses:
        members = []
        member_start = 0
        delimiter_count = 0
        for index, character in enumerate(clause):
            if character not in _ENUMERATION_DELIMITERS:
                continue
            delimiter_count += 1
            if member_start < index:
                member = clause[member_start:index].strip(
                    _ENUMERATION_MEMBER_TRIM)
                if member:
                    members.append(member)
            member_start = index + 1
        if member_start < len(clause):
            member = clause[member_start:].strip(_ENUMERATION_MEMBER_TRIM)
            if member:
                members.append(member)
        if delimiter_count and len(members) >= 2:
            values.extend(members)
    return tuple(values)


def _signal(
        *,
        support: bool,
        counter: bool,
        support_reason: str,
        counter_reason: str,
        undetermined_reason: str,
        ) -> tuple[str, str]:
    """把互斥机械判据归并为信号状态及稳定理由。"""
    if support and counter:
        raise BroadQaExternalDataError(
            "source inference training census 信号判据冲突")
    if support:
        return "MECHANICAL_SUPPORT_SIGNAL", support_reason
    if counter:
        return "MECHANICAL_COUNTER_SIGNAL", counter_reason
    return "UNDETERMINED", undetermined_reason


def _item_signals(value: dict[str, object]) -> dict[str, tuple[str, str]]:
    """为单题计算六类 operator 的非语义机械信号。"""
    terminal = value["terminal_source"]
    training = value["training_source"]
    wikitext = str(terminal["wikitext"])
    passage_texts = tuple(
        str(item["text"]) for item in terminal["passages"])
    passage_source = "".join(passage_texts)
    normalized_wikitext = normalize_external_text(wikitext)
    normalized_passages = normalize_external_text(passage_source)
    normalized_gold = tuple(
        normalize_external_text(str(item))
        for item in training["gold_answers"])
    if any(not item for item in normalized_gold):
        raise BroadQaExternalDataError(
            "source inference training census normalized gold 为空")
    exact_wikitext_hit = any(
        str(item) in wikitext for item in training["gold_answers"])
    normalized_wikitext_hit = any(
        item in normalized_wikitext for item in normalized_gold)
    normalized_passage_hit = any(
        item in normalized_passages for item in normalized_gold)

    parenthetical = tuple(filter(None, (
        normalize_external_text(item)
        for item in _parenthetical_contents(wikitext))))
    parenthetical_hit = any(
        gold in content
        for gold in normalized_gold for content in parenthetical)
    enumeration = tuple(filter(None, (
        normalize_external_text(item)
        for item in _enumeration_members(str(terminal["plain_text"])))))
    enumeration_hit = any(
        gold == member
        for gold in normalized_gold for member in enumeration)
    minimum_segments = min(
        minimum_source_segments(normalized_passages, item)
        for item in normalized_gold)

    signals = {}
    signals["NORMALIZATION_EQUIVALENCE"] = _signal(
        support=normalized_wikitext_hit and not exact_wikitext_hit,
        counter=exact_wikitext_hit,
        support_reason="NORMALIZATION_RESCUES_EXACT_MISS",
        counter_reason="EXACT_SOURCE_HIT_NEEDS_NO_NORMALIZATION_RESCUE",
        undetermined_reason="NO_NORMALIZED_SOURCE_HIT",
    )
    signals["SOURCE_SPAN_SELECTION"] = _signal(
        support=normalized_passage_hit,
        counter=normalized_wikitext_hit and not normalized_passage_hit,
        support_reason="GOLD_OCCURS_IN_PROJECTED_PASSAGE",
        counter_reason="GOLD_OCCURS_ONLY_OUTSIDE_PROJECTED_PASSAGES",
        undetermined_reason="GOLD_ABSENT_FROM_NORMALIZED_TERMINAL_SOURCE",
    )
    signals["PARENTHETICAL_EXPANSION"] = _signal(
        support=parenthetical_hit,
        counter=bool(parenthetical) and normalized_wikitext_hit
        and not parenthetical_hit,
        support_reason="GOLD_OCCURS_IN_PARENTHETICAL_BYTES",
        counter_reason="PARENTHETICAL_BYTES_EXIST_WITHOUT_GOLD",
        undetermined_reason="NO_DECISIVE_PARENTHETICAL_BYTE_PATTERN",
    )
    signals["ENUMERATION_MEMBER_SELECTION"] = _signal(
        support=enumeration_hit,
        counter=bool(enumeration) and normalized_wikitext_hit
        and not enumeration_hit,
        support_reason="GOLD_EQUALS_DELIMITED_MEMBER_BYTES",
        counter_reason="GOLD_OCCURS_OUTSIDE_EXACT_DELIMITED_MEMBERS",
        undetermined_reason="NO_DECISIVE_ENUMERATION_BYTE_PATTERN",
    )
    signals["EXPLICIT_UNIT_ERA_FORMAT_MAPPING"] = (
        "UNDETERMINED",
        "NO_FROZEN_LEARNED_UNIT_ERA_SURFACE_PACK",
    )
    signals["FINITE_ROLE_COMPOSITION"] = _signal(
        support=2 <= minimum_segments <= 4,
        counter=minimum_segments == _UNREACHABLE,
        support_reason="GOLD_REQUIRES_TWO_TO_FOUR_ORDERED_SOURCE_SEGMENTS",
        counter_reason="GOLD_UNREACHABLE_FROM_ORDERED_PASSAGE_BYTES",
        undetermined_reason=(
            "DIRECT_OR_HIGH_SEGMENT_BYTE_PATTERN_NOT_ROLE_EVIDENCE"),
    )
    if set(signals) != set(SOURCE_INFERENCE_OPERATOR_FAMILIES):
        raise BroadQaExternalDataError(
            "source inference training census operator inventory 漂移")
    return signals


def audit_source_inference_training_operator_census(
        *,
        run_root: str | Path,
        dossier_manifest_path: str | Path,
        dossier_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """发布六类 operator 的机械信号库存，不读取或写入规则。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError(
            "source inference training census run root 不存在")
    manifest_file = _within(
        root, dossier_manifest_path, label="dossier_manifest_path")
    dossier_file = _within(root, dossier_path, label="dossier_path")
    target = _within(root, target_dir, label="target_dir")
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference training census target 已存在")
    dossier_manifest = _read_dossier_manifest(manifest_file)
    if (_sha256_file(dossier_file) != dossier_manifest.get("dossier_sha256")
            or dossier_file.stat().st_size
            != dossier_manifest.get("dossier_bytes")):
        raise BroadQaExternalDataError(
            "source inference training dossier commitment 漂移")
    dossier = read_source_inference_training_dossier(dossier_file)
    if len(dossier) != dossier_manifest.get("dossier_record_count"):
        raise BroadQaExternalDataError(
            "source inference training dossier count 漂移")

    records = []
    family_state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_family_state_counts: dict[
        str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter))
    assignment_family_state_counts: dict[
        str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter))
    for value in dossier:
        source_key = str(value["training_source"]["source_key"])
        assignment = str(value["training_assignment"])
        if (source_key not in SOURCE_INFERENCE_TRAINING_SOURCES
                or assignment not in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS):
            raise BroadQaExternalDataError(
                "source inference training census source/assignment 漂移")
        for family, (state, reason) in _item_signals(value).items():
            family_state_counts[family][state] += 1
            source_family_state_counts[source_key][family][state] += 1
            assignment_family_state_counts[assignment][family][state] += 1
            records.append({
                "format_version": 1,
                "item_id": value["item_id"],
                "mechanical_reason": reason,
                "mechanical_signal_state": state,
                "operator_family": family,
                "record_kind": SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND,
                "rules_written": 0,
                "semantic_label_written": 0,
                "source_key": source_key,
                "training_assignment": assignment,
            })

    target.mkdir(parents=True)
    records_path = target / "operator-census.records.jsonl"
    with records_path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))

    def state_counts(values) -> dict[str, int]:
        """补零并按固定三态顺序投影计数。"""
        return {state: values[state] for state in MECHANICAL_SIGNAL_STATES}

    census_manifest = {
        "artifact_kind": SOURCE_INFERENCE_TRAINING_CENSUS_KIND,
        "assignment_family_state_counts": {
            assignment: {
                family: state_counts(
                    assignment_family_state_counts[assignment][family])
                for family in SOURCE_INFERENCE_OPERATOR_FAMILIES
            }
            for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
        },
        "dossier_manifest_sha256": _sha256_file(manifest_file),
        "dossier_sha256": _sha256_file(dossier_file),
        "format_version": 1,
        "item_count": len(dossier),
        "learner_read_count": 0,
        "operator_family_state_counts": {
            family: state_counts(family_state_counts[family])
            for family in SOURCE_INFERENCE_OPERATOR_FAMILIES
        },
        "operator_preassigned_count": 0,
        "record_count": len(records),
        "records_sha256": _sha256_file(records_path),
        "rules_written": 0,
        "semantic_labels_written": 0,
        "source_family_state_counts": {
            source: {
                family: state_counts(
                    source_family_state_counts[source][family])
                for family in SOURCE_INFERENCE_OPERATOR_FAMILIES
            }
            for source in SOURCE_INFERENCE_TRAINING_SOURCES
        },
        "status": "MECHANICAL_CENSUS_ONLY_NOT_LEARNED",
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(census_manifest))
    return {
        **census_manifest,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _work_path(value: str) -> Path:
    """要求训练盘点大数据路径为显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从未学习 training dossier 发布只读 operator census。"""
    parser = argparse.ArgumentParser(
        description="Audit mechanical source-inference operator signals.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--dossier-manifest", type=_work_path, required=True)
    parser.add_argument("--dossier", type=_work_path, required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    args = parser.parse_args(argv)
    report = audit_source_inference_training_operator_census(
        run_root=args.run_root,
        dossier_manifest_path=args.dossier_manifest,
        dossier_path=args.dossier,
        target_dir=args.target_dir,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MECHANICAL_SIGNAL_STATES",
    "SOURCE_INFERENCE_TRAINING_CENSUS_KIND",
    "SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND",
    "audit_source_inference_training_operator_census",
    "main",
]
