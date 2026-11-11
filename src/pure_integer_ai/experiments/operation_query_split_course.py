"""Separate operation request inputs from train-fact ownership and evaluator labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes


SPLITS = ("train", "held_out", "negative", "conflict")
OPERATION_DUTIES = (
    "direct", "explanation", "comparison", "causal",
    "temporal", "summary", "rewrite", "reference",
)
LABEL_POLICIES = {"held_out": frozenset({"ANSWER_CLOSED"}),
                  "negative": frozenset({"UNKNOWN", "REFUSE_UNAUTHORIZED"}),
                  "conflict": frozenset({"CLARIFY_CONFLICT"})}
_QUERY_FIELDS = frozenset({
    "case_id", "chain_id", "duty", "question", "split", "owner", "license_id",
})
_LABEL_FIELDS = frozenset({"case_id", "policy", "split", "owner", "license_id"})


class OperationQuerySplitError(ValueError):
    """Query and evaluator owners do not form an isolated four-way course."""


@dataclass(frozen=True, slots=True)
class OperationQueryCase:
    """One natural input whose identity is independent of any teacher fact row."""

    case_id: str
    chain_id: str
    duty: str
    question: str
    split: str
    owner: str
    license_id: str


@dataclass(frozen=True, slots=True)
class OperationQuerySplitAudit:
    """Structural split receipt; it cannot qualify semantic behavior."""

    case_counts: tuple[tuple[str, int], ...]
    query_sha256: tuple[tuple[str, str], ...]
    evaluator_label_sha256: tuple[tuple[str, str], ...]
    held_out_duty_count: int
    train_duty_count: int
    status: str
    semantic_behavior_qualification: str = "UNKNOWN_NOT_RUN"
    novel_topology_status: str = "UNKNOWN_NOT_REPRESENTED"
    production_authorized: int = 0
    free_dialogue_complete: int = 0


def _read_jsonl(path: Path, fields: frozenset[str]) -> tuple[dict, ...]:
    """读取有界 canonical JSONL，禁止宽松解析和字段扩张。"""
    raw = path.read_bytes()
    if (not raw or len(raw) > 16 * 1024 * 1024 or not raw.endswith(b"\n")
            or raw.endswith(b"\n\n")):
        raise OperationQuerySplitError("query course needs bounded JSONL with one final newline")
    rows = []
    for ordinal, line in enumerate(raw.splitlines(), 1):
        try:
            row = parse_canonical_json_bytes(line, require_object=True)
        except ValueError as error:
            raise OperationQuerySplitError(f"query course line {ordinal} is not canonical JSON") from error
        if set(row) != fields:
            raise OperationQuerySplitError(f"query course line {ordinal} fields drifted")
        rows.append(row)
    return tuple(rows)


def _text(value: object, where: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise OperationQuerySplitError(f"{where} must be nonempty trimmed text")
    return value


def read_train_operation_queries(path: str | Path) -> tuple[OperationQueryCase, ...]:
    """只读取 train 问句，不接受 held-out、negative、conflict 或事实 seed。"""
    source = Path(path).resolve(strict=True)
    if "evaluator" in source.parts:
        raise OperationQuerySplitError(
            "learner query source must be train input only and cannot be under the evaluator owner")
    rows = _read_jsonl(source, _QUERY_FIELDS)
    cases = tuple(OperationQueryCase(
        _text(row["case_id"], "case_id"), _text(row["chain_id"], "chain_id"),
        _text(row["duty"], "duty"), _text(row["question"], "question"),
        _text(row["split"], "split"), _text(row["owner"], "owner"),
        _text(row["license_id"], "license_id")) for row in rows)
    if (not cases or any(case.split != "train" or case.owner != "teacher"
            or case.license_id != "CC0-1.0" or case.duty not in OPERATION_DUTIES
            for case in cases)
            or len({case.case_id for case in cases}) != len(cases)
            or len({case.question for case in cases}) != len(cases)):
        raise OperationQuerySplitError("learner query source must be unique licensed teacher train input only")
    return cases


def _read_split_queries(path: Path, split: str) -> tuple[OperationQueryCase, ...]:
    rows = _read_jsonl(path, _QUERY_FIELDS)
    expected_owner = "teacher" if split == "train" else "evaluator"
    cases = tuple(OperationQueryCase(
        _text(row["case_id"], "case_id"), _text(row["chain_id"], "chain_id"),
        _text(row["duty"], "duty"), _text(row["question"], "question"),
        _text(row["split"], "split"), _text(row["owner"], "owner"),
        _text(row["license_id"], "license_id")) for row in rows)
    if (any(case.split != split or case.owner != expected_owner
            or case.license_id != "CC0-1.0" or case.duty not in OPERATION_DUTIES
            for case in cases)
            or len({case.case_id for case in cases}) != len(cases)
            or len({case.question for case in cases}) != len(cases)):
        raise OperationQuerySplitError(f"{split} query owner, license, or identity is invalid")
    return cases


def _read_labels(path: Path, split: str, case_ids: set[str]) -> tuple[dict, ...]:
    rows = _read_jsonl(path, _LABEL_FIELDS)
    expected_policy = LABEL_POLICIES[split]
    seen = set()
    for row in rows:
        case_id = _text(row["case_id"], "label case_id")
        if (case_id not in case_ids or case_id in seen or row["split"] != split
                or row["owner"] != "evaluator" or row["license_id"] != "CC0-1.0"
                or row["policy"] not in expected_policy):
            raise OperationQuerySplitError(f"{split} evaluator labels do not match isolated query ownership")
        seen.add(case_id)
    if seen != case_ids:
        raise OperationQuerySplitError(f"{split} evaluator labels do not exactly cover query cases")
    return rows


def audit_operation_query_splits(
        train_path: str | Path,
        held_out_path: str | Path,
        negative_path: str | Path,
        conflict_path: str | Path,
        held_out_labels_path: str | Path,
        negative_labels_path: str | Path,
        conflict_labels_path: str | Path,
        *, minimum_cases_per_split: int = 20,
        ) -> tuple[dict[str, tuple[OperationQueryCase, ...]], OperationQuerySplitAudit]:
    """独立核四路自然输入/label 隔离；不读取或编译 train facts。"""
    if type(minimum_cases_per_split) is not int or minimum_cases_per_split < 1:
        raise ValueError("minimum_cases_per_split must be a positive integer")
    paths = {
        "train": Path(train_path).resolve(strict=True),
        "held_out": Path(held_out_path).resolve(strict=True),
        "negative": Path(negative_path).resolve(strict=True),
        "conflict": Path(conflict_path).resolve(strict=True),
    }
    if len(set(paths.values())) != len(SPLITS):
        raise OperationQuerySplitError("each query split needs its own physical source file")
    if ("evaluator" not in paths["held_out"].parts
            or "evaluator" not in paths["negative"].parts
            or "evaluator" not in paths["conflict"].parts
            or "evaluator" in paths["train"].parts):
        raise OperationQuerySplitError("evaluation query sources must remain under the evaluator owner")

    courses = {split: _read_split_queries(paths[split], split) for split in SPLITS}
    labels_by_split = {
        "held_out": Path(held_out_labels_path).resolve(strict=True),
        "negative": Path(negative_labels_path).resolve(strict=True),
        "conflict": Path(conflict_labels_path).resolve(strict=True),
    }
    if len(set(labels_by_split.values())) != 3 or any(
            "evaluator" not in path.parts for path in labels_by_split.values()):
        raise OperationQuerySplitError("evaluator label owners must be three separate physical files")
    if set(paths.values()).intersection(labels_by_split.values()):
        raise OperationQuerySplitError("query cases and evaluator labels need separate physical files")

    all_ids: set[str] = set()
    all_questions: set[str] = set()
    chain_owners: dict[str, str] = {}
    counts, duties = {}, {}
    for split in SPLITS:
        cases = courses[split]
        if len(cases) < minimum_cases_per_split:
            raise OperationQuerySplitError(f"{split} has fewer than {minimum_cases_per_split} query cases")
        duties[split] = {case.duty for case in cases}
        counts[split] = len(cases)
        for case in cases:
            if case.case_id in all_ids or case.question in all_questions:
                raise OperationQuerySplitError("case identity or exact natural input crosses query splits")
            all_ids.add(case.case_id)
            all_questions.add(case.question)
            prior = chain_owners.setdefault(case.chain_id, split)
            if prior != split:
                raise OperationQuerySplitError("source/document/dialogue chain crosses query splits")

    all_duties = set(OPERATION_DUTIES)
    if not all_duties <= duties["train"] or not all_duties <= duties["held_out"]:
        raise OperationQuerySplitError("train and held-out inputs must represent all eight actual duties")
    for split, path in labels_by_split.items():
        _read_labels(path, split, {case.case_id for case in courses[split]})

    query_hashes = tuple((split, hashlib.sha256(paths[split].read_bytes()).hexdigest())
                         for split in SPLITS)
    label_hashes = tuple((split, hashlib.sha256(path.read_bytes()).hexdigest())
                         for split, path in sorted(labels_by_split.items()))
    audit = OperationQuerySplitAudit(
        tuple((split, counts[split]) for split in SPLITS), query_hashes, label_hashes,
        len(duties["held_out"]), len(duties["train"]), "PASS_QUERY_SPLIT_STRUCTURE_ONLY")
    return courses, audit
