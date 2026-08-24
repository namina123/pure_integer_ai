"""公开 24 问广域问答的可读主证据审计。

该切片复用仓库登记的 CC0 开发问题和真实 K 盘 20k SQLite。它不把
``ANSWER`` 状态码当作成功，而是逐题检查主证据是否包含预声明事实词、
是否仍被 Category/表格残片占据，并重复一次核对确定性。它是开发回归，
不是 held-out 或正式通用问答评测。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    BroadQaResult,
    query_broad_qa,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


DEV_AUDIT_PROTOCOL_V1 = 1
DEV_AUDIT_PASS = "PASS"
DEV_AUDIT_FAIL = "FAIL"
DEV_AUDIT_NE = "NE"
_TRACE_DOMAIN = "pure_integer_ai.broad_qa.dev.surface.audit.v1"
_QUESTIONS_PATH = Path("data/ph2/broad_qa_dev_questions_v1.json")
_QUESTIONS_SHA256 = (
    "fe1b5f8ca9ce9904936442604e7c5a901f1ecebc4781c5f19e9c08392958ad93")

# Each inner tuple is an OR group; every group must have one matching surface.
_EVIDENCE_GROUPS: tuple[tuple[tuple[str, ...], ...], ...] = (
    (("2,928米", "2,928 米"),),
    (("35公里",),), (("5座",),),
    (("大写模式", "大寫模式"),), (("14公里",),),
    (("2,100米", "2,100 米"), ("4,000米", "4,000 米")),
    (("粤式粥点", "粵式粥點"),), (("银牌", "銀牌"),),
    (("2012年3月31日",),), (("355米",),),
    (("维基媒体基金会", "維基媒體基金會"),), (("CC0",),),
    (("1340年6月24日",),), (("每年7月第二个周末",),),
    (("6,000 光年", "6,000光年"),), (("爱德华·斯特凡", "愛德華·斯特凡"),),
    (("罗斯福", "羅斯福"),), (("4公里",),),
    (("80万", "80萬"), ("100万人", "100萬人")),
    (("外交",), ("贸易", "貿易"), ("国际人道主义救援", "國際人道主義救援")),
    (("福建",), ("贵州", "貴州")), (("机组人员", "機組人員"),),
    (), (("艾希莉·辛普森", "艾希莉·辛普森"),),
)
_EXPECTED_STATUS = tuple(
    "UNKNOWN" if index == 22 else "ANSWER" for index in range(24))


class BroadQaDevAuditError(ValueError):
    """开发问题 artifact、K 盘边界或报告合同无效。"""


def _text(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise BroadQaDevAuditError(f"{where} 必须是规范字符串")
    if not allow_empty and not value:
        raise BroadQaDevAuditError(f"{where} 不能为空")
    if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
        raise BroadQaDevAuditError(f"{where} 含非 Unicode scalar")
    return value


def _load_questions(project_root: str | Path) -> tuple[str, ...]:
    path = Path(project_root).resolve() / _QUESTIONS_PATH
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaDevAuditError("公开广域问题 artifact 不可回读") from error
    if (hashlib.sha256(payload).hexdigest() != _QUESTIONS_SHA256
            or canonical_json_bytes(value) + b"\n" != payload):
        raise BroadQaDevAuditError("公开广域问题 artifact SHA/规范字节漂移")
    if (not isinstance(value, dict)
            or value.get("artifact_kind") != "PH2_BROAD_QA_DEV_QUESTIONS_V1"
            or value.get("format_version") != 1
            or value.get("license_id") != "CC0-1.0"
            or value.get("scope") != "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT"
            or not isinstance(value.get("questions"), list)
            or len(value["questions"]) != 24
            or len(set(value["questions"])) != 24):
        raise BroadQaDevAuditError("公开广域问题 artifact envelope 漂移")
    questions = tuple(_text(item, "question") for item in value["questions"])
    return questions


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BroadQaDevObservation:
    ordinal: int
    question: str
    status: str
    title: str | None
    primary_evidence: str
    answer: str
    expected_status: str
    evidence_hit: int
    primary_surface_clean: int

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer or None,
            "evidence_hit": bool(self.evidence_hit),
            "expected_status": self.expected_status,
            "ordinal": self.ordinal,
            "primary_evidence": self.primary_evidence or None,
            "primary_surface_clean": bool(self.primary_surface_clean),
            "question": self.question,
            "status": self.status,
            "title": self.title,
        }

    def canonical_record(self) -> tuple[int, ...]:
        result = [DEV_AUDIT_PROTOCOL_V1, self.ordinal, self.evidence_hit,
                  self.primary_surface_clean]
        for value in (self.question, self.status, self.title or "",
                      self.primary_evidence, self.answer,
                      self.expected_status):
            scalars = tuple(ord(item) for item in value)
            result.extend((len(scalars), *scalars))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class BroadQaDevAuditReport:
    status: str
    database_name: str
    database_sha256: str
    question_artifact_sha256: str
    question_count: int
    answer_count: int
    unknown_count: int
    clarify_count: int
    evidence_expected_count: int
    evidence_hit_count: int
    primary_surface_clean_count: int
    long_answer_count: int
    replay_bit_identical: bool
    observations: tuple[BroadQaDevObservation, ...]
    trace: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_count": self.answer_count,
            "clarify_count": self.clarify_count,
            "database_name": self.database_name,
            "database_sha256": self.database_sha256,
            "evidence_expected_count": self.evidence_expected_count,
            "evidence_hit_count": self.evidence_hit_count,
            "format_version": DEV_AUDIT_PROTOCOL_V1,
            "long_answer_count": self.long_answer_count,
            "observations": [item.to_dict() for item in self.observations],
            "primary_surface_clean_count": self.primary_surface_clean_count,
            "question_artifact_sha256": self.question_artifact_sha256,
            "question_count": self.question_count,
            "replay_bit_identical": self.replay_bit_identical,
            "status": self.status,
            "trace_u": list(self.trace),
            "unknown_count": self.unknown_count,
        }


def _primary(result: BroadQaResult) -> str:
    chain = result.evidence_chain
    return (chain[0].selected_text if chain and chain[0].selected_text
            else (result.evidence_text or ""))


def _observe(ordinal: int, question: str, result: BroadQaResult,
             expected_status: str,
             expected_groups: tuple[tuple[str, ...], ...],
             ) -> BroadQaObservation:
    primary = _primary(result)
    hit = int(
        bool(expected_groups)
        and result.status == expected_status
        and all(any(option in primary for option in group)
                for group in expected_groups))
    clean = int(
        result.status == "ANSWER"
        and bool(primary)
        and not primary.lstrip().startswith(("Category:", "category:", "{|")))
    return BroadQaDevObservation(
        ordinal, question, result.status, result.title, primary,
        result.answer or "", expected_status, hit, clean)


def build_broad_qa_dev_audit(*, project_root: str | Path,
                             database_path: str | Path,
                             ) -> BroadQaDevAuditReport:
    """运行公开 24 问开发问题两次并比较完整纯值观察。"""
    root = Path(project_root).resolve()
    database = Path(database_path).resolve()
    if database.drive.upper() != "K:" or not database.is_file():
        raise BroadQaDevAuditError("database 必须是存在的 K 盘文件")
    questions = _load_questions(root)
    if len(_EVIDENCE_GROUPS) != len(questions):
        raise BroadQaDevAuditError("evidence inventory 与问题数不一致")
    first_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        first_results = tuple(query_broad_qa(first_connection, item)
                              for item in questions)
    finally:
        first_connection.close()
    replay_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        replay_results = tuple(query_broad_qa(replay_connection, item)
                               for item in questions)
    finally:
        replay_connection.close()
    observations = tuple(
        _observe(index, question, result, _EXPECTED_STATUS[index],
                 _EVIDENCE_GROUPS[index])
        for index, (question, result) in enumerate(
            zip(questions, first_results)))
    replay_observations = tuple(
        _observe(index, question, result, _EXPECTED_STATUS[index],
                 _EVIDENCE_GROUPS[index])
        for index, (question, result) in enumerate(
            zip(questions, replay_results)))
    replay_identical = observations == replay_observations
    expected_count = sum(bool(item) for item in _EVIDENCE_GROUPS)
    evidence_hits = sum(item.evidence_hit for item in observations)
    clean_count = sum(item.primary_surface_clean for item in observations)
    answers = sum(item.status == "ANSWER" for item in observations)
    unknown = sum(item.status == "UNKNOWN" for item in observations)
    clarify = sum(item.status == "CLARIFY" for item in observations)
    long_answers = sum(
        len(item.answer.encode("utf-8")) >= 48
        for item in observations if item.status == "ANSWER")
    passed = (
        len(observations) == 24
        and tuple(item.status for item in observations) == _EXPECTED_STATUS
        and evidence_hits == expected_count
        and clean_count == answers
        and replay_identical
    )
    status = DEV_AUDIT_PASS if passed else DEV_AUDIT_FAIL
    trace_values = [DEV_AUDIT_PROTOCOL_V1, len(observations), answers,
                    unknown, clarify, expected_count, evidence_hits,
                    clean_count, int(replay_identical)]
    for item in observations:
        trace_values.extend(item.canonical_record())
    trace = integer_tuple_fingerprint(
        tuple(trace_values), domain=_TRACE_DOMAIN)
    return BroadQaDevAuditReport(
        status, database.name, _sha256_path(database), _QUESTIONS_SHA256,
        len(observations), answers, unknown, clarify, expected_count,
        evidence_hits, clean_count, long_answers, replay_identical,
        observations, trace,
    )


def write_broad_qa_dev_audit(report: BroadQaDevAuditReport,
                             output_path: str | Path) -> str:
    output = Path(output_path).resolve()
    if output.drive.upper() != "K:" or output.exists():
        raise ValueError("broad QA audit output 必须是不存在的 K 盘文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":")) + "\n",
                       encoding="utf-8")
    return str(output)


__all__ = [
    "BroadQaDevAuditError", "BroadQaDevAuditReport",
    "BroadQaDevObservation", "DEV_AUDIT_FAIL", "DEV_AUDIT_NE",
    "DEV_AUDIT_PASS", "build_broad_qa_dev_audit",
    "write_broad_qa_dev_audit",
]
