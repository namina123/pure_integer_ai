"""来源内归纳 review decision ledger 与最终三层 family 冻结合同。

ledger 必须逐项覆盖已冻结 roster。抽取项需要终页逐字支持；可推导项需要已通过
的 source-inference record 承诺；冲突项需要两个不同来源承诺；其余只能拒绝。
本模块不代替 reviewer 作语义判断，也不会自动把来源缺失映射为可推导或冲突。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_DECISIONS,
    SOURCE_INFERENCE_ROSTER_KIND,
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    parse_broad_qa_source_inference_record,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_review import (
    SOURCE_INFERENCE_DOSSIER_KIND,
    SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


SOURCE_INFERENCE_DECISION_LEDGER_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_DECISION_LEDGER_V1")
SOURCE_INFERENCE_DECISION_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_DECISION_V1")
SOURCE_INFERENCE_REVIEW_WORKSHEET_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_WORKSHEET_V1")
SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_V1")
SOURCE_INFERENCE_DEVELOPMENT_PACK_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_DEVELOPMENT_PACK_V1")
SOURCE_INFERENCE_DEVELOPMENT_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_DEVELOPMENT_RECORD_V1")
SOURCE_INFERENCE_REVIEW_INPUT_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_INPUT_V1")
SOURCE_INFERENCE_RATIONALE_CODES = (
    "EXACT_TERMINAL_SUPPORT",
    "AUDITABLE_SOURCE_DERIVATION",
    "DISTINCT_SOURCE_ASSERTION_CONFLICT",
    "INSUFFICIENT_SOURCE_SUPPORT",
)
_DECISION_RATIONALE = {
    "EXTRACTIVE": "EXACT_TERMINAL_SUPPORT",
    "SOURCE_DERIVABLE": "AUDITABLE_SOURCE_DERIVATION",
    "SOURCE_CONFLICT": "DISTINCT_SOURCE_ASSERTION_CONFLICT",
    "REJECT": "INSUFFICIENT_SOURCE_SUPPORT",
}


def _sha256(value: object, *, label: str) -> str:
    """要求值是小写规范 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    """流式计算 ledger 输入或输出文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    """要求 JSON object 字段集合精确匹配。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise BroadQaExternalDataError(f"{label} 字段漂移")
    return value


def _read_manifest(
        path: Path,
        *,
        artifact_kind: str,
        status: str,
        ) -> dict[str, object]:
    """严格回读规范 manifest 及其冻结状态。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("source inference manifest 不可读") from error
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or value.get("artifact_kind") != artifact_kind
            or value.get("format_version") != 1
            or value.get("status") != status):
        raise BroadQaExternalDataError("source inference manifest 漂移")
    return value


def _read_jsonl(
        path: Path,
        *,
        record_kind: str,
        ) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL 并要求 item identity 唯一。"""
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                item_id = value.get("item_id") if isinstance(value, dict) else None
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or value.get("record_kind") != record_kind
                        or not isinstance(item_id, str) or not item_id
                        or item_id in identities):
                    raise BroadQaExternalDataError(
                        f"source inference JSONL 漂移: {line_number}")
                identities.add(item_id)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("source inference JSONL 非法") from error
    if not values:
        raise BroadQaExternalDataError("source inference JSONL 为空")
    return tuple(values)


def _read_roster(path: Path) -> tuple[dict[str, object], ...]:
    """回读机械 roster 的最小决策坐标。"""
    values = _read_jsonl(
        path, record_kind=SOURCE_INFERENCE_ROSTER_RECORD_KIND)
    expected = {
        "assignment", "format_version", "item_id", "question_sha256",
        "record_kind", "source_alignment_status", "source_key",
        "terminal_page_id", "terminal_revision_id", "title_key",
    }
    if any(set(value) != expected or value["format_version"] != 1
           for value in values):
        raise BroadQaExternalDataError("source inference roster 字段漂移")
    return values


def _read_dossier(path: Path) -> tuple[dict[str, object], ...]:
    """回读完整 dossier 并核验来源承诺所需字段存在。"""
    values = _read_jsonl(
        path, record_kind=SOURCE_INFERENCE_DOSSIER_RECORD_KIND)
    expected = {
        "assignment", "format_version", "item_id", "record_kind",
        "review_source", "roster_commitment", "terminal_source",
    }
    for value in values:
        if set(value) != expected or value["format_version"] != 1:
            raise BroadQaExternalDataError("source inference dossier 字段漂移")
        review = value["review_source"]
        terminal = value["terminal_source"]
        if (not isinstance(review, dict) or not isinstance(terminal, dict)
                or not isinstance(review.get("context_sha256"), str)
                or not isinstance(terminal.get("wikitext_sha256"), str)
                or not isinstance(terminal.get("plain_text"), str)
                or not isinstance(terminal.get("snapshot_id"), str)
                or not isinstance(terminal.get("license_id"), str)
                or type(terminal.get("page_id")) is not int
                or type(terminal.get("revision_id")) is not int
                or not isinstance(terminal.get("title"), str)
                or not isinstance(terminal.get("passages"), list)
                or not isinstance(review.get("gold_answers"), list)):
            raise BroadQaExternalDataError(
                "source inference dossier 来源承诺缺失")
        required_passage = {
            "ordinal", "raw_end", "raw_sha256", "raw_start",
            "section_title", "text", "text_sha256",
        }
        if any(not isinstance(item, dict) or set(item) != required_passage
               for item in terminal["passages"]):
            raise BroadQaExternalDataError(
                "source inference dossier passage 承诺缺失")
    return values


def _gold_answer_sha256s(gold_answers: tuple[str, ...]) -> tuple[str, ...]:
    """返回固定 gold 文本集合的规范 SHA 承诺。"""
    return tuple(sorted({
        hashlib.sha256(item.encode("utf-8")).hexdigest()
        for item in gold_answers
    }))


def _validate_inference_record_binding(
        record,
        *,
        item_id: str,
        roster_record: dict[str, object],
        dossier_record: dict[str, object],
        gold_answers: tuple[str, ...],
        ) -> None:
    """核验 inference record 只使用当前题目的固定终页来源。"""
    terminal = dossier_record["terminal_source"]
    if (record.item_id != item_id
            or record.question_sha256 != roster_record["question_sha256"]
            or record.gold_answer_sha256s
            != _gold_answer_sha256s(gold_answers)
            or record.terminal_wikitext_sha256
            != terminal["wikitext_sha256"]):
        raise BroadQaExternalDataError(
            "SOURCE_DERIVABLE inference record 题目承诺漂移")
    rendered = normalize_external_text(record.claim.rendered_text)
    if not any(
            rendered == normalize_external_text(answer)
            for answer in gold_answers):
        raise BroadQaExternalDataError(
            "SOURCE_DERIVABLE inference record 未派生目标 gold")
    passages = {
        (
            value["ordinal"], value["raw_start"], value["raw_end"],
            value["raw_sha256"], value["text"],
        )
        for value in terminal["passages"]
    }
    for premise in record.claim.premises:
        observation = premise.observation
        passage = (
            observation.passage_ordinal,
            observation.raw_start,
            observation.raw_end,
            observation.raw_sha256,
            observation.evidence_text,
        )
        if (observation.page_id != roster_record["terminal_page_id"]
                or observation.revision_id
                != roster_record["terminal_revision_id"]
                or observation.title != terminal["title"]
                or observation.snapshot_id != terminal["snapshot_id"]
                or observation.license_id != terminal["license_id"]
                or passage not in passages):
            raise BroadQaExternalDataError(
                "SOURCE_DERIVABLE inference record 终页证据漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceDecision:
    """一个 reviewer 对固定 roster item 的四态、来源化裁决。"""

    item_id: str
    decision: str
    rationale_code: str
    source_commitment_sha256s: tuple[str, ...]
    inference_record_sha256: str | None
    reviewer_note_sha256: str

    def __post_init__(self) -> None:
        """核验决策、理由、来源承诺和 inference record 条件。"""
        _sha256(self.item_id, label="decision item_id")
        if (self.decision not in SOURCE_INFERENCE_DECISIONS
                or self.rationale_code not in SOURCE_INFERENCE_RATIONALE_CODES
                or self.rationale_code != _DECISION_RATIONALE[self.decision]):
            raise BroadQaExternalDataError("source inference decision/rationale 漂移")
        if (not isinstance(self.source_commitment_sha256s, tuple)
                or not self.source_commitment_sha256s
                or self.source_commitment_sha256s
                != tuple(sorted(set(self.source_commitment_sha256s)))):
            raise BroadQaExternalDataError(
                "source inference 来源承诺必须非空唯一排序")
        for item in self.source_commitment_sha256s:
            _sha256(item, label="decision source commitment")
        if self.decision == "SOURCE_CONFLICT":
            if len(self.source_commitment_sha256s) < 2:
                raise BroadQaExternalDataError(
                    "SOURCE_CONFLICT 必须保留至少两个不同来源承诺")
        elif len(self.source_commitment_sha256s) != 1:
            raise BroadQaExternalDataError(
                "非冲突 decision 必须绑定唯一主来源承诺")
        if self.decision == "SOURCE_DERIVABLE":
            _sha256(
                self.inference_record_sha256,
                label="decision inference record")
        elif self.inference_record_sha256 is not None:
            raise BroadQaExternalDataError(
                "非 SOURCE_DERIVABLE 不得携带 inference record")
        _sha256(self.reviewer_note_sha256, label="decision reviewer note")

    def to_dict(self) -> dict[str, object]:
        """导出不泄露 reviewer 文本的规范决策记录。"""
        return {
            "decision": self.decision,
            "format_version": 1,
            "inference_record_sha256": self.inference_record_sha256,
            "item_id": self.item_id,
            "rationale_code": self.rationale_code,
            "record_kind": SOURCE_INFERENCE_DECISION_RECORD_KIND,
            "reviewer_note_sha256": self.reviewer_note_sha256,
            "source_commitment_sha256s": list(
                self.source_commitment_sha256s),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceInferenceDecision":
        """从字段精确的 JSON object 恢复决策。"""
        raw = _exact(value, {
            "decision", "format_version", "inference_record_sha256",
            "item_id", "rationale_code", "record_kind",
            "reviewer_note_sha256", "source_commitment_sha256s",
        }, label="source inference decision")
        if (raw["format_version"] != 1
                or raw["record_kind"] != SOURCE_INFERENCE_DECISION_RECORD_KIND
                or not isinstance(raw["source_commitment_sha256s"], list)):
            raise BroadQaExternalDataError(
                "source inference decision envelope 漂移")
        return cls(
            raw["item_id"], raw["decision"], raw["rationale_code"],
            tuple(raw["source_commitment_sha256s"]),
            raw["inference_record_sha256"], raw["reviewer_note_sha256"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceDecisionLedger:
    """绑定 roster、dossier 和精确全覆盖决策的规范 ledger。"""

    roster_sha256: str
    dossier_sha256: str
    decisions: tuple[BroadQaSourceInferenceDecision, ...]
    status: str = "FROZEN_REVIEWED_NOT_RUN"

    def __post_init__(self) -> None:
        """核验输入承诺、决策排序、唯一性和冻结状态。"""
        _sha256(self.roster_sha256, label="ledger roster")
        _sha256(self.dossier_sha256, label="ledger dossier")
        if (not isinstance(self.decisions, tuple) or not self.decisions
                or any(not isinstance(item, BroadQaSourceInferenceDecision)
                       for item in self.decisions)
                or self.decisions != tuple(sorted(
                    self.decisions, key=lambda item: item.item_id))
                or len({item.item_id for item in self.decisions})
                != len(self.decisions)):
            raise BroadQaExternalDataError(
                "source inference ledger decisions 未唯一规范排序")
        if self.status != "FROZEN_REVIEWED_NOT_RUN":
            raise BroadQaExternalDataError("source inference ledger 状态漂移")

    def to_dict(self) -> dict[str, object]:
        """导出 ledger envelope、分账和全部决策。"""
        counts = Counter(item.decision for item in self.decisions)
        return {
            "artifact_kind": SOURCE_INFERENCE_DECISION_LEDGER_KIND,
            "decision_counts": {
                key: counts[key] for key in SOURCE_INFERENCE_DECISIONS},
            "decisions": [item.to_dict() for item in self.decisions],
            "dossier_sha256": self.dossier_sha256,
            "format_version": 1,
            "item_count": len(self.decisions),
            "roster_sha256": self.roster_sha256,
            "status": self.status,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 ledger 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回规范 ledger SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> "BroadQaSourceInferenceDecisionLedger":
        """从字段精确的 JSON object 恢复 ledger。"""
        raw = _exact(value, {
            "artifact_kind", "decision_counts", "decisions",
            "dossier_sha256", "format_version", "item_count",
            "roster_sha256", "status",
        }, label="source inference decision ledger")
        if (raw["artifact_kind"] != SOURCE_INFERENCE_DECISION_LEDGER_KIND
                or raw["format_version"] != 1
                or not isinstance(raw["decisions"], list)):
            raise BroadQaExternalDataError(
                "source inference decision ledger envelope 漂移")
        ledger = cls(
            raw["roster_sha256"], raw["dossier_sha256"],
            tuple(BroadQaSourceInferenceDecision.from_dict(item)
                  for item in raw["decisions"]),
            raw["status"],
        )
        if ledger.to_dict() != raw:
            raise BroadQaExternalDataError(
                "source inference decision ledger 分账漂移")
        return ledger


def parse_source_inference_decision_ledger(
        payload: bytes,
        ) -> BroadQaSourceInferenceDecisionLedger:
    """严格回读单行规范 ledger，拒绝字段和字节漂移。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError(
            "source inference decision ledger 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "source inference decision ledger JSON 非规范") from error
    ledger = BroadQaSourceInferenceDecisionLedger.from_dict(value)
    if ledger.canonical_bytes() != payload:
        raise BroadQaExternalDataError(
            "source inference decision ledger 字节漂移")
    return ledger


def validate_source_inference_decision_ledger(
        ledger: BroadQaSourceInferenceDecisionLedger,
        *,
        roster_path: str | Path,
        dossier_path: str | Path,
        inference_record_paths: Iterable[str | Path] = (),
        ) -> dict[str, object]:
    """对固定 roster/dossier 核验全覆盖、assignment 和来源承诺。"""
    if not isinstance(ledger, BroadQaSourceInferenceDecisionLedger):
        raise TypeError("source inference ledger 类型非法")
    roster_file = Path(roster_path).resolve()
    dossier_file = Path(dossier_path).resolve()
    if (ledger.roster_sha256 != _sha256_file(roster_file)
            or ledger.dossier_sha256 != _sha256_file(dossier_file)):
        raise BroadQaExternalDataError(
            "source inference ledger 输入 commitment 漂移")
    roster = _read_roster(roster_file)
    dossier = _read_dossier(dossier_file)
    roster_by_id = {str(item["item_id"]): item for item in roster}
    dossier_by_id = {str(item["item_id"]): item for item in dossier}
    decision_by_id = {item.item_id: item for item in ledger.decisions}
    if set(roster_by_id) != set(dossier_by_id) or set(roster_by_id) != set(decision_by_id):
        raise BroadQaExternalDataError(
            "source inference ledger 未精确覆盖固定 roster")

    verified_inference_records = {}
    for raw_path in inference_record_paths:
        path = Path(raw_path).resolve()
        try:
            payload = path.read_bytes()
            record = parse_broad_qa_source_inference_record(payload)
        except (OSError, ValueError) as error:
            raise BroadQaExternalDataError(
                "source inference record 不可验证") from error
        digest = hashlib.sha256(payload).hexdigest()
        if digest in verified_inference_records:
            raise BroadQaExternalDataError(
                "source inference record identity 重复")
        verified_inference_records[digest] = record

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item_id, decision in decision_by_id.items():
        roster_record = roster_by_id[item_id]
        dossier_record = dossier_by_id[item_id]
        if (dossier_record["assignment"] != roster_record["assignment"]
                or dossier_record["roster_commitment"]["title_key"]
                != roster_record["title_key"]):
            raise BroadQaExternalDataError(
                "source inference dossier/roster binding 漂移")
        review = dossier_record["review_source"]
        terminal = dossier_record["terminal_source"]
        terminal_sha = _sha256(
            terminal["wikitext_sha256"], label="terminal source commitment")
        review_sha = _sha256(
            review["context_sha256"], label="review source commitment")
        gold_answers = tuple(str(item) for item in review["gold_answers"])
        terminal_text = normalize_external_text(str(terminal["plain_text"]))
        terminal_hit = any(
            normalize_external_text(answer) in terminal_text
            for answer in gold_answers)
        if decision.decision == "EXTRACTIVE":
            if (roster_record["assignment"] != "EXTRACTIVE_CANDIDATE"
                    or not terminal_hit
                    or decision.source_commitment_sha256s != (terminal_sha,)):
                raise BroadQaExternalDataError(
                    "EXTRACTIVE decision 缺少终页逐字支持")
        elif decision.decision == "SOURCE_DERIVABLE":
            if (roster_record["assignment"] != "NON_EXTRACTIVE_REVIEW"
                    or terminal_hit
                    or decision.source_commitment_sha256s != (terminal_sha,)
                    or decision.inference_record_sha256
                    not in verified_inference_records):
                raise BroadQaExternalDataError(
                    "SOURCE_DERIVABLE decision 来源边界漂移")
            _validate_inference_record_binding(
                verified_inference_records[
                    decision.inference_record_sha256],
                item_id=item_id,
                roster_record=roster_record,
                dossier_record=dossier_record,
                gold_answers=gold_answers,
            )
        elif decision.decision == "SOURCE_CONFLICT":
            if (roster_record["assignment"] != "NON_EXTRACTIVE_REVIEW"
                    or terminal_hit
                    or decision.source_commitment_sha256s
                    != tuple(sorted((review_sha, terminal_sha)))):
                raise BroadQaExternalDataError(
                    "SOURCE_CONFLICT decision 来源边界漂移")
        else:
            expected = terminal_sha if terminal_sha else review_sha
            if decision.source_commitment_sha256s != (expected,):
                raise BroadQaExternalDataError(
                    "REJECT decision 主来源承诺漂移")
        source_counts[str(roster_record["source_key"])][decision.decision] += 1
    counts = Counter(item.decision for item in ledger.decisions)
    return {
        "decision_counts": {
            key: counts[key] for key in SOURCE_INFERENCE_DECISIONS},
        "item_count": len(ledger.decisions),
        "ledger_sha256": ledger.sha256(),
        "verified_inference_record_count": len(verified_inference_records),
        "source_decision_counts": {
            source: {decision: values[decision]
                     for decision in SOURCE_INFERENCE_DECISIONS}
            for source, values in sorted(source_counts.items())
        },
        "status": "VALIDATED_REVIEWED_NOT_RUN",
    }


def _read_review_input(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读 reviewer 写入的四态输入，不接受自动语义标签。"""
    values = _read_jsonl(
        path, record_kind=SOURCE_INFERENCE_REVIEW_INPUT_KIND)
    expected = {
        "decision", "format_version", "inference_record_path", "item_id",
        "record_kind", "reviewer_note",
    }
    for value in values:
        if (set(value) != expected or value["format_version"] != 1
                or value["decision"] not in SOURCE_INFERENCE_DECISIONS
                or not isinstance(value["reviewer_note"], str)
                or not value["reviewer_note"].strip()
                or value["reviewer_note"].strip()
                != value["reviewer_note"]
                or (value["inference_record_path"] is not None
                    and not isinstance(
                        value["inference_record_path"], str))):
            raise BroadQaExternalDataError(
                "source inference review input 字段漂移")
        _sha256(value["item_id"], label="review input item_id")
        if ((value["decision"] == "SOURCE_DERIVABLE")
                != (value["inference_record_path"] is not None)):
            raise BroadQaExternalDataError(
                "source inference review input record path 漂移")
    return values


def compile_source_inference_decision_ledger(
        *,
        roster_path: str | Path,
        dossier_path: str | Path,
        review_input_path: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """把人工四态输入编译为规范 ledger、校验报告和 SHA manifest。"""
    roster_file = Path(roster_path).resolve()
    dossier_file = Path(dossier_path).resolve()
    input_file = Path(review_input_path).resolve()
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference decision target 已存在")
    roster = _read_roster(roster_file)
    dossier = _read_dossier(dossier_file)
    review_input = _read_review_input(input_file)
    roster_by_id = {str(item["item_id"]): item for item in roster}
    dossier_by_id = {str(item["item_id"]): item for item in dossier}
    input_by_id = {str(item["item_id"]): item for item in review_input}
    if (len(input_by_id) != len(review_input)
            or set(roster_by_id) != set(dossier_by_id)
            or set(roster_by_id) != set(input_by_id)):
        raise BroadQaExternalDataError(
            "source inference review input 未精确覆盖固定 roster")

    decisions = []
    inference_paths = []
    for item_id in sorted(roster_by_id):
        dossier_record = dossier_by_id[item_id]
        review = dossier_record["review_source"]
        terminal = dossier_record["terminal_source"]
        value = input_by_id[item_id]
        decision = str(value["decision"])
        terminal_sha = _sha256(
            terminal["wikitext_sha256"], label="terminal commitment")
        if decision == "SOURCE_CONFLICT":
            source_sha256s = tuple(sorted((
                _sha256(
                    review["context_sha256"], label="review commitment"),
                terminal_sha,
            )))
        else:
            source_sha256s = (terminal_sha,)
        inference_sha = None
        if decision == "SOURCE_DERIVABLE":
            raw_path = Path(str(value["inference_record_path"]))
            path = (raw_path if raw_path.is_absolute()
                    else (input_file.parent / raw_path)).resolve()
            if not path.is_file():
                raise BroadQaExternalDataError(
                    "source inference review input record 不存在")
            inference_sha = _sha256_file(path)
            inference_paths.append(path)
        note_sha = hashlib.sha256(
            str(value["reviewer_note"]).encode("utf-8")).hexdigest()
        decisions.append(BroadQaSourceInferenceDecision(
            item_id,
            decision,
            _DECISION_RATIONALE[decision],
            source_sha256s,
            inference_sha,
            note_sha,
        ))
    ledger = BroadQaSourceInferenceDecisionLedger(
        _sha256_file(roster_file),
        _sha256_file(dossier_file),
        tuple(decisions),
    )
    validation = validate_source_inference_decision_ledger(
        ledger,
        roster_path=roster_file,
        dossier_path=dossier_file,
        inference_record_paths=inference_paths,
    )
    target.mkdir(parents=True)
    ledger_path = target / "decision.ledger.json"
    report_path = target / "validation.report.json"
    manifest_path = target / "manifest.json"
    ledger_path.write_bytes(ledger.canonical_bytes())
    report_path.write_bytes(canonical_json_line(validation))
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_DECISION_LEDGER_KIND,
        "decision_counts": validation["decision_counts"],
        "dossier_sha256": _sha256_file(dossier_file),
        "format_version": 1,
        "inference_record_count": len(inference_paths),
        "item_count": len(decisions),
        "ledger_sha256": _sha256_file(ledger_path),
        "review_input_sha256": _sha256_file(input_file),
        "roster_sha256": _sha256_file(roster_file),
        "status": "VALIDATED_REVIEWED_NOT_RUN",
        "validation_report_sha256": _sha256_file(report_path),
    }
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def publish_source_inference_review_worksheet(
        dossier_path: str | Path,
        *,
        target_path: str | Path,
        context_radius: int = 120,
        ) -> dict[str, object]:
    """为全部固定 item 发布无决策 worksheet，便于逐项审阅。"""
    dossier_file = Path(dossier_path).resolve()
    target = Path(target_path).resolve()
    if (target.exists() or type(context_radius) is not int
            or not 32 <= context_radius <= 1000):
        raise BroadQaExternalDataError(
            "source inference worksheet 输出或预算非法")
    dossier = _read_dossier(dossier_file)
    records = []
    for value in dossier:
        review = value["review_source"]
        terminal = value["terminal_source"]
        context = str(review["context"])
        snippets = []
        for answer in review["gold_answers"]:
            answer = str(answer)
            start = context.find(answer)
            if start < 0:
                continue
            snippets.append(context[
                max(0, start - context_radius):
                min(len(context), start + len(answer) + context_radius)
            ])
        records.append({
            "allowed_decisions": list(SOURCE_INFERENCE_DECISIONS),
            "assignment": value["assignment"],
            "decision": "UNREVIEWED",
            "format_version": 1,
            "gold_answers": review["gold_answers"],
            "item_id": value["item_id"],
            "old_context_answer_snippets": snippets,
            "question": review["question"],
            "record_kind": SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND,
            "review_context_sha256": review["context_sha256"],
            "terminal_passages": terminal["passages"],
            "terminal_title": terminal["title"],
            "terminal_wikitext_sha256": terminal["wikitext_sha256"],
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_REVIEW_WORKSHEET_KIND,
        "dossier_sha256": _sha256_file(dossier_file),
        "format_version": 1,
        "item_count": len(records),
        "review_decisions_written": 0,
        "status": "UNREVIEWED_NOT_A_DECISION_LEDGER",
        "worksheet_bytes": target.stat().st_size,
        "worksheet_sha256": _sha256_file(target),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def _balanced_decision_selection(
        values: tuple[dict[str, object], ...],
        *,
        quota: int,
        ) -> tuple[dict[str, object], ...]:
    """按来源等额、item SHA 稳定选择一个已审决策层。"""
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for value in values:
        by_source[str(value["source_key"])].append(value)
    sources = tuple(sorted(by_source))
    if type(quota) is not int or quota <= 0 or not sources:
        raise BroadQaExternalDataError("source inference stratum quota 非法")
    base, remainder = divmod(quota, len(sources))
    selected = []
    for index, source in enumerate(sources):
        source_quota = base + int(index < remainder)
        candidates = sorted(by_source[source], key=lambda item: item["item_id"])
        if len(candidates) < source_quota:
            raise BroadQaExternalDataError(
                f"source inference stratum 来源库存不足: {source}")
        selected.extend(candidates[:source_quota])
    return tuple(selected)


def freeze_source_inference_development_pack(
        ledger: BroadQaSourceInferenceDecisionLedger,
        *,
        roster_path: str | Path,
        dossier_path: str | Path,
        target_dir: str | Path,
        stratum_quota: int,
        inference_record_paths: Iterable[str | Path] = (),
        ) -> dict[str, object]:
    """从已验证 ledger 等额冻结三层 development family，不包含 REJECT。"""
    validation = validate_source_inference_decision_ledger(
        ledger,
        roster_path=roster_path,
        dossier_path=dossier_path,
        inference_record_paths=inference_record_paths,
    )
    roster = _read_roster(Path(roster_path).resolve())
    dossier = _read_dossier(Path(dossier_path).resolve())
    roster_by_id = {str(item["item_id"]): item for item in roster}
    dossier_by_id = {str(item["item_id"]): item for item in dossier}
    decision_by_id = {item.item_id: item for item in ledger.decisions}
    selected = []
    for stratum in ("EXTRACTIVE", "SOURCE_DERIVABLE", "SOURCE_CONFLICT"):
        available = tuple({
            **roster_by_id[item_id],
            "decision": decision.to_dict(),
        } for item_id, decision in decision_by_id.items()
          if decision.decision == stratum)
        selected.extend(_balanced_decision_selection(
            available, quota=stratum_quota))
    if len({item["title_key"] for item in selected}) != len(selected):
        raise BroadQaExternalDataError(
            "source inference development pack 标题重复")
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference development target 已存在")
    target.mkdir(parents=True)
    records = []
    for value in sorted(selected, key=lambda item: (
            item["decision"]["decision"], item["item_id"])):
        dossier_record = dossier_by_id[str(value["item_id"])]
        records.append({
            "decision": value["decision"],
            "format_version": 1,
            "item_id": value["item_id"],
            "question": dossier_record["review_source"]["question"],
            "record_kind": SOURCE_INFERENCE_DEVELOPMENT_RECORD_KIND,
            "source_key": value["source_key"],
            "stratum": value["decision"]["decision"],
            "title_key": value["title_key"],
        })
    records_path = target / "development.records.jsonl"
    with records_path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
    counts = Counter(item["stratum"] for item in records)
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_DEVELOPMENT_PACK_KIND,
        "development_records_sha256": _sha256_file(records_path),
        "format_version": 1,
        "ledger_sha256": ledger.sha256(),
        "question_count": len(records),
        "selection_rule": (
            "VALIDATED_DECISION_THEN_SOURCE_BALANCE_THEN_ITEM_SHA256_V1"),
        "status": "FROZEN_NOT_RUN",
        "stratum_counts": {
            key: counts[key]
            for key in ("EXTRACTIVE", "SOURCE_DERIVABLE", "SOURCE_CONFLICT")},
        "stratum_quota": stratum_quota,
        "title_count": len(records),
        "validation": validation,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


__all__ = [
    "BroadQaSourceInferenceDecision",
    "BroadQaSourceInferenceDecisionLedger",
    "SOURCE_INFERENCE_DECISION_LEDGER_KIND",
    "SOURCE_INFERENCE_DECISION_RECORD_KIND",
    "SOURCE_INFERENCE_REVIEW_INPUT_KIND",
    "compile_source_inference_decision_ledger",
    "SOURCE_INFERENCE_DEVELOPMENT_PACK_KIND",
    "SOURCE_INFERENCE_RATIONALE_CODES",
    "SOURCE_INFERENCE_REVIEW_WORKSHEET_KIND",
    "freeze_source_inference_development_pack",
    "parse_source_inference_decision_ledger",
    "publish_source_inference_review_worksheet",
    "validate_source_inference_decision_ledger",
]
