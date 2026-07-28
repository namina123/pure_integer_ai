"""P3-Ia 对 LC-07/09/13/15 与能力基线的不可覆盖修订合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
LEDGER_KEYS = ("LC-00", "LC-07", "LC-09", "LC-13", "LC-15")
P3IA_COURSE_STATUS = "COURSE_FROZEN"
P3IA_PRODUCTION_CONTRACT_STATUS = "CONTRACT_READY"
P3IB_STATUS = "NE"
P3IB_PHASE = "PH3"
FORMAL_RUNTIME_STATUS = "NOT_STARTED"
FOCUSED_RUNTIME_EVIDENCE = "PASS"
EXECUTION_STATE = {
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}
INVARIANTS = {
    "base_artifact_preserved": 1,
    "cross_language_pass_forbidden": 1,
    "course_frozen_is_not_mastered": 1,
    "formal_training_not_executed": 1,
    "p3ib_deferred_to_ph3": 1,
    "runtime_test_is_not_training": 1,
}
ARTIFACT_KINDS = {
    "LC-00": "PH2_LANGUAGE_CAPABILITY_BASELINE_P3IA_REVISION",
    "LC-07": "PH2_LC07_DISCOURSE_INFORMATION_P3IA_REVISION",
    "LC-09": "PH2_LC09_TRANSFER_AXIS_P3IA_REVISION",
    "LC-13": "PH2_LC13_DIRECTIONAL_CONSUMER_P3IA_REVISION",
    "LC-15": "PH2_LC15_LEARNING_OBJECTIVE_P3IA_REVISION",
}
ARTIFACT_STATUSES = {
    "LC-00": "BASELINE_FROZEN",
    "LC-07": "COURSE_FROZEN",
    "LC-09": "CONTRACT_FROZEN",
    "LC-13": "CONTRACT_FROZEN",
    "LC-15": "COURSE_FROZEN",
}
ARTIFACT_VERSIONS = {
    "LC-00": "LG-LC-MD-GG-baseline-v40-supersedes-v39",
    "LC-07": "LC-07-discourse-information-course-v2-supersedes-v1",
    "LC-09": "LC-09-transfer-axis-manifest-v2-supersedes-v1",
    "LC-13": "LC-13-directional-consumer-manifest-v2-supersedes-v1",
    "LC-15": "LC-15-final-learning-objectives-v2-supersedes-v1",
}
LEDGER_FACTS = {
    "LC-00": {
        "bound_ledger_revisions": ["LC-07", "LC-09", "LC-13", "LC-15"],
        "p3ia_capability_state": "CONTRACT_READY",
        "p3ib_capability_state": "NE",
        "p3ib_phase": "PH3",
    },
    "LC-07": {
        "base_course_preserved": 1,
        "p3ia_private_label_isolation_required": 1,
        "p3ia_raw_hierarchy_course_bound": 1,
        "p3ia_same_language_recall_course_bound": 1,
    },
    "LC-09": {
        "base_pack_inventory_count": 16,
        "code_switch_transfer_claim_state": "NE",
        "cross_language_transfer_claim_state": "NE",
        "current_pack_inventory_count": 17,
        "p3ia_pack_registered": 1,
        "p3ia_transfer_claim_state": "NE",
    },
    "LC-13": {
        "base_route_count": 60,
        "consumer_direction": "UNDERSTANDING",
        "consumer_status": "CONTRACT_READY",
        "formal_directional_verdict": "NE",
        "p3ia_consumer_bound": 1,
    },
    "LC-15": {
        "ablation_results_observed": 0,
        "base_course_source_count": 9,
        "candidate_eliminations_executed": 0,
        "current_course_source_count": 10,
        "p3ia_objective_course_bound": 1,
    },
}
EVIDENCE_ROOTS = ("REPOSITORY", "WORKSPACE")
EVIDENCE_ROLES = (
    "COURSE",
    "EVALUATOR",
    "PACK",
    "RUNTIME",
    "SAMPLE",
    "SUPERSEDED_LEDGER",
    "TEST",
    "UPSTREAM_REVISION",
)


class P3IaLedgerRevisionError(RuntimeError):
    """P3-Ia 修订范围、证据身份或诚实状态不闭合。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise P3IaLedgerRevisionError(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise P3IaLedgerRevisionError(f"{where} 必须是安全相对路径")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise P3IaLedgerRevisionError(f"{where} 必须是 SHA-256")
    return text


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise P3IaLedgerRevisionError(f"{where} 必须是非负严格整数")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise P3IaLedgerRevisionError(f"{where} 字段不精确")
    return value


@dataclass(frozen=True)
class LedgerEvidenceFile:
    """一个仓库或工作区相对的不可变证据文件身份。"""

    root_key: str
    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if self.root_key not in EVIDENCE_ROOTS:
            raise P3IaLedgerRevisionError("evidence root_key 未登记")
        _relative_path(self.relative_path, where="evidence relative_path")
        if self.role not in EVIDENCE_ROLES:
            raise P3IaLedgerRevisionError("evidence role 未登记")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise P3IaLedgerRevisionError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    @property
    def identity_key(self) -> str:
        return f"{self.root_key}/{self.relative_path}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "root_key": self.root_key,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LedgerEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "root_key", "sha256",
        }, where="LedgerEvidenceFile")
        return cls(
            str(raw["root_key"]), str(raw["relative_path"]),
            str(raw["role"]), raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True)
class P3IaLedgerRevision:
    """一个只追加的 P3-Ia 账目修订，不扩张为训练或迁移 PASS。"""

    format_version: int
    ledger_key: str
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    supersedes_relative_path: str
    supersedes_sha256: str
    p3ia_course_status: str
    p3ia_production_contract_status: str
    p3ib_status: str
    p3ib_phase: str
    language_scope: tuple[str, ...]
    code_switch_status: str
    cross_language_pass_authority: int
    formal_runtime_status: str
    focused_runtime_evidence: str
    ledger_facts: CanonicalJsonObject
    evidence_files: tuple[LedgerEvidenceFile, ...]
    invariants: CanonicalJsonObject
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise P3IaLedgerRevisionError("format_version 非法")
        if self.ledger_key not in LEDGER_KEYS:
            raise P3IaLedgerRevisionError("ledger_key 未登记")
        if self.artifact_kind != ARTIFACT_KINDS[self.ledger_key]:
            raise P3IaLedgerRevisionError("artifact_kind 与 ledger 不匹配")
        if self.artifact_version != ARTIFACT_VERSIONS[self.ledger_key]:
            raise P3IaLedgerRevisionError("artifact_version 与 ledger 不匹配")
        if self.artifact_status != ARTIFACT_STATUSES[self.ledger_key]:
            raise P3IaLedgerRevisionError("artifact_status 与 ledger 不匹配")
        _relative_path(
            self.supersedes_relative_path, where="supersedes_relative_path")
        _sha256(self.supersedes_sha256, where="supersedes_sha256")
        if self.p3ia_course_status != P3IA_COURSE_STATUS:
            raise P3IaLedgerRevisionError("P3-Ia course status 不诚实")
        if self.p3ia_production_contract_status != P3IA_PRODUCTION_CONTRACT_STATUS:
            raise P3IaLedgerRevisionError("P3-Ia production status 不诚实")
        if self.p3ib_status != P3IB_STATUS or self.p3ib_phase != P3IB_PHASE:
            raise P3IaLedgerRevisionError("P3-Ib 必须保持 NE/PH3")
        if self.language_scope != ("zh",):
            raise P3IaLedgerRevisionError("R-01 语言范围只能是 zh")
        if self.code_switch_status != "NE":
            raise P3IaLedgerRevisionError("code-switch 必须保持 NE")
        if self.cross_language_pass_authority != 0:
            raise P3IaLedgerRevisionError("不得签发跨语言 PASS")
        if self.formal_runtime_status != FORMAL_RUNTIME_STATUS:
            raise P3IaLedgerRevisionError("正式 runtime 不得冒充已启动")
        if self.focused_runtime_evidence != FOCUSED_RUNTIME_EVIDENCE:
            raise P3IaLedgerRevisionError("聚焦 runtime 证据状态非法")
        if (not isinstance(self.ledger_facts, CanonicalJsonObject)
                or self.ledger_facts.to_value() != LEDGER_FACTS[self.ledger_key]):
            raise P3IaLedgerRevisionError("ledger_facts 漂移")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or not all(isinstance(item, LedgerEvidenceFile)
                           for item in self.evidence_files)):
            raise P3IaLedgerRevisionError("evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.root_key, item.relative_path, item.role)))
        object.__setattr__(self, "evidence_files", evidence)
        identities = tuple(item.identity_key for item in evidence)
        if len(identities) != len(set(identities)):
            raise P3IaLedgerRevisionError("evidence 文件身份重复")
        superseded = tuple(
            item for item in evidence if item.role == "SUPERSEDED_LEDGER")
        if (len(superseded) != 1
                or superseded[0].root_key != "REPOSITORY"
                or superseded[0].relative_path != self.supersedes_relative_path
                or superseded[0].sha256 != self.supersedes_sha256):
            raise P3IaLedgerRevisionError("supersedes 证据未闭合")
        required_roles = {
            "COURSE", "EVALUATOR", "PACK", "RUNTIME", "SAMPLE",
            "SUPERSEDED_LEDGER", "TEST",
        }
        if not required_roles.issubset({item.role for item in evidence}):
            raise P3IaLedgerRevisionError("P3-Ia 文件级证据角色未闭合")
        if self.ledger_key == "LC-00":
            upstream = tuple(
                item for item in evidence if item.role == "UPSTREAM_REVISION")
            if len(upstream) != 4:
                raise P3IaLedgerRevisionError("baseline 未绑定四个 LC 修订")
        if (not isinstance(self.invariants, CanonicalJsonObject)
                or self.invariants.to_value() != INVARIANTS):
            raise P3IaLedgerRevisionError("invariants 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise P3IaLedgerRevisionError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "code_switch_status": self.code_switch_status,
            "cross_language_pass_authority": self.cross_language_pass_authority,
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "focused_runtime_evidence": self.focused_runtime_evidence,
            "formal_runtime_status": self.formal_runtime_status,
            "format_version": self.format_version,
            "invariants": self.invariants.to_value(),
            "language_scope": list(self.language_scope),
            "ledger_facts": self.ledger_facts.to_value(),
            "ledger_key": self.ledger_key,
            "p3ia_course_status": self.p3ia_course_status,
            "p3ia_production_contract_status": (
                self.p3ia_production_contract_status),
            "p3ib_phase": self.p3ib_phase,
            "p3ib_status": self.p3ib_status,
            "supersedes_relative_path": self.supersedes_relative_path,
            "supersedes_sha256": self.supersedes_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "P3IaLedgerRevision":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "code_switch_status", "cross_language_pass_authority",
            "evidence_files", "execution_state", "focused_runtime_evidence",
            "formal_runtime_status", "format_version", "invariants",
            "language_scope", "ledger_facts", "ledger_key",
            "p3ia_course_status", "p3ia_production_contract_status",
            "p3ib_phase", "p3ib_status", "supersedes_relative_path",
            "supersedes_sha256",
        }, where="P3IaLedgerRevision")
        return cls(
            raw["format_version"], str(raw["ledger_key"]),
            str(raw["artifact_kind"]), str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            str(raw["supersedes_relative_path"]),
            str(raw["supersedes_sha256"]),
            str(raw["p3ia_course_status"]),
            str(raw["p3ia_production_contract_status"]),
            str(raw["p3ib_status"]), str(raw["p3ib_phase"]),
            tuple(str(item) for item in raw["language_scope"]),
            str(raw["code_switch_status"]),
            raw["cross_language_pass_authority"],
            str(raw["formal_runtime_status"]),
            str(raw["focused_runtime_evidence"]),
            CanonicalJsonObject.from_value(raw["ledger_facts"]),
            tuple(LedgerEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
            CanonicalJsonObject.from_value(raw["invariants"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )


def read_p3ia_ledger_revision(path: str | Path) -> P3IaLedgerRevision:
    """严格回读规范修订 artifact。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise P3IaLedgerRevisionError("revision newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        revision = P3IaLedgerRevision.from_dict(value)
    except P3IaLedgerRevisionError:
        raise
    except Exception as error:
        raise P3IaLedgerRevisionError("revision artifact 损坏") from error
    if revision.canonical_bytes() != payload:
        raise P3IaLedgerRevisionError("revision artifact 非规范字节")
    return revision


def write_p3ia_ledger_revision(
        revision: P3IaLedgerRevision,
        path: str | Path,
        ) -> Path:
    """独占或幂等写修订 artifact，禁止同版本覆盖。"""
    if not isinstance(revision, P3IaLedgerRevision):
        raise P3IaLedgerRevisionError("revision 类型非法")
    target = Path(path)
    payload = revision.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise P3IaLedgerRevisionError("revision 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise P3IaLedgerRevisionError("revision 无法写入") from error
    return target


def verify_p3ia_ledger_revision_files(
        revision: P3IaLedgerRevision,
        *,
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> None:
    """逐字节回验修订绑定的仓库与外部 pack 文件。"""
    roots = {
        "REPOSITORY": Path(repository_root).resolve(),
        "WORKSPACE": Path(workspace_root).resolve(),
    }
    for item in revision.evidence_files:
        root = roots[item.root_key]
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise P3IaLedgerRevisionError("evidence 路径逃逸") from error
        if not path.is_file():
            raise P3IaLedgerRevisionError("evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise P3IaLedgerRevisionError("evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_STATUSES",
    "ARTIFACT_VERSIONS",
    "EXECUTION_STATE",
    "INVARIANTS",
    "LEDGER_FACTS",
    "LEDGER_KEYS",
    "LedgerEvidenceFile",
    "P3IaLedgerRevision",
    "P3IaLedgerRevisionError",
    "read_p3ia_ledger_revision",
    "verify_p3ia_ledger_revision_files",
    "write_p3ia_ledger_revision",
]
