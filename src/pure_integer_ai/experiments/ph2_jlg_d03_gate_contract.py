"""J-LG-D03 最终发布前合取 artifact 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    PublicFileIdentity,
    inventory_public_files,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_LG_D03_CONJUNCTION_GATE"
ARTIFACT_PATH = "data/ph2/manifests/j_lg_d03_gate_v3.json"
VERDICTS = ("BLOCKED", "NE", "PASS", "REJECT")
MAIN_CONDITION_KEYS = (
    "J-LG-D03-01-SOURCE-EXIT",
    "J-LG-D03-02-GENERATION-GENERALIZATION",
    "J-LG-D03-03-MEMORY-DYNAMICS-DECISION",
    "J-LG-D03-04-PACK-AUDITABILITY",
    "J-LG-D03-05-LEGACY-IDENTITY-CLEAR",
    "J-LG-D03-06-SECRET-CLEAR",
    "J-LG-D03-07-ZERO-FORMAL-EXECUTION",
    "J-LG-D03-08-CAPABILITY-LEDGERS-FROZEN",
    "J-LG-D03-09-CORE-COURSES-FROZEN",
    "J-LG-D03-10-TRANSFER-AXES-FROZEN",
    "J-LG-D03-11-DIRECTIONAL-CONSUMERS-FROZEN",
    "J-LG-D03-12-RETENTION-ROLLBACK-FROZEN",
)
SUPPLEMENTAL_CHECK_KEYS = (
    "SUP-CC-CEDICT-HISTORICAL-BLOCKER-PRESERVED",
    "SUP-NL-00-SCOPE-DECIDED",
    "SUP-PAPER-BYTE-IDENTITY",
    "SUP-RI-00-SCOPE-DECIDED",
    "SUP-WIKTIONARY-DOUBLE-PASS",
)
EXECUTION_STATE_KEYS = (
    "assessment_updates",
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "evaluator_label_writes",
    "formal_training_runs",
    "mastered_claims",
    "memory_learning_writes",
    "readiness_claims",
    "teacher_calls",
    "use_learning_writes",
    "w01_started",
)
PAPER_SHA256 = {
    "paper/main.pdf": (
        "04cfb5d7741117d5888ef8a6018de5de0979f759915b4f863f4df0d77ea04898"),
    "paper/main.tex": (
        "fedde37d06790b919373c23e1bc507275c8fecdcb1150a23d5b20590ef7a15c1"),
}


class JLGD03GateContractError(RuntimeError):
    """最终闸不完整、非规范或越权声明范围。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JLGD03GateContractError(f"{where} must be canonical text")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise JLGD03GateContractError(f"{where} must be a nonnegative int")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise JLGD03GateContractError(f"{where} must be 0 or 1")
    return value


def _digest(value: Any, length: int, *, where: str) -> str:
    text = _text(value, where=where)
    if (len(text) != length
            or any(character not in "0123456789abcdef" for character in text)):
        raise JLGD03GateContractError(f"{where} has an invalid digest")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise JLGD03GateContractError(f"{where} must be a safe relative path")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise JLGD03GateContractError(f"{where} must be a tuple")
    result = tuple(_text(item, where=where) for item in value)
    if not allow_empty and not result:
        raise JLGD03GateContractError(f"{where} cannot be empty")
    if tuple(sorted(set(result))) != result:
        raise JLGD03GateContractError(f"{where} must be sorted and unique")
    return result


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise JLGD03GateContractError(f"{where} fields are not exact")
    return value


def _resolve_under(root: Path, relative_path: str) -> Path:
    path = (root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(root):
        raise JLGD03GateContractError("evidence path escapes its root")
    return path


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError as error:
        raise JLGD03GateContractError("evidence file cannot be read") from error
    return size, digest.hexdigest()


@dataclass(frozen=True, order=True)
class GateEvidenceIdentity:
    """公开仓库工作树外的证据文件身份。"""

    relative_path: str
    scope: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, where="external evidence path"))
        if self.scope not in {"RAW_EVIDENCE", "WORKSPACE_ARTIFACT"}:
            raise JLGD03GateContractError("external evidence scope is invalid")
        _nonnegative(self.size_bytes, where="external evidence size")
        object.__setattr__(self, "sha256", _digest(
            self.sha256, 64, where="external evidence SHA-256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "scope": self.scope,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GateEvidenceIdentity":
        raw = _exact_keys(value, {
            "relative_path", "scope", "sha256", "size_bytes",
        }, where="GateEvidenceIdentity")
        return cls(
            str(raw["relative_path"]), str(raw["scope"]),
            raw["size_bytes"], str(raw["sha256"]),
        )


@dataclass(frozen=True)
class GateCondition:
    """一项有独立直接证据的闸条件及诚实裁决。"""

    condition_key: str
    verdict: str
    statement: str
    evidence_refs: tuple[str, ...]
    facts: CanonicalJsonObject

    def __post_init__(self) -> None:
        _text(self.condition_key, where="condition key")
        if self.verdict not in VERDICTS:
            raise JLGD03GateContractError("condition verdict is invalid")
        _text(self.statement, where="condition statement")
        refs = _strict_text_tuple(self.evidence_refs, where="condition evidence")
        object.__setattr__(self, "evidence_refs", tuple(
            _relative_path(item, where="condition evidence") for item in refs))
        if not isinstance(self.facts, CanonicalJsonObject):
            raise JLGD03GateContractError("condition facts are not canonical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "evidence_refs": list(self.evidence_refs),
            "facts": self.facts.to_value(),
            "statement": self.statement,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GateCondition":
        raw = _exact_keys(value, {
            "condition_key", "evidence_refs", "facts", "statement", "verdict",
        }, where="GateCondition")
        return cls(
            str(raw["condition_key"]), str(raw["verdict"]),
            str(raw["statement"]),
            tuple(str(item) for item in raw["evidence_refs"]),
            CanonicalJsonObject.from_value(raw["facts"]),
        )


@dataclass(frozen=True)
class FinalPublicGate:
    """最终候选清单扫描，规范 artifact 自身按规则排除。"""

    scope_file_count: int
    scanned_text_file_count: int
    legacy_rule_keys: tuple[str, ...]
    legacy_finding_count: int
    secret_rule_keys: tuple[str, ...]
    secret_finding_count: int
    binary_paths: tuple[str, ...]
    unreadable_paths: tuple[str, ...]
    artifact_self_excluded: int
    post_publish_self_scan_required: int
    public_candidate_clear: int

    def __post_init__(self) -> None:
        _nonnegative(self.scope_file_count, where="public scope count")
        _nonnegative(self.scanned_text_file_count, where="public text count")
        _nonnegative(self.legacy_finding_count, where="legacy finding count")
        _nonnegative(self.secret_finding_count, where="secret finding count")
        for name in ("legacy_rule_keys", "secret_rule_keys"):
            object.__setattr__(self, name, _strict_text_tuple(
                getattr(self, name), where=name))
        for name in ("binary_paths", "unreadable_paths"):
            values = _strict_text_tuple(
                getattr(self, name), where=name, allow_empty=True)
            object.__setattr__(self, name, tuple(
                _relative_path(item, where=name) for item in values))
        _flag(self.artifact_self_excluded, where="artifact_self_excluded")
        _flag(
            self.post_publish_self_scan_required,
            where="post_publish_self_scan_required",
        )
        _flag(self.public_candidate_clear, where="public_candidate_clear")
        if (self.scanned_text_file_count + len(self.binary_paths)
                + len(self.unreadable_paths) != self.scope_file_count):
            raise JLGD03GateContractError("public scan scope is incomplete")
        expected_clear = int(not (
            self.legacy_finding_count or self.secret_finding_count
            or self.binary_paths or self.unreadable_paths))
        if self.public_candidate_clear != expected_clear:
            raise JLGD03GateContractError("public scan verdict is dishonest")
        if self.artifact_self_excluded != 1:
            raise JLGD03GateContractError("canonical self exclusion is required")
        if self.post_publish_self_scan_required != 1:
            raise JLGD03GateContractError("post-publication self scan is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_self_excluded": self.artifact_self_excluded,
            "binary_paths": list(self.binary_paths),
            "legacy_finding_count": self.legacy_finding_count,
            "legacy_rule_keys": list(self.legacy_rule_keys),
            "post_publish_self_scan_required": self.post_publish_self_scan_required,
            "public_candidate_clear": self.public_candidate_clear,
            "scanned_text_file_count": self.scanned_text_file_count,
            "scope_file_count": self.scope_file_count,
            "secret_finding_count": self.secret_finding_count,
            "secret_rule_keys": list(self.secret_rule_keys),
            "unreadable_paths": list(self.unreadable_paths),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinalPublicGate":
        raw = _exact_keys(value, {
            "artifact_self_excluded", "binary_paths", "legacy_finding_count",
            "legacy_rule_keys", "post_publish_self_scan_required",
            "public_candidate_clear", "scanned_text_file_count",
            "scope_file_count", "secret_finding_count", "secret_rule_keys",
            "unreadable_paths",
        }, where="FinalPublicGate")
        return cls(
            raw["scope_file_count"], raw["scanned_text_file_count"],
            tuple(str(item) for item in raw["legacy_rule_keys"]),
            raw["legacy_finding_count"],
            tuple(str(item) for item in raw["secret_rule_keys"]),
            raw["secret_finding_count"],
            tuple(str(item) for item in raw["binary_paths"]),
            tuple(str(item) for item in raw["unreadable_paths"]),
            raw["artifact_self_excluded"],
            raw["post_publish_self_scan_required"],
            raw["public_candidate_clear"],
        )


@dataclass(frozen=True)
class JLGD03GateManifest:
    """可审计且只允许后续会话发布 D-03 的决断。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_key: str
    head_sha1: str
    origin_master_sha1: str
    tracked_change_count: int
    staged_change_count: int
    untracked_file_count: int
    inventory_exclusions: tuple[str, ...]
    file_inventory: tuple[PublicFileIdentity, ...]
    paper_files: tuple[PublicFileIdentity, ...]
    external_evidence: tuple[GateEvidenceIdentity, ...]
    final_public_gate: FinalPublicGate
    conditions: tuple[GateCondition, ...]
    supplemental_checks: tuple[GateCondition, ...]
    execution_state: CanonicalJsonObject
    conjunction_passed: int
    d03_release_decision: str
    d03_published: int

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise JLGD03GateContractError("gate format version is invalid")
        _text(self.artifact_version, where="artifact version")
        if self.task_key != "J-LG-D03":
            raise JLGD03GateContractError("task key is not J-LG-D03")
        object.__setattr__(self, "head_sha1", _digest(
            self.head_sha1, 40, where="HEAD"))
        object.__setattr__(self, "origin_master_sha1", _digest(
            self.origin_master_sha1, 40, where="origin/master"))
        if self.head_sha1 != self.origin_master_sha1:
            raise JLGD03GateContractError("HEAD and origin/master disagree")
        for name in (
                "tracked_change_count", "staged_change_count",
                "untracked_file_count"):
            _nonnegative(getattr(self, name), where=name)
        if self.tracked_change_count or self.staged_change_count:
            raise JLGD03GateContractError("tracked or staged changes are present")
        exclusions = _strict_text_tuple(
            self.inventory_exclusions, where="inventory exclusions")
        exclusions = tuple(_relative_path(
            item, where="inventory exclusion") for item in exclusions)
        object.__setattr__(self, "inventory_exclusions", exclusions)
        if exclusions != (ARTIFACT_PATH,):
            raise JLGD03GateContractError("only the gate artifact may be excluded")
        for name in ("file_inventory", "paper_files"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or not all(isinstance(item, PublicFileIdentity)
                               for item in values)):
                raise JLGD03GateContractError(f"{name} is invalid")
            values = tuple(sorted(values, key=lambda item: item.relative_path))
            object.__setattr__(self, name, values)
            paths = tuple(item.relative_path for item in values)
            if len(paths) != len(set(paths)):
                raise JLGD03GateContractError(f"{name} has duplicate paths")
        if len(self.file_inventory) + 1 != self.untracked_file_count:
            raise JLGD03GateContractError("untracked inventory is incomplete")
        paper = {item.relative_path: item.sha256 for item in self.paper_files}
        if paper != PAPER_SHA256:
            raise JLGD03GateContractError("paper byte identity changed")
        if (not isinstance(self.external_evidence, tuple)
                or not all(isinstance(item, GateEvidenceIdentity)
                           for item in self.external_evidence)):
            raise JLGD03GateContractError("external evidence is invalid")
        external = tuple(sorted(self.external_evidence))
        object.__setattr__(self, "external_evidence", external)
        external_paths = tuple(item.relative_path for item in external)
        if len(external_paths) != len(set(external_paths)):
            raise JLGD03GateContractError("external evidence path is duplicated")
        if not isinstance(self.final_public_gate, FinalPublicGate):
            raise JLGD03GateContractError("final public gate is invalid")
        if self.final_public_gate.scope_file_count != len(self.file_inventory):
            raise JLGD03GateContractError("public gate and inventory disagree")
        for name, expected in (
                ("conditions", MAIN_CONDITION_KEYS),
                ("supplemental_checks", SUPPLEMENTAL_CHECK_KEYS)):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or not all(isinstance(item, GateCondition) for item in values)):
                raise JLGD03GateContractError(f"{name} is invalid")
            values = tuple(sorted(values, key=lambda item: item.condition_key))
            object.__setattr__(self, name, values)
            if tuple(item.condition_key for item in values) != expected:
                raise JLGD03GateContractError(f"{name} is incomplete")
        known_refs = {
            item.relative_path for item in self.file_inventory
        } | {
            item.relative_path for item in self.paper_files
        } | set(external_paths)
        for item in (*self.conditions, *self.supplemental_checks):
            if not set(item.evidence_refs).issubset(known_refs):
                raise JLGD03GateContractError(
                    "condition references evidence without a file identity")
        if not isinstance(self.execution_state, CanonicalJsonObject):
            raise JLGD03GateContractError("execution state is invalid")
        execution = self.execution_state.to_value()
        if set(execution) != set(EXECUTION_STATE_KEYS):
            raise JLGD03GateContractError("execution state fields are incomplete")
        if any(value != 0 for value in execution.values()):
            raise JLGD03GateContractError("forbidden execution state is nonzero")
        _flag(self.conjunction_passed, where="conjunction_passed")
        _flag(self.d03_published, where="d03_published")
        if self.d03_published != 0:
            raise JLGD03GateContractError("this artifact cannot publish D-03")
        all_pass = all(
            item.verdict == "PASS"
            for item in (*self.conditions, *self.supplemental_checks))
        expected_pass = int(
            all_pass and self.final_public_gate.public_candidate_clear == 1)
        if self.conjunction_passed != expected_pass:
            raise JLGD03GateContractError("conjunction verdict is dishonest")
        expected_status = "PASS" if expected_pass else "BLOCKED"
        if self.artifact_status != expected_status:
            raise JLGD03GateContractError("artifact status disagrees with conjunction")
        expected_decision = (
            "ALLOW_NEXT_SESSION_TO_PUBLISH_D03"
            if expected_pass else "DO_NOT_PUBLISH_D03")
        if self.d03_release_decision != expected_decision:
            raise JLGD03GateContractError("D-03 release decision is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "conditions": [item.to_dict() for item in self.conditions],
            "conjunction_passed": self.conjunction_passed,
            "d03_published": self.d03_published,
            "d03_release_decision": self.d03_release_decision,
            "execution_state": self.execution_state.to_value(),
            "external_evidence": [
                item.to_dict() for item in self.external_evidence],
            "file_inventory": [item.to_dict() for item in self.file_inventory],
            "final_public_gate": self.final_public_gate.to_dict(),
            "format_version": self.format_version,
            "head_sha1": self.head_sha1,
            "inventory_exclusions": list(self.inventory_exclusions),
            "origin_master_sha1": self.origin_master_sha1,
            "paper_files": [item.to_dict() for item in self.paper_files],
            "staged_change_count": self.staged_change_count,
            "supplemental_checks": [
                item.to_dict() for item in self.supplemental_checks],
            "task_key": self.task_key,
            "tracked_change_count": self.tracked_change_count,
            "untracked_file_count": self.untracked_file_count,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JLGD03GateManifest":
        raw = _exact_keys(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "conditions", "conjunction_passed", "d03_published",
            "d03_release_decision", "execution_state", "external_evidence",
            "file_inventory", "final_public_gate", "format_version",
            "head_sha1", "inventory_exclusions", "origin_master_sha1",
            "paper_files", "staged_change_count", "supplemental_checks",
            "task_key", "tracked_change_count", "untracked_file_count",
        }, where="JLGD03GateManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise JLGD03GateContractError("artifact kind is invalid")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]), str(raw["task_key"]),
            str(raw["head_sha1"]), str(raw["origin_master_sha1"]),
            raw["tracked_change_count"], raw["staged_change_count"],
            raw["untracked_file_count"],
            tuple(str(item) for item in raw["inventory_exclusions"]),
            tuple(PublicFileIdentity.from_dict(item)
                  for item in raw["file_inventory"]),
            tuple(PublicFileIdentity.from_dict(item)
                  for item in raw["paper_files"]),
            tuple(GateEvidenceIdentity.from_dict(item)
                  for item in raw["external_evidence"]),
            FinalPublicGate.from_dict(raw["final_public_gate"]),
            tuple(GateCondition.from_dict(item) for item in raw["conditions"]),
            tuple(GateCondition.from_dict(item)
                  for item in raw["supplemental_checks"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
            raw["conjunction_passed"], str(raw["d03_release_decision"]),
            raw["d03_published"],
        )


def read_jlg_d03_gate_manifest(path: str | Path) -> JLGD03GateManifest:
    """只读恢复规范 gate artifact，不执行训练或 runtime。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise JLGD03GateContractError("gate newline is invalid")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = JLGD03GateManifest.from_dict(value)
    except JLGD03GateContractError:
        raise
    except Exception as error:
        raise JLGD03GateContractError("gate artifact is damaged") from error
    if manifest.canonical_bytes() != payload:
        raise JLGD03GateContractError("gate artifact bytes are non-canonical")
    return manifest


def write_jlg_d03_gate_manifest(
        manifest: JLGD03GateManifest,
        path: str | Path,
        ) -> Path:
    """只发布一次，已存在时只接受逐字节相同 artifact。"""
    if not isinstance(manifest, JLGD03GateManifest):
        raise JLGD03GateContractError("gate manifest type is invalid")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise JLGD03GateContractError("gate artifact already differs")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise JLGD03GateContractError("gate artifact cannot be published") from error
    return target


def verify_jlg_d03_gate_files(
        manifest: JLGD03GateManifest,
        *,
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> None:
    """复核公开文件、论文、原始报告和 pack manifest 的全部身份。"""
    if not isinstance(manifest, JLGD03GateManifest):
        raise JLGD03GateContractError("gate manifest type is invalid")
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    current = inventory_public_files(
        repository,
        tuple(item.relative_path for item in manifest.file_inventory),
    )
    if current != manifest.file_inventory:
        raise JLGD03GateContractError("public file identity changed")
    papers = inventory_public_files(
        repository,
        tuple(item.relative_path for item in manifest.paper_files),
    )
    if papers != manifest.paper_files:
        raise JLGD03GateContractError("paper file identity changed")
    if not _resolve_under(repository, ARTIFACT_PATH).is_file():
        raise JLGD03GateContractError("self-excluded gate artifact is missing")
    for item in manifest.external_evidence:
        path = _resolve_under(workspace, item.relative_path)
        if not path.is_file():
            raise JLGD03GateContractError("external evidence is missing")
        size, digest = _hash_file(path)
        if size != item.size_bytes or digest != item.sha256:
            raise JLGD03GateContractError("external evidence identity changed")


__all__ = [
    "ARTIFACT_PATH",
    "EXECUTION_STATE_KEYS",
    "FinalPublicGate",
    "GateCondition",
    "GateEvidenceIdentity",
    "JLGD03GateContractError",
    "JLGD03GateManifest",
    "MAIN_CONDITION_KEYS",
    "PAPER_SHA256",
    "SUPPLEMENTAL_CHECK_KEYS",
    "read_jlg_d03_gate_manifest",
    "verify_jlg_d03_gate_files",
    "write_jlg_d03_gate_manifest",
]
