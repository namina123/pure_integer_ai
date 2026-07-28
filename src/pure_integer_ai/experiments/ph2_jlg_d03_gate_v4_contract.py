"""J-LG-D03 v4 补充吸收后的最终合取 gate 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_capability_baseline_v41_contract import (
    VERSION_KEYS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_jlg_d03_gate_contract import (
    FinalPublicGate,
    GateCondition,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    PublicFileIdentity,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_LG_D03_CONJUNCTION_GATE_V4"
ARTIFACT_VERSION = "J-LG-D03-prepublication-gate-v4-supersedes-v3"
ARTIFACT_PATH = "data/ph2/manifests/j_lg_d03_gate_v4.json"
V3_PATH = "data/ph2/manifests/j_lg_d03_gate_v3.json"
V3_SHA256 = "9de926e95d9386e782816a705e3525c74fcaf38b80f98126534d757956b5bad6"
PAPER_SHA256 = {
    "paper/main.pdf": (
        "04cfb5d7741117d5888ef8a6018de5de0979f759915b4f863f4df0d77ea04898"),
    "paper/main.tex": (
        "fedde37d06790b919373c23e1bc507275c8fecdcb1150a23d5b20590ef7a15c1"),
}
CONDITION_KEYS = (
    "J-LG-D03-V4-01-P3IA-COURSE-CONSUMER",
    "J-LG-D03-V4-02-STORAGE-ABSORPTION",
    "J-LG-D03-V4-03-RECOVERY-ABSORPTION",
    "J-LG-D03-V4-04-AUTHORIZED-GENERATION",
    "J-LG-D03-V4-05-TYPED-PROOF-FAMILIES",
    "J-LG-D03-V4-06-LONG-CONTEXT",
    "J-LG-D03-V4-07-D02-MD-GG-BASELINE",
    "J-LG-D03-V4-08-VERSION-KEYS",
    "J-LG-D03-V4-09-FILE-DEPENDENCY-GRAPH",
    "J-LG-D03-V4-10-P3IB-NE-PH3",
    "J-LG-D03-V4-11-PUBLIC-SCAN",
    "J-LG-D03-V4-12-SOURCE-LICENSE",
    "J-LG-D03-V4-13-PAPER-BYTES",
    "J-LG-D03-V4-14-ZERO-EXECUTION",
    "J-LG-D03-V4-15-V3-PRESERVED",
    "J-LG-D03-V4-16-D03-HOLD",
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
REQUIRED_NODE_SPECS = {
    "BASELINE_V40": (
        "data/ph2/manifests/language_capability_baseline_v40.json",
        "BASELINE_FROZEN"),
    "BASELINE_V41": (
        "data/ph2/manifests/language_capability_baseline_v41.json",
        "BASELINE_FROZEN"),
    "D02_SOURCE": (
        "data/ph2/manifests/d02_source_pack_coverage_v1.json",
        "6_PACK_FROZEN_1_BLOCKED"),
    "GG01": (
        "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
        "CONTRACT_FROZEN"),
    "GG02": (
        "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
        "BRIDGE_FROZEN"),
    "GG03": (
        "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
        "COURSE_FROZEN"),
    "LC07_V2": (
        "data/ph2/manifests/lc07_discourse_information_course_v2.json",
        "COURSE_FROZEN"),
    "LC09_V2": (
        "data/ph2/manifests/lc09_transfer_axis_manifest_v2.json",
        "CONTRACT_FROZEN"),
    "LC13_V2": (
        "data/ph2/manifests/lc13_directional_consumer_manifest_v2.json",
        "CONTRACT_FROZEN"),
    "LC15_V2": (
        "data/ph2/manifests/lc15_final_learning_objectives_v2.json",
        "COURSE_FROZEN"),
    "MD01": (
        "data/ph2/manifests/md01_memory_dynamics_contract_v1.json",
        "CONTRACT_FROZEN"),
    "MD02": (
        "data/ph2/manifests/md02_situation_state_adapter_v1.json",
        "ADAPTER_FROZEN"),
    "MD03": (
        "data/ph2/manifests/md03_directional_center_adapter_v1.json",
        "ADAPTER_FROZEN"),
    "MD04_PLAN": (
        "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
        "PREREGISTERED"),
    "MD04_RUNS": (
        "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
        "RESULTS_OBSERVED"),
    "MD05": (
        "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
        "PASS"),
    "P3IA_COURSE": (
        "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json",
        "COURSE_FROZEN"),
    "R02": (
        "data/ph2/manifests/r02_storage_absorption_v1.json",
        "PRODUCTION_EVIDENCED"),
    "R03": (
        "data/ph2/manifests/r03_correction_recovery_absorption_v1.json",
        "PRODUCTION_EVIDENCED"),
    "R04": (
        "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json",
        "PRODUCTION_EVIDENCED"),
    "R05": (
        "data/ph2/manifests/r05_typed_proof_family_absorption_v1.json",
        "PRODUCTION_EVIDENCED"),
    "R06": (
        "data/ph2/manifests/r06_long_context_absorption_v1.json",
        "PRODUCTION_EVIDENCED"),
    "V3_GATE": (V3_PATH, "PASS"),
}
_V41_DEPENDENCIES = tuple(sorted(
    key for key in REQUIRED_NODE_SPECS if key not in {"BASELINE_V41", "V3_GATE"}))
REQUIRED_EDGE_PAIRS = tuple(sorted((
    *(("BASELINE_V41", key) for key in _V41_DEPENDENCIES),
    ("GATE_V4", "BASELINE_V41"),
    ("GATE_V4", "R02"),
    ("GATE_V4", "R03"),
    ("GATE_V4", "R04"),
    ("GATE_V4", "R05"),
    ("GATE_V4", "R06"),
    ("GATE_V4", "V3_GATE"),
)))


class JLGD03GateV4Error(RuntimeError):
    """v4 gate 合取、文件图、扫描或停止边界不闭合。"""


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise JLGD03GateV4Error(f"{where} 必须是非空规范文本")
    return value


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise JLGD03GateV4Error(f"{where} 必须是安全相对路径")
    return text


def _digest(value: Any, length: int, *, where: str) -> str:
    text = _text(value, where=where)
    if (len(text) != length
            or any(item not in "0123456789abcdef" for item in text)):
        raise JLGD03GateV4Error(f"{where} digest 非法")
    return text


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise JLGD03GateV4Error(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise JLGD03GateV4Error(f"{where} 必须是 0/1")
    return value


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise JLGD03GateV4Error(f"{where} 字段不精确")
    return value


@dataclass(frozen=True, order=True)
class GateV4DependencyNode:
    """一个有字节身份和诚实状态的 v4 依赖 artifact 节点。"""

    node_key: str
    relative_path: str
    status: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _text(self.node_key, where="dependency node key")
        _relative_path(self.relative_path, where="dependency path")
        _text(self.status, where="dependency status")
        if _nonnegative(self.byte_count, where="dependency byte_count") == 0:
            raise JLGD03GateV4Error("dependency artifact 不得为空")
        _digest(self.sha256, 64, where="dependency sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "node_key": self.node_key,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GateV4DependencyNode":
        raw = _exact(value, {
            "byte_count", "node_key", "relative_path", "sha256", "status",
        }, where="GateV4DependencyNode")
        return cls(
            str(raw["node_key"]), str(raw["relative_path"]),
            str(raw["status"]), raw["byte_count"], str(raw["sha256"]))


@dataclass(frozen=True, order=True)
class GateV4DependencyEdge:
    """consumer 到直接 dependency 的显式有向边。"""

    consumer_key: str
    dependency_key: str

    def __post_init__(self) -> None:
        _text(self.consumer_key, where="edge consumer")
        _text(self.dependency_key, where="edge dependency")
        if self.consumer_key == self.dependency_key:
            raise JLGD03GateV4Error("dependency edge 不得自环")

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer_key": self.consumer_key,
            "dependency_key": self.dependency_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GateV4DependencyEdge":
        raw = _exact(value, {
            "consumer_key", "dependency_key",
        }, where="GateV4DependencyEdge")
        return cls(str(raw["consumer_key"]), str(raw["dependency_key"]))


@dataclass(frozen=True)
class JLGD03GateV4Manifest:
    """当前工作树和全部补充吸收证据的不可覆盖 v4 合取。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    task_key: str
    head_sha1: str
    origin_master_sha1: str
    tracked_change_count: int
    staged_change_count: int
    untracked_file_count: int
    candidate_file_count: int
    candidate_paths: tuple[str, ...]
    inventory_exclusions: tuple[str, ...]
    file_inventory: tuple[PublicFileIdentity, ...]
    paper_files: tuple[PublicFileIdentity, ...]
    dependency_nodes: tuple[GateV4DependencyNode, ...]
    dependency_edges: tuple[GateV4DependencyEdge, ...]
    version_keys: CanonicalJsonObject
    final_public_gate: FinalPublicGate
    conditions: tuple[GateCondition, ...]
    execution_state: CanonicalJsonObject
    conjunction_passed: int
    d03_release_decision: str
    d03_published: int

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise JLGD03GateV4Error("format_version 非法")
        if self.artifact_kind != ARTIFACT_KIND:
            raise JLGD03GateV4Error("artifact_kind 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise JLGD03GateV4Error("artifact_version 非法")
        if self.task_key != "J-LG-D03":
            raise JLGD03GateV4Error("task_key 非法")
        _digest(self.head_sha1, 40, where="HEAD")
        _digest(self.origin_master_sha1, 40, where="origin/master")
        if self.head_sha1 != self.origin_master_sha1:
            raise JLGD03GateV4Error("HEAD 与 origin/master 不一致")
        for name in (
                "tracked_change_count", "staged_change_count",
                "untracked_file_count", "candidate_file_count"):
            _nonnegative(getattr(self, name), where=name)
        if self.staged_change_count != 0:
            raise JLGD03GateV4Error("staged change 必须为零")
        if self.candidate_file_count != (
                self.tracked_change_count + self.untracked_file_count):
            raise JLGD03GateV4Error("candidate 文件计数不闭合")
        exclusions = tuple(sorted(_relative_path(
            item, where="inventory exclusion")
                                  for item in self.inventory_exclusions))
        object.__setattr__(self, "inventory_exclusions", exclusions)
        if exclusions != (ARTIFACT_PATH,):
            raise JLGD03GateV4Error("只允许排除 v4 自身")
        candidate_paths = tuple(sorted(_relative_path(
            item, where="candidate path") for item in self.candidate_paths))
        object.__setattr__(self, "candidate_paths", candidate_paths)
        if (len(candidate_paths) != len(set(candidate_paths))
                or ARTIFACT_PATH in candidate_paths
                or len(candidate_paths) + 1 != self.candidate_file_count):
            raise JLGD03GateV4Error("candidate paths 不完整")
        for name in ("file_inventory", "paper_files"):
            values = getattr(self, name)
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(item, PublicFileIdentity)
                           for item in values)):
                raise JLGD03GateV4Error(f"{name} 非法")
            values = tuple(sorted(values, key=lambda item: item.relative_path))
            object.__setattr__(self, name, values)
            paths = tuple(item.relative_path for item in values)
            if len(paths) != len(set(paths)):
                raise JLGD03GateV4Error(f"{name} 路径重复")
        if ARTIFACT_PATH in {
                item.relative_path for item in self.file_inventory}:
            raise JLGD03GateV4Error("v4 自身不得进入哈希 inventory")
        if not set(candidate_paths).issubset({
                item.relative_path for item in self.file_inventory}):
            raise JLGD03GateV4Error("candidate inventory 不完整")
        paper = {item.relative_path: item.sha256 for item in self.paper_files}
        if paper != PAPER_SHA256:
            raise JLGD03GateV4Error("paper byte identity 漂移")

        if (not isinstance(self.dependency_nodes, tuple)
                or any(not isinstance(item, GateV4DependencyNode)
                       for item in self.dependency_nodes)):
            raise JLGD03GateV4Error("dependency_nodes 非法")
        nodes = tuple(sorted(self.dependency_nodes))
        object.__setattr__(self, "dependency_nodes", nodes)
        node_specs = {
            item.node_key: (item.relative_path, item.status) for item in nodes}
        if node_specs != REQUIRED_NODE_SPECS:
            raise JLGD03GateV4Error("dependency node 集合不完整")
        if len(nodes) != len({item.relative_path for item in nodes}):
            raise JLGD03GateV4Error("dependency path 重复")
        inventory_paths = {item.relative_path for item in self.file_inventory}
        if not {item.relative_path for item in nodes}.issubset(inventory_paths):
            raise JLGD03GateV4Error("dependency 未进入文件 inventory")

        if (not isinstance(self.dependency_edges, tuple)
                or any(not isinstance(item, GateV4DependencyEdge)
                       for item in self.dependency_edges)):
            raise JLGD03GateV4Error("dependency_edges 非法")
        edges = tuple(sorted(self.dependency_edges))
        object.__setattr__(self, "dependency_edges", edges)
        pairs = tuple((item.consumer_key, item.dependency_key) for item in edges)
        if pairs != REQUIRED_EDGE_PAIRS:
            raise JLGD03GateV4Error("dependency edge 集合不完整")
        known_consumers = {"GATE_V4", *node_specs}
        if any(item.consumer_key not in known_consumers
               or item.dependency_key not in node_specs for item in edges):
            raise JLGD03GateV4Error("dependency edge 引用未知节点")

        if (not isinstance(self.version_keys, CanonicalJsonObject)
                or self.version_keys.to_value() != VERSION_KEYS):
            raise JLGD03GateV4Error("version keys 漂移")
        if not isinstance(self.final_public_gate, FinalPublicGate):
            raise JLGD03GateV4Error("final public gate 非法")
        if self.final_public_gate.scope_file_count != len(self.file_inventory):
            raise JLGD03GateV4Error("public scan 与 inventory 不一致")
        if (not isinstance(self.conditions, tuple)
                or any(not isinstance(item, GateCondition)
                       for item in self.conditions)):
            raise JLGD03GateV4Error("conditions 非法")
        conditions = tuple(sorted(
            self.conditions, key=lambda item: item.condition_key))
        object.__setattr__(self, "conditions", conditions)
        if tuple(item.condition_key for item in conditions) != CONDITION_KEYS:
            raise JLGD03GateV4Error("conditions 不完整")
        known_refs = inventory_paths | set(PAPER_SHA256)
        if any(not set(item.evidence_refs).issubset(known_refs)
               for item in conditions):
            raise JLGD03GateV4Error("condition 引用无文件身份证据")
        rows = {item.condition_key: item.facts.to_value() for item in conditions}
        self._verify_direct_facts(rows)

        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or set(self.execution_state.to_value())
                != set(EXECUTION_STATE_KEYS)
                or any(value != 0
                       for value in self.execution_state.to_value().values())):
            raise JLGD03GateV4Error("execution state 非零或不完整")
        _flag(self.conjunction_passed, where="conjunction_passed")
        _flag(self.d03_published, where="d03_published")
        if self.d03_published != 0:
            raise JLGD03GateV4Error("v4 不得发布 D-03")
        expected_pass = int(
            all(item.verdict == "PASS" for item in conditions)
            and self.final_public_gate.public_candidate_clear == 1)
        if self.conjunction_passed != expected_pass:
            raise JLGD03GateV4Error("conjunction verdict 不诚实")
        if self.artifact_status != ("PASS" if expected_pass else "BLOCKED"):
            raise JLGD03GateV4Error("artifact_status 与合取不一致")
        expected_decision = (
            "ALLOW_FUTURE_CONFIRMED_SESSION_TO_PUBLISH_D03"
            if expected_pass else "DO_NOT_PUBLISH_D03")
        if self.d03_release_decision != expected_decision:
            raise JLGD03GateV4Error("D-03 release decision 不诚实")

    @staticmethod
    def _verify_direct_facts(rows: dict[str, Any]) -> None:
        if rows[CONDITION_KEYS[0]] != {
                "consumer_status": "CONTRACT_READY",
                "course_status": "COURSE_FROZEN",
                "focused_runtime_evidence": "PASS",
                "formal_runtime_status": "NOT_STARTED",
                "label_owner_isolated": 1,
            }:
            raise JLGD03GateV4Error("R-01 direct facts 漂移")
        for index, expected in (
                (1, "PRODUCTION_EVIDENCED"),
                (2, "PRODUCTION_EVIDENCED"),
                (3, "PRODUCTION_EVIDENCED"),
                (4, "PRODUCTION_EVIDENCED"),
                (5, "PRODUCTION_EVIDENCED")):
            if rows[CONDITION_KEYS[index]].get("artifact_status") != expected:
                raise JLGD03GateV4Error(f"R-0{index + 1} direct facts 漂移")
        if rows[CONDITION_KEYS[4]].get("proof_family_count") != 5:
            raise JLGD03GateV4Error("R-05 proof family 数量漂移")
        if rows[CONDITION_KEYS[5]].get("mechanism_count") != 3:
            raise JLGD03GateV4Error("R-06 mechanism 数量漂移")
        if rows[CONDITION_KEYS[6]] != {
                "baseline_version": "LG-LC-MD-GG-baseline-v41-supersedes-v40",
                "d02_source_entry_count": 7,
                "gg03_course_status": "COURSE_FROZEN",
                "md05_decision": "PASS",
            }:
            raise JLGD03GateV4Error("D-02/MD/GG baseline facts 漂移")
        if rows[CONDITION_KEYS[7]] != VERSION_KEYS:
            raise JLGD03GateV4Error("condition version keys 漂移")
        graph = rows[CONDITION_KEYS[8]]
        if not (
                graph.get("dependency_node_count") == len(REQUIRED_NODE_SPECS)
                and graph.get("dependency_edge_count") == len(REQUIRED_EDGE_PAIRS)
                and graph.get("file_identity_complete") == 1):
            raise JLGD03GateV4Error("file dependency graph facts 漂移")
        if rows[CONDITION_KEYS[9]] != {
                "code_switch_status": "NE",
                "cross_language_pass_authority": 0,
                "p3ib_phase": "PH3",
                "p3ib_status": "NE",
            }:
            raise JLGD03GateV4Error("P3-Ib facts 漂移")
        public = rows[CONDITION_KEYS[10]]
        if not (
                public.get("artifact_self_excluded") == 1
                and public.get("post_publish_self_scan_required") == 1
                and public.get("legacy_finding_count") == 0
                and public.get("secret_finding_count") == 0
                and public.get("binary_count") == 0
                and public.get("unreadable_count") == 0):
            raise JLGD03GateV4Error("public scan facts 漂移")
        if rows[CONDITION_KEYS[11]] != {
                "blocked_source_count": 1,
                "frozen_license_pack_count": 6,
                "source_entry_count": 7,
            }:
            raise JLGD03GateV4Error("source/license facts 漂移")
        if rows[CONDITION_KEYS[12]] != PAPER_SHA256:
            raise JLGD03GateV4Error("paper condition facts 漂移")
        if any(value != 0 for value in rows[CONDITION_KEYS[13]].values()):
            raise JLGD03GateV4Error("zero execution condition 漂移")
        if rows[CONDITION_KEYS[14]] != {
                "supersedes": "v3",
                "v3_artifact_status": "PASS",
                "v3_sha256": V3_SHA256,
            }:
            raise JLGD03GateV4Error("v3 preservation facts 漂移")
        if rows[CONDITION_KEYS[15]] != {
                "d03_published": 0,
                "publication_scope": "FUTURE_CONFIRMED_SESSION_ONLY",
                "w01_started": 0,
            }:
            raise JLGD03GateV4Error("D-03 hold facts 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "candidate_file_count": self.candidate_file_count,
            "candidate_paths": list(self.candidate_paths),
            "conditions": [item.to_dict() for item in self.conditions],
            "conjunction_passed": self.conjunction_passed,
            "d03_published": self.d03_published,
            "d03_release_decision": self.d03_release_decision,
            "dependency_edges": [item.to_dict() for item in self.dependency_edges],
            "dependency_nodes": [item.to_dict() for item in self.dependency_nodes],
            "execution_state": self.execution_state.to_value(),
            "file_inventory": [item.to_dict() for item in self.file_inventory],
            "final_public_gate": self.final_public_gate.to_dict(),
            "format_version": self.format_version,
            "head_sha1": self.head_sha1,
            "inventory_exclusions": list(self.inventory_exclusions),
            "origin_master_sha1": self.origin_master_sha1,
            "paper_files": [item.to_dict() for item in self.paper_files],
            "staged_change_count": self.staged_change_count,
            "task_key": self.task_key,
            "tracked_change_count": self.tracked_change_count,
            "untracked_file_count": self.untracked_file_count,
            "version_keys": self.version_keys.to_value(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JLGD03GateV4Manifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "candidate_file_count", "candidate_paths", "conditions",
            "conjunction_passed",
            "d03_published", "d03_release_decision", "dependency_edges",
            "dependency_nodes", "execution_state", "file_inventory",
            "final_public_gate", "format_version", "head_sha1",
            "inventory_exclusions", "origin_master_sha1", "paper_files",
            "staged_change_count", "task_key", "tracked_change_count",
            "untracked_file_count", "version_keys",
        }, where="JLGD03GateV4Manifest")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["artifact_status"]),
            str(raw["task_key"]), str(raw["head_sha1"]),
            str(raw["origin_master_sha1"]), raw["tracked_change_count"],
            raw["staged_change_count"], raw["untracked_file_count"],
            raw["candidate_file_count"],
            tuple(str(item) for item in raw["candidate_paths"]),
            tuple(str(item) for item in raw["inventory_exclusions"]),
            tuple(PublicFileIdentity.from_dict(item)
                  for item in raw["file_inventory"]),
            tuple(PublicFileIdentity.from_dict(item)
                  for item in raw["paper_files"]),
            tuple(GateV4DependencyNode.from_dict(item)
                  for item in raw["dependency_nodes"]),
            tuple(GateV4DependencyEdge.from_dict(item)
                  for item in raw["dependency_edges"]),
            CanonicalJsonObject.from_value(raw["version_keys"]),
            FinalPublicGate.from_dict(raw["final_public_gate"]),
            tuple(GateCondition.from_dict(item) for item in raw["conditions"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
            raw["conjunction_passed"], str(raw["d03_release_decision"]),
            raw["d03_published"],
        )


def read_jlg_d03_gate_v4(path: str | Path) -> JLGD03GateV4Manifest:
    """严格回读 canonical v4 gate。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise JLGD03GateV4Error("v4 newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = JLGD03GateV4Manifest.from_dict(value)
    except JLGD03GateV4Error:
        raise
    except Exception as error:
        raise JLGD03GateV4Error("v4 gate 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise JLGD03GateV4Error("v4 gate 非规范字节")
    return manifest


def write_jlg_d03_gate_v4(
        manifest: JLGD03GateV4Manifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 v4，禁止同版本异内容覆盖。"""
    if not isinstance(manifest, JLGD03GateV4Manifest):
        raise JLGD03GateV4Error("v4 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise JLGD03GateV4Error("v4 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise JLGD03GateV4Error("v4 无法写入") from error
    return target


def verify_jlg_d03_gate_v4_files(
        manifest: JLGD03GateV4Manifest,
        *, repository_root: str | Path,
        ) -> None:
    """逐字节回验 candidate inventory、paper 与 dependency nodes。"""
    root = Path(repository_root).resolve()
    identities = (*manifest.file_inventory, *manifest.paper_files)
    for item in identities:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise JLGD03GateV4Error("file identity 路径逃逸") from error
        if not path.is_file():
            raise JLGD03GateV4Error("file identity 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.size_bytes
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise JLGD03GateV4Error("file identity 漂移")
    inventory = {
        item.relative_path: (item.size_bytes, item.sha256)
        for item in manifest.file_inventory}
    for node in manifest.dependency_nodes:
        if inventory[node.relative_path] != (node.byte_count, node.sha256):
            raise JLGD03GateV4Error("dependency node 与 inventory 漂移")


__all__ = [
    "ARTIFACT_KIND",
    "ARTIFACT_PATH",
    "ARTIFACT_VERSION",
    "CONDITION_KEYS",
    "EXECUTION_STATE_KEYS",
    "PAPER_SHA256",
    "REQUIRED_EDGE_PAIRS",
    "REQUIRED_NODE_SPECS",
    "V3_PATH",
    "V3_SHA256",
    "GateV4DependencyEdge",
    "GateV4DependencyNode",
    "JLGD03GateV4Error",
    "JLGD03GateV4Manifest",
    "read_jlg_d03_gate_v4",
    "verify_jlg_d03_gate_v4_files",
    "write_jlg_d03_gate_v4",
]
