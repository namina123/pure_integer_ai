"""W-08 五维公共合同、payload inventory 与运行请求边界。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import ArtifactFileIdentity
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_AUTHORITY_RELATIVE_PATH,
    W08_DIMENSION_KEYS,
    W08_FUTURE_PACK_KEYS,
    W08_SUBTASK_ORDER,
    read_w08_authority,
)


W08_STAGE_KEY = "W-08"
W08_OWNER_KEY = "PH2_W08_TRANSACTION_OWNER"
W08_RUNNER_KEY = "PH2_LANGUAGE_STAGE8_DISCOURSE_GENERATION"
W08_CANDIDATE_OWNER = "PH2_TRAIN_CANDIDATE"
W08_TEACHER_OWNER = "PH2_TRAINING_EVIDENCE"
W08_EVALUATOR_OWNER = "PH2_PRIVATE_EVALUATOR"
W08_ALLOWED_MODES = ("fresh", "restart", "resume")
W08_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W08_FAILURE_POINT_KEYS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)
W08_STOP_STATES = (
    "RESOLVED",
    "CLARIFY",
    "UNKNOWN",
    "ACCESS_BLOCKED",
    "GROUNDING_BLOCKED",
    "BUDGET_EXHAUSTED",
)
W08_CONSUMER_KEYS = ("UNDERSTANDING", "REASONING", "GENERATION")
W08_CARRIER_KEYS = (
    "DOCUMENT_CONTAINER",
    "HTML",
    "MARKDOWN",
    "MATH_NOTATION",
    "PLAIN_TEXT",
    "REFERENCE_LINK_EMBED",
    "SOURCE_CODE",
    "TABLE_GRID",
    "TRANSCRIBED_OCR_ASR",
)
W08_LEARNING_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--discourse-revision-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc07-discourse-information-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc08-open-set-clarification-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc14-attribution-quotation-v1",
    "ZHWIKIPEDIA_20260701--CC-BY-SA-4.0--source-pack-v1",
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1",
)
W08_RESOURCE_BUDGET = {
    "max_checkpoint_count": 2048,
    "max_logic_operations": 8000000,
    "max_payload_bytes": 536870912,
    "max_payload_gets": 524288,
    "max_recompute_objects": 800000,
    "max_records": 800000,
    "max_segments": 32768,
    "max_workers": 4,
}
W08_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W07_RUNTIME_EVIDENCED": 1,
    "W08_STARTED": 0,
    "W09_STARTED": 0,
    "companion_writes": 0,
    "formal_w08_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}
W08_STAGE_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/stages/w08_stage_manifest_v1.json"
)
W08_GLOBAL_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)


class W08ContractError(RuntimeError):
    """W-08 owner、schema、inventory、request 或资源边界漂移。"""


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W08ContractError("W-08 path is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or "//" in value:
        raise W08ContractError("W-08 path escapes repository")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W08ContractError(f"missing W-08 contract parent: {path.name}") from error
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class W08FileBinding:
    """绑定单个 pack 文件的完整路径、owner、split、phase 与双摘要。"""

    relative_path: str
    pack_key: str
    access_phase: str
    identity: ArtifactFileIdentity

    def __post_init__(self) -> None:
        _safe_relative(self.relative_path)
        if self.access_phase not in {"candidate", "teacher", "evaluator", "forbidden"}:
            raise W08ContractError("W-08 file access phase is invalid")
        if not isinstance(self.identity, ArtifactFileIdentity):
            raise W08ContractError("W-08 file identity is missing")
        if not self.relative_path.endswith("/" + self.identity.relative_path):
            raise W08ContractError("W-08 file path and identity drifted")
        pair = (self.identity.owner_kind, self.identity.split)
        allowed = {
            "candidate": {("source", None), ("observation", "train")},
            "teacher": {("teacher", "train")},
            "evaluator": {
                ("observation", "held_out"),
                ("evaluator", "held_out"),
            },
            "forbidden": {
                ("observation", "dev"),
                ("evaluator", "dev"),
                ("teacher", "held_out"),
            },
        }
        if pair not in allowed[self.access_phase]:
            raise W08ContractError("W-08 file owner/split does not match phase")

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_phase": self.access_phase,
            "identity": self.identity.to_dict(),
            "pack_key": self.pack_key,
            "relative_path": self.relative_path,
        }


@dataclass
class W08PayloadAudit:
    """累计 transport、交付和所有禁止读取/写入计数。"""

    transport_attempts: int = 0
    transport_bytes: int = 0
    payload_gets: int = 0
    payload_bytes: int = 0
    source_ref_reads: int = 0
    observation_reads: int = 0
    teacher_evidence_reads: int = 0
    evaluator_label_reads: int = 0
    held_out_reads: int = 0
    future_payload_reads: int = 0
    teacher_calls: int = 0
    learning_writes: int = 0
    memory_learning_writes: int = 0


@dataclass(frozen=True)
class W08FrozenContract:
    """首次 payload transport 前闭合的 W-08 公共合同。"""

    authority_sha256: str
    baseline_public_head_commit_sha1: str
    stage_train_pack_keys: tuple[str, ...]
    stage_dev_pack_keys: tuple[str, ...]
    stage_held_out_pack_keys: tuple[str, ...]
    stage_evaluator_pack_keys: tuple[str, ...]
    future_pack_keys: tuple[str, ...]
    learning_pack_keys: tuple[str, ...]
    candidate_bindings: tuple[W08FileBinding, ...]
    teacher_bindings: tuple[W08FileBinding, ...]
    evaluator_bindings: tuple[W08FileBinding, ...]
    forbidden_bindings: tuple[W08FileBinding, ...]
    future_forbidden_paths: tuple[str, ...]
    subtask_order: tuple[str, ...]
    dimension_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    consumer_keys: tuple[str, ...]
    stop_states: tuple[str, ...]
    carrier_keys: tuple[str, ...]
    discourse_state_components: tuple[str, ...]
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    logical_shard_count: int
    merge_barrier_key: str
    cursor_version: str
    logical_clock_version: str
    resource_budget: tuple[tuple[str, int], ...]
    owner_key: str
    candidate_owner: str
    teacher_owner: str
    evaluator_owner: str
    runner_key: str
    execution_state: tuple[tuple[str, int], ...]
    base_fence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.authority_sha256) != 64 or len(self.baseline_public_head_commit_sha1) != 40:
            raise W08ContractError("W-08 authority identity is invalid")
        if (
            self.future_pack_keys != W08_FUTURE_PACK_KEYS
            or self.learning_pack_keys != W08_LEARNING_PACK_KEYS
            or set(self.future_pack_keys) & set(self.stage_train_pack_keys)
        ):
            raise W08ContractError("W-08 train/future boundary drifted")
        phases = (
            (self.candidate_bindings, "candidate"),
            (self.teacher_bindings, "teacher"),
            (self.evaluator_bindings, "evaluator"),
            (self.forbidden_bindings, "forbidden"),
        )
        paths: list[str] = []
        for bindings, phase in phases:
            if any(item.access_phase != phase for item in bindings):
                raise W08ContractError("W-08 binding phase drifted")
            paths.extend(item.relative_path for item in bindings)
        if len(paths) != len(set(paths)):
            raise W08ContractError("W-08 payload path appears in multiple phases")
        if any(pack not in self.learning_pack_keys for pack in (
            *(item.pack_key for item in self.candidate_bindings),
            *(item.pack_key for item in self.teacher_bindings),
        )):
            raise W08ContractError("W-08 candidate binding uses non-learning pack")
        if (
            self.subtask_order != W08_SUBTASK_ORDER
            or self.dimension_keys != W08_DIMENSION_KEYS
            or self.ablation_keys != W08_ABLATION_KEYS
            or self.consumer_keys != W08_CONSUMER_KEYS
            or self.stop_states != W08_STOP_STATES
            or self.carrier_keys != W08_CARRIER_KEYS
            or self.discourse_state_components
            != ("APPEND_ONLY_EVENT_LOG", "CURRENT_PROJECTION", "DEPENDENCY_INDEX")
        ):
            raise W08ContractError("W-08 schema/registry order drifted")
        if (
            self.allowed_worker_counts != W08_ALLOWED_WORKER_COUNTS
            or self.failure_point_keys != W08_FAILURE_POINT_KEYS
            or self.logical_shard_count != 16
            or self.merge_barrier_key != "PH2-D03-STABLE-MERGE-BARRIER-V1"
            or self.cursor_version != "PH2-D03-CURSOR-V1"
            or self.logical_clock_version != "PH2-W08-LOGICAL-CLOCK-V1"
            or dict(self.resource_budget) != W08_RESOURCE_BUDGET
        ):
            raise W08ContractError("W-08 recovery/resource contract drifted")
        if (
            self.owner_key != W08_OWNER_KEY
            or self.candidate_owner != W08_CANDIDATE_OWNER
            or self.teacher_owner != W08_TEACHER_OWNER
            or self.evaluator_owner != W08_EVALUATOR_OWNER
            or self.runner_key != W08_RUNNER_KEY
            or dict(self.execution_state) != W08_ZERO_EXECUTION_STATE
            or len(self.base_fence_key) != 32
        ):
            raise W08ContractError("W-08 owner/state/base fence drifted")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "ablation_keys": list(self.ablation_keys),
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "authority_sha256": self.authority_sha256,
            "baseline_public_head_commit_sha1": self.baseline_public_head_commit_sha1,
            "candidate_bindings": [item.to_dict() for item in self.candidate_bindings],
            "carrier_keys": list(self.carrier_keys),
            "consumer_keys": list(self.consumer_keys),
            "cursor_version": self.cursor_version,
            "dimension_keys": list(self.dimension_keys),
            "discourse_state_components": list(self.discourse_state_components),
            "evaluator_bindings": [item.to_dict() for item in self.evaluator_bindings],
            "failure_point_keys": list(self.failure_point_keys),
            "forbidden_bindings": [item.to_dict() for item in self.forbidden_bindings],
            "future_forbidden_paths": list(self.future_forbidden_paths),
            "future_pack_keys": list(self.future_pack_keys),
            "learning_pack_keys": list(self.learning_pack_keys),
            "logical_clock_version": self.logical_clock_version,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "owner_key": self.owner_key,
            "resource_budget": dict(self.resource_budget),
            "runner_key": self.runner_key,
            "stop_states": list(self.stop_states),
            "subtask_order": list(self.subtask_order),
            "teacher_bindings": [item.to_dict() for item in self.teacher_bindings],
        })


@dataclass(frozen=True)
class W08RunRequest:
    """Candidate 的精确 contract/base/owner/resource/payload 请求。"""

    stage_key: str
    owner_key: str
    runner_key: str
    contract_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]
    forbidden_payload_paths: tuple[str, ...] = ()

    def execution_identity_key(self) -> tuple[int, ...]:
        return digest_value({
            "base_fence_key": list(self.base_fence_key),
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "contract_key": list(self.contract_key),
            "owner_key": self.owner_key,
            "resource_budget": dict(self.resource_budget),
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        return digest_value({
            "execution": list(self.execution_identity_key()),
            "mode": self.mode,
            "worker_count": self.worker_count,
        })


def validate_w08_request(
    context: W08FrozenContract, request: W08RunRequest
) -> W08RunRequest:
    if not isinstance(context, W08FrozenContract) or not isinstance(request, W08RunRequest):
        raise W08ContractError("W-08 request/context type is invalid")
    if (
        request.stage_key != W08_STAGE_KEY
        or request.owner_key != context.owner_key
        or request.runner_key != context.runner_key
        or request.contract_key != context.stable_key()
        or request.base_fence_key != context.base_fence_key
    ):
        raise W08ContractError("W-08 request owner/contract/base fence drifted")
    if request.worker_count not in W08_ALLOWED_WORKER_COUNTS or request.mode not in W08_ALLOWED_MODES:
        raise W08ContractError("W-08 worker count or mode is invalid")
    if request.resource_budget != tuple(sorted(W08_RESOURCE_BUDGET.items())):
        raise W08ContractError("W-08 resource budget drifted")
    candidate = tuple(item.relative_path for item in context.candidate_bindings)
    teacher = tuple(item.relative_path for item in context.teacher_bindings)
    if request.candidate_payload_paths != candidate or request.teacher_evidence_paths != teacher:
        raise W08ContractError("W-08 request is not the exact train whitelist")
    if request.forbidden_payload_paths:
        raise W08ContractError("W-08 candidate request contains forbidden paths")
    return request


def make_w08_request(
    context: W08FrozenContract, *, worker_count: int = 1, mode: str = "fresh"
) -> W08RunRequest:
    request = W08RunRequest(
        W08_STAGE_KEY,
        context.owner_key,
        context.runner_key,
        context.stable_key(),
        context.base_fence_key,
        worker_count,
        mode,
        tuple(sorted(W08_RESOURCE_BUDGET.items())),
        tuple(item.relative_path for item in context.candidate_bindings),
        tuple(item.relative_path for item in context.teacher_bindings),
    )
    return validate_w08_request(context, request)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W08ContractError(f"cannot read W-08 JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise W08ContractError("W-08 JSON must be an object")
    return value


def _binding(
    pack_root: str,
    pack_key: str,
    phase: str,
    identity: ArtifactFileIdentity,
) -> W08FileBinding:
    return W08FileBinding(
        f"{pack_root}/{identity.relative_path}", pack_key, phase, identity
    )


def open_w08_frozen_contract(repository_root: str | Path) -> W08FrozenContract:
    """只读打开 authority、D-03 inventory 和允许 pack manifest。"""
    repository = Path(repository_root).resolve()
    authority = read_w08_authority(repository)
    authority_path = repository / W08_AUTHORITY_RELATIVE_PATH
    stage = _read_json(repository / W08_STAGE_MANIFEST_PATH)
    global_manifest = _read_json(repository / W08_GLOBAL_MANIFEST_PATH)
    visibility = stage.get("data_visibility", {})
    by_pack = {
        item.get("pack_key"): item
        for item in global_manifest.get("pack_bindings", [])
        if isinstance(item, dict)
    }
    candidate: list[W08FileBinding] = []
    teacher: list[W08FileBinding] = []
    evaluator: list[W08FileBinding] = []
    forbidden: list[W08FileBinding] = []
    future_paths: list[str] = []
    stage_keys = set((
        *visibility.get("train_pack_keys", []),
        *visibility.get("dev_pack_keys", []),
        *visibility.get("held_out_pack_keys", []),
        *visibility.get("evaluator_pack_keys", []),
    ))
    for pack_key in sorted(stage_keys):
        declared = by_pack.get(pack_key)
        if not isinstance(declared, dict):
            raise W08ContractError(f"missing D-03 pack binding: {pack_key}")
        manifest_info = declared.get("manifest_identity", {})
        manifest_relative = _safe_relative(manifest_info.get("relative_path"))
        manifest_path = repository / manifest_relative
        if (
            manifest_path.stat().st_size != manifest_info.get("size_bytes")
            or _sha256(manifest_path) != manifest_info.get("sha256")
        ):
            raise W08ContractError(f"W-08 pack manifest identity drifted: {pack_key}")
        manifest = read_artifact_manifest(manifest_path)
        pack_root = PurePosixPath(manifest_relative).parent.as_posix()
        for identity in manifest.files:
            pair = (identity.owner_kind, identity.split)
            if pack_key in W08_LEARNING_PACK_KEYS and pair in {
                ("source", None), ("observation", "train")
            }:
                candidate.append(_binding(pack_root, pack_key, "candidate", identity))
            elif pack_key in W08_LEARNING_PACK_KEYS and pair == ("teacher", "train"):
                teacher.append(_binding(pack_root, pack_key, "teacher", identity))
            elif pair in {
                ("observation", "held_out"), ("evaluator", "held_out")
            }:
                evaluator.append(_binding(pack_root, pack_key, "evaluator", identity))
            elif pair in {
                ("observation", "dev"), ("evaluator", "dev"),
                ("teacher", "held_out"),
            }:
                forbidden.append(_binding(pack_root, pack_key, "forbidden", identity))
    for pack_key in W08_FUTURE_PACK_KEYS:
        declared = by_pack.get(pack_key)
        if not isinstance(declared, dict):
            raise W08ContractError(f"missing future pack binding: {pack_key}")
        paths = (
            declared.get("train_observation_paths", []),
            declared.get("dev_observation_paths", []),
            declared.get("held_out_observation_paths", []),
            declared.get("teacher_evidence_paths", []),
            declared.get("evaluator_label_paths", []),
        )
        for group in paths:
            for relative in group:
                future_paths.append(_safe_relative(relative))
        future_paths.append(_safe_relative(
            declared.get("manifest_identity", {}).get("relative_path")
        ))
    candidate.sort(key=lambda item: (W08_LEARNING_PACK_KEYS.index(item.pack_key), item.identity.owner_kind))
    teacher.sort(key=lambda item: W08_LEARNING_PACK_KEYS.index(item.pack_key))
    evaluator.sort(key=lambda item: (item.pack_key, item.identity.owner_kind, item.relative_path))
    forbidden.sort(key=lambda item: (item.pack_key, item.identity.owner_kind, item.relative_path))
    future_tuple = tuple(sorted(set(future_paths)))
    base_fence = digest_value({
        "authority_sha256": _sha256(authority_path),
        "baseline_public_head_commit_sha1": authority["baseline_public_head_commit_sha1"],
        "candidate_paths": [item.relative_path for item in candidate],
        "evaluator_paths": [item.relative_path for item in evaluator],
        "future_paths": list(future_tuple),
        "resource_budget": W08_RESOURCE_BUDGET,
    })
    return W08FrozenContract(
        _sha256(authority_path),
        authority["baseline_public_head_commit_sha1"],
        tuple(visibility.get("train_pack_keys", [])),
        tuple(visibility.get("dev_pack_keys", [])),
        tuple(visibility.get("held_out_pack_keys", [])),
        tuple(visibility.get("evaluator_pack_keys", [])),
        tuple(visibility.get("future_pack_keys", [])),
        W08_LEARNING_PACK_KEYS,
        tuple(candidate),
        tuple(teacher),
        tuple(evaluator),
        tuple(forbidden),
        future_tuple,
        W08_SUBTASK_ORDER,
        W08_DIMENSION_KEYS,
        W08_ABLATION_KEYS,
        W08_CONSUMER_KEYS,
        W08_STOP_STATES,
        W08_CARRIER_KEYS,
        ("APPEND_ONLY_EVENT_LOG", "CURRENT_PROJECTION", "DEPENDENCY_INDEX"),
        W08_ALLOWED_WORKER_COUNTS,
        W08_FAILURE_POINT_KEYS,
        16,
        "PH2-D03-STABLE-MERGE-BARRIER-V1",
        "PH2-D03-CURSOR-V1",
        "PH2-W08-LOGICAL-CLOCK-V1",
        tuple(sorted(W08_RESOURCE_BUDGET.items())),
        W08_OWNER_KEY,
        W08_CANDIDATE_OWNER,
        W08_TEACHER_OWNER,
        W08_EVALUATOR_OWNER,
        W08_RUNNER_KEY,
        tuple(sorted(W08_ZERO_EXECUTION_STATE.items())),
        base_fence,
    )


__all__ = [
    "W08_ALLOWED_WORKER_COUNTS",
    "W08_CARRIER_KEYS",
    "W08_CONSUMER_KEYS",
    "W08_FAILURE_POINT_KEYS",
    "W08_LEARNING_PACK_KEYS",
    "W08_RESOURCE_BUDGET",
    "W08_STOP_STATES",
    "W08ContractError",
    "W08FileBinding",
    "W08FrozenContract",
    "W08PayloadAudit",
    "W08RunRequest",
    "make_w08_request",
    "open_w08_frozen_contract",
    "validate_w08_request",
]
