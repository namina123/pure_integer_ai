"""W-09 独立 owner、pack registry、file binding 与运行请求合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    StableRecordKey,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_artifact_manifest,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_ALLOWED_WORKER_COUNTS,
    W09_AUTHORITY_RELATIVE_PATH,
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_FAILURE_POINT_KEYS,
    W09_GLOBAL_MANIFEST_PATH,
    W09_RESOURCE_BUDGET,
    W09_STOP_STATES,
    W09_SUBTASK_ORDER,
    read_w09_authority,
)


W09_STAGE_KEY = "W-09"
W09_OWNER_KEY = "PH2_W09_TRANSACTION_OWNER"
W09_RUNNER_KEY = "PH2_LANGUAGE_STAGE9_RETENTION_CONTINUAL_LEARNING"
W09_CANDIDATE_OWNER = "PH2_TRAIN_CANDIDATE"
W09_TRAINING_MATERIAL_OWNER = "PH2_TRAINING_EVIDENCE"
W09_DEV_OWNER = "PH2_W09_DEV_CALIBRATION_OWNER"
W09_EVALUATOR_OWNER = "PH2_PRIVATE_EVALUATOR"
W09_ALLOWED_MODES = ("fresh", "restart", "resume")
W09_ACCESS_PHASES = (
    "candidate",
    "training_material",
    "dev",
    "evaluator",
    "forbidden",
)


class W09ContractError(RuntimeError):
    """W-09 owner、registry、request 或资源合同发生漂移。"""


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W09ContractError("W-09 path is not canonical")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
        or "//" in value
        or "!" in value
    ):
        raise W09ContractError("W-09 path is outside the exact registry")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W09ContractError("W-09 metadata parent is unavailable") from error
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W09ContractError("W-09 metadata JSON is invalid") from error
    if not isinstance(value, dict):
        raise W09ContractError("W-09 metadata JSON must be an object")
    return value


def _has_link_component(root: Path, relative: str) -> bool:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
        is_junction = getattr(current, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
    return False


def _metadata_file(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    if _has_link_component(root, safe):
        raise W09ContractError("W-09 metadata path contains a link")
    target = (root / Path(*PurePosixPath(safe).parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise W09ContractError("W-09 metadata path escapes repository") from error
    if not target.is_file():
        raise W09ContractError("W-09 metadata file is missing")
    return target


@dataclass(frozen=True, order=True)
class W09FileBinding:
    """一个文件的 pack、相位、owner/split 与 transport/content identity。"""

    relative_path: str
    pack_key: str
    access_phase: str
    artifact_key: StableRecordKey
    identity: ArtifactFileIdentity

    def __post_init__(self) -> None:
        _safe_relative(self.relative_path)
        if not isinstance(self.pack_key, str) or not self.pack_key:
            raise W09ContractError("W-09 pack key is invalid")
        if self.access_phase not in W09_ACCESS_PHASES:
            raise W09ContractError("W-09 file access phase is invalid")
        if not isinstance(self.artifact_key, StableRecordKey):
            raise W09ContractError("W-09 artifact key is missing")
        if not isinstance(self.identity, ArtifactFileIdentity):
            raise W09ContractError("W-09 file identity is missing")
        if not self.relative_path.endswith("/" + self.identity.relative_path):
            raise W09ContractError("W-09 file path and identity drifted")
        pair = (self.identity.owner_kind, self.identity.split)
        allowed = {
            "candidate": {("source", None), ("observation", "train")},
            "training_material": {("teacher", "train")},
            "dev": {("observation", "dev"), ("evaluator", "dev")},
            "evaluator": {
                ("source", None),
                ("observation", "held_out"),
                ("evaluator", "held_out"),
            },
            "forbidden": {
                ("evaluator", "train"),
                ("teacher", "dev"),
                ("teacher", "held_out"),
            },
        }
        if pair not in allowed[self.access_phase]:
            raise W09ContractError("W-09 file owner/split does not match phase")

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_phase": self.access_phase,
            "artifact_key": self.artifact_key.to_list(),
            "identity": self.identity.to_dict(),
            "pack_key": self.pack_key,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class W09HostWriteSnapshot:
    """dev/evaluator 调用边界的六个 production host 写计数。"""

    core_writes: int = 0
    evidence_writes: int = 0
    use_writes: int = 0
    memory_writes: int = 0
    assessment_writes: int = 0
    clock_writes: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.to_tuple()):
            raise W09ContractError("W-09 host write snapshot is invalid")

    def to_tuple(self) -> tuple[int, ...]:
        return (
            self.core_writes,
            self.evidence_writes,
            self.use_writes,
            self.memory_writes,
            self.assessment_writes,
            self.clock_writes,
        )

    @property
    def is_zero(self) -> bool:
        return self.to_tuple() == (0, 0, 0, 0, 0, 0)


@dataclass
class W09PayloadAudit:
    """只累计安全计数，不保存 path、surface、expected、label 或异常文本。"""

    transport_attempts: int = 0
    transport_bytes: int = 0
    payload_gets: int = 0
    payload_bytes: int = 0
    source_ref_reads: int = 0
    observation_reads: int = 0
    training_evidence_reads: int = 0
    dev_reads: int = 0
    held_out_reads: int = 0
    evaluator_label_reads: int = 0
    redacted_candidate_fields: int = 0
    teacher_calls: int = 0
    api_calls: int = 0
    llm_calls: int = 0
    core_writes: int = 0
    evidence_writes: int = 0
    use_writes: int = 0
    memory_writes: int = 0
    assessment_writes: int = 0
    clock_writes: int = 0

    def safe_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(
            (name, value)
            for name, value in vars(self).items()
            if isinstance(value, int)
        ))


@dataclass(frozen=True)
class W09FrozenContract:
    """首次 payload transport 前冻结的 W-09 完整公共合同。"""

    authority_sha256: str
    baseline_public_head_commit_sha1: str
    candidate_pack_keys: tuple[str, ...]
    dev_pack_keys: tuple[str, ...]
    held_out_pack_keys: tuple[str, ...]
    evaluator_pack_keys: tuple[str, ...]
    future_pack_keys: tuple[str, ...]
    candidate_bindings: tuple[W09FileBinding, ...]
    training_material_bindings: tuple[W09FileBinding, ...]
    dev_bindings: tuple[W09FileBinding, ...]
    evaluator_bindings: tuple[W09FileBinding, ...]
    forbidden_bindings: tuple[W09FileBinding, ...]
    carrier_keys: tuple[str, ...]
    consumer_keys: tuple[str, ...]
    dimension_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    stop_states: tuple[str, ...]
    subtask_order: tuple[str, ...]
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    logical_shard_count: int
    resource_budget: tuple[tuple[str, int], ...]
    execution_state: tuple[tuple[str, int], ...]
    owner_key: str
    candidate_owner: str
    training_material_owner: str
    dev_owner: str
    evaluator_owner: str
    runner_key: str
    base_fence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.authority_sha256) != 64 or len(self.baseline_public_head_commit_sha1) != 40:
            raise W09ContractError("W-09 authority identity is invalid")
        registries = (
            (self.candidate_pack_keys, 34),
            (self.dev_pack_keys, 1),
            (self.held_out_pack_keys, 37),
            (self.evaluator_pack_keys, 37),
        )
        if any(len(keys) != count or len(set(keys)) != count for keys, count in registries):
            raise W09ContractError("W-09 pack registry cardinality drifted")
        if self.held_out_pack_keys != self.evaluator_pack_keys or self.future_pack_keys:
            raise W09ContractError("W-09 evaluator/future registry drifted")
        phases = (
            (self.candidate_bindings, "candidate", self.candidate_pack_keys),
            (
                self.training_material_bindings,
                "training_material",
                self.candidate_pack_keys,
            ),
            (self.dev_bindings, "dev", self.dev_pack_keys),
            (self.evaluator_bindings, "evaluator", self.evaluator_pack_keys),
        )
        for bindings, phase, pack_keys in phases:
            if any(item.access_phase != phase for item in bindings):
                raise W09ContractError("W-09 binding phase drifted")
            if {item.pack_key for item in bindings} != set(pack_keys):
                raise W09ContractError("W-09 binding pack registry is incomplete")
        expected_shapes = {
            "candidate": {("source", None), ("observation", "train")},
            "training_material": {("teacher", "train")},
            "dev": {("observation", "dev"), ("evaluator", "dev")},
            "evaluator": {
                ("source", None),
                ("observation", "held_out"),
                ("evaluator", "held_out"),
            },
        }
        for bindings, phase, pack_keys in phases:
            for pack_key in pack_keys:
                shape = {
                    (item.identity.owner_kind, item.identity.split)
                    for item in bindings
                    if item.pack_key == pack_key
                }
                count = sum(item.pack_key == pack_key for item in bindings)
                if shape != expected_shapes[phase] or count != len(expected_shapes[phase]):
                    raise W09ContractError("W-09 per-pack file shape drifted")
        if any(item.access_phase != "forbidden" for item in self.forbidden_bindings):
            raise W09ContractError("W-09 forbidden binding phase drifted")
        by_path: dict[str, list[W09FileBinding]] = {}
        for bindings, _, _ in phases:
            for item in bindings:
                by_path.setdefault(item.relative_path, []).append(item)
        for items in by_path.values():
            if len(items) == 1:
                continue
            if (
                len(items) != 2
                or {item.access_phase for item in items} != {"candidate", "evaluator"}
                or any(item.identity.owner_kind != "source" for item in items)
            ):
                raise W09ContractError("W-09 payload path crosses access phases")
        if (
            self.carrier_keys != W09_CARRIER_KEYS
            or self.consumer_keys != W09_CONSUMER_KEYS
            or self.dimension_keys != W09_DIMENSION_KEYS
            or self.ablation_keys != W09_ABLATION_KEYS
            or self.stop_states != W09_STOP_STATES
            or self.subtask_order != W09_SUBTASK_ORDER
        ):
            raise W09ContractError("W-09 typed registry order drifted")
        if (
            self.allowed_worker_counts != W09_ALLOWED_WORKER_COUNTS
            or self.failure_point_keys != W09_FAILURE_POINT_KEYS
            or self.logical_shard_count != 16
            or dict(self.resource_budget) != W09_RESOURCE_BUDGET
        ):
            raise W09ContractError("W-09 recovery/resource contract drifted")
        if (
            self.owner_key != W09_OWNER_KEY
            or self.candidate_owner != W09_CANDIDATE_OWNER
            or self.training_material_owner != W09_TRAINING_MATERIAL_OWNER
            or self.dev_owner != W09_DEV_OWNER
            or self.evaluator_owner != W09_EVALUATOR_OWNER
            or self.runner_key != W09_RUNNER_KEY
            or len(self.base_fence_key) != 32
        ):
            raise W09ContractError("W-09 owner/base fence drifted")
        state = dict(self.execution_state)
        if (
            state.get("W09_STARTED") != 0
            or state.get("formal_w09_training_runs") != 0
            or state.get("teacher_calls") != 0
            or state.get("LANGUAGE_CAPABILITY_MASTERED") != 0
            or state.get("LANGUAGE_READINESS") != 0
        ):
            raise W09ContractError("W-09 zero execution state drifted")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "ablation_keys": list(self.ablation_keys),
            "authority_sha256": self.authority_sha256,
            "base_fence_key": list(self.base_fence_key),
            "candidate_bindings": [item.to_dict() for item in self.candidate_bindings],
            "candidate_pack_keys": list(self.candidate_pack_keys),
            "carrier_keys": list(self.carrier_keys),
            "consumer_keys": list(self.consumer_keys),
            "dev_bindings": [item.to_dict() for item in self.dev_bindings],
            "dev_pack_keys": list(self.dev_pack_keys),
            "dimension_keys": list(self.dimension_keys),
            "evaluator_bindings": [item.to_dict() for item in self.evaluator_bindings],
            "evaluator_pack_keys": list(self.evaluator_pack_keys),
            "owner_key": self.owner_key,
            "resource_budget": dict(self.resource_budget),
            "runner_key": self.runner_key,
            "training_material_bindings": [
                item.to_dict() for item in self.training_material_bindings
            ],
        })


@dataclass(frozen=True)
class W09RunRequest:
    """Candidate 的精确 owner/base/resource/34-pack 请求。"""

    stage_key: str
    owner_key: str
    runner_key: str
    contract_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_pack_keys: tuple[str, ...]
    candidate_payload_paths: tuple[str, ...]
    training_material_paths: tuple[str, ...]
    forbidden_payload_paths: tuple[str, ...] = ()

    def execution_identity_key(self) -> tuple[int, ...]:
        return digest_value({
            "base_fence_key": list(self.base_fence_key),
            "candidate_pack_keys": list(self.candidate_pack_keys),
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "contract_key": list(self.contract_key),
            "owner_key": self.owner_key,
            "resource_budget": dict(self.resource_budget),
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "training_material_paths": list(self.training_material_paths),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        return digest_value({
            "execution": list(self.execution_identity_key()),
            "mode": self.mode,
            "worker_count": self.worker_count,
        })


def validate_w09_request(
    context: W09FrozenContract,
    request: W09RunRequest,
) -> W09RunRequest:
    if not isinstance(context, W09FrozenContract) or not isinstance(request, W09RunRequest):
        raise W09ContractError("W-09 request/context type is invalid")
    if (
        request.stage_key != W09_STAGE_KEY
        or request.owner_key != context.owner_key
        or request.runner_key != context.runner_key
        or request.contract_key != context.stable_key()
        or request.base_fence_key != context.base_fence_key
    ):
        raise W09ContractError("W-09 request owner/contract/base fence drifted")
    if request.worker_count not in W09_ALLOWED_WORKER_COUNTS or request.mode not in W09_ALLOWED_MODES:
        raise W09ContractError("W-09 worker count or mode is invalid")
    if request.resource_budget != tuple(sorted(W09_RESOURCE_BUDGET.items())):
        raise W09ContractError("W-09 resource budget drifted")
    candidate_paths = tuple(item.relative_path for item in context.candidate_bindings)
    training_paths = tuple(
        item.relative_path for item in context.training_material_bindings
    )
    if (
        request.candidate_pack_keys != context.candidate_pack_keys
        or request.candidate_payload_paths != candidate_paths
        or request.training_material_paths != training_paths
        or request.forbidden_payload_paths
    ):
        raise W09ContractError("W-09 request is not the exact 34-pack whitelist")
    return request


def make_w09_request(
    context: W09FrozenContract,
    *,
    worker_count: int = 1,
    mode: str = "fresh",
) -> W09RunRequest:
    return validate_w09_request(
        context,
        W09RunRequest(
            W09_STAGE_KEY,
            context.owner_key,
            context.runner_key,
            context.stable_key(),
            context.base_fence_key,
            worker_count,
            mode,
            tuple(sorted(W09_RESOURCE_BUDGET.items())),
            context.candidate_pack_keys,
            tuple(item.relative_path for item in context.candidate_bindings),
            tuple(
                item.relative_path for item in context.training_material_bindings
            ),
        ),
    )


def _binding(
    pack_root: str,
    pack_key: str,
    phase: str,
    artifact_key: StableRecordKey,
    identity: ArtifactFileIdentity,
) -> W09FileBinding:
    return W09FileBinding(
        f"{pack_root}/{identity.relative_path}",
        pack_key,
        phase,
        artifact_key,
        identity,
    )


def open_w09_frozen_contract(repository_root: str | Path) -> W09FrozenContract:
    """回读 authority 并只用 pack manifest 元数据冻结 34/1/37 registry。"""
    repository = Path(repository_root).resolve()
    authority = read_w09_authority(repository)
    authority_path = _metadata_file(repository, W09_AUTHORITY_RELATIVE_PATH)
    global_manifest = _read_json(_metadata_file(repository, W09_GLOBAL_MANIFEST_PATH))
    inventory = authority["stage_inventory"]
    candidate_pack_keys = tuple(inventory["train_pack_keys"])
    dev_pack_keys = tuple(inventory["dev_pack_keys"])
    held_out_pack_keys = tuple(inventory["held_out_pack_keys"])
    evaluator_pack_keys = tuple(inventory["evaluator_pack_keys"])
    by_pack = {
        item.get("pack_key"): item
        for item in global_manifest.get("pack_bindings", [])
        if isinstance(item, dict)
    }
    candidate: list[W09FileBinding] = []
    training_material: list[W09FileBinding] = []
    dev: list[W09FileBinding] = []
    evaluator: list[W09FileBinding] = []
    forbidden: list[W09FileBinding] = []
    for pack_key in held_out_pack_keys:
        declared = by_pack.get(pack_key)
        if not isinstance(declared, dict):
            raise W09ContractError("W-09 pack is missing from global registry")
        manifest_info = declared.get("manifest_identity")
        if not isinstance(manifest_info, dict):
            raise W09ContractError("W-09 pack manifest identity is invalid")
        manifest_relative = _safe_relative(manifest_info.get("relative_path"))
        manifest_path = _metadata_file(repository, manifest_relative)
        if (
            manifest_path.stat().st_size != manifest_info.get("size_bytes")
            or _sha256(manifest_path) != manifest_info.get("sha256")
        ):
            raise W09ContractError("W-09 pack manifest identity drifted")
        try:
            manifest = read_artifact_manifest(manifest_path)
        except (DatasetArtifactIOError, KeyError, TypeError, ValueError) as error:
            raise W09ContractError("W-09 pack manifest schema is invalid") from error
        pack_root = PurePosixPath(manifest_relative).parent.as_posix()
        for identity in manifest.files:
            pair = (identity.owner_kind, identity.split)
            if pair == ("source", None):
                if pack_key in candidate_pack_keys:
                    candidate.append(_binding(
                        pack_root, pack_key, "candidate", manifest.stable_key, identity
                    ))
                evaluator.append(_binding(
                    pack_root, pack_key, "evaluator", manifest.stable_key, identity
                ))
            elif pair == ("observation", "train"):
                if pack_key not in candidate_pack_keys:
                    raise W09ContractError("W-09 unregistered pack exposes train payload")
                candidate.append(_binding(
                    pack_root, pack_key, "candidate", manifest.stable_key, identity
                ))
            elif pair == ("teacher", "train"):
                if pack_key not in candidate_pack_keys:
                    raise W09ContractError("W-09 unregistered pack exposes train Evidence")
                training_material.append(
                    _binding(
                        pack_root,
                        pack_key,
                        "training_material",
                        manifest.stable_key,
                        identity,
                    )
                )
            elif pair in {("observation", "dev"), ("evaluator", "dev")}:
                if pack_key not in dev_pack_keys:
                    raise W09ContractError("W-09 dev payload appears outside dev registry")
                dev.append(_binding(
                    pack_root, pack_key, "dev", manifest.stable_key, identity
                ))
            elif pair in {
                ("observation", "held_out"),
                ("evaluator", "held_out"),
            }:
                evaluator.append(_binding(
                    pack_root, pack_key, "evaluator", manifest.stable_key, identity
                ))
            elif pair in {
                ("evaluator", "train"),
                ("teacher", "dev"),
                ("teacher", "held_out"),
            }:
                forbidden.append(_binding(
                    pack_root, pack_key, "forbidden", manifest.stable_key, identity
                ))
            else:
                raise W09ContractError("W-09 pack contains an unregistered owner/split")
    pack_rank = {key: index for index, key in enumerate(held_out_pack_keys)}
    phase_rank = {"source": 0, "observation": 1, "teacher": 2, "evaluator": 3}
    sort_key = lambda item: (
        pack_rank[item.pack_key],
        phase_rank[item.identity.owner_kind],
        item.relative_path,
    )
    for bindings in (candidate, training_material, dev, evaluator, forbidden):
        bindings.sort(key=sort_key)
    base_fence = digest_value({
        "authority_sha256": _sha256(authority_path),
        "baseline_public_head_commit_sha1": authority[
            "baseline_public_head_commit_sha1"
        ],
        "candidate_bindings": [item.to_dict() for item in candidate],
        "dev_bindings": [item.to_dict() for item in dev],
        "evaluator_bindings": [item.to_dict() for item in evaluator],
        "resource_budget": W09_RESOURCE_BUDGET,
        "training_material_bindings": [
            item.to_dict() for item in training_material
        ],
    })
    return W09FrozenContract(
        _sha256(authority_path),
        authority["baseline_public_head_commit_sha1"],
        candidate_pack_keys,
        dev_pack_keys,
        held_out_pack_keys,
        evaluator_pack_keys,
        tuple(inventory["future_pack_keys"]),
        tuple(candidate),
        tuple(training_material),
        tuple(dev),
        tuple(evaluator),
        tuple(forbidden),
        tuple(authority["carrier_keys"]),
        tuple(authority["consumer_keys"]),
        tuple(authority["dimension_keys"]),
        tuple(authority["ablation_keys"]),
        tuple(authority["stop_states"]),
        tuple(authority["subtask_order"]),
        tuple(authority["transaction"]["allowed_worker_counts"]),
        tuple(authority["transaction"]["failure_point_keys"]),
        authority["transaction"]["logical_shard_count"],
        tuple(sorted(authority["resource_budget"].items())),
        tuple(sorted(authority["execution_state"].items())),
        W09_OWNER_KEY,
        W09_CANDIDATE_OWNER,
        W09_TRAINING_MATERIAL_OWNER,
        W09_DEV_OWNER,
        W09_EVALUATOR_OWNER,
        W09_RUNNER_KEY,
        base_fence,
    )


__all__ = [
    "W09_ALLOWED_MODES",
    "W09_CANDIDATE_OWNER",
    "W09_DEV_OWNER",
    "W09_EVALUATOR_OWNER",
    "W09_OWNER_KEY",
    "W09_RUNNER_KEY",
    "W09_STAGE_KEY",
    "W09_TRAINING_MATERIAL_OWNER",
    "W09ContractError",
    "W09FileBinding",
    "W09FrozenContract",
    "W09HostWriteSnapshot",
    "W09PayloadAudit",
    "W09RunRequest",
    "make_w09_request",
    "open_w09_frozen_contract",
    "validate_w09_request",
]
