"""GG-03 executable generation formal family 的冻结与公开预检编排。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    sha1_text,
    sha256_text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_evaluation_kernel.guard import (
    EvaluationOneShotGuard,
    build_available_guard_for_identity,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    PRIVATE_OWNER_ARTIFACT_KIND,
    GenerationGeneralizationCodeFileIdentity,
    GenerationGeneralizationCodeIdentity,
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationObservationInventoryIdentity,
    GenerationGeneralizationObservationRecordIdentity,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    build_generation_generalization_code_identity,
    double_scan_generation_generalization_observation_inventory,
    generation_generalization_sha256_bytes,
    generation_generalization_sha256_file,
    read_generation_generalization_private_label_owner_receipt,
    scan_generation_generalization_observation_inventory,
    strict_generation_generalization_relative_path,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationBatch,
    GenerationGeneralizationEvaluationPolicy,
    run_generation_generalization_evaluation_batch,
)
from pure_integer_ai.experiments.train_context import TrainContext


FAMILY_ARTIFACT_KIND = "PH2_GG03_EXECUTABLE_EVALUATION_FAMILY_FREEZE_V1"
FAMILY_STATUS = "FROZEN_NOT_RUN_LABELS_UNREAD"
PUBLIC_DRY_RUN_ARTIFACT_KIND = "PH2_GG03_PUBLIC_DRY_RUN_RECEIPT_V1"
FAMILY_MANIFEST_NAME = "family-freeze.json"
FAMILY_GUARD_NAME = "guard.available.json"
FORMAL_PUBLICATION_PATHS = (
    "run.intent.json",
    "run.outcome.json",
    "predictions.seal.json",
    "publication/aggregate.json",
    "publication/decision.json",
    "publication/runtime_receipt.json",
    "publication/failure_seal.json",
)
FORMAL_EXECUTION_ORDER = (
    "VERIFY_FAMILY_AND_LABEL_FREE_INVENTORY",
    "VERIFY_AVAILABLE_GUARD",
    "CONSUME_UNIQUE_GUARD_AND_PUBLISH_INTENT",
    "MATERIALIZE_LABEL_FREE_OBSERVATIONS",
    "RUN_SHARED_E05D_RUNNER_WITHOUT_LABELS",
    "SEAL_PREDICTIONS_AND_RUNTIME_AUDIT",
    "READ_PRIVATE_LABELS",
    "SCORE_FROZEN_REQUIREMENTS",
    "PUBLISH_AGGREGATE_AND_RECEIPT_OR_FAILURE_SEAL",
)


def _within(root: Path, value: str | Path, *, where: str) -> Path:
    """解析路径并拒绝越出显式物理 root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 越出物理 root")
    return path


def _disjoint(left: Path, right: Path) -> None:
    """要求 candidate-visible 与 private-label root 互不包含。"""
    if left == right or left.is_relative_to(right) or right.is_relative_to(left):
        raise GenerationGeneralizationEvaluationFamilyError(
            "candidate-visible 与 private-label root 未物理隔离")


def _git_value(repository: Path, *arguments: str) -> str:
    """读取本地 Git identity，不调用网络或修改配置。"""
    try:
        result = subprocess.run(
            ("git", *arguments), cwd=repository, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family Git identity 不可用") from error
    return result.stdout.strip()


def _published_git_head(repository: Path) -> str:
    """只接受 clean 且 HEAD 与 origin/master 相同的公开提交。"""
    head = _git_value(repository, "rev-parse", "HEAD")
    origin = _git_value(repository, "rev-parse", "origin/master")
    dirty = _git_value(repository, "status", "--porcelain=v1")
    sha1_text(head, where="GG-03 family public head")
    if head != origin or dirty:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family freeze 要求 clean 且 HEAD=origin/master")
    return head


def require_generation_generalization_k_run_root(value: str | Path) -> Path:
    """要求 production family 的全部运行物位于显式 K 盘 root。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 evaluation run root 必须是已存在的 K 盘目录")
    return root


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationPublicDryRunReceipt:
    """同一 E-05D runner 的 label-free public PASS 证明。"""

    candidate_payload_sha256: str
    code_identity_sha256: str
    policy_sha256: str
    batch_sha256: str
    observation_inventory_sha256: str
    run_count: int
    status: str
    teacher_call_count: int = 0
    label_read_count: int = 0
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        for name in (
                "candidate_payload_sha256", "code_identity_sha256",
                "policy_sha256", "batch_sha256",
                "observation_inventory_sha256"):
            sha256_text(getattr(self, name), where=f"GG-03 dry run {name}")
        if type(self.run_count) is not int or self.run_count <= 0:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 dry run count 非法")
        if self.status != "PASS":
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 public dry-run 未 PASS")
        if any(getattr(self, name) != 0 for name in (
                "teacher_call_count", "label_read_count",
                "host_learning_write_count")):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 public dry-run 零调用/零写审计失败")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": PUBLIC_DRY_RUN_ARTIFACT_KIND,
            "batch_sha256": self.batch_sha256,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "code_identity_sha256": self.code_identity_sha256,
            "format_version": 1,
            "host_learning_write_count": self.host_learning_write_count,
            "label_read_count": self.label_read_count,
            "observation_inventory_sha256": (
                self.observation_inventory_sha256),
            "policy_sha256": self.policy_sha256,
            "run_count": self.run_count,
            "status": self.status,
            "teacher_call_count": self.teacher_call_count,
        }

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "GenerationGeneralizationPublicDryRunReceipt":
        """从精确 canonical object 恢复 public dry-run receipt。"""
        if (not isinstance(value, dict) or set(value) != {
                "artifact_kind", "batch_sha256", "candidate_payload_sha256",
                "code_identity_sha256", "format_version",
                "host_learning_write_count", "label_read_count",
                "observation_inventory_sha256", "policy_sha256",
                "run_count", "status", "teacher_call_count",
                }):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 public dry-run receipt 字段漂移")
        if (value["artifact_kind"] != PUBLIC_DRY_RUN_ARTIFACT_KIND
                or value["format_version"] != 1):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 public dry-run receipt kind/version 漂移")
        return cls(
            str(value["candidate_payload_sha256"]),
            str(value["code_identity_sha256"]),
            str(value["policy_sha256"]),
            str(value["batch_sha256"]),
            str(value["observation_inventory_sha256"]),
            value["run_count"],
            str(value["status"]),
            value["teacher_call_count"],
            value["label_read_count"],
            value["host_learning_write_count"],
        )


def build_generation_generalization_public_dry_run_receipt(
        host_ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observations: tuple[GenerationGeneralizationEvaluationObservation, ...],
        *,
        code_identity: GenerationGeneralizationCodeIdentity,
        policy: GenerationGeneralizationEvaluationPolicy | None = None,
        ) -> GenerationGeneralizationPublicDryRunReceipt:
    """直接调用 E-05D shared runner，形成不含 surface 的公开预检 receipt。"""
    if not isinstance(code_identity, GenerationGeneralizationCodeIdentity):
        raise TypeError("GG-03 dry-run code identity 类型错误")
    policy = policy or GenerationGeneralizationEvaluationPolicy()
    batch = run_generation_generalization_evaluation_batch(
        host_ctx, loaded, observations, policy)
    if (not isinstance(batch, GenerationGeneralizationEvaluationBatch)
            or batch.status != "PASS"
            or batch.coverage != INDEPENDENT_VERIFIER_REQUIREMENTS):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 public dry-run 未闭合六路 hard conjunction")
    return GenerationGeneralizationPublicDryRunReceipt(
        loaded.pack.sha256(), code_identity.aggregate_sha256,
        generation_generalization_sha256_bytes(
            canonical_json_bytes(policy.to_dict())),
        generation_generalization_sha256_bytes(
            canonical_json_bytes(list(batch.stable_key()))),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict() for item in observations])),
        len(batch.runs), batch.status,
        sum(item.teacher_call_count for item in batch.runs),
        sum(item.label_read_count for item in batch.runs),
        sum(item.host_learning_write_count for item in batch.runs),
    )


def publish_generation_generalization_public_dry_run_receipt(
        receipt: GenerationGeneralizationPublicDryRunReceipt,
        *,
        run_root: str | Path,
        target_path: str | Path,
        ) -> dict[str, object]:
    """在 K 盘不可覆盖发布不含 surface/label 的 public dry-run receipt。"""
    if not isinstance(receipt, GenerationGeneralizationPublicDryRunReceipt):
        raise TypeError("GG-03 public dry-run receipt 类型错误")
    root = require_generation_generalization_k_run_root(run_root)
    target = _within(root, target_path, where="GG-03 public dry-run receipt")
    if target.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 public dry-run receipt 已存在")
    write_immutable_json(receipt.to_dict(), target)
    if read_canonical_object(target) != receipt.to_dict():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 public dry-run receipt 回读漂移")
    return {
        **receipt.to_dict(),
        "receipt_sha256": generation_generalization_sha256_file(target),
    }


def read_generation_generalization_public_dry_run_receipt(
        path: str | Path,
        ) -> GenerationGeneralizationPublicDryRunReceipt:
    """严格回读 K 盘 public dry-run receipt。"""
    return GenerationGeneralizationPublicDryRunReceipt.from_dict(
        read_canonical_object(path))


def _threshold_contract() -> dict[str, object]:
    """冻结无分数阈值的六路逻辑 hard conjunction。"""
    return {
        "fail_condition": "ANY_REQUIREMENT_FAIL",
        "hidden_or_numeric_threshold_count": 0,
        "kind": "NO_SCORE_THRESHOLD_ALL_REQUIREMENTS_HARD_CONJUNCT",
        "ne_condition": "NO_FAIL_AND_ANY_REQUIREMENT_NE",
        "pass_condition": "EVERY_REQUIREMENT_PASS",
        "requirements": list(INDEPENDENT_VERIFIER_REQUIREMENTS),
        "status_precedence": ["FAIL", "NE", "PASS"],
    }


def _aggregate_contract() -> dict[str, object]:
    """冻结未来 E-06 aggregate、receipt 与 failure seal 的公开身份。"""
    core = {
        "aggregate_statuses": ["PASS", "FAIL", "NE"],
        "failure_seal_statuses": ["FAIL", "NE"],
        "immutable_paths": list(FORMAL_PUBLICATION_PATHS),
        "requirement_order": list(INDEPENDENT_VERIFIER_REQUIREMENTS),
        "runtime_receipt_statuses": ["PASS"],
        "surface_or_label_fields_public": 0,
    }
    return {
        **core,
        "sealed_verdict_identity_sha256": (
            generation_generalization_sha256_bytes(canonical_json_bytes(core))),
    }


def generation_generalization_verdict_contract_sha256() -> str:
    """返回 private owner 必须在读 label 前绑定的 sealed verdict identity。"""
    return str(_aggregate_contract()["sealed_verdict_identity_sha256"])


def build_generation_generalization_evaluation_family_freeze(
        *,
        public_head_sha1: str,
        candidate_manifest_relative_path: str,
        candidate_manifest_sha256: str,
        candidate_manifest_size_bytes: int,
        candidate_payload_sha256: str,
        candidate_training_artifact_sha256: str,
        code_identity: GenerationGeneralizationCodeIdentity,
        policy: GenerationGeneralizationEvaluationPolicy,
        observation_inventory_relative_path: str,
        observation_inventory: GenerationGeneralizationObservationInventoryIdentity,
        private_owner_receipt_relative_path: str,
        private_owner_receipt_sha256: str,
        private_owner: GenerationGeneralizationPrivateLabelOwnerReceipt,
        public_dry_run: GenerationGeneralizationPublicDryRunReceipt,
        ) -> dict[str, object]:
    """从已核验 identity 构造不含路径机密和 private label 的 freeze。"""
    sha1_text(public_head_sha1, where="GG-03 family public head")
    for name, value in (
            ("candidate manifest", candidate_manifest_sha256),
            ("candidate payload", candidate_payload_sha256),
            ("candidate training artifact", candidate_training_artifact_sha256),
            ("private owner receipt", private_owner_receipt_sha256)):
        sha256_text(value, where=f"GG-03 family {name}")
    if (type(candidate_manifest_size_bytes) is not int
            or candidate_manifest_size_bytes <= 0):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 candidate manifest bytes 非法")
    for where, value in (
            ("GG-03 candidate manifest", candidate_manifest_relative_path),
            ("GG-03 Observation inventory", observation_inventory_relative_path),
            ("GG-03 private owner receipt", private_owner_receipt_relative_path)):
        strict_generation_generalization_relative_path(value, where=where)
    if (private_owner.observation_inventory_sha256
            != observation_inventory.transport_sha256
            or private_owner.label_record_count != observation_inventory.record_count):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private label owner 与 Observation inventory 不闭合")
    policy_sha = generation_generalization_sha256_bytes(
        canonical_json_bytes(policy.to_dict()))
    if (public_dry_run.candidate_payload_sha256 != candidate_payload_sha256
            or public_dry_run.code_identity_sha256
            != code_identity.aggregate_sha256
            or public_dry_run.policy_sha256 != policy_sha):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 public dry-run 未绑定当前 candidate/code/policy")
    aggregate = _aggregate_contract()
    if private_owner.verdict_contract_sha256 != aggregate[
            "sealed_verdict_identity_sha256"]:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 private owner verdict contract 漂移")
    core = {
        "aggregate_contract": aggregate,
        "artifact_kind": FAMILY_ARTIFACT_KIND,
        "candidate": {
            "manifest_relative_path": candidate_manifest_relative_path,
            "manifest_sha256": candidate_manifest_sha256,
            "manifest_size_bytes": candidate_manifest_size_bytes,
            "payload_sha256": candidate_payload_sha256,
            "training_artifact_sha256": candidate_training_artifact_sha256,
        },
        "code_identity": code_identity.to_dict(),
        "execution_order": list(FORMAL_EXECUTION_ORDER),
        "format_version": 1,
        "formal_run_count": 0,
        "host_learning_write_count": 0,
        "label_read_count_before_prediction_seal": 0,
        "observation_inventory": {
            **observation_inventory.to_dict(),
            "double_pass_equal": 1,
            "relative_path": observation_inventory_relative_path,
        },
        "physical_layout": {
            "candidate_visible_root": "candidate-visible",
            "private_label_root": "private-label-owner",
            "roots_disjoint": 1,
        },
        "policy": policy.to_dict(),
        "policy_sha256": policy_sha,
        "private_owner": {
            "label_commitment_sha256": private_owner.label_commitment_sha256,
            "owner_receipt_relative_path": private_owner_receipt_relative_path,
            "owner_receipt_sha256": private_owner_receipt_sha256,
            "status": private_owner.status,
        },
        "public_dry_run": public_dry_run.to_dict(),
        "public_head_sha1": public_head_sha1,
        "runner_contract": {
            "formal_and_public_runner_symbol": (
                "run_generation_generalization_evaluation_batch"),
            "parallel_private_generation_logic_allowed": 0,
            "selection_inputs": [
                "CANDIDATE_PACK", "LABEL_FREE_OBSERVATION", "FROZEN_POLICY"],
        },
        "status": FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
        "threshold_contract": _threshold_contract(),
        "unique_formal_run_limit": 1,
    }
    return {
        **core,
        "family_commitment_sha256": (
            generation_generalization_sha256_bytes(canonical_json_bytes(core))),
    }


def _prepare_generation_generalization_evaluation_family_freeze(
        *,
        repository_root: str | Path,
        run_root: str | Path,
        candidate_visible_root: str | Path,
        private_label_root: str | Path,
        loaded_candidate: LoadedGenerationCandidatePack,
        observation_inventory_path: str | Path,
        private_owner_receipt_path: str | Path,
        public_dry_run: GenerationGeneralizationPublicDryRunReceipt,
        policy: GenerationGeneralizationEvaluationPolicy,
        resource_ceiling: GenerationGeneralizationEvaluationBudget,
        ) -> tuple[dict[str, object], str]:
    """核验 production 物理边界并准备 freeze；整个过程不打开 label 文件。"""
    repository = Path(repository_root).resolve()
    root = require_generation_generalization_k_run_root(run_root)
    candidate_root = _within(
        root, candidate_visible_root, where="GG-03 candidate-visible root")
    private_root = _within(
        root, private_label_root, where="GG-03 private-label root")
    if (not candidate_root.is_dir() or not private_root.is_dir()
            or candidate_root.name != "candidate-visible"
            or private_root.name != "private-label-owner"):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 physical roots 名称或存在性漂移")
    _disjoint(candidate_root, private_root)
    if not isinstance(loaded_candidate, LoadedGenerationCandidatePack):
        raise TypeError("GG-03 loaded candidate 类型错误")
    candidate_manifest = _within(
        candidate_root, loaded_candidate.manifest_path,
        where="GG-03 candidate manifest")
    observation_path = _within(
        candidate_root, observation_inventory_path,
        where="GG-03 Observation inventory")
    owner_path = _within(
        private_root, private_owner_receipt_path,
        where="GG-03 private owner receipt")
    inventory = double_scan_generation_generalization_observation_inventory(
        observation_path, resource_ceiling=resource_ceiling)
    owner, owner_sha = read_generation_generalization_private_label_owner_receipt(
        owner_path)
    code = build_generation_generalization_code_identity(repository)
    freeze = build_generation_generalization_evaluation_family_freeze(
        public_head_sha1=_published_git_head(repository),
        candidate_manifest_relative_path=(
            candidate_manifest.relative_to(candidate_root).as_posix()),
        candidate_manifest_sha256=(
            generation_generalization_sha256_file(candidate_manifest)),
        candidate_manifest_size_bytes=candidate_manifest.stat().st_size,
        candidate_payload_sha256=loaded_candidate.pack.sha256(),
        candidate_training_artifact_sha256=(
            loaded_candidate.pack.training_artifact_sha256),
        code_identity=code,
        policy=policy,
        observation_inventory_relative_path=(
            observation_path.relative_to(candidate_root).as_posix()),
        observation_inventory=inventory,
        private_owner_receipt_relative_path=(
            owner_path.relative_to(private_root).as_posix()),
        private_owner_receipt_sha256=owner_sha,
        private_owner=owner,
        public_dry_run=public_dry_run,
    )
    return freeze, owner_sha


def publish_generation_generalization_evaluation_family_freeze(
        *,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """在 K 盘不可覆盖发布 family freeze 与共享 available guard。"""
    root = require_generation_generalization_k_run_root(arguments["run_root"])
    target = _within(root, target_dir, where="GG-03 family target")
    if target.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family target 已存在")
    freeze, owner_sha = (
        _prepare_generation_generalization_evaluation_family_freeze(
            **arguments))
    temporary = Path(tempfile.mkdtemp(
        prefix=".gg03-family-building-", dir=target.parent)).resolve()
    try:
        manifest_path = temporary / FAMILY_MANIFEST_NAME
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_line(freeze))
        manifest_sha = generation_generalization_sha256_file(manifest_path)
        guard = build_available_guard_for_identity(
            manifest_sha,
            str(freeze["family_commitment_sha256"]),
            owner_sha,
            str(freeze["candidate"]["payload_sha256"]),
            str(freeze["code_identity"]["aggregate_sha256"]),
        )
        write_immutable_json(guard.to_dict(), temporary / FAMILY_GUARD_NAME)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        **freeze,
        "available_guard_sha256": guard.sha256(),
        "manifest_sha256": manifest_sha,
    }


def read_generation_generalization_evaluation_family_freeze(
        family_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """严格回读 family 并重算 candidate/code/inventory/owner 安全 identity。"""
    run_root = require_generation_generalization_k_run_root(arguments["run_root"])
    root = _within(run_root, family_dir, where="GG-03 family directory")
    freeze, owner_sha = (
        _prepare_generation_generalization_evaluation_family_freeze(
            **arguments))
    manifest_path = root / FAMILY_MANIFEST_NAME
    guard_path = root / FAMILY_GUARD_NAME
    if (not root.is_dir()
            or {item.name for item in root.iterdir()}
            != {FAMILY_MANIFEST_NAME, FAMILY_GUARD_NAME}):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family physical inventory 漂移")
    try:
        payload = manifest_path.read_bytes()
        stored = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family manifest 不可读") from error
    if (canonical_json_line(stored) != payload or stored != freeze):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family manifest 与 live identity 漂移")
    manifest_sha = generation_generalization_sha256_bytes(payload)
    guard = EvaluationOneShotGuard.from_dict(read_canonical_object(guard_path))
    expected_guard = build_available_guard_for_identity(
        manifest_sha,
        str(freeze["family_commitment_sha256"]),
        owner_sha,
        str(freeze["candidate"]["payload_sha256"]),
        str(freeze["code_identity"]["aggregate_sha256"]),
    )
    if guard != expected_guard:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 family available guard 漂移")
    return {
        **freeze,
        "available_guard_sha256": guard.sha256(),
        "manifest_sha256": manifest_sha,
    }


__all__ = [
    "FAMILY_ARTIFACT_KIND",
    "FAMILY_GUARD_NAME",
    "FAMILY_MANIFEST_NAME",
    "FAMILY_STATUS",
    "FORMAL_EXECUTION_ORDER",
    "FORMAL_PUBLICATION_PATHS",
    "PRIVATE_OWNER_ARTIFACT_KIND",
    "GenerationGeneralizationCodeFileIdentity",
    "GenerationGeneralizationCodeIdentity",
    "GenerationGeneralizationEvaluationFamilyError",
    "GenerationGeneralizationObservationInventoryIdentity",
    "GenerationGeneralizationObservationRecordIdentity",
    "GenerationGeneralizationPrivateLabelOwnerReceipt",
    "GenerationGeneralizationPublicDryRunReceipt",
    "build_generation_generalization_code_identity",
    "build_generation_generalization_evaluation_family_freeze",
    "build_generation_generalization_public_dry_run_receipt",
    "double_scan_generation_generalization_observation_inventory",
    "generation_generalization_verdict_contract_sha256",
    "publish_generation_generalization_evaluation_family_freeze",
    "publish_generation_generalization_public_dry_run_receipt",
    "read_generation_generalization_evaluation_family_freeze",
    "read_generation_generalization_public_dry_run_receipt",
    "read_generation_generalization_private_label_owner_receipt",
    "require_generation_generalization_k_run_root",
    "scan_generation_generalization_observation_inventory",
]
