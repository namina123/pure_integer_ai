"""GG-03 V2 semantic formal family 的冻结、发布与严格回读。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    _disjoint,
    _published_git_head,
    _within,
    require_generation_generalization_k_run_root,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationCodeIdentity,
    GenerationGeneralizationEvaluationFamilyError,
    GenerationGeneralizationObservationInventoryIdentity,
    GenerationGeneralizationPrivateLabelOwnerReceipt,
    build_generation_generalization_code_identity_for_roots,
    double_scan_generation_generalization_observation_inventory,
    generation_generalization_sha256_bytes,
    generation_generalization_sha256_file,
    read_generation_generalization_private_label_owner_receipt,
    strict_generation_generalization_relative_path,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationBudget,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationPolicy,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    SEMANTIC_LABEL_ARTIFACT_KIND,
    generation_generalization_semantic_verdict_contract_sha256,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_preflight import (
    SEMANTIC_PUBLIC_DRY_RUN_RECEIPT_NAME,
    GenerationGeneralizationSemanticPublicDryRunReceipt,
    read_generation_generalization_semantic_public_dry_run_receipt,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_protocol import (
    SEMANTIC_FORMAL_AGGREGATE_ARTIFACT_KIND,
    SEMANTIC_FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND,
    SEMANTIC_FORMAL_FAILURE_SEAL_ARTIFACT_KIND,
    SEMANTIC_FORMAL_RUNTIME_RECEIPT_ARTIFACT_KIND,
    SEMANTIC_PREDICTION_SEAL_ARTIFACT_KIND,
)


SEMANTIC_FAMILY_ARTIFACT_KIND = (
    "PH2_GG03_EXECUTABLE_SEMANTIC_EVALUATION_FAMILY_FREEZE_V2")
SEMANTIC_FAMILY_STATUS = "FROZEN_NOT_RUN_SEMANTIC_LABELS_UNREAD"
SEMANTIC_FAMILY_MANIFEST_NAME = "family-freeze.json"
SEMANTIC_FAMILY_GUARD_NAME = "guard.available.json"
SEMANTIC_FORMAL_PUBLICATION_PATHS = (
    "run.intent.json",
    "run.outcome.json",
    "predictions.seal.json",
    "publication/aggregate.json",
    "publication/decision.json",
    "publication/runtime_receipt.json",
    "publication/failure_seal.json",
    "publication/failure_diagnostic.json",
)
SEMANTIC_FORMAL_EXECUTION_ORDER = (
    "VERIFY_SEMANTIC_FAMILY_AND_LABEL_FREE_INVENTORY",
    "VERIFY_AVAILABLE_GUARD",
    "CONSUME_UNIQUE_GUARD_AND_PUBLISH_INTENT",
    "MATERIALIZE_LABEL_FREE_OBSERVATIONS",
    "RUN_SHARED_E05D_RUNNER_WITHOUT_LABELS",
    "PROJECT_AND_SEAL_SEMANTIC_PREDICTIONS",
    "READ_PRIVATE_SEMANTIC_LABELS",
    "SCORE_FROZEN_REQUIREMENTS_AND_SEMANTIC_CONTRACT",
    "PUBLISH_SEMANTIC_AGGREGATE_AND_RECEIPT_OR_FAILURE_SEAL",
)
_SEMANTIC_CODE_ROOT_MODULES = (
    "pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner",
    "pure_integer_ai.experiments.ph2_generation_generalization_semantic_family",
    "pure_integer_ai.experiments.ph2_generation_generalization_semantic_formal_runner",
    "pure_integer_ai.experiments.ph2_generation_generalization_semantic_preflight",
)


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


def _semantic_aggregate_contract() -> dict[str, object]:
    """冻结 V2 semantic aggregate、receipt 和 failure seal 公开身份。"""
    return {
        "aggregate_artifact_kind": SEMANTIC_FORMAL_AGGREGATE_ARTIFACT_KIND,
        "aggregate_statuses": ["PASS", "FAIL", "NE"],
        "failure_diagnostic": {
            "artifact_kind": (
                SEMANTIC_FORMAL_FAILURE_DIAGNOSTIC_ARTIFACT_KIND),
            "message_or_input_field_count": 0,
        },
        "failure_seal_artifact_kind": (
            SEMANTIC_FORMAL_FAILURE_SEAL_ARTIFACT_KIND),
        "failure_seal_statuses": ["FAIL", "NE"],
        "immutable_paths": list(SEMANTIC_FORMAL_PUBLICATION_PATHS),
        "label_artifact_kind": SEMANTIC_LABEL_ARTIFACT_KIND,
        "prediction_seal_artifact_kind": (
            SEMANTIC_PREDICTION_SEAL_ARTIFACT_KIND),
        "projection_verdict_contract_sha256": (
            generation_generalization_semantic_verdict_contract_sha256()),
        "requirement_order": list(INDEPENDENT_VERIFIER_REQUIREMENTS),
        "runtime_receipt_artifact_kind": (
            SEMANTIC_FORMAL_RUNTIME_RECEIPT_ARTIFACT_KIND),
        "runtime_receipt_statuses": ["PASS"],
        "surface_or_label_fields_public": 0,
    }


def build_generation_generalization_semantic_evaluation_family_freeze(
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
        public_dry_run_receipt_relative_path: str,
        public_dry_run_receipt_sha256: str,
        public_dry_run: GenerationGeneralizationSemanticPublicDryRunReceipt,
        ) -> dict[str, object]:
    """从已核验 identity 构造不含路径机密和 private label 的 V2 freeze。"""
    sha1_text(public_head_sha1, where="GG-03 semantic family public head")
    for name, value in (
            ("candidate manifest", candidate_manifest_sha256),
            ("candidate payload", candidate_payload_sha256),
            ("candidate training artifact", candidate_training_artifact_sha256),
            ("private owner receipt", private_owner_receipt_sha256),
            ("public semantic dry-run receipt",
             public_dry_run_receipt_sha256)):
        sha256_text(value, where=f"GG-03 semantic family {name}")
    if (type(candidate_manifest_size_bytes) is not int
            or candidate_manifest_size_bytes <= 0):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic candidate manifest bytes 非法")
    for where, value in (
            ("GG-03 semantic candidate manifest",
             candidate_manifest_relative_path),
            ("GG-03 semantic Observation inventory",
             observation_inventory_relative_path),
            ("GG-03 semantic private owner receipt",
             private_owner_receipt_relative_path),
            ("GG-03 semantic public dry-run receipt",
             public_dry_run_receipt_relative_path)):
        strict_generation_generalization_relative_path(value, where=where)
    if Path(public_dry_run_receipt_relative_path).name != (
            SEMANTIC_PUBLIC_DRY_RUN_RECEIPT_NAME):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run receipt 文件名漂移")
    if (private_owner.observation_inventory_sha256
            != observation_inventory.transport_sha256
            or private_owner.label_record_count
            != observation_inventory.record_count):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic private owner 与 Observation inventory 不闭合")
    verdict_contract = (
        generation_generalization_semantic_verdict_contract_sha256())
    if private_owner.verdict_contract_sha256 != verdict_contract:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic private owner verdict contract 漂移")
    policy_sha = generation_generalization_sha256_bytes(
        canonical_json_bytes(policy.to_dict()))
    if (public_dry_run.candidate_payload_sha256 != candidate_payload_sha256
            or public_dry_run.code_identity_sha256
            != code_identity.aggregate_sha256
            or public_dry_run.policy_sha256 != policy_sha
            or public_dry_run.verdict_contract_sha256 != verdict_contract):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run 未绑定 candidate/code/policy/contract")
    formal_stable_keys = {
        item.stable_key_sha256 for item in observation_inventory.records}
    if formal_stable_keys & set(
            public_dry_run.observation_stable_key_sha256s):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic formal Observation 复用了 public dry-run 输入")
    formal_content = {
        item.content_sha256 for item in observation_inventory.records}
    if formal_content & set(public_dry_run.observation_content_sha256s):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic formal Observation 复用了 public dry-run 内容")
    aggregate_contract = _semantic_aggregate_contract()
    core = {
        "aggregate_contract": aggregate_contract,
        "artifact_kind": SEMANTIC_FAMILY_ARTIFACT_KIND,
        "candidate": {
            "manifest_relative_path": candidate_manifest_relative_path,
            "manifest_sha256": candidate_manifest_sha256,
            "manifest_size_bytes": candidate_manifest_size_bytes,
            "payload_sha256": candidate_payload_sha256,
            "training_artifact_sha256": candidate_training_artifact_sha256,
        },
        "code_identity": code_identity.to_dict(),
        "contamination_audit": {
            "formal_public_content_overlap_count": 0,
            "formal_public_stable_key_overlap_count": 0,
            "policy": (
                "FORMAL_OBSERVATIONS_AND_CONTENT_DISJOINT_FROM_PUBLIC_PREFLIGHT"),
        },
        "execution_order": list(SEMANTIC_FORMAL_EXECUTION_ORDER),
        "format_version": 2,
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
            "label_commitment_sha256": (
                private_owner.label_commitment_sha256),
            "owner_receipt_relative_path": private_owner_receipt_relative_path,
            "owner_receipt_sha256": private_owner_receipt_sha256,
            "status": private_owner.status,
            "verdict_contract_sha256": verdict_contract,
        },
        "public_semantic_dry_run": {
            **public_dry_run.to_dict(),
            "receipt_relative_path": public_dry_run_receipt_relative_path,
            "receipt_sha256": public_dry_run_receipt_sha256,
        },
        "public_head_sha1": public_head_sha1,
        "runner_contract": {
            "formal_and_public_runner_symbol": (
                "run_generation_generalization_evaluation_batch"),
            "parallel_private_generation_logic_allowed": 0,
            "semantic_projection_symbol": (
                "build_actual_generation_generalization_semantic_projection"),
            "selection_inputs": [
                "CANDIDATE_PACK", "LABEL_FREE_OBSERVATION", "FROZEN_POLICY"],
        },
        "status": SEMANTIC_FAMILY_STATUS,
        "teacher_api_llm_call_count": 0,
        "threshold_contract": _threshold_contract(),
        "unique_formal_run_limit": 1,
    }
    return {
        **core,
        "family_commitment_sha256": (
            generation_generalization_sha256_bytes(canonical_json_bytes(core))),
    }


def _prepare_generation_generalization_semantic_family_freeze(
        *,
        repository_root: str | Path,
        run_root: str | Path,
        candidate_visible_root: str | Path,
        private_label_root: str | Path,
        loaded_candidate: LoadedGenerationCandidatePack,
        observation_inventory_path: str | Path,
        private_owner_receipt_path: str | Path,
        public_dry_run_receipt_path: str | Path,
        policy: GenerationGeneralizationEvaluationPolicy,
        resource_ceiling: GenerationGeneralizationEvaluationBudget,
        ) -> tuple[dict[str, object], str]:
    """核验 V2 production 边界并准备 freeze，全程不打开 label 文件。"""
    repository = Path(repository_root).resolve()
    root = require_generation_generalization_k_run_root(run_root)
    candidate_root = _within(
        root, candidate_visible_root,
        where="GG-03 semantic candidate-visible root")
    private_root = _within(
        root, private_label_root,
        where="GG-03 semantic private-label root")
    if (not candidate_root.is_dir() or not private_root.is_dir()
            or candidate_root.name != "candidate-visible"
            or private_root.name != "private-label-owner"):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic physical roots 名称或存在性漂移")
    _disjoint(candidate_root, private_root)
    if not isinstance(loaded_candidate, LoadedGenerationCandidatePack):
        raise TypeError("GG-03 semantic loaded candidate 类型错误")
    candidate_manifest = _within(
        candidate_root, loaded_candidate.manifest_path,
        where="GG-03 semantic candidate manifest")
    observation_path = _within(
        candidate_root, observation_inventory_path,
        where="GG-03 semantic Observation inventory")
    owner_path = _within(
        private_root, private_owner_receipt_path,
        where="GG-03 semantic private owner receipt")
    public_preflight_root = root / "public-preflight"
    public_dry_run_path = _within(
        root, public_dry_run_receipt_path,
        where="GG-03 semantic public dry-run receipt")
    if (public_dry_run_path.parent != public_preflight_root
            or public_dry_run_path.name
            != SEMANTIC_PUBLIC_DRY_RUN_RECEIPT_NAME
            or public_dry_run_path.is_relative_to(private_root)):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run receipt 物理布局漂移")
    inventory = double_scan_generation_generalization_observation_inventory(
        observation_path, resource_ceiling=resource_ceiling)
    owner, owner_sha = read_generation_generalization_private_label_owner_receipt(
        owner_path)
    public_dry_run = (
        read_generation_generalization_semantic_public_dry_run_receipt(
            public_dry_run_path))
    public_dry_run_sha = generation_generalization_sha256_file(
        public_dry_run_path)
    code = build_generation_generalization_code_identity_for_roots(
        repository, _SEMANTIC_CODE_ROOT_MODULES)
    freeze = build_generation_generalization_semantic_evaluation_family_freeze(
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
        public_dry_run_receipt_relative_path=(
            public_dry_run_path.relative_to(root).as_posix()),
        public_dry_run_receipt_sha256=public_dry_run_sha,
        public_dry_run=public_dry_run,
    )
    return freeze, owner_sha


def publish_generation_generalization_semantic_evaluation_family_freeze(
        *,
        target_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """在 K 盘不可覆盖发布 V2 family freeze 与 available guard。"""
    root = require_generation_generalization_k_run_root(arguments["run_root"])
    target = _within(root, target_dir, where="GG-03 semantic family target")
    if target.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic family target 已存在")
    freeze, owner_sha = (
        _prepare_generation_generalization_semantic_family_freeze(
            **arguments))
    temporary = Path(tempfile.mkdtemp(
        prefix=".gg03-semantic-family-building-",
        dir=target.parent,
    )).resolve()
    try:
        manifest_path = temporary / SEMANTIC_FAMILY_MANIFEST_NAME
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
        write_immutable_json(
            guard.to_dict(), temporary / SEMANTIC_FAMILY_GUARD_NAME)
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


def read_generation_generalization_semantic_evaluation_family_freeze(
        family_dir: str | Path,
        **arguments: object,
        ) -> dict[str, object]:
    """严格回读 V2 family 并重算 live identity 与 available guard。"""
    run_root = require_generation_generalization_k_run_root(
        arguments["run_root"])
    root = _within(run_root, family_dir, where="GG-03 semantic family dir")
    freeze, owner_sha = (
        _prepare_generation_generalization_semantic_family_freeze(
            **arguments))
    manifest_path = root / SEMANTIC_FAMILY_MANIFEST_NAME
    guard_path = root / SEMANTIC_FAMILY_GUARD_NAME
    if (not root.is_dir()
            or {item.name for item in root.iterdir()} != {
                SEMANTIC_FAMILY_MANIFEST_NAME,
                SEMANTIC_FAMILY_GUARD_NAME,
            }):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic family physical inventory 漂移")
    try:
        payload = manifest_path.read_bytes()
        stored = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic family manifest 不可读") from error
    if (canonical_json_line(stored) != payload or stored != freeze):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic family manifest 与 live identity 漂移")
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
            "GG-03 semantic family available guard 漂移")
    return {
        **freeze,
        "available_guard_sha256": guard.sha256(),
        "manifest_sha256": manifest_sha,
    }


__all__ = [
    "SEMANTIC_FAMILY_ARTIFACT_KIND",
    "SEMANTIC_FAMILY_GUARD_NAME",
    "SEMANTIC_FAMILY_MANIFEST_NAME",
    "SEMANTIC_FAMILY_STATUS",
    "SEMANTIC_FORMAL_EXECUTION_ORDER",
    "SEMANTIC_FORMAL_PUBLICATION_PATHS",
    "build_generation_generalization_semantic_evaluation_family_freeze",
    "publish_generation_generalization_semantic_evaluation_family_freeze",
    "read_generation_generalization_semantic_evaluation_family_freeze",
]
