"""GG-03 V2 语义 public preflight 的构造、发布与严格回读。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    sha256_text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_generation_candidate_pack import (
    LoadedGenerationCandidatePack,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family import (
    _within,
    require_generation_generalization_k_run_root,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationCodeIdentity,
    GenerationGeneralizationEvaluationFamilyError,
    generation_generalization_observation_content_sha256,
    generation_generalization_sha256_bytes,
    generation_generalization_sha256_file,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationBatch,
    GenerationGeneralizationEvaluationPolicy,
    run_generation_generalization_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_generation_generalization_formal_labels import (
    generation_generalization_observation_key_sha256,
)
from pure_integer_ai.experiments.ph2_generation_generalization_semantic_labels import (
    build_actual_generation_generalization_semantic_projection,
    build_expected_generation_generalization_semantic_projection,
    generation_generalization_semantic_verdict_contract_sha256,
)
from pure_integer_ai.experiments.train_context import TrainContext


SEMANTIC_PUBLIC_DRY_RUN_ARTIFACT_KIND = (
    "PH2_GG03_PUBLIC_SEMANTIC_DRY_RUN_RECEIPT_V2")
SEMANTIC_PUBLIC_DRY_RUN_RECEIPT_NAME = (
    "public-semantic-dry-run.receipt.json")
_RECEIPT_FIELDS = frozenset({
    "artifact_kind", "batch_sha256", "candidate_payload_sha256",
    "code_identity_sha256", "format_version", "host_learning_write_count",
    "label_read_count", "observation_content_sha256s",
    "observation_inventory_sha256", "observation_stable_key_sha256s",
    "policy_sha256", "run_count", "semantic_projection_inventory_sha256",
    "status", "teacher_call_count", "verdict_contract_sha256",
})


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSemanticPublicDryRunReceipt:
    """同一生产 runner 的 public semantic PASS 内容锁。"""

    candidate_payload_sha256: str
    code_identity_sha256: str
    policy_sha256: str
    batch_sha256: str
    observation_inventory_sha256: str
    observation_stable_key_sha256s: tuple[str, ...]
    observation_content_sha256s: tuple[str, ...]
    semantic_projection_inventory_sha256: str
    verdict_contract_sha256: str
    run_count: int
    status: str
    teacher_call_count: int = 0
    label_read_count: int = 0
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        for name in (
                "candidate_payload_sha256", "code_identity_sha256",
                "policy_sha256", "batch_sha256",
                "observation_inventory_sha256",
                "semantic_projection_inventory_sha256",
                "verdict_contract_sha256"):
            sha256_text(
                getattr(self, name), where=f"GG-03 semantic dry run {name}")
        if self.verdict_contract_sha256 != (
                generation_generalization_semantic_verdict_contract_sha256()):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic dry run verdict contract 漂移")
        if type(self.run_count) is not int or self.run_count <= 0:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic dry run count 非法")
        for name in (
                "observation_stable_key_sha256s",
                "observation_content_sha256s"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or len(values) != self.run_count
                    or values != tuple(sorted(values))
                    or len(set(values)) != self.run_count):
                raise GenerationGeneralizationEvaluationFamilyError(
                    f"GG-03 semantic dry run {name} 非法")
            for item in values:
                sha256_text(
                    item, where=f"GG-03 semantic dry run {name} item")
        if self.status != "PASS":
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic public dry-run 未 PASS")
        if any(getattr(self, name) != 0 for name in (
                "teacher_call_count", "label_read_count",
                "host_learning_write_count")):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic public dry-run 零调用/零写失败")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": SEMANTIC_PUBLIC_DRY_RUN_ARTIFACT_KIND,
            "batch_sha256": self.batch_sha256,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "code_identity_sha256": self.code_identity_sha256,
            "format_version": 2,
            "host_learning_write_count": self.host_learning_write_count,
            "label_read_count": self.label_read_count,
            "observation_content_sha256s": list(
                self.observation_content_sha256s),
            "observation_inventory_sha256": (
                self.observation_inventory_sha256),
            "observation_stable_key_sha256s": list(
                self.observation_stable_key_sha256s),
            "policy_sha256": self.policy_sha256,
            "run_count": self.run_count,
            "semantic_projection_inventory_sha256": (
                self.semantic_projection_inventory_sha256),
            "status": self.status,
            "teacher_call_count": self.teacher_call_count,
            "verdict_contract_sha256": self.verdict_contract_sha256,
        }

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "GenerationGeneralizationSemanticPublicDryRunReceipt":
        """从精确 canonical object 恢复 semantic preflight receipt。"""
        raw = exact_dict(
            value, _RECEIPT_FIELDS,
            where="GG-03 semantic public dry-run receipt",
        )
        if (raw["artifact_kind"] != SEMANTIC_PUBLIC_DRY_RUN_ARTIFACT_KIND
                or raw["format_version"] != 2
                or not isinstance(raw["observation_stable_key_sha256s"], list)
                or not isinstance(raw["observation_content_sha256s"], list)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic public dry-run kind/version/list 漂移")
        return cls(
            str(raw["candidate_payload_sha256"]),
            str(raw["code_identity_sha256"]),
            str(raw["policy_sha256"]),
            str(raw["batch_sha256"]),
            str(raw["observation_inventory_sha256"]),
            tuple(str(item) for item in raw[
                "observation_stable_key_sha256s"]),
            tuple(str(item) for item in raw[
                "observation_content_sha256s"]),
            str(raw["semantic_projection_inventory_sha256"]),
            str(raw["verdict_contract_sha256"]),
            raw["run_count"],
            str(raw["status"]),
            raw["teacher_call_count"],
            raw["label_read_count"],
            raw["host_learning_write_count"],
        )


def build_generation_generalization_semantic_public_dry_run_receipt(
        host_ctx: TrainContext,
        loaded: LoadedGenerationCandidatePack,
        observations: tuple[GenerationGeneralizationEvaluationObservation, ...],
        *,
        code_identity: GenerationGeneralizationCodeIdentity,
        policy: GenerationGeneralizationEvaluationPolicy | None = None,
        ) -> GenerationGeneralizationSemanticPublicDryRunReceipt:
    """真实执行 public batch，并要求 actual/expected semantic 全量相等。"""
    if not isinstance(code_identity, GenerationGeneralizationCodeIdentity):
        raise TypeError("GG-03 semantic dry-run code identity 类型错误")
    policy = policy or GenerationGeneralizationEvaluationPolicy()
    batch = run_generation_generalization_evaluation_batch(
        host_ctx, loaded, observations, policy)
    if (not isinstance(batch, GenerationGeneralizationEvaluationBatch)
            or batch.status != "PASS"
            or batch.coverage != INDEPENDENT_VERIFIER_REQUIREMENTS
            or tuple(item.observation for item in batch.runs) != observations):
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run 内部 hard conjunction 未闭合")
    projection_rows = []
    for run in batch.runs:
        actual = build_actual_generation_generalization_semantic_projection(
            run)
        expected = build_expected_generation_generalization_semantic_projection(
            run.observation)
        if actual is None or actual != expected:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic public dry-run projection 未 PASS")
        projection_rows.append({
            "observation_stable_key_sha256": (
                generation_generalization_observation_key_sha256(
                    run.observation)),
            "semantic_projection_sha256": actual.sha256(),
        })
    return GenerationGeneralizationSemanticPublicDryRunReceipt(
        loaded.pack.sha256(),
        code_identity.aggregate_sha256,
        generation_generalization_sha256_bytes(
            canonical_json_bytes(policy.to_dict())),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            list(batch.stable_key()))),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            [item.to_dict() for item in observations])),
        tuple(sorted(
            generation_generalization_observation_key_sha256(item)
            for item in observations)),
        tuple(sorted(
            generation_generalization_observation_content_sha256(item)
            for item in observations)),
        generation_generalization_sha256_bytes(canonical_json_bytes(
            projection_rows)),
        generation_generalization_semantic_verdict_contract_sha256(),
        len(batch.runs),
        "PASS",
        sum(item.teacher_call_count for item in batch.runs),
        sum(item.label_read_count for item in batch.runs),
        sum(item.host_learning_write_count for item in batch.runs),
    )


def publish_generation_generalization_semantic_public_dry_run_receipt(
        receipt: GenerationGeneralizationSemanticPublicDryRunReceipt,
        *,
        run_root: str | Path,
        target_path: str | Path,
        ) -> dict[str, object]:
    """在 K 盘不可覆盖发布不含 surface/label 的 semantic receipt。"""
    if not isinstance(
            receipt, GenerationGeneralizationSemanticPublicDryRunReceipt):
        raise TypeError("GG-03 semantic public dry-run receipt 类型错误")
    root = require_generation_generalization_k_run_root(run_root)
    target = _within(
        root, target_path, where="GG-03 semantic public dry-run receipt")
    if target.exists():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run receipt 已存在")
    write_immutable_json(receipt.to_dict(), target)
    if read_canonical_object(target) != receipt.to_dict():
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic public dry-run receipt 回读漂移")
    return {
        **receipt.to_dict(),
        "receipt_sha256": generation_generalization_sha256_file(target),
    }


def read_generation_generalization_semantic_public_dry_run_receipt(
        path: str | Path,
        ) -> GenerationGeneralizationSemanticPublicDryRunReceipt:
    """严格回读 semantic public preflight receipt。"""
    return GenerationGeneralizationSemanticPublicDryRunReceipt.from_dict(
        read_canonical_object(path))


__all__ = [
    "SEMANTIC_PUBLIC_DRY_RUN_ARTIFACT_KIND",
    "SEMANTIC_PUBLIC_DRY_RUN_RECEIPT_NAME",
    "GenerationGeneralizationSemanticPublicDryRunReceipt",
    "build_generation_generalization_semantic_public_dry_run_receipt",
    "publish_generation_generalization_semantic_public_dry_run_receipt",
    "read_generation_generalization_semantic_public_dry_run_receipt",
]
