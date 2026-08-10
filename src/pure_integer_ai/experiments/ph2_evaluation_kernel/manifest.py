"""Immutable family manifest binding kernel, plugin, source, owner and gates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    positive,
    read_canonical_object,
    sha1_text,
    sha256_text,
    string_tuple,
    text,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_EVALUATOR_STAGES,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    build_v2_private_family_registration,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.owner_receipt import (
    EvaluationOwnerBinding,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.records import (
    EvaluationKernelContractError,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.source_binding import (
    EvaluationSourceBinding,
)


EVALUATION_KERNEL_MANIFEST_KIND = "PH2_EVALUATION_KERNEL_FAMILY_MANIFEST"
EVALUATION_KERNEL_MANIFEST_VERSION = "PH2-EVALUATION-KERNEL-MANIFEST-V1"
EVALUATION_NE_POLICIES = ("BLOCK",)
EVALUATION_BLOCKED_POLICIES = ("SEAL_AND_STOP",)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class EvaluationThreshold:
    """One preregistered no-compromise hard-conjunct threshold."""

    result_key: str
    min_pass_numerator: int
    min_pass_denominator: int
    max_failed_count: int = 0
    ne_policy: str = "BLOCK"
    blocked_policy: str = "SEAL_AND_STOP"

    def __post_init__(self) -> None:
        text(self.result_key, where="evaluation threshold result_key")
        positive(self.min_pass_numerator, where="threshold numerator")
        positive(self.min_pass_denominator, where="threshold denominator")
        if self.min_pass_numerator > self.min_pass_denominator:
            raise EvaluationKernelContractError("evaluation threshold ratio exceeds one")
        if type(self.max_failed_count) is not int or self.max_failed_count != 0:
            raise EvaluationKernelContractError(
                "evaluation threshold may not tolerate failed records")
        if self.ne_policy not in EVALUATION_NE_POLICIES:
            raise EvaluationKernelContractError("evaluation NE policy was weakened")
        if self.blocked_policy not in EVALUATION_BLOCKED_POLICIES:
            raise EvaluationKernelContractError("evaluation BLOCKED policy was weakened")

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_policy": self.blocked_policy,
            "max_failed_count": self.max_failed_count,
            "min_pass_denominator": self.min_pass_denominator,
            "min_pass_numerator": self.min_pass_numerator,
            "ne_policy": self.ne_policy,
            "result_key": self.result_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationThreshold":
        raw = exact_dict(value, {
            "blocked_policy", "max_failed_count", "min_pass_denominator",
            "min_pass_numerator", "ne_policy", "result_key",
        }, where="EvaluationThreshold")
        return cls(
            str(raw["result_key"]), raw["min_pass_numerator"],
            raw["min_pass_denominator"], raw["max_failed_count"],
            str(raw["ne_policy"]), str(raw["blocked_policy"]),
        )


def _dimension_declaration_value(
        bearing_dimension_keys: tuple[str, ...],
        generation_hard_conjunct_key: str,
        support_dimension_keys: tuple[str, ...],
        thresholds: tuple[EvaluationThreshold, ...],
        ) -> dict[str, Any]:
    hard = (*bearing_dimension_keys, generation_hard_conjunct_key,
            *support_dimension_keys)
    return {
        "bearing_dimension_keys": list(bearing_dimension_keys),
        "generation_hard_conjunct_key": generation_hard_conjunct_key,
        "hard_conjunct_keys": list(hard),
        "support_dimension_keys": list(support_dimension_keys),
        "thresholds": [item.to_dict() for item in thresholds],
    }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class EvaluationKernelManifest:
    """The only family identity accepted by the generic runtime."""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    stage_key: str
    family_key: str
    revision_key: str
    public_head_sha1: str
    family_commitment: str
    kernel_semantic_sha256: str
    stage_manifest_sha256: str
    plugin: EvaluationPluginDeclaration
    source_binding: EvaluationSourceBinding
    owner_binding: EvaluationOwnerBinding
    candidate_artifact_sha256: str
    code_freeze_sha256: str
    transport_family_commitment: str
    resource_budget: V2EvaluatorResourceBudget
    dimension_declaration_sha256: str
    consumed_lineage_sha256: str
    bearing_dimension_keys: tuple[str, ...]
    generation_hard_conjunct_key: str
    support_dimension_keys: tuple[str, ...]
    thresholds: tuple[EvaluationThreshold, ...]
    status: str = "FAMILY_MANIFEST_FROZEN"

    def __post_init__(self) -> None:
        if (type(self.format_version) is not int or self.format_version != 1
                or self.artifact_kind != EVALUATION_KERNEL_MANIFEST_KIND
                or self.artifact_version != EVALUATION_KERNEL_MANIFEST_VERSION):
            raise EvaluationKernelContractError("evaluation manifest identity drifted")
        for name in ("release_key", "family_key", "revision_key"):
            text(getattr(self, name), where=f"evaluation manifest {name}")
        sha1_text(self.public_head_sha1, where="evaluation manifest public_head_sha1")
        if self.stage_key not in V2_EVALUATOR_STAGES:
            raise EvaluationKernelContractError("evaluation manifest stage is invalid")
        for name in (
                "family_commitment", "kernel_semantic_sha256",
                "stage_manifest_sha256", "candidate_artifact_sha256",
                "code_freeze_sha256", "transport_family_commitment",
                "dimension_declaration_sha256", "consumed_lineage_sha256"):
            sha256_text(getattr(self, name), where=f"evaluation manifest {name}")
        if (not isinstance(self.plugin, EvaluationPluginDeclaration)
                or self.plugin.stage_key != self.stage_key):
            raise EvaluationKernelContractError("evaluation manifest plugin drifted")
        if not isinstance(self.source_binding, EvaluationSourceBinding):
            raise EvaluationKernelContractError("evaluation source binding is invalid")
        if not isinstance(self.owner_binding, EvaluationOwnerBinding):
            raise EvaluationKernelContractError("evaluation owner binding is invalid")
        if self.owner_binding.owner_key != "PH2_V2_PRIVATE_EVALUATOR":
            raise EvaluationKernelContractError("evaluation private owner key drifted")
        if not isinstance(self.resource_budget, V2EvaluatorResourceBudget):
            raise EvaluationKernelContractError("evaluation resource budget is invalid")
        if self.owner_binding.source_ref_count != self.source_binding.record_count:
            raise EvaluationKernelContractError(
                "evaluation owner and source binding counts do not close")
        bearings = string_tuple(
            self.bearing_dimension_keys, where="evaluation bearing dimensions")
        support = string_tuple(
            self.support_dimension_keys, where="evaluation support dimensions")
        text(self.generation_hard_conjunct_key, where="evaluation generation key")
        if bearings != self.bearing_dimension_keys or support != self.support_dimension_keys:
            raise EvaluationKernelContractError("evaluation dimension order drifted")
        hard = self.hard_conjunct_keys
        if len(hard) != len(set(hard)):
            raise EvaluationKernelContractError("evaluation hard conjunct keys overlap")
        if (not isinstance(self.thresholds, tuple)
                or tuple(item.result_key for item in self.thresholds) != hard):
            raise EvaluationKernelContractError(
                "evaluation thresholds must exactly follow hard conjunct order")
        if self.plugin.result_keys != hard:
            raise EvaluationKernelContractError(
                "evaluation plugin results differ from manifest hard conjuncts")
        declaration_sha = hashlib.sha256(canonical_json_bytes(
            _dimension_declaration_value(
                self.bearing_dimension_keys,
                self.generation_hard_conjunct_key,
                self.support_dimension_keys,
                self.thresholds,
            ))).hexdigest()
        if self.dimension_declaration_sha256 != declaration_sha:
            raise EvaluationKernelContractError(
                "evaluation dimension declaration commitment drifted")
        code_freeze_sha = _code_freeze_sha256(
            self.kernel_semantic_sha256, self.stage_manifest_sha256,
            self.plugin.semantic_sha256, self.source_binding.source_contract_sha256,
            self.dimension_declaration_sha256)
        if self.code_freeze_sha256 != code_freeze_sha:
            raise EvaluationKernelContractError("evaluation code freeze commitment drifted")
        registration = self.transport_registration()
        if registration.policy.hard_conjunct_keys != hard:
            raise EvaluationKernelContractError(
                "evaluation dimensions differ from the V2 boundary policy")
        if registration.family_commitment != self.transport_family_commitment:
            raise EvaluationKernelContractError(
                "evaluation transport family commitment drifted")
        if self.family_commitment != self.computed_family_commitment():
            raise EvaluationKernelContractError("evaluation family commitment drifted")
        if self.status != "FAMILY_MANIFEST_FROZEN":
            raise EvaluationKernelContractError("evaluation manifest status drifted")

    @property
    def hard_conjunct_keys(self) -> tuple[str, ...]:
        return (
            *self.bearing_dimension_keys,
            self.generation_hard_conjunct_key,
            *self.support_dimension_keys,
        )

    def _commitment_value(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("family_commitment")
        return value

    def computed_family_commitment(self) -> str:
        return hashlib.sha256(canonical_json_bytes(
            self._commitment_value())).hexdigest()

    def transport_registration(self) -> V2PrivateFamilyRegistration:
        """Rebuild the existing V2 firewall registration without payload reads."""
        return build_v2_private_family_registration(
            self.stage_key,
            payload_commitment=self.owner_binding.payload_commitment,
            case_commitment=self.owner_binding.case_commitment,
            label_commitment=self.owner_binding.label_commitment,
            cluster_commitment=self.owner_binding.cluster_commitment,
            candidate_freeze_sha256=self.candidate_artifact_sha256,
            code_freeze_sha256=self.code_freeze_sha256,
            resource_budget=self.resource_budget,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "bearing_dimension_keys": list(self.bearing_dimension_keys),
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "code_freeze_sha256": self.code_freeze_sha256,
            "consumed_lineage_sha256": self.consumed_lineage_sha256,
            "dimension_declaration_sha256": self.dimension_declaration_sha256,
            "family_commitment": self.family_commitment,
            "family_key": self.family_key,
            "format_version": self.format_version,
            "generation_hard_conjunct_key": self.generation_hard_conjunct_key,
            "kernel_semantic_sha256": self.kernel_semantic_sha256,
            "owner_binding": self.owner_binding.to_dict(),
            "plugin": self.plugin.to_dict(),
            "public_head_sha1": self.public_head_sha1,
            "release_key": self.release_key,
            "revision_key": self.revision_key,
            "source_binding": self.source_binding.to_dict(),
            "stage_key": self.stage_key,
            "stage_manifest_sha256": self.stage_manifest_sha256,
            "status": self.status,
            "support_dimension_keys": list(self.support_dimension_keys),
            "thresholds": [item.to_dict() for item in self.thresholds],
            "transport_family_commitment": self.transport_family_commitment,
            "resource_budget": self.resource_budget.to_dict(),
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "EvaluationKernelManifest":
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "bearing_dimension_keys",
            "candidate_artifact_sha256", "code_freeze_sha256",
            "consumed_lineage_sha256",
            "dimension_declaration_sha256", "family_commitment", "family_key",
            "format_version", "generation_hard_conjunct_key",
            "kernel_semantic_sha256", "owner_binding", "plugin", "release_key",
            "public_head_sha1",
            "resource_budget", "revision_key", "source_binding", "stage_key",
            "stage_manifest_sha256", "status", "support_dimension_keys",
            "thresholds", "transport_family_commitment",
        }, where="EvaluationKernelManifest")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            str(raw["stage_key"]), str(raw["family_key"]),
            str(raw["revision_key"]), str(raw["public_head_sha1"]),
            str(raw["family_commitment"]),
            str(raw["kernel_semantic_sha256"]),
            str(raw["stage_manifest_sha256"]),
            EvaluationPluginDeclaration.from_dict(raw["plugin"]),
            EvaluationSourceBinding.from_dict(raw["source_binding"]),
            EvaluationOwnerBinding.from_dict(raw["owner_binding"]),
            str(raw["candidate_artifact_sha256"]),
            str(raw["code_freeze_sha256"]),
            str(raw["transport_family_commitment"]),
            V2EvaluatorResourceBudget.from_dict(raw["resource_budget"]),
            str(raw["dimension_declaration_sha256"]),
            str(raw["consumed_lineage_sha256"]),
            tuple(str(item) for item in raw["bearing_dimension_keys"]),
            str(raw["generation_hard_conjunct_key"]),
            tuple(str(item) for item in raw["support_dimension_keys"]),
            tuple(EvaluationThreshold.from_dict(item) for item in raw["thresholds"]),
            str(raw["status"]),
        )


def build_evaluation_manifest(
        *,
        release_key: str,
        stage_key: str,
        family_key: str,
        revision_key: str,
        public_head_sha1: str,
        kernel_semantic_sha256: str,
        stage_manifest_sha256: str,
        plugin: EvaluationPluginDeclaration,
        source_binding: EvaluationSourceBinding,
        owner_binding: EvaluationOwnerBinding,
        candidate_artifact_sha256: str,
        resource_budget: V2EvaluatorResourceBudget,
        consumed_lineage_sha256: str,
        bearing_dimension_keys: tuple[str, ...],
        generation_hard_conjunct_key: str,
        support_dimension_keys: tuple[str, ...],
        thresholds: tuple[EvaluationThreshold, ...],
        ) -> EvaluationKernelManifest:
    """Build a self-committing family manifest without reading any payload."""
    dimension_sha = hashlib.sha256(canonical_json_bytes(
        _dimension_declaration_value(
            bearing_dimension_keys, generation_hard_conjunct_key,
            support_dimension_keys, thresholds))).hexdigest()
    code_freeze_sha = _code_freeze_sha256(
        kernel_semantic_sha256, stage_manifest_sha256,
        plugin.semantic_sha256, source_binding.source_contract_sha256,
        dimension_sha)
    registration = build_v2_private_family_registration(
        stage_key,
        payload_commitment=owner_binding.payload_commitment,
        case_commitment=owner_binding.case_commitment,
        label_commitment=owner_binding.label_commitment,
        cluster_commitment=owner_binding.cluster_commitment,
        candidate_freeze_sha256=candidate_artifact_sha256,
        code_freeze_sha256=code_freeze_sha,
        resource_budget=resource_budget,
    )
    provisional = EvaluationKernelManifest.__new__(EvaluationKernelManifest)
    values = (
        ("format_version", 1),
        ("artifact_kind", EVALUATION_KERNEL_MANIFEST_KIND),
        ("artifact_version", EVALUATION_KERNEL_MANIFEST_VERSION),
        ("release_key", release_key), ("stage_key", stage_key),
        ("family_key", family_key), ("revision_key", revision_key),
        ("public_head_sha1", public_head_sha1),
        ("family_commitment", "0" * 64),
        ("kernel_semantic_sha256", kernel_semantic_sha256),
        ("stage_manifest_sha256", stage_manifest_sha256),
        ("plugin", plugin), ("source_binding", source_binding),
        ("owner_binding", owner_binding),
        ("candidate_artifact_sha256", candidate_artifact_sha256),
        ("code_freeze_sha256", code_freeze_sha),
        ("transport_family_commitment", registration.family_commitment),
        ("resource_budget", resource_budget),
        ("dimension_declaration_sha256", dimension_sha),
        ("consumed_lineage_sha256", consumed_lineage_sha256),
        ("bearing_dimension_keys", bearing_dimension_keys),
        ("generation_hard_conjunct_key", generation_hard_conjunct_key),
        ("support_dimension_keys", support_dimension_keys),
        ("thresholds", thresholds), ("status", "FAMILY_MANIFEST_FROZEN"),
    )
    for name, value in values:
        object.__setattr__(provisional, name, value)
    commitment = provisional.computed_family_commitment()
    return EvaluationKernelManifest(*(
        value if name != "family_commitment" else commitment
        for name, value in values
    ))


def _code_freeze_sha256(
        kernel_semantic_sha256: str,
        stage_manifest_sha256: str,
        plugin_semantic_sha256: str,
        source_contract_sha256: str,
        dimension_declaration_sha256: str,
        ) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "dimension_declaration_sha256": dimension_declaration_sha256,
        "kernel_semantic_sha256": kernel_semantic_sha256,
        "plugin_semantic_sha256": plugin_semantic_sha256,
        "source_contract_sha256": source_contract_sha256,
        "stage_manifest_sha256": stage_manifest_sha256,
    })).hexdigest()


def publish_evaluation_manifest(
        manifest: EvaluationKernelManifest,
        path: str | Path,
        ) -> Path:
    """Publish append-only, or accept the exact canonical manifest idempotently."""
    if not isinstance(manifest, EvaluationKernelManifest):
        raise EvaluationKernelContractError("evaluation manifest type is invalid")
    target = Path(path)
    write_immutable_json(manifest.to_dict(), target)
    if read_evaluation_manifest(target) != manifest:
        raise EvaluationKernelContractError("evaluation manifest readback drifted")
    return target


def read_evaluation_manifest(path: str | Path) -> EvaluationKernelManifest:
    """Read a canonical family manifest and rerun every binding guard."""
    return EvaluationKernelManifest.from_dict(read_canonical_object(Path(path)))


__all__ = [
    "EVALUATION_BLOCKED_POLICIES",
    "EVALUATION_KERNEL_MANIFEST_KIND",
    "EVALUATION_KERNEL_MANIFEST_VERSION",
    "EVALUATION_NE_POLICIES",
    "EvaluationKernelManifest",
    "EvaluationThreshold",
    "build_evaluation_manifest",
    "publish_evaluation_manifest",
    "read_evaluation_manifest",
]
