"""PH2-D03-V2 evaluator isolation, family preregistration and safe reports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_OWNER_KEYS,
    V2_RELEASE_KEY,
)
from pure_integer_ai.experiments.ph2_d03_v2_catalog import (
    V2_CONTRACT_PATH,
    read_v2_successor_contract,
)


V2_EVALUATOR_BOUNDARY_KIND = "PH2_D03_V2_EVALUATOR_BOUNDARY"
V2_EVALUATOR_BOUNDARY_VERSION = "PH2-D03-V2-evaluator-boundary-v1"
V2_EVALUATOR_BOUNDARY_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_ft00_06_evaluator_boundary_v1.json"
)
V2_EVALUATOR_STAGES = tuple(f"W-{ordinal:02d}" for ordinal in range(2, 10))
V2_EVALUATOR_SPLITS = ("train", "dev", "held_out", "adversarial", "wall")
V2_PRIVATE_SPLITS = ("held_out", "adversarial", "wall")
V2_REPORT_STATUSES = ("FAIL", "NE", "PASS")
V2_NE_POLICY = "BLOCK"
V2_ZERO_CALL_WINDOW_COUNT = 3
V2_REPORT_GUARD_VERSION = "PH2-D03-V2-SAFE-REPORT-GUARD-V1"
V2_EXPOSURE_POLICY_VERSION = "PH2-D03-V2-EXPOSURE-LEDGER-V1"

V2_RESOURCE_HARD_LIMITS = {
    "max_checkpoint_count": 2_304,
    "max_logic_operations": 9_000_000,
    "max_payload_bytes": 603_979_776,
    "max_payload_gets": 589_824,
    "max_records": 900_000,
    "max_workers": 4,
}
V2_ZERO_WRITE_FIELDS = (
    "assessment_writes",
    "candidate_writes",
    "clock_writes",
    "companion_writes",
    "core_writes",
    "evaluator_label_writes",
    "evidence_writes",
    "host_writes",
    "memory_writes",
    "use_writes",
)
V2_PATH_LAYOUTS = (
    ("source_ref", "source/source_refs.jsonl.gz"),
    ("observation", "observations/{split}.jsonl.gz"),
    ("teacher_evidence", "teacher/train.evidence.jsonl.gz"),
    ("evaluator_label", "evaluator/{split}.labels.jsonl.gz"),
)

_STAGE_BEARINGS = (
    ("W-02", (
        "W-02-V2-BOUNDARY-WITHDRAWAL",
        "W-02-V2-MULTI-CANDIDATE",
        "W-02-V2-NEW-CONTENT-MORPHOLOGY",
        "W-02-V2-OOV",
    )),
    ("W-03", (
        "W-03-V2-CONCEPT-SPLIT",
        "W-03-V2-POLYSEMY-COMPETITION",
        "W-03-V2-SOURCE-CONFLICT",
        "W-03-V2-SUPERSEDE",
    )),
    ("W-04", (
        "W-04-V2-CONTENT-REPLACEMENT",
        "W-04-V2-CUE-REPLACEMENT",
        "W-04-V2-EVIDENCE-ABLATION",
        "W-04-V2-SEED-ABLATION",
    )),
    ("W-05", (
        "W-05-V2-OCCURRENCE-IDENTITY",
        "W-05-V2-PROPOSITION-CONSUMER",
        "W-05-V2-ROLE-SWAP",
        "W-05-V2-SCOPE",
    )),
    ("W-06", (
        "W-06-V2-CAUSES",
        "W-06-V2-MEREOLOGY",
        "W-06-V2-PRECEDES",
        "W-06-V2-PROPERTY",
        "W-06-V2-PURE-ALIAS-REFERS",
        "W-06-V2-SIMILAR-ANTONYM",
        "W-06-V2-SUBSET-MEMBER",
    )),
    ("W-07", (
        "W-07-V2-AND-OR",
        "W-07-V2-CONDITION",
        "W-07-V2-EXISTS",
        "W-07-V2-FORALL",
        "W-07-V2-MODAL",
        "W-07-V2-NESTED-SCOPE",
        "W-07-V2-NOT",
    )),
    ("W-08", (
        "W-08-V2-CHINESE-VARIATION",
        "W-08-V2-DISCOURSE",
        "W-08-V2-LOCAL-RECOMPUTE",
        "W-08-V2-LONG-CONTEXT",
        "W-08-V2-P3IA",
    )),
    ("W-09", (
        "W-09-V2-DIMENSIONAL-PASS",
        "W-09-V2-RESOURCE-STOP",
        "W-09-V2-ROLLBACK",
        "W-09-V2-TEACHER-ZERO-WINDOW",
        "W-09-V2-V06-CLONE",
    )),
)

_FORBIDDEN_REPORT_KEYS = frozenset({
    "absolute_path", "case", "case_key", "error_message", "exception",
    "expected", "expected_payload", "expected_state", "label", "label_key",
    "message", "observed_surface", "path", "private_path", "prompt",
    "raw_observation", "raw_text", "relative_path", "surface", "surface_form",
    "text", "typed_payload",
})
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\|file:)", re.IGNORECASE)


class V2EvaluatorBoundaryError(D03ContractError):
    """The successor evaluator boundary or preregistration drifted."""


class V2ReportExposureError(V2EvaluatorBoundaryError):
    """A safe report contained a forbidden field, value or type."""

    def __init__(self, exposure_kind: str) -> None:
        self.exposure_kind = exposure_kind
        super().__init__(f"v2 safe report exposure: {exposure_kind}")


def _strict_sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise V2EvaluatorBoundaryError(f"{where} must be lowercase SHA-256")
    return value


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise V2EvaluatorBoundaryError(f"{where} must be a positive integer")
    return value


def _flag(value: object, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise V2EvaluatorBoundaryError(f"{where} must be an integer flag")
    return value


def _ordered_strings(value: object, *, where: str) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise V2EvaluatorBoundaryError(f"{where} must be non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise V2EvaluatorBoundaryError(f"{where} must be unique")
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_ordered(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2StageEvaluationPolicy:
    """Exact successor dimensions and hard gates for one W stage."""

    stage_key: str
    bearing_dimension_keys: tuple[str, ...]
    generation_hard_conjunct_key: str
    hard_conjunct_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    min_pass_numerator: int = 1
    min_pass_denominator: int = 1
    max_fail_count: int = 0
    ne_policy: str = V2_NE_POLICY
    zero_call_window_count: int = V2_ZERO_CALL_WINDOW_COUNT

    def __post_init__(self) -> None:
        if self.stage_key not in V2_EVALUATOR_STAGES:
            raise V2EvaluatorBoundaryError("v2 evaluator stage is not registered")
        bearings = _ordered_strings(
            self.bearing_dimension_keys, where="v2 bearing dimensions")
        if any(not item.startswith(f"{self.stage_key}-V2-") for item in bearings):
            raise V2EvaluatorBoundaryError("v2 bearing dimension identity drifted")
        expected_generation = f"{self.stage_key}-V2-GENERATION-HARD-CONJUNCT"
        if self.generation_hard_conjunct_key != expected_generation:
            raise V2EvaluatorBoundaryError("v2 generation hard conjunct drifted")
        support = (
            f"{self.stage_key}-V2-RESOURCE",
            f"{self.stage_key}-V2-ROLLBACK",
            f"{self.stage_key}-V2-ZERO-CALL-WINDOWS",
            f"{self.stage_key}-V2-V06-CLONE",
        )
        expected_hard = _unique_ordered((*bearings, expected_generation, *support))
        if self.hard_conjunct_keys != expected_hard:
            raise V2EvaluatorBoundaryError("v2 hard conjunct order drifted")
        if self.ablation_keys != tuple(f"{item}-ABLATION" for item in expected_hard):
            raise V2EvaluatorBoundaryError("v2 ablation order drifted")
        if (self.min_pass_numerator != 1 or self.min_pass_denominator != 1
                or self.max_fail_count != 0 or self.ne_policy != V2_NE_POLICY
                or self.zero_call_window_count != V2_ZERO_CALL_WINDOW_COUNT):
            raise V2EvaluatorBoundaryError("v2 evaluator threshold was weakened")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_keys": list(self.ablation_keys),
            "bearing_dimension_keys": list(self.bearing_dimension_keys),
            "generation_hard_conjunct_key": self.generation_hard_conjunct_key,
            "hard_conjunct_keys": list(self.hard_conjunct_keys),
            "max_fail_count": self.max_fail_count,
            "min_pass_denominator": self.min_pass_denominator,
            "min_pass_numerator": self.min_pass_numerator,
            "ne_policy": self.ne_policy,
            "stage_key": self.stage_key,
            "zero_call_window_count": self.zero_call_window_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2StageEvaluationPolicy":
        raw = exact_dict(value, {
            "ablation_keys", "bearing_dimension_keys",
            "generation_hard_conjunct_key", "hard_conjunct_keys",
            "max_fail_count", "min_pass_denominator", "min_pass_numerator",
            "ne_policy", "stage_key", "zero_call_window_count",
        }, where="V2StageEvaluationPolicy")
        return cls(
            str(raw["stage_key"]),
            _ordered_strings(raw["bearing_dimension_keys"], where="bearing dimensions"),
            str(raw["generation_hard_conjunct_key"]),
            _ordered_strings(raw["hard_conjunct_keys"], where="hard conjuncts"),
            _ordered_strings(raw["ablation_keys"], where="ablations"),
            raw["min_pass_numerator"], raw["min_pass_denominator"],
            raw["max_fail_count"], str(raw["ne_policy"]),
            raw["zero_call_window_count"],
        )


def _stage_policy(stage_key: str, bearings: tuple[str, ...]) -> V2StageEvaluationPolicy:
    generation = f"{stage_key}-V2-GENERATION-HARD-CONJUNCT"
    hard = _unique_ordered((
        *bearings, generation,
        f"{stage_key}-V2-RESOURCE",
        f"{stage_key}-V2-ROLLBACK",
        f"{stage_key}-V2-ZERO-CALL-WINDOWS",
        f"{stage_key}-V2-V06-CLONE",
    ))
    return V2StageEvaluationPolicy(
        stage_key, bearings, generation, hard,
        tuple(f"{item}-ABLATION" for item in hard),
    )


V2_STAGE_EVALUATION_POLICIES = tuple(
    _stage_policy(stage, bearings) for stage, bearings in _STAGE_BEARINGS)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2OwnerRootPolicy:
    """An opaque physical root and the only records visible through it."""

    owner_key: str
    root_key: str
    purpose: str
    allowed_splits: tuple[str, ...]
    allowed_record_kinds: tuple[str, ...]
    candidate_freeze_required: int
    code_freeze_required: int
    zero_write_required: int

    def __post_init__(self) -> None:
        if self.owner_key not in V2_OWNER_KEYS:
            raise V2EvaluatorBoundaryError("v2 evaluator owner is not registered")
        if not isinstance(self.root_key, str) or not self.root_key:
            raise V2EvaluatorBoundaryError("v2 evaluator root key is invalid")
        if not isinstance(self.purpose, str) or not self.purpose:
            raise V2EvaluatorBoundaryError("v2 evaluator purpose is invalid")
        splits = _ordered_strings(self.allowed_splits, where="v2 owner splits")
        if splits != tuple(item for item in V2_EVALUATOR_SPLITS if item in splits):
            raise V2EvaluatorBoundaryError("v2 owner split order drifted")
        kinds = _ordered_strings(self.allowed_record_kinds, where="v2 owner record kinds")
        if any(item not in dict(V2_PATH_LAYOUTS) for item in kinds):
            raise V2EvaluatorBoundaryError("v2 owner record kind is invalid")
        for name in (
            "candidate_freeze_required", "code_freeze_required",
            "zero_write_required",
        ):
            _flag(getattr(self, name), where=f"v2 owner {name}")
        is_private = self.owner_key == "PH2_V2_PRIVATE_EVALUATOR"
        if ((self.candidate_freeze_required, self.code_freeze_required)
                != ((1, 1) if is_private else (0, 0))):
            raise V2EvaluatorBoundaryError("v2 private freeze gate drifted")
        if self.zero_write_required != 1:
            raise V2EvaluatorBoundaryError("v2 payload access must be read-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_record_kinds": list(self.allowed_record_kinds),
            "allowed_splits": list(self.allowed_splits),
            "candidate_freeze_required": self.candidate_freeze_required,
            "code_freeze_required": self.code_freeze_required,
            "owner_key": self.owner_key,
            "purpose": self.purpose,
            "root_key": self.root_key,
            "zero_write_required": self.zero_write_required,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2OwnerRootPolicy":
        raw = exact_dict(value, {
            "allowed_record_kinds", "allowed_splits",
            "candidate_freeze_required", "code_freeze_required", "owner_key",
            "purpose", "root_key", "zero_write_required",
        }, where="V2OwnerRootPolicy")
        return cls(
            str(raw["owner_key"]), str(raw["root_key"]), str(raw["purpose"]),
            _ordered_strings(raw["allowed_splits"], where="owner splits"),
            _ordered_strings(raw["allowed_record_kinds"], where="owner kinds"),
            raw["candidate_freeze_required"], raw["code_freeze_required"],
            raw["zero_write_required"],
        )


V2_OWNER_ROOT_POLICIES = (
    V2OwnerRootPolicy(
        "PH2_V2_CANDIDATE", "CANDIDATE_TRAIN_ROOT", "TRAIN_INTAKE",
        ("train",), ("source_ref", "observation"), 0, 0, 1),
    V2OwnerRootPolicy(
        "PH2_V2_TEACHER", "TEACHER_TRAIN_ROOT", "TRAIN_EVIDENCE",
        ("train",), ("source_ref", "teacher_evidence"), 0, 0, 1),
    V2OwnerRootPolicy(
        "PH2_V2_DEV_CALIBRATOR", "DEV_CALIBRATION_ROOT", "DEV_CALIBRATION",
        ("dev",), ("source_ref", "observation", "evaluator_label"), 0, 0, 1),
    V2OwnerRootPolicy(
        "PH2_V2_SHADOW_AUDITOR", "SHADOW_AUDIT_ROOT", "SHADOW_AUDIT",
        ("train", "dev"), ("source_ref", "observation"), 0, 0, 1),
    V2OwnerRootPolicy(
        "PH2_V2_PRIVATE_EVALUATOR", "PRIVATE_EVALUATOR_ROOT",
        "PRIVATE_EVALUATION", V2_PRIVATE_SPLITS,
        ("source_ref", "observation", "evaluator_label"), 1, 1, 1),
)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2EvaluatorResourceBudget:
    """A family-specific positive budget bounded by the FT00 hard limits."""

    max_checkpoint_count: int
    max_logic_operations: int
    max_payload_bytes: int
    max_payload_gets: int
    max_records: int
    max_workers: int

    def __post_init__(self) -> None:
        for name, hard_limit in V2_RESOURCE_HARD_LIMITS.items():
            value = _positive(getattr(self, name), where=f"v2 resource {name}")
            if value > hard_limit:
                raise V2EvaluatorBoundaryError(f"v2 resource {name} exceeds hard limit")
        if self.max_workers not in {1, 2, 4}:
            raise V2EvaluatorBoundaryError("v2 evaluator workers must be 1, 2 or 4")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in sorted(V2_RESOURCE_HARD_LIMITS)}

    @classmethod
    def from_dict(cls, value: Any) -> "V2EvaluatorResourceBudget":
        raw = exact_dict(value, set(V2_RESOURCE_HARD_LIMITS),
                         where="V2EvaluatorResourceBudget")
        return cls(**{name: raw[name] for name in V2_RESOURCE_HARD_LIMITS})


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2PrivateFamilyRegistration:
    """Metadata-only freeze created after Candidate and code are frozen."""

    stage_key: str
    family_commitment: str
    payload_commitment: str
    case_commitment: str
    label_commitment: str
    cluster_commitment: str
    candidate_freeze_sha256: str
    code_freeze_sha256: str
    policy: V2StageEvaluationPolicy
    resource_budget: V2EvaluatorResourceBudget
    private_owner_key: str = "PH2_V2_PRIVATE_EVALUATOR"
    formal_run_count: int = 0
    private_payload_reads: int = 0
    exposure_incident_count: int = 0
    blind_pass_eligible: int = 1

    def __post_init__(self) -> None:
        policies = {item.stage_key: item for item in V2_STAGE_EVALUATION_POLICIES}
        if self.stage_key not in policies or self.policy != policies[self.stage_key]:
            raise V2EvaluatorBoundaryError("v2 private family stage policy drifted")
        for name in (
            "family_commitment", "payload_commitment", "case_commitment",
            "label_commitment", "cluster_commitment", "candidate_freeze_sha256",
            "code_freeze_sha256",
        ):
            _strict_sha256(getattr(self, name), where=f"v2 family {name}")
        if self.private_owner_key != "PH2_V2_PRIVATE_EVALUATOR":
            raise V2EvaluatorBoundaryError("v2 private family owner drifted")
        if any(type(value) is not int or value != 0 for value in (
                self.formal_run_count, self.private_payload_reads,
                self.exposure_incident_count)):
            raise V2EvaluatorBoundaryError("v2 family must freeze before any private run")
        if self.blind_pass_eligible != 1:
            raise V2EvaluatorBoundaryError("an exposed family cannot be preregistered")
        if self.family_commitment != self.computed_family_commitment():
            raise V2EvaluatorBoundaryError("v2 private family commitment drifted")

    def _commitment_value(self) -> dict[str, Any]:
        return _family_commitment_value(
            self.stage_key,
            payload_commitment=self.payload_commitment,
            case_commitment=self.case_commitment,
            label_commitment=self.label_commitment,
            cluster_commitment=self.cluster_commitment,
            candidate_freeze_sha256=self.candidate_freeze_sha256,
            code_freeze_sha256=self.code_freeze_sha256,
            policy=self.policy,
            resource_budget=self.resource_budget,
        )

    def computed_family_commitment(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self._commitment_value()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._commitment_value(),
            "blind_pass_eligible": self.blind_pass_eligible,
            "exposure_incident_count": self.exposure_incident_count,
            "family_commitment": self.family_commitment,
            "formal_run_count": self.formal_run_count,
            "private_payload_reads": self.private_payload_reads,
        }


def _family_commitment_value(
        stage_key: str,
        *,
        payload_commitment: str,
        case_commitment: str,
        label_commitment: str,
        cluster_commitment: str,
        candidate_freeze_sha256: str,
        code_freeze_sha256: str,
        policy: V2StageEvaluationPolicy,
        resource_budget: V2EvaluatorResourceBudget,
        ) -> dict[str, Any]:
    return {
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "case_commitment": case_commitment,
        "cluster_commitment": cluster_commitment,
        "code_freeze_sha256": code_freeze_sha256,
        "label_commitment": label_commitment,
        "payload_commitment": payload_commitment,
        "policy": policy.to_dict(),
        "private_owner_key": "PH2_V2_PRIVATE_EVALUATOR",
        "resource_budget": resource_budget.to_dict(),
        "stage_key": stage_key,
    }


def build_v2_private_family_registration(
        stage_key: str,
        *,
        payload_commitment: str,
        case_commitment: str,
        label_commitment: str,
        cluster_commitment: str,
        candidate_freeze_sha256: str,
        code_freeze_sha256: str,
        resource_budget: V2EvaluatorResourceBudget,
        ) -> V2PrivateFamilyRegistration:
    """Build a zero-read registration without opening any family payload."""
    policies = {item.stage_key: item for item in V2_STAGE_EVALUATION_POLICIES}
    if stage_key not in policies or not isinstance(resource_budget, V2EvaluatorResourceBudget):
        raise V2EvaluatorBoundaryError("v2 private family inputs are invalid")
    payload = _strict_sha256(payload_commitment, where="payload")
    cases = _strict_sha256(case_commitment, where="case")
    labels = _strict_sha256(label_commitment, where="label")
    clusters = _strict_sha256(cluster_commitment, where="cluster")
    candidate = _strict_sha256(candidate_freeze_sha256, where="candidate freeze")
    code = _strict_sha256(code_freeze_sha256, where="code freeze")
    policy = policies[stage_key]
    family_commitment = _sha256_bytes(canonical_json_bytes(
        _family_commitment_value(
            stage_key,
            payload_commitment=payload,
            case_commitment=cases,
            label_commitment=labels,
            cluster_commitment=clusters,
            candidate_freeze_sha256=candidate,
            code_freeze_sha256=code,
            policy=policy,
            resource_budget=resource_budget,
        )
    ))
    return V2PrivateFamilyRegistration(
        stage_key, family_commitment, payload, cases, labels, clusters,
        candidate, code, policy, resource_budget,
    )


def validate_v2_safe_report(value: object) -> None:
    """Recursively reject private fields, absolute paths and non-JSON values."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise V2ReportExposureError("NON_STRING_KEY")
            lowered = key.lower()
            if (lowered in _FORBIDDEN_REPORT_KEYS or lowered.endswith((
                    "_path", "_surface", "_text", "_payload", "_expected"))):
                raise V2ReportExposureError("FORBIDDEN_FIELD")
            validate_v2_safe_report(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_v2_safe_report(item)
    elif isinstance(value, str):
        if _ABSOLUTE_PATH.match(value):
            raise V2ReportExposureError("ABSOLUTE_PATH")
        if "/" in value or "\\" in value:
            raise V2ReportExposureError("PATH_STRING")
    elif value is not None and type(value) not in {bool, int}:
        raise V2ReportExposureError("UNSAFE_VALUE_TYPE")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class V2EvaluatorBoundaryContract:
    """The public, payload-free FT00-06 release boundary."""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    successor_contract: D03FileIdentity
    successor_contract_sha256: str
    owner_root_policies: tuple[V2OwnerRootPolicy, ...]
    path_layouts: tuple[tuple[str, str], ...]
    stage_policies: tuple[V2StageEvaluationPolicy, ...]
    resource_hard_limits: dict[str, int]
    zero_write_fields: tuple[str, ...]
    report_guard_version: str
    exposure_policy_version: str
    exposure_policy: str
    private_authorization_policy: str
    initial_state: dict[str, int]
    status: str

    def __post_init__(self) -> None:
        if (self.format_version != 1 or type(self.format_version) is not int
                or self.artifact_kind != V2_EVALUATOR_BOUNDARY_KIND
                or self.artifact_version != V2_EVALUATOR_BOUNDARY_VERSION
                or self.release_key != V2_RELEASE_KEY):
            raise V2EvaluatorBoundaryError("v2 evaluator boundary identity drifted")
        if not isinstance(self.successor_contract, D03FileIdentity):
            raise V2EvaluatorBoundaryError("v2 successor contract identity is invalid")
        _strict_sha256(self.successor_contract_sha256,
                       where="v2 successor canonical contract")
        if self.owner_root_policies != V2_OWNER_ROOT_POLICIES:
            raise V2EvaluatorBoundaryError("v2 owner root policies drifted")
        if self.path_layouts != V2_PATH_LAYOUTS:
            raise V2EvaluatorBoundaryError("v2 evaluator path layouts drifted")
        if self.stage_policies != V2_STAGE_EVALUATION_POLICIES:
            raise V2EvaluatorBoundaryError("v2 stage evaluator policies drifted")
        if self.resource_hard_limits != V2_RESOURCE_HARD_LIMITS:
            raise V2EvaluatorBoundaryError("v2 evaluator hard limits drifted")
        if self.zero_write_fields != V2_ZERO_WRITE_FIELDS:
            raise V2EvaluatorBoundaryError("v2 evaluator zero-write account drifted")
        if (self.report_guard_version != V2_REPORT_GUARD_VERSION
                or self.exposure_policy_version != V2_EXPOSURE_POLICY_VERSION
                or self.exposure_policy
                != "APPEND_ONLY_INCIDENT_REVOKES_BLIND_PASS_PERMANENTLY"
                or self.private_authorization_policy
                != "CANDIDATE_AND_CODE_FREEZE_THEN_BLIND_FAMILY_ONLY"):
            raise V2EvaluatorBoundaryError("v2 evaluator safety policy drifted")
        expected_state = {
            "FT00_COMPLETE": 0,
            "blind_families_exposed": 0,
            "candidate_writes": 0,
            "formal_private_evaluation_runs": 0,
            "formal_training_runs": 0,
            "private_payload_reads": 0,
        }
        if self.initial_state != expected_state or self.status != "EVALUATOR_BOUNDARY_FROZEN":
            raise V2EvaluatorBoundaryError("v2 evaluator initial state drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "exposure_policy": self.exposure_policy,
            "exposure_policy_version": self.exposure_policy_version,
            "format_version": self.format_version,
            "initial_state": dict(self.initial_state),
            "owner_root_policies": [item.to_dict() for item in self.owner_root_policies],
            "path_layouts": [
                {"record_kind": kind, "root_relative_layout": layout}
                for kind, layout in self.path_layouts
            ],
            "private_authorization_policy": self.private_authorization_policy,
            "release_key": self.release_key,
            "report_guard_version": self.report_guard_version,
            "resource_hard_limits": dict(sorted(self.resource_hard_limits.items())),
            "stage_policies": [item.to_dict() for item in self.stage_policies],
            "status": self.status,
            "successor_contract": self.successor_contract.to_dict(),
            "successor_contract_sha256": self.successor_contract_sha256,
            "zero_write_fields": list(self.zero_write_fields),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2EvaluatorBoundaryContract":
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "exposure_policy",
            "exposure_policy_version", "format_version", "initial_state",
            "owner_root_policies", "path_layouts", "private_authorization_policy",
            "release_key", "report_guard_version", "resource_hard_limits",
            "stage_policies", "status", "successor_contract",
            "successor_contract_sha256", "zero_write_fields",
        }, where="V2EvaluatorBoundaryContract")
        if not isinstance(raw["path_layouts"], list):
            raise V2EvaluatorBoundaryError("v2 path layouts must be an array")
        layouts = []
        for item in raw["path_layouts"]:
            pair = exact_dict(item, {"record_kind", "root_relative_layout"},
                              where="v2 path layout")
            layouts.append((str(pair["record_kind"]), str(pair["root_relative_layout"])))
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            D03FileIdentity.from_dict(raw["successor_contract"]),
            str(raw["successor_contract_sha256"]),
            tuple(V2OwnerRootPolicy.from_dict(item)
                  for item in raw["owner_root_policies"]),
            tuple(layouts),
            tuple(V2StageEvaluationPolicy.from_dict(item)
                  for item in raw["stage_policies"]),
            dict(raw["resource_hard_limits"]),
            tuple(str(item) for item in raw["zero_write_fields"]),
            str(raw["report_guard_version"]),
            str(raw["exposure_policy_version"]), str(raw["exposure_policy"]),
            str(raw["private_authorization_policy"]), dict(raw["initial_state"]),
            str(raw["status"]),
        )

    def sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self.to_dict()))


def build_v2_evaluator_boundary_contract(
        repository_root: str | Path) -> V2EvaluatorBoundaryContract:
    """Build the public contract without opening any evaluator payload."""
    root = Path(repository_root).resolve()
    successor = read_v2_successor_contract(root)
    target = root / Path(*V2_CONTRACT_PATH.split("/"))
    payload = target.read_bytes()
    identity = D03FileIdentity(
        V2_CONTRACT_PATH, len(payload), _sha256_bytes(payload))
    return V2EvaluatorBoundaryContract(
        1, V2_EVALUATOR_BOUNDARY_KIND, V2_EVALUATOR_BOUNDARY_VERSION,
        V2_RELEASE_KEY, identity, successor.sha256(), V2_OWNER_ROOT_POLICIES,
        V2_PATH_LAYOUTS, V2_STAGE_EVALUATION_POLICIES,
        dict(V2_RESOURCE_HARD_LIMITS), V2_ZERO_WRITE_FIELDS,
        V2_REPORT_GUARD_VERSION, V2_EXPOSURE_POLICY_VERSION,
        "APPEND_ONLY_INCIDENT_REVOKES_BLIND_PASS_PERMANENTLY",
        "CANDIDATE_AND_CODE_FREEZE_THEN_BLIND_FAMILY_ONLY",
        {
            "FT00_COMPLETE": 0,
            "blind_families_exposed": 0,
            "candidate_writes": 0,
            "formal_private_evaluation_runs": 0,
            "formal_training_runs": 0,
            "private_payload_reads": 0,
        },
        "EVALUATOR_BOUNDARY_FROZEN",
    )


def publish_v2_evaluator_boundary_contract(
        repository_root: str | Path,
        path: str | Path | None = None,
        ) -> Path:
    """Publish the canonical public boundary by exclusive or idempotent write."""
    root = Path(repository_root).resolve()
    target = (root / Path(*V2_EVALUATOR_BOUNDARY_PATH.split("/"))
              if path is None else Path(path).resolve())
    contract = build_v2_evaluator_boundary_contract(root)
    write_immutable_json(contract.to_dict(), target)
    if read_v2_evaluator_boundary_contract(root, target) != contract:
        raise V2EvaluatorBoundaryError("v2 evaluator boundary readback drifted")
    return target


def read_v2_evaluator_boundary_contract(
        repository_root: str | Path,
        path: str | Path | None = None,
        ) -> V2EvaluatorBoundaryContract:
    """Strictly read the public boundary and revalidate the successor parent."""
    root = Path(repository_root).resolve()
    target = (root / Path(*V2_EVALUATOR_BOUNDARY_PATH.split("/"))
              if path is None else Path(path).resolve())
    contract = V2EvaluatorBoundaryContract.from_dict(read_canonical_object(target))
    expected = build_v2_evaluator_boundary_contract(root)
    if contract != expected:
        raise V2EvaluatorBoundaryError("v2 evaluator boundary parent or policy drifted")
    return contract


__all__ = [
    "V2_EVALUATOR_BOUNDARY_PATH", "V2_EVALUATOR_SPLITS",
    "V2_EVALUATOR_STAGES", "V2_EXPOSURE_POLICY_VERSION", "V2_NE_POLICY",
    "V2_OWNER_ROOT_POLICIES", "V2_PATH_LAYOUTS", "V2_PRIVATE_SPLITS",
    "V2_REPORT_GUARD_VERSION", "V2_RESOURCE_HARD_LIMITS",
    "V2_STAGE_EVALUATION_POLICIES", "V2_ZERO_CALL_WINDOW_COUNT",
    "V2_ZERO_WRITE_FIELDS", "V2EvaluatorBoundaryContract",
    "V2EvaluatorBoundaryError", "V2EvaluatorResourceBudget",
    "V2OwnerRootPolicy", "V2PrivateFamilyRegistration", "V2ReportExposureError",
    "V2StageEvaluationPolicy", "build_v2_evaluator_boundary_contract",
    "build_v2_private_family_registration",
    "publish_v2_evaluator_boundary_contract",
    "read_v2_evaluator_boundary_contract", "validate_v2_safe_report",
]
