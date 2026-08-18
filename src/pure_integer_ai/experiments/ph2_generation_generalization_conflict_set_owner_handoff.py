"""Independent owner-handoff contract for the GG03 ``CONFLICT_SET`` family.

This module is a label-free boundary only.  It does not load private labels,
run a candidate, publish a receipt, register a response act, or train a model.
All values are immutable typed records so the later owner and evaluator stages
cannot silently reuse the legacy single-claim family.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)

# object-model: value; representation=struct; interop=pending
# Every record below is a frozen, slot-backed struct.  No runtime owner or
# evaluator behavior is hidden behind these data carriers.

OWNER_HANDOFF_ARTIFACT_KIND = "PH2_GG03_CONFLICT_SET_OWNER_HANDOFF_V1"
CAPABILITY_KEY = "GG03_CONFLICT_SET_RUNTIME"
CODE_IDENTITY = "GG03_CONFLICT_SET_PUBLIC_V1"
EVALUATOR_OWNER = "GG03_CONFLICT_SET_EVALUATOR_OWNER_V1"
SOURCE_OWNER = "GG03_CONFLICT_SET_SOURCE_OWNER_V1"
FAMILY_NAMESPACE = "GG03_CONFLICT_SET_FORMAL_V1"
RESPONSE_ACT = "CONFLICT_SET"
PROJECTION_VERSION = "CONFLICT_SET_PROJECTION_V1"
GENERATION_CONTRACT = "REAL_G02_S07_R01_G04_MULTI_SENTENCE"
HANDOFF_STATUS = "OWNER_HANDOFF_READY"
SPLITS = ("train", "dev", "held_out")
FORMAL_RUN_LIMIT = 1
NEGATIVE_MATRIX_CASE_COUNT = 17

ARTIFACT_ROLES = (
    "code_freeze",
    "observation_pack",
    "source_manifest",
    "candidate_manifest",
    "public_preflight",
    "private_labels",
    "prediction_seal",
    "aggregate_report",
    "runtime_receipt",
    "formal_failure_report",
)

_SPLIT_ORDER = {name: index for index, name in enumerate(SPLITS)}
_LEGACY_IDENTITIES = frozenset({
    "PH2_GG03_EXECUTABLE_EVALUATION_FAMILY_FREEZE_V1",
    "PH2_GG03_EXECUTABLE_SEMANTIC_EVALUATION_FAMILY_FREEZE_V2",
    "PH2_GG03_FORMAL_AGGREGATE_V1",
    "PH2_GG03_FORMAL_SEMANTIC_AGGREGATE_V2",
    "GG03_GENERATION_GENERALIZATION",
})


class ConflictSetOwnerHandoffError(ValueError):
    """A handoff value violates the independent public contract."""


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConflictSetOwnerHandoffError(f"{where} must be non-empty text")
    return value


def _positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConflictSetOwnerHandoffError(
            f"{where} must be a positive strict integer")
    return value


def _zero(value: object, *, where: str) -> int:
    if type(value) is not int or value != 0:
        raise ConflictSetOwnerHandoffError(f"{where} must be zero")
    return value


def _one(value: object, *, where: str) -> int:
    if type(value) is not int or value != 1:
        raise ConflictSetOwnerHandoffError(f"{where} must be one")
    return value


def _sha(value: object, *, where: str, length: int) -> str:
    result = _text(value, where=where).lower()
    if (len(result) != length
            or any(char not in "0123456789abcdef" for char in result)):
        raise ConflictSetOwnerHandoffError(f"{where} must be hex SHA-{length * 4}")
    return result


def _int_key(value: object, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int for item in value)):
        raise ConflictSetOwnerHandoffError(
            f"{where} must be a non-empty strict integer key")
    return tuple(value)


def _strings(
        value: object,
        *,
        where: str,
        sorted_unique: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConflictSetOwnerHandoffError(f"{where} must be an array")
    result = tuple(_text(item, where=f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise ConflictSetOwnerHandoffError(f"{where} must be unique")
    if sorted_unique and tuple(sorted(result)) != result:
        raise ConflictSetOwnerHandoffError(f"{where} must be sorted")
    return result


def _exact(value: object, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ConflictSetOwnerHandoffError(
            f"{where} has missing or unknown fields")
    return value


def _source_ref(value: object, *, where: str) -> SourceRef:
    try:
        key = _int_key(value, where=f"{where}.stable_key")
        return SourceRef.from_stable_key(key)
    except (TypeError, ValueError, ConflictSetOwnerHandoffError) as error:
        if isinstance(error, ConflictSetOwnerHandoffError):
            raise
        raise ConflictSetOwnerHandoffError(
            f"{where} is not a valid SourceRef") from error


def _canonical_source_ids(
        value: object, *, where: str) -> tuple[str, ...]:
    result = _strings(value, where=where, sorted_unique=True)
    if len(result) < 2:
        raise ConflictSetOwnerHandoffError(
            f"{where} must contain at least two sources")
    return result


@dataclass(frozen=True, slots=True)
class ConflictSetResourceBudget:
    """Per-observation integer resource ceiling, not a performance claim."""

    max_claim_count: int
    max_source_count: int
    max_sentence_count: int
    max_output_units: int
    max_recovered_objects: int

    def __post_init__(self) -> None:
        for name in (
                "max_claim_count", "max_source_count", "max_sentence_count",
                "max_output_units", "max_recovered_objects"):
            _positive(getattr(self, name), where=f"resource.{name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_claim_count": self.max_claim_count,
            "max_output_units": self.max_output_units,
            "max_recovered_objects": self.max_recovered_objects,
            "max_sentence_count": self.max_sentence_count,
            "max_source_count": self.max_source_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetResourceBudget":
        raw = _exact(value, {
            "max_claim_count", "max_output_units", "max_recovered_objects",
            "max_sentence_count", "max_source_count",
        }, where="resource_budget")
        return cls(
            raw["max_claim_count"], raw["max_source_count"],
            raw["max_sentence_count"], raw["max_output_units"],
            raw["max_recovered_objects"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetSplitAxes:
    """Axes used to prevent random string-only splits."""

    claim_count: int
    source_count: int
    surface_family: str
    order_family: str
    source_cluster_ids: tuple[str, ...]
    lexical_structure: str

    def __post_init__(self) -> None:
        _positive(self.claim_count, where="split_axes.claim_count")
        _positive(self.source_count, where="split_axes.source_count")
        _text(self.surface_family, where="split_axes.surface_family")
        _text(self.order_family, where="split_axes.order_family")
        _strings(
            self.source_cluster_ids,
            where="split_axes.source_cluster_ids",
            sorted_unique=True,
        )
        _text(self.lexical_structure, where="split_axes.lexical_structure")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_count": self.claim_count,
            "lexical_structure": self.lexical_structure,
            "order_family": self.order_family,
            "source_cluster_ids": list(self.source_cluster_ids),
            "source_count": self.source_count,
            "surface_family": self.surface_family,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetSplitAxes":
        raw = _exact(value, {
            "claim_count", "lexical_structure", "order_family",
            "source_cluster_ids", "source_count", "surface_family",
        }, where="split_axes")
        return cls(
            raw["claim_count"], raw["source_count"],
            raw["surface_family"], raw["order_family"],
            _strings(raw["source_cluster_ids"],
                     where="split_axes.source_cluster_ids", sorted_unique=True),
            raw["lexical_structure"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetSourceManifestEntry:
    """Public source identity; stance is deliberately absent."""

    source_id: str
    source: SourceRef
    source_cluster_id: str
    document_scope: tuple[int, ...]
    content_sha256: str
    source_type: str
    license_id: str
    public_status: str
    split: str

    def __post_init__(self) -> None:
        _text(self.source_id, where="source_manifest.source_id")
        if not isinstance(self.source, SourceRef):
            raise TypeError("source_manifest.source must be SourceRef")
        _text(self.source_cluster_id, where="source_manifest.source_cluster_id")
        _int_key(self.document_scope, where="source_manifest.document_scope")
        _sha(self.content_sha256, where="source_manifest.content_sha256", length=64)
        _text(self.source_type, where="source_manifest.source_type")
        _text(self.license_id, where="source_manifest.license_id")
        if self.public_status != "PUBLIC":
            raise ConflictSetOwnerHandoffError(
                "source_manifest.public_status must be PUBLIC")
        if self.split not in SPLITS:
            raise ConflictSetOwnerHandoffError("source_manifest.split is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "document_scope": list(self.document_scope),
            "license_id": self.license_id,
            "public_status": self.public_status,
            "source_cluster_id": self.source_cluster_id,
            "source_id": self.source_id,
            "source_ref": list(self.source.stable_key()),
            "source_type": self.source_type,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetSourceManifestEntry":
        raw = _exact(value, {
            "content_sha256", "document_scope", "license_id", "public_status",
            "source_cluster_id", "source_id", "source_ref", "source_type",
            "split",
        }, where="source_manifest.entry")
        return cls(
            raw["source_id"],
            _source_ref(raw["source_ref"], where="source_manifest.source_ref"),
            raw["source_cluster_id"],
            _int_key(raw["document_scope"], where="source_manifest.document_scope"),
            raw["content_sha256"],
            raw["source_type"],
            raw["license_id"],
            raw["public_status"],
            raw["split"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetObservationClaim:
    """Label-free claim identity and its declared source closure."""

    claim_id: str
    proposition_key: tuple[int, ...]
    source_ids: tuple[str, ...]
    surface_course_family: str

    def __post_init__(self) -> None:
        _text(self.claim_id, where="observation.claim_id")
        _int_key(self.proposition_key, where="observation.proposition_key")
        if self.source_ids != _canonical_source_ids(
                self.source_ids, where="observation.claim.source_ids"):
            raise ConflictSetOwnerHandoffError(
                "observation claim source_ids must be canonical")
        _text(self.surface_course_family,
              where="observation.surface_course_family")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "proposition_key": list(self.proposition_key),
            "source_ids": list(self.source_ids),
            "surface_course_family": self.surface_course_family,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetObservationClaim":
        raw = _exact(value, {
            "claim_id", "proposition_key", "source_ids",
            "surface_course_family",
        }, where="observation.claim")
        return cls(
            raw["claim_id"],
            _int_key(raw["proposition_key"], where="observation.proposition_key"),
            _canonical_source_ids(raw["source_ids"],
                                  where="observation.claim.source_ids"),
            raw["surface_course_family"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetObservationSourceBinding:
    """One-to-one public source_id to SourceRef mapping."""

    source_id: str
    source: SourceRef

    def __post_init__(self) -> None:
        _text(self.source_id, where="observation.source_binding.source_id")
        if not isinstance(self.source, SourceRef):
            raise TypeError("observation.source_binding.source must be SourceRef")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_ref": list(self.source.stable_key()),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetObservationSourceBinding":
        raw = _exact(value, {"source_id", "source_ref"},
                     where="observation.source_binding")
        return cls(
            raw["source_id"],
            _source_ref(raw["source_ref"],
                        where="observation.source_binding.source_ref"),
        )


@dataclass(frozen=True, slots=True)
class ConflictSetOwnerObservation:
    """Label-free formal input visible to both owners."""

    observation_id: str
    scope_id: int
    response_act: str
    split: str
    claim_order: tuple[str, ...]
    claims: tuple[ConflictSetObservationClaim, ...]
    source_bindings: tuple[ConflictSetObservationSourceBinding, ...]
    generation_contract: str
    split_axes: ConflictSetSplitAxes
    resource_budget: ConflictSetResourceBudget

    def __post_init__(self) -> None:
        _text(self.observation_id, where="observation.observation_id")
        _positive(self.scope_id, where="observation.scope_id")
        if self.response_act != RESPONSE_ACT:
            raise ConflictSetOwnerHandoffError(
                "observation.response_act must be CONFLICT_SET")
        if self.split not in SPLITS:
            raise ConflictSetOwnerHandoffError("observation.split is invalid")
        if (len(self.claim_order) < 2
                or len(set(self.claim_order)) != len(self.claim_order)):
            raise ConflictSetOwnerHandoffError(
                "observation.claim_order must contain two or more unique claims")
        if (len(self.claims) != len(self.claim_order)
                or tuple(item.claim_id for item in self.claims)
                != self.claim_order):
            raise ConflictSetOwnerHandoffError(
                "observation claims must follow claim_order exactly")
        if (not self.source_bindings
                or tuple(item.source_id for item in self.source_bindings)
                != tuple(sorted({item.source_id for item in self.source_bindings}))
                or len({item.source_id for item in self.source_bindings})
                != len(self.source_bindings)
                or len({item.source for item in self.source_bindings})
                != len(self.source_bindings)):
            raise ConflictSetOwnerHandoffError(
                "observation source_bindings must be sorted and one-to-one")
        claim_sources = {
            source_id
            for claim in self.claims
            for source_id in claim.source_ids
        }
        bound_sources = {item.source_id for item in self.source_bindings}
        if claim_sources != bound_sources:
            raise ConflictSetOwnerHandoffError(
                "observation claim/source coverage is incomplete")
        if self.generation_contract != GENERATION_CONTRACT:
            raise ConflictSetOwnerHandoffError(
                "observation generation_contract is not the real runtime")
        if self.split_axes.claim_count != len(self.claims):
            raise ConflictSetOwnerHandoffError("split_axes claim_count drift")
        if self.split_axes.source_count != len(self.source_bindings):
            raise ConflictSetOwnerHandoffError("split_axes source_count drift")
        if self.split_axes.source_count < self.split_axes.claim_count * 2:
            raise ConflictSetOwnerHandoffError(
                "each claim must retain at least two source mappings")
        if self.resource_budget.max_claim_count < len(self.claims):
            raise ConflictSetOwnerHandoffError("resource claim ceiling is too small")
        if self.resource_budget.max_source_count < len(self.source_bindings):
            raise ConflictSetOwnerHandoffError("resource source ceiling is too small")
        if self.resource_budget.max_sentence_count < len(self.claims):
            raise ConflictSetOwnerHandoffError("resource sentence ceiling is too small")
        if self.resource_budget.max_recovered_objects < (
                len(self.claims) + len(self.source_bindings)):
            raise ConflictSetOwnerHandoffError(
                "resource recovery ceiling is too small")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_order": list(self.claim_order),
            "claims": [item.to_dict() for item in self.claims],
            "generation_contract": self.generation_contract,
            "observation_id": self.observation_id,
            "resource_budget": self.resource_budget.to_dict(),
            "response_act": self.response_act,
            "scope_id": self.scope_id,
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "split": self.split,
            "split_axes": self.split_axes.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetOwnerObservation":
        raw = _exact(value, {
            "claim_order", "claims", "generation_contract", "observation_id",
            "resource_budget", "response_act", "scope_id", "source_bindings",
            "split", "split_axes",
        }, where="observation")
        claims = tuple(
            ConflictSetObservationClaim.from_dict(item)
            for item in raw["claims"]
        )
        bindings = tuple(
            ConflictSetObservationSourceBinding.from_dict(item)
            for item in raw["source_bindings"]
        )
        return cls(
            raw["observation_id"], raw["scope_id"],
            raw["response_act"], raw["split"],
            _strings(raw["claim_order"], where="observation.claim_order"),
            claims, bindings, raw["generation_contract"],
            ConflictSetSplitAxes.from_dict(raw["split_axes"]),
            ConflictSetResourceBudget.from_dict(raw["resource_budget"]),
        )


@dataclass(frozen=True, slots=True)
class ConflictSetPublicPreflight:
    """Evidence that only the public runtime was exercised."""

    public_head_sha1: str
    status: str
    working_tree_clean: int
    positive_runtime_case_count: int
    negative_matrix_case_count: int
    compileall_passed: int
    diff_check_passed: int
    legacy_scan_clear: int
    secret_scan_clear: int
    teacher_api_llm_call_count: int
    private_label_read_count: int
    formal_run_count: int

    def __post_init__(self) -> None:
        _sha(self.public_head_sha1, where="public_preflight.public_head_sha1", length=40)
        if self.status != "PASS":
            raise ConflictSetOwnerHandoffError("public_preflight.status must be PASS")
        for name in (
                "working_tree_clean", "compileall_passed", "diff_check_passed",
                "legacy_scan_clear", "secret_scan_clear"):
            _one(getattr(self, name), where=f"public_preflight.{name}")
        _positive(self.positive_runtime_case_count,
                  where="public_preflight.positive_runtime_case_count")
        if self.negative_matrix_case_count != NEGATIVE_MATRIX_CASE_COUNT:
            raise ConflictSetOwnerHandoffError(
                "public_preflight negative matrix count is not frozen")
        for name in (
                "teacher_api_llm_call_count", "private_label_read_count",
                "formal_run_count"):
            _zero(getattr(self, name), where=f"public_preflight.{name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "compileall_passed": self.compileall_passed,
            "diff_check_passed": self.diff_check_passed,
            "formal_run_count": self.formal_run_count,
            "legacy_scan_clear": self.legacy_scan_clear,
            "negative_matrix_case_count": self.negative_matrix_case_count,
            "positive_runtime_case_count": self.positive_runtime_case_count,
            "private_label_read_count": self.private_label_read_count,
            "public_head_sha1": self.public_head_sha1,
            "secret_scan_clear": self.secret_scan_clear,
            "status": self.status,
            "teacher_api_llm_call_count": self.teacher_api_llm_call_count,
            "working_tree_clean": self.working_tree_clean,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetPublicPreflight":
        raw = _exact(value, {
            "compileall_passed", "diff_check_passed", "formal_run_count",
            "legacy_scan_clear", "negative_matrix_case_count",
            "positive_runtime_case_count", "private_label_read_count",
            "public_head_sha1", "secret_scan_clear", "status",
            "teacher_api_llm_call_count", "working_tree_clean",
        }, where="public_preflight")
        return cls(
            raw["public_head_sha1"], raw["status"],
            raw["working_tree_clean"], raw["positive_runtime_case_count"],
            raw["negative_matrix_case_count"], raw["compileall_passed"],
            raw["diff_check_passed"], raw["legacy_scan_clear"],
            raw["secret_scan_clear"], raw["teacher_api_llm_call_count"],
            raw["private_label_read_count"], raw["formal_run_count"],
        )


@dataclass(frozen=True, slots=True)
class ConflictSetArtifactRole:
    """One independently owned artifact role in the future formal family."""

    role: str
    artifact_kind: str
    owner: str
    visibility: str

    def __post_init__(self) -> None:
        _text(self.role, where="artifact_role.role")
        _text(self.artifact_kind, where="artifact_role.artifact_kind")
        _text(self.owner, where="artifact_role.owner")
        if self.visibility not in {"PUBLIC", "PRIVATE"}:
            raise ConflictSetOwnerHandoffError("artifact_role.visibility is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "owner": self.owner,
            "role": self.role,
            "visibility": self.visibility,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetArtifactRole":
        raw = _exact(value, {"artifact_kind", "owner", "role", "visibility"},
                     where="artifact_role")
        return cls(
            raw["role"], raw["artifact_kind"], raw["owner"],
            raw["visibility"],
        )


def expected_conflict_set_artifact_roles() -> tuple[ConflictSetArtifactRole, ...]:
    """Return the fixed role inventory; no role may be omitted or added."""
    private = {"PRIVATE"}
    specs = (
        ("code_freeze", "CODE_FREEZE_V1", CODE_IDENTITY, "PUBLIC"),
        ("observation_pack", "OBSERVATION_PACK_V1", SOURCE_OWNER, "PUBLIC"),
        ("source_manifest", "SOURCE_MANIFEST_V1", SOURCE_OWNER, "PUBLIC"),
        ("candidate_manifest", "CANDIDATE_MANIFEST_V1", CODE_IDENTITY, "PUBLIC"),
        ("public_preflight", "PUBLIC_PREFLIGHT_V1", CODE_IDENTITY, "PUBLIC"),
        ("private_labels", "PRIVATE_LABELS_V1", EVALUATOR_OWNER, "PRIVATE"),
        ("prediction_seal", "PREDICTION_SEAL_V1", EVALUATOR_OWNER, "PRIVATE"),
        ("aggregate_report", "AGGREGATE_REPORT_V1", EVALUATOR_OWNER, "PUBLIC"),
        ("runtime_receipt", "RUNTIME_RECEIPT_V1", EVALUATOR_OWNER, "PUBLIC"),
        ("formal_failure_report", "FORMAL_FAILURE_REPORT_V1", EVALUATOR_OWNER,
         "PUBLIC"),
    )
    return tuple(
        ConflictSetArtifactRole(
            role, f"{FAMILY_NAMESPACE}_{kind}", owner,
            "PRIVATE" if visibility in private else visibility,
        )
        for role, kind, owner, visibility in specs
    )


@dataclass(frozen=True, slots=True)
class ConflictSetOwnerHandoff:
    """Complete public handoff metadata before any private evaluator exists."""

    capability_key: str
    code_identity: str
    evaluator_owner: str
    source_owner: str
    family_namespace: str
    response_act: str
    projection_version: str
    public_preflight: ConflictSetPublicPreflight
    source_manifest: tuple[ConflictSetSourceManifestEntry, ...]
    observations: tuple[ConflictSetOwnerObservation, ...]
    artifact_roles: tuple[ConflictSetArtifactRole, ...]
    formal_run_limit: int
    status: str

    def __post_init__(self) -> None:
        if self.capability_key != CAPABILITY_KEY:
            raise ConflictSetOwnerHandoffError("capability_key is not independent")
        if self.code_identity != CODE_IDENTITY:
            raise ConflictSetOwnerHandoffError("code_identity is not independent")
        if self.evaluator_owner != EVALUATOR_OWNER:
            raise ConflictSetOwnerHandoffError("evaluator_owner is not independent")
        if self.source_owner != SOURCE_OWNER or self.source_owner == self.evaluator_owner:
            raise ConflictSetOwnerHandoffError("source_owner is not independent")
        if self.family_namespace != FAMILY_NAMESPACE:
            raise ConflictSetOwnerHandoffError("family_namespace is legacy or invalid")
        if self.family_namespace in _LEGACY_IDENTITIES:
            raise ConflictSetOwnerHandoffError("legacy family identity is forbidden")
        if self.response_act != RESPONSE_ACT:
            raise ConflictSetOwnerHandoffError("response_act is invalid")
        if self.projection_version != PROJECTION_VERSION:
            raise ConflictSetOwnerHandoffError("projection_version is invalid")
        if not isinstance(self.public_preflight, ConflictSetPublicPreflight):
            raise TypeError("public_preflight type is invalid")
        if self.formal_run_limit != FORMAL_RUN_LIMIT:
            raise ConflictSetOwnerHandoffError("formal run limit must be one")
        if self.status != HANDOFF_STATUS:
            raise ConflictSetOwnerHandoffError("handoff status is invalid")
        if (not self.source_manifest
                or tuple(item.source_id for item in self.source_manifest)
                != tuple(sorted(item.source_id for item in self.source_manifest))
                or len({item.source_id for item in self.source_manifest})
                != len(self.source_manifest)
                or len({item.source for item in self.source_manifest})
                != len(self.source_manifest)):
            raise ConflictSetOwnerHandoffError(
                "source manifest must be sorted and one-to-one")
        if (not self.observations
                or tuple(
                    (_SPLIT_ORDER[item.split], item.observation_id)
                    for item in self.observations
                ) != tuple(sorted(
                    (_SPLIT_ORDER[item.split], item.observation_id)
                    for item in self.observations))
                or len({item.observation_id for item in self.observations})
                != len(self.observations)):
            raise ConflictSetOwnerHandoffError(
                "observations must be ordered and unique")
        if set(item.split for item in self.observations) != set(SPLITS):
            raise ConflictSetOwnerHandoffError(
                "train/dev/held_out must all be present")
        manifest_by_id = {item.source_id: item for item in self.source_manifest}
        used_ids: set[str] = set()
        clusters_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
        for observation in self.observations:
            expected_clusters: set[str] = set()
            for binding in observation.source_bindings:
                entry = manifest_by_id.get(binding.source_id)
                if entry is None or entry.source != binding.source:
                    raise ConflictSetOwnerHandoffError(
                        "observation source mapping drifted from manifest")
                if entry.split != observation.split:
                    raise ConflictSetOwnerHandoffError(
                        "observation/source split drift")
                used_ids.add(binding.source_id)
                expected_clusters.add(entry.source_cluster_id)
            if observation.split_axes.source_cluster_ids != tuple(sorted(expected_clusters)):
                raise ConflictSetOwnerHandoffError(
                    "observation source-cluster axis is incomplete")
            clusters_by_split[observation.split].update(expected_clusters)
        if used_ids != set(manifest_by_id):
            raise ConflictSetOwnerHandoffError(
                "source manifest contains unused or missing mappings")
        for index, split in enumerate(SPLITS):
            if any(clusters_by_split[split] & clusters_by_split[other]
                   for other in SPLITS[index + 1:]):
                raise ConflictSetOwnerHandoffError(
                    "source clusters must be isolated across splits")
        if self.artifact_roles != expected_conflict_set_artifact_roles():
            raise ConflictSetOwnerHandoffError(
                "artifact role inventory is incomplete or independently owned")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": OWNER_HANDOFF_ARTIFACT_KIND,
            "artifact_roles": [item.to_dict() for item in self.artifact_roles],
            "capability_key": self.capability_key,
            "code_identity": self.code_identity,
            "evaluator_owner": self.evaluator_owner,
            "family_namespace": self.family_namespace,
            "formal_run_limit": self.formal_run_limit,
            "observations": [item.to_dict() for item in self.observations],
            "projection_version": self.projection_version,
            "public_preflight": self.public_preflight.to_dict(),
            "response_act": self.response_act,
            "source_manifest": [item.to_dict() for item in self.source_manifest],
            "source_owner": self.source_owner,
            "status": self.status,
            "format_version": 1,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConflictSetOwnerHandoff":
        raw = _exact(value, {
            "artifact_kind", "artifact_roles", "capability_key", "code_identity",
            "evaluator_owner", "family_namespace", "formal_run_limit",
            "format_version", "observations", "projection_version",
            "public_preflight", "response_act", "source_manifest", "source_owner",
            "status",
        }, where="owner_handoff")
        if (raw["artifact_kind"] != OWNER_HANDOFF_ARTIFACT_KIND
                or raw["format_version"] != 1):
            raise ConflictSetOwnerHandoffError(
                "artifact_kind or format_version is not this family")
        try:
            source_manifest = tuple(
                ConflictSetSourceManifestEntry.from_dict(item)
                for item in raw["source_manifest"]
            )
            observations = tuple(
                ConflictSetOwnerObservation.from_dict(item)
                for item in raw["observations"]
            )
            roles = tuple(
                ConflictSetArtifactRole.from_dict(item)
                for item in raw["artifact_roles"]
            )
            return cls(
                raw["capability_key"], raw["code_identity"],
                raw["evaluator_owner"], raw["source_owner"],
                raw["family_namespace"], raw["response_act"],
                raw["projection_version"],
                ConflictSetPublicPreflight.from_dict(raw["public_preflight"]),
                source_manifest, observations, roles,
                raw["formal_run_limit"], raw["status"],
            )
        except ConflictSetOwnerHandoffError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            raise ConflictSetOwnerHandoffError(
                "owner_handoff contains an invalid typed value") from error

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_conflict_set_owner_handoff_bytes(
        payload: bytes) -> ConflictSetOwnerHandoff:
    """Parse exactly one canonical JSONL handoff record."""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise ConflictSetOwnerHandoffError("handoff must be one JSONL record")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except (TypeError, ValueError) as error:
        raise ConflictSetOwnerHandoffError("handoff JSON is not canonical") from error
    if canonical_json_line(value) != payload:
        raise ConflictSetOwnerHandoffError("handoff JSON bytes are not canonical")
    return ConflictSetOwnerHandoff.from_dict(value)


def read_conflict_set_owner_handoff(
        path: str | Path) -> ConflictSetOwnerHandoff:
    """Read one immutable public handoff record without side effects."""
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise ConflictSetOwnerHandoffError("handoff file is unreadable") from error
    return parse_conflict_set_owner_handoff_bytes(payload)


__all__ = [
    "ARTIFACT_ROLES",
    "CAPABILITY_KEY",
    "CODE_IDENTITY",
    "EVALUATOR_OWNER",
    "FAMILY_NAMESPACE",
    "FORMAL_RUN_LIMIT",
    "GENERATION_CONTRACT",
    "HANDOFF_STATUS",
    "NEGATIVE_MATRIX_CASE_COUNT",
    "OWNER_HANDOFF_ARTIFACT_KIND",
    "PROJECTION_VERSION",
    "RESPONSE_ACT",
    "SOURCE_OWNER",
    "SPLITS",
    "ConflictSetArtifactRole",
    "ConflictSetObservationClaim",
    "ConflictSetObservationSourceBinding",
    "ConflictSetOwnerHandoff",
    "ConflictSetOwnerHandoffError",
    "ConflictSetOwnerObservation",
    "ConflictSetPublicPreflight",
    "ConflictSetResourceBudget",
    "ConflictSetSourceManifestEntry",
    "ConflictSetSplitAxes",
    "expected_conflict_set_artifact_roles",
    "parse_conflict_set_owner_handoff_bytes",
    "read_conflict_set_owner_handoff",
]
