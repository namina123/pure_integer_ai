"""Public-only W-03 V2 evaluation records and exact source binding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_core import PUBLIC_LICENSE_IDS
from pure_integer_ai.experiments.ph2_evaluation_kernel.plugin import (
    EvaluationPluginDeclaration,
    EvaluationPluginRunContext,
)
from pure_integer_ai.experiments.ph2_evaluation_kernel.source_binding import (
    EvaluationSourceBinding,
    EvaluationSourceSlice,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload


W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION = "PH2-W03-V2-PUBLIC-SOURCE-V1"


# object-model: exception
class W03V2PublicSourceError(ValueError):
    """The proposed public stream is incomplete, private, or cross-wired."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _cluster_key(source: SourceRefRecord) -> str:
    return ".".join(str(item) for item in source.source_cluster_key.components)


def _record_key(record: object) -> tuple[int, ...]:
    stable_key = getattr(record, "stable_key", None)
    components = getattr(stable_key, "components", None)
    if (not isinstance(components, tuple) or not components
            or any(type(item) is not int for item in components)):
        raise W03V2PublicSourceError("public evaluation record key is invalid")
    return components


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicSourceRecord:
    """One public SourceRef record authorized for a bounded W-03 stream."""

    source_binding_sha256: str
    record: SourceRefRecord

    def __post_init__(self) -> None:
        if (not isinstance(self.source_binding_sha256, str)
                or len(self.source_binding_sha256) != 64):
            raise W03V2PublicSourceError("public source binding SHA is invalid")
        if not isinstance(self.record, SourceRefRecord):
            raise W03V2PublicSourceError("public source record type is invalid")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicObservationEvidencePair:
    """One public train Observation and its exact public teacher Evidence."""

    source_binding_sha256: str
    observation: ObservationRecord
    evidence: TeacherEvidenceRecord

    def __post_init__(self) -> None:
        if (not isinstance(self.source_binding_sha256, str)
                or len(self.source_binding_sha256) != 64):
            raise W03V2PublicSourceError("public pair binding SHA is invalid")
        if (not isinstance(self.observation, ObservationRecord)
                or not isinstance(self.evidence, TeacherEvidenceRecord)):
            raise W03V2PublicSourceError("public observation/evidence pair drifted")
        if (self.observation.split != "train"
                or self.evidence.observation_key != self.observation.stable_key
                or self.evidence.source_ref_key
                != self.observation.source_ref_key):
            raise W03V2PublicSourceError("public pair identity is not closed")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicEvaluationBatch:
    """Canonical SourceRef-first public records for an experimental run."""

    source_binding: EvaluationSourceBinding
    source_records: tuple[W03V2PublicSourceRecord, ...]
    pairs: tuple[W03V2PublicObservationEvidencePair, ...]
    record_commitment: str
    transport_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_binding, EvaluationSourceBinding):
            raise W03V2PublicSourceError("public EvaluationSourceBinding is invalid")
        if (not self.source_records or not self.pairs
                or any(not isinstance(item, W03V2PublicSourceRecord)
                       for item in self.source_records)
                or any(not isinstance(item, W03V2PublicObservationEvidencePair)
                       for item in self.pairs)):
            raise W03V2PublicSourceError("public evaluation batch is empty or invalid")
        binding_sha = self.source_binding.sha256()
        if any(item.source_binding_sha256 != binding_sha
               for item in (*self.source_records, *self.pairs)):
            raise W03V2PublicSourceError("public stream escaped its source binding")
        if self.source_binding.record_count != len(self.source_records):
            raise W03V2PublicSourceError("public source count drifted")
        if (not isinstance(self.record_commitment, str)
                or len(self.record_commitment) != 64
                or type(self.transport_bytes) is not int
                or self.transport_bytes <= 0):
            raise W03V2PublicSourceError("public record commitment is invalid")
        source_keys = tuple(
            _record_key(item.record) for item in self.source_records)
        pair_keys = tuple(
            _record_key(item.observation) for item in self.pairs)
        if (source_keys != tuple(sorted(source_keys))
                or pair_keys != tuple(sorted(pair_keys))
                or len(source_keys) != len(set(source_keys))
                or len(pair_keys) != len(set(pair_keys))):
            raise W03V2PublicSourceError("public stream order or uniqueness drifted")

    @property
    def records(self) -> tuple[object, ...]:
        """Return the only authorized SourceRef-first plugin stream."""
        return (*self.source_records, *self.pairs)

    def training_payload(self) -> W03TrainingPayload:
        """Recover the existing W-03 train payload without evaluator labels."""
        return W03TrainingPayload(
            tuple(item.record for item in self.source_records),
            tuple(item.observation for item in self.pairs),
            tuple(item.evidence for item in self.pairs),
        )


def build_w03_v2_public_evaluation_batch(
        payload: W03TrainingPayload,
        ) -> W03V2PublicEvaluationBatch:
    """Authorize only closed, redistributable train pairs for public P0-P2."""
    if not isinstance(payload, W03TrainingPayload):
        raise TypeError("public W-03 source adapter requires W03TrainingPayload")
    source_by_key: dict[object, SourceRefRecord] = {}
    for source in payload.source_refs:
        if not isinstance(source, SourceRefRecord):
            raise W03V2PublicSourceError("public source inventory type drifted")
        if source.stable_key in source_by_key:
            raise W03V2PublicSourceError("public source inventory contains duplicates")
        source_by_key[source.stable_key] = source
    observations = tuple(sorted(
        payload.observations, key=lambda item: _record_key(item)))
    evidence = tuple(sorted(
        payload.teacher_evidence, key=lambda item: _record_key(item)))
    if (not observations or any(not isinstance(item, ObservationRecord)
                                for item in observations)
            or any(item.split != "train" for item in observations)):
        raise W03V2PublicSourceError("public W-03 stream must be nonempty train-only")
    if len({_record_key(item) for item in observations}) != len(observations):
        raise W03V2PublicSourceError("public observations contain duplicate keys")
    evidence_by_observation: dict[object, TeacherEvidenceRecord] = {}
    for item in evidence:
        if not isinstance(item, TeacherEvidenceRecord):
            raise W03V2PublicSourceError("public evidence inventory type drifted")
        if item.observation_key in evidence_by_observation:
            raise W03V2PublicSourceError("public observation has multiple evidence rows")
        evidence_by_observation[item.observation_key] = item
    observation_keys = {item.stable_key for item in observations}
    if set(evidence_by_observation) != observation_keys:
        raise W03V2PublicSourceError("public observation/evidence inventory is not closed")

    selected_sources = []
    for source_key in sorted({item.source_ref_key for item in observations}):
        source = source_by_key.get(source_key)
        if source is None:
            raise W03V2PublicSourceError("public observation references unknown SourceRef")
        if (source.redistribution_policy != "PUBLIC"
                or source.license_id not in PUBLIC_LICENSE_IDS):
            raise W03V2PublicSourceError("nonpublic source entered public evaluation")
        if source.record_ordinal <= 0:
            raise W03V2PublicSourceError("public source ordinal must be positive")
        selected_sources.append(source)
    selected = tuple(sorted(selected_sources, key=_record_key))
    selected_by_key = {item.stable_key: item for item in selected}
    for observation in observations:
        source = selected_by_key[observation.source_ref_key]
        item = evidence_by_observation[observation.stable_key]
        if (observation.license_partition != source.license_id
                or item.source_ref_key != source.stable_key):
            raise W03V2PublicSourceError("public source/pair license or identity drifted")

    source_ref_commitment = _sha([item.to_dict() for item in selected])
    pair_commitment = _sha([
        {
            "evidence": evidence_by_observation[item.stable_key].to_dict(),
            "observation": item.to_dict(),
        }
        for item in observations
    ])
    source_contract_sha256 = _sha({
        "adapter_version": W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION,
        "pair_commitment": pair_commitment,
        "record_order": "SOURCE_REF_FIRST_THEN_OBSERVATION_EVIDENCE_PAIRS",
        "source_ref_commitment": source_ref_commitment,
    })
    slices = tuple(sorted(
        EvaluationSourceSlice(
            item.source_key,
            "train",
            _cluster_key(item),
            item.record_ordinal,
            item.record_ordinal,
            1,
            _sha(item.to_dict()),
        )
        for item in selected
    ))
    binding = EvaluationSourceBinding(
        source_contract_sha256, slices, source_ref_commitment)
    binding_sha = binding.sha256()
    source_records = tuple(
        W03V2PublicSourceRecord(binding_sha, item) for item in selected)
    pairs = tuple(
        W03V2PublicObservationEvidencePair(
            binding_sha, item, evidence_by_observation[item.stable_key])
        for item in observations
    )
    record_payload = {
        "pairs": [
            {
                "evidence": item.evidence.to_dict(),
                "observation": item.observation.to_dict(),
            }
            for item in pairs
        ],
        "sources": [item.record.to_dict() for item in source_records],
    }
    record_bytes = canonical_json_bytes(record_payload)
    return W03V2PublicEvaluationBatch(
        binding, source_records, pairs,
        hashlib.sha256(record_bytes).hexdigest(), len(record_bytes),
    )


def build_w03_v2_public_run_context(
        batch: W03V2PublicEvaluationBatch,
        declaration: EvaluationPluginDeclaration,
        ) -> EvaluationPluginRunContext:
    """Build a payload-free experimental context without a formal family/guard."""
    if (not isinstance(batch, W03V2PublicEvaluationBatch)
            or not isinstance(declaration, EvaluationPluginDeclaration)):
        raise TypeError("public W-03 context inputs are invalid")
    identity = {
        "mode": "PUBLIC_ONLY_EXPERIMENTAL_P0_P2",
        "plugin_declaration_sha256": declaration.sha256(),
        "record_commitment": batch.record_commitment,
        "source_binding_sha256": batch.source_binding.sha256(),
    }
    return EvaluationPluginRunContext(
        _sha({"manifest_projection": identity}),
        _sha({"family_projection": identity}),
        batch.source_binding.sha256(),
        _sha({"owner": "PUBLIC_ONLY_NO_FORMAL_OWNER"}),
        1,
    )


__all__ = [
    "W03_V2_PUBLIC_SOURCE_ADAPTER_VERSION",
    "W03V2PublicEvaluationBatch",
    "W03V2PublicObservationEvidencePair",
    "W03V2PublicSourceError",
    "W03V2PublicSourceRecord",
    "build_w03_v2_public_evaluation_batch",
    "build_w03_v2_public_run_context",
]
