"""V5-first streaming I/O for the successor V3 R4 blind owner."""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Iterator

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    validate_blind_private_owner_record_v5,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
    V2PrivateFamilyRegistration,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2AccessPermit,
    V2AccessRequest,
    V2PhysicalRoots,
    V2WriteAccount,
    authorize_v2_access,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_contract import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLITS,
    W02MorphologySuccessorV3PrivateR4FileIdentity,
)


# object-model: exception
class W02MorphologySuccessorV3PrivateR4IOError(RuntimeError):
    """An R4 permit, V5 source closure, stream, or pair binding drifted."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def r4_private_file_by_layout(
        files: tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        layout_key: str,
        ) -> W02MorphologySuccessorV3PrivateR4FileIdentity:
    matches = tuple(row for row in files if row.layout_key == layout_key)
    if len(matches) != 1 or matches[0].root_key != "PRIVATE_EVALUATOR_ROOT":
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private owner layout is not unique")
    return matches[0]


def r4_private_split_layout(split: str, kind: str) -> str:
    if split not in W02_MORPH_V3_PRIVATE_SPLITS or kind not in {
            "observation", "label"}:
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private split/kind is not registered")
    suffix = "OBSERVATION" if kind == "observation" else "LABEL"
    return f"PRIVATE_{split.upper()}_{suffix}"


def authorize_w02_morphology_successor_v3_private_r4_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        ) -> dict[str, V2AccessPermit]:
    """Bridge the frozen CC-BY-SA-4.0 identities into the V2 firewall."""
    if tuple(row.layout_key for row in files) != W02_MORPH_V3_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private file inventory drifted")
    permits = {}
    for layout_key in W02_MORPH_V3_PRIVATE_LAYOUTS:
        identity = r4_private_file_by_layout(files, layout_key)
        split = identity.split or "held_out"
        request = V2AccessRequest(
            "W-02", "PH2_V2_PRIVATE_EVALUATOR", split,
            identity.record_kind, W02_MORPH_V3_PRIVATE_PATHS[layout_key],
            identity.transport_sha256, identity.transport_size_bytes,
            "PRIVATE_EVALUATION", registration.candidate_freeze_sha256,
            registration.code_freeze_sha256, V2WriteAccount())
        permits[layout_key] = authorize_v2_access(
            boundary, roots, request, registration=registration)
    return permits


def iter_w02_morphology_successor_v3_private_r4_records(
        identity: W02MorphologySuccessorV3PrivateR4FileIdentity,
        permit: V2AccessPermit,
        ) -> Iterator[object]:
    """Read one authorized gzip with V5 authority and close both identities."""
    if (not isinstance(identity, W02MorphologySuccessorV3PrivateR4FileIdentity)
            or not isinstance(permit, V2AccessPermit)
            or permit.root_key != "PRIVATE_EVALUATOR_ROOT"
            or permit.record_kind != identity.record_kind
            or permit.content_sha256 != identity.transport_sha256
            or permit.content_size_bytes != identity.transport_size_bytes):
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private permit does not match frozen identity")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key = None
    last_key = None
    previous_key = None
    try:
        with permit.target_path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02MorphologySuccessorV3PrivateR4IOError(
                            f"R4 private JSONL line {line_number} newline drifted")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_blind_private_owner_record_v5(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorV3PrivateR4IOError(
                            "R4 private record kind drifted")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02MorphologySuccessorV3PrivateR4IOError(
                            "R4 private record split drifted")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02MorphologySuccessorV3PrivateR4IOError(
                            "R4 private stable keys are not strictly ordered")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except W02MorphologySuccessorV3PrivateR4IOError:
        raise
    except (OSError, EOFError, ValueError) as error:
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private gzip/JSONL read failed") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private content identity drifted")
    size, digest = _sha256_file(permit.target_path)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private transport drifted during content read")


def read_and_close_w02_morphology_successor_v3_private_r4_sources(
        files: tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        permits: dict[str, V2AccessPermit],
        ) -> tuple[SourceRefRecord, ...]:
    """Fully close every V5 SourceRef before any observation or label read."""
    sources = tuple(iter_w02_morphology_successor_v3_private_r4_records(
        r4_private_file_by_layout(files, "PRIVATE_SOURCE"),
        permits["PRIVATE_SOURCE"]))
    if (len(sources) != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or any(not isinstance(row, SourceRefRecord) for row in sources)):
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 V5 SourceRef closure did not complete")
    return sources


def iter_w02_morphology_successor_v3_private_r4_pairs(
        files: tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        permits: dict[str, V2AccessPermit], split: str,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """Pair observation and label streams after SourceRef closure."""
    observation_key = r4_private_split_layout(split, "observation")
    label_key = r4_private_split_layout(split, "label")
    observations = iter_w02_morphology_successor_v3_private_r4_records(
        r4_private_file_by_layout(files, observation_key),
        permits[observation_key])
    labels = iter_w02_morphology_successor_v3_private_r4_records(
        r4_private_file_by_layout(files, label_key), permits[label_key])
    count = 0
    for observation, evaluation in zip(observations, labels, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)
                or observation.split != split
                or evaluation.observation_key != observation.stable_key
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02MorphologySuccessorV3PrivateR4IOError(
                "R4 private pair owner/binding drifted")
        count += 1
        yield observation, evaluation
    if count != r4_private_file_by_layout(files, observation_key).record_count:
        raise W02MorphologySuccessorV3PrivateR4IOError(
            "R4 private pair count drifted")


__all__ = [
    "W02MorphologySuccessorV3PrivateR4IOError",
    "authorize_w02_morphology_successor_v3_private_r4_files",
    "iter_w02_morphology_successor_v3_private_r4_pairs",
    "iter_w02_morphology_successor_v3_private_r4_records",
    "r4_private_file_by_layout",
    "r4_private_split_layout",
    "read_and_close_w02_morphology_successor_v3_private_r4_sources",
]
