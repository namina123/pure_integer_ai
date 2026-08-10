"""successor V3 R5 盲测 owner 的 V6-first 流式 I/O。"""
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
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v6 import (
    validate_blind_private_owner_record_v6,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_contract import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLITS,
    W02MorphologySuccessorV3PrivateR5FileIdentity,
)


# object-model: exception
class W02MorphologySuccessorV3PrivateR5IOError(RuntimeError):
    """R5 permit、V6 source 闭合、stream 或 lzh 绑定发生漂移。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def r5_private_file_by_layout(
        files: tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        layout_key: str,
        ) -> W02MorphologySuccessorV3PrivateR5FileIdentity:
    matches = tuple(row for row in files if row.layout_key == layout_key)
    if len(matches) != 1 or matches[0].root_key != "PRIVATE_EVALUATOR_ROOT":
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private owner layout is not unique")
    return matches[0]


def r5_private_split_layout(split: str, kind: str) -> str:
    if split not in W02_MORPH_V3_PRIVATE_SPLITS or kind not in {
            "observation", "label"}:
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private split/kind is not registered")
    suffix = "OBSERVATION" if kind == "observation" else "LABEL"
    return f"PRIVATE_{split.upper()}_{suffix}"


def authorize_w02_morphology_successor_v3_private_r5_files(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        ) -> dict[str, V2AccessPermit]:
    """把冻结的 R5 文件身份接入 V2 firewall。"""
    if tuple(row.layout_key for row in files) != W02_MORPH_V3_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private file inventory drifted")
    permits = {}
    for layout_key in W02_MORPH_V3_PRIVATE_LAYOUTS:
        identity = r5_private_file_by_layout(files, layout_key)
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


def iter_w02_morphology_successor_v3_private_r5_records(
        identity: W02MorphologySuccessorV3PrivateR5FileIdentity,
        permit: V2AccessPermit,
        ) -> Iterator[object]:
    """按 V6 authority 读取一个获准 gzip，并闭合内容与传输身份。"""
    if (not isinstance(identity, W02MorphologySuccessorV3PrivateR5FileIdentity)
            or not isinstance(permit, V2AccessPermit)
            or permit.root_key != "PRIVATE_EVALUATOR_ROOT"
            or permit.record_kind != identity.record_kind
            or permit.content_sha256 != identity.transport_sha256
            or permit.content_size_bytes != identity.transport_size_bytes):
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private permit does not match frozen identity")
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
                        raise W02MorphologySuccessorV3PrivateR5IOError(
                            f"R5 private JSONL line {line_number} newline drifted")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_blind_private_owner_record_v6(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02MorphologySuccessorV3PrivateR5IOError(
                            "R5 private record kind drifted")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02MorphologySuccessorV3PrivateR5IOError(
                            "R5 private record split drifted")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02MorphologySuccessorV3PrivateR5IOError(
                            "R5 private stable keys are not strictly ordered")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except W02MorphologySuccessorV3PrivateR5IOError:
        raise
    except (OSError, EOFError, ValueError) as error:
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private gzip/JSONL read failed") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private content identity drifted")
    size, digest = _sha256_file(permit.target_path)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private transport drifted during content read")


def read_and_close_w02_morphology_successor_v3_private_r5_sources(
        files: tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        permits: dict[str, V2AccessPermit],
        ) -> tuple[SourceRefRecord, ...]:
    """读取任何 observation 或 label 前，先完整闭合全部 V6 SourceRef。"""
    sources = tuple(iter_w02_morphology_successor_v3_private_r5_records(
        r5_private_file_by_layout(files, "PRIVATE_SOURCE"),
        permits["PRIVATE_SOURCE"]))
    if (len(sources) != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            or any(not isinstance(row, SourceRefRecord) for row in sources)):
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 V6 SourceRef closure did not complete")
    return sources


def iter_w02_morphology_successor_v3_private_r5_pairs(
        files: tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...],
        permits: dict[str, V2AccessPermit], split: str,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """SourceRef 闭合后配对原始 lzh observation 与 label。"""
    observation_key = r5_private_split_layout(split, "observation")
    label_key = r5_private_split_layout(split, "label")
    observations = iter_w02_morphology_successor_v3_private_r5_records(
        r5_private_file_by_layout(files, observation_key),
        permits[observation_key])
    labels = iter_w02_morphology_successor_v3_private_r5_records(
        r5_private_file_by_layout(files, label_key), permits[label_key])
    count = 0
    for observation, evaluation in zip(observations, labels, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)
                or observation.language != "lzh"
                or observation.split != split
                or evaluation.observation_key != observation.stable_key
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02MorphologySuccessorV3PrivateR5IOError(
                "R5 private pair language/owner/binding drifted")
        count += 1
        yield observation, evaluation
    if count != r5_private_file_by_layout(files, observation_key).record_count:
        raise W02MorphologySuccessorV3PrivateR5IOError(
            "R5 private pair count drifted")


__all__ = [
    "W02MorphologySuccessorV3PrivateR5IOError",
    "authorize_w02_morphology_successor_v3_private_r5_files",
    "iter_w02_morphology_successor_v3_private_r5_pairs",
    "iter_w02_morphology_successor_v3_private_r5_records",
    "r5_private_file_by_layout",
    "r5_private_split_layout",
    "read_and_close_w02_morphology_successor_v3_private_r5_sources",
]
